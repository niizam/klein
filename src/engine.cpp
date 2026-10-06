// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

#include "common.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "graph.h"
#include "vision.h"

namespace klein {

// VRAM kept for decode-time compute buffers (verify batches of a few tokens, MTP passes, logits).
static constexpr size_t kDecodeReserve = 192ull << 20;

// Lookup drafting: a match of at least kLookupStrong tokens is used on its own; a shorter one (>= kLookupMin) only
// when the MTP head's first draft agrees with it (two independent sources agreeing; HyperQwen's rule).
static constexpr int kLookupStrong = 8;
static constexpr int kLookupMin = 3;
static constexpr int kLookupMaxMatch = 32;
static constexpr int kLookupScan = 1 << 20;  // positions searched back (the whole context in practice)

static ggml_type parse_kv_type(const std::string& s) {
    if (s == "f16") return GGML_TYPE_F16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    fatal("unsupported KV cache type '%s' (f16, q8_0, q4_0)", s.c_str());
}

Engine::Engine(const EngineConfig& cfg) : cfg_(cfg) {
    gpu_ = ggml_backend_cuda_init(0);
    if (!gpu_) fatal("no CUDA device");
    cpu_ = ggml_backend_cpu_init();
    if (cfg_.n_threads <= 0) cfg_.n_threads = std::max(1, (int) std::thread::hardware_concurrency() * 3 / 4);
    ggml_backend_cpu_set_n_threads(cpu_, cfg_.n_threads);

    ModelOptions mo;
    mo.path = cfg.model_path;
    mo.mtp_path = cfg.mtp_path;
    mo.load_mtp = cfg.n_draft > 0;
    mo.cpu_repack = cfg.cpu_repack;
    model_ = std::make_unique<Model>(mo);
    tok_.load(model_->gguf());
    const HParams& hp = model_->hp;

    // --- plan ---
    size_t vram_free = 0, vram_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(gpu_), &vram_free, &vram_total);
    PlanInput pin;
    pin.vram_free = vram_free;
    pin.vram_margin = cfg.vram_margin_mb << 20;
    pin.state.n_ctx = std::min(cfg.n_ctx > 0 ? cfg.n_ctx : hp.n_ctx_train, hp.n_ctx_train);
    pin.state.n_snapshots = (model_->has_mtp() ? cfg.n_draft : 0) + 1;
    pin.state.max_batch = std::max(cfg.n_ubatch, cfg.n_draft + 1);
    pin.state.mtp_window = std::max(cfg.mtp_window, 2 * pin.state.max_batch);
    pin.state.snap_type = cfg.snap_type == "f32" ? GGML_TYPE_F32 : cfg.snap_type == "bf16" ? GGML_TYPE_BF16 : GGML_TYPE_F16;
    pin.verify_batch = (model_->has_mtp() ? cfg.n_draft : 0) + 1;
    const std::string kvt = cfg.kv_type == "auto" ? (pin.state.n_ctx > 131072 ? "q4_0" : "q8_0") : cfg.kv_type;
    pin.state.type_k = pin.state.type_v = parse_kv_type(kvt);
    pin.kv_auto = cfg.kv_place == "auto";
    if (!pin.kv_auto) pin.state.kv_place = cfg.kv_place == "gpu" ? Place::Gpu : Place::Host;
    // VRAM is planned for prompts up to `base_kv` positions; longer prompts make room by demoting FFN blocks
    // (context-elastic placement, see ensure_prefill_vram()).
    n_ctx_cfg_ = pin.state.n_ctx;
    // Planned for decoding; every prefill makes room for its own scratch by demoting FFN blocks, then promotes them.
    pin.compute_reserve = cfg.compute_reserve_mb > 0 ? (cfg.compute_reserve_mb << 20) : kDecodeReserve;
    plan_ = plan_placement(*model_, pin);
    // enough demotable blocks (RAM copies prepared at load) to fit the prefill scratch of a full-context prompt
    int n_elastic = 0;
    {
        const size_t worst = compute_need(pin.state.n_ctx, cfg.n_ubatch);
        const size_t extra = worst > pin.compute_reserve ? worst - pin.compute_reserve : 0;
        size_t got = 0;
        for (size_t i = plan_.n_spilled_blocks; i < plan_.block_order.size() && got < extra; ++i, ++n_elastic) {
            for (auto& w : model_->weights) {
                const Layer& L = model_->layers[plan_.block_order[i]];
                if (w.t == L.ffn_gate || w.t == L.ffn_up || w.t == L.ffn_down) got += ggml_nbytes(w.t);
            }
        }
        if (n_elastic > 0) n_elastic += 1;  // headroom for estimate error
    }
    KLOG_INFO("VRAM free %.2f / %.2f GiB; compute reserve %.0f MiB", vram_free / GiB, vram_total / GiB, pin.compute_reserve / MiB);
    KLOG_INFO("plan: %s", plan_.summary().c_str());

    // --- load ---
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_);
    ggml_backend_buffer_type_t host_buft = ggml_backend_cuda_host_buffer_type();  // pinned: fast streaming to the GPU
    model_->load(gpu_buft, host_buft, plan_.block_order, plan_.n_spilled_blocks, n_elastic);
    margin_ = pin.vram_margin;
    // A KV cache in RAM is mapped into the GPU's address space: attention runs on the GPU and reads the cells it
    // needs over PCIe, instead of round-tripping every attention layer through the CPU.
    ggml_backend_buffer_type_t kv_host_buft = cfg.kv_zero_copy ? ggml_backend_cuda_mapped_host_buffer_type(0) : nullptr;
    kv_mapped_ = kv_host_buft != nullptr;
    if (!kv_host_buft) kv_host_buft = host_buft;
    state_ = std::make_unique<State>(*model_, plan_.state, gpu_buft, kv_host_buft);
    if (!cfg.mmproj_path.empty()) {
        vision_ = std::make_unique<VisionModel>(cfg.mmproj_path, host_buft);
        if (vision_->hp.proj_dim != hp.n_embd) fatal("the vision encoder projects to %d dimensions, the model has %d", vision_->hp.proj_dim, hp.n_embd);
        image_pad_id_ = tok_.find("<|image_pad|>");
        if (image_pad_id_ < 0) fatal("the model's vocabulary has no <|image_pad|> token");
    }

    // --- scheduler ---
    ggml_backend_t backends[2] = {gpu_, cpu_};
    sched_ = ggml_backend_sched_new(backends, nullptr, 2, 32768, false, true);
    mtp_logits_.resize(hp.n_vocab);

    warmup();
    KLOG_INFO("ready: %.2f GiB VRAM still free", this->vram_free() / GiB);
}

Engine::~Engine() {
    vision_.reset();
    state_.reset();
    model_.reset();
    if (sched_) ggml_backend_sched_free(sched_);
    if (cpu_) ggml_backend_free(cpu_);
    if (gpu_) ggml_backend_free(gpu_);
}

size_t Engine::compute_need(int n_kv, int n_tokens) const {
    // Prefill compute buffer: activations of one chunk, streamed copies of spilled weights, and the f16 K/V copy that
    // flash attention makes of a quantized cache (the dominant term for long prompts).
    const HParams& hp = model_->hp;
    const size_t act = (size_t) 4 * std::min(n_tokens, cfg_.n_ubatch) * hp.n_ff * sizeof(float);
    const size_t streamed = 160ull << 20;
    const size_t fa_conv = (size_t) n_kv * hp.kv_row() * 2 * sizeof(uint16_t);
    return act + streamed + fa_conv;
}

void Engine::ensure_prefill_vram(int n_kv_end, int n_tokens) {
    const size_t need = compute_need(padded_n_kv(n_kv_end, n_ctx_cfg_), n_tokens);
    const size_t have_sched = ggml_backend_sched_get_buffer_size(sched_, gpu_);
    int n = 0;
    while (vram_free() + have_sched < need + margin_ && model_->next_demote_bytes() > 0 && model_->demote()) ++n;
    if (n > 0) KLOG_INFO("long prompt (%d positions): moved %d FFN blocks to RAM for the prefill scratch (%d blocks in RAM)", n_kv_end, n,
                         model_->n_host_blocks());
}

void Engine::relax_after_prefill() {
    // Drop the prefill-sized compute buffers, then bring FFN blocks back to VRAM while it has room. Decode needs
    // little scratch, and every block in VRAM saves CPU time on every token.
    if (model_->n_host_blocks() == 0) return;
    const size_t decode_need = kDecodeReserve;
    const size_t sched_buf = ggml_backend_sched_get_buffer_size(sched_, gpu_);
    if (sched_buf <= decode_need && vram_free() < margin_ + decode_need + model_->next_promote_bytes()) return;
    ggml_backend_sched_free(sched_);
    ggml_backend_t backends[2] = {gpu_, cpu_};
    sched_ = ggml_backend_sched_new(backends, nullptr, 2, 32768, false, true);
    int n = 0;
    while (model_->next_promote_bytes() > 0 && vram_free() > margin_ + decode_need + model_->next_promote_bytes() &&
           model_->n_host_blocks() > plan_.n_spilled_blocks && model_->promote())
        ++n;
    if (n > 0) KLOG_DEBUG("moved %d FFN blocks back to VRAM for decoding (%d in RAM)", n, model_->n_host_blocks());
}

void Engine::warmup() {
    // CUDA loads kernel modules lazily on first use; run every graph shape once so the first request does not pay
    // for it (and the CUDA memory pool reaches its working size).
    const double t0 = now_ms();
    std::vector<int32_t> toks(64, tok_.n_vocab() > 1000 ? 1000 : 0);
    std::vector<float> logits;
    Sampler smp(SamplerParams{});
    generate(toks, 3, smp, [](int32_t) { return true; });
    reset();
    if (vision_) {
        ImageRGB grey;
        grey.w = grey.h = 64;
        grey.px.assign((size_t) 64 * 64 * 3, 128);
        encode_image(grey);
    }
    KLOG_DEBUG("warm-up %.0f ms", now_ms() - t0);
}

void Engine::reset() {
    state_->clear(gpu_);
    mtp_ready_ = false;
    cache_.clear();
    cache_rope_after_.clear();
}

int Engine::rope_next() const { return cache_rope_after_.empty() ? 0 : cache_rope_after_.back(); }

void Engine::run(bool mtp, const RunArgs& a, ggml_tensor* h_src, ggml_tensor* h_dst) {
    const int n_ctx = state_->cfg().n_ctx;
    if (a.cell0 + a.n > n_ctx) fatal("context full (%d tokens)", n_ctx);
    // the MTP layer's cache is a ring buffer of mtp_cells() positions; the main caches are indexed by cell
    const int ring = mtp && mtp_cells(state_->cfg()) < n_ctx ? mtp_cells(state_->cfg()) : 0;
    const int n_kv = ring ? std::min(ring, padded_n_kv(a.cell0 + a.n, n_ctx)) : padded_n_kv(a.cell0 + a.n, n_ctx);
    bool embd_in = false;
    for (int i = 0; i < a.n && !embd_in; ++i) embd_in = a.cells[i].emb != nullptr;

    // One metadata buffer per graph kind: ggml-cuda caches CUDA graphs by the address of each split's first node, so
    // kinds built in the same memory would keep invalidating each other's cached graphs.
    std::vector<uint8_t>& meta = meta_buf_[(mtp ? 2 : 0) + (a.n >= 32 ? 1 : 0)];
    if (meta.empty()) meta.resize(ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false));
    ggml_init_params ip{meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(ip);
    FwdGraph g = mtp ? build_mtp_graph(ctx, *model_, *state_, a.n, n_kv, a.n_out, h_src, h_dst, embd_in)
                     : build_main_graph(ctx, *model_, *state_, a.n, n_kv, a.n_out, embd_in);

    ggml_backend_sched_reset(sched_);
    // With the KV cache in plain RAM, small batches attend on the CPU (copying the cache to the GPU would cost more).
    if (state_->cfg().kv_place == Place::Host && !state_->kv_mapped() && a.n < 32) {
        for (ggml_tensor* t : g.attn_nodes) ggml_backend_sched_set_tensor_backend(sched_, t, cpu_);
    }
    if (!ggml_backend_sched_alloc_graph(sched_, g.gf)) fatal("failed to allocate the compute graph");

    // inputs: tokens, or embedding rows (image cells as given, text cells looked up in token_embd on the host)
    if (embd_in) {
        const HParams& hp = model_->hp;
        std::vector<float> e((size_t) a.n * hp.n_embd);
        const ggml_tensor* te = model_->tok_embd;
        const auto* traits = ggml_get_type_traits(te->type);
        for (int i = 0; i < a.n; ++i) {
            float* dst = e.data() + (size_t) i * hp.n_embd;
            if (a.cells[i].emb) {
                std::memcpy(dst, a.cells[i].emb, (size_t) hp.n_embd * sizeof(float));
            } else {
                const char* row = (const char*) te->data + (size_t) a.cells[i].id * te->nb[1];
                traits->to_float(row, dst, hp.n_embd);
            }
        }
        ggml_backend_tensor_set(g.in.embd, e.data(), 0, e.size() * sizeof(float));
    } else {
        std::vector<int32_t> toks(a.n);
        for (int i = 0; i < a.n; ++i) toks[i] = a.cells[i].id;
        ggml_backend_tensor_set(g.in.tokens, toks.data(), 0, toks.size() * sizeof(int32_t));
    }
    // M-RoPE positions, section-major: [t..., y..., x..., 0...]; text cells have t = y = x
    std::vector<int32_t> pos((size_t) a.n * 4);
    std::vector<int64_t> idx(a.n);
    for (int i = 0; i < a.n; ++i) {
        for (int j = 0; j < 3; ++j) pos[(size_t) j * a.n + i] = a.cells[i].rope[j];
        pos[(size_t) 3 * a.n + i] = 0;
        idx[i] = ring > 0 ? (a.cell0 + i) % ring : a.cell0 + i;
    }
    ggml_backend_tensor_set(g.in.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.in.kv_idx, idx.data(), 0, idx.size() * sizeof(int64_t));
    {
        std::vector<ggml_fp16_t> mask((size_t) n_kv * a.n);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        const int end = a.cell0 + a.n - 1;  // last cell written by this pass
        for (int i = 0; i < a.n; ++i) {
            const int p = a.cell0 + i;
            ggml_fp16_t* row = mask.data() + (size_t) i * n_kv;
            if (ring == 0) {
                for (int j = 0; j < n_kv; ++j) row[j] = (j >= a.kv_lo && j <= p) ? zero : ninf;
            } else {
                // ring slot j holds the newest cell q <= end with q % ring == j (after this pass's writes)
                for (int j = 0; j < n_kv; ++j) {
                    const int q = end - ((end - j) % ring + ring) % ring;
                    row[j] = (q >= a.kv_lo && q <= p && q > p - ring) ? zero : ninf;
                }
            }
        }
        ggml_backend_tensor_set(g.in.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    if (g.in.out_ids) {
        std::vector<int32_t> out(a.n_out);
        for (int i = 0; i < a.n_out; ++i) out[i] = a.n - a.n_out + i;
        ggml_backend_tensor_set(g.in.out_ids, out.data(), 0, out.size() * sizeof(int32_t));
    }

    if (ggml_backend_sched_graph_compute(sched_, g.gf) != GGML_STATUS_SUCCESS) fatal("graph compute failed");
    if (a.logits && g.logits) {
        a.logits->resize((size_t) a.n_out * n_vocab());
        ggml_backend_tensor_get(g.logits, a.logits->data(), 0, a.logits->size() * sizeof(float));
    }
    ggml_free(ctx);
}

void Engine::eval_cells(const Cell* cells, int n, int n_out, std::vector<float>* logits) {
    KLEIN_ASSERT(n <= state_->cfg().max_batch);
    run(false, RunArgs{cells, n, state_->n_past, 0, n_out, logits}, nullptr, nullptr);
    state_->n_past += n;
    for (int i = 0; i < n; ++i) {
        cache_.push_back(cells[i].id);
        cache_rope_after_.push_back(cells[i].rope_after);
    }
}

std::vector<Engine::Cell> Engine::text_cells(const int32_t* tokens, int n, int rope0) {
    std::vector<Cell> c(n);
    for (int i = 0; i < n; ++i) c[i] = Cell{tokens[i], nullptr, {rope0 + i, rope0 + i, rope0 + i}, rope0 + i + 1};
    return c;
}

void Engine::eval(const int32_t* tokens, int n, int n_out, std::vector<float>* logits) {
    const auto c = text_cells(tokens, n, rope_next());
    eval_cells(c.data(), n, n_out, logits);
}

void Engine::mtp_pass(const Cell* cells, int n, int cell0, bool from_main, int h_row0, int n_out) {
    ggml_tensor* hsrc_base = from_main ? state_->hidden() : state_->mtp_hidden();
    // The MTP pass's own hidden states go to mtp_hidden rows right after the rows it reads (no overlap).
    const int dst_row0 = from_main ? 0 : h_row0 + 1;
    KLEIN_ASSERT(dst_row0 + n <= state_->cfg().max_batch && h_row0 + n <= state_->cfg().max_batch);

    ggml_init_params ip{ggml_tensor_overhead() * 4, nullptr, true};
    ggml_context* vctx = ggml_init(ip);
    ggml_tensor* h_src = ggml_view_2d(vctx, hsrc_base, hsrc_base->ne[0], n, hsrc_base->nb[1], (size_t) h_row0 * hsrc_base->nb[1]);
    ggml_tensor* h_dst = ggml_view_2d(vctx, state_->mtp_hidden(), hsrc_base->ne[0], n, hsrc_base->nb[1],
                                      (size_t) dst_row0 * hsrc_base->nb[1]);
    ggml_backend_view_init(h_src);
    ggml_backend_view_init(h_dst);
    std::vector<float> tmp;
    run(true, RunArgs{cells, n, cell0, 1, n_out, n_out > 0 ? &tmp : nullptr}, h_src, h_dst);
    if (n_out > 0) {
        std::copy(tmp.end() - n_vocab(), tmp.end(), mtp_logits_.begin());
        mtp_ready_ = true;
    }
    mtp_h_row_ = dst_row0 + n - 1;
    ggml_free(vctx);
}

int Engine::lookup_drafts(int32_t next, int n, std::vector<int32_t>& out) const {
    // Find the earlier occurrence of the longest suffix of (cached tokens + next) and propose the n tokens that
    // followed it (as in HyperQwen's lookup drafting). The continuation may overlap the suffix itself, so repeating
    // patterns are proposed from their own period. Image cells have negative ids and never match.
    out.clear();
    const int len = (int) cache_.size() + 1;
    auto at = [&](int i) { return i == len - 1 ? next : cache_[i]; };
    int best_len = 0, best_pos = -1;
    const int max_scan = std::min(len - 1, kLookupScan);
    for (int j = len - 2; j >= len - 1 - max_scan && j >= 0; --j) {
        if (at(j) != next) continue;
        int l = 1;
        while (l < kLookupMaxMatch && j - l >= 0 && at(j - l) == at(len - 1 - l)) ++l;
        if (l > best_len) {
            best_len = l;
            best_pos = j;
            if (l >= kLookupMaxMatch) break;
        }
    }
    if (best_pos < 0) return 0;
    for (int k = 1; k <= n && best_pos + k < len; ++k) {
        const int32_t t = at(best_pos + k);
        if (t < 0) break;
        out.push_back(t);
    }
    return best_len;
}

int32_t Engine::mtp_draft_token() const {
    return (int32_t) (std::max_element(mtp_logits_.begin(), mtp_logits_.end()) - mtp_logits_.begin());
}

void Engine::prefill_cells(const std::vector<Cell>& cells, std::vector<float>& last_logits, GenStats* stats) {
    const double t0 = now_ms();
    const bool use_mtp = model_->has_mtp() && cfg_.n_draft > 0;
    const int n = (int) cells.size();
    if (n > cfg_.n_draft + 1) ensure_prefill_vram(state_->n_past + n, n);
    for (int i = 0; i < n; i += cfg_.n_ubatch) {
        const int nb = std::min(cfg_.n_ubatch, n - i);
        const bool last = i + nb >= n;
        const int cell0 = state_->n_past;
        eval_cells(cells.data() + i, nb, last ? 1 : 0, last ? &last_logits : nullptr);
        // MTP: cell c takes cell c's input and the main hidden state of cell c - 1. The chunk's last cell needs the
        // next input, which for the final chunk is only known after sampling (see generate()).
        if (use_mtp) {
            const int m = last ? nb - 1 : nb;
            if (m > 0) mtp_pass(cells.data() + i + 1, m, cell0 + 1, true, 0, 0);
        }
    }
    if (stats) {
        stats->n_prompt += n;
        stats->t_prompt_ms += now_ms() - t0;
    }
}

void Engine::prefill(const std::vector<int32_t>& tokens, std::vector<float>& last_logits, GenStats* stats) {
    prefill_cells(text_cells(tokens.data(), (int) tokens.size(), rope_next()), last_logits, stats);
}

std::vector<Engine::Cell> Engine::build_cells(const std::vector<int32_t>& prompt, const std::vector<PromptImage>& images) const {
    // Each image placeholder token (<|image_pad|>) becomes the image's grid of embeddings. Image cells share the
    // M-RoPE time position; their height/width positions follow the grid; the text after an image continues at
    // max(grid_w, grid_h) past the image's start (Qwen-VL convention, as in llama.cpp's mtmd).
    std::vector<Cell> cells;
    cells.reserve(prompt.size());
    int r = 0;
    size_t img = 0;
    for (int32_t t : prompt) {
        if (t == image_pad_id_ && img < images.size()) {
            const PromptImage& im = images[img++];
            const int after = r + std::max(im.grid_x, im.grid_y);
            for (int k = 0; k < im.grid_x * im.grid_y; ++k) {
                // pseudo token id: negative, derived from the image content and the cell, so prompt reuse compares
                // images by content
                const uint64_t h = (im.hash ^ (0x9E3779B97F4A7C15ull * (uint64_t) (k + 1))) * 0xBF58476D1CE4E5B9ull;
                const int32_t id = -1 - (int32_t) (h >> 34);
                cells.push_back(Cell{id, im.embd.data() + (size_t) k * model_->hp.n_embd,
                                     {r, r + k / im.grid_x, r + k % im.grid_x}, after});
            }
            r = after;
        } else {
            cells.push_back(Cell{t, nullptr, {r, r, r}, r + 1});
            ++r;
        }
    }
    if (img < images.size()) fatal("%zu image(s) given but only %zu image placeholder(s) in the prompt", images.size(), img);
    return cells;
}

GenStats Engine::generate(const std::vector<int32_t>& prompt, int n_predict, Sampler& sampler,
                          const std::function<bool(int32_t)>& on_token) {
    return generate(prompt, {}, n_predict, sampler, on_token);
}

GenStats Engine::generate(const std::vector<int32_t>& prompt, const std::vector<PromptImage>& images, int n_predict, Sampler& sampler,
                          const std::function<bool(int32_t)>& on_token) {
    GenStats st;
    const int n_vocab_ = n_vocab();
    std::vector<float> logits;
    const std::vector<Cell> all = build_cells(prompt, images);
    st.n_prompt_total = (int) all.size();
    if ((int) all.size() + cfg_.n_draft + 2 > n_ctx())
        throw std::runtime_error(format("the prompt (%zu tokens) does not fit the context (%d)", all.size(), n_ctx()));
    size_t common = 0;
    while (common < cache_.size() && common < all.size() && cache_[common] == all[common].id) ++common;
    std::vector<Cell> todo;
    if (common > 0 && common == cache_.size() && common < all.size()) {
        todo.assign(all.begin() + common, all.end());
        // the cached part may have shifted rope positions (images): continue from where the cache ends
        const int shift = rope_next() - all[common].rope[0];
        if (shift != 0)
            for (Cell& c : todo) {
                for (int& p : c.rope) p += shift;
                c.rope_after += shift;
            }
        KLOG_DEBUG("reusing %zu cached tokens, prefilling %zu", common, todo.size());
    } else {
        reset();
        todo = all;
    }
    prefill_cells(todo, logits, &st);
    relax_after_prefill();

    const double t0 = now_ms();
    const bool use_mtp = model_->has_mtp() && cfg_.n_draft > 0;
    int32_t next = sampler.sample(logits.data(), n_vocab_);
    sampler.accept(next);
    st.n_gen = 1;
    if (!on_token(next) || tok_.is_eog(next) || n_predict <= 1) {
        st.t_gen_ms = now_ms() - t0;
        return st;
    }
    // finish the MTP pass for the prompt's last cell, now that its next token is known
    if (use_mtp) {
        const auto c = text_cells(&next, 1, rope_next());
        mtp_pass(c.data(), 1, state_->n_past, true, (int) ((todo.size() - 1) % cfg_.n_ubatch), 1);
    }

    std::vector<int32_t> batch;
    bool stop = false;
    while (!stop && st.n_gen < n_predict && state_->n_past + cfg_.n_draft + 2 < n_ctx()) {
        const int base = state_->n_past;  // cell of `next`
        const int rbase = rope_next();    // its rope position
        batch.assign(1, next);
        double tp = now_ms();
        if (use_mtp) {
            // draft: d1 from the pending MTP logits. When the text is being reproduced from the context (a long
            // earlier occurrence of what was just written), the rest comes from that occurrence for free; otherwise
            // the MTP head is chained on its own hidden states.
            int32_t d = mtp_draft_token();
            batch.push_back(d);
            std::vector<int32_t> look;
            const int match = cfg_.lookup ? lookup_drafts(next, cfg_.n_draft, look) : 0;
            const bool strong = match >= kLookupStrong && (int) look.size() == cfg_.n_draft;
            const bool agreed = match >= kLookupMin && (int) look.size() == cfg_.n_draft && look[0] == d;
            if (strong || agreed) {
                batch.assign(1, next);
                batch.insert(batch.end(), look.begin(), look.end());
                st.n_lookup++;
            } else {
                for (int j = 1; j < cfg_.n_draft; ++j) {
                    const auto c = text_cells(&d, 1, rbase + j);
                    mtp_pass(c.data(), 1, base + j, false, mtp_h_row_, 1);
                    d = mtp_draft_token();
                    batch.push_back(d);
                }
            }
        }
        const int nb = (int) batch.size();
        st.t_draft_ms += now_ms() - tp;
        tp = now_ms();
        const auto vc = text_cells(batch.data(), nb, rbase);
        eval_cells(vc.data(), nb, nb, &logits);
        st.t_verify_ms += now_ms() - tp;
        tp = now_ms();
        st.n_steps++;
        st.n_drafted += nb - 1;

        // verify: sample the target at each position, accept drafts while they match
        int accepted = 0;
        for (int i = 0; i < nb; ++i) {
            const int32_t t = sampler.sample(logits.data() + (size_t) i * n_vocab_, n_vocab_);
            sampler.accept(t);
            st.n_gen++;
            next = t;
            if (!on_token(t) || tok_.is_eog(t) || st.n_gen >= n_predict) { stop = true; accepted = i; break; }
            if (i + 1 < nb && batch[i + 1] == t) { accepted = i + 1; continue; }
            accepted = i;
            break;
        }
        st.n_accepted += accepted;
        st.t_sample_ms += now_ms() - tp;
        tp = now_ms();
        // keep `next` + accepted drafts in the caches; roll the DeltaNet state back past the rejected ones
        const int rejected = (nb - 1) - accepted;
        state_->n_past = base + accepted + 1;
        cache_.resize(state_->n_past);
        cache_rope_after_.resize(state_->n_past);
        if (rejected > 0) state_->rollback_recurrent(gpu_, rejected);
        st.t_rollback_ms += now_ms() - tp;
        tp = now_ms();
        if (stop) break;
        if (use_mtp) {
            // true MTP pass over the kept cells base+1 .. base+accepted+1 (main hidden rows 0..accepted)
            std::vector<int32_t> mt(batch.begin() + 1, batch.begin() + 1 + accepted);
            mt.push_back(next);
            const auto mc = text_cells(mt.data(), (int) mt.size(), rbase + 1);
            mtp_pass(mc.data(), (int) mc.size(), base + 1, true, 0, 1);
        }
        st.t_mtp_ms += now_ms() - tp;
    }
    st.t_gen_ms = now_ms() - t0;
    KLOG_DEBUG("compute buffers after decode: GPU %.0f MiB, CPU %.0f MiB; VRAM free %.0f MiB; %d FFN blocks in RAM",
               ggml_backend_sched_get_buffer_size(sched_, gpu_) / MiB, ggml_backend_sched_get_buffer_size(sched_, cpu_) / MiB,
               vram_free() / MiB, model_->n_host_blocks());
    return st;
}

std::vector<std::pair<std::string, double>> Engine::vision_debug(const ImageInput& in) {
    if (!vision_) throw std::runtime_error("no vision encoder (--mmproj)");
    std::vector<std::pair<std::string, double>> sums;
    std::vector<uint8_t>& meta = vision_meta_;
    if (meta.empty()) meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params ip{meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(ip);
    VisionModel::Graph g = vision_->build(ctx, in.w, in.h);
    ggml_backend_sched_reset(sched_);
    struct Cb {
        std::vector<std::pair<std::string, double>>* sums;
    } cb{&sums};
    ggml_backend_sched_set_eval_callback(
        sched_,
        [](ggml_tensor* t, bool ask, void* ud) -> bool {
            const std::string n = ggml_get_name(t);
            const bool want = n == "patch_bias" || n == "inp_pos_emb" || n.rfind("layer_out-", 0) == 0 || n == "post_ln" || n == "v_embd";
            if (ask) return want;
            if (want && t->type == GGML_TYPE_F32) {
                std::vector<float> v(ggml_nelements(t));
                ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
                double s = 0;
                for (float x : v) s += x;
                ((Cb*) ud)->sums->push_back({n, s});
            }
            return true;
        },
        &cb);
    if (!ggml_backend_sched_alloc_graph(sched_, g.gf)) fatal("failed to allocate the vision graph");
    ggml_backend_tensor_set(g.inp_raw, in.planar.data(), 0, in.planar.size() * sizeof(float));
    const auto pos = vision_->positions(in.w, in.h);
    ggml_backend_tensor_set(g.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(sched_, g.gf) != GGML_STATUS_SUCCESS) fatal("vision graph compute failed");
    ggml_backend_sched_set_eval_callback(sched_, nullptr, nullptr);
    ggml_free(ctx);
    return sums;
}

PromptImage Engine::encode_image(const ImageRGB& img) {
    if (!vision_) throw std::runtime_error("this server has no vision encoder (start it with --mmproj FILE)");
    const double t0 = now_ms();
    const VisionModel::HParams& vh = vision_->hp;
    ImageInput in = preprocess_qwen_vl(img, vision_->align(), cfg_.image_min_tokens, cfg_.image_max_tokens, vh.mean, vh.std);
    const int n_patches = (in.w / vh.patch) * (in.h / vh.patch);

    // VRAM for the encoder's activations (its weights are streamed from RAM): borrow it from FFN blocks
    const size_t need = (size_t) n_patches * 64 * 1024 + (192ull << 20);
    const size_t have_sched = ggml_backend_sched_get_buffer_size(sched_, gpu_);
    int demoted = 0;
    while (vram_free() + have_sched < need + margin_ && model_->next_demote_bytes() > 0 && model_->demote()) ++demoted;

    std::vector<uint8_t>& meta = vision_meta_;
    if (meta.empty()) meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(8192, false));
    ggml_init_params ip{meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(ip);
    VisionModel::Graph g = vision_->build(ctx, in.w, in.h);
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, g.gf)) fatal("failed to allocate the vision graph");
    ggml_backend_tensor_set(g.inp_raw, in.planar.data(), 0, in.planar.size() * sizeof(float));
    const auto pos = vision_->positions(in.w, in.h);
    ggml_backend_tensor_set(g.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(sched_, g.gf) != GGML_STATUS_SUCCESS) fatal("vision graph compute failed");

    PromptImage out;
    out.grid_x = g.grid_x;
    out.grid_y = g.grid_y;
    out.embd.resize((size_t) g.n_tokens * vh.proj_dim);
    ggml_backend_tensor_get(g.out, out.embd.data(), 0, out.embd.size() * sizeof(float));
    ggml_free(ctx);
    // content hash (FNV-1a over the pixels and size) for prompt reuse
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    mix((uint64_t) img.w);
    mix((uint64_t) img.h);
    for (uint8_t b : img.px) mix(b);
    out.hash = h;
    relax_after_prefill();
    KLOG_INFO("image %dx%d -> %dx%d pixels -> %d tokens (%dx%d) in %.0f ms%s", img.w, img.h, in.w, in.h, g.n_tokens, g.grid_x, g.grid_y,
              now_ms() - t0, demoted ? format(", %d FFN blocks lent to the encoder", demoted).c_str() : "");
    return out;
}

}  // namespace klein

namespace klein {
size_t Engine::vram_free() const {
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(gpu_), &f, &t);
    return f;
}
}  // namespace klein
