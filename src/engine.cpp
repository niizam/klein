// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "common.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "graph.h"

namespace klein {

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
    ggml_backend_cpu_set_n_threads(cpu_, cfg.n_threads);

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
    pin.verify_batch = (model_->has_mtp() ? cfg.n_draft : 0) + 1;
    const std::string kvt = cfg.kv_type == "auto" ? (pin.state.n_ctx > 131072 ? "q4_0" : "q8_0") : cfg.kv_type;
    pin.state.type_k = pin.state.type_v = parse_kv_type(kvt);
    pin.kv_auto = cfg.kv_place == "auto";
    if (!pin.kv_auto) pin.state.kv_place = cfg.kv_place == "gpu" ? Place::Gpu : Place::Host;
    if (cfg.compute_reserve_mb > 0) {
        pin.compute_reserve = cfg.compute_reserve_mb << 20;
    } else {
        // activations of one prefill chunk + room for streamed (spilled) weight copies + the f16 K/V conversion
        // that flash attention does for quantized caches during prefill
        const size_t act = (size_t) 4 * cfg.n_ubatch * hp.n_ff * sizeof(float);
        const size_t streamed = 160ull << 20;
        const size_t fa_conv = (size_t) std::min(pin.state.n_ctx, 65536) * hp.kv_row() * 2 * sizeof(uint16_t);
        pin.compute_reserve = act + streamed + fa_conv;
    }
    plan_ = plan_placement(*model_, pin);
    KLOG_INFO("VRAM free %.2f / %.2f GiB; compute reserve %.0f MiB", vram_free / GiB, vram_total / GiB, pin.compute_reserve / MiB);
    KLOG_INFO("plan: %s", plan_.summary().c_str());

    // --- load ---
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_);
    ggml_backend_buffer_type_t host_buft = ggml_backend_cuda_host_buffer_type();  // pinned: fast streaming to the GPU
    model_->load(gpu_buft, host_buft);
    state_ = std::make_unique<State>(*model_, plan_.state, gpu_buft, host_buft);

    // --- scheduler ---
    ggml_backend_t backends[2] = {gpu_, cpu_};
    sched_ = ggml_backend_sched_new(backends, nullptr, 2, 32768, false, true);
    meta_buf_.resize(ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false));
    mtp_logits_.resize(hp.n_vocab);

    size_t f2 = 0, t2 = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(gpu_), &f2, &t2);
    KLOG_INFO("ready: %.2f GiB VRAM still free", f2 / GiB);
}

Engine::~Engine() {
    state_.reset();
    model_.reset();
    if (sched_) ggml_backend_sched_free(sched_);
    if (cpu_) ggml_backend_free(cpu_);
    if (gpu_) ggml_backend_free(gpu_);
}

void Engine::reset() {
    state_->clear(gpu_);
    mtp_ready_ = false;
    cache_.clear();
}

void Engine::run(bool mtp, const RunArgs& a, ggml_tensor* h_src, ggml_tensor* h_dst) {
    const int n_ctx = state_->cfg().n_ctx;
    if (a.pos0 + a.n > n_ctx) fatal("context full (%d tokens)", n_ctx);
    const int n_kv = padded_n_kv(a.pos0 + a.n, n_ctx);

    ggml_init_params ip{meta_buf_.size(), meta_buf_.data(), true};
    ggml_context* ctx = ggml_init(ip);
    FwdGraph g = mtp ? build_mtp_graph(ctx, *model_, *state_, a.n, n_kv, a.n_out, h_src, h_dst)
                     : build_main_graph(ctx, *model_, *state_, a.n, n_kv, a.n_out);

    ggml_backend_sched_reset(sched_);
    // With the KV cache in RAM, small batches attend on the CPU (copying the cache to the GPU would cost more).
    if (state_->cfg().kv_place == Place::Host && a.n < 32) {
        for (ggml_tensor* t : g.attn_nodes) ggml_backend_sched_set_tensor_backend(sched_, t, cpu_);
    }
    if (!ggml_backend_sched_alloc_graph(sched_, g.gf)) fatal("failed to allocate the compute graph");

    // inputs
    ggml_backend_tensor_set(g.in.tokens, a.tokens, 0, (size_t) a.n * sizeof(int32_t));
    std::vector<int32_t> pos((size_t) a.n * 4);
    std::vector<int64_t> idx(a.n);
    for (int i = 0; i < a.n; ++i) {
        for (int j = 0; j < 3; ++j) pos[(size_t) j * a.n + i] = a.pos0 + i;
        pos[(size_t) 3 * a.n + i] = 0;
        idx[i] = a.pos0 + i;
    }
    ggml_backend_tensor_set(g.in.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.in.kv_idx, idx.data(), 0, idx.size() * sizeof(int64_t));
    {
        std::vector<ggml_fp16_t> mask((size_t) n_kv * a.n);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < a.n; ++i) {
            const int p = a.pos0 + i;
            ggml_fp16_t* row = mask.data() + (size_t) i * n_kv;
            for (int j = 0; j < n_kv; ++j) row[j] = (j >= a.kv_lo && j <= p) ? zero : ninf;
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

void Engine::eval(const int32_t* tokens, int n, int n_out, std::vector<float>* logits) {
    KLEIN_ASSERT(n <= state_->cfg().max_batch);
    run(false, RunArgs{tokens, n, state_->n_past, 0, n_out, logits}, nullptr, nullptr);
    state_->n_past += n;
    cache_.insert(cache_.end(), tokens, tokens + n);
}

void Engine::mtp_pass(const int32_t* tokens, int n, int pos0, bool from_main, int h_row0, int n_out) {
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
    std::vector<float>* out = n_out > 0 ? &mtp_logits_ : nullptr;
    std::vector<float> tmp;
    run(true, RunArgs{tokens, n, pos0, 1, n_out, n_out > 0 ? &tmp : nullptr}, h_src, h_dst);
    if (out) {
        std::copy(tmp.end() - n_vocab(), tmp.end(), mtp_logits_.begin());
        mtp_ready_ = true;
    }
    mtp_h_row_ = dst_row0 + n - 1;
    ggml_free(vctx);
}

int32_t Engine::mtp_draft_token() const {
    return (int32_t) (std::max_element(mtp_logits_.begin(), mtp_logits_.end()) - mtp_logits_.begin());
}

void Engine::prefill(const std::vector<int32_t>& tokens, std::vector<float>& last_logits, GenStats* stats) {
    const double t0 = now_ms();
    const bool use_mtp = model_->has_mtp() && cfg_.n_draft > 0;
    const int n = (int) tokens.size();
    for (int i = 0; i < n; i += cfg_.n_ubatch) {
        const int nb = std::min(cfg_.n_ubatch, n - i);
        const bool last = i + nb >= n;
        const int pos0 = state_->n_past;
        eval(tokens.data() + i, nb, last ? 1 : 0, last ? &last_logits : nullptr);
        // MTP: position p takes token p and the main hidden state of position p - 1. The chunk's last position
        // needs the next token, which for the final chunk is only known after sampling (see generate()).
        if (use_mtp) {
            const int m = last ? nb - 1 : nb;
            if (m > 0) mtp_pass(tokens.data() + i + 1, m, pos0 + 1, true, 0, 0);
        }
    }
    if (stats) {
        stats->n_prompt += n;
        stats->t_prompt_ms += now_ms() - t0;
    }
}

GenStats Engine::generate(const std::vector<int32_t>& prompt, int n_predict, Sampler& sampler,
                          const std::function<bool(int32_t)>& on_token) {
    GenStats st;
    const int n_vocab_ = n_vocab();
    std::vector<float> logits;
    size_t common = 0;
    while (common < cache_.size() && common < prompt.size() && cache_[common] == prompt[common]) ++common;
    std::vector<int32_t> todo;
    if (common > 0 && common == cache_.size() && common < prompt.size()) {
        todo.assign(prompt.begin() + common, prompt.end());
        KLOG_DEBUG("reusing %zu cached tokens, prefilling %zu", common, todo.size());
    } else {
        reset();
        todo = prompt;
    }
    prefill(todo, logits, &st);

    const double t0 = now_ms();
    const bool use_mtp = model_->has_mtp() && cfg_.n_draft > 0;
    int32_t next = sampler.sample(logits.data(), n_vocab_);
    sampler.accept(next);
    st.n_gen = 1;
    if (!on_token(next) || tok_.is_eog(next) || n_predict <= 1) {
        st.t_gen_ms = now_ms() - t0;
        return st;
    }
    // finish the MTP pass for the prompt's last position, now that its next token is known
    if (use_mtp) mtp_pass(&next, 1, state_->n_past, true, (int) ((todo.size() - 1) % cfg_.n_ubatch), 1);

    std::vector<int32_t> batch;
    bool stop = false;
    while (!stop && st.n_gen < n_predict && state_->n_past + cfg_.n_draft + 2 < n_ctx()) {
        const int base = state_->n_past;  // position of `next`
        batch.assign(1, next);
        double tp = now_ms();
        if (use_mtp) {
            // draft: d1 from the pending MTP logits, then chain the MTP head on its own hidden states
            int32_t d = mtp_draft_token();
            batch.push_back(d);
            for (int j = 1; j < cfg_.n_draft; ++j) {
                mtp_pass(&d, 1, base + j, false, mtp_h_row_, 1);
                d = mtp_draft_token();
                batch.push_back(d);
            }
        }
        const int nb = (int) batch.size();
        st.t_draft_ms += now_ms() - tp;
        tp = now_ms();
        eval(batch.data(), nb, nb, &logits);
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
        if (rejected > 0) state_->rollback_recurrent(gpu_, rejected);
        st.t_rollback_ms += now_ms() - tp;
        tp = now_ms();
        if (stop) break;
        if (use_mtp) {
            // true MTP pass over the kept positions base+1 .. base+accepted+1 (main hidden rows 0..accepted)
            std::vector<int32_t> mt(batch.begin() + 1, batch.begin() + 1 + accepted);
            mt.push_back(next);
            mtp_pass(mt.data(), (int) mt.size(), base + 1, true, 0, 1);
        }
        st.t_mtp_ms += now_ms() - tp;
    }
    st.t_gen_ms = now_ms() - t0;
    return st;
}

}  // namespace klein

namespace klein {
size_t Engine::vram_free() const {
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(gpu_), &f, &t);
    return f;
}
}  // namespace klein
