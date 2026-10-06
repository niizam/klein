// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "model.h"

#include <algorithm>
#include <cstring>

#include "common.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "transcode.h"

namespace klein {

static constexpr const char* ARCH = "qwen35";

Model::Model(const ModelOptions& opt) : cpu_repack_(opt.cpu_repack) {
    main_ = std::make_unique<GgufFile>(opt.path);
    const std::string arch = main_->get_str("general.architecture");
    if (arch != ARCH) fatal("'%s' has architecture '%s'; klein supports '%s' (Qwen3.5/3.6/3.8 dense)", opt.path.c_str(), arch.c_str(), ARCH);
    read_hparams();

    if (opt.load_mtp && !opt.mtp_path.empty()) {
        mtp_ = std::make_unique<GgufFile>(opt.mtp_path);
        if (mtp_->get_str("general.architecture") != ARCH) fatal("MTP file '%s' is not a %s model", opt.mtp_path.c_str(), ARCH);
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 2048;
    ip.no_alloc = true;
    ctx_ = ggml_init(ip);

    tok_embd = bind("token_embd.weight", -1, true);
    output_norm = bind("output_norm.weight", -1, true);
    output = bind("output.weight", -1, false);
    if (!output) fatal("tied embeddings are not supported (no output.weight)");

    const int n_total = hp.n_layer + (opt.load_mtp ? hp.n_mtp : 0);
    layers.resize(n_total);
    auto blk = [](int il, const char* name) { return format("blk.%d.%s", il, name); };
    for (int il = 0; il < hp.n_layer; ++il) {
        Layer& L = layers[il];
        L.recurrent = hp.is_recurrent(il);
        L.attn_norm = bind(blk(il, "attn_norm.weight"), il, true);
        L.post_norm = bind(blk(il, "post_attention_norm.weight"), il, true);
        if (L.recurrent) {
            L.wqkv = bind(blk(il, "attn_qkv.weight"), il, true);
            L.wz = bind(blk(il, "attn_gate.weight"), il, true);
            L.ssm_conv1d = bind(blk(il, "ssm_conv1d.weight"), il, true);
            L.ssm_dt = bind(blk(il, "ssm_dt.bias"), il, true);
            L.ssm_a = bind(blk(il, "ssm_a"), il, true);
            L.ssm_beta = bind(blk(il, "ssm_beta.weight"), il, true);
            L.ssm_alpha = bind(blk(il, "ssm_alpha.weight"), il, true);
            L.ssm_norm = bind(blk(il, "ssm_norm.weight"), il, true);
            L.ssm_out = bind(blk(il, "ssm_out.weight"), il, true);
        } else {
            L.wq = bind(blk(il, "attn_q.weight"), il, true);
            L.wk = bind(blk(il, "attn_k.weight"), il, true);
            L.wv = bind(blk(il, "attn_v.weight"), il, true);
            L.wo = bind(blk(il, "attn_output.weight"), il, true);
            L.q_norm = bind(blk(il, "attn_q_norm.weight"), il, true);
            L.k_norm = bind(blk(il, "attn_k_norm.weight"), il, true);
        }
        L.ffn_gate = bind(blk(il, "ffn_gate.weight"), il, true);
        L.ffn_up = bind(blk(il, "ffn_up.weight"), il, true);
        L.ffn_down = bind(blk(il, "ffn_down.weight"), il, true);
    }
    for (int il = hp.n_layer; il < n_total; ++il) {
        const GgufFile* src = mtp_ ? mtp_.get() : main_.get();
        if (!src->tensor(blk(il, "nextn.eh_proj.weight"))) {
            KLOG_WARN("no MTP block in '%s'; speculative decoding with MTP is disabled", src->path().c_str());
            layers.resize(hp.n_layer);
            break;
        }
        Layer& L = layers[il];
        L.attn_norm = bind(blk(il, "attn_norm.weight"), il, true, src);
        L.post_norm = bind(blk(il, "post_attention_norm.weight"), il, true, src);
        L.wq = bind(blk(il, "attn_q.weight"), il, true, src);
        L.wk = bind(blk(il, "attn_k.weight"), il, true, src);
        L.wv = bind(blk(il, "attn_v.weight"), il, true, src);
        L.wo = bind(blk(il, "attn_output.weight"), il, true, src);
        L.q_norm = bind(blk(il, "attn_q_norm.weight"), il, true, src);
        L.k_norm = bind(blk(il, "attn_k_norm.weight"), il, true, src);
        L.ffn_gate = bind(blk(il, "ffn_gate.weight"), il, true, src);
        L.ffn_up = bind(blk(il, "ffn_up.weight"), il, true, src);
        L.ffn_down = bind(blk(il, "ffn_down.weight"), il, true, src);
        L.eh_proj = bind(blk(il, "nextn.eh_proj.weight"), il, true, src);
        L.enorm = bind(blk(il, "nextn.enorm.weight"), il, true, src);
        L.hnorm = bind(blk(il, "nextn.hnorm.weight"), il, true, src);
        L.head_norm = bind(blk(il, "nextn.shared_head_norm.weight"), il, false, src);
    }

    // The input embedding is only looked up (one row per token): it always stays in host memory.
    for (auto& w : weights) {
        if (w.t == tok_embd) w.place = Place::Host;
    }
}

Model::~Model() {
    for (auto b : buffers_) ggml_backend_buffer_free(b);
    if (ctx_) ggml_free(ctx_);
}

void Model::read_hparams() {
    const GgufFile& f = *main_;
    auto k = [](const char* s) { return std::string(ARCH) + "." + s; };
    const int block_count = (int) f.get_u32(k("block_count"));
    hp.n_mtp = (int) f.get_u32(k("nextn_predict_layers"), 0);
    hp.n_layer = block_count - hp.n_mtp;
    hp.n_embd = (int) f.get_u32(k("embedding_length"));
    hp.n_ff = (int) f.get_u32(k("feed_forward_length"));
    hp.n_ctx_train = (int) f.get_u32(k("context_length"));
    hp.full_attn_interval = (int) f.get_u32(k("full_attention_interval"), 4);
    hp.n_head = (int) f.get_u32(k("attention.head_count"));
    hp.n_head_kv = (int) f.get_u32(k("attention.head_count_kv"));
    hp.head_dim = (int) f.get_u32(k("attention.key_length"));
    if ((int) f.get_u32(k("attention.value_length"), hp.head_dim) != hp.head_dim) fatal("key_length != value_length is not supported");
    hp.n_rot = (int) f.get_u32(k("rope.dimension_count"));
    const auto sec = f.get_arr_int(k("rope.dimension_sections"));
    for (size_t i = 0; i < 4 && i < sec.size(); ++i) hp.rope_sections[i] = (int) sec[i];
    hp.rope_freq_base = f.get_f32(k("rope.freq_base"), 10000.0f);
    hp.rms_eps = f.get_f32(k("attention.layer_norm_rms_epsilon"), 1e-6f);
    hp.ssm_conv_kernel = (int) f.get_u32(k("ssm.conv_kernel"));
    hp.ssm_d_state = (int) f.get_u32(k("ssm.state_size"));
    hp.ssm_n_group = (int) f.get_u32(k("ssm.group_count"));
    hp.ssm_dt_rank = (int) f.get_u32(k("ssm.time_step_rank"));
    hp.ssm_d_inner = (int) f.get_u32(k("ssm.inner_size"));
    hp.n_vocab = (int) f.get_arr_n("tokenizer.ggml.tokens");
    if (hp.n_layer % hp.full_attn_interval != 0) fatal("n_layer %d is not a multiple of full_attention_interval", hp.n_layer);
    if (hp.ssm_d_inner != hp.ssm_dt_rank * hp.ssm_d_state) fatal("unexpected Gated DeltaNet shape");
}

ggml_tensor* Model::bind(const std::string& name, int layer, bool required, const GgufFile* from) {
    const GgufFile* src = from ? from : main_.get();
    ggml_tensor* meta = src->tensor(name);
    if (!meta) {
        if (required) fatal("tensor '%s' missing in '%s'", name.c_str(), src->path().c_str());
        return nullptr;
    }
    ggml_tensor* t = ggml_new_tensor(ctx_, meta->type, GGML_MAX_DIMS, meta->ne);
    ggml_set_name(t, name.c_str());
    WeightInfo w;
    w.t = t;
    w.src = src;
    w.src_meta = meta;
    w.layer = layer;
    weights.push_back(w);
    return t;
}

size_t Model::bytes(Place p) const {
    size_t s = 0;
    for (const auto& w : weights)
        if (w.place == p) s += ggml_nbytes(w.t);
    return s;
}

size_t Model::total_bytes() const { return bytes(Place::Gpu) + bytes(Place::Host); }

void Model::load(ggml_backend_buffer_type_t gpu_buft, ggml_backend_buffer_type_t host_buft) {
    const double t0 = now_ms();
    auto alloc_group = [&](Place p, ggml_backend_buffer_type_t buft) -> ggml_backend_buffer_t {
        size_t size = 0;
        const size_t align = ggml_backend_buft_get_alignment(buft);
        for (const auto& w : weights)
            if (w.place == p) size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, w.t), align);
        if (size == 0) return nullptr;
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, size);
        if (!buf) fatal("cannot allocate %.2f GiB of %s memory for weights", size / GiB, ggml_backend_buft_name(buft));
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_tallocr ta = ggml_tallocr_new(buf);
        for (auto& w : weights)
            if (w.place == p) ggml_tallocr_alloc(&ta, w.t);
        buffers_.push_back(buf);
        return buf;
    };
    alloc_group(Place::Gpu, gpu_buft);
    alloc_group(Place::Host, host_buft);

    // Read in file order, through a staging buffer for device tensors.
    std::vector<WeightInfo*> order;
    for (auto& w : weights) order.push_back(&w);
    std::sort(order.begin(), order.end(), [](const WeightInfo* a, const WeightInfo* b) {
        if (a->src != b->src) return a->src < b->src;
        return a->src->tensor_file_offset(a->src_meta) < b->src->tensor_file_offset(b->src_meta);
    });
    std::vector<uint8_t> staging(64u << 20);
    size_t done = 0;
    for (WeightInfo* w : order) {
        const size_t n = ggml_nbytes(w->t);
        if (ggml_backend_buffer_is_host(w->t->buffer)) {
            w->src->read_tensor_data(w->src_meta, w->t->data, 0, n);
        } else {
            for (size_t off = 0; off < n; off += staging.size()) {
                const size_t len = std::min(staging.size(), n - off);
                w->src->read_tensor_data(w->src_meta, staging.data(), off, len);
                ggml_backend_tensor_set(w->t, staging.data(), off, len);
            }
        }
        done += n;
    }
    if (cpu_repack_) make_cpu_copies();
    KLOG_INFO("loaded %.2f GiB of weights in %.1f s (%.2f GiB GPU, %.2f GiB host + %.2f GiB repacked CPU copies)", done / GiB,
              (now_ms() - t0) / 1000.0, bytes(Place::Gpu) / GiB, bytes(Place::Host) / GiB, cpu_alt_bytes / GiB);
}

void Model::make_cpu_copies() {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t repack = nullptr;
    if (get_extra) {
        for (auto p = get_extra(cpu_dev); p && *p; ++p) {
            if (std::strcmp(ggml_backend_buft_name(*p), "CPU_REPACK") == 0) repack = *p;
        }
    }
    if (!repack) {
        KLOG_WARN("no repacked CPU layout available; spilled weights use the plain CPU kernels");
        return;
    }
    // Would the repack buffer accept a matrix of this type and shape for a small-batch mul_mat?
    auto repackable = [&](ggml_type type, const int64_t* ne) {
        ggml_init_params ip{ggml_tensor_overhead() * 4, nullptr, true};
        ggml_context* c = ggml_init(ip);
        ggml_tensor* w = ggml_new_tensor_2d(c, type, ne[0], ne[1]);
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, ne[0], 4);
        ggml_tensor* y = ggml_mul_mat(c, w, x);
        ggml_backend_buffer_t tb = ggml_backend_buft_alloc_buffer(repack, 0x100);
        w->buffer = tb;
        const bool ok = ggml_backend_dev_supports_op(cpu_dev, y);
        w->buffer = nullptr;
        ggml_backend_buffer_free(tb);
        ggml_free(c);
        return ok;
    };

    struct Job {
        WeightInfo* w;
        ggml_tensor* alt;
    };
    std::vector<Job> jobs;
    for (auto& w : weights) {
        if (w.place != Place::Host || w.t == tok_embd || ggml_n_dims(w.t) != 2) continue;
        const ggml_type target = w.t->type == GGML_TYPE_IQ4_XS ? GGML_TYPE_IQ4_NL : w.t->type;
        if (!repackable(target, w.t->ne)) continue;
        ggml_tensor* alt = ggml_new_tensor_2d(ctx_, target, w.t->ne[0], w.t->ne[1]);
        ggml_format_name(alt, "%s.cpu", ggml_get_name(w.t));
        jobs.push_back({&w, alt});
    }
    if (jobs.empty()) return;
    const size_t align = ggml_backend_buft_get_alignment(repack);
    size_t size = 0;
    for (auto& j : jobs) size += GGML_PAD(ggml_backend_buft_get_alloc_size(repack, j.alt), align);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(repack, size);
    if (!buf) fatal("cannot allocate %.2f GiB for repacked CPU weights (use --no-repack)", size / GiB);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    buffers_.push_back(buf);
    ggml_tallocr ta = ggml_tallocr_new(buf);
    std::vector<uint8_t> tmp;
    for (auto& j : jobs) {
        ggml_tallocr_alloc(&ta, j.alt);
        const void* src = j.w->t->data;
        if (j.w->t->type != j.alt->type) {
            tmp.resize(ggml_nbytes(j.alt));
            transcode_iq4xs_to_iq4nl(src, tmp.data(), ggml_nelements(j.w->t));
            src = tmp.data();
        }
        ggml_backend_tensor_set(j.alt, src, 0, ggml_nbytes(j.alt));  // the repack buffer converts the layout here
        cpu_alt[j.w->t] = j.alt;
        cpu_alt_bytes += ggml_nbytes(j.alt);
    }
}

}  // namespace klein
