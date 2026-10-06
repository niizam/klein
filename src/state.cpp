// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "state.h"

#include "common.h"
#include <algorithm>

#include "ggml-alloc.h"

namespace klein {

static int64_t conv_row(const HParams& hp) { return (int64_t) (hp.ssm_conv_kernel - 1) * hp.conv_channels(); }
static int64_t ssm_row(const HParams& hp) { return (int64_t) hp.ssm_d_state * hp.ssm_d_state * hp.ssm_dt_rank; }

int mtp_cells(const StateConfig& cfg) { return std::min(cfg.n_ctx, cfg.mtp_window); }

size_t state_kv_bytes(const HParams& hp, const StateConfig& cfg, bool with_mtp) {
    (void) with_mtp;  // the MTP ring buffer is counted in state_gpu_fixed_bytes()
    const size_t row = ggml_row_size(cfg.type_k, hp.kv_row()) + ggml_row_size(cfg.type_v, hp.kv_row());
    return (size_t) hp.n_attn_layers() * row * (size_t) cfg.n_ctx;
}

size_t state_mtp_kv_bytes(const HParams& hp, const StateConfig& cfg) {
    const size_t row = ggml_row_size(cfg.type_k, hp.kv_row()) + ggml_row_size(cfg.type_v, hp.kv_row());
    return row * (size_t) mtp_cells(cfg);
}

size_t state_recurrent_bytes(const HParams& hp, const StateConfig& cfg) {
    const int n_rec = hp.n_layer - hp.n_attn_layers();
    return (size_t) n_rec * (size_t) (conv_row(hp) + ssm_row(hp)) * sizeof(float) * (size_t) cfg.n_snapshots;
}

State::State(const Model& model, const StateConfig& cfg, ggml_backend_buffer_type_t gpu_buft, ggml_backend_buffer_type_t host_buft)
    : model_(model), cfg_(cfg) {
    const HParams& hp = model.hp;
    const int n_all = (int) model.layers.size();
    k_.assign(n_all, nullptr);
    v_.assign(n_all, nullptr);
    conv_.assign(n_all, nullptr);
    ssm_.assign(n_all, nullptr);

    ggml_init_params ip{ggml_tensor_overhead() * (size_t) (n_all * 4 + 8), nullptr, true};
    ctx_ = ggml_init(ip);

    std::vector<ggml_tensor*> kv_tensors, gpu_tensors;
    for (int il = 0; il < n_all; ++il) {
        const bool is_mtp = il >= hp.n_layer;
        if (is_mtp) {
            // The MTP head attends over a sliding window kept as a ring buffer in VRAM: drafts only need recent
            // context (the main model verifies every token), and the cost stays flat as the context grows.
            const int cells = mtp_cells(cfg);
            k_[il] = ggml_new_tensor_2d(ctx_, cfg.type_k, hp.kv_row(), cells);
            v_[il] = ggml_new_tensor_2d(ctx_, cfg.type_v, hp.kv_row(), cells);
            ggml_format_name(k_[il], "cache_k_mtp");
            ggml_format_name(v_[il], "cache_v_mtp");
            gpu_tensors.push_back(k_[il]);
            gpu_tensors.push_back(v_[il]);
        } else if (!hp.is_recurrent(il)) {
            k_[il] = ggml_new_tensor_2d(ctx_, cfg.type_k, hp.kv_row(), cfg.n_ctx);
            v_[il] = ggml_new_tensor_2d(ctx_, cfg.type_v, hp.kv_row(), cfg.n_ctx);
            ggml_format_name(k_[il], "cache_k_l%d", il);
            ggml_format_name(v_[il], "cache_v_l%d", il);
            kv_tensors.push_back(k_[il]);
            kv_tensors.push_back(v_[il]);
        } else {
            conv_[il] = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, conv_row(hp), cfg.n_snapshots);
            ssm_[il] = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, ssm_row(hp), cfg.n_snapshots);
            ggml_format_name(conv_[il], "cache_conv_l%d", il);
            ggml_format_name(ssm_[il], "cache_ssm_l%d", il);
            gpu_tensors.push_back(conv_[il]);
            gpu_tensors.push_back(ssm_[il]);
        }
    }
    hidden_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, hp.n_embd, cfg.max_batch);
    ggml_set_name(hidden_, "hidden_last");
    gpu_tensors.push_back(hidden_);
    mtp_hidden_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, hp.n_embd, cfg.max_batch);
    ggml_set_name(mtp_hidden_, "mtp_hidden_last");
    gpu_tensors.push_back(mtp_hidden_);
    if (cfg.kv_place == Place::Gpu) gpu_tensors.insert(gpu_tensors.end(), kv_tensors.begin(), kv_tensors.end());

    auto alloc = [&](const std::vector<ggml_tensor*>& ts, ggml_backend_buffer_type_t buft, const char* what) {
        if (ts.empty()) return;
        const size_t align = ggml_backend_buft_get_alignment(buft);
        size_t size = 0;
        for (auto t : ts) size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, size);
        if (!buf) fatal("cannot allocate %.2f GiB for the %s", size / GiB, what);
        ggml_tallocr ta = ggml_tallocr_new(buf);
        for (auto t : ts) ggml_tallocr_alloc(&ta, t);
        buffers_.push_back(buf);
    };
    alloc(gpu_tensors, gpu_buft, "GPU state (recurrent states / KV cache)");
    if (cfg.kv_place == Place::Host) alloc(kv_tensors, host_buft, "host KV cache");

    for (auto t : kv_tensors) kv_bytes_ += ggml_nbytes(t);
    for (int il = 0; il < n_all; ++il) {
        if (conv_[il]) rec_bytes_ += ggml_nbytes(conv_[il]) + ggml_nbytes(ssm_[il]);
    }
    for (auto b : buffers_) ggml_backend_buffer_clear(b, 0);
}

State::~State() {
    for (auto b : buffers_) ggml_backend_buffer_free(b);
    if (ctx_) ggml_free(ctx_);
}

void State::clear(ggml_backend_t gpu) {
    (void) gpu;
    // KV cells beyond n_past are masked, so only the recurrent state must be zeroed.
    for (size_t il = 0; il < conv_.size(); ++il) {
        if (conv_[il]) {
            ggml_backend_tensor_memset(conv_[il], 0, 0, ggml_nbytes(conv_[il]));
            ggml_backend_tensor_memset(ssm_[il], 0, 0, ggml_nbytes(ssm_[il]));
        }
    }
    n_past = 0;
    n_past_mtp = 0;
}

void State::rollback_recurrent(ggml_backend_t gpu, int s) {
    if (s <= 0) return;
    KLEIN_ASSERT(s < cfg_.n_snapshots);
    // One small graph of device-side copies: slot s -> slot 0 for every DeltaNet layer.
    ggml_init_params ip{ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true};
    ggml_context* c = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(c);
    for (size_t il = 0; il < conv_.size(); ++il) {
        for (ggml_tensor* t : {conv_[il], ssm_[il]}) {
            if (!t) continue;
            ggml_tensor* src = ggml_view_1d(c, t, t->ne[0], (size_t) s * t->nb[1]);
            ggml_tensor* dst = ggml_view_1d(c, t, t->ne[0], 0);
            ggml_backend_view_init(src);
            ggml_backend_view_init(dst);
            ggml_build_forward_expand(gf, ggml_cpy(c, src, dst));
        }
    }
    ggml_backend_graph_compute(gpu, gf);
    ggml_free(c);
}

}  // namespace klein
