// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <vector>

#include "ggml.h"
#include "model.h"
#include "state.h"

namespace klein {

// Inputs of one forward pass over `n_tokens` consecutive positions starting at `pos0` (single sequence).
struct FwdInputs {
    ggml_tensor* tokens = nullptr;   // I32 [n_tokens]
    ggml_tensor* pos = nullptr;      // I32 [4 * n_tokens], M-RoPE sections: [p..., p..., p..., 0...]
    ggml_tensor* kv_idx = nullptr;   // I64 [n_tokens], KV cells written (= positions)
    ggml_tensor* mask = nullptr;     // F16 [n_kv, n_tokens], causal mask
    ggml_tensor* out_ids = nullptr;  // I32 [n_out], rows whose logits are computed
};

struct FwdGraph {
    ggml_cgraph* gf = nullptr;
    FwdInputs in;
    ggml_tensor* logits = nullptr;   // F32 [n_vocab, n_out]
    std::vector<ggml_tensor*> attn_nodes;  // flash-attention nodes (pinned to the CPU when the KV cache is in host memory)
    int n_tokens = 0, n_kv = 0, n_out = 0;
};

// Main model: embeddings -> 64 hybrid layers -> final norm (stored into State::hidden) -> LM head.
FwdGraph build_main_graph(ggml_context* ctx, const Model& m, const State& st, int n_tokens, int n_kv, int n_out);

// MTP head over `n_tokens` positions: input token embeddings + hidden states from `h_src` (a view of
// [n_embd, n_tokens]). Writes its own final-norm hidden states to `h_dst` (may be nullptr).
FwdGraph build_mtp_graph(ggml_context* ctx, const Model& m, const State& st, int n_tokens, int n_kv, int n_out,
                         ggml_tensor* h_src, ggml_tensor* h_dst);

// KV view length used for a pass ending at position `n_past_after` (rounded up so graph shapes repeat).
int padded_n_kv(int n_past_after, int n_ctx);

}  // namespace klein
