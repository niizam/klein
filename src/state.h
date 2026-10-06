// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "model.h"

namespace klein {

struct StateConfig {
    int n_ctx = 0;
    ggml_type type_k = GGML_TYPE_Q8_0;
    ggml_type type_v = GGML_TYPE_Q8_0;
    Place kv_place = Place::Gpu;
    int n_snapshots = 1;      // recurrent state slots: 1 + max draft tokens (for rollback after rejected drafts)
    ggml_type snap_type = GGML_TYPE_BF16;  // storage of the rollback snapshots (slots 1..); slot 0 is always f32
    int max_batch = 512;      // largest batch a forward pass takes (bounds the hidden-state buffer)
    int mtp_window = 16384;   // MTP head attention window (ring buffer cells, kept in VRAM)
};

// The memory of one sequence: the KV cache of the full-attention layers (and of the MTP layer), the Gated
// DeltaNet recurrent states with rollback snapshots, and the last hidden states (input of the MTP head).
class State {
public:
    State(const Model& model, const StateConfig& cfg, ggml_backend_buffer_type_t gpu_buft, ggml_backend_buffer_type_t host_buft);
    ~State();
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    const StateConfig& cfg() const { return cfg_; }

    // KV cache per full-attention layer, indexed by decoder layer id (nullptr for recurrent layers);
    // the MTP layer's cache is at index n_layer. Shape [n_head_kv * head_dim, n_ctx].
    ggml_tensor* k(int il) const { return k_[il]; }
    ggml_tensor* v(int il) const { return v_[il]; }
    // For a KV cache in RAM: the same memory seen as a plain host buffer. Large batches (prefill) use this view: the
    // scheduler writes it on the CPU and copies the used range to the GPU in bulk before attention, which beats
    // reading it in small pieces across PCIe. Small batches use k()/v(), mapped into the GPU (zero-copy).
    ggml_tensor* k_bulk(int il) const { return k_host_.empty() || !k_host_[il] ? k_[il] : k_host_[il]; }
    ggml_tensor* v_bulk(int il) const { return v_host_.empty() || !v_host_[il] ? v_[il] : v_host_[il]; }
    // Recurrent state per DeltaNet layer. conv: [row, n_snapshots] f32, slot 0 = current, slot s = s tokens back.
    // ssm: [row, 1] f32 (current); ssm_snap: [row, n_snapshots - 1] in snap_type, row s - 1 = s tokens back.
    ggml_tensor* conv(int il) const { return conv_[il]; }
    ggml_tensor* ssm(int il) const { return ssm_[il]; }
    ggml_tensor* ssm_snap(int il) const { return ssm_snap_.empty() ? nullptr : ssm_snap_[il]; }
    // Final-norm hidden states of the last main forward pass: [n_embd, max_batch].
    ggml_tensor* hidden() const { return hidden_; }
    // Final-norm hidden states written by MTP passes (input of the next chained draft): [n_embd, max_batch].
    ggml_tensor* mtp_hidden() const { return mtp_hidden_; }

    size_t kv_bytes() const { return kv_bytes_; }
    // KV cache in RAM that the GPU reads directly (mapped pinned memory); false for VRAM or plain RAM.
    bool kv_mapped() const { return kv_mapped_; }
    size_t recurrent_bytes() const { return rec_bytes_; }

    int n_past = 0;      // tokens in the main model's cache
    int n_past_mtp = 0;  // positions whose MTP-layer KV was computed from the main model's true hidden states

    void clear(ggml_backend_t gpu);

    // Make slot `s` the current recurrent state (after `s` trailing tokens of the last batch were rejected).
    void rollback_recurrent(ggml_backend_t gpu, int s);

private:
    const Model& model_;
    StateConfig cfg_;
    ggml_context* ctx_ = nullptr;
    std::vector<ggml_backend_buffer_t> buffers_;
    std::vector<ggml_tensor*> k_, v_, conv_, ssm_;
    std::vector<ggml_tensor*> k_host_, v_host_;
    std::vector<ggml_tensor*> ssm_snap_;
    bool kv_mapped_ = false;
    ggml_tensor* hidden_ = nullptr;
    ggml_tensor* mtp_hidden_ = nullptr;
    size_t kv_bytes_ = 0, rec_bytes_ = 0;
};

// Bytes the state needs, for planning before anything is allocated.
size_t state_kv_bytes(const HParams& hp, const StateConfig& cfg, bool with_mtp);
size_t state_recurrent_bytes(const HParams& hp, const StateConfig& cfg);
size_t state_mtp_kv_bytes(const HParams& hp, const StateConfig& cfg);
int mtp_cells(const StateConfig& cfg);

}  // namespace klein
