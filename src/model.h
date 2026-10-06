// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf_file.h"

namespace klein {

// Hyperparameters of the GGUF architecture "qwen35" (Qwen3.5 / 3.6 / 3.8 dense hybrid models).
struct HParams {
    int n_layer = 0;           // decoder layers (without MTP)
    int n_mtp = 0;             // MTP (nextn) layers stored after the decoder layers
    int n_embd = 0;
    int n_ff = 0;
    int n_vocab = 0;
    int n_ctx_train = 0;
    int full_attn_interval = 4;
    // full attention
    int n_head = 0, n_head_kv = 0, head_dim = 0;
    int n_rot = 0;
    int rope_sections[4] = {0, 0, 0, 0};
    float rope_freq_base = 10000.0f;
    float rms_eps = 1e-6f;
    // gated delta net
    int ssm_conv_kernel = 4;
    int ssm_d_state = 128;     // head dim of k and v
    int ssm_n_group = 16;      // k heads
    int ssm_dt_rank = 48;      // v heads
    int ssm_d_inner = 6144;    // v heads * head dim

    bool is_recurrent(int il) const { return il < n_layer && ((il + 1) % full_attn_interval) != 0; }
    int n_attn_layers() const { return n_layer / full_attn_interval; }
    int conv_channels() const { return ssm_d_inner + 2 * ssm_n_group * ssm_d_state; }
    int64_t kv_row() const { return (int64_t) n_head_kv * head_dim; }  // K (or V) elements per token per layer
};

struct Layer {
    bool recurrent = false;
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* post_norm = nullptr;
    // full attention (also used by the MTP block)
    ggml_tensor* wq = nullptr;  // query + output gate, interleaved per head: [n_embd, 2 * n_head * head_dim]
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* q_norm = nullptr;
    ggml_tensor* k_norm = nullptr;
    // gated delta net
    ggml_tensor* wqkv = nullptr;
    ggml_tensor* wz = nullptr;
    ggml_tensor* ssm_conv1d = nullptr;
    ggml_tensor* ssm_dt = nullptr;
    ggml_tensor* ssm_a = nullptr;
    ggml_tensor* ssm_beta = nullptr;
    ggml_tensor* ssm_alpha = nullptr;
    ggml_tensor* ssm_norm = nullptr;
    ggml_tensor* ssm_out = nullptr;
    // FFN
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
    // MTP (nextn) block only
    ggml_tensor* eh_proj = nullptr;
    ggml_tensor* enorm = nullptr;
    ggml_tensor* hnorm = nullptr;
    ggml_tensor* head_norm = nullptr;
};

enum class Place { Gpu, Host };

struct WeightInfo {
    ggml_tensor* t = nullptr;    // the tensor used in graphs
    const GgufFile* src = nullptr;
    ggml_tensor* src_meta = nullptr;
    Place place = Place::Gpu;
    int layer = -1;              // -1: global tensor
};

struct ModelOptions {
    std::string path;
    std::string mtp_path;        // optional separate GGUF holding the MTP block (blk.<n_layer>.*)
    bool load_mtp = true;
    bool cpu_repack = true;      // spilled weights also get a CPU-optimized (repacked) copy for small batches
};

class Model {
public:
    HParams hp;
    ggml_tensor* tok_embd = nullptr;
    ggml_tensor* output_norm = nullptr;
    ggml_tensor* output = nullptr;
    std::vector<Layer> layers;   // n_layer decoder layers, then n_mtp MTP layers

    std::vector<WeightInfo> weights;

    // For spilled weights: a copy in ggml's repacked CPU layout (IQ4_XS transcoded to IQ4_NL first), used for
    // batches below the GPU offload threshold. The original stays in pinned memory for streaming during prefill.
    std::unordered_map<const ggml_tensor*, ggml_tensor*> cpu_alt;
    size_t cpu_alt_bytes = 0;

    explicit Model(const ModelOptions& opt);
    ~Model();

    bool has_mtp() const { return hp.n_mtp > 0 && layers.size() > (size_t) hp.n_layer && layers[hp.n_layer].eh_proj; }
    const Layer& mtp_layer() const { return layers[hp.n_layer]; }

    // Total bytes of weights per placement (after plan()).
    size_t bytes(Place p) const;
    size_t total_bytes() const;

    // Allocates backend buffers according to each WeightInfo::place and reads the data from disk.
    void load(ggml_backend_buffer_type_t gpu_buft, ggml_backend_buffer_type_t host_buft);

    const GgufFile& gguf() const { return *main_; }

private:
    std::unique_ptr<GgufFile> main_;
    std::unique_ptr<GgufFile> mtp_;
    bool cpu_repack_ = true;
    void make_cpu_copies();
    ggml_context* ctx_ = nullptr;  // tensor objects (no data)
    std::vector<ggml_backend_buffer_t> buffers_;

    void read_hparams();
    ggml_tensor* bind(const std::string& name, int layer, bool required, const GgufFile* from = nullptr);
};

}  // namespace klein
