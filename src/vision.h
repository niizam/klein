// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf_file.h"
#include "image.h"

namespace klein {

// The Qwen3-VL vision encoder (mmproj GGUF, projector "qwen3vl_merger"): a ViT over 16x16 patches with 2D M-RoPE and
// learned position embeddings, then a 2x2 patch merger projecting into the language model's embedding space.
// Weights stay in (pinned) RAM; the scheduler streams them to the GPU while an image is encoded, so the encoder
// takes no VRAM between images.
class VisionModel {
public:
    struct HParams {
        int n_embd = 0, n_ff = 0, n_head = 0, n_layer = 0, patch = 0, merge = 2, proj_dim = 0, pos_side = 0;
        float eps = 1e-6f;
        float mean[3] = {0.5f, 0.5f, 0.5f};
        float std[3] = {0.5f, 0.5f, 0.5f};
    } hp;

    VisionModel(const std::string& path, ggml_backend_buffer_type_t buft);
    ~VisionModel();

    int align() const { return hp.patch * hp.merge; }
    size_t weight_bytes() const { return weight_bytes_; }

    struct Graph {
        ggml_cgraph* gf = nullptr;
        ggml_tensor* inp_raw = nullptr;    // F32 [w, h, 3]
        ggml_tensor* positions = nullptr;  // I32 [4 * n_patches]
        ggml_tensor* out = nullptr;        // F32 [proj_dim, n_tokens]
        int n_tokens = 0, grid_x = 0, grid_y = 0;  // merged-token grid
    };
    Graph build(ggml_context* ctx, int w, int h) const;
    // M-RoPE positions of the encoder's patches for an image of w x h pixels.
    std::vector<int32_t> positions(int w, int h) const;

private:
    struct Layer {
        ggml_tensor *ln1_w, *ln1_b, *ln2_w, *ln2_b, *qkv_w, *qkv_b, *o_w, *o_b, *up_w, *up_b, *down_w, *down_b;
    };
    std::unique_ptr<GgufFile> f_;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    size_t weight_bytes_ = 0;
    ggml_tensor *patch0_ = nullptr, *patch1_ = nullptr, *patch_b_ = nullptr, *pos_ = nullptr;
    ggml_tensor *post_w_ = nullptr, *post_b_ = nullptr, *mm0_w_ = nullptr, *mm0_b_ = nullptr, *mm2_w_ = nullptr, *mm2_b_ = nullptr;
    std::vector<Layer> layers_;
};

}  // namespace klein
