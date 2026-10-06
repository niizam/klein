// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The Qwen3-VL vision encoder as a ggml graph. The structure follows llama.cpp's tools/mtmd/models/qwen3vl.cpp and
// clip.cpp (MIT), which klein uses as the numerical reference.
#include "vision.h"

#include <cmath>

#include "common.h"
#include "ggml-alloc.h"
#include "gguf.h"

namespace klein {

VisionModel::VisionModel(const std::string& path, ggml_backend_buffer_type_t buft) {
    f_ = std::make_unique<GgufFile>(path);
    const GgufFile& f = *f_;
    if (f.get_str("general.architecture") != "clip" || f.get_str("clip.projector_type") != "qwen3vl_merger")
        fatal("'%s' is not a Qwen3-VL vision projector (mmproj with projector_type qwen3vl_merger)", path.c_str());
    hp.n_embd = (int) f.get_u32("clip.vision.embedding_length");
    hp.n_ff = (int) f.get_u32("clip.vision.feed_forward_length");
    hp.n_head = (int) f.get_u32("clip.vision.attention.head_count");
    hp.n_layer = (int) f.get_u32("clip.vision.block_count");
    hp.patch = (int) f.get_u32("clip.vision.patch_size");
    hp.merge = (int) f.get_u32("clip.vision.spatial_merge_size", 2);
    hp.proj_dim = (int) f.get_u32("clip.vision.projection_dim");
    hp.eps = f.get_f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
    if (hp.merge != 2) fatal("unsupported spatial merge size %d", hp.merge);
    {
        gguf_context* g = f.gguf();
        auto read3 = [&](const char* key, float* dst) {
            const int64_t id = gguf_find_key(g, key);
            if (id >= 0 && gguf_get_arr_n(g, id) >= 3) {
                const float* d = (const float*) gguf_get_arr_data(g, id);
                for (int i = 0; i < 3; ++i) dst[i] = d[i];
            }
        };
        read3("clip.vision.image_mean", hp.mean);
        read3("clip.vision.image_std", hp.std);
    }
    for (int il = 0; il < hp.n_layer; ++il) {
        if (f.get_arr_n("clip.vision.is_deepstack_layers") > 0) {
            const auto ds = f.get_arr_int("clip.vision.is_deepstack_layers");
            if (il < (int) ds.size() && ds[il]) fatal("vision encoders with deepstack layers are not supported yet");
        }
    }

    ggml_init_params ip{ggml_tensor_overhead() * 512, nullptr, true};
    ctx_ = ggml_init(ip);
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> pairs;  // (ours, file meta)
    auto bind = [&](const std::string& name) {
        ggml_tensor* meta = f.tensor(name);
        if (!meta) fatal("vision tensor '%s' missing in '%s'", name.c_str(), path.c_str());
        ggml_tensor* t = ggml_new_tensor(ctx_, meta->type, GGML_MAX_DIMS, meta->ne);
        ggml_set_name(t, name.c_str());
        pairs.push_back({t, meta});
        return t;
    };
    patch0_ = bind("v.patch_embd.weight");
    patch1_ = bind("v.patch_embd.weight.1");
    patch_b_ = bind("v.patch_embd.bias");
    pos_ = bind("v.position_embd.weight");
    post_w_ = bind("v.post_ln.weight");
    post_b_ = bind("v.post_ln.bias");
    mm0_w_ = bind("mm.0.weight");
    mm0_b_ = bind("mm.0.bias");
    mm2_w_ = bind("mm.2.weight");
    mm2_b_ = bind("mm.2.bias");
    for (int il = 0; il < hp.n_layer; ++il) {
        auto n = [&](const char* s) { return format("v.blk.%d.%s", il, s); };
        layers_.push_back({bind(n("ln1.weight")), bind(n("ln1.bias")), bind(n("ln2.weight")), bind(n("ln2.bias")),
                           bind(n("attn_qkv.weight")), bind(n("attn_qkv.bias")), bind(n("attn_out.weight")), bind(n("attn_out.bias")),
                           bind(n("ffn_up.weight")), bind(n("ffn_up.bias")), bind(n("ffn_down.weight")), bind(n("ffn_down.bias"))});
    }
    hp.pos_side = (int) std::lround(std::sqrt((double) pos_->ne[1]));

    buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);
    if (!buf_) fatal("cannot allocate %s for the vision encoder", ggml_backend_buft_name(buft));
    ggml_backend_buffer_set_usage(buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> tmp;
    for (auto& [t, meta] : pairs) {
        const size_t n = ggml_nbytes(t);
        tmp.resize(n);
        f.read_tensor_data(meta, tmp.data(), 0, n);
        ggml_backend_tensor_set(t, tmp.data(), 0, n);
        weight_bytes_ += n;
    }
    KLOG_INFO("vision encoder: %d layers, %d-pixel patches, %.2f GiB (in RAM, streamed to the GPU per image)", hp.n_layer, hp.patch,
              weight_bytes_ / GiB);
}

VisionModel::~VisionModel() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

std::vector<int32_t> VisionModel::positions(int w, int h) const {
    const int pw = w / hp.patch, ph = h / hp.patch, n = pw * ph;
    std::vector<int32_t> pos((size_t) n * 4);
    int ptr = 0;
    for (int y = 0; y < ph; y += hp.merge)
        for (int x = 0; x < pw; x += hp.merge)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    pos[ptr] = y + dy;
                    pos[n + ptr] = x + dx;
                    pos[2 * n + ptr] = y + dy;
                    pos[3 * n + ptr] = x + dx;
                    ++ptr;
                }
    return pos;
}

VisionModel::Graph VisionModel::build(ggml_context* ctx, int w, int h) const {
    Graph g;
    g.gf = ggml_new_graph_custom(ctx, 8192, false);
    const int px = w / hp.patch, py = h / hp.patch, n_pos = px * py;
    const int E = hp.n_embd, H = hp.n_head, D = E / H;
    g.grid_x = px / 2;
    g.grid_y = py / 2;
    g.n_tokens = n_pos / 4;

    g.inp_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w, h, 3);
    ggml_set_name(g.inp_raw, "v_inp_raw");
    ggml_set_input(g.inp_raw);
    g.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n_pos * 4);
    ggml_set_name(g.positions, "v_positions");
    ggml_set_input(g.positions);

    // a still image is both frames of the temporal patch: conv with each half of the 3D patch kernel and add
    ggml_tensor* inp = ggml_add(ctx, ggml_conv_2d(ctx, patch0_, g.inp_raw, hp.patch, hp.patch, 0, 0, 1, 1),
                                ggml_conv_2d(ctx, patch1_, g.inp_raw, hp.patch, hp.patch, 0, 0, 1, 1));  // [px, py, E]

    // reorder patches so each 2x2 merge window is contiguous
    auto merge_order = [&](ggml_tensor* t) {
        t = ggml_cont_4d(ctx, t, E * 2, px / 2, py, 1);
        t = ggml_reshape_4d(ctx, t, E * 2, px / 2, 2, py / 2);
        t = ggml_permute(ctx, t, 0, 2, 1, 3);
        return ggml_cont_3d(ctx, t, E, n_pos, 1);
    };
    inp = merge_order(ggml_permute(ctx, inp, 1, 2, 0, 3));
    inp = ggml_add(ctx, inp, patch_b_);
    ggml_set_name(inp, "patch_bias");

    // learned position embeddings, bilinearly resized from the trained grid
    ggml_tensor* pe = pos_;
    if (px != hp.pos_side || py != hp.pos_side) {
        pe = ggml_reshape_3d(ctx, pe, E, hp.pos_side, hp.pos_side);
        pe = ggml_permute(ctx, pe, 2, 0, 1, 3);
        pe = ggml_interpolate(ctx, pe, px, py, E, 1, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
        pe = ggml_permute(ctx, pe, 1, 2, 0, 3);
        pe = ggml_cont_2d(ctx, pe, E, n_pos);
    }
    inp = ggml_add(ctx, inp, merge_order(pe));
    ggml_set_name(inp, "inp_pos_emb");

    int sections[4] = {D / 4, D / 4, D / 4, D / 4};
    const float kq_scale = 1.0f / std::sqrt((float) D);
    ggml_tensor* x = inp;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[il];
        ggml_tensor* cur = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, hp.eps), L.ln1_w), L.ln1_b);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, L.qkv_w, cur), L.qkv_b);
        ggml_tensor* q = ggml_view_3d(ctx, cur, D, H, n_pos, ggml_row_size(cur->type, D), cur->nb[1], 0);
        ggml_tensor* k = ggml_view_3d(ctx, cur, D, H, n_pos, ggml_row_size(cur->type, D), cur->nb[1], ggml_row_size(cur->type, E));
        ggml_tensor* v = ggml_view_3d(ctx, cur, D, H, n_pos, ggml_row_size(cur->type, D), cur->nb[1], ggml_row_size(cur->type, 2 * E));
        q = ggml_rope_multi(ctx, q, g.positions, nullptr, D / 2, sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
        k = ggml_rope_multi(ctx, k, g.positions, nullptr, D / 2, sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_cast(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), GGML_TYPE_F16);
        v = ggml_cast(ctx, ggml_permute(ctx, v, 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor* a = ggml_flash_attn_ext(ctx, q, k, v, nullptr, kq_scale, 0.0f, 0.0f);
        ggml_prec_set_acc(a, GGML_PREC_F32);
        a = ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2] * a->ne[3]);
        a = ggml_add(ctx, ggml_mul_mat(ctx, L.o_w, a), L.o_b);
        x = ggml_add(ctx, a, x);

        cur = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, hp.eps), L.ln2_w), L.ln2_b);
        cur = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, L.up_w, cur), L.up_b));
        cur = ggml_add(ctx, ggml_mul_mat(ctx, L.down_w, cur), L.down_b);
        x = ggml_add(ctx, x, cur);
        ggml_format_name(x, "layer_out-%d", il);
    }
    x = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, hp.eps), post_w_), post_b_);
    ggml_set_name(x, "post_ln");

    // merge 2x2 patches into one token and project into the language model's embedding space
    x = ggml_reshape_3d(ctx, x, E * 4, n_pos / 4, 1);
    x = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, mm0_w_, x), mm0_b_));
    x = ggml_add(ctx, ggml_mul_mat(ctx, mm2_w_, x), mm2_b_);
    x = ggml_reshape_2d(ctx, x, hp.proj_dim, n_pos / 4);
    ggml_set_name(x, "v_embd");
    ggml_set_output(x);
    ggml_build_forward_expand(g.gf, x);
    g.out = x;
    return g;
}

}  // namespace klein
