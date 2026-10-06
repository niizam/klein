// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The qwen35 architecture as ggml graphs. The structure follows llama.cpp's `qwen35` implementation
// (MIT, src/models/qwen35.cpp and delta-net-base.cpp), specialised to a single sequence.
#include "graph.h"

#include <algorithm>
#include <cmath>

#include "common.h"

namespace klein {

int padded_n_kv(int n_past_after, int n_ctx) {
    const int pad = 256;
    return std::min(n_ctx, (n_past_after + pad - 1) / pad * pad);
}

namespace {

// ggml-cuda's default op-offload threshold: from this batch size on, ops on host weights run on the GPU.
constexpr int64_t kGpuOffloadBatch = 32;

struct Builder {
    ggml_context* ctx;
    const Model& m;
    const State& st;
    const HParams& hp;
    ggml_cgraph* gf;
    FwdInputs in;
    int n_tokens, n_kv;
    std::vector<ggml_tensor*> attn_nodes;

    ggml_tensor* rms(ggml_tensor* x, ggml_tensor* w) {
        x = ggml_rms_norm(ctx, x, hp.rms_eps);
        return w ? ggml_mul(ctx, x, w) : x;
    }

    ggml_tensor* l2(ggml_tensor* x) {
        const float n = (float) x->ne[0];
        return ggml_scale(ctx, ggml_rms_norm(ctx, x, hp.rms_eps / n), 1.0f / std::sqrt(n));
    }

    void make_inputs(bool need_mask) {
        in.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(in.tokens, "inp_tokens");
        ggml_set_input(in.tokens);
        in.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n_tokens * 4);
        ggml_set_name(in.pos, "inp_pos");
        ggml_set_input(in.pos);
        in.kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
        ggml_set_name(in.kv_idx, "inp_kv_idx");
        ggml_set_input(in.kv_idx);
        if (need_mask) {
            in.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens);
            ggml_set_name(in.mask, "inp_mask");
            ggml_set_input(in.mask);
        }
    }

    // Spilled weights: small batches use the repacked CPU copy; large batches use the original, which the scheduler
    // streams to the GPU (op offload).
    ggml_tensor* mm(ggml_tensor* w, ggml_tensor* x) {
        if (x->ne[1] < kGpuOffloadBatch && w->buffer && ggml_backend_buffer_is_host(w->buffer)) {
            auto it = m.cpu_alt.find(w);
            if (it != m.cpu_alt.end()) w = it->second;
        }
        return ggml_mul_mat(ctx, w, x);
    }

    ggml_tensor* ffn(const Layer& L, ggml_tensor* x) {
        ggml_tensor* g = mm(L.ffn_gate, x);
        ggml_tensor* u = mm(L.ffn_up, x);
        ggml_tensor* h = ggml_swiglu_split(ctx, g, u);
        return mm(L.ffn_down, h);
    }

    // Full attention with output gate, Q/K norm, interleaved M-RoPE and a quantized KV cache.
    ggml_tensor* attention(const Layer& L, int il, ggml_tensor* x) {
        const int hd = hp.head_dim, nh = hp.n_head, nkv = hp.n_head_kv;
        ggml_tensor* qfull = ggml_mul_mat(ctx, L.wq, x);  // [2 * hd * nh, n]
        ggml_tensor* kcur = ggml_mul_mat(ctx, L.wk, x);
        ggml_tensor* vcur = ggml_mul_mat(ctx, L.wv, x);

        const size_t es = ggml_element_size(qfull);
        ggml_tensor* q = ggml_view_3d(ctx, qfull, hd, nh, n_tokens, es * hd * 2, es * hd * 2 * nh, 0);
        q = rms(q, L.q_norm);
        ggml_tensor* gate = ggml_view_3d(ctx, qfull, hd, nh, n_tokens, es * hd * 2, es * hd * 2 * nh, es * hd);
        gate = ggml_cont_2d(ctx, gate, (int64_t) hd * nh, n_tokens);

        kcur = rms(ggml_reshape_3d(ctx, kcur, hd, nkv, n_tokens), L.k_norm);
        vcur = ggml_reshape_3d(ctx, vcur, hd, nkv, n_tokens);

        int sections[4] = {hp.rope_sections[0], hp.rope_sections[1], hp.rope_sections[2], hp.rope_sections[3]};
        q = ggml_rope_multi(ctx, q, in.pos, nullptr, hp.n_rot, sections, GGML_ROPE_TYPE_IMROPE, hp.n_ctx_train,
                            hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        kcur = ggml_rope_multi(ctx, kcur, in.pos, nullptr, hp.n_rot, sections, GGML_ROPE_TYPE_IMROPE, hp.n_ctx_train,
                               hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        ggml_tensor* ck = st.k(il);
        ggml_tensor* cv = st.v(il);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, ck, ggml_reshape_2d(ctx, kcur, hp.kv_row(), n_tokens), in.kv_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cv, ggml_reshape_2d(ctx, vcur, hp.kv_row(), n_tokens), in.kv_idx));

        ggml_tensor* k = ggml_view_3d(ctx, ck, hd, nkv, n_kv, ggml_row_size(ck->type, hd), ggml_row_size(ck->type, hp.kv_row()), 0);
        ggml_tensor* v = ggml_view_3d(ctx, cv, hd, nkv, n_kv, ggml_row_size(cv->type, hd), ggml_row_size(cv->type, hp.kv_row()), 0);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);  // [hd, n_kv, nkv]
        v = ggml_permute(ctx, v, 0, 2, 1, 3);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [hd, n, nh]

        ggml_tensor* o = ggml_flash_attn_ext(ctx, q, k, v, in.mask, 1.0f / std::sqrt((float) hd), 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        attn_nodes.push_back(o);
        o = ggml_reshape_2d(ctx, o, (int64_t) hd * nh, n_tokens);
        o = ggml_mul(ctx, o, ggml_sigmoid(ctx, gate));
        return ggml_mul_mat(ctx, L.wo, o);
    }

    // Gated DeltaNet (linear attention) with causal conv1d; writes rollback snapshots into the state.
    ggml_tensor* deltanet(const Layer& L, int il, ggml_tensor* x) {
        const int64_t S = hp.ssm_d_state, Hk = hp.ssm_n_group, Hv = hp.ssm_dt_rank;
        const int64_t ch = hp.conv_channels(), kc = hp.ssm_conv_kernel;
        const int K = st.cfg().n_snapshots;

        ggml_tensor* qkv = ggml_mul_mat(ctx, L.wqkv, x);  // [ch, n]
        ggml_tensor* z = ggml_mul_mat(ctx, L.wz, x);      // [Hv * S, n]
        ggml_tensor* beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, ggml_mul_mat(ctx, L.ssm_beta, x), 1, Hv, n_tokens, 1));
        ggml_tensor* alpha = ggml_mul_mat(ctx, L.ssm_alpha, x);  // [Hv, n]
        ggml_tensor* g = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, L.ssm_dt)), L.ssm_a);
        g = ggml_reshape_4d(ctx, g, 1, Hv, n_tokens, 1);

        // causal conv1d over [previous kc-1 inputs | this batch]
        ggml_tensor* cs_all = st.conv(il);
        ggml_tensor* cs = ggml_view_1d(ctx, cs_all, cs_all->ne[0], 0);
        cs = ggml_reshape_3d(ctx, cs, kc - 1, ch, 1);
        ggml_tensor* xt = ggml_transpose(ctx, ggml_reshape_3d(ctx, qkv, ch, n_tokens, 1));
        ggml_tensor* conv_in = ggml_concat(ctx, cs, xt, 0);  // [kc - 1 + n, ch, 1]
        for (int t = 1; t <= K; ++t) {
            const int64_t s_idx = std::max<int64_t>(0, conv_in->ne[0] - (kc - 1) - K + t);
            const int slot = K - t;
            if (slot >= n_tokens && slot > 0) continue;  // only states inside this batch can be rolled back to
            ggml_tensor* src = ggml_view_3d(ctx, conv_in, kc - 1, ch, 1, conv_in->nb[1], conv_in->nb[2],
                                            ggml_row_size(conv_in->type, s_idx));
            ggml_tensor* dst = ggml_view_1d(ctx, cs_all, cs_all->ne[0], (size_t) slot * cs_all->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
        }
        ggml_tensor* conv = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, L.ssm_conv1d));  // [ch, n, 1]

        const int64_t nb1 = ggml_row_size(conv->type, ch);
        ggml_tensor* q = ggml_view_4d(ctx, conv, S, Hk, n_tokens, 1, ggml_row_size(conv->type, S), nb1, nb1 * n_tokens, 0);
        ggml_tensor* k = ggml_view_4d(ctx, conv, S, Hk, n_tokens, 1, ggml_row_size(conv->type, S), nb1, nb1 * n_tokens,
                                      ggml_row_size(conv->type, S * Hk));
        ggml_tensor* v = ggml_view_4d(ctx, conv, S, Hv, n_tokens, 1, ggml_row_size(conv->type, S), nb1, nb1 * n_tokens,
                                      ggml_row_size(conv->type, 2 * S * Hk));
        q = l2(q);
        k = l2(k);

        ggml_tensor* ss_all = st.ssm(il);
        ggml_tensor* s0 = ggml_reshape_4d(ctx, ggml_view_1d(ctx, ss_all, ss_all->ne[0], 0), S, S, Hv, 1);
        ggml_tensor* r = ggml_gated_delta_net(ctx, q, k, v, g, beta, s0, K);

        const int64_t attn_elems = S * Hv * n_tokens;
        ggml_tensor* out = ggml_view_4d(ctx, r, S, Hv, n_tokens, 1, ggml_row_size(r->type, S), ggml_row_size(r->type, S * Hv),
                                        ggml_row_size(r->type, attn_elems), 0);
        const int n_written = std::min<int>(n_tokens, K);
        const int64_t D = S * S * Hv;
        ggml_tensor* snaps = ggml_view_2d(ctx, r, D, n_written, ggml_row_size(r->type, D), ggml_row_size(r->type, attn_elems));
        ggml_tensor* dst = ggml_view_2d(ctx, ss_all, D, n_written, ss_all->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, snaps, dst));

        // gated RMS norm, then output projection
        ggml_tensor* zz = ggml_reshape_4d(ctx, z, S, Hv, n_tokens, 1);
        ggml_tensor* o = ggml_mul(ctx, rms(out, L.ssm_norm), ggml_silu(ctx, zz));
        o = ggml_reshape_2d(ctx, o, S * Hv, n_tokens);
        return ggml_mul_mat(ctx, L.ssm_out, o);
    }

    ggml_tensor* block(const Layer& L, int il, ggml_tensor* x, bool recurrent) {
        ggml_tensor* cur = rms(x, L.attn_norm);
        cur = recurrent ? deltanet(L, il, cur) : attention(L, il, cur);
        cur = ggml_add(ctx, cur, x);
        ggml_tensor* res = cur;
        cur = ffn(L, rms(cur, L.post_norm));
        return ggml_add(ctx, cur, res);
    }

    ggml_tensor* lm_head(ggml_tensor* h, int n_out) {
        if (n_out != n_tokens) {
            in.out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_out);
            ggml_set_name(in.out_ids, "inp_out_ids");
            ggml_set_input(in.out_ids);
            h = ggml_get_rows(ctx, h, in.out_ids);
        }
        return ggml_mul_mat(ctx, m.output, h);
    }
};

}  // namespace

FwdGraph build_main_graph(ggml_context* ctx, const Model& m, const State& st, int n_tokens, int n_kv, int n_out) {
    Builder b{ctx, m, st, m.hp, ggml_new_graph_custom(ctx, 16384, false), {}, n_tokens, n_kv};
    b.make_inputs(true);
    ggml_tensor* x = ggml_get_rows(ctx, m.tok_embd, b.in.tokens);
    for (int il = 0; il < m.hp.n_layer; ++il) x = b.block(m.layers[il], il, x, m.hp.is_recurrent(il));
    ggml_tensor* h = b.rms(x, m.output_norm);
    ggml_tensor* hdst = ggml_view_2d(ctx, st.hidden(), m.hp.n_embd, n_tokens, st.hidden()->nb[1], 0);
    ggml_build_forward_expand(b.gf, ggml_cpy(ctx, h, hdst));
    FwdGraph g;
    g.logits = n_out > 0 ? b.lm_head(h, n_out) : nullptr;
    if (g.logits) {
        ggml_set_name(g.logits, "logits");
        ggml_set_output(g.logits);
        ggml_build_forward_expand(b.gf, g.logits);
    }
    g.gf = b.gf;
    g.in = b.in;
    g.attn_nodes = b.attn_nodes;
    g.n_tokens = n_tokens;
    g.n_kv = n_kv;
    g.n_out = n_out;
    return g;
}

FwdGraph build_mtp_graph(ggml_context* ctx, const Model& m, const State& st, int n_tokens, int n_kv, int n_out,
                         ggml_tensor* h_src, ggml_tensor* h_dst) {
    KLEIN_ASSERT(m.has_mtp());
    const Layer& L = m.mtp_layer();
    const int il = m.hp.n_layer;
    Builder b{ctx, m, st, m.hp, ggml_new_graph_custom(ctx, 2048, false), {}, n_tokens, n_kv};
    b.make_inputs(true);
    ggml_tensor* e = b.rms(ggml_get_rows(ctx, m.tok_embd, b.in.tokens), L.enorm);
    ggml_tensor* h = b.rms(h_src, L.hnorm);
    ggml_tensor* x = ggml_mul_mat(ctx, L.eh_proj, ggml_concat(ctx, e, h, 0));
    x = b.block(L, il, x, false);
    ggml_tensor* hout = b.rms(x, L.head_norm ? L.head_norm : m.output_norm);
    if (h_dst) ggml_build_forward_expand(b.gf, ggml_cpy(ctx, hout, h_dst));
    FwdGraph g;
    g.logits = n_out > 0 ? b.lm_head(hout, n_out) : nullptr;
    if (g.logits) {
        ggml_set_name(g.logits, "mtp_logits");
        ggml_set_output(g.logits);
        ggml_build_forward_expand(b.gf, g.logits);
    }
    g.gf = b.gf;
    g.in = b.in;
    g.attn_nodes = b.attn_nodes;
    g.n_tokens = n_tokens;
    g.n_kv = n_kv;
    g.n_out = n_out;
    return g;
}

}  // namespace klein
