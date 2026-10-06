// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "transcode.h"

#include <cstring>

#include "common.h"
#include "ggml.h"

namespace klein {

namespace {
// Layouts as in ggml-common.h (block_iq4_xs, block_iq4_nl).
constexpr int QK_K = 256;
struct BlockIq4Xs {
    ggml_fp16_t d;
    uint16_t scales_h;
    uint8_t scales_l[QK_K / 64];
    uint8_t qs[QK_K / 2];
};
struct BlockIq4Nl {
    ggml_fp16_t d;
    uint8_t qs[16];
};
static_assert(sizeof(BlockIq4Xs) == 2 + 2 + 4 + 128, "iq4_xs block size");
static_assert(sizeof(BlockIq4Nl) == 18, "iq4_nl block size");
}  // namespace

void transcode_iq4xs_to_iq4nl(const void* src, void* dst, int64_t n_elements) {
    KLEIN_ASSERT(n_elements % QK_K == 0);
    const BlockIq4Xs* x = (const BlockIq4Xs*) src;
    BlockIq4Nl* y = (BlockIq4Nl*) dst;
    const int64_t nb = n_elements / QK_K;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4);
            BlockIq4Nl& o = y[i * (QK_K / 32) + ib];
            o.d = ggml_fp32_to_fp16(d * (float) (ls - 32));
            std::memcpy(o.qs, x[i].qs + 16 * ib, 16);
        }
    }
}

}  // namespace klein
