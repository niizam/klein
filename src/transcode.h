// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>

namespace klein {

// IQ4_XS -> IQ4_NL. Both use the same 16-value non-linear codebook and the same nibble layout per 32 values; IQ4_XS
// stores one fp16 scale per 256 values plus a 6-bit sub-scale per 32, IQ4_NL one fp16 scale per 32. The IQ4_NL scale
// d * (ls - 32) is rounded to fp16 (relative error <= 2^-11), so the conversion is lossless up to that rounding.
// IQ4_NL has fast repacked AVX2 kernels for small batches; IQ4_XS has none.
void transcode_iq4xs_to_iq4nl(const void* src, void* dst, int64_t n_elements);

}  // namespace klein
