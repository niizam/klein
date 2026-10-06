// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace klein {

// 8-bit RGB image, row-major, 3 bytes per pixel.
struct ImageRGB {
    int w = 0, h = 0;
    std::vector<uint8_t> px;
};

// Decodes PNG, JPEG, BMP, GIF (first frame), TGA, PSD, HDR or PNM from memory. Throws std::runtime_error.
ImageRGB decode_image(const uint8_t* data, size_t size);

// Qwen-VL input: the image resized (aspect ratio kept, sides a multiple of `align` = patch * merge) and normalized to
// [-1, 1], stored planar: [3][h][w] floats.
struct ImageInput {
    int w = 0, h = 0;
    std::vector<float> planar;
};

// `min_tokens` / `max_tokens` bound the number of merged image tokens (each covers align x align pixels).
ImageInput preprocess_qwen_vl(const ImageRGB& img, int align, int min_tokens, int max_tokens, const float mean[3], const float std[3]);

}  // namespace klein
