// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Image decoding and Qwen-VL preprocessing. The resize ("smart resize" with aspect-preserving padding) and the
// Pillow-compatible bicubic resampler follow llama.cpp's tools/mtmd/mtmd-image.cpp (MIT), so klein feeds the vision
// encoder the same pixels as the reference implementation.
#include "image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace klein {

ImageRGB decode_image(const uint8_t* data, size_t size) {
    int w = 0, h = 0, ch = 0;
    unsigned char* p = stbi_load_from_memory(data, (int) size, &w, &h, &ch, 3);
    if (!p) throw std::runtime_error(std::string("cannot decode image: ") + stbi_failure_reason());
    ImageRGB img;
    img.w = w;
    img.h = h;
    img.px.assign(p, p + (size_t) w * h * 3);
    stbi_image_free(p);
    return img;
}

namespace {

// Pillow-compatible separable bicubic resampling (a = -0.5), fixed point with 22 fractional bits.
ImageRGB resize_bicubic(const ImageRGB& img, int tw, int th) {
    const int PRECISION_BITS = 32 - 8 - 2;
    const double filter_support = 2.0;
    auto filter = [](double x) -> double {
        x = std::fabs(x);
        constexpr double a = -0.5;
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    auto clip8 = [](int v) -> uint8_t { return (uint8_t) std::min(255, std::max(0, v)); };
    auto weights_for = [&](int in, int out, std::vector<int>& bounds, std::vector<int32_t>& weights) {
        double scale = (double) in / out, fscale = std::max(1.0, scale);
        const double support = filter_support * fscale;
        const int ksize = (int) std::ceil(support) * 2 + 1;
        std::vector<double> pre((size_t) out * ksize);
        bounds.resize((size_t) out * 2);
        for (int xx = 0; xx < out; ++xx) {
            const double center = (xx + 0.5) * scale;
            double ww = 0.0;
            const double ss = 1.0 / fscale;
            int xmin = std::max(0, (int) (center - support + 0.5));
            int xmax = std::min(in, (int) (center + support + 0.5)) - xmin;
            int x = 0;
            for (; x < xmax; ++x) {
                const double w = filter((x + xmin - center + 0.5) * ss);
                pre[(size_t) xx * ksize + x] = w;
                ww += w;
            }
            for (x = 0; x < xmax; ++x)
                if (ww != 0.0) pre[(size_t) xx * ksize + x] /= ww;
            for (; x < ksize; ++x) pre[(size_t) xx * ksize + x] = 0;
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        weights.resize(pre.size());
        const double fxp = std::ldexp(1.0, PRECISION_BITS);
        for (size_t i = 0; i < pre.size(); ++i) weights[i] = (int32_t) (pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5));
        return ksize;
    };

    std::vector<uint8_t> cur = img.px;
    int cw = img.w, chh = img.h;
    if (tw != cw) {
        std::vector<int> b;
        std::vector<int32_t> k;
        const int ks = weights_for(cw, tw, b, k);
        std::vector<uint8_t> out((size_t) tw * chh * 3);
        for (int y = 0; y < chh; ++y) {
            const uint8_t* row = cur.data() + (size_t) y * cw * 3;
            uint8_t* dst = out.data() + (size_t) y * tw * 3;
            for (int xx = 0; xx < tw; ++xx) {
                const int xmin = b[xx * 2], xcnt = b[xx * 2 + 1];
                const int32_t* kk = &k[(size_t) xx * ks];
                const uint8_t* p = row + (size_t) xmin * 3;
                int32_t s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xcnt; ++x, p += 3) {
                    s0 += p[0] * kk[x];
                    s1 += p[1] * kk[x];
                    s2 += p[2] * kk[x];
                }
                dst[xx * 3] = clip8(s0 >> PRECISION_BITS);
                dst[xx * 3 + 1] = clip8(s1 >> PRECISION_BITS);
                dst[xx * 3 + 2] = clip8(s2 >> PRECISION_BITS);
            }
        }
        cur.swap(out);
        cw = tw;
    }
    if (th != chh) {
        std::vector<int> b;
        std::vector<int32_t> k;
        const int ks = weights_for(chh, th, b, k);
        const size_t row = (size_t) cw * 3;
        std::vector<uint8_t> out(row * th);
        std::vector<int32_t> acc(row);
        for (int yy = 0; yy < th; ++yy) {
            const int ymin = b[yy * 2], ycnt = b[yy * 2 + 1];
            const int32_t* kk = &k[(size_t) yy * ks];
            std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));
            for (int y = 0; y < ycnt; ++y) {
                const uint8_t* src = cur.data() + (size_t) (ymin + y) * row;
                for (size_t i = 0; i < row; ++i) acc[i] += src[i] * kk[y];
            }
            uint8_t* dst = out.data() + (size_t) yy * row;
            for (size_t i = 0; i < row; ++i) dst[i] = clip8(acc[i] >> PRECISION_BITS);
        }
        cur.swap(out);
        chh = th;
    }
    ImageRGB r;
    r.w = cw;
    r.h = chh;
    r.px.swap(cur);
    return r;
}

}  // namespace

ImageInput preprocess_qwen_vl(const ImageRGB& img, int align, int min_tokens, int max_tokens, const float mean[3], const float std[3]) {
    if (img.w <= 0 || img.h <= 0) throw std::runtime_error("empty image");
    const long long min_px = (long long) min_tokens * align * align, max_px = (long long) max_tokens * align * align;
    auto round_f = [&](double x) { return (int) std::round(x / align) * align; };
    auto ceil_f = [&](double x) { return (int) std::ceil(x / align) * align; };
    auto floor_f = [&](double x) { return (int) std::floor(x / align) * align; };
    int w = std::max(align, round_f(img.w)), h = std::max(align, round_f(img.h));
    if ((long long) w * h > max_px) {
        const double beta = std::sqrt((double) img.h * img.w / (double) max_px);
        h = std::max(align, floor_f(img.h / beta));
        w = std::max(align, floor_f(img.w / beta));
    } else if ((long long) w * h < min_px) {
        const double beta = std::sqrt((double) min_px / ((double) img.h * img.w));
        h = ceil_f(img.h * beta);
        w = ceil_f(img.w * beta);
    }

    // aspect-preserving resize into (w, h), centered on black (llama.cpp PAD_CEIL)
    ImageRGB canvas;
    canvas.w = w;
    canvas.h = h;
    if (img.w == w && img.h == h) {
        canvas.px = img.px;
    } else {
        const double scale = std::min((double) w / img.w, (double) h / img.h);
        const int nw = std::min((int) std::ceil(img.w * scale), w), nh = std::min((int) std::ceil(img.h * scale), h);
        ImageRGB r = resize_bicubic(img, nw, nh);
        canvas.px.assign((size_t) w * h * 3, 0);
        const int ox = (w - nw) / 2, oy = (h - nh) / 2;
        for (int y = 0; y < nh; ++y)
            std::copy(r.px.begin() + (size_t) y * nw * 3, r.px.begin() + (size_t) (y + 1) * nw * 3,
                      canvas.px.begin() + ((size_t) (y + oy) * w + ox) * 3);
    }

    ImageInput in;
    in.w = w;
    in.h = h;
    const size_t n = (size_t) w * h;
    in.planar.resize(3 * n);
    for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) in.planar[c * n + i] = ((float) canvas.px[i * 3 + c] / 255.0f - mean[c]) / std[c];
    return in;
}

}  // namespace klein
