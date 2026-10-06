// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// klein-cpubench: measures CPU matrix x small-batch throughput per weight type, on a weight set much larger
// than the CPU caches (like real decoding), with ggml's plain and repacked ("extra") CPU buffer layouts.
// Output: effective weight bytes per second for batch sizes 1..8. These numbers drive the placement planner.
#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

using namespace klein;

static std::vector<ggml_backend_buffer_type_t> extra_bufts(ggml_backend_dev_t dev) {
    std::vector<ggml_backend_buffer_type_t> out;
    auto reg = ggml_backend_dev_backend_reg(dev);
    auto fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (fn) {
        for (auto p = fn(dev); p && *p; ++p) out.push_back(*p);
    }
    return out;
}

struct Result {
    double gbps[9] = {0};
};

static Result bench_type(ggml_backend_t cpu, ggml_backend_buffer_type_t buft, ggml_type type, int64_t k, int64_t m, int copies,
                         const std::vector<int>& batches, int reps) {
    Result res;
    // weights
    ggml_init_params ip{ggml_tensor_overhead() * (copies + 64), nullptr, true};
    ggml_context* wctx = ggml_init(ip);
    std::vector<ggml_tensor*> W;
    for (int c = 0; c < copies; ++c) W.push_back(ggml_new_tensor_2d(wctx, type, k, m));
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
    if (!wbuf) fatal("alloc failed");

    // valid quantized data: quantize random floats once, reuse for every copy
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0.0f, 0.02f);
        std::vector<float> src((size_t) k * m);
        for (auto& v : src) v = nd(rng);
        std::vector<float> imat((size_t) k, 1.0f);
        std::vector<uint8_t> q(ggml_row_size(type, k) * m);
        ggml_quantize_chunk(type, src.data(), q.data(), 0, m, k, imat.data());
        for (auto t : W) ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
    const double wbytes = (double) ggml_row_size(type, k) * m * copies;

    for (int n : batches) {
        ggml_init_params gp{ggml_tensor_overhead() * (copies * 2 + 16) + ggml_graph_overhead(), nullptr, true};
        ggml_context* gctx = ggml_init(gp);
        ggml_tensor* x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, k, n);
        ggml_cgraph* gf = ggml_new_graph(gctx);
        for (auto t : W) ggml_build_forward_expand(gf, ggml_mul_mat(gctx, t, x));
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
        ggml_gallocr_alloc_graph(ga, gf);
        std::vector<float> xv((size_t) k * n, 0.01f);
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
        ggml_backend_graph_compute(cpu, gf);  // warm-up
        const double t0 = now_ms();
        for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(cpu, gf);
        const double dt = (now_ms() - t0) / reps;
        res.gbps[n] = wbytes / (dt * 1e6);
        ggml_gallocr_free(ga);
        ggml_free(gctx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return res;
}

int main(int argc, char** argv) {
    int threads = 8;
    std::string types = "q4_0,q8_0,q2_K,q3_K,q4_K,q5_K,q6_K,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_xs,iq4_nl";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-t" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--types" && i + 1 < argc) types = argv[++i];
        else { std::fprintf(stderr, "usage: %s [-t threads] [--types a,b,...]\n", argv[0]); return 1; }
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, threads);
    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    auto extras = extra_bufts(dev);

    const int64_t k = 5120, m = 17408;  // one Qwen3.8-27B FFN up/gate matrix
    const std::vector<int> batches = {1, 2, 3, 4, 8};
    std::printf("threads=%d, weight set ~= 8 FFN matrices (>> L3), GB/s of weight bytes\n", threads);
    std::printf("%-8s %-12s %8s %8s %8s %8s %8s\n", "type", "layout", "n=1", "n=2", "n=3", "n=4", "n=8");

    size_t pos = 0;
    while (pos < types.size()) {
        size_t e = types.find(',', pos);
        if (e == std::string::npos) e = types.size();
        const std::string name = types.substr(pos, e - pos);
        pos = e + 1;
        ggml_type type = GGML_TYPE_COUNT;
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
            const char* tn = ggml_type_name((ggml_type) t);
            if (tn && name == tn) type = (ggml_type) t;
        }
        if (type == GGML_TYPE_COUNT) { std::fprintf(stderr, "unknown type %s\n", name.c_str()); continue; }
        const int copies = 8;
        std::vector<std::pair<std::string, ggml_backend_buffer_type_t>> layouts = {{"plain", ggml_backend_cpu_buffer_type()}};
        for (auto b : extras) layouts.push_back({ggml_backend_buft_name(b), b});
        for (auto& [lname, buft] : layouts) {
            // the repack buffer only takes types it can repack; detect by trying a test op
            if (buft != ggml_backend_cpu_buffer_type()) {
                ggml_init_params ip{ggml_tensor_overhead() * 4, nullptr, true};
                ggml_context* c = ggml_init(ip);
                ggml_tensor* w = ggml_new_tensor_2d(c, type, k, m);
                ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, k, 4);
                ggml_tensor* y = ggml_mul_mat(c, w, x);
                ggml_backend_buffer_t tb = ggml_backend_buft_alloc_buffer(buft, 0x100);
                w->buffer = tb;
                const bool ok = ggml_backend_dev_supports_op(dev, y) && ggml_backend_dev_supports_buft(dev, buft);
                w->buffer = nullptr;
                ggml_backend_buffer_free(tb);
                ggml_free(c);
                if (!ok) continue;
            }
            Result r = bench_type(cpu, buft, type, k, m, copies, batches, 5);
            std::printf("%-8s %-12s %8.1f %8.1f %8.1f %8.1f %8.1f\n", name.c_str(), lname.c_str(), r.gbps[1], r.gbps[2], r.gbps[3],
                        r.gbps[4], r.gbps[8]);
            std::fflush(stdout);
        }
    }
    ggml_backend_free(cpu);
    return 0;
}
