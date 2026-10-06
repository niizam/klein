// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "planner.h"

#include <algorithm>
#include <map>

#include "common.h"

namespace klein {

double cpu_weight_gbps(ggml_type type, double dram_gbps, int batch) {
    // GB/s of the GGUF's weight bytes, measured with klein-cpubench (weights >> L3, 8 threads, Ryzen 7 5700X,
    // DDR4-3200) in the layout klein uses on the CPU: repacked for Q4_0/Q4_K/IQ4_NL, IQ4_XS transcoded to IQ4_NL
    // and repacked (rate per IQ4_XS byte), plain otherwise. `batch` 1 = plain decode, 4 = MTP verify of 3 drafts.
    // See docs/BENCHMARKS.md.
    struct Rate {
        ggml_type t;
        double n1, n4;
    };
    static const Rate rates[] = {
        {GGML_TYPE_Q4_0, 37.0, 29.7},   {GGML_TYPE_Q8_0, 36.7, 36.0},   {GGML_TYPE_Q2_K, 36.5, 25.8},
        {GGML_TYPE_Q3_K, 35.1, 16.2},   {GGML_TYPE_Q4_K, 37.9, 34.5},   {GGML_TYPE_Q5_K, 37.4, 20.7},
        {GGML_TYPE_Q6_K, 33.2, 27.6},   {GGML_TYPE_IQ2_XS, 28.1, 7.1},  {GGML_TYPE_IQ2_S, 25.6, 6.2},
        {GGML_TYPE_IQ3_XXS, 21.4, 6.1}, {GGML_TYPE_IQ3_S, 23.1, 6.1},   {GGML_TYPE_IQ4_XS, 35.6, 29.8},
        {GGML_TYPE_IQ4_NL, 37.7, 31.6},
    };
    double g = 10.0;
    for (const Rate& r : rates) {
        if (r.t == type) {
            // linear interpolation between the measured batch sizes
            const double f = std::clamp((batch - 1) / 3.0, 0.0, 1.0);
            g = r.n1 + f * (r.n4 - r.n1);
        }
    }
    return std::min(g, dram_gbps);
}

std::string Plan::summary() const {
    std::string s = format("VRAM for weights %.2f GiB: %.2f GiB of weights on GPU, %.2f GiB in RAM (%d FFN blocks", budget / GiB,
                           weights_gpu / GiB, weights_host / GiB, n_spilled_blocks);
    if (!spilled_layers.empty()) {
        s += ": layers";
        for (int l : spilled_layers) s += format(" %d", l);
    }
    s += format("); KV cache %.2f GiB (%s, %s/%s) in %s; recurrent state %.0f MiB; est. CPU time %.1f ms/step",
                kv_bytes / GiB, format("%d ctx", state.n_ctx).c_str(), ggml_type_name(state.type_k), ggml_type_name(state.type_v),
                state.kv_place == Place::Gpu ? "VRAM" : "RAM", rec_bytes / MiB, est_cpu_ms);
    return s;
}

Plan plan_placement(Model& model, const PlanInput& in) {
    const HParams& hp = model.hp;
    Plan p;
    p.state = in.state;
    p.rec_bytes = state_recurrent_bytes(hp, p.state);
    p.kv_bytes = state_kv_bytes(hp, p.state, model.has_mtp());
    const size_t hidden_bytes = 2ull * hp.n_embd * p.state.max_batch * sizeof(float);

    size_t all_weights = 0;
    for (auto& w : model.weights) {
        w.place = (w.t == model.tok_embd) ? Place::Host : Place::Gpu;
        if (w.place == Place::Gpu) all_weights += ggml_nbytes(w.t);
    }

    const size_t fixed = in.vram_margin + in.compute_reserve + p.rec_bytes + hidden_bytes;
    if (in.vram_free < fixed) fatal("not enough free VRAM: %.2f GiB free, %.2f GiB needed before any weights", in.vram_free / GiB, fixed / GiB);
    const size_t avail = in.vram_free - fixed;

    // A large KV cache goes to RAM: every KV byte in VRAM would push a weight byte out, and weight bytes are read on
    // every step while KV bytes are read only up to the current context. A small one stays in VRAM, where attention
    // needs no CPU work and no PCIe copies.
    if (in.kv_auto) {
        const bool small = p.kv_bytes <= in.kv_gpu_max;
        p.state.kv_place = (small || all_weights + p.kv_bytes <= avail) ? Place::Gpu : Place::Host;
    }
    p.budget = avail - (p.state.kv_place == Place::Gpu ? std::min(avail, p.kv_bytes) : 0);

    // Spill units: one decoder layer's FFN (gate + up + down) each, so every spilled layer costs one GPU->CPU->GPU
    // round trip. Cheapest CPU time per freed byte first.
    struct Unit {
        int layer;
        size_t bytes;
        double ms;
        std::vector<WeightInfo*> ws;
    };
    std::map<int, Unit> units;
    for (auto& w : model.weights) {
        if (w.layer < 0 || w.layer >= hp.n_layer) continue;
        const Layer& L = model.layers[w.layer];
        if (w.t != L.ffn_gate && w.t != L.ffn_up && w.t != L.ffn_down) continue;
        Unit& u = units[w.layer];
        u.layer = w.layer;
        u.bytes += ggml_nbytes(w.t);
        u.ms += ggml_nbytes(w.t) / (cpu_weight_gbps(w.t->type, in.dram_gbps, in.verify_batch) * 1e6);
        u.ws.push_back(&w);
    }
    std::vector<Unit*> order;
    for (auto& [l, u] : units) order.push_back(&u);
    std::sort(order.begin(), order.end(), [](const Unit* a, const Unit* b) { return a->ms / a->bytes < b->ms / b->bytes; });

    size_t gpu = all_weights;
    for (Unit* u : order) {
        if (gpu <= p.budget) break;
        for (WeightInfo* w : u->ws) w->place = Place::Host;
        gpu -= u->bytes;
        p.est_cpu_ms += u->ms;
        p.spilled_layers.push_back(u->layer);
    }
    if (gpu > p.budget) fatal("the model does not fit even with every FFN in RAM (%.2f GiB needed on GPU, %.2f GiB available)", gpu / GiB, p.budget / GiB);
    std::sort(p.spilled_layers.begin(), p.spilled_layers.end());
    p.n_spilled_blocks = (int) p.spilled_layers.size();
    p.weights_gpu = model.bytes(Place::Gpu);
    p.weights_host = model.bytes(Place::Host);
    return p;
}

}  // namespace klein
