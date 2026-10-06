// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <string>
#include <vector>

#include "model.h"
#include "state.h"

namespace klein {

struct PlanInput {
    size_t vram_free = 0;          // measured after the CUDA context exists
    size_t vram_margin = 384ull << 20;  // left free for the driver, the CUDA pool and fragmentation
    size_t compute_reserve = 0;    // compute buffers (graph activations)
    StateConfig state;             // n_ctx, KV types, snapshots; kv_place is decided by the planner when kv_auto
    bool kv_auto = true;
    size_t kv_gpu_max = 768ull << 20;  // auto: KV caches up to this size stay in VRAM
    double dram_gbps = 38.0;       // upper bound of CPU weight streaming (see docs/BENCHMARKS.md)
    int verify_batch = 1;          // tokens per decode step on the spilled weights (1 + MTP drafts)
};

struct Plan {
    StateConfig state;
    size_t budget = 0;             // VRAM available for weights
    size_t weights_gpu = 0, weights_host = 0;
    size_t kv_bytes = 0, rec_bytes = 0;
    int n_spilled_blocks = 0;
    std::vector<int> spilled_layers;
    std::vector<int> block_order;  // every decoder layer's FFN block, cheapest to run on the CPU first
    double est_cpu_ms = 0.0;       // estimated CPU time per single-token step for the spilled weights
    std::string summary() const;
};

// CPU throughput (GB/s of weight bytes, single token) for a quant type on the reference CPU, capped at DRAM speed.
double cpu_weight_gbps(ggml_type type, double dram_gbps, int batch);

// Decides where every weight goes (sets WeightInfo::place) and where the KV cache goes.
Plan plan_placement(Model& model, const PlanInput& in);

}  // namespace klein
