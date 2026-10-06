// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace klein {

struct SamplerParams {
    float temperature = 1.0f;   // <= 0: greedy
    int top_k = 20;             // <= 0: no limit
    float top_p = 0.95f;
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;          // 0: random
};

// Samples one token from logits. Remembers the tokens it is told about (accept()) for the penalties.
class Sampler {
public:
    explicit Sampler(const SamplerParams& p);
    void reset();
    void accept(int32_t token);
    int32_t sample(const float* logits, int n_vocab);
    const SamplerParams& params() const { return p_; }

private:
    SamplerParams p_;
    std::mt19937_64 rng_;
    std::unordered_map<int32_t, int> counts_;
    std::vector<std::pair<float, int32_t>> cand_;
};

}  // namespace klein
