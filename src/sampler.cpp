// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace klein {

Sampler::Sampler(const SamplerParams& p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {}

void Sampler::reset() { counts_.clear(); }

void Sampler::accept(int32_t token) { counts_[token]++; }

int32_t Sampler::sample(const float* logits, int n_vocab) {
    const bool penalize = p_.presence_penalty != 0.0f || p_.frequency_penalty != 0.0f || p_.repetition_penalty != 1.0f;
    auto logit = [&](int32_t id) {
        float l = logits[id];
        if (penalize) {
            auto it = counts_.find(id);
            if (it != counts_.end()) {
                if (p_.repetition_penalty != 1.0f) l = l > 0 ? l / p_.repetition_penalty : l * p_.repetition_penalty;
                l -= p_.presence_penalty + p_.frequency_penalty * (float) it->second;
            }
        }
        return l;
    };

    if (p_.temperature <= 0.0f) {
        int32_t best = 0;
        float bl = -INFINITY;
        for (int32_t i = 0; i < n_vocab; ++i) {
            const float l = logit(i);
            if (l > bl) { bl = l; best = i; }
        }
        return best;
    }

    // top-k over (penalized) logits
    const int k = (p_.top_k > 0 && p_.top_k < n_vocab) ? p_.top_k : n_vocab;
    cand_.resize(n_vocab);
    for (int32_t i = 0; i < n_vocab; ++i) cand_[i] = {logit(i), i};
    auto cmp = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) { return a.first > b.first; };
    if (k < n_vocab) {
        std::partial_sort(cand_.begin(), cand_.begin() + k, cand_.end(), cmp);
        cand_.resize(k);
    } else {
        std::sort(cand_.begin(), cand_.end(), cmp);
    }

    // softmax with temperature
    const float maxl = cand_[0].first;
    double sum = 0.0;
    for (auto& c : cand_) {
        c.first = std::exp((c.first - maxl) / p_.temperature);
        sum += c.first;
    }
    for (auto& c : cand_) c.first = (float) (c.first / sum);

    // min-p
    if (p_.min_p > 0.0f) {
        const float thr = cand_[0].first * p_.min_p;
        size_t n = 1;
        while (n < cand_.size() && cand_[n].first >= thr) ++n;
        cand_.resize(n);
    }
    // top-p
    if (p_.top_p < 1.0f) {
        double cum = 0.0;
        size_t n = 0;
        while (n < cand_.size()) {
            cum += cand_[n].first;
            ++n;
            if (cum >= p_.top_p) break;
        }
        cand_.resize(std::max<size_t>(n, 1));
    }
    double total = 0.0;
    for (auto& c : cand_) total += c.first;
    std::uniform_real_distribution<double> u(0.0, total);
    double r = u(rng_);
    for (auto& c : cand_) {
        r -= c.first;
        if (r <= 0.0) return c.second;
    }
    return cand_.back().second;
}

}  // namespace klein
