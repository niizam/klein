// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "model.h"
#include "planner.h"
#include "sampler.h"
#include "state.h"
#include "tokenizer.h"

namespace klein {

struct EngineConfig {
    std::string model_path;
    std::string mtp_path;          // optional separate MTP GGUF (smaller MTP block)
    int n_ctx = 262144;
    std::string kv_type = "auto";  // auto | f16 | q8_0 | q4_0
    std::string kv_place = "auto"; // auto | gpu | host
    int n_threads = 0;             // 0 = 3/4 of the hardware threads (12 on an 8-core/16-thread CPU, measured best)
    int n_ubatch = 2048;           // prefill chunk
    int n_draft = 3;               // MTP draft tokens per step (0 = no speculation)
    size_t vram_margin_mb = 256;
    size_t compute_reserve_mb = 0; // 0 = estimate
    bool cpu_repack = true;        // repacked CPU copies of spilled weights (faster small-batch decode, more RAM)
    bool kv_zero_copy = true;      // host KV cache read by the GPU over PCIe (else: attention on the CPU)
    int mtp_window = 16384;        // MTP head attention window (positions)
};

struct GenStats {
    int n_prompt = 0;
    int n_gen = 0;
    double t_prompt_ms = 0.0;
    double t_gen_ms = 0.0;
    int n_steps = 0;          // verify passes
    int n_drafted = 0;
    int n_accepted = 0;
    // time per phase of the decode loop
    double t_draft_ms = 0, t_verify_ms = 0, t_sample_ms = 0, t_rollback_ms = 0, t_mtp_ms = 0;
    double prompt_tps() const { return t_prompt_ms > 0 ? n_prompt * 1000.0 / t_prompt_ms : 0.0; }
    double gen_tps() const { return t_gen_ms > 0 ? n_gen * 1000.0 / t_gen_ms : 0.0; }
};

class Engine {
public:
    explicit Engine(const EngineConfig& cfg);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    const Model& model() const { return *model_; }
    const Tokenizer& tokenizer() const { return tok_; }
    State& state() { return *state_; }
    const Plan& plan() const { return plan_; }
    int n_vocab() const { return model_->hp.n_vocab; }
    int n_ctx() const { return state_->cfg().n_ctx; }
    size_t vram_free() const;

    // Resets the sequence.
    void reset();

    // Main model over `n` tokens at positions state.n_past..; appends them to the cache. Logits of the last `n_out`
    // tokens are written to `logits` (n_out x n_vocab) when it is non-null.
    void eval(const int32_t* tokens, int n, int n_out, std::vector<float>* logits);

    // Prefill a prompt (chunked), feeding the MTP head along the way. Returns the logits of the last token.
    void prefill(const std::vector<int32_t>& tokens, std::vector<float>& last_logits, GenStats* stats);

    // Generates up to n_predict tokens after a prefill. `on_token` returns false to stop. When the cached tokens are
    // a prefix of `prompt` (e.g. the previous turn of the same chat), only the new suffix is prefilled.
    GenStats generate(const std::vector<int32_t>& prompt, int n_predict, Sampler& sampler,
                      const std::function<bool(int32_t)>& on_token);

private:
    EngineConfig cfg_;
    ggml_backend_t gpu_ = nullptr;
    ggml_backend_t cpu_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    std::unique_ptr<Model> model_;
    std::unique_ptr<State> state_;
    Tokenizer tok_;
    Plan plan_;
    std::vector<uint8_t> meta_buf_;
    bool mtp_ready_ = false;            // MTP logits for the next draft are in mtp_logits_
    bool kv_mapped_ = false;            // host KV cache is mapped into the GPU address space (zero-copy)
    std::vector<float> mtp_logits_;
    std::vector<int32_t> cache_;        // tokens at positions 0 .. n_past-1 (for prompt-prefix reuse)
    int mtp_h_row_ = 0;                 // row of State::mtp_hidden holding the last MTP hidden state

    struct RunArgs {
        const int32_t* tokens;
        int n;
        int pos0;
        int kv_lo;                      // first valid KV cell for the mask
        int n_out;
        std::vector<float>* logits;
    };
    void run(bool mtp, const RunArgs& a, ggml_tensor* h_src, ggml_tensor* h_dst);
    size_t compute_need(int n_kv, int n_tokens) const;
    void ensure_prefill_vram(int n_kv_end, int n_tokens);
    void relax_after_prefill();
    void warmup();
    int n_ctx_cfg_ = 0;
    size_t margin_ = 0;
    // MTP pass over tokens at positions pos0.., reading main hidden rows [h_row0, h_row0 + n).
    void mtp_pass(const int32_t* tokens, int n, int pos0, bool from_main, int h_row0, int n_out);
    int32_t mtp_draft_token() const;
};

}  // namespace klein
