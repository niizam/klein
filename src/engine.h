// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "image.h"
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
    std::string snap_type = "f16";  // rollback snapshot storage: f16 (half the VRAM, 10-bit mantissa), bf16 or f32
    std::string mmproj_path;       // Qwen3-VL vision encoder GGUF (enables images)
    int image_min_tokens = 8;      // bounds on the tokens one image becomes (each covers 32x32 pixels)
    int image_max_tokens = 1024;
};

// An image encoded for the language model: grid_x * grid_y embeddings (row-major), one per 32x32-pixel cell.
struct PromptImage {
    std::vector<float> embd;
    int grid_x = 0, grid_y = 0;
    uint64_t hash = 0;  // of the pixels, for prompt reuse
};

struct GenStats {
    int n_prompt = 0;        // prompt cells prefilled (after prompt reuse)
    int n_prompt_total = 0;  // prompt length in cells, images expanded
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
    // Same, with images: the i-th <|image_pad|> token of `prompt` stands for images[i].
    GenStats generate(const std::vector<int32_t>& prompt, const std::vector<PromptImage>& images, int n_predict, Sampler& sampler,
                      const std::function<bool(int32_t)>& on_token);

    bool has_vision() const { return vision_ != nullptr; }
    // Runs the vision encoder on the GPU (weights streamed from RAM). Throws if there is no encoder.
    PromptImage encode_image(const ImageRGB& img);
    // Runs the encoder on an already preprocessed input and reports the sum of each checkpoint tensor
    // (patch_bias, inp_pos_emb, layer_out-N, post_ln, v_embd), for comparison with llama.cpp's llama-mtmd-debug.
    std::vector<std::pair<std::string, double>> vision_debug(const ImageInput& in);

private:
    EngineConfig cfg_;
    ggml_backend_t gpu_ = nullptr;
    ggml_backend_t cpu_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    std::unique_ptr<Model> model_;
    std::unique_ptr<State> state_;
    Tokenizer tok_;
    Plan plan_;
    std::vector<uint8_t> meta_buf_[4];   // graph metadata per kind: main/MTP x small/large batch
    bool mtp_ready_ = false;            // MTP logits for the next draft are in mtp_logits_
    bool kv_mapped_ = false;            // host KV cache is mapped into the GPU address space (zero-copy)
    std::vector<float> mtp_logits_;
    std::vector<int32_t> cache_;        // ids of cells 0 .. n_past-1 (tokens; negative for image cells), for prompt reuse
    std::vector<int32_t> cache_rope_after_;  // rope position that follows each cached cell
    std::unique_ptr<class VisionModel> vision_;
    std::vector<uint8_t> vision_meta_;
    int32_t image_pad_id_ = -1;
    int mtp_h_row_ = 0;                 // row of State::mtp_hidden holding the last MTP hidden state

    // One position of the sequence: a token, or an embedding row (image), with its M-RoPE position (t, y, x).
    struct Cell {
        int32_t id;          // token id; negative pseudo id for image cells
        const float* emb;    // image embedding row, or nullptr for tokens
        int32_t rope[3];
        int32_t rope_after;  // rope position of whatever follows this cell
    };
    struct RunArgs {
        const Cell* cells;
        int n;
        int cell0;                      // first KV cell written
        int kv_lo;                      // first valid KV cell for the mask
        int n_out;
        std::vector<float>* logits;
    };
    int rope_next() const;
    static std::vector<Cell> text_cells(const int32_t* tokens, int n, int rope0);
    std::vector<Cell> build_cells(const std::vector<int32_t>& prompt, const std::vector<PromptImage>& images) const;
    void eval_cells(const Cell* cells, int n, int n_out, std::vector<float>* logits);
    void prefill_cells(const std::vector<Cell>& cells, std::vector<float>& last_logits, GenStats* stats);
    void run(bool mtp, const RunArgs& a, ggml_tensor* h_src, ggml_tensor* h_dst);
    size_t compute_need(int n_kv, int n_tokens) const;
    void ensure_prefill_vram(int n_kv_end, int n_tokens);
    void relax_after_prefill();
    void warmup();
    int n_ctx_cfg_ = 0;
    size_t margin_ = 0;
    // MTP pass over cells written at KV cells cell0.., reading main hidden rows [h_row0, h_row0 + n).
    void mtp_pass(const Cell* cells, int n, int cell0, bool from_main, int h_row0, int n_out);
    int32_t mtp_draft_token() const;
};

}  // namespace klein
