// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace klein {

class GgufFile;

// Byte-level BPE tokenizer for GGUF `tokenizer.ggml.model == "gpt2"` with pre-tokenizer `qwen35`
// (Qwen3.5 / Qwen3.6 / Qwen3.8). Output must match llama.cpp's tokenizer exactly.
class Tokenizer {
public:
    void load(const GgufFile& f);

    // Encodes UTF-8 text. With parse_special, control/user-defined tokens written literally in the text
    // (e.g. "<|im_start|>") become their single token ids; without it they are tokenized as plain text.
    std::vector<int32_t> encode(const std::string& text, bool parse_special) const;

    // The UTF-8 bytes a token stands for (byte-level BPE decoded). Control tokens return their text
    // (e.g. "<|im_end|>") only when `special` is true, otherwise "".
    std::string token_to_piece(int32_t id, bool special = false) const;
    std::string decode(const std::vector<int32_t>& ids, bool special = false) const;

    int32_t n_vocab() const { return (int32_t) id_to_text_.size(); }
    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    int32_t pad() const { return pad_; }
    bool is_eog(int32_t id) const;               // end-of-generation: eos, <|im_end|>, <|endoftext|>, ...
    bool is_control(int32_t id) const;
    int32_t find(const std::string& token_text) const;  // exact vocab text lookup, -1 if missing

private:
    std::vector<std::string> id_to_text_;               // vocab strings as stored (byte-level encoded)
    std::vector<int32_t> token_type_;                   // GGUF tokenizer.ggml.token_type
    std::unordered_map<std::string, int32_t> text_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;  // key: left + ' ' + right
    std::vector<int32_t> special_ids_;                  // control + user-defined, longest text first
    std::vector<int32_t> eog_ids_;
    int32_t bos_ = -1, eos_ = -1, pad_ = -1;

    void bpe_word(const std::string& word, std::vector<int32_t>& out) const;
    void encode_text(const std::string& text, std::vector<int32_t>& out) const;
};

}  // namespace klein
