// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "tokenizer.h"

#include "common.h"
#include "gguf_file.h"
#include "unicode.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace klein {

// Byte-level BPE tokenizer for Qwen3.5/3.6/3.8 (GGUF `tokenizer.ggml.model == "gpt2"` with the `qwen35`
// pre-tokenizer). The merge loop and the special-token partition follow llama.cpp's `llama-vocab.cpp`
// (MIT License): `llm_tokenizer_bpe_session` and `tokenizer_st_partition`.
namespace {

// Qwen3.5/3.8 pre-tokenizer, exactly as llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35. `unicode_regex_split`
// dispatches on this literal to its hand-written qwen35 splitter.
const char* const k_qwen35_regex =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

}  // namespace

void Tokenizer::load(const GgufFile& f) {
    const std::string model = f.get_str("tokenizer.ggml.model", "");
    if (model != "gpt2") {
        fatal("tokenizer: unsupported tokenizer.ggml.model '%s' (expected 'gpt2')", model.c_str());
    }

    const size_t n = f.get_arr_n("tokenizer.ggml.tokens");
    id_to_text_.resize(n);
    text_to_id_.clear();
    text_to_id_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        id_to_text_[i] = f.get_arr_str("tokenizer.ggml.tokens", i);
        text_to_id_[id_to_text_[i]] = (int32_t) i;  // last wins, as in llama.cpp
    }

    token_type_.assign(n, 1);  // LLAMA_TOKEN_TYPE_NORMAL
    const std::vector<int64_t> types = f.get_arr_int("tokenizer.ggml.token_type");
    for (size_t i = 0; i < n && i < types.size(); ++i) {
        token_type_[i] = (int32_t) types[i];
    }

    merge_rank_.clear();
    const size_t n_merges = f.get_arr_n("tokenizer.ggml.merges");
    merge_rank_.reserve(n_merges * 2);
    for (size_t i = 0; i < n_merges; ++i) {
        const std::string m = f.get_arr_str("tokenizer.ggml.merges", i);
        // llama.cpp: split after the first space (search starts at index 1), so rank keys are "left right".
        const size_t pos = m.find(' ', 1);
        std::string first;
        std::string second;
        if (pos != std::string::npos) {
            first = m.substr(0, pos);
            second = m.substr(pos + 1);
        }
        merge_rank_.emplace(first + " " + second, (int32_t) i);  // first wins (std::unordered_map::emplace)
    }

    bos_ = (int32_t) f.get_u32("tokenizer.ggml.bos_token_id", 0xFFFFFFFFu);
    eos_ = (int32_t) f.get_u32("tokenizer.ggml.eos_token_id", 0xFFFFFFFFu);
    pad_ = (int32_t) f.get_u32("tokenizer.ggml.padding_token_id", 0xFFFFFFFFu);

    // Special-token candidates: CONTROL and USER_DEFINED, longest text first (matches llama.cpp's cache).
    special_ids_.clear();
    for (int32_t id = 0; id < (int32_t) n; ++id) {
        if (token_type_[id] == 3 /*CONTROL*/ || token_type_[id] == 4 /*USER_DEFINED*/) {
            special_ids_.push_back(id);
        }
    }
    std::sort(special_ids_.begin(), special_ids_.end(), [this](int32_t a, int32_t b) {
        return id_to_text_[a].size() > id_to_text_[b].size();
    });

    eog_ids_.clear();
    if (eos_ >= 0 && eos_ < (int32_t) n) eog_ids_.push_back(eos_);
    static const char* const eog_names[] = {
        "<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>", "<|return|>",
    };
    for (const char* name : eog_names) {
        const auto it = text_to_id_.find(name);
        if (it == text_to_id_.end()) continue;
        if (std::find(eog_ids_.begin(), eog_ids_.end(), it->second) == eog_ids_.end()) {
            eog_ids_.push_back(it->second);
        }
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int32_t> out;
    if (text.empty()) return out;

    // Split the input at special tokens, like llama.cpp's tokenizer_st_partition (no lstrip/rstrip here).
    struct Fragment {
        bool is_token;
        std::string text;
        int32_t id;
    };

    std::vector<Fragment> fragments;
    fragments.push_back(Fragment{false, text, -1});

    for (const int32_t sid : special_ids_) {
        const int type = sid < (int32_t) token_type_.size() ? token_type_[sid] : 1;
        // Without parse_special, CONTROL tokens are not matched; USER_DEFINED tokens still are.
        if (!parse_special && type != 4) continue;

        const std::string& special = id_to_text_[sid];
        if (special.empty()) continue;

        std::vector<Fragment> next;
        next.reserve(fragments.size());
        for (const Fragment& frag : fragments) {
            if (frag.is_token) {
                next.push_back(frag);
                continue;
            }
            const std::string& s = frag.text;
            size_t pos = 0;
            while (pos <= s.size()) {
                const size_t match = s.find(special, pos);
                if (match == std::string::npos) {
                    if (pos < s.size()) next.push_back(Fragment{false, s.substr(pos), -1});
                    break;
                }
                if (match > pos) next.push_back(Fragment{false, s.substr(pos, match - pos), -1});
                next.push_back(Fragment{true, std::string(), sid});
                pos = match + special.size();
            }
        }
        fragments.swap(next);
    }

    for (const Fragment& frag : fragments) {
        if (frag.is_token) {
            out.push_back(frag.id);
        } else {
            encode_text(frag.text, out);
        }
    }
    return out;
}

void Tokenizer::encode_text(const std::string& text, std::vector<int32_t>& out) const {
    static const std::vector<std::string> regex_exprs = { k_qwen35_regex };
    const std::vector<std::string> words = unicode_regex_split(text, regex_exprs, /*byte_encode =*/true);
    for (const std::string& word : words) {
        bpe_word(word, out);
    }
}

// BPE over one pre-tokenized (already byte-encoded) word. The merge loop mirrors llama.cpp's
// llm_tokenizer_bpe_session: a priority queue ordered by merge rank (ties resolved leftmost first).
void Tokenizer::bpe_word(const std::string& word, std::vector<int32_t>& out) const {
    struct Symbol {
        int prev;
        int next;
        size_t pos;  // offset into `word`
        size_t n;    // byte length; 0 once merged into the left neighbour
    };

    struct Bigram {
        int left;
        int right;
        int rank;
        std::string text;
    };

    struct BigramCompare {
        bool operator()(const Bigram& l, const Bigram& r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    std::vector<Symbol> symbols;
    symbols.reserve(word.size());
    {
        size_t offset = 0;
        int index = 0;
        int prev = -1;
        while (offset < word.size()) {
            size_t len = unicode_len_utf8(word[offset]);
            if (len == 0) len = 1;
            len = std::min(len, word.size() - offset);
            Symbol sym;
            sym.prev = prev;
            sym.next = -1;
            sym.pos = offset;
            sym.n = len;
            if (prev >= 0) symbols[prev].next = index;
            symbols.push_back(sym);
            prev = index;
            offset += len;
            ++index;
        }
    }

    std::priority_queue<Bigram, std::vector<Bigram>, BigramCompare> queue;

    auto add_bigram = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        const Symbol& ls = symbols[left];
        const Symbol& rs = symbols[right];
        if (ls.n == 0 || rs.n == 0) return;
        const std::string left_token = word.substr(ls.pos, ls.n);
        const std::string right_token = word.substr(rs.pos, rs.n);
        const auto it = merge_rank_.find(left_token + " " + right_token);
        if (it == merge_rank_.end()) return;
        queue.push(Bigram{left, right, it->second, left_token + right_token});
    };

    for (int i = 1; i < (int) symbols.size(); ++i) {
        add_bigram(i - 1, i);
    }

    while (!queue.empty()) {
        const Bigram bigram = queue.top();
        queue.pop();

        Symbol& left_symbol = symbols[bigram.left];
        Symbol& right_symbol = symbols[bigram.right];
        if (left_symbol.n == 0 || right_symbol.n == 0) continue;

        const std::string left_token = word.substr(left_symbol.pos, left_symbol.n);
        const std::string right_token = word.substr(right_symbol.pos, right_symbol.n);
        if (left_token + right_token != bigram.text) continue;  // stale bigram

        // merge the right symbol into the left one and unlink it
        left_symbol.n += right_symbol.n;
        right_symbol.n = 0;
        left_symbol.next = right_symbol.next;
        if (right_symbol.next >= 0) {
            symbols[right_symbol.next].prev = bigram.left;
        }

        add_bigram(left_symbol.prev, bigram.left);
        add_bigram(bigram.left, left_symbol.next);
    }

    // Map the surviving symbols to ids; unknown symbols fall back to their individual byte tokens.
    for (const Symbol& sym : symbols) {
        if (sym.n == 0) continue;
        const std::string str = word.substr(sym.pos, sym.n);
        const auto it = text_to_id_.find(str);
        if (it != text_to_id_.end()) {
            out.push_back(it->second);
            continue;
        }
        // Each codepoint of the byte-encoded word is exactly one byte's byte-encoded string.
        size_t off = 0;
        while (off < str.size()) {
            size_t len = unicode_len_utf8(str[off]);
            if (len == 0 || len > str.size() - off) len = 1;
            const auto bit = text_to_id_.find(str.substr(off, len));
            if (bit != text_to_id_.end()) out.push_back(bit->second);
            off += len;
        }
    }
}

std::string Tokenizer::token_to_piece(int32_t id, bool special) const {
    if (id < 0 || id >= (int32_t) id_to_text_.size()) return "";

    const int type = id < (int32_t) token_type_.size() ? token_type_[id] : 1;
    const std::string& text = id_to_text_[id];

    // UNKNOWN / CONTROL: returned only when `special` is set (as in llama.cpp).
    if (type == 2 || type == 3) {
        return special ? text : std::string();
    }

    // USER_DEFINED: llama.cpp returns the raw text (not byte-decoded).
    if (type == 4) {
        return text;
    }

    if (type == 1) {  // NORMAL: byte-decode the GPT-2 byte-to-unicode encoded text
        std::string out;
        const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
        for (const uint32_t cpt : cpts) {
            const std::string utf8 = unicode_cpt_to_utf8(cpt);
            try {
                out += unicode_utf8_to_byte(utf8);
            } catch (const std::out_of_range&) {
                out += "[UNK_BYTE_0x";
                for (const char c : utf8) out += format("%02x", (uint8_t) c);
                out += text + "]";
            }
        }
        return out;
    }

    if (type == 6) {  // BYTE: text is "<0xXX>"
        if (text.size() == 6 && text[0] == '<' && text[1] == '0' && text[2] == 'x' && text[5] == '>') {
            const int hi = hex_digit(text[3]);
            const int lo = hex_digit(text[4]);
            if (hi >= 0 && lo >= 0) {
                return std::string(1, (char) ((hi << 4) | lo));
            }
        }
        return "";
    }

    // UNUSED / UNDEFINED (and any unsupported type): empty.
    return "";
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool special) const {
    std::string out;
    for (const int32_t id : ids) {
        out += token_to_piece(id, special);
    }
    return out;
}

bool Tokenizer::is_eog(int32_t id) const {
    return std::find(eog_ids_.begin(), eog_ids_.end(), id) != eog_ids_.end();
}

bool Tokenizer::is_control(int32_t id) const {
    return id >= 0 && id < (int32_t) token_type_.size() && token_type_[id] == 3;
}

int32_t Tokenizer::find(const std::string& token_text) const {
    const auto it = text_to_id_.find(token_text);
    return it == text_to_id_.end() ? -1 : it->second;
}

}  // namespace klein
