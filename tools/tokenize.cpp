// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// klein-tokenize: prints token ids for a text, in the same format as llama.cpp's `llama-tokenize --ids`
// so the two can be compared directly.
#include <cstring>
#include <fstream>
#include <sstream>

#include "common.h"
#include "gguf_file.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
    std::string model, text;
    bool parse_special = true, roundtrip = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model = argv[++i];
        else if (a == "-p" && i + 1 < argc) text = argv[++i];
        else if (a == "-f" && i + 1 < argc) {
            std::ifstream f(argv[++i], std::ios::binary);
            std::stringstream ss;
            ss << f.rdbuf();
            text = ss.str();
        } else if (a == "--no-parse-special") parse_special = false;
        else if (a == "--roundtrip") roundtrip = true;
        else {
            std::fprintf(stderr, "usage: %s -m model.gguf (-p text | -f file) [--no-parse-special] [--roundtrip]\n", argv[0]);
            return 1;
        }
    }
    if (model.empty()) {
        std::fprintf(stderr, "missing -m\n");
        return 1;
    }
    klein::GgufFile f(model);
    klein::Tokenizer tok;
    tok.load(f);
    const double t0 = klein::now_ms();
    const auto ids = tok.encode(text, parse_special);
    const double t1 = klein::now_ms();
    std::printf("[");
    for (size_t i = 0; i < ids.size(); ++i) std::printf(i ? ", %d" : "%d", ids[i]);
    std::printf("]\n");
    std::fprintf(stderr, "%zu tokens in %.2f ms\n", ids.size(), t1 - t0);
    if (roundtrip) {
        const std::string back = tok.decode(ids, true);
        std::fprintf(stderr, "roundtrip %s\n", back == text ? "OK" : "MISMATCH");
        return back == text ? 0 : 2;
    }
    return 0;
}
