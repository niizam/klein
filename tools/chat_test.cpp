// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// klein-chat-test: renders every case in the JSON file given as argv[1] with render_chat and writes one JSON object
// per line ({"name": ..., "output": ...}, or "error" when render_chat throws) so it can be diffed against the real
// Jinja template rendered by tests/chat/render_reference.py.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "chat.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s cases.json\n", argv[0]);
        return 2;
    }

    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();

    klein::json cases;
    try {
        cases = klein::json::parse(buffer.str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "invalid cases json: %s\n", e.what());
        return 2;
    }
    if (!cases.is_array()) {
        std::fprintf(stderr, "cases json must be an array\n");
        return 2;
    }

    for (const auto& test_case : cases) {
        const std::string name = test_case.value("name", std::string());
        klein::json messages = test_case.value("messages", klein::json::array());
        klein::json tools = test_case.contains("tools") ? test_case["tools"] : klein::json(nullptr);

        klein::ChatOptions options;
        if (test_case.contains("options")) {
            const klein::json& o = test_case["options"];
            if (o.contains("add_generation_prompt")) options.add_generation_prompt = o["add_generation_prompt"];
            if (o.contains("enable_thinking")) options.enable_thinking = o["enable_thinking"];
            if (o.contains("reasoning_effort")) options.reasoning_effort = o["reasoning_effort"];
            if (o.contains("preserve_thinking")) options.preserve_thinking = o["preserve_thinking"];
        }

        klein::json line;
        line["name"] = name;
        try {
            line["output"] = klein::render_chat(messages, tools, options);
        } catch (const std::exception& e) {
            line["error"] = std::string(e.what());
        }
        std::cout << line.dump() << "\n";
    }
    return 0;
}
