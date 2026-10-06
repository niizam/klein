// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <string>

namespace klein {

class Engine;

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;           // empty: no auth (only allowed on loopback hosts)
    std::string model_name = "qwen3.8-27b";
    int default_max_tokens = -1;   // -1: until the context is full
};

// Serves an OpenAI-compatible HTTP API backed by `engine` (one request generates at a time). Blocks until stopped.
int run_server(Engine& engine, const ServerOptions& opt);

}  // namespace klein
