// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <stdexcept>
#include <string>

#include "json.hpp"

namespace klein {

using json = nlohmann::ordered_json;

// Options of the Qwen3.8 chat template (the GGUF's tokenizer.chat_template, reproduced in C++).
struct ChatOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    std::string reasoning_effort = "xhigh";  // xhigh | medium | low (used only when enable_thinking)
    bool preserve_thinking = true;
};

// Renders OpenAI-style `messages` (and optional `tools`, an array of OpenAI tool objects or null) into the prompt
// text exactly as the Qwen3.8 Jinja template does. Throws std::runtime_error with the template's message on
// invalid input (e.g. "No user query found in messages.").
std::string render_chat(const json& messages, const json& tools, const ChatOptions& opt);

// The model's output after the prompt, split into reasoning, answer text and tool calls.
struct ParsedOutput {
    std::string reasoning;   // text inside <think> ... </think> (trimmed)
    std::string content;     // answer text outside thinking and tool calls (trimmed)
    json tool_calls = json::array();  // [{ "id", "type": "function", "function": { "name", "arguments": "<JSON string>" } }]
};

// `thinking_open`: the prompt ended inside a <think> block (enable_thinking), so the output starts as reasoning.
// `tools` is used to type parameter values (non-"string" schema types are parsed as JSON when possible).
ParsedOutput parse_output(const std::string& text, bool thinking_open, const json& tools);

// Incremental splitter for streaming: feed decoded text pieces, get reasoning/content deltas. Text that might be
// the start of a tag (<think>, </think>, <tool_call>) is held back until it is decided. Tool-call blocks are not
// emitted as content; they are reported by parse_output() at the end.
class StreamSplitter {
public:
    explicit StreamSplitter(bool thinking_open);
    struct Delta {
        std::string reasoning;
        std::string content;
    };
    Delta feed(const std::string& piece);
    Delta flush();  // end of generation: release held-back text

private:
    enum class Mode { Reasoning, Content, ToolCall } mode_;
    std::string pending_;
    bool content_started_ = false;  // leading whitespace after </think> is dropped
};

}  // namespace klein
