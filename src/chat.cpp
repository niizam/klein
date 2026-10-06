// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "chat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

namespace klein {
namespace {

const char* const k_effort_xhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
    "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const char* const k_effort_low =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
    "unnecessary elaboration.";

const char* const k_tools_header =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";

const char* const k_tool_instructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
    "within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but "
    "NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n"
    "</IMPORTANT>";

const char* const k_vision_image = "<|vision_start|><|image_pad|><|vision_end|>";
const char* const k_vision_video = "<|vision_start|><|video_pad|><|vision_end|>";

// Built from adjacent literals so the source never contains the raw tag text.
const char* const k_think_open = "<" "think>";
const char* const k_think_close = "</" "think>";

// Python str.strip() over the whitespace the task specifies (space, \t, \n, \r, \v, \f), matching Jinja's `|trim`.
bool is_ws(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

std::string strip(const std::string& s) {
    size_t begin = 0;
    size_t end = s.size();
    while (begin < end && is_ws((unsigned char) s[begin])) ++begin;
    while (end > begin && is_ws((unsigned char) s[end - 1])) --end;
    return s.substr(begin, end - begin);
}

bool starts_with(const std::string& s, const char* prefix) {
    const size_t n = std::strlen(prefix);
    return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

bool ends_with(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && std::memcmp(s.data() + (s.size() - n), suffix, n) == 0;
}

const json& member(const json& object, const char* key) {
    static const json k_null = json(nullptr);
    if (object.is_object() && object.contains(key)) return object[key];
    return k_null;
}

// ----------------------------------------------------------------------------------------------------------------
// json -> the exact text `json.dumps(x, ensure_ascii=False, indent=None, separators=(", ", ": "), sort_keys=False)`
// produces, which is what the template's `tojson` filter emits.
// ----------------------------------------------------------------------------------------------------------------

std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (const unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
        }
    }
    out.push_back('"');
    return out;
}

// Python's repr() of a finite float, which is also what json.dumps emits for it.
std::string py_float_repr(double value) {
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return value < 0 ? "-Infinity" : "Infinity";
    if (value == 0.0) return std::signbit(value) ? "-0.0" : "0.0";

    const bool negative = value < 0;
    const double magnitude = negative ? -value : value;

    // Shortest %e representation that round-trips, the same number of digits Python's repr uses.
    char buf[64];
    int precision = 17;
    for (int p = 1; p <= 17; ++p) {
        std::snprintf(buf, sizeof(buf), "%.*e", p - 1, magnitude);
        if (std::strtod(buf, nullptr) == magnitude) {
            precision = p;
            break;
        }
    }
    std::snprintf(buf, sizeof(buf), "%.*e", precision - 1, magnitude);

    const std::string sci = buf;
    const size_t epos = sci.find('e');
    const int exponent = std::atoi(sci.c_str() + epos + 1);
    std::string digits;
    for (const char c : sci.substr(0, epos)) {
        if (c >= '0' && c <= '9') digits.push_back(c);
    }
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();

    const int decpt = exponent + 1;  // digits before the decimal point
    std::string out;
    if (decpt <= -4 || decpt > 16) {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char ebuf[16];
        std::snprintf(ebuf, sizeof(ebuf), "%+03d", decpt - 1);
        out += "e";
        out += ebuf;
    } else if (decpt <= 0) {
        out = "0." + std::string((size_t) -decpt, '0') + digits;
    } else if ((size_t) decpt >= digits.size()) {
        out = digits + std::string((size_t) decpt - digits.size(), '0') + ".0";
    } else {
        out = digits.substr(0, (size_t) decpt) + "." + digits.substr((size_t) decpt);
    }
    return negative ? "-" + out : out;
}

std::string to_python_json(const json& value) {
    switch (value.type()) {
        case json::value_t::null: return "null";
        case json::value_t::boolean: return value.get<bool>() ? "true" : "false";
        case json::value_t::string: return escape_json_string(value.get<std::string>());
        case json::value_t::number_unsigned: return std::to_string(value.get<uint64_t>());
        case json::value_t::number_integer: return std::to_string(value.get<int64_t>());
        case json::value_t::number_float: return py_float_repr(value.get<double>());
        case json::value_t::array: {
            std::string out = "[";
            bool first = true;
            for (const auto& element : value) {
                if (!first) out += ", ";
                out += to_python_json(element);
                first = false;
            }
            out += "]";
            return out;
        }
        case json::value_t::object: {
            std::string out = "{";
            bool first = true;
            for (auto it = value.begin(); it != value.end(); ++it) {
                if (!first) out += ", ";
                out += escape_json_string(it.key());
                out += ": ";
                out += to_python_json(it.value());
                first = false;
            }
            out += "}";
            return out;
        }
        default: return "null";
    }
}

// ----------------------------------------------------------------------------------------------------------------
// render_content macro
// ----------------------------------------------------------------------------------------------------------------

std::string render_content(const json& content, bool is_system_content) {
    if (content.is_string()) return content.get<std::string>();

    if (content.is_array()) {
        std::string out;
        for (const auto& item : content) {
            const bool is_image = item.is_object() && (item.contains("image") || item.contains("image_url") ||
                                                       (item.contains("type") && item["type"] == "image"));
            const bool is_video = item.is_object() && (item.contains("video") ||
                                                       (item.contains("type") && item["type"] == "video"));
            if (is_image) {
                if (is_system_content) throw std::runtime_error("System message cannot contain images.");
                out += k_vision_image;
            } else if (is_video) {
                if (is_system_content) throw std::runtime_error("System message cannot contain videos.");
                out += k_vision_video;
            } else if (item.is_object() && item.contains("text")) {
                const json& text = item["text"];
                if (text.is_string()) out += text.get<std::string>();
                else if (text.is_number_unsigned()) out += std::to_string(text.get<uint64_t>());
                else if (text.is_number_integer()) out += std::to_string(text.get<int64_t>());
                else if (text.is_number_float()) out += py_float_repr(text.get<double>());
                else if (text.is_boolean()) out += text.get<bool>() ? "True" : "False";
                else if (text.is_null()) out += "None";
            } else {
                throw std::runtime_error("Unexpected item type in content.");
            }
        }
        return out;
    }

    if (content.is_null()) return "";
    throw std::runtime_error("Unexpected content type.");
}

// ----------------------------------------------------------------------------------------------------------------
// parse_output helpers
// ----------------------------------------------------------------------------------------------------------------

std::string random_hex8() {
    static std::mt19937_64 rng(std::random_device{}());
    static const char* digits = "0123456789abcdef";
    std::uniform_int_distribution<int> dist(0, 15);
    std::string out;
    out.reserve(8);
    for (int i = 0; i < 8; ++i) out.push_back(digits[dist(rng)]);
    return out;
}

bool schema_param_type(const json& tools, const std::string& function_name, const std::string& param,
                       std::string& type_out) {
    if (!tools.is_array()) return false;
    for (const auto& tool : tools) {
        const json* fn = nullptr;
        if (tool.is_object() && tool.contains("function")) fn = &tool["function"];
        else if (tool.is_object()) fn = &tool;
        if (fn == nullptr || !fn->is_object()) continue;
        if (fn->value("name", std::string()) != function_name) continue;
        if (!fn->contains("parameters") || !(*fn)["parameters"].is_object()) return false;
        const json& params = (*fn)["parameters"];
        if (!params.contains("properties") || !params["properties"].is_object()) return false;
        const json& props = params["properties"];
        if (!props.contains(param) || !props[param].is_object() || !props[param].contains("type")) return false;
        type_out = props[param]["type"].get<std::string>();
        return true;
    }
    return false;
}

bool parse_tool_block(const std::string& block, const json& tools, const std::string& id, json& call_out) {
    const size_t function_pos = block.find("<function=");
    if (function_pos == std::string::npos) return false;
    const size_t name_begin = function_pos + std::strlen("<function=");
    const size_t name_end = block.find('>', name_begin);
    if (name_end == std::string::npos) return false;
    const std::string name = block.substr(name_begin, name_end - name_begin);

    const size_t function_close = block.find("</function>", name_end);
    if (function_close == std::string::npos) return false;
    const std::string body = block.substr(name_end + 1, function_close - (name_end + 1));

    json arguments = json::object();
    size_t pos = 0;
    while (true) {
        const size_t param_pos = body.find("<parameter=", pos);
        if (param_pos == std::string::npos) break;
        const size_t pname_begin = param_pos + std::strlen("<parameter=");
        const size_t pname_end = body.find('>', pname_begin);
        if (pname_end == std::string::npos) break;
        const std::string param_name = body.substr(pname_begin, pname_end - pname_begin);
        const size_t param_close = body.find("</parameter>", pname_end);
        if (param_close == std::string::npos) break;
        std::string value = body.substr(pname_end + 1, param_close - (pname_end + 1));
        if (!value.empty() && value.front() == '\n') value.erase(0, 1);
        if (!value.empty() && value.back() == '\n') value.pop_back();

        std::string type;
        json typed;
        if (schema_param_type(tools, name, param_name, type) && type != "string") {
            try {
                typed = json::parse(value);
            } catch (const std::exception&) {
                typed = value;
            }
        } else {
            typed = value;
        }
        arguments[param_name] = typed;
        pos = param_close + std::strlen("</parameter>");
    }

    json function;
    function["name"] = name;
    function["arguments"] = arguments.dump();
    json call;
    call["id"] = id;
    call["type"] = "function";
    call["function"] = function;
    call_out = call;
    return true;
}

}  // namespace

std::string render_chat(const json& messages, const json& tools, const ChatOptions& opt) {
    if (!messages.is_array() || messages.empty()) throw std::runtime_error("No messages provided.");

    std::string reasoning_instructions;
    if (opt.enable_thinking) {
        const std::string& effort = opt.reasoning_effort;
        if (effort != "xhigh" && effort != "medium" && effort != "low") {
            throw std::runtime_error("Unexpected reasoning effort " + effort +
                                     ". Supported types are xhigh (default), medium, and low.");
        }
        if (effort == "xhigh") reasoning_instructions = k_effort_xhigh;
        else if (effort == "low") reasoning_instructions = k_effort_low;
    }

    const std::string first_role = member(messages[0], "role").is_string()
                                       ? messages[0].value("role", std::string())
                                       : std::string();
    const bool has_tools = tools.is_array() && !tools.empty();

    std::string out;
    if (has_tools) {
        out += "<|im_start|>system\n";
        if (!reasoning_instructions.empty()) out += reasoning_instructions + "\n\n";
        out += k_tools_header;
        for (const auto& tool : tools) {
            out += "\n";
            out += to_python_json(tool);
        }
        out += "\n</tools>";
        out += k_tool_instructions;
        if (first_role == "system") {
            const std::string content = strip(render_content(member(messages[0], "content"), true));
            if (!content.empty()) out += "\n\n" + content;
        }
        out += "<|im_end|>\n";
    } else if (first_role == "system") {
        const std::string content = strip(render_content(member(messages[0], "content"), true));
        if (!content.empty()) {
            out += "<|im_start|>system\n";
            if (!reasoning_instructions.empty()) out += reasoning_instructions + "\n\n";
            out += content + "<|im_end|>\n";
        } else if (!reasoning_instructions.empty()) {
            out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
        }
    } else if (!reasoning_instructions.empty()) {
        out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
    }

    // Find the last genuine user query (scanning backwards), skipping tool-response-only user messages.
    bool multi_step_tool = true;
    int last_query_index = (int) messages.size() - 1;
    for (int i = (int) messages.size() - 1; i >= 0 && multi_step_tool; --i) {
        const json& message = messages[i];
        if (message.value("role", std::string()) != "user") continue;
        const std::string content = strip(render_content(member(message, "content"), false));
        if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) {
            multi_step_tool = false;
            last_query_index = i;
        }
    }
    if (multi_step_tool) throw std::runtime_error("No user query found in messages.");

    const int count = (int) messages.size();
    for (int i = 0; i < count; ++i) {
        const json& message = messages[i];
        const std::string role = message.value("role", std::string());
        const std::string content = strip(render_content(member(message, "content"), false));

        if (role == "system") {
            if (i != 0) throw std::runtime_error("System message must be at the beginning.");
        } else if (role == "user") {
            out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string reasoning_content;
            if (message.contains("reasoning_content") && message["reasoning_content"].is_string()) {
                reasoning_content = message["reasoning_content"].get<std::string>();
            }
            reasoning_content = strip(reasoning_content);
            if (opt.preserve_thinking || i > last_query_index) {
                out += "<|im_start|>" + role + "\n" + k_think_open + "\n" + reasoning_content + "\n" +
                       k_think_close + "\n\n" + content;
            } else {
                out += "<|im_start|>" + role + "\n" + content;
            }

            if (message.contains("tool_calls") && message["tool_calls"].is_array() &&
                !message["tool_calls"].empty()) {
                bool first_call = true;
                for (const auto& raw_call : message["tool_calls"]) {
                    json tool_call = raw_call;
                    if (tool_call.is_object() && tool_call.contains("function")) tool_call = tool_call["function"];
                    const std::string name =
                        tool_call.is_object() ? tool_call.value("name", std::string()) : std::string();

                    if (first_call) {
                        if (!strip(content).empty()) out += "\n\n<tool_call>\n<function=" + name + ">\n";
                        else out += "<tool_call>\n<function=" + name + ">\n";
                    } else {
                        out += "\n<tool_call>\n<function=" + name + ">\n";
                    }
                    first_call = false;

                    if (tool_call.is_object() && tool_call.contains("arguments")) {
                        const json& raw_args = tool_call["arguments"];
                        const bool empty_string = raw_args.is_string() && raw_args.get<std::string>().empty();
                        if (!empty_string) {
                            json arguments = raw_args;
                            if (arguments.is_string()) arguments = json::parse(arguments.get<std::string>());
                            if (!arguments.is_object()) {
                                throw std::runtime_error("The 'items' filter requires a mapping.");
                            }
                            for (auto it = arguments.begin(); it != arguments.end(); ++it) {
                                out += "<parameter=" + it.key() + ">\n";
                                if (it.value().is_string()) out += it.value().get<std::string>();
                                else out += to_python_json(it.value());
                                out += "\n</parameter>\n";
                            }
                        }
                    }
                    out += "</function>\n</tool_call>";
                }
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            const bool previous_is_tool =
                i > 0 && messages[i - 1].value("role", std::string()) == "tool";
            const bool next_is_tool =
                i + 1 < count && messages[i + 1].value("role", std::string()) == "tool";
            if (!previous_is_tool) out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (i == count - 1 || !next_is_tool) out += "<|im_end|>\n";
        } else {
            throw std::runtime_error("Unexpected message role.");
        }
    }

    if (opt.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (!opt.enable_thinking) {
            out += std::string(k_think_open) + "\n\n" + k_think_close + "\n\n";
        } else {
            out += std::string(k_think_open) + "\n";
        }
    }
    return out;
}

ParsedOutput parse_output(const std::string& text, bool thinking_open, const json& tools) {
    ParsedOutput result;

    std::string reasoning;
    std::string remainder;
    if (thinking_open) {
        const size_t close = text.find(k_think_close);
        if (close == std::string::npos) {
            reasoning = text;
        } else {
            reasoning = text.substr(0, close);
            remainder = text.substr(close + std::strlen(k_think_close));
        }
    } else {
        const size_t open = text.find(k_think_open);
        if (open == std::string::npos) {
            remainder = text;
        } else {
            const size_t close = text.find(k_think_close, open + std::strlen(k_think_open));
            if (close == std::string::npos) {
                remainder = text;
            } else {
                reasoning =
                    text.substr(open + std::strlen(k_think_open), close - (open + std::strlen(k_think_open)));
                remainder = text.substr(close + std::strlen(k_think_close));
            }
        }
    }
    result.reasoning = strip(reasoning);

    std::string content;
    size_t pos = 0;
    size_t call_index = 0;
    while (true) {
        const size_t start = remainder.find("<tool_call>", pos);
        if (start == std::string::npos) {
            content += remainder.substr(pos);
            break;
        }
        const size_t end = remainder.find("</tool_call>", start);
        if (end == std::string::npos) {
            content += remainder.substr(pos, start - pos);
            break;  // unterminated tool call at the end of generation: ignore it
        }
        content += remainder.substr(pos, start - pos);
        const std::string block = remainder.substr(start + std::strlen("<tool_call>"),
                                                   end - (start + std::strlen("<tool_call>")));
        json call;
        if (parse_tool_block(block, tools, "call_" + std::to_string(call_index) + "_" + random_hex8(), call)) {
            result.tool_calls.push_back(call);
            ++call_index;
        }
        pos = end + std::strlen("</tool_call>");
    }
    result.content = strip(content);
    return result;
}

StreamSplitter::StreamSplitter(bool thinking_open)
    : mode_(thinking_open ? Mode::Reasoning : Mode::Content) {}

StreamSplitter::Delta StreamSplitter::feed(const std::string& piece) {
    pending_ += piece;
    Delta delta;
    static const char* const tags[] = {k_think_open, k_think_close, "<tool_call>", "</tool_call>"};
    const int tag_count = 4;

    auto emit = [&](const std::string& text) {
        if (mode_ == Mode::Reasoning) {
            delta.reasoning += text;
        } else if (mode_ == Mode::Content) {
            size_t skip = 0;
            if (!content_started_) {
                while (skip < text.size() && is_ws((unsigned char) text[skip])) ++skip;
            }
            if (skip < text.size()) {
                delta.content += text.substr(skip);
                content_started_ = true;
            }
        }
    };

    while (true) {
        size_t best_pos = std::string::npos;
        int best_tag = -1;
        for (int i = 0; i < tag_count; ++i) {
            const size_t p = pending_.find(tags[i]);
            if (p != std::string::npos && (best_pos == std::string::npos || p < best_pos)) {
                best_pos = p;
                best_tag = i;
            }
        }

        size_t hold = 0;
        for (int i = 0; i < tag_count; ++i) {
            const size_t tag_len = std::strlen(tags[i]);
            const size_t max_k = std::min(tag_len - 1, pending_.size());
            for (size_t k = max_k; k >= 1; --k) {
                if (pending_.compare(pending_.size() - k, k, tags[i], k) == 0) {
                    hold = std::max(hold, k);
                    break;
                }
            }
        }

        const size_t safe_end = pending_.size() >= hold ? pending_.size() - hold : 0;
        if (best_pos != std::string::npos && best_pos < safe_end) {
            emit(pending_.substr(0, best_pos));
            if (best_tag == 0) {  // k_think_open
                mode_ = Mode::Content;
                content_started_ = false;
            } else if (best_tag == 1) {  // k_think_close
                if (mode_ == Mode::Reasoning) {
                    mode_ = Mode::Content;
                    content_started_ = false;
                }
            } else if (best_tag == 2) {  // <tool_call>
                mode_ = Mode::ToolCall;
            } else if (best_tag == 3) {  // </tool_call>
                mode_ = Mode::Content;
            }
            pending_.erase(0, best_pos + std::strlen(tags[best_tag]));
        } else {
            emit(pending_.substr(0, safe_end));
            pending_.erase(0, safe_end);
            break;
        }
    }
    return delta;
}

StreamSplitter::Delta StreamSplitter::flush() {
    Delta delta;
    if (mode_ != Mode::ToolCall && !pending_.empty()) {
        if (mode_ == Mode::Reasoning) {
            delta.reasoning += pending_;
        } else if (mode_ == Mode::Content) {
            size_t skip = 0;
            if (!content_started_) {
                while (skip < pending_.size() && is_ws((unsigned char) pending_[skip])) ++skip;
            }
            if (skip < pending_.size()) {
                delta.content += pending_.substr(skip);
                content_started_ = true;
            }
        }
    }
    pending_.clear();
    return delta;
}

}  // namespace klein
