// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "httplib.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat.h"
#include "image.h"
#include "index_html.h"
#include "common.h"
#include "engine.h"
#include "sampler.h"
#include "server.h"
#include "tokenizer.h"

namespace klein {
namespace {

// ----------------------------------------------------------------------------------------------------------------
// small helpers
// ----------------------------------------------------------------------------------------------------------------

std::string random_hex(int n) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    static const char* digits = "0123456789abcdef";
    std::uniform_int_distribution<int> dist(0, 15);
    std::string out;
    out.reserve((size_t) n);
    for (int i = 0; i < n; ++i) out.push_back(digits[dist(rng)]);
    return out;
}

bool is_loopback_host(const std::string& host) {
    return host == "127.0.0.1" || host == "::1" || host == "localhost";
}

// http://localhost[:port] or http://127.0.0.1[:port]
bool is_allowed_origin(const std::string& origin) {
    auto match = [&](const char* base) -> bool {
        const size_t n = std::strlen(base);
        if (origin.size() < n || origin.compare(0, n, base) != 0) return false;
        if (origin.size() == n) return true;
        if (origin[n] != ':') return false;
        if (origin.size() == n + 1) return false;
        for (size_t i = n + 1; i < origin.size(); ++i) {
            if (!std::isdigit((unsigned char) origin[i])) return false;
        }
        return true;
    };
    return match("http://localhost") || match("http://127.0.0.1");
}

void apply_cors(const httplib::Request& req, httplib::Response& res) {
    const std::string origin = req.get_header_value("Origin");
    if (is_allowed_origin(origin)) res.set_header("Access-Control-Allow-Origin", origin);
}

void send_error(httplib::Response& res, int status, const std::string& message,
                const char* type = "invalid_request_error") {
    json error;
    error["message"] = message;
    error["type"] = type;
    json out;
    out["error"] = error;
    res.status = status;
    res.set_content(out.dump(), "application/json");
}

bool authorized(const httplib::Request& req, const std::string& api_key) {
    if (api_key.empty()) return true;
    const std::string auth = req.get_header_value("Authorization");
    const std::string prefix = "Bearer ";
    if (auth.size() > prefix.size() && auth.compare(0, prefix.size(), prefix) == 0 &&
        auth.substr(prefix.size()) == api_key) {
        return true;
    }
    return req.get_header_value("x-api-key") == api_key;
}

bool parse_json(const httplib::Request& req, json& out, httplib::Response& res) {
    if (req.body.empty()) {
        send_error(res, 400, "Empty request body.");
        return false;
    }
    try {
        out = json::parse(req.body);
    } catch (const std::exception& e) {
        send_error(res, 400, std::string("Invalid JSON: ") + e.what());
        return false;
    }
    if (!out.is_object()) {
        send_error(res, 400, "Request body must be a JSON object.");
        return false;
    }
    return true;
}

bool get_number(const json& body, const char* key, double& out) {
    if (!body.is_object() || !body.contains(key)) return false;
    const json& v = body[key];
    if (!v.is_number()) return false;
    out = v.get<double>();
    return true;
}

bool get_integer(const json& body, const char* key, long long& out) {
    if (!body.is_object() || !body.contains(key)) return false;
    const json& v = body[key];
    if (v.is_number_integer()) {
        out = v.get<long long>();
        return true;
    }
    if (v.is_number_unsigned()) {
        out = (long long) v.get<unsigned long long>();
        return true;
    }
    if (v.is_number_float()) {
        out = (long long) v.get<double>();
        return true;
    }
    return false;
}

bool get_bool(const json& body, const char* key, bool& out) {
    if (!body.is_object() || !body.contains(key)) return false;
    if (!body[key].is_boolean()) return false;
    out = body[key].get<bool>();
    return true;
}

void parse_sampling(const json& body, SamplerParams& sp) {
    double d;
    long long i;
    if (get_number(body, "temperature", d)) sp.temperature = (float) d;
    if (get_number(body, "top_p", d)) sp.top_p = (float) d;
    if (get_integer(body, "top_k", i)) sp.top_k = (int) i;
    if (get_number(body, "min_p", d)) sp.min_p = (float) d;
    if (get_number(body, "presence_penalty", d)) sp.presence_penalty = (float) d;
    if (get_number(body, "frequency_penalty", d)) sp.frequency_penalty = (float) d;
    if (get_number(body, "repetition_penalty", d)) sp.repetition_penalty = (float) d;
    if (get_integer(body, "seed", i)) sp.seed = (uint64_t) i;
}

void parse_stop(const json& body, std::vector<std::string>& out) {
    if (!body.is_object() || !body.contains("stop")) return;
    const json& s = body["stop"];
    if (s.is_string()) {
        const std::string v = s.get<std::string>();
        if (!v.empty()) out.push_back(v);
    } else if (s.is_array()) {
        for (const auto& e : s) {
            if (e.is_string()) {
                const std::string v = e.get<std::string>();
                if (!v.empty()) out.push_back(v);
            }
        }
    }
}

bool map_effort(const std::string& effort, bool& enable_thinking, std::string& mapped, std::string& err) {
    if (effort == "none" || effort == "minimal") {
        enable_thinking = false;
        return true;
    }
    if (effort == "low") {
        enable_thinking = true;
        mapped = "low";
        return true;
    }
    if (effort == "medium") {
        enable_thinking = true;
        mapped = "medium";
        return true;
    }
    if (effort == "high" || effort == "xhigh") {
        enable_thinking = true;
        mapped = "xhigh";
        return true;
    }
    err = "Unknown reasoning_effort '" + effort +
          "'. Supported: none, minimal, low, medium, high, xhigh.";
    return false;
}

// Cell count such that prompt + prediction fits: -1 with a message on error.
int resolve_max_tokens(const json& body, int prompt_len, int n_ctx, int fallback, std::string& err) {
    long long value = 0;
    const bool has = get_integer(body, "max_completion_tokens", value) ||
                     get_integer(body, "max_tokens", value);
    const long long room = (long long) n_ctx - (long long) prompt_len;
    if (room <= 0) {
        err = "The prompt is too long: it does not fit in the " + std::to_string(n_ctx) +
              "-token context window.";
        return -1;
    }
    long long want = has ? value : (long long) fallback;
    if (want <= 0) {
        err = "max_tokens must be a positive integer.";
        return -1;
    }
    if (want > room) want = room;
    return (int) want;
}

json make_usage(int completion, int prompt) {
    json usage;
    usage["prompt_tokens"] = prompt;
    usage["completion_tokens"] = completion;
    usage["total_tokens"] = prompt + completion;
    return usage;
}

json make_timings(const GenStats& s, int completion) {
    json t;
    t["prompt_n"] = s.n_prompt;
    t["prompt_ms"] = s.t_prompt_ms;
    t["prompt_per_second"] = s.prompt_tps();
    t["predicted_n"] = completion;
    t["predicted_ms"] = s.t_gen_ms;
    t["predicted_per_second"] = s.t_gen_ms > 0.0 ? completion * 1000.0 / s.t_gen_ms : 0.0;
    t["draft_n"] = s.n_drafted;
    t["draft_n_accepted"] = s.n_accepted;
    return t;
}

// ----------------------------------------------------------------------------------------------------------------
// UTF-8 aware output buffer with stop-string handling
// ----------------------------------------------------------------------------------------------------------------

size_t utf8_complete_len(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        size_t need = 1;
        if (c >= 0xF0) {
            need = 4;
        } else if (c >= 0xE0) {
            need = 3;
        } else if (c >= 0xC0) {
            need = 2;
        }
        if (i + need > s.size()) break;
        i += need;
    }
    return i;
}

size_t utf8_floor(const std::string& s, size_t pos) {
    while (pos > 0 && pos < s.size() && (((unsigned char) s[pos]) & 0xC0) == 0x80) --pos;
    return pos;
}

class StopBuffer {
public:
    explicit StopBuffer(const std::vector<std::string>& stops) {
        for (const auto& s : stops) {
            if (s.empty()) continue;
            stops_.push_back(s);
            max_stop_ = std::max(max_stop_, s.size());
        }
    }

    bool stopped() const { return stopped_; }

    void feed(const std::string& piece, const std::function<void(const std::string&)>& emit) {
        partial_ += piece;
        const size_t n = utf8_complete_len(partial_);
        chars_ += partial_.substr(0, n);
        partial_.erase(0, n);
        drain(false, emit);
    }

    void drain(bool final, const std::function<void(const std::string&)>& emit) {
        if (stopped_) return;
        if (stops_.empty()) {
            if (!chars_.empty()) {
                emit(chars_);
                chars_.clear();
            }
            if (final && !partial_.empty()) {
                emit(partial_);
                partial_.clear();
            }
            return;
        }
        for (;;) {
            size_t best = std::string::npos;
            for (const auto& s : stops_) {
                const size_t p = chars_.find(s);
                if (p != std::string::npos && (best == std::string::npos || p < best)) best = p;
            }
            if (best != std::string::npos) {
                if (best > 0) emit(chars_.substr(0, best));
                chars_.clear();
                partial_.clear();
                stopped_ = true;
                return;
            }
            if (final) {
                if (!chars_.empty()) {
                    emit(chars_);
                    chars_.clear();
                }
                if (!partial_.empty()) {
                    emit(partial_);
                    partial_.clear();
                }
                return;
            }
            const size_t hold = max_stop_ > 0 ? max_stop_ - 1 : 0;
            size_t safe = chars_.size() > hold ? chars_.size() - hold : 0;
            safe = utf8_floor(chars_, safe);
            if (safe == 0) return;
            emit(chars_.substr(0, safe));
            chars_.erase(0, safe);
        }
    }

private:
    std::vector<std::string> stops_;
    std::string chars_;
    std::string partial_;
    size_t max_stop_ = 0;
    bool stopped_ = false;
};

// ----------------------------------------------------------------------------------------------------------------
// generation core, shared by streaming and non-streaming endpoints
// ----------------------------------------------------------------------------------------------------------------

struct GenResult {
    GenStats stats;
    std::string text;      // emitted text (truncated at the first stop string)
    int generated = 0;     // generated tokens, EOG excluded
    bool hit_eog = false;
    bool stopped = false;  // a stop string was matched
    bool write_failed = false;
};

// Base64 (standard alphabet, padding optional, whitespace ignored).
std::string base64_decode(const std::string& in) {
    static int8_t map[256];
    static bool init = false;
    if (!init) {
        std::fill(std::begin(map), std::end(map), (int8_t) -1);
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int k = 0; k < 64; ++k) map[(uint8_t) a[k]] = (int8_t) k;
        map[(uint8_t) '-'] = 62;  // base64url
        map[(uint8_t) '_'] = 63;
        init = true;
    }
    std::string out;
    out.reserve(in.size() * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    for (unsigned char c : in) {
        if (c == '=') break;
        const int v = map[c];
        if (v < 0) {
            if (std::isspace(c)) continue;
            throw std::runtime_error("invalid base64 in image data");
        }
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char) ((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// Images of an OpenAI-style message list, in order: content items {"type":"image_url","image_url":{"url":...}} or
// {"type":"image","image":...}, as data: URLs (or bare base64). Remote URLs are refused: klein does not fetch.
std::vector<ImageRGB> extract_images(const json& messages) {
    std::vector<ImageRGB> out;
    if (!messages.is_array()) return out;
    for (const auto& m : messages) {
        if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
        for (const auto& item : m["content"]) {
            if (!item.is_object()) continue;
            std::string url;
            if (item.contains("image_url")) {
                const auto& iu = item["image_url"];
                url = iu.is_string() ? iu.get<std::string>() : (iu.is_object() && iu.contains("url") ? iu["url"].get<std::string>() : "");
            } else if (item.contains("image") && item["image"].is_string()) {
                url = item["image"].get<std::string>();
            } else {
                continue;
            }
            std::string b64 = url;
            if (url.rfind("data:", 0) == 0) {
                const size_t comma = url.find(',');
                if (comma == std::string::npos || url.substr(0, comma).find(";base64") == std::string::npos)
                    throw std::runtime_error("image data URLs must be base64-encoded");
                b64 = url.substr(comma + 1);
            } else if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
                throw std::runtime_error("image URLs are not fetched; send the image as a data: URL (base64)");
            }
            const std::string bytes = base64_decode(b64);
            out.push_back(decode_image((const uint8_t*) bytes.data(), bytes.size()));
        }
    }
    return out;
}

GenResult run_generation(Engine& engine, std::mutex& mutex, const std::vector<int32_t>& tokens,
                         int n_predict, const SamplerParams& sp, const std::vector<std::string>& stops,
                         const std::function<bool(const std::string&)>& emit,
                         const std::vector<ImageRGB>& images_rgb = {}) {
    GenResult result;
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<PromptImage> images;
    for (const ImageRGB& im : images_rgb) images.push_back(engine.encode_image(im));
    Sampler sampler(sp);
    StopBuffer buffer(stops);
    bool ok = true;
    std::function<void(const std::string&)> sink_emit = [&](const std::string& text) {
        if (!ok) return;
        result.text += text;
        if (!emit(text)) ok = false;
    };
    result.stats = engine.generate(tokens, images, n_predict, sampler, [&](int32_t token) -> bool {
        const Tokenizer& tok = engine.tokenizer();
        if (tok.is_eog(token)) {
            result.hit_eog = true;
            return false;
        }
        ++result.generated;
        buffer.feed(tok.token_to_piece(token, false), sink_emit);
        if (buffer.stopped()) {
            result.stopped = true;
            return false;
        }
        if (!ok) {
            result.write_failed = true;
            return false;
        }
        return true;
    });
    buffer.drain(true, sink_emit);
    if (!ok) result.write_failed = true;
    return result;
}

void log_request(const char* endpoint, int prompt_len, const GenResult& r) {
    if (r.stats.n_drafted > 0) {
        KLOG_INFO("%s: prompt %d tokens, generated %d, prompt %.1f tok/s, gen %.1f tok/s, draft %d/%d (%.0f%%)", endpoint,
                  prompt_len, r.generated, r.stats.prompt_tps(), r.stats.gen_tps(), r.stats.n_accepted, r.stats.n_drafted,
                  r.stats.n_drafted ? 100.0 * r.stats.n_accepted / r.stats.n_drafted : 0.0);
    } else {
        KLOG_INFO("%s: prompt %d tokens, generated %d, prompt %.1f tok/s, gen %.1f tok/s", endpoint, prompt_len,
                  r.generated, r.stats.prompt_tps(), r.stats.gen_tps());
    }
}

std::string finish_reason(const GenResult& r) {
    if (r.hit_eog || r.stopped) return "stop";
    return "length";
}

// ----------------------------------------------------------------------------------------------------------------
// streaming
// ----------------------------------------------------------------------------------------------------------------

struct StreamReq {
    Engine* engine = nullptr;
    std::mutex* mutex = nullptr;
    ServerOptions opt;
    std::string endpoint;
    std::string id;
    std::vector<int32_t> tokens;
    int n_predict = 0;
    SamplerParams sp;
    std::vector<std::string> stops;
    bool chat = true;
    bool thinking_open = true;
    bool include_usage = false;
    json tools;
    std::vector<ImageRGB> images;
};

void serve_stream(const StreamReq& st, httplib::DataSink& sink) {
    const long long created = (long long) std::time(nullptr);
    bool write_ok = true;

    auto send = [&](const json& obj) -> bool {
        const std::string data = "data: " + obj.dump() + "\n\n";
        return sink.write(data.data(), data.size());
    };
    auto base = [&]() {
        json obj;
        obj["id"] = st.id;
        obj["object"] = st.chat ? "chat.completion.chunk" : "text_completion";
        obj["created"] = created;
        obj["model"] = st.opt.model_name;
        return obj;
    };
    auto choice = [&](const json& delta, const json& finish) {
        json c;
        c["index"] = 0;
        c["delta"] = delta;
        c["finish_reason"] = finish;
        return c;
    };

    if (st.chat) {
        json delta;
        delta["role"] = "assistant";
        delta["content"] = "";
        json obj = base();
        obj["choices"] = json::array({choice(delta, json(nullptr))});
        if (!send(obj)) write_ok = false;
    }

    StreamSplitter splitter(st.thinking_open);
    std::function<bool(const std::string&)> emit;
    if (st.chat) {
        emit = [&](const std::string& text) -> bool {
            if (!write_ok) return false;
            const StreamSplitter::Delta d = splitter.feed(text);
            if (!d.reasoning.empty()) {
                json delta;
                delta["reasoning_content"] = d.reasoning;
                json obj = base();
                obj["choices"] = json::array({choice(delta, json(nullptr))});
                if (!send(obj)) {
                    write_ok = false;
                    return false;
                }
            }
            if (!d.content.empty()) {
                json delta;
                delta["content"] = d.content;
                json obj = base();
                obj["choices"] = json::array({choice(delta, json(nullptr))});
                if (!send(obj)) {
                    write_ok = false;
                    return false;
                }
            }
            return true;
        };
    } else {
        emit = [&](const std::string& text) -> bool {
            if (!write_ok) return false;
            json c;
            c["index"] = 0;
            c["text"] = text;
            c["finish_reason"] = json(nullptr);
            json obj = base();
            obj["choices"] = json::array({c});
            if (!send(obj)) {
                write_ok = false;
                return false;
            }
            return true;
        };
    }

    GenResult result;
    if (write_ok) {
        try {
            result = run_generation(*st.engine, *st.mutex, st.tokens, st.n_predict, st.sp, st.stops, emit, st.images);
        } catch (const std::exception& e) {
            result.write_failed = true;
            if (write_ok) {
                json error;
                error["message"] = e.what();
                error["type"] = "server_error";
                json obj;
                obj["error"] = error;
                send(obj);
            }
            write_ok = false;
        }
    }

    if (st.chat && write_ok) {
        const StreamSplitter::Delta d = splitter.flush();
        if (!d.reasoning.empty()) {
            json delta;
            delta["reasoning_content"] = d.reasoning;
            json obj = base();
            obj["choices"] = json::array({choice(delta, json(nullptr))});
            if (!send(obj)) write_ok = false;
        }
        if (!d.content.empty()) {
            json delta;
            delta["content"] = d.content;
            json obj = base();
            obj["choices"] = json::array({choice(delta, json(nullptr))});
            if (!send(obj)) write_ok = false;
        }
    }

    std::string finish = finish_reason(result);
    ParsedOutput parsed;
    if (st.chat) {
        parsed = parse_output(result.text, st.thinking_open, st.tools);
        if (!parsed.tool_calls.empty()) finish = "tool_calls";
    }

    if (st.chat && write_ok && !parsed.tool_calls.empty()) {
        json calls = json::array();
        int index = 0;
        for (const auto& call : parsed.tool_calls) {
            json c = call;
            c["index"] = index++;
            calls.push_back(c);
        }
        json delta;
        delta["tool_calls"] = calls;
        json obj = base();
        obj["choices"] = json::array({choice(delta, json(nullptr))});
        if (!send(obj)) write_ok = false;
    }

    if (write_ok) {
        json delta = json::object();
        json obj = base();
        obj["choices"] = json::array({choice(delta, finish)});
        if (!send(obj)) write_ok = false;
    }

    if (st.include_usage && write_ok) {
        json obj = base();
        obj["choices"] = json::array();
        obj["usage"] = make_usage(result.generated, std::max((int) st.tokens.size(), result.stats.n_prompt_total));
        if (!send(obj)) write_ok = false;
    }

    if (write_ok) {
        const std::string done = "data: [DONE]\n\n";
        sink.write(done.data(), done.size());
    }
    sink.done();

    log_request(st.endpoint.c_str(), (int) st.tokens.size(), result);
}

}  // namespace

int run_server(Engine& engine, const ServerOptions& opt) {
    if (opt.api_key.empty() && !is_loopback_host(opt.host)) {
        KLOG_ERROR("refusing to listen on %s without an API key; set an API key or bind to a loopback host",
                   opt.host.c_str());
        return 1;
    }

    std::mutex generation_mutex;
    httplib::Server svr;

    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        if (req.method == "OPTIONS") {
            apply_cors(req, res);
            res.status = 204;
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key");
            res.set_header("Access-Control-Max-Age", "86400");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (req.path == "/health" || req.path == "/") return httplib::Server::HandlerResponse::Unhandled;
        if (!opt.api_key.empty() && !authorized(req, opt.api_key)) {
            send_error(res, 401, "Incorrect API key provided.");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    svr.set_post_routing_handler([](const httplib::Request& req, httplib::Response& res) { apply_cors(req, res); });

    // The chat page (index.html, compiled in). It calls the API from the same origin.
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(reinterpret_cast<const char*>(klein_index_html), klein_index_html_len, "text/html; charset=utf-8");
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    auto models = [&](const httplib::Request&, httplib::Response& res) {
        json model;
        model["id"] = opt.model_name;
        model["object"] = "model";
        model["owned_by"] = "klein";
        model["context_length"] = engine.n_ctx();
        model["modalities"] = engine.has_vision() ? json::array({"text", "image"}) : json::array({"text"});
        json out;
        out["object"] = "list";
        out["data"] = json::array({model});
        res.set_content(out.dump(), "application/json");
    };
    svr.Get("/v1/models", models);
    svr.Get("/models", models);

    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        if (!parse_json(req, body, res)) return;

        bool stream = false;
        get_bool(body, "stream", stream);
        bool include_usage = false;
        if (body.contains("stream_options") && body["stream_options"].is_object()) {
            get_bool(body["stream_options"], "include_usage", include_usage);
        }

        bool enable_thinking = true;
        std::string effort = "xhigh";
        bool preserve_thinking = true;
        std::string err;
        if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string()) {
            if (!map_effort(body["reasoning_effort"].get<std::string>(), enable_thinking, effort, err)) {
                send_error(res, 400, err);
                return;
            }
        }
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
            const json& kwargs = body["chat_template_kwargs"];
            if (kwargs.contains("reasoning_effort") && kwargs["reasoning_effort"].is_string()) {
                if (!map_effort(kwargs["reasoning_effort"].get<std::string>(), enable_thinking, effort, err)) {
                    send_error(res, 400, err);
                    return;
                }
            }
            if (kwargs.contains("enable_thinking") && kwargs["enable_thinking"].is_boolean()) {
                enable_thinking = kwargs["enable_thinking"].get<bool>();
            }
            if (kwargs.contains("preserve_thinking") && kwargs["preserve_thinking"].is_boolean()) {
                preserve_thinking = kwargs["preserve_thinking"].get<bool>();
            }
        }

        SamplerParams sp;
        if (enable_thinking) {
            sp.temperature = 1.0f;
            sp.top_p = 0.95f;
            sp.top_k = 20;
            sp.min_p = 0.0f;
            sp.presence_penalty = 0.0f;
        } else {
            sp.temperature = 0.7f;
            sp.top_p = 0.8f;
            sp.top_k = 20;
            sp.min_p = 0.0f;
            sp.presence_penalty = 1.5f;
        }
        sp.frequency_penalty = 0.0f;
        sp.repetition_penalty = 1.0f;
        parse_sampling(body, sp);

        std::vector<std::string> stops;
        parse_stop(body, stops);

        json tools = json(nullptr);
        if (body.contains("tools")) tools = body["tools"];
        if (body.contains("tool_choice") && body["tool_choice"].is_string() &&
            body["tool_choice"].get<std::string>() == "none") {
            tools = json::array();
        }

        json messages = json(nullptr);
        if (body.contains("messages")) messages = body["messages"];

        ChatOptions chat_options;
        chat_options.add_generation_prompt = true;
        chat_options.enable_thinking = enable_thinking;
        chat_options.reasoning_effort = effort;
        chat_options.preserve_thinking = preserve_thinking;

        std::string prompt;
        try {
            prompt = render_chat(messages, tools, chat_options);
        } catch (const std::exception& e) {
            send_error(res, 400, e.what());
            return;
        }

        std::vector<ImageRGB> images;
        try {
            images = extract_images(messages);
        } catch (const std::exception& e) {
            send_error(res, 400, e.what());
            return;
        }
        if (!images.empty() && !engine.has_vision()) {
            send_error(res, 400, "this model was started without a vision encoder; restart klein serve with --mmproj FILE");
            return;
        }

        std::vector<int32_t> tokens = engine.tokenizer().encode(prompt, true);
        const int fallback = opt.default_max_tokens >= 0 ? opt.default_max_tokens : engine.n_ctx() - (int) tokens.size();
        const int n_predict = resolve_max_tokens(body, (int) tokens.size(), engine.n_ctx(), fallback, err);
        if (n_predict < 0) {
            send_error(res, 400, err);
            return;
        }

        const std::string id = "chatcmpl-" + random_hex(24);

        if (stream) {
            auto st = std::make_shared<StreamReq>();
            st->engine = &engine;
            st->mutex = &generation_mutex;
            st->opt = opt;
            st->endpoint = "/v1/chat/completions";
            st->id = id;
            st->tokens = std::move(tokens);
            st->n_predict = n_predict;
            st->sp = sp;
            st->stops = std::move(stops);
            st->chat = true;
            st->thinking_open = enable_thinking;
            st->include_usage = include_usage;
            st->tools = tools;
            st->images = std::move(images);
            res.set_chunked_content_provider(
                "text/event-stream", [st](size_t, httplib::DataSink& sink) -> bool {
                    serve_stream(*st, sink);
                    return true;
                });
            return;
        }

        GenResult result;
        try {
            result = run_generation(engine, generation_mutex, tokens, n_predict, sp, stops,
                                    [](const std::string&) { return true; }, images);
        } catch (const std::exception& e) {
            send_error(res, 500, e.what(), "server_error");
            return;
        }
        const ParsedOutput parsed = parse_output(result.text, enable_thinking, tools);

        json message;
        message["role"] = "assistant";
        if (!parsed.tool_calls.empty() && parsed.content.empty()) {
            message["content"] = nullptr;
        } else {
            message["content"] = parsed.content;
        }
        if (!parsed.reasoning.empty()) message["reasoning_content"] = parsed.reasoning;
        if (!parsed.tool_calls.empty()) message["tool_calls"] = parsed.tool_calls;

        std::string finish = finish_reason(result);
        if (!parsed.tool_calls.empty()) finish = "tool_calls";

        json choice;
        choice["index"] = 0;
        choice["message"] = message;
        choice["finish_reason"] = finish;

        json out;
        out["id"] = id;
        out["object"] = "chat.completion";
        out["created"] = (long long) std::time(nullptr);
        out["model"] = opt.model_name;
        out["choices"] = json::array({choice});
        out["usage"] = make_usage(result.generated, std::max((int) tokens.size(), result.stats.n_prompt_total));
        out["timings"] = make_timings(result.stats, result.generated);
        res.set_content(out.dump(), "application/json");

        log_request("/v1/chat/completions", (int) tokens.size(), result);
    });

    svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        if (!parse_json(req, body, res)) return;
        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            send_error(res, 400, "The 'prompt' field is required and must be a string.");
            return;
        }
        const std::string prompt = body["prompt"].get<std::string>();
        bool stream = false;
        get_bool(body, "stream", stream);
        bool include_usage = false;
        if (body.contains("stream_options") && body["stream_options"].is_object()) {
            get_bool(body["stream_options"], "include_usage", include_usage);
        }

        SamplerParams sp;
        sp.temperature = 0.7f;
        sp.top_p = 0.8f;
        sp.top_k = 20;
        sp.min_p = 0.0f;
        sp.presence_penalty = 0.0f;
        sp.frequency_penalty = 0.0f;
        sp.repetition_penalty = 1.0f;
        parse_sampling(body, sp);

        std::vector<std::string> stops;
        parse_stop(body, stops);

        std::vector<int32_t> tokens = engine.tokenizer().encode(prompt, true);
        const int fallback = opt.default_max_tokens >= 0 ? opt.default_max_tokens : 256;
        std::string err;
        const int n_predict = resolve_max_tokens(body, (int) tokens.size(), engine.n_ctx(), fallback, err);
        if (n_predict < 0) {
            send_error(res, 400, err);
            return;
        }

        const std::string id = "cmpl-" + random_hex(24);

        if (stream) {
            auto st = std::make_shared<StreamReq>();
            st->engine = &engine;
            st->mutex = &generation_mutex;
            st->opt = opt;
            st->endpoint = "/v1/completions";
            st->id = id;
            st->tokens = std::move(tokens);
            st->n_predict = n_predict;
            st->sp = sp;
            st->stops = std::move(stops);
            st->chat = false;
            st->include_usage = include_usage;
            res.set_chunked_content_provider(
                "text/event-stream", [st](size_t, httplib::DataSink& sink) -> bool {
                    serve_stream(*st, sink);
                    return true;
                });
            return;
        }

        GenResult result;
        try {
            result = run_generation(engine, generation_mutex, tokens, n_predict, sp, stops,
                                    [](const std::string&) { return true; });
        } catch (const std::exception& e) {
            send_error(res, 500, e.what(), "server_error");
            return;
        }

        json choice;
        choice["index"] = 0;
        choice["text"] = result.text;
        choice["finish_reason"] = finish_reason(result);

        json out;
        out["id"] = id;
        out["object"] = "text_completion";
        out["created"] = (long long) std::time(nullptr);
        out["model"] = opt.model_name;
        out["choices"] = json::array({choice});
        out["usage"] = make_usage(result.generated, std::max((int) tokens.size(), result.stats.n_prompt_total));
        out["timings"] = make_timings(result.stats, result.generated);
        res.set_content(out.dump(), "application/json");

        log_request("/v1/completions", (int) tokens.size(), result);
    });

    svr.Post("/completion", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        if (!parse_json(req, body, res)) return;
        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            send_error(res, 400, "The 'prompt' field is required and must be a string.");
            return;
        }
        const std::string prompt = body["prompt"].get<std::string>();

        SamplerParams sp;
        sp.temperature = 0.7f;
        sp.top_p = 0.8f;
        sp.top_k = 20;
        sp.min_p = 0.0f;
        sp.presence_penalty = 0.0f;
        sp.frequency_penalty = 0.0f;
        sp.repetition_penalty = 1.0f;
        parse_sampling(body, sp);

        std::vector<int32_t> tokens = engine.tokenizer().encode(prompt, true);
        const long long room = (long long) engine.n_ctx() - (long long) tokens.size();
        if (room <= 0) {
            send_error(res, 400, "The prompt is too long: it does not fit in the context window.");
            return;
        }

        long long requested = 256;
        get_integer(body, "n_predict", requested);
        if (requested <= 0) {
            requested = opt.default_max_tokens >= 0 ? opt.default_max_tokens : 256;
        }
        if (requested > room) requested = room;

        const std::vector<std::string> no_stops;
        GenResult result;
        try {
            result = run_generation(engine, generation_mutex, tokens, (int) requested, sp, no_stops,
                                    [](const std::string&) { return true; });
        } catch (const std::exception& e) {
            send_error(res, 500, e.what(), "server_error");
            return;
        }

        json out;
        out["content"] = result.text;
        out["tokens_predicted"] = result.generated;
        out["tokens_evaluated"] = result.stats.n_prompt;
        out["timings"] = make_timings(result.stats, result.generated);
        out["stop"] = true;
        res.set_content(out.dump(), "application/json");

        log_request("/completion", (int) tokens.size(), result);
    });

    KLOG_INFO("listening on http://%s:%d", opt.host.c_str(), opt.port);
    if (!svr.listen(opt.host, opt.port)) {
        KLOG_ERROR("failed to listen on %s:%d", opt.host.c_str(), opt.port);
        return 1;
    }
    return 0;
}

}  // namespace klein
