// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "common.h"

#include <mutex>
#include <vector>

namespace klein {

static LogLevel g_level = LogLevel::Info;
static std::mutex g_log_mutex;

void set_log_level(LogLevel level) { g_level = level; }
LogLevel log_level() { return g_level; }

void log_msg(LogLevel level, const char* fmt, ...) {
    if (level < g_level) return;
    static const char* tags[] = {"debug", "info", "warn", "error"};
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::fprintf(stderr, "klein %s: ", tags[(int) level]);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

void fatal(const char* fmt, ...) {
    std::fprintf(stderr, "klein fatal: ");
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::fflush(stderr);
    std::exit(1);
}

std::string format(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::vector<char> buf((size_t) n + 1);
    std::vsnprintf(buf.data(), buf.size(), fmt, ap2);
    va_end(ap2);
    return std::string(buf.data(), (size_t) n);
}

}  // namespace klein
