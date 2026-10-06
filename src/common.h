// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace klein {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

void set_log_level(LogLevel level);
LogLevel log_level();
void log_msg(LogLevel level, const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#define KLOG_DEBUG(...) ::klein::log_msg(::klein::LogLevel::Debug, __VA_ARGS__)
#define KLOG_INFO(...)  ::klein::log_msg(::klein::LogLevel::Info,  __VA_ARGS__)
#define KLOG_WARN(...)  ::klein::log_msg(::klein::LogLevel::Warn,  __VA_ARGS__)
#define KLOG_ERROR(...) ::klein::log_msg(::klein::LogLevel::Error, __VA_ARGS__)

[[noreturn]] void fatal(const char* fmt, ...);

#define KLEIN_ASSERT(cond)                                                              \
    do {                                                                                \
        if (!(cond)) ::klein::fatal("assertion failed: %s (%s:%d)", #cond, __FILE__, __LINE__); \
    } while (0)

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
constexpr double MiB = 1024.0 * 1024.0;

std::string format(const char* fmt, ...);

}  // namespace klein
