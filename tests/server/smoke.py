#!/usr/bin/env python3
"""End-to-end smoke test of `klein serve` (OpenAI-compatible API).

    python tests/server/smoke.py http://127.0.0.1:8080

Checks /health, /v1/models, a non-thinking chat, a streamed thinking chat (reasoning_content deltas), a tool
call, prefix reuse on a follow-up turn, and /completion. Exits non-zero on the first failure.
"""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8080"


def req(method, path, body=None, stream=False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(r, timeout=1800)
    if stream:
        return resp
    return json.loads(resp.read())


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        sys.exit(1)


h = req("GET", "/health")
check(h.get("status") == "ok", "/health")
m = req("GET", "/v1/models")
check(m["data"] and m["data"][0]["id"], "/v1/models: " + m["data"][0]["id"])

# 1. non-thinking chat
t0 = time.time()
r = req("POST", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    "reasoning_effort": "none", "max_tokens": 16, "temperature": 0})
msg = r["choices"][0]["message"]
check("Paris" in (msg.get("content") or ""), "chat (no thinking): %r in %.1fs" % (msg.get("content"), time.time() - t0))
check(r["usage"]["prompt_tokens"] > 0 and r["usage"]["completion_tokens"] > 0, "usage: %s" % r["usage"])
print("     timings:", r.get("timings"))

# 2. streamed thinking chat
resp = req("POST", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "Is 91 a prime number? Think briefly, then answer yes or no."}],
    "reasoning_effort": "low", "max_tokens": 600, "stream": True, "stream_options": {"include_usage": True},
    "temperature": 0}, stream=True)
reasoning, content, finish, usage, done = "", "", None, None, False
for raw in resp:
    line = raw.decode("utf-8").strip()
    if not line.startswith("data: "):
        continue
    payload = line[6:]
    if payload == "[DONE]":
        done = True
        break
    ev = json.loads(payload)
    if ev.get("usage"):
        usage = ev["usage"]
    for ch in ev.get("choices", []):
        d = ch.get("delta", {})
        reasoning += d.get("reasoning_content") or ""
        content += d.get("content") or ""
        finish = ch.get("finish_reason") or finish
check(done, "stream ended with [DONE]")
check(len(reasoning) > 0, "stream: reasoning_content deltas (%d chars)" % len(reasoning))
check("no" in content.lower(), "stream: answer %r (finish=%s)" % (content.strip()[:80], finish))
check(usage is not None, "stream: usage chunk %s" % usage)

# 3. tool call
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "days": {"type": "integer", "description": "Forecast days"}}, "required": ["city"]}}}]
r = req("POST", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "What's the weather in Tokyo for the next 3 days? Use the tool."}],
    "tools": tools, "reasoning_effort": "none", "max_tokens": 200, "temperature": 0})
msg = r["choices"][0]["message"]
calls = msg.get("tool_calls") or []
check(len(calls) == 1 and calls[0]["function"]["name"] == "get_weather", "tool call: %s" % calls)
args = json.loads(calls[0]["function"]["arguments"])
check(args.get("city", "").lower().startswith("tokyo") and args.get("days") == 3, "tool args typed: %s" % args)
check(r["choices"][0]["finish_reason"] == "tool_calls", "finish_reason tool_calls")

# 4. follow-up turn reuses the cached prefix (only the new suffix is prefilled)
conv = [{"role": "user", "content": "Write a haiku about the sea."}]
r1 = req("POST", "/v1/chat/completions", {"messages": conv, "reasoning_effort": "none", "max_tokens": 60, "temperature": 0})
conv.append({"role": "assistant", "content": r1["choices"][0]["message"]["content"]})
conv.append({"role": "user", "content": "Now one about mountains."})
r2 = req("POST", "/v1/chat/completions", {"messages": conv, "reasoning_effort": "none", "max_tokens": 60, "temperature": 0})
pn, pt = r2["timings"]["prompt_n"], r2["usage"]["prompt_tokens"]
check(pn < pt, "follow-up prefilled %d of %d prompt tokens (prefix reused)" % (pn, pt))

# 5. llama.cpp-style /completion
r = req("POST", "/completion", {"prompt": "The quick brown fox", "n_predict": 8, "temperature": 0})
check(r.get("tokens_predicted", 0) > 0 and "timings" in r, "/completion: %r" % r.get("content"))
print("all server checks passed")
