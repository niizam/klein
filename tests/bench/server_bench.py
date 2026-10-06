#!/usr/bin/env python3
# Usage: python tests/bench/server_bench.py http://127.0.0.1:8080  (klein serve or llama-server)
# Sends fixed greedy requests to an OpenAI-compatible /completion server and prints llama.cpp-style timings.
import json, sys, time, urllib.request

url = sys.argv[1]
prompts = [
    "<|im_start|>user\nWrite a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two doctests.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|im_start|>user\nExplain in about 200 words how a CPU cache hierarchy works.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
]
for p in prompts:
    body = json.dumps({"prompt": p, "n_predict": 256, "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(url + "/completion", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    t = r.get("timings", {})
    print(f"wall {time.time()-t0:.1f}s  prompt {t.get('prompt_n')} tok @ {t.get('prompt_per_second',0):.1f} t/s  "
          f"gen {t.get('predicted_n')} tok @ {t.get('predicted_per_second',0):.2f} t/s  "
          f"draft {t.get('draft_n')} accepted {t.get('draft_n_accepted')}")
    print("   ", repr(r.get("content", "")[:120]))
