#!/usr/bin/env python3
"""Sends the same greedy image requests to an OpenAI-compatible server and saves the answers as JSON.

    python tests/bench/vision_compare.py http://127.0.0.1:8080 out.json image1.jpg [image2.png ...]

Run it against `klein serve --mmproj ...` and `llama-server --mmproj ...` and compare the outputs: with greedy
sampling the texts should agree (up to numerical near-ties between the two engines).
"""
import base64
import json
import mimetypes
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
base, out_path, images = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3:]
prompts = ["Describe this image in two sentences.", "List every piece of text you can read in this image."]
results = []
for path in images:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    data = base64.b64encode(open(path, "rb").read()).decode()
    for prompt in prompts:
        body = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                {"type": "text", "text": prompt}]}],
            "temperature": 0, "top_k": 1, "max_tokens": 160,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        text = r["choices"][0]["message"]["content"]
        results.append({"image": path, "prompt": prompt, "text": text, "seconds": round(time.time() - t0, 2),
                        "prompt_tokens": r.get("usage", {}).get("prompt_tokens")})
        print(f"--- {path} | {prompt} ({results[-1]['seconds']} s, {results[-1]['prompt_tokens']} prompt tokens)\n{text}\n")
json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
