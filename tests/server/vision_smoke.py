#!/usr/bin/env python3
"""Image checks for `klein serve --mmproj ...`.

    python tests/server/vision_smoke.py http://127.0.0.1:8080 image.jpg

Checks that /v1/models advertises image input, that an image question is answered with usage counting the image
tokens, that a follow-up turn reuses the cached image (only the new text is prefilled), and that a remote image URL
is refused with a 400.
"""
import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
BASE, IMG = sys.argv[1].rstrip("/"), sys.argv[2]


def post(body):
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        sys.exit(1)


m = json.loads(urllib.request.urlopen(BASE + "/v1/models").read())["data"][0]
check("image" in m.get("modalities", []), "/v1/models advertises image input: %s" % m.get("modalities"))

mime = mimetypes.guess_type(IMG)[0] or "image/jpeg"
url = "data:%s;base64,%s" % (mime, base64.b64encode(open(IMG, "rb").read()).decode())
conv = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                     {"type": "text", "text": "What is the main headline in this image?"}]}]
r1 = post({"messages": conv, "reasoning_effort": "none", "max_tokens": 60, "temperature": 0})
a1 = r1["choices"][0]["message"]["content"]
check(len(a1) > 0, "image answer: %r" % a1[:100])
check(r1["usage"]["prompt_tokens"] > 100, "usage counts image tokens: %d prompt tokens" % r1["usage"]["prompt_tokens"])

conv += [{"role": "assistant", "content": a1}, {"role": "user", "content": "And what is the date on it?"}]
r2 = post({"messages": conv, "reasoning_effort": "none", "max_tokens": 40, "temperature": 0})
pn, pt = r2["timings"]["prompt_n"], r2["usage"]["prompt_tokens"]
check(pn < 60 and pt > 100, "follow-up reuses the image: prefilled %d of %d prompt tokens" % (pn, pt))
print("     answer:", r2["choices"][0]["message"]["content"][:100])

try:
    post({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                                                     {"type": "text", "text": "hi"}]}], "max_tokens": 5})
    check(False, "remote image URL refused")
except urllib.error.HTTPError as e:
    check(e.code == 400, "remote image URL refused with %d" % e.code)
print("all vision checks passed")
