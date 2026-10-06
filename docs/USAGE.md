# Using klein

`klein.exe <command> -m MODEL.gguf [options]`. `klein.exe` without arguments prints every option.

The model is any **Qwen3.5 / 3.6 / 3.8 dense** GGUF (architecture `qwen35`), for example Unsloth's
`Qwen3.8-27B-UD-IQ4_XS.gguf`. If the GGUF contains the MTP layer (`blk.64.nextn.*`; Unsloth's do), klein uses
it for speculative decoding. `--mtp FILE` takes the MTP block from another GGUF instead (e.g. a smaller one).

## Chat in the terminal

```powershell
klein run -m Qwen3.8-27B-UD-IQ4_XS.gguf -p "Write a Python function that reverses a linked list."
klein run -m Qwen3.8-27B-UD-IQ4_XS.gguf -p "Prove that sqrt(2) is irrational." --think -n 2000 --temp 1.0 --top-p 0.95
```

`run` uses the Qwen chat format with thinking off by default (`--think` turns it on) and greedy sampling unless
`--temp` is given. Speed statistics go to stderr.

## Images

With the vision encoder (`mmproj-BF16.gguf` from Unsloth's Qwen3.8-27B repository), klein reads images:

```powershell
klein run -m Qwen3.8-27B-UD-IQ4_XS.gguf --mmproj mmproj-BF16.gguf --image photo.jpg -p "What is in this picture?"
klein serve -m Qwen3.8-27B-UD-IQ4_XS.gguf --mmproj mmproj-BF16.gguf
```

- `--image FILE` can be given several times. PNG, JPEG, BMP, GIF (first frame), TGA, PSD, HDR and PNM are read.
- Each image becomes one token per 32x32 pixels after resizing, between 8 and `--image-max-tokens` (default 1024,
  about one megapixel). A 640x480 photo is 300 tokens.
- The encoder's weights (0.87 GiB) stay in RAM. For each image klein borrows VRAM from a few FFN blocks, runs the
  encoder on the GPU (about 0.1-0.4 s), and gives the VRAM back, so text-only chats lose no speed.
- Server: send images as OpenAI `image_url` content items with `data:` URLs (base64). Remote `http(s)` URLs are
  refused: klein does not fetch anything. `/v1/models` lists `"modalities": ["text", "image"]`.
- Chat page: the image button, paste or drag-and-drop attaches images; they are downscaled in the browser first.

## Server (OpenAI-compatible)

```powershell
klein serve -m Qwen3.8-27B-UD-IQ4_XS.gguf                    # http://127.0.0.1:8080
klein serve -m Qwen3.8-27B-UD-IQ4_XS.gguf --host 0.0.0.0 --api-key SECRET --port 8080
```

Open **http://127.0.0.1:8080/** in a browser for the built-in chat page (`src/web/index.html`, compiled into
`klein.exe`): streaming answers, a thinking switch (Off/Low/Medium/High) with the reasoning folded away, code blocks
with a copy button, tables, and the speed of each answer. Conversations are kept in the browser. With
`--api-key KEY`, open `http://HOST:PORT/?key=KEY` once; the page remembers the key.

A non-loopback `--host` is refused without `--api-key`. With a key, every endpoint except `/health` needs
`Authorization: Bearer <key>` (or `x-api-key: <key>`).

| Endpoint | |
| --- | --- |
| `POST /v1/chat/completions` | messages, tools, streaming (SSE), `reasoning_content`, tool calls |
| `POST /v1/completions` | raw prompt, no template |
| `POST /completion` | llama.cpp-style (`prompt`, `n_predict`), returns `timings` |
| `GET /v1/models`, `GET /models` | the loaded model and its context length |
| `GET /` | the chat page |
| `GET /health` | `{"status":"ok"}` |

### Chat completions

- **Thinking.** On by default, as the model's template does. Turn it off with `"reasoning_effort": "none"`, or
  choose `"low"`, `"medium"` or `"high"` (= the template's `xhigh`). The template's own switches also work:
  `"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low", "preserve_thinking": true}`.
  Reasoning text comes back as `message.reasoning_content` (streaming: `delta.reasoning_content`).
- **Sampling defaults** follow Qwen's recommendations and can be overridden per request (`temperature`, `top_p`,
  `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `seed`):
  - thinking: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0
  - no thinking: temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5
- **Tools.** OpenAI `tools` are rendered into the model's format. Calls come back as `message.tool_calls`, with
  `arguments` typed from the tool's JSON schema (an `integer` parameter arrives as a number). `finish_reason` is
  `tool_calls`.
- **Stop strings.** `stop` (a string or an array) ends generation and is cut from the output.
- **Limits.** `max_tokens` / `max_completion_tokens`. Without one, generation runs until the end-of-turn token or
  the context is full.
- **Prompt reuse.** When a request extends the previous one (the next turn of the same conversation), only the new
  tokens are prefilled. `timings.prompt_n` shows how many were.
- **One request at a time.** The engine serves requests in order; others wait.

Every response also carries llama.cpp-style `timings` (`prompt_per_second`, `predicted_per_second`, `draft_n`,
`draft_n_accepted`).

### Connecting apps

Any OpenAI-compatible client: base URL `http://127.0.0.1:8080/v1`, any model name, any API key (or the one you set).

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": "Hi!"}],
                                   extra_body={"reasoning_effort": "none"})
print(r.choices[0].message.content)
```

## Benchmarks and checks

```powershell
klein bench -m MODEL.gguf -pp 4096 -n 256        # prefill a 4096-token prompt, generate 256 tokens
klein ppl -m MODEL.gguf -f text.txt --ppl-ctx 512 --chunks 8
klein ppl -m MODEL.gguf --kld-base base.bin       # KL divergence vs llama-perplexity --kl-divergence-base
klein check-spec -m MODEL.gguf -c 8192 -n 300 -p "..."
```

`bench` and `run` print the time per decode-step phase (draft, verify, sample, rollback, MTP update).

## Memory and the context length

The default context is the model's native **262,144 tokens**. At start-up klein measures the free VRAM and plans
where everything goes (the log's `plan:` line):

- **KV cache.** In RAM when it is large (4.5 GiB at 262K with the q4_0 default). Attention still runs on the GPU:
  decoding reads the cache directly over PCIe, and prefill copies the used range to VRAM in bulk.
- **FFN weights.** Whole decoder-layer FFN blocks that do not fit in VRAM stay in RAM, and the CPU computes them.
  klein picks the blocks whose quant types the CPU handles fastest, and keeps a repacked copy for the AVX2
  kernels.
- **Long prompts.** VRAM is planned for decoding. A long prompt temporarily moves a few more FFN blocks to RAM to
  make room for the attention scratch, and they move back once the prompt is read.

RAM use at 262K with the IQ4_XS model is about 4.5 GiB of KV cache + 3.5–5 GiB of spilled weights and their
repacked copies + the OS file cache. `-c` lowers the context and the KV cache with it. `--no-repack` saves RAM at
the cost of decode speed.

Context-length options:

| Context | KV cache (q4_0 / q8_0) | Notes |
| --- | --- | --- |
| 8,192 | 0.14 / 0.27 GiB | KV cache stays in VRAM |
| 32,768 | 0.56 / 1.06 GiB | q8_0 by default |
| 131,072 | 2.25 / 4.25 GiB | q8_0 by default |
| 262,144 | 4.50 / 8.50 GiB | q4_0 by default |
