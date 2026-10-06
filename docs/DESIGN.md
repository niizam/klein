# klein design

klein is an inference engine for **Qwen3.8-27B** (GGUF architecture `qwen35`) on one consumer NVIDIA GPU plus
system RAM. The reference machine is also the largest one klein targets:

| Part | Reference machine |
| --- | --- |
| GPU | RTX 3080 Ti, 12 GiB GDDR6X (~912 GB/s), sm_86, PCIe 4.0 x16 |
| CPU | Ryzen 7 5700X, 8 cores / 16 threads, AVX2 (no AVX-512) |
| RAM | 32 GiB DDR4-3200, dual channel (51.2 GB/s theoretical) |
| OS | Windows 11, WDDM driver 591.86 (CUDA 13.1 runtime level), CUDA toolkit 13.4 |

Goals, in priority order:

1. Fastest prompt reading (prefill) and answer writing (decode) on the reference machine, with CUDA.
2. The model's native 262,144-token context must work within 12 GiB VRAM + 32 GiB RAM.
3. The smaller the machine needed, the better. Nothing may require more than the reference machine.

Measured results are in [BENCHMARKS.md](BENCHMARKS.md).

## The model

From `config.json` and the GGUF metadata:

| | |
| --- | --- |
| Layers | 64 decoder layers + 1 MTP (multi-token prediction) layer |
| Layer pattern | 3 Gated DeltaNet (linear attention) layers, then 1 full-attention layer, repeated 16 times |
| Hidden size | 5,120; FFN 17,408 (SwiGLU) |
| Full attention | 24 query heads, 4 KV heads, head dim 256, sigmoid output gate, Q/K RMS norm, interleaved M-RoPE (64 of 256 dims, theta 1e7) |
| Gated DeltaNet | 16 key heads, 48 value heads, head dim 128, causal conv1d (kernel 4) |
| Vocabulary | 248,320 tokens; untied input embedding and LM head |
| Context | 262,144 native |

Only the 16 full-attention layers keep a KV cache. The 48 DeltaNet layers keep a fixed-size recurrent state.

| Component (IQ4_XS GGUF) | Size | Placement |
| --- | ---: | --- |
| `token_embd` (Q3_K) | 0.51 GiB | RAM: only one row per token is read |
| LM head `output` (Q5_K) | 0.81 GiB | VRAM: read by every target and every draft step |
| MTP layer | 0.36 GiB | VRAM |
| Decoder trunk | 11.58 GiB | VRAM, except whole FFN blocks the planner spills (3.4-3.8 GiB) |
| Recurrent state | 144 MiB f32 + 3 bf16 rollback snapshots (216 MiB) | VRAM |
| KV cache | 18 KiB/token at q4_0 (4.5 GiB at 262,144) | RAM when large (pinned, GPU-mapped) |

About 10.8 GiB of VRAM is free on the reference machine with the Windows desktop running, so at least ~2.5 GiB of
weights must live in RAM even before any KV cache.

## The performance model

Decoding is memory-bound: every weight byte is read once per forward pass. Bytes in VRAM are read at ~700 GB/s.
Bytes in RAM are read by the CPU at ~33-37 GB/s (measured end to end), and reading them over PCIe is no faster
(~24 GB/s) and draws on the same DRAM. So:

```
time per step ~ CPU time for the spilled weights + GPU time + fixed overhead (launches, syncs)
tokens/s      ~ tokens accepted per step / time per step
```

klein's techniques each attack one term.

## Techniques

### 1. Placement planner (`src/planner.cpp`)

The unit that moves to RAM is one decoder layer's FFN (gate, up, down). One unit means one GPU->CPU->GPU round trip
per forward pass, and FFNs are 75% of the trunk. Units are ranked by **CPU milliseconds per byte freed**, from
measured per-type CPU throughput (`klein-cpubench`, weights much larger than L3, in the layout klein uses on the
CPU). IQ2/IQ3 tensors cost 4-5x more CPU time per byte than K-quants or IQ4 at a 4-token verify batch, so the
planner never spills them while cheaper bytes exist. The planner fills VRAM with:

1. everything except `token_embd` and the spilled FFN blocks,
2. the recurrent state, the MTP ring cache, hidden-state buffers,
3. a decode compute reserve (192 MiB) and a margin (256 MiB),
4. the KV cache, only when it is small (<= 768 MiB) or everything fits. A KV byte in VRAM pushes out a weight
   byte, and weight bytes are read every step while KV bytes are read only up to the current context.

### 2. CPU copies of spilled weights (`Model::make_cpu_copies`, `src/transcode.cpp`)

ggml's repacked AVX2 kernels (Q4_0, Q4_K, IQ4_NL, ...) keep a 4-token batch near DRAM speed. The plain kernels for
IQ4_XS, the dominant type in Unsloth's quant, drop to half that. IQ4_XS and IQ4_NL share the codebook and the
nibble layout, so klein transcodes spilled IQ4_XS blocks to IQ4_NL (each 32-value scale `d*(ls-32)` rounded to
fp16, +6% bytes) and repacks them. The original stays in pinned RAM and is streamed to the GPU for prefill
batches (>= 32 tokens). The repacked copy serves decode and verify batches.

| Type (CPU, 8 threads) | n=1 | n=4 (verify) |
| --- | ---: | ---: |
| IQ4_XS plain | 35.8 GB/s | 17.8 GB/s |
| IQ4_XS -> IQ4_NL repacked | 35.6 | 29.8 |
| Q4_K repacked | 37.9 | 34.5 |
| IQ3_S plain | 23.1 | 6.1 |

Repacked kernels process 4 columns at a time, so verify batches of exactly 4 tokens (3 drafts) are the efficient
size: 2 drafts measured slower per step than 3.

### 3. MTP speculative decoding (`Engine::generate`)

Qwen3.8 ships one MTP layer. Each step:

1. **Draft.** The pending MTP logits give d1. The MTP head is then chained on its own hidden state for d2 and d3.
2. **Verify.** The main model runs on `[next, d1, d2, d3]` in one pass (the spilled weights are read once for 4
   tokens).
3. **Accept.** klein samples the target at each position and accepts drafts while they match. This is exact for
   any sampler: every emitted token is a target sample given the correct prefix.
4. **Roll back.** KV entries past the accepted prefix are masked out; the DeltaNet state is restored from a
   snapshot.
5. **MTP update.** The MTP head runs over the accepted positions with the main model's true hidden states, which
   rewrites its KV entries and yields the next d1.

Details that matter:
- **Snapshots.** ggml's fused `gated_delta_net` op emits the state after each of the last K tokens. klein keeps the
  current state in f32 and the older ones in **bf16** (half the VRAM). Rollback copies a snapshot back, about 0.5
  ms.
- **MTP ring cache.** The MTP head attends over a 16,384-position **sliding window** stored as a ring buffer in
  VRAM. Draft cost no longer grows with the context (39 -> 4.6 ms per step at 32K). Drafts only need to be
  plausible, since the main model verifies every token.
- **Correctness.** `klein check-spec` generates greedily with drafts, then replays without speculation. Over 300
  tokens, 1 position differed, with a logit margin of 0.07 (a numerical near-tie between batch shapes).

### 4. KV cache in RAM, read by the GPU (`ggml-cuda` mapped buffer, `State`)

A 262K cache does not fit next to the weights, so it lives in pinned RAM. klein adds a **mapped host buffer type**
to ggml-cuda (pinned memory under unified addressing; see [THIRD_PARTY.md](THIRD_PARTY.md)). The same memory has
two views:

- **Decode/verify** (< 32 tokens): the GPU-mapped view. Attention runs on the GPU and reads only the cells in use
  over PCIe. This replaces 16 GPU->CPU->GPU round trips per pass with CPU attention.
  - Verify step at 8K context: 197 -> 168 ms.
- **Prefill** (>= 32 tokens): a plain host view. The scheduler writes new K/V on the CPU and copies the used range
  to VRAM in bulk before attention. Flash attention re-reads K/V for every query tile, which is slow when each read
  crosses PCIe.
  - Prefill at 8K context: 941 -> 1271 tok/s.

### 5. Context-elastic VRAM (`Engine::ensure_prefill_vram`, `relax_after_prefill`, `Model::demote/promote`)

VRAM is planned for decoding. Prefilling a long prompt needs scratch memory that grows with the context, mostly
the f16 copy of the quantized K/V that flash attention makes (1 GiB at 262K). Instead of reserving that
permanently:

- Every FFN block has its own buffers. Blocks behind the spill frontier also keep a pinned RAM copy and a repacked
  copy, prepared at load time.
- Before a prefill, klein demotes as many blocks as the prefill needs: a pointer swap plus freeing their VRAM.
- After the prefill, it drops the prefill-sized compute buffers and promotes blocks back with a host-to-device
  copy.

Short chats get the decode-optimal placement, and a 262K prompt still fits. The same mechanism lets the default
prefill chunk be 2,048 tokens: its activations only take VRAM while the prompt is read.

### 6. Prompt reuse

When a request extends the cached token sequence (the next turn of a conversation), only the new suffix is
prefilled. The DeltaNet state cannot be rewound to an arbitrary earlier position, so reuse requires the cache to be
an exact prefix of the new prompt.

### 7. CUDA graphs

klein builds ggml with `GGML_CUDA_GRAPHS=ON` (ggml leaves it off outside llama.cpp) and gives every graph kind
(main/MTP x small/large batch) its own metadata buffer. ggml-cuda caches CUDA graphs by the address of each split's
first node, so kinds built in the same memory would keep invalidating each other's cached graphs. Without graphs,
one token cost ~1,700 individual kernel launches at ~10 µs of CPU time each.

## Correctness checks

- **Tokenizer.** Identical token ids to llama.cpp on code, CJK, emoji, numbers, special tokens and the full GPL
  text (6,887 tokens); decoding round-trips.
- **Chat template.** The C++ renderer matches the GGUF's Jinja template byte-for-byte on 44 cases (tools, thinking
  modes, multi-turn tool calls, errors).
- **Logits.** KL divergence vs llama.cpp on the same tokens is 0.00187, with 98.33% top-1 agreement.
  llama.cpp against itself with a different batch size gives 0.00153 and 98.33%, so klein is within numerical
  noise of the reference.
- **Speculative decoding.** `check-spec` (above).
- **Server.** `tests/server/smoke.py`: chat, streaming with reasoning, typed tool calls, prompt reuse,
  `/completion`.

## Code map

| File | |
| --- | --- |
| `src/gguf_file.*` | GGUF metadata and tensor reads |
| `src/model.*` | hyperparameters, tensor binding, FFN blocks (load, demote, promote), repacked CPU copies |
| `src/planner.*` | VRAM budget, KV placement, FFN spill ranking (measured CPU rates) |
| `src/state.*` | KV caches (VRAM or mapped RAM with a bulk view), MTP ring cache, DeltaNet state and snapshots |
| `src/graph.*` | the qwen35 graph and the MTP head as ggml graphs |
| `src/engine.*` | backends, scheduler, prefill, speculative decode loop, elastic VRAM, warm-up |
| `src/tokenizer.*`, `src/chat.*`, `src/sampler.*` | BPE tokenizer, Qwen3.8 chat template + output parser, sampler |
| `src/server.*` | OpenAI-compatible HTTP server |
| `src/transcode.*` | IQ4_XS -> IQ4_NL |
| `tools/` | `klein-tokenize`, `klein-cpubench`, `klein-chat-test` |

## Ideas not yet done

- Splitting each spilled FFN by rows between GPU and CPU so both work at once. The gain is bounded by the GPU time
  of those layers (~10% of a step).
- A smaller draft LM head: drafting with only the most frequent tokens' rows of `output`, cutting the 0.81 GiB read
  per draft.
- Chunked attention with log-sum-exp merging, so long-context prefill needs no full-length f16 K/V scratch.
