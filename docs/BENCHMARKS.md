# klein benchmarks

Every number here was measured on the reference machine, with the command that produced it. Higher is better.

| | |
| --- | --- |
| GPU | RTX 3080 Ti 12 GB (sm_86), PCIe 4.0 x16, WDDM driver 591.86 |
| CPU | Ryzen 7 5700X (8 cores / 16 threads, AVX2) |
| RAM | 32 GB DDR4-3200, dual channel |
| OS | Windows 11 Pro 10.0.26100, desktop running (≈1 GB of VRAM in use by Windows and apps) |
| Model | `Qwen3.8-27B-UD-IQ4_XS.gguf` (Unsloth, 13.26 GiB, MTP layer included) |
| Reference engine | llama.cpp release b11435, built from source with CUDA 13.4 for sm_86 |
| klein | commit `cb466b2` (bf16 snapshots) unless noted |

All processes ran at below-normal priority. tok/s = tokens per second.

## Headline: klein vs llama.cpp at the full 262,144-token context

Same two prompts sent to each engine's server (`/completion`, greedy, 256 tokens max, prompt cache off): a code
request and a prose request. Script: `tests/bench/server_bench.py` (two fixed prompts in Qwen chat format, thinking off).

| | llama.cpp | klein | |
| --- | ---: | ---: | ---: |
| Code prompt: decode | 12.28 tok/s | **22.24 tok/s** | 1.81x |
| Code prompt: drafts accepted | 166 / 186 | 178 / 198 | |
| Prose prompt: decode | 8.76 tok/s | **15.98 tok/s** | 1.82x |
| Prose prompt: drafts accepted | 159 / 284 | 160 / 273 | |

llama.cpp was given the best configuration found for this machine (sections below): MTP drafts, KV cache in RAM,
CPU-friendly FFN blocks in RAM, 12 threads:

```
llama-server -m Qwen3.8-27B-UD-IQ4_XS.gguf -ngl 99 -fa on -t 12 -c 262144 -ctk q4_0 -ctv q4_0 -nkvo \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ot "blk\.(13|21|22|23|24|25|26|27|28|36|37|38|39|40|41|42|43|44|48|49|50)\.ffn_(up|gate|down)\.weight=CPU"
klein serve -m Qwen3.8-27B-UD-IQ4_XS.gguf            # defaults: 262144 context, 3 MTP drafts
```

Prompt reading (prefill), 4,096-token prompt, 2,048-token chunks, KV cache in RAM:

| | tok/s |
| --- | ---: |
| llama.cpp (`llama-bench -p 4096 -ub 2048 -nkvo 1 -ctk q4_0 -ctv q4_0 -t 12`, same `-ot`) | 1,004 |
| klein (`klein bench -pp 4096`) | **1,169** |

## klein across prompt lengths (262,144-token context)

`klein bench -m Qwen3.8-27B-UD-IQ4_XS.gguf -pp N -n 256` (synthetic code prompt, greedy):

| Prompt | Prefill | Decode | Tokens / step | Drafts accepted | Verify step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 954 tok/s | 19.30 tok/s | 3.32 | 77% | 160 ms |
| 4,096 | 1,169 tok/s | 21.24 tok/s | 3.61 | 86% | 158 ms |
| 32,768 | 989 tok/s | 18.78 tok/s | 3.41 | 80% | 172 ms |
| 4,096, `--draft 0` (no speculation) | 1,215 tok/s | 7.19 tok/s | 1.00 | - | 140 ms |

Decode speed varies with how predictable the text is (draft acceptance). The verify step grows slowly with context
because attention reads the RAM-resident KV cache over PCIe.

## How each technique moved the numbers

In the order they were built; each row changes one thing (262K context unless noted).

| Change | Before | After |
| --- | ---: | ---: |
| First working klein (planner, MTP; at 262K) vs llama.cpp + MTP (at 8K), code prompt | 12.06 tok/s | 11.90 tok/s |
| Repacked CPU copies of spilled weights (IQ4_XS → IQ4_NL) | 11.90 | 14.27 tok/s |
| Zero-copy KV in RAM (GPU reads over PCIe), 8K ctx, KV in RAM | 13.53 | 15.76 tok/s |
| MTP sliding-window ring cache, 32K prompt | 13.18 | 15.07 tok/s (draft 39 → 4.6 ms/step) |
| Dual-view KV: bulk copy for prefill, 8K ctx, KV in RAM, prefill | 941 | 1,271 tok/s |
| Elastic VRAM + 2,048-token chunks + warm-up, 512-token prompt, prefill | 615 | 954 tok/s |
| 12 CPU threads instead of 8 | 17.54 | 18.42 tok/s |
| bf16 rollback snapshots (22 instead of 25 FFN blocks in RAM) | 18.87 | 20.89 tok/s |

## Where the time goes

Per decode step at the 262K setting (`klein bench -pp 512 -n 256`, printed as `per step (ms)`):

| Phase | ms |
| --- | ---: |
| MTP drafts (2 passes) | 6.6 |
| Verify (main model on 4 tokens) | 160.2 |
| Sampling | 1.1 |
| DeltaNet rollback | 0.5 |
| MTP update | 3.6 |

The verify pass is ~3.4 GiB of spilled FFN weights on the CPU (≈115 ms at ~30 GB/s for a 4-token batch) plus the
GPU's share and per-pass overhead. An Nsight Systems profile of plain decoding (`--draft 0`) at this commit showed
~111,000 individual kernel launches (≈10.5 µs CPU each) and ~8,300 stream syncs over the run: CUDA graphs were
compiled out of klein's ggml build. That is being fixed (`GGML_CUDA_GRAPHS=ON`); results will be added here.

## CPU throughput per quant type

`klein-cpubench -t 8`: one Qwen FFN matrix shape (5120 x 17408), 8 copies so the weights far exceed the 32 MB L3,
GB/s of weight bytes. This table drives the placement planner.

| Type | Layout | n=1 | n=2 | n=3 | n=4 | n=8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| q4_0 | plain | 36.7 | 34.2 | 27.2 | 22.2 | 9.6 |
| q4_0 | repacked | 37.0 | 26.5 | 21.8 | 29.7 | 15.1 |
| q8_0 | plain | 36.7 | 39.1 | 39.5 | 36.0 | 24.3 |
| q2_K | plain | 36.5 | 39.6 | 29.3 | 25.8 | 6.2 |
| q3_K | plain | 35.1 | 30.3 | 21.0 | 16.2 | 6.8 |
| q4_K | plain | 34.8 | 37.3 | 31.0 | 25.5 | 9.6 |
| q4_K | repacked | 37.9 | 26.5 | 23.9 | 34.5 | 18.0 |
| q5_K | plain | 37.4 | 31.1 | 25.9 | 20.7 | 11.0 |
| q6_K | plain | 33.2 | 35.2 | 30.6 | 27.6 | 10.7 |
| iq2_xs | plain | 28.1 | 12.4 | 8.9 | 7.1 | 4.6 |
| iq2_s | plain | 25.6 | 11.1 | 7.2 | 6.2 | 4.2 |
| iq3_xxs | plain | 21.4 | 12.9 | 8.6 | 6.1 | 5.0 |
| iq3_s | plain | 23.1 | 11.0 | 7.9 | 6.1 | 6.1 |
| iq4_xs | plain | 35.8 | 25.9 | 19.0 | 17.8 | 8.7 |
| iq4_nl | plain | 32.6 | 31.6 | 25.0 | 19.0 | 9.9 |
| iq4_nl | repacked | 37.7 | 27.6 | 23.2 | 31.6 | 16.7 |

Repacked kernels shine at batches of exactly 4 (they process 4 columns at a time), which is why klein drafts 3
tokens: `--draft 2` measured 219.6 ms per verify step (3 tokens) against 177.6 ms for `--draft 3` (4 tokens), and
`--draft 4` 253.2 ms.

Whole-model CPU streaming rate: `llama-bench -ngl 0 -t 8 -n 16` = 2.39 tok/s, i.e. ≈33 GB/s end to end including
the slow IQ2/IQ3 tensors.

## llama.cpp baselines (how its best configuration was found)

`llama-bench`, 8 threads, short context, decode of 64 tokens:

| Configuration | Prefill pp512 | Decode |
| --- | ---: | ---: |
| FFN of layers 0-16 on the CPU (2.3 GiB; many IQ2/IQ3 tensors) | 981 tok/s | 8.82 tok/s |
| 48 CPU-friendliest FFN tensors on the CPU (2.3 GiB) | 922 tok/s | 9.81 tok/s |
| 18 CPU-friendliest whole FFN blocks on the CPU (2.36 GiB) | - | 9.86 tok/s |
| Threads 4 / 8 / 16, layers 0-16 | - | 7.14 / 8.39 / 8.91 tok/s |

`llama-server` with `--spec-type draft-mtp --spec-draft-n-max 3`, 8K context, 2.76 GiB of FFN on the CPU:
12.06 tok/s (code prompt, 180/192 drafts accepted), 8.21 tok/s (prose).

## Accuracy

| Check | Result |
| --- | --- |
| KL divergence vs llama.cpp logits (4 x 512 tokens of novel text, f16 KV) | mean 0.00187, same top-1 98.33% |
| llama.cpp vs itself, batch 64 instead of 512 (noise floor) | mean 0.00153, same top-1 98.33% |
| Perplexity on the same tokens | klein 3.3340, llama.cpp 3.3309 |
| `check-spec`, 300 greedy tokens, f32 snapshots | 1 near-tie (margin 0.027) |
| `check-spec`, 300 greedy tokens, bf16 snapshots | 1 near-tie (margin 0.067) |

Commands:

```
llama-perplexity -m MODEL -f novel.txt -c 512 --chunks 4 -ngl 99 -fa on --kl-divergence-base base.bin
klein ppl -m MODEL --kld-base base.bin --kv f16 --ppl-batch 512
klein check-spec -m MODEL -c 8192 -n 300 -p "Explain how a hash table handles collisions, with a short C example."
```

(`novel.txt` = klein's own docs and sources, text the model has never seen. Note that llama.cpp reads `-f` files in
text mode on Windows, converting CRLF to LF; compare on the token ids stored in the base file, as `--kld-base` does.)
