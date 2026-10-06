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
GPU's share. Nsight Systems profiles of plain decoding (`--draft 0`, 64-token prompt, 64 tokens):

| | CUDA graphs off (`cb466b2`) | CUDA graphs on (`ae44df4`) |
| --- | ---: | ---: |
| `cudaLaunchKernel` calls / CPU time | 111,411 / 1,172 ms | 10,059 / 75 ms |
| `cudaGraphLaunch` calls / CPU time | 0 | 1,300 / 160 ms |
| `cudaStreamSynchronize` calls / time | 8,320 / 1,928 ms | 8,339 / 2,227 ms |

Launches are asynchronous, so their CPU time mostly overlapped GPU work: CUDA graphs cut it by ~80% but decode
only by 3-4%. What remains is waiting, ~65 syncs per token, one at each GPU->CPU hand-off around a spilled FFN
block, i.e. the CPU computing the spilled weights. Decode is bound by reading those weights from RAM.

Same matrix as above with CUDA graphs on (`klein bench`, 262K context):

| Prompt | Prefill | Decode | Verify step | before (graphs off) |
| ---: | ---: | ---: | ---: | --- |
| 512 | 960 tok/s | 19.66 tok/s | 155.8 ms | 19.30 tok/s, 160.2 ms |
| 4,096 | 1,156 tok/s | 20.52 tok/s | 164.7 ms | 21.24 tok/s, 157.6 ms (acceptance 86% vs 88%) |
| 32,768 | 984 tok/s | 18.98 tok/s | 169.3 ms | 18.78 tok/s, 172.3 ms |
| 4,096, `--draft 0` | 1,210 tok/s | 7.49 tok/s | 134.3 ms | 7.19 tok/s, 139.9 ms |

`check-spec` with CUDA graphs: 2 near-ties in 300 tokens (worst margin 0.024).

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

## Quantization: speed vs quality

Unsloth's `UD-IQ3_XXS` (10.18 GiB) fits in VRAM almost entirely (only `token_embd`, which is just looked up, stays
in RAM), so decoding no longer waits on the CPU. `klein bench -m Qwen3.8-27B-UD-IQ3_XXS.gguf`, 262K context:

| Prompt | Prefill | Decode | Verify step | IQ4_XS decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1,049 tok/s | 49.96 tok/s | 61.5 ms | 19.66 tok/s |
| 4,096 | 1,123 tok/s | 48.80 tok/s | 61.1 ms | 20.52 tok/s |
| 32,768 | 930 tok/s | 30.51 tok/s | 92.6 ms | 18.98 tok/s |

At 32K the KV cache (in RAM, read over PCIe) becomes the main cost of a step.

Quality against the most precise quant available, `UD-Q4_K_S` (14.30 GiB), on 4,080 tokens of text the model has
never seen (klein's own docs and sources, 16 chunks of 512):

| Quant | Size | Mean KLD vs Q4_K_S | Same top token | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| UD-Q4_K_S (reference) | 14.30 GiB | - | - | 5.562 |
| UD-IQ4_XS (default) | 13.27 GiB | 0.0172 | 92.60% | 5.578 (+0.3%) |
| UD-IQ3_XXS | 10.18 GiB | 0.0800 | 85.64% | 5.773 (+3.8%) |

```
llama-perplexity -m Qwen3.8-27B-UD-Q4_K_S.gguf -f novel_big.txt -c 512 --chunks 16 -ngl 99 -fa on -t 12 \
  -ot "<33 FFN blocks>=CPU" --kl-divergence-base kld_q4ks.bin
klein ppl -m Qwen3.8-27B-UD-IQ4_XS.gguf  --kld-base kld_q4ks.bin --ppl-batch 512 --kv f16
klein ppl -m Qwen3.8-27B-UD-IQ3_XXS.gguf --kld-base kld_q4ks.bin --ppl-batch 512 --kv f16
```

IQ3_XXS writes answers ~2.4x faster at a clearly measurable quality cost (4.6x the divergence of IQ4_XS); which
one to run is the user's call (`-m`).

## Rollback snapshot precision

`check-spec` (300 greedy tokens, IQ4_XS, 8K context), positions that differ from plain decoding:

| Snapshots | Differing positions | Worst logit margin |
| --- | ---: | ---: |
| f32 | 1 | 0.027 |
| bf16 | 1 | 0.067 |
| f16 (default) | 1 | 0.0096 |

f16 and bf16 take the same VRAM; f16 keeps 10 mantissa bits to bf16's 7 (as HyperQwen notes for this model's state).

## Vision encoder

Numerical agreement with llama.cpp: both encoders on the same raw 448x448 checkerboard (784 patches, position
embeddings interpolated from 48x48 to 28x28), sum of each checkpoint tensor:

| Checkpoint | llama.cpp (`llama-mtmd-debug -p encode -n 448 --image cb`) | klein (`klein vision-debug -pp 448`) |
| --- | ---: | ---: |
| patch embedding + bias | 23345.943 | 23343.399 |
| + position embeddings | 23648.951 | 23649.116 |
| layer 0 output | 56597.652 | 56598.149 |
| layer 13 output | 10203.045 | 10203.294 |
| layer 26 output | 824895.750 | 824902.045 |
| post layer norm | -298.317 | -298.320 |
| image embeddings (output) | 2329.092 | 2329.145 |

The first row differs most because llama.cpp computes the patch convolution in f16, klein in f32.

Encoding time (IQ4_XS model, 262K context, weights streamed from RAM): a 640x480 photo (300 tokens) 123-255 ms,
a 929x861 screenshot (783 tokens) 367-430 ms, including borrowing and returning VRAM. Decoding after an image runs
at the usual speed (15.4 tok/s on the first answer about the photo, 8K context).

Greedy answers to the same two images and prompts from `klein serve` and `llama-server` (both `--mmproj`,
`tests/bench/vision_compare.py`) agree word for word for the first ~30 tokens of the description and then differ in
phrasing ("Below the headline" / "Below the main headline"), as greedy decoding does after a near-tie.

## Lookup drafting (from HyperQwen), measured and left off

Drafting from the earlier occurrence of what was just written (`--lookup`: a match of 8+ tokens is used alone, 3-7
only when the MTP head's first draft agrees), IQ4_XS, 262K context, greedy:

| Task | Without lookup | With lookup |
| --- | ---: | ---: |
| Rewrite a 1,013-token C++ file with two renames (976 tokens out) | 23.88 tok/s, 99% of drafts accepted | 24.10 tok/s, 99%; 239 of 246 steps from the context |
| `klein bench -pp 512 -n 256` | 19.89 tok/s, 79% | 19.08 tok/s, 77% |

HyperQwen's gain comes from verifying 15-token blocks while the text is copied. klein verifies 4 tokens per step
(the batch the repacked AVX2 kernels handle in one pass), and within 4 tokens the MTP head already drafts copies
at 99%, so lookup only saves the ~7 ms of draft passes. Longer verify blocks would need 7 rollback snapshots instead
of 3 (+~290 MiB of VRAM, about 8% slower decoding everywhere) for ~20% on copies. Lookup stays available as an
option, off by default.
