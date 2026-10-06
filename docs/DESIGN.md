# klein design

klein is an inference engine for **Qwen3.8-27B** (GGUF architecture `qwen35`) on one consumer NVIDIA GPU
plus system RAM. The reference machine is also the largest one klein targets:

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

## The model

From `config.json` and the GGUF metadata:

| | |
| --- | --- |
| Layers | 64 decoder layers + 1 MTP (multi-token prediction) layer |
| Layer pattern | 3 Gated DeltaNet (linear attention) layers, then 1 full-attention layer, repeated 16 times |
| Hidden size | 5,120; FFN 17,408 (SwiGLU) |
| Full attention | 24 query heads, 4 KV heads, head dim 256, output gate (sigmoid), Q/K RMS norm, partial MRoPE (64 of 256 dims, theta 1e7) |
| Gated DeltaNet | 16 key heads, 48 value heads, head dim 128, conv kernel 4 |
| Vocabulary | 248,320 tokens; untied input embedding and LM head |
| Context | 262,144 native |

Only the 16 full-attention layers keep a KV cache. The 48 DeltaNet layers keep a fixed-size recurrent state.

### Memory per component

| Component | Size | Notes |
| --- | ---: | --- |
| `token_embd` (Q3_K) | 0.51 GiB | Only one row per token is read: stays in RAM |
| `output` LM head (Q5_K) | 0.81 GiB | Read fully every target **and** every draft step |
| MTP layer (blk.64, Q6_K/Q8_0) | 0.36 GiB | Read every draft step |
| Decoder trunk (blk.0-63) | 11.58 GiB | Mixed IQ2_XS…Q6_K (Unsloth Dynamic) |
| Recurrent state | 144 MiB + 4.5 MiB conv | f32, per sequence (×K for rollback snapshots) |
| KV cache per token | 64 KiB f16 / 34 KiB q8_0 / 18 KiB q4_0 | 16 layers × (K+V) × 4 heads × 256 |
| KV cache at 262,144 | 16 GiB f16 / 8.5 GiB q8_0 / 4.5 GiB q4_0 | |

Free VRAM on the reference machine with the Windows desktop running is about **10.7 GiB**. The trunk +
LM head + MTP head alone are 12.75 GiB, so **at least ~2.5 GiB of weights must live in system RAM**, before any
KV cache.

## Performance model

Decoding one token is memory-bound: every weight byte is read once per forward pass.

- Bytes in VRAM are read at ~700-800 GB/s (achieved by quantized mat-vec kernels): ~10 GiB costs ~14 ms.
- Bytes in system RAM are read by the CPU at ~40 GB/s (dual-channel DDR4): 2.5 GiB costs ~65 ms.
- Streaming them to the GPU over PCIe instead is no better: ~22 GB/s, and it draws on the same DRAM bandwidth.

So **DRAM bandwidth spent per decode step is the single number that decides decode speed**:

```
time per step ≈ max(spilled_bytes / DRAM_BW, gpu_bytes / VRAM_BW) + per-step overhead
tokens/s      ≈ tokens accepted per step / time per step
```

It holds for KV cache bytes too: a KV byte in RAM costs the same as a weight byte in RAM, but only once the
context actually reaches it. This gives klein's levers:

1. **Spill as few bytes as possible.** Fill VRAM exactly: weights first, because every weight byte is read on
   every step, while KV bytes are read only up to the current context length. Keep `token_embd` in RAM because
   it is only looked up. Keep the LM head and the MTP layer on the GPU because drafts read them too.
2. **Read spilled bytes at full DRAM speed.** Prefer to spill tensors whose CPU kernels are fastest (repackable
   K-quants and Q8_0 over IQ2/IQ3 grid lookups), using a measured per-type throughput table.
3. **Overlap CPU and GPU work.** Split each layer's FFN by rows between the GPU and the CPU so both run at the
   same time, instead of running CPU layers and GPU layers one after another.
4. **Get more tokens per step.** Use the model's own MTP head for speculative decoding: one verify pass reads
   the spilled weights once for several tokens. DeltaNet state rollback uses the snapshot slots of ggml's fused
   `gated_delta_net` op.
5. **Keep per-step overhead small.** Use CUDA graphs, few host/device syncs and pinned transfer buffers.

Prefill is compute-bound (~54 GFLOP per token for the weights, plus attention that grows with the context).
Spilled weights are streamed to the GPU over PCIe for large batches. At ~22 GB/s, 2.5 GiB takes ~0.12 s per
2,048-token batch, which is small next to the batch's compute, and klein prefetches the next tensor while the
current one computes.

## Building blocks

klein builds on **ggml** (MIT; vendored under `third_party/`) for tensors, the CUDA and CPU backends and the
quantized kernels for every GGUF quant type. klein's own code covers the parts that decide speed on this
machine: the placement planner, the qwen35 graph (trunk and MTP), KV and recurrent-state memory, the hybrid
CPU/GPU execution schedule, speculative decoding, sampling, tokenizer, chat formatting and the server.

The architecture port follows llama.cpp's `qwen35` implementation (MIT), which is the reference for numerical
correctness.

## Status

This document records the design as it is built. Measured numbers go to [BENCHMARKS.md](BENCHMARKS.md), and
every claim of speed has a measurement there with the command and machine that produced it.
