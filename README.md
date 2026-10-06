# klein

**A fast local inference engine for Qwen3.8-27B on a 12 GB NVIDIA GPU.**

klein runs Qwen3.8-27B at ~4 bits (Unsloth's `UD-IQ4_XS` GGUF, 13.3 GiB) with its native **262,144-token
context** on a 12 GB graphics card plus 32 GB of RAM, using CUDA. It is about **1.8x faster at writing answers**
than llama.cpp's best configuration on the same PC, and reads prompts faster too.

Measured on an RTX 3080 Ti (12 GB), Ryzen 7 5700X, 32 GB DDR4-3200, Windows 11, 262,144-token context, the same
two prompts sent to each engine's server:

| | llama.cpp b11435, best config | **klein** |
| --- | ---: | ---: |
| Writes answers: code prompt | 12.3 tok/s | **22.2 tok/s** |
| Writes answers: prose prompt | 8.8 tok/s | **16.0 tok/s** |
| Reads a 4,096-token prompt | 1,004 tok/s | **1,169 tok/s** |

llama.cpp's configuration was tuned for this machine: MTP drafts, q4_0 KV cache in RAM, the CPU-friendliest FFN
tensors in RAM, 12 threads. All numbers, commands and more cases are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Why it is faster

The model needs about 13 GiB and the card has about 11 GiB free, so some weights must stay in RAM, where the CPU
reads them at ~35 GB/s instead of ~700 GB/s. That gap decides the speed. klein:

- **Picks which weights go to RAM** by measured CPU cost, whole FFN blocks at a time, and keeps a repacked copy
  (IQ4_XS transcoded to IQ4_NL) for the AVX2 kernels. A 4-token verify then runs near DRAM speed instead of half.
- **Speculates with the model's own MTP head**: 3 drafts per step, verified in one pass, so each pass over the RAM
  weights yields ~3.5 tokens. DeltaNet state rollback uses bf16 snapshots. The MTP head attends over a 16K
  sliding window kept in VRAM, so drafting does not slow down as the context grows.
- **Keeps a 262K KV cache in RAM without slowing decoding**: the GPU reads it directly over PCIe (zero-copy,
  through a small ggml-cuda patch). Prompt reading instead copies it to VRAM in bulk.
- **Moves VRAM between weights and prompt scratch on demand**: long prompts borrow VRAM from a few FFN blocks
  while they are read; decoding gets it back.
- Uses **CUDA graphs**, reuses the cached prompt prefix between turns, and warms up at start-up.

The details and the measurements behind each choice: [docs/DESIGN.md](docs/DESIGN.md).

## Quick start

Requirements: Windows 11 (tested), an NVIDIA GPU with 12 GB (the default build targets RTX 30 series / sm_86;
other architectures need `-DCMAKE_CUDA_ARCHITECTURES=...`), 32 GB RAM, Visual Studio 2022 with C++, the CUDA
toolkit (tested with 13.4). See [docs/BUILDING.md](docs/BUILDING.md).

```powershell
git clone <this repository> klein; cd klein
scripts\build.ps1                       # ~40 min the first time (CUDA kernels), then seconds

# get the model (13.3 GiB)
hf download unsloth/Qwen3.8-27B-GGUF --include "*UD-IQ4_XS*" --local-dir models

# chat in the terminal
build\klein.exe run -m models\Qwen3.8-27B-UD-IQ4_XS.gguf -p "Write a haiku about GPUs."

# OpenAI-compatible server on http://127.0.0.1:8080/v1
build\klein.exe serve -m models\Qwen3.8-27B-UD-IQ4_XS.gguf
```

Point any OpenAI-compatible app at `http://127.0.0.1:8080/v1`. Streaming, thinking (`reasoning_content`, effort
low/medium/high/none) and tool calls are supported. Commands, options and the API: [docs/USAGE.md](docs/USAGE.md).

## Smaller machines

The reference machine is the largest klein targets. Less RAM or VRAM works with a shorter context (`-c 32768`
keeps the KV cache at ~0.6-1 GiB) or without the repacked CPU copies (`--no-repack`, less RAM, slower answers).
The planner adapts to the VRAM it finds free.

## Status

Working and measured: the qwen35 architecture (dense Qwen3.5/3.6/3.8), GGUF quants supported by ggml, CUDA
GPUs (Ampere and newer by default), Windows. Not supported: images (the vision tower), MoE Qwen variants,
several concurrent requests, Linux builds (not tried).

## License

GPL-3.0-or-later ([LICENSE](LICENSE)). klein includes ggml and parts of llama.cpp (MIT), cpp-httplib (MIT) and
nlohmann/json (MIT); see [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md). The model has its own license (Apache-2.0
for Qwen3.8-27B).
