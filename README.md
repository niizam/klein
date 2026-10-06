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
  weights yields ~3.5 tokens. DeltaNet state rollback uses f16 snapshots. The MTP head attends over a 16K
  sliding window kept in VRAM, so drafting does not slow down as the context grows.
- **Keeps a 262K KV cache in RAM without slowing decoding**: the GPU reads it directly over PCIe (zero-copy,
  through a small ggml-cuda patch). Prompt reading instead copies it to VRAM in bulk.
- **Moves VRAM between weights and prompt scratch on demand**: long prompts borrow VRAM from a few FFN blocks
  while they are read; decoding gets it back.
- Uses **CUDA graphs**, reuses the cached prompt prefix between turns, and warms up at start-up.

The details and the measurements behind each choice: [docs/DESIGN.md](docs/DESIGN.md).

## Quick start

### Download (Windows, prebuilt)

From the [latest release](https://github.com/niizam/klein/releases/latest), download both zips and unpack them into
the same folder:

- `klein-<version>-win-cuda13-x64.zip`: `klein.exe` and its tools
- `cuda13-runtime-win-x64.zip`: the NVIDIA cuBLAS libraries klein needs (`cublas64_13.dll`, `cublasLt64_13.dll`)

Requirements:

| | |
| --- | --- |
| GPU | NVIDIA RTX 30, 40 or 50 series with 12 GB (prebuilt for sm_86, sm_89 and sm_120) |
| Driver | an NVIDIA driver for CUDA 13 (tested with 591.86) |
| CPU | x86-64 with AVX2 (any Ryzen; Intel from 2013 on) |
| RAM | 32 GB for the full 262K context (less with `-c`) |
| OS | Windows 10/11 x64 with the [Microsoft Visual C++ Redistributable 2015-2022](https://aka.ms/vs/17/release/vc_redist.x64.exe) |

### Build from source

Visual Studio 2022 with C++ and the CUDA toolkit (tested with 13.4). See [docs/BUILDING.md](docs/BUILDING.md).

```powershell
git clone https://github.com/niizam/klein; cd klein
scripts\build.ps1                       # ~40 min the first time (CUDA kernels), then seconds
```

### Run

```powershell
# get the model (13.3 GiB)
hf download unsloth/Qwen3.8-27B-GGUF --include "*UD-IQ4_XS*" --local-dir models

# chat in the terminal
klein.exe run -m models\Qwen3.8-27B-UD-IQ4_XS.gguf -p "Write a haiku about GPUs."

# chat page on http://127.0.0.1:8080/ and OpenAI-compatible API on http://127.0.0.1:8080/v1
klein.exe serve -m models\Qwen3.8-27B-UD-IQ4_XS.gguf
```

(From a source build, the executable is `build\klein.exe`.)

For images, also download `mmproj-BF16.gguf` from the same repository and add `--mmproj mmproj-BF16.gguf`; then
attach pictures in the chat page or send them to the API. Details: [docs/USAGE.md](docs/USAGE.md#images).

Point any OpenAI-compatible app at `http://127.0.0.1:8080/v1`. Streaming, thinking (`reasoning_content`, effort
low/medium/high/none) and tool calls are supported. Commands, options and the API: [docs/USAGE.md](docs/USAGE.md).

## Which model file

| Unsloth GGUF | Size | Writes answers (262K context) | Quality vs the Q4_K_S reference |
| --- | ---: | ---: | --- |
| `UD-IQ4_XS` (default) | 13.3 GiB | ~20 tok/s | KL divergence 0.017, same top token 92.6% |
| `UD-IQ3_XXS` | 10.2 GiB | ~49 tok/s | KL divergence 0.080, same top token 85.6% |

IQ3_XXS fits in 12 GB of VRAM almost entirely, so it is ~2.4x faster, at a clearly measurable quality cost. How
both were measured: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#quantization-speed-vs-quality).

## Smaller machines

The reference machine is the largest klein targets. Less RAM or VRAM works with a shorter context (`-c 32768`
keeps the KV cache at ~0.6-1 GiB) or without the repacked CPU copies (`--no-repack`, less RAM, slower answers).
The planner adapts to the VRAM it finds free.

## Status

Working and measured: the qwen35 architecture (dense Qwen3.5/3.6/3.8), images (Qwen3-VL vision encoder), GGUF
quants supported by ggml, CUDA GPUs (Ampere and newer by default), Windows. Not supported: video, MoE Qwen
variants, several concurrent requests, Linux builds (not tried).

## Acknowledgments

klein stands on other people's work:

- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** (MIT): klein's tensor library, its CPU and CUDA
  kernels for every GGUF quant type, and the reference implementations klein's qwen35 graph, tokenizer, vision
  encoder and image preprocessing were ported from and checked against. Its `llama-perplexity`,
  `llama-mtmd-debug`, `llama-bench` and `llama-server` were the yardsticks for every accuracy and speed number.
- **[Strata](https://github.com/Niko1221/Strata)** by Niko1221: the hybrid GPU/CPU approach for Qwen models on
  12 GB cards, KV caches kept in RAM, and the measured, plain-words documentation style klein follows.
- **[HyperQwen](https://github.com/syv-ai/HyperQwen)** by syv-ai: the observation that this model's recurrent state
  keeps its precision in fp16 (klein's rollback snapshots) and the lookup drafting design (implemented and
  measured in klein, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
- **[Unsloth](https://unsloth.ai)**: the Dynamic GGUF quants and the vision encoder GGUF klein runs, and the
  Qwen3.8 run guide.
- **[Qwen](https://huggingface.co/Qwen/Qwen3.8-27B)**: the Qwen3.8-27B model (Apache-2.0).
- **[Pillow](https://github.com/python-pillow/Pillow)**: the bicubic resampler (through llama.cpp's port), so
  images are resized exactly as the model's own preprocessing does.
- **[cpp-httplib](https://github.com/yhirose/cpp-httplib)**, **[nlohmann/json](https://github.com/nlohmann/json)**
  and **[stb_image](https://github.com/nothings/stb)**: the HTTP server, JSON and image decoding.

## License

GPL-3.0-or-later ([LICENSE](LICENSE)). klein includes ggml and parts of llama.cpp (MIT), cpp-httplib (MIT),
nlohmann/json (MIT) and stb_image (public domain / MIT); see [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md). The
prebuilt release ships NVIDIA's cuBLAS libraries under the
[CUDA EULA](https://docs.nvidia.com/cuda/eula/) (redistributable files). The model has its own license
(Apache-2.0 for Qwen3.8-27B).
