# Building klein

klein builds on Windows with Visual Studio 2022 and the CUDA toolkit. Everything else (ggml, cpp-httplib,
nlohmann/json, the Unicode tables) is vendored under `third_party/`.

## Requirements

| | Tested with |
| --- | --- |
| OS | Windows 11 Pro 24H2 (10.0.26100) |
| Compiler | Visual Studio Community 2022 (MSVC 19.44), "Desktop development with C++" |
| CUDA toolkit | 13.4 (nvcc 13.4.92) |
| NVIDIA driver | 591.86 (CUDA 13.1 runtime level), so the build emits native code |
| CMake + Ninja | the copies bundled with Visual Studio (found by `scripts/build.ps1`) |
| GPU | Ampere or newer for the defaults (`86-real` = RTX 30 series) |

The build compiles **native SASS only** (`CMAKE_CUDA_ARCHITECTURES=86-real`). PTX from a newer toolkit cannot be
JIT-compiled by an older driver, and native code also skips the JIT step at start-up. For another GPU, pass its
architecture, e.g. `-DCMAKE_CUDA_ARCHITECTURES=89-real` for an RTX 40 series card.

## Build

From PowerShell in the repository:

```powershell
scripts\build.ps1                              # configure (first time) and build everything into build\
scripts\build.ps1 -Targets klein               # only the main executable
scripts\build.ps1 -Jobs 8 -BuildDir build-dbg -Config RelWithDebInfo
```

`scripts\build.ps1` enters the Visual Studio developer environment, configures CMake with Ninja and builds at
**below-normal priority** with 4 parallel jobs by default, so the desktop stays usable during the CUDA compile.
The first build compiles ggml's CUDA kernels and takes about 40 minutes with 4 jobs on a Ryzen 7 5700X. Later
builds only recompile klein's own sources (seconds).

Outputs in `build\`:

| File | What it is |
| --- | --- |
| `klein.exe` | the engine: `run`, `bench`, `ppl`, `check-spec`, `serve` (see [USAGE.md](USAGE.md)) |
| `klein-tokenize.exe` | prints token ids, same format as llama.cpp's `llama-tokenize --ids` |
| `klein-chat-test.exe` | renders chat-template test cases (see `tests/chat/`) |
| `klein-cpubench.exe` | CPU matrix x small-batch throughput per quant type and layout |

## Tests

```powershell
# chat template: C++ renderer vs the real Jinja template (needs Python with jinja2)
build\klein-chat-test.exe tests\chat\cases.json > out_cpp.jsonl
python tests\chat\render_reference.py tests\chat\cases.json > out_ref.jsonl
python tests\chat\compare.py out_ref.jsonl out_cpp.jsonl

# speculative decoding emits what plain decoding emits (greedy), up to numerical near-ties
build\klein.exe check-spec -m Qwen3.8-27B-UD-IQ4_XS.gguf -c 8192 -n 300 -p "Explain how a hash table handles collisions."

# numerical agreement with llama.cpp: KL divergence against llama-perplexity's logits
llama-perplexity -m MODEL.gguf -f text.txt -c 512 --chunks 4 --kl-divergence-base base.bin
build\klein.exe ppl -m MODEL.gguf --kld-base base.bin --ppl-batch 512

# server end to end (start `klein serve` first)
python tests\server\smoke.py http://127.0.0.1:8080
```
