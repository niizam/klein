# Third-party code

klein is GPL-3.0-or-later. It includes the following code under GPL-compatible licenses; each directory keeps its
license file.

| Directory | Origin | License | Use |
| --- | --- | --- | --- |
| `third_party/ggml` | ggml from [llama.cpp](https://github.com/ggml-org/llama.cpp) release b11435 | MIT | tensors, CPU and CUDA backends, quantized kernels, GGUF reader |
| `third_party/llama-unicode` | `src/unicode*.{cpp,h}` from llama.cpp b11435 | MIT | Unicode categories and the pre-tokenizer regex split |
| `third_party/httplib` | [cpp-httplib](https://github.com/yhirose/cpp-httplib) 0.59.0 | MIT | HTTP server |
| `third_party/json` | [nlohmann/json](https://github.com/nlohmann/json) 3.12.0 | MIT | JSON |

The qwen35 graph (`src/graph.cpp`) and the BPE tokenizer (`src/tokenizer.cpp`) follow llama.cpp's
implementations (`src/models/qwen35.cpp`, `delta-net-base.cpp`, `llama-vocab.cpp`, MIT), which klein uses as the
reference for numerical and token-level correctness.

## Changes to the vendored ggml

ggml is trimmed to the core, CPU and CUDA backends (the other backends' directories are removed). Code changes are
marked `[klein]` in the sources:

1. **Mapped host buffer type** (`ggml-cuda.cu`, `include/ggml-cuda.h`): `ggml_backend_cuda_mapped_host_buffer_type()`
   allocates pinned host memory (`cudaMallocHost`) and hands it to the CUDA backend as a device buffer. Under
   unified virtual addressing the host pointer is valid in kernels, so ops on tensors in it run on the GPU and read
   or write the memory over PCIe without first copying the whole tensor. klein puts a RAM-resident KV cache in it.
   - `ggml_backend_cuda_buffer_context` gets a `host_mapped` flag. Mapped buffers are freed with `cudaFreeHost`,
     memset and cleared on the host, and copied with `cudaMemcpyDefault`.
   - Same-device tensor copies use `cudaMemcpyDefault` (works for both device and mapped memory).

## Build configuration

`CMakeLists.txt` sets for ggml: `GGML_CUDA=ON`, `GGML_CUDA_GRAPHS=ON` (off by default outside llama.cpp),
`GGML_CUDA_FA_ALL_QUANTS=OFF` (the KV types klein uses, f16/q8_0/q4_0, are in the default set),
`GGML_NATIVE=ON`, static libraries.
