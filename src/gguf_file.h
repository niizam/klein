// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace klein {

// A GGUF file opened for metadata. Tensor descriptors live in a no-alloc ggml context; their data is read
// on demand with read_tensor_data().
class GgufFile {
public:
    explicit GgufFile(const std::string& path);
    ~GgufFile();
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    const std::string& path() const { return path_; }
    gguf_context* gguf() const { return gguf_; }

    bool has_key(const std::string& key) const;
    uint32_t get_u32(const std::string& key, uint32_t def) const;
    uint32_t get_u32(const std::string& key) const;  // fatal if missing
    float get_f32(const std::string& key, float def) const;
    std::string get_str(const std::string& key, const std::string& def = "") const;
    std::vector<int64_t> get_arr_int(const std::string& key) const;  // any integer array type
    size_t get_arr_n(const std::string& key) const;
    std::string get_arr_str(const std::string& key, size_t i) const;

    // Tensor metadata (no data). nullptr if missing.
    ggml_tensor* tensor(const std::string& name) const;
    const std::vector<ggml_tensor*>& tensors() const { return tensors_; }
    uint64_t tensor_file_offset(const ggml_tensor* t) const;  // absolute offset in the file

    // Reads `size` bytes of a tensor's data starting at `offset` within the tensor.
    void read_tensor_data(const ggml_tensor* t, void* dst, size_t offset, size_t size) const;

private:
    std::string path_;
    gguf_context* gguf_ = nullptr;
    ggml_context* meta_ = nullptr;
    std::vector<ggml_tensor*> tensors_;
    std::unordered_map<std::string, ggml_tensor*> by_name_;
    std::unordered_map<const ggml_tensor*, uint64_t> offsets_;
    mutable std::FILE* file_ = nullptr;
};

}  // namespace klein
