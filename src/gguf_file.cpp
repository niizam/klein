// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "gguf_file.h"

#include "common.h"
#include "ggml.h"
#include "gguf.h"

namespace klein {

static int seek64(std::FILE* f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (__int64) off, SEEK_SET);
#else
    return fseeko(f, (off_t) off, SEEK_SET);
#endif
}

GgufFile::GgufFile(const std::string& path) : path_(path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &meta_;
    gguf_ = gguf_init_from_file(path.c_str(), params);
    if (!gguf_) fatal("cannot read GGUF file '%s'", path.c_str());

    const size_t data_off = gguf_get_data_offset(gguf_);
    const int64_t n = gguf_get_n_tensors(gguf_);
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(meta_, name);
        KLEIN_ASSERT(t != nullptr);
        tensors_.push_back(t);
        by_name_[name] = t;
        offsets_[t] = data_off + gguf_get_tensor_offset(gguf_, i);
    }
    file_ = std::fopen(path.c_str(), "rb");
    if (!file_) fatal("cannot open '%s'", path.c_str());
}

GgufFile::~GgufFile() {
    if (file_) std::fclose(file_);
    if (meta_) ggml_free(meta_);
    if (gguf_) gguf_free(gguf_);
}

bool GgufFile::has_key(const std::string& key) const { return gguf_find_key(gguf_, key.c_str()) >= 0; }

uint32_t GgufFile::get_u32(const std::string& key, uint32_t def) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(gguf_, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf_, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(gguf_, id);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(gguf_, id);
        case GGUF_TYPE_INT64:  return (uint32_t) gguf_get_val_i64(gguf_, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(gguf_, id);
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(gguf_, id);
        default: fatal("GGUF key '%s' is not an integer", key.c_str());
    }
}

uint32_t GgufFile::get_u32(const std::string& key) const {
    if (!has_key(key)) fatal("GGUF '%s' lacks required key '%s'", path_.c_str(), key.c_str());
    return get_u32(key, 0);
}

float GgufFile::get_f32(const std::string& key, float def) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(gguf_, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf_, id);
        case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(gguf_, id);
        default: return (float) get_u32(key, 0);
    }
}

std::string GgufFile::get_str(const std::string& key, const std::string& def) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(gguf_, id);
}

size_t GgufFile::get_arr_n(const std::string& key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY) return 0;
    return gguf_get_arr_n(gguf_, id);
}

std::string GgufFile::get_arr_str(const std::string& key, size_t i) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    KLEIN_ASSERT(id >= 0);
    return gguf_get_arr_str(gguf_, id, i);
}

std::vector<int64_t> GgufFile::get_arr_int(const std::string& key) const {
    std::vector<int64_t> out;
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) return out;
    if (gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY) {
        out.push_back(get_u32(key, 0));
        return out;
    }
    const size_t n = gguf_get_arr_n(gguf_, id);
    const void* d = gguf_get_arr_data(gguf_, id);
    out.resize(n);
    switch (gguf_get_arr_type(gguf_, id)) {
        case GGUF_TYPE_INT32:  for (size_t i = 0; i < n; ++i) out[i] = ((const int32_t*) d)[i]; break;
        case GGUF_TYPE_UINT32: for (size_t i = 0; i < n; ++i) out[i] = ((const uint32_t*) d)[i]; break;
        case GGUF_TYPE_INT64:  for (size_t i = 0; i < n; ++i) out[i] = ((const int64_t*) d)[i]; break;
        case GGUF_TYPE_UINT64: for (size_t i = 0; i < n; ++i) out[i] = (int64_t) ((const uint64_t*) d)[i]; break;
        case GGUF_TYPE_INT8:   for (size_t i = 0; i < n; ++i) out[i] = ((const int8_t*) d)[i]; break;
        case GGUF_TYPE_UINT8:  for (size_t i = 0; i < n; ++i) out[i] = ((const uint8_t*) d)[i]; break;
        case GGUF_TYPE_INT16:  for (size_t i = 0; i < n; ++i) out[i] = ((const int16_t*) d)[i]; break;
        case GGUF_TYPE_UINT16: for (size_t i = 0; i < n; ++i) out[i] = ((const uint16_t*) d)[i]; break;
        case GGUF_TYPE_BOOL:   for (size_t i = 0; i < n; ++i) out[i] = ((const int8_t*) d)[i] ? 1 : 0; break;
        default: fatal("GGUF key '%s' is not an integer array", key.c_str());
    }
    return out;
}

ggml_tensor* GgufFile::tensor(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : it->second;
}

uint64_t GgufFile::tensor_file_offset(const ggml_tensor* t) const {
    auto it = offsets_.find(t);
    KLEIN_ASSERT(it != offsets_.end());
    return it->second;
}

void GgufFile::read_tensor_data(const ggml_tensor* t, void* dst, size_t offset, size_t size) const {
    KLEIN_ASSERT(offset + size <= ggml_nbytes(t));
    if (seek64(file_, tensor_file_offset(t) + offset) != 0) fatal("seek failed in '%s'", path_.c_str());
    if (std::fread(dst, 1, size, file_) != size) fatal("short read in '%s'", path_.c_str());
}

}  // namespace klein
