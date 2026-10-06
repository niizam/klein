cmake_minimum_required(VERSION 3.21)
project(klein VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# The reference GPU is an RTX 3080 Ti (sm_86). Real SASS only: the installed driver may be older than the
# toolkit, and PTX from a newer toolkit cannot be JIT-compiled by an older driver.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86-real" CACHE STRING "")
endif()

# ggml (vendored, MIT): core + CPU + CUDA backends, statically linked.
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "")
set(GGML_CUDA_FA_ALL_QUANTS OFF CACHE BOOL "")
set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)  # replay decode graphs with one launch (off by default outside llama.cpp)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml)

add_library(klein_unicode STATIC
    third_party/llama-unicode/unicode.cpp
    third_party/llama-unicode/unicode-data.cpp)
target_include_directories(klein_unicode PUBLIC third_party/llama-unicode)

add_library(klein_core STATIC
    src/common.cpp
    src/gguf_file.cpp
    src/tokenizer.cpp
    src/model.cpp
    src/state.cpp
    src/graph.cpp
    src/planner.cpp
    src/sampler.cpp
    src/engine.cpp
    src/transcode.cpp
    src/chat.cpp
    src/server.cpp
    src/image.cpp
    src/vision.cpp)
target_include_directories(klein_core PUBLIC src third_party/json third_party/httplib third_party/stb)

# The chat page (src/web/index.html) is compiled into the binary and served at "/".
include(cmake/embed.cmake)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/web/index.html)
embed_file(${CMAKE_CURRENT_SOURCE_DIR}/src/web/index.html ${CMAKE_CURRENT_BINARY_DIR}/generated/index_html.h klein_index_html)
target_include_directories(klein_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
if (WIN32)
    target_link_libraries(klein_core PUBLIC ws2_32)
    target_compile_definitions(klein_core PUBLIC _WIN32_WINNT=0x0A00)
endif()
target_link_libraries(klein_core PUBLIC ggml ggml-base ggml-cpu ggml-cuda klein_unicode)
if (MSVC)
    target_compile_options(klein_core PRIVATE /W3 /utf-8)
    target_compile_options(klein_unicode PRIVATE /utf-8)
    target_compile_definitions(klein_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

add_executable(klein src/main.cpp)
target_link_libraries(klein PRIVATE klein_core)

add_executable(klein-tokenize tools/tokenize.cpp)
target_link_libraries(klein-tokenize PRIVATE klein_core)

add_executable(klein-chat-test tools/chat_test.cpp)
target_link_libraries(klein-chat-test PRIVATE klein_core)

add_executable(klein-cpubench tools/cpubench.cpp)
target_link_libraries(klein-cpubench PRIVATE klein_core)
