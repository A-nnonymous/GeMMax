// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef INCLUDE_JIT_COMPILER_HPP_
#define INCLUDE_JIT_COMPILER_HPP_

#include <cuda.h>
#include <nvrtc.h>
#include <memory>
#include "./cache_manager.hpp"
#include "./fp8_gemm_types.hpp"

// Error checking macros
#define CHECK_NVRTC(expr)                                                 \
  do {                                                                    \
    nvrtcResult result = (expr);                                          \
    if (result != NVRTC_SUCCESS) {                                        \
      throw std::runtime_error("NVRTC error: " +                          \
                               std::string(nvrtcGetErrorString(result))); \
    }                                                                     \
  } while (0)

#define CHECK_CUDA(expr)                                               \
  do {                                                                 \
    CUresult result = (expr);                                          \
    if (result != CUDA_SUCCESS) {                                      \
      const char* err_str;                                             \
      cuGetErrorString(result, &err_str);                              \
      throw std::runtime_error("CUDA error: " + std::string(err_str)); \
    }                                                                  \
  } while (0)

/**
 * @brief JIT compiler for CUDA FP8 GEMM kernels
 *
 * This class handles runtime compilation of specialized FP8 GEMM kernels
 * using NVRTC, with caching support for compiled binaries.
 */
class JITCompiler {
 public:
  /**
   * @brief Construct a new JIT Compiler object
   * @param cache_manager Pointer to cache manager for storing compiled kernels
   */
  explicit JITCompiler(std::shared_ptr<CacheManager> cache_manager);

  /**
   * @brief Destroy the JIT Compiler object
   */
  ~JITCompiler();

  /**
   * @brief Compile or retrieve from cache a GEMM kernel for the given
   * configuration
   * @param config GEMM configuration parameters
   * @return Handle to the compiled CUDA function
   */
  CUfunction compile_gemm_kernel(const GemmConfig& config);

 private:
  std::shared_ptr<CacheManager> cache_manager_;
  CUcontext cuda_context_;

  /**
   * @brief Generate CUDA source code for a GEMM kernel with the given
   * configuration
   * @param config GEMM configuration parameters
   * @param block_size_m Block size for M dimension
   * @param block_size_n Block size for N dimension
   * @param block_size_k Block size for K dimension
   * @return Generated source code as string
   */
  std::string generate_gemm_source(const GemmConfig& config,
                                   int block_size_m,
                                   int block_size_n,
                                   int block_size_k) const;

  /**
   * @brief Get kernel from cache or compile new one if missing
   * @param config GEMM configuration parameters
   * @return Handle to the compiled CUDA function
   */
  CUfunction get_or_compile_kernel(const GemmConfig& config);

  /**
   * @brief Compile source code to cubin binary
   * @param source CUDA source code
   * @return Compiled cubin data as vector of bytes
   */
  std::vector<char> compile_to_cubin(const std::string& source) const;

  /**
   * @brief Load kernel function from cubin data
   * @param cubin_data Compiled cubin binary data
   * @param kernel_name Name of the kernel function to load
   * @return Handle to the loaded CUDA function
   */
  CUfunction load_kernel_from_cubin(const std::vector<char>& cubin_data,
                                    const std::string& kernel_name) const;

  /**
   * @brief Select optimal block sizes based on matrix dimensions
   * @param config GEMM configuration parameters
   * @param block_size_m Output block size for M dimension
   * @param block_size_n Output block size for N dimension
   * @param block_size_k Output block size for K dimension
   */
  void select_block_sizes(const GemmConfig& config,
                          int& block_size_m,                  // NOLINT
                          int& block_size_n,                  // NOLINT
                          int& block_size_k) const noexcept;  // NOLINT
};

#endif  // INCLUDE_JIT_COMPILER_HPP_
