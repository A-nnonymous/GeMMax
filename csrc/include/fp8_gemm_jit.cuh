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

#pragma once

#include <cuda.h>
#include <nvrtc.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "fp8_gemm_types.h"

/**
 * @brief JIT compiler and executor for FP8 GEMM on H100
 *
 * Handles runtime compilation of FP8 matrix multiplication kernels using NVRTC,
 * with caching of compiled binaries to avoid repeated compilation. Uses
 * rapidhash for cache key generation and supports efficient kernel execution.
 */
class Fp8GemmJit {
 private:
  CUdevice device;                 ///< CUDA device handle
  CUcontext context;               ///< CUDA context handle
  std::string cache_path;          ///< Path to cache file
  mutable std::mutex cache_mutex;  ///< Mutex for thread-safe cache access

  // Cache storage: hash -> {config, cubin, kernel}
  struct CacheEntry {
    CachedGemmData data;
    CUmodule module;
    CUfunction function;
  };
  std::unordered_map<uint64_t, CacheEntry> cache;

  // Cache statistics
  size_t cache_hits = 0;
  size_t cache_misses = 0;

  /**
   * @brief Load cache from disk
   */
  void load_cache();

  /**
   * @brief Save cache to disk
   */
  void save_cache() const;

  /**
   * @brief Read kernel template from file
   * @return Template content as string
   */
  std::string read_kernel_template() const;

  /**
   * @brief Replace placeholders in template with actual values
   * @param template_str Template content
   * @param config Configuration parameters
   * @return Modified source code with replaced values
   */
  std::string replace_template_placeholders(const std::string& template_str,
                                            const GemmConfig& config) const;

  /**
   * @brief Compile kernel source to PTX using NVRTC
   * @param source Kernel source code
   * @return Compiled PTX
   */
  std::string compile_to_ptx(const std::string& source) const;

  /**
   * @brief Compile PTX to cubin using CUDA driver API
   * @param ptx PTX source code
   * @return Compiled cubin binary
   */
  std::vector<char> compile_to_cubin(const std::string& ptx) const;

  /**
   * @brief Create kernel function from cubin data
   * @param cubin Cubin binary data
   * @param module Output module handle
   * @param function Output function handle
   */
  void create_kernel_from_cubin(const std::vector<char>& cubin,
                                CUmodule& module,
                                CUfunction& function) const;

  /**
   * @brief Compute combined hash of config and cubin data
   * @param config GEMM configuration
   * @param cubin Cubin binary data
   * @return Combined 64-bit hash
   */
  static uint64_t compute_combined_hash(
      const GemmConfig& config, const std::vector<char>& cubin) noexcept;

 public:
  /**
   * @brief Constructor
   * @param device_id CUDA device ID to use
   * @param cache_file Path to cache file
   */
  Fp8GemmJit(int device_id = 0,
             const std::string& cache_file = "fp8_gemm_cache.bin");

  /**
   * @brief Destructor - saves cache to disk
   */
  ~Fp8GemmJit();

  /**
   * @brief Execute FP8 GEMM with given configuration
   * @param A Pointer to matrix A (FP8) on device
   * @param B Pointer to matrix B (FP8) on device
   * @param C Pointer to matrix C (float) on device
   * @param config GEMM configuration parameters
   * @return CUDA error code
   */
  cudaError_t execute(const __nv_fp8_e4m3fn* A,
                      const __nv_fp8_e4m3fn* B,
                      float* C,
                      const GemmConfig& config);

  /**
   * @brief Get cache statistics
   * @param hits Output: number of cache hits
   * @param misses Output: number of cache misses
   */
  void get_cache_stats(size_t& hits, size_t& misses) const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    hits = cache_hits;
    misses = cache_misses;
  }

  /**
   * @brief Clear cache (both memory and disk)
   */
  void clear_cache();

  // Delete copy constructor and assignment operator
  Fp8GemmJit(const Fp8GemmJit&) = delete;
  Fp8GemmJit& operator=(const Fp8GemmJit&) = delete;

  // Allow move operations
  Fp8GemmJit(Fp8GemmJit&&) = default;
  Fp8GemmJit& operator=(Fp8GemmJit&&) = default;
};
