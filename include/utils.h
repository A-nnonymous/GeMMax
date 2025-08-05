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
#ifndef INCLUDE_UTILS_H_
#define INCLUDE_UTILS_H_

#include <cuda.h>
#include <nvrtc.h>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include "./fp8_gemm_types.hpp"

#ifndef CHECK_CUDA
// Error checking macros
#define CHECK_CUDA(expr)                                     \
  do {                                                       \
    CUresult result = (expr);                                \
    if (result != CUDA_SUCCESS) {                            \
      const char* error_str;                                 \
      cuGetErrorString(result, &error_str);                  \
      throw std::runtime_error(std::string(__FILE__) + ":" + \
                               std::to_string(__LINE__) +    \
                               " CUDA error: " + error_str); \
    }                                                        \
  } while (0)
#endif

#ifndef CHECK_NVRTC
#define CHECK_NVRTC(expr)                                          \
  do {                                                             \
    nvrtcResult result = (expr);                                   \
    if (result != NVRTC_SUCCESS) {                                 \
      throw std::runtime_error(                                    \
          std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
          " NVRTC error: " + nvrtcGetErrorString(result));         \
    }                                                              \
  } while (0)
#endif

/**
 * @brief Generates a random FP8 matrix on the host
 */
std::vector<fp8_e4m3fn> generate_random_fp8_matrix(int rows, int cols);

/**
 * @brief Generates a random float matrix on the host
 */
std::vector<float> generate_random_float_matrix(int rows, int cols);

/**
 * @brief Copies data from host to device
 */
template <typename T>
T* copy_to_device(const std::vector<T>& host_data);

/**
 * @brief Copies data from device to host
 */
template <typename T>
std::vector<T> copy_from_device(const T* device_data, size_t count);

/**
 * @brief Frees device memory
 */
template <typename T>
void free_device_memory(T* device_ptr);

/**
 * @brief CPU reference implementation of GEMM for verification
 */
void gemm_cpu_reference(const std::vector<fp8_e4m3fn>& A,
                        const std::vector<fp8_e4m3fn>& B,
                        std::vector<float>& C,  // NOLINT
                        const GemmConfig& config);

/**
 * @brief Verifies GEMM result against CPU reference
 */
bool verify_gemm_result(const std::vector<fp8_e4m3fn>& A,
                        const std::vector<fp8_e4m3fn>& B,
                        const std::vector<float>& C_initial,
                        const std::vector<float>& C_result,
                        const GemmConfig& config,
                        float tolerance = 1e-3f);

#include "utils.inl"

#endif  // INCLUDE_UTILS_H_
