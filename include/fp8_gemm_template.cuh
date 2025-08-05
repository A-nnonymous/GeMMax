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

#ifndef FP8_GEMM_TP
#define FP8_GEMM_TP

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "fp8_gemm_types.hpp"

// H100 native FP8 type (E4M3 format with IEEE 754-style NaNs and infs)
using fp8_e4m3fn = __nv_fp8_e4m3;

/**
 * @brief Converts float to fp8_e4m3fn
 */
__host__ __device__ inline fp8_e4m3fn float_to_fp8(float value) {
  return static_cast<fp8_e4m3fn>(value);
}

/**
 * @brief Converts fp8_e4m3fn to float
 */
__host__ __device__ inline float fp8_to_float(fp8_e4m3fn value) {
  return float(value);
}

/**
 * @brief FP8 GEMM kernel (C = alpha*A*B + beta*C)
 *
 * Template parameters:
 *  - BLOCK_SIZE_M: Rows handled by each thread block
 *  - BLOCK_SIZE_N: Columns handled by each thread block
 *  - BLOCK_SIZE_K: Tile size for K dimension
 *  - CONFIG: constexpr GEMM configuration
 *
 * @param A Input matrix A (m x k) in fp8_e4m3fn format
 * @param B Input matrix B (k x n) in fp8_e4m3fn format
 * @param C Output matrix C (m x n) in float format
 */
template <int BLOCK_SIZE_M,
          int BLOCK_SIZE_N,
          int BLOCK_SIZE_K,
          const GemmConfig& CONFIG>
__global__ void gemm_fp8_kernel(const fp8_e4m3fn* __restrict__ A,
                                const fp8_e4m3fn* __restrict__ B,
                                float* __restrict__ C) {
  // Thread indices within block
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Block indices
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Global row and column indices
  const int row = by * BLOCK_SIZE_M + ty;
  const int col = bx * BLOCK_SIZE_N + tx;

  // Shared memory for tile data (padded to avoid bank conflicts)
  __shared__ fp8_e4m3fn sA[BLOCK_SIZE_M][BLOCK_SIZE_K + 1];
  __shared__ fp8_e4m3fn sB[BLOCK_SIZE_K + 1][BLOCK_SIZE_N];

  // Accumulator register
  float sum = 0.0f;

  // Loop over K dimension in tiles
  for (int bk = 0; bk < (CONFIG.k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++bk) {
    // Load tile from A matrix into shared memory
    const int a_col = bk * BLOCK_SIZE_K + tx;
    if (row < CONFIG.m && a_col < CONFIG.k) {
      sA[ty][tx] = A[row * CONFIG.lda + a_col];
    } else {
      sA[ty][tx] = float_to_fp8(0.0f);
    }

    // Load tile from B matrix into shared memory
    const int b_row = bk * BLOCK_SIZE_K + ty;
    if (b_row < CONFIG.k && col < CONFIG.n) {
      sB[ty][tx] = B[b_row * CONFIG.ldb + col];
    } else {
      sB[ty][tx] = float_to_fp8(0.0f);
    }

    __syncthreads();

// Compute matrix multiplication for this tile
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE_K; ++k) {
      sum += fp8_to_float(sA[ty][k]) * fp8_to_float(sB[k][tx]);
    }

    __syncthreads();
  }

  // Store result to global memory if within bounds
  if (row < CONFIG.m && col < CONFIG.n) {
    C[row * CONFIG.ldc + col] =
        CONFIG.alpha * sum + CONFIG.beta * C[row * CONFIG.ldc + col];
  }
}

#endif  // FP8_GEMM_TP
