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

#include <rapidhash.h>
#include <array>
#include <cstdint>
#include <cstring>

// Forward declaration
struct GemmConfig;

/**
 * @brief Structure combining configuration and cubin data with precomputed hash
 *
 * Stores all necessary information for a cached GEMM kernel, including
 * configuration parameters, compiled cubin data, and a precomputed hash
 * for efficient equality checks.
 */
struct CachedGemmData {
  GemmConfig config;        ///< GEMM configuration parameters
  uint64_t hash;            ///< Precomputed hash for fast equality checks
  std::vector<char> cubin;  ///< Compiled cubin binary data

  /**
   * @brief Default constructor
   */
  constexpr CachedGemmData() = default;

  /**
   * @brief Constructor with config and cubin data
   * @param cfg GEMM configuration
   * @param cbn Compiled cubin data
   */
  CachedGemmData(const GemmConfig& cfg, std::vector<char>&& cbn);

  /**
   * @brief Equality operator with fast hash check first
   * @param other Other CachedGemmData instance to compare
   * @return True if both instances are equal
   */
  bool operator==(const CachedGemmData& other) const noexcept {
    // Fast hash check first, then full data comparison
    if (hash != other.hash) return false;
    if (!(config == other.config)) return false;
    return cubin == other.cubin;
  }
};

/**
 * @brief Configuration parameters for FP8 GEMM
 *
 * Contains all tunable parameters for FP8 matrix multiplication, including
 * matrix dimensions, transformation flags, scaling factors, and kernel tuning
 * parameters like block sizes and unroll factors.
 */
struct GemmConfig {
  // Matrix dimensions and layout
  size_t m = 0;          ///< Rows of matrix A and C
  size_t n = 0;          ///< Columns of matrix B and C
  size_t k = 0;          ///< Columns of matrix A and rows of matrix B
  bool trans_a = false;  ///< Transpose matrix A
  bool trans_b = false;  ///< Transpose matrix B

  // Scaling factors for FP8 conversion
  float scale_A = 1.0f;  ///< Scaling factor for matrix A
  float scale_B = 1.0f;  ///< Scaling factor for matrix B
  float scale_C = 1.0f;  ///< Scaling factor for matrix C

  // GEMM coefficients
  float alpha = 1.0f;  ///< Alpha coefficient
  float beta = 0.0f;   ///< Beta coefficient

  // Kernel tuning parameters
  size_t block_size_m = 32;  ///< Block size for M dimension
  size_t block_size_n = 32;  ///< Block size for N dimension
  size_t block_size_k = 16;  ///< Block size for K dimension
  size_t unroll_factor = 4;  ///< Loop unrolling factor

  /**
   * @brief Default constructor
   */
  constexpr GemmConfig() = default;

  /**
   * @brief Constructor with matrix dimensions
   * @param m_rows Rows of matrix A and C
   * @param n_cols Columns of matrix B and C
   * @param k_dim Columns of matrix A and rows of matrix B
   */
  constexpr GemmConfig(size_t m_rows, size_t n_cols, size_t k_dim)
      : m(m_rows), n(n_cols), k(k_dim) {}

  /**
   * @brief Equality operator
   * @param other Other GemmConfig instance to compare
   * @return True if all parameters are equal
   */
  bool operator==(const GemmConfig& other) const noexcept {
    return m == other.m && n == other.n && k == other.k &&
           trans_a == other.trans_a && trans_b == other.trans_b &&
           scale_A == other.scale_A && scale_B == other.scale_B &&
           scale_C == other.scale_C && alpha == other.alpha &&
           beta == other.beta && block_size_m == other.block_size_m &&
           block_size_n == other.block_size_n &&
           block_size_k == other.block_size_k &&
           unroll_factor == other.unroll_factor;
  }

  /**
   * @brief Compute hash of configuration using rapidhash
   * @return 64-bit hash value
   */
  uint64_t compute_hash() const noexcept {
    return rapidhash(this, sizeof(GemmConfig));
  }
};

// Specialization of std::hash for CachedGemmData
namespace std {
template <>
struct hash<CachedGemmData> {
  size_t operator()(const CachedGemmData& data) const noexcept {
    return data.hash;
  }
};
}  // namespace std
