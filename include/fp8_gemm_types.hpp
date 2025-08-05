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

#include <array>
#include <cstdint>
#include <cstring>
#include "./rapidhash.h"

/**
 * @brief Configuration structure for GEMM operations
 *
 * Contains all static parameters needed to define a GEMM operation.
 * Designed for use as a constexpr template parameter.
 */
struct alignas(16) GemmConfig {
  int m;        // Rows in matrix A and C
  int n;        // Columns in matrix B and C
  int k;        // Columns in A and rows in B
  int lda;      // Leading dimension of A
  int ldb;      // Leading dimension of B
  int ldc;      // Leading dimension of C
  float alpha;  // Scaling factor for A*B product
  float beta;   // Scaling factor for existing C values

  __host__ __device__ constexpr GemmConfig(int m_,
                                           int n_,
                                           int k_,
                                           int lda_,
                                           int ldb_,
                                           int ldc_,
                                           float alpha_,
                                           float beta_)
      : m(m_),
        n(n_),
        k(k_),
        lda(lda_),
        ldb(ldb_),
        ldc(ldc_),
        alpha(alpha_),
        beta(beta_) {}
};
