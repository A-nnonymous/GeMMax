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

#include "./utils.h"
#include <cstring>

std::vector<fp8_e4m3fn> generate_random_fp8_matrix(int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const size_t size = static_cast<size_t>(rows) * cols;
  std::vector<fp8_e4m3fn> matrix(size);

  for (size_t i = 0; i < size; ++i) {
    matrix[i] = float_to_fp8(dist(gen));
  }

  return matrix;
}

std::vector<float> generate_random_float_matrix(int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const size_t size = static_cast<size_t>(rows) * cols;
  std::vector<float> matrix(size);

  for (size_t i = 0; i < size; ++i) {
    matrix[i] = dist(gen);
  }

  return matrix;
}

void gemm_cpu_reference(const std::vector<fp8_e4m3fn>& A,
                        const std::vector<fp8_e4m3fn>& B,
                        std::vector<float>& C,  // NOLINT
                        const GemmConfig& config) {
  // Create temporary copy of C to use for beta calculation
  const std::vector<float> C_initial = C;

  // CPU-based GEMM implementation
  for (int i = 0; i < config.m; ++i) {
    for (int j = 0; j < config.n; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < config.k; ++k) {
        const float a_val = fp8_to_float(A[i * config.lda + k]);
        const float b_val = fp8_to_float(B[k * config.ldb + j]);
        sum += a_val * b_val;
      }
      C[i * config.ldc + j] =
          config.alpha * sum + config.beta * C_initial[i * config.ldc + j];
    }
  }
}

bool verify_gemm_result(const std::vector<fp8_e4m3fn>& A,
                        const std::vector<fp8_e4m3fn>& B,
                        const std::vector<float>& C_initial,
                        const std::vector<float>& C_result,
                        const GemmConfig& config,
                        float tolerance) {
  // Compute reference result on CPU
  std::vector<float> C_reference = C_initial;
  gemm_cpu_reference(A, B, C_reference, config);

  // Check dimensions match
  if (C_result.size() != C_reference.size()) {
    return false;
  }

  // Check each element
  for (size_t i = 0; i < C_result.size(); ++i) {
    const float diff = std::abs(C_result[i] - C_reference[i]);
    const float abs_val = std::abs(C_reference[i]);

    // Handle special case where reference is zero
    const float rel_error = (abs_val < 1e-9f) ? diff : diff / abs_val;

    if (rel_error > tolerance) {
      return false;
    }
  }

  return true;
}
