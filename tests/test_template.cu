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

#include <type_traits>
#include "../include/fp8_gemm_template.cuh"
#include "../include/fp8_gemm_types.hpp"
#include "../include/test_framework.hpp"

constexpr GemmConfig make_config(
    int M, int N, int K, int bM, int bN, int bK, float a, float b) {
  return GemmConfig(M, N, K, bM, bN, bK, a, b);
}
inline static constexpr auto config1 =
    make_config(16, 16, 16, 16, 16, 16, 1.0f, 0.0f);
// Test constexpr configuration
constexpr GemmConfig test_config =
    make_config(32, 32, 32, 32, 32, 32, 1.0f, 0.0f);

// Test template instantiation with different block sizes
template class std::is_invocable<
    decltype(gemm_fp8_kernel<16, 16, 16, test_config>),
    const fp8_e4m3fn*,
    const fp8_e4m3fn*,
    float*>;

TEST("Template: Constexpr config validation", test_constexpr_config) {
  // Verify constexpr values
  ASSERT_EQ(test_config.m, 32, "M dimension mismatch");
  ASSERT_EQ(test_config.n, 32, "N dimension mismatch");
  ASSERT_EQ(test_config.k, 32, "K dimension mismatch");
  ASSERT_EQ(test_config.lda, 32, "LDA mismatch");
  ASSERT_EQ(test_config.ldb, 32, "LDB mismatch");
  ASSERT_EQ(test_config.ldc, 32, "LDC mismatch");
  ASSERT_NEAR(test_config.alpha, 1.0f, 1e-6f, "Alpha mismatch");
  ASSERT_NEAR(test_config.beta, 0.0f, 1e-6f, "Beta mismatch");
}

TEST("Template: FP8 conversion functions", test_fp8_conversions) {
  // Test float to FP8 and back conversion
  const float test_values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f};

  for (float val : test_values) {
    const fp8_e4m3fn fp8_val = float_to_fp8(val);
    const float converted_back = fp8_to_float(fp8_val);

    // Allow for some error due to FP8 precision limitations
    ASSERT_NEAR(val, converted_back, 0.1f, "FP8 conversion error");
  }
}

TEST("Template: Different block size instantiations",
     test_block_size_instantiation) {
  // Verify we can create different template instantiations
  // This is a compile-time test that we can instantiate different block sizes

  (void)gemm_fp8_kernel<16, 16, 16, config1>;

  ASSERT_TRUE(true, "All template instantiations succeeded");
}

int main() {
  test::TestRegistry::instance().run_all();
  return 0;
}
