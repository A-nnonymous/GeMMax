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

#include <chrono>
#include <filesystem>
#include "../include/cache_manager.hpp"
#include "../include/jit_compiler.hpp"
#include "../include/test_framework.hpp"
#include "../include/utils.h"

TEST("JIT: Basic compilation", test_jit_compilation) {
  auto cache_manager = std::make_shared<CacheManager>("./jit_test_cache");
  JITCompiler compiler(cache_manager);

  // Simple GEMM configuration
  const GemmConfig config = {.m = 64,
                             .n = 64,
                             .k = 64,
                             .lda = 64,
                             .ldb = 64,
                             .ldc = 64,
                             .alpha = 1.0f,
                             .beta = 0.0f};

  // Compile kernel
  CUfunction kernel = compiler.compile_gemm_kernel(config);

  // Verify kernel is valid (non-null)
  ASSERT_TRUE(kernel != nullptr, "Kernel compilation failed");

  // Cleanup
  std::filesystem::remove_all("./jit_test_cache");
}

TEST("JIT: Cache functionality", test_jit_cache) {
  auto cache_manager = std::make_shared<CacheManager>("./cache_test_cache");
  JITCompiler compiler(cache_manager);

  // Define configuration
  const GemmConfig config = {.m = 128,
                             .n = 128,
                             .k = 128,
                             .lda = 128,
                             .ldb = 128,
                             .ldc = 128,
                             .alpha = 1.0f,
                             .beta = 0.0f};

  // First compilation - should generate cache
  auto start = std::chrono::high_resolution_clock::now();
  CUfunction kernel1 = compiler.compile_gemm_kernel(config);
  auto end = std::chrono::high_resolution_clock::now();
  const auto first_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  // Second compilation - should use cache
  start = std::chrono::high_resolution_clock::now();
  CUfunction kernel2 = compiler.compile_gemm_kernel(config);
  end = std::chrono::high_resolution_clock::now();
  const auto second_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  // Verify kernels are valid
  ASSERT_TRUE(kernel1 != nullptr, "First kernel compilation failed");
  ASSERT_TRUE(kernel2 != nullptr, "Second kernel compilation failed");

  // Cache version should be significantly faster (at least 5x)
  ASSERT_LT(second_time * 5, first_time, "Cache not working as expected");

  // Cleanup
  std::filesystem::remove_all("./cache_test_cache");
}

TEST("JIT: Different configurations generate different kernels",
     test_jit_config_differentiation) {
  auto cache_manager = std::make_shared<CacheManager>("./config_test_cache");
  JITCompiler compiler(cache_manager);

  // Create two different configurations
  const GemmConfig config1 = {.m = 64,
                              .n = 64,
                              .k = 64,
                              .lda = 64,
                              .ldb = 64,
                              .ldc = 64,
                              .alpha = 1.0f,
                              .beta = 0.0f};

  const GemmConfig config2 = {.m = 128,
                              .n = 128,
                              .k = 128,  // Different dimensions
                              .lda = 128,
                              .ldb = 128,
                              .ldc = 128,
                              .alpha = 1.0f,
                              .beta = 0.0f};

  // Generate different cache keys
  const uint64_t key1 = cache_manager->generate_key(config1);
  const uint64_t key2 = cache_manager->generate_key(config2);

  // Keys should be different for different configurations
  ASSERT_NE(key1, key2, "Different configurations generated same cache key");

  // Compile both kernels
  CUfunction kernel1 = compiler.compile_gemm_kernel(config1);
  CUfunction kernel2 = compiler.compile_gemm_kernel(config2);

  ASSERT_TRUE(kernel1 != nullptr, "Kernel1 compilation failed");
  ASSERT_TRUE(kernel2 != nullptr, "Kernel2 compilation failed");

  // Cleanup
  std::filesystem::remove_all("./config_test_cache");
}

int main() {
  test::TestRegistry::instance().run_all();
  return 0;
}
