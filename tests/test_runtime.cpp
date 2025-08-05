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

#include <cuda.h>
#include <chrono>
#include <filesystem>
#include "../include/cache_manager.hpp"
#include "../include/jit_compiler.hpp"
#include "../include/test_framework.hpp"
#include "../include/utils.h"

TEST("Runtime: Small matrix multiplication", test_small_matrix_gemm) {
  auto cache_manager = std::make_shared<CacheManager>("./runtime_test_cache");
  JITCompiler compiler(cache_manager);

  // Configuration
  const int m = 32;
  const int n = 32;
  const int k = 32;

  const GemmConfig config = {.m = m,
                             .n = n,
                             .k = k,
                             .lda = k,
                             .ldb = n,
                             .ldc = n,
                             .alpha = 1.0f,
                             .beta = 0.0f};

  // Generate input matrices
  auto A_host = generate_random_fp8_matrix(m, k);
  auto B_host = generate_random_fp8_matrix(k, n);
  auto C_host = generate_random_float_matrix(m, n);

  // Copy to device
  fp8_e4m3fn* A_device = copy_to_device(A_host);
  fp8_e4m3fn* B_device = copy_to_device(B_host);
  float* C_device = copy_to_device(C_host);

  // Compile kernel
  CUfunction kernel = compiler.compile_gemm_kernel(config);

  // Launch kernel
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  void* args[] = {&A_device, &B_device, &C_device};
  CHECK_CUDA(cuLaunchKernel(kernel,
                            grid.x,
                            grid.y,
                            grid.z,
                            block.x,
                            block.y,
                            block.z,
                            0,
                            nullptr,
                            args,
                            nullptr));
  CHECK_CUDA(cuCtxSynchronize());

  // Copy result back
  auto C_result = copy_from_device(C_device, m * n);

  // Verify result
  const bool valid =
      verify_gemm_result(A_host, B_host, C_host, C_result, config);
  ASSERT_TRUE(valid, "Small matrix GEMM result is invalid");

  // Cleanup
  free_device_memory(A_device);
  free_device_memory(B_device);
  free_device_memory(C_device);
  std::filesystem::remove_all("./runtime_test_cache");
}

TEST("Runtime: Medium non-square matrix multiplication",
     test_medium_matrix_gemm) {
  auto cache_manager = std::make_shared<CacheManager>("./medium_test_cache");
  JITCompiler compiler(cache_manager);

  // Non-square configuration
  const int m = 128;
  const int n = 256;
  const int k = 64;

  const GemmConfig config = {
      .m = m,
      .n = n,
      .k = k,
      .lda = k,
      .ldb = n,
      .ldc = n,
      .alpha = 0.5f,
      .beta = 0.5f  // Test with scaling
  };

  // Generate input matrices
  auto A_host = generate_random_fp8_matrix(m, k);
  auto B_host = generate_random_fp8_matrix(k, n);
  auto C_host = generate_random_float_matrix(m, n);

  // Copy to device
  fp8_e4m3fn* A_device = copy_to_device(A_host);
  fp8_e4m3fn* B_device = copy_to_device(B_host);
  float* C_device = copy_to_device(C_host);

  // Compile kernel
  CUfunction kernel = compiler.compile_gemm_kernel(config);

  // Launch kernel
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  void* args[] = {&A_device, &B_device, &C_device};
  CHECK_CUDA(cuLaunchKernel(kernel,
                            grid.x,
                            grid.y,
                            grid.z,
                            block.x,
                            block.y,
                            block.z,
                            0,
                            nullptr,
                            args,
                            nullptr));
  CHECK_CUDA(cuCtxSynchronize());

  // Copy result back
  auto C_result = copy_from_device(C_device, m * n);

  // Verify result
  const bool valid =
      verify_gemm_result(A_host, B_host, C_host, C_result, config);
  ASSERT_TRUE(valid, "Medium matrix GEMM result is invalid");

  // Cleanup
  free_device_memory(A_device);
  free_device_memory(B_device);
  free_device_memory(C_device);
  std::filesystem::remove_all("./medium_test_cache");
}

TEST("Runtime: Performance measurement", test_gemm_performance) {
  auto cache_manager = std::make_shared<CacheManager>("./perf_test_cache");
  JITCompiler compiler(cache_manager);

  // Larger matrix for meaningful performance measurement
  const int m = 1024;
  const int n = 1024;
  const int k = 1024;

  const GemmConfig config = {.m = m,
                             .n = n,
                             .k = k,
                             .lda = k,
                             .ldb = n,
                             .ldc = n,
                             .alpha = 1.0f,
                             .beta = 0.0f};

  // Generate input matrices
  auto A_host = generate_random_fp8_matrix(m, k);
  auto B_host = generate_random_fp8_matrix(k, n);
  auto C_host = generate_random_float_matrix(m, n);

  // Copy to device
  fp8_e4m3fn* A_device = copy_to_device(A_host);
  fp8_e4m3fn* B_device = copy_to_device(B_host);
  float* C_device = copy_to_device(C_host);

  // Compile kernel (warm-up)
  CUfunction kernel = compiler.compile_gemm_kernel(config);

  // Launch parameters
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  void* args[] = {&A_device, &B_device, &C_device};

  // Measure execution time
  const int iterations = 10;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    CHECK_CUDA(cuLaunchKernel(kernel,
                              grid.x,
                              grid.y,
                              grid.z,
                              block.x,
                              block.y,
                              block.z,
                              0,
                              nullptr,
                              args,
                              nullptr));
  }
  CHECK_CUDA(cuCtxSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  const float avg_time = static_cast<float>(duration) / iterations /
                         1000.0f;  // Average time in seconds

  // Calculate TFLOPS (2 operations per multiply-add)
  const float flops = 2.0f * m * n * k;
  const float tflops = flops / (avg_time * 1e12f);

  std::cout << "\nPerformance: " << tflops << " TFLOPS" << std::endl;

  // For H100, we expect at least 100 TFLOPS for FP8 GEMM
  ASSERT_GT(tflops, 100.0f, "Performance below expected threshold");

  // Cleanup
  free_device_memory(A_device);
  free_device_memory(B_device);
  free_device_memory(C_device);
  std::filesystem::remove_all("./perf_test_cache");
}

int main() {
  test::TestRegistry::instance().run_all();
  return 0;
}
