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
#include <iostream>
#include <memory>
#include "./include/cache_manager.hpp"
#include "./include/jit_compiler.hpp"
#include "./include/utils.h"

int main() {
  try {
    // Initialize cache manager
    auto cache_manager = std::make_shared<CacheManager>("./gemm_cache");

    // Initialize JIT compiler
    JITCompiler jit_compiler(cache_manager);

    // Define GEMM problem dimensions
    const int m = 2048;
    const int n = 2048;
    const int k = 2048;

    // Create GEMM configuration
    const GemmConfig config{.m = m,
                            .n = n,
                            .k = k,
                            .lda = k,
                            .ldb = n,
                            .ldc = n,
                            .alpha = 1.0f,
                            .beta = 0.0f};

    // Generate random input matrices
    std::cout << "Generating random FP8 matrices (" << m << "x" << k << " and "
              << k << "x" << n << ")..." << std::endl;
    auto A_host = generate_random_fp8_matrix(m, k);
    auto B_host = generate_random_fp8_matrix(k, n);
    auto C_host = generate_random_float_matrix(m, n);

    // Copy data to device
    std::cout << "Copying data to device memory..." << std::endl;
    fp8_e4m3fn* A_device = copy_to_device(A_host);
    fp8_e4m3fn* B_device = copy_to_device(B_host);
    float* C_device = copy_to_device(C_host);

    // First run - compile or load from cache
    std::cout << "First GEMM execution..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    CUfunction kernel = jit_compiler.compile_gemm_kernel(config);

    // Configure kernel launch
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    // Launch kernel
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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "First execution time: " << duration.count() << " seconds"
              << std::endl;

    // Second run - should use cache
    std::cout << "Second GEMM execution (cached)..." << std::endl;
    start = std::chrono::high_resolution_clock::now();

    kernel = jit_compiler.compile_gemm_kernel(config);

    // Launch kernel again
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

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Second execution time: " << duration.count() << " seconds"
              << std::endl;

    // Copy result back to host
    std::cout << "Copying result back to host..." << std::endl;
    auto C_result = copy_from_device(C_device, m * n);

    // Verify result
    std::cout << "Verifying GEMM result..." << std::endl;
    bool valid = verify_gemm_result(A_host, B_host, C_host, C_result, config);
    std::cout << "Result verification: " << (valid ? "PASSED" : "FAILED")
              << std::endl;

    // Calculate performance
    const float flops = 2.0f * m * n * k;
    const float tflops = flops / (duration.count() * 1e12f);
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // Cleanup
    free_device_memory(A_device);
    free_device_memory(B_device);
    free_device_memory(C_device);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
