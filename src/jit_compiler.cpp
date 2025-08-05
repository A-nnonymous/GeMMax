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

#include "./jit_compiler.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>

JITCompiler::JITCompiler(std::shared_ptr<CacheManager> cache_manager)
    : cache_manager_(std::move(cache_manager)), cuda_context_(nullptr) {
  if (!cache_manager_) {
    throw std::invalid_argument("Cache manager cannot be null");
  }

  // Initialize CUDA context
  CHECK_CUDA(cuInit(0));

  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, 0));

  CHECK_CUDA(cuCtxCreate(&cuda_context_, 0, device));
}

JITCompiler::~JITCompiler() {
  // Clean up CUDA context
  if (cuda_context_ != nullptr) {
    cuCtxDestroy(cuda_context_);
  }
}

CUfunction JITCompiler::compile_gemm_kernel(const GemmConfig& config) {
  return get_or_compile_kernel(config);
}

std::string JITCompiler::generate_gemm_source(const GemmConfig& config,
                                              int block_size_m,
                                              int block_size_n,
                                              int block_size_k) const {
  std::stringstream ss;

  // Include necessary headers
  ss << "#include <cuda_fp16.h>\n";
  ss << "#include <cstdint>\n";
  ss << "#include \"gemm_fp8.cuh\"\n\n";

  // Define constexpr configuration
  ss << "constexpr GemmConfig kernel_config = {\n";
  ss << "    .m = " << config.m << ",\n";
  ss << "    .n = " << config.n << ",\n";
  ss << "    .k = " << config.k << ",\n";
  ss << "    .lda = " << config.lda << ",\n";
  ss << "    .ldb = " << config.ldb << ",\n";
  ss << "    .ldc = " << config.ldc << ",\n";
  ss << "    .alpha = " << config.alpha << ",\n";
  ss << "    .beta = " << config.beta << "\n";
  ss << "};\n\n";

  // Explicit template instantiation
  ss << "template __global__ void gemm_fp8_kernel<" << block_size_m << ", "
     << block_size_n << ", " << block_size_k << ", kernel_config>("
     << "const fp8_e4m3fn*, const fp8_e4m3fn*, float*);\n\n";

  // Kernel wrapper with C linkage
  ss << "extern \"C\" __global__ void gemm_fp8_wrapper(\n";
  ss << "    const fp8_e4m3fn* A, const fp8_e4m3fn* B, float* C) {\n";
  ss << "    gemm_fp8_kernel<" << block_size_m << ", " << block_size_n << ", "
     << block_size_k << ", kernel_config>(A, B, C);\n";
  ss << "}\n";

  return ss.str();
}

CUfunction JITCompiler::get_or_compile_kernel(const GemmConfig& config) {
  // Try cache first
  std::vector<char> cubin_data;
  if (cache_manager_->get(config, cubin_data)) {
    try {
      return load_kernel_from_cubin(cubin_data, "gemm_fp8_wrapper");
    } catch (const std::exception&) {
      // Cache entry is invalid, proceed to recompile
    }
  }

  // Select optimal block sizes
  int block_size_m, block_size_n, block_size_k;
  select_block_sizes(config, block_size_m, block_size_n, block_size_k);

  // Generate kernel source
  const std::string source =
      generate_gemm_source(config, block_size_m, block_size_n, block_size_k);

  // Compile to cubin
  cubin_data = compile_to_cubin(source);

  // Asynchronously store in cache
  cache_manager_->put_async(config, cubin_data);

  // Load and return kernel
  return load_kernel_from_cubin(cubin_data, "gemm_fp8_wrapper");
}

std::vector<char> JITCompiler::compile_to_cubin(
    const std::string& source) const {
  // Create NVRTC program
  nvrtcProgram program;
  const char* source_ptr = source.c_str();
  CHECK_NVRTC(
      nvrtcCreateProgram(&program,
                         source_ptr,
                         "gemm_fp8_kernel.cu",  // Name for error reporting
                         0,
                         nullptr,
                         nullptr));

  // Compilation options - H100 optimized
  const std::vector<const char*> options = {
      "--gpu-architecture=sm_90",  // H100 architecture
      "--use_fast_math",
      "-default-device",
      "-Xptxas=-v",
      "--std=c++17",
      "-maxrregcount=255"};

  // Compile program
  const nvrtcResult compile_result =
      nvrtcCompileProgram(program, options.size(), options.data());

  // Get and log any compilation errors
  std::string log;
  size_t log_size;
  if (nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS &&
      log_size > 1) {
    log.resize(log_size);
    nvrtcGetProgramLog(program, log.data());
  }

  if (compile_result != NVRTC_SUCCESS) {
    throw std::runtime_error("NVRTC compilation failed: " + log);
  }

  // Get cubin size and data
  size_t cubin_size;
  CHECK_NVRTC(nvrtcGetCUBINSize(program, &cubin_size));

  std::vector<char> cubin_data(cubin_size);
  CHECK_NVRTC(nvrtcGetCUBIN(program, cubin_data.data()));

  // Cleanup
  CHECK_NVRTC(nvrtcDestroyProgram(&program));

  return cubin_data;
}

CUfunction JITCompiler::load_kernel_from_cubin(
    const std::vector<char>& cubin_data, const std::string& kernel_name) const {
  // Load module from cubin data
  CUmodule module;
  CHECK_CUDA(cuModuleLoadData(&module, cubin_data.data()));

  // Get kernel function
  CUfunction kernel;
  CHECK_CUDA(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));

  // Note: In production code, add module reference counting
  return kernel;
}

void JITCompiler::select_block_sizes(const GemmConfig& config,
                                     int& block_size_m,
                                     int& block_size_n,
                                     int& block_size_k) const noexcept {
  // Heuristic-based block size selection optimized for H100
  if (config.m >= 2048 && config.n >= 2048) {
    block_size_m = 32;
    block_size_n = 32;
    block_size_k = 16;
  } else if (config.m >= 1024 && config.n >= 1024) {
    block_size_m = 32;
    block_size_n = 32;
    block_size_k = 8;
  } else if (config.m >= 512 && config.n >= 512) {
    block_size_m = 16;
    block_size_n = 16;
    block_size_k = 16;
  } else {
    block_size_m = 16;
    block_size_n = 16;
    block_size_k = 8;
  }
}
