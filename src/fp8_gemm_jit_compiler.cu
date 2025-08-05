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

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "./fp8_gemm_jit.cuh"

// Error checking macros
#define CHECK_NVRTC(expr)                                                 \
  do {                                                                    \
    nvrtcResult result = (expr);                                          \
    if (result != NVRTC_SUCCESS) {                                        \
      throw std::runtime_error("NVRTC error: " +                          \
                               std::string(nvrtcGetErrorString(result))); \
    }                                                                     \
  } while (0)

#define CHECK_CUDA(expr)                                               \
  do {                                                                 \
    CUresult result = (expr);                                          \
    if (result != CUDA_SUCCESS) {                                      \
      const char* err_str;                                             \
      cuGetErrorString(result, &err_str);                              \
      throw std::runtime_error("CUDA error: " + std::string(err_str)); \
    }                                                                  \
  } while (0)

// Implement CachedGemmData constructor
CachedGemmData::CachedGemmData(const GemmConfig& cfg, std::vector<char>&& cbn)
    : config(cfg), cubin(std::move(cbn)) {
  hash = Fp8GemmJit::compute_combined_hash(config, cubin);
}

// Fp8GemmJit implementation
Fp8GemmJit::Fp8GemmJit(int device_id, const std::string& cache_file)
    : cache_path(cache_file) {
  // Initialize CUDA driver
  CHECK_CUDA(cuInit(0));

  // Get device handle
  CHECK_CUDA(cuDeviceGet(&device, device_id));

  // Create context
  CHECK_CUDA(cuCtxCreate(&context, 0, device));

  // Load existing cache
  try {
    load_cache();
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to load cache - " << e.what() << std::endl;
    std::cerr << "Proceeding with empty cache" << std::endl;
  }
}

Fp8GemmJit::~Fp8GemmJit() {
  // Destroy all modules
  std::lock_guard<std::mutex> lock(cache_mutex);
  for (auto& entry : cache) {
    if (entry.second.module) {
      cuModuleUnload(entry.second.module);
    }
  }

  // Save cache to disk
  try {
    save_cache();
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to save cache - " << e.what() << std::endl;
  }

  // Destroy context
  if (context) {
    cuCtxDestroy(context);
  }
}

void Fp8GemmJit::load_cache() {
  std::ifstream file(cache_path, std::ios::binary);
  if (!file.is_open()) {
    return;  // No existing cache, nothing to load
  }

  std::lock_guard<std::mutex> lock(cache_mutex);
  cache.clear();

  // Read number of entries
  size_t num_entries;
  file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));

  for (size_t i = 0; i < num_entries; ++i) {
    // Read CachedGemmData
    CachedGemmData data;

    // Read config
    file.read(reinterpret_cast<char*>(&data.config), sizeof(GemmConfig));

    // Read hash
    file.read(reinterpret_cast<char*>(&data.hash), sizeof(uint64_t));

    // Read cubin size and data
    size_t cubin_size;
    file.read(reinterpret_cast<char*>(&cubin_size), sizeof(cubin_size));
    data.cubin.resize(cubin_size);
    file.read(data.cubin.data(), cubin_size);

    // Create module and function
    CacheEntry entry;
    entry.data = std::move(data);

    try {
      create_kernel_from_cubin(entry.data.cubin, entry.module, entry.function);
      cache[entry.data.hash] = std::move(entry);
    } catch (const std::exception& e) {
      std::cerr << "Warning: Failed to load cached kernel - " << e.what()
                << std::endl;
      std::cerr << "This entry will be ignored" << std::endl;
    }
  }
}

void Fp8GemmJit::save_cache() const {
  std::ofstream file(cache_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open cache file for writing: " +
                             cache_path);
  }

  std::lock_guard<std::mutex> lock(cache_mutex);

  // Write number of entries
  size_t num_entries = cache.size();
  file.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));

  // Write each entry
  for (const auto& pair : cache) {
    const CacheEntry& entry = pair.second;
    const CachedGemmData& data = entry.data;

    // Write config
    file.write(reinterpret_cast<const char*>(&data.config), sizeof(GemmConfig));

    // Write hash
    file.write(reinterpret_cast<const char*>(&data.hash), sizeof(uint64_t));

    // Write cubin size and data
    size_t cubin_size = data.cubin.size();
    file.write(reinterpret_cast<const char*>(&cubin_size), sizeof(cubin_size));
    file.write(data.cubin.data(), cubin_size);
  }
}

std::string Fp8GemmJit::read_kernel_template() const {
  std::ifstream file("fp8_gemm_kernels.template");
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open kernel template file");
  }

  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

std::string Fp8GemmJit::replace_template_placeholders(
    const std::string& template_str, const GemmConfig& config) const {
  // Preallocate buffer with extra space to avoid reallocations
  std::string result;
  result.reserve(template_str.size() + 1024);

  size_t pos = 0;
  size_t start, end;

  while ((start = template_str.find("{{", pos)) != std::string::npos) {
    // Copy characters before the placeholder
    result.append(template_str, pos, start - pos);

    // Find end of placeholder
    end = template_str.find("}}", start + 2);
    if (end == std::string::npos) {
      throw std::runtime_error("Unclosed placeholder in template");
    }

    // Extract placeholder name
    std::string placeholder = template_str.substr(start + 2, end - start - 2);

    // Replace with actual value
    if (placeholder == "BLOCK_SIZE_M") {
      result += std::to_string(config.block_size_m);
    } else if (placeholder == "BLOCK_SIZE_N") {
      result += std::to_string(config.block_size_n);
    } else if (placeholder == "BLOCK_SIZE_K") {
      result += std::to_string(config.block_size_k);
    } else if (placeholder == "UNROLL_FACTOR") {
      result += std::to_string(config.unroll_factor);
    } else {
      throw std::runtime_error("Unknown placeholder in template: " +
                               placeholder);
    }

    pos = end + 2;
  }

  // Copy remaining characters
  result.append(template_str, pos);

  return result;
}

std::string Fp8GemmJit::compile_to_ptx(const std::string& source) const {
  nvrtcProgram program;
  const char* source_ptr = source.c_str();

  // Create program
  CHECK_NVRTC(nvrtcCreateProgram(
      &program, source_ptr, "fp8_gemm_kernel.cu", 0, nullptr, nullptr));

  // Compile options for H100
  const std::vector<const char*> options = {"--gpu-architecture=sm_90",
                                            "--use_fast_math",
                                            "-default-device",
                                            "-Xptxas=-v"};

  // Compile
  nvrtcResult compile_result =
      nvrtcCompileProgram(program, options.size(), options.data());

  // Check for compilation errors
  if (compile_result != NVRTC_SUCCESS) {
    size_t log_size;
    CHECK_NVRTC(nvrtcGetProgramLogSize(program, &log_size));
    std::string log(log_size, '\0');
    CHECK_NVRTC(nvrtcGetProgramLog(program, log.data()));

    nvrtcDestroyProgram(&program);
    throw std::runtime_error("PTX compilation failed:\n" + log);
  }

  // Get PTX code
  size_t ptx_size;
  CHECK_NVRTC(nvrtcGetPTXSize(program, &ptx_size));
  std::string ptx(ptx_size, '\0');
  CHECK_NVRTC(nvrtcGetPTX(program, ptx.data()));

  // Cleanup
  CHECK_NVRTC(nvrtcDestroyProgram(&program));

  return ptx;
}

std::vector<char> Fp8GemmJit::compile_to_cubin(const std::string& ptx) const {
  CUmodule module;
  CHECK_CUDA(cuModuleLoadData(&module, ptx.c_str()));

  // Get cubin data size
  size_t cubin_size;
  CHECK_CUDA(cuModuleGetData(module, nullptr, &cubin_size));

  // Allocate buffer and get cubin data
  std::vector<char> cubin(cubin_size);
  CHECK_CUDA(cuModuleGetData(
      module, reinterpret_cast<void**>(cubin.data()), &cubin_size));

  // Unload module (we'll reload it when needed)
  CHECK_CUDA(cuModuleUnload(module));

  return cubin;
}

void Fp8GemmJit::create_kernel_from_cubin(const std::vector<char>& cubin,
                                          CUmodule& module,
                                          CUfunction& function) const {
  // Load module from cubin data
  CHECK_CUDA(cuModuleLoadData(&module, cubin.data()));

  // Get function handle
  CHECK_CUDA(cuModuleGetFunction(&function, module, "fp8_gemm_kernel"));
}

uint64_t Fp8GemmJit::compute_combined_hash(
    const GemmConfig& config, const std::vector<char>& cubin) noexcept {
  // First compute hash of config
  uint64_t config_hash = config.compute_hash();

  // Then compute hash of cubin data using config_hash as seed
  if (cubin.empty()) {
    return config_hash;
  }

  return rapidhash_with_seed(cubin.data(), cubin.size(), config_hash);
}

cudaError_t Fp8GemmJit::execute(const __nv_fp8_e4m3fn* A,
                                const __nv_fp8_e4m3fn* B,
                                float* C,
                                const GemmConfig& config) {
  try {
    // Calculate hash for this configuration
    uint64_t config_hash = config.compute_hash();

    CacheEntry* entry = nullptr;

    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      auto it = cache.find(config_hash);

      // Check if we have a cache hit
      if (it != cache.end() && it->second.data.config == config) {
        entry = &it->second;
        cache_hits++;
      } else {
        cache_misses++;
      }
    }

    // If no cache hit, compile new kernel
    if (!entry) {
      // Read and process template
      std::string kernel_template = read_kernel_template();
      std::string kernel_source =
          replace_template_placeholders(kernel_template, config);

      // Compile to PTX and then to cubin
      std::string ptx = compile_to_ptx(kernel_source);
      std::vector<char> cubin = compile_to_cubin(ptx);

      // Create cached entry
      CachedGemmData data(config, std::move(cubin));

      // Create module and function
      CacheEntry new_entry;
      new_entry.data = std::move(data);
      create_kernel_from_cubin(
          new_entry.data.cubin, new_entry.module, new_entry.function);

      // Add to cache
      std::lock_guard<std::mutex> lock(cache_mutex);
      entry = &cache[new_entry.data.hash];
      *entry = std::move(new_entry);
    }

    // Calculate leading dimensions
    size_t lda = config.trans_a ? config.m : config.k;
    size_t ldb = config.trans_b ? config.k : config.n;
    size_t ldc = config.n;

    // Configure kernel launch parameters
    dim3 block(config.block_size_n, config.block_size_m);
    dim3 grid((config.n + block.x - 1) / block.x,
              (config.m + block.y - 1) / block.y);

    // Kernel parameters
    void* args[] = {const_cast<__nv_fp8_e4m3fn*>(A),
                    const_cast<__nv_fp8_e4m3fn*>(B),
                    C,
                    &config.m,
                    &config.n,
                    &config.k,
                    &lda,
                    &ldb,
                    &ldc,
                    &config.trans_a,
                    &config.trans_b,
                    &config.scale_A,
                    &config.scale_B,
                    &config.scale_C,
                    &config.alpha,
                    &config.beta};

    // Launch kernel
    CHECK_CUDA(cuLaunchKernel(entry->function,
                              grid.x,
                              grid.y,
                              grid.z,  // Grid dimensions
                              block.x,
                              block.y,
                              block.z,  // Block dimensions
                              0,
                              nullptr,  // Shared memory and stream
                              args,
                              nullptr  // Kernel arguments
                              ));

    // Synchronize to ensure kernel completes
    CHECK_CUDA(cuCtxSynchronize());

    return cudaSuccess;
  } catch (const std::exception& e) {
    std::cerr << "Error executing GEMM: " << e.what() << std::endl;
    return cudaErrorUnknown;
  }
}

void Fp8GemmJit::clear_cache() {
  std::lock_guard<std::mutex> lock(cache_mutex);

  // Unload all modules
  for (auto& entry : cache) {
    if (entry.second.module) {
      cuModuleUnload(entry.second.module);
    }
  }

  // Clear cache
  cache.clear();
  cache_hits = 0;
  cache_misses = 0;

  // Remove cache file
  std::remove(cache_path.c_str());
}
