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

#include <rapidhash.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace {
// Helper function to format hash value as hex string
std::string format_hash(uint64_t hash) noexcept {
  std::stringstream ss;
  ss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return ss.str();
}
}  // namespace

CacheManager::CacheManager(std::filesystem::path cache_dir)
    : cache_dir_(std::move(cache_dir)) {
  ensure_directory_exists(cache_dir_);
}

CacheManager::~CacheManager() {
  // Wait for all asynchronous writes to complete
  std::lock_guard<std::mutex> lock(write_mutex_);
  for (auto& task : write_tasks_) {
    if (task.valid()) {
      task.wait();
    }
  }
}

bool CacheManager::get(const GemmConfig& config,
                       std::vector<char>& data) const {
  const uint64_t key = generate_key(config);
  const std::string filename = hash_to_filename(key);
  const std::filesystem::path path = cache_dir_ / filename;

  if (!std::filesystem::exists(path)) {
    return false;
  }

  // Read file contents
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }

  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  data.resize(static_cast<size_t>(size));
  return file.read(data.data(), size).good();
}

void CacheManager::put_async(const GemmConfig& config,
                             const std::vector<char>& data) {
  std::lock_guard<std::mutex> lock(write_mutex_);

  // Launch asynchronous write
  write_tasks_.emplace_back(std::async(
      std::launch::async, &CacheManager::put_sync, this, config, data));

  // Clean up completed tasks
  auto it = std::remove_if(
      write_tasks_.begin(), write_tasks_.end(), [](const std::future<void>& f) {
        return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
      });
  write_tasks_.erase(it, write_tasks_.end());
}

uint64_t CacheManager::generate_key(const GemmConfig& config) const noexcept {
  // Use rapidhash correctly with pointer and size
  return rapidhash(&config, sizeof(GemmConfig));
}

std::string CacheManager::hash_to_filename(uint64_t hash) const noexcept {
  return format_hash(hash) + ".cubin";
}

void CacheManager::put_sync(const GemmConfig& config,
                            const std::vector<char>& data) const {
  const uint64_t key = generate_key(config);
  const std::string filename = hash_to_filename(key);
  const std::filesystem::path path = cache_dir_ / filename;

  // Write data to file
  std::ofstream file(path, std::ios::binary);
  if (file.is_open()) {
    file.write(data.data(), data.size());
  }
}

void CacheManager::ensure_directory_exists(
    const std::filesystem::path& dir) const {
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
}
