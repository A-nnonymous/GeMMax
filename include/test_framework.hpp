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

#ifndef INCLUDE_TEST_FRAMEWORK_HPP_
#define INCLUDE_TEST_FRAMEWORK_HPP_

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace test {
// Test result tracking
struct TestResult {
  std::string name;
  bool passed;
  std::string message;
};

// Test registry
class TestRegistry {
 private:
  std::vector<std::pair<std::string, std::function<void()>>> tests_;
  std::vector<TestResult> results_;

  TestRegistry() = default;

 public:
  // Singleton instance
  static TestRegistry& instance() {
    static TestRegistry registry;
    return registry;
  }

  // Register a test
  void register_test(const std::string& name, std::function<void()> test_func) {
    tests_.emplace_back(name, std::move(test_func));
  }

  // Run all tests
  void run_all() {
    results_.clear();
    std::cout << "Running " << tests_.size() << " tests...\n" << std::endl;

    for (const auto& [name, test_func] : tests_) {
      TestResult result{name, true, ""};
      try {
        test_func();
        std::cout << "PASS: " << name << std::endl;
      } catch (const std::exception& e) {
        result.passed = false;
        result.message = e.what();
        std::cout << "FAIL: " << name << " - " << e.what() << std::endl;
      } catch (...) {
        result.passed = false;
        result.message = "Unknown error";
        std::cout << "FAIL: " << name << " - Unknown error" << std::endl;
      }
      results_.push_back(result);
    }

    // Print summary
    const auto passed = std::count_if(
        results_.begin(), results_.end(), [](const TestResult& r) {
          return r.passed;
        });

    std::cout << "\nTest summary: " << passed << "/" << results_.size()
              << " passed" << std::endl;

    if (passed != results_.size()) {
      std::cout << "Failed tests:" << std::endl;
      for (const auto& result : results_) {
        if (!result.passed) {
          std::cout << "  " << result.name << ": " << result.message
                    << std::endl;
        }
      }
    }
  }
};

// Test registration helper
struct TestRegistrar {
  TestRegistrar(const std::string& name, std::function<void()> test_func) {
    TestRegistry::instance().register_test(name, std::move(test_func));
  }
};

// Assertion macros
#define ASSERT_TRUE(condition, message)                       \
  do {                                                        \
    if (!(condition)) {                                       \
      throw std::runtime_error("Assertion failed: " message); \
    }                                                         \
  } while (0)

#define ASSERT_FALSE(condition, message) ASSERT_TRUE(!(condition), message)

#define ASSERT_EQ(a, b, message)                                            \
  do {                                                                      \
    if ((a) != (b)) {                                                       \
      throw std::runtime_error("Assertion failed: " message " (" #a " = " + \
                               std::to_string(a) + ", " #b " = " +          \
                               std::to_string(b) + ")");                    \
    }                                                                       \
  } while (0)

#define ASSERT_NE(a, b, message)                                            \
  do {                                                                      \
    if ((a) == (b)) {                                                       \
      throw std::runtime_error("Assertion failed: " message " (" #a " = " + \
                               std::to_string(a) + ", " #b " = " +          \
                               std::to_string(b) + ")");                    \
    }                                                                       \
  } while (0)

#define ASSERT_LT(a, b, message)                                            \
  do {                                                                      \
    if ((a) >= (b)) {                                                       \
      throw std::runtime_error("Assertion failed: " message " (" #a " = " + \
                               std::to_string(a) + ", " #b " = " +          \
                               std::to_string(b) + ")");                    \
    }                                                                       \
  } while (0)

#define ASSERT_GT(a, b, message)                                            \
  do {                                                                      \
    if ((a) <= (b)) {                                                       \
      throw std::runtime_error("Assertion failed: " message " (" #a " = " + \
                               std::to_string(a) + ", " #b " = " +          \
                               std::to_string(b) + ")");                    \
    }                                                                       \
  } while (0)

#define ASSERT_NEAR(a, b, tolerance, message)                              \
  do {                                                                     \
    if (std::abs((a) - (b)) > (tolerance)) {                               \
      throw std::runtime_error(                                            \
          "Assertion failed: " message " (" #a " = " + std::to_string(a) + \
          ", " #b " = " + std::to_string(b) +                              \
          ", difference = " + std::to_string(std::abs(a - b)) + ")");      \
    }                                                                      \
  } while (0)

// Test registration macro
#define TEST(name, test_case)                                             \
  static void test_case();                                                \
  static test::TestRegistrar test_registrar_##test_case(name, test_case); \
  static void test_case()
}  // namespace test

#endif  // INCLUDE_TEST_FRAMEWORK_HPP_
