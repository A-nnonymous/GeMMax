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

#include <cuda.h>          // NOLINT
#include <cuda_runtime.h>  // NOLINT
#include <functional>
#include <iostream>
#include <vector>
#include "paddle/extension.h"

void dispatch_trivial_gemm(const paddle::Tensor& lhs,
                           const paddle::Tensor& lhs_scale,
                           const paddle::Tensor& rhs,
                           const paddle::Tensor& rhs_scale,
                           const bool is_lhs_1d_scaled,
                           const bool is_rhs_1d_scaled,
                           paddle::Tensor& out  // NOLINT
) {
  using gemm_func_t = std::function<void()>;
  gemm_func_t gemm_func;

  auto trivial_gemm_1dx1d_scaled = [&]() {
    std::cout << "1dx1d_scaled" << std::endl;
    return;  // not-implemented yet
  };
  auto trivial_gemm_1dx2d_scaled = [&]() {
    std::cout << "1dx2d_scaled" << std::endl;
    return;  // not-implemented yet
  };
  auto trivial_gemm_2dx1d_scaled = [&]() {
    std::cout << "2dx1d_scaled" << std::endl;
    return;  // not-implemented yet
  };
  auto trivial_gemm_2dx2d_scaled = [&]() {
    std::cout << "2dx2d_scaled" << std::endl;
    return;  // not-implemented yet
  };

  gemm_func = is_lhs_1d_scaled
                  ? (is_rhs_1d_scaled ? (gemm_func_t)trivial_gemm_1dx1d_scaled
                                      : (gemm_func_t)trivial_gemm_1dx2d_scaled)
                  : (is_rhs_1d_scaled ? (gemm_func_t)trivial_gemm_2dx1d_scaled
                                      : (gemm_func_t)trivial_gemm_2dx2d_scaled);
  gemm_func();
  return;
}

std::vector<paddle::Tensor> trivial_gemm(const paddle::Tensor& lhs,
                                         const paddle::Tensor& lhs_scale,
                                         const paddle::Tensor& rhs,
                                         const paddle::Tensor& rhs_scale,
                                         const bool is_lhs_1d_scaled,
                                         const bool is_rhs_1d_scaled) {
  // ----------------- Arguments check-------------------
  auto check_args = [&]() -> bool {
    // ----------- Type check ----------
    PADDLE_ENFORCE_EQ(
        lhs.is_dense_tensor() && rhs.is_dense_tensor(),
        true,
        common::errors::InvalidArgument("Input tensors must be dense tensors"));

    PADDLE_ENFORCE_EQ(lhs.dtype(),
                      rhs.dtype(),
                      common::errors::InvalidArgument(
                          "The dtype of lhs must be same as rhs"));

    PADDLE_ENFORCE_EQ(lhs.dtype(),
                      paddle::DataType::FLOAT8_E4M3FN,
                      common::errors::InvalidArgument(
                          "The dtype of lhs must be float8_e4m3fn"));

    PADDLE_ENFORCE_EQ(
        lhs_scale.dtype(),
        paddle::DataType::FLOAT32,
        common::errors::InvalidArgument("lhs_scale must be float32"));

    PADDLE_ENFORCE_EQ(
        rhs_scale.dtype(),
        paddle::DataType::FLOAT32,
        common::errors::InvalidArgument("rhs_scale must be float32"));

    // ------------- Shape check --------------
    const size_t m = lhs.shape()[0];
    const size_t n = rhs.shape()[0];

    if (m == 0 or n == 0) return true;  // NOLINT

    PADDLE_ENFORCE_EQ(
        lhs.dims().size(),
        2,
        common::errors::InvalidArgument("lhs must be 2D tensor, but got %d-D",
                                        lhs.dims().size()));

    PADDLE_ENFORCE_EQ(
        rhs.dims().size(),
        2,
        common::errors::InvalidArgument("rhs must be 2D tensor, but got %d-D",
                                        rhs.dims().size()));

    PADDLE_ENFORCE_EQ(lhs.shape()[1],
                      rhs.shape()[1],
                      common::errors::InvalidArgument(
                          "Matrix multiplication(NT) dimension mismatch: "
                          "lhs.shape[1]=%d != rhs.shape[1]=%d",
                          lhs.shape()[1],
                          rhs.shape()[1]));

    if (is_lhs_1d_scaled) {
      const size_t quanted_shape1 = (lhs.shape()[1] + 127) / 128;
      PADDLE_ENFORCE_EQ(
          lhs_scale.dims().size(),
          2,
          common::errors::InvalidArgument("lhs_scale must be 2D"));
      PADDLE_ENFORCE_EQ(lhs_scale.shape()[0],
                        lhs.shape()[0],
                        common::errors::InvalidArgument(
                            "lhs_scale size[0] mismatch: expected %d, got %d",
                            lhs.shape()[0],
                            lhs_scale.shape()[0]));
      PADDLE_ENFORCE_EQ(lhs_scale.shape()[1],
                        quanted_shape1,
                        common::errors::InvalidArgument(
                            "lhs_scale size[1] illegal: expected %d, got %d",
                            quanted_shape1,
                            lhs_scale.shape()[0]));
    } else {
      const size_t quanted_shape0 = (lhs.shape()[0] + 127) / 128;
      const size_t quanted_shape1 = (lhs.shape()[1] + 127) / 128;
      PADDLE_ENFORCE_EQ(lhs_scale.shape()[0],
                        (lhs.shape()[0] + 127) / 128,
                        common::errors::InvalidArgument(
                            "lhs_scale size[0] illegal: expected %d, got %d",
                            quanted_shape0,
                            lhs_scale.shape()[0]));
      PADDLE_ENFORCE_EQ(lhs_scale.shape()[1],
                        (lhs.shape()[1] + 127) / 128,
                        common::errors::InvalidArgument(
                            "lhs_scale size[1] illegal: expected %d, got %d",
                            quanted_shape1,
                            lhs_scale.shape()[0]));
    }

    if (is_rhs_1d_scaled) {
      const size_t quanted_shape1 = (lhs.shape()[1] + 127) / 128;
      PADDLE_ENFORCE_EQ(
          rhs_scale.dims().size(),
          2,
          common::errors::InvalidArgument("rhs_scale must be 2D"));
      PADDLE_ENFORCE_EQ(rhs_scale.shape()[0],
                        rhs.shape()[0],
                        common::errors::InvalidArgument(
                            "rhs_scale size mismatch: expected %d, got %d",
                            rhs.shape()[0],
                            rhs_scale.shape()[0]));
    } else {
      const size_t quanted_shape0 = (rhs.shape()[0] + 127) / 128;
      const size_t quanted_shape1 = (rhs.shape()[1] + 127) / 128;
      PADDLE_ENFORCE_EQ(rhs_scale.shape()[0],
                        (rhs.shape()[0] + 127) / 128,
                        common::errors::InvalidArgument(
                            "rhs_scale size[0] illegal: expected %d, got %d",
                            quanted_shape0,
                            lhs_scale.shape()[0]));
      PADDLE_ENFORCE_EQ(rhs_scale.shape()[1],
                        (rhs.shape()[1] + 127) / 128,
                        common::errors::InvalidArgument(
                            "rhs_scale size[1] illegal: expected %d, got %d",
                            quanted_shape1,
                            lhs_scale.shape()[0]));
    }

    // ----------- Place check ------------
    PADDLE_ENFORCE_EQ(lhs.place(),
                      rhs.place(),
                      common::errors::InvalidArgument(
                          "lhs and rhs must be on the same device"));

    PADDLE_ENFORCE_EQ(lhs.place(),
                      lhs_scale.place(),
                      common::errors::InvalidArgument(
                          "lhs and lhs_scale must be on the same device"));

    PADDLE_ENFORCE_EQ(rhs.place(),
                      rhs_scale.place(),
                      common::errors::InvalidArgument(
                          "rhs and rhs_scale must be on the same device"));

    // ----------- Space check -----------
    const size_t total_output_elements = m * n;
    constexpr size_t bytes_per_output_element = 2;  // bfloat16
    const size_t total_bytes = total_output_elements * bytes_per_output_element;

    PADDLE_ENFORCE_LE(total_bytes,
                      static_cast<size_t>(1) << 36,  // 64GB limit
                      common::errors::ResourceExhausted(
                          "Output tensor too large: %d bytes", total_bytes));
    return false;
  };  // check_args

  auto allocate_outputs = [&]() {
    const auto m = lhs.shape()[0];
    const auto n = rhs.shape()[0];
    return paddle::empty({m, n}, paddle::DataType::BFLOAT16, lhs.place());
  };  // allocate_outputs

  auto launch = [&](auto& outputs) -> std::vector<paddle::Tensor> {
    dispatch_trivial_gemm(lhs,
                          lhs_scale,
                          rhs,
                          rhs_scale,
                          is_lhs_1d_scaled,
                          is_rhs_1d_scaled,
                          outputs);
    return {outputs};
  };  // launch

  try {
    auto is_0size = check_args();
    auto outputs = allocate_outputs();
    if (is_0size) {
      return {outputs};
    }
    return launch(outputs);
  } catch (const std::exception& e) {
    PADDLE_THROW(common::errors::Fatal("trivial_gemm failed: %s", e.what()));
  }  // try-catches
}

PD_BUILD_OP(trivial_gemm)
    .Inputs({"lhs", "lhs_scale", "rhs", "rhs_scale"})
    .Outputs({"out"})
    .Attrs({"is_lhs_1d_scaled: bool", "is_rhs_1d_scaled: bool"})
    .SetKernelFn(PD_KERNEL(trivial_gemm));
