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
#include "paddle/extension.h"

void dispatch_trivial_gemm(const paddle::Tensor& lhs,
                           const paddle::Tensor& rhs,
                           paddle::Tensor& out  // NOLINT
) {
  return;
}

std::vector<paddle::Tensor> gemm_nt(const paddle::Tensor& lhs,
                                    const paddle::Tensor& rhs) {
  const int m = lhs.shape()[0];  // lhs [m, k]
  const int k = lhs.shape()[1];
  const int n = rhs.shape()[1];  // rhs [k, n]

  PADDLE_ENFORCE_EQ(
      lhs.dtype(),
      rhs.dtype(),
      common::errors::InvalidArgument("The dtype of lhs must be same as rhs"));

  PADDLE_ENFORCE_EQ(k,
                    rhs.shape()[0],
                    common::errors::InvalidArgument(
                        "The shape 0 of rhs must equal to shape 1 of lhs."));

  //------------------------ 输出四张量 ------------------------
  auto out = paddle::empty({m, n}, lhs.dtype(), lhs.place());

  dispatch_trivial_gemm(lhs, rhs, out);

  return {out};
}
PD_BUILD_OP(trivial_gemm)
    .Inputs({"lhs", "rhs"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(trivial_gemm));
