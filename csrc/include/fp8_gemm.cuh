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
#pragma once

#include "jit_tma_impl.cuh"
#include "jit_tma_warp_spec_impl.cuh"
#include "jit_tma_warp_spec_sched_hyper_tuned_impl.cuh"
#include "jit_tma_warp_spec_sched_impl.cuh"
#include "jit_tma_warp_spec_sched_tuned_impl.cuh"
#include "jit_trivial_impl.cuh"
#include "trivial_impl.cuh"

template <typename Derived>
class GemmHelper {
 protected:
  using GemmFunc = std::function<void(const paddle::Tensor&,
                                      const paddle::Tensor&,
                                      const paddle::Tensor&,
                                      paddle::Tensor&)>;

  void dispatch_impl(const paddle::Tensor& lhs,
                     const paddle::Tensor& lhs_scale,
                     const paddle::Tensor& rhs,
                     const paddle::Tensor& rhs_scale,
                     bool is_lhs_1d_scaled,
                     bool is_rhs_1d_scaled,
                     paddle::Tensor& out) {
    auto select_impl = [this](bool lhs_1d, bool rhs_1d) -> GemmFunc {
      if (lhs_1d) {
        return rhs_1d ? static_cast<Derived*>(this)->impl_1dx1d
                      : static_cast<Derived*>(this)->impl_1dx2d;
      }
      return rhs_1d ? static_cast<Derived*>(this)->impl_2dx1d
                    : static_cast<Derived*>(this)->impl_2dx2d;
    };
    select_impl(is_lhs_1d_scaled, is_rhs_1d_scaled)(lhs, rhs, out);
  }
};
