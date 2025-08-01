# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest
from typing import Tuple

import paddle
from paddle.incubate.fp8 import deep_gemm
from paddle.incubate.fp8.deep_gemm import (
    calc_diff,
    ceil_div,
    get_col_major_tma_aligned_tensor,
)
from paddle import Tensor
from paddle.distributed.fleet.utils.timer_helper import _GPUEventTimer as Timer


class TestDeepGemm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        paddle.seed(0)
        random.seed(0)
        cls.rmse_threshold = 7e-4

    def per_token_cast_to_fp8(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        self.assertEqual(x.dim(), 2)
        self.assertEqual(x.shape[1] % 128, 0)
        m, n = x.shape
        x_view = paddle.view(x, (m, -1, 128))
        x_abs = paddle.abs(x_view).astype(paddle.float32)
        x_amax = paddle.amax(x_abs, axis=2)
        x_amax = paddle.view(x_amax, (m, -1))
        x_amax = paddle.clip(x_amax, min=1e-4)
        scaled_x = x_view * (448.0 / x_amax.unsqueeze(2))
        scaled_x_converted = paddle.view(scaled_x.astype(paddle.float8_e4m3fn), (m, n))

        x_amax_scaled = paddle.view((x_amax / 448.0), (m, -1))

        result = (scaled_x_converted, x_amax_scaled)
        return result

    def per_block_cast_to_fp8(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        self.assertEqual(x.dim(), 2)
        m, n = x.shape
        x_padded = paddle.zeros(
            (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype
        )
        x_padded[:m, :n] = x
        x_view = paddle.view(x_padded, (-1, 128, x_padded.shape[1] // 128, 128))

        x_abs = paddle.abs(x_view).astype(paddle.float32)
        x_amax = paddle.amax(x_abs, axis=(1, 3), keepdim=True)
        x_amax = paddle.clip(x_amax, min=1e-4)
        x_scaled = (x_view * (448.0 / x_amax)).astype(paddle.float8_e4m3fn)

        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
            paddle.view(x_amax / 448.0, (x_view.shape[0], x_view.shape[2]))
        )

    def construct(
        self, m: int, k: int, n: int
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor, Tensor]:
        x = paddle.randn((m, k), dtype=paddle.bfloat16)
        y = paddle.randn((n, k), dtype=paddle.bfloat16)
        out = paddle.empty((m, n), dtype=paddle.bfloat16)
        ref_out = x @ y.t()

        x_fp8, y_fp8 = self.per_token_cast_to_fp8(x), self.per_block_cast_to_fp8(y)
        # Transpose earlier so that the testing will not trigger transposing kernels
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
        return x_fp8, y_fp8, out, ref_out

    def construct_grouped(
        self, num_groups: int, m: int, k: int, n: int, is_masked: bool
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor, Tensor]:
        x = paddle.randn((num_groups, m, k), dtype=paddle.bfloat16)
        y = paddle.randn((num_groups, n, k), dtype=paddle.bfloat16)
        out = paddle.empty((num_groups, m, n), dtype=paddle.bfloat16)
        ref_out = paddle.einsum("gmk,gnk->gmn", x, y)

        self.assertEqual(m % 4, 0, f"TMA alignment error: {m}")
        x_fp8 = (
            paddle.empty_like(x, dtype=paddle.float8_e4m3fn),
            paddle.empty((num_groups, m, k // 128), dtype=paddle.float32),
        )
        y_fp8 = (
            paddle.empty_like(y, dtype=paddle.float8_e4m3fn),
            paddle.empty(
                (num_groups, (n + 127) // 128, k // 128), dtype=paddle.float32
            ),
        )
        for i in range(num_groups):
            x_fp8_0_i, x_fp8_1_i = self.per_token_cast_to_fp8(x[i])
            paddle.assign(x_fp8_0_i, x_fp8[0][i])
            paddle.assign(x_fp8_1_i, x_fp8[1][i])
            y_fp8_0_i, y_fp8_1_i = self.per_block_cast_to_fp8(y[i])
            paddle.assign(y_fp8_0_i, y_fp8[0][i])
            paddle.assign(y_fp8_1_i, y_fp8[1][i])

        # For non-masked input, we must merge the group and M dims
        if not is_masked:
            x_fp8 = (
                paddle.view(x_fp8[0], (-1, k)),
                self.per_token_cast_to_fp8(paddle.view(x, (-1, k)))[1],
            )
            out, ref_out = paddle.view(out, (-1, n)), paddle.view(ref_out, (-1, n))

        # Transpose earlier so that the testing will not trigger transposing kernels
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
        return x_fp8, y_fp8, out, ref_out

    def performance_test(self, func, *args, test_name="", reps=[100, 1000, 10000]):
        """Performance stability test with multiple repetitions"""
        times = {}

        for rep in reps:
            timer = Timer("perf")
            timer.start()
            for _ in range(rep):
                func(*args)
            timer.stop()
            avg_time = timer.elapsed(reset=False) / rep
            times[rep] = avg_time
            print(f"{test_name} - {rep} reps: {avg_time * 1000:.6f} ms/op")

        # Check performance stability (variance should be reasonable)
        if len(reps) >= 2:
            time_values = list(times.values())
            max_time = max(time_values)
            min_time = min(time_values)
            variance = (max_time - min_time) / min_time
            print(f"{test_name} - Performance variance: {variance:.2%}")
            # Allow up to 50% variance between different repetition counts
            self.assertLess(
                variance, 0.5, f"Performance variance too high: {variance:.2%}"
            )

    def test_gemm_correctness(self):
        """Test basic GEMM correctness"""
        print("Testing GEMM correctness:")
        for m in (64, 128, 4096):
            for k, n in [
                (7168, 2112),
                (1536, 24576),
                (512, 32768),
                (16384, 7168),
                (7168, 4096),
                (2048, 7168),
            ]:
                with self.subTest(m=m, k=k, n=n):
                    x_fp8, y_fp8, out, ref_out = self.construct(m, k, n)
                    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
                    diff = calc_diff(out, ref_out)
                    self.assertLess(
                        diff,
                        self.rmse_threshold,
                        f"RMSE too high: m={m}, k={k}, n={n}, diff={diff:.8f}",
                    )
        print("GEMM correctness test passed\n")

    def test_gemm_performance(self):
        """Test basic GEMM performance stability"""
        print("Testing GEMM performance:")
        # Test with a representative case
        m, k, n = 4096, 7168, 4096
        x_fp8, y_fp8, out, ref_out = self.construct(m, k, n)

        def gemm_op():
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

        self.performance_test(gemm_op, test_name=f"GEMM({m}x{k}x{n})")
        print("GEMM performance test passed\n")

    def test_m_grouped_gemm_contiguous_correctness(self):
        """Test grouped contiguous GEMM correctness"""
        print("Testing grouped contiguous GEMM correctness:")

        for num_groups, m, k, n in (
            (8, 4096, 7168, 4096),
            (8, 4096, 2048, 7168),
            (4, 8192, 2048, 7168),
            (4, 8192, 7168, 4096),
        ):
            with self.subTest(num_groups=num_groups, m=m, k=k, n=n):
                x_fp8, y_fp8, out, ref_out = self.construct_grouped(
                    num_groups, m, k, n, is_masked=False
                )
                m_indices = paddle.arange(0, num_groups, dtype=paddle.int32)
                m_indices = paddle.flatten(
                    paddle.expand(
                        paddle.unsqueeze(m_indices, -1), shape=[num_groups, m]
                    )
                )
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    x_fp8, y_fp8, out, m_indices
                )
                diff = calc_diff(out, ref_out)
                self.assertLess(
                    diff,
                    self.rmse_threshold,
                    f"RMSE too high: m={m * num_groups}, k={k}, n={n}, diff={diff:.8f}",
                )
        print("Grouped contiguous GEMM correctness test passed\n")

    def test_m_grouped_gemm_contiguous_performance(self):
        """Test grouped contiguous GEMM performance stability"""
        print("Testing grouped contiguous GEMM performance:")
        # Test with a representative case
        num_groups, m, k, n = 8, 4096, 7168, 4096
        x_fp8, y_fp8, out, ref_out = self.construct_grouped(
            num_groups, m, k, n, is_masked=False
        )
        m_indices = paddle.arange(0, num_groups, dtype=paddle.int32)
        m_indices = paddle.flatten(
            paddle.expand(paddle.unsqueeze(m_indices, -1), shape=[num_groups, m])
        )

        def grouped_gemm_op():
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                x_fp8, y_fp8, out, m_indices
            )

        self.performance_test(
            grouped_gemm_op, test_name=f"GroupedGEMM({num_groups}x{m}x{k}x{n})"
        )
        print("Grouped contiguous GEMM performance test passed\n")

    def test_m_grouped_gemm_masked_correctness(self):
        """Test grouped masked GEMM correctness"""
        print("Testing grouped masked GEMM correctness:")

        for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
            for k, n in ((7168, 4096), (2048, 7168)):
                with self.subTest(num_groups=num_groups, m=m, k=k, n=n):
                    # Test correctness
                    masked_m_candidates = list(
                        filter(
                            lambda candidate: candidate <= m,
                            (64, 128, 192, 256, 320, 384),
                        )
                    )
                    for i in range(5):  # Reduced iterations for unit test
                        x_fp8, y_fp8, out, ref_out = self.construct_grouped(
                            num_groups, m, k, n, is_masked=True
                        )
                        masked_m = paddle.empty((num_groups,), dtype=paddle.int32)
                        for j in range(num_groups):
                            masked_m[j] = random.choice(masked_m_candidates)

                        masked_m_float = paddle.cast(masked_m, "float32")
                        masked_m_mean = paddle.mean(masked_m_float)
                        masked_m_mean_int = paddle.cast(masked_m_mean, "int32")
                        expected_m = min(int(masked_m_mean_int + 1), m)

                        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                            x_fp8, y_fp8, out, masked_m, expected_m
                        )

                        for j in range(num_groups):
                            diff = calc_diff(
                                out[j, : masked_m[j].item()],
                                ref_out[j, : masked_m[j].item()],
                            )
                            self.assertLess(
                                diff,
                                self.rmse_threshold,
                                f"RMSE too high: m={m}, k={k}, n={n}, j={j}, "
                                f"masked_m={masked_m[j]}, num_groups={num_groups}, diff={diff:.8f}",
                            )
        print("Grouped masked GEMM correctness test passed\n")

    def test_m_grouped_gemm_masked_performance(self):
        """Test grouped masked GEMM performance stability"""
        print("Testing grouped masked GEMM performance:")
        # Test with a representative case
        num_groups, m, k, n = 4, 256, 7168, 4096
        x_fp8, y_fp8, out, ref_out = self.construct_grouped(
            num_groups, m, k, n, is_masked=True
        )

        # Fixed masked_m for consistent performance testing
        masked_m = paddle.to_tensor([128, 192, 256, 192], dtype=paddle.int32)
        expected_m = 256

        def masked_gemm_op():
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                x_fp8, y_fp8, out, masked_m, expected_m
            )

        self.performance_test(
            masked_gemm_op, test_name=f"MaskedGEMM({num_groups}x{m}x{k}x{n})"
        )
        print("Grouped masked GEMM performance test passed\n")


if __name__ == "__main__":
    # Enable GPU if available
    if paddle.device.cuda.device_count() > 0:
        paddle.device.set_device("gpu:0")

    unittest.main(verbosity=2)
