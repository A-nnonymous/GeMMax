# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import multiprocessing
import os


def run(func):
    p = multiprocessing.Process(target=func)
    p.start()
    p.join()


def change_pwd():
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


def setup_gemmax():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    change_pwd()
    setup(
        name="gemmax",
        ext_modules=CUDAExtension(
            sources=[
                "csrc/trivial_gemm.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-w", "-Wno-abi", "-fPIC", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "-DCUTE_ARCH_MMA_SM90A_ENABLE",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-lineinfo",
                    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
                    "-gencode=arch=compute_90a,code=sm_90a",
                    "-DNDEBUG",
                ],
            },
        ),
    )


run(setup_gemmax)
