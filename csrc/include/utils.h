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
#include "paddle/extension.h"

#define DEFAULT_THROW(NAME, TYPE)                           \
  default:                                                  \
    do {                                                    \
      PD_THROW(#NAME, " not implemented for '", TYPE, "'"); \
    } while (0);                                            \
    break
#define DISPATCH_REAL_TYPE(TYPEIN, ...)      \
  switch (TYPEIN) {                          \
    case paddle::DataType::FLOAT32: {        \
      using paddle_t = float;                \
      __VA_ARGS__;                           \
      break;                                 \
    }                                        \
    case paddle::DataType::FLOAT16: {        \
      using paddle_t = phi::dtype::float16;  \
      __VA_ARGS__;                           \
      break;                                 \
    }                                        \
    case paddle::DataType::BFLOAT16: {       \
      using paddle_t = phi::dtype::bfloat16; \
      __VA_ARGS__;                           \
      break;                                 \
    }                                        \
      DEFAULT_THROW(NAME, TYPEIN);           \
  }
