// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

// Ref
// https://github.com/HazyResearch/flash-attention/blob/main/csrc/flash_attn/src/static_switch.h
// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, ({
///     some_function<BoolConst>(...);
/// }));
/// ```
/// We need "({" and "})" to make sure that the code is a single argument being
/// passed to the macro.
#define BOOL_SWITCH(COND, CONST_NAME, CODE) \
  if (COND) {                               \
    constexpr bool CONST_NAME = true;       \
    CODE;                                   \
  } else {                                  \
    constexpr bool CONST_NAME = false;      \
    CODE;                                   \
  }

// modified from BOOL_SWITCH
// because MSVC cannot handle std::conditional with constexpr variable
#define FP16_SWITCH(COND, CODE)      \
  if (COND) {                        \
    using elem_type = __nv_bfloat16; \
    CODE;                            \
  } else {                           \
    using elem_type = __half;        \
    CODE;                            \
  }\
