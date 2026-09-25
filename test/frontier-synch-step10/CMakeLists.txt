# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
# the License for details. You may not use this file except in compliance with the License. THIS
# SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
# License.
cmake_minimum_required(VERSION 3.20)
project(FrontierSynchStep10 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(PTO_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_executable(frontier-step10-core core.cpp)
target_include_directories(frontier-step10-core PRIVATE "${PTO_SOURCE_DIR}/include")
target_compile_options(frontier-step10-core PRIVATE -Wall -Wextra -Werror -UNDEBUG)
option(STEP10_SANITIZE "Run address and undefined-behavior instrumentation" OFF)
if(STEP10_SANITIZE)
  target_compile_options(frontier-step10-core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(frontier-step10-core PRIVATE -fsanitize=address,undefined)
endif()
enable_testing()
add_test(NAME frontier-step10-core COMMAND frontier-step10-core)

# Native tests are a main-project target (pto-frontier-exact-test), so they use
# PTOTransforms' actual transitive LLVM/MLIR dependencies. Do not relink a
# separately discovered static archive with a guessed library closure.
