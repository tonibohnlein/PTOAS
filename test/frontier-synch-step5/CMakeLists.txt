# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
# the License for details. You may not use this file except in compliance with the License. THIS
# SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
# License.
# Focused standalone test tree; no alteration of the user's existing build.
cmake_minimum_required(VERSION 3.20)
project(FrontierSynchStep5 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(PTO_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_executable(frontier-step5-arithmetic arithmetic.cpp)
target_include_directories(frontier-step5-arithmetic PRIVATE "${PTO_SOURCE_DIR}/include")
# Assertions are the test oracle even with a Release LLVM/PTO build.
target_compile_options(frontier-step5-arithmetic PRIVATE -UNDEBUG)
enable_testing()
add_test(NAME frontier-step5-arithmetic COMMAND frontier-step5-arithmetic)
option(STEP5_NATIVE "Build native MLIR endpoint tests" OFF)
if(STEP5_NATIVE)
  if(NOT PTO_BUILD_DIR)
    message(FATAL_ERROR "Set PTO_BUILD_DIR to the prepared PTOAS build (generated PTO headers)")
  endif()
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)
  add_executable(frontier-step5-native native.cpp)
  target_include_directories(frontier-step5-native PRIVATE
    "${PTO_SOURCE_DIR}/include" "${PTO_BUILD_DIR}/include"
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  target_compile_options(frontier-step5-native PRIVATE -UNDEBUG)
  add_definitions(${LLVM_DEFINITIONS})
  target_link_libraries(frontier-step5-native PRIVATE MLIRParser MLIRIR
    MLIRArithDialect MLIRSCFDialect MLIRFuncDialect MLIRSupport)
  add_test(NAME frontier-step5-native COMMAND frontier-step5-native)
endif()
