# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.20)
project(PTOLogicalRelationTests LANGUAGES C CXX)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
if(NOT LLVM_VERSION_MAJOR EQUAL 19)
  message(FATAL_ERROR "The native relation parity suite requires the pinned LLVM 19 build")
endif()
get_filename_component(PTO_RELATION_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_executable(pto-logical-relations-test relations_driver.cpp
  "${PTO_RELATION_ROOT}/lib/PTO/Transforms/InsertSync/LogicalSyncRelations.cpp")
target_compile_features(pto-logical-relations-test PRIVATE cxx_std_17)
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(pto-logical-relations-test PRIVATE -fno-rtti)
endif()
target_include_directories(pto-logical-relations-test PRIVATE
  "${PTO_RELATION_ROOT}/include" ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(pto-logical-relations-test PRIVATE MLIRPresburger LLVMSupport)
