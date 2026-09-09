# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(PTOLogicalRelationTests LANGUAGES C CXX)
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)
endif()
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

if(TARGET pto-logical-sync-test AND PTO_ENABLE_PYTHON_BINDING)
  set(_oahs_command "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_focused.py"
    --relation-driver "$<TARGET_FILE:pto-logical-relations-test>"
    --occurrence-driver "$<TARGET_FILE:pto-sync-occurrences-test>"
    --native-driver "$<TARGET_FILE:pto-logical-sync-test>"
    --candidates-driver "$<TARGET_FILE:pto-sync-candidates-test>"
    --slot-mapping-driver "$<TARGET_FILE:pto-sync-slots-test>"
    --opt "$<TARGET_FILE:pto-test-opt>"
    --python-root "${CMAKE_BINARY_DIR}/python"
    --output-root "${CMAKE_BINARY_DIR}/test-results/oahs")
  add_test(NAME oahs_focused COMMAND ${_oahs_command})
  set_tests_properties(oahs_focused PROPERTIES LABELS "OAHS" RUN_SERIAL TRUE TIMEOUT 600)
  add_custom_target(check-oahs-focused
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -R "^oahs_focused$" --parallel 1
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    DEPENDS pto-logical-relations-test pto-sync-occurrences-test pto-logical-sync-test pto-sync-candidates-test pto-sync-slots-test
            pto-test-opt PTOAS_Python PTOASCompiler
    USES_TERMINAL)
endif()
