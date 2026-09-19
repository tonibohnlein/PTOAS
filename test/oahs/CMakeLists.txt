# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.16)
project(OAHSCoreTests LANGUAGES CXX)
enable_testing()
# Compile the same production core once for every standalone test executable.
add_library(oahs-core-under-test STATIC ../../lib/PTO/Transforms/OAHS/Plan.cpp
  ../../lib/PTO/Transforms/OAHS/CausalFrontier.cpp
  ../../lib/PTO/Transforms/OAHS/SelectedLedger.cpp
  ../../lib/PTO/Transforms/OAHS/SelectedControl.cpp
  ../../lib/PTO/Transforms/OAHS/SelectedReplay.cpp
  ../../lib/PTO/Transforms/OAHS/SelectedGroups.cpp
  ../../lib/PTO/Transforms/OAHS/SelectedAllocation.cpp
  ../../lib/PTO/Transforms/OAHS/SelectedPlan.cpp
  ../../lib/PTO/Transforms/OAHS/CyclicFrontiers.cpp)
target_include_directories(oahs-core-under-test PUBLIC ../../include)
target_compile_features(oahs-core-under-test PUBLIC cxx_std_17)
target_compile_options(oahs-core-under-test PRIVATE -Wall -Wextra -Werror -pedantic)
set(OAHS_TEST_SUITES scaling analysis storage_frontier observed phase replay lifecycle causal_frontier
  selected_plan selected_cyclic selected_update selected_lookahead observation_union bank_occurrence first_use fifo_slot)
foreach(suite ${OAHS_TEST_SUITES})
  add_executable(oahs-${suite}-test ${suite}_test.cpp)
  target_link_libraries(oahs-${suite}-test PRIVATE oahs-core-under-test)
  target_compile_options(oahs-${suite}-test PRIVATE -Wall -Wextra -Werror -pedantic)
  add_test(NAME oahs_${suite} COMMAND oahs-${suite}-test)
endforeach()

# Test-only reference collectors; no Python dependency in the live pass.
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

add_test(NAME oahs_phase_vendor COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/reference/phase_vendor_check.py)

add_executable(oahs-phase-driver phase_driver.cpp)
target_link_libraries(oahs-phase-driver PRIVATE oahs-core-under-test)
target_compile_options(oahs-phase-driver PRIVATE -Wall -Wextra -Werror -pedantic)
add_test(NAME oahs_phase_reference COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/reference/phase_bridge.py --driver $<TARGET_FILE:oahs-phase-driver>)

add_executable(oahs-causal-frontier-driver causal_frontier_driver.cpp)
target_link_libraries(oahs-causal-frontier-driver PRIVATE oahs-core-under-test)
target_compile_options(oahs-causal-frontier-driver PRIVATE -Wall -Wextra -Werror -pedantic)
add_test(NAME oahs_causal_frontier_reference
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/reference/causal_frontier_bridge.py
    --driver $<TARGET_FILE:oahs-causal-frontier-driver>)

add_executable(oahs-selected-plan-driver selected_plan_driver.cpp)
target_link_libraries(oahs-selected-plan-driver PRIVATE oahs-core-under-test)
target_compile_options(oahs-selected-plan-driver PRIVATE -Wall -Wextra -Werror -pedantic)
add_test(NAME oahs_selected_plan_reference
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/reference/selected_plan_bridge.py
    --driver $<TARGET_FILE:oahs-selected-plan-driver>)

# Pure immutable-index regressions can also be compiled without MLIR or the core.
add_executable(oahs-lookahead-index-test lookahead_index_test.cpp)
target_compile_features(oahs-lookahead-index-test PRIVATE cxx_std_17)
target_compile_options(oahs-lookahead-index-test PRIVATE -Wall -Wextra -Werror -pedantic)
add_test(NAME oahs_lookahead_index COMMAND oahs-lookahead-index-test)
