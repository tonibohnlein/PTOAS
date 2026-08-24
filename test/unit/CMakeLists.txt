# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_executable(pto-schedule-graph-unit-test
  ScheduleGraphTest.cpp
)
target_include_directories(pto-schedule-graph-unit-test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
)
add_test(NAME pto_schedule_graph_unit
  COMMAND pto-schedule-graph-unit-test
)
set_tests_properties(pto_schedule_graph_unit PROPERTIES LABELS "PTO;unit")

add_executable(pto-sync-cover-graph-unit-test
  SyncCoverGraphTest.cpp
  "${PROJECT_SOURCE_DIR}/lib/PTO/Transforms/CanonicalSync/SyncCoverGraph.cpp"
)
target_include_directories(pto-sync-cover-graph-unit-test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
)
add_test(NAME pto_sync_cover_graph_unit
  COMMAND pto-sync-cover-graph-unit-test
)
set_tests_properties(pto_sync_cover_graph_unit PROPERTIES LABELS "PTO;unit")

add_executable(pto-sync-cover-coverage-unit-test
  SyncCoverCoverageTest.cpp
  "${PROJECT_SOURCE_DIR}/lib/PTO/Transforms/CanonicalSync/SyncCoverCoverage.cpp"
  "${PROJECT_SOURCE_DIR}/lib/PTO/Transforms/CanonicalSync/SyncCoverGraph.cpp"
)
target_include_directories(pto-sync-cover-coverage-unit-test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
)
add_test(NAME pto_sync_cover_coverage_unit
  COMMAND pto-sync-cover-coverage-unit-test
)
set_tests_properties(pto_sync_cover_coverage_unit PROPERTIES LABELS "PTO;unit")

add_executable(pto-canonical-sync-algorithms-unit-test
  CanonicalSyncAlgorithmsTest.cpp
  "${PROJECT_SOURCE_DIR}/lib/PTO/Transforms/CanonicalSync/CanonicalSyncColoring.cpp"
  "${PROJECT_SOURCE_DIR}/lib/PTO/Transforms/CanonicalSync/CanonicalSyncCriticalPath.cpp"
  "${PROJECT_SOURCE_DIR}/lib/PTO/Transforms/CanonicalSync/CanonicalSyncToken.cpp"
  "${PROJECT_SOURCE_DIR}/lib/PTO/Transforms/CanonicalSync/CompletionReduction.cpp"
)
target_include_directories(pto-canonical-sync-algorithms-unit-test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
)
add_test(NAME pto_canonical_sync_algorithms_unit
  COMMAND pto-canonical-sync-algorithms-unit-test
)
set_tests_properties(pto_canonical_sync_algorithms_unit
  PROPERTIES LABELS "PTO;unit")

add_executable(pto-canonical-sync-ownership-verification-unit-test
  CanonicalSyncOwnershipVerificationTest.cpp
)
target_include_directories(pto-canonical-sync-ownership-verification-unit-test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/lib/PTO/Transforms/CanonicalSync"
)
target_link_libraries(pto-canonical-sync-ownership-verification-unit-test PRIVATE
  MLIRIR
  PTOTransforms
  LLVMSupport
)
add_test(NAME pto_canonical_sync_ownership_verification_unit
  COMMAND pto-canonical-sync-ownership-verification-unit-test
)
set_tests_properties(pto_canonical_sync_ownership_verification_unit
  PROPERTIES LABELS "PTO;unit")
