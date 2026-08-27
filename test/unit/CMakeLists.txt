# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

add_executable(pto-ir-translator-unit-test
  PTOIRTranslatorTest.cpp
)
target_include_directories(pto-ir-translator-unit-test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(pto-ir-translator-unit-test PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  PTOIR
  PTOTransforms
  LLVMSupport
)
add_test(NAME pto_ir_translator_unit COMMAND pto-ir-translator-unit-test)
set_tests_properties(pto_ir_translator_unit PROPERTIES LABELS "PTO;unit")

add_executable(pto-canonical-sync-analysis-unit-test
  CanonicalSyncAnalysisTest.cpp
)
target_include_directories(pto-canonical-sync-analysis-unit-test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(pto-canonical-sync-analysis-unit-test PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  PTOIR
  PTOTransforms
  LLVMSupport
)
add_test(NAME pto_canonical_sync_analysis_unit
  COMMAND pto-canonical-sync-analysis-unit-test
)
set_tests_properties(pto_canonical_sync_analysis_unit
  PROPERTIES LABELS "PTO;unit")

add_executable(pto-sync-cover-graph-unit-test
  SyncCoverGraphTest.cpp
)
target_link_libraries(pto-sync-cover-graph-unit-test PRIVATE
  PTOSyncCoverCore
)
add_test(NAME pto_sync_cover_graph_unit
  COMMAND pto-sync-cover-graph-unit-test
)
set_tests_properties(pto_sync_cover_graph_unit PROPERTIES LABELS "PTO;unit")

add_executable(pto-sync-cover-coverage-unit-test
  SyncCoverCoverageTest.cpp
)
target_link_libraries(pto-sync-cover-coverage-unit-test PRIVATE
  PTOSyncCoverCore
)
add_test(NAME pto_sync_cover_coverage_unit
  COMMAND pto-sync-cover-coverage-unit-test
)
set_tests_properties(pto_sync_cover_coverage_unit PROPERTIES LABELS "PTO;unit")
