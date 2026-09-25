# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(pto-test-opt
  pto-test-opt.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(pto-test-opt PRIVATE
  PTOIR
  PTOTransforms
  MLIRMlirOptMain
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
)

add_dependencies(pto-test-opt
  PTOOpsIncGen
  PTOPassesIncGen
)

add_llvm_executable(pto-vpto-scheduler-tracker-test
  pto-vpto-scheduler-tracker-test.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(pto-vpto-scheduler-tracker-test PRIVATE
  PTOIR
  PTOTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_dependencies(pto-vpto-scheduler-tracker-test
  PTOOpsIncGen
  PTOPassesIncGen
)

add_llvm_executable(pto-vfsimt-size-patcher-test
  pto-vfsimt-size-patcher-test.cpp
  PARTIAL_SOURCES_INTENDED
)

target_include_directories(pto-vfsimt-size-patcher-test PRIVATE
  "${CMAKE_SOURCE_DIR}/tools/ptoas"
)

target_link_libraries(pto-vfsimt-size-patcher-test PRIVATE
  PTOASVFSIMTSizePatcher
  LLVMCore
  LLVMIRReader
  LLVMSupport
  MLIRSupport
)

add_llvm_executable(pto-bisheng-scheduler-test
  pto-bisheng-scheduler-test.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(pto-bisheng-scheduler-test PRIVATE PTOASBishengScheduler)

add_llvm_executable(pto-frontier-analysis-test
  pto-frontier-analysis-test.cpp
  FactoredProvenanceNativeTests.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-analysis-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
)
add_dependencies(pto-frontier-analysis-test PTOOpsIncGen PTOPassesIncGen)

# The production original-point/control core is intentionally MLIR-independent.
add_llvm_executable(pto-frontier-interval-core-test
  pto-frontier-interval-core-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_include_directories(pto-frontier-interval-core-test PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/lib/PTO/Transforms/FrontierSynch"
)

add_llvm_executable(pto-frontier-interval-test
  pto-frontier-interval-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-interval-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
)
add_dependencies(pto-frontier-interval-test PTOOpsIncGen PTOPassesIncGen)
# The same production expression transfer can be checked without an MLIR runtime.
add_llvm_executable(pto-factored-provenance-test
  pto-factored-provenance-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_include_directories(pto-factored-provenance-test PRIVATE "${CMAKE_SOURCE_DIR}/include")

# Phase A semantic tests use production records and always-active assertions.
add_llvm_executable(pto-frontier-obligations-test
  ../../test/standalone/frontier_obligations_test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_include_directories(pto-frontier-obligations-test PRIVATE "${CMAKE_SOURCE_DIR}/include")
add_llvm_executable(pto-frontier-value-arithmetic-test
  ../../test/frontier-synch-step5/arithmetic.cpp
  PARTIAL_SOURCES_INTENDED
)
target_include_directories(pto-frontier-value-arithmetic-test PRIVATE "${CMAKE_SOURCE_DIR}/include")
add_llvm_executable(pto-frontier-value-test
  ../../test/frontier-synch-step5/native.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-value-test PRIVATE
  MLIRIR MLIRParser MLIRSupport MLIRFuncDialect MLIRArithDialect MLIRSCFDialect
)
add_dependencies(pto-frontier-value-test PTOOpsIncGen)
add_llvm_executable(pto-frontier-lifetime-test
  pto-frontier-lifetime-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-lifetime-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
)
add_dependencies(pto-frontier-lifetime-test PTOOpsIncGen PTOPassesIncGen)

# Step 7: one portable source-DAG oracle and the actual native Phase A adapter.
add_llvm_executable(pto-frontier-d1-core-test
  ../../test/standalone/frontier_d1_sources_test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_include_directories(pto-frontier-d1-core-test PRIVATE "${CMAKE_SOURCE_DIR}/include")
add_llvm_executable(pto-frontier-d1-test
  pto-frontier-d1-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-d1-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
)
add_dependencies(pto-frontier-d1-test PTOOpsIncGen PTOPassesIncGen)
add_llvm_executable(pto-frontier-periodic-core-test
  ../../test/frontier-synch-step8/core.cpp
  PARTIAL_SOURCES_INTENDED
)
target_include_directories(pto-frontier-periodic-core-test PRIVATE "${CMAKE_SOURCE_DIR}/include")
add_llvm_executable(pto-frontier-periodic-test
  ../../test/frontier-synch-step8/native.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-periodic-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
)
add_dependencies(pto-frontier-periodic-test PTOOpsIncGen PTOPassesIncGen)
# D4 uses the production factored transfer and a qualified original-use algebra.
add_llvm_executable(pto-frontier-d4-test
  pto-frontier-d4-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_include_directories(pto-frontier-d4-test PRIVATE "${CMAKE_SOURCE_DIR}/include")
add_llvm_executable(pto-frontier-d4-native-test
  pto-frontier-d4-native-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-d4-native-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
)
add_dependencies(pto-frontier-d4-native-test PTOOpsIncGen PTOPassesIncGen)
# Step 10: pure structural algebra and public native original-boundary queries.
add_llvm_executable(pto-frontier-exact-core-test
  ../../test/frontier-synch-step10/core.cpp
  PARTIAL_SOURCES_INTENDED
)
target_include_directories(pto-frontier-exact-core-test PRIVATE "${CMAKE_SOURCE_DIR}/include")
add_llvm_executable(pto-frontier-exact-test
  ../../test/frontier-synch-step10/native.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-exact-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
)
add_dependencies(pto-frontier-exact-test PTOOpsIncGen PTOPassesIncGen)
# I.3 uses the same production core in portable and native interface checks.
add_llvm_executable(pto-frontier-covering-core-test
  pto-frontier-covering-core-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_include_directories(pto-frontier-covering-core-test PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/lib/PTO/Transforms/FrontierSynch"
)
add_llvm_executable(pto-frontier-covering-test
  pto-frontier-covering-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-covering-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
)
add_dependencies(pto-frontier-covering-test PTOOpsIncGen PTOPassesIncGen)
add_llvm_executable(pto-frontier-descriptor-core-test
  ../../test/standalone/frontier_descriptor_slots.cpp
  PARTIAL_SOURCES_INTENDED
)
target_include_directories(pto-frontier-descriptor-core-test PRIVATE "${CMAKE_SOURCE_DIR}/include")
add_llvm_executable(pto-frontier-request-test
  pto-frontier-request-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-request-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
)
add_dependencies(pto-frontier-request-test PTOOpsIncGen PTOPassesIncGen)

add_llvm_executable(pto-frontier-preparation-test
  pto-frontier-preparation-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-preparation-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
)
add_dependencies(pto-frontier-preparation-test PTOOpsIncGen PTOPassesIncGen)
