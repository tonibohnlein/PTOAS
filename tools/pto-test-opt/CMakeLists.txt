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

# Inspect shared SyncInput without a second effects/control representation.
add_llvm_executable(pto-frontier-analysis-test
  pto-frontier-analysis-test.cpp
  frontier-structured-driver.cpp
  frontier-sync-target-driver.cpp
  frontier-gm-alias-driver.cpp
  frontier-atomic-store-driver.cpp
  frontier-regional-request-driver.cpp
  frontier-fixed-body-upper-driver.cpp
  frontier-stationary-cell-driver.cpp
  frontier-counted-reader-driver.cpp
  frontier-finite-physical-driver.cpp
  frontier-one-way-repair-driver.cpp
  frontier-periodic-allocation-driver.cpp
  frontier-structured-counter-driver.cpp
  frontier-symbolic-allocation-driver.cpp
  frontier-adjacent-local-upper-driver.cpp
  frontier-general-composition-driver.cpp
  frontier-periodic-nested-query-driver.cpp
  frontier-single-stream-driver.cpp
  frontier-general-counted-driver.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-analysis-test PRIVATE
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRControlFlowDialect
  MLIRControlFlowInterfaces MLIRLoopLikeInterface MLIRAffineDialect MLIRMemRefDialect
)
add_dependencies(pto-frontier-analysis-test PTOOpsIncGen PTOPassesIncGen)

# Exercise explicit demand analysis against an independent finite graph oracle.
add_llvm_executable(pto-frontier-demand-test
  pto-frontier-demand-test.cpp
  frontier-guarded-driver.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-demand-test PRIVATE PTOIR PTOTransforms MLIRIR MLIRSupport)
add_dependencies(pto-frontier-demand-test PTOOpsIncGen PTOPassesIncGen)

# Supplied exact rotating fragments; independent finite physical-access oracle.
add_llvm_executable(pto-frontier-rotating-test
  pto-frontier-rotating-test.cpp
  frontier-periodic-driver.cpp
  frontier-symbolic-driver.cpp
  frontier-signed-driver.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-frontier-rotating-test PRIVATE PTOIR PTOTransforms MLIRIR MLIRSupport)
add_dependencies(pto-frontier-rotating-test PTOOpsIncGen PTOPassesIncGen)
