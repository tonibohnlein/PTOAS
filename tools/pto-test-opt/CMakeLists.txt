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

add_llvm_executable(pto-protocol-sync-one-shot-test
  pto-protocol-sync-one-shot-test.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(pto-protocol-sync-one-shot-test PRIVATE
  PTOIR
  PTOTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
)

add_dependencies(pto-protocol-sync-one-shot-test
  PTOOpsIncGen
  PTOPassesIncGen
)

add_llvm_executable(pto-protocol-sync-ready-release-test
  pto-protocol-sync-ready-release-test.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(pto-protocol-sync-ready-release-test PRIVATE
  PTOIR
  PTOTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
)

add_dependencies(pto-protocol-sync-ready-release-test
  PTOOpsIncGen
  PTOPassesIncGen
)

add_llvm_executable(pto-protocol-sync-direct-repair-test
  pto-protocol-sync-direct-repair-test.cpp
  PARTIAL_SOURCES_INTENDED
)

target_link_libraries(pto-protocol-sync-direct-repair-test PRIVATE
  PTOIR
  PTOTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
)

add_dependencies(pto-protocol-sync-direct-repair-test
  PTOOpsIncGen
  PTOPassesIncGen
)

add_llvm_executable(pto-protocol-sync-mixed-test
  pto-protocol-sync-mixed-test.cpp
  PARTIAL_SOURCES_INTENDED
)

add_llvm_executable(pto-protocol-sync-local-memory-test
  pto-protocol-sync-local-memory-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-protocol-sync-local-memory-test PRIVATE
  PTOIR PTOTransforms MLIRArithDialect MLIRFuncDialect MLIRIR MLIRParser MLIRSCFDialect MLIRSupport
)
add_dependencies(pto-protocol-sync-local-memory-test PTOOpsIncGen PTOPassesIncGen)

add_llvm_executable(pto-protocol-sync-scoreboard-test
  pto-protocol-sync-scoreboard-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-protocol-sync-scoreboard-test PRIVATE
  PTOIR PTOTransforms MLIRArithDialect MLIRFuncDialect MLIRIR MLIRParser MLIRSCFDialect MLIRSupport
)
add_dependencies(pto-protocol-sync-scoreboard-test PTOOpsIncGen PTOPassesIncGen)

target_link_libraries(pto-protocol-sync-mixed-test PRIVATE
  PTOIR
  PTOTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
)

add_dependencies(pto-protocol-sync-mixed-test
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
