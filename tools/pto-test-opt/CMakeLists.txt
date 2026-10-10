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

# Shared effect/pipe extraction is usable independently of dependency analysis.
add_llvm_executable(pto-sync-input-test
  pto-sync-input-test.cpp
  FiniteOverlayChecks.cpp
  IntegerRelationChecks.cpp
  GeneralArithmeticSelectorChecks.cpp
  RepeatedReadOnlyStorageChecks.cpp
  PeriodicSharedAllocationChecks.cpp
  SharedHandoffAllocationChecks.cpp
  FiniteAllocationQueryChecks.cpp
  NumericalHierarchyChecks.cpp
  BoundingRegionalChecks.cpp
  RequirementProvenanceChecks.cpp
  CertifiedPartialReductionChecks.cpp
  ControlOriginDistancesChecks.cpp
  CompactWriterReaderChecks.cpp
  CompactWriterReaderInputChecks.cpp
  CompactLowerFactsChecks.cpp
  PeriodicExcessChecks.cpp
  CompactOrderBoundsChecks.cpp
  BalancedCompactBodyChecks.cpp
  GuardedCompactMatchingChecks.cpp
  ConditionalCompactInputChecks.cpp
  BoundingSequenceChecks.cpp
  BoundaryExcessChecks.cpp
  FiniteRequirementReplacementChecks.cpp
  CompactStorageBoundaryChecks.cpp
  CompactBoundaryRanksChecks.cpp
  BoundingRepetitionChecks.cpp
  RepeatedExcessChecks.cpp
  CompactClassRepetitionChecks.cpp
  CompactBoundingPipelineChecks.cpp
  CompactBoundingAllocationChecks.cpp
  GeneralArithmeticAllocationChecks.cpp
  MixedStrideChecks.cpp
  PhysicalExecutedCounterChecks.cpp
  StorageLaneAllocationChecks.cpp
  BoundedLifetimeProvenanceChecks.cpp
  BoundedLifetimeAllocationChecks.cpp
  ArithmeticStorageSelectorChecks.cpp
  RegionalRelationChecks.cpp
  ArithmeticDemandChecks.cpp
  ArithmeticSelectorChecks.cpp
  ArithmeticProgramDump.cpp
  ProgramRecognitionDump.cpp
  NumericTemplateDump.cpp
  NormalizedControlChecks.cpp
  GuardedPeriodicChecks.cpp
  PeriodicAnalysisDump.cpp
  LogicalEndpointDump.cpp
  LogicalEndpointChecks.cpp
  SyncLogicalInsertionChecks.cpp
  PreparedInsertionChecks.cpp
  PeriodicAllocationDump.cpp
  PeriodicAllocationChecks.cpp
  PeriodicDemandChecks.cpp
  ExplicitAnalysisDump.cpp
  RotatingAnalysisDump.cpp
  RotatingBoundaryChecks.cpp
  NumericalWeightedRepetitionChecks.cpp
  NumericalRepeatedSquaringChecks.cpp
  FiniteVisitChecks.cpp
  FiniteVisitInputChecks.cpp
  RotatingRegionChecks.cpp
  LifetimeStreamChecks.cpp
  ArithmeticPeriodicConversionChecks.cpp
  ResolvedLifetimeStreamChecks.cpp
  GuardedRankChecks.cpp
  RepeatedRegionChecks.cpp
  CarriedScalarChecks.cpp
  ScalarConstraintChecks.cpp
  RepeatedStorageChecks.cpp
  RepeatedPhaseChecks.cpp
  NestedRegionalChecks.cpp
  RegionExpressionChecks.cpp
  ExpressionRelationChecks.cpp
  MixedSymbolicRegionChecks.cpp
  ProgramStructureChecks.cpp
  SyncAliasChecks.cpp
  SyncRegionContractChecks.cpp
  SyncRegionDigitChecks.cpp
  SyncStep0Audit.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-sync-input-test PRIVATE
  MLIRDLTIDialect
  PTOIR PTOTransforms MLIRIR MLIRParser MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRLLVMDialect
)
add_dependencies(pto-sync-input-test PTOOpsIncGen PTOPassesIncGen)

# Exercise primitive classification, normalization and producer contract limits.
add_llvm_executable(pto-arithmetic-recognition-test
  pto-arithmetic-recognition-test.cpp
  PARTIAL_SOURCES_INTENDED
)
target_link_libraries(pto-arithmetic-recognition-test PRIVATE PTOIR PTOTransforms MLIRIR MLIRSupport)
add_dependencies(pto-arithmetic-recognition-test PTOOpsIncGen PTOPassesIncGen)
