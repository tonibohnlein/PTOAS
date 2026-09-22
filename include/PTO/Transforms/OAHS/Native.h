// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_NATIVE_H
#define PTO_TRANSFORMS_OAHS_NATIVE_H
#include "PTO/Transforms/OAHS/Analysis.h"
#include "PTO/IR/SyncProtocolModel.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>
namespace mlir::pto::oahs {
struct SelectedPlan;
struct SelectedOptions;
struct NativeAnalysis {
  Program program;
  // Preserved lowering-owned protocols; handles refer to unchanged original IR.
  llvm::SmallVector<SyncProtocolModel, 0> protocols;
  AnalysisResult analysis;
  // Analytical-phase to original operation mapping, valid while the caller
  // keeps the IR unchanged. Periodic effects can map several phases to the
  // same instruction; these are not cloned native payload operations.
  llvm::SmallVector<mlir::Operation *> phases;
  // Original SSA roots indexed by Cell::storageOrigins. Equal numeric local
  // addresses may have several roots; root identity alone is not disjointness.
  llvm::SmallVector<Value> storageRoots;
  // Original command anchors. Null entries are unavailable synthetic decisions.
  llvm::SmallVector<mlir::Operation *> cuts;
  // Representative reachable cut per phase (NoControlId if unreachable);
  // refined observations can have
  // additional sites. Inspect program.observed for all qualified occurrences.
  llvm::SmallVector<Cut> phaseCuts;
  std::vector<std::string> observationNotes;
};
// Import and analyze the unsynchronized native program without changing its IR.
// Success means analysis completed, not that residual synchronization is
// absent.
LogicalResult analyzeHandoffSync(func::FuncOp function, NativeAnalysis &result);
// Compatibility entry point: performs the same analysis and discards its
// report.
LogicalResult analyzeHandoffSync(func::FuncOp function);
// Uses selected-plan construction with the causal frontier, shared production
// translation/alias analysis, and SyncCodegen. Failure
// leaves the original function unchanged; no backend fallback is performed.
LogicalResult runHandoffSync(func::FuncOp function);
namespace testing {
// Test/report entry for the same selected constructor and checker as
// runHandoffSync, including import, transaction, emission and reconstruction.
// The optional report describes construction; LogicalResult additionally covers
// emission/reconstruction. No alternative constructor or fallback is used.
LogicalResult runSelectedHandoffSyncWithMutation(
    func::FuncOp function, llvm::function_ref<void(func::FuncOp)> mutate = {},
    SelectedPlan *report = nullptr, const SelectedOptions *options = nullptr);

// Import with the same qualified observation policy used by the selected
// constructor. This is a read-only diagnostic/test entry point: it selects no
// commands and grants no completion or event credit.
LogicalResult analyzeSelectedHandoffSync(func::FuncOp function,
                                         NativeAnalysis &result, bool classInvariantInputs = false,
                                         bool firstWriteConsumers = false, bool finalReadSources = false);

// Exercise arithmetic read-back directly, without the earlier whole-IR identity
// gate masking decoder failures. This test hook grants no observation/target
// qualification and is never used for production acceptance.
LogicalResult checkHandoffObservationPredicate(
    const OriginalObservation &observation, mlir::Operation *anchor,
    llvm::ArrayRef<std::pair<std::size_t, mlir::Operation *>> originalLoopOwners,
    Value condition);
// Mutation occurs only on the private working copy, before reconstruction.
LogicalResult
runHandoffSyncWithMutation(func::FuncOp function,
                           llvm::function_ref<void(func::FuncOp)> mutate);
} // namespace testing
} // namespace mlir::pto::oahs
#endif
