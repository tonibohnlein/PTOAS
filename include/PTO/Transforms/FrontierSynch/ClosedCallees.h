// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Read-only recognition of delegation through independently closed invocations.
// This proves the wrapper shape, not callee closure or allocation feasibility.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_CLOSEDCALLEES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_CLOSEDCALLEES_H
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
namespace mlir::pto::frontiersynch {
struct ClosedCalleeCandidate {
    RecognitionResult result;
    SmallVector<func::FuncOp> callees;
    unsigned activeCores = 0;
};
struct ClosedCalleeModule {
    DenseMap<Operation*, ClosedCalleeCandidate> wrappers;
    // A callee cannot inherit distinct-formal-root assumptions from its caller:
    // two actual arguments may be the same pointer, or overlapping views.
    DenseSet<Operation*> calledFunctions;
    DenseMap<Operation*, SmallVector<Operation*>> callers;
};
ClosedCalleeModule recognizeClosedCallees(ModuleOp module);
// Compile leaves first, then accept wrappers only after closure is established.
// Ordinary direct calls and original physical-core guards remain in the IR.
using PrepareCallee = llvm::function_ref<FailureOr<std::unique_ptr<PreparedLogicalPlan>>(func::FuncOp, GMAliasPolicy)>;
LogicalResult insertModuleSynchronization(ModuleOp module, GMAliasPolicy policy, PrepareCallee prepare);
} // namespace mlir::pto::frontiersynch
#endif
