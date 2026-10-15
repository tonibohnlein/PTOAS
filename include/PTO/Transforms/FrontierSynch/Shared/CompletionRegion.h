// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owned conservative completion evidence; independent of exact demands/queries.
#ifndef PTO_FRONTIERSYNCH_COMPLETIONREGION_H
#define PTO_FRONTIERSYNCH_COMPLETIONREGION_H
#include "PTO/Transforms/FrontierSynch/Shared/CompletionTransfer.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include <memory>
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
class PhaseIndex;
class ClosedCompletionChild;
struct ClosedCompletionImportContext;
struct PreparedLogicalPlan;
class CompletionRegion {
public:
    static std::shared_ptr<const CompletionRegion> build(func::FuncOp function,
        std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error);
    // Explicit conservative request: no fine storage partition or exact
    // backend is constructed. This does not replace a retained exact result.
    static std::shared_ptr<const CompletionRegion> buildCoarse(
        func::FuncOp function, GMAliasPolicy policy, std::string& error);
    // Qualify existing event-free code without inserting completion commands.
    // No executable plan escapes: the summary may retain pending exit effects.
    static FailureOr<CompletionSummary> summarizeEventFree(func::FuncOp function,
        std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error);
    // Qualify a private child with no local events. Calls retain independently
    // closed resource owners; no completion command or executable plan is added.
    static std::shared_ptr<const CompletionRegion> qualifyExistingChild(func::FuncOp function,
        std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error,
        ClosedCompletionImportContext& context);
    const CompletionTransferResult& transfer() const { return result; }
    std::shared_ptr<const SyncInput> inputOwner() const { return input; }
    // The parent publishes no events. Imported children own their closed IDs;
    // the parent needs no additional physical assignment.
    LogicalResult insert(func::FuncOp function, PreparedLogicalPlan& plan) const;
    bool ownsClosedCall(Operation* call) const;
    bool validateClosedChildren() const;
    bool validateClosedChildren(DenseSet<const ClosedCompletionChild*>& checked) const;
    ArrayRef<func::FuncOp> importedFunctions() const { return childFunctions; }

private:
    static std::shared_ptr<const CompletionRegion> buildImpl(func::FuncOp function,
        std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error, bool existingOnly,
        ClosedCompletionImportContext& context);
    bool summaryOnly = false;
    func::FuncOp original;
    std::shared_ptr<const SyncInput> input;
    CompletionTransferResult result;
    std::vector<std::shared_ptr<const ClosedCompletionChild>> children;
    DenseSet<Operation*> importedCalls;
    SmallVector<func::FuncOp> childFunctions;
    std::vector<Operation*> cuts, originalOperations;
    std::string originalText;
    DictionaryAttr moduleAttributes;
};
} // namespace mlir::pto::frontiersynch
#endif
