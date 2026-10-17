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
namespace mlir { class IRMapping; }
namespace mlir::pto::frontiersynch {
class PhaseIndex;
class ClosedCompletionChild;
struct ClosedCompletionImportContext;
struct PreparedLogicalPlan;
class FiniteAnalysis;
class CompletionInlineChild;
class CompletionSourceSnapshot;

class CompletionRegion {
public:
    // Shared across scopes of one immutable analysis session. Captures source
    // identity and validates the input/index association once.
    static std::shared_ptr<const CompletionSourceSnapshot> captureSource(func::FuncOp function,
        std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error);
    // Proper scopes are adjacent original roots; no implicit invocation drain.
    // Inline children retain exact event-free implementations, not just demands.
    static std::shared_ptr<const CompletionRegion> buildRegion(func::FuncOp function,
        std::shared_ptr<const SyncInput> input, const PhaseIndex& index, ArrayRef<Operation*> roots,
        ArrayRef<std::shared_ptr<const CompletionInlineChild>> children, std::string& error,
        std::shared_ptr<const CompletionSourceSnapshot> source = {});
    static std::shared_ptr<const CompletionRegion> buildWithChildren(func::FuncOp function,
        std::shared_ptr<const SyncInput> input, const PhaseIndex& index,
        ArrayRef<std::shared_ptr<const CompletionInlineChild>> children, std::string& error,
        std::shared_ptr<const CompletionSourceSnapshot> source = {});
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
    struct ChildContracts {
        uint64_t eventFree = 0, completedExit = 0, closedDescendants = 0;
    };
    ChildContracts childContracts() const;
    bool completesInvocation() const { return !summaryOnly && wholeInvocation; }
    struct ConstructionWork {
        uint64_t scopedOperations = 0, scopedPhases = 0, graphVertices = 0;
    };
    const ConstructionWork& constructionWork() const { return work; }
    std::shared_ptr<const CompletionSourceSnapshot> sourceOwner() const { return source; }
    uint64_t inlineChildren() const { return inlinePlans.size(); }
    ArrayRef<Operation*> originalRoots() const { return roots; }
    const CompletionTransferResult& transfer() const { return result; }
    std::shared_ptr<const SyncInput> inputOwner() const { return input; }
    // Proper owners are append-only fragments, never independently inserted.
    // The parent publishes no events. Imported children own their closed IDs;
    // the parent needs no additional physical assignment.
    LogicalResult insert(func::FuncOp function, PreparedLogicalPlan& plan) const;
    bool ownsClosedCall(Operation* call) const;
    bool validateClosedChildren() const;
    bool validateClosedChildren(DenseSet<const ClosedCompletionChild*>& checked) const;
    ArrayRef<func::FuncOp> importedFunctions() const { return childFunctions; }

private:
    friend class CompletionInlineChild;
    LogicalResult append(PreparedLogicalPlan& destination, IRMapping& mapping,
                         SmallVectorImpl<Operation*>& completionCuts,
                         DenseSet<const CompletionSourceSnapshot*>& checkedSources,
                         DenseSet<const ClosedCompletionChild*>& checkedCalls) const;
    static std::shared_ptr<const CompletionRegion> buildImpl(func::FuncOp function,
        std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error, bool existingOnly,
        ClosedCompletionImportContext& context, ArrayRef<Operation*> roots = {},
        ArrayRef<std::shared_ptr<const CompletionInlineChild>> children = {}, bool wholeInvocation = true,
        std::shared_ptr<const CompletionSourceSnapshot> source = {});
    bool wholeInvocation = true;
    SmallVector<Operation*> roots;
    std::vector<std::shared_ptr<const CompletionInlineChild>> inlinePlans;
    bool summaryOnly = false;
    func::FuncOp original;
    std::shared_ptr<const SyncInput> input;
    CompletionTransferResult result;
    ConstructionWork work;
    std::vector<std::shared_ptr<const ClosedCompletionChild>> children;
    DenseSet<Operation*> importedCalls;
    SmallVector<func::FuncOp> childFunctions;
    std::vector<Operation*> cuts;
    std::shared_ptr<const CompletionSourceSnapshot> source;
};
} // namespace mlir::pto::frontiersynch
#endif
