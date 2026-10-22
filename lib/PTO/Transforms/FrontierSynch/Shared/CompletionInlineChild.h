// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Retained finite implementations with separately qualified notification
// resources. Mathematical exactness alone cannot import eventful children.
#ifndef PTO_COMPLETION_INLINE_CHILD_H
#define PTO_COMPLETION_INLINE_CHILD_H
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionTransfer.h"
#include "mlir/IR/IRMapping.h"
#include <map>
#include "PTO/Transforms/FrontierSynch/Shared/LogicalFamilyFragment.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
namespace mlir::pto::frontiersynch {
struct SequenceAnalysis;
struct BoundedLifetimeDemandResult;
class CompletionSourceSnapshot;
class CompletionStorage;
class ClosedCompletionChild;
struct CompletionChildAssembly {
    uint64_t nextRecord = 0;
    SmallVector<Attribute> allocationGroups;
};
class CompletionInlineChild {
public:
    static std::shared_ptr<const CompletionInlineChild> qualify(std::shared_ptr<FiniteAnalysis> finite);
    static std::shared_ptr<const CompletionInlineChild> qualify(
        std::shared_ptr<FiniteAnalysis> finite, ArrayRef<int64_t> eligibleIds);
    static std::shared_ptr<const CompletionInlineChild> qualify(std::shared_ptr<const CompletionRegion> completion);
    static std::shared_ptr<const CompletionInlineChild> qualify(
        std::shared_ptr<BoundedLifetimeDemandResult> bounded, std::shared_ptr<const SyncInput> input);
    // Provider-only adapter for an exact direct whole-loop protocol. The
    // producer owns its matching proof; this adapter checks scope/resources.
    using VisitPreparation = std::function<FailureOr<std::unique_ptr<PreparedLogicalPlan>>() >;
    static std::shared_ptr<const CompletionInlineChild> qualifyVisit(
        std::shared_ptr<const void> owner, std::shared_ptr<const SyncInput> input,
        scf::ForOp loop, VisitPreparation prepare);
    // Whole-loop confinement supplies boundary order, not payload completion.
    struct PrefixBoundary {
        scf::ForOp invocation;
        std::map<uint32_t, CompletionFacts> pendingByPipe;
        std::map<std::pair<uint32_t, uint32_t>, uint8_t> palette;
    };
    std::optional<PrefixBoundary> prefixBoundary(const CompletionStorage&, const PhaseIndex&) const;
    // Same assignment, translated labels/frames. No new coloring or analysis.
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> preparePrefixImport(ArrayRef<int64_t> eligibleIds) const;
    const CompletionSummary& summary() const { return effects; }
    std::optional<uint64_t> payloadCount() const;
    bool exactRealizedOrder() const;
    bool usesEvents() const;
    bool returnsPendingEvents() const;
    using ResourceFootprint = std::map<std::pair<uint32_t, uint32_t>, uint8_t>;
    // Missing means unknown participating resources, never an empty footprint.
    std::optional<ResourceFootprint> resourceFootprint() const;
    CompletionSummary classifiedSummary(const CompletionStorage& storage, const PhaseIndex& index) const;
    ArrayRef<Operation*> roots() const;
    const SyncInput* input() const;
    LogicalResult append(PreparedLogicalPlan& destination, IRMapping& mapping,
                         SmallVectorImpl<Operation*>& completionCuts,
                         DenseSet<const CompletionSourceSnapshot*>& checkedSources,
                         DenseSet<const ClosedCompletionChild*>& checkedCalls, CompletionChildAssembly& assembly) const;
private:
    std::shared_ptr<FiniteAnalysis> finite;
    std::shared_ptr<const void> visitOwner;
    std::shared_ptr<const SyncInput> visitInput;
    SmallVector<Operation*> visitRoots;
    func::FuncOp visitFunction;
    // Legacy producers may construct detached SSA. Retain their immutable
    // mathematics, not live uses of source SSA across function replacement.
    VisitPreparation visitPreparation;
    SmallVector<Value> enclosingVisits;
    std::shared_ptr<const CompletionRegion> completion;
    CompletionSummary effects;
    std::shared_ptr<const LogicalFamilyFragment> fragment;
    // Within-visit assignment only. Parent CFG must certify completion between
    // eventful visits; matching closure alone does not establish consumption.
    std::optional<PhysicalAllocationPlan> eventAssignment;
};
// A failed qualification is specific to this immutable mathematical owner.
// Upgraded representations publish their own export and never mutate this one.
std::function<std::shared_ptr<const CompletionInlineChild>()> completionChildExport(
    std::shared_ptr<FiniteAnalysis> finite);
std::function<std::shared_ptr<const CompletionInlineChild>()> completionChildExport(
    std::shared_ptr<SequenceAnalysis> sequence);
std::function<std::shared_ptr<const CompletionInlineChild>()> completionChildExport(
    std::shared_ptr<BoundedLifetimeDemandResult> bounded, std::shared_ptr<const SyncInput> input);
} // namespace mlir::pto::frontiersynch
#endif
