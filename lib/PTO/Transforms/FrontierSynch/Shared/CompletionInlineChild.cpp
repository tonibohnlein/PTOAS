// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve a qualified exact child's barriers through the common materializer.
#include "CompletionInlineChild.h"
#include "CompletionStorage.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
namespace mlir::pto::frontiersynch {
std::function<std::shared_ptr<const CompletionInlineChild>()> completionChildExport(
    std::shared_ptr<FiniteAnalysis> finite)
{
    if (!finite) { return {}; }
    using Result = std::shared_ptr<const CompletionInlineChild>;
    auto retained = std::make_shared<std::optional<Result>>();
    return [finite = std::move(finite), retained]() -> Result {
        if (!*retained) { *retained = CompletionInlineChild::qualify(finite); }
        return **retained;
    };
}
namespace {
bool contains(const DenseSet<Operation*>& roots, Operation* operation)
{
    for (auto* op = operation; op; op = op->getParentOp()) {
        if (roots.contains(op)) { return true; }
    }
    return false;
}
bool eventFree(const PreparedLogicalPlan& plan)
{
    if (plan.completeInvocation || plan.privateState || plan.completionOwner || plan.compactBoundingOwner ||
        plan.physicalOutput || plan.executedOutput || plan.deferredOutput) { return false; }
    if (llvm::any_of(plan.families, [](const auto& family) {
        return !family.local || family.sourceChoices || family.targetChoices;
    })) { return false; }
    if (llvm::any_of(plan.endpoints, [](const auto& endpoint) {
        return endpoint.kind != LogicalCommandKind::Barrier;
    })) { return false; }
    if (plan.familyRecipe) {
        auto shapes = familyMatchingShapes(*plan.familyRecipe);
        if (failed(shapes) || llvm::any_of(*shapes, [](const auto& entry) {
            return entry.second.hasSet || entry.second.hasWait;
        })) { return false; }
    }
    return true;
}
}
std::shared_ptr<const CompletionInlineChild> CompletionInlineChild::qualify(std::shared_ptr<FiniteAnalysis> finite)
{
    if (!finite || !finite->originalInput() || !finite->error().empty() || finite->originalInput()->wholeFunction() ||
        finite->originalInput()->roots().empty()) { return {}; }
    auto plan = finite->prepare();
    if (failed(plan) || !eventFree(**plan) || !(*plan)->endpoints.empty() ||
        !(*plan)->preparation.empty() || (*plan)->familyProtocol) { return {}; }
    auto fragment = LogicalFamilyFragment::capture(**plan);
    if (!fragment) { return {}; }
    auto child = std::shared_ptr<CompletionInlineChild>(new CompletionInlineChild);
    child->finite = std::move(finite);
    child->fragment = std::move(fragment);
    uint8_t modes = 0;
    const auto& input = *child->finite->originalInput();
    for (auto* root : input.roots()) {
        root->walk([&](Operation* op) {
            for (const auto* phase : input.index().phasesFor(op)) {
                if (!phase->useVec.empty()) { modes |= 1; }
                if (!phase->defVec.empty()) { modes |= 2; }
            }
        });
    }
    // Internal barriers remain in the implementation. The coarse import need
    // not reconstruct which effects they complete; it conservatively carries all.
    child->effects = {{modes}, {modes}, {modes}, true};
    return child;
}
CompletionSummary CompletionInlineChild::classifiedSummary(
    const CompletionStorage& storage, const PhaseIndex& index) const
{
    // Proper completion children use the same full-input class interpretation.
    // Finite children retain their exact code but import conservative effects.
    return completion ? effects : storage.summarize(roots(), index);
}
ArrayRef<Operation*> CompletionInlineChild::roots() const
{
    return completion ? completion->originalRoots() : finite->originalInput()->roots();
}
const SyncInput* CompletionInlineChild::input() const
{
    return completion ? completion->inputOwner().get() : &finite->originalInput()->storage();
}
std::shared_ptr<const CompletionInlineChild> CompletionInlineChild::qualify(
    std::shared_ptr<const CompletionRegion> completion)
{
    if (!completion || completion->completesInvocation() || completion->summaryOnly ||
        completion->originalRoots().empty()) { return {}; }
    auto child = std::shared_ptr<CompletionInlineChild>(new CompletionInlineChild);
    child->effects = completion->transfer().summary;
    child->completion = std::move(completion);
    return child;
}
LogicalResult CompletionInlineChild::append(PreparedLogicalPlan& destination, IRMapping& mapping,
    SmallVectorImpl<Operation*>& completionCuts, DenseSet<const CompletionSourceSnapshot*>& checkedSources,
    DenseSet<const ClosedCompletionChild*>& checkedCalls) const
{
    if (completion) { return completion->append(destination, mapping, completionCuts, checkedSources, checkedCalls); }
    auto fresh = fragment->instantiate();
    auto& plan = *fresh;
    if (failed(materializeLogicalFamilies(plan)) || !eventFree(plan)) { return failure(); }
    DenseSet<Operation*> rootSet(roots().begin(), roots().end());
    for (const auto& stage : plan.preparation) {
        // The common materializer can hoist safe scalar work outside the
        // child interval. Its certified original placement must still belong
        // to this invocation; only command cuts are confined to child roots.
        if (!stage.before || stage.before->getParentOfType<func::FuncOp>() !=
            finite->originalInput()->function()) { return failure(); }
        auto* cut = mapping.lookupOrNull(stage.before);
        if (!cut) { return failure(); }
        auto& block = destination.addPreparation(cut);
        for (auto& operation : *stage.code) { block.push_back(operation.clone(mapping)); }
    }
    for (auto endpoint : plan.endpoints) {
        if (!contains(rootSet, endpoint.before)) { return failure(); }
        endpoint.before = mapping.lookupOrNull(endpoint.before);
        endpoint.guard = mapping.lookupOrDefault(endpoint.guard);
        if (!endpoint.before) { return failure(); }
        // No notification matching remains. Keep exact guards and command order;
        // optional exclusion/provenance metadata is not needed for these barriers.
        endpoint.domain.clear();
        endpoint.records.clear();
        endpoint.piece = -1;
        endpoint.record = destination.endpoints.size();
        destination.endpoints.push_back(std::move(endpoint));
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
