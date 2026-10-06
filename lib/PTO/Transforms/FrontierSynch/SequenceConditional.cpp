// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Guard original arm interfaces; mutually exclusive arms may share the sequence
// composer because all query and boundary membership is masked by arm presence.
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
namespace {
void maskSelectors(RegionalStorageSelectors& selectors, RegionExpressions& arena, Expr guard)
{
    auto mask = [&](auto& values) {
        for (auto& value : values) { value.present = arena.select(guard, value.present, arena.boolean(false)); }
    };
    mask(selectors.firstWriters); mask(selectors.lastWriters);
    for (auto& [pipe, values] : selectors.firstReaders) { mask(values); }
    for (auto& [pipe, values] : selectors.lastReaders) { mask(values); }
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> maskPlan(
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepared,
    const std::shared_ptr<RegionExpressions>& arena, Expr guard)
{
    if (failed(prepared)) { return failure(); }
    auto& plan = **prepared;
    plan.completeInvocation = false;
    // Original arm cuts already suppress commands when the arm is not taken.
    // Allocation summaries also need that presence; copy borrowed summaries.
    if (plan.regionalAllocation) {
        plan.regionalAllocation = std::make_shared<RegionalAllocationSummary>(*plan.regionalAllocation);
        for (auto& group : plan.regionalAllocation->groups) {
            for (auto& member : group.members) {
                member.active = arena->select(guard, member.active, arena->boolean(false));
            }
        }
    }
    return std::move(*prepared);
}
RegionalAnalysis guardedArm(RegionalAnalysis body, Expr guard)
{
    auto owner = std::make_shared<const RegionalAnalysis>(std::move(body));
    auto out = *owner;
    auto arena = owner->expressions;
    auto mask = [&](auto& values) {
        for (auto& value : values) { value.present = arena->select(guard, value.present, arena->boolean(false)); }
    };
    for (auto& boundary : out.storageBoundary) {
        mask(boundary.firstWriters); mask(boundary.lastWriters);
        for (auto& [pipe, values] : boundary.firstReaders) { mask(values); }
        for (auto& [pipe, values] : boundary.lastReaders) { mask(values); }
    }
    for (auto* boundaries : {&out.accessBoundary, &out.deferredAccessBoundary}) {
        for (auto& boundary : *boundaries) {
            boundary.first.present = arena->select(guard, boundary.first.present, arena->boolean(false));
            boundary.last.present = arena->select(guard, boundary.last.present, arena->boolean(false));
        }
    }
    for (auto& [pipe, values] : out.firstPayloads) { mask(values); }
    for (auto& [pipe, values] : out.lastPayloads) { mask(values); }
    out.presence = [owner, arena, guard](RegionalEvent event) -> std::optional<Expr> {
        auto present = regionalPresence(*owner, event);
        return present ? std::optional<Expr>(arena->select(guard, *present, arena->boolean(false))) : std::nullopt;
    };
    out.reachability = [owner, arena, guard](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
        auto answer = regionalReachability(*owner, a, b);
        return answer ? std::optional<Expr>(arena->select(guard, *answer, arena->boolean(false))) : std::nullopt;
    };
    if (owner->storageSelectors) {
        out.storageSelectors = [owner, arena, guard](RegionalByteAddress address) {
            auto selectors = owner->storageSelectors(address);
            if (selectors) { maskSelectors(*selectors, *arena, guard); }
            return selectors;
        };
    }
    if (owner->prepare) {
        out.prepare = [owner, arena, guard]() { return maskPlan(owner->prepare(), arena, guard); };
    }
    if (owner->prepareWithVisits) {
        out.prepareWithVisits = [owner, arena, guard](ArrayRef<scf::ForOp> loops) {
            return maskPlan(owner->prepareWithVisits(loops), arena, guard);
        };
    }
    if (owner->prepareFiltered) {
        out.prepareFiltered = [owner, arena, guard](const RegionalDemandFilter& filter) {
            return maskPlan(owner->prepareFiltered(filter), arena, guard);
        };
    }
    out.capabilities.contextualGuards = true;
    return out;
}
} // namespace
bool SequenceAnalysisState::conditionalChild(const StructureNode& node)
{
    auto branch = dyn_cast_or_null<scf::IfOp>(node.anchor);
    if (!branch) { return fail("conditional region requires an scf.if operation"); }
    const bool requiresPrerequisite = index.needsValuePrerequisite(branch);
    if (branch.getNumResults() || requiresPrerequisite) {
        return fail("conditional region requires mapped control prerequisites and result-free arms");
    }
    auto condition = expressions.input(branch.getCondition());
    for (auto id : node.children) {
        const auto& arm = program->nodes[id];
        if (!arm.payloadCount) { continue; }
        auto analysis = analyzeSequenceRegion(function, *input, *program, id, arena, indexOwner);
        if (!analysis.error.empty()) { return fail("conditional arm: " + analysis.error); }
        auto body = sequenceRegionalResult(analysis);
        bool thenArm = arm.region == &branch.getThenRegion();
        auto guard = thenArm ? condition : negate(condition);
        Child child;
        child.regional = guardedArm(std::move(body), guard);
        child.anchors = child.regional.anchors;
        children.push_back(std::move(child));
    }
    return true;
}
} // namespace mlir::pto::frontiersynch
