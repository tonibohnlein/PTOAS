// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Normalize a wrapper with at most one payload-bearing visit without proving
// recurrence of its proper children. Original cuts and coordinates survive.
#include "BoundarySlices.h"
#include "PrefixOccurrence.h"
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/Shared/RepeatedPhases.h"
namespace mlir::pto::frontiersynch {
bool SequenceAnalysisState::singletonBoundaryChild(const StructureNode& node, Expr trips)
{
    auto loop = dyn_cast_or_null<scf::ForOp>(node.anchor);
    const bool supported = loop && node.children.size() == 1 && !index.hasUnprovedCarriedState(loop);
    if (!supported) {
        return false;
    }
    RegionExpressions::Transaction transaction(expressions);
    std::string diagnostic;
    auto slices = collectBoundarySlices(loop, index, expressions, trips, DenseMap<Value, Expr>{}, diagnostic,
        BoundaryControlPolicy::RetainChildControl);
    if (!slices) { return false; }
    PhaseNormalization normalizer(loop, index, expressions);
    const BoundarySlice* active = nullptr;
    SmallVector<std::size_t> selected;
    for (const auto& slice : *slices) {
        SmallVector<std::size_t> parts;
        std::function<bool(std::size_t)> select = [&](std::size_t id) {
            const auto& child = program->nodes[id];
            if (!child.payloadCount) { return true; }
            if (child.kind == StructureKind::Sequence) {
                const bool selectedBody = llvm::none_of(child.children, [&](std::size_t nested) {
                    return program->nodes[nested].kind == StructureKind::Conditional;
                });
                if (selectedBody) { parts.push_back(id); return true; }
                for (auto nested : child.children) { if (!select(nested)) { return false; } }
                return true;
            }
            if (child.kind == StructureKind::Conditional) {
                auto branch = dyn_cast_or_null<scf::IfOp>(child.anchor);
                if (!branch) { return false; }
                auto bound = boundaryGuard(branch.getCondition(), normalizer, slice.bindings, expressions);
                auto known = bound ? expressions.constantValue(*bound) : std::nullopt;
                if (!known) {
                    // Retain opaque, data-dependent control as one complete
                    // proper child. It is never assumed true or replayed.
                    parts.push_back(id);
                    return true;
                }
                const bool mappedControl = !index.hasRelevantResults(branch) && !index.needsValuePrerequisite(branch);
                if (!mappedControl) {
                    return false;
                }
                for (auto arm : child.children) {
                    const bool thenArm = program->nodes[arm].region == &branch.getThenRegion();
                    const bool takeArm = thenArm == (*known != 0);
                    if (takeArm) {
                        if (!select(arm)) { return false; }
                    }
                }
                return true;
            }
            // A proper loop retains its own control, effects and mathematical
            // provider. Do not inspect or unroll its runtime visits here.
            parts.push_back(id);
            return true;
        };
        if (!select(node.children.front())) { return false; }
        if (parts.empty()) { continue; }
        // Two active intervals would require cross-visit storage reasoning.
        if (active || !slice.maximumLength || *slice.maximumLength > 1) { return false; }
        active = &slice;
        selected = std::move(parts);
    }
    if (!active) { return false; }
    // Child requests may populate session caches in this same arena. Retain
    // their expression IDs even if a later capability cannot be exported.
    transaction.commit();
    auto unavailable = [&](const std::string& reason) {
        if (!repeatedAttempt.empty()) { repeatedAttempt += "; "; }
        repeatedAttempt += "singleton boundary: " + reason;
        return false;
    };
    std::vector<RegionalAnalysis> parts;
    for (auto id : selected) {
        auto view = originalRegion(id, diagnostic);
        if (failed(view)) { return unavailable(diagnostic); }
        parts.push_back(std::move(*view));
    }
    SmallVector<scf::ForOp> context(node.loops.begin(), node.loops.end());
    context.push_back(loop);
    auto composed = composeRegionalSequenceWithin(
                function, arena, std::move(parts), true, false, context, index.sourceFacts(), allocationPolicy);
    if (!composed.error.empty()) { return unavailable(composed.error); }
    composed.state->constructionOrigin = "singleton-body";
    composed.state->constructionAnchor = loop.getOperation();
    composed.state->completeInvocation = false;
    composed.state->requiredOuterLoops = context;
    auto body = sequenceRegionalResult(composed);
    const auto prefixBody = body;
    auto lifted = repeatPhasedRegions(function, loop, {std::move(body)}, active->interval.end,
        node.loops, active->interval.begin, uint64_t{1}, {}, allocationPolicy);
    if (!lifted.error.empty()) { return unavailable(lifted.error); }
    // This proof covers every payload-bearing execution of the original loop,
    // unlike an arbitrary clipped one-visit regional view. Original cuts retain
    // branch control and nested-loop resets; there are no intervisit demands.
    liftSingletonPrefixContribution(prefixBody, lifted.regional);
    Child child;
    child.regional = std::move(lifted.regional);
    child.anchors = child.regional.anchors;
    children.push_back(std::move(child));
    return true;
}
} // namespace mlir::pto::frontiersynch
