// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- MixedLoopFrontier.cpp - Complete native serial-loop alternative -----===//
// Keep this alternative atomic, including boundary handoffs and the exit drain.
// Canonical local requirements replace timeline rejection only under its
// complete order certificate; all nonlocal effects still pass F interpretation.

#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto::protocol_sync;

namespace {
bool sameEdges(ArrayRef<SyncLoopFrontierEdge> first, ArrayRef<SyncLoopFrontierEdge> second)
{
    return first.size() == second.size() && llvm::equal(first, second, [](const auto& a, const auto& b) {
               return std::tie(a.source, a.target, a.sourcePipe, a.targetPipe, a.eventId) ==
                      std::tie(b.source, b.target, b.sourcePipe, b.targetPipe, b.eventId);
           });
}

bool sameRequirements(ArrayRef<SyncResidualObligation> first, ArrayRef<SyncResidualObligation> second)
{
    return first.size() == second.size() && llvm::equal(first, second, [](const auto& a, const auto& b) {
               return std::tie(
                          a.id, a.localRequirement, a.kind, a.source, a.target, a.sourceAccess, a.targetAccess,
                          a.control, a.iteration.kind, a.iteration.distance, a.iteration.carrier, a.precision,
                          a.generation, a.channel, a.detail) ==
                          std::tie(
                              b.id, b.localRequirement, b.kind, b.source, b.target, b.sourceAccess, b.targetAccess,
                              b.control, b.iteration.kind, b.iteration.distance, b.iteration.carrier, b.precision,
                              b.generation, b.channel, b.detail) &&
                      a.atoms == b.atoms;
           });
}

SyncMixedWorldCost costFor(const SyncLoopFrontierPlan& repair)
{
    SyncMixedWorldCost cost;
    std::map<std::pair<pto::PIPE, pto::PIPE>, unsigned> domains;
    const auto count = [&](const SyncLoopFrontierEdge& edge) {
        if (edge.eventId) {
            ++cost.generatedEventPairs;
            cost.staticActions += 2;
            cost.eventPressure =
                std::max<std::uint64_t>(cost.eventPressure, ++domains[{edge.sourcePipe, edge.targetPipe}]);
        } else {
            ++cost.targetedBarriers;
            ++cost.staticActions;
        }
    };
    for (const auto& edge : repair.edges) {
        count(edge);
    }
    for (const auto& edge : repair.boundaryEdges) {
        count(edge);
    }
    // The backedge has an extra prime/drain pair, or an extra named drain.
    cost.staticActions += repair.edges.back().eventId ? 2 : 1;
    cost.targetedBarriers += repair.edges.back().eventId ? 0 : 1;
    cost.fixedExitDrains = 1;
    ++cost.staticActions;
    return cost;
}
} // namespace

FailureOr<std::optional<SyncMixedProtocolPlan>> mlir::pto::protocol_sync::buildMixedLoopFrontierPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels)
{
    auto repair = buildLoopFrontierRepairPlan(schedule, {}, true);
    if (failed(repair)) {
        return failure();
    }
    if (repair->status != SyncLoopFrontierStatus::Ready) {
        return std::optional<SyncMixedProtocolPlan>{};
    }
    auto world = buildLoopFrontierWorld(schedule);
    if (failed(world)) {
        return failure();
    }
    auto interpreted = interpretSelectedWorld(schedule, stages, timelines, channels, *world);
    if (failed(interpreted)) {
        return failure();
    }
    if (!interpreted->isComplete()) {
        return std::optional<SyncMixedProtocolPlan>{};
    }
    SyncMixedProtocolPlan plan;
    plan.status = SyncMixedPlanStatus::Ready;
    plan.selectedWorldKind = SyncMixedWorldKind::LoopFrontier;
    plan.selectedWorld = std::move(*world);
    plan.selectedCost = costFor(*repair);
    plan.initialResidualCount = repair->requirements.size();
    plan.candidateCountBeforeDeletion = 1;
    plan.loopFrontier = std::move(*repair);
    return std::optional<SyncMixedProtocolPlan>(std::move(plan));
}

LogicalResult mlir::pto::protocol_sync::verifyMixedLoopFrontierPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncMixedProtocolPlan& plan)
{
    auto expected = buildMixedLoopFrontierPlan(schedule, stages, timelines, channels);
    const bool valid = succeeded(expected) && *expected && plan.loopFrontier && !plan.hasProtocol() &&
                       plan.directObligations.empty() && plan.directRepair.candidates.empty() && plan.failures.empty();
    if (!valid) {
        return failure();
    }
    const auto& reference = **expected;
    const auto& a = *plan.loopFrontier;
    const auto& b = *reference.loopFrontier;
    const auto& x = plan.selectedCost;
    const auto& y = reference.selectedCost;
    return success(
        plan.status == reference.status && plan.selectedWorldKind == reference.selectedWorldKind &&
        a.status == b.status && a.loop == b.loop && a.includeBoundaries == b.includeBoundaries &&
        a.requirementCount == b.requirementCount && sameEdges(a.edges, b.edges) &&
        sameEdges(a.boundaryEdges, b.boundaryEdges) && sameRequirements(a.requirements, b.requirements) &&
        plan.initialResidualCount == reference.initialResidualCount && plan.candidateCountBeforeDeletion == 1 &&
        plan.reverseDeletionAttempts == 0 && plan.reverseDeletionRemoved == 0 &&
        plan.selectedWorld.orderedLoop == reference.selectedWorld.orderedLoop &&
        plan.selectedWorld.exitCompletedPhases == reference.selectedWorld.exitCompletedPhases &&
        plan.selectedWorld.completions.empty() && plan.selectedWorld.visibility.empty() &&
        plan.selectedWorld.protocols.empty() &&
        std::tie(x.generatedEventPairs, x.targetedBarriers, x.fixedExitDrains, x.eventPressure, x.staticActions) ==
            std::tie(y.generatedEventPairs, y.targetedBarriers, y.fixedExitDrains, y.eventPressure, y.staticActions));
}
