// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Native direct loop path; all nonlocal/SSA/exit obligations still pass F.
#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto::protocol_sync;

namespace {
auto key(const SyncSelectedCompletion& edge)
{
    return std::make_tuple(
        edge.source, edge.target, edge.control, edge.iteration.kind, edge.iteration.distance, edge.iteration.carrier);
}
bool sameWorld(const SyncSelectedWorld& a, const SyncSelectedWorld& b)
{
    return !a.orderedLoop && a.acknowledgedPhases.empty() && a.protocols.empty() && a.visibility.empty() &&
           a.exitCompletedPhases == b.exitCompletedPhases && a.completions.size() == b.completions.size() &&
           llvm::equal(a.completions, b.completions, [](const auto& x, const auto& y) { return key(x) == key(y); });
}
} // namespace

LogicalResult mlir::pto::protocol_sync::appendSelectiveLoopEventGenerations(
    const StructuredSyncIR& schedule, const SyncSelectiveLoopPlan& plan,
    SmallVectorImpl<SyncEventGeneration>& generations)
{
    for (const auto& edge : plan.edges) {
        const auto* phase = schedule.findPhase(edge.completion.source);
        const bool validPhase = phase && phase->iterationDomain.loops.size() == 1;
        if (!validPhase) {
            return failure();
        }
        if (!edge.isEvent()) {
            if (edge.placement.eventId) {
                return failure();
            }
            continue;
        }
        SyncEventGeneration generation;
        generation.id = generations.size();
        generation.kind = SyncEventGenerationKind::DirectRepair;
        generation.core = phase->core;
        generation.sourcePipe = edge.placement.sourcePipe;
        generation.targetPipe = edge.placement.targetPipe;
        generation.setAnchor = edge.placement.source;
        generation.waitAnchor = edge.placement.target;
        generation.recurrenceOwner = phase->iterationDomain.loops.front();
        generation.recurring = true;
        generation.eventId = edge.placement.eventId;
        generations.push_back(generation);
    }
    return success();
}

FailureOr<std::optional<SyncMixedProtocolPlan>> mlir::pto::protocol_sync::buildMixedSelectiveLoopPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels)
{
    auto repair = buildSelectiveLoopRepair(schedule);
    if (failed(repair)) {
        return failure();
    }
    if (!repair->plan) {
        SyncMixedProtocolPlan rejected;
        rejected.status = SyncMixedPlanStatus::Unsupported;
        rejected.selectedWorldKind = SyncMixedWorldKind::SelectiveLoop;
        const auto reason = repair->status == SyncSelectiveLoopStatus::AnalysisLimit ?
                                SyncMixedPlanRejection::SelectiveAnalysisLimit :
                            repair->status == SyncSelectiveLoopStatus::UnprovedTokenContract ?
                                SyncMixedPlanRejection::UnprovedTokenContract :
                                SyncMixedPlanRejection::IncompleteDirectRepair;
        rejected.failures.push_back({reason, repair->detail});
        return std::optional<SyncMixedProtocolPlan>(std::move(rejected));
    }
    SyncInterpretationOptions options;
    options.isolatedLoopIsModeled = true;
    auto result = interpretSelectedWorld(schedule, stages, timelines, channels, repair->plan->world, nullptr, options);
    if (failed(result)) {
        return failure();
    }
    if (!result->isComplete()) {
        return std::optional<SyncMixedProtocolPlan>{};
    }
    SyncMixedProtocolPlan plan;
    plan.status = SyncMixedPlanStatus::Ready;
    plan.selectedWorldKind = SyncMixedWorldKind::SelectiveLoop;
    plan.selectiveLoop = std::move(*repair->plan);
    plan.selectedWorld = plan.selectiveLoop->world;
    plan.initialResidualCount = result->localRequirements;
    plan.candidateCountBeforeDeletion = plan.selectiveLoop->candidateCountBeforeDeletion;
    plan.reverseDeletionAttempts = plan.selectiveLoop->deletionAttempts;
    plan.reverseDeletionRemoved = plan.selectiveLoop->deletionRemoved;
    std::map<std::pair<pto::PIPE, pto::PIPE>, unsigned> domains;
    for (const auto& edge : plan.selectiveLoop->edges) {
        if (edge.isEvent()) {
            ++plan.selectedCost.generatedEventPairs;
            plan.selectedCost.staticActions += edge.completion.iteration.distance == 1 ? 4 : 2;
            const auto pressure = ++domains[{edge.placement.sourcePipe, edge.placement.targetPipe}];
            plan.selectedCost.eventPressure = std::max<std::uint64_t>(plan.selectedCost.eventPressure, pressure);
        } else {
            ++plan.selectedCost.targetedBarriers;
            ++plan.selectedCost.staticActions;
        }
    }
    plan.selectedCost.fixedExitDrains = 1;
    ++plan.selectedCost.staticActions;
    return std::optional<SyncMixedProtocolPlan>(std::move(plan));
}

LogicalResult mlir::pto::protocol_sync::verifyMixedSelectiveLoopPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncMixedProtocolPlan& plan)
{
    auto rebuilt = buildMixedSelectiveLoopPlan(schedule, stages, timelines, channels);
    const bool valid = succeeded(rebuilt) && *rebuilt && !plan.hasProtocol() && !plan.loopFrontier &&
                       !plan.structuredFrontier && plan.directObligations.empty() &&
                       plan.directRepair.candidates.empty();
    if (!valid) {
        return failure();
    }
    auto expected = **rebuilt;
    const bool resourceFailure = plan.status == SyncMixedPlanStatus::ResourceInfeasible ||
                                 plan.status == SyncMixedPlanStatus::AllocationAnalysisLimit;
    if (resourceFailure && failed(allocateMixedProtocolEvents(schedule, expected))) {
        return failure();
    }
    const auto& x = plan.selectedCost;
    const auto& y = expected.selectedCost;
    const bool cost =
        std::tie(x.generatedEventPairs, x.targetedBarriers, x.fixedExitDrains, x.eventPressure, x.staticActions) ==
        std::tie(y.generatedEventPairs, y.targetedBarriers, y.fixedExitDrains, y.eventPressure, y.staticActions);
    const bool failuresMatch = plan.failures.size() == expected.failures.size() &&
                               llvm::equal(plan.failures, expected.failures, [](const auto& a, const auto& b) {
                                   return a.reason == b.reason && a.detail == b.detail;
                               });
    const bool common = cost && failuresMatch && plan.status == expected.status &&
                        plan.allocationFailure == expected.allocationFailure &&
                        plan.selectedWorldKind == expected.selectedWorldKind &&
                        plan.initialResidualCount == expected.initialResidualCount &&
                        plan.candidateCountBeforeDeletion == expected.candidateCountBeforeDeletion &&
                        plan.reverseDeletionAttempts == expected.reverseDeletionAttempts &&
                        plan.reverseDeletionRemoved == expected.reverseDeletionRemoved &&
                        plan.selectiveLoop.has_value() == expected.selectiveLoop.has_value() &&
                        sameWorld(plan.selectedWorld, expected.selectedWorld);
    if (!common || !expected.selectiveLoop) {
        return success(common);
    }
    const auto& a = *plan.selectiveLoop;
    const auto& b = *expected.selectiveLoop;
    const bool edges =
        a.edges.size() == b.edges.size() && llvm::equal(a.edges, b.edges, [](const auto& x, const auto& y) {
            return std::tie(x.placement.source, x.placement.target, x.placement.sourcePipe, x.placement.targetPipe) ==
                       std::tie(
                           y.placement.source, y.placement.target, y.placement.sourcePipe, y.placement.targetPipe) &&
                   key(x.completion) == key(y.completion);
        });
    SmallVector<SyncEventGeneration, 16> generations;
    const bool validAssignment = succeeded(appendSelectiveLoopEventGenerations(schedule, a, generations)) &&
                                 succeeded(verifySyncEventGenerationAssignment(
                                     ProtocolSyncTarget::resolve(schedule.getFunction()), {}, generations));
    if (!validAssignment) {
        return failure();
    }
    return success(
        a.loop == b.loop && edges && a.candidateCountBeforeDeletion == b.candidateCountBeforeDeletion &&
        a.deletionAttempts == b.deletionAttempts && a.deletionRemoved == b.deletionRemoved &&
        sameWorld(a.world, b.world) && sameWorld(plan.selectedWorld, b.world));
}
