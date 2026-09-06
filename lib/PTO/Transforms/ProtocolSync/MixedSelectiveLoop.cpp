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

FailureOr<std::optional<SyncMixedProtocolPlan>> mlir::pto::protocol_sync::buildMixedSelectiveLoopPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels)
{
    auto repair = buildSelectiveLoopRepair(schedule);
    if (failed(repair)) {
        return failure();
    }
    if (!*repair) {
        return std::optional<SyncMixedProtocolPlan>{};
    }
    SyncInterpretationOptions options;
    options.isolatedLoopIsModeled = true;
    auto result = interpretSelectedWorld(schedule, stages, timelines, channels, (**repair).world, nullptr, options);
    if (failed(result)) {
        return failure();
    }
    if (!result->isComplete()) {
        return std::optional<SyncMixedProtocolPlan>{};
    }
    SyncMixedProtocolPlan plan;
    plan.status = SyncMixedPlanStatus::Ready;
    plan.selectedWorldKind = SyncMixedWorldKind::SelectiveLoop;
    plan.selectiveLoop = std::move(**repair);
    plan.selectedWorld = plan.selectiveLoop->world;
    plan.initialResidualCount = result->localRequirements;
    plan.candidateCountBeforeDeletion = plan.selectiveLoop->edges.size();
    std::map<std::pair<pto::PIPE, pto::PIPE>, unsigned> domains;
    for (const auto& edge : plan.selectiveLoop->edges) {
        if (edge.placement.eventId) {
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
    const bool valid = succeeded(rebuilt) && *rebuilt && plan.selectiveLoop && !plan.hasProtocol() &&
                       !plan.loopFrontier && !plan.structuredFrontier && plan.directObligations.empty() &&
                       plan.directRepair.candidates.empty() && plan.failures.empty();
    if (!valid) {
        return failure();
    }
    const auto& expected = **rebuilt;
    const auto& a = *plan.selectiveLoop;
    const auto& b = *expected.selectiveLoop;
    const auto& x = plan.selectedCost;
    const auto& y = expected.selectedCost;
    const bool cost =
        std::tie(x.generatedEventPairs, x.targetedBarriers, x.fixedExitDrains, x.eventPressure, x.staticActions) ==
        std::tie(y.generatedEventPairs, y.targetedBarriers, y.fixedExitDrains, y.eventPressure, y.staticActions);
    const bool edges =
        a.edges.size() == b.edges.size() && llvm::equal(a.edges, b.edges, [](const auto& x, const auto& y) {
            return std::tie(
                       x.placement.source, x.placement.target, x.placement.sourcePipe, x.placement.targetPipe,
                       x.placement.eventId) ==
                       std::tie(
                           y.placement.source, y.placement.target, y.placement.sourcePipe, y.placement.targetPipe,
                           y.placement.eventId) &&
                   key(x.completion) == key(y.completion);
        });
    return success(
        plan.status == expected.status && plan.selectedWorldKind == expected.selectedWorldKind && a.loop == b.loop &&
        edges && cost && plan.allocationFailure == SyncEventAllocationFailure::None &&
        plan.initialResidualCount == expected.initialResidualCount &&
        plan.candidateCountBeforeDeletion == expected.candidateCountBeforeDeletion &&
        plan.reverseDeletionAttempts == 0 && plan.reverseDeletionRemoved == 0 && sameWorld(a.world, b.world) &&
        sameWorld(plan.selectedWorld, b.world));
}
