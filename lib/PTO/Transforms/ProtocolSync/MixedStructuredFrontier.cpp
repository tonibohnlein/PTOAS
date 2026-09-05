// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- MixedStructuredFrontier.cpp - Atomic balanced structured alternative ===//
#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"
#include "llvm/ADT/STLExtras.h"
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {
bool sameRequirements(ArrayRef<SyncResidualObligation> a, ArrayRef<SyncResidualObligation> b)
{
    return a.size() == b.size() && llvm::equal(a, b, [](const auto& x, const auto& y) {
               return std::tie(
                          x.id, x.localRequirement, x.kind, x.source, x.target, x.sourceAccess, x.targetAccess,
                          x.control, x.participation, x.iteration.kind, x.iteration.distance, x.iteration.carrier,
                          x.precision, x.generation, x.channel, x.detail) ==
                          std::tie(
                              y.id, y.localRequirement, y.kind, y.source, y.target, y.sourceAccess, y.targetAccess,
                              y.control, y.participation, y.iteration.kind, y.iteration.distance, y.iteration.carrier,
                              y.precision, y.generation, y.channel, y.detail) &&
                      x.atoms == y.atoms;
           });
}
} // namespace

FailureOr<std::optional<SyncMixedProtocolPlan>> mlir::pto::protocol_sync::buildMixedStructuredFrontierPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels)
{
    auto repair = buildStructuredFrontierPlan(schedule);
    if (failed(repair)) {
        return failure();
    }
    if (!*repair) {
        return std::optional<SyncMixedProtocolPlan>{};
    }
    SyncMixedProtocolPlan plan;
    plan.structuredFrontier = std::move(**repair);
    for (const auto& phase : plan.structuredFrontier->phases) {
        plan.selectedWorld.acknowledgedPhases.push_back(phase.phase);
        plan.selectedWorld.exitCompletedPhases.push_back(phase.phase);
        if (phase.pipe == PIPE::PIPE_V) {
            ++plan.selectedCost.targetedBarriers;
            ++plan.selectedCost.staticActions;
        } else {
            plan.selectedCost.generatedEventPairs += 2;
            plan.selectedCost.staticActions += 4;
            plan.selectedCost.eventPressure = 1;
        }
    }
    ++plan.selectedCost.staticActions;
    plan.selectedCost.fixedExitDrains = 1;
    auto interpreted = interpretSelectedWorld(schedule, stages, timelines, channels, plan.selectedWorld);
    if (failed(interpreted)) {
        return failure();
    }
    if (!interpreted->isComplete()) {
        return std::optional<SyncMixedProtocolPlan>{};
    }
    plan.status = SyncMixedPlanStatus::Ready;
    plan.selectedWorldKind = SyncMixedWorldKind::StructuredFrontier;
    plan.initialResidualCount = plan.structuredFrontier->requirements.size();
    plan.candidateCountBeforeDeletion = 1;
    return std::optional<SyncMixedProtocolPlan>(std::move(plan));
}

LogicalResult mlir::pto::protocol_sync::verifyMixedStructuredFrontierPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncMixedProtocolPlan& plan)
{
    auto expected = buildMixedStructuredFrontierPlan(schedule, stages, timelines, channels);
    const bool valid = succeeded(expected) && *expected && plan.structuredFrontier && !plan.loopFrontier &&
                       !plan.hasProtocol() && plan.directObligations.empty() && plan.directRepair.candidates.empty() &&
                       plan.failures.empty() && !plan.selectedWorld.orderedLoop &&
                       plan.selectedWorld.protocols.empty() && plan.selectedWorld.completions.empty() &&
                       plan.selectedWorld.visibility.empty();
    if (!valid) {
        return failure();
    }
    const auto& reference = **expected;
    const auto& a = *plan.structuredFrontier;
    const auto& b = *reference.structuredFrontier;
    const bool phases =
        a.phases.size() == b.phases.size() && llvm::equal(a.phases, b.phases, [](const auto& x, const auto& y) {
            return std::tie(x.phase, x.operation, x.pipe, x.acquireId, x.releaseId) ==
                   std::tie(y.phase, y.operation, y.pipe, y.acquireId, y.releaseId);
        });
    const auto& x = plan.selectedCost;
    const auto& y = reference.selectedCost;
    return success(
        phases && sameRequirements(a.requirements, b.requirements) && plan.status == reference.status &&
        plan.selectedWorldKind == reference.selectedWorldKind &&
        plan.initialResidualCount == reference.initialResidualCount && plan.candidateCountBeforeDeletion == 1 &&
        plan.reverseDeletionAttempts == 0 && plan.reverseDeletionRemoved == 0 &&
        plan.selectedWorld.acknowledgedPhases == reference.selectedWorld.acknowledgedPhases &&
        plan.selectedWorld.exitCompletedPhases == reference.selectedWorld.exitCompletedPhases &&
        std::tie(x.generatedEventPairs, x.targetedBarriers, x.fixedExitDrains, x.eventPressure, x.staticActions) ==
            std::tie(y.generatedEventPairs, y.targetedBarriers, y.fixedExitDrains, y.eventPressure, y.staticActions));
}
