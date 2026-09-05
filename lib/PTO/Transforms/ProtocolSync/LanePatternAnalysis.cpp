// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LanePatternAnalysis.cpp - Lane pattern experiment ---------------===//

#include "PTO/Transforms/ProtocolSync/LanePatternAnalysis.h"

#include "LanePatternInternal.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;
using namespace mlir::pto::protocol_sync::detail;

namespace {

SyncLaneTargetQuery queryTarget(
    const ProtocolSyncTarget& target, const LaneFrontierAnalysisResult& frontiers,
    const SyncLanePatternCandidate& candidate)
{
    if (!target.isSupported()) {
        return SyncLaneTargetQuery::UnsupportedTarget;
    }
    const SyncExecutionLane* source = frontiers.findLane(candidate.sourceLane);
    const SyncExecutionLane* destination = frontiers.findLane(candidate.targetLane);
    if (!source || !destination) {
        return SyncLaneTargetQuery::UnsupportedMechanism;
    }
    const ProtocolSyncResource sourceResource{source->core, source->pipe};
    const ProtocolSyncResource targetResource{destination->core, destination->pipe};
    bool supported = false;
    switch (candidate.kind) {
        case SyncLanePatternKind::SharedOneShotFrontier:
            supported = target.supportsEvent(sourceResource, targetResource);
            break;
        case SyncLanePatternKind::SameLaneCompletionCut:
            supported = candidate.sourceLane == candidate.targetLane && target.supportsPipeBarrier(sourceResource);
            break;
        case SyncLanePatternKind::ChoiceBalancedRoundTrip:
            supported = target.supportsReadyRelease(source->core, source->pipe, destination->pipe);
            break;
    }
    return supported ? SyncLaneTargetQuery::Supported : SyncLaneTargetQuery::UnsupportedMechanism;
}

void updateStatistics(const SyncLanePatternCandidate& candidate, ProtocolSyncStatistics* statistics)
{
    if (!statistics) {
        return;
    }
    ++statistics->lanePatternCandidates;
    ++statistics->lanePatternKinds[stringifySyncLanePatternKind(candidate.kind).str()];
    switch (candidate.kind) {
        case SyncLanePatternKind::SharedOneShotFrontier:
            ++statistics->sharedOneShotFrontiers;
            break;
        case SyncLanePatternKind::SameLaneCompletionCut:
            ++statistics->sameLaneCompletionCuts;
            break;
        case SyncLanePatternKind::ChoiceBalancedRoundTrip:
            ++statistics->choiceBalancedRoundTrips;
            break;
    }
    if (candidate.targetQuery == SyncLaneTargetQuery::Supported) {
        ++statistics->lanePatternTargetSupported;
    } else {
        ++statistics->lanePatternTargetRejected;
    }
    switch (candidate.checkpointE) {
        case SyncCheckpointEStatus::Admitted:
            ++statistics->lanePatternCheckpointEAdmitted;
            break;
        case SyncCheckpointEStatus::Rejected:
        case SyncCheckpointEStatus::Unavailable:
            ++statistics->lanePatternCheckpointERejected;
            break;
        case SyncCheckpointEStatus::NotApplicable:
            ++statistics->lanePatternCheckpointENotApplicable;
            break;
    }
    statistics->lanePatternLogicalCost += candidate.cost.logicalCandidates;
    statistics->lanePatternSteadyStateActions += candidate.cost.steadyStateActions;
}

void queryRawPairCompletion(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& frontiers, const ProtocolSyncTarget& target,
    SyncLaneRawAccessPair& pair)
{
    if (pair.sourceLane == kInvalidSyncId || pair.sourceLane != pair.targetLane) {
        pair.completion = ProtocolSyncSameLaneCompletion::NotApplicable;
        return;
    }
    const SyncExecutionLane* lane = frontiers.findLane(pair.sourceLane);
    const SyncPhase* source = schedule.findPhase(pair.sourcePhase);
    const SyncPhase* destination = schedule.findPhase(pair.targetPhase);
    if (!lane || !source || !destination) {
        pair.completion = ProtocolSyncSameLaneCompletion::UnsupportedMechanism;
        return;
    }
    pair.completion =
        target.querySameLaneCompletion({lane->core, lane->pipe}, source->operation, destination->operation);
}

void updateRawPairStatistics(const SyncLaneRawAccessPair& pair, ProtocolSyncStatistics* statistics)
{
    if (!statistics) {
        return;
    }
    switch (pair.completion) {
        case ProtocolSyncSameLaneCompletion::Intrinsic:
            ++statistics->rawAccessPairCompletionIntrinsic;
            break;
        case ProtocolSyncSameLaneCompletion::PipeBarrier:
            ++statistics->rawAccessPairCompletionPipeBarrier;
            break;
        case ProtocolSyncSameLaneCompletion::UnsupportedTarget:
            ++statistics->rawAccessPairCompletionUnsupportedTarget;
            break;
        case ProtocolSyncSameLaneCompletion::UnsupportedMechanism:
            ++statistics->rawAccessPairCompletionUnsupportedMechanism;
            break;
        case ProtocolSyncSameLaneCompletion::NotApplicable:
            ++statistics->rawAccessPairCompletionNotApplicable;
            break;
    }
}

} // namespace

LanePatternAnalysisResult mlir::pto::protocol_sync::analyzeLanePatterns(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const LaneFrontierAnalysisResult& frontiers, ProtocolSyncStatistics* statistics)
{
    LanePatternAnalysisResult result;
    result.rawAccessPairs = buildLaneRawAccessPairs(schedule, timelines, frontiers);
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    for (SyncLaneRawAccessPair& pair : result.rawAccessPairs) {
        queryRawPairCompletion(schedule, frontiers, target, pair);
        updateRawPairStatistics(pair, statistics);
    }
    if (statistics) {
        statistics->rawAccessPairs += result.rawAccessPairs.size();
    }
    appendSharedOneShotFrontiers(schedule, timelines, frontiers, result.candidates);
    appendSameLaneCompletionCuts(schedule, stages, result.rawAccessPairs, result.candidates);
    appendChoiceBalancedRoundTrips(schedule, stages, timelines, channels, frontiers, result.candidates);

    for (auto [id, candidate] : llvm::enumerate(result.candidates)) {
        candidate.id = static_cast<SyncLanePatternCandidateId>(id);
        candidate.targetQuery = queryTarget(target, frontiers, candidate);
        updateStatistics(candidate, statistics);
    }
    return result;
}

StringRef mlir::pto::protocol_sync::stringifySyncLanePatternKind(SyncLanePatternKind kind)
{
    switch (kind) {
        case SyncLanePatternKind::SharedOneShotFrontier:
            return "shared-one-shot-frontier";
        case SyncLanePatternKind::SameLaneCompletionCut:
            return "same-lane-completion-cut";
        case SyncLanePatternKind::ChoiceBalancedRoundTrip:
            return "choice-balanced-round-trip";
    }
    return "shared-one-shot-frontier";
}

StringRef mlir::pto::protocol_sync::stringifySyncLaneReferencePattern(SyncLaneReferencePattern pattern)
{
    switch (pattern) {
        case SyncLaneReferencePattern::SharedEventFrontier:
            return "shared-event-frontier";
        case SyncLaneReferencePattern::MultiDemandPipeBarrier:
            return "multi-demand-pipe-barrier";
        case SyncLaneReferencePattern::LiftedChoiceReadyRelease:
            return "lifted-choice-ready-release";
    }
    return "shared-event-frontier";
}

StringRef mlir::pto::protocol_sync::stringifySyncLaneReferenceMechanism(SyncLanePatternKind kind)
{
    switch (kind) {
        case SyncLanePatternKind::SharedOneShotFrontier:
            return "event";
        case SyncLanePatternKind::SameLaneCompletionCut:
            return "barrier";
        case SyncLanePatternKind::ChoiceBalancedRoundTrip:
            return "recurring-event";
    }
    return "event";
}

StringRef mlir::pto::protocol_sync::stringifySyncLaneTargetQuery(SyncLaneTargetQuery query)
{
    switch (query) {
        case SyncLaneTargetQuery::Supported:
            return "supported";
        case SyncLaneTargetQuery::UnsupportedTarget:
            return "unsupported-target";
        case SyncLaneTargetQuery::UnsupportedMechanism:
            return "unsupported-mechanism";
    }
    return "unsupported-mechanism";
}

StringRef mlir::pto::protocol_sync::stringifySyncCheckpointEStatus(SyncCheckpointEStatus status)
{
    switch (status) {
        case SyncCheckpointEStatus::Admitted:
            return "admitted";
        case SyncCheckpointEStatus::Rejected:
            return "rejected";
        case SyncCheckpointEStatus::NotApplicable:
            return "not-applicable";
        case SyncCheckpointEStatus::Unavailable:
            return "unavailable";
    }
    return "unavailable";
}

StringRef mlir::pto::protocol_sync::stringifySyncLaneRawHazardKind(SyncLaneRawHazardKind kind)
{
    switch (kind) {
        case SyncLaneRawHazardKind::ReadAfterWrite:
            return "raw";
        case SyncLaneRawHazardKind::WriteAfterRead:
            return "war";
        case SyncLaneRawHazardKind::WriteAfterWrite:
            return "waw";
    }
    return "waw";
}
