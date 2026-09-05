// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTransitionFrontiers.cpp - Track transition cuts ---------===//

#include "StorageTrackInternal.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

void setCheckpointEStatus(
    const SyncGenerationTimeline& timeline, const SyncReadyReleasePlan* plan, SyncStorageTransitionFrontier& transition)
{
    if (timeline.generationKind == SyncGenerationKind::OneShot) {
        transition.checkpointE = SyncCheckpointEStatus::NotApplicable;
        return;
    }
    if (!plan) {
        transition.checkpointE = SyncCheckpointEStatus::Unavailable;
        transition.checkpointERejection = SyncReadyReleaseRejection::InternalInvariant;
        return;
    }
    if (plan->status == SyncReadyReleasePlanStatus::Ready && plan->generation == timeline.id) {
        transition.checkpointE = SyncCheckpointEStatus::Admitted;
        return;
    }
    if (plan->status == SyncReadyReleasePlanStatus::Unsupported) {
        transition.checkpointE = SyncCheckpointEStatus::Rejected;
        if (!plan->rejections.empty()) {
            transition.checkpointERejection = plan->rejections.front().reason;
        }
        return;
    }
    transition.checkpointE = SyncCheckpointEStatus::NotApplicable;
}

void appendUniqueTrack(SmallVectorImpl<SyncStorageTrackId>& tracks, SyncStorageTrackId track)
{
    if (!llvm::is_contained(tracks, track)) {
        tracks.push_back(track);
    }
}

void appendUniqueRawPair(SmallVectorImpl<SyncLaneRawAccessPairId>& pairs, SyncLaneRawAccessPairId pair)
{
    if (!llvm::is_contained(pairs, pair)) {
        pairs.push_back(pair);
    }
}

SmallVector<SyncStorageTrackId, 4> intersectTracks(
    ArrayRef<SyncStorageTrackId> first, ArrayRef<SyncStorageTrackId> second)
{
    SmallVector<SyncStorageTrackId, 4> common;
    for (SyncStorageTrackId track : first) {
        if (llvm::is_contained(second, track)) {
            common.push_back(track);
        }
    }
    return common;
}

SmallVector<SyncStorageTrackId, 4> collectStageTracks(
    const SyncGenerationTimeline& timeline, ArrayRef<SyncStageId> stages,
    ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks)
{
    SmallVector<SyncStorageTrackId, 4> result;
    for (const SyncRawAccessEndpoint& endpoint : timeline.rawAccesses) {
        const bool usableEndpoint =
            llvm::is_contained(stages, endpoint.stage) && endpoint.access < accessToTracks.size();
        if (!usableEndpoint) {
            continue;
        }
        for (SyncStorageTrackId track : accessToTracks[endpoint.access]) {
            appendUniqueTrack(result, track);
        }
    }
    llvm::sort(result);
    return result;
}

void recordTransition(
    SyncStorageTransitionFrontier transition, SmallVectorImpl<SyncStorageTransitionFrontier>& transitions,
    ProtocolSyncStatistics* statistics)
{
    transition.id = static_cast<SyncStorageTransitionId>(transitions.size());
    llvm::sort(transition.tracks);
    llvm::sort(transition.rawPairMembers);
    transitions.push_back(std::move(transition));
    if (!statistics) {
        return;
    }
    const SyncStorageTransitionFrontier& stored = transitions.back();
    ++statistics->storageTransitionFrontiers;
    ++statistics->storageTransitionKinds[stringifySyncStorageTransitionKind(stored.kind).str()];
    switch (stored.kind) {
        case SyncStorageTransitionKind::Ready:
        case SyncStorageTransitionKind::Release:
            ++statistics->storageLifecycleTransitions;
            break;
        case SyncStorageTransitionKind::Completion:
            ++statistics->storageCompletionTransitions;
            break;
        case SyncStorageTransitionKind::Residual:
            ++statistics->storageResidualTransitions;
            break;
    }
    const bool supported = stored.targetResult == SyncStorageTargetResult::Intrinsic ||
                           stored.targetResult == SyncStorageTargetResult::PipeBarrier ||
                           stored.targetResult == SyncStorageTargetResult::Event;
    const bool rejected = stored.targetResult == SyncStorageTargetResult::UnsupportedTarget ||
                          stored.targetResult == SyncStorageTargetResult::UnsupportedMechanism;
    statistics->storageTransitionTargetSupported += supported ? 1 : 0;
    statistics->storageTransitionTargetRejected += rejected ? 1 : 0;
}

bool transitionCoversPair(
    const SyncLaneFrontierExperiment& experiment, const SyncLaneRawAccessPair& pair,
    ArrayRef<SyncStorageTrackId> tracks, ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks)
{
    const bool matchingEndpoints = experiment.sourceLane == pair.sourceLane &&
                                   experiment.targetLane == pair.targetLane &&
                                   llvm::is_contained(experiment.sourceStages, pair.sourceStage) &&
                                   llvm::is_contained(experiment.targetStages, pair.targetStage);
    const bool projectedEndpoints =
        pair.sourceAccess < accessToTracks.size() && pair.targetAccess < accessToTracks.size();
    if (!matchingEndpoints || !projectedEndpoints) {
        return false;
    }
    SmallVector<SyncStorageTrackId, 4> pairTracks =
        intersectTracks(accessToTracks[pair.sourceAccess], accessToTracks[pair.targetAccess]);
    return llvm::any_of(pairTracks, [&](SyncStorageTrackId track) { return llvm::is_contained(tracks, track); });
}

void appendLifecycleTransitions(
    const StorageTimelineAnalysisResult& timelines, const SyncReadyReleasePlan* checkpointEPlan,
    const LaneFrontierAnalysisResult& laneFrontiers, const LanePatternAnalysisResult& lanePatterns,
    ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks, const ProtocolSyncTarget& target,
    llvm::SmallBitVector& covered, SmallVectorImpl<SyncStorageTransitionFrontier>& transitions,
    ProtocolSyncStatistics* statistics)
{
    for (const SyncLaneFrontierExperiment& experiment : laneFrontiers.getExperiments()) {
        const bool eligible = experiment.isFound() && experiment.demand != SyncLaneDemandKind::Residual &&
                              experiment.generation < timelines.getTimelines().size();
        if (!eligible) {
            continue;
        }
        const SyncGenerationTimeline& timeline = timelines.getTimelines()[experiment.generation];
        SmallVector<SyncStorageTrackId, 4> sourceTracks =
            collectStageTracks(timeline, experiment.sourceStages, accessToTracks);
        SmallVector<SyncStorageTrackId, 4> targetTracks =
            collectStageTracks(timeline, experiment.targetStages, accessToTracks);
        SmallVector<SyncStorageTrackId, 4> tracks = intersectTracks(sourceTracks, targetTracks);
        if (tracks.empty()) {
            continue;
        }
        SyncStorageTransitionFrontier transition;
        transition.kind = experiment.demand == SyncLaneDemandKind::Ready ? SyncStorageTransitionKind::Ready :
                                                                           SyncStorageTransitionKind::Release;
        transition.origin = SyncStorageTransitionOrigin::LaneFrontier;
        transition.generation = experiment.generation;
        transition.sourceLane = experiment.sourceLane;
        transition.targetLane = experiment.targetLane;
        transition.sourceFrontier = experiment.sourceFrontier;
        transition.targetFrontier = experiment.targetFrontier;
        transition.iterationDistance = experiment.iterationDistance;
        transition.recurring = timeline.generationKind == SyncGenerationKind::LoopIteration;
        transition.tracks = std::move(tracks);
        transition.cost.steadyStateActions = 2;
        transition.targetResult = mlir::pto::protocol_sync::detail::queryStorageLaneTransfer(
            target, laneFrontiers, experiment.sourceLane, experiment.targetLane);
        setCheckpointEStatus(timeline, checkpointEPlan, transition);
        for (const SyncLaneRawAccessPair& pair : lanePatterns.getRawAccessPairs()) {
            if (transitionCoversPair(experiment, pair, transition.tracks, accessToTracks)) {
                appendUniqueRawPair(transition.rawPairMembers, pair.id);
                if (pair.id < covered.size()) {
                    covered.set(pair.id);
                }
            }
        }
        mlir::pto::protocol_sync::detail::refineSameLaneStorageTarget(lanePatterns, transition);
        recordTransition(std::move(transition), transitions, statistics);
    }
}

void appendCompletionTransitions(
    const LanePatternAnalysisResult& lanePatterns, ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks,
    llvm::SmallBitVector& covered, SmallVectorImpl<SyncStorageTransitionFrontier>& transitions,
    ProtocolSyncStatistics* statistics)
{
    for (const SyncLanePatternCandidate& candidate : lanePatterns.getCandidates()) {
        if (candidate.kind != SyncLanePatternKind::SameLaneCompletionCut) {
            continue;
        }
        SyncStorageTransitionFrontier transition;
        transition.kind = SyncStorageTransitionKind::Completion;
        transition.origin = SyncStorageTransitionOrigin::CompletionCut;
        transition.sourceLane = candidate.sourceLane;
        transition.targetLane = candidate.targetLane;
        transition.sourceFrontier = candidate.firstSource;
        transition.targetFrontier = candidate.firstSource;
        transition.cost = candidate.cost;
        transition.targetResult = candidate.targetQuery == SyncLaneTargetQuery::Supported ?
                                      SyncStorageTargetResult::PipeBarrier :
                                      SyncStorageTargetResult::UnsupportedMechanism;
        for (SyncLaneRawAccessPairId pairId : candidate.rawPairMembers) {
            const bool usablePair = pairId < lanePatterns.getRawAccessPairs().size() && !covered.test(pairId);
            if (!usablePair) {
                continue;
            }
            const SyncLaneRawAccessPair& pair = lanePatterns.getRawAccessPairs()[pairId];
            const bool projectedEndpoints =
                pair.sourceAccess < accessToTracks.size() && pair.targetAccess < accessToTracks.size();
            if (!projectedEndpoints) {
                continue;
            }
            SmallVector<SyncStorageTrackId, 4> pairTracks =
                intersectTracks(accessToTracks[pair.sourceAccess], accessToTracks[pair.targetAccess]);
            if (pairTracks.empty()) {
                continue;
            }
            for (SyncStorageTrackId track : pairTracks) {
                appendUniqueTrack(transition.tracks, track);
            }
            appendUniqueRawPair(transition.rawPairMembers, pair.id);
            covered.set(pair.id);
        }
        if (!transition.rawPairMembers.empty()) {
            recordTransition(std::move(transition), transitions, statistics);
        }
    }
}

void appendResidualTransitions(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& laneFrontiers,
    const LanePatternAnalysisResult& lanePatterns, ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks,
    const ProtocolSyncTarget& target, llvm::SmallBitVector& covered,
    SmallVectorImpl<SyncStorageTransitionFrontier>& transitions, ProtocolSyncStatistics* statistics)
{
    for (const SyncLaneRawAccessPair& pair : lanePatterns.getRawAccessPairs()) {
        const bool usablePair = pair.id < covered.size() && !covered.test(pair.id) &&
                                pair.sourceAccess < accessToTracks.size() && pair.targetAccess < accessToTracks.size();
        if (!usablePair) {
            continue;
        }
        SmallVector<SyncStorageTrackId, 4> tracks =
            intersectTracks(accessToTracks[pair.sourceAccess], accessToTracks[pair.targetAccess]);
        if (tracks.empty()) {
            continue;
        }
        SyncStorageTransitionFrontier transition;
        transition.kind = SyncStorageTransitionKind::Residual;
        transition.origin = SyncStorageTransitionOrigin::RawAccessPair;
        transition.generation = pair.sourceGeneration == pair.targetGeneration ? pair.sourceGeneration : kInvalidSyncId;
        transition.sourceLane = pair.sourceLane;
        transition.targetLane = pair.targetLane;
        const SyncPhase* source = schedule.findPhase(pair.sourcePhase);
        const SyncPhase* destination = schedule.findPhase(pair.targetPhase);
        transition.sourceFrontier.points.push_back(
            {pair.sourceAfter, source ? source->guard : SmallVector<SyncControlAtom, 2>()});
        transition.targetFrontier.points.push_back(
            {pair.targetBefore, destination ? destination->guard : SmallVector<SyncControlAtom, 2>()});
        transition.tracks = std::move(tracks);
        transition.rawPairMembers.push_back(pair.id);
        transition.targetResult =
            pair.sourceLane == pair.targetLane ?
                mlir::pto::protocol_sync::detail::mapSameLaneCompletionToStorageTarget(pair.completion) :
                mlir::pto::protocol_sync::detail::queryStorageLaneTransfer(
                    target, laneFrontiers, pair.sourceLane, pair.targetLane);
        switch (transition.targetResult) {
            case SyncStorageTargetResult::Intrinsic:
                transition.cost.steadyStateActions = 0;
                break;
            case SyncStorageTargetResult::PipeBarrier:
                transition.cost.steadyStateActions = 1;
                break;
            case SyncStorageTargetResult::Event:
                transition.cost.steadyStateActions = 2;
                break;
            case SyncStorageTargetResult::UnsupportedTarget:
            case SyncStorageTargetResult::UnsupportedMechanism:
            case SyncStorageTargetResult::NotApplicable:
                break;
        }
        covered.set(pair.id);
        recordTransition(std::move(transition), transitions, statistics);
    }
}

} // namespace

void mlir::pto::protocol_sync::detail::appendStorageTransitionFrontiers(
    const StructuredSyncIR& schedule, const StorageTimelineAnalysisResult& timelines,
    const SyncReadyReleasePlan* checkpointEPlan, const LaneFrontierAnalysisResult& laneFrontiers,
    const LanePatternAnalysisResult& lanePatterns, ArrayRef<SyncStorageTrack> tracks,
    ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks,
    SmallVectorImpl<SyncStorageTransitionFrontier>& transitions, ProtocolSyncStatistics* statistics)
{
    (void)tracks;
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    llvm::SmallBitVector covered(lanePatterns.getRawAccessPairs().size());
    appendLifecycleTransitions(
        timelines, checkpointEPlan, laneFrontiers, lanePatterns, accessToTracks, target, covered, transitions,
        statistics);
    appendCompletionTransitions(lanePatterns, accessToTracks, covered, transitions, statistics);
    appendResidualTransitions(
        schedule, laneFrontiers, lanePatterns, accessToTracks, target, covered, transitions, statistics);
    if (statistics) {
        statistics->storageRawPairsCovered += covered.count();
        statistics->storageRawPairsUncovered += covered.size() - covered.count();
    }
}
