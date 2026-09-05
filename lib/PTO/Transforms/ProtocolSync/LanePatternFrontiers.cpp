// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LanePatternFrontiers.cpp - Shared and choice lane patterns ------===//

#include "LanePatternInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

struct ReadyWindow {
    const SyncLaneFrontierExperiment* experiment = nullptr;
    SyncProgramPointId source = kInvalidSyncId;
    SyncProgramPointId target = kInvalidSyncId;
};

struct ReadyWindowGroup {
    SyncExecutionLaneId sourceLane = kInvalidSyncId;
    SyncExecutionLaneId targetLane = kInvalidSyncId;
    SyncRegionId region = kInvalidSyncId;
    SmallVector<SyncControlAtom, 2> guard;
    SmallVector<ReadyWindow, 8> windows;
};

bool sameControlAtom(const SyncControlAtom& first, const SyncControlAtom& second)
{
    return first.choice == second.choice && first.arm == second.arm;
}

bool sameGuard(ArrayRef<SyncControlAtom> first, ArrayRef<SyncControlAtom> second)
{
    return first.size() == second.size() && llvm::equal(first, second, sameControlAtom);
}

const SyncGuardedProgramPoint* getOnlyPoint(const SyncProgramFrontier& frontier)
{
    return frontier.points.size() == 1 ? &frontier.points.front() : nullptr;
}

const SyncGenerationTimeline* findTimeline(const StorageTimelineAnalysisResult& timelines, SyncGenerationId generation)
{
    const ArrayRef<SyncGenerationTimeline> all = timelines.getTimelines();
    return generation < all.size() ? &all[generation] : nullptr;
}

const SyncChannel* findChannel(const ChannelAnalysisResult& channels, SyncGenerationId generation)
{
    auto match = llvm::find_if(
        channels.getChannels(), [&](const SyncChannel& channel) { return channel.generation == generation; });
    return match == channels.getChannels().end() ? nullptr : &*match;
}

void appendUniqueGeneration(SmallVectorImpl<SyncGenerationId>& generations, SyncGenerationId generation)
{
    if (!llvm::is_contained(generations, generation)) {
        generations.push_back(generation);
    }
}

SmallVector<ReadyWindowGroup, 8> collectReadyWindowGroups(
    const StructuredSyncIR& schedule, const StorageTimelineAnalysisResult& timelines,
    const LaneFrontierAnalysisResult& frontiers)
{
    SmallVector<ReadyWindowGroup, 8> groups;
    for (const SyncLaneFrontierExperiment& experiment : frontiers.getExperiments()) {
        const SyncGenerationTimeline* timeline = findTimeline(timelines, experiment.generation);
        const SyncGuardedProgramPoint* source = getOnlyPoint(experiment.sourceFrontier);
        const SyncGuardedProgramPoint* target = getOnlyPoint(experiment.targetFrontier);
        const bool ineligible = experiment.demand != SyncLaneDemandKind::Ready || !experiment.isFound() ||
                                experiment.topology != SyncLaneTopology::CrossLane || !timeline ||
                                timeline->generationKind != SyncGenerationKind::OneShot || !source || !target ||
                                source->point >= schedule.getProgramPoints().size() ||
                                target->point >= schedule.getProgramPoints().size();
        if (ineligible) {
            continue;
        }
        const SyncRegionId sourceRegion = schedule.getProgramPoints()[source->point].region;
        const SyncRegionId targetRegion = schedule.getProgramPoints()[target->point].region;
        if (sourceRegion != targetRegion || !sameGuard(source->guard, target->guard)) {
            continue;
        }
        auto group = llvm::find_if(groups, [&](const ReadyWindowGroup& current) {
            return current.sourceLane == experiment.sourceLane && current.targetLane == experiment.targetLane &&
                   current.region == sourceRegion && sameGuard(current.guard, source->guard);
        });
        if (group == groups.end()) {
            ReadyWindowGroup next;
            next.sourceLane = experiment.sourceLane;
            next.targetLane = experiment.targetLane;
            next.region = sourceRegion;
            next.guard.assign(source->guard.begin(), source->guard.end());
            groups.push_back(std::move(next));
            group = std::prev(groups.end());
        }
        group->windows.push_back({&experiment, source->point, target->point});
    }
    return groups;
}

void appendSharedCandidate(
    const ReadyWindowGroup& group, ArrayRef<ReadyWindow> windows, SmallVectorImpl<SyncLanePatternCandidate>& candidates)
{
    const bool hasMultipleWindows = windows.size() >= 2;
    if (!hasMultipleWindows) {
        return;
    }
    const ReadyWindow& latestSource = *llvm::max_element(
        windows, [](const ReadyWindow& left, const ReadyWindow& right) { return left.source < right.source; });
    const ReadyWindow& earliestTarget = *llvm::min_element(
        windows, [](const ReadyWindow& left, const ReadyWindow& right) { return left.target < right.target; });
    if (latestSource.source >= earliestTarget.target) {
        return;
    }

    SyncLanePatternCandidate candidate;
    candidate.kind = SyncLanePatternKind::SharedOneShotFrontier;
    candidate.referencePattern = SyncLaneReferencePattern::SharedEventFrontier;
    candidate.sourceLane = group.sourceLane;
    candidate.targetLane = group.targetLane;
    candidate.firstSource.points.push_back({latestSource.source, group.guard});
    candidate.firstTarget.points.push_back({earliestTarget.target, group.guard});
    candidate.cost.steadyStateActions = 2;
    for (const ReadyWindow& window : windows) {
        candidate.frontierMembers.push_back(window.experiment->id);
        appendUniqueGeneration(candidate.generations, window.experiment->generation);
    }
    candidates.push_back(std::move(candidate));
}

bool stageCoversChoiceArm(const SyncStage& stage, SyncRegionId choice, unsigned arm)
{
    return llvm::any_of(
        stage.guard, [&](const SyncControlAtom& atom) { return atom.choice == choice && atom.arm == arm; });
}

bool consumersCoverEveryArm(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncGenerationTimeline& timeline,
    const SyncRegion& choice)
{
    SmallVector<unsigned, 2> arms;
    for (const SyncRegionElement& element : choice.elements) {
        const SyncRegion* child =
            element.kind == SyncRegionElement::Kind::ChildRegion ? schedule.findRegion(element.child) : nullptr;
        if (child && child->kind == SyncRegionKind::Alternative) {
            arms.push_back(child->arm);
        }
    }
    const bool hasCompleteArmSet = arms.size() >= 2 && timeline.consumers.size() >= arms.size();
    if (!hasCompleteArmSet) {
        return false;
    }
    for (SyncStageId consumerId : timeline.consumers) {
        const SyncStage* consumer = stages.findStage(consumerId);
        if (!consumer ||
            !llvm::any_of(arms, [&](unsigned arm) { return stageCoversChoiceArm(*consumer, choice.id, arm); })) {
            return false;
        }
    }
    return llvm::all_of(arms, [&](unsigned arm) {
        return llvm::any_of(timeline.consumers, [&](SyncStageId consumerId) {
            const SyncStage* consumer = stages.findStage(consumerId);
            return consumer && stageCoversChoiceArm(*consumer, choice.id, arm);
        });
    });
}

} // namespace

void mlir::pto::protocol_sync::detail::appendSharedOneShotFrontiers(
    const StructuredSyncIR& schedule, const StorageTimelineAnalysisResult& timelines,
    const LaneFrontierAnalysisResult& frontiers, SmallVectorImpl<SyncLanePatternCandidate>& candidates)
{
    for (ReadyWindowGroup& group : collectReadyWindowGroups(schedule, timelines, frontiers)) {
        llvm::sort(group.windows, [](const ReadyWindow& left, const ReadyWindow& right) {
            return left.source < right.source;
        });
        SmallVector<ReadyWindow, 8> compatible;
        SyncProgramPointId latestSource = 0;
        SyncProgramPointId earliestTarget = kInvalidSyncId;
        for (const ReadyWindow& window : group.windows) {
            const SyncProgramPointId nextSource =
                compatible.empty() ? window.source : std::max(latestSource, window.source);
            const SyncProgramPointId nextTarget =
                compatible.empty() ? window.target : std::min(earliestTarget, window.target);
            const bool startsNewGroup = !compatible.empty() && nextSource >= nextTarget;
            if (startsNewGroup) {
                appendSharedCandidate(group, compatible, candidates);
                compatible.clear();
                latestSource = window.source;
                earliestTarget = window.target;
            } else {
                latestSource = nextSource;
                earliestTarget = nextTarget;
            }
            compatible.push_back(window);
        }
        appendSharedCandidate(group, compatible, candidates);
    }
}

void mlir::pto::protocol_sync::detail::appendChoiceBalancedRoundTrips(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const LaneFrontierAnalysisResult& frontiers, SmallVectorImpl<SyncLanePatternCandidate>& candidates)
{
    for (const SyncLaneFrontierExperiment& ready : frontiers.getExperiments()) {
        const SyncGenerationTimeline* timeline = findTimeline(timelines, ready.generation);
        const SyncGuardedProgramPoint* choiceEntry = getOnlyPoint(ready.targetFrontier);
        const bool ineligible = ready.demand != SyncLaneDemandKind::Ready || !ready.isFound() ||
                                ready.targetPlacement != SyncFrontierPlacement::ChoiceEntry || !timeline ||
                                timeline->generationKind != SyncGenerationKind::LoopIteration ||
                                !timeline->nextOverwrite || timeline->nextOverwrite->iterationDistance == 0 ||
                                ready.targetStages.size() != timeline->consumers.size() || !choiceEntry ||
                                choiceEntry->point >= schedule.getProgramPoints().size();
        if (ineligible) {
            continue;
        }
        const SyncProgramPoint& entryPoint = schedule.getProgramPoints()[choiceEntry->point];
        const SyncRegion* choice = schedule.findRegion(entryPoint.region);
        if (!choice || choice->kind != SyncRegionKind::Choice || entryPoint.kind != SyncProgramPointKind::RegionEntry ||
            !consumersCoverEveryArm(schedule, stages, *timeline, *choice)) {
            continue;
        }
        for (const SyncLaneFrontierExperiment& release : frontiers.getExperiments()) {
            const SyncGuardedProgramPoint* choiceExit = getOnlyPoint(release.sourceFrontier);
            const bool doesNotPair = release.generation != ready.generation ||
                                     release.demand != SyncLaneDemandKind::Release || !release.isFound() ||
                                     release.sourcePlacement != SyncFrontierPlacement::ChoiceExit ||
                                     release.sourceLane != ready.targetLane || release.targetLane != ready.sourceLane ||
                                     release.sourceStages.size() != timeline->consumers.size() || !choiceExit ||
                                     choiceExit->point >= schedule.getProgramPoints().size();
            if (doesNotPair) {
                continue;
            }
            const SyncProgramPoint& exitPoint = schedule.getProgramPoints()[choiceExit->point];
            if (exitPoint.kind != SyncProgramPointKind::RegionExit || exitPoint.region != choice->id) {
                continue;
            }
            SyncLanePatternCandidate candidate;
            candidate.kind = SyncLanePatternKind::ChoiceBalancedRoundTrip;
            candidate.referencePattern = SyncLaneReferencePattern::LiftedChoiceReadyRelease;
            candidate.sourceLane = ready.sourceLane;
            candidate.targetLane = ready.targetLane;
            candidate.firstSource = ready.sourceFrontier;
            candidate.firstTarget = ready.targetFrontier;
            candidate.secondSource = release.sourceFrontier;
            candidate.secondTarget = release.targetFrontier;
            candidate.generations.push_back(ready.generation);
            candidate.frontierMembers = {ready.id, release.id};
            candidate.cost.steadyStateActions = 4;
            const SyncChannel* channel = findChannel(channels, ready.generation);
            candidate.checkpointE =
                channel ? (channel->isAdmitted() ? SyncCheckpointEStatus::Admitted : SyncCheckpointEStatus::Rejected) :
                          SyncCheckpointEStatus::Unavailable;
            candidate.checkpointERejection = channel ? channel->rejection : SyncChannelRejection::TimelineRejected;
            candidates.push_back(std::move(candidate));
            break;
        }
    }
}
