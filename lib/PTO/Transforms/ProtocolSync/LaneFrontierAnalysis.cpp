// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LaneFrontierAnalysis.cpp - Lane/frontier experiment -------------===//

#include "PTO/Transforms/ProtocolSync/LaneFrontierAnalysis.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <iterator>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

enum class FrontierSide : std::uint8_t { Before, After };

struct UniqueLane {
    SyncExecutionLaneId id = kInvalidSyncId;
    bool multiple = false;
};

struct StageLaneGroup {
    SyncExecutionLaneId lane = kInvalidSyncId;
    SmallVector<SyncStageId, 4> stages;
    SmallVector<unsigned, 4> indices;
};

bool sameControlAtom(const SyncControlAtom& first, const SyncControlAtom& second)
{
    return first.choice == second.choice && first.arm == second.arm;
}

bool sameGuard(ArrayRef<SyncControlAtom> first, ArrayRef<SyncControlAtom> second)
{
    return first.size() == second.size() && llvm::equal(first, second, sameControlAtom);
}

unsigned commonGuardPrefix(ArrayRef<SyncGuardedProgramPoint> first, ArrayRef<SyncGuardedProgramPoint> second)
{
    const bool missingPoints = first.empty() || second.empty();
    if (missingPoints) {
        return 0;
    }
    const ArrayRef<SyncControlAtom> reference = first.front().guard;
    unsigned length = reference.size();
    auto shorten = [&](ArrayRef<SyncControlAtom> guard) {
        length = std::min<unsigned>(length, guard.size());
        unsigned index = 0;
        while (index < length && sameControlAtom(reference[index], guard[index])) {
            ++index;
        }
        length = index;
    };
    for (const SyncGuardedProgramPoint& point : first) {
        shorten(point.guard);
    }
    for (const SyncGuardedProgramPoint& point : second) {
        shorten(point.guard);
    }
    return length;
}

bool deriveFrontier(
    const StructuredSyncIR& schedule, const SyncProgramFrontier& endpoint, const SyncProgramFrontier& peer,
    FrontierSide side, SyncProgramFrontier& result, SyncFrontierPlacement& placement)
{
    const bool missingEndpoint = endpoint.points.empty() || peer.points.empty();
    if (missingEndpoint) {
        return false;
    }

    const unsigned prefix = commonGuardPrefix(endpoint.points, peer.points);
    const bool endpointEntersChoice = llvm::all_of(
        endpoint.points, [&](const SyncGuardedProgramPoint& point) { return point.guard.size() > prefix; });
    const bool peerStopsBeforeChoice =
        llvm::all_of(peer.points, [&](const SyncGuardedProgramPoint& point) { return point.guard.size() == prefix; });
    if (endpointEntersChoice && peerStopsBeforeChoice) {
        const SyncRegionId choice = endpoint.points.front().guard[prefix].choice;
        const bool sameChoice = llvm::all_of(endpoint.points, [&](const SyncGuardedProgramPoint& point) {
            return point.guard[prefix].choice == choice;
        });
        const SyncRegion* region = sameChoice ? schedule.findRegion(choice) : nullptr;
        const ArrayRef<SyncControlAtom> expectedGuard =
            ArrayRef<SyncControlAtom>(endpoint.points.front().guard).take_front(prefix);
        if (!region || region->kind != SyncRegionKind::Choice || !sameGuard(region->guard, expectedGuard)) {
            return false;
        }
        result.points.push_back({side == FrontierSide::Before ? region->entry : region->exit, region->guard});
        placement =
            side == FrontierSide::Before ? SyncFrontierPlacement::ChoiceEntry : SyncFrontierPlacement::ChoiceExit;
        return true;
    }

    const ArrayRef<SyncControlAtom> guard = endpoint.points.front().guard;
    if (!llvm::all_of(
            endpoint.points, [&](const SyncGuardedProgramPoint& point) { return sameGuard(point.guard, guard); })) {
        return false;
    }
    const ArrayRef<SyncProgramPoint> programPoints = schedule.getProgramPoints();
    if (llvm::any_of(endpoint.points, [&](const SyncGuardedProgramPoint& point) {
            return point.point >= programPoints.size();
        })) {
        return false;
    }
    const SyncRegionId region = programPoints[endpoint.points.front().point].region;
    if (!llvm::all_of(endpoint.points, [&](const SyncGuardedProgramPoint& point) {
            return programPoints[point.point].region == region;
        })) {
        return false;
    }

    auto compare = [](const SyncGuardedProgramPoint& left, const SyncGuardedProgramPoint& right) {
        return left.point < right.point;
    };
    const auto selected = side == FrontierSide::Before ? llvm::min_element(endpoint.points, compare) :
                                                         llvm::max_element(endpoint.points, compare);
    result.points.push_back(*selected);
    placement = endpoint.points.size() == 1 ? SyncFrontierPlacement::Exact : SyncFrontierPlacement::LinearCoalesced;
    return true;
}

SyncExecutionLaneId findExistingLane(const LaneFrontierAnalysisResult& result, SyncPhysicalCore core, PIPE pipe)
{
    for (const SyncExecutionLane& lane : result.getLanes()) {
        if (lane.core == core && lane.pipe == pipe) {
            return lane.id;
        }
    }
    return kInvalidSyncId;
}

SyncExecutionLaneId getStageLane(
    const PipelineStageAnalysisResult& stages, const LaneFrontierAnalysisResult& result, SyncStageId stageId)
{
    const SyncStage* stage = stages.findStage(stageId);
    if (!stage || stage->phases.empty()) {
        return kInvalidSyncId;
    }
    const SyncExecutionLane* lane = result.findLaneForPhase(stage->phases.front());
    if (!lane) {
        return kInvalidSyncId;
    }
    const bool spansLanes = llvm::any_of(stage->phases, [&](SyncPhaseId phase) {
        const SyncExecutionLane* current = result.findLaneForPhase(phase);
        return !current || current->id != lane->id;
    });
    return spansLanes ? kInvalidSyncId : lane->id;
}

UniqueLane getUniqueLane(
    ArrayRef<SyncStageId> stageIds, const PipelineStageAnalysisResult& stages, const LaneFrontierAnalysisResult& result)
{
    UniqueLane lane;
    for (SyncStageId stageId : stageIds) {
        const SyncExecutionLaneId current = getStageLane(stages, result, stageId);
        if (current == kInvalidSyncId) {
            lane.id = kInvalidSyncId;
            return lane;
        }
        if (lane.id == kInvalidSyncId) {
            lane.id = current;
        } else if (lane.id != current) {
            lane.id = kInvalidSyncId;
            lane.multiple = true;
            return lane;
        }
    }
    return lane;
}

SmallVector<StageLaneGroup, 4> groupStagesByLane(
    ArrayRef<SyncStageId> stageIds, const PipelineStageAnalysisResult& stages, const LaneFrontierAnalysisResult& result)
{
    SmallVector<StageLaneGroup, 4> groups;
    for (auto [index, stageId] : llvm::enumerate(stageIds)) {
        const SyncExecutionLaneId lane = getStageLane(stages, result, stageId);
        auto group = llvm::find_if(groups, [&](const StageLaneGroup& current) { return current.lane == lane; });
        if (group == groups.end()) {
            groups.push_back({lane, {}, {}});
            group = std::prev(groups.end());
        }
        group->stages.push_back(stageId);
        group->indices.push_back(index);
    }
    return groups;
}

SyncProgramFrontier selectFrontiers(ArrayRef<SyncProgramFrontier> frontiers, ArrayRef<unsigned> indices)
{
    SyncProgramFrontier result;
    for (unsigned index : indices) {
        if (index < frontiers.size()) {
            result.points.append(frontiers[index].points.begin(), frontiers[index].points.end());
        }
    }
    return result;
}

SyncChannelRejection findChannelRejection(SyncGenerationId generation, const ChannelAnalysisResult& channels)
{
    auto channel = llvm::find_if(
        channels.getChannels(), [&](const SyncChannel& current) { return current.generation == generation; });
    return channel == channels.getChannels().end() ? SyncChannelRejection::TimelineRejected : channel->rejection;
}

SyncLaneTopology classifyTopology(
    SyncExecutionLaneId sourceId, SyncExecutionLaneId targetId, const LaneFrontierAnalysisResult& result)
{
    const SyncExecutionLane* source = result.findLane(sourceId);
    const SyncExecutionLane* target = result.findLane(targetId);
    if (!source || !target || source->core == SyncPhysicalCore::Unknown || target->core == SyncPhysicalCore::Unknown ||
        source->pipe == PIPE::PIPE_UNASSIGNED || target->pipe == PIPE::PIPE_UNASSIGNED) {
        return SyncLaneTopology::Unknown;
    }
    if (sourceId == targetId) {
        return SyncLaneTopology::SameLane;
    }
    return source->core == target->core ? SyncLaneTopology::CrossLane : SyncLaneTopology::CrossCore;
}

void updateStatistics(const SyncLaneFrontierExperiment& experiment, ProtocolSyncStatistics* statistics)
{
    if (!statistics) {
        return;
    }
    ++statistics->laneFrontierExperiments;
    if (experiment.demand == SyncLaneDemandKind::Ready) {
        ++statistics->laneReadyExperiments;
    } else if (experiment.demand == SyncLaneDemandKind::Release) {
        ++statistics->laneReleaseExperiments;
    } else {
        ++statistics->laneResidualExperiments;
    }
    if (!experiment.isFound()) {
        ++statistics->laneFrontierRejections[stringifySyncLaneFrontierStatus(experiment.status).str()];
        return;
    }
    ++statistics->laneFrontiersFound;
    if (experiment.topology == SyncLaneTopology::SameLane) {
        ++statistics->sameLaneFrontiersFound;
    } else if (experiment.topology == SyncLaneTopology::CrossLane) {
        ++statistics->crossLaneFrontiersFound;
    }
    if (experiment.sourcePlacement == SyncFrontierPlacement::LinearCoalesced ||
        experiment.targetPlacement == SyncFrontierPlacement::LinearCoalesced) {
        ++statistics->linearFrontiersCoalesced;
    }
    if (experiment.sourcePlacement == SyncFrontierPlacement::ChoiceExit ||
        experiment.targetPlacement == SyncFrontierPlacement::ChoiceEntry) {
        ++statistics->choiceBoundaryFrontiersFound;
    }
    if (experiment.channelRejection != SyncChannelRejection::None) {
        ++statistics->frontiersFoundForRejectedChannels;
    }
}

SyncLaneFrontierExperiment finishExperiment(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& result, SyncLaneFrontierExperiment experiment,
    SyncLaneFrontierExperimentId id, const SyncProgramFrontier& source, const SyncProgramFrontier& target,
    bool multipleSourceLanes, bool multipleTargetLanes, ProtocolSyncStatistics* statistics)
{
    experiment.id = id;
    experiment.topology = classifyTopology(experiment.sourceLane, experiment.targetLane, result);
    if (!schedule.getFailures().empty()) {
        experiment.status = SyncLaneFrontierStatus::UnresolvedSchedule;
    } else if (source.points.empty() || target.points.empty()) {
        experiment.status = SyncLaneFrontierStatus::MissingEndpoint;
    } else if (multipleSourceLanes) {
        experiment.status = SyncLaneFrontierStatus::MultipleSourceLanes;
    } else if (multipleTargetLanes) {
        experiment.status = SyncLaneFrontierStatus::MultipleTargetLanes;
    } else if (experiment.topology == SyncLaneTopology::Unknown) {
        experiment.status = SyncLaneFrontierStatus::UnresolvedLane;
    } else {
        const bool sourceFound = deriveFrontier(
            schedule, source, target, FrontierSide::After, experiment.sourceFrontier, experiment.sourcePlacement);
        const bool targetFound = deriveFrontier(
            schedule, target, source, FrontierSide::Before, experiment.targetFrontier, experiment.targetPlacement);
        if (!sourceFound || !targetFound) {
            experiment.status = SyncLaneFrontierStatus::UnsupportedControl;
        } else if (experiment.topology == SyncLaneTopology::CrossCore) {
            experiment.status = SyncLaneFrontierStatus::CrossCore;
        } else {
            experiment.status = SyncLaneFrontierStatus::Found;
        }
    }
    updateStatistics(experiment, statistics);
    return experiment;
}

SmallVector<SyncLaneFrontierExperiment, 4> buildReadyExperiments(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncGenerationTimeline& timeline,
    SyncChannelRejection channelRejection, const LaneFrontierAnalysisResult& result,
    SyncLaneFrontierExperimentId firstId, ProtocolSyncStatistics* statistics)
{
    SmallVector<SyncLaneFrontierExperiment, 4> experiments;
    const UniqueLane sourceLane = getUniqueLane(timeline.producers, stages, result);
    for (const StageLaneGroup& targets : groupStagesByLane(timeline.consumers, stages, result)) {
        SyncLaneFrontierExperiment experiment;
        experiment.generation = timeline.id;
        experiment.demand = SyncLaneDemandKind::Ready;
        experiment.sourceLane = sourceLane.id;
        experiment.targetLane = targets.lane;
        experiment.sourceStages = timeline.producers;
        experiment.targetStages = targets.stages;
        experiment.channelRejection = channelRejection;
        SyncProgramFrontier target = selectFrontiers(timeline.acquisitions, targets.indices);
        experiments.push_back(finishExperiment(
            schedule, result, std::move(experiment), firstId + experiments.size(), timeline.publication, target,
            sourceLane.multiple, false, statistics));
    }
    return experiments;
}

SmallVector<SyncLaneFrontierExperiment, 4> buildReleaseExperiments(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncGenerationTimeline& timeline,
    SyncChannelRejection channelRejection, const LaneFrontierAnalysisResult& result,
    SyncLaneFrontierExperimentId firstId, ProtocolSyncStatistics* statistics)
{
    SmallVector<SyncLaneFrontierExperiment, 4> experiments;
    if (!timeline.nextOverwrite) {
        return experiments;
    }
    const UniqueLane targetLane = getUniqueLane(timeline.producers, stages, result);
    for (const StageLaneGroup& sources : groupStagesByLane(timeline.consumers, stages, result)) {
        SyncLaneFrontierExperiment experiment;
        experiment.generation = timeline.id;
        experiment.demand = SyncLaneDemandKind::Release;
        experiment.sourceLane = sources.lane;
        experiment.targetLane = targetLane.id;
        experiment.sourceStages.assign(sources.stages.begin(), sources.stages.end());
        experiment.targetStages.assign(timeline.producers.begin(), timeline.producers.end());
        experiment.channelRejection = channelRejection;
        experiment.iterationDistance = timeline.nextOverwrite->iterationDistance;
        SyncProgramFrontier source = selectFrontiers(timeline.finalUses, sources.indices);
        experiments.push_back(finishExperiment(
            schedule, result, std::move(experiment), firstId + experiments.size(), source,
            timeline.nextOverwrite->frontier, false, targetLane.multiple, statistics));
    }
    return experiments;
}

SyncLaneFrontierExperiment buildResidualExperiment(
    const SyncGenerationTimeline& timeline, SyncChannelRejection channelRejection, SyncLaneFrontierExperimentId id,
    ProtocolSyncStatistics* statistics)
{
    SyncLaneFrontierExperiment experiment;
    experiment.id = id;
    experiment.generation = timeline.id;
    experiment.demand = SyncLaneDemandKind::Residual;
    experiment.status = SyncLaneFrontierStatus::TimelineRejected;
    experiment.timelineRejection = timeline.rejection;
    experiment.channelRejection = channelRejection;
    updateStatistics(experiment, statistics);
    return experiment;
}

} // namespace

const SyncExecutionLane* LaneFrontierAnalysisResult::findLane(SyncExecutionLaneId id) const
{
    return id < lanes.size() ? &lanes[id] : nullptr;
}

const SyncExecutionLane* LaneFrontierAnalysisResult::findLaneForPhase(SyncPhaseId phase) const
{
    return phase < phaseToLane.size() ? findLane(phaseToLane[phase]) : nullptr;
}

LaneFrontierAnalysisResult mlir::pto::protocol_sync::analyzeLaneFrontiers(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    ProtocolSyncStatistics* statistics)
{
    LaneFrontierAnalysisResult result;
    result.phaseToLane.assign(schedule.getPhases().size(), kInvalidSyncId);
    for (const SyncPhase& phase : schedule.getPhases()) {
        SyncExecutionLaneId laneId = findExistingLane(result, phase.core, phase.pipe);
        if (laneId == kInvalidSyncId) {
            laneId = result.lanes.size();
            result.lanes.push_back({laneId, phase.core, phase.pipe, {}});
        }
        const SyncStage* stage = stages.findStageForPhase(phase.id);
        SyncLaneOccurrence occurrence;
        occurrence.phase = phase.id;
        occurrence.stage = stage ? stage->id : kInvalidSyncId;
        occurrence.region = phase.region;
        occurrence.before = phase.before;
        occurrence.after = phase.after;
        occurrence.guard = phase.guard;
        occurrence.iterationDomain = phase.iterationDomain;
        result.lanes[laneId].occurrences.push_back(std::move(occurrence));
        result.phaseToLane[phase.id] = laneId;
    }
    if (statistics) {
        statistics->executionLanes = result.lanes.size();
        statistics->laneOccurrences = schedule.getPhases().size();
    }

    for (const SyncGenerationTimeline& timeline : timelines.getTimelines()) {
        const SyncChannelRejection channelRejection = findChannelRejection(timeline.id, channels);
        if (!timeline.isAdmitted()) {
            result.experiments.push_back(
                buildResidualExperiment(timeline, channelRejection, result.experiments.size(), statistics));
            continue;
        }
        SmallVector<SyncLaneFrontierExperiment, 4> ready = buildReadyExperiments(
            schedule, stages, timeline, channelRejection, result, result.experiments.size(), statistics);
        result.experiments.append(std::make_move_iterator(ready.begin()), std::make_move_iterator(ready.end()));
        SmallVector<SyncLaneFrontierExperiment, 4> release = buildReleaseExperiments(
            schedule, stages, timeline, channelRejection, result, result.experiments.size(), statistics);
        result.experiments.append(std::make_move_iterator(release.begin()), std::make_move_iterator(release.end()));
    }
    return result;
}

StringRef mlir::pto::protocol_sync::stringifySyncLaneDemandKind(SyncLaneDemandKind kind)
{
    switch (kind) {
        case SyncLaneDemandKind::Ready:
            return "ready";
        case SyncLaneDemandKind::Release:
            return "release";
        case SyncLaneDemandKind::Residual:
            return "residual";
    }
    return "residual";
}

StringRef mlir::pto::protocol_sync::stringifySyncLaneTopology(SyncLaneTopology topology)
{
    switch (topology) {
        case SyncLaneTopology::SameLane:
            return "same-lane";
        case SyncLaneTopology::CrossLane:
            return "cross-lane";
        case SyncLaneTopology::CrossCore:
            return "cross-core";
        case SyncLaneTopology::Unknown:
            return "unknown";
    }
    return "unknown";
}

StringRef mlir::pto::protocol_sync::stringifySyncFrontierPlacement(SyncFrontierPlacement placement)
{
    switch (placement) {
        case SyncFrontierPlacement::Exact:
            return "exact";
        case SyncFrontierPlacement::LinearCoalesced:
            return "linear-coalesced";
        case SyncFrontierPlacement::ChoiceEntry:
            return "choice-entry";
        case SyncFrontierPlacement::ChoiceExit:
            return "choice-exit";
        case SyncFrontierPlacement::Unavailable:
            return "unavailable";
    }
    return "unavailable";
}

StringRef mlir::pto::protocol_sync::stringifySyncLaneFrontierStatus(SyncLaneFrontierStatus status)
{
    switch (status) {
        case SyncLaneFrontierStatus::Found:
            return "found";
        case SyncLaneFrontierStatus::TimelineRejected:
            return "timeline-rejected";
        case SyncLaneFrontierStatus::UnresolvedSchedule:
            return "unresolved-schedule";
        case SyncLaneFrontierStatus::MissingEndpoint:
            return "missing-endpoint";
        case SyncLaneFrontierStatus::UnresolvedLane:
            return "unresolved-lane";
        case SyncLaneFrontierStatus::MultipleSourceLanes:
            return "multiple-source-lanes";
        case SyncLaneFrontierStatus::MultipleTargetLanes:
            return "multiple-target-lanes";
        case SyncLaneFrontierStatus::UnsupportedControl:
            return "unsupported-control";
        case SyncLaneFrontierStatus::CrossCore:
            return "cross-core";
    }
    return "missing-endpoint";
}
