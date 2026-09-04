// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LaneFrontierDump.cpp - Stable lane/frontier diagnostics ---------===//

#include "PTO/Transforms/ProtocolSync/LaneFrontierAnalysis.h"

#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;
using namespace llvm;

namespace {

void printGuard(ArrayRef<SyncControlAtom> guard, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(
        guard, output, [&](const SyncControlAtom& atom) { output << '#' << atom.choice << ':' << atom.arm; });
    output << ']';
}

void printRegionIds(ArrayRef<SyncRegionId> ids, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(ids, output, [&](SyncRegionId id) { output << '#' << id; });
    output << ']';
}

void printStageIds(ArrayRef<SyncStageId> ids, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(ids, output, [&](SyncStageId id) { output << '#' << id; });
    output << ']';
}

void printLaneId(SyncExecutionLaneId id, raw_ostream& output)
{
    if (id == kInvalidSyncId) {
        output << "unknown";
    } else {
        output << '#' << id;
    }
}

void printFrontier(const SyncProgramFrontier& frontier, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(frontier.points, output, [&](const SyncGuardedProgramPoint& point) {
        output << "pp" << point.point << " guard=";
        printGuard(point.guard, output);
    });
    output << ']';
}

void printLane(const SyncExecutionLane& lane, raw_ostream& output)
{
    output << "  execution-lane #" << lane.id << " core=" << stringifySyncPhysicalCore(lane.core)
           << " pipe=" << stringifyPIPE(lane.pipe)
           << " order=structured-partial occurrences=" << lane.occurrences.size() << '\n';
    for (const SyncLaneOccurrence& occurrence : lane.occurrences) {
        output << "    occurrence phase=#" << occurrence.phase << " stage=#" << occurrence.stage << " region=#"
               << occurrence.region << " before=pp" << occurrence.before << " after=pp" << occurrence.after
               << " guard=";
        printGuard(occurrence.guard, output);
        output << " loops=";
        printRegionIds(occurrence.iterationDomain.loops, output);
        output << '\n';
    }
}

void printExperiment(const SyncLaneFrontierExperiment& experiment, raw_ostream& output)
{
    output << "  experiment #" << experiment.id << " generation=#" << experiment.generation
           << " demand=" << stringifySyncLaneDemandKind(experiment.demand) << " source-execution-lane=";
    printLaneId(experiment.sourceLane, output);
    output << " target-execution-lane=";
    printLaneId(experiment.targetLane, output);
    output << " topology=" << stringifySyncLaneTopology(experiment.topology)
           << " status=" << stringifySyncLaneFrontierStatus(experiment.status) << '\n';
    output << "    source-stages=";
    printStageIds(experiment.sourceStages, output);
    output << " source-frontier=";
    printFrontier(experiment.sourceFrontier, output);
    output << " placement=" << stringifySyncFrontierPlacement(experiment.sourcePlacement) << '\n';
    output << "    target-stages=";
    printStageIds(experiment.targetStages, output);
    output << " target-frontier=";
    printFrontier(experiment.targetFrontier, output);
    output << " placement=" << stringifySyncFrontierPlacement(experiment.targetPlacement) << '\n';
    output << "    channel-reason=" << stringifySyncChannelRejection(experiment.channelRejection)
           << " timeline-reason=" << stringifySyncTimelineRejection(experiment.timelineRejection)
           << " iteration-distance=" << experiment.iterationDistance
           << " target-proof=" << (experiment.isFound() ? "required" : "not-applicable") << " selectable=no\n";
}

bool frontierIsPoint(const SyncProgramFrontier& frontier, SyncProgramPointId point)
{
    return frontier.points.size() == 1 && frontier.points.front().point == point &&
           frontier.points.front().guard.empty();
}

const SyncLaneFrontierExperiment* findUniqueExperiment(
    const LaneFrontierAnalysisResult& analysis, SyncGenerationId generation, SyncLaneDemandKind demand)
{
    const SyncLaneFrontierExperiment* match = nullptr;
    for (const SyncLaneFrontierExperiment& experiment : analysis.getExperiments()) {
        if (experiment.generation != generation || experiment.demand != demand || !experiment.isFound()) {
            continue;
        }
        if (match) {
            return nullptr;
        }
        match = &experiment;
    }
    return match;
}

} // namespace

void mlir::pto::protocol_sync::printLaneFrontierAnalysis(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& analysis, raw_ostream& output)
{
    output << "PROTOCOL-SYNC lane-frontiers function=@" << schedule.getFunction().getSymName() << '\n';
    output << "  semantics=structural-only lane-order=partial target-ordering=unproven selectable=no\n";
    for (const SyncExecutionLane& lane : analysis.getLanes()) {
        printLane(lane, output);
    }
    for (const SyncLaneFrontierExperiment& experiment : analysis.getExperiments()) {
        printExperiment(experiment, output);
    }
    output << "PROTOCOL-SYNC lane-frontiers-end function=@" << schedule.getFunction().getSymName() << '\n';
}

void mlir::pto::protocol_sync::printReadyReleaseFrontierComparison(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& analysis, const SyncReadyReleasePlan& plan,
    raw_ostream& output)
{
    const SyncPhase* producer = schedule.findPhase(plan.producerPhase);
    const SyncPhase* consumer = schedule.findPhase(plan.consumerPhase);
    const SyncExecutionLane* producerLane = analysis.findLaneForPhase(plan.producerPhase);
    const SyncExecutionLane* consumerLane = analysis.findLaneForPhase(plan.consumerPhase);
    const SyncLaneFrontierExperiment* ready =
        findUniqueExperiment(analysis, plan.generation, SyncLaneDemandKind::Ready);
    const SyncLaneFrontierExperiment* release =
        findUniqueExperiment(analysis, plan.generation, SyncLaneDemandKind::Release);

    const bool endpointLanesMatch = producer && consumer && producerLane && consumerLane &&
                                    producerLane->core == plan.core && consumerLane->core == plan.core &&
                                    producerLane->pipe == plan.producerPipe && consumerLane->pipe == plan.consumerPipe;
    const bool directionsMatch = endpointLanesMatch && ready && release && ready->sourceLane == producerLane->id &&
                                 ready->targetLane == consumerLane->id && release->sourceLane == consumerLane->id &&
                                 release->targetLane == producerLane->id;
    const bool publicationMatches = producer && ready && frontierIsPoint(ready->sourceFrontier, producer->after);
    const bool acquisitionMatches = consumer && ready && frontierIsPoint(ready->targetFrontier, consumer->before);
    const bool finalUseMatches = consumer && release && frontierIsPoint(release->sourceFrontier, consumer->after);
    const bool nextOverwriteMatches = producer && release &&
                                      frontierIsPoint(release->targetFrontier, producer->before) &&
                                      release->iterationDistance == plan.capacity;
    const bool matches = plan.status == SyncReadyReleasePlanStatus::Ready && endpointLanesMatch && directionsMatch &&
                         publicationMatches && acquisitionMatches && finalUseMatches && nextOverwriteMatches;

    output << "PROTOCOL-SYNC lane-ready-release-differential function=@" << schedule.getFunction().getSymName()
           << " status=" << (matches ? "match" : "mismatch") << " capacity=" << plan.capacity << '\n';
    output << "  endpoint-execution-lanes=" << (endpointLanesMatch ? "match" : "mismatch")
           << " directions=" << (directionsMatch ? "match" : "mismatch")
           << " publication=" << (publicationMatches ? "match" : "mismatch")
           << " acquisition=" << (acquisitionMatches ? "match" : "mismatch")
           << " final-use=" << (finalUseMatches ? "match" : "mismatch")
           << " next-overwrite=" << (nextOverwriteMatches ? "match" : "mismatch") << '\n';
}
