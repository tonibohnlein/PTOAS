// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LanePatternDump.cpp - Stable lane pattern diagnostics -----------===//

#include "PTO/Transforms/ProtocolSync/LanePatternAnalysis.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto::protocol_sync;
using namespace llvm;

namespace {

void printIds(ArrayRef<std::uint32_t> ids, StringRef prefix, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(ids, output, [&](std::uint32_t id) { output << prefix << id; });
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
        output << "pp" << point.point << " guard=[";
        llvm::interleaveComma(
            point.guard, output, [&](const SyncControlAtom& atom) { output << '#' << atom.choice << ':' << atom.arm; });
        output << ']';
    });
    output << ']';
}

void printPointDescription(const StructuredSyncIR& schedule, SyncProgramPointId id, raw_ostream& output)
{
    if (id >= schedule.getProgramPoints().size()) {
        output << "unknown(pp" << id << ')';
        return;
    }
    const SyncProgramPoint& point = schedule.getProgramPoints()[id];
    Operation* operation = nullptr;
    bool before = false;
    switch (point.kind) {
        case SyncProgramPointKind::RegionEntry:
        case SyncProgramPointKind::RegionExit: {
            const SyncRegion* region = schedule.findRegion(point.region);
            operation = region ? region->operation : nullptr;
            before = point.kind == SyncProgramPointKind::RegionEntry;
            break;
        }
        case SyncProgramPointKind::SemanticActionBefore:
        case SyncProgramPointKind::SemanticActionAfter: {
            const SyncSemanticAction* action = schedule.findSemanticAction(point.action);
            operation = action ? action->operation : nullptr;
            before = point.kind == SyncProgramPointKind::SemanticActionBefore;
            break;
        }
        case SyncProgramPointKind::PhaseBefore:
        case SyncProgramPointKind::PhaseAfter: {
            const SyncPhase* phase = schedule.findPhase(point.phase);
            operation = phase ? phase->operation : nullptr;
            before = point.kind == SyncProgramPointKind::PhaseBefore;
            break;
        }
    }
    if (!operation) {
        output << "unknown(pp" << id << ')';
        return;
    }
    output << (before ? "before(" : "after(") << operation->getName() << ')';
}

void printReferenceEndpoint(const StructuredSyncIR& schedule, const SyncProgramFrontier& frontier, raw_ostream& output)
{
    const bool isSinglePoint = frontier.points.size() == 1;
    if (!isSinglePoint) {
        output << "multi-point";
        return;
    }
    printPointDescription(schedule, frontier.points.front().point, output);
}

void printReferencePlacement(
    const StructuredSyncIR& schedule, const SyncLanePatternCandidate& candidate, raw_ostream& output)
{
    output << "    reference-placement=";
    printReferenceEndpoint(schedule, candidate.firstSource, output);
    if (candidate.kind != SyncLanePatternKind::SameLaneCompletionCut) {
        output << "->";
        printReferenceEndpoint(schedule, candidate.firstTarget, output);
    }
    if (candidate.kind == SyncLanePatternKind::ChoiceBalancedRoundTrip) {
        output << ';';
        printReferenceEndpoint(schedule, candidate.secondSource, output);
        output << "->";
        printReferenceEndpoint(schedule, candidate.secondTarget, output);
    }
    output << '\n';
}

void printRawEndpoints(const SyncGenerationTimeline& timeline, raw_ostream& output)
{
    output << "  raw-endpoints generation=#" << timeline.id
           << " timeline-reason=" << stringifySyncTimelineRejection(timeline.rejection) << " endpoints=[";
    llvm::interleaveComma(timeline.rawAccesses, output, [&](const SyncRawAccessEndpoint& endpoint) {
        output << "a#" << endpoint.access << " phase=#" << endpoint.phase << " stage=#" << endpoint.stage
               << " before=pp" << endpoint.before << " after=pp" << endpoint.after;
    });
    output << "]\n";
}

void printRawPair(const SyncLaneRawAccessPair& pair, raw_ostream& output)
{
    output << "  raw-pair #" << pair.id << " hazard=" << stringifySyncLaneRawHazardKind(pair.hazard) << " accesses=a#"
           << pair.sourceAccess << "->a#" << pair.targetAccess << " generations=#" << pair.sourceGeneration << "->#"
           << pair.targetGeneration << " phases=#" << pair.sourcePhase << "->#" << pair.targetPhase << " lanes=";
    printLaneId(pair.sourceLane, output);
    output << "->";
    printLaneId(pair.targetLane, output);
    output << " interval=after(pp" << pair.sourceAfter << ")->before(pp" << pair.targetBefore
           << ") completion-query=" << stringifyProtocolSyncSameLaneCompletion(pair.completion) << '\n';
}

void printCandidateFrontiers(const SyncLanePatternCandidate& candidate, raw_ostream& output)
{
    if (candidate.kind == SyncLanePatternKind::SameLaneCompletionCut) {
        output << "    completion-cut=";
        printFrontier(candidate.firstSource, output);
        output << '\n';
        return;
    }
    output << "    ready-source=";
    printFrontier(candidate.firstSource, output);
    output << " ready-target=";
    printFrontier(candidate.firstTarget, output);
    output << '\n';
    if (candidate.kind == SyncLanePatternKind::ChoiceBalancedRoundTrip) {
        output << "    release-source=";
        printFrontier(candidate.secondSource, output);
        output << " release-target=";
        printFrontier(candidate.secondTarget, output);
        output << '\n';
    }
}

void printCandidate(const StructuredSyncIR& schedule, const SyncLanePatternCandidate& candidate, raw_ostream& output)
{
    output << "  lane-pattern #" << candidate.id << " kind=" << stringifySyncLanePatternKind(candidate.kind)
           << " old-pattern=" << stringifySyncLaneReferencePattern(candidate.referencePattern)
           << " old-mechanism=" << stringifySyncLaneReferenceMechanism(candidate.kind) << " lanes=";
    printLaneId(candidate.sourceLane, output);
    output << "->";
    printLaneId(candidate.targetLane, output);
    output << '\n';
    printCandidateFrontiers(candidate, output);
    printReferencePlacement(schedule, candidate, output);
    output << "    generations=";
    printIds(candidate.generations, "#", output);
    output << " frontier-members=";
    printIds(candidate.frontierMembers, "e#", output);
    output << " raw-pair-members=";
    printIds(candidate.rawPairMembers, "r#", output);
    output << '\n';
    output << "    target-query=" << stringifySyncLaneTargetQuery(candidate.targetQuery)
           << " checkpoint-e=" << stringifySyncCheckpointEStatus(candidate.checkpointE)
           << " checkpoint-e-reason=" << stringifySyncChannelRejection(candidate.checkpointERejection)
           << " cost-logical=" << candidate.cost.logicalCandidates
           << " cost-steady-actions=" << candidate.cost.steadyStateActions << " selectable=no\n";
}

} // namespace

void mlir::pto::protocol_sync::printLanePatternAnalysis(
    const StructuredSyncIR& schedule, const StorageTimelineAnalysisResult& timelines,
    const LanePatternAnalysisResult& analysis, raw_ostream& output)
{
    output << "PROTOCOL-SYNC lane-patterns function=@" << schedule.getFunction().getSymName() << '\n';
    output << "  semantics=diagnostic-only raw-order=linear-block-only target-query=advisory selectable=no\n";
    for (const SyncGenerationTimeline& timeline : timelines.getTimelines()) {
        printRawEndpoints(timeline, output);
    }
    for (const SyncLaneRawAccessPair& pair : analysis.getRawAccessPairs()) {
        printRawPair(pair, output);
    }
    for (const SyncLanePatternCandidate& candidate : analysis.getCandidates()) {
        printCandidate(schedule, candidate, output);
    }
    output << "PROTOCOL-SYNC lane-patterns-end function=@" << schedule.getFunction().getSymName() << '\n';
}
