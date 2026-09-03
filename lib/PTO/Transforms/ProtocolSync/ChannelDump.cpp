// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ChannelDump.cpp - Deterministic timeline/channel diagnostics -----===//

#include "PTO/Transforms/ProtocolSync/ChannelProtocolIR.h"

#include "mlir/IR/AsmState.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;
using namespace llvm;

namespace {

void printIntervals(ArrayRef<SyncByteInterval> intervals, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(
        intervals, output, [&](const SyncByteInterval& interval) { output << interval.begin << '+' << interval.size; });
    output << ']';
}

void printStageIds(ArrayRef<SyncStageId> ids, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(ids, output, [&](SyncStageId id) { output << '#' << id; });
    output << ']';
}

void printGuard(ArrayRef<SyncControlAtom> guard, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(
        guard, output, [&](const SyncControlAtom& atom) { output << '#' << atom.choice << ':' << atom.arm; });
    output << ']';
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

void printFrontiers(ArrayRef<SyncProgramFrontier> frontiers, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(
        frontiers, output, [&](const SyncProgramFrontier& frontier) { printFrontier(frontier, output); });
    output << ']';
}

void printSlot(const std::optional<SyncSlotExpression>& slot, AsmState& state, raw_ostream& output)
{
    if (!slot) {
        output << "implicit-depth-one";
        return;
    }
    slot->selector.printAsOperand(output, state);
    output << '/' << slot->depth << " expr=" << stringifySyncSlotExpressionKind(slot->kind);
    if (slot->kind == SyncSlotExpressionKind::Constant) {
        output << "(offset=" << slot->offset << ",modulus=" << slot->modulus << ')';
    } else if (slot->kind == SyncSlotExpressionKind::AffineModulo) {
        output << "(loop=#" << slot->loop << ",coefficient=" << slot->coefficient << ",offset=" << slot->offset
               << ",modulus=" << slot->modulus << ')';
    }
}

void printStage(const SyncStage& stage, raw_ostream& output)
{
    output << "  stage #" << stage.id << " region=#" << stage.region << " role=" << stringifySyncStageRole(stage.role)
           << " core=" << stringifySyncPhysicalCore(stage.core) << " pipe=" << stringifyPIPE(stage.pipe) << " phases=";
    output << '[';
    llvm::interleaveComma(stage.phases, output, [&](SyncPhaseId id) { output << '#' << id; });
    output << "] accesses=";
    output << '[';
    llvm::interleaveComma(stage.accesses, output, [&](SyncAccessId id) { output << '#' << id; });
    output << "] guard=";
    printGuard(stage.guard, output);
    output << " loops=";
    output << '[';
    llvm::interleaveComma(stage.iterationDomain.loops, output, [&](SyncRegionId id) { output << '#' << id; });
    output << "]\n";
}

void printTimeline(
    const StructuredSyncIR& schedule, const SyncGenerationTimeline& timeline, AsmState& state, raw_ostream& output)
{
    const SyncStorageFamily* family = schedule.findStorageFamily(timeline.family);
    output << "  timeline #" << timeline.id << " family=#" << timeline.family
           << " generation=" << stringifySyncGenerationKind(timeline.generationKind)
           << " status=" << (timeline.isAdmitted() ? "admitted" : "rejected")
           << " reason=" << stringifySyncTimelineRejection(timeline.rejection) << " slice=";
    printIntervals(timeline.slice, output);
    output << " slot=";
    printSlot(timeline.slot, state, output);
    output << " carrying-loop=";
    if (timeline.carryingRegion == kInvalidSyncId) {
        output << "none";
    } else {
        output << '#' << timeline.carryingRegion;
    }
    output << " producers=";
    printStageIds(timeline.producers, output);
    output << " consumers=";
    printStageIds(timeline.consumers, output);
    output << " role=" << (family ? stringifySyncStorageRole(family->role) : "unknown");
    output << '\n';
    if (!timeline.isAdmitted()) {
        return;
    }
    output << "    publication=";
    printFrontier(timeline.publication, output);
    output << " acquisitions=";
    printFrontiers(timeline.acquisitions, output);
    output << " final-uses=";
    printFrontiers(timeline.finalUses, output);
    output << " next-overwrite=";
    if (timeline.nextOverwrite) {
        printFrontier(timeline.nextOverwrite->frontier, output);
        output << " distance=" << timeline.nextOverwrite->iterationDistance;
    } else {
        output << "none";
    }
    output << " reuse=";
    if (timeline.reuseDistance) {
        output << *timeline.reuseDistance;
    } else {
        output << "none";
    }
    output << '\n';
}

void printChannel(const SyncChannel& channel, const StorageTimelineAnalysisResult& timelines, raw_ostream& output)
{
    output << "  channel #" << channel.id << " generation=#" << channel.generation
           << " kind=" << stringifySyncChannelKind(channel.kind) << " capacity=" << channel.capacity
           << " status=" << (channel.isAdmitted() ? "admitted" : "rejected")
           << " reason=" << stringifySyncChannelRejection(channel.rejection);
    if (channel.generation < timelines.getTimelines().size()) {
        const SyncGenerationTimeline& timeline = timelines.getTimelines()[channel.generation];
        output << " producers=";
        printStageIds(timeline.producers, output);
        output << " consumers=";
        printStageIds(timeline.consumers, output);
    }
    output << " oracle-ready=" << stringifySyncDemandOracleStatus(channel.readyOracle)
           << " oracle-release=" << stringifySyncDemandOracleStatus(channel.releaseOracle) << '\n';
}

} // namespace

void mlir::pto::protocol_sync::printProtocolSyncChannels(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels, raw_ostream& output)
{
    output << "PROTOCOL-SYNC channels function=@" << schedule.getFunction().getSymName() << '\n';
    AsmState state(schedule.getFunction());
    for (const SyncStage& stage : stages.getStages()) {
        printStage(stage, output);
    }
    for (const SyncGenerationTimeline& timeline : timelines.getTimelines()) {
        printTimeline(schedule, timeline, state, output);
    }
    for (const SyncChannel& channel : channels.getChannels()) {
        printChannel(channel, timelines, output);
    }
    output << "PROTOCOL-SYNC channels-end function=@" << schedule.getFunction().getSymName() << '\n';
}
