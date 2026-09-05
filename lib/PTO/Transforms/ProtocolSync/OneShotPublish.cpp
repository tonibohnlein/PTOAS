// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- OneShotPublish.cpp - Channel-scoped one-shot protocols ----------===//

#include "PTO/Transforms/ProtocolSync/OneShotPublish.h"

#include "PTO/IR/PTO.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <chrono>
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using OneShotPublishClock = std::chrono::steady_clock;
using CandidateKey = std::tuple<std::uint8_t, std::uint8_t, std::uint8_t, SyncPhaseId, SyncPhaseId, std::uint8_t>;

constexpr StringLiteral kGeneratedAttr = "pto.protocol_sync.generated";
constexpr StringLiteral kProtocolAttr = "pto.protocol_sync.protocol_id";
constexpr StringLiteral kProtocolKindAttr = "pto.protocol_sync.protocol_kind";
constexpr StringLiteral kProtocolKind = "one-shot-publish";
constexpr StringLiteral kRoleAttr = "pto.protocol_sync.role";

std::uint64_t elapsedMicroseconds(OneShotPublishClock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(OneShotPublishClock::now() - start).count();
}

const SyncGenerationTimeline* findTimeline(const StorageTimelineAnalysisResult& timelines, SyncGenerationId generation)
{
    ArrayRef<SyncGenerationTimeline> records = timelines.getTimelines();
    const bool valid = generation < records.size() && records[generation].id == generation;
    return valid ? &records[generation] : nullptr;
}

const SyncChannel* findChannel(const ChannelAnalysisResult& channels, SyncChannelId id)
{
    ArrayRef<SyncChannel> records = channels.getChannels();
    const bool valid = id < records.size() && records[id].id == id;
    return valid ? &records[id] : nullptr;
}

const SyncPhase* getOnlyPhase(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, SyncStageId stageId)
{
    const SyncStage* stage = stages.findStage(stageId);
    const bool invalidStage = !stage || stage->phases.size() != 1;
    if (invalidStage) {
        return nullptr;
    }
    const SyncPhase* phase = schedule.findPhase(stage->phases.front());
    return phase && phase->completion == SyncCompletionKind::PhaseEnd ? phase : nullptr;
}

bool frontierIs(const SyncProgramFrontier& frontier, SyncProgramPointId point)
{
    return frontier.points.size() == 1 && frontier.points.front().point == point &&
           frontier.points.front().guard.empty();
}

bool operationsAreOrdered(Operation* source, Operation* target)
{
    return source && target && source != target && source->getBlock() == target->getBlock() &&
           source->isBeforeInBlock(target);
}

std::optional<SyncOneShotPublishKind> implementationKind(
    const ProtocolSyncTarget& target, const SyncPhase& source, const SyncPhase& targetPhase)
{
    if (source.pipe == targetPhase.pipe && source.pipe == PIPE::PIPE_S) {
        return SyncOneShotPublishKind::IntrinsicOrder;
    }
    if (source.pipe == targetPhase.pipe) {
        return target.supportsPipeBarrier({source.core, source.pipe}) ?
                   std::optional<SyncOneShotPublishKind>(SyncOneShotPublishKind::PipeBarrier) :
                   std::nullopt;
    }
    return target.supportsEvent({source.core, source.pipe}, {targetPhase.core, targetPhase.pipe}) ?
               std::optional<SyncOneShotPublishKind>(SyncOneShotPublishKind::DirectedEvent) :
               std::nullopt;
}

CandidateKey candidateKey(const SyncOneShotPublishCandidate& candidate)
{
    return {
        static_cast<std::uint8_t>(candidate.core),
        static_cast<std::uint8_t>(candidate.sourcePipe),
        static_cast<std::uint8_t>(candidate.targetPipe),
        candidate.sourcePhase,
        candidate.targetPhase,
        static_cast<std::uint8_t>(candidate.kind)};
}

bool candidateMatches(const SyncOneShotPublishCandidate& selected, const SyncOneShotPublishCandidate& authoritative)
{
    return candidateKey(selected) == candidateKey(authoritative) &&
           selected.sourceOperation == authoritative.sourceOperation &&
           selected.targetOperation == authoritative.targetOperation && selected.channels == authoritative.channels &&
           selected.generations == authoritative.generations;
}

bool isReserved(const StructuredSyncIR& schedule, PIPE source, PIPE target, unsigned eventId)
{
    return llvm::any_of(schedule.getSummaries(), [&](const SyncOpSummary& summary) {
        return llvm::any_of(summary.eventReservations, [&](const SyncEventReservation& reservation) {
            return reservation.source == source && reservation.target == target &&
                   llvm::is_contained(reservation.eventIds, eventId);
        });
    });
}

void tagGenerated(Operation* operation, OpBuilder& builder, SyncOneShotPublishId candidate, StringRef role)
{
    operation->setAttr(kGeneratedAttr, builder.getUnitAttr());
    operation->setAttr(kProtocolAttr, builder.getI32IntegerAttr(static_cast<std::int32_t>(candidate)));
    operation->setAttr(kProtocolKindAttr, builder.getStringAttr(kProtocolKind));
    operation->setAttr(kRoleAttr, builder.getStringAttr(role));
}

bool isBefore(Operation* first, Operation* second)
{
    return first && second && first->getBlock() == second->getBlock() && first->isBeforeInBlock(second);
}

bool onlyGeneratedBetween(Operation* first, Operation* second)
{
    if (!isBefore(first, second)) {
        return false;
    }
    for (Operation* operation = first->getNextNode(); operation && operation != second;
         operation = operation->getNextNode()) {
        if (!operation->hasAttrOfType<UnitAttr>(kGeneratedAttr)) {
            return false;
        }
    }
    return true;
}

struct ConcreteCandidate {
    Operation* barrier = nullptr;
    Operation* set = nullptr;
    Operation* wait = nullptr;
};

LogicalResult collectConcreteCandidates(
    func::FuncOp clone, unsigned candidateCount, llvm::DenseMap<SyncOneShotPublishId, ConcreteCandidate>& records)
{
    bool malformed = false;
    clone.walk([&](Operation* operation) {
        auto kind = operation->getAttrOfType<StringAttr>(kProtocolKindAttr);
        const bool belongsToCandidate = kind && kind.getValue() == kProtocolKind;
        if (!belongsToCandidate) {
            return;
        }
        auto id = operation->getAttrOfType<IntegerAttr>(kProtocolAttr);
        auto role = operation->getAttrOfType<StringAttr>(kRoleAttr);
        const bool valid = isFixedSyncOperation(operation) && operation->hasAttrOfType<UnitAttr>(kGeneratedAttr) &&
                           id && id.getInt() >= 0 && static_cast<std::uint64_t>(id.getInt()) < candidateCount && role;
        if (!valid) {
            malformed = true;
            return;
        }
        ConcreteCandidate& record = records[static_cast<SyncOneShotPublishId>(id.getInt())];
        const StringRef roleValue = role.getValue();
        if (roleValue == "publish-barrier") {
            malformed |= record.barrier || !isa<BarrierOp>(operation);
            record.barrier = operation;
        } else if (roleValue == "publish-set") {
            malformed |= record.set || !isa<SetFlagOp>(operation);
            record.set = operation;
        } else if (roleValue == "publish-wait") {
            malformed |= record.wait || !isa<WaitFlagOp>(operation);
            record.wait = operation;
        } else {
            malformed = true;
        }
    });
    return failure(malformed);
}

} // namespace

FailureOr<SyncOneShotPublishPlan> mlir::pto::protocol_sync::buildOneShotPublishCandidates(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels)
{
    if (!schedule.isFrozen()) {
        return failure();
    }
    SyncOneShotPublishPlan plan;
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    const bool unsupportedSchedule = !target.isSupported() || !schedule.getFailures().empty();
    if (unsupportedSchedule) {
        return plan;
    }

    std::map<CandidateKey, unsigned> candidateByFrontier;
    for (const SyncChannel& channel : channels.getChannels()) {
        const SyncGenerationTimeline* timeline = findTimeline(timelines, channel.generation);
        const bool candidateChannel =
            timeline && timeline->isAdmitted() && channel.isAdmitted() && channel.kind == SyncChannelKind::OneShot &&
            channel.capacity != 0 && timeline->generationKind == SyncGenerationKind::OneShot &&
            timeline->producers.size() == 1 && timeline->consumers.size() == 1 && !timeline->nextOverwrite &&
            channel.readyOracle == SyncDemandOracleStatus::Match &&
            channel.releaseOracle != SyncDemandOracleStatus::Mismatch;
        if (!candidateChannel) {
            continue;
        }
        const SyncStorageFamily* family = schedule.findStorageFamily(timeline->family);
        const SyncPhase* producer = getOnlyPhase(schedule, stages, timeline->producers.front());
        const SyncPhase* consumer = getOnlyPhase(schedule, stages, timeline->consumers.front());
        const bool exactLocalFrontiers =
            family && family->role == SyncStorageRole::LocalBuffer && producer && consumer &&
            producer->core == consumer->core && producer->core != SyncPhysicalCore::Unknown &&
            producer->guard.empty() && consumer->guard.empty() && producer->iterationDomain.loops.empty() &&
            consumer->iterationDomain.loops.empty() && frontierIs(timeline->publication, producer->after) &&
            timeline->acquisitions.size() == 1 && frontierIs(timeline->acquisitions.front(), consumer->before) &&
            operationsAreOrdered(producer->operation, consumer->operation);
        if (!exactLocalFrontiers) {
            continue;
        }
        std::optional<SyncOneShotPublishKind> kind = implementationKind(target, *producer, *consumer);
        if (!kind) {
            continue;
        }
        SyncOneShotPublishCandidate candidate;
        candidate.kind = *kind;
        candidate.core = producer->core;
        candidate.sourcePipe = producer->pipe;
        candidate.targetPipe = consumer->pipe;
        candidate.sourcePhase = producer->id;
        candidate.targetPhase = consumer->id;
        candidate.sourceOperation = producer->operation;
        candidate.targetOperation = consumer->operation;
        const CandidateKey key = candidateKey(candidate);
        auto found = candidateByFrontier.find(key);
        if (found == candidateByFrontier.end()) {
            candidate.id = plan.candidates.size();
            candidate.channels.push_back(channel.id);
            candidate.generations.push_back(timeline->id);
            candidateByFrontier[key] = candidate.id;
            plan.candidates.push_back(std::move(candidate));
            continue;
        }
        SyncOneShotPublishCandidate& shared = plan.candidates[found->second];
        shared.channels.push_back(channel.id);
        shared.generations.push_back(timeline->id);
    }
    return plan;
}

LogicalResult mlir::pto::protocol_sync::appendOneShotPublishSelectedWorld(
    const SyncOneShotPublishPlan& plan, const ChannelAnalysisResult& channels, SyncSelectedWorld& world)
{
    llvm::DenseSet<SyncChannelId> selectedChannels;
    for (auto [index, candidate] : llvm::enumerate(plan.candidates)) {
        const bool validCandidate = candidate.id == index && candidate.sourcePhase != kInvalidSyncId &&
                                    candidate.targetPhase != kInvalidSyncId &&
                                    candidate.channels.size() == candidate.generations.size() &&
                                    !candidate.channels.empty();
        if (!validCandidate) {
            return failure();
        }
        world.completions.push_back(
            {candidate.sourcePhase,
             candidate.targetPhase,
             SyncControlRelation::MustExecute,
             {SyncIterationRelationKind::SameIteration, 0}});
        for (auto [channelId, generation] : llvm::zip_equal(candidate.channels, candidate.generations)) {
            const SyncChannel* channel = findChannel(channels, channelId);
            const bool validChannel = selectedChannels.insert(channelId).second && channel && channel->isAdmitted() &&
                                      channel->kind == SyncChannelKind::OneShot && channel->generation == generation &&
                                      channel->capacity != 0;
            if (!validChannel) {
                return failure();
            }
            world.protocols.push_back(
                {SyncSelectedProtocolKind::OneShotPublish, channelId, generation, channel->capacity});
        }
    }
    return success();
}

LogicalResult mlir::pto::protocol_sync::verifyOneShotPublishPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncOneShotPublishPlan& plan)
{
    FailureOr<SyncOneShotPublishPlan> authoritative =
        buildOneShotPublishCandidates(schedule, stages, timelines, channels);
    if (failed(authoritative)) {
        return failure();
    }
    llvm::DenseSet<unsigned> matched;
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    for (auto [index, candidate] : llvm::enumerate(plan.candidates)) {
        if (candidate.id != index) {
            return failure();
        }
        auto found = llvm::find_if(authoritative->candidates, [&](const SyncOneShotPublishCandidate& expected) {
            return candidateMatches(candidate, expected);
        });
        const bool unmatchedCandidate = found == authoritative->candidates.end() || !matched.insert(found->id).second;
        if (unmatchedCandidate) {
            return failure();
        }
        if (candidate.kind == SyncOneShotPublishKind::DirectedEvent && candidate.eventId) {
            const unsigned eventId = *candidate.eventId;
            const bool validEvent = llvm::is_contained(target.getCompilerEventIds(), eventId) &&
                                    !isReserved(schedule, candidate.sourcePipe, candidate.targetPipe, eventId);
            if (!validEvent) {
                return failure();
            }
        } else if (candidate.kind != SyncOneShotPublishKind::DirectedEvent && candidate.eventId) {
            return failure();
        }
    }
    SyncSelectedWorld world;
    return appendOneShotPublishSelectedWorld(plan, channels, world);
}

LogicalResult mlir::pto::protocol_sync::materializeOneShotPublishPlan(
    func::FuncOp clone, const IRMapping& mapping, const SyncOneShotPublishPlan& plan,
    ProtocolSyncStatistics* statistics)
{
    const OneShotPublishClock::time_point start = OneShotPublishClock::now();
    OpBuilder builder(clone.getContext());
    for (const SyncOneShotPublishCandidate& candidate : plan.candidates) {
        Operation* source = mapping.lookupOrNull(candidate.sourceOperation);
        Operation* target = mapping.lookupOrNull(candidate.targetOperation);
        if (!source || !target) {
            return failure();
        }
        if (candidate.kind == SyncOneShotPublishKind::IntrinsicOrder) {
            continue;
        }
        if (candidate.kind == SyncOneShotPublishKind::PipeBarrier) {
            builder.setInsertionPoint(target);
            auto barrier =
                builder.create<BarrierOp>(target->getLoc(), PipeAttr::get(clone.getContext(), candidate.sourcePipe));
            tagGenerated(barrier, builder, candidate.id, "publish-barrier");
            if (statistics) {
                ++statistics->materializationTransitions;
            }
            continue;
        }
        if (!candidate.eventId) {
            return failure();
        }
        builder.setInsertionPointAfter(source);
        auto set = builder.create<SetFlagOp>(
            source->getLoc(), PipeAttr::get(clone.getContext(), candidate.sourcePipe),
            PipeAttr::get(clone.getContext(), candidate.targetPipe),
            EventAttr::get(clone.getContext(), static_cast<EVENT>(*candidate.eventId)));
        tagGenerated(set, builder, candidate.id, "publish-set");
        builder.setInsertionPoint(target);
        auto wait = builder.create<WaitFlagOp>(
            target->getLoc(), PipeAttr::get(clone.getContext(), candidate.sourcePipe),
            PipeAttr::get(clone.getContext(), candidate.targetPipe),
            EventAttr::get(clone.getContext(), static_cast<EVENT>(*candidate.eventId)));
        tagGenerated(wait, builder, candidate.id, "publish-wait");
        if (statistics) {
            statistics->materializationTransitions += 2;
        }
    }
    if (statistics) {
        statistics->materializationUs += elapsedMicroseconds(start);
    }
    return success();
}

LogicalResult mlir::pto::protocol_sync::verifyOneShotPublishMaterialization(
    const StructuredSyncIR& schedule, func::FuncOp clone, const IRMapping& mapping, const SyncOneShotPublishPlan& plan,
    ProtocolSyncStatistics* statistics)
{
    const OneShotPublishClock::time_point start = OneShotPublishClock::now();
    llvm::DenseMap<SyncOneShotPublishId, ConcreteCandidate> records;
    if (failed(collectConcreteCandidates(clone, plan.candidates.size(), records))) {
        return failure();
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    for (const SyncOneShotPublishCandidate& candidate : plan.candidates) {
        const ConcreteCandidate concrete = records.lookup(candidate.id);
        Operation* source = mapping.lookupOrNull(candidate.sourceOperation);
        Operation* destination = mapping.lookupOrNull(candidate.targetOperation);
        if (!source || !destination) {
            return failure();
        }
        if (candidate.kind == SyncOneShotPublishKind::IntrinsicOrder) {
            if (concrete.barrier || concrete.set || concrete.wait) {
                return failure();
            }
            continue;
        }
        if (candidate.kind == SyncOneShotPublishKind::PipeBarrier) {
            auto barrier = dyn_cast_or_null<BarrierOp>(concrete.barrier);
            const bool validBarrier =
                barrier && !concrete.set && !concrete.wait && barrier.getPipe().getPipe() == candidate.sourcePipe &&
                isBefore(source, concrete.barrier) && onlyGeneratedBetween(concrete.barrier, destination);
            if (!validBarrier) {
                return failure();
            }
            continue;
        }
        auto set = dyn_cast_or_null<SetFlagOp>(concrete.set);
        auto wait = dyn_cast_or_null<WaitFlagOp>(concrete.wait);
        if (!candidate.eventId || !set || !wait || concrete.barrier) {
            return failure();
        }
        const unsigned eventId = static_cast<unsigned>(set.getEventId().getEvent());
        const bool validEvent =
            set.getSrcPipe().getPipe() == candidate.sourcePipe && set.getDstPipe().getPipe() == candidate.targetPipe &&
            wait.getSrcPipe().getPipe() == candidate.sourcePipe &&
            wait.getDstPipe().getPipe() == candidate.targetPipe &&
            wait.getEventId().getEvent() == set.getEventId().getEvent() && eventId == *candidate.eventId &&
            target.supportsEvent({candidate.core, candidate.sourcePipe}, {candidate.core, candidate.targetPipe}) &&
            onlyGeneratedBetween(source, concrete.set) && isBefore(concrete.set, concrete.wait) &&
            onlyGeneratedBetween(concrete.wait, destination);
        if (!validEvent) {
            return failure();
        }
    }
    if (statistics) {
        statistics->verificationUs += elapsedMicroseconds(start);
    }
    return success();
}

StringRef mlir::pto::protocol_sync::stringifySyncOneShotPublishKind(SyncOneShotPublishKind kind)
{
    switch (kind) {
        case SyncOneShotPublishKind::IntrinsicOrder:
            return "intrinsic-order";
        case SyncOneShotPublishKind::PipeBarrier:
            return "pipe-barrier";
        case SyncOneShotPublishKind::DirectedEvent:
            return "directed-event";
    }
    return "unknown";
}
