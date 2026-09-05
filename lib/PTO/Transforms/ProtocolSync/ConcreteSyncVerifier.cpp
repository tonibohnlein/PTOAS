// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ConcreteSyncVerifier.cpp - Verify emitted synchronization -------===//

#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/ChannelProtocolIR.h"
#include "PTO/Transforms/ProtocolSync/EventAllocation.h"
#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "PTO/Transforms/ProtocolSync/LoopFrontierRepair.h"
#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"
#include "PTO/Transforms/ProtocolSync/StorageTimeline.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using EventKey = std::tuple<std::uint8_t, std::uint8_t, unsigned>;
using CompletionKey = std::tuple<SyncPhaseId, SyncPhaseId, std::uint8_t, std::uint8_t, unsigned, SyncRegionId>;

struct ConcreteState {
    const StructuredSyncIR& schedule;
    ProtocolSyncTarget target;
    SyncSelectedWorld world;
    llvm::SmallVector<SyncEventReservation, 8> reservations;
    llvm::SmallVector<SyncEventGeneration, 16> generations;
    llvm::DenseSet<Operation*> consumedSynchronization;
    std::set<CompletionKey> completions;
};

bool sameGuard(ArrayRef<SyncControlAtom> first, ArrayRef<SyncControlAtom> second)
{
    return first.size() == second.size() && llvm::equal(first, second, [](const auto& left, const auto& right) {
               return left.choice == right.choice && left.arm == right.arm;
           });
}

SyncControlRelation controlRelation(const SyncPhase& source, const SyncPhase& target)
{
    if (!sameGuard(source.guard, target.guard)) {
        return SyncControlRelation::Unknown;
    }
    return source.guard.empty() ? SyncControlRelation::MustExecute : SyncControlRelation::SameGuard;
}

CompletionKey completionKey(const SyncSelectedCompletion& completion)
{
    return {
        completion.source,
        completion.target,
        static_cast<std::uint8_t>(completion.control),
        static_cast<std::uint8_t>(completion.iteration.kind),
        completion.iteration.distance,
        completion.iteration.carrier};
}

LogicalResult addCompletion(ConcreteState& state, const SyncSelectedCompletion& completion)
{
    if (completion.control == SyncControlRelation::Unknown) {
        return failure();
    }
    if (state.completions.insert(completionKey(completion)).second) {
        state.world.completions.push_back(completion);
    }
    return success();
}

LogicalResult addSameIterationCompletion(ConcreteState& state, const SyncPhase& source, const SyncPhase& target)
{
    const bool exactForwardPair = source.id < target.id && source.iterationDomain.loops == target.iterationDomain.loops;
    if (!exactForwardPair) {
        return failure();
    }
    return addCompletion(
        state, {source.id, target.id, controlRelation(source, target), {SyncIterationRelationKind::SameIteration, 0}});
}

const SyncRegion* enclosingPhysicalSection(const StructuredSyncIR& schedule, const SyncPhase& phase)
{
    const SyncRegion* region = schedule.findRegion(phase.region);
    while (region && region->kind != SyncRegionKind::PhysicalSection) {
        region = region->parent == kInvalidSyncId ? nullptr : schedule.findRegion(region->parent);
    }
    return region;
}

const SyncRegion* findLoopRegion(const StructuredSyncIR& schedule, Operation* loop)
{
    auto found = llvm::find_if(schedule.getRegions(), [&](const SyncRegion& region) {
        return region.kind == SyncRegionKind::Loop && region.operation == loop;
    });
    return found == schedule.getRegions().end() ? nullptr : &*found;
}

SmallVector<const SyncPhase*, 8> phasesBefore(
    const StructuredSyncIR& schedule, Operation* operation, std::optional<PIPE> pipe = std::nullopt)
{
    SmallVector<const SyncPhase*, 8> result;
    for (const SyncPhase& phase : schedule.getPhases()) {
        const bool eligible = phase.operation && phase.operation->getBlock() == operation->getBlock() &&
                              phase.operation->isBeforeInBlock(operation) && (!pipe || phase.pipe == *pipe);
        if (eligible) {
            result.push_back(&phase);
        }
    }
    return result;
}

SmallVector<const SyncPhase*, 8> phasesAfter(
    const StructuredSyncIR& schedule, Operation* operation, std::optional<PIPE> pipe = std::nullopt)
{
    SmallVector<const SyncPhase*, 8> result;
    for (const SyncPhase& phase : schedule.getPhases()) {
        const bool eligible = phase.operation && phase.operation->getBlock() == operation->getBlock() &&
                              operation->isBeforeInBlock(phase.operation) && (!pipe || phase.pipe == *pipe);
        if (eligible) {
            result.push_back(&phase);
        }
    }
    return result;
}

SmallVector<const SyncPhase*, 4> phasesBetween(const StructuredSyncIR& schedule, Operation* first, Operation* second)
{
    SmallVector<const SyncPhase*, 4> result;
    const bool invalidInterval =
        !first || !second || first->getBlock() != second->getBlock() || !first->isBeforeInBlock(second);
    if (invalidInterval) {
        return result;
    }
    for (const SyncPhase& phase : schedule.getPhases()) {
        const bool between = phase.operation && phase.operation->getBlock() == first->getBlock() &&
                             first->isBeforeInBlock(phase.operation) && phase.operation->isBeforeInBlock(second);
        if (between) {
            result.push_back(&phase);
        }
    }
    return result;
}

bool isCompilerEventId(const ConcreteState& state, unsigned eventId)
{
    return llvm::is_contained(state.target.getCompilerEventIds(), eventId);
}

bool isReserved(const ConcreteState& state, PIPE source, PIPE target, unsigned eventId)
{
    return llvm::any_of(state.reservations, [&](const SyncEventReservation& reservation) {
        return reservation.source == source && reservation.target == target &&
               llvm::is_contained(reservation.eventIds, eventId);
    });
}

EventKey getEventKey(PIPE source, PIPE target, unsigned eventId)
{
    return {static_cast<std::uint8_t>(source), static_cast<std::uint8_t>(target), eventId};
}

std::optional<unsigned> getStaticEventId(Operation* operation)
{
    if (auto set = dyn_cast<SetFlagOp>(operation)) {
        return static_cast<unsigned>(set.getEventId().getEvent());
    }
    if (auto wait = dyn_cast<WaitFlagOp>(operation)) {
        return static_cast<unsigned>(wait.getEventId().getEvent());
    }
    return std::nullopt;
}

std::pair<PIPE, PIPE> getStaticEventDirection(Operation* operation)
{
    if (auto set = dyn_cast<SetFlagOp>(operation)) {
        return {set.getSrcPipe().getPipe(), set.getDstPipe().getPipe()};
    }
    auto wait = cast<WaitFlagOp>(operation);
    return {wait.getSrcPipe().getPipe(), wait.getDstPipe().getPipe()};
}

std::pair<PIPE, PIPE> getDynamicEventDirection(Operation* operation)
{
    if (auto set = dyn_cast<SetFlagDynOp>(operation)) {
        return {set.getSrcPipe().getPipe(), set.getDstPipe().getPipe()};
    }
    auto wait = cast<WaitFlagDynOp>(operation);
    return {wait.getSrcPipe().getPipe(), wait.getDstPipe().getPipe()};
}

Value getDynamicEventValue(Operation* operation)
{
    if (auto set = dyn_cast<SetFlagDynOp>(operation)) {
        return set.getEventId();
    }
    return cast<WaitFlagDynOp>(operation).getEventId();
}

std::optional<unsigned> getConstantIndex(Value value)
{
    auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
    const bool valid = constant && constant.value() >= 0 &&
                       static_cast<std::uint64_t>(constant.value()) <= std::numeric_limits<unsigned>::max();
    return valid ? std::optional<unsigned>(static_cast<unsigned>(constant.value())) : std::nullopt;
}

Value stripIndexCast(Value value)
{
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        return cast.getIn();
    }
    return value;
}

struct EventSelector {
    Value slot;
    llvm::SmallVector<unsigned, 2> eventIds;
};

FailureOr<EventSelector> parseEventSelector(Value value)
{
    auto select = value.getDefiningOp<arith::SelectOp>();
    auto compare = select ? select.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp();
    auto remainder = compare ? compare.getLhs().getDefiningOp<arith::RemUIOp>() : arith::RemUIOp();
    const std::optional<unsigned> laneOne = compare ? getConstantIndex(compare.getRhs()) : std::nullopt;
    const std::optional<unsigned> modulus = remainder ? getConstantIndex(remainder.getRhs()) : std::nullopt;
    const std::optional<unsigned> laneOneEvent = select ? getConstantIndex(select.getTrueValue()) : std::nullopt;
    const std::optional<unsigned> laneZeroEvent = select ? getConstantIndex(select.getFalseValue()) : std::nullopt;
    const bool valid = select && compare && compare.getPredicate() == arith::CmpIPredicate::eq && remainder &&
                       laneOne && *laneOne == 1 && modulus && *modulus == 2 && laneZeroEvent && laneOneEvent;
    if (!valid) {
        return failure();
    }
    return EventSelector{stripIndexCast(remainder.getLhs()), {*laneZeroEvent, *laneOneEvent}};
}

bool accessUsesSelector(const StructuredSyncIR& schedule, const SyncPhase& phase, Value selector)
{
    return llvm::any_of(phase.accesses, [&](SyncAccessId id) {
        const SyncAccess* access = schedule.findAccess(id);
        return access && access->slot && access->slot->kind == SyncSlotExpressionKind::AffineModulo &&
               access->slot->selector == selector && access->slot->depth == 2 && access->slot->modulus == 2;
    });
}

SmallVector<Operation*, 4> bodyEventOperations(scf::ForOp loop)
{
    SmallVector<Operation*, 4> operations;
    for (Operation& operation : loop.getBody()->without_terminator()) {
        if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp>(&operation)) {
            operations.push_back(&operation);
        }
    }
    return operations;
}

SmallVector<Operation*, 2> adjacentStaticEvents(Operation* anchor, bool before, bool set)
{
    SmallVector<Operation*, 2> result;
    Operation* operation = before ? anchor->getPrevNode() : anchor->getNextNode();
    bool adjacent = operation && (set ? isa<SetFlagOp>(operation) : isa<WaitFlagOp>(operation));
    while (adjacent) {
        result.push_back(operation);
        operation = before ? operation->getPrevNode() : operation->getNextNode();
        adjacent = operation && (set ? isa<SetFlagOp>(operation) : isa<WaitFlagOp>(operation));
    }
    if (before) {
        llvm::reverse(result);
    }
    return result;
}

LogicalResult verifyEventIds(
    const ConcreteState& state, PIPE source, PIPE target, ArrayRef<unsigned> eventIds, bool requireDistinct)
{
    llvm::DenseSet<unsigned> seen;
    for (unsigned eventId : eventIds) {
        const bool valid = isCompilerEventId(state, eventId) && !isReserved(state, source, target, eventId) &&
                           (!requireDistinct || seen.insert(eventId).second);
        if (!valid) {
            return failure();
        }
    }
    return success();
}

LogicalResult verifyReadyReleaseBoundary(
    ConcreteState& state, scf::ForOp loop, PIPE releaseSource, PIPE releaseTarget, ArrayRef<unsigned> releaseIds,
    SmallVectorImpl<Operation*>& primes, SmallVectorImpl<Operation*>& drains)
{
    const SmallVector<Operation*, 2> adjacentSets = adjacentStaticEvents(loop, true, true);
    const SmallVector<Operation*, 2> adjacentWaits = adjacentStaticEvents(loop, false, false);
    llvm::DenseSet<unsigned> expected(releaseIds.begin(), releaseIds.end());
    llvm::DenseSet<unsigned> primed;
    llvm::DenseSet<unsigned> drained;
    for (Operation* operation : adjacentSets) {
        const auto [source, target] = getStaticEventDirection(operation);
        const std::optional<unsigned> eventId = getStaticEventId(operation);
        if (source != releaseSource || target != releaseTarget || !eventId || !expected.contains(*eventId)) {
            continue;
        }
        if (!primed.insert(*eventId).second) {
            return failure();
        }
        primes.push_back(operation);
        state.consumedSynchronization.insert(operation);
    }
    for (Operation* operation : adjacentWaits) {
        const auto [source, target] = getStaticEventDirection(operation);
        const std::optional<unsigned> eventId = getStaticEventId(operation);
        if (source != releaseSource || target != releaseTarget || !eventId || !expected.contains(*eventId)) {
            continue;
        }
        if (!drained.insert(*eventId).second) {
            return failure();
        }
        drains.push_back(operation);
        state.consumedSynchronization.insert(operation);
    }
    return success(primed.size() == expected.size() && drained.size() == expected.size());
}

LogicalResult addReadyReleaseGenerations(
    ConcreteState& state, const SyncRegion& loop, const SyncPhase& producer, const SyncPhase& consumer,
    ArrayRef<unsigned> readyIds, ArrayRef<unsigned> releaseIds)
{
    for (unsigned eventId : readyIds) {
        state.generations.push_back(
            {static_cast<SyncEventGenerationId>(state.generations.size()), SyncEventGenerationKind::ReadyReleaseReady,
             producer.core, producer.pipe, consumer.pipe, producer.operation, consumer.operation, producer.guard,
             loop.id, true, eventId});
    }
    for (unsigned eventId : releaseIds) {
        state.generations.push_back(
            {static_cast<SyncEventGenerationId>(state.generations.size()), SyncEventGenerationKind::ReadyReleaseRelease,
             producer.core, consumer.pipe, producer.pipe, consumer.operation, producer.operation, producer.guard,
             loop.id, true, eventId});
    }
    return success();
}

bool readyReleaseTokenWitnessIsBalanced(unsigned capacity, unsigned trips)
{
    SmallVector<bool, 2> release(capacity, true);
    SmallVector<bool, 2> ready(capacity, false);
    for (unsigned iteration = 0; iteration < trips; ++iteration) {
        const unsigned lane = iteration % capacity;
        if (!release[lane] || ready[lane]) {
            return false;
        }
        release[lane] = false;
        ready[lane] = true;
        if (release[lane] || !ready[lane]) {
            return false;
        }
        ready[lane] = false;
        release[lane] = true;
    }
    for (unsigned lane = 0; lane < capacity; ++lane) {
        if (!release[lane] || ready[lane]) {
            return false;
        }
        release[lane] = false;
    }
    return llvm::none_of(release, [](bool token) { return token; }) &&
           llvm::none_of(ready, [](bool token) { return token; });
}

bool verifyReadyReleaseTokenSemantics(unsigned capacity)
{
    if (capacity == 0 || capacity > 2) {
        return false;
    }
    const unsigned horizon = 2 * capacity + 1;
    for (unsigned trips = 0; trips <= horizon; ++trips) {
        if (!readyReleaseTokenWitnessIsBalanced(capacity, trips)) {
            return false;
        }
    }
    return true;
}

LogicalResult reconstructReadyRelease(ConcreteState& state, scf::ForOp loop)
{
    SmallVector<Operation*, 4> events = bodyEventOperations(loop);
    if (events.empty()) {
        return success();
    }
    const bool invalidBodyShape =
        events.size() != 4 || !isa<WaitFlagOp, WaitFlagDynOp>(events[0]) || !isa<SetFlagOp, SetFlagDynOp>(events[1]) ||
        !isa<WaitFlagOp, WaitFlagDynOp>(events[2]) || !isa<SetFlagOp, SetFlagDynOp>(events[3]);
    if (invalidBodyShape) {
        return failure();
    }
    SmallVector<const SyncPhase*, 4> producerPhases = phasesBetween(state.schedule, events[0], events[1]);
    SmallVector<const SyncPhase*, 4> consumerPhases = phasesBetween(state.schedule, events[2], events[3]);
    const bool exactPhases = producerPhases.size() == 1 && consumerPhases.size() == 1 &&
                             phasesBetween(state.schedule, events[1], events[2]).empty();
    if (!exactPhases) {
        return failure();
    }
    const SyncPhase& producer = *producerPhases.front();
    const SyncPhase& consumer = *consumerPhases.front();
    const SyncRegion* loopRegion = findLoopRegion(state.schedule, loop);
    const bool exactLoop =
        loopRegion && producer.core != SyncPhysicalCore::Unknown && producer.core == consumer.core &&
        producer.pipe != consumer.pipe && producer.iterationDomain.loops == consumer.iterationDomain.loops &&
        !producer.iterationDomain.loops.empty() && producer.iterationDomain.loops.back() == loopRegion->id;
    if (!exactLoop) {
        return failure();
    }

    const bool staticBody =
        llvm::all_of(events, [](Operation* operation) { return isa<SetFlagOp, WaitFlagOp>(operation); });
    const bool dynamicBody =
        llvm::all_of(events, [](Operation* operation) { return isa<SetFlagDynOp, WaitFlagDynOp>(operation); });
    if (!staticBody && !dynamicBody) {
        return failure();
    }

    const auto [releaseSource, releaseTarget] =
        staticBody ? getStaticEventDirection(events[0]) : getDynamicEventDirection(events[0]);
    const auto [readySource, readyTarget] =
        staticBody ? getStaticEventDirection(events[1]) : getDynamicEventDirection(events[1]);
    const auto [readyWaitSource, readyWaitTarget] =
        staticBody ? getStaticEventDirection(events[2]) : getDynamicEventDirection(events[2]);
    const auto [releaseSetSource, releaseSetTarget] =
        staticBody ? getStaticEventDirection(events[3]) : getDynamicEventDirection(events[3]);
    const bool validDirections = releaseSource == consumer.pipe && releaseTarget == producer.pipe &&
                                 readySource == producer.pipe && readyTarget == consumer.pipe &&
                                 readyWaitSource == readySource && readyWaitTarget == readyTarget &&
                                 releaseSetSource == releaseSource && releaseSetTarget == releaseTarget;
    if (!validDirections) {
        return failure();
    }

    SmallVector<unsigned, 2> readyIds;
    SmallVector<unsigned, 2> releaseIds;
    unsigned capacity = 0;
    if (staticBody) {
        const std::optional<unsigned> releaseWait = getStaticEventId(events[0]);
        const std::optional<unsigned> readySet = getStaticEventId(events[1]);
        const std::optional<unsigned> readyWait = getStaticEventId(events[2]);
        const std::optional<unsigned> releaseSet = getStaticEventId(events[3]);
        const bool valid =
            releaseWait && readySet && readyWait && releaseSet && readySet == readyWait && releaseWait == releaseSet;
        if (!valid) {
            return failure();
        }
        readyIds.push_back(*readySet);
        releaseIds.push_back(*releaseWait);
        capacity = 1;
    } else {
        const Value releaseWait = getDynamicEventValue(events[0]);
        const Value readySet = getDynamicEventValue(events[1]);
        const Value readyWait = getDynamicEventValue(events[2]);
        const Value releaseSet = getDynamicEventValue(events[3]);
        FailureOr<EventSelector> readySelector = parseEventSelector(readySet);
        FailureOr<EventSelector> releaseSelector = parseEventSelector(releaseWait);
        const bool valid = readySet == readyWait && releaseWait == releaseSet && succeeded(readySelector) &&
                           succeeded(releaseSelector) && readySelector->slot == releaseSelector->slot &&
                           accessUsesSelector(state.schedule, producer, readySelector->slot);
        if (!valid) {
            return failure();
        }
        readyIds = readySelector->eventIds;
        releaseIds = releaseSelector->eventIds;
        capacity = 2;
    }

    const bool legalProtocol = state.target.supportsReadyRelease(producer.core, producer.pipe, consumer.pipe) &&
                               succeeded(verifyEventIds(state, producer.pipe, consumer.pipe, readyIds, true)) &&
                               succeeded(verifyEventIds(state, consumer.pipe, producer.pipe, releaseIds, true));
    if (!legalProtocol) {
        return failure();
    }
    SmallVector<Operation*, 2> primes;
    SmallVector<Operation*, 2> drains;
    if (failed(verifyReadyReleaseBoundary(state, loop, consumer.pipe, producer.pipe, releaseIds, primes, drains))) {
        return failure();
    }
    if (!verifyReadyReleaseTokenSemantics(capacity)) {
        return failure();
    }
    for (Operation* event : events) {
        state.consumedSynchronization.insert(event);
    }
    const bool invalidEffects =
        failed(addSameIterationCompletion(state, producer, consumer)) ||
        failed(addCompletion(
            state, {consumer.id,
                    producer.id,
                    controlRelation(consumer, producer),
                    {SyncIterationRelationKind::LoopCarried, capacity, loopRegion->id}})) ||
        failed(addReadyReleaseGenerations(state, *loopRegion, producer, consumer, readyIds, releaseIds));
    if (invalidEffects) {
        return failure();
    }
    state.world.exitCompletedPhases.push_back(producer.id);
    state.world.exitCompletedPhases.push_back(consumer.id);
    return success();
}

LogicalResult reconstructReadyReleaseProtocols(ConcreteState& state, func::FuncOp function)
{
    unsigned protocols = 0;
    WalkResult result = function.walk([&](scf::ForOp loop) {
        if (bodyEventOperations(loop).empty()) {
            return WalkResult::advance();
        }
        ++protocols;
        return failed(reconstructReadyRelease(state, loop)) ? WalkResult::interrupt() : WalkResult::advance();
    });
    return success(!result.wasInterrupted() && protocols <= 1);
}

LogicalResult addConcreteEventPair(ConcreteState& state, Operation* set, Operation* wait)
{
    const auto [sourcePipe, targetPipe] = getStaticEventDirection(set);
    const auto waitDirection = getStaticEventDirection(wait);
    const std::optional<unsigned> eventId = getStaticEventId(set);
    const bool exactPair = eventId && waitDirection == std::pair<PIPE, PIPE>{sourcePipe, targetPipe} &&
                           getStaticEventId(wait) == eventId && set->getBlock() == wait->getBlock() &&
                           set->isBeforeInBlock(wait) && isCompilerEventId(state, *eventId) &&
                           !isReserved(state, sourcePipe, targetPipe, *eventId);
    if (!exactPair) {
        return failure();
    }
    SmallVector<const SyncPhase*, 8> sources = phasesBefore(state.schedule, set, sourcePipe);
    SmallVector<const SyncPhase*, 8> targets = phasesAfter(state.schedule, wait, targetPipe);
    bool added = false;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    ArrayRef<SyncControlAtom> guard;
    for (const SyncPhase* source : sources) {
        for (const SyncPhase* target : targets) {
            const bool compatible = source->core != SyncPhysicalCore::Unknown && source->core == target->core &&
                                    source->iterationDomain.loops.empty() && target->iterationDomain.loops.empty() &&
                                    state.target.supportsEvent({source->core, sourcePipe}, {target->core, targetPipe});
            if (!compatible) {
                continue;
            }
            if (failed(addSameIterationCompletion(state, *source, *target))) {
                return failure();
            }
            const bool inconsistentGeneration = added && (source->core != core || !sameGuard(source->guard, guard));
            if (inconsistentGeneration) {
                return failure();
            }
            core = source->core;
            guard = source->guard;
            added = true;
        }
    }
    if (!added) {
        return failure();
    }
    state.generations.push_back(
        {static_cast<SyncEventGenerationId>(state.generations.size()), SyncEventGenerationKind::DirectRepair, core,
         sourcePipe, targetPipe, set, wait, SmallVector<SyncControlAtom, 2>(guard.begin(), guard.end()), kInvalidSyncId,
         false, *eventId});
    state.consumedSynchronization.insert(set);
    state.consumedSynchronization.insert(wait);
    return success();
}

LogicalResult reconstructStaticEvents(ConcreteState& state, func::FuncOp function)
{
    std::map<std::pair<Block*, EventKey>, Operation*> open;
    bool malformed = false;
    function.walk([&](Operation* operation) {
        const bool skipOperation =
            malformed || state.consumedSynchronization.contains(operation) || !isa<SetFlagOp, WaitFlagOp>(operation);
        if (skipOperation) {
            return;
        }
        const auto [source, target] = getStaticEventDirection(operation);
        const std::optional<unsigned> eventId = getStaticEventId(operation);
        if (!eventId) {
            malformed = true;
            return;
        }
        const auto key = std::make_pair(operation->getBlock(), getEventKey(source, target, *eventId));
        if (isa<SetFlagOp>(operation)) {
            const bool duplicateSet = open.find(key) != open.end();
            if (duplicateSet) {
                malformed = true;
                return;
            }
            open[key] = operation;
            return;
        }
        auto found = open.find(key);
        const bool invalidWait = found == open.end() || failed(addConcreteEventPair(state, found->second, operation));
        if (invalidWait) {
            malformed = true;
            return;
        }
        open.erase(found);
    });
    return success(!malformed && open.empty());
}

LogicalResult addBarrierCompletions(ConcreteState& state, BarrierOp barrier)
{
    const PIPE pipe = barrier.getPipe().getPipe();
    SmallVector<const SyncPhase*, 8> sources = phasesBefore(state.schedule, barrier, pipe);
    // A named barrier completes its own pipeline prefix. It does not transfer
    // that completion to another pipeline; that requires a directed handoff.
    SmallVector<const SyncPhase*, 8> targets = phasesAfter(state.schedule, barrier, pipe);
    bool added = false;
    for (const SyncPhase* source : sources) {
        if (!state.target.supportsPipeBarrier({source->core, pipe})) {
            return failure();
        }
        if (!source->iterationDomain.loops.empty()) {
            continue;
        }
        // A named barrier remains valid without a later access on this pipe,
        // but does not establish completion before function or section exit.
        added = true;
        for (const SyncPhase* target : targets) {
            const bool compatible = source->core == target->core && source->iterationDomain.loops.empty() &&
                                    target->iterationDomain.loops.empty();
            if (!compatible) {
                continue;
            }
            if (failed(addSameIterationCompletion(state, *source, *target))) {
                return failure();
            }
        }
    }
    return success(added);
}

LogicalResult addTailCompletions(ConcreteState& state, BarrierOp barrier)
{
    Operation* operation = barrier;
    Operation* sectionOperation = nullptr;
    for (Operation* parent = operation->getParentOp(); parent; parent = parent->getParentOp()) {
        if (isa<SectionVectorOp, SectionCubeOp>(parent)) {
            sectionOperation = parent;
            break;
        }
        if (isa<func::FuncOp>(parent)) {
            break;
        }
    }
    if (sectionOperation) {
        const bool atSectionExit = operation->getBlock() && &operation->getBlock()->back() == operation;
        if (!atSectionExit) {
            return failure();
        }
    } else if (!isa_and_nonnull<func::ReturnOp>(operation->getNextNode())) {
        return failure();
    }

    bool added = false;
    llvm::DenseSet<SyncPhaseId> existing;
    existing.insert(state.world.exitCompletedPhases.begin(), state.world.exitCompletedPhases.end());
    for (const SyncPhase& phase : state.schedule.getPhases()) {
        const SyncRegion* section = enclosingPhysicalSection(state.schedule, phase);
        const bool sameDomain = sectionOperation ? section && section->operation == sectionOperation : !section;
        if (sameDomain) {
            if (existing.insert(phase.id).second) {
                state.world.exitCompletedPhases.push_back(phase.id);
            }
            added = true;
        }
    }
    return success(added);
}

LogicalResult reconstructBarriers(ConcreteState& state, func::FuncOp function)
{
    bool malformed = false;
    function.walk([&](BarrierOp barrier) {
        if (malformed) {
            return;
        }
        const PIPE pipe = barrier.getPipe().getPipe();
        const LogicalResult result =
            pipe == PIPE::PIPE_ALL ? addTailCompletions(state, barrier) : addBarrierCompletions(state, barrier);
        if (failed(result)) {
            malformed = true;
            return;
        }
        state.consumedSynchronization.insert(barrier);
    });
    return failure(malformed);
}

LogicalResult verifyAllSynchronizationConsumed(const ConcreteState& state, func::FuncOp function)
{
    bool unmodeled = false;
    function.walk([&](Operation* operation) {
        const bool unmodeledSynchronization =
            isFixedSyncOperation(operation) && !state.consumedSynchronization.contains(operation);
        if (unmodeledSynchronization) {
            unmodeled = true;
        }
    });
    return failure(unmodeled);
}

LogicalResult rejectAtStage(StringRef stage, StringRef* firstFailedStage)
{
    if (firstFailedStage) {
        *firstFailedStage = stage;
    }
    return failure();
}

LogicalResult reconstructConcreteWorld(ConcreteState& state, func::FuncOp function, StringRef* firstFailedStage)
{
    for (const SyncOpSummary& summary : state.schedule.getSummaries()) {
        state.reservations.append(summary.eventReservations.begin(), summary.eventReservations.end());
        if (summary.queue) {
            return rejectAtStage("queue-supply", firstFailedStage);
        }
    }
    if (failed(reconstructReadyReleaseProtocols(state, function))) {
        return rejectAtStage("recurring-events", firstFailedStage);
    }
    if (failed(reconstructStaticEvents(state, function))) {
        return rejectAtStage("static-events", firstFailedStage);
    }
    if (failed(reconstructBarriers(state, function))) {
        return rejectAtStage("barriers", firstFailedStage);
    }
    if (failed(verifyAllSynchronizationConsumed(state, function))) {
        return rejectAtStage("unmodeled-fixed-sync", firstFailedStage);
    }
    if (failed(verifySyncEventGenerationAssignment(
            state.target, state.reservations, state.generations, /*allowUnallocated=*/false))) {
        return rejectAtStage("event-generation-assignment", firstFailedStage);
    }
    return success();
}

} // namespace

FailureOr<SyncSelectedWorld> mlir::pto::protocol_sync::reconstructFixedSyncSupply(
    const StructuredSyncIR& schedule, SmallVectorImpl<SyncEventReservation>* occupiedEvents)
{
    const bool hasFixed = llvm::any_of(schedule.getSummaries(), [](const SyncOpSummary& summary) {
        return summary.provider == SyncSummaryProvider::FixedSynchronization;
    });
    if (!hasFixed) {
        return SyncSelectedWorld{};
    }
    ConcreteState state{schedule, ProtocolSyncTarget::resolve(schedule.getFunction())};
    const bool validSupply =
        state.target.isSupported() && succeeded(reconstructConcreteWorld(state, schedule.getFunction(), nullptr));
    if (!validSupply) {
        return failure();
    }
    if (occupiedEvents) {
        for (const SyncEventGeneration& generation : state.generations) {
            if (!generation.eventId) {
                return failure();
            }
            occupiedEvents->push_back({generation.sourcePipe, generation.targetPipe, {*generation.eventId}});
        }
    }
    return std::move(state.world);
}

LogicalResult mlir::pto::protocol_sync::verifyFreshConcreteSyncSemantics(
    func::FuncOp function, ProtocolSyncStatistics* statistics)
{
    LegacySyncIRAdapter adapter;
    LegacySyncSnapshot snapshot;
    if (failed(adapter.buildSnapshot(function, snapshot))) {
        return failure();
    }
    SyncSemanticContext context = adapter.buildSemanticContext(snapshot);
    return verifyConcreteSyncSemantics(context, function, statistics);
}

LogicalResult mlir::pto::protocol_sync::verifyConcreteSyncSemantics(
    const SyncSemanticContext& context, func::FuncOp function, ProtocolSyncStatistics* statistics,
    StringRef* firstFailedStage)
{
    if (firstFailedStage) {
        *firstFailedStage = "none";
    }
    if (!function || failed(mlir::verify(function))) {
        return rejectAtStage("ir-verification", firstFailedStage);
    }
    StructuredSyncIR schedule(function);
    StructuredSyncIRBuilder builder(context);
    const bool invalidSchedule = failed(builder.build(function, schedule)) || !schedule.getFailures().empty();
    if (invalidSchedule) {
        return rejectAtStage("schedule", firstFailedStage);
    }
    FailureOr<PipelineStageAnalysisResult> stages = analyzePipelineStages(schedule);
    if (failed(stages)) {
        return rejectAtStage("pipeline-stages", firstFailedStage);
    }
    StorageTimelineAnalysisResult timelines = analyzeStorageTimelines(schedule, *stages);
    ChannelAnalysisResult channels = analyzeChannels(schedule, *stages, timelines);
    ConcreteState state{schedule, ProtocolSyncTarget::resolve(function)};
    if (!state.target.isSupported()) {
        return rejectAtStage("target", firstFailedStage);
    }
    const bool loopFrontier = succeeded(verifyConcreteLoopFrontierRepair(schedule, true));
    if (loopFrontier) {
        auto world = buildLoopFrontierWorld(schedule);
        if (failed(world)) {
            return rejectAtStage("loop-frontier-world", firstFailedStage);
        }
        state.world = std::move(*world);
    } else {
        if (failed(reconstructConcreteWorld(state, function, firstFailedStage))) {
            return failure();
        }
    }
    const SyncInterpretationOptions options{/*fixedSynchronizationIsModeled=*/true};
    FailureOr<SyncInterpretationResult> result =
        interpretSelectedWorld(schedule, *stages, timelines, channels, state.world, nullptr, options);
    if (statistics) {
        const std::uint64_t semanticTransitions = succeeded(result) ? result->transitions : 0;
        const std::uint64_t concreteTransitions = state.consumedSynchronization.size();
        const std::uint64_t combined =
            semanticTransitions > std::numeric_limits<std::uint64_t>::max() - concreteTransitions ?
                std::numeric_limits<std::uint64_t>::max() :
                semanticTransitions + concreteTransitions;
        const std::uint64_t available = std::numeric_limits<std::uint64_t>::max() - statistics->verifierTransitions;
        statistics->verifierTransitions += std::min(available, combined);
    }
    if (failed(result)) {
        return rejectAtStage("interpretation", firstFailedStage);
    }
    if (!result->isComplete()) {
        return rejectAtStage("residual-obligations", firstFailedStage);
    }
    // The loop checker establishes total dynamic phase order, including all
    // boundary paths. The straight-line scoreboard cannot interpret recurrence.
    if (!loopFrontier && failed(verifyLocalMemoryCoverage(schedule, state.world))) {
        return rejectAtStage("local-memory-coverage", firstFailedStage);
    }
    if (!loopFrontier && failed(verifyConcreteLocalScoreboard(schedule))) {
        return rejectAtStage("local-concrete-scoreboard", firstFailedStage);
    }
    return success();
}
