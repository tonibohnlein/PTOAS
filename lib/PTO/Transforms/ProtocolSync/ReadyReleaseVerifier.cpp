// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ReadyReleaseVerifier.cpp - Independent recurring verifier -------===//

#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

constexpr StringLiteral kGeneratedAttr = "pto.protocol_sync.generated";
constexpr StringLiteral kProtocolAttr = "pto.protocol_sync.protocol_id";
constexpr StringLiteral kProtocolKindAttr = "pto.protocol_sync.protocol_kind";
constexpr StringLiteral kReadyReleaseKind = "ready-release";
constexpr StringLiteral kRoleAttr = "pto.protocol_sync.role";
constexpr StringLiteral kLaneAttr = "pto.protocol_sync.logical_lane";
constexpr StringLiteral kLanesAttr = "pto.protocol_sync.logical_lanes";
constexpr StringLiteral kDirectCandidateAttr = "pto.protocol_sync.direct_candidate_id";

struct ExpectedReadyRelease {
    const SyncPhase* producer = nullptr;
    const SyncPhase* consumer = nullptr;
    Operation* loop = nullptr;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE producerPipe = PIPE::PIPE_UNASSIGNED;
    PIPE consumerPipe = PIPE::PIPE_UNASSIGNED;
    unsigned capacity = 0;
    Value slotSelector;
};

struct ConcreteReadyRelease {
    llvm::SmallVector<Operation*, 2> releasePrimeSets;
    Operation* releaseBodyWait = nullptr;
    Operation* readyBodySet = nullptr;
    Operation* readyBodyWait = nullptr;
    Operation* releaseBodySet = nullptr;
    llvm::SmallVector<Operation*, 2> releaseDrainWaits;
};

bool isReserved(const StructuredSyncIR& schedule, PIPE source, PIPE target, unsigned eventId)
{
    for (const SyncOpSummary& summary : schedule.getSummaries()) {
        for (const SyncEventReservation& reservation : summary.eventReservations) {
            if (reservation.source == source && reservation.target == target &&
                llvm::is_contained(reservation.eventIds, eventId)) {
                return true;
            }
        }
    }
    return false;
}

bool sameCanonicalSlotExpression(const SyncSlotExpression& first, const SyncSlotExpression& second)
{
    return first.kind == second.kind && first.depth == second.depth && first.loop == second.loop &&
           first.coefficient == second.coefficient && first.offset == second.offset && first.modulus == second.modulus;
}

bool rangesOverlap(ArrayRef<SyncByteInterval> first, ArrayRef<SyncByteInterval> second)
{
    for (const SyncByteInterval& lhs : first) {
        const bool invalidLeft = lhs.size == 0 || lhs.begin > std::numeric_limits<std::uint64_t>::max() - lhs.size;
        if (invalidLeft) {
            return true;
        }
        const std::uint64_t lhsEnd = lhs.begin + lhs.size;
        for (const SyncByteInterval& rhs : second) {
            const bool invalidRight = rhs.size == 0 || rhs.begin > std::numeric_limits<std::uint64_t>::max() - rhs.size;
            if (invalidRight) {
                return true;
            }
            const std::uint64_t rhsEnd = rhs.begin + rhs.size;
            if (lhs.begin < rhsEnd && rhs.begin < lhsEnd) {
                return true;
            }
        }
    }
    return false;
}

bool globalEffectsAreIndependentlyDisjoint(ArrayRef<const SyncAccess*> accesses)
{
    for (auto [index, first] : llvm::enumerate(accesses)) {
        for (const SyncAccess* second : accesses.drop_front(index + 1)) {
            const bool hasWrite = first->mode != SyncAccessMode::Read || second->mode != SyncAccessMode::Read;
            if (!hasWrite || first->storage.space != second->storage.space) {
                continue;
            }
            const bool exactDistinctRanges = first->family != second->family && first->storage.physical &&
                                             second->storage.physical && !first->storage.unknownRange &&
                                             !second->storage.unknownRange && !first->storage.aliasesUnknownRange &&
                                             !second->storage.aliasesUnknownRange &&
                                             !first->storage.intervals.empty() && !second->storage.intervals.empty();
            if (!exactDistinctRanges || rangesOverlap(first->storage.intervals, second->storage.intervals)) {
                return false;
            }
        }
    }
    return true;
}

// Reconstruct capacity from the allocation itself so verification does not trust
// the planner's cached storage-family metadata.
std::optional<unsigned> getAllocationCapacity(Value root)
{
    if (!root) {
        return std::nullopt;
    }
    if (auto allocation = root.getDefiningOp<AllocTileOp>()) {
        return isa<TileBufType>(allocation.getResult().getType()) ? std::optional<unsigned>(1) : std::nullopt;
    }
    if (auto allocation = root.getDefiningOp<AllocMultiTileOp>()) {
        auto type = dyn_cast<MultiTileBufType>(allocation.getResult().getType());
        if (!type) {
            return std::nullopt;
        }
        return type.getCount();
    }
    return std::nullopt;
}

FailureOr<ExpectedReadyRelease> reconstructExpectedProtocol(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages)
{
    const bool validBaseShape =
        schedule.isFrozen() && schedule.getFailures().empty() && schedule.getSemanticActions().empty();
    if (!validBaseShape) {
        return failure();
    }
    if (llvm::any_of(schedule.getSummaries(), [](const SyncOpSummary& summary) {
            return !summary.suppliedProtocols.empty() || summary.queue.has_value();
        })) {
        return failure();
    }

    const SyncRegion* loopRegion = nullptr;
    for (const SyncRegion& region : schedule.getRegions()) {
        if (region.kind == SyncRegionKind::Choice || region.kind == SyncRegionKind::Alternative ||
            region.kind == SyncRegionKind::PhysicalSection) {
            return failure();
        }
        if (region.kind == SyncRegionKind::Loop) {
            if (loopRegion || region.cardinality != SyncCardinality::ZeroOrMore) {
                return failure();
            }
            loopRegion = &region;
        }
    }
    auto loop = loopRegion ? dyn_cast_or_null<scf::ForOp>(loopRegion->operation) : scf::ForOp();
    if (!loop) {
        return failure();
    }

    const SyncPhase* producer = nullptr;
    const SyncPhase* consumer = nullptr;
    unsigned accessedLocalFamilies = 0;
    unsigned capacity = 0;
    std::optional<SyncSlotExpression> slot;
    Value producerSelector;
    llvm::DenseMap<SyncStorageFamilyId, llvm::SmallVector<const SyncAccess*, 4>> accessesByFamily;
    for (const SyncAccess& access : schedule.getAccesses()) {
        accessesByFamily[access.family].push_back(&access);
    }
    for (const SyncStorageFamily& family : schedule.getStorageFamilies()) {
        if (family.role != SyncStorageRole::LocalBuffer) {
            continue;
        }
        bool familyAccessed = false;
        auto familyAccesses = accessesByFamily.find(family.id);
        if (familyAccesses == accessesByFamily.end()) {
            continue;
        }
        bool hasLoopAccess = false;
        bool hasOutsideAccess = false;
        for (const SyncAccess* access : familyAccesses->second) {
            const SyncPhase* phase = schedule.findPhase(access->phase);
            const bool inLoop = phase && llvm::is_contained(phase->iterationDomain.loops, loopRegion->id);
            hasLoopAccess |= inLoop;
            hasOutsideAccess |= !inLoop;
        }
        if (hasLoopAccess && hasOutsideAccess) {
            return failure();
        }
        if (!hasLoopAccess) {
            continue;
        }
        const std::optional<unsigned> allocationCapacity = getAllocationCapacity(family.root);
        if (!family.physical || family.unknownRange || family.aliasesUnknownRange || family.capacityConflict ||
            !allocationCapacity || !family.slotCount || *family.slotCount != *allocationCapacity ||
            *allocationCapacity < 1 || *allocationCapacity > 2) {
            return failure();
        }
        if (capacity != 0 && capacity != *allocationCapacity) {
            return failure();
        }
        capacity = *allocationCapacity;
        for (const SyncAccess* accessPtr : familyAccesses->second) {
            const SyncAccess& access = *accessPtr;
            familyAccessed = true;
            const SyncPhase* phase = schedule.findPhase(access.phase);
            const bool exactStorage = access.storage.physical && !access.storage.unknownRange &&
                                      !access.storage.aliasesUnknownRange &&
                                      access.storage.intervals.size() == family.intervals.size() &&
                                      llvm::equal(
                                          access.storage.intervals, family.intervals,
                                          [](const SyncByteInterval& left, const SyncByteInterval& right) {
                                              return left.begin == right.begin && left.size == right.size;
                                          });
            const bool validSingleSlot = capacity == 1 && !access.slot;
            const bool validDoubleSlot =
                capacity == 2 && access.slot && access.slot->kind == SyncSlotExpressionKind::AffineModulo &&
                access.slot->selector && access.slot->induction && access.slot->loop == loopRegion->id &&
                access.slot->depth == 2 && access.slot->modulus == 2 && access.slot->coefficient == 1 &&
                access.slot->offset >= 0 && access.slot->offset < 2;
            const bool validAccess = phase && (validSingleSlot || validDoubleSlot) &&
                                     access.visibility == SyncVisibilityClass::Local && exactStorage;
            if (!validAccess) {
                return failure();
            }
            if (validDoubleSlot) {
                const bool inconsistentSlot = slot && !sameCanonicalSlotExpression(*slot, *access.slot);
                if (inconsistentSlot) {
                    return failure();
                }
                if (!slot) {
                    slot = access.slot;
                }
            }
            if (access.mode == SyncAccessMode::Write) {
                if (producer && producer != phase) {
                    return failure();
                }
                producer = phase;
                if (validDoubleSlot && !producerSelector) {
                    producerSelector = access.slot->selector;
                }
            } else if (access.mode == SyncAccessMode::Read) {
                if (consumer && consumer != phase) {
                    return failure();
                }
                consumer = phase;
            } else {
                return failure();
            }
        }
        accessedLocalFamilies += familyAccessed ? 1U : 0U;
    }
    const bool validCapacity = (capacity == 1 && !slot) || (capacity == 2 && slot && producerSelector);
    if (accessedLocalFamilies != 1 || !validCapacity || !producer || !consumer || producer == consumer ||
        !producer->operation || !consumer->operation || producer->core == SyncPhysicalCore::Unknown ||
        producer->core != consumer->core || producer->pipe == consumer->pipe ||
        producer->operation->getBlock() != loop.getBody() || consumer->operation->getBlock() != loop.getBody() ||
        !producer->operation->isBeforeInBlock(consumer->operation)) {
        return failure();
    }
    const SyncStage* producerStage = stages.findStageForPhase(producer->id);
    const SyncStage* consumerStage = stages.findStageForPhase(consumer->id);
    const bool exactStages = producerStage && consumerStage && producerStage->phases.size() == 1 &&
                             consumerStage->phases.size() == 1 && producerStage->iterationDomain.loops.size() == 1 &&
                             producerStage->iterationDomain.loops.front() == loopRegion->id &&
                             consumerStage->iterationDomain.loops == producerStage->iterationDomain.loops &&
                             producerStage->guard.empty() && consumerStage->guard.empty();
    if (!exactStages) {
        return failure();
    }
    const bool loopHasOnlyProtocolPhases = llvm::all_of(schedule.getPhases(), [&](const SyncPhase& phase) {
        const bool inLoop = llvm::is_contained(phase.iterationDomain.loops, loopRegion->id);
        return !inLoop || phase.id == producer->id || phase.id == consumer->id;
    });
    if (!loopHasOnlyProtocolPhases) {
        return failure();
    }

    llvm::DenseMap<SyncStorageFamilyId, unsigned> globalModes;
    llvm::SmallVector<const SyncAccess*, 8> globalAccesses;
    for (const SyncAccess& access : schedule.getAccesses()) {
        if (access.visibility == SyncVisibilityClass::Unknown || access.mode == SyncAccessMode::Ordered ||
            access.mode == SyncAccessMode::ReadWrite) {
            return failure();
        }
        const SyncPhase* phase = schedule.findPhase(access.phase);
        if (access.visibility == SyncVisibilityClass::Global && (!phase || phase->pipe == PIPE::PIPE_S)) {
            return failure();
        }
        if (access.visibility == SyncVisibilityClass::Global) {
            unsigned& modes = globalModes[access.family];
            modes |= access.mode == SyncAccessMode::Read ? 1U : 2U;
            if (modes == 3U) {
                return failure();
            }
            globalAccesses.push_back(&access);
        }
    }
    if (!globalEffectsAreIndependentlyDisjoint(globalAccesses)) {
        return failure();
    }

    ExpectedReadyRelease result;
    result.producer = producer;
    result.consumer = consumer;
    result.loop = loop.getOperation();
    result.core = producer->core;
    result.producerPipe = producer->pipe;
    result.consumerPipe = consumer->pipe;
    result.capacity = capacity;
    if (producerSelector) {
        result.slotSelector = producerSelector;
    }
    return result;
}

std::optional<unsigned> getLogicalLane(IntegerAttr lane, unsigned capacity)
{
    if (!lane) {
        return std::nullopt;
    }
    const APInt& value = lane.getValue();
    const bool validValue = !value.isNegative() && value.ult(capacity);
    if (!validValue) {
        return std::nullopt;
    }
    return static_cast<unsigned>(value.getLimitedValue());
}

LogicalResult collectConcreteProtocol(func::FuncOp clone, unsigned capacity, ConcreteReadyRelease& protocol)
{
    protocol.releasePrimeSets.assign(capacity, nullptr);
    protocol.releaseDrainWaits.assign(capacity, nullptr);
    llvm::DenseMap<StringRef, Operation**> bodyRoles = {
        {"release-body-wait", &protocol.releaseBodyWait},
        {"ready-body-set", &protocol.readyBodySet},
        {"ready-body-wait", &protocol.readyBodyWait},
        {"release-body-set", &protocol.releaseBodySet},
    };
    bool malformed = false;
    clone.walk([&](Operation* operation) {
        const bool fixedSync = isFixedSyncOperation(operation);
        const bool generated = operation->hasAttr(kGeneratedAttr);
        if (!fixedSync && !generated) {
            return;
        }
        auto kind = operation->getAttrOfType<StringAttr>(kProtocolKindAttr);
        const bool belongsToOtherCandidate =
            operation->hasAttr(kDirectCandidateAttr) ||
            (kind && (kind.getValue() == "one-shot" || kind.getValue() == "one-shot-publish"));
        if (belongsToOtherCandidate) {
            return;
        }
        auto role = operation->getAttrOfType<StringAttr>(kRoleAttr);
        auto protocolId = operation->getAttrOfType<IntegerAttr>(kProtocolAttr);
        const bool validBaseTag = fixedSync && operation->hasAttrOfType<UnitAttr>(kGeneratedAttr) && kind && role &&
                                  protocolId && protocolId.getInt() == 0 && kind.getValue() == kReadyReleaseKind;
        if (!validBaseTag) {
            malformed = true;
            return;
        }

        const bool prime = role.getValue() == "release-prime-set";
        const bool drain = role.getValue() == "release-drain-wait";
        if (prime || drain) {
            auto lane = operation->getAttrOfType<IntegerAttr>(kLaneAttr);
            std::optional<unsigned> laneValue = getLogicalLane(lane, capacity);
            const bool validLane = laneValue && !operation->hasAttr(kLanesAttr);
            if (!validLane) {
                malformed = true;
                return;
            }
            llvm::SmallVectorImpl<Operation*>& actions = prime ? protocol.releasePrimeSets : protocol.releaseDrainWaits;
            Operation*& action = actions[*laneValue];
            if (action) {
                malformed = true;
                return;
            }
            action = operation;
            return;
        }

        auto found = bodyRoles.find(role.getValue());
        const bool duplicateOrUnknownBodyRole = found == bodyRoles.end() || *found->second;
        if (duplicateOrUnknownBodyRole) {
            malformed = true;
            return;
        }
        auto lane = operation->getAttrOfType<IntegerAttr>(kLaneAttr);
        auto lanes = operation->getAttrOfType<DenseI32ArrayAttr>(kLanesAttr);
        const bool validSingleLane = capacity == 1 && lane && lane.getInt() == 0 && !lanes;
        const bool validDoubleLane =
            capacity == 2 && !lane && lanes && lanes.size() == 2 && lanes[0] == 0 && lanes[1] == 1;
        if (!validSingleLane && !validDoubleLane) {
            malformed = true;
            return;
        }
        *found->second = operation;
    });
    const bool completeBoundaries =
        llvm::all_of(protocol.releasePrimeSets, [](Operation* operation) { return operation != nullptr; }) &&
        llvm::all_of(protocol.releaseDrainWaits, [](Operation* operation) { return operation != nullptr; });
    const bool completeBody =
        protocol.releaseBodyWait && protocol.readyBodySet && protocol.readyBodyWait && protocol.releaseBodySet;
    return success(!malformed && completeBoundaries && completeBody);
}

bool eventIs(Operation* operation, bool set, PIPE source, PIPE target, unsigned& eventId)
{
    if (set) {
        auto event = dyn_cast_or_null<SetFlagOp>(operation);
        const bool validSet = event && event.getSrcPipe().getPipe() == source && event.getDstPipe().getPipe() == target;
        if (!validSet) {
            return false;
        }
        eventId = static_cast<unsigned>(event.getEventId().getEvent());
        return true;
    }
    auto event = dyn_cast_or_null<WaitFlagOp>(operation);
    const bool validWait = event && event.getSrcPipe().getPipe() == source && event.getDstPipe().getPipe() == target;
    if (!validWait) {
        return false;
    }
    eventId = static_cast<unsigned>(event.getEventId().getEvent());
    return true;
}

bool dynamicEventIs(Operation* operation, bool set, PIPE source, PIPE target, Value& eventId)
{
    if (set) {
        auto event = dyn_cast_or_null<SetFlagDynOp>(operation);
        const bool validSet = event && event.getSrcPipe().getPipe() == source && event.getDstPipe().getPipe() == target;
        if (!validSet) {
            return false;
        }
        eventId = event.getEventId();
        return true;
    }
    auto event = dyn_cast_or_null<WaitFlagDynOp>(operation);
    const bool validWait = event && event.getSrcPipe().getPipe() == source && event.getDstPipe().getPipe() == target;
    if (!validWait) {
        return false;
    }
    eventId = event.getEventId();
    return true;
}

std::optional<unsigned> getConstantIndex(Value value)
{
    auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
    const bool validConstant = constant && constant.value() >= 0 &&
                               static_cast<std::uint64_t>(constant.value()) <= std::numeric_limits<unsigned>::max();
    if (!validConstant) {
        return std::nullopt;
    }
    return static_cast<unsigned>(constant.value());
}

Value stripIndexCast(Value value)
{
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        return cast.getIn();
    }
    return value;
}

bool matchEventSelector(Value value, Value expectedSelector, llvm::SmallVectorImpl<unsigned>& eventIds)
{
    auto select = value.getDefiningOp<arith::SelectOp>();
    auto compare = select ? select.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp();
    const bool validSelect = select && compare && compare.getPredicate() == arith::CmpIPredicate::eq;
    if (!validSelect) {
        return false;
    }
    auto selectedLane = compare.getLhs().getDefiningOp<arith::RemUIOp>();
    std::optional<unsigned> laneOne = getConstantIndex(compare.getRhs());
    std::optional<unsigned> modulus = selectedLane ? getConstantIndex(selectedLane.getRhs()) : std::nullopt;
    std::optional<unsigned> laneOneEvent = getConstantIndex(select.getTrueValue());
    std::optional<unsigned> laneZeroEvent = getConstantIndex(select.getFalseValue());
    const bool validSelector = selectedLane && stripIndexCast(selectedLane.getLhs()) == expectedSelector && laneOne &&
                               *laneOne == 1 && modulus && *modulus == 2 && laneZeroEvent && laneOneEvent;
    if (!validSelector) {
        return false;
    }
    eventIds.assign({*laneZeroEvent, *laneOneEvent});
    return true;
}

LogicalResult verifyEvents(
    const StructuredSyncIR& schedule, const ProtocolSyncTarget& target, const ExpectedReadyRelease& expected,
    const ConcreteReadyRelease& protocol, const IRMapping& mapping)
{
    llvm::SmallVector<unsigned, 2> releaseIds;
    for (unsigned lane = 0; lane < expected.capacity; ++lane) {
        unsigned prime = 0;
        unsigned drain = 0;
        const bool validBoundary =
            eventIs(protocol.releasePrimeSets[lane], true, expected.consumerPipe, expected.producerPipe, prime) &&
            eventIs(protocol.releaseDrainWaits[lane], false, expected.consumerPipe, expected.producerPipe, drain) &&
            prime == drain;
        if (!validBoundary) {
            return failure();
        }
        releaseIds.push_back(prime);
    }

    llvm::SmallVector<unsigned, 2> readyIds;
    if (expected.capacity == 1) {
        unsigned bodyReleaseWait = 0;
        unsigned bodyReadySet = 0;
        unsigned bodyReadyWait = 0;
        unsigned bodyReleaseSet = 0;
        const bool validBody =
            eventIs(protocol.releaseBodyWait, false, expected.consumerPipe, expected.producerPipe, bodyReleaseWait) &&
            eventIs(protocol.readyBodySet, true, expected.producerPipe, expected.consumerPipe, bodyReadySet) &&
            eventIs(protocol.readyBodyWait, false, expected.producerPipe, expected.consumerPipe, bodyReadyWait) &&
            eventIs(protocol.releaseBodySet, true, expected.consumerPipe, expected.producerPipe, bodyReleaseSet) &&
            releaseIds.front() == bodyReleaseWait && releaseIds.front() == bodyReleaseSet &&
            bodyReadySet == bodyReadyWait;
        if (!validBody) {
            return failure();
        }
        readyIds.push_back(bodyReadySet);
    } else {
        Value releaseWait;
        Value releaseSet;
        Value readySet;
        Value readyWait;
        const bool validDynamicBody =
            dynamicEventIs(
                protocol.releaseBodyWait, false, expected.consumerPipe, expected.producerPipe, releaseWait) &&
            dynamicEventIs(protocol.readyBodySet, true, expected.producerPipe, expected.consumerPipe, readySet) &&
            dynamicEventIs(protocol.readyBodyWait, false, expected.producerPipe, expected.consumerPipe, readyWait) &&
            dynamicEventIs(protocol.releaseBodySet, true, expected.consumerPipe, expected.producerPipe, releaseSet) &&
            releaseWait == releaseSet && readySet == readyWait;
        Value expectedSelector = mapping.lookupOrNull(expected.slotSelector);
        llvm::SmallVector<unsigned, 2> selectedReleaseIds;
        if (!validDynamicBody || !expectedSelector ||
            !matchEventSelector(releaseWait, expectedSelector, selectedReleaseIds) ||
            !matchEventSelector(readySet, expectedSelector, readyIds) || selectedReleaseIds != releaseIds) {
            return failure();
        }
    }

    const bool completeIdSets = readyIds.size() == expected.capacity && releaseIds.size() == expected.capacity;
    const bool uniqueDoubleLaneIds =
        expected.capacity != 2 || (completeIdSets && readyIds[0] != readyIds[1] && releaseIds[0] != releaseIds[1]);
    if (!completeIdSets || !uniqueDoubleLaneIds) {
        return failure();
    }
    if (!target.supportsReadyRelease(expected.core, expected.producerPipe, expected.consumerPipe)) {
        return failure();
    }
    for (unsigned lane = 0; lane < expected.capacity; ++lane) {
        const bool legalIds = llvm::is_contained(target.getCompilerEventIds(), readyIds[lane]) &&
                              llvm::is_contained(target.getCompilerEventIds(), releaseIds[lane]) &&
                              !isReserved(schedule, expected.producerPipe, expected.consumerPipe, readyIds[lane]) &&
                              !isReserved(schedule, expected.consumerPipe, expected.producerPipe, releaseIds[lane]);
        if (!legalIds) {
            return failure();
        }
    }
    return success();
}

LogicalResult verifyPlacement(
    const ExpectedReadyRelease& expected, const ConcreteReadyRelease& protocol, const IRMapping& mapping)
{
    Operation* loop = mapping.lookupOrNull(expected.loop);
    Operation* producer = mapping.lookupOrNull(expected.producer->operation);
    Operation* consumer = mapping.lookupOrNull(expected.consumer->operation);
    if (!loop || !producer || !consumer) {
        return failure();
    }
    bool boundaryPlacement = protocol.releasePrimeSets.front()->getBlock() == loop->getBlock() &&
                             protocol.releaseDrainWaits.front()->getBlock() == loop->getBlock();
    for (unsigned lane = 1; lane < expected.capacity; ++lane) {
        boundaryPlacement &= protocol.releasePrimeSets[lane - 1]->getNextNode() == protocol.releasePrimeSets[lane] &&
                             protocol.releaseDrainWaits[lane - 1]->getNextNode() == protocol.releaseDrainWaits[lane];
    }
    boundaryPlacement &= protocol.releasePrimeSets.back()->getNextNode() == loop &&
                         loop->getNextNode() == protocol.releaseDrainWaits.front();
    const bool bodyPlacement =
        protocol.releaseBodyWait->getNextNode() == producer && producer->getNextNode() == protocol.readyBodySet &&
        protocol.readyBodySet->getNextNode() == protocol.readyBodyWait &&
        protocol.readyBodyWait->getNextNode() == consumer && consumer->getNextNode() == protocol.releaseBodySet &&
        protocol.releaseBodyWait->getBlock() == cast<scf::ForOp>(loop).getBody();
    bool hasBodyBarrier = false;
    cast<scf::ForOp>(loop).walk([&](BarrierOp) { hasBodyBarrier = true; });
    return success(boundaryPlacement && bodyPlacement && !hasBodyBarrier);
}

} // namespace

LogicalResult mlir::pto::protocol_sync::verifyReadyReleaseProtocolMaterialization(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, func::FuncOp clone,
    const IRMapping& mapping, ProtocolSyncStatistics* statistics)
{
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    FailureOr<ExpectedReadyRelease> expected = reconstructExpectedProtocol(schedule, stages);
    const bool validInputs = target.isSupported() && succeeded(expected);
    if (!validInputs) {
        return failure();
    }
    ConcreteReadyRelease protocol;
    const bool validMaterialization = succeeded(collectConcreteProtocol(clone, expected->capacity, protocol)) &&
                                      succeeded(verifyEvents(schedule, target, *expected, protocol, mapping)) &&
                                      succeeded(verifyPlacement(*expected, protocol, mapping));
    if (!validMaterialization) {
        return failure();
    }
    if (statistics) {
        statistics->verifierTransitions += 4 + 2 * expected->capacity;
    }
    return success();
}
