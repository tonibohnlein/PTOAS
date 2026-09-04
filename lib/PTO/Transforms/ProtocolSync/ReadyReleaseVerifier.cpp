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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

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

struct ExpectedReadyRelease {
    const SyncPhase* producer = nullptr;
    const SyncPhase* consumer = nullptr;
    Operation* loop = nullptr;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE producerPipe = PIPE::PIPE_UNASSIGNED;
    PIPE consumerPipe = PIPE::PIPE_UNASSIGNED;
};

struct ConcreteReadyRelease {
    Operation* releasePrimeSet = nullptr;
    Operation* releaseBodyWait = nullptr;
    Operation* readyBodySet = nullptr;
    Operation* readyBodyWait = nullptr;
    Operation* releaseBodySet = nullptr;
    Operation* releaseDrainWait = nullptr;
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

FailureOr<ExpectedReadyRelease> reconstructExpectedProtocol(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages)
{
    const bool validBaseShape = schedule.isFrozen() && schedule.getFailures().empty() &&
                                schedule.getSemanticActions().empty() && schedule.getPhases().size() == 2 &&
                                stages.getStages().size() == 2;
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
    llvm::DenseMap<SyncStorageFamilyId, llvm::SmallVector<const SyncAccess*, 4>> accessesByFamily;
    for (const SyncAccess& access : schedule.getAccesses()) {
        accessesByFamily[access.family].push_back(&access);
    }
    for (const SyncStorageFamily& family : schedule.getStorageFamilies()) {
        if (family.role != SyncStorageRole::LocalBuffer) {
            continue;
        }
        if (!family.physical || family.unknownRange || family.aliasesUnknownRange || family.capacityConflict ||
            !family.slotCount || *family.slotCount != 1) {
            return failure();
        }
        bool familyAccessed = false;
        auto familyAccesses = accessesByFamily.find(family.id);
        if (familyAccesses == accessesByFamily.end()) {
            continue;
        }
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
            if (!phase || access.slot || access.visibility != SyncVisibilityClass::Local || !exactStorage) {
                return failure();
            }
            if (access.mode == SyncAccessMode::Write) {
                if (producer && producer != phase) {
                    return failure();
                }
                producer = phase;
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
    if (accessedLocalFamilies != 1 || !producer || !consumer || producer == consumer || !producer->operation ||
        !consumer->operation || producer->core == SyncPhysicalCore::Unknown || producer->core != consumer->core ||
        producer->pipe == consumer->pipe || producer->operation->getBlock() != loop.getBody() ||
        consumer->operation->getBlock() != loop.getBody() ||
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

    llvm::DenseMap<SyncStorageFamilyId, unsigned> globalModes;
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
        }
    }

    ExpectedReadyRelease result;
    result.producer = producer;
    result.consumer = consumer;
    result.loop = loop.getOperation();
    result.core = producer->core;
    result.producerPipe = producer->pipe;
    result.consumerPipe = consumer->pipe;
    return result;
}

LogicalResult collectConcreteProtocol(func::FuncOp clone, ConcreteReadyRelease& protocol)
{
    llvm::DenseMap<StringRef, Operation**> roles = {
        {"release-prime-set", &protocol.releasePrimeSet}, {"release-body-wait", &protocol.releaseBodyWait},
        {"ready-body-set", &protocol.readyBodySet},       {"ready-body-wait", &protocol.readyBodyWait},
        {"release-body-set", &protocol.releaseBodySet},   {"release-drain-wait", &protocol.releaseDrainWait},
    };
    bool malformed = false;
    clone.walk([&](Operation* operation) {
        const bool fixedSync = isFixedSyncOperation(operation);
        const bool generated = operation->hasAttr(kGeneratedAttr);
        if (!fixedSync && !generated) {
            return;
        }
        auto kind = operation->getAttrOfType<StringAttr>(kProtocolKindAttr);
        auto role = operation->getAttrOfType<StringAttr>(kRoleAttr);
        auto protocolId = operation->getAttrOfType<IntegerAttr>(kProtocolAttr);
        auto lane = operation->getAttrOfType<IntegerAttr>(kLaneAttr);
        const bool validTag = fixedSync && operation->hasAttrOfType<UnitAttr>(kGeneratedAttr) && kind && role &&
                              protocolId && protocolId.getInt() == 0 && lane && lane.getInt() == 0 &&
                              kind.getValue() == kReadyReleaseKind;
        if (!validTag) {
            malformed = true;
            return;
        }
        auto found = roles.find(role.getValue());
        const bool duplicateOrUnknownRole = found == roles.end() || (found != roles.end() && *found->second);
        if (duplicateOrUnknownRole) {
            malformed = true;
            return;
        }
        *found->second = operation;
    });
    const bool complete = protocol.releasePrimeSet && protocol.releaseBodyWait && protocol.readyBodySet &&
                          protocol.readyBodyWait && protocol.releaseBodySet && protocol.releaseDrainWait;
    return success(!malformed && complete);
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

LogicalResult verifyEvents(
    const StructuredSyncIR& schedule, const ProtocolSyncTarget& target, const ExpectedReadyRelease& expected,
    const ConcreteReadyRelease& protocol)
{
    unsigned primeRelease = 0;
    unsigned bodyReleaseWait = 0;
    unsigned bodyReadySet = 0;
    unsigned bodyReadyWait = 0;
    unsigned bodyReleaseSet = 0;
    unsigned drainRelease = 0;
    const bool kindsAndDirections =
        eventIs(protocol.releasePrimeSet, true, expected.consumerPipe, expected.producerPipe, primeRelease) &&
        eventIs(protocol.releaseBodyWait, false, expected.consumerPipe, expected.producerPipe, bodyReleaseWait) &&
        eventIs(protocol.readyBodySet, true, expected.producerPipe, expected.consumerPipe, bodyReadySet) &&
        eventIs(protocol.readyBodyWait, false, expected.producerPipe, expected.consumerPipe, bodyReadyWait) &&
        eventIs(protocol.releaseBodySet, true, expected.consumerPipe, expected.producerPipe, bodyReleaseSet) &&
        eventIs(protocol.releaseDrainWait, false, expected.consumerPipe, expected.producerPipe, drainRelease);
    if (!kindsAndDirections || primeRelease != bodyReleaseWait || primeRelease != bodyReleaseSet ||
        primeRelease != drainRelease || bodyReadySet != bodyReadyWait) {
        return failure();
    }
    const bool legalDirections =
        target.supportsEvent({expected.core, expected.producerPipe}, {expected.core, expected.consumerPipe}) &&
        target.supportsEvent({expected.core, expected.consumerPipe}, {expected.core, expected.producerPipe});
    const bool legalIds = llvm::is_contained(target.getCompilerEventIds(), bodyReadySet) &&
                          llvm::is_contained(target.getCompilerEventIds(), primeRelease) &&
                          !isReserved(schedule, expected.producerPipe, expected.consumerPipe, bodyReadySet) &&
                          !isReserved(schedule, expected.consumerPipe, expected.producerPipe, primeRelease);
    return success(legalDirections && legalIds);
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
    const bool boundaryPlacement = protocol.releasePrimeSet->getBlock() == loop->getBlock() &&
                                   protocol.releaseDrainWait->getBlock() == loop->getBlock() &&
                                   protocol.releasePrimeSet->getNextNode() == loop &&
                                   loop->getNextNode() == protocol.releaseDrainWait;
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
    const bool validMaterialization = succeeded(collectConcreteProtocol(clone, protocol)) &&
                                      succeeded(verifyEvents(schedule, target, *expected, protocol)) &&
                                      succeeded(verifyPlacement(*expected, protocol, mapping));
    if (!validMaterialization) {
        return failure();
    }
    if (statistics) {
        statistics->verifierTransitions += 6;
    }
    return success();
}
