// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
//===- OneShotVerifier.cpp - Independent emitted-protocol verifier -------===//
#include "PTO/Transforms/ProtocolSync/OneShotProtocol.h"

#include "PTO/IR/PTO.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

constexpr StringLiteral kGeneratedAttr = "pto.protocol_sync.generated";
constexpr StringLiteral kProtocolAttr = "pto.protocol_sync.protocol_id";
constexpr StringLiteral kRoleAttr = "pto.protocol_sync.role";
constexpr StringLiteral kTailRole = "tail-drain";
constexpr StringLiteral kSectionLocalTailAttr = "pto.auto_sync_tail_section_local";

bool isBefore(Operation* first, Operation* second)
{
    return first && second && first->getBlock() == second->getBlock() && first->isBeforeInBlock(second);
}

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

const SyncRegion* findEnclosingPhysicalSection(const StructuredSyncIR& schedule, const SyncPhase& phase)
{
    const SyncRegion* region = schedule.findRegion(phase.region);
    while (region) {
        if (region->kind == SyncRegionKind::PhysicalSection) {
            return region;
        }
        if (region->parent == kInvalidSyncId) {
            return nullptr;
        }
        region = schedule.findRegion(region->parent);
    }
    return nullptr;
}

bool physicalSectionMatchesCore(const SyncRegion& section, SyncPhysicalCore core)
{
    if (!section.operation) {
        return false;
    }
    if (core == SyncPhysicalCore::Vector) {
        return isa<SectionVectorOp>(section.operation);
    }
    if (core == SyncPhysicalCore::Cube) {
        return isa<SectionCubeOp>(section.operation);
    }
    return false;
}

bool hasExcludedSemanticInput(const StructuredSyncIR& schedule)
{
    const bool hasDirectExclusion = !schedule.getFailures().empty() || !schedule.getSemanticActions().empty();
    if (hasDirectExclusion) {
        return true;
    }
    if (llvm::any_of(schedule.getSummaries(), [](const SyncOpSummary& summary) {
            return !summary.suppliedProtocols.empty() || summary.queue.has_value();
        })) {
        return true;
    }
    const unsigned physicalSections = llvm::count_if(schedule.getRegions(), [](const SyncRegion& region) {
        return region.kind == SyncRegionKind::PhysicalSection;
    });
    const bool unsupportedRegion = llvm::any_of(schedule.getRegions(), [](const SyncRegion& region) {
        return region.kind == SyncRegionKind::Choice || region.kind == SyncRegionKind::Alternative ||
               region.kind == SyncRegionKind::Loop;
    });
    return physicalSections > 1 || unsupportedRegion;
}

struct VerifiedScheduleChain {
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    SmallVector<const SyncPhase*, 8> phases;
    Operation* tailSection = nullptr;
};

FailureOr<VerifiedScheduleChain> reconstructScheduleChain(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages)
{
    const bool scheduleIsEligible = schedule.isFrozen() && !hasExcludedSemanticInput(schedule);
    if (!scheduleIsEligible) {
        return failure();
    }

    VerifiedScheduleChain chain;
    const SyncRegion* commonSection = nullptr;
    bool sectionInitialized = false;
    Operation* previous = nullptr;
    llvm::DenseSet<Operation*> operations;
    for (const SyncPhase& phase : schedule.getPhases()) {
        const SyncStage* stage = stages.findStageForPhase(phase.id);
        const bool duplicateOperation = phase.operation && !operations.insert(phase.operation).second;
        const bool validStageShape = stage && stage->id != kInvalidSyncId && stage->phases.size() == 1 &&
                                     phase.operation && phase.guard.empty() && phase.iterationDomain.loops.empty() &&
                                     !duplicateOperation && phase.core != SyncPhysicalCore::Unknown;
        if (!validStageShape) {
            return failure();
        }
        if (chain.core == SyncPhysicalCore::Unknown) {
            chain.core = phase.core;
        } else if (chain.core != phase.core) {
            return failure();
        }
        if (previous && !isBefore(previous, phase.operation)) {
            return failure();
        }
        previous = phase.operation;

        const SyncRegion* section = findEnclosingPhysicalSection(schedule, phase);
        if (!sectionInitialized) {
            commonSection = section;
            sectionInitialized = true;
        } else if (commonSection != section) {
            return failure();
        }
        if (section && !physicalSectionMatchesCore(*section, phase.core)) {
            return failure();
        }
        chain.phases.push_back(&phase);
    }
    if (chain.phases.empty()) {
        return failure();
    }
    chain.tailSection = commonSection ? commonSection->operation : nullptr;
    return chain;
}

bool verifyVisibilitySubset(const StructuredSyncIR& schedule)
{
    bool seenGlobalWrite = false;
    for (const SyncPhase& phase : schedule.getPhases()) {
        for (SyncAccessId accessId : phase.accesses) {
            const SyncAccess* access = schedule.findAccess(accessId);
            if (!access || access->visibility == SyncVisibilityClass::Unknown) {
                return false;
            }
            if (access->visibility != SyncVisibilityClass::Global) {
                continue;
            }
            if (phase.pipe == PIPE::PIPE_S || access->mode == SyncAccessMode::Ordered ||
                access->mode == SyncAccessMode::ReadWrite) {
                return false;
            }
            if (access->mode == SyncAccessMode::Read && seenGlobalWrite) {
                return false;
            }
            if (access->mode == SyncAccessMode::Write) {
                seenGlobalWrite = true;
            }
        }
    }
    return true;
}

struct ConcreteProtocol {
    Operation* barrier = nullptr;
    Operation* set = nullptr;
    Operation* wait = nullptr;
};

LogicalResult collectConcreteProtocols(func::FuncOp clone, unsigned boundaryCount,
    llvm::DenseMap<SyncOneShotProtocolId, ConcreteProtocol>& records,
    SmallVectorImpl<Operation*>& tailBarriers)
{
    bool malformed = false;
    clone.walk([&](Operation* operation) {
        const bool fixedSync = isFixedSyncOperation(operation);
        const bool generated = operation->hasAttr(kGeneratedAttr);
        if (!fixedSync && !generated) {
            return;
        }
        if (!fixedSync || !operation->hasAttrOfType<UnitAttr>(kGeneratedAttr)) {
            malformed = true;
            return;
        }
        auto role = operation->getAttrOfType<StringAttr>(kRoleAttr);
        if (!role) {
            malformed = true;
            return;
        }
        const StringRef roleValue = role.getValue();
        if (roleValue == kTailRole) {
            if (!isa<BarrierOp>(operation)) {
                malformed = true;
            } else {
                tailBarriers.push_back(operation);
            }
            return;
        }
        auto protocolAttr = operation->getAttrOfType<IntegerAttr>(kProtocolAttr);
        const bool validProtocolId = protocolAttr && protocolAttr.getInt() >= 0 &&
                                     static_cast<std::uint64_t>(protocolAttr.getInt()) < boundaryCount;
        if (!validProtocolId) {
            malformed = true;
            return;
        }
        ConcreteProtocol& record = records[static_cast<SyncOneShotProtocolId>(protocolAttr.getInt())];
        if (roleValue == "barrier") {
            if (record.barrier || !isa<BarrierOp>(operation)) {
                malformed = true;
            } else {
                record.barrier = operation;
            }
        } else if (roleValue == "event-set") {
            if (record.set || !isa<SetFlagOp>(operation)) {
                malformed = true;
            } else {
                record.set = operation;
            }
        } else if (roleValue == "event-wait") {
            if (record.wait || !isa<WaitFlagOp>(operation)) {
                malformed = true;
            } else {
                record.wait = operation;
            }
        } else {
            malformed = true;
        }
    });
    return failure(malformed);
}

LogicalResult verifyTail(
    func::FuncOp clone, const IRMapping& mapping, Operation* expectedSection, ArrayRef<Operation*> tailBarriers)
{
    if (expectedSection) {
        Operation* section = mapping.lookupOrNull(expectedSection);
        const bool validSection =
            section && section->getNumRegions() == 1 && llvm::hasSingleElement(section->getRegion(0));
        if (!validSection) {
            return failure();
        }
        Block& body = section->getRegion(0).front();
        if (body.empty()) {
            return failure();
        }
        Operation* final = &body.back();
        auto barrier = dyn_cast<BarrierOp>(final);
        auto role = final->getAttrOfType<StringAttr>(kRoleAttr);
        return success(
            barrier && final->hasAttrOfType<UnitAttr>(kGeneratedAttr) && role && role.getValue() == kTailRole &&
            !final->hasAttr(kProtocolAttr) && barrier.getPipe().getPipe() == PIPE::PIPE_ALL &&
            final->hasAttrOfType<UnitAttr>(kSectionLocalTailAttr) &&
            tailBarriers.size() == 1 && tailBarriers.front() == final);
    }

    unsigned returnCount = 0;
    bool invalid = false;
    clone.walk([&](func::ReturnOp operation) {
        ++returnCount;
        Operation* previous = operation->getPrevNode();
        auto barrier = dyn_cast_or_null<BarrierOp>(previous);
        auto role = previous ? previous->getAttrOfType<StringAttr>(kRoleAttr) : StringAttr();
        const bool validTail = barrier && previous->hasAttrOfType<UnitAttr>(kGeneratedAttr) && role &&
                               role.getValue() == kTailRole && !previous->hasAttr(kProtocolAttr) &&
                               !previous->hasAttr(kSectionLocalTailAttr) &&
                               barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
        if (!validTail) {
            invalid = true;
        }
    });
    return success(!invalid && returnCount != 0 && tailBarriers.size() == returnCount);
}

std::uint64_t eventKey(SyncPhysicalCore core, PIPE source, PIPE target, unsigned eventId)
{
    return (static_cast<std::uint64_t>(core) << 24) | (static_cast<std::uint64_t>(source) << 16) |
           (static_cast<std::uint64_t>(target) << 8) | eventId;
}

LogicalResult verifyBoundary(const StructuredSyncIR& schedule, const ProtocolSyncTarget& target,
    const SyncPhase& expectedSource, const SyncPhase& expectedTarget, Operation* source,
    Operation* destination, const ConcreteProtocol& record, llvm::DenseSet<std::uint64_t>& allocatedEvents)
{
    if (expectedSource.pipe == expectedTarget.pipe) {
        auto barrier = dyn_cast_or_null<BarrierOp>(record.barrier);
        return success(barrier && !record.set && !record.wait &&
                       barrier.getPipe().getPipe() == expectedSource.pipe &&
                       target.supportsPipeBarrier({expectedSource.core, expectedSource.pipe}) &&
                       destination->getPrevNode() == record.barrier);
    }

    auto set = dyn_cast_or_null<SetFlagOp>(record.set);
    auto wait = dyn_cast_or_null<WaitFlagOp>(record.wait);
    const bool validEvent = set && wait && !record.barrier &&
                            set.getSrcPipe().getPipe() == expectedSource.pipe &&
                            set.getDstPipe().getPipe() == expectedTarget.pipe &&
                            wait.getSrcPipe().getPipe() == expectedSource.pipe &&
                            wait.getDstPipe().getPipe() == expectedTarget.pipe &&
                            set.getEventId().getEvent() == wait.getEventId().getEvent() &&
                            source->getNextNode() == record.set && destination->getPrevNode() == record.wait &&
                            isBefore(record.set, record.wait) &&
                            target.supportsEvent(
                                {expectedSource.core, expectedSource.pipe},
                                {expectedTarget.core, expectedTarget.pipe});
    if (!validEvent) {
        return failure();
    }

    const unsigned eventId = static_cast<unsigned>(set.getEventId().getEvent());
    return success(llvm::is_contained(target.getCompilerEventIds(), eventId) &&
                   !isReserved(schedule, expectedSource.pipe, expectedTarget.pipe, eventId) &&
                   allocatedEvents
                       .insert(eventKey(expectedSource.core, expectedSource.pipe, expectedTarget.pipe, eventId))
                       .second);
}

LogicalResult verifyBoundaries(const StructuredSyncIR& schedule, const ProtocolSyncTarget& target,
    const VerifiedScheduleChain& chain, const IRMapping& mapping,
    const llvm::DenseMap<SyncOneShotProtocolId, ConcreteProtocol>& records, ProtocolSyncStatistics* statistics)
{
    unsigned expectedConcreteRecords = 0;
    llvm::DenseSet<std::uint64_t> allocatedEvents;
    for (unsigned index = 0; index + 1 < chain.phases.size(); ++index) {
        const SyncPhase& expectedSource = *chain.phases[index];
        const SyncPhase& expectedTarget = *chain.phases[index + 1];
        Operation* source = mapping.lookupOrNull(expectedSource.operation);
        Operation* destination = mapping.lookupOrNull(expectedTarget.operation);
        if (!source || !destination || !isBefore(source, destination)) {
            return failure();
        }

        auto found = records.find(index);
        const bool intrinsic = expectedSource.pipe == PIPE::PIPE_S && expectedTarget.pipe == PIPE::PIPE_S;
        if (intrinsic) {
            if (found != records.end()) {
                return failure();
            }
        } else {
            ++expectedConcreteRecords;
            const bool validBoundary = found != records.end() &&
                                       succeeded(verifyBoundary(schedule, target, expectedSource, expectedTarget,
                                           source, destination, found->second, allocatedEvents));
            if (!validBoundary) {
                return failure();
            }
        }
        if (statistics) {
            ++statistics->verifierTransitions;
        }
    }
    return success(records.size() == expectedConcreteRecords);
}

} // namespace

LogicalResult mlir::pto::protocol_sync::verifyOneShotProtocolMaterialization(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, func::FuncOp clone,
    const IRMapping& mapping, ProtocolSyncStatistics* statistics)
{
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    FailureOr<VerifiedScheduleChain> chain = reconstructScheduleChain(schedule, stages);
    const bool validInputs = target.isSupported() && verifyVisibilitySubset(schedule) && succeeded(chain);
    if (!validInputs) {
        return failure();
    }

    llvm::DenseMap<SyncOneShotProtocolId, ConcreteProtocol> records;
    SmallVector<Operation*, 4> tailBarriers;
    const unsigned boundaryCount = chain->phases.size() - 1;
    if (failed(collectConcreteProtocols(clone, boundaryCount, records, tailBarriers))) {
        return failure();
    }

    const bool verified = succeeded(verifyBoundaries(schedule, target, *chain, mapping, records, statistics)) &&
                          succeeded(verifyTail(clone, mapping, chain->tailSection, tailBarriers));
    if (!verified) {
        return failure();
    }
    return success();
}
