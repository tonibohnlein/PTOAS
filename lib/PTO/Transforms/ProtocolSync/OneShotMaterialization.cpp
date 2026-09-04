// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
//===- OneShotMaterialization.cpp - Clone, emit, verify, commit ----------===//
#include "PTO/Transforms/ProtocolSync/OneShotProtocol.h"

#include "PTO/IR/PTO.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <chrono>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using OneShotClock = std::chrono::steady_clock;
constexpr StringLiteral kGeneratedAttr = "pto.protocol_sync.generated";
constexpr StringLiteral kProtocolAttr = "pto.protocol_sync.protocol_id";
constexpr StringLiteral kRoleAttr = "pto.protocol_sync.role";
constexpr StringLiteral kTailRole = "tail-drain";

std::uint64_t elapsedMicroseconds(OneShotClock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(OneShotClock::now() - start).count();
}

void tagGenerated(Operation* operation, OpBuilder& builder, SyncOneShotProtocolId protocol, StringRef role)
{
    operation->setAttr(kGeneratedAttr, builder.getUnitAttr());
    operation->setAttr(kProtocolAttr, builder.getI32IntegerAttr(static_cast<std::int32_t>(protocol)));
    operation->setAttr(kRoleAttr, builder.getStringAttr(role));
}

void tagTail(BarrierOp barrier, OpBuilder& builder)
{
    barrier->setAttr(kGeneratedAttr, builder.getUnitAttr());
    barrier->setAttr(kRoleAttr, builder.getStringAttr(kTailRole));
    barrier->setAttr("pto.auto_sync_tail_barrier", builder.getUnitAttr());
    barrier->setAttr("pto.auto_sync_tail_hint", builder.getStringAttr("barrier_all"));
}

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

const SyncChannel* findUniqueChannelForGeneration(const ChannelAnalysisResult& channels, SyncGenerationId generation)
{
    const SyncChannel* result = nullptr;
    for (const SyncChannel& channel : channels.getChannels()) {
        if (channel.generation != generation) {
            continue;
        }
        if (result) {
            return nullptr;
        }
        result = &channel;
    }
    return result;
}

const SyncPhase* getOnlyPhase(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, SyncStageId id)
{
    const SyncStage* stage = stages.findStage(id);
    if (!stage || stage->phases.size() != 1) {
        return nullptr;
    }
    return schedule.findPhase(stage->phases.front());
}

bool isRelevantLocalTimeline(const StructuredSyncIR& schedule, const SyncGenerationTimeline& timeline)
{
    const SyncStorageFamily* family = schedule.findStorageFamily(timeline.family);
    return family && family->role == SyncStorageRole::LocalBuffer;
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
    if (!schedule.getFailures().empty() || !schedule.getSemanticActions().empty()) {
        return true;
    }
    if (llvm::any_of(schedule.getSummaries(), [](const SyncOpSummary& summary) {
            return !summary.suppliedProtocols.empty() || summary.queue.has_value();
        })) {
        return true;
    }
    return llvm::any_of(schedule.getRegions(), [](const SyncRegion& region) {
        return region.kind == SyncRegionKind::Choice || region.kind == SyncRegionKind::Alternative ||
               region.kind == SyncRegionKind::Loop;
    });
}

struct VerifiedScheduleChain {
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    SmallVector<const SyncPhase*, 8> phases;
    Operation* tailSection = nullptr;
};

FailureOr<VerifiedScheduleChain> reconstructScheduleChain(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages)
{
    if (!schedule.isFrozen() || hasExcludedSemanticInput(schedule)) {
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
        if (!stage || stage->id == kInvalidSyncId || stage->phases.size() != 1 || !phase.operation ||
            !phase.guard.empty() || !phase.iterationDomain.loops.empty() || duplicateOperation ||
            phase.core == SyncPhysicalCore::Unknown) {
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

LogicalResult verifyTail(
    func::FuncOp clone, const IRMapping& mapping, Operation* expectedSection, ArrayRef<Operation*> tailBarriers)
{
    if (expectedSection) {
        Operation* section = mapping.lookupOrNull(expectedSection);
        if (!section || section->getNumRegions() != 1 || !llvm::hasSingleElement(section->getRegion(0))) {
            return failure();
        }
        Operation* terminator = section->getRegion(0).front().getTerminator();
        Operation* previous = terminator ? terminator->getPrevNode() : nullptr;
        auto barrier = dyn_cast_or_null<BarrierOp>(previous);
        auto role = previous ? previous->getAttrOfType<StringAttr>(kRoleAttr) : StringAttr();
        return success(
            barrier && previous->hasAttrOfType<UnitAttr>(kGeneratedAttr) && role && role.getValue() == kTailRole &&
            barrier.getPipe().getPipe() == PIPE::PIPE_ALL && tailBarriers.size() == 1 &&
            tailBarriers.front() == previous);
    }

    unsigned returnCount = 0;
    bool invalid = false;
    clone.walk([&](func::ReturnOp operation) {
        ++returnCount;
        Operation* previous = operation->getPrevNode();
        auto barrier = dyn_cast_or_null<BarrierOp>(previous);
        auto role = previous ? previous->getAttrOfType<StringAttr>(kRoleAttr) : StringAttr();
        if (!barrier || !previous->hasAttrOfType<UnitAttr>(kGeneratedAttr) || !role || role.getValue() != kTailRole ||
            barrier.getPipe().getPipe() != PIPE::PIPE_ALL) {
            invalid = true;
        }
    });
    return success(!invalid && returnCount != 0 && tailBarriers.size() == returnCount);
}

LogicalResult verifyRelevantChannels(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    ArrayRef<const SyncPhase*> phaseOrder, const SyncOneShotPlan& plan)
{
    llvm::DenseMap<SyncPhaseId, unsigned> phasePosition;
    for (auto [position, phase] : llvm::enumerate(phaseOrder)) {
        phasePosition[phase->id] = position;
    }
    SmallVector<SmallVector<SyncChannelId, 2>, 8> expectedChannels(plan.protocols.size());
    for (const SyncGenerationTimeline& timeline : timelines.getTimelines()) {
        if (!isRelevantLocalTimeline(schedule, timeline)) {
            continue;
        }
        if (!timeline.isAdmitted() || timeline.generationKind != SyncGenerationKind::OneShot ||
            timeline.producers.size() != 1 || timeline.consumers.size() != 1) {
            return failure();
        }
        const SyncChannel* channel = findUniqueChannelForGeneration(channels, timeline.id);
        if (!channel || !channel->isAdmitted() || channel->kind != SyncChannelKind::OneShot ||
            channel->readyOracle != SyncDemandOracleStatus::Match ||
            channel->releaseOracle == SyncDemandOracleStatus::Mismatch) {
            return failure();
        }
        const SyncPhase* producer = getOnlyPhase(schedule, stages, timeline.producers.front());
        const SyncPhase* consumer = getOnlyPhase(schedule, stages, timeline.consumers.front());
        if (!producer || !consumer) {
            return failure();
        }
        auto producerPosition = phasePosition.find(producer->id);
        auto consumerPosition = phasePosition.find(consumer->id);
        if (producerPosition == phasePosition.end() || consumerPosition == phasePosition.end() ||
            producerPosition->second >= consumerPosition->second) {
            return failure();
        }
        for (unsigned boundary = producerPosition->second; boundary < consumerPosition->second; ++boundary) {
            if (boundary >= expectedChannels.size()) {
                return failure();
            }
            expectedChannels[boundary].push_back(channel->id);
        }
    }
    for (auto [boundary, protocol] : llvm::enumerate(plan.protocols)) {
        ArrayRef<SyncChannelId> expected = expectedChannels[boundary];
        if (protocol.channels.size() != expected.size()) {
            return failure();
        }
        for (SyncChannelId channel : protocol.channels) {
            if (llvm::count(protocol.channels, channel) != 1 || !llvm::is_contained(expected, channel)) {
                return failure();
            }
        }
    }
    return success();
}

LogicalResult verifyConcreteProtocolShape(
    func::FuncOp clone, const IRMapping& mapping, const StructuredSyncIR& schedule,
    const PipelineStageAnalysisResult& stages, const StorageTimelineAnalysisResult& timelines,
    const ChannelAnalysisResult& channels, const SyncOneShotPlan& plan, ProtocolSyncStatistics* statistics)
{
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    FailureOr<VerifiedScheduleChain> expectedChain = reconstructScheduleChain(schedule, stages);
    if (!target.isSupported() || !verifyVisibilitySubset(schedule) || failed(expectedChain) ||
        plan.functionCore != expectedChain->core || !plan.emitTailBarrier ||
        plan.tailSectionOperation != expectedChain->tailSection ||
        plan.phaseOrder.size() != expectedChain->phases.size() ||
        plan.protocols.size() + 1 != expectedChain->phases.size()) {
        return failure();
    }
    for (auto [index, phase] : llvm::enumerate(expectedChain->phases)) {
        if (plan.phaseOrder[index] != phase->id) {
            return failure();
        }
    }

    llvm::DenseMap<SyncOneShotProtocolId, ConcreteProtocol> records;
    llvm::SmallVector<Operation*, 4> tailBarriers;
    bool malformedTag = false;
    clone.walk([&](Operation* operation) {
        if (!operation->hasAttrOfType<UnitAttr>(kGeneratedAttr)) {
            return;
        }
        auto role = operation->getAttrOfType<StringAttr>(kRoleAttr);
        if (!role) {
            malformedTag = true;
            return;
        }
        if (role.getValue() == kTailRole) {
            tailBarriers.push_back(operation);
            return;
        }
        auto protocolAttr = operation->getAttrOfType<IntegerAttr>(kProtocolAttr);
        if (!protocolAttr) {
            malformedTag = true;
            return;
        }
        const auto protocolId = static_cast<SyncOneShotProtocolId>(protocolAttr.getInt());
        ConcreteProtocol& record = records[protocolId];
        if (role.getValue() == "barrier") {
            if (record.barrier || !isa<BarrierOp>(operation)) {
                malformedTag = true;
            } else {
                record.barrier = operation;
            }
        } else if (role.getValue() == "event-set") {
            if (record.set || !isa<SetFlagOp>(operation)) {
                malformedTag = true;
            } else {
                record.set = operation;
            }
        } else if (role.getValue() == "event-wait") {
            if (record.wait || !isa<WaitFlagOp>(operation)) {
                malformedTag = true;
            } else {
                record.wait = operation;
            }
        } else {
            malformedTag = true;
        }
    });
    const unsigned expectedConcreteRecords = llvm::count_if(plan.protocols, [](const SyncOneShotProtocol& protocol) {
        return protocol.kind != SyncOneShotProtocolKind::IntrinsicOrder;
    });
    if (malformedTag || records.size() != expectedConcreteRecords) {
        return failure();
    }

    for (auto [index, protocol] : llvm::enumerate(plan.protocols)) {
        const SyncPhase& expectedSource = *expectedChain->phases[index];
        const SyncPhase& expectedTarget = *expectedChain->phases[index + 1];
        if (protocol.id != index || protocol.sourcePhase != expectedSource.id ||
            protocol.targetPhase != expectedTarget.id || protocol.sourceOperation != expectedSource.operation ||
            protocol.targetOperation != expectedTarget.operation || protocol.core != expectedSource.core ||
            expectedSource.core != expectedTarget.core || protocol.sourcePipe != expectedSource.pipe ||
            protocol.targetPipe != expectedTarget.pipe) {
            return failure();
        }
        Operation* source = mapping.lookupOrNull(expectedSource.operation);
        Operation* destination = mapping.lookupOrNull(expectedTarget.operation);
        if (!source || !destination || !isBefore(source, destination)) {
            return failure();
        }
        auto found = records.find(protocol.id);
        const bool intrinsic = expectedSource.pipe == PIPE::PIPE_S && expectedTarget.pipe == PIPE::PIPE_S;
        const bool barrier = expectedSource.pipe == expectedTarget.pipe && !intrinsic;
        if (intrinsic && protocol.kind != SyncOneShotProtocolKind::IntrinsicOrder) {
            return failure();
        }
        if (barrier && protocol.kind != SyncOneShotProtocolKind::PipeBarrier) {
            return failure();
        }
        if (!intrinsic && !barrier && protocol.kind != SyncOneShotProtocolKind::DirectedEvent) {
            return failure();
        }
        if (protocol.kind == SyncOneShotProtocolKind::IntrinsicOrder) {
            if (found != records.end() || protocol.sourcePipe != PIPE::PIPE_S || protocol.targetPipe != PIPE::PIPE_S) {
                return failure();
            }
        } else if (protocol.kind == SyncOneShotProtocolKind::PipeBarrier) {
            if (found == records.end()) {
                return failure();
            }
            const ConcreteProtocol& record = found->second;
            auto barrier = dyn_cast_or_null<BarrierOp>(record.barrier);
            if (!barrier || record.set || record.wait || barrier.getPipe().getPipe() != protocol.sourcePipe ||
                !target.supportsPipeBarrier({protocol.core, protocol.sourcePipe}) ||
                !isBefore(source, record.barrier) || !isBefore(record.barrier, destination)) {
                return failure();
            }
        } else {
            if (found == records.end()) {
                return failure();
            }
            const ConcreteProtocol& record = found->second;
            auto set = dyn_cast_or_null<SetFlagOp>(record.set);
            auto wait = dyn_cast_or_null<WaitFlagOp>(record.wait);
            if (!set || !wait || record.barrier || !protocol.eventId ||
                set.getSrcPipe().getPipe() != protocol.sourcePipe ||
                set.getDstPipe().getPipe() != protocol.targetPipe ||
                wait.getSrcPipe().getPipe() != protocol.sourcePipe ||
                wait.getDstPipe().getPipe() != protocol.targetPipe ||
                static_cast<unsigned>(set.getEventId().getEvent()) != *protocol.eventId ||
                static_cast<unsigned>(wait.getEventId().getEvent()) != *protocol.eventId ||
                !target.supportsEvent({protocol.core, protocol.sourcePipe}, {protocol.core, protocol.targetPipe}) ||
                !llvm::is_contained(target.getCompilerEventIds(), *protocol.eventId) ||
                isReserved(schedule, protocol.sourcePipe, protocol.targetPipe, *protocol.eventId) ||
                !isBefore(source, record.set) || !isBefore(record.set, record.wait) ||
                !isBefore(record.wait, destination)) {
                return failure();
            }
        }
        if (statistics) {
            ++statistics->verifierTransitions;
        }
    }

    for (auto [index, first] : llvm::enumerate(plan.protocols)) {
        if (first.kind != SyncOneShotProtocolKind::DirectedEvent || !first.eventId) {
            continue;
        }
        for (const SyncOneShotProtocol& second : llvm::drop_begin(plan.protocols, index + 1)) {
            const bool sameKey = second.kind == SyncOneShotProtocolKind::DirectedEvent && second.eventId &&
                                 first.core == second.core && first.sourcePipe == second.sourcePipe &&
                                 first.targetPipe == second.targetPipe && *first.eventId == *second.eventId;
            if (sameKey) {
                return failure();
            }
        }
    }

    if (failed(verifyRelevantChannels(schedule, stages, timelines, channels, expectedChain->phases, plan)) ||
        failed(verifyTail(clone, mapping, expectedChain->tailSection, tailBarriers))) {
        return failure();
    }
    return success();
}

} // namespace

LogicalResult mlir::pto::protocol_sync::materializeAndVerifyOneShotProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels, const SyncOneShotPlan& plan,
    ProtocolSyncStatistics* statistics)
{
    if (!schedule.isFrozen() || plan.status != SyncOneShotPlanStatus::Ready || !plan.emitTailBarrier ||
        plan.phaseOrder.empty() || plan.protocols.size() + 1 != plan.phaseOrder.size()) {
        return failure();
    }
    for (const SyncOneShotProtocol& protocol : plan.protocols) {
        if (protocol.kind == SyncOneShotProtocolKind::DirectedEvent && !protocol.eventId) {
            return failure();
        }
    }

    func::FuncOp function = schedule.getFunction();
    IRMapping mapping;
    OwningOpRef<ModuleOp> stagingModule = ModuleOp::create(function.getLoc());
    func::FuncOp clone = cast<func::FuncOp>(function->clone(mapping));
    stagingModule->push_back(clone);
    if (ModuleOp sourceModule = function->getParentOfType<ModuleOp>()) {
        for (func::FuncOp declaration : sourceModule.getOps<func::FuncOp>()) {
            if (declaration != function && declaration.isDeclaration()) {
                stagingModule->push_back(cast<func::FuncOp>(declaration->clone()));
            }
        }
    }

    const OneShotClock::time_point materializationStart = OneShotClock::now();
    OpBuilder builder(clone.getContext());
    for (const SyncOneShotProtocol& protocol : plan.protocols) {
        Operation* source = mapping.lookupOrNull(protocol.sourceOperation);
        Operation* destination = mapping.lookupOrNull(protocol.targetOperation);
        if (!source || !destination) {
            return failure();
        }
        if (protocol.kind == SyncOneShotProtocolKind::IntrinsicOrder) {
            continue;
        }
        if (protocol.kind == SyncOneShotProtocolKind::PipeBarrier) {
            builder.setInsertionPoint(destination);
            auto barrier = builder.create<BarrierOp>(
                destination->getLoc(), PipeAttr::get(clone.getContext(), protocol.sourcePipe));
            tagGenerated(barrier, builder, protocol.id, "barrier");
            if (statistics) {
                ++statistics->materializationTransitions;
            }
            continue;
        }
        builder.setInsertionPointAfter(source);
        auto set = builder.create<SetFlagOp>(
            source->getLoc(), PipeAttr::get(clone.getContext(), protocol.sourcePipe),
            PipeAttr::get(clone.getContext(), protocol.targetPipe),
            EventAttr::get(clone.getContext(), static_cast<EVENT>(*protocol.eventId)));
        tagGenerated(set, builder, protocol.id, "event-set");
        builder.setInsertionPoint(destination);
        auto wait = builder.create<WaitFlagOp>(
            destination->getLoc(), PipeAttr::get(clone.getContext(), protocol.sourcePipe),
            PipeAttr::get(clone.getContext(), protocol.targetPipe),
            EventAttr::get(clone.getContext(), static_cast<EVENT>(*protocol.eventId)));
        tagGenerated(wait, builder, protocol.id, "event-wait");
        if (statistics) {
            statistics->materializationTransitions += 2;
        }
    }

    if (plan.tailSectionOperation) {
        Operation* section = mapping.lookupOrNull(plan.tailSectionOperation);
        if (!section || section->getNumRegions() != 1 || !llvm::hasSingleElement(section->getRegion(0))) {
            return failure();
        }
        Operation* terminator = section->getRegion(0).front().getTerminator();
        if (!terminator) {
            return failure();
        }
        builder.setInsertionPoint(terminator);
        auto barrier =
            builder.create<BarrierOp>(terminator->getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
        tagTail(barrier, builder);
        if (statistics) {
            ++statistics->materializationTransitions;
        }
    } else {
        clone.walk([&](func::ReturnOp operation) {
            builder.setInsertionPoint(operation);
            auto barrier =
                builder.create<BarrierOp>(operation.getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
            tagTail(barrier, builder);
            if (statistics) {
                ++statistics->materializationTransitions;
            }
        });
    }
    if (statistics) {
        statistics->materializationUs += elapsedMicroseconds(materializationStart);
    }

    const OneShotClock::time_point verificationStart = OneShotClock::now();
    if (failed(verifyConcreteProtocolShape(clone, mapping, schedule, stages, timelines, channels, plan, statistics)) ||
        failed(mlir::verify(*stagingModule))) {
        if (statistics) {
            statistics->verificationUs += elapsedMicroseconds(verificationStart);
        }
        function.emitError("ProtocolSync rejected its staged one-shot materialization; original IR is unchanged");
        return failure();
    }
    if (statistics) {
        statistics->verificationUs += elapsedMicroseconds(verificationStart);
    }
    function.getBody().takeBody(clone.getBody());
    return success();
}
