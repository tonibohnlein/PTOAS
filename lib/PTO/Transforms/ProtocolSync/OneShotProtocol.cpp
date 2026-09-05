// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
//===- OneShotProtocol.cpp - Checkpoint D planning and allocation --------===//
#include "PTO/Transforms/ProtocolSync/OneShotProtocol.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/ProtocolSync/EventAllocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using ChannelIndex = llvm::DenseMap<SyncGenerationId, const SyncChannel*>;

ChannelIndex buildChannelIndex(const ChannelAnalysisResult& channels)
{
    ChannelIndex index;
    for (const SyncChannel& channel : channels.getChannels()) {
        auto [entry, inserted] = index.try_emplace(channel.generation, &channel);
        if (!inserted) {
            entry->second = nullptr;
        }
    }
    return index;
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

void reject(SyncOneShotPlan& plan, SyncChannelId channel, SyncOneShotRejection reason, StringRef detail)
{
    plan.status = SyncOneShotPlanStatus::Unsupported;
    plan.rejections.push_back({channel, reason, detail.str()});
}

bool hasExistingSynchronization(const StructuredSyncIR& schedule)
{
    return llvm::any_of(schedule.getSummaries(), [](const SyncOpSummary& summary) {
        return !summary.suppliedProtocols.empty() || summary.queue.has_value();
    });
}

bool operationsAreStrictlyOrdered(Operation* source, Operation* target)
{
    return source && target && source != target && source->getBlock() == target->getBlock() &&
           source->isBeforeInBlock(target);
}

const SyncRegion* enclosingPhysicalSection(const StructuredSyncIR& schedule, const SyncPhase& phase)
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

bool sectionMatchesCore(const SyncRegion& section, SyncPhysicalCore core)
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

bool hasUnsupportedStructuredControl(const StructuredSyncIR& schedule)
{
    return llvm::any_of(schedule.getRegions(), [](const SyncRegion& region) {
        return region.kind == SyncRegionKind::Choice || region.kind == SyncRegionKind::Alternative ||
               region.kind == SyncRegionKind::Loop;
    });
}

bool hasMultiplePhysicalSections(const StructuredSyncIR& schedule)
{
    return llvm::count_if(schedule.getRegions(), [](const SyncRegion& region) {
               return region.kind == SyncRegionKind::PhysicalSection;
           }) > 1;
}

/// Checkpoint D does not yet have the generation-aware residual-obligation
/// interpreter. It therefore admits only a deliberately total completion chain.
/// Global publication/acquisition requiring cache semantics is outside this
/// milestone and must fail closed.
bool hasUnsupportedVisibility(const StructuredSyncIR& schedule, StringRef& detail)
{
    bool seenGlobalWrite = false;
    for (const SyncPhase& phase : schedule.getPhases()) {
        for (SyncAccessId accessId : phase.accesses) {
            const SyncAccess* access = schedule.findAccess(accessId);
            if (!access) {
                detail = "a phase references a missing access";
                return true;
            }
            if (access->visibility == SyncVisibilityClass::Unknown) {
                detail = "an access has unknown local/global visibility";
                return true;
            }
            if (access->visibility != SyncVisibilityClass::Global) {
                continue;
            }
            if (phase.pipe == PIPE::PIPE_S) {
                detail = "scalar GM visibility requires a cache-maintenance protocol";
                return true;
            }
            if (access->mode == SyncAccessMode::Ordered || access->mode == SyncAccessMode::ReadWrite) {
                detail = "ordered or read-write GM effects require residual visibility analysis";
                return true;
            }
            if (access->mode == SyncAccessMode::Read && seenGlobalWrite) {
                detail = "a GM read follows a possible GM write and requires a publication contract";
                return true;
            }
            if (access->mode == SyncAccessMode::Write) {
                seenGlobalWrite = true;
            }
        }
    }
    return false;
}

LogicalResult validateRelevantChannels(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const llvm::DenseMap<SyncPhaseId, unsigned>& phasePosition, SyncOneShotPlan& plan)
{
    const ChannelIndex channelIndex = buildChannelIndex(channels);
    for (const SyncGenerationTimeline& timeline : timelines.getTimelines()) {
        if (!isRelevantLocalTimeline(schedule, timeline)) {
            continue;
        }
        if (!timeline.isAdmitted()) {
            reject(
                plan, kInvalidSyncId, SyncOneShotRejection::NonOneShotChannel,
                "Checkpoint D requires every local-storage timeline to be admitted");
            return failure();
        }
        if (timeline.producers.size() != 1 || timeline.consumers.size() != 1) {
            reject(
                plan, kInvalidSyncId, SyncOneShotRejection::NonOneShotChannel,
                "Checkpoint D requires every local timeline to have exactly one producer and one consumer");
            return failure();
        }
        auto foundChannel = channelIndex.find(timeline.id);
        const SyncChannel* channel = foundChannel == channelIndex.end() ? nullptr : foundChannel->second;
        if (!channel) {
            reject(
                plan, kInvalidSyncId, SyncOneShotRejection::IncompleteChannelSet,
                "a relevant local timeline does not have exactly one channel record");
            return failure();
        }
        if (!channel->isAdmitted()) {
            reject(
                plan, channel->id, SyncOneShotRejection::NonOneShotChannel,
                "Checkpoint D requires every relevant local timeline to be an admitted one-shot channel");
            return failure();
        }
        if (timeline.generationKind != SyncGenerationKind::OneShot || channel->kind != SyncChannelKind::OneShot ||
            timeline.producers.size() != 1 || timeline.consumers.size() != 1) {
            reject(
                plan, channel->id, SyncOneShotRejection::NonOneShotChannel,
                "Checkpoint D supports only one-producer/one-consumer one-shot channels");
            return failure();
        }
        if (channel->readyOracle != SyncDemandOracleStatus::Match ||
            channel->releaseOracle == SyncDemandOracleStatus::Mismatch) {
            reject(
                plan, channel->id, SyncOneShotRejection::UnverifiedChannel,
                "the legacy demand oracle did not authenticate the one-shot channel");
            return failure();
        }
        const SyncPhase* producer = getOnlyPhase(schedule, stages, timeline.producers.front());
        const SyncPhase* consumer = getOnlyPhase(schedule, stages, timeline.consumers.front());
        if (!producer || !consumer) {
            reject(
                plan, channel->id, SyncOneShotRejection::UnsupportedStageShape,
                "channel endpoints must each contain one exact physical phase");
            return failure();
        }
        auto producerPosition = phasePosition.find(producer->id);
        auto consumerPosition = phasePosition.find(consumer->id);
        if (producerPosition == phasePosition.end() || consumerPosition == phasePosition.end() ||
            producerPosition->second >= consumerPosition->second) {
            reject(
                plan, channel->id, SyncOneShotRejection::UnorderedEndpoints,
                "one-shot channel endpoints are not forward ordered in the certified phase chain");
            return failure();
        }
        for (unsigned boundary = producerPosition->second; boundary < consumerPosition->second; ++boundary) {
            if (boundary >= plan.protocols.size()) {
                return failure();
            }
            plan.protocols[boundary].channels.push_back(channel->id);
        }
    }
    return success();
}

} // namespace

FailureOr<SyncOneShotPlan> mlir::pto::protocol_sync::buildOneShotProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    ProtocolSyncStatistics* statistics)
{
    if (!schedule.isFrozen()) {
        return failure();
    }

    SyncOneShotPlan plan;
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.supportsOneShotEmission()) {
        reject(
            plan, kInvalidSyncId, SyncOneShotRejection::UnsupportedTarget,
            target.getUnsupportedReason(ProtocolSyncEmissionMode::OneShot));
        return plan;
    }
    plan.targetKind = target.getKind();
    plan.capabilityProfile = target.getCapabilityProfile();
    if (!schedule.getFailures().empty()) {
        reject(
            plan, kInvalidSyncId, SyncOneShotRejection::ScheduleFailure,
            "the immutable schedule contains unsupported or incomplete semantic facts");
        return plan;
    }
    if (hasExistingSynchronization(schedule)) {
        reject(
            plan, kInvalidSyncId, SyncOneShotRejection::ExistingSynchronization,
            "Checkpoint D does not compose generated protocols with existing synchronization");
        return plan;
    }
    if (!schedule.getSemanticActions().empty()) {
        reject(
            plan, kInvalidSyncId, SyncOneShotRejection::SemanticAction,
            "Checkpoint D does not yet compose generated protocols with ordered semantic actions");
        return plan;
    }
    if (hasUnsupportedStructuredControl(schedule)) {
        reject(
            plan, kInvalidSyncId, SyncOneShotRejection::UnsupportedControlFlow,
            "Checkpoint D supports only an exactly-once linear phase chain");
        return plan;
    }
    if (hasMultiplePhysicalSections(schedule)) {
        reject(
            plan, kInvalidSyncId, SyncOneShotRejection::MixedPhysicalSections,
            "Checkpoint D supports at most one physical section per function");
        return plan;
    }

    StringRef visibilityDetail;
    if (hasUnsupportedVisibility(schedule, visibilityDetail)) {
        reject(plan, kInvalidSyncId, SyncOneShotRejection::UnsupportedVisibility, visibilityDetail);
        return plan;
    }

    const SyncRegion* commonSection = nullptr;
    bool sectionChoiceInitialized = false;
    Operation* previousOperation = nullptr;
    llvm::DenseMap<Operation*, SyncPhaseId> operationToPhase;
    llvm::DenseMap<SyncPhaseId, unsigned> phasePosition;

    for (const SyncPhase& phase : schedule.getPhases()) {
        const SyncStage* stage = stages.findStageForPhase(phase.id);
        if (!stage || stage->phases.size() != 1 || stage->id == kInvalidSyncId || !phase.operation ||
            !phase.guard.empty() || !phase.iterationDomain.loops.empty()) {
            reject(
                plan, kInvalidSyncId, SyncOneShotRejection::UnsupportedStageShape,
                "every phase must be one unguarded, once-only exact physical stage");
            return plan;
        }
        if (operationToPhase.count(phase.operation) != 0) {
            reject(
                plan, kInvalidSyncId, SyncOneShotRejection::UnsupportedStageShape,
                "Checkpoint D does not lower multi-phase macros");
            return plan;
        }
        operationToPhase[phase.operation] = phase.id;

        if (phase.core == SyncPhysicalCore::Unknown) {
            reject(
                plan, kInvalidSyncId, SyncOneShotRejection::MixedPhysicalCores, "a physical phase has no known core");
            return plan;
        }
        if (plan.functionCore == SyncPhysicalCore::Unknown) {
            plan.functionCore = phase.core;
        } else if (plan.functionCore != phase.core) {
            reject(
                plan, kInvalidSyncId, SyncOneShotRejection::MixedPhysicalCores,
                "Checkpoint D supports only one physical core per function");
            return plan;
        }
        if (previousOperation && !operationsAreStrictlyOrdered(previousOperation, phase.operation)) {
            reject(
                plan, kInvalidSyncId, SyncOneShotRejection::UnorderedEndpoints,
                "physical phases must form one strict lexical block order");
            return plan;
        }
        previousOperation = phase.operation;

        const SyncRegion* section = enclosingPhysicalSection(schedule, phase);
        if (!sectionChoiceInitialized) {
            commonSection = section;
            sectionChoiceInitialized = true;
        } else if (commonSection != section) {
            reject(
                plan, kInvalidSyncId, SyncOneShotRejection::MixedPhysicalSections,
                "physical phases span different section ownership contexts");
            return plan;
        }
        if (section && !sectionMatchesCore(*section, phase.core)) {
            reject(
                plan, kInvalidSyncId, SyncOneShotRejection::MixedPhysicalSections,
                "the physical section kind disagrees with the phase core");
            return plan;
        }

        phasePosition[phase.id] = plan.phaseOrder.size();
        plan.phaseOrder.push_back(phase.id);
    }

    if (plan.phaseOrder.empty()) {
        plan.status = SyncOneShotPlanStatus::Empty;
        return plan;
    }

    plan.emitTailBarrier = true;
    plan.tailSectionOperation = commonSection ? commonSection->operation : nullptr;

    for (unsigned index = 0; index + 1 < plan.phaseOrder.size(); ++index) {
        const SyncPhase* source = schedule.findPhase(plan.phaseOrder[index]);
        const SyncPhase* destination = schedule.findPhase(plan.phaseOrder[index + 1]);
        if (!source || !destination) {
            return failure();
        }
        SyncOneShotProtocol protocol;
        protocol.id = plan.protocols.size();
        protocol.core = source->core;
        protocol.sourcePipe = source->pipe;
        protocol.targetPipe = destination->pipe;
        protocol.sourcePhase = source->id;
        protocol.targetPhase = destination->id;
        protocol.sourceOperation = source->operation;
        protocol.targetOperation = destination->operation;

        if (source->pipe == destination->pipe && source->pipe == PIPE::PIPE_S) {
            protocol.kind = SyncOneShotProtocolKind::IntrinsicOrder;
        } else if (source->pipe == destination->pipe) {
            protocol.kind = SyncOneShotProtocolKind::PipeBarrier;
            if (!target.supportsPipeBarrier({source->core, source->pipe})) {
                reject(
                    plan, kInvalidSyncId, SyncOneShotRejection::UnsupportedBarrier,
                    "the target does not support the required same-pipeline barrier");
                return plan;
            }
        } else {
            protocol.kind = SyncOneShotProtocolKind::DirectedEvent;
            if (!target.supportsEvent({source->core, source->pipe}, {destination->core, destination->pipe})) {
                reject(
                    plan, kInvalidSyncId, SyncOneShotRejection::UnsupportedEventDirection,
                    "the target does not support the required adjacent directed event");
                return plan;
            }
        }
        plan.protocols.push_back(std::move(protocol));
    }

    if (failed(validateRelevantChannels(schedule, stages, timelines, channels, phasePosition, plan))) {
        if (plan.status != SyncOneShotPlanStatus::Unsupported) {
            return failure();
        }
        return plan;
    }

    plan.status = SyncOneShotPlanStatus::Ready;
    if (statistics) {
        statistics->protocolCandidates += plan.protocols.size();
        statistics->selectedOneShotProtocols += plan.protocols.size();
        for (const SyncOneShotProtocol& protocol : plan.protocols) {
            if (protocol.kind == SyncOneShotProtocolKind::PipeBarrier) {
                ++statistics->logicalActions;
                ++statistics->selectedSamePipeBarriers;
            } else if (protocol.kind == SyncOneShotProtocolKind::DirectedEvent) {
                statistics->logicalActions += 2;
                ++statistics->selectedDirectedEventPairs;
            }
        }
        ++statistics->logicalActions; // mandatory PIPE_ALL tail drain
        ++statistics->selectedTailDrains;
    }
    return plan;
}

LogicalResult mlir::pto::protocol_sync::allocateOneShotProtocolEvents(
    const StructuredSyncIR& schedule, SyncOneShotPlan& plan, ProtocolSyncStatistics* statistics)
{
    if (plan.status != SyncOneShotPlanStatus::Ready) {
        return success();
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    const bool targetMatchesPlan = target.supportsOneShotEmission() &&
                                   target.getCapabilityProfile() == plan.capabilityProfile;
    if (!targetMatchesPlan) {
        return failure();
    }

    SmallVector<SyncEventReservation, 8> reservations;
    for (const SyncOpSummary& summary : schedule.getSummaries()) {
        reservations.append(summary.eventReservations.begin(), summary.eventReservations.end());
    }
    SmallVector<SyncEventGeneration, 8> generations;
    SmallVector<SyncOneShotProtocol*, 8> eventProtocols;
    for (SyncOneShotProtocol& protocol : plan.protocols) {
        if (protocol.kind != SyncOneShotProtocolKind::DirectedEvent) {
            continue;
        }
        if (protocol.eventId) {
            return failure();
        }
        SyncEventGeneration generation;
        generation.id = generations.size();
        generation.kind = SyncEventGenerationKind::OneShot;
        generation.core = protocol.core;
        generation.sourcePipe = protocol.sourcePipe;
        generation.targetPipe = protocol.targetPipe;
        generation.setAnchor = protocol.sourceOperation;
        generation.waitAnchor = protocol.targetOperation;
        generations.push_back(std::move(generation));
        eventProtocols.push_back(&protocol);
    }
    FailureOr<SyncEventAllocationResult> allocation = allocateSyncEventGenerations(target, reservations, generations);
    if (failed(allocation)) {
        return failure();
    }
    if (statistics) {
        recordSyncEventAllocationStatistics(*allocation, *statistics);
    }
    if (allocation->status == SyncEventAllocationStatus::ResourceInfeasible) {
        const SyncOneShotProtocol* witness = eventProtocols.empty() ? nullptr : eventProtocols.front();
        reject(
            plan, !witness || witness->channels.empty() ? kInvalidSyncId : witness->channels.front(),
            SyncOneShotRejection::EventCapacity,
            "interfering event generations exhaust the compiler event pool in one directed domain");
        return success();
    }
    if (allocation->status != SyncEventAllocationStatus::Allocated) {
        return failure();
    }
    for (auto [protocol, eventId] : llvm::zip_equal(eventProtocols, allocation->eventIds)) {
        protocol->eventId = eventId;
    }
    return success();
}

StringRef mlir::pto::protocol_sync::stringifySyncOneShotProtocolKind(SyncOneShotProtocolKind kind)
{
    switch (kind) {
        case SyncOneShotProtocolKind::IntrinsicOrder:
            return "intrinsic-order";
        case SyncOneShotProtocolKind::PipeBarrier:
            return "pipe-barrier";
        case SyncOneShotProtocolKind::DirectedEvent:
            return "directed-event";
    }
    return "unknown";
}

StringRef mlir::pto::protocol_sync::stringifySyncOneShotPlanStatus(SyncOneShotPlanStatus status)
{
    switch (status) {
        case SyncOneShotPlanStatus::Empty:
            return "empty";
        case SyncOneShotPlanStatus::Ready:
            return "ready";
        case SyncOneShotPlanStatus::Unsupported:
            return "unsupported";
    }
    return "unsupported";
}

StringRef mlir::pto::protocol_sync::stringifySyncOneShotRejection(SyncOneShotRejection rejection)
{
    switch (rejection) {
        case SyncOneShotRejection::None:
            return "none";
        case SyncOneShotRejection::UnsupportedTarget:
            return "unsupported-target";
        case SyncOneShotRejection::ExistingSynchronization:
            return "existing-synchronization";
        case SyncOneShotRejection::ScheduleFailure:
            return "schedule-failure";
        case SyncOneShotRejection::SemanticAction:
            return "semantic-action";
        case SyncOneShotRejection::UnsupportedControlFlow:
            return "unsupported-control-flow";
        case SyncOneShotRejection::UnsupportedStageShape:
            return "unsupported-stage-shape";
        case SyncOneShotRejection::MixedPhysicalCores:
            return "mixed-physical-cores";
        case SyncOneShotRejection::MixedPhysicalSections:
            return "mixed-physical-sections";
        case SyncOneShotRejection::UnorderedEndpoints:
            return "unordered-endpoints";
        case SyncOneShotRejection::UnsupportedBarrier:
            return "unsupported-barrier";
        case SyncOneShotRejection::UnsupportedEventDirection:
            return "unsupported-event-direction";
        case SyncOneShotRejection::UnsupportedVisibility:
            return "unsupported-visibility";
        case SyncOneShotRejection::NonOneShotChannel:
            return "non-one-shot-channel";
        case SyncOneShotRejection::UnverifiedChannel:
            return "unverified-channel";
        case SyncOneShotRejection::IncompleteChannelSet:
            return "incomplete-channel-set";
        case SyncOneShotRejection::EventCapacity:
            return "event-capacity";
        case SyncOneShotRejection::InternalInvariant:
            return "internal-invariant";
    }
    return "internal-invariant";
}

void mlir::pto::protocol_sync::printOneShotProtocolPlan(
    func::FuncOp function, const SyncOneShotPlan& plan, raw_ostream& output)
{
    output << "PROTOCOL-SYNC one-shot-plan function=@" << function.getSymName()
           << " status=" << stringifySyncOneShotPlanStatus(plan.status) << " phases=" << plan.phaseOrder.size()
           << " protocols=" << plan.protocols.size() << " core=" << stringifySyncPhysicalCore(plan.functionCore)
           << " target=" << stringifyProtocolSyncTargetKind(plan.targetKind)
           << " profile=" << stringifyProtocolSyncCapabilityProfile(plan.capabilityProfile) << '\n';
    for (const SyncOneShotProtocol& protocol : plan.protocols) {
        output << "  protocol #" << protocol.id << " kind=" << stringifySyncOneShotProtocolKind(protocol.kind)
               << " phases=#" << protocol.sourcePhase << "->#" << protocol.targetPhase
               << " source-pipe=" << static_cast<unsigned>(protocol.sourcePipe)
               << " target-pipe=" << static_cast<unsigned>(protocol.targetPipe) << " source="
               << (protocol.sourceOperation ? protocol.sourceOperation->getName().getStringRef() : StringRef("<none>"))
               << " target="
               << (protocol.targetOperation ? protocol.targetOperation->getName().getStringRef() : StringRef("<none>"))
               << " event-id=";
        if (protocol.eventId) {
            output << *protocol.eventId;
        } else {
            output << "none";
        }
        output << " channels=[";
        llvm::interleaveComma(protocol.channels, output);
        output << "]\n";
    }
    output << "  tail-drain=" << (plan.emitTailBarrier ? "true" : "false")
           << " placement=" << (plan.tailSectionOperation ? "physical-section" : "function-return") << '\n';
    for (const SyncOneShotPlanRejection& rejection : plan.rejections) {
        output << "  rejection channel=";
        if (rejection.channel == kInvalidSyncId) {
            output << "none";
        } else {
            output << '#' << rejection.channel;
        }
        output << " reason=" << stringifySyncOneShotRejection(rejection.reason) << " detail=\"" << rejection.detail
               << "\"\n";
    }
}
