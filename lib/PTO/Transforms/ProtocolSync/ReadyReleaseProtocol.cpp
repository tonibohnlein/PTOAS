// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ReadyReleaseProtocol.cpp - Recurring protocol planning ----------===//

#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

constexpr unsigned kCheckpointEMaximumCapacity = 2;
constexpr unsigned kTokenWitnessHorizon = 6;

std::uint32_t eventDomain(SyncPhysicalCore core, PIPE source, PIPE target)
{
    return (static_cast<std::uint32_t>(core) << 16) | (static_cast<std::uint32_t>(source) << 8) |
           static_cast<std::uint32_t>(target);
}

std::uint32_t reservationDomain(PIPE source, PIPE target)
{
    return (static_cast<std::uint32_t>(source) << 8) | static_cast<std::uint32_t>(target);
}

void reject(SyncReadyReleasePlan& plan, SyncChannelId channel, SyncReadyReleaseRejection reason, StringRef detail)
{
    plan.status = SyncReadyReleasePlanStatus::Unsupported;
    plan.rejections.push_back({channel, reason, detail.str()});
}

bool hasExistingSynchronization(const StructuredSyncIR& schedule)
{
    return llvm::any_of(schedule.getSummaries(), [](const SyncOpSummary& summary) {
        return !summary.suppliedProtocols.empty() || summary.queue.has_value();
    });
}

bool hasUnsupportedControl(const StructuredSyncIR& schedule, const SyncRegion*& loopRegion)
{
    loopRegion = nullptr;
    for (const SyncRegion& region : schedule.getRegions()) {
        if (region.kind == SyncRegionKind::Choice || region.kind == SyncRegionKind::Alternative ||
            region.kind == SyncRegionKind::PhysicalSection) {
            return true;
        }
        if (region.kind != SyncRegionKind::Loop) {
            continue;
        }
        if (loopRegion || region.cardinality != SyncCardinality::ZeroOrMore ||
            !isa_and_nonnull<scf::ForOp>(region.operation)) {
            return true;
        }
        loopRegion = &region;
    }
    return !loopRegion;
}

bool hasUnsupportedVisibility(const StructuredSyncIR& schedule)
{
    llvm::DenseMap<SyncStorageFamilyId, unsigned> globalModes;
    for (const SyncAccess& access : schedule.getAccesses()) {
        if (access.visibility == SyncVisibilityClass::Unknown) {
            return true;
        }
        if (access.visibility != SyncVisibilityClass::Global) {
            continue;
        }
        const SyncPhase* phase = schedule.findPhase(access.phase);
        if (!phase || phase->pipe == PIPE::PIPE_S || access.mode == SyncAccessMode::Ordered ||
            access.mode == SyncAccessMode::ReadWrite) {
            return true;
        }
        unsigned& modes = globalModes[access.family];
        modes |= access.mode == SyncAccessMode::Read ? 1U : 2U;
        if (modes == 3U) {
            return true;
        }
    }
    return false;
}

const SyncPhase* getOnlyPhase(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, SyncStageId id)
{
    const SyncStage* stage = stages.findStage(id);
    return stage && stage->phases.size() == 1 ? schedule.findPhase(stage->phases.front()) : nullptr;
}

bool frontierIs(const SyncProgramFrontier& frontier, SyncProgramPointId point)
{
    return frontier.points.size() == 1 && frontier.points.front().point == point &&
           frontier.points.front().guard.empty();
}

bool sameCanonicalSlotExpression(const SyncSlotExpression& first, const SyncSlotExpression& second)
{
    return first.kind == second.kind && first.depth == second.depth && first.loop == second.loop &&
           first.coefficient == second.coefficient && first.offset == second.offset && first.modulus == second.modulus;
}

std::optional<SyncSlotExpression> findProducerSlot(
    const StructuredSyncIR& schedule, const SyncGenerationTimeline& timeline, const SyncPhase& producer)
{
    for (const SyncAccess& access : schedule.getAccesses()) {
        const bool matchesProducer = access.phase == producer.id && access.family == timeline.family &&
                                     access.mode == SyncAccessMode::Write && access.slot && timeline.slot &&
                                     sameCanonicalSlotExpression(*access.slot, *timeline.slot);
        if (matchesProducer) {
            return access.slot;
        }
    }
    return std::nullopt;
}

unsigned slotAtIteration(const SyncReadyReleasePlan& plan, unsigned iteration)
{
    if (plan.capacity == 1) {
        return 0;
    }
    if (!plan.slot || plan.slot->modulus == 0) {
        return plan.capacity;
    }
    const SyncSlotExpression& slot = *plan.slot;
    const std::int64_t signedModulus = static_cast<std::int64_t>(slot.modulus);
    const auto normalize = [&](std::int64_t value) {
        const std::int64_t remainder = value % signedModulus;
        return static_cast<std::uint64_t>(remainder < 0 ? remainder + signedModulus : remainder);
    };
    const std::uint64_t coefficient = normalize(slot.coefficient);
    const std::uint64_t offset = normalize(slot.offset);
    const std::uint64_t iterationModulo = static_cast<std::uint64_t>(iteration) % slot.modulus;
    const std::uint64_t product = (coefficient * iterationModulo) % slot.modulus;
    return static_cast<unsigned>((product + offset) % slot.modulus);
}

bool applyTokenTransfer(
    const SyncReadyReleasePlan& plan, unsigned iteration, llvm::MutableArrayRef<bool> free,
    llvm::MutableArrayRef<bool> ready)
{
    const unsigned lane = slotAtIteration(plan, iteration);
    if (lane >= plan.capacity || !free[lane] || ready[lane]) {
        return false;
    }
    free[lane] = false;
    ready[lane] = true;
    if (free[lane] || !ready[lane]) {
        return false;
    }
    ready[lane] = false;
    free[lane] = true;
    return true;
}

bool simulateTripCount(const SyncReadyReleasePlan& plan, unsigned tripCount, unsigned& transitionApplications)
{
    llvm::SmallVector<bool, 2> free(plan.capacity, true);
    llvm::SmallVector<bool, 2> ready(plan.capacity, false);
    for (unsigned iteration = 0; iteration < tripCount; ++iteration) {
        ++transitionApplications;
        if (!applyTokenTransfer(plan, iteration, free, ready)) {
            return false;
        }
    }
    return llvm::all_of(free, [](bool value) { return value; }) &&
           llvm::none_of(ready, [](bool value) { return value; });
}

bool canonicalStateIsInductive(const SyncReadyReleasePlan& plan, unsigned& transitionApplications)
{
    const bool validSingleLane = plan.capacity == 1 && !plan.slot;
    const bool validDoubleLane = plan.capacity == 2 && plan.slot &&
                                 plan.slot->kind == SyncSlotExpressionKind::AffineModulo && plan.slot->depth == 2 &&
                                 plan.slot->modulus == 2 && plan.slot->coefficient == 1 && plan.slot->offset >= 0 &&
                                 plan.slot->offset < 2;
    if (!validSingleLane && !validDoubleLane) {
        return false;
    }

    // Starting from the canonical Free state, one selector period visits each
    // lane exactly once and every complete body transfer returns that lane to
    // Free. Repetition of this period is the arbitrary-trip induction step;
    // the finite runs below remain boundary witnesses rather than the proof.
    llvm::SmallVector<bool, 2> free(plan.capacity, true);
    llvm::SmallVector<bool, 2> ready(plan.capacity, false);
    llvm::SmallVector<bool, 2> visited(plan.capacity, false);
    for (unsigned iteration = 0; iteration < plan.capacity; ++iteration) {
        const unsigned lane = slotAtIteration(plan, iteration);
        if (lane >= plan.capacity || visited[lane]) {
            return false;
        }
        visited[lane] = true;
        ++transitionApplications;
        if (!applyTokenTransfer(plan, iteration, free, ready)) {
            return false;
        }
    }
    return llvm::all_of(visited, [](bool value) { return value; }) &&
           llvm::all_of(free, [](bool value) { return value; }) &&
           llvm::none_of(ready, [](bool value) { return value; });
}

bool buildTokenCertificate(SyncReadyReleasePlan& plan)
{
    unsigned& transitionApplications = plan.tokenCertificate.transitionApplications;
    transitionApplications = 0;
    plan.tokenCertificate.witnessHorizon = kTokenWitnessHorizon;
    for (unsigned iteration = 0; iteration < kTokenWitnessHorizon; ++iteration) {
        plan.tokenCertificate.slotWitness.push_back(slotAtIteration(plan, iteration));
    }
    plan.tokenCertificate.zeroTripSafe = simulateTripCount(plan, 0, transitionApplications);
    plan.tokenCertificate.oneTripSafe = simulateTripCount(plan, 1, transitionApplications);
    plan.tokenCertificate.oddEvenSafe =
        simulateTripCount(plan, 2, transitionApplications) && simulateTripCount(plan, 3, transitionApplications) &&
        simulateTripCount(plan, 4, transitionApplications) && simulateTripCount(plan, 5, transitionApplications);
    plan.tokenCertificate.steadyStateStable = canonicalStateIsInductive(plan, transitionApplications);
    return plan.tokenCertificate.zeroTripSafe && plan.tokenCertificate.oneTripSafe &&
           plan.tokenCertificate.oddEvenSafe && plan.tokenCertificate.steadyStateStable;
}

bool validateTimelineFrontiers(
    const SyncGenerationTimeline& timeline, const SyncPhase& producer, const SyncPhase& consumer)
{
    return frontierIs(timeline.publication, producer.after) && timeline.acquisitions.size() == 1 &&
           frontierIs(timeline.acquisitions.front(), consumer.before) && timeline.finalUses.size() == 1 &&
           frontierIs(timeline.finalUses.front(), consumer.after) && timeline.nextOverwrite &&
           frontierIs(timeline.nextOverwrite->frontier, producer.before) &&
           timeline.nextOverwrite->iterationDistance == timeline.reuseDistance;
}

bool validateEndpointShape(const SyncPhase& producer, const SyncPhase& consumer, const SyncRegion& loopRegion)
{
    auto loop = dyn_cast_or_null<scf::ForOp>(loopRegion.operation);
    return loop && producer.operation && consumer.operation && producer.operation->getBlock() == loop.getBody() &&
           consumer.operation->getBlock() == loop.getBody() && producer.operation != consumer.operation &&
           producer.operation->getNextNode() == consumer.operation && producer.guard.empty() &&
           consumer.guard.empty() && producer.iterationDomain.loops.size() == 1 &&
           producer.iterationDomain.loops.front() == loopRegion.id &&
           consumer.iterationDomain.loops == producer.iterationDomain.loops;
}

} // namespace

FailureOr<SyncReadyReleasePlan> mlir::pto::protocol_sync::buildReadyReleaseProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    ProtocolSyncStatistics* statistics)
{
    if (!schedule.isFrozen()) {
        return failure();
    }

    SyncReadyReleasePlan plan;
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.isSupported()) {
        reject(plan, kInvalidSyncId, SyncReadyReleaseRejection::UnsupportedTarget, target.getUnsupportedReason());
        return plan;
    }
    if (!target.supportsReadyReleaseEmission()) {
        reject(
            plan, kInvalidSyncId, SyncReadyReleaseRejection::UnsupportedTarget,
            "ReadyRelease emission is qualified only for the explicit ProtocolSync A3 target");
        return plan;
    }
    if (schedule.getPhases().empty()) {
        return plan;
    }
    if (!schedule.getFailures().empty()) {
        reject(
            plan, kInvalidSyncId, SyncReadyReleaseRejection::ScheduleFailure,
            "the immutable schedule contains unsupported or incomplete semantic facts");
        return plan;
    }
    if (hasExistingSynchronization(schedule)) {
        reject(
            plan, kInvalidSyncId, SyncReadyReleaseRejection::ExistingSynchronization,
            "Checkpoint E does not compose ReadyRelease with existing synchronization");
        return plan;
    }
    if (!schedule.getSemanticActions().empty()) {
        reject(
            plan, kInvalidSyncId, SyncReadyReleaseRejection::SemanticAction,
            "Checkpoint E does not compose ReadyRelease with ordered semantic actions");
        return plan;
    }
    const SyncRegion* loopRegion = nullptr;
    if (hasUnsupportedControl(schedule, loopRegion)) {
        reject(
            plan, kInvalidSyncId, SyncReadyReleaseRejection::UnsupportedControlFlow,
            "ReadyRelease requires one non-nested scf.for and no branch or physical-section control");
        return plan;
    }
    if (hasUnsupportedVisibility(schedule)) {
        reject(
            plan, kInvalidSyncId, SyncReadyReleaseRejection::UnsupportedVisibility,
            "ReadyRelease requires exact non-scalar GM effects on disjoint read-only and write-only families");
        return plan;
    }
    const bool hasOneCandidate = timelines.getTimelines().size() == 1 && channels.getChannels().size() == 1;
    if (!hasOneCandidate) {
        reject(
            plan, kInvalidSyncId, SyncReadyReleaseRejection::IncompleteChannelSet,
            "the initial ReadyRelease slice requires exactly one local timeline and channel");
        return plan;
    }

    const SyncGenerationTimeline& timeline = timelines.getTimelines().front();
    const SyncChannel& channel = channels.getChannels().front();
    plan.channel = channel.id;
    plan.generation = timeline.id;
    plan.capacity = channel.capacity;
    plan.slot = timeline.slot;
    const bool isReadyReleaseChannel =
        timeline.isAdmitted() && channel.isAdmitted() && channel.generation == timeline.id &&
        channel.kind == SyncChannelKind::ReadyRelease && timeline.generationKind == SyncGenerationKind::LoopIteration &&
        timeline.nextOverwrite && timeline.reuseDistance;
    if (!isReadyReleaseChannel) {
        reject(
            plan, channel.id, SyncReadyReleaseRejection::NonReadyReleaseChannel,
            "the complete local lifecycle is not an admitted recurring ready/release channel");
        return plan;
    }
    if (channel.readyOracle != SyncDemandOracleStatus::Match ||
        channel.releaseOracle != SyncDemandOracleStatus::Match) {
        reject(
            plan, channel.id, SyncReadyReleaseRejection::UnverifiedChannel,
            "the legacy demand oracle did not authenticate both ready and release relations");
        return plan;
    }
    if (plan.capacity == 0 || plan.capacity > kCheckpointEMaximumCapacity || *timeline.reuseDistance != plan.capacity) {
        reject(
            plan, channel.id, SyncReadyReleaseRejection::UnsupportedCapacity,
            "ReadyRelease requires authoritative capacity and reuse distance one or two");
        return plan;
    }
    const bool validSingleLane = plan.capacity == 1 && !timeline.slot;
    const bool validDoubleLane =
        plan.capacity == 2 && timeline.slot && timeline.slot->kind == SyncSlotExpressionKind::AffineModulo &&
        timeline.slot->selector && timeline.slot->induction && timeline.slot->loop == loopRegion->id &&
        timeline.slot->depth == 2 && timeline.slot->modulus == 2 && timeline.slot->coefficient == 1 &&
        timeline.slot->offset >= 0 && timeline.slot->offset < 2;
    if (!validSingleLane && !validDoubleLane) {
        reject(
            plan, channel.id, SyncReadyReleaseRejection::UnsupportedCapacity,
            "ReadyRelease requires an implicit single slot or exact unit-stride modulo-two selector");
        return plan;
    }
    const bool hasExactEndpointSet = timeline.producers.size() == 1 && timeline.consumers.size() == 1 &&
                                     schedule.getPhases().size() == 2 && stages.getStages().size() == 2;
    if (!hasExactEndpointSet) {
        reject(
            plan, channel.id, SyncReadyReleaseRejection::UnsupportedFunctionShape,
            "the initial ReadyRelease slice requires exactly one producer and one consumer phase");
        return plan;
    }

    const SyncPhase* producer = getOnlyPhase(schedule, stages, timeline.producers.front());
    const SyncPhase* consumer = getOnlyPhase(schedule, stages, timeline.consumers.front());
    const bool validEndpoints = producer && consumer && loopRegion && timeline.carryingRegion == loopRegion->id &&
                                producer->core != SyncPhysicalCore::Unknown && producer->core == consumer->core &&
                                producer->pipe != consumer->pipe &&
                                validateEndpointShape(*producer, *consumer, *loopRegion);
    if (!validEndpoints) {
        reject(
            plan, channel.id, SyncReadyReleaseRejection::UnsupportedFunctionShape,
            "ReadyRelease endpoints must be distinct ordered pipes in one exact loop body and physical core");
        return plan;
    }
    plan.core = producer->core;
    plan.producerPipe = producer->pipe;
    plan.consumerPipe = consumer->pipe;
    plan.producerPhase = producer->id;
    plan.consumerPhase = consumer->id;
    plan.loopOperation = loopRegion->operation;
    plan.producerOperation = producer->operation;
    plan.consumerOperation = consumer->operation;
    if (plan.capacity == 2) {
        plan.slot = findProducerSlot(schedule, timeline, *producer);
        if (!plan.slot) {
            reject(
                plan, channel.id, SyncReadyReleaseRejection::InternalInvariant,
                "the admitted depth-two timeline has no canonical producer slot selector");
            return plan;
        }
    }

    if (!target.supportsReadyRelease(plan.core, plan.producerPipe, plan.consumerPipe)) {
        reject(
            plan, channel.id, SyncReadyReleaseRejection::UnsupportedEventDirection,
            "the target does not support both ready and reverse-release event directions");
        return plan;
    }
    if (!validateTimelineFrontiers(timeline, *producer, *consumer)) {
        reject(
            plan, channel.id, SyncReadyReleaseRejection::InvalidFrontier,
            "publication, acquisition, final-use, or next-overwrite frontier is incomplete");
        return plan;
    }

    for (unsigned lane = 0; lane < plan.capacity; ++lane) {
        plan.lanes.push_back({lane, std::nullopt, std::nullopt});
    }
    if (!buildTokenCertificate(plan)) {
        reject(
            plan, channel.id, SyncReadyReleaseRejection::InvalidTokenTransfer,
            "the prime/body/drain token transfer failed its zero/one/odd/even certificate");
        return plan;
    }
    plan.status = SyncReadyReleasePlanStatus::Ready;
    if (statistics) {
        ++statistics->protocolCandidates;
        statistics->interpreterTransitions += plan.tokenCertificate.transitionApplications;
        statistics->interpreterPeakStates = std::max<std::uint64_t>(statistics->interpreterPeakStates, 1);
    }
    return plan;
}

LogicalResult mlir::pto::protocol_sync::allocateReadyReleaseProtocolEvents(
    const ProtocolSyncTarget& target, ArrayRef<SyncEventReservation> importedReservations, SyncReadyReleasePlan& plan,
    ProtocolSyncStatistics* statistics)
{
    if (plan.status != SyncReadyReleasePlanStatus::Ready) {
        return success();
    }
    const bool validAllocationShape =
        plan.lanes.size() == plan.capacity && plan.capacity != 0 && plan.capacity <= kCheckpointEMaximumCapacity;
    if (!validAllocationShape) {
        return failure();
    }
    const bool alreadyAllocated = llvm::any_of(plan.lanes, [](const SyncReadyReleaseLane& lane) {
        return lane.readyEventId.has_value() || lane.releaseEventId.has_value();
    });
    if (alreadyAllocated) {
        return failure();
    }
    const bool targetSupportsProtocol = target.isSupported() && target.supportsReadyReleaseEmission() &&
                                        target.supportsReadyRelease(plan.core, plan.producerPipe, plan.consumerPipe);
    if (!targetSupportsProtocol) {
        return failure();
    }

    llvm::DenseMap<std::uint32_t, llvm::SmallVector<unsigned, 8>> reservations;
    for (const SyncEventReservation& reservation : importedReservations) {
        auto& ids = reservations[reservationDomain(reservation.source, reservation.target)];
        ids.append(reservation.eventIds.begin(), reservation.eventIds.end());
    }

    const auto allocateDomain = [&](PIPE source, PIPE targetPipe, llvm::SmallVectorImpl<unsigned>& allocated) {
        const ArrayRef<unsigned> reserved = reservations[reservationDomain(source, targetPipe)];
        for (unsigned lane = 0; lane < plan.capacity; ++lane) {
            auto selected = llvm::find_if(target.getCompilerEventIds(), [&](unsigned eventId) {
                return !llvm::is_contained(reserved, eventId) && !llvm::is_contained(allocated, eventId);
            });
            if (selected == target.getCompilerEventIds().end()) {
                return false;
            }
            allocated.push_back(*selected);
        }
        return true;
    };

    llvm::SmallVector<unsigned, 2> readyIds;
    llvm::SmallVector<unsigned, 2> releaseIds;
    const bool allocatedReady = allocateDomain(plan.producerPipe, plan.consumerPipe, readyIds);
    const bool allocatedRelease = allocateDomain(plan.consumerPipe, plan.producerPipe, releaseIds);
    if (!allocatedReady || !allocatedRelease) {
        reject(
            plan, plan.channel, SyncReadyReleaseRejection::EventCapacity,
            "the complete ReadyRelease candidate does not fit the compiler event pool");
        return success();
    }
    for (unsigned lane = 0; lane < plan.capacity; ++lane) {
        plan.lanes[lane].readyEventId = readyIds[lane];
        plan.lanes[lane].releaseEventId = releaseIds[lane];
    }
    if (statistics) {
        ++statistics->selectedReadyReleaseProtocols;
        statistics->selectedReadyReleaseLanes += plan.lanes.size();
        statistics->logicalActions += 4 + 2 * plan.lanes.size();
        statistics->allocationGraphVertices += 2 * plan.capacity;
        statistics->allocationGraphEdges += plan.capacity > 0 ? 2 * (plan.capacity - 1) : 0;
        statistics->eventDomains += eventDomain(plan.core, plan.producerPipe, plan.consumerPipe) ==
                                            eventDomain(plan.core, plan.consumerPipe, plan.producerPipe) ?
                                        1 :
                                        2;
        statistics->maxEventDomainPressure = std::max<std::uint64_t>(statistics->maxEventDomainPressure, plan.capacity);
        for (const SyncReadyReleaseLane& lane : plan.lanes) {
            statistics->maximumEventIdPlusOne = std::max<std::uint64_t>(
                statistics->maximumEventIdPlusOne, std::max(*lane.readyEventId, *lane.releaseEventId) + 1);
        }
    }
    return success();
}

LogicalResult mlir::pto::protocol_sync::allocateReadyReleaseProtocolEvents(
    const StructuredSyncIR& schedule, SyncReadyReleasePlan& plan, ProtocolSyncStatistics* statistics)
{
    llvm::SmallVector<SyncEventReservation, 4> importedReservations;
    for (const SyncOpSummary& summary : schedule.getSummaries()) {
        importedReservations.append(summary.eventReservations.begin(), summary.eventReservations.end());
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    return allocateReadyReleaseProtocolEvents(target, importedReservations, plan, statistics);
}

StringRef mlir::pto::protocol_sync::stringifySyncReadyReleasePlanStatus(SyncReadyReleasePlanStatus status)
{
    switch (status) {
        case SyncReadyReleasePlanStatus::Empty:
            return "empty";
        case SyncReadyReleasePlanStatus::Ready:
            return "ready";
        case SyncReadyReleasePlanStatus::Unsupported:
            return "unsupported";
    }
    return "unsupported";
}

StringRef mlir::pto::protocol_sync::stringifySyncReadyReleaseRejection(SyncReadyReleaseRejection rejection)
{
    switch (rejection) {
        case SyncReadyReleaseRejection::None:
            return "none";
        case SyncReadyReleaseRejection::UnsupportedTarget:
            return "unsupported-target";
        case SyncReadyReleaseRejection::ExistingSynchronization:
            return "existing-synchronization";
        case SyncReadyReleaseRejection::ScheduleFailure:
            return "schedule-failure";
        case SyncReadyReleaseRejection::SemanticAction:
            return "semantic-action";
        case SyncReadyReleaseRejection::UnsupportedControlFlow:
            return "unsupported-control-flow";
        case SyncReadyReleaseRejection::UnsupportedFunctionShape:
            return "unsupported-function-shape";
        case SyncReadyReleaseRejection::IncompleteChannelSet:
            return "incomplete-channel-set";
        case SyncReadyReleaseRejection::NonReadyReleaseChannel:
            return "non-ready-release-channel";
        case SyncReadyReleaseRejection::UnverifiedChannel:
            return "unverified-channel";
        case SyncReadyReleaseRejection::UnsupportedCapacity:
            return "unsupported-capacity";
        case SyncReadyReleaseRejection::UnsupportedEventDirection:
            return "unsupported-event-direction";
        case SyncReadyReleaseRejection::UnsupportedVisibility:
            return "unsupported-visibility";
        case SyncReadyReleaseRejection::InvalidFrontier:
            return "invalid-frontier";
        case SyncReadyReleaseRejection::InvalidTokenTransfer:
            return "invalid-token-transfer";
        case SyncReadyReleaseRejection::EventCapacity:
            return "event-capacity";
        case SyncReadyReleaseRejection::InternalInvariant:
            return "internal-invariant";
    }
    return "internal-invariant";
}

void mlir::pto::protocol_sync::printReadyReleaseProtocolPlan(
    func::FuncOp function, const SyncReadyReleasePlan& plan, raw_ostream& output)
{
    output << "PROTOCOL-SYNC ready-release-plan function=@" << function.getSymName()
           << " status=" << stringifySyncReadyReleasePlanStatus(plan.status) << " channel=";
    if (plan.channel == kInvalidSyncId) {
        output << "none";
    } else {
        output << '#' << plan.channel;
    }
    output << " capacity=" << plan.capacity << " core=" << stringifySyncPhysicalCore(plan.core)
           << " producer-pipe=" << static_cast<unsigned>(plan.producerPipe)
           << " consumer-pipe=" << static_cast<unsigned>(plan.consumerPipe) << '\n';
    for (const SyncReadyReleaseLane& lane : plan.lanes) {
        output << "  lane #" << lane.logicalLane << " ready-event-id=";
        if (lane.readyEventId) {
            output << *lane.readyEventId;
        } else {
            output << "none";
        }
        output << " release-event-id=";
        if (lane.releaseEventId) {
            output << *lane.releaseEventId;
        } else {
            output << "none";
        }
        output << '\n';
    }
    const SyncReadyReleaseTokenCertificate& certificate = plan.tokenCertificate;
    output << "  token-certificate zero-trip=" << (certificate.zeroTripSafe ? "safe" : "unsafe")
           << " one-trip=" << (certificate.oneTripSafe ? "safe" : "unsafe")
           << " odd-even=" << (certificate.oddEvenSafe ? "safe" : "unsafe")
           << " steady-state=" << (certificate.steadyStateStable ? "stable" : "unstable") << " slots=[";
    llvm::interleaveComma(certificate.slotWitness, output);
    output << "] transition-applications=" << certificate.transitionApplications << '\n';
    for (const SyncReadyReleasePlanRejection& rejection : plan.rejections) {
        output << "  rejection channel=";
        if (rejection.channel == kInvalidSyncId) {
            output << "none";
        } else {
            output << '#' << rejection.channel;
        }
        output << " reason=" << stringifySyncReadyReleaseRejection(rejection.reason) << " detail=\"" << rejection.detail
               << "\"\n";
    }
}
