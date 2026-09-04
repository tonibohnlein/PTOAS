// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- MixedProtocolPlan.cpp - Select complete Checkpoint-F worlds -----===//

#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

bool hasInternalRejection(const SyncOneShotPlan& plan)
{
    return llvm::any_of(plan.rejections, [](const SyncOneShotPlanRejection& rejection) {
        return rejection.reason == SyncOneShotRejection::InternalInvariant;
    });
}

bool hasInternalRejection(const SyncReadyReleasePlan& plan)
{
    return llvm::any_of(plan.rejections, [](const SyncReadyReleasePlanRejection& rejection) {
        return rejection.reason == SyncReadyReleaseRejection::InternalInvariant;
    });
}

bool hasInternalRejection(const SyncDirectRepairPlan& plan)
{
    return llvm::any_of(plan.rejections, [](const SyncDirectRepairPlanRejection& rejection) {
        return rejection.reason == SyncDirectRepairRejection::InternalInvariant;
    });
}

FailureOr<SyncSelectedWorld> buildProtocolWorld(
    const SyncMixedProtocolPlan& plan, const ChannelAnalysisResult& channels, bool includeProtocol = true)
{
    if (!includeProtocol || !plan.hasProtocol()) {
        return SyncSelectedWorld{};
    }
    if (plan.oneShot && plan.readyRelease) {
        return failure();
    }
    if (plan.oneShot) {
        return buildSelectedWorld(*plan.oneShot, channels);
    }
    return buildSelectedWorld(*plan.readyRelease);
}

SmallVector<SyncDirectCandidateId, 8> allDirectCandidates(const SyncDirectRepairPlan& direct)
{
    SmallVector<SyncDirectCandidateId, 8> selected;
    for (const SyncDirectRepairCandidate& candidate : direct.candidates) {
        selected.push_back(candidate.id);
    }
    return selected;
}

FailureOr<SyncSelectedWorld> buildCompleteWorld(
    const SyncMixedProtocolPlan& plan, const ChannelAnalysisResult& channels,
    ArrayRef<SyncDirectCandidateId> selectedDirect, bool includeProtocol = true)
{
    FailureOr<SyncSelectedWorld> world = buildProtocolWorld(plan, channels, includeProtocol);
    const LogicalResult applied =
        failed(world) ? failure() :
                        applyDirectRepairCandidates(plan.directRepair, plan.directObligations, selectedDirect, *world);
    if (failed(applied)) {
        return failure();
    }
    return world;
}

LogicalResult pruneDirectCandidates(SyncMixedProtocolPlan& plan, ArrayRef<SyncDirectCandidateId> selected)
{
    const bool retainsEveryCandidate = selected.size() == plan.directRepair.candidates.size();
    if (retainsEveryCandidate) {
        return success();
    }

    llvm::BitVector selectedBits(plan.directRepair.candidates.size());
    llvm::BitVector retainedObligations(plan.directObligations.size());
    for (SyncDirectCandidateId id : selected) {
        const bool validSelection =
            id < selectedBits.size() && !selectedBits.test(id) && plan.directRepair.candidates[id].id == id;
        if (!validSelection) {
            return failure();
        }
        selectedBits.set(id);
        for (SyncObligationId obligation : plan.directRepair.candidates[id].obligations) {
            if (obligation >= retainedObligations.size()) {
                return failure();
            }
            retainedObligations.set(obligation);
        }
    }

    SmallVector<SyncResidualObligation, 16> obligations;
    llvm::DenseMap<SyncObligationId, SyncObligationId> remappedObligations;
    for (int index = retainedObligations.find_first(); index >= 0; index = retainedObligations.find_next(index)) {
        const SyncObligationId oldId = static_cast<SyncObligationId>(index);
        SyncResidualObligation obligation = plan.directObligations[oldId];
        obligation.id = obligations.size();
        remappedObligations[oldId] = obligation.id;
        obligations.push_back(std::move(obligation));
    }

    SyncDirectRepairPlan direct;
    direct.obligationCount = obligations.size();
    for (SyncDirectCandidateId oldId : selected) {
        SyncDirectRepairCandidate candidate = plan.directRepair.candidates[oldId];
        candidate.id = direct.candidates.size();
        for (SyncObligationId& obligation : candidate.obligations) {
            auto remapped = remappedObligations.find(obligation);
            if (remapped == remappedObligations.end()) {
                return failure();
            }
            obligation = remapped->second;
        }
        direct.candidates.push_back(std::move(candidate));
    }
    direct.status = direct.candidates.empty() ? SyncDirectRepairPlanStatus::Empty : SyncDirectRepairPlanStatus::Ready;
    plan.directObligations = std::move(obligations);
    plan.directRepair = std::move(direct);
    return success();
}

LogicalResult reverseDeleteCandidates(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels, SyncMixedProtocolPlan& plan,
    ProtocolSyncStatistics* statistics)
{
    SmallVector<SyncDirectCandidateId, 8> selected = allDirectCandidates(plan.directRepair);
    for (std::size_t reverseIndex = selected.size(); reverseIndex > 0; --reverseIndex) {
        SmallVector<SyncDirectCandidateId, 8> trial(selected);
        trial.erase(trial.begin() + reverseIndex - 1);
        ++plan.reverseDeletionAttempts;
        FailureOr<SyncSelectedWorld> world = buildCompleteWorld(plan, channels, trial);
        if (failed(world)) {
            return failure();
        }
        FailureOr<SyncInterpretationResult> result =
            interpretSelectedWorld(schedule, stages, timelines, channels, *world, statistics);
        if (failed(result)) {
            return failure();
        }
        if (result->isComplete()) {
            selected = std::move(trial);
            ++plan.reverseDeletionRemoved;
        }
    }

    if (plan.hasProtocol()) {
        ++plan.reverseDeletionAttempts;
        FailureOr<SyncSelectedWorld> world = buildCompleteWorld(plan, channels, selected, false);
        if (failed(world)) {
            return failure();
        }
        FailureOr<SyncInterpretationResult> result =
            interpretSelectedWorld(schedule, stages, timelines, channels, *world, statistics);
        if (failed(result)) {
            return failure();
        }
        if (result->isComplete()) {
            plan.oneShot.reset();
            plan.readyRelease.reset();
            ++plan.reverseDeletionRemoved;
        }
    }

    return pruneDirectCandidates(plan, selected);
}

void clearEventIds(SyncMixedProtocolPlan& plan)
{
    if (plan.oneShot) {
        for (SyncOneShotProtocol& protocol : plan.oneShot->protocols) {
            protocol.eventId.reset();
        }
    }
    if (plan.readyRelease) {
        for (SyncReadyReleaseLane& lane : plan.readyRelease->lanes) {
            lane.readyEventId.reset();
            lane.releaseEventId.reset();
        }
    }
    for (SyncDirectRepairCandidate& candidate : plan.directRepair.candidates) {
        candidate.eventId.reset();
    }
}

bool hasAllocatedEvent(const SyncMixedProtocolPlan& plan)
{
    const bool oneShotAllocated = plan.oneShot && llvm::any_of(plan.oneShot->protocols, [](const auto& protocol) {
                                      return protocol.eventId.has_value();
                                  });
    const bool readyReleaseAllocated =
        plan.readyRelease && llvm::any_of(plan.readyRelease->lanes, [](const auto& lane) {
            return lane.readyEventId.has_value() || lane.releaseEventId.has_value();
        });
    const bool directAllocated =
        llvm::any_of(plan.directRepair.candidates, [](const auto& candidate) { return candidate.eventId.has_value(); });
    return oneShotAllocated || readyReleaseAllocated || directAllocated;
}

void appendReservation(SmallVectorImpl<SyncEventReservation>& reservations, PIPE source, PIPE target, unsigned eventId)
{
    auto found = llvm::find_if(reservations, [&](const SyncEventReservation& reservation) {
        return reservation.source == source && reservation.target == target;
    });
    if (found == reservations.end()) {
        reservations.push_back({source, target, {eventId}});
    } else if (!llvm::is_contained(found->eventIds, eventId)) {
        found->eventIds.push_back(eventId);
    }
}

void appendProtocolReservations(const SyncMixedProtocolPlan& plan, SmallVectorImpl<SyncEventReservation>& reservations)
{
    if (plan.oneShot) {
        for (const SyncOneShotProtocol& protocol : plan.oneShot->protocols) {
            if (protocol.kind == SyncOneShotProtocolKind::DirectedEvent && protocol.eventId) {
                appendReservation(reservations, protocol.sourcePipe, protocol.targetPipe, *protocol.eventId);
            }
        }
    }
    if (plan.readyRelease) {
        for (const SyncReadyReleaseLane& lane : plan.readyRelease->lanes) {
            appendReservation(
                reservations, plan.readyRelease->producerPipe, plan.readyRelease->consumerPipe, *lane.readyEventId);
            appendReservation(
                reservations, plan.readyRelease->consumerPipe, plan.readyRelease->producerPipe, *lane.releaseEventId);
        }
    }
}

using EventDomain = std::tuple<std::uint8_t, std::uint8_t, std::uint8_t>;

LogicalResult recordEvent(
    std::map<EventDomain, std::set<unsigned>>& events, SyncPhysicalCore core, PIPE source, PIPE destination,
    unsigned eventId, const ProtocolSyncTarget& target)
{
    if (!llvm::is_contained(target.getCompilerEventIds(), eventId)) {
        return failure();
    }
    auto& domain = events[{
        static_cast<std::uint8_t>(core), static_cast<std::uint8_t>(source), static_cast<std::uint8_t>(destination)}];
    return success(domain.insert(eventId).second);
}

LogicalResult collectAllocatedEvents(
    const SyncMixedProtocolPlan& plan, const ProtocolSyncTarget& target,
    std::map<EventDomain, std::set<unsigned>>& events)
{
    if (plan.oneShot) {
        for (const SyncOneShotProtocol& protocol : plan.oneShot->protocols) {
            if (protocol.kind == SyncOneShotProtocolKind::DirectedEvent &&
                (!protocol.eventId ||
                 failed(recordEvent(
                     events, protocol.core, protocol.sourcePipe, protocol.targetPipe, *protocol.eventId, target)))) {
                return failure();
            }
        }
    }
    if (plan.readyRelease) {
        for (const SyncReadyReleaseLane& lane : plan.readyRelease->lanes) {
            if (!lane.readyEventId || !lane.releaseEventId ||
                failed(recordEvent(
                    events, plan.readyRelease->core, plan.readyRelease->producerPipe, plan.readyRelease->consumerPipe,
                    *lane.readyEventId, target)) ||
                failed(recordEvent(
                    events, plan.readyRelease->core, plan.readyRelease->consumerPipe, plan.readyRelease->producerPipe,
                    *lane.releaseEventId, target))) {
                return failure();
            }
        }
    }
    for (const SyncDirectRepairCandidate& candidate : plan.directRepair.candidates) {
        if (candidate.kind == SyncDirectRepairKind::DirectedEvent &&
            (!candidate.eventId ||
             failed(recordEvent(
                 events, candidate.core, candidate.sourcePipe, candidate.targetPipe, *candidate.eventId, target)))) {
            return failure();
        }
    }
    return success();
}

bool hasCapacityRejection(const SyncOneShotPlan& plan)
{
    return llvm::any_of(plan.rejections, [](const SyncOneShotPlanRejection& rejection) {
        return rejection.reason == SyncOneShotRejection::EventCapacity;
    });
}

bool hasCapacityRejection(const SyncReadyReleasePlan& plan)
{
    return llvm::any_of(plan.rejections, [](const SyncReadyReleasePlanRejection& rejection) {
        return rejection.reason == SyncReadyReleaseRejection::EventCapacity;
    });
}

void recordSelectedStatistics(
    const SyncMixedProtocolPlan& plan, const std::map<EventDomain, std::set<unsigned>>& events,
    ProtocolSyncStatistics& statistics)
{
    if (plan.oneShot) {
        statistics.selectedOneShotProtocols += plan.oneShot->protocols.size();
        for (const SyncOneShotProtocol& protocol : plan.oneShot->protocols) {
            if (protocol.kind == SyncOneShotProtocolKind::PipeBarrier) {
                ++statistics.selectedSamePipeBarriers;
                ++statistics.logicalActions;
            } else if (protocol.kind == SyncOneShotProtocolKind::DirectedEvent) {
                ++statistics.selectedDirectedEventPairs;
                statistics.logicalActions += 2;
            }
        }
        ++statistics.selectedTailDrains;
        ++statistics.logicalActions;
    }
    if (plan.readyRelease) {
        ++statistics.selectedReadyReleaseProtocols;
        statistics.selectedReadyReleaseLanes += plan.readyRelease->lanes.size();
        statistics.logicalActions += 4 + 2 * plan.readyRelease->lanes.size();
    }
    statistics.selectedDirectRepairs += plan.directRepair.candidates.size();
    for (const SyncDirectRepairCandidate& candidate : plan.directRepair.candidates) {
        statistics.logicalActions += candidate.kind == SyncDirectRepairKind::DirectedEvent ? 2 : 1;
    }
    statistics.eventDomains += events.size();
    for (const auto& [domain, eventIds] : events) {
        (void)domain;
        statistics.allocationGraphVertices += eventIds.size();
        statistics.allocationGraphEdges += eventIds.size() * (eventIds.size() - 1) / 2;
        statistics.maxEventDomainPressure = std::max<std::uint64_t>(statistics.maxEventDomainPressure, eventIds.size());
        if (!eventIds.empty()) {
            statistics.maximumEventIdPlusOne =
                std::max<std::uint64_t>(statistics.maximumEventIdPlusOne, *eventIds.rbegin() + 1);
        }
    }
}

} // namespace

FailureOr<SyncMixedProtocolPlan> mlir::pto::protocol_sync::buildMixedProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels, bool enableProtocols,
    ProtocolSyncStatistics* statistics)
{
    if (!schedule.isFrozen()) {
        return failure();
    }
    SyncMixedProtocolPlan plan;
    plan.protocolsEnabled = enableProtocols;
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.isSupported()) {
        plan.status = SyncMixedPlanStatus::Unsupported;
        plan.failures.push_back({SyncMixedPlanRejection::UnsupportedTarget, target.getUnsupportedReason().str()});
        return plan;
    }

    if (enableProtocols) {
        FailureOr<SyncReadyReleasePlan> readyRelease = buildReadyReleaseProtocolPlan(
            schedule, stages, timelines, channels, nullptr, SyncReadyReleasePlanningScope::MixedCandidate);
        const bool readyReleaseFailed = failed(readyRelease) || hasInternalRejection(*readyRelease);
        if (readyReleaseFailed) {
            return failure();
        }
        if (readyRelease->status == SyncReadyReleasePlanStatus::Ready) {
            plan.readyRelease = std::move(*readyRelease);
            if (statistics) {
                ++statistics->protocolCandidates;
                statistics->tokenCertificateTransitions += plan.readyRelease->tokenCertificate.transitionApplications;
            }
        } else {
            FailureOr<SyncOneShotPlan> oneShot =
                buildOneShotProtocolPlan(schedule, stages, timelines, channels, nullptr);
            const bool oneShotFailed = failed(oneShot) || hasInternalRejection(*oneShot);
            if (oneShotFailed) {
                return failure();
            }
            if (oneShot->status == SyncOneShotPlanStatus::Ready) {
                plan.oneShot = std::move(*oneShot);
                if (statistics) {
                    statistics->protocolCandidates += plan.oneShot->protocols.size();
                }
            }
        }
    }

    FailureOr<SyncSelectedWorld> protocolWorld = buildProtocolWorld(plan, channels);
    if (failed(protocolWorld)) {
        return failure();
    }
    FailureOr<SyncInterpretationResult> residuals =
        interpretSelectedWorld(schedule, stages, timelines, channels, *protocolWorld, statistics);
    if (failed(residuals)) {
        return failure();
    }
    plan.initialResidualCount = residuals->obligations.size();
    plan.directObligations = residuals->obligations;
    FailureOr<SyncDirectRepairPlan> direct =
        buildDirectRepairPlan(schedule, stages, plan.directObligations, statistics);
    const LogicalResult directVerified =
        failed(direct) ? failure() :
                         verifyDirectRepairPlan(schedule, stages, plan.directObligations, *direct, statistics);
    const bool directPlanValid = succeeded(directVerified) && !hasInternalRejection(*direct);
    if (!directPlanValid) {
        return failure();
    }
    plan.directRepair = std::move(*direct);
    if (!plan.directRepair.isComplete()) {
        plan.status = SyncMixedPlanStatus::Unsupported;
        plan.failures.push_back(
            {SyncMixedPlanRejection::IncompleteDirectRepair,
             "selected protocols leave residual obligations outside the direct-repair subset"});
        if (statistics) {
            statistics->mixedSelectionCandidates += plan.hasProtocol() ? 1 : 0;
            statistics->mixedSelectionCandidates += plan.directRepair.candidates.size();
        }
        return plan;
    }

    if (failed(selectMixedProtocolCandidates(schedule, stages, timelines, channels, plan, statistics))) {
        return failure();
    }
    return plan;
}

LogicalResult mlir::pto::protocol_sync::selectMixedProtocolCandidates(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels, SyncMixedProtocolPlan& plan,
    ProtocolSyncStatistics* statistics)
{
    const bool validCandidatePool = schedule.isFrozen() && plan.directRepair.isComplete() &&
                                    !(plan.oneShot && plan.readyRelease) && plan.failures.empty() &&
                                    !hasAllocatedEvent(plan);
    if (!validCandidatePool) {
        return failure();
    }
    plan.candidateCountBeforeDeletion = plan.directRepair.candidates.size() + (plan.hasProtocol() ? 1 : 0);
    plan.reverseDeletionAttempts = 0;
    plan.reverseDeletionRemoved = 0;
    if (failed(reverseDeleteCandidates(schedule, stages, timelines, channels, plan, statistics))) {
        return failure();
    }
    SmallVector<SyncDirectCandidateId, 8> selectedDirect = allDirectCandidates(plan.directRepair);
    FailureOr<SyncSelectedWorld> selectedWorld = buildCompleteWorld(plan, channels, selectedDirect);
    if (failed(selectedWorld)) {
        return failure();
    }
    FailureOr<SyncInterpretationResult> final =
        interpretSelectedWorld(schedule, stages, timelines, channels, *selectedWorld, statistics);
    const bool finalWorldComplete = succeeded(final) && final->isComplete();
    if (!finalWorldComplete) {
        return failure();
    }
    plan.selectedWorld = std::move(*selectedWorld);
    plan.status = plan.hasProtocol() || !plan.directRepair.candidates.empty() ? SyncMixedPlanStatus::Ready :
                                                                                SyncMixedPlanStatus::Empty;
    if (statistics) {
        statistics->mixedSelectionCandidates += plan.candidateCountBeforeDeletion;
        statistics->reverseDeletionAttempts += plan.reverseDeletionAttempts;
        statistics->reverseDeletionRemoved += plan.reverseDeletionRemoved;
    }
    return success();
}

LogicalResult mlir::pto::protocol_sync::allocateMixedProtocolEvents(
    const StructuredSyncIR& schedule, SyncMixedProtocolPlan& plan, ProtocolSyncStatistics* statistics)
{
    const bool canAllocate = schedule.isFrozen() && plan.isComplete() && !hasAllocatedEvent(plan);
    if (!canAllocate) {
        return failure();
    }
    if (plan.status == SyncMixedPlanStatus::Empty) {
        return success();
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.isSupported()) {
        return failure();
    }

    SyncMixedProtocolPlan allocated = plan;
    if (allocated.oneShot) {
        const LogicalResult allocatedOneShot = allocateOneShotProtocolEvents(schedule, *allocated.oneShot);
        const bool oneShotReady =
            succeeded(allocatedOneShot) && allocated.oneShot->status == SyncOneShotPlanStatus::Ready;
        if (!oneShotReady) {
            const bool capacity = hasCapacityRejection(*allocated.oneShot);
            clearEventIds(plan);
            if (!capacity) {
                return failure();
            }
            plan.status = SyncMixedPlanStatus::ResourceInfeasible;
            plan.failures.push_back(
                {SyncMixedPlanRejection::EventCapacity, "one-shot protocol exhausted a selected event domain"});
            return success();
        }
    }
    if (allocated.readyRelease) {
        const LogicalResult allocatedReadyRelease =
            allocateReadyReleaseProtocolEvents(schedule, *allocated.readyRelease);
        const bool readyReleaseReady =
            succeeded(allocatedReadyRelease) && allocated.readyRelease->status == SyncReadyReleasePlanStatus::Ready;
        if (!readyReleaseReady) {
            const bool capacity = hasCapacityRejection(*allocated.readyRelease);
            clearEventIds(plan);
            if (!capacity) {
                return failure();
            }
            plan.status = SyncMixedPlanStatus::ResourceInfeasible;
            plan.failures.push_back(
                {SyncMixedPlanRejection::EventCapacity, "ReadyRelease protocol exhausted a selected event domain"});
            return success();
        }
    }

    SmallVector<SyncEventReservation, 8> reservations;
    for (const SyncOpSummary& summary : schedule.getSummaries()) {
        reservations.append(summary.eventReservations.begin(), summary.eventReservations.end());
    }
    appendProtocolReservations(allocated, reservations);
    if (allocated.directRepair.status == SyncDirectRepairPlanStatus::Ready) {
        if (failed(allocateDirectRepairEvents(target, reservations, allocated.directRepair))) {
            return failure();
        }
        if (allocated.directRepair.status == SyncDirectRepairPlanStatus::ResourceInfeasible) {
            clearEventIds(plan);
            plan.status = SyncMixedPlanStatus::ResourceInfeasible;
            plan.failures.push_back(
                {SyncMixedPlanRejection::EventCapacity, "direct repair exhausted a selected event domain"});
            return success();
        }
    }

    std::map<EventDomain, std::set<unsigned>> events;
    if (failed(collectAllocatedEvents(allocated, target, events))) {
        return failure();
    }
    plan = std::move(allocated);
    if (statistics) {
        recordSelectedStatistics(plan, events, *statistics);
    }
    return success();
}

StringRef mlir::pto::protocol_sync::stringifySyncMixedPlanStatus(SyncMixedPlanStatus status)
{
    switch (status) {
        case SyncMixedPlanStatus::Empty:
            return "empty";
        case SyncMixedPlanStatus::Ready:
            return "ready";
        case SyncMixedPlanStatus::Unsupported:
            return "unsupported";
        case SyncMixedPlanStatus::ResourceInfeasible:
            return "resource-infeasible";
    }
    return "unsupported";
}

StringRef mlir::pto::protocol_sync::stringifySyncMixedPlanRejection(SyncMixedPlanRejection rejection)
{
    switch (rejection) {
        case SyncMixedPlanRejection::None:
            return "none";
        case SyncMixedPlanRejection::UnsupportedTarget:
            return "unsupported-target";
        case SyncMixedPlanRejection::IncompleteDirectRepair:
            return "incomplete-direct-repair";
        case SyncMixedPlanRejection::EventCapacity:
            return "event-capacity";
        case SyncMixedPlanRejection::InternalInvariant:
            return "internal-invariant";
    }
    return "internal-invariant";
}

void mlir::pto::protocol_sync::printMixedProtocolPlan(
    func::FuncOp function, const SyncMixedProtocolPlan& plan, llvm::raw_ostream& output)
{
    StringRef protocol = plan.readyRelease ? "ready-release" : plan.oneShot ? "one-shot" : "none";
    output << "PROTOCOL-SYNC mixed-plan function=@" << function.getSymName()
           << " status=" << stringifySyncMixedPlanStatus(plan.status) << " protocol=" << protocol
           << " initial-residuals=" << plan.initialResidualCount
           << " direct-candidates=" << plan.directRepair.candidates.size()
           << " candidates-before-deletion=" << plan.candidateCountBeforeDeletion
           << " reverse-attempts=" << plan.reverseDeletionAttempts << " reverse-removed=" << plan.reverseDeletionRemoved
           << '\n';
    for (const SyncMixedPlanFailure& failure : plan.failures) {
        output << "  reason=" << stringifySyncMixedPlanRejection(failure.reason) << " detail=\"" << failure.detail
               << "\"\n";
    }
    if (plan.oneShot) {
        printOneShotProtocolPlan(function, *plan.oneShot, output);
    }
    if (plan.readyRelease) {
        printReadyReleaseProtocolPlan(function, *plan.readyRelease, output);
    }
    if (plan.directRepair.status != SyncDirectRepairPlanStatus::Empty) {
        printDirectRepairPlan(function, plan.directRepair, output);
    }
}
