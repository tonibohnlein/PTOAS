// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- MixedProtocolVerifier.cpp - Verify complete mixed worlds --------===//

#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"

#include "PTO/IR/PTO.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <chrono>
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using MixedVerifierClock = std::chrono::steady_clock;
constexpr StringLiteral kGeneratedAttr = "pto.protocol_sync.generated";
constexpr StringLiteral kProtocolAttr = "pto.protocol_sync.protocol_id";
constexpr StringLiteral kProtocolKindAttr = "pto.protocol_sync.protocol_kind";
constexpr StringLiteral kDirectCandidateAttr = "pto.protocol_sync.direct_candidate_id";
constexpr StringLiteral kOneShotKind = "one-shot";
constexpr StringLiteral kReadyReleaseKind = "ready-release";

std::uint64_t elapsedMicroseconds(MixedVerifierClock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(MixedVerifierClock::now() - start).count();
}

bool sameIteration(const SyncIterationRelation& first, const SyncIterationRelation& second)
{
    return first.kind == second.kind && first.distance == second.distance && first.carrier == second.carrier;
}

bool sameProtocol(const SyncSelectedProtocol& first, const SyncSelectedProtocol& second)
{
    return first.kind == second.kind && first.channel == second.channel && first.generation == second.generation &&
           first.capacity == second.capacity;
}

bool sameCompletion(const SyncSelectedCompletion& first, const SyncSelectedCompletion& second)
{
    return first.source == second.source && first.target == second.target && first.control == second.control &&
           sameIteration(first.iteration, second.iteration);
}

bool sameVisibility(const SyncSelectedVisibility& first, const SyncSelectedVisibility& second)
{
    return first.source == second.source && first.target == second.target && first.control == second.control &&
           sameIteration(first.iteration, second.iteration);
}

bool sameWorld(const SyncSelectedWorld& first, const SyncSelectedWorld& second)
{
    return llvm::equal(first.protocols, second.protocols, sameProtocol) &&
           llvm::equal(first.completions, second.completions, sameCompletion) &&
           llvm::equal(first.visibility, second.visibility, sameVisibility) &&
           first.exitCompletedPhases == second.exitCompletedPhases;
}

using ObligationKey = std::tuple<
    SyncObligationKind, SyncPhaseId, SyncPhaseId, std::optional<SyncGenerationId>, std::optional<SyncChannelId>,
    SyncControlRelation, SyncIterationRelationKind, unsigned, SyncRegionId, std::string>;

ObligationKey obligationKey(const SyncResidualObligation& obligation)
{
    return {
        obligation.kind,
        obligation.source,
        obligation.target,
        obligation.generation,
        obligation.channel,
        obligation.control,
        obligation.iteration.kind,
        obligation.iteration.distance,
        obligation.iteration.carrier,
        obligation.detail};
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
    return plan.oneShot ? buildSelectedWorld(*plan.oneShot, channels) : buildSelectedWorld(*plan.readyRelease);
}

SmallVector<SyncDirectCandidateId, 8> allDirectCandidates(const SyncMixedProtocolPlan& plan)
{
    SmallVector<SyncDirectCandidateId, 8> selected;
    for (const SyncDirectRepairCandidate& candidate : plan.directRepair.candidates) {
        selected.push_back(candidate.id);
    }
    return selected;
}

FailureOr<SyncSelectedWorld> buildSelectedWorldFromPlan(
    const SyncMixedProtocolPlan& plan, const ChannelAnalysisResult& channels,
    ArrayRef<SyncDirectCandidateId> directCandidates, bool includeProtocol = true)
{
    FailureOr<SyncSelectedWorld> world = buildProtocolWorld(plan, channels, includeProtocol);
    const LogicalResult applied =
        failed(world) ?
            failure() :
            applyDirectRepairCandidates(plan.directRepair, plan.directObligations, directCandidates, *world);
    if (failed(applied)) {
        return failure();
    }
    return world;
}

bool hasOnlyFailure(const SyncMixedProtocolPlan& plan, SyncMixedPlanRejection rejection)
{
    return plan.failures.size() == 1 && plan.failures.front().reason == rejection &&
           !plan.failures.front().detail.empty();
}

LogicalResult verifyRetainedResiduals(
    ArrayRef<SyncResidualObligation> initial, ArrayRef<SyncResidualObligation> retained)
{
    std::map<ObligationKey, unsigned> unmatched;
    for (const SyncResidualObligation& obligation : initial) {
        ++unmatched[obligationKey(obligation)];
    }
    for (auto [index, obligation] : llvm::enumerate(retained)) {
        if (obligation.id != index) {
            return failure();
        }
        auto found = unmatched.find(obligationKey(obligation));
        if (found == unmatched.end()) {
            return failure();
        }
        if (--found->second == 0) {
            unmatched.erase(found);
        }
    }
    return success();
}

LogicalResult verifyReverseMinimality(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncMixedProtocolPlan& plan, ProtocolSyncStatistics* statistics)
{
    SmallVector<SyncDirectCandidateId, 8> selected = allDirectCandidates(plan);
    for (std::size_t index = 0; index < selected.size(); ++index) {
        SmallVector<SyncDirectCandidateId, 8> trial(selected);
        trial.erase(trial.begin() + index);
        FailureOr<SyncSelectedWorld> world = buildSelectedWorldFromPlan(plan, channels, trial);
        if (failed(world)) {
            return failure();
        }
        FailureOr<SyncInterpretationResult> result =
            interpretSelectedWorld(schedule, stages, timelines, channels, *world, statistics);
        const bool candidateIsEssential = succeeded(result) && !result->isComplete();
        if (!candidateIsEssential) {
            return failure();
        }
    }
    if (plan.hasProtocol()) {
        FailureOr<SyncSelectedWorld> world = buildSelectedWorldFromPlan(plan, channels, selected, false);
        if (failed(world)) {
            return failure();
        }
        FailureOr<SyncInterpretationResult> result =
            interpretSelectedWorld(schedule, stages, timelines, channels, *world, statistics);
        const bool protocolIsEssential = succeeded(result) && !result->isComplete();
        if (!protocolIsEssential) {
            return failure();
        }
    }
    return success();
}

std::uint64_t eventKey(SyncPhysicalCore core, PIPE source, PIPE target, unsigned eventId)
{
    return (static_cast<std::uint64_t>(core) << 24) | (static_cast<std::uint64_t>(source) << 16) |
           (static_cast<std::uint64_t>(target) << 8) | eventId;
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

LogicalResult appendAllocatedEvent(
    const StructuredSyncIR& schedule, const ProtocolSyncTarget& target, llvm::DenseSet<std::uint64_t>& events,
    SyncPhysicalCore core, PIPE source, PIPE destination, std::optional<unsigned> eventId,
    unsigned& eventBearingCandidates, unsigned& allocatedCandidates)
{
    ++eventBearingCandidates;
    if (!eventId) {
        return success();
    }
    ++allocatedCandidates;
    const bool valid = llvm::is_contained(target.getCompilerEventIds(), *eventId) &&
                       !isReserved(schedule, source, destination, *eventId) &&
                       events.insert(eventKey(core, source, destination, *eventId)).second;
    return success(valid);
}

LogicalResult verifyAllocationState(const StructuredSyncIR& schedule, const SyncMixedProtocolPlan& plan)
{
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.isSupported()) {
        return failure();
    }
    llvm::DenseSet<std::uint64_t> events;
    unsigned eventBearingCandidates = 0;
    unsigned allocatedCandidates = 0;
    if (plan.oneShot) {
        for (const SyncOneShotProtocol& protocol : plan.oneShot->protocols) {
            if (protocol.kind == SyncOneShotProtocolKind::DirectedEvent &&
                failed(appendAllocatedEvent(
                    schedule, target, events, protocol.core, protocol.sourcePipe, protocol.targetPipe, protocol.eventId,
                    eventBearingCandidates, allocatedCandidates))) {
                return failure();
            }
        }
    }
    if (plan.readyRelease) {
        for (const SyncReadyReleaseLane& lane : plan.readyRelease->lanes) {
            if (failed(appendAllocatedEvent(
                    schedule, target, events, plan.readyRelease->core, plan.readyRelease->producerPipe,
                    plan.readyRelease->consumerPipe, lane.readyEventId, eventBearingCandidates, allocatedCandidates)) ||
                failed(appendAllocatedEvent(
                    schedule, target, events, plan.readyRelease->core, plan.readyRelease->consumerPipe,
                    plan.readyRelease->producerPipe, lane.releaseEventId, eventBearingCandidates,
                    allocatedCandidates))) {
                return failure();
            }
        }
    }
    for (const SyncDirectRepairCandidate& candidate : plan.directRepair.candidates) {
        if (candidate.kind == SyncDirectRepairKind::DirectedEvent &&
            failed(appendAllocatedEvent(
                schedule, target, events, candidate.core, candidate.sourcePipe, candidate.targetPipe, candidate.eventId,
                eventBearingCandidates, allocatedCandidates))) {
            return failure();
        }
    }
    const bool allAllocated = allocatedCandidates == eventBearingCandidates;
    const bool noneAllocated = allocatedCandidates == 0;
    return success(
        (allAllocated || noneAllocated) && (plan.status != SyncMixedPlanStatus::ResourceInfeasible || noneAllocated));
}

LogicalResult verifyGeneratedOwnership(func::FuncOp function, const SyncMixedProtocolPlan& plan)
{
    bool malformed = false;
    function.walk([&](Operation* operation) {
        const bool fixed = isFixedSyncOperation(operation);
        const bool generated = operation->hasAttrOfType<UnitAttr>(kGeneratedAttr);
        if (!fixed && !generated) {
            return;
        }
        if (!fixed || !generated) {
            malformed = true;
            return;
        }
        auto direct = operation->getAttrOfType<IntegerAttr>(kDirectCandidateAttr);
        auto protocol = operation->getAttrOfType<StringAttr>(kProtocolKindAttr);
        const bool exactlyOneOwner = static_cast<bool>(direct) != static_cast<bool>(protocol);
        if (!exactlyOneOwner) {
            malformed = true;
            return;
        }
        if (direct) {
            const bool validDirect =
                direct.getInt() >= 0 &&
                static_cast<std::uint64_t>(direct.getInt()) < plan.directRepair.candidates.size() &&
                !operation->hasAttr(kProtocolAttr);
            malformed |= !validDirect;
            return;
        }
        auto protocolId = operation->getAttrOfType<IntegerAttr>(kProtocolAttr);
        const StringRef protocolKind = protocol.getValue();
        if (protocolKind == kReadyReleaseKind) {
            malformed |= !plan.readyRelease || plan.oneShot || !protocolId || protocolId.getInt() != 0;
            return;
        }
        if (protocolKind == kOneShotKind) {
            const bool validOneShot =
                plan.oneShot && !plan.readyRelease &&
                (!protocolId || (protocolId.getInt() >= 0 &&
                                 static_cast<std::uint64_t>(protocolId.getInt()) < plan.oneShot->protocols.size()));
            malformed |= !validOneShot;
            return;
        }
        malformed = true;
    });
    return failure(malformed);
}

void buildIdentityMapping(func::FuncOp function, IRMapping& mapping)
{
    function.walk([&](Operation* operation) {
        mapping.map(operation, operation);
        for (Value result : operation->getResults()) {
            mapping.map(result, result);
        }
        for (Region& region : operation->getRegions()) {
            for (Block& block : region) {
                for (BlockArgument argument : block.getArguments()) {
                    mapping.map(argument, argument);
                }
            }
        }
    });
}

} // namespace

LogicalResult mlir::pto::protocol_sync::verifyMixedProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncMixedProtocolPlan& plan, ProtocolSyncStatistics* statistics)
{
    const bool malformedPlan = !schedule.isFrozen() || (plan.oneShot && plan.readyRelease);
    if (malformedPlan) {
        return failure();
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.isSupported()) {
        return success(
            plan.status == SyncMixedPlanStatus::Unsupported && !plan.hasProtocol() &&
            plan.directRepair.candidates.empty() && hasOnlyFailure(plan, SyncMixedPlanRejection::UnsupportedTarget));
    }
    const LogicalResult directVerified =
        verifyDirectRepairPlan(schedule, stages, plan.directObligations, plan.directRepair, statistics);
    const LogicalResult allocationVerified = verifyAllocationState(schedule, plan);
    const bool planVerified =
        succeeded(directVerified) && succeeded(allocationVerified) && !hasInternalRejection(plan.directRepair);
    if (!planVerified) {
        return failure();
    }

    FailureOr<SyncSelectedWorld> protocolWorld = buildProtocolWorld(plan, channels);
    if (failed(protocolWorld)) {
        return failure();
    }
    FailureOr<SyncInterpretationResult> initial =
        interpretSelectedWorld(schedule, stages, timelines, channels, *protocolWorld, statistics);
    const LogicalResult retainedResidualsVerified =
        failed(initial) ? failure() : verifyRetainedResiduals(initial->obligations, plan.directObligations);
    const bool initialResidualsMatch = succeeded(initial) && plan.initialResidualCount == initial->obligations.size() &&
                                       succeeded(retainedResidualsVerified);
    if (!initialResidualsMatch) {
        return failure();
    }

    if (plan.status == SyncMixedPlanStatus::Unsupported) {
        return success(
            !plan.directRepair.isComplete() && hasOnlyFailure(plan, SyncMixedPlanRejection::IncompleteDirectRepair));
    }
    if (plan.status != SyncMixedPlanStatus::Ready && plan.status != SyncMixedPlanStatus::Empty &&
        plan.status != SyncMixedPlanStatus::ResourceInfeasible) {
        return failure();
    }
    if (!plan.directRepair.isComplete()) {
        return failure();
    }
    const std::uint64_t retainedCandidates = plan.directRepair.candidates.size() + (plan.hasProtocol() ? 1 : 0);
    const bool validDeletionAccounting =
        plan.candidateCountBeforeDeletion == retainedCandidates + plan.reverseDeletionRemoved &&
        plan.reverseDeletionAttempts == plan.candidateCountBeforeDeletion;
    if (!validDeletionAccounting) {
        return failure();
    }
    if (plan.status == SyncMixedPlanStatus::ResourceInfeasible) {
        return success(hasOnlyFailure(plan, SyncMixedPlanRejection::EventCapacity));
    }
    if (!plan.failures.empty()) {
        return failure();
    }

    SmallVector<SyncDirectCandidateId, 8> selected = allDirectCandidates(plan);
    FailureOr<SyncSelectedWorld> selectedWorld = buildSelectedWorldFromPlan(plan, channels, selected);
    const bool selectedWorldMatches = succeeded(selectedWorld) && sameWorld(*selectedWorld, plan.selectedWorld);
    if (!selectedWorldMatches) {
        return failure();
    }
    if (failed(verifyReverseMinimality(schedule, stages, timelines, channels, plan, statistics))) {
        return failure();
    }
    FailureOr<SyncInterpretationResult> final =
        interpretSelectedWorld(schedule, stages, timelines, channels, *selectedWorld, statistics);
    const bool finalWorldComplete = succeeded(final) && final->isComplete();
    if (!finalWorldComplete) {
        return failure();
    }
    const bool expectedEmpty = !plan.hasProtocol() && plan.directRepair.candidates.empty();
    return success((plan.status == SyncMixedPlanStatus::Empty) == expectedEmpty);
}

LogicalResult mlir::pto::protocol_sync::materializeAndVerifyMixedProtocolPlanInDisposableModule(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncMixedProtocolPlan& plan, ProtocolSyncStatistics* statistics)
{
    if (plan.status != SyncMixedPlanStatus::Ready ||
        failed(verifyMixedProtocolPlan(schedule, stages, timelines, channels, plan, statistics))) {
        return failure();
    }
    func::FuncOp function = schedule.getFunction();
    IRMapping mapping;
    buildIdentityMapping(function, mapping);
    if (plan.oneShot && failed(materializeOneShotProtocolPlan(function, mapping, *plan.oneShot, statistics))) {
        return failure();
    }
    if (plan.readyRelease &&
        failed(materializeReadyReleaseProtocolPlan(function, mapping, *plan.readyRelease, statistics))) {
        return failure();
    }
    if (plan.directRepair.status == SyncDirectRepairPlanStatus::Ready &&
        failed(materializeDirectRepairPlan(function, mapping, plan.directRepair, statistics))) {
        return failure();
    }

    const MixedVerifierClock::time_point start = MixedVerifierClock::now();
    const bool validOneShot = !plan.oneShot || succeeded(verifyOneShotProtocolMaterialization(
                                                   schedule, stages, function, mapping, statistics));
    const bool validReadyRelease = !plan.readyRelease || succeeded(verifyReadyReleaseProtocolMaterialization(
                                                             schedule, stages, function, mapping, statistics));
    const bool validDirect =
        plan.directRepair.status != SyncDirectRepairPlanStatus::Ready ||
        succeeded(verifyDirectRepairMaterialization(
            schedule, stages, plan.directObligations, function, mapping, plan.directRepair, statistics));
    const bool verified = validOneShot && validReadyRelease && validDirect &&
                          succeeded(verifyGeneratedOwnership(function, plan)) && succeeded(mlir::verify(function));
    if (statistics) {
        statistics->verificationUs += elapsedMicroseconds(start);
    }
    if (!verified) {
        function.emitError("ProtocolSync rejected the complete mixed plan in disposable staging IR");
        return failure();
    }
    return success();
}
