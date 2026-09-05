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
#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"
#include "PTO/Transforms/ProtocolSync/EventAllocation.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <chrono>
#include <map>
#include <tuple>
#include <vector>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using MixedVerifierClock = std::chrono::steady_clock;
constexpr StringLiteral kGeneratedAttr = "pto.protocol_sync.generated";
constexpr StringLiteral kProtocolAttr = "pto.protocol_sync.protocol_id";
constexpr StringLiteral kProtocolKindAttr = "pto.protocol_sync.protocol_kind";
constexpr StringLiteral kDirectCandidateAttr = "pto.protocol_sync.direct_candidate_id";
constexpr StringLiteral kOneShotKind = "one-shot-publish";
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
    SyncControlRelation, SyncIterationRelationKind, unsigned, SyncRegionId, std::string, std::optional<std::uint32_t>,
    std::vector<std::uint32_t>, SyncAccessId, SyncAccessId, SyncRegionPrecision>;

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
        obligation.detail,
        obligation.localRequirement,
        {obligation.atoms.begin(), obligation.atoms.end()},
        obligation.sourceAccess,
        obligation.targetAccess,
        obligation.precision};
}

bool hasInternalRejection(const SyncDirectRepairPlan& plan)
{
    return llvm::any_of(plan.rejections, [](const SyncDirectRepairPlanRejection& rejection) {
        return rejection.reason == SyncDirectRepairRejection::InternalInvariant;
    });
}

SmallVector<SyncOneShotPublishId, 4> allOneShotCandidates(const SyncMixedProtocolPlan& plan)
{
    SmallVector<SyncOneShotPublishId, 4> selected;
    if (plan.oneShot) {
        for (const SyncOneShotPublishCandidate& candidate : plan.oneShot->candidates) {
            selected.push_back(candidate.id);
        }
    }
    return selected;
}

FailureOr<SyncOneShotPublishPlan> selectOneShotCandidates(
    const SyncMixedProtocolPlan& plan, ArrayRef<SyncOneShotPublishId> selected)
{
    SyncOneShotPublishPlan result;
    if (selected.empty()) {
        return result;
    }
    if (!plan.oneShot) {
        return failure();
    }
    llvm::BitVector seen(plan.oneShot->candidates.size());
    for (SyncOneShotPublishId id : selected) {
        const bool valid = id < seen.size() && !seen.test(id) && plan.oneShot->candidates[id].id == id;
        if (!valid) {
            return failure();
        }
        seen.set(id);
        SyncOneShotPublishCandidate candidate = plan.oneShot->candidates[id];
        candidate.id = result.candidates.size();
        result.candidates.push_back(std::move(candidate));
    }
    return result;
}

FailureOr<SyncSelectedWorld> buildProtocolWorld(
    const SyncMixedProtocolPlan& plan, const ChannelAnalysisResult& channels,
    ArrayRef<SyncOneShotPublishId> selectedOneShot, bool includeReadyRelease)
{
    SyncSelectedWorld world;
    if (includeReadyRelease && plan.readyRelease) {
        FailureOr<SyncSelectedWorld> readyRelease = buildSelectedWorld(*plan.readyRelease);
        if (failed(readyRelease)) {
            return failure();
        }
        world = std::move(*readyRelease);
    }
    FailureOr<SyncOneShotPublishPlan> oneShot = selectOneShotCandidates(plan, selectedOneShot);
    const bool invalidOneShot = failed(oneShot) || failed(appendOneShotPublishSelectedWorld(*oneShot, channels, world));
    if (invalidOneShot) {
        return failure();
    }
    return world;
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
    ArrayRef<SyncDirectCandidateId> directCandidates, ArrayRef<SyncOneShotPublishId> oneShotCandidates,
    bool includeReadyRelease)
{
    FailureOr<SyncSelectedWorld> world = buildProtocolWorld(plan, channels, oneShotCandidates, includeReadyRelease);
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

SyncMixedWorldKind classifySelectedWorld(const SyncMixedProtocolPlan& plan)
{
    if (plan.oneShot && plan.readyRelease) {
        return SyncMixedWorldKind::CombinedProtocols;
    }
    if (plan.readyRelease) {
        return SyncMixedWorldKind::ReadyRelease;
    }
    return plan.oneShot ? SyncMixedWorldKind::OneShotPublish : SyncMixedWorldKind::DirectOnly;
}

SyncMixedWorldCost reconstructSelectedCost(const SyncMixedProtocolPlan& plan)
{
    SyncMixedWorldCost cost;
    cost.eventPressure = plan.selectedCost.eventPressure;
    if (plan.oneShot) {
        for (const SyncOneShotPublishCandidate& candidate : plan.oneShot->candidates) {
            if (candidate.kind == SyncOneShotPublishKind::DirectedEvent) {
                ++cost.generatedEventPairs;
                cost.staticActions += 2;
            } else if (candidate.kind == SyncOneShotPublishKind::PipeBarrier) {
                ++cost.targetedBarriers;
                ++cost.staticActions;
            }
        }
    }
    if (plan.readyRelease) {
        cost.generatedEventPairs += 2 * plan.readyRelease->lanes.size();
        cost.staticActions += 4 + 2 * plan.readyRelease->lanes.size();
    }
    for (const SyncDirectRepairCandidate& candidate : plan.directRepair.candidates) {
        if (candidate.kind == SyncDirectRepairKind::DirectedEvent) {
            ++cost.generatedEventPairs;
            cost.staticActions += 2;
        } else if (candidate.kind == SyncDirectRepairKind::PipeBarrier) {
            ++cost.targetedBarriers;
            ++cost.staticActions;
        } else if (candidate.kind == SyncDirectRepairKind::ExitBarrier) {
            ++cost.fixedExitDrains;
        }
    }
    return cost;
}

bool selectedCostMatches(const SyncMixedProtocolPlan& plan)
{
    const SyncMixedWorldCost expected = reconstructSelectedCost(plan);
    return plan.selectedWorldKind == classifySelectedWorld(plan) &&
           plan.selectedCost.generatedEventPairs == expected.generatedEventPairs &&
           plan.selectedCost.targetedBarriers == expected.targetedBarriers &&
           plan.selectedCost.fixedExitDrains == expected.fixedExitDrains &&
           plan.selectedCost.staticActions == expected.staticActions && plan.completeWorldsAttempted != 0 &&
           plan.completeWorldsFeasible != 0 && plan.completeWorldsFeasible <= plan.completeWorldsAttempted;
}

bool infeasibleCostMatches(const SyncMixedProtocolPlan& plan)
{
    const SyncMixedWorldCost expected = reconstructSelectedCost(plan);
    return plan.selectedWorldKind == classifySelectedWorld(plan) &&
           plan.selectedCost.generatedEventPairs == expected.generatedEventPairs &&
           plan.selectedCost.targetedBarriers == expected.targetedBarriers &&
           plan.selectedCost.fixedExitDrains == expected.fixedExitDrains &&
           plan.selectedCost.staticActions == expected.staticActions && plan.completeWorldsAttempted != 0 &&
           plan.completeWorldsFeasible == 0;
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
    SmallVector<SyncOneShotPublishId, 4> selectedOneShot = allOneShotCandidates(plan);
    for (std::size_t index = 0; index < selected.size(); ++index) {
        SmallVector<SyncDirectCandidateId, 8> trial(selected);
        trial.erase(trial.begin() + index);
        FailureOr<SyncSelectedWorld> world =
            buildSelectedWorldFromPlan(plan, channels, trial, selectedOneShot, plan.readyRelease.has_value());
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
    for (std::size_t index = 0; index < selectedOneShot.size(); ++index) {
        SmallVector<SyncOneShotPublishId, 4> trial(selectedOneShot);
        trial.erase(trial.begin() + index);
        FailureOr<SyncSelectedWorld> world =
            buildSelectedWorldFromPlan(plan, channels, selected, trial, plan.readyRelease.has_value());
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
    if (plan.readyRelease) {
        FailureOr<SyncSelectedWorld> world =
            buildSelectedWorldFromPlan(plan, channels, selected, selectedOneShot, false);
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
    return success();
}

LogicalResult verifyAllocationState(const StructuredSyncIR& schedule, const SyncMixedProtocolPlan& plan)
{
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.supportsMixedEmission()) {
        return failure();
    }
    SmallVector<SyncEventReservation, 8> reservations;
    for (const SyncOpSummary& summary : schedule.getSummaries()) {
        reservations.append(summary.eventReservations.begin(), summary.eventReservations.end());
    }
    if (failed(reconstructFixedSyncSupply(schedule, &reservations))) {
        return failure();
    }
    SmallVector<SyncEventGeneration, 16> generations;
    const auto appendGeneration = [&](SyncEventGenerationKind kind, SyncPhysicalCore core, PIPE sourcePipe,
                                      PIPE targetPipe, Operation* setAnchor, Operation* waitAnchor,
                                      ArrayRef<SyncControlAtom> guard, SyncRegionId recurrenceOwner, bool recurring,
                                      std::optional<unsigned> eventId) {
        SyncEventGeneration generation;
        generation.id = generations.size();
        generation.kind = kind;
        generation.core = core;
        generation.sourcePipe = sourcePipe;
        generation.targetPipe = targetPipe;
        generation.setAnchor = setAnchor;
        generation.waitAnchor = waitAnchor;
        generation.guard.assign(guard.begin(), guard.end());
        generation.recurrenceOwner = recurrenceOwner;
        generation.recurring = recurring;
        generation.eventId = eventId;
        generations.push_back(std::move(generation));
    };
    if (plan.oneShot) {
        for (const SyncOneShotPublishCandidate& candidate : plan.oneShot->candidates) {
            if (candidate.kind != SyncOneShotPublishKind::DirectedEvent) {
                continue;
            }
            const SyncPhase* source = schedule.findPhase(candidate.sourcePhase);
            if (!source) {
                return failure();
            }
            appendGeneration(
                SyncEventGenerationKind::OneShot, candidate.core, candidate.sourcePipe, candidate.targetPipe,
                candidate.sourceOperation, candidate.targetOperation, source->guard, kInvalidSyncId, false,
                candidate.eventId);
        }
    }
    if (plan.readyRelease) {
        for (const SyncReadyReleaseLane& lane : plan.readyRelease->lanes) {
            appendGeneration(
                SyncEventGenerationKind::ReadyReleaseReady, plan.readyRelease->core, plan.readyRelease->producerPipe,
                plan.readyRelease->consumerPipe, plan.readyRelease->producerOperation,
                plan.readyRelease->consumerOperation, {}, plan.readyRelease->loopRegion, true, lane.readyEventId);
            appendGeneration(
                SyncEventGenerationKind::ReadyReleaseRelease, plan.readyRelease->core, plan.readyRelease->consumerPipe,
                plan.readyRelease->producerPipe, plan.readyRelease->consumerOperation,
                plan.readyRelease->producerOperation, {}, plan.readyRelease->loopRegion, true, lane.releaseEventId);
        }
    }
    for (const SyncDirectRepairCandidate& candidate : plan.directRepair.candidates) {
        if (candidate.kind != SyncDirectRepairKind::DirectedEvent) {
            continue;
        }
        const SyncPhase* source = schedule.findPhase(candidate.sourcePhase);
        if (!source) {
            return failure();
        }
        appendGeneration(
            SyncEventGenerationKind::DirectRepair, candidate.core, candidate.sourcePipe, candidate.targetPipe,
            candidate.sourceOperation, candidate.targetOperation, source->guard, kInvalidSyncId, false,
            candidate.eventId);
    }
    const bool anyAllocated =
        llvm::any_of(generations, [](const SyncEventGeneration& generation) { return generation.eventId.has_value(); });
    SmallVector<SyncEventGeneration, 16> unallocated(generations);
    for (SyncEventGeneration& generation : unallocated) {
        generation.eventId.reset();
    }
    FailureOr<SyncEventAllocationResult> authoritative =
        allocateSyncEventGenerations(target, reservations, unallocated);
    const bool expectsResourceInfeasible = plan.status == SyncMixedPlanStatus::ResourceInfeasible;
    const bool authoritativeStatusMatches =
        succeeded(authoritative) &&
        (expectsResourceInfeasible ? authoritative->status == SyncEventAllocationStatus::ResourceInfeasible :
                                     authoritative->status == SyncEventAllocationStatus::Allocated) &&
        authoritative->maximumDomainPressure == plan.selectedCost.eventPressure;
    return success(
        succeeded(verifySyncEventGenerationAssignment(target, reservations, generations)) &&
        (plan.status != SyncMixedPlanStatus::ResourceInfeasible || !anyAllocated) && authoritativeStatusMatches);
}

LogicalResult verifyGeneratedOwnership(const StructuredSyncIR& schedule, const SyncMixedProtocolPlan& plan)
{
    // Only operations from the pre-emission schedule are fixed supply. An
    // untagged newly emitted operation must still fail ownership validation.
    llvm::DenseSet<Operation*> fixedSupply;
    for (const SyncOpSummary& summary : schedule.getSummaries()) {
        if (summary.provider == SyncSummaryProvider::FixedSynchronization) {
            fixedSupply.insert(summary.operation);
        }
    }
    bool malformed = false;
    schedule.getFunction().walk([&](Operation* operation) {
        if (fixedSupply.contains(operation)) {
            return;
        }
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
            malformed |= !plan.readyRelease || !protocolId || protocolId.getInt() != 0;
            return;
        }
        if (protocolKind == kOneShotKind) {
            const bool validOneShot = plan.oneShot && protocolId && protocolId.getInt() >= 0 &&
                                      static_cast<std::uint64_t>(protocolId.getInt()) < plan.oneShot->candidates.size();
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
    if (plan.loopFrontier) {
        return verifyMixedLoopFrontierPlan(schedule, stages, timelines, channels, plan);
    }
    if (plan.selectedWorld.orderedLoop) {
        return failure();
    }
    const bool malformedPlan = !schedule.isFrozen();
    if (malformedPlan) {
        return failure();
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.supportsMixedEmission()) {
        return success(
            plan.status == SyncMixedPlanStatus::Unsupported && !plan.hasProtocol() &&
            plan.directRepair.candidates.empty() && hasOnlyFailure(plan, SyncMixedPlanRejection::UnsupportedTarget));
    }
    const LogicalResult directVerified =
        verifyDirectRepairPlan(schedule, stages, plan.directObligations, plan.directRepair, statistics);
    const LogicalResult oneShotVerified =
        plan.oneShot ? verifyOneShotPublishPlan(schedule, stages, timelines, channels, *plan.oneShot) : success();
    const LogicalResult allocationVerified =
        plan.status == SyncMixedPlanStatus::Unsupported ? success() : verifyAllocationState(schedule, plan);
    const bool planVerified = succeeded(directVerified) && succeeded(oneShotVerified) &&
                              succeeded(allocationVerified) && !hasInternalRejection(plan.directRepair);
    if (!planVerified) {
        return failure();
    }

    SmallVector<SyncOneShotPublishId, 4> selectedOneShot = allOneShotCandidates(plan);
    FailureOr<SyncSelectedWorld> protocolWorld =
        buildProtocolWorld(plan, channels, selectedOneShot, plan.readyRelease.has_value());
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
    std::uint64_t retainedCandidates = plan.directRepair.candidates.size();
    retainedCandidates += plan.oneShot ? plan.oneShot->candidates.size() : 0;
    retainedCandidates += plan.readyRelease ? 1 : 0;
    const bool validDeletionAccounting =
        plan.candidateCountBeforeDeletion == retainedCandidates + plan.reverseDeletionRemoved &&
        plan.reverseDeletionAttempts == plan.candidateCountBeforeDeletion;
    if (!validDeletionAccounting) {
        return failure();
    }
    if (plan.status == SyncMixedPlanStatus::ResourceInfeasible) {
        const bool malformedResourceFailure =
            !hasOnlyFailure(plan, SyncMixedPlanRejection::EventCapacity) || !infeasibleCostMatches(plan);
        if (malformedResourceFailure) {
            return failure();
        }
        FailureOr<SyncMixedProtocolPlan> authoritative =
            buildMixedProtocolPlan(schedule, stages, timelines, channels, plan.protocolsEnabled, nullptr);
        const bool authoritativeResourceFailure =
            succeeded(authoritative) && authoritative->status == SyncMixedPlanStatus::ResourceInfeasible &&
            authoritative->selectedWorldKind == plan.selectedWorldKind &&
            authoritative->selectedCost.generatedEventPairs == plan.selectedCost.generatedEventPairs &&
            authoritative->selectedCost.targetedBarriers == plan.selectedCost.targetedBarriers &&
            authoritative->selectedCost.fixedExitDrains == plan.selectedCost.fixedExitDrains &&
            authoritative->selectedCost.eventPressure == plan.selectedCost.eventPressure &&
            authoritative->selectedCost.staticActions == plan.selectedCost.staticActions &&
            authoritative->completeWorldsAttempted == plan.completeWorldsAttempted &&
            authoritative->completeWorldsFeasible == plan.completeWorldsFeasible &&
            sameWorld(authoritative->selectedWorld, plan.selectedWorld);
        return success(authoritativeResourceFailure);
    }
    if (!plan.failures.empty()) {
        return failure();
    }
    if (!selectedCostMatches(plan)) {
        return failure();
    }

    FailureOr<SyncMixedProtocolPlan> authoritative =
        buildMixedProtocolPlan(schedule, stages, timelines, channels, plan.protocolsEnabled, nullptr);
    const bool optimalWorldMatches =
        succeeded(authoritative) && authoritative->isComplete() &&
        authoritative->selectedWorldKind == plan.selectedWorldKind &&
        authoritative->selectedCost.generatedEventPairs == plan.selectedCost.generatedEventPairs &&
        authoritative->selectedCost.targetedBarriers == plan.selectedCost.targetedBarriers &&
        authoritative->selectedCost.fixedExitDrains == plan.selectedCost.fixedExitDrains &&
        authoritative->selectedCost.eventPressure == plan.selectedCost.eventPressure &&
        authoritative->selectedCost.staticActions == plan.selectedCost.staticActions &&
        authoritative->completeWorldsAttempted == plan.completeWorldsAttempted &&
        authoritative->completeWorldsFeasible == plan.completeWorldsFeasible &&
        sameWorld(authoritative->selectedWorld, plan.selectedWorld);
    if (!optimalWorldMatches) {
        return failure();
    }

    SmallVector<SyncDirectCandidateId, 8> selected = allDirectCandidates(plan);
    FailureOr<SyncSelectedWorld> selectedWorld =
        buildSelectedWorldFromPlan(plan, channels, selected, selectedOneShot, plan.readyRelease.has_value());
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
    const SyncMixedProtocolPlan& plan, const SyncSemanticContext& context, ProtocolSyncStatistics* statistics)
{
    if (plan.status != SyncMixedPlanStatus::Ready ||
        failed(verifyMixedProtocolPlan(schedule, stages, timelines, channels, plan, statistics))) {
        return failure();
    }
    func::FuncOp function = schedule.getFunction();
    IRMapping mapping;
    buildIdentityMapping(function, mapping);
    if (plan.loopFrontier) {
        if (failed(materializeLoopFrontierRepair(function, mapping, *plan.loopFrontier))) {
            return failure();
        }
        return verifyFreshConcreteSyncSemantics(function, statistics);
    }
    if (plan.oneShot && failed(materializeOneShotPublishPlan(function, mapping, *plan.oneShot, statistics))) {
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
    const bool validOneShot = !plan.oneShot || succeeded(verifyOneShotPublishMaterialization(
                                                   schedule, function, mapping, *plan.oneShot, statistics));
    const bool validReadyRelease = !plan.readyRelease || succeeded(verifyReadyReleaseProtocolMaterialization(
                                                             schedule, stages, function, mapping, statistics));
    const bool validDirect =
        plan.directRepair.status != SyncDirectRepairPlanStatus::Ready ||
        succeeded(verifyDirectRepairMaterialization(
            schedule, stages, plan.directObligations, function, mapping, plan.directRepair, statistics));
    const bool verified =
        validOneShot && validReadyRelease && validDirect && succeeded(verifyGeneratedOwnership(schedule, plan)) &&
        succeeded(verifyConcreteSyncSemantics(context, function, statistics)) && succeeded(mlir::verify(function));
    if (statistics) {
        statistics->verificationUs += elapsedMicroseconds(start);
    }
    if (!verified) {
        function.emitError("ProtocolSync rejected the complete mixed plan in disposable staging IR");
        return failure();
    }
    return success();
}
