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

#include "PTO/Transforms/ProtocolSync/EventAllocation.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

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

SmallVector<SyncOneShotPublishId, 4> allOneShotCandidates(const SyncMixedProtocolPlan& plan)
{
    SmallVector<SyncOneShotPublishId, 4> selected;
    if (!plan.oneShot) {
        return selected;
    }
    for (const SyncOneShotPublishCandidate& candidate : plan.oneShot->candidates) {
        selected.push_back(candidate.id);
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
    ArrayRef<SyncDirectCandidateId> selectedDirect, ArrayRef<SyncOneShotPublishId> selectedOneShot,
    bool includeReadyRelease)
{
    FailureOr<SyncSelectedWorld> world = buildProtocolWorld(plan, channels, selectedOneShot, includeReadyRelease);
    const LogicalResult applied =
        failed(world) ? failure() :
                        applyDirectRepairCandidates(plan.directRepair, plan.directObligations, selectedDirect, *world);
    if (failed(applied)) {
        return failure();
    }
    return world;
}

void pruneOneShotCandidates(SyncMixedProtocolPlan& plan, ArrayRef<SyncOneShotPublishId> selected)
{
    if (!plan.oneShot) {
        return;
    }
    SyncOneShotPublishPlan retained;
    for (SyncOneShotPublishId id : selected) {
        SyncOneShotPublishCandidate candidate = plan.oneShot->candidates[id];
        candidate.id = retained.candidates.size();
        retained.candidates.push_back(std::move(candidate));
    }
    if (retained.candidates.empty()) {
        plan.oneShot.reset();
    } else {
        plan.oneShot = std::move(retained);
    }
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
    SmallVector<SyncOneShotPublishId, 4> selectedOneShot = allOneShotCandidates(plan);
    for (std::size_t reverseIndex = selected.size(); reverseIndex > 0; --reverseIndex) {
        SmallVector<SyncDirectCandidateId, 8> trial(selected);
        trial.erase(trial.begin() + reverseIndex - 1);
        ++plan.reverseDeletionAttempts;
        FailureOr<SyncSelectedWorld> world =
            buildCompleteWorld(plan, channels, trial, selectedOneShot, plan.readyRelease.has_value());
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

    for (std::size_t reverseIndex = selectedOneShot.size(); reverseIndex > 0; --reverseIndex) {
        SmallVector<SyncOneShotPublishId, 4> trial(selectedOneShot);
        trial.erase(trial.begin() + reverseIndex - 1);
        ++plan.reverseDeletionAttempts;
        FailureOr<SyncSelectedWorld> world =
            buildCompleteWorld(plan, channels, selected, trial, plan.readyRelease.has_value());
        if (failed(world)) {
            return failure();
        }
        FailureOr<SyncInterpretationResult> result =
            interpretSelectedWorld(schedule, stages, timelines, channels, *world, statistics);
        if (failed(result)) {
            return failure();
        }
        if (result->isComplete()) {
            selectedOneShot = std::move(trial);
            ++plan.reverseDeletionRemoved;
        }
    }

    if (plan.readyRelease) {
        ++plan.reverseDeletionAttempts;
        FailureOr<SyncSelectedWorld> world = buildCompleteWorld(plan, channels, selected, selectedOneShot, false);
        if (failed(world)) {
            return failure();
        }
        FailureOr<SyncInterpretationResult> result =
            interpretSelectedWorld(schedule, stages, timelines, channels, *world, statistics);
        if (failed(result)) {
            return failure();
        }
        if (result->isComplete()) {
            plan.readyRelease.reset();
            ++plan.reverseDeletionRemoved;
        }
    }

    pruneOneShotCandidates(plan, selectedOneShot);
    return pruneDirectCandidates(plan, selected);
}

void clearEventIds(SyncMixedProtocolPlan& plan)
{
    if (plan.oneShot) {
        for (SyncOneShotPublishCandidate& candidate : plan.oneShot->candidates) {
            candidate.eventId.reset();
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
    const bool oneShotAllocated = plan.oneShot && llvm::any_of(plan.oneShot->candidates, [](const auto& candidate) {
                                      return candidate.eventId.has_value();
                                  });
    const bool readyReleaseAllocated =
        plan.readyRelease && llvm::any_of(plan.readyRelease->lanes, [](const auto& lane) {
            return lane.readyEventId.has_value() || lane.releaseEventId.has_value();
        });
    const bool directAllocated =
        llvm::any_of(plan.directRepair.candidates, [](const auto& candidate) { return candidate.eventId.has_value(); });
    return oneShotAllocated || readyReleaseAllocated || directAllocated;
}

LogicalResult buildMixedEventGenerations(
    const StructuredSyncIR& schedule, SyncMixedProtocolPlan& plan, SmallVectorImpl<SyncEventGeneration>& generations,
    SmallVectorImpl<std::optional<unsigned>*>& assignmentSlots)
{
    const auto appendGeneration = [&](SyncEventGenerationKind kind, SyncPhysicalCore core, PIPE sourcePipe,
                                      PIPE targetPipe, Operation* setAnchor, Operation* waitAnchor,
                                      ArrayRef<SyncControlAtom> guard, SyncRegionId recurrenceOwner, bool recurring,
                                      std::optional<unsigned>& eventId) {
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
        assignmentSlots.push_back(&eventId);
    };
    if (plan.oneShot) {
        for (SyncOneShotPublishCandidate& candidate : plan.oneShot->candidates) {
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
        SyncReadyReleasePlan& readyRelease = *plan.readyRelease;
        for (SyncReadyReleaseLane& lane : readyRelease.lanes) {
            appendGeneration(
                SyncEventGenerationKind::ReadyReleaseReady, readyRelease.core, readyRelease.producerPipe,
                readyRelease.consumerPipe, readyRelease.producerOperation, readyRelease.consumerOperation, {},
                readyRelease.loopRegion, true, lane.readyEventId);
            appendGeneration(
                SyncEventGenerationKind::ReadyReleaseRelease, readyRelease.core, readyRelease.consumerPipe,
                readyRelease.producerPipe, readyRelease.consumerOperation, readyRelease.producerOperation, {},
                readyRelease.loopRegion, true, lane.releaseEventId);
        }
    }
    for (SyncDirectRepairCandidate& candidate : plan.directRepair.candidates) {
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
    return success();
}

void recordSelectedStatistics(const SyncMixedProtocolPlan& plan, ProtocolSyncStatistics& statistics)
{
    if (plan.oneShot) {
        statistics.selectedOneShotProtocols += plan.oneShot->candidates.size();
        for (const SyncOneShotPublishCandidate& candidate : plan.oneShot->candidates) {
            if (candidate.kind == SyncOneShotPublishKind::PipeBarrier) {
                ++statistics.selectedSamePipeBarriers;
                ++statistics.logicalActions;
            } else if (candidate.kind == SyncOneShotPublishKind::DirectedEvent) {
                ++statistics.selectedDirectedEventPairs;
                statistics.logicalActions += 2;
            }
        }
    }
    if (plan.readyRelease) {
        ++statistics.selectedReadyReleaseProtocols;
        statistics.selectedReadyReleaseLanes += plan.readyRelease->lanes.size();
        statistics.logicalActions += 4 + 2 * plan.readyRelease->lanes.size();
    }
    statistics.selectedDirectRepairs += plan.directRepair.candidates.size();
    for (const SyncDirectRepairCandidate& candidate : plan.directRepair.candidates) {
        statistics.logicalActions += candidate.kind == SyncDirectRepairKind::DirectedEvent ? 2 : 1;
        statistics.selectedTailDrains += candidate.kind == SyncDirectRepairKind::ExitBarrier ? 1 : 0;
    }
}

SyncMixedWorldKind classifyWorld(const SyncMixedProtocolPlan& plan)
{
    if (plan.oneShot && plan.readyRelease) {
        return SyncMixedWorldKind::CombinedProtocols;
    }
    if (plan.readyRelease) {
        return SyncMixedWorldKind::ReadyRelease;
    }
    return plan.oneShot ? SyncMixedWorldKind::OneShotPublish : SyncMixedWorldKind::DirectOnly;
}

SyncMixedWorldCost computeWorldCost(const SyncMixedProtocolPlan& plan, std::uint64_t eventPressure)
{
    SyncMixedWorldCost cost;
    cost.eventPressure = eventPressure;
    if (plan.oneShot) {
        for (const SyncOneShotPublishCandidate& candidate : plan.oneShot->candidates) {
            cost.generatedEventPairs += candidate.kind == SyncOneShotPublishKind::DirectedEvent ? 1 : 0;
            cost.targetedBarriers += candidate.kind == SyncOneShotPublishKind::PipeBarrier ? 1 : 0;
            cost.staticActions += candidate.kind == SyncOneShotPublishKind::DirectedEvent ? 2 :
                                  candidate.kind == SyncOneShotPublishKind::PipeBarrier   ? 1 :
                                                                                            0;
        }
    }
    if (plan.readyRelease) {
        cost.generatedEventPairs += 2 * plan.readyRelease->lanes.size();
        cost.staticActions += 4 + 2 * plan.readyRelease->lanes.size();
    }
    for (const SyncDirectRepairCandidate& candidate : plan.directRepair.candidates) {
        cost.generatedEventPairs += candidate.kind == SyncDirectRepairKind::DirectedEvent ? 1 : 0;
        cost.targetedBarriers += candidate.kind == SyncDirectRepairKind::PipeBarrier ? 1 : 0;
        cost.fixedExitDrains += candidate.kind == SyncDirectRepairKind::ExitBarrier ? 1 : 0;
        if (candidate.kind != SyncDirectRepairKind::ExitBarrier) {
            cost.staticActions += candidate.kind == SyncDirectRepairKind::DirectedEvent ? 2 : 1;
        }
    }
    return cost;
}

unsigned worldTieBreak(SyncMixedWorldKind kind)
{
    switch (kind) {
        case SyncMixedWorldKind::CombinedProtocols:
            return 0;
        case SyncMixedWorldKind::ReadyRelease:
            return 1;
        case SyncMixedWorldKind::OneShotPublish:
            return 2;
        case SyncMixedWorldKind::DirectOnly:
            return 3;
    }
    return 3;
}

bool worldCostsLess(const SyncMixedProtocolPlan& first, const SyncMixedProtocolPlan& second)
{
    const SyncMixedWorldCost& left = first.selectedCost;
    const SyncMixedWorldCost& right = second.selectedCost;
    const std::uint64_t leftProtocols =
        (first.oneShot ? first.oneShot->candidates.size() : 0) + (first.readyRelease ? 1 : 0);
    const std::uint64_t rightProtocols =
        (second.oneShot ? second.oneShot->candidates.size() : 0) + (second.readyRelease ? 1 : 0);
    const auto leftKey = std::make_tuple(
        left.generatedEventPairs, left.targetedBarriers, left.eventPressure, left.staticActions,
        worldTieBreak(first.selectedWorldKind), std::numeric_limits<std::uint64_t>::max() - leftProtocols);
    const auto rightKey = std::make_tuple(
        right.generatedEventPairs, right.targetedBarriers, right.eventPressure, right.staticActions,
        worldTieBreak(second.selectedWorldKind), std::numeric_limits<std::uint64_t>::max() - rightProtocols);
    return leftKey < rightKey;
}

struct ProtocolSelection {
    SmallVector<SyncOneShotPublishId, 4> oneShot;
    bool readyRelease = false;
};

FailureOr<std::optional<SyncOneShotPublishPlan>> selectOneShotAlternative(
    const std::optional<SyncOneShotPublishPlan>& candidates, ArrayRef<SyncOneShotPublishId> selected)
{
    if (selected.empty()) {
        return std::optional<SyncOneShotPublishPlan>{};
    }
    if (!candidates) {
        return failure();
    }
    SyncMixedProtocolPlan source;
    source.oneShot = candidates;
    FailureOr<SyncOneShotPublishPlan> subset = selectOneShotCandidates(source, selected);
    if (failed(subset)) {
        return failure();
    }
    return std::optional<SyncOneShotPublishPlan>(std::move(*subset));
}

void recordSelectionAllocationWork(const ProtocolSyncStatistics& source, ProtocolSyncStatistics* destination)
{
    if (!destination) {
        return;
    }
    destination->allocationGraphVertices += source.allocationGraphVertices;
    destination->allocationGraphEdges += source.allocationGraphEdges;
    destination->allocationBacktrackingNodes += source.allocationBacktrackingNodes;
    destination->allocationSearchLimitHits += source.allocationSearchLimitHits;
    destination->eventDomains += source.eventDomains;
    destination->maxEventDomainPressure = std::max(destination->maxEventDomainPressure, source.maxEventDomainPressure);
    destination->maximumEventIdPlusOne = std::max(destination->maximumEventIdPlusOne, source.maximumEventIdPlusOne);
}

FailureOr<SyncMixedProtocolPlan> buildCompleteCandidateWorld(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    std::optional<SyncOneShotPublishPlan> oneShot, std::optional<SyncReadyReleasePlan> readyRelease,
    ProtocolSyncStatistics* statistics)
{
    SyncMixedProtocolPlan plan;
    plan.oneShot = std::move(oneShot);
    plan.readyRelease = std::move(readyRelease);
    SmallVector<SyncOneShotPublishId, 4> selectedOneShot = allOneShotCandidates(plan);
    FailureOr<SyncSelectedWorld> protocolWorld =
        buildProtocolWorld(plan, channels, selectedOneShot, plan.readyRelease.has_value());
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
    if (failed(direct)) {
        return failure();
    }
    const LogicalResult directVerified =
        verifyDirectRepairPlan(schedule, stages, plan.directObligations, *direct, statistics);
    const bool invalidDirectPlan = failed(directVerified) || hasInternalRejection(*direct);
    if (invalidDirectPlan) {
        return failure();
    }
    plan.directRepair = std::move(*direct);
    if (!plan.directRepair.isComplete()) {
        plan.status = SyncMixedPlanStatus::Unsupported;
        plan.failures.push_back(
            {SyncMixedPlanRejection::IncompleteDirectRepair,
             "this complete protocol world leaves obligations outside the direct-repair subset"});
        return plan;
    }
    if (failed(selectMixedProtocolCandidates(schedule, stages, timelines, channels, plan, statistics))) {
        return failure();
    }
    return plan;
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
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.isSupported()) {
        SyncMixedProtocolPlan plan;
        plan.protocolsEnabled = enableProtocols;
        plan.status = SyncMixedPlanStatus::Unsupported;
        plan.failures.push_back({SyncMixedPlanRejection::UnsupportedTarget, target.getUnsupportedReason().str()});
        return plan;
    }

    std::optional<SyncOneShotPublishPlan> oneShot;
    std::optional<SyncReadyReleasePlan> readyRelease;
    if (enableProtocols) {
        FailureOr<SyncOneShotPublishPlan> candidates =
            buildOneShotPublishCandidates(schedule, stages, timelines, channels);
        if (failed(candidates)) {
            return failure();
        }
        if (!candidates->candidates.empty()) {
            if (failed(verifyOneShotPublishPlan(schedule, stages, timelines, channels, *candidates))) {
                return failure();
            }
            if (statistics) {
                statistics->protocolCandidates += candidates->candidates.size();
            }
            oneShot = std::move(*candidates);
        }

        FailureOr<SyncReadyReleasePlan> candidate = buildReadyReleaseProtocolPlan(
            schedule, stages, timelines, channels, nullptr, SyncReadyReleasePlanningScope::MixedCandidate);
        const bool readyReleaseFailed = failed(candidate) || hasInternalRejection(*candidate);
        if (readyReleaseFailed) {
            return failure();
        }
        if (candidate->status == SyncReadyReleasePlanStatus::Ready) {
            if (statistics) {
                ++statistics->protocolCandidates;
                statistics->tokenCertificateTransitions += candidate->tokenCertificate.transitionApplications;
            }
            readyRelease = std::move(*candidate);
        }
    }

    std::optional<SyncMixedProtocolPlan> best;
    std::optional<SyncMixedProtocolPlan> unsupported;
    std::optional<SyncMixedProtocolPlan> resourceInfeasible;
    ProtocolSelection bestSelection;
    std::uint64_t worldsAttempted = 0;
    std::uint64_t worldsFeasible = 0;
    const auto evaluateSelection = [&](const ProtocolSelection& selection) -> LogicalResult {
        ++worldsAttempted;
        FailureOr<std::optional<SyncOneShotPublishPlan>> selectedOneShot =
            selectOneShotAlternative(oneShot, selection.oneShot);
        if (failed(selectedOneShot)) {
            return failure();
        }
        std::optional<SyncReadyReleasePlan> selectedReadyRelease;
        if (selection.readyRelease) {
            if (!readyRelease) {
                return failure();
            }
            selectedReadyRelease = readyRelease;
        }
        FailureOr<SyncMixedProtocolPlan> candidate = buildCompleteCandidateWorld(
            schedule, stages, timelines, channels, std::move(*selectedOneShot), std::move(selectedReadyRelease),
            statistics);
        if (failed(candidate)) {
            return failure();
        }
        candidate->protocolsEnabled = enableProtocols;
        candidate->selectedWorldKind = classifyWorld(*candidate);
        if (!candidate->isComplete()) {
            if (!unsupported) {
                unsupported = std::move(*candidate);
            }
            return success();
        }

        SyncMixedProtocolPlan allocated = *candidate;
        ProtocolSyncStatistics allocationStatistics;
        if (failed(allocateMixedProtocolEvents(schedule, allocated, &allocationStatistics))) {
            return failure();
        }
        recordSelectionAllocationWork(allocationStatistics, statistics);
        if (allocated.status == SyncMixedPlanStatus::ResourceInfeasible) {
            allocated.selectedCost = computeWorldCost(*candidate, allocationStatistics.maxEventDomainPressure);
            if (!resourceInfeasible) {
                resourceInfeasible = std::move(allocated);
            }
            return success();
        }
        candidate->selectedCost = computeWorldCost(*candidate, allocationStatistics.maxEventDomainPressure);
        ++worldsFeasible;
        if (!best || worldCostsLess(*candidate, *best)) {
            best = std::move(*candidate);
            bestSelection = selection;
        }
        return success();
    };

    ProtocolSelection emptySelection;
    if (failed(evaluateSelection(emptySelection))) {
        return failure();
    }
    if (oneShot) {
        for (const SyncOneShotPublishCandidate& candidate : oneShot->candidates) {
            ProtocolSelection singleton;
            singleton.oneShot.push_back(candidate.id);
            if (failed(evaluateSelection(singleton))) {
                return failure();
            }
        }
    }
    if (readyRelease) {
        ProtocolSelection singleton;
        singleton.readyRelease = true;
        if (failed(evaluateSelection(singleton))) {
            return failure();
        }
    }

    const unsigned protocolCandidateCount = (oneShot ? oneShot->candidates.size() : 0) + (readyRelease ? 1 : 0);
    if (!best && protocolCandidateCount > 1) {
        ProtocolSelection allProtocols;
        if (oneShot) {
            for (const SyncOneShotPublishCandidate& candidate : oneShot->candidates) {
                allProtocols.oneShot.push_back(candidate.id);
            }
        }
        allProtocols.readyRelease = readyRelease.has_value();
        if (failed(evaluateSelection(allProtocols))) {
            return failure();
        }
    }
    if (best) {
        if (oneShot) {
            for (const SyncOneShotPublishCandidate& candidate : oneShot->candidates) {
                if (llvm::is_contained(bestSelection.oneShot, candidate.id)) {
                    continue;
                }
                ProtocolSelection trial = bestSelection;
                trial.oneShot.push_back(candidate.id);
                llvm::sort(trial.oneShot);
                if (failed(evaluateSelection(trial))) {
                    return failure();
                }
            }
        }
        if (readyRelease && !bestSelection.readyRelease) {
            ProtocolSelection trial = bestSelection;
            trial.readyRelease = true;
            if (failed(evaluateSelection(trial))) {
                return failure();
            }
        }
    }

    if (statistics) {
        statistics->completeWorldsAttempted += worldsAttempted;
        statistics->completeWorldsFeasible += worldsFeasible;
    }
    if (best) {
        best->completeWorldsAttempted = worldsAttempted;
        best->completeWorldsFeasible = worldsFeasible;
        if (statistics) {
            statistics->selectedWorldEventPairs += best->selectedCost.generatedEventPairs;
            statistics->selectedWorldTargetedBarriers += best->selectedCost.targetedBarriers;
            statistics->selectedWorldFixedExitDrains += best->selectedCost.fixedExitDrains;
        }
        return std::move(*best);
    }
    SyncMixedProtocolPlan result = resourceInfeasible ? std::move(*resourceInfeasible) :
                                   unsupported        ? std::move(*unsupported) :
                                                        SyncMixedProtocolPlan{};
    result.protocolsEnabled = enableProtocols;
    result.completeWorldsAttempted = worldsAttempted;
    result.completeWorldsFeasible = 0;
    if (!resourceInfeasible && !unsupported) {
        result.status = SyncMixedPlanStatus::Unsupported;
        result.failures.push_back(
            {SyncMixedPlanRejection::IncompleteDirectRepair, "no complete protocol/direct world was constructible"});
    }
    return result;
}

LogicalResult mlir::pto::protocol_sync::selectMixedProtocolCandidates(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels, SyncMixedProtocolPlan& plan,
    ProtocolSyncStatistics* statistics)
{
    const bool validCandidatePool =
        schedule.isFrozen() && plan.directRepair.isComplete() && plan.failures.empty() && !hasAllocatedEvent(plan);
    if (!validCandidatePool) {
        return failure();
    }
    plan.candidateCountBeforeDeletion = plan.directRepair.candidates.size();
    plan.candidateCountBeforeDeletion += plan.oneShot ? plan.oneShot->candidates.size() : 0;
    plan.candidateCountBeforeDeletion += plan.readyRelease ? 1 : 0;
    plan.reverseDeletionAttempts = 0;
    plan.reverseDeletionRemoved = 0;
    if (failed(reverseDeleteCandidates(schedule, stages, timelines, channels, plan, statistics))) {
        return failure();
    }
    SmallVector<SyncDirectCandidateId, 8> selectedDirect = allDirectCandidates(plan.directRepair);
    SmallVector<SyncOneShotPublishId, 4> selectedOneShot = allOneShotCandidates(plan);
    FailureOr<SyncSelectedWorld> selectedWorld =
        buildCompleteWorld(plan, channels, selectedDirect, selectedOneShot, plan.readyRelease.has_value());
    if (failed(selectedWorld)) {
        return failure();
    }
    FailureOr<SyncInterpretationResult> final =
        interpretSelectedWorld(schedule, stages, timelines, channels, *selectedWorld, statistics);
    const bool finalWorldComplete = succeeded(final) && final->isComplete();
    if (!finalWorldComplete) {
        return failure();
    }
    FailureOr<SyncSelectedWorld> protocolWorld =
        buildProtocolWorld(plan, channels, selectedOneShot, plan.readyRelease.has_value());
    if (failed(protocolWorld)) {
        return failure();
    }
    FailureOr<SyncInterpretationResult> initial =
        interpretSelectedWorld(schedule, stages, timelines, channels, *protocolWorld, statistics);
    if (failed(initial)) {
        return failure();
    }
    plan.initialResidualCount = initial->obligations.size();
    plan.selectedWorld = std::move(*selectedWorld);
    plan.selectedWorldKind = classifyWorld(plan);
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
    SmallVector<SyncEventReservation, 8> reservations;
    for (const SyncOpSummary& summary : schedule.getSummaries()) {
        reservations.append(summary.eventReservations.begin(), summary.eventReservations.end());
    }
    SmallVector<SyncEventGeneration, 16> generations;
    SmallVector<std::optional<unsigned>*, 16> assignmentSlots;
    if (failed(buildMixedEventGenerations(schedule, allocated, generations, assignmentSlots))) {
        return failure();
    }
    FailureOr<SyncEventAllocationResult> allocation = allocateSyncEventGenerations(target, reservations, generations);
    if (failed(allocation)) {
        return failure();
    }
    if (statistics) {
        recordSyncEventAllocationStatistics(*allocation, *statistics);
    }
    if (allocation->status == SyncEventAllocationStatus::ResourceInfeasible) {
        clearEventIds(plan);
        plan.status = SyncMixedPlanStatus::ResourceInfeasible;
        plan.failures.push_back(
            {SyncMixedPlanRejection::EventCapacity,
             "interfering mixed-protocol event generations exhaust a selected event domain"});
        return success();
    }
    if (allocation->status != SyncEventAllocationStatus::Allocated) {
        return failure();
    }
    for (auto [slot, eventId] : llvm::zip_equal(assignmentSlots, allocation->eventIds)) {
        *slot = eventId;
    }
    plan = std::move(allocated);
    if (statistics) {
        recordSelectedStatistics(plan, *statistics);
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

StringRef mlir::pto::protocol_sync::stringifySyncMixedWorldKind(SyncMixedWorldKind kind)
{
    switch (kind) {
        case SyncMixedWorldKind::DirectOnly:
            return "direct-only";
        case SyncMixedWorldKind::OneShotPublish:
            return "one-shot-publish";
        case SyncMixedWorldKind::ReadyRelease:
            return "ready-release";
        case SyncMixedWorldKind::CombinedProtocols:
            return "combined-protocols";
    }
    return "direct-only";
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
    output << "PROTOCOL-SYNC mixed-plan function=@" << function.getSymName()
           << " status=" << stringifySyncMixedPlanStatus(plan.status)
           << " world=" << stringifySyncMixedWorldKind(plan.selectedWorldKind)
           << " initial-residuals=" << plan.initialResidualCount
           << " direct-candidates=" << plan.directRepair.candidates.size()
           << " candidates-before-deletion=" << plan.candidateCountBeforeDeletion
           << " reverse-attempts=" << plan.reverseDeletionAttempts << " reverse-removed=" << plan.reverseDeletionRemoved
           << " worlds-attempted=" << plan.completeWorldsAttempted << " worlds-feasible=" << plan.completeWorldsFeasible
           << " event-pairs=" << plan.selectedCost.generatedEventPairs
           << " targeted-barriers=" << plan.selectedCost.targetedBarriers
           << " fixed-exit-drains=" << plan.selectedCost.fixedExitDrains
           << " event-pressure=" << plan.selectedCost.eventPressure << '\n';
    for (const SyncMixedPlanFailure& failure : plan.failures) {
        output << "  reason=" << stringifySyncMixedPlanRejection(failure.reason) << " detail=\"" << failure.detail
               << "\"\n";
    }
    if (plan.oneShot) {
        for (const SyncOneShotPublishCandidate& candidate : plan.oneShot->candidates) {
            output << "  one-shot-publish #" << candidate.id
                   << " kind=" << stringifySyncOneShotPublishKind(candidate.kind) << " phases=#"
                   << candidate.sourcePhase << "->#" << candidate.targetPhase << " channels=[";
            llvm::interleaveComma(candidate.channels, output, [&](SyncChannelId channel) { output << '#' << channel; });
            output << "] event=";
            if (candidate.eventId) {
                output << *candidate.eventId;
            } else {
                output << "unallocated";
            }
            output << '\n';
        }
    }
    if (plan.readyRelease) {
        printReadyReleaseProtocolPlan(function, *plan.readyRelease, output);
    }
    if (plan.directRepair.status != SyncDirectRepairPlanStatus::Empty) {
        printDirectRepairPlan(function, plan.directRepair, output);
    }
}
