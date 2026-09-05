// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- DirectRepair.cpp - Plan targeted residual synchronization -------===//

#include "PTO/Transforms/ProtocolSync/DirectRepair.h"

#include "PTO/Transforms/ProtocolSync/EventAllocation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

bool sameGuard(ArrayRef<SyncControlAtom> first, ArrayRef<SyncControlAtom> second)
{
    return first.size() == second.size() && llvm::equal(first, second, [](const auto& lhs, const auto& rhs) {
               return lhs.choice == rhs.choice && lhs.arm == rhs.arm;
           });
}

SyncControlRelation phaseControlRelation(const SyncPhase& source, const SyncPhase& target)
{
    if (!sameGuard(source.guard, target.guard)) {
        return SyncControlRelation::Unknown;
    }
    return source.guard.empty() ? SyncControlRelation::MustExecute : SyncControlRelation::SameGuard;
}

bool isBefore(Operation* source, Operation* target)
{
    return source && target && source != target && source->getBlock() == target->getBlock() &&
           source->isBeforeInBlock(target);
}

bool hasExactPhaseStage(const PipelineStageAnalysisResult& stages, const SyncPhase& phase)
{
    const SyncStage* stage = stages.findStageForPhase(phase.id);
    return stage && stage->phases.size() == 1 && stage->phases.front() == phase.id &&
           phase.completion == SyncCompletionKind::PhaseEnd;
}

const SyncRegion* enclosingPhysicalSection(const StructuredSyncIR& schedule, const SyncPhase& phase)
{
    const SyncRegion* region = schedule.findRegion(phase.region);
    while (region) {
        if (region->kind == SyncRegionKind::PhysicalSection) {
            return region;
        }
        region = region->parent == kInvalidSyncId ? nullptr : schedule.findRegion(region->parent);
    }
    return nullptr;
}

bool supportsDirectCompletion(SyncObligationKind kind)
{
    return kind == SyncObligationKind::Completion || kind == SyncObligationKind::Reclamation ||
           kind == SyncObligationKind::SSACompletion;
}

void rejectObligation(
    SyncDirectRepairPlan& plan, const SyncResidualObligation& obligation, SyncDirectRepairRejection reason,
    StringRef detail)
{
    plan.uncoveredObligations.push_back(obligation.id);
    plan.rejections.push_back({obligation.id, reason, detail.str()});
}

struct RepairInterval {
    const SyncResidualObligation* obligation = nullptr;
    const SyncPhase* source = nullptr;
    const SyncPhase* target = nullptr;
};

struct FrontierGroup {
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE sourcePipe = PIPE::PIPE_UNASSIGNED;
    PIPE targetPipe = PIPE::PIPE_UNASSIGNED;
    Block* block = nullptr;
    SyncControlRelation control = SyncControlRelation::Unknown;
    llvm::SmallVector<SyncControlAtom, 2> guard;
    llvm::SmallVector<RepairInterval, 4> intervals;
};

bool matchesGroup(
    const FrontierGroup& group, const SyncPhase& source, const SyncPhase& target, SyncControlRelation control)
{
    return group.core == source.core && group.sourcePipe == source.pipe && group.targetPipe == target.pipe &&
           group.block == source.operation->getBlock() && group.control == control &&
           sameGuard(group.guard, source.guard);
}

void addIntervalToGroups(
    const RepairInterval& interval, SmallVectorImpl<FrontierGroup>& groups,
    llvm::DenseMap<Block*, SmallVector<unsigned, 4>>& groupsByBlock)
{
    const SyncPhase& source = *interval.source;
    const SyncPhase& target = *interval.target;
    const SyncControlRelation control = phaseControlRelation(source, target);
    SmallVector<unsigned, 4>& blockGroups = groupsByBlock[source.operation->getBlock()];
    for (unsigned index : blockGroups) {
        if (matchesGroup(groups[index], source, target, control)) {
            groups[index].intervals.push_back(interval);
            return;
        }
    }
    const unsigned index = groups.size();
    FrontierGroup group;
    group.core = source.core;
    group.sourcePipe = source.pipe;
    group.targetPipe = target.pipe;
    group.block = source.operation->getBlock();
    group.control = control;
    group.guard.assign(source.guard.begin(), source.guard.end());
    group.intervals.push_back(interval);
    groups.push_back(std::move(group));
    blockGroups.push_back(index);
}

LogicalResult appendFrontierCandidates(const FrontierGroup& group, SyncDirectRepairPlan& plan)
{
    llvm::DenseMap<Operation*, unsigned> operationRanks;
    for (auto [rank, operation] : llvm::enumerate(*group.block)) {
        operationRanks[&operation] = rank;
    }
    SmallVector<RepairInterval, 8> intervals(group.intervals.begin(), group.intervals.end());
    for (const RepairInterval& interval : intervals) {
        const bool endpointsRanked = operationRanks.count(interval.source->operation) != 0 &&
                                     operationRanks.count(interval.target->operation) != 0;
        if (!endpointsRanked) {
            return failure();
        }
    }
    llvm::stable_sort(intervals, [&](const RepairInterval& first, const RepairInterval& second) {
        const unsigned firstTarget = operationRanks.lookup(first.target->operation);
        const unsigned secondTarget = operationRanks.lookup(second.target->operation);
        if (firstTarget != secondTarget) {
            return firstTarget < secondTarget;
        }
        const unsigned firstSource = operationRanks.lookup(first.source->operation);
        const unsigned secondSource = operationRanks.lookup(second.source->operation);
        return firstSource != secondSource ? firstSource < secondSource : first.obligation->id < second.obligation->id;
    });

    struct CandidateFrontier {
        const SyncPhase* target = nullptr;
        const SyncPhase* latestSource = nullptr;
        llvm::SmallVector<SyncObligationId, 4> obligations;
    };
    SmallVector<CandidateFrontier, 8> frontiers;
    SmallVector<unsigned, 8> frontierRanks;
    for (const RepairInterval& interval : intervals) {
        const unsigned sourceRank = operationRanks.lookup(interval.source->operation);
        const bool needsFrontier = frontierRanks.empty() || sourceRank >= frontierRanks.back();
        if (needsFrontier) {
            frontiers.push_back({interval.target, nullptr, {}});
            frontierRanks.push_back(operationRanks.lookup(interval.target->operation));
        }
    }
    for (const RepairInterval& interval : intervals) {
        const unsigned sourceRank = operationRanks.lookup(interval.source->operation);
        const unsigned targetRank = operationRanks.lookup(interval.target->operation);
        auto found = llvm::upper_bound(frontierRanks, sourceRank);
        const bool noCoveringFrontier = found == frontierRanks.end() || *found > targetRank;
        if (noCoveringFrontier) {
            return failure();
        }
        CandidateFrontier& frontier = frontiers[std::distance(frontierRanks.begin(), found)];
        frontier.obligations.push_back(interval.obligation->id);
        const bool isLaterSource =
            !frontier.latestSource || operationRanks.lookup(frontier.latestSource->operation) < sourceRank;
        if (isLaterSource) {
            frontier.latestSource = interval.source;
        }
    }

    for (CandidateFrontier& frontier : frontiers) {
        if (!frontier.latestSource || frontier.obligations.empty()) {
            return failure();
        }
        SyncDirectRepairCandidate candidate;
        candidate.id = plan.candidates.size();
        candidate.kind = group.sourcePipe == group.targetPipe ? SyncDirectRepairKind::PipeBarrier :
                                                                SyncDirectRepairKind::DirectedEvent;
        candidate.core = group.core;
        candidate.sourcePipe = group.sourcePipe;
        candidate.targetPipe = group.targetPipe;
        candidate.sourcePhase = frontier.latestSource->id;
        candidate.targetPhase = frontier.target->id;
        candidate.sourceOperation = frontier.latestSource->operation;
        candidate.targetOperation = frontier.target->operation;
        candidate.control = group.control;
        candidate.iteration = {SyncIterationRelationKind::SameIteration, 0};
        candidate.obligations = std::move(frontier.obligations);
        llvm::sort(candidate.obligations);
        plan.candidates.push_back(std::move(candidate));
    }
    return success();
}

struct ExitGroup {
    Operation* section = nullptr;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    llvm::SmallVector<const SyncResidualObligation*, 4> obligations;
};

ExitGroup* findExitGroup(SmallVectorImpl<ExitGroup>& groups, Operation* section)
{
    auto found = llvm::find_if(groups, [&](const ExitGroup& group) { return group.section == section; });
    return found == groups.end() ? nullptr : &*found;
}

void appendExitCandidates(ArrayRef<ExitGroup> groups, SyncDirectRepairPlan& plan)
{
    for (const ExitGroup& group : groups) {
        SyncDirectRepairCandidate candidate;
        candidate.id = plan.candidates.size();
        candidate.kind = SyncDirectRepairKind::ExitBarrier;
        candidate.core = group.core;
        candidate.sourcePipe = PIPE::PIPE_ALL;
        candidate.targetPipe = PIPE::PIPE_ALL;
        candidate.tailSectionOperation = group.section;
        candidate.control = SyncControlRelation::MustExecute;
        candidate.iteration = {SyncIterationRelationKind::SameIteration, 0};
        for (const SyncResidualObligation* obligation : group.obligations) {
            candidate.obligations.push_back(obligation->id);
            candidate.sourcePhase = candidate.sourcePhase == kInvalidSyncId ?
                                        obligation->source :
                                        std::max(candidate.sourcePhase, obligation->source);
        }
        llvm::sort(candidate.obligations);
        plan.candidates.push_back(std::move(candidate));
    }
}

using CompletionKey =
    std::tuple<SyncPhaseId, SyncPhaseId, SyncControlRelation, SyncIterationRelationKind, unsigned, SyncRegionId>;

CompletionKey completionKey(const SyncSelectedCompletion& completion)
{
    return {
        completion.source,
        completion.target,
        completion.control,
        completion.iteration.kind,
        completion.iteration.distance,
        completion.iteration.carrier};
}

LogicalResult allocateDirectRepairEventsImpl(
    const ProtocolSyncTarget& target, ArrayRef<SyncEventReservation> reservations, const StructuredSyncIR* schedule,
    SyncDirectRepairPlan& plan, ProtocolSyncStatistics* statistics)
{
    const bool canAllocate =
        target.supportsDirectRepairEmission() && plan.status == SyncDirectRepairPlanStatus::Ready &&
        llvm::none_of(plan.candidates, [](const SyncDirectRepairCandidate& candidate) {
            return candidate.eventId.has_value();
        });
    if (!canAllocate) {
        return failure();
    }
    SmallVector<SyncEventGeneration, 8> generations;
    SmallVector<SyncDirectRepairCandidate*, 8> eventCandidates;
    for (SyncDirectRepairCandidate& candidate : plan.candidates) {
        if (candidate.kind != SyncDirectRepairKind::DirectedEvent) {
            continue;
        }
        SyncEventGeneration generation;
        generation.id = generations.size();
        generation.kind = SyncEventGenerationKind::DirectRepair;
        generation.core = candidate.core;
        generation.sourcePipe = candidate.sourcePipe;
        generation.targetPipe = candidate.targetPipe;
        generation.setAnchor = candidate.sourceOperation;
        generation.waitAnchor = candidate.targetOperation;
        if (schedule) {
            const SyncPhase* source = schedule->findPhase(candidate.sourcePhase);
            if (!source) {
                return failure();
            }
            generation.guard.assign(source->guard.begin(), source->guard.end());
        }
        generations.push_back(std::move(generation));
        eventCandidates.push_back(&candidate);
    }
    FailureOr<SyncEventAllocationResult> allocation = allocateSyncEventGenerations(target, reservations, generations);
    if (failed(allocation)) {
        return failure();
    }
    if (statistics) {
        recordSyncEventAllocationStatistics(*allocation, *statistics);
    }
    if (allocation->status == SyncEventAllocationStatus::ResourceInfeasible) {
        plan.status = SyncDirectRepairPlanStatus::ResourceInfeasible;
        plan.rejections.push_back(
            {kInvalidSyncId, SyncDirectRepairRejection::EventCapacity,
             "interfering direct event generations exhaust the compiler event pool"});
        return success();
    }
    if (allocation->status != SyncEventAllocationStatus::Allocated) {
        return failure();
    }
    for (auto [candidate, eventId] : llvm::zip_equal(eventCandidates, allocation->eventIds)) {
        candidate->eventId = eventId;
    }
    return success();
}

} // namespace

FailureOr<SyncDirectRepairPlan> mlir::pto::protocol_sync::buildDirectRepairPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    ArrayRef<SyncResidualObligation> obligations, ProtocolSyncStatistics* statistics)
{
    const bool invalidInput = !schedule.isFrozen() || obligations.size() >= kInvalidSyncId;
    if (invalidInput) {
        return failure();
    }
    SyncDirectRepairPlan plan;
    plan.obligationCount = static_cast<std::uint32_t>(obligations.size());
    for (auto [index, obligation] : llvm::enumerate(obligations)) {
        if (obligation.id != index) {
            return failure();
        }
    }

    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.supportsDirectRepairEmission()) {
        plan.status = SyncDirectRepairPlanStatus::Unsupported;
        plan.rejections.push_back({
            kInvalidSyncId, SyncDirectRepairRejection::UnsupportedTarget,
            target.getUnsupportedReason(ProtocolSyncEmissionMode::DirectRepair)});
        for (const SyncResidualObligation& obligation : obligations) {
            plan.uncoveredObligations.push_back(obligation.id);
        }
        if (statistics) {
            statistics->directRepairUncovered += obligations.size();
        }
        return plan;
    }

    SmallVector<FrontierGroup, 8> frontierGroups;
    llvm::DenseMap<Block*, SmallVector<unsigned, 4>> groupsByBlock;
    SmallVector<ExitGroup, 2> exitGroups;
    for (const SyncResidualObligation& obligation : obligations) {
        if (obligation.kind == SyncObligationKind::ExitCompletion) {
            const SyncPhase* phase = schedule.findPhase(obligation.source);
            if (!phase || phase->id != obligation.target || !phase->operation ||
                phase->core == SyncPhysicalCore::Unknown || !hasExactPhaseStage(stages, *phase)) {
                rejectObligation(
                    plan, obligation, SyncDirectRepairRejection::InvalidEndpoint,
                    "exit completion has no exact physical phase endpoint");
                continue;
            }
            const SyncRegion* section = enclosingPhysicalSection(schedule, *phase);
            Operation* sectionOperation = section ? section->operation : nullptr;
            ExitGroup* group = findExitGroup(exitGroups, sectionOperation);
            if (!group) {
                exitGroups.push_back({sectionOperation, phase->core, {&obligation}});
                continue;
            }
            if (group->core != phase->core) {
                rejectObligation(
                    plan, obligation, SyncDirectRepairRejection::MixedPhysicalCores,
                    "one exit barrier cannot cover mixed physical cores");
                continue;
            }
            group->obligations.push_back(&obligation);
            continue;
        }
        if (!supportsDirectCompletion(obligation.kind)) {
            rejectObligation(
                plan, obligation, SyncDirectRepairRejection::UnsupportedObligation,
                "obligation requires a dedicated protocol rather than direct completion repair");
            continue;
        }
        const SyncPhase* source = schedule.findPhase(obligation.source);
        const SyncPhase* destination = schedule.findPhase(obligation.target);
        if (!source || !destination || !source->operation || !destination->operation || source == destination) {
            rejectObligation(
                plan, obligation, SyncDirectRepairRejection::InvalidEndpoint,
                "direct repair requires two distinct physical phase endpoints");
            continue;
        }
        if (source->core == SyncPhysicalCore::Unknown || source->core != destination->core) {
            rejectObligation(
                plan, obligation, SyncDirectRepairRejection::MixedPhysicalCores,
                "direct repair does not synthesize cross-core synchronization");
            continue;
        }
        const bool supportedControl = obligation.control == phaseControlRelation(*source, *destination) &&
                                      obligation.control != SyncControlRelation::Unknown;
        if (!supportedControl) {
            rejectObligation(
                plan, obligation, SyncDirectRepairRejection::UnsupportedControl,
                "direct repair endpoints must have one exact guard class");
            continue;
        }
        const bool recurring = obligation.iteration.kind != SyncIterationRelationKind::SameIteration ||
                               obligation.iteration.distance != 0 || obligation.iteration.carrier != kInvalidSyncId ||
                               !source->iterationDomain.loops.empty() || !destination->iterationDomain.loops.empty();
        if (recurring) {
            rejectObligation(
                plan, obligation, SyncDirectRepairRejection::UnsupportedRecurrence,
                "direct events are not reused across a recurring execution domain");
            continue;
        }
        const bool exactStages = hasExactPhaseStage(stages, *source) && hasExactPhaseStage(stages, *destination);
        if (!exactStages) {
            rejectObligation(
                plan, obligation, SyncDirectRepairRejection::UnsupportedStageShape,
                "direct repair requires one exact phase at each physical operation frontier");
            continue;
        }
        if (!isBefore(source->operation, destination->operation)) {
            rejectObligation(
                plan, obligation, SyncDirectRepairRejection::UnorderedEndpoints,
                "direct repair endpoints do not form a forward interval in one block");
            continue;
        }
        if (source->pipe == destination->pipe) {
            const bool supportedBarrier =
                source->pipe != PIPE::PIPE_S && target.supportsPipeBarrier({source->core, source->pipe});
            if (!supportedBarrier) {
                rejectObligation(
                    plan, obligation, SyncDirectRepairRejection::UnsupportedBarrier,
                    "target has no legal barrier for the same-pipe residual");
                continue;
            }
        } else if (!target.supportsEvent({source->core, source->pipe}, {destination->core, destination->pipe})) {
            rejectObligation(
                plan, obligation, SyncDirectRepairRejection::UnsupportedEventDirection,
                "target has no legal directed event for the residual");
            continue;
        }
        addIntervalToGroups({&obligation, source, destination}, frontierGroups, groupsByBlock);
    }

    for (const FrontierGroup& group : frontierGroups) {
        if (failed(appendFrontierCandidates(group, plan))) {
            return failure();
        }
    }
    appendExitCandidates(exitGroups, plan);

    llvm::BitVector covered(obligations.size());
    for (const SyncDirectRepairCandidate& candidate : plan.candidates) {
        for (SyncObligationId obligation : candidate.obligations) {
            const bool invalidCoverage = obligation >= covered.size() || covered.test(obligation);
            if (invalidCoverage) {
                return failure();
            }
            covered.set(obligation);
        }
        if (statistics) {
            ++statistics->directRepairCandidates;
            const bool sharedCandidate = candidate.obligations.size() > 1;
            if (sharedCandidate) {
                ++statistics->directRepairSharedCandidates;
            }
        }
    }
    llvm::sort(plan.uncoveredObligations);
    llvm::BitVector uncovered(obligations.size());
    for (SyncObligationId obligation : plan.uncoveredObligations) {
        const bool invalidUncovered = obligation >= uncovered.size() || uncovered.test(obligation);
        if (invalidUncovered) {
            return failure();
        }
        uncovered.set(obligation);
    }
    for (const SyncResidualObligation& obligation : obligations) {
        const bool newlyUncovered = !covered.test(obligation.id) && !uncovered.test(obligation.id);
        if (newlyUncovered) {
            rejectObligation(
                plan, obligation, SyncDirectRepairRejection::InternalInvariant,
                "eligible residual was not assigned to a physical candidate");
            uncovered.set(obligation.id);
        }
    }
    llvm::sort(plan.uncoveredObligations);
    if (statistics) {
        statistics->directRepairUncovered += plan.uncoveredObligations.size();
    }
    if (obligations.empty()) {
        plan.status = SyncDirectRepairPlanStatus::Empty;
    } else if (plan.uncoveredObligations.empty()) {
        plan.status = SyncDirectRepairPlanStatus::Ready;
    } else {
        plan.status = SyncDirectRepairPlanStatus::Partial;
    }
    return plan;
}

LogicalResult mlir::pto::protocol_sync::applyDirectRepairCandidates(
    const SyncDirectRepairPlan& plan, ArrayRef<SyncResidualObligation> obligations,
    ArrayRef<SyncDirectCandidateId> selected, SyncSelectedWorld& world)
{
    if (plan.obligationCount != obligations.size()) {
        return failure();
    }
    SyncSelectedWorld staged = world;
    llvm::BitVector selectedCandidates(plan.candidates.size());
    std::set<CompletionKey> knownCompletions;
    for (const SyncSelectedCompletion& completion : staged.completions) {
        knownCompletions.insert(completionKey(completion));
    }
    std::set<SyncPhaseId> knownExitCompletions(staged.exitCompletedPhases.begin(), staged.exitCompletedPhases.end());
    for (SyncDirectCandidateId candidateId : selected) {
        const bool invalidCandidate = candidateId >= plan.candidates.size() || selectedCandidates.test(candidateId);
        if (invalidCandidate) {
            return failure();
        }
        selectedCandidates.set(candidateId);
        const SyncDirectRepairCandidate& candidate = plan.candidates[candidateId];
        if (candidate.id != candidateId) {
            return failure();
        }
        for (SyncObligationId obligationId : candidate.obligations) {
            const bool invalidObligation =
                obligationId >= obligations.size() || obligations[obligationId].id != obligationId;
            if (invalidObligation) {
                return failure();
            }
            const SyncResidualObligation& obligation = obligations[obligationId];
            if (obligation.kind == SyncObligationKind::ExitCompletion) {
                if (knownExitCompletions.insert(obligation.source).second) {
                    staged.exitCompletedPhases.push_back(obligation.source);
                }
                continue;
            }
            if (!supportsDirectCompletion(obligation.kind)) {
                return failure();
            }
            const SyncSelectedCompletion completion{
                obligation.source, obligation.target, obligation.control, obligation.iteration};
            if (knownCompletions.insert(completionKey(completion)).second) {
                staged.completions.push_back(completion);
            }
        }
    }
    world = std::move(staged);
    return success();
}

LogicalResult mlir::pto::protocol_sync::allocateDirectRepairEvents(
    const StructuredSyncIR& schedule, SyncDirectRepairPlan& plan, ProtocolSyncStatistics* statistics)
{
    if (!schedule.isFrozen()) {
        return failure();
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.supportsDirectRepairEmission()) {
        return failure();
    }

    SmallVector<SyncEventReservation, 8> reservations;
    for (const SyncOpSummary& summary : schedule.getSummaries()) {
        reservations.append(summary.eventReservations.begin(), summary.eventReservations.end());
    }
    return allocateDirectRepairEventsImpl(target, reservations, &schedule, plan, statistics);
}

LogicalResult mlir::pto::protocol_sync::allocateDirectRepairEvents(
    const ProtocolSyncTarget& target, ArrayRef<SyncEventReservation> reservations, SyncDirectRepairPlan& plan,
    ProtocolSyncStatistics* statistics)
{
    return allocateDirectRepairEventsImpl(target, reservations, nullptr, plan, statistics);
}
