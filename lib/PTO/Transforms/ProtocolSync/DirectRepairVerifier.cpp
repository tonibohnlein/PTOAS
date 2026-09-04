// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- DirectRepairVerifier.cpp - Verify logical and emitted repairs ----===//

#include "PTO/Transforms/ProtocolSync/DirectRepair.h"

#include "PTO/IR/PTO.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

constexpr StringLiteral kGeneratedAttr = "pto.protocol_sync.generated";
constexpr StringLiteral kCandidateAttr = "pto.protocol_sync.direct_candidate_id";
constexpr StringLiteral kRoleAttr = "pto.protocol_sync.role";
constexpr StringLiteral kProtocolKindAttr = "pto.protocol_sync.protocol_kind";
constexpr StringLiteral kSectionLocalTailAttr = "pto.auto_sync_tail_section_local";

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

bool isBeforeOrEqual(Operation* source, Operation* target) { return source == target || isBefore(source, target); }

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

bool isReserved(const StructuredSyncIR& schedule, PIPE source, PIPE target, unsigned eventId)
{
    return llvm::any_of(schedule.getSummaries(), [&](const SyncOpSummary& summary) {
        return llvm::any_of(summary.eventReservations, [&](const SyncEventReservation& reservation) {
            return reservation.source == source && reservation.target == target &&
                   llvm::is_contained(reservation.eventIds, eventId);
        });
    });
}

std::uint64_t eventKey(const SyncDirectRepairCandidate& candidate, unsigned eventId)
{
    return (static_cast<std::uint64_t>(candidate.core) << 24) |
           (static_cast<std::uint64_t>(candidate.sourcePipe) << 16) |
           (static_cast<std::uint64_t>(candidate.targetPipe) << 8) | eventId;
}

LogicalResult verifyExitCandidate(
    const StructuredSyncIR& schedule, ArrayRef<SyncResidualObligation> obligations,
    const SyncDirectRepairCandidate& candidate)
{
    if (candidate.targetPhase != kInvalidSyncId || candidate.sourceOperation || candidate.targetOperation ||
        candidate.sourcePipe != PIPE::PIPE_ALL || candidate.targetPipe != PIPE::PIPE_ALL || candidate.eventId ||
        candidate.obligations.empty() || candidate.control != SyncControlRelation::MustExecute ||
        candidate.iteration.kind != SyncIterationRelationKind::SameIteration || candidate.iteration.distance != 0 ||
        candidate.iteration.carrier != kInvalidSyncId) {
        return failure();
    }
    SyncPhaseId maximumSource = kInvalidSyncId;
    for (SyncObligationId id : candidate.obligations) {
        if (id >= obligations.size()) {
            return failure();
        }
        const SyncResidualObligation& obligation = obligations[id];
        const SyncPhase* phase = schedule.findPhase(obligation.source);
        if (obligation.id != id || obligation.kind != SyncObligationKind::ExitCompletion || !phase ||
            obligation.target != phase->id || phase->core != candidate.core) {
            return failure();
        }
        const SyncRegion* section = enclosingPhysicalSection(schedule, *phase);
        Operation* expectedSection = section ? section->operation : nullptr;
        if (expectedSection != candidate.tailSectionOperation) {
            return failure();
        }
        maximumSource = maximumSource == kInvalidSyncId ? phase->id : std::max(maximumSource, phase->id);
    }
    return success(candidate.sourcePhase == maximumSource);
}

LogicalResult verifyFrontierCandidate(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    ArrayRef<SyncResidualObligation> obligations, const ProtocolSyncTarget& target,
    const SyncDirectRepairCandidate& candidate, llvm::DenseSet<std::uint64_t>& allocatedEvents)
{
    const SyncPhase* frontierSource = schedule.findPhase(candidate.sourcePhase);
    const SyncPhase* frontierTarget = schedule.findPhase(candidate.targetPhase);
    const bool invalidFrontier = !frontierSource || !frontierTarget || !frontierSource->operation ||
                                 !frontierTarget->operation || candidate.sourceOperation != frontierSource->operation ||
                                 candidate.targetOperation != frontierTarget->operation ||
                                 candidate.tailSectionOperation || candidate.core == SyncPhysicalCore::Unknown ||
                                 candidate.obligations.empty() || !hasExactPhaseStage(stages, *frontierSource) ||
                                 !hasExactPhaseStage(stages, *frontierTarget) ||
                                 !isBefore(frontierSource->operation, frontierTarget->operation);
    if (invalidFrontier) {
        return failure();
    }
    const bool samePipe = candidate.sourcePipe == candidate.targetPipe;
    if (candidate.kind == SyncDirectRepairKind::PipeBarrier) {
        if (!samePipe || candidate.eventId || candidate.sourcePipe == PIPE::PIPE_S ||
            !target.supportsPipeBarrier({candidate.core, candidate.sourcePipe})) {
            return failure();
        }
    } else if (candidate.kind == SyncDirectRepairKind::DirectedEvent) {
        if (samePipe ||
            !target.supportsEvent({candidate.core, candidate.sourcePipe}, {candidate.core, candidate.targetPipe})) {
            return failure();
        }
        if (candidate.eventId) {
            const unsigned eventId = *candidate.eventId;
            const bool validId = llvm::is_contained(target.getCompilerEventIds(), eventId) &&
                                 !isReserved(schedule, candidate.sourcePipe, candidate.targetPipe, eventId) &&
                                 allocatedEvents.insert(eventKey(candidate, eventId)).second;
            if (!validId) {
                return failure();
            }
        }
    } else {
        return failure();
    }

    Operation* latestSource = nullptr;
    Operation* earliestTarget = nullptr;
    for (SyncObligationId id : candidate.obligations) {
        if (id >= obligations.size()) {
            return failure();
        }
        const SyncResidualObligation& obligation = obligations[id];
        const SyncPhase* source = schedule.findPhase(obligation.source);
        const SyncPhase* destination = schedule.findPhase(obligation.target);
        const bool invalidObligation =
            obligation.id != id || !supportsDirectCompletion(obligation.kind) || !source || !destination ||
            !source->operation || !destination->operation || source->core != candidate.core ||
            destination->core != candidate.core || source->pipe != candidate.sourcePipe ||
            destination->pipe != candidate.targetPipe || obligation.control != candidate.control ||
            phaseControlRelation(*source, *destination) != candidate.control ||
            obligation.iteration.kind != SyncIterationRelationKind::SameIteration ||
            obligation.iteration.distance != 0 || obligation.iteration.carrier != kInvalidSyncId ||
            !source->iterationDomain.loops.empty() || !destination->iterationDomain.loops.empty() ||
            source->operation->getBlock() != frontierSource->operation->getBlock() ||
            destination->operation->getBlock() != frontierTarget->operation->getBlock() ||
            !sameGuard(source->guard, frontierSource->guard) ||
            !isBeforeOrEqual(source->operation, frontierSource->operation) ||
            !isBeforeOrEqual(frontierTarget->operation, destination->operation);
        if (invalidObligation) {
            return failure();
        }
        if (!latestSource || isBefore(latestSource, source->operation)) {
            latestSource = source->operation;
        }
        if (!earliestTarget || isBefore(destination->operation, earliestTarget)) {
            earliestTarget = destination->operation;
        }
    }
    return success(
        latestSource == frontierSource->operation && earliestTarget == frontierTarget->operation &&
        frontierSource->core == candidate.core && frontierTarget->core == candidate.core &&
        frontierSource->pipe == candidate.sourcePipe && frontierTarget->pipe == candidate.targetPipe &&
        phaseControlRelation(*frontierSource, *frontierTarget) == candidate.control);
}

struct ConcreteCandidate {
    Operation* barrier = nullptr;
    Operation* set = nullptr;
    Operation* wait = nullptr;
    SmallVector<Operation*, 2> tails;
};

LogicalResult collectConcreteCandidates(
    func::FuncOp clone, unsigned candidateCount, llvm::DenseMap<SyncDirectCandidateId, ConcreteCandidate>& records)
{
    bool malformed = false;
    clone.walk([&](Operation* operation) {
        const bool fixed = isFixedSyncOperation(operation);
        const bool generated = operation->hasAttr(kGeneratedAttr);
        if (!fixed && !generated) {
            return;
        }
        const bool belongsToOtherProtocol =
            !operation->hasAttr(kCandidateAttr) && operation->hasAttr(kProtocolKindAttr);
        if (belongsToOtherProtocol) {
            return;
        }
        auto candidateAttr = operation->getAttrOfType<IntegerAttr>(kCandidateAttr);
        auto role = operation->getAttrOfType<StringAttr>(kRoleAttr);
        const bool validCandidate = fixed && generated && operation->hasAttrOfType<UnitAttr>(kGeneratedAttr) &&
                                    candidateAttr && candidateAttr.getInt() >= 0 &&
                                    static_cast<std::uint64_t>(candidateAttr.getInt()) < candidateCount && role;
        if (!validCandidate) {
            malformed = true;
            return;
        }
        ConcreteCandidate& record = records[static_cast<SyncDirectCandidateId>(candidateAttr.getInt())];
        const StringRef roleValue = role.getValue();
        if (roleValue == "direct-barrier") {
            if (record.barrier || !isa<BarrierOp>(operation)) {
                malformed = true;
            } else {
                record.barrier = operation;
            }
        } else if (roleValue == "direct-event-set") {
            if (record.set || !isa<SetFlagOp>(operation)) {
                malformed = true;
            } else {
                record.set = operation;
            }
        } else if (roleValue == "direct-event-wait") {
            if (record.wait || !isa<WaitFlagOp>(operation)) {
                malformed = true;
            } else {
                record.wait = operation;
            }
        } else if (roleValue == "direct-tail-drain") {
            if (!isa<BarrierOp>(operation)) {
                malformed = true;
            } else {
                record.tails.push_back(operation);
            }
        } else {
            malformed = true;
        }
    });
    return failure(malformed);
}

bool onlyGeneratedBetween(Operation* first, Operation* second)
{
    if (!isBefore(first, second)) {
        return false;
    }
    for (Operation* operation = first->getNextNode(); operation && operation != second;
         operation = operation->getNextNode()) {
        if (!operation->hasAttrOfType<UnitAttr>(kGeneratedAttr)) {
            return false;
        }
    }
    return true;
}

LogicalResult verifyConcreteFrontier(
    const StructuredSyncIR& schedule, const ProtocolSyncTarget& target, const SyncDirectRepairCandidate& candidate,
    const ConcreteCandidate& concrete, const IRMapping& mapping, llvm::DenseSet<std::uint64_t>& allocatedEvents)
{
    Operation* source = mapping.lookupOrNull(schedule.findPhase(candidate.sourcePhase)->operation);
    Operation* destination = mapping.lookupOrNull(schedule.findPhase(candidate.targetPhase)->operation);
    if (!source || !destination) {
        return failure();
    }
    if (candidate.kind == SyncDirectRepairKind::PipeBarrier) {
        auto barrier = dyn_cast_or_null<BarrierOp>(concrete.barrier);
        return success(
            barrier && !concrete.set && !concrete.wait && concrete.tails.empty() &&
            barrier.getPipe().getPipe() == candidate.sourcePipe && isBefore(source, concrete.barrier) &&
            onlyGeneratedBetween(concrete.barrier, destination));
    }
    auto set = dyn_cast_or_null<SetFlagOp>(concrete.set);
    auto wait = dyn_cast_or_null<WaitFlagOp>(concrete.wait);
    if (!candidate.eventId || !set || !wait || concrete.barrier || !concrete.tails.empty()) {
        return failure();
    }
    const unsigned eventId = static_cast<unsigned>(set.getEventId().getEvent());
    const bool valid =
        set.getSrcPipe().getPipe() == candidate.sourcePipe && set.getDstPipe().getPipe() == candidate.targetPipe &&
        wait.getSrcPipe().getPipe() == candidate.sourcePipe && wait.getDstPipe().getPipe() == candidate.targetPipe &&
        wait.getEventId().getEvent() == set.getEventId().getEvent() && eventId == *candidate.eventId &&
        target.supportsEvent({candidate.core, candidate.sourcePipe}, {candidate.core, candidate.targetPipe}) &&
        llvm::is_contained(target.getCompilerEventIds(), eventId) &&
        !isReserved(schedule, candidate.sourcePipe, candidate.targetPipe, eventId) &&
        allocatedEvents.insert(eventKey(candidate, eventId)).second && onlyGeneratedBetween(source, concrete.set) &&
        isBefore(concrete.set, concrete.wait) && onlyGeneratedBetween(concrete.wait, destination);
    return success(valid);
}

LogicalResult verifyConcreteTail(
    func::FuncOp clone, const SyncDirectRepairCandidate& candidate, const ConcreteCandidate& concrete,
    const IRMapping& mapping)
{
    if (concrete.barrier || concrete.set || concrete.wait || concrete.tails.empty()) {
        return failure();
    }
    if (candidate.tailSectionOperation) {
        Operation* section = mapping.lookupOrNull(candidate.tailSectionOperation);
        const bool validSection = section && section->getNumRegions() == 1 &&
                                  llvm::hasSingleElement(section->getRegion(0)) && concrete.tails.size() == 1;
        if (!validSection) {
            return failure();
        }
        Operation* tail = concrete.tails.front();
        Block& body = section->getRegion(0).front();
        auto barrier = dyn_cast<BarrierOp>(tail);
        return success(
            &body.back() == tail && barrier && barrier.getPipe().getPipe() == PIPE::PIPE_ALL &&
            tail->hasAttrOfType<UnitAttr>(kSectionLocalTailAttr));
    }
    unsigned returns = 0;
    bool invalid = false;
    clone.walk([&](func::ReturnOp operation) {
        ++returns;
        Operation* previous = operation->getPrevNode();
        auto barrier = dyn_cast_or_null<BarrierOp>(previous);
        const bool validTail = barrier && llvm::is_contained(concrete.tails, previous) &&
                               barrier.getPipe().getPipe() == PIPE::PIPE_ALL &&
                               !previous->hasAttr(kSectionLocalTailAttr);
        if (!validTail) {
            invalid = true;
        }
    });
    return success(!invalid && returns != 0 && concrete.tails.size() == returns);
}

struct VerifiedFrontierGroup {
    SyncDirectRepairKind kind = SyncDirectRepairKind::PipeBarrier;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE sourcePipe = PIPE::PIPE_UNASSIGNED;
    PIPE targetPipe = PIPE::PIPE_UNASSIGNED;
    Block* block = nullptr;
    SyncControlRelation control = SyncControlRelation::Unknown;
    llvm::SmallVector<SyncControlAtom, 2> guard;
    llvm::SmallVector<const SyncDirectRepairCandidate*, 4> candidates;
};

bool matchesVerifiedGroup(
    const VerifiedFrontierGroup& group, const SyncDirectRepairCandidate& candidate, ArrayRef<SyncControlAtom> guard)
{
    return group.kind == candidate.kind && group.core == candidate.core && group.sourcePipe == candidate.sourcePipe &&
           group.targetPipe == candidate.targetPipe && group.block == candidate.sourceOperation->getBlock() &&
           group.control == candidate.control && sameGuard(group.guard, guard);
}

void addVerifiedCandidateToGroups(
    const StructuredSyncIR& schedule, const SyncDirectRepairCandidate& candidate,
    SmallVectorImpl<VerifiedFrontierGroup>& groups, llvm::DenseMap<Block*, SmallVector<unsigned, 4>>& groupsByBlock)
{
    const SyncPhase* source = schedule.findPhase(candidate.sourcePhase);
    SmallVector<unsigned, 4>& blockGroups = groupsByBlock[candidate.sourceOperation->getBlock()];
    for (unsigned index : blockGroups) {
        if (matchesVerifiedGroup(groups[index], candidate, source->guard)) {
            groups[index].candidates.push_back(&candidate);
            return;
        }
    }
    const unsigned index = groups.size();
    VerifiedFrontierGroup group;
    group.kind = candidate.kind;
    group.core = candidate.core;
    group.sourcePipe = candidate.sourcePipe;
    group.targetPipe = candidate.targetPipe;
    group.block = candidate.sourceOperation->getBlock();
    group.control = candidate.control;
    group.guard.assign(source->guard.begin(), source->guard.end());
    group.candidates.push_back(&candidate);
    groups.push_back(std::move(group));
    blockGroups.push_back(index);
}

LogicalResult verifyGroupHasNoMergeableFrontiers(const VerifiedFrontierGroup& group)
{
    llvm::DenseMap<Operation*, unsigned> operationRanks;
    for (auto [rank, operation] : llvm::enumerate(*group.block)) {
        operationRanks[&operation] = rank;
    }
    SmallVector<const SyncDirectRepairCandidate*, 8> candidates(group.candidates.begin(), group.candidates.end());
    llvm::stable_sort(candidates, [&](const auto* first, const auto* second) {
        const unsigned firstSource = operationRanks.lookup(first->sourceOperation);
        const unsigned secondSource = operationRanks.lookup(second->sourceOperation);
        if (firstSource != secondSource) {
            return firstSource < secondSource;
        }
        return operationRanks.lookup(first->targetOperation) < operationRanks.lookup(second->targetOperation);
    });
    std::optional<unsigned> furthestTarget;
    for (const SyncDirectRepairCandidate* candidate : candidates) {
        const bool ranked = operationRanks.count(candidate->sourceOperation) != 0 &&
                            operationRanks.count(candidate->targetOperation) != 0;
        if (!ranked) {
            return failure();
        }
        const unsigned sourceRank = operationRanks.lookup(candidate->sourceOperation);
        const unsigned targetRank = operationRanks.lookup(candidate->targetOperation);
        if (furthestTarget && sourceRank < *furthestTarget) {
            return failure();
        }
        furthestTarget = furthestTarget ? std::max(*furthestTarget, targetRank) : targetRank;
    }
    return success();
}

LogicalResult verifyCandidatesCannotShareFrontier(
    const StructuredSyncIR& schedule, ArrayRef<SyncDirectRepairCandidate> candidates)
{
    bool flatExitSeen = false;
    llvm::DenseSet<Operation*> sectionExits;
    SmallVector<VerifiedFrontierGroup, 8> groups;
    llvm::DenseMap<Block*, SmallVector<unsigned, 4>> groupsByBlock;
    for (const SyncDirectRepairCandidate& candidate : candidates) {
        if (candidate.kind != SyncDirectRepairKind::ExitBarrier) {
            addVerifiedCandidateToGroups(schedule, candidate, groups, groupsByBlock);
            continue;
        }
        if (!candidate.tailSectionOperation) {
            if (flatExitSeen) {
                return failure();
            }
            flatExitSeen = true;
            continue;
        }
        if (!sectionExits.insert(candidate.tailSectionOperation).second) {
            return failure();
        }
    }
    for (const VerifiedFrontierGroup& group : groups) {
        if (failed(verifyGroupHasNoMergeableFrontiers(group))) {
            return failure();
        }
    }
    return success();
}

} // namespace

LogicalResult mlir::pto::protocol_sync::verifyDirectRepairPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    ArrayRef<SyncResidualObligation> obligations, const SyncDirectRepairPlan& plan, ProtocolSyncStatistics* statistics)
{
    const bool validInput =
        schedule.isFrozen() && plan.obligationCount == obligations.size() && obligations.size() < kInvalidSyncId;
    if (!validInput) {
        return failure();
    }
    for (auto [index, obligation] : llvm::enumerate(obligations)) {
        if (obligation.id != index) {
            return failure();
        }
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.isSupported()) {
        const bool canonicalUncovered = llvm::all_of(llvm::enumerate(plan.uncoveredObligations), [](auto indexed) {
            return indexed.value() == indexed.index();
        });
        const bool validUnsupported =
            plan.status == SyncDirectRepairPlanStatus::Unsupported && plan.candidates.empty() &&
            plan.uncoveredObligations.size() == obligations.size() && canonicalUncovered &&
            plan.rejections.size() == 1 && plan.rejections.front().obligation == kInvalidSyncId &&
            plan.rejections.front().reason == SyncDirectRepairRejection::UnsupportedTarget &&
            plan.rejections.front().detail == target.getUnsupportedReason();
        return success(validUnsupported);
    }
    if (plan.status == SyncDirectRepairPlanStatus::Unsupported) {
        return failure();
    }

    llvm::BitVector covered(obligations.size());
    llvm::DenseSet<std::uint64_t> allocatedEvents;
    for (auto [index, candidate] : llvm::enumerate(plan.candidates)) {
        if (candidate.id != index || !llvm::is_sorted(candidate.obligations)) {
            return failure();
        }
        const LogicalResult verified =
            candidate.kind == SyncDirectRepairKind::ExitBarrier ?
                verifyExitCandidate(schedule, obligations, candidate) :
                verifyFrontierCandidate(schedule, stages, obligations, target, candidate, allocatedEvents);
        if (failed(verified)) {
            return failure();
        }
        for (SyncObligationId id : candidate.obligations) {
            const bool invalidCoverage = id >= covered.size() || covered.test(id);
            if (invalidCoverage) {
                return failure();
            }
            covered.set(id);
            if (statistics) {
                ++statistics->verifierTransitions;
            }
        }
    }
    if (failed(verifyCandidatesCannotShareFrontier(schedule, plan.candidates))) {
        return failure();
    }

    if (!llvm::is_sorted(plan.uncoveredObligations)) {
        return failure();
    }
    llvm::BitVector uncovered(obligations.size());
    for (SyncObligationId id : plan.uncoveredObligations) {
        const bool invalidUncovered = id >= uncovered.size() || uncovered.test(id) || covered.test(id);
        if (invalidUncovered) {
            return failure();
        }
        uncovered.set(id);
    }
    const bool completePartition = covered.count() + uncovered.count() == obligations.size();
    if (!completePartition) {
        return failure();
    }
    const bool empty = obligations.empty() && plan.candidates.empty() && plan.rejections.empty() &&
                       plan.status == SyncDirectRepairPlanStatus::Empty;
    const bool ready = !obligations.empty() && uncovered.none() && plan.rejections.empty() &&
                       plan.status == SyncDirectRepairPlanStatus::Ready;
    const bool partial = !uncovered.none() && plan.status == SyncDirectRepairPlanStatus::Partial;
    const bool resourceInfeasible = !obligations.empty() && uncovered.none() && !plan.candidates.empty() &&
                                    plan.status == SyncDirectRepairPlanStatus::ResourceInfeasible;
    const bool validStatus = empty || ready || partial || resourceInfeasible;
    if (!validStatus) {
        return failure();
    }
    llvm::BitVector rejectedObligations(obligations.size());
    unsigned planLevelRejections = 0;
    for (const SyncDirectRepairPlanRejection& rejection : plan.rejections) {
        if (rejection.reason == SyncDirectRepairRejection::None || rejection.detail.empty()) {
            return failure();
        }
        if (rejection.obligation == kInvalidSyncId) {
            ++planLevelRejections;
            continue;
        }
        const bool invalidRejection = rejection.obligation >= uncovered.size() ||
                                      !uncovered.test(rejection.obligation) ||
                                      rejectedObligations.test(rejection.obligation);
        if (invalidRejection) {
            return failure();
        }
        rejectedObligations.set(rejection.obligation);
    }
    if (partial && (planLevelRejections != 0 || rejectedObligations != uncovered)) {
        return failure();
    }
    if (resourceInfeasible) {
        const bool oneCapacityRejection = plan.rejections.size() == 1 && planLevelRejections == 1 &&
                                          plan.rejections.front().reason == SyncDirectRepairRejection::EventCapacity;
        const bool hasDirectedEvent = llvm::any_of(plan.candidates, [](const SyncDirectRepairCandidate& candidate) {
            return candidate.kind == SyncDirectRepairKind::DirectedEvent;
        });
        const bool allUnallocated = llvm::all_of(
            plan.candidates, [](const SyncDirectRepairCandidate& candidate) { return !candidate.eventId; });
        if (!oneCapacityRejection || !hasDirectedEvent || !allUnallocated) {
            return failure();
        }
    }
    if (ready) {
        const auto directed = llvm::make_filter_range(plan.candidates, [](const SyncDirectRepairCandidate& candidate) {
            return candidate.kind == SyncDirectRepairKind::DirectedEvent;
        });
        const bool anyAllocated = llvm::any_of(directed, [](const auto& candidate) { return candidate.eventId; });
        const bool allAllocated = llvm::all_of(directed, [](const auto& candidate) { return candidate.eventId; });
        if (anyAllocated != allAllocated) {
            return failure();
        }
    }
    return success();
}

LogicalResult mlir::pto::protocol_sync::verifyDirectRepairMaterialization(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    ArrayRef<SyncResidualObligation> obligations, func::FuncOp clone, const IRMapping& mapping,
    const SyncDirectRepairPlan& plan, ProtocolSyncStatistics* statistics)
{
    if (plan.status != SyncDirectRepairPlanStatus::Ready ||
        failed(verifyDirectRepairPlan(schedule, stages, obligations, plan, statistics))) {
        return failure();
    }
    llvm::DenseMap<SyncDirectCandidateId, ConcreteCandidate> records;
    const bool completeRecords = succeeded(collectConcreteCandidates(clone, plan.candidates.size(), records)) &&
                                 records.size() == plan.candidates.size();
    if (!completeRecords) {
        return failure();
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    llvm::DenseSet<std::uint64_t> allocatedEvents;
    for (const SyncDirectRepairCandidate& candidate : plan.candidates) {
        auto found = records.find(candidate.id);
        if (found == records.end()) {
            return failure();
        }
        const LogicalResult verified =
            candidate.kind == SyncDirectRepairKind::ExitBarrier ?
                verifyConcreteTail(clone, candidate, found->second, mapping) :
                verifyConcreteFrontier(schedule, target, candidate, found->second, mapping, allocatedEvents);
        if (failed(verified)) {
            return failure();
        }
        if (statistics) {
            ++statistics->verifierTransitions;
        }
    }
    return success();
}
