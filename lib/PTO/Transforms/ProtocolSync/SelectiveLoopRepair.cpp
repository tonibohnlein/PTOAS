// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Direct local hazard handoffs; never an adjacent-phase total-order cycle.
#include "PTO/Transforms/ProtocolSync/SelectiveLoopRepair.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "PTO/Transforms/ProtocolSync/RecurringEventLifetime.h"
#include "PTO/Transforms/ProtocolSync/GMAliasPolicy.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {
bool alreadySupplied(
    const SyncSelectedWorld& world, unsigned count, unsigned source, unsigned target, unsigned distance)
{
    llvm::BitVector visited(2 * count);
    SmallVector<std::pair<unsigned, unsigned>, 16> pending{{source, 0}};
    while (!pending.empty()) {
        const auto [phase, iteration] = pending.pop_back_val();
        if (phase == target && iteration == distance) {
            return true;
        }
        const unsigned node = iteration * count + phase;
        if (visited.test(node)) {
            continue;
        }
        visited.set(node);
        for (const auto& edge : world.completions) {
            const unsigned next = iteration + edge.iteration.distance;
            if (edge.source == phase && next <= distance) {
                pending.push_back({edge.target, next});
            }
        }
    }
    return false;
}

template <typename Op>
void emitEvent(OpBuilder& builder, const SyncLoopFrontierEdge& edge, Location location)
{
    builder.create<Op>(
        location, PipeAttr::get(builder.getContext(), edge.sourcePipe),
        PipeAttr::get(builder.getContext(), edge.targetPipe),
        EventAttr::get(builder.getContext(), static_cast<EVENT>(*edge.eventId)));
}

SyncRecurringEventProof proveTokens(const SyncSelectiveLoopPlan& plan)
{
    SmallVector<SyncRecurringEventChannel, 16> channels;
    const unsigned stride = 2 * plan.edges.size() + 1;
    for (auto [id, edge] : llvm::enumerate(plan.edges)) {
        if (!edge.isEvent()) {
            continue;
        }
        const auto& completion = edge.completion;
        const bool carried = completion.iteration.distance == 1;
        channels.push_back(
            {static_cast<unsigned>(edge.placement.sourcePipe), static_cast<unsigned>(edge.placement.targetPipe),
             completion.source * stride + static_cast<unsigned>(plan.edges.size() + id),
             completion.target * stride + static_cast<unsigned>(id), completion.iteration.distance, carried, carried});
    }
    return proveRecurringEventLifetimes(channels);
}

bool fitsConcreteBudget(const StructuredSyncIR& schedule, const SyncSelectiveLoopPlan& plan)
{
    unsigned operations = 0;
    schedule.getFunction().walk([&](Operation*) { ++operations; });
    const auto barriers = llvm::count_if(plan.edges, [](const auto& edge) { return !edge.isEvent(); });
    // Conservative pre-emission bound: each body cut contributes at most two
    // full phase-pair products. Never publish a candidate that can exceed the
    // concrete checker's expansion budget after materialization.
    const std::uint64_t phases = schedule.getPhases().size();
    const std::uint64_t expansion = 2 * plan.edges.size() * phases * phases;
    return plan.edges.size() - barriers <= kSelectiveLoopMaximumChannels && barriers <= kSelectiveLoopMaximumBarriers &&
           expansion <= kSelectiveLoopMaximumCompletions &&
           operations + 4 * plan.edges.size() + 1 <= kSelectiveLoopMaximumOperations;
}
} // namespace

FailureOr<SyncSelectiveLoopAttempt> mlir::pto::protocol_sync::buildSelectiveLoopRepair(const StructuredSyncIR& schedule)
{
    const auto carrier = findIsolatedSyncLoop(schedule, false);
    if (!carrier) {
        return SyncSelectiveLoopAttempt{
            SyncSelectiveLoopStatus::Unsupported, "requires an isolated unconditional loop"};
    }
    SyncLocalFlowOptions options;
    options.analyzeSingleLoop = true;
    auto local = analyzeLocalMemory(schedule, options);
    if (failed(local)) {
        return failure();
    }
    if (local->loopStatus != SyncLocalLoopStatus::Complete) {
        return SyncSelectiveLoopAttempt{
            local->loopStatus == SyncLocalLoopStatus::LimitExceeded ? SyncSelectiveLoopStatus::AnalysisLimit :
                                                                      SyncSelectiveLoopStatus::Unsupported,
            "local loop memory analysis did not complete"};
    }
    std::set<std::tuple<SyncPhaseId, SyncPhaseId, unsigned>> pairs;
    for (const auto& requirement : local->requirements) {
        const bool within = requirement.iteration.kind == SyncIterationRelationKind::SameIteration &&
                            requirement.iteration.distance == 0;
        const bool carried = requirement.iteration.kind == SyncIterationRelationKind::LoopCarried &&
                             requirement.iteration.distance == 1 && requirement.iteration.carrier == *carrier;
        if (!within && !carried) {
            return SyncSelectiveLoopAttempt{SyncSelectiveLoopStatus::Unsupported, "unsupported occurrence relation"};
        }
        pairs.emplace(requirement.source, requirement.target, carried ? 1 : 0);
    }
    // Nonlocal census is deliberately conservative. Alias uncertainty changes
    // hazard existence, never turns completion into publication.
    for (const auto& first : schedule.getAccesses()) {
        for (const auto& second : schedule.getAccesses()) {
            const bool gm = first.storage.space == AddressSpace::GM && second.storage.space == AddressSpace::GM;
            const bool readRead = first.mode == SyncAccessMode::Read && second.mode == SyncAccessMode::Read;
            if (!gm || readRead || haveDisjointGMArgumentRoots(schedule, first, second)) {
                continue;
            }
            if (first.mode != SyncAccessMode::Read && second.mode != SyncAccessMode::Write) {
                return SyncSelectiveLoopAttempt{SyncSelectiveLoopStatus::Unsupported, "GM publication is unqualified"};
            }
            if (first.phase < second.phase) {
                pairs.emplace(first.phase, second.phase, 0);
            }
            pairs.emplace(first.phase, second.phase, 1);
        }
    }
    const bool bounded = pairs.size() <= 256;
    if (!bounded) {
        return SyncSelectiveLoopAttempt{SyncSelectiveLoopStatus::AnalysisLimit, "requirement pair budget exceeded"};
    }
    const auto target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.supportsDirectRepairEmission()) {
        return SyncSelectiveLoopAttempt{SyncSelectiveLoopStatus::Unsupported, "target does not support direct repair"};
    }
    SyncSelectiveLoopPlan plan;
    plan.loop = schedule.findRegion(*carrier)->operation;
    SmallVector<std::tuple<SyncPhaseId, SyncPhaseId, unsigned>, 16> ordered(pairs.begin(), pairs.end());
    const auto samePipe = [&](const auto& pair) {
        return schedule.findPhase(std::get<0>(pair))->pipe == schedule.findPhase(std::get<1>(pair))->pipe;
    };
    llvm::stable_sort(ordered, [&](const auto& a, const auto& b) { return samePipe(a) < samePipe(b); });
    // First establish exact cross-lane readiness/reclamation. A same-pipe
    // carried WAW may already follow through those handoffs; blindly adding a
    // body barrier could drain an independent current-iteration load as well.
    for (auto [sourceId, targetId, distance] : ordered) {
        const auto& source = *schedule.findPhase(sourceId);
        const auto& destination = *schedule.findPhase(targetId);
        SyncLoopFrontierEdge placement{source.operation, destination.operation, source.pipe, destination.pipe, {}};
        if (source.pipe == destination.pipe) {
            if (alreadySupplied(plan.world, schedule.getPhases().size(), sourceId, targetId, distance)) {
                continue;
            }
            if (!target.supportsPipeBarrier({source.core, source.pipe})) {
                return SyncSelectiveLoopAttempt{SyncSelectiveLoopStatus::Unsupported, "unqualified same-pipe barrier"};
            }
        } else {
            if (!target.supportsEvent({source.core, source.pipe}, {destination.core, destination.pipe})) {
                return SyncSelectiveLoopAttempt{SyncSelectiveLoopStatus::Unsupported, "unqualified event direction"};
            }
        }
        const SyncIterationRelation relation{
            distance == 0 ? SyncIterationRelationKind::SameIteration : SyncIterationRelationKind::LoopCarried, distance,
            distance == 0 ? kInvalidSyncId : *carrier};
        SyncSelectedCompletion completion{sourceId, targetId, SyncControlRelation::MustExecute, relation};
        plan.edges.push_back({placement, completion});
        plan.world.completions.push_back(completion);
    }
    for (const auto& phase : schedule.getPhases()) {
        plan.world.exitCompletedPhases.push_back(phase.id);
    }
    if (!fitsConcreteBudget(schedule, plan)) {
        return SyncSelectiveLoopAttempt{
            SyncSelectiveLoopStatus::AnalysisLimit, "concrete verification budget exceeded"};
    }
    const auto proof = proveTokens(plan);
    if (proof.status != SyncRecurringEventStatus::Proven) {
        return SyncSelectiveLoopAttempt{
            proof.status == SyncRecurringEventStatus::AnalysisLimit ? SyncSelectiveLoopStatus::AnalysisLimit :
                                                                      SyncSelectiveLoopStatus::UnprovedTokenContract,
            "recurring event consumption contract not proved", proof};
    }
    if (failed(verifySelectiveLoopMemory(schedule, plan.world))) {
        return failure();
    }
    plan.candidateCountBeforeDeletion = plan.edges.size();
    for (unsigned index = plan.edges.size(); index > 0; --index) {
        if (!plan.edges[index - 1].isEvent()) {
            continue;
        }
        auto trial = plan;
        trial.edges.erase(trial.edges.begin() + index - 1);
        trial.world.completions.erase(trial.world.completions.begin() + index - 1);
        ++plan.deletionAttempts;
        if (proveTokens(trial).status == SyncRecurringEventStatus::Proven &&
            succeeded(verifySelectiveLoopMemory(schedule, trial.world))) {
            plan.edges = std::move(trial.edges);
            plan.world = std::move(trial.world);
            ++plan.deletionRemoved;
        }
    }
    SyncSelectiveLoopAttempt attempt;
    attempt.status = SyncSelectiveLoopStatus::Ready;
    attempt.tokenProof = proveTokens(plan);
    attempt.plan = std::move(plan);
    return attempt;
}

LogicalResult mlir::pto::protocol_sync::materializeSelectiveLoopRepair(
    func::FuncOp function, const SyncSelectiveLoopPlan& plan)
{
    auto loop = dyn_cast_or_null<scf::ForOp>(plan.loop);
    const bool assigned =
        llvm::all_of(plan.edges, [](const auto& edge) { return edge.isEvent() == edge.placement.eventId.has_value(); });
    const bool valid = loop && loop->getParentOp() == function && assigned &&
                       proveTokens(plan).status == SyncRecurringEventStatus::Proven;
    if (!valid) {
        return failure();
    }
    OpBuilder builder(function.getContext());
    builder.setInsertionPoint(loop);
    for (const auto& edge : plan.edges) {
        if (edge.placement.eventId && edge.completion.iteration.distance == 1) {
            emitEvent<SetFlagOp>(builder, edge.placement, loop.getLoc());
        }
    }
    // Materialize once per anchor, preserving the exact event order certified
    // above. Repeated setInsertionPointAfter would reverse shared-anchor sets.
    SmallVector<Operation*, 16> body;
    for (Operation& operation : *loop.getBody()) {
        body.push_back(&operation);
    }
    for (Operation* operation : body) {
        builder.setInsertionPoint(operation);
        for (const auto& edge : plan.edges) {
            if (edge.placement.target != operation) {
                continue;
            }
            if (edge.placement.eventId) {
                emitEvent<WaitFlagOp>(builder, edge.placement, operation->getLoc());
            } else {
                builder.create<BarrierOp>(
                    operation->getLoc(), PipeAttr::get(function.getContext(), edge.placement.targetPipe));
            }
        }
        builder.setInsertionPointAfter(operation);
        for (const auto& edge : plan.edges) {
            if (edge.placement.source == operation && edge.placement.eventId) {
                emitEvent<SetFlagOp>(builder, edge.placement, operation->getLoc());
            }
        }
    }
    builder.setInsertionPointAfter(loop);
    for (const auto& edge : plan.edges) {
        if (edge.placement.eventId && edge.completion.iteration.distance == 1) {
            emitEvent<WaitFlagOp>(builder, edge.placement, loop.getLoc());
        }
    }
    builder.setInsertionPoint(function.getBody().front().getTerminator());
    builder.create<BarrierOp>(function.getLoc(), PipeAttr::get(function.getContext(), PIPE::PIPE_ALL));
    return success();
}
