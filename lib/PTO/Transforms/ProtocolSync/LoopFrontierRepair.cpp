// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LoopFrontierRepair.cpp - Conservative acknowledged loop handoffs ----===//

#include "PTO/Transforms/ProtocolSync/LoopFrontierRepair.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using EventKey = std::tuple<PIPE, PIPE, unsigned>;

bool ordinarySummary(const SyncOpSummary& summary)
{
    return (summary.provider == SyncSummaryProvider::Structural || summary.provider == SyncSummaryProvider::Pipeline) &&
           summary.eventReservations.empty() && summary.suppliedProtocols.empty() && !summary.queue;
}

bool reserved(PIPE source, PIPE target, unsigned id, ArrayRef<SyncEventReservation> reservations)
{
    return llvm::any_of(reservations, [&](const SyncEventReservation& reservation) {
        return reservation.source == source && reservation.target == target &&
               llvm::is_contained(reservation.eventIds, id);
    });
}

template <typename OpTy>
void emitEvent(OpBuilder& builder, const SyncLoopFrontierEdge& edge, Location location)
{
    builder.create<OpTy>(
        location, PipeAttr::get(builder.getContext(), edge.sourcePipe),
        PipeAttr::get(builder.getContext(), edge.targetPipe),
        EventAttr::get(builder.getContext(), static_cast<EVENT>(*edge.eventId)));
}

void emitAcquire(OpBuilder& builder, const SyncLoopFrontierEdge& edge, Location location)
{
    if (edge.eventId) {
        emitEvent<WaitFlagOp>(builder, edge, location);
    } else {
        builder.create<BarrierOp>(location, PipeAttr::get(builder.getContext(), edge.targetPipe));
    }
}

} // namespace

FailureOr<SyncLoopFrontierPlan> mlir::pto::protocol_sync::buildLoopFrontierRepairPlan(
    const StructuredSyncIR& schedule, ArrayRef<SyncEventReservation> reservations, bool includeBoundaries)
{
    SyncLoopFrontierPlan plan;
    plan.detail = "needs one isolated ordinary loop with bounded local footprints and no fixed synchronization";
    SyncLocalFlowOptions options;
    options.analyzeSingleLoop = true;
    auto local = analyzeLocalMemory(schedule, options);
    if (failed(local)) {
        return failure();
    }
    const bool supported = local->loopStatus == SyncLocalLoopStatus::Complete && !schedule.getPhases().empty() &&
                           llvm::all_of(schedule.getSummaries(), ordinarySummary);
    if (!supported) {
        return plan;
    }
    const SyncRegion* carrier = schedule.findRegion(local->loopCarrier);
    auto loop = carrier ? dyn_cast_or_null<scf::ForOp>(carrier->operation) : scf::ForOp();
    if (!loop) {
        return failure();
    }
    SmallVector<const SyncPhase*, 8> phases;
    SmallVector<const SyncPhase*, 8> outer;
    for (const SyncPhase& phase : schedule.getPhases()) {
        const bool outerPhase = includeBoundaries && phase.operation && phase.operation->getBlock() == loop->getBlock();
        if (outerPhase) {
            outer.push_back(&phase);
            continue;
        }
        const bool isolated = phase.operation && phase.operation->getBlock() == loop.getBody() &&
                              (phases.empty() || phases.back()->operation != phase.operation);
        if (!isolated) {
            return plan;
        }
        phases.push_back(&phase);
    }
    if (phases.empty()) {
        return plan;
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!target.supportsDirectRepairEmission()) {
        plan.detail = target.getUnsupportedReason();
        return plan;
    }
    std::set<EventKey> occupied;
    SmallVector<SyncLoopFrontierEdge, 8> edges;
    for (unsigned index = 0; index < phases.size(); ++index) {
        const SyncPhase& source = *phases[index];
        const SyncPhase& destination = *phases[(index + 1) % phases.size()];
        SyncLoopFrontierEdge edge{source.operation, destination.operation, source.pipe, destination.pipe, std::nullopt};
        if (source.pipe == destination.pipe) {
            if (!target.supportsPipeBarrier({source.core, source.pipe})) {
                return plan;
            }
        } else {
            if (!target.supportsEvent({source.core, source.pipe}, {destination.core, destination.pipe})) {
                return plan;
            }
            for (unsigned id : target.getCompilerEventIds()) {
                const bool available = !reserved(source.pipe, destination.pipe, id, reservations) &&
                                       occupied.emplace(source.pipe, destination.pipe, id).second;
                if (available) {
                    edge.eventId = id;
                    break;
                }
            }
            if (!edge.eventId) {
                plan.status = SyncLoopFrontierStatus::ResourceInfeasible;
                plan.detail = "no unreserved event ID for a recurring frontier generation";
                return plan;
            }
        }
        edges.push_back(edge);
    }
    SmallVector<SyncLoopFrontierEdge, 8> boundaries;
    if (includeBoundaries) {
        SmallVector<Operation*, 8> nodes;
        std::map<Operation*, PIPE> pipes;
        for (const SyncPhase* phase : outer) {
            if (phase->operation->isBeforeInBlock(loop)) {
                nodes.push_back(phase->operation);
            }
            pipes.emplace(phase->operation, phase->pipe);
        }
        nodes.push_back(loop);
        for (const SyncPhase* phase : outer) {
            if (loop->isBeforeInBlock(phase->operation)) {
                nodes.push_back(phase->operation);
            }
        }
        for (unsigned index = 1; index < nodes.size(); ++index) {
            Operation* source = nodes[index - 1];
            Operation* destination = nodes[index];
            const PIPE src = source == loop ? phases.front()->pipe : pipes.at(source);
            const PIPE dst = destination == loop ? phases.back()->pipe : pipes.at(destination);
            SyncLoopFrontierEdge edge{source, destination, src, dst, std::nullopt};
            if (src == dst) {
                if (!target.supportsPipeBarrier({SyncPhysicalCore::Vector, src})) {
                    return plan;
                }
            } else {
                if (!target.supportsEvent({SyncPhysicalCore::Vector, src}, {SyncPhysicalCore::Vector, dst})) {
                    return plan;
                }
                for (unsigned id : target.getCompilerEventIds()) {
                    const bool available =
                        !reserved(src, dst, id, reservations) && occupied.emplace(src, dst, id).second;
                    if (available) {
                        edge.eventId = id;
                        break;
                    }
                }
                if (!edge.eventId) {
                    plan.status = SyncLoopFrontierStatus::ResourceInfeasible;
                    plan.detail = "no unreserved event ID for an outer frontier";
                    return plan;
                }
            }
            boundaries.push_back(edge);
        }
    }
    plan.loop = loop;
    plan.edges = std::move(edges);
    plan.boundaryEdges = std::move(boundaries);
    plan.includeBoundaries = includeBoundaries;
    plan.requirements = local->requirements;
    plan.requirementCount = local->requirements.size();
    plan.status = SyncLoopFrontierStatus::Ready;
    plan.detail = "local completion cycle only; full F verification is still required";
    return plan;
}

LogicalResult mlir::pto::protocol_sync::materializeLoopFrontierRepair(
    func::FuncOp clone, const IRMapping& mapping, const SyncLoopFrontierPlan& plan)
{
    if (!clone || plan.status != SyncLoopFrontierStatus::Ready || plan.edges.empty()) {
        return failure();
    }
    auto loop = dyn_cast_or_null<scf::ForOp>(mapping.lookupOrNull(plan.loop));
    const bool inClone = loop && loop->getParentOfType<func::FuncOp>() == clone;
    if (!inClone) {
        return failure();
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(clone);
    SmallVector<SyncLoopFrontierEdge, 8> mapped;
    std::set<EventKey> keys;
    for (const SyncLoopFrontierEdge& edge : plan.edges) {
        Operation* source = mapping.lookupOrNull(edge.source);
        Operation* destination = mapping.lookupOrNull(edge.target);
        const bool inside =
            source && destination && source->getBlock() == loop.getBody() && destination->getBlock() == loop.getBody();
        if (!inside) {
            return failure();
        }
        const bool samePipe = edge.sourcePipe == edge.targetPipe;
        const bool legal =
            samePipe ?
                !edge.eventId && target.supportsPipeBarrier({SyncPhysicalCore::Vector, edge.sourcePipe}) :
                edge.eventId && llvm::is_contained(target.getCompilerEventIds(), *edge.eventId) &&
                    target.supportsEvent(
                        {SyncPhysicalCore::Vector, edge.sourcePipe}, {SyncPhysicalCore::Vector, edge.targetPipe}) &&
                    keys.emplace(edge.sourcePipe, edge.targetPipe, *edge.eventId).second;
        if (!legal) {
            return failure();
        }
        mapped.push_back({source, destination, edge.sourcePipe, edge.targetPipe, edge.eventId});
    }
    for (unsigned index = 0; index < mapped.size(); ++index) {
        const auto& edge = mapped[index];
        const bool cycle = edge.target == mapped[(index + 1) % mapped.size()].source;
        const bool forward = index + 1 == mapped.size() || edge.source->isBeforeInBlock(edge.target);
        if (!cycle || !forward) {
            return failure();
        }
    }
    OpBuilder builder(clone.getContext());
    const auto& backedge = mapped.back();
    if (backedge.eventId) {
        builder.setInsertionPoint(loop);
        emitEvent<SetFlagOp>(builder, backedge, loop.getLoc());
    }
    for (const auto& edge : mapped) {
        if (edge.eventId) {
            builder.setInsertionPointAfter(edge.source);
            emitEvent<SetFlagOp>(builder, edge, edge.source->getLoc());
        }
        builder.setInsertionPoint(edge.target);
        emitAcquire(builder, edge, edge.target->getLoc());
    }
    builder.setInsertionPointAfter(loop);
    emitAcquire(builder, backedge, loop.getLoc());
    Operation* loopDrain = loop->getNextNode();
    for (const auto& edge : plan.boundaryEdges) {
        Operation* source = mapping.lookupOrNull(edge.source);
        Operation* destination = mapping.lookupOrNull(edge.target);
        const bool validBoundary = source && destination && source->getBlock() == loop->getBlock() &&
                                   destination->getBlock() == loop->getBlock() && source->isBeforeInBlock(destination);
        if (!validBoundary) {
            return failure();
        }
        // Insert entry acquisition before the prime, not merely before scf.for.
        Operation* acquireAnchor = destination == loop && backedge.eventId ? loop->getPrevNode() : destination;
        if (edge.eventId) {
            builder.setInsertionPointAfter(source == loop ? loopDrain : source);
            emitEvent<SetFlagOp>(builder, edge, source->getLoc());
        }
        builder.setInsertionPoint(acquireAnchor);
        emitAcquire(builder, edge, destination->getLoc());
    }
    if (plan.includeBoundaries) {
        builder.setInsertionPoint(clone.getBody().front().getTerminator());
        builder.create<BarrierOp>(clone.getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
    }
    return success();
}
