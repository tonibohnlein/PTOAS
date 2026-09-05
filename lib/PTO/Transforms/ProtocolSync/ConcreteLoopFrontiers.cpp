// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ConcreteLoopFrontiers.cpp - Reconstruct an acknowledged cycle ------===//
// This checker consumes actual lexical operations, never a planner recipe.
// Each forward handoff orders adjacent phases, and the last handoff closes
// the iteration cycle. Unique event keys plus the cycle prove a consuming wait
// precedes every rearm; prime/drain handles zero trips. This proof is only for
// the declared isolated-loop local-completion subset, not arbitrary protocols.

#include "PTO/Transforms/ProtocolSync/LoopFrontierRepair.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using EventKey = std::tuple<PIPE, PIPE, unsigned>;

bool allowedSummary(const SyncOpSummary& summary)
{
    if (summary.provider == SyncSummaryProvider::FixedSynchronization) {
        return summary.operation && isa<SetFlagOp, WaitFlagOp, BarrierOp>(summary.operation);
    }
    return (summary.provider == SyncSummaryProvider::Structural || summary.provider == SyncSummaryProvider::Pipeline) &&
           summary.eventReservations.empty() && summary.suppliedProtocols.empty() && !summary.queue;
}

bool synchronization(Operation& operation)
{
    return isa<SetFlagOp, WaitFlagOp, BarrierOp, SetFlagDynOp, WaitFlagDynOp>(&operation);
}

template <typename OpTy>
std::optional<unsigned> readEvent(Operation* operation, PIPE source, PIPE target, const ProtocolSyncTarget& model)
{
    auto event = dyn_cast_or_null<OpTy>(operation);
    const bool matches = event && event.getSrcPipe().getPipe() == source && event.getDstPipe().getPipe() == target;
    if (!matches) {
        return std::nullopt;
    }
    const unsigned id = static_cast<unsigned>(event.getEventId().getEvent());
    const bool legal = model.supportsEvent({SyncPhysicalCore::Vector, source}, {SyncPhysicalCore::Vector, target}) &&
                       llvm::is_contained(model.getCompilerEventIds(), id);
    return legal ? std::optional<unsigned>(id) : std::nullopt;
}

class CycleChecker {
public:
    CycleChecker(const ProtocolSyncTarget& model, ArrayRef<Operation*> body) : model(model), body(body) {}

    LogicalResult run(ArrayRef<const SyncPhase*> phases, ArrayRef<Operation*> outer, Operation* loop)
    {
        const PIPE first = phases.front()->pipe;
        const PIPE last = phases.back()->pipe;
        const bool eventBackedge = first != last;
        const std::size_t expectedOuter = eventBackedge ? 3 : 2;
        const bool outerShape = outer.size() == expectedOuter && outer[eventBackedge ? 1 : 0] == loop;
        if (!outerShape) {
            return failure();
        }
        std::optional<unsigned> backedgeId;
        if (eventBackedge) {
            backedgeId = readEvent<SetFlagOp>(outer.front(), last, first, model);
            if (!backedgeId) {
                return failure();
            }
        }
        if (failed(acquire(last, first, backedgeId))) {
            return failure();
        }
        for (unsigned index = 0; index < phases.size(); ++index) {
            const bool phaseMatches = take() == phases[index]->operation;
            if (!phaseMatches) {
                return failure();
            }
            const PIPE source = phases[index]->pipe;
            const PIPE target = phases[(index + 1) % phases.size()]->pipe;
            std::optional<unsigned> id;
            if (source != target) {
                id = readEvent<SetFlagOp>(take(), source, target, model);
                if (!id) {
                    return failure();
                }
            }
            const bool finalPhase = index + 1 == phases.size();
            if (finalPhase) {
                if (id != backedgeId) {
                    return failure();
                }
            } else if (failed(acquire(source, target, id))) {
                return failure();
            }
        }
        if (cursor != body.size()) {
            return failure();
        }
        Operation* drain = outer.back();
        if (eventBackedge) {
            return success(readEvent<WaitFlagOp>(drain, last, first, model) == backedgeId);
        }
        return barrier(drain, first);
    }

private:
    Operation* take() { return cursor < body.size() ? body[cursor++] : nullptr; }

    LogicalResult barrier(Operation* operation, PIPE pipe)
    {
        auto fence = dyn_cast_or_null<BarrierOp>(operation);
        const bool legal =
            fence && fence.getPipe().getPipe() == pipe && model.supportsPipeBarrier({SyncPhysicalCore::Vector, pipe});
        return success(legal);
    }

    LogicalResult acquire(PIPE source, PIPE target, std::optional<unsigned> expected)
    {
        Operation* operation = take();
        if (source == target) {
            return barrier(operation, target);
        }
        const auto id = readEvent<WaitFlagOp>(operation, source, target, model);
        if (!id || id != expected || !keys.emplace(source, target, *id).second) {
            return failure();
        }
        return success();
    }

    const ProtocolSyncTarget& model;
    ArrayRef<Operation*> body;
    std::size_t cursor = 0;
    std::set<EventKey> keys;
};

} // namespace

LogicalResult mlir::pto::protocol_sync::verifyConcreteLoopFrontierRepair(const StructuredSyncIR& schedule)
{
    SyncLocalFlowOptions options;
    options.analyzeSingleLoop = true;
    auto local = analyzeLocalMemory(schedule, options);
    const bool supported = succeeded(local) && local->loopStatus == SyncLocalLoopStatus::Complete &&
                           !schedule.getPhases().empty() && llvm::all_of(schedule.getSummaries(), allowedSummary);
    if (!supported) {
        return failure();
    }
    const SyncRegion* carrier = schedule.findRegion(local->loopCarrier);
    auto loop = carrier ? dyn_cast_or_null<scf::ForOp>(carrier->operation) : scf::ForOp();
    if (!loop) {
        return failure();
    }
    llvm::DenseMap<Operation*, const SyncPhase*> phaseAt;
    for (const SyncPhase& phase : schedule.getPhases()) {
        const bool inLoop = phase.operation && phase.operation->getBlock() == loop.getBody();
        if (!inLoop || !phaseAt.try_emplace(phase.operation, &phase).second) {
            return failure();
        }
    }
    SmallVector<const SyncPhase*, 8> phases;
    SmallVector<Operation*, 16> body;
    for (Operation& operation : *loop.getBody()) {
        auto found = phaseAt.find(&operation);
        if (found != phaseAt.end()) {
            phases.push_back(found->second);
            body.push_back(&operation);
        } else if (synchronization(operation)) {
            body.push_back(&operation);
        }
    }
    SmallVector<Operation*, 4> outer;
    for (Operation& operation : *loop->getBlock()) {
        const bool significant = &operation == loop.getOperation() || synchronization(operation);
        if (significant) {
            outer.push_back(&operation);
        }
    }
    const ProtocolSyncTarget model = ProtocolSyncTarget::resolve(schedule.getFunction());
    const bool hasCycle = model.supportsDirectRepairEmission() && !phases.empty();
    if (!hasCycle) {
        return failure();
    }
    CycleChecker checker(model, body);
    return checker.run(phases, outer, loop);
}
