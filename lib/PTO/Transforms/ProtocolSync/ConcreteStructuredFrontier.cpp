// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ConcreteStructuredFrontier.cpp - Reconstruct balanced local packages ===//
// This checker does not consume a planner recipe. Entry/exit pairs are matched
// from actual neighboring instructions on the SAME block/guard as their phase.
// Each package starts and ends with no live token and completed work on V;
// sequence, choice (including empty arms), and loop compose that invariant.
#include "PTO/Transforms/ProtocolSync/StructuredFrontier.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {
bool pair(
    Operation* first, PIPE source, PIPE target, const ProtocolSyncTarget& model, llvm::DenseSet<Operation*>& consumed)
{
    auto set = dyn_cast_or_null<SetFlagOp>(first);
    auto wait = first ? dyn_cast_or_null<WaitFlagOp>(first->getNextNode()) : WaitFlagOp();
    if (!set || !wait) {
        return false;
    }
    const unsigned id = static_cast<unsigned>(set.getEventId().getEvent());
    const bool matches = set.getSrcPipe().getPipe() == source && set.getDstPipe().getPipe() == target &&
                         wait.getSrcPipe().getPipe() == source && wait.getDstPipe().getPipe() == target &&
                         wait.getEventId() == set.getEventId() && llvm::is_contained(model.getCompilerEventIds(), id) &&
                         model.supportsEvent({SyncPhysicalCore::Vector, source}, {SyncPhysicalCore::Vector, target});
    return matches && consumed.insert(set).second && consumed.insert(wait).second;
}
} // namespace

FailureOr<SyncSelectedWorld> mlir::pto::protocol_sync::reconstructStructuredFrontier(const StructuredSyncIR& schedule)
{
    if (!supportsStructuredFrontier(schedule, true)) {
        return failure();
    }
    const auto model = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!model.supportsDirectRepairEmission()) {
        return failure();
    }
    llvm::DenseSet<Operation*> consumed;
    SyncSelectedWorld world;
    for (const auto& phase : schedule.getPhases()) {
        Operation* op = phase.operation;
        if (phase.pipe == PIPE::PIPE_V) {
            auto barrier = dyn_cast_or_null<BarrierOp>(op->getNextNode());
            const bool valid = barrier && barrier.getPipe().getPipe() == PIPE::PIPE_V &&
                               model.supportsPipeBarrier({phase.core, phase.pipe}) && consumed.insert(barrier).second;
            if (!valid) {
                return failure();
            }
        } else {
            Operation* before = op->getPrevNode();
            const bool missingRoundTrip = !before ||
                                          !pair(before->getPrevNode(), PIPE::PIPE_V, phase.pipe, model, consumed) ||
                                          !pair(op->getNextNode(), phase.pipe, PIPE::PIPE_V, model, consumed);
            if (missingRoundTrip) {
                return failure();
            }
        }
        world.acknowledgedPhases.push_back(phase.id);
        world.exitCompletedPhases.push_back(phase.id);
    }
    auto* terminator = schedule.getFunction().getBody().front().getTerminator();
    auto drain = dyn_cast_or_null<BarrierOp>(terminator->getPrevNode());
    const bool terminal = isa<func::ReturnOp>(terminator) && drain && drain.getPipe().getPipe() == PIPE::PIPE_ALL &&
                          consumed.insert(drain).second;
    if (!terminal) {
        return failure();
    }
    for (const auto& summary : schedule.getSummaries()) {
        const bool unconsumed = isFixedSyncOperation(summary.operation) && !consumed.contains(summary.operation);
        if (unconsumed) {
            return failure();
        }
    }
    return world;
}
