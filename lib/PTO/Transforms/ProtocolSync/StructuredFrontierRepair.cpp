// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StructuredFrontierRepair.cpp - Balanced region-interface baseline ----===//
// Every phase starts/ends at the V completion interface. Non-V phases use an
// acknowledged round trip; V phases end with a named barrier. An entire package
// is branch-local. Sequential packages can reuse their keys only because the
// reverse acquisition precedes the next forward publication.
#include "PTO/Transforms/ProtocolSync/StructuredFrontier.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {
template <typename Op>
void event(OpBuilder& builder, Location loc, PIPE source, PIPE target, unsigned id)
{
    builder.create<Op>(
        loc, PipeAttr::get(builder.getContext(), source), PipeAttr::get(builder.getContext(), target),
        EventAttr::get(builder.getContext(), static_cast<EVENT>(id)));
}
} // namespace

FailureOr<std::optional<SyncStructuredFrontierPlan>> mlir::pto::protocol_sync::buildStructuredFrontierPlan(
    const StructuredSyncIR& schedule)
{
    if (!supportsStructuredFrontier(schedule, false)) {
        return std::optional<SyncStructuredFrontierPlan>{};
    }
    const auto target = ProtocolSyncTarget::resolve(schedule.getFunction());
    const bool unsupportedTarget = !target.supportsDirectRepairEmission() || target.getCompilerEventIds().empty() ||
                                   !target.supportsPipeBarrier({SyncPhysicalCore::Vector, PIPE::PIPE_V});
    if (unsupportedTarget) {
        return std::optional<SyncStructuredFrontierPlan>{};
    }
    SyncLocalFlowOptions options;
    options.analyzeStructured = true;
    auto local = analyzeLocalMemory(schedule, options);
    if (failed(local)) {
        return failure();
    }
    if (local->structuredStatus != SyncLocalStructuredStatus::Complete) {
        return std::optional<SyncStructuredFrontierPlan>{};
    }
    SyncStructuredFrontierPlan plan;
    for (const auto& phase : schedule.getPhases()) {
        if (phase.pipe != PIPE::PIPE_V &&
            (!target.supportsEvent({phase.core, PIPE::PIPE_V}, {phase.core, phase.pipe}) ||
             !target.supportsEvent({phase.core, phase.pipe}, {phase.core, PIPE::PIPE_V}))) {
            return std::optional<SyncStructuredFrontierPlan>{};
        }
        const unsigned id = phase.pipe == PIPE::PIPE_V ? 0 : target.getCompilerEventIds().front();
        plan.phases.push_back({phase.id, phase.operation, phase.pipe, id, id});
    }
    plan.requirements = local->requirements;
    return std::optional<SyncStructuredFrontierPlan>(std::move(plan));
}

LogicalResult mlir::pto::protocol_sync::materializeStructuredFrontier(
    func::FuncOp clone, const IRMapping& mapping, const SyncStructuredFrontierPlan& plan)
{
    if (!clone || !llvm::hasSingleElement(clone.getBody())) {
        return failure();
    }
    OpBuilder builder(clone.getContext());
    for (const auto& phase : plan.phases) {
        Operation* op = mapping.lookupOrNull(phase.operation);
        const bool invalidClone = !op || op->getParentOfType<func::FuncOp>() != clone;
        if (invalidClone) {
            return failure();
        }
        if (phase.pipe != PIPE::PIPE_V) {
            builder.setInsertionPoint(op);
            event<SetFlagOp>(builder, op->getLoc(), PIPE::PIPE_V, phase.pipe, phase.acquireId);
            event<WaitFlagOp>(builder, op->getLoc(), PIPE::PIPE_V, phase.pipe, phase.acquireId);
        }
        builder.setInsertionPointAfter(op);
        if (phase.pipe == PIPE::PIPE_V) {
            builder.create<BarrierOp>(op->getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_V));
        } else {
            event<SetFlagOp>(builder, op->getLoc(), phase.pipe, PIPE::PIPE_V, phase.releaseId);
            event<WaitFlagOp>(builder, op->getLoc(), phase.pipe, PIPE::PIPE_V, phase.releaseId);
        }
    }
    builder.setInsertionPoint(clone.getBody().front().getTerminator());
    builder.create<BarrierOp>(clone.getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
    return success();
}
