// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- DirectRepairMaterialization.cpp - Emit and verify direct repairs -===//

#include "PTO/Transforms/ProtocolSync/DirectRepair.h"

#include "PTO/IR/PTO.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

#include <chrono>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using DirectRepairClock = std::chrono::steady_clock;
constexpr StringLiteral kGeneratedAttr = "pto.protocol_sync.generated";
constexpr StringLiteral kCandidateAttr = "pto.protocol_sync.direct_candidate_id";
constexpr StringLiteral kRoleAttr = "pto.protocol_sync.role";
constexpr StringLiteral kSectionLocalTailAttr = "pto.auto_sync_tail_section_local";

std::uint64_t elapsedMicroseconds(DirectRepairClock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(DirectRepairClock::now() - start).count();
}

void tagGenerated(Operation* operation, OpBuilder& builder, SyncDirectCandidateId candidate, StringRef role)
{
    operation->setAttr(kGeneratedAttr, builder.getUnitAttr());
    operation->setAttr(kCandidateAttr, builder.getI32IntegerAttr(static_cast<std::int32_t>(candidate)));
    operation->setAttr(kRoleAttr, builder.getStringAttr(role));
}

LogicalResult materializeDirectRepairPlanOnClone(
    func::FuncOp clone, const IRMapping& mapping, const SyncDirectRepairPlan& plan, ProtocolSyncStatistics* statistics)
{
    if (plan.status != SyncDirectRepairPlanStatus::Ready || plan.candidates.empty()) {
        return failure();
    }
    const DirectRepairClock::time_point start = DirectRepairClock::now();
    OpBuilder builder(clone.getContext());
    for (const SyncDirectRepairCandidate& candidate : plan.candidates) {
        if (candidate.kind == SyncDirectRepairKind::PipeBarrier) {
            Operation* destination = mapping.lookupOrNull(candidate.targetOperation);
            if (!destination) {
                return failure();
            }
            builder.setInsertionPoint(destination);
            auto barrier = builder.create<BarrierOp>(
                destination->getLoc(), PipeAttr::get(clone.getContext(), candidate.sourcePipe));
            tagGenerated(barrier, builder, candidate.id, "direct-barrier");
            if (statistics) {
                ++statistics->materializationTransitions;
            }
            continue;
        }
        if (candidate.kind == SyncDirectRepairKind::DirectedEvent) {
            Operation* source = mapping.lookupOrNull(candidate.sourceOperation);
            Operation* destination = mapping.lookupOrNull(candidate.targetOperation);
            if (!source || !destination || !candidate.eventId) {
                return failure();
            }
            builder.setInsertionPointAfter(source);
            auto set = builder.create<SetFlagOp>(
                source->getLoc(), PipeAttr::get(clone.getContext(), candidate.sourcePipe),
                PipeAttr::get(clone.getContext(), candidate.targetPipe),
                EventAttr::get(clone.getContext(), static_cast<EVENT>(*candidate.eventId)));
            tagGenerated(set, builder, candidate.id, "direct-event-set");
            builder.setInsertionPoint(destination);
            auto wait = builder.create<WaitFlagOp>(
                destination->getLoc(), PipeAttr::get(clone.getContext(), candidate.sourcePipe),
                PipeAttr::get(clone.getContext(), candidate.targetPipe),
                EventAttr::get(clone.getContext(), static_cast<EVENT>(*candidate.eventId)));
            tagGenerated(wait, builder, candidate.id, "direct-event-wait");
            if (statistics) {
                statistics->materializationTransitions += 2;
            }
            continue;
        }
        if (candidate.kind != SyncDirectRepairKind::ExitBarrier) {
            return failure();
        }
        if (candidate.tailSectionOperation) {
            Operation* section = mapping.lookupOrNull(candidate.tailSectionOperation);
            const bool validSection =
                section && section->getNumRegions() == 1 && llvm::hasSingleElement(section->getRegion(0));
            if (!validSection) {
                return failure();
            }
            Block& body = section->getRegion(0).front();
            builder.setInsertionPointToEnd(&body);
            auto barrier =
                builder.create<BarrierOp>(section->getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
            tagGenerated(barrier, builder, candidate.id, "direct-tail-drain");
            barrier->setAttr(kSectionLocalTailAttr, builder.getUnitAttr());
            if (statistics) {
                ++statistics->materializationTransitions;
            }
            continue;
        }
        clone.walk([&](func::ReturnOp operation) {
            builder.setInsertionPoint(operation);
            auto barrier =
                builder.create<BarrierOp>(operation.getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
            tagGenerated(barrier, builder, candidate.id, "direct-tail-drain");
            if (statistics) {
                ++statistics->materializationTransitions;
            }
        });
    }
    if (statistics) {
        statistics->materializationUs += elapsedMicroseconds(start);
    }
    return success();
}

} // namespace

LogicalResult mlir::pto::protocol_sync::materializeAndVerifyDirectRepairPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    ArrayRef<SyncResidualObligation> obligations, const SyncDirectRepairPlan& plan, ProtocolSyncStatistics* statistics)
{
    if (!schedule.isFrozen()) {
        return failure();
    }
    func::FuncOp function = schedule.getFunction();
    ModuleOp sourceModule = function->getParentOfType<ModuleOp>();
    if (!sourceModule) {
        return failure();
    }
    IRMapping mapping;
    OwningOpRef<ModuleOp> stagingModule = cast<ModuleOp>(sourceModule->clone(mapping));
    auto clone = dyn_cast_or_null<func::FuncOp>(mapping.lookupOrNull(function.getOperation()));
    if (!clone || failed(materializeDirectRepairPlanOnClone(clone, mapping, plan, statistics))) {
        return failure();
    }
    const DirectRepairClock::time_point verificationStart = DirectRepairClock::now();
    const bool verified =
        succeeded(verifyDirectRepairMaterialization(schedule, stages, obligations, clone, mapping, plan, statistics)) &&
        succeeded(mlir::verify(*stagingModule));
    if (statistics) {
        statistics->verificationUs += elapsedMicroseconds(verificationStart);
    }
    if (!verified) {
        function.emitError("ProtocolSync rejected its staged direct repair; original IR is unchanged");
        return failure();
    }
    function.getBody().takeBody(clone.getBody());
    return success();
}

LogicalResult mlir::pto::protocol_sync::materializeAndVerifyDirectRepairPlanInDisposableModule(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    ArrayRef<SyncResidualObligation> obligations, const SyncDirectRepairPlan& plan, ProtocolSyncStatistics* statistics)
{
    if (!schedule.isFrozen()) {
        return failure();
    }
    func::FuncOp function = schedule.getFunction();
    IRMapping identityMapping;
    function.walk([&](Operation* operation) { identityMapping.map(operation, operation); });
    if (failed(materializeDirectRepairPlanOnClone(function, identityMapping, plan, statistics))) {
        return failure();
    }
    const DirectRepairClock::time_point verificationStart = DirectRepairClock::now();
    const LogicalResult verified =
        verifyDirectRepairMaterialization(schedule, stages, obligations, function, identityMapping, plan, statistics);
    if (statistics) {
        statistics->verificationUs += elapsedMicroseconds(verificationStart);
    }
    if (failed(verified)) {
        function.emitError("ProtocolSync rejected direct repair in disposable staging IR");
        return failure();
    }
    return success();
}
