// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
//===- OneShotMaterialization.cpp - Clone, emit, verify, commit ----------===//
#include "PTO/Transforms/ProtocolSync/OneShotProtocol.h"

#include "PTO/IR/PTO.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

#include <chrono>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using OneShotClock = std::chrono::steady_clock;
constexpr StringLiteral kGeneratedAttr = "pto.protocol_sync.generated";
constexpr StringLiteral kProtocolAttr = "pto.protocol_sync.protocol_id";
constexpr StringLiteral kRoleAttr = "pto.protocol_sync.role";
constexpr StringLiteral kTailRole = "tail-drain";
constexpr StringLiteral kSectionLocalTailAttr = "pto.auto_sync_tail_section_local";

std::uint64_t elapsedMicroseconds(OneShotClock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(OneShotClock::now() - start).count();
}

void tagGenerated(Operation* operation, OpBuilder& builder, SyncOneShotProtocolId protocol, StringRef role)
{
    operation->setAttr(kGeneratedAttr, builder.getUnitAttr());
    operation->setAttr(kProtocolAttr, builder.getI32IntegerAttr(static_cast<std::int32_t>(protocol)));
    operation->setAttr(kRoleAttr, builder.getStringAttr(role));
}

void tagTail(BarrierOp barrier, OpBuilder& builder, bool sectionLocal)
{
    barrier->setAttr(kGeneratedAttr, builder.getUnitAttr());
    barrier->setAttr(kRoleAttr, builder.getStringAttr(kTailRole));
    barrier->setAttr("pto.auto_sync_tail_barrier", builder.getUnitAttr());
    barrier->setAttr("pto.auto_sync_tail_hint", builder.getStringAttr("barrier_all"));
    if (sectionLocal) {
        barrier->setAttr(kSectionLocalTailAttr, builder.getUnitAttr());
    }
}

} // namespace

LogicalResult mlir::pto::protocol_sync::materializeOneShotProtocolPlan(
    func::FuncOp clone, const IRMapping& mapping, const SyncOneShotPlan& plan, ProtocolSyncStatistics* statistics)
{
    const bool validPlanShape = plan.status == SyncOneShotPlanStatus::Ready && plan.emitTailBarrier &&
                                !plan.phaseOrder.empty() &&
                                plan.protocols.size() + 1 == plan.phaseOrder.size();
    if (!validPlanShape) {
        return failure();
    }

    const OneShotClock::time_point materializationStart = OneShotClock::now();
    OpBuilder builder(clone.getContext());
    for (const SyncOneShotProtocol& protocol : plan.protocols) {
        Operation* source = mapping.lookupOrNull(protocol.sourceOperation);
        Operation* destination = mapping.lookupOrNull(protocol.targetOperation);
        if (!source || !destination) {
            return failure();
        }
        if (protocol.kind == SyncOneShotProtocolKind::IntrinsicOrder) {
            continue;
        }
        if (protocol.kind == SyncOneShotProtocolKind::PipeBarrier) {
            builder.setInsertionPoint(destination);
            auto barrier = builder.create<BarrierOp>(
                destination->getLoc(), PipeAttr::get(clone.getContext(), protocol.sourcePipe));
            tagGenerated(barrier, builder, protocol.id, "barrier");
            if (statistics) {
                ++statistics->materializationTransitions;
            }
            continue;
        }
        if (protocol.kind != SyncOneShotProtocolKind::DirectedEvent || !protocol.eventId) {
            return failure();
        }
        builder.setInsertionPointAfter(source);
        auto set = builder.create<SetFlagOp>(
            source->getLoc(), PipeAttr::get(clone.getContext(), protocol.sourcePipe),
            PipeAttr::get(clone.getContext(), protocol.targetPipe),
            EventAttr::get(clone.getContext(), static_cast<EVENT>(*protocol.eventId)));
        tagGenerated(set, builder, protocol.id, "event-set");
        builder.setInsertionPoint(destination);
        auto wait = builder.create<WaitFlagOp>(
            destination->getLoc(), PipeAttr::get(clone.getContext(), protocol.sourcePipe),
            PipeAttr::get(clone.getContext(), protocol.targetPipe),
            EventAttr::get(clone.getContext(), static_cast<EVENT>(*protocol.eventId)));
        tagGenerated(wait, builder, protocol.id, "event-wait");
        if (statistics) {
            statistics->materializationTransitions += 2;
        }
    }

    if (plan.tailSectionOperation) {
        Operation* section = mapping.lookupOrNull(plan.tailSectionOperation);
        if (!section || section->getNumRegions() != 1 || !llvm::hasSingleElement(section->getRegion(0))) {
            return failure();
        }
        Block& body = section->getRegion(0).front();
        builder.setInsertionPointToEnd(&body);
        auto barrier = builder.create<BarrierOp>(section->getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
        tagTail(barrier, builder, /*sectionLocal=*/true);
        if (statistics) {
            ++statistics->materializationTransitions;
        }
    } else {
        clone.walk([&](func::ReturnOp operation) {
            builder.setInsertionPoint(operation);
            auto barrier =
                builder.create<BarrierOp>(operation.getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
            tagTail(barrier, builder, /*sectionLocal=*/false);
            if (statistics) {
                ++statistics->materializationTransitions;
            }
        });
    }
    if (statistics) {
        statistics->materializationUs += elapsedMicroseconds(materializationStart);
    }
    return success();
}

LogicalResult mlir::pto::protocol_sync::materializeAndVerifyOneShotProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncOneShotPlan& plan,
    ProtocolSyncStatistics* statistics)
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
    if (!clone) {
        return failure();
    }

    if (failed(materializeOneShotProtocolPlan(clone, mapping, plan, statistics))) {
        return failure();
    }
    const OneShotClock::time_point verificationStart = OneShotClock::now();
    const bool verifiedMaterialization =
        succeeded(verifyOneShotProtocolMaterialization(schedule, stages, clone, mapping, statistics)) &&
        succeeded(mlir::verify(*stagingModule));
    if (!verifiedMaterialization) {
        if (statistics) {
            statistics->verificationUs += elapsedMicroseconds(verificationStart);
        }
        function.emitError("ProtocolSync rejected its staged one-shot materialization; original IR is unchanged");
        return failure();
    }
    if (statistics) {
        statistics->verificationUs += elapsedMicroseconds(verificationStart);
    }
    function.getBody().takeBody(clone.getBody());
    return success();
}

LogicalResult mlir::pto::protocol_sync::materializeAndVerifyOneShotProtocolPlanInPlace(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncOneShotPlan& plan,
    ProtocolSyncStatistics* statistics)
{
    if (!schedule.isFrozen()) {
        return failure();
    }

    func::FuncOp function = schedule.getFunction();
    IRMapping identityMapping;
    function.walk([&](Operation* operation) { identityMapping.map(operation, operation); });
    if (failed(materializeOneShotProtocolPlan(function, identityMapping, plan, statistics))) {
        return failure();
    }

    const OneShotClock::time_point verificationStart = OneShotClock::now();
    if (failed(verifyOneShotProtocolMaterialization(schedule, stages, function, identityMapping, statistics))) {
        if (statistics) {
            statistics->verificationUs += elapsedMicroseconds(verificationStart);
        }
        function.emitError("ProtocolSync rejected its staged one-shot materialization; original IR is unchanged");
        return failure();
    }
    if (statistics) {
        statistics->verificationUs += elapsedMicroseconds(verificationStart);
    }
    return success();
}
