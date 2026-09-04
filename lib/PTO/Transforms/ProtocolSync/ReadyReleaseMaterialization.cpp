// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ReadyReleaseMaterialization.cpp - Atomic recurring emission ------===//

#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

#include <chrono>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using ReadyReleaseClock = std::chrono::steady_clock;
constexpr StringLiteral kGeneratedAttr = "pto.protocol_sync.generated";
constexpr StringLiteral kProtocolAttr = "pto.protocol_sync.protocol_id";
constexpr StringLiteral kProtocolKindAttr = "pto.protocol_sync.protocol_kind";
constexpr StringLiteral kReadyReleaseKind = "ready-release";
constexpr StringLiteral kRoleAttr = "pto.protocol_sync.role";
constexpr StringLiteral kLaneAttr = "pto.protocol_sync.logical_lane";

std::uint64_t elapsedMicroseconds(ReadyReleaseClock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(ReadyReleaseClock::now() - start).count();
}

void tagAction(Operation* operation, OpBuilder& builder, StringRef role, std::optional<unsigned> lane)
{
    operation->setAttr(kGeneratedAttr, builder.getUnitAttr());
    operation->setAttr(kProtocolAttr, builder.getI32IntegerAttr(0));
    operation->setAttr(kProtocolKindAttr, builder.getStringAttr(kReadyReleaseKind));
    operation->setAttr(kRoleAttr, builder.getStringAttr(role));
    if (lane) {
        operation->setAttr(kLaneAttr, builder.getI32IntegerAttr(static_cast<std::int32_t>(*lane)));
    }
}

template <typename OpTy>
Operation* createEvent(
    OpBuilder& builder, Location location, func::FuncOp clone, PIPE source, PIPE target, unsigned eventId,
    StringRef role, std::optional<unsigned> lane = std::nullopt)
{
    auto operation = builder.create<OpTy>(
        location, PipeAttr::get(clone.getContext(), source), PipeAttr::get(clone.getContext(), target),
        EventAttr::get(clone.getContext(), static_cast<EVENT>(eventId)));
    tagAction(operation, builder, role, lane);
    return operation.getOperation();
}

} // namespace

LogicalResult mlir::pto::protocol_sync::materializeReadyReleaseProtocolPlan(
    func::FuncOp clone, const IRMapping& mapping, const SyncReadyReleasePlan& plan, ProtocolSyncStatistics* statistics)
{
    const bool validPlan = plan.status == SyncReadyReleasePlanStatus::Ready && plan.capacity == 1 &&
                           plan.lanes.size() == 1 && plan.lanes.front().logicalLane == 0 &&
                           plan.lanes.front().readyEventId && plan.lanes.front().releaseEventId &&
                           plan.tokenCertificate.zeroTripSafe && plan.tokenCertificate.oneTripSafe &&
                           plan.tokenCertificate.oddEvenSafe && plan.tokenCertificate.steadyStateStable;
    if (!validPlan) {
        return failure();
    }

    Operation* loopOperation = mapping.lookupOrNull(plan.loopOperation);
    Operation* producer = mapping.lookupOrNull(plan.producerOperation);
    Operation* consumer = mapping.lookupOrNull(plan.consumerOperation);
    auto loop = dyn_cast_or_null<scf::ForOp>(loopOperation);
    const bool validMapping = loop && producer && consumer && producer->getBlock() == loop.getBody() &&
                              consumer->getBlock() == loop.getBody() && producer->isBeforeInBlock(consumer);
    if (!validMapping) {
        return failure();
    }

    const ReadyReleaseClock::time_point start = ReadyReleaseClock::now();
    const SyncReadyReleaseLane& lane = plan.lanes.front();
    OpBuilder builder(clone.getContext());
    builder.setInsertionPoint(loop);
    createEvent<SetFlagOp>(
        builder, loop.getLoc(), clone, plan.consumerPipe, plan.producerPipe, *lane.releaseEventId, "release-prime-set",
        lane.logicalLane);

    builder.setInsertionPoint(producer);
    createEvent<WaitFlagOp>(
        builder, producer->getLoc(), clone, plan.consumerPipe, plan.producerPipe, *lane.releaseEventId,
        "release-body-wait", lane.logicalLane);
    builder.setInsertionPointAfter(producer);
    createEvent<SetFlagOp>(
        builder, producer->getLoc(), clone, plan.producerPipe, plan.consumerPipe, *lane.readyEventId, "ready-body-set",
        lane.logicalLane);

    builder.setInsertionPoint(consumer);
    createEvent<WaitFlagOp>(
        builder, consumer->getLoc(), clone, plan.producerPipe, plan.consumerPipe, *lane.readyEventId, "ready-body-wait",
        lane.logicalLane);
    builder.setInsertionPointAfter(consumer);
    createEvent<SetFlagOp>(
        builder, consumer->getLoc(), clone, plan.consumerPipe, plan.producerPipe, *lane.releaseEventId,
        "release-body-set", lane.logicalLane);

    builder.setInsertionPointAfter(loop);
    createEvent<WaitFlagOp>(
        builder, loop.getLoc(), clone, plan.consumerPipe, plan.producerPipe, *lane.releaseEventId, "release-drain-wait",
        lane.logicalLane);

    if (statistics) {
        statistics->materializationTransitions += 6;
        statistics->materializationUs += elapsedMicroseconds(start);
    }
    return success();
}

LogicalResult mlir::pto::protocol_sync::materializeAndVerifyReadyReleaseProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncReadyReleasePlan& plan,
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
    if (!clone || failed(materializeReadyReleaseProtocolPlan(clone, mapping, plan, statistics))) {
        return failure();
    }

    const ReadyReleaseClock::time_point start = ReadyReleaseClock::now();
    const bool verified =
        succeeded(verifyReadyReleaseProtocolMaterialization(schedule, stages, clone, mapping, statistics)) &&
        succeeded(mlir::verify(*stagingModule));
    if (statistics) {
        statistics->verificationUs += elapsedMicroseconds(start);
    }
    if (!verified) {
        function.emitError("ProtocolSync rejected its staged ReadyRelease materialization; original IR is unchanged");
        return failure();
    }
    function.getBody().takeBody(clone.getBody());
    return success();
}

LogicalResult mlir::pto::protocol_sync::materializeAndVerifyReadyReleaseProtocolPlanInPlace(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncReadyReleasePlan& plan,
    ProtocolSyncStatistics* statistics)
{
    if (!schedule.isFrozen()) {
        return failure();
    }
    func::FuncOp function = schedule.getFunction();
    IRMapping identityMapping;
    function.walk([&](Operation* operation) { identityMapping.map(operation, operation); });
    if (failed(materializeReadyReleaseProtocolPlan(function, identityMapping, plan, statistics))) {
        return failure();
    }

    const ReadyReleaseClock::time_point start = ReadyReleaseClock::now();
    const LogicalResult verified =
        verifyReadyReleaseProtocolMaterialization(schedule, stages, function, identityMapping, statistics);
    if (statistics) {
        statistics->verificationUs += elapsedMicroseconds(start);
    }
    if (failed(verified)) {
        function.emitError("ProtocolSync rejected its staged ReadyRelease materialization; original IR is unchanged");
        return failure();
    }
    return success();
}
