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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

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
constexpr StringLiteral kLanesAttr = "pto.protocol_sync.logical_lanes";

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

void tagDynamicAction(Operation* operation, OpBuilder& builder, StringRef role, unsigned capacity)
{
    tagAction(operation, builder, role, std::nullopt);
    llvm::SmallVector<std::int32_t, 2> lanes;
    for (unsigned lane = 0; lane < capacity; ++lane) {
        lanes.push_back(static_cast<std::int32_t>(lane));
    }
    operation->setAttr(kLanesAttr, builder.getDenseI32ArrayAttr(lanes));
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

template <typename OpTy>
Operation* createDynamicEvent(
    OpBuilder& builder, Location location, func::FuncOp clone, PIPE source, PIPE target, Value eventId, StringRef role,
    unsigned capacity)
{
    auto operation = builder.create<OpTy>(
        location, PipeAttr::get(clone.getContext(), source), PipeAttr::get(clone.getContext(), target), eventId);
    tagDynamicAction(operation, builder, role, capacity);
    return operation.getOperation();
}

Value createLaneOnePredicate(OpBuilder& builder, Location location, Value selector, unsigned capacity)
{
    const bool needsIndexCast = selector.getType() != builder.getIndexType();
    if (needsIndexCast) {
        selector = builder.create<arith::IndexCastOp>(location, builder.getIndexType(), selector);
    }
    Value capacityValue = builder.create<arith::ConstantIndexOp>(location, capacity);
    Value selectedLane = builder.create<arith::RemUIOp>(location, selector, capacityValue);
    Value laneOne = builder.create<arith::ConstantIndexOp>(location, 1);
    return builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, selectedLane, laneOne);
}

Value createEventSelector(
    OpBuilder& builder, Location location, Value selectsLaneOne, ArrayRef<SyncReadyReleaseLane> lanes, bool ready)
{
    const unsigned laneZeroId = ready ? *lanes[0].readyEventId : *lanes[0].releaseEventId;
    const unsigned laneOneId = ready ? *lanes[1].readyEventId : *lanes[1].releaseEventId;
    Value laneZeroEvent = builder.create<arith::ConstantIndexOp>(location, laneZeroId);
    Value laneOneEvent = builder.create<arith::ConstantIndexOp>(location, laneOneId);
    return builder.create<arith::SelectOp>(location, selectsLaneOne, laneOneEvent, laneZeroEvent);
}

} // namespace

LogicalResult mlir::pto::protocol_sync::materializeReadyReleaseProtocolPlan(
    func::FuncOp clone, const IRMapping& mapping, const SyncReadyReleasePlan& plan, ProtocolSyncStatistics* statistics)
{
    const bool validCapacity = plan.capacity == 1 || plan.capacity == 2;
    const bool validLanes =
        plan.lanes.size() == plan.capacity && llvm::all_of(llvm::enumerate(plan.lanes), [](auto indexedLane) {
            const SyncReadyReleaseLane& lane = indexedLane.value();
            return lane.logicalLane == indexedLane.index() && lane.readyEventId && lane.releaseEventId;
        });
    const bool validSelector = plan.capacity == 1 ? !plan.slot : plan.slot.has_value();
    const bool validPlan = plan.status == SyncReadyReleasePlanStatus::Ready && validCapacity && validLanes &&
                           validSelector && plan.tokenCertificate.zeroTripSafe && plan.tokenCertificate.oneTripSafe &&
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
    OpBuilder builder(clone.getContext());
    builder.setInsertionPoint(loop);
    for (const SyncReadyReleaseLane& lane : plan.lanes) {
        createEvent<SetFlagOp>(
            builder, loop.getLoc(), clone, plan.consumerPipe, plan.producerPipe, *lane.releaseEventId,
            "release-prime-set", lane.logicalLane);
    }

    builder.setInsertionPoint(producer);
    if (plan.capacity == 1) {
        const SyncReadyReleaseLane& lane = plan.lanes.front();
        createEvent<WaitFlagOp>(
            builder, producer->getLoc(), clone, plan.consumerPipe, plan.producerPipe, *lane.releaseEventId,
            "release-body-wait", lane.logicalLane);
        builder.setInsertionPointAfter(producer);
        createEvent<SetFlagOp>(
            builder, producer->getLoc(), clone, plan.producerPipe, plan.consumerPipe, *lane.readyEventId,
            "ready-body-set", lane.logicalLane);

        builder.setInsertionPoint(consumer);
        createEvent<WaitFlagOp>(
            builder, consumer->getLoc(), clone, plan.producerPipe, plan.consumerPipe, *lane.readyEventId,
            "ready-body-wait", lane.logicalLane);
        builder.setInsertionPointAfter(consumer);
        createEvent<SetFlagOp>(
            builder, consumer->getLoc(), clone, plan.consumerPipe, plan.producerPipe, *lane.releaseEventId,
            "release-body-set", lane.logicalLane);
    } else {
        Value selector = mapping.lookupOrNull(plan.slot->selector);
        if (!selector) {
            return failure();
        }
        Value selectsLaneOne = createLaneOnePredicate(builder, producer->getLoc(), selector, plan.capacity);
        Value releaseEvent = createEventSelector(builder, producer->getLoc(), selectsLaneOne, plan.lanes, false);
        Value readyEvent = createEventSelector(builder, producer->getLoc(), selectsLaneOne, plan.lanes, true);
        createDynamicEvent<WaitFlagDynOp>(
            builder, producer->getLoc(), clone, plan.consumerPipe, plan.producerPipe, releaseEvent, "release-body-wait",
            plan.capacity);
        builder.setInsertionPointAfter(producer);
        createDynamicEvent<SetFlagDynOp>(
            builder, producer->getLoc(), clone, plan.producerPipe, plan.consumerPipe, readyEvent, "ready-body-set",
            plan.capacity);

        builder.setInsertionPoint(consumer);
        createDynamicEvent<WaitFlagDynOp>(
            builder, consumer->getLoc(), clone, plan.producerPipe, plan.consumerPipe, readyEvent, "ready-body-wait",
            plan.capacity);
        builder.setInsertionPointAfter(consumer);
        createDynamicEvent<SetFlagDynOp>(
            builder, consumer->getLoc(), clone, plan.consumerPipe, plan.producerPipe, releaseEvent, "release-body-set",
            plan.capacity);
    }

    builder.setInsertionPointAfter(loop);
    for (const SyncReadyReleaseLane& lane : plan.lanes) {
        createEvent<WaitFlagOp>(
            builder, loop.getLoc(), clone, plan.consumerPipe, plan.producerPipe, *lane.releaseEventId,
            "release-drain-wait", lane.logicalLane);
    }

    if (statistics) {
        statistics->materializationTransitions += 4 + 2 * plan.capacity;
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
    function.walk([&](Operation* operation) {
        identityMapping.map(operation, operation);
        for (Value result : operation->getResults()) {
            identityMapping.map(result, result);
        }
        for (Region& region : operation->getRegions()) {
            for (Block& block : region) {
                for (BlockArgument argument : block.getArguments()) {
                    identityMapping.map(argument, argument);
                }
            }
        }
    });
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
