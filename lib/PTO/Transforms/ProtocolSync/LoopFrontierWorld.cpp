// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LoopFrontierWorld.cpp - Completion-only structured certificate ------===//
// A compact logical certificate for a completely acknowledged serial loop and
// its outer chain. It cannot supply publication. Concrete callers reconstruct
// its actions before importing it; planners must validate the complete recipe.

#include "PTO/Transforms/ProtocolSync/LoopFrontierRepair.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::pto::protocol_sync;

FailureOr<SyncSelectedWorld> mlir::pto::protocol_sync::buildLoopFrontierWorld(const StructuredSyncIR& schedule)
{
    SyncLocalFlowOptions options;
    options.analyzeSingleLoop = true;
    auto local = analyzeLocalMemory(schedule, options);
    const bool complete = succeeded(local) && local->loopStatus == SyncLocalLoopStatus::Complete;
    if (!complete) {
        return failure();
    }
    SyncSelectedWorld world;
    world.orderedLoop = local->loopCarrier;
    for (const SyncPhase& phase : schedule.getPhases()) {
        world.exitCompletedPhases.push_back(phase.id);
    }
    return world;
}

bool mlir::pto::protocol_sync::loopFrontierOrders(
    const StructuredSyncIR& schedule, SyncRegionId carrier, SyncPhaseId source, SyncPhaseId target,
    const SyncIterationRelation& relation)
{
    const SyncRegion* region = schedule.findRegion(carrier);
    auto loop = region ? dyn_cast_or_null<scf::ForOp>(region->operation) : scf::ForOp();
    const SyncPhase* first = schedule.findPhase(source);
    const SyncPhase* second = schedule.findPhase(target);
    const bool valid = loop && first && second && first->operation && second->operation && first->guard.empty() &&
                       second->guard.empty();
    if (!valid) {
        return false;
    }
    const bool sourceBody = first->operation->getBlock() == loop.getBody();
    const bool targetBody = second->operation->getBlock() == loop.getBody();
    const bool sourceOuter = first->operation->getBlock() == loop->getBlock();
    const bool targetOuter = second->operation->getBlock() == loop->getBlock();
    switch (relation.kind) {
        case SyncIterationRelationKind::SameIteration:
            return relation.distance == 0 && relation.carrier == kInvalidSyncId && source < target &&
                   ((sourceBody && targetBody) || (sourceOuter && targetOuter));
        case SyncIterationRelationKind::LoopCarried:
            return relation.carrier == carrier && relation.distance > 0 && sourceBody && targetBody;
        case SyncIterationRelationKind::LoopEntry:
            return relation.carrier == carrier && relation.distance == 0 && sourceOuter && targetBody &&
                   first->operation->isBeforeInBlock(loop);
        case SyncIterationRelationKind::LoopExit:
            return relation.carrier == carrier && relation.distance == 0 && sourceBody && targetOuter &&
                   loop->isBeforeInBlock(second->operation);
        case SyncIterationRelationKind::LoopBypass:
            return relation.carrier == carrier && relation.distance == 0 && sourceOuter && targetOuter &&
                   first->operation->isBeforeInBlock(loop) && loop->isBeforeInBlock(second->operation);
        default:
            return false;
    }
}
