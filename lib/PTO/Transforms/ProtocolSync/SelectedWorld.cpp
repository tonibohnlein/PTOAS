// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SelectedWorld.cpp - Adapt complete plans to logical effects ------===//

#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"

#include "PTO/Transforms/ProtocolSync/OneShotProtocol.h"
#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

const SyncChannel* findChannel(const ChannelAnalysisResult& channels, SyncChannelId id)
{
    if (id >= channels.getChannels().size()) {
        return nullptr;
    }
    const SyncChannel& channel = channels.getChannels()[id];
    return channel.id == id ? &channel : nullptr;
}

bool appendOneShotCompletion(
    const SyncOneShotPlan& plan, unsigned index, SyncSelectedWorld& world,
    llvm::DenseSet<SyncChannelId>& selectedChannels)
{
    const bool invalidIndex = index >= plan.protocols.size() || index + 1 >= plan.phaseOrder.size();
    if (invalidIndex) {
        return false;
    }
    const SyncOneShotProtocol& protocol = plan.protocols[index];
    if (protocol.sourcePhase != plan.phaseOrder[index] || protocol.targetPhase != plan.phaseOrder[index + 1]) {
        return false;
    }
    world.completions.push_back(
        {protocol.sourcePhase,
         protocol.targetPhase,
         SyncControlRelation::MustExecute,
         {SyncIterationRelationKind::SameIteration, 0}});
    for (SyncChannelId channel : protocol.channels) {
        selectedChannels.insert(channel);
    }
    return true;
}

bool appendSelectedOneShotProtocols(
    const ChannelAnalysisResult& channels, const llvm::DenseSet<SyncChannelId>& selectedChannels,
    SyncSelectedWorld& world)
{
    for (const SyncChannel& channel : channels.getChannels()) {
        const bool selected = selectedChannels.contains(channel.id);
        if (!selected) {
            continue;
        }
        const bool invalidChannel = !channel.isAdmitted() || channel.kind != SyncChannelKind::OneShot ||
                                    channel.generation == kInvalidSyncId || channel.capacity == 0;
        if (invalidChannel) {
            return false;
        }
        world.protocols.push_back(
            {SyncSelectedProtocolKind::OneShotPublish, channel.id, channel.generation, channel.capacity});
    }
    for (SyncChannelId selected : selectedChannels) {
        if (!findChannel(channels, selected)) {
            return false;
        }
    }
    return true;
}

bool hasCanonicalReadyReleaseLanes(const SyncReadyReleasePlan& plan)
{
    const bool invalidShape = plan.capacity == 0 || plan.lanes.size() != plan.capacity;
    if (invalidShape) {
        return false;
    }
    llvm::BitVector lanes(plan.capacity);
    for (const SyncReadyReleaseLane& lane : plan.lanes) {
        if (lane.logicalLane >= plan.capacity || lanes.test(lane.logicalLane)) {
            return false;
        }
        lanes.set(lane.logicalLane);
    }
    return lanes.all();
}

} // namespace

FailureOr<SyncSelectedWorld> mlir::pto::protocol_sync::buildSelectedWorld(
    const SyncOneShotPlan& plan, const ChannelAnalysisResult& channels)
{
    const bool invalidPlan = plan.status != SyncOneShotPlanStatus::Ready || plan.phaseOrder.empty() ||
                             !plan.emitTailBarrier || plan.protocols.size() != plan.phaseOrder.size() - 1;
    if (invalidPlan) {
        return failure();
    }

    SyncSelectedWorld world;
    llvm::DenseSet<SyncChannelId> selectedChannels;
    for (unsigned index = 0; index < plan.protocols.size(); ++index) {
        if (!appendOneShotCompletion(plan, index, world, selectedChannels)) {
            return failure();
        }
    }
    if (!appendSelectedOneShotProtocols(channels, selectedChannels, world)) {
        return failure();
    }
    world.exitCompletedPhases.append(plan.phaseOrder.begin(), plan.phaseOrder.end());
    return world;
}

FailureOr<SyncSelectedWorld> mlir::pto::protocol_sync::buildSelectedWorld(const SyncReadyReleasePlan& plan)
{
    if (plan.status != SyncReadyReleasePlanStatus::Ready || plan.channel == kInvalidSyncId ||
        plan.generation == kInvalidSyncId || plan.producerPhase == kInvalidSyncId ||
        plan.consumerPhase == kInvalidSyncId || plan.loopRegion == kInvalidSyncId ||
        !hasCanonicalReadyReleaseLanes(plan)) {
        return failure();
    }

    SyncSelectedWorld world;
    world.protocols.push_back({SyncSelectedProtocolKind::ReadyRelease, plan.channel, plan.generation, plan.capacity});
    world.completions.push_back(
        {plan.producerPhase,
         plan.consumerPhase,
         SyncControlRelation::MustExecute,
         {SyncIterationRelationKind::SameIteration, 0}});
    world.completions.push_back(
        {plan.consumerPhase,
         plan.producerPhase,
         SyncControlRelation::MustExecute,
         {SyncIterationRelationKind::LoopCarried, plan.capacity, plan.loopRegion}});
    world.exitCompletedPhases.push_back(plan.producerPhase);
    world.exitCompletedPhases.push_back(plan.consumerPhase);
    return world;
}
