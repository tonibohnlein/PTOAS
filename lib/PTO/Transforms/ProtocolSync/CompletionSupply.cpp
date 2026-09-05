// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CompletionSupply.cpp - Forward transfer of certified completion -----===//
// Phase-before summaries contain only explicit completion-path supply. Lane
// identity locates the receiver; lexical or same-pipe succession adds no edge.
// Retained input-edge indices support backward explanation without consulting
// memory requirements, protocol recognition, or physical event allocation.

#include "PTO/Transforms/ProtocolSync/CompletionSupply.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto::protocol_sync;

namespace {
bool supportedDomain(const StructuredSyncIR& schedule)
{
    Block* block = nullptr;
    for (const SyncPhase& phase : schedule.getPhases()) {
        if (!phase.operation || phase.macroPhase || phase.core != SyncPhysicalCore::Vector ||
            phase.completion != SyncCompletionKind::PhaseEnd || !phase.guard.empty() ||
            !phase.iterationDomain.loops.empty()) {
            return false;
        }
        if (block && block != phase.operation->getBlock()) {
            return false;
        }
        block = phase.operation->getBlock();
    }
    return true;
}
} // namespace

FailureOr<SyncCompletionSupply> mlir::pto::protocol_sync::buildCompletionSupply(
    const StructuredSyncIR& schedule, const SyncSelectedWorld& world, std::size_t maximumMemberships)
{
    if (!schedule.isFrozen()) {
        return failure();
    }
    SyncCompletionSupply result;
    const bool unsupported = world.orderedLoop || !world.acknowledgedPhases.empty() || !supportedDomain(schedule);
    if (unsupported) {
        return result;
    }
    const std::size_t count = schedule.getPhases().size();
    if (count && count > maximumMemberships / count) {
        result.status = SyncCompletionSupplyStatus::LimitExceeded;
        return result;
    }
    const bool exhaustedIds = count >= kInvalidSyncId || world.completions.size() >= kInvalidSyncId;
    if (exhaustedIds) {
        return failure();
    }
    result.incoming.resize(count);
    for (const SyncPhase& phase : schedule.getPhases()) {
        if (phase.id != result.frontiers.size()) {
            return failure();
        }
        result.frontiers.push_back(
            {phase.id, phase.before, phase.region, phase.core, phase.pipe, llvm::BitVector(count)});
    }
    result.edges.assign(world.completions.begin(), world.completions.end());
    for (auto [index, edge] : llvm::enumerate(result.edges)) {
        if (edge.source >= count || edge.target >= count || edge.source >= edge.target ||
            edge.control != SyncControlRelation::MustExecute ||
            edge.iteration.kind != SyncIterationRelationKind::SameIteration || edge.iteration.distance != 0 ||
            edge.iteration.carrier != kInvalidSyncId) {
            return failure();
        }
        auto& completed = result.frontiers[edge.target].completed;
        if (completed.test(edge.source)) {
            return failure(); // Reject duplicate input edges before closure.
        }
        completed.set(edge.source);
        result.incoming[edge.target].push_back(static_cast<unsigned>(index));
    }
    for (SyncCompletionFrontier& frontier : result.frontiers) {
        for (unsigned edge : result.incoming[frontier.target]) {
            frontier.completed |= result.frontiers[result.edges[edge].source].completed;
        }
    }
    result.status = SyncCompletionSupplyStatus::Complete;
    return result;
}

const SyncCompletionFrontier* SyncCompletionSupply::getBefore(SyncPhaseId target) const
{
    if (status != SyncCompletionSupplyStatus::Complete || target >= frontiers.size()) {
        return nullptr;
    }
    return &frontiers[target];
}

bool SyncCompletionSupply::covers(SyncPhaseId source, SyncPhaseId target, const SyncIterationRelation& relation) const
{
    const auto* frontier = getBefore(target);
    return frontier && source < frontier->completed.size() &&
           relation.kind == SyncIterationRelationKind::SameIteration && relation.distance == 0 &&
           relation.carrier == kInvalidSyncId && frontier->completed.test(source);
}

llvm::SmallVector<unsigned, 4> SyncCompletionSupply::witness(SyncPhaseId source, SyncPhaseId target) const
{
    llvm::SmallVector<unsigned, 4> path;
    if (!covers(source, target, {SyncIterationRelationKind::SameIteration, 0})) {
        return path;
    }
    SyncPhaseId current = target;
    while (current != source) {
        bool found = false;
        for (unsigned index : incoming[current]) {
            const auto& edge = edges[index];
            if (edge.source == source || frontiers[edge.source].completed.test(source)) {
                path.push_back(index);
                current = edge.source; // Strictly decreases; no cyclic witness.
                found = true;
                break;
            }
        }
        if (!found) {
            return {};
        }
    }
    std::reverse(path.begin(), path.end());
    return path;
}
