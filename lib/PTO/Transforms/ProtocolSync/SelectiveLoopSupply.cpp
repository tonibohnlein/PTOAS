// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Completion relations for once-only boundaries and fixed-footprint recurrence.
#include "PTO/Transforms/ProtocolSync/SelectiveLoopRepair.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::pto::protocol_sync;

namespace {
bool closeRelations(SmallVectorImpl<llvm::BitVector>& graph)
{
    for (unsigned middle = 0; middle < graph.size(); ++middle) {
        for (auto& row : graph) {
            if (row.test(middle)) {
                row |= graph[middle];
            }
        }
    }
    for (unsigned index = 0; index < graph.size(); ++index) {
        if (graph[index].test(index)) {
            return false;
        }
    }
    return true;
}
} // namespace

bool SyncSelectiveLoopSupply::covers(
    SyncPhaseId source, SyncPhaseId target, const SyncIterationRelation& relation) const
{
    const unsigned count = body.size();
    if (source >= count || target >= count) {
        return false;
    }
    const bool a = body.test(source);
    const bool b = body.test(target);
    switch (relation.kind) {
        case SyncIterationRelationKind::SameIteration:
            if (relation.distance != 0 || a != b) {
                return false;
            }
            return positive[source].test(target) &&
                   (a ? positive[count + source].test(count + target) : bypass[source].test(target));
        case SyncIterationRelationKind::LoopEntry:
            return relation.carrier == carrier && relation.distance == 0 && !a && b && positive[source].test(target) &&
                   positive[source].test(count + target);
        case SyncIterationRelationKind::LoopExit:
            return relation.carrier == carrier && relation.distance == 0 && a && !b && positive[source].test(target) &&
                   positive[count + source].test(target);
        case SyncIterationRelationKind::LoopBypass:
            return relation.carrier == carrier && relation.distance == 0 && !a && !b && bypass[source].test(target);
        case SyncIterationRelationKind::LoopCarried:
            return relation.carrier == carrier && relation.distance == 1 && a && b &&
                   positive[source].test(count + target);
        default:
            return false;
    }
}

FailureOr<SyncSelectiveLoopSupply> mlir::pto::protocol_sync::buildSelectiveLoopSupply(
    const StructuredSyncIR& schedule, const SyncSelectedWorld& world)
{
    auto carrier = findIsolatedSyncLoop(schedule, true);
    if (!carrier) {
        return failure();
    }
    auto loop = cast<scf::ForOp>(schedule.findRegion(*carrier)->operation);
    const unsigned count = schedule.getPhases().size();
    SyncSelectiveLoopSupply result;
    result.carrier = *carrier;
    result.body.resize(count);
    result.positive.assign(2 * count, llvm::BitVector(2 * count));
    result.bypass.assign(count, llvm::BitVector(count));
    for (const auto& phase : schedule.getPhases()) {
        const bool body = phase.operation->getBlock() == loop.getBody();
        if (body) {
            result.body.set(phase.id);
        }
    }
    for (const auto& edge : world.completions) {
        const bool valid =
            edge.source < count && edge.target < count && edge.control == SyncControlRelation::MustExecute;
        if (!valid) {
            return failure();
        }
        const bool a = result.body.test(edge.source);
        const bool b = result.body.test(edge.target);
        const auto& relation = edge.iteration;
        const bool forward = edge.source < edge.target;
        const bool boundary = relation.carrier == *carrier && relation.distance == 0 && forward;
        if (relation.kind == SyncIterationRelationKind::SameIteration && relation.distance == 0 && a == b && forward) {
            result.positive[edge.source].set(edge.target);
            if (a) {
                result.positive[count + edge.source].set(count + edge.target);
            } else {
                result.bypass[edge.source].set(edge.target);
            }
        } else if (relation.kind == SyncIterationRelationKind::LoopEntry && boundary && !a && b) {
            result.positive[edge.source].set(edge.target);
            result.positive[edge.source].set(count + edge.target);
        } else if (relation.kind == SyncIterationRelationKind::LoopExit && boundary && a && !b) {
            result.positive[edge.source].set(edge.target);
            result.positive[count + edge.source].set(edge.target);
        } else if (relation.kind == SyncIterationRelationKind::LoopBypass && boundary && !a && !b) {
            result.bypass[edge.source].set(edge.target);
        } else if (
            relation.kind == SyncIterationRelationKind::LoopCarried && relation.carrier == *carrier &&
            relation.distance == 1 && a && b) {
            result.positive[edge.source].set(count + edge.target);
        } else {
            return failure();
        }
    }
    const bool positiveAcyclic = closeRelations(result.positive);
    const bool bypassAcyclic = closeRelations(result.bypass);
    if (!positiveAcyclic || !bypassAcyclic) {
        return failure();
    }
    return result;
}
