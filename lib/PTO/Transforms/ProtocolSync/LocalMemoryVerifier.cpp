// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LocalMemoryVerifier.cpp - Independent all-pair completion check ------===//
// Deliberately does not call analyzeLocalMemory or consume its atoms, sparse
// requirement chains, or generation records. It shares only operation-local
// footprint recovery with the planner; the byte-set oracle tests that boundary.

#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

bool hasSameExecution(const SyncPhase& source, const SyncPhase& target)
{
    const bool sameGuard = source.guard.size() == target.guard.size() &&
                           llvm::equal(source.guard, target.guard, [](const auto& first, const auto& second) {
                               return first.choice == second.choice && first.arm == second.arm;
                           });
    return source.operation && target.operation && source.core == SyncPhysicalCore::Vector &&
           target.core == source.core && source.operation->getBlock() == target.operation->getBlock() &&
           source.id < target.id && source.iterationDomain.loops.empty() && target.iterationDomain.loops.empty() &&
           sameGuard;
}

bool reaches(SyncPhaseId source, SyncPhaseId target, ArrayRef<SmallVector<SyncPhaseId, 4>> successors)
{
    llvm::BitVector visited(successors.size());
    SmallVector<SyncPhaseId, 16> worklist{source};
    while (!worklist.empty()) {
        const SyncPhaseId current = worklist.pop_back_val();
        if (current == target) {
            return true;
        }
        if (visited.test(current)) {
            continue;
        }
        visited.set(current);
        worklist.append(successors[current].begin(), successors[current].end());
    }
    return false;
}

} // namespace

LogicalResult mlir::pto::protocol_sync::verifyLocalMemoryCoverage(
    const StructuredSyncIR& schedule, const SyncSelectedWorld& concreteWorld, SyncAccessId* uncoveredSource,
    SyncAccessId* uncoveredTarget)
{
    if (uncoveredSource) {
        *uncoveredSource = kInvalidSyncId;
    }
    if (uncoveredTarget) {
        *uncoveredTarget = kInvalidSyncId;
    }
    if (!schedule.isFrozen()) {
        return failure();
    }
    SmallVector<SmallVector<SyncPhaseId, 4>, 16> successors(schedule.getPhases().size());
    for (const SyncSelectedCompletion& completion : concreteWorld.completions) {
        const SyncPhase* source = schedule.findPhase(completion.source);
        const SyncPhase* target = schedule.findPhase(completion.target);
        if (!source || !target) {
            return failure();
        }
        const bool sameIteration =
            completion.iteration.kind == SyncIterationRelationKind::SameIteration && completion.iteration.distance == 0;
        if (sameIteration && hasSameExecution(*source, *target)) {
            successors[source->id].push_back(target->id);
        }
    }
    SmallVector<SyncLocalAccessRegion, 16> regions;
    for (const SyncAccess& access : schedule.getAccesses()) {
        regions.push_back(recoverLocalAccessRegion(access));
    }
    for (const SyncAccess& source : schedule.getAccesses()) {
        const SyncLocalAccessRegion& first = regions[source.id];
        if (first.precision == SyncRegionPrecision::Unknown) {
            continue;
        }
        for (const SyncAccess& target : schedule.getAccesses()) {
            const SyncLocalAccessRegion& second = regions[target.id];
            const bool readRead = source.mode == SyncAccessMode::Read && target.mode == SyncAccessMode::Read;
            if (second.precision == SyncRegionPrecision::Unknown || readRead) {
                continue;
            }
            const SyncPhase* sourcePhase = schedule.findPhase(source.phase);
            const SyncPhase* targetPhase = schedule.findPhase(target.phase);
            if (!sourcePhase || !targetPhase) {
                return failure();
            }
            const bool overlap = first.interval.begin < second.interval.begin + second.interval.size &&
                                 second.interval.begin < first.interval.begin + first.interval.size;
            if (!overlap || !hasSameExecution(*sourcePhase, *targetPhase)) {
                continue;
            }
            if (!reaches(source.phase, target.phase, successors)) {
                if (uncoveredSource) {
                    *uncoveredSource = source.id;
                }
                if (uncoveredTarget) {
                    *uncoveredTarget = target.id;
                }
                return failure();
            }
        }
    }
    return success();
}
