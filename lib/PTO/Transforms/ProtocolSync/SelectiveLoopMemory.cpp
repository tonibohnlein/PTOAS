// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Independent occurrence-pair check. No sparse atom/generation construction.
#include "PTO/Transforms/ProtocolSync/SelectiveLoopRepair.h"
#include "PTO/Transforms/ProtocolSync/GMAliasPolicy.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

std::optional<SyncRegionId> mlir::pto::protocol_sync::findIsolatedSyncLoop(
    const StructuredSyncIR& schedule, bool allowFixed)
{
    const bool bounded = schedule.isFrozen() && schedule.getFailures().empty() && !schedule.getPhases().empty() &&
                         schedule.getPhases().size() <= kSelectiveLoopMaximumPhases &&
                         schedule.getAccesses().size() <= kSelectiveLoopMaximumAccesses;
    if (!bounded) {
        return std::nullopt;
    }
    if (failed(verifySyncDescriptorBindings(schedule))) {
        return std::nullopt;
    }
    const SyncRegion* carrier = nullptr;
    for (const SyncRegion& region : schedule.getRegions()) {
        if (region.kind == SyncRegionKind::PhysicalSection) {
            return std::nullopt;
        }
        if (region.kind == SyncRegionKind::Choice) {
            // Concrete boundary guards contain synchronization only. Their
            // predicates and token participation are checked by reconstruction.
            const bool physical = llvm::any_of(schedule.getPhases(), [&](const SyncPhase& phase) {
                return region.operation->isAncestor(phase.operation);
            });
            if (!allowFixed || physical) {
                return std::nullopt;
            }
        }
        if (region.kind == SyncRegionKind::Loop) {
            if (carrier) {
                return std::nullopt;
            }
            carrier = &region;
        }
    }
    auto loop = carrier ? dyn_cast_or_null<scf::ForOp>(carrier->operation) : scf::ForOp();
    const bool ordinaryLoop = loop && loop.getNumRegionIterArgs() == 0 && loop->getParentOp() == schedule.getFunction();
    if (!ordinaryLoop) {
        return std::nullopt;
    }
    IntegerAttr step;
    const bool positiveStep = matchPattern(loop.getStep(), m_Constant(&step)) && step.getValue().isStrictlyPositive();
    if (!positiveStep) {
        return std::nullopt;
    }
    unsigned operationCount = 0;
    schedule.getFunction().walk([&](Operation*) {
        ++operationCount;
        return operationCount > kSelectiveLoopMaximumOperations ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (operationCount > kSelectiveLoopMaximumOperations) {
        return std::nullopt;
    }
    llvm::DenseSet<Operation*> operations;
    for (const SyncPhase& phase : schedule.getPhases()) {
        const bool supported =
            phase.operation &&
            (phase.operation->getBlock() == loop.getBody() || phase.operation->getBlock() == loop->getBlock()) &&
            phase.operation->getNumResults() == 0 && phase.guard.empty() && phase.core == SyncPhysicalCore::Vector &&
            !phase.macroPhase && phase.completion == SyncCompletionKind::PhaseEnd &&
            (phase.pipe == PIPE::PIPE_MTE2 || phase.pipe == PIPE::PIPE_V || phase.pipe == PIPE::PIPE_MTE3) &&
            operations.insert(phase.operation).second;
        if (!supported) {
            return std::nullopt;
        }
    }
    for (const auto& summary : schedule.getSummaries()) {
        const bool ordinary = summary.provider == SyncSummaryProvider::Pipeline ||
                              summary.provider == SyncSummaryProvider::Structural ||
                              summary.provider == SyncSummaryProvider::Descriptor;
        const bool fixed = allowFixed && isa<SetFlagOp, WaitFlagOp, BarrierOp>(summary.operation);
        const bool unsupported = (!ordinary && !fixed) || !summary.eventReservations.empty() || summary.queue;
        if (unsupported) {
            return std::nullopt;
        }
    }
    for (const SyncAccess& access : schedule.getAccesses()) {
        const bool ordinary = access.mode == SyncAccessMode::Read || access.mode == SyncAccessMode::Write ||
                              access.mode == SyncAccessMode::ReadWrite;
        if (!ordinary || access.slot) {
            return std::nullopt;
        }
        if (access.storage.space == AddressSpace::VEC) {
            if (recoverLocalAccessRegion(access).precision == SyncRegionPrecision::Unknown) {
                return std::nullopt;
            }
        } else if (access.storage.space != AddressSpace::GM) {
            return std::nullopt;
        }
    }
    return carrier->id;
}

LogicalResult mlir::pto::protocol_sync::verifySelectiveLoopMemory(
    const StructuredSyncIR& schedule, const SyncSelectedWorld& world)
{
    const auto carrier = findIsolatedSyncLoop(schedule, true);
    const bool unsupported =
        !carrier || world.orderedLoop || !world.acknowledgedPhases.empty() || !world.protocols.empty();
    if (unsupported) {
        return failure();
    }
    auto supply = buildSelectiveLoopSupply(schedule, world);
    if (failed(supply)) {
        return failure();
    }
    for (const SyncAccess& first : schedule.getAccesses()) {
        for (const SyncAccess& second : schedule.getAccesses()) {
            const bool sourceBody = supply->body.test(first.phase);
            const bool targetBody = supply->body.test(second.phase);
            if (first.phase >= second.phase && !(sourceBody && targetBody)) {
                continue;
            }
            const bool readRead = first.mode == SyncAccessMode::Read && second.mode == SyncAccessMode::Read;
            if (readRead || first.storage.space != second.storage.space) {
                continue;
            }
            if (first.storage.space == AddressSpace::GM) {
                if (haveDisjointGMArgumentRoots(schedule, first, second)) {
                    continue;
                }
                const bool publication = first.mode != SyncAccessMode::Read && second.mode != SyncAccessMode::Write;
                if (publication) {
                    return failure(); // No qualified GM publication rule in this slice.
                }
            } else {
                const auto a = recoverLocalAccessRegion(first);
                const auto b = recoverLocalAccessRegion(second);
                const bool overlap = a.interval.begin < b.interval.begin + b.interval.size &&
                                     b.interval.begin < a.interval.begin + a.interval.size;
                if (!overlap) {
                    continue;
                }
            }
            // Only body occurrences repeat. Once-only prefix/suffix accesses
            // must never create fictitious carried pairs or reverse GM hazards.
            bool missing = false;
            if (first.phase < second.phase) {
                const auto kind = sourceBody == targetBody ? SyncIterationRelationKind::SameIteration :
                                  targetBody               ? SyncIterationRelationKind::LoopEntry :
                                                             SyncIterationRelationKind::LoopExit;
                missing = !supply->covers(first.phase, second.phase, {kind, 0, *carrier});
            }
            if (sourceBody && targetBody) {
                missing |=
                    !supply->covers(first.phase, second.phase, {SyncIterationRelationKind::LoopCarried, 1, *carrier});
            }
            if (missing) {
                return failure();
            }
        }
    }
    // Fixed footprints: writer self-recurrence composes distance-one coverage
    // to all positive distances. This is not a bounded-unrolling proof for
    // symbolic slots, choices, or general memory visibility.
    return success();
}
