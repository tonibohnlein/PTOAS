// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ConcreteLocalScoreboard.cpp - Instruction-ordered local check -------===//
// Recover the hardware-graph scoreboard's trust boundary, not its planner:
// pending effects stay outstanding; sets capture source-prefix knowledge;
// waits transfer a consumed token to its destination; named barriers establish
// only local knowledge. No planner completion graph is consulted.
// This supplements, and does not replace, full verification. Loop instances,
// cross-region effects, macro internals, and nonlocal visibility remain gated.

#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"
#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using EventKey = std::tuple<PIPE, PIPE, unsigned>;

struct PipeState {
    llvm::BitVector issued;
    llvm::BitVector known;
};

class LocalScoreboard {
public:
    LocalScoreboard(const StructuredSyncIR& schedule, SyncAccessId* source, SyncAccessId* target)
        : schedule(schedule),
          targetModel(ProtocolSyncTarget::resolve(schedule.getFunction())),
          uncoveredSource(source),
          uncoveredTarget(target)
    {}

    LogicalResult run(Block& block)
    {
        for (Operation& operation : block) {
            const bool hasNestedRegion = operation.getNumRegions() != 0;
            if (hasNestedRegion) {
                return success(); // Boundary tokens need structured interpretation.
            }
        }
        for (const SyncPhase& phase : schedule.getPhases()) {
            const bool outsideBlock = !phase.operation || phase.operation->getBlock() != &block;
            if (outsideBlock) {
                continue;
            }
            const bool ordinary = phase.core == SyncPhysicalCore::Vector && !phase.macroPhase &&
                                  phase.completion == SyncCompletionKind::PhaseEnd &&
                                  phase.iterationDomain.loops.empty();
            if (!ordinary) {
                return success(); // Outside this supplementary verifier's subset.
            }
            const bool uniquePhase = phases.try_emplace(phase.operation, &phase).second;
            if (!uniquePhase) {
                return success(); // Multiple physical phases need macro modeling.
            }
        }
        for (Operation& operation : block) {
            auto phase = phases.find(&operation);
            const bool invalidPhase = phase != phases.end() && failed(issue(*phase->second));
            if (invalidPhase) {
                return failure();
            }
            if (failed(synchronize(operation))) {
                return failure();
            }
        }
        return success(tokens.empty());
    }

private:
    PipeState& pipeState(PIPE pipe)
    {
        auto [entry, inserted] = pipes.try_emplace(pipe);
        if (inserted) {
            entry->second.issued.resize(schedule.getAccesses().size());
            entry->second.known.resize(schedule.getAccesses().size());
        }
        return entry->second;
    }

    LogicalResult checkAccess(const SyncAccess& access, const SyncLocalAccessRegion& region, PIPE pipe)
    {
        const llvm::BitVector& known = pipeState(pipe).known;
        for (const SyncLocalAccessRegion& previous : pending) {
            const SyncAccess* source = schedule.findAccess(previous.access);
            if (!source) {
                return failure();
            }
            const bool readRead = source->mode == SyncAccessMode::Read && access.mode == SyncAccessMode::Read;
            const bool overlaps = previous.interval.begin < region.interval.begin + region.interval.size &&
                                  region.interval.begin < previous.interval.begin + previous.interval.size;
            if (source->phase == access.phase || readRead || !overlaps || known.test(source->id)) {
                continue;
            }
            if (uncoveredSource) {
                *uncoveredSource = source->id;
            }
            if (uncoveredTarget) {
                *uncoveredTarget = access.id;
            }
            return failure();
        }
        return success();
    }

    LogicalResult issue(const SyncPhase& phase)
    {
        for (const SyncAccess& access : schedule.getAccesses()) {
            if (access.phase != phase.id || access.mode == SyncAccessMode::Ordered) {
                continue;
            }
            const SyncLocalAccessRegion region = recoverLocalAccessRegion(access);
            if (region.precision == SyncRegionPrecision::Unknown) {
                continue;
            }
            if (failed(checkAccess(access, region, phase.pipe))) {
                return failure();
            }
            pending.push_back(region);
            pipeState(phase.pipe).issued.set(access.id);
        }
        return success();
    }

    LogicalResult event(PIPE source, PIPE target, unsigned id, bool set)
    {
        const bool legal =
            targetModel.supportsEvent({SyncPhysicalCore::Vector, source}, {SyncPhysicalCore::Vector, target}) &&
            llvm::is_contained(targetModel.getCompilerEventIds(), id);
        if (!legal) {
            return failure();
        }
        const EventKey key{source, target, id};
        if (set) {
            PipeState& producer = pipeState(source);
            llvm::BitVector payload = producer.issued;
            payload |= producer.known;
            return success(tokens.try_emplace(key, std::move(payload)).second);
        }
        auto token = tokens.find(key);
        if (token == tokens.end()) {
            return failure();
        }
        pipeState(target).known |= token->second;
        tokens.erase(token);
        return success();
    }

    LogicalResult synchronize(Operation& operation)
    {
        if (auto set = dyn_cast<SetFlagOp>(&operation)) {
            return event(
                set.getSrcPipe().getPipe(), set.getDstPipe().getPipe(),
                static_cast<unsigned>(set.getEventId().getEvent()), true);
        }
        if (auto wait = dyn_cast<WaitFlagOp>(&operation)) {
            return event(
                wait.getSrcPipe().getPipe(), wait.getDstPipe().getPipe(),
                static_cast<unsigned>(wait.getEventId().getEvent()), false);
        }
        if (auto barrier = dyn_cast<BarrierOp>(&operation)) {
            const PIPE pipe = barrier.getPipe().getPipe();
            if (pipe == PIPE::PIPE_ALL) {
                // Only the full verifier may certify terminal drain placement.
                // Do not grant a body-wide completion edge here.
                return success();
            }
            if (!targetModel.supportsPipeBarrier({SyncPhysicalCore::Vector, pipe})) {
                return failure();
            }
            PipeState& local = pipeState(pipe);
            local.known |= local.issued;
        }
        return success();
    }

    const StructuredSyncIR& schedule;
    ProtocolSyncTarget targetModel;
    SyncAccessId* uncoveredSource;
    SyncAccessId* uncoveredTarget;
    llvm::DenseMap<Operation*, const SyncPhase*> phases;
    std::map<PIPE, PipeState> pipes;
    std::map<EventKey, llvm::BitVector> tokens;
    SmallVector<SyncLocalAccessRegion, 16> pending;
};

} // namespace

LogicalResult mlir::pto::protocol_sync::verifyConcreteLocalScoreboard(
    const StructuredSyncIR& schedule, SyncAccessId* uncoveredSource, SyncAccessId* uncoveredTarget)
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
    llvm::DenseSet<Block*> visited;
    for (const SyncPhase& phase : schedule.getPhases()) {
        Block* block = phase.operation ? phase.operation->getBlock() : nullptr;
        if (!block || !visited.insert(block).second) {
            continue;
        }
        LocalScoreboard scoreboard(schedule, uncoveredSource, uncoveredTarget);
        if (failed(scoreboard.run(*block))) {
            return failure();
        }
    }
    return success();
}
