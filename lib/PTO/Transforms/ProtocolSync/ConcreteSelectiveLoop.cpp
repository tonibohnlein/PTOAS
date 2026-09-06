// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Reconstruct literal event instances independently of selected repair edges.
#include "PTO/Transforms/ProtocolSync/SelectiveLoopRepair.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "PTO/Transforms/ProtocolSync/RecurringEventLifetime.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {
using Key = std::tuple<PIPE, PIPE, unsigned>;
struct Channel {
    Operation* set = nullptr;
    Operation* wait = nullptr;
    Operation* prime = nullptr;
    Operation* drain = nullptr;
};

bool assign(Operation*& destination, Operation* operation)
{
    if (destination) {
        return false;
    }
    destination = operation;
    return true;
}

bool appendPrefixSupply(
    const StructuredSyncIR& schedule, SyncSelectedWorld& world, Operation* set, Operation* wait, PIPE sourcePipe,
    PIPE targetPipe, unsigned distance, SyncRegionId carrier)
{
    for (const auto& source : schedule.getPhases()) {
        const bool before = source.pipe == sourcePipe && source.operation->isBeforeInBlock(set);
        if (!before) {
            continue;
        }
        for (const auto& target : schedule.getPhases()) {
            if (target.pipe == targetPipe && wait->isBeforeInBlock(target.operation)) {
                const bool exhausted = world.completions.size() >= kSelectiveLoopMaximumCompletions;
                if (exhausted) {
                    return false;
                }
                world.completions.push_back(
                    {source.id,
                     target.id,
                     SyncControlRelation::MustExecute,
                     {distance == 0 ? SyncIterationRelationKind::SameIteration : SyncIterationRelationKind::LoopCarried,
                      distance, distance == 0 ? kInvalidSyncId : carrier}});
            }
        }
    }
    return true;
}
} // namespace

FailureOr<SyncSelectedWorld> mlir::pto::protocol_sync::reconstructSelectiveLoop(const StructuredSyncIR& schedule)
{
    const auto carrier = findIsolatedSyncLoop(schedule, true);
    if (!carrier) {
        return failure();
    }
    auto loop = cast<scf::ForOp>(schedule.findRegion(*carrier)->operation);
    const auto target = ProtocolSyncTarget::resolve(schedule.getFunction());
    std::map<Key, Channel> channels;
    SmallVector<Operation*, 8> barriers;
    llvm::DenseMap<Operation*, unsigned> order;
    unsigned index = 0;
    for (Operation& operation : *loop.getBody()) {
        order[&operation] = index++;
    }
    Operation* tail = schedule.getFunction().getBody().front().getTerminator()->getPrevNode();
    bool exitDrain = false;
    bool invalid = false;
    schedule.getFunction().walk([&](Operation* operation) {
        if (auto barrier = dyn_cast<BarrierOp>(operation)) {
            const PIPE pipe = barrier.getPipe().getPipe();
            if (operation == tail && pipe == PIPE::PIPE_ALL) {
                exitDrain = true;
            } else if (
                operation->getBlock() == loop.getBody() &&
                target.supportsPipeBarrier({SyncPhysicalCore::Vector, pipe})) {
                barriers.push_back(operation);
            } else {
                invalid = true;
            }
            return;
        }
        if (!isa<SetFlagOp, WaitFlagOp>(operation)) {
            return;
        }
        auto set = dyn_cast<SetFlagOp>(operation);
        auto wait = dyn_cast<WaitFlagOp>(operation);
        const PIPE source = set ? set.getSrcPipe().getPipe() : wait.getSrcPipe().getPipe();
        const PIPE destination = set ? set.getDstPipe().getPipe() : wait.getDstPipe().getPipe();
        const unsigned id = static_cast<unsigned>(set ? set.getEventId().getEvent() : wait.getEventId().getEvent());
        const bool legal =
            target.supportsEvent({SyncPhysicalCore::Vector, source}, {SyncPhysicalCore::Vector, destination}) &&
            llvm::is_contained(target.getCompilerEventIds(), id);
        if (!legal) {
            invalid = true;
            return;
        }
        auto& channel = channels[{source, destination, id}];
        const bool inBody = operation->getBlock() == loop.getBody();
        if (inBody) {
            invalid |= !assign(set ? channel.set : channel.wait, operation);
        } else if (operation->getBlock() == loop->getBlock()) {
            if (set && operation->isBeforeInBlock(loop)) {
                invalid |= !assign(channel.prime, operation);
            } else if (wait && loop->isBeforeInBlock(operation) && operation->isBeforeInBlock(tail)) {
                invalid |= !assign(channel.drain, operation);
            } else {
                invalid = true;
            }
        } else {
            invalid = true;
        }
    });
    const bool unbounded =
        channels.size() > kSelectiveLoopMaximumChannels || barriers.size() > kSelectiveLoopMaximumBarriers;
    if (invalid || !exitDrain || unbounded) {
        return failure();
    }
    SyncSelectedWorld world;
    SmallVector<SyncRecurringEventChannel, 16> tokenChannels;
    for (const auto& entry : channels) {
        const auto [source, destination, id] = entry.first;
        const auto& channel = entry.second;
        const bool carried = channel.prime && channel.drain;
        const bool partial = static_cast<bool>(channel.prime) != static_cast<bool>(channel.drain);
        if (!channel.set || !channel.wait || partial) {
            return failure();
        }
        if (!carried && !channel.set->isBeforeInBlock(channel.wait)) {
            return failure();
        }
        tokenChannels.push_back(
            {static_cast<unsigned>(source), static_cast<unsigned>(destination), order.lookup(channel.set),
             order.lookup(channel.wait), carried ? 1U : 0U, carried, carried});
        if (!appendPrefixSupply(
                schedule, world, channel.set, channel.wait, source, destination, carried ? 1 : 0, *carrier)) {
            return failure();
        }
    }
    if (proveRecurringEventLifetimes(tokenChannels).status != SyncRecurringEventStatus::Proven) {
        return failure();
    }
    for (Operation* operation : barriers) {
        const PIPE pipe = cast<BarrierOp>(operation).getPipe().getPipe();
        if (!appendPrefixSupply(schedule, world, operation, operation, pipe, pipe, 0, *carrier)) {
            return failure();
        }
        // A body barrier also drains all preceding-iteration work on its pipe.
        for (const auto& source : schedule.getPhases()) {
            for (const auto& destination : schedule.getPhases()) {
                const bool covered = source.pipe == pipe && destination.pipe == pipe &&
                                     operation->isBeforeInBlock(destination.operation);
                if (covered) {
                    const bool exhausted = world.completions.size() >= kSelectiveLoopMaximumCompletions;
                    if (exhausted) {
                        return failure();
                    }
                    world.completions.push_back(
                        {source.id,
                         destination.id,
                         SyncControlRelation::MustExecute,
                         {SyncIterationRelationKind::LoopCarried, 1, *carrier}});
                }
            }
        }
    }
    for (const auto& phase : schedule.getPhases()) {
        world.exitCompletedPhases.push_back(phase.id);
    }
    const auto completionKey = [](const SyncSelectedCompletion& edge) {
        return std::make_tuple(
            edge.source, edge.target, edge.control, edge.iteration.kind, edge.iteration.distance,
            edge.iteration.carrier);
    };
    llvm::sort(world.completions, [&](const auto& a, const auto& b) { return completionKey(a) < completionKey(b); });
    const auto end = std::unique(world.completions.begin(), world.completions.end(), [&](const auto& a, const auto& b) {
        return completionKey(a) == completionKey(b);
    });
    world.completions.erase(end, world.completions.end());
    if (failed(verifySelectiveLoopMemory(schedule, world))) {
        return failure();
    }
    return world;
}
