// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Reconstruct actual guarded boundary channels and recurring binary tokens.
#include "PTO/Transforms/ProtocolSync/SelectiveLoopRepair.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {
using Kind = SyncIterationRelationKind;
using Key = std::tuple<PIPE, PIPE, unsigned>;
enum class Guard { None, Nonempty, First, Last, Invalid };
struct Action {
    Operation* operation = nullptr;
    Operation* anchor = nullptr;
    Guard guard = Guard::Invalid;
};
struct Channel {
    SmallVector<Action, 2> sets;
    SmallVector<Action, 2> waits;
};

Action readAction(Operation* operation, scf::ForOp loop)
{
    Action result{operation, operation, Guard::None};
    auto choice = dyn_cast<scf::IfOp>(operation->getParentOp());
    if (!choice) {
        const bool ordinary = operation->getBlock() == loop.getBody() || operation->getBlock() == loop->getBlock();
        result.guard = ordinary ? Guard::None : Guard::Invalid;
        return result;
    }
    result.anchor = choice;
    result.guard = Guard::Invalid;
    const bool singleton = choice.getNumResults() == 0 && choice.getElseRegion().empty() &&
                           &choice.getThenRegion().front() == operation->getBlock() &&
                           choice.thenBlock()->getOperations().size() == 2;
    auto comparison = choice.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!singleton || !comparison) {
        return result;
    }
    const auto predicate = comparison.getPredicate();
    const Value left = comparison.getLhs();
    const Value right = comparison.getRhs();
    const bool forwardNonempty =
        predicate == arith::CmpIPredicate::slt && left == loop.getLowerBound() && right == loop.getUpperBound();
    const bool reversedNonempty =
        predicate == arith::CmpIPredicate::sgt && right == loop.getLowerBound() && left == loop.getUpperBound();
    const bool nonempty = choice->getBlock() == loop->getBlock() && (forwardNonempty || reversedNonempty);
    const bool firstOperands = (left == loop.getInductionVar() && right == loop.getLowerBound()) ||
                               (right == loop.getInductionVar() && left == loop.getLowerBound());
    const bool first = choice->getBlock() == loop.getBody() && predicate == arith::CmpIPredicate::eq && firstOperands;
    const bool last =
        choice->getBlock() == loop.getBody() && predicate == arith::CmpIPredicate::ule && right == loop.getStep();
    if (nonempty) {
        result.guard = Guard::Nonempty;
    } else if (first) {
        result.guard = Guard::First;
    } else if (last) {
        auto difference = left.getDefiningOp<arith::SubIOp>();
        const bool remaining = difference && difference.getLhs() == loop.getUpperBound() &&
                               difference.getRhs() == loop.getInductionVar() &&
                               difference.getOverflowFlags() == arith::IntegerOverflowFlags::none;
        if (remaining) {
            result.guard = Guard::Last;
        }
    }
    return result;
}

bool before(Operation* first, Operation* second)
{
    return first->getBlock() == second->getBlock() && first->isBeforeInBlock(second);
}

bool appendSupply(
    const StructuredSyncIR& schedule, SyncSelectedWorld& world, scf::ForOp loop, SyncRegionId carrier, Operation* set,
    Operation* wait, PIPE p, PIPE q, Kind kind, bool allBodySources = false)
{
    for (const auto& source : schedule.getPhases()) {
        const bool body = source.operation->getBlock() == loop.getBody();
        const bool bodySource = kind == Kind::LoopCarried || kind == Kind::LoopExit;
        const bool selected = source.pipe == p && (allBodySources ? body : before(source.operation, set)) &&
                              (kind == Kind::SameIteration || body == bodySource);
        if (!selected) {
            continue;
        }
        for (const auto& target : schedule.getPhases()) {
            if (target.pipe != q || !before(wait, target.operation)) {
                continue;
            }
            const bool exhausted = world.completions.size() >= kSelectiveLoopMaximumCompletions;
            if (exhausted) {
                return false;
            }
            world.completions.push_back(
                {source.id,
                 target.id,
                 SyncControlRelation::MustExecute,
                 {kind, kind == Kind::LoopCarried ? 1U : 0U, kind == Kind::SameIteration ? kInvalidSyncId : carrier}});
        }
    }
    return true;
}

bool emptyPrimeAndLateDrain(
    const StructuredSyncIR& schedule, scf::ForOp loop, const Action& prime, const Action& drain,
    ArrayRef<Action> actions)
{
    const bool misplaced = prime.guard != Guard::None || drain.guard != Guard::None || !before(prime.anchor, loop) ||
                           !before(loop, drain.anchor);
    if (misplaced) {
        return false;
    }
    for (const auto& phase : schedule.getPhases()) {
        const bool outside = phase.operation->getBlock() == loop->getBlock();
        if (!outside) {
            continue;
        }
        const bool latePrime = before(phase.operation, loop) && !before(prime.anchor, phase.operation);
        if (latePrime) {
            return false;
        }
        const bool earlyDrain = before(loop, phase.operation) && !before(phase.operation, drain.anchor);
        if (earlyDrain) {
            return false;
        }
    }
    for (const auto& action : actions) {
        const bool outstanding =
            action.guard == Guard::Nonempty && before(loop, action.anchor) && !before(action.anchor, drain.anchor);
        if (outstanding) {
            return false;
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
    const auto core = schedule.getPhases().front().core;
    std::map<Key, Channel> channels;
    SmallVector<Action, 8> barriers;
    SmallVector<Action, 16> actions;
    llvm::DenseMap<Operation*, unsigned> order;
    unsigned index = 0;
    for (Operation& operation : *loop.getBody()) {
        order[&operation] = index++;
    }
    Operation* tail = schedule.getFunction().getBody().front().getTerminator()->getPrevNode();
    bool exitDrain = false;
    bool invalid = false;
    schedule.getFunction().walk([&](Operation* operation) {
        if (!isa<SetFlagOp, WaitFlagOp, BarrierOp>(operation)) {
            return;
        }
        Action action = readAction(operation, loop);
        if (action.guard == Guard::Invalid) {
            invalid = true;
            return;
        }
        actions.push_back(action);
        if (auto barrier = dyn_cast<BarrierOp>(operation)) {
            const PIPE pipe = barrier.getPipe().getPipe();
            if (operation == tail && pipe == PIPE::PIPE_ALL) {
                exitDrain = true;
            } else if (target.supportsPipeBarrier({core, pipe})) {
                barriers.push_back(action);
            } else {
                invalid = true;
            }
            return;
        }
        auto set = dyn_cast<SetFlagOp>(operation);
        auto wait = dyn_cast<WaitFlagOp>(operation);
        const PIPE p = set ? set.getSrcPipe().getPipe() : wait.getSrcPipe().getPipe();
        const PIPE q = set ? set.getDstPipe().getPipe() : wait.getDstPipe().getPipe();
        const unsigned id = static_cast<unsigned>(set ? set.getEventId().getEvent() : wait.getEventId().getEvent());
        const bool supported =
            target.supportsEvent({core, p}, {core, q}) && llvm::is_contained(target.getCompilerEventIds(), id);
        if (!supported) {
            invalid = true;
            return;
        }
        auto& channel = channels[{p, q, id}];
        (set ? channel.sets : channel.waits).push_back(action);
    });
    const bool rejected = invalid || !exitDrain || channels.size() > kSelectiveLoopMaximumChannels ||
                          barriers.size() > kSelectiveLoopMaximumBarriers;
    if (rejected) {
        return failure();
    }
    SyncSelectedWorld world;
    SmallVector<SyncRecurringEventChannel, 16> tokens;
    for (const auto& entry : channels) {
        const auto [p, q, id] = entry.first;
        const auto& channel = entry.second;
        const bool unbalanced =
            channel.sets.size() != channel.waits.size() || channel.sets.empty() || channel.sets.size() > 2;
        if (unbalanced) {
            return failure();
        }
        const bool carried = channel.sets.size() == 2;
        const Action* set = &channel.sets.front();
        const Action* wait = &channel.waits.front();
        if (carried) {
            if (!emptyPrimeAndLateDrain(schedule, loop, channel.sets.front(), channel.waits.back(), actions)) {
                return failure();
            }
            set = &channel.sets.back();
        }
        Kind kind = Kind::Unknown;
        const bool bodyPair = set->anchor->getBlock() == loop.getBody() && wait->anchor->getBlock() == loop.getBody();
        if (set->guard == Guard::None && wait->guard == Guard::None && bodyPair) {
            if (!carried && !before(set->anchor, wait->anchor)) {
                return failure();
            }
            kind = carried ? Kind::LoopCarried : Kind::SameIteration;
            tokens.push_back(
                {static_cast<unsigned>(p), static_cast<unsigned>(q), order.lookup(set->anchor),
                 order.lookup(wait->anchor), carried ? 1U : 0U, carried, carried});
        } else if (
            !carried && set->guard == Guard::Nonempty && wait->guard == Guard::First && before(set->anchor, loop)) {
            kind = Kind::LoopEntry;
        } else if (
            !carried && set->guard == Guard::Last && wait->guard == Guard::Nonempty && before(loop, wait->anchor)) {
            kind = Kind::LoopExit;
        } else if (
            !carried && set->guard == Guard::None && wait->guard == Guard::None &&
            set->anchor->getBlock() == loop->getBlock() && before(set->anchor, wait->anchor)) {
            kind = Kind::SameIteration;
        } else {
            return failure();
        }
        if (!appendSupply(schedule, world, loop, *carrier, set->anchor, wait->anchor, p, q, kind)) {
            return failure();
        }
    }
    if (proveRecurringEventLifetimes(tokens).status != SyncRecurringEventStatus::Proven) {
        return failure();
    }
    for (const Action& action : barriers) {
        const PIPE pipe = cast<BarrierOp>(action.operation).getPipe().getPipe();
        const bool inBody = action.anchor->getBlock() == loop.getBody();
        const bool suffix = before(loop, action.anchor);
        if (action.guard == Guard::First && inBody) {
            if (!appendSupply(schedule, world, loop, *carrier, loop, action.anchor, pipe, pipe, Kind::LoopEntry)) {
                return failure();
            }
        } else if (action.guard == Guard::Nonempty && suffix) {
            if (!appendSupply(schedule, world, loop, *carrier, loop, action.anchor, pipe, pipe, Kind::LoopExit, true)) {
                return failure();
            }
        } else if (action.guard == Guard::None) {
            if (!appendSupply(
                    schedule, world, loop, *carrier, action.anchor, action.anchor, pipe, pipe, Kind::SameIteration)) {
                return failure();
            }
            if (inBody || suffix) {
                const Kind kind = inBody ? Kind::LoopCarried : Kind::LoopExit;
                if (!appendSupply(schedule, world, loop, *carrier, loop, action.anchor, pipe, pipe, kind, true)) {
                    return failure();
                }
            }
        } else {
            return failure();
        }
    }
    for (const auto& phase : schedule.getPhases()) {
        world.exitCompletedPhases.push_back(phase.id);
    }
    const auto key = [](const SyncSelectedCompletion& edge) {
        return std::make_tuple(
            edge.source, edge.target, edge.iteration.kind, edge.iteration.distance, edge.iteration.carrier);
    };
    llvm::sort(world.completions, [&](const auto& a, const auto& b) { return key(a) < key(b); });
    const auto end = std::unique(world.completions.begin(), world.completions.end(), [&](const auto& a, const auto& b) {
        return key(a) == key(b);
    });
    world.completions.erase(end, world.completions.end());
    if (failed(verifySelectiveLoopMemory(schedule, world))) {
        return failure();
    }
    return world;
}
