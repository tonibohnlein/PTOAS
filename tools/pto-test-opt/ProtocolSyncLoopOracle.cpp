// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ProtocolSyncLoopOracle.cpp - Exhaustive per-pipe executions --------===//
// Test oracle: expand actual IR into FIFO commands, enumerate enabled pipe
// interleavings and independent asynchronous completions, and reject local
// hazards, live-key rearming, leaks or deadlock. Same-pipe issue is not completion.
// Uses fixture allocation byte ranges, not production requirements or recipes.
// Global drains rendezvous at the preceding command prefix on every lane.
// Return checks completion rather than implicitly waiting for pending work.

#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"

#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

enum class CommandKind { Phase, Set, Wait, Barrier };
struct Command {
    CommandKind kind;
    unsigned identity;
};
struct Boundary {
    std::vector<unsigned> prefixes;
    unsigned phaseCount = 0;
    bool returns = false;
};
using EventKey = std::tuple<PIPE, PIPE, unsigned>;

bool hazard(const SyncPhase& first, const SyncPhase& second, const StructuredSyncIR& schedule)
{
    for (SyncAccessId a : first.accesses) {
        const SyncAccess* left = schedule.findAccess(a);
        if (!left || left->storage.space == AddressSpace::GM) {
            continue;
        }
        for (SyncAccessId b : second.accesses) {
            const SyncAccess* right = schedule.findAccess(b);
            const bool local = right && right->storage.space == left->storage.space && first.core == second.core &&
                               (left->mode != SyncAccessMode::Read || right->mode != SyncAccessMode::Read);
            if (!local) {
                continue;
            }
            // All admitted test fixtures use addressed 16x16xf16 allocations.
            auto leftAlloc = left->value.getDefiningOp<AllocTileOp>();
            auto rightAlloc = right->value.getDefiningOp<AllocTileOp>();
            if (!leftAlloc || !rightAlloc) {
                return true;
            }
            auto leftAddr = leftAlloc.getAddr().getDefiningOp<arith::ConstantOp>();
            auto rightAddr = rightAlloc.getAddr().getDefiningOp<arith::ConstantOp>();
            if (!leftAddr || !rightAddr) {
                return true;
            }
            const auto x = cast<IntegerAttr>(leftAddr.getValue()).getInt();
            const auto y = cast<IntegerAttr>(rightAddr.getValue()).getInt();
            // Compare fixture byte intervals directly, independent of atom masks.
            if (x < y + 512 && y < x + 512) {
                return true;
            }
        }
    }
    return false;
}

class ExecutionOracle {
public:
    explicit ExecutionOracle(const StructuredSyncIR& schedule) : schedule(schedule) {}

    void append(Operation& operation)
    {
        for (const SyncPhase& phase : schedule.getPhases()) {
            if (phase.operation == &operation) {
                const unsigned lane = laneId(phase.pipe);
                positions.push_back({lane, static_cast<unsigned>(queues[lane].size())});
                queues[lane].push_back({CommandKind::Phase, static_cast<unsigned>(phases.size())});
                phases.push_back(&phase);
                return;
            }
        }
        if (auto set = dyn_cast<SetFlagOp>(&operation)) {
            event(
                set.getSrcPipe().getPipe(), set.getDstPipe().getPipe(),
                static_cast<unsigned>(set.getEventId().getEvent()), true);
        } else if (auto wait = dyn_cast<WaitFlagOp>(&operation)) {
            event(
                wait.getSrcPipe().getPipe(), wait.getDstPipe().getPipe(),
                static_cast<unsigned>(wait.getEventId().getEvent()), false);
        } else if (auto barrier = dyn_cast<BarrierOp>(&operation)) {
            const bool allLanes = barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
            if (allLanes) {
                boundary(false);
            } else {
                queues[laneId(barrier.getPipe().getPipe())].push_back({CommandKind::Barrier, 0});
            }
        } else if (isa<func::ReturnOp>(operation)) {
            boundary(true);
        }
    }

    bool run(
        bool* unsafeWitness = nullptr, std::optional<std::pair<SyncPhaseId, SyncPhaseId>> overlap = std::nullopt,
        bool* overlapWitness = nullptr)
    {
        if (overlapWitness) {
            *overlapWitness = false;
        }
        if (unsafeWitness) {
            *unsafeWitness = false;
        }
        const auto unsafe = [&]() {
            if (unsafeWitness) {
                *unsafeWitness = true;
            }
            return false;
        };
        const bool hasReturn = !boundaries.empty() && boundaries.back().returns;
        if (!hasReturn) {
            return false;
        }
        std::vector<std::vector<unsigned>> predecessors(phases.size());
        for (unsigned target = 0; target < phases.size(); ++target) {
            for (unsigned source = 0; source < target; ++source) {
                if (hazard(*phases[source], *phases[target], schedule)) {
                    predecessors[target].push_back(source);
                }
            }
        }
        using State = std::vector<unsigned>;
        std::set<State> visited;
        const unsigned completionBase = queues.size() + events.size();
        const unsigned boundaryIndex = completionBase + phases.size();
        std::vector<State> work{State(boundaryIndex + 1, 0)};
        visited.insert(work.front());
        constexpr unsigned maximumStates = 100000;
        for (unsigned index = 0; index < work.size(); ++index) {
            const State state = work[index];
            if (overlap && overlapWitness) {
                *overlapWitness |= hasPendingOverlap(state, completionBase, *overlap);
            }
            bool advanced = false;
            bool finished = true;
            const auto enqueue = [&](State next) {
                if (visited.insert(next).second) {
                    work.push_back(std::move(next));
                }
            };
            const Boundary* cut =
                state[boundaryIndex] < boundaries.size() ? &boundaries[state[boundaryIndex]] : nullptr;
            if (cut) {
                finished = false;
                bool reached = true;
                for (unsigned lane = 0; lane < queues.size(); ++lane) {
                    reached &= state[lane] == prefix(*cut, lane);
                }
                if (reached) {
                    bool completed = true;
                    for (unsigned phase = 0; phase < cut->phaseCount; ++phase) {
                        completed &= state[completionBase + phase] != 0;
                    }
                    if (cut->returns && !completed) {
                        return unsafe(); // Return must not drain asynchronous work.
                    }
                    if (completed) {
                        State next = state;
                        ++next[boundaryIndex];
                        enqueue(std::move(next));
                        advanced = true;
                    }
                }
            }
            // An issued phase may complete independently of subsequent issue.
            for (unsigned phase = 0; phase < phases.size(); ++phase) {
                const auto [owner, position] = positions[phase];
                const bool pending = state[owner] > position && state[completionBase + phase] == 0;
                if (pending) {
                    State next = state;
                    next[completionBase + phase] = 1;
                    enqueue(std::move(next));
                    advanced = true;
                    finished = false;
                }
            }
            for (unsigned lane = 0; lane < queues.size(); ++lane) {
                if (state[lane] == queues[lane].size()) {
                    continue;
                }
                finished = false;
                if (cut && state[lane] >= prefix(*cut, lane)) {
                    continue; // No lane crosses a global boundary early.
                }
                const Command command = queues[lane][state[lane]];
                const unsigned token = queues.size() + command.identity;
                if (command.kind == CommandKind::Set || command.kind == CommandKind::Barrier) {
                    bool pending = false;
                    for (unsigned phase = 0; phase < phases.size(); ++phase) {
                        const auto [owner, position] = positions[phase];
                        pending |= owner == lane && position < state[lane] && state[completionBase + phase] == 0;
                    }
                    if (pending) {
                        continue;
                    }
                }
                if (command.kind == CommandKind::Wait && state[token] == 0) {
                    continue;
                }
                if (command.kind == CommandKind::Set && state[token] != 0) {
                    return unsafe();
                }
                if (command.kind == CommandKind::Phase) {
                    for (unsigned previous : predecessors[command.identity]) {
                        if (state[completionBase + previous] == 0) {
                            return unsafe();
                        }
                    }
                }
                State next = state;
                ++next[lane];
                if (command.kind == CommandKind::Set || command.kind == CommandKind::Wait) {
                    next[token] = command.kind == CommandKind::Set ? 1 : 0;
                }
                advanced = true;
                enqueue(std::move(next));
            }
            const bool limit = work.size() >= maximumStates;
            if (limit) {
                return false; // Test budget exhaustion is not a passing proof.
            }
            if (!advanced && !finished) {
                return unsafe();
            }
            if (finished) {
                for (unsigned token = queues.size(); token < completionBase; ++token) {
                    if (state[token] != 0) {
                        return unsafe();
                    }
                }
            }
        }
        return true;
    }

private:
    bool hasPendingOverlap(
        const std::vector<unsigned>& state, unsigned completionBase, std::pair<SyncPhaseId, SyncPhaseId> overlap) const
    {
        bool firstPending = false;
        bool secondPending = false;
        for (unsigned phase = 0; phase < phases.size(); ++phase) {
            const auto [owner, position] = positions[phase];
            const bool pending = state[owner] > position && state[completionBase + phase] == 0;
            firstPending |= pending && phases[phase]->id == overlap.first;
            secondPending |= pending && phases[phase]->id == overlap.second;
        }
        return firstPending && secondPending;
    }

    static unsigned prefix(const Boundary& cut, unsigned lane)
    {
        return lane < cut.prefixes.size() ? cut.prefixes[lane] : 0;
    }

    void boundary(bool returns)
    {
        Boundary cut;
        cut.phaseCount = phases.size();
        cut.returns = returns;
        for (const auto& queue : queues) {
            cut.prefixes.push_back(queue.size());
        }
        boundaries.push_back(std::move(cut));
    }

    unsigned laneId(PIPE pipe)
    {
        auto [found, inserted] = lanes.emplace(pipe, queues.size());
        if (inserted) {
            queues.emplace_back();
        }
        return found->second;
    }

    void event(PIPE source, PIPE target, unsigned id, bool set)
    {
        const unsigned key = events.emplace(EventKey{source, target, id}, events.size()).first->second;
        const unsigned lane = laneId(set ? source : target);
        queues[lane].push_back({set ? CommandKind::Set : CommandKind::Wait, key});
    }

    const StructuredSyncIR& schedule;
    std::map<PIPE, unsigned> lanes;
    std::map<EventKey, unsigned> events;
    std::vector<std::vector<Command>> queues;
    std::vector<const SyncPhase*> phases;
    std::vector<std::pair<unsigned, unsigned>> positions;
    std::vector<Boundary> boundaries;
};

} // namespace

bool checkIndependentReadinessInterleavings(
    const StructuredSyncIR& schedule, SyncPhaseId first, SyncPhaseId second, bool& overlapWitness)
{
    ExecutionOracle oracle(schedule);
    for (Operation& operation : schedule.getFunction().getBody().front()) {
        oracle.append(operation);
    }
    // The caller requires both exhaustive safety and an overlapping execution.
    // Finding a witness before an unsafe state or budget limit is not success.
    return oracle.run(nullptr, std::make_pair(first, second), &overlapWitness);
}

bool checkStructuredFrontierInterleavings(
    const StructuredSyncIR& schedule, unsigned trips, std::uint64_t choices, bool* unsafeWitness)
{
    if (unsafeWitness) {
        *unsafeWitness = false;
    }
    ExecutionOracle oracle(schedule);
    unsigned decision = 0;
    const auto expand = [&](const auto& self, Block& block) -> bool {
        for (Operation& op : block) {
            if (auto branch = dyn_cast<scf::IfOp>(op)) {
                if (decision >= 64) {
                    return false;
                }
                const bool takeThen = (choices & (std::uint64_t{1} << decision++)) != 0;
                Region& arm = takeThen ? branch.getThenRegion() : branch.getElseRegion();
                const bool invalidArm = !arm.empty() && !self(self, arm.front());
                if (invalidArm) {
                    return false;
                }
            } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
                for (unsigned iteration = 0; iteration < trips; ++iteration) {
                    if (!self(self, *loop.getBody())) {
                        return false;
                    }
                }
            } else {
                oracle.append(op);
            }
        }
        return true;
    };
    return expand(expand, schedule.getFunction().getBody().front()) && oracle.run(unsafeWitness);
}

bool checkSelectiveLoopInterleavings(const StructuredSyncIR& schedule, unsigned trips, bool* overlapWitness)
{
    func::FuncOp function = schedule.getFunction();
    ExecutionOracle oracle(schedule);
    for (Operation& operation : function.getBody().front()) {
        if (auto loop = dyn_cast<scf::ForOp>(&operation)) {
            for (unsigned iteration = 0; iteration < trips; ++iteration) {
                for (Operation& nested : *loop.getBody()) {
                    oracle.append(nested);
                }
            }
        } else {
            oracle.append(operation);
        }
    }
    const auto overlap = overlapWitness ? std::optional<std::pair<SyncPhaseId, SyncPhaseId>>({1, 2}) : std::nullopt;
    return oracle.run(nullptr, overlap, overlapWitness);
}

bool checkLoopFrontierInterleavings(const StructuredSyncIR& schedule, unsigned trips)
{
    return checkSelectiveLoopInterleavings(schedule, trips, nullptr);
}

namespace {
// Evaluate raw scalar instructions instead of recognizing the planner's guard
// templates. APInt models fixed-width wrap without host signed overflow.
class BoundaryExpansion {
public:
    BoundaryExpansion(ExecutionOracle& oracle, scf::ForOp loop, unsigned trips)
        : oracle(oracle), loop(loop), trips(trips)
    {}

    bool initialize()
    {
        auto lower = loop.getLowerBound().getDefiningOp<arith::ConstantOp>();
        auto step = loop.getStep().getDefiningOp<arith::ConstantOp>();
        if (!lower || !step) {
            return false;
        }
        const auto low = cast<IntegerAttr>(lower.getValue()).getValue().sextOrTrunc(64);
        const auto stride = cast<IntegerAttr>(step.getValue()).getValue().sextOrTrunc(64);
        values[loop.getUpperBound()] = low + stride * trips;
        if (trips != 0) {
            // Non-dividing upper bounds exercise the final partial step.
            values[loop.getUpperBound()] -= stride - 1;
        }
        return true;
    }

    bool expand(Block& block)
    {
        for (Operation& operation : block) {
            if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
                if (auto value = dyn_cast<IntegerAttr>(constant.getValue())) {
                    values[constant.getResult()] = value.getValue().sextOrTrunc(64);
                }
            } else if (auto subtract = dyn_cast<arith::SubIOp>(operation)) {
                const bool operands = values.contains(subtract.getLhs()) && values.contains(subtract.getRhs());
                if (!operands) {
                    return false;
                }
                values[subtract.getResult()] = values[subtract.getLhs()] - values[subtract.getRhs()];
            } else if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
                if (!comparison(compare)) {
                    return false;
                }
            } else if (auto choice = dyn_cast<scf::IfOp>(operation)) {
                auto found = values.find(choice.getCondition());
                if (found == values.end()) {
                    return false;
                }
                Region& arm = found->second.isZero() ? choice.getElseRegion() : choice.getThenRegion();
                const bool validArm = arm.empty() || expand(arm.front());
                if (!validArm) {
                    return false;
                }
            } else if (auto recurring = dyn_cast<scf::ForOp>(operation)) {
                const bool supported =
                    recurring == loop && values.contains(loop.getLowerBound()) && values.contains(loop.getStep());
                if (!supported) {
                    return false;
                }
                auto iv = values[loop.getLowerBound()];
                for (unsigned iteration = 0; iteration < trips; ++iteration) {
                    if (!iv.slt(values[loop.getUpperBound()])) {
                        return false;
                    }
                    values[loop.getInductionVar()] = iv;
                    if (!expand(*loop.getBody())) {
                        return false;
                    }
                    iv += values[loop.getStep()];
                }
                if (iv.slt(values[loop.getUpperBound()])) {
                    return false;
                }
            } else {
                oracle.append(operation);
            }
        }
        return true;
    }

private:
    bool comparison(arith::CmpIOp operation)
    {
        const bool operands = values.contains(operation.getLhs()) && values.contains(operation.getRhs());
        if (!operands) {
            return false;
        }
        const auto a = values[operation.getLhs()];
        const auto b = values[operation.getRhs()];
        bool result = false;
        switch (operation.getPredicate()) {
            case arith::CmpIPredicate::eq:
                result = a == b;
                break;
            case arith::CmpIPredicate::ne:
                result = a != b;
                break;
            case arith::CmpIPredicate::slt:
                result = a.slt(b);
                break;
            case arith::CmpIPredicate::sle:
                result = a.sle(b);
                break;
            case arith::CmpIPredicate::sgt:
                result = a.sgt(b);
                break;
            case arith::CmpIPredicate::sge:
                result = a.sge(b);
                break;
            case arith::CmpIPredicate::ult:
                result = a.ult(b);
                break;
            case arith::CmpIPredicate::ule:
                result = a.ule(b);
                break;
            case arith::CmpIPredicate::ugt:
                result = a.ugt(b);
                break;
            case arith::CmpIPredicate::uge:
                result = a.uge(b);
                break;
            default:
                return false;
        }
        values[operation.getResult()] = llvm::APInt(64, result);
        return true;
    }
    ExecutionOracle& oracle;
    scf::ForOp loop;
    unsigned trips;
    llvm::DenseMap<Value, llvm::APInt> values;
};
} // namespace

bool checkSelectiveBoundaryInterleavings(const StructuredSyncIR& schedule, unsigned trips, bool* overlapWitness)
{
    ExecutionOracle oracle(schedule);
    auto loop = *schedule.getFunction().getOps<scf::ForOp>().begin();
    BoundaryExpansion expansion(oracle, loop, trips);
    const bool expanded = expansion.initialize() && expansion.expand(schedule.getFunction().getBody().front());
    if (!expanded) {
        return false;
    }
    // Prefix A is phase 1, independent suffix use of B is phase 4.
    const auto overlap = overlapWitness ? std::optional<std::pair<SyncPhaseId, SyncPhaseId>>({1, 4}) : std::nullopt;
    return oracle.run(nullptr, overlap, overlapWitness);
}
