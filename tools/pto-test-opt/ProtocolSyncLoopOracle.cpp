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

#include <map>
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
        if (!left || left->storage.space != AddressSpace::VEC) {
            continue;
        }
        for (SyncAccessId b : second.accesses) {
            const SyncAccess* right = schedule.findAccess(b);
            const bool local = right && right->storage.space == AddressSpace::VEC &&
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

    bool run(bool* unsafeWitness = nullptr)
    {
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

bool checkLoopFrontierInterleavings(const StructuredSyncIR& schedule, unsigned trips)
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
    return oracle.run();
}
