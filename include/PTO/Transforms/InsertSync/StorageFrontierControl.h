// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERCONTROL_H
#define PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERCONTROL_H

#include "PTO/Transforms/InsertSync/StorageFrontierRelations.h"
#include <map>
#include <set>
#include <tuple>

namespace mlir::pto::insert_sync_frontier {
struct GuardVariable {
    unsigned scope = kInvalid;
    // Empty means an unknown integer domain, not an impossible value.
    std::vector<int64_t> finiteValues;
};
struct EdgeAssumption {
    unsigned from = kInvalid, to = kInvalid, variable = kInvalid;
    int64_t value = 0;
    bool equal = true;
};
struct GuardUpdate {
    unsigned variable = kInvalid;
    enum class Kind { Assign, AddModulo };
    Kind kind = Kind::Assign;
    int64_t value = 0;
    unsigned modulus = 0;
};
struct GuardControl {
    std::vector<GuardVariable> variables;
    std::vector<EdgeAssumption> assumptions;
    // Forget SSA definitions recreated by this dynamic loop iteration.
    std::vector<std::vector<unsigned>> forgetAtNode;
    std::vector<std::vector<GuardUpdate>> updatesAtNode;
};
struct ValueConstraint {
    std::optional<int64_t> equal;
    std::set<int64_t> excluded;
    bool operator<(const ValueConstraint& v) const { return std::tie(equal, excluded) < std::tie(v.equal, v.excluded); }
};
using GuardEnvironment = std::map<unsigned, ValueConstraint>;
inline bool assume(GuardEnvironment& env, const GuardVariable& domain, unsigned var, int64_t value, bool equal)
{
    auto& state = env[var];
    if (
        equal) {
        if (
            (state.equal && *state.equal != value) || state.excluded.count(value)) {
            return false;
        }
        state.equal = value;
        state.excluded.clear();
    } else {
        if (
            state.equal) {
            return *state.equal != value;
        }
        state.excluded.insert(value);
    }
    if (
        !domain.finiteValues.empty()) {
        bool possible = false;
        for (
            int64_t v : domain.finiteValues) {
            possible |= state.equal ? *state.equal == v : !state.excluded.count(v);
        }
        if (
            !possible) {
            return false;
        }
    }
    return true;
}
struct GuardPartitionResult {
    enum class Status { Complete, InvalidInput, AnalysisLimit };
    Status status = Status::InvalidInput;
    Program program;
    std::vector<unsigned> origins;
    std::vector<GuardEnvironment> environments;
    std::vector<std::vector<unsigned>> copies;
};
// Finite symbolic path partitioning. This enumerates guard STATES, never loop
// iterations. Backedges return to the same state after the appropriate scope
// invalidation. Unknown equalities add possible paths rather than proving them.
inline GuardPartitionResult partitionGuards(
    const Program& input, const GuardControl& control, Budget& budget, unsigned maximumNodes = 8192)
{
    GuardPartitionResult result;
    if (
        control.variables.size() > 128) {
        result.status = GuardPartitionResult::Status::AnalysisLimit;
        return result;
    }
    if (
        !valid(input) || (!control.forgetAtNode.empty() && control.forgetAtNode.size() != input.nodes.size()) ||
        (!control.updatesAtNode.empty() && control.updatesAtNode.size() != input.nodes.size())) {
        return result;
    }
    std::map<std::pair<unsigned, unsigned>, std::vector<EdgeAssumption>> byEdge;
    for (
        const auto& a : control.assumptions) {
        if (
            a.from >= input.nodes.size() || a.to >= input.nodes.size() || a.variable >= control.variables.size() ||
            std::find(input.nodes[a.from].next.begin(), input.nodes[a.from].next.end(), a.to) ==
                input.nodes[a.from].next.end()) {
            return result;
        }
        byEdge[{a.from, a.to}].push_back(a);
    }
    for (
        const auto& list : control.forgetAtNode) {
        for (
            unsigned v : list) {
            if (
                v >= control.variables.size()) {
                return result;
            }
        }
    }
    for (
        const auto& list : control.updatesAtNode) {
        for (
            const auto& u : list) {
            if (
                u.variable >= control.variables.size() ||
                (u.kind == GuardUpdate::Kind::AddModulo && (!u.modulus || u.modulus > 16))) {
                return result;
            }
        }
    }
    result.program.lanes = input.lanes;
    result.program.phaseLane = input.phaseLane;
    result.program.keys = input.keys;
    result.program.allowUnrepresentedPhases = true;
    result.copies.resize(input.nodes.size());
    std::map<std::pair<unsigned, GuardEnvironment>, unsigned> ids;
    std::deque<unsigned> queue;
    auto create = [&](unsigned node, GuardEnvironment env) -> unsigned {
        if (
            !control.forgetAtNode.empty()) {
            for (
                unsigned variable : control.forgetAtNode[node]) {
                env.erase(variable);
            }
        }
        if (
            !control.updatesAtNode.empty()) {
            for (
                const auto& update : control.updatesAtNode[node]) {
                if (
                    update.kind == GuardUpdate::Kind::Assign) {
                    env[update.variable] = ValueConstraint{update.value, {}};
                } else {
                    auto old = env.find(update.variable);
                    if (
                        old != env.end() && old->second.equal) {
                        old->second = ValueConstraint{
                            int64_t(residue(__int128(*old->second.equal) + update.value, update.modulus)), {}};
                    } else {
                        // An unproved old residue cannot justify a fabricated successor.
                        env.erase(update.variable);
                    }
                }
            }
        }
        auto key = std::make_pair(node, env);
        if (
            auto it = ids.find(key); it != ids.end()) {
            return it->second;
        }
        if (
            result.program.nodes.size() >= maximumNodes || !budget.spend(1 + env.size())) {
            return kInvalid;
        }
        unsigned id = result.program.nodes.size();
        ids.emplace(std::move(key), id);
        Node copy = input.nodes[node];
        copy.next.clear();
        result.program.nodes.push_back(std::move(copy));
        result.origins.push_back(node);
        result.environments.push_back(std::move(env));
        result.copies[node].push_back(id);
        queue.push_back(id);
        return id;
    };
    if (
        create(0, {}) == kInvalid) {
        result.status = GuardPartitionResult::Status::AnalysisLimit;
        return result;
    }
    while (
        !queue.empty()) {
        unsigned current = queue.front();
        queue.pop_front();
        unsigned original = result.origins[current];
        for (
            unsigned next : input.nodes[original].next) {
            if (
                !budget.spend(1)) {
                result.status = GuardPartitionResult::Status::AnalysisLimit;
                return result;
            }
            auto env = result.environments[current];
            bool possible = true;
            for (
                const auto& a : byEdge[{original, next}]) {
                if (
                    !assume(env, control.variables[a.variable], a.variable, a.value, a.equal)) {
                    possible = false;
                    break;
                }
            }
            if (
                !possible) {
                continue;
            }
            unsigned target = create(next, std::move(env));
            if (
                target == kInvalid) {
                result.status = GuardPartitionResult::Status::AnalysisLimit;
                return result;
            }
            result.program.nodes[current].next.push_back(target);
        }
    }
    // Keep only connected boundary alternatives, not a Cartesian product of
    // mutually exclusive paths. Stop at the first exit of this region.
    for (
        const auto& r : input.regions) {
        for (
            unsigned entry : result.copies[r.entry]) {
            std::vector<bool> seen(result.program.nodes.size(), false);
            std::deque<unsigned> work{entry};
            seen[entry] = true;
            while (
                !work.empty()) {
                if (
                    !budget.spend()) {
                    result.status = GuardPartitionResult::Status::AnalysisLimit;
                    return result;
                }
                unsigned n = work.front();
                work.pop_front();
                if (
                    result.origins[n] == r.exit) {
                    if (
                        result.program.regions.size() >= 4096) {
                        result.status = GuardPartitionResult::Status::AnalysisLimit;
                        return result;
                    }
                    RegionScope region{entry, n, r.kind, kInvalid};
                    Bits included(result.program.nodes.size());
                    std::deque<unsigned> members{entry};
                    while (!members.empty()) {
                        unsigned member = members.front(); members.pop_front();
                        if (included.test(member)) continue;
                        if (!budget.spend(1 + result.program.nodes[member].next.size())) {
                            result.status = GuardPartitionResult::Status::AnalysisLimit;
                            return result;
                        }
                        included.set(member);
                        region.members.push_back(member);
                        if (result.origins[member] == r.exit) continue;
                        for (unsigned next : result.program.nodes[member].next)
                            if (result.origins[next] >= r.entry && result.origins[next] <= r.exit)
                                members.push_back(next);
                    }
                    result.program.regions.push_back(std::move(region));
                    continue;
                }
                for (
                    unsigned next : result.program.nodes[n].next) {
                    unsigned origin = result.origins[next];
                    if (
                        origin < r.entry || origin > r.exit || seen[next]) {
                        continue;
                    }
                    seen[next] = true;
                    work.push_back(next);
                }
            }
        }
    }
    result.status = GuardPartitionResult::Status::Complete;
    return result;
}
// Shared lowering of the exact loop-control language 0 | one | first middle* last.
// Body callbacks retain static effect identities; they must not clone physical
// IR. A middle backedge represents arbitrary repetition rather than a test horizon.
struct IterationClass {
    bool first = false, last = false;
};
template <class Tails, class Gate, class Body, class Merge, class Backedge>
Tails structuredLoopTransfer(Tails incoming, Gate gate, Body body, Merge merge, Backedge backedge)
{
    Tails empty = gate(incoming, 0);
    Tails one = body(gate(incoming, 1), IterationClass{true, true});
    Tails first = body(gate(incoming, 2), IterationClass{true, false});
    Tails hub = merge(first);
    Tails middle = body(hub, IterationClass{false, false});
    backedge(middle, hub);
    Tails last = body(hub, IterationClass{false, true});
    empty.insert(empty.end(), one.begin(), one.end());
    empty.insert(empty.end(), last.begin(), last.end());
    return merge(empty);
}
} // namespace mlir::pto::insert_sync_frontier
#endif
