// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/StructuredSyncComposition.h"
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <tuple>

using namespace mlir::pto::structured_sync;
namespace c = mlir::pto::structured_sync::composition;
static uint64_t checks = 0;
static void requireAt(bool value, unsigned line)
{
    ++checks;
    if (!value) {
        std::cerr << "failed check at line " << line << '\n';
        std::abort(); // Also enforce checks in Release/NDEBUG native builds.
    }
}
#define require(value) requireAt((value), __LINE__)

// Independent finite oracle. Payload has distinct issue/completion vertices;
// FIFO issue never implies completion. Events/barriers acquire actual queue
// prefixes. Check original hazards and one-bit event rearm causality by graph
// search, without using the production State or its loop summaries.
struct Oracle {
    std::vector<std::vector<unsigned>> edges;
    std::array<int, c::LaneCount> control;
    std::array<std::vector<unsigned>, c::LaneCount> issued;
    struct Access {
        unsigned issue, done;
        c::Effects effects;
    };
    std::vector<Access> accesses;
    using Key = std::tuple<unsigned, unsigned, unsigned>;
    struct Token {
        int publication = -1, consumption = -1;
    };
    std::map<Key, Token> tokens;
    std::vector<std::pair<unsigned, unsigned>> rearms;
    Oracle() { control.fill(-1); }
    unsigned vertex()
    {
        edges.emplace_back();
        return edges.size() - 1;
    }
    void edge(int a, unsigned b)
    {
        if (a >= 0)
            edges[a].push_back(b);
    }
    void command(unsigned lane, unsigned v, bool prefix)
    {
        edge(control[lane], v);
        if (prefix)
            for (auto done : issued[lane])
                edge(done, v);
        control[lane] = v;
    }
    void set(unsigned a, unsigned b, unsigned key)
    {
        auto v = vertex();
        edge(control[a], v);
        for (auto done : issued[a])
            edge(done, v);
        // Publication FIRE observes prior completion, but does not block later
        // source issue. Only WAIT and barriers advance the source control gate.
        auto& token = tokens[{a, b, key}];
        if (token.publication >= 0) {
            require(token.consumption >= 0);
            rearms.emplace_back(token.consumption, v);
        }
        token = {int(v), -1};
    }
    void wait(unsigned a, unsigned b, unsigned key)
    {
        auto v = vertex();
        command(b, v, false);
        auto& token = tokens[{a, b, key}];
        require(token.publication >= 0 && token.consumption < 0);
        edge(token.publication, v);
        token.consumption = v;
    }
    void mechanism(const c::Mechanism& m)
    {
        if (m.kind == c::Mechanism::Barrier) {
            auto v = vertex();
            command(m.first, v, true);
            return;
        }
        if (m.kind == c::Mechanism::Publish) {
            set(m.first, m.second, m.forwardKey);
            return;
        }
        if (m.kind == c::Mechanism::Acquire) {
            wait(m.first, m.second, m.forwardKey);
            return;
        }
        set(m.first, m.second, m.forwardKey);
        wait(m.first, m.second, m.forwardKey);
        set(m.second, m.first, m.reverseKey);
        wait(m.second, m.first, m.reverseKey);
    }
    void payload(const c::Node& n)
    {
        unsigned issue = vertex(), done = vertex();
        edge(control[n.lane], issue);
        edge(issue, done);
        issued[n.lane].push_back(done);
        accesses.push_back({issue, done, n.effects});
    }
    bool reaches(unsigned a, unsigned b)
    {
        std::vector<bool> seen(edges.size());
        std::vector<unsigned> todo{a};
        while (!todo.empty()) {
            unsigned x = todo.back();
            todo.pop_back();
            if (x == b)
                return true;
            if (seen[x])
                continue;
            seen[x] = true;
            for (auto y : edges[x])
                todo.push_back(y);
        }
        return false;
    }
    void check()
    {
        for (unsigned a = 0; a < accesses.size(); ++a)
            for (unsigned b = a + 1; b < accesses.size(); ++b) {
                bool hazard = false;
                for (unsigned i = 0; i < accesses[a].effects.size(); ++i) {
                    auto x = accesses[a].effects[i], y = accesses[b].effects[i];
                    hazard |= (x.writers && (y.readers || y.writers)) || (x.readers && y.writers);
                }
                if (hazard)
                    require(reaches(accesses[a].done, accesses[b].issue));
            }
        for (auto [wait, set] : rearms)
            require(reaches(wait, set));
        for (auto [k, t] : tokens)
            require(t.consumption >= 0);
        // Host enqueue order gives an acyclic serialization witness for this
        // packet vocabulary; every dependency points to a later graph vertex.
        for (unsigned a = 0; a < edges.size(); ++a)
            for (auto b : edges[a])
                require(a < b);
    }
};
static unsigned add(c::Program& p, c::Node::Kind kind, std::vector<unsigned> children = {})
{
    c::Node n;
    n.kind = kind;
    n.effects.resize(p.cells);
    n.children = std::move(children);
    p.nodes.push_back(n);
    return p.nodes.size() - 1;
}
static unsigned op(c::Program& p, unsigned lane, unsigned cell, bool write)
{
    auto id = add(p, c::Node::Operation);
    auto& n = p.nodes[id];
    n.lane = lane;
    (write ? n.effects[cell].writers : n.effects[cell].readers) = 1u << lane;
    return id;
}
static unsigned sequence(c::Program& p, std::vector<unsigned> children)
{
    children.push_back(add(p, c::Node::Sequence));
    return add(p, c::Node::Sequence, std::move(children));
}
// Decisions belong to a particular dynamic visit, not to a lexical loop or a
// shared global trip count. Keep this object across whole-program invocations.
struct ExecutionPolicy {
    std::map<unsigned, unsigned> visits;
    std::map<unsigned, unsigned> nextTrips, lastTrips, iteration;
    std::function<unsigned(unsigned, unsigned)> trips;
    std::function<unsigned(unsigned, unsigned)> choice;
};
static void execute(const c::Program& p, const c::Result& r, unsigned id, ExecutionPolicy& policy, Oracle& oracle)
{
    for (auto& m : r.before[id]) {
        bool participates = true;
        if (m.participation == c::Mechanism::First)
            participates = policy.iteration.at(m.loop) == 0;
        else if (m.participation == c::Mechanism::NonEmpty) {
            if (id <= m.loop) {
                if (!policy.nextTrips.count(m.loop))
                    policy.nextTrips[m.loop] = policy.trips(m.loop, policy.visits[m.loop]);
                participates = policy.nextTrips.at(m.loop) != 0;
            } else
                participates = policy.lastTrips.at(m.loop) != 0;
        }
        if (participates)
            oracle.mechanism(m);
    }
    const auto& n = p.nodes[id];
    auto run = [&](unsigned child) { execute(p, r, child, policy, oracle); };
    switch (n.kind) {
        case c::Node::Operation:
            oracle.payload(n);
            break;
        case c::Node::Sequence:
            for (auto x : n.children)
                run(x);
            break;
        case c::Node::Choice:
            run(n.children[policy.choice(id, policy.visits[id]++)]);
            break;
        case c::Node::For: {
            unsigned trips = policy.nextTrips.count(id) ? policy.nextTrips.at(id) : policy.trips(id, policy.visits[id]);
            ++policy.visits[id];
            policy.nextTrips.erase(id);
            for (unsigned i = 0; i < trips; ++i) {
                policy.iteration[id] = i;
                run(n.children[0]);
            }
            policy.lastTrips[id] = trips;
            break;
        }
        case c::Node::While: {
            unsigned trips = policy.trips(id, policy.visits[id]++);
            for (unsigned i = 0; i <= trips; ++i) {
                run(n.children[0]);
                if (i < trips)
                    run(n.children[1]);
            }
            break;
        }
    }
}
static void execute(
    const c::Program& p, const c::Result& r, unsigned id, unsigned trips, unsigned pattern, unsigned& branch,
    Oracle& oracle)
{
    ExecutionPolicy policy;
    policy.trips = [=](unsigned, unsigned) { return trips; };
    policy.choice = [&](unsigned, unsigned) { return (pattern >> (branch++ % 4)) & 1; };
    execute(p, r, id, policy, oracle);
}
int main()
{
    {
        // A parent-side completion handoff between two possible publication
        // cuts must not invalidate or invent shared incoming-prefix coverage.
        c::Program p;
        p.cells = 4;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto produce = op(p, b, 3, true);
        auto wa = op(p, a, 0, true), wb = op(p, a, 1, true);
        p.nodes[wb].effects[3].readers = 1u << a;
        auto unrelated = op(p, a, 2, true), rb = op(p, b, 1, false), ra = op(p, b, 0, false);
        auto loop = add(p, c::Node::For, {sequence(p, {unrelated, rb, ra})});
        p.nodes[loop].entryGuardStart = produce;
        auto after = op(p, a, 0, true);
        auto outer = add(p, c::Node::For, {sequence(p, {produce, wa, wb, loop, after})});
        sequence(p, {outer});
        auto plan = c::constructDemands(p);
        require(plan.success && plan.entryEpisodes == 1 && c::verifyDemands(p, plan.before).success);
        require(std::any_of(plan.before[wb].begin(), plan.before[wb].end(), [](const auto& m) {
            return m.kind == c::Mechanism::Acquire || m.kind == c::Mechanism::Rendezvous;
        }));
        ExecutionPolicy policy;
        policy.trips = [](unsigned id, unsigned visit) { return (id + visit) % 4; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 5; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
    }
    {
        c::Program p;
        p.cells = 3;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto wa = op(p, a, 0, true), wb = op(p, a, 1, true);
        auto unrelated = op(p, a, 2, true), ra = op(p, b, 0, false), rb = op(p, b, 1, false);
        auto loop = add(p, c::Node::For, {sequence(p, {unrelated, ra, rb})});
        p.nodes[loop].entryGuardStart = wa;
        auto after = op(p, a, 0, true);
        auto outer = add(p, c::Node::For, {sequence(p, {wa, wb, loop, after})});
        sequence(p, {outer});
        auto plan = c::constructDemands(p);
        require(plan.success && plan.entryEpisodes == 2 && c::verifyDemands(p, plan.before).success);
        require(std::count_if(plan.before[after].begin(), plan.before[after].end(), [](const auto& m) {
                    return m.participation == c::Mechanism::NonEmpty && m.kind == c::Mechanism::Publish;
                }) == 1);
        auto shared = p;
        auto& sharedChildren = shared.nodes[p.nodes[loop].children[0]].children;
        std::swap(sharedChildren[1], sharedChildren[2]);
        auto sharedPlan = c::constructDemands(shared);
        require(
            sharedPlan.success && sharedPlan.entryEpisodes == 1 && c::verifyDemands(shared, sharedPlan.before).success);
        auto without = p;
        without.nodes[loop].entryGuardStart = ~0u;
        require(c::constructDemands(without).success && c::constructDemands(without).entryEpisodes == 0);
        auto rejected = c::testing::constructDemandsRejectingEntryProposal(p);
        require(rejected.success && rejected.rejectedEntryProposals == 1 && rejected.entryEpisodes == 0);
        require(rejected.before == c::constructDemands(without).before);
        require(c::verifyDemands(p, rejected.before).success);
        require(!c::verifyDemands(without, plan.before).success);
        auto futureWrite = p;
        futureWrite.nodes[unrelated].effects[0].writers = 1u << a;
        require(!c::verifyDemands(futureWrite, plan.before).success);
        auto broken = plan.before;
        auto first = std::find_if(
            broken[ra].begin(), broken[ra].end(), [](const auto& m) { return m.participation == c::Mechanism::First; });
        require(first != broken[ra].end());
        for (bool forward : {false, true}) {
            auto collision = plan.before;
            c::Mechanism packet{c::Mechanism::Rendezvous, std::min(a, b), std::max(a, b), 0, 0};
            (forward ? packet.forwardKey : packet.reverseKey) = first->forwardKey;
            collision[ra].insert(collision[ra].begin(), packet);
            require(!c::verifyDemands(p, collision).success);
        }
        first->participation = c::Mechanism::Every;
        require(!c::verifyDemands(p, broken).success);
        ExecutionPolicy policy;
        policy.trips = [](unsigned id, unsigned visit) { return (id + visit) % 4; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 5; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
        Oracle sharedOracle;
        for (unsigned invocation = 0; invocation < 5; ++invocation) {
            execute(shared, sharedPlan, shared.nodes.size() - 1, policy, sharedOracle);
            sharedOracle.check();
        }
    }
    {
        // Many distinct logical generations fit one physical key per direction
        // because each return demand acknowledges the preceding acquisition.
        c::Program p;
        p.cells = 1;
        p.target.compilerKeys = {0, 1}; // zero remains the canonical reservation
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        std::vector<unsigned> body;
        for (unsigned i = 0; i < 12; ++i) {
            body.push_back(op(p, a, 0, true));
            body.push_back(op(p, b, 0, false));
        }
        auto loop = add(p, c::Node::For, {sequence(p, body)});
        sequence(p, {loop});
        auto plan = c::constructDemands(p);
        require(plan.success && c::verifyDemands(p, plan.before).success);
        require(plan.protocolKeys == 2 && plan.sharedProtocolKeys >= 20);
        require(plan.allocationFallbackScopes == 0 && plan.directHandoffs == 24);
        require(plan.demandFallbacks == 0);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return visit % 4; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 4; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
    }
    {
        // Overlapping publications cannot share just because their lexical
        // waits exist. Keep feasible handoffs, but never share their physical
        // colors with the following independently executing scope.
        c::Program p;
        p.cells = 3;
        p.target.compilerKeys = {0, 1};
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto wa = op(p, a, 0, true), wb = op(p, a, 1, true);
        auto ra = op(p, b, 0, false), rb = op(p, b, 1, false);
        auto first = add(p, c::Node::For, {sequence(p, {wa, wb, ra, rb})});
        auto wc = op(p, a, 2, true), rc = op(p, b, 2, false);
        auto second = add(p, c::Node::For, {sequence(p, {wc, rc})});
        sequence(p, {first, second});
        auto plan = c::testing::constructDemandsWithoutAllocationReplay(p);
        require(plan.success && c::verifyDemands(p, plan.before).success);
        require(plan.allocationFallbackScopes == 2 && plan.protocolKeys == 2);
        require(plan.demandFallbacks > 0 && plan.directHandoffs == 2);
        ExecutionPolicy policy;
        policy.trips = [=](unsigned id, unsigned visit) { return (id + visit) % 3; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 4; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
    }
    {
        // Scarcity in the middle of a scope supplies more than the original
        // early prefix. Reuse it for later cells without skipping a new write.
        c::Program p;
        p.cells = 3;
        p.target.compilerKeys = {0, 1};
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto wa = op(p, a, 0, true), wb = op(p, a, 1, true), wc = op(p, a, 2, true);
        auto ra = op(p, b, 0, false), rb = op(p, b, 1, false), rc = op(p, b, 2, false);
        auto rewrite = op(p, a, 0, true), reread = op(p, b, 0, false);
        auto body = sequence(p, {wa, wb, wc, ra, rb, rc, rewrite, reread});
        auto loop = add(p, c::Node::For, {body});
        sequence(p, {loop});
        auto old = c::testing::constructDemandsWithoutAllocationReplay(p);
        auto plan = c::constructDemands(p);
        require(old.success && plan.success && c::verifyDemands(p, plan.before).success);
        require(plan.allocationReplays == 1 && plan.rejectedAllocationReplays == 0);
        require(plan.replayedFallbackDemands > 0 && plan.replayCommandsRemoved > 0);
        // A nonzero canonical ID must not collide with virtual logical IDs.
        auto shifted = p;
        shifted.target.compilerKeys = {3, 5};
        auto shiftedPlan = c::constructDemands(shifted);
        require(shiftedPlan.success && c::verifyDemands(shifted, shiftedPlan.before).success);
        require(shiftedPlan.replayedFallbackDemands > 0 && shiftedPlan.replayCommandsRemoved > 0);
        auto rejected = c::testing::constructDemandsRejectingAllocationReplay(p);
        require(rejected.success && rejected.rejectedAllocationReplays == 1 && rejected.before == old.before);
        ExecutionPolicy policy;
        policy.trips = [](unsigned id, unsigned visit) { return (id + visit) % 4; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        Oracle shiftedOracle;
        for (unsigned invocation = 0; invocation < 5; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
            execute(shifted, shiftedPlan, shifted.nodes.size() - 1, policy, shiftedOracle);
            shiftedOracle.check();
        }
    }
    {
        // The shared Program contract assigns effects only to physical leaves.
        // All constructors/checkers reject a malformed structural effect row.
        c::Program p;
        p.cells = 1;
        auto payload = op(p, unsigned(Pipe::V), 0, true);
        auto root = sequence(p, {payload});
        auto actual = c::construct(p).before;
        p.nodes[root].effects[0].writers = 1u << p.nodes[root].lane;
        require(!c::construct(p).success && !c::verify(p, actual).success);
        require(!c::constructCuts(p).success && !c::verifyCuts(p, actual).success);
        require(!c::constructDemands(p).success && !c::verifyDemands(p, actual).success);
    }
    {
        // Matching lexical endpoints alone do not prove rearm. This complete
        // straight-line word still needs a causal return before a repeated visit.
        c::Program p;
        p.cells = 1;
        auto start = add(p, c::Node::Sequence), finish = add(p, c::Node::Sequence);
        sequence(p, {start, finish});
        std::vector<std::vector<c::Mechanism>> actual(p.nodes.size());
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        actual[start].push_back({c::Mechanism::Publish, a, b, 1});
        actual[finish].push_back({c::Mechanism::Acquire, a, b, 1});
        require(!c::verifyDemands(p, actual).success);
        actual[finish].push_back({c::Mechanism::Publish, b, a, 1});
        actual[finish].push_back({c::Mechanism::Acquire, b, a, 1});
        require(c::verifyDemands(p, actual).success);
        actual[start][0].forwardKey = 0; // reserved conservative key
        require(!c::verifyDemands(p, actual).success);
    }
    {
        c::Program p;
        p.cells = 1;
        auto write = op(p, unsigned(Pipe::MTE2), 0, true);
        auto read = op(p, unsigned(Pipe::V), 0, false);
        auto loop = add(p, c::Node::For, {sequence(p, {write, read})});
        sequence(p, {loop});
        auto plan = c::constructDemands(p);
        require(plan.success && c::verifyDemands(p, plan.before).success);
        require(plan.completionRefinements == 1 && plan.rejectedRefinements == 0);
        require(c::verifyDemands(p, plan.before).ownedRefinements > 0);
        auto rejected = c::testing::constructDemandsRejectingRefinement(p);
        require(rejected.success && rejected.completionRefinements == 1 && rejected.rejectedRefinements == 1);
        require(c::verifyDemands(p, rejected.before).success);
        require(plan.reusedAcknowledgments == 2 && plan.sharedAcknowledgments == 0);
        for (unsigned trips = 0; trips < 5; ++trips) {
            unsigned branch = 0;
            Oracle oracle;
            execute(p, plan, p.nodes.size() - 1, trips, 0, branch, oracle);
            execute(p, plan, p.nodes.size() - 1, 4 - trips, 0, branch, oracle);
            oracle.check();
            Oracle rollback;
            execute(p, rejected, p.nodes.size() - 1, trips, 0, branch, rollback);
            execute(p, rejected, p.nodes.size() - 1, 4 - trips, 0, branch, rollback);
            rollback.check();
        }
        p.target.compilerKeys = {0};
        auto scarce = c::constructDemands(p);
        require(scarce.success && c::verifyDemands(p, scarce.before).success);
        require(scarce.directHandoffs == 0 && scarce.demandFallbacks != 0);
        unsigned packets = 0;
        for (const auto& commands : scarce.before)
            for (const auto& m : commands)
                packets += m.kind == c::Mechanism::Rendezvous;
        require(packets != 0);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return visit % 4; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle repeated;
        for (unsigned invocation = 0; invocation < 5; ++invocation) {
            execute(p, scarce, p.nodes.size() - 1, policy, repeated);
            repeated.check();
        }
        auto corrupt = scarce.before;
        bool changed = false;
        for (auto& commands : corrupt)
            for (auto& m : commands)
                if (!changed && m.kind == c::Mechanism::Rendezvous) {
                    std::swap(m.first, m.second);
                    changed = true;
                }
        require(changed && !c::verifyDemands(p, corrupt).success);
    }
    {
        // One consumer needs two physical cells; completion is a producer
        // prefix, not two cell-owned protocols. X after that prefix must not
        // become a prerequisite, and later source work must remain pending.
        c::Program p;
        p.cells = 4;
        unsigned source = unsigned(Pipe::MTE2), target = unsigned(Pipe::V);
        auto a = op(p, source, 0, true), b = op(p, source, 1, true);
        auto x = op(p, source, 2, true);
        auto consumer = op(p, target, 0, false);
        p.nodes[consumer].effects[1].readers = 1u << target;
        auto newer = op(p, source, 3, true);
        auto second = op(p, target, 3, false);
        auto reused = op(p, target, 0, false);
        sequence(p, {a, b, x, consumer, newer, second, reused});
        auto plan = c::constructDemands(p);
        require(plan.success);
        require(c::verifyDemands(p, plan.before).success);
        require(plan.directHandoffs == 2 && plan.sharedAcknowledgments == 1);
        require(plan.demands[0].publication == x && plan.demands[0].acquisition == consumer);
        require(plan.demands[0].cells.size() == 2);
        require(plan.before[reused].empty());
        unsigned branch = 0;
        Oracle oracle;
        execute(p, plan, p.nodes.size() - 1, 1, 0, branch, oracle);
        oracle.check();
        // The independent graph also checks the absent, unnecessary ordering.
        require(!oracle.reaches(oracle.accesses[2].done, oracle.accesses[3].issue));
        auto broken = plan.before;
        broken[a].push_back(broken[x].front());
        broken[x].erase(broken[x].begin());
        require(!c::verifyDemands(p, broken).success);
        broken = plan.before;
        broken[second].resize(broken[second].size() - 2); // drop shared reply
        require(!c::verifyDemands(p, broken).success);
        broken = plan.before;
        broken[consumer].clear();
        require(!c::verifyDemands(p, broken).success);
    }
    // A publication for y can complete an earlier x write. It cannot complete
    // the same write moved AFTER that publication. x deliberately has a return
    // access on its first lane, so it gets no independent cyclic precision.
    for (unsigned variant = 0; variant < 4; ++variant) {
        bool newer = variant != 0;
        c::Program p;
        p.cells = 2;
        auto x = op(p, unsigned(Pipe::MTE2), 1, true);
        auto y = op(p, unsigned(Pipe::MTE2), 0, true);
        auto readY = op(p, unsigned(Pipe::V), 0, false);
        auto readX = op(p, unsigned(Pipe::V), 1, false);
        auto returnX = op(p, unsigned(Pipe::MTE2), 1, false);
        auto writeX = x;
        if (variant == 2) {
            auto yes = sequence(p, {x}), no = add(p, c::Node::Sequence);
            writeX = add(p, c::Node::Choice, {yes, no});
        }
        if (variant == 3)
            writeX = add(p, c::Node::For, {sequence(p, {x})});
        auto body = sequence(
            p, newer ? std::vector<unsigned>{y, writeX, readY, readX, returnX} :
                       std::vector<unsigned>{x, y, readY, readX, returnX});
        auto loop = add(p, c::Node::For, {body});
        sequence(p, {loop});
        auto plan = c::constructCuts(p);
        require(plan.success);
        require(c::verifyCuts(p, plan.before).success);
        require(plan.cutCycles == 1);
        auto fallback = [](const c::Mechanism& m) {
            return m.kind == c::Mechanism::Barrier || m.kind == c::Mechanism::Rendezvous;
        };
        require(std::any_of(plan.before[readX].begin(), plan.before[readX].end(), fallback) == newer);
        for (unsigned trips = 0; trips < 4; ++trips)
            for (unsigned pattern = 0; pattern < 8; ++pattern) {
                unsigned branch = 0;
                Oracle oracle;
                execute(p, plan, p.nodes.size() - 1, trips, pattern, branch, oracle);
                oracle.check();
            }
        if (newer) {
            auto& commands = plan.before[readX];
            commands.erase(std::remove_if(commands.begin(), commands.end(), fallback), commands.end());
            require(!c::verifyCuts(p, plan.before).success);
        }
    }
    for (unsigned groups : {2u, 4u, 32u, 34u}) {
        c::Program p;
        p.cells = 1;
        auto incoming = op(p, unsigned(Pipe::MTE3), 0, true);
        std::vector<unsigned> body;
        for (unsigned i = 0; i < groups; ++i)
            body.push_back(op(p, unsigned(i % 2 ? Pipe::V : Pipe::MTE2), 0, i % 2 == 0));
        auto inner = add(p, c::Node::For, {sequence(p, std::move(body))});
        auto outside = op(p, unsigned(Pipe::MTE3), 0, true);
        auto outer = add(p, c::Node::For, {sequence(p, {inner, outside})});
        sequence(p, {incoming, outer});
        auto plan = c::constructCuts(p);
        require(plan.success);
        require(c::verifyCuts(p, plan.before).success);
        for (unsigned trips = 0; trips < 3; ++trips) {
            unsigned branch = 0;
            Oracle oracle;
            execute(p, plan, p.nodes.size() - 1, trips, 0, branch, oracle);
            oracle.check();
        }
        // Scarce keys may reduce optional precision, never baseline support.
        p.target.compilerKeys = {0};
        auto conservative = c::constructCuts(p);
        require(conservative.success);
        require(c::verifyCuts(p, conservative.before).success);
    }
    {
        c::Program p;
        p.cells = 1;
        auto a = op(p, unsigned(Pipe::MTE2), 0, true);
        auto read = op(p, unsigned(Pipe::V), 0, false);
        auto empty = add(p, c::Node::Sequence);
        auto choice = add(p, c::Node::Choice, {sequence(p, {read}), empty});
        auto before = sequence(p, {a, choice});
        auto after = add(p, c::Node::Sequence);
        auto loop = add(p, c::Node::While, {before, after});
        sequence(p, {loop});
        auto plan = c::constructCuts(p);
        require(plan.success);
        require(c::verifyCuts(p, plan.before).success);
        for (unsigned trips = 0; trips < 3; ++trips)
            for (unsigned pattern = 0; pattern < 16; ++pattern) {
                unsigned branch = 0;
                Oracle oracle;
                execute(p, plan, p.nodes.size() - 1, trips, pattern, branch, oracle);
                oracle.check();
            }
    }
    for (unsigned depth : {2u, 3u}) {
        c::Program p;
        p.cells = 2 * depth;
        std::vector<unsigned> slots;
        for (unsigned slot = 0; slot < depth; ++slot) {
            auto load = op(p, unsigned(Pipe::MTE2), slot, true);
            auto compute = op(p, unsigned(Pipe::V), slot, false);
            p.nodes[compute].effects[depth + slot].writers = 1u << unsigned(Pipe::V);
            auto store = op(p, unsigned(Pipe::MTE3), depth + slot, false);
            auto end = add(p, c::Node::Sequence);
            auto body = add(p, c::Node::Sequence, {load, compute, store, end});
            auto empty = add(p, c::Node::Sequence);
            slots.push_back(add(p, c::Node::Choice, {body, empty}));
        }
        slots.push_back(add(p, c::Node::Sequence));
        auto body = add(p, c::Node::Sequence, slots);
        auto loop = add(p, c::Node::For, {body});
        auto end = add(p, c::Node::Sequence);
        add(p, c::Node::Sequence, {loop, end});
        auto plan = c::constructCuts(p);
        require(plan.success);
        require(c::verifyCuts(p, plan.before).success);
        unsigned publications = 0, conservative = 0;
        for (auto& site : plan.before)
            for (auto& m : site) {
                publications += m.kind == c::Mechanism::Publish;
                conservative += m.kind == c::Mechanism::Rendezvous || m.kind == c::Mechanism::Barrier;
            }
        require(publications == 8 * depth);
        require(conservative == 0);
        for (unsigned trips = 0; trips < 5; ++trips)
            for (unsigned pattern = 0; pattern < 16; ++pattern) {
                unsigned branch = 0;
                Oracle oracle;
                execute(p, plan, p.nodes.size() - 1, trips, pattern, branch, oracle);
                oracle.check();
                // A second invocation uses the SAME physical key population. The
                // final release consumption must causally precede its next seed FIRE.
                execute(p, plan, p.nodes.size() - 1, trips, pattern ^ 15, branch, oracle);
                oracle.check();
            }
        auto broken = plan.before;
        broken[loop].erase(broken[loop].begin());
        require(!c::verifyCuts(p, broken).success);
        broken = plan.before;
        broken[end].clear();
        require(!c::verifyCuts(p, broken).success);
    }
    {
        // A pending B write remains pending at B after its reply SET, even
        // though A's reply WAIT has acquired that write. Challenge the actual
        // consumer next, not merely a standalone rendezvous mask helper.
        c::Program p;
        p.cells = 1;
        auto write = op(p, unsigned(Pipe::MTE2), 0, true);
        auto read = op(p, unsigned(Pipe::V), 0, false);
        auto overwrite = op(p, unsigned(Pipe::MTE2), 0, true);
        add(p, c::Node::Sequence, {write, read, overwrite});
        auto plan = c::construct(p);
        require(plan.success);
        unsigned branch = 0;
        Oracle oracle;
        execute(p, plan, p.nodes.size() - 1, 1, 0, branch, oracle);
        oracle.check();
        c::State state(1);
        state.seed(p.nodes[write].effects);
        state.rendezvous(unsigned(Pipe::V), unsigned(Pipe::MTE2));
        require(state.demands(unsigned(Pipe::MTE2), p.nodes[overwrite].effects) != 0);
        require(state.demands(unsigned(Pipe::V), p.nodes[read].effects) == 0);
    }
    {
        // The round trip is needed for a DIFFERENT cell. B's later overwrite of
        // x must still wait for its earlier write of x; the reply SET did not
        // drain B. This fails if either production or the oracle treats SET FIRE
        // as a gate on subsequent source issue.
        c::Program p;
        p.cells = 2;
        auto x = op(p, unsigned(Pipe::MTE2), 0, true);
        auto y = op(p, unsigned(Pipe::V), 1, true);
        auto readY = op(p, unsigned(Pipe::MTE2), 1, false);
        auto overwriteX = op(p, unsigned(Pipe::MTE2), 0, true);
        add(p, c::Node::Sequence, {x, y, readY, overwriteX});
        auto plan = c::construct(p);
        require(plan.success);
        require(plan.before[overwriteX].size() == 1);
        require(plan.before[overwriteX][0].kind == c::Mechanism::Barrier);
        unsigned branch = 0;
        Oracle oracle;
        execute(p, plan, p.nodes.size() - 1, 1, 0, branch, oracle);
        oracle.check();
        plan.before[overwriteX].clear();
        require(!c::verify(p, plan.before).success);
    }
    std::mt19937 random(19371);
    // Independently varying siblings, nested invocations, skipped paths, and
    // repeated entry with the same physical key population. The asymmetric
    // child counts specifically cover the old all-loops-use-trips blind spot.
    for (unsigned sample = 0; sample < 32; ++sample) {
        c::Program p;
        p.cells = 2;
        auto a = op(p, unsigned(Pipe::MTE2), 0, true);
        auto b = op(p, unsigned(Pipe::V), 0, false);
        auto first = add(p, c::Node::For, {sequence(p, {a, b})});
        auto c0 = op(p, unsigned(Pipe::V), 0, true);
        auto d = op(p, unsigned(Pipe::MTE3), 0, false);
        auto choice = add(p, c::Node::Choice, {sequence(p, {c0, d}), add(p, c::Node::Sequence)});
        auto second = add(p, c::Node::For, {sequence(p, {choice})});
        auto before = op(p, unsigned(Pipe::MTE2), 1, true);
        auto after = op(p, unsigned(Pipe::V), 1, false);
        auto w = add(p, c::Node::While, {sequence(p, {before}), sequence(p, {after})});
        auto outer = add(p, c::Node::For, {sequence(p, {first, second, w})});
        sequence(p, {outer});
        for (unsigned mode : {0u, 1u, 2u}) {
            auto plan = mode == 2 ? c::constructDemands(p) : mode == 1 ? c::constructCuts(p) : c::construct(p);
            require(plan.success);
            require((mode == 2 ? c::verifyDemands(p, plan.before) :
                     mode == 1 ? c::verifyCuts(p, plan.before) :
                                 c::verify(p, plan.before))
                        .success);
            ExecutionPolicy policy;
            policy.trips = [=](unsigned node, unsigned invocation) {
                if (node == outer)
                    return 2u;
                if (invocation == 0 && node == first)
                    return 0u;
                if (invocation == 0 && node == second)
                    return 3u;
                return (sample + 3 * node + invocation * 7) % 4;
            };
            policy.choice = [=](unsigned node, unsigned invocation) {
                return ((sample * 13 + node * 7 + invocation * 5) >> (invocation % 5)) & 1;
            };
            Oracle oracle;
            for (unsigned invocation = 0; invocation < 3; ++invocation) {
                execute(p, plan, p.nodes.size() - 1, policy, oracle);
                oracle.check();
            }
            require(policy.visits[first] == 6 && policy.visits[second] == 6);
        }
    }
    for (unsigned sample = 0; sample < 64; ++sample) {
        c::Program p;
        p.cells = 3;
        unsigned lanes[] = {unsigned(Pipe::V), unsigned(Pipe::MTE2), unsigned(Pipe::MTE3)};
        std::vector<unsigned> body;
        for (unsigned i = 0; i < 8; ++i) {
            if (i == 2)
                body.push_back(op(p, unsigned(Pipe::MTE2), 0, true));
            if (i == 6)
                body.push_back(op(p, unsigned(Pipe::V), 0, false));
            auto access = op(p, lanes[random() % 3], 1 + random() % 2, random() % 2);
            if (random() % 2) {
                auto yes = sequence(p, {access}), no = add(p, c::Node::Sequence);
                access = add(p, c::Node::Choice, {yes, no});
            }
            body.push_back(access);
        }
        auto loop = add(p, c::Node::For, {sequence(p, std::move(body))});
        sequence(p, {loop});
        auto plan = c::constructCuts(p);
        require(plan.success);
        require(c::verifyCuts(p, plan.before).success);
        for (unsigned trips = 0; trips < 3; ++trips)
            for (unsigned pattern = 0; pattern < 8; ++pattern) {
                unsigned branch = 0;
                Oracle oracle;
                execute(p, plan, p.nodes.size() - 1, trips, pattern, branch, oracle);
                oracle.check();
                execute(p, plan, p.nodes.size() - 1, trips, pattern ^ 7, branch, oracle);
                oracle.check();
            }
    }
    for (unsigned sample = 0; sample < 240; ++sample) {
        c::Program p;
        p.cells = 3;
        unsigned lanes[] = {unsigned(Pipe::V), unsigned(Pipe::MTE2), unsigned(Pipe::MTE3)};
        auto payload = [&]() { return op(p, lanes[random() % 3], random() % 3, random() % 2); };
        auto a = payload(), b = payload();
        auto choice = add(p, c::Node::Choice, {a, b});
        auto tail = payload();
        auto body = add(p, c::Node::Sequence, {choice, tail});
        auto loop = add(p, c::Node::For, {body});
        auto before = payload(), after = payload();
        auto other = add(p, c::Node::While, {before, after});
        auto siblings = add(p, c::Node::Sequence, {loop, other});
        add(p, c::Node::For, {siblings});
        auto result = c::construct(p);
        require(result.success);
        require(result.nodeVisits == p.nodes.size());
        require(c::verify(p, result.before).success);
        auto demanded = c::constructDemands(p);
        require(demanded.success);
        require(c::verifyDemands(p, demanded.before).success);
        ExecutionPolicy varying;
        varying.trips = [=](unsigned node, unsigned invocation) { return (sample * 11 + node + 3 * invocation) % 3; };
        varying.choice = [=](unsigned node, unsigned invocation) {
            return ((sample + 3 * node + invocation * 7) >> (invocation % 3)) & 1;
        };
        Oracle repeated;
        for (unsigned invocation = 0; invocation < 3; ++invocation) {
            execute(p, demanded, p.nodes.size() - 1, varying, repeated);
            repeated.check();
        }
        for (unsigned trips = 0; trips < 3; ++trips)
            for (unsigned pattern = 0; pattern < 8; ++pattern) {
                unsigned branch = 0;
                Oracle oracle;
                execute(p, result, p.nodes.size() - 1, trips, pattern, branch, oracle);
                oracle.check();
            }
        for (auto& site : result.before)
            if (!site.empty() && site[0].kind == c::Mechanism::Rendezvous) {
                ++site[0].forwardKey;
                require(!c::verify(p, result.before).success);
                break;
            }
    }
    c::Program p;
    p.core = Core::AIC;
    p.cells = 1;
    auto a = op(p, unsigned(Pipe::FIX), 0, true), b = op(p, unsigned(Pipe::MTE2), 0, false);
    add(p, c::Node::Sequence, {a, b});
    auto routed = c::construct(p);
    require(routed.success);
    require(routed.before[b].size() == 2);
    require(c::verify(p, routed.before).success);
    routed.before[b].clear();
    require(!c::verify(p, routed.before).success);
    // A valid completion rendezvous must NOT discharge same-address GM
    // publication, including after a barrier or on a widened loop backedge.
    auto complete = c::construct(p);
    require(complete.success);
    p.globalMemory = {true};
    require(!c::construct(p).success);
    require(!c::verify(p, complete.before).success);
    p.globalMemory.clear();
    p.target.compilerKeys.clear();
    require(!c::construct(p).success);
    p.cells = c::MaxCells + 1;
    require(!c::construct(p).success);
    // Fixed resource width: increasing sequence length does not create larger
    // occurrence spaces or closure matrices. Both passes visit each node once.
    for (unsigned size : {64u, 256u, 1024u, 4096u}) {
        c::Program linear;
        linear.cells = 4;
        std::vector<unsigned> sequence;
        for (unsigned i = 0; i < size; ++i)
            sequence.push_back(op(linear, unsigned(Pipe::V), i % 4, true));
        add(linear, c::Node::Sequence, sequence);
        auto plan = c::construct(linear);
        require(plan.success);
        auto checked = c::verify(linear, plan.before);
        require(checked.success);
        require(plan.nodeVisits == size + 1 && checked.nodeVisits == size + 1);
        require(plan.cellVisits == (size + 1) * 4 * c::LaneCount);
    }
    c::Program deep;
    deep.cells = c::MaxCells;
    unsigned root = op(deep, unsigned(Pipe::V), c::MaxCells - 1, true);
    for (unsigned depth = 0; depth < 128; ++depth)
        root = add(deep, c::Node::For, {root});
    auto nested = c::construct(deep);
    require(nested.success);
    require(nested.nodeVisits == deep.nodes.size());
    require(c::verify(deep, nested.before).success);
    auto bounded = c::constructDemands(deep);
    require(bounded.success);
    auto boundedCheck = c::verifyDemands(deep, bounded.before);
    require(boundedCheck.success);
    require(bounded.nodeVisits <= 8 * deep.nodes.size());
    require(boundedCheck.nodeVisits <= 3 * deep.nodes.size());
    std::cout << checks << " compositional native-helper/independent finite graph assertions passed\n";
}
