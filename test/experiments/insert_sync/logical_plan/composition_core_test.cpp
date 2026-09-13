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
    uint64_t commands = 0;
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
        ++commands;
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
        unsigned first = m.word == ~0u ? 0 : unsigned(*p.nodes[m.word].firstActive() - p.nodes[m.word].periodicLower);
        if (m.participation == c::Mechanism::First)
            participates = policy.iteration.at(m.loop) == 0;
        else if (m.participation == c::Mechanism::Previous)
            participates = policy.iteration.at(m.loop) != first;
        else if (m.participation == c::Mechanism::LoopExit)
            participates = policy.lastTrips.at(m.loop) > first;
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
            if (n.periodicOwner != ~0u) {
                unsigned ordinal = policy.iteration.at(n.periodicOwner);
                bool take = n.periodicResidues & (uint32_t(1) << (ordinal % n.periodicPeriod));
                run(n.children[take ? 0 : 1]);
            } else
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
        auto cost = c::testing::deferredDiscoveryReservation(64, 4, 6, 20);
        require(bool(cost));
        require(c::testing::deferredDiscoveryReservation(64, 4, 6, 20, *cost) == cost);
        require(!c::testing::deferredDiscoveryReservation(64, 4, 6, 20, *cost - 1));
        require(!c::testing::deferredDiscoveryReservation(64, 4, 6, 20, 0));
        for (unsigned field = 0; field < 5; ++field) {
            std::array<uint64_t, 5> args{64, 4, 6, 20, c::DeferredDiscoveryLimit};
            args[field] = UINT64_MAX;
            require(!c::testing::deferredDiscoveryReservation(args[0], args[1], args[2], args[3], args[4]));
        }
        require(!c::testing::deferredDiscoveryReservation(1u << 20, 2, 6, 20));
        require(!c::testing::deferredDiscoveryReservation(64, 4, 65, 20));
        require(!c::testing::deferredDiscoveryReservation(64, 4, 6, (1u << 20) + 1));
    }
    // Closed per-visit rings: no header seed, no unconsumed last publication,
    // and one shared protocol for identical multi-cell witnesses. The finite
    // graph starts keys idle and preserves them across skipped/repeated runs.
    for (unsigned cells : {1u, 2u}) {
        c::Program p;
        p.cells = cells;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto write = op(p, a, 0, true), read = op(p, b, 0, false);
        if (cells == 2) {
            p.nodes[write].effects[1].writers = 1u << a;
            p.nodes[read].effects[1].readers = 1u << b;
        }
        auto body = sequence(p, {write, read});
        auto loop = add(p, c::Node::For, {body});
        sequence(p, {loop});
        auto old = c::testing::constructDemandsWithoutRings(p);
        auto plan = c::constructDemands(p);
        require(old.success && plan.success && plan.ringCandidates == 1);
        require(plan.rejectedRings == 1 && plan.before == old.before); // Equal cost is not an improvement.
        require(c::verifyDemands(p, plan.before).success);
        require(plan.before[loop].empty());
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned invocation) { return (invocation * 3) % 5; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 8; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
        auto missing = plan.before;
        missing[write].erase(missing[write].begin());
        require(!c::verifyDemands(p, missing).success);
        auto reversed = plan.before;
        std::reverse(reversed[write].begin(), reversed[write].end());
        require(!c::verifyDemands(p, reversed).success);
        auto outside = p;
        // Changing an immutable cell witness invalidates the selected word;
        // a second observer cannot borrow the first observer's cycle credit.
        outside.nodes[read].lane = unsigned(Pipe::MTE3);
        outside.nodes[read].effects[0].readers = 1u << unsigned(Pipe::MTE3);
        require(!c::verifyDemands(outside, plan.before).success);
    }
    for (unsigned copies : {1u, 2u}) {
        c::Program p;
        p.cells = 2 * copies;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto load = op(p, a, 0, true), compute = op(p, b, 0, false), store = op(p, d, 1, false);
        p.nodes[compute].effects[1].writers = 1u << b;
        if (copies == 2) {
            p.nodes[load].effects[2] = p.nodes[load].effects[0];
            p.nodes[compute].effects[2] = p.nodes[compute].effects[0];
            p.nodes[compute].effects[3] = p.nodes[compute].effects[1];
            p.nodes[store].effects[3] = p.nodes[store].effects[1];
        }
        auto loop = add(p, c::Node::For, {sequence(p, {load, compute, store})});
        sequence(p, {loop});
        auto plan = c::constructDemands(p);
        require(plan.success && plan.cutCycles == 2 && plan.ringCandidateCommandsRemoved > 0);
        require(c::verifyDemands(p, plan.before).success && plan.before[loop].empty());
        auto rejected = c::testing::constructDemandsRejectingRings(p);
        require(
            rejected.success && rejected.rejectedRings == 1 &&
            rejected.before == c::testing::constructDemandsWithoutRings(p).before);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return (visit * 3) % 5; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 8; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
    }
    {
        // Deferred wrap: start from two already selected two-group closed
        // rings spanning three lanes. Move each wrap SET to the exact
        // post-last-group cut, consume it on later visits, and drain the final
        // generation only on nonempty exit.
        c::Program p;
        p.cells = 5;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto guard = add(p, c::Node::Sequence);
        auto load = op(p, a, 0, true), compute = op(p, b, 0, false);
        p.nodes[compute].effects[1].writers = 1u << b;
        auto laterB = add(p, c::Node::Operation);
        p.nodes[laterB].lane = b;
        // Unrelated read-only work on untouched cells has real physical
        // effects, but no invented next-iteration write/reuse obligation.
        p.nodes[laterB].effects[2].readers = 1u << b;
        auto store = op(p, d, 1, false), laterD = add(p, c::Node::Operation);
        // The last MTE3 group also writes an independent destination. Its next
        // generation is ordered only through the deferred MTE3->V receipt and
        // the ordinary V->MTE3 return, matching a TStore destination hazard.
        p.nodes[store].effects[3].writers = 1u << d;
        p.nodes[laterD].lane = d;
        p.nodes[laterD].effects[4].readers = 1u << d;
        auto body = sequence(p, {load, compute, laterB, store, laterD});
        auto loop = add(p, c::Node::For, {body});
        auto after = add(p, c::Node::Sequence);
        sequence(p, {guard, loop, after});
        p.nodes[loop].entryGuardStart = guard;

        auto plan = c::constructDemands(p);
        require(
            plan.success && plan.deferredRingCandidates == 2 && plan.deferredRings == 2 &&
            plan.rejectedDeferredRings == 0 && plan.deferredProtocolSteps > 0 &&
            plan.deferredProtocolSteps <= (1u << 20));
        require(c::verifyDemands(p, plan.before).success);
        require(std::count_if(plan.before[load].begin(), plan.before[load].end(), [](const auto& mechanism) {
                    return mechanism.participation == c::Mechanism::Previous;
                }) == 1);
        require(std::count_if(plan.before[compute].begin(), plan.before[compute].end(), [](const auto& mechanism) {
                    return mechanism.participation == c::Mechanism::Previous;
                }) == 1);
        require(std::count_if(plan.before[after].begin(), plan.before[after].end(), [](const auto& mechanism) {
                    return mechanism.participation == c::Mechanism::LoopExit;
                }) == 2);
        // These are the cuts immediately after the last relevant reader. The
        // later same-lane payload is deliberately excluded from the snapshot.
        require(std::any_of(plan.before[laterB].begin(), plan.before[laterB].end(), [&](const auto& mechanism) {
            return mechanism.kind == c::Mechanism::Publish && mechanism.first == b && mechanism.second == a;
        }));
        require(std::any_of(plan.before[laterD].begin(), plan.before[laterD].end(), [&](const auto& mechanism) {
            return mechanism.kind == c::Mechanism::Publish && mechanism.first == d && mechanism.second == b;
        }));

        auto rollback = c::testing::constructDemandsRejectingDeferredRings(p);
        require(
            rollback.success && rollback.deferredRingCandidates == 2 && rollback.deferredRings == 0 &&
            rollback.rejectedDeferredRings == 2 && c::verifyDemands(p, rollback.before).success);
        auto noDiscovery = c::testing::constructDemandsWithoutDeferredDiscovery(p);
        require(
            noDiscovery.success && noDiscovery.before == rollback.before && noDiscovery.deferredDiscoveryWork == 0 &&
            noDiscovery.deferredDiscoveryRefusals == 1 && plan.deferredDiscoveryWork > 0 &&
            plan.deferredDiscoveryWork <= c::DeferredDiscoveryLimit);
        for (unsigned trips : {0u, 1u, 2u}) {
            ExecutionPolicy deferredPolicy, closedPolicy;
            deferredPolicy.trips = [=](unsigned, unsigned) { return trips; };
            closedPolicy.trips = deferredPolicy.trips;
            deferredPolicy.choice = closedPolicy.choice = [](unsigned, unsigned) { return 0u; };
            Oracle deferredOracle, closedOracle;
            execute(p, plan, p.nodes.size() - 1, deferredPolicy, deferredOracle);
            execute(p, rollback, p.nodes.size() - 1, closedPolicy, closedOracle);
            deferredOracle.check();
            closedOracle.check();
            // Zero visits execute no ring command. For T>0, T SETs and
            // (T-1)+1 WAITs preserve the closed ring's dynamic command count.
            require(deferredOracle.commands == closedOracle.commands);
        }
        ExecutionPolicy repeated;
        const std::array<unsigned, 6> visits{0, 1, 2, 0, 2, 1};
        repeated.trips = [&](unsigned, unsigned invocation) { return visits[invocation % visits.size()]; };
        repeated.choice = [](unsigned, unsigned) { return 0u; };
        Oracle repeatedOracle;
        for (unsigned invocation = 0; invocation < visits.size(); ++invocation) {
            execute(p, plan, p.nodes.size() - 1, repeated, repeatedOracle);
            repeatedOracle.check();
        }
        ExecutionPolicy overlap;
        overlap.trips = [](unsigned, unsigned) { return 2u; };
        overlap.choice = [](unsigned, unsigned) { return 0u; };
        Oracle overlapOracle;
        execute(p, plan, p.nodes.size() - 1, overlap, overlapOracle);
        overlapOracle.check();
        // The moved V->MTE2 SET contains the first iteration's compute prefix,
        // not later unrelated V work. Thus the next load may issue without
        // waiting for laterB to complete; this checks graph causality rather
        // than merely checking the lexical SET site.
        require(overlapOracle.accesses.size() == 10);
        require(!overlapOracle.reaches(overlapOracle.accesses[2].done, overlapOracle.accesses[5].issue));

        auto mutate = [&](auto change) {
            auto broken = plan.before;
            change(broken);
            require(!c::verifyDemands(p, broken).success);
        };
        mutate([&](auto& broken) {
            auto& commands = broken[load];
            commands.erase(std::find_if(commands.begin(), commands.end(), [](const auto& mechanism) {
                return mechanism.participation == c::Mechanism::Previous;
            }));
        });
        mutate([&](auto& broken) {
            auto& commands = broken[after];
            commands.erase(std::find_if(commands.begin(), commands.end(), [](const auto& mechanism) {
                return mechanism.participation == c::Mechanism::LoopExit;
            }));
        });
        mutate([&](auto& broken) {
            auto previous = std::find_if(broken[load].begin(), broken[load].end(), [](const auto& mechanism) {
                return mechanism.participation == c::Mechanism::Previous;
            });
            ++previous->forwardKey;
        });
        mutate([&](auto& broken) {
            auto previous = std::find_if(broken[load].begin(), broken[load].end(), [](const auto& mechanism) {
                return mechanism.participation == c::Mechanism::Previous;
            });
            previous->participation = c::Mechanism::First;
        });
        mutate([&](auto& broken) {
            auto previous = std::find_if(broken[load].begin(), broken[load].end(), [](const auto& mechanism) {
                return mechanism.participation == c::Mechanism::Previous;
            });
            previous->participation = c::Mechanism::Every;
            previous->loop = ~0u;
        });
        mutate([&](auto& broken) {
            auto& commands = broken[laterB];
            commands.erase(std::find_if(commands.begin(), commands.end(), [&](const auto& mechanism) {
                return mechanism.kind == c::Mechanism::Publish && mechanism.first == b && mechanism.second == a;
            }));
        });
        mutate([&](auto& broken) {
            auto publication = std::find_if(broken[laterB].begin(), broken[laterB].end(), [&](const auto& mechanism) {
                return mechanism.kind == c::Mechanism::Publish && mechanism.first == b && mechanism.second == a;
            });
            broken[store].push_back(*publication);
            broken[laterB].erase(publication);
        });
        mutate([&](auto& broken) {
            auto publication = std::find_if(broken[laterB].begin(), broken[laterB].end(), [&](const auto& mechanism) {
                return mechanism.kind == c::Mechanism::Publish && mechanism.first == b && mechanism.second == a;
            });
            broken[compute].push_back(*publication);
            broken[laterB].erase(publication);
        });
        auto oversized = plan.before;
        auto splitExit = plan.before;
        auto exit = std::find_if(splitExit[after].begin(), splitExit[after].end(), [](const auto& m) {
            return m.participation == c::Mechanism::LoopExit;
        });
        splitExit[p.nodes.back().children.back()].push_back(*exit);
        splitExit[after].erase(exit);
        require(!c::verifyDemands(p, splitExit).success);
        auto interleaved = plan.before;
        interleaved[after].insert(interleaved[after].begin() + 1, {c::Mechanism::Barrier, d, d});
        require(!c::verifyDemands(p, interleaved).success);
        auto relayed = plan.before;
        relayed[after].push_back({c::Mechanism::Rendezvous, std::min(a, b), std::max(a, b), 0, 0});
        auto relayCheck = c::verifyDemands(p, relayed);
        require(
            !relayCheck.success && relayCheck.reason == "deferred loop exit may stall a later synchronization command");
        for (unsigned i = 0; i < 30000; ++i)
            oversized[guard].push_back({c::Mechanism::Barrier, a, a});
        auto bounded = c::verifyDemands(p, oversized);
        require(
            !bounded.success && bounded.deferredRejectionStage == "protocol" &&
            bounded.deferredRejectionReason == "deferred demand ring protocol exceeds optional work bound");
    }
    for (unsigned period : {1u, 2u, 3u, 32u}) {
        // A complete word selected on the last residue. First means first
        // ACTIVE visit, not the first loop iteration. Empty/skipped iterations
        // do not publish; repeated whole owners share actual hardware keys.
        c::Program p;
        p.cells = 4;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto guard = add(p, c::Node::Sequence);
        auto load = op(p, a, 0, true), use = op(p, b, 0, false);
        p.nodes[use].effects[1].writers = 1u << b;
        auto later = op(p, b, 2, false), store = op(p, d, 1, false);
        p.nodes[store].effects[3].writers = 1u << d;
        auto word = sequence(p, {load, use, later, store});
        auto empty = add(p, c::Node::Sequence);
        auto choice = add(p, c::Node::Choice, {word, empty});
        auto body = sequence(p, {choice});
        auto loop = add(p, c::Node::For, {body});
        auto after = add(p, c::Node::Sequence);
        sequence(p, {guard, loop, after});
        p.nodes[loop].entryGuardStart = guard;
        uint32_t all = period == 32 ? UINT32_MAX : (uint32_t(1) << period) - 1;
        uint32_t active = uint32_t(1) << (period - 1);
        std::function<void(unsigned, uint32_t)> annotate = [&](unsigned id, uint32_t mask) {
            auto& node = p.nodes[id];
            if (node.kind != c::Node::Operation) {
                node.periodicOwner = loop;
                node.periodicPeriod = period;
                node.periodicLower = 5;
                node.periodicResidues = node.kind == c::Node::Choice ? active : mask;
            }
            if (node.kind == c::Node::Choice) {
                annotate(node.children[0], mask & active);
                annotate(node.children[1], mask & ~active);
            } else
                for (unsigned child : node.children)
                    annotate(child, mask);
        };
        annotate(body, all);
        auto plan = c::constructDemands(p);
        require(plan.success && c::verifyDemands(p, plan.before).success);
        if (period < 32) {
            require(plan.deferredRings == 2 && plan.rejectedDeferredRings == 0);
            require(std::any_of(plan.before[load].begin(), plan.before[load].end(), [&](const auto& m) {
                return m.participation == c::Mechanism::Previous && m.word == word;
            }));
            auto wrongFirst = plan.before;
            for (auto& m : wrongFirst[load])
                if (m.participation == c::Mechanism::Previous)
                    m.word = ~0u;
            require(!c::verifyDemands(p, wrongFirst).success);
            auto wrongExit = plan.before;
            for (auto& m : wrongExit[after])
                if (m.participation == c::Mechanism::LoopExit)
                    m.word = ~0u;
            require(!c::verifyDemands(p, wrongExit).success);
            auto wrongDomain = p;
            wrongDomain.nodes[word].periodicResidues = 0;
            require(!c::verifyDemands(wrongDomain, plan.before).success);
        } else {
            require(
                plan.deferredRings == 0 && plan.rejectedDeferredRings == 2 &&
                plan.deferredRejectionStage == "protocol");
        }
        ExecutionPolicy policy;
        std::vector<unsigned> trips{0, period - 1, period, period + 1, 2 * period + 3, 0, 1};
        policy.trips = [&](unsigned, unsigned visit) { return trips[visit % trips.size()]; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < trips.size(); ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
        auto overflowing = p;
        overflowing.nodes[word].periodicLower = INT64_MAX;
        require(c::constructDemands(overflowing).success); // precision failure, not semantic refusal
        if (period < 32) {
            for (bool writer : {false, true}) {
                auto outside = p;
                auto& node = outside.nodes[empty];
                node.kind = c::Node::Operation;
                node.periodicOwner = ~0u;
                node.lane = b;
                (writer ? node.effects[0].writers : node.effects[0].readers) = 1u << b;
                require(!c::verifyDemands(outside, plan.before).success);
            }
        }
    }
    {
        // A later sibling's receiving-lane work and cleanup are not part of
        // this owner's retirement packet. Only the final owner may defer here.
        c::Program p;
        p.cells = 4;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto guard = add(p, c::Node::Sequence);
        auto child = [&](unsigned cell) {
            auto load = op(p, a, cell, true), use = op(p, b, cell, false);
            p.nodes[use].effects[cell + 1].writers = 1u << b;
            auto store = op(p, d, cell + 1, false);
            return add(p, c::Node::For, {sequence(p, {load, use, store})});
        };
        auto first = child(0), between = add(p, c::Node::Sequence), second = child(2);
        auto after = add(p, c::Node::Sequence);
        sequence(p, {guard, first, between, second, after});
        p.nodes[first].entryGuardStart = guard;
        p.nodes[second].entryGuardStart = between;
        auto plan = c::constructDemands(p);
        require(plan.success && plan.deferredRings == 2 && c::verifyDemands(p, plan.before).success);
        for (const auto& commands : plan.before)
            for (const auto& m : commands)
                if (m.participation == c::Mechanism::LoopExit)
                    require(m.loop == second);
        auto wrongOwner = plan.before;
        for (auto& m : wrongOwner[after])
            if (m.participation == c::Mechanism::LoopExit)
                m.loop = first;
        require(!c::verifyDemands(p, wrongOwner).success);
        ExecutionPolicy policy;
        policy.trips = [=](unsigned id, unsigned visit) { return (visit + (id == first ? 1 : 0)) % 4; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 6; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
    }
    {
        // Ten otherwise eligible, disjoint families fit the directed key
        // pools. The optional selector takes a deterministic batch of eight;
        // unselected words keep their complete closed protocols.
        c::Program p;
        p.cells = 10;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto guard = add(p, c::Node::Sequence);
        std::vector<unsigned> words;
        for (unsigned slot = 0; slot < 5; ++slot) {
            unsigned cell = 2 * slot;
            auto load = op(p, a, cell, true), use = op(p, b, cell, false);
            p.nodes[use].effects[cell + 1].writers = 1u << b;
            auto store = op(p, d, cell + 1, false);
            auto word = sequence(p, {load, use, store});
            auto empty = add(p, c::Node::Sequence);
            words.push_back(add(p, c::Node::Choice, {word, empty}));
        }
        auto body = sequence(p, words), loop = add(p, c::Node::For, {body});
        auto after = add(p, c::Node::Sequence);
        sequence(p, {guard, loop, after});
        p.nodes[loop].entryGuardStart = guard;
        std::function<void(unsigned, uint32_t)> annotate = [&](unsigned id, uint32_t active) {
            auto& node = p.nodes[id];
            if (node.kind != c::Node::Operation) {
                node.periodicOwner = loop;
                node.periodicPeriod = 1;
                node.periodicResidues = node.kind == c::Node::Choice ? 1 : active;
            }
            if (node.kind == c::Node::Choice) {
                annotate(node.children[0], active);
                annotate(node.children[1], 0);
            } else
                for (unsigned child : node.children)
                    annotate(child, active);
        };
        annotate(body, 1);
        auto plan = c::constructDemands(p);
        require(
            plan.success && plan.deferredRings == 8 && plan.deferredSkippedFamilies == 2 &&
            c::verifyDemands(p, plan.before).success);
        require(plan.before == c::constructDemands(p).before);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return visit % 3; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 6; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
    }
    {
        // Real same-lane writes after each proposed tail SET are not covered by
        // its source-prefix receipt. Their next-generation WAW obligations make
        // deferred wrap inapplicable; retain the independently verified closed
        // rings rather than erasing that suffix history.
        c::Program p;
        p.cells = 4;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto guard = add(p, c::Node::Sequence);
        auto load = op(p, a, 0, true), compute = op(p, b, 0, false);
        p.nodes[compute].effects[1].writers = 1u << b;
        auto laterB = op(p, b, 2, true), store = op(p, d, 1, false), laterD = op(p, d, 3, true);
        auto loop = add(p, c::Node::For, {sequence(p, {load, compute, laterB, store, laterD})});
        auto after = add(p, c::Node::Sequence);
        sequence(p, {guard, loop, after});
        p.nodes[loop].entryGuardStart = guard;
        auto plan = c::constructDemands(p);
        require(
            plan.success && plan.deferredRingCandidates == 2 && plan.deferredRings == 0 &&
            plan.rejectedDeferredRings == 2 && plan.deferredRejectionStage == "physical" &&
            !plan.deferredRejectionReason.empty() && c::verifyDemands(p, plan.before).success);
        ExecutionPolicy policy;
        const std::array<unsigned, 6> visits{0, 1, 2, 0, 2, 1};
        policy.trips = [&](unsigned, unsigned invocation) { return visits[invocation % visits.size()]; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < visits.size(); ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
    }
    {
        // The typed wrap receipt clears only its covered source prefix. Current
        // source work before Previous, source work after the tail SET, and
        // another lane's work all remain pending. A global cell also confirms
        // that generation normalization never supplies GM visibility.
        c::Program p;
        p.cells = 3;
        p.globalMemory = {false, false, true};
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto guard = add(p, c::Node::Sequence);
        auto load = op(p, a, 0, true);
        auto earlyD = add(p, c::Node::Operation);
        p.nodes[earlyD].lane = d;
        auto compute = op(p, b, 0, false);
        p.nodes[compute].effects[1].writers = 1u << b;
        auto store = op(p, d, 1, false);
        auto lateD = add(p, c::Node::Operation), lateA = add(p, c::Node::Operation);
        p.nodes[lateD].lane = d;
        p.nodes[lateA].lane = a;
        auto loop = add(p, c::Node::For, {sequence(p, {load, earlyD, compute, store, lateD, lateA})});
        auto after = add(p, c::Node::Sequence);
        sequence(p, {guard, loop, after});
        p.nodes[loop].entryGuardStart = guard;
        auto plan = c::constructDemands(p);
        require(plan.success && plan.deferredRings == 2 && c::verifyDemands(p, plan.before).success);
        auto mustRetain = [&](unsigned producer) {
            auto changed = p;
            changed.nodes[producer].effects[2].writers = 1u << changed.nodes[producer].lane;
            changed.nodes[compute].effects[2].readers = 1u << b;
            auto checked = c::verifyDemands(changed, plan.before);
            require(
                !checked.success && checked.deferredRejectionStage == "physical" &&
                !checked.deferredRejectionReason.empty());
        };
        mustRetain(earlyD);
        mustRetain(lateD);
        mustRetain(lateA);
    }
    {
        // A branch-contained closed ring remains the existing baseline even
        // when its owner has guard metadata. Deferred wrap requires the ring
        // region to be exactly the For body, so a skipped branch cannot execute
        // a partial deferred word.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto guard = add(p, c::Node::Sequence);
        auto load = op(p, a, 0, true), compute = op(p, b, 0, false);
        p.nodes[compute].effects[1].writers = 1u << b;
        auto store = op(p, d, 1, false);
        auto conditionalWord = sequence(p, {load, compute, store});
        auto empty = add(p, c::Node::Sequence);
        auto choice = add(p, c::Node::Choice, {conditionalWord, empty});
        auto body = sequence(p, {choice});
        auto loop = add(p, c::Node::For, {body});
        auto after = add(p, c::Node::Sequence);
        sequence(p, {guard, loop, after});
        p.nodes[loop].entryGuardStart = guard;
        auto plan = c::constructDemands(p);
        require(
            plan.success && plan.deferredRingCandidates == 0 && plan.deferredRings == 0 &&
            c::verifyDemands(p, plan.before).success);
        for (const auto& commands : plan.before)
            for (const auto& mechanism : commands)
                require(
                    mechanism.participation != c::Mechanism::Previous &&
                    mechanism.participation != c::Mechanism::LoopExit);
    }
    for (unsigned scenario = 0; scenario < 4; ++scenario) {
        // LoopExit may retire at function exit past unrelated-lane work. It
        // must not stall later first-lane work, cross a Choice continuation,
        // or escape into an enclosing loop backedge.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto guard = add(p, c::Node::Sequence);
        auto load = op(p, a, 0, true), compute = op(p, b, 0, false), store = op(p, d, 1, false);
        p.nodes[compute].effects[1].writers = 1u << b;
        auto loop = add(p, c::Node::For, {sequence(p, {load, compute, store})});
        auto after = add(p, c::Node::Sequence);
        p.nodes[loop].entryGuardStart = guard;
        if (scenario == 2) {
            auto branch = sequence(p, {guard, loop, after});
            auto empty = add(p, c::Node::Sequence);
            auto choice = add(p, c::Node::Choice, {branch, empty});
            auto laterA = add(p, c::Node::Operation), laterB = add(p, c::Node::Operation);
            p.nodes[laterA].lane = a;
            p.nodes[laterB].lane = b;
            sequence(p, {choice, laterA, laterB});
        } else if (scenario == 3) {
            auto innerScope = sequence(p, {guard, loop, after});
            auto outer = add(p, c::Node::For, {innerScope});
            sequence(p, {outer});
        } else {
            std::vector<unsigned> root{guard, loop, after};
            auto later = add(p, c::Node::Operation);
            p.nodes[later].lane = scenario == 0 ? d : a;
            root.push_back(later);
            if (scenario == 1) {
                auto laterB = add(p, c::Node::Operation);
                p.nodes[laterB].lane = b;
                root.push_back(laterB);
            }
            sequence(p, std::move(root));
        }
        auto plan = c::constructDemands(p);
        require(plan.success && c::verifyDemands(p, plan.before).success);
        if (scenario == 0)
            require(plan.deferredRingCandidates == 2 && plan.deferredRings == 2);
        else {
            require(plan.deferredRingCandidates == 0 && plan.deferredRings == 0);
            for (const auto& commands : plan.before)
                for (const auto& mechanism : commands)
                    require(
                        mechanism.participation != c::Mechanism::Previous &&
                        mechanism.participation != c::Mechanism::LoopExit);
        }
    }
    {
        // Endpoint population is a structural multiset, not node-ID order.
        // Create siblings in reverse order, then execute load/compute/store.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto guard = add(p, c::Node::Sequence);
        auto store = op(p, d, 1, false), compute = op(p, b, 0, false), load = op(p, a, 0, true);
        p.nodes[compute].effects[1].writers = 1u << b;
        auto loop = add(p, c::Node::For, {sequence(p, {load, compute, store})});
        auto after = add(p, c::Node::Sequence);
        sequence(p, {guard, loop, after});
        p.nodes[loop].entryGuardStart = guard;
        auto plan = c::constructDemands(p);
        require(
            plan.success && plan.deferredRingCandidates == 2 && plan.deferredRings == 2 &&
            c::verifyDemands(p, plan.before).success);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return visit % 3; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 6; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
    }
    {
        // Distinct source cuts, one actual consumer: sharing keeps the common
        // wait and publishes the latest necessary prefix, not one pair/cell.
        c::Program p;
        p.cells = 4;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto first = op(p, a, 0, true), second = op(p, a, 1, true);
        auto unrelated = op(p, b, 3, false);
        auto use = op(p, b, 0, false), output = op(p, d, 2, false);
        p.nodes[use].effects[1].readers = 1u << b;
        p.nodes[use].effects[2].writers = 1u << b;
        auto loop = add(p, c::Node::For, {sequence(p, {first, unrelated, second, use, output})});
        sequence(p, {loop});
        auto plan = c::constructDemands(p);
        require(plan.success && plan.cutCycles == 2 && plan.ringCandidateCommandsRemoved > 0);
        require(c::verifyDemands(p, plan.before).success);
        unsigned readiness = 0;
        for (const auto& m : plan.before[use])
            readiness += m.kind == c::Mechanism::Acquire && m.first == a && m.second == b;
        require(readiness == 1);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return (visit * 3) % 5; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 8; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
            for (unsigned i = 0; i < oracle.accesses.size(); i += 5)
                require(!oracle.reaches(oracle.accesses[i + 1].done, oracle.accesses[i + 2].issue));
        }
        auto corrupted = plan.before;
        corrupted[use].erase(
            std::remove_if(
                corrupted[use].begin(), corrupted[use].end(),
                [&](const auto& m) { return m.kind == c::Mechanism::Acquire && m.first == a && m.second == b; }),
            corrupted[use].end());
        require(!c::verifyDemands(p, corrupted).success);
    }
    {
        // A collapsed single-lane nested loop is not one cell generation.
        // Refuse only the optional ring, not general structural composition.
        c::Program p;
        p.cells = 1;
        auto write = op(p, unsigned(Pipe::MTE2), 0, true);
        auto inner = add(p, c::Node::For, {sequence(p, {write})});
        auto read = op(p, unsigned(Pipe::V), 0, false);
        auto outer = add(p, c::Node::For, {sequence(p, {inner, read})});
        sequence(p, {outer});
        auto plan = c::constructDemands(p);
        require(plan.success && plan.ringCandidates == 0 && c::verifyDemands(p, plan.before).success);
        ExecutionPolicy policy;
        policy.trips = [=](unsigned id, unsigned visit) { return id == outer ? 2u : visit % 4; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 4; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
    }
    {
        // Interleaved cell words cannot borrow a single later readiness wait.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto x = op(p, a, 0, true), rx = op(p, b, 0, false);
        auto y = op(p, a, 1, true), ry = op(p, b, 1, false);
        auto loop = add(p, c::Node::For, {sequence(p, {x, rx, y, ry})});
        sequence(p, {loop});
        auto plan = c::constructDemands(p);
        require(plan.success && c::verifyDemands(p, plan.before).success);
        std::vector<std::vector<c::Mechanism>> merged(p.nodes.size());
        merged[x] = {{c::Mechanism::Publish, b, a, 1, 0}, {c::Mechanism::Acquire, b, a, 1, 0}};
        merged[ry] = {{c::Mechanism::Publish, a, b, 1, 0}, {c::Mechanism::Acquire, a, b, 1, 0}};
        require(!c::verifyDemands(p, merged).success);
    }
    {
        c::Program p;
        p.cells = c::MaxCells;
        auto write = op(p, unsigned(Pipe::MTE2), 0, true), read = op(p, unsigned(Pipe::V), 0, false);
        std::vector<unsigned> body{write, read};
        for (unsigned i = 0; i < 4096; ++i)
            body.push_back(add(p, c::Node::Sequence));
        auto loop = add(p, c::Node::For, {sequence(p, std::move(body))});
        sequence(p, {loop});
        auto plan = c::constructDemands(p);
        require(plan.success && plan.ringCandidates == 0 && c::verifyDemands(p, plan.before).success);
    }
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
