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
static void execute(
    const c::Program& p, const c::Result& r, unsigned id, unsigned trips, unsigned pattern, unsigned& branch,
    Oracle& oracle)
{
    for (auto& m : r.before[id])
        oracle.mechanism(m);
    const auto& n = p.nodes[id];
    auto run = [&](unsigned child) { execute(p, r, child, trips, pattern, branch, oracle); };
    switch (n.kind) {
        case c::Node::Operation:
            oracle.payload(n);
            break;
        case c::Node::Sequence:
            for (auto x : n.children)
                run(x);
            break;
        case c::Node::Choice:
            run(n.children[(pattern >> (branch++ % 4)) & 1]);
            break;
        case c::Node::For:
            for (unsigned i = 0; i < trips; ++i)
                run(n.children[0]);
            break;
        case c::Node::While:
            for (unsigned i = 0; i <= trips; ++i) {
                run(n.children[0]);
                if (i < trips)
                    run(n.children[1]);
            }
            break;
    }
}
int main()
{
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
    std::cout << checks << " compositional native-helper/independent finite graph assertions passed\n";
}
