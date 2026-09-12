// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/StructuredSyncComposition.h"
#include <algorithm>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <tuple>

using namespace mlir::pto::structured_sync;
namespace c = mlir::pto::structured_sync::composition;
namespace {
void merge(c::Effects& a, const c::Effects& b)
{
    for (unsigned i = 0; i < a.size(); ++i) {
        a[i].readers |= b[i].readers;
        a[i].writers |= b[i].writers;
    }
}
Lane lane(const c::Program& p, unsigned id) { return {p.core, static_cast<Pipe>(id)}; }
std::optional<unsigned> key(const c::Program& p, unsigned a, unsigned b)
{
    if (!p.target.event(lane(p, a), lane(p, b)))
        return {};
    for (auto k : p.target.compilerKeys)
        if (p.target.available(lane(p, a), lane(p, b), k))
            return k;
    return {};
}
bool validMechanism(const c::Program& p, const c::Mechanism& m)
{
    if (m.first >= c::LaneCount || m.second >= c::LaneCount)
        return false;
    if (m.kind == c::Mechanism::Barrier)
        return p.target.barrier(lane(p, m.first));
    if (m.kind != c::Mechanism::Rendezvous || m.first >= m.second)
        return false;
    // Fixed orientation AND keys throughout the lifetime are part of the
    // reusable protocol. Reversing the next exchange could rearm its reply
    // before the preceding reply was consumed.
    return key(p, m.first, m.second) == m.forwardKey && key(p, m.second, m.first) == m.reverseKey;
}
void apply(c::State& state, const c::Mechanism& m)
{
    if (m.kind == c::Mechanism::Barrier)
        state.barrier(m.first);
    else
        state.rendezvous(m.first, m.second);
}
bool summarize(const c::Program& p, std::vector<c::Effects>& summaries, std::string& reason)
{
    if (!p.cells || p.cells > c::MaxCells || p.nodes.empty()) {
        reason = "invalid composition storage population";
        return false;
    }
    if (!p.globalMemory.empty() && p.globalMemory.size() != p.cells) {
        reason = "invalid visibility cell population";
        return false;
    }
    std::vector<unsigned> parents(p.nodes.size());
    for (unsigned id = 0; id < p.nodes.size(); ++id) {
        const auto& n = p.nodes[id];
        if (n.effects.size() != p.cells || n.lane >= c::LaneCount ||
            (n.kind == c::Node::Operation && (!n.children.empty() || !p.target.barrier(lane(p, n.lane)))) ||
            (n.kind == c::Node::For && n.children.size() != 1) ||
            ((n.kind == c::Node::Choice || n.kind == c::Node::While) && n.children.size() != 2) ||
            n.kind > c::Node::While) {
            reason = "invalid composition node";
            return false;
        }
        for (const auto& e : n.effects)
            if ((e.readers | e.writers) & ~(1u << n.lane)) {
                reason = "effect lane differs from operation lane";
                return false;
            }
        summaries.push_back(n.effects);
        for (unsigned child : n.children) {
            if (child >= id || ++parents[child] != 1) {
                reason = "composition is not a postorder tree";
                return false;
            }
            merge(summaries.back(), summaries[child]);
        }
    }
    for (unsigned i = 0; i + 1 < parents.size(); ++i)
        if (parents[i] != 1) {
            reason = "orphan composition node";
            return false;
        }
    return true;
}

bool needsVisibility(const c::Program& p, const c::Node& n, const c::State& state)
{
    for (unsigned i = 0; i < p.globalMemory.size(); ++i)
        if (p.globalMemory[i] && n.effects[i].readers && state.written[i])
            return true;
    return false;
}

// Bidirectional, target-qualified shortest path; fixed lane count. Each
// rendezvous transports completion acquired at its preceding path vertex.
bool acquire(const c::Program& p, unsigned source, unsigned target, c::State& state, std::vector<c::Mechanism>& out)
{
    if (source == target) {
        if (!p.target.barrier(lane(p, source)))
            return false;
        c::Mechanism m{c::Mechanism::Barrier, source, source};
        apply(state, m);
        out.push_back(m);
        return true;
    }
    std::array<int, c::LaneCount> previous;
    previous.fill(-1);
    previous[source] = source;
    std::queue<unsigned> todo;
    todo.push(source);
    while (!todo.empty() && previous[target] < 0) {
        unsigned a = todo.front();
        todo.pop();
        for (unsigned b = 0; b < c::LaneCount; ++b)
            if (previous[b] < 0 && key(p, a, b) && key(p, b, a)) {
                previous[b] = a;
                todo.push(b);
            }
    }
    if (previous[target] < 0)
        return false;
    std::vector<unsigned> path{target};
    while (path.back() != source)
        path.push_back(previous[path.back()]);
    std::reverse(path.begin(), path.end());
    for (unsigned i = 1; i < path.size(); ++i) {
        auto a = std::min(path[i - 1], path[i]);
        auto b = std::max(path[i - 1], path[i]);
        c::Mechanism m{c::Mechanism::Rendezvous, a, b, *key(p, a, b), *key(p, b, a)};
        apply(state, m);
        out.push_back(m);
    }
    return true;
}
} // namespace

c::State::State(unsigned cells)
{
    for (auto& history : pending)
        history.resize(cells);
    written.resize(cells);
}
void c::State::join(const State& other)
{
    for (unsigned lane = 0; lane < LaneCount; ++lane)
        merge(pending[lane], other.pending[lane]);
    for (unsigned i = 0; i < written.size(); ++i)
        written[i] |= other.written[i];
}
void c::State::seed(const Effects& effects)
{
    for (auto& history : pending)
        merge(history, effects);
    for (unsigned i = 0; i < effects.size(); ++i)
        written[i] |= effects[i].writers;
}
void c::State::barrier(unsigned lane)
{
    for (auto& e : pending[lane]) {
        e.readers &= ~(1u << lane);
        e.writers &= ~(1u << lane);
    }
}
void c::State::rendezvous(unsigned a, unsigned b)
{
    for (unsigned i = 0; i < pending[a].size(); ++i) {
        History h;
        h.readers = pending[a][i].readers & pending[b][i].readers;
        h.writers = pending[a][i].writers & pending[b][i].writers;
        h.readers &= ~(1u << a);
        h.writers &= ~(1u << a);
        // B's WAIT acquires A's prefix and previous knowledge. Its following
        // reply SET does NOT block B's later issue until B's own prefix completes.
        pending[b][i] = h;
        // Only A's final reply WAIT also acquires B's prefix. Never model SET
        // FIRE as a same-lane barrier or as ordering subsequent source issue.
        h.readers &= ~(1u << b);
        h.writers &= ~(1u << b);
        pending[a][i] = h;
    }
}
uint8_t c::State::demands(unsigned observer, const Effects& effects) const
{
    uint8_t result = 0;
    for (unsigned i = 0; i < effects.size(); ++i) {
        if (effects[i].readers || effects[i].writers)
            result |= pending[observer][i].writers;
        if (effects[i].writers)
            result |= pending[observer][i].readers;
    }
    return result;
}
bool c::Mechanism::operator==(const Mechanism& m) const
{
    return kind == m.kind && first == m.first && second == m.second && forwardKey == m.forwardKey &&
           reverseKey == m.reverseKey;
}

c::Result c::construct(const Program& p)
{
    Result result;
    result.before.resize(p.nodes.size());
    std::vector<Effects> summaries;
    if (!summarize(p, summaries, result.reason))
        return result;
    std::function<bool(unsigned, State&)> visit = [&](unsigned id, State& state) {
        ++result.nodeVisits;
        result.cellVisits += p.cells * LaneCount;
        const auto& n = p.nodes[id];
        if (n.kind == Node::Operation) {
            if (needsVisibility(p, n, state)) {
                result.reason = "visibility has no qualified compositional realization";
                return false;
            }
            for (unsigned source = 0; source < LaneCount; ++source)
                if (state.demands(n.lane, n.effects) & (1u << source)) {
                    if (!acquire(p, source, n.lane, state, result.before[id])) {
                        result.reason = "target cannot realize conservative completion";
                        return false;
                    }
                    ++result.acquisitions;
                }
            state.seed(n.effects);
            return true;
        }
        if (n.kind == Node::Choice) {
            State alternative = state;
            if (!visit(n.children[0], state) || !visit(n.children[1], alternative))
                return false;
            state.join(alternative);
            return true;
        }
        if (n.kind == Node::For || n.kind == Node::While) {
            State entry = state;
            // This is a deliberately broad inductive entry: every effect the whole
            // body can leave outstanding is already present for every observer.
            // Transfers only clear bits or add body effects, so one visit suffices.
            state.seed(summaries[id]);
            if (!visit(n.children[0], state))
                return false;
            if (n.kind == Node::While) {
                State exit = state; // condition false: BEFORE executes, AFTER does not
                if (!visit(n.children[1], state))
                    return false;
                state = std::move(exit);
            } else
                state.join(entry); // zero-trip for
            return true;
        }
        for (auto child : n.children)
            if (!visit(child, state))
                return false;
        return true;
    };
    State state(p.cells);
    result.success = visit(p.nodes.size() - 1, state);
    if (!result.success)
        result.before.clear();
    return result;
}

c::Result c::verify(const Program& p, const std::vector<std::vector<Mechanism>>& actual)
{
    Result result;
    std::vector<Effects> summaries;
    if (actual.size() != p.nodes.size() || !summarize(p, summaries, result.reason))
        return result;
    // Independently walk actual mechanisms and test physical demands. Never
    // call construct(), acquire(), or compare against its chosen action list.
    std::function<bool(unsigned, State&)> check = [&](unsigned id, State& state) {
        ++result.nodeVisits;
        result.cellVisits += p.cells * LaneCount;
        const auto& n = p.nodes[id];
        for (const auto& m : actual[id]) {
            if (!validMechanism(p, m)) {
                result.reason = "invalid reusable event protocol";
                return false;
            }
            apply(state, m);
        }
        switch (n.kind) {
            case Node::Operation:
                if (needsVisibility(p, n, state)) {
                    result.reason = "same-address GM visibility is not completion";
                    return false;
                }
                if (state.demands(n.lane, n.effects)) {
                    result.reason = "uncovered physical completion requirement";
                    return false;
                }
                state.seed(n.effects);
                return true;
            case Node::Choice: {
                auto other = state;
                if (!check(n.children[0], state) || !check(n.children[1], other))
                    return false;
                state.join(other);
                return true;
            }
            case Node::For: {
                auto empty = state;
                state.seed(summaries[id]);
                if (!check(n.children[0], state))
                    return false;
                state.join(empty);
                return true;
            }
            case Node::While: {
                state.seed(summaries[id]);
                if (!check(n.children[0], state))
                    return false;
                auto exits = state;
                if (!check(n.children[1], state))
                    return false;
                state = std::move(exits);
                return true;
            }
            case Node::Sequence:
                for (auto child : n.children)
                    if (!check(child, state))
                        return false;
                return true;
        }
        return false;
    };
    State state(p.cells);
    result.success = check(p.nodes.size() - 1, state);
    return result;
}

namespace {
// A word is a transfer over original structural cuts, not a schedule expanded
// by trip count. Alternative cuts name mutually exclusive arms of one choice.
constexpr unsigned NoCut = ~0u;
constexpr unsigned MaxGroups = 32, MaxAlternatives = 8;
struct Group {
    unsigned lane;
    std::vector<unsigned> first, last;
};
struct Word {
    bool valid = true;
    std::vector<Group> groups;
};
struct Cycle {
    unsigned owner, region;
    std::vector<unsigned> cells;
    std::vector<Group> groups;
    std::map<std::pair<unsigned, unsigned>, unsigned> keys;
};
struct Cuts {
    std::vector<unsigned> parent, next, owner;
    std::vector<c::Effects> effects;
    std::vector<Cycle> cycles;
};
bool alternatives(std::vector<unsigned>& a, const std::vector<unsigned>& b)
{
    for (unsigned id : b)
        if (std::find(a.begin(), a.end(), id) == a.end())
            a.push_back(id);
    return a.size() <= MaxAlternatives;
}
bool append(Word& a, const Word& b)
{
    if (!a.valid || !b.valid)
        return a.valid = false;
    for (const auto& g : b.groups) {
        if (!a.groups.empty() && a.groups.back().lane == g.lane)
            a.groups.back().last = g.last;
        else
            a.groups.push_back(g);
    }
    return a.valid = a.groups.size() <= MaxGroups;
}
bool sameWord(const Word& a, const Word& b)
{
    if (!a.valid || !b.valid || a.groups.size() != b.groups.size())
        return false;
    for (unsigned i = 0; i < a.groups.size(); ++i)
        if (a.groups[i].lane != b.groups[i].lane)
            return false;
    return true;
}
// Derive words from immutable effects. Unknown/mixed child recurrences lose
// precision locally; the baseline still composes them. No Boolean products,
// periodic occurrence models, dependence-pair enumeration or closure matrices.
bool discoverCuts(const c::Program& p, Cuts& cuts, std::string& reason)
{
    if (!summarize(p, cuts.effects, reason))
        return false;
    unsigned size = p.nodes.size();
    cuts.parent.assign(size, NoCut);
    cuts.next.assign(size, NoCut);
    cuts.owner.assign(size, NoCut);
    for (unsigned id = 0; id < size; ++id) {
        const auto& n = p.nodes[id];
        for (unsigned child : n.children)
            cuts.parent[child] = id;
        if (n.kind == c::Node::Sequence)
            for (unsigned i = 1; i < n.children.size(); ++i)
                cuts.next[n.children[i - 1]] = n.children[i];
    }
    for (unsigned id = size; id-- > 0;)
        for (unsigned child : p.nodes[id].children)
            cuts.owner[child] =
                (p.nodes[id].kind == c::Node::For || p.nodes[id].kind == c::Node::While) ? id : cuts.owner[id];
    for (unsigned cell = 0; cell < p.cells; ++cell) {
        if (!p.globalMemory.empty() && p.globalMemory[cell])
            continue;
        std::vector<Word> words(size);
        std::vector<unsigned> accesses(size);
        for (unsigned id = 0; id < size; ++id) {
            const auto& n = p.nodes[id];
            auto& word = words[id];
            auto e = cuts.effects[id][cell];
            unsigned mask = e.readers | e.writers;
            if (n.kind == c::Node::Operation) {
                if (mask) {
                    word.groups.push_back({n.lane, {id}, {id}});
                    accesses[id] = 1;
                }
                continue;
            }
            for (unsigned child : n.children)
                accesses[id] += accesses[child];
            if (n.kind == c::Node::Sequence) {
                for (unsigned child : n.children)
                    if (!append(word, words[child]))
                        break;
            } else if (n.kind == c::Node::Choice) {
                const auto &a = words[n.children[0]], &b = words[n.children[1]];
                if (sameWord(a, b)) {
                    word = a;
                    for (unsigned i = 0; i < word.groups.size(); ++i) {
                        word.valid &= alternatives(word.groups[i].first, b.groups[i].first);
                        word.valid &= alternatives(word.groups[i].last, b.groups[i].last);
                    }
                } else if (mask && !(mask & (mask - 1))) {
                    // A possibly empty single-lane child has a common entry/exit cut.
                    unsigned pipe = 0;
                    while (!(mask & (1u << pipe)))
                        ++pipe;
                    word.groups.push_back({pipe, {id}, {id}});
                } else
                    word.valid = false;
            } else if (mask && !(mask & (mask - 1))) {
                unsigned pipe = 0;
                while (!(mask & (1u << pipe)))
                    ++pipe;
                word.groups.push_back({pipe, {id}, {id}});
            } else if (mask)
                word.valid = false;
        }
        // Choose the smallest complete access population within one repeated
        // owner. A slot-specific branch may be skipped arbitrarily: its protocol
        // advances on visits, not on a guessed value of the surrounding IV.
        for (unsigned id = 0; id < size; ++id) {
            const auto& word = words[id];
            if (p.nodes[id].kind != c::Node::Sequence || !word.valid || word.groups.size() < 2 ||
                word.groups.front().lane == word.groups.back().lane || !cuts.effects[id][cell].writers)
                continue;
            unsigned owner = cuts.owner[id];
            if (owner == NoCut || cuts.next[owner] == NoCut || accesses[id] != accesses[owner])
                continue;
            bool qualified = true;
            for (unsigned i = 0; i < word.groups.size(); ++i) {
                const auto &g = word.groups[i], &following = word.groups[(i + 1) % word.groups.size()];
                qualified &= bool(key(p, g.lane, following.lane));
                for (unsigned last : g.last)
                    qualified &= cuts.next[last] != NoCut;
            }
            if (qualified)
                cuts.cycles.push_back({owner, id, {cell}, word.groups, {}});
            if (qualified)
                break;
        }
    }
    return true;
}
using Commands = std::vector<std::vector<c::Mechanism>>;
c::Mechanism event(c::Mechanism::Kind kind, unsigned source, unsigned target, unsigned k)
{
    return {kind, source, target, k, 0};
}
struct CycleSites {
    Commands published, seed, waits, finished;
    std::vector<std::vector<std::pair<unsigned, unsigned>>> acquired; // cycle,lane
    std::vector<std::vector<std::pair<unsigned, unsigned>>> finishedAcquired;
    std::vector<std::vector<unsigned>> initialized;
    explicit CycleSites(unsigned size)
        : published(size),
          seed(size),
          waits(size),
          finished(size),
          acquired(size),
          finishedAcquired(size),
          initialized(size)
    {}
    void complete()
    {
        for (unsigned id = 0; id < acquired.size(); ++id)
            acquired[id].insert(acquired[id].end(), finishedAcquired[id].begin(), finishedAcquired[id].end());
    }
};
void addCycleSites(const Cycle& cycle, unsigned index, const Cuts& cuts, CycleSites& sites)
{
    const auto& groups = cycle.groups;
    unsigned last = groups.back().lane, first = groups.front().lane;
    sites.seed[cycle.owner].push_back(event(c::Mechanism::Publish, last, first, cycle.keys.at({last, first})));
    sites.initialized[cycle.owner].push_back(index);
    auto finish = cuts.next[cycle.owner];
    sites.waits[finish].push_back(event(c::Mechanism::Acquire, last, first, cycle.keys.at({last, first})));
    sites.acquired[finish].push_back({index, first});
    // The final release WAIT is on FIRST, not LAST. Carry that consumption
    // back to LAST before a reentrant invocation can publish its next seed.
    // This is an empty forward visit, not a reset of event state or PIPE_ALL.
    for (unsigned i = 0; i + 1 < groups.size(); ++i) {
        unsigned a = groups[i].lane, b = groups[i + 1].lane, k = cycle.keys.at({a, b});
        sites.finished[finish].push_back(event(c::Mechanism::Publish, a, b, k));
        sites.finished[finish].push_back(event(c::Mechanism::Acquire, a, b, k));
        sites.finishedAcquired[finish].push_back({index, b});
    }
    for (unsigned i = 0; i < groups.size(); ++i) {
        const auto &g = groups[i], &previous = groups[(i + groups.size() - 1) % groups.size()];
        const auto& next = groups[(i + 1) % groups.size()];
        for (unsigned node : g.first) {
            sites.waits[node].push_back(
                event(c::Mechanism::Acquire, previous.lane, g.lane, cycle.keys.at({previous.lane, g.lane})));
            sites.acquired[node].push_back({index, g.lane});
        }
        for (unsigned node : g.last)
            sites.published[cuts.next[node]].push_back(
                event(c::Mechanism::Publish, g.lane, next.lane, cycle.keys.at({g.lane, next.lane})));
    }
}
void acquireCells(c::State& state, const Cycle& cycle, unsigned observer)
{
    for (unsigned cell : cycle.cells)
        state.pending[observer][cell] = {};
}
c::Effects cycleDemands(const c::Program& p, const Cycle& cycle)
{
    c::Effects effects(p.cells);
    for (unsigned cell : cycle.cells)
        effects[cell].writers = 1;
    return effects;
}
// Completion is a source-prefix property, not a property of the motivating
// storage cell. For each live, certified physical key retain the MAY work not
// covered by its publication. Later payload is added to every receipt, so a
// WAIT never completes a newer occurrence merely because its cell/lane match.
// The population is bounded by the target's finite directed-key pool.
struct PrefixState : c::State {
    using EventKey = std::tuple<unsigned, unsigned, unsigned>;
    std::map<EventKey, c::Effects> receipts;
    explicit PrefixState(unsigned cells) : State(cells) {}
    void seed(const c::Effects& effects)
    {
        State::seed(effects);
        for (auto& [key, remaining] : receipts)
            merge(remaining, effects);
    }
    void enterLoop(const c::Effects& effects)
    {
        // Forget optional completion evidence, NEVER the actual token protocol.
        // A static publication outside/before this visit cannot describe every
        // incoming backedge generation. The separately checked cycle invariant
        // still supplies its storage-specific credits.
        receipts.clear();
        State::seed(effects);
    }
    void join(const PrefixState& other)
    {
        State::join(other);
        for (auto it = receipts.begin(); it != receipts.end();) {
            auto found = other.receipts.find(it->first);
            if (found == other.receipts.end())
                it = receipts.erase(it);
            else {
                merge(it->second, found->second);
                ++it;
            }
        }
    }
    void publish(const c::Mechanism& m)
    {
        auto remaining = pending[m.first];
        for (auto& h : remaining) {
            h.readers &= ~(1u << m.first);
            h.writers &= ~(1u << m.first);
        }
        receipts[{m.first, m.second, m.forwardKey}] = std::move(remaining);
    }
    void acquirePrefix(const c::Mechanism& m)
    {
        auto it = receipts.find({m.first, m.second, m.forwardKey});
        if (it == receipts.end())
            return;
        for (unsigned cell = 0; cell < it->second.size(); ++cell) {
            pending[m.second][cell].readers &= it->second[cell].readers;
            pending[m.second][cell].writers &= it->second[cell].writers;
        }
        receipts.erase(it);
    }
};
} // namespace

namespace {
c::Result constructCutCandidate(const c::Program& p, bool reserveFallback)
{
    using namespace c;
    c::Result result;
    Cuts cuts;
    if (!discoverCuts(p, cuts, result.reason))
        return result;
    // One global allocation population. Precision and ordinary protocols never
    // alias live keys. A retry can reserve a conservative key, but fitting
    // populations may use the whole pool. Repeated directions WITHIN a cycle share its causal
    // consumption/return path, not a lexical assertion that WAIT has finished.
    std::map<std::pair<unsigned, unsigned>, std::set<unsigned>> occupied;
    std::vector<Cycle> allocated;
    for (auto cycle : cuts.cycles) {
        auto trial = occupied;
        bool valid = true;
        for (unsigned i = 0; i < cycle.groups.size(); ++i) {
            auto a = cycle.groups[i].lane, b = cycle.groups[(i + 1) % cycle.groups.size()].lane;
            auto direction = std::make_pair(a, b);
            if (cycle.keys.count(direction))
                continue;
            auto fallback = key(p, a, b);
            std::optional<unsigned> selected;
            for (unsigned k : p.target.compilerKeys)
                if ((!reserveFallback || k != fallback) && !trial[direction].count(k) &&
                    p.target.available(lane(p, a), lane(p, b), k)) {
                    selected = k;
                    break;
                }
            if (!selected) {
                valid = false;
                break;
            }
            cycle.keys[direction] = *selected;
            trial[direction].insert(*selected);
        }
        if (valid) {
            occupied = std::move(trial);
            allocated.push_back(std::move(cycle));
        }
    }
    cuts.cycles = std::move(allocated);
    result.cutCycles = cuts.cycles.size();
    auto fallbackProgram = p;
    for (const auto& cycle : cuts.cycles)
        for (auto [direction, k] : cycle.keys)
            fallbackProgram.target.reservations.push_back({lane(p, direction.first), lane(p, direction.second), k});
    CycleSites sites(p.nodes.size());
    for (unsigned i = 0; i < cuts.cycles.size(); ++i)
        addCycleSites(cuts.cycles[i], i, cuts, sites);
    sites.complete();
    result.before.resize(p.nodes.size());
    std::function<bool(unsigned, PrefixState&)> visit = [&](unsigned id, PrefixState& state) {
        ++result.nodeVisits;
        result.cellVisits += p.cells * LaneCount;
        const auto& n = p.nodes[id];
        auto& commands = result.before[id];
        commands = sites.published[id];
        for (const auto& m : sites.published[id])
            state.publish(m);
        for (unsigned i : sites.initialized[id]) {
            const auto& cycle = cuts.cycles[i];
            auto effects = cycleDemands(p, cycle);
            unsigned observer = cycle.groups.back().lane;
            for (unsigned source = 0; source < LaneCount; ++source)
                if (state.demands(observer, effects) & (1u << source))
                    if (!acquire(fallbackProgram, source, observer, state, commands))
                        return false;
        }
        commands.insert(commands.end(), sites.seed[id].begin(), sites.seed[id].end());
        commands.insert(commands.end(), sites.waits[id].begin(), sites.waits[id].end());
        commands.insert(commands.end(), sites.finished[id].begin(), sites.finished[id].end());
        for (const auto& m : sites.seed[id])
            state.publish(m);
        unsigned acquisition = 0;
        auto receive = [&](const Mechanism& m) {
            state.acquirePrefix(m);
            auto [i, observer] = sites.acquired[id][acquisition++];
            acquireCells(state, cuts.cycles[i], observer);
        };
        for (const auto& m : sites.waits[id])
            receive(m);
        for (const auto& m : sites.finished[id]) {
            if (m.kind == Mechanism::Publish)
                state.publish(m);
            else
                receive(m);
        }
        if (n.kind == Node::Operation) {
            if (needsVisibility(p, n, state)) {
                result.reason = "visibility has no qualified compositional realization";
                return false;
            }
            for (unsigned source = 0; source < LaneCount; ++source)
                if (state.demands(n.lane, n.effects) & (1u << source)) {
                    if (!acquire(fallbackProgram, source, n.lane, state, commands))
                        return false;
                    ++result.acquisitions;
                }
            state.seed(n.effects);
            return true;
        }
        if (n.kind == Node::Choice) {
            auto other = state;
            if (!visit(n.children[0], state) || !visit(n.children[1], other))
                return false;
            state.join(other);
            return true;
        }
        if (n.kind == Node::For || n.kind == Node::While) {
            auto entry = state;
            state.enterLoop(cuts.effects[id]);
            if (!visit(n.children[0], state))
                return false;
            if (n.kind == Node::While) {
                auto exit = state;
                if (!visit(n.children[1], state))
                    return false;
                state = std::move(exit);
            } else
                state.join(entry);
            return true;
        }
        for (unsigned child : n.children)
            if (!visit(child, state))
                return false;
        return true;
    };
    PrefixState state(p.cells);
    result.success = visit(p.nodes.size() - 1, state);
    if (!result.success) {
        result.before.clear();
        if (result.reason.empty())
            result.reason = "target cannot realize cut initialization";
    }
    return result;
}
} // namespace

c::Result c::constructCuts(const Program& p)
{
    auto result = constructCutCandidate(p, false);
    // A single transactional retry reserves a conservative key if precision
    // used a direction needed by the remaining physical demands. No search over
    // colorings or repeated refinement, and no partial first candidate escapes.
    if (!result.success) {
        result = constructCutCandidate(p, true);
        result.allocationRetries = 1;
    }
    return result;
}

c::Result c::verifyCuts(const Program& p, const std::vector<std::vector<Mechanism>>& input)
{
    Result result;
    Cuts cuts;
    if (input.size() != p.nodes.size() || !discoverCuts(p, cuts, result.reason))
        return result;
    using Direction = std::pair<unsigned, unsigned>;
    using Position = std::pair<unsigned, unsigned>; // original cut, command kind
    using Key = std::pair<Direction, unsigned>;
    std::map<Key, std::vector<Position>> population;
    for (unsigned id = 0; id < input.size(); ++id)
        for (const auto& m : input[id]) {
            if (m.kind != Mechanism::Publish && m.kind != Mechanism::Acquire)
                continue;
            if (m.first >= LaneCount || m.second >= LaneCount ||
                !p.target.available(lane(p, m.first), lane(p, m.second), m.forwardKey) ||
                !p.target.event(lane(p, m.first), lane(p, m.second))) {
                result.reason = "invalid cut event";
                return result;
            }
            population[{{m.first, m.second}, m.forwardKey}].push_back({id, unsigned(m.kind)});
        }
    for (auto& [k, positions] : population)
        std::sort(positions.begin(), positions.end());
    std::set<Key> used;
    std::vector<Cycle> certified;
    // Infer numeric protocols from the actual IR, not from constructor receipts
    // or allocation decisions. Exact cut populations establish participation;
    // the complete cyclic lane word establishes consumption-before-rearm.
    for (auto cycle : cuts.cycles) {
        std::map<Direction, std::vector<Position>> wanted;
        unsigned first = cycle.groups.front().lane, last = cycle.groups.back().lane;
        wanted[{last, first}].push_back({cycle.owner, unsigned(Mechanism::Publish)});
        wanted[{last, first}].push_back({cuts.next[cycle.owner], unsigned(Mechanism::Acquire)});
        for (unsigned i = 0; i + 1 < cycle.groups.size(); ++i) {
            auto direction = std::make_pair(cycle.groups[i].lane, cycle.groups[i + 1].lane);
            wanted[direction].push_back({cuts.next[cycle.owner], unsigned(Mechanism::Publish)});
            wanted[direction].push_back({cuts.next[cycle.owner], unsigned(Mechanism::Acquire)});
        }
        for (unsigned i = 0; i < cycle.groups.size(); ++i) {
            const auto &g = cycle.groups[i], &following = cycle.groups[(i + 1) % cycle.groups.size()];
            const auto& previous = cycle.groups[(i + cycle.groups.size() - 1) % cycle.groups.size()];
            for (unsigned id : g.first)
                wanted[{previous.lane, g.lane}].push_back({id, unsigned(Mechanism::Acquire)});
            for (unsigned id : g.last)
                wanted[{g.lane, following.lane}].push_back({cuts.next[id], unsigned(Mechanism::Publish)});
        }
        bool valid = true;
        for (auto& [direction, positions] : wanted) {
            std::sort(positions.begin(), positions.end());
            bool found = false;
            for (unsigned k : p.target.compilerKeys) {
                Key candidate = {direction, k};
                auto it = population.find(candidate);
                if (!used.count(candidate) && it != population.end() && it->second == positions) {
                    cycle.keys[direction] = k;
                    found = true;
                    break;
                }
            }
            if (!found) {
                valid = false;
                break;
            }
        }
        if (valid) {
            for (auto [direction, k] : cycle.keys)
                used.insert({direction, k});
            certified.push_back(std::move(cycle));
        }
    }
    auto fallbackProgram = p;
    for (const auto& k : used)
        fallbackProgram.target.reservations.push_back({lane(p, k.first.first), lane(p, k.first.second), k.second});
    Commands actual(input.size());
    for (unsigned id = 0; id < input.size(); ++id) {
        const auto& commands = input[id];
        for (unsigned i = 0; i < commands.size();) {
            auto m = commands[i];
            if ((m.kind != Mechanism::Publish && m.kind != Mechanism::Acquire) ||
                used.count({{m.first, m.second}, m.forwardKey})) {
                actual[id].push_back(m);
                ++i;
                continue;
            }
            // Numeric keys not owned by any reconstructed cut cycle may implement
            // the ordinary four-command protocol. Infer it from actual commands.
            if (i + 4 > commands.size() || m.kind != Mechanism::Publish || m.first >= m.second) {
                result.reason = "unmatched cut protocol or conservative packet";
                return result;
            }
            auto reply = commands[i + 2];
            Mechanism packet{Mechanism::Rendezvous, m.first, m.second, m.forwardKey, reply.forwardKey};
            if (!(commands[i + 1] == event(Mechanism::Acquire, m.first, m.second, m.forwardKey)) ||
                !(reply == event(Mechanism::Publish, m.second, m.first, reply.forwardKey)) ||
                !(commands[i + 3] == event(Mechanism::Acquire, m.second, m.first, reply.forwardKey)) ||
                !validMechanism(fallbackProgram, packet)) {
                result.reason = "invalid or cycle-aliased conservative packet";
                return result;
            }
            actual[id].push_back(packet);
            i += 4;
        }
    }
    cuts.cycles = std::move(certified);
    CycleSites sites(p.nodes.size());
    result.cutCycles = cuts.cycles.size();
    for (unsigned i = 0; i < cuts.cycles.size(); ++i)
        addCycleSites(cuts.cycles[i], i, cuts, sites);
    sites.complete();
    for (unsigned id = 0; id < actual.size(); ++id) {
        auto expected = sites.published[id];
        expected.insert(expected.end(), sites.seed[id].begin(), sites.seed[id].end());
        expected.insert(expected.end(), sites.waits[id].begin(), sites.waits[id].end());
        expected.insert(expected.end(), sites.finished[id].begin(), sites.finished[id].end());
        unsigned index = 0;
        for (const auto& m : actual[id])
            if (m.kind == Mechanism::Publish || m.kind == Mechanism::Acquire) {
                if (index >= expected.size() || !(m == expected[index++])) {
                    result.reason = "cut protocol command order changed";
                    return result;
                }
            }
        if (index != expected.size()) {
            result.reason = "missing cut protocol command";
            return result;
        }
    }
    std::function<bool(unsigned, PrefixState&)> check = [&](unsigned id, PrefixState& state) {
        ++result.nodeVisits;
        result.cellVisits += p.cells * LaneCount;
        unsigned publications = 0, acquisitions = 0;
        for (const auto& m : actual[id]) {
            if (m.kind == Mechanism::Publish) {
                state.publish(m);
                if (publications++ >= sites.published[id].size() &&
                    publications <= sites.published[id].size() + sites.seed[id].size()) {
                    unsigned seed = publications - sites.published[id].size() - 1;
                    const auto& cycle = cuts.cycles[sites.initialized[id][seed]];
                    if (state.demands(cycle.groups.back().lane, cycleDemands(p, cycle))) {
                        result.reason = "cut protocol initialization omits incoming physical work";
                        return false;
                    }
                }
            } else if (m.kind == Mechanism::Acquire) {
                state.acquirePrefix(m);
                auto [cycle, observer] = sites.acquired[id][acquisitions++];
                acquireCells(state, cuts.cycles[cycle], observer);
            } else {
                if (!validMechanism(fallbackProgram, m)) {
                    result.reason = "invalid conservative mechanism in cut plan";
                    return false;
                }
                apply(state, m);
            }
        }
        const auto& n = p.nodes[id];
        if (n.kind == Node::Operation) {
            if (needsVisibility(p, n, state) || state.demands(n.lane, n.effects)) {
                result.reason = "uncovered original physical obligation in cut plan";
                return false;
            }
            state.seed(n.effects);
            return true;
        }
        if (n.kind == Node::Choice) {
            auto other = state;
            if (!check(n.children[0], state) || !check(n.children[1], other))
                return false;
            state.join(other);
            return true;
        }
        if (n.kind == Node::For || n.kind == Node::While) {
            auto entry = state;
            state.enterLoop(cuts.effects[id]);
            if (!check(n.children[0], state))
                return false;
            if (n.kind == Node::While) {
                auto exit = state;
                if (!check(n.children[1], state))
                    return false;
                state = std::move(exit);
            } else
                state.join(entry);
            return true;
        }
        for (unsigned child : n.children)
            if (!check(child, state))
                return false;
        return true;
    };
    PrefixState state(p.cells);
    result.success = check(p.nodes.size() - 1, state);
    return result;
}
