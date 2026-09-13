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
uint64_t commandCost(const c::Mechanism& mechanism)
{
    if (mechanism.kind == c::Mechanism::Rendezvous)
        return 4;
    if (mechanism.kind == c::Mechanism::Visibility)
        return mechanism.visibilityAction == c::VisibilityAction::FenceOnly ? 1 : 2;
    return 1;
}
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
    if (m.participation != c::Mechanism::Every || m.loop != ~0u || m.word != ~0u)
        return false;
    if (m.first >= c::LaneCount || m.second >= c::LaneCount)
        return false;
    if (m.kind == c::Mechanism::Barrier)
        return p.target.barrier(lane(p, m.first));
    if (m.kind == c::Mechanism::Visibility)
        return p.core == Core::AIV && m.first == 0 && m.second == 0 && m.forwardKey == 0 && m.reverseKey == 0 &&
               m.visibilityAction <= c::VisibilityAction::InvalidateTarget;
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
    else if (m.kind == c::Mechanism::Visibility)
        state.visibility(m.visibilityAction);
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
        if (n.periodicOwner != ~0u &&
            (n.periodicOwner >= p.nodes.size() || p.nodes[n.periodicOwner].kind != c::Node::For ||
             (n.kind != c::Node::Sequence && n.kind != c::Node::Choice) || !n.periodicPeriod || n.periodicPeriod > 32 ||
             n.periodicLower < 0 || (n.periodicPeriod < 32 && (n.periodicResidues >> n.periodicPeriod)))) {
            reason = "invalid optional periodic domain";
            return false;
        }
        const auto operationLane = lane(p, n.lane);
        if (n.effects.size() != p.cells || n.lane >= c::LaneCount ||
            (n.entryGuardStart != ~0u && (n.kind != c::Node::For || n.entryGuardStart >= p.nodes.size())) ||
            (n.kind == c::Node::Operation &&
             (!n.children.empty() || !p.target.supports(operationLane) ||
              (n.lane != unsigned(Pipe::S) && !p.target.barrier(operationLane)))) ||
            (n.kind == c::Node::For && n.children.size() != 1) ||
            ((n.kind == c::Node::Choice || n.kind == c::Node::While) && n.children.size() != 2) ||
            n.kind > c::Node::While) {
            reason = "invalid composition node";
            return false;
        }
        for (const auto& e : n.effects)
            if (((e.readers | e.writers) & ~(1u << n.lane)) ||
                (n.kind != c::Node::Operation && (e.readers || e.writers))) {
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
    if (std::any_of(p.nodes.begin(), p.nodes.end(), [](const auto& n) { return n.periodicOwner != ~0u; })) {
        std::vector<uint64_t> begin(p.nodes.size()), end(p.nodes.size());
        std::vector<std::pair<unsigned, bool>> stack{{unsigned(p.nodes.size() - 1), false}};
        uint64_t position = 0;
        while (!stack.empty()) {
            auto [id, exiting] = stack.back();
            stack.pop_back();
            if (exiting) {
                end[id] = position++;
                continue;
            }
            begin[id] = position++;
            stack.push_back({id, true});
            for (unsigned child : p.nodes[id].children)
                stack.push_back({child, false});
        }
        for (unsigned id = 0; id < p.nodes.size(); ++id) {
            unsigned owner = p.nodes[id].periodicOwner;
            if (owner != ~0u && !(begin[owner] < begin[id] && end[id] < end[owner])) {
                reason = "periodic owner is not an original ancestor";
                return false;
            }
        }
    }
    return true;
}

enum class VisibilityNeed { None, FenceOnly, CleanSource, InvalidateTarget, UnsupportedMte3ToMte2 };

VisibilityNeed visibilityNeed(const c::Program& p, const c::Node& n, const c::State& state)
{
    const unsigned observer = n.lane;
    bool fenceOnly = false, cleanSource = false, invalidateTarget = false, unsupportedMte3ToMte2 = false;
    for (unsigned i = 0; i < p.globalMemory.size(); ++i) {
        bool targetReads = n.effects[i].readers != 0;
        bool targetWrites = n.effects[i].writers != 0;
        if (!p.globalMemory[i] || (!targetReads && !targetWrites))
            continue;
        for (unsigned source = 0; source < c::LaneCount; ++source) {
            if (!(state.written[observer][i] & (uint8_t(1u) << source)) || source == observer)
                continue;
            // The qualified hardware model requires an explicit cache/fence
            // recipe only when a GM value crosses the scalar data cache.  It
            // separately keeps MTE3->MTE2 same-address publication fail-closed.
            // Other non-scalar pairs still need completion, which is tracked
            // independently by State::pending and a directed handoff.
            bool scalarCrossing = (source == unsigned(Pipe::S)) != (observer == unsigned(Pipe::S));
            unsupportedMte3ToMte2 |=
                targetReads && source == unsigned(Pipe::MTE3) && observer == unsigned(Pipe::MTE2);
            cleanSource |= scalarCrossing && observer != unsigned(Pipe::S);
            invalidateTarget |= scalarCrossing && observer == unsigned(Pipe::S) && targetReads;
            fenceOnly |= scalarCrossing && observer == unsigned(Pipe::S) && targetWrites && !targetReads;
        }
    }
    // A cache recipe must never hide the separately unqualified direct GM
    // publication case when conservative alias merging puts both histories in
    // one cell.
    if (unsupportedMte3ToMte2)
        return VisibilityNeed::UnsupportedMte3ToMte2;
    if (cleanSource)
        return VisibilityNeed::CleanSource;
    if (invalidateTarget)
        return VisibilityNeed::InvalidateTarget;
    if (fenceOnly)
        return VisibilityNeed::FenceOnly;
    return VisibilityNeed::None;
}

bool visibilityCovered(
    const c::Program& p, const c::Node& n, const c::State& state,
    const std::vector<c::Mechanism>& mechanisms)
{
    auto need = visibilityNeed(p, n, state);
    if (need == VisibilityNeed::None)
        return true;
    if (need == VisibilityNeed::UnsupportedMte3ToMte2)
        return false;
    auto expected =
        need == VisibilityNeed::CleanSource        ? c::VisibilityAction::CleanSource :
        need == VisibilityNeed::InvalidateTarget ? c::VisibilityAction::InvalidateTarget :
                                                   c::VisibilityAction::FenceOnly;
    return std::any_of(mechanisms.begin(), mechanisms.end(), [&](const auto& mechanism) {
        return mechanism.kind == c::Mechanism::Visibility &&
               mechanism.participation == c::Mechanism::Every &&
               mechanism.visibilityAction == expected;
    });
}

bool realizeVisibility(
    const c::Program& p, const c::Node& n, c::State& state, std::vector<c::Mechanism>& commands,
    std::string& reason, uint64_t& requirements)
{
    auto need = visibilityNeed(p, n, state);
    if (need == VisibilityNeed::None)
        return true;
    if (need == VisibilityNeed::UnsupportedMte3ToMte2) {
        reason = "MTE3-to-MTE2 GM publication has no qualified compositional realization";
        return false;
    }
    c::Mechanism mechanism;
    mechanism.kind = c::Mechanism::Visibility;
    mechanism.visibilityAction =
        need == VisibilityNeed::CleanSource        ? c::VisibilityAction::CleanSource :
        need == VisibilityNeed::InvalidateTarget ? c::VisibilityAction::InvalidateTarget :
                                                   c::VisibilityAction::FenceOnly;
    apply(state, mechanism);
    commands.push_back(mechanism);
    ++requirements;
    return true;
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
    for (auto& history : written)
        history.resize(cells);
}
void c::State::join(const State& other)
{
    for (unsigned lane = 0; lane < LaneCount; ++lane)
        merge(pending[lane], other.pending[lane]);
    for (unsigned observer = 0; observer < LaneCount; ++observer)
        for (unsigned i = 0; i < written[observer].size(); ++i)
            written[observer][i] |= other.written[observer][i];
}
void c::State::seed(const Effects& effects)
{
    for (auto& history : pending)
        merge(history, effects);
    for (auto& history : written)
        for (unsigned i = 0; i < effects.size(); ++i)
            history[i] |= effects[i].writers;
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
void c::State::visibility(VisibilityAction action)
{
    // The qualified GM fence drains all AIV physical pipelines. Completion
    // becomes universal, while cache maintenance discharges only the matching
    // scalar-cache direction. The disputed MTE3->MTE2 publication bit remains.
    for (auto& observer : pending)
        for (auto& effect : observer)
            effect = {};
    const uint8_t scalar = uint8_t(1u) << unsigned(Pipe::S);
    if (action == VisibilityAction::CleanSource) {
        for (unsigned observer = 0; observer < LaneCount; ++observer)
            if (observer != unsigned(Pipe::S))
                for (auto& sources : written[observer])
                    sources &= ~scalar;
    } else if (action == VisibilityAction::InvalidateTarget) {
        // A non-scalar source is globally published by the fence. A scalar
        // read additionally invalidates DCache.
        for (auto& sources : written[unsigned(Pipe::S)])
            sources &= scalar;
    }
    // FenceOnly discharges one write-only WAW at its exact target cut. It does
    // not invalidate DCache and therefore cannot erase visibility history.
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
    // The admitted scalar payload contract is synchronous within PIPE_S. It
    // does not imply scalar-to-other-pipe completion and never licenses a
    // PIPE_S barrier; those demands still require a qualified directed event.
    if (observer == unsigned(Pipe::S))
        result &= ~(uint8_t(1u) << unsigned(Pipe::S));
    return result;
}
bool c::Mechanism::operator==(const Mechanism& m) const
{
    return kind == m.kind && first == m.first && second == m.second && forwardKey == m.forwardKey &&
           reverseKey == m.reverseKey && participation == m.participation && loop == m.loop && word == m.word &&
           visibilityAction == m.visibilityAction;
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
            if (!realizeVisibility(
                    p, n, state, result.before[id], result.reason,
                    result.visibilityRequirements))
                return false;
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
                if (!visibilityCovered(p, n, state, actual[id])) {
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
        acquireRemaining(m.second, it->second);
        receipts.erase(it);
    }
    void acquireRemaining(unsigned observer, const c::Effects& remaining)
    {
        for (unsigned cell = 0; cell < remaining.size(); ++cell) {
            pending[observer][cell].readers &= remaining[cell].readers;
            pending[observer][cell].writers &= remaining[cell].writers;
        }
    }
};

// A closed, single-Sequence ring has no seed or cleanup outside its visit.
// The last lane publishes at the NEXT visit's first cut, after its previous
// work in that lane's queue. The normal two-copy event-word proof establishes
// consumption-before-rearm; skipped visits execute no partial protocol.
// This optional certificate concerns internal cell generations only. Every
// incoming history bit is retained until an ordinary prefix transfer proves it.
struct DemandRings {
    // Optional precision has a representation-work ceiling, independent of
    // numeric trip counts. Exceeding it leaves ordinary demands unchanged.
    static constexpr uint64_t MaxCells = 1u << 20;
    static bool affordable(const c::Program& p) { return !p.nodes.empty() && p.cells <= MaxCells / p.nodes.size(); }
    Cuts cuts;
    Commands sites;
    std::map<PrefixState::EventKey, unsigned> owners;
    std::set<unsigned> loopOwners;
    explicit DemandRings(unsigned size) : sites(size) {}
    bool discover(const c::Program& p, std::string& reason)
    {
        if (!affordable(p))
            return true;
        if (!discoverCuts(p, cuts, reason))
            return false;
        std::vector<Cycle> shapes;
        std::map<std::vector<unsigned>, unsigned> shared;
        for (auto cycle : cuts.cycles) {
            bool straight = p.nodes[cycle.owner].kind == c::Node::For;
            // No cell-accessing nested segment is hidden between endpoints.
            // The path from this Sequence to its nearest For contains only
            // Sequence/Choice nodes: one iteration executes the entire word
            // once or skips it. No nested trip count is assigned a generation.
            for (unsigned child : p.nodes[cycle.region].children)
                if (p.nodes[child].kind != c::Node::Operation)
                    for (unsigned cell : cycle.cells) {
                        const auto& e = cuts.effects[child][cell];
                        straight &= !(e.readers | e.writers);
                    }
            std::set<std::pair<unsigned, unsigned>> directions;
            for (unsigned g = 0; g < cycle.groups.size(); ++g)
                straight &=
                    directions.insert({cycle.groups[g].lane, cycle.groups[(g + 1) % cycle.groups.size()].lane}).second;
            std::vector<unsigned> signature{cycle.owner, cycle.region};
            for (const auto& g : cycle.groups) {
                straight &= g.first.size() == 1 && g.last.size() == 1;
                for (unsigned id : g.first)
                    straight &= cuts.parent[id] == cycle.region && p.nodes[id].kind == c::Node::Operation;
                for (unsigned id : g.last)
                    straight &= cuts.parent[id] == cycle.region && p.nodes[id].kind == c::Node::Operation;
                signature.insert(signature.end(), {g.lane, g.first.front(), g.last.front()});
            }
            if (!straight)
                continue;
            auto [it, inserted] = shared.emplace(signature, shapes.size());
            if (inserted)
                shapes.push_back(std::move(cycle));
            else
                shapes[it->second].cells.insert(shapes[it->second].cells.end(), cycle.cells.begin(), cycle.cells.end());
        }
        cuts.cycles = std::move(shapes);
        // Two-group directional sharing: one group's exact boundaries are
        // common, and the other group's union remains one noninterleaving lane
        // interval. Within-visit edges retain a common publication/acquisition.
        // The wrap SET/WAIT may move to an earlier first cut together: its SET
        // still follows the common last lane's prior visit and cannot capture
        // MORE intervening source work. Incoming E remains a separate demand.
        // Two linear grouping passes, not pairwise family/set-cover search.
        std::vector<unsigned> position(p.nodes.size());
        for (const auto& n : p.nodes)
            for (unsigned i = 0; i < n.children.size(); ++i)
                position[n.children[i]] = i;
        std::set<unsigned> firstPassFamilies;
        for (unsigned anchor : {0u, 1u}) {
            std::map<std::vector<unsigned>, unsigned> families;
            std::vector<Cycle> merged;
            for (auto cycle : cuts.cycles) {
                if (cycle.groups.size() != 2 || (anchor == 1 && firstPassFamilies.count(cycle.cells.front()))) {
                    merged.push_back(std::move(cycle));
                    continue;
                }
                const auto& fixed = cycle.groups[anchor];
                std::vector<unsigned> signature{cycle.owner,          cycle.region,        cycle.groups[0].lane,
                                                cycle.groups[1].lane, fixed.first.front(), fixed.last.front()};
                auto found = families.find(signature);
                if (found != families.end()) {
                    auto groups = merged[found->second].groups;
                    auto& extent = groups[1 - anchor];
                    const auto& extra = cycle.groups[1 - anchor];
                    if (position[extra.first.front()] < position[extent.first.front()])
                        extent.first = extra.first;
                    if (position[extra.last.front()] > position[extent.last.front()])
                        extent.last = extra.last;
                    if (position[groups[0].last.front()] < position[groups[1].first.front()]) {
                        auto& family = merged[found->second];
                        family.groups = std::move(groups);
                        family.cells.insert(family.cells.end(), cycle.cells.begin(), cycle.cells.end());
                        if (anchor == 0)
                            firstPassFamilies.insert(family.cells.front());
                        continue;
                    }
                }
                families[std::move(signature)] = merged.size();
                merged.push_back(std::move(cycle));
            }
            cuts.cycles = std::move(merged);
        }
        return true;
    }
    struct Endpoint {
        unsigned cut;
        c::Mechanism::Kind kind;
        unsigned source, target;
    };
    std::vector<Endpoint> pattern(const Cycle& cycle) const
    {
        std::vector<Endpoint> out;
        out.push_back(
            {cycle.groups.front().first.front(), c::Mechanism::Publish, cycle.groups.back().lane,
             cycle.groups.front().lane});
        for (unsigned g = 0; g < cycle.groups.size(); ++g) {
            const auto& group = cycle.groups[g];
            const auto& previous = cycle.groups[(g + cycle.groups.size() - 1) % cycle.groups.size()];
            out.push_back({group.first.front(), c::Mechanism::Acquire, previous.lane, group.lane});
            if (g + 1 < cycle.groups.size())
                out.push_back(
                    {cuts.next[group.last.front()], c::Mechanism::Publish, group.lane, cycle.groups[g + 1].lane});
        }
        return out;
    }
    void index()
    {
        for (unsigned i = 0; i < cuts.cycles.size(); ++i) {
            const auto& cycle = cuts.cycles[i];
            loopOwners.insert(cycle.owner);
            for (const auto& e : pattern(cycle))
                sites[e.cut].push_back(event(e.kind, e.source, e.target, cycle.keys.at({e.source, e.target})));
            for (auto [direction, key] : cycle.keys)
                owners.emplace(PrefixState::EventKey{direction.first, direction.second, key}, i);
        }
    }
    bool infer(const c::Program& p, const Commands& actual, std::string& reason)
    {
        if (!discover(p, reason))
            return false;
        using Position = std::pair<unsigned, unsigned>;
        std::map<PrefixState::EventKey, std::vector<Position>> population;
        for (unsigned id = 0; id < actual.size(); ++id)
            for (const auto& m : actual[id])
                if (m.participation == c::Mechanism::Every &&
                    (m.kind == c::Mechanism::Publish || m.kind == c::Mechanism::Acquire))
                    population[{m.first, m.second, m.forwardKey}].push_back({id, unsigned(m.kind)});
        for (auto& [key, positions] : population)
            std::sort(positions.begin(), positions.end());
        std::set<PrefixState::EventKey> used;
        std::vector<Cycle> certified;
        for (auto cycle : cuts.cycles) {
            std::map<std::pair<unsigned, unsigned>, std::vector<Position>> wanted;
            for (const auto& e : pattern(cycle))
                wanted[{e.source, e.target}].push_back({e.cut, unsigned(e.kind)});
            bool found = true;
            for (auto& [direction, positions] : wanted) {
                std::sort(positions.begin(), positions.end());
                bool matched = false;
                for (unsigned k : p.target.compilerKeys) {
                    PrefixState::EventKey key{direction.first, direction.second, k};
                    auto it = population.find(key);
                    if (!used.count(key) && it != population.end() && it->second == positions) {
                        cycle.keys[direction] = k;
                        matched = true;
                        break;
                    }
                }
                found &= matched;
            }
            if (found) {
                for (auto [direction, k] : cycle.keys)
                    used.insert({direction.first, direction.second, k});
                certified.push_back(std::move(cycle));
            }
        }
        cuts.cycles = std::move(certified);
        index();
        // Endpoint populations alone do not establish same-cut ordering.
        for (unsigned id = 0; id < actual.size(); ++id) {
            std::vector<c::Mechanism> word;
            for (const auto& m : actual[id])
                if (m.participation == c::Mechanism::Every && owners.count({m.first, m.second, m.forwardKey}) &&
                    (m.kind == c::Mechanism::Publish || m.kind == c::Mechanism::Acquire))
                    word.push_back(m);
            if (word != sites[id]) {
                reason = "recurring demand ring command order changed";
                return false;
            }
        }
        return true;
    }
    bool credit(
        const c::Mechanism& m, PrefixState& state, const std::map<unsigned, c::State>& incoming,
        std::string& reason) const
    {
        if (m.kind != c::Mechanism::Acquire || m.participation != c::Mechanism::Every)
            return true;
        auto found = owners.find({m.first, m.second, m.forwardKey});
        if (found == owners.end())
            return true;
        const auto& cycle = cuts.cycles[found->second];
        auto entry = incoming.find(cycle.owner);
        if (entry == incoming.end()) {
            reason = "recurring demand ring has no incoming generation scope";
            return false;
        }
        for (unsigned cell : cycle.cells) {
            auto& pending = state.pending[m.second][cell];
            const auto& external = entry->second.pending[m.second][cell];
            const auto& internal = cuts.effects[cycle.owner][cell];
            if ((pending.readers & ~(external.readers | internal.readers)) ||
                (pending.writers & ~(external.writers | internal.writers))) {
                reason = "recurring demand ring omits external physical history";
                return false;
            }
            pending.readers &= external.readers;
            pending.writers &= external.writers;
        }
        return true;
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
            if (!realizeVisibility(
                    p, n, state, commands, result.reason,
                    result.visibilityRequirements))
                return false;
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
            if (!visibilityCovered(p, n, state, actual[id]) || state.demands(n.lane, n.effects)) {
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

namespace {
// Static demand placement, independent of event numbering and periodic words.
// A previous child contributes its full MAY effects at its common exit. Within
// one sequence the last relevant read/write on a lane names a sufficient
// prefix, even when that child is an arbitrary conditional or nested loop.
struct DemandAnalysis {
    using EntryWitnessKey = std::pair<unsigned, unsigned>;
    using UniversalDemandKey = std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, std::vector<unsigned>>;
    static constexpr uint64_t MaxIncomingEntryWork = 1u << 20;
    static constexpr uint64_t MaxChoiceIncomingWork = 1u << 20;
    struct ChoiceIncomingOptions {
        struct CommonChoiceProposal {
            c::CompletionDemand demand;
            c::Effects witness;
        };
        struct UniversalChoiceFamily {
            unsigned choice = NoCut, parent = NoCut, publication = NoCut;
            unsigned source = NoCut, observer = NoCut;
            std::vector<unsigned> cells;
            std::vector<c::CompletionDemand> alternatives;
            uint64_t prefixSteps = 0;
        };
        bool enabled = false, commonEnabled = true, corrupt = false;
        bool alternativesEnabled = false, corruptAlternative = false;
        uint64_t limit = MaxChoiceIncomingWork;
        // One allowance shared by discovery, refinement, allocation replay
        // and ring attempts. A new DemandAnalysis must not reset this budget.
        mutable uint64_t charged = 0;
        mutable bool exhausted = false;
        mutable uint64_t passes = 0, exhaustionPass = 0;
        mutable bool commonAnalyzed = false;
        mutable std::vector<CommonChoiceProposal> common;
        const std::vector<c::CompletionDemand>* baselineDemands = nullptr;
        mutable bool alternativesAnalyzed = false;
        mutable std::vector<UniversalChoiceFamily> alternatives;
    };
    struct FirstLaneSummary {
        std::array<unsigned, MaxAlternatives> operations{};
        unsigned count = 0;
        bool mayNoLane = true, valid = true;

        bool insert(unsigned id)
        {
            if (std::find(operations.begin(), operations.begin() + count, id) != operations.begin() + count)
                return true;
            if (count == operations.size())
                return valid = false;
            operations[count++] = id;
            return true;
        }
    };
    using FirstSummary = std::array<FirstLaneSummary, c::LaneCount>;
    std::vector<c::Effects> effects;
    std::vector<uint64_t> subtreeNodes;
    std::vector<uint8_t> laneMask, followingLaneMask;
    std::vector<FirstSummary> entryFirst;
    std::vector<std::vector<unsigned>> owned;
    std::vector<unsigned> parent;
    std::vector<unsigned> scopeOrder;
    std::vector<unsigned> position, next;
    std::vector<std::vector<c::CompletionDemand>> entries;
    std::vector<std::vector<c::CompletionDemand>> requests;
    std::vector<std::set<unsigned>> capture;
    std::vector<std::vector<PrefixState::EventKey>> release;
    std::map<EntryWitnessKey, c::Effects> entryWitnesses;
    std::map<EntryWitnessKey, c::Effects> choiceWitnesses;
    std::vector<ChoiceIncomingOptions::UniversalChoiceFamily> universalChoices;
    std::map<UniversalDemandKey, unsigned> universalChoiceRequests;
    uint64_t entrySummarySlots = 0, entrySummaryScans = 0, entryStorageUnits = 0;
    uint64_t entryCandidatePairs = 0;
    uint64_t entryWitnessCells = 0, entryWitnessCount = 0, entrySourceOverlapRejections = 0;
    uint64_t entrySummarySkipped = 0;
    uint64_t choiceDemandCandidates = 0, choiceDemandWork = 0;

    std::optional<unsigned> universalChoiceFamily(const c::CompletionDemand& demand) const
    {
        auto found = universalChoiceRequests.find(
            {demand.scope, demand.publication, demand.acquisition, demand.source, demand.observer, demand.cells});
        return found == universalChoiceRequests.end() ? std::nullopt : std::optional<unsigned>(found->second);
    }

    const c::Effects* entryWitness(const c::CompletionDemand& demand) const
    {
        auto found = entryWitnesses.find({demand.acquisition, demand.observer});
        return found == entryWitnesses.end() ? nullptr : &found->second;
    }

    const c::Effects* choiceWitness(const c::CompletionDemand& demand) const
    {
        auto found = choiceWitnesses.find({demand.acquisition, demand.observer});
        return found == choiceWitnesses.end() ? nullptr : &found->second;
    }

    const c::CompletionDemand* entryDemand(
        unsigned loop, unsigned acquisition, unsigned source, unsigned observer) const
    {
        if (loop >= entries.size())
            return nullptr;
        auto found = std::find_if(entries[loop].begin(), entries[loop].end(), [&](const auto& demand) {
            return demand.acquisition == acquisition && demand.source == source && demand.observer == observer;
        });
        return found == entries[loop].end() ? nullptr : &*found;
    }

    const FirstLaneSummary* firstSummary(unsigned root, unsigned observer) const
    {
        if (root >= entryFirst.size() || observer >= c::LaneCount)
            return nullptr;
        return &entryFirst[root][observer];
    }

    const c::CompletionDemand* lateEntryDemand(unsigned loop, unsigned site, unsigned source, unsigned observer) const
    {
        if (loop >= entries.size())
            return nullptr;
        const c::CompletionDemand* found = nullptr;
        for (const auto& demand : entries[loop]) {
            if (demand.source != source || demand.observer != observer || demand.acquisition >= effects.size() ||
                effects.size() <= site)
                continue;
            const auto* summary = firstSummary(demand.acquisition, observer);
            if (!summary || !summary->valid || summary->mayNoLane || !summary->count ||
                summary->count > MaxAlternatives ||
                std::find(summary->operations.begin(), summary->operations.begin() + summary->count, site) ==
                    summary->operations.begin() + summary->count)
                continue;
            if (found)
                return nullptr;
            found = &demand;
        }
        return found;
    }

    void recordEntryStats(c::Result& result) const
    {
        result.entrySummarySlots = entrySummarySlots;
        result.entrySummaryScans = entrySummaryScans;
        result.entryStorageUnits = entryStorageUnits;
        result.entryCandidatePairs = entryCandidatePairs;
        result.entryWitnessCells = entryWitnessCells;
        result.entryWitnesses = entryWitnessCount;
        result.entrySourceOverlapRejections = entrySourceOverlapRejections;
        result.entrySummarySkipped = entrySummarySkipped;
        result.choiceDemandCandidates = choiceDemandCandidates;
        result.choiceDemandWork = choiceDemandWork;
    }

    bool build(const c::Program& p, std::string& reason, const ChoiceIncomingOptions* choiceIncoming = nullptr)
    {
        if (choiceIncoming && choiceIncoming->enabled)
            ++choiceIncoming->passes;
        if (!summarize(p, effects, reason))
            return false;
        parent.assign(p.nodes.size(), NoCut);
        position.assign(p.nodes.size(), NoCut);
        next.assign(p.nodes.size(), NoCut);
        entries.resize(p.nodes.size());
        subtreeNodes.assign(p.nodes.size(), 1);
        laneMask.assign(p.nodes.size(), 0);
        followingLaneMask.assign(p.nodes.size(), 0);
        requests.resize(p.nodes.size());
        capture.resize(p.nodes.size());
        release.resize(p.nodes.size());
        owned.resize(p.nodes.size());
        uint64_t incomingEntryScanWork = 0, incomingEntryStorage = 0;
        auto chargeEntryScans = [&](uint64_t count) {
            if (count > MaxIncomingEntryWork - incomingEntryScanWork) {
                entrySummarySkipped = 1;
                return false;
            }
            incomingEntryScanWork += count;
            entrySummaryScans += count;
            return true;
        };
        auto chargeEntryStorage = [&](uint64_t count) {
            if (count > MaxIncomingEntryWork - incomingEntryStorage) {
                entrySummarySkipped = 1;
                return false;
            }
            incomingEntryStorage += count;
            entryStorageUnits += count;
            return true;
        };
        const bool choiceEnabled = choiceIncoming && choiceIncoming->enabled;
        const uint64_t choiceLimit = choiceEnabled ? std::min(choiceIncoming->limit, MaxChoiceIncomingWork) : 0;
        uint64_t choiceWork = 0;
        bool choiceExhausted = false;
        auto chargeChoiceWork = [&](uint64_t count) {
            if (!choiceEnabled)
                return false;
            if (choiceExhausted || choiceIncoming->exhausted || count > choiceLimit - choiceIncoming->charged) {
                choiceExhausted = true;
                if (!choiceIncoming->exhausted)
                    choiceIncoming->exhaustionPass = choiceIncoming->passes;
                choiceIncoming->exhausted = true;
                return false;
            }
            choiceIncoming->charged += count;
            choiceWork += count;
            choiceDemandWork = choiceWork;
            return true;
        };
        // Mandatory structural metadata is independent of the optional first-
        // consumer population. Build it first so the cheap pre-scan can require
        // the same qualified owner/body shape as demand placement.
        for (unsigned id = 0; id < p.nodes.size(); ++id) {
            const auto& node = p.nodes[id];
            if (node.kind == c::Node::Operation)
                laneMask[id] |= uint8_t(1u << p.nodes[id].lane);
            for (unsigned child : node.children)
                laneMask[id] |= laneMask[child];
        }
        for (unsigned scope = 0; scope < p.nodes.size(); ++scope) {
            const auto& children = p.nodes[scope].children;
            for (unsigned i = 0; i < children.size(); ++i) {
                parent[children[i]] = scope;
                position[children[i]] = i;
                if (i + 1 < children.size())
                    next[children[i]] = children[i + 1];
            }
            for (unsigned child : children)
                subtreeNodes[scope] += subtreeNodes[child];
            if (p.nodes[scope].kind == c::Node::Sequence) {
                uint8_t later = 0;
                for (auto child = children.rbegin(); child != children.rend(); ++child) {
                    followingLaneMask[*child] = later;
                    later |= laneMask[*child];
                }
            }
        }
        bool needsEntryFirst = false;
        if (chargeEntryScans(p.nodes.size())) {
            for (unsigned owner = 0; owner < p.nodes.size() && !needsEntryFirst; ++owner) {
                const auto& node = p.nodes[owner];
                if (node.kind != c::Node::For || node.entryGuardStart == NoCut || node.children.empty() ||
                    parent[owner] == NoCut || p.nodes[parent[owner]].kind != c::Node::Sequence ||
                    next[owner] == NoCut || parent[node.entryGuardStart] != parent[owner] ||
                    position[node.entryGuardStart] > position[owner] ||
                    p.nodes[node.children[0]].kind != c::Node::Sequence)
                    continue;
                needsEntryFirst = std::any_of(
                    p.nodes[node.children[0]].children.begin(), p.nodes[node.children[0]].children.end(),
                    [&](unsigned child) { return p.nodes[child].kind == c::Node::Choice; });
            }
        }
        bool needsChoiceFirst = false;
        if (choiceEnabled && choiceIncoming->commonEnabled && chargeChoiceWork(p.nodes.size()))
            for (unsigned id = 0; id < p.nodes.size() && !needsChoiceFirst; ++id)
                needsChoiceFirst = p.nodes[id].kind == c::Node::Choice && parent[id] != NoCut &&
                                   p.nodes[parent[id]].kind == c::Node::Sequence && position[id] > 0;
        bool hasEntryFirst = false;
        const uint64_t slotWidth = uint64_t(c::LaneCount) * MaxAlternatives;
        const uint64_t summaryWidth = uint64_t(c::LaneCount) * (MaxAlternatives + 3);
        bool summaryFits = p.nodes.size() <= (MaxIncomingEntryWork - 1) / summaryWidth;
        if ((needsEntryFirst || needsChoiceFirst) && summaryFits) {
            uint64_t slots = p.nodes.size() * slotWidth;
            uint64_t storage = p.nodes.size() * summaryWidth + 1;
            bool reserved = needsEntryFirst ? chargeEntryStorage(storage) : chargeChoiceWork(storage);
            if (reserved) {
                if (needsEntryFirst)
                    entrySummarySlots = slots;
                entryFirst.resize(p.nodes.size());
                hasEntryFirst = true;
            }
        }
        if (!needsEntryFirst || !hasEntryFirst)
            entrySummarySkipped = 1;
        // Scan charges are made immediately before the work they cover. If the
        // finite optional allowance is exhausted, discard the incomplete
        // population. Structural order intentionally decides which later
        // proposals are forgone; this is a precision limit, not an admission
        // rule or a claim of globally optimal selection.
        for (unsigned id = 0; id < p.nodes.size() && hasEntryFirst; ++id) {
            const auto& node = p.nodes[id];
            auto chargeSummary = [&](uint64_t count) {
                return needsEntryFirst ? chargeEntryScans(count) : chargeChoiceWork(count);
            };
            if (!chargeSummary(1)) {
                hasEntryFirst = false;
                break;
            }
            if (node.kind == c::Node::Operation) {
                auto& summary = entryFirst[id][node.lane];
                summary.operations[summary.count++] = id;
                summary.mayNoLane = false;
                continue;
            }
            if (node.kind == c::Node::For || node.kind == c::Node::While) {
                if (!chargeSummary(c::LaneCount)) {
                    hasEntryFirst = false;
                    break;
                }
                for (auto& summary : entryFirst[id])
                    summary.valid = false;
                continue;
            }
            if (node.kind == c::Node::Sequence) {
                for (unsigned child : node.children) {
                    for (unsigned observer = 0; observer < c::LaneCount; ++observer) {
                        auto& summary = entryFirst[id][observer];
                        const auto& part = entryFirst[child][observer];
                        if (!summary.mayNoLane)
                            continue;
                        if (!chargeSummary(1 + (part.valid ? part.count : 0))) {
                            hasEntryFirst = false;
                            break;
                        }
                        summary.valid &= part.valid;
                        if (summary.valid)
                            for (unsigned i = 0; i < part.count; ++i)
                                summary.insert(part.operations[i]);
                        summary.mayNoLane &= part.mayNoLane;
                    }
                    if (!hasEntryFirst)
                        break;
                }
                if (!hasEntryFirst)
                    break;
                continue;
            }
            for (unsigned observer = 0; observer < c::LaneCount; ++observer) {
                auto& summary = entryFirst[id][observer];
                summary.mayNoLane = false;
                for (unsigned child : node.children) {
                    const auto& part = entryFirst[child][observer];
                    if (!chargeSummary(1 + (part.valid ? part.count : 0))) {
                        hasEntryFirst = false;
                        break;
                    }
                    summary.valid &= part.valid;
                    if (summary.valid)
                        for (unsigned i = 0; i < part.count; ++i)
                            summary.insert(part.operations[i]);
                    summary.mayNoLane |= part.mayNoLane;
                }
                if (!hasEntryFirst)
                    break;
            }
        }
        if (!hasEntryFirst)
            entryFirst.clear();
        auto reserveIncomingEntry = [&](unsigned root) {
            if (!hasEntryFirst)
                return false;
            if (!chargeEntryScans(c::LaneCount))
                return false;
            for (const auto& summary : entryFirst[root]) {
                if (!summary.valid) {
                    entrySummarySkipped = 1;
                    return false;
                }
            }
            return true;
        };
        // Optional ordinary-Choice requests use the same first-site summary as
        // loop entry, but a separate allowance and an ordinary Every protocol.
        // Build the whole bounded population transactionally before exposing
        // captures to the mandatory demand pass.
        std::vector<ChoiceIncomingOptions::CommonChoiceProposal> choiceProposals;
        if (choiceEnabled && choiceIncoming->commonEnabled && choiceIncoming->commonAnalyzed && needsChoiceFirst &&
            hasEntryFirst && !choiceExhausted) {
            uint64_t copyWork = 0;
            for (const auto& proposal : choiceIncoming->common) {
                uint64_t cells = proposal.demand.cells.size() + proposal.witness.size() + 8;
                if (cells > MaxChoiceIncomingWork - copyWork) {
                    choiceExhausted = true;
                    break;
                }
                copyWork += cells;
            }
            if (!choiceExhausted && chargeChoiceWork(copyWork))
                choiceProposals = choiceIncoming->common;
        }
        if (choiceEnabled && choiceIncoming->commonEnabled && !choiceIncoming->commonAnalyzed && needsChoiceFirst &&
            hasEntryFirst && !choiceExhausted) {
            if (!chargeChoiceWork(p.nodes.size()))
                choiceExhausted = true;
            std::vector<uint8_t> hasRecurrence;
            if (!choiceExhausted)
                hasRecurrence.resize(p.nodes.size());
            for (unsigned id = 0; id < p.nodes.size() && !choiceExhausted; ++id) {
                hasRecurrence[id] = p.nodes[id].kind == c::Node::For || p.nodes[id].kind == c::Node::While;
                for (unsigned nested : p.nodes[id].children)
                    hasRecurrence[id] |= hasRecurrence[nested];
            }
            for (unsigned scope = 0; scope < p.nodes.size() && !choiceExhausted; ++scope) {
                const auto& children = p.nodes[scope].children;
                if (p.nodes[scope].kind != c::Node::Sequence || children.empty())
                    continue;
                uint64_t scopeWork = uint64_t(p.cells) * c::LaneCount * (children.size() + 2);
                if (!chargeChoiceWork(scopeWork))
                    break;
                using Positions = std::array<int, c::LaneCount>;
                Positions missing;
                missing.fill(-1);
                std::vector<Positions> reads(p.cells, missing), writes(p.cells, missing);
                for (unsigned index = 0; index < children.size() && !choiceExhausted; ++index) {
                    unsigned child = children[index];
                    const auto& node = p.nodes[child];
                    if (node.kind == c::Node::Choice && index && !hasRecurrence[child])
                        for (unsigned observer = 0; observer < c::LaneCount && !choiceExhausted; ++observer) {
                            const auto& summary = entryFirst[child][observer];
                            if (!summary.valid || summary.mayNoLane || !summary.count ||
                                summary.count > MaxAlternatives)
                                continue;
                            if (!chargeChoiceWork(uint64_t(summary.count) * p.cells))
                                break;
                            // Every arm must expose the same exact first
                            // observer effect. Do not broaden this witness to
                            // the full Choice MAY summary or merge a later B
                            // frontier into the first A prefix.
                            std::array<c::History, c::MaxCells> projected{};
                            std::array<unsigned, c::MaxCells> firstCells{};
                            unsigned firstCount = 0;
                            bool sameCoverage = true;
                            for (unsigned alternative = 0; alternative < summary.count; ++alternative) {
                                unsigned site = summary.operations[alternative];
                                if (site >= p.nodes.size() || p.nodes[site].kind != c::Node::Operation ||
                                    p.nodes[site].lane != observer) {
                                    sameCoverage = false;
                                    break;
                                }
                                bool nonempty = false;
                                for (unsigned cell = 0; cell < p.cells; ++cell) {
                                    c::History effect;
                                    uint8_t bit = uint8_t(1u << observer);
                                    effect.readers = p.nodes[site].effects[cell].readers & bit;
                                    effect.writers = p.nodes[site].effects[cell].writers & bit;
                                    nonempty |= effect.readers || effect.writers;
                                    if (!alternative) {
                                        projected[cell] = effect;
                                        if (effect.readers || effect.writers)
                                            firstCells[firstCount++] = cell;
                                    } else
                                        sameCoverage &= effect.readers == projected[cell].readers &&
                                                        effect.writers == projected[cell].writers;
                                }
                                sameCoverage &= nonempty;
                            }
                            if (!sameCoverage || !firstCount)
                                continue;
                            for (unsigned source = 0; source < c::LaneCount && !choiceExhausted; ++source) {
                                if (source == observer)
                                    continue;
                                // Charge rejected alternatives and the search
                                // for a demonstrably later source operation
                                // before doing either scan.
                                uint64_t scan = uint64_t(summary.count + 1) * firstCount;
                                scan += uint64_t(index) * p.cells;
                                if (!chargeChoiceWork(scan))
                                    break;
                                int commonLast = -2;
                                bool exactFrontier = true;
                                uint8_t sourceBit = uint8_t(1u << source);
                                for (unsigned alternative = 0; alternative < summary.count; ++alternative) {
                                    int latest = -1;
                                    for (unsigned i = 0; i < firstCount; ++i) {
                                        unsigned cell = firstCells[i];
                                        int previous = writes[cell][source];
                                        if (projected[cell].writers)
                                            previous = std::max(previous, reads[cell][source]);
                                        latest = std::max(latest, previous);
                                    }
                                    if (latest < 0 || (commonLast != -2 && latest != commonLast)) {
                                        exactFrontier = false;
                                        break;
                                    }
                                    commonLast = latest;
                                }
                                // A source generation on a demanded cell in
                                // the Choice is not an incoming first prefix.
                                // Retain the ordinary branch demand instead.
                                for (unsigned i = 0; i < firstCount && exactFrontier; ++i) {
                                    const auto& effect = effects[child][firstCells[i]];
                                    exactFrontier &= !((effect.readers | effect.writers) & sourceBit);
                                }
                                if (!exactFrontier || commonLast < 0 || unsigned(commonLast + 1) >= index)
                                    continue;
                                bool laterSource = false;
                                for (unsigned before = unsigned(commonLast + 1); before < index && !laterSource;
                                     ++before)
                                    for (unsigned cell = 0; cell < p.cells; ++cell) {
                                        const auto& effect = effects[children[before]][cell];
                                        if ((effect.readers | effect.writers) & sourceBit) {
                                            laterSource = true;
                                            break;
                                        }
                                    }
                                if (!laterSource)
                                    continue;
                                constexpr uint64_t ProposalBookkeeping = 12;
                                if (!chargeChoiceWork(uint64_t(p.cells) + firstCount + ProposalBookkeeping))
                                    break;
                                c::CompletionDemand demand{
                                    scope, children[unsigned(commonLast + 1)], child, source, observer, {}};
                                demand.cells.assign(firstCells.begin(), firstCells.begin() + firstCount);
                                c::Effects witness(p.cells);
                                for (unsigned i = 0; i < firstCount; ++i)
                                    witness[firstCells[i]] = projected[firstCells[i]];
                                choiceProposals.push_back({std::move(demand), std::move(witness)});
                            }
                        }
                    for (unsigned cell = 0; cell < p.cells; ++cell)
                        for (unsigned source = 0; source < c::LaneCount; ++source) {
                            if (effects[child][cell].readers & (1u << source))
                                reads[cell][source] = index;
                            if (effects[child][cell].writers & (1u << source))
                                writes[cell][source] = index;
                        }
                }
            }
        }
        if (choiceIncoming && choiceIncoming->commonEnabled && !choiceIncoming->commonAnalyzed) {
            choiceIncoming->commonAnalyzed = true;
            if (!choiceExhausted)
                choiceIncoming->common = choiceProposals;
        }
        if (choiceExhausted)
            choiceProposals.clear();
        else
            for (auto& proposal : choiceProposals) {
                auto key = EntryWitnessKey{proposal.demand.acquisition, proposal.demand.observer};
                choiceWitnesses.try_emplace(key, std::move(proposal.witness));
                capture[proposal.demand.publication].insert(proposal.demand.source);
                requests[proposal.demand.acquisition].push_back(std::move(proposal.demand));
                ++choiceDemandCandidates;
            }
        // The same optional transaction can promote a branch-local ordinary
        // demand to one parent-owned family. Discovery is based on the exact
        // demands of the verified disabled plan; later analysis passes only
        // replay these immutable identities. No command population is edited
        // after numbering.
        if (choiceEnabled && choiceIncoming->alternativesEnabled && choiceIncoming->baselineDemands &&
            !choiceIncoming->alternativesAnalyzed && !choiceExhausted) {
            choiceIncoming->alternativesAnalyzed = true;
            std::vector<ChoiceIncomingOptions::UniversalChoiceFamily> proposals;
            auto contains = [&](unsigned root, unsigned node) {
                for (unsigned at = node; at != NoCut; at = parent[at])
                    if (at == root)
                        return true;
                return false;
            };
            auto relevant = [&](const c::CompletionDemand& demand, const c::Effects& summary) {
                if (demand.acquisition >= p.nodes.size())
                    return false;
                const auto& consumer = p.nodes[demand.acquisition].effects;
                uint8_t source = uint8_t(1u << demand.source);
                for (unsigned cell : demand.cells) {
                    if (cell >= p.cells)
                        return true;
                    bool touches = consumer[cell].readers || consumer[cell].writers;
                    if ((touches && (summary[cell].writers & source)) ||
                        (consumer[cell].writers && (summary[cell].readers & source)))
                        return true;
                }
                return false;
            };
            auto invalidatingPath = [&](const c::CompletionDemand& demand, unsigned choice) {
                unsigned child = demand.acquisition;
                for (unsigned scope = parent[child]; scope != NoCut && child != choice;
                     child = scope, scope = parent[scope]) {
                    if (p.nodes[scope].kind != c::Node::Sequence)
                        continue;
                    for (unsigned i = 0; i < position[child]; ++i)
                        if (relevant(demand, effects[p.nodes[scope].children[i]]))
                            return true;
                }
                return false;
            };
            for (const auto& demand : *choiceIncoming->baselineDemands) {
                // Charge every rejected ancestry/effect scan before performing
                // it. With at most eight retained families/sites, exhausting
                // this shared allowance simply disables the optional proposal.
                if (!chargeChoiceWork(uint64_t(p.nodes.size()) * (uint64_t(p.cells) + 8)))
                    break;
                if (demand.acquisition >= p.nodes.size() || p.nodes[demand.acquisition].kind != c::Node::Operation ||
                    demand.source >= c::LaneCount || demand.observer >= c::LaneCount ||
                    demand.source == demand.observer)
                    continue;
                unsigned choice = NoCut;
                for (unsigned at = parent[demand.acquisition]; at != NoCut; at = parent[at])
                    if (p.nodes[at].kind == c::Node::Choice) {
                        choice = at;
                        break;
                    }
                if (choice == NoCut || !contains(choice, demand.publication) || parent[choice] == NoCut ||
                    p.nodes[parent[choice]].kind != c::Node::Sequence || invalidatingPath(demand, choice))
                    continue;
                unsigned choiceParent = parent[choice];
                const auto& siblings = p.nodes[choiceParent].children;
                int last = -1;
                for (unsigned i = 0; i < position[choice]; ++i)
                    if (relevant(demand, effects[siblings[i]]))
                        last = i;
                if (last < 0)
                    continue;
                uint64_t prefixSteps = 0;
                uint8_t sourceBit = uint8_t(1u << demand.source);
                for (unsigned i = unsigned(last + 1); i < position[choice]; ++i)
                    prefixSteps += std::any_of(
                        effects[siblings[i]].begin(), effects[siblings[i]].end(),
                        [&](const auto& effect) { return (effect.readers | effect.writers) & sourceBit; });
                if (!prefixSteps)
                    continue;
                unsigned publication = siblings[unsigned(last + 1)];
                auto found = std::find_if(proposals.begin(), proposals.end(), [&](const auto& proposal) {
                    return proposal.choice == choice && proposal.parent == choiceParent &&
                           proposal.publication == publication && proposal.source == demand.source &&
                           proposal.observer == demand.observer && proposal.cells == demand.cells;
                });
                if (found == proposals.end()) {
                    if (proposals.size() == MaxAlternatives) {
                        choiceExhausted = true;
                        break;
                    }
                    proposals.push_back(
                        {choice,
                         choiceParent,
                         publication,
                         demand.source,
                         demand.observer,
                         demand.cells,
                         {},
                         prefixSteps});
                    found = std::prev(proposals.end());
                }
                if (found->alternatives.size() == MaxAlternatives) {
                    choiceExhausted = true;
                    break;
                }
                found->alternatives.push_back(demand);
            }
            if (!choiceExhausted)
                proposals.erase(
                    std::remove_if(
                        proposals.begin(), proposals.end(),
                        [&](const auto& proposal) {
                            std::set<unsigned> sites;
                            for (const auto& demand : proposal.alternatives)
                                sites.insert(demand.acquisition);
                            std::function<std::optional<uint8_t>(unsigned)> cardinality =
                                [&](unsigned id) -> std::optional<uint8_t> {
                                if (!chargeChoiceWork(1))
                                    return {};
                                const auto& node = p.nodes[id];
                                if (node.kind == c::Node::For || node.kind == c::Node::While)
                                    return {};
                                unsigned own = sites.count(id) ? 1 : 0;
                                auto add = [](uint8_t left, uint8_t right) {
                                    uint8_t result = 0;
                                    for (unsigned a = 0; a < 3; ++a)
                                        if (left & (1u << a))
                                            for (unsigned b = 0; b < 3; ++b)
                                                if (right & (1u << b))
                                                    result |= uint8_t(1u << std::min(2u, a + b));
                                    return result;
                                };
                                uint8_t children = 1;
                                if (node.kind == c::Node::Choice) {
                                    children = 0;
                                    for (unsigned child : node.children) {
                                        auto part = cardinality(child);
                                        if (!part)
                                            return {};
                                        children |= *part;
                                    }
                                } else
                                    for (unsigned child : node.children) {
                                        auto part = cardinality(child);
                                        if (!part)
                                            return {};
                                        children = add(children, *part);
                                    }
                                return add(uint8_t(1u << own), children);
                            };
                            auto counts = cardinality(proposal.choice);
                            return !counts || *counts != uint8_t(1u << 1) || sites.size() < 2;
                        }),
                    proposals.end());
            if (choiceExhausted)
                proposals.clear();
            choiceIncoming->alternatives = std::move(proposals);
        }
        if (choiceIncoming && choiceIncoming->alternativesEnabled && !choiceIncoming->exhausted)
            universalChoices = choiceIncoming->alternatives;
        for (unsigned scope = 0; scope < p.nodes.size(); ++scope) {
            const auto& children = p.nodes[scope].children;
            if (p.nodes[scope].kind != c::Node::Sequence || children.empty())
                continue;
            using Positions = std::array<int, c::LaneCount>;
            Positions missing;
            missing.fill(-1);
            std::vector<Positions> reads(p.cells, missing), writes(p.cells, missing);
            for (unsigned index = 0; index < children.size(); ++index) {
                unsigned child = children[index];
                const auto& n = p.nodes[child];
                if (n.kind == c::Node::For && n.entryGuardStart != NoCut && next[child] != NoCut &&
                    parent[n.entryGuardStart] == scope && position[n.entryGuardStart] <= index &&
                    p.nodes[n.children[0]].kind == c::Node::Sequence) {
                    std::array<std::vector<bool>, c::LaneCount> seen;
                    for (auto& cells : seen)
                        cells.resize(p.cells);
                    for (unsigned first : p.nodes[n.children[0]].children) {
                        const auto& consumer = p.nodes[first];
                        bool summarized = consumer.kind == c::Node::Operation ||
                                          (consumer.kind == c::Node::Choice && reserveIncomingEntry(first));
                        if (summarized)
                            for (unsigned observer = 0; observer < c::LaneCount; ++observer) {
                                std::array<unsigned, MaxAlternatives> alternatives{};
                                unsigned count = 0;
                                if (consumer.kind == c::Node::Operation) {
                                    if (consumer.lane != observer)
                                        continue;
                                    alternatives[count++] = first;
                                } else {
                                    const auto& summary = entryFirst[first][observer];
                                    count = summary.count;
                                    std::copy_n(summary.operations.begin(), count, alternatives.begin());
                                }
                                if (!count)
                                    continue;
                                uint64_t unionScans = uint64_t(count) * p.cells;
                                if (!chargeEntryScans(unionScans))
                                    break;
                                // Sparse stack storage avoids allocating an
                                // Effects vector for an empty or rejected
                                // observer/source pair. This union is computed
                                // once and shared by every candidate source.
                                std::array<unsigned, c::MaxCells> firstCells{};
                                std::array<c::History, c::MaxCells> firstEffects{};
                                std::array<unsigned, c::MaxCells> witnessCells{};
                                unsigned firstCount = 0, witnessCount = 0;
                                uint8_t observerBit = uint8_t(1u << observer);
                                for (unsigned cell = 0; cell < p.cells; ++cell) {
                                    c::History combined;
                                    for (unsigned i = 0; i < count; ++i) {
                                        const auto& effect = p.nodes[alternatives[i]].effects[cell];
                                        combined.readers |= effect.readers & observerBit;
                                        combined.writers |= effect.writers & observerBit;
                                    }
                                    if (!combined.readers && !combined.writers)
                                        continue;
                                    firstCells[firstCount] = cell;
                                    firstEffects[firstCount++] = combined;
                                    if (!seen[observer][cell])
                                        witnessCells[witnessCount++] = cell;
                                }
                                if (!firstCount || !witnessCount)
                                    continue;
                                for (unsigned source = 0; source < c::LaneCount; ++source) {
                                    if (source == observer)
                                        continue;
                                    ++entryCandidatePairs;
                                    if (!chargeEntryScans(firstCount))
                                        break;
                                    c::CompletionDemand d{scope, n.entryGuardStart, first, source, observer, {}};
                                    uint8_t sourceBit = uint8_t(1u << source);
                                    int last = -1;
                                    bool incomingOnly = true;
                                    for (unsigned i = 0; i < firstCount; ++i) {
                                        unsigned cell = firstCells[i];
                                        const auto& effect = firstEffects[i];
                                        // Do not spend an entry key on only a
                                        // subset of a first consumer's source
                                        // prerequisites. If it also needs a
                                        // fresh in-owner generation, leave
                                        // placement to the ordinary constructor.
                                        incomingOnly &= !(
                                            (effects[child][cell].readers | effects[child][cell].writers) & sourceBit);
                                        if (seen[observer][cell])
                                            continue;
                                        int previous = writes[cell][source];
                                        if (effect.writers)
                                            previous = std::max(previous, reads[cell][source]);
                                        last = std::max(last, previous);
                                    }
                                    if (!incomingOnly) {
                                        ++entrySourceOverlapRejections;
                                        continue;
                                    }
                                    bool newWitness = !entryWitnesses.count({first, observer});
                                    constexpr uint64_t DemandBookkeeping = 8;
                                    constexpr uint64_t WitnessBookkeeping = 4;
                                    uint64_t storage = uint64_t(witnessCount) + DemandBookkeeping;
                                    if (newWitness)
                                        storage += uint64_t(p.cells) + WitnessBookkeeping;
                                    if (!chargeEntryStorage(storage))
                                        break;
                                    d.cells.assign(witnessCells.begin(), witnessCells.begin() + witnessCount);
                                    // Keep the source-prefix identity exact.
                                    // A lexical source uses its latest relevant
                                    // cut (never before guard availability).
                                    // Only a genuinely incoming prefix has no
                                    // lexical source and publishes at the guard.
                                    if (last >= 0)
                                        d.publication =
                                            children[std::max(unsigned(last + 1), position[n.entryGuardStart])];
                                    if (newWitness) {
                                        c::Effects witness(p.cells);
                                        for (unsigned i = 0; i < firstCount; ++i)
                                            if (!seen[observer][firstCells[i]])
                                                witness[firstCells[i]] = firstEffects[i];
                                        entryWitnessCells += p.cells;
                                        ++entryWitnessCount;
                                        entryWitnesses.emplace(EntryWitnessKey{first, observer}, std::move(witness));
                                    }
                                    capture[d.publication].insert(source);
                                    entries[child].push_back(std::move(d));
                                }
                            }
                        for (unsigned observer = 0; observer < c::LaneCount; ++observer)
                            for (unsigned cell = 0; cell < p.cells; ++cell)
                                seen[observer][cell] =
                                    seen[observer][cell] ||
                                    ((effects[first][cell].readers | effects[first][cell].writers) & (1u << observer));
                    }
                }
                if (n.kind == c::Node::Operation)
                    for (unsigned source = 0; source < c::LaneCount; ++source) {
                        if (source == n.lane)
                            continue;
                        c::CompletionDemand demand{scope, children.front(), child, source, n.lane, {}};
                        int last = -1;
                        for (unsigned cell = 0; cell < p.cells; ++cell) {
                            const auto& e = n.effects[cell];
                            if (e.readers || e.writers) {
                                demand.cells.push_back(cell);
                                last = std::max(last, writes[cell][source]);
                            }
                            if (e.writers)
                                last = std::max(last, reads[cell][source]);
                        }
                        if (demand.cells.empty())
                            continue;
                        if (last >= 0)
                            demand.publication = children[last + 1];
                        for (unsigned family = 0; family < universalChoices.size(); ++family) {
                            const auto& proposal = universalChoices[family];
                            auto original = std::find_if(
                                proposal.alternatives.begin(), proposal.alternatives.end(), [&](const auto& old) {
                                    return old.scope == demand.scope && old.publication == demand.publication &&
                                           old.acquisition == demand.acquisition && old.source == demand.source &&
                                           old.observer == demand.observer && old.cells == demand.cells;
                                });
                            if (original == proposal.alternatives.end())
                                continue;
                            demand.publication = proposal.publication;
                            universalChoiceRequests[{
                                demand.scope, demand.publication, demand.acquisition, demand.source, demand.observer,
                                demand.cells}] = family;
                            break;
                        }
                        capture[demand.publication].insert(source);
                        requests[child].push_back(std::move(demand));
                    }
                for (unsigned cell = 0; cell < p.cells; ++cell)
                    for (unsigned source = 0; source < c::LaneCount; ++source) {
                        if (effects[child][cell].readers & (1u << source))
                            reads[cell][source] = index;
                        if (effects[child][cell].writers & (1u << source))
                            writes[cell][source] = index;
                    }
            }
            // Backward continuation demand: keep a prospective prefix only
            // until its last possible consumer in this execution domain. This
            // is independent of which proposals the forward pass will need.
            std::set<PrefixState::EventKey> needed;
            for (auto it = children.rbegin(); it != children.rend(); ++it)
                for (const auto& demand : requests[*it]) {
                    PrefixState::EventKey cut{demand.source, demand.source, demand.publication};
                    if (needed.insert(cut).second)
                        release[*it].push_back(cut);
                }
        }
        // Entry snapshots can be needed inside a later loop. Do not release
        // them at the last ordinary same-scope consumer. The existing bounded
        // proposal cache can still discard one safely before selection.
        std::set<PrefixState::EventKey> entrySnapshots;
        for (const auto& group : entries)
            for (const auto& d : group)
                entrySnapshots.insert({d.source, d.source, d.publication});
        for (auto& group : release)
            group.erase(
                std::remove_if(group.begin(), group.end(), [&](const auto& k) { return entrySnapshots.count(k); }),
                group.end());
        // Find the smallest sequence containing every access to a physical
        // cell. No access outside that domain can create new history for it.
        // The LCA of its first/last DFS accesses contains all intervening ones.
        // This costs O(nodes + cells * depth), with the fixed cell bound.
        std::vector<unsigned> first(p.cells, NoCut), last(p.cells, NoCut), depth(p.nodes.size());
        std::function<void(unsigned)> walk = [&](unsigned id) {
            if (p.nodes[id].kind == c::Node::Sequence)
                scopeOrder.push_back(id);
            if (p.nodes[id].kind == c::Node::Operation)
                for (unsigned cell = 0; cell < p.cells; ++cell)
                    if (p.nodes[id].effects[cell].readers || p.nodes[id].effects[cell].writers) {
                        if (first[cell] == NoCut)
                            first[cell] = id;
                        last[cell] = id;
                    }
            for (unsigned child : p.nodes[id].children) {
                depth[child] = depth[id] + 1;
                walk(child);
            }
        };
        walk(p.nodes.size() - 1);
        for (unsigned cell = 0; cell < p.cells; ++cell) {
            if (first[cell] == NoCut || (!p.globalMemory.empty() && p.globalMemory[cell]))
                continue;
            unsigned a = first[cell], b = last[cell];
            while (a != b) {
                if (depth[a] >= depth[b])
                    a = parent[a];
                else
                    b = parent[b];
            }
            while (a != NoCut && p.nodes[a].kind != c::Node::Sequence)
                a = parent[a];
            if (a != NoCut) {
                unsigned enclosing = parent[a];
                while (enclosing != NoCut && p.nodes[enclosing].kind != c::Node::For &&
                       p.nodes[enclosing].kind != c::Node::While)
                    enclosing = parent[enclosing];
                if (enclosing != NoCut)
                    owned[a].push_back(cell);
            }
        }
        return true;
    }
};

// Verify one closed protocol word and the rearm edges to its next copy. SET
// observes the source prefix but does not gate later source issue. Callers
// validate target directions, keys, and the structural execution domain.
struct ProtocolFacts {
    struct Use {
        uint64_t firstPublish = 0, nextPublish = 0, consumed = 0;
    };
    std::map<PrefixState::EventKey, Use> uses;
    std::vector<PrefixState::EventKey> order;
};
using CausalTransfer = std::array<uint8_t, c::LaneCount>;
struct TransferPoint {
    size_t position = 0;
    CausalTransfer transfer{};
};
struct StructuredProtocolWord {
    std::vector<c::Mechanism> commands;
    std::vector<TransferPoint> transfers;
};

CausalTransfer identityTransfer()
{
    CausalTransfer result{};
    for (unsigned lane = 0; lane < c::LaneCount; ++lane)
        result[lane] = uint8_t(1u << lane);
    return result;
}

bool verifyProtocolWordImpl(
    const std::vector<c::Mechanism>& word, const std::vector<TransferPoint>* transfers, std::string& reason,
    ProtocolFacts* facts)
{
    using Clock = std::array<uint64_t, c::LaneCount>;
    std::array<Clock, c::LaneCount> gates{};
    Clock serial{};
    struct Token {
        Clock prefix{};
        uint64_t consumed = 0;
        bool live = false;
    };
    std::map<PrefixState::EventKey, Token> tokens;
    if (facts)
        *facts = {};
    for (unsigned repetition = 0; repetition < 2; ++repetition) {
        size_t transferIndex = 0;
        for (size_t position = 0; position <= word.size(); ++position) {
            while (transfers && transferIndex < transfers->size() && (*transfers)[transferIndex].position == position) {
                // Every row reads the same pre-transfer lane population. An
                // in-place update would invent ordering between unrelated rows.
                auto before = gates;
                for (unsigned target = 0; target < c::LaneCount; ++target) {
                    Clock joined{};
                    for (unsigned source = 0; source < c::LaneCount; ++source)
                        if ((*transfers)[transferIndex].transfer[target] & (1u << source))
                            for (unsigned lane = 0; lane < c::LaneCount; ++lane)
                                joined[lane] = std::max(joined[lane], before[source][lane]);
                    gates[target] = joined;
                }
                ++transferIndex;
            }
            if (position == word.size())
                break;
            const auto& m = word[position];
            PrefixState::EventKey k{m.first, m.second, m.forwardKey};
            auto& token = tokens[k];
            if (m.kind == c::Mechanism::Publish) {
                if (token.live || gates[m.first][m.second] < token.consumed) {
                    reason = "demand publication can rearm before consumption";
                    return false;
                }
                token.prefix = gates[m.first];
                token.live = true;
                if (facts && m.forwardKey != NoCut) {
                    if (repetition == 0) {
                        if (facts->uses.count(k)) {
                            reason = "logical key has multiple publications";
                            return false;
                        }
                        facts->order.push_back(k);
                        facts->uses[k].firstPublish = gates[m.first][m.second];
                    } else
                        facts->uses[k].nextPublish = gates[m.first][m.second];
                }
            } else {
                if (!token.live) {
                    reason = "demand acquisition has no participating publication";
                    return false;
                }
                auto& target = gates[m.second];
                for (unsigned lane = 0; lane < c::LaneCount; ++lane)
                    target[lane] = std::max(target[lane], token.prefix[lane]);
                target[m.second] = ++serial[m.second];
                token.consumed = target[m.second];
                token.live = false;
                if (facts && m.forwardKey != NoCut && repetition == 0)
                    facts->uses[k].consumed = token.consumed;
            }
        }
        if (transfers && transferIndex != transfers->size()) {
            reason = "child causal transfer is outside its protocol word";
            return false;
        }
        for (const auto& [key, token] : tokens)
            if (token.live) {
                reason = "demand protocol exports an unmatched publication";
                return false;
            }
    }
    return true;
}

bool verifyProtocolWord(const std::vector<c::Mechanism>& word, std::string& reason, ProtocolFacts* facts = nullptr)
{
    return verifyProtocolWordImpl(word, nullptr, reason, facts);
}

bool verifyProtocolWord(const StructuredProtocolWord& word, std::string& reason, ProtocolFacts* facts = nullptr)
{
    return verifyProtocolWordImpl(word.commands, &word.transfers, reason, facts);
}

// A deferred wrap is a different dynamic protocol from the closed per-visit
// ring above.  Its wrap SET executes immediately after the last group.  A
// Previous WAIT consumes that token before each later visit, and one LoopExit
// WAIT consumes the final token iff the counted loop was nonempty.  The shape
// is deliberately limited to the direct body Sequence of a qualified For.
// No predicate is solved here: entryGuardStart is the lowering-owned witness
// that the two fixed guards are available.
struct DeferredDemandRings {
    using Key = PrefixState::EventKey;
    struct Endpoint {
        unsigned cut = NoCut;
        c::Mechanism::Kind kind = c::Mechanism::Publish;
        c::Mechanism::Participation participation = c::Mechanism::Every;
        unsigned source = 0, target = 0, loop = NoCut, word = NoCut;
    };
    struct Span {
        unsigned firstRead = NoCut, lastRead = NoCut;
        unsigned firstWrite = NoCut, lastWrite = NoCut;
    };
    struct SourceReceipt {
        unsigned owner = NoCut;
        unsigned publicationCut = NoCut, publicationIndex = NoCut;
        unsigned previousCut = NoCut, previousIndex = NoCut;
        c::Effects publishedPrefix, currentPrefix, unacquiredSuffix, outsideHistory;
    };
    using CellSpans = std::array<Span, c::LaneCount>;
    Cuts cuts;
    Commands sites;
    std::map<Key, unsigned> owners;
    std::map<Key, SourceReceipt> sourceReceipts;
    std::set<Key> keys;
    std::set<unsigned> loopOwners;
    uint64_t extentWork = 0;
    uint64_t eligibilityWork = 0;
    uint64_t protocolSteps = 0;
    explicit DeferredDemandRings(unsigned size) : sites(size) {}

    // Payload-free synchronization may still relay an exit stall. Summarize
    // actual issuing lanes once, including commands in later nested regions.
    // A certified LoopExit only retires a token; any later publication that
    // could forward its stall is still counted. This is placement eligibility,
    // not a completion or event-consumption proof.
    static std::vector<uint8_t> followingCommands(
        const c::Program& p, const Commands& actual, unsigned retiringOwner = NoCut, unsigned retirementCut = NoCut)
    {
        std::vector<uint8_t> subtree(p.nodes.size()), following(p.nodes.size());
        for (unsigned id = 0; id < p.nodes.size(); ++id) {
            for (const auto& m : actual[id]) {
                if (m.first >= c::LaneCount || m.second >= c::LaneCount) {
                    subtree[id] = UINT8_MAX;
                    continue;
                }
                if (m.participation == c::Mechanism::LoopExit && m.loop == retiringOwner && id == retirementCut)
                    continue;
                if (m.kind == c::Mechanism::Visibility)
                    subtree[id] = UINT8_MAX;
                else if (m.kind == c::Mechanism::Acquire)
                    subtree[id] |= uint8_t(1u << m.second);
                else {
                    subtree[id] |= uint8_t(1u << m.first);
                    if (m.kind == c::Mechanism::Rendezvous)
                        subtree[id] |= uint8_t(1u << m.second);
                }
            }
            for (unsigned child : p.nodes[id].children)
                subtree[id] |= subtree[child];
            if (p.nodes[id].kind == c::Node::Sequence) {
                uint8_t later = 0;
                for (auto child = p.nodes[id].children.rbegin(); child != p.nodes[id].children.rend(); ++child) {
                    following[*child] = later;
                    later |= subtree[*child];
                }
            }
        }
        return following;
    }

    static bool continuationSafe(
        const c::Program& p, const DemandAnalysis& analysis, const Cycle& cycle, const std::vector<uint8_t>& following)
    {
        uint8_t target = uint8_t(1u << cycle.groups.front().lane);
        for (unsigned cursor = cycle.owner; analysis.parent[cursor] != NoCut; cursor = analysis.parent[cursor])
            if (following[cursor] & target)
                return false;
        return true;
    }

    static bool chargeEligibility(const c::Program& p, uint64_t& work)
    {
        // Bound repeated owner/path/physical-cell scans before visiting them.
        const uint64_t factor = 3 * (uint64_t(p.cells) + 1);
        if (p.nodes.size() > (DemandRings::MaxCells - work) / factor)
            return false;
        work += p.nodes.size() * factor;
        return true;
    }

    static bool eligible(
        const c::Program& p, const DemandAnalysis& analysis, const Cycle& cycle, uint64_t* overlapRejections = nullptr)
    {
        if (cycle.owner >= p.nodes.size() || cycle.region >= p.nodes.size())
            return false;
        const auto& owner = p.nodes[cycle.owner];
        bool periodic = owner.children.size() == 1 && owner.children[0] != cycle.region;
        const auto& word = p.nodes[cycle.region];
        if (owner.kind != c::Node::For || owner.children.size() != 1 ||
            p.nodes[cycle.region].kind != c::Node::Sequence || owner.entryGuardStart == NoCut ||
            analysis.next[cycle.owner] == NoCut)
            return false;
        if (periodic && (word.periodicOwner != cycle.owner || !word.firstActive()))
            return false;
        if (periodic) {
            unsigned period = word.periodicPeriod;
            uint32_t all = period == 32 ? UINT32_MAX : (uint32_t(1) << period) - 1;
            std::function<bool(unsigned, uint32_t)> complete = [&](unsigned id, uint32_t active) {
                const auto& node = p.nodes[id];
                if (node.kind == c::Node::For || node.kind == c::Node::While)
                    return false;
                if (node.kind == c::Node::Sequence || node.kind == c::Node::Choice) {
                    if (node.periodicOwner != cycle.owner || node.periodicPeriod != period ||
                        node.periodicLower != word.periodicLower)
                        return false;
                    if (node.kind == c::Node::Sequence && node.periodicResidues != active)
                        return false;
                }
                if (node.kind == c::Node::Choice)
                    return complete(node.children[0], active & node.periodicResidues) &&
                           complete(node.children[1], active & ~node.periodicResidues);
                for (unsigned child : node.children)
                    if (!complete(child, active))
                        return false;
                return true;
            };
            if (!complete(owner.children[0], all))
                return false;
        }
        // The source-prefix receipt below is reconstructed from one linear
        // body word. Empty Sequences represent scalar/terminator cuts; nested
        // control remains on the already verified closed-ring baseline.
        for (unsigned child : p.nodes[cycle.region].children)
            if (p.nodes[child].kind != c::Node::Operation &&
                !(p.nodes[child].kind == c::Node::Sequence && p.nodes[child].children.empty()))
                return false;
        unsigned tail = analysis.next[cycle.groups.back().last.front()];
        if (tail == NoCut || analysis.parent[tail] != cycle.region ||
            analysis.position[cycle.groups.front().first.front()] >= analysis.position[tail])
            return false; // publication moves strictly before the next visit
        if (periodic) {
            // MAY effects outside this complete word remain uncovered by its
            // earlier wrap publication. Reject obvious same-cell source WAW
            // cases up front; this filters a family, never an entire program.
            auto outside = outsideWord(p, analysis, cycle);
            // discoverCuts already requires equal complete access populations
            // for each witness cell in word and owner. Keep the exclusion
            // explicit here: no residue-dependent hidden reader/writer belongs
            // to an internal generation normalized by this certificate.
            for (unsigned cell : cycle.cells)
                if (outside[cell].readers || outside[cell].writers)
                    return false;
            uint8_t source = uint8_t(1u << cycle.groups.back().lane);
            for (unsigned cell = 0; cell < p.cells; ++cell)
                if (outside[cell].writers & analysis.effects[cycle.region][cell].writers & source) {
                    if (overlapRejections)
                        ++*overlapRejections;
                    return false;
                }
        }
        unsigned scope = analysis.parent[cycle.owner];
        if (scope == NoCut || p.nodes[scope].kind != c::Node::Sequence ||
            analysis.parent[owner.entryGuardStart] != scope ||
            analysis.position[owner.entryGuardStart] > analysis.position[cycle.owner])
            return false;
        // LoopExit can stall its first/target lane. Reject if that lane may run
        // later on any continuation to function exit, including after an
        // enclosing Choice. An enclosing loop is rejected wholesale because
        // its backedge may reach earlier first-lane work in the next visit.
        uint8_t target = uint8_t(1u << cycle.groups.front().lane);
        for (unsigned cursor = cycle.owner; analysis.parent[cursor] != NoCut;) {
            unsigned parent = analysis.parent[cursor];
            if (p.nodes[parent].kind == c::Node::For || p.nodes[parent].kind == c::Node::While)
                return false;
            if (p.nodes[parent].kind == c::Node::Sequence && (analysis.followingLaneMask[cursor] & target))
                return false;
            cursor = parent;
        }
        return true;
    }

    static c::Effects outsideWord(const c::Program& p, const DemandAnalysis& analysis, const Cycle& cycle)
    {
        c::Effects result(p.cells);
        for (unsigned cursor = cycle.region; cursor != cycle.owner;) {
            unsigned parent = analysis.parent[cursor];
            if (parent == NoCut)
                return analysis.effects[cycle.owner];
            for (unsigned sibling : p.nodes[parent].children)
                if (sibling != cursor)
                    merge(result, analysis.effects[sibling]);
            cursor = parent;
        }
        return result;
    }

    std::vector<Endpoint> pattern(const Cycle& cycle, const DemandAnalysis& analysis) const
    {
        std::vector<Endpoint> out;
        const auto& groups = cycle.groups;
        for (unsigned g = 0; g < groups.size(); ++g) {
            const auto& group = groups[g];
            const auto& previous = groups[(g + groups.size() - 1) % groups.size()];
            out.push_back(
                {group.first.front(), c::Mechanism::Acquire, g == 0 ? c::Mechanism::Previous : c::Mechanism::Every,
                 previous.lane, group.lane, g == 0 ? cycle.owner : NoCut});
            if (g + 1 < groups.size())
                out.push_back(
                    {cuts.next[group.last.front()], c::Mechanism::Publish, c::Mechanism::Every, group.lane,
                     groups[g + 1].lane, NoCut});
        }
        unsigned last = groups.back().lane, first = groups.front().lane;
        out.push_back(
            {cuts.next[groups.back().last.front()], c::Mechanism::Publish, c::Mechanism::Every, last, first, NoCut});
        out.push_back(
            {analysis.next[cycle.owner], c::Mechanism::Acquire, c::Mechanism::LoopExit, last, first, cycle.owner});
        if (pWord(cycle, analysis)) {
            out.front().word = cycle.region;
            out.back().word = cycle.region;
        }
        return out;
    }

    static bool pWord(const Cycle& cycle, const DemandAnalysis& analysis)
    {
        return analysis.parent[cycle.region] != cycle.owner;
    }

    bool buildSourceReceipts(
        const c::Program& p, const DemandAnalysis& analysis, const Commands& actual, std::string& reason)
    {
        std::map<unsigned, std::vector<CellSpans>> spans;
        for (const auto& cycle : cuts.cycles) {
            if (spans.count(cycle.region))
                continue;
            const auto& children = p.nodes[cycle.region].children;
            uint64_t remaining = DemandRings::MaxCells - extentWork;
            if (p.cells && children.size() > remaining / p.cells) {
                reason = "deferred demand ring source envelope exceeds optional work bound";
                return false;
            }
            extentWork += uint64_t(p.cells) * children.size();
            auto [where, inserted] = spans.emplace(cycle.region, std::vector<CellSpans>(p.cells));
            (void)inserted;
            auto& ownerSpans = where->second;
            for (unsigned position = 0; position < children.size(); ++position) {
                const auto& effects = analysis.effects[children[position]];
                for (unsigned cell = 0; cell < p.cells; ++cell)
                    for (unsigned source = 0; source < c::LaneCount; ++source) {
                        auto& span = ownerSpans[cell][source];
                        if (effects[cell].readers & (1u << source)) {
                            if (span.firstRead == NoCut)
                                span.firstRead = position;
                            span.lastRead = position;
                        }
                        if (effects[cell].writers & (1u << source)) {
                            if (span.firstWrite == NoCut)
                                span.firstWrite = position;
                            span.lastWrite = position;
                        }
                    }
            }
        }
        for (const auto& cycle : cuts.cycles) {
            uint64_t remaining = DemandRings::MaxCells - extentWork;
            if (p.cells > remaining) {
                reason = "deferred demand ring receipt population exceeds optional work bound";
                return false;
            }
            extentWork += p.cells;
            unsigned firstCut = cycle.groups.front().first.front();
            unsigned tailCut = cuts.next[cycle.groups.back().last.front()];
            if (analysis.parent[firstCut] != cycle.region || analysis.parent[tailCut] != cycle.region) {
                reason = "deferred demand ring endpoints are not direct body cuts";
                return false;
            }
            unsigned firstPosition = analysis.position[firstCut];
            unsigned tailPosition = analysis.position[tailCut];
            unsigned source = cycle.groups.back().lane, target = cycle.groups.front().lane;
            SourceReceipt receipt;
            receipt.owner = cycle.owner;
            receipt.publicationCut = tailCut;
            receipt.previousCut = firstCut;
            receipt.publishedPrefix.resize(p.cells);
            receipt.currentPrefix.resize(p.cells);
            receipt.unacquiredSuffix.resize(p.cells);
            const auto& ownerSpans = spans.at(cycle.region);
            // Receipt construction repeats the outside-word traversal after
            // eligibility. Charge that traversal explicitly before doing it;
            // its ancestor-sibling population is at most the original tree.
            if (p.nodes.size() > (DemandRings::MaxCells - extentWork) / p.cells) {
                reason = "deferred outside receipt exceeds optional work bound";
                return false;
            }
            extentWork += uint64_t(p.cells) * p.nodes.size();
            auto outside = outsideWord(p, analysis, cycle);
            receipt.outsideHistory = outside;
            for (unsigned cell = 0; cell < p.cells; ++cell) {
                const auto& span = ownerSpans[cell][source];
                uint8_t sourceBit = uint8_t(1u << source);
                if (span.firstRead != NoCut && span.firstRead < tailPosition)
                    receipt.publishedPrefix[cell].readers |= sourceBit;
                if (span.firstWrite != NoCut && span.firstWrite < tailPosition)
                    receipt.publishedPrefix[cell].writers |= sourceBit;
                if (span.firstRead != NoCut && span.firstRead < firstPosition)
                    receipt.currentPrefix[cell].readers |= sourceBit;
                if (span.firstWrite != NoCut && span.firstWrite < firstPosition)
                    receipt.currentPrefix[cell].writers |= sourceBit;
                if (span.lastRead != NoCut && span.lastRead >= tailPosition)
                    receipt.unacquiredSuffix[cell].readers |= sourceBit;
                if (span.lastWrite != NoCut && span.lastWrite >= tailPosition)
                    receipt.unacquiredSuffix[cell].writers |= sourceBit;
                receipt.unacquiredSuffix[cell].readers |= outside[cell].readers & sourceBit;
                receipt.unacquiredSuffix[cell].writers |= outside[cell].writers & sourceBit;
            }
            Key wrap{source, target, cycle.keys.at({source, target})};
            auto locate = [&](unsigned cut, c::Mechanism::Kind kind, c::Mechanism::Participation participation,
                              unsigned loop) -> std::optional<unsigned> {
                std::optional<unsigned> found;
                for (unsigned index = 0; index < actual[cut].size(); ++index) {
                    const auto& mechanism = actual[cut][index];
                    if (mechanism.kind == kind && mechanism.participation == participation &&
                        mechanism.first == source && mechanism.second == target &&
                        mechanism.forwardKey == std::get<2>(wrap) && mechanism.loop == loop) {
                        if (found)
                            return {};
                        found = index;
                    }
                }
                return found;
            };
            auto publication = locate(tailCut, c::Mechanism::Publish, c::Mechanism::Every, NoCut);
            auto previous = locate(firstCut, c::Mechanism::Acquire, c::Mechanism::Previous, cycle.owner);
            if (!publication || !previous) {
                reason = "deferred demand ring has no exact actual source receipt";
                return false;
            }
            receipt.publicationIndex = *publication;
            receipt.previousIndex = *previous;
            if (!sourceReceipts.emplace(wrap, std::move(receipt)).second) {
                reason = "deferred demand rings share a wrap event key";
                return false;
            }
        }
        return true;
    }

    void appendActual(
        const c::Program& p, std::vector<c::Mechanism>& word, const std::vector<c::Mechanism>& commands, unsigned owner,
        unsigned trips, std::optional<unsigned> iteration) const
    {
        for (const auto& mechanism : commands) {
            bool participate = mechanism.participation == c::Mechanism::Every;
            if (mechanism.loop == owner) {
                unsigned first = 0;
                if (mechanism.word != NoCut) {
                    const auto& domain = p.nodes[mechanism.word];
                    first = unsigned(*domain.firstActive() - domain.periodicLower);
                }
                if (mechanism.participation == c::Mechanism::NonEmpty ||
                    mechanism.participation == c::Mechanism::LoopExit)
                    participate = trips > first;
                else if (mechanism.participation == c::Mechanism::First)
                    participate = iteration && *iteration == 0;
                else if (mechanism.participation == c::Mechanism::Previous)
                    participate = iteration && *iteration != first;
            }
            if (!participate)
                continue;
            if (mechanism.kind == c::Mechanism::Publish || mechanism.kind == c::Mechanism::Acquire)
                word.push_back(mechanism);
            else if (mechanism.kind == c::Mechanism::Rendezvous) {
                word.push_back(event(c::Mechanism::Publish, mechanism.first, mechanism.second, mechanism.forwardKey));
                word.push_back(event(c::Mechanism::Acquire, mechanism.first, mechanism.second, mechanism.forwardKey));
                word.push_back(event(c::Mechanism::Publish, mechanism.second, mechanism.first, mechanism.reverseKey));
                word.push_back(event(c::Mechanism::Acquire, mechanism.second, mechanism.first, mechanism.reverseKey));
            }
        }
    }

    std::vector<c::Mechanism> invocation(
        const c::Program& p, const DemandAnalysis& analysis, const Commands& actual, unsigned owner,
        unsigned trips) const
    {
        std::vector<c::Mechanism> word;
        unsigned scope = analysis.parent[owner];
        appendActual(p, word, actual[scope], owner, trips, std::nullopt);
        for (unsigned child : p.nodes[scope].children) {
            appendActual(p, word, actual[child], owner, trips, std::nullopt);
            if (child != owner)
                continue;
            unsigned body = p.nodes[owner].children[0];
            for (unsigned iteration = 0; iteration < trips; ++iteration) {
                std::function<void(unsigned)> visit = [&](unsigned id) {
                    appendActual(p, word, actual[id], owner, trips, iteration);
                    const auto& node = p.nodes[id];
                    if (node.kind == c::Node::Choice) {
                        bool take = node.periodicResidues & (uint32_t(1) << (iteration % node.periodicPeriod));
                        visit(node.children[take ? 0 : 1]);
                    } else
                        for (unsigned nested : node.children)
                            visit(nested);
                };
                visit(body);
            }
        }
        return word;
    }

    bool verifyCombined(
        const c::Program& p, const DemandAnalysis& analysis, const Commands& actual, std::string& reason)
    {
        // This is a representation-work bound, not a wall-time claim. Cache
        // direct command costs once so many owners sharing one parent cannot
        // turn the optional verifier into an unbounded owner-by-parent scan.
        // Rendezvous expands to four event operations; barriers still cost one
        // because appendActual must inspect them.
        constexpr uint64_t MaxProtocolSteps = 1u << 20;
        std::vector<uint64_t> direct(actual.size()), sequence(actual.size()), subtree(actual.size());
        auto boundedAdd = [](uint64_t& total, uint64_t extra) {
            constexpr uint64_t limit = (1u << 20) + 1;
            total = total >= limit || extra >= limit - total ? limit : total + extra;
        };
        for (unsigned id = 0; id < actual.size(); ++id) {
            for (const auto& mechanism : actual[id])
                boundedAdd(direct[id], commandCost(mechanism));
            subtree[id] = direct[id] + 1; // include original node traversal
            for (unsigned child : p.nodes[id].children)
                boundedAdd(subtree[id], subtree[child]);
            if (p.nodes[id].kind == c::Node::Sequence) {
                sequence[id] = direct[id];
                for (unsigned child : p.nodes[id].children)
                    boundedAdd(sequence[id], direct[child]);
            }
        }
        protocolSteps = 0;
        for (unsigned owner : loopOwners) {
            uint64_t parent = sequence[analysis.parent[owner]];
            unsigned bodyId = p.nodes[owner].children[0];
            uint64_t body = subtree[bodyId];
            uint64_t period = std::max(1u, p.nodes[bodyId].periodicPeriod);
            uint64_t maxTrips = 2 * period;
            uint64_t pairs = (maxTrips + 1) * (maxTrips + 1);
            uint64_t factor = 4 * pairs;
            if (body > (MaxProtocolSteps - protocolSteps) / (factor * maxTrips) ||
                parent > (MaxProtocolSteps - protocolSteps - factor * maxTrips * body) / factor) {
                reason = "deferred demand ring protocol exceeds optional work bound";
                return false;
            }
            // Every first/steady-period exit residue, two owner invocations
            // per pair and the two-copy recurrence check. Period one is the
            // original nine 0/1/2 trip pairs. Charge before materializing words.
            protocolSteps += factor * (parent + maxTrips * body);
        }
        // Check first/steady/final behavior and retain physical token state
        // across two complete owner invocations.  Unlike the ordinary
        // per-Sequence two-copy check, this word contains the actual ordinary
        // mechanisms in their original order as well as deferred endpoints.
        // This is not a finite-enumeration argument for unbounded execution:
        // the first copy establishes initialization/finalization and the
        // second establishes the same consumption-before-next-publication
        // recurrence used by verifyProtocolWord for arbitrary repetitions.
        for (unsigned owner : loopOwners) {
            unsigned maxTrips = 2 * std::max(1u, p.nodes[p.nodes[owner].children[0]].periodicPeriod);
            for (unsigned firstTrips = 0; firstTrips <= maxTrips; ++firstTrips)
                for (unsigned secondTrips = 0; secondTrips <= maxTrips; ++secondTrips) {
                    auto word = invocation(p, analysis, actual, owner, firstTrips);
                    auto second = invocation(p, analysis, actual, owner, secondTrips);
                    word.insert(word.end(), second.begin(), second.end());
                    if (!verifyProtocolWord(word, reason))
                        return false;
                }
        }
        return true;
    }

    bool infer(const c::Program& p, const DemandAnalysis& analysis, const Commands& actual, std::string& reason)
    {
        bool active = false;
        for (const auto& commands : actual)
            active |= std::any_of(commands.begin(), commands.end(), [](const auto& mechanism) {
                return mechanism.participation == c::Mechanism::Previous ||
                       mechanism.participation == c::Mechanism::LoopExit;
            });
        if (!active)
            return true;
        uint64_t commandRecords = p.nodes.size();
        for (const auto& commands : actual) {
            if (commandRecords > DemandRings::MaxCells || commands.size() > DemandRings::MaxCells - commandRecords) {
                reason = "deferred demand ring command population exceeds optional work bound";
                return false;
            }
            commandRecords += commands.size();
        }
        DemandRings shapes(p.nodes.size());
        if (!shapes.discover(p, reason))
            return false;
        cuts = std::move(shapes.cuts);
        using Position = std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned>;
        std::map<Key, std::vector<Position>> population;
        for (unsigned id = 0; id < actual.size(); ++id)
            for (unsigned index = 0; index < actual[id].size(); ++index) {
                const auto& mechanism = actual[id][index];
                if (mechanism.kind == c::Mechanism::Publish || mechanism.kind == c::Mechanism::Acquire)
                    population[{mechanism.first, mechanism.second, mechanism.forwardKey}].push_back(
                        {id, unsigned(mechanism.kind), unsigned(mechanism.participation), mechanism.loop,
                         mechanism.word});
            }
        for (auto& [eventKey, positions] : population) {
            (void)eventKey;
            std::sort(positions.begin(), positions.end());
        }
        std::set<Key> used;
        std::vector<Cycle> certified;
        for (auto cycle : cuts.cycles) {
            if (!chargeEligibility(p, eligibilityWork)) {
                reason = "deferred demand ring eligibility exceeds optional work bound";
                return false;
            }
            if (!eligible(p, analysis, cycle))
                continue;
            std::map<std::pair<unsigned, unsigned>, std::vector<Endpoint>> wanted;
            for (const auto& endpoint : pattern(cycle, analysis))
                wanted[{endpoint.source, endpoint.target}].push_back(endpoint);
            bool found = true;
            for (const auto& [direction, endpoints] : wanted) {
                std::vector<Position> expected;
                for (const auto& endpoint : endpoints)
                    expected.push_back(
                        {endpoint.cut, unsigned(endpoint.kind), unsigned(endpoint.participation), endpoint.loop,
                         endpoint.word});
                std::sort(expected.begin(), expected.end());
                bool matched = false;
                for (unsigned candidate : p.target.compilerKeys) {
                    Key eventKey{direction.first, direction.second, candidate};
                    if (used.count(eventKey) || key(p, direction.first, direction.second) == candidate ||
                        !p.target.available(lane(p, direction.first), lane(p, direction.second), candidate))
                        continue;
                    auto observed = population.find(eventKey);
                    if (observed == population.end() || observed->second != expected)
                        continue;
                    cycle.keys[direction] = candidate;
                    matched = true;
                    break;
                }
                found &= matched;
            }
            if (!found)
                continue;
            unsigned index = certified.size();
            for (const auto& [direction, assigned] : cycle.keys) {
                Key eventKey{direction.first, direction.second, assigned};
                used.insert(eventKey);
                keys.insert(eventKey);
                owners[eventKey] = index;
            }
            loopOwners.insert(cycle.owner);
            certified.push_back(std::move(cycle));
        }
        cuts.cycles = std::move(certified);
        if (!buildSourceReceipts(p, analysis, actual, reason))
            return false;
        for (unsigned index = 0; index < cuts.cycles.size(); ++index)
            for (const auto& endpoint : pattern(cuts.cycles[index], analysis)) {
                auto mechanism = event(
                    endpoint.kind, endpoint.source, endpoint.target,
                    cuts.cycles[index].keys.at({endpoint.source, endpoint.target}));
                mechanism.participation = endpoint.participation;
                mechanism.loop = endpoint.loop;
                mechanism.word = endpoint.word;
                sites[endpoint.cut].push_back(mechanism);
            }
        // Population equality above prevents extra sites for an owned key;
        // this comparison additionally fixes command order at shared cuts.
        for (unsigned id = 0; id < actual.size(); ++id) {
            std::vector<c::Mechanism> observed;
            for (const auto& mechanism : actual[id])
                if (keys.count({mechanism.first, mechanism.second, mechanism.forwardKey}) &&
                    (mechanism.kind == c::Mechanism::Publish || mechanism.kind == c::Mechanism::Acquire))
                    observed.push_back(mechanism);
            if (observed != sites[id]) {
                reason = "deferred demand ring command population or order changed";
                return false;
            }
        }
        for (const auto& commands : actual)
            for (const auto& mechanism : commands)
                if (mechanism.kind == c::Mechanism::Rendezvous &&
                    (keys.count({mechanism.first, mechanism.second, mechanism.forwardKey}) ||
                     keys.count({mechanism.second, mechanism.first, mechanism.reverseKey}))) {
                    reason = "canonical packet collides with a deferred demand ring";
                    return false;
                }
        for (unsigned owner : loopOwners) {
            if (commandRecords > DemandRings::MaxCells - eligibilityWork) {
                reason = "deferred demand ring continuation exceeds optional work bound";
                return false;
            }
            eligibilityWork += commandRecords;
            // One owner's adjacent final waits form its retirement packet.
            // Do not exempt another owner's cleanup from continuation analysis.
            unsigned exit = analysis.next[owner], prefix = 0;
            for (const auto& m : actual[exit]) {
                if (m.kind != c::Mechanism::Acquire || m.participation != c::Mechanism::LoopExit || m.loop != owner)
                    break;
                ++prefix;
            }
            if (prefix != std::count_if(cuts.cycles.begin(), cuts.cycles.end(), [&](const auto& cycle) {
                    return cycle.owner == owner;
                })) {
                reason = "deferred retirement packet is not an adjacent immediate-exit prefix";
                return false;
            }
            auto following = followingCommands(p, actual, owner, exit);
            for (const auto& cycle : cuts.cycles)
                if (cycle.owner == owner && !continuationSafe(p, analysis, cycle, following)) {
                    reason = "deferred loop exit may stall a later synchronization command";
                    return false;
                }
        }
        return true;
    }

    bool credit(
        const c::Mechanism& mechanism, PrefixState& state, const std::map<unsigned, c::State>& incoming,
        std::string& reason) const
    {
        if (mechanism.kind != c::Mechanism::Acquire || mechanism.participation == c::Mechanism::LoopExit)
            return true;
        auto found = owners.find({mechanism.first, mechanism.second, mechanism.forwardKey});
        if (found == owners.end())
            return true;
        const auto& cycle = cuts.cycles[found->second];
        auto entry = incoming.find(cycle.owner);
        if (entry == incoming.end()) {
            reason = "deferred demand ring has no whole-owner entry state";
            return false;
        }
        if (mechanism.participation == c::Mechanism::Previous) {
            auto receipt = sourceReceipts.find({mechanism.first, mechanism.second, mechanism.forwardKey});
            if (receipt == sourceReceipts.end() || receipt->second.owner != cycle.owner) {
                reason = "deferred demand ring has no typed source receipt";
                return false;
            }
            uint8_t sourceBit = uint8_t(1u << mechanism.first);
            for (unsigned cell = 0; cell < state.pending[mechanism.second].size(); ++cell) {
                auto& pending = state.pending[mechanism.second][cell];
                const auto& external = entry->second.pending[mechanism.second][cell];
                const auto& prefix = receipt->second.publishedPrefix[cell];
                const auto& current = receipt->second.currentPrefix[cell];
                const auto& suffix = receipt->second.unacquiredSuffix[cell];
                if ((pending.readers & sourceBit &
                     ~(external.readers | prefix.readers | current.readers | suffix.readers)) ||
                    (pending.writers & sourceBit &
                     ~(external.writers | prefix.writers | current.writers | suffix.writers))) {
                    reason = "deferred demand ring source prefix omits physical history";
                    return false;
                }
                // The exact actual SET certifies the complete source prefix I.
                // The first visit has no old generation; steady visits consume
                // I through the actual Previous WAIT. In both cases retain E,
                // source work U after the SET, and current-visit source work C
                // before the WAIT. Clear only this source bit; other lanes and
                // the independent GM-visibility state remain unchanged.
                pending.readers &= uint8_t(~sourceBit | external.readers | current.readers | suffix.readers);
                pending.writers &= uint8_t(~sourceBit | external.writers | current.writers | suffix.writers);
            }
        }
        // This is generation normalization, not credit from a first-visit
        // Previous WAIT (which does not execute). At the certified first cut,
        // no demanded-cell MAY effect precedes the first group, so the initial
        // synthetic loop seed may be intersected with E. On later visits the
        // same intersection is justified by the actual Previous WAIT.
        for (unsigned cell : cycle.cells) {
            auto& pending = state.pending[mechanism.second][cell];
            const auto& external = entry->second.pending[mechanism.second][cell];
            const auto& internal = cuts.effects[cycle.region][cell];
            unsigned source = cycle.groups.back().lane, target = cycle.groups.front().lane;
            const auto& outside =
                sourceReceipts.at({source, target, cycle.keys.at({source, target})}).outsideHistory[cell];
            if ((pending.readers & ~(external.readers | internal.readers | outside.readers)) ||
                (pending.writers & ~(external.writers | internal.writers | outside.writers))) {
                reason = "deferred demand ring omits external physical history";
                return false;
            }
            pending.readers &= external.readers | outside.readers;
            pending.writers &= external.writers | outside.writers;
        }
        return true;
    }
};

void appendProtocolCommands(
    const std::vector<c::Mechanism>& source, std::vector<c::Mechanism>& word, bool expandPackets, bool virtualPackets,
    const std::set<PrefixState::EventKey>* excluded)
{
    for (const auto& m : source)
        if (m.participation != c::Mechanism::Every)
            continue;
        else if (
            excluded && (excluded->count({m.first, m.second, m.forwardKey}) ||
                         (m.kind == c::Mechanism::Rendezvous && excluded->count({m.second, m.first, m.reverseKey}))))
            continue;
        else if (m.kind == c::Mechanism::Publish || m.kind == c::Mechanism::Acquire)
            word.push_back(m);
        else if (expandPackets && m.kind == c::Mechanism::Rendezvous) {
            // Canonical packets participate in causality but are not logical
            // colors. Virtual IDs remain disjoint from raw logical events.
            auto forward = virtualPackets ? NoCut : m.forwardKey;
            auto reverse = virtualPackets ? NoCut : m.reverseKey;
            word.push_back(event(c::Mechanism::Publish, m.first, m.second, forward));
            word.push_back(event(c::Mechanism::Acquire, m.first, m.second, forward));
            word.push_back(event(c::Mechanism::Publish, m.second, m.first, reverse));
            word.push_back(event(c::Mechanism::Acquire, m.second, m.first, reverse));
        }
}

uint64_t protocolCommandCount(
    const std::vector<c::Mechanism>& source, bool expandPackets, const std::set<PrefixState::EventKey>* excluded)
{
    uint64_t count = 0;
    for (const auto& m : source) {
        if (m.participation != c::Mechanism::Every)
            continue;
        if (excluded && (excluded->count({m.first, m.second, m.forwardKey}) ||
                         (m.kind == c::Mechanism::Rendezvous && excluded->count({m.second, m.first, m.reverseKey}))))
            continue;
        if (m.kind == c::Mechanism::Publish || m.kind == c::Mechanism::Acquire)
            ++count;
        else if (expandPackets && m.kind == c::Mechanism::Rendezvous)
            count += 4;
    }
    return count;
}

std::vector<c::Mechanism> demandWord(
    const c::Program& p, const Commands& commands, unsigned scope, bool expandPackets = false,
    bool virtualPackets = false, const std::set<PrefixState::EventKey>* excluded = nullptr)
{
    std::vector<c::Mechanism> word;
    for (unsigned child : p.nodes[scope].children)
        appendProtocolCommands(commands[child], word, expandPackets, virtualPackets, excluded);
    return word;
}

bool exportCausalTransfer(const StructuredProtocolWord& word, CausalTransfer& transfer, std::string& reason)
{
    std::array<uint8_t, c::LaneCount> gates = identityTransfer();
    struct Token {
        uint8_t prefix = 0;
        bool live = false;
    };
    std::map<PrefixState::EventKey, Token> tokens;
    size_t transferIndex = 0;
    for (size_t position = 0; position <= word.commands.size(); ++position) {
        while (transferIndex < word.transfers.size() && word.transfers[transferIndex].position == position) {
            auto before = gates;
            for (unsigned target = 0; target < c::LaneCount; ++target) {
                uint8_t joined = 0;
                for (unsigned source = 0; source < c::LaneCount; ++source)
                    if (word.transfers[transferIndex].transfer[target] & (1u << source))
                        joined |= before[source];
                gates[target] = joined;
            }
            ++transferIndex;
        }
        if (position == word.commands.size())
            break;
        const auto& mechanism = word.commands[position];
        PrefixState::EventKey key{mechanism.first, mechanism.second, mechanism.forwardKey};
        auto& token = tokens[key];
        if (mechanism.kind == c::Mechanism::Publish) {
            if (token.live) {
                reason = "child causal summary has an overlapping publication";
                return false;
            }
            token.prefix = gates[mechanism.first];
            token.live = true;
        } else {
            if (!token.live) {
                reason = "child causal summary has an unmatched acquisition";
                return false;
            }
            gates[mechanism.second] |= token.prefix;
            token.live = false;
        }
    }
    if (transferIndex != word.transfers.size()) {
        reason = "child causal summary transfer is outside its word";
        return false;
    }
    for (const auto& [key, token] : tokens)
        if (token.live) {
            (void)key;
            reason = "child causal summary exports a live publication";
            return false;
        }
    transfer = gates;
    return true;
}

struct ChildReturnSummaries {
    static constexpr uint64_t MaxWork = 1u << 22;
    std::vector<std::optional<CausalTransfer>> body;
    uint64_t work = 0;
    bool exhausted = false;

    bool reserve(uint64_t count, uint64_t width, uint64_t limit)
    {
        if (exhausted || work > limit || (width && count > (limit - work) / width)) {
            exhausted = true;
            return false;
        }
        work += count * width;
        return true;
    }

    bool build(
        const c::Program& p, const Commands& commands, bool expandPackets, bool virtualPackets,
        const std::set<PrefixState::EventKey>* excluded, uint64_t limit, std::string& reason,
        const std::map<unsigned, CausalTransfer>* certifiedChoiceReturns = nullptr)
    {
        limit = std::min(limit, MaxWork);
        // Reserve the optional summary population before allocating it. Work
        // below is representation work, not a wall-clock estimate.
        if (!reserve(p.nodes.size(), c::LaneCount + 4, limit))
            return false;
        body.resize(p.nodes.size());
        auto appendChild = [&](StructuredProtocolWord& word, unsigned child) {
            // Charge the complete input group before scanning it, then the
            // worst-case expanded storage before copying any command.
            if (!reserve(commands[child].size(), 4, limit))
                return false;
            uint64_t emitted = protocolCommandCount(commands[child], expandPackets, excluded);
            if (!reserve(emitted, 4 * c::LaneCount + 12, limit))
                return false;
            if (!body[child])
                return false;
            appendProtocolCommands(commands[child], word.commands, expandPackets, virtualPackets, excluded);
            if (*body[child] != identityTransfer()) {
                if (!reserve(1, c::LaneCount + 4, limit))
                    return false;
                word.transfers.push_back({word.commands.size(), *body[child]});
            }
            return true;
        };
        auto proveAndExport = [&](const StructuredProtocolWord& word, CausalTransfer& transfer) {
            // Two numeric copies establish closed rearm. The separate pure
            // symbolic pass applies every 7x7 transfer once and exports only
            // the first-copy lane-origin transform. It starts from identity;
            // no numeric clock or live token is imported at child entry.
            constexpr uint64_t matrix = uint64_t(c::LaneCount) * c::LaneCount;
            constexpr uint64_t numericMatrix = matrix * c::LaneCount;
            if (!reserve(word.commands.size(), 3 * c::LaneCount + 12, limit) ||
                !reserve(word.transfers.size(), 2 * numericMatrix + matrix + 6, limit))
                return false;
            return verifyProtocolWord(word, reason) && exportCausalTransfer(word, transfer, reason);
        };
        for (unsigned id = 0; id < p.nodes.size(); ++id) {
            const auto& node = p.nodes[id];
            if (node.kind == c::Node::Operation || node.kind == c::Node::For || node.kind == c::Node::While) {
                body[id] = identityTransfer();
                continue;
            }
            if (node.kind == c::Node::Choice) {
                std::optional<CausalTransfer> common;
                for (unsigned arm : node.children) {
                    StructuredProtocolWord word;
                    if (!appendChild(word, arm))
                        return false;
                    CausalTransfer transfer;
                    if (!proveAndExport(word, transfer))
                        return false;
                    if (!common)
                        common = transfer;
                    else
                        for (unsigned lane = 0; lane < c::LaneCount; ++lane)
                            (*common)[lane] &= transfer[lane];
                }
                if (!common)
                    return false;
                if (certifiedChoiceReturns) {
                    auto transfer = certifiedChoiceReturns->find(id);
                    if (transfer != certifiedChoiceReturns->end())
                        for (unsigned lane = 0; lane < c::LaneCount; ++lane)
                            (*common)[lane] |= transfer->second[lane];
                }
                body[id] = *common;
                continue;
            }
            StructuredProtocolWord word;
            for (unsigned child : node.children)
                if (!appendChild(word, child))
                    return false;
            CausalTransfer transfer;
            if (!proveAndExport(word, transfer))
                return false;
            body[id] = transfer;
        }
        return true;
    }

    std::optional<StructuredProtocolWord> word(
        const c::Program& p, const Commands& commands, unsigned scope, bool expandPackets, bool virtualPackets,
        const std::set<PrefixState::EventKey>* excluded, uint64_t limit)
    {
        StructuredProtocolWord result;
        for (unsigned child : p.nodes[scope].children) {
            if (!reserve(commands[child].size(), 4, limit))
                return {};
            uint64_t emitted = protocolCommandCount(commands[child], expandPackets, excluded);
            if (!reserve(emitted, 4 * c::LaneCount + 12, limit) || !body[child])
                return {};
            appendProtocolCommands(commands[child], result.commands, expandPackets, virtualPackets, excluded);
            if (*body[child] != identityTransfer()) {
                if (!reserve(1, c::LaneCount + 4, limit))
                    return {};
                result.transfers.push_back({result.commands.size(), *body[child]});
            }
        }
        return result;
    }
};

struct ChildReturnOptions {
    bool enabled = false, corrupt = false;
    uint64_t limit = ChildReturnSummaries::MaxWork;
    mutable uint64_t charged = 0;
    mutable uint64_t candidates = 0, rejected = 0;
    mutable uint64_t checks = 0, budgetCheck = 0;
    mutable bool exhausted = false;

    bool reserve(uint64_t count, uint64_t width = 1) const
    {
        uint64_t allowance = std::min(limit, ChildReturnSummaries::MaxWork);
        if (exhausted || charged > allowance || (width && count > (allowance - charged) / width)) {
            exhausted = true;
            return false;
        }
        charged += count * width;
        return true;
    }
};

// The checker derives closed protocol words from actual IR, not from demands
// or constructor families. Each raw key has matched SET/WAIT uses in one
// structural execution domain and is exclusive to it. Two copies establish
// every consumption -> next publication edge in the periodic word; translating
// those edges proves arbitrary repetitions and skips of the whole domain.
// Nested domains use disjoint keys and need not finish their payload on return.
bool verifyFirstCardinality(
    const c::Program& p, const DemandAnalysis& analysis, const Commands& actual, const PrefixState::EventKey& key,
    unsigned loop, unsigned origin, const std::vector<unsigned>& actualSites, std::string& reason)
{
    using namespace c;
    if (actualSites.size() == 1 && actualSites.front() == origin)
        return true;
    const auto* summary = analysis.firstSummary(origin, std::get<1>(key));
    if (origin >= p.nodes.size() || p.nodes[origin].kind != Node::Choice || !summary || !summary->valid ||
        summary->mayNoLane || !summary->count || summary->count > MaxAlternatives) {
        reason = "late first-consumer wait has no total bounded Choice summary";
        return false;
    }
    std::vector<unsigned> expected(summary->operations.begin(), summary->operations.begin() + summary->count);
    auto observed = actualSites;
    std::sort(expected.begin(), expected.end());
    std::sort(observed.begin(), observed.end());
    if (expected != observed) {
        reason = "late first-consumer wait population differs from original first sites";
        return false;
    }
    uint8_t observerBit = uint8_t(1u << std::get<1>(key));
    for (unsigned site : expected)
        if (std::none_of(p.nodes[site].effects.begin(), p.nodes[site].effects.end(), [&](const auto& effect) {
                return (effect.readers | effect.writers) & observerBit;
            })) {
            reason = "late first-consumer site has no observer effect witness";
            return false;
        }
    // Bits denote possible path states, not a Boolean "some path consumed".
    // A mixed Choice result is 1|2 == 3 and cannot satisfy the final exact
    // singleton Consumed check. It also cannot execute another matching WAIT.
    constexpr uint8_t Unconsumed = 1, Consumed = 2;
    auto matching = [&](const Mechanism& mechanism) {
        return mechanism.kind == Mechanism::Acquire && mechanism.participation == Mechanism::First &&
               mechanism.loop == loop && mechanism.first == std::get<0>(key) && mechanism.second == std::get<1>(key) &&
               mechanism.forwardKey == std::get<2>(key);
    };
    std::function<bool(unsigned, uint8_t, uint8_t&)> walk = [&](unsigned id, uint8_t state, uint8_t& out) {
        unsigned waits = std::count_if(actual[id].begin(), actual[id].end(), matching);
        if (waits) {
            if (waits != 1 || (state & Consumed) || !std::binary_search(expected.begin(), expected.end(), id))
                return false;
            state = Consumed;
        }
        const auto& node = p.nodes[id];
        if (node.kind == Node::Operation && node.lane == std::get<1>(key) && state != Consumed)
            return false;
        if (node.kind == Node::Choice) {
            uint8_t left = 0, right = 0;
            if (!walk(node.children[0], state, left) || !walk(node.children[1], state, right))
                return false;
            out = left | right;
            return true;
        }
        if (node.kind == Node::For || node.kind == Node::While) {
            // Even a recurrence lexically after a first site is outside this
            // finite cardinality certificate. Keep the common-cut baseline.
            return false;
        }
        for (unsigned child : node.children) {
            uint8_t next = 0;
            if (!walk(child, state, next))
                return false;
            state = next;
        }
        out = state;
        return true;
    };
    uint8_t outcome = 0;
    if (!walk(origin, Unconsumed, outcome) || outcome != Consumed) {
        reason = "late first-consumer waits are not exactly once on every Choice path";
        return false;
    }
    return true;
}

bool projectEntryWord(
    const c::Program& p, const Commands& actual, unsigned id, unsigned loop, std::vector<c::Mechanism>& word,
    std::string& reason)
{
    using namespace c;
    std::vector<Mechanism> direct;
    for (const auto& mechanism : actual[id])
        if ((mechanism.participation == Mechanism::First || mechanism.participation == Mechanism::NonEmpty) &&
            mechanism.loop == loop)
            direct.push_back(mechanism);
    const auto& node = p.nodes[id];
    if (node.kind == Node::Choice) {
        std::vector<Mechanism> left, right;
        if (!projectEntryWord(p, actual, node.children[0], loop, left, reason) ||
            !projectEntryWord(p, actual, node.children[1], loop, right, reason))
            return false;
        if (left != right) {
            reason = "entry episode command order differs across Choice paths";
            return false;
        }
        word.insert(word.end(), direct.begin(), direct.end());
        word.insert(word.end(), left.begin(), left.end());
        return true;
    }
    if (node.kind == Node::For || node.kind == Node::While) {
        std::vector<Mechanism> nested;
        for (unsigned child : node.children)
            if (!projectEntryWord(p, actual, child, loop, nested, reason))
                return false;
        if (!nested.empty()) {
            reason = "entry episode command is nested in an unknown recurrence";
            return false;
        }
        word.insert(word.end(), direct.begin(), direct.end());
        return true;
    }
    word.insert(word.end(), direct.begin(), direct.end());
    for (unsigned child : node.children)
        if (!projectEntryWord(p, actual, child, loop, word, reason))
            return false;
    return true;
}

bool verifyEntryProtocols(
    const c::Program& p, const DemandAnalysis& analysis, const Commands& actual,
    const std::set<PrefixState::EventKey>& deferredKeys, std::set<PrefixState::EventKey>& reserved, std::string& reason)
{
    using namespace c;
    using Key = PrefixState::EventKey;
    struct Protocol {
        unsigned loop = NoCut, provider = NoCut, first = NoCut, firstSite = NoCut, ackSet = NoCut, ackWait = NoCut;
        std::vector<unsigned> additionalFirstSites;
    };
    std::map<Key, Protocol> population;
    std::map<unsigned, std::set<Key>> firstKeys, replyKeys;
    auto refuse = [&]() {
        reason = "invalid first-consumer entry episode or participation";
        return false;
    };
    for (unsigned id = 0; id < actual.size(); ++id)
        for (unsigned index = 0; index < actual[id].size(); ++index) {
            const auto& m = actual[id][index];
            if (m.participation == Mechanism::Every) {
                if (m.loop != NoCut || m.word != NoCut)
                    return refuse();
                continue;
            }
            Key mechanismKey{m.first, m.second, m.forwardKey};
            if (m.participation == Mechanism::Previous || m.participation == Mechanism::LoopExit) {
                if (!deferredKeys.count(mechanismKey))
                    return refuse();
                continue;
            }
            if (m.word != NoCut)
                return refuse();
            if (m.loop >= p.nodes.size() || p.nodes[m.loop].kind != Node::For ||
                p.nodes[m.loop].entryGuardStart == NoCut || analysis.next[m.loop] == NoCut || m.first >= LaneCount ||
                m.second >= LaneCount || !p.target.event(lane(p, m.first), lane(p, m.second)) ||
                !p.target.available(lane(p, m.first), lane(p, m.second), m.forwardKey) ||
                key(p, m.first, m.second) == m.forwardKey ||
                (m.kind != Mechanism::Publish && m.kind != Mechanism::Acquire))
                return refuse();
            Key key{m.first, m.second, m.forwardKey};
            auto& pop = population[key];
            if (pop.loop != NoCut && pop.loop != m.loop)
                return refuse();
            pop.loop = m.loop;
            auto scope = analysis.parent[m.loop];
            if (scope == NoCut || p.nodes[scope].kind != Node::Sequence ||
                analysis.parent[p.nodes[m.loop].entryGuardStart] != scope)
                return refuse();
            if (m.participation == Mechanism::First) {
                if (m.kind != Mechanism::Acquire)
                    return refuse();
                const auto* demand = analysis.entryDemand(m.loop, id, m.first, m.second);
                bool late = false;
                if (!demand) {
                    demand = analysis.lateEntryDemand(m.loop, id, m.first, m.second);
                    late = true;
                }
                if (!demand || (!late && analysis.parent[id] != p.nodes[m.loop].children[0]) ||
                    (late && (p.nodes[demand->acquisition].kind != Node::Choice ||
                              p.nodes[id].kind != Node::Operation || p.nodes[id].lane != m.second)))
                    return refuse();
                const auto* witness = analysis.entryWitness(*demand);
                if (!witness || (!late && p.nodes[id].kind == Node::Operation && p.nodes[id].lane != m.second) ||
                    (!late && p.nodes[id].kind != Node::Operation && p.nodes[id].kind != Node::Choice))
                    return refuse();
                for (unsigned cell : demand->cells)
                    if (((*witness)[cell].readers | (*witness)[cell].writers) & (1u << m.second)) {
                        if ((analysis.effects[m.loop][cell].readers | analysis.effects[m.loop][cell].writers) &
                            (1u << m.first))
                            return refuse();
                    } else
                        return refuse();
                if (pop.first != NoCut && pop.first != demand->acquisition)
                    return refuse();
                pop.first = demand->acquisition;
                if (pop.firstSite == NoCut)
                    pop.firstSite = id;
                else {
                    // Reject an oversized actual population before growing
                    // storage. A valid exclusive family has at most eight
                    // sites, including the first one retained inline.
                    if (pop.additionalFirstSites.size() >= MaxAlternatives - 1)
                        return refuse();
                    pop.additionalFirstSites.push_back(id);
                }
                firstKeys[m.loop].insert(key);
            } else if (m.participation == Mechanism::NonEmpty) {
                if (analysis.parent[id] != scope)
                    return refuse();
                if (analysis.position[id] <= analysis.position[m.loop]) {
                    if (m.kind != Mechanism::Publish ||
                        analysis.position[id] < analysis.position[p.nodes[m.loop].entryGuardStart])
                        return refuse();
                    if (pop.provider != NoCut)
                        return refuse();
                    pop.provider = id;
                } else {
                    if (id != analysis.next[m.loop])
                        return refuse();
                    if (m.kind == Mechanism::Publish) {
                        if (index + 1 == actual[id].size())
                            return refuse();
                        const auto& wait = actual[id][index + 1];
                        if (wait.kind != Mechanism::Acquire || wait.participation != m.participation ||
                            wait.loop != m.loop || wait.first != m.first || wait.second != m.second ||
                            wait.forwardKey != m.forwardKey)
                            return refuse();
                        if (pop.ackSet != NoCut)
                            return refuse();
                        pop.ackSet = id;
                        replyKeys[m.loop].insert(key);
                    } else {
                        if (index == 0)
                            return refuse();
                        const auto& set = actual[id][index - 1];
                        if (set.kind != Mechanism::Publish || set.participation != m.participation ||
                            set.loop != m.loop || set.first != m.first || set.second != m.second ||
                            set.forwardKey != m.forwardKey)
                            return refuse();
                        if (pop.ackWait != NoCut)
                            return refuse();
                        pop.ackWait = id;
                    }
                }
            } else
                return refuse();
        }
    std::set<Key> lateKeys;
    std::set<unsigned> lateLoops;
    for (const auto& [key, pop] : population)
        if (pop.provider != NoCut && (pop.firstSite != pop.first || !pop.additionalFirstSites.empty())) {
            lateKeys.insert(key);
            lateLoops.insert(pop.loop);
        }
    constexpr uint64_t MaxEntryProtocolWork = 1u << 20;
    uint64_t protocolWork = 0;
    std::vector<uint64_t> subtreeRecords;
    auto chargeProtocol = [&](uint64_t count) {
        if (count > MaxEntryProtocolWork - protocolWork) {
            reason = "late entry episode verification exceeds optional work bound";
            return false;
        }
        protocolWork += count;
        return true;
    };
    if (!lateKeys.empty()) {
        if (!chargeProtocol(actual.size()))
            return false;
        subtreeRecords.resize(actual.size());
        for (unsigned id = 0; id < actual.size(); ++id) {
            if (!chargeProtocol(actual[id].size()))
                return false;
            subtreeRecords[id] = 1 + actual[id].size();
            for (unsigned child : p.nodes[id].children) {
                if (subtreeRecords[child] > MaxEntryProtocolWork - subtreeRecords[id])
                    subtreeRecords[id] = MaxEntryProtocolWork;
                else
                    subtreeRecords[id] += subtreeRecords[child];
            }
        }
    }
    for (const auto& [k, pop] : population) {
        if (pop.loop == NoCut)
            return refuse();
        if (pop.provider != NoCut) {
            if (pop.first == NoCut || pop.ackSet != NoCut || pop.ackWait != NoCut)
                return refuse();
            if (lateKeys.count(k)) {
                if (!chargeProtocol(subtreeRecords[pop.first]))
                    return false;
                std::vector<unsigned> firstSites{pop.firstSite};
                firstSites.insert(firstSites.end(), pop.additionalFirstSites.begin(), pop.additionalFirstSites.end());
                uint64_t effectScans = uint64_t(firstSites.size()) * p.cells;
                if (!chargeProtocol(effectScans) ||
                    !verifyFirstCardinality(p, analysis, actual, k, pop.loop, pop.first, firstSites, reason))
                    return false;
            }
        } else if (pop.ackSet == NoCut || pop.ackWait == NoCut || pop.first != NoCut)
            return refuse();
        reserved.insert(k);
    }
    std::set<unsigned> loops;
    for (const auto& [key, pop] : population)
        loops.insert(pop.loop);
    for (unsigned loop : loops) {
        if (firstKeys[loop].empty() || replyKeys[loop].empty())
            return refuse();
        std::vector<Mechanism> word;
        auto append = [&](unsigned cut) {
            for (const auto& m : actual[cut])
                if (m.participation != Mechanism::Every && m.loop == loop)
                    word.push_back(m);
        };
        auto appendLate = [&](unsigned cut) {
            for (const auto& m : actual[cut])
                if ((m.participation == Mechanism::First || m.participation == Mechanism::NonEmpty) && m.loop == loop)
                    word.push_back(m);
        };
        bool late = lateLoops.count(loop);
        const auto& siblings = p.nodes[analysis.parent[loop]].children;
        for (unsigned i = 0; i <= analysis.position[loop]; ++i) {
            if (late)
                appendLate(siblings[i]);
            else
                append(siblings[i]);
        }
        unsigned body = p.nodes[loop].children[0];
        if (late) {
            uint64_t depthProduct = subtreeRecords[body];
            if (analysis.subtreeNodes[body] && depthProduct > MaxEntryProtocolWork / analysis.subtreeNodes[body])
                return refuse();
            depthProduct *= analysis.subtreeNodes[body];
            if (!chargeProtocol(depthProduct))
                return false;
            std::vector<Mechanism> bodyWord;
            if (!projectEntryWord(p, actual, body, loop, bodyWord, reason))
                return false;
            word.insert(word.end(), bodyWord.begin(), bodyWord.end());
        } else
            for (unsigned child : p.nodes[body].children)
                append(child);
        if (late)
            appendLate(analysis.next[loop]);
        else
            append(analysis.next[loop]);
        // Reconstruct the actual combined nonempty visit, preserving command
        // order rather than assuming independent per-cell acknowledgment pairs.
        // The empty visit has no episode commands. Other domains use disjoint
        // keys and provide no assumed causality to this two-copy proof.
        if (!verifyProtocolWord(word, reason))
            return false;
    }
    return true;
}

struct AlternativeChoiceCertificates {
    using Key = PrefixState::EventKey;
    static constexpr uint64_t MaxWork = 1u << 22;
    struct Endpoint {
        unsigned site = NoCut, index = 0;
        c::Mechanism mechanism;
        bool canonical = false;
    };
    std::set<Key> familyKeys;
    std::map<unsigned, CausalTransfer> choiceReturns;
    uint64_t work = 0;
    uint64_t families = 0, sites = 0, prefixSteps = 0;
    std::set<unsigned> sourceScopes;
    bool exhausted = false;

    bool reserve(uint64_t count, uint64_t width, uint64_t limit)
    {
        limit = std::min(limit, MaxWork);
        if (work > limit || (width && count > (limit - work) / width)) {
            exhausted = true;
            return false;
        }
        work += count * width;
        return true;
    }

    bool build(
        const c::Program& p, const DemandAnalysis& analysis, const Commands& actual, const std::set<Key>& candidates,
        uint64_t limit, std::string& reason)
    {
        using namespace c;
        if (candidates.empty())
            return true;
        if (candidates.size() > MaxAlternatives) {
            reason = "too many alternative Choice families";
            return false;
        }
        uint64_t commands = 0;
        if (!reserve(actual.size(), 2, limit))
            return false;
        for (const auto& group : actual) {
            if (!reserve(group.size(), 8, limit))
                return false;
            commands += group.size();
        }
        (void)commands;
        // Build structural intervals once. Repeated parent-chain membership
        // tests inside an ancestor scan would otherwise multiply tree depth.
        if (!reserve(p.nodes.size(), 10, limit))
            return false;
        std::vector<uint64_t> begin(p.nodes.size()), end(p.nodes.size());
        std::vector<std::pair<unsigned, bool>> stack{{unsigned(p.nodes.size() - 1), false}};
        uint64_t position = 0;
        while (!stack.empty()) {
            auto [id, leaving] = stack.back();
            stack.pop_back();
            if (leaving) {
                end[id] = position++;
                continue;
            }
            begin[id] = position++;
            stack.push_back({id, true});
            for (unsigned child : p.nodes[id].children)
                stack.push_back({child, false});
        }
        std::map<Key, std::vector<Endpoint>> uses;
        for (unsigned site = 0; site < actual.size(); ++site)
            for (unsigned index = 0; index < actual[site].size(); ++index) {
                const auto& mechanism = actual[site][index];
                if (mechanism.kind == Mechanism::Publish || mechanism.kind == Mechanism::Acquire)
                    uses[{mechanism.first, mechanism.second, mechanism.forwardKey}].push_back(
                        {site, index, mechanism, false});
                else if (mechanism.kind == Mechanism::Rendezvous) {
                    uses[{mechanism.first, mechanism.second, mechanism.forwardKey}].push_back(
                        {site, index, mechanism, true});
                    uses[{mechanism.second, mechanism.first, mechanism.reverseKey}].push_back(
                        {site, index, mechanism, true});
                }
            }
        auto contains = [&](unsigned root, unsigned node) {
            return begin[root] <= begin[node] && end[node] <= end[root];
        };
        for (const auto& key : candidates) {
            auto found = uses.find(key);
            if (found == uses.end() || !reserve(found->second.size(), 12, limit))
                return false;
            const Endpoint* publication = nullptr;
            std::vector<const Endpoint*> waits;
            for (const auto& endpoint : found->second) {
                const auto& mechanism = endpoint.mechanism;
                if (endpoint.canonical || mechanism.participation != Mechanism::Every || mechanism.loop != NoCut ||
                    mechanism.word != NoCut ||
                    (mechanism.kind != Mechanism::Publish && mechanism.kind != Mechanism::Acquire)) {
                    reason = "alternative Choice family shares a physical key";
                    return false;
                }
                if (mechanism.kind == Mechanism::Publish) {
                    if (publication) {
                        reason = "alternative Choice family has multiple publications";
                        return false;
                    }
                    publication = &endpoint;
                } else
                    waits.push_back(&endpoint);
            }
            if (!publication || waits.size() < 2 || waits.size() > MaxAlternatives) {
                reason = "alternative Choice family has an unsupported endpoint population";
                return false;
            }
            if (!reserve(p.nodes.size(), waits.size() + p.cells + 4, limit))
                return false;
            unsigned choice = NoCut;
            for (unsigned ancestor = waits.front()->site; ancestor != NoCut; ancestor = analysis.parent[ancestor])
                if (p.nodes[ancestor].kind == Node::Choice &&
                    std::all_of(
                        waits.begin(), waits.end(), [&](const auto* wait) { return contains(ancestor, wait->site); })) {
                    choice = ancestor;
                    break;
                }
            if (choice == NoCut || analysis.parent[choice] == NoCut ||
                p.nodes[analysis.parent[choice]].kind != Node::Sequence ||
                analysis.parent[publication->site] != analysis.parent[choice] ||
                analysis.position[publication->site] >= analysis.position[choice]) {
                reason = "alternative Choice publication is outside its parent interval";
                return false;
            }
            // A compact structural cardinality proof. Bit n means that a path
            // contains n matching waits, with bit 2 also representing >2.
            std::function<std::optional<uint8_t>(unsigned)> cardinality = [&](unsigned id) -> std::optional<uint8_t> {
                if (!reserve(1 + actual[id].size(), 2, limit))
                    return {};
                const auto& node = p.nodes[id];
                if (node.kind == Node::For || node.kind == Node::While)
                    return {};
                unsigned own = std::count_if(actual[id].begin(), actual[id].end(), [&](const auto& mechanism) {
                    return mechanism.kind == Mechanism::Acquire && mechanism.participation == Mechanism::Every &&
                           Key{mechanism.first, mechanism.second, mechanism.forwardKey} == key;
                });
                if (own > 1)
                    own = 2;
                auto add = [](uint8_t left, uint8_t right) {
                    uint8_t result = 0;
                    for (unsigned a = 0; a < 3; ++a)
                        if (left & (1u << a))
                            for (unsigned b = 0; b < 3; ++b)
                                if (right & (1u << b))
                                    result |= uint8_t(1u << std::min(2u, a + b));
                    return result;
                };
                uint8_t children = 1;
                if (node.kind == Node::Choice) {
                    children = 0;
                    for (unsigned child : node.children) {
                        auto part = cardinality(child);
                        if (!part)
                            return {};
                        children |= *part;
                    }
                } else
                    for (unsigned child : node.children) {
                        auto part = cardinality(child);
                        if (!part)
                            return {};
                        children = add(children, *part);
                    }
                return add(uint8_t(1u << own), children);
            };
            auto counts = cardinality(choice);
            if (!counts || *counts != uint8_t(1u << 1)) {
                reason = "alternative Choice wait is not exactly once on every path";
                return false;
            }
            std::set<Key> acknowledgments;
            if (!reserve(waits.size(), 6, limit))
                return false;
            std::vector<std::vector<Mechanism>> alternatives;
            alternatives.reserve(waits.size());
            for (const auto* wait : waits) {
                if (wait->index + 2 >= actual[wait->site].size()) {
                    reason = "alternative Choice wait has no adjacent return ACK";
                    return false;
                }
                const auto& ackSet = actual[wait->site][wait->index + 1];
                const auto& ackWait = actual[wait->site][wait->index + 2];
                Key ack{ackSet.first, ackSet.second, ackSet.forwardKey};
                if (ackSet.kind != Mechanism::Publish || ackWait.kind != Mechanism::Acquire ||
                    ackSet.participation != Mechanism::Every || ackWait.participation != Mechanism::Every ||
                    ackSet.loop != NoCut || ackWait.loop != NoCut || ackSet.word != NoCut || ackWait.word != NoCut ||
                    ackSet.first != std::get<1>(key) || ackSet.second != std::get<0>(key) ||
                    ackWait.first != ackSet.first || ackWait.second != ackSet.second ||
                    ackWait.forwardKey != ackSet.forwardKey || ack == key || !acknowledgments.insert(ack).second) {
                    reason = "alternative Choice return ACK is not dedicated and adjacent";
                    return false;
                }
                auto ackUses = uses.find(ack);
                if (ackUses == uses.end() || ackUses->second.size() != 2 ||
                    std::any_of(ackUses->second.begin(), ackUses->second.end(), [](const auto& endpoint) {
                        return endpoint.canonical || endpoint.mechanism.participation != Mechanism::Every;
                    })) {
                    reason = "alternative Choice return ACK shares a physical key";
                    return false;
                }
                if (!reserve(4, 4 * LaneCount + 16, limit)) {
                    reason = "alternative Choice family exceeds its work bound";
                    return false;
                }
                alternatives.push_back({publication->mechanism, wait->mechanism, ackSet, ackWait});
            }
            // Every prior/next arm pair is checked, including i==j. The first
            // chain proves WAIT(F)i -> SET(Ri)i -> WAIT(Ri)i -> next SET(F).
            // The second proves an Ri token can be reused after skipped arms:
            // WAIT(Ri)i -> SET(F)j -> WAIT(F)j -> SET(Rj)j, followed on a
            // later return to i by SET(Ri)i. At most 8x8 fixed eight-command
            // projections are examined; production never enumerates traces.
            for (const auto& previous : alternatives)
                for (const auto& next : alternatives) {
                    if (!reserve(8, 4 * LaneCount + 16, limit)) {
                        reason = "alternative Choice pair proof exceeds its work bound";
                        return false;
                    }
                    std::vector<Mechanism> pair;
                    pair.reserve(8);
                    pair.insert(pair.end(), previous.begin(), previous.end());
                    pair.insert(pair.end(), next.begin(), next.end());
                    std::string projectedReason;
                    if (!verifyProtocolWord(pair, projectedReason)) {
                        reason = std::move(projectedReason);
                        return false;
                    }
                }
            if (familyKeys.count(key) || std::any_of(
                                             acknowledgments.begin(), acknowledgments.end(),
                                             [&](const auto& ack) { return familyKeys.count(ack); })) {
                reason = "alternative Choice family shares a certified key";
                return false;
            }
            if (!familyKeys.insert(key).second) {
                reason = "duplicate alternative Choice forward family";
                return false;
            }
            familyKeys.insert(acknowledgments.begin(), acknowledgments.end());
            auto [source, observer, number] = key;
            (void)number;
            auto transfer = choiceReturns.emplace(choice, identityTransfer()).first;
            // This edge is exported only after the actual four-endpoint word
            // has been proved for every ordered arm pair. It says that every
            // exit from this Choice returns the observer's causal history to
            // the source; it is neither payload-completion credit nor a fact
            // inferred from the projected remainder protocol.
            transfer->second[source] |= uint8_t(1u << observer);
            ++families;
            sites += waits.size();
            sourceScopes.insert(analysis.parent[choice]);
            const auto& siblings = p.nodes[analysis.parent[choice]].children;
            for (unsigned position = analysis.position[publication->site]; position < analysis.position[choice];
                 ++position)
                if (std::any_of(
                        analysis.effects[siblings[position]].begin(), analysis.effects[siblings[position]].end(),
                        [sourceLane = source](const auto& effect) {
                            return (effect.readers | effect.writers) & (1u << sourceLane);
                        }))
                    ++prefixSteps;
        }
        return true;
    }
};

bool verifyDemandProtocols(
    const c::Program& p, const DemandAnalysis& analysis, const std::vector<std::vector<c::Mechanism>>& actual,
    const DeferredDemandRings& deferred, std::string& reason, uint64_t childReturnLimit, uint64_t& childReturnWork,
    bool& usedChildReturns, bool& budgetExhausted, uint64_t alternativeChoiceLimit, uint64_t& alternativeChoiceWork,
    uint64_t& alternativeChoiceFamilies, uint64_t& alternativeChoiceSites, uint64_t& alternativeChoiceSourceScopes,
    uint64_t& alternativeChoicePrefixSteps, bool& alternativeChoiceBudgetExhausted, bool& usedAlternativeChoices)
{
    using Key = PrefixState::EventKey;
    usedChildReturns = false;
    usedAlternativeChoices = false;
    std::set<Key> entryKeys = deferred.keys;
    if (!verifyEntryProtocols(p, analysis, actual, deferred.keys, entryKeys, reason))
        return false;
    std::set<Key> packetKeys;
    for (const auto& commands : actual)
        for (const auto& mechanism : commands)
            if (mechanism.kind == c::Mechanism::Rendezvous) {
                packetKeys.insert({mechanism.first, mechanism.second, mechanism.forwardKey});
                packetKeys.insert({mechanism.second, mechanism.first, mechanism.reverseKey});
            }
    struct Population {
        unsigned scope = NoCut, sets = 0, waits = 0;
        bool multipleScopes = false;
    };
    std::map<Key, Population> population;
    for (unsigned id = 0; id < actual.size(); ++id)
        for (const auto& m : actual[id]) {
            Key forward{m.first, m.second, m.forwardKey};
            if ((m.kind == c::Mechanism::Publish || m.kind == c::Mechanism::Acquire) && deferred.keys.count(forward))
                continue;
            if (m.kind == c::Mechanism::Rendezvous &&
                (deferred.keys.count(forward) || deferred.keys.count({m.second, m.first, m.reverseKey}))) {
                reason = "canonical packet collides with a deferred demand ring";
                return false;
            }
            if (m.participation != c::Mechanism::Every)
                continue;
            if (validMechanism(p, m)) {
                if (m.kind == c::Mechanism::Rendezvous && (entryKeys.count({m.first, m.second, m.forwardKey}) ||
                                                           entryKeys.count({m.second, m.first, m.reverseKey}))) {
                    reason = "canonical packet collides with an entry episode";
                    return false;
                }
                // Actual canonical packets are complete-or-skipped templates.
                // Their fixed global orientation/keys permit sequential reuse
                // across nested domains, unlike independently numbered raw
                // words. Check the expanded template without flattening those
                // domains or imposing raw-key single-site constraints on it.
                if (m.kind == c::Mechanism::Rendezvous &&
                    !verifyProtocolWord(
                        {event(c::Mechanism::Publish, m.first, m.second, m.forwardKey),
                         event(c::Mechanism::Acquire, m.first, m.second, m.forwardKey),
                         event(c::Mechanism::Publish, m.second, m.first, m.reverseKey),
                         event(c::Mechanism::Acquire, m.second, m.first, m.reverseKey)},
                        reason))
                    return false;
                continue;
            }
            if ((m.kind != c::Mechanism::Publish && m.kind != c::Mechanism::Acquire) || m.first >= c::LaneCount ||
                m.second >= c::LaneCount || !p.target.event(lane(p, m.first), lane(p, m.second)) ||
                !p.target.available(lane(p, m.first), lane(p, m.second), m.forwardKey) ||
                std::find(p.target.compilerKeys.begin(), p.target.compilerKeys.end(), m.forwardKey) ==
                    p.target.compilerKeys.end() ||
                packetKeys.count(forward) || analysis.parent[id] == NoCut ||
                entryKeys.count({m.first, m.second, m.forwardKey}) ||
                p.nodes[analysis.parent[id]].kind != c::Node::Sequence) {
                reason = "invalid demand protocol mechanism or execution domain";
                return false;
            }
            auto& pop = population[{m.first, m.second, m.forwardKey}];
            pop.multipleScopes |= pop.scope != NoCut && pop.scope != analysis.parent[id];
            pop.scope = analysis.parent[id];
            (m.kind == c::Mechanism::Publish ? pop.sets : pop.waits)++;
        }
    std::set<Key> alternativeCandidates;
    for (const auto& [key, pop] : population)
        if (pop.sets == 1 && pop.waits >= 2 && pop.waits <= MaxAlternatives)
            alternativeCandidates.insert(key);
    AlternativeChoiceCertificates alternatives;
    if (!alternatives.build(p, analysis, actual, alternativeCandidates, alternativeChoiceLimit, reason)) {
        alternativeChoiceWork = alternatives.work;
        alternativeChoiceBudgetExhausted = alternatives.exhausted;
        return false;
    }
    alternativeChoiceWork = alternatives.work;
    alternativeChoiceFamilies = alternatives.families;
    alternativeChoiceSites = alternatives.sites;
    alternativeChoiceSourceScopes = alternatives.sourceScopes.size();
    alternativeChoicePrefixSteps = alternatives.prefixSteps;
    alternativeChoiceBudgetExhausted = alternatives.exhausted;
    usedAlternativeChoices = !alternatives.familyKeys.empty();
    for (const auto& [key, pop] : population) {
        if (alternatives.familyKeys.count(key))
            continue;
        if (pop.scope == NoCut || pop.multipleScopes) {
            reason = "event key shared by independently executing domains";
            return false;
        }
        if (!pop.sets || pop.sets != pop.waits) {
            reason = "demand key needs balanced publications and acquisitions per domain visit";
            return false;
        }
    }
    std::set<Key> ordinaryExcluded = deferred.keys;
    ordinaryExcluded.insert(alternatives.familyKeys.begin(), alternatives.familyKeys.end());
    std::map<unsigned, std::vector<c::Mechanism>> words;
    for (unsigned scope = 0; scope < p.nodes.size(); ++scope)
        if (p.nodes[scope].kind == c::Node::Sequence)
            words[scope] = demandWord(p, actual, scope, true, false, &ordinaryExcluded);
    // Preserve the existing flat path exactly. Structured summaries are an
    // optional rescue only when that path cannot establish rearm.
    std::string flatReason;
    bool flat = true;
    for (const auto& [scope, word] : words) {
        (void)scope;
        if (!verifyProtocolWord(word, flatReason)) {
            flat = false;
            break;
        }
    }
    if (flat)
        return true;

    // When an alternative family needs child-return rescue, both proofs share
    // the caller's one optional allowance. Do not renew a separate allowance
    // for the causal summaries after spending it on arm certificates.
    if (!alternatives.familyKeys.empty())
        childReturnLimit = std::min(childReturnLimit, alternativeChoiceLimit - alternativeChoiceWork);

    // Guarded entry/deferred keys have a different dynamic population and are
    // never imported into an ordinary child transfer. Build every structured
    // scope transactionally. If any summary is unavailable or exceeds its
    // optional allowance, report the mandatory flat check's failure.
    std::set<Key> protectedKeys = entryKeys;
    protectedKeys.insert(alternatives.familyKeys.begin(), alternatives.familyKeys.end());
    for (const auto& commands : actual)
        for (const auto& m : commands)
            if (m.participation != c::Mechanism::Every) {
                protectedKeys.insert({m.first, m.second, m.forwardKey});
                if (m.kind == c::Mechanism::Rendezvous)
                    protectedKeys.insert({m.second, m.first, m.reverseKey});
            }
    ChildReturnSummaries summaries;
    std::map<unsigned, StructuredProtocolWord> structured;
    bool useStructured = childReturnLimit != 0;
    std::string optionalReason;
    if (useStructured)
        useStructured = summaries.build(
            p, actual, true, false, &protectedKeys, childReturnLimit, optionalReason, &alternatives.choiceReturns);
    if (useStructured)
        for (const auto& [scope, flat] : words) {
            (void)flat;
            auto word = summaries.word(p, actual, scope, true, false, &protectedKeys, childReturnLimit);
            if (!word) {
                useStructured = false;
                break;
            }
            constexpr uint64_t matrix = uint64_t(c::LaneCount) * c::LaneCount;
            constexpr uint64_t numericMatrix = matrix * c::LaneCount;
            if (!summaries.reserve(word->commands.size(), 2 * c::LaneCount + 8, childReturnLimit) ||
                !summaries.reserve(word->transfers.size(), 2 * numericMatrix + matrix + 4, childReturnLimit)) {
                useStructured = false;
                break;
            }
            structured.emplace(scope, std::move(*word));
        }
    childReturnWork += summaries.work;
    budgetExhausted = summaries.exhausted || childReturnLimit == 0;
    if (useStructured) {
        std::string structuredReason;
        for (const auto& [scope, word] : structured) {
            (void)scope;
            if (!verifyProtocolWord(word, structuredReason)) {
                useStructured = false;
                break;
            }
        }
        usedChildReturns = useStructured;
    }
    if (!useStructured) {
        reason = std::move(flatReason);
        return false;
    }
    return true;
}
} // namespace

namespace {
using DemandInvariants = std::map<unsigned, c::State>;
using DemandIdentity = std::array<unsigned, 5>;
using DemandFallbacks = std::set<DemandIdentity>;
DemandIdentity identity(const c::CompletionDemand& d)
{
    return {d.scope, d.publication, d.acquisition, d.source, d.observer};
}
bool applyEntryCredit(PrefixState& state, const c::Mechanism& m, const DemandAnalysis& analysis, std::string& reason)
{
    if (m.participation == c::Mechanism::NonEmpty) {
        // A conditional ACK does not establish unconditional exit completion.
        // Zero-trip state must survive the ordinary loop join unchanged.
        if (m.kind == c::Mechanism::Publish)
            state.publish(m);
        else
            state.receipts.erase({m.first, m.second, m.forwardKey});
        return true;
    }
    auto found = state.receipts.find({m.first, m.second, m.forwardKey});
    if (found == state.receipts.end()) {
        reason = "first-consumer receipt has no actual provider";
        return false;
    }
    const auto& remaining = found->second;
    const auto& effects = analysis.effects[m.loop];
    for (unsigned cell = 0; cell < remaining.size(); ++cell)
        if ((effects[cell].readers & ~remaining[cell].readers) || (effects[cell].writers & ~remaining[cell].writers)) {
            reason = "entry credit does not retain the complete loop MAY effects";
            return false;
        }
    // The guarded first WAIT acquires this old prefix exactly once. Thereafter
    // P := P & remaining is an idempotent credit: remaining contains the entire
    // loop MAY summary, so reapplying it abstractly cannot erase new generations.
    // This is an episode-scoped completion fact, not repeated token consumption.
    state.acquireRemaining(m.second, remaining);
    return true;
}
c::Result verifyDemandImpl(
    const c::Program& p, const std::vector<std::vector<c::Mechanism>>& actual, DemandInvariants* invariants,
    uint64_t childReturnLimit = ChildReturnSummaries::MaxWork, bool* usedChildReturns = nullptr,
    uint64_t alternativeChoiceLimit = AlternativeChoiceCertificates::MaxWork, bool* usedAlternativeChoices = nullptr);
c::Result verifyConstructionDemands(
    const c::Program& p, const Commands& actual, DemandInvariants* invariants, const ChildReturnOptions* childReturns,
    bool* usedChildReturns = nullptr, const DemandAnalysis::ChoiceIncomingOptions* choiceIncoming = nullptr);
c::Result constructDemandCandidate(
    const c::Program& p, const DemandInvariants* invariants, DemandFallbacks& unassigned,
    const DemandFallbacks* forced = nullptr, const Cuts* recurring = nullptr,
    const DemandAnalysis::ChoiceIncomingOptions* choiceIncoming = nullptr,
    const ChildReturnOptions* childReturns = nullptr)
{
    using namespace c;
    c::Result result;
    DemandAnalysis analysis;
    if (!analysis.build(p, result.reason, choiceIncoming))
        return result;
    analysis.recordEntryStats(result);
    result.before.resize(p.nodes.size());
    std::vector<std::vector<Mechanism>> publications(p.nodes.size());
    using Direction = std::pair<unsigned, unsigned>;
    using FamilyKey = std::tuple<unsigned, unsigned, unsigned>;
    std::map<Direction, std::set<unsigned>> entryKeys;
    struct Family {
        unsigned acknowledgment, last;
    };
    std::map<FamilyKey, Family> families;
    std::vector<PrefixState::EventKey> demandKeys;
    struct UniversalRuntime {
        std::optional<unsigned> forward;
        std::set<unsigned> realized;
    };
    std::vector<UniversalRuntime> universalRuntime(analysis.universalChoices.size());
    std::set<PrefixState::EventKey> universalForwardKeys;
    std::set<FamilyKey> universalAckFamilies;
    bool universalFailed = false;
    unsigned nextLogicalKey = 1;
    if (!analysis.universalChoices.empty())
        for (unsigned physical : p.target.compilerKeys) {
            if (physical == NoCut) {
                result.reason = "universal Choice logical namespace unavailable";
                return result;
            }
            nextLogicalKey = std::max(nextLogicalKey, physical + 1);
        }
    DemandRings rings(p.nodes.size());
    std::map<unsigned, State> ringIncoming;
    if (recurring) {
        result.cellVisits += uint64_t(p.cells) * p.nodes.size();
        rings.cuts = *recurring; // Reuse immutable discovery, charge before its state copy.
        // A virtual namespace disjoint from physical entry keys. Final global
        // numbering and protocol checks remain the ordinary demand allocator.
        for (unsigned k : p.target.compilerKeys) {
            if (k == NoCut) {
                result.reason = "optional ring logical namespace unavailable";
                return result;
            }
            nextLogicalKey = std::max(nextLogicalKey, k + 1);
        }
        for (auto& cycle : rings.cuts.cycles)
            for (unsigned g = 0; g < cycle.groups.size(); ++g) {
                auto direction = std::make_pair(cycle.groups[g].lane, cycle.groups[(g + 1) % cycle.groups.size()].lane);
                if (!cycle.keys.count(direction)) {
                    if (nextLogicalKey == NoCut) {
                        result.reason = "optional ring logical namespace exhausted";
                        return result;
                    }
                    cycle.keys[direction] = nextLogicalKey++;
                }
            }
        rings.index();
        result.cutCycles = rings.cuts.cycles.size();
    }
    auto allocate = [&](unsigned a, unsigned b) -> std::optional<unsigned> {
        if (!p.target.event(lane(p, a), lane(p, b)))
            return {};
        for (unsigned k : p.target.compilerKeys)
            if (key(p, a, b) != k && p.target.available(lane(p, a), lane(p, b), k) && nextLogicalKey != NoCut)
                return nextLogicalKey++;
        return {};
    };
    auto tryDirectDemand = [&](const CompletionDemand& demand, const Effects& witness, unsigned observer,
                               unsigned acquisition, PrefixState& state) {
        if (forced && forced->count(identity(demand))) {
            ++result.replayedFallbackDemands;
            return false;
        }
        auto universal = analysis.universalChoiceFamily(demand);
        PrefixState::EventKey receiptKey{demand.source, demand.source, demand.publication};
        if (universal) {
            if (*universal >= universalRuntime.size() || !universalRuntime[*universal].forward) {
                universalFailed = true;
                return false;
            }
            receiptKey = {demand.source, observer, *universalRuntime[*universal].forward};
        }
        auto receipt = state.receipts.find(receiptKey);
        if (receipt == state.receipts.end())
            return false;
        auto trialState = state;
        trialState.acquireRemaining(observer, receipt->second);
        if (trialState.demands(observer, witness) & (1u << demand.source))
            return false;
        FamilyKey familyKey{demand.scope, demand.source, observer};
        auto family = families.find(familyKey);
        unsigned savedLogicalKey = nextLogicalKey;
        auto forward = universal ? universalRuntime[*universal].forward : allocate(demand.source, observer);
        auto ack = family == families.end() ? allocate(observer, demand.source) :
                                              std::optional<unsigned>(family->second.acknowledgment);
        if (!forward || !ack) {
            nextLogicalKey = savedLogicalKey;
            return false;
        }
        families[familyKey] = {*ack, acquisition};
        if (!universal)
            publications[demand.publication].push_back(event(Mechanism::Publish, demand.source, observer, *forward));
        bool corruptChoice = choiceIncoming && choiceIncoming->corrupt && analysis.choiceWitness(demand);
        bool corruptUniversal = universal && choiceIncoming && choiceIncoming->corruptAlternative &&
                                universalRuntime[*universal].realized.empty();
        if (!corruptChoice && !corruptUniversal)
            result.before[acquisition].push_back(event(Mechanism::Acquire, demand.source, observer, *forward));
        state.acquireRemaining(observer, receipt->second);
        result.demands.push_back(demand);
        demandKeys.push_back({demand.source, observer, *forward});
        if (universal) {
            universalRuntime[*universal].realized.insert(acquisition);
            universalForwardKeys.insert({demand.source, observer, *forward});
            universalAckFamilies.insert(familyKey);
        }
        ++result.directHandoffs;
        return true;
    };
    std::function<bool(unsigned, PrefixState&)> visit = [&](unsigned id, PrefixState& state) {
        ++result.nodeVisits;
        result.cellVisits += p.cells * LaneCount;
        if (invariants && !analysis.owned[id].empty()) {
            auto found = invariants->find(id);
            if (found != invariants->end())
                for (unsigned observer = 0; observer < LaneCount; ++observer)
                    for (unsigned cell : analysis.owned[id]) {
                        auto& h = state.pending[observer][cell];
                        h.readers &= found->second.pending[observer][cell].readers;
                        h.writers &= found->second.pending[observer][cell].writers;
                    }
        }
        // Prospective publications are pure snapshots, not assumed hardware
        // actions. Keep at most eight cuts per source, independently of IR size.
        // Discarding a proposal loses precision, never an outstanding event.
        for (unsigned source : analysis.capture[id]) {
            unsigned count = 0;
            for (const auto& entry : state.receipts)
                count += std::get<0>(entry.first) == source && std::get<1>(entry.first) == source;
            if (count >= MaxAlternatives) {
                auto old = std::find_if(state.receipts.begin(), state.receipts.end(), [&](const auto& entry) {
                    return std::get<0>(entry.first) == source && std::get<1>(entry.first) == source;
                });
                state.receipts.erase(old);
            }
            state.publish(event(Mechanism::Publish, source, source, id));
        }
        for (const auto& m : result.before[id])
            if (m.participation != Mechanism::Every && !applyEntryCredit(state, m, analysis, result.reason))
                return false;
        for (const auto& m : rings.sites[id]) {
            result.before[id].push_back(m);
            if (m.kind == Mechanism::Publish)
                state.publish(m);
            else {
                if (!state.receipts.count({m.first, m.second, m.forwardKey})) {
                    result.reason = "optional ring acquisition has no publication";
                    return false;
                }
                state.acquirePrefix(m);
                if (!rings.credit(m, state, ringIncoming, result.reason))
                    return false;
            }
        }
        const auto& n = p.nodes[id];
        if (n.kind == Node::Operation) {
            if (!realizeVisibility(
                    p, n, state, result.before[id], result.reason,
                    result.visibilityRequirements))
                return false;
            for (unsigned source = 0; source < LaneCount; ++source) {
                if (!(state.demands(n.lane, n.effects) & (1u << source)))
                    continue;
                ++result.acquisitions;
                bool direct = false;
                for (const auto& demand : analysis.requests[id]) {
                    if (demand.source != source)
                        continue;
                    // A bounded replay propagates the real canonical transfer
                    // at this original demand cut. Do not substitute a later
                    // proposal after the selected demand was forced to the
                    // canonical path.
                    if (forced && forced->count(identity(demand))) {
                        ++result.replayedFallbackDemands;
                        break;
                    }
                    // The complete current operation demand, not merely its
                    // local cell list, must be discharged by the proposed
                    // source snapshot.
                    if ((direct = tryDirectDemand(demand, n.effects, n.lane, id, state)))
                        break;
                }
                if (!direct) {
                    ++result.demandFallbacks;
                    if (!acquire(p, source, n.lane, state, result.before[id])) {
                        result.reason = "target cannot realize remaining completion demand";
                        return false;
                    }
                }
            }
            state.seed(n.effects);
            for (const auto& cut : analysis.release[id])
                state.receipts.erase(cut);
            return true;
        }
        if (n.kind == Node::Choice) {
            std::vector<PrefixState::EventKey> universalReceipts;
            for (unsigned family = 0; family < analysis.universalChoices.size(); ++family) {
                const auto& proposal = analysis.universalChoices[family];
                if (proposal.choice != id)
                    continue;
                auto sourceReceipt = state.receipts.find({proposal.source, proposal.source, proposal.publication});
                auto forward = allocate(proposal.source, proposal.observer);
                if (sourceReceipt == state.receipts.end() || !forward) {
                    universalFailed = true;
                    continue;
                }
                universalRuntime[family].forward = *forward;
                auto mechanism = event(Mechanism::Publish, proposal.source, proposal.observer, *forward);
                publications[proposal.publication].push_back(mechanism);
                PrefixState::EventKey receipt{proposal.source, proposal.observer, *forward};
                state.receipts[receipt] = sourceReceipt->second;
                universalReceipts.push_back(receipt);
            }
            // A universal first-observer request executes before branch state
            // is copied. It remains an ordinary Every SET/WAIT family; failure
            // to realize this optional prefix leaves branch-local demands to
            // the unchanged operation path.
            for (const auto& demand : analysis.requests[id]) {
                const auto* witness = analysis.choiceWitness(demand);
                if (!witness || !(state.demands(demand.observer, *witness) & (1u << demand.source)))
                    continue;
                ++result.acquisitions;
                tryDirectDemand(demand, *witness, demand.observer, id, state);
            }
            auto other = state;
            if (!visit(n.children[0], state) || !visit(n.children[1], other))
                return false;
            for (const auto& receipt : universalReceipts) {
                state.receipts.erase(receipt);
                other.receipts.erase(receipt);
            }
            state.join(other);
            for (unsigned family = 0; family < analysis.universalChoices.size(); ++family) {
                const auto& proposal = analysis.universalChoices[family];
                if (proposal.choice != id)
                    continue;
                std::set<unsigned> expected;
                for (const auto& demand : proposal.alternatives)
                    expected.insert(demand.acquisition);
                universalFailed |= universalRuntime[family].realized != expected;
            }
            for (const auto& cut : analysis.release[id])
                state.receipts.erase(cut);
            return true;
        }
        if (n.kind == Node::For || n.kind == Node::While) {
            auto entry = state;
            if (rings.loopOwners.count(id))
                ringIncoming.insert_or_assign(id, entry);
            std::vector<PrefixState::EventKey> episodeReceipts;
            std::map<Direction, Effects> acquiredEntry;
            std::map<Direction, unsigned> entryReplies;
            for (const auto& demand : analysis.entries[id]) {
                const auto* witness = analysis.entryWitness(demand);
                if (!witness) {
                    result.reason = "entry demand has no independently derived first-consumer witness";
                    return false;
                }
                // A previous first consumer may already acquire this whole
                // incoming prefix. The source has no body effects on these
                // demanded cells, so that credit survives all later visits.
                auto acquired = acquiredEntry.find({demand.source, demand.observer});
                if (acquired != acquiredEntry.end()) {
                    auto covered = state;
                    covered.seed(analysis.effects[id]);
                    covered.acquireRemaining(demand.observer, acquired->second);
                    if (!(covered.demands(demand.observer, *witness) & (1u << demand.source)))
                        continue;
                }
                auto receipt = state.receipts.find({demand.source, demand.source, demand.publication});
                if (receipt == state.receipts.end())
                    continue;
                auto remaining = receipt->second;
                merge(remaining, analysis.effects[id]);
                auto trialState = state;
                trialState.seed(analysis.effects[id]);
                if (!(trialState.demands(demand.observer, *witness) & (1u << demand.source)))
                    continue;
                trialState.acquireRemaining(demand.observer, remaining);
                if (trialState.demands(demand.observer, *witness) & (1u << demand.source))
                    continue;
                auto trial = entryKeys;
                auto allocateEntry = [&](unsigned a, unsigned b) -> std::optional<unsigned> {
                    for (unsigned k : p.target.compilerKeys)
                        if (key(p, a, b) != k && p.target.event(lane(p, a), lane(p, b)) &&
                            p.target.available(lane(p, a), lane(p, b), k) && !trial[{a, b}].count(k)) {
                            trial[{a, b}].insert(k);
                            return k;
                        }
                    return {};
                };
                auto forward = allocateEntry(demand.source, demand.observer);
                auto existingReply = entryReplies.find({demand.source, demand.observer});
                auto reverse = existingReply == entryReplies.end() ? allocateEntry(demand.observer, demand.source) :
                                                                     std::optional<unsigned>(existingReply->second);
                if (!forward || !reverse)
                    continue;
                entryKeys = std::move(trial);
                auto make = [&](Mechanism::Kind kind, unsigned a, unsigned b, unsigned k,
                                Mechanism::Participation participation) {
                    auto m = event(kind, a, b, k);
                    m.participation = participation;
                    m.loop = id;
                    return m;
                };
                publications[demand.publication].push_back(
                    make(Mechanism::Publish, demand.source, demand.observer, *forward, Mechanism::NonEmpty));
                result.before[demand.acquisition].push_back(
                    make(Mechanism::Acquire, demand.source, demand.observer, *forward, Mechanism::First));
                if (existingReply == entryReplies.end()) {
                    result.before[analysis.next[id]].push_back(
                        make(Mechanism::Publish, demand.observer, demand.source, *reverse, Mechanism::NonEmpty));
                    result.before[analysis.next[id]].push_back(
                        make(Mechanism::Acquire, demand.observer, demand.source, *reverse, Mechanism::NonEmpty));
                    entryReplies[{demand.source, demand.observer}] = *reverse;
                    ++result.entryReplyFamilies;
                }
                // The entry provider is only a temporary abstract receipt for
                // the nonempty body visit. Keep it out of the zero-trip state:
                // the guarded SET does not execute when the loop is skipped.
                state.receipts[{demand.source, demand.observer, *forward}] = receipt->second;
                episodeReceipts.push_back({demand.source, demand.observer, *forward});
                if (acquired == acquiredEntry.end())
                    acquiredEntry[{demand.source, demand.observer}] = std::move(remaining);
                else
                    for (unsigned cell = 0; cell < p.cells; ++cell) {
                        acquired->second[cell].readers &= remaining[cell].readers;
                        acquired->second[cell].writers &= remaining[cell].writers;
                    }
                ++result.entryEpisodes;
            }
            state.seed(analysis.effects[id]);
            if (invariants) {
                auto found = invariants->find(id);
                if (found != invariants->end())
                    for (unsigned observer = 0; observer < LaneCount; ++observer)
                        for (unsigned cell = 0; cell < p.cells; ++cell) {
                            auto& h = state.pending[observer][cell];
                            const auto& hint = found->second.pending[observer][cell];
                            const auto& incoming = entry.pending[observer][cell];
                            h.readers = (h.readers & hint.readers) | incoming.readers;
                            h.writers = (h.writers & hint.writers) | incoming.writers;
                        }
            }
            if (!visit(n.children[0], state))
                return false;
            // Stable entry credit is scoped to this body. No guarded episode
            // receipt may leak into the zero-trip join or a later sibling.
            for (const auto& receipt : episodeReceipts)
                state.receipts.erase(receipt);
            if (n.kind == Node::While) {
                auto exit = state;
                if (!visit(n.children[1], state))
                    return false;
                state = std::move(exit);
            } else
                state.join(entry);
            ringIncoming.erase(id);
            return true;
        }
        for (unsigned child : n.children)
            if (!visit(child, state))
                return false;
        return true;
    };
    PrefixState state(p.cells);
    result.success = visit(p.nodes.size() - 1, state);
    for (const auto& proposal : analysis.universalChoices)
        for (const auto& demand : proposal.alternatives) {
            auto family = families.find({demand.scope, demand.source, demand.observer});
            universalFailed |= family == families.end() || family->second.last != demand.acquisition;
        }
    if (!result.success || universalFailed) {
        if (result.reason.empty())
            result.reason = "universal Choice family was not realized exactly";
        result.success = false;
        result.before.clear();
        return result;
    }
    // One reply can acknowledge several different storage demands. Its SET
    // follows every family acquisition on the observer lane. The reply WAIT
    // gates the source before the next visit's publications. Do not give the
    // forward constructor unearned payload completion credit for this reply.
    for (const auto& [key, family] : families) {
        auto [scope, source, observer] = key;
        result.before[family.last].push_back(event(Mechanism::Publish, observer, source, family.acknowledgment));
        result.before[family.last].push_back(event(Mechanism::Acquire, observer, source, family.acknowledgment));
        ++result.sharedAcknowledgments;
    }
    for (unsigned id = 0; id < publications.size(); ++id)
        result.before[id].insert(result.before[id].begin(), publications[id].begin(), publications[id].end());
    // Prefer the return causality already supplied by required handoffs. Each
    // trial removes only an optional reply, never a physical demand or a
    // readiness/release endpoint. Check only the changed word: at most one
    // trial per directed family (fixed lane population) in each scope.
    std::set<FamilyKey> removedAcknowledgments;
    for (const auto& [familyKey, family] : families) {
        if (universalAckFamilies.count(familyKey))
            continue;
        auto [scope, source, observer] = familyKey;
        auto& commands = result.before[family.last];
        auto saved = commands;
        commands.erase(
            std::remove_if(
                commands.begin(), commands.end(),
                [sourceLane = source, targetLane = observer, ack = family.acknowledgment](const auto& m) {
                    return m.participation == Mechanism::Every &&
                           (m.kind == Mechanism::Publish || m.kind == Mechanism::Acquire) && m.first == targetLane &&
                           m.second == sourceLane && m.forwardKey == ack;
                }),
            commands.end());
        std::string why;
        if (!verifyProtocolWord(demandWord(p, result.before, scope, true, true), why))
            commands = std::move(saved);
        else {
            --result.sharedAcknowledgments;
            ++result.reusedAcknowledgments;
            removedAcknowledgments.insert(familyKey);
        }
    }
    // Number only the finished logical protocol. A memory cell does not own
    // a key, nor does each logical handoff need a distinct physical key. The
    // certificate uses actual WAIT-consumption ticks, not lexical intervals.
    // For each color prove consecutive rearm AND the last->first copy edge.
    // Greedy failure is not a proof of infeasibility. Retain successful colors
    // and replace only unassigned logical acquisitions by stronger canonical
    // packets at the same cut. Remove their publications, then freshly recheck
    // the complete actual word (including packets) and physical requirements.
    std::map<Direction, std::set<unsigned>> occupied = entryKeys;
    for (const auto& [direction, keys] : entryKeys)
        result.protocolKeys += keys.size();
    AlternativeChoiceCertificates logicalAlternatives;
    std::map<PrefixState::EventKey, unsigned> universalNumbering;
    std::set<PrefixState::EventKey> universalPhysicalKeys;
    if (!universalForwardKeys.empty()) {
        uint64_t remaining = 0;
        if (choiceIncoming && !choiceIncoming->exhausted && choiceIncoming->charged <= choiceIncoming->limit)
            remaining = choiceIncoming->limit - choiceIncoming->charged;
        std::string alternativeReason;
        bool certified =
            logicalAlternatives.build(p, analysis, result.before, universalForwardKeys, remaining, alternativeReason);
        if (choiceIncoming) {
            // Exhausted proposals still pay for the work completed before refusal.
            if (logicalAlternatives.work > remaining)
                choiceIncoming->exhausted = true;
            else
                choiceIncoming->charged += logicalAlternatives.work;
            choiceIncoming->exhausted |= logicalAlternatives.exhausted;
        }
        if (!certified || logicalAlternatives.families != analysis.universalChoices.size()) {
            result.success = false;
            result.reason = alternativeReason.empty() ? "universal Choice logical certificate failed" :
                                                        std::move(alternativeReason);
            result.before.clear();
            return result;
        }
        auto trial = occupied;
        for (const auto& logical : logicalAlternatives.familyKeys) {
            unsigned source = std::get<0>(logical), observer = std::get<1>(logical);
            auto physical = std::find_if(p.target.compilerKeys.begin(), p.target.compilerKeys.end(), [&](unsigned k) {
                return key(p, source, observer) != k && p.target.available(lane(p, source), lane(p, observer), k) &&
                       !trial[{source, observer}].count(k);
            });
            if (physical == p.target.compilerKeys.end()) {
                result.success = false;
                result.reason = "universal Choice family exhausted dedicated event keys";
                result.before.clear();
                return result;
            }
            trial[{source, observer}].insert(*physical);
            universalNumbering.emplace(logical, *physical);
            universalPhysicalKeys.insert({source, observer, *physical});
            ++result.protocolKeys;
        }
        occupied = std::move(trial);
        result.alternativeChoiceCandidates = analysis.universalChoices.size();
        result.alternativeChoiceFamilies = logicalAlternatives.families;
        result.alternativeChoiceSites = logicalAlternatives.sites;
        for (const auto& family : analysis.universalChoices)
            result.alternativeChoiceSetsRemoved += family.alternatives.size() - 1;
        result.alternativeChoiceSourceScopes = logicalAlternatives.sourceScopes.size();
        result.alternativeChoicePrefixSteps = logicalAlternatives.prefixSteps;
        result.alternativeChoiceWork = choiceIncoming ? choiceIncoming->charged : logicalAlternatives.work;
    }
    std::set<unsigned> fallbackScopes;
    std::set<PrefixState::EventKey> fallbackKeys;
    std::map<FamilyKey, unsigned> numberedAcknowledgments;
    bool retainAckIdentity = childReturns && childReturns->enabled && childReturns->reserve(families.size(), 8);

    // If an entire directed-domain population fits, every logical protocol can
    // own a distinct physical key. In that case the canonical packet key is not
    // needed for fallback and may be used by one direct protocol. This is a
    // complete-population decision made before scope-local coloring: partial
    // use of the canonical key could otherwise collide with a later fallback.
    // Over-capacity domains retain the existing sharing-first policy.
    using ScopedEventKey = std::tuple<unsigned, unsigned, unsigned, unsigned>;
    std::map<Direction, std::set<ScopedEventKey>> domainPopulation;
    std::set<Direction> packetDirections;
    for (unsigned site = 0; site < result.before.size(); ++site)
        for (const auto& mechanism : result.before[site]) {
            if (mechanism.participation != Mechanism::Every)
                continue;
            if (mechanism.kind == Mechanism::Rendezvous) {
                packetDirections.insert({mechanism.first, mechanism.second});
                packetDirections.insert({mechanism.second, mechanism.first});
                continue;
            }
            if (mechanism.kind != Mechanism::Publish && mechanism.kind != Mechanism::Acquire)
                continue;
            PrefixState::EventKey logical{mechanism.first, mechanism.second, mechanism.forwardKey};
            unsigned scope = analysis.parent[site];
            // Alternative Choice numbering has already assigned both the
            // forward family and every arm-local return acknowledgment.  None
            // of those logical keys belong to the ordinary population below.
            if (scope != NoCut && !logicalAlternatives.familyKeys.count(logical))
                domainPopulation[{mechanism.first, mechanism.second}].insert(
                    {scope, mechanism.first, mechanism.second, mechanism.forwardKey});
        }
    std::map<ScopedEventKey, unsigned> dedicatedNumbering;
    std::set<std::pair<unsigned, unsigned>> consideredPairs;
    for (const auto& [direction, ignored] : domainPopulation) {
        (void)ignored;
        std::pair<unsigned, unsigned> pair = std::minmax(direction.first, direction.second);
        if (!consideredPairs.insert(pair).second)
            continue;
        std::array<Direction, 2> directions{{{pair.first, pair.second}, {pair.second, pair.first}}};
        std::array<std::vector<unsigned>, 2> available;
        bool feasible = true, usesCanonical = false;
        for (unsigned i = 0; i < directions.size(); ++i) {
            const auto& candidate = directions[i];
            if (packetDirections.count(candidate)) {
                feasible = false;
                break;
            }
            auto population = domainPopulation.find(candidate);
            size_t populationSize = population == domainPopulation.end() ? 0 : population->second.size();
            unsigned noncanonical = 0;
            auto canonical = key(p, candidate.first, candidate.second);
            for (unsigned physical : p.target.compilerKeys)
                if (p.target.available(lane(p, candidate.first), lane(p, candidate.second), physical) &&
                    !occupied[candidate].count(physical)) {
                    available[i].push_back(physical);
                    noncanonical += !canonical || physical != *canonical;
                }
            feasible &= populationSize <= available[i].size();
            usesCanonical |= populationSize > noncanonical;
        }
        // A canonical packet occupies both directions. Dedicate canonical keys
        // only if every raw protocol in both directions fits, so no later
        // allocation fallback can need that packet. Otherwise preserve the
        // established sharing-first policy and stable numbering unchanged.
        if (!feasible || !usesCanonical)
            continue;
        for (unsigned i = 0; i < directions.size(); ++i) {
            const auto& candidate = directions[i];
            auto population = domainPopulation.find(candidate);
            if (population == domainPopulation.end() || population->second.empty())
                continue;
            auto physical = available[i].begin();
            for (const auto& scoped : population->second) {
                dedicatedNumbering.emplace(scoped, *physical);
                occupied[candidate].insert(*physical++);
            }
            ++result.dedicatedAllocationDomains;
            result.dedicatedAllocationKeys += population->second.size();
            result.protocolKeys += population->second.size();
        }
    }
    for (unsigned scope : analysis.scopeOrder) {
        const auto* excluded = logicalAlternatives.familyKeys.empty() ? nullptr : &logicalAlternatives.familyKeys;
        auto word = demandWord(p, result.before, scope, true, true, excluded);
        // An all-alternative scope still contains endpoints that must receive
        // their reserved physical numbers even though its ordinary word is empty.
        ProtocolFacts facts;
        if (!verifyProtocolWord(word, result.reason, &facts)) {
            result.success = false;
            result.before.clear();
            return result;
        }
        struct Color {
            unsigned number;
            uint64_t nextFirst, lastConsumed;
            bool ring;
        };
        std::map<Direction, std::vector<Color>> colors;
        std::map<PrefixState::EventKey, unsigned> numbering;
        auto trial = occupied;
        unsigned count = 0, dedicatedCount = 0;
        for (const auto& logical : facts.order) {
            unsigned a = std::get<0>(logical), b = std::get<1>(logical);
            auto dedicated = dedicatedNumbering.find({scope, a, b, std::get<2>(logical)});
            if (dedicated != dedicatedNumbering.end()) {
                numbering[logical] = dedicated->second;
                ++dedicatedCount;
                continue;
            }
            const auto& use = facts.uses.at(logical);
            auto& choices = colors[{a, b}];
            auto reusable = std::find_if(choices.begin(), choices.end(), [&](const Color& c) {
                return !c.ring && !rings.owners.count(logical) && use.firstPublish >= c.lastConsumed &&
                       c.nextFirst >= use.consumed;
            });
            if (reusable != choices.end()) {
                numbering[logical] = reusable->number;
                reusable->lastConsumed = use.consumed;
                continue;
            }
            auto fresh = std::find_if(p.target.compilerKeys.begin(), p.target.compilerKeys.end(), [&](unsigned k) {
                return key(p, a, b) != k && p.target.available(lane(p, a), lane(p, b), k) && !trial[{a, b}].count(k);
            });
            if (fresh == p.target.compilerKeys.end()) {
                fallbackKeys.insert(logical);
                fallbackScopes.insert(scope);
                continue;
            }
            trial[{a, b}].insert(*fresh);
            choices.push_back({*fresh, use.nextPublish, use.consumed, bool(rings.owners.count(logical))});
            numbering[logical] = *fresh;
            ++count;
        }
        occupied = std::move(trial);
        result.protocolKeys += count;
        result.sharedProtocolKeys += numbering.size() - count - dedicatedCount;
        if (retainAckIdentity)
            for (auto it = families.lower_bound({scope, 0, 0}); it != families.end() && std::get<0>(it->first) == scope;
                 ++it) {
                auto [owner, source, observer] = it->first;
                (void)owner;
                auto number = numbering.find({observer, source, it->second.acknowledgment});
                if (number != numbering.end())
                    numberedAcknowledgments.emplace(it->first, number->second);
            }
        for (unsigned child : p.nodes[scope].children) {
            auto original = std::move(result.before[child]);
            auto& commands = result.before[child];
            commands.clear();
            for (auto m : original) {
                if (m.participation != Mechanism::Every) {
                    commands.push_back(m);
                    continue;
                }
                if (m.kind != Mechanism::Publish && m.kind != Mechanism::Acquire) {
                    commands.push_back(m);
                    continue;
                }
                PrefixState::EventKey logical{m.first, m.second, m.forwardKey};
                auto universalNumber = universalNumbering.find(logical);
                if (universalNumber != universalNumbering.end()) {
                    m.forwardKey = universalNumber->second;
                    commands.push_back(m);
                    continue;
                }
                if (fallbackKeys.count(logical) && rings.owners.count(logical)) {
                    result.success = false;
                    result.reason = "optional recurring demand ring exhausted event keys";
                    result.before.clear();
                    return result;
                }
                if (!fallbackKeys.count(logical)) {
                    m.forwardKey = numbering.at(logical);
                    commands.push_back(m);
                    continue;
                }
                if (m.kind == Mechanism::Publish)
                    continue;
                State dummy(p.cells);
                if (!acquire(p, m.first, m.second, dummy, commands)) {
                    result.success = false;
                    result.reason = "target cannot realize demand allocation fallback";
                    result.before.clear();
                    return result;
                }
                ++result.demandFallbacks;
            }
        }
        // Packet replacement preserves (and may strengthen) the original
        // acquisition's causal edges. Do not accept that argument alone: check
        // actual renamed raw events together with the actual canonical packets.
        if (!verifyProtocolWord(
                demandWord(
                    p, result.before, scope, true, false,
                    universalPhysicalKeys.empty() ? nullptr : &universalPhysicalKeys),
                result.reason)) {
            result.success = false;
            result.before.clear();
            return result;
        }
    }
    result.allocationFallbackScopes = fallbackScopes.size();
    result.allocationFallbackKeys = fallbackKeys.size();
    // Filter receipts/statistics once, not once per failed allocation.
    std::vector<CompletionDemand> retained;
    std::set<FamilyKey> retainedFamilies;
    for (unsigned i = 0; i < result.demands.size(); ++i) {
        const auto& d = result.demands[i];
        if (!fallbackKeys.count(demandKeys[i])) {
            retained.push_back(d);
            retainedFamilies.insert({d.scope, d.source, d.observer});
        } else
            unassigned.insert(identity(d));
    }
    result.demands = std::move(retained);
    result.directHandoffs = result.demands.size();
    result.choiceDemandFamilies = std::count_if(result.demands.begin(), result.demands.end(), [&](const auto& d) {
        return analysis.choiceWitness(d) != nullptr;
    });
    for (const auto& [familyKey, family] : families)
        if (removedAcknowledgments.count(familyKey)) {
            if (!retainedFamilies.count(familyKey))
                --result.reusedAcknowledgments;
        } else if (fallbackKeys.count({std::get<2>(familyKey), std::get<1>(familyKey), family.acknowledgment}))
            --result.sharedAcknowledgments;

    // Post-allocation removal only: coloring and its reserved-key population
    // are final. Discover the exact adjacent numbered ACK endpoints, remove a
    // single combined set, and accept it only after fresh structured protocol
    // and physical verification. A failed optional trial restores the exact
    // numbered command population.
    if (retainAckIdentity && childReturns->reserve(p.nodes.size(), 2)) {
        uint64_t commandCount = 0;
        for (const auto& commands : result.before) {
            if (commands.size() > ChildReturnSummaries::MaxWork - commandCount) {
                commandCount = ChildReturnSummaries::MaxWork;
                break;
            }
            commandCount += commands.size();
        }
        struct AckCandidate {
            FamilyKey family;
            unsigned choice = NoCut, site = NoCut;
            size_t index = 0;
            PrefixState::EventKey key{};
        };
        std::vector<AckCandidate> candidates;
        bool affordable = childReturns->reserve(commandCount, 4) && childReturns->reserve(families.size(), 16) &&
                          childReturns->reserve(result.demands.size(), 4);
        std::map<FamilyKey, unsigned> choiceRoots;
        std::set<FamilyKey> ambiguousChoices;
        if (affordable)
            for (const auto& demand : result.demands) {
                if (demand.acquisition >= p.nodes.size() || p.nodes[demand.acquisition].kind != Node::Choice)
                    continue;
                FamilyKey family{demand.scope, demand.source, demand.observer};
                auto [found, inserted] = choiceRoots.emplace(family, demand.acquisition);
                if (!inserted && found->second != demand.acquisition)
                    ambiguousChoices.insert(family);
            }
        std::set<PrefixState::EventKey> protectedKeys;
        protectedKeys.insert(universalPhysicalKeys.begin(), universalPhysicalKeys.end());
        if (affordable)
            for (const auto& commands : result.before)
                for (const auto& mechanism : commands)
                    if (mechanism.participation != Mechanism::Every) {
                        protectedKeys.insert({mechanism.first, mechanism.second, mechanism.forwardKey});
                        if (mechanism.kind == Mechanism::Rendezvous)
                            protectedKeys.insert({mechanism.second, mechanism.first, mechanism.reverseKey});
                    }
        if (affordable)
            for (const auto& [familyKey, family] : families) {
                if (universalAckFamilies.count(familyKey))
                    continue;
                auto choice = choiceRoots.find(familyKey);
                auto ackNumber = numberedAcknowledgments.find(familyKey);
                if (choice == choiceRoots.end() || ambiguousChoices.count(familyKey) ||
                    removedAcknowledgments.count(familyKey) || !retainedFamilies.count(familyKey) ||
                    family.last >= result.before.size() || ackNumber == numberedAcknowledgments.end())
                    continue;
                auto [scope, source, observer] = familyKey;
                (void)scope;
                const auto& commands = result.before[family.last];
                if (!childReturns->reserve(commands.size(), 2)) {
                    affordable = false;
                    break;
                }
                std::optional<size_t> match;
                for (size_t i = 0; i + 1 < commands.size(); ++i) {
                    const auto& publish = commands[i];
                    const auto& acquire = commands[i + 1];
                    bool adjacent = publish.kind == Mechanism::Publish && acquire.kind == Mechanism::Acquire &&
                                    publish.participation == Mechanism::Every &&
                                    acquire.participation == Mechanism::Every && publish.first == observer &&
                                    publish.second == source && acquire.first == observer && acquire.second == source &&
                                    publish.forwardKey == acquire.forwardKey && publish.forwardKey == ackNumber->second;
                    if (!adjacent)
                        continue;
                    if (match) {
                        match.reset();
                        break;
                    }
                    match = i;
                }
                if (!match)
                    continue;
                PrefixState::EventKey physical{observer, source, commands[*match].forwardKey};
                if (protectedKeys.count(physical))
                    continue;
                if (candidates.size() == MaxAlternatives) {
                    affordable = false;
                    break;
                }
                candidates.push_back({familyKey, choice->second, family.last, *match, physical});
            }
        result.childReturnCandidates = candidates.size();
        childReturns->candidates += candidates.size();
        if (affordable && !candidates.empty() && childReturns->reserve(commandCount, 2) &&
            childReturns->reserve(p.nodes.size()) && childReturns->reserve(candidates.size(), 12)) {
            auto saved = result.before;
            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return std::tie(a.site, a.index) > std::tie(b.site, b.index);
            });
            for (const auto& candidate : candidates) {
                auto& commands = result.before[candidate.site];
                commands.erase(commands.begin() + candidate.index, commands.begin() + candidate.index + 2);
            }
            if (childReturns->corrupt) {
                const auto& candidate = candidates.front();
                auto [scope, source, observer] = candidate.family;
                (void)scope;
                auto& commands = result.before[candidate.choice];
                auto found = std::find_if(
                    commands.begin(), commands.end(),
                    [sourceLane = source, targetLane = observer](const auto& mechanism) {
                        return mechanism.kind == Mechanism::Acquire && mechanism.participation == Mechanism::Every &&
                               mechanism.first == sourceLane && mechanism.second == targetLane;
                    });
                if (found != commands.end())
                    commands.erase(found);
            }
            uint64_t allowance = std::min(childReturns->limit, ChildReturnSummaries::MaxWork);
            uint64_t remaining = allowance - childReturns->charged;
            ChildReturnSummaries summaries;
            std::string optionalReason;
            bool proved = summaries.build(
                p, result.before, true, false, &protectedKeys, remaining, optionalReason,
                logicalAlternatives.choiceReturns.empty() ? nullptr : &logicalAlternatives.choiceReturns);
            childReturns->charged += summaries.work;
            childReturns->exhausted |= summaries.exhausted;
            if (proved)
                for (const auto& candidate : candidates) {
                    auto [scope, source, observer] = candidate.family;
                    (void)scope;
                    if (candidate.choice >= summaries.body.size() || !summaries.body[candidate.choice] ||
                        !((*summaries.body[candidate.choice])[source] & (1u << observer))) {
                        proved = false;
                        break;
                    }
                }
            // Reserve the complete fresh checker population before invoking it.
            // Its own structured-summary work is separately capped by the
            // remaining shared allowance and charged after it returns.
            uint64_t width = uint64_t(p.cells) * LaneCount;
            // The actual receipt population is bounded by the command count,
            // not by the proposal cache's eight alternatives. Include copies
            // during the checker's at-most-two additional subtree probes.
            uint64_t checkWidth = 4 + 3 * (width + uint64_t(p.cells) * commandCount);
            if (proved && (!childReturns->reserve(p.nodes.size(), checkWidth) ||
                           !childReturns->reserve(commandCount, 3 * p.cells + 2 * LaneCount + 8)))
                proved = false;
            c::Result checked;
            bool usedStructured = false;
            if (proved) {
                checked =
                    verifyConstructionDemands(p, result.before, nullptr, childReturns, &usedStructured, choiceIncoming);
                proved &= checked.success && usedStructured;
            }
            result.nodeVisits += checked.nodeVisits;
            result.cellVisits += checked.cellVisits;
            if (!proved) {
                result.before = std::move(saved);
                result.rejectedChildReturns = candidates.size();
                childReturns->rejected += candidates.size();
            } else {
                result.childReturnAcksRemoved = candidates.size();
                result.sharedAcknowledgments -= candidates.size();
                result.reusedAcknowledgments += candidates.size();
            }
        } else
            childReturns->rejected += candidates.size();
        result.childReturnWork = childReturns->charged;
    } else if (childReturns)
        result.childReturnWork = childReturns->charged;
    return result;
}

c::Result verifyDemandImpl(
    const c::Program& p, const std::vector<std::vector<c::Mechanism>>& actual, DemandInvariants* invariants,
    uint64_t childReturnLimit, bool* usedChildReturns, uint64_t alternativeChoiceLimit, bool* usedAlternativeChoices)
{
    using namespace c;
    c::Result result;
    if (usedChildReturns)
        *usedChildReturns = false;
    if (usedAlternativeChoices)
        *usedAlternativeChoices = false;
    DemandAnalysis analysis;
    if (actual.size() != p.nodes.size() || !analysis.build(p, result.reason))
        return result;
    analysis.recordEntryStats(result);
    for (const auto& commands : actual)
        for (const auto& m : commands)
            if (m.first >= LaneCount || m.second >= LaneCount ||
                (m.word != NoCut &&
                 (m.word >= p.nodes.size() || !p.nodes[m.word].firstActive() ||
                  p.nodes[m.word].periodicOwner != m.loop ||
                  (m.participation != Mechanism::Previous && m.participation != Mechanism::LoopExit)))) {
                result.reason = "invalid actual demand lane or periodic word";
                return result;
            }
    DeferredDemandRings deferred(p.nodes.size());
    bool inferred = deferred.infer(p, analysis, actual, result.reason);
    result.deferredEligibilityWork = deferred.eligibilityWork;
    result.deferredReceiptCells = deferred.extentWork;
    result.cellVisits += deferred.extentWork * LaneCount + deferred.eligibilityWork;
    if (!inferred) {
        result.deferredRejectionStage = "inference";
        result.deferredRejectionReason = result.reason;
        return result;
    }
    if (!deferred.verifyCombined(p, analysis, actual, result.reason)) {
        result.deferredProtocolSteps = deferred.protocolSteps;
        result.deferredRejectionStage = "protocol";
        result.deferredRejectionReason = result.reason;
        return result;
    }
    result.deferredProtocolSteps = deferred.protocolSteps;
    bool usedStructured = false, usedAlternatives = false;
    if (!verifyDemandProtocols(
            p, analysis, actual, deferred, result.reason, childReturnLimit, result.childReturnWork, usedStructured,
            result.childReturnBudgetExhausted, alternativeChoiceLimit, result.alternativeChoiceWork,
            result.alternativeChoiceFamilies, result.alternativeChoiceSites, result.alternativeChoiceSourceScopes,
            result.alternativeChoicePrefixSteps, result.alternativeChoiceBudgetExhausted, usedAlternatives)) {
        result.deferredRejectionStage = "protocol";
        result.deferredRejectionReason = result.reason;
        return result;
    }
    if (usedChildReturns)
        *usedChildReturns = usedStructured;
    if (usedAlternativeChoices)
        *usedAlternativeChoices = usedAlternatives;
    DemandRings rings(p.nodes.size());
    if (!rings.infer(p, actual, result.reason)) {
        if (!deferred.keys.empty()) {
            result.deferredRejectionStage = "inference";
            result.deferredRejectionReason = result.reason;
        }
        return result;
    }
    if (DemandRings::affordable(p))
        result.cellVisits += uint64_t(p.cells) * p.nodes.size();
    result.cutCycles = rings.cuts.cycles.size();
    result.cutCycles += deferred.cuts.cycles.size();
    std::map<unsigned, State> ringIncoming;
    // Fresh physical requirements and actual event-prefix transfer. No chosen
    // demand, source-cut proposal, family membership, or initialization receipt
    // is consumed by this checker.
    std::map<unsigned, std::set<PrefixState::EventKey>> entryProviders;
    for (unsigned id = 0; id < actual.size(); ++id)
        for (const auto& m : actual[id])
            if (m.participation == c::Mechanism::NonEmpty && m.kind == c::Mechanism::Publish &&
                m.loop < p.nodes.size() && analysis.parent[m.loop] != NoCut &&
                analysis.parent[id] == analysis.parent[m.loop] && analysis.position[id] <= analysis.position[m.loop])
                entryProviders[m.loop].insert({m.first, m.second, m.forwardKey});
    uint64_t probeBudget = 2 * p.nodes.size();
    std::function<bool(unsigned, PrefixState&, bool)> check = [&](unsigned id, PrefixState& state, bool validate) {
        ++result.nodeVisits;
        result.cellVisits += p.cells * LaneCount;
        for (const auto& m : actual[id]) {
            if (m.participation == Mechanism::Previous) {
                if (!deferred.credit(m, state, ringIncoming, result.reason))
                    return false;
            } else if (m.participation == Mechanism::LoopExit) {
                // This dynamic WAIT retires the final token only on a nonempty
                // execution. It supplies no unconditional completion credit
                // across the zero-trip join.
                continue;
            } else if (m.participation != Mechanism::Every) {
                if (!applyEntryCredit(state, m, analysis, result.reason))
                    return false;
            } else if (m.kind == Mechanism::Publish)
                state.publish(m);
            else if (m.kind == Mechanism::Acquire) {
                if (!state.receipts.count({m.first, m.second, m.forwardKey})) {
                    result.reason = "missing actual demand publication receipt";
                    return false;
                }
                state.acquirePrefix(m);
                if (!rings.credit(m, state, ringIncoming, result.reason))
                    return false;
                if (!deferred.credit(m, state, ringIncoming, result.reason))
                    return false;
            } else
                apply(state, m);
        }
        const auto& n = p.nodes[id];
        bool narrowedOwned = false;
        State ownedSeed;
        if (validate && !analysis.owned[id].empty() && analysis.subtreeNodes[id] - 1 <= probeBudget) {
            ++result.ownedRefinements;
            probeBudget -= analysis.subtreeNodes[id] - 1;
            auto probe = state;
            for (unsigned observer = 0; observer < LaneCount; ++observer)
                for (unsigned cell : analysis.owned[id])
                    probe.pending[observer][cell] = analysis.effects[id][cell];
            for (unsigned child : n.children)
                if (!check(child, probe, false))
                    return false;
            // Initial owned-cell history is empty. All later visits inherit
            // only this domain's effects; outside payload cannot invalidate
            // completion for these cells. F(Top) is an inductive visit invariant,
            // even when enclosing conditionals skip arbitrarily many visits.
            for (unsigned observer = 0; observer < LaneCount; ++observer)
                for (unsigned cell : analysis.owned[id]) {
                    auto& h = state.pending[observer][cell];
                    h.readers &= probe.pending[observer][cell].readers;
                    h.writers &= probe.pending[observer][cell].writers;
                }
            ownedSeed = probe;
            narrowedOwned = true;
            if (invariants)
                (*invariants)[id] = ownedSeed;
        }
        if (n.kind == Node::Operation) {
            if (validate &&
                (!visibilityCovered(p, n, state, actual[id]) || state.demands(n.lane, n.effects))) {
                result.reason = "uncovered physical demand or GM visibility requirement at node " + std::to_string(id) +
                                " lane " + std::to_string(n.lane) + " demand " +
                                std::to_string(state.demands(n.lane, n.effects));
                return false;
            }
            state.seed(n.effects);
            return true;
        }
        if (n.kind == Node::Choice) {
            auto other = state;
            if (!check(n.children[0], state, validate) || !check(n.children[1], other, validate))
                return false;
            state.join(other);
            return true;
        }
        if (n.kind == Node::For || n.kind == Node::While) {
            // NonEmpty entry providers are guarded by the loop trip predicate.
            // They must not survive the zero-trip branch of this abstract join;
            // ordinary receipts and ordinary commands at the same cut do.
            auto entry = state;
            if (rings.loopOwners.count(id) || deferred.loopOwners.count(id))
                ringIncoming.insert_or_assign(id, entry);
            for (const auto& receipt : entryProviders[id])
                entry.receipts.erase(receipt);
            state.seed(analysis.effects[id]);
            // Bounded invariant narrowing for a FIXED actual plan. Top=E|S
            // is inductive. Monotonicity gives E|F(Top) <= Top, itself an
            // inductive invariant. The probe checks no hazards and recursively
            // uses only the conservative rule; the real visit below proves all
            // accesses and checks closure. At most two extra whole-tree visits
            // are allowed across ALL nested loops, not two visits per depth.
            uint64_t cost = analysis.subtreeNodes[id] - 1;
            if (validate && cost <= probeBudget) {
                probeBudget -= cost;
                auto probe = state;
                for (unsigned child : n.children)
                    if (!check(child, probe, false))
                        return false;
                for (unsigned observer = 0; observer < LaneCount; ++observer)
                    for (unsigned cell = 0; cell < p.cells; ++cell) {
                        auto& h = state.pending[observer][cell];
                        h.readers = entry.pending[observer][cell].readers | probe.pending[observer][cell].readers;
                        h.writers = entry.pending[observer][cell].writers | probe.pending[observer][cell].writers;
                    }
            }
            State seed = state;
            if (validate && invariants)
                (*invariants)[id] = seed;
            if (!check(n.children[0], state, validate))
                return false;
            auto exit = state;
            if (n.kind == Node::While) {
                if (!check(n.children[1], state, validate))
                    return false;
            }
            if (validate)
                for (unsigned observer = 0; observer < LaneCount; ++observer)
                    for (unsigned cell = 0; cell < p.cells; ++cell) {
                        const auto& out = state.pending[observer][cell];
                        const auto& in = seed.pending[observer][cell];
                        if ((out.readers & ~in.readers) || (out.writers & ~in.writers)) {
                            result.reason = "demand loop completion invariant did not close";
                            return false;
                        }
                    }
            if (n.kind == Node::While)
                state = std::move(exit);
            else
                state.join(entry);
            ringIncoming.erase(id);
            return true;
        }
        for (unsigned child : n.children)
            if (!check(child, state, validate))
                return false;
        if (narrowedOwned)
            for (unsigned observer = 0; observer < LaneCount; ++observer)
                for (unsigned cell : analysis.owned[id]) {
                    const auto& out = state.pending[observer][cell];
                    const auto& in = ownedSeed.pending[observer][cell];
                    if ((out.readers & ~in.readers) || (out.writers & ~in.writers)) {
                        result.reason = "owned-cell completion invariant did not close";
                        return false;
                    }
                }
        return true;
    };
    PrefixState state(p.cells);
    result.success = check(p.nodes.size() - 1, state, true);
    if (!result.success && !deferred.keys.empty()) {
        result.deferredRejectionStage = "physical";
        result.deferredRejectionReason = result.reason;
    }
    return result;
}
} // namespace

namespace {
c::Result verifyConstructionDemands(
    const c::Program& p, const Commands& actual, DemandInvariants* invariants, const ChildReturnOptions* childReturns,
    bool* usedChildReturns, const DemandAnalysis::ChoiceIncomingOptions* choiceIncoming)
{
    uint64_t childRemaining = 0;
    if (childReturns && childReturns->enabled) {
        ++childReturns->checks;
        uint64_t allowance = std::min(childReturns->limit, ChildReturnSummaries::MaxWork);
        childRemaining = childReturns->exhausted ? 0 : allowance - childReturns->charged;
    }
    uint64_t alternativeRemaining = 0;
    if (choiceIncoming && choiceIncoming->alternativesEnabled && !choiceIncoming->exhausted &&
        choiceIncoming->charged <= choiceIncoming->limit)
        alternativeRemaining = choiceIncoming->limit - choiceIncoming->charged;
    auto checked = verifyDemandImpl(p, actual, invariants, childRemaining, usedChildReturns, alternativeRemaining);
    if (childReturns && checked.childReturnWork > childRemaining) {
        checked.success = false;
        checked.childReturnBudgetExhausted = true;
    } else if (childReturns)
        childReturns->charged += checked.childReturnWork;
    if (childReturns && checked.childReturnBudgetExhausted) {
        childReturns->exhausted = true;
        if (!childReturns->budgetCheck)
            childReturns->budgetCheck = childReturns->checks;
    }
    if (choiceIncoming && choiceIncoming->alternativesEnabled) {
        uint64_t used = checked.alternativeChoiceWork + checked.childReturnWork;
        // Charge partial proof work independently of the eventual proof status.
        if (used <= alternativeRemaining)
            choiceIncoming->charged += used;
        if (used > alternativeRemaining || checked.alternativeChoiceBudgetExhausted ||
            checked.childReturnBudgetExhausted) {
            choiceIncoming->exhausted = true;
            checked.success = false;
        }
    }
    return checked;
}

c::Result constructDemandsImpl(
    const c::Program& p, bool corruptRefinement, bool replayAllocation = true, bool corruptReplay = false,
    bool corruptEntry = false, bool allowEntryFallback = true,
    const DemandAnalysis::ChoiceIncomingOptions* choiceIncoming = nullptr,
    const ChildReturnOptions* childReturns = nullptr)
{
    DemandFallbacks unassigned;
    auto initial = constructDemandCandidate(p, nullptr, unassigned, nullptr, nullptr, choiceIncoming, childReturns);
    if (corruptEntry)
        for (auto& commands : initial.before)
            commands.erase(
                std::remove_if(
                    commands.begin(), commands.end(),
                    [](const auto& m) { return m.participation == c::Mechanism::First; }),
                commands.end());
    DemandInvariants invariants;
    auto checked = initial.success ? verifyConstructionDemands(
                                         p, initial.before, &invariants, childReturns, nullptr, choiceIncoming) :
                                     c::Result{};
    if (!initial.success || !checked.success) {
        bool hasEntryContract =
            std::any_of(p.nodes.begin(), p.nodes.end(), [](const auto& n) { return n.entryGuardStart != NoCut; });
        if (allowEntryFallback && hasEntryContract) {
            auto conservativeEntry = p;
            for (auto& n : conservativeEntry.nodes)
                n.entryGuardStart = NoCut;
            auto fallback = constructDemandsImpl(
                conservativeEntry, corruptRefinement, replayAllocation, corruptReplay, false, false, choiceIncoming,
                childReturns);
            fallback.nodeVisits += initial.nodeVisits + checked.nodeVisits;
            fallback.cellVisits += initial.cellVisits + checked.cellVisits;
            fallback.rejectedEntryProposals = 1;
            return fallback;
        }
        if (initial.success)
            initial.reason = checked.reason;
        initial.success = false;
        initial.before.clear();
        return initial;
    }
    initial.cellVisits += checked.cellVisits;
    initial.nodeVisits += checked.nodeVisits;
    const DemandInvariants* selectedInvariants = nullptr;
    // Exactly one optional reconstruction using already proved prefix
    // invariants. The candidate must prove its OWN invariant independently.
    // This happens before native emission; a failed emitted-IR checker is never
    // converted into fallback success.
    if (!invariants.empty()) {
        DemandFallbacks refinedUnassigned;
        auto refined =
            constructDemandCandidate(p, &invariants, refinedUnassigned, nullptr, nullptr, choiceIncoming, childReturns);
        if (corruptRefinement)
            for (auto& commands : refined.before)
                commands.clear();
        auto rechecked = refined.success ? verifyConstructionDemands(
                                               p, refined.before, nullptr, childReturns, nullptr, choiceIncoming) :
                                           c::Result{};
        if (refined.success && rechecked.success) {
            refined.cellVisits += initial.cellVisits + rechecked.cellVisits;
            refined.nodeVisits += initial.nodeVisits + rechecked.nodeVisits;
            initial = std::move(refined);
            unassigned = std::move(refinedUnassigned);
            selectedInvariants = &invariants;
        } else {
            initial.cellVisits += refined.cellVisits + rechecked.cellVisits;
            initial.nodeVisits += refined.nodeVisits + rechecked.nodeVisits;
            initial.rejectedRefinements = 1;
        }
        initial.completionRefinements = 1;
    }
    if (unassigned.empty() || corruptRefinement || !replayAllocation)
        return initial;

    // One replay of the same constructor, with the same structural requests
    // and invariant hints. Never recurse or chase allocation to a fixed point.
    // Its only new information is which original forward demands must use the
    // existing canonical acquire() transfer during construction, rather than
    // after numbering. Later consumers can reuse the resulting completion.
    DemandFallbacks newlyUnassigned;
    auto replay = constructDemandCandidate(
        p, selectedInvariants, newlyUnassigned, &unassigned, nullptr, choiceIncoming, childReturns);
    if (corruptReplay)
        for (auto& commands : replay.before)
            commands.clear();
    auto verified = replay.success ?
                        verifyConstructionDemands(p, replay.before, nullptr, childReturns, nullptr, choiceIncoming) :
                        c::Result{};
    bool acceptable =
        replay.success && verified.success &&
        std::includes(unassigned.begin(), unassigned.end(), newlyUnassigned.begin(), newlyUnassigned.end());
    if (acceptable)
        for (unsigned id = 0; id < p.nodes.size(); ++id) {
            auto guarded = [](const auto& commands) {
                std::vector<c::Mechanism> out;
                for (const auto& m : commands)
                    if (m.participation != c::Mechanism::Every)
                        out.push_back(m);
                return out;
            };
            if (guarded(initial.before[id]) != guarded(replay.before[id])) {
                acceptable = false;
                break;
            }
        }
    uint64_t removed = 0;
    if (acceptable) {
        // Every current command participates on every visit to its immediate
        // Sequence. Non-increasing command cost per scope therefore protects
        // independently varying loop/branch executions, not just static totals.
        for (const auto& n : p.nodes) {
            if (n.kind != c::Node::Sequence)
                continue;
            auto cost = [&](const c::Result& r) {
                uint64_t total = 0;
                for (unsigned child : n.children)
                    for (const auto& m : r.before[child])
                        total += commandCost(m);
                return total;
            };
            auto before = cost(initial), after = cost(replay);
            if (after > before) {
                acceptable = false;
                break;
            }
            removed += before - after;
        }
    }
    initial.allocationReplays = 1;
    if (acceptable) {
        replay.cellVisits += initial.cellVisits + verified.cellVisits;
        replay.nodeVisits += initial.nodeVisits + verified.nodeVisits;
        replay.completionRefinements = initial.completionRefinements;
        replay.rejectedRefinements = initial.rejectedRefinements;
        replay.allocationReplays = 1;
        replay.replayCommandsRemoved = removed;
        return replay;
    }
    initial.cellVisits += replay.cellVisits + verified.cellVisits;
    initial.nodeVisits += replay.nodeVisits + verified.nodeVisits;
    initial.rejectedAllocationReplays = 1;
    return initial;
}
} // namespace

namespace {
c::Result constructWithDemandRings(
    const c::Program& p, bool corrupt, const DemandAnalysis::ChoiceIncomingOptions* choiceIncoming = nullptr,
    const ChildReturnOptions* childReturns = nullptr)
{
    using namespace c;
    auto baseline = constructDemandsImpl(p, false, true, false, false, true, choiceIncoming, childReturns);
    if (!baseline.success)
        return baseline;
    DemandRings shapes(p.nodes.size());
    std::string why;
    if (DemandRings::affordable(p))
        baseline.cellVisits += uint64_t(p.cells) * p.nodes.size();
    if (!shapes.discover(p, why) || shapes.cuts.cycles.empty())
        return baseline;
    baseline.ringCandidates = 1;
    // Cheap necessary resource and local savings checks precede construction.
    // They may forgo a useful candidate, but never affect baseline admission.
    std::map<unsigned, uint64_t> minimum;
    std::map<std::pair<unsigned, unsigned>, unsigned> required;
    std::set<PrefixState::EventKey> entryKeys;
    for (const auto& commands : baseline.before)
        for (const auto& m : commands)
            if (m.participation != Mechanism::Every)
                entryKeys.insert({m.first, m.second, m.forwardKey});
    for (const auto& cycle : shapes.cuts.cycles) {
        minimum[cycle.region] += 2 * cycle.groups.size();
        for (unsigned g = 0; g < cycle.groups.size(); ++g)
            ++required[{cycle.groups[g].lane, cycle.groups[(g + 1) % cycle.groups.size()].lane}];
    }
    bool useful = false, feasible = true;
    for (const auto& [scope, lower] : minimum) {
        uint64_t old = 0;
        for (unsigned child : p.nodes[scope].children)
            for (const auto& m : baseline.before[child])
                if (m.participation == Mechanism::Every)
                    old += commandCost(m);
        feasible &= lower <= old;
        useful |= lower < old;
    }
    for (const auto& [direction, count] : required) {
        unsigned available = 0;
        for (unsigned k : p.target.compilerKeys)
            available += key(p, direction.first, direction.second) != k &&
                         p.target.available(lane(p, direction.first), lane(p, direction.second), k) &&
                         !entryKeys.count({direction.first, direction.second, k});
        feasible &= count <= available;
    }
    if (!useful || !feasible) {
        baseline.rejectedRings = 1;
        return baseline;
    }
    DemandFallbacks unassigned;
    auto candidate =
        constructDemandCandidate(p, nullptr, unassigned, nullptr, &shapes.cuts, choiceIncoming, childReturns);
    if (corrupt)
        for (auto& commands : candidate.before)
            commands.clear();
    auto checked = candidate.success ?
                       verifyConstructionDemands(p, candidate.before, nullptr, childReturns, nullptr, choiceIncoming) :
                       c::Result{};
    bool accepted = candidate.success && checked.success;
    uint64_t removed = 0;
    if (accepted)
        for (unsigned id = 0; id < p.nodes.size(); ++id) {
            // Preserve every conditional entry episode exactly. For ordinary
            // commands require non-increase in EACH independently executed
            // Sequence, not an amortized trip-count or whole-function total.
            auto guarded = [](const auto& commands) {
                std::vector<Mechanism> out;
                for (const auto& m : commands)
                    if (m.participation != Mechanism::Every)
                        out.push_back(m);
                return out;
            };
            if (guarded(baseline.before[id]) != guarded(candidate.before[id])) {
                accepted = false;
                break;
            }
            if (p.nodes[id].kind != Node::Sequence)
                continue;
            auto cost = [&](const c::Result& plan) {
                uint64_t total = 0;
                for (unsigned child : p.nodes[id].children)
                    for (const auto& m : plan.before[child])
                        if (m.participation == Mechanism::Every)
                            total += commandCost(m);
                return total;
            };
            auto old = cost(baseline), next = cost(candidate);
            if (next > old) {
                accepted = false;
                break;
            }
            removed += old - next;
        }
    baseline.nodeVisits += checked.nodeVisits;
    baseline.cellVisits += checked.cellVisits;
    baseline.ringCandidates = 1;
    if (!accepted || !removed) {
        baseline.rejectedRings = 1;
        return baseline;
    }
    candidate.nodeVisits = baseline.nodeVisits;
    candidate.cellVisits = baseline.cellVisits;
    candidate.ringCandidates = 1;
    candidate.ringCandidateCommandsRemoved = removed;
    return candidate;
}

std::optional<uint64_t> discoveryReservation(
    uint64_t nodes, uint64_t cells, uint64_t keys, uint64_t commands, uint64_t limit)
{
    // Establish small operand bounds before EVERY subsequent product/sum.
    // NC <= 2^20, K <= 64 and log(NC),log(commands) <= 21 keep all
    // intermediate widths below 2^18. Only reserve() multiplies a width by
    // population, after division against the remaining allowance.
    if (!nodes || !cells || nodes > DemandRings::MaxCells || cells > DemandRings::MaxCells / nodes || keys > 64 ||
        commands > DemandRings::MaxCells || limit > c::DeferredDiscoveryLimit)
        return {};
    const uint64_t scanCells = cells * nodes;
    auto logarithm = [](uint64_t size) {
        uint64_t bits = 1;
        while (size >>= 1)
            ++bits;
        return bits;
    };
    uint64_t work = 0;
    auto reserve = [&](uint64_t count, uint64_t width) {
        if (width && count > (limit - work) / width)
            return false;
        work += count * width;
        return true;
    };
    const uint64_t wordWidth = MaxGroups * (MaxAlternatives * MaxAlternatives + 2 * MaxAlternatives + keys);
    if (!reserve(nodes, 8) || !reserve(scanCells, wordWidth + 8 * c::LaneCount * (1 + logarithm(scanCells)) + 64) ||
        !reserve(cells, MaxGroups * (keys + 1) * (MaxGroups + logarithm(scanCells))) ||
        !reserve(commands, 32 + 4 * logarithm(commands)))
        return {};
    return work;
}

c::Result constructWithDeferredRings(
    const c::Program& p, bool corrupt, uint64_t discoveryLimit = c::DeferredDiscoveryLimit,
    const DemandAnalysis::ChoiceIncomingOptions* choiceIncoming = nullptr,
    const ChildReturnOptions* childReturns = nullptr)
{
    using namespace c;
    // The starting point is already independently verified by the existing
    // closed-ring selector. Deferred wrap never rescues a rejected ring shape
    // and never changes ordinary fallback mechanisms.
    auto baseline = constructWithDemandRings(p, false, choiceIncoming, childReturns);
    if (!baseline.success || baseline.cutCycles == 0 || !DemandRings::affordable(p))
        return baseline;
    // Reserve representation work BEFORE rerunning discovery/inference. The
    // word scan has bounded groups/alternatives, at most one cycle per cell,
    // and fixed-lane demand scans; signature/key matching and command sorting
    // are charged separately. This is a conservative reservation, not a CPU
    // instruction count or wall-clock bound. Overflow/limit refusal leaves the
    // already verified baseline unchanged.
    const uint64_t scanNodes = p.nodes.size();
    uint64_t commands = 0;
    for (const auto& site : baseline.before) {
        if (site.size() > (1u << 20) - commands) {
            ++baseline.deferredDiscoveryRefusals;
            return baseline;
        }
        commands += site.size();
    }
    auto discoveryWork =
        discoveryReservation(scanNodes, p.cells, p.target.compilerKeys.size(), commands, discoveryLimit);
    if (!discoveryWork) {
        ++baseline.deferredDiscoveryRefusals;
        return baseline;
    }
    baseline.deferredDiscoveryWork += *discoveryWork;
    baseline.cellVisits += *discoveryWork;
    baseline.nodeVisits += 2 * scanNodes;
    DemandAnalysis analysis;
    DemandRings closed(p.nodes.size());
    std::string reason;
    if (!analysis.build(p, reason) || !closed.infer(p, baseline.before, reason))
        return baseline;
    auto following = DeferredDemandRings::followingCommands(p, baseline.before);
    std::vector<Cycle> eligible;
    uint64_t eligibilityWork = 0;
    for (const auto& cycle : closed.cuts.cycles) {
        if (!DeferredDemandRings::chargeEligibility(p, eligibilityWork)) {
            baseline.cellVisits += eligibilityWork;
            baseline.deferredEligibilityWork += eligibilityWork;
            return baseline;
        }
        if (DeferredDemandRings::eligible(p, analysis, cycle, &baseline.periodicWriteOverlapRejections) &&
            DeferredDemandRings::continuationSafe(p, analysis, cycle, following))
            eligible.push_back(cycle);
    }
    baseline.cellVisits += eligibilityWork;
    baseline.deferredEligibilityWork += eligibilityWork;
    if (eligible.empty())
        return baseline;

    // Select a bounded population of complete, disjoint witness/key families
    // before trying emission. One combined check accepts this subset or the
    // original baseline; no per-family proof/backtracking campaign is run.
    constexpr unsigned MaxDeferredFamilies = 8;
    std::vector<Cycle> selected;
    std::set<unsigned> selectedCells;
    std::set<PrefixState::EventKey> selectedKeys;
    for (auto& cycle : eligible) {
        bool disjoint = selected.size() < MaxDeferredFamilies;
        for (unsigned cell : cycle.cells)
            disjoint &= !selectedCells.count(cell);
        for (auto [direction, eventKey] : cycle.keys)
            disjoint &= !selectedKeys.count({direction.first, direction.second, eventKey});
        if (!disjoint) {
            ++baseline.deferredSkippedFamilies;
            continue;
        }
        selectedCells.insert(cycle.cells.begin(), cycle.cells.end());
        for (auto [direction, eventKey] : cycle.keys)
            selectedKeys.insert({direction.first, direction.second, eventKey});
        selected.push_back(std::move(cycle));
    }
    eligible = std::move(selected);

    baseline.deferredRingCandidates = eligible.size();
    auto candidate = baseline;
    Commands moved(p.nodes.size()), exits(p.nodes.size());
    bool transformed = true;
    auto matches = [](const Mechanism& mechanism, Mechanism::Kind kind, unsigned source, unsigned target,
                      unsigned eventKey) {
        return mechanism.kind == kind && mechanism.participation == Mechanism::Every && mechanism.loop == NoCut &&
               mechanism.first == source && mechanism.second == target && mechanism.forwardKey == eventKey;
    };
    for (const auto& cycle : eligible) {
        const auto& firstGroup = cycle.groups.front();
        const auto& lastGroup = cycle.groups.back();
        unsigned source = lastGroup.lane, target = firstGroup.lane;
        unsigned eventKey = cycle.keys.at({source, target});
        unsigned firstCut = firstGroup.first.front();
        unsigned lastCut = closed.cuts.next[lastGroup.last.front()];
        unsigned exitCut = analysis.next[cycle.owner];
        auto& firstCommands = candidate.before[firstCut];
        auto publication = std::find_if(firstCommands.begin(), firstCommands.end(), [&](const auto& mechanism) {
            return matches(mechanism, Mechanism::Publish, source, target, eventKey);
        });
        auto previous = std::find_if(firstCommands.begin(), firstCommands.end(), [&](const auto& mechanism) {
            return matches(mechanism, Mechanism::Acquire, source, target, eventKey);
        });
        if (publication == firstCommands.end() || previous == firstCommands.end() || lastCut == NoCut ||
            exitCut == NoCut) {
            transformed = false;
            break;
        }
        previous->participation = Mechanism::Previous;
        previous->loop = cycle.owner;
        previous->word = DeferredDemandRings::pWord(cycle, analysis) ? cycle.region : NoCut;
        firstCommands.erase(publication);
        moved[lastCut].push_back(event(Mechanism::Publish, source, target, eventKey));
        auto finalWait = event(Mechanism::Acquire, source, target, eventKey);
        finalWait.participation = Mechanism::LoopExit;
        finalWait.loop = cycle.owner;
        finalWait.word = DeferredDemandRings::pWord(cycle, analysis) ? cycle.region : NoCut;
        exits[exitCut].push_back(finalWait);
    }
    if (transformed)
        for (unsigned id = 0; id < p.nodes.size(); ++id) {
            // The moved SET is the first command after the last group. It
            // snapshots no later unrelated source prefix. The final WAIT is
            // likewise the first command at the original post-loop anchor.
            candidate.before[id].insert(candidate.before[id].begin(), moved[id].begin(), moved[id].end());
            candidate.before[id].insert(candidate.before[id].begin(), exits[id].begin(), exits[id].end());
        }
    if (corrupt && transformed)
        for (auto& commands : candidate.before) {
            auto wait = std::find_if(commands.begin(), commands.end(), [](const auto& mechanism) {
                return mechanism.participation == Mechanism::LoopExit;
            });
            if (wait != commands.end()) {
                commands.erase(wait);
                break;
            }
        }
    auto checked = transformed ?
                       verifyConstructionDemands(p, candidate.before, nullptr, childReturns, nullptr, choiceIncoming) :
                       c::Result{};
    baseline.nodeVisits += checked.nodeVisits;
    baseline.cellVisits += checked.cellVisits;
    baseline.deferredProtocolSteps += checked.deferredProtocolSteps;
    baseline.deferredEligibilityWork += checked.deferredEligibilityWork;
    baseline.deferredReceiptCells += checked.deferredReceiptCells;
    if (!transformed || !checked.success) {
        baseline.deferredRejectionStage = transformed ? checked.deferredRejectionStage : "placement";
        baseline.deferredRejectionReason =
            transformed ? checked.deferredRejectionReason : "closed-ring endpoints could not be moved exactly";
        if (baseline.deferredRejectionStage.empty())
            baseline.deferredRejectionStage = "physical";
        if (baseline.deferredRejectionReason.empty())
            baseline.deferredRejectionReason = checked.reason;
        baseline.rejectedDeferredRings = eligible.size();
        return baseline;
    }
    candidate.nodeVisits = baseline.nodeVisits;
    candidate.cellVisits = baseline.cellVisits;
    candidate.deferredProtocolSteps = baseline.deferredProtocolSteps;
    candidate.deferredEligibilityWork = baseline.deferredEligibilityWork;
    candidate.deferredReceiptCells = baseline.deferredReceiptCells;
    candidate.deferredRingCandidates = eligible.size();
    candidate.deferredRings = eligible.size();
    candidate.periodicDeferredRings = std::count_if(eligible.begin(), eligible.end(), [&](const auto& cycle) {
        return DeferredDemandRings::pWord(cycle, analysis);
    });
    return candidate;
}

c::Result constructWithLateEntry(
    const c::Program& p, bool corrupt, uint64_t limit = 1u << 22,
    const DemandAnalysis::ChoiceIncomingOptions* choiceIncoming = nullptr,
    const ChildReturnOptions* childReturns = nullptr)
{
    using namespace c;
    auto baseline = constructWithDeferredRings(p, false, c::DeferredDiscoveryLimit, choiceIncoming, childReturns);
    if (!baseline.success || !baseline.entryEpisodes || baseline.before.size() != p.nodes.size())
        return baseline;
    // Two aggregate allowances cover discovery and the worst-case single copy
    // plus eight placements for every command. Reserve each stage before its
    // scans/allocations; no per-family budget can admit an unbounded aggregate.
    constexpr uint64_t MaxLateEntryWork = 1u << 22;
    if (limit > MaxLateEntryWork)
        limit = MaxLateEntryWork;
    uint64_t work = 0;
    auto reserve = [&](uint64_t count, uint64_t width) {
        if (work > limit || (width && count > (limit - work) / width))
            return false;
        work += count * width;
        return true;
    };
    uint64_t commands = 0;
    if (!reserve(p.nodes.size(), 1))
        return baseline;
    for (const auto& site : baseline.before) {
        if (site.size() > MaxLateEntryWork - commands)
            return baseline;
        commands += site.size();
    }
    if (!reserve(commands, 1) || !reserve(baseline.demands.size(), 1))
        return baseline;
    uint64_t demandCells = 0;
    for (const auto& demand : baseline.demands) {
        if (demand.cells.size() > MaxLateEntryWork - demandCells)
            return baseline;
        demandCells += demand.cells.size();
    }
    baseline.cellVisits += work;
    bool hasCommonChoice = false;
    for (unsigned id = 0; id < baseline.before.size() && !hasCommonChoice; ++id)
        hasCommonChoice = p.nodes[id].kind == Node::Choice &&
                          std::any_of(baseline.before[id].begin(), baseline.before[id].end(), [](const auto& m) {
                              return m.kind == Mechanism::Acquire && m.participation == Mechanism::First;
                          });
    if (!hasCommonChoice)
        return baseline;
    uint64_t prior = work;
    uint64_t cells = p.cells;
    if (!reserve(p.nodes.size(), 16) || !reserve(p.nodes.size(), cells * LaneCount) ||
        !reserve(commands, MaxAlternatives + 4) || !reserve(baseline.demands.size(), 8) || !reserve(demandCells, 1))
        return baseline;
    baseline.cellVisits += work - prior;
    DemandAnalysis analysis;
    std::string reason;
    if (!analysis.build(p, reason))
        return baseline;
    std::vector<uint8_t> hasRecurrence(p.nodes.size());
    for (unsigned id = 0; id < p.nodes.size(); ++id) {
        hasRecurrence[id] = p.nodes[id].kind == Node::For || p.nodes[id].kind == Node::While;
        for (unsigned child : p.nodes[id].children)
            hasRecurrence[id] |= hasRecurrence[child];
    }
    struct Family {
        unsigned origin;
        Mechanism wait;
        std::vector<unsigned> sites;
    };
    std::vector<Family> families;
    for (unsigned id = 0; id < baseline.before.size(); ++id) {
        if (p.nodes[id].kind != Node::Choice)
            continue;
        for (const auto& mechanism : baseline.before[id]) {
            if (mechanism.kind != Mechanism::Acquire || mechanism.participation != Mechanism::First)
                continue;
            const auto* demand = analysis.entryDemand(mechanism.loop, id, mechanism.first, mechanism.second);
            const auto* summary = analysis.firstSummary(id, mechanism.second);
            if (!demand || !summary || !summary->valid || summary->mayNoLane || !summary->count ||
                summary->count > MaxAlternatives || hasRecurrence[id])
                continue;
            prior = work;
            if (!reserve(summary->count, cells + 2))
                return baseline;
            baseline.cellVisits += work - prior;
            Family family{id, mechanism, {}};
            family.sites.assign(summary->operations.begin(), summary->operations.begin() + summary->count);
            if (std::any_of(family.sites.begin(), family.sites.end(), [&](unsigned site) {
                    return site >= p.nodes.size() || p.nodes[site].kind != Node::Operation ||
                           p.nodes[site].lane != mechanism.second ||
                           std::none_of(
                               p.nodes[site].effects.begin(), p.nodes[site].effects.end(), [&](const auto& effect) {
                                   return (effect.readers | effect.writers) & (1u << mechanism.second);
                               });
                }))
                continue;
            families.push_back(std::move(family));
        }
    }
    if (families.empty())
        return baseline;
    baseline.lateEntryCandidates = 1;
    auto candidate = baseline;
    std::vector<std::vector<Mechanism>> moved(p.nodes.size());
    bool transformed = true;
    uint64_t sites = 0;
    for (const auto& family : families) {
        auto& common = candidate.before[family.origin];
        auto found = std::find(common.begin(), common.end(), family.wait);
        if (found == common.end()) {
            transformed = false;
            break;
        }
        common.erase(found);
        for (unsigned site : family.sites)
            moved[site].push_back(family.wait);
        sites += family.sites.size();
    }
    if (transformed)
        for (unsigned id = 0; id < moved.size(); ++id)
            candidate.before[id].insert(candidate.before[id].begin(), moved[id].begin(), moved[id].end());
    if (corrupt && transformed) {
        for (const auto& family : families) {
            auto& commandsAtSite = candidate.before[family.sites.front()];
            auto found = std::find(commandsAtSite.begin(), commandsAtSite.end(), family.wait);
            if (found != commandsAtSite.end()) {
                commandsAtSite.erase(found);
                break;
            }
        }
    }
    auto checked = transformed ?
                       verifyConstructionDemands(p, candidate.before, nullptr, childReturns, nullptr, choiceIncoming) :
                       c::Result{};
    baseline.nodeVisits += checked.nodeVisits;
    baseline.cellVisits += checked.cellVisits;
    if (!transformed || !checked.success) {
        baseline.rejectedLateEntryFamilies = families.size();
        return baseline;
    }
    candidate.nodeVisits = baseline.nodeVisits;
    candidate.cellVisits = baseline.cellVisits;
    candidate.lateEntryCandidates = 1;
    candidate.lateEntryFamilies = families.size();
    candidate.lateEntrySites = sites;
    return candidate;
}

c::Result constructChoiceTransaction(
    const c::Program& p, c::Result baseline, DemandAnalysis::ChoiceIncomingOptions& options,
    const ChildReturnOptions* childReturns)
{
    using namespace c;
    if (!baseline.success || !options.enabled || !options.limit)
        return baseline;
    const bool universalPhase = options.alternativesEnabled;
    const uint64_t allowance = std::min(options.limit, uint64_t(DemandAnalysis::MaxChoiceIncomingWork));
    uint64_t work = 0;
    auto charge = [&](uint64_t amount) {
        if (amount > allowance - work)
            return false;
        work += amount;
        return true;
    };
    if (!charge(p.nodes.size()))
        return baseline;
    bool hasShape = false;
    for (const auto& node : p.nodes) {
        if (node.kind != Node::Sequence || node.children.size() < 2)
            continue;
        hasShape |= std::any_of(node.children.begin() + 1, node.children.end(), [&](unsigned child) {
            return p.nodes[child].kind == Node::Choice;
        });
    }
    if (!hasShape)
        return baseline;

    // Reserve represented state work before rerunning optional construction.
    // This covers a fixed number of constructor/checker passes and their
    // node/cell/receipt populations, not an exact allocator-byte or time bound.
    uint64_t commands = 0;
    for (const auto& group : baseline.before)
        commands += group.size();
    const uint64_t width = uint64_t(p.cells) * LaneCount;
    const uint64_t perNode = 8 * (1 + width * (MaxAlternatives + 1));
    const uint64_t perCommand = 4 * (1 + width);
    if (p.nodes.size() > (allowance - work) / perNode || !charge(p.nodes.size() * perNode) ||
        commands > (allowance - work) / perCommand || !charge(commands * perCommand))
        return baseline;

    options.limit = allowance - work;
    options.baselineDemands = &baseline.demands;
    const uint64_t initialReservation = work;
    auto recordChoiceWork = [&](c::Result& result) {
        if (universalPhase) {
            result.alternativeChoiceWork = work + options.charged;
            result.alternativeChoiceBudgetExhausted = options.exhausted;
        } else {
            result.choiceDemandWork = work + options.charged;
            result.choiceDemandReservedWork = initialReservation;
            result.choiceDemandAnalysisWork = options.charged;
            result.choiceDemandAnalysisPasses = options.passes;
            result.choiceDemandBudgetPass = options.exhaustionPass;
        }
    };
    DemandAnalysis proposed;
    std::string proposalReason;
    bool proposedOK = proposed.build(p, proposalReason, &options);
    recordChoiceWork(baseline);
    if (options.exhausted || !proposedOK ||
        (universalPhase ? proposed.universalChoices.empty() : !proposed.choiceDemandCandidates))
        return baseline;
    auto candidate = constructWithLateEntry(p, false, 1u << 22, &options, childReturns);
    recordChoiceWork(baseline);
    if (options.exhausted || (universalPhase && childReturns && childReturns->exhausted)) {
        if (universalPhase) {
            baseline.alternativeChoiceCandidates = proposed.universalChoices.size();
            baseline.rejectedAlternativeChoices = proposed.universalChoices.size();
            baseline.alternativeChoiceBudgetExhausted = true;
        } else {
            baseline.choiceDemandCandidates = proposed.choiceDemandCandidates;
            baseline.rejectedChoiceDemands = proposed.choiceDemandCandidates;
        }
        baseline.nodeVisits += candidate.nodeVisits;
        baseline.cellVisits += candidate.cellVisits;
        return baseline;
    }
    if (candidate.success && (candidate.choiceDemandFamilies || candidate.alternativeChoiceFamilies)) {
        uint64_t candidateCommands = 0;
        for (const auto& group : candidate.before)
            candidateCommands += group.size();
        // Charge the extra fresh check and the bounded baseline-benefit,
        // ancestry, unchanged-region and per-owner path-delta scans separately.
        const uint64_t nodes = p.nodes.size();
        uint64_t finalWork = nodes * (1 + width) + candidateCommands * (1 + width);
        finalWork += nodes * nodes;
        finalWork += uint64_t(candidate.demands.size()) * (nodes + MaxAlternatives * (baseline.demands.size() + nodes));
        if (finalWork > allowance - work - options.charged) {
            if (universalPhase) {
                baseline.alternativeChoiceCandidates = proposed.universalChoices.size();
                baseline.rejectedAlternativeChoices = proposed.universalChoices.size();
                baseline.alternativeChoiceBudgetExhausted = true;
            } else {
                baseline.choiceDemandCandidates = proposed.choiceDemandCandidates;
                baseline.rejectedChoiceDemands = proposed.choiceDemandCandidates;
            }
            baseline.nodeVisits += candidate.nodeVisits;
            baseline.cellVisits += candidate.cellVisits;
            return baseline;
        }
        work += finalWork;
        options.limit -= finalWork;
    }
    bool hasCandidate =
        candidate.success && (universalPhase ? candidate.alternativeChoiceFamilies == proposed.universalChoices.size() :
                                               candidate.choiceDemandFamilies != 0);
    if (candidate.success && candidate.alternativeChoiceFamilies) {
        auto sameDemand = [](const auto& left, const auto& right) {
            return left.scope == right.scope && left.publication == right.publication &&
                   left.acquisition == right.acquisition && left.source == right.source &&
                   left.observer == right.observer && left.cells == right.cells;
        };
        for (const auto& demand : candidate.demands) {
            bool promoted = std::any_of(
                proposed.universalChoices.begin(), proposed.universalChoices.end(), [&](const auto& family) {
                    return std::any_of(family.alternatives.begin(), family.alternatives.end(), [&](const auto& old) {
                        return old.acquisition == demand.acquisition && old.source == demand.source &&
                               old.observer == demand.observer && old.cells == demand.cells;
                    });
                });
            bool existed = std::any_of(baseline.demands.begin(), baseline.demands.end(), [&](const auto& old) {
                return sameDemand(old, demand);
            });
            candidate.alternativeChoiceContinuationDemands += !promoted && !existed;
        }
    }
    auto checked = hasCandidate ?
                       verifyConstructionDemands(p, candidate.before, nullptr, childReturns, nullptr, &options) :
                       c::Result{};
    bool accepted = hasCandidate && checked.success;
    if (accepted && universalPhase)
        for (const auto& old : baseline.demands) {
            if (p.nodes[old.acquisition].kind != Node::Choice)
                continue;
            // Recoloring is allowed; losing an accepted Common cut, witness or
            // execution domain as a side effect of the new family is not.
            accepted = std::any_of(candidate.demands.begin(), candidate.demands.end(), [&](const auto& fresh) {
                return old.scope == fresh.scope && old.publication == fresh.publication &&
                       old.acquisition == fresh.acquisition && old.source == fresh.source &&
                       old.observer == fresh.observer && old.cells == fresh.cells;
            });
            if (!accepted)
                break;
        }

    // A plausible earlier cut alone is not evidence that the selected
    // baseline waits too broadly. Require an actual baseline handoff at each
    // original first site, published within this Choice after the intervening
    // parent-lane work. Otherwise this optional proposal has no proved gain.
    if (accepted && !universalPhase)
        for (const auto& demand : candidate.demands) {
            if (p.nodes[demand.acquisition].kind != Node::Choice)
                continue;
            const auto* summary = proposed.firstSummary(demand.acquisition, demand.observer);
            if (!summary || !summary->valid || summary->mayNoLane || !summary->count) {
                accepted = false;
                break;
            }
            for (unsigned i = 0; i < summary->count; ++i) {
                bool laterBaseline = false;
                for (const auto& old : baseline.demands) {
                    if (old.acquisition != summary->operations[i] || old.source != demand.source ||
                        old.observer != demand.observer)
                        continue;
                    unsigned ancestor = old.publication;
                    while (ancestor != NoCut && ancestor != demand.acquisition)
                        ancestor = proposed.parent[ancestor];
                    laterBaseline |= ancestor == demand.acquisition;
                }
                if (!laterBaseline) {
                    accepted = false;
                    break;
                }
            }
        }

    // This ordinary refinement must not perturb conditional entry/deferred
    // protocols as an indirect consequence of a different demand plan.
    if (accepted)
        for (unsigned id = 0; id < p.nodes.size(); ++id) {
            auto guarded = [](const auto& commands) {
                std::vector<Mechanism> out;
                for (const auto& mechanism : commands)
                    if (mechanism.participation != Mechanism::Every)
                        out.push_back(mechanism);
                return out;
            };
            if (guarded(baseline.before[id]) != guarded(candidate.before[id])) {
                accepted = false;
                break;
            }
        }

    std::map<unsigned, uint64_t> familiesByScope;
    if (accepted)
        for (const auto& demand : candidate.demands)
            if (demand.acquisition < p.nodes.size() && p.nodes[demand.acquisition].kind == Node::Choice)
                ++familiesByScope[demand.scope];
    if (accepted)
        for (const auto& family : proposed.universalChoices)
            ++familiesByScope[family.parent];
    if (accepted)
        for (unsigned id = 0; id < p.nodes.size(); ++id) {
            unsigned ancestor = proposed.parent[id];
            while (ancestor != NoCut && !familiesByScope.count(ancestor))
                ancestor = proposed.parent[ancestor];
            if (ancestor != NoCut)
                continue;
            // Global coloring may rename keys, but it must not change any
            // unrelated region's mechanisms, placement or direction.
            const auto& old = baseline.before[id];
            const auto& fresh = candidate.before[id];
            if (old.size() != fresh.size()) {
                accepted = false;
                break;
            }
            for (unsigned i = 0; i < old.size(); ++i) {
                auto renamed = fresh[i];
                renamed.forwardKey = old[i].forwardKey;
                renamed.reverseKey = old[i].reverseKey;
                if (!(renamed == old[i])) {
                    accepted = false;
                    break;
                }
            }
        }
    auto pathDelta = [&](unsigned root) {
        std::function<std::optional<int64_t>(unsigned)> fold = [&](unsigned id) -> std::optional<int64_t> {
            auto cost = [](const auto& group) {
                int64_t total = 0;
                for (const auto& m : group)
                    total += commandCost(m);
                return total;
            };
            int64_t own = cost(candidate.before[id]) - cost(baseline.before[id]);
            const auto& node = p.nodes[id];
            if (node.kind == Node::For || node.kind == Node::While) {
                // Independently varying nested visits cannot be flattened to
                // one iteration in a path-cost comparison. This candidate
                // leaves their complete interior command population unchanged.
                std::function<bool(unsigned)> unchanged = [&](unsigned nested) {
                    if (candidate.before[nested] != baseline.before[nested])
                        return false;
                    for (unsigned child : p.nodes[nested].children)
                        if (!unchanged(child))
                            return false;
                    return true;
                };
                for (unsigned child : node.children)
                    if (!unchanged(child))
                        return {};
                return own;
            }
            if (node.kind == Node::Choice) {
                int64_t arm = INT64_MIN;
                for (unsigned child : node.children) {
                    auto delta = fold(child);
                    if (!delta)
                        return {};
                    arm = std::max(arm, *delta);
                }
                return own + arm;
            }
            for (unsigned child : node.children) {
                auto delta = fold(child);
                if (!delta)
                    return {};
                own += *delta;
            }
            return own;
        };
        return fold(root);
    };
    if (accepted)
        for (const auto& [scope, families] : familiesByScope) {
            if (scope >= p.nodes.size() || p.nodes[scope].kind != Node::Sequence) {
                accepted = false;
                break;
            }
            auto delta = pathDelta(scope);
            // Sequence sums DELTAS and Choice takes their maximum. This allows at
            // most one additional ordinary SET/WAIT pair per owner, regardless
            // of the number of promoted requests or directed families,
            // on every structural path, without trip-count algebra.
            if (!delta || *delta > 2) {
                candidate.alternativeChoiceCostRejections += !proposed.universalChoices.empty();
                accepted = false;
                break;
            }
        }

    uint64_t candidateNodeVisits = candidate.nodeVisits + checked.nodeVisits;
    uint64_t candidateCellVisits = candidate.cellVisits + checked.cellVisits;
    if (!accepted) {
        baseline.nodeVisits += candidateNodeVisits;
        baseline.cellVisits += candidateCellVisits;
        if (universalPhase) {
            baseline.alternativeChoiceCandidates = proposed.universalChoices.size();
            baseline.rejectedAlternativeChoices = proposed.universalChoices.size();
            baseline.alternativeChoiceContinuationDemands = candidate.alternativeChoiceContinuationDemands;
            baseline.alternativeChoiceCostRejections = candidate.alternativeChoiceCostRejections;
        } else {
            baseline.choiceDemandCandidates = candidate.choiceDemandCandidates;
            baseline.rejectedChoiceDemands = candidate.choiceDemandCandidates;
        }
        recordChoiceWork(baseline);
        return baseline;
    }
    candidate.nodeVisits += baseline.nodeVisits + checked.nodeVisits;
    candidate.cellVisits += baseline.cellVisits + checked.cellVisits;
    if (universalPhase) {
        candidate.choiceDemandCandidates = baseline.choiceDemandCandidates;
        candidate.choiceDemandFamilies = baseline.choiceDemandFamilies;
        candidate.rejectedChoiceDemands = baseline.rejectedChoiceDemands;
        candidate.choiceDemandWork = baseline.choiceDemandWork;
        candidate.choiceDemandReservedWork = baseline.choiceDemandReservedWork;
        candidate.choiceDemandAnalysisWork = baseline.choiceDemandAnalysisWork;
        candidate.choiceDemandAnalysisPasses = baseline.choiceDemandAnalysisPasses;
        candidate.choiceDemandBudgetPass = baseline.choiceDemandBudgetPass;
    }
    recordChoiceWork(candidate);
    return candidate;
}

c::Result constructWithChoiceDemands(
    const c::Program& p, bool corrupt, uint64_t limit = 1u << 20, const ChildReturnOptions* childReturns = nullptr,
    bool enableAlternatives = false, bool corruptAlternative = false,
    uint64_t alternativeLimit = AlternativeChoiceCertificates::MaxWork)
{
    // A/B/C are fixed phases, not recursive candidate attempts. Common retains
    // its existing allowance. The alternative allowance applies only to C;
    // every C refusal returns the already verified B exactly.
    auto result = constructWithLateEntry(p, false, 1u << 22, nullptr, childReturns);
    DemandAnalysis::ChoiceIncomingOptions common;
    common.enabled = true;
    common.corrupt = corrupt;
    common.limit = limit;
    result = constructChoiceTransaction(p, std::move(result), common, childReturns);
    if (!enableAlternatives || !alternativeLimit || !result.success || (childReturns && childReturns->exhausted))
        return result;

    DemandAnalysis::ChoiceIncomingOptions alternatives;
    alternatives.enabled = true;
    alternatives.alternativesEnabled = true;
    alternatives.corruptAlternative = corruptAlternative;
    alternatives.limit = alternativeLimit;
    alternatives.commonAnalyzed = true;
    // Move the bounded immutable proposal population, rather than copying
    // analysis state before the alternative allowance has been established.
    if (result.choiceDemandFamilies)
        alternatives.common = std::move(common.common);
    // Failed Common proposals must not be rediscovered or rescued by C.
    alternatives.commonEnabled = !alternatives.common.empty();

    ChildReturnOptions trialChild;
    const ChildReturnOptions* trial = nullptr;
    if (childReturns && childReturns->enabled) {
        trialChild.enabled = true;
        trialChild.corrupt = childReturns->corrupt;
        trialChild.limit = std::min(childReturns->limit, ChildReturnSummaries::MaxWork) - childReturns->charged;
        trial = &trialChild;
    }
    result = constructChoiceTransaction(p, std::move(result), alternatives, trial);
    if (trial) {
        // Charge work, but do not let exhaustion in an optional C trial
        // invalidate a proof already accepted in B.
        childReturns->charged += trialChild.charged;
        childReturns->candidates += trialChild.candidates;
        childReturns->rejected += trialChild.rejected;
        childReturns->checks += trialChild.checks;
    }
    return result;
}

c::Result constructWithChildReturns(
    const c::Program& p, bool corrupt, uint64_t limit = ChildReturnSummaries::MaxWork, bool enableAlternatives = false,
    bool corruptAlternative = false, uint64_t alternativeLimit = AlternativeChoiceCertificates::MaxWork)
{
    ChildReturnOptions options;
    options.enabled = limit != 0;
    options.corrupt = corrupt;
    options.limit = limit;
    auto result = constructWithChoiceDemands(
        p, false, 1u << 20, options.enabled ? &options : nullptr, enableAlternatives, corruptAlternative,
        alternativeLimit);
    if (options.exhausted) {
        // No earlier accepted removal may escape an exhausted whole attempt.
        // Reconstruct the exact disabled plan once, never once per candidate.
        auto fallback = constructWithChoiceDemands(
            p, false, 1u << 20, nullptr, enableAlternatives, corruptAlternative, alternativeLimit);
        fallback.nodeVisits += result.nodeVisits;
        fallback.cellVisits += result.cellVisits;
        result = std::move(fallback);
    }
    result.childReturnWork = options.charged;
    result.childReturnCandidates = options.candidates;
    result.rejectedChildReturns = options.exhausted ? options.candidates : options.rejected;
    result.childReturnChecks = options.checks;
    result.childReturnBudgetCheck = options.budgetCheck;
    result.childReturnBudgetExhausted = options.exhausted;
    return result;
}

c::Result constructWithAlternativeChoices(
    const c::Program& p, bool corrupt, uint64_t limit = AlternativeChoiceCertificates::MaxWork)
{
    // This is configuration for the single upstream Choice-demand
    // transaction, not a second command-editing pass.
    return constructWithChildReturns(p, false, ChildReturnSummaries::MaxWork, true, corrupt, limit);
}
} // namespace
c::Result c::constructDemands(const Program& p) { return constructWithAlternativeChoices(p, false); }
std::optional<uint64_t> c::testing::deferredDiscoveryReservation(
    uint64_t nodes, uint64_t cells, uint64_t keys, uint64_t commands, uint64_t limit)
{
    return discoveryReservation(nodes, cells, keys, commands, limit);
}
c::Result c::testing::constructDemandsRejectingRings(const Program& p) { return constructWithDemandRings(p, true); }
c::Result c::testing::constructDemandsRejectingDeferredRings(const Program& p)
{
    return constructWithDeferredRings(p, true);
}
c::Result c::testing::constructDemandsWithoutDeferredDiscovery(const Program& p)
{
    return constructWithDeferredRings(p, false, 0);
}
c::Result c::testing::constructDemandsWithoutLateEntry(const Program& p)
{
    return constructWithDeferredRings(p, false);
}
c::Result c::testing::constructDemandsWithLateEntryWorkLimit(const Program& p, uint64_t limit)
{
    return constructWithLateEntry(p, false, limit);
}
c::Result c::testing::constructDemandsRejectingLateEntry(const Program& p) { return constructWithLateEntry(p, true); }
c::Result c::testing::constructDemandsWithoutChoiceDemands(const Program& p)
{
    return constructWithLateEntry(p, false);
}
c::Result c::testing::constructDemandsWithChoiceWorkLimit(const Program& p, uint64_t limit)
{
    return constructWithChoiceDemands(p, false, limit);
}
c::Result c::testing::constructDemandsRejectingChoiceDemands(const Program& p)
{
    return constructWithChoiceDemands(p, true);
}
c::Result c::testing::constructDemandsWithoutChildReturns(const Program& p)
{
    return constructWithChoiceDemands(p, false);
}
c::Result c::testing::constructDemandsWithChildReturnWorkLimit(const Program& p, uint64_t limit)
{
    return constructWithChildReturns(p, false, limit);
}
c::Result c::testing::constructDemandsRejectingChildReturns(const Program& p)
{
    return constructWithChildReturns(p, true);
}
c::Result c::testing::constructDemandsWithoutAlternativeChoices(const Program& p)
{
    return constructWithChildReturns(p, false);
}
c::Result c::testing::constructDemandsWithAlternativeChoiceWorkLimit(const Program& p, uint64_t limit)
{
    return constructWithAlternativeChoices(p, false, limit);
}
c::Result c::testing::constructDemandsRejectingAlternativeChoices(const Program& p)
{
    return constructWithAlternativeChoices(p, true);
}
c::Result c::testing::constructDemandsWithoutRings(const Program& p) { return constructDemandsImpl(p, false); }
c::Result c::testing::constructDemandsRejectingRefinement(const Program& p) { return constructDemandsImpl(p, true); }
c::Result c::testing::constructDemandsRejectingEntryProposal(const Program& p)
{
    return constructDemandsImpl(p, false, true, false, true);
}
c::Result c::testing::constructDemandsWithoutAllocationReplay(const Program& p)
{
    return constructDemandsImpl(p, false, false);
}
c::Result c::testing::constructDemandsRejectingAllocationReplay(const Program& p)
{
    return constructDemandsImpl(p, false, true, true);
}

c::Result c::verifyDemands(const Program& p, const std::vector<std::vector<Mechanism>>& actual)
{
    return verifyDemandImpl(p, actual, nullptr);
}
