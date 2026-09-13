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
    uint64_t executedCommands = 0;
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
        executedCommands += m.kind == c::Mechanism::Rendezvous ? 4 : 1;
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
        // The only PIPE_S payload admitted by the native composition adapter
        // has synchronous same-pipeline completion. Keep this oracle rule
        // independent of the production State::demands implementation.
        if (n.lane == unsigned(Pipe::S))
            for (auto prior : issued[n.lane])
                edge(prior, issue);
        edge(issue, done);
        issued[n.lane].push_back(done);
        accesses.push_back({issue, done, n.effects});
    }
    void macro(const c::Node& n)
    {
        for (unsigned index = 0; index < n.macroPhases.size(); ++index) {
            c::Node phase;
            phase.kind = c::Node::Operation;
            phase.lane = n.macroPhases[index].lane;
            phase.effects = n.macroPhases[index].effects;
            payload(phase);
            for (const auto& transfer : n.macroTransfers)
                if (transfer.afterPhase == index) {
                    auto v = vertex();
                    command(transfer.observer, v, false);
                    for (auto done : issued[transfer.source])
                        edge(done, v);
                }
        }
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
static unsigned p2pMacro(c::Program& p, unsigned sourceCell, unsigned stagingCell, unsigned targetCell)
{
    auto id = add(p, c::Node::Macro);
    auto& n = p.nodes[id];
    c::MacroPhase first{unsigned(Pipe::MTE2), c::Effects(p.cells)};
    first.effects[sourceCell].readers = 1u << first.lane;
    first.effects[stagingCell].writers = 1u << first.lane;
    c::MacroPhase second{unsigned(Pipe::MTE3), c::Effects(p.cells)};
    second.effects[stagingCell].readers = 1u << second.lane;
    second.effects[targetCell].writers = 1u << second.lane;
    n.macroPhases = {std::move(first), std::move(second)};
    n.macroTransfers.push_back({0, unsigned(Pipe::MTE2), unsigned(Pipe::MTE3)});
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
        case c::Node::Macro:
            oracle.macro(n);
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
static void testRemoteSignals()
{
    // Blocking remote signals are immutable cross-core protocol cuts. A
    // TNOTIFY-like publication is accepted only after the local GM prefix has
    // been completed and made visible; TWAIT-like acquisition must not
    // manufacture completion for unrelated local pipeline work.
    unsigned mte2 = unsigned(Pipe::MTE2), mte3 = unsigned(Pipe::MTE3);
    unsigned vector = unsigned(Pipe::V), scalar = unsigned(Pipe::S);
    c::FixedAction notify{c::FixedAction::RemoteNotify};
    c::FixedAction wait{c::FixedAction::RemoteWait};

    auto makeNotify = [&](unsigned lane, bool write) {
        c::Program p;
        p.cells = 1;
        p.globalMemory = {true};
        auto payload = op(p, lane, 0, write);
        auto site = add(p, c::Node::Sequence);
        add(p, c::Node::Sequence, {payload, site});
        p.fixedBefore.resize(p.nodes.size());
        p.fixedBefore[site] = {notify};
        return std::pair(std::move(p), site);
    };

    auto [unfinished, unfinishedSite] = makeNotify(mte3, true);
    auto unfinishedPlan = c::constructDemands(unfinished);
    require(
        !unfinishedPlan.success &&
        unfinishedPlan.reason == "authored remote notify has an unfinished local producer prefix");

    // Completing the MTE3 queue is insufficient for publication of its GM
    // write. The following fence is a separate visibility obligation.
    unfinished.fixedBefore[unfinishedSite].insert(
        unfinished.fixedBefore[unfinishedSite].begin(), {c::FixedAction::Barrier, mte3});
    auto unpublished = c::constructDemands(unfinished);
    require(
        !unpublished.success &&
        unpublished.reason == "authored remote notify has an unpublished non-scalar GM write");

    auto [published, publishedSite] = makeNotify(mte3, true);
    published.fixedBefore[publishedSite].insert(
        published.fixedBefore[publishedSite].begin(), {c::FixedAction::Fence});
    auto publishedPlan = c::constructDemands(published);
    require(
        publishedPlan.success && publishedPlan.fixedActions == 2 && publishedPlan.acquisitions == 0 &&
        c::verifyDemands(published, publishedPlan.before).success);
    published.core = Core::AIC;
    require(!c::constructDemands(published).success);

    // A read releases remote storage after source-pipeline completion; it does
    // not require a GM write-publication fence.
    auto [released, releasedSite] = makeNotify(mte2, false);
    released.fixedBefore[releasedSite].insert(
        released.fixedBefore[releasedSite].begin(), {c::FixedAction::Barrier, mte2});
    auto releasedPlan = c::constructDemands(released);
    require(
        releasedPlan.success && releasedPlan.fixedActions == 2 && releasedPlan.acquisitions == 0 &&
        c::verifyDemands(released, releasedPlan.before).success);

    // The same named barrier releases MTE2 for remote publication but does not
    // transfer that completion to V. The local consumer still needs its own
    // qualified cross-pipeline handoff.
    c::Program named;
    named.cells = 1;
    auto namedWrite = op(named, mte2, 0, true);
    auto namedRead = op(named, vector, 0, false);
    add(named, c::Node::Sequence, {namedWrite, namedRead});
    named.fixedBefore.resize(named.nodes.size());
    named.fixedBefore[namedRead] = {{c::FixedAction::Barrier, mte2}};
    auto namedPlan = c::constructDemands(named);
    require(
        namedPlan.success && namedPlan.acquisitions == 1 &&
        c::verifyDemands(named, namedPlan.before).success);

    auto [scalarUnpublished, scalarSite] = makeNotify(scalar, true);
    scalarUnpublished.fixedBefore[scalarSite].insert(
        scalarUnpublished.fixedBefore[scalarSite].begin(), {c::FixedAction::Fence});
    auto dirtyScalar = c::constructDemands(scalarUnpublished);
    require(
        !dirtyScalar.success && dirtyScalar.reason == "authored remote notify has an unpublished scalar GM write");
    c::FixedAction clean{c::FixedAction::CacheMaintenance};
    clean.cells = {1};
    scalarUnpublished.fixedBefore[scalarSite].insert(
        scalarUnpublished.fixedBefore[scalarSite].begin(), clean);
    auto scalarPublished = c::constructDemands(scalarUnpublished);
    require(
        scalarPublished.success && scalarPublished.fixedActions == 3 &&
        c::verifyDemands(scalarUnpublished, scalarPublished.before).success);

    c::Program local;
    local.cells = 1;
    local.globalMemory = {false};
    auto localWrite = op(local, mte2, 0, true);
    auto localWait = add(local, c::Node::Sequence);
    auto localRead = op(local, vector, 0, false);
    add(local, c::Node::Sequence, {localWrite, localWait, localRead});
    local.fixedBefore.resize(local.nodes.size());
    local.fixedBefore[localWait] = {wait};
    auto localPlan = c::constructDemands(local);
    require(
        localPlan.success && localPlan.fixedActions == 1 && localPlan.acquisitions == 1 &&
        c::verifyDemands(local, localPlan.before).success);
    std::vector<std::vector<c::Mechanism>> empty(local.nodes.size());
    require(!c::verifyDemands(local, empty).success);

    c::Program remoteRead;
    remoteRead.cells = 1;
    remoteRead.globalMemory = {true};
    auto remoteWait = add(remoteRead, c::Node::Sequence);
    auto scalarRead = op(remoteRead, scalar, 0, false);
    add(remoteRead, c::Node::Sequence, {remoteWait, scalarRead});
    remoteRead.fixedBefore.resize(remoteRead.nodes.size());
    remoteRead.fixedBefore[remoteWait] = {wait};
    auto remoteReadPlan = c::constructDemands(remoteRead);
    require(
        remoteReadPlan.success && remoteReadPlan.visibilityRequirements == 1 &&
        remoteReadPlan.before[scalarRead].size() == 1 &&
        remoteReadPlan.before[scalarRead][0].kind == c::Mechanism::Visibility &&
        remoteReadPlan.before[scalarRead][0].visibilityAction == c::VisibilityAction::InvalidateTarget &&
        c::verifyDemands(remoteRead, remoteReadPlan.before).success);
    std::vector<std::vector<c::Mechanism>> stale(remoteRead.nodes.size());
    require(!c::verifyDemands(remoteRead, stale).success);

    // A wait at the end of one loop visit can stale the scalar cache before a
    // read at the beginning of the next. The bounded loop seed must therefore
    // include fixed remote-wait effects, not only payload effects.
    c::Program loopCarried;
    loopCarried.cells = 1;
    loopCarried.globalMemory = {true};
    auto loopRead = op(loopCarried, scalar, 0, false);
    auto loopWait = add(loopCarried, c::Node::Sequence);
    auto loopBody = add(loopCarried, c::Node::Sequence, {loopRead, loopWait});
    add(loopCarried, c::Node::For, {loopBody});
    loopCarried.fixedBefore.resize(loopCarried.nodes.size());
    loopCarried.fixedBefore[loopWait] = {wait};
    auto loopCarriedPlan = c::constructDemands(loopCarried);
    require(
        loopCarriedPlan.success && loopCarriedPlan.before[loopRead].size() == 1 &&
        loopCarriedPlan.before[loopRead][0].kind == c::Mechanism::Visibility &&
        loopCarriedPlan.before[loopRead][0].visibilityAction == c::VisibilityAction::InvalidateTarget &&
        c::verifyDemands(loopCarried, loopCarriedPlan.before).success);
    std::vector<std::vector<c::Mechanism>> loopStale(loopCarried.nodes.size());
    require(!c::verifyDemands(loopCarried, loopStale).success);
}
int main()
{
    {
        // General source cuts, not a kernel recipe: residual B follows A in
        // both arms, while independent C is issued after B on the source.
        c::Program p;
        p.cells = 3;
        unsigned source = unsigned(Pipe::MTE2), observer = unsigned(Pipe::V);
        auto a = op(p, source, 0, true), b = op(p, source, 1, true), later = op(p, source, 2, true);
        std::array<unsigned, 2> first{}, second{}, arms{};
        for (unsigned arm = 0; arm < 2; ++arm) {
            first[arm] = op(p, observer, 0, false);
            second[arm] = op(p, observer, 1, false);
            arms[arm] = sequence(p, {first[arm], second[arm]});
        }
        auto choice = add(p, c::Node::Choice, {arms[0], arms[1]});
        auto root = sequence(p, {a, b, later, choice});
        auto baseline = c::testing::constructDemandsWithoutAlternativeChoices(p);
        auto selected = c::constructDemands(p);
        require(baseline.success && selected.success && selected.alternativeChoiceFamilies == 1);
        require(selected.dedicatedAllocationDomains == 0 && selected.dedicatedAllocationKeys == 0);
        require(selected.protocolKeys == 5);
        require(selected.alternativeChoiceSetsRemoved == 1 && selected.alternativeChoiceSites == 2);
        require(selected.alternativeChoicePrefixSteps == 1 && !selected.alternativeChoiceBudgetExhausted);
        require(c::verifyDemands(p, selected.before).success);
        auto set = std::find_if(selected.before[later].begin(), selected.before[later].end(), [&](const auto& m) {
            return m.kind == c::Mechanism::Publish && m.first == source && m.second == observer;
        });
        require(set != selected.before[later].end());
        auto wait = *set;
        wait.kind = c::Mechanism::Acquire;
        for (unsigned consumer : second)
            require(
                std::find(selected.before[consumer].begin(), selected.before[consumer].end(), wait) !=
                selected.before[consumer].end());
        auto rejected = c::testing::constructDemandsRejectingAlternativeChoices(p);
        require(rejected.success && rejected.before == baseline.before && rejected.rejectedAlternativeChoices == 1);
        for (uint64_t limit : {uint64_t(0), uint64_t(1), uint64_t(p.nodes.size())}) {
            auto limited = c::testing::constructDemandsWithAlternativeChoiceWorkLimit(p, limit);
            require(limited.success && limited.before == baseline.before && limited.alternativeChoiceWork <= limit);
        }
        require(c::testing::constructDemandsWithAlternativeChoiceWorkLimit(p, UINT64_MAX).before == selected.before);
        auto exact = c::testing::constructDemandsWithAlternativeChoiceWorkLimit(p, selected.alternativeChoiceWork);
        auto shortBudget =
            c::testing::constructDemandsWithAlternativeChoiceWorkLimit(p, selected.alternativeChoiceWork - 1);
        require(exact.before == selected.before && !exact.alternativeChoiceBudgetExhausted);
        require(
            shortBudget.success && shortBudget.before == baseline.before &&
            shortBudget.alternativeChoiceBudgetExhausted);
        require(shortBudget.alternativeChoiceWork < selected.alternativeChoiceWork);
        for (const auto& keys :
             {std::vector<unsigned>{0}, std::vector<unsigned>{0, 1}, std::vector<unsigned>{0, 1, 2},
              std::vector<unsigned>{0, 1, 2, 3}, std::vector<unsigned>{3, 5}}) {
            auto scarce = p;
            scarce.target.compilerKeys = keys;
            auto oldScarce = c::testing::constructDemandsWithoutAlternativeChoices(scarce);
            auto newScarce = c::constructDemands(scarce);
            require(oldScarce.success && newScarce.success);
            if (keys == std::vector<unsigned>({0, 1, 2})) {
                // This selected word mixes a Universal Choice family with an
                // ordinary exact-fit population. Universal forward and return
                // keys were numbered first and must not inflate the latter.
                require(newScarce.alternativeChoiceFamilies == 1);
                require(newScarce.dedicatedAllocationDomains == 2);
                require(newScarce.dedicatedAllocationKeys == 2 && newScarce.protocolKeys == 5);
                bool directCanonical = false;
                for (const auto& group : newScarce.before)
                    for (const auto& mechanism : group)
                        directCanonical |=
                            (mechanism.kind == c::Mechanism::Publish || mechanism.kind == c::Mechanism::Acquire) &&
                            mechanism.forwardKey == 0;
                require(directCanonical);
            } else if (keys == std::vector<unsigned>({0, 1, 2, 3})) {
                require(newScarce.alternativeChoiceFamilies == 1);
                require(newScarce.dedicatedAllocationDomains == 0 && newScarce.dedicatedAllocationKeys == 0);
                require(newScarce.protocolKeys == 5);
            } else {
                require(!newScarce.alternativeChoiceFamilies);
                require(newScarce.before == oldScarce.before);
            }
            require(c::verifyDemands(scarce, newScarce.before).success);
        }
        for (unsigned arm = 0; arm < 2; ++arm) {
            Oracle oracle;
            ExecutionPolicy policy;
            policy.trips = [](unsigned, unsigned) { return 0u; };
            policy.choice = [arm](unsigned, unsigned) { return arm; };
            execute(p, selected, root, policy, oracle);
            oracle.check();
        }
        // The old broad B receipt also served a later C consumer. Earlier
        // demand selection must leave that C history pending. The ordinary
        // constructor may supply it, or the complete optional transaction may
        // decline on cost; the old removal-only output remains invalid.
        auto continuation = p;
        auto last = p.nodes[root].children.back();
        continuation.nodes[last].kind = c::Node::Operation;
        continuation.nodes[last].lane = observer;
        continuation.nodes[last].effects[2].readers = 1u << observer;
        auto oldContinuation = c::testing::constructDemandsWithoutAlternativeChoices(continuation);
        auto newContinuation = c::constructDemands(continuation);
        require(oldContinuation.success && newContinuation.success);
        require(c::verifyDemands(continuation, newContinuation.before).success);
        if (!newContinuation.alternativeChoiceFamilies)
            require(newContinuation.before == oldContinuation.before);
        require(!c::verifyDemands(continuation, selected.before).success);
        for (unsigned arm = 0; arm < 2; ++arm) {
            Oracle oracle;
            ExecutionPolicy policy;
            policy.trips = [](unsigned, unsigned) { return 0u; };
            policy.choice = [arm](unsigned, unsigned) { return arm; };
            execute(continuation, newContinuation, root, policy, oracle);
            oracle.check();
        }
    }
    {
        // Cross-Choice continuation and storage reuse are ordinary physical
        // demands. Exercise independently varying owners and branches rather
        // than a fixed trip count or a kernel-specific prefetch shape.
        for (bool withLaterProducer : {false, true})
            for (bool recurring : {false, true})
                for (unsigned overwrite : {0u, 1u, 2u}) {
                    c::Program p;
                    p.cells = 4;
                    unsigned source = unsigned(Pipe::MTE2), observer = unsigned(Pipe::V);
                    auto a = op(p, source, 0, true), b = op(p, source, 1, true);
                    auto later = op(p, source, 2, true);
                    std::vector<unsigned> arms;
                    for (unsigned arm = 0; arm < 2; ++arm) {
                        auto first = op(p, observer, 0, false);
                        auto second = op(p, observer, 1, false);
                        arms.push_back(sequence(p, {first, second}));
                    }
                    auto choice = add(p, c::Node::Choice, arms);
                    std::vector<unsigned> body{a, b, later, choice};
                    if (withLaterProducer)
                        body.push_back(op(p, source, 3, true));
                    auto consumer = op(p, observer, 2, false);
                    if (withLaterProducer)
                        p.nodes[consumer].effects[3].readers = 1u << observer;
                    body.push_back(consumer);
                    for (unsigned i = 0; i < overwrite; ++i) {
                        body.push_back(op(p, source, 2, true));
                        body.push_back(op(p, observer, 2, false));
                    }
                    auto owner = sequence(p, body);
                    auto root = recurring ? sequence(p, {add(p, c::Node::For, {owner})}) : owner;
                    auto baseline = c::testing::constructDemandsWithoutAlternativeChoices(p);
                    auto candidate = c::constructDemands(p);
                    require(baseline.success && candidate.success);
                    require(c::verifyDemands(p, candidate.before).success);
                    if (withLaterProducer && !recurring && !overwrite) {
                        require(candidate.alternativeChoiceFamilies == 1);
                        require(!candidate.alternativeChoiceCostRejections);
                        auto missingContinuation = candidate.before;
                        missingContinuation[consumer].clear();
                        require(!c::verifyDemands(p, missingContinuation).success);
                    }
                    if (!withLaterProducer) {
                        require(candidate.alternativeChoiceContinuationDemands > 0);
                        require(candidate.alternativeChoiceCostRejections == 1);
                    }
                    if (!candidate.alternativeChoiceFamilies)
                        require(candidate.before == baseline.before);
                    for (unsigned pattern = 0; pattern < 4; ++pattern) {
                        Oracle oldOracle, newOracle;
                        ExecutionPolicy oldPolicy, newPolicy;
                        oldPolicy.trips = newPolicy.trips = [pattern](unsigned, unsigned visit) {
                            constexpr unsigned trips[] = {0, 1, 3, 0, 2};
                            return trips[(visit + pattern) % 5];
                        };
                        oldPolicy.choice =
                            newPolicy.choice = [pattern](unsigned, unsigned visit) { return (visit + pattern) % 2; };
                        uint64_t ownerVisits = 0;
                        for (unsigned invocation = 0; invocation < (recurring ? 5u : 1u); ++invocation) {
                            execute(p, baseline, root, oldPolicy, oldOracle);
                            execute(p, candidate, root, newPolicy, newOracle);
                            oldOracle.check();
                            newOracle.check();
                            ownerVisits += recurring ? oldPolicy.lastTrips.at(p.nodes[root].children.front()) : 1;
                            require(newOracle.executedCommands <= oldOracle.executedCommands + 2 * ownerVisits);
                            require(oldOracle.accesses.size() == newOracle.accesses.size());
                            for (unsigned i = 0; i < oldOracle.accesses.size(); ++i)
                                for (unsigned cell = 0; cell < p.cells; ++cell) {
                                    const auto& old = oldOracle.accesses[i].effects[cell];
                                    const auto& fresh = newOracle.accesses[i].effects[cell];
                                    require(old.readers == fresh.readers && old.writers == fresh.writers);
                                }
                        }
                    }
                }
    }
    {
        // A branch-local newer generation, an absent consumer, or a consumed
        // prefix inside independently repeated control cannot be promoted as
        // one universally participating parent family.
        for (unsigned mutation = 0; mutation < 3; ++mutation) {
            c::Program p;
            p.cells = 3;
            unsigned source = unsigned(Pipe::MTE2), observer = unsigned(Pipe::V);
            auto a = op(p, source, 0, true), b = op(p, source, 1, true);
            auto later = op(p, source, 2, true);
            std::vector<unsigned> arms;
            for (unsigned arm = 0; arm < 2; ++arm) {
                std::vector<unsigned> body{op(p, observer, 0, false)};
                if (arm == 0 && mutation == 0)
                    body.push_back(op(p, source, 1, true));
                if (!(arm == 0 && mutation == 1)) {
                    unsigned consumer = op(p, observer, 1, false);
                    if (arm == 0 && mutation == 2)
                        consumer = add(p, c::Node::For, {sequence(p, {consumer})});
                    body.push_back(consumer);
                }
                arms.push_back(sequence(p, body));
            }
            auto choice = add(p, c::Node::Choice, arms);
            auto root = sequence(p, {a, b, later, choice});
            auto baseline = c::testing::constructDemandsWithoutAlternativeChoices(p);
            auto candidate = c::constructDemands(p);
            require(baseline.success && candidate.success && !candidate.alternativeChoiceFamilies);
            require(candidate.before == baseline.before);
            for (unsigned take = 0; take < 2; ++take)
                for (unsigned trips : {0u, 1u, 3u}) {
                    Oracle oracle;
                    ExecutionPolicy policy;
                    policy.trips = [trips](unsigned, unsigned) { return trips; };
                    policy.choice = [take](unsigned, unsigned) { return take; };
                    execute(p, candidate, root, policy, oracle);
                    oracle.check();
                }
        }
    }
    {
        // One actual parent publication, alternative late acquisitions. The
        // dedicated arm returns close the same forward key on every path;
        // repeated invocations may choose different arms without resetting it.
        c::Program p;
        p.cells = 1;
        unsigned source = unsigned(Pipe::MTE2), observer = unsigned(Pipe::V);
        auto start = add(p, c::Node::Sequence);
        std::array<unsigned, 2> waits{}, returns{}, arms{};
        for (unsigned i = 0; i < 2; ++i) {
            waits[i] = add(p, c::Node::Sequence);
            returns[i] = add(p, c::Node::Sequence);
            arms[i] = sequence(p, {waits[i], returns[i]});
        }
        auto choice = add(p, c::Node::Choice, {arms[0], arms[1]});
        auto root = sequence(p, {start, choice});
        c::Result plan;
        plan.before.resize(p.nodes.size());
        plan.before[start] = {{c::Mechanism::Publish, source, observer, 1}};
        for (unsigned arm = 0; arm < 2; ++arm) {
            plan.before[waits[arm]] = {
                {c::Mechanism::Acquire, source, observer, 1},
                {c::Mechanism::Publish, observer, source, arm + 2},
                {c::Mechanism::Acquire, observer, source, arm + 2}};
        }
        require(c::verifyDemands(p, plan.before).success);
        // Disjoint keys still share pipeline queues. Keep a second complete
        // family on the observer, both before and after the alternative WAIT,
        // and compare repeated arm switching with the independent graph.
        for (bool beforeWait : {false, true}) {
            auto interleaved = plan;
            unsigned third = unsigned(Pipe::MTE3);
            for (unsigned arm = 0; arm < 2; ++arm) {
                std::vector<c::Mechanism> packet{
                    {c::Mechanism::Publish, third, observer, arm + 4},
                    {c::Mechanism::Acquire, third, observer, arm + 4},
                    {c::Mechanism::Publish, observer, third, arm + 4},
                    {c::Mechanism::Acquire, observer, third, arm + 4}};
                auto& at = interleaved.before[waits[arm]];
                at.insert(beforeWait ? at.begin() : at.end(), packet.begin(), packet.end());
            }
            require(c::verifyDemands(p, interleaved.before).success);
            Oracle oracle;
            ExecutionPolicy policy;
            policy.trips = [](unsigned, unsigned) { return 0u; };
            policy.choice = [](unsigned, unsigned visit) { return visit % 2; };
            for (unsigned invocation = 0; invocation < 7; ++invocation) {
                execute(p, interleaved, root, policy, oracle);
                oracle.check();
            }
            // The certificate is not a blanket repair for another family's
            // rearm. Removing its return must still be rejected.
            interleaved.before[waits[0]].erase(
                interleaved.before[waits[0]].begin() + (beforeWait ? 2 : 5),
                interleaved.before[waits[0]].begin() + (beforeWait ? 4 : 7));
            require(!c::verifyDemands(p, interleaved.before).success);
        }
        for (unsigned pattern = 0; pattern < 16; ++pattern) {
            ExecutionPolicy policy;
            policy.trips = [](unsigned, unsigned) { return 0u; };
            policy.choice = [pattern](unsigned, unsigned visit) { return (pattern >> (visit % 4)) & 1; };
            Oracle oracle;
            for (unsigned visit = 0; visit < 8; ++visit) {
                execute(p, plan, root, policy, oracle);
                oracle.check();
            }
        }
        // All mutations are judged from actual commands, without a selected
        // family/cut certificate supplied by the constructor.
        for (unsigned arm = 0; arm < 2; ++arm) {
            auto missing = plan.before;
            missing[waits[arm]].clear();
            require(!c::verifyDemands(p, missing).success);
            auto duplicate = plan.before;
            duplicate[waits[arm]].push_back(duplicate[waits[arm]].front());
            require(!c::verifyDemands(p, duplicate).success);
            auto wrong = plan.before;
            wrong[waits[arm]][0].forwardKey = 5;
            require(!c::verifyDemands(p, wrong).success);
            auto missingReturn = plan.before;
            missingReturn[waits[arm]].resize(1);
            require(!c::verifyDemands(p, missingReturn).success);
            auto earlyReturn = plan.before;
            std::rotate(
                earlyReturn[waits[arm]].begin(), earlyReturn[waits[arm]].begin() + 1, earlyReturn[waits[arm]].end());
            require(!c::verifyDemands(p, earlyReturn).success);
        }
        auto duplicateSet = plan.before;
        duplicateSet[start].push_back(duplicateSet[start].front());
        require(!c::verifyDemands(p, duplicateSet).success);
        auto lateSet = plan.before;
        lateSet[p.nodes[root].children.back()] = lateSet[start];
        lateSet[start].clear();
        require(!c::verifyDemands(p, lateSet).success);
        auto foreignUse = plan.before;
        foreignUse[p.nodes[root].children.back()] = {
            {c::Mechanism::Publish, observer, source, 2}, {c::Mechanism::Acquire, observer, source, 2}};
        require(!c::verifyDemands(p, foreignUse).success);
        // A body-only consumer cannot discharge the publication on a zero-trip
        // path. Keep the original parent/Choice otherwise unchanged.
        c::Program skipped;
        skipped.cells = 1;
        auto publication = add(skipped, c::Node::Sequence);
        auto loopWait = add(skipped, c::Node::Sequence), loopReturn = add(skipped, c::Node::Sequence);
        auto body = sequence(skipped, {loopWait, loopReturn});
        auto loop = add(skipped, c::Node::For, {body});
        auto armLoop = sequence(skipped, {loop});
        auto otherWait = add(skipped, c::Node::Sequence), otherReturn = add(skipped, c::Node::Sequence);
        auto otherArm = sequence(skipped, {otherWait, otherReturn});
        auto skipChoice = add(skipped, c::Node::Choice, {armLoop, otherArm});
        sequence(skipped, {publication, skipChoice});
        std::vector<std::vector<c::Mechanism>> unbalanced(skipped.nodes.size());
        unbalanced[publication] = plan.before[start];
        unbalanced[loopWait] = plan.before[waits[0]];
        unbalanced[otherWait] = plan.before[waits[1]];
        require(!c::verifyDemands(skipped, unbalanced).success);
    }
    {
        // Physical-prefix checking remains independent of alternative-key
        // participation. An A receipt must not cover later source work on B,
        // nor a newer A generation issued after the common publication.
        c::Program p;
        p.cells = 2;
        unsigned source = unsigned(Pipe::MTE2), observer = unsigned(Pipe::V);
        auto writeA = op(p, source, 0, true), writeB = op(p, source, 1, true);
        std::array<unsigned, 2> readers{}, returns{}, arms{};
        for (unsigned arm = 0; arm < 2; ++arm) {
            readers[arm] = op(p, observer, 0, false);
            returns[arm] = add(p, c::Node::Sequence);
            arms[arm] = sequence(p, {readers[arm], returns[arm]});
        }
        auto choice = add(p, c::Node::Choice, {arms[0], arms[1]});
        auto root = sequence(p, {writeA, writeB, choice});
        c::Result plan;
        plan.before.resize(p.nodes.size());
        // Explicit entry reconciliation supplies storage release on repeated
        // whole invocations; the acquisition ACK precedes the current reader.
        plan.before[writeA] = {
            {c::Mechanism::Rendezvous, std::min(source, observer), std::max(source, observer), 0, 0},
            {c::Mechanism::Barrier, source}};
        plan.before[writeB] = {{c::Mechanism::Publish, source, observer, 1}};
        for (unsigned arm = 0; arm < 2; ++arm) {
            plan.before[readers[arm]] = {
                {c::Mechanism::Acquire, source, observer, 1},
                {c::Mechanism::Publish, observer, source, arm + 2},
                {c::Mechanism::Acquire, observer, source, arm + 2}};
        }
        require(c::verifyDemands(p, plan.before).success);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned) { return 0u; };
        policy.choice = [](unsigned, unsigned visit) { return visit % 2; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 5; ++invocation) {
            execute(p, plan, root, policy, oracle);
            oracle.check();
        }
        auto newerA = p;
        newerA.nodes[writeB].effects[1].writers = 0;
        newerA.nodes[writeB].effects[0].writers = 1u << source;
        require(!c::verifyDemands(newerA, plan.before).success);
        auto missingB = p;
        auto continuation = p.nodes[root].children.back();
        missingB.nodes[continuation].kind = c::Node::Operation;
        missingB.nodes[continuation].lane = observer;
        missingB.nodes[continuation].effects[1].readers = 1u << observer;
        require(!c::verifyDemands(missingB, plan.before).success);
        for (unsigned arm = 0; arm < 2; ++arm) {
            auto late = plan.before;
            auto commands = late[readers[arm]];
            late[readers[arm]].clear();
            late[returns[arm]] = commands;
            require(!c::verifyDemands(p, late).success);
        }
    }
    {
        // A parent's WAIT-consumption may return through either closed child
        // protocol. The key domains remain disjoint; memory independence does
        // not provide this evidence. Reconstruct it from the actual commands.
        c::Program p;
        p.cells = 1;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto start = add(p, c::Node::Sequence);
        std::array<unsigned, 2> cuts{}, arms{};
        for (unsigned i = 0; i < 2; ++i) {
            cuts[i] = add(p, c::Node::Sequence);
            arms[i] = sequence(p, {cuts[i]});
        }
        auto choice = add(p, c::Node::Choice, {arms[0], arms[1]});
        auto root = sequence(p, {start, choice});
        c::Result plan;
        plan.before.resize(p.nodes.size());
        plan.before[start] = {{c::Mechanism::Publish, a, b, 1}};
        plan.before[choice] = {{c::Mechanism::Acquire, a, b, 1}};
        for (unsigned i = 0; i < 2; ++i) {
            unsigned key = i + 2;
            plan.before[cuts[i]] = {
                {c::Mechanism::Publish, b, a, key},
                {c::Mechanism::Acquire, b, a, key},
                {c::Mechanism::Publish, a, b, key},
                {c::Mechanism::Acquire, a, b, key}};
        }
        require(c::verifyDemands(p, plan.before).success);
        ExecutionPolicy policy;
        policy.trips = [](unsigned id, unsigned visit) { return (id + visit) % 4; };
        policy.choice = [](unsigned, unsigned visit) { return visit % 2; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 8; ++invocation) {
            execute(p, plan, root, policy, oracle);
            oracle.check();
        }
        for (unsigned arm = 0; arm < 2; ++arm) {
            auto empty = plan.before;
            empty[cuts[arm]].clear();
            require(!c::verifyDemands(p, empty).success);
            for (unsigned command = 0; command < 4; ++command) {
                auto missing = plan.before;
                missing[cuts[arm]].erase(missing[cuts[arm]].begin() + command);
                require(!c::verifyDemands(p, missing).success);
                auto wrong = plan.before;
                wrong[cuts[arm]][command].forwardKey = 7;
                require(!c::verifyDemands(p, wrong).success);
            }
        }
        auto sharedKey = plan.before;
        for (auto& command : sharedKey[cuts[1]])
            command.forwardKey = 2;
        require(!c::verifyDemands(p, sharedKey).success);
        // The common acquisition must execute before the child returns its
        // incoming history. A later WAIT cannot retroactively join that return.
        auto tooLate = plan.before;
        tooLate[p.nodes[root].children.back()] = tooLate[choice];
        tooLate[choice].clear();
        require(!c::verifyDemands(p, tooLate).success);
        // Child work issued after its return is not covered by that causal
        // summary. The separate physical checker must retain this hazard.
        auto payloadAfterReturn = p;
        for (unsigned cut : cuts) {
            payloadAfterReturn.nodes[cut].kind = c::Node::Operation;
            payloadAfterReturn.nodes[cut].lane = b;
            payloadAfterReturn.nodes[cut].effects[0].writers = 1u << b;
        }
        auto continuation = p.nodes[root].children.back();
        payloadAfterReturn.nodes[continuation].kind = c::Node::Operation;
        payloadAfterReturn.nodes[continuation].lane = a;
        payloadAfterReturn.nodes[continuation].effects[0].readers = 1u << a;
        require(!c::verifyDemands(payloadAfterReturn, plan.before).success);
    }
    {
        // One visit of B<->C followed by A<->B does NOT carry A-entry history
        // into C. Two visits do. Exporting a summary after the rearm check's
        // second copy would incorrectly accept this parent C->A protocol.
        c::Program p;
        p.cells = 1;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto start = add(p, c::Node::Sequence);
        auto first = add(p, c::Node::Sequence), second = add(p, c::Node::Sequence);
        auto child = sequence(p, {first, second});
        sequence(p, {start, child});
        std::vector<std::vector<c::Mechanism>> actual(p.nodes.size());
        actual[start] = {{c::Mechanism::Publish, d, a, 1}, {c::Mechanism::Acquire, d, a, 1}};
        actual[first] = {
            {c::Mechanism::Publish, b, d, 2},
            {c::Mechanism::Acquire, b, d, 2},
            {c::Mechanism::Publish, d, b, 2},
            {c::Mechanism::Acquire, d, b, 2}};
        actual[second] = {
            {c::Mechanism::Publish, a, b, 3},
            {c::Mechanism::Acquire, a, b, 3},
            {c::Mechanism::Publish, b, a, 3},
            {c::Mechanism::Acquire, b, a, 3}};
        require(!c::verifyDemands(p, actual).success);
        std::swap(actual[first], actual[second]);
        require(c::verifyDemands(p, actual).success);
        // The same non-idempotent transfer in a child's entry commands must
        // not execute again as part of the child's body summary.
        actual[first].clear();
        actual[second].clear();
        actual[child] = {{c::Mechanism::Rendezvous, b, d, 0, 0}, {c::Mechanism::Rendezvous, b, a, 0, 0}};
        require(!c::verifyDemands(p, actual).success);
        std::swap(actual[child][0], actual[child][1]);
        require(c::verifyDemands(p, actual).success);
    }
    for (auto kind : {c::Node::For, c::Node::While}) {
        // Loop bodies do not promise an unconditional return transfer. For
        // can skip every visit; While intentionally exports identity here too.
        c::Program p;
        p.cells = 1;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto start = add(p, c::Node::Sequence), inside = add(p, c::Node::Sequence);
        auto body = sequence(p, {inside});
        unsigned loop;
        if (kind == c::Node::For)
            loop = add(p, kind, {body});
        else {
            auto other = sequence(p, {});
            loop = add(p, kind, {other, body});
        }
        auto root = sequence(p, {start, loop});
        std::vector<std::vector<c::Mechanism>> actual(p.nodes.size());
        actual[start] = {{c::Mechanism::Publish, a, b, 1}, {c::Mechanism::Acquire, a, b, 1}};
        actual[inside] = {
            {c::Mechanism::Publish, b, a, 2},
            {c::Mechanism::Acquire, b, a, 2},
            {c::Mechanism::Publish, a, b, 2},
            {c::Mechanism::Acquire, a, b, 2}};
        require(!c::verifyDemands(p, actual).success);
        actual[p.nodes[root].children.back()] = {{c::Mechanism::Publish, b, a, 1}, {c::Mechanism::Acquire, b, a, 1}};
        require(c::verifyDemands(p, actual).success);
    }
    {
        // Differential qualification of universal arm transfers: six ordered
        // two-packet words per arm, all six directed parent demands. The
        // independent graph injects fresh consumption on each visit and tests
        // both arm outcomes without relying on production transfer matrices.
        std::array<unsigned, 3> lanes{unsigned(Pipe::V), unsigned(Pipe::MTE2), unsigned(Pipe::MTE3)};
        std::array<std::pair<unsigned, unsigned>, 3> pairs{{{0, 1}, {0, 2}, {1, 2}}};
        std::vector<std::pair<unsigned, unsigned>> words;
        for (unsigned first = 0; first < pairs.size(); ++first)
            for (unsigned second = 0; second < pairs.size(); ++second)
                if (first != second)
                    words.push_back({first, second});
        for (const auto& left : words)
            for (const auto& right : words)
                for (unsigned source : lanes)
                    for (unsigned target : lanes) {
                        if (source == target)
                            continue;
                        c::Program p;
                        p.cells = 1;
                        auto start = add(p, c::Node::Sequence);
                        std::array<unsigned, 2> cuts{}, arms{};
                        for (unsigned arm = 0; arm < 2; ++arm) {
                            cuts[arm] = add(p, c::Node::Sequence);
                            arms[arm] = sequence(p, {cuts[arm]});
                        }
                        auto choice = add(p, c::Node::Choice, {arms[0], arms[1]});
                        auto root = sequence(p, {start, choice});
                        c::Result plan;
                        plan.before.resize(p.nodes.size());
                        plan.before[start] = {
                            {c::Mechanism::Publish, source, target, 1}, {c::Mechanism::Acquire, source, target, 1}};
                        for (unsigned arm = 0; arm < 2; ++arm) {
                            auto word = arm ? right : left;
                            for (unsigned packet : {word.first, word.second}) {
                                auto [a, b] = pairs[packet];
                                auto& commands = plan.before[cuts[arm]];
                                unsigned key = arm + 2;
                                commands.push_back({c::Mechanism::Publish, lanes[a], lanes[b], key});
                                commands.push_back({c::Mechanism::Acquire, lanes[a], lanes[b], key});
                                commands.push_back({c::Mechanism::Publish, lanes[b], lanes[a], key});
                                commands.push_back({c::Mechanism::Acquire, lanes[b], lanes[a], key});
                            }
                        }
                        Oracle oracle;
                        ExecutionPolicy policy;
                        policy.trips = [](unsigned, unsigned) { return 0u; };
                        policy.choice = [](unsigned, unsigned visit) { return visit % 2; };
                        for (unsigned invocation = 0; invocation < 3; ++invocation)
                            execute(p, plan, root, policy, oracle);
                        bool expected = true;
                        for (auto [wait, set] : oracle.rearms)
                            expected &= oracle.reaches(wait, set);
                        require(c::verifyDemands(p, plan.before).success == expected);
                    }
    }
    {
        // Ordinary Choice: the first A consumer must acquire the prefix
        // after writeA, not the later B write. B remains an independent demand.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto writeA = op(p, a, 0, true), writeB = op(p, a, 1, true);
        auto first = op(p, b, 0, false), later = op(p, b, 1, false);
        auto otherFirst = op(p, b, 0, false), otherLater = op(p, b, 1, false);
        auto choice = add(p, c::Node::Choice, {sequence(p, {first, later}), sequence(p, {otherFirst, otherLater})});
        auto root = add(p, c::Node::For, {sequence(p, {writeA, writeB, choice})});
        auto residualB = p;
        for (unsigned id : {later, otherLater}) {
            p.nodes[id].effects[1].readers = 0;
            p.nodes[id].effects[0].readers = 1u << b;
        }
        auto baseline = c::testing::constructDemandsWithoutChoiceDemands(p);
        auto plan = c::constructDemands(p);
        require(baseline.success && plan.success && c::verifyDemands(p, plan.before).success);
        require(plan.choiceDemandFamilies > 0);
        auto early = std::find_if(plan.before[writeB].begin(), plan.before[writeB].end(), [&](const auto& m) {
            return m.kind == c::Mechanism::Publish && m.participation == c::Mechanism::Every && m.first == a &&
                   m.second == b;
        });
        require(early != plan.before[writeB].end());
        auto waiting = *early;
        waiting.kind = c::Mechanism::Acquire;
        require(
            std::find(plan.before[choice].begin(), plan.before[choice].end(), waiting) != plan.before[choice].end());
        auto rejected = c::testing::constructDemandsRejectingChoiceDemands(p);
        require(rejected.success && rejected.before == baseline.before && rejected.rejectedChoiceDemands > 0);
        for (uint64_t limit : {uint64_t(0), uint64_t(1), uint64_t(p.nodes.size())}) {
            auto exhausted = c::testing::constructDemandsWithChoiceWorkLimit(p, limit);
            require(exhausted.success && exhausted.before == baseline.before && exhausted.choiceDemandFamilies == 0);
        }
        require(c::testing::constructDemandsWithChoiceWorkLimit(p, UINT64_MAX).before == plan.before);
        require(plan.choiceDemandWork > 1 && plan.choiceDemandWork <= (1u << 20));
        auto exactBudget = c::testing::constructDemandsWithChoiceWorkLimit(p, plan.choiceDemandWork);
        auto shortBudget = c::testing::constructDemandsWithChoiceWorkLimit(p, plan.choiceDemandWork - 1);
        require(exactBudget.before == plan.before && exactBudget.choiceDemandWork <= plan.choiceDemandWork);
        require(
            shortBudget.success && shortBudget.before == baseline.before &&
            shortBudget.choiceDemandWork < plan.choiceDemandWork);
        require(
            plan.choiceDemandAnalysisPasses >= 3 && plan.choiceDemandAnalysisWork > 0 &&
            plan.choiceDemandReservedWork + plan.choiceDemandAnalysisWork < plan.choiceDemandWork);
        bool laterPassExhausted = false;
        for (unsigned fraction = 1; fraction < 32 && !laterPassExhausted; ++fraction) {
            uint64_t allowance = plan.choiceDemandReservedWork + plan.choiceDemandAnalysisWork * fraction / 32;
            auto limited = c::testing::constructDemandsWithChoiceWorkLimit(p, allowance);
            require(limited.success && limited.choiceDemandWork <= allowance);
            if (limited.choiceDemandBudgetPass) {
                require(limited.before == baseline.before);
                laterPassExhausted = limited.choiceDemandBudgetPass >= 3;
            }
        }
        require(laterPassExhausted);
        auto missing = plan.before;
        missing[choice].erase(std::find(missing[choice].begin(), missing[choice].end(), waiting));
        require(!c::verifyDemands(p, missing).success);
        auto duplicate = plan.before;
        duplicate[choice].push_back(waiting);
        require(!c::verifyDemands(p, duplicate).success);
        auto earlyPublish = plan.before;
        earlyPublish[writeB].erase(std::find(earlyPublish[writeB].begin(), earlyPublish[writeB].end(), *early));
        earlyPublish[writeA].push_back(*early);
        require(!c::verifyDemands(p, earlyPublish).success);
        auto outsideLoop = plan.before;
        outsideLoop[writeB].erase(std::find(outsideLoop[writeB].begin(), outsideLoop[writeB].end(), *early));
        outsideLoop[root].push_back(*early);
        require(!c::verifyDemands(p, outsideLoop).success);
        auto wrongKey = plan.before;
        auto wrong = std::find(wrongKey[choice].begin(), wrongKey[choice].end(), waiting);
        ++wrong->forwardKey;
        require(!c::verifyDemands(p, wrongKey).success);
        // The A-only receipt must not cover a later B consumer. Child B
        // handoffs can return the parent's consumption, avoiding its separate
        // ACK while preserving both distinct physical readiness frontiers.
        require(!c::verifyDemands(residualB, plan.before).success);
        auto residualBaseline = c::testing::constructDemandsWithoutChildReturns(residualB);
        auto residualPlan = c::constructDemands(residualB);
        require(
            residualPlan.success && residualPlan.choiceDemandFamilies > 0 && residualPlan.childReturnAcksRemoved > 0 &&
            c::verifyDemands(residualB, residualPlan.before).success);
        auto rejectedChild = c::testing::constructDemandsRejectingChildReturns(residualB);
        require(
            rejectedChild.success && rejectedChild.before == residualBaseline.before &&
            rejectedChild.rejectedChildReturns > 0);
        for (uint64_t limit : {uint64_t(0), uint64_t(1)}) {
            auto limited = c::testing::constructDemandsWithChildReturnWorkLimit(residualB, limit);
            require(limited.success && limited.before == residualBaseline.before && limited.childReturnWork <= limit);
        }
        require(
            c::testing::constructDemandsWithChildReturnWorkLimit(residualB, UINT64_MAX).before == residualPlan.before);
        require(residualPlan.childReturnChecks > 1 && !residualPlan.childReturnBudgetExhausted);
        auto exactChildBudget =
            c::testing::constructDemandsWithChildReturnWorkLimit(residualB, residualPlan.childReturnWork);
        require(exactChildBudget.before == residualPlan.before && !exactChildBudget.childReturnBudgetExhausted);
        auto shortChildBudget =
            c::testing::constructDemandsWithChildReturnWorkLimit(residualB, residualPlan.childReturnWork - 1);
        require(
            shortChildBudget.success && shortChildBudget.childReturnBudgetExhausted &&
            shortChildBudget.before == residualBaseline.before &&
            shortChildBudget.childReturnWork < residualPlan.childReturnWork);
        require(shortChildBudget.childReturnBudgetCheck > 1);
        bool refusedAfterDiscovery = false;
        for (unsigned fraction = 1; fraction < 16; ++fraction) {
            uint64_t limit = residualPlan.childReturnWork * fraction / 16;
            auto limited = c::testing::constructDemandsWithChildReturnWorkLimit(residualB, limit);
            require(limited.success && limited.childReturnWork <= limit);
            require(c::verifyDemands(residualB, limited.before).success);
            if (limited.childReturnBudgetExhausted)
                require(limited.before == residualBaseline.before);
            if (limited.rejectedChildReturns && !limited.childReturnAcksRemoved) {
                require(limited.before == residualBaseline.before);
                refusedAfterDiscovery = true;
            }
        }
        require(refusedAfterDiscovery);

        // A rejected structured-ring trial is detached from the already
        // accepted child-return transaction. Exhausting only the episode
        // allowance must return that post-baseline plan byte for byte.
        c::Program transactional;
        transactional.cells = 3;
        auto parentA = op(transactional, a, 0, true), parentB = op(transactional, a, 1, true);
        auto childA = op(transactional, b, 0, false), childB = op(transactional, b, 1, false);
        auto otherA = op(transactional, b, 0, false), otherB = op(transactional, b, 1, false);
        auto childChoice = add(
            transactional, c::Node::Choice,
            {sequence(transactional, {childA, childB}), sequence(transactional, {otherA, otherB})});
        unsigned c = unsigned(Pipe::MTE3);
        auto w0 = op(transactional, c, 2, true), r0 = op(transactional, b, 2, false);
        auto w1 = op(transactional, c, 2, true), r1 = op(transactional, b, 2, false);
        auto transactionalLoop = add(
            transactional, c::Node::For,
            {sequence(transactional, {parentA, parentB, childChoice, w0, r0, w1, r1})});
        sequence(transactional, {transactionalLoop});
        auto childBaseline = c::testing::constructDemandsWithoutStructuredRings(transactional);
        require(childBaseline.success && childBaseline.childReturnAcksRemoved > 0);
        auto rejectedEpisode = c::testing::constructDemandsWithEpisodeWorkLimit(transactional, 0);
        require(
            rejectedEpisode.success && rejectedEpisode.before == childBaseline.before &&
            rejectedEpisode.childReturnAcksRemoved == childBaseline.childReturnAcksRemoved &&
            rejectedEpisode.rejectedRecurringEpisodes > 0);
        require(c::verifyDemands(transactional, rejectedEpisode.before).success);

        ExecutionPolicy residualPolicy;
        residualPolicy.trips = [](unsigned, unsigned visit) { return (visit * 3) % 5; };
        residualPolicy.choice = [](unsigned, unsigned visit) { return visit % 2; };
        Oracle residualOracle;
        for (unsigned invocation = 0; invocation < 8; ++invocation) {
            execute(residualB, residualPlan, root, residualPolicy, residualOracle);
            residualOracle.check();
        }
        auto empty = p;
        empty.nodes[otherFirst].effects.assign(p.cells, {});
        empty.nodes[otherLater].effects.assign(p.cells, {});
        auto emptyBaseline = c::testing::constructDemandsWithoutChoiceDemands(empty);
        auto emptyPlan = c::constructDemands(empty);
        require(emptyPlan.success && emptyPlan.choiceDemandFamilies == 0 && emptyPlan.before == emptyBaseline.before);
        auto sourceInside = p;
        sourceInside.nodes[otherFirst].lane = a;
        sourceInside.nodes[otherFirst].effects.assign(p.cells, {});
        sourceInside.nodes[otherFirst].effects[0].writers = 1u << a;
        auto insideBaseline = c::testing::constructDemandsWithoutChoiceDemands(sourceInside);
        auto insidePlan = c::constructDemands(sourceInside);
        require(
            insidePlan.success && insidePlan.choiceDemandFamilies == 0 && insidePlan.before == insideBaseline.before);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return (visit * 3) % 5; };
        policy.choice = [](unsigned, unsigned visit) { return visit % 2; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 8; ++invocation) {
            execute(p, plan, root, policy, oracle);
            oracle.check();
        }
    }
    {
        // Three unknown paths share the same physical first demand. Branch
        // identity is not a generation, and varying it needs no new predicate.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto writeA = op(p, a, 0, true), writeB = op(p, a, 1, true);
        std::array<unsigned, 3> reads{}, arms{};
        for (unsigned i = 0; i < reads.size(); ++i) {
            reads[i] = op(p, b, 0, false);
            arms[i] = sequence(p, {reads[i]});
        }
        auto inner = add(p, c::Node::Choice, {arms[0], arms[1]});
        auto choice = add(p, c::Node::Choice, {sequence(p, {inner}), arms[2]});
        auto root = add(p, c::Node::For, {sequence(p, {writeA, writeB, choice})});
        auto plan = c::constructDemands(p);
        require(plan.success && plan.choiceDemandFamilies == 1 && c::verifyDemands(p, plan.before).success);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return (visit + 1) % 4; };
        policy.choice = [](unsigned id, unsigned visit) { return (id + visit) % 2; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 8; ++invocation) {
            execute(p, plan, root, policy, oracle);
            oracle.check();
        }
        auto different = p;
        different.nodes[reads[2]].effects[0].readers = 0;
        different.nodes[reads[2]].effects[1].readers = 1u << b;
        auto differentPlan = c::constructDemands(different);
        auto differentBaseline = c::testing::constructDemandsWithoutChoiceDemands(different);
        require(
            differentPlan.success && differentPlan.choiceDemandFamilies == 0 &&
            differentPlan.before == differentBaseline.before);
        require(!c::verifyDemands(different, plan.before).success);
    }
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
        // A later whole-GM visibility fence drains every AIV lane. Deferred
        // loop-exit retirement must not be retained as if that fence issued
        // only on PIPE_S; use the ordinary closed protocol instead.
        c::Program p;
        p.cells = 3;
        p.globalMemory = {false, false, true};
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto guard = add(p, c::Node::Sequence);
        auto load = op(p, a, 0, true), compute = op(p, b, 0, false), store = op(p, d, 1, false);
        p.nodes[compute].effects[1].writers = 1u << b;
        auto loop = add(p, c::Node::For, {sequence(p, {load, compute, store})});
        p.nodes[loop].entryGuardStart = guard;
        auto scalarWrite = op(p, unsigned(Pipe::S), 2, true);
        auto vectorRead = op(p, b, 2, false);
        sequence(p, {guard, loop, scalarWrite, vectorRead});
        auto plan = c::constructDemands(p);
        require(
            plan.success && plan.deferredRingCandidates == 0 && plan.deferredRings == 0 &&
            plan.visibilityRequirements == 1 && c::verifyDemands(p, plan.before).success);
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
        // Equivalent lane words in mutually exclusive arms are one recurring
        // episode family.  The exact cuts differ, but every prior/next arm pair
        // must consume each event before rearming it.
        c::Program p;
        p.cells = 1;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto wa = op(p, a, 0, true), ra = op(p, b, 0, false);
        auto wb = op(p, a, 0, true), rb = op(p, b, 0, false);
        auto left = sequence(p, {wa, ra}), right = sequence(p, {wb, rb});
        auto choice = add(p, c::Node::Choice, {left, right});
        auto loop = add(p, c::Node::For, {sequence(p, {choice})});
        sequence(p, {loop});
        auto selected = c::constructDemands(p);
        require(selected.success && selected.ringCandidates == 1);
        auto plan = c::testing::constructDemandsForcingRings(p);
        require(plan.success && plan.cutCycles == 1);
        require(plan.recurringEpisodeWords == 1 && plan.recurringEpisodePairs == 1);
        require(plan.recurringChoiceEndpoints == 4);
        require(c::verifyDemands(p, plan.before).success);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return visit % 4; };
        policy.choice = [](unsigned, unsigned visit) { return visit % 2; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 8; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
        auto missing = plan.before;
        auto& commands = missing[wb];
        auto wait = std::find_if(commands.begin(), commands.end(), [&](const auto& m) {
            return m.kind == c::Mechanism::Acquire && m.first == b && m.second == a;
        });
        require(wait != commands.end());
        commands.erase(wait);
        require(!c::verifyDemands(p, missing).success);

        auto ordinary = c::testing::constructDemandsWithoutStructuredRings(p);
        auto bounded = c::testing::constructDemandsWithEpisodeWorkLimit(p, 0);
        require(ordinary.success && bounded.success && bounded.before == ordinary.before);
        require(bounded.recurringEpisodeBudgetExhausted && bounded.rejectedRecurringEpisodes > 0);
        require(c::verifyDemands(p, bounded.before).success);
    }
    {
        // A completely skipped arm executes no partial protocol.  Empty and
        // active visits are both finite episode words, so arbitrary runs of
        // skipped iterations cannot strand or prematurely rearm a key.
        c::Program p;
        p.cells = 1;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto write = op(p, a, 0, true), read = op(p, b, 0, false);
        auto active = sequence(p, {write, read});
        auto empty = sequence(p, {});
        auto choice = add(p, c::Node::Choice, {active, empty});
        auto body = sequence(p, {choice});
        auto loop = add(p, c::Node::For, {body});
        sequence(p, {loop});
        auto plan = c::testing::constructDemandsForcingRings(p);
        require(plan.success && plan.cutCycles == 1);
        require(plan.recurringEpisodeWords == 2 && plan.recurringEpisodePairs == 4);
        require(c::verifyDemands(p, plan.before).success);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return 1 + visit % 4; };
        policy.choice = [](unsigned, unsigned visit) { return (visit % 3) != 0; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 8; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
    }
    {
        // The final lane group needs no same-visit successor cut: its next
        // publication is intentionally at the next active visit's first cut.
        // This is a structured-episode extension; the legacy-only selector
        // correctly retains the independently verified non-ring baseline.
        c::Program p;
        p.cells = 1;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto write = op(p, a, 0, true), read = op(p, b, 0, false);
        auto body = add(p, c::Node::Sequence, {write, read});
        auto loop = add(p, c::Node::For, {body});
        sequence(p, {loop});
        auto plan = c::testing::constructDemandsForcingRings(p);
        require(plan.success && plan.cutCycles == 1 && c::verifyDemands(p, plan.before).success);
        auto legacy = c::testing::constructDemandsWithoutStructuredRings(p);
        require(
            legacy.success && legacy.cutCycles == 0 && legacy.before == c::constructDemands(p).before &&
            c::verifyDemands(p, legacy.before).success);
    }
    {
        // Repeated lane directions in one finite word reuse the same key only
        // through the intervening reverse transfer.  This is certified from
        // actual event generations, not from lexical WAIT placement.
        c::Program p;
        p.cells = 1;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto w0 = op(p, a, 0, true), r0 = op(p, b, 0, false);
        auto w1 = op(p, a, 0, true), r1 = op(p, b, 0, false);
        auto loop = add(p, c::Node::For, {sequence(p, {w0, r0, w1, r1})});
        sequence(p, {loop});
        auto plan = c::testing::constructDemandsForcingRings(p);
        require(plan.success && plan.cutCycles == 1);
        require(plan.recurringEpisodeWords == 1 && plan.recurringEpisodePairs == 1);
        require(plan.recurringRepeatedDirections == 2);
        require(c::verifyDemands(p, plan.before).success);
        ExecutionPolicy policy;
        policy.trips = [](unsigned, unsigned visit) { return (visit * 3) % 5; };
        policy.choice = [](unsigned, unsigned) { return 0u; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 6; ++invocation) {
            execute(p, plan, p.nodes.size() - 1, policy, oracle);
            oracle.check();
        }
        auto missing = plan.before;
        auto& commands = missing[w1];
        auto wait = std::find_if(commands.begin(), commands.end(), [&](const auto& m) {
            return m.kind == c::Mechanism::Acquire && m.first == b && m.second == a;
        });
        require(wait != commands.end());
        commands.erase(wait);
        require(!c::verifyDemands(p, missing).success);

        // Preserve the exact endpoint population and cardinality, but execute
        // WAIT before SET at one repeated cut. Only the structured lifecycle
        // proof can reject this balanced mutation.
        auto reordered = plan.before;
        auto& pair = reordered[w1];
        auto publication = std::find_if(pair.begin(), pair.end(), [&](const auto& m) {
            return m.kind == c::Mechanism::Publish && m.first == b && m.second == a;
        });
        auto acquisition = std::find_if(pair.begin(), pair.end(), [&](const auto& m) {
            return m.kind == c::Mechanism::Acquire && m.first == b && m.second == a;
        });
        require(publication != pair.end() && acquisition != pair.end() && publication < acquisition);
        std::iter_swap(publication, acquisition);
        auto reorderedCheck = c::verifyDemands(p, reordered);
        require(
            !reorderedCheck.success &&
            reorderedCheck.reason.find("recurring episode lifecycle failed") == 0);
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
        // A guarded incoming episode and a raw recurring episode may use the
        // same lane direction, but never the same physical key. Each protocol
        // remains valid in isolation; ownership overlap must still reject.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto incoming = op(p, a, 1, true);
        auto w0 = op(p, a, 0, true), r0 = op(p, b, 0, false);
        auto incomingUse = op(p, b, 1, false);
        auto w1 = op(p, a, 0, true), r1 = op(p, b, 0, false);
        auto loop = add(p, c::Node::For, {sequence(p, {w0, r0, incomingUse, w1, r1})});
        p.nodes[loop].entryGuardStart = incoming;
        auto after = add(p, c::Node::Sequence);
        sequence(p, {incoming, loop, after});
        auto plan = c::testing::constructDemandsForcingRings(p);
        require(plan.success && plan.entryEpisodes > 0 && plan.cutCycles == 1);
        require(c::verifyDemands(p, plan.before).success);
        auto entry = std::find_if(plan.before[incomingUse].begin(), plan.before[incomingUse].end(), [&](const auto& m) {
            return m.kind == c::Mechanism::Acquire && m.participation != c::Mechanism::Every && m.first == a &&
                   m.second == b;
        });
        require(entry != plan.before[incomingUse].end());
        auto collided = plan.before;
        bool changed = false;
        for (auto& commands : collided)
            for (auto& m : commands)
                if (m.participation == c::Mechanism::Every &&
                    (m.kind == c::Mechanism::Publish || m.kind == c::Mechanism::Acquire) && m.first == a &&
                    m.second == b) {
                    m.forwardKey = entry->forwardKey;
                    changed = true;
                }
        require(changed);
        auto collision = c::verifyDemands(p, collided);
        require(
            !collision.success &&
            collision.reason == "structured recurring key collides with a guarded entry or deferred protocol");
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
        // A loop-carried source prefix can be acquired once before a common
        // direct-body Choice.  The nested branch is unknown, and the outer
        // empty arm still executes the balanced first-visit episode without
        // receiving invented payload credit.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto preload = op(p, a, 0, true), guard = add(p, c::Node::Sequence);
        auto prefix = op(p, b, 1, false);
        auto leftUse = op(p, b, 0, false), rightUse = op(p, b, 0, true);
        auto sourceMarker = op(p, d, 1, false);
        auto nested = add(p, c::Node::Choice, {sequence(p, {leftUse}), sequence(p, {rightUse, sourceMarker})});
        auto common = add(p, c::Node::Choice, {sequence(p, {nested}), sequence(p, {})});
        auto inner = add(p, c::Node::For, {sequence(p, {prefix, common})});
        auto nextGeneration = op(p, a, 0, true);
        auto outer = add(p, c::Node::For, {sequence(p, {guard, inner, nextGeneration})});
        // Model repeated whole invocations structurally too: the portable core
        // does not include the native function's terminal ALL retirement.
        auto root = add(p, c::Node::For, {sequence(p, {preload, outer})});
        p.nodes[inner].entryGuardStart = guard;

        auto plan = c::constructDemands(p);
        require(plan.success && plan.entryEpisodes == 1 && plan.entryReplyFamilies == 1);
        require(plan.entrySummarySlots > 0 && plan.entrySummaryScans > 0 && plan.entryWitnesses > 0);
        require(plan.entryStorageUnits >= plan.entrySummarySlots + plan.entryWitnessCells);
        require(plan.entryStorageUnits <= (1u << 20) && plan.entrySummaryScans <= (1u << 20));
        require(c::verifyDemands(p, plan.before).success);
        auto isEntry = [&](const c::Mechanism& m, c::Mechanism::Kind kind, c::Mechanism::Participation part) {
            return m.kind == kind && m.participation == part && m.loop == inner && m.first == a && m.second == b;
        };
        require(std::count_if(plan.before[guard].begin(), plan.before[guard].end(), [&](const auto& m) {
                    return isEntry(m, c::Mechanism::Publish, c::Mechanism::NonEmpty);
                }) == 1);
        require(std::count_if(plan.before[common].begin(), plan.before[common].end(), [&](const auto& m) {
                    return isEntry(m, c::Mechanism::Acquire, c::Mechanism::First);
                }) == 1);
        require(std::none_of(plan.before[leftUse].begin(), plan.before[leftUse].end(), [](const auto& m) {
            return m.participation == c::Mechanism::First;
        }));
        require(std::none_of(plan.before[rightUse].begin(), plan.before[rightUse].end(), [](const auto& m) {
            return m.participation == c::Mechanism::First;
        }));
        require(
            std::count_if(plan.before[nextGeneration].begin(), plan.before[nextGeneration].end(), [&](const auto& m) {
                return m.participation == c::Mechanism::NonEmpty && m.first == b && m.second == a;
            }) == 2);

        // Exercise zero/nonzero inner and outer visits, both common arms, both
        // nested arms, and repeated whole-program execution with live keys.
        const std::array<unsigned, 7> outerTrips{0, 1, 2, 1, 0, 2, 1};
        const std::array<unsigned, 8> innerTrips{0, 1, 3, 2, 0, 2, 1, 3};
        ExecutionPolicy policy;
        policy.trips = [&](unsigned id, unsigned visit) {
            return id == root  ? 2u :
                   id == outer ? outerTrips[visit % outerTrips.size()] :
                                 innerTrips[visit % innerTrips.size()];
        };
        policy.choice = [&](unsigned id, unsigned visit) {
            return id == common ? unsigned((visit + 1) % 3 == 0) : visit % 2;
        };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < outerTrips.size(); ++invocation) {
            execute(p, plan, root, policy, oracle);
            oracle.check();
        }

        // The empty arm executes no observer payload, but the common guarded
        // WAIT still consumes its provider and the reply retires the episode.
        // Record the four-command precision cost rather than hiding it.
        auto without = p;
        without.nodes[inner].entryGuardStart = ~0u;
        auto baseline = c::constructDemands(without);
        require(baseline.success && baseline.entryEpisodes == 0 && c::verifyDemands(without, baseline.before).success);
        auto oneVisit = [&](unsigned, unsigned) { return 1u; };
        auto emptyArm = [&](unsigned id, unsigned) { return id == common ? 1u : 0u; };
        ExecutionPolicy selectedPolicy, baselinePolicy;
        selectedPolicy.trips = baselinePolicy.trips = oneVisit;
        selectedPolicy.choice = baselinePolicy.choice = emptyArm;
        Oracle selectedOracle, baselineOracle;
        execute(p, plan, root, selectedPolicy, selectedOracle);
        execute(without, baseline, root, baselinePolicy, baselineOracle);
        selectedOracle.check();
        baselineOracle.check();
        require(selectedOracle.commands == baselineOracle.commands + 4);

        // No source history means no speculative episode.  Conversely, an
        // empty initial generation followed only by outer-loop recurrence is
        // valid: the first SET publishes an empty prefix, later ones publish
        // the preceding outer visit.
        auto noIncoming = p;
        noIncoming.nodes[preload].effects[0] = {};
        noIncoming.nodes[nextGeneration].effects[0] = {};
        auto noIncomingPlan = c::constructDemands(noIncoming);
        require(noIncomingPlan.success && noIncomingPlan.entryEpisodes == 0);
        auto recurrenceOnly = p;
        recurrenceOnly.nodes[preload].effects[0] = {};
        auto recurrencePlan = c::constructDemands(recurrenceOnly);
        require(
            recurrencePlan.success && recurrencePlan.entryEpisodes == 1 &&
            c::verifyDemands(recurrenceOnly, recurrencePlan.before).success);

        // Fresh reconstruction rejects changes to the common cut, owner,
        // participation, key population, provider, acknowledgment, and source
        // absence premise.  A branch-local WAIT cannot replace the common one.
        auto first = std::find_if(plan.before[common].begin(), plan.before[common].end(), [&](const auto& m) {
            return isEntry(m, c::Mechanism::Acquire, c::Mechanism::First);
        });
        require(first != plan.before[common].end());
        auto droppedFirst = plan.before;
        droppedFirst[common].erase(droppedFirst[common].begin() + (first - plan.before[common].begin()));
        require(!c::verifyDemands(p, droppedFirst).success);
        auto branchLocal = droppedFirst;
        branchLocal[leftUse].push_back(*first);
        require(!c::verifyDemands(p, branchLocal).success);
        auto wrongParticipation = plan.before;
        for (auto& m : wrongParticipation[common])
            if (m.participation == c::Mechanism::First)
                m.participation = c::Mechanism::Every;
        require(!c::verifyDemands(p, wrongParticipation).success);
        auto wrongOwner = plan.before;
        for (auto& m : wrongOwner[common])
            if (m.participation == c::Mechanism::First)
                m.loop = outer;
        require(!c::verifyDemands(p, wrongOwner).success);
        auto wrongKey = plan.before;
        for (auto& m : wrongKey[common])
            if (m.participation == c::Mechanism::First)
                ++m.forwardKey;
        require(!c::verifyDemands(p, wrongKey).success);
        auto noProvider = plan.before;
        noProvider[guard].erase(
            std::remove_if(
                noProvider[guard].begin(), noProvider[guard].end(),
                [&](const auto& m) { return isEntry(m, c::Mechanism::Publish, c::Mechanism::NonEmpty); }),
            noProvider[guard].end());
        require(!c::verifyDemands(p, noProvider).success);
        auto noReply = plan.before;
        noReply[nextGeneration].erase(
            std::remove_if(
                noReply[nextGeneration].begin(), noReply[nextGeneration].end(),
                [](const auto& m) { return m.participation == c::Mechanism::NonEmpty; }),
            noReply[nextGeneration].end());
        require(!c::verifyDemands(p, noReply).success);
        auto sourceInside = p;
        sourceInside.nodes[sourceMarker].lane = a;
        sourceInside.nodes[sourceMarker].effects.assign(p.cells, {});
        sourceInside.nodes[sourceMarker].effects[0].writers = 1u << a;
        require(!c::verifyDemands(sourceInside, plan.before).success);
        auto sourceInsidePlan = c::constructDemands(sourceInside);
        require(sourceInsidePlan.success && sourceInsidePlan.entryEpisodes == 0);
        auto earlierObserver = p;
        earlierObserver.nodes[prefix].effects[0].readers = 1u << b;
        require(!c::verifyDemands(earlierObserver, plan.before).success);
        auto earlierPlan = c::constructDemands(earlierObserver);
        require(earlierPlan.success && c::verifyDemands(earlierObserver, earlierPlan.before).success);
        require(std::none_of(earlierPlan.before[common].begin(), earlierPlan.before[common].end(), [](const auto& m) {
            return m.participation == c::Mechanism::First;
        }));
        auto global = p;
        global.globalMemory = {true, false};
        auto globalPlan = c::constructDemands(global);
        require(globalPlan.success && c::verifyDemands(global, globalPlan.before).success);
    }
    {
        // The common witness contains only each arm's first observer site.
        // A later access to B must not move the entry publication past the
        // intervening B write or receive early completion credit from A.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto writeA = op(p, a, 0, true), writeB = op(p, a, 1, true);
        auto firstA = op(p, b, 0, false), laterB = op(p, b, 1, false);
        auto otherFirstA = op(p, b, 0, false), otherLaterB = op(p, b, 1, false);
        auto common = add(p, c::Node::Choice, {sequence(p, {firstA, laterB}), sequence(p, {otherFirstA, otherLaterB})});
        auto loop = add(p, c::Node::For, {sequence(p, {common})});
        auto after = add(p, c::Node::Sequence);
        auto root = add(p, c::Node::For, {sequence(p, {writeA, writeB, loop, after})});
        p.nodes[loop].entryGuardStart = writeA;
        auto plan = c::testing::constructDemandsWithoutLateEntry(p);
        require(plan.success && plan.entryEpisodes == 1 && c::verifyDemands(p, plan.before).success);
        auto first = std::find_if(plan.before[common].begin(), plan.before[common].end(), [](const auto& m) {
            return m.participation == c::Mechanism::First;
        });
        require(first != plan.before[common].end() && first->first == a && first->second == b);
        // Commands at writeB execute after writeA and before writeB itself.
        // The resulting receipt covers A while retaining B as a later suffix.
        require(std::any_of(plan.before[writeB].begin(), plan.before[writeB].end(), [&](const auto& m) {
            return m.kind == c::Mechanism::Publish && m.participation == c::Mechanism::NonEmpty && m.loop == loop &&
                   m.first == a && m.second == b && m.forwardKey == first->forwardKey;
        }));
        require(std::none_of(plan.before[loop].begin(), plan.before[loop].end(), [&](const auto& m) {
            return m.kind == c::Mechanism::Publish && m.participation == c::Mechanism::NonEmpty && m.loop == loop &&
                   m.first == a && m.second == b && m.forwardKey == first->forwardKey;
        }));
        for (unsigned late : {laterB, otherLaterB}) {
            auto missingLate = plan.before;
            missingLate[late].clear();
            require(!c::verifyDemands(p, missingLate).success);
        }
        ExecutionPolicy policy;
        policy.trips = [=](unsigned id, unsigned visit) { return id == root ? 2u : (visit + 1) % 4; };
        policy.choice = [](unsigned, unsigned visit) { return visit % 2; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 6; ++invocation) {
            execute(p, plan, root, policy, oracle);
            oracle.check();
        }
    }
    {
        // One incoming prefix, three exclusive first consumers. Independent
        // lane work precedes the consumer in each arm. Late First placement
        // must preserve exactly one dynamic consumption, including changing
        // choices across iterations and repeated whole-program invocations.
        c::Program p;
        p.cells = 3;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto initial = op(p, a, 0, true), guard = add(p, c::Node::Sequence);
        std::array<unsigned, 3> first{}, later{}, independent{};
        std::array<unsigned, 3> arms{};
        for (unsigned i = 0; i < arms.size(); ++i) {
            independent[i] = op(p, d, 2, false);
            first[i] = op(p, b, 0, false);
            later[i] = op(p, b, 1, false);
            arms[i] = sequence(p, {independent[i], first[i], later[i]});
        }
        auto nested = add(p, c::Node::Choice, {arms[0], arms[1]});
        auto common = add(p, c::Node::Choice, {sequence(p, {nested}), arms[2]});
        auto loop = add(p, c::Node::For, {sequence(p, {common})});
        auto next = op(p, a, 0, true);
        auto root = add(p, c::Node::For, {sequence(p, {initial, guard, loop, next})});
        p.nodes[loop].entryGuardStart = guard;
        auto baseline = c::testing::constructDemandsWithoutLateEntry(p);
        auto plan = c::constructDemands(p);
        require(baseline.success && plan.success && c::verifyDemands(p, plan.before).success);
        require(plan.lateEntryFamilies == 1 && plan.lateEntrySites == 3);
        require(std::none_of(plan.before[common].begin(), plan.before[common].end(), [](const auto& m) {
            return m.participation == c::Mechanism::First;
        }));
        c::Mechanism shared;
        for (unsigned id : first) {
            auto found = std::find_if(plan.before[id].begin(), plan.before[id].end(), [](const auto& m) {
                return m.participation == c::Mechanism::First;
            });
            require(found != plan.before[id].end());
            require(
                found->kind == c::Mechanism::Acquire && found->loop == loop && found->first == a && found->second == b);
            if (id == first[0])
                shared = *found;
            else
                require(*found == shared);
        }
        auto rejected = c::testing::constructDemandsRejectingLateEntry(p);
        require(rejected.success && rejected.before == baseline.before && rejected.rejectedLateEntryFamilies > 0);
        for (uint64_t limit : {uint64_t(0), uint64_t(1), uint64_t(p.nodes.size())}) {
            auto exhausted = c::testing::constructDemandsWithLateEntryWorkLimit(p, limit);
            require(exhausted.success && exhausted.before == baseline.before && exhausted.lateEntryFamilies == 0);
        }
        auto clamped = c::testing::constructDemandsWithLateEntryWorkLimit(p, UINT64_MAX);
        require(clamped.success && clamped.before == plan.before);
        auto eraseFirst = [&](auto& commands, unsigned id) {
            commands[id].erase(
                std::remove_if(
                    commands[id].begin(), commands[id].end(),
                    [](const auto& m) { return m.participation == c::Mechanism::First; }),
                commands[id].end());
        };
        for (unsigned i = 0; i < first.size(); ++i) {
            auto missing = plan.before;
            eraseFirst(missing, first[i]);
            require(!c::verifyDemands(p, missing).success);
            auto duplicate = plan.before;
            duplicate[first[i]].push_back(shared);
            require(!c::verifyDemands(p, duplicate).success);
            auto tooLate = missing;
            tooLate[later[i]].insert(tooLate[later[i]].begin(), shared);
            require(!c::verifyDemands(p, tooLate).success);
            auto early = missing;
            early[independent[i]].push_back(shared);
            require(!c::verifyDemands(p, early).success);
            auto wrongOwner = plan.before;
            for (auto& m : wrongOwner[first[i]])
                if (m.participation == c::Mechanism::First)
                    m.loop = root;
            require(!c::verifyDemands(p, wrongOwner).success);
            auto wrongKey = plan.before;
            for (auto& m : wrongKey[first[i]])
                if (m.participation == c::Mechanism::First)
                    ++m.forwardKey;
            require(!c::verifyDemands(p, wrongKey).success);
        }
        auto mixed = plan.before;
        mixed[common].push_back(shared);
        require(!c::verifyDemands(p, mixed).success);
        // A new earlier observer or a removed arm cannot inherit the old
        // first-site certificate, even though the other paths still match.
        auto earlier = p;
        earlier.nodes[independent[1]].lane = b;
        earlier.nodes[independent[1]].effects.assign(p.cells, {});
        earlier.nodes[independent[1]].effects[0].readers = 1u << b;
        require(!c::verifyDemands(earlier, plan.before).success);
        auto empty = p;
        // Keep all nodes in the tree: replace the payload effects by unrelated
        // lane accesses instead of deleting original nodes from its population.
        for (unsigned id : {first[2], later[2]}) {
            empty.nodes[id].lane = d;
            empty.nodes[id].effects.assign(p.cells, {});
            empty.nodes[id].effects[2].readers = 1u << d;
        }
        auto emptyPlan = c::constructDemands(empty);
        auto emptyBaseline = c::testing::constructDemandsWithoutLateEntry(empty);
        require(emptyPlan.success && emptyPlan.lateEntryFamilies == 0 && emptyPlan.before == emptyBaseline.before);
        require(!c::verifyDemands(empty, plan.before).success);
        auto sourceInside = p;
        sourceInside.nodes[independent[0]].lane = a;
        sourceInside.nodes[independent[0]].effects.assign(p.cells, {});
        sourceInside.nodes[independent[0]].effects[0].writers = 1u << a;
        require(!c::verifyDemands(sourceInside, plan.before).success);
        ExecutionPolicy policy, oldPolicy;
        policy.trips =
            oldPolicy.trips = [=](unsigned id, unsigned visit) { return id == root ? 2u : (visit * 3 + 1) % 5; };
        policy.choice = oldPolicy.choice = [](unsigned id, unsigned visit) { return (id + visit) % 2; };
        Oracle oracle, oldOracle;
        for (unsigned invocation = 0; invocation < 6; ++invocation) {
            execute(p, plan, root, policy, oracle);
            execute(p, baseline, root, oldPolicy, oldOracle);
            oracle.check();
            oldOracle.check();
            require(oracle.commands == oldOracle.commands);
        }
    }
    {
        // A nested observer-free Choice may precede the first consumer.
        // Its outgoing state is uniformly Unconsumed, not an error: only
        // the complete first-consumer domain must finish uniformly Consumed.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto initial = op(p, a, 0, true), guard = add(p, c::Node::Sequence);
        auto independent = op(p, d, 1, false);
        auto before = add(p, c::Node::Choice, {sequence(p, {independent}), sequence(p, {})});
        auto left = op(p, b, 0, false), right = op(p, b, 0, false);
        auto common = add(p, c::Node::Choice, {sequence(p, {before, left}), sequence(p, {right})});
        auto loop = add(p, c::Node::For, {sequence(p, {common})});
        auto after = add(p, c::Node::Sequence);
        auto root = add(p, c::Node::For, {sequence(p, {initial, guard, loop, after})});
        p.nodes[loop].entryGuardStart = guard;
        auto plan = c::constructDemands(p);
        require(plan.success && plan.lateEntryFamilies == 1 && plan.lateEntrySites == 2);
        require(c::verifyDemands(p, plan.before).success);
        ExecutionPolicy policy;
        policy.trips = [=](unsigned id, unsigned visit) { return id == root ? 2u : visit % 3; };
        policy.choice = [](unsigned id, unsigned visit) { return (id + visit) % 2; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 5; ++invocation) {
            execute(p, plan, root, policy, oracle);
            oracle.check();
        }
    }
    {
        // Cardinality is necessary but does not justify flattening the order
        // of different entry families. The projected combined word must agree
        // across both arms, including their reverse acknowledgment protocols.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V), d = unsigned(Pipe::MTE3);
        auto initialA = op(p, a, 0, true), initialD = op(p, d, 1, true);
        auto guard = add(p, c::Node::Sequence);
        auto left = op(p, b, 0, false), right = op(p, b, 0, false);
        p.nodes[left].effects[1].readers = p.nodes[right].effects[1].readers = 1u << b;
        auto common = add(p, c::Node::Choice, {sequence(p, {left}), sequence(p, {right})});
        auto loop = add(p, c::Node::For, {sequence(p, {common})});
        auto nextA = op(p, a, 0, true), nextD = op(p, d, 1, true);
        auto root = add(p, c::Node::For, {sequence(p, {initialA, initialD, guard, loop, nextA, nextD})});
        p.nodes[loop].entryGuardStart = guard;
        auto plan = c::constructDemands(p);
        require(plan.success && plan.lateEntryFamilies == 2 && plan.lateEntrySites == 4);
        require(c::verifyDemands(p, plan.before).success);
        auto reversed = plan.before;
        std::vector<unsigned> indices;
        for (unsigned i = 0; i < reversed[right].size(); ++i)
            if (reversed[right][i].participation == c::Mechanism::First)
                indices.push_back(i);
        require(indices.size() == 2);
        std::swap(reversed[right][indices[0]], reversed[right][indices[1]]);
        auto rejected = c::verifyDemands(p, reversed);
        require(!rejected.success && rejected.reason == "entry episode command order differs across Choice paths");
        auto oversized = plan.before;
        for (unsigned i = 0; i < 9; ++i)
            oversized[left].push_back(plan.before[left][indices[0]]);
        require(!c::verifyDemands(p, oversized).success);
        ExecutionPolicy policy;
        policy.trips = [=](unsigned id, unsigned visit) { return id == root ? 2u : (visit + 1) % 4; };
        policy.choice = [](unsigned, unsigned visit) { return visit % 2; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 5; ++invocation) {
            execute(p, plan, root, policy, oracle);
            oracle.check();
        }
    }
    {
        // A possible-first-site union is not an exclusive partition. On the
        // true inner path, both possible sites execute; on the false path,
        // only the second does. mayNoLane=false must not authorize two WAITs.
        c::Program p;
        p.cells = 1;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto initial = op(p, a, 0, true), guard = add(p, c::Node::Sequence);
        auto optional = op(p, b, 0, false), afterOptional = op(p, b, 0, false);
        auto other = op(p, b, 0, false);
        auto inner = add(p, c::Node::Choice, {sequence(p, {optional}), sequence(p, {})});
        auto common = add(p, c::Node::Choice, {sequence(p, {inner, afterOptional}), sequence(p, {other})});
        auto loop = add(p, c::Node::For, {sequence(p, {common})});
        auto after = add(p, c::Node::Sequence);
        sequence(p, {initial, guard, loop, after});
        p.nodes[loop].entryGuardStart = guard;
        auto baseline = c::testing::constructDemandsWithoutLateEntry(p);
        auto plan = c::constructDemands(p);
        require(plan.success && baseline.entryEpisodes == 1 && plan.before == baseline.before);
        require(plan.lateEntryFamilies == 0 && c::verifyDemands(p, plan.before).success);
        auto forged = plan.before;
        auto found = std::find_if(forged[common].begin(), forged[common].end(), [](const auto& m) {
            return m.participation == c::Mechanism::First;
        });
        require(found != forged[common].end());
        auto wait = *found;
        forged[common].erase(found);
        for (unsigned id : {optional, afterOptional, other})
            forged[id].insert(forged[id].begin(), wait);
        require(!c::verifyDemands(p, forged).success);
    }
    {
        // Nine possible first sites exceed the fixed alternative bound.  This
        // disables only the optional common-cut summary; ordinary construction
        // and independent verification remain available.
        c::Program p;
        p.cells = 1;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto produce = op(p, a, 0, true), guard = add(p, c::Node::Sequence);
        unsigned alternatives = sequence(p, {op(p, b, 0, false)});
        for (unsigned i = 1; i < 9; ++i)
            alternatives = add(p, c::Node::Choice, {alternatives, sequence(p, {op(p, b, 0, false)})});
        auto loop = add(p, c::Node::For, {sequence(p, {alternatives})});
        auto after = add(p, c::Node::Sequence);
        auto root = add(p, c::Node::For, {sequence(p, {produce, guard, loop, after})});
        p.nodes[loop].entryGuardStart = guard;
        auto plan = c::constructDemands(p);
        require(plan.success && plan.entryEpisodes == 0 && c::verifyDemands(p, plan.before).success);
        require(plan.entrySummarySlots > 0 && plan.entryWitnessCells == 0);
        ExecutionPolicy policy;
        policy.trips = [=](unsigned id, unsigned visit) { return id == root ? 2u : visit % 3; };
        policy.choice = [](unsigned id, unsigned visit) { return (id + visit) % 2; };
        Oracle oracle;
        for (unsigned invocation = 0; invocation < 5; ++invocation) {
            execute(p, plan, root, policy, oracle);
            oracle.check();
        }
    }
    {
        // An entry key is not useful for only part of a first consumer that
        // also needs a fresh in-owner source generation. Removing that fresh
        // effect enables a common episode, including the empty-arm path to a
        // later same-cell use. This is a structural, not opcode-specific rule.
        c::Program p;
        p.cells = 2;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto initial = op(p, a, 0, true), fresh = op(p, a, 1, true);
        auto first = op(p, b, 0, false);
        p.nodes[first].effects[1].readers = 1u << b;
        auto choice = add(p, c::Node::Choice, {sequence(p, {first}), sequence(p, {})});
        auto later = op(p, b, 0, false);
        auto loop = add(p, c::Node::For, {sequence(p, {fresh, choice, later})});
        auto after = add(p, c::Node::Sequence);
        auto root = add(p, c::Node::For, {sequence(p, {initial, loop, after})});
        p.nodes[loop].entryGuardStart = initial;
        auto mixed = c::constructDemands(p);
        require(mixed.success && mixed.entryEpisodes == 0 && c::verifyDemands(p, mixed.before).success);
        require(mixed.entrySourceOverlapRejections > 0);
        auto stable = p;
        stable.nodes[fresh].effects[1] = {};
        auto plan = c::constructDemands(stable);
        require(plan.success && plan.entryEpisodes == 1 && c::verifyDemands(stable, plan.before).success);
        require(std::any_of(plan.before[choice].begin(), plan.before[choice].end(), [](const auto& m) {
            return m.participation == c::Mechanism::First;
        }));
        require(std::none_of(plan.before[later].begin(), plan.before[later].end(), [=](const auto& m) {
            return m.kind == c::Mechanism::Acquire && m.first == a && m.second == b;
        }));
        ExecutionPolicy policy, mixedPolicy;
        policy.trips = mixedPolicy.trips = [=](unsigned id, unsigned visit) { return id == root ? 2u : visit % 4; };
        policy.choice = mixedPolicy.choice = [](unsigned, unsigned visit) { return visit % 2; };
        Oracle oracle, mixedOracle;
        for (unsigned invocation = 0; invocation < 4; ++invocation) {
            execute(stable, plan, root, policy, oracle);
            execute(p, mixed, root, mixedPolicy, mixedOracle);
            oracle.check();
            mixedOracle.check();
        }
    }
    {
        // Exceed the fixed first-site storage allowance before allocation,
        // while keeping the physical program tiny. No partial first-summary
        // witness may escape; the ordinary constructor must still succeed.
        c::Program p;
        p.cells = 1;
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto produce = op(p, a, 0, true);
        std::vector<unsigned> children{produce};
        for (unsigned i = 0; i < (1u << 20) / (c::LaneCount * 8) + 1; ++i)
            children.push_back(add(p, c::Node::Sequence));
        auto use = op(p, b, 0, false);
        auto choice = add(p, c::Node::Choice, {sequence(p, {use}), sequence(p, {})});
        auto loop = add(p, c::Node::For, {sequence(p, {choice})});
        p.nodes[loop].entryGuardStart = children.back();
        children.push_back(loop);
        children.push_back(add(p, c::Node::Sequence));
        sequence(p, children);
        auto plan = c::constructDemands(p);
        require(plan.success && c::verifyDemands(p, plan.before).success);
        require(plan.entrySummarySkipped == 1 && plan.entrySummarySlots == 0 && plan.entryEpisodes == 0);
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
        // Two complete directed populations fit the two-key pool only when
        // the canonical packet key is available for direct assignment. Since
        // no fallback packet is needed, both sibling scopes remain direct.
        c::Program p;
        p.cells = 2;
        p.target.compilerKeys = {0, 1};
        unsigned a = unsigned(Pipe::MTE2), b = unsigned(Pipe::V);
        auto wa = op(p, a, 0, true), ra = op(p, b, 0, false);
        auto first = add(p, c::Node::For, {sequence(p, {wa, ra})});
        auto wb = op(p, a, 1, true), rb = op(p, b, 1, false);
        auto second = add(p, c::Node::For, {sequence(p, {wb, rb})});
        sequence(p, {first, second});
        auto plan = c::testing::constructDemandsWithoutAllocationReplay(p);
        require(plan.success && c::verifyDemands(p, plan.before).success);
        require(plan.dedicatedAllocationDomains == 2);
        require(plan.dedicatedAllocationKeys == 4);
        require(plan.allocationFallbackScopes == 0 && plan.demandFallbacks == 0);
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
        // Exact scalar-pipeline contracts: same-PIPE_S hazards need no illegal
        // barrier, while S->V completion uses a qualified event.
        c::Program p;
        p.cells = 1;
        auto first = op(p, unsigned(Pipe::S), 0, true);
        auto second = op(p, unsigned(Pipe::S), 0, true);
        auto vectorRead = op(p, unsigned(Pipe::V), 0, false);
        add(p, c::Node::Sequence, {first, second, vectorRead});
        for (bool demandDriven : {false, true}) {
            auto plan = demandDriven ? c::constructDemands(p) : c::construct(p);
            require(plan.success);
            require(plan.before[second].empty());
            require(!plan.before[vectorRead].empty());
            require((demandDriven ? c::verifyDemands(p, plan.before) : c::verify(p, plan.before)).success);
            unsigned branch = 0;
            Oracle oracle;
            execute(p, plan, p.nodes.size() - 1, 1, 0, branch, oracle);
            oracle.check();
        }
        auto broken = c::constructDemands(p);
        broken.before[vectorRead].clear();
        require(!c::verifyDemands(p, broken.before).success);
        p.core = Core::AIC;
        require(!c::constructDemands(p).success);
    }
    {
        // GM completion and GM visibility are separate target facts. Ordinary
        // non-scalar transfers use the event protocol. Scalar-cache crossings
        // use the qualified AIV CMO/fence recipe, while the disputed
        // MTE3->MTE2 publication direction remains fail-closed.
        auto check = [](unsigned source, unsigned observer, bool sourceWrites, bool targetReads, bool targetWrites,
                        bool accepted, std::optional<c::VisibilityAction> action) {
            c::Program p;
            p.cells = 1;
            p.globalMemory = {true};
            auto first = op(p, source, 0, sourceWrites);
            if (!sourceWrites) {
                p.nodes[first].effects[0].readers = uint8_t(1u << source);
                p.nodes[first].effects[0].writers = 0;
            }
            auto second = op(p, observer, 0, targetWrites);
            p.nodes[second].effects[0].readers = targetReads ? uint8_t(1u << observer) : 0;
            p.nodes[second].effects[0].writers = targetWrites ? uint8_t(1u << observer) : 0;
            auto root = sequence(p, {first, second});
            auto plan = c::constructDemands(p);
            require(plan.success == accepted);
            if (!accepted)
                return;
            require(c::verifyDemands(p, plan.before).success);
            auto found = std::find_if(plan.before[second].begin(), plan.before[second].end(), [](const auto& m) {
                return m.kind == c::Mechanism::Visibility;
            });
            require((found != plan.before[second].end()) == action.has_value());
            if (!action) {
                ExecutionPolicy policy;
                policy.trips = [](unsigned, unsigned) { return 0u; };
                policy.choice = [](unsigned, unsigned) { return 0u; };
                Oracle oracle;
                execute(p, plan, root, policy, oracle);
                oracle.check();
                return;
            }
            require(found->visibilityAction == *action);
            require(plan.visibilityRequirements == 1);
            auto missing = plan.before;
            missing[second].erase(missing[second].begin() + (found - plan.before[second].begin()));
            require(!c::verifyDemands(p, missing).success);
            auto reversed = plan.before;
            auto wrong = std::find_if(reversed[second].begin(), reversed[second].end(), [](const auto& m) {
                return m.kind == c::Mechanism::Visibility;
            });
            wrong->visibilityAction = *action == c::VisibilityAction::CleanSource ?
                                          c::VisibilityAction::InvalidateTarget :
                                          c::VisibilityAction::CleanSource;
            require(!c::verifyDemands(p, reversed).success);
            if (*action == c::VisibilityAction::InvalidateTarget) {
                wrong->visibilityAction = c::VisibilityAction::FenceOnly;
                require(!c::verifyDemands(p, reversed).success);
            }
        };
        // Non-scalar RAW and scalar same-lane accesses need no cache recipe.
        check(unsigned(Pipe::MTE2), unsigned(Pipe::V), true, true, false, true, std::nullopt);
        check(unsigned(Pipe::MTE2), unsigned(Pipe::MTE3), true, true, false, true, std::nullopt);
        check(unsigned(Pipe::S), unsigned(Pipe::S), true, true, false, true, std::nullopt);
        // Scalar-crossing RAW and WAW, including the write-only fence case.
        check(unsigned(Pipe::S), unsigned(Pipe::V), true, true, false, true, c::VisibilityAction::CleanSource);
        check(unsigned(Pipe::S), unsigned(Pipe::V), true, false, true, true, c::VisibilityAction::CleanSource);
        check(unsigned(Pipe::V), unsigned(Pipe::S), true, true, false, true, c::VisibilityAction::InvalidateTarget);
        check(unsigned(Pipe::V), unsigned(Pipe::S), true, false, true, true, c::VisibilityAction::FenceOnly);
        // Pure WAR crosses the scalar cache but transfers no value.
        check(unsigned(Pipe::S), unsigned(Pipe::V), false, false, true, true, std::nullopt);
        check(unsigned(Pipe::V), unsigned(Pipe::S), false, false, true, true, std::nullopt);
        // Only MTE3->MTE2 RAW is the disputed publication direction; WAW is
        // completion-only under the hardware reference.
        check(unsigned(Pipe::MTE3), unsigned(Pipe::MTE2), true, true, false, false, std::nullopt);
        check(unsigned(Pipe::MTE3), unsigned(Pipe::MTE2), true, false, true, true, std::nullopt);

        // A scalar write on only one branch remains a MAY-visible source at
        // the join. The common non-scalar consumer still needs the recipe.
        c::Program branch;
        branch.cells = 1;
        branch.globalMemory = {true};
        auto scalarWrite = op(branch, unsigned(Pipe::S), 0, true);
        auto yes = sequence(branch, {scalarWrite});
        auto no = sequence(branch, {});
        auto choice = add(branch, c::Node::Choice, {yes, no});
        auto vectorRead = op(branch, unsigned(Pipe::V), 0, false);
        sequence(branch, {choice, vectorRead});
        auto branchPlan = c::constructDemands(branch);
        require(
            branchPlan.success && branchPlan.visibilityRequirements == 1 &&
            c::verifyDemands(branch, branchPlan.before).success);

        // A fence-only WAW on X must not erase the independent target-cache
        // obligation for Y. The later scalar read still needs invalidation.
        c::Program independent;
        independent.cells = 2;
        independent.globalMemory = {true, true};
        auto writeY = op(independent, unsigned(Pipe::V), 1, true);
        auto writeX = op(independent, unsigned(Pipe::V), 0, true);
        auto overwriteX = op(independent, unsigned(Pipe::S), 0, true);
        auto readY = op(independent, unsigned(Pipe::S), 1, false);
        sequence(independent, {writeY, writeX, overwriteX, readY});
        auto independentPlan = c::constructDemands(independent);
        require(
            independentPlan.success && independentPlan.visibilityRequirements == 2 &&
            c::verifyDemands(independent, independentPlan.before).success);
        auto hasAction = [&](unsigned site, c::VisibilityAction expected) {
            return std::any_of(
                independentPlan.before[site].begin(), independentPlan.before[site].end(), [&](const auto& mechanism) {
                    return mechanism.kind == c::Mechanism::Visibility && mechanism.visibilityAction == expected;
                });
        };
        require(hasAction(overwriteX, c::VisibilityAction::FenceOnly));
        require(hasAction(readY, c::VisibilityAction::InvalidateTarget));
    }
    {
        // Authored synchronization is part of the immutable program transfer,
        // not a generated-plan receipt. PIPE_ALL discharges completion, while
        // CMO/fence ordering controls scalar-cache visibility independently.
        c::Program p;
        p.cells = 2;
        p.globalMemory = {false, true};
        unsigned mte2 = unsigned(Pipe::MTE2), vector = unsigned(Pipe::V), scalar = unsigned(Pipe::S);
        auto localWrite = op(p, mte2, 0, true);
        auto localRead = op(p, vector, 0, false);
        auto scalarWrite = op(p, scalar, 1, true);
        auto vectorRead = op(p, vector, 1, false);
        sequence(p, {localWrite, localRead, scalarWrite, vectorRead});
        p.fixedBefore.resize(p.nodes.size());
        p.fixedBefore[localRead].push_back({c::FixedAction::BarrierAll});
        c::FixedAction clean{c::FixedAction::CacheMaintenance};
        clean.cells = {0, 1};
        p.fixedBefore[vectorRead] = {clean, {c::FixedAction::Fence}};
        auto plan = c::constructDemands(p);
        require(
            plan.success && plan.fixedActions == 3 && plan.visibilityRequirements == 0 && plan.acquisitions == 0 &&
            c::verifyDemands(p, plan.before).success);

        // Reversing clean/fence cannot publish the preceding scalar write.
        auto reversed = p;
        std::reverse(reversed.fixedBefore[vectorRead].begin(), reversed.fixedBefore[vectorRead].end());
        auto repaired = c::constructDemands(reversed);
        require(
            repaired.success && repaired.visibilityRequirements == 1 &&
            c::verifyDemands(reversed, repaired.before).success);
        std::vector<std::vector<c::Mechanism>> empty(reversed.nodes.size());
        require(!c::verifyDemands(reversed, empty).success);

        // An addressed CMO with unproved range coverage is represented by an
        // all-zero cell mask and receives no visibility credit.
        auto unproved = p;
        unproved.fixedBefore[vectorRead][0].cells = {0, 0};
        auto conservative = c::constructDemands(unproved);
        require(
            conservative.success && conservative.visibilityRequirements == 1 &&
            c::verifyDemands(unproved, conservative.before).success);

        // Fence then invalidate is the opposite qualified direction.
        c::Program targetCache;
        targetCache.cells = 1;
        targetCache.globalMemory = {true};
        auto vectorWrite = op(targetCache, vector, 0, true);
        auto scalarRead = op(targetCache, scalar, 0, false);
        sequence(targetCache, {vectorWrite, scalarRead});
        targetCache.fixedBefore.resize(targetCache.nodes.size());
        c::FixedAction invalidate{c::FixedAction::CacheMaintenance};
        invalidate.cells = {1};
        targetCache.fixedBefore[scalarRead] = {{c::FixedAction::Fence}, invalidate};
        auto targetPlan = c::constructDemands(targetCache);
        require(
            targetPlan.success && targetPlan.fixedActions == 2 && targetPlan.visibilityRequirements == 0 &&
            targetPlan.acquisitions == 0 && c::verifyDemands(targetCache, targetPlan.before).success);

        // AIC fences drain only MTE2/MTE3/FIX. They can supply completion for
        // those resources, but cannot publish a synthetic PIPE_S generation.
        c::Program aic;
        aic.core = Core::AIC;
        aic.cells = 1;
        auto aicWrite = op(aic, mte2, 0, true);
        auto aicRead = op(aic, unsigned(Pipe::FIX), 0, false);
        sequence(aic, {aicWrite, aicRead});
        aic.fixedBefore.resize(aic.nodes.size());
        aic.fixedBefore[aicRead] = {{c::FixedAction::Fence}};
        auto aicPlan = c::constructDemands(aic);
        require(aicPlan.success && aicPlan.acquisitions == 0 && c::verifyDemands(aic, aicPlan.before).success);
        c::Program unsupportedAic;
        unsupportedAic.core = Core::AIC;
        unsupportedAic.cells = 1;
        unsupportedAic.globalMemory = {true};
        auto aicScalarWrite = op(unsupportedAic, scalar, 0, true);
        auto aicMte2Read = op(unsupportedAic, mte2, 0, false);
        sequence(unsupportedAic, {aicScalarWrite, aicMte2Read});
        unsupportedAic.fixedBefore.resize(unsupportedAic.nodes.size());
        c::FixedAction aicClean{c::FixedAction::CacheMaintenance};
        aicClean.cells = {1};
        unsupportedAic.fixedBefore[aicMte2Read] = {aicClean, {c::FixedAction::Fence}};
        require(!c::constructDemands(unsupportedAic).success);

        // A fixed visibility recipe in only one Choice arm is not an
        // unconditional post-join proof.
        c::Program choiceProgram;
        choiceProgram.cells = 1;
        choiceProgram.globalMemory = {true};
        auto beforeChoice = op(choiceProgram, scalar, 0, true);
        auto cleanedArm = sequence(choiceProgram, {});
        auto emptyArm = sequence(choiceProgram, {});
        auto choice = add(choiceProgram, c::Node::Choice, {cleanedArm, emptyArm});
        auto afterChoice = op(choiceProgram, vector, 0, false);
        sequence(choiceProgram, {beforeChoice, choice, afterChoice});
        choiceProgram.fixedBefore.resize(choiceProgram.nodes.size());
        c::FixedAction armClean{c::FixedAction::CacheMaintenance};
        armClean.cells = {1};
        choiceProgram.fixedBefore[cleanedArm] = {armClean, {c::FixedAction::Fence}};
        auto choicePlan = c::constructDemands(choiceProgram);
        require(
            choicePlan.success && choicePlan.visibilityRequirements == 1 &&
            c::verifyDemands(choiceProgram, choicePlan.before).success);
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
    // Atomic P2P macros expose ordered phase effects without exposing internal
    // event cuts. The forward hidden transfer covers staging; final MTE3 work
    // remains outstanding to later observers.
    c::Program macroProgram;
    macroProgram.cells = 3;
    auto macro = p2pMacro(macroProgram, 0, 1, 2);
    auto consume = op(macroProgram, unsigned(Pipe::MTE2), 2, false);
    add(macroProgram, c::Node::Sequence, {macro, consume});
    for (unsigned eventKey : {0u, 1u}) {
        macroProgram.target.reservations.push_back(
            {{Core::AIV, Pipe::MTE2}, {Core::AIV, Pipe::MTE3}, eventKey});
        macroProgram.target.reservations.push_back(
            {{Core::AIV, Pipe::MTE3}, {Core::AIV, Pipe::MTE2}, eventKey});
    }
    auto macroPlan = c::construct(macroProgram);
    require(macroPlan.success);
    require(c::verify(macroProgram, macroPlan.before).success);
    require(!c::constructCuts(macroProgram).success);
    require(!c::verifyCuts(macroProgram, macroPlan.before).success);
    require(!c::constructDemands(macroProgram).success);
    require(!c::verifyDemands(macroProgram, macroPlan.before).success);
    require(macroPlan.before[macro].empty());
    require(macroPlan.before[consume].size() == 1);
    require(macroPlan.before[consume][0].kind == c::Mechanism::Rendezvous);
    require(macroPlan.before[consume][0].forwardKey >= 2);
    require(macroPlan.before[consume][0].reverseKey >= 2);
    ExecutionPolicy macroPolicy;
    macroPolicy.trips = [](unsigned, unsigned) { return 1u; };
    macroPolicy.choice = [](unsigned, unsigned) { return 0u; };
    Oracle macroOracle;
    execute(macroProgram, macroPlan, macroProgram.nodes.size() - 1, macroPolicy, macroOracle);
    macroOracle.check();
    auto missingMacroCompletion = macroPlan.before;
    missingMacroCompletion[consume].clear();
    require(!c::verify(macroProgram, missingMacroCompletion).success);
    auto invalidMacro = macroProgram;
    invalidMacro.nodes[macro].macroTransfers.clear();
    require(!c::construct(invalidMacro).success);
    auto reversedMacro = macroProgram;
    reversedMacro.nodes[macro].macroTransfers = {
        {1, unsigned(Pipe::MTE3), unsigned(Pipe::MTE2)}};
    require(!c::construct(reversedMacro).success);
    auto mismatchedMacro = macroProgram;
    mismatchedMacro.nodes[macro].macroTransfers[0].source = unsigned(Pipe::V);
    require(!c::construct(mismatchedMacro).success);
    auto mismatchedObserver = macroProgram;
    mismatchedObserver.nodes[macro].macroTransfers[0].observer = unsigned(Pipe::V);
    require(!c::construct(mismatchedObserver).success);
    auto gmMacro = macroProgram;
    gmMacro.globalMemory = {false, false, true};
    require(!c::construct(gmMacro).success);

    c::Program incomingMacro;
    incomingMacro.cells = 3;
    auto producer = op(incomingMacro, unsigned(Pipe::V), 0, true);
    auto incoming = p2pMacro(incomingMacro, 0, 1, 2);
    add(incomingMacro, c::Node::Sequence, {producer, incoming});
    auto incomingPlan = c::construct(incomingMacro);
    require(incomingPlan.success && !incomingPlan.before[incoming].empty());
    require(c::verify(incomingMacro, incomingPlan.before).success);
    incomingPlan.before[incoming].clear();
    require(!c::verify(incomingMacro, incomingPlan.before).success);

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
    // A valid MTE3->MTE2 completion rendezvous must NOT discharge the
    // separately unqualified same-address GM publication requirement.
    c::Program gm;
    gm.core = Core::AIC;
    gm.cells = 1;
    auto store = op(gm, unsigned(Pipe::MTE3), 0, true);
    auto load = op(gm, unsigned(Pipe::MTE2), 0, false);
    add(gm, c::Node::Sequence, {store, load});
    auto complete = c::construct(gm);
    require(complete.success);
    gm.globalMemory = {true};
    require(!c::construct(gm).success);
    require(!c::verify(gm, complete.before).success);
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
    testRemoteSignals();
    std::cout << checks << " compositional native-helper/independent finite graph assertions passed\n";
}
