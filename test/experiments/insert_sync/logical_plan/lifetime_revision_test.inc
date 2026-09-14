// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
// Included by composition_core_test.cpp: exercises the actual production
// lifetime constructor and open verifier, not a second implementation.
#include <set>
static void testLifetimeRevision()
{
    c::Program p;
    p.core = Core::AIC;
    p.cells = 1;
    p.globalMemory = {false};
    auto node = [&](c::Node::Kind kind, std::vector<unsigned> children = {}) {
        unsigned id = p.nodes.size();
        c::Node n;
        n.kind = kind;
        n.children = std::move(children);
        n.effects.resize(p.cells);
        p.nodes.push_back(std::move(n));
        return id;
    };
    const auto writer = node(c::Node::Operation);
    p.nodes[writer].lane = unsigned(Pipe::M);
    p.nodes[writer].effects[0].writers = 1u << unsigned(Pipe::M);
    p.nodes[writer].byteEffects = p.nodes[writer].effects;
    const auto reader = node(c::Node::Operation);
    p.nodes[reader].lane = unsigned(Pipe::FIX);
    // The true read remains subject to the original ACC resource exclusion.
    p.nodes[reader].effects[0] = {uint8_t(1u << unsigned(Pipe::FIX)), uint8_t(1u << unsigned(Pipe::FIX))};
    p.nodes[reader].byteEffects.resize(1);
    p.nodes[reader].byteEffects[0].readers = 1u << unsigned(Pipe::FIX);
    // Match the native tree's explicit yield/exit insertion cut.
    const auto bodyExit = node(c::Node::Sequence);
    const auto body = node(c::Node::Sequence, {writer, reader, bodyExit});
    const auto loop = node(c::Node::For, {body});
    const auto tail = node(c::Node::Sequence);
    node(c::Node::Sequence, {loop, tail});
    p.fixedBefore.resize(p.nodes.size());
    p.fixedBefore[tail].push_back({c::FixedAction::BarrierAll, 0, {}});

    auto summaries = c::testing::summarizeStorageLifetimes(p);
    require(summaries.size() == 1);
    require(summaries.front().producer == unsigned(Pipe::M));
    require(summaries.front().readers == (1u << unsigned(Pipe::FIX)));
    auto withoutProjection = p;
    withoutProjection.nodes[reader].byteEffects.clear();
    require(c::testing::summarizeStorageLifetimes(withoutProjection).empty());

    auto plan = c::constructDemands(p);
    require(plan.success);
    // The persistent prefix protocol and residual provider retain the
    // stronger ACC exclusion and verify against the original effects.
    require(plan.lifetimeCandidates == 1 && plan.persistentLifetimes == 1 &&
            plan.rejectedPersistentLifetimes == 0);
    require(plan.lifetimeStrongerRejections == 0 && plan.lifetimeOverlapRejections == 0);
    require(plan.lifetimeEligibilityWork == p.cells * p.nodes.size() + 1);
    require(c::verifyDemands(p, plan.before).success);
    require(!c::testing::verifyOpenDemands(p, std::vector<std::vector<c::Mechanism>>(p.nodes.size())).success);
    auto corrupted = plan.before;
    bool removed = false;
    for (auto& site : corrupted) {
        for (auto it = site.begin(); it != site.end(); ++it) {
            if (it->kind == c::Mechanism::Acquire && it->first == unsigned(Pipe::M) &&
                it->second == unsigned(Pipe::FIX)) {
                site.erase(it); removed = true; break;
            }
        }
        if (removed) break;
    }
    require(removed && !c::verifyDemands(p, corrupted).success);
    // Original obligations are unchanged by producer/reader discovery.
    require(p.nodes[reader].effects[0].writers == (1u << unsigned(Pipe::FIX)));

    auto ordinary = p;
    ordinary.nodes[reader].effects[0].writers = 0;
    auto selected = c::constructDemands(ordinary);
    require(selected.success && selected.persistentLifetimes == 1);
    uint64_t sets = 0, waits = 0, barriers = 0;
    std::set<std::tuple<unsigned,unsigned,unsigned>> keys;
    for (const auto& site : selected.before)
        for (const auto& m : site) {
            sets += m.kind == c::Mechanism::Rendezvous ? 2 : m.kind == c::Mechanism::Publish;
            waits += m.kind == c::Mechanism::Rendezvous ? 2 : m.kind == c::Mechanism::Acquire;
            barriers += m.kind == c::Mechanism::Barrier;
            if (m.kind == c::Mechanism::Rendezvous || m.kind == c::Mechanism::Publish || m.kind == c::Mechanism::Acquire)
                keys.emplace(m.first,m.second,m.forwardKey);
            if (m.kind == c::Mechanism::Rendezvous)
                keys.emplace(m.second,m.first,m.reverseKey);
        }
    require(selected.selectedCountsValid);
    require(selected.selectedSetSites == sets && selected.selectedWaitSites == waits);
    require(selected.selectedNamedBarriers == barriers && selected.protocolKeys == keys.size());
    require(c::testing::summarizeStorageLifetimes(p, 0).empty());

    auto invalid = p;
    invalid.nodes[reader].byteEffects[0].readers = 0;
    require(!c::constructDemands(invalid).success);
}
