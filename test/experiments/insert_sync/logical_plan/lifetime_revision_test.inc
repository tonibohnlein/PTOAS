// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
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
    // Each readiness, release, priming and cleanup endpoint belongs to the
    // actual reusable protocol. Deleting any one must fail reconstruction.
    for (unsigned site = 0; site < plan.before.size(); ++site)
        for (unsigned index = 0; index < plan.before[site].size(); ++index) {
            auto kind = plan.before[site][index].kind;
            if (kind != c::Mechanism::Publish && kind != c::Mechanism::Acquire) continue;
            auto missing = plan.before;
            missing[site].erase(missing[site].begin() + index);
            require(!c::testing::verifyOpenDemands(p, missing).success);
        }
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

    // A final native ALL can replace the explicit exit acknowledgment, but
    // cannot consume the final credit or repair an in-body missing endpoint.
    auto retired = p;
    retired.terminalRetirementCut = tail;
    auto retiredPlan = c::constructDemands(retired);
    require(retiredPlan.success && retiredPlan.persistentLifetimes == 1);
    require(retiredPlan.selectedSetSites + 1 == plan.selectedSetSites);
    require(retiredPlan.selectedWaitSites + 1 == plan.selectedWaitSites);
    require(c::testing::verifyOpenDemands(retired, retiredPlan.before).success);
    require(!c::testing::verifyOpenDemands(p, retiredPlan.before).success);
    auto misplacedRetirement = retired;
    misplacedRetirement.terminalRetirementCut = bodyExit;
    require(!c::testing::verifyOpenDemands(misplacedRetirement, retiredPlan.before).success);
    // Retirement preserves write generations across invocations. A GM reload
    // after the previous invocation's store still needs publication, even
    // though the intervening ALL completed that store and every event.
    c::Program global;
    global.cells = 2;
    global.globalMemory = {true, true};
    auto gmRead = op(global, unsigned(Pipe::MTE2), 0, false);
    auto gmWrite = op(global, unsigned(Pipe::MTE3), 0, true);
    auto gmTail = add(global, c::Node::Sequence, {});
    add(global, c::Node::Sequence, {gmRead, gmWrite, gmTail});
    global.terminalRetirementCut = gmTail;
    std::vector<std::vector<c::Mechanism>> gmCommands(global.nodes.size());
    gmCommands[gmWrite] = {{c::Mechanism::Publish, unsigned(Pipe::MTE2), unsigned(Pipe::MTE3)},
                           {c::Mechanism::Acquire, unsigned(Pipe::MTE2), unsigned(Pipe::MTE3)}};
    require(!c::testing::verifyOpenDemands(global, gmCommands).success);
    global.nodes[gmWrite].effects[1] = global.nodes[gmWrite].effects[0];
    global.nodes[gmWrite].effects[0] = {};
    require(c::testing::verifyOpenDemands(global, gmCommands).success);
    for (unsigned site = 0; site < retiredPlan.before.size(); ++site)
        for (unsigned index = 0; index < retiredPlan.before[site].size(); ++index) {
            auto kind = retiredPlan.before[site][index].kind;
            if (kind != c::Mechanism::Publish && kind != c::Mechanism::Acquire) continue;
            auto mutation = retiredPlan.before;
            mutation[site].erase(mutation[site].begin() + index);
            require(!c::testing::verifyOpenDemands(retired, mutation).success);
        }

    // The ACC lifetime supersedes an entry-only FIX-to-M helper. The first
    // MMAD also reads an operand never accessed by FIX; that operand remains
    // an ordinary obligation, not a reason to reject the ACC witness.
    c::Program guarded;
    guarded.core = Core::AIC;
    guarded.cells = 2;
    guarded.globalMemory = {false, false};
    auto guard = add(guarded, c::Node::Sequence);
    auto operand = op(guarded, unsigned(Pipe::MTE1), 1, true);
    auto update = op(guarded, unsigned(Pipe::M), 0, true);
    guarded.nodes[update].effects[1].readers = 1u << unsigned(Pipe::M);
    guarded.nodes[update].byteEffects = guarded.nodes[update].effects;
    auto inner = add(guarded, c::Node::For, {sequence(guarded, {operand, update})});
    guarded.nodes[inner].entryGuardStart = guard;
    guarded.nodes[inner].nonEmpty = true;
    auto fix = op(guarded, unsigned(Pipe::FIX), 0, false);
    guarded.nodes[fix].byteEffects = guarded.nodes[fix].effects;
    guarded.nodes[fix].effects[0].writers = 1u << unsigned(Pipe::FIX);
    auto outer = add(guarded, c::Node::For, {sequence(guarded, {guard, inner, fix})});
    sequence(guarded, {outer});
    auto guardedBaseline = c::testing::constructDemandsWithoutPersistentLifetimes(guarded);
    auto guardedPlan = c::constructDemands(guarded);
    require(guardedBaseline.success && guardedBaseline.entryEpisodes > 0);
    if (!guardedPlan.persistentLifetimes)
        std::cerr << "ACC entry replacement: " << guardedPlan.deferredRejectionReason << '\n';
    require(guardedPlan.success && guardedPlan.persistentLifetimes == 2);
    require(c::testing::verifyOpenDemands(guarded, guardedPlan.before).success);
    ExecutionPolicy guardedPolicy;
    guardedPolicy.trips = [&](unsigned id, unsigned visit) { return id == inner ? 1 + visit % 2 : visit % 3; };
    guardedPolicy.choice = [](unsigned, unsigned) { return 0u; };
    Oracle guardedOracle;
    for (unsigned invocation = 0; invocation < 6; ++invocation) {
        execute(guarded, guardedPlan, guarded.nodes.size() - 1, guardedPolicy, guardedOracle);
        guardedOracle.check();
        auto retirement = guardedOracle.vertex();
        for (unsigned lane = 0; lane < c::LaneCount; ++lane) guardedOracle.command(lane, retirement, true);
    }
    auto unrelatedGuard = guarded;
    unrelatedGuard.nodes[fix].effects[1].writers = 1u << unsigned(Pipe::FIX);
    unrelatedGuard.nodes[fix].byteEffects[1].writers = 1u << unsigned(Pipe::FIX);
    auto unrelatedBaseline = c::testing::constructDemandsWithoutPersistentLifetimes(unrelatedGuard);
    auto unrelatedPlan = c::constructDemands(unrelatedGuard);
    require(unrelatedPlan.success && !unrelatedPlan.persistentLifetimes &&
            unrelatedPlan.lifetimeProtocolRejections > 0 && unrelatedPlan.before == unrelatedBaseline.before);

    // Multiple ACC updates and multiple FIX visits retain their original
    // exclusion; optional lifetime discovery must not collapse these uses.
    auto repeated = p;
    auto insertBefore = [&](unsigned id, c::Node n) {
        for (auto& existing : repeated.nodes)
            for (auto& child : existing.children)
                if (child >= id) ++child;
        repeated.nodes.insert(repeated.nodes.begin() + id, std::move(n));
        repeated.fixedBefore.insert(repeated.fixedBefore.begin() + id, std::vector<c::FixedAction>{});
    };
    insertBefore(reader, p.nodes[writer]);
    insertBefore(reader + 2, p.nodes[reader]);
    repeated.nodes[body + 2].children = {writer, reader, reader + 1, reader + 2, bodyExit + 2};
    auto repeatedPlan = c::constructDemands(repeated);
    require(repeatedPlan.success && repeatedPlan.persistentLifetimes == 1);
    require(c::testing::verifyOpenDemands(repeated, repeatedPlan.before).success);

    // An independent ordinary region must keep its useful early acquisition
    // when an ACC lifetime is selected elsewhere in the same function.
    auto mixed = p;
    mixed.cells = 2;
    mixed.globalMemory = {false, false};
    for (auto& n : mixed.nodes) {
        n.effects.resize(2);
        if (!n.byteEffects.empty()) n.byteEffects.resize(2);
    }
    mixed.nodes.pop_back();
    mixed.fixedBefore.pop_back();
    auto ordinaryOp = [&](Pipe pipe, bool write) {
        c::Node n;
        n.kind = c::Node::Operation;
        n.lane = unsigned(pipe);
        n.effects.resize(2);
        if (write) n.effects[1].writers = 1u << unsigned(pipe);
        else n.effects[1].readers = 1u << unsigned(pipe);
        unsigned id = mixed.nodes.size();
        mixed.nodes.push_back(std::move(n));
        return id;
    };
    auto ordinaryWrite = ordinaryOp(Pipe::MTE2, true);
    auto ordinaryRead = ordinaryOp(Pipe::MTE1, false);
    auto ordinaryOverwrite = ordinaryOp(Pipe::MTE1, true);
    auto root = p.nodes.back();
    root.effects.resize(2);
    root.children = {ordinaryWrite, ordinaryRead, ordinaryOverwrite, loop, tail};
    mixed.nodes.push_back(std::move(root));
    mixed.fixedBefore.resize(mixed.nodes.size());
    auto ordinaryBaseline = c::testing::constructDemandsWithoutPersistentLifetimes(mixed);
    auto combined = c::constructDemands(mixed);
    require(ordinaryBaseline.success && combined.success && combined.persistentLifetimes == 1);
    require(c::testing::verifyOpenDemands(mixed, combined.before).success);
    bool ordinaryAcquisition = false;
    for (const auto& m : ordinaryBaseline.before[ordinaryRead]) {
        if (m.kind != c::Mechanism::Acquire && m.kind != c::Mechanism::Rendezvous) continue;
        ordinaryAcquisition = true;
        auto fresh = std::find_if(combined.before[ordinaryRead].begin(), combined.before[ordinaryRead].end(),
            [&](const auto& command) { return command.kind == m.kind && command.first == m.first && command.second == m.second; });
        require(fresh != combined.before[ordinaryRead].end());
        if (m.kind == c::Mechanism::Acquire)
            for (unsigned site = 0; site < ordinaryBaseline.before.size(); ++site)
                for (const auto& old : ordinaryBaseline.before[site])
                    if (old.kind == c::Mechanism::Publish && old.first == m.first &&
                        old.second == m.second && old.forwardKey == m.forwardKey)
                        require(std::any_of(combined.before[site].begin(), combined.before[site].end(),
                            [&](const auto& command) {
                                return command.kind == c::Mechanism::Publish && command.first == fresh->first &&
                                    command.second == fresh->second && command.forwardKey == fresh->forwardKey;
                            }));
    }
    require(ordinaryAcquisition);

    // A shared publication acquired in either Choice arm is an independent
    // all-Every provider, even though its family counter denotes a refinement.
    auto alternatives = p;
    alternatives.cells = 4;
    alternatives.globalMemory.assign(4, false);
    for (auto& n : alternatives.nodes) {
        n.effects.resize(4);
        if (!n.byteEffects.empty()) n.byteEffects.resize(4);
    }
    alternatives.nodes.pop_back();
    alternatives.fixedBefore.pop_back();
    auto append = [&](c::Node n) {
        unsigned id = alternatives.nodes.size();
        n.effects.resize(4);
        alternatives.nodes.push_back(std::move(n));
        return id;
    };
    auto access = [&](Pipe lane, unsigned cell, bool write) {
        c::Node n;
        n.kind = c::Node::Operation;
        n.lane = unsigned(lane);
        n.effects.resize(4);
        if (write) n.effects[cell].writers = 1u << unsigned(lane);
        else n.effects[cell].readers = 1u << unsigned(lane);
        return append(std::move(n));
    };
    auto firstSource = access(Pipe::MTE2, 1, true);
    auto secondSource = access(Pipe::MTE2, 2, true);
    auto laterSource = access(Pipe::MTE2, 3, true);
    std::vector<unsigned> arms;
    for (unsigned arm = 0; arm < 2; ++arm) {
        auto first = access(Pipe::MTE1, 1, false);
        auto second = access(Pipe::MTE1, 2, false);
        c::Node sequence;
        sequence.kind = c::Node::Sequence;
        sequence.children = {first, second};
        arms.push_back(append(std::move(sequence)));
    }
    c::Node choice;
    choice.kind = c::Node::Choice;
    choice.children = arms;
    auto choiceId = append(std::move(choice));
    c::Node alternativeRoot;
    alternativeRoot.kind = c::Node::Sequence;
    alternativeRoot.children = {firstSource, secondSource, laterSource, choiceId, loop, tail};
    append(std::move(alternativeRoot));
    alternatives.fixedBefore.resize(alternatives.nodes.size());
    auto alternativeBaseline = c::testing::constructDemandsWithoutPersistentLifetimes(alternatives);
    auto alternativePlan = c::constructDemands(alternatives);
    require(alternativeBaseline.success && alternativeBaseline.alternativeChoiceFamilies == 1);
    if (!alternativePlan.persistentLifetimes) {
        std::cerr << "ACC/alternative coexistence: " << alternativePlan.deferredRejectionStage << ": "
                  << alternativePlan.deferredRejectionReason << '\n';
        for (const auto& lifetime : c::testing::summarizeStorageLifetimes(alternatives)) {
            std::cerr << "lifetime producer " << lifetime.producer << " scope " << lifetime.scope << " cells";
            for (unsigned cell : lifetime.cells) std::cerr << ' ' << cell;
            std::cerr << '\n';
        }
    }
    require(alternativePlan.success && alternativePlan.persistentLifetimes == 1);
    require(c::testing::verifyOpenDemands(alternatives, alternativePlan.before).success);
    for (unsigned site = firstSource; site <= choiceId; ++site)
        for (const auto& old : alternativeBaseline.before[site])
            require(std::any_of(alternativePlan.before[site].begin(), alternativePlan.before[site].end(),
                [&](const auto& fresh) {
                    return fresh.kind == old.kind && fresh.first == old.first && fresh.second == old.second &&
                        fresh.participation == old.participation;
                }));
    ExecutionPolicy policy;
    policy.trips = [](unsigned, unsigned visit) { return visit % 3; };
    policy.choice = [](unsigned, unsigned visit) { return visit % 2; };
    Oracle oracle;
    for (unsigned visit = 0; visit < 6; ++visit) {
        unsigned firstAccess = oracle.accesses.size();
        execute(alternatives, alternativePlan, alternatives.nodes.size() - 1, policy, oracle);
        oracle.check();
        // The later unrelated source must remain outside the prefix acquired
        // by the second consumer in either Choice arm.
        require(!oracle.reaches(oracle.accesses[firstAccess + 2].done,
                                oracle.accesses[firstAccess + 4].issue));
        // Native invocations end in an ALL retirement, which is deliberately
        // outside the portable core. Preserve event history across calls but
        // model that actual boundary before issuing the next invocation.
        auto retirement = oracle.vertex();
        for (unsigned lane = 0; lane < c::LaneCount; ++lane)
            oracle.command(lane, retirement, true);
    }

    auto scarce = mixed;
    for (auto [site, lane, write] : {
             std::tuple<unsigned, Pipe, bool>{ordinaryWrite, Pipe::M, true},
             {ordinaryRead, Pipe::FIX, false}, {ordinaryOverwrite, Pipe::FIX, true}}) {
        scarce.nodes[site].lane = unsigned(lane);
        scarce.nodes[site].effects[1] = {};
        if (write) scarce.nodes[site].effects[1].writers = 1u << unsigned(lane);
        else scarce.nodes[site].effects[1].readers = 1u << unsigned(lane);
    }
    scarce.target.compilerKeys = {0};
    auto scarceBaseline = c::testing::constructDemandsWithoutPersistentLifetimes(scarce);
    auto scarcePlan = c::constructDemands(scarce);
    require(scarcePlan.success && !scarcePlan.persistentLifetimes);
    require(scarcePlan.before == scarceBaseline.before && scarcePlan.lifetimeAllocationRejections == 1);
    require(scarcePlan.lifetimeAnalysisWork > scarcePlan.lifetimeEligibilityWork);

    auto invalid = p;
    invalid.nodes[reader].byteEffects[0].readers = 0;
    require(!c::constructDemands(invalid).success);
}
