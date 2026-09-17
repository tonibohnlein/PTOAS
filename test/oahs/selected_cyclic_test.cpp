// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include <numeric>
#include <set>
using namespace selected_test;
namespace {
void checkSlots(unsigned slots)
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V;
    auto body = base(slots, slots + 1);
    for (unsigned a = 0; a < o::PipeCount; ++a) {
        body.target.supported[a] = a == unsigned(P) || a == unsigned(Q);
        for (unsigned b = 0; b < o::PipeCount; ++b) {
            if (!((a == unsigned(P) && b == unsigned(Q)) || (a == unsigned(Q) && b == unsigned(P)))) {
                body.target.keys[a][b].clear();
            }
        }
    }
    body.reservations = {{P, Q, 0, false}, {Q, P, 0, false}};
    body.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}})};
    std::vector<unsigned> cells(slots);
    std::iota(cells.begin(), cells.end(), 0);
    auto input = o::makePeriodicLoop(body, slots, {{0, 0, cells, 1, 0}, {1, 0, cells, 1, 0}});
    require(input.success, "periodic input: " + input.reason);
    auto result = accepted(input.program);
    require(result.channels.size() == 2 * slots, "cyclic qualifier must derive both roles per slot");
    require(result.work.acknowledgments == 0 && count(result, o::Command::Barrier) == 0 &&
            count(result, o::Command::BarrierAll) == 0, "qualified slot cycle must not select extra barriers/helpers");
    for (const auto& channel : result.channels) {
        require(channel.key == channel.cell + 1, "stable slots reserve lowest unreserved directional keys");
    }
    unsigned deleted = 0;
    for (const auto& endpoint : result.ledger) {
        auto words = result.commands;
        const auto& original = result.commands[endpoint.cut];
        auto found = std::find_if(original.begin(), original.end(), [&](const auto& c) {
            return c.kind == endpoint.command.kind && c.source == endpoint.command.source &&
                   c.observer == endpoint.command.observer && c.key == endpoint.command.key;
        });
        require(found != original.end(), "ledger endpoint absent from emitted word");
        const auto position = std::size_t(found - original.begin());
        const auto observation = input.program.observed->sites[endpoint.cut].observation;
        for (std::size_t site = 0; site < words.size(); ++site) {
            if (input.program.observed->sites[site].observation == observation) {
                words[site].erase(words[site].begin() + position);
            }
        }
        require(!o::checkCausalFrontier(input.program, words).accepted, "deleted cyclic endpoint was accepted");
        ++deleted;
    }
    // One original observation cannot receive different words in two contexts.
    auto words = result.commands;
    bool mutated = false;
    for (std::size_t site = 0; site < words.size(); ++site) {
        if (!words[site].empty() && o::canonicalCommandCut(input.program, site) != site) {
            words[site].clear();
            mutated = true;
            break;
        }
    }
    if (mutated) {
        require(!o::checkCausalFrontier(input.program, words).accepted, "nonuniform observation accepted");
    }
    auto unavailable = input.program;
    for (auto& observation : unavailable.observed->observations) {
        observation.available = false;
    }
    require(!o::constructSelectedPlan(unavailable).success, "unavailable original guards accepted");
    auto scarce = input.program;
    scarce.target.keys[unsigned(P)][unsigned(Q)].pop_back();
    auto refusal = o::constructSelectedPlan(scarce);
    require(!refusal.success && refusal.failure == o::SelectedFailure::EventResource,
            "insufficient qualified role keys must be a resource refusal, not fallback");
    std::cout << "slots=" << slots << " channels=" << result.channels.size()
              << " mutated=" << deleted << " fixedpoint=" << result.work.invariantSiteEvaluations << '\n';
}
o::Program refine(o::Program p, std::size_t owner)
{
    const auto& q = *p.observed;
    o::CountedLoopRegion loop;
    loop.owner = owner;
    loop.header = q.sites[owner].successors.front();
    loop.bodyEntry = q.sites[loop.header].successors[0];
    loop.continuation = q.sites[loop.header].successors[1];
    std::vector<std::size_t> todo{loop.bodyEntry};
    std::set<std::size_t> seen;
    while (!todo.empty()) {
        auto site = todo.back(); todo.pop_back();
        if (site == loop.header || !seen.insert(site).second) continue;
        loop.bodySites.push_back(site);
        const auto& next = q.sites[site].successors;
        todo.insert(todo.end(), next.begin(), next.end());
    }
    auto refined = o::refineCountedLoop(p, loop);
    require(refined.success, refined.reason);
    return refined.program;
}
void regional()
{
    auto p = base(2, 8);
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V;
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                    op(P, {{1, false, true}}), op(Q, {{1, true, false}}),
                    op(P, {{0, false, true}, {1, false, true}})};
    o::Region first{o::Region::For, {seq({leaf(0), leaf(1)})}, 0, true};
    o::Region second{o::Region::For, {seq({leaf(2), leaf(3)})}, 0, true};
    p.body = seq({first, second, leaf(4)});
    auto input = o::addStructuredBoundaryCuts(p);
    require(input.success, input.reason);
    std::vector<std::size_t> owners;
    for (const auto& scope : input.program.observed->scopes)
        if (scope.kind == o::AnalysisContext::ForBody) owners.push_back(scope.ownerSite);
    require(owners.size() == 2, "two loop scopes");
    auto qualified = refine(refine(input.program, owners[0]), owners[1]);
    auto result = accepted(qualified);
    require(result.channels.size() == 4, "independent regions retain four roles");
    require(result.loops.size() == 2, "entry/backedge/exit certificates for both regions");
    for (const auto& loop : result.loops) {
        require(loop.incoming.reachable() && loop.outgoing.reachable(), "loop interface lost context");
        for (const auto& clause : loop.clauses) {
            require(clause.version == loop.version && clause.version == result.ledger.size(), "stale role clause");
            require(clause.beforeIssue == result.certificate.cuts[clause.cut].beforeIssue, "invented conditional credit");
        }
    }
    // The same inner protocol is entered repeatedly by an unrefined outer loop.
    p.operations.resize(2);
    p.body = {o::Region::For, {first}, 0, true};
    input = o::addStructuredBoundaryCuts(p);
    require(input.success, input.reason);
    owners.clear();
    for (const auto& scope : input.program.observed->scopes)
        if (scope.kind == o::AnalysisContext::ForBody) owners.push_back(scope.ownerSite);
    auto nested = accepted(refine(input.program, owners.back()));
    require(nested.channels.size() == 2, "inner role protocol retained through repeated entries");
    unsigned removedReturns = 0;
    for (const auto& channel : nested.channels) {
        if (channel.source != Q || channel.observer != P) continue;
        for (auto cut : channel.publications) {
            if (std::find(channel.acquisitions.begin(), channel.acquisitions.end(), cut) == channel.acquisitions.end()) continue;
            auto words = nested.commands;
            const auto qualifiedProgram = refine(input.program, owners.back());
            const auto observation = qualifiedProgram.observed->sites[cut].observation;
            for (std::size_t site = 0; site < words.size(); ++site) {
                if (qualifiedProgram.observed->sites[site].observation != observation) continue;
                auto& word = words[site];
                word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
                    return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
                        command.source == Q && command.observer == P && command.key == channel.key;
                }), word.end());
            }
            require(!o::checkCausalFrontier(qualifiedProgram, words).accepted,
                    "re-entered first use accepted without the final consumption return");
            ++removedReturns;
        }
    }
    require(removedReturns != 0, "re-entry acknowledgment mutation was not exercised");
}
void contextual()
{
    auto body = base(2, 8);
    // Multiple readers, unrelated effects and may-write intervals must remain
    // ordinary access obligations; no definite-content assertion is necessary.
    body.operations = {op(o::Pipe::MTE2, {{0, false, true}}),
                       op(o::Pipe::V, {{0, true, false}, {1, false, true}}),
                       op(o::Pipe::V, {{0, true, false}})};
    auto input = o::makePeriodicLoop(body, 1, {});
    require(input.success, input.reason);
    auto& p = input.program;
    auto& q = *p.observed;
    const auto preOp = p.operations.size();
    p.operations.push_back(op(o::Pipe::MTE3, {{0, false, true}}));
    const auto postOp = p.operations.size();
    p.operations.push_back(op(o::Pipe::MTE1, {{0, true, false}, {1, false, true}}));
    auto node = [&](std::size_t operation) {
        const auto site = q.sites.size();
        const auto observation = q.observations.size();
        q.observations.push_back({1000 + site, {}, true});
        q.sites.push_back({operation, observation, {}, {}, 0});
        return site;
    };
    const auto oldEntry = q.entry, oldExit = q.exit;
    q.entry = node(preOp);
    const auto post = node(postOp);
    q.exit = node(o::NoControlId);
    q.sites[q.entry].successors = {oldEntry};
    q.sites[oldExit].successors = {post};
    q.sites[post].successors = {q.exit};
    p.target.barrierAll = true;
    p.invocation.retirement = o::Program::InvocationContract::DrainAllAtReturn;
    auto result = accepted(p);
    require(result.channels.size() == 2, "open loop lost ready/release roles");
    require(result.work.frontierVisits > 0 && !result.decisions.empty(), "surrounding requirements bypassed ordinary construction");
    require(count(result, o::Command::BarrierAll) == 1, "retirement belongs only at invocation exit");
    require(result.commands[q.exit].back().kind == o::Command::BarrierAll, "wrong retirement cut");
    // Deleting an entry-context handoff or an exit-context handoff must remain
    // observable even though the local recurring role protocol is intact.
    unsigned rejected = 0;
    for (const auto& endpoint : result.ledger) {
        if (endpoint.command.kind != o::Command::Acquire) continue;
        auto words = result.commands;
        const auto& original = result.commands[endpoint.cut];
        auto found = std::find_if(original.begin(), original.end(), [&](const auto& c) {
            return c.kind == endpoint.command.kind && c.source == endpoint.command.source &&
                c.observer == endpoint.command.observer && c.key == endpoint.command.key;
        });
        const auto offset = std::size_t(found - original.begin());
        for (std::size_t site = 0; site < words.size(); ++site)
            if (q.sites[site].observation == q.sites[endpoint.cut].observation)
                words[site].erase(words[site].begin() + offset);
        require(!o::checkCausalFrontier(p, words).accepted, "contextual acquisition deletion accepted");
        ++rejected;
    }
    require(rejected > 2, "contextual mutation population absent");
    // A fixed event can be live across the child. It belongs to the same global
    // key ledger; the role allocator must neither borrow nor reset it.
    o::Commands fixed(o::commandCutCount(p));
    fixed[q.entry].push_back({o::Command::Publish, o::Pipe::MTE2, o::Pipe::V, 7});
    fixed[q.exit].push_back({o::Command::Acquire, o::Pipe::MTE2, o::Pipe::V, 7});
    auto withIncoming = accepted(p, fixed);
    require(withIncoming.channels.size() == 2, "fixed events disable role qualification");
    for (const auto& channel : withIncoming.channels)
        require(channel.key != 7, "recurring channel reused a fixed key");
    const auto& keys = withIncoming.certificate.keys;
    auto found = std::find_if(keys.begin(), keys.end(), [](const auto& key) {
        return key.source == o::Pipe::MTE2 && key.observer == o::Pipe::V && key.key == 7;
    });
    require(found != keys.end(), "fixed key disappeared");
    require(withIncoming.loops[0].incoming.facts()->events[found - keys.begin()].occupancy == 2,
            "child interface reset an incoming live event");
}
void sharedRecurringPrefixes()
{
    const auto P = o::Pipe::MTE1, Q = o::Pipe::M;
    auto body = base(2, 4);
    body.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                       op(Q, {{0, true, false}, {1, true, false}})};
    const auto input = o::makePeriodicLoop(body, 1, {});
    require(input.success, input.reason);
    const auto result = accepted(input.program);
    require(result.channels.size() == 2,
            "common matrix consumer and reuse frontier must share recurring ready/release prefixes");
    for (const auto& channel : result.channels) {
        require(channel.cells == std::vector<unsigned>({0, 1}),
                "shared recurring channel must retain both physical obligations");
    }
}
} // namespace
int main()
{
    regional();
    contextual();
    sharedRecurringPrefixes();
    for (unsigned slots = 1; slots <= 4; ++slots) {
        checkSlots(slots);
    }
}
