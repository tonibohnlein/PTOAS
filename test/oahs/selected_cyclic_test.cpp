// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
#include <functional>
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
    require(result.work.frontierSameVisit != 0,
            "recurring readiness requirements lost same-visit correspondence");
    require(result.work.frontierPreviousUse != 0,
            "recurring release requirements lost previous-use correspondence");
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
void distinctReleaseDeadlines()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::MTE1;
    auto body = base(2, 4);
    body.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                       op(Q, {{0, true, false}}), op(Q, {{1, true, false}})};
    const auto input = o::makePeriodicLoop(body, 1, {});
    require(input.success, input.reason);
    const auto& p = input.program;
    const auto plan = accepted(p);
    require(plan.channels.size() == 4, "different reader prefixes must keep separate releases");
    const auto& g = *p.observed;
    for (unsigned iterations : {0u, 1u, 2u, 6u}) {
        std::vector<o::Cut> path;
        std::set<std::pair<o::Cut, unsigned>> active, dead;
        std::function<bool(o::Cut, unsigned)> walk = [&](o::Cut at, unsigned offset) {
            const auto key = std::make_pair(at, offset);
            if (active.count(key) || dead.count(key)) return false;
            const auto& node = g.sites[at];
            if (node.operation != o::NoControlId) {
                if (offset == 4 * iterations || p.operations[node.operation].original != offset % 4) return false;
                ++offset;
            }
            path.push_back(at);
            if (at == g.exit && offset == 4 * iterations) return true;
            active.insert(key);
            for (auto next : node.successors) if (walk(next, offset)) return true;
            active.erase(key); dead.insert(key); path.pop_back(); return false;
        };
        require(walk(g.entry, 0), "missing separate-release repeated trace");
        auto flat = p; flat.observed.reset(); flat.body = {}; flat.operations.clear();
        o::Commands words; std::vector<o::Command> pending;
        for (auto cut : path) {
            pending.insert(pending.end(), plan.commands[cut].begin(), plan.commands[cut].end());
            const auto operation = g.sites[cut].operation;
            if (operation == o::NoControlId) continue;
            flat.operations.push_back(p.operations[operation]);
            words.push_back(std::move(pending)); pending.clear();
        }
        words.push_back(std::move(pending));
        std::vector<unsigned> visits(flat.operations.size());
        std::iota(visits.begin(), visits.end(), 0);
        std::vector<std::pair<unsigned, unsigned>> forbidden;
        for (unsigned i = 1; i < iterations; ++i) forbidden.emplace_back(4*i-1, 4*i);
        require(bool(oahs_oracle::graph(flat, words, visits, forbidden)),
                "later B reader now gates the next independent A overwrite");
    }
}

void transitiveRecurringCoverage()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
    auto body = base(1, 8);
    body.operations = {op(P, {{0, false, true}}),
                       op(Q, {{0, true, true}}), op(R, {{0, true, false}})};
    body.body = {o::Region::For, {seq({leaf(0), leaf(1), leaf(2)})}, 0, true};
    auto input = o::addStructuredBoundaryCuts(body);
    require(input.success, input.reason);
    auto &q = *input.program.observed;
    const auto originalSize = q.sites.size();
    for (std::size_t site = 0; site < originalSize; ++site) {
        if (q.sites[site].operation == o::NoControlId) continue;
        const auto cut = q.sites.size(), observation = q.observations.size();
        auto boundary = q.sites[site];
        boundary.operation = o::NoControlId;
        boundary.observation = observation;
        q.observations.push_back({1000 + cut, {}, true});
        q.sites.push_back(std::move(boundary));
        q.sites[site].successors = {cut};
        q.sites[site].backedgeOwners.clear();
    }
    const auto program = refine(input.program, input.program.observed->scopes[1].ownerSite);
    const auto result = accepted(program);
    std::cout << "transitive channels=" << result.channels.size()
              << " removed=" << result.work.redundantRecurringChannels << '\n';
    require(result.channels.size() == 3 && result.work.redundantRecurringChannels == 0 && result.work.recurringTrials == 0,
            "load/RMW/store recurrence must retain only its ready/ready/release chain");
    require(result.work.acknowledgments == 0, "real storage release already supplies rearming");
    // A remaining storage-release channel is also part of the rearming proof.
    // Deleting any complete channel must invalidate the combined cycle.
    for (const auto &channel : result.channels) {
        auto words = result.commands;
        for (auto &word : words)
            word.erase(std::remove_if(word.begin(), word.end(), [&](const auto &command) {
                return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
                    command.source == channel.source && command.observer == channel.observer && command.key == channel.key;
            }), word.end());
        require(!o::checkCausalFrontier(program, words).accepted,
                "removing a necessary whole channel was accepted");
    }
}

// A retained normalization value and an independently loaded operand share
// actual completion through an in-place pipeline and its final-reader return.
void pipelineOperandSupport()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
    auto body = base(4, 8);
    body.operations = {
        op(Q, {{0, true, false}, {2, false, true}}),
        op(P, {{0, false, true}, {3, false, true}}), op(P, {{1, false, true}}),
        op(Q, {{0, true, true}, {3, true, true}, {2, true, false}}),
        op(Q, {{0, true, true}, {3, true, true}, {1, true, false}}),
        op(Q, {{0, true, true}, {3, true, false}}), op(R, {{0, true, false}})};
    o::Region inner{o::Region::For, {seq({leaf(1), leaf(2), leaf(3), leaf(4), leaf(5), leaf(6)})}};
    body.body = {o::Region::For, {seq({leaf(0), inner})}};
    auto build = [&](const o::Program& original) {
        auto input = o::addStructuredBoundaryCuts(original);
        require(input.success, input.reason);
        auto& q = *input.program.observed;
        const auto owner = q.scopes.back().ownerSite;
        const auto size = q.sites.size();
        for (std::size_t site = 0; site < size; ++site) {
            if (q.sites[site].operation == o::NoControlId) continue;
            const auto cut = q.sites.size(), observation = q.observations.size();
            auto boundary = q.sites[site];
            boundary.operation = o::NoControlId;
            boundary.observation = observation;
            q.observations.push_back({1000 + cut, {}, true});
            q.sites.push_back(std::move(boundary));
            q.sites[site].successors = {cut};
            q.sites[site].backedgeOwners.clear();
        }
        return refine(input.program, owner);
    };
    const auto program = build(body);
    const auto result = accepted(program);
    require(result.channels.size() == 4 && result.work.recurringTrials == 0,
            "pipeline must select two early readiness roles, result readiness and final-reader return directly");
    unsigned ready = 0;
    for (const auto& channel : result.channels) {
        ready += channel.source == P && channel.observer == Q;
        require(channel.source != Q || channel.observer != P,
                "supported operand still allocated its own release channel");
        auto broken = result.commands;
        for (auto& word : broken)
            word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
                return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
                    command.source == channel.source && command.observer == channel.observer &&
                    command.key == channel.key;
            }), word.end());
        require(!o::checkCausalFrontier(program, broken).accepted,
                "removing a supporting pipeline hop must lose memory or rearming evidence");
    }
    require(ready == 2, "input and operand readiness were broadened into one prefix");
    const auto& graph = *program.observed;
    std::vector<std::vector<unsigned>> lengths{{}};
    for (unsigned first : {0u, 1u, 2u, 4u}) {
        lengths.push_back({first});
        for (unsigned second : {0u, 1u, 2u, 4u}) lengths.push_back({first, second});
    }
    lengths.push_back({4, 0, 1});
    lengths.push_back({0, 2, 0, 4});
    for (const auto& entries : lengths) {
        std::vector<unsigned> expected;
        for (auto iterations : entries) {
            expected.push_back(0);
            for (unsigned i = 0; i < iterations; ++i)
                for (unsigned op = 1; op <= 6; ++op) expected.push_back(op);
        }
        std::set<std::pair<std::size_t, std::size_t>> active, dead;
        std::vector<std::size_t> path;
        std::function<bool(std::size_t, std::size_t)> walk = [&](std::size_t at, std::size_t offset) {
            const auto key = std::make_pair(at, offset);
            if (active.count(key) || dead.count(key)) return false;
            const auto& node = graph.sites[at];
            if (node.operation != o::NoControlId) {
                if (offset == expected.size() || node.operation != expected[offset]) return false;
                ++offset;
            }
            path.push_back(at);
            if (at == graph.exit && offset == expected.size()) return true;
            active.insert(key);
            for (auto next : node.successors) if (walk(next, offset)) return true;
            active.erase(key); dead.insert(key); path.pop_back(); return false;
        };
        require(walk(graph.entry, 0), "pipeline trace missing zero/first/steady/tail/reentry case");
        auto flat = program;
        flat.observed.reset(); flat.body = {}; flat.operations.clear();
        o::Commands words;
        std::vector<o::Command> pending;
        std::vector<std::pair<unsigned, unsigned>> forbidden;
        unsigned vectorFences = 0;
        for (auto site : path) {
            pending.insert(pending.end(), result.commands[site].begin(), result.commands[site].end());
            for (const auto& command : result.commands[site])
                vectorFences += command.kind == o::Command::Barrier && command.source == Q;
            const auto operation = graph.sites[site].operation;
            if (operation == o::NoControlId) continue;
            if (operation == 3)
                require(std::none_of(pending.begin(), pending.end(), [&](const auto& command) {
                    return command.kind == o::Command::Barrier && command.source == Q;
                }), "retained normalization completion was repaired again at the row operation");
            if (operation == 2) forbidden.emplace_back(flat.operations.size(), flat.operations.size() + 1);
            flat.operations.push_back(program.operations[operation]);
            words.push_back(std::move(pending)); pending.clear();
        }
        words.push_back(std::move(pending));
        std::vector<unsigned> sequence(flat.operations.size());
        std::iota(sequence.begin(), sequence.end(), 0);
        require(bool(oahs_oracle::graph(flat, words, sequence, forbidden)),
                "pipeline lost memory/rearming or delayed early input use for the independent operand");
        require(vectorFences >= 2 * std::accumulate(entries.begin(), entries.end(), 0u),
                "genuine in-place vector dependencies lost their barriers");
    }
    // An independent reader has no path into the final output receipt.
    auto independent = body;
    independent.operations.push_back(op(o::Pipe::MTE1, {{1, true, false}}));
    independent.body.children[0].children[1].children[0].children.push_back(leaf(7));
    const auto withReader = build(independent);
    const auto separate = accepted(withReader);
    auto missing = separate.commands;
    for (auto& word : missing)
        word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
            return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
                command.source == o::Pipe::MTE1;
        }), word.end());
    require(!o::checkCausalFrontier(withReader, missing).accepted,
            "the output return falsely covered an independent operand reader");
}

void guardedReaderEpisode()
{
    const auto P = o::Pipe::MTE1, Q = o::Pipe::M;
    auto input = base(3, 4);
    input.operations = {
        op(P, {{0, false, true}}),
        op(P, {{1, false, true}}),
        op(Q, {{0, true, false}, {1, true, false}, {2, false, true}}),
        op(o::Pipe::S, {})};
    input.target.keys[unsigned(Q)][unsigned(P)].resize(1);
    input.body = {o::Region::For,
                  {{o::Region::Choice, {seq({leaf(0), leaf(1), leaf(2), leaf(3)}), {}}}},
                  0, true};
    auto converted = o::addStructuredBoundaryCuts(input);
    require(converted.success, converted.reason);
    const auto plan = accepted(converted.program);
    require(plan.channels.size() == 3,
            "shared guarded reader episode reserved per-cell release channels");
    unsigned readiness = 0, release = 0;
    for (const auto& channel : plan.channels) {
        require(channel.owner == o::NoAnalysisId && channel.period == 0,
                "guarded episode invented periodic correspondence");
        if (channel.source == P && channel.observer == Q) {
            ++readiness;
            require(channel.cells.size() == 1,
                    "independent early readiness prefixes were combined");
        } else if (channel.source == Q && channel.observer == P) {
            ++release;
            require(channel.cells == std::vector<unsigned>({0, 1}),
                    "common guarded reader completion was not composed");
        }
    }
    require(readiness == 2 && release == 1 && plan.work.recurringTrials == 0,
            "guarded episode used physical-key trials instead of logical composition");
    require(o::checkCausalFrontier(converted.program, plan.commands).accepted,
            "composed guarded episode failed the independent final checker");

    o::Commands fixed(o::commandCutCount(converted.program));
    fixed[converted.program.observed->entry].push_back(
        {o::Command::Barrier, P, P, 0});
    const auto authored = accepted(converted.program, fixed);
    require(authored.channels.empty(),
            "guarded qualification reinterpreted an authored synchronization protocol");

    auto distinct = input;
    distinct.operations = {
        op(P, {{0, false, true}}), op(P, {{1, false, true}}),
        op(Q, {{0, true, false}, {2, false, true}}),
        op(Q, {{1, true, false}, {2, false, true}})};
    distinct.body = {o::Region::For,
                     {{o::Region::Choice, {seq({leaf(0), leaf(1), leaf(2), leaf(3)}), {}}}},
                     0, true};
    auto separated = o::addStructuredBoundaryCuts(distinct);
    require(separated.success, separated.reason);
    const auto ordinary = accepted(separated.program);
    require(ordinary.channels.empty(),
            "different guarded reader deadlines were composed into a recurring protocol");
}
// Distinct physical-bank episodes stay open across sibling child loops. This
// exercises unrefined control: no residue modes, loop unrolling or new guards.
// Repeated readers keep a single input generation until their own child exit.
// The next first-bank write must not acquire the second child's reader prefix.
o::Program readerRegionProgram(o::Program input)
{
    auto imported = o::addStructuredBoundaryCuts(input);
    require(imported.success, imported.reason);
    auto& p = imported.program;
    auto& g = *p.observed;
    const auto oldSize = g.sites.size();
    for (o::Cut site = 0; site < oldSize; ++site) {
        if (g.sites[site].operation == o::NoControlId) continue;
        auto boundary = g.sites[site];
        boundary.operation = o::NoControlId;
        boundary.observation = g.observations.size();
        g.observations.push_back({1000 + g.sites.size(), {}, true});
        g.sites[site].successors = {g.sites.size()};
        g.sites[site].backedgeOwners.clear();
        g.sites.push_back(std::move(boundary));
    }
    // Portable counterpart of native constant-bound reader-loop metadata.
    for (const auto& scope : g.scopes) {
        if (scope.kind != o::AnalysisContext::ForBody) continue;
        const auto owner = scope.ownerSite, header = g.sites[owner].successors.front();
        o::ObservedLoop loop;
        loop.owner = loop.entry = owner;
        loop.bodyEntry = g.sites[header].successors.front();
        loop.exit = g.sites[header].successors.back(); loop.atLeastOnce = true;
        std::set<o::Cut> seen;
        std::vector<o::Cut> todo{loop.bodyEntry};
        while (!todo.empty()) {
            const auto site = todo.back(); todo.pop_back();
            if (site == header || site == loop.exit || !seen.insert(site).second) continue;
            for (auto next : g.sites[site].successors) todo.push_back(next);
        }
        loop.sites.assign(seen.begin(), seen.end());
        g.loops.push_back(std::move(loop));
    }
    return p;
}

void readerRegionCycles()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::MTE1;
    auto input = base(2, 6);
    input.operations = {op(P, {{0, false, true}}), op(P, {{1, false, true}}),
                        op(Q, {{0, true, false}}), op(Q, {{0, true, false}}),
                        op(Q, {{1, true, false}}), op(Q, {{1, true, false}})};
    input.body = {o::Region::For, {seq({leaf(0), leaf(1),
        {o::Region::For, {seq({leaf(2), leaf(3)})}, 0, true},
        {o::Region::For, {seq({leaf(4), leaf(5)})}, 0, true}})}, 0, true};
    auto p = readerRegionProgram(input);
    const auto& g = *p.observed;
    const auto plan = accepted(p);
    require(plan.channels.size() == 4 && plan.work.recurringTrials == 0,
            "reader regions need two complete cycles without omission trials");
    for (const auto& channel : plan.channels) {
        auto broken = plan.commands;
        for (auto& word : broken) word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
            return command.source == channel.source && command.observer == channel.observer &&
                command.key == channel.key && (command.kind == o::Command::Publish || command.kind == o::Command::Acquire);
        }), word.end());
        require(!o::checkCausalFrontier(p, broken).accepted, "reader-region support was assumed without its channel");
    }
    for (const auto& lengths : std::vector<std::vector<unsigned>>{{}, {1, 2}, {3, 1, 2, 4}, {0, 0, 1, 3}}) {
        std::vector<unsigned> expected;
        for (unsigned pair = 0; pair < lengths.size(); pair += 2) {
            expected.insert(expected.end(), {0, 1});
            for (unsigned i = 0; i < lengths[pair]; ++i) expected.insert(expected.end(), {2, 3});
            for (unsigned i = 0; i < lengths[pair + 1]; ++i) expected.insert(expected.end(), {4, 5});
        }
        std::vector<o::Cut> path;
        std::set<std::pair<o::Cut, unsigned>> active, dead;
        std::function<bool(o::Cut, unsigned)> walk = [&](o::Cut at, unsigned offset) {
            const auto key = std::make_pair(at, offset);
            if (active.count(key) || dead.count(key)) return false;
            const auto& node = g.sites[at];
            if (node.operation != o::NoControlId) {
                if (offset == expected.size() || node.operation != expected[offset]) return false;
                ++offset;
            }
            path.push_back(at);
            if (at == g.exit && offset == expected.size()) return true;
            active.insert(key);
            for (auto next : node.successors) if (walk(next, offset)) return true;
            active.erase(key); dead.insert(key); path.pop_back(); return false;
        };
        require(walk(g.entry, 0), "reader-region varying-length path missing");
        auto flat = p; flat.observed.reset(); flat.body = {}; flat.operations.clear();
        o::Commands words; std::vector<o::Command> pending;
        for (auto cut : path) {
            pending.insert(pending.end(), plan.commands[cut].begin(), plan.commands[cut].end());
            const auto operation = g.sites[cut].operation;
            if (operation == o::NoControlId) continue;
            flat.operations.push_back(p.operations[operation]);
            words.push_back(std::move(pending)); pending.clear();
        }
        words.push_back(std::move(pending));
        std::vector<unsigned> visits(flat.operations.size()); std::iota(visits.begin(), visits.end(), 0);
        std::vector<std::pair<unsigned, unsigned>> forbidden;
        for (unsigned i = 0; i < expected.size(); ++i) if (expected[i] == 0)
            for (unsigned j = 0; j < i; ++j) if (expected[j] == 4 || expected[j] == 5)
                forbidden.emplace_back(j, i);
        require(bool(oahs_oracle::graph(flat, words, visits, forbidden)),
                "unrelated second reader region gates first-bank refill");
    }
    auto independent = p;
    independent.operations[3].pipe = o::Pipe::M;
    const auto separate = accepted(independent);
    require(std::none_of(separate.channels.begin(), separate.channels.end(), [](const auto& channel) {
        return std::find(channel.cells.begin(), channel.cells.end(), 0) != channel.cells.end();
    }), "an independent reader was covered by another engine's return");
    // A producer inside the reader region invalidates the invariant generation.
    auto regenerated = p;
    regenerated.operations[3].pipe = P;
    regenerated.operations[3].accesses = {{0, false, true}};
    const auto conservative = accepted(regenerated);
    require(std::none_of(conservative.channels.begin(), conservative.channels.end(), [](const auto& channel) {
        return std::find(channel.cells.begin(), channel.cells.end(), 0) != channel.cells.end();
    }), "regenerated reader input admitted as invariant");
}

// Final original visits expose the physical release before trailing Q work.
void lastReaderWithinChild()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::MTE1;
    auto input = base(2, 6);
    input.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                        op(Q, {{1, true, false}})};
    input.body = {o::Region::For, {seq({leaf(0),
        {o::Region::For, {seq({leaf(1), leaf(2)})}, 0, true}})}, 0, true};
    auto original = readerRegionProgram(input);
    const auto child = std::find_if(original.observed->loops.begin(), original.observed->loops.end(),
        [&](const auto& loop) {
            return std::none_of(loop.sites.begin(),loop.sites.end(),[&](auto site) {
                return original.observed->sites[site].operation == 0;
            });
        });
    require(child != original.observed->loops.end(), "last-reader child missing");
    const auto owner = child->owner;
    const auto read = *std::find_if(child->sites.begin(),child->sites.end(),[&](auto site) {
        return original.observed->sites[site].operation == 1;
    });
    const auto anchor = original.observed->sites[read].successors.front();
    auto refined = o::refineLastVisit(original, owner, {anchor});
    require(refined.success, refined.reason);
    const auto& p = refined.program;
    auto baseline = accepted(original), plan = accepted(p);
    require(plan.channels.size() == 2, "last-reader cycle not selected");
    require(std::any_of(plan.channels.begin(), plan.channels.end(), [&](const auto& channel) {
        return channel.source == Q && std::any_of(channel.publications.begin(), channel.publications.end(),
            [&](auto cut) {
                const auto& word = p.observed->observations[p.observed->sites[cut].observation];
                return std::any_of(word.atoms.begin(), word.atoms.end(), [&](const auto& a) {
                    return a.kind == o::ObservationAtom::LoopHasNext && a.owner == owner && a.value == 0;
                });
            });
    }), "release remained at the child exit");
    auto evaluate = [&](const o::Program& program, const o::Commands& commands,
                        const std::vector<unsigned>& expected, oahs_oracle::PayloadOrder& order,
                        const std::vector<std::pair<unsigned,unsigned>>& forbidden) {
        const auto& g = *program.observed;
        std::vector<o::Cut> path;
        std::set<std::pair<o::Cut,unsigned>> active, dead;
        std::function<bool(o::Cut,unsigned)> walk = [&](o::Cut at, unsigned offset) {
            const auto key = std::make_pair(at,offset);
            if (active.count(key) || dead.count(key)) return false;
            const auto& site = g.sites[at];
            if (site.operation != o::NoControlId) {
                if (offset == expected.size() || site.operation != expected[offset]) return false;
                ++offset;
            }
            path.push_back(at);
            if (at == g.exit && offset == expected.size()) return true;
            active.insert(key);
            for (auto next : site.successors) if (walk(next,offset)) return true;
            active.erase(key); dead.insert(key); path.pop_back(); return false;
        };
        require(walk(g.entry,0), "last-reader trace missing");
        auto flat = program; flat.observed.reset(); flat.body = {}; flat.operations.clear();
        o::Commands words; std::vector<o::Command> pending;
        for (auto cut : path) {
            const auto observation = g.sites[cut].observation;
            if (observation != o::NoControlId) for (const auto& atom : g.observations[observation].atoms) {
                if (atom.kind != o::ObservationAtom::LoopHasNext || atom.owner != owner) continue;
                bool more = false;
                for (auto i = flat.operations.size(); i < expected.size() && expected[i] != 0; ++i)
                    more |= expected[i] == 1;
                require(atom.value == unsigned(more), "final observation differs from original remaining visits");
            }
            pending.insert(pending.end(),commands[cut].begin(),commands[cut].end());
            const auto op = g.sites[cut].operation;
            if (op == o::NoControlId) continue;
            flat.operations.push_back(program.operations[op]);
            words.push_back(std::move(pending)); pending.clear();
        }
        words.push_back(std::move(pending));
        std::vector<unsigned> visits(flat.operations.size()); std::iota(visits.begin(),visits.end(),0);
        return bool(oahs_oracle::graph(flat,words,visits,forbidden,nullptr,nullptr,&order));
    };
    unsigned removed = 0, paths = 0;
    for (auto lengths : std::vector<std::vector<unsigned>>{{}, {1}, {2}, {4}, {1,3}, {3,1}, {2,4,1}}) {
        std::vector<unsigned> expected;
        std::vector<std::pair<unsigned,unsigned>> forbidden;
        for (auto length : lengths) {
            if (!expected.empty()) forbidden.emplace_back(expected.size()-1,expected.size());
            expected.push_back(0);
            for (unsigned i = 0; i < length; ++i) expected.insert(expected.end(),{1,2});
        }
        oahs_oracle::PayloadOrder before, after;
        require(evaluate(original,baseline.commands,expected,before,{}), "last-reader baseline oracle failed");
        require(evaluate(p,plan.commands,expected,after,forbidden), "trailing Q work still gates the next P write");
        require(std::includes(before.begin(),before.end(),after.begin(),after.end()), "last-reader placement added order");
        removed += before.size()-after.size(); ++paths;
        if (!forbidden.empty()) {
            oahs_oracle::PayloadOrder ignored;
            require(!evaluate(original,baseline.commands,expected,ignored,forbidden), "baseline lacks the discriminating edge");
        }
    }
    require(removed != 0, "last-reader placement removed no ordering");
    for (const auto& channel : plan.channels) {
        auto broken = plan.commands;
        for (auto& word : broken) word.erase(std::remove_if(word.begin(),word.end(),[&](const auto& command) {
            return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
                command.source == channel.source && command.observer == channel.observer && command.key == channel.key;
        }),word.end());
        require(!o::checkCausalFrontier(p,broken).accepted, "last-reader cycle inferred missing support");
        oahs_oracle::PayloadOrder ignored;
        require(!evaluate(p,broken,{0,1,2,0,1,2},ignored,{}), "last-reader independent oracle inferred missing support");
    }
    auto empty = original;
    for (auto& loop : empty.observed->loops) if (loop.owner == owner) loop.atLeastOnce = false;
    require(!o::refineLastVisit(empty,owner,{anchor}).success, "unknown/empty loop admitted as nonempty");
    require(!o::refineLastVisit(p,owner,{anchor}).success, "last-visit refinement repeated");
    auto guarded = original;
    guarded.observed->sites[read].successors.push_back(anchor);
    require(!o::refineLastVisit(guarded,owner,{anchor}).success, "guarded body admitted as a straight last visit");
    auto repeated = plan.commands;
    for (const auto& channel : plan.channels) if (channel.source == Q)
        for (auto cut : channel.publications) {
            const auto& observation = p.observed->observations[p.observed->sites[cut].observation];
            if (observation.atoms.empty()) continue;
            for (const auto& command : plan.commands[cut])
                if (command.kind == o::Command::Publish && command.source == Q && command.observer == P)
                    repeated[anchor].push_back(command);
        }
    require(!o::checkCausalFrontier(p,repeated).accepted, "release on every body visit passed the checker");
    oahs_oracle::PayloadOrder ignored;
    require(!evaluate(p,repeated,{0,1,2,1,2,0,1,2},ignored,{}), "oracle accepted repeated release");
    auto laterRead = p; laterRead.operations[2].accesses = {{0,true,false}};
    const auto fallback = accepted(laterRead);
    for (const auto& channel : fallback.channels) if (channel.source == Q)
        for (auto cut : channel.publications)
            require(p.observed->observations[p.observed->sites[cut].observation].atoms.empty(),
                    "later physical read ignored by early release");
    std::cout << "last-reader paths=" << paths << " removed-payload-relations=" << removed << '\n';
}

// One generation is read by two children, followed by its next overwrite.
// Region summaries guide endpoints; only the realized cycle grants credit.
void retainedReaderRegions()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::MTE1;
    auto input = base(2, 6);
    input.operations = {op(P, {{0, false, true}}), op(Q, {}),
                        op(Q, {{0, true, false}}), op(Q, {{0, true, false}}),
                        op(Q, {{0, true, false}}), op(Q, {{0, true, false}})};
    input.body = {o::Region::For, {seq({leaf(0),
        {o::Region::For, {seq({leaf(2), leaf(3)})}, 0, true},
        {o::Region::For, {seq({leaf(4), leaf(5)})}, 0, true}, leaf(1)})}, 0, true};
    auto p = readerRegionProgram(input);
    const auto plan = accepted(p);
    require(plan.channels.size() == 2 && plan.work.recurringTrials == 0,
            "retained input needs one complete cycle across both children");
    for (const auto& channel : plan.channels) {
        require(channel.cells == std::vector<unsigned>{0}, "unrelated writer joined retained cycle");
        require(channel.publications.size() == (channel.source == P ? 1u : 2u),
                "retained input published at an intermediate child boundary");
        require(channel.acquisitions.size() == (channel.source == P ? 1u : 2u),
                "retained readiness reacquired for the second child");
    }
    o::SelectedOptions ordinary;
    ordinary.recurring = false;
    const auto baseline = o::constructSelectedPlan(p, {}, ordinary);
    require(baseline.success, baseline.reason);
    const auto& g = *p.observed;
    auto premature = plan.commands;
    const auto& release = *std::find_if(plan.channels.begin(), plan.channels.end(),
        [&](const auto& channel) { return channel.source == Q; });
    std::vector<o::Cut> children;
    for (const auto& loop : g.loops) {
        if (std::any_of(loop.sites.begin(), loop.sites.end(), [&](auto site) {
                return g.sites[site].operation == 0;
            })) continue;
        children.push_back(loop.exit);
    }
    require(children.size() == 2, "retained fixture child count");
    const auto early = o::canonicalCommandCut(p, children.front());
    const auto late = o::canonicalCommandCut(p, children.back());
    auto& word = premature[late];
    auto endpoint = std::find_if(word.begin(), word.end(), [&](const auto& command) {
        return command.kind == o::Command::Publish && command.source == Q &&
            command.observer == P && command.key == release.key;
    });
    require(endpoint != word.end(), "last-reader release missing");
    const auto command = *endpoint;
    word.erase(endpoint); premature[early].push_back(command);
    require(!o::checkCausalFrontier(p, premature).accepted,
            "release after first child incorrectly covers the second child");
    unsigned removed = 0;
    for (const auto& lengths : std::vector<std::vector<unsigned>>{{}, {1, 1}, {0, 2}, {3, 0},
            {0, 0}, {1, 3, 2, 1}, {2, 0, 0, 3}, {3, 2, 1, 4}}) {
        std::vector<unsigned> expected;
        for (unsigned pair = 0; pair < lengths.size(); pair += 2) {
            expected.push_back(0);
            for (unsigned i = 0; i < lengths[pair]; ++i) expected.insert(expected.end(), {2, 3});
            for (unsigned i = 0; i < lengths[pair + 1]; ++i) expected.insert(expected.end(), {4, 5});
            expected.push_back(1);
        }
        std::vector<o::Cut> path;
        std::set<std::pair<o::Cut, unsigned>> active, dead;
        std::function<bool(o::Cut, unsigned)> walk = [&](o::Cut at, unsigned offset) {
            const auto key = std::make_pair(at, offset);
            if (active.count(key) || dead.count(key)) return false;
            const auto operation = g.sites[at].operation;
            if (operation != o::NoControlId) {
                if (offset == expected.size() || operation != expected[offset]) return false;
                ++offset;
            }
            path.push_back(at);
            if (at == g.exit && offset == expected.size()) return true;
            active.insert(key);
            for (auto next : g.sites[at].successors) if (walk(next, offset)) return true;
            active.erase(key); dead.insert(key); path.pop_back(); return false;
        };
        require(walk(g.entry, 0), "retained-reader varying-length trace missing");
        auto flatten = [&](const o::Commands& commands, oahs_oracle::PayloadOrder& order) {
            auto flat = p; flat.observed.reset(); flat.body = {}; flat.operations.clear();
            o::Commands words; std::vector<o::Command> pending;
            for (auto cut : path) {
                pending.insert(pending.end(), commands[cut].begin(), commands[cut].end());
                const auto operation = g.sites[cut].operation;
                if (operation == o::NoControlId) continue;
                flat.operations.push_back(p.operations[operation]);
                words.push_back(std::move(pending)); pending.clear();
            }
            words.push_back(std::move(pending));
            std::vector<unsigned> visits(flat.operations.size()); std::iota(visits.begin(), visits.end(), 0);
            return oahs_oracle::graph(flat, words, visits, {}, nullptr, nullptr, &order);
        };
        oahs_oracle::PayloadOrder before, after;
        require(bool(flatten(baseline.commands, before)) && bool(flatten(plan.commands, after)),
                "retained cycle failed independent memory/balance/rearming checks");
        require(std::includes(before.begin(), before.end(), after.begin(), after.end()),
                "retained cycle introduced payload ordering");
        if (lengths.size() >= 4 && lengths[1]) {
            oahs_oracle::PayloadOrder bad;
            require(!flatten(premature, bad), "independent oracle accepted premature child release");
        }
        removed += before.size() - after.size();
    }
    require(removed != 0, "retained cycle did not improve the broad ordinary repair");
    for (const auto& channel : plan.channels) {
        auto broken = plan.commands;
        for (auto& word : broken) word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
            return command.source == channel.source && command.observer == channel.observer &&
                command.key == channel.key && (command.kind == o::Command::Publish || command.kind == o::Command::Acquire);
        }), word.end());
        require(!o::checkCausalFrontier(p, broken).accepted, "retained generation credited without actual transfer");
    }
    auto independent = p;
    independent.operations[4].pipe = o::Pipe::M;
    const auto separate = accepted(independent);
    require(separate.channels.empty(), "independent engine completion inferred from final reader");
    auto reloaded = input;
    reloaded.operations.push_back(op(P, {{0, false, true}}));
    reloaded.body.children[0].children.insert(reloaded.body.children[0].children.begin() + 2, leaf(6));
    const auto reloadPlan = accepted(readerRegionProgram(reloaded));
    require(reloadPlan.channels.size() == 2, "reloaded input lost complete per-generation cycle");
    for (const auto& channel : reloadPlan.channels) if (channel.source == P)
        require(channel.publications.size() == 2 && channel.acquisitions.size() == 2,
                "old readiness was retained across an actual reload");
    auto unrelated = input;
    unrelated.operations[1] = op(P, {{1, false, true}});
    // Put the unrelated write after the retained write: removing the earlier
    // producer fence must not move its repair across the current generation.
    unrelated.body.children[0].children.pop_back();
    unrelated.body.children[0].children.insert(unrelated.body.children[0].children.begin() + 1, leaf(1));
    const auto declined = accepted(readerRegionProgram(unrelated));
    require(declined.channels.empty(), "uncertified producer-fence motion admitted");
    std::cout << "retained-reader paths=8 removed-payload-relations=" << removed << '\n';
}

// Original observed paths are flattened only in the independent test oracle.
// Production qualification never enumerates child lengths or parent visits.
bool readerTraceOrder(const o::Program& p, const o::Commands& commands,
                      const std::vector<unsigned>& expected, oahs_oracle::PayloadOrder& order)
{
    const auto& g = *p.observed;
    std::vector<o::Cut> path;
    std::set<std::pair<o::Cut, unsigned>> active, dead;
    std::function<bool(o::Cut, unsigned)> walk = [&](o::Cut at, unsigned offset) {
        const auto key = std::make_pair(at, offset);
        if (active.count(key) || dead.count(key)) return false;
        const auto operation = g.sites[at].operation;
        if (operation != o::NoControlId) {
            if (offset == expected.size() || operation != expected[offset]) return false;
            ++offset;
        }
        path.push_back(at);
        if (at == g.exit && offset == expected.size()) return true;
        active.insert(key);
        for (auto next : g.sites[at].successors) if (walk(next, offset)) return true;
        active.erase(key); dead.insert(key); path.pop_back(); return false;
    };
    require(walk(g.entry, 0), "multi-input original path missing");
    auto flat = p; flat.observed.reset(); flat.body = {}; flat.operations.clear();
    o::Commands words; std::vector<o::Command> pending;
    for (auto cut : path) {
        pending.insert(pending.end(), commands[cut].begin(), commands[cut].end());
        const auto operation = g.sites[cut].operation;
        if (operation == o::NoControlId) continue;
        flat.operations.push_back(p.operations[operation]);
        words.push_back(std::move(pending)); pending.clear();
    }
    words.push_back(std::move(pending));
    std::vector<unsigned> visits(flat.operations.size()); std::iota(visits.begin(), visits.end(), 0);
    return bool(oahs_oracle::graph(flat, words, visits, {}, nullptr, nullptr, &order));
}

o::Program multiReaderInput(bool secondRetained)
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::MTE1;
    auto input = base(2, 6);
    input.operations = {op(P, {{0, false, true}}), op(P, {{1, false, true}}),
                        op(Q, {{0, true, false}}), op(Q, {{1, true, false}}),
                        op(Q, {{0, true, false}}), op(Q, {{1, true, false}}), op(Q, {})};
    input.body = {o::Region::For, {seq({leaf(0), leaf(1),
        {o::Region::For, {seq({leaf(2)})}, 0, true},
        {o::Region::For, {seq({leaf(3)})}, 0, true},
        {o::Region::For, {seq({leaf(4)})}, 0, true}})}, 0, true};
    auto& sequence = input.body.children.front().children;
    if (secondRetained) sequence.push_back({o::Region::For, {seq({leaf(5)})}, 0, true});
    else input.operations.erase(input.operations.begin() + 5);
    sequence.push_back(leaf(secondRetained ? 6 : 5));
    return input;
}

void retainedProducerCohort()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::MTE1;
    unsigned paths = 0, removed = 0;
    for (bool secondRetained : {false, true}) {
        auto input = multiReaderInput(secondRetained);
        const auto p = readerRegionProgram(input);
        const auto plan = accepted(p);
        require(plan.channels.size() == 4 && plan.work.recurringTrials == 0,
                "multi-input reader cohort lost separate complete cycles");
        require(plan.work.rejectedSupportProposals == 0 &&
                std::none_of(plan.fences.begin(), plan.fences.end(), [&](const auto& f) { return f.observer == P; }),
                "accepted producer cohort still needs a local repair");
        std::set<std::vector<o::Cut>> sources, consumers, releases;
        for (const auto& channel : plan.channels) {
            if (channel.source == P) {
                sources.insert(channel.publications); consumers.insert(channel.acquisitions);
            } else releases.insert(channel.publications);
        }
        require(sources.size() == 2 && consumers.size() == 2 && releases.size() == 2,
                "distinct readiness or last-reader boundaries were merged");
        o::SelectedOptions ordinary; ordinary.recurring = false;
        const auto baseline = o::constructSelectedPlan(p, {}, ordinary);
        require(baseline.success, baseline.reason);
        for (const auto& lengths : std::vector<std::vector<unsigned>>{
                {}, {1,1,1,1}, {0,2,3,0}, {3,0,0,2}, {0,0,0,0},
                {1,3,2,1, 2,0,1,3}, {0,0,0,0, 2,1,0,3}, {3,2,1,4, 1,4,3,2}}) {
            std::vector<unsigned> expected;
            for (unsigned entry = 0; entry < lengths.size(); entry += 4) {
                expected.insert(expected.end(), {0,1});
                for (unsigned child = 0; child < (secondRetained ? 4u : 3u); ++child)
                    expected.insert(expected.end(), lengths[entry + child], 2 + child);
                expected.push_back(secondRetained ? 6 : 5);
            }
            oahs_oracle::PayloadOrder before, after;
            require(readerTraceOrder(p, baseline.commands, expected, before) &&
                    readerTraceOrder(p, plan.commands, expected, after),
                    "multi-input trace violates memory, balance or rearming");
            require(std::includes(before.begin(), before.end(), after.begin(), after.end()),
                    "multi-input admission moved a repair and added payload ordering");
            if (!lengths.empty() && lengths[0])
                require(!after.count({3,4}), "later Y load gates the first X extraction");
            removed += before.size() - after.size();
            ++paths;
        }
        unsigned missingSupportFailures = 0;
        for (const auto& channel : plan.channels) {
            auto broken = plan.commands;
            for (auto& word : broken) word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
                return command.source == channel.source && command.observer == channel.observer &&
                    command.key == channel.key && (command.kind == o::Command::Publish || command.kind == o::Command::Acquire);
            }), word.end());
            const auto accepted = o::checkCausalFrontier(p, broken).accepted;
            oahs_oracle::PayloadOrder bad;
            const std::vector<unsigned> visits = secondRetained
                ? std::vector<unsigned>{0,1,2,3,4,5,6,0,1,2,3,4,5,6}
                : std::vector<unsigned>{0,1,2,3,4,5,0,1,2,3,4,5};
            require(readerTraceOrder(p, broken, visits, bad) == accepted,
                    "independent oracle disagrees on removed multi-input support");
            if (!accepted) ++missingSupportFailures;
            if (channel.source == P || channel.cell == 0)
                require(!accepted, "required readiness/early X return was inferred without its transfer");
        }
        // An earlier Y reader can already be covered by X's later return.
        // That optional sharing is deliberately not a channel-deletion policy.
        require(missingSupportFailures >= 3, "missing-support negatives were not discriminating");
        const auto release = std::find_if(plan.channels.begin(), plan.channels.end(),
            [&](const auto& channel) { return channel.source == Q && channel.cell == 0; });
        auto premature = plan.commands;
        const auto& graph = *p.observed;
        const auto firstChild = std::find_if(graph.loops.begin(), graph.loops.end(), [&](const auto& loop) {
            return std::any_of(loop.sites.begin(), loop.sites.end(), [&](auto site) { return graph.sites[site].operation == 2; }) &&
                std::none_of(loop.sites.begin(), loop.sites.end(), [&](auto site) { return graph.sites[site].operation == 0; });
        });
        require(firstChild != graph.loops.end(), "first X child missing");
        const auto early = o::canonicalCommandCut(p, firstChild->exit);
        for (auto cut : release->publications) {
            if (cut == o::canonicalCommandCut(p, graph.entry)) continue;
            auto& word = premature[cut];
            auto publication = std::find_if(word.begin(), word.end(), [&](const auto& command) {
                return command.kind == o::Command::Publish && command.source == Q &&
                    command.observer == P && command.key == release->key;
            });
            require(publication != word.end(), "X release not found");
            const auto command = *publication; word.erase(publication); premature[early].push_back(command);
        }
        require(!o::checkCausalFrontier(p, premature).accepted, "multi-input X released after only its first child");
        oahs_oracle::PayloadOrder prematureOrder;
        const std::vector<unsigned> repeated = secondRetained
            ? std::vector<unsigned>{0,1,2,3,4,5,6,0,1,2,3,4,5,6}
            : std::vector<unsigned>{0,1,2,3,4,5,0,1,2,3,4,5};
        require(!readerTraceOrder(p, premature, repeated, prematureOrder), "independent oracle accepted premature multi-input release");
        auto reload = input;
        reload.operations.push_back(op(P, {{0, false, true}}));
        auto& sequence = reload.body.children.front().children;
        sequence.insert(sequence.begin() + 4, leaf(reload.operations.size() - 1));
        const auto reloaded = accepted(readerRegionProgram(reload));
        for (const auto& channel : reloaded.channels) if (channel.cell == 0 && channel.source == P)
            require(channel.publications.size() == 2 && channel.acquisitions.size() == 2,
                    "multi-input readiness survived an actual X reload");
        auto tight = input;
        tight.target.keys[unsigned(P)][unsigned(Q)] = {0, 1};
        tight.target.keys[unsigned(Q)][unsigned(P)] = {0};
        const auto tightProgram = readerRegionProgram(tight);
        const auto fallback = o::constructSelectedPlan(tightProgram);
        const auto tightOrdinary = o::constructSelectedPlan(tightProgram, {}, ordinary);
        require(fallback.channels.empty() && fallback.work.rejectedResourceProposals == 1,
                "partial multi-input cohort escaped capacity fallback");
        require(fallback.success == tightOrdinary.success && fallback.reason == tightOrdinary.reason,
                "capacity fallback changed ordinary construction's outcome");
        auto uncovered = input;
        uncovered.operations[3].accesses.clear(); uncovered.operations[5].accesses.clear();
        const auto declined = accepted(readerRegionProgram(uncovered));
        require(declined.channels.empty(), "uncovered Y writer allowed X's fence to move");
        auto independent = input;
        independent.operations[4].pipe = o::Pipe::M;
        const auto otherReader = accepted(readerRegionProgram(independent));
        require(std::none_of(otherReader.channels.begin(), otherReader.channels.end(),
                    [](const auto& channel) { return channel.cell == 0; }),
                "independent X reader was covered by a different engine's release");
    }
    require(removed != 0, "multi-input cycle did not recover any payload independence");
    std::cout << "retained-cohort paths=" << paths << " removed-payload-relations=" << removed << '\n';
}

void independentBankEpisodes()
{
    const auto P = o::Pipe::MTE1, Q = o::Pipe::M;
    auto input = base(3, 4);
    input.operations = {
        op(P, {{0, false, true}}), op(P, {{1, false, true}}),
        op(Q, {{0, true, false}}), op(Q, {{1, true, false}}),
        op(Q, {{0, true, false}}), op(o::Pipe::MTE2, {{2, false, true}}),
        op(o::Pipe::S, {})};
    const auto episode = seq({leaf(0), leaf(1),
                              {o::Region::Choice, {leaf(2), leaf(4)}}, leaf(3)});
    const o::Region child{o::Region::For,
                         {seq({leaf(6), {o::Region::Choice, {episode, seq({})}}})}, 0, true};
    auto secondChild = child;
    std::function<void(o::Region&)> renumber = [&](o::Region& region) {
        if (region.kind == o::Region::Operation) region.operation += 7;
        for (auto& nested : region.children) renumber(nested);
    };
    renumber(secondChild);
    const auto firstOperations = input.operations;
    input.operations.insert(input.operations.end(), firstOperations.begin(), firstOperations.end());
    input.body = {o::Region::For, {seq({leaf(5), child, leaf(12), secondChild})}, 0, true};
    auto imported = o::addStructuredBoundaryCuts(input);
    require(imported.success, imported.reason);
    // Native SCF import has legal after-operation/yield cuts on both arms.
    auto& control = *imported.program.observed;
    const auto originalSites = control.sites.size();
    for (o::Cut site = 0; site < originalSites; ++site) {
        if (control.sites[site].operation == o::NoControlId) continue;
        const auto after = control.sites.size();
        auto boundary = control.sites[site];
        boundary.operation = o::NoControlId;
        boundary.observation = control.observations.size();
        control.observations.push_back({1000 + after, {}, true});
        control.sites.push_back(std::move(boundary));
        control.sites[site].successors = {after};
        control.sites[site].backedgeOwners.clear();
    }
    const auto& p = imported.program;
    const auto plan = accepted(p);
    require(plan.channels.size() == 4 && plan.work.recurringTrials == 0,
            "distinct banks must select their own complete interfaces directly");
    for (const auto& channel : plan.channels) {
        require(channel.cells.size() == 1 && channel.owner == o::NoAnalysisId && channel.period == 0,
                "distinct reader frontiers were merged or assigned a fabricated period");
        auto broken = plan.commands;
        for (auto& word : broken)
            word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
                return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
                    command.source == channel.source && command.observer == channel.observer &&
                    command.key == channel.key;
            }), word.end());
        require(!o::checkCausalFrontier(p, broken).accepted,
                "missing bank readiness/release was credited by the interface alone");
    }
    const auto& g = *p.observed;
    for (const auto& lengths : std::vector<std::vector<unsigned>>{{}, {0, 0}, {1, 2},
                                                               {3, 0, 0, 2}, {2, 3, 1, 4}}) {
        for (unsigned mask = 0; mask < 8; ++mask) {
            std::vector<unsigned> expected;
            unsigned iteration = 0, phase = 0;
            for (auto length : lengths) {
                const auto base = 7 * (phase++ % 2);
                expected.push_back(base + 5);
                for (unsigned i = 0; i < length; ++i, ++iteration) {
                    expected.push_back(base + 6);
                    if (!(mask & (1u << (iteration % 3)))) continue;
                    for (auto operation : {0u, 1u, iteration % 2 ? 4u : 2u, 3u})
                        expected.push_back(base + operation);
                }
            }
            std::vector<o::Cut> path;
            std::set<std::pair<o::Cut, unsigned>> active, dead;
            std::function<bool(o::Cut, unsigned)> walk = [&](o::Cut at, unsigned offset) {
                const auto key = std::make_pair(at, offset);
                if (active.count(key) || dead.count(key)) return false;
                const auto& node = g.sites[at];
                if (node.operation != o::NoControlId) {
                    if (offset == expected.size() || node.operation != expected[offset]) return false;
                    ++offset;
                }
                path.push_back(at);
                if (at == g.exit && offset == expected.size()) return true;
                active.insert(key);
                for (auto next : node.successors) if (walk(next, offset)) return true;
                active.erase(key); dead.insert(key); path.pop_back(); return false;
            };
            require(walk(g.entry, 0), "missing skipped/sibling/reentered bank trace");
            auto flat = p; flat.observed.reset(); flat.body = {}; flat.operations.clear();
            o::Commands words; std::vector<o::Command> pending;
            for (auto cut : path) {
                pending.insert(pending.end(), plan.commands[cut].begin(), plan.commands[cut].end());
                const auto operation = g.sites[cut].operation;
                if (operation == o::NoControlId) continue;
                flat.operations.push_back(p.operations[operation]);
                words.push_back(std::move(pending)); pending.clear();
            }
            words.push_back(std::move(pending));
            std::vector<unsigned> visits(flat.operations.size());
            std::iota(visits.begin(), visits.end(), 0);
            std::vector<std::pair<unsigned, unsigned>> forbidden;
            unsigned lastB = unsigned(expected.size());
            for (unsigned i = 0; i < expected.size(); ++i) {
                const auto operation = expected[i] % 7;
                if (operation == 3) lastB = i;
                if ((operation == 0 || operation == 5) && lastB < i)
                    forbidden.emplace_back(lastB, i);
                if (operation == 2 || operation == 4)
                    forbidden.emplace_back(i - 1, i);
            }
            require(bool(oahs_oracle::graph(flat, words, visits, forbidden)),
                    "bank protocol delayed early readiness, acquired the wrong reader, or drained at child exit");
        }
    }
}

// The frontend specializes physical effects, never payload order. Check that
// ACC reuse remains a compute obligation and cannot gate the next bank fill.
void carriedBankEffects(bool reentered, unsigned banks)
{
    auto p = base(banks + 1, banks);
    const auto P = o::Pipe::MTE1, Q = o::Pipe::M;
    p.operations = {op(P, {}), op(Q, {{banks, true, true}})};
    for (unsigned cell = 0; cell < banks; ++cell) {
        p.operations[0].accesses.push_back({cell, false, true});
        p.operations[1].accesses.push_back({cell, true, false});
    }
    p.operations[0].original = 0;
    p.operations[1].original = 1;
    p.body = {o::Region::For, {seq({leaf(0), leaf(1)})}, 0, true};
    if (reentered) p.body = {o::Region::For, {p.body}, 0, true};
    auto input = o::addStructuredBoundaryCuts(p);
    require(input.success, input.reason);
    const auto &q = *input.program.observed;
    o::CountedLoopRegion loop;
    loop.owner = q.scopes.back().ownerSite;
    loop.header = q.sites[loop.owner].successors.front();
    loop.bodyEntry = q.sites[loop.header].successors.front();
    loop.continuation = q.sites[loop.header].successors.back();
    loop.period = banks;
    std::set<o::Cut> seen;
    std::vector<o::Cut> todo{loop.bodyEntry};
    while (!todo.empty()) {
        auto at = todo.back(); todo.pop_back();
        if (at == loop.header || !seen.insert(at).second) continue;
        loop.bodySites.push_back(at);
        for (auto next : q.sites[at].successors) todo.push_back(next);
    }
    loop.effects = {{0, {}}, {1, {}}};
    for (unsigned residue = 0; residue < banks; ++residue) {
        const unsigned cell = banks == 3 ? (2 * residue + 1) % 3 : residue;
        loop.effects[0].residues.push_back({{cell, false, true}});
        loop.effects[1].residues.push_back({{cell, true, false}, {banks, true, true}});
    }
    const auto refined = o::refineCountedLoop(input.program, loop);
    require(refined.success, refined.reason);
    const auto &program = refined.program;
    const auto plan = accepted(program);
    require(plan.channels.size() == 2 * banks, "banks need separate ready/release pairs");
    for (const auto &channel : plan.channels)
        require(std::find(channel.cells.begin(), channel.cells.end(), banks) == channel.cells.end(),
                "ACC reuse merged into operand-bank release");
    const auto &control = *program.observed;
    if (reentered) {
        for (const auto &channel : plan.channels) {
            if (channel.source != Q) continue;
            require(std::find(channel.publications.begin(), channel.publications.end(), control.entry) !=
                        channel.publications.end(), "open release lacks one invocation prime");
            require(std::find(channel.acquisitions.begin(), channel.acquisitions.end(), control.exit) !=
                        channel.acquisitions.end(), "open release lacks final invocation consumption");
            const auto key = std::find_if(plan.certificate.keys.begin(), plan.certificate.keys.end(),
                [&](const auto &k) { return k.source == Q && k.observer == P && k.key == channel.key; });
            require(key != plan.certificate.keys.end(), "open release key missing");
            const auto index = key - plan.certificate.keys.begin();
            const auto interface = std::find_if(plan.loops.begin(), plan.loops.end(),
                [&](const auto &l) { return l.owner == loop.owner; });
            require(interface != plan.loops.end() &&
                        interface->outgoing.facts()->events[index].occupancy == 2,
                    "child exit consumed or forgot its outstanding bank release");
            auto broken = plan.commands;
            auto &word = broken[control.exit];
            word.erase(std::remove_if(word.begin(), word.end(), [&](const auto &cmd) {
                return cmd.kind == o::Command::Acquire && cmd.source == Q &&
                       cmd.observer == P && cmd.key == channel.key;
            }), word.end());
            require(!o::checkCausalFrontier(program, broken).accepted,
                    "final invocation consumption deletion accepted");
        }
    }
    for (unsigned iterations = 0; iterations <= 2 * banks + 3; ++iterations) {
        std::vector<o::Cut> path;
        std::vector<unsigned> visits(control.sites.size());
        const unsigned entries = reentered ? 2 : 1, total = 2 * iterations * entries;
        std::function<bool(o::Cut, unsigned, unsigned, unsigned)> trace =
            [&](o::Cut at, unsigned payloads, unsigned iteration, unsigned entered) {
            if (payloads > total || visits[at] > 2 * (iterations + 1)) return false;
            if (at == loop.owner) { iteration = 0; if (++entered > entries) return false; }
            if (at == loop.continuation && iteration != iterations) return false;
            const auto &node = control.sites[at];
            if (node.observation != o::NoAnalysisId) {
                for (const auto &atom : control.observations[node.observation].atoms) {
                    if (atom.owner != loop.owner) continue;
                    bool actual = true;
                    if (atom.kind == o::ObservationAtom::LoopResidue)
                        actual = iteration % atom.parameter == atom.value;
                    else if (atom.kind == o::ObservationAtom::LoopHasPrevious)
                        actual = (iteration >= atom.parameter) == bool(atom.value);
                    else if (atom.kind == o::ObservationAtom::LoopHasNext)
                        actual = (iterations > iteration && iterations - iteration > atom.parameter) == bool(atom.value);
                    if (!actual) return false;
                }
            }
            path.push_back(at);
            ++visits[at];
            if (at == control.exit && payloads == total && entered == entries) return true;
            const auto op = node.operation;
            if (op != o::NoAnalysisId) {
                if (payloads == total || program.operations[op].original != payloads % 2) {
                    --visits[at]; path.pop_back(); return false;
                }
                ++payloads;
            }
            for (std::size_t edge = 0; edge < node.successors.size(); ++edge) {
                const bool back = !node.backedgeOwners.empty() && node.backedgeOwners[edge] == loop.owner;
                if (trace(node.successors[edge], payloads, iteration + unsigned(back), entered)) return true;
            }
            --visits[at]; path.pop_back(); return false;
        };
        require(trace(control.entry, 0, 0, 0), "missing first/steady/tail/reentry trace");
        auto flat = program;
        flat.observed.reset(); flat.body = {}; flat.operations.clear();
        o::Commands words;
        std::vector<o::Command> pending;
        for (auto site : path) {
            pending.insert(pending.end(), plan.commands[site].begin(), plan.commands[site].end());
            const auto operation = control.sites[site].operation;
            if (operation == o::NoAnalysisId) continue;
            flat.operations.push_back(program.operations[operation]);
            words.push_back(std::move(pending)); pending.clear();
        }
        words.push_back(std::move(pending));
        std::vector<unsigned> sequence(flat.operations.size());
        std::iota(sequence.begin(), sequence.end(), 0);
        std::vector<std::pair<unsigned, unsigned>> forbidden;
        for (unsigned entry = 0; entry < entries; ++entry)
            for (unsigned i = 0; i + 1 < iterations; ++i) {
                const unsigned base = 2 * entry * iterations;
                forbidden.push_back({base+2*i+1, base+2*i+2});
            }
        require(bool(oahs_oracle::graph(flat, words, sequence, forbidden)),
                "next-bank preparation acquired unrelated current-bank compute (or lost required order)");
    }
    auto malformed = loop;
    malformed.effects[0].residues.pop_back();
    require(!o::refineCountedLoop(input.program, malformed).success, "partial orbit binding accepted");
    malformed = loop;
    malformed.effects[0].residues[0][0].definiteWrite = true;
    require(!o::refineCountedLoop(input.program, malformed).success, "geometry conferred full-write credit");
}
} // namespace
int main()
{
    for (unsigned banks : {2u, 3u}) {
        carriedBankEffects(false, banks);
        carriedBankEffects(true, banks);
    }
    regional();
    contextual();
    sharedRecurringPrefixes();
    distinctReleaseDeadlines();
    transitiveRecurringCoverage();
    pipelineOperandSupport();
    guardedReaderEpisode();
    independentBankEpisodes();
    readerRegionCycles();
    lastReaderWithinChild();
    retainedReaderRegions();
    retainedProducerCohort();
    for (unsigned slots = 1; slots <= 4; ++slots) {
        checkSlots(slots);
    }
}
