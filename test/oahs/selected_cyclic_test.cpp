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
    require(result.work.recurringLocalPackets != 0 && result.work.recurringAnalysisSites == 0 &&
            result.work.recurringReplaySites == 0,
            "simple slot cycle did not use immutable local qualification");
    require(result.work.recurringActivations != 0 && !result.declinedRecurring,
            "normal constructor did not activate required cyclic support");
    for (const auto& activation : result.activations) {
        require(!activation.before.empty() && activation.after.size() < activation.before.size(),
                "recurring support was installed without actual deadline progress");
    }
    const auto covered = accepted(input.program, result.commands);
    require(covered.work.recurringFamilies != 0 && covered.activations.empty() && covered.channels.empty(),
            "already covered recurring opportunities allocated another protocol");
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
    // Both physical obligations have identical publication and acquisition
    // gaps. Sharing must not manufacture equality by moving either endpoint.
    body.operations = {op(P, {{0, false, true, true}, {1, false, true, true}}),
                       op(Q, {{0, true, false}, {1, true, false}})};
    const auto input = o::makePeriodicLoop(body, 1, {});
    require(input.success, input.reason);
    const auto result = accepted(input.program);
    require(result.channels.size() == 2,
            "common matrix consumer and reuse frontier must share recurring ready/release prefixes");
    require(result.work.recurringFamilies == 2 && result.activations.size() == 1 &&
                result.activations.front().families.size() == 2,
            "identical complete protocols lost their constituent physical witnesses");
    for (const auto& channel : result.channels) {
        require(channel.cells == std::vector<unsigned>({0, 1}),
                "shared recurring channel must retain both physical obligations");
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
    // The old three-channel result was obtained by completed-population
    // omission trials. Its quality recovery needs a finite RMW support recipe;
    // keep this service/negative witness without restoring that search policy.
    require(result.work.recurringTrials == 0, "recurring construction restored omission search");
    std::set<std::tuple<o::Pipe, o::Pipe, unsigned>> channels;
    for (const auto& word : result.commands) {
        for (const auto& command : word) {
            if (command.kind == o::Command::Acquire) {
                channels.emplace(command.source, command.observer, command.key);
            }
        }
    }
    require(!channels.empty(), "RMW recurrence lost all actual transfers");
    unsigned necessary = 0;
    for (const auto& channel : channels) {
        auto words = result.commands;
        for (auto& word : words) {
            word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
                return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
                    std::make_tuple(command.source, command.observer, command.key) == channel;
            }), word.end());
        }
        necessary += !o::checkCausalFrontier(program, words).accepted;
    }
    require(necessary != 0, "RMW channel deletion negatives became vacuous");
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
    transitiveRecurringCoverage();
    for (unsigned slots = 1; slots <= 4; ++slots) {
        checkSlots(slots);
    }
}
