// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
#include "../../lib/PTO/Transforms/OAHS/SelectedInternal.h"
#include <functional>
#include <set>
using namespace selected_test;
namespace {
o::CountedLoopRegion region(const o::Program& p, std::size_t owner)
{
    o::CountedLoopRegion loop;
    const auto& q = *p.observed;
    loop.owner = owner;
    loop.header = q.sites[owner].successors.front();
    loop.bodyEntry = q.sites[loop.header].successors.front();
    loop.continuation = q.sites[loop.header].successors.back();
    loop.period = 2;
    loop.atLeastOnce = true;
    std::vector<std::size_t> todo{loop.bodyEntry};
    std::set<std::size_t> seen;
    while (!todo.empty()) {
        const auto at = todo.back();
        todo.pop_back();
        if (at == loop.header || !seen.insert(at).second) {
            continue;
        }
        loop.bodySites.push_back(at);
        for (auto next : q.sites[at].successors) {
            todo.push_back(next);
        }
    }
    return loop;
}

struct Fixture {
    o::Program child;
    o::CountedLoopRegion bank;
};
Fixture fixture(bool guarded, bool reentered, unsigned banks)
{
    auto p = base(2 * banks + 4, 8);
    p.operations = {op(o::Pipe::MTE2, {}), op(o::Pipe::MTE2, {}),
                    op(o::Pipe::MTE1, {}), op(o::Pipe::MTE1, {}), op(o::Pipe::M, {})};
    for (unsigned side = 0; side < 2; ++side) {
        for (unsigned bank = 0; bank < banks; ++bank) {
            p.operations[side].accesses.push_back({side * banks + bank, false, true});
            p.operations[side + 2].accesses.push_back({side * banks + bank, true, false});
        }
        for (unsigned bank = 0; bank < 2; ++bank) {
            p.operations[side + 2].accesses.push_back({2 * banks + side * 2 + bank, false, true});
            p.operations[4].accesses.push_back({2 * banks + side * 2 + bank, true, false});
        }
    }
    for (std::size_t i = 0; i < p.operations.size(); ++i) {
        p.operations[i].original = i;
    }
    auto episode = seq({leaf(0), leaf(1),
        {o::Region::For, {seq({leaf(2), leaf(3), leaf(4)})}}});
    if (guarded) {
        episode = {o::Region::Choice, {episode, seq({})}};
    }
    p.body = {o::Region::For, {episode}};
    if (reentered) {
        p.body = {o::Region::For, {p.body}};
    }
    auto imported = o::addStructuredBoundaryCuts(p);
    require(imported.success, imported.reason);
    std::vector<std::size_t> owners;
    for (const auto& scope : imported.program.observed->scopes) {
        if (scope.kind == o::AnalysisContext::ForBody) {
            owners.push_back(scope.ownerSite);
        }
    }
    require(owners.size() == (reentered ? 3 : 2), "nested original regions");
    auto child = region(imported.program, owners.back());
    for (unsigned operation = 2; operation < 5; ++operation) {
        o::CountedLoopRegion::PeriodicEffects binding{operation, {}};
        for (unsigned residue = 0; residue < 2; ++residue) {
            std::vector<o::Access> effects;
            for (const auto& access : p.operations[operation].accesses) {
                if (access.cell < 2 * banks || access.cell % 2 == residue) {
                    effects.push_back(access);
                }
            }
            binding.residues.push_back(std::move(effects));
        }
        child.effects.push_back(std::move(binding));
    }
    auto refined = o::refineCountedLoop(imported.program, child);
    require(refined.success, refined.reason);
    auto bank = region(refined.program, owners[owners.size() - 2]);
    bank.period = banks;
    for (unsigned operation = 0; operation < 4; ++operation) {
        o::CountedLoopRegion::PeriodicEffects binding{operation, {}};
        for (unsigned residue = 0; residue < banks; ++residue) {
            const auto physical = banks == 3 ? (residue * 2) % banks : residue;
            std::vector<o::Access> effects;
            for (const auto& access : p.operations[operation].accesses) {
                if (access.cell >= 2 * banks || access.cell % banks == physical) {
                    effects.push_back(access);
                }
            }
            binding.residues.push_back(std::move(effects));
        }
        bank.effects.push_back(std::move(binding));
    }
    return {std::move(refined.program), std::move(bank)};
}

void concrete(const o::Program& p, const o::Commands& commands, const std::vector<std::size_t>& path)
{
    auto truth = p;
    truth.observed.reset();
    truth.body = {};
    truth.operations.clear();
    o::Commands words;
    std::vector<o::Command> pending;
    std::vector<unsigned> visits, previousComputes, currentComputes;
    std::vector<std::pair<unsigned, unsigned>> forbidden;
    unsigned previousBank = std::numeric_limits<unsigned>::max();
    for (auto at : path) {
        pending.insert(pending.end(), commands[at].begin(), commands[at].end());
        const auto physical = p.observed->sites[at].operation;
        if (physical == o::NoControlId) {
            continue;
        }
        const auto& operation = p.operations[physical];
        const auto visit = unsigned(visits.size());
        if (operation.original == 0) {
            const auto& effect = operation.accesses.front();
            previousComputes = effect.cell == previousBank ? std::vector<unsigned>{} : currentComputes;
            currentComputes.clear();
            previousBank = effect.cell;
        }
        if (operation.original >= 2) {
            require(operation.accesses.size() == 2, "enclosing qualification broadened child effects");
        }
        if (operation.pipe == o::Pipe::MTE2) {
            for (auto source : previousComputes) {
                forbidden.emplace_back(source, visit);
            }
        }
        if (operation.pipe == o::Pipe::M) {
            currentComputes.push_back(visit);
        }
        words.push_back(std::move(pending));
        pending.clear();
        visits.push_back(visit);
        truth.operations.push_back(operation);
    }
    words.push_back(std::move(pending));
    require(bool(oahs_oracle::graph(truth, words, visits, forbidden)),
        "bank interface lost memory/event correctness or gates next-bank DMA on current compute");
}

void run(bool guarded, bool reentered, unsigned banks = 2)
{
    const auto f = fixture(guarded, reentered, banks);
    const auto imported = o::refineBankOccurrences(f.child, f.bank);
    require(imported.success, imported.reason);
    const auto& p = imported.program;
    require(p.observed->sites.size() == f.child.observed->sites.size() + (banks - 1) * (f.bank.bodySites.size() + 1),
        "bank interface introduced enclosing first/tail modes");
    unsigned pairedChildren = 0;
    for (const auto& child : p.observed->loops) {
        if (child.occurrences.empty()) continue;
        ++pairedChildren;
        require(child.occurrences.size() == banks, "lost an independent child occurrence");
        for (const auto& occurrence : o::loopEntryOccurrences(child)) {
            require(occurrence.entry != o::NoControlId && occurrence.exit != o::NoControlId,
                    "child occurrence lost its paired boundaries");
            require(!occurrence.sites.empty(), "child occurrence lost its original member positions");
            for (auto site : occurrence.sites) {
                require(std::find(child.sites.begin(), child.sites.end(), site) != child.sites.end(),
                        "child occurrence escaped its original owner");
            }
        }
    }
    require(pairedChildren != 0, "enclosing refinement lost child correspondence");
    auto broken = p;
    for (auto& child : broken.observed->loops) {
        if (child.occurrences.empty()) continue;
        child.occurrences.front().entry = broken.observed->exit;
        break;
    }
    require(!o::validateProgram(broken).success, "accepted a child entry outside its original owner");
    const auto plan = accepted(p);
    require(!plan.channels.empty(), "bank-qualified readiness/release channels absent");
    unsigned bankReady = 0, bankRelease = 0;
    for (const auto& channel : plan.channels) {
        if (channel.owner != f.bank.owner) continue;
        if (channel.source == o::Pipe::MTE2 && channel.observer == o::Pipe::MTE1) {
            require(channel.cells.size() == 1, "distinct bank readiness prefixes were combined");
            ++bankReady;
        } else if (channel.source == o::Pipe::MTE1 && channel.observer == o::Pipe::MTE2) {
            require(channel.cells.size() == 1,
                    "different physical reader releases were combined by endpoint motion");
            ++bankRelease;
        }
    }
    require(bankReady == 2 * banks && bankRelease == 2 * banks,
            "unexpected bank readiness/release interface population");
    std::vector<std::size_t> path;
    std::map<std::size_t, unsigned> backedges;
    unsigned traces = 0;
    std::function<void(std::size_t, unsigned)> walk = [&](std::size_t at, unsigned payloads) {
        payloads += p.observed->sites[at].operation != o::NoControlId;
        if (payloads > (banks == 3 ? 22 : 18)) {
            return;
        }
        require(path.size() < 4096, "fixture has a payload-free cycle");
        path.push_back(at);
        if (at == p.observed->exit) {
            concrete(p, plan.commands, path);
            ++traces;
        } else {
            const auto& site = p.observed->sites[at];
            for (std::size_t edge = 0; edge < site.successors.size(); ++edge) {
                const auto owner = site.backedgeOwners.empty() ? o::NoControlId : site.backedgeOwners[edge];
                if (owner != o::NoControlId && backedges[owner] == (banks == 3 ? 4u : 3u)) {
                    continue;
                }
                if (owner != o::NoControlId) {
                    ++backedges[owner];
                }
                walk(site.successors[edge], payloads);
                if (owner != o::NoControlId) {
                    --backedges[owner];
                }
            }
        }
        path.pop_back();
    };
    walk(p.observed->entry, 0);
    require(traces > 0, "no occurrence traces");
    auto malformed = f.bank;
    malformed.effects.front().residues.pop_back();
    require(!o::refineBankOccurrences(f.child, malformed).success, "malformed bank binding accepted");
    malformed = f.bank;
    malformed.bodySites.pop_back();
    require(!o::refineBankOccurrences(f.child, malformed).success, "escaping bank body accepted");
    require(!o::refineBankOccurrences(p, f.bank).success, "duplicate bank dimension accepted");
    std::cout << "banks=" << banks << " guarded=" << guarded << " reentered=" << reentered
              << " bank occurrence traces=" << traces << " channels=" << plan.channels.size()
              << " sites=" << p.observed->sites.size() << '\n';
}
// Two distinct reader owners use one physical generation. Reloads and outside
// readers are represented as accesses, rather than another child-loop pattern.
void composedReaders(unsigned variant)
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
    auto p = base(4, 8);
    p.operations = {op(P, {{0, false, true, true}, {1, false, true, true}}),
        op(Q, {{0, true, false}, {1, true, false}, {2, false, true, true}}),
        op(R, {{2, true, false}}),
        op(Q, {{0, true, false}, {1, true, false}, {2, false, true, true}}),
        op(R, {{2, true, false}}),
        op(Q, {{0, true, false}, {1, true, false}}),
        op(P, {{0, false, true, true}, {1, false, true, true}}),
        op(P, {{3, false, true, true}}), op(P, {{3, false, true, true}})};
    for (std::size_t i = 0; i < p.operations.size(); ++i) {
        p.operations[i].original = i;
        for (auto& access : p.operations[i].accesses) {
            access.definiteWrite = false;
        }
    }
    o::Region first{o::Region::For, {seq({leaf(1), leaf(2)})}, 0, true};
    o::Region second{o::Region::For, {seq({leaf(3), leaf(4)})}, 0, true};
    auto body = seq({leaf(0)});
    if (variant == 3) {
        body.children.push_back(leaf(8));
    }
    body.children.push_back(first);
    if (variant == 1) {
        body.children.push_back(leaf(6));
    }
    body.children.push_back(second);
    if (variant == 2) {
        body.children.push_back(leaf(5));
    }
    p.body = {o::Region::For, {body}, 0, true};
    if (variant == 3) {
        p.body = seq({leaf(7), p.body});
    }
    if (variant == 4) {
        p.body = seq({p.body, leaf(7)});
    }
    const auto originalOperations = p.operations;
    p.operations.clear();
    std::map<unsigned, unsigned> operationIds;
    std::function<void(o::Region&)> remap = [&](o::Region& node) {
        if (node.kind == o::Region::Operation) {
            const auto old = unsigned(node.operation);
            operationIds.emplace(old, unsigned(p.operations.size()));
            node.operation = operationIds.at(old);
            p.operations.push_back(originalOperations[old]);
            p.operations.back().original = node.operation;
        }
        for (auto& child : node.children) {
            remap(child);
        }
    };
    remap(p.body);
    auto imported = o::addStructuredBoundaryCuts(p);
    require(imported.success, imported.reason);
    auto program = imported.program;
    std::vector<o::Cut> owners;
    for (const auto& scope : program.observed->scopes) {
        if (scope.kind == o::AnalysisContext::ForBody) {
            owners.push_back(scope.ownerSite);
        }
    }
    require(owners.size() == 3, "two child reader owners expected");
    for (unsigned child = 1; child < 3; ++child) {
        auto loop = region(program, owners[child]);
        loop.period = 1;
        auto refined = o::refineCountedLoop(program, loop);
        require(refined.success, refined.reason);
        program = std::move(refined.program);
    }
    auto bank = region(program, owners.front());
    for (unsigned operation : {0u, 1u, 3u, 5u, 6u}) {
        if ((operation == 5 && variant != 2) || (operation == 6 && variant != 1)) {
            continue;
        }
        o::CountedLoopRegion::PeriodicEffects binding{operationIds.at(operation), {}};
        for (unsigned residue = 0; residue < 2; ++residue) {
            std::vector<o::Access> effects;
            for (const auto& access : originalOperations[operation].accesses) {
                if (access.cell >= 2 || access.cell == residue) {
                    auto physical = access;
                    physical.definiteWrite = false;
                    effects.push_back(physical);
                }
            }
            binding.residues.push_back(std::move(effects));
        }
        bank.effects.push_back(std::move(binding));
    }
    auto refined = o::refineBankOccurrences(program, bank);
    require(refined.success, refined.reason);
    if (variant == 5) {
        o::selected::Control control(refined.program);
        o::StorageFrontierAnalysis storage(refined.program);
        o::selected::RequirementFrontiers facts(refined.program, control, storage);
        o::Cut firstReader = o::NoAnalysisId, continuingReader = o::NoAnalysisId;
        for (o::Cut site = 0; site < control.graph.sites.size(); ++site) {
            if (facts.use(site, 0).roles != 1 ||
                refined.program.operations[control.graph.operations[site]].original != operationIds.at(1)) {
                continue;
            }
            const auto& role = facts.readerParticipation(site, 0);
            if (role.proved() && role.first) {
                firstReader = site;
            } else if (role.proved()) {
                continuingReader = site;
            }
        }
        require(firstReader != o::NoAnalysisId && continuingReader != o::NoAnalysisId,
                "shared-word fixture lacks distinct participating roles");
        auto& graph = *refined.program.observed;
        graph.sites[continuingReader].observation = graph.sites[firstReader].observation;
        require(o::validateProgram(refined.program).success, "shared-word fixture is not valid input");
        o::selected::Control sharedControl(refined.program);
        o::StorageFrontierAnalysis sharedStorage(refined.program);
        o::selected::RequirementFrontiers sharedFacts(refined.program, sharedControl, sharedStorage);
        const auto requests = o::selected::qualifyCyclicFrontiers(refined.program, sharedControl, sharedFacts);
        require(std::none_of(requests.begin(), requests.end(), [&](const auto& request) {
                    return request.owner == bank.owner && request.source == P && request.observer == Q &&
                        std::find(request.cells.begin(), request.cells.end(), 0) != request.cells.end();
                }), "conflicting shared-word first/continuing roles produced one readiness acquisition");
        return;
    }
    const auto plan = o::constructSelectedPlan(refined.program);
    if (variant == 0) {
        require(plan.work.unsummarizedBackedges != 0 && plan.work.contextualReplays != 0,
                "cyclic occurrence interfaces lost original-edge replay");
    }
    if (variant == 3) {
        require(plan.declinedRecurring && plan.declinedRecurring->reason.find("producer repair") != std::string::npos,
                "unsupported Y repair crossed the newly supported X overwrite");
    } else {
        require(plan.success && o::checkCausalFrontier(refined.program, plan.commands).accepted,
                "composed reader construction was not independently accepted");
        require(!plan.declinedRecurring, "composed reader protocol was declined");
        const auto ready = std::count_if(plan.channels.begin(), plan.channels.end(), [&](const auto& channel) {
            return channel.owner == bank.owner && channel.source == P && channel.observer == Q;
        });
        require(ready == 2, "shared generation across reader children was not constructed per bank");
        const auto releases = std::count_if(plan.channels.begin(), plan.channels.end(), [&](const auto& channel) {
            return channel.owner == bank.owner && channel.source == Q && channel.observer == P;
        });
        require(releases == 2, "composed generation lost a bank release");
        for (const auto& channel : plan.channels) {
            if (channel.owner != bank.owner || channel.source != Q || channel.observer != P) {
                continue;
            }
            auto broken = plan.commands;
            for (auto& word : broken) {
                word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
                    return command.kind == o::Command::Acquire && command.source == Q &&
                           command.observer == P && command.key == channel.key;
                }), word.end());
            }
            require(!o::checkCausalFrontier(refined.program, broken).accepted,
                    "removed generation return was accepted");
        }
    }
}

} // namespace
int main()
{
    for (unsigned variant = 0; variant < 6; ++variant) {
        composedReaders(variant);
    }
    run(false, false);
    run(true, false);
    run(false, true);
    run(true, true);
    run(false, true, 3);
    return 0;
}
