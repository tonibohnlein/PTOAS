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
    const auto plan = accepted(p);
    require(plan.channels.size() == 3 * banks + 4, "bank-qualified readiness/release channels absent");
    unsigned bankReady = 0, bankRelease = 0;
    for (const auto& channel : plan.channels) {
        if (channel.owner != f.bank.owner) continue;
        if (channel.source == o::Pipe::MTE2 && channel.observer == o::Pipe::MTE1) {
            require(channel.cells.size() == 1, "distinct bank readiness prefixes were combined");
            ++bankReady;
        } else if (channel.source == o::Pipe::MTE1 && channel.observer == o::Pipe::MTE2) {
            require(channel.cells.size() == 2 && channel.cells[0] % banks == channel.cells[1] % banks,
                    "release did not join the same physical bank episode");
            ++bankRelease;
        }
    }
    require(bankReady == 2 * banks && bankRelease == banks,
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
} // namespace
int main()
{
    run(false, false);
    run(true, false);
    run(false, true);
    run(true, true);
    run(false, true, 3);
    return 0;
}
