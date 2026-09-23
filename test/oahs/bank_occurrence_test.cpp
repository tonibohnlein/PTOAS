// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
#include "../../lib/PTO/Transforms/OAHS/OriginalReadQueries.h"
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

void accumulatorOverlay(bool reversed, bool outside = false)
{
    auto f = fixture(false, false, 2);
    auto& p = f.child;
    p.target = mlir::pto::a3SyncProfile(mlir::pto::SyncCore::Cube);
    p.nativeAccumulatorClasses = 2;
    o::Cell accumulator;
    accumulator.domain = o::Cell::Domain::Accumulator;
    accumulator.storage = o::Cell::Storage::CanonicalInterval;
    accumulator.coordinateSpace = "physical-local-acc";
    accumulator.ranges = {{0, 131072}};
    const auto cell = unsigned(p.cells.size());
    p.cells.push_back(accumulator);
    std::size_t outsidePhase = o::NoControlId;
    for (std::size_t i = 0; i < p.operations.size(); ++i) {
        auto& operation = p.operations[i];
        if (operation.pipe != o::Pipe::M) {
            continue;
        }
        operation.nativeMmadAccumulate = true;
        operation.accesses.push_back({cell, true, true, false, 0});
        auto other = operation.accesses;
        other.back().nativeAccumulatorClass = 1;
        // Same bytes and roles, different proved access contracts. An enclosing
        // overlay must retain this distinction through a refined child.
        f.bank.effects.push_back({i, reversed ? std::vector<std::vector<o::Access>>{other, operation.accesses} :
            std::vector<std::vector<o::Access>>{operation.accesses, other}});
        if (outside) {
            // The outside occurrence retains the conservative union; neither
            // specialized residue equals that original physical phase.
            operation.accesses.push_back({cell, true, true, false, 1});
            outsidePhase = i;
        }
    }
    if (outside) {
        auto& q = *p.observed;
        auto node = q.sites[q.entry];
        node.operation = outsidePhase;
        node.successors = {q.entry};
        node.backedgeOwners.clear();
        node.observation = q.observations.size();
        q.observations.push_back({node.observation, {}, true});
        q.entry = q.sites.size();
        q.sites.push_back(std::move(node));
    }
    const auto refined = o::refineBankOccurrences(p, f.bank);
    if (outside && refined.success) {
        const auto& accesses = refined.program.operations[outsidePhase].accesses;
        require(std::count_if(accesses.begin(), accesses.end(), [&](const auto& access) {
                    return access.cell == cell;
                }) == 2, "enclosing refinement mutated an outside occurrence's physical effects");
    }
    require(refined.success, refined.reason);
    bool seen[2] = {false, false};
    const auto& q = *refined.program.observed;
    for (const auto& site : q.sites) {
        if (site.operation == o::NoControlId || site.observation == o::NoControlId ||
            refined.program.operations[site.operation].pipe != o::Pipe::M) {
            continue;
        }
        for (const auto& atom : q.observations[site.observation].atoms) {
            if (atom.owner != f.bank.owner || atom.kind != o::ObservationAtom::LoopResidue) {
                continue;
            }
            require(atom.value < 2, "unexpected enclosing contract residue");
            seen[atom.value] = true;
            const auto& accesses = refined.program.operations[site.operation].accesses;
            const auto access = std::find_if(accesses.begin(), accesses.end(), [&](const auto& a) {
                return a.cell == cell;
            });
            const auto expected = reversed ? 1 - atom.value : atom.value;
            require(access != accesses.end() && access->nativeAccumulatorClass == expected,
                    "enclosing refinement replaced an exact child ACC access contract");
        }
    }
    require(seen[0] && seen[1], "both enclosing access contracts must remain represented");
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
        op(P, {{3, false, true, true}}), op(P, {{3, false, true, true}}),
        op(P, {{3, false, true, true}})};
    for (std::size_t i = 0; i < p.operations.size(); ++i) {
        p.operations[i].original = i;
        for (auto& access : p.operations[i].accesses) {
            access.definiteWrite = false;
        }
    }
    o::Region first{o::Region::For, {seq({leaf(1), leaf(2)})}, 0, true};
    o::Region second{o::Region::For, {seq({leaf(3), leaf(4)})}, 0, true};
    auto body = seq({leaf(0)});
    if (variant == 3 || variant == 6) {
        body.children.push_back(leaf(8));
        if (variant == 6) { body.children.push_back(leaf(9)); }
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
    if (variant == 3 || variant == 6) {
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
            const auto& role = facts.readerBoundaries(site, 0);
            const bool first = role.proved() && role.first.hit();
            if (first) {
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
        require(std::none_of(requests.roles.begin(), requests.roles.end(), [&](const auto& request) {
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
        require(plan.success && o::checkCausalFrontier(refined.program, plan.commands).accepted,
                "ordinary producer support did not reconstruct");
        require(plan.work.recurringActivations != 0 && !plan.declinedRecurring,
                "unrelated producer repair still disabled the x lifetime");
        bool hasProducerFence = false;
        for (const auto& word : plan.commands) {
            hasProducerFence |= std::any_of(word.begin(), word.end(), [&](const auto& command) {
                return command.kind == o::Command::Barrier && command.source == P;
            });
        }
        require(hasProducerFence, "typed ordinary producer support omitted its fence");
        const auto ready = std::count_if(plan.channels.begin(), plan.channels.end(), [&](const auto& channel) {
            return channel.owner == bank.owner && channel.source == P && channel.observer == Q;
        });
        const auto release = std::count_if(plan.channels.begin(), plan.channels.end(), [&](const auto& channel) {
            return channel.owner == bank.owner && channel.source == Q && channel.observer == P;
        });
        require(ready == 2 && release == 2, "unrelated z repair erased the two-bank x protocol");
        o::selected::Control selectedControl(refined.program);
        bool fencedSeed = false;
        for (o::Cut site = 0; site < selectedControl.graph.operations.size(); ++site) {
            const auto opId = selectedControl.graph.operations[site];
            const bool selectedSeed = opId == operationIds.at(0) && selectedControl.reachable[site];
            if (!selectedSeed) { continue; }
            const auto word = selectedControl.canonicalCut[site];
            const bool populated = word < plan.commands.size() && !plan.commands[word].empty();
            if (!populated) { continue; }
            const auto& last = plan.commands[word].back();
            fencedSeed |= last.kind == o::Command::Barrier && last.source == P;
        }
        require(fencedSeed, "ordinary z support fence was not at the x producer seed");
        auto missing = plan.commands;
        for (auto& word : missing) {
            word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
                return command.kind == o::Command::Barrier && command.source == P;
            }), word.end());
        }
        require(!o::checkCausalFrontier(refined.program, missing).accepted,
                "removing ordinary producer support kept the z protocol valid");
    } else if (variant == 6) {
        require(plan.success && o::checkCausalFrontier(refined.program, plan.commands).accepted,
                "intervening z access lost conservative service");
    } else {
        require(plan.success && o::checkCausalFrontier(refined.program, plan.commands).accepted,
                "composed reader construction was not independently accepted");
        require(!plan.declinedRecurring, "composed reader protocol was declined");
        require(plan.work.recurringActivations != 0 &&
                    std::any_of(plan.activations.begin(), plan.activations.end(), [](const auto& activation) {
                        return activation.families.size() >= 2 && activation.after.size() < activation.before.size();
                    }), "normal constructor did not close required multi-bank support from an actual residual");
        const auto ready = std::count_if(plan.channels.begin(), plan.channels.end(), [&](const auto& channel) {
            return channel.owner == bank.owner && channel.source == P && channel.observer == Q;
        });
        if (ready != 2) {
            for (const auto& refusal : plan.recurringRefusals) {
                std::cerr << "variant=" << variant << " family=" << refusal.family << " at=" << refusal.deadline
                          << " refusal=" << refusal.reason << '\n';
            }
        }
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

void participationDemandScaling()
{
    uint64_t previous = 0;
    for (unsigned count : {16u, 32u, 64u}) {
        auto p = base(1);
        p.operations = {op(o::Pipe::MTE2, {{0, false, true}}), op(o::Pipe::V, {{0, true, false}})};
        auto body = seq({leaf(0), leaf(1)});
        for (unsigned i = 0; i < count; ++i) {
            const auto operation = p.operations.size();
            p.operations.push_back(op(o::Pipe::V, {{0, true, false}}));
            o::Region child{o::Region::For, {seq({leaf(operation)})}};
            child.zeroTripPossible = true; body.children.push_back(child);
        }
        p.body = {o::Region::For, {body}};
        const auto imported = o::addStructuredBoundaryCuts(p);
        require(imported.success, imported.reason);
        o::storage_detail::OriginalReadQueries reads(imported.program);
        const auto demands = reads.participationDemands();
        require(demands.size() == count, "optional sibling demand disappeared");
        const auto work = reads.compositionParts;
        require(!previous || work <= previous * 2 + 32, "linear frontier DAG materialized quadratic subject sets");
        previous = work;
    }
}

void participationBatchScaling()
{
    uint64_t previous = 0;
    for (unsigned count : {4u, 8u, 16u}) {
        auto p = base(1);
        o::ObservedControl q; q.qualification = "test original independently sampled Boolean intervals";
        q.scopes.push_back({0, o::NoControlId, o::NoControlId});
        q.sites.resize(5 * count + 1); q.exit = 5 * count;
        std::vector<o::ParticipationRegion> requests;
        for (unsigned site = 0; site < q.sites.size(); ++site) {
            q.sites[site].observation = q.observations.size();
            q.observations.push_back({site, {}, true});
            if (site != q.exit) { q.sites[site].successors = {site + 1}; }
        }
        for (unsigned i = 0; i < count; ++i) {
            const auto start = 5 * i, decision = start + 2, stop = start + 4;
            for (auto site : {start + 1, start + 3}) {
                q.sites[site].operation = p.operations.size();
                p.operations.push_back(op(o::Pipe::V, {{0, true, false}}));
            }
            q.sites[decision].successors = {start + 3, stop};
            requests.push_back({start, stop, decision,
                {o::ObservationAtom::LoopNonEmpty, decision, 0, 1}, {stop}, {start + 3}, true});
        }
        p.observed = q;
        auto overlap = requests.front(); overlap.predicate.owner += q.sites.size();
        requests.push_back(overlap);
        const auto batch = o::refineParticipations(p, requests);
        require(batch.success && batch.accepted.size() == count,
                "disjoint demands lost to unrelated overlapping demand");
        require(batch.copies == 1 && batch.validations == 2 && batch.refreshes == 1,
                "participation repeats whole-program preparation per interval");
        require(!batch.refusals.back().empty(), "unsupported overlap was silently accepted");
        require(!previous || batch.work <= previous * 2 + 4, "disjoint predicate refinement work is not linear");
        previous = batch.work;
        require(batch.program.observed->sites.size() == 13 * count + 1,
                "independent predicates formed a global product");
        auto cyclic = p;
        cyclic.observed->sites[1].successors.push_back(0);
        const auto prefixCycle = o::refineParticipations(cyclic, {requests.front()});
        require(prefixCycle.success && prefixCycle.accepted.empty() &&
                prefixCycle.refusals.front().find("entry repeats") != std::string::npos,
                "prefix cycle skipped the original unguarded entry word");
        auto unrelated = p;
        const auto other = 5 * (count - 1);
        o::ObservedLoop legacy{other, other, other + 4, {other + 1, other + 2, other + 3}, other + 1, false};
        legacy.entries = {other}; legacy.exits = {other + 4};
        unrelated.observed->loops.push_back(legacy);
        const auto local = o::refineParticipations(unrelated, {requests.front()});
        require(local.success && local.accepted.size() == 1 &&
                local.program.observed->loops.back().occurrences.empty() &&
                local.program.observed->loops.back().sites == legacy.sites,
                "unrelated unpaired loop blocked or changed a local participation interval");
        auto alias = p;
        alias.observed->sites[q.exit].observation = q.sites[2].observation;
        const auto outside = o::refineParticipations(alias, {requests.front()});
        require(outside.success && outside.accepted.empty() &&
                outside.refusals.front().find("command words") != std::string::npos,
                "outside word: " + outside.reason + (outside.refusals.empty() ? "" : outside.refusals.front()));
        alias = p;
        alias.observed->sites[2].observation = q.sites[0].observation;
        const auto shared = o::refineParticipations(alias, {requests.front()});
        require(shared.success && shared.accepted.empty() &&
                shared.refusals.front().find("source gap") != std::string::npos,
                "shared unguarded entry word was accepted");
    }
}

void optionalReaderParticipation()
{
    auto p = base(1, 2);
    p.operations = {op(o::Pipe::MTE2, {{0, false, true}}),
                    op(o::Pipe::V, {{0, true, false}}), op(o::Pipe::V, {{0, true, false}})};
    for (unsigned i = 0; i < p.operations.size(); ++i) { p.operations[i].original = i; }
    o::Region a{o::Region::For, {seq({leaf(1)})}};
    o::Region b{o::Region::For, {seq({leaf(2)})}}; b.zeroTripPossible = true;
    p.body = {o::Region::For, {seq({leaf(0), a, b})}};
    auto input = o::addStructuredBoundaryCuts(p);
    require(input.success, input.reason);
    p = std::move(input.program);
    std::vector<o::Cut> owners;
    for (const auto& scope : p.observed->scopes) {
        if (scope.kind == o::AnalysisContext::ForBody) { owners.push_back(scope.ownerSite); }
    }
    for (unsigned child = 1; child <= 2; ++child) {
        auto model = region(p, owners[child]); model.period = 1; model.atLeastOnce = child == 1;
        auto refined = o::refineCountedLoop(p, model);
        require(refined.success, refined.reason); p = std::move(refined.program);
    }
    auto parent = region(p, owners.front());
    p.observed->loops.push_back({parent.owner, parent.owner, parent.continuation,
                                parent.bodySites, parent.bodyEntry, true});
    p.observed->loops.back().sites.push_back(parent.header);
    o::selected::Control control(p);
    o::StorageFrontierAnalysis storage(p);
    o::selected::RequirementFrontiers frontiers(p, control, storage);
    o::storage_detail::OriginalReadQueries reads(p);
    const auto demands = reads.participationDemands();
    require(demands.size() == 1 && demands.front().predicate.owner == owners[2], "D3 sibling participation demand");
    o::ParticipationRegion model;
    model.entry = owners[1]; model.exit = parent.header; model.decision = owners[2];
    model.predicate = demands.front().predicate; model.available = true;
    const auto choices = p.observed->sites[model.decision].successors;
    model.whenFalse = {choices.front()}; model.whenTrue.assign(choices.begin() + 1, choices.end());
    auto unavailable = model; unavailable.available = false;
    require(!o::refineParticipation(p, unavailable).success, "unavailable predicate accepted");
    auto malformed = model; malformed.whenTrue.push_back(choices.front());
    require(!o::refineParticipation(p, malformed).success, "overlapping alternatives accepted");
    auto refined = o::refineParticipation(p, model);
    require(refined.success, refined.reason);
    const auto plan = accepted(refined.program);
    require(!plan.declinedRecurring && !plan.activations.empty(), "optional reader family not activated");
    require(plan.channels.size() == 2, "optional child created a private protocol");
    auto schema = o::exportObservedSchema(refined.program, plan.commands);
    require(schema.complete && o::reconstructObservedSchema(refined.program, schema).success,
            "optional participation reconstruction failed");
    bool aRelease = false, bRelease = false;
    for (const auto& channel : plan.channels) {
        if (channel.source != o::Pipe::V) { continue; }
        for (auto cut : channel.publications) {
            const auto observation = refined.program.observed->sites[cut].observation;
            if (observation == o::NoControlId) { continue; }
            for (const auto& atom : refined.program.observed->observations[observation].atoms) {
                if (atom.kind == o::ObservationAtom::LoopNonEmpty && atom.owner == owners[2]) {
                    aRelease |= atom.value == 0; bRelease |= atom.value == 1;
                }
            }
        }
    }
    require(aRelease && bRelease, "final release lost original child participation");
    const auto baseline = accepted(p);
    const auto& q = *refined.program.observed;
    std::vector<o::Cut> oldWords(q.observations.size(), o::NoControlId);
    for (std::size_t word = 0; word < q.observations.size(); ++word) {
        auto original = q.observations[word];
        original.atoms.erase(std::remove_if(original.atoms.begin(), original.atoms.end(), [&](const auto& atom) {
            return atom.kind == o::ObservationAtom::LoopNonEmpty && atom.owner == owners[2];
        }), original.atoms.end());
        for (o::Cut site = 0; site < p.observed->sites.size(); ++site) {
            const auto prior = p.observed->sites[site].observation;
            if (prior == o::NoControlId) { continue; }
            const auto& before = p.observed->observations[prior];
            const bool sameAtoms = std::equal(before.atoms.begin(), before.atoms.end(),
                original.atoms.begin(), original.atoms.end(), [](const auto& a, const auto& b) {
                    return std::tie(a.kind, a.owner, a.parameter, a.value) ==
                           std::tie(b.kind, b.owner, b.parameter, b.value);
                });
            if (before.anchor == original.anchor && sameAtoms) {
                oldWords[word] = site; break;
            }
        }
    }
    unsigned traces = 0, removed = 0;
    bool skipped = false, present = false, changedOnReentry = false;
    std::vector<o::Cut> path;
    std::map<o::Cut, unsigned> backedges;
    auto checkTrace = [&]() {
        auto truth = refined.program; truth.observed.reset(); truth.operations.clear();
        std::array<o::Commands, 2> words;
        std::array<std::vector<o::Command>, 2> pending;
        std::vector<unsigned> visits;
        for (auto site : path) {
            const auto& node = q.sites[site];
            for (const auto& command : plan.commands[site]) { pending[1].push_back(command); }
            if (node.observation != o::NoControlId) {
                require(oldWords[node.observation] != o::NoControlId, "participation lost original word projection");
                for (const auto& command : baseline.commands[oldWords[node.observation]]) {
                    pending[0].push_back(command);
                }
            }
            if (node.operation == o::NoControlId) { continue; }
            visits.push_back(truth.operations.size());
            truth.operations.push_back(refined.program.operations[node.operation]);
            for (unsigned version = 0; version < 2; ++version) {
                words[version].push_back(std::move(pending[version])); pending[version].clear();
            }
        }
        if (visits.empty()) { return; }
        std::vector<unsigned> optionalReads;
        for (const auto& operation : truth.operations) {
            if (operation.original == 0) { optionalReads.push_back(0); }
            if (operation.original == 2) { ++optionalReads.back(); }
        }
        for (std::size_t episode = 0; episode < optionalReads.size(); ++episode) {
            skipped |= optionalReads[episode] == 0;
            present |= optionalReads[episode] != 0;
            if (episode) {
                changedOnReentry |= bool(optionalReads[episode]) != bool(optionalReads[episode - 1]);
            }
        }
        std::array<std::set<std::pair<unsigned, unsigned>>, 2> relations;
        for (unsigned version = 0; version < 2; ++version) {
            words[version].push_back(std::move(pending[version]));
            require(bool(oahs_oracle::graph(truth, words[version], visits, {}, nullptr, nullptr, &relations[version])),
                    "optional-reader concrete event or memory failure");
        }
        require(std::includes(relations[0].begin(), relations[0].end(), relations[1].begin(), relations[1].end()),
                "participation added payload order on a complete nonempty trace");
        removed += relations[0].size() - relations[1].size();
        ++traces;
    };
    std::function<void(o::Cut, unsigned)> walk = [&](o::Cut at, unsigned payloads) {
        payloads += q.sites[at].operation != o::NoControlId;
        if (payloads > 10) { return; }
        require(path.size() < 512, "participation introduced a payload-free cycle");
        path.push_back(at);
        if (at == q.exit) { checkTrace(); }
        else {
            const auto& node = q.sites[at];
            for (std::size_t edge = 0; edge < node.successors.size(); ++edge) {
                const auto owner = node.backedgeOwners.empty() ? o::NoControlId : node.backedgeOwners[edge];
                if (owner != o::NoControlId && backedges[owner] == 2) { continue; }
                if (owner != o::NoControlId) { ++backedges[owner]; }
                walk(node.successors[edge], payloads);
                if (owner != o::NoControlId) { --backedges[owner]; }
            }
        }
        path.pop_back();
    };
    walk(q.entry, 0);
    require(traces > 0 && skipped && present && changedOnReentry,
            "optional-reader comparison missed empty/nonempty/reentered participation");
    auto missing = plan.commands;
    for (auto& word : missing) {
        word.erase(std::remove_if(word.begin(), word.end(), [](const auto& command) {
            return command.kind == o::Command::Publish && command.source == o::Pipe::V &&
                   command.observer == o::Pipe::MTE2;
        }), word.end());
    }
    require(!o::checkCausalFrontier(refined.program, missing).accepted, "missing final release was accepted");
    std::cout << "optional reader traces=" << traces << " removed relations=" << removed << '\n';
}

} // namespace
int main()
{
    participationDemandScaling();
    participationBatchScaling();
    optionalReaderParticipation();
    accumulatorOverlay(false);
    accumulatorOverlay(true);
    accumulatorOverlay(false, true);
    for (unsigned variant = 0; variant < 7; ++variant) {
        composedReaders(variant);
    }
    run(false, false);
    run(true, false);
    run(false, true);
    run(true, true);
    run(false, true, 3);
    return 0;
}
