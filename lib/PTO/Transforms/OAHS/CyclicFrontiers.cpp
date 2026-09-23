// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"
#include <algorithm>
#include <deque>
#include <limits>
#include <tuple>

namespace mlir::pto::oahs::selected {
namespace {
// Project one physical cell's access roles, without making storage succession
// imply completion. All other payload remains in the selected full-graph replay.
bool balanced(const Control&, const std::vector<Cut>&, const std::vector<Cut>&);
std::vector<RecurringRequirement> qualifyCell(
    const Program& p, const Control& c, const RequirementFrontiers& frontiers,
    const ObservedLoop& loop, unsigned cell)
{
    std::set<Id> members(loop.sites.begin(), loop.sites.end());
    const std::set<Id> entries = loop.entries.empty() ? std::set<Id>{loop.entry}
        : std::set<Id>(loop.entries.begin(), loop.entries.end());
    const std::set<Id> exits = loop.exits.empty() ? std::set<Id>{loop.exit}
        : std::set<Id>(loop.exits.begin(), loop.exits.end());
    if (loop.entry >= c.graph.sites.size() || loop.exit >= c.graph.sites.size() ||
        members.count(loop.entry) || members.count(loop.exit) || members.count(c.graph.entry)) return {};
    // Metadata is a proposal, not a trusted interface. Check all entries and
    // exits, including a root entry with no predecessor edge.
    for (Id site = 0; site < c.graph.sites.size(); ++site) {
        for (auto next : c.graph.sites[site].successors) {
            const bool unqualifiedEntry = members.count(next) && !members.count(site) &&
                !entries.count(site);
            const bool unqualifiedExit = members.count(site) && !members.count(next) &&
                !exits.count(next);
            if (unqualifiedEntry || unqualifiedExit) {
                return {};
            }
        }
    }
    if (p.cells[cell].exclusive) return {};
    std::vector<OccurrenceMode> modes(c.graph.sites.size());
    std::vector<unsigned> roles(c.graph.sites.size());
    Pipe producer = Pipe::Count, consumer = Pipe::Count;
    uint64_t residue = NoAnalysisId, period = 0;
    for (auto site : members) {
        if (site >= c.graph.sites.size()) return {};
        const auto operation = c.graph.operations[site];
        if (!c.reachable[site] || operation == NoAnalysisId) continue;
        const auto& op = p.operations[operation];
        const auto& use = frontiers.use(site, cell);
        const auto role = use.roles;
        if (!role) continue;
        const auto m = use.occurrence;
        if (role == 3 || !m.valid || m.owner != loop.owner ||
            (period && (period != m.period || residue != m.residue)) ||
            p.target.synchronous[unsigned(op.pipe)]) return {};
        period = m.period;
        residue = m.residue;
        auto& pipe = role == 2 ? producer : consumer;
        if (pipe != Pipe::Count && pipe != op.pipe) return {};
        pipe = op.pipe;
        modes[site] = m;
        roles[site] = role;
    }
    if (producer == Pipe::Count || consumer == Pipe::Count || producer == consumer) return {};
    const std::vector<Cut> boundaries(exits.begin(), exits.end());
    auto nextAccess = [&](const std::vector<Id>& starts) -> std::optional<std::set<Id>> {
        const auto& next = frontiers.nextUses(starts, cell, boundaries);
        if (!next.complete) {
            return std::nullopt;
        }
        std::set<Id> out(next.boundaries.begin(), next.boundaries.end());
        for (const auto& access : next.accesses) {
            out.insert(access.site);
        }
        return out;
    };
    const auto firstUses = nextAccess(std::vector<Id>(entries.begin(), entries.end()));
    if (!firstUses) {
        return {};
    }
    for (auto first : *firstUses) {
        const bool invalidFirstAccess = !exits.count(first) &&
            (roles[first] != 2 || modes[first].previous);
        if (invalidFirstAccess) {
            return {};
        }
    }
    // Track first/reused permission from the real local entry and all backedges.
    std::vector<unsigned> previous(c.graph.sites.size());
    std::deque<Id> queue(entries.begin(), entries.end());
    for (auto entry : entries) {
        previous[entry] = 1;
    }
    while (!queue.empty()) {
        auto site = queue.front(); queue.pop_front();
        auto bits = roles[site] == 1 ? 2u : previous[site];
        for (auto next : c.graph.sites[site].successors) {
            if (!members.count(next)) continue;
            auto joined = previous[next] | bits;
            if (joined != previous[next]) { previous[next] = joined; queue.push_back(next); }
        }
    }
    std::vector<unsigned> predecessors(c.graph.sites.size());
    for (auto site : members) if (roles[site]) {
        const auto next = nextAccess(c.graph.sites[site].successors);
        if (!next || next->empty()) {
            return {};
        }
        for (auto target : *next) {
            if (exits.count(target)) {
                if (roles[site] == 2 || modes[site].next) return {};
            } else {
                predecessors[target] |= roles[site];
                if (roles[site] == 2 && (roles[target] != 1 || !(modes[site] == modes[target]))) return {};
                if (roles[site] == 1 && roles[target] == 1 && !(modes[site] == modes[target])) return {};
                if (roles[site] == 1 && roles[target] == 2 && !modes[site].next) return {};
            }
        }
    }
    // If surrounding control can re-enter this initializer, the last ready
    // consumption must reach its publisher before the next first-use SET.
    // A final local return is justified by that specific rearm obligation.
    bool reentered = false;
    std::vector<Id> later(exits.begin(), exits.end());
    std::vector<bool> visited(c.graph.sites.size());
    while (!later.empty()) {
        const auto at = later.back(); later.pop_back();
        if (entries.count(at)) { reentered = true; break; }
        if (visited[at]) continue;
        visited[at] = true;
        const auto& next = c.graph.sites[at].successors;
        later.insert(later.end(), next.begin(), next.end());
    }
    RecurringRequirement ready, release;
    ready.qualifiedCycle = release.qualifiedCycle = true;
    release.storageRelease = true;
    ready.cell = release.cell = cell;
    ready.cells = release.cells = {cell};
    ready.source = producer;
    ready.observer = consumer;
    release.source = consumer;
    release.observer = producer;
    ready.owner = release.owner = loop.owner;
    ready.period = release.period = period;
    std::vector<Cut> finalAcquisitions;
    if (reentered && period > 1) {
        for (auto site : members) {
            if (!c.graph.legalCuts[site] || c.graph.operations[site] != NoAnalysisId) continue;
            const auto m = occurrenceMode(p, site);
            if (!m.valid || m.owner != loop.owner || m.period != period ||
                (!m.previous && m.residue < residue)) continue;
            const auto &node = p.observed->sites[site];
            if (std::find(node.backedgeOwners.begin(), node.backedgeOwners.end(), loop.owner) ==
                node.backedgeOwners.end()) continue;
            const auto &observation = p.observed->observations[node.observation];
            if (std::any_of(observation.atoms.begin(), observation.atoms.end(), [&](const auto &atom) {
                    return atom.owner == loop.owner && atom.kind == ObservationAtom::LoopHasNext &&
                           atom.parameter == 1 && atom.value == 0;
                })) finalAcquisitions.push_back(canonicalCommandCut(p, site));
        }
    }
    for (auto site : members) {
        if (!roles[site]) continue;
        const auto cut = canonicalCommandCut(p, site);
        const auto endpoint = frontiers.recurringRelease(site, cell);
        if (endpoint == NoAnalysisId) return {};
        if (roles[site] == 2) {
            if (previous[site] != (modes[site].previous ? 2u : 1u)) return {};
            ready.publications.push_back(endpoint);
            if (modes[site].previous) release.acquisitions.push_back(cut);
        } else {
            if (predecessors[site] == 2) ready.acquisitions.push_back(cut);
            else if (predecessors[site] != 1) return {}; // ambiguous first reader
            const auto next = nextAccess(c.graph.sites[site].successors);
            if (!next) {
                return {};
            }
            unsigned nextRoles = 0;
            for (auto target : *next) {
                nextRoles |= exits.count(target) ? 4u : roles[target];
            }
            if ((nextRoles & 1) && nextRoles != 1) return {}; // ambiguous last reader
            if (!(nextRoles & 1) && (modes[site].next || reentered)) {
                release.publications.push_back(endpoint);
                if (!modes[site].next && finalAcquisitions.empty()) release.acquisitions.push_back(endpoint);
            }
        }
    }
    release.acquisitions.insert(release.acquisitions.end(), finalAcquisitions.begin(), finalAcquisitions.end());
    // A child exit is not a storage reuse deadline. For a re-entered bank
    // protocol, try carrying the release token through the parent continuation.
    // Prime once at the owning invocation entry, acquire at EVERY bank write
    // (including first use after re-entry), and consume the final token at the
    // invocation exit. Zero-trip paths retain the prime unchanged. The original
    // graph must prove alternation; an intervening/conditional bank use cannot
    // be justified by a reset local iteration counter.
    if (reentered && period > 1 && c.graph.legalCuts[c.graph.entry] &&
        c.graph.legalCuts[c.graph.exit]) {
        auto open = release;
        open.publications = {canonicalCommandCut(p, c.graph.entry)};
        open.acquisitions = {canonicalCommandCut(p, c.graph.exit)};
        for (auto site : members) {
            if (roles[site] == 2)
                open.acquisitions.push_back(canonicalCommandCut(p, site));
            else if (roles[site] == 1) {
                const auto next = nextAccess(c.graph.sites[site].successors);
                if (!next) {
                    return {};
                }
                if (std::none_of(next->begin(), next->end(), [&](Id target) {
                        return !exits.count(target) && roles[target] == 1;
                    }))
                    open.publications.push_back(frontiers.recurringRelease(site, cell));
            }
        }
        if (balanced(c, open.publications, open.acquisitions)) release = std::move(open);
    }
    for (auto* request : {&ready, &release}) {
        for (auto* cuts : {&request->publications, &request->acquisitions}) {
            std::sort(cuts->begin(), cuts->end());
            cuts->erase(std::unique(cuts->begin(), cuts->end()), cuts->end());
        }
        if (request->publications.empty() || request->acquisitions.empty()) return {};
    }
    return {std::move(ready), std::move(release)};
}

// Compose a refined child reader episode with its original enclosing loop.
// This is deliberately narrower than nested residue refinement: it retains the
// child's existing finite quotient and derives one complete producer/readers
// cycle for an exact physical cell. The selected protocol is available before
// ordinary construction decides whether a same-pipe overwrite needs a fence.
std::vector<RecurringRequirement> qualifyEnclosingCell(
    const Program& p, const Control& c, const RequirementFrontiers& frontiers,
    const ObservedLoop& loop, unsigned cell)
{
    if (loop.bodyEntry == NoAnalysisId || !loop.atLeastOnce ||
        loop.entry >= c.graph.sites.size() || loop.exit >= c.graph.sites.size() ||
        !c.graph.legalCuts[c.graph.entry] || !c.graph.legalCuts[c.graph.exit] ||
        p.cells[cell].exclusive) return {};
    const std::set<Id> members(loop.sites.begin(), loop.sites.end());
    if (!members.count(loop.bodyEntry) || members.count(loop.entry) || members.count(loop.exit)) return {};
    std::vector<unsigned> roles(c.graph.sites.size());
    Pipe producer = Pipe::Count, consumer = Pipe::Count;
    uint64_t bankPeriod = 0, bankResidue = NoAnalysisId;
    for (auto site : members) {
        if (site >= c.graph.sites.size() || !c.reachable[site]) continue;
        const auto operation = c.graph.operations[site];
        if (operation == NoAnalysisId) continue;
        const auto& op = p.operations[operation];
        const auto role = frontiers.use(site, cell).roles;
        if (!role) continue;
        if (role == 3 || p.target.synchronous[unsigned(op.pipe)]) return {};
        const auto observation = p.observed->sites[site].observation;
        if (observation != NoAnalysisId) {
            for (const auto& atom : p.observed->observations[observation].atoms) {
                if (atom.kind != ObservationAtom::LoopResidue || atom.owner != loop.owner) {
                    continue;
                }
                if (bankPeriod && (bankPeriod != atom.parameter || bankResidue != atom.value)) {
                    return {};
                }
                bankPeriod = atom.parameter;
                bankResidue = atom.value;
            }
        }
        auto& pipe = role == 2 ? producer : consumer;
        if (pipe != Pipe::Count && pipe != op.pipe) return {};
        pipe = op.pipe;
        roles[site] = role;
    }
    if (producer == Pipe::Count || consumer == Pipe::Count || producer == consumer) return {};

    RecurringRequirement ready, release;
    ready.qualifiedCycle = release.qualifiedCycle = true;
    release.storageRelease = true;
    ready.cell = release.cell = cell;
    ready.cells = release.cells = {cell};
    ready.source = producer;
    ready.observer = consumer;
    release.source = consumer;
    release.observer = producer;
    ready.owner = release.owner = loop.owner;
    ready.period = release.period = bankPeriod ? bankPeriod : 1;
    release.publications.push_back(canonicalCommandCut(p, c.graph.entry));
    release.acquisitions.push_back(canonicalCommandCut(p, c.graph.exit));
    std::set<Cut> checkedReaderWords;
    for (auto site : members) {
        if (!roles[site]) continue;
        const auto endpoint = frontiers.recurringRelease(site, cell);
        if (endpoint == NoAnalysisId) return {};
        if (roles[site] == 2) {
            ready.publications.push_back(endpoint);
            release.acquisitions.push_back(canonicalCommandCut(p, site));
            ready.supportSeeds.push_back(canonicalCommandCut(p, site));
            continue;
        }
        const auto& participation = frontiers.readerBoundaries(site, cell, loop.owner);
        if (!participation.proved()) {
            return {};
        }
        if (checkedReaderWords.insert(c.canonicalCut[site]).second) {
            for (auto occurrence : c.wordOccurrences[c.canonicalCut[site]]) {
                if (!c.reachable[occurrence]) {
                    continue;
                }
                const auto& other = frontiers.readerBoundaries(occurrence, cell, loop.owner);
                const bool sameRoles = other.first.status == participation.first.status &&
                    other.final.status == participation.final.status;
                const bool bothTyped = other.originalInterval && participation.originalInterval;
                const bool sameInterval = !bothTyped ||
                    (other.interval.owner == participation.interval.owner &&
                     other.interval.begin == participation.interval.begin &&
                     other.interval.end == participation.interval.end);
                const bool compatible = other.proved() && sameRoles && sameInterval;
                if (!compatible) {
                    return {};
                }
            }
        }
        if (participation.first.hit()) {
            ready.acquisitions.push_back(canonicalCommandCut(p, site));
        }
        if (participation.final.hit()) {
            release.publications.push_back(endpoint);
        }
    }
    for (auto* request : {&ready, &release}) {
        for (auto* cuts : {&request->publications, &request->acquisitions}) {
            std::sort(cuts->begin(), cuts->end());
            cuts->erase(std::unique(cuts->begin(), cuts->end()), cuts->end());
        }
        if (request->publications.empty() || request->acquisitions.empty() ||
            !balanced(c, request->publications, request->acquisitions)) return {};
    }
    return {std::move(ready), std::move(release)};
}

bool balanced(const Control& c, const std::vector<Cut>& publications,
              const std::vector<Cut>& acquisitions)
{
    std::set<Cut> publish(publications.begin(), publications.end());
    std::set<Cut> acquire(acquisitions.begin(), acquisitions.end());
    for (auto cut : publish) if (acquire.count(cut)) return false;
    // Endpoint positions are distinct in this recurring vocabulary. Matching
    // itself is shared with ordinary placement and binding, over all original
    // occurrences rather than a separate empty/full-only monitor.
    return c.correspondence(publications, acquisitions).proved();
}

} // namespace

RecurringFrontiers qualifyCyclicFrontiers(
    const Program& p, const Control& c, const RequirementFrontiers& frontiers)
{
    RecurringFrontiers out;
    if (!p.observed) { return out; }
    using RoleKey = std::tuple<Pipe, Pipe, std::vector<Cut>, std::vector<Cut>>;
    std::map<RoleKey, Id> roles;
    for (const auto& loop : p.observed->loops) {
        for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
            auto local = loop.bodyEntry == NoAnalysisId
                ? qualifyCell(p, c, frontiers, loop, cell)
                : qualifyEnclosingCell(p, c, frontiers, loop, cell);
            if (local.empty()) { continue; }
            RecurringFamily family;
            family.owner = loop.owner;
            family.cells = {cell};
            family.support = local;
            for (auto& request : local) {
                const RoleKey key{request.source, request.observer, request.publications, request.acquisitions};
                const auto found = roles.find(key);
                if (found == roles.end()) {
                    const auto id = out.roles.size();
                    roles.emplace(key, id);
                    family.roles.push_back(id);
                    out.roles.push_back(std::move(request));
                } else {
                    family.roles.push_back(found->second);
                    auto& role = out.roles[found->second];
                    role.cells.insert(role.cells.end(), request.cells.begin(), request.cells.end());
                    std::sort(role.cells.begin(), role.cells.end());
                    role.cells.erase(std::unique(role.cells.begin(), role.cells.end()), role.cells.end());
                }
            }
            std::set<Cut> deadlines;
            for (auto site : loop.sites) {
                const auto operation = c.graph.operations[site];
                if (operation == NoAnalysisId || !c.reachable[site]) { continue; }
                if (frontiers.use(site, cell).roles) { deadlines.insert(c.canonicalCut[site]); }
            }
            family.deadlines.assign(deadlines.begin(), deadlines.end());
            const auto id = out.families.size();
            for (auto deadline : family.deadlines) { out.at[{deadline, cell}].push_back(id); }
            out.families.push_back(std::move(family));
        }
    }
    return out;
}
ProducerSupportScope Constructor::producerScope(const std::vector<RecurringRequirement>& obligations)
{
    ProducerSupportScope out;
    std::array<std::vector<Cut>, PipeCount> seeds;
    for (const auto& obligation : obligations) {
        // Support belongs to the selected family even when a role is shared.
        auto& sites = seeds[unsigned(obligation.source)];
        sites.insert(sites.end(), obligation.supportSeeds.begin(), obligation.supportSeeds.end());
    }
    for (auto& sites : seeds) {
        for (auto& site : sites) {
            site = control.canonicalCut[site];
        }
        std::sort(sites.begin(), sites.end());
        sites.erase(std::unique(sites.begin(), sites.end()), sites.end());
    }
    const bool needsSupport = std::any_of(seeds.begin(), seeds.end(), [](const auto& sites) {
        return !sites.empty();
    });
    if (needsSupport) {
        // This certificate protects producer-repair relocation, not arbitrary
        // endpoint motion. Both sides use original reachability, including
        // backedges and continuations; an unrelated cell is never excluded.
        for (unsigned pipe = 0; pipe < PipeCount; ++pipe) {
            if (seeds[pipe].empty()) {
                continue;
            }
            auto reached = [&](bool backward) {
                std::vector<bool> seen(control.graph.sites.size());
                std::vector<Cut> todo;
                for (auto word : seeds[pipe]) {
                    const auto& occurrences = control.wordOccurrences[control.canonicalCut[word]];
                    todo.insert(todo.end(), occurrences.begin(), occurrences.end());
                }
                while (!todo.empty()) {
                    const auto site = todo.back();
                    todo.pop_back();
                    if (seen[site] || !control.reachable[site]) {
                        continue;
                    }
                    seen[site] = true;
                    ++result.work.producerSupportWork;
                    const auto& next = backward ? control.predecessors[site] : control.graph.sites[site].successors;
                    todo.insert(todo.end(), next.begin(), next.end());
                }
                return seen;
            };
            const auto before = reached(true), after = reached(false);
            std::vector<bool> precedingOperations(program.operations.size());
            for (Cut site = 0; site < before.size(); ++site) {
                ++result.work.producerSupportWork;
                const auto operation = control.graph.operations[site];
                if (before[site] && operation != NoAnalysisId && !precedingOperations[operation]) {
                    precedingOperations[operation] = true;
                    const auto& op = program.operations[operation];
                    for (const auto& access : op.accesses) {
                        const auto base = (Id(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
                        if (access.read) {
                            out.classes[pipe].insert(base);
                        }
                        if (access.write) {
                            out.classes[pipe].insert(base + 1);
                        }
                    }
                }
            }
            for (Cut site = 0; site < after.size(); ++site) {
                const auto operation = control.graph.operations[site];
                if (after[site] && operation != NoAnalysisId &&
                    unsigned(program.operations[operation].pipe) == pipe) {
                    out.consumers.push_back(site);
                }
            }

        }
    }
    return out;
}
bool Constructor::supportsRecurring(Id id, Cut site, const FrontierRequirement& requirement) const
{
    const auto& family = recurringFrontiers.families[id];
    const auto observer = program.operations[control.graph.operations[site]].pipe;
    const bool cell = std::find(family.cells.begin(), family.cells.end(), requirement.cell) != family.cells.end();
    if (!cell) { return false; }
    return std::any_of(family.support.begin(), family.support.end(), [&](const auto& role) {
        const bool incoming = role.observer == observer && role.source == requirement.source &&
            requirement.sourceWrite != role.storageRelease;
        const bool writerReturn = role.source == observer && requirement.source == observer &&
            requirement.sourceWrite && requirement.consumerWrite && !role.storageRelease;
        return incoming || writerReturn;
    });
}
const Constructor::SupportLinks& Constructor::recurringSupportLinks(Id id)
{
    auto found = recurringLinks.find(id);
    const bool cached = found != recurringLinks.end() && found->second.version == ledger.version();
    if (cached) { return found->second; }
    auto scope = recurringScopes.find(id);
    if (scope == recurringScopes.end()) {
        scope = recurringScopes.emplace(id, producerScope(recurringFrontiers.families[id].support)).first;
    }
    SupportLinks links;
    links.version = ledger.version();
    std::set<Id> dependencies;
    for (auto site : scope->second.consumers) {
        ++result.work.recurringSupportQueries;
        const auto operation = control.graph.operations[site];
        const auto missing = frontier.inspect(cache.cuts[site].before.causal, operation).residuals;
        const auto pipe = unsigned(program.operations[operation].pipe);
        for (const auto& r : missing) {
            if (!scope->second.classes[pipe].count(accessClass(r))) { continue; }
            const auto candidates = recurringFrontiers.at.find({control.canonicalCut[site], r.cell});
            Id selected = NoAnalysisId;
            if (candidates != recurringFrontiers.at.end()) {
                for (auto family : candidates->second) {
                    ++result.work.recurringSupportQueries;
                    if (supportsRecurring(family, site, r)) { selected = family; break; }
                }
            }
            if (selected == NoAnalysisId) {
                links.reason = "generation packet leaves a producer repair without a supporting recipe";
                return recurringLinks.insert_or_assign(id, std::move(links)).first->second;
            }
            dependencies.insert(selected);
        }
    }
    links.complete = true;
    links.families.assign(dependencies.begin(), dependencies.end());
    return recurringLinks.insert_or_assign(id, std::move(links)).first->second;
}
std::optional<RecurringPacket> Constructor::prepareRecurring(
    const std::vector<RecurringRequirement>& requests, std::string& reason,
    const ProducerSupportScope* support, const std::vector<Id>* families)
{
    RecurringPacket proposal;
    proposal.requests = requests;
    proposal.supportClasses = producerSupportClasses;
    proposal.supportConsumers = producerSupportConsumers;
    std::set<Id> used;
    for (const auto& request : requests) {
        Id selected = NoAnalysisId;
        for (Id key = 0; key < frontier.keys().size(); ++key) {
            const auto& identity = frontier.keys()[key];
            const bool eligible = identity.source == request.source && identity.observer == request.observer &&
                !used.count(key) && unownedKey(key) && ledger.eventUses(identity).empty();
            if (eligible) { selected = key; break; }
        }
        if (selected == NoAnalysisId) {
            reason = "qualified recurring roles exceed their eligible unused key pool";
            return {};
        }
        used.insert(selected);
        proposal.keys.push_back(selected);
    }
    OrderedPacket endpoints;
    for (Id i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        const auto number = frontier.keys()[proposal.keys[i]].key;
        const auto channel = result.channels.size() + i;
        for (auto cut : request.publications) {
            endpoints.push_back({cut, {Command::Publish, request.source, request.observer, number},
                                 EndpointPurpose::RecurringCompletion, channel});
        }
        for (auto cut : request.acquisitions) {
            endpoints.push_back({cut, {Command::Acquire, request.source, request.observer, number},
                                 EndpointPurpose::RecurringCompletion, channel});
        }
    }
    auto owned = prepareOwnedPacket(endpoints);
    if (!owned) { reason = "recurring packet lacks complete ownership"; return {}; }
    proposal.packet = std::move(*owned);
    const auto scope = support ? *support : producerScope(requests);
    if (!scope.consumers.empty()) { proposal.supportConsumers.resize(control.graph.sites.size()); }
    for (unsigned pipe = 0; pipe < PipeCount; ++pipe) {
        proposal.supportClasses[pipe].insert(scope.classes[pipe].begin(), scope.classes[pipe].end());
    }
    for (auto site : scope.consumers) { proposal.supportConsumers[site] = true; }
    if (families && qualifyRecurringInterface(*families, proposal, scope)) {
        ++result.work.recurringLocalPackets;
        return proposal;
    }
    ++result.work.recurringLocalDeclines;
    const auto words = ledger.withPacket(proposal.packet.prepared);
    if (!words) { reason = "recurring packet changed during preparation"; return {}; }
    const auto checked = analyze(program, *words, {false});
    ++result.work.ownershipChecks;
    result.work.ownershipCheckSites += checked.stats.siteEvaluations;
    result.work.recurringAnalysisSites += checked.stats.siteEvaluations;
    if (!acceptOwnedPacket(proposal.packet, checked)) {
        reason = "recurring packet lacks event or established support";
        return {};
    }
    const auto view = ledger.packetView(proposal.packet.prepared);
    if (!view) { reason = "recurring overlay changed during qualification"; return {}; }
    proposal.evaluated = evaluateContextual(&*view, proposal.supportClasses, proposal.supportConsumers, 0);
    result.work.recurringReplaySites += proposal.evaluated.evaluations;
    if (!proposal.evaluated.success) { reason = proposal.evaluated.reason; return {}; }
    return proposal;
}
bool Constructor::commitRecurring(RecurringPacket& proposal)
{
    SelectedDecision record;
    if (!commitOwnedPacket(proposal.packet, record)) { return false; }
    producerSupportClasses = proposal.supportClasses;
    producerSupportConsumers = proposal.supportConsumers;
    for (Id i = 0; i < proposal.requests.size(); ++i) {
        const auto& request = proposal.requests[i];
        const auto key = proposal.keys[i];
        recurringKeys.insert(key);
        closedKeys.insert(key);
        result.channels.push_back({request.cell, frontier.keys()[key].key, request.cells,
            request.source, request.observer, request.publications, request.acquisitions,
            request.owner, request.period});
    }
    needsContextualReplay = true;
    result.work.recurringChannels = result.channels.size();
    if (proposal.localCertificate) {
        // Only the selected packet updates actual causal state. Failure here
        // is mandatory failure, never hidden by the immutable interface proof.
        cache = evaluateContextual(nullptr, producerSupportClasses, producerSupportConsumers, 0);
        if (!cache.success) { return fail(SelectedFailure::SelectedUpdate, cache.reason, cache.failureCut); }
    } else {
        cache = std::move(proposal.evaluated);
    }
    refreshSources();
    SelectedUpdate update;
    update.version = ledger.version();
    update.siteEvaluations = cache.evaluations;
    update.contextual = true;
    update.finalizedQueries = std::count(finalized.begin(), finalized.end(), true);
    update.changedCuts = ledger.changes();
    result.updates.push_back(std::move(update));
    ledger.clearChanges();
    ++result.work.selectedUpdates;
    ++result.work.unreusedUpdates;
    return true;
}
bool Constructor::activateRecurring()
{
    const auto operation = control.graph.operations[current];
    if (operation == NoAnalysisId || recurringFrontiers.families.empty()) { return true; }
    const auto word = control.canonicalCut[current];
    bool indexed = false;
    for (const auto& access : program.operations[operation].accesses) {
        indexed |= recurringFrontiers.at.count({word, access.cell}) != 0;
    }
    if (!indexed) { return true; }
    if (!recurringBaseline) {
        // Hypothesis traversal is not the residual used to select a recurrence.
        // Establish one cold, command-free fixed point of the selected ledger.
        needsContextualReplay = true;
        cache = {};
        if (!contextualReplay()) { return false; }
        recurringBaseline = true;
    }
    auto classes = [](const std::vector<FrontierRequirement>& residuals) {
        std::set<Id> out;
        for (const auto& r : residuals) { out.insert(accessClass(r)); }
        return out;
    };
    while (true) {
        const auto before = residual();
        if (before.empty()) { return true; }
        std::set<Id> candidates;
        for (const auto& r : before) {
            const auto found = recurringFrontiers.at.find({word, r.cell});
            if (found == recurringFrontiers.at.end()) { continue; }
            result.work.recurringCandidates += found->second.size();
            for (auto id : found->second) {
                if (supportsRecurring(id, current, r)) { candidates.insert(id); }
            }
        }
        bool activated = false;
        for (auto id : candidates) {
            auto& attempt = attemptedFamilies[id];
            if (activeFamilies[id]) { continue; }
            const bool cachedRefusal = attempt.evaluated && attempt.version == ledger.version() &&
                !attempt.improving.count(current);
            if (cachedRefusal) { continue; }
            attempt = {ledger.version(), true, {}};
            ++result.work.recurringAttempts;
            const auto& family = recurringFrontiers.families[id];
            std::set<Id> closure, roles;
            std::vector<Id> pending{id};
            ProducerSupportScope support;
            std::set<Cut> supportConsumers;
            std::string reason;
            bool supported = true;
            while (!pending.empty()) {
                const auto member = pending.back(); pending.pop_back();
                if (!closure.insert(member).second) { continue; }
                ++result.work.recurringSupportQueries;
                const auto& recipe = recurringFrontiers.families[member];
                roles.insert(recipe.roles.begin(), recipe.roles.end());
                const auto& links = recurringSupportLinks(member);
                const auto& scope = recurringScopes.at(member);
                for (unsigned pipe = 0; pipe < PipeCount; ++pipe) {
                    support.classes[pipe].insert(scope.classes[pipe].begin(), scope.classes[pipe].end());
                }
                supportConsumers.insert(scope.consumers.begin(), scope.consumers.end());
                if (!links.complete) { supported = false; reason = links.reason; break; }
                pending.insert(pending.end(), links.families.begin(), links.families.end());
            }
            std::vector<RecurringRequirement> requests;
            std::vector<Id> newRoles;
            for (auto role : roles) {
                if (activeRoles.count(role)) { continue; }
                newRoles.push_back(role);
                requests.push_back(recurringFrontiers.roles[role]);
            }
            if (supported && requests.empty()) { continue; }
            support.consumers.assign(supportConsumers.begin(), supportConsumers.end());
            const std::vector<Id> familyIds(closure.begin(), closure.end());
            auto proposal = supported ? prepareRecurring(requests, reason, &support, &familyIds) : std::nullopt;
            if (!proposal) {
                ++result.work.recurringDeclines;
                result.recurringRefusals.push_back({current, id, ledger.version(), reason});
                continue;
            }
            // One staged fixed point answers every occurrence of this recipe.
            // Remember improving deadlines, not a blanket family refusal: an
            // unrelated prelude can leave this same family useful later with
            // no intervening ledger change.
            for (auto deadline : family.deadlines) {
                for (auto site : control.wordOccurrences[deadline]) {
                    const auto op = control.graph.operations[site];
                    if (op == NoAnalysisId || !cache.cuts[site].before.causal.reachable()) { continue; }
                    const auto old = frontier.inspect(cache.cuts[site].before.causal, op);
                    const auto oldSet = classes(old.residuals);
                    auto newSet = oldSet;
                    bool valid = true;
                    if (proposal->localCertificate) {
                        for (auto covered : proposal->guaranteed[site]) { newSet.erase(covered); }
                    } else {
                        const auto next = frontier.inspect(proposal->evaluated.cuts[site].before.causal, op);
                        newSet = classes(next.residuals);
                        valid = next.failure == FrontierFailure::None || next.failure == FrontierFailure::Payload;
                    }
                    const bool reduced = valid && newSet.size() < oldSet.size() &&
                        std::includes(oldSet.begin(), oldSet.end(), newSet.begin(), newSet.end());
                    if (reduced) {
                        attempt.improving.insert(site);
                    }
                }
            }
            if (!attempt.improving.count(current)) { ++result.work.recurringDeclines; continue; }
            auto newClasses = classes(before);
            if (proposal->localCertificate) {
                for (auto covered : proposal->guaranteed[current]) { newClasses.erase(covered); }
            } else {
                const auto candidate = frontier.inspect(proposal->evaluated.cuts[current].before.causal, operation);
                newClasses = classes(candidate.residuals);
            }
            const auto firstChannel = result.channels.size();
            if (!commitRecurring(*proposal)) { return false; }
            for (Id i = 0; i < newRoles.size(); ++i) { activeRoles.emplace(newRoles[i], firstChannel + i); }
            for (auto member : closure) { activeFamilies[member] = true; }
            const auto actualClasses = classes(residual());
            const bool coveredResult = std::includes(newClasses.begin(), newClasses.end(),
                                                    actualClasses.begin(), actualClasses.end());
            const bool sameResult = coveredResult && cache.version == ledger.version();
            if (!sameResult) {
                return fail(SelectedFailure::MissingParticipation,
                    "selected recurring packet did not reduce the complete actual residual", current);
            }
            ++result.work.recurringActivations;
            result.activations.push_back({current, {closure.begin(), closure.end()},
                                          before, residual(), ledger.version()});
            activated = true;
            break;
        }
        if (!activated) { return true; }
    }
}
} // namespace mlir::pto::oahs::selected
