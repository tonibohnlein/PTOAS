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
        const auto& participation = frontiers.readerParticipation(site, cell);
        if (!participation.proved()) {
            return {};
        }
        if (checkedReaderWords.insert(c.canonicalCut[site]).second) {
            for (auto occurrence : c.wordOccurrences[c.canonicalCut[site]]) {
                if (!c.reachable[occurrence]) {
                    continue;
                }
                const auto& other = frontiers.readerParticipation(occurrence, cell);
                if (!other.proved() || other.first != participation.first || other.final != participation.final) {
                    return {};
                }
            }
        }
        if (participation.first) {
            ready.acquisitions.push_back(canonicalCommandCut(p, site));
        }
        if (participation.final) {
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

std::vector<RecurringRequirement> qualifyRelationships(
    const Program& p, const Control& c, const RequirementFrontiers& frontiers)
{
    using Key = std::tuple<Id, uint64_t, Pipe, Pipe, Id, Id>;
    std::map<Key, RecurringRequirement> grouped;
    if (!p.observed) return {};
    for (const auto& loop : p.observed->loops) {
        const std::set<Id> members(loop.sites.begin(), loop.sites.end());
        for (auto target : loop.sites) {
            if (target >= c.graph.sites.size() || !c.reachable[target]) continue;
            const auto targetOperation = c.graph.operations[target];
            if (targetOperation == NoAnalysisId) continue;
            const auto targetMode = occurrenceMode(p, target);
            if (!targetMode.valid || targetMode.owner != loop.owner) continue;
            const auto targetPipe = p.operations[targetOperation].pipe;
            if (p.target.synchronous[unsigned(targetPipe)]) continue;
            for (const auto& requirement : frontiers.at(target)) {
                const auto& relationship = requirement.relationship;
                const auto sourceSite = relationship.source.site;
                if (!members.count(sourceSite) || sourceSite >= c.graph.sites.size()) continue;
                const auto sourceOperation = c.graph.operations[sourceSite];
                if (sourceOperation == NoAnalysisId || relationship.cell >= p.cells.size() ||
                    p.cells[relationship.cell].exclusive) continue;
                const auto sourcePipe = p.operations[sourceOperation].pipe;
                const auto sourceMode = occurrenceMode(p, sourceSite);
                if (sourcePipe == targetPipe || p.target.synchronous[unsigned(sourcePipe)] ||
                    !sourceMode.valid || sourceMode.owner != loop.owner ||
                    sourceMode.period != targetMode.period) continue;
                const auto publication = frontiers.recurringRelease(sourceSite, relationship.cell);
                if (publication == NoAnalysisId) continue;
                for (const auto key : {
                        Key{loop.owner, sourceMode.period, sourcePipe, targetPipe,
                            NoAnalysisId, NoAnalysisId},
                        Key{loop.owner, sourceMode.period, sourcePipe, targetPipe,
                            sourceOperation, targetOperation}}) {
                    auto& request = grouped[key];
                    request.source = sourcePipe;
                    request.observer = targetPipe;
                    request.owner = loop.owner;
                    request.period = sourceMode.period;
                    request.cells.push_back(relationship.cell);
                    request.publications.push_back(publication);
                    request.acquisitions.push_back(canonicalCommandCut(p, target));
                }
            }
        }
    }
    std::vector<RecurringRequirement> out;
    auto normalize = [&](RecurringRequirement& request) {
        std::sort(request.cells.begin(), request.cells.end());
        request.cells.erase(std::unique(request.cells.begin(), request.cells.end()), request.cells.end());
        for (auto* values : {&request.publications, &request.acquisitions}) {
            std::sort(values->begin(), values->end());
            values->erase(std::unique(values->begin(), values->end()), values->end());
        }
        request.cell = request.cells.front();
    };
    for (auto& [key, request] : grouped) normalize(request);
    std::set<std::tuple<Id, uint64_t, Pipe, Pipe>> coarseAccepted;
    for (auto& [key, request] : grouped) {
        const auto [owner, period, source, observer, sourceOperation, targetOperation] = key;
        const auto coarse = std::make_tuple(owner, period, source, observer);
        if (sourceOperation != NoAnalysisId) continue;
        if (balanced(c, request.publications, request.acquisitions)) {
            coarseAccepted.insert(coarse);
            out.push_back(request);
        }
    }
    for (auto& [key, request] : grouped) {
        const auto [owner, period, source, observer, sourceOperation, targetOperation] = key;
        const auto coarse = std::make_tuple(owner, period, source, observer);
        if (sourceOperation == NoAnalysisId) continue;
        if (!coarseAccepted.count(coarse) && balanced(c, request.publications, request.acquisitions)) {
            out.push_back(request);
        }
    }
    return out;
}

bool protocolCompatible(const Program& p, const Control& c,
                        const std::vector<RecurringRequirement>& requests)
{
    Commands commands(commandCutCount(p));
    std::map<std::pair<Pipe, Pipe>, std::set<unsigned>> used;
    for (const auto& reservation : p.reservations) {
        used[{reservation.source, reservation.observer}].insert(reservation.key);
    }
    for (const auto& request : requests) {
        const auto direction = std::make_pair(request.source, request.observer);
        unsigned number = std::numeric_limits<unsigned>::max();
        for (auto candidate : p.target.keys[unsigned(request.source)][unsigned(request.observer)]) {
            if (!used[direction].count(candidate)) {
                number = candidate;
                break;
            }
        }
        if (number == std::numeric_limits<unsigned>::max()) return false;
        used[direction].insert(number);
        auto add = [&](Cut cut, Command::Kind kind) {
            const auto canonical = c.canonicalCut[cut];
            for (auto site : c.wordOccurrences[canonical]) {
                commands[site].push_back({kind, request.source, request.observer, number});
            }
        };
        for (auto cut : request.publications) add(cut, Command::Publish);
        for (auto cut : request.acquisitions) add(cut, Command::Acquire);
    }
    const auto checked = analyze(p, commands, {false});
    if (!checked.complete || !checked.diagnostics.empty()) return false;
    return std::none_of(checked.protocol.begin(), checked.protocol.end(), [](const auto& obligation) {
        return obligation.kind != ProtocolObligation::ReceiptNotEstablished;
    });
}
} // namespace

std::vector<RecurringRequirement> qualifyCyclicFrontiers(
    const Program& p, const Control& c, const RequirementFrontiers& frontiers)
{
    std::vector<RecurringRequirement> requests;
    if (!p.observed) return requests;
    for (const auto& loop : p.observed->loops) {
        for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
            auto local = loop.bodyEntry == NoAnalysisId
                ? qualifyCell(p, c, frontiers, loop, cell)
                : qualifyEnclosingCell(p, c, frontiers, loop, cell);
            for (auto& request : local) {
                // Several conservative storage witnesses can name the same
                // physical role. One actual prefix serves their conjunction.
                const auto duplicate = std::find_if(requests.begin(), requests.end(), [&](const auto& old) {
                    return old.source == request.source && old.observer == request.observer &&
                        old.publications == request.publications && old.acquisitions == request.acquisitions;
                });
                if (duplicate == requests.end()) requests.push_back(std::move(request));
                else {
                    duplicate->qualifiedCycle &= request.qualifiedCycle;
                    duplicate->supportSeeds.insert(duplicate->supportSeeds.end(),
                        request.supportSeeds.begin(), request.supportSeeds.end());
                    duplicate->cells.insert(duplicate->cells.end(), request.cells.begin(), request.cells.end());
                    std::sort(duplicate->cells.begin(), duplicate->cells.end());
                    duplicate->cells.erase(std::unique(duplicate->cells.begin(), duplicate->cells.end()),
                                           duplicate->cells.end());
                }
            }
        }
    }
    // Complex loops can contain RMW accesses and more than two participating
    // pipelines. Build their recurring frontiers from the shared storage
    // succession relation. Prefer a direction-wide word; if its uses do not
    // alternate, retain independently balanced operation-pair words. Every
    // retained word has exact original-control participation under the finite
    // balance monitor, and the combined protocol is checked below.
    // This grants no completion credit: the completed combined ledger is still
    // checked by the causal frontier before emission.
    const auto ordinary = requests;
    auto relationshipRequests = qualifyRelationships(p, c, frontiers);
    for (auto& candidate : relationshipRequests) {
        auto subset = [](const std::vector<Cut>& a, const std::vector<Cut>& b) {
            return std::includes(b.begin(), b.end(), a.begin(), a.end());
        };
        requests.erase(std::remove_if(requests.begin(), requests.end(), [&](const auto& old) {
            const bool replaced = old.owner == candidate.owner && old.source == candidate.source &&
                old.observer == candidate.observer && subset(old.publications, candidate.publications) &&
                subset(old.acquisitions, candidate.acquisitions);
            if (replaced) {
                candidate.supportSeeds.insert(candidate.supportSeeds.end(),
                    old.supportSeeds.begin(), old.supportSeeds.end());
            }
            return replaced;
        }), requests.end());
        requests.push_back(std::move(candidate));
    }
    if (!relationshipRequests.empty() && !protocolCompatible(p, c, requests)) requests = ordinary;
    // Sharing is position-preserving. Occurrence balance and straight paths
    // do not prove that delaying a publication or advancing an acquisition
    // preserves the surrounding payload order. Keep distinct boundaries until
    // a contextual ordering certificate can justify their movement.
    for (Id i = 0; i < requests.size(); ++i) {
        auto& a = requests[i];
        for (Id j = i + 1; j < requests.size();) {
            const auto& b = requests[j];
            if (a.source != b.source || a.observer != b.observer || a.owner != b.owner ||
                a.period != b.period || a.storageRelease != b.storageRelease ||
                a.publications != b.publications || a.acquisitions != b.acquisitions) {
                ++j;
                continue;
            }
            a.qualifiedCycle &= b.qualifiedCycle;
            a.supportSeeds.insert(a.supportSeeds.end(), b.supportSeeds.begin(), b.supportSeeds.end());
            a.cells.insert(a.cells.end(), b.cells.begin(), b.cells.end());
            std::sort(a.cells.begin(), a.cells.end());
            a.cells.erase(std::unique(a.cells.begin(), a.cells.end()), a.cells.end());
            a.cell = a.cells.front();
            requests.erase(requests.begin() + j);
        }
    }
    return requests;
}
bool Constructor::recurring(const std::vector<RecurringRequirement>& requests)
{
    auto& reserved = recurringKeys;
    std::set<Id> fixedKeys;
    for (const auto& endpoint : ledger.records()) {
        const auto& command = endpoint.command;
        if (command.kind != Command::Publish && command.kind != Command::Acquire) continue;
        for (Id key = 0; key < frontier.keys().size(); ++key) {
            const auto& identity = frontier.keys()[key];
            if (identity.source == command.source && identity.observer == command.observer && identity.key == command.key)
                fixedKeys.insert(key);
        }
    }
    // Allocate the proposed words without committing them to the ledger. A
    // channel is not necessary merely because its participation is qualified.
    std::vector<Id> keys;
    std::set<Id> proposedKeys;
    for (const auto& request : requests) {
        Id selected = NoAnalysisId;
        for (Id key = 0; key < frontier.keys().size(); ++key) {
            const auto& identity = frontier.keys()[key];
            if (identity.source == request.source && identity.observer == request.observer && !proposedKeys.count(key) && !fixedKeys.count(key)) {
                selected = key;
                break;
            }
        }
        if (selected == NoAnalysisId) {
            return fail(SelectedFailure::EventResource, "qualified recurring roles exceed their eligible key pool");
        }
        proposedKeys.insert(selected);
        keys.push_back(selected);
    }
    std::vector<bool> retained(requests.size(), true);
    auto packet = [&]() {
        OrderedPacket endpoints;
        Id channel = result.channels.size();
        for (Id i = 0; i < requests.size(); ++i) {
            if (!retained[i]) {
                continue;
            }
            const auto& request = requests[i];
            const auto number = frontier.keys()[keys[i]].key;
            auto add = [&](Cut cut, Command::Kind kind) {
                endpoints.push_back({cut, {kind, request.source, request.observer, number},
                                     EndpointPurpose::RecurringCompletion, channel});
            };
            for (auto cut : request.publications) {
                add(cut, Command::Publish);
            }
            for (auto cut : request.acquisitions) {
                add(cut, Command::Acquire);
            }
            ++channel;
        }
        return endpoints;
    };
    auto prepared = ledger.preparePacket(packet());
    if (!prepared.valid()) {
        return fail(SelectedFailure::SelectedUpdate, prepared.reason());
    }
    auto candidate = [&]() { return *ledger.withPacket(prepared); };
    auto alternativeRoute = [&](Id omitted) {
        // Immutable topology is only a cheap opportunity filter. Actual prefix,
        // occurrence, and consumption coverage must pass full replay below.
        std::set<Pipe> reached{requests[omitted].source};
        bool changed = true;
        while (changed) {
            changed = false;
            for (Id i = 0; i < requests.size(); ++i)
                if (i != omitted && retained[i] && reached.count(requests[i].source))
                    changed |= reached.insert(requests[i].observer).second;
        }
        return reached.count(requests[omitted].observer) != 0;
    };
    auto requirementKeys = [](const AnalysisResult& report) {
        std::set<std::tuple<Cut, Id, Id, unsigned, unsigned, unsigned>> out;
        for (const auto& r : report.residuals)
            out.emplace(r.consumerCut, r.demand.producer, r.demand.consumer,
                        r.demand.cell, unsigned(r.kind), unsigned(r.demand.property));
        return out;
    };
    std::optional<AnalysisResult> selected;
    for (Id index = requests.size(); index-- > 0;) {
        if (requests[index].qualifiedCycle) continue;
        if (!alternativeRoute(index)) continue;
        if (!selected) {
            selected = analyze(program, candidate(), {false});
            result.work.recurringAnalysisSites += selected->stats.siteEvaluations;
        }
        if (!selected->complete || !selected->diagnostics.empty() ||
            !selected->protocol.empty() || !selected->phaseResources.empty()) break;
        retained[index] = false;
        auto previousPacket = std::move(prepared);
        prepared = ledger.preparePacket(packet());
        if (!prepared.valid()) {
            return fail(SelectedFailure::SelectedUpdate, prepared.reason());
        }
        auto trial = analyze(program, candidate(), {false});
        ++result.work.recurringTrials;
        result.work.recurringAnalysisSites += trial.stats.siteEvaluations;
        const auto before = requirementKeys(*selected), after = requirementKeys(trial);
        // An otherwise memory-redundant return may be the only acknowledgment
        // for another key. Require all remaining event preconditions, not only
        // byte completion, and never accept a newly uncovered payload demand.
        const bool covered = trial.complete && trial.diagnostics.empty() &&
            trial.protocol.empty() && trial.phaseResources.empty() &&
            std::includes(before.begin(), before.end(), after.begin(), after.end()) &&
            std::all_of(trial.retirement.begin(), trial.retirement.end(), [&](const auto& r) {
                return std::any_of(selected->retirement.begin(), selected->retirement.end(), [&](const auto& old) {
                    return r.operation == old.operation && r.observer == old.observer;
                });
            });
        if (covered) {
            selected = std::move(trial);
            ++result.work.redundantRecurringChannels;
        } else {
            retained[index] = true;
            prepared = std::move(previousPacket);
        }
    }
    std::array<std::vector<Cut>, PipeCount> seeds;
    for (Id index = 0; index < requests.size(); ++index) {
        // A replacement or omission can preserve the cycle's causal effect.
        // Support belongs to the proposal, not the private channel identity.
        auto& sites = seeds[unsigned(requests[index].source)];
        sites.insert(sites.end(), requests[index].supportSeeds.begin(), requests[index].supportSeeds.end());
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
        if (!selected) {
            selected = analyze(program, candidate(), {false});
            result.work.recurringAnalysisSites += selected->stats.siteEvaluations;
        }
        if (!selected->complete || !selected->diagnostics.empty() || !selected->protocol.empty() ||
            !selected->phaseResources.empty()) {
            return fail(SelectedFailure::LoopInvariant, "generation packet has unestablished protocol support");
        }
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
                if (before[site] && operation != NoAnalysisId) {
                    precedingOperations[operation] = true;
                }
            }
            for (const auto& residual : selected->residuals) {
                ++result.work.producerSupportWork;
                if (unsigned(program.operations[residual.demand.consumer].pipe) != pipe ||
                    residual.consumerCut >= after.size() || !after[residual.consumerCut]) {
                    continue;
                }
                if (precedingOperations[residual.demand.producer]) {
                    return fail(SelectedFailure::LoopInvariant,
                                "generation packet leaves a producer repair crossing its overwrite",
                                residual.consumerCut);
                }
            }
        }
    }
    if (ledger.appendPacket(prepared).size() != prepared.size()) {
        return fail(SelectedFailure::SelectedUpdate, "recurring packet changed after checking");
    }
    for (Id index = 0; index < requests.size(); ++index) {
        if (!retained[index]) continue;
        const auto& request = requests[index];
        reserved.insert(keys[index]);
        needsContextualReplay = true;
        const auto number = frontier.keys()[keys[index]].key;
        result.channels.push_back({request.cell, number, request.cells, request.source, request.observer,
                                   request.publications, request.acquisitions, request.owner,
                                   request.period});
    }
    result.work.recurringChannels = result.channels.size();
    // These are physical access roles, not definite-write/content certificates.
    // They are symbolic obligations, not assumed fresh-entry receipts.
    // finish() checks the entire selected ledger from the original root, through
    // every original entry/backedge/exit. Failure exports no executable program.
    return true;
}
} // namespace mlir::pto::oahs::selected
