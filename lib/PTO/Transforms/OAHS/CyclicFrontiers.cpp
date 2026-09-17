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
struct Mode {
    Id owner = NoAnalysisId;
    uint64_t period = 0, residue = 0;
    bool previous = false, next = false, valid = false;
    bool operator==(const Mode& b) const
    {
        return valid && b.valid && owner == b.owner && period == b.period && residue == b.residue &&
               previous == b.previous && next == b.next;
    }
    bool operator<(const Mode& b) const
    {
        return std::tie(owner, period, residue, previous, next, valid) <
               std::tie(b.owner, b.period, b.residue, b.previous, b.next, b.valid);
    }
};
Mode mode(const Program& p, Cut cut)
{
    Mode out;
    const auto observation = p.observed->sites[cut].observation;
    if (observation == NoAnalysisId) {
        return out;
    }
    const auto& value = p.observed->observations[observation];
    if (!value.available || (value.atoms.size() != 3 && value.atoms.size() != 4)) {
        return out;
    }
    uint64_t period = 0;
    Id owner = NoAnalysisId;
    for (const auto &atom : value.atoms)
        if (atom.kind == ObservationAtom::LoopResidue) { period = atom.parameter; owner = atom.owner; }
    unsigned seen = 0;
    for (const auto& atom : value.atoms) {
        if (period > 1 && atom.kind == ObservationAtom::LoopHasNext && atom.parameter == 1 && atom.owner == owner) continue;
        if (seen && (atom.owner != out.owner || atom.parameter != out.period)) {
            return {};
        }
        out.owner = atom.owner;
        out.period = atom.parameter;
        if (atom.kind == ObservationAtom::LoopResidue) {
            out.residue = atom.value;
            seen |= 1;
        } else if (atom.kind == ObservationAtom::LoopHasPrevious) {
            out.previous = atom.value != 0;
            seen |= 2;
        } else if (atom.kind == ObservationAtom::LoopHasNext) {
            out.next = atom.value != 0;
            seen |= 4;
        } else {
            return {};
        }
    }
    out.valid = seen == 7 && out.period != 0 && out.residue < out.period;
    return out;
}
Cut after(const Program& p, const Control& c, Id site, const Mode& expected)
{
    // A unique legal source position in the unchanged current-visit corridor.
    std::set<Id> candidates;
    auto todo = c.graph.sites[site].successors;
    std::vector<bool> seen(c.graph.sites.size());
    while (!todo.empty()) {
        const auto at = todo.back();
        todo.pop_back();
        if (seen[at]) {
            continue;
        }
        seen[at] = true;
        if (c.graph.legalCuts[at]) {
            if (!(mode(p, at) == expected)) {
                return NoAnalysisId;
            }
            candidates.insert(canonicalCommandCut(p, at));
            continue;
        }
        if (c.graph.operations[at] != NoAnalysisId || at == c.graph.exit) {
            return NoAnalysisId;
        }
        const auto& next = c.graph.sites[at].successors;
        todo.insert(todo.end(), next.begin(), next.end());
    }
    return candidates.size() == 1 ? *candidates.begin() : NoAnalysisId;
}
// Project one physical cell's access roles, without making storage succession
// imply completion. All other payload remains in the selected full-graph replay.
std::vector<RecurringRequirement> qualifyCell(
    const Program& p, const Control& c, const ObservedLoop& loop, unsigned cell)
{
    std::set<Id> members(loop.sites.begin(), loop.sites.end());
    if (loop.entry >= c.graph.sites.size() || loop.exit >= c.graph.sites.size() ||
        members.count(loop.entry) || members.count(loop.exit) || members.count(c.graph.entry)) return {};
    // Metadata is a proposal, not a trusted interface. Check all entries and
    // exits, including a root entry with no predecessor edge.
    for (Id site = 0; site < c.graph.sites.size(); ++site) {
        for (auto next : c.graph.sites[site].successors) {
            if (members.count(next) && !members.count(site) && site != loop.entry) return {};
            if (members.count(site) && !members.count(next) && next != loop.exit) return {};
        }
    }
    if (p.cells[cell].exclusive) return {};
    std::vector<Mode> modes(c.graph.sites.size());
    std::vector<unsigned> roles(c.graph.sites.size());
    Pipe producer = Pipe::Count, consumer = Pipe::Count;
    uint64_t residue = NoAnalysisId, period = 0;
    for (auto site : members) {
        if (site >= c.graph.sites.size()) return {};
        const auto operation = c.graph.operations[site];
        if (!c.reachable[site] || operation == NoAnalysisId) continue;
        const auto& op = p.operations[operation];
        unsigned role = 0;
        for (const auto& a : op.accesses) if (a.cell == cell) role |= unsigned(a.read) | (unsigned(a.write) << 1);
        if (!role) continue;
        const auto m = mode(p, site);
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
    auto nextAccess = [&](const std::vector<Id>& starts) {
        std::set<Id> out;
        auto todo = starts;
        std::vector<bool> seen(c.graph.sites.size());
        while (!todo.empty()) {
            const auto at = todo.back(); todo.pop_back();
            if (seen[at]) continue;
            seen[at] = true;
            if (at == loop.exit || roles[at]) out.insert(at);
            else if (members.count(at) || at == loop.entry) {
                const auto& next = c.graph.sites[at].successors;
                todo.insert(todo.end(), next.begin(), next.end());
            }
        }
        return out;
    };
    for (auto first : nextAccess({loop.entry})) {
        if (first != loop.exit && (roles[first] != 2 || modes[first].previous)) return {};
    }
    // Track first/reused permission from the real local entry and all backedges.
    std::vector<unsigned> previous(c.graph.sites.size());
    std::deque<Id> queue{loop.entry};
    previous[loop.entry] = 1;
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
        if (next.empty()) return {};
        for (auto target : next) {
            if (target == loop.exit) {
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
    std::vector<Id> later{loop.exit};
    std::vector<bool> visited(c.graph.sites.size());
    while (!later.empty()) {
        const auto at = later.back(); later.pop_back();
        if (at == loop.entry) { reentered = true; break; }
        if (visited[at]) continue;
        visited[at] = true;
        const auto& next = c.graph.sites[at].successors;
        later.insert(later.end(), next.begin(), next.end());
    }
    RecurringRequirement ready, release;
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
            const auto m = mode(p, site);
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
        const auto endpoint = after(p, c, site, modes[site]);
        if (endpoint == NoAnalysisId) return {};
        if (roles[site] == 2) {
            if (previous[site] != (modes[site].previous ? 2u : 1u)) return {};
            ready.publications.push_back(endpoint);
            if (modes[site].previous) release.acquisitions.push_back(cut);
        } else {
            if (predecessors[site] == 2) ready.acquisitions.push_back(cut);
            else if (predecessors[site] != 1) return {}; // ambiguous first reader
            const auto next = nextAccess(c.graph.sites[site].successors);
            unsigned nextRoles = 0;
            for (auto target : next) nextRoles |= target == loop.exit ? 4u : roles[target];
            if ((nextRoles & 1) && nextRoles != 1) return {}; // ambiguous last reader
            if (!(nextRoles & 1) && (modes[site].next || reentered)) {
                release.publications.push_back(endpoint);
                if (!modes[site].next && finalAcquisitions.empty()) release.acquisitions.push_back(endpoint);
            }
        }
    }
    release.acquisitions.insert(release.acquisitions.end(), finalAcquisitions.begin(), finalAcquisitions.end());
    for (auto* request : {&ready, &release}) {
        for (auto* cuts : {&request->publications, &request->acquisitions}) {
            std::sort(cuts->begin(), cuts->end());
            cuts->erase(std::unique(cuts->begin(), cuts->end()), cuts->end());
        }
        if (request->publications.empty() || request->acquisitions.empty()) return {};
    }
    return {std::move(ready), std::move(release)};
}

bool balanced(const Control& c, const std::vector<Cut>& publications,
              const std::vector<Cut>& acquisitions)
{
    std::set<Cut> publish(publications.begin(), publications.end());
    std::set<Cut> acquire(acquisitions.begin(), acquisitions.end());
    for (auto cut : publish) if (acquire.count(cut)) return false;
    std::vector<uint8_t> incoming(c.graph.sites.size());
    std::deque<Id> queue;
    std::vector<bool> queued(c.graph.sites.size());
    bool invalid = false;
    incoming[c.graph.entry] = 1; // bit 0: empty; bit 1: full; bit 2: invalid
    queue.push_back(c.graph.entry);
    queued[c.graph.entry] = true;
    while (!queue.empty()) {
        const auto site = queue.front();
        queue.pop_front();
        queued[site] = false;
        auto state = incoming[site];
        const auto cut = c.canonicalCut[site];
        if (publish.count(cut)) {
            uint8_t next = state & 4;
            if (state & 1) next |= 2;
            if (state & 2) {
                next |= 4;
                invalid = true;
            }
            state = next;
        }
        if (acquire.count(cut)) {
            uint8_t next = state & 4;
            if (state & 2) next |= 1;
            if (state & 1) {
                next |= 4;
                invalid = true;
            }
            state = next;
        }
        for (auto successor : c.graph.sites[site].successors) {
            const auto joined = uint8_t(incoming[successor] | state);
            if (joined != incoming[successor]) {
                incoming[successor] = joined;
                if (!queued[successor]) {
                    queue.push_back(successor);
                    queued[successor] = true;
                }
            }
        }
    }
    return !invalid && incoming[c.graph.exit] == 1;
}

std::vector<RecurringRequirement> qualifyRelationships(
    const Program& p, const Control& c, const StorageFrontierAnalysis& storage)
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
            const auto targetMode = mode(p, target);
            if (!targetMode.valid || targetMode.owner != loop.owner) continue;
            const auto targetPipe = p.operations[targetOperation].pipe;
            if (p.target.synchronous[unsigned(targetPipe)]) continue;
            for (const auto& relationship : storage.relationshipsAt(target)) {
                const auto sourceSite = relationship.source.site;
                if (!members.count(sourceSite) || sourceSite >= c.graph.sites.size()) continue;
                const auto sourceOperation = c.graph.operations[sourceSite];
                if (sourceOperation == NoAnalysisId || relationship.cell >= p.cells.size() ||
                    p.cells[relationship.cell].exclusive) continue;
                const auto sourcePipe = p.operations[sourceOperation].pipe;
                const auto sourceMode = mode(p, sourceSite);
                if (sourcePipe == targetPipe || p.target.synchronous[unsigned(sourcePipe)] ||
                    !sourceMode.valid || sourceMode.owner != loop.owner ||
                    sourceMode.period != targetMode.period) continue;
                const auto publication = after(p, c, sourceSite, sourceMode);
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
    const Program& p, const Control& c, const StorageFrontierAnalysis& storage)
{
    std::vector<RecurringRequirement> requests;
    if (!p.observed) return requests;
    for (const auto& loop : p.observed->loops) {
        for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
            auto local = qualifyCell(p, c, loop, cell);
            for (auto& request : local) {
                // Several conservative storage witnesses can name the same
                // physical role. One actual prefix serves their conjunction.
                const auto duplicate = std::find_if(requests.begin(), requests.end(), [&](const auto& old) {
                    return old.source == request.source && old.observer == request.observer &&
                        old.publications == request.publications && old.acquisitions == request.acquisitions;
                });
                if (duplicate == requests.end()) requests.push_back(std::move(request));
                else {
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
    auto relationshipRequests = qualifyRelationships(p, c, storage);
    for (auto& candidate : relationshipRequests) {
        auto subset = [](const std::vector<Cut>& a, const std::vector<Cut>& b) {
            return std::includes(b.begin(), b.end(), a.begin(), a.end());
        };
        requests.erase(std::remove_if(requests.begin(), requests.end(), [&](const auto& old) {
            return old.owner == candidate.owner && old.source == candidate.source &&
                old.observer == candidate.observer && subset(old.publications, candidate.publications) &&
                subset(old.acquisitions, candidate.acquisitions);
        }), requests.end());
        requests.push_back(std::move(candidate));
    }
    if (!relationshipRequests.empty() && !protocolCompatible(p, c, requests)) requests = ordinary;
    // Apply F3/F4 to recurring roles too. If one side of two role interfaces
    // is already the same frontier, a later compatible source prefix (or an
    // earlier compatible acquisition) can serve the conjunction. We only
    // merge one unambiguous cut per original occurrence mode; alternatives
    // remain separate until their participation correspondence is proved.
    auto indexed = [&](const std::vector<Cut>& cuts) {
        std::map<Mode, Cut> out;
        for (auto cut : cuts) {
            const auto key = mode(p, cut);
            if (!key.valid || !out.emplace(key, cut).second) return std::map<Mode, Cut>{};
        }
        return out;
    };
    auto combine = [&](const std::vector<Cut>& a, const std::vector<Cut>& b, bool later,
                       std::vector<Cut>& out) {
        const auto left = indexed(a), right = indexed(b);
        if (left.empty() || left.size() != right.size()) return false;
        out.clear();
        for (const auto& [key, x] : left) {
            const auto found = right.find(key);
            if (found == right.end()) return false;
            const auto y = found->second;
            if (c.straight(x, y)) out.push_back(later ? y : x);
            else if (c.straight(y, x)) out.push_back(later ? x : y);
            else return false;
        }
        std::sort(out.begin(), out.end());
        return true;
    };
    for (Id i = 0; i < requests.size(); ++i) {
        for (Id j = i + 1; j < requests.size();) {
            auto& a = requests[i];
            auto& b = requests[j];
            if (a.source != b.source || a.observer != b.observer || a.owner != b.owner ||
                a.period != b.period) {
                ++j;
                continue;
            }
            std::vector<Cut> merged;
            bool compatible = false;
            if (a.acquisitions == b.acquisitions && combine(a.publications, b.publications, true, merged)) {
                a.publications = std::move(merged);
                compatible = true;
            } else if (a.publications == b.publications &&
                       combine(a.acquisitions, b.acquisitions, false, merged)) {
                a.acquisitions = std::move(merged);
                compatible = true;
            }
            if (!compatible) {
                ++j;
                continue;
            }
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
    auto candidate = [&]() {
        auto commands = ledger.commands();
        for (Id i = 0; i < requests.size(); ++i) {
            if (!retained[i]) continue;
            const auto& r = requests[i];
            const auto number = frontier.keys()[keys[i]].key;
            auto add = [&](Cut cut, Command::Kind kind) {
                for (auto site : control.wordOccurrences[control.canonicalCut[cut]])
                    commands[site].push_back({kind, r.source, r.observer, number});
            };
            for (auto cut : r.publications) add(cut, Command::Publish);
            for (auto cut : r.acquisitions) add(cut, Command::Acquire);
        }
        return commands;
    };
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
        if (!alternativeRoute(index)) continue;
        if (!selected) {
            selected = analyze(program, candidate(), {false});
            result.work.recurringAnalysisSites += selected->stats.siteEvaluations;
        }
        if (!selected->complete || !selected->diagnostics.empty() ||
            !selected->protocol.empty() || !selected->phaseResources.empty()) break;
        retained[index] = false;
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
        } else retained[index] = true;
    }
    for (Id index = 0; index < requests.size(); ++index) {
        if (!retained[index]) continue;
        const auto& request = requests[index];
        reserved.insert(keys[index]);
        const auto number = frontier.keys()[keys[index]].key;
        const auto id = result.channels.size();
        for (auto cut : request.publications) {
            ledger.append(cut, {Command::Publish, request.source, request.observer, number},
                EndpointPurpose::RecurringCompletion, id);
        }
        for (auto cut : request.acquisitions) {
            ledger.append(cut, {Command::Acquire, request.source, request.observer, number},
                EndpointPurpose::RecurringCompletion, id);
        }
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
