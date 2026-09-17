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
};
Mode mode(const Program& p, Cut cut)
{
    Mode out;
    const auto observation = p.observed->sites[cut].observation;
    if (observation == NoAnalysisId) {
        return out;
    }
    const auto& value = p.observed->observations[observation];
    if (!value.available || value.atoms.size() != 3) {
        return out;
    }
    unsigned seen = 0;
    for (const auto& atom : value.atoms) {
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
    RecurringRequirement ready{cell, producer, consumer, {}, {}}, release{cell, consumer, producer, {}, {}};
    ready.owner = release.owner = loop.owner;
    ready.period = release.period = period;
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
                if (!modes[site].next) release.acquisitions.push_back(endpoint);
            }
        }
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
} // namespace

std::vector<RecurringRequirement> qualifyCyclicFrontiers(const Program& p, const Control& c)
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
            }
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
    for (const auto& request : requests) {
        Id selected = NoAnalysisId;
        for (Id key = 0; key < frontier.keys().size(); ++key) {
            const auto& identity = frontier.keys()[key];
            if (identity.source == request.source && identity.observer == request.observer && !reserved.count(key) && !fixedKeys.count(key)) {
                selected = key;
                break;
            }
        }
        if (selected == NoAnalysisId) {
            return fail(SelectedFailure::EventResource, "qualified recurring roles exceed their eligible key pool");
        }
        reserved.insert(selected);
        const auto number = frontier.keys()[selected].key;
        const auto id = result.channels.size();
        for (auto cut : request.publications) {
            ledger.append(cut, {Command::Publish, request.source, request.observer, number},
                EndpointPurpose::RecurringCompletion, id);
        }
        for (auto cut : request.acquisitions) {
            ledger.append(cut, {Command::Acquire, request.source, request.observer, number},
                EndpointPurpose::RecurringCompletion, id);
        }
        result.channels.push_back({request.cell, number, request.source, request.observer,
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
