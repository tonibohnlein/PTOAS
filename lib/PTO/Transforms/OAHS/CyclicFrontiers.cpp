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
std::set<Id> nextPayload(const Control& c, const std::vector<Id>& starts)
{
    std::set<Id> out;
    auto todo = starts;
    std::vector<bool> seen(c.graph.sites.size());
    while (!todo.empty()) {
        const auto site = todo.back();
        todo.pop_back();
        if (seen[site]) {
            continue;
        }
        seen[site] = true;
        if (c.graph.operations[site] != NoAnalysisId || site == c.graph.exit) {
            out.insert(site);
        } else {
            const auto& next = c.graph.sites[site].successors;
            todo.insert(todo.end(), next.begin(), next.end());
        }
    }
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
std::vector<unsigned> priorReader(const Program& p, const Control& c, unsigned cell)
{
    std::vector<unsigned> masks(c.graph.sites.size());
    std::deque<Id> queue{c.graph.entry};
    masks[c.graph.entry] = 1; // no previous read, versus a previous read (bit 1)
    while (!queue.empty()) {
        const auto site = queue.front();
        queue.pop_front();
        auto bits = masks[site];
        const auto operation = c.graph.operations[site];
        if (operation != NoAnalysisId) {
            const auto& a = p.operations[operation].accesses.front();
            if (a.cell == cell && a.read) {
                bits = 2;
            }
        }
        for (auto next : c.graph.sites[site].successors) {
            const auto joined = masks[next] | bits;
            if (joined != masks[next]) {
                masks[next] = joined;
                queue.push_back(next);
            }
        }
    }
    return masks;
}
unsigned nextWriter(const Program& p, const Control& c, Id reader, unsigned cell)
{
    unsigned possibilities = 0;
    auto todo = c.graph.sites[reader].successors;
    std::vector<bool> seen(c.graph.sites.size());
    while (!todo.empty()) {
        const auto site = todo.back();
        todo.pop_back();
        if (seen[site]) {
            continue;
        }
        seen[site] = true;
        if (site == c.graph.exit) {
            possibilities |= 1;
            continue;
        }
        const auto operation = c.graph.operations[site];
        if (operation != NoAnalysisId) {
            const auto& a = p.operations[operation].accesses.front();
            if (a.cell == cell && a.write) {
                possibilities |= 2;
                continue;
            }
        }
        const auto& next = c.graph.sites[site].successors;
        todo.insert(todo.end(), next.begin(), next.end());
    }
    return possibilities;
}
bool exactSlots(const Program& p)
{
    for (Id i = 0; i < p.cells.size(); ++i) {
        const auto& a = p.cells[i];
        if (a.storage != Cell::Storage::CanonicalInterval || a.exclusive || a.unknownRange) {
            return false;
        }
        for (Id j = 0; j < i; ++j) {
            const auto& b = p.cells[j];
            if (a.addressSpace != b.addressSpace || a.coordinateSpace != b.coordinateSpace) {
                return false; // this first qualifier requires one exact storage space
            }
            const auto x = a.ranges.front(), y = b.ranges.front();
            if (x.first < y.first + y.second && y.first < x.first + x.second) {
                return false;
            }
        }
    }
    return true;
}
} // namespace

std::vector<RecurringRequirement> qualifyCyclicFrontiers(const Program& p, const Control& c)
{
    if (!p.observed || p.invocation.retirement != Program::InvocationContract::NoRetirement ||
        p.operations.empty() || p.cells.empty() || !exactSlots(p)) {
        return {};
    }
    std::vector<Mode> modes(c.graph.sites.size());
    std::vector<unsigned> slot(p.cells.size(), std::numeric_limits<unsigned>::max());
    std::vector<bool> writes(c.graph.sites.size());
    Pipe producer = Pipe::Count, consumer = Pipe::Count;
    Id owner = NoAnalysisId;
    std::vector<Id> payloads;
    for (Id site = 0; site < c.graph.sites.size(); ++site) {
        const auto operation = c.graph.operations[site];
        if (!c.reachable[site] || operation == NoAnalysisId) {
            continue;
        }
        const auto& op = p.operations[operation];
        const auto m = mode(p, site);
        if (!m.valid || m.period != p.cells.size() || (owner != NoAnalysisId && owner != m.owner) ||
            op.accesses.size() != 1 || op.accesses[0].read == op.accesses[0].write ||
            !op.resources.empty() || !op.visibility.empty() || !op.authoredEvents.empty() ||
            !op.internalTransfers.empty() || op.finalBlock || p.target.synchronous[unsigned(op.pipe)]) {
            return {};
        }
        owner = m.owner;
        const auto& a = op.accesses[0];
        if (a.write && !a.definiteWrite) {
            return {};
        }
        if (slot[m.residue] != std::numeric_limits<unsigned>::max() && slot[m.residue] != a.cell) {
            return {};
        }
        slot[m.residue] = a.cell;
        auto& pipe = a.write ? producer : consumer;
        if (pipe != Pipe::Count && pipe != op.pipe) {
            return {};
        }
        pipe = op.pipe;
        modes[site] = m;
        writes[site] = a.write;
        payloads.push_back(site);
    }
    if (producer == Pipe::Count || consumer == Pipe::Count || producer == consumer ||
        std::set<unsigned>(slot.begin(), slot.end()).size() != p.cells.size() ||
        std::find(slot.begin(), slot.end(), std::numeric_limits<unsigned>::max()) != slot.end()) {
        return {};
    }
    for (auto first : nextPayload(c, {c.graph.entry})) {
        if (first != c.graph.exit && (!writes[first] || modes[first].residue != 0 || modes[first].previous)) {
            return {};
        }
    }
    for (auto site : payloads) {
        const auto next = nextPayload(c, c.graph.sites[site].successors);
        if (next.empty()) {
            return {};
        }
        for (auto target : next) {
            if (writes[site]) {
                if (target == c.graph.exit || writes[target] || !(modes[site] == modes[target])) {
                    return {};
                }
            } else if (target != c.graph.exit &&
                       (!writes[target] || modes[target].residue != (modes[site].residue + 1) % p.cells.size())) {
                return {};
            }
        }
    }
    std::vector<RecurringRequirement> requests;
    for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
        const auto previous = priorReader(p, c, cell);
        RecurringRequirement ready{cell, producer, consumer, {}, {}}, release{cell, consumer, producer, {}, {}};
        for (auto site : payloads) {
            const auto& a = p.operations[c.graph.operations[site]].accesses[0];
            if (a.cell != cell) {
                continue;
            }
            const auto endpoint = after(p, c, site, modes[site]);
            if (endpoint == NoAnalysisId) {
                return {};
            }
            if (a.write) {
                if (previous[site] != (modes[site].previous ? 2u : 1u)) {
                    return {};
                }
                ready.publications.push_back(endpoint);
                if (modes[site].previous) {
                    release.acquisitions.push_back(canonicalCommandCut(p, site));
                }
            } else {
                if (nextWriter(p, c, site, cell) != (modes[site].next ? 2u : 1u)) {
                    return {};
                }
                ready.acquisitions.push_back(canonicalCommandCut(p, site));
                if (modes[site].next) {
                    release.publications.push_back(endpoint);
                }
            }
        }
        for (auto* request : {&ready, &release}) {
            for (auto* cuts : {&request->publications, &request->acquisitions}) {
                std::sort(cuts->begin(), cuts->end());
                cuts->erase(std::unique(cuts->begin(), cuts->end()), cuts->end());
            }
            if (request->publications.empty() || request->acquisitions.empty()) {
                return {};
            }
            requests.push_back(std::move(*request));
        }
    }
    return requests;
}
bool Constructor::recurring(const std::vector<RecurringRequirement>& requests)
{
    std::set<Id> reserved;
    for (const auto& request : requests) {
        Id selected = NoAnalysisId;
        for (Id key = 0; key < frontier.keys().size(); ++key) {
            const auto& identity = frontier.keys()[key];
            if (identity.source == request.source && identity.observer == request.observer && !reserved.count(key)) {
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
                EndpointPurpose::Completion, id);
        }
        for (auto cut : request.acquisitions) {
            ledger.append(cut, {Command::Acquire, request.source, request.observer, number},
                EndpointPurpose::Completion, id);
        }
        result.channels.push_back({request.cell, number, request.source, request.observer,
                                   request.publications, request.acquisitions});
    }
    result.work.recurringChannels = result.channels.size();
    // These are symbolic role obligations, not assumed fresh-entry receipts.
    // finish() checks the entire selected ledger from the original root, through
    // every original entry/backedge/exit. Failure exports no executable program.
    return true;
}
} // namespace mlir::pto::oahs::selected
