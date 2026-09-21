// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"
#include <algorithm>
#include <chrono>
#include <deque>
#include <limits>
#include <tuple>

namespace mlir::pto::oahs::selected {
namespace {
Cut after(const Program& p, const Control& c, Id site, const OccurrenceMode& expected)
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
            if (!(occurrenceMode(p, at) == expected)) {
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
bool balanced(const Control&, const std::vector<Cut>&, const std::vector<Cut>&, bool = false);
std::vector<RecurringRequirement> qualifyCell(
    const Program& p, const Control& c, const ObservedLoop& loop, unsigned cell, Pipe intermediate = Pipe::Count)
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
        if (op.pipe == intermediate) continue;
        unsigned role = 0;
        for (const auto& a : op.accesses) if (a.cell == cell) role |= unsigned(a.read) | (unsigned(a.write) << 1);
        if (!role) continue;
        const auto m = occurrenceMode(p, site);
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
            const bool foundAccess = exits.count(at) || roles[at];
            if (foundAccess) {
                out.insert(at);
            } else if (members.count(at) || entries.count(at)) {
                const auto& next = c.graph.sites[at].successors;
                todo.insert(todo.end(), next.begin(), next.end());
            }
        }
        return out;
    };
    for (auto first : nextAccess(std::vector<Id>(entries.begin(), entries.end()))) {
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
        if (next.empty()) return {};
        for (auto target : next) {
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
            for (auto target : next) {
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
                if (std::none_of(next.begin(), next.end(), [&](Id target) {
                        return !exits.count(target) && roles[target] == 1;
                    }))
                    open.publications.push_back(after(p, c, site, modes[site]));
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

// A restricted three-engine storage cycle: producer write, in-place work,
// then a final reader. Projecting out the middle engine only derives the outer
// occurrence correspondence; the actual protocol includes both readiness hops.
// It grants no receipt until the selected endpoints are causally propagated.
std::vector<RecurringRequirement> qualifyPipelineCycle(
    const Program& p, const Control& c, const ObservedLoop& loop, unsigned cell,
    const std::vector<RecurringRequirement>& ordinary)
{
    if (loop.bodyEntry != NoAnalysisId || p.cells[cell].exclusive ||
        p.cells[cell].unknownRange || p.cells[cell].storage != Cell::Storage::CanonicalInterval)
        return {};
    Pipe middle = Pipe::Count;
    std::vector<Id> accesses;
    for (auto site : loop.sites) {
        const auto operation = c.graph.operations[site];
        if (!c.reachable[site] || operation == NoAnalysisId) continue;
        const auto& op = p.operations[operation];
        unsigned role = 0;
        for (const auto& a : op.accesses)
            if (a.cell == cell) role |= unsigned(a.read) | (unsigned(a.write) << 1);
        if (!role) continue;
        const auto mode = occurrenceMode(p, site);
        if (!mode.valid || mode.owner != loop.owner || mode.period != 1 ||
            p.target.synchronous[unsigned(op.pipe)]) return {};
        accesses.push_back(site);
        if (role == 3) {
            if (middle != Pipe::Count && middle != op.pipe) return {};
            middle = op.pipe;
        }
    }
    if (middle == Pipe::Count) return {};
    auto outer = qualifyCell(p, c, loop, cell, middle);
    if (outer.size() != 2) return {};
    auto ready = outer[0], result = outer[0];
    ready.observer = middle;
    ready.acquisitions.clear();
    result.source = middle;
    result.publications.clear();
    std::set<Id> assigned;
    // Each original occurrence must have a single straight corridor from the
    // producer's publication through the middle accesses to its final reader.
    // Guards, partial middle episodes, and coupled bank schedules decline.
    for (auto publication : outer[0].publications) {
        std::vector<Id> body;
        Cut reader = NoAnalysisId;
        for (auto acquisition : outer[0].acquisitions)
            if (occurrenceMode(p, publication) == occurrenceMode(p, acquisition) &&
                c.straight(publication, acquisition)) {
                if (reader != NoAnalysisId) return {};
                reader = acquisition;
            }
        if (reader == NoAnalysisId) return {};
        for (auto site : accesses)
            if (p.operations[c.graph.operations[site]].pipe == middle &&
                c.straight(publication, site) && c.straight(site, reader)) body.push_back(site);
        if (body.empty() || std::any_of(body.begin(), body.end(), [&](Id site) {
                return !(occurrenceMode(p, site) == occurrenceMode(p, publication));
            })) return {};
        std::sort(body.begin(), body.end(), [&](Id a, Id b) { return c.position[a] < c.position[b]; });
        const auto& first = p.operations[c.graph.operations[body.front()]];
        const auto& last = p.operations[c.graph.operations[body.back()]];
        auto has = [&](const Operation& op, bool write) {
            return std::any_of(op.accesses.begin(), op.accesses.end(), [&](const auto& a) {
                return a.cell == cell && (write ? a.write : a.read);
            });
        };
        // Both hops are independently required by the original storage uses.
        if (!has(first, false) || !has(last, true)) return {};
        const auto endpoint = after(p, c, body.back(), occurrenceMode(p, body.back()));
        if (endpoint == NoAnalysisId) return {};
        ready.acquisitions.push_back(canonicalCommandCut(p, body.front()));
        result.publications.push_back(endpoint);
        assigned.insert(body.begin(), body.end());
    }
    for (auto site : accesses)
        if (p.operations[c.graph.operations[site]].pipe == middle && !assigned.count(site)) return {};
    for (auto* request : {&ready, &result}) {
        for (auto* cuts : {&request->publications, &request->acquisitions}) std::sort(cuts->begin(), cuts->end());
        if (!balanced(c, request->publications, request->acquisitions)) return {};
    }
    // Independent two-engine operands retain their early readiness. Their
    // private return is unnecessary only when this cycle's producer starts
    // earlier and its middle publication follows every operand reader in the
    // SAME occurrence. The final-reader return then transports both completion
    // and readiness-consumption evidence before the operand's next overwrite.
    std::vector<RecurringRequirement> supported;
    for (const auto& operand : ordinary) {
        if (operand.owner != loop.owner) continue;
        if (!operand.qualifiedCycle || operand.period != 1 || operand.storageRelease ||
            operand.source != ready.source || operand.observer != middle) continue;
        const auto reverse = std::find_if(ordinary.begin(), ordinary.end(), [&](const auto& r) {
            return r.owner == operand.owner && r.period == 1 && r.qualifiedCycle && r.storageRelease &&
                r.source == middle && r.observer == ready.source && r.cells == operand.cells;
        });
        if (reverse == ordinary.end()) return {};
        for (auto cut : operand.publications)
            if (std::none_of(ready.publications.begin(), ready.publications.end(), [&](Cut earlier) {
                    return occurrenceMode(p, earlier) == occurrenceMode(p, cut) && c.straight(earlier, cut);
                })) return {};
        for (auto cut : reverse->publications)
            if (std::none_of(result.publications.begin(), result.publications.end(), [&](Cut later) {
                    return occurrenceMode(p, cut) == occurrenceMode(p, later) && c.straight(cut, later);
                })) return {};
        supported.push_back(operand);
    }
    // Keep this admission local: do not replace unrelated owner protocols.
    for (const auto& old : ordinary) {
        if (old.owner != loop.owner) continue;
        if (std::none_of(supported.begin(), supported.end(), [&](const auto& operand) {
                return old.cells == operand.cells &&
                    ((old.source == operand.source && old.observer == operand.observer) ||
                     (old.source == operand.observer && old.observer == operand.source));
            })) return {};
    }
    std::vector<RecurringRequirement> out{ready, result, outer[1]};
    out.insert(out.end(), supported.begin(), supported.end());
    return out;
}

// Nearest storage role over the original control graph. Callers provide
// physical accesses or qualified region boundaries, never selected credit.
// Reader=1, writer=2, invocation boundary=4: constant-height propagation.
std::vector<unsigned> nearestRoles(const Control& c, const std::vector<unsigned>& roles, bool backward,
                                   uint64_t* visits = nullptr)
{
    const auto n = c.graph.sites.size();
    if (visits) *visits += n;
    std::vector<unsigned> facts(n);
    std::deque<Cut> todo;
    std::vector<bool> queued(n);
    auto add = [&](Cut site, unsigned bits) {
        const auto joined = facts[site] | bits;
        if (facts[site] == joined) return;
        facts[site] = joined;
        if (!queued[site]) { queued[site] = true; todo.push_back(site); }
    };
    auto edges = [&](Cut site) -> const std::vector<Id>& {
        return backward ? c.predecessors[site] : c.graph.sites[site].successors;
    };
    for (Cut site = 0; site < n; ++site)
        if (roles[site]) for (auto next : edges(site)) add(next, roles[site]);
    add(backward ? c.graph.exit : c.graph.entry, 4);
    while (!todo.empty()) {
        const auto site = todo.front(); todo.pop_front(); queued[site] = false;
        if (visits) ++*visits;
        if (roles[site]) continue;
        for (auto next : edges(site)) add(next, facts[site]);
    }
    return facts;
}

// Derive complete guarded producer/reader episodes before assigning physical
// event keys. A skipped branch preserves the bank release token, so the next
// participating writer is paired with the preceding participating reader.
std::vector<RecurringRequirement> qualifyGuardedBanks(const Program& p, const Control& c)
{
    std::vector<RecurringRequirement> out;
    if (!p.observed || !c.complete || !c.graph.legalCuts[c.graph.entry] || !c.graph.legalCuts[c.graph.exit]) return out;
    const auto n = c.graph.sites.size();
    std::vector<std::vector<std::pair<Cut, unsigned>>> incidences(p.cells.size());
    for (Cut site = 0; site < n; ++site) {
        const auto operation = c.graph.operations[site];
        if (!c.reachable[site] || operation == NoAnalysisId) continue;
        for (const auto& access : p.operations[operation].accesses)
            incidences[access.cell].push_back({site, unsigned(access.read) | (unsigned(access.write) << 1)});
    }
    for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
        const auto& physical = p.cells[cell];
        if (physical.storage != Cell::Storage::CanonicalInterval || physical.unknownRange || physical.exclusive)
            continue;
        std::vector<unsigned> roles(n);
        for (auto [site, role] : incidences[cell]) roles[site] |= role;
        Pipe writer = Pipe::Count, reader = Pipe::Count;
        bool admitted = true, cyclic = false;
        for (auto [site, ignored] : incidences[cell]) {
            (void)ignored;
            const auto role = roles[site];
            const auto pipe = p.operations[c.graph.operations[site]].pipe;
            if (role == 3 || !role || p.target.synchronous[unsigned(pipe)]) { admitted = false; break; }
            auto& engine = role == 2 ? writer : reader;
            if (engine != Pipe::Count && engine != pipe) { admitted = false; break; }
            engine = pipe;
            cyclic |= c.components[c.component[site]].cyclic;
        }
        if (!admitted || !cyclic || writer == Pipe::Count || reader == Pipe::Count || writer == reader) continue;
        const auto previous = nearestRoles(c, roles, false), next = nearestRoles(c, roles, true);
        RecurringRequirement ready, release;
        ready.qualifiedCycle = release.qualifiedCycle = true;
        release.storageRelease = true;
        ready.cell = release.cell = cell;
        ready.cells = release.cells = {cell};
        ready.source = writer; ready.observer = reader;
        release.source = reader; release.observer = writer;
        // Invocation-owned, not a fabricated periodic-loop qualification.
        ready.owner = release.owner = NoAnalysisId;
        release.publications = {c.canonicalCut[c.graph.entry]};
        release.acquisitions = {c.canonicalCut[c.graph.exit]};
        for (Cut site = 0; site < n && admitted; ++site) {
            const auto role = roles[site];
            if (!role) continue;
            const auto after = c.after(site);
            if (!c.graph.legalCuts[site] || after == NoAnalysisId) { admitted = false; break; }
            if (role == 2) {
                // One writer per phase; every write must reach a reader before
                // another write or exit. Consecutive writes remain ordinary.
                admitted = previous[site] && !(previous[site] & ~5u) && next[site] == 1;
                release.acquisitions.push_back(c.canonicalCut[site]);
                ready.publications.push_back(c.canonicalCut[after]);
            } else {
                // Multiple readers on one engine form a conjunction. First and
                // last endpoints must be unambiguous at their ORIGINAL cuts.
                admitted = (previous[site] == 1 || previous[site] == 2) &&
                    (next[site] == 1 || (next[site] && !(next[site] & ~6u)));
                if (previous[site] == 2) ready.acquisitions.push_back(c.canonicalCut[site]);
                if (!(next[site] & 1)) release.publications.push_back(c.canonicalCut[after]);
            }
        }
        if (!admitted) continue;
        for (auto* r : {&ready, &release})
            for (auto* cuts : {&r->publications, &r->acquisitions}) {
                std::sort(cuts->begin(), cuts->end());
                cuts->erase(std::unique(cuts->begin(), cuts->end()), cuts->end());
            }
        // Shared emitted words must participate exactly like the analytical
        // endpoints, including zero visits, backedges, and invocation exit.
        if (!balanced(c, ready.publications, ready.acquisitions, true) ||
            !balanced(c, release.publications, release.acquisitions, true)) continue;
        out.push_back(std::move(ready));
        out.push_back(std::move(release));
    }
    return out;
}

// Reader-only children can retain one storage generation across their visits
// and siblings. The first qualified reader acquires readiness; the last child
// exit releases the generation before subsequent unrelated reader-pipe work.
// Select both directions so actual readiness supports return and rearming.
std::vector<RecurringRequirement> qualifyReaderRegionCycles(
    const Program& p, const Control& c, const std::vector<RecurringRequirement>& selected,
    bool shareReturns, SelectedWork* work)
{
    std::vector<RecurringRequirement> out;
    if (!p.observed) return out;
    struct Candidate {
        RecurringRequirement ready, release;
        std::set<Cut> writes;
        bool retained;
    };
    std::vector<Candidate> candidates;
    std::set<unsigned> owned;
    for (const auto& request : selected)
        owned.insert(request.cells.begin(), request.cells.end());
    std::vector<std::vector<Cut>> accesses(p.cells.size());
    for (Cut site = 0; site < c.graph.sites.size(); ++site) {
        const auto operation = c.graph.operations[site];
        if (!c.reachable[site] || operation == NoAnalysisId) continue;
        for (const auto& access : p.operations[operation].accesses)
            accesses[access.cell].push_back(site);
    }
    for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
        const auto& physical = p.cells[cell];
        if (owned.count(cell) || physical.storage != Cell::Storage::CanonicalInterval ||
            physical.unknownRange || physical.exclusive) continue;
        Pipe writer = Pipe::Count, reader = Pipe::Count;
        std::set<Cut> writes, reads;
        bool admitted = true;
        for (auto site : accesses[cell]) {
            const auto& operation = p.operations[c.graph.operations[site]];
            for (const auto& access : operation.accesses) {
                if (access.cell != cell) continue;
                if (access.read == access.write) { admitted = false; break; }
                auto& pipe = access.write ? writer : reader;
                if (pipe != Pipe::Count && pipe != operation.pipe) { admitted = false; break; }
                pipe = operation.pipe;
                (access.write ? writes : reads).insert(site);
            }
            if (!admitted) break;
        }
        if (!admitted || writes.empty() || reads.empty() || writer == reader ||
            p.target.synchronous[unsigned(writer)] || p.target.synchronous[unsigned(reader)]) continue;
        if (std::any_of(writes.begin(), writes.end(), [&](Cut site) {
                return !c.components[c.component[site]].cyclic;
            })) continue;
        struct ReaderRegion { Cut entry, acquisition, exit, publication; };
        std::vector<ReaderRegion> regions;
        std::set<Cut> covered;
        for (const auto& loop : p.observed->loops) {
            if (!loop.atLeastOnce || loop.bodyEntry == NoAnalysisId ||
                !loop.entries.empty() || !loop.exits.empty()) continue;
            const std::set<Cut> members(loop.sites.begin(), loop.sites.end());
            // Use leaf reader regions; do not turn an enclosing multi-generation
            // region into one completion episode.
            if (std::any_of(p.observed->loops.begin(), p.observed->loops.end(), [&](const auto& child) {
                    return child.owner != loop.owner && members.count(child.entry);
                })) continue;
            std::vector<Cut> local;
            for (auto site : reads) if (members.count(site)) local.push_back(site);
            if (local.empty() || std::any_of(writes.begin(), writes.end(),
                    [&](Cut site) { return members.count(site); })) continue;
            const auto facts = std::find_if(c.loopEntries.begin(), c.loopEntries.end(),
                [&](const auto& entry) { return entry.entry == loop.entry; });
            if (facts == c.loopEntries.end() || facts->issuedPipes.count(writer)) continue;
            Cut acquisition;
            const auto& first = facts->firstConsumers[unsigned(reader)];
            if (!first.empty() && std::all_of(first.begin(), first.end(), [&](Cut site) {
                    return reads.count(site);
                })) acquisition = c.canonicalCut[loop.entry];
            else {
                const auto input = std::find_if(facts->firstInputConsumers.begin(), facts->firstInputConsumers.end(),
                    [&](Cut site) { return reads.count(site); });
                if (input == facts->firstInputConsumers.end()) continue;
                acquisition = c.canonicalCut[*input];
            }
            if (!c.graph.legalCuts[loop.exit]) { admitted = false; break; }
            // A final-visit observation can expose the last physical reader
            // before unrelated trailing work. Query the same cell/control
            // view; no lexical boundary or observation supplies completion.
            Cut publication = c.canonicalCut[loop.exit];
            std::vector<std::pair<Cut, Cut>> lastCandidates;
            for (auto site : local) {
                const auto afterRead = c.after(site);
                if (afterRead == NoAnalysisId || !members.count(afterRead)) continue;
                const auto observation = p.observed->sites[afterRead].observation;
                if (observation == NoAnalysisId) continue;
                const auto& atoms = p.observed->observations[observation].atoms;
                if (std::any_of(atoms.begin(), atoms.end(), [&](const auto& atom) {
                        return atom.kind == ObservationAtom::LoopHasNext && atom.owner == loop.owner &&
                               atom.parameter == loop.lastVisitDistance && atom.value == 0;
                    })) lastCandidates.emplace_back(site, c.canonicalCut[afterRead]);
            }
            if (!lastCandidates.empty()) {
                std::vector<unsigned> lastRoles(c.graph.sites.size());
                for (auto site : local) lastRoles[site] = 1;
                lastRoles[loop.exit] = 2;
                const auto following = nearestRoles(c, lastRoles, true);
                std::set<Cut> lastCuts;
                for (auto [site, cut] : lastCandidates) if (following[site] == 2) lastCuts.insert(cut);
                if (lastCuts.size() == 1 && balanced(c, {c.canonicalCut[loop.entry]}, {*lastCuts.begin()}))
                    publication = *lastCuts.begin();
            }
            regions.push_back({loop.entry, acquisition, loop.exit, publication});
            covered.insert(local.begin(), local.end());
        }
        if (!admitted || covered != reads) continue;
        // Compose reader-only children by physical generation. A sibling exit
        // is a use boundary, not automatically a release; the following child
        // retains readiness unless an intervening writer starts a new phase.
        std::vector<unsigned> beforeRoles(c.graph.sites.size()), afterRoles(beforeRoles.size());
        for (auto site : writes) beforeRoles[site] = afterRoles[site] = 2;
        for (const auto& region : regions) {
            beforeRoles[region.exit] |= 1;
            afterRoles[region.entry] |= 1;
        }
        const auto previous = nearestRoles(c, beforeRoles, false);
        const auto next = nearestRoles(c, afterRoles, true);
        std::set<Cut> firstConsumers, publications{c.canonicalCut[c.graph.entry]};
        for (const auto& region : regions) {
            // Native sibling exit/entry anchors can be the same original cut.
            // That boundary itself is the neighboring role in this view.
            const auto predecessor = beforeRoles[region.entry] ? beforeRoles[region.entry] : previous[region.entry];
            if (predecessor == 2) firstConsumers.insert(region.acquisition);
            else if (predecessor != 1) { admitted = false; break; }
            const auto successor = afterRoles[region.exit] ? afterRoles[region.exit] : next[region.exit];
            if (successor && !(successor & ~6u)) publications.insert(region.publication);
            else if (successor != 1) { admitted = false; break; }
        }
        const bool retainsAcrossChildren = firstConsumers.size() < regions.size();
        // Mixed first/retained or last/non-last paths need an original
        // participation witness; do not introduce an event-derived guard.
        if (!admitted) continue;
        RecurringRequirement returned;
        returned.cell = cell; returned.cells = {cell};
        returned.source = reader; returned.observer = writer;
        returned.qualifiedCycle = returned.storageRelease = true;
        returned.publications.assign(publications.begin(), publications.end());
        std::set<Cut> acquisitions{c.canonicalCut[c.graph.exit]};
        for (auto site : writes) acquisitions.insert(c.canonicalCut[site]);
        returned.acquisitions.assign(acquisitions.begin(), acquisitions.end());
        // Exact original participation, including skipped parent episodes,
        // first entry, backedges, and tail reuse. No fresh-scope reset.
        RecurringRequirement ready;
        ready.cell = cell; ready.cells = {cell};
        ready.source = writer; ready.observer = reader; ready.qualifiedCycle = true;
        ready.acquisitions.assign(firstConsumers.begin(), firstConsumers.end());
        for (auto site : writes) {
            const auto after = c.after(site);
            if (after == NoAnalysisId) { admitted = false; break; }
            ready.publications.push_back(c.canonicalCut[after]);
        }
        std::sort(ready.publications.begin(), ready.publications.end());
        ready.publications.erase(std::unique(ready.publications.begin(), ready.publications.end()),
                                 ready.publications.end());
        if (admitted && balanced(c, ready.publications, ready.acquisitions, true) &&
            balanced(c, returned.publications, returned.acquisitions, true)) {
            candidates.push_back({std::move(ready), std::move(returned), std::move(writes), retainsAcrossChildren});
        }
    }
    // A retained child can remove a producer fence. Do not let an uncovered
    // write move the remaining fence after the next generation. Index complete
    // candidates first; multi-input admission needs a closed producer cohort,
    // not a per-cell assertion that the other cycles will appear later.
    std::map<Pipe, std::set<unsigned>> written;
    std::set<Pipe> exclusive;
    for (const auto& operation : p.operations) {
        for (const auto& access : operation.accesses) {
            if (access.write) written[operation.pipe].insert(access.cell);
            if (p.cells[access.cell].exclusive) exclusive.insert(operation.pipe);
        }
    }
    std::map<unsigned, Id> byCell;
    for (Id i = 0; i < candidates.size(); ++i) byCell.emplace(candidates[i].ready.cell, i);
    std::map<Pipe, bool> closedCohorts;
    for (const auto& candidate : candidates) {
        const auto producer = candidate.ready.source;
        const auto& cells = written[producer];
        if (candidate.retained && cells.size() > 1 && !closedCohorts.count(producer)) {
            bool closed = true;
            Cut anchor = NoAnalysisId;
            for (auto cell : cells) {
                const auto found = byCell.find(cell);
                if (found == byCell.end()) { closed = false; break; }
                const auto& support = candidates[found->second];
                // First extension: one straight producer corridor, one writer
                // occurrence per cell, and one reader engine. Reloads and
                // branch-dependent producer sequences retain the old fallback.
                if (support.ready.source != producer || support.ready.observer != candidate.ready.observer ||
                    support.writes.size() != 1) { closed = false; break; }
                const auto site = *support.writes.begin();
                if (anchor == NoAnalysisId) anchor = site;
                else if (!c.straight(anchor, site) && !c.straight(site, anchor)) { closed = false; break; }
            }
            closedCohorts.emplace(producer, closed);
        }
    }
    std::vector<bool> admitted(candidates.size()), shared(candidates.size());
    for (Id i = 0; i < candidates.size(); ++i) {
        auto& candidate = candidates[i];
        const auto producer = candidate.ready.source;
        if (candidate.retained && exclusive.count(producer)) continue;
        if (candidate.retained && written[producer].size() > 1) {
            if (!closedCohorts.at(producer)) continue;
            // This is a conditional support assertion. The single staged
            // mandatory solve below must discharge it before any reservation.
            candidate.ready.repairFreeProducers.insert(producer);
        }
        admitted[i] = true;
    }
    // Keep each readiness boundary. A later required reader return can also
    // cover an earlier reader phase, but only if its existing acquisition is
    // before the other's overwrite deadline. Never move the supporting wait.
    // This selects a composed lifetime interface BEFORE allocating any keys;
    // it is not an omission trial on a completed synchronization plan.
    const auto entry = c.canonicalCut[c.graph.entry], exit = c.canonicalCut[c.graph.exit];
    auto bodyPublication = [&](const Candidate& candidate) -> Cut {
        const auto& release = candidate.release;
        if (candidate.writes.size() != 1 || release.publications.size() != 2 ||
            release.acquisitions.size() != 2 ||
            !std::binary_search(release.publications.begin(), release.publications.end(), entry) ||
            !std::binary_search(release.acquisitions.begin(), release.acquisitions.end(), exit))
            return NoAnalysisId;
        const auto cut = release.publications[release.publications.front() == entry ? 1 : 0];
        return c.wordOccurrences[cut].size() == 1 ? cut : NoAnalysisId;
    };
    if (shareReturns) for (Id y = 0; y < candidates.size(); ++y) {
        if (!admitted[y]) continue;
        auto& victim = candidates[y];
        const auto producer = victim.ready.source;
        if (!closedCohorts.count(producer) || !closedCohorts.at(producer) || exclusive.count(producer)) continue;
        const auto yRelease = bodyPublication(victim);
        if (yRelease == NoAnalysisId) continue;
        for (Id x = 0; x < candidates.size(); ++x) {
            if (x == y || !admitted[x] || shared[x]) continue;
            auto& support = candidates[x];
            if (support.ready.source != producer || support.ready.observer != victim.ready.observer) continue;
            const auto xRelease = bodyPublication(support);
            if (xRelease == NoAnalysisId || xRelease == yRelease ||
                support.writes == victim.writes ||
                !c.straight(*support.writes.begin(), *victim.writes.begin())) continue;
            if (work) ++work->returnSharingQueries;
            // Every path to X's release must cross Y's final-reader boundary
            // in this generation. A writer or invocation boundary kills that
            // correspondence. The existing nearest-role view handles empty
            // reader children and varying visit counts without an unrolling.
            std::vector<unsigned> roles(c.graph.sites.size());
            for (const auto& member : candidates) if (member.ready.source == producer)
                for (auto site : member.writes) roles[site] = 2;
            roles[yRelease] = 1;
            const auto previous = nearestRoles(c, roles, false,
                work ? &work->returnSharingSiteVisits : nullptr);
            if (previous[xRelease] != 1) continue;
            support.release.cells.insert(support.release.cells.end(),
                victim.release.cells.begin(), victim.release.cells.end());
            std::sort(support.release.cells.begin(), support.release.cells.end());
            support.release.sharedReturns += 1 + victim.release.sharedReturns;
            // The actual staged words must discharge ALL producer demands,
            // including zero-reader WAW, plus event consumption/republication.
            support.ready.repairFreeProducers.insert(producer);
            shared[y] = true;
            break;
        }
    }
    for (Id i = 0; i < candidates.size(); ++i) {
        if (!admitted[i]) continue;
        auto& candidate = candidates[i];
        out.push_back(std::move(candidate.ready));
        if (!shared[i]) out.push_back(std::move(candidate.release));
    }
    return out;
}

// Compose a refined child reader episode with its original enclosing loop.
// This is deliberately narrower than nested residue refinement: it retains the
// child's existing finite quotient and derives one complete producer/readers
// cycle for an exact physical cell. The selected protocol is available before
// ordinary construction decides whether a same-pipe overwrite needs a fence.
std::vector<RecurringRequirement> qualifyEnclosingCell(
    const Program& p, const Control& c, const ObservedLoop& loop, unsigned cell)
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
        unsigned role = 0;
        for (const auto& access : op.accesses)
            if (access.cell == cell) role |= unsigned(access.read) | (unsigned(access.write) << 1);
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

    auto atomsEqual = [&](Cut a, Cut b) {
        const auto oa = p.observed->sites[a].observation;
        const auto ob = p.observed->sites[b].observation;
        if (oa == NoAnalysisId || ob == NoAnalysisId) return false;
        auto left = p.observed->observations[oa].atoms;
        auto right = p.observed->observations[ob].atoms;
        auto less = [](const auto& x, const auto& y) {
            return std::tie(x.owner, x.kind, x.parameter, x.value) <
                   std::tie(y.owner, y.kind, y.parameter, y.value);
        };
        std::sort(left.begin(), left.end(), less);
        std::sort(right.begin(), right.end(), less);
        return left.size() == right.size() &&
            std::equal(left.begin(), left.end(), right.begin(), [&](const auto& x, const auto& y) {
                return !less(x, y) && !less(y, x);
            });
    };
    auto afterObservation = [&](Id source) {
        std::set<Cut> cuts;
        auto todo = c.graph.sites[source].successors;
        std::vector<bool> seen(c.graph.sites.size());
        while (!todo.empty()) {
            const auto at = todo.back(); todo.pop_back();
            if (seen[at]) continue;
            seen[at] = true;
            if (c.graph.legalCuts[at]) {
                if (!atomsEqual(source, at)) return Cut(NoAnalysisId);
                cuts.insert(canonicalCommandCut(p, at));
                continue;
            }
            if (c.graph.operations[at] != NoAnalysisId || at == c.graph.exit)
                return Cut(NoAnalysisId);
            const auto& next = c.graph.sites[at].successors;
            todo.insert(todo.end(), next.begin(), next.end());
        }
        return cuts.size() == 1 ? *cuts.begin() : Cut(NoAnalysisId);
    };
    struct ChildVisit { Id owner = NoAnalysisId; bool first = false, last = false, valid = false; };
    auto childVisit = [&](Id site) {
        ChildVisit out;
        const auto observation = p.observed->sites[site].observation;
        if (observation == NoAnalysisId) return out;
        const auto& atoms = p.observed->observations[observation].atoms;
        for (const auto& before : atoms) {
            if (before.kind != ObservationAtom::LoopHasPrevious || before.parameter != 1 ||
                before.owner == loop.owner) continue;
            for (const auto& after : atoms) {
                if (after.kind != ObservationAtom::LoopHasNext || after.parameter != 1 ||
                    after.owner != before.owner) continue;
                if (out.valid && out.owner != before.owner) return ChildVisit{};
                out = {before.owner, before.value == 0, after.value == 0, true};
            }
        }
        return out;
    };

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
    Id childOwner = NoAnalysisId;
    for (auto site : members) {
        if (!roles[site]) continue;
        const auto endpoint = afterObservation(site);
        if (endpoint == NoAnalysisId) return {};
        if (roles[site] == 2) {
            ready.publications.push_back(endpoint);
            release.acquisitions.push_back(canonicalCommandCut(p, site));
            continue;
        }
        const auto visit = childVisit(site);
        if (!visit.valid || visit.owner == loop.owner ||
            (childOwner != NoAnalysisId && childOwner != visit.owner)) return {};
        childOwner = visit.owner;
        if (visit.first) ready.acquisitions.push_back(canonicalCommandCut(p, site));
        if (visit.last) release.publications.push_back(endpoint);
    }
    if (childOwner == NoAnalysisId) return {};
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
              const std::vector<Cut>& acquisitions, bool adjacent)
{
    std::set<Cut> publish(publications.begin(), publications.end());
    std::set<Cut> acquire(acquisitions.begin(), acquisitions.end());
    for (auto cut : publish) if (!adjacent && acquire.count(cut)) return false;
    std::vector<uint8_t> incoming(c.graph.sites.size());
    std::deque<Id> queue;
    std::vector<bool> queued(c.graph.sites.size());
    bool invalid = false;
    uint8_t exitState = 0;
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
        if (site == c.graph.exit) exitState = state;
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
    return !invalid && exitState == 1;
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
    const Program& p, const Control& c, const RequirementFrontiers& frontiers,
    bool allowGuardedEpisodes, bool movingFrontiers, bool shareReaderReturns, SelectedWork* work)
{
    std::vector<RecurringRequirement> requests;
    if (!p.observed) return requests;
    if (allowGuardedEpisodes && p.alternatingSlots) {
        // Complete the two-slot read/write ownership cycle before ordinary
        // repair. The previous episode's return covers the older same-slot
        // writer; this episode's return may therefore follow its read. The
        // acknowledgment carries that read's completion before the next write.
        // One logical pair of directions serves the FIFO, not one per cell.
        const auto& slots = *p.alternatingSlots;
        RecurringRequirement returned, acknowledged;
        returned.cell = acknowledged.cell = slots.cells.front();
        returned.cells = acknowledged.cells = slots.cells;
        returned.source = acknowledged.observer = slots.writer;
        returned.observer = acknowledged.source = slots.reader;
        returned.qualifiedCycle = acknowledged.qualifiedCycle = true;
        // Preserve WAIT return -> SET acknowledgment -> WAIT acknowledgment
        // within each after-read word (not the guarded publication-first form).
        returned.period = acknowledged.period = 1;
        std::set<Cut> reads, afterReads;
        for (auto site : slots.reads) {
            auto next = c.after(site);
            if (next == NoAnalysisId) return {};
            reads.insert(c.canonicalCut[site]);
            afterReads.insert(c.canonicalCut[next]);
        }
        returned.publications.assign(reads.begin(), reads.end());
        returned.acquisitions.assign(afterReads.begin(), afterReads.end());
        acknowledged.publications = acknowledged.acquisitions = returned.acquisitions;
        if (!balanced(c, returned.publications, returned.acquisitions)) return {};
        return {returned, acknowledged};
    }
    auto append = [&](RecurringRequirement request) {
        // Several conservative storage witnesses can name the same physical
        // role. One actual prefix serves their conjunction.
        const auto duplicate = std::find_if(requests.begin(), requests.end(), [&](const auto& old) {
            return old.source == request.source && old.observer == request.observer &&
                old.publications == request.publications && old.acquisitions == request.acquisitions;
        });
        if (duplicate == requests.end()) requests.push_back(std::move(request));
        else {
            duplicate->qualifiedCycle &= request.qualifiedCycle;
            duplicate->repairFreeProducers.insert(request.repairFreeProducers.begin(), request.repairFreeProducers.end());
            duplicate->sharedReturns += request.sharedReturns;
            duplicate->cells.insert(duplicate->cells.end(), request.cells.begin(), request.cells.end());
            std::sort(duplicate->cells.begin(), duplicate->cells.end());
            duplicate->cells.erase(std::unique(duplicate->cells.begin(), duplicate->cells.end()),
                                   duplicate->cells.end());
        }
    };
    for (const auto& loop : p.observed->loops) {
        for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
            auto local = loop.bodyEntry == NoAnalysisId
                ? qualifyCell(p, c, loop, cell)
                : qualifyEnclosingCell(p, c, loop, cell);
            for (auto& request : local) append(std::move(request));
        }
    }
    std::set<Id> pipelineOwners;
    // Select a complete supported pipeline interface before binding keys or
    // repairing same-engine remainders. No changed-plan omission trial is used.
    if (allowGuardedEpisodes) {
        for (const auto& loop : p.observed->loops) {
            for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
                auto pipeline = qualifyPipelineCycle(p, c, loop, cell, requests);
                if (pipeline.empty()) continue;
                pipelineOwners.insert(loop.owner);
                requests.erase(std::remove_if(requests.begin(), requests.end(), [&](const auto& r) {
                    return r.owner == loop.owner;
                }), requests.end());
                for (auto& request : pipeline) append(std::move(request));
                // The selected cycle's source prefix covers the other cells
                // touched by these same operations; don't allocate per-cell rings.
                break;
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
    relationshipRequests.erase(std::remove_if(relationshipRequests.begin(), relationshipRequests.end(),
        [&](const auto& r) { return pipelineOwners.count(r.owner); }), relationshipRequests.end());
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
    using EndpointOccurrence = std::vector<std::tuple<Id, unsigned, uint64_t, uint64_t>>;
    auto indexed = [&](const std::vector<Cut>& cuts) {
        std::map<EndpointOccurrence, Cut> out;
        for (auto cut : cuts) {
            if (cut >= p.observed->sites.size()) return std::map<EndpointOccurrence, Cut>{};
            const auto observation = p.observed->sites[cut].observation;
            if (observation == NoAnalysisId || observation >= p.observed->observations.size() ||
                !p.observed->observations[observation].available)
                return std::map<EndpointOccurrence, Cut>{};
            EndpointOccurrence key;
            for (const auto& atom : p.observed->observations[observation].atoms)
                key.emplace_back(atom.owner, unsigned(atom.kind), atom.parameter, atom.value);
            std::sort(key.begin(), key.end());
            if (!out.emplace(std::move(key), cut).second)
                return std::map<EndpointOccurrence, Cut>{};
        }
        return out;
    };
    auto combine = [&](const std::vector<Cut>& a, const std::vector<Cut>& b, bool later,
                       std::vector<Cut>& out) {
        if (!movingFrontiers) {
            if (a != b) return false;
            out = a;
            return true;
        }
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
    // Unrefined physical episodes retain their release across child exits.
    // Sharing a final reader is optional: a bank with a distinct reader
    // frontier still needs its own readiness and previous-use return.
    if (requests.empty() && allowGuardedEpisodes) {
        // Compose exact guarded banks by their shared first-reader episode
        // before assigning keys. Distinct readiness publications stay at their
        // early source prefixes; a common reader completion supplies one
        // release token acquired at the earliest corresponding reuse deadline.
        auto guarded = qualifyGuardedBanks(p, c);
        using Episode = std::tuple<Pipe, Pipe, std::vector<Cut>>;
        std::map<Episode, std::vector<Id>> episodes;
        for (Id i = 0; i + 1 < guarded.size(); i += 2) {
            const auto& ready = guarded[i];
            const auto& release = guarded[i + 1];
            if (release.source != ready.observer || release.observer != ready.source ||
                ready.cell != release.cell) continue;
            episodes[{ready.source, ready.observer, ready.acquisitions}].push_back(i);
        }
        // Admit independent banks as one logical cohort. Do not append a
        // per-cell population to an already composed common-reader protocol,
        // or mix unrelated producer/reader directions into this specialization.
        const bool independent = episodes.size() > 1 &&
            std::all_of(episodes.begin(), episodes.end(), [&](const auto& item) {
                return item.second.size() == 1 &&
                    std::get<0>(item.first) == std::get<0>(episodes.begin()->first) &&
                    std::get<1>(item.first) == std::get<1>(episodes.begin()->first);
            });
        for (auto& [episode, members] : episodes) {
            (void)episode;
            if (members.size() == 1) {
                if (!independent) continue;
                // The role/balance proofs above already admit this episode.
                // Do not force ordinary common-cut repair merely because no
                // other storage cell has the exact same reader frontier.
                // Capacity is checked for the complete logical proposal before
                // any endpoint is committed by recurring().
                const auto index = members.front();
                append(std::move(guarded[index]));
                append(std::move(guarded[index + 1]));
                continue;
            }
            std::optional<RecurringRequirement> release;
            std::vector<RecurringRequirement> readiness;
            for (auto index : members) {
                readiness.push_back(std::move(guarded[index]));
                auto next = std::move(guarded[index + 1]);
                if (!release) {
                    release = std::move(next);
                    continue;
                }
                if (release->publications != next.publications ||
                    release->acquisitions.size() != next.acquisitions.size()) {
                    release.reset();
                    break;
                }
                std::vector<Cut> acquisitions;
                for (Id i = 0; i < release->acquisitions.size(); ++i) {
                    const auto a = release->acquisitions[i];
                    const auto b = next.acquisitions[i];
                    if (a == b || c.straight(a, b)) acquisitions.push_back(a);
                    else if (c.straight(b, a)) acquisitions.push_back(b);
                    else {
                        release.reset();
                        break;
                    }
                }
                if (!release) break;
                release->acquisitions = std::move(acquisitions);
                release->cells.insert(release->cells.end(), next.cells.begin(), next.cells.end());
                std::sort(release->cells.begin(), release->cells.end());
                release->cells.erase(std::unique(release->cells.begin(), release->cells.end()),
                                     release->cells.end());
                release->cell = release->cells.front();
            }
            if (!release ||
                !balanced(c, release->publications, release->acquisitions, true)) continue;
            for (auto& ready : readiness) append(std::move(ready));
            append(std::move(*release));
        }
    }
    if (allowGuardedEpisodes)
        for (auto& request : qualifyReaderRegionCycles(p, c, requests, shareReaderReturns, work))
            append(std::move(request));
    // Distinct release publications retain their own prefixes. Matching bank
    // phases and a balanced merged token stream do not justify waiting for a
    // later reader at an earlier overwrite. Identical publication frontiers
    // can still share through the common-prefix rule above.
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
            // Recurring qualification is an optional construction shortcut.
            // No endpoint has been committed yet, so decline the whole proposal
            // and let ordinary demand-driven construction reuse the key pool.
            ++result.work.rejectedResourceProposals;
            return true;
        }
        proposedKeys.insert(selected);
        keys.push_back(selected);
    }
    std::vector<bool> retained(requests.size(), true);
    const bool guarded = std::all_of(requests.begin(), requests.end(), [](const auto& request) {
        return request.owner == NoAnalysisId && request.period == 0;
    });
    struct PendingEndpoint { Cut cut; Command command; Id request; };
    // Materialize the exact canonical words before checking. In particular a
    // publication-first convention can remove consumption credit carried by a
    // return, so it must never be applied only after admission.
    auto materialize = [&]() {
        std::vector<PendingEndpoint> endpoints;
        for (Id i = 0; i < requests.size(); ++i) {
            if (!retained[i]) continue;
            const auto& r = requests[i];
            const auto number = frontier.keys()[keys[i]].key;
            for (auto cut : r.publications)
                endpoints.push_back({control.canonicalCut[cut], {Command::Publish, r.source, r.observer, number}, i});
            for (auto cut : r.acquisitions)
                endpoints.push_back({control.canonicalCut[cut], {Command::Acquire, r.source, r.observer, number}, i});
        }
        if (guarded) std::stable_partition(endpoints.begin(), endpoints.end(), [](const auto& endpoint) {
            return endpoint.command.kind == Command::Publish;
        });
        return endpoints;
    };
    auto candidate = [&](const std::vector<PendingEndpoint>& endpoints) {
        auto commands = ledger.commands();
        for (const auto& endpoint : endpoints)
            for (auto site : control.wordOccurrences[endpoint.cut])
                commands[site].push_back(endpoint.command);
        return commands;
    };
    auto endpoints = materialize();
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
    // Missing payload completion remains pending; invalid mandatory protocol
    // must not commit endpoints/reservations and poison ordinary construction.
    const auto checkStart = std::chrono::steady_clock::now();
    std::optional<AnalysisResult> selected = analyze(program, candidate(endpoints), {false});
    result.work.proposalCheckSites += selected->stats.siteEvaluations;
    result.work.proposalCheckMicroseconds += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - checkStart).count();
    if (!selected->complete || !selected->diagnostics.empty() ||
        !selected->protocol.empty() || !selected->phaseResources.empty()) {
        ++result.work.rejectedProtocolProposals;
        return true;
    }
    std::set<Pipe> repairFreeProducers;
    for (const auto& request : requests)
        repairFreeProducers.insert(request.repairFreeProducers.begin(), request.repairFreeProducers.end());
    if (std::any_of(selected->residuals.begin(), selected->residuals.end(), [&](const auto& residual) {
            return repairFreeProducers.count(program.operations[residual.demand.consumer].pipe);
        })) {
        // The protocol may be safe while its missing producer repair would
        // broaden ordering. Decline unchanged; do not move or invent a fence.
        ++result.work.rejectedSupportProposals;
        return true;
    }
    // An exactly fitting cohort can strand an uncovered ordinary demand.
    // Conservative direct-vocabulary admission, not a global allocation proof
    // or a fixed spare-key heuristic. Full-pool fully supported cycles may pass.
    for (const auto& residual : selected->residuals) {
        const auto source = program.operations[residual.demand.producer].pipe;
        const auto observer = program.operations[residual.demand.consumer].pipe;
        if (source == observer) continue;
        bool proposed = false, ordinary = false;
        for (Id key = 0; key < frontier.keys().size(); ++key) {
            const auto& identity = frontier.keys()[key];
            if (identity.source != source || identity.observer != observer) continue;
            proposed |= proposedKeys.count(key) != 0;
            ordinary |= !proposedKeys.count(key) && !fixedKeys.count(key);
        }
        if (proposed && !ordinary) {
            ++result.work.rejectedResourceProposals;
            return true;
        }
    }
    for (Id index = requests.size(); index-- > 0;) {
        if (!options.recurringOmissionTrials) break;
        if (requests[index].qualifiedCycle) continue;
        if (!alternativeRoute(index)) continue;
        if (!selected) {
            selected = analyze(program, candidate(endpoints), {false});
            result.work.recurringAnalysisSites += selected->stats.siteEvaluations;
        }
        if (!selected->complete || !selected->diagnostics.empty() ||
            !selected->protocol.empty() || !selected->phaseResources.empty()) break;
        retained[index] = false;
        auto trialEndpoints = materialize();
        auto trial = analyze(program, candidate(trialEndpoints), {false});
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
            endpoints = std::move(trialEndpoints);
            ++result.work.redundantRecurringChannels;
        } else retained[index] = true;
    }
    std::vector<Id> channels(requests.size(), NoAnalysisId);
    for (Id index = 0; index < requests.size(); ++index) {
        if (!retained[index]) continue;
        const auto& request = requests[index];
        result.work.sharedReaderReturns += request.sharedReturns;
        reserved.insert(keys[index]);
        needsContextualReplay = true;
        const auto number = frontier.keys()[keys[index]].key;
        channels[index] = result.channels.size();
        result.channels.push_back({request.cell, number, request.cells, request.source, request.observer,
                                   request.publications, request.acquisitions, request.owner,
                                   request.period});
    }
    for (const auto& endpoint : endpoints)
        ledger.append(endpoint.cut, endpoint.command, EndpointPurpose::RecurringCompletion,
                      channels[endpoint.request]);
    result.work.recurringChannels = result.channels.size();
    // These are physical access roles, not definite-write/content certificates.
    // They are symbolic obligations, not assumed fresh-entry receipts.
    // finish() checks the entire selected ledger from the original root, through
    // every original entry/backedge/exit. Failure exports no executable program.
    return true;
}
} // namespace mlir::pto::oahs::selected
