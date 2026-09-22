// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"
#include <algorithm>
#include <functional>
#include <queue>

namespace mlir::pto::oahs::selected {
namespace {
// Iterative DFS avoids making compiler stack depth depend on source nesting.
std::vector<Id> finishingOrder(const detail::ControlGraph& graph, const std::vector<bool>& reachable)
{
    std::vector<bool> seen(graph.sites.size());
    std::vector<Id> order;
    for (Id root = 0; root < graph.sites.size(); ++root) {
        if (!reachable[root] || seen[root]) {
            continue;
        }
        std::vector<std::pair<Id, Id>> stack{{root, 0}};
        seen[root] = true;
        while (!stack.empty()) {
            auto& frame = stack.back();
            const auto& successors = graph.sites[frame.first].successors;
            if (frame.second == successors.size()) {
                order.push_back(frame.first);
                stack.pop_back();
                continue;
            }
            const auto next = successors[frame.second++];
            if (!seen[next]) {
                seen[next] = true;
                stack.emplace_back(next, 0);
            }
        }
    }
    return order;
}
std::vector<std::vector<Id>> strongComponents(
    const detail::ControlGraph& graph, const std::vector<std::vector<Id>>& predecessors,
    const std::vector<bool>& reachable, std::vector<Id>& membership)
{
    auto order = finishingOrder(graph, reachable);
    std::vector<std::vector<Id>> groups;
    membership.assign(graph.sites.size(), NoAnalysisId);
    for (auto at = order.rbegin(); at != order.rend(); ++at) {
        if (membership[*at] != NoAnalysisId) {
            continue;
        }
        const auto group = groups.size();
        groups.emplace_back();
        std::vector<Id> todo{*at};
        membership[*at] = group;
        while (!todo.empty()) {
            const auto site = todo.back();
            todo.pop_back();
            groups.back().push_back(site);
            for (auto before : predecessors[site]) {
                if (reachable[before] && membership[before] == NoAnalysisId) {
                    membership[before] = group;
                    todo.push_back(before);
                }
            }
        }
        std::sort(groups.back().begin(), groups.back().end());
    }
    return groups;
}
Id rank(const detail::ControlGraph& graph, Id site)
{
    return site < graph.cutRanks.size() ? graph.cutRanks[site] : site;
}
void loopSummaries(Control& control)
{
    const auto& graph = control.graph;
    control.constructionEdges.resize(graph.sites.size());
    control.headerAccesses.resize(graph.sites.size());
    std::map<Id, std::set<Id>> loops;
    std::map<Id, std::set<Id>> tails;
    for (Id source = 0; source < graph.sites.size(); ++source) {
        if (!control.reachable[source]) {
            continue;
        }
        const auto& node = graph.sites[source];
        for (Id i = 0; i < node.successors.size(); ++i) {
            const auto header = node.successors[i];
            if (node.backedgeOwners[i] == NoAnalysisId) {
                control.constructionEdges[source].push_back(header);
                continue;
            }
            std::set<Id> body{header, source};
            std::vector<Id> work;
            if (source != header) {
                work.push_back(source);
            }
            while (!work.empty()) {
                const auto site = work.back();
                work.pop_back();
                for (auto before : control.predecessors[site]) {
                    if (control.reachable[before] && body.insert(before).second && before != header) {
                        work.push_back(before);
                    }
                }
            }
            bool singleEntry = !body.count(graph.entry) || header == graph.entry;
            for (auto site : body) {
                for (auto before : control.predecessors[site]) {
                    singleEntry &= site == header || !control.reachable[before] || body.count(before);
                }
            }
            if (singleEntry) {
                loops[header].insert(body.begin(), body.end());
                tails[header].insert(source);
            }
        }
    }
    for (const auto& loop : loops) {
        std::set<Id> exits, operations;
        for (auto site : loop.second) {
            if (graph.operations[site] != NoAnalysisId) {
                operations.insert(graph.operations[site]);
            }
            for (auto next : graph.sites[site].successors) {
                if (!loop.second.count(next)) {
                    exits.insert(next);
                }
            }
        }
        control.headerAccesses[loop.first].assign(operations.begin(), operations.end());
        for (auto tail : tails[loop.first]) {
            auto& edges = control.constructionEdges[tail];
            edges.insert(edges.end(), exits.begin(), exits.end());
            std::sort(edges.begin(), edges.end());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        }
    }
}
bool bodyOrder(const Control& control, const std::vector<Id>& membership,
               Id group, Component& component)
{
    const auto& graph = control.graph;
    std::map<Id, Id> indegree;
    using Ranked = std::pair<Id, Id>;
    std::priority_queue<Ranked, std::vector<Ranked>, std::greater<Ranked>> ready;
    for (auto site : component.sites) {
        indegree[site] = 0;
    }
    for (auto site : component.sites) {
        for (auto next : control.constructionEdges[site]) {
            if (membership[next] == group) {
                ++indegree[next];
            }
        }
    }
    for (const auto& entry : indegree) {
        if (entry.second == 0) {
            ready.emplace(rank(graph, entry.first), entry.first);
        }
    }
    while (!ready.empty()) {
        const auto site = ready.top().second;
        ready.pop();
        component.order.push_back(site);
        for (auto target : control.constructionEdges[site]) {
            if (membership[target] != group) {
                continue;
            }
            --indegree[target];
            if (indegree[target] == 0) {
                ready.emplace(rank(graph, target), target);
            }
        }
    }
    return component.order.size() == component.sites.size();
}
} // namespace

Control::Control(const Program& program)
{
    const auto valid = validateProgram(program);
    if (!valid.success) {
        reason = valid.reason;
        return;
    }
    validInput = true;
    graph = detail::buildControlGraph(program);
    sitePipes.resize(graph.sites.size(), Pipe::Count);
    for (Id site = 0; site < graph.sites.size(); ++site)
        if (graph.operations[site] != NoAnalysisId)
            sitePipes[site] = program.operations[graph.operations[site]].pipe;
    reachable = detail::reachableSites(graph);
    predecessors.resize(graph.sites.size());
    for (Id site = 0; site < graph.sites.size(); ++site) {
        for (auto next : graph.sites[site].successors) {
            predecessors[next].push_back(site);
        }
    }
    loopSummaries(*this);
    std::vector<Id> membership;
    const auto groups = strongComponents(graph, predecessors, reachable, membership);
    std::vector<std::set<Id>> edges(groups.size());
    std::vector<Id> indegree(groups.size());
    for (Id site = 0; site < graph.sites.size(); ++site) {
        if (!reachable[site]) {
            continue;
        }
        for (auto next : graph.sites[site].successors) {
            const auto a = membership[site], b = membership[next];
            if (a != b && edges[a].insert(b).second) {
                ++indegree[b];
            }
        }
    }
    using Ranked = std::pair<Id, Id>;
    std::priority_queue<Ranked, std::vector<Ranked>, std::greater<Ranked>> ready;
    for (Id group = 0; group < groups.size(); ++group) {
        if (indegree[group] == 0) {
            ready.emplace(rank(graph, groups[group].front()), group);
        }
    }
    component.assign(graph.sites.size(), NoAnalysisId);
    position.assign(graph.sites.size(), NoAnalysisId);
    frame.assign(graph.sites.size(), NoAnalysisId);
    Id ordinal = 0;
    while (!ready.empty()) {
        const auto group = ready.top().second;
        ready.pop();
        Component block;
        block.sites = groups[group];
        block.cyclic = block.sites.size() > 1;
        for (auto site : block.sites) {
            const auto& next = graph.sites[site].successors;
            block.cyclic |= std::find(next.begin(), next.end(), site) != next.end();
            bool entry = site == graph.entry;
            for (auto before : predecessors[site]) {
                entry |= reachable[before] && membership[before] != group;
            }
            if (entry) {
                block.entries.push_back(site);
            }
            component[site] = components.size();
        }
        if (!bodyOrder(*this, membership, group, block) || (block.cyclic && block.entries.size() != 1)) {
            reason = "original loop needs a reducible entry and qualified backedge labels";
            return;
        }
        for (auto site : block.order) {
            position[site] = ordinal++;
            frame[site] = site;
            if (!block.cyclic) {
                if (predecessors[site].size() == 1) {
                    const auto before = predecessors[site].front();
                    if (frame[before] != NoAnalysisId && graph.sites[before].successors.size() == 1)
                        frame[site] = frame[before];
                }
                continue;
            }
            // Crossing a repeated component needs the occurrence vocabulary
            // carried by observed control. Raw structured loops retain the
            // conservative common-cut path; treating their static body order
            // as a generation certificate can create an unrearmable event.
            if (!program.observed) continue;
            std::vector<Id> forwardPredecessors;
            for (auto before : predecessors[site]) {
                const auto& successors = graph.sites[before].successors;
                const auto owners = graph.sites[before].backedgeOwners;
                bool hasForward = false;
                for (std::size_t edge = 0; edge < successors.size(); ++edge) {
                    if (successors[edge] != site) continue;
                    hasForward |= owners.empty() || owners[edge] == NoAnalysisId;
                }
                if (hasForward && membership[before] == group)
                    forwardPredecessors.push_back(before);
            }
            if (forwardPredecessors.size() == 1) {
                const auto before = forwardPredecessors.front();
                unsigned forwardSuccessors = 0;
                for (std::size_t edge = 0; edge < graph.sites[before].successors.size(); ++edge) {
                    const auto owner = graph.sites[before].backedgeOwners.empty()
                        ? NoAnalysisId : graph.sites[before].backedgeOwners[edge];
                    forwardSuccessors += owner == NoAnalysisId;
                }
                if (frame[before] != NoAnalysisId && forwardSuccessors == 1) frame[site] = frame[before];
            }
        }
        components.push_back(std::move(block));
        for (auto next : edges[group]) {
            --indegree[next];
            if (indegree[next] == 0) {
                ready.emplace(rank(graph, groups[next].front()), next);
            }
        }
    }
    complete = components.size() == groups.size();
    if (!complete) return;
    std::vector<std::vector<Id>> originalEdges(graph.sites.size());
    std::vector<bool> payloadSites(graph.sites.size());
    std::vector<LookaheadIndex::ClassIssue> classIssues;
    for (Id site = 0; site < graph.sites.size(); ++site) {
        originalEdges[site] = graph.sites[site].successors;
        const auto operation = graph.operations[site];
        payloadSites[site] = operation != NoAnalysisId;
        if (!reachable[site] || operation == NoAnalysisId || frame[site] == NoAnalysisId) continue;
        const auto& op = program.operations[operation];
        for (const auto& effect : op.accesses) {
            const auto base = (Id(effect.cell) * PipeCount + unsigned(op.pipe)) * 2;
            if (effect.read) classIssues.push_back({frame[site], position[site], base});
            if (effect.write) classIssues.push_back({frame[site], position[site], base + 1});
        }
    }
    if (!lookahead.build(originalEdges, payloadSites, classIssues)) {
        complete = false;
        reason = "invalid immutable lookahead dimensions";
        return;
    }
    // Prepare region entry placement facts once, before selecting any event.
    // Construction queries these summaries instead of rediscovering invariant
    // classes and earlier observer work at every residual repair.
    if (program.observed) for (const auto& loop : program.observed->loops) {
        if (!loop.atLeastOnce || loop.bodyEntry == NoAnalysisId) continue;
        LoopEntryFacts facts;
        facts.entry = loop.entry;
        facts.sites = loop.sites;
        facts.firstConsumer.fill(NoAnalysisId);
        for (const auto& [boundary, consumer] : loop.firstWriteFrontiers) {
            if (boundary >= graph.sites.size() || consumer >= graph.sites.size() ||
                graph.operations[boundary] != NoAnalysisId ||
                graph.operations[consumer] == NoAnalysisId ||
                graph.sites[boundary].successors != std::vector<Id>{consumer} ||
                !lookahead.balancedTransfer({loop.entry}, boundary, graph.entry, graph.exit)) {
                complete = false;
                reason = "invalid first-write receipt boundary";
                return;
            }
            facts.firstWriteFrontiers.emplace_back(boundary, consumer);
            receiptGaps.emplace(boundary, program.operations[graph.operations[consumer]].pipe);
        }
        std::set<Pipe> observers;
        for (auto site : loop.sites) {
            ++loopEntryPreparationSites;
            const auto operation = graph.operations[site];
            if (operation == NoAnalysisId) continue;
            const auto& op = program.operations[operation];
            const auto observation = program.observed->sites[site].observation;
            if (observation != NoAnalysisId) {
                const auto& atoms = program.observed->observations[observation].atoms;
                if (std::any_of(atoms.begin(), atoms.end(), [&](const auto& a) {
                        return a.kind == ObservationAtom::LoopHasPrevious &&
                            a.owner == loop.owner && a.parameter == 1 && a.value == 0;
                    }) && lookahead.balancedTransfer({loop.entry}, site, graph.entry, graph.exit))
                    facts.firstInputConsumers.push_back(site);
            }
            observers.insert(op.pipe);
            facts.issuedPipes.insert(op.pipe);
            for (const auto& access : op.accesses) {
                const auto base = (Id(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
                if (access.read) facts.issuedClasses.insert(base);
                if (access.write) facts.issuedClasses.insert(base + 1);
            }
        }
        for (auto observer : observers) {
            std::vector<bool> seen(graph.sites.size());
            std::vector<Cut> todo{loop.bodyEntry};
            std::set<Cut> first;
            bool bypass = false;
            while (!todo.empty()) {
                const auto at = todo.back(); todo.pop_back();
                if (seen[at]) continue;
                seen[at] = true;
                ++loopEntryPreparationSites;
                if (at == loop.exit) { bypass = true; break; }
                const auto operation = graph.operations[at];
                if (operation != NoAnalysisId && program.operations[operation].pipe == observer) {
                    first.insert(at);
                    continue;
                }
                const auto& next = graph.sites[at].successors;
                todo.insert(todo.end(), next.begin(), next.end());
            }
            if (!bypass && !first.empty()) {
                facts.firstConsumers[unsigned(observer)].assign(first.begin(), first.end());
                if (first.size() == 1) facts.firstConsumer[unsigned(observer)] = *first.begin();
                std::fill(seen.begin(), seen.end(), false);
                todo = graph.sites[loop.entry].successors;
                while (!todo.empty()) {
                    const auto at = todo.back(); todo.pop_back();
                    if (at == loop.exit || at == loop.entry || seen[at]) continue;
                    seen[at] = true;
                    ++loopEntryPreparationSites;
                    facts.crossedWords[unsigned(observer)].push_back(at);
                    if (first.count(at)) continue;
                    const auto& next = graph.sites[at].successors;
                    todo.insert(todo.end(), next.begin(), next.end());
                }
            }
        }
        loopEntries.push_back(std::move(facts));
    }
    // One pass over the sites fixes the canonical word of each site, the sites
    // sharing it, and the range of components it spans. canonicalCommandCut
    // names the EARLIEST site carrying an observation, so the first occurrence
    // encountered in site order is that canonical site. Unreachable and
    // later-canonical occurrences are retained deliberately: an endpoint
    // aggregate over a shared word must not straddle a reuse boundary.
    canonicalCut.resize(graph.sites.size());
    wordOccurrences.assign(graph.sites.size(), {});
    wordSpan.assign(graph.sites.size(), {NoAnalysisId, 0});
    std::map<Id, Cut> earliest;
    for (Id site = 0; site < graph.sites.size(); ++site) {
        canonicalCut[site] = site;
        if (program.observed && legalCommandCut(program, site)) {
            canonicalCut[site] =
                earliest.emplace(program.observed->sites[site].observation, site).first->second;
        }
        wordOccurrences[canonicalCut[site]].push_back(site);
        const auto block = component[site];
        if (block == NoAnalysisId) {
            continue;
        }
        auto& span = wordSpan[canonicalCut[site]];
        span.first = span.first == NoAnalysisId ? block : std::min(span.first, block);
        span.second = std::max(span.second, block);
    }
    if (program.observed) for (const auto& loop : program.observed->loops)
        for (auto cut : loop.firstVisitPrefix) {
            if (cut >= canonicalCut.size()) {
                complete = false;
                reason = "invalid first-visit prefix cut";
                return;
            }
            firstPrefixWords.insert(canonicalCut[cut]);
            if (!loop.firstWriteFrontiers.empty()) firstWriteWords.insert(canonicalCut[cut]);
        }
    if (program.observed) for (Cut cut = 0; cut < graph.sites.size(); ++cut) {
        if (!reachable[cut] || canonicalCut[cut] != cut) continue;
        const auto id = program.observed->sites[cut].observation;
        if (id == NoAnalysisId) continue;
        const auto& observation = program.observed->observations[id];
        if (observation.available && observation.beforeSharedWord &&
            std::any_of(observation.atoms.begin(), observation.atoms.end(), [](const auto& atom) {
                return atom.kind == ObservationAtom::LoopHasNext && atom.value == 0;
            })) finalReadGaps.push_back(cut);
    }
    prepareChoiceFrontiers(program);
}
bool Control::straight(Id a, Id b) const
{
    return a < frame.size() && b < frame.size() && frame[a] != NoAnalysisId &&
           frame[a] == frame[b] && position[a] <= position[b];
}
bool Control::sourceCut(Id cut, Pipe source) const
{
    const auto receipt = receiptGaps.find(cut);
    return graph.legalCuts[cut] &&
        (receipt == receiptGaps.end() || receipt->second == source);
}
Cut Control::after(Id origin) const
{
    auto site = origin;
    while (graph.sites[site].successors.size() == 1) {
        site = graph.sites[site].successors.front();
        if (!straight(origin, site)) {
            break;
        }
        if (sourceCut(site, sitePipes[origin])) {
            return site;
        }
    }
    return NoAnalysisId;
}

OccurrenceMode occurrenceMode(const Program& program, Cut cut)
{
    OccurrenceMode out;
    if (!program.observed || cut >= program.observed->sites.size()) return out;
    const auto observation = program.observed->sites[cut].observation;
    if (observation == NoAnalysisId || observation >= program.observed->observations.size()) return out;
    const auto& value = program.observed->observations[observation];
    const bool insufficientObservation = !value.available || value.atoms.size() < 3;
    if (insufficientObservation) {
        return out;
    }
    uint64_t period = 0;
    Id owner = NoAnalysisId;
    for (const auto& atom : value.atoms)
        if (atom.kind == ObservationAtom::LoopResidue) {
            const bool counted = std::any_of(value.atoms.begin(), value.atoms.end(), [&](const auto& other) {
                return other.owner == atom.owner && other.parameter == atom.parameter &&
                    other.kind == ObservationAtom::LoopHasPrevious;
            });
            if (counted) {
                if (owner != NoAnalysisId) {
                    return {};
                }
                period = atom.parameter;
                owner = atom.owner;
            }
        }
    if (!period) return out;
    unsigned seen = 0;
    for (const auto& atom : value.atoms) {
        if (atom.owner != owner && atom.kind == ObservationAtom::LoopResidue) {
            continue;
        }
        if (period > 1 && atom.owner == owner && atom.parameter == 1 &&
            (atom.kind == ObservationAtom::LoopHasPrevious || atom.kind == ObservationAtom::LoopHasNext))
            continue;
        if (seen && (atom.owner != out.owner || atom.parameter != out.period)) return {};
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


} // namespace mlir::pto::oahs::selected
