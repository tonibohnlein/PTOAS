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
            if (block.cyclic) {
                continue;
            }
            frame[site] = site;
            if (predecessors[site].size() == 1) {
                const auto before = predecessors[site].front();
                if (frame[before] != NoAnalysisId && graph.sites[before].successors.size() == 1) {
                    frame[site] = frame[before];
                }
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
}
bool Control::straight(Id a, Id b) const
{
    return a < frame.size() && b < frame.size() && frame[a] != NoAnalysisId &&
           frame[a] == frame[b] && position[a] <= position[b];
}
Cut Control::after(Id origin) const
{
    auto site = origin;
    while (graph.sites[site].successors.size() == 1) {
        site = graph.sites[site].successors.front();
        if (!straight(origin, site)) {
            break;
        }
        if (graph.legalCuts[site]) {
            return site;
        }
    }
    return NoAnalysisId;
}
} // namespace mlir::pto::oahs::selected
