// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/CompletionTransfer.h"
#include <limits>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
void count(uint64_t& value)
{
    if (value != std::numeric_limits<uint64_t>::max()) { ++value; }
}
bool valid(const CompletionSummary& summary, std::size_t classes)
{
    if (summary.all.size() != classes || summary.exposed.size() != classes ||
        summary.suffix.size() != classes) { return false; }
    for (std::size_t c = 0; c < classes; ++c) {
        if (summary.all[c] > 3 || (summary.exposed[c] & ~summary.all[c]) ||
            (summary.suffix[c] & ~summary.all[c])) { return false; }
    }
    return true;
}
bool conflict(const CompletionFacts& a, const CompletionFacts& b)
{
    for (std::size_t c = 0; c < a.size(); ++c) {
        if (((a[c] & 2) && b[c]) || (a[c] && (b[c] & 2))) { return true; }
    }
    return false;
}
void join(CompletionFacts& a, const CompletionFacts& b)
{
    for (std::size_t c = 0; c < a.size(); ++c) { a[c] |= b[c]; }
}
struct Flow {
    std::vector<CompletionFacts> incoming, outgoing;
};
// Iterative Kosaraju traversal: source components precede their successors.
// The explicit DFS stack also handles deeply nested/long source CFGs.
std::vector<std::vector<std::size_t>> components(
    const std::vector<CompletionVertex>& graph, const std::vector<bool>& reachable)
{
    std::vector<std::vector<std::size_t>> reverse(graph.size()), result;
    std::vector<bool> seen(graph.size(), false);
    std::vector<std::size_t> finish;
    for (std::size_t v = 0; v < graph.size(); ++v) {
        if (!reachable[v]) { continue; }
        for (auto successor : graph[v].successors) { reverse[successor].push_back(v); }
        if (seen[v]) { continue; }
        std::vector<std::pair<std::size_t, std::size_t>> stack{{v, 0}};
        seen[v] = true;
        while (!stack.empty()) {
            const auto node = stack.back().first;
            auto& edge = stack.back().second;
            if (edge == graph[node].successors.size()) {
                finish.push_back(node); stack.pop_back(); continue;
            }
            const auto successor = graph[node].successors[edge++];
            if (!seen[successor]) { seen[successor] = true; stack.emplace_back(successor, 0); }
        }
    }
    seen.assign(graph.size(), false);
    for (auto it = finish.rbegin(); it != finish.rend(); ++it) {
        if (seen[*it]) { continue; }
        result.push_back({*it}); seen[*it] = true;
        auto& nodes = result.back();
        for (std::size_t next = 0; next < nodes.size(); ++next) {
            for (auto predecessor : reverse[nodes[next]]) {
                if (!seen[predecessor]) { seen[predecessor] = true; nodes.push_back(predecessor); }
            }
        }
    }
    return result;
}
class ComponentTransfer {
public:
    ComponentTransfer(const std::vector<CompletionVertex>& graph, std::size_t classes,
                      CompletionTransferResult& result)
        : graph(graph), classes(classes), result(result),
          flow{std::vector<CompletionFacts>(graph.size(), CompletionFacts(classes)),
               std::vector<CompletionFacts>(graph.size(), CompletionFacts(classes))},
          external(graph.size(), CompletionFacts(classes)), internal(graph.size()), boundary(graph.size()) {}
    Flow run()
    {
        const auto groups = components(graph, result.reachable);
        std::vector<std::size_t> owner(graph.size(), graph.size());
        for (std::size_t i = 0; i < groups.size(); ++i) {
            for (auto v : groups[i]) { owner[v] = i; }
        }
        for (const auto& nodes : groups) {
            for (auto v : nodes) {
                for (auto successor : graph[v].successors) {
                    (owner[v] == owner[successor] ? internal[v] : boundary[v]).push_back(successor);
                }
            }
        }
        for (const auto& nodes : groups) { finalize(nodes); }
        return std::move(flow);
    }
private:
    const std::vector<CompletionVertex>& graph;
    std::size_t classes;
    CompletionTransferResult& result;
    Flow flow;
    std::vector<CompletionFacts> external;
    std::vector<std::vector<std::size_t>> internal, boundary;
    std::vector<std::tuple<std::size_t, std::size_t, uint8_t>> pending;
    void publish(std::size_t vertex, std::size_t cell, uint8_t mode)
    {
        if (!(flow.outgoing[vertex][cell] & mode)) {
            flow.outgoing[vertex][cell] |= mode;
            pending.emplace_back(vertex, cell, mode);
            count(result.work.propagatedFacts);
        }
    }
    void solve(const std::vector<std::size_t>& nodes)
    {
        count(result.work.componentSolves);
        pending.clear();
        for (auto v : nodes) {
            flow.incoming[v] = external[v];
            flow.outgoing[v].assign(classes, 0);
            for (std::size_t c = 0; c < classes; ++c) {
                const auto initial = graph[v].summary.suffix[c] |
                    ((!result.protectEntry[v] && graph[v].summary.transparent) ? external[v][c] : 0);
                for (uint8_t mode : {uint8_t(1), uint8_t(2)}) {
                    if (initial & mode) { publish(v, c, mode); }
                }
            }
        }
        for (std::size_t next = 0; next < pending.size(); ++next) {
            const auto [v, cell, mode] = pending[next];
            for (auto successor : internal[v]) {
                count(result.work.edgeFactVisits); count(result.work.internalEdgeFactVisits);
                if (flow.incoming[successor][cell] & mode) { continue; }
                flow.incoming[successor][cell] |= mode;
                if (!result.protectEntry[successor] && graph[successor].summary.transparent) {
                    publish(successor, cell, mode);
                }
            }
        }
    }
    void finalize(const std::vector<std::size_t>& nodes)
    {
        count(result.work.components);
        const bool cyclic = nodes.size() > 1 || !internal[nodes.front()].empty();
        if (cyclic) { count(result.work.cyclicComponents); solve(nodes); }
        for (auto v : nodes) {
            result.protectEntry[v] = result.protectEntry[v] ||
                conflict(cyclic ? flow.incoming[v] : external[v], graph[v].summary.exposed);
        }
        solve(nodes);
        for (auto v : nodes) {
            for (auto successor : boundary[v]) {
                for (std::size_t c = 0; c < classes; ++c) {
                    for (uint8_t mode : {uint8_t(1), uint8_t(2)}) {
                        if (flow.outgoing[v][c] & mode) {
                            count(result.work.edgeFactVisits); count(result.work.externalEdgeFactVisits);
                            external[successor][c] |= mode;
                        }
                    }
                }
            }
        }
    }
};
} // namespace
CompletionTransferResult buildCompletionTransfer(const std::vector<CompletionVertex>& graph,
    std::size_t entry, std::size_t exit, std::size_t classCount)
{
    CompletionTransferResult result;
    if (entry >= graph.size() || exit >= graph.size() || !graph[exit].successors.empty()) {
        result.error = "completion CFG requires valid entry and terminal exit"; return result;
    }
    for (const auto& vertex : graph) {
        if (!valid(vertex.summary, classCount)) {
            result.error = "completion summary has inconsistent storage classes or modes"; return result;
        }
        for (auto successor : vertex.successors) {
            if (successor >= graph.size()) {
                result.error = "completion CFG has an invalid successor"; return result;
            }
        }
    }
    result.reachable.assign(graph.size(), false);
    result.protectEntry.assign(graph.size(), false);
    std::vector<std::size_t> pending{entry};
    result.reachable[entry] = true;
    for (std::size_t next = 0; next < pending.size(); ++next) {
        for (auto successor : graph[pending[next]].successors) {
            if (!result.reachable[successor]) {
                result.reachable[successor] = true; pending.push_back(successor);
            }
        }
    }
    for (auto v : pending) { result.protectEntry[v] = graph[v].requiresEntryCompletion; }
    auto finalized = ComponentTransfer(graph, classCount, result).run();
    auto& summary = result.summary;
    summary.all.assign(classCount, 0);
    summary.exposed.assign(classCount, 0);
    summary.suffix = std::move(finalized.outgoing[exit]);
    summary.transparent = false;
    for (auto v : pending) { join(summary.all, graph[v].summary.all); }
    // Only paths with no preceding completion contribute exposed effects.
    // The same reachability determines whether an empty-boundary visit exists.
    std::vector<bool> beforeCompletion(graph.size(), false);
    pending.assign(1, entry); beforeCompletion[entry] = true;
    for (std::size_t next = 0; next < pending.size(); ++next) {
        const auto v = pending[next];
        if (result.protectEntry[v]) { continue; }
        join(summary.exposed, graph[v].summary.exposed);
        if (!graph[v].summary.transparent) { continue; }
        if (v == exit) { summary.transparent = true; }
        for (auto successor : graph[v].successors) {
            if (!beforeCompletion[successor]) {
                beforeCompletion[successor] = true; pending.push_back(successor);
            }
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
