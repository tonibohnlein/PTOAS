// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve coactive command order; refine only groups that participate in cycles.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ENDPOINTORDER_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ENDPOINTORDER_H
#include <algorithm>
#include <set>
#include <utility>
#include <vector>
namespace mlir::pto::frontiersynch {
struct EndpointOrder {
    std::vector<std::size_t> positions;
    std::set<std::size_t> cyclic;
};
// Edges refer to this immutable graph's dense node indices. Grouping may make
// these constraints cyclic even though the original alternatives are ordered.
// Acyclic descendants of a cycle must retain their compact representation.
inline EndpointOrder orderEndpointGroups(const std::vector<std::set<std::size_t>>& edges)
{
    EndpointOrder result;
    const auto count = edges.size();
    std::vector<std::size_t> incoming(count, 0);
    for (const auto& successors : edges) {
        for (auto target : successors) { ++incoming[target]; }
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (!incoming[i]) { result.positions.push_back(i); }
    }
    for (std::size_t i = 0; i < result.positions.size(); ++i) {
        for (auto target : edges[result.positions[i]]) {
            if (!--incoming[target]) { result.positions.push_back(target); }
        }
    }
    if (result.positions.size() == count) { return result; }
    // Iterative Kosaraju traversal avoids recursion proportional to the command
    // inventory. Both traversals are linear in the supplied group graph.
    std::vector<bool> seen(count, false);
    std::vector<std::size_t> finished;
    using Frame = std::pair<std::size_t, std::set<std::size_t>::const_iterator>;
    std::vector<Frame> stack;
    for (std::size_t root = 0; root < count; ++root) {
        if (seen[root]) { continue; }
        seen[root] = true;
        stack.emplace_back(root, edges[root].begin());
        while (!stack.empty()) {
            auto& [node, next] = stack.back();
            if (next == edges[node].end()) {
                finished.push_back(node);
                stack.pop_back();
                continue;
            }
            auto target = *next++;
            if (!seen[target]) {
                seen[target] = true;
                stack.emplace_back(target, edges[target].begin());
            }
        }
    }
    std::vector<std::vector<std::size_t>> predecessors(count);
    for (std::size_t source = 0; source < count; ++source) {
        for (auto target : edges[source]) { predecessors[target].push_back(source); }
    }
    std::fill(seen.begin(), seen.end(), false);
    std::vector<std::size_t> pending, component;
    for (auto it = finished.rbegin(); it != finished.rend(); ++it) {
        if (seen[*it]) { continue; }
        component.clear();
        pending.push_back(*it);
        seen[*it] = true;
        while (!pending.empty()) {
            auto node = pending.back();
            pending.pop_back();
            component.push_back(node);
            for (auto source : predecessors[node]) {
                if (!seen[source]) { seen[source] = true; pending.push_back(source); }
            }
        }
        if (component.size() > 1 || edges[component.front()].count(component.front())) {
            result.cyclic.insert(component.begin(), component.end());
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
#endif
