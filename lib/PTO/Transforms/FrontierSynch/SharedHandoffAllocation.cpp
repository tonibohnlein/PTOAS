// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Greedy shared-pool walk, followed when necessary by an exact chain partition.
#include "PTO/Transforms/FrontierSynch/SharedHandoffAllocation.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <algorithm>
#include <functional>
#include <queue>
#include <numeric>
#include <optional>
namespace mlir::pto::frontiersynch {
namespace {
using Edges = std::vector<std::vector<uint32_t>>;
bool prepare(Edges& edges, std::vector<uint32_t>& order, std::string& error)
{
    std::vector<uint32_t> indegree(edges.size(), 0);
    for (uint32_t i = 0; i < edges.size(); ++i) {
        auto& row = edges[i];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        for (auto j : row) {
            if (j >= edges.size() || i == j) {
                error = "shared handoff reuse has an invalid or reflexive edge";
                return false;
            }
            ++indegree[j];
        }
    }
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> ready;
    for (uint32_t i = 0; i < edges.size(); ++i) {
        if (!indegree[i]) { ready.push(i); }
    }
    while (!ready.empty()) {
        const auto i = ready.top(); ready.pop();
        order.push_back(i);
        for (auto j : edges[i]) {
            --indegree[j];
            if (!indegree[j]) { ready.push(j); }
        }
    }
    if (order.size() != edges.size()) {
        error = "shared handoff reuse contains a cycle";
        return false;
    }
    for (const auto& row : edges) {
        for (auto j : row) {
            if (!std::includes(row.begin(), row.end(), edges[j].begin(), edges[j].end())) {
                error = "shared handoff reuse must include its transitive closure";
                return false;
            }
        }
    }
    return true;
}
std::optional<SharedHandoffAllocation> greedy(llvm::ArrayRef<SharedHandoff> handoffs,
    llvm::ArrayRef<uint32_t> order, const std::function<bool(uint32_t, uint32_t)>& precedes, uint64_t capacity)
{
    SharedHandoffAllocation result;
    result.lanes.resize(handoffs.size());
    std::vector<uint32_t> last;
    for (auto next : order) {
        uint32_t selected = UINT32_MAX, free = UINT32_MAX;
        for (uint32_t lane = 0; lane < last.size(); ++lane) {
            const auto prior = last[lane];
            if (!precedes(prior, next)) { continue; }
            if (free == UINT32_MAX) { free = lane; }
            if (handoffs[prior].sourcePipe == handoffs[next].sourcePipe &&
                handoffs[prior].targetPipe == handoffs[next].targetPipe) {
                selected = lane;
                break;
            }
        }
        if (selected == UINT32_MAX) { selected = free; }
        if (selected == UINT32_MAX) {
            if (last.size() == capacity) { return std::nullopt; }
            selected = static_cast<uint32_t>(last.size());
            last.push_back(next);
        } else { last[selected] = next; }
        result.lanes[next] = selected;
    }
    result.budget = last.size();
    return result;
}
bool augment(uint32_t root, const Edges& edges, std::vector<uint32_t>& left, std::vector<uint32_t>& right)
{
    std::vector<uint32_t> parent(edges.size(), UINT32_MAX), queue{root};
    std::vector<uint8_t> seen(edges.size(), 0);
    seen[root] = 1;
    for (std::size_t head = 0; head < queue.size(); ++head) {
        const auto u = queue[head];
        for (auto v : edges[u]) {
            if (parent[v] != UINT32_MAX) { continue; }
            parent[v] = u;
            if (right[v] == UINT32_MAX) {
                auto current = v;
                while (current != UINT32_MAX) {
                    const auto source = parent[current], old = left[source];
                    left[source] = current; right[current] = source;
                    current = old;
                }
                return true;
            }
            if (!seen[right[v]]) {
                seen[right[v]] = 1;
                queue.push_back(right[v]);
            }
        }
    }
    return false;
}
SharedHandoffAllocation minimum(const Edges& edges, llvm::ArrayRef<uint32_t> order)
{
    std::vector<uint32_t> left(edges.size(), UINT32_MAX), right(edges.size(), UINT32_MAX);
    for (auto root : order) {
        augment(root, edges, left, right);
    }
    SharedHandoffAllocation result;
    result.lanes.assign(edges.size(), UINT32_MAX);
    result.exactMinimum = true;
    for (auto root : order) {
        if (right[root] != UINT32_MAX) { continue; }
        for (auto current = root; current != UINT32_MAX; current = left[current]) {
            result.lanes[current] = static_cast<uint32_t>(result.budget);
        }
        ++result.budget;
    }
    return result;
}
} // namespace
SharedHandoffAllocation allocateSharedHandoffs(
    llvm::ArrayRef<SharedHandoff> handoffs, llvm::ArrayRef<std::vector<uint32_t>> successors, uint64_t capacity)
{
    SharedHandoffAllocation invalid;
    if (handoffs.size() > UINT32_MAX || handoffs.size() != successors.size()) {
        invalid.error = "shared handoff reuse dimensions do not match representable handoff indices";
        return invalid;
    }
    Edges edges(successors.begin(), successors.end());
    std::vector<uint32_t> order;
    if (!prepare(edges, order, invalid.error)) { return invalid; }
    if (handoffs.empty()) { invalid.exactMinimum = true; return invalid; }
    auto result = greedy(handoffs, order, [&](uint32_t from, uint32_t to) {
        return std::binary_search(edges[from].begin(), edges[from].end(), to);
    }, capacity);
    return result ? *result : minimum(edges, order);
}
SharedHandoffAllocation allocateSharedHandoffsByQuery(
    llvm::ArrayRef<SharedHandoff> handoffs,
    const std::function<bool(uint32_t, uint32_t)>& precedes, uint64_t capacity)
{
    SharedHandoffAllocation result;
    if (handoffs.size() > UINT32_MAX || !precedes) {
        result.error = "certified handoff query is missing or its indices are unrepresentable";
        return result;
    }
    std::vector<uint32_t> order(handoffs.size());
    std::iota(order.begin(), order.end(), 0);
    if (auto assigned = greedy(handoffs, order, precedes, capacity)) {
        assigned->exactMinimum = handoffs.empty();
        return *assigned;
    }
    // Greedy has stopped at the first exhausted pool. No quadratic relation is
    // allocated on the successful path, including smaller supplied ID subsets.
    Edges edges(handoffs.size());
    for (uint32_t i = 0; i < handoffs.size(); ++i) {
        for (uint32_t j = i + 1; j < handoffs.size(); ++j) {
            if (precedes(i, j)) { edges[i].push_back(j); }
        }
    }
    order.clear();
    SharedHandoffAllocation invalid;
    if (!prepare(edges, order, invalid.error)) { return invalid; }
    return minimum(edges, order);
}
namespace {
using Directions = std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>;
Directions directionGroups(llvm::ArrayRef<SharedHandoff> handoffs)
{
    Directions groups;
    for (uint32_t i = 0; i < handoffs.size(); ++i) {
        groups[{handoffs[i].sourcePipe, handoffs[i].targetPipe}].push_back(i);
    }
    return groups;
}
void mergeDirection(SharedHandoffAllocation& output, const SharedHandoffAllocation& part,
                    const std::vector<uint32_t>& indices)
{
    output.budget = std::max(output.budget, part.budget);
    output.exactMinimum &= part.exactMinimum;
    for (std::size_t i = 0; i < indices.size(); ++i) { output.lanes[indices[i]] = part.lanes[i]; }
}
} // namespace
SharedHandoffAllocation allocateDirectedHandoffs(
    llvm::ArrayRef<SharedHandoff> handoffs, llvm::ArrayRef<std::vector<uint32_t>> successors, uint64_t capacity)
{
    SharedHandoffAllocation result;
    if (handoffs.size() >= UINT32_MAX || handoffs.size() != successors.size()) {
        result.error = "directed handoff dimensions are invalid"; return result;
    }
    for (const auto& row : successors) {
        if (llvm::any_of(row, [&](uint32_t id) { return id >= handoffs.size(); })) {
            result.error = "directed handoff successor is out of range"; return result;
        }
    }
    result.lanes.resize(handoffs.size()); result.exactMinimum = true;
    for (const auto& group : directionGroups(handoffs)) {
        const auto& direction = group.first;
        const auto& indices = group.second;
        std::map<uint32_t, uint32_t> local;
        for (uint32_t i = 0; i < indices.size(); ++i) { local[indices[i]] = i; }
        Edges edges(indices.size());
        for (std::size_t i = 0; i < indices.size(); ++i) {
            for (auto next : successors[indices[i]]) {
                auto found = local.find(next);
                if (found != local.end()) { edges[i].push_back(found->second); }
            }
        }
        std::vector<SharedHandoff> part(indices.size(), {direction.first, direction.second});
        auto assigned = allocateSharedHandoffs(part, edges, capacity);
        if (!assigned.error.empty()) { return assigned; }
        mergeDirection(result, assigned, indices);
    }
    return result;
}
SharedHandoffAllocation allocateDirectedHandoffsByQuery(
    llvm::ArrayRef<SharedHandoff> handoffs,
    const std::function<bool(uint32_t, uint32_t)>& precedes, uint64_t capacity)
{
    SharedHandoffAllocation result;
    if (handoffs.size() >= UINT32_MAX || !precedes) {
        result.error = "directed handoff query is missing or unrepresentable"; return result;
    }
    result.lanes.resize(handoffs.size()); result.exactMinimum = true;
    for (const auto& group : directionGroups(handoffs)) {
        const auto& direction = group.first;
        const auto& indices = group.second;
        std::vector<SharedHandoff> part(indices.size(), {direction.first, direction.second});
        auto assigned = allocateSharedHandoffsByQuery(part, [&](uint32_t from, uint32_t to) {
            return precedes(indices[from], indices[to]);
        }, capacity);
        if (!assigned.error.empty()) { return assigned; }
        mergeDirection(result, assigned, indices);
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
