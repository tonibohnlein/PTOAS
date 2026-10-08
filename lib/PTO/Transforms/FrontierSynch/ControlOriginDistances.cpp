// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ControlOriginDistances.h"
#include <algorithm>
#include <deque>
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
constexpr uint64_t Unreachable = std::numeric_limits<uint64_t>::max();
struct Arc { uint32_t target; uint8_t weight; };
using Adjacency = std::vector<std::vector<Arc>>;
bool validate(const ControlOriginGraph& graph, std::size_t origins, ControlOriginDistances& result)
{
    if (graph.payloads.size() > graph.vertices / 2) {
        result.error = "control graph has more payload cuts than vertices";
        return false;
    }
    std::vector<uint8_t> role(graph.vertices, 0);
    for (const auto& payload : graph.payloads) {
        if (payload.input >= graph.vertices || payload.output >= graph.vertices || payload.input == payload.output ||
            role[payload.input] || role[payload.output]) {
            result.error = "control payload cuts must be distinct unique graph vertices";
            return false;
        }
        role[payload.input] = 1;
        role[payload.output] = 2;
    }
    for (const auto& edge : graph.edges) {
        if (edge.source >= graph.vertices || edge.target >= graph.vertices || edge.weight > 1 ||
            role[edge.source] == 1 || role[edge.target] == 2) {
            result.error = "control edge has invalid endpoints, weight or payload through-edge bypass";
            return false;
        }
    }
    // Every pass below visits each vertex/edge a constant number of times.
    // Reserve a conservative counter bound before any per-origin traversal.
    const auto limit = std::numeric_limits<uint64_t>::max();
    if (graph.edges.size() > limit - graph.vertices - graph.payloads.size()) {
        result.error = "control graph size exceeds distance counter representation";
        return false;
    }
    const uint64_t work = uint64_t(graph.vertices) + graph.edges.size() + graph.payloads.size();
    constexpr uint64_t CounterPassBound = 32;
    if (work > limit / CounterPassBound || (work && origins > limit / (CounterPassBound * work))) {
        result.error = "control origin traversal counters would overflow";
        return false;
    }
    result.cost.vertices = graph.vertices;
    result.cost.edges = uint64_t(graph.edges.size()) + graph.payloads.size();
    result.cost.origins = origins;
    return true;
}
class OriginAnalysis {
public:
    OriginAnalysis(const ControlOriginGraph& graph, ControlOriginDistances& result)
        : graph(graph), result(result), forward(graph.vertices), reverse(graph.vertices) {}
    bool run(const ControlStorageOrigin& origin);
private:
    bool connect(const ControlStorageOrigin& origin);
    void shortest(uint32_t source);
    std::vector<uint32_t> finish(uint32_t source);
    uint32_t components(uint32_t source);
    bool maximums(uint32_t count, std::vector<uint64_t>& maximum, std::vector<bool>& infinite);
    const ControlOriginGraph& graph;
    ControlOriginDistances& result;
    Adjacency forward, reverse;
    std::vector<uint64_t> minimum;
    std::vector<uint32_t> component;
};
bool OriginAnalysis::connect(const ControlStorageOrigin& origin)
{
    if (origin.payload >= graph.payloads.size() || origin.definiteKillPayloads.size() > graph.payloads.size()) {
        result.error = "control origin has invalid payload or oversized kill list";
        return false;
    }
    std::vector<bool> killed(graph.payloads.size(), false);
    for (auto payload : origin.definiteKillPayloads) {
        if (payload >= killed.size() || killed[payload]) {
            result.error = "control origin kill payloads must be valid and unique";
            return false;
        }
        killed[payload] = true;
    }
    auto add = [&](uint32_t source, uint32_t target, uint8_t weight) {
        forward[source].push_back({target, weight});
        reverse[target].push_back({source, weight});
        ++result.cost.edgeVisits;
    };
    for (const auto& edge : graph.edges) { add(edge.source, edge.target, edge.weight); }
    for (std::size_t id = 0; id < graph.payloads.size(); ++id) {
        const auto& payload = graph.payloads[id];
        if (!killed[id]) { add(payload.input, payload.output, 0); }
    }
    return true;
}
void OriginAnalysis::shortest(uint32_t source)
{
    minimum.assign(graph.vertices, Unreachable);
    std::vector<bool> settled(graph.vertices, false);
    std::deque<std::pair<uint32_t, uint64_t>> pending;
    pending.push_back({source, 0});
    minimum[source] = 0;
    while (!pending.empty()) {
        auto [vertex, distance] = pending.front();
        pending.pop_front();
        ++result.cost.vertexVisits;
        if (settled[vertex] || distance != minimum[vertex]) { continue; }
        settled[vertex] = true;
        for (const auto& edge : forward[vertex]) {
            ++result.cost.edgeVisits;
            // A finite shortest path is simple: distance <= V-1 <= UINT32_MAX.
            const auto candidate = distance + edge.weight;
            if (candidate >= minimum[edge.target]) { continue; }
            minimum[edge.target] = candidate;
            if (edge.weight) { pending.push_back({edge.target, candidate}); }
            else { pending.push_front({edge.target, candidate}); }
        }
    }
}
std::vector<uint32_t> OriginAnalysis::finish(uint32_t source)
{
    struct Frame { uint32_t vertex; std::size_t next; };
    std::vector<Frame> stack{{source, 0}};
    std::vector<bool> visited(graph.vertices, false);
    std::vector<uint32_t> order;
    visited[source] = true;
    while (!stack.empty()) {
        auto& frame = stack.back();
        if (frame.next == forward[frame.vertex].size()) {
            order.push_back(frame.vertex);
            stack.pop_back();
            ++result.cost.vertexVisits;
            continue;
        }
        const auto next = forward[frame.vertex][frame.next++].target;
        ++result.cost.edgeVisits;
        if (!visited[next]) { visited[next] = true; stack.push_back({next, 0}); }
    }
    return order;
}
uint32_t OriginAnalysis::components(uint32_t source)
{
    const auto order = finish(source);
    component.assign(graph.vertices, UINT32_MAX);
    uint32_t count = 0;
    for (auto position = order.rbegin(); position != order.rend(); ++position) {
        if (component[*position] != UINT32_MAX) { continue; }
        std::vector<uint32_t> pending{*position};
        component[*position] = count;
        while (!pending.empty()) {
            const auto vertex = pending.back();
            pending.pop_back();
            ++result.cost.vertexVisits;
            for (const auto& edge : reverse[vertex]) {
                ++result.cost.edgeVisits;
                if (minimum[edge.target] == Unreachable || component[edge.target] != UINT32_MAX) { continue; }
                component[edge.target] = count;
                pending.push_back(edge.target);
            }
        }
        ++count;
    }
    return count;
}
bool OriginAnalysis::maximums(uint32_t count, std::vector<uint64_t>& maximum, std::vector<bool>& infinite)
{
    Adjacency condensed(count);
    std::vector<std::size_t> incoming(count, 0);
    maximum.assign(count, 0);
    infinite.assign(count, false);
    for (uint32_t vertex = 0; vertex < graph.vertices; ++vertex) {
        ++result.cost.vertexVisits;
        if (minimum[vertex] == Unreachable) { continue; }
        for (const auto& edge : forward[vertex]) {
            ++result.cost.edgeVisits;
            const auto source = component[vertex], target = component[edge.target];
            if (source == target) { infinite[source] = infinite[source] || edge.weight != 0; }
            else { condensed[source].push_back({target, edge.weight}); ++incoming[target]; }
        }
    }
    std::vector<uint32_t> pending;
    for (uint32_t id = 0; id < count; ++id) { if (!incoming[id]) { pending.push_back(id); } }
    for (std::size_t next = 0; next < pending.size(); ++next) {
        const auto source = pending[next];
        ++result.cost.vertexVisits;
        for (const auto& edge : condensed[source]) {
            ++result.cost.edgeVisits;
            infinite[edge.target] = infinite[edge.target] || infinite[source];
            // A finite condensation path has at most V-1 edges of weight <=1.
            if (!infinite[edge.target]) {
                maximum[edge.target] = std::max(maximum[edge.target], maximum[source] + edge.weight);
            }
            if (--incoming[edge.target] == 0) { pending.push_back(edge.target); }
        }
    }
    if (pending.size() != count) {
        result.error = "control origin condensation is not acyclic";
        return false;
    }
    return true;
}
bool OriginAnalysis::run(const ControlStorageOrigin& origin)
{
    if (!connect(origin)) { return false; }
    const auto& payload = graph.payloads[origin.payload];
    shortest(payload.output);
    if (minimum[payload.input] == 0) {
        result.error = "control origin can revisit its input without advancing an iteration";
        return false;
    }
    const auto count = components(payload.output);
    std::vector<uint64_t> maximum;
    std::vector<bool> infinite;
    if (!maximums(count, maximum, infinite)) { return false; }
    std::vector<OriginDistanceInterval> row(graph.vertices);
    for (uint32_t vertex = 0; vertex < graph.vertices; ++vertex) {
        ++result.cost.vertexVisits;
        if (minimum[vertex] == Unreachable) { continue; }
        const auto group = component[vertex];
        row[vertex] = {true, minimum[vertex], infinite[group] ? std::nullopt : std::optional<uint64_t>(maximum[group])};
    }
    result.distances.push_back(std::move(row));
    return true;
}
} // namespace
ControlOriginDistances computeControlOriginDistances(
    const ControlOriginGraph& graph, llvm::ArrayRef<ControlStorageOrigin> origins)
{
    ControlOriginDistances result;
    if (!validate(graph, origins.size(), result)) { return result; }
    for (const auto& origin : origins) {
        OriginAnalysis analysis(graph, result);
        if (!analysis.run(origin)) { result.distances.clear(); return result; }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
