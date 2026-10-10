// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One completion-seeded shortest-path index per pipe; shared final-edge tests.
#include "PeriodicAnalysisInternal.h"
#include <set>
namespace mlir::pto::frontiersynch::periodic {
namespace {
using Distance = std::pair<uint64_t, uint32_t>;
bool distances(const Graph& graph, const PeriodicAnalysis& output, uint32_t row,
               PipeFrontier& frontier)
{
    frontier.distances.resize(graph.outgoing.size());
    std::set<Distance> queue; // At most one pending entry per vertex.
    for (uint32_t type = 0; type < output.payloads.size(); ++type) {
        if (output.sourceRows[type] == row) {
            const uint64_t seed = frontier.count - output.localRanks[type];
            frontier.distances[2 * type + 1] = seed;
            queue.emplace(seed, 2 * type + 1);
        }
    }
    while (!queue.empty()) {
        auto [distance, vertex] = *queue.begin();
        queue.erase(queue.begin());
        if (frontier.distances[vertex] != distance) {
            continue;
        }
        for (auto id : graph.outgoing[vertex]) {
            const auto& edge = graph.edges[id];
            uint64_t weight = 0, candidate = 0;
            if (!multiply(edge.weight, frontier.count, weight) || !add(distance, weight, candidate)) {
                return false;
            }
            auto& old = frontier.distances[edge.target];
            if (!old || candidate < *old) {
                if (old) {
                    queue.erase({*old, edge.target});
                }
                old = candidate;
                queue.emplace(candidate, edge.target);
            }
        }
    }
    return true;
}
struct FinalEdge {
    uint64_t score = 0;
    uint32_t identity = 0;
};
struct BestTwo {
    std::optional<FinalEdge> first;
    std::optional<FinalEdge> second;
    void insert(FinalEdge candidate)
    {
        // Each graph edge is visited once; equal scores retain both identities.
        if (!first || candidate.score < first->score) {
            second = first;
            first = candidate;
        } else if (!second || candidate.score < second->score) {
            second = candidate;
        }
    }
    std::optional<uint64_t> excluding(uint32_t identity) const
    {
        const auto& selected = first && first->identity != identity ? first : second;
        return selected ? std::optional<uint64_t>(selected->score) : std::nullopt;
    }
};
bool retain(const Graph& graph, const PeriodicAnalysis& output, uint32_t row,
            std::vector<bool>& redundant)
{
    const auto& frontier = output.frontiers[row];
    std::vector<BestTwo> incoming(graph.outgoing.size());
    for (uint32_t id = 0; id < graph.edges.size(); ++id) {
        const auto& edge = graph.edges[id];
        const auto distance = frontier.distances[edge.source];
        if (!distance) {
            continue;
        }
        uint64_t weight = 0, score = 0;
        if (!multiply(edge.weight, frontier.count, weight) || !add(*distance, weight, score)) {
            return false;
        }
        incoming[edge.target].insert({score, id});
    }
    for (std::size_t i = 0; i < output.generators.size(); ++i) {
        const auto& record = output.generators[i];
        if (output.sourceRows[record.source] != row) {
            continue;
        }
        const auto score = incoming[2 * record.target].excluding(graph.recordEdges[i]);
        if (score) {
            redundant[i] = threshold(*score, output.localRanks[record.source], frontier.count) <=
                record.displacement;
        }
    }
    return true;
}
} // namespace
bool computeFrontiers(const Graph& graph, PeriodicAnalysis& output)
{
    std::vector<bool> redundant(output.generators.size(), false);
    for (uint32_t row = 0; row < output.frontiers.size(); ++row) {
        if (!distances(graph, output, row, output.frontiers[row]) || !retain(graph, output, row, redundant)) {
            return false;
        }
    }
    for (uint32_t i = 0; i < redundant.size(); ++i) {
        if (!redundant[i]) {
            output.retained.push_back(i);
        }
    }
    return true;
}
} // namespace mlir::pto::frontiersynch::periodic
