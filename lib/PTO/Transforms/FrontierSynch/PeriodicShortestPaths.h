// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Private quotient graph helpers. The indexed heap holds at most one entry per
// vertex, keeping log(V) operations even with many parallel weighted records.
#ifndef PTO_FRONTIERSYNCH_PERIODICSHORTESTPATHS_H
#define PTO_FRONTIERSYNCH_PERIODICSHORTESTPATHS_H

#include "PTO/Transforms/FrontierSynch/PeriodicDemandAnalysis.h"
#include <utility>

namespace mlir::pto::frontiersynch::detail {
struct QuotientEdge {
    std::size_t source = 0;
    std::size_t target = 0;
    llvm::DynamicAPInt weight;
};
struct QuotientGraph {
    SmallVector<QuotientEdge> edges;
    SmallVector<SmallVector<std::size_t>> outgoing;
    explicit QuotientGraph(std::size_t vertices) : outgoing(vertices) {}
    void add(std::size_t source, std::size_t target, const llvm::DynamicAPInt& weight)
    {
        outgoing[source].push_back(edges.size());
        edges.push_back({source, target, weight});
    }
};

class DistanceHeap {
public:
    explicit DistanceHeap(ArrayRef<PeriodicDistance> distances)
        : values(distances), positions(distances.size(), distances.size())
    {}
    bool empty() const { return heap.empty(); }
    void decreased(std::size_t vertex)
    {
        if (positions[vertex] == values.size()) {
            positions[vertex] = heap.size();
            heap.push_back(vertex);
        }
        std::size_t child = positions[vertex];
        while (child != 0) {
            const std::size_t parent = (child - 1) / 2;
            if (!less(child, parent)) {
                break;
            }
            exchange(child, parent);
            child = parent;
        }
    }
    std::size_t pop()
    {
        const auto vertex = heap.front();
        exchange(0, heap.size() - 1);
        heap.pop_back();
        positions[vertex] = values.size();
        std::size_t parent = 0;
        // Check before doubling: an internal node is strictly below n/2.
        while (parent < heap.size() / 2) {
            std::size_t child = 2 * parent + 1;
            if (child + 1 < heap.size() && less(child + 1, child)) {
                ++child;
            }
            if (!less(child, parent)) {
                break;
            }
            exchange(child, parent);
            parent = child;
        }
        return vertex;
    }

private:
    bool less(std::size_t left, std::size_t right) const { return *values[heap[left]] < *values[heap[right]]; }
    void exchange(std::size_t left, std::size_t right)
    {
        std::swap(heap[left], heap[right]);
        positions[heap[left]] = left;
        positions[heap[right]] = right;
    }
    ArrayRef<PeriodicDistance> values;
    SmallVector<std::size_t> heap;
    SmallVector<std::size_t> positions;
};

// Seeds and weights are nonnegative; one heap slot per vertex. Scaling never
// expands a weight into edges or periods.
inline SmallVector<PeriodicDistance> shortestPaths(
    const QuotientGraph& graph, SmallVector<PeriodicDistance> distances, const llvm::DynamicAPInt& scale)
{
    DistanceHeap heap(distances);
    for (std::size_t vertex = 0; vertex < distances.size(); ++vertex) {
        if (distances[vertex]) {
            heap.decreased(vertex);
        }
    }
    while (!heap.empty()) {
        const auto vertex = heap.pop();
        for (auto id : graph.outgoing[vertex]) {
            const auto& edge = graph.edges[id];
            llvm::DynamicAPInt candidate = *distances[vertex] + scale * edge.weight;
            if (!distances[edge.target] || candidate < *distances[edge.target]) {
                distances[edge.target] = candidate;
                heap.decreased(edge.target);
            }
        }
    }
    return distances;
}

struct IncomingMinimum {
    std::size_t edge = 0;
    PeriodicDistance distance;
};
struct TwoIncomingMinima {
    IncomingMinimum first;
    IncomingMinimum second;
    // Each identity is visited once, so equal values still occupy both entries.
    void consider(std::size_t edge, const llvm::DynamicAPInt& value)
    {
        if (!first.distance || value < *first.distance) {
            second = first;
            first = {edge, value};
        } else if (!second.distance || value < *second.distance) {
            second = {edge, value};
        }
    }
    PeriodicDistance excluding(std::size_t edge) const
    {
        return first.distance && first.edge == edge ? second.distance : first.distance;
    }
};

inline SmallVector<TwoIncomingMinima> incomingMinima(
    const QuotientGraph& graph, ArrayRef<PeriodicDistance> distances, const llvm::DynamicAPInt& scale)
{
    SmallVector<TwoIncomingMinima> minima(graph.outgoing.size());
    for (auto [id, edge] : llvm::enumerate(graph.edges)) {
        if (distances[edge.source]) {
            minima[edge.target].consider(id, *distances[edge.source] + scale * edge.weight);
        }
    }
    return minima;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
