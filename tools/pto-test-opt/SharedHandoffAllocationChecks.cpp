// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent exhaustive chain partitions for small causal lifetime orders.
#include "PTO/Transforms/FrontierSynch/SharedHandoffAllocation.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <set>
namespace fs = mlir::pto::frontiersynch;
namespace {
using Matrix = std::vector<std::vector<uint8_t>>;
using Edges = std::vector<std::vector<uint32_t>>;
Edges adjacency(const Matrix& order)
{
    Edges edges(order.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        for (uint32_t j = 0; j < order.size(); ++j) {
            if (order[i][j]) { edges[i].push_back(j); }
        }
    }
    return edges;
}
unsigned bruteMinimum(const Matrix& order)
{
    unsigned best = order.size();
    std::vector<std::vector<unsigned>> chains;
    // Enumerate set partitions, checking every pair in each candidate chain.
    // This uses no matching, source order, or greedy lane choice.
    std::function<void(unsigned)> visit = [&](unsigned next) {
        if (next == order.size()) { best = std::min(best, unsigned(chains.size())); return; }
        if (chains.size() >= best) { return; }
        for (std::size_t lane = 0; lane < chains.size(); ++lane) {
            bool comparable = std::all_of(chains[lane].begin(), chains[lane].end(), [&](unsigned prior) {
                return order[prior][next] || order[next][prior];
            });
            if (!comparable) { continue; }
            chains[lane].push_back(next); visit(next + 1); chains[lane].pop_back();
        }
        chains.push_back({next}); visit(next + 1); chains.pop_back();
    };
    visit(0);
    return best;
}
bool valid(const fs::SharedHandoffAllocation& result, const Matrix& order)
{
    if (!result.error.empty() || result.lanes.size() != order.size()) { return false; }
    std::set<uint32_t> used;
    for (unsigned i = 0; i < order.size(); ++i) {
        if (result.lanes[i] >= result.budget) { return false; }
        used.insert(result.lanes[i]);
        for (unsigned j = 0; j < i; ++j) {
            if (result.lanes[i] == result.lanes[j] && !order[i][j] && !order[j][i]) { return false; }
        }
    }
    return used.size() == result.budget;
}
bool exhaustive()
{
    constexpr unsigned count = 5, choices = count * (count - 1) / 2;
    std::vector<fs::SharedHandoff> handoffs;
    for (unsigned i = 0; i < count; ++i) { handoffs.push_back({i % 3, (i + 1) % 3}); }
    for (unsigned mask = 0; mask < (1U << choices); ++mask) {
        Matrix order(count, std::vector<uint8_t>(count));
        unsigned bit = 0;
        for (unsigned i = 0; i < count; ++i) {
            for (unsigned j = i + 1; j < count; ++j) { order[i][j] = (mask >> bit++) & 1U; }
        }
        for (unsigned k = 0; k < count; ++k) {
            for (unsigned i = 0; i < count; ++i) {
                for (unsigned j = 0; j < count; ++j) { order[i][j] |= order[i][k] && order[k][j]; }
            }
        }
        const auto queryWidth = bruteMinimum(order);
        for (uint64_t capacity : {uint64_t(0), uint64_t(queryWidth), uint64_t(count)}) {
            uint64_t queries = 0;
            auto queried = fs::allocateSharedHandoffsByQuery(handoffs, [&](uint32_t from, uint32_t to) {
                ++queries; return order[from][to] != 0;
            }, capacity);
            if (!valid(queried, order) || (queried.exactMinimum && queried.budget != queryWidth) ||
                (capacity == queryWidth && queried.budget != queryWidth) ||
                (!queried.exactMinimum && queries > capacity * count)) { return false; }
        }
        // Reverse numerical IDs to exercise non-reference-ordered input too.
        Matrix reversed(count, std::vector<uint8_t>(count));
        for (unsigned i = 0; i < count; ++i) {
            for (unsigned j = 0; j < count; ++j) { reversed[count - 1 - i][count - 1 - j] = order[i][j]; }
        }
        for (const auto& relation : {order, reversed}) {
            const auto edges = adjacency(relation);
            const auto width = bruteMinimum(relation);
            auto exact = fs::allocateSharedHandoffs(handoffs, edges, 0);
            auto fit = fs::allocateSharedHandoffs(handoffs, edges, width);
            auto scarce = fs::allocateSharedHandoffs(handoffs, edges, width - 1);
            auto greedy = fs::allocateSharedHandoffs(handoffs, edges, count);
            auto repeated = fs::allocateSharedHandoffs(handoffs, edges, 0);
            if (!valid(exact, relation) || !exact.exactMinimum || exact.budget != width ||
                !valid(fit, relation) || fit.budget != width || !valid(scarce, relation) ||
                !scarce.exactMinimum || scarce.budget != width || !valid(greedy, relation) ||
                greedy.exactMinimum || exact.lanes != repeated.lanes) { return false; }
        }
    }
    return true;
}
bool examples()
{
    // Greedy chooses 0->2 and gets stuck at 3; matching uses 0->3 and 1->2.
    std::vector<fs::SharedHandoff> handoffs{{0, 1}, {2, 3}, {0, 1}, {3, 0}};
    Edges edges{{2, 3}, {2}, {}, {}};
    auto greedy = fs::allocateSharedHandoffs(handoffs, edges, 4);
    auto exact = fs::allocateSharedHandoffs(handoffs, edges, 2);
    if (greedy.budget != 3 || greedy.exactMinimum || exact.budget != 2 || !exact.exactMinimum) { return false; }
    // Same-direction preference chooses lane 1 even though lane 0 is also free.
    std::vector<fs::SharedHandoff> directions{{0, 1}, {1, 2}, {1, 2}, {0, 1}};
    auto preference = fs::allocateSharedHandoffs(directions, {{2, 3}, {2, 3}, {}, {}}, 4);
    if (preference.lanes != std::vector<uint32_t>({0, 1, 1, 0})) { return false; }
    // Direction labels and a shared command pipe never themselves prove reuse.
    std::vector<fs::SharedHandoff> chain{{0, 1}, {1, 2}};
    auto incomparable = fs::allocateSharedHandoffs(chain, {{}, {}}, 1);
    auto commandOrdered = fs::allocateSharedHandoffs(chain, {{1}, {}}, 1);
    if (incomparable.budget != 2 || !incomparable.exactMinimum || commandOrdered.budget != 1) { return false; }
    auto empty = fs::allocateSharedHandoffs({}, {}, 0);
    auto singleton = fs::allocateSharedHandoffs({{0, 1}}, {{}}, 0);
    return empty.error.empty() && empty.exactMinimum && !empty.budget && empty.lanes.empty() &&
           singleton.error.empty() && singleton.exactMinimum && singleton.budget == 1;
}
bool queryExamples()
{
    // A long causal chain must stay on the O(Eh) path: constructing its dense
    // relation would require millions of queries, not one query per successor.
    constexpr unsigned count = 4096;
    std::vector<fs::SharedHandoff> chain(count, {0, 1});
    uint64_t queries = 0;
    auto simple = fs::allocateSharedHandoffsByQuery(chain, [&](uint32_t from, uint32_t to) {
        ++queries; return from < to;
    }, 6);
    if (!simple.error.empty() || simple.exactMinimum || simple.budget != 1 || queries != count - 1) { return false; }
    std::vector<fs::SharedHandoff> handoffs{{0, 1}, {2, 3}, {0, 1}, {3, 0}};
    auto reuse = [](uint32_t from, uint32_t to) { return (from == 0 && to >= 2) || (from == 1 && to == 2); };
    auto exact = fs::allocateSharedHandoffsByQuery(handoffs, reuse, 2);
    if (!exact.error.empty() || !exact.exactMinimum || exact.budget != 2) { return false; }
    queries = 0;
    auto scarce = fs::allocateSharedHandoffsByQuery(std::vector<fs::SharedHandoff>(8, {0, 1}),
        [&](uint32_t, uint32_t) { ++queries; return false; }, 6);
    // Stop greedy at its seventh handoff (21 queries), then inspect 28 pairs.
    if (!scarce.error.empty() || !scarce.exactMinimum || scarce.budget != 8 || queries != 49) { return false; }
    auto empty = fs::allocateSharedHandoffsByQuery({}, reuse, 0);
    auto missing = fs::allocateSharedHandoffsByQuery(handoffs, {}, 2);
    return empty.error.empty() && empty.exactMinimum && !empty.budget && !missing.error.empty();
}
bool directedNamespaces()
{
    std::vector<fs::SharedHandoff> handoffs{{0, 1}, {1, 0}, {0, 1}};
    auto separate = fs::allocateDirectedHandoffs(handoffs, {{2}, {}, {}}, 1);
    auto scarce = fs::allocateDirectedHandoffs(handoffs, {{}, {}, {}}, 1);
    unsigned queries = 0;
    auto queried = fs::allocateDirectedHandoffsByQuery(handoffs, [&](uint32_t from, uint32_t to) {
        ++queries; return from == 0 && to == 2;
    }, 1);
    return separate.error.empty() && separate.budget == 1 &&
        separate.lanes == std::vector<uint32_t>({0, 0, 0}) &&
        scarce.error.empty() && scarce.budget == 2 &&
        queried.error.empty() && queried.budget == 1 && queries == 1;
}
bool invalid()
{
    std::vector<fs::SharedHandoff> handoffs(3, {0, 1});
    for (const auto& edges : std::vector<Edges>{{{}, {}}, {{3}, {}, {}}, {{0}, {}, {}},
                                               {{1}, {2}, {0}}, {{1}, {2}, {}}}) {
        auto result = fs::allocateSharedHandoffs(handoffs, edges, 3);
        if (result.error.empty() || !result.lanes.empty()) { return false; }
    }
    auto duplicates = fs::allocateSharedHandoffs(handoffs, {{2, 1, 1, 2}, {2}, {}}, 1);
    return duplicates.error.empty() && duplicates.budget == 1;
}
} // namespace
int runSharedHandoffAllocationChecks()
{
    if (!examples() || !queryExamples() || !invalid() || !exhaustive() || !directedNamespaces()) {
        llvm::errs() << "shared handoff allocation check failed\n";
        return 1;
    }
    llvm::outs() << "shared handoff allocation: greedy fallback and 2048 orders match brute chain partitions\n";
    return 0;
}
