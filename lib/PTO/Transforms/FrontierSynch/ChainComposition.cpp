// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "ChainInterfaceInternal.h"
#include <algorithm>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Rows = std::vector<std::vector<uint32_t>>;
struct Projection {
    std::array<std::vector<uint32_t>, 2> selected;
    std::array<Rows, 2> next;
};
bool validIndex(const NumericalChainInterface& index)
{
    const auto size = index.chain.size(), chains = index.chains.size();
    if (!index.error.empty() || size > UINT32_MAX || chains > UINT32_MAX || index.rank.size() != size ||
        index.forward.size() != size || index.reverse.size() != size) { return false; }
    uint64_t seen = 0;
    for (uint32_t c = 0; c < chains; ++c) {
        if (index.chains[c].size() > size) { return false; }
        for (uint32_t r = 0; r < index.chains[c].size(); ++r) {
            auto id = index.chains[c][r]; ++seen;
            if (id >= size || index.chain[id] != c || index.rank[id] != r) { return false; }
        }
    }
    if (seen != size) { return false; }
    for (uint32_t id = 0; id < size; ++id) {
        if (index.forward[id].size() != chains || index.reverse[id].size() != chains) { return false; }
        for (uint32_t c = 0; c < chains; ++c) {
            if (index.forward[id][c] > index.chains[c].size() || index.reverse[id][c] > index.chains[c].size()) {
                return false;
            }
        }
    }
    return true;
}
bool initialize(NumericalChainMerge& out, Projection& projection,
                const std::vector<NumericalChainSelection>& selection)
{
    if (!out.children[0] || !out.children[1] || selection.size() > UINT32_MAX) { return false; }
    uint64_t chainCount = 0;
    std::set<uint32_t> directory;
    for (unsigned child = 0; child < 2; ++child) {
        const auto& index = *out.children[child];
        const auto& maps = out.parentChains[child];
        if (!validIndex(index) || maps.size() != index.chains.size() ||
            std::set<uint32_t>(maps.begin(), maps.end()).size() != maps.size()) { return false; }
        for (auto target : maps) {
            chainCount = std::max(chainCount, uint64_t(target) + 1); directory.insert(target);
        }
        projection.selected[child].assign(index.chain.size(), UINT32_MAX);
    }
    if (chainCount > UINT32_MAX || directory.size() != chainCount) { return false; }
    for (uint32_t id = 0; id < selection.size(); ++id) {
        const auto value = selection[id];
        if (value.child > 1 || value.event >= projection.selected[value.child].size() ||
            projection.selected[value.child][value.event] != UINT32_MAX) { return false; }
        projection.selected[value.child][value.event] = id;
    }
    Rows chains(chainCount);
    for (unsigned child = 0; child < 2; ++child) {
        const auto& index = *out.children[child];
        out.selectedRanks[child].resize(index.chains.size()); out.parentRanks[child].resize(index.chains.size());
        for (uint32_t c = 0; c < index.chains.size(); ++c) {
            auto& chain = chains[out.parentChains[child][c]];
            for (uint32_t rank = 0; rank < index.chains[c].size(); ++rank) {
                ++out.index.operations;
                auto id = projection.selected[child][index.chains[c][rank]];
                if (id == UINT32_MAX) { continue; }
                out.selectedRanks[child][c].push_back(rank);
                out.parentRanks[child][c].push_back(chain.size()); chain.push_back(id);
            }
        }
    }
    auto operations = out.index.operations;
    out.index = chain::initialize(std::move(chains)); out.index.operations += operations;
    return out.index.error.empty();
}
void projectRows(NumericalChainMerge& out, Projection& projection,
                 const std::vector<NumericalChainSelection>& selection)
{
    for (auto& row : out.index.forward) {
        for (uint32_t c = 0; c < row.size(); ++c) { row[c] = out.index.chains[c].size(); ++out.index.operations; }
    }
    for (unsigned child = 0; child < 2; ++child) {
        const auto& index = *out.children[child];
        projection.next[child].resize(index.chains.size());
        for (uint32_t c = 0; c < index.chains.size(); ++c) {
            auto target = out.parentChains[child][c];
            auto& next = projection.next[child][c];
            next.assign(index.chains[c].size() + 1, out.index.chains[target].size());
            for (std::size_t r = index.chains[c].size(); r; --r) {
                auto id = projection.selected[child][index.chains[c][r - 1]];
                next[r - 1] = id == UINT32_MAX ? next[r] : out.index.rank[id]; ++out.index.operations;
            }
        }
    }
    for (uint32_t id = 0; id < selection.size(); ++id) {
        auto selected = selection[id];
        const auto& index = *out.children[selected.child];
        for (uint32_t c = 0; c < index.chains.size(); ++c) {
            out.index.forward[id][out.parentChains[selected.child][c]] =
                projection.next[selected.child][c][index.forward[selected.event][c]];
            ++out.index.operations;
        }
    }
}
void activateCrossings(NumericalChainMerge& out, const Projection& projection,
                       const std::vector<NumericalCrossing>& edges)
{
    const auto& left = *out.children[0]; const auto& right = *out.children[1];
    for (uint32_t c = 0; c < left.chains.size(); ++c) {
        std::vector<std::vector<uint32_t>> buckets(left.chains[c].size());
        for (uint32_t edge = 0; edge < edges.size(); ++edge) {
            const auto last = left.reverse[edges[edge].source][c]; ++out.index.operations;
            if (last) { buckets[last - 1].push_back(edge); }
        }
        std::vector<uint32_t> best(right.chains.size());
        for (uint32_t d = 0; d < best.size(); ++d) {
            best[d] = out.index.chains[out.parentChains[1][d]].size();
        }
        for (std::size_t r = buckets.size(); r; --r) {
            for (auto edge : buckets[r - 1]) {
                for (uint32_t d = 0; d < best.size(); ++d) {
                    best[d] = std::min(best[d], projection.next[1][d][right.forward[edges[edge].target][d]]);
                    ++out.index.operations;
                }
            }
            auto id = projection.selected[0][left.chains[c][r - 1]];
            if (id == UINT32_MAX) { continue; }
            for (uint32_t d = 0; d < best.size(); ++d) {
                auto& value = out.index.forward[id][out.parentChains[1][d]];
                value = std::min(value, best[d]); ++out.index.operations;
            }
        }
    }
}
void indexCrossings(NumericalChainMerge& out, const std::vector<NumericalCrossing>& edges)
{
    const auto& left = *out.children[0]; const auto& right = *out.children[1];
    out.crossings.resize(left.chains.size(), std::vector<NumericalCrossingGroup>(right.chains.size()));
    std::vector<std::vector<uint32_t>> bySource(left.chain.size()), byTarget(right.chain.size());
    for (uint32_t id = 0; id < edges.size(); ++id) {
        bySource[edges[id].source].push_back(id); byTarget[edges[id].target].push_back(id); ++out.index.operations;
    }
    for (uint32_t c = 0; c < left.chains.size(); ++c) {
        for (auto source : left.chains[c]) {
            for (auto id : bySource[source]) {
                auto target = edges[id].target;
                out.crossings[c][right.chain[target]].forward.push_back({left.rank[source], right.rank[target]});
                ++out.index.operations;
            }
        }
    }
    for (uint32_t d = 0; d < right.chains.size(); ++d) {
        for (auto target : right.chains[d]) {
            for (auto id : byTarget[target]) {
                auto source = edges[id].source;
                out.crossings[left.chain[source]][d].reverse.push_back({right.rank[target], left.rank[source] + 1});
                ++out.index.operations;
            }
        }
    }
    for (auto& row : out.crossings) {
        for (auto& group : row) {
            for (std::size_t i = group.forward.size(); i > 1; --i) {
                group.forward[i - 2].second = std::min(group.forward[i - 2].second, group.forward[i - 1].second);
                ++out.index.operations;
            }
            for (std::size_t i = 1; i < group.reverse.size(); ++i) {
                group.reverse[i].second = std::max(group.reverse[i].second, group.reverse[i - 1].second);
                ++out.index.operations;
            }
        }
    }
}
} // namespace
NumericalChainMerge buildNumericalChainMerge(
    std::shared_ptr<const NumericalChainInterface> left, std::shared_ptr<const NumericalChainInterface> right,
    const std::vector<NumericalCrossing>& edges, const std::vector<NumericalChainSelection>& selection,
    const std::vector<uint32_t>& leftChains, const std::vector<uint32_t>& rightChains)
{
    NumericalChainMerge out;
    out.children = {std::move(left), std::move(right)}; out.parentChains = {leftChains, rightChains};
    Projection projection;
    if (!initialize(out, projection, selection)) { out.index.error = "invalid numerical merge projection"; return out; }
    auto retained = chain::reduce(*out.children[0], *out.children[1], edges, out.index.operations);
    if (!retained) { out.index.error = "invalid numerical merge crossings"; return out; }
    projectRows(out, projection, selection);
    activateCrossings(out, projection, edges);
    chain::invert(out.index);
    std::vector<NumericalCrossing> reduced;
    for (std::size_t i = 0; i < edges.size(); ++i) { if ((*retained)[i]) { reduced.push_back(edges[i]); } }
    indexCrossings(out, reduced);
    return out;
}
} // namespace mlir::pto::frontiersynch
