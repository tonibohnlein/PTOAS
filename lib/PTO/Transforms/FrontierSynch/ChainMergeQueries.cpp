// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
namespace {
bool valid(const NumericalChainInterface& index, const std::vector<uint32_t>& thresholds)
{
    if (!index.error.empty() || thresholds.size() != index.chains.size()) { return false; }
    for (std::size_t c = 0; c < thresholds.size(); ++c) {
        if (thresholds[c] > index.chains[c].size()) { return false; }
    }
    return true;
}
std::optional<uint32_t> crossing(const NumericalCrossingGroup& group, uint32_t threshold,
                                  bool reverse, NumericalChainQueryCost& cost)
{
    const auto& values = reverse ? group.reverse : group.forward;
    auto found = std::lower_bound(values.begin(), values.end(), threshold, [&](const auto& pair, uint32_t value) {
        ++cost.indexOperations; return pair.first < value;
    });
    if (reverse) { return found == values.begin() ? std::nullopt : std::optional<uint32_t>((--found)->second); }
    return found == values.end() ? std::nullopt : std::optional<uint32_t>(found->second);
}
std::vector<uint32_t> absent(const NumericalChainInterface& index, bool reverse)
{
    std::vector<uint32_t> result(index.chains.size(), 0);
    if (!reverse) {
        for (std::size_t c = 0; c < result.size(); ++c) { result[c] = index.chains[c].size(); }
    }
    return result;
}
std::vector<uint32_t> across(const NumericalChainMerge& merge, const std::vector<uint32_t>& thresholds,
                             bool reverse, NumericalChainQueryCost& cost)
{
    const auto& other = *merge.children[reverse ? 0 : 1];
    auto frontier = absent(other, reverse);
    for (uint32_t c = 0; c < merge.crossings.size(); ++c) {
        for (uint32_t d = 0; d < merge.crossings[c].size(); ++d) {
            auto value = crossing(merge.crossings[c][d], thresholds[reverse ? d : c], reverse, cost);
            if (!value) { continue; }
            auto& entry = frontier[reverse ? c : d];
            entry = reverse ? std::max(entry, *value) : std::min(entry, *value); ++cost.indexOperations;
        }
    }
    auto result = absent(other, reverse);
    for (uint32_t c = 0; c < frontier.size(); ++c) {
        if (reverse ? frontier[c] == 0 : frontier[c] == other.chains[c].size()) { continue; }
        const auto id = other.chains[c][reverse ? frontier[c] - 1 : frontier[c]];
        const auto& row = reverse ? other.reverse[id] : other.forward[id];
        for (uint32_t d = 0; d < result.size(); ++d) {
            result[d] = reverse ? std::max(result[d], row[d]) : std::min(result[d], row[d]);
            ++cost.indexOperations;
        }
    }
    return result;
}
void project(const NumericalChainMerge& merge, uint32_t child, const std::vector<uint32_t>& thresholds,
             bool reverse, std::vector<uint32_t>& result, NumericalChainQueryCost& cost)
{
    for (uint32_t c = 0; c < thresholds.size(); ++c) {
        const auto& positions = merge.selectedRanks[child][c];
        auto found = std::lower_bound(positions.begin(), positions.end(), thresholds[c],
            [&](uint32_t rank, uint32_t bound) { ++cost.indexOperations; return rank < bound; });
        const auto offset = static_cast<std::size_t>(found - positions.begin());
        auto& target = result[merge.parentChains[child][c]];
        if (reverse && offset) { target = std::max(target, merge.parentRanks[child][c][offset - 1] + 1); }
        if (!reverse && found != positions.end()) { target = std::min(target, merge.parentRanks[child][c][offset]); }
        ++cost.indexOperations;
    }
}
} // namespace
std::optional<std::vector<uint32_t>> numericalLeafThresholds(const NumericalChainInterface& index,
    const std::function<std::optional<bool>(uint32_t, bool)>& query, bool reverse, NumericalChainQueryCost& cost)
{
    if (!index.error.empty() || !query) { return std::nullopt; }
    std::vector<uint32_t> result;
    for (const auto& chain : index.chains) {
        uint32_t begin = 0, end = chain.size();
        while (begin < end) {
            const auto middle = begin + (end - begin) / 2;
            auto answer = query(chain[middle], reverse); ++cost.leafQueries;
            if (!answer) { return std::nullopt; }
            if (*answer == reverse) { begin = middle + 1; } else { end = middle; }
        }
        result.push_back(begin);
    }
    return result;
}
std::optional<std::vector<uint32_t>> NumericalChainMerge::propagate(uint32_t child,
    const std::vector<uint32_t>& thresholds, bool reverse, NumericalChainQueryCost& cost) const
{
    if (!index.error.empty() || child > 1 || !children[child] || !valid(*children[child], thresholds)) {
        return std::nullopt;
    }
    auto result = absent(index, reverse);
    project(*this, child, thresholds, reverse, result, cost);
    if (child == (reverse ? 1U : 0U)) {
        auto other = across(*this, thresholds, reverse, cost);
        project(*this, 1 - child, other, reverse, result, cost);
    }
    return result;
}
std::optional<bool> NumericalChainMerge::crosses(const std::vector<uint32_t>& forward,
    const std::vector<uint32_t>& reverse, NumericalChainQueryCost& cost) const
{
    if (!index.error.empty() || !children[0] || !children[1] ||
        !valid(*children[0], forward) || !valid(*children[1], reverse)) { return std::nullopt; }
    for (uint32_t c = 0; c < crossings.size(); ++c) {
        for (uint32_t d = 0; d < crossings[c].size(); ++d) {
            auto first = crossing(crossings[c][d], forward[c], false, cost);
            ++cost.indexOperations;
            if (first && *first < reverse[d]) { return true; }
        }
    }
    return false;
}
} // namespace mlir::pto::frontiersynch
