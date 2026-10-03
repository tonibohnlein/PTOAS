// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/OneWayRepair.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <deque>
#include <limits>
#include <optional>
#include <utility>
#include <vector>
namespace mlir::pto::frontiersynch {
namespace {
using Int = llvm::DynamicAPInt;
struct Line {
    Int slope, intercept;
    std::size_t cut;
    Int at(const Int& x) const { return slope * x + intercept; }
};
bool redundant(const Line& a, const Line& b, const Line& c)
{
    // Strictly decreasing slopes give positive denominators. Compare the
    // two intersection abscissae exactly, including negative intersections.
    return (b.intercept - a.intercept) * (b.slope - c.slope) >=
           (c.intercept - b.intercept) * (a.slope - b.slope);
}
} // namespace
OneWayRepair repairOneWay(llvm::ArrayRef<OneWayCover> covers,
    const Int& nP, const Int& nQ, llvm::ArrayRef<unsigned> eligibleIds)
{
    OneWayRepair result;
    if (nP < Int(0) || nQ < Int(0)) {
        result.reason = "one-way payload counts must be nonnegative"; return result;
    }
    llvm::DenseSet<unsigned> uniqueIds;
    for (auto id : eligibleIds) {
        if (!uniqueIds.insert(id).second) {
            result.reason = "one-way eligible IDs must be distinct"; return result;
        }
    }
    Int previousSource(0), previousConsumer(0);
    for (const auto& cover : covers) {
        if (cover.sourceRank <= previousSource || cover.consumerRank <= previousConsumer ||
            cover.sourceRank > nP || cover.consumerRank > nQ) {
            result.reason = "one-way covers require bounded strictly increasing positive ranks"; return result;
        }
        previousSource = cover.sourceRank; previousConsumer = cover.consumerRank;
    }
    if (covers.empty()) { result.status = OneWayRepairStatus::Ready; return result; }
    if (eligibleIds.empty()) {
        result.status = OneWayRepairStatus::NoCapacity;
        result.reason = "one-way selected staircase needs at least one eligible ID; available 0";
        return result;
    }
    const auto m = covers.size(), t = std::min(m, eligibleIds.size());
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (m == limit || t == limit || (t + 1) > limit / (m + 1) ||
        (t + 1) * (m + 1) > limit / sizeof(std::size_t) ||
        m + 1 > std::vector<Int>().max_size() ||
        m + 1 > std::vector<std::optional<Int>>().max_size() ||
        m + 1 > std::vector<std::size_t>().max_size() ||
        t + 1 > std::vector<std::vector<std::size_t>>().max_size()) {
        result.reason = "one-way partition storage size is not representable"; return result;
    }
    std::vector<Int> weights(m + 1, Int(0)), products(m + 1, Int(0));
    for (std::size_t v = 1; v <= m; ++v) {
        const auto next = v == m ? nQ + Int(1) : covers[v].consumerRank;
        const auto weight = next - covers[v - 1].consumerRank;
        weights[v] = weights[v - 1] + weight;
        products[v] = products[v - 1] + weight * covers[v - 1].sourceRank;
    }
    std::vector<std::optional<Int>> previous(m + 1), current(m + 1);
    std::vector<std::vector<std::size_t>> cuts(t + 1, std::vector<std::size_t>(m + 1, 0));
    previous[0] = Int(0);
    for (std::size_t layer = 1; layer <= t; ++layer) {
        std::fill(current.begin(), current.end(), std::nullopt);
        std::deque<Line> envelope;
        for (std::size_t v = layer; v <= m; ++v) {
            const auto q = v - 1;
            if (previous[q]) {
                Line line{-weights[q], *previous[q] + products[q], q};
                while (envelope.size() >= 2 &&
                       redundant(envelope[envelope.size() - 2], envelope.back(), line)) {
                    envelope.pop_back();
                }
                envelope.push_back(std::move(line));
            }
            if (envelope.empty()) {
                result.reason = "one-way partition has no feasible last cut"; return result;
            }
            const auto& rank = covers[v - 1].sourceRank;
            // Keeping the older line on equality selects the smaller cut.
            while (envelope.size() >= 2 && envelope[1].at(rank) < envelope.front().at(rank)) {
                envelope.pop_front();
            }
            current[v] = rank * weights[v] - products[v] + envelope.front().at(rank);
            cuts[layer][v] = envelope.front().cut;
        }
        previous.swap(current);
    }
    if (!previous[m]) {
        result.reason = "one-way final partition value unavailable"; return result;
    }
    result.excess = *previous[m];
    result.blocks.resize(t);
    std::size_t end = m;
    for (std::size_t layer = t; layer > 0; --layer) {
        const auto q = cuts[layer][end];
        if (q >= end || q < layer - 1) {
            result.blocks.clear(); result.reason = "one-way reconstruction cut invalid"; return result;
        }
        result.blocks[layer - 1] = {q, end - 1, eligibleIds[layer - 1]};
        end = q;
    }
    if (end) {
        result.blocks.clear(); result.reason = "one-way reconstruction did not cover the staircase"; return result;
    }
    result.status = OneWayRepairStatus::Ready;
    return result;
}
} // namespace mlir::pto::frontiersynch
