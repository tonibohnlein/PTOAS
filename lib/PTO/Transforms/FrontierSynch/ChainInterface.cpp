// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
#include <algorithm>
#include <limits>
namespace mlir::pto::frontiersynch {
bool NumericalChainInterface::reaches(uint32_t source, uint32_t target) const
{
    return error.empty() && source < chain.size() && target < chain.size() &&
           forward[source][chain[target]] <= rank[target];
}
NumericalChainInterface buildNumericalChainInterface(
    std::vector<std::vector<uint32_t>> chains, const std::function<std::optional<bool>(uint32_t, uint32_t)>& query)
{
    NumericalChainInterface out;
    out.chains = std::move(chains);
    if (out.chains.size() > UINT32_MAX) {
        out.error = "chain count overflow";
        return out;
    }
    std::size_t count = 0;
    for (const auto& chain : out.chains) {
        if (chain.size() > UINT32_MAX - count) {
            out.error = "chain event count overflow";
            return out;
        }
        count += chain.size();
    }
    out.chain.assign(count, UINT32_MAX);
    out.rank.resize(count);
    for (uint32_t c = 0; c < out.chains.size(); ++c) {
        for (uint32_t r = 0; r < out.chains[c].size(); ++r) {
            auto id = out.chains[c][r];
            if (id >= count || out.chain[id] != UINT32_MAX) {
                out.error = "chain identities must be unique dense IDs";
                return out;
            }
            out.chain[id] = c;
            out.rank[id] = r;
        }
    }
    out.forward.assign(count, std::vector<uint32_t>(out.chains.size()));
    out.reverse.assign(count, std::vector<uint32_t>(out.chains.size()));
    for (uint32_t c = 0; c < out.chains.size(); ++c) {
        for (uint32_t d = 0; d < out.chains.size(); ++d) {
            const auto& source = out.chains[c];
            const auto& target = out.chains[d];
            uint32_t cursor = 0;
            for (auto id : source) {
                while (cursor < target.size()) {
                    auto answer = query(id, target[cursor]);
                    ++out.queries;
                    if (!answer) {
                        out.error = "child chain query unavailable";
                        return out;
                    }
                    if (*answer) {
                        break;
                    }
                    ++cursor;
                }
                out.forward[id][d] = cursor;
            }
            cursor = 0;
            for (uint32_t j = 0; j < target.size(); ++j) {
                while (cursor < source.size() && out.forward[source[cursor]][d] <= j) {
                    ++cursor;
                }
                out.reverse[target[j]][c] = cursor;
            }
        }
    }
    return out;
}
std::optional<std::vector<bool>> reduceNumericalCrossings(
    const NumericalChainInterface& left, const NumericalChainInterface& right,
    const std::vector<NumericalCrossing>& edges)
{
    if (!left.error.empty() || !right.error.empty() || edges.size() > UINT32_MAX) {
        return std::nullopt;
    }
    for (const auto& edge : edges) {
        if (edge.source >= left.chain.size() || edge.target >= right.chain.size()) {
            return std::nullopt;
        }
    }
    std::vector<bool> retained(edges.size(), true);
    for (uint32_t c = 0; c < left.chains.size(); ++c) {
        for (uint32_t d = 0; d < right.chains.size(); ++d) {
            const auto length = left.chains[c].size();
            std::vector<std::vector<uint32_t>> points(length), tests(length);
            for (uint32_t id = 0; id < edges.size(); ++id) {
                const auto& edge = edges[id];
                if (left.chain[edge.source] == c && right.chain[edge.target] == d) {
                    points[left.rank[edge.source]].push_back(id);
                }
                const auto threshold = left.forward[edge.source][c];
                if (threshold < length && right.reverse[edge.target][d]) {
                    tests[threshold].push_back(id);
                }
            }
            using Candidate = std::pair<uint32_t, uint32_t>;
            const Candidate absent{UINT32_MAX, UINT32_MAX};
            Candidate first = absent, second = absent;
            for (std::size_t position = length; position; --position) {
                for (auto id : points[position - 1]) {
                    Candidate next{right.rank[edges[id].target], id};
                    if (next < first) {
                        second = first;
                        first = next;
                    } else if (next < second) {
                        second = next;
                    }
                }
                for (auto id : tests[position - 1]) {
                    const auto other = first.second == id ? second : first;
                    if (other.first < right.reverse[edges[id].target][d]) {
                        retained[id] = false;
                    }
                }
            }
        }
    }
    return retained;
}
} // namespace mlir::pto::frontiersynch
