// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
// Numerical specialization only: symbolic guards retain the general query path.
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
namespace mlir::pto::frontiersynch {
bool SequenceAnalysisState::numericalCrossingReduction()
{
    if (children.size() != 2 || !portChoices.empty() || children[0].regional.referenceBefore ||
        children[1].regional.referenceBefore) {
        return false;
    }
    // Symbolic guards require the general reduction, which can discard
    // mutually exclusive candidates before asking any child queries. Check
    // this eligibility before constructing the numerical child interfaces.
    for (const auto& edge : crossings) {
        if (!expressions.constantValue(edge.guard)) { return false; }
    }
    if (auto incomingTarget = incoming.find(1); incomingTarget != incoming.end()) {
        for (const auto& link : incomingTarget->second) {
            if (!expressions.constantValue(link.guard)) { return false; }
        }
    }
    std::vector<uint32_t> ids[2];
    std::map<std::pair<uint32_t, unsigned>, std::vector<uint32_t>> lists[2];
    std::vector<uint32_t> local(2 * ports.size(), UINT32_MAX);
    for (uint32_t p = 0; p < ports.size(); ++p) {
        const auto& port = ports[p];
        if (port.child > 1 || !expressions.constantValue(port.ordinal) || !port.visits.empty() ||
            expressions.constantValue(present(p)) != 1) {
            return false;
        }
        for (unsigned kind = 0; kind < 2; ++kind) {
            local[2 * p + kind] = ids[port.child].size();
            lists[port.child][{pipe(p), kind}].push_back(local[2 * p + kind]);
            ids[port.child].push_back(2 * p + kind);
        }
    }
    NumericalChainInterface indices[2];
    for (unsigned child = 0; child < 2; ++child) {
        std::vector<std::vector<uint32_t>> chains;
        for (auto& [key, list] : lists[child]) {
            std::stable_sort(list.begin(), list.end(), [&](uint32_t a, uint32_t b) {
                const auto& x = ports[ids[child][a] / 2];
                const auto& y = ports[ids[child][b] / 2];
                return std::make_pair(*expressions.constantValue(x.ordinal), x.type) <
                       std::make_pair(*expressions.constantValue(y.ordinal), y.type);
            });
            // Coincident roles need identity normalization before chain sweeps.
            for (std::size_t i = 1; i < list.size(); ++i) {
                const auto& x = ports[ids[child][list[i - 1]] / 2];
                const auto& y = ports[ids[child][list[i]] / 2];
                if (x.type == y.type && expressions.constantValue(x.ordinal) == expressions.constantValue(y.ordinal)) {
                    return false;
                }
            }
            chains.push_back(std::move(list));
        }
        indices[child] =
            buildNumericalChainInterface(std::move(chains), [&](uint32_t a, uint32_t b) -> std::optional<bool> {
                auto answer = eventReachability(ids[child][a], ids[child][b]);
                auto value = answer ? expressions.constantValue(*answer) : std::nullopt;
                if (!value) {
                    return std::nullopt;
                }
                return *value != 0;
            });
        if (!indices[child].error.empty()) {
            return false;
        }
    }
    std::vector<NumericalCrossing> links;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> lookup;
    auto found = incoming.find(1);
    if (found != incoming.end()) {
        for (const auto& link : found->second) {
            auto guard = expressions.constantValue(link.guard);
            if (!guard || ports[link.source / 2].child != 0 || ports[link.target / 2].child != 1) {
                return false;
            }
            if (!*guard) {
                continue;
            }
            auto pair = std::make_pair(local[link.source], local[link.target]);
            if (lookup.emplace(pair, links.size()).second) {
                links.push_back({pair.first, pair.second});
            }
        }
    }
    auto retained = reduceNumericalCrossings(indices[0], indices[1], links);
    if (!retained) {
        return false;
    }
    std::vector<Expr> guards;
    for (const auto& edge : crossings) {
        auto guard = expressions.constantValue(edge.guard);
        if (!guard || ports[edge.source].child != 0 || ports[edge.target].child != 1) {
            return false;
        }
        auto found = lookup.find({local[2 * edge.source + 1], local[2 * edge.target]});
        if (*guard && found == lookup.end()) {
            return false;
        }
        guards.push_back(expressions.boolean(*guard && (*retained)[found->second]));
    }
    for (std::size_t i = 0; i < guards.size(); ++i) {
        crossings[i].guard = guards[i];
    }
    return true;
}
} // namespace mlir::pto::frontiersynch
