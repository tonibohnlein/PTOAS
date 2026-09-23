// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared original-control SCC query; no selected commands or causal credit.
#ifndef PTO_OAHS_CONTROL_COMPONENTS_H
#define PTO_OAHS_CONTROL_COMPONENTS_H
#include "Control.h"
#include <algorithm>

namespace mlir::pto::oahs::detail {
// Iterative DFS avoids making compiler stack depth depend on source nesting.
inline std::vector<std::size_t> finishingOrder(const ControlGraph& graph, const std::vector<bool>& reachable)
{
    std::vector<bool> seen(graph.sites.size());
    std::vector<std::size_t> order;
    for (std::size_t root = 0; root < graph.sites.size(); ++root) {
        if (!reachable[root] || seen[root]) {
            continue;
        }
        std::vector<std::pair<std::size_t, std::size_t>> stack{{root, 0}};
        seen[root] = true;
        while (!stack.empty()) {
            auto& frame = stack.back();
            const auto& successors = graph.sites[frame.first].successors;
            if (frame.second == successors.size()) {
                order.push_back(frame.first);
                stack.pop_back();
                continue;
            }
            const auto next = successors[frame.second++];
            if (!seen[next]) {
                seen[next] = true;
                stack.emplace_back(next, 0);
            }
        }
    }
    return order;
}
inline std::vector<std::vector<std::size_t>> strongComponents(
    const ControlGraph& graph, const std::vector<std::vector<std::size_t>>& predecessors,
    const std::vector<bool>& reachable, std::vector<std::size_t>& membership)
{
    auto order = finishingOrder(graph, reachable);
    std::vector<std::vector<std::size_t>> groups;
    membership.assign(graph.sites.size(), NoAnalysisId);
    for (auto at = order.rbegin(); at != order.rend(); ++at) {
        if (membership[*at] != NoAnalysisId) {
            continue;
        }
        const auto group = groups.size();
        groups.emplace_back();
        std::vector<std::size_t> todo{*at};
        membership[*at] = group;
        while (!todo.empty()) {
            const auto site = todo.back();
            todo.pop_back();
            groups.back().push_back(site);
            for (auto before : predecessors[site]) {
                if (reachable[before] && membership[before] == NoAnalysisId) {
                    membership[before] = group;
                    todo.push_back(before);
                }
            }
        }
        std::sort(groups.back().begin(), groups.back().end());
    }
    return groups;
}
} // namespace mlir::pto::oahs::detail
#endif
