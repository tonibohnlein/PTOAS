// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/SourcePipeServices.h"
#include <algorithm>
#include <deque>
#include <unordered_set>
namespace mlir::pto::frontiersynch {
std::size_t SourcePipeFactHash::operator()(const SourcePipeFact& fact) const
{
    auto seed = std::hash<std::size_t>{}(fact.cell);
    seed ^= std::hash<uint32_t>{}(fact.pipe) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
    return seed ^ std::hash<uint8_t>{}(fact.mode);
}
namespace {
std::string validate(
    const std::vector<SourcePipeNode>& graph, const std::vector<SourcePipeSite>& sites, std::size_t classes)
{
    for (const auto& site : sites) {
        if (site.modes.size() != classes) {
            return "inconsistent class inventory";
        }
        for (auto mode : site.modes) {
            if (mode > 3) {
                return "invalid read/write mode";
            }
        }
        for (auto cell : site.refreshed) {
            if (cell >= classes) {
                return "invalid class-wide refresh";
            }
        }
    }
    for (const auto& node : graph) {
        if (node.site && *node.site >= sites.size()) {
            return "invalid CFG site";
        }
        std::unordered_set<std::size_t> edges;
        for (auto next : node.successors) {
            if (next >= graph.size() || !edges.insert(next).second) {
                return "invalid or repeated CFG edge";
            }
        }
    }
    return {};
}
} // namespace
SourcePipeServices solveSourcePipeServices(
    const std::vector<SourcePipeNode>& graph, const std::vector<SourcePipeSite>& sites, std::size_t classes)
{
    SourcePipeServices result;
    result.error = validate(graph, sites, classes);
    if (!result.error.empty()) {
        return result;
    }
    result.incoming.resize(graph.size());
    result.outgoing.resize(graph.size());
    result.possible.resize(sites.size());
    std::vector<std::unordered_set<std::size_t>> kills;
    for (const auto& site : sites) {
        kills.emplace_back(site.refreshed.begin(), site.refreshed.end());
        result.work.preparedModes += site.modes.size();
    }
    std::deque<std::pair<std::size_t, SourcePipeFact>> queue;
    auto publish = [&](std::size_t node, SourcePipeFact fact) {
        if (result.outgoing[node].insert(fact).second) {
            ++result.work.outgoing;
            queue.emplace_back(node, fact);
        }
    };
    for (std::size_t node = 0; node < graph.size(); ++node) {
        result.work.preparedEdges += graph[node].successors.size();
        if (!graph[node].site) {
            continue;
        }
        const auto& site = sites[*graph[node].site];
        for (std::size_t cell = 0; cell < classes; ++cell) {
            for (uint8_t mode : {uint8_t(1), uint8_t(2)}) {
                if (site.modes[cell] & mode) {
                    publish(node, {cell, site.pipe, mode});
                    ++result.work.generated;
                }
            }
        }
    }
    while (!queue.empty()) {
        auto [node, fact] = queue.front();
        queue.pop_front();
        for (auto next : graph[node].successors) {
            ++result.work.transmissions;
            if (!result.incoming[next].insert(fact).second) {
                continue;
            }
            ++result.work.incoming;
            if (graph[next].site && kills[*graph[next].site].count(fact.cell)) {
                ++result.work.killed;
            } else {
                publish(next, fact);
            }
        }
    }
    std::vector<std::unordered_set<uint32_t>> selected(sites.size());
    for (std::size_t node = 0; node < graph.size(); ++node) {
        if (!graph[node].site) {
            continue;
        }
        auto site = *graph[node].site;
        for (const auto& fact : result.incoming[node]) {
            ++result.work.inspected;
            auto target = sites[site].modes[fact.cell];
            if (target && (fact.mode == 2 || (target & 2))) {
                selected[site].insert(fact.pipe);
            }
        }
    }
    for (std::size_t site = 0; site < sites.size(); ++site) {
        result.possible[site].assign(selected[site].begin(), selected[site].end());
        std::sort(result.possible[site].begin(), result.possible[site].end());
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
