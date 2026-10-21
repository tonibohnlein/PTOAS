// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite source-pipe projection of union/GEN/class-wide-KILL transfers.
// No origin identities, distances, readiness, or executable cuts are inferred.
#include "PTO/Transforms/FrontierSynch/Shared/ServicePruning.h"
#include <algorithm>
#include <array>
#include <deque>
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
bool contains(const std::vector<uint32_t>& mask, uint32_t pipe)
{
    return std::find(mask.begin(), mask.end(), pipe) != mask.end();
}
std::string validate(
    const std::vector<SourcePipeNode>& graph, const std::vector<SourcePipeSite>& sites, const ServiceMask& baseline,
    const ServiceMask& candidate, const ServiceMask& protectedPaths)
{
    if (baseline.size() != sites.size() || candidate.size() != sites.size() || protectedPaths.size() != sites.size()) {
        return "inconsistent service mask inventory";
    }
    const auto classes = sites.empty() ? 0 : sites.front().modes.size();
    for (std::size_t i = 0; i < sites.size(); ++i) {
        if (sites[i].modes.size() != classes) {
            return "inconsistent class inventory";
        }
        std::set<uint32_t> unique;
        auto next = baseline[i].begin();
        for (auto pipe : candidate[i]) {
            next = std::find(next, baseline[i].end(), pipe);
            if (next == baseline[i].end() || !unique.insert(pipe).second) {
                return "candidate is not an ordered subset";
            }
            ++next;
        }
        for (auto pipe : protectedPaths[i]) {
            if (!contains(candidate[i], pipe)) {
                return "protected non-storage path removed";
            }
        }
        if (contains(baseline[i], sites[i].pipe) && !contains(candidate[i], sites[i].pipe)) {
            return "self-pipe delivery removed";
        }
    }
    for (const auto& node : graph) {
        if (node.site && *node.site >= sites.size()) {
            return "invalid graph site";
        }
        for (auto next : node.successors) {
            if (next >= graph.size()) {
                return "invalid graph edge";
            }
        }
    }
    return {};
}
using Debt = std::array<std::size_t, 4>; // class, mode, source, destination
struct DebtHash {
    std::size_t operator()(const Debt& debt) const
    {
        std::size_t seed = 0;
        for (auto field : debt) {
            seed ^= std::hash<std::size_t>{}(field) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
} // namespace
DeliveryCertificate certifyServiceDelivery(
    const std::vector<SourcePipeNode>& graph, const std::vector<SourcePipeSite>& sites, const ServiceMask& baseline,
    const ServiceMask& candidate, const ServiceMask& protectedPaths)
{
    DeliveryCertificate result;
    result.error = validate(graph, sites, baseline, candidate, protectedPaths);
    if (!result.error.empty()) {
        return result;
    }
    std::set<uint32_t> pipes;
    for (const auto& site : sites) {
        pipes.insert(site.pipe);
    }
    std::vector<std::unordered_set<Debt, DebtHash>> incoming(graph.size()), outgoing(graph.size());
    std::deque<std::pair<std::size_t, Debt>> queue;
    auto publish = [&](std::size_t node, const Debt& debt) {
        if (outgoing[node].insert(debt).second) {
            ++result.retained;
            queue.emplace_back(node, debt);
        }
    };
    std::vector<std::unordered_set<uint32_t>> masks;
    for (const auto& row : candidate) {
        masks.emplace_back(row.begin(), row.end());
    }
    auto delivered = [&](std::size_t node, const Debt& debt) {
        return graph[node].site && debt[3] == sites[*graph[node].site].pipe && masks[*graph[node].site].count(debt[2]);
    };
    for (std::size_t node = 0; node < graph.size(); ++node) {
        if (!graph[node].site) {
            continue;
        }
        const auto& site = sites[*graph[node].site];
        for (std::size_t cell = 0; cell < site.modes.size(); ++cell) {
            for (std::size_t mode : {std::size_t(1), std::size_t(2)}) {
                if (!(site.modes[cell] & mode)) {
                    continue;
                }
                for (auto destination : pipes) {
                    publish(node, {cell, mode, site.pipe, destination});
                }
            }
        }
    }
    while (!queue.empty()) {
        auto [node, debt] = queue.front();
        queue.pop_front();
        for (auto next : graph[node].successors) {
            ++result.transmissions;
            if (incoming[next].insert(debt).second && !delivered(next, debt)) {
                publish(next, debt);
            }
        }
    }
    for (std::size_t node = 0; node < graph.size(); ++node) {
        if (!graph[node].site) {
            continue;
        }
        const auto& site = sites[*graph[node].site];
        for (const auto& debt : incoming[node]) {
            ++result.inspected;
            auto mode = site.modes[debt[0]];
            if (debt[3] == site.pipe && !delivered(node, debt) && mode && (debt[1] == 2 || (mode & 2))) {
                ++result.violations;
            }
        }
    }
    result.accepted = !result.violations;
    if (!result.accepted) {
        result.error = "joint storage delivery failed";
    }
    return result;
}
ServiceProposal proposeLocalServices(
    const std::vector<SourcePipeNode>& graph, const std::vector<SourcePipeSite>& sites, const ServiceMask& baseline,
    const ServiceMask& protectedPaths)
{
    ServiceProposal result;
    result.kept = baseline;
    result.error = validate(graph, sites, baseline, baseline, protectedPaths);
    if (!result.error.empty()) {
        return result;
    }
    for (const auto& site : sites) {
        if (!site.refreshed.empty()) {
            result.error = "refresh baseline requires a separate relay contract";
            return result;
        }
    }
    std::vector<std::vector<std::size_t>> parents(graph.size());
    for (std::size_t node = 0; node < graph.size(); ++node) {
        for (auto next : graph[node].successors) {
            parents[next].push_back(node);
        }
    }
    auto leader = [&](std::size_t node) {
        return parents[node].size() != 1 || graph[parents[node].front()].successors.size() != 1;
    };
    std::vector<std::size_t> starts;
    for (std::size_t node = 0; node < graph.size(); ++node) {
        if (leader(node)) {
            starts.push_back(node);
        }
    }
    for (std::size_t node = 0; node < graph.size(); ++node) {
        starts.push_back(node);
    }
    std::vector<bool> visited(graph.size());
    for (auto start : starts) {
        std::map<uint32_t, std::map<uint32_t, std::vector<uint8_t>>> dirty;
        auto node = start;
        while (!visited[node]) {
            visited[node] = true;
            ++result.nodes;
            if (!graph[node].site) {
                dirty.clear(); // Also reset at empty structured-control boundaries.
            }
            if (graph[node].site) {
                const auto id = *graph[node].site;
                const auto& site = sites[id];
                for (auto source : baseline[id]) {
                    if (source == site.pipe) {
                        continue;
                    }
                    ++result.tested;
                    auto oldSource = dirty.find(source);
                    auto* old = oldSource == dirty.end() ? nullptr : &oldSource->second;
                    auto anchor = old ? old->find(site.pipe) : std::map<uint32_t, std::vector<uint8_t>>::iterator{};
                    bool anchored = old && anchor != old->end(), conflict = false;
                    if (anchored) {
                        for (std::size_t cell = 0; cell < site.modes.size(); ++cell) {
                            ++result.classTests;
                            auto before = anchor->second[cell], now = site.modes[cell];
                            conflict |= before && now && ((before | now) & 2);
                        }
                    }
                    if (anchored && !conflict && !contains(protectedPaths[id], source)) {
                        auto& kept = result.kept[id];
                        kept.erase(std::remove(kept.begin(), kept.end(), source), kept.end());
                        ++result.deleted;
                    } else {
                        dirty[source][site.pipe].assign(site.modes.size(), 0);
                        result.cleared += site.modes.size();
                    }
                }
                auto active = dirty.find(site.pipe);
                if (active != dirty.end()) {
                    for (auto& [target, facts] : active->second) {
                        for (std::size_t cell = 0; cell < facts.size(); ++cell) {
                            ++result.updates;
                            facts[cell] |= site.modes[cell];
                        }
                    }
                }
            }
            if (graph[node].successors.size() != 1) {
                break;
            }
            auto next = graph[node].successors.front();
            if (leader(next)) {
                break;
            }
            node = next;
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
