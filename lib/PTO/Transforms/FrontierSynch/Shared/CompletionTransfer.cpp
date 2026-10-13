// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/CompletionTransfer.h"
#include <limits>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
void count(uint64_t& value)
{
    if (value != std::numeric_limits<uint64_t>::max()) { ++value; }
}
bool valid(const CompletionSummary& summary, std::size_t classes)
{
    if (summary.all.size() != classes || summary.exposed.size() != classes ||
        summary.suffix.size() != classes) { return false; }
    for (std::size_t c = 0; c < classes; ++c) {
        if (summary.all[c] > 3 || (summary.exposed[c] & ~summary.all[c]) ||
            (summary.suffix[c] & ~summary.all[c])) { return false; }
    }
    return true;
}
bool conflict(const CompletionFacts& a, const CompletionFacts& b)
{
    for (std::size_t c = 0; c < a.size(); ++c) {
        if (((a[c] & 2) && b[c]) || (a[c] && (b[c] & 2))) { return true; }
    }
    return false;
}
void join(CompletionFacts& a, const CompletionFacts& b)
{
    for (std::size_t c = 0; c < a.size(); ++c) { a[c] |= b[c]; }
}
struct Flow {
    std::vector<CompletionFacts> incoming, outgoing;
};
Flow solve(const std::vector<CompletionVertex>& graph, const std::vector<bool>& reachable,
           const std::vector<bool>& protect, std::size_t classes, CompletionTransferWork& work)
{
    Flow flow{std::vector<CompletionFacts>(graph.size(), CompletionFacts(classes)),
              std::vector<CompletionFacts>(graph.size(), CompletionFacts(classes))};
    std::vector<std::tuple<std::size_t, std::size_t, uint8_t>> pending;
    auto publish = [&](std::size_t vertex, std::size_t cell, uint8_t mode) {
        if (!(flow.outgoing[vertex][cell] & mode)) {
            flow.outgoing[vertex][cell] |= mode;
            pending.emplace_back(vertex, cell, mode);
            count(work.propagatedFacts);
        }
    };
    for (std::size_t v = 0; v < graph.size(); ++v) {
        if (!reachable[v]) { continue; }
        for (std::size_t c = 0; c < classes; ++c) {
            for (uint8_t mode : {uint8_t(1), uint8_t(2)}) {
                if (graph[v].summary.suffix[c] & mode) { publish(v, c, mode); }
            }
        }
    }
    for (std::size_t next = 0; next < pending.size(); ++next) {
        const auto [v, cell, mode] = pending[next];
        for (auto successor : graph[v].successors) {
            count(work.edgeFactVisits);
            if (flow.incoming[successor][cell] & mode) { continue; }
            flow.incoming[successor][cell] |= mode;
            if (!protect[successor] && graph[successor].summary.transparent) {
                publish(successor, cell, mode);
            }
        }
    }
    return flow;
}
} // namespace
CompletionTransferResult buildCompletionTransfer(const std::vector<CompletionVertex>& graph,
    std::size_t entry, std::size_t exit, std::size_t classCount)
{
    CompletionTransferResult result;
    if (entry >= graph.size() || exit >= graph.size() || !graph[exit].successors.empty()) {
        result.error = "completion CFG requires valid entry and terminal exit"; return result;
    }
    for (const auto& vertex : graph) {
        if (!valid(vertex.summary, classCount)) {
            result.error = "completion summary has inconsistent storage classes or modes"; return result;
        }
        for (auto successor : vertex.successors) {
            if (successor >= graph.size()) {
                result.error = "completion CFG has an invalid successor"; return result;
            }
        }
    }
    result.reachable.assign(graph.size(), false);
    result.protectEntry.assign(graph.size(), false);
    std::vector<std::size_t> pending{entry};
    result.reachable[entry] = true;
    for (std::size_t next = 0; next < pending.size(); ++next) {
        for (auto successor : graph[pending[next]].successors) {
            if (!result.reachable[successor]) {
                result.reachable[successor] = true; pending.push_back(successor);
            }
        }
    }
    {
        auto first = solve(graph, result.reachable, result.protectEntry, classCount, result.work);
        for (auto v : pending) {
            result.protectEntry[v] = graph[v].requiresEntryCompletion ||
                                    conflict(first.incoming[v], graph[v].summary.exposed);
        }
    }
    auto second = solve(graph, result.reachable, result.protectEntry, classCount, result.work);
    auto& summary = result.summary;
    summary.all.assign(classCount, 0);
    summary.exposed.assign(classCount, 0);
    summary.suffix = std::move(second.outgoing[exit]);
    summary.transparent = false;
    for (auto v : pending) { join(summary.all, graph[v].summary.all); }
    // Only paths with no preceding completion contribute exposed effects.
    // The same reachability determines whether an empty-boundary visit exists.
    std::vector<bool> beforeCompletion(graph.size(), false);
    pending.assign(1, entry); beforeCompletion[entry] = true;
    for (std::size_t next = 0; next < pending.size(); ++next) {
        const auto v = pending[next];
        if (result.protectEntry[v]) { continue; }
        join(summary.exposed, graph[v].summary.exposed);
        if (!graph[v].summary.transparent) { continue; }
        if (v == exit) { summary.transparent = true; }
        for (auto successor : graph[v].successors) {
            if (!beforeCompletion[successor]) {
                beforeCompletion[successor] = true; pending.push_back(successor);
            }
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
