// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numerical quotient: starts/completions, native chains, and supplied demands.
#include "PeriodicAnalysisInternal.h"
#include <algorithm>
#include <tuple>
#include <unordered_map>
namespace mlir::pto::frontiersynch {
namespace {
bool before(const PeriodicRecord& a, const PeriodicRecord& b)
{
    return std::tie(a.source, a.target, a.displacement) < std::tie(b.source, b.target, b.displacement);
}
bool same(const PeriodicRecord& a, const PeriodicRecord& b)
{
    return a.source == b.source && a.target == b.target && a.displacement == b.displacement;
}
PeriodicAnalysis reject(const char* message)
{
    PeriodicAnalysis result;
    result.error = message;
    return result;
}
void addEdge(periodic::Graph& graph, uint32_t source, uint32_t target, uint64_t weight)
{
    const auto id = static_cast<uint32_t>(graph.edges.size());
    graph.outgoing[source].push_back(id);
    graph.edges.push_back({source, target, weight});
}
void buildNative(periodic::Graph& graph, PeriodicAnalysis& output)
{
    std::vector<std::vector<uint32_t>> types;
    for (uint32_t i = 0; i < output.payloads.size(); ++i) {
        const auto pipe = output.payloads[i].pipe;
        auto [found, inserted] = output.pipeRows.emplace(pipe, output.pipeRows.size());
        if (inserted) {
            output.frontiers.push_back({pipe, 0, {}});
            types.emplace_back();
        }
        const auto row = found->second;
        types[row].push_back(i);
        output.sourceRows.push_back(row);
        output.localRanks.push_back(++output.frontiers[row].count);
        addEdge(graph, 2 * i, 2 * i + 1, 0);
    }
    for (const auto& sequence : types) {
        for (std::size_t j = 0; j < sequence.size(); ++j) {
            const auto current = sequence[j], next = sequence[(j + 1) % sequence.size()];
            const auto wrap = j + 1 == sequence.size() ? 1 : 0;
            addEdge(graph, 2 * current, 2 * next, wrap);
            addEdge(graph, 2 * current + 1, 2 * next + 1, wrap);
        }
    }
}
} // namespace
PeriodicAnalysis analyzePeriodicDemands(llvm::ArrayRef<PeriodicPayload> payloads,
                                       llvm::ArrayRef<PeriodicRecord> records)
{
    const uint64_t maxID = std::numeric_limits<uint32_t>::max();
    // Exactly 3*m native/payload edges and at most records.size() demand edges.
    const bool invalidSize = payloads.size() > maxID / 3 || records.size() > maxID ||
        3 * uint64_t(payloads.size()) + records.size() > maxID;
    if (invalidSize) {
        return reject("quotient identity overflow");
    }
    for (const auto& record : records) {
        const bool absent = record.source >= payloads.size() || record.target >= payloads.size();
        const bool backward = record.displacement == 0 && record.source >= record.target;
        if (absent || backward) {
            return reject("nonforward or absent periodic endpoint");
        }
    }
    PeriodicAnalysis output;
    output.payloads.assign(payloads.begin(), payloads.end());
    output.generators.assign(records.begin(), records.end());
    std::sort(output.generators.begin(), output.generators.end(), before);
    output.generators.erase(std::unique(output.generators.begin(), output.generators.end(), same),
                            output.generators.end());
    periodic::Graph graph;
    graph.outgoing.resize(2 * payloads.size());
    buildNative(graph, output);
    for (const auto& record : output.generators) {
        graph.recordEdges.push_back(static_cast<uint32_t>(graph.edges.size()));
        addEdge(graph, 2 * record.source + 1, 2 * record.target, record.displacement);
    }
    output.graphEdges = graph.edges.size();
    if (!periodic::computeFrontiers(graph, output)) {
        return reject("quotient distance arithmetic overflow");
    }
    return output;
}
} // namespace mlir::pto::frontiersynch
