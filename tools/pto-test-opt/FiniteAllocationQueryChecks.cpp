// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Graph = std::vector<std::vector<bool>>;
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (std::size_t j = 0; j < graph.size(); ++j) {
                graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
            }
        }
    }
}
bool check(func::FuncOp function, ArrayRef<uint32_t> pipes, ArrayRef<fs::StorageGenerator> edges)
{
    fs::ExplicitAnalysis analysis;
    for (uint32_t i = 0; i < pipes.size(); ++i) {
        fs::ExplicitEffects occurrence; occurrence.payload = i; occurrence.pipe = pipes[i];
        analysis.occurrences.push_back(occurrence);
    }
    analysis.reduction = fs::reduceExplicitDemands(analysis.occurrences, edges);
    if (!analysis.reduction.error.empty()) { return false; }
    auto certificate = fs::explicitAllocationCertificate(analysis, 7, function.getContext());
    if (!certificate || certificate.getAs<StringAttr>("mode").getValue() != "rank-query") { return false; }
    auto rows = certificate.getAs<ArrayAttr>("handoffs");
    const auto& reduction = analysis.reduction;
    Graph graph(2*pipes.size(), std::vector<bool>(2*pipes.size()));
    for (std::size_t i = 0; i < pipes.size(); ++i) {
        graph[2*i][2*i] = graph[2*i+1][2*i+1] = graph[2*i][2*i+1] = true;
        for (std::size_t j = i+1; j < pipes.size(); ++j) {
            if (pipes[i] == pipes[j]) { graph[2*i][2*j] = graph[2*i+1][2*j+1] = true; }
        }
    }
    for (const auto& edge : edges) { graph[2*edge.source+1][2*edge.target] = true; }
    for (const auto& edge : reduction.retained) {
        if (pipes[edge.source] != pipes[edge.target]) { continue; }
        for (std::size_t prior = 0; prior < edge.target; ++prior) {
            if (pipes[prior] == pipes[edge.target]) { graph[2*prior+1][2*edge.target] = true; }
        }
    }
    close(graph);
    auto allocation = fs::decodeFiniteAllocation(function, certificate, {0, 1, 2, 3, 4, 5});
    if (failed(allocation) || allocation->records.size() != rows.size()) { return false; }
    std::map<int64_t, uint32_t> last;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        auto row = cast<DictionaryAttr>(rows[i]);
        const auto record = row.getAs<IntegerAttr>("record").getInt();
        const auto& edge = reduction.retained[record];
        auto profile = row.getAs<DenseI64ArrayAttr>("evidence");
        if (!profile || static_cast<std::size_t>(profile.size()) != reduction.pipeLabels.size()) { return false; }
        // Check every compact rank answer against an independently closed graph.
        for (auto other : rows) {
            auto targetRow = cast<DictionaryAttr>(other);
            const auto targetRecord = targetRow.getAs<IntegerAttr>("record").getInt();
            const auto& next = reduction.retained[targetRecord];
            const bool same = pipes[edge.target] == pipes[next.source];
            auto targetProfile = targetRow.getAs<DenseI64ArrayAttr>("evidence");
            const bool rankAnswer = same ? edge.target <= next.source :
                targetProfile[reduction.pipeColumns[edge.target]] >= reduction.localRanks[edge.target];
            if (rankAnswer != graph[2*edge.target + (same ? 0 : 1)][2*next.source]) { return false; }
        }
        const auto& assigned = allocation->records[i];
        if (assigned.record != record || assigned.ids.size() != 1) { return false; }
        auto found = last.find(assigned.ids.front());
        if (found != last.end()) {
            const bool same = pipes[found->second] == pipes[edge.source];
            if (!graph[2*found->second + (same ? 0 : 1)][2*edge.source]) { return false; }
        }
        last[assigned.ids.front()] = edge.target;
    }
    return true;
}
bool compact(MLIRContext& context)
{
    fs::ExplicitAnalysis analysis;
    std::vector<fs::StorageGenerator> edges;
    constexpr unsigned count = 256;
    for (uint32_t i = 0; i < count; ++i) {
        fs::ExplicitEffects occurrence; occurrence.payload = i; occurrence.pipe = i%2;
        analysis.occurrences.push_back(occurrence);
        if (i) { edges.push_back({i-1, i}); }
    }
    analysis.reduction = fs::reduceExplicitDemands(analysis.occurrences, edges);
    auto certificate = fs::explicitAllocationCertificate(analysis, 0, &context);
    if (!certificate) { return false; }
    auto rows = certificate.getAs<ArrayAttr>("handoffs");
    if (rows.size() != count-1) { return false; }
    uint64_t stored = 0;
    for (auto row : rows) { stored += cast<DictionaryAttr>(row).getAs<DenseI64ArrayAttr>("evidence").size(); }
    return stored == 2*(count-1);
}
} // namespace
int runFiniteAllocationQueryChecks()
{
    MLIRContext context;
    context.disableMultithreading(); context.loadDialect<func::FuncDialect>();
    Builder b(&context);
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(
        b.getUnknownLoc(), "finite_rank_query_checks", b.getFunctionType({}, {})));
    if (!check(*function, {}, {}) || !check(*function, {0, 1}, {{0, 1}}) ||
        !check(*function, {0, 1, 0, 1, 2, 0}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}}) ||
        !check(*function, {0, 1, 0, 2, 0, 1}, {{0, 1}, {0, 4}, {3, 5}}) || !compact(context)) {
        llvm::errs() << "finite rank-query certificate check failed\n"; return 1;
    }
    llvm::outs() << "finite rank-query compact profiles and command-order oracle passed\n";
    return 0;
}
