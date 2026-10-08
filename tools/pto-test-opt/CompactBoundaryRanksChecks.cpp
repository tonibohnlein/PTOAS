// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CompactBoundaryRanks.h"
#include "PTO/Transforms/FrontierSynch/CompactOrderBounds.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Kind = fs::PeriodicEventKind;
using Graph = std::vector<std::vector<bool>>;
Graph unfold(llvm::ArrayRef<fs::PeriodicPayload> word, llvm::ArrayRef<fs::PeriodicRecord> records,
             llvm::ArrayRef<fs::PeriodicRecord> native, uint64_t trips)
{
    const auto n = word.size() * trips;
    Graph graph(2 * n, std::vector<bool>(2 * n));
    for (std::size_t a = 0; a < n; ++a) {
        graph[2*a][2*a] = graph[2*a+1][2*a+1] = graph[2*a][2*a+1] = true;
        for (std::size_t b = a + 1; b < n; ++b) {
            if (word[a % word.size()].pipe == word[b % word.size()].pipe) {
                graph[2*a][2*b] = graph[2*a+1][2*b+1] = true;
            }
        }
    }
    for (auto edges : {records, native}) {
        for (auto edge : edges) {
            for (uint64_t i = 0; i < trips; ++i) {
                if (edge.displacement < trips - i) {
                    graph[2*(word.size()*i+edge.source)+1][2*(word.size()*(i+edge.displacement)+edge.target)] = true;
                }
            }
        }
    }
    for (std::size_t via = 0; via < graph.size(); ++via) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) {
                graph[a][b] = graph[a][b] || (graph[a][via] && graph[via][b]);
            }
        }
    }
    return graph;
}
bool compare(const fs::CompactBoundaryRanks& ranks, bool upper)
{
    const auto& snapshot = upper ? ranks.upperSnapshot() : ranks.lowerSnapshot();
    const auto& index = snapshot->analysis();
    auto& arena = *snapshot->context()->expressions();
    const auto graph = unfold(index.payloads, index.generators, index.nativePrerequisites, ranks.trips());
    const auto& selectors = upper ? ranks.upper() : ranks.lower();
    for (std::size_t p = 0; p < ranks.ports().size(); ++p) {
        const auto& event = ranks.ports()[p];
        const auto ordinal = *arena.constantValue(event.ordinal);
        const bool present = ordinal < ranks.trips();
        if (ranks.present()[p] != present) { return false; }
        const auto vertex = 2*(index.payloads.size()*ordinal+event.type) + (event.kind == Kind::Completion ? 1 : 0);
        for (std::size_t pipe = 0; pipe < ranks.pipeLabels().size(); ++pipe) {
            uint64_t rank = 0, prefix = 0, excluded = ranks.counts()[pipe];
            for (std::size_t a = 0; a < graph.size() / 2; ++a) {
                if (index.payloads[a % index.payloads.size()].pipe != ranks.pipeLabels()[pipe]) { continue; }
                ++rank;
                if (present && graph[2*a+1][vertex]) { prefix = rank; }
                if (present && graph[vertex][2*a]) { excluded = std::min(excluded, rank - 1); }
            }
            if (selectors.prefix[p][pipe] != prefix || selectors.suffixExcluded[p][pipe] != excluded ||
                ranks.counts()[pipe] != rank) { return false; }
        }
    }
    return true;
}
fs::PeriodicExcessSnapshot snapshot(fs::CompactFixedBody domain, llvm::ArrayRef<fs::PeriodicRecord> edges,
                                    llvm::ArrayRef<fs::PeriodicRecord> native = {})
{
    std::string error;
    auto graph = fs::analyzePeriodicDemands(domain->payloads(), edges, native);
    auto result = fs::bindPeriodicExcessGraph(domain, graph, fs::ReductionQuality::Covers, error);
    // The factory owns a deep graph snapshot; discarded mutable source rows
    // must not change selectors subsequently generated from it.
    graph.payloads.clear(); graph.frontiers.clear(); graph.generators.clear();
    return result;
}
bool finite(fs::CompactFixedBody domain)
{
    auto& arena = *domain->context()->expressions();
    const auto last = static_cast<uint32_t>(domain->payloads().size() - 1);
    const std::vector<fs::PeriodicRecord> choices{{0, 0, 1}, {last, 0, 2}, {0, last, uint64_t(last ? 0 : 2)}};
    const std::vector<fs::PeriodicRecord> native{{last, 0, 3}};
    for (unsigned mask = 0; mask < 8; ++mask) {
        std::vector<fs::PeriodicRecord> lower;
        for (unsigned bit = 0; bit < choices.size(); ++bit) {
            if (mask & (1U << bit)) { lower.push_back(choices[bit]); }
        }
        auto a = snapshot(domain, choices, native), b = snapshot(domain, lower, native);
        if (!a || !b) { return false; }
        for (uint64_t trips = 0; trips <= 4; ++trips) {
            std::vector<fs::RegionalEvent> ports;
            for (uint64_t ordinal = 0; ordinal <= trips; ++ordinal) {
                for (uint32_t type = 0; type < domain->payloads().size(); ++type) {
                    for (auto kind : {Kind::Start, Kind::Completion}) {
                        ports.push_back({type, arena.constant(ordinal), kind});
                    }
                }
            }
            std::string error;
            auto ranks = fs::captureCompactBoundaryRanks(a, b, trips, ports, error);
            const uint64_t expected = 8 * trips * domain->payloads().size() * domain->payloads().size();
            if (!ranks || !error.empty() || ranks->thresholdQueries() != expected ||
                ranks->frontierEntries() != 2*a->analysis().frontiers.size()*domain->payloads().size() ||
                !compare(*ranks, true) || !compare(*ranks, false)) { return false; }
        }
    }
    return true;
}
bool huge(fs::CompactFixedBody domain)
{
    auto graph = snapshot(domain, {});
    if (!graph) { return false; }
    auto& arena = *domain->context()->expressions();
    uint64_t width = 0;
    for (const auto& row : graph->analysis().frontiers) { width = std::max(width, uint64_t(row.count)); }
    const uint64_t trips = UINT64_MAX / width;
    std::vector<fs::RegionalEvent> ports{{0, arena.constant(trips-1), Kind::Completion},
                                        {0, arena.constant(trips-1), Kind::Start}};
    std::string error;
    auto ranks = fs::captureCompactBoundaryRanks(graph, graph, trips, ports, error);
    if (!ranks) { return false; }
    const auto row = graph->analysis().sourceRows[0];
    const uint64_t h = graph->analysis().frontiers[row].count;
    if (ranks->upper().prefix[0][row] != h*(trips-1)+1 || ranks->upper().prefix[1][row] != 0 ||
        ranks->upper().suffixExcluded[0][row] != h*trips ||
        ranks->upper().suffixExcluded[1][row] != h*(trips-1)) { return false; }
    if (width > 1 && fs::captureCompactBoundaryRanks(graph, graph, UINT64_MAX, ports, error)) { return false; }
    constexpr uint64_t distance = uint64_t(1) << 40;
    auto delayed = snapshot(domain, {{0, 0, distance}});
    ports = {{0, arena.constant(0), Kind::Completion}, {0, arena.constant(distance), Kind::Start}};
    auto binary = fs::captureCompactBoundaryRanks(delayed, graph, distance+2, ports, error);
    return binary && binary->upper().prefix[1][row] == 1 &&
        binary->upper().suffixExcluded[0][row] == h*distance &&
        binary->thresholdQueries() == 8*domain->payloads().size();
}
bool malformed(fs::CompactFixedBody domain, fs::CompactFixedBody foreign)
{
    auto graph = snapshot(domain, {}), other = snapshot(foreign, {}), strong = snapshot(domain, {{0, 0, 1}});
    auto& arena = *domain->context()->expressions();
    std::string error;
    const fs::RegionalEvent valid{0, arena.constant(0), Kind::Start};
    if (!graph || !other || !strong || fs::captureCompactBoundaryRanks({}, graph, 1, {valid}, error) ||
        fs::captureCompactBoundaryRanks(graph, other, 1, {valid}, error) ||
        fs::captureCompactBoundaryRanks(graph, strong, 0, {}, error)) { return false; }
    auto differentNative = snapshot(domain, {}, {{0, 0, 1}});
    if (fs::captureCompactBoundaryRanks(differentNative, graph, 1, {}, error)) { return false; }
    std::vector<fs::RegionalEvent> invalid(5, valid);
    invalid[0].ordinal = domain->trips();
    invalid[1].ordinal = arena.boolean(true);
    invalid[2].type = static_cast<uint32_t>(domain->payloads().size());
    invalid[3].visits = {arena.constant(0)};
    invalid[4].kind = static_cast<Kind>(10);
    for (const auto& port : invalid) {
        if (fs::captureCompactBoundaryRanks(graph, graph, 1, {port}, error) || error.empty()) { return false; }
    }
    auto empty = fs::captureCompactBoundaryRanks(graph, graph, 0, {}, error);
    return empty && empty->ports().empty() && !empty->thresholdQueries();
}
} // namespace
int runCompactBoundaryRanksChecks(func::FuncOp function, const pto::SyncInput& input)
{
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return 1; }
    scf::ForOp loop;
    for (auto candidate : function.getOps<scf::ForOp>()) { if (loop) { return 1; } loop = candidate; }
    if (!loop) { return 1; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    const auto upper = arena->input(loop.getUpperBound());
    const auto trips = arena->select(arena->slt(arena->constant(0), upper), upper, arena->constant(0));
    std::string error;
    auto domain = fs::captureCompactFixedBodyContext(loop, input, index, arena, trips, error);
    if (function->hasAttr("test.compact_nested")) { return !domain && !error.empty() ? 0 : 1; }
    auto foreign = fs::captureCompactFixedBodyContext(loop, input, index, arena, trips, error);
    if (!domain || !foreign || domain->payloads().empty() || !finite(domain) || !huge(domain) ||
        !malformed(domain, foreign)) {
        llvm::errs() << "compact boundary ranks failed: " << function.getSymName() << "\n"; return 1;
    }
    llvm::outs() << "compact boundary ranks passed: " << function.getSymName() << "\n";
    return 0;
}
