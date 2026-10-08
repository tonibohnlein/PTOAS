// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent dense DAG oracle for projections, crossing reductions and queries.
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
#include "PTO/Transforms/FrontierSynch/RegionalNumericalInterface.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Graph = std::vector<std::vector<bool>>;
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) {
                graph[a][b] = graph[a][b] || (graph[a][k] && graph[k][b]);
            }
        }
    }
}
bool numerical(unsigned mask, unsigned selectedMask, uint64_t& checked)
{
    Graph graph(8, std::vector<bool>(8));
    for (unsigned a = 0; a < 8; ++a) {
        graph[a][a] = true;
        for (unsigned b = a + 1; b < 8; ++b) { if (a % 2 == b % 2) { graph[a][b] = true; } }
    }
    const std::array<std::pair<unsigned, unsigned>, 8> optional{{{0, 1}, {1, 2}, {2, 3}, {4, 5},
                                                               {5, 6}, {6, 7}, {0, 7}, {3, 4}}};
    for (unsigned i = 0; i < optional.size(); ++i) {
        if (mask & (1U << i)) { graph[optional[i].first][optional[i].second] = true; }
    }
    auto direct = graph; close(graph);
    auto leaf = [&](unsigned start) {
        return std::make_shared<const fs::NumericalChainInterface>(fs::buildNumericalChainInterface(
            {{0, 2}, {1, 3}}, [&](uint32_t a, uint32_t b) {
                return std::optional<bool>(graph[start + a][start + b]);
            }));
    };
    auto left = leaf(0), right = leaf(4);
    std::vector<fs::NumericalCrossing> links;
    for (unsigned a = 0; a < 4; ++a) {
        for (unsigned b = 4; b < 8; ++b) { if (direct[a][b]) { links.push_back({a, b - 4}); } }
    }
    std::vector<fs::NumericalChainSelection> selection;
    // Reverse dense output IDs to exercise filtering without numerical-ID order.
    for (unsigned a = 8; a; --a) {
        if (selectedMask & (1U << (a - 1))) { selection.push_back({(a - 1) / 4, (a - 1) % 4}); }
    }
    auto merge = fs::buildNumericalChainMerge(left, right, links, selection, {0, 1}, {0, 1});
    if (!merge.index.error.empty() || merge.index.queries) { return false; }
    auto original = [&](unsigned id) { return 4 * selection[id].child + selection[id].event; };
    for (unsigned a = 0; a < selection.size(); ++a) {
        for (unsigned b = 0; b < selection.size(); ++b) {
            if (merge.index.reaches(a, b) != graph[original(a)][original(b)]) { return false; }
            ++checked;
        }
    }
    fs::NumericalChainQueryCost cost;
    for (unsigned event = 0; event < 8; ++event) {
        const auto child = event / 4;
        auto index = child ? right : left;
        for (bool reverse : {false, true}) {
            auto vector = fs::numericalLeafThresholds(*index, [&](uint32_t port, bool back) {
                return std::optional<bool>(back ? graph[4 * child + port][event] : graph[event][4 * child + port]);
            }, reverse, cost);
            auto propagated = vector ? merge.propagate(child, *vector, reverse, cost) : std::nullopt;
            if (!propagated) { return false; }
            for (unsigned target = 0; target < selection.size(); ++target) {
                auto c = merge.index.chain[target], r = merge.index.rank[target];
                const bool value = reverse ? r < (*propagated)[c] : r >= (*propagated)[c];
                const bool expected = reverse ? graph[original(target)][event] : graph[event][original(target)];
                if (value != expected) { return false; }
                ++checked;
            }
        }
    }
    for (unsigned a = 0; a < 4; ++a) {
        for (unsigned b = 0; b < 4; ++b) {
            auto answer = merge.crosses(left->forward[a], right->reverse[b], cost);
            if (answer != std::optional<bool>(graph[a][b + 4])) { return false; }
        }
    }
    return true;
}
bool hierarchy(func::FuncOp function, uint64_t& checked)
{
    constexpr uint32_t leaves = 4, trips = 3, types = 2, payloads = leaves * trips * types;
    Graph graph(2 * payloads, std::vector<bool>(2 * payloads));
    for (uint32_t a = 0; a < payloads; ++a) {
        graph[2*a][2*a] = graph[2*a+1][2*a+1] = graph[2*a][2*a+1] = true;
        for (uint32_t b = a + 1; b < payloads; ++b) {
            if (a % types == b % types) { graph[2*a][2*b] = graph[2*a+1][2*b+1] = true; }
        }
        if (a % types == 0) { graph[2*a+1][2*(a+1)] = true; }
    }
    close(graph);
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::vector<pto::CompoundInstanceElement> phases;
    phases.reserve(leaves * types);
    std::vector<fs::RegionalAnalysis> regions;
    uint64_t queries = 0;
    for (uint32_t leaf = 0; leaf < leaves; ++leaf) {
        fs::RegionalAnalysis region;
        region.expressions = arena; region.capabilities = {true, true, true, false};
        for (uint32_t type = 0; type < types; ++type) {
            const auto pipe = type ? pto::PipelineType::PIPE_V : pto::PipelineType::PIPE_MTE1;
            phases.emplace_back(phases.size(), SmallVector<const pto::BaseMemInfo*>{},
                SmallVector<const pto::BaseMemInfo*>{}, pipe, function->getName());
            phases.back().elementOp = function;
            region.anchors.push_back({&phases.back(), {}, {}, {}}); region.occurrenceLoops.push_back({});
            region.firstPayloads[static_cast<uint32_t>(pipe)].push_back(
                {{type, arena->constant(0), fs::PeriodicEventKind::Start}, arena->boolean(true)});
            region.lastPayloads[static_cast<uint32_t>(pipe)].push_back(
                {{type, arena->constant(trips - 1), fs::PeriodicEventKind::Start}, arena->boolean(true)});
        }
        region.presence = [arena](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
            return arena->lt(event.ordinal, arena->constant(trips));
        };
        region.reachability = [arena, leaf, &graph, &queries](fs::RegionalEvent a, fs::RegionalEvent b)
            -> std::optional<fs::RegionExpressions::Id> {
            ++queries;
            auto av = arena->constantValue(a.ordinal), bv = arena->constantValue(b.ordinal);
            if (!av || !bv) { return std::nullopt; }
            if (*av >= trips || *bv >= trips) { return arena->boolean(false); }
            auto ai = 2 * (leaf * trips * types + *av * types + a.type) + (a.kind == fs::PeriodicEventKind::Completion);
            auto bi = 2 * (leaf * trips * types + *bv * types + b.type) + (b.kind == fs::PeriodicEventKind::Completion);
            return arena->boolean(graph[ai][bi]);
        };
        regions.push_back(std::move(region));
    }
    auto a = fs::composeRegionalSequence(function, arena, {regions[0], regions[1]}, false, false);
    auto b = fs::composeRegionalSequence(function, arena, {regions[2], regions[3]}, false, false);
    if (!a.error.empty() || !b.error.empty()) { return false; }
    auto first = fs::sequenceRegionalResult(a), second = fs::sequenceRegionalResult(b);
    if (!first.numerical || !second.numerical) { return false; }
    const auto before = queries;
    auto root = fs::composeRegionalSequence(function, arena, {first, second}, false, false);
    auto exported = fs::sequenceRegionalResult(root);
    if (!root.error.empty() || !exported.numerical || queries != before || root.cost.numericalMerges != 3 ||
        root.cost.numericalReusedChildren != 2) { return false; }
    auto event = [&](uint32_t id) {
        const auto payload = id / 2, leaf = payload / (trips * types), ordinal = (payload / types) % trips;
        return fs::RegionalEvent{leaf * types + payload % types, arena->constant(ordinal),
            id % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start};
    };
    for (uint32_t x = 0; x < graph.size(); ++x) {
        for (uint32_t y = 0; y < graph.size(); ++y) {
            auto actual = exported.reachability(event(x), event(y));
            if (!actual || arena->constantValue(*actual) != uint64_t(graph[x][y])) { return false; }
            ++checked;
        }
    }
    auto counts = fs::sequenceNumericalQueryCounts(root);
    return counts.leafQueries && counts.indexOperations;
}
} // namespace
int runNumericalHierarchyChecks(func::FuncOp function)
{
    uint64_t checked = 0;
    for (unsigned mask = 0; mask < 256; ++mask) {
        for (unsigned selected : {0U, 1U, 85U, 170U, 255U}) {
            if (!numerical(mask, selected, checked)) {
                llvm::errs() << "numerical projection/query oracle failed at " << mask << ", " << selected << "\n";
                return 1;
            }
        }
    }
    if (!hierarchy(function, checked)) { llvm::errs() << "numerical sequence hierarchy oracle failed\n"; return 1; }
    llvm::outs() << "numerical hierarchy checked " << checked << " independent event pairs\n";
    return 0;
}
