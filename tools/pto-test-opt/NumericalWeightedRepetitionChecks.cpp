// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite unfolding: Boolean BFS never uses scaled ranks or scores.
#include "PTO/Transforms/FrontierSynch/NumericalWeightedRepetition.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <numeric>
namespace fs = mlir::pto::frontiersynch;
namespace {
using Matrix = std::vector<std::vector<bool>>;
using Crossing = fs::NumericalWeightedCrossing;
void close(Matrix& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (std::size_t j = 0; j < graph.size(); ++j) {
                graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
            }
        }
    }
}
std::vector<bool> unfold(const Matrix& internal, const std::vector<Crossing>& records,
    uint32_t source, uint32_t horizon, std::optional<uint32_t> excluded = std::nullopt,
    const std::vector<bool>* retained = nullptr)
{
    const auto p = internal.size();
    std::vector<std::vector<std::size_t>> edges((horizon + 1) * p);
    for (uint32_t visit = 0; visit <= horizon; ++visit) {
        for (std::size_t a = 0; a < p; ++a) {
            for (std::size_t b = 0; b < p; ++b) {
                if (internal[a][b]) { edges[visit * p + a].push_back(visit * p + b); }
            }
        }
        for (uint32_t id = 0; id < records.size(); ++id) {
            const auto& record = records[id];
            if (excluded == id || (retained && !(*retained)[id]) || record.displacement > horizon - visit) {
                continue;
            }
            edges[visit * p + record.source].push_back((visit + record.displacement) * p + record.target);
        }
    }
    std::vector<bool> reached(edges.size(), false);
    std::vector<std::size_t> ready{source}; reached[source] = true;
    for (std::size_t next = 0; next < ready.size(); ++next) {
        for (auto target : edges[ready[next]]) {
            if (!reached[target]) { reached[target] = true; ready.push_back(target); }
        }
    }
    return reached;
}
fs::NumericalChainInterface makeIndex(std::vector<std::vector<uint32_t>> chains, Matrix graph)
{
    close(graph);
    return fs::buildNumericalChainInterface(std::move(chains),
        [graph = std::move(graph)](uint32_t a, uint32_t b) -> std::optional<bool> { return graph[a][b]; });
}
bool check(unsigned p, unsigned chainCount, unsigned seed, uint64_t& pairs)
{
    uint32_t random = seed + 1;
    auto next = [&]() { random = random * 1664525U + 1013904223U; return random; };
    std::vector<std::vector<uint32_t>> chains(chainCount);
    for (uint32_t i = 0; i < p; ++i) { chains[i % chainCount].push_back(i); }
    Matrix graph(p, std::vector<bool>(p, false));
    for (uint32_t i = 0; i < p; ++i) { graph[i][i] = true; }
    for (const auto& chain : chains) {
        for (std::size_t i = 1; i < chain.size(); ++i) { graph[chain[i - 1]][chain[i]] = true; }
    }
    for (uint32_t i = 0; i < p; ++i) {
        for (uint32_t j = i + 1; j < p; ++j) { if ((next() >> 24) % 4 == 0) { graph[i][j] = true; } }
    }
    auto child = makeIndex(chains, graph);
    std::vector<Crossing> records;
    for (const auto& chain : chains) { records.push_back({chain.back(), chain.front(), 1, true}); }
    for (unsigned edge = 0; edge < 2 * p; ++edge) {
        auto a = next() % p, b = next() % p;
        records.push_back({a, b, 1 + next() % 3, false});
    }
    // Duplicate an ordinary record and a native one with opposite priority.
    records.push_back(records.back());
    auto duplicate = records.front(); duplicate.native = false; records.push_back(duplicate);
    auto result = fs::buildNumericalWeightedRepetition(child, records);
    if (!result.error.empty() || !result.index || result.originalToCanonical.size() != records.size()) {
        llvm::errs() << "weighted index construction: " << result.error << "\n"; return false;
    }
    const auto& canonical = result.crossings;
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& mapped = canonical[result.originalToCanonical[i]];
        if (records[i].source != mapped.source || records[i].target != mapped.target ||
            records[i].displacement != mapped.displacement || (records[i].native && !mapped.native)) { return false; }
    }
    if (result.cost.relaxations > chainCount * (result.cost.internalEdges + result.cost.crossingEdges) ||
        result.cost.prefixEntries != chainCount * canonical.size()) { return false; }
    // Any shortest simple weighted path has <=p-1 positive edges of weight<=3.
    // Native carries fill every later gap; this horizon includes the minimum.
    const uint32_t horizon = 3 * p + 2;
    for (uint32_t source = 0; source < p; ++source) {
        auto required = unfold(graph, canonical, source, horizon);
        auto reduced = unfold(graph, canonical, source, horizon, std::nullopt, &result.retained);
        if (required != reduced) { llvm::errs() << "weighted reduced closure mismatch\n"; return false; }
        for (uint32_t target = 0; target < p; ++target) {
            std::optional<uint64_t> first;
            for (uint32_t gap = 0; gap <= horizon; ++gap) {
                const bool expected = required[gap * p + target];
                auto query = result.index->query(source, target, gap);
                ++pairs;
                if (!query || *query != expected) { llvm::errs() << "weighted port query mismatch\n"; return false; }
                if (expected && !first) { first = gap; }
                if (gap) {
                    uint64_t probes = 0;
                    auto arbitrary = result.index->across(child.forward[source], child.reverse[target], gap, probes);
                    if (!arbitrary || *arbitrary != expected || probes > chainCount * chainCount) { return false; }
                }
            }
            auto distance = result.index->distance(source, target);
            if (!distance || distance->reachable != first.has_value() || distance->displacement != first) {
                llvm::errs() << "weighted recovered distance mismatch\n"; return false;
            }
        }
    }
    for (uint32_t id = 0; id < canonical.size(); ++id) {
        const auto& record = canonical[id];
        auto without = unfold(graph, canonical, record.source, static_cast<uint32_t>(record.displacement), id);
        const bool cover = record.native || !without[record.displacement * p + record.target];
        if (result.retained[id] != cover) { llvm::errs() << "weighted crossing cover mismatch\n"; return false; }
    }
    // The index must not borrow mutable rows from the caller.
    child.forward.clear(); child.chains.clear();
    if (result.index->query(0, 0, 1) != true || result.index->query(p, 0, 1).has_value()) { return false; }
    return true;
}
bool boundaries()
{
    auto empty = fs::buildNumericalWeightedRepetition({}, {});
    if (!empty.error.empty() || !empty.index || !empty.crossings.empty()) { return false; }
    Matrix graph(3, std::vector<bool>(3, false));
    for (unsigned i = 0; i < 3; ++i) { graph[i][i] = true; }
    auto body = makeIndex({{0}, {1}, {2}}, graph);
    std::vector<Crossing> edges{{0, 0, 1, true}, {1, 1, 1, true}, {2, 2, 1, true},
                              {0, 1, UINT64_MAX, false}, {1, 2, UINT64_MAX, false}};
    auto result = fs::buildNumericalWeightedRepetition(body, edges);
    if (!result.error.empty() || !result.index) { return false; }
    auto huge = result.index->distance(0, 2), largest = result.index->distance(0, 1);
    auto unreachable = result.index->distance(2, 0);
    if (!huge || !huge->reachable || huge->displacement || !largest || largest->displacement != UINT64_MAX ||
        !unreachable || unreachable->reachable || result.index->query(0, 2, UINT64_MAX) != false ||
        result.index->query(0, 1, UINT64_MAX) != true) { return false; }
    // Equal exclusion scores must keep distinct identities: the tested first
    // score is excluded, but the second still proves the redundant crossing.
    auto tiedBody = makeIndex({{0}, {1}, {2}}, {{true, true, false},
        {false, true, false}, {false, false, true}});
    auto tied = fs::buildNumericalWeightedRepetition(tiedBody, {{0, 0, 1, true},
        {1, 1, 1, true}, {2, 2, 1, true}, {0, 2, 1, false}, {1, 2, 1, false}});
    if (!tied.error.empty() || tied.retained[tied.originalToCanonical[3]] ||
        !tied.retained[tied.originalToCanonical[4]]) { return false; }
    auto zero = edges; zero.back().displacement = 0;
    if (fs::buildNumericalWeightedRepetition(body, zero).error.empty()) { return false; }
    auto missing = edges; missing.erase(missing.begin());
    if (fs::buildNumericalWeightedRepetition(body, missing).error.empty()) { return false; }
    auto malformed = body; malformed.reverse[0].clear();
    if (fs::buildNumericalWeightedRepetition(malformed, edges).error.empty()) { return false; }
    auto alias = makeIndex({{0}, {1}}, {{true, true}, {true, true}});
    if (fs::buildNumericalWeightedRepetition(alias, {{0, 0, 1, true}, {1, 1, 1, true}}).error.empty()) {
        return false;
    }
    auto singleton = makeIndex({{0}}, {{true}});
    auto priority = fs::buildNumericalWeightedRepetition(singleton, {{0, 0, 1, false}, {0, 0, 1, true}});
    if (!priority.error.empty() || priority.crossings.size() != 1 || !priority.crossings[0].native ||
        priority.representatives != std::vector<uint32_t>{1} || !priority.retained[0]) { return false; }
    uint64_t probes = 0;
    if (result.index->across({0, 0, 0}, {1, 1, 1}, 0, probes).has_value() ||
        result.index->across({2, 0, 0}, {1, 1, 1}, 1, probes).has_value()) { return false; }
    return true;
}
} // namespace
int runNumericalWeightedRepetitionChecks()
{
    uint64_t pairs = 0, graphs = 0;
    if (!boundaries()) { llvm::errs() << "weighted repetition boundary checks failed\n"; return 1; }
    for (unsigned ports = 1; ports <= 6; ++ports) {
        for (unsigned chains = 1; chains <= std::min(ports, 3U); ++chains) {
            for (unsigned seed = 0; seed < 12; ++seed) {
                if (!check(ports, chains, seed, pairs)) {
                    llvm::errs() << "weighted repetition oracle failed P=" << ports << " c=" << chains
                                 << " seed=" << seed << "\n";
                    return 1;
                }
                ++graphs;
            }
        }
    }
    llvm::outs() << "weighted repetition: " << graphs << " graphs and " << pairs << " finite port queries passed\n";
    return 0;
}
