// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite-state execution oracle, including arbitrary backedges.
#include "PTO/Transforms/FrontierSynch/Shared/CompletionTransfer.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <random>
#include <tuple>
namespace fs = mlir::pto::frontiersynch;
namespace {
unsigned bits(const fs::CompletionFacts& facts)
{
    unsigned value = 0;
    for (std::size_t c = 0; c < facts.size(); ++c) { value |= unsigned(facts[c]) << (2 * c); }
    return value;
}
fs::CompletionFacts facts(unsigned value) { return {uint8_t(value & 3), uint8_t((value >> 2) & 3)}; }
struct Executions {
    std::vector<unsigned> incoming;
    unsigned all = 0, exposed = 0, suffix = 0;
    bool transparent = false;
};
Executions execute(const std::vector<fs::CompletionVertex>& graph, const std::vector<bool>& protect)
{
    Executions result;
    result.incoming.resize(graph.size());
    using Visit = std::tuple<std::size_t, unsigned, bool>;
    std::set<Visit> seen;
    std::vector<Visit> work{{0, 0, false}};
    seen.insert(work.front());
    for (std::size_t next = 0; next < work.size(); ++next) {
        const auto [vertex, incoming, completed] = work[next];
        const auto& summary = graph[vertex].summary;
        result.incoming[vertex] |= incoming;
        result.all |= bits(summary.all);
        if (!completed && !protect[vertex]) { result.exposed |= bits(summary.exposed); }
        const bool boundary = protect[vertex] || !summary.transparent;
        const auto outgoing = (boundary ? 0 : incoming) | bits(summary.suffix);
        if (vertex == graph.size() - 1) {
            result.suffix |= outgoing;
            result.transparent |= !completed && !boundary;
        }
        for (auto successor : graph[vertex].successors) {
            Visit visit{successor, outgoing, completed || boundary};
            if (seen.insert(visit).second) { work.push_back(visit); }
        }
    }
    return result;
}
bool conflicts(unsigned incoming, unsigned exposed)
{
    for (unsigned c = 0; c < 2; ++c) {
        const auto a = (incoming >> (2 * c)) & 3;
        const auto b = (exposed >> (2 * c)) & 3;
        if (a && b && ((a | b) & 2)) { return true; }
    }
    return false;
}
// Independent oracle: transitive closure identifies components, and concrete
// state exploration supplies incoming facts. No production delta equations.
std::vector<bool> referenceProtection(const std::vector<fs::CompletionVertex>& graph)
{
    const auto size = graph.size();
    std::vector<std::vector<bool>> reaches(size, std::vector<bool>(size));
    std::vector<bool> protect(size), done(size);
    for (std::size_t v = 0; v < size; ++v) {
        reaches[v][v] = true;
        for (auto next : graph[v].successors) { reaches[v][next] = true; }
    }
    for (std::size_t k = 0; k < size; ++k) {
        for (std::size_t i = 0; i < size; ++i) {
            for (std::size_t j = 0; j < size; ++j) {
                reaches[i][j] = reaches[i][j] || (reaches[i][k] && reaches[k][j]);
            }
        }
    }
    for (std::size_t v = 0; v < size; ++v) {
        done[v] = !reaches[0][v];
        protect[v] = reaches[0][v] && graph[v].requiresEntryCompletion;
    }
    for (std::size_t pass = 0; pass < size; ++pass) {
        for (std::size_t v = 0; v < size; ++v) {
            if (done[v]) { continue; }
            bool ready = true;
            for (std::size_t u = 0; u < size; ++u) {
                if (!done[u] && reaches[u][v] && !reaches[v][u]) { ready = false; }
            }
            if (!ready) { continue; }
            const auto preliminary = execute(graph, protect);
            for (std::size_t u = 0; u < size; ++u) {
                if (reaches[v][u] && reaches[u][v]) {
                    protect[u] = protect[u] || conflicts(preliminary.incoming[u], bits(graph[u].summary.exposed));
                    done[u] = true;
                }
            }
        }
    }
    return protect;
}
bool check(std::vector<fs::CompletionVertex> graph)
{
    auto first = execute(graph, std::vector<bool>(graph.size()));
    std::vector<bool> protect(graph.size());
    for (std::size_t v = 0; v < graph.size(); ++v) {
        protect[v] = graph[v].requiresEntryCompletion;
        const auto exposed = bits(graph[v].summary.exposed);
        for (unsigned c = 0; c < 2; ++c) {
            const auto a = (first.incoming[v] >> (2 * c)) & 3;
            const auto b = (exposed >> (2 * c)) & 3;
            protect[v] = protect[v] || (a && b && ((a | b) & 2));
        }
    }
    const auto selected = referenceProtection(graph);
    const auto expected = execute(graph, selected);
    const auto actual = fs::buildCompletionTransfer(graph, 0, graph.size() - 1, 2);
    for (std::size_t v = 0; v < graph.size(); ++v) {
        if (actual.protectEntry[v] && !protect[v]) { return false; }
        if (actual.reachable[v] && !selected[v] &&
            conflicts(expected.incoming[v], bits(graph[v].summary.exposed))) { return false; }
    }
    std::size_t edges = 0;
    for (const auto& vertex : graph) { edges += vertex.successors.size(); }
    return actual.error.empty() && actual.protectEntry == selected &&
           bits(actual.summary.all) == expected.all && bits(actual.summary.exposed) == expected.exposed &&
           bits(actual.summary.suffix) == expected.suffix && actual.summary.transparent == expected.transparent &&
           actual.work.propagatedFacts <= 8 * graph.size() && actual.work.edgeFactVisits <= 8 * edges &&
           actual.work.externalEdgeFactVisits <= 4 * edges &&
           actual.work.edgeFactVisits == actual.work.internalEdgeFactVisits + actual.work.externalEdgeFactVisits &&
           actual.work.componentSolves == actual.work.components + actual.work.cyclicComponents;
}
} // namespace
int runCompletionTransferChecks()
{
    uint64_t checked = 0;
    const fs::CompletionSummary empty{facts(0), facts(0), facts(0), true};
    for (unsigned modes = 0; modes < 256; ++modes) {
        for (unsigned edges = 0; edges < 16; ++edges) {
            for (unsigned boundaries = 0; boundaries < 4; ++boundaries) {
                std::vector<fs::CompletionVertex> graph(5, {empty, {}, false});
                graph[0].successors = {1}; graph[1].successors = {2};
                graph[2].successors = {3}; graph[3].successors = {4};
                if (edges & 1) { graph[0].successors.push_back(4); }
                if (edges & 2) { graph[1].successors.push_back(3); }
                if (edges & 4) { graph[3].successors.push_back(1); }
                if (edges & 8) { graph[2].successors.push_back(2); }
                const unsigned a = modes & 15, b = modes >> 4;
                graph[1].summary = {facts(a), facts(a), facts(a), true};
                graph[2].summary = {facts(b), facts(b), facts(b), true};
                if (boundaries & 1) {
                    // A child completes after its exposed accesses; no suffix.
                    graph[2].summary.suffix = facts(0); graph[2].summary.transparent = false;
                }
                graph[3].requiresEntryCompletion = boundaries & 2;
                if (!check(graph)) { llvm::errs() << "completion execution oracle mismatch\n"; return 1; }
                ++checked;
            }
        }
    }
    std::vector<fs::CompletionVertex> emptyGraph(1, {empty, {}, false});
    auto zero = fs::buildCompletionTransfer(emptyGraph, 0, 0, 2);
    if (!zero.error.empty() || !zero.summary.transparent || bits(zero.summary.all)) { return 1; }
    auto invalid = emptyGraph;
    invalid[0].summary.exposed[0] = 2;
    if (fs::buildCompletionTransfer(invalid, 0, 0, 2).error.empty()) { return 1; }
    invalid = emptyGraph; invalid[0].successors = {1};
    if (fs::buildCompletionTransfer(invalid, 0, 0, 2).error.empty()) { return 1; }
    // Unreachable writer and forward prerequisite do not affect the result.
    std::vector<fs::CompletionVertex> unreachable(3, {empty, {}, false});
    unreachable[0].successors = {2};
    unreachable[1] = {{facts(2), facts(2), facts(2), true}, {2}, true};
    if (!check(unreachable)) { return 1; }
    std::vector<fs::CompletionVertex> mixed(4, {empty, {}, false});
    mixed[0].successors = {1};
    mixed[1] = {{facts(3), facts(1), facts(2), false}, {2}, false};
    mixed[2] = {{facts(12), facts(0), facts(8), false}, {3, 1}, false};
    if (!check(mixed)) { return 1; }
    // No terminating visit: effects remain sound but transparency is false.
    mixed[0].successors = {0};
    if (!check(mixed)) { return 1; }
    std::vector<fs::CompletionVertex> scalar(1, {{{}, {}, {}, true}, {}, true});
    auto required = fs::buildCompletionTransfer(scalar, 0, 0, 0);
    if (!required.error.empty() || !required.protectEntry[0] || required.summary.transparent ||
        required.work.propagatedFacts || required.work.edgeFactVisits) { return 1; }
    std::vector<fs::CompletionVertex> reads(4, {empty, {}, false});
    for (std::size_t v = 0; v + 1 < reads.size(); ++v) { reads[v].successors = {v + 1}; }
    reads[0].summary = {facts(2), facts(2), facts(2), true};
    reads[1].summary = reads[2].summary = {facts(1), facts(1), facts(1), true};
    const auto compact = fs::buildCompletionTransfer(reads, 0, 3, 2);
    if (!check(reads) || compact.protectEntry != std::vector<bool>({false, true, false, false}) ||
        compact.work.componentSolves != 4 || compact.work.cyclicComponents) { return 1; }
    std::mt19937 random(621);
    for (unsigned trial = 0; trial < 1000; ++trial) {
        std::vector<fs::CompletionVertex> graph(2 + random() % 7, {empty, {}, false});
        for (std::size_t v = 0; v + 1 < graph.size(); ++v) {
            const auto all = random() % 16;
            graph[v].summary = {facts(all), facts(all & random()), facts(all & random()), (random() % 2) != 0};
            graph[v].requiresEntryCompletion = random() % 7 == 0;
            for (std::size_t u = 0; u < graph.size(); ++u) {
                if (random() % 4 == 0) { graph[v].successors.push_back(u); }
            }
        }
        if (!check(graph)) { llvm::errs() << "random completion execution oracle mismatch\n"; return 1; }
    }
    // A small cycle with a long acyclic descendant chain must not trigger
    // repeated whole-graph solves or recursion on the host stack.
    std::vector<fs::CompletionVertex> chain(100000, {empty, {}, false});
    for (std::size_t v = 0; v + 1 < chain.size(); ++v) { chain[v].successors = {v + 1}; }
    chain[1].successors.push_back(0);
    chain[0].summary = {facts(2), facts(2), facts(2), true};
    const auto large = fs::buildCompletionTransfer(chain, 0, chain.size() - 1, 2);
    if (!large.error.empty() || large.work.components != chain.size() - 1 ||
        large.work.cyclicComponents != 1 || large.work.componentSolves != chain.size() ||
        large.work.externalEdgeFactVisits > chain.size() || !large.protectEntry[0]) { return 1; }
    llvm::outs() << "completion transfer: " << checked
                 << " execution graphs; SCC-local transfer; 1000 random graphs; 100000-node chain\n";
    return 0;
}
