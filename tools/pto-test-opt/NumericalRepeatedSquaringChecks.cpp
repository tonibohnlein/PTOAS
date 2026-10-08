// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Small independent occurrence graphs verify the shared definition algebra.
#include "PTO/Transforms/FrontierSynch/NumericalRepeatedSquaring.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <set>
namespace fs = mlir::pto::frontiersynch;
namespace {
using Matrix = std::vector<std::vector<bool>>;
using Link = fs::NumericalRepeatedLink;
void close(Matrix& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) {
                graph[a][b] = graph[a][b] || (graph[a][k] && graph[k][b]);
            }
        }
    }
}
Matrix unfold(const Matrix& leaf, const std::vector<Link>& links, uint64_t copies)
{
    const auto p = leaf.size();
    Matrix graph(p * copies, std::vector<bool>(p * copies, false));
    for (uint64_t copy = 0; copy < copies; ++copy) {
        for (std::size_t a = 0; a < p; ++a) {
            for (std::size_t b = 0; b < p; ++b) { graph[copy * p + a][copy * p + b] = leaf[a][b]; }
        }
        if (copy + 1 < copies) {
            for (const auto& edge : links) { graph[copy * p + edge.source][(copy + 1) * p + edge.target] = true; }
        }
    }
    close(graph); return graph;
}
using Edges = std::set<std::pair<uint64_t, uint64_t>>;
void expand(const fs::NumericalRepeatedNode& node, uint64_t offset, uint32_t ports, Edges& demands)
{
    for (const auto& seam : node.seams) {
        if (!seam.native) {
            demands.emplace((offset + seam.source.copy) * ports + seam.source.port,
                            (offset + seam.target.copy) * ports + seam.target.port);
        }
    }
    if (node.merge) {
        expand(*node.children[0], offset, ports, demands);
        expand(*node.children[1], offset + node.children[0]->copies, ports, demands);
    }
}
bool check(const Matrix& body, std::vector<std::vector<uint32_t>> chains,
           const std::vector<Link>& links, uint64_t copies, uint64_t& queries)
{
    const uint32_t p = static_cast<uint32_t>(body.size());
    auto leaf = std::make_shared<fs::NumericalChainInterface>(fs::buildNumericalChainInterface(std::move(chains),
        [&body](uint32_t a, uint32_t b) -> std::optional<bool> { return body[a][b]; }));
    auto result = fs::buildNumericalRepeatedSquaring(leaf, links, copies);
    if (!result.error.empty() || !result.root || result.root->copies != copies ||
        result.root->ports.size() > 2 * p) {
        llvm::errs() << "numerical squaring: " << result.error << "\n"; return false;
    }
    auto graph = unfold(body, links, copies);
    for (uint32_t a = 0; a < result.root->ports.size(); ++a) {
        const auto& source = result.root->ports[a];
        for (uint32_t b = 0; b < result.root->ports.size(); ++b) {
            const auto& target = result.root->ports[b];
            const bool expected = graph[source.copy * p + source.port][target.copy * p + target.port];
            if (result.root->index->reaches(a, b) != expected) {
                llvm::errs() << "shared boundary index mismatch\n"; return false;
            }
        }
    }
    Edges actual, expected;
    expand(*result.root, 0, p, actual);
    for (uint64_t copy = 0; copy + 1 < copies; ++copy) {
        for (const auto& link : result.links) {
            if (link.native) { continue; }
            const auto a = copy * p + link.source, b = (copy + 1) * p + link.target;
            bool redundant = false;
            for (std::size_t z = 0; z < graph.size(); ++z) {
                if (z != a && z != b) { redundant |= graph[a][z] && graph[z][b]; }
            }
            if (!redundant) { expected.emplace(a, b); }
        }
    }
    if (actual != expected) { llvm::errs() << "shared seam demand mismatch\n"; return false; }
    fs::NumericalSquaringQueryCost cost;
    for (std::size_t a = 0; a < graph.size(); ++a) {
        for (std::size_t b = 0; b < graph.size(); ++b) {
            auto answer = result.query(a / p, a % p, b / p, b % p, cost);
            ++queries;
            if (!answer || *answer != graph[a][b]) { llvm::errs() << "shared query mismatch\n"; return false; }
            if (a / p != b / p) {
                auto arbitrary = result.across(a / p, leaf->forward[a % p], b / p, leaf->reverse[b % p], cost);
                if (!arbitrary || *arbitrary != graph[a][b]) { return false; }
            }
        }
        for (bool reverse : {false, true}) {
            auto thresholds = result.thresholds(a / p, reverse ? leaf->reverse[a % p] : leaf->forward[a % p],
                                                reverse, cost);
            if (!thresholds) { return false; }
            const auto& index = *result.root->index;
            for (uint32_t c = 0; c < index.chains.size(); ++c) {
                uint32_t expected = reverse ? 0 : static_cast<uint32_t>(index.chains[c].size());
                for (uint32_t rank = 0; rank < index.chains[c].size(); ++rank) {
                    const auto& point = result.root->ports[index.chains[c][rank]];
                    const auto b = point.copy * p + point.port;
                    if (reverse ? graph[b][a] : graph[a][b]) {
                        expected = reverse ? rank + 1 : std::min(expected, rank);
                    }
                }
                if ((*thresholds)[c] != expected) { llvm::errs() << "shared projection mismatch\n"; return false; }
            }
        }
    }
    if (result.query(copies, 0, 0, 0, cost).has_value() ||
        result.across(0, {}, 0, {}, cost).has_value()) { return false; }
    // Snapshot lifetime: callers may mutate or release their original index.
    leaf->forward.clear(); leaf->chains.clear();
    if (copies && result.query(0, 0, copies - 1, 0, cost) != true) { return false; }
    return true;
}
bool largeAndInvalid()
{
    auto leaf = std::make_shared<fs::NumericalChainInterface>(fs::buildNumericalChainInterface({{0, 1}},
        [](uint32_t a, uint32_t b) -> std::optional<bool> { return a <= b; }));
    std::vector<Link> links{{1, 0, true}, {1, 0, false}};
    auto large = fs::buildNumericalRepeatedSquaring(leaf, links, UINT64_MAX);
    if (!large.error.empty() || !large.root || large.root->copies != UINT64_MAX || large.cost.merges > 126 ||
        large.cost.definitions != large.cost.merges + 1 || large.cost.boundaryPorts > 4 * large.cost.definitions ||
        large.links.size() != 1 || !large.links[0].native) { return false; }
    std::set<const fs::NumericalRepeatedNode*> unique;
    std::vector<const fs::NumericalRepeatedNode*> pending{large.root.get()};
    while (!pending.empty()) {
        auto* node = pending.back(); pending.pop_back();
        if (!unique.insert(node).second) { continue; }
        if (node->ports.size() > 4) { return false; }
        if (node->merge) {
            pending.push_back(node->children[0].get()); pending.push_back(node->children[1].get());
        }
    }
    if (unique.size() != large.cost.definitions) { return false; }
    fs::NumericalSquaringQueryCost cost;
    if (large.query(0, 1, UINT64_MAX - 1, 0, cost) != true ||
        large.query(UINT64_MAX - 1, 0, 0, 1, cost) != false || cost.nodes > 512) { return false; }
    cost.nodes = UINT64_MAX;
    if (large.query(0, 0, 0, 0, cost).has_value()) { return false; }
    cost = {}; cost.indexOperations = UINT64_MAX;
    if (large.query(0, 0, 0, 1, cost).has_value()) { return false; }
    if (fs::buildNumericalRepeatedSquaring({}, links, 2).error.empty() ||
        fs::buildNumericalRepeatedSquaring(leaf, {}, 2).error.empty() ||
        fs::buildNumericalRepeatedSquaring(leaf, {{2, 0, true}}, 2).error.empty()) { return false; }
    auto invalid = std::make_shared<fs::NumericalChainInterface>(*leaf); invalid->reverse[0].clear();
    if (fs::buildNumericalRepeatedSquaring(invalid, links, 2).error.empty()) { return false; }
    auto empty = fs::buildNumericalRepeatedSquaring(
        std::make_shared<const fs::NumericalChainInterface>(), {}, UINT64_MAX);
    if (!empty.error.empty() || !empty.root || !empty.root->ports.empty() || empty.cost.merges > 126) { return false; }
    // A shared fragment can be the next repetition's leaf. No expanded inner
    // events are required by the builder; its owner stays available to callers.
    auto inner = fs::buildNumericalRepeatedSquaring(leaf, links, 7);
    std::vector<Link> outerLinks;
    for (const auto& chain : inner.root->index->chains) { outerLinks.push_back({chain.back(), chain.front(), true}); }
    auto outer = fs::buildNumericalRepeatedSquaring(inner.root->index, outerLinks, UINT64_MAX);
    return outer.error.empty() && outer.root && outer.root->ports.size() == 8 && outer.cost.merges <= 126;
}
} // namespace
int runNumericalRepeatedSquaringChecks()
{
    if (!largeAndInvalid()) { llvm::errs() << "numerical squaring boundary checks failed\n"; return 1; }
    uint64_t queries = 0, cases = 0;
    for (uint32_t p = 1; p <= 5; ++p) {
        for (uint32_t c = 1; c <= std::min(p, 3U); ++c) {
            std::vector<std::vector<uint32_t>> chains(c);
            Matrix body(p, std::vector<bool>(p, false));
            for (uint32_t i = 0; i < p; ++i) { chains[i % c].push_back(i); body[i][i] = true; }
            for (const auto& chain : chains) {
                for (std::size_t i = 1; i < chain.size(); ++i) { body[chain[i - 1]][chain[i]] = true; }
            }
            for (unsigned variant = 0; variant < 4; ++variant) {
                auto graph = body;
                for (uint32_t a = 0; a < p; ++a) {
                    for (uint32_t b = a + 1; b < p; ++b) {
                        if ((a + 3 * b + variant) % 4 == 0) { graph[a][b] = true; }
                    }
                }
                close(graph);
                std::vector<Link> links;
                for (const auto& chain : chains) { links.push_back({chain.back(), chain.front(), true}); }
                for (uint32_t a = 0; a < p; ++a) {
                    links.push_back({a, (a + variant) % p, false});
                }
                links.push_back(links.back());
                for (uint64_t copies : {0u, 1u, 2u, 3u, 5u, 6u}) {
                    if (!check(graph, chains, links, copies, queries)) {
                        llvm::errs() << "numerical squaring failed P=" << p << " c=" << c
                                     << " variant=" << variant << " T=" << copies << "\n"; return 1;
                    }
                    ++cases;
                }
            }
        }
    }
    llvm::outs() << "numerical squaring: " << cases << " graphs and " << queries << " query pairs passed\n";
    return 0;
}
