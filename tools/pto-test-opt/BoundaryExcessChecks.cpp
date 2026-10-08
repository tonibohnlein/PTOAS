// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite event DAG oracle and analytical huge-count certificates.
#include "PTO/Transforms/FrontierSynch/BoundaryExcess.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Graph = fs::BoundaryPortClosure;
struct Payload { uint32_t region = 0, pipe = 0; };
Graph native(llvm::ArrayRef<Payload> word)
{
    Graph graph(2 * word.size(), std::vector<bool>(2 * word.size()));
    for (std::size_t a = 0; a < word.size(); ++a) {
        graph[2*a][2*a] = graph[2*a+1][2*a+1] = graph[2*a][2*a+1] = true;
        for (std::size_t b = a + 1; b < word.size(); ++b) {
            if (word[a].pipe == word[b].pipe) { graph[2*a][2*b] = graph[2*a+1][2*b+1] = true; }
        }
    }
    return graph;
}
void close(Graph& graph)
{
    for (std::size_t via = 0; via < graph.size(); ++via) {
        for (std::size_t source = 0; source < graph.size(); ++source) {
            for (std::size_t target = 0; target < graph.size(); ++target) {
                graph[source][target] = graph[source][target] || (graph[source][via] && graph[via][target]);
            }
        }
    }
}
fs::BoundaryExcessInput frame(llvm::ArrayRef<Payload> word, uint32_t regions, uint32_t pipes,
                             const Graph& lower, const Graph& upper)
{
    fs::BoundaryExcessInput input;
    input.pipes = pipes;
    input.regions.resize(regions);
    for (auto& region : input.regions) { region.counts.resize(pipes); region.exactInternalDifference = true; }
    std::vector<uint64_t> ranks;
    for (const auto& payload : word) {
        ranks.push_back(++input.regions[payload.region].counts[payload.pipe]);
        input.ports.push_back({payload.region, true}); input.ports.push_back({payload.region, true});
    }
    for (bool positive : {false, true}) {
        const auto& graph = positive ? upper : lower;
        auto& selectors = positive ? input.upper : input.lower;
        for (std::size_t port = 0; port < input.ports.size(); ++port) {
            selectors.prefix.emplace_back(pipes, 0);
            selectors.suffixExcluded.push_back(input.regions[input.ports[port].region].counts);
            for (std::size_t payload = 0; payload < word.size(); ++payload) {
                if (word[payload].region != input.ports[port].region) { continue; }
                auto& prefix = selectors.prefix.back()[word[payload].pipe];
                auto& excluded = selectors.suffixExcluded.back()[word[payload].pipe];
                if (graph[2*payload+1][port]) { prefix = std::max(prefix, ranks[payload]); }
                if (graph[port][2*payload]) { excluded = std::min(excluded, ranks[payload] - 1); }
            }
        }
    }
    for (std::size_t a = 0; a < word.size(); ++a) {
        for (std::size_t b = 0; b < word.size(); ++b) {
            if (word[a].region == word[b].region && upper[2*a+1][2*b] && !lower[2*a+1][2*b]) {
                input.regions[word[a].region].internalBound += 1;
            }
        }
    }
    return input;
}
fs::NumericalChainInterface index(llvm::ArrayRef<Payload> word, const Graph& graph)
{
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> groups;
    for (uint32_t payload = 0; payload < word.size(); ++payload) {
        groups[{word[payload].pipe, 0}].push_back(2 * payload);
        groups[{word[payload].pipe, 1}].push_back(2 * payload + 1);
    }
    std::vector<std::vector<uint32_t>> chains;
    for (const auto& group : groups) { chains.push_back(group.second); }
    return fs::buildNumericalChainInterface(chains, [&graph](uint32_t a, uint32_t b) -> std::optional<bool> {
        return graph[a][b];
    });
}
bool compare(llvm::ArrayRef<Payload> word, uint32_t regions, uint32_t pipes, Graph lower, Graph upper)
{
    close(lower); close(upper);
    const auto input = frame(word, regions, pipes, lower, upper);
    const auto general = fs::countBoundaryExcess(input, lower, upper);
    uint64_t difference = 0, internal = 0;
    for (std::size_t a = 0; a < word.size(); ++a) {
        for (std::size_t b = 0; b < word.size(); ++b) {
            if (lower[2*a+1][2*b] && !upper[2*a+1][2*b]) { return false; }
            if (upper[2*a+1][2*b] && !lower[2*a+1][2*b]) {
                ++difference; internal += word[a].region == word[b].region;
            }
        }
    }
    if (!general.error.empty() || general.total != difference || general.internal != internal ||
        general.cross != difference - internal || !general.exactDifference) { return false; }
    if (regions != 2) { return true; }
    const auto compressed = fs::countBinaryBoundaryExcess(input, index(word, lower), index(word, upper));
    if (!compressed.error.empty()) { llvm::errs() << compressed.error << "\n"; }
    return compressed.error.empty() && compressed.total == general.total && compressed.cross == general.cross &&
        compressed.rectangles <= 4 * uint64_t(pipes) * pipes * pipes * input.ports.size();
}
bool graphs(uint64_t& checked)
{
    const std::vector<Payload> word{{0, 0}, {0, 1}, {0, 0}, {1, 1}, {1, 0}, {1, 1}};
    const std::pair<uint32_t, uint32_t> edges[]{{0, 1}, {1, 2}, {0, 3}, {2, 3}, {2, 5}, {3, 4}, {4, 5}};
    for (uint32_t mask = 0; mask < 128; ++mask) {
        auto lower = native(word), upper = native(word);
        for (uint32_t edge = 0; edge < 7; ++edge) {
            const auto [a, b] = edges[edge];
            if ((mask >> edge) & 1U) { upper[2*a+1][2*b] = true; }
            if (((mask >> edge) & 1U) && edge % 2) { lower[2*a+1][2*b] = true; }
        }
        if (!compare(word, 2, 2, lower, upper)) { llvm::errs() << "boundary mask " << mask << "\n"; return false; }
        ++checked;
    }
    // Empty middle region: common native chains still carry dependencies across it.
    const std::vector<Payload> empty{{0, 0}, {0, 1}, {2, 0}, {2, 1}};
    auto low = native(empty), high = low;
    high[1][2] = true; low[5][6] = high[5][6] = true;
    return compare(empty, 3, 2, low, high);
}
bool exactCrossing()
{
    for (uint32_t count : {1, 2, 5}) {
        std::vector<Payload> word{{0, 0}, {0, 1}, {0, 0}, {0, 1}};
        word.insert(word.end(), count, Payload{1, 2});
        auto lower = native(word), upper = lower;
        lower[1][2] = lower[1][6] = true;
        upper[1][2] = upper[5][6] = true;
        lower[7][8] = upper[7][8] = true; // The crossing itself is EXACT.
        close(lower); close(upper);
        const auto input = frame(word, 2, 3, lower, upper);
        const auto result = fs::countBinaryBoundaryExcess(input, index(word, lower), index(word, upper));
        if (!result.error.empty() || result.internal != 1 || result.cross != count || result.total != count + 1 ||
            !compare(word, 2, 3, lower, upper)) { return false; }
    }
    return true;
}
bool rectangleOracle()
{
    for (uint32_t code = 0; code < 256; ++code) {
        uint32_t remaining = code;
        std::vector<fs::BoundaryRankRectangle> rectangles;
        for (unsigned i = 0; i < 4; ++i) {
            const auto choice = remaining % 4; remaining /= 4;
            rectangles.push_back({1 + choice, (choice + i) % 5});
        }
        uint64_t expected = 0;
        for (uint64_t a = 1; a <= 4; ++a) {
            for (uint64_t b = 1; b <= 4; ++b) {
                bool present = false;
                for (const auto& rectangle : rectangles) {
                    present |= a <= rectangle.sourcePrefix && b > rectangle.targetExcluded;
                }
                expected += present;
            }
        }
        const auto actual = fs::countBoundaryRectangles(rectangles, 4);
        if (!actual.error.empty() || actual.pairs != expected) { return false; }
    }
    return fs::countBoundaryRectangles({}, 0).pairs == 0 &&
        !fs::countBoundaryRectangles({{1, 2}}, 1).error.empty();
}
bool hugeAndMalformed()
{
    const auto n = UINT64_MAX;
    fs::BoundaryExcessInput input;
    input.pipes = 2;
    input.regions = {{{n, 0}, llvm::APInt(256, 0), true}, {{0, n}, llvm::APInt(256, 0), true}};
    input.ports = {{0, true}, {1, true}};
    input.lower.prefix = input.upper.prefix = {{n, 0}, {0, 0}};
    input.lower.suffixExcluded = input.upper.suffixExcluded = {{n, 0}, {0, 0}};
    const Graph lower{{true, false}, {false, true}}, upper{{true, true}, {false, true}};
    const auto result = fs::countBoundaryExcess(input, lower, upper);
    const auto expected = llvm::APInt(256, n) * llvm::APInt(256, n);
    if (!result.error.empty() || result.cross != expected || result.total != expected ||
        result.total.getActiveBits() != 128) { return false; }
    auto bad = input; bad.lower.prefix[0][0] = 0; bad.upper.prefix[0][0] = 1;
    if (fs::countBoundaryExcess(bad, upper, lower).error.empty()) { return false; }
    bad = input; bad.lower.suffixExcluded[1][1] = n; bad.upper.suffixExcluded[1][1] = n - 1;
    bad.regions[0].internalBound = llvm::APInt::getMaxValue(256);
    if (fs::countBoundaryExcess(bad, lower, upper).error.empty()) { return false; }
    bad = input; bad.lower.prefix[0].pop_back();
    if (fs::countBoundaryExcess(bad, lower, upper).error.empty()) { return false; }
    bad = input; bad.ports[0].present = false;
    if (fs::countBoundaryExcess(bad, lower, upper).error.empty()) { return false; }
    bad = input; bad.pipes = 0;
    if (fs::countBoundaryExcess(bad, lower, upper).error.empty()) { return false; }
    bad = input; bad.ports[0].region = 2;
    if (fs::countBoundaryExcess(bad, lower, upper).error.empty()) { return false; }
    const std::vector<Payload> word{{0, 0}, {1, 0}};
    auto graph = native(word); close(graph);
    auto empty = frame(word, 2, 1, graph, graph);
    auto goodIndex = index(word, graph), badIndex = goodIndex;
    badIndex.reverse[0].pop_back();
    if (fs::countBinaryBoundaryExcess(empty, badIndex, goodIndex).error.empty()) { return false; }
    badIndex = goodIndex; badIndex.rank[0] = UINT32_MAX;
    if (fs::countBinaryBoundaryExcess(empty, goodIndex, badIndex).error.empty()) { return false; }
    badIndex = goodIndex; badIndex.reverse[2][0] = 0;
    if (fs::countBinaryBoundaryExcess(empty, badIndex, goodIndex).error.empty()) { return false; }
    return fs::countBoundaryExcess({}, {}, {}).error.empty();
}
bool hierarchy()
{
    const std::vector<Payload> word{{0, 0}, {0, 1}, {1, 1}, {2, 0}, {3, 1}, {3, 0}};
    auto lower = native(word), upper = lower;
    lower[5][6] = upper[5][6] = true;
    lower[7][8] = upper[7][8] = true;
    upper[1][2] = upper[3][4] = upper[9][10] = true;
    close(lower); close(upper);
    const auto leaves = frame(word, 4, 2, lower, upper);
    const auto flat = fs::countBoundaryExcess(leaves, lower, upper);
    auto rootWord = word;
    for (auto& payload : rootWord) { payload.region = payload.region < 2 ? 0 : 1; }
    auto rootInput = frame(rootWord, 2, 2, lower, upper);
    llvm::APInt childTotal(256, 0);
    for (uint32_t child = 0; child < 2; ++child) {
        std::vector<uint32_t> chosen;
        std::vector<Payload> childWord;
        for (uint32_t i = 0; i < word.size(); ++i) {
            if (rootWord[i].region == child) {
                chosen.push_back(i);
                childWord.push_back({word[i].region % 2, word[i].pipe});
            }
        }
        Graph a(2 * chosen.size(), std::vector<bool>(2 * chosen.size())), b = a;
        for (std::size_t i = 0; i < a.size(); ++i) {
            for (std::size_t j = 0; j < a.size(); ++j) {
                a[i][j] = lower[2*chosen[i/2]+i%2][2*chosen[j/2]+j%2];
                b[i][j] = upper[2*chosen[i/2]+i%2][2*chosen[j/2]+j%2];
            }
        }
        auto childInput = frame(childWord, 2, 2, a, b);
        auto result = fs::countBinaryBoundaryExcess(childInput, index(childWord, a), index(childWord, b));
        if (!result.error.empty() || result.total != rootInput.regions[child].internalBound) { return false; }
        rootInput.regions[child].internalBound = result.total;
        childTotal += result.total;
    }
    const auto root = fs::countBinaryBoundaryExcess(rootInput, index(rootWord, lower), index(rootWord, upper));
    return flat.error.empty() && root.error.empty() && root.total == flat.total &&
        root.total == childTotal + root.cross && root.exactDifference && compare(word, 4, 2, lower, upper);
}
} // namespace
int runBoundaryExcessChecks()
{
    uint64_t checked = 0;
    if (!rectangleOracle() || !graphs(checked) || !exactCrossing() || !hugeAndMalformed() || !hierarchy()) {
        llvm::errs() << "boundary excess checks failed\n"; return 1;
    }
    llvm::outs() << "boundary excess checks passed: " << checked << " finite DAG bounds\n";
    return 0;
}
