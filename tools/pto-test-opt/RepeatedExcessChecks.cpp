// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RepeatedExcess.h"
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
namespace fs = mlir::pto::frontiersynch;
namespace {
using Graph = std::vector<std::vector<bool>>;
using Kind = fs::PeriodicEventKind;
llvm::APInt wide(uint64_t value) { return llvm::APInt(256, value); }
Graph unfold(llvm::ArrayRef<fs::PeriodicRecord> edges, uint64_t occurrences)
{
    Graph graph(2*occurrences, std::vector<bool>(2*occurrences));
    for (uint64_t a = 0; a < occurrences; ++a) {
        graph[2*a][2*a] = graph[2*a+1][2*a+1] = graph[2*a][2*a+1] = true;
        for (uint64_t b = a + 1; b < occurrences; ++b) {
            if (a % 2 == b % 2) { graph[2*a][2*b] = graph[2*a+1][2*b+1] = true; }
        }
    }
    for (auto edge : edges) {
        for (uint64_t iteration = 0; 4*iteration+edge.source < occurrences; ++iteration) {
            const auto target = 4*(iteration+edge.displacement)+edge.target;
            if (target < occurrences) { graph[2*(4*iteration+edge.source)+1][2*target] = true; }
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
uint64_t difference(const Graph& upper, const Graph& lower)
{
    uint64_t result = 0;
    for (std::size_t a = 1; a < upper.size(); a += 2) {
        for (std::size_t b = 0; b < upper.size(); b += 2) {
            if (upper[a][b] && !lower[a][b]) { ++result; }
        }
    }
    return result;
}
fs::BoundaryExcessSelectors selectors(const Graph& graph)
{
    fs::BoundaryExcessSelectors result;
    result.prefix.assign(8, std::vector<uint64_t>(2));
    result.suffixExcluded.assign(8, std::vector<uint64_t>(2, 2));
    for (uint32_t port = 0; port < 8; ++port) {
        for (uint32_t payload = 0; payload < 4; ++payload) {
            const auto pipe = payload % 2, rank = payload / 2;
            if (graph[2*payload+1][port]) { result.prefix[port][pipe] = rank + 1; }
            if (graph[port][2*payload]) {
                result.suffixExcluded[port][pipe] = std::min(result.suffixExcluded[port][pipe], uint64_t(rank));
            }
        }
    }
    return result;
}
fs::RepeatedPortDistances distances(llvm::ArrayRef<fs::PeriodicRecord> edges)
{
    const auto graph = fs::analyzePeriodicDemands({{0}, {1}, {0}, {1}}, edges);
    fs::RepeatedPortDistances result(8, std::vector<std::optional<uint64_t>>(8));
    for (uint32_t a = 0; a < 8; ++a) {
        for (uint32_t b = 0; b < 8; ++b) {
            const auto value = graph.eventThreshold({a/2, a%2 ? Kind::Completion : Kind::Start},
                {b/2, b%2 ? Kind::Completion : Kind::Start});
            result[a][b] = value.displacement;
        }
    }
    return result;
}
fs::RepeatedChainFrontiers compress(const fs::RepeatedPortDistances& lower, const fs::RepeatedPortDistances& upper)
{
    fs::RepeatedChainFrontiers result;
    result.chains = {{0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (const auto* distances : {&lower, &upper}) {
        std::vector<std::vector<std::optional<uint64_t>>> rows(4, std::vector<std::optional<uint64_t>>(8));
        for (std::size_t chain = 0; chain < 4; ++chain) {
            for (uint32_t rank = 0; rank < 2; ++rank) {
                for (uint32_t target = 0; target < 8; ++target) {
                    const auto d = (*distances)[result.chains[chain][rank]][target];
                    if (!d) { continue; }
                    const auto rho = 1 - rank + 2*(*d);
                    auto& value = rows[chain][target];
                    value = value ? std::min(*value, rho) : rho;
                }
            }
        }
        if (distances == &lower) { result.lower = std::move(rows); }
        else { result.upper = std::move(rows); }
    }
    return result;
}
bool finite()
{
    const std::vector<fs::PeriodicRecord> lower{{1, 0, 1}}; // Exact crossing in both graphs.
    const std::vector<fs::PeriodicRecord> optional{{0, 1, 0}, {2, 3, 0}, {3, 0, 2}};
    const auto lo = distances(lower);
    for (uint32_t mask = 0; mask < 8; ++mask) {
        auto upper = lower;
        for (uint32_t bit = 0; bit < optional.size(); ++bit) {
            if (mask & (1U << bit)) { upper.push_back(optional[bit]); }
        }
        fs::RepeatedExcessInput input;
        input.period = {{2, 2}, wide(difference(unfold(upper, 4), unfold(lower, 4))), true};
        input.prefixes = {{{0, 0}, wide(0), true},
            {{1, 1}, wide(difference(unfold(upper, 2), unfold(lower, 2))), true}};
        input.present.assign(8, true);
        input.lower = selectors(unfold(lower, 4)); input.upper = selectors(unfold(upper, 4));
        const auto hi = distances(upper);
        const auto chains = compress(lo, hi);
        for (uint64_t periods = 0; periods < 5; ++periods) {
            const auto general = fs::countRepeatedExcess(input, lo, hi, periods);
            const auto compressed = fs::countChainRepeatedExcess(input, chains, periods);
            if (!general.error.empty() || !compressed.error.empty() || general.prefixes.size() != 2 ||
                compressed.thresholdEntries != 32) { return false; }
            if (mask == 1 && periods == 2 && general.prefixes[0].cross.isZero()) { return false; }
            for (uint64_t prefix = 0; prefix < 2; ++prefix) {
                const auto expected = difference(unfold(upper, 4*periods+2*prefix), unfold(lower, 4*periods+2*prefix));
                if (general.prefixes[prefix].total != expected || !general.prefixes[prefix].exactDifference ||
                    compressed.prefixes[prefix].total != general.prefixes[prefix].total) { return false; }
            }
        }
        auto malformed = chains;
        malformed.upper[0][0] = std::nullopt;
        if (fs::countChainRepeatedExcess(input, malformed, 0).error.empty()) { return false; }
        malformed = chains; malformed.chains[1][0] = malformed.chains[0][0];
        if (fs::countChainRepeatedExcess(input, malformed, 0).error.empty()) { return false; }
    }
    return true;
}
bool rectangleOracle()
{
    for (uint64_t seed = 0; seed < 128; ++seed) {
        std::vector<fs::ThresholdRankRectangle> rectangles;
        uint64_t state = seed + 1;
        for (uint64_t i = 0; i < 6; ++i) {
            state = state * 17 + 13;
            rectangles.push_back({state%5, (state/7)%5, (state/11)%6});
        }
        for (uint64_t n = 0; n < 7; ++n) {
            for (bool full : {false, true}) {
                uint64_t expected = 0;
                for (uint64_t s = 1; s <= n; ++s) {
                    for (uint64_t x = 1; x <= 4; ++x) {
                        for (uint64_t y = 1; y <= 4; ++y) {
                            bool present = false;
                            for (auto rectangle : rectangles) {
                                present |= rectangle.activation <= s && x <= rectangle.width && y <= rectangle.height;
                            }
                            if (present) { expected += full ? n-s : 1; }
                        }
                    }
                }
                auto actual = fs::sumThresholdRectangles(rectangles, n, full);
                if (!actual.error.empty() || actual.weighted != expected ||
                    actual.skylineUpdates > 2*rectangles.size()) {
                    return false;
                }
            }
        }
    }
    return true;
}
fs::RepeatedExcessInput nested(uint64_t count)
{
    fs::RepeatedExcessInput input;
    input.period = {{count, count}, wide(0), true};
    input.prefixes = {{{0, 0}, wide(0), true}, {{count, 0}, wide(0), true}, {{count, count}, wide(0), true}};
    input.present = {true, true};
    // Only the last P completion and first Q start are needed, regardless of
    // the inner payload count. Both signs have native-only period interiors.
    input.lower.prefix = {{count, 0}, {0, 0}};
    input.lower.suffixExcluded = {{count, count}, {count, 0}};
    input.upper = input.lower;
    return input;
}
bool hugeAndMalformed()
{
    constexpr uint64_t delay = uint64_t(1) << 40;
    fs::RepeatedPortDistances lower{{0, std::nullopt}, {std::nullopt, 0}}, upper = lower;
    upper[0][1] = delay;
    for (uint64_t count : {uint64_t(1), uint64_t(3), uint64_t(1) << 40, UINT64_MAX}) {
        const auto input = nested(count);
        for (uint64_t n : {uint64_t(0), delay-1, delay, delay+1, UINT64_MAX}) {
            auto result = fs::countRepeatedExcess(input, lower, upper, n);
            if (!result.error.empty()) { return false; }
            const auto tail = n > delay ? n-delay : 0;
            // Divide the triangular factor before multiplying a huge square.
            const auto exactFull = (wide(tail)*(wide(tail)+1)).udiv(wide(2))*wide(count)*wide(count);
            const auto partial = n >= delay ? wide(n-delay+1)*wide(count)*wide(count) : wide(0);
            if (result.prefixes[0].total != exactFull || result.prefixes[1].total != exactFull ||
                result.prefixes[2].total != exactFull + partial) { return false; }
        }
    }
    auto input = nested(2);
    fs::RepeatedExcessInput empty;
    empty.period.exactInternalDifference = true;
    empty.prefixes = {{{}, wide(0), true}};
    const auto emptyResult = fs::countRepeatedExcess(empty, {}, {}, UINT64_MAX);
    const auto emptyChain = fs::countChainRepeatedExcess(empty, {}, UINT64_MAX);
    if (!emptyResult.error.empty() || !emptyChain.error.empty() || emptyResult.prefixes[0].total != 0 ||
        !emptyResult.prefixes[0].exactDifference) { return false; }
    auto bad = lower; bad[0][1] = 0;
    if (fs::countRepeatedExcess(input, bad, upper, 0).error.empty()) { return false; }
    bad = upper; bad[0].clear();
    if (fs::countRepeatedExcess(input, lower, bad, 1).error.empty()) { return false; }
    input.prefixes[0].counts[0] = 1;
    if (fs::countRepeatedExcess(input, lower, upper, 1).error.empty()) { return false; }
    input = nested(2); input.period.internalBound = llvm::APInt::getMaxValue(256);
    if (fs::countRepeatedExcess(input, lower, upper, 2).error.empty()) { return false; }
    input = nested(2); input.lower.prefix[1][0] = 1;
    if (fs::countRepeatedExcess(input, lower, upper, 1).error.empty()) { return false; }
    input = nested(2);
    input.present.push_back(false);
    for (auto* selectors : {&input.lower, &input.upper}) {
        selectors->prefix.push_back({0, 0}); selectors->suffixExcluded.push_back({2, 2});
    }
    for (auto* matrix : {&lower, &upper}) {
        for (auto& row : *matrix) { row.push_back(std::nullopt); }
        matrix->push_back({std::nullopt, std::nullopt, std::nullopt});
    }
    auto absent = fs::countRepeatedExcess(input, lower, upper, delay+1);
    if (!absent.error.empty() || absent.prefixes[0].total != 4) { return false; }
    input.upper.prefix.back()[0] = 1;
    return !fs::countRepeatedExcess(input, lower, upper, delay+1).error.empty();
}
} // namespace
int runRepeatedExcessChecks()
{
    if (!rectangleOracle()) { llvm::errs() << "threshold rectangle sweep oracle failed\n"; return 1; }
    if (!finite()) { llvm::errs() << "repeated excess finite graph oracle failed\n"; return 1; }
    if (!hugeAndMalformed()) { llvm::errs() << "repeated excess huge/malformed checks failed\n"; return 1; }
    llvm::outs() << "repeated excess checks passed: finite graph, compressed frontier, threshold and huge-count "
                 << "oracles\n";
    return 0;
}
