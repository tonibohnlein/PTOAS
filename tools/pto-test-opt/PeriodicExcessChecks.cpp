// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite event closure and all-type threshold/profile oracles.
#include "PTO/Transforms/FrontierSynch/PeriodicExcessCertificate.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Matrix = std::vector<std::vector<uint64_t>>;
constexpr uint64_t Infinity = UINT64_MAX / 4;
llvm::APInt wide(uint64_t value) { return llvm::APInt(256, value); }
void close(Matrix& graph)
{
    for (std::size_t middle = 0; middle < graph.size(); ++middle) {
        for (std::size_t from = 0; from < graph.size(); ++from) {
            for (std::size_t to = 0; to < graph.size(); ++to) {
                graph[from][to] = std::min(graph[from][to], graph[from][middle] + graph[middle][to]);
            }
        }
    }
}
Matrix thresholdOracle(const fs::PeriodicAnalysis& analysis)
{
    const uint32_t m = static_cast<uint32_t>(analysis.payloads.size());
    Matrix graph(2 * m, std::vector<uint64_t>(2 * m, Infinity));
    auto edge = [&graph](uint32_t from, uint32_t to, uint64_t distance) {
        graph[from][to] = std::min(graph[from][to], distance);
    };
    for (uint32_t a = 0; a < m; ++a) {
        edge(2 * a, 2 * a, 0); edge(2 * a + 1, 2 * a + 1, 0); edge(2 * a, 2 * a + 1, 0);
        // All native pairs, independently of production adjacent-chain assembly.
        for (uint32_t b = 0; b < m; ++b) {
            if (analysis.payloads[a].pipe == analysis.payloads[b].pipe) {
                edge(2 * a, 2 * b, a < b ? 0 : 1);
                edge(2 * a + 1, 2 * b + 1, a < b ? 0 : 1);
            }
        }
    }
    for (const auto& record : analysis.generators) {
        edge(2 * record.source + 1, 2 * record.target, record.displacement);
    }
    for (const auto& record : analysis.nativePrerequisites) {
        edge(2 * record.source + 1, 2 * record.target, record.displacement);
    }
    close(graph);
    return graph;
}
Matrix finiteOracle(const fs::PeriodicAnalysis& analysis, uint32_t trips)
{
    const uint32_t m = static_cast<uint32_t>(analysis.payloads.size()), n = m * trips;
    Matrix graph(2 * n, std::vector<uint64_t>(2 * n, Infinity));
    for (uint32_t a = 0; a < n; ++a) {
        graph[2 * a][2 * a] = graph[2 * a + 1][2 * a + 1] = graph[2 * a][2 * a + 1] = 0;
        for (uint32_t b = a + 1; b < n; ++b) {
            if (analysis.payloads[a % m].pipe == analysis.payloads[b % m].pipe) {
                graph[2 * a][2 * b] = graph[2 * a + 1][2 * b + 1] = 0;
            }
        }
    }
    auto add = [&graph, m, trips](llvm::ArrayRef<fs::PeriodicRecord> records) {
        for (const auto& record : records) {
            for (uint32_t t = 0; t < trips; ++t) {
                if (record.displacement < trips - t) {
                    graph[2 * (t * m + record.source) + 1]
                         [2 * ((t + record.displacement) * m + record.target)] = 0;
                }
            }
        }
    };
    add(analysis.generators); add(analysis.nativePrerequisites); close(graph);
    return graph;
}
llvm::APInt pairs(uint64_t threshold, uint64_t trips)
{
    if (threshold == Infinity || threshold >= trips) { return wide(0); }
    const auto count = wide(trips - threshold);
    return (count * (count + 1)).udiv(wide(2));
}
bool compare(const fs::PeriodicAnalysis& upper, const fs::PeriodicAnalysis& lower, uint32_t trips)
{
    const auto result = fs::analyzePeriodicExcess(upper, lower, trips);
    if (!result.error.empty()) { llvm::errs() << result.error << "\n"; return false; }
    const auto a = thresholdOracle(upper), b = thresholdOracle(lower);
    const auto fa = finiteOracle(upper, trips), fb = finiteOracle(lower, trips);
    llvm::APInt expected = wide(0), gamma = wide(0), quadratic = wide(0);
    bool equal = true;
    for (std::size_t source = 0; source < upper.payloads.size(); ++source) {
        for (std::size_t target = 0; target < upper.payloads.size(); ++target) {
            const auto x = a[2 * source + 1][2 * target], y = b[2 * source + 1][2 * target];
            if (y < x) { return false; }
            expected += pairs(x, trips) - pairs(y, trips);
            equal &= x == y;
            if (x != Infinity && y == Infinity) { quadratic += 1; }
            if (x != Infinity && y != Infinity) { gamma += wide(y - x); }
        }
    }
    uint64_t finite = 0;
    for (std::size_t source = 0; source < fa.size() / 2; ++source) {
        for (std::size_t target = 0; target < fa.size() / 2; ++target) {
            const bool up = fa[2 * source + 1][2 * target] != Infinity;
            const bool low = fb[2 * source + 1][2 * target] != Infinity;
            if (low && !up) { return false; }
            finite += up && !low;
        }
    }
    const uint64_t m = upper.payloads.size(), k = upper.frontiers.size();
    return result.excess == expected && result.excess == finite && result.quadraticPairs == quadratic &&
        result.sameSupport == quadratic.isZero() && result.equalFrontiers == equal &&
        result.frontierEntries == m * k && result.supportSteps == 2 * k * k * k &&
        (result.sameSupport ? result.gamma == gamma && result.linearBound == gamma * wide(trips) :
                              !result.gamma && !result.linearBound) &&
        (!result.linearBound || result.excess.ule(*result.linearBound)) &&
        (!result.counterpartBound || result.excess.ule(*result.counterpartBound));
}
bool exhaustive(uint64_t& cases)
{
    const std::vector<fs::PeriodicPayload> word{{9}, {2}, {9}};
    const std::vector<fs::PeriodicRecord> edges{{0, 1, 0}, {1, 2, 0}, {2, 0, 1}, {0, 2, 0}};
    for (uint32_t mask = 0; mask < 81; ++mask) {
        uint32_t choice = mask;
        std::vector<fs::PeriodicRecord> upper, lower;
        for (auto edge : edges) {
            const auto mode = choice % 3; choice /= 3;
            if (!mode) { continue; }
            upper.push_back(edge);
            if (mode == 2) { edge.displacement += 2; lower.push_back(edge); }
        }
        for (bool native : {false, true}) {
            std::vector<fs::PeriodicRecord> fixed;
            if (native) { fixed.push_back({1, 0, 1}); }
            const auto a = fs::analyzePeriodicDemands(word, upper, fixed);
            const auto b = fs::analyzePeriodicDemands(word, lower, fixed);
            for (uint32_t trips = 0; trips <= 5; ++trips) {
                if (!compare(a, b, trips)) {
                    llvm::errs() << "excess case " << mask << ":" << trips << "\n";
                    return false;
                }
                ++cases;
            }
        }
    }
    return true;
}
bool largeAndSupport()
{
    const std::vector<fs::PeriodicPayload> word{{0}, {1}};
    const auto upper = fs::analyzePeriodicDemands(word, {{0, 1, 0}});
    const auto lower = fs::analyzePeriodicDemands(word, {{0, 1, 7}});
    const auto native = fs::analyzePeriodicDemands(word, {});
    const uint64_t trips = UINT64_MAX;
    const auto linear = fs::analyzePeriodicExcess(upper, lower, trips);
    const auto quadratic = fs::analyzePeriodicExcess(upper, native, trips);
    if (!linear.error.empty() || !quadratic.error.empty() || linear.gamma != wide(7) ||
        linear.excess != pairs(0, trips) - pairs(7, trips) || quadratic.excess != pairs(0, trips) ||
        quadratic.quadraticPairs != 1 || quadratic.sameSupport || quadratic.excess.getActiveBits() <= 64 ||
        fs::periodicExcessDecimal(quadratic.excess) != "170141183460469231722463931679029329920") { return false; }
    const auto feedbackA = fs::analyzePeriodicDemands(word, {{0, 1, 0}}, {{1, 0, 2}});
    const auto feedbackB = fs::analyzePeriodicDemands(word, {{0, 1, 7}}, {{1, 0, 2}});
    const auto feedback = fs::analyzePeriodicExcess(feedbackA, feedbackB, trips);
    if (!feedback.error.empty() || !feedback.sameSupport || feedback.gamma != wide(21)) { return false; }
    const auto single = fs::analyzePeriodicDemands({{5}}, {});
    const auto local = fs::analyzePeriodicDemands({{5}}, {{0, 0, 3}});
    const auto noCycle = fs::analyzePeriodicExcess(single, single, 0);
    const auto cycle = fs::analyzePeriodicExcess(local, single, 3);
    return noCycle.error.empty() && noCycle.equalFrontiers && noCycle.quadraticPairs == 0 &&
        cycle.error.empty() && cycle.quadraticPairs == 1 && cycle.excess == 0 && !cycle.equalFrontiers;
}
bool malformed()
{
    const auto good = fs::analyzePeriodicDemands({{3}, {4}}, {{0, 1, 0}});
    auto rejected = [&good](const fs::PeriodicAnalysis& bad) {
        return !fs::analyzePeriodicExcess(good, bad, 1).error.empty();
    };
    auto bad = good; bad.sourceRows.pop_back(); if (!rejected(bad)) { return false; }
    bad = good; bad.sourceRows[0] = UINT32_MAX; if (!rejected(bad)) { return false; }
    bad = good; bad.localRanks[0] = 0; if (!rejected(bad)) { return false; }
    bad = good; bad.frontiers[0].count = 0; if (!rejected(bad)) { return false; }
    bad = good; bad.frontiers[0].distances.pop_back(); if (!rejected(bad)) { return false; }
    bad = good; bad.pipeRows[3] = 1; if (!rejected(bad)) { return false; }
    bad = good; bad.retained.push_back(UINT32_MAX); if (!rejected(bad)) { return false; }
    bad = good; bad.generators[0].target = UINT32_MAX; if (!rejected(bad)) { return false; }
    bad = good; bad.frontiers[1].distances[0] = 0; if (!rejected(bad)) { return false; }
    bad = good; bad.frontiers[0].distances[2].reset(); if (!rejected(bad)) { return false; }
    bad = good; bad.nativePrerequisites.push_back({1, 0, 1}); if (!rejected(bad)) { return false; }
    const auto later = fs::analyzePeriodicDemands({{3}, {4}}, {{0, 1, 4}});
    if (fs::analyzePeriodicExcess(later, good, 0).error.empty()) { return false; }
    const auto empty = fs::analyzePeriodicDemands({}, {});
    return fs::analyzePeriodicExcess(empty, empty, UINT64_MAX).error.empty() &&
        !fs::certifyPeriodicExcess({}, {}, 0).error.empty();
}
} // namespace
int runPeriodicExcessChecks()
{
    uint64_t cases = 0;
    if (!exhaustive(cases) || !largeAndSupport() || !malformed()) {
        llvm::errs() << "periodic excess checks failed\n";
        return 1;
    }
    llvm::outs() << "periodic excess checks passed: " << cases << " finite/profile cases\n";
    return 0;
}
