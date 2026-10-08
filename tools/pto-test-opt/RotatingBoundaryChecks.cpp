// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
// Independent unfolded all-conflict graph, including cross-visit buffer reuse.
#include "PTO/Transforms/FrontierSynch/RotatingBoundary.h"
#include "llvm/Support/raw_ostream.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include <set>
#include <tuple>
namespace fs = mlir::pto::frontiersynch;
namespace {
using Pair = std::pair<unsigned, unsigned>;
bool check(unsigned slots, unsigned stride, unsigned slope, unsigned intercept, unsigned trips,
           bool scalar, bool relationships = false)
{
    std::vector<fs::PeriodicPayload> payloads{{0}, {1}, {2}};
    fs::StorageProtectionPolicy protection;
    if (scalar) {
        payloads[1].pipe = 0;
        protection.scalarPipe = 0;
    }
    std::vector<fs::RotatingFragment> effects{
        {0, 0, 0, slots, stride, 0, false, true, 0},
        {1, 0, 0, slots, stride, 0, true, false, 0},
        {2, 0, 0, slots, stride, slots - 1, true, false, 0}};
    std::vector<fs::RotatingBoundaryCell> cells;
    for (unsigned slot = 0; slot < slots; ++slot) {
        cells.push_back({0, 0, slot});
    }
    auto certificate = fs::buildRotatingBoundaryCertificate(payloads, effects, cells, {}, 256, protection);
    if (relationships) {
        auto extracted = fs::extractRotatingGenerators(payloads, effects, protection);
        extracted.generators.push_back({1, 2, 0});
        extracted.generators.push_back({2, 1, 1});
        auto quotient = fs::analyzePeriodicDemands(payloads, extracted.generators, {{0, 1, 0}});
        certificate = fs::buildRotatingBoundaryCertificate(
            std::move(quotient), effects, cells, {{1, 2}, {2, 1}}, 256, protection);
    }
    if (!certificate.error.empty()) {
        llvm::errs() << certificate.error << "\n";
        return false;
    }
    auto analysis = fs::analyzeAffineRotatingVisits(std::move(certificate), slope, intercept);
    if (!analysis.error.empty()) {
        llvm::errs() << analysis.error << "\n";
        return false;
    }
    struct Event {
        unsigned type, iteration, slot;
        bool write;
    };
    std::vector<Event> events;
    std::set<Pair> nativeRequirements;
    std::vector<unsigned> starts, lengths;
    for (unsigned t = 0; t < trips; ++t) {
        starts.push_back(events.size());
        lengths.push_back(slope * t + intercept);
        for (unsigned i = 0; i < lengths.back(); ++i) {
            for (const auto& effect : effects) {
                events.push_back({effect.payload, i, unsigned((stride * i + effect.offset) % slots), effect.write});
            }
        }
    }
    const unsigned n = events.size();
    std::vector<std::vector<bool>> graph(2 * n, std::vector<bool>(2 * n));
    for (unsigned i = 0; i < n; ++i) {
        graph[2 * i][2 * i + 1] = true;
        for (unsigned j = i + 1; j < n; ++j) {
            const auto p = payloads[events[i].type].pipe, q = payloads[events[j].type].pipe;
            if (p == q) {
                graph[2 * i][2 * j] = graph[2 * i + 1][2 * j + 1] = true;
            }
            if (relationships && events[i].type == 0 && events[j].type == 1 && j == i + 1) {
                graph[2 * i + 1][2 * j] = true;
                nativeRequirements.emplace(i, j);
            }
            if (relationships && !protection.protectsScalar(p, q) &&
                ((events[i].type == 1 && events[j].type == 2) ||
                 (events[i].type == 2 && events[j].type == 1))) {
                graph[2 * i + 1][2 * j] = true;
            }
            if (!protection.protectsScalar(p, q) && events[i].slot == events[j].slot &&
                (events[i].write || events[j].write)) {
                graph[2 * i + 1][2 * j] = true;
            }
        }
    }
    for (unsigned k = 0; k < 2 * n; ++k) {
        for (unsigned i = 0; i < k; ++i) {
            if (!graph[i][k]) {
                continue;
            }
            for (unsigned j = k + 1; j < 2 * n; ++j) {
                graph[i][j] = graph[i][j] || graph[k][j];
            }
        }
    }
    std::set<Pair> expected, actual;
    for (unsigned a = 0; a < n; ++a) {
        for (unsigned b = a + 1; b < n; ++b) {
            if (!graph[2 * a + 1][2 * b]) {
                continue;
            }
            bool redundant = false;
            for (unsigned z = 2 * a + 2; z < 2 * b; ++z) {
                redundant |= graph[2 * a + 1][z] && graph[z][2 * b];
            }
            if (!redundant && !nativeRequirements.count({a, b})) {
                expected.emplace(a, b);
            }
        }
    }
    for (unsigned t = 0; t < trips; ++t) {
        for (auto id : analysis.child.quotient.retained) {
            const auto& e = analysis.child.quotient.generators[id];
            for (unsigned i = 0; i < lengths[t]; ++i) {
                if (e.displacement < lengths[t] - i) {
                    actual.emplace(starts[t] + 3 * i + e.source, starts[t] + 3 * (i + e.displacement) + e.target);
                }
            }
        }
        for (const auto& edge : analysis.crossingInto(t)) {
            actual.emplace(
                starts[t - 1] + 3 * edge.source.at(lengths[t - 1]) + edge.source.type,
                starts[t] + 3 * edge.target.at(lengths[t]) + edge.target.type);
        }
    }
    if (expected != actual) {
        llvm::errs() << "rotating boundary mismatch " << slots << "," << stride << "," << slope << "," << intercept
                     << "," << trips << "\n";
        return false;
    }
    return true;
}
bool checkRelationshipPorts()
{
    // Sites 1 and 2 are neither pipe extrema nor storage extrema.
    std::vector<fs::PeriodicPayload> payloads{{0}, {0}, {0}, {0}};
    auto quotient = fs::analyzePeriodicDemands(payloads, {{1, 2, 0}, {2, 1, 1}});
    auto certificate = fs::buildRotatingBoundaryCertificate(
        std::move(quotient), {}, {}, {{1, 2}, {2, 1}});
    if (!certificate.error.empty()) { return false; }
    auto left = certificate.select(3), right = certificate.select(5);
    auto crossings = certificate.crossings(left, right);
    return crossings && crossings->size() == 1 &&
        crossings->front().source.type == 2 && crossings->front().target.type == 1;
}
bool checkProtectionScope()
{
    std::vector<fs::PeriodicPayload> payloads{{0}, {0}};
    auto quotient = fs::analyzePeriodicDemands(payloads, {});
    const std::vector<fs::RotatingFragment> inconsistent{
        {0, 0, 0, 2, 1, 0, false, true, 0}, {0, 0, 0, 2, 0, 0, false, true, 0}};
    if (fs::buildRotatingBoundaryCertificate(quotient, inconsistent, {{0, 0, 0}, {0, 0, 1}}, {}).error.empty()) {
        return false;
    }
    for (unsigned mode = 0; mode < 4; ++mode) {
        const uint64_t group = 1 | (mode == 0 ? 0 : fs::invocationProtectionBit);
        std::vector<fs::RotatingFragment> effects{
            {0, 0, 0, 1, 0, 0, false, true, group},
            {1, 0, 0, 1, 0, 0, false, true, group}};
        std::vector<fs::RotatingBoundaryCell> cells{{0, 0, 0}};
        if (mode == 2) { effects[0].protectionGroup |= fs::protectionResetBit; }
        if (mode == 3) {
            // Same endpoints also conflict on an unprotected physical atom.
            effects.push_back({0, 0, 1, 1, 0, 0, false, true, 0});
            effects.push_back({1, 0, 1, 1, 0, 0, false, true, 0});
            cells.push_back({0, 1, 0});
        }
        auto certificate = fs::buildRotatingBoundaryCertificate(quotient, effects, cells, {});
        if (!certificate.error.empty()) { return false; }
        auto crossings = certificate.crossings(certificate.select(2), certificate.select(3));
        if (!crossings || crossings->size() != (mode == 1 ? 0U : 1U)) { return false; }
    }
    return true;
}
} // namespace
int runRotatingBoundaryChecks()
{
    if (!checkRelationshipPorts() || !checkProtectionScope()) {
        llvm::errs() << "boundary relationship or protection adapter failed\n";
        return 1;
    }
    unsigned checked = 0;
    for (unsigned slots = 1; slots <= 3; ++slots) {
        for (unsigned stride = 0; stride <= slots; ++stride) {
            for (unsigned slope = 1; slope <= 2; ++slope) {
                for (unsigned intercept = 0; intercept <= 3; ++intercept) {
                    for (unsigned trips : {0U, 1U, 2U, 5U, 9U}) {
                        for (bool scalar : {false, true}) {
                            if (!check(slots, stride, slope, intercept, trips, scalar)) {
                                return 1;
                            }
                            if (!check(slots, stride, slope, intercept, trips, scalar, true)) {
                                return 1;
                            }
                            checked += 2;
                        }
                    }
                }
            }
        }
    }
    llvm::outs() << "rotating boundary certificates: " << checked << " unfolded comparisons passed\n";
    return 0;
}
