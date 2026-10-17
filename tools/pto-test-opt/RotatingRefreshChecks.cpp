// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetime.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/RotatingNormalization.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <set>
namespace fs = mlir::pto::frontiersynch;
bool checkCollectiveRotatingRefresh()
{
    const std::vector<fs::PeriodicPayload> payloads{{0}, {1}, {2}};
    uint64_t cases = 0;
    for (uint64_t slots = 1; slots <= 10; ++slots) {
        for (uint64_t stride = 0; stride <= slots; ++stride) {
            for (uint64_t first = 0; first < slots; ++first) {
                for (uint64_t second = 0; second < slots; ++second) {
                    for (unsigned mask = 0; mask < 4; ++mask) {
                        std::vector<fs::RotatingFragment> fragments{
                            {0, 0, 0, slots, stride, first, false, true, 0},
                            {1, 0, 0, slots, stride, second, true, true, 0},
                            {2, 0, 1, slots, stride, 0, true, false, 0}};
                        std::vector<uint8_t> guaranteed{uint8_t(mask & 1), uint8_t(mask & 2), 0};
                        auto result = fs::certifyRotatingRefreshCoverage(payloads, fragments, guaranteed);
                        // Independent finite orbit enumeration: every possible
                        // writable slot, every start, then first guaranteed write.
                        std::set<uint64_t> written;
                        for (uint64_t n = 0; n < slots; ++n) {
                            written.insert((stride * n + first) % slots);
                            written.insert((stride * n + second) % slots);
                        }
                        uint64_t expected = 0;
                        bool complete = true;
                        for (auto slot : written) {
                            for (uint64_t start = 0; start < slots; ++start) {
                                uint64_t delay = 1;
                                for (; delay <= slots; ++delay) {
                                    const auto n = start + delay;
                                    if (((mask & 1) && (stride * n + first) % slots == slot) ||
                                        ((mask & 2) && (stride * n + second) % slots == slot)) { break; }
                                }
                                complete &= delay <= slots;
                                expected = std::max(expected, delay);
                            }
                        }
                        if (result.error.empty() != complete || (complete && result.span != expected)) {
                            llvm::errs() << "collective refresh differs from physical orbit oracle\n";
                            return false;
                        }
                        if (complete) {
                            auto prerequisite = fs::certifyRotatingRefreshCoverage(
                                payloads, fragments, guaranteed, slots + 3);
                            if (!prerequisite.error.empty() || prerequisite.span != slots + 3) { return false; }
                        }
                        ++cases;
                    }
                }
            }
        }
    }
    std::vector<fs::RotatingFragment> pair{
        {0, 0, 0, 4, 1, 0, false, true, 0}, {1, 0, 0, 4, 1, 2, false, true, 0}};
    auto collective = fs::certifyRotatingRefreshCoverage(payloads, pair, {1, 1});
    auto individual = fs::certifyRotatingRefreshCoverage(payloads, pair, {1, 0});
    if (!collective.error.empty() || collective.span != 2 || individual.span != 4) { return false; }
    // Compare source-anchored covers from the smaller certified window and the
    // older sufficient window. Presence includes clipped short invocations.
    for (unsigned trips = 0; trips <= 7; ++trips) {
        auto covers = [&](unsigned span) {
            fs::RegionExpressions arena;
            fs::LifetimeWindowInput input;
            input.sites = 2; input.span = span;
            for (unsigned n = 0; n <= span; ++n) {
                for (unsigned site = 0; site < 2; ++site) {
                    const auto present = arena.boolean(n < trips);
                    const auto id = static_cast<uint32_t>(input.payloads.size());
                    input.payloads.push_back({site, present});
                    input.accesses.push_back({id, (n + 2 * site) % 4, arena.boolean(false), present});
                }
            }
            auto result = fs::analyzeLifetimeWindow(arena, input);
            std::set<std::pair<uint32_t, uint32_t>> edges;
            if (!result.error.empty()) { edges.emplace(UINT32_MAX, UINT32_MAX); }
            for (const auto& edge : result.sourceDemands) {
                if (arena.constantValue(edge.guard) == 1) { edges.emplace(edge.source, edge.target); }
            }
            return edges;
        };
        if (covers(2) != covers(4)) { return false; }
    }
    // Normalization scales with supplied fragments, independent of slot count.
    for (unsigned count : {64, 256, 1024, 4096}) {
        std::vector<fs::RotatingFragment> fragments;
        for (unsigned i = 0; i < count; ++i) {
            fragments.push_back({i % 3, 0, 0, uint64_t(1) << 48, 1, i, false, true, 0});
        }
        auto normalized = fs::normalizeRotatingFragments(payloads, fragments);
        if (!normalized.error.empty() || normalized.inspectedFragments != count ||
            normalized.normalizedFamilies != 1 || normalized.accesses.size() != count) { return false; }
    }
    std::vector<fs::RotatingFragment> interleaved{
        {0, 0, 0, 4, 1, 0, false, true, 0}, {1, 1, 1, 6, 2, 1, false, true, 0},
        {2, 0, 0, 4, 1, 2, false, true, 0}, {1, 1, 1, 6, 2, 3, false, true, 0}};
    auto mixed = fs::certifyRotatingRefreshCoverage(payloads, interleaved, {1, 1, 1, 0});
    if (!mixed.error.empty() || mixed.span != 3) { return false; }
    auto duplicate = pair;
    duplicate.push_back(pair.front()); duplicate.back().protectionGroup = 1;
    if (fs::normalizeRotatingFragments(payloads, duplicate).error.empty()) { return false; }
    auto invalid = pair;
    invalid[0].slots = 0;
    if (fs::normalizeRotatingFragments(payloads, invalid).error.empty()) { return false; }
    invalid = pair; invalid[1].stride = 2;
    if (fs::normalizeRotatingFragments(payloads, invalid).error.empty()) { return false; }
    invalid = pair; invalid[0].protectionGroup = 1;
    if (fs::certifyRotatingRefreshCoverage(payloads, invalid, {1, 1}).error.empty()) { return false; }
    llvm::outs() << "collective rotating refresh: " << cases << " physical-orbit cases passed\n";
    return true;
}
