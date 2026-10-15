// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_RESTRICTED_INCOMING_CHECKS_H
#define PTO_RESTRICTED_INCOMING_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/WriterPairIndex.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
namespace {
bool writerPairIndexChecks()
{
    namespace fs = mlir::pto::frontiersynch;
    using Pair = std::pair<std::size_t, std::size_t>;
    for (unsigned left = 0; left < 256; ++left) {
        fs::WriterPairIndex a(8, [&](std::size_t i) { return !(left & (1U << i)); });
        std::vector<Pair> expected, actual;
        for (std::size_t i = 0; i < 8; ++i) {
            for (std::size_t j = i + 1; j < 8; ++j) {
                if (left & ((1U << i) | (1U << j))) {
                    expected.emplace_back(i, j);
                }
            }
        }
        a.pairs([&](auto i, auto j) {
            actual.emplace_back(i, j);
            return true;
        });
        if (actual != expected) {
            return false;
        }
        for (unsigned right = 0; right < 64; ++right) {
            fs::WriterPairIndex b(6, [&](std::size_t i) { return !(right & (1U << i)); });
            expected.clear();
            actual.clear();
            for (std::size_t i = 0; i < 8; ++i) {
                for (std::size_t j = 0; j < 6; ++j) {
                    if ((left & (1U << i)) || (right & (1U << j))) {
                        expected.emplace_back(i, j);
                    }
                }
            }
            a.cross(b, [&](auto i, auto j) {
                actual.emplace_back(i, j);
                return true;
            });
            if (actual != expected) {
                return false;
            }
        }
    }
    for (unsigned read = 0; read < 64; ++read) {
        for (unsigned needed = 0; needed < 64; ++needed) {
            fs::WriterPairIndex index(
                6, [&](std::size_t i) { return (read >> i) & 1; }, [&](std::size_t i) { return (needed >> i) & 1; });
            std::vector<Pair> actual, expected;
            index.pairs([&](auto a, auto b) {
                actual.emplace_back(a, b);
                return true;
            });
            for (std::size_t a = 0; a < 6; ++a) {
                for (std::size_t b = a + 1; b < 6; ++b) {
                    if (((needed >> a) & 1 || (needed >> b) & 1) && !((read >> a) & 1 && (read >> b) & 1)) {
                        expected.emplace_back(a, b);
                    }
                }
            }
            if (actual != expected) {
                return false;
            }
        }
    }
    for (std::size_t count : {0U, 1U, 1024U, 65536U}) {
        for (bool needed : {false, true}) {
            std::size_t visited = 0;
            fs::WriterPairIndex resolved(
                count, [](std::size_t) { return false; }, [&](std::size_t i) { return needed && i == count / 2; });
            resolved.pairs([&](auto, auto) {
                ++visited;
                return true;
            });
            if (visited != (needed && count ? count - 1 : 0)) {
                return false;
            }
        }
    }
    // Residual phase eligibility must not reverse the original conflict pair:
    // an initializer publishes protection but cannot consume it. Exercise both
    // orientations with either phase alone requiring an overlap query.
    for (unsigned needs = 1; needs < 4; ++needs) {
        for (unsigned reset = 0; reset < 2; ++reset) {
            fs::WriterPairIndex writers(2, [](std::size_t) { return false; });
            unsigned conflicts = 0;
            writers.pairs([&](std::size_t a, std::size_t b) {
                if (!(needs & ((1U << a) | (1U << b)))) {
                    return true;
                }
                const uint64_t first = 1 | (reset == 0 ? fs::protectionResetBit : 0);
                const uint64_t second = 1 | (reset == 1 ? fs::protectionResetBit : 0);
                conflicts += !fs::hardwareProtectsConflict(2, first, 2, second);
                return true;
            });
            if (conflicts != reset) {
                return false;
            }
        }
    }
    for (std::size_t count : {0U, 1U, 64U, 4096U, 65536U}) {
        for (std::size_t writers : {0U, 1U, 3U}) {
            writers = std::min(writers, count);
            std::size_t inspected = 0, visited = 0;
            fs::WriterPairIndex index(count, [&](std::size_t i) {
                ++inspected;
                return i >= writers;
            });
            index.pairs([&](auto, auto) {
                ++visited;
                return true;
            });
            const auto expected = writers * (count - writers) + (writers ? writers * (writers - 1) / 2 : 0);
            if (inspected != count || visited != expected || index.hasWriters() != bool(writers)) {
                return false;
            }
        }
    }
    return true;
}
bool restrictedIncomingScaling(mlir::MLIRContext& context)
{
    namespace fs = mlir::pto::frontiersynch;
    for (uint32_t count : {16U, 64U, 256U}) {
        mlir::Block inputs;
        fs::RegionExpressions e;
        std::vector<fs::GuardedRankPayload> payloads;
        std::vector<fs::GuardedRankEdge> edges;
        std::vector<fs::RegionExpressions::Id> predicates;
        for (uint32_t i = 0; i < count; ++i) {
            auto value = inputs.addArgument(mlir::IntegerType::get(&context, 1), mlir::UnknownLoc::get(&context));
            predicates.push_back(e.input(value));
            payloads.push_back({0, predicates.back()});
            edges.push_back({i, count, e.boolean(true)});
        }
        payloads.push_back({1, e.boolean(true)});
        auto reduced = fs::reduceGuardedRanks(e, payloads, edges, {}, fs::GuardedRankLabels::Reference);
        if (!reduced.error.empty() || e.size() > 32 * uint64_t(count) + 100) {
            return false;
        }
        for (uint32_t pattern = 0; pattern < 20; ++pattern) {
            fs::RegionExpressions::Transaction scratch(e);
            std::vector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> values;
            std::optional<uint32_t> last;
            for (uint32_t i = 0; i < count; ++i) {
                const bool active =
                    pattern == 1 || (pattern == 2 && i % 2) || (pattern > 2 && i == (17 * pattern) % count);
                values.emplace_back(predicates[i], e.boolean(active));
                if (active) {
                    last = i;
                }
            }
            fs::RegionExpressions::Substitution bindings(values);
            std::vector<uint32_t> selected;
            for (const auto& edge : reduced.retained) {
                if (e.constantValue(e.substitute(edge.guard, bindings)) == 1) {
                    selected.push_back(edge.source);
                }
            }
            if (selected != (last ? std::vector<uint32_t>{*last} : std::vector<uint32_t>{})) {
                return false;
            }
            for (uint32_t i = 0; i < count; ++i) {
                auto reaches =
                    reduced.query(e, {i, fs::PeriodicEventKind::Completion}, {count, fs::PeriodicEventKind::Start});
                if (!reaches) {
                    return false;
                }
                const bool present = e.constantValue(e.substitute(predicates[i], bindings)) == 1;
                if (e.constantValue(e.substitute(*reaches, bindings)) != unsigned(present && last && i <= *last)) {
                    return false;
                }
            }
        }
    }
    return true;
}
} // namespace
#endif
