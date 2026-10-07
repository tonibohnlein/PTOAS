// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "FiniteGuardedInternal.h"
namespace mlir::pto::frontiersynch {
void FiniteGuardedState::closeAndReduce()
{
    const auto count = effects.size();
    std::map<std::pair<uint32_t, uint32_t>, Expr> demands;
    auto add = [&](uint32_t source, uint32_t target, Expr guard) {
        auto [it, inserted] = demands.emplace(std::make_pair(source, target), guard);
        if (!inserted) {
            it->second = either(it->second, guard);
        }
    };
    std::map<uint32_t, std::vector<std::pair<uint32_t, CellAccess>>> accesses;
    for (uint32_t i = 0; i < count; ++i) {
        for (auto mode : effects[i].accesses) {
            accesses[mode.atom].push_back({i, mode});
        }
    }
    for (const auto& [cell, uses] : accesses) {
        // Protected writer pairs and simultaneous macro phases change the
        // lifetime-chain proof. Retain their exact raw pair generators; rank
        // reduction still applies, with the same O(W) candidate bound.
        bool plain = true;
        for (auto [i, x] : uses) {
            for (auto [j, y] : uses) {
                if (i == j) {
                    continue;
                }
                plain &= anchors[i].phase->elementOp != anchors[j].phase->elementOp &&
                         !ptoStorageProtection().protectsScalar(pipe(i), pipe(j)) &&
                         !hardwareProtectsConflict(pipe(i), x.protectionGroup, pipe(j), y.protectionGroup);
            }
        }
        for (std::size_t a = 0; a < uses.size(); ++a) {
            auto [i, x] = uses[a];
            auto noWriter = yes();
            for (std::size_t b = a + 1; b < uses.size(); ++b) {
                auto [j, y] = uses[b];
                ++cost.crossingCandidates;
                if (anchors[i].phase->elementOp != anchors[j].phase->elementOp && (x.write || y.write) &&
                    !ptoStorageProtection().protectsScalar(pipe(i), pipe(j)) &&
                    !hardwareProtectsConflict(pipe(i), x.protectionGroup, pipe(j), y.protectionGroup)) {
                    add(i, j, both(both(presence[i], presence[j]), noWriter));
                }
                if (plain && y.write) {
                    noWriter = both(noWriter, negate(presence[j]));
                }
            }
        }
    }
    for (auto edge : residual) {
        add(edge.source, edge.target, both(presence[edge.source], presence[edge.target]));
    }
    std::vector<GuardedRankPayload> payloads;
    std::vector<GuardedRankEdge> generators, native;
    for (uint32_t i = 0; i < count; ++i) {
        payloads.push_back({pipe(i), presence[i]});
    }
    for (auto [pair, guard] : demands) {
        generators.push_back({pair.first, pair.second, guard});
    }
    for (auto edge : nativePrerequisites) {
        native.push_back({edge.source, edge.target, yes()});
    }
    rankIndex = reduceGuardedRanks(*arena, payloads, generators, native);
    retained.clear();
    for (auto edge : rankIndex.retained) {
        retained.push_back({edge.source, edge.target, edge.guard});
    }
}

void FiniteGuardedState::summarize(const SyncInput& input)
{
    auto selector = [&](uint32_t type, Expr present) {
        return RegionalSelector{{type, arena->constant(0), PeriodicEventKind::Start}, present};
    };
    std::map<uint32_t, std::vector<std::pair<uint32_t, CellAccess>>> cells;
    std::map<uint32_t, Expr> preceding, following;
    for (uint32_t i = 0; i < effects.size(); ++i) {
        for (auto mode : effects[i].accesses) {
            cells[mode.atom].push_back({i, mode});
        }
        auto [seen, added] = preceding.emplace(pipe(i), no());
        firstPayloads[pipe(i)].push_back(selector(i, both(presence[i], negate(seen->second))));
        seen->second = either(seen->second, presence[i]);
        ++cost.selectorComparisons;
    }
    for (std::size_t j = effects.size(); j; --j) {
        const auto i = static_cast<uint32_t>(j - 1);
        auto [seen, added] = following.emplace(pipe(i), no());
        lastPayloads[pipe(i)].push_back(selector(i, both(presence[i], negate(seen->second))));
        seen->second = either(seen->second, presence[i]);
        ++cost.selectorComparisons;
    }
    for (const auto& [cell, uses] : cells) {
        RegionalStorageBoundary boundary;
        boundary.cell = input.accesses().cells()[cell];
        for (auto [i, mode] : uses) {
            auto first = presence[i], last = presence[i];
            if (mode.write) {
                for (auto [j, other] : uses) {
                    if (!other.write || anchors[i].phase->elementOp == anchors[j].phase->elementOp) {
                        continue;
                    }
                    ++cost.selectorComparisons;
                    if (j < i) {
                        first = both(first, negate(presence[j]));
                    }
                    if (j > i) {
                        last = both(last, negate(presence[j]));
                    }
                }
                boundary.firstWriters.push_back(selector(i, first));
                boundary.lastWriters.push_back(selector(i, last));
            } else if (mode.read) {
                for (auto [j, other] : uses) {
                    ++cost.selectorComparisons;
                    bool supersedes = anchors[i].phase->elementOp != anchors[j].phase->elementOp &&
                                      (other.write || (other.read && pipe(i) == pipe(j)));
                    if (supersedes && j < i) {
                        first = both(first, negate(presence[j]));
                    }
                    if (supersedes && j > i) {
                        last = both(last, negate(presence[j]));
                    }
                }
                boundary.firstReaders[pipe(i)].push_back(selector(i, first));
                boundary.lastReaders[pipe(i)].push_back(selector(i, last));
            }
        }
        storageBoundary.push_back(std::move(boundary));
    }
}
} // namespace mlir::pto::frontiersynch
