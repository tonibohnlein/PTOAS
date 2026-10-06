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
    graph.assign(2*count, std::vector<Expr>(2*count, no()));
    std::vector<std::vector<Expr>> demands(count, std::vector<Expr>(count, no()));
    std::map<uint32_t, std::vector<std::pair<uint32_t, CellAccess>>> accesses;
    for (uint32_t i = 0; i < count; ++i) {
        graph[2*i][2*i] = graph[2*i+1][2*i+1] = graph[2*i][2*i+1] = presence[i];
        for (uint32_t j = i+1; j < count; ++j) {
            if (pipe(i) == pipe(j)) {
                graph[2*i][2*j] = graph[2*i+1][2*j+1] = both(presence[i],presence[j]);
            }
        }
        for (auto mode : effects[i].accesses) { accesses[mode.atom].push_back({i, mode}); }
    }
    for (const auto& [cell, uses] : accesses) {
        (void)cell;
        for (std::size_t a = 0; a < uses.size(); ++a) {
            auto [i, x] = uses[a];
            for (std::size_t b = a+1; b < uses.size(); ++b) {
                auto [j, y] = uses[b];
                ++cost.crossingCandidates;
                if (!(x.write || y.write) || hardwareProtectsConflict(pipe(i), x.protectionGroup,
                                                                     pipe(j), y.protectionGroup)) { continue; }
                demands[i][j] = either(demands[i][j], both(presence[i], presence[j]));
            }
        }
    }
    for (const auto& edge : residual) {
        demands[edge.source][edge.target] = either(demands[edge.source][edge.target],
            both(presence[edge.source], presence[edge.target]));
    }
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t j = i+1; j < count; ++j) { graph[2*i+1][2*j] = demands[i][j]; }
    }
    for (const auto& edge : nativePrerequisites) {
        auto native = both(presence[edge.source], presence[edge.target]);
        graph[2*edge.source+1][2*edge.target] = either(graph[2*edge.source+1][2*edge.target], native);
        demands[edge.source][edge.target] = both(demands[edge.source][edge.target], negate(native));
    }
    // Reflexive diagonals are presence-gated. Every nontrivial edge points
    // forward in this potential-event order; mutually exclusive arms never join.
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = k+1; j < graph.size(); ++j) {
                graph[i][j] = either(graph[i][j], both(graph[i][k], graph[k][j]));
            }
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t j = i+1; j < count; ++j) {
            if (arena->constantValue(demands[i][j]) == 0) { continue; }
            Expr alternative = no();
            for (std::size_t z = 2*i+2; z < 2*j; ++z) {
                alternative = either(alternative, both(graph[2*i+1][z], graph[z][2*j]));
            }
            auto guard = both(demands[i][j], negate(alternative));
            if (arena->constantValue(guard) != 0) { retained.push_back({i,j,guard}); }
        }
    }
}
void FiniteGuardedState::summarize(const SyncInput& input)
{
    auto selector = [&](uint32_t type, Expr present) {
        return RegionalSelector{{type, arena->constant(0), PeriodicEventKind::Start}, present};
    };
    std::map<uint32_t, std::vector<std::pair<uint32_t, CellAccess>>> cells;
    for (uint32_t i = 0; i < effects.size(); ++i) {
        for (auto mode : effects[i].accesses) { cells[mode.atom].push_back({i,mode}); }
        auto first = presence[i], last = presence[i];
        for (uint32_t j = 0; j < effects.size(); ++j) {
            if (pipe(i) != pipe(j)) { continue; }
            ++cost.selectorComparisons;
            if (j < i) { first = both(first, negate(presence[j])); }
            if (j > i) { last = both(last, negate(presence[j])); }
        }
        firstPayloads[pipe(i)].push_back(selector(i,first));
        lastPayloads[pipe(i)].push_back(selector(i,last));
    }
    for (const auto& [cell, uses] : cells) {
        RegionalStorageBoundary boundary;
        boundary.cell = input.accesses().cells()[cell];
        for (auto [i, mode] : uses) {
            auto first = presence[i], last = presence[i];
            if (mode.write) {
                for (auto [j, other] : uses) {
                    if (!other.write) { continue; }
                    ++cost.selectorComparisons;
                    if (j < i) { first = both(first, negate(presence[j])); }
                    if (j > i) { last = both(last, negate(presence[j])); }
                }
                boundary.firstWriters.push_back(selector(i,first));
                boundary.lastWriters.push_back(selector(i,last));
            } else if (mode.read) {
                for (auto [j, other] : uses) {
                    ++cost.selectorComparisons;
                    bool supersedes = other.write || (other.read && pipe(i) == pipe(j));
                    if (supersedes && j < i) { first = both(first, negate(presence[j])); }
                    if (supersedes && j > i) { last = both(last, negate(presence[j])); }
                }
                boundary.firstReaders[pipe(i)].push_back(selector(i,first));
                boundary.lastReaders[pipe(i)].push_back(selector(i,last));
            }
        }
        storageBoundary.push_back(std::move(boundary));
    }
}
} // namespace mlir::pto::frontiersynch
