// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact Hungarian assignment with absent edges, then checked cycle potentials.
#include "PTO/Transforms/FrontierSynch/PeriodicSharedAllocation.h"
#include "llvm/ADT/DynamicAPInt.h"
#include <utility>
#include <map>

namespace mlir::pto::frontiersynch {
namespace {
using Integer = llvm::DynamicAPInt;
using Status = PeriodicSharedAllocationStatus;
PeriodicSharedAllocation fail(Status status, const char* message)
{
    PeriodicSharedAllocation result;
    result.status = status;
    result.error = message;
    return result;
}
Integer integer(uint64_t value)
{
    // DynamicAPInt's small constructor is signed; widen before doubling.
    return Integer(static_cast<int64_t>(value >> 1)) * Integer(2) + Integer(static_cast<int64_t>(value & 1));
}
bool zeroCycle(const PeriodicReuseMatrix& weights)
{
    const auto n = weights.size();
    std::vector<std::size_t> degree(n, 0), ready;
    for (const auto& row : weights) {
        for (std::size_t j = 0; j < n; ++j) {
            if (row[j] && *row[j] == 0) { ++degree[j]; }
        }
    }
    for (std::size_t j = 0; j < n; ++j) {
        if (!degree[j]) { ready.push_back(j); }
    }
    for (std::size_t cursor = 0; cursor < ready.size(); ++cursor) {
        for (std::size_t j = 0; j < n; ++j) {
            if (weights[ready[cursor]][j] && *weights[ready[cursor]][j] == 0) {
                --degree[j];
                if (!degree[j]) { ready.push_back(j); }
            }
        }
    }
    return ready.size() != n;
}
class Assignment {
public:
    explicit Assignment(const PeriodicReuseMatrix& weights)
        : weights(weights), n(weights.size()), row(n + 1), column(n + 1), matched(n + 1), previous(n + 1) {}
    std::optional<std::vector<uint32_t>> run()
    {
        for (std::size_t source = 1; source <= n; ++source) {
            if (!augment(source)) { return std::nullopt; }
        }
        std::vector<uint32_t> next(n);
        for (std::size_t target = 1; target <= n; ++target) {
            next[matched[target] - 1] = static_cast<uint32_t>(target - 1);
        }
        return next;
    }
private:
    const PeriodicReuseMatrix& weights;
    std::size_t n;
    std::vector<Integer> row, column;
    std::vector<std::size_t> matched, previous;
    std::optional<std::pair<std::size_t, Integer>> relax(
        std::size_t current, const std::vector<bool>& seen, std::vector<std::optional<Integer>>& best)
    {
        std::optional<std::pair<std::size_t, Integer>> choice;
        const auto source = matched[current];
        for (std::size_t j = 1; j <= n; ++j) {
            if (seen[j]) { continue; }
            if (auto weight = weights[source - 1][j - 1]) {
                auto reduced = integer(*weight) - row[source] - column[j];
                if (!best[j] || reduced < *best[j]) { best[j] = reduced; previous[j] = current; }
            }
            // A previous alternating-tree row may reach this column even
            // when the current row has no finite edge to it.
            if (best[j] && (!choice || *best[j] < choice->second)) { choice = std::make_pair(j, *best[j]); }
        }
        return choice;
    }
    void shift(const Integer& delta, const std::vector<bool>& seen, std::vector<std::optional<Integer>>& best)
    {
        for (std::size_t j = 0; j <= n; ++j) {
            if (seen[j]) { row[matched[j]] += delta; column[j] -= delta; }
            else if (best[j]) { *best[j] -= delta; }
        }
    }
    bool augment(std::size_t source)
    {
        matched[0] = source;
        std::size_t current = 0;
        std::vector<std::optional<Integer>> best(n + 1);
        std::vector<bool> seen(n + 1, false);
        do {
            seen[current] = true;
            auto choice = relax(current, seen, best);
            if (!choice) { return false; }
            shift(choice->second, seen, best);
            current = choice->first;
        } while (matched[current]);
        do {
            const auto parent = previous[current];
            matched[current] = matched[parent];
            current = parent;
        } while (current);
        return true;
    }
};
bool appendCycle(const PeriodicReuseMatrix& weights, const std::vector<uint32_t>& next,
                 uint32_t start, std::vector<bool>& seen, PeriodicSharedAllocation& result)
{
    std::vector<uint32_t> cycle;
    uint64_t width = 0;
    auto current = start;
    do {
        seen[current] = true;
        cycle.push_back(current);
        const auto weight = *weights[current][next[current]];
        if (weight > UINT64_MAX - width) { return false; }
        width += weight;
        current = next[current];
    } while (current != start);
    if (!width || width > UINT64_MAX - result.budget) { return false; }
    const auto id = static_cast<uint32_t>(result.cycles.size());
    result.cycles.push_back({start, result.budget, width});
    uint64_t prefix = 0;
    for (auto phase : cycle) {
        const auto offset = prefix ? (width - prefix) % width : 0;
        result.phases[phase] = {next[phase], id, result.budget, width, offset};
        prefix += *weights[phase][next[phase]]; // Bounded by the checked cycle sum.
    }
    result.budget += width;
    return true;
}
} // namespace
PeriodicSharedAllocation allocatePeriodicShared(const PeriodicReuseMatrix& weights)
{
    const auto n = weights.size();
    if (n >= UINT32_MAX || n >= std::vector<Integer>().max_size() ||
        n >= std::vector<std::optional<Integer>>().max_size()) {
        return fail(Status::InvalidInput, "periodic phase count exceeds assignment representation");
    }
    for (const auto& row : weights) {
        if (row.size() != n) { return fail(Status::InvalidInput, "periodic reuse matrix must be square"); }
    }
    if (zeroCycle(weights)) {
        return fail(Status::ZeroWeightCycle, "periodic reuse graph contains a zero-weight cycle");
    }
    auto next = Assignment(weights).run();
    if (!next) { return fail(Status::NoFiniteCover, "periodic reuse graph has no finite cycle cover"); }
    PeriodicSharedAllocation result;
    result.phases.resize(n);
    std::vector<bool> seen(n, false);
    for (uint32_t start = 0; start < n; ++start) {
        if (!seen[start] && !appendCycle(weights, *next, start, seen, result)) {
            return fail(Status::Overflow, "minimum periodic cycle-cover budget exceeds uint64 representation");
        }
    }
    return result;
}
PeriodicSharedAllocation allocateDirectedPeriodic(const PeriodicReuseMatrix& weights,
    const std::vector<std::pair<uint32_t, uint32_t>>& directions)
{
    if (weights.size() != directions.size() || weights.size() >= UINT32_MAX) {
        return fail(Status::InvalidInput, "directed periodic phase dimensions disagree");
    }
    for (const auto& row : weights) {
        if (row.size() != weights.size()) { return fail(Status::InvalidInput, "directed reuse matrix must be square"); }
    }
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> groups;
    for (uint32_t i = 0; i < directions.size(); ++i) { groups[directions[i]].push_back(i); }
    PeriodicSharedAllocation result;
    result.phases.resize(weights.size());
    for (const auto& group : groups) {
        const auto& indices = group.second;
        PeriodicReuseMatrix part(indices.size(), std::vector<std::optional<uint64_t>>(indices.size()));
        for (std::size_t i = 0; i < indices.size(); ++i) {
            for (std::size_t j = 0; j < indices.size(); ++j) { part[i][j] = weights[indices[i]][indices[j]]; }
        }
        auto assigned = allocatePeriodicShared(part);
        if (assigned.status != Status::Success) { return assigned; }
        const auto firstCycle = result.cycles.size();
        for (auto cycle : assigned.cycles) {
            cycle.firstPhase = indices[cycle.firstPhase]; result.cycles.push_back(cycle);
        }
        for (std::size_t i = 0; i < indices.size(); ++i) {
            auto phase = assigned.phases[i];
            phase.successor = indices[phase.successor]; phase.cycle += static_cast<uint32_t>(firstCycle);
            result.phases[indices[i]] = phase;
        }
        result.budget = std::max(result.budget, assigned.budget);
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
