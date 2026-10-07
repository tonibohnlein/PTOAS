// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RepeatedLaneAllocation.h"
#include "RepeatedRegionInternal.h"
#include <algorithm>
#include <limits>
#include <map>

namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
constexpr uint64_t maximumPaletteWidth = 6;
constexpr int64_t unavailableCost = maximumPaletteWidth + 1;
using Costs = std::vector<std::vector<int64_t>>;
struct Lane {
    RegionalAllocationGroup group;
    RepeatedAllocationExtrema bounds;
    Id active = RegionExpressions::invalid;
    uint64_t delay = 0;
    bool child = true;
};
using Directions = std::map<std::pair<uint32_t, uint32_t>, std::vector<Lane>>;

// Hungarian assignment: O(n^3) integer operations and O(n^2) cost storage.
// Missing edges have cost 7; an accepted assignment costs at most 6 and hence
// contains no missing edge. This bounds the numeric search without claiming
// that a failed proof is an unreachable ordering relation.
std::optional<std::vector<std::size_t>> minimumAssignment(const Costs& costs)
{
    const auto n = costs.size();
    if (n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / 32)) { return std::nullopt; }
    const auto infinity = std::numeric_limits<int64_t>::max() / 4;
    std::vector<int64_t> row(n + 1), column(n + 1);
    std::vector<std::size_t> matched(n + 1), previous(n + 1);
    for (std::size_t i = 1; i <= n; ++i) {
        matched[0] = i;
        std::size_t current = 0;
        std::vector<int64_t> best(n + 1, infinity);
        std::vector<bool> seen(n + 1, false);
        do {
            seen[current] = true;
            const auto source = matched[current];
            int64_t delta = infinity;
            std::size_t next = 0;
            for (std::size_t j = 1; j <= n; ++j) {
                if (seen[j]) { continue; }
                auto reduced = costs[source - 1][j - 1] - row[source] - column[j];
                if (reduced < best[j]) { best[j] = reduced; previous[j] = current; }
                if (best[j] < delta) { delta = best[j]; next = j; }
            }
            if (!next) { return std::nullopt; }
            for (std::size_t j = 0; j <= n; ++j) {
                if (seen[j]) { row[matched[j]] += delta; column[j] -= delta; }
                else { best[j] -= delta; }
            }
            current = next;
        } while (matched[current]);
        do {
            const auto next = previous[current];
            matched[current] = matched[next];
            current = next;
        } while (current);
    }
    std::vector<std::size_t> result(n);
    uint64_t total = 0;
    for (std::size_t j = 1; j <= n; ++j) {
        auto cost = costs[matched[j] - 1][j - 1];
        if (cost < 0 || static_cast<uint64_t>(cost) > maximumPaletteWidth - total) { return std::nullopt; }
        total += static_cast<uint64_t>(cost);
        result[matched[j] - 1] = j - 1;
    }
    return result;
}

bool addLane(RepeatedRegionState& state, Directions& directions,
             RegionalAllocationGroup group, uint64_t delay, bool child)
{
    if (group.budget != 1 || group.members.empty()) { return false; }
    for (const auto& member : group.members) {
        if (child && (!member.tupleRule || !validPhysicalTupleRule(*member.tupleRule, 1) ||
                      !canRestrictRepeatedAllocationMember(state, member))) { return false; }
    }
    auto bounds = repeatedAllocationExtrema(state.body, group);
    if (!bounds || bounds->first.empty() || bounds->second.empty()) { return false; }
    auto active = state.e().boolean(false);
    for (const auto& selector : bounds->first) { active = state.e().lor(active, selector.present); }
    auto direction = std::make_pair(group.sourcePipe, group.targetPipe);
    directions[direction].push_back({std::move(group), std::move(*bounds), active, delay, child});
    return true;
}

bool addCrossings(RepeatedRegionState& state, Directions& directions,
                  ArrayRef<std::pair<uint32_t, std::size_t>> crossings)
{
    for (const auto& [record, index] : crossings) {
        if (index >= state.crossings.size()) { return false; }
        const auto& crossing = state.crossings[index];
        if (crossing.source.type >= state.body.anchors.size() ||
            crossing.target.type >= state.body.anchors.size()) { return false; }
        const auto& source = state.body.anchors[crossing.source.type];
        const auto& target = state.body.anchors[crossing.target.type];
        if (!source.phase || !target.phase) { return false; }
        auto p = static_cast<uint32_t>(source.phase->kPipeValue);
        auto q = static_cast<uint32_t>(target.phase->kPipeValue);
        if (p == q) { continue; }
        auto first = crossing.source, last = crossing.target;
        first.kind = PeriodicEventKind::Start;
        last.kind = PeriodicEventKind::Completion;
        RegionalAllocationGroup group{p, q, 1, {{record, 0, 0, first, last, crossing.guard}}};
        if (!addLane(state, directions, std::move(group), 1, false)) { return false; }
    }
    return true;
}

int64_t reuseCost(RepeatedRegionState& state, const Lane& from, const Lane& to)
{
    auto& e = state.e();
    // A cycle must be wholly present or wholly absent in each parameter case.
    // Pairwise conditional edges alone do not justify skipping a middle lane.
    if (!e.implies(from.active, to.active) || !e.implies(to.active, from.active)) { return unavailableCost; }
    // All spans here have one direction. Their source pipes agree, so a
    // consumer cannot precede a publication in an earlier source period.
    // Only nonnegative displacements need be considered for this contract.
    for (uint64_t gap = from.delay; gap <= maximumPaletteWidth; ++gap) {
        bool proven = true;
        for (const auto& end : from.bounds.second) {
            for (const auto& begin : to.bounds.first) {
                auto active = e.land(end.present, begin.present);
                if (e.constantValue(active) == 0) { continue; }
                // The WAIT endpoint of an outer handoff is already one period
                // later than its SET. Subtract that consumer delay exactly once.
                auto query = state.relativeQuery(end.event, begin.event, gap - from.delay);
                if (!query || !e.implies(active, *query)) { proven = false; break; }
            }
            if (!proven) { break; }
        }
        if (proven) { return static_cast<int64_t>(gap); }
    }
    return unavailableCost;
}

bool appendLane(RepeatedRegionState& state, RegionalAllocationGroup& result,
                const Lane& lane, uint64_t base, uint64_t width, uint64_t potential)
{
    for (auto member : lane.group.members) {
        auto arity = lane.child ? member.tupleRule->coordinateCount : 0;
        if (arity == UINT64_MAX) { return false; }
        PhysicalTupleRule rule{arity + 1, base,
            {{lane.child ? uint64_t(1) : uint64_t(0), 1, (width - potential % width) % width, width, 1}}};
        member.tupleRule = std::move(rule);
        member.stride = 0;
        member.phase = 0;
        if (!liftRepeatedAllocationEnvelope(state, member, lane.delay)) { return false; }
        result.members.push_back(std::move(member));
    }
    return true;
}

std::optional<RegionalAllocationGroup> allocateDirection(RepeatedRegionState& state, ArrayRef<Lane> lanes)
{
    const auto n = lanes.size();
    if (!n || n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / 32) ||
        n > std::vector<int64_t>().max_size() / n) { return std::nullopt; }
    Costs costs(n, std::vector<int64_t>(n, unavailableCost));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) { costs[i][j] = reuseCost(state, lanes[i], lanes[j]); }
    }
    auto next = minimumAssignment(costs);
    if (!next) { return std::nullopt; }
    RegionalAllocationGroup result{lanes.front().group.sourcePipe, lanes.front().group.targetPipe, 0, {}};
    std::vector<bool> seen(n, false);
    for (std::size_t start = 0; start < n; ++start) {
        if (seen[start]) { continue; }
        std::vector<std::size_t> cycle;
        uint64_t width = 0;
        auto current = start;
        do {
            if (seen[current]) { return std::nullopt; }
            seen[current] = true;
            cycle.push_back(current);
            width += static_cast<uint64_t>(costs[current][(*next)[current]]);
            current = (*next)[current];
        } while (current != start);
        // A nonempty notification chain cannot form a zero-distance cycle.
        // Reject vacuous/insufficient certificates instead of inventing an ID.
        if (!width || width > maximumPaletteWidth - result.budget) { return std::nullopt; }
        uint64_t potential = 0;
        for (auto index : cycle) {
            if (!appendLane(state, result, lanes[index], result.budget, width, potential)) { return std::nullopt; }
            potential += static_cast<uint64_t>(costs[index][(*next)[index]]);
        }
        result.budget += width;
    }
    return result;
}
} // namespace

std::shared_ptr<RegionalAllocationSummary> periodicLaneAllocation(
    RepeatedRegionState& state, const RegionalAllocationSummary& coalesced,
    ArrayRef<std::pair<uint32_t, std::size_t>> crossings)
{
    Directions directions;
    for (const auto& group : coalesced.groups) {
        for (auto lane : constantAllocationLanes(group)) {
            if (!addLane(state, directions, std::move(lane), 0, true)) { return {}; }
        }
    }
    if (!addCrossings(state, directions, crossings)) { return {}; }
    auto result = std::make_shared<RegionalAllocationSummary>();
    for (const auto& entry : directions) {
        auto allocated = allocateDirection(state, entry.second);
        if (!allocated) { return {}; }
        result->groups.push_back(std::move(*allocated));
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
