// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RepeatedLaneAllocation.h"
#include "RepeatedRegionInternal.h"
#include "PTO/Transforms/FrontierSynch/RegionalLaneExports.h"
#include "PTO/Transforms/FrontierSynch/PeriodicSharedAllocation.h"
#include <map>

namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
constexpr uint64_t maximumPaletteWidth = 6;
struct Lane {
    RegionalAllocationGroup group;
    RepeatedAllocationExtrema bounds;
    uint64_t delay = 0;
    bool child = true;
};
bool addLane(RepeatedRegionState& state, std::vector<Lane>& lanes,
             RegionalAllocationGroup group, uint64_t delay, bool child)
{
    if (group.budget != 1 || group.members.empty()) { return false; }
    for (const auto& member : group.members) {
        if (child && (!member.tupleRule || !validPhysicalTupleRule(*member.tupleRule, 1) ||
                      !canRestrictRepeatedAllocationMember(state, member))) { return false; }
    }
    // Intersect each member's invocation with the retained phase interval
    // before constructing the reuse graph. A clipped-away handoff still needs
    // a record mapping, but must not become a phantom vertex in a cycle.
    for (auto& member : group.members) {
        auto clipped = member;
        if (!liftRepeatedAllocationEnvelope(state, clipped, delay)) { return false; }
        member.active = clipped.active;
    }
    auto bounds = repeatedAllocationExtrema(state.body, group);
    if (!bounds || bounds->first.empty() != bounds->second.empty()) { return false; }
    lanes.push_back({std::move(group), std::move(*bounds), delay, child});
    return true;
}

bool addCrossings(RepeatedRegionState& state, std::vector<Lane>& lanes,
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
        if (!addLane(state, lanes, std::move(group), 1, false)) { return false; }
    }
    return true;
}

std::optional<uint64_t> reuseCost(RepeatedRegionState& state, const Lane& from, const Lane& to)
{
    auto& e = state.e();
    // Costs propose an assignment only. A separate all-pair check below
    // certifies reuse when intervening handoffs have different activity.
    // Restrict to certified nonnegative source displacements. This is a
    // sufficient matrix, not the set of every possible hardware assignment.
    for (uint64_t gap = from.delay; gap <= maximumPaletteWidth; ++gap) {
        bool proven = true;
        for (const auto& end : from.bounds.second) {
            for (const auto& begin : to.bounds.first) {
                auto active = e.land(end.present, begin.present);
                if (e.implies(active, e.boolean(false))) { continue; }
                auto consumer = end.event, producer = begin.event;
                const auto& anchors = state.body.anchors;
                if (consumer.type >= anchors.size() || producer.type >= anchors.size() ||
                    !anchors[consumer.type].phase || !anchors[producer.type].phase) { return std::nullopt; }
                // Match regionalHandoffReuse: a WAIT precedes its payload's
                // start, while a SET follows its payload's issue on that pipe.
                // Native start order therefore suffices on one pipe, including
                // the same occurrence; cross-pipe reuse requires completion.
                producer.kind = PeriodicEventKind::Start;
                consumer.kind = anchors[consumer.type].phase->kPipeValue ==
                    anchors[producer.type].phase->kPipeValue ?
                    PeriodicEventKind::Start : PeriodicEventKind::Completion;
                // The WAIT endpoint of an outer handoff is already one period
                // later than its SET. Subtract that consumer delay exactly once.
                auto query = state.relativeQuery(consumer, producer, gap - from.delay);
                if (!query || !(e.implies(active, *query) || e.constantUnder(active, active) == 0 ||
                                e.constantUnder(active, *query) == 1)) { proven = false; break; }
            }
            if (!proven) { break; }
        }
        if (proven) { return gap; }
    }
    return std::nullopt;
}

// Export each physical label's actual first/last member occurrences after
// clipping. A member's source visits form one interval; its consumer is delayed
// by a fixed number of repeats. Only the at-most-six physical labels expand.
void appendBounds(RepeatedRegionState& state, RegionalAllocationGroup& result,
                  const RegionalAllocationMember& member, uint64_t delay,
                  uint64_t base, uint64_t width, uint64_t offset)
{
    if (width == 1) {
        result.lanes[base].firstSources.push_back({member.firstSource, member.active});
        result.lanes[base].lastTargets.push_back({member.lastTarget, member.active});
        return;
    }
    auto& e = state.e();
    const auto first = member.firstSource.visits.front();
    const auto last = e.sub(member.lastTarget.visits.front(), e.constant(delay));
    for (uint64_t lane = 0; lane < width; ++lane) {
        const auto residue = (lane + width - offset) % width;
        const auto shift = e.rem(e.add(e.constant(residue),
            e.sub(e.constant(width), e.rem(first, e.constant(width)))), e.constant(width));
        // Compare the shift before adding it, so an absent first+shift cannot
        // wrap into the live interval near the machine integer limit.
        const auto room = e.sub(last, first);
        const auto active = e.land(member.active, e.le(shift, room));
        const auto begin = e.select(active, e.add(first, shift), first);
        const auto end = e.sub(last, e.rem(e.sub(last, begin), e.constant(width)));
        auto source = member.firstSource, target = member.lastTarget;
        source.visits.front() = begin;
        target.visits.front() = e.add(end, e.constant(delay));
        result.lanes[base + lane].firstSources.push_back({std::move(source), active});
        result.lanes[base + lane].lastTargets.push_back({std::move(target), active});
    }
}

bool appendLane(RepeatedRegionState& state, RegionalAllocationGroup& result,
                const Lane& lane, uint64_t base, uint64_t width, uint64_t offset)
{
    for (auto member : lane.group.members) {
        auto direction = regionalAllocationDirection(lane.group, member);
        member.sourcePipe = direction.first;
        member.targetPipe = direction.second;
        auto arity = lane.child ? member.tupleRule->coordinateCount : 0;
        if (arity == UINT64_MAX) { return false; }
        PhysicalTupleRule rule{arity + 1, base,
            {{lane.child ? uint64_t(1) : uint64_t(0), 1, offset, width, 1}}};
        member.tupleRule = std::move(rule);
        member.stride = 0;
        member.phase = 0;
        if (!liftRepeatedAllocationEnvelope(state, member, lane.delay)) { return false; }
        appendBounds(state, result, member, lane.delay, base, width, offset);
        result.members.push_back(std::move(member));
    }
    return true;
}

// The formula (visit+offset)%width collides first at this positive gap.
// Larger collisions are safe through native start order at the destination
// source site. Its activity is immutable across repeats, and a witness between
// two retained occurrences stays inside the contiguous clipped interval.
bool compatible(const PeriodicReuseMatrix& costs, std::size_t from, std::size_t to,
                uint64_t fromOffset, uint64_t toOffset, uint64_t width)
{
    if (!costs[from][to]) { return false; }
    auto gap = (fromOffset + width - toOffset) % width;
    if (!gap) {
        // Distinct phases using one label in the same visit require an actual
        // command ordering; unrelated absent intermediates cannot supply it.
        if (from != to && *costs[from][to] != 0 && (!costs[to][from] || *costs[to][from] != 0)) {
            return false;
        }
        gap = width;
    }
    return *costs[from][to] <= gap;
}

bool certifyAssignment(const PeriodicReuseMatrix& costs, const PeriodicSharedAllocation& assignment)
{
    if (assignment.status != PeriodicSharedAllocationStatus::Success ||
        !assignment.budget || assignment.budget > maximumPaletteWidth ||
        assignment.phases.size() != costs.size()) { return false; }
    for (std::size_t i = 0; i < costs.size(); ++i) {
        const auto& from = assignment.phases[i];
        for (std::size_t j = 0; j < costs.size(); ++j) {
            const auto& to = assignment.phases[j];
            if (from.cycle != to.cycle) { continue; }
            if (from.laneCount != to.laneCount || !from.laneCount ||
                from.offset >= from.laneCount || to.offset >= to.laneCount ||
                !compatible(costs, i, j, from.offset, to.offset, from.laneCount)) { return false; }
        }
    }
    return true;
}

// A deterministic polynomial fallback when the cheapest cycle cover has an
// uncertified skipped phase (or conditional zero edges form a candidate cycle).
// Try at most six widths and six offsets; this is sufficient, not a minimum.
std::optional<std::pair<uint64_t, std::vector<uint64_t>>> greedyAssignment(const PeriodicReuseMatrix& costs)
{
    for (uint64_t width = 1; width <= maximumPaletteWidth; ++width) {
        std::vector<uint64_t> offsets;
        for (std::size_t i = 0; i < costs.size(); ++i) {
            std::optional<uint64_t> selected;
            for (uint64_t offset = 0; offset < width; ++offset) {
                bool valid = compatible(costs, i, i, offset, offset, width);
                for (std::size_t j = 0; valid && j < i; ++j) {
                    valid = compatible(costs, i, j, offset, offsets[j], width) &&
                            compatible(costs, j, i, offsets[j], offset, width);
                }
                if (valid) { selected = offset; break; }
            }
            if (!selected) { break; }
            offsets.push_back(*selected);
        }
        if (offsets.size() == costs.size()) { return std::make_pair(width, std::move(offsets)); }
    }
    return std::nullopt;
}

// Keep independent palettes when some pairs have no certified order. Each
// insertion reruns the direct all-pair proof; a missing intermediate is never
// used as a witness. The bounded offset search costs O(6^2 n^3) in total.
std::optional<PeriodicSharedAllocation> partitionAssignment(const PeriodicReuseMatrix& costs)
{
    if (auto together = greedyAssignment(costs)) {
        PeriodicSharedAllocation result;
        result.budget = together->first;
        result.phases.resize(costs.size());
        for (std::size_t i = 0; i < costs.size(); ++i) {
            result.phases[i].laneCount = result.budget;
            result.phases[i].offset = together->second[i];
        }
        return result;
    }
    struct Palette {
        std::vector<std::size_t> members;
        uint64_t width = 0;
        std::vector<uint64_t> offsets;
    };
    std::vector<Palette> palettes;
    for (std::size_t i = 0; i < costs.size(); ++i) {
        if (!costs[i][i]) { return std::nullopt; }
        auto increase = std::max(uint64_t(1), *costs[i][i]);
        if (increase > maximumPaletteWidth) { return std::nullopt; }
        std::optional<std::size_t> chosen;
        Palette next{{i}, increase, {0}};
        for (std::size_t g = 0; g < palettes.size(); ++g) {
            auto members = palettes[g].members; members.push_back(i);
            PeriodicReuseMatrix subset(members.size(), std::vector<std::optional<uint64_t>>(members.size()));
            for (std::size_t a = 0; a < members.size(); ++a) {
                for (std::size_t b = 0; b < members.size(); ++b) { subset[a][b] = costs[members[a]][members[b]]; }
            }
            auto placed = greedyAssignment(subset);
            if (!placed || placed->first < palettes[g].width ||
                placed->first - palettes[g].width >= increase) { continue; }
            increase = placed->first - palettes[g].width;
            chosen = g;
            next = {std::move(members), placed->first, std::move(placed->second)};
        }
        if (chosen) { palettes[*chosen] = std::move(next); }
        else { palettes.push_back(std::move(next)); }
    }
    PeriodicSharedAllocation result;
    result.phases.resize(costs.size());
    for (std::size_t g = 0; g < palettes.size(); ++g) {
        const auto& palette = palettes[g];
        if (palette.width > maximumPaletteWidth - result.budget) { return std::nullopt; }
        for (std::size_t i = 0; i < palette.members.size(); ++i) {
            auto& phase = result.phases[palette.members[i]];
            phase.cycle = g; phase.laneBegin = result.budget;
            phase.laneCount = palette.width; phase.offset = palette.offsets[i];
        }
        result.budget += palette.width;
    }
    return result;
}

std::optional<RegionalAllocationGroup> allocateLanes(RepeatedRegionState& state, ArrayRef<Lane> lanes)
{
    std::vector<std::size_t> live;
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        if (!lanes[i].bounds.first.empty()) { live.push_back(i); }
    }
    const auto n = live.size();
    if (n > UINT32_MAX || (n && n > std::vector<std::optional<uint64_t>>().max_size() / n)) {
        return std::nullopt;
    }
    PeriodicReuseMatrix costs(n, std::vector<std::optional<uint64_t>>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            costs[i][j] = reuseCost(state, lanes[live[i]], lanes[live[j]]);
        }
    }
    auto assignment = allocatePeriodicShared(costs);
    if (!certifyAssignment(costs, assignment)) {
        auto partitioned = partitionAssignment(costs);
        if (!partitioned || (n && !certifyAssignment(costs, *partitioned))) { return std::nullopt; }
        assignment = std::move(*partitioned);
    }
    // The existing typed record mapping requires a positive placeholder budget
    // even when every guarded record is absent; no command uses that label.
    const auto budget = std::max(uint64_t(1), assignment.budget);
    RegionalAllocationGroup result{lanes.front().group.sourcePipe, lanes.front().group.targetPipe, budget, {}};
    result.lanes.resize(budget);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& phase = assignment.phases[i];
        if (!appendLane(state, result, lanes[live[i]], phase.laneBegin, phase.laneCount, phase.offset)) {
            return std::nullopt;
        }
    }
    // Absent records execute no commands and consume no additional lane.
    for (const auto& lane : lanes) {
        if (lane.bounds.first.empty() && !appendLane(state, result, lane, 0, 1, 0)) { return std::nullopt; }
    }
    return result;
}
} // namespace

std::shared_ptr<RegionalAllocationSummary> periodicLaneAllocation(
    RepeatedRegionState& state, const RegionalAllocationSummary& coalesced,
    ArrayRef<std::pair<uint32_t, std::size_t>> crossings)
{
    std::vector<Lane> lanes;
    for (const auto& group : coalesced.groups) {
        for (auto lane : constantAllocationLanes(group)) {
            if (!addLane(state, lanes, std::move(lane), 0, true)) { return {}; }
        }
    }
    if (!addCrossings(state, lanes, crossings)) { return {}; }
    auto result = std::make_shared<RegionalAllocationSummary>();
    if (lanes.empty()) { return result; }
    std::map<std::pair<uint32_t, uint32_t>, std::vector<Lane>> directions;
    for (auto& lane : lanes) {
        const auto direction = regionalAllocationDirection(lane.group, lane.group.members.front());
        if (llvm::any_of(lane.group.members, [&](const auto& member) {
            return regionalAllocationDirection(lane.group, member) != direction;
        })) { return {}; }
        directions[direction].push_back(std::move(lane));
    }
    for (auto& direction : directions) {
        auto allocated = allocateLanes(state, direction.second);
        if (!allocated) { return {}; }
        result->groups.push_back(std::move(*allocated));
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
