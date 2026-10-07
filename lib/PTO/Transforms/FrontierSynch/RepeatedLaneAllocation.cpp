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

namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
constexpr uint64_t maximumPaletteWidth = 6;
struct Lane {
    RegionalAllocationGroup group;
    RepeatedAllocationExtrema bounds;
    Id active = RegionExpressions::invalid;
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
    auto bounds = repeatedAllocationExtrema(state.body, group);
    if (!bounds || bounds->first.empty() || bounds->second.empty()) { return false; }
    auto active = state.e().boolean(false);
    for (const auto& selector : bounds->first) { active = state.e().lor(active, selector.present); }
    lanes.push_back({std::move(group), std::move(*bounds), active, delay, child});
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

bool completePeriods(RepeatedRegionState& state)
{
    auto& e = state.e();
    if (!state.phaseCount) { return false; }
    auto begin = state.originalBegin == RegionExpressions::invalid ? e.constant(0) : state.originalBegin;
    auto end = state.originalTrips == RegionExpressions::invalid ? state.trips : state.originalTrips;
    auto modulus = e.constant(state.phaseCount);
    return e.constantValue(e.rem(begin, modulus)) == 0 && e.constantValue(e.rem(end, modulus)) == 0;
}

bool samePhase(const RepeatedRegionState& state, const Lane& from, const Lane& to)
{
    if (state.typePhases.empty()) { return true; }
    std::optional<uint32_t> phase;
    for (const auto* lane : {&from, &to}) {
        for (const auto* selectors : {&lane->bounds.first, &lane->bounds.second}) {
            for (const auto& selector : *selectors) {
                if (selector.event.type >= state.typePhases.size()) { return false; }
                auto next = state.typePhases[selector.event.type];
                if (phase && *phase != next) { return false; }
                phase = next;
            }
        }
    }
    return true;
}

std::optional<uint64_t> reuseCost(RepeatedRegionState& state, const Lane& from,
                                const Lane& to, bool wholePeriods)
{
    auto& e = state.e();
    // A cycle must be wholly present or wholly absent in each parameter case.
    // Pairwise conditional edges alone do not justify skipping a middle lane.
    if (!e.implies(from.active, to.active) || !e.implies(to.active, from.active)) { return std::nullopt; }
    // A partial period can omit one phase while retaining another. Without a
    // clipped-chain proof, share only lanes with identical phase boundaries.
    if (!wholePeriods && !samePhase(state, from, to)) { return std::nullopt; }
    // Restrict to certified nonnegative source displacements. This is a
    // sufficient matrix, not the set of every possible hardware assignment.
    for (uint64_t gap = from.delay; gap <= maximumPaletteWidth; ++gap) {
        bool proven = true;
        for (const auto& end : from.bounds.second) {
            for (const auto& begin : to.bounds.first) {
                auto active = e.land(end.present, begin.present);
                if (e.constantValue(active) == 0) { continue; }
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
                if (!query || !e.implies(active, *query)) { proven = false; break; }
            }
            if (!proven) { break; }
        }
        if (proven) { return gap; }
    }
    return std::nullopt;
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
        result.members.push_back(std::move(member));
    }
    return true;
}

std::optional<RegionalAllocationGroup> allocateLanes(RepeatedRegionState& state, ArrayRef<Lane> lanes)
{
    const auto n = lanes.size();
    if (!n || n > UINT32_MAX || n > std::vector<std::optional<uint64_t>>().max_size() / n) {
        return std::nullopt;
    }
    PeriodicReuseMatrix costs(n, std::vector<std::optional<uint64_t>>(n));
    const bool wholePeriods = completePeriods(state);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            costs[i][j] = reuseCost(state, lanes[i], lanes[j], wholePeriods);
        }
    }
    auto assignment = allocatePeriodicShared(costs);
    if (assignment.status != PeriodicSharedAllocationStatus::Success ||
        !assignment.budget || assignment.budget > maximumPaletteWidth) { return std::nullopt; }
    RegionalAllocationGroup result{lanes.front().group.sourcePipe, lanes.front().group.targetPipe,
                                   assignment.budget, {}};
    // Equal activity and nonnegative displacements make each cycle a chain.
    // Every intermediate consumer command precedes the next publication.
    // With gap >= delay, between retained publications all intermediate handoffs remain
    // inside the finite interval; missing terminal crossings cannot break reuse.
    // The phase check above gives all lanes the same interval boundaries.
    // Retain each original record once and index its cycle by the outer source
    // coordinate. This exports conservative envelopes, never stale lane data.
    for (std::size_t i = 0; i < n; ++i) {
        const auto& phase = assignment.phases[i];
        if (!appendLane(state, result, lanes[i], phase.laneBegin, phase.laneCount, phase.offset)) {
            return std::nullopt;
        }
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
    auto allocated = allocateLanes(state, lanes);
    if (!allocated) { return {}; }
    result->groups.push_back(std::move(*allocated));
    return result;
}
} // namespace mlir::pto::frontiersynch
