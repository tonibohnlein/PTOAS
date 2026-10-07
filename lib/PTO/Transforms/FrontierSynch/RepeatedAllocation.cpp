// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RepeatedAllocation.h"
#include "RepeatedLaneAllocation.h"
#include "RepeatedRegionInternal.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
constexpr uint64_t maximumPaletteWidth = 6;
using Selectors = std::vector<RegionalSelector>;

// Preserve original event coordinates. Selecting extrema changes guards only,
// avoiding selected-coordinate queries in the nested reachability backend.
bool addExtremum(const RegionalAnalysis& body, Selectors& values, RegionalSelector next, bool first)
{
    auto& e = *body.expressions;
    for (auto& prior : values) {
        auto earlier = first ? regionalReferenceBefore(body, next.event, prior.event) :
                               regionalReferenceBefore(body, prior.event, next.event);
        if (!earlier) { return false; }
        auto old = prior.present;
        // Coordinates may be undefined outside their selector's presence.
        // Compare only when both alternatives are present; keep each original
        // presence expression intact when its competitor is absent.
        auto choose = e.select(next.present, e.select(old, *earlier, e.boolean(true)), e.boolean(false));
        prior.present = e.select(choose, e.boolean(false), old);
        next.present = choose;
    }
    values.erase(std::remove_if(values.begin(), values.end(), [&](const auto& value) {
        return e.constantValue(value.present) == 0;
    }), values.end());
    if (e.constantValue(next.present) != 0) { values.push_back(std::move(next)); }
    return true;
}
std::optional<std::pair<Selectors, Selectors>> extrema(
    const RegionalAnalysis& body, const RegionalAllocationGroup& group)
{
    Selectors first, last;
    auto& e = *body.expressions;
    for (const auto& member : group.members) {
        auto ps = regionalPresence(body, member.firstSource), pt = regionalPresence(body, member.lastTarget);
        if (!ps || !pt) { return std::nullopt; }
        auto active = e.select(member.active,
            e.select(*ps, *pt, e.boolean(false)), e.boolean(false));
        if (e.constantValue(active) == 0) { continue; }
        if (!addExtremum(body, first, {member.firstSource, active}, true) ||
            !addExtremum(body, last, {member.lastTarget, active}, false)) { return std::nullopt; }
    }
    return std::make_pair(std::move(first), std::move(last));
}
std::optional<uint64_t> lag(RepeatedRegionState& state, const RegionalAllocationGroup& group, uint64_t delay)
{
    if (!group.budget || group.budget > maximumPaletteWidth) { return std::nullopt; }
    auto bounds = extrema(state.body, group);
    if (!bounds) { return std::nullopt; }
    auto& e = state.e();
    for (uint64_t distance = 1; distance <= maximumPaletteWidth / group.budget; ++distance) {
        // No two source visits in this finite invocation can reuse this bank.
        auto firstPeriod = state.originalBegin == RegionExpressions::invalid ? e.constant(0) :
            e.div(state.originalBegin, e.constant(state.phaseCount));
        auto count = e.constantValue(e.sub(state.trips, firstPeriod));
        if (count && *count <= distance) { return distance; }
        if (distance < delay) { continue; }
        bool certified = true;
        for (const auto& end : bounds->second) {
            for (const auto& begin : bounds->first) {
                auto active = e.land(end.present, begin.present);
                if (e.constantValue(active) == 0) { continue; }
                auto reaches = state.relativeQuery(end.event, begin.event, distance - delay);
                if (!reaches || !e.implies(active, *reaches)) {
                    certified = false; break;
                }
            }
            if (!certified) { break; }
        }
        if (certified) { return distance; }
    }
    return std::nullopt;
}
std::optional<std::pair<Id, Id>> phaseInterval(RepeatedRegionState& state, uint32_t type)
{
    auto& e = state.e();
    auto q = state.phaseCount;
    if (!q || (!state.typePhases.empty() && type >= state.typePhases.size())) { return std::nullopt; }
    auto phase = e.constant(state.typePhases.empty() ? 0 : state.typePhases[type]);
    auto begin = state.originalBegin == RegionExpressions::invalid ? e.constant(0) : state.originalBegin;
    auto end = state.originalTrips == RegionExpressions::invalid ? state.trips : state.originalTrips;
    auto edge = [&](Id limit) {
        return e.add(e.div(limit, e.constant(q)),
            e.select(e.lt(phase, e.rem(limit, e.constant(q))), e.constant(1), e.constant(0)));
    };
    return std::make_pair(edge(begin), edge(end));
}
bool liftEnvelope(RepeatedRegionState& state, RegionalAllocationMember& member, uint64_t delay)
{
    auto source = phaseInterval(state, member.firstSource.type), target = phaseInterval(state, member.lastTarget.type);
    if (!source || !target) { return false; }
    auto& e = state.e();
    auto shifted = [&](Id value) {
        return e.select(e.lt(value, e.constant(delay)), e.constant(0), e.sub(value, e.constant(delay)));
    };
    auto targetFirst = shifted(target->first);
    auto first = e.select(e.lt(source->first, targetFirst), targetFirst, source->first);
    auto end = e.minimum(source->second, shifted(target->second));
    member.active = e.land(member.active, e.lt(first, end));
    member.firstSource.visits.insert(member.firstSource.visits.begin(), first);
    // The subtraction is observed only under first < end; empty spans are absent.
    auto last = e.add(e.sub(end, e.constant(1)), e.constant(delay));
    member.lastTarget.visits.insert(member.lastTarget.visits.begin(), last);
    return true;
}
bool singlePhase(const RepeatedRegionState& state, const RegionalAllocationMember& member)
{
    if (state.typePhases.empty()) { return true; }
    auto source = member.firstSource.type, target = member.lastTarget.type;
    return source < state.typePhases.size() && target < state.typePhases.size() &&
        state.typePhases[source] == state.typePhases[target];
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
std::optional<RegionalAllocationGroup> liftGroup(
    RepeatedRegionState& state, RegionalAllocationGroup group, uint64_t delay, bool child)
{
    if (child && llvm::any_of(group.members, [&](const auto& member) {
        return !canRestrictRepeatedAllocationMember(state, member);
    })) {
        // A partial phase must omit a complete child chain. Complete periods
        // preserve cross-phase chains too; otherwise actual clipped extrema
        // must come from the producer.
        return std::nullopt;
    }
    auto distance = lag(state, group, delay);
    if (!distance) { return std::nullopt; }
    for (auto& member : group.members) {
        PhysicalTupleRule rule;
        if (child) {
            if (!member.tupleRule || !validPhysicalTupleRule(*member.tupleRule, group.budget)) {
                return std::nullopt;
            }
            rule = *member.tupleRule;
            if (rule.coordinateCount == UINT64_MAX) { return std::nullopt; }
            ++rule.coordinateCount;
            for (auto& term : rule.terms) { if (term.coordinate) { ++term.coordinate; } }
            rule.terms.push_back({1, 1, 0, *distance, group.budget});
        } else { rule.terms.push_back({0, 1, 0, *distance, 1}); }
        member.tupleRule = std::move(rule);
        member.singletonHandoff = false;
        if (!liftEnvelope(state, member, delay)) { return std::nullopt; }
    }
    group.budget *= *distance;
    return group;
}
std::optional<uint64_t> constantOffset(const PhysicalTupleRule& rule)
{
    uint64_t offset = rule.base;
    for (const auto& term : rule.terms) {
        if (!term.scale) { continue; }
        if (!term.modulus || term.stride % term.modulus) { return std::nullopt; }
        offset += term.scale * (term.phase % term.modulus);
    }
    return offset;
}
// Constant child lanes are independent certified chains. Keep their individual
// envelopes instead of reserving the entire palette until its last consumer.
// This enumerates existing records only, never loop visits or payloads.
std::vector<RegionalAllocationGroup> constantLanes(const RegionalAllocationGroup& group)
{
    std::map<uint64_t, RegionalAllocationGroup> lanes;
    for (const auto& member : group.members) {
        if (!member.tupleRule || !validPhysicalTupleRule(*member.tupleRule, group.budget)) { return {group}; }
        auto offset = constantOffset(*member.tupleRule);
        if (!offset) { return {group}; }
        auto [entry, added] = lanes.try_emplace(*offset);
        if (added) { entry->second = {group.sourcePipe, group.targetPipe, 1, {}}; }
        auto copy = member;
        copy.tupleRule = PhysicalTupleRule{member.tupleRule->coordinateCount, 0, {}};
        copy.stride = 0;
        copy.phase = 0;
        entry->second.members.push_back(std::move(copy));
    }
    std::vector<RegionalAllocationGroup> result;
    for (auto& [offset, lane] : lanes) { result.push_back(std::move(lane)); }
    return result;
}
} // namespace
std::optional<RepeatedAllocationExtrema> repeatedAllocationExtrema(
    const RegionalAnalysis& body, const RegionalAllocationGroup& group)
{
    return extrema(body, group);
}
bool canRestrictRepeatedAllocationMember(RepeatedRegionState& state, const RegionalAllocationMember& member)
{
    return member.singletonHandoff || singlePhase(state, member) || completePeriods(state);
}
bool liftRepeatedAllocationEnvelope(RepeatedRegionState& state, RegionalAllocationMember& member, uint64_t delay)
{
    if (!liftEnvelope(state, member, delay)) { return false; }
    member.singletonHandoff = false;
    return true;
}
std::vector<RegionalAllocationGroup> constantAllocationLanes(const RegionalAllocationGroup& group)
{
    return constantLanes(group);
}
std::shared_ptr<RegionalAllocationSummary> repeatedRegionalAllocation(
    RepeatedRegionState& state, const RegionalAllocationSummary& child,
    ArrayRef<std::pair<uint32_t, std::size_t>> crossings)
{
    auto coalesced = coalesceRegionalAllocation(state.body, child);
    if (!coalesced) { return {}; }
    if (auto periodic = periodicLaneAllocation(state, *coalesced, crossings)) { return periodic; }
    auto result = std::make_shared<RegionalAllocationSummary>();
    for (const auto& group : coalesced->groups) {
        for (auto lane : constantLanes(group)) {
            auto lifted = liftGroup(state, std::move(lane), 0, true);
            if (!lifted) { return {}; }
            result->groups.push_back(std::move(*lifted));
        }
    }
    for (const auto& [record, index] : crossings) {
        if (index >= state.crossings.size()) { return {}; }
        const auto& crossing = state.crossings[index];
        if (crossing.source.type >= state.body.anchors.size() ||
            crossing.target.type >= state.body.anchors.size()) { return {}; }
        const auto& source = state.body.anchors[crossing.source.type];
        const auto& target = state.body.anchors[crossing.target.type];
        if (!source.phase || !target.phase) { return {}; }
        auto p = static_cast<uint32_t>(source.phase->kPipeValue);
        auto q = static_cast<uint32_t>(target.phase->kPipeValue);
        if (p == q) { continue; }
        auto first = crossing.source, last = crossing.target;
        first.kind = PeriodicEventKind::Start;
        last.kind = PeriodicEventKind::Completion;
        RegionalAllocationGroup group{p, q, 1, {{record, 0, 0, first, last, crossing.guard}}};
        auto lifted = liftGroup(state, std::move(group), 1, false);
        if (!lifted) { return {}; }
        result->groups.push_back(std::move(*lifted));
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
