// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RegionalLaneExports.h"
#include "PTO/Transforms/FrontierSynch/PeriodicSharedCertificate.h"

namespace mlir::pto::frontiersynch {
std::pair<uint32_t, uint32_t> regionalAllocationDirection(
    const RegionalAllocationGroup& group, const RegionalAllocationMember& member)
{
    return {member.sourcePipe.value_or(group.sourcePipe), member.targetPipe.value_or(group.targetPipe)};
}
bool exportConstantRegionalLanes(RegionalAllocationGroup& group)
{
    if (!group.budget || group.budget > 6) { return false; }
    if (!group.lanes.empty()) { return group.lanes.size() == group.budget; }
    std::vector<RegionalAllocationLane> lanes(group.budget);
    for (const auto& member : group.members) {
        uint64_t offset = member.phase % group.budget;
        if (member.tupleRule) {
            const auto& rule = *member.tupleRule;
            if (!validPhysicalTupleRule(rule, group.budget)) { return false; }
            offset = rule.base;
            for (const auto& term : rule.terms) {
                if (!term.scale) { continue; }
                if (term.stride % term.modulus) { return false; }
                offset += term.scale * (term.phase % term.modulus);
            }
        } else if (member.stride % group.budget) { return false; }
        lanes[offset].firstSources.push_back({member.firstSource, member.active});
        lanes[offset].lastTargets.push_back({member.lastTarget, member.active});
    }
    group.lanes = std::move(lanes);
    return true;
}
std::shared_ptr<RegionalAllocationSummary> exportPeriodicSharedLanes(
    const RegionalAnalysis& region, const PeriodicAnalysis& periodic, RegionExpressions::Id trips)
{
    if (!region.expressions || trips >= region.expressions->size() || region.expressions->isBoolean(trips) ||
        region.anchors.size() != periodic.payloads.size() ||
        llvm::any_of(region.outerLoops, [](const auto& loops) { return !loops.empty(); })) { return {}; }
    auto assignment = buildPeriodicSharedAssignment(periodic);
    if (!assignment) { return {}; }
    auto out = std::make_shared<RegionalAllocationSummary>();
    auto& e = *region.expressions;
    const auto& allocation = assignment->allocation;
    for (const auto& cycle : allocation.cycles) {
        // Enumerate physical lanes only; never expand a numerical distance,
        // source ordinal domain, or trip count to make an export fit.
        if (!cycle.laneCount || cycle.laneCount > 6) { return {}; }
        const auto& first = periodic.generators[assignment->records[cycle.firstPhase]];
        RegionalAllocationGroup group{periodic.payloads[first.source].pipe,
            periodic.payloads[first.target].pipe, cycle.laneCount, {}};
        group.lanes.resize(cycle.laneCount);
        out->groups.push_back(std::move(group));
    }
    for (std::size_t phase = 0; phase < assignment->records.size(); ++phase) {
        const auto id = assignment->records[phase];
        const auto& record = periodic.generators[id];
        const auto& assigned = allocation.phases[phase];
        auto& group = out->groups[assigned.cycle];
        // Every chosen edge weighs d+h, so d <= its positive cycle width.
        if (record.displacement > group.budget) { return {}; }
        const auto distance = e.constant(record.displacement), modulus = e.constant(group.budget);
        const auto active = e.lt(distance, trips);
        const auto limit = e.select(active, e.sub(trips, distance), e.constant(0));
        RegionalAllocationMember member{id, 1, assigned.offset,
            {record.source, e.constant(0), PeriodicEventKind::Start},
            {record.target, e.sub(trips, e.constant(1)), PeriodicEventKind::Completion}, active};
        member.sourcePipe = periodic.payloads[record.source].pipe;
        member.targetPipe = periodic.payloads[record.target].pipe;
        member.tupleRule = PhysicalTupleRule{1, 0, {{0, 1, assigned.offset, group.budget, 1}}};
        group.members.push_back(std::move(member));
        for (uint64_t lane = 0; lane < group.budget; ++lane) {
            const auto residue = (lane + group.budget - assigned.offset) % group.budget;
            const auto first = e.constant(residue), present = e.lt(first, limit);
            // Guard before subtracting: inactive phases keep defined arithmetic
            // and cannot manufacture a boundary event for an empty invocation.
            const auto end = e.select(present, e.sub(limit, e.constant(1)), first);
            const auto last = e.sub(end, e.rem(e.sub(end, first), modulus));
            group.lanes[lane].firstSources.push_back(
                {{record.source, first, PeriodicEventKind::Start}, present});
            group.lanes[lane].lastTargets.push_back(
                {{record.target, e.add(last, distance), PeriodicEventKind::Completion}, present});
        }
    }
    return out;
}
} // namespace mlir::pto::frontiersynch
