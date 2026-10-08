// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Reserve contiguous child palettes using their certified lifetime envelopes.
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include "RegionalConstantPalette.h"
#include "PTO/Transforms/FrontierSynch/RegionalLaneExports.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
namespace mlir::pto::frontiersynch {
std::shared_ptr<RegionalAllocationSummary> coalesceRegionalAllocation(
    const RegionalAnalysis& region, const RegionalAllocationSummary& input)
{
    if (!region.expressions || !region.capabilities.exactQueries) { return {}; }
    auto& e = *region.expressions;
    const auto& groups = input.groups;
    std::set<uint32_t> records;
    for (const auto& group : groups) {
        if (!group.budget || group.budget > INT64_MAX || group.members.empty()) { return {}; }
        if (!group.lanes.empty() && group.lanes.size() != group.budget) { return {}; }
        for (const auto& member : group.members) {
            if (!records.insert(member.record).second || !validRegionalEvent(region, member.firstSource) ||
                !validRegionalEvent(region, member.lastTarget) || member.active >= e.size() ||
                !e.isBoolean(member.active) ||
                (member.tupleRule && !validPhysicalTupleRule(*member.tupleRule, group.budget))) { return {}; }
        }
    }
    // Preserve dynamic lane mappings. Split mixed constant groups by direction;
    // only same-direction lifetimes need a proof for numeric-ID sharing.
    if (llvm::any_of(groups, [&](const auto& group) {
        return !group.lanes.empty() || llvm::any_of(group.members, [&](const auto& member) {
            return regionalAllocationDirection(group, member) != std::make_pair(group.sourcePipe, group.targetPipe);
        });
    })) {
        return coalesceConstantRegionalAllocation(region, input);
    }
    using EventKey = std::tuple<uint32_t, RegionExpressions::Id, PeriodicEventKind,
                                std::vector<RegionExpressions::Id>>;
    using QueryKey = std::pair<EventKey, EventKey>;
    std::map<QueryKey, std::optional<RegionExpressions::Id>> queryCache;
    auto reaches = [&](const RegionalEvent& a, const RegionalEvent& b) {
        QueryKey key{{a.type, a.ordinal, a.kind, a.visits}, {b.type, b.ordinal, b.kind, b.visits}};
        auto found = queryCache.find(key);
        if (found != queryCache.end()) { return found->second; }
        auto answer = regionalReachability(region, a, b);
        queryCache.emplace(std::move(key), answer);
        return answer;
    };
    auto compatible = [&](const RegionalAllocationGroup& a, const RegionalAllocationGroup& b) {
        for (const auto& x : a.members) {
            for (const auto& y : b.members) {
                auto active = e.land(x.active, y.active);
                if (e.constantValue(active) == 0) { continue; }
                auto xy = reaches(x.lastTarget, y.firstSource);
                if (xy && e.implies(active, *xy)) { continue; }
                auto yx = reaches(y.lastTarget, x.firstSource);
                if (!xy || !yx || !e.implies(active, e.lor(*xy, *yx))) { return false; }
            }
        }
        return true;
    };
    // Greedy contiguous palette placement is sufficient, not a minimum-width
    // allocator. No successful placement introduces a synchronization edge.
    std::vector<std::size_t> order(groups.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return groups[a].budget > groups[b].budget;
    });
    std::vector<std::optional<uint64_t>> offsets(groups.size());
    for (auto i : order) {
        const auto& group = groups[i];
        std::vector<std::pair<uint64_t, uint64_t>> occupied;
        for (std::size_t j = 0; j < groups.size(); ++j) {
            if (!offsets[j] || groups[j].sourcePipe != group.sourcePipe ||
                groups[j].targetPipe != group.targetPipe || compatible(group, groups[j])) { continue; }
            occupied.emplace_back(*offsets[j], *offsets[j] + groups[j].budget);
        }
        std::sort(occupied.begin(), occupied.end());
        uint64_t offset = 0;
        for (auto interval : occupied) {
            if (interval.second <= offset) { continue; }
            if (interval.first >= offset && interval.first - offset >= group.budget) { break; }
            offset = interval.second;
        }
        if (offset > static_cast<uint64_t>(INT64_MAX) - group.budget) { return {}; }
        offsets[i] = offset;
    }
    auto result = std::make_shared<RegionalAllocationSummary>();
    std::map<std::pair<uint32_t, uint32_t>, std::size_t> directions;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        const auto& group = groups[i];
        auto [entry, inserted] = directions.emplace(std::make_pair(group.sourcePipe, group.targetPipe),
                                                    result->groups.size());
        if (inserted) { result->groups.push_back({group.sourcePipe, group.targetPipe, 0, {}}); }
        auto& combined = result->groups[entry->second];
        combined.budget = std::max(combined.budget, *offsets[i] + group.budget);
        for (auto member : group.members) {
            if (!member.tupleRule) {
                member.tupleRule = PhysicalTupleRule{1, 0,
                    {{0, member.stride % group.budget, member.phase % group.budget, group.budget, 1}}};
            }
            member.tupleRule->base += *offsets[i];
            combined.members.push_back(std::move(member));
        }
    }
    for (auto& group : result->groups) { exportConstantRegionalLanes(group); }
    return result;
}
} // namespace mlir::pto::frontiersynch
