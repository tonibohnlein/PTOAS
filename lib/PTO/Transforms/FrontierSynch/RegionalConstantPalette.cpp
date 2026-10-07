// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RegionalConstantPalette.h"
#include "PTO/Transforms/FrontierSynch/RegionalLaneExports.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using EventKey = std::tuple<uint32_t, Id, std::vector<Id>>;
using QueryKey = std::pair<EventKey, EventKey>;
std::optional<uint64_t> constantLane(const RegionalAllocationGroup& group,
                                   const RegionalAllocationMember& member)
{
    if (!member.tupleRule) {
        return member.stride % group.budget ? std::nullopt :
            std::optional<uint64_t>(member.phase % group.budget);
    }
    const auto& rule = *member.tupleRule;
    if (!validPhysicalTupleRule(rule, group.budget)) { return std::nullopt; }
    uint64_t lane = rule.base;
    for (const auto& term : rule.terms) {
        if (!term.scale) { continue; }
        if (term.stride % term.modulus) { return std::nullopt; }
        // validPhysicalTupleRule bounds every term and their sum by budget.
        lane += term.scale * (term.phase % term.modulus);
    }
    return lane;
}
struct Compatibility {
    const RegionalAnalysis& region;
    std::map<QueryKey, std::optional<Id>> queries;
    std::map<std::pair<Id, Id>, bool> proofs;
    bool proves(Id premise, Id consequence)
    {
        auto key = std::make_pair(premise, consequence);
        if (auto found = proofs.find(key); found != proofs.end()) { return found->second; }
        auto& e = *region.expressions;
        const bool answer = e.implies(premise, consequence) || e.constantUnder(premise, premise) == 0 ||
            e.constantUnder(premise, consequence) == 1;
        proofs.emplace(key, answer);
        return answer;
    }
    std::optional<Id> query(const RegionalEvent& a, const RegionalEvent& b)
    {
        QueryKey key{{a.type, a.ordinal, a.visits}, {b.type, b.ordinal, b.visits}};
        if (auto found = queries.find(key); found != queries.end()) { return found->second; }
        auto answer = regionalHandoffReuse(region, a, b);
        queries.emplace(std::move(key), answer);
        return answer;
    }
    bool compatible(const RegionalAllocationGroup& a, const RegionalAllocationGroup& b)
    {
        auto& e = *region.expressions;
        for (const auto& x : a.members) {
            for (const auto& y : b.members) {
                auto active = e.land(x.active, y.active);
                if (proves(active, e.boolean(false))) { continue; }
                auto xy = query(x.lastTarget, y.firstSource);
                if (xy && proves(active, *xy)) { continue; }
                auto yx = query(y.lastTarget, x.firstSource);
                if (yx && proves(active, *yx)) { continue; }
                if (!xy || !yx || !proves(active, e.lor(*xy, *yx))) { return false; }
            }
        }
        return true;
    }
};
std::optional<std::vector<RegionalAllocationGroup>> split(const RegionalAllocationGroup& group)
{
    std::map<uint64_t, RegionalAllocationGroup> lanes;
    for (const auto& member : group.members) {
        auto lane = constantLane(group, member);
        if (!lane) { return std::nullopt; }
        auto [entry, added] = lanes.try_emplace(*lane);
        if (added) { entry->second = {group.sourcePipe, group.targetPipe, 1, {}}; }
        auto copy = member;
        const auto direction = regionalAllocationDirection(group, member);
        copy.sourcePipe = direction.first;
        copy.targetPipe = direction.second;
        copy.tupleRule = PhysicalTupleRule{member.tupleRule ? member.tupleRule->coordinateCount : 1, 0, {}};
        copy.stride = copy.phase = 0;
        entry->second.members.push_back(std::move(copy));
    }
    std::vector<RegionalAllocationGroup> result;
    for (auto& entry : lanes) { result.push_back(std::move(entry.second)); }
    return result;
}
} // namespace
std::shared_ptr<RegionalAllocationSummary> coalesceConstantRegionalAllocation(
    const RegionalAnalysis& region, const RegionalAllocationSummary& input)
{
    if (!region.expressions || !region.capabilities.exactQueries) { return {}; }
    auto out = std::make_shared<RegionalAllocationSummary>();
    std::vector<std::size_t> constants;
    Compatibility proof{region, {}, {}};
    for (const auto& group : input.groups) {
        if (!group.budget) { return {}; }
        auto lanes = split(group);
        if (!lanes) {
            auto copy = group;
            for (auto& member : copy.members) {
                if (!member.tupleRule) {
                    member.tupleRule = PhysicalTupleRule{1, 0,
                        {{0, member.stride % group.budget, member.phase % group.budget, group.budget, 1}}};
                }
            }
            out->groups.push_back(std::move(copy));
            continue;
        }
        for (auto& lane : *lanes) {
            std::optional<std::size_t> destination;
            for (auto id : constants) {
                // Check all original lifetimes against all existing users. A
                // guarded compatibility relation is not assumed transitive.
                if (proof.compatible(out->groups[id], lane)) { destination = id; break; }
            }
            if (!destination) {
                constants.push_back(out->groups.size());
                out->groups.push_back(std::move(lane));
                continue;
            }
            auto& merged = out->groups[*destination];
            for (auto& member : lane.members) { merged.members.push_back(std::move(member)); }
        }
    }
    for (auto id : constants) {
        // Formulas were remapped to lane zero; rebuild matching selectors.
        if (!exportConstantRegionalLanes(out->groups[id])) { return {}; }
    }
    return out;
}
} // namespace mlir::pto::frontiersynch
