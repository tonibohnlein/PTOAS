// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RegionalConstantPalette.h"
#include "RegionalCoalescingProfile.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalLaneExports.h"
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
    RegionalCoalescingProfile& work;
    std::map<QueryKey, std::optional<Id>> queries;
    std::map<std::pair<Id, Id>, bool> proofs;
    bool proves(Id premise, Id consequence)
    {
        work.visit(work.proofRequests);
        auto key = std::make_pair(premise, consequence);
        if (auto found = proofs.find(key); found != proofs.end()) { work.visit(work.proofHits); return found->second; }
        RegionalCoalescingProfile::Scope timer(work, work.proofNs);
        auto& e = *region.expressions;
        const bool answer = e.implies(premise, consequence) || e.constantUnder(premise, premise) == 0 ||
            e.constantUnder(premise, consequence) == 1;
        proofs.emplace(key, answer);
        return answer;
    }
    std::optional<Id> query(const RegionalEvent& a, const RegionalEvent& b)
    {
        work.visit(work.queryRequests);
        QueryKey key{{a.type, a.ordinal, a.visits}, {b.type, b.ordinal, b.visits}};
        if (auto found = queries.find(key); found != queries.end()) {
            work.visit(work.queryHits);
            return found->second;
        }
        work.visit(work.queryConstructions);
        RegionalCoalescingProfile::Scope timer(work, work.queryNs);
        auto answer = regionalHandoffReuse(region, a, b);
        queries.emplace(std::move(key), answer);
        return answer;
    }
    struct Envelope { RegionalEvent first, last; Id active; };
    std::optional<RegionalEvent> extremum(RegionalEvent a, RegionalEvent b, Id active, bool first)
    {
        // Certify native event order, not just record or reference numbering.
        // Both operands have the same kind and original pipe.
        work.visit(work.envelopeQueryRequests);
        auto ab = regionalReachability(region, a, b);
        if (ab && proves(active, *ab)) { return first ? a : b; }
        work.visit(work.envelopeQueryRequests);
        auto ba = regionalReachability(region, b, a);
        if (ba && proves(active, *ba)) { return first ? b : a; }
        return std::nullopt;
    }
    std::optional<Envelope> combine(const std::optional<Envelope>& a, const std::optional<Envelope>& b)
    {
        // A bound must exist whenever any member of the combined group does.
        // Different activities need a selected endpoint, not an unconditional
        // choice of one group's bound. Keep the all-member fallback there.
        if (!a || !b || a->active != b->active) { return std::nullopt; }
        auto first = extremum(a->first, b->first, a->active, true);
        auto last = extremum(a->last, b->last, a->active, false);
        if (!first || !last) { return std::nullopt; }
        return Envelope{*first, *last, a->active};
    }
    std::optional<Envelope> envelope(const RegionalAllocationGroup& group)
    {
        work.visit(work.envelopeRequests);
        RegionalCoalescingProfile::Scope timer(work, work.envelopeNs);
        if (group.sourcePipe == group.targetPipe) { return std::nullopt; }
        std::optional<Envelope> result;
        for (const auto& member : group.members) {
            if ((result && result->active != member.active) ||
                member.firstSource.type >= region.anchors.size() ||
                member.lastTarget.type >= region.anchors.size()) { return std::nullopt; }
            const auto& source = region.anchors[member.firstSource.type];
            const auto& target = region.anchors[member.lastTarget.type];
            if (!source.phase || !target.phase ||
                static_cast<uint32_t>(source.phase->kPipeValue) != group.sourcePipe ||
                static_cast<uint32_t>(target.phase->kPipeValue) != group.targetPipe) { return std::nullopt; }
            auto ps = regionalPresence(region, member.firstSource);
            auto pt = regionalPresence(region, member.lastTarget);
            if (!ps || !pt || !proves(member.active, *ps) || !proves(member.active, *pt)) {
                return std::nullopt;
            }
            Envelope next{member.firstSource, member.lastTarget, member.active};
            next.first.kind = PeriodicEventKind::Start;
            next.last.kind = PeriodicEventKind::Completion;
            result = result ? combine(result, next) : std::optional<Envelope>(next);
            if (!result) { return std::nullopt; }
        }
        if (result) { work.visit(work.envelopeAccepted); }
        return result;
    }
    bool separated(Id active, const RegionalEvent& lastA, const RegionalEvent& firstB,
                   const RegionalEvent& lastB, const RegionalEvent& firstA)
    {
        auto& e = *region.expressions;
        if (proves(active, e.boolean(false))) { return true; }
        auto ab = query(lastA, firstB);
        if (ab && proves(active, *ab)) { return true; }
        auto ba = query(lastB, firstA);
        if (ba && proves(active, *ba)) { return true; }
        return ab && ba && proves(active, e.lor(*ab, *ba));
    }
    bool compatible(const RegionalAllocationGroup& a, const RegionalAllocationGroup& b,
                    const std::optional<Envelope>& boundsA, const std::optional<Envelope>& boundsB)
    {
        auto& e = *region.expressions;
        // Native bounds exist under the group's exact common activity and
        // cover every member lifetime there. Separation under both activities
        // therefore covers the full Cartesian product.
        // Overlapping/unavailable bounds do not reject interleaved lifetimes:
        // retain the original all-pair sufficient proof in those cases.
        if (boundsA && boundsB && separated(e.land(boundsA->active, boundsB->active),
                boundsA->last, boundsB->first, boundsB->last, boundsA->first)) { return true; }
        for (const auto& x : a.members) {
            for (const auto& y : b.members) {
                work.visit(work.memberPairs);
                if (!separated(e.land(x.active, y.active), x.lastTarget, y.firstSource,
                               y.lastTarget, x.firstSource)) { return false; }
            }
        }
        return true;
    }
};
std::optional<std::vector<RegionalAllocationGroup>> split(const RegionalAllocationGroup& group)
{
    std::map<std::tuple<uint32_t, uint32_t, uint64_t>, RegionalAllocationGroup> lanes;
    for (const auto& member : group.members) {
        auto lane = constantLane(group, member);
        if (!lane) { return std::nullopt; }
        const auto direction = regionalAllocationDirection(group, member);
        auto [entry, added] = lanes.try_emplace(std::make_tuple(direction.first, direction.second, *lane));
        if (added) { entry->second = {direction.first, direction.second, 1, {}}; }
        auto copy = member;
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
    const RegionalAnalysis& region, const RegionalAllocationSummary& input, AllocationProfile* profile)
{
    if (!region.expressions || !region.capabilities.exactQueries) { return {}; }
    RegionalCoalescingProfile work(profile, "constant");
    auto out = std::make_shared<RegionalAllocationSummary>();
    struct ConstantPalette { std::size_t index; std::optional<Compatibility::Envelope> bounds; };
    std::vector<ConstantPalette> constants;
    Compatibility proof{region, work, {}, {}};
    for (const auto& group : input.groups) {
        work.visit(work.groups);
        if (work.profile) { accumulateCost(work.members, group.members.size()); }
        if (!group.budget) { return {}; }
        std::optional<std::vector<RegionalAllocationGroup>> lanes;
        {
            RegionalCoalescingProfile::Scope timer(work, work.summaryNs);
            lanes = split(group);
        }
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
            auto bounds = proof.envelope(lane);
            for (std::size_t i = 0; i < constants.size(); ++i) {
                // Check all original lifetimes against all existing users. A
                // guarded compatibility relation is not assumed transitive.
                const auto& candidate = out->groups[constants[i].index];
                if (candidate.sourcePipe != lane.sourcePipe || candidate.targetPipe != lane.targetPipe) { continue; }
                if (proof.compatible(candidate, lane, constants[i].bounds, bounds)) { destination = i; break; }
            }
            if (!destination) {
                constants.push_back({out->groups.size(), std::move(bounds)});
                out->groups.push_back(std::move(lane));
                continue;
            }
            auto& palette = constants[*destination];
            palette.bounds = proof.combine(palette.bounds, bounds);
            auto& merged = out->groups[palette.index];
            for (auto& member : lane.members) { merged.members.push_back(std::move(member)); }
        }
    }
    for (const auto& palette : constants) {
        // Formulas were remapped to lane zero; rebuild matching selectors.
        if (!exportConstantRegionalLanes(out->groups[palette.index])) { return {}; }
    }
    return out;
}
} // namespace mlir::pto::frontiersynch
