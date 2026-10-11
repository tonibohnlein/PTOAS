// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Mandatory native-pipe envelopes make ordinary bounded palettes composable.
// Optional guards never supply assumed first/last occurrences; storage-backed
// coordinates require a distinct regional adapter and are not exported here.
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "BoundedLifetimeAllocationInternal.h"
#include <algorithm>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
// A guaranteed occurrence brackets every optional endpoint on its native pipe.
// Its existence in each visit, not the optional endpoint's guard, justifies the
// regional lifetime envelope. No global query is inferred from a window row.
std::optional<uint32_t> mandatoryEnvelope(const LifetimeWindowInput& w,
    llvm::ArrayRef<uint8_t> unconditional, uint32_t endpoint, bool first)
{
    std::optional<uint32_t> result;
    for (uint32_t site = 0; site < w.sites; ++site) {
        if (!unconditional[site] || w.payloads[site].pipe != w.payloads[endpoint].pipe) { continue; }
        if (first && site <= endpoint) { return site; }
        if (!first && site >= endpoint) { result = site; }
    }
    return result;
}
// A conditionally executed site may still be mandatory at the invocation
// boundary. Its retained presence query substitutes the original lower/step.
std::optional<uint32_t> certifiedEnvelope(RegionExpressions& e, const LifetimeWindowInput& w,
    llvm::ArrayRef<uint8_t> unconditional, uint32_t endpoint, bool first,
    RegionExpressions::Id ordinal, RegionExpressions::Id active,
    const detail::BoundedAllocationPresence& boundary)
{
    if (auto site = mandatoryEnvelope(w, unconditional, endpoint, first)) { return site; }
    if (!boundary.presence || boundary.expressions.get() != &e) { return {}; }
    for (uint32_t site = 0; site < w.sites; ++site) {
        if (w.payloads[site].pipe != w.payloads[endpoint].pipe ||
            (first ? site > endpoint : site < endpoint)) { continue; }
        auto kind = first ? PeriodicEventKind::Start : PeriodicEventKind::Completion;
        auto present = boundary.presence({site, ordinal, kind});
        if (!present || *present >= e.size() || !e.isBoolean(*present)) { continue; }
        if (e.implies(active, *present) || e.constantUnder(active, *present) == 1) { return site; }
    }
    return {};
}
struct RegionalPalette {
    uint64_t begin, width, offset;
    std::size_t group;
};
std::optional<std::vector<RegionalPalette>> regionalPalettes(
    const detail::BoundedLifetimeAllocationAssignment& assignment, RegionalAllocationSummary& out)
{
    using Direction = std::pair<uint32_t, uint32_t>;
    std::map<Direction, std::size_t> positions;
    std::vector<RegionalPalette> phases;
    for (std::size_t index = 0; index < assignment.palettes.size(); ++index) {
        const auto& palette = assignment.palettes[index];
        uint64_t begin = 0, width = palette.gap, offset = 0;
        if (assignment.shared) {
            const auto& phase = assignment.shared->phases[index];
            begin = phase.laneBegin; width = phase.laneCount; offset = phase.offset;
        }
        // Hardware allocation remains allocation-only. Never expand a wide
        // encoded palette to try to make it fit the six-ID pool.
        if (!width || begin > 6 || width > 6 - begin) { return {}; }
        std::size_t group = out.groups.size();
        if (assignment.shared) {
            auto inserted = positions.emplace(Direction{palette.source, palette.target}, group);
            group = inserted.first->second;
        }
        if (group == out.groups.size()) {
            out.groups.push_back({palette.source, palette.target, 0, {}});
        }
        out.groups[group].budget = std::max(out.groups[group].budget, begin + width);
        phases.push_back({begin, width, offset, group});
    }
    for (auto& group : out.groups) { group.lanes.resize(group.budget); }
    return phases;
}
bool appendRegionalMember(RegionExpressions& e, const LifetimeWindowInput& w,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> demands,
    RegionExpressions::Id trips, uint32_t record, bool inactive, const RegionalPalette& phase,
    RegionalAllocationGroup& group, const detail::BoundedAllocationPresence& boundary)
{
    const auto& edge = demands[record];
    const uint32_t target = edge.target % w.sites;
    if (inactive) {
        RegionalAllocationMember member{record, 0, 0,
            {edge.source, e.constant(0), PeriodicEventKind::Start},
            {target, e.constant(0), PeriodicEventKind::Completion}, e.boolean(false)};
        member.tupleRule = PhysicalTupleRule{1, phase.begin, {{0, 1, phase.offset, phase.width, 1}}};
        group.members.push_back(std::move(member));
        return true;
    }
    const auto active = e.lt(e.constant(edge.target / w.sites), trips);
    // Keep inactive coordinates defined even for zero trips. Active implies
    // that both mandatory envelope occurrences actually execute.
    const auto finalVisit = e.sub(e.select(active, trips, e.constant(1)), e.constant(1));
    auto first = certifiedEnvelope(e, w, unconditional, edge.source, true, e.constant(0), active, boundary);
    auto last = certifiedEnvelope(e, w, unconditional, target, false, finalVisit, active, boundary);
    if (!first || !last) { return false; }
    RegionalAllocationMember member{record, 0, 0,
        {*first, e.constant(0), PeriodicEventKind::Start},
        {*last, finalVisit, PeriodicEventKind::Completion}, active};
    member.tupleRule = PhysicalTupleRule{1, phase.begin, {{0, 1, phase.offset, phase.width, 1}}};
    group.members.push_back(member);
    // Each lane's actual handoffs are contained in this envelope, even when
    // opaque guards suppress arbitrary visits. Broader bounds only prevent
    // an optimization; they cannot authorize otherwise overlapping reuse.
    for (uint64_t lane = phase.begin; lane < phase.begin + phase.width; ++lane) {
        group.lanes[lane].firstSources.push_back({member.firstSource, active});
        group.lanes[lane].lastTargets.push_back({member.lastTarget, active});
    }
    return true;
}
} // namespace
std::shared_ptr<RegionalAllocationSummary> detail::boundedLifetimeRegionalAllocationWithPresence(
    RegionExpressions& producerExpressions, RegionExpressions& outputExpressions,
    const LifetimeWindowInput& w, llvm::ArrayRef<uint8_t> unconditional,
    llvm::ArrayRef<GuardedRankEdge> demands, RegionExpressions::Id trips,
    const BoundedAllocationPresence& boundary)
{
    if (trips >= outputExpressions.size() || outputExpressions.isBoolean(trips) ||
        demands.size() > UINT32_MAX || !outputExpressions.constructionError().empty() ||
        !producerExpressions.constructionError().empty()) { return {}; }
    if (boundary.presence && boundary.expressions.get() != &outputExpressions) { return {}; }
    RegionExpressions::Transaction transaction(outputExpressions);
    auto assignment = detail::boundedLifetimeAllocationAssignment(producerExpressions, w, unconditional, demands);
    if (!assignment) { return {}; }
    auto result = std::make_shared<RegionalAllocationSummary>();
    auto phases = regionalPalettes(*assignment, *result);
    if (!phases) { return {}; }
    for (std::size_t i = 0; i < assignment->palettes.size(); ++i) {
        for (auto record : assignment->palettes[i].records) {
            if (!appendRegionalMember(outputExpressions, w, unconditional, demands, trips,
                    static_cast<uint32_t>(record), assignment->inactiveRecords[record],
                    (*phases)[i], result->groups[(*phases)[i].group], boundary)) { return {}; }
        }
    }
    if (!outputExpressions.constructionError().empty()) { return {}; }
    transaction.commit();
    return result;
}
std::shared_ptr<RegionalAllocationSummary> boundedLifetimeRegionalAllocation(
    RegionExpressions& producerExpressions, RegionExpressions& outputExpressions,
    const LifetimeWindowInput& window, llvm::ArrayRef<uint8_t> unconditional,
    llvm::ArrayRef<GuardedRankEdge> demands, RegionExpressions::Id trips)
{
    return detail::boundedLifetimeRegionalAllocationWithPresence(producerExpressions,
        outputExpressions, window, unconditional, demands, trips, {});
}
} // namespace mlir::pto::frontiersynch
