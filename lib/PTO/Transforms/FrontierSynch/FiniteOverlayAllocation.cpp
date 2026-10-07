// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FiniteOverlayAllocation.h"
#include "PTO/Transforms/FrontierSynch/RegionalLaneExports.h"
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
struct Record {
    uint32_t source = 0, target = 0;
    uint64_t arity = 0;
};
std::optional<std::map<uint32_t, Record>> recordShapes(const PreparedLogicalPlan& plan)
{
    std::map<uint32_t, Record> records;
    for (const auto& family : plan.families) {
        if (family.local) { continue; }
        for (const auto& member : family.members) {
            if (!records.emplace(member.record, Record{family.sourcePipe, family.targetPipe, 0}).second) {
                return std::nullopt;
            }
        }
    }
    for (const auto& endpoint : plan.endpoints) {
        if (endpoint.kind == LogicalCommandKind::Barrier) { continue; }
        for (auto record : endpoint.records) {
            auto found = records.find(record);
            const auto arity = endpoint.memberCoordinates.size();
            if (found == records.end() || !arity || (found->second.arity && found->second.arity != arity)) {
                return std::nullopt;
            }
            found->second.arity = arity;
        }
    }
    return records;
}
bool normalizeMembers(RegionalAllocationSummary& summary, const std::map<uint32_t, Record>& records)
{
    std::set<uint32_t> covered;
    for (auto& group : summary.groups) {
        if (!group.budget) { return false; }
        for (auto& member : group.members) {
            auto found = records.find(member.record);
            if (found == records.end() || !found->second.arity || !covered.insert(member.record).second ||
                regionalAllocationDirection(group, member) !=
                    std::make_pair(found->second.source, found->second.target)) { return false; }
            if (!member.tupleRule) {
                member.tupleRule = PhysicalTupleRule{found->second.arity, 0,
                    {{0, member.stride, member.phase, group.budget, 1}}};
            }
            if (member.tupleRule->coordinateCount != found->second.arity ||
                !validPhysicalTupleRule(*member.tupleRule, group.budget)) { return false; }
        }
    }
    return covered.size() == records.size();
}
} // namespace
std::shared_ptr<RegionalAllocationSummary> finiteOverlayAllocation(
    const FiniteOverlayAnalysis& analysis, const PreparedLogicalPlan& plan,
    ArrayRef<std::optional<uint32_t>> overlayRecords)
{
    if (!analysis.error.empty() || !analysis.regional.expressions ||
        analysis.demands.size() != analysis.retained.size() || overlayRecords.size() != analysis.demands.size()) {
        return {};
    }
    auto records = recordShapes(plan);
    if (!records) { return {}; }
    auto result = plan.regionalAllocation ? std::make_shared<RegionalAllocationSummary>(*plan.regionalAllocation) :
        std::make_shared<RegionalAllocationSummary>();
    for (std::size_t i = 0; i < overlayRecords.size(); ++i) {
        if (!overlayRecords[i]) { continue; }
        auto found = records->find(*overlayRecords[i]);
        // Local added covers need no notification lane.
        if (found == records->end()) { continue; }
        const auto& demand = analysis.demands[i];
        if (!validRegionalEvent(analysis.regional, demand.source) ||
            !validRegionalEvent(analysis.regional, demand.target)) { return {}; }
        RegionalAllocationGroup group;
        group.sourcePipe = found->second.source; group.targetPipe = found->second.target; group.budget = 1;
        RegionalAllocationMember member;
        member.record = *overlayRecords[i]; member.active = analysis.retained[i]; member.singletonHandoff = true;
        member.firstSource = demand.source; member.firstSource.kind = PeriodicEventKind::Start;
        member.lastTarget = demand.target; member.lastTarget.kind = PeriodicEventKind::Completion;
        member.sourcePipe = group.sourcePipe; member.targetPipe = group.targetPipe;
        member.tupleRule = PhysicalTupleRule{1, 0, {}};
        group.lanes.push_back({{{member.firstSource, member.active}}, {{member.lastTarget, member.active}}});
        group.members.push_back(std::move(member));
        result->groups.push_back(std::move(group));
    }
    if (!normalizeMembers(*result, *records)) { return {}; }
    return result;
}
} // namespace mlir::pto::frontiersynch
