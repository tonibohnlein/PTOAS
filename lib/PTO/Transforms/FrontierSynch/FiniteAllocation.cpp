// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Allocation-only lowering of a certified logical plan. No scarcity repair.
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "mlir/IR/Builders.h"
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
DictionaryAttr certificate(Builder& b, int64_t plan, ArrayRef<Attribute> directions) {
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(1)),
        b.getNamedAttr("kind", b.getStringAttr("finite")),
        b.getNamedAttr("plan", b.getI64IntegerAttr(plan)),
        b.getNamedAttr("directions", b.getArrayAttr(directions))});
}
DictionaryAttr direction(Builder& b, uint32_t p, uint32_t q, StringRef mode,
                         ArrayRef<int64_t> records, ArrayRef<Attribute> evidence) {
    return b.getDictionaryAttr({b.getNamedAttr("source", b.getI64IntegerAttr(p)),
        b.getNamedAttr("target", b.getI64IntegerAttr(q)),
        b.getNamedAttr("mode", b.getStringAttr(mode)),
        b.getNamedAttr("records", b.getDenseI64ArrayAttr(records)),
        b.getNamedAttr("evidence", b.getArrayAttr(evidence))});
}
} // namespace
DictionaryAttr explicitAllocationCertificate(const ExplicitAnalysis& analysis, int64_t plan, MLIRContext* context)
{
    Builder b(context);
    using Key = std::pair<uint32_t, uint32_t>;
    std::map<Key, std::vector<std::size_t>> groups;
    const auto& r = analysis.reduction;
    for (std::size_t i = 0; i < r.retained.size(); ++i) {
        const auto& edge = r.retained[i];
        auto p = analysis.occurrences[edge.source].pipe, q = analysis.occurrences[edge.target].pipe;
        if (p != q) { groups[{p,q}].push_back(i); }
    }
    SmallVector<Attribute> directions;
    for (const auto& [key, indices] : groups) {
        SmallVector<int64_t> records;
        SmallVector<Attribute> releases;
        std::size_t next = 0;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const auto& edge = r.retained[indices[i]];
            next = std::max(next, i+1);
            while (next < indices.size()) {
                auto source = r.retained[indices[next]].source;
                if (r.startRanks[source][r.pipeColumns[edge.target]] >= r.localRanks[edge.target]) { break; }
                ++next;
            }
            records.push_back(indices[i]);
            releases.push_back(b.getI64IntegerAttr(next));
        }
        directions.push_back(direction(b,key.first,key.second,"intervals",records,releases));
    }
    return certificate(b,plan,directions);
}
DictionaryAttr finiteRegionalAllocationCertificate(const RegionalAnalysis& region, const PreparedLogicalPlan& plan)
{
    if (llvm::any_of(region.outerLoops, [](const auto& loops) { return !loops.empty(); }) ||
        region.anchors.empty() || !region.capabilities.exactQueries || !region.expressions ||
        !region.presence || !region.reachability ||
        region.anchors.size() != region.occurrenceLoops.size()) { return {}; }
    auto& arena = *region.expressions;
    using CutPipe = std::pair<Operation*, uint32_t>;
    std::map<CutPipe, uint32_t> starts, finishes;
    for (uint32_t i = 0; i < region.anchors.size(); ++i) {
        const auto& a = region.anchors[i];
        if (!a.phase || !a.before.before || !a.after.before ||
            region.occurrenceLoops[i] || !a.coordinates.empty() ||
            !starts.emplace(CutPipe{a.before.before, static_cast<uint32_t>(a.phase->kPipeValue)},i).second ||
            !finishes.emplace(CutPipe{a.after.before, static_cast<uint32_t>(a.phase->kPipeValue)},i).second) {
            return {};
        }
    }
    struct Handoff { int64_t record; uint32_t source, target; RegionExpressions::Id active; };
    std::map<std::pair<uint32_t,uint32_t>,std::vector<Handoff>> groups;
    auto zero = arena.constant(0);
    for (const auto& family : plan.families) {
        if (family.local) { continue; }
        auto s = finishes.find({family.sourceCut.before, family.sourcePipe});
        auto t = starts.find({family.targetCut.before, family.targetPipe});
        if (s == finishes.end() || t == starts.end() || family.members.size() != 1) { return {}; }
        if (family.sourcePipe != static_cast<uint32_t>(region.anchors[s->second].phase->kPipeValue) ||
            family.targetPipe != static_cast<uint32_t>(region.anchors[t->second].phase->kPipeValue)) { return {}; }
        const auto& member = family.members.front();
        if (!member.sourceCoordinates.empty() || !member.targetCoordinates.empty()) { return {}; }
        auto sp = region.presence({s->second,zero,PeriodicEventKind::Start});
        auto tp = region.presence({t->second,zero,PeriodicEventKind::Start});
        if (!sp || !tp) { return {}; }
        // Endpoint presence overapproximates retention, so proving compatibility
        // under this stronger activation condition is safe but may lose sharing.
        groups[{family.sourcePipe,family.targetPipe}].push_back(
            {member.record,s->second,t->second,arena.land(*sp,*tp)});
    }
    Builder b(plan.endpoints.empty() ? region.anchors.front().before.before->getContext() :
                                     plan.endpoints.front().before->getContext());
    SmallVector<Attribute> directions;
    for (const auto& [key, handoffs] : groups) {
        SmallVector<int64_t> records;
        SmallVector<Attribute> conflicts;
        for (std::size_t i = 0; i < handoffs.size(); ++i) {
            const auto& x = handoffs[i];
            records.push_back(x.record);
            SmallVector<int64_t> incompatible;
            for (std::size_t j = 0; j < i; ++j) {
                const auto& y = handoffs[j];
                auto xy = region.reachability({x.target,zero,PeriodicEventKind::Completion},
                                               {y.source,zero,PeriodicEventKind::Start});
                auto yx = region.reachability({y.target,zero,PeriodicEventKind::Completion},
                                               {x.source,zero,PeriodicEventKind::Start});
                if (!xy || !yx) { return {}; }
                if (!arena.implies(arena.land(x.active,y.active),arena.lor(*xy,*yx))) {
                    incompatible.push_back(j);
                }
            }
            conflicts.push_back(b.getDenseI64ArrayAttr(incompatible));
        }
        directions.push_back(direction(b,key.first,key.second,"compatibility",records,conflicts));
    }
    return certificate(b,plan.planId,directions);
}
} // namespace mlir::pto::frontiersynch
