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
DictionaryAttr certificate(Builder& b, int64_t plan, StringRef mode, ArrayRef<Attribute> handoffs) {
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
        b.getNamedAttr("kind", b.getStringAttr("finite")),
        b.getNamedAttr("plan", b.getI64IntegerAttr(plan)),
        b.getNamedAttr("mode", b.getStringAttr(mode)),
        b.getNamedAttr("handoffs", b.getArrayAttr(handoffs))});
}
DictionaryAttr handoff(Builder& b, int64_t record, uint32_t p, uint32_t q,
                       ArrayRef<int64_t> evidence) {
    return b.getDictionaryAttr({b.getNamedAttr("record", b.getI64IntegerAttr(record)),
        b.getNamedAttr("source", b.getI64IntegerAttr(p)),
        b.getNamedAttr("target", b.getI64IntegerAttr(q)),
        b.getNamedAttr("evidence", b.getDenseI64ArrayAttr(evidence))});
}
} // namespace
DictionaryAttr explicitAllocationCertificate(const ExplicitAnalysis& analysis, int64_t plan, MLIRContext* context)
{
    Builder b(context);
    const auto& r = analysis.reduction;
    // Local insertion places a barrier before the consumer. For nonadjacent
    // local demands this adds order: retain it in the fixed-plan reuse query.
    std::vector<StorageGenerator> edges;
    std::vector<std::vector<uint32_t>> byPipe(r.pipeLabels.size());
    for (uint32_t i = 0; i < analysis.occurrences.size(); ++i) {
        byPipe[r.pipeColumns[i]].push_back(i);
    }
    for (uint32_t i = 0; i < analysis.occurrences.size(); ++i) {
        for (uint32_t p = 0; p < byPipe.size(); ++p) {
            if (auto rank = r.startRanks[i][p]) { edges.push_back({byPipe[p][rank - 1], i}); }
        }
    }
    bool strengthened = false;
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < r.retained.size(); ++i) {
        const auto& edge = r.retained[i];
        if (r.pipeColumns[edge.source] != r.pipeColumns[edge.target]) { indices.push_back(i); continue; }
        auto previous = byPipe[r.pipeColumns[edge.target]][r.localRanks[edge.target] - 2];
        if (previous != edge.source) { edges.push_back({previous, edge.target}); strengthened = true; }
    }
    auto actual = strengthened ? reduceExplicitDemands(analysis.occurrences, edges) : ExplicitReduction{};
    if (!actual.error.empty()) { return {}; }
    const auto& order = strengthened ? actual : r;
    SmallVector<Attribute> handoffs;
    for (auto index : indices) {
        const auto& edge = r.retained[index];
        SmallVector<int64_t> successors;
        for (auto [j, other] : llvm::enumerate(indices)) {
            auto next = r.retained[other].source;
            const bool samePipe = r.pipeColumns[edge.target] == r.pipeColumns[next];
            // WAIT before its consumer precedes SET after that same or a later
            // payload on this pipe, without requiring consumer completion.
            bool reuse = samePipe ? edge.target <= next :
                order.startRanks[next][order.pipeColumns[edge.target]] >= order.localRanks[edge.target];
            if (reuse) { successors.push_back(j); }
        }
        handoffs.push_back(handoff(b, index, analysis.occurrences[edge.source].pipe,
                                  analysis.occurrences[edge.target].pipe, successors));
    }
    return certificate(b, plan, "reuse-order", handoffs);
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
    std::vector<Handoff> handoffs;
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
        handoffs.push_back(
            {member.record,s->second,t->second,arena.land(*sp,*tp)});
    }
    Builder b(plan.endpoints.empty() ? region.anchors.front().before.before->getContext() :
                                     plan.endpoints.front().before->getContext());
    SmallVector<Attribute> entries;
    for (std::size_t i = 0; i < handoffs.size(); ++i) {
        const auto& x = handoffs[i];
        SmallVector<int64_t> incompatible;
        for (std::size_t j = 0; j < i; ++j) {
            const auto& y = handoffs[j];
            auto xy = regionalHandoffReuse(region, {x.target,zero,PeriodicEventKind::Completion},
                                           {y.source,zero,PeriodicEventKind::Start});
            auto yx = regionalHandoffReuse(region, {y.target,zero,PeriodicEventKind::Completion},
                                           {x.source,zero,PeriodicEventKind::Start});
            if (!xy || !yx) { return {}; }
            if (!arena.implies(arena.land(x.active,y.active),arena.lor(*xy,*yx))) { incompatible.push_back(j); }
        }
        entries.push_back(handoff(b, x.record, static_cast<uint32_t>(region.anchors[x.source].phase->kPipeValue),
                                  static_cast<uint32_t>(region.anchors[x.target].phase->kPipeValue), incompatible));
    }
    return certificate(b,plan.planId,"compatibility",entries);
}
} // namespace mlir::pto::frontiersynch
