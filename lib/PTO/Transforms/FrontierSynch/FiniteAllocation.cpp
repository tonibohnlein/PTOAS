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
    if (!context || plan < 0 || !analysis.error.empty() || !analysis.reduction.error.empty() ||
        analysis.occurrences.size() > UINT32_MAX) { return {}; }
    Builder b(context);
    const auto& r = analysis.reduction;
    const auto n = analysis.occurrences.size();
    if (r.pipeColumns.size() != n || r.localRanks.size() != n || r.startRanks.size() != n) { return {}; }
    std::vector<std::vector<uint32_t>> byPipe(r.pipeLabels.size());
    for (uint32_t i = 0; i < n; ++i) {
        if (r.pipeColumns[i] >= byPipe.size() || r.startRanks[i].size() != byPipe.size() ||
            r.pipeLabels[r.pipeColumns[i]] != analysis.occurrences[i].pipe) { return {}; }
        auto& row = byPipe[r.pipeColumns[i]];
        row.push_back(i);
        if (r.localRanks[i] != row.size()) { return {}; }
    }
    // Local insertion places a barrier before the consumer. For nonadjacent
    // local demands this adds order: retain it in the fixed-plan reuse query.
    std::vector<StorageGenerator> edges;
    std::vector<std::vector<std::size_t>> bySource(n);
    bool strengthened = false;
    for (std::size_t i = 0; i < r.retained.size(); ++i) {
        const auto& edge = r.retained[i];
        if (edge.source >= edge.target || edge.target >= n || i > INT64_MAX) { return {}; }
        if (r.pipeColumns[edge.source] != r.pipeColumns[edge.target]) {
            bySource[edge.source].push_back(i); continue;
        }
        if (r.localRanks[edge.target] < 2) { return {}; }
        auto previous = byPipe[r.pipeColumns[edge.target]][r.localRanks[edge.target] - 2];
        if (previous != edge.source) { edges.push_back({previous, edge.target}); strengthened = true; }
    }
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t p = 0; p < byPipe.size(); ++p) {
            const auto rank = r.startRanks[i][p];
            if (rank > byPipe[p].size() || (rank && byPipe[p][rank - 1] >= i)) { return {}; }
            if (rank && strengthened) { edges.push_back({byPipe[p][rank - 1], i}); }
        }
    }
    auto actual = strengthened ? reduceExplicitDemands(analysis.occurrences, edges) : ExplicitReduction{};
    if (!actual.error.empty()) { return {}; }
    const auto& order = strengthened ? actual : r;
    SmallVector<Attribute> handoffs;
    // Source buckets supply a topological handoff order in O(n+h) work. Keep
    // original record IDs while serializing only O(k) rank data per handoff.
    for (const auto& bucket : bySource) {
        for (auto index : bucket) {
            const auto& edge = r.retained[index];
            SmallVector<int64_t> ranks(order.startRanks[edge.source].begin(), order.startRanks[edge.source].end());
            NamedAttrList attributes(handoff(b, index, analysis.occurrences[edge.source].pipe,
                                             analysis.occurrences[edge.target].pipe, ranks));
            attributes.set("source_position", b.getI64IntegerAttr(edge.source));
            attributes.set("target_position", b.getI64IntegerAttr(edge.target));
            attributes.set("target_rank", b.getI64IntegerAttr(order.localRanks[edge.target]));
            handoffs.push_back(attributes.getDictionary(context));
        }
    }
    SmallVector<int64_t> labels(order.pipeLabels.begin(), order.pipeLabels.end());
    NamedAttrList attributes(certificate(b, plan, "rank-query", handoffs));
    attributes.set("pipe_labels", b.getDenseI64ArrayAttr(labels));
    return attributes.getDictionary(context);
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
    struct Handoff { int64_t record; uint32_t source, target; RegionExpressions::Id active; bool cleanup; };
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
        const bool cleanup = family.targetChoices && !family.targetChoices->loop();
        handoffs.push_back(
            {member.record,s->second,t->second,cleanup ? *sp : arena.land(*sp,*tp),cleanup});
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
            if (x.cleanup) { xy = arena.boolean(false); }
            if (y.cleanup) { yx = arena.boolean(false); }
            if (!arena.implies(arena.land(x.active,y.active),arena.lor(*xy,*yx))) { incompatible.push_back(j); }
        }
        entries.push_back(handoff(b, x.record, static_cast<uint32_t>(region.anchors[x.source].phase->kPipeValue),
                                  static_cast<uint32_t>(region.anchors[x.target].phase->kPipeValue), incompatible));
    }
    return certificate(b,plan.planId,"compatibility",entries);
}
} // namespace mlir::pto::frontiersynch
