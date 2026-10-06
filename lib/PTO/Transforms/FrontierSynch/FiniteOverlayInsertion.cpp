// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FiniteOverlayInsertion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFiniteOverlayInsertion(
    func::FuncOp function, const FiniteOverlayAnalysis& analysis,
    std::string& error)
{
    const auto& base = analysis.base;
    if (!function || !analysis.error.empty() || !analysis.retainBase || !base.prepareFiltered ||
        !base.expressions || analysis.demands.size() != analysis.retained.size() ||
        base.anchors.size() != base.occurrenceLoops.size()) {
        error = "finite overlay lacks filtered base endpoint recipes"; return failure();
    }
    for (const auto& anchor : base.anchors) {
        if (!anchor.phase || !anchor.phase->elementOp ||
            anchor.phase->elementOp->getParentOfType<func::FuncOp>() != function ||
            !anchor.before.before || !anchor.after.before ||
            anchor.before.before->getParentOfType<func::FuncOp>() != function ||
            anchor.after.before->getParentOfType<func::FuncOp>() != function) {
            error = "finite overlay anchors belong to another function or invalid cuts"; return failure();
        }
    }
    auto supplied = base.prepareFiltered(analysis.retainBase);
    if (failed(supplied)) { error = "finite overlay base endpoint refinement failed"; return failure(); }
    auto plan = std::move(*supplied);
    plan->allocationCertificate = {};
    auto& arena = *base.expressions;
    using Expr = RegionExpressions::Id;
    std::map<Operation*, Block*> blocks;
    std::map<Operation*, RegionExpressions::CutEmission> contexts;
    auto emit = [&](Expr value, Operation* cut) -> FailureOr<Value> {
        if (!cut) { return failure(); }
        auto found = blocks.find(cut);
        if (found == blocks.end()) { found = blocks.emplace(cut, &plan->addPreparation(cut)).first; }
        OpBuilder builder(function.getContext());
        builder.setInsertionPointToEnd(found->second);
        return arena.emitContextual(value, builder, cut, contexts[cut]);
    };
    uint64_t nextRecord = 0;
    for (const auto& family : plan->families) {
        nextRecord = std::max(nextRecord, uint64_t(family.id) + 1);
        for (const auto& member : family.members) { nextRecord = std::max(nextRecord, uint64_t(member.record) + 1); }
    }
    if (plan->families.empty() && !plan->endpoints.empty()) {
        error = "finite overlay base has no endpoint provenance"; return failure();
    }
    if (!plan->independentPieces) {
        for (auto& endpoint : plan->endpoints) {
            if (endpoint.record < 0 || static_cast<uint64_t>(endpoint.record) >= nextRecord) {
                error = "finite overlay base has invalid record identity"; return failure();
            }
            if (!endpoint.memberCoordinates.empty()) {
                error = "finite overlay needs a filtered base with ungrouped or independent recipes"; return failure();
            }
            endpoint.records = {static_cast<uint32_t>(endpoint.record)};
            if (endpoint.kind != LogicalCommandKind::Barrier) {
                auto member = emit(arena.constant(endpoint.record), endpoint.before);
                if (failed(member)) { return failure(); }
                endpoint.memberCoordinates.push_back(*member);
            }
        }
    }
    plan->groupedFamilies = true;
    plan->independentPieces = true;
    auto cutGuard = [&](RegionalEvent event, Expr retained) -> std::optional<Expr> {
        if (event.type >= base.anchors.size()) { return std::nullopt; }
        const auto& anchor = base.anchors[event.type];
        auto loop = base.occurrenceLoops[event.type];
        if (loop) {
            APInt step;
            if (!matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isSignedIntN(64) ||
                step.getSExtValue() <= 0 || !loop.getInductionVar().getType().isIndex()) { return std::nullopt; }
            auto current = arena.div(arena.sub(arena.input(loop.getInductionVar()),
                arena.input(loop.getLowerBound())), arena.constant(step.getSExtValue()));
            retained = arena.land(retained, arena.eq(current, event.ordinal));
        } else { retained = arena.land(retained, arena.eq(event.ordinal, arena.constant(0))); }
        for (auto coordinate : anchor.coordinates) {
            retained = arena.land(retained, arena.eq(arena.input(coordinate.loop.getInductionVar()),
                                                      arena.constant(coordinate.induction)));
        }
        return retained;
    };
    for (std::size_t i = 0; i < analysis.demands.size(); ++i) {
        auto retained = analysis.retained[i];
        if (arena.constantValue(retained) == 0) { continue; }
        const auto& demand = analysis.demands[i];
        if (demand.source.type >= base.anchors.size() || demand.target.type >= base.anchors.size() ||
            nextRecord >= UINT32_MAX) { error = "finite overlay endpoint identity overflow"; return failure(); }
        const auto& a = base.anchors[demand.source.type]; const auto& b = base.anchors[demand.target.type];
        if (!a.phase || !b.phase) { error = "finite overlay has no payload anchors"; return failure(); }
        auto p = static_cast<uint32_t>(a.phase->kPipeValue), q = static_cast<uint32_t>(b.phase->kPipeValue);
        auto record = static_cast<uint32_t>(nextRecord++);
        EndpointFamily family;
        family.id = record; family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
        family.sourceCut = a.after; family.targetCut = b.before;
        family.members.push_back({record, demand.source.type, demand.target.type, a.coordinates, b.coordinates});
        plan->families.push_back(std::move(family));
        for (bool publish : {true, false}) {
            if (publish && p == q) { continue; }
            auto* cut = publish ? a.after.before : b.before.before;
            auto expression = cutGuard(publish ? demand.source : demand.target, retained);
            if (!expression) { error = "finite overlay endpoint coordinate unsupported"; return failure(); }
            auto guard = emit(*expression, cut);
            if (failed(guard)) { error = "finite overlay endpoint guard unavailable"; return failure(); }
            auto kind = p == q ? LogicalCommandKind::Barrier :
                        publish ? LogicalCommandKind::Set : LogicalCommandKind::Wait;
            PreparedLogicalEndpoint endpoint{cut, kind, p, q, record, *guard, {}};
            endpoint.records.push_back(record);
            if (p != q) {
                auto identity = emit(arena.constant(0), cut), member = emit(arena.constant(record), cut);
                if (failed(identity) || failed(member)) { return failure(); }
                endpoint.identity = *identity; endpoint.memberCoordinates.push_back(*member);
            }
            plan->endpoints.push_back(std::move(endpoint));
        }
    }
    using Namespace = std::tuple<uint32_t, uint32_t, uint64_t>;
    std::map<Namespace, uint32_t> namespaces;
    std::map<uint32_t, Namespace> records;
    for (const auto& family : plan->families) {
        Namespace key{family.sourcePipe, family.targetPipe, family.displacement};
        for (const auto& member : family.members) {
            auto [it, inserted] = namespaces.emplace(key, member.record);
            it->second = std::min(it->second, member.record); records.emplace(member.record, key);
        }
    }
    int64_t piece = 0;
    for (auto& endpoint : plan->endpoints) {
        if (endpoint.records.empty() || !records.count(endpoint.records.front())) {
            error = "finite overlay endpoint lacks record provenance"; return failure();
        }
        endpoint.record = namespaces.at(records.at(endpoint.records.front())); endpoint.piece = piece++;
    }
    if (!arena.constructionError().empty()) { error = arena.constructionError(); return failure(); }
    return plan;
}
RegionalAnalysis finiteOverlayRegionalResult(func::FuncOp function,
    const FiniteOverlayAnalysis& analysis)
{
    auto result = analysis.regional;
    if (!function || !analysis.error.empty() || !analysis.base.prepareFiltered || !analysis.retainBase) {
        return result;
    }
    auto owned = std::make_shared<FiniteOverlayAnalysis>(analysis);
    result.prepare = [owned, function]() -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        std::string error;
        auto plan = prepareFiniteOverlayInsertion(function, *owned, error);
        if (succeeded(plan)) { (*plan)->completeInvocation = false; }
        return plan;
    };
    result.capabilities.endpointRecipes = true;
    return result;
}
} // namespace mlir::pto::frontiersynch
