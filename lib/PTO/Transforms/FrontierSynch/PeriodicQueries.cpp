// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Symbolic entry/exit queries for a qualified stationary period, not unfolding.
#include "RegionalRequests.h"
#include "mlir/IR/Matchers.h"
namespace mlir::pto::frontiersynch {
LogicalResult liftPeriodicQueries(SelectedAnalysis& selected, const SignedInputs& inputs, std::string& reason)
{
    using Int = llvm::DynamicAPInt;
    using Kind = PeriodicEventKind;
    auto space = inputs.context->space();
    APInt step;
    if (!selected.loop || !matchPattern(selected.loop.getStep(), m_ConstantInt(&step)) || !step.isOne() ||
        space->period() != Int(1)) {
        reason = "periodic symbolic queries require qualified unit progression and unit signed period";
        return failure();
    }
    auto schema = space->schema();
    if (selected.sites.size() != selected.periodic.sites().size()) {
        reason = "periodic symbolic query site-count identity mismatch"; return failure();
    }
    for (auto [position, site] : llvm::enumerate(selected.sites)) {
        if (site >= schema->sites().size() || schema->sites()[site].phase != selected.periodic.sites()[position] ||
            schema->sites()[site].coordinates.size() != 1 ||
            schema->sites()[site].coordinates.front() != selected.loop.getInductionVar()) {
            reason = "periodic symbolic query coordinate is not the sole original loop induction value";
            return failure();
        }
    }
    SmallVector<SignedPiece> queries, demands, present;
    auto piece = [&](std::size_t a, std::size_t b, Kind source, Kind target, const Int& distance, bool exact) {
        SignedPiece value;
        value.domain = {selected.sites[a], source};
        value.range = {selected.sites[b], target};
        value.residues.assign(2 + schema->parameters().size(), Int(0));
        // Original IV difference equals executed ordinal difference for unit progression.
        SignedAtom lower;
        lower.terms = {{{SignedAxisRole::Domain, 0}, 1}, {{SignedAxisRole::Range, 0}, -1}};
        lower.bound = -distance;
        value.atoms.push_back(lower);
        if (exact) {
            SignedAtom upper;
            upper.terms = {{{SignedAxisRole::Range, 0}, 1}, {{SignedAxisRole::Domain, 0}, -1}};
            upper.bound = distance;
            value.atoms.push_back(upper);
        }
        return value;
    };
    for (std::size_t a = 0; a < selected.sites.size(); ++a) {
        for (std::size_t b = 0; b < selected.sites.size(); ++b) {
            for (auto target : {Kind::Start, Kind::Completion}) {
                auto threshold = selected.periodic.threshold(a, b, target);
                if (failed(threshold)) { reason = "periodic threshold unavailable"; return failure(); }
                if (*threshold) { queries.push_back(piece(a, b, Kind::Completion, target, **threshold, false)); }
                auto origin = selected.periodic.threshold(a, Kind::Start, b, target);
                if (failed(origin)) {
                    reason = "periodic start threshold unavailable";
                    return failure();
                }
                auto start = *origin;
                if (start) { queries.push_back(piece(a, b, Kind::Start, target, *start, false)); }
            }
        }
    }
    for (auto id : selected.periodic.retained()) {
        const auto& edge = selected.periodic.generators()[id].edge;
        demands.push_back(piece(edge.source, edge.consumer, Kind::Completion, Kind::Start, edge.distance, true));
    }
    for (const auto& occurrence : inputs.present->pieces()) {
        for (auto kind : {Kind::Start, Kind::Completion}) {
            auto event = occurrence;
            event.range.kind = kind;
            present.push_back(std::move(event));
        }
    }
    auto events = SignedRelation::import(space, SymbolicTuple::Unit, SymbolicTuple::Event, present);
    auto restrict = [&](ArrayRef<SignedPiece> pieces) -> SignedRelationHandle {
        auto relation = SignedRelation::import(space, SymbolicTuple::Event, SymbolicTuple::Event, pieces);
        if (!events.succeeded() || !relation.succeeded()) { return {}; }
        auto context = relation.value->restrictContext(inputs.context);
        if (!context.succeeded()) { return {}; }
        auto domain = context.value->restrictDomain(events.value);
        if (!domain.succeeded()) { return {}; }
        auto range = domain.value->restrictRange(events.value);
        return range.succeeded() ? range.value : SignedRelationHandle{};
    };
    auto reach = restrict(queries), minimum = restrict(demands);
    if (!reach || !minimum) { reason = "periodic symbolic presence/context restriction failed"; return failure(); }
    selected.context = inputs.context;
    selected.reachability = reach;
    selected.minimum = minimum;
    selected.contract.interfaces |= interfaceBit(DemandInterface::RegionQueries) |
                                    interfaceBit(DemandInterface::UniformMembership);
    return success();
}
} // namespace mlir::pto::frontiersynch
