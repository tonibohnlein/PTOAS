// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact arithmetic crossing adapter over proper original MLIR children. Child
// queries are signed relations, never opaque callbacks treated as formulas.
#include "RegionalRequests.h"
namespace mlir::pto::frontiersynch {
namespace {
using Relation = SignedRelationHandle;
struct Algebra {
    bool valid = true;
    Relation take(SignedResult<Relation> result)
    {
        valid &= result.succeeded();
        return result.value;
    }
    Relation unite(Relation a, Relation b) { return valid ? take(a->unite(b)) : Relation{}; }
    Relation compose(Relation a, Relation b) { return valid ? take(a->compose(b)) : Relation{}; }
    Relation subtract(Relation a, Relation b) { return valid ? take(a->subtract(b)) : Relation{}; }
};
Relation filter(Relation relation, ArrayRef<std::size_t> owners, std::optional<std::size_t> child)
{
    SmallVector<SignedPiece> pieces;
    for (const auto& piece : relation->pieces()) {
        bool accepted =
            child ? ((!piece.domain.site || owners[*piece.domain.site] == *child) &&
                     (!piece.range.site || owners[*piece.range.site] == *child)) :
                    (piece.domain.site && piece.range.site && owners[*piece.domain.site] < owners[*piece.range.site]);
        if (accepted) {
            pieces.push_back(piece);
        }
    }
    auto result = SignedRelation::import(relation->space(), relation->domain(), relation->range(), pieces);
    return result.succeeded() ? result.value : Relation{};
}
} // namespace
FailureOr<SelectedAnalysisHandle> composeEndpointRegions(
    func::FuncOp function, StructuredInputHandle input, const SignedInputs& primitives,
    SmallVectorImpl<AnalysisAttempt>& attempts, std::string& reason, CostLedger& costs,
    RegionalRequests* regional, const AnalysisNeeds& childNeeds, bool childMatching, RegionalRoutePolicy policy)
{
    CostScope merge(costs, CostStage::Merge);
    DenseMap<Operation*, std::size_t> ids;
    SmallVector<Operation*> children;
    SmallVector<std::size_t> owners;
    for (const auto& site : input->sites()) {
        Operation* child = site.anchor;
        while (child->getParentOp() != function) {
            child = child->getParentOp();
        }
        auto [found, added] = ids.try_emplace(child, children.size());
        if (added) {
            children.push_back(child);
        }
        owners.push_back(found->second);
    }
    if (children.size() < 2) {
        attempts.push_back(
            {&function.getBody(), "structured-composition", "not-applicable", "no proper sequence merge",
             merge.nanoseconds()});
        return failure();
    }
    auto space = primitives.context->space();
    auto empty = SignedRelation::import(space, SymbolicTuple::Occurrence, SymbolicTuple::Occurrence, {});
    if (!empty.succeeded()) {
        reason = "empty crossing relation could not be represented";
        return failure();
    }
    SignedInputs nativeInputs = primitives;
    nativeInputs.generators = empty.value;
    nativeInputs.extras = empty.value;
    auto native = [&]() {
        CostScope backend(costs, CostStage::Backend);
        return SignedDemandAnalysis::build(space, nativeInputs);
    }();
    if (!native.succeeded()) {
        reason = signedDiagnostic(native.status).str();
        return failure();
    }
    auto result = std::make_shared<SelectedAnalysis>();
    result->kind = SelectedAnalysis::Kind::Signed;
    result->route = "structured-composition";
    result->structured = input;
    result->context = primitives.context;
    result->native = native.value->native();
    Algebra algebra;
    Relation reach, internal;
    // Request all-event queries from each proper child. The context, storage
    // model, site identities and schema are common to all child results.
    for (std::size_t child = 0; child < children.size(); ++child) {
        CostScope childAttempt(costs, CostStage::Merge);
        SelectedAnalysisHandle childResult;
        if (regional) {
            auto needs = childNeeds;
            needs.interfaces |= interfaceBit(DemandInterface::RegionQueries);
            auto requested = regional->request(children[child], regional->contextFor(children[child]),
                RegionalMode::Modeled, needs, RegionalRepresentation::ArithmeticRelations,
                childMatching ? RegionalPreparation::SelectorMatching : RegionalPreparation::Mathematical, policy);
            childResult = requested.analysis;
            if (!childResult || !childResult->reachability || !childResult->minimum ||
                childResult->reachability->space() != space) {
                reason = requested.obligation.empty() ? "child query uses a different admitted schema" :
                                                       requested.obligation;
                return failure();
            }
            if (childResult->contract.closure == SelectedClosure::SoundUpper) {
                result->contract.closure = SelectedClosure::SoundUpper;
            }
            result->regionalChildren.push_back(childResult);
            reach = reach ? algebra.unite(reach, childResult->reachability) : childResult->reachability;
            internal = internal ? algebra.unite(internal, childResult->minimum) : childResult->minimum;
        } else {
            SignedInputs sliced = primitives;
            sliced.present = filter(primitives.present, owners, child);
            sliced.reference = filter(primitives.reference, owners, child);
            sliced.extras = filter(primitives.extras, owners, child);
            sliced.generators = filter(primitives.generators, owners, child);
            if (!sliced.present || !sliced.reference || !sliced.extras || !sliced.generators) {
                reason = "child relation restriction failed";
                return failure();
            }
            auto built = [&]() {
                CostScope backend(costs, CostStage::Backend);
                return SignedDemandAnalysis::build(space, sliced);
            }();
            if (!built.succeeded()) {
                reason = "child minimum/query obligation: " + signedDiagnostic(built.status).str();
                attempts.push_back(
                    {children[child]->getParentRegion(), "child-integer-octagons", "unmet-obligation", reason,
                     childAttempt.nanoseconds()});
                return failure();
            }
            auto analysis = built.value;
            reach = reach ? algebra.unite(reach, analysis->reachability()) : analysis->reachability();
            internal = internal ? algebra.unite(internal, analysis->minimum()) : analysis->minimum();
        }
        ++result->inspectedRegions;
        attempts.push_back(
            {children[child]->getParentRegion(), regional ? "child-symbolic-queries" : "child-integer-octagons",
             "ready", "", childAttempt.nanoseconds()});
    }
    // The shared analyzer supplies every crossing conflict. Presence and strict
    // reference order qualify it before converting C->I occurrence pairs.
    auto allGenerators = primitives.generators->unite(primitives.extras);
    auto admitted =
        allGenerators.succeeded() ? allGenerators.value->restrictContext(primitives.context) : allGenerators;
    auto ordered = admitted.succeeded() ? admitted.value->intersect(primitives.reference) : admitted;
    auto present = ordered.succeeded() ? ordered.value->restrictDomain(primitives.present) : ordered;
    auto both = present.succeeded() ? present.value->restrictRange(primitives.present) : present;
    if (!both.succeeded()) {
        reason = "crossing occurrence qualification failed";
        return failure();
    }
    SmallVector<SignedPiece> events;
    for (auto piece : both.value->pieces()) {
        piece.domain.kind = PeriodicEventKind::Completion;
        piece.range.kind = PeriodicEventKind::Start;
        events.push_back(std::move(piece));
    }
    auto generators = SignedRelation::import(space, SymbolicTuple::Event, SymbolicTuple::Event, events);
    if (!generators.succeeded()) {
        reason = "crossing event conversion failed";
        return failure();
    }
    auto crossingG = filter(generators.value, owners, std::nullopt);
    auto crossingN = filter(result->native, owners, std::nullopt);
    if (!crossingG || !crossingN) {
        reason = "complete crossing relation could not be represented";
        return failure();
    }
    auto crossing = algebra.unite(crossingG, crossingN);
    // Each crossing advances to a proper later child. Doubling path capacity
    // terminates after log(children) rounds, independent of numeric loop trips.
    for (std::size_t span = 1; span < children.size() && algebra.valid;) {
        reach = algebra.unite(reach, algebra.compose(algebra.compose(reach, crossing), reach));
        span = span > children.size() / 2 ? children.size() : span * 2;
    }
    auto strict = algebra.subtract(reach, native.value->identity());
    auto alternatives = algebra.compose(strict, strict);
    auto retained = algebra.subtract(algebra.subtract(crossingG, result->native), alternatives);
    result->minimum = algebra.unite(internal, retained);
    result->reachability = reach;
    if (!algebra.valid) {
        reason = "exact child-query/crossing algebra failed";
        attempts.push_back(
            {&function.getBody(), "structured-composition", "unmet-obligation", reason, merge.nanoseconds()});
        return failure();
    }
    // The regional caller records Ready only after requested capabilities and
    // selector qualification have succeeded. Internal mathematical readiness
    // must not survive a failed admission of that same parent request.
    if (!regional) {
        attempts.push_back({&function.getBody(), "structured-composition", "ready", "", merge.nanoseconds()});
    }
    return SelectedAnalysisHandle(result);
}
} // namespace mlir::pto::frontiersynch
