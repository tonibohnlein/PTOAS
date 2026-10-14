// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Common exact regional contract. All expressions belong to the supplied arena.
#include "RepeatedRegionInternal.h"
#include "OccurrenceBinding.h"
#include "MixedFamilyInsertion.h"
#include "RepeatedAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
namespace mlir::pto::frontiersynch
{
namespace
{
class RepeatedPreparation
{
  public:
    using Id = RegionExpressions::Id;
    explicit RepeatedPreparation(RepeatedRegionState &state)
        : state(state), body(state.body), function(state.function), loop(state.loop), trips(state.trips)
    {
    }
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> run(ArrayRef<scf::ForOp> enclosing)
    {
        SmallVector<scf::ForOp> context(enclosing.begin(), enclosing.end());
        context.push_back(loop);
        auto internal =
            body.prepareWithVisits
                ? body.prepareWithVisits(context)
                : (body.prepare ? body.prepare() : FailureOr<std::unique_ptr<PreparedLogicalPlan>>(failure()));
        if (failed(internal))
        {
            state.error = "repeated body endpoint preparation is unavailable";
            return failure();
        }
        plan = std::move(*internal);
        plan->completeInvocation = false;
        plan->nestedIdentities = true;
        plan->allocationCertificate = {};
        auto childView = takeRegionalAllocationView(*plan, true);
        if (!childView) {
            state.error = "repeated child allocation view is unavailable";
            return failure();
        }
        current = ordinal(loop);
        if (!current)
        {
            state.error = "repeated endpoints require a positive constant index step";
            return failure();
        }
        originalVisit = *current;
        current = e().div(*current, e().constant(state.phaseCount));
        if (failed(prefixInternal()) || failed(addCrossings()))
        {
            return failure();
        }
        const bool unitCrossings = llvm::all_of(state.crossings,
            [](const RepeatedCrossing& crossing) { return crossing.displacement == 1; });
        if (unitCrossings) {
            plan->allocationPreparation = [owner = state.shared_from_this(), childView,
                crossings = std::move(allocationCrossings)](PreparedLogicalPlan& prepared) {
                prepareAllocationSupport(*childView, prepared.allocationProfile);
                auto allocation = childView->regionalAllocation;
                if (!allocation) { allocation = finiteRegionalAllocation(owner->body, *childView); }
                if (allocation) {
                    prepared.regionalAllocation = repeatedRegionalAllocation(
                        *owner, *allocation, crossings, prepared.allocationProfile);
                }
            };
        }
        renameFamilies();
        return std::move(plan);
    }

  private:
    RepeatedRegionState &state;
    RegionalAnalysis &body;
    func::FuncOp function;
    scf::ForOp loop;
    Id trips;
    std::unique_ptr<PreparedLogicalPlan> plan;
    std::optional<Id> current;
    Id originalVisit = RegionExpressions::invalid;
    uint64_t nextRecord = 0;
    std::vector<std::pair<uint32_t, std::size_t>> allocationCrossings;
    std::map<Operation *, Block *> blocks;
    std::map<Operation *, RegionExpressions::CutEmission> memos;
    RegionExpressions &e() { return state.e(); }
    LogicalResult fail(const char *reason)
    {
        state.error = reason;
        return failure();
    }
    FailureOr<Value> emit(Id expression, Operation *cut)
    {
        auto [it, added] = blocks.try_emplace(cut);
        if (added)
        {
            it->second = &plan->addPreparation(cut);
        }
        OpBuilder builder(function.getContext());
        builder.setInsertionPointToEnd(it->second);
        return e().emitContextual(expression, builder, cut, memos[cut]);
    }
    std::optional<Id> ordinal(scf::ForOp coordinate)
    {
        if (!coordinate || !coordinate.getInductionVar().getType().isIndex()) { return std::nullopt; }
        return originalLoopOrdinal(e(), coordinate);
    }
    std::optional<Id> at(const RegionalEvent &event, Id guard)
    {
        if (!validRegionalEvent(body, event) || event.type >= body.anchors.size())
        {
            return std::nullopt;
        }
        if (!state.typePhases.empty()) {
            guard = e().land(guard, e().eq(e().rem(originalVisit, e().constant(state.phaseCount)),
                e().constant(state.typePhases[event.type])));
        }
        // This producer's existing counted-loop contract requires index IVs.
        auto indexLoop = [](scf::ForOp loop) { return !loop || loop.getInductionVar().getType().isIndex(); };
        if (!indexLoop(body.occurrenceLoops[event.type])) { return std::nullopt; }
        if (!body.outerLoops.empty() && !llvm::all_of(body.outerLoops[event.type], indexLoop)) { return std::nullopt; }
        auto conditions = regionalOccurrenceConstraints(body, event, state.error);
        if (failed(conditions)) { return std::nullopt; }
        for (auto condition : *conditions) { guard = e().land(guard, condition); }
        return guard;
    }
    LogicalResult prefixInternal()
    {
        for (const auto &family : plan->families)
        {
            nextRecord = std::max(nextRecord, uint64_t(family.id) + 1);
            for (const auto &member : family.members)
            {
                nextRecord = std::max(nextRecord, uint64_t(member.record) + 1);
            }
        }
        if (plan->familyRecipe) {
            auto prefixed = prefixFamilyInvocation(plan->familyRecipe, body.expressions, *current);
            if (failed(prefixed)) { return fail("repeat visit is unavailable at a child family cut"); }
            plan->familyRecipe = std::move(*prefixed);
        }
        for (auto &endpoint : plan->endpoints)
        {
            if (!plan->independentPieces)
            {
                if (endpoint.record < 0 || static_cast<uint64_t>(endpoint.record) >= nextRecord)
                {
                    return fail("repeated child endpoint has invalid record provenance");
                }
                endpoint.records = {static_cast<uint32_t>(endpoint.record)};
                if (endpoint.kind != LogicalCommandKind::Barrier)
                {
                    if (!endpoint.memberCoordinates.empty())
                    {
                        return fail("ambiguous repeated member coordinate");
                    }
                    auto member = emit(e().constant(endpoint.record), endpoint.before);
                    if (failed(member))
                    {
                        return fail("repeated member identity unavailable");
                    }
                    endpoint.memberCoordinates.push_back(*member);
                }
            }
            endpoint.piece = &endpoint - plan->endpoints.data();
            if (endpoint.kind == LogicalCommandKind::Barrier)
            {
                continue;
            }
            if (endpoint.memberCoordinates.empty())
            {
                return fail("repeated child lacks a canonical member");
            }
            auto visit = emit(*current, endpoint.before);
            if (failed(visit))
            {
                return fail("repeat visit is unavailable at an internal endpoint");
            }
            endpoint.memberCoordinates.insert(endpoint.memberCoordinates.begin() + 1, *visit);
        }
        plan->independentPieces = true;
        plan->groupedFamilies = true;
        return success();
    }
    LogicalResult addCrossings()
    {
        std::vector<uint8_t> qualified;
        auto compact = prepareRepeatedCrossingFamilies(state, *plan, allocationCrossings, nextRecord, qualified);
        if (failed(compact)) { return fail("repeated crossing family preparation failed"); }
        if (*compact) { return success(); }
        std::vector<ContextualEndpointFamily> families;
        SmallVector<uint8_t> numericalEndpoints;
        for (const auto &crossing : state.crossings)
        {
            if (crossing.native || e().constantValue(crossing.guard) == 0)
            {
                continue;
            }
            if (nextRecord >= UINT32_MAX)
            {
                return fail("repeated record identity overflow");
            }
            const auto record = static_cast<uint32_t>(nextRecord++);
            allocationCrossings.emplace_back(record, &crossing - state.crossings.data());
            const auto &a = body.anchors[crossing.source.type];
            const auto &b = body.anchors[crossing.target.type];
            auto p = static_cast<uint32_t>(a.phase->kPipeValue), q = static_cast<uint32_t>(b.phase->kPipeValue);
            EndpointFamily family;
            family.id = record;
            family.sourcePipe = p;
            family.targetPipe = q;
            family.local = p == q;
            family.displacement = crossing.displacement;
            family.sourceCut = a.after;
            family.targetCut = b.before;
            family.members.push_back(
                {record, crossing.source.type, crossing.target.type, a.coordinates, b.coordinates});
            plan->families.push_back(std::move(family));
            for (bool publish : {true, false})
            {
                if (p == q && publish)
                {
                    continue;
                }
                const auto &event = publish ? crossing.source : crossing.target;
                auto cut = publish ? a.after.before : b.before.before;
                const auto distance = e().constant(crossing.displacement);
                // Both guards prove the matching coordinate arithmetic is in
                // range. Its total modular value off-guard is never executed.
                auto boundary = publish ? e().land(e().lt(distance, trips),
                    e().lt(*current, e().sub(trips, distance))) :
                    e().land(e().le(distance, *current), e().lt(*current, trips));
                if (!state.typePhases.empty()) {
                    auto other = publish ? e().add(*current, distance) : e().sub(*current, distance);
                    auto type = publish ? crossing.target.type : crossing.source.type;
                    auto full = e().div(state.originalTrips, e().constant(state.phaseCount));
                    auto tail = e().rem(state.originalTrips, e().constant(state.phaseCount));
                    auto below = [&](Id visit, uint32_t kind) {
                        return e().lor(e().lt(visit, full), e().land(e().eq(visit, full),
                            e().lt(e().constant(state.typePhases[kind]), tail)));
                    };
                    boundary = e().land(boundary, e().land(below(other, type), below(*current, event.type)));
                    if (state.originalBegin != RegionExpressions::invalid) {
                        auto first = e().div(state.originalBegin, e().constant(state.phaseCount));
                        auto head = e().rem(state.originalBegin, e().constant(state.phaseCount));
                        auto above = [&](Id visit, uint32_t kind) {
                            return e().lor(e().lt(first, visit), e().land(e().eq(visit, first),
                                e().le(head, e().constant(state.typePhases[kind]))));
                        };
                        boundary = e().land(boundary, e().land(above(other, type), above(*current, event.type)));
                    }
                }
                auto guard = at(event, e().land(crossing.guard, boundary));
                if (!guard)
                {
                    return fail("repeated crossing coordinate unavailable");
                }
                ContextualEndpointFamily endpoint;
                endpoint.piece.namespaceId = record;
                endpoint.piece.sourcePipe = p; endpoint.piece.targetPipe = q;
                endpoint.piece.displacement = crossing.displacement;
                endpoint.piece.kind = p == q ? EndpointKind::Barrier :
                    (publish ? EndpointKind::Set : EndpointKind::Wait);
                endpoint.piece.cut = {cut->getBlock(), cut};
                endpoint.piece.members.push_back({record, body.anchors[event.type].coordinates});
                endpoint.activity = *guard;
                endpoint.ordinal = publish ? *current : e().sub(*current, distance);
                if (!state.typePhases.empty()) {
                    endpoint.residueCoordinate = originalVisit;
                    endpoint.modulus = state.phaseCount;
                    endpoint.residue = state.typePhases[event.type];
                }
                families.push_back(std::move(endpoint));
                numericalEndpoints.push_back(qualified[&crossing - state.crossings.data()]);
            }
        }
        if (!families.empty()) {
            ContextualFamilyExpressions expressions(body.expressions);
            auto recipe = buildMixedFamilyEndpointRecipe(expressions, families, numericalEndpoints, loop);
            if (failed(recipe)) { recipe = buildContextualFamilyEndpointRecipe(expressions, families); }
            if (failed(recipe)) { return fail("repeated crossing expressions unavailable at their cuts"); }
            auto joined = appendFamilyEndpointRecipe(plan->familyRecipe, *recipe, 0, plan->endpoints.size());
            if (failed(joined)) { return fail("repeated crossing family composition failed"); }
            plan->familyRecipe = std::move(*joined);
        }
        return success();
    }
    void renameFamilies()
    {
        const auto names = familyRecordNamespaces(plan->families);
        for (auto &endpoint : plan->endpoints)
        {
            endpoint.record = names.at(endpoint.records.front());
        }
    }
};
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> RepeatedRegionState::prepare(ArrayRef<scf::ForOp> enclosing)
{
    return RepeatedPreparation(*this).run(enclosing);
}
} // namespace mlir::pto::frontiersynch
