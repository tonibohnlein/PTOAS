// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Common exact regional contract. All expressions belong to the supplied arena.
#include "RepeatedRegionInternal.h"
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
        plan->regionalAllocation.reset();
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
        APInt step;
        if (!matchPattern(coordinate.getStep(), m_ConstantInt(&step)) || !step.isSignedIntN(64) ||
            step.getSExtValue() <= 0 || !coordinate.getInductionVar().getType().isIndex())
        {
            return std::nullopt;
        }
        return e().div(e().sub(e().input(coordinate.getInductionVar()), e().input(coordinate.getLowerBound())),
                       e().constant(step.getSExtValue()));
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
        if (body.endpointEventGuard) {
            auto value = body.endpointEventGuard(event);
            if (!value) { return std::nullopt; }
            guard = e().land(guard, *value);
        }
        auto leaf = body.occurrenceLoops[event.type];
        if (leaf)
        {
            auto value = ordinal(leaf);
            if (!value)
            {
                return std::nullopt;
            }
            guard = e().land(guard, e().eq(*value, event.ordinal));
        }
        if (!body.outerLoops.empty())
        {
            const auto &frame = body.outerLoops[event.type];
            for (std::size_t i = 0; i < frame.size(); ++i)
            {
                auto value = ordinal(frame[i]);
                if (!value)
                {
                    return std::nullopt;
                }
                if (!body.outerDivisors.empty()) {
                    value = e().div(*value, e().constant(body.outerDivisors[event.type][i]));
                }
                guard = e().land(guard, e().eq(*value, event.visits[i]));
            }
        }
        for (auto coordinate : body.anchors[event.type].coordinates)
        {
            guard = e().land(guard,
                             e().eq(e().input(coordinate.loop.getInductionVar()), e().constant(coordinate.induction)));
        }
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
            const auto &a = body.anchors[crossing.source.type];
            const auto &b = body.anchors[crossing.target.type];
            auto p = static_cast<uint32_t>(a.phase->kPipeValue), q = static_cast<uint32_t>(b.phase->kPipeValue);
            EndpointFamily family;
            family.id = record;
            family.sourcePipe = p;
            family.targetPipe = q;
            family.local = p == q;
            family.displacement = 1;
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
                auto boundary =
                    publish ? e().lt(*current, e().sub(trips, e().constant(1))) : e().lt(e().constant(0), *current);
                if (!state.typePhases.empty()) {
                    auto other = publish ? e().add(*current, e().constant(1)) :
                                           e().sub(*current, e().constant(1));
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
                auto predicate = emit(*guard, cut);
                if (failed(predicate))
                {
                    return fail("repeated crossing predicate unavailable at its cut");
                }
                PreparedLogicalEndpoint endpoint{
                    cut,
                    p == q ? LogicalCommandKind::Barrier :
                        (publish ? LogicalCommandKind::Set : LogicalCommandKind::Wait),
                    p,
                    q,
                    record,
                    *predicate,
                    {}};
                endpoint.records = {record};
                endpoint.piece = plan->endpoints.size();
                if (p != q)
                {
                    auto identity = emit(publish ? *current : e().sub(*current, e().constant(1)), cut);
                    auto member = emit(e().constant(record), cut);
                    if (failed(identity) || failed(member))
                    {
                        return fail("repeated matching identity unavailable");
                    }
                    endpoint.identity = *identity;
                    endpoint.memberCoordinates.push_back(*member);
                }
                plan->endpoints.push_back(std::move(endpoint));
            }
        }
        return success();
    }
    void renameFamilies()
    {
        using Key = std::tuple<uint32_t, uint32_t, uint64_t>;
        std::map<Key, uint32_t> names;
        std::map<uint32_t, Key> keys;
        for (const auto &family : plan->families)
        {
            const Key key{family.sourcePipe, family.targetPipe, family.displacement};
            for (const auto &member : family.members)
            {
                auto [entry, added] = names.emplace(key, member.record);
                entry->second = std::min(entry->second, member.record);
                keys.emplace(member.record, key);
            }
        }
        for (auto &endpoint : plan->endpoints)
        {
            endpoint.record = names.at(keys.at(endpoint.records.front()));
        }
    }
};
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> RepeatedRegionState::prepare(ArrayRef<scf::ForOp> enclosing)
{
    return RepeatedPreparation(*this).run(enclosing);
}
} // namespace mlir::pto::frontiersynch
