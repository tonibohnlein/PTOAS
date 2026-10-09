// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicInsertion.h"
#include "CountedLoop.h"
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
class Preparer {
public:
    Preparer(func::FuncOp function, const GuardedPeriodicEndpointInput& analysis, ArrayRef<Expr> selected,
             PreparedLogicalPlan& plan, std::string& error,
             const RegionalDemandFilter& filter)
        : function(function), analysis(analysis), selected(selected), plan(plan), arena(*analysis.expressions),
          builder(function.getContext()), error(error), filter(filter) {}
    LogicalResult run()
    {
        auto loop = analysis.loop;
        const auto domain = CountedLoop::get(loop);
        if (!domain) {
            error = "guarded endpoint domain requires a proved counted-loop ordinal and positive constant step";
            return failure();
        }
        trips = domain->trips(arena);
        auto originalOrdinal = domain->ordinal(arena);
        ordinal = arena.div(originalOrdinal, arena.constant(analysis.period));
        residue = arena.rem(originalOrdinal, arena.constant(analysis.period));
        builder.setInsertionPointToEnd(&plan.addPreparation(loop));
        auto count = arena.emitContextual(trips, builder, loop, entry);
        if (failed(count)) { return failure(); }
        tripValue = *count;
        for (std::size_t i = 0; i < analysis.generators.size(); ++i) {
            auto retained = selected[i];
            if (arena.implies(retained, arena.boolean(false))) { continue; }
            auto distance = analysis.generators[i].displacement;
            auto enabledValue = arena.emitContextual(retained, builder, loop, entry);
            auto distanceValue = arena.emitContextual(distance, builder, loop, entry);
            if (failed(enabledValue) || failed(distanceValue)) { return failure(); }
            invariants.push_back({i, *enabledValue, *distanceValue});
        }
        for (const auto& item : invariants) {
            if (failed(endpoints(item))) { return failure(); }
        }
        return success();
    }
private:
    struct Invariant { std::size_t record; Value retained, distance; };
    func::FuncOp function;
    const GuardedPeriodicEndpointInput& analysis;
    ArrayRef<Expr> selected;
    PreparedLogicalPlan& plan;
    RegionExpressions& arena;
    OpBuilder builder;
    RegionExpressions::CutEmission entry;
    Expr trips = RegionExpressions::invalid, ordinal = RegionExpressions::invalid;
    Expr residue = RegionExpressions::invalid;
    Value tripValue;
    std::vector<Invariant> invariants;
    std::string& error;
    const RegionalDemandFilter& filter;
    struct Cut {
        Block* block = nullptr;
        RegionExpressions::CutEmission context;
    };
    std::map<Operation*, Cut> cuts;
    LogicalResult endpoint(const Invariant& item, bool publish, bool local)
    {
        const auto& edge = analysis.generators[item.record];
        auto* operation = analysis.phases[publish ? edge.source : edge.target]->elementOp;
        auto* before = publish ? operation->getNextNode() : operation;
        if (!before) { return failure(); }
        auto& cut = cuts[before];
        if (!cut.block) {
            cut.block = &plan.addPreparation(before);
            // Entry preparation dominates every original body cut. Preserve
            // its computed invariants through contextual branch specialization
            // instead of rebuilding their DAGs independently at each endpoint.
            for (auto [expression, value] : entry.values) {
                cut.context.values[expression] = value;
                cut.context.cofactors[expression] = expression;
            }
        }
        builder.setInsertionPointToEnd(cut.block);
        auto retained = selected[item.record];
        cut.context.values[retained] = item.retained;
        cut.context.values[edge.displacement] = item.distance;
        cut.context.values[trips] = tripValue;
        // At a visited body cut ordinal < trips. Subtract before comparing to
        // avoid an overflowing source ordinal + displacement.
        auto count = [&](uint32_t type) {
            auto r = arena.constant(analysis.residues.empty() ? 0 : analysis.residues[type]);
            auto nonempty = arena.lt(r, trips);
            auto amount = arena.add(arena.div(arena.sub(arena.sub(trips, arena.constant(1)), r),
                                              arena.constant(analysis.period)), arena.constant(1));
            return arena.select(nonempty, amount, arena.constant(0));
        };
        auto available = publish ?
            arena.land(arena.lt(ordinal, count(edge.target)),
                       arena.lt(edge.displacement, arena.sub(count(edge.target), ordinal))) :
            arena.land(arena.le(edge.displacement, ordinal),
                       arena.lt(arena.sub(ordinal, edge.displacement), count(edge.source)));
        const auto selectedType = publish ? edge.source : edge.target;
        auto selectedResidue = arena.constant(analysis.residues.empty() ? 0 : analysis.residues[selectedType]);
        auto guardExpression = arena.land(retained, arena.land(available, arena.eq(residue, selectedResidue)));
        if (filter) {
            auto sourceOrdinal = publish ? ordinal : arena.sub(ordinal, edge.displacement);
            auto targetOrdinal = publish ? arena.add(ordinal, edge.displacement) : ordinal;
            auto condition = filter({edge.source, sourceOrdinal, PeriodicEventKind::Completion},
                                    {edge.target, targetOrdinal, PeriodicEventKind::Start});
            if (!condition || !arena.isBoolean(*condition)) { return failure(); }
            guardExpression = arena.land(guardExpression, *condition);
        }
        // A filter can introduce predicates beyond the loop invariants. Reuse
        // seeded values and the same safe contextual replay rules as regions.
        auto guard = filter ? arena.emitContextual(guardExpression, builder, before, cut.context) :
                              arena.emit(guardExpression, builder, before, cut.context.values);
        auto identity = arena.emit(publish ? ordinal : arena.sub(ordinal, edge.displacement),
                                   builder, before, cut.context.values);
        if (failed(guard) || failed(identity)) { return failure(); }
        auto kind = local ? LogicalCommandKind::Barrier :
                    publish ? LogicalCommandKind::Set : LogicalCommandKind::Wait;
        plan.endpoints.push_back({before, kind, analysis.payloads[edge.source].pipe,
            analysis.payloads[edge.target].pipe, static_cast<int64_t>(item.record), *guard,
            local ? Value() : *identity});
        return success();
    }
    LogicalResult endpoints(const Invariant& item)
    {
        const auto& edge = analysis.generators[item.record];
        const bool local = analysis.payloads[edge.source].pipe == analysis.payloads[edge.target].pipe;
        auto* source = analysis.phases[edge.source]->elementOp;
        auto* target = analysis.phases[edge.target]->elementOp;
        EndpointFamily family;
        family.id = static_cast<uint32_t>(item.record);
        family.sourcePipe = analysis.payloads[edge.source].pipe;
        family.targetPipe = analysis.payloads[edge.target].pipe;
        family.local = local;
        family.sourceCut = {source->getBlock(), source->getNextNode()};
        family.targetCut = {target->getBlock(), target};
        family.members.push_back({static_cast<uint32_t>(item.record), edge.source, edge.target, {}, {}});
        plan.families.push_back(std::move(family));
        if (!local && failed(endpoint(item, true, false))) { return failure(); }
        return endpoint(item, false, local);
    }
};
} // namespace
static FailureOr<std::unique_ptr<PreparedLogicalPlan>> preparePeriodicEndpoints(
    func::FuncOp function, const GuardedPeriodicEndpointInput& analysis, ArrayRef<Expr> selected, std::string& error,
    const RegionalDemandFilter& filter)
{
    if (!analysis.loop || analysis.loop->getParentOfType<func::FuncOp>() != function || !analysis.expressions ||
        !analysis.period || analysis.payloads.size() != analysis.phases.size() ||
        (analysis.residues.empty() ? analysis.period != 1 : analysis.residues.size() != analysis.phases.size()) ||
        analysis.generators.size() != selected.size() ||
        analysis.generators.size() >= static_cast<std::size_t>(UINT32_MAX)) {
        error = "guarded periodic analysis has no valid endpoint contract";
        return failure();
    }
    for (std::size_t i = 0; i < analysis.phases.size(); ++i) {
        if (!analysis.phases[i] || !analysis.phases[i]->elementOp ||
            !analysis.loop->isProperAncestor(analysis.phases[i]->elementOp) ||
            (!analysis.residues.empty() && analysis.residues[i] >= analysis.period)) {
            error = "guarded periodic endpoint has an invalid original cut or residue"; return failure();
        }
    }
    for (const auto& edge : analysis.generators) {
        if (edge.source >= analysis.payloads.size() || edge.target >= analysis.payloads.size()) {
            error = "guarded periodic record has invalid endpoint types"; return failure();
        }
    }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->completeInvocation = !analysis.phases.empty();
    for (auto* phase : analysis.phases) { analysis.expressions->forbidRecomputation(phase->elementOp); }
    if (failed(Preparer(function, analysis, selected, *plan, error, filter).run())) {
        if (error.empty()) {
            error = "guarded periodic endpoint invariants or bounds unavailable: " + analysis.expressions->error();
        }
        return failure();
    }

    return plan;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGuardedPeriodicEndpoints(
    func::FuncOp function, const GuardedPeriodicEndpointInput& input, std::string& error,
    const RegionalDemandFilter& filter)
{
    const bool valid = input.periodic && input.periodic->error.empty() &&
                       input.periodic->expressions == input.expressions;
    if (!valid) { error = "guarded periodic analysis has no valid endpoint contract"; return failure(); }
    return preparePeriodicEndpoints(function, input, input.periodic->retained, error, filter);
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareNumericalPeriodicEndpoints(
    func::FuncOp function, const NumericalPeriodicEndpointInput& input, std::string& error,
    const RegionalDemandFilter& filter)
{
    const bool valid = input.expressions && input.periodic && input.periodic->error.empty();
    if (!valid) { error = "numerical periodic analysis has no valid endpoint contract"; return failure(); }
    auto& e = *input.expressions;
    std::vector<GuardedPeriodicPayload> payloads;
    std::vector<GuardedPeriodicRecord> records;
    std::vector<Expr> selected(input.periodic->generators.size(), e.boolean(false));
    for (const auto& payload : input.periodic->payloads) { payloads.push_back({payload.pipe, e.boolean(true)}); }
    for (const auto& record : input.periodic->generators) {
        records.push_back({record.source, record.target, e.constant(record.displacement),
                           e.boolean(true), record.displacement});
    }
    for (auto index : input.periodic->retained) {
        if (index >= selected.size()) { error = "invalid retained numerical record identity"; return failure(); }
        selected[index] = e.boolean(true);
    }
    GuardedPeriodicEndpointInput domain{input.loop, input.expressions, input.phases, payloads, records,
                                       nullptr, input.period, input.residues};
    return preparePeriodicEndpoints(function, domain, selected, error, filter);
}
} // namespace mlir::pto::frontiersynch
