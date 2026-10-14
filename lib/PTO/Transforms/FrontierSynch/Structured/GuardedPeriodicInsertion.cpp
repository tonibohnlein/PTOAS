// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Structured/GuardedPeriodicInsertion.h"
#include "../Shared/CountedLoop.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
// Retain exact periodic families and entry invariants before scalar lowering.
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
class Preparer {
public:
    Preparer(const GuardedPeriodicEndpointInput& analysis, ArrayRef<Expr> selected,
             PreparedLogicalPlan& plan, std::string& error,
             const RegionalDemandFilter& filter)
        : analysis(analysis), selected(selected), plan(plan), arena(*analysis.expressions),
          expressions(analysis.expressions), error(error), filter(filter) {}
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
        if (failed(share(trips))) { return failure(); }
        for (std::size_t i = 0; i < analysis.generators.size(); ++i) {
            auto retained = selected[i];
            if (arena.implies(retained, arena.boolean(false))) { continue; }
            if (failed(share(retained)) || failed(share(analysis.generators[i].displacement)) ||
                failed(endpoints(i))) { return failure(); }
        }
        if (!families.empty()) {
            const auto namespaces = familyRecordNamespaces(plan.families);
            for (auto& family : families) {
                family.piece.namespaceId = namespaces.at(family.piece.members.front().record);
            }
            auto recipe = buildContextualFamilyEndpointRecipe(expressions, families, shared);
            if (failed(recipe)) { return failure(); }
            plan.familyRecipe = std::move(*recipe);
        }
        plan.groupedFamilies = true;
        plan.independentPieces = true;
        return success();
    }
private:
    const GuardedPeriodicEndpointInput& analysis;
    ArrayRef<Expr> selected;
    PreparedLogicalPlan& plan;
    RegionExpressions& arena;
    ContextualFamilyExpressions expressions;
    Expr trips = RegionExpressions::invalid, ordinal = RegionExpressions::invalid;
    Expr residue = RegionExpressions::invalid;
    std::vector<ContextualEndpointFamily> families;
    std::vector<ContextualSharedExpression> shared;
    std::string& error;
    const RegionalDemandFilter& filter;
    LogicalResult share(Expr expression)
    {
        // Preserve entry availability and sharing, including safe scalar replay.
        // This validates a recipe and retains no generated Values.
        if (failed(expressions.prepare(expression, analysis.loop))) { return failure(); }
        shared.push_back({expression, analysis.loop});
        return success();
    }
    LogicalResult endpoint(std::size_t record, bool publish, bool local)
    {
        const auto& edge = analysis.generators[record];
        auto* operation = analysis.phases[publish ? edge.source : edge.target]->elementOp;
        auto* before = publish ? operation->getNextNode() : operation;
        if (!before) { return failure(); }
        auto retained = selected[record];
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
        ContextualEndpointFamily family;
        family.piece.cut = {before->getBlock(), before};
        family.piece.kind = local ? EndpointKind::Barrier : publish ? EndpointKind::Set : EndpointKind::Wait;
        family.piece.sourcePipe = analysis.payloads[edge.source].pipe;
        family.piece.targetPipe = analysis.payloads[edge.target].pipe;
        family.piece.members.push_back({static_cast<uint32_t>(record), {}});
        family.activity = guardExpression;
        family.ordinal = publish ? ordinal : arena.sub(ordinal, edge.displacement);
        families.push_back(std::move(family));
        return success();
    }
    LogicalResult endpoints(std::size_t record)
    {
        const auto& edge = analysis.generators[record];
        const bool local = analysis.payloads[edge.source].pipe == analysis.payloads[edge.target].pipe;
        auto* source = analysis.phases[edge.source]->elementOp;
        auto* target = analysis.phases[edge.target]->elementOp;
        EndpointFamily family;
        family.id = static_cast<uint32_t>(record);
        family.sourcePipe = analysis.payloads[edge.source].pipe;
        family.targetPipe = analysis.payloads[edge.target].pipe;
        family.local = local;
        family.sourceCut = {source->getBlock(), source->getNextNode()};
        family.targetCut = {target->getBlock(), target};
        family.members.push_back({static_cast<uint32_t>(record), edge.source, edge.target, {}, {}});
        plan.families.push_back(std::move(family));
        if (!local && failed(endpoint(record, true, false))) { return failure(); }
        return endpoint(record, false, local);
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
    if (failed(Preparer(analysis, selected, *plan, error, filter).run())) {
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
