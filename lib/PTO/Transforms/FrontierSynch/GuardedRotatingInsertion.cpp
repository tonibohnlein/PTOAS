// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/GuardedRotatingInsertion.h"
#include "PTO/Transforms/FrontierSynch/CompactAllocation.h"
#include "RecognitionInternal.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
std::optional<int64_t> constant(Value value)
{
    APInt number;
    if (!matchPattern(value, m_ConstantInt(&number)) || number.getBitWidth() > 64) { return std::nullopt; }
    return number.getSExtValue();
}
class Preparer {
public:
    Preparer(func::FuncOp function, GuardedRotatingAnalysis& analysis, PreparedLogicalPlan& plan, std::string& error,
             const RegionalDemandFilter& filter)
        : function(function), analysis(analysis), plan(plan), arena(*analysis.expressions),
          builder(function.getContext()), error(error), filter(filter) {}
    LogicalResult run()
    {
        auto loop = analysis.loop;
        auto low = constant(loop.getLowerBound()), step = constant(loop.getStep());
        const auto bits = DataLayout::closest(function).getTypeSizeInBits(builder.getIndexType());
        if (!low || *low < 0 || !step || *step <= 0 || bits.isScalable() || bits.getFixedValue() != 64) {
            error = "guarded endpoint domain requires a nonnegative start, positive step and 64-bit index";
            return failure();
        }
        auto zero = arena.constant(0), one = arena.constant(1);
        auto lower = arena.input(loop.getLowerBound()), upper = arena.input(loop.getUpperBound());
        auto stride = arena.constant(*step);
        auto span = arena.select(arena.slt(lower, upper), arena.sub(upper, lower), zero);
        trips = arena.add(arena.div(span, stride),
                          arena.select(arena.eq(arena.rem(span, stride), zero), zero, one));
        ordinal = arena.div(arena.sub(arena.input(loop.getInductionVar()), lower), stride);
        builder.setInsertionPointToEnd(&plan.addPreparation(loop));
        auto count = arena.emitContextual(trips, builder, loop, entry);
        if (failed(count)) { return failure(); }
        tripValue = *count;
        for (std::size_t i = 0; i < analysis.generators.size(); ++i) {
            auto retained = analysis.periodic.retained[i];
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
    GuardedRotatingAnalysis& analysis;
    PreparedLogicalPlan& plan;
    RegionExpressions& arena;
    OpBuilder builder;
    RegionExpressions::CutEmission entry;
    Expr trips = RegionExpressions::invalid, ordinal = RegionExpressions::invalid;
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
        auto retained = analysis.periodic.retained[item.record];
        cut.context.values[retained] = item.retained;
        cut.context.values[edge.displacement] = item.distance;
        cut.context.values[trips] = tripValue;
        // At a visited body cut ordinal < trips. Subtract before comparing to
        // avoid an overflowing source ordinal + displacement.
        auto available = publish ? arena.lt(edge.displacement, arena.sub(trips, ordinal)) :
                                   arena.le(edge.displacement, ordinal);
        auto guardExpression = arena.land(retained, available);
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
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGuardedRotatingEndpoints(
    func::FuncOp function, GuardedRotatingAnalysis& analysis, std::string& error,
    const RegionalDemandFilter& filter)
{
    if (!analysis.error.empty() || !analysis.loop || !analysis.expressions ||
        analysis.generators.size() != analysis.periodic.retained.size() ||
        analysis.generators.size() >= static_cast<std::size_t>(UINT32_MAX)) {
        error = "guarded rotating analysis has no valid endpoint contract";
        return failure();
    }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->completeInvocation = !analysis.phases.empty();
    for (auto* phase : analysis.phases) { analysis.expressions->forbidRecomputation(phase->elementOp); }
    if (failed(Preparer(function, analysis, *plan, error, filter).run())) {
        if (error.empty()) {
            error = "guarded rotating endpoint invariants or bounds unavailable: " + analysis.expressions->error();
        }
        return failure();
    }
    if (!filter) {
        plan->allocationCertificate = guardedAllocationCertificate(analysis, plan->planId, function.getContext());
    }
    return plan;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGuardedRotatingInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program)
{
    if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) { return failure(); }
    const StructureNode* selected = nullptr;
    for (const auto& node : program.nodes) {
        if (node.kind == StructureKind::Loop && node.anchor->getParentOp() == function) {
            if (selected || !node.guardedRotatingResult ||
                node.guardedRotatingResult->result.state != RecognitionState::Applicable) { return failure(); }
            selected = &node;
        }
    }
    if (!selected) { return failure(); }
    auto loop = dyn_cast<scf::ForOp>(selected->anchor);
    if (!loop || llvm::any_of(input.instructions(), [&](const auto* phase) {
        return !loop->isProperAncestor(phase->elementOp);
    })) { return failure(); }
    PhaseIndex index;
    if (failed(index.build(function, input))) { return failure(); }
    RecognitionResult outside;
    for (auto& operation : function.front()) {
        if (&operation != loop.getOperation()) {
            detail::inspectLeaf(operation, index, outside);
            if (operation.getNumRegions()) { return failure(); }
        }
    }
    if (outside.state != RecognitionState::Applicable) { return failure(); }
    auto analysis = analyzeGuardedRotating(loop, input, *selected->guardedRotatingResult);
    std::string error;
    return prepareGuardedRotatingEndpoints(function, analysis, error);
}
} // namespace mlir::pto::frontiersynch
