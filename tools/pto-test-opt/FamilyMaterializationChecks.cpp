// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Construction-size and staged ownership checks for shared family lowering.
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/MixedFamilyInsertion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
bool runFamilyMaterializationChecks(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    OpBuilder builder(context);
    const auto location = builder.getUnknownLoc();
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(location, "shared_family_cache",
        builder.getFunctionType({builder.getIndexType()}, {})));
    auto* body = function->addEntryBlock();
    builder.setInsertionPointToEnd(body);
    SmallVector<Operation*> cuts;
    for (unsigned i = 0; i < 64; ++i) {
        cuts.push_back(builder.create<arith::ConstantIndexOp>(location, i).getOperation());
    }
    builder.create<func::ReturnOp>(location);
    auto arena = std::make_shared<fs::RegionExpressions>();
    const auto value = arena->input(body->getArgument(0));
    SmallVector<fs::ContextualSharedExpression> shared;
    for (unsigned i = 0; i < 512; ++i) {
        shared.push_back({arena->eq(value, arena->constant(i)), cuts.front()});
    }
    SmallVector<fs::ContextualEndpointFamily> endpoints;
    for (unsigned i = 0; i < cuts.size(); ++i) {
        fs::ContextualEndpointFamily endpoint;
        endpoint.piece.cut = {cuts[i]->getBlock(), cuts[i]};
        endpoint.piece.kind = fs::EndpointKind::Barrier;
        endpoint.piece.sourcePipe = 1; endpoint.piece.targetPipe = 1;
        endpoint.piece.members.push_back({i, {}});
        endpoint.activity = shared[i].expression;
        endpoints.push_back(endpoint);
    }
    auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints, shared);
    if (failed(recipe)) { return false; }
    auto forms = fs::familyEndpointForms(**recipe);
    if (failed(forms) || forms->sharedRoots != shared.size() ||
        forms->sharedLookups != cuts.size() || forms->sharedCandidates != cuts.size() ||
        forms->sharedBindings != cuts.size()) { return false; }
    // Reusing a recipe does not repeat its construction. A mixed leaf carries
    // the construction work of its retained contextual part exactly once.
    auto twice = fs::appendFamilyEndpointRecipe(*recipe, *recipe, 64, 0);
    if (failed(twice)) { return false; }
    auto repeatedForms = fs::familyEndpointForms(**twice);
    if (failed(repeatedForms) || repeatedForms->contextualSites != 2 * cuts.size() ||
        repeatedForms->sharedBindings != forms->sharedBindings ||
        repeatedForms->sharedLookups != forms->sharedLookups) { return false; }
    fs::ContextualFamilyExpressions expressions(arena);
    SmallVector<uint8_t> residual(endpoints.size(), 0);
    auto mixed = fs::buildMixedFamilyEndpointRecipe(expressions, endpoints, residual, {}, shared);
    if (failed(mixed)) { return false; }
    auto mixedForms = fs::familyEndpointForms(**mixed);
    if (failed(mixedForms) || mixedForms->sharedBindings != forms->sharedBindings ||
        mixedForms->sharedLookups != forms->sharedLookups ||
        mixedForms->sharedCandidates != forms->sharedCandidates) { return false; }
    fs::PreparedLogicalPlan plan(0);
    plan.familyRecipe = *recipe;
    if (failed(fs::materializeLogicalFamilies(plan)) || plan.endpoints.size() != cuts.size()) { return false; }
    // The shared source memo is indexed once. Each other command imports just
    // its selected predicate, not the 512 predicates and their dependencies.
    if (plan.familyWork.contexts != cuts.size() || plan.familyWork.sharedImports != cuts.size() - 1 ||
        plan.familyWork.sharedRequests != cuts.size() - 1 ||
        plan.familyWork.sharedCandidates != cuts.size() - 1 ||
        plan.familyWork.localEntries != plan.familyWork.sharedPublished + cuts.size() - 1) { return false; }
    fs::PreparedLogicalPlan retry(0);
    retry.familyRecipe = *recipe;
    if (failed(fs::materializeLogicalFamilies(retry)) ||
        retry.endpoints.front().guard == plan.endpoints.front().guard ||
        retry.familyWork.localEntries != plan.familyWork.localEntries) { return false; }
    // A value prepared at a later original cut cannot seed an earlier cut,
    // even though all shared roots are materialized first in detached blocks.
    SmallVector<fs::ContextualEndpointFamily> reversed{endpoints.front(), endpoints.back()};
    reversed.back().activity = reversed.front().activity;
    auto late = fs::buildContextualFamilyEndpointRecipe(arena, reversed,
        {{reversed.front().activity, cuts.back()}});
    if (failed(late)) { return false; }
    fs::PreparedLogicalPlan ordered(0);
    ordered.familyRecipe = *late;
    if (failed(fs::materializeLogicalFamilies(ordered)) || ordered.familyWork.sharedImports != 0 ||
        ordered.endpoints.front().guard == ordered.endpoints.back().guard) { return false; }
    // Self-contained leaves and wrappers share one cut context across the
    // complete traversal, while preserving the detached dependency order.
    auto leaf = fs::buildContextualFamilyEndpointRecipe(arena, {endpoints.front()});
    if (failed(leaf)) { return false; }
    auto masked = fs::maskFamilyActivity(*leaf, arena, endpoints.front().activity);
    if (failed(masked)) { return false; }
    fs::PreparedLogicalPlan composed(0);
    for (unsigned i = 0; i < 32; ++i) {
        auto joined = fs::appendFamilyEndpointRecipe(composed.familyRecipe, *masked, i, 0);
        if (failed(joined)) { return false; }
        composed.familyRecipe = *joined;
        fs::EndpointFamily family;
        family.sourcePipe = 1; family.targetPipe = 1; family.local = true;
        family.members.push_back({i, 0, 0, {}, {}});
        composed.families.push_back(std::move(family));
    }
    if (failed(fs::materializeLogicalFamilies(composed)) || composed.familyWork.contexts != 1) { return false; }
    Value predicate;
    llvm::DenseSet<Value> available;
    for (const auto& stage : composed.preparation) {
        for (auto& operation : *stage.code) {
            for (Value operand : operation.getOperands()) {
                auto* definition = operand.getDefiningOp();
                if (definition && !definition->getParentOp() && !available.contains(operand)) { return false; }
            }
            for (Value result : operation.getResults()) { available.insert(result); }
        }
    }
    for (const auto& endpoint : composed.endpoints) {
        auto mask = endpoint.guard.getDefiningOp<arith::SelectOp>();
        if (!mask || mask.getCondition() != mask.getTrueValue()) { return false; }
        if (predicate && predicate != mask.getCondition()) { return false; }
        predicate = mask.getCondition();
    }
    // Binding 128 wrappers must not rediscover the entire child chain for
    // each wrapper. Availability still runs at every applicable original cut.
    auto mixedEndpoints = endpoints;
    for (unsigned i = 0; i < mixedEndpoints.size(); i += 2) {
        mixedEndpoints[i].piece.kind = fs::EndpointKind::Set;
        mixedEndpoints[i].piece.targetPipe = 2;
        mixedEndpoints[i].ordinal = arena->constant(0);
    }
    auto nested = fs::buildContextualFamilyEndpointRecipe(arena, mixedEndpoints);
    if (failed(nested)) { return false; }
    constexpr unsigned depth = 128;
    for (unsigned i = 0; i < depth; ++i) {
        nested = i % 2 ? fs::prefixFamilyInvocation(*nested, arena, arena->constant(i)) :
                         fs::maskFamilyActivity(*nested, arena, endpoints.front().activity);
        if (failed(nested)) { return false; }
    }
    // A cached cut inventory is not an availability certificate for a new
    // expression. An unresolved invocation input must still be rejected.
    Block foreign;
    auto unavailable = foreign.addArgument(builder.getI1Type(), location);
    if (succeeded(fs::maskFamilyActivity(*nested, arena, arena->input(unavailable)))) { return false; }
    auto nestedForms = fs::familyEndpointForms(**nested);
    const auto bindings = depth / 2 * (cuts.size() + cuts.size() / 2);
    if (failed(nestedForms) || nestedForms->wrapperDiscoveryNodes != 1 ||
        nestedForms->wrapperDiscoveryPieces != cuts.size() || nestedForms->wrapperIndexImports != depth - 1 ||
        nestedForms->wrapperIndexEntries != 0 || nestedForms->wrapperBindingRequests != bindings) { return false; }
    auto arities = fs::familyMatchingArities(**nested);
    if (failed(arities) || arities->size() != cuts.size() / 2) { return false; }
    for (const auto& [record, arity] : *arities) {
        if (record % 2 || arity != depth / 2 + 1) { return false; }
    }
    // Appending a new cut must form a new union; the old wrapper index cannot
    // hide it. This also exercises imports from an already indexed subtree.
    auto extra = endpoints.front();
    extra.piece.cut = {body, body->getTerminator()};
    auto tail = fs::buildContextualFamilyEndpointRecipe(arena, {extra});
    if (failed(tail)) { return false; }
    auto extended = fs::appendFamilyEndpointRecipe(*nested, *tail, 64, 0);
    if (failed(extended)) { return false; }
    auto outer = fs::maskFamilyActivity(*extended, arena, endpoints.front().activity);
    if (failed(outer)) { return false; }
    auto outerForms = fs::familyEndpointForms(**outer);
    if (failed(outerForms) || outerForms->wrapperDiscoveryNodes != 4 ||
        outerForms->wrapperDiscoveryPieces != cuts.size() + 1 ||
        outerForms->wrapperIndexEntries != cuts.size() ||
        outerForms->wrapperBindingRequests != bindings + cuts.size() + 1) { return false; }
    return composed.endpoints.size() == 32;
}

bool runSparseSharedBindingChecks(mlir::Operation* cut)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    using Arena = fs::RegionExpressions;
    Arena arena;
    llvm::DenseMap<Arena::Id, Value> published;
    fs::PreparedLogicalPlan plan(0);
    OpBuilder builder(cut->getContext());
    builder.setInsertionPointToEnd(&plan.addPreparation(cut));
    SmallVector<Arena::Id> roots;
    for (unsigned i = 0; i < 512; ++i) {
        auto root = arena.constant(i + 1);
        roots.push_back(root);
        published[root] = builder.create<arith::ConstantIndexOp>(cut->getLoc(), i + 1);
    }
    auto lookup = [&](Arena::Id id) { return published.lookup(id); };
    auto input = [](Value) { return Value{}; };
    Arena::SharedCutBindings bindings{lookup, input};
    for (unsigned i = 0; i < 128; ++i) {
        Arena::CutEmission planning, local;
        auto known = [&](Arena::Id id) { return published.count(id); };
        Arena::SharedCutPlanning planned{known};
        auto recipe = arena.prepareContextual(roots[i], cut, planning, &planned);
        if (planning.plannedValues.size() != 1 || planning.cofactors.size() != 1 ||
            planning.planningSharedRequests != 1) { return false; }
        if (failed(recipe)) { return false; }
        auto& code = plan.addPreparation(cut);
        builder.setInsertionPointToEnd(&code);
        auto result = arena.emitPreparedContextual(**recipe, builder, cut, local, &bindings);
        if (failed(result) || *result != published.lookup(roots[i]) || !code.empty() ||
            local.values.size() != 1 || local.sharedRequests != 1 || local.sharedImports != 1) { return false; }
        Arena::CutEmission missing;
        auto absent = [](Arena::Id) { return Value{}; };
        Arena::SharedCutBindings unavailable{absent, input};
        if (succeeded(arena.emitPreparedContextual(**recipe, builder, cut, missing, &unavailable)) ||
            !code.empty()) { return false; }
    }
    // A required replay-input binding is imported independently of expression
    // IDs. An absent borrowed binding still fails before constructing scalar IR.
    Block placeholders;
    const auto source = cut->getOperand(0);
    const auto placeholder = placeholders.addArgument(source.getType(), cut->getLoc());
    Arena::CutEmission planning, local;
    planning.inputs[placeholder] = source;
    auto recipe = arena.prepareContextual(arena.input(placeholder), cut, planning);
    if (failed(recipe)) { return false; }
    auto borrowedInput = [&](Value original) { return original == placeholder ? source : Value{}; };
    Arena::SharedCutBindings inputBindings{lookup, borrowedInput};
    auto& code = plan.addPreparation(cut);
    builder.setInsertionPointToEnd(&code);
    auto result = arena.emitPreparedContextual(**recipe, builder, cut, local, &inputBindings);
    if (failed(result) || *result != source || !code.empty() || local.inputs.size() != 1 ||
        local.sharedImports != 1) { return false; }
    Arena::CutEmission missing;
    return failed(arena.emitPreparedContextual(**recipe, builder, cut, missing, &bindings)) && code.empty();
}
