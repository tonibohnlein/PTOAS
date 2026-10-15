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
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/FamilyMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "FamilyProtocolChecks.h"
#include "DeferredFamilyChecks.h"
#include "DeferredCleanupChecks.h"
#include "DeferredLocalChecks.h"
#include "FamilyConstructionChecks.h"
#include "FamilyResidueCompactionChecks.h"
#include "FamilyNormalizationConstructionChecks.h"
#include "OwnedSelectorSinkingChecks.h"
namespace {
bool checkCountedSharedEnvironments(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    OpBuilder builder(context);
    const auto loc = builder.getUnknownLoc();
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(loc, "counted_shared_environments",
        builder.getFunctionType({builder.getIndexType(), builder.getI1Type()}, {})));
    auto* body = function->addEntryBlock();
    builder.setInsertionPointToEnd(body);
    auto branch = builder.create<scf::IfOp>(loc, body->getArgument(1), true);
    builder.create<func::ReturnOp>(loc);
    auto* before = branch.getOperation();
    auto* yes = branch.thenBlock()->getTerminator();
    auto* no = branch.elseBlock()->getTerminator();
    auto arena = std::make_shared<fs::RegionExpressions>();
    const auto root = arena->add(arena->input(body->getArgument(0)), arena->constant(7));
    fs::ContextualEndpointFamily endpoint;
    endpoint.piece.cut = {body, before}; endpoint.piece.kind = fs::EndpointKind::Barrier;
    endpoint.piece.sourcePipe = endpoint.piece.targetPipe = 1;
    endpoint.piece.members.push_back({0, {}}); endpoint.activity = arena->boolean(true);
    auto firstOwner = fs::buildContextualFamilyEndpointRecipe(arena, {endpoint});
    auto secondOwner = fs::buildContextualFamilyEndpointRecipe(arena, {endpoint});
    if (failed(firstOwner) || failed(secondOwner)) { return false; }
    // Keep emitted values alive until both environment directories are destroyed.
    std::vector<std::unique_ptr<Block>> blocks;
    fs::detail::ScalarConstruction construction;
    fs::FamilyConstruction counted(construction);
    fs::FamilyMaterialization actual;
    uint64_t operations = 0;
    auto emit = [&](Operation* cut, const fs::FamilyEndpointRecipe* owner, bool bound, bool succeeds) {
        fs::RegionExpressions::CutEmission planning;
        if (bound) { planning.plannedValues.insert(root); }
        auto recipe = arena->prepareContextual(root, cut, planning);
        if (failed(recipe)) { return false; }
        auto predicted = counted.emit(*arena, **recipe, cut, owner);
        blocks.push_back(std::make_unique<Block>());
        builder.setInsertionPointToEnd(blocks.back().get());
        auto value = actual.emit(*arena, **recipe, builder, cut, owner);
        operations += blocks.back()->getOperations().size();
        if (succeeded(predicted) != succeeds || succeeded(value) != succeeds ||
            construction.operations() != operations) { return false; }
        if (succeeds && !bound) {
            counted.bind(*arena, *owner, cut, root, *predicted);
            actual.bind(*arena, *owner, cut, root, *value);
            counted.publish(*arena, *owner); actual.publish(*arena, *owner);
        }
        return true;
    };
    // One dominating publication serves both branch cuts. A distinct owner in
    // the same arena cannot borrow it, and its own then-only publication cannot
    // flow into the else branch. Same-cut reuse remains available.
    return emit(before, firstOwner->get(), false, true) &&
        emit(yes, firstOwner->get(), true, true) && emit(no, firstOwner->get(), true, true) &&
        emit(yes, secondOwner->get(), true, false) && emit(yes, secondOwner->get(), false, true) &&
        emit(yes, secondOwner->get(), true, true) && emit(no, secondOwner->get(), true, false);
}
} // namespace
bool runFamilyMaterializationChecks(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    auto check = [](bool passed, llvm::StringRef stage) {
        if (!passed) { llvm::errs() << "family materialization check failed: " << stage << "\n"; }
        return passed;
    };
    // These fixtures own independent IR. Run every fixture before returning so
    // one failed oracle does not hide later failures and require another build.
    bool passed = true;
    passed &= check(checkCountedSharedEnvironments(context), "shared environments");
    passed &= check(checkNumericalConstructionSchedules(context), "numerical schedules");
    passed &= check(checkResidueCompaction(context), "residue compaction");
    passed &= check(checkScalarConstructionTrace(context), "scalar construction");
    passed &= check(checkNormalizedConstructionParity(context), "normalization parity");
    passed &= check(checkOwnedSelectorBoundaries(context), "owned selector boundaries");
    passed &= check(checkIdenticalSelectorGuardBoundaries(context), "identical guards");
    passed &= check(checkSelectorGuardPartitionCost(context), "guard partition cost");
    passed &= check(checkDifferingSelectorGroups(context), "differing guard groups");
    passed &= check(checkOwnedSelectorRoutes(context), "identical-guard selector routes");
    passed &= check(checkOwnedSelectorRoutes(context, true), "differing-guard selector routes");
    passed &= check(checkFamilyProtocols(context), "family protocols");
    passed &= check(checkDeferredFamilies(context), "deferred families");
    passed &= check(checkDeferredCleanup(context), "deferred cleanup");
    passed &= check(checkDeferredLocalFamilies(context), "deferred local families");
    if (!passed) { return false; }
    auto tailFailure = [](unsigned line) {
        llvm::errs() << "family materialization shared-cache tail failed at line " << line << "\n";
        return false;
    };
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
    if (failed(recipe)) { return tailFailure(__LINE__); }
    auto forms = fs::familyEndpointForms(**recipe);
    if (failed(forms) || forms->sharedRoots != shared.size() ||
        forms->sharedLookups != cuts.size() || forms->sharedCandidates != cuts.size() ||
        forms->sharedBindings != cuts.size()) { return tailFailure(__LINE__); }
    // Reusing a recipe does not repeat its construction. A mixed leaf carries
    // the construction work of its retained contextual part exactly once.
    auto twice = fs::appendFamilyEndpointRecipe(*recipe, *recipe, 64, 0);
    if (failed(twice)) { return tailFailure(__LINE__); }
    auto repeatedForms = fs::familyEndpointForms(**twice);
    if (failed(repeatedForms) || repeatedForms->contextualSites != 2 * cuts.size() ||
        repeatedForms->sharedBindings != forms->sharedBindings ||
        repeatedForms->sharedLookups != forms->sharedLookups) { return tailFailure(__LINE__); }
    fs::ContextualFamilyExpressions expressions(arena);
    SmallVector<uint8_t> residual(endpoints.size(), 0);
    auto mixed = fs::buildMixedFamilyEndpointRecipe(expressions, endpoints, residual, {}, shared);
    if (failed(mixed)) { return tailFailure(__LINE__); }
    auto mixedForms = fs::familyEndpointForms(**mixed);
    if (failed(mixedForms) || mixedForms->sharedBindings != forms->sharedBindings ||
        mixedForms->sharedLookups != forms->sharedLookups ||
        mixedForms->sharedCandidates != forms->sharedCandidates) { return tailFailure(__LINE__); }
    fs::PreparedLogicalPlan plan(0);
    plan.familyRecipe = *recipe;
    if (failed(fs::materializeLogicalFamilies(plan)) || plan.endpoints.size() != cuts.size()) {
        return tailFailure(__LINE__);
    }
    // Construct only the 64 live predicates from the 512-root inventory. Each
    // other command imports its selected predicate from one indexed source memo.
    uint64_t comparisons = 0;
    for (const auto& stage : plan.preparation) {
        stage.code->walk([&](arith::CmpIOp) { ++comparisons; });
    }
    if (comparisons != cuts.size()) { return tailFailure(__LINE__); }
    if (plan.familyWork.contexts != cuts.size() || plan.familyWork.sharedImports != cuts.size() - 1 ||
        plan.familyWork.sharedRequests != cuts.size() - 1 ||
        plan.familyWork.sharedCandidates != cuts.size() - 1 ||
        plan.familyWork.localEntries != plan.familyWork.sharedPublished + cuts.size() - 1) {
        return tailFailure(__LINE__);
    }
    fs::PreparedLogicalPlan retry(0);
    retry.familyRecipe = *recipe;
    if (failed(fs::materializeLogicalFamilies(retry)) ||
        retry.endpoints.front().guard == plan.endpoints.front().guard ||
        retry.familyWork.localEntries != plan.familyWork.localEntries) { return tailFailure(__LINE__); }
    // A value prepared at a later original cut cannot seed an earlier cut,
    // even though all shared roots are materialized first in detached blocks.
    SmallVector<fs::ContextualEndpointFamily> reversed{endpoints.front(), endpoints.back()};
    reversed.back().activity = reversed.front().activity;
    auto late = fs::buildContextualFamilyEndpointRecipe(arena, reversed,
        {{reversed.front().activity, cuts.back()}});
    if (failed(late)) { return tailFailure(__LINE__); }
    fs::PreparedLogicalPlan ordered(0);
    ordered.familyRecipe = *late;
    if (failed(fs::materializeLogicalFamilies(ordered)) || ordered.familyWork.sharedImports != 0 ||
        ordered.endpoints.front().guard == ordered.endpoints.back().guard) { return tailFailure(__LINE__); }
    // Self-contained leaves and wrappers share one cut context across the
    // complete traversal, while preserving the detached dependency order.
    auto leaf = fs::buildContextualFamilyEndpointRecipe(arena, {endpoints.front()});
    if (failed(leaf)) { return tailFailure(__LINE__); }
    auto masked = fs::maskFamilyActivity(*leaf, arena, endpoints.front().activity);
    if (failed(masked)) { return tailFailure(__LINE__); }
    fs::PreparedLogicalPlan composed(0);
    for (unsigned i = 0; i < 32; ++i) {
        auto joined = fs::appendFamilyEndpointRecipe(composed.familyRecipe, *masked, i, 0);
        if (failed(joined)) { return tailFailure(__LINE__); }
        composed.familyRecipe = *joined;
        fs::EndpointFamily family;
        family.sourcePipe = 1; family.targetPipe = 1; family.local = true;
        family.members.push_back({i, 0, 0, {}, {}});
        composed.families.push_back(std::move(family));
    }
    if (failed(fs::materializeLogicalFamilies(composed)) || composed.familyWork.contexts != 1) {
        return tailFailure(__LINE__);
    }
    Value predicate;
    llvm::DenseSet<Value> available;
    for (const auto& stage : composed.preparation) {
        for (auto& operation : *stage.code) {
            for (Value operand : operation.getOperands()) {
                auto* definition = operand.getDefiningOp();
                if (definition && !definition->getParentOp() && !available.contains(operand)) {
                    return tailFailure(__LINE__);
                }
            }
            for (Value result : operation.getResults()) { available.insert(result); }
        }
    }
    for (const auto& endpoint : composed.endpoints) {
        auto mask = endpoint.guard.getDefiningOp<arith::SelectOp>();
        if (!mask || mask.getCondition() != mask.getTrueValue()) { return tailFailure(__LINE__); }
        if (predicate && predicate != mask.getCondition()) { return tailFailure(__LINE__); }
        predicate = mask.getCondition();
    }
    // Binding 128 wrappers must not rediscover the entire child chain for
    // each wrapper. Availability still runs at every applicable original cut.
    auto mixedEndpoints = endpoints;
    for (unsigned i = 0; i < mixedEndpoints.size(); i += 2) {
        mixedEndpoints[i].piece.kind = fs::EndpointKind::Set;
        mixedEndpoints[i].piece.targetPipe = 2;
        mixedEndpoints[i].ordinal = arena->constant(0);
        mixedEndpoints[i].matchingCoordinates = {arena->constant(i + 7), value};
    }
    auto nested = fs::buildContextualFamilyEndpointRecipe(arena, mixedEndpoints);
    if (failed(nested)) { return tailFailure(__LINE__); }
    constexpr unsigned depth = 128;
    for (unsigned i = 0; i < depth; ++i) {
        nested = i % 2 ? fs::prefixFamilyInvocation(*nested, arena, arena->constant(i)) :
                         fs::maskFamilyActivity(*nested, arena, endpoints.front().activity);
        if (failed(nested)) { return tailFailure(__LINE__); }
    }
    // A cached cut inventory is not an availability certificate for a new
    // expression. An unresolved invocation input must still be rejected.
    Block foreign;
    auto unavailable = foreign.addArgument(builder.getI1Type(), location);
    if (succeeded(fs::maskFamilyActivity(*nested, arena, arena->input(unavailable)))) { return tailFailure(__LINE__); }
    auto nestedForms = fs::familyEndpointForms(**nested);
    const auto bindings = depth / 2 * (cuts.size() + cuts.size() / 2);
    if (failed(nestedForms) || nestedForms->wrapperDiscoveryNodes != 1 ||
        nestedForms->wrapperDiscoveryPieces != cuts.size() || nestedForms->wrapperIndexImports != depth - 1 ||
        nestedForms->wrapperIndexEntries != 0 || nestedForms->wrapperBindingRequests != bindings) {
        return tailFailure(__LINE__);
    }
    auto arities = fs::familyMatchingArities(**nested);
    if (failed(arities) || arities->size() != cuts.size() / 2) { return tailFailure(__LINE__); }
    for (const auto& [record, arity] : *arities) {
        if (record % 2 || arity != depth / 2 + 3) { return tailFailure(__LINE__); }
    }
    fs::PreparedLogicalPlan coordinates(0);
    coordinates.familyRecipe = *nested;
    // Wrappers translate record namespaces using the owned original-family
    // directory, including local barriers as well as communication endpoints.
    for (const auto& endpoint : mixedEndpoints) {
        fs::EndpointFamily family;
        family.id = endpoint.piece.members.front().record;
        family.sourcePipe = endpoint.piece.sourcePipe;
        family.targetPipe = endpoint.piece.targetPipe;
        family.displacement = endpoint.piece.displacement;
        family.local = endpoint.piece.kind == fs::EndpointKind::Barrier;
        family.sourceCut = family.targetCut = endpoint.piece.cut;
        family.sourceOrder = family.targetOrder = endpoint.piece.order;
        for (const auto& member : endpoint.piece.members) {
            family.members.push_back({member.record, 0, 0, member.coordinates, member.coordinates});
        }
        coordinates.families.push_back(std::move(family));
    }
    if (!coordinates.preparation.empty() || failed(fs::materializeLogicalFamilies(coordinates)) ||
        coordinates.endpoints.size() != cuts.size()) { return tailFailure(__LINE__); }
    for (unsigned i = 0; i < coordinates.endpoints.size(); i += 2) {
        const auto& tuple = coordinates.endpoints[i].memberCoordinates;
        APInt constant;
        if (tuple.size() != depth / 2 + 3 || tuple.back() != body->getArgument(0) ||
            !matchPattern(tuple[tuple.size() - 2], m_ConstantInt(&constant)) || constant != i + 7) {
            return tailFailure(__LINE__);
        }
        for (unsigned j = 0; j < depth / 2; ++j) {
            if (!matchPattern(tuple[j + 1], m_ConstantInt(&constant)) || constant != depth - 1 - 2 * j) {
                return tailFailure(__LINE__);
            }
        }
    }
    auto invalidCoordinates = endpoints.front();
    invalidCoordinates.matchingCoordinates = {value};
    if (succeeded(fs::buildContextualFamilyEndpointRecipe(arena, {invalidCoordinates}))) {
        return tailFailure(__LINE__);
    }
    invalidCoordinates = mixedEndpoints.front();
    invalidCoordinates.matchingCoordinates = {arena->boolean(true)};
    if (succeeded(fs::buildContextualFamilyEndpointRecipe(arena, {invalidCoordinates}))) {
        return tailFailure(__LINE__);
    }
    // Appending a new cut must form a new union; the old wrapper index cannot
    // hide it. This also exercises imports from an already indexed subtree.
    auto extra = endpoints.front();
    extra.piece.cut = {body, body->getTerminator()};
    auto tail = fs::buildContextualFamilyEndpointRecipe(arena, {extra});
    if (failed(tail)) { return tailFailure(__LINE__); }
    auto extended = fs::appendFamilyEndpointRecipe(*nested, *tail, 64, 0);
    if (failed(extended)) { return tailFailure(__LINE__); }
    auto outer = fs::maskFamilyActivity(*extended, arena, endpoints.front().activity);
    if (failed(outer)) { return tailFailure(__LINE__); }
    auto outerForms = fs::familyEndpointForms(**outer);
    if (failed(outerForms) || outerForms->wrapperDiscoveryNodes != 4 ||
        outerForms->wrapperDiscoveryPieces != cuts.size() + 1 ||
        outerForms->wrapperIndexEntries != cuts.size() ||
        outerForms->wrapperBindingRequests != bindings + cuts.size() + 1) { return tailFailure(__LINE__); }
    return composed.endpoints.size() == 32 || tailFailure(__LINE__);
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
