// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Retain numerical and contextual family recipes through composition; lower at insertion.
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "FamilyMaterialization.h"
#include "PrivateStatePlan.h"
#include "PhysicalMapInputs.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "MixedFamilyInsertion.h"
#include "MixedEndpointPieces.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
#include "PTO/Transforms/FrontierSynch/Shared/EndpointPieces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>
#include <limits>
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
enum class FamilyContextKind { Invocation, Activity };
struct FamilyContextRecipe {
    FamilyContextKind kind = FamilyContextKind::Invocation;
    std::shared_ptr<RegionExpressions> arena;
    std::map<Operation*, std::shared_ptr<const CutExpressionRecipe>> cuts;
    uint64_t discoveryNodes = 0, discoveryPieces = 0, indexImports = 0, indexEntries = 0, bindingRequests = 0;
};
// Original cuts do not change under namespace translation, activity masking or
// invocation-coordinate binding. Only nonlocal commands need tuple bindings.
using FamilyCutIndex = std::map<Operation*, bool>;
struct NormalizedFamilyInventory;
class FamilyEndpointRecipe {
public:
    // A leaf owns numerical selectors or contextual bindings. A composition references earlier
    // recipes and records namespace/placement transforms; append is constant work.
    std::shared_ptr<const FamilyEndpointRecipe> previous, child;
    std::shared_ptr<const NormalizedFamilyInventory> normalized;
    std::shared_ptr<const FamilyContextRecipe> context;
    uint32_t recordBase = 0, maximumRecord = 0;
    std::size_t endpointIndex = 0, maximumEndpointIndex = 0, pieceCount = 0;
    struct Binding {
        std::shared_ptr<const CutExpressionRecipe> activity, ordinal, residueCoordinate;
        uint64_t modulus = 0, residue = 0;
    };
    std::shared_ptr<RegionExpressions> arena;
    std::vector<Binding> bindings;
    struct Shared {
        ContextualSharedExpression expression;
        std::shared_ptr<const CutExpressionRecipe> binding;
    };
    std::vector<Shared> shared;
    uint64_t sharedLookups = 0, sharedCandidates = 0, sharedBindings = 0;
    // Lazy metadata owned by this compilation; no expression availability is
    // cached here. Wrappers share the immutable index of their unchanged child.
    mutable std::shared_ptr<const FamilyCutIndex> originalCuts;
    scf::ForOp outer;
    std::vector<EndpointPiece> pieces;
    std::vector<std::shared_ptr<const FamilyExpressionRecipe>> selectors;
    struct Cut { TemplateEndpointCut cut; SmallVector<std::size_t> pieces; };
    std::vector<Cut> cuts;
};
struct FamilyContextPath {
    const FamilyContextRecipe* recipe;
    std::size_t parent;
    uint64_t invocationDepth;
};
struct FamilyPart {
    const FamilyEndpointRecipe* recipe;
    uint32_t recordBase;
    std::size_t endpointIndex, contextPath;
};
struct FamilyContribution {
    // A null leaf selects an immutable concrete contribution by inventory index.
    const FamilyEndpointRecipe* leaf = nullptr;
    uint32_t recordBase = 0;
    std::size_t contextPath = SIZE_MAX, concreteIndex = 0;
};
struct NormalizedFamilyInventory {
    SmallVector<FamilyContextPath> contexts;
    SmallVector<FamilyPart> leaves;
    SmallVector<FamilyContribution> stream;
    std::vector<std::shared_ptr<const FamilyEndpointRecipe>> owners;
    SmallVector<std::shared_ptr<const EndpointCutChoices>> selectionOwners;
    SmallVector<uint64_t> snapshot, identity;
    bool translateNamespaces = false;
};
struct PhysicalFamilyOutput {
    std::shared_ptr<const FamilyEndpointRecipe> protocol;
    DictionaryAttr certificate;
    SmallVector<int64_t> eligibleIds;
    std::map<const FamilyContribution*, SmallVector<std::shared_ptr<const PhysicalEndpointMap>>> deferred;
    SmallVector<std::shared_ptr<const PhysicalEndpointMap>> concrete;
    struct Binding {
        uint32_t source = 0, target = 0;
        std::shared_ptr<const PhysicalEndpointMap> mapping;
    };
    std::map<std::tuple<Operation*, LogicalCommandKind, uint32_t>, Binding> bindings;
};
namespace {
class FamilyPreparer {
public:
    FamilyPreparer(
        const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared, uint32_t recordBase,
        const std::map<uint32_t, uint32_t>& namespaces, FamilyMaterialization& materialization,
        ArrayRef<std::shared_ptr<const PhysicalEndpointMap>> maps = {})
        : recipe(recipe),
          prepared(prepared),
          recordBase(recordBase),
          namespaces(namespaces),
          materialization(materialization),
          maps(maps),
          builder(recipe.cuts.front().cut.before->getContext())
    {}
    LogicalResult run()
    {
        bool needsOrdinals = maps.empty();
        for (auto [index, piece] : llvm::enumerate(recipe.pieces)) {
            needsOrdinals |= piece.displacement || (piece.kind != EndpointKind::Barrier &&
                                                    (maps.empty() || physicalMapNeedsCoordinate(*maps[index], 0)));
        }
        if (recipe.outer && needsOrdinals) {
            initializeOrdinals();
        }
        if (recipe.arena) {
            return prepareContextual();
        }
        for (const auto& cut : recipe.cuts) {
            if (!cut.pieces.empty()) {
                prepareCut(cut.cut, cut.pieces);
            }
        }
        return success();
    }

private:
    const FamilyEndpointRecipe& recipe;
    PreparedLogicalPlan& prepared;
    uint32_t recordBase;
    const std::map<uint32_t, uint32_t>& namespaces;
    FamilyMaterialization& materialization;
    ArrayRef<std::shared_ptr<const PhysicalEndpointMap>> maps;
    OpBuilder builder;
    Value trips;
    Value ordinal;
    Value zero;
    Value one;
    LogicalResult prepareContextual()
    {
        const auto* environment = recipe.shared.empty() ? nullptr : &recipe;
        for (const auto& shared : recipe.shared) {
            auto* cut = shared.expression.cut;
            Block code;
            builder.setInsertionPointToEnd(&code);
            auto value = materialization.emit(*recipe.arena, *shared.binding, builder, cut, environment);
            if (failed(value)) {
                return failure();
            }
            materialization.bind(*recipe.arena, recipe, cut, shared.expression.expression, *value);
            adopt(code, cut);
        }
        if (environment) {
            materialization.publish(*recipe.arena, recipe);
        }
        for (std::size_t i = 0; i < recipe.pieces.size(); ++i) {
            const auto& piece = recipe.pieces[i];
            const auto& binding = recipe.bindings[i];
            auto* cut = piece.cut.before;
            Block code;
            builder.setInsertionPointToEnd(&code);
            if (!recipe.selectors.empty() && recipe.selectors[i]) {
                if (!recipe.outer && (maps.empty() || (maps[i] && physicalMapNeedsCoordinate(*maps[i], 0)))) {
                    ordinal = number(0, cut->getLoc());
                }
                preparePiece(cut, i, boundary(piece, cut->getLoc()));
                adopt(code, cut);
                continue;
            }
            auto emit = [&](const std::shared_ptr<const CutExpressionRecipe>& value) {
                return materialization.emit(*recipe.arena, *value, builder, cut, environment);
            };
            auto guard = emit(binding.activity);
            if (failed(guard)) {
                return failure();
            }
            const bool local = piece.kind == EndpointKind::Barrier;
            auto kind = local ? LogicalCommandKind::Barrier :
                                (piece.kind == EndpointKind::Set ? LogicalCommandKind::Set : LogicalCommandKind::Wait);
            const auto record = piece.members.front().record + recordBase;
            PreparedLogicalEndpoint endpoint{
                cut,
                kind,
                piece.sourcePipe,
                piece.targetPipe,
                namespaces.empty() ? piece.namespaceId + recordBase : namespaces.at(record),
                *guard,
                {}};
            endpoint.records = {record};
            if (!maps.empty()) {
                endpoint.physicalMapping = maps[i];
            }
            retainEndpointCoordinates(endpoint, piece.members.front().coordinates);
            if (binding.residueCoordinate) {
                auto coordinate = emit(binding.residueCoordinate);
                if (failed(coordinate)) {
                    return failure();
                }
                if (endpoint.domain.empty()) {
                    endpoint.domain.emplace_back();
                }
                for (auto& box : endpoint.domain) {
                    box.push_back({*coordinate, binding.modulus, static_cast<int64_t>(binding.residue)});
                }
            }
            if (!local) {
                if (!endpoint.physicalMapping || physicalMapNeedsCoordinate(*endpoint.physicalMapping, 0)) {
                    auto ordinal = emit(binding.ordinal);
                    if (failed(ordinal)) {
                        return failure();
                    }
                    endpoint.identity = *ordinal;
                }
                endpoint.memberCoordinates.push_back(
                    endpoint.physicalMapping && !physicalMapRequiresMember(*endpoint.physicalMapping) ?
                        Value{} :
                        number(record, cut->getLoc()));
            }
            endpoint.piece = prepared.endpoints.size();
            prepared.endpoints.push_back(std::move(endpoint));
            adopt(code, cut);
        }
        return success();
    }
    void adopt(Block& code, Operation* cut)
    {
        if (code.empty()) { return; }
        auto& target = prepared.addPreparation(cut);
        target.getOperations().splice(target.end(), code.getOperations());
    }
    Value number(uint64_t value, Location location)
    {
        auto type = builder.getIndexType();
        return builder.create<arith::ConstantOp>(location, type, IntegerAttr::get(type, APInt(64, value)));
    }
    Value compare(arith::CmpIPredicate predicate, Value a, Value b, Location location)
    {
        return builder.create<arith::CmpIOp>(location, predicate, a, b);
    }
    void initializeOrdinals()
    {
        auto loop = recipe.outer;
        auto location = loop.getLoc();
        builder.setInsertionPointToEnd(&prepared.addPreparation(loop));
        zero = number(0, location);
        one = number(1, location);
        auto positive = compare(arith::CmpIPredicate::sgt, loop.getUpperBound(), loop.getLowerBound(), location);
        auto difference = builder.create<arith::SubIOp>(location, loop.getUpperBound(), loop.getLowerBound());
        auto span = builder.create<arith::SelectOp>(location, positive, difference, zero);
        auto quotient = builder.create<arith::DivUIOp>(location, span, loop.getStep());
        auto remainder = builder.create<arith::RemUIOp>(location, span, loop.getStep());
        auto partial = compare(arith::CmpIPredicate::ne, remainder, zero, location);
        auto extra = builder.create<arith::SelectOp>(location, partial, one, zero);
        trips = builder.create<arith::AddIOp>(location, quotient, extra);
        builder.setInsertionPointToEnd(&prepared.addPreparation(&loop.getBody()->front()));
        auto offset = builder.create<arith::SubIOp>(location, loop.getInductionVar(), loop.getLowerBound());
        ordinal = builder.create<arith::DivUIOp>(location, offset, loop.getStep());
    }
    Value boundary(const EndpointPiece& piece, Location location)
    {
        const auto displacement = piece.displacement;
        if (!displacement) {
            return builder.create<arith::ConstantIntOp>(location, 1, 1);
        }
        auto delay = number(displacement, location);
        if (piece.kind != EndpointKind::Set) {
            return compare(arith::CmpIPredicate::uge, ordinal, delay, location);
        }
        auto remaining = builder.create<arith::SubIOp>(location, trips, delay);
        auto hasTarget = compare(arith::CmpIPredicate::ult, delay, trips, location);
        auto beforeEnd = compare(arith::CmpIPredicate::ult, ordinal, remaining, location);
        return builder.create<arith::AndIOp>(location, hasTarget, beforeEnd);
    }
    void preparePiece(Operation* before, std::size_t index, Value partnerPresent)
    {
        auto location = before->getLoc();
        const auto& piece = recipe.pieces[index];
        const auto mapping = maps.empty() ? nullptr : maps[index];
        const bool needMember = !mapping || physicalMapRequiresMember(*mapping);
        auto selector = needMember ?
                            emitFamilyExpressions(builder, location, *recipe.selectors[index]) :
                            FamilyExpressions{{}, emitFamilyPresence(builder, location, *recipe.selectors[index]), {}};
        Value guard = builder.create<arith::AndIOp>(location, selector.present, partnerPresent);
        Value identity;
        auto kind = LogicalCommandKind::Barrier;
        const bool local = piece.kind == EndpointKind::Barrier;
        if (!local) {
            kind = piece.kind == EndpointKind::Set ? LogicalCommandKind::Set : LogicalCommandKind::Wait;
            if (!mapping || physicalMapNeedsCoordinate(*mapping, 0)) {
                identity = ordinal;
                if (piece.kind == EndpointKind::Wait && piece.displacement) {
                    identity = builder.create<arith::SubIOp>(location, ordinal, number(piece.displacement, location));
                }
            }
        }
        PreparedLogicalEndpoint endpoint{
            before,
            kind,
            piece.sourcePipe,
            piece.targetPipe,
            namespaces.empty() ? piece.namespaceId + recordBase :
                                 namespaces.at(piece.members.front().record + recordBase),
            guard,
            identity,
            {}};
        if (!local) {
            Value member = selector.member;
            if (member && recordBase) {
                member = builder.create<arith::AddIOp>(location, member, number(recordBase, location));
            }
            endpoint.memberCoordinates.push_back(member);
        }
        endpoint.physicalMapping = mapping;
        for (const auto& member : piece.members) {
            retainEndpointCoordinates(endpoint, member.coordinates);
            endpoint.records.push_back(member.record + recordBase);
        }
        endpoint.piece = static_cast<int64_t>(prepared.endpoints.size());
        prepared.endpoints.push_back(std::move(endpoint));
    }
    void prepareCut(TemplateEndpointCut cut, ArrayRef<std::size_t> selected)
    {
        builder.setInsertionPointToEnd(&prepared.addPreparation(cut.before));
        if (!recipe.outer && llvm::any_of(selected, [&](auto index) {
                return maps.empty() || (maps[index] && physicalMapNeedsCoordinate(*maps[index], 0));
            })) {
            ordinal = number(0, cut.before->getLoc());
        }
        std::map<std::pair<bool, uint64_t>, Value> predicates;
        for (auto index : selected) {
            const auto& piece = recipe.pieces[index];
            auto [entry, inserted] =
                predicates.try_emplace(std::make_pair(piece.kind == EndpointKind::Set, piece.displacement));
            if (inserted) {
                entry->second = boundary(piece, cut.before->getLoc());
            }
            preparePiece(cut.before, index, entry->second);
        }
    }
};
FamilyExpressionPlan buildPieceSelector(const EndpointPiece& piece)
{
    SmallVector<SmallVector<TemplateCoordinate>> coordinates;
    SmallVector<int64_t> labels;
    for (const auto& member : piece.members) {
        coordinates.push_back(member.coordinates);
        labels.push_back(member.record);
    }
    return buildFamilyExpressions(coordinates, labels);
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildRecipe(const NumericTemplateEndpoints& plan)
{
    if (!plan.logical.error.empty()) {
        return failure();
    }
    auto pieces = buildEndpointPieces(plan);
    if (!pieces.error.empty()) {
        if (plan.outer) { plan.outer->emitError(pieces.error); }
        return failure();
    }
    auto recipe = std::make_shared<FamilyEndpointRecipe>();
    recipe->outer = plan.outer;
    recipe->pieces = std::move(pieces.pieces);
    recipe->pieceCount = recipe->pieces.size();
    for (const auto& piece : recipe->pieces) {
        recipe->maximumRecord = std::max(recipe->maximumRecord, piece.namespaceId);
        for (const auto& member : piece.members) {
            recipe->maximumRecord = std::max(recipe->maximumRecord, member.record);
        }
    }
    std::map<Operation*, std::size_t> cuts;
    for (const auto& group : plan.groups) {
        if (!group.cut.before || group.cut.block != group.cut.before->getBlock() ||
            !cuts.emplace(group.cut.before, recipe->cuts.size()).second) { return failure(); }
        recipe->cuts.push_back({group.cut, {}});
    }
    for (const auto& piece : recipe->pieces) {
        auto cut = cuts.find(piece.cut.before);
        if (cut == cuts.end() || recipe->cuts[cut->second].cut.block != piece.cut.block) { return failure(); }
        recipe->cuts[cut->second].pieces.push_back(recipe->selectors.size());
        auto selection = buildPieceSelector(piece);
        if (!selection.error.empty()) {
            if (plan.outer) { plan.outer->emitError(selection.error); }
            return failure();
        }
        recipe->selectors.push_back(std::move(selection.recipe));
    }
    for (auto& cut : recipe->cuts) {
        llvm::sort(cut.pieces, [&](std::size_t a, std::size_t b) {
            return recipe->pieces[a].order < recipe->pieces[b].order;
        });
    }
    return std::shared_ptr<const FamilyEndpointRecipe>(std::move(recipe));
}
} // namespace
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildFamilyEndpointRecipe(const NumericTemplateEndpoints& plan)
{
    if (!plan.outer) { return failure(); }
    return buildRecipe(plan);
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildFiniteFamilyEndpointRecipe(
    const NumericTemplateEndpoints& plan)
{
    if (plan.outer || plan.logical.recipes.empty() ||
        llvm::any_of(plan.logical.recipes, [](const auto& recipe) { return recipe.displacement != 0; })) {
        return failure();
    }
    return buildRecipe(plan);
}
FailureOr<std::shared_ptr<const CutExpressionRecipe>> ContextualFamilyExpressions::prepare(
    RegionExpressions::Id expression, Operation* cut, ArrayRef<RegionExpressions::Id> shared)
{
    if (!owner) { return failure(); }
    auto& context = contexts[cut];
    auto& cache = recipes[cut];
    bool changed = false;
    for (auto bound : shared) {
        if (bound >= owner->size()) { return failure(); }
        changed |= context.plannedValues.insert(bound).second;
    }
    if (changed) {
        context.cofactors.clear();
        cache.clear();
    }
    auto found = cache.find(expression);
    if (found != cache.end()) { return found->second; }
    auto lookup = [&](RegionExpressions::Id id) { return sharedLookup && sharedLookup(id, cut); };
    RegionExpressions::SharedCutPlanning bindings{lookup};
    auto recipe = owner->prepareContextual(expression, cut, context, sharedLookup ? &bindings : nullptr);
    if (failed(recipe)) { return failure(); }
    cache.emplace(expression, *recipe);
    return *recipe;
}
std::pair<uint64_t, uint64_t> ContextualFamilyExpressions::sharedBindingWork() const
{
    uint64_t lookups = 0, bindings = 0;
    for (const auto& [cut, context] : contexts) {
        accumulateCost(lookups, context.planningSharedRequests);
        accumulateCost(bindings, context.plannedValues.size());
    }
    return {lookups, bindings};
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildContextualFamilyEndpointRecipe(
    std::shared_ptr<RegionExpressions> arena, ArrayRef<ContextualEndpointFamily> families,
    ArrayRef<ContextualSharedExpression> shared)
{
    ContextualFamilyExpressions expressions(std::move(arena));
    return buildContextualFamilyEndpointRecipe(expressions, families, shared);
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildContextualFamilyEndpointRecipe(
    ContextualFamilyExpressions& expressions, ArrayRef<ContextualEndpointFamily> families,
    ArrayRef<ContextualSharedExpression> shared)
{
    auto arena = expressions.arena();
    if (!arena || families.empty() || !families.front().piece.cut.before) { return failure(); }
    auto function = families.front().piece.cut.before->getParentOfType<func::FuncOp>();
    if (!function) { return failure(); }
    auto recipe = std::make_shared<FamilyEndpointRecipe>();
    recipe->arena = std::move(arena);
    std::map<Operation*, llvm::DenseSet<RegionExpressions::Id>> sharedRoots;
    for (const auto& expression : shared) {
        if (!expression.cut || expression.cut->getParentOfType<func::FuncOp>() != function) { return failure(); }
        if (!sharedRoots[expression.cut].insert(expression.expression).second) { continue; }
        auto planned = expressions.prepare(expression.expression, expression.cut);
        if (failed(planned)) { return failure(); }
        recipe->shared.push_back({expression, std::move(*planned)});
    }
    // Shared-root dependencies belong to this immutable recipe, not to the
    // caller's availability cache or a later independent family build.
    DominanceInfo dominance(function);
    std::map<RegionExpressions::Id, SmallVector<Operation*>> available;
    for (const auto& root : recipe->shared) {
        available[root.expression.expression].push_back(root.expression.cut);
    }
    ContextualFamilyExpressions boundExpressions(recipe->arena, [&](RegionExpressions::Id id, Operation* cut) {
        auto found = available.find(id);
        if (found == available.end()) { return false; }
        for (auto* source : found->second) {
            accumulateCost(recipe->sharedCandidates, 1);
            if (source == cut || dominance.properlyDominates(source, cut)) { return true; }
        }
        return false;
    });
    auto& endpointExpressions = shared.empty() ? expressions : boundExpressions;
    for (const auto& family : families) {
        const auto& piece = family.piece;
        auto* cut = piece.cut.before;
        if (!cut || piece.cut.block != cut->getBlock() || cut->getParentOfType<func::FuncOp>() != function ||
            piece.members.size() != 1 ||
            (piece.kind != EndpointKind::Set && piece.kind != EndpointKind::Wait &&
             piece.kind != EndpointKind::Barrier) ||
            ((piece.sourcePipe == piece.targetPipe) != (piece.kind == EndpointKind::Barrier))) { return failure(); }
        auto prepare = [&](RegionExpressions::Id expression, bool boolean) {
            if (expression >= recipe->arena->size() || recipe->arena->isBoolean(expression) != boolean) {
                return FailureOr<std::shared_ptr<const CutExpressionRecipe>>(failure());
            }
            return endpointExpressions.prepare(expression, cut);
        };
        auto guard = prepare(family.activity, true);
        if (failed(guard)) { return failure(); }
        FamilyEndpointRecipe::Binding binding;
        binding.activity = std::move(*guard);
        if (piece.kind != EndpointKind::Barrier) {
            auto ordinal = prepare(family.ordinal, false);
            if (failed(ordinal)) { return failure(); }
            binding.ordinal = std::move(*ordinal);
        }
        if (family.residueCoordinate != RegionExpressions::invalid) {
            if (!family.modulus || family.residue >= family.modulus || family.residue > INT64_MAX) { return failure(); }
            auto coordinate = prepare(family.residueCoordinate, false);
            if (failed(coordinate)) { return failure(); }
            binding.residueCoordinate = std::move(*coordinate);
            binding.modulus = family.modulus; binding.residue = family.residue;
        }
        recipe->maximumRecord = std::max({recipe->maximumRecord, piece.namespaceId, piece.members.front().record});
        recipe->cuts.push_back({piece.cut, {}});
        recipe->pieces.push_back(piece);
        recipe->bindings.push_back(std::move(binding));
    }
    recipe->pieceCount = recipe->pieces.size();
    std::tie(recipe->sharedLookups, recipe->sharedBindings) = boundExpressions.sharedBindingWork();
    return std::shared_ptr<const FamilyEndpointRecipe>(std::move(recipe));
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildMixedFamilyEndpointRecipe(
    ContextualFamilyExpressions& expressions, ArrayRef<ContextualEndpointFamily> families,
    ArrayRef<uint8_t> numerical, scf::ForOp outer, ArrayRef<ContextualSharedExpression> shared)
{
    if (families.size() != numerical.size() || families.empty()) { return failure(); }
    if (llvm::none_of(numerical, [](uint8_t value) { return value != 0; })) {
        return buildContextualFamilyEndpointRecipe(expressions, families, shared);
    }
    auto* firstCut = families.front().piece.cut.before;
    auto function = firstCut ? firstCut->getParentOfType<func::FuncOp>() : func::FuncOp{};
    if (!function || (outer && outer->getParentOfType<func::FuncOp>() != function)) { return failure(); }
    using Key = std::tuple<uint32_t, EndpointKind, Operation*>;
    std::map<Key, std::size_t> origins, residuals;
    SmallVector<EndpointPiece> points;
    SmallVector<ContextualEndpointFamily> residual;
    for (std::size_t i = 0; i < families.size(); ++i) {
        const auto& piece = families[i].piece;
        if (!piece.cut.before || piece.cut.before->getParentOfType<func::FuncOp>() != function ||
            piece.members.size() != 1 || (!outer && numerical[i] && piece.displacement)) { return failure(); }
        Key key{piece.members.front().record, piece.kind, piece.cut.before};
        if (!origins.emplace(key, i).second) { return failure(); }
        points.push_back(piece);
        if (!numerical[i]) {
            residuals.emplace(key, residual.size());
            residual.push_back(families[i]);
        }
    }
    auto grouped = buildMixedEndpointPieces(points, numerical);
    if (!grouped.error.empty()) { return failure(); }
    auto recipe = std::make_shared<FamilyEndpointRecipe>();
    std::shared_ptr<const FamilyEndpointRecipe> contextual;
    if (!residual.empty()) {
        auto planned = buildContextualFamilyEndpointRecipe(expressions, residual, shared);
        if (failed(planned)) { return failure(); }
        contextual = *planned;
        recipe->arena = contextual->arena;
        recipe->shared = contextual->shared;
        recipe->sharedLookups = contextual->sharedLookups;
        recipe->sharedCandidates = contextual->sharedCandidates;
        recipe->sharedBindings = contextual->sharedBindings;
    } else {
        recipe->arena = expressions.arena();
    }
    if (!recipe->arena) { return failure(); }
    recipe->outer = outer;
    llvm::sort(grouped.pieces, [](const auto& a, const auto& b) { return a.order < b.order; });
    for (auto& piece : grouped.pieces) {
        Key key{piece.members.front().record, piece.kind, piece.cut.before};
        FamilyEndpointRecipe::Binding binding;
        std::shared_ptr<const FamilyExpressionRecipe> selector;
        if (numerical[origins.at(key)]) {
            auto selected = buildPieceSelector(piece);
            if (!selected.error.empty()) { return failure(); }
            selector = std::move(selected.recipe);
        } else {
            binding = contextual->bindings[residuals.at(key)];
        }
        recipe->maximumRecord = std::max(recipe->maximumRecord, piece.namespaceId);
        for (const auto& member : piece.members) {
            recipe->maximumRecord = std::max(recipe->maximumRecord, member.record);
        }
        recipe->cuts.push_back({piece.cut, {}});
        recipe->pieces.push_back(std::move(piece));
        recipe->bindings.push_back(std::move(binding));
        recipe->selectors.push_back(std::move(selector));
    }
    recipe->pieceCount = recipe->pieces.size();
    return std::shared_ptr<const FamilyEndpointRecipe>(std::move(recipe));
}
std::map<uint32_t, uint32_t> familyRecordNamespaces(ArrayRef<EndpointFamily> families)
{
    using Key = std::tuple<uint32_t, uint32_t, uint64_t>;
    std::map<Key, uint32_t> minima;
    for (const auto& family : families) {
        Key key{family.sourcePipe, family.targetPipe, family.displacement};
        for (const auto& member : family.members) {
            auto entry = minima.emplace(key, member.record).first;
            entry->second = std::min(entry->second, member.record);
        }
    }
    std::map<uint32_t, uint32_t> result;
    for (const auto& family : families) {
        Key key{family.sourcePipe, family.targetPipe, family.displacement};
        for (const auto& member : family.members) { result.emplace(member.record, minima.at(key)); }
    }
    return result;
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> appendFamilyEndpointRecipe(
    std::shared_ptr<const FamilyEndpointRecipe> previous,
    std::shared_ptr<const FamilyEndpointRecipe> child, uint32_t recordBase, std::size_t endpointIndex)
{
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (!child || child->maximumRecord > UINT32_MAX - recordBase ||
        child->maximumEndpointIndex > limit - endpointIndex ||
        (previous && child->pieceCount > limit - previous->pieceCount)) { return failure(); }
    auto recipe = std::make_shared<FamilyEndpointRecipe>();
    recipe->maximumRecord = std::max(previous ? previous->maximumRecord : 0U, child->maximumRecord + recordBase);
    recipe->maximumEndpointIndex = std::max(previous ? previous->maximumEndpointIndex : 0,
                                           endpointIndex + child->maximumEndpointIndex);
    recipe->pieceCount = child->pieceCount + (previous ? previous->pieceCount : 0);
    recipe->recordBase = recordBase;
    recipe->endpointIndex = endpointIndex;
    recipe->previous = std::move(previous);
    recipe->child = std::move(child);
    return std::shared_ptr<const FamilyEndpointRecipe>(std::move(recipe));
}
namespace {
std::shared_ptr<const FamilyCutIndex> originalFamilyCuts(const FamilyEndpointRecipe& recipe,
                                                       FamilyContextRecipe& work)
{
    if (recipe.originalCuts) {
        accumulateCost(work.indexImports, 1);
        return recipe.originalCuts;
    }
    auto index = std::make_shared<FamilyCutIndex>();
    SmallVector<const FamilyEndpointRecipe*> pending{&recipe};
    llvm::DenseSet<const FamilyEndpointRecipe*> seen;
    while (!pending.empty()) {
        const auto* current = pending.pop_back_val();
        if (!seen.insert(current).second) { continue; }
        accumulateCost(work.discoveryNodes, 1);
        if (current->originalCuts) {
            accumulateCost(work.indexImports, 1);
            accumulateCost(work.indexEntries, current->originalCuts->size());
            for (const auto& [cut, nonlocal] : *current->originalCuts) { (*index)[cut] |= nonlocal; }
            continue;
        }
        if (current->child) { pending.push_back(current->child.get()); }
        if (current->previous) { pending.push_back(current->previous.get()); }
        for (const auto& piece : current->pieces) {
            accumulateCost(work.discoveryPieces, 1);
            (*index)[piece.cut.before] |= piece.sourcePipe != piece.targetPipe;
        }
    }
    recipe.originalCuts = index;
    return index;
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> bindFamilyContext(
    std::shared_ptr<const FamilyEndpointRecipe> child, std::shared_ptr<RegionExpressions> arena,
    RegionExpressions::Id expression, FamilyContextKind kind)
{
    if (!child || !arena || expression >= arena->size() ||
        arena->isBoolean(expression) != (kind == FamilyContextKind::Activity)) { return failure(); }
    auto context = std::make_shared<FamilyContextRecipe>();
    context->kind = kind;
    context->arena = std::move(arena);
    auto cuts = originalFamilyCuts(*child, *context);
    for (const auto& [cut, nonlocal] : *cuts) {
        if (kind == FamilyContextKind::Invocation && !nonlocal) { continue; }
        accumulateCost(context->bindingRequests, 1);
        RegionExpressions::CutEmission emission;
        auto prepared = context->arena->prepareContextual(expression, cut, emission);
        if (failed(prepared)) { return failure(); }
        context->cuts.emplace(cut, std::move(*prepared));
    }
    auto recipe = std::make_shared<FamilyEndpointRecipe>();
    recipe->maximumRecord = child->maximumRecord;
    recipe->maximumEndpointIndex = child->maximumEndpointIndex;
    recipe->pieceCount = child->pieceCount;
    recipe->originalCuts = std::move(cuts);
    recipe->child = std::move(child);
    recipe->context = std::move(context);
    return std::shared_ptr<const FamilyEndpointRecipe>(std::move(recipe));
}
} // namespace
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> prefixFamilyInvocation(
    std::shared_ptr<const FamilyEndpointRecipe> child, std::shared_ptr<RegionExpressions> arena,
    RegionExpressions::Id visit)
{
    return bindFamilyContext(std::move(child), std::move(arena), visit, FamilyContextKind::Invocation);
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> maskFamilyActivity(
    std::shared_ptr<const FamilyEndpointRecipe> child, std::shared_ptr<RegionExpressions> arena,
    RegionExpressions::Id activity)
{
    return bindFamilyContext(std::move(child), std::move(arena), activity, FamilyContextKind::Activity);
}
namespace {
LogicalResult collectFamilyParts(const FamilyEndpointRecipe& recipe, SmallVectorImpl<FamilyPart>& leaves,
                                 SmallVectorImpl<FamilyContextPath>& contexts)
{
    if (recipe.normalized) {
        llvm::append_range(leaves, recipe.normalized->leaves);
        llvm::append_range(contexts, recipe.normalized->contexts);
        return success();
    }
    SmallVector<FamilyPart> pending{{&recipe, 0, 0, SIZE_MAX}};
    // Preserve inventory order without recursive host stack use. Coordinate
    // metadata and scalar lowering consume this same composition traversal.
    while (!pending.empty()) {
        auto part = pending.pop_back_val();
        if (!part.recipe->child) {
            if (!leaves.empty() && leaves.back().endpointIndex > part.endpointIndex) { return failure(); }
            leaves.push_back(part);
            continue;
        }
        auto path = part.contextPath;
        if (part.recipe->context) {
            uint64_t depth = path == SIZE_MAX ? 0 : contexts[path].invocationDepth;
            if (part.recipe->context->kind == FamilyContextKind::Invocation) {
                if (depth == UINT64_MAX) { return failure(); }
                ++depth;
            }
            path = contexts.size();
            contexts.push_back({part.recipe->context.get(), part.contextPath, depth});
        }
        pending.push_back({part.recipe->child.get(), part.recordBase + part.recipe->recordBase,
                           part.endpointIndex + part.recipe->endpointIndex, path});
        if (part.recipe->previous) {
            pending.push_back({part.recipe->previous.get(), part.recordBase, part.endpointIndex, part.contextPath});
        }
    }
    return success();
}
class FamilyContextLowering {
public:
    FamilyContextLowering(PreparedLogicalPlan& plan, FamilyMaterialization& materialization)
        : plan(plan), materialization(materialization)
    {}
    LogicalResult apply(PreparedLogicalEndpoint& endpoint, std::size_t path, ArrayRef<FamilyContextPath> paths)
    {
        // The innermost binding is visited first. Inserting at member one then
        // leaves enclosing invocation coordinates before their descendants.
        while (path != SIZE_MAX) {
            const auto& item = paths[path];
            path = item.parent;
            if (item.recipe->kind == FamilyContextKind::Invocation && endpoint.kind == LogicalCommandKind::Barrier) {
                continue;
            }
            if (item.recipe->kind == FamilyContextKind::Invocation && endpoint.physicalMapping &&
                !physicalMapNeedsCoordinate(*endpoint.physicalMapping, item.invocationDepth)) {
                endpoint.memberCoordinates.insert(endpoint.memberCoordinates.begin() + 1, Value{});
                continue;
            }
            auto value = lower(*item.recipe, endpoint.before);
            if (failed(value)) {
                return failure();
            }
            if (item.recipe->kind == FamilyContextKind::Invocation) {
                if (endpoint.memberCoordinates.empty()) {
                    return failure();
                }
                endpoint.memberCoordinates.insert(endpoint.memberCoordinates.begin() + 1, *value);
            } else {
                // The leaf guard can be in a later stage than the shared cut
                // value. Append after both dependencies, never into that cache's block.
                OpBuilder builder(endpoint.before->getContext());
                builder.setInsertionPointToEnd(&plan.addPreparation(endpoint.before));
                auto location = endpoint.before->getLoc();
                auto no = builder.create<arith::ConstantIntOp>(location, 0, 1);
                endpoint.guard = builder.create<arith::SelectOp>(location, *value, endpoint.guard, no);
            }
        }
        return success();
    }

private:
    PreparedLogicalPlan& plan;
    FamilyMaterialization& materialization;
    FailureOr<Value> lower(const FamilyContextRecipe& recipe, Operation* cut)
    {
        auto found = recipe.cuts.find(cut);
        if (found == recipe.cuts.end()) { return failure(); }
        // Shared values may have been emitted by a preceding leaf. Append after
        // that preparation rather than returning to an earlier wrapper block.
        Block code;
        OpBuilder builder(cut->getContext());
        builder.setInsertionPointToEnd(&code);
        auto value = materialization.emit(*recipe.arena, *found->second, builder, cut);
        if (failed(value)) { return failure(); }
        if (!code.empty()) {
            auto& target = plan.addPreparation(cut);
            target.getOperations().splice(target.end(), code.getOperations());
        }
        return value;
    }
};
} // namespace
std::size_t familyEndpointPieceCount(const FamilyEndpointRecipe& recipe) { return recipe.pieceCount; }
FailureOr<FamilyEndpointForms> familyEndpointForms(const FamilyEndpointRecipe& recipe)
{
    SmallVector<FamilyPart> leaves;
    SmallVector<FamilyContextPath> contexts;
    if (failed(collectFamilyParts(recipe, leaves, contexts))) { return failure(); }
    FamilyEndpointForms result;
    llvm::DenseSet<const FamilyContextRecipe*> countedContexts;
    for (const auto& path : contexts) {
        const auto* context = path.recipe;
        if (!countedContexts.insert(context).second) { continue; }
        accumulateCost(result.wrapperDiscoveryNodes, context->discoveryNodes);
        accumulateCost(result.wrapperDiscoveryPieces, context->discoveryPieces);
        accumulateCost(result.wrapperIndexImports, context->indexImports);
        accumulateCost(result.wrapperIndexEntries, context->indexEntries);
        accumulateCost(result.wrapperBindingRequests, context->bindingRequests);
    }
    llvm::DenseSet<const FamilyEndpointRecipe*> counted;
    for (const auto& part : leaves) {
        const auto& leaf = *part.recipe;
        if (counted.insert(&leaf).second) {
            accumulateCost(result.sharedRoots, leaf.shared.size());
            accumulateCost(result.sharedLookups, leaf.sharedLookups);
            accumulateCost(result.sharedCandidates, leaf.sharedCandidates);
            accumulateCost(result.sharedBindings, leaf.sharedBindings);
        }
        std::size_t numerical = 0;
        for (std::size_t i = 0; i < leaf.pieces.size(); ++i) {
            numerical += !leaf.arena || (!leaf.selectors.empty() && leaf.selectors[i]);
        }
        result.numericalSites += numerical;
        result.contextualSites += leaf.pieces.size() - numerical;
        result.mixedLeaves += numerical && numerical < leaf.pieces.size();
    }
    return result;
}
FailureOr<std::map<uint32_t, uint64_t>> familyMatchingArities(const FamilyEndpointRecipe& recipe)
{
    SmallVector<FamilyPart> leaves;
    SmallVector<FamilyContextPath> contexts;
    if (failed(collectFamilyParts(recipe, leaves, contexts))) { return failure(); }
    std::map<uint32_t, uint64_t> result;
    for (const auto& part : leaves) {
        const uint64_t depth = part.contextPath == SIZE_MAX ? 0 : contexts[part.contextPath].invocationDepth;
        if (depth == UINT64_MAX) { return failure(); }
        for (const auto& piece : part.recipe->pieces) {
            if (piece.kind == EndpointKind::Barrier) { continue; }
            for (const auto& member : piece.members) {
                auto [entry, added] = result.emplace(member.record + part.recordBase, depth + 1);
                if (!added && entry->second != depth + 1) { return failure(); }
            }
        }
    }
    return result;
}
namespace {
void appendCoordinates(ArrayRef<TemplateCoordinate> coordinates, SmallVectorImpl<uint64_t>& key)
{
    key.push_back(coordinates.size());
    for (auto coordinate : coordinates) {
        key.push_back(reinterpret_cast<uintptr_t>(coordinate.loop.getOperation()));
        key.push_back(static_cast<uint64_t>(coordinate.induction));
    }
}
void appendPieceKey(const EndpointPiece& piece, SmallVectorImpl<uint64_t>& key)
{
    key.append(
        {piece.namespaceId, piece.sourcePipe, piece.targetPipe, piece.displacement, uint64_t(piece.kind),
         reinterpret_cast<uintptr_t>(piece.cut.block), reinterpret_cast<uintptr_t>(piece.cut.before), piece.order,
         piece.members.size()});
    for (const auto& member : piece.members) {
        key.push_back(member.record);
        appendCoordinates(member.coordinates, key);
    }
}
// Canonicalize generated operand identities structurally. The key stores only
// integers/opaque source identities, so later CSE cannot leave dangling Values
// in an allocation cache. It is never decoded as executable semantics.
SmallVector<uint64_t> familyPlanSnapshot(const PreparedLogicalPlan& plan)
{
    SmallVector<uint64_t> key{uint64_t(plan.planId),  plan.completeInvocation, plan.nestedIdentities,
                              plan.independentPieces, plan.groupedFamilies,    plan.preparation.size()};
    DenseMap<Value, uint64_t> generated;
    uint64_t next = 0;
    auto value = [&](Value input) {
        auto found = generated.find(input);
        key.push_back(found == generated.end() ? 0 : 1);
        key.push_back(
            found == generated.end() ? reinterpret_cast<uintptr_t>(input.getAsOpaquePointer()) : found->second);
    };
    for (const auto& stage : plan.preparation) {
        key.push_back(reinterpret_cast<uintptr_t>(stage.before));
        key.push_back(bool(stage.code));
        if (!stage.code) {
            continue;
        }
        key.push_back(stage.code->getOperations().size());
        for (auto& operation : *stage.code) {
            key.push_back(reinterpret_cast<uintptr_t>(operation.getName().getStringRef().data()));
            key.push_back(reinterpret_cast<uintptr_t>(operation.getAttrDictionary().getAsOpaquePointer()));
            key.push_back(operation.getNumOperands());
            for (auto operand : operation.getOperands()) {
                value(operand);
            }
            key.push_back(operation.getNumResults());
            for (auto result : operation.getResults()) {
                key.push_back(reinterpret_cast<uintptr_t>(result.getType().getAsOpaquePointer()));
                generated[result] = next++;
            }
            // Concrete region-bearing compatibility code has no canonical
            // recipe here. Its unique owner prevents unsafe cross-plan reuse.
            key.push_back(operation.getNumRegions() ? reinterpret_cast<uintptr_t>(&operation) : 0);
        }
    }
    key.push_back(plan.endpoints.size());
    for (const auto& endpoint : plan.endpoints) {
        key.append(
            {reinterpret_cast<uintptr_t>(endpoint.before), uint64_t(endpoint.kind), endpoint.sourcePipe,
             endpoint.targetPipe, uint64_t(endpoint.record), uint64_t(endpoint.piece)});
        value(endpoint.guard);
        value(endpoint.identity);
        key.push_back(endpoint.memberCoordinates.size());
        for (auto coordinate : endpoint.memberCoordinates) {
            value(coordinate);
        }
        key.push_back(endpoint.records.size());
        for (auto record : endpoint.records) {
            key.push_back(record);
        }
        key.push_back(endpoint.domain.size());
        for (const auto& box : endpoint.domain) {
            key.push_back(box.size());
            for (const auto& constraint : box) {
                value(constraint.coordinate);
                key.push_back(constraint.modulus);
                key.push_back(uint64_t(constraint.value));
            }
        }
    }
    key.push_back(plan.families.size());
    for (const auto& family : plan.families) {
        key.append(
            {family.id, family.sourcePipe, family.targetPipe, family.displacement, family.local,
             reinterpret_cast<uintptr_t>(family.sourceCut.before), reinterpret_cast<uintptr_t>(family.targetCut.before),
             family.sourceOrder, family.targetOrder});
        for (const auto* choice : {family.sourceChoices.get(), family.targetChoices.get()}) {
            key.push_back(bool(choice));
            if (!choice) {
                continue;
            }
            key.push_back(reinterpret_cast<uintptr_t>(choice->loop().getOperation()));
            key.push_back(choice->pendingMaterialization());
            value(choice->consumerPredicate());
            // The immutable arena/predicate owns the pending singleton proof.
            // Fresh wrappers can retain that same evidence across preparation.
            if (choice->pendingMaterialization()) {
                choice->appendPendingSelectionKey(key);
            }
            key.push_back(choice->cuts().size());
            for (auto cut : choice->cuts()) {
                key.push_back(reinterpret_cast<uintptr_t>(cut.before));
            }
        }
        key.push_back(family.members.size());
        for (const auto& member : family.members) {
            key.append({member.record, member.source, member.target});
            appendCoordinates(member.sourceCoordinates, key);
            appendCoordinates(member.targetCoordinates, key);
        }
    }
    return key;
}
void appendLeafKey(const FamilyPart& part, SmallVectorImpl<uint64_t>& key)
{
    const auto& leaf = *part.recipe;
    auto outer = leaf.outer;
    key.append(
        {part.recordBase, part.contextPath, reinterpret_cast<uintptr_t>(outer.getOperation()),
         reinterpret_cast<uintptr_t>(leaf.arena.get()), leaf.pieces.size()});
    for (std::size_t i = 0; i < leaf.pieces.size(); ++i) {
        appendPieceKey(leaf.pieces[i], key);
        const bool numerical = !leaf.arena || (!leaf.selectors.empty() && leaf.selectors[i]);
        key.push_back(numerical);
        if (numerical) {
            appendFamilyExpressionKey(*leaf.selectors[i], key);
        } else {
            const auto& binding = leaf.bindings[i];
            appendCutExpressionKey(*binding.activity, key);
            key.push_back(bool(binding.ordinal));
            if (binding.ordinal) {
                appendCutExpressionKey(*binding.ordinal, key);
            }
            key.push_back(bool(binding.residueCoordinate));
            if (binding.residueCoordinate) {
                appendCutExpressionKey(*binding.residueCoordinate, key);
            }
            key.append({binding.modulus, binding.residue});
        }
    }
    key.push_back(leaf.shared.size());
    for (const auto& shared : leaf.shared) {
        appendCutExpressionKey(*shared.binding, key);
    }
}
bool numericalLeaf(const FamilyEndpointRecipe& recipe)
{
    if (!recipe.shared.empty() || recipe.selectors.size() != recipe.pieces.size()) {
        return false;
    }
    return llvm::all_of(recipe.selectors, [](const auto& selector) { return bool(selector); });
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> mergeNumericalLeaves(
    ArrayRef<FamilyPart> parts, const std::map<uint32_t, uint32_t>& namespaces)
{
    SmallVector<EndpointPiece> points;
    SmallVector<uint8_t> numerical;
    for (const auto& part : parts) {
        const auto& leaf = *part.recipe;
        SmallVector<std::size_t> selected;
        if (leaf.arena) {
            for (std::size_t i = 0; i < leaf.pieces.size(); ++i) {
                selected.push_back(i);
            }
        } else {
            for (const auto& cut : leaf.cuts) {
                llvm::append_range(selected, cut.pieces);
            }
        }
        for (auto index : selected) {
            const auto& piece = leaf.pieces[index];
            for (auto member : piece.members) {
                member.record += part.recordBase;
                // Copy fixed metadata only. Copying piece here duplicates its
                // complete member inventory once for every singleton output.
                EndpointPiece point{piece.namespaceId + part.recordBase, piece.sourcePipe, piece.targetPipe,
                    piece.displacement, piece.kind, piece.cut, piece.order, {}};
                point.members.push_back(std::move(member));
                points.push_back(std::move(point));
                numerical.push_back(1);
            }
        }
    }
    auto grouped = buildMixedEndpointPieces(points, numerical);
    if (!grouped.error.empty()) {
        return failure();
    }
    auto merged = std::make_shared<FamilyEndpointRecipe>();
    merged->outer = parts.front().recipe->outer;
    llvm::sort(grouped.pieces, [](const auto& a, const auto& b) { return a.order < b.order; });
    std::map<Operation*, std::size_t> cuts;
    for (auto& piece : grouped.pieces) {
        if (!namespaces.empty()) {
            piece.namespaceId = namespaces.at(piece.members.front().record);
        }
        merged->maximumRecord = std::max(merged->maximumRecord, piece.namespaceId);
        for (const auto& member : piece.members) {
            merged->maximumRecord = std::max(merged->maximumRecord, member.record);
        }
        auto selected = buildPieceSelector(piece);
        if (!selected.error.empty()) {
            return failure();
        }
        auto [cut, added] = cuts.emplace(piece.cut.before, merged->cuts.size());
        if (added) {
            merged->cuts.push_back({piece.cut, {}});
        }
        merged->cuts[cut->second].pieces.push_back(merged->pieces.size());
        merged->pieces.push_back(std::move(piece));
        merged->selectors.push_back(std::move(selected.recipe));
    }
    merged->pieceCount = merged->pieces.size();
    return std::shared_ptr<const FamilyEndpointRecipe>(std::move(merged));
}
} // namespace
LogicalResult normalizeFamilyEndpointPlan(PreparedLogicalPlan& plan)
{
    if (!plan.familyRecipe) {
        return success();
    }
    if (plan.familyProtocol && plan.familyProtocol != plan.familyRecipe) {
        return failure();
    }
    if (plan.familyRecipe->normalized) {
        return success(plan.familyRecipe->normalized->snapshot == familyPlanSnapshot(plan));
    }
    const auto original = plan.familyRecipe;
    if (!original->child && (!plan.preparation.empty() || !plan.endpoints.empty())) {
        return failure();
    }
    SmallVector<FamilyPart> leaves;
    auto inventory = std::make_shared<NormalizedFamilyInventory>();
    for (const auto& family : plan.families) {
        for (const auto& choice : {family.sourceChoices, family.targetChoices}) {
            if (choice && choice->pendingMaterialization()) {
                inventory->selectionOwners.push_back(choice);
            }
        }
    }
    if (failed(collectFamilyParts(*original, leaves, inventory->contexts))) {
        return failure();
    }
    inventory->snapshot = familyPlanSnapshot(plan);
    inventory->translateNamespaces = bool(original->child);
    const auto namespaces =
        inventory->translateNamespaces ? familyRecordNamespaces(plan.families) : std::map<uint32_t, uint32_t>{};
    uint64_t inputSites = 0;
    for (const auto& part : leaves) {
        if (part.recipe->pieces.size() > UINT64_MAX - inputSites) {
            return failure();
        }
        inputSites += part.recipe->pieces.size();
        for (const auto& piece : part.recipe->pieces) {
            if (piece.members.empty()) {
                return failure();
            }
            std::optional<uint32_t> space;
            for (const auto& member : piece.members) {
                if (member.record > UINT32_MAX - part.recordBase) {
                    return failure();
                }
                if (!inventory->translateNamespaces) {
                    continue;
                }
                auto found = namespaces.find(member.record + part.recordBase);
                if (found == namespaces.end() || (space && *space != found->second)) {
                    return failure();
                }
                space = found->second;
            }
        }
    }
    // Group only adjacent numerical contributions with identical invocation
    // and binding ownership. Concrete entries and contextual leaves are fixed
    // anchors. No unresolved predicate is used as exclusion evidence.
    for (std::size_t i = 0; i < leaves.size();) {
        auto part = leaves[i];
        std::size_t end = i + 1;
        if (numericalLeaf(*part.recipe)) {
            while (end < leaves.size() && numericalLeaf(*leaves[end].recipe) &&
                   leaves[end].endpointIndex == part.endpointIndex && leaves[end].contextPath == part.contextPath &&
                   leaves[end].recipe->outer == part.recipe->outer) {
                ++end;
            }
        }
        if (end > i + 1) {
            auto merged = mergeNumericalLeaves(ArrayRef<FamilyPart>(leaves).slice(i, end - i), namespaces);
            if (failed(merged)) {
                return failure();
            }
            inventory->owners.push_back(*merged);
            part.recipe = merged->get();
            part.recordBase = 0;
        }
        inventory->leaves.push_back(part);
        i = end;
    }
    std::size_t concrete = 0;
    for (const auto& part : inventory->leaves) {
        if (part.endpointIndex > plan.endpoints.size()) {
            return failure();
        }
        while (concrete < part.endpointIndex) {
            inventory->stream.push_back({nullptr, 0, SIZE_MAX, concrete++});
        }
        inventory->stream.push_back({part.recipe, part.recordBase, part.contextPath, 0});
    }
    while (concrete < plan.endpoints.size()) {
        inventory->stream.push_back({nullptr, 0, SIZE_MAX, concrete++});
    }
    inventory->identity = inventory->snapshot;
    inventory->identity.push_back(inventory->contexts.size());
    for (const auto& path : inventory->contexts) {
        inventory->identity.append(
            {uint64_t(path.recipe->kind), path.parent, path.invocationDepth,
             reinterpret_cast<uintptr_t>(path.recipe->arena.get()), path.recipe->cuts.size()});
        for (const auto& [cut, binding] : path.recipe->cuts) {
            (void)cut;
            appendCutExpressionKey(*binding, inventory->identity);
        }
    }
    inventory->identity.push_back(inventory->stream.size());
    for (const auto& contribution : inventory->stream) {
        inventory->identity.push_back(bool(contribution.leaf));
        if (!contribution.leaf) {
            inventory->identity.push_back(contribution.concreteIndex);
        } else {
            appendLeafKey(
                {contribution.leaf, contribution.recordBase, 0, contribution.contextPath}, inventory->identity);
        }
    }
    auto normalized = std::make_shared<FamilyEndpointRecipe>();
    normalized->child = original;
    normalized->maximumRecord = original->maximumRecord;
    normalized->maximumEndpointIndex = original->maximumEndpointIndex;
    for (const auto& part : inventory->leaves) {
        normalized->pieceCount += part.recipe->pieces.size();
    }
    normalized->normalized = std::move(inventory);
    plan.familyWork.normalizationInputs = inputSites;
    plan.familyWork.normalizedSites = normalized->pieceCount;
    plan.familyRecipe = normalized;
    plan.familyProtocol = std::move(normalized);
    return success();
}
std::shared_ptr<const FamilyEndpointRecipe> familyRecipeForComposition(
    std::shared_ptr<const FamilyEndpointRecipe> recipe)
{
    return recipe && recipe->normalized ? recipe->child : recipe;
}
ArrayRef<uint64_t> normalizedFamilyProtocolKey(const FamilyEndpointRecipe& recipe)
{
    return recipe.normalized ? ArrayRef<uint64_t>(recipe.normalized->identity) : ArrayRef<uint64_t>{};
}
bool validFamilyEndpointProtocol(const PreparedLogicalPlan& plan)
{
    if (!plan.familyProtocol) {
        return !plan.physicalOutput;
    }
    if (!plan.familyProtocol->normalized) {
        return false;
    }
    const auto& snapshot = plan.familyProtocol->normalized->snapshot;
    return snapshot.size() >= 3 && snapshot[0] == uint64_t(plan.planId) && snapshot[1] == plan.completeInvocation &&
           snapshot[2] == plan.nestedIdentities &&
           (plan.familyRecipe ? plan.familyRecipe == plan.familyProtocol : plan.familyWork.materializations == 1);
}
bool validPhysicalFamilyOutput(const PreparedLogicalPlan& plan)
{
    return !plan.physicalOutput || (plan.familyProtocol == plan.physicalOutput->protocol &&
                                    plan.allocationCertificate == plan.physicalOutput->certificate);
}
bool validPhysicalFamilyEndpoint(const PreparedLogicalPlan& plan, const PreparedLogicalEndpoint& endpoint)
{
    if (!plan.physicalOutput) {
        return !endpoint.physicalMapping;
    }
    if (endpoint.kind == LogicalCommandKind::Barrier) {
        return !endpoint.physicalMapping;
    }
    if (!endpoint.physicalMapping || endpoint.records.empty() ||
        physicalMapTupleArity(*endpoint.physicalMapping) != endpoint.memberCoordinates.size()) {
        return false;
    }
    for (auto record : endpoint.records) {
        auto found = plan.physicalOutput->bindings.find({endpoint.before, endpoint.kind, record});
        if (found == plan.physicalOutput->bindings.end() || found->second.source != endpoint.sourcePipe ||
            found->second.target != endpoint.targetPipe ||
            (found->second.mapping != endpoint.physicalMapping &&
             !equivalentPhysicalMaps(*found->second.mapping, *endpoint.physicalMapping))) {
            return false;
        }
    }
    return true;
}
LogicalResult planPhysicalFamilyOutput(func::FuncOp function, PreparedLogicalPlan& plan, ArrayRef<int64_t> eligibleIds)
{
    if (plan.privateState) {
        return plan.privateState->requestPhysical(function, plan, eligibleIds);
    }
    if (plan.physicalOutput || plan.completionOwner || !plan.familyRecipe || !plan.allocationCertificate ||
        failed(preflightPhysicalOutput(function, plan.allocationCertificate)) ||
        failed(normalizeFamilyEndpointPlan(plan))) {
        return failure();
    }
    std::set<int64_t> pool;
    for (auto id : eligibleIds) {
        if (id < 0 || id > 5 || !pool.insert(id).second) {
            return failure();
        }
    }
    if (pool.empty()) {
        return failure();
    }
    auto decoded = decodePhysicalAllocationCertificate(function, plan.allocationCertificate, eligibleIds);
    if (failed(decoded) || decoded->planId != plan.planId) {
        return failure();
    }
    auto arities = familyMatchingArities(*plan.familyRecipe);
    if (failed(arities)) {
        return failure();
    }
    for (const auto& endpoint : plan.endpoints) {
        if (endpoint.kind == LogicalCommandKind::Barrier) {
            continue;
        }
        if (endpoint.memberCoordinates.empty()) {
            return failure();
        }
        for (auto record : endpoint.records) {
            auto [found, added] = arities->emplace(record, endpoint.memberCoordinates.size());
            if (!added && found->second != endpoint.memberCoordinates.size()) {
                return failure();
            }
        }
    }
    DominanceInfo dominance(function);
    auto build = [&](const PhysicalAllocationPlan& allocation) -> FailureOr<std::shared_ptr<PhysicalFamilyOutput>> {
        auto result = std::make_shared<PhysicalFamilyOutput>();
        result->protocol = plan.familyProtocol;
        result->certificate = plan.allocationCertificate;
        result->eligibleIds.assign(eligibleIds.begin(), eligibleIds.end());
        std::map<uint32_t, const PhysicalRecordAllocation*> records;
        for (const auto& record : allocation.records) {
            if (record.record < 0 || record.record > UINT32_MAX || !records.emplace(record.record, &record).second) {
                return failure();
            }
        }
        auto mapping = [&](Operation* cut, LogicalCommandKind kind, uint32_t source, uint32_t target,
                           ArrayRef<uint32_t> labels, ArrayRef<SmallVector<TemplateCoordinate>> points)
            -> FailureOr<std::shared_ptr<const PhysicalEndpointMap>> {
            SmallVector<PhysicalRecordAllocation> members;
            std::optional<uint64_t> arity;
            for (auto label : labels) {
                auto found = records.find(label);
                auto shape = arities->find(label);
                if (found == records.end() || shape == arities->end() || found->second->sourcePipe != source ||
                    found->second->targetPipe != target || (arity && *arity != shape->second)) {
                    return failure();
                }
                arity = shape->second;
                members.push_back(*found->second);
            }
            if (!arity) {
                return failure();
            }
            auto planned = planPhysicalEndpointMap(cut, members, points, *arity, plan.nestedIdentities, dominance);
            if (failed(planned)) {
                return failure();
            }
            for (auto record : labels) {
                if (!result->bindings
                         .emplace(
                             std::make_tuple(cut, kind, record),
                             PhysicalFamilyOutput::Binding{source, target, *planned})
                         .second) {
                    return failure();
                }
            }
            return *planned;
        };
        std::map<uint32_t, const EndpointFamilyMember*> provenance;
        for (const auto& family : plan.families) {
            for (const auto& member : family.members) {
                provenance.emplace(member.record, &member);
            }
        }
        for (const auto& endpoint : plan.endpoints) {
            if (endpoint.kind == LogicalCommandKind::Barrier) {
                result->concrete.push_back(nullptr);
                continue;
            }
            SmallVector<SmallVector<TemplateCoordinate>> points;
            for (auto record : endpoint.records) {
                auto found = provenance.find(record);
                if (found == provenance.end()) {
                    return failure();
                }
                points.push_back(
                    endpoint.kind == LogicalCommandKind::Set ? found->second->sourceCoordinates :
                                                               found->second->targetCoordinates);
            }
            auto planned = mapping(
                endpoint.before, endpoint.kind, endpoint.sourcePipe, endpoint.targetPipe, endpoint.records, points);
            if (failed(planned)) {
                return failure();
            }
            result->concrete.push_back(*planned);
        }
        for (const auto& contribution : plan.familyProtocol->normalized->stream) {
            if (!contribution.leaf) {
                continue;
            }
            auto& maps = result->deferred[&contribution];
            for (const auto& piece : contribution.leaf->pieces) {
                if (piece.kind == EndpointKind::Barrier) {
                    maps.push_back(nullptr);
                    continue;
                }
                SmallVector<uint32_t> labels;
                SmallVector<SmallVector<TemplateCoordinate>> points;
                for (const auto& member : piece.members) {
                    labels.push_back(member.record + contribution.recordBase);
                    points.push_back(member.coordinates);
                }
                auto planned = mapping(
                    piece.cut.before,
                    piece.kind == EndpointKind::Set ? LogicalCommandKind::Set : LogicalCommandKind::Wait,
                    piece.sourcePipe, piece.targetPipe, labels, points);
                if (failed(planned)) {
                    return failure();
                }
                maps.push_back(*planned);
            }
        }
        return result;
    };
    auto incumbent = build(*decoded);
    if (failed(incumbent)) {
        return failure();
    }
    auto cost = [](const PhysicalFamilyOutput& output) -> std::optional<uint64_t> {
        uint64_t total = 0;
        auto add = [&](const auto& map) {
            if (!map) {
                return true;
            }
            auto value = physicalMapCost(*map);
            if (!value || *value > uint64_t(INT64_MAX) - total) {
                return false;
            }
            total += *value;
            return true;
        };
        for (const auto& map : output.concrete) {
            if (!add(map)) {
                return std::nullopt;
            }
        }
        for (const auto& [contribution, maps] : output.deferred) {
            (void)contribution;
            for (const auto& map : maps) {
                if (!add(map)) {
                    return std::nullopt;
                }
            }
        }
        return total;
    };
    // Both candidates carry independent producer certificates. The secondary
    // estimate cannot authorize a new assignment or change the logical protocol.
    if (auto alternative = plan.allocationCertificate.getAs<DictionaryAttr>("regular_candidate");
        alternative && physicalCandidateFits(alternative, eligibleIds.size())) {
        FailureOr<std::shared_ptr<PhysicalFamilyOutput>> other = failure();
        {
            ScopedDiagnosticHandler optionalDiagnostics(function.getContext(), [](Diagnostic&) { return success(); });
            auto candidate = decodePhysicalAllocationCertificate(function, alternative, eligibleIds);
            if (succeeded(candidate) && candidate->planId == plan.planId) {
                other = build(*candidate);
            }
        }
        if (succeeded(other)) {
            const auto oldCost = cost(**incumbent), newCost = cost(**other);
            if (oldCost && newCost && *newCost < *oldCost) {
                incumbent = std::move(other);
            }
        }
    }
    plan.physicalOutput = std::move(*incumbent);
    return success();
}
LogicalResult materializeFamilyEndpointCode(const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared)
{
    if (!validFamilyEndpointProtocol(prepared) || !validPhysicalFamilyOutput(prepared) || !recipe.normalized ||
        prepared.familyWork.materializations) {
        return failure();
    }
    if (prepared.endpoints.size() > static_cast<uint64_t>(INT64_MAX) ||
        recipe.maximumEndpointIndex > prepared.endpoints.size() ||
        recipe.pieceCount > static_cast<uint64_t>(INT64_MAX) - prepared.endpoints.size()) {
        return failure();
    }
    SmallVector<FamilyPart> leaves;
    SmallVector<FamilyContextPath> contexts;
    if (failed(collectFamilyParts(recipe, leaves, contexts))) {
        return failure();
    }
    // Sequence namespace normalization applies to delayed and concrete records
    // alike. Validate the translated provenance before allocating any operations.
    auto namespaces = recipe.normalized->translateNamespaces ? familyRecordNamespaces(prepared.families) :
                                                               std::map<uint32_t, uint32_t>{};
    if (recipe.normalized->translateNamespaces) {
        for (const auto& part : leaves) {
            for (const auto& piece : part.recipe->pieces) {
                if (piece.members.empty()) { return failure(); }
                std::optional<uint32_t> space;
                for (const auto& member : piece.members) {
                    auto found = namespaces.find(member.record + part.recordBase);
                    if (found == namespaces.end() || (space && *space != found->second)) { return failure(); }
                    space = found->second;
                }
            }
        }
    }
    // Stage all lowering independently: unavailable contextual bindings must
    // not consume the incumbent concrete inventory or partially lower a recipe.
    PreparedLogicalPlan lowered(prepared.planId);
    FamilyMaterialization materialization;
    FamilyContextLowering contextLowering(lowered, materialization);
    for (const auto& contribution : recipe.normalized->stream) {
        if (!contribution.leaf) {
            lowered.endpoints.push_back(prepared.endpoints[contribution.concreteIndex]);
            if (prepared.physicalOutput) {
                lowered.endpoints.back().physicalMapping =
                    prepared.physicalOutput->concrete[contribution.concreteIndex];
                auto& endpoint = lowered.endpoints.back();
                if (endpoint.physicalMapping) {
                    if (!physicalMapNeedsCoordinate(*endpoint.physicalMapping, 0)) {
                        endpoint.identity = {};
                    }
                    if (!physicalMapRequiresMember(*endpoint.physicalMapping)) {
                        endpoint.memberCoordinates.front() = {};
                    }
                    for (std::size_t i = 1; i < endpoint.memberCoordinates.size(); ++i) {
                        if (!physicalMapNeedsCoordinate(*endpoint.physicalMapping, i)) {
                            endpoint.memberCoordinates[i] = {};
                        }
                    }
                }
            }
            continue;
        }
        const auto begin = lowered.endpoints.size();
        if (!contribution.leaf->pieces.empty() &&
            failed(FamilyPreparer(
                       *contribution.leaf, lowered, contribution.recordBase, namespaces, materialization,
                       prepared.physicalOutput ? ArrayRef<std::shared_ptr<const PhysicalEndpointMap>>(
                                                     prepared.physicalOutput->deferred.at(&contribution)) :
                                                 ArrayRef<std::shared_ptr<const PhysicalEndpointMap>>{})
                       .run())) {
            return failure();
        }
        for (std::size_t i = begin; i < lowered.endpoints.size(); ++i) {
            if (failed(contextLowering.apply(lowered.endpoints[i], contribution.contextPath, contexts))) {
                return failure();
            }
        }
    }
    for (std::size_t i = 0; i < lowered.endpoints.size(); ++i) {
        lowered.endpoints[i].piece = static_cast<int64_t>(i);
    }
    // Only pending cleanup witnesses need a final guard binding. Index their
    // record/cut requests once instead of scanning all endpoints per family.
    std::map<std::pair<uint32_t, Operation*>, Value> primaryGuards;
    for (const auto& family : prepared.families) {
        if (!family.targetChoices || !family.targetChoices->pendingMaterialization()) { continue; }
        if (!family.targetChoices->validPendingSelection() || family.members.size() != 1) { return failure(); }
        if (!primaryGuards.emplace(std::make_pair(family.members.front().record, family.targetCut.before),
                                   Value()).second) { return failure(); }
    }
    if (!primaryGuards.empty()) {
        for (const auto& endpoint : lowered.endpoints) {
            if (endpoint.kind != LogicalCommandKind::Wait) { continue; }
            for (auto record : endpoint.records) {
                auto found = primaryGuards.find({record, endpoint.before});
                if (found == primaryGuards.end()) { continue; }
                if (found->second) { return failure(); }
                found->second = endpoint.guard;
            }
        }
    }
    std::vector<std::pair<std::size_t, std::shared_ptr<const EndpointCutChoices>>> resolvedChoices;
    for (std::size_t f = 0; f < prepared.families.size(); ++f) {
        const auto& family = prepared.families[f];
        if (!family.targetChoices || !family.targetChoices->pendingMaterialization()) { continue; }
        auto predicate = primaryGuards.at({family.members.front().record, family.targetCut.before});
        if (!predicate) { return failure(); }
        auto choices = qualifyTerminalCleanupCuts(family.targetChoices->cuts(), predicate);
        if (!choices) { return failure(); }
        resolvedChoices.push_back({f, std::move(choices)});
    }
    for (auto& choice : resolvedChoices) { prepared.families[choice.first].targetChoices = std::move(choice.second); }
    const auto inputs = prepared.familyWork.normalizationInputs, sites = prepared.familyWork.normalizedSites;
    prepared.familyWork = materialization.work();
    prepared.familyWork.normalizationInputs = inputs;
    prepared.familyWork.normalizedSites = sites;
    prepared.familyWork.materializations = 1;
    prepared.endpoints = std::move(lowered.endpoints);
    for (auto& stage : lowered.preparation) { prepared.preparation.push_back(std::move(stage)); }
    prepared.groupedFamilies = true;
    prepared.independentPieces = true;
    return success();
}
} // namespace mlir::pto::frontiersynch
