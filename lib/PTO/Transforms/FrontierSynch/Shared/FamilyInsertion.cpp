// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Retain numerical and contextual family recipes through composition; lower at insertion.
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/DeferredFamily.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "FamilyMaterialization.h"
#include "ControlRootProfile.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "FamilyEndpointNormalization.h"
#include "CountedCleanupSelection.h"
#include "SelectorGuardPlan.h"
#include "CertifiedAllocationCandidates.h"
#include "PrivateStatePlan.h"
#include "PhysicalMapInputs.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalExecutedCounters.h"
#include "MixedFamilyInsertion.h"
#include "MixedEndpointPieces.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
#include "PTO/Transforms/FrontierSynch/Shared/EndpointPieces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
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
        SmallVector<std::shared_ptr<const CutExpressionRecipe>> matchingCoordinates;
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
    std::vector<EndpointFamily> families;
    std::vector<std::shared_ptr<const FamilyEndpointRecipe>> owners;
    SmallVector<std::shared_ptr<const EndpointCutChoices>> selectionOwners;
    SmallVector<uint64_t> snapshot, identity;
    bool translateNamespaces = false;
};
struct PhysicalFamilyOutput {
    bool emptyInvocation = false;
    int64_t emptyPlanId = 0;
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
template<class Scalar> struct ContextualFamilyInputs {
    Scalar guard, residue, ordinal;
    SmallVector<Scalar> members;
};
// Both emission and counting walk this exact root order. The consumer only
// supplies the scalar representation and the original-cut storage adapter.
class ContextualFamilySchedule {
public:
    ContextualFamilySchedule(const FamilyEndpointRecipe& recipe,
        ArrayRef<std::shared_ptr<const PhysicalEndpointMap>> maps, uint64_t invocationDepth, bool executed)
        : recipe(recipe), maps(maps), invocationDepth(invocationDepth), executed(executed) {}
    template<class Consumer> LogicalResult run(Consumer& consumer) const
    {
        auto live = liveSharedExpressions();
        if (failed(live)) { return failure(); }
        const auto* environment = recipe.shared.empty() ? nullptr : &recipe;
        for (auto [index, shared] : llvm::enumerate(recipe.shared)) {
            if (!(*live)[index]) { continue; }
            const auto& sharedRoot = shared;
            auto* cut = sharedRoot.expression.cut;
            if (failed(consumer.at(cut, [&]() {
                    auto value = consumer.emit(*sharedRoot.binding, cut, environment);
                    if (failed(value)) { return failure(); }
                    consumer.bind(cut, sharedRoot.expression.expression, *value);
                    return success();
                }))) { return failure(); }
        }
        if (environment) { consumer.publish(); }
        for (std::size_t i = 0; i < recipe.pieces.size(); ++i) {
            const auto& piece = recipe.pieces[i];
            const auto& binding = recipe.bindings[i];
            auto* cut = piece.cut.before;
            if (failed(consumer.at(cut, [&]() {
                    if (!recipe.selectors.empty() && recipe.selectors[i]) { return consumer.numerical(i); }
                    ContextualFamilyInputs<typename Consumer::Scalar> inputs;
                    auto emit = [&](const std::shared_ptr<const CutExpressionRecipe>& root, auto& value) {
                        auto result = consumer.emit(*root, cut, environment);
                        if (failed(result)) { return failure(); }
                        value = *result;
                        return success();
                    };
                    if (failed(emit(binding.activity, inputs.guard))) { return failure(); }
                    if (binding.residueCoordinate && failed(emit(binding.residueCoordinate, inputs.residue))) {
                        return failure();
                    }
                    if (piece.kind != EndpointKind::Barrier) {
                        if (!executed && (maps.empty() || physicalMapNeedsCoordinate(*maps[i], 0)) &&
                            failed(emit(binding.ordinal, inputs.ordinal))) { return failure(); }
                        inputs.members.push_back(executed ||
                            (!maps.empty() && !physicalMapRequiresMember(*maps[i])) ? typename Consumer::Scalar{} :
                            consumer.number(piece.members.front().record, cut));
                        for (auto [position, coordinate] : llvm::enumerate(binding.matchingCoordinates)) {
                            typename Consumer::Scalar value;
                            if (!executed && (maps.empty() ||
                                physicalMapNeedsCoordinate(*maps[i], invocationDepth + position + 1)) &&
                                failed(emit(coordinate, value))) { return failure(); }
                            inputs.members.push_back(value);
                        }
                    }
                    return consumer.endpoint(i, inputs);
                }))) { return failure(); }
        }
        return success();
    }
private:
    const FamilyEndpointRecipe& recipe;
    ArrayRef<std::shared_ptr<const PhysicalEndpointMap>> maps;
    uint64_t invocationDepth;
    bool executed;
    using SharedRequest = std::pair<const CutExpressionRecipe*, Operation*>;
    SmallVector<SharedRequest> liveContextualExpressions() const
    {
        SmallVector<SharedRequest> result;
        for (std::size_t i = 0; i < recipe.pieces.size(); ++i) {
            if (!recipe.selectors.empty() && recipe.selectors[i]) { continue; }
            const auto& piece = recipe.pieces[i];
            const auto& binding = recipe.bindings[i];
            auto add = [&](const std::shared_ptr<const CutExpressionRecipe>& expression) {
                if (expression) { result.push_back({expression.get(), piece.cut.before}); }
            };
            add(binding.activity);
            add(binding.residueCoordinate);
            if (piece.kind == EndpointKind::Barrier || executed) { continue; }
            if (maps.empty() || physicalMapNeedsCoordinate(*maps[i], 0)) { add(binding.ordinal); }
            for (auto [position, coordinate] : llvm::enumerate(binding.matchingCoordinates)) {
                if (maps.empty() || physicalMapNeedsCoordinate(*maps[i], invocationDepth + position + 1)) {
                    add(coordinate);
                }
            }
        }
        return result;
    }
    FailureOr<SmallVector<uint8_t>> liveSharedExpressions() const
    {
        SmallVector<uint8_t> live(recipe.shared.size(), 0);
        if (recipe.shared.empty()) { return live; }
        std::map<RegionExpressions::Id, SmallVector<std::size_t>> candidates;
        for (auto [index, shared] : llvm::enumerate(recipe.shared)) {
            candidates[shared.expression.expression].push_back(index);
        }
        DominanceInfo dominance(recipe.cuts.front().cut.before->getParentOfType<func::FuncOp>());
        auto pending = liveContextualExpressions();
        while (!pending.empty()) {
            const auto request = pending.pop_back_val();
            const auto* expression = request.first;
            auto* cut = request.second;
            for (auto id : cutExpressionSharedExpressions(*expression)) {
                auto found = candidates.find(id);
                if (found == candidates.end()) { return failure(); }
                auto selected = llvm::find_if(found->second, [&](std::size_t index) {
                    auto* source = recipe.shared[index].expression.cut;
                    return source == cut || dominance.properlyDominates(source, cut);
                });
                if (selected == found->second.end()) { return failure(); }
                if (live[*selected]) { continue; }
                live[*selected] = 1;
                const auto& shared = recipe.shared[*selected];
                pending.push_back({shared.binding.get(), shared.expression.cut});
            }
        }
        return live;
    }
};
// No loop/selector arithmetic is re-estimated: both consumers execute these
// exact scalar primitives with one local trip/ordinal state per leaf.
class NumericalFamilyScalars {
public:
    using Scalar = detail::ScalarToken;
    explicit NumericalFamilyScalars(detail::ScalarConstruction& construction) : construction(construction) {}
    void initializeTrips(scf::ForOp loop)
    {
        zero = construction.number(0);
        one = construction.number(1);
        const auto upper = construction.input(loop.getUpperBound()), lower = construction.input(loop.getLowerBound());
        const auto step = construction.input(loop.getStep());
        auto positive = construction.compare(arith::CmpIPredicate::sgt, upper, lower);
        auto difference = construction.binary<arith::SubIOp>(upper, lower);
        auto span = construction.select(positive, difference, zero);
        auto quotient = construction.binary<arith::DivUIOp>(span, step);
        auto remainder = construction.binary<arith::RemUIOp>(span, step);
        auto partial = construction.compare(arith::CmpIPredicate::ne, remainder, zero);
        auto extra = construction.select(partial, one, zero);
        trips = construction.binary<arith::AddIOp>(quotient, extra);
    }
    void initializeOrdinal(scf::ForOp loop)
    {
        auto offset = construction.binary<arith::SubIOp>(construction.input(loop.getInductionVar()),
            construction.input(loop.getLowerBound()));
        ordinal = construction.binary<arith::DivUIOp>(offset, construction.input(loop.getStep()));
    }
    void zeroOrdinal() { ordinal = construction.number(0); }
    Scalar boundary(const EndpointPiece& piece)
    {
        if (!piece.displacement) { return construction.boolean(true); }
        auto delay = construction.number(piece.displacement);
        if (piece.kind != EndpointKind::Set) {
            return construction.compare(arith::CmpIPredicate::uge, ordinal, delay);
        }
        auto remaining = construction.binary<arith::SubIOp>(trips, delay);
        auto hasTarget = construction.compare(arith::CmpIPredicate::ult, delay, trips);
        auto beforeEnd = construction.compare(arith::CmpIPredicate::ult, ordinal, remaining);
        return construction.binary<arith::AndIOp>(hasTarget, beforeEnd);
    }
    ContextualFamilyInputs<Scalar> piece(const FamilyEndpointRecipe& recipe, std::size_t index,
        const PhysicalEndpointMap* mapping, uint32_t recordBase, Scalar partnerPresent)
    {
        const auto& piece = recipe.pieces[index];
        const bool needMember = !mapping || physicalMapRequiresMember(*mapping);
        auto selector = needMember ? detail::constructFamilyExpressions(construction, *recipe.selectors[index]) :
            detail::FamilyScalarTokens{detail::constructFamilyPresence(construction, *recipe.selectors[index]), {}};
        ContextualFamilyInputs<Scalar> result;
        result.guard = construction.binary<arith::AndIOp>(selector.present, partnerPresent);
        if (piece.kind != EndpointKind::Barrier) {
            if (!mapping || physicalMapNeedsCoordinate(*mapping, 0)) {
                result.ordinal = ordinal;
                if (piece.kind == EndpointKind::Wait && piece.displacement) {
                    result.ordinal = construction.binary<arith::SubIOp>(
                        ordinal, construction.number(piece.displacement));
                }
            }
            auto member = selector.member;
            if (member && recordBase) {
                member = construction.binary<arith::AddIOp>(member, construction.number(recordBase));
            }
            result.members.push_back(member);
        }
        return result;
    }
private:
    detail::ScalarConstruction& construction;
    Scalar trips, ordinal, zero, one;
};
template<class Consumer> LogicalResult runFamilyPreparation(const FamilyEndpointRecipe& recipe,
    ArrayRef<std::shared_ptr<const PhysicalEndpointMap>> maps, uint64_t invocationDepth, bool executed,
    Consumer& consumer)
{
    bool needsOrdinals = maps.empty();
    for (auto [index, piece] : llvm::enumerate(recipe.pieces)) {
        needsOrdinals |= piece.displacement || (piece.kind != EndpointKind::Barrier &&
            (maps.empty() || physicalMapNeedsCoordinate(*maps[index], 0)));
    }
    if (recipe.outer && needsOrdinals) {
        auto loop = recipe.outer;
        if (failed(consumer.at(loop, [&]() { consumer.initializeTrips(loop); return success(); })) ||
            failed(consumer.at(&loop.getBody()->front(), [&]() {
                consumer.initializeOrdinal(loop); return success();
            }))) { return failure(); }
    }
    if (recipe.arena) { return ContextualFamilySchedule(recipe, maps, invocationDepth, executed).run(consumer); }
    for (const auto& cut : recipe.cuts) {
        if (cut.pieces.empty()) { continue; }
        if (failed(consumer.at(cut.cut.before, [&]() {
                if (!recipe.outer && llvm::any_of(cut.pieces, [&](auto index) {
                        return maps.empty() || (maps[index] && physicalMapNeedsCoordinate(*maps[index], 0));
                    })) { consumer.zeroOrdinal(); }
                std::map<std::pair<bool, uint64_t>, typename Consumer::Scalar> predicates;
                for (auto index : cut.pieces) {
                    const auto& piece = recipe.pieces[index];
                    auto [entry, added] = predicates.try_emplace(
                        std::make_pair(piece.kind == EndpointKind::Set, piece.displacement));
                    if (added) { entry->second = consumer.boundary(piece); }
                    if (failed(consumer.numericalPiece(index, entry->second))) { return failure(); }
                }
                return success();
            }))) { return failure(); }
    }
    return success();
}
class FamilyPreparer {
public:
    FamilyPreparer(
        const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared, uint32_t recordBase,
        const std::map<uint32_t, uint32_t>& namespaces, FamilyMaterialization& materialization,
        ArrayRef<std::shared_ptr<const PhysicalEndpointMap>> maps = {}, uint64_t invocationDepth = 0,
        bool executed = false)
        : recipe(recipe),
          prepared(prepared),
          recordBase(recordBase),
          namespaces(namespaces),
          materialization(materialization),
          maps(maps),
          invocationDepth(invocationDepth),
          executed(executed),
          builder(recipe.cuts.front().cut.before->getContext()),
          scalar(builder, recipe.cuts.front().cut.before->getLoc()), numericalScalars(scalar)
    {}
    LogicalResult run() { return runFamilyPreparation(recipe, maps, invocationDepth, executed, *this); }

private:
    const FamilyEndpointRecipe& recipe;
    PreparedLogicalPlan& prepared;
    uint32_t recordBase;
    const std::map<uint32_t, uint32_t>& namespaces;
    FamilyMaterialization& materialization;
    ArrayRef<std::shared_ptr<const PhysicalEndpointMap>> maps;
    uint64_t invocationDepth;
    bool executed;
    OpBuilder builder;
    detail::ScalarConstruction scalar;
    NumericalFamilyScalars numericalScalars;
    template<class Consumer> friend LogicalResult runFamilyPreparation(const FamilyEndpointRecipe&,
        ArrayRef<std::shared_ptr<const PhysicalEndpointMap>>, uint64_t, bool, Consumer&);
    friend class ContextualFamilySchedule;
    using Scalar = Value;
    template<class Action> LogicalResult at(Operation* cut, Action&& action)
    {
        Block code;
        builder.setInsertionPointToEnd(&code);
        scalar.setLocation(cut->getLoc());
        if (failed(action())) { return failure(); }
        adopt(code, cut);
        return success();
    }
    FailureOr<Value> emit(const CutExpressionRecipe& binding, Operation* cut,
        const FamilyEndpointRecipe* environment)
    {
        return materialization.emit(*recipe.arena, binding, builder, cut, environment);
    }
    void bind(Operation* cut, RegionExpressions::Id expression, Value value)
    {
        materialization.bind(*recipe.arena, recipe, cut, expression, value);
    }
    void publish() { materialization.publish(*recipe.arena, recipe); }
    Value number(uint32_t record, Operation*) { return scalar.number(uint64_t(record) + recordBase).value; }
    void initializeTrips(scf::ForOp loop) { numericalScalars.initializeTrips(loop); }
    void initializeOrdinal(scf::ForOp loop) { numericalScalars.initializeOrdinal(loop); }
    void zeroOrdinal() { numericalScalars.zeroOrdinal(); }
    Value boundary(const EndpointPiece& piece) { return numericalScalars.boundary(piece).value; }
    LogicalResult numerical(std::size_t i)
    {
        if (!recipe.outer && (maps.empty() || (maps[i] && physicalMapNeedsCoordinate(*maps[i], 0)))) {
            zeroOrdinal();
        }
        return numericalPiece(i, boundary(recipe.pieces[i]));
    }
    LogicalResult endpoint(std::size_t i, const ContextualFamilyInputs<Value>& inputs)
    {
        const auto& piece = recipe.pieces[i];
        const auto& binding = recipe.bindings[i];
        const bool local = piece.kind == EndpointKind::Barrier;
        auto kind = local ? LogicalCommandKind::Barrier :
            (piece.kind == EndpointKind::Set ? LogicalCommandKind::Set : LogicalCommandKind::Wait);
        const auto record = piece.members.front().record + recordBase;
        PreparedLogicalEndpoint endpoint{piece.cut.before, kind, piece.sourcePipe, piece.targetPipe,
            namespaces.empty() ? piece.namespaceId + recordBase : namespaces.at(record), inputs.guard, inputs.ordinal};
        endpoint.records = {record};
        endpoint.memberCoordinates = inputs.members;
        if (!maps.empty()) { endpoint.physicalMapping = maps[i]; }
        retainEndpointCoordinates(endpoint, piece.members.front().coordinates);
        if (binding.residueCoordinate) {
            if (endpoint.domain.empty()) { endpoint.domain.emplace_back(); }
            for (auto& box : endpoint.domain) {
                box.push_back({inputs.residue, binding.modulus, static_cast<int64_t>(binding.residue)});
            }
        }
        endpoint.piece = prepared.endpoints.size();
        prepared.endpoints.push_back(std::move(endpoint));
        return success();
    }
    void adopt(Block& code, Operation* cut)
    {
        if (code.empty()) { return; }
        auto& target = prepared.addPreparation(cut);
        target.getOperations().splice(target.end(), code.getOperations());
    }
    LogicalResult numericalPiece(std::size_t index, Value partnerPresent)
    {
        const auto& piece = recipe.pieces[index];
        const auto mapping = maps.empty() ? nullptr : maps[index];
        auto inputs = numericalScalars.piece(recipe, index, mapping.get(), recordBase, scalar.input(partnerPresent));
        const auto kind = piece.kind == EndpointKind::Barrier ? LogicalCommandKind::Barrier :
            (piece.kind == EndpointKind::Set ? LogicalCommandKind::Set : LogicalCommandKind::Wait);
        PreparedLogicalEndpoint endpoint{piece.cut.before, kind, piece.sourcePipe, piece.targetPipe,
            namespaces.empty() ? piece.namespaceId + recordBase :
                namespaces.at(piece.members.front().record + recordBase), inputs.guard.value, inputs.ordinal.value, {}};
        for (auto member : inputs.members) { endpoint.memberCoordinates.push_back(member.value); }
        endpoint.physicalMapping = mapping;
        for (const auto& member : piece.members) {
            retainEndpointCoordinates(endpoint, member.coordinates);
            endpoint.records.push_back(member.record + recordBase);
        }
        endpoint.piece = static_cast<int64_t>(prepared.endpoints.size());
        prepared.endpoints.push_back(std::move(endpoint));
        return success();
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
            for (auto coordinate : family.matchingCoordinates) {
                auto planned = prepare(coordinate, false);
                if (failed(planned)) { return failure(); }
                binding.matchingCoordinates.push_back(std::move(*planned));
            }
        } else if (!family.matchingCoordinates.empty()) {
            return failure();
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
            piece.members.size() != 1 || (!outer && numerical[i] && piece.displacement) ||
            (numerical[i] && !families[i].matchingCoordinates.empty())) { return failure(); }
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
detail::ScalarToken constructFamilyActivity(detail::ScalarConstruction& construction,
    detail::ScalarToken condition, detail::ScalarToken guard)
{
    return construction.select(condition, guard, construction.boolean(false));
}
template<class Endpoint, class Consumer> LogicalResult applyFamilyContexts(Endpoint& endpoint,
    std::size_t path, ArrayRef<FamilyContextPath> paths, Consumer& consumer)
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
            endpoint.memberCoordinates.insert(endpoint.memberCoordinates.begin() + 1, decltype(endpoint.guard){});
            continue;
        }
        auto value = consumer.lower(*item.recipe, endpoint.before);
        if (failed(value)) { return failure(); }
        if (item.recipe->kind == FamilyContextKind::Invocation) {
            if (endpoint.memberCoordinates.empty()) { return failure(); }
            endpoint.memberCoordinates.insert(endpoint.memberCoordinates.begin() + 1, *value);
        } else {
            endpoint.guard = consumer.mask(*value, endpoint.guard, endpoint.before);
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
        return applyFamilyContexts(endpoint, path, paths, *this);
    }

private:
    template<class Endpoint, class Consumer> friend LogicalResult applyFamilyContexts(
        Endpoint&, std::size_t, ArrayRef<FamilyContextPath>, Consumer&);
    Value mask(Value condition, Value guard, Operation* cut)
    {
        // Append after both the leaf guard and any cached wrapper root.
        OpBuilder builder(cut->getContext());
        builder.setInsertionPointToEnd(&plan.addPreparation(cut));
        detail::ScalarConstruction construction(builder, cut->getLoc());
        return constructFamilyActivity(construction, construction.input(condition), construction.input(guard)).value;
    }
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
struct CountedFamilyEndpoint {
    Operation* before = nullptr;
    LogicalCommandKind kind = LogicalCommandKind::Barrier;
    uint32_t sourcePipe = 0, targetPipe = 0, record = 0;
    detail::ScalarToken guard, identity;
    SmallVector<uint32_t> records;
    struct Constraint { detail::ScalarToken coordinate; uint64_t modulus; int64_t value; };
    SmallVector<SmallVector<Constraint>> domain;
    SmallVector<detail::ScalarToken> memberCoordinates;
    std::shared_ptr<const PhysicalEndpointMap> physicalMapping;
};
// One counter owns every token across all original cuts and shared leaf
// environments. No generated operation, root recipe or allocation is built.
class FamilyCounter {
public:
    using Scalar = detail::ScalarToken;
    FamilyCounter(const FamilyEndpointRecipe& recipe, uint32_t recordBase,
        ArrayRef<std::shared_ptr<const PhysicalEndpointMap>> maps,
        FamilyConstruction& materialization, detail::ScalarConstruction& construction,
        const std::map<uint32_t, uint32_t>& namespaces)
        : recipe(recipe), recordBase(recordBase), maps(maps), materialization(materialization),
          construction(construction), numericalScalars(construction), namespaces(namespaces) {}
    template<class Action> LogicalResult at(Operation* cut, Action&& action)
    {
        construction.setInsertionCut(cut);
        return action();
    }
    FailureOr<Scalar> emit(const CutExpressionRecipe& binding, Operation* cut,
        const FamilyEndpointRecipe* environment)
    {
        return materialization.emit(*recipe.arena, binding, cut, environment);
    }
    void bind(Operation* cut, RegionExpressions::Id expression, Scalar value)
    {
        materialization.bind(*recipe.arena, recipe, cut, expression, value);
    }
    void publish() { materialization.publish(*recipe.arena, recipe); }
    Scalar number(uint32_t record, Operation*) { return construction.number(uint64_t(record) + recordBase); }
    void initializeTrips(scf::ForOp loop) { numericalScalars.initializeTrips(loop); }
    void initializeOrdinal(scf::ForOp loop) { numericalScalars.initializeOrdinal(loop); }
    void zeroOrdinal() { numericalScalars.zeroOrdinal(); }
    Scalar boundary(const EndpointPiece& piece) { return numericalScalars.boundary(piece); }
    LogicalResult numerical(std::size_t i)
    {
        if (!recipe.outer && (maps.empty() || (maps[i] && physicalMapNeedsCoordinate(*maps[i], 0)))) {
            zeroOrdinal();
        }
        return numericalPiece(i, boundary(recipe.pieces[i]));
    }
    LogicalResult numericalPiece(std::size_t index, Scalar partnerPresent)
    {
        return endpoint(index, numericalScalars.piece(recipe, index,
            maps.empty() ? nullptr : maps[index].get(), recordBase, partnerPresent));
    }
    LogicalResult endpoint(std::size_t index, const ContextualFamilyInputs<Scalar>& inputs)
    {
        const auto& piece = recipe.pieces[index];
        CountedFamilyEndpoint endpoint;
        endpoint.before = piece.cut.before;
        endpoint.kind = piece.kind == EndpointKind::Barrier ? LogicalCommandKind::Barrier :
            (piece.kind == EndpointKind::Set ? LogicalCommandKind::Set : LogicalCommandKind::Wait);
        endpoint.sourcePipe = piece.sourcePipe; endpoint.targetPipe = piece.targetPipe;
        const auto record = piece.members.front().record + recordBase;
        endpoint.record = namespaces.empty() ? piece.namespaceId + recordBase : namespaces.at(record);
        endpoint.guard = inputs.guard; endpoint.identity = inputs.ordinal;
        endpoint.memberCoordinates = inputs.members;
        endpoint.physicalMapping = maps.empty() ? nullptr : maps[index];
        for (const auto& member : piece.members) {
            endpoint.records.push_back(member.record + recordBase);
            if (member.coordinates.empty()) { continue; }
            SmallVector<CountedFamilyEndpoint::Constraint> box;
            for (auto coordinate : member.coordinates) {
                box.push_back({construction.input(coordinate.loop.getInductionVar()), 0, coordinate.induction});
            }
            endpoint.domain.push_back(std::move(box));
        }
        const bool contextual = recipe.arena && (recipe.selectors.empty() || !recipe.selectors[index]);
        if (contextual && recipe.bindings[index].residueCoordinate) {
            const auto& binding = recipe.bindings[index];
            if (endpoint.domain.empty()) { endpoint.domain.emplace_back(); }
            for (auto& box : endpoint.domain) {
                box.push_back({inputs.residue, binding.modulus, static_cast<int64_t>(binding.residue)});
            }
        }
        endpoints.push_back(std::move(endpoint));
        return success();
    }
    SmallVector<CountedFamilyEndpoint, 0> endpoints;
private:
    const FamilyEndpointRecipe& recipe;
    uint32_t recordBase;
    ArrayRef<std::shared_ptr<const PhysicalEndpointMap>> maps;
    FamilyConstruction& materialization;
    detail::ScalarConstruction& construction;
    NumericalFamilyScalars numericalScalars;
    const std::map<uint32_t, uint32_t>& namespaces;
};
class FamilyContextCounter {
public:
    FamilyContextCounter(FamilyConstruction& materialization, detail::ScalarConstruction& construction)
        : materialization(materialization), construction(construction) {}
    FailureOr<detail::ScalarToken> lower(const FamilyContextRecipe& recipe, Operation* cut)
    {
        auto found = recipe.cuts.find(cut);
        if (found == recipe.cuts.end()) { return failure(); }
        construction.setInsertionCut(cut);
        return materialization.emit(*recipe.arena, *found->second, cut);
    }
    detail::ScalarToken mask(detail::ScalarToken condition, detail::ScalarToken guard, Operation* cut)
    {
        construction.setInsertionCut(cut);
        return constructFamilyActivity(construction, condition, guard);
    }
private:
    FamilyConstruction& materialization;
    detail::ScalarConstruction& construction;
};
// Only arithmetic preparation admitted by insertion is represented. Original
// values remain external identities; generated values retain exact properties,
// types, result numbers and original insertion cuts in one comparison owner.
class FamilyNormalizationCounter {
public:
    using Scalar = detail::ScalarToken;
    FamilyNormalizationCounter(detail::ScalarConstruction& construction, const detail::ScalarDeduplication& aliases)
        : construction(construction), aliases(aliases)
    {
        for (std::size_t i = 0; i < construction.trace().size(); ++i) {
            for (auto result : construction.trace()[i].results) { definitions[result] = i; }
        }
    }
    Scalar canonical(Scalar value) const
    {
        return value ? construction.lookup(aliases.canonical(value.id)) : Scalar{};
    }
    template<class Action> LogicalResult at(Operation* cut, Action action)
    {
        construction.setInsertionCut(cut);
        return action();
    }
    Scalar original(Value value) { return construction.input(value); }
    bool less(Scalar a, Scalar b) const { return a.id < b.id; }
    detail::ScalarGuardPattern<Scalar> pattern(Scalar value)
    {
        value = canonical(value);
        if (value.value) {
            auto* operation = value.value.getDefiningOp();
            if (!operation) { return {}; }
            SmallVector<Scalar> operands;
            for (auto operand : operation->getOperands()) { operands.push_back(original(operand)); }
            return detail::scalarGuardPattern<Scalar>(operation->getName().getStringRef(),
                operation->getPropertiesAsAttribute(), operands);
        }
        auto found = definitions.find(value.id);
        if (found == definitions.end()) { return {}; }
        const auto& instruction = construction.trace()[found->second];
        SmallVector<Scalar> operands;
        for (auto operand : instruction.operands) { operands.push_back(canonical(construction.lookup(operand))); }
        return detail::scalarGuardPattern<Scalar>(instruction.name.getValue(), instruction.properties, operands);
    }
    Scalar select(Scalar condition, Scalar yes, Scalar no) { return construction.select(condition, yes, no); }
    Scalar lor(Scalar a, Scalar b) { return construction.binary<arith::OrIOp>(a, b); }
    Scalar land(Scalar a, Scalar b) { return construction.binary<arith::AndIOp>(a, b); }
    std::pair<Scalar, Scalar> selection(const FamilyExpressionRecipe& recipe, bool member)
    {
        if (!member) { return {detail::constructFamilyPresence(construction, recipe), {}}; }
        auto selected = detail::constructFamilyExpressions(construction, recipe);
        return {selected.present, selected.member};
    }
private:
    detail::ScalarConstruction& construction;
    const detail::ScalarDeduplication& aliases;
    llvm::DenseMap<uint64_t, std::size_t> definitions;
};
// One request directory for the actual prepared WAIT guards in both routes.
// Pending evidence proves multiplicity only; it never substitutes for this guard.
template<class Endpoint>
FailureOr<std::map<std::pair<uint32_t, Operation*>, decltype(Endpoint{}.guard)>> pendingCleanupGuards(
    ArrayRef<EndpointFamily> families, ArrayRef<Endpoint> endpoints)
{
    using Scalar = decltype(Endpoint{}.guard);
    std::map<std::pair<uint32_t, Operation*>, Scalar> guards;
    for (const auto& family : families) {
        if (!family.targetChoices || !family.targetChoices->pendingMaterialization()) { continue; }
        if (!family.targetChoices->validPendingSelection() || family.members.size() != 1) { return failure(); }
        if (!guards.emplace(std::make_pair(family.members.front().record, family.targetCut.before),
                            Scalar{}).second) { return failure(); }
    }
    if (guards.empty()) { return guards; }
    for (const auto& endpoint : endpoints) {
        if (endpoint.kind != LogicalCommandKind::Wait) { continue; }
        for (auto record : endpoint.records) {
            auto found = guards.find({record, endpoint.before});
            if (found == guards.end()) { continue; }
            if (found->second) { return failure(); }
            found->second = endpoint.guard;
        }
    }
    for (const auto& [key, guard] : guards) {
        if (!guard) { return failure(); }
    }
    return guards;
}
// Reproduce the post-dedup owned preparation graph without scalar emission.
// Non-SSA metadata keeps operations alive, but does not make an eager use a
// guarded one. Only actual physical-map inputs are command consumers.
void countSharedSelectorGuards(detail::ScalarConstruction& construction,
    const detail::ScalarDeduplication& aliases, ArrayRef<CountedFamilyEndpoint> endpoints,
    ArrayRef<detail::ScalarToken> evidence)
{
    SmallVector<detail::SelectorUse> uses;
    llvm::DenseSet<uint64_t> roots;
    auto root = [&](detail::ScalarToken token) { if (token) { roots.insert(aliases.canonical(token.id)); } };
    auto use = [&](detail::ScalarToken token, Operation* cut, detail::ScalarToken guard) {
        if (token) { uses.push_back({aliases.canonical(token.id), aliases.canonical(guard.id), cut}); }
    };
    for (auto token : evidence) { root(token); use(token, nullptr, {}); }
    for (const auto& endpoint : endpoints) {
        root(endpoint.guard); root(endpoint.identity);
        for (auto member : endpoint.memberCoordinates) { root(member); }
        for (const auto& box : endpoint.domain) {
            for (auto item : box) { root(item.coordinate); use(item.coordinate, nullptr, {}); }
        }
        use(endpoint.guard, endpoint.before, {});
        if (!endpoint.physicalMapping) { continue; }
        if (physicalMapNeedsCoordinate(*endpoint.physicalMapping, 0)) {
            use(endpoint.identity, endpoint.before, endpoint.guard);
        }
        for (auto entry : llvm::enumerate(endpoint.memberCoordinates)) {
            const bool live = entry.index() ? physicalMapNeedsCoordinate(*endpoint.physicalMapping, entry.index()) :
                physicalMapRequiresMember(*endpoint.physicalMapping);
            if (live) { use(entry.value(), endpoint.before, endpoint.guard); }
        }
    }
    llvm::DenseSet<std::size_t> redundant(aliases.redundant.begin(), aliases.redundant.end());
    SmallVector<detail::ScalarInstruction, 0> source;
    for (auto entry : llvm::enumerate(construction.trace())) {
        if (redundant.count(entry.index())) { continue; }
        auto instruction = entry.value();
        for (auto& operand : instruction.operands) { operand = aliases.canonical(operand); }
        source.push_back(std::move(instruction));
    }
    std::vector<bool> live(source.size(), false);
    for (std::size_t i = source.size(); i-- > 0;) {
        live[i] = llvm::any_of(source[i].results, [&](uint64_t id) { return roots.count(id); });
        if (live[i]) { for (auto id : source[i].operands) { roots.insert(id); } }
    }
    SmallVector<detail::ScalarInstruction, 0> retained;
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (live[i]) { retained.push_back(std::move(source[i])); }
    }
    struct Counter {
        detail::ScalarConstruction& construction;
        detail::ScalarToken lookup(uint64_t id) { return construction.lookup(id); }
        detail::ScalarToken lor(detail::ScalarToken a, detail::ScalarToken b)
        { return construction.binary<arith::OrIOp>(a, b); }
        void begin(const detail::SelectorGuardGroup&, detail::ScalarToken) {}
        void zero(Type type)
        {
            if (type.isIndex()) { construction.number(0); }
            else { construction.integer(APInt(cast<IntegerType>(type).getWidth(), 0)); }
        }
        void finish(const detail::SelectorGuardGroup&) {}
    } counter{construction};
    const auto groups = detail::planSelectorGuards(retained, uses);
    for (const auto& group : groups) {
        construction.setInsertionCut(group.cut);
        detail::constructSelectorGuardGroup(group, counter);
    }
}
std::optional<uint64_t> ownedFamilyConstruction(
    const PhysicalFamilyOutput& output, bool sharedSelectors, bool logical = false)
{
    if (!output.protocol || !output.protocol->normalized || !output.concrete.empty()) { return std::nullopt; }
    const auto& inventory = *output.protocol->normalized;
    Operation* first = nullptr;
    for (const auto& contribution : inventory.stream) {
        if (!contribution.leaf) { return std::nullopt; }
        if (!first && !contribution.leaf->pieces.empty()) { first = contribution.leaf->pieces.front().cut.before; }
    }
    if (!first) { return uint64_t(0); }
    detail::ScalarConstruction construction;
    construction.enableTrace(first->getContext());
    FamilyConstruction materialization(construction);
    FamilyContextCounter contexts(materialization, construction);
    struct Plan {
        SmallVector<CountedFamilyEndpoint, 0> endpoints;
        ArrayRef<EndpointFamily> families;
        bool independentPieces = true;
    } plan;
    plan.families = inventory.families;
    const auto namespaces = inventory.translateNamespaces ? familyRecordNamespaces(inventory.families) :
        std::map<uint32_t, uint32_t>{};
    for (const auto& contribution : inventory.stream) {
        const auto& leaf = *contribution.leaf;
        ArrayRef<std::shared_ptr<const PhysicalEndpointMap>> maps;
        if (!logical) {
            auto found = output.deferred.find(&contribution);
            if (found == output.deferred.end() || found->second.size() != leaf.pieces.size()) { return std::nullopt; }
            maps = found->second;
            for (auto [piece, mapping] : llvm::zip(leaf.pieces, maps)) {
                if (!mapping && piece.kind != EndpointKind::Barrier) { return std::nullopt; }
            }
        }
        if (leaf.pieces.empty()) { continue; }
        FamilyCounter counter(leaf, contribution.recordBase, maps, materialization, construction, namespaces);
        const auto depth = contribution.contextPath == SIZE_MAX ? 0 :
            inventory.contexts[contribution.contextPath].invocationDepth;
        if (failed(runFamilyPreparation(leaf, maps, depth, false, counter))) { return std::nullopt; }
        // This leaf's wrappers run before the next leaf can consume the shared
        // arena/cut cache, exactly as in actual materialization.
        for (auto& endpoint : counter.endpoints) {
            if (failed(applyFamilyContexts(endpoint, contribution.contextPath, inventory.contexts, contexts))) {
                return std::nullopt;
            }
            plan.endpoints.push_back(std::move(endpoint));
        }
    }
    if (!construction.validTrace()) { return std::nullopt; }
    auto cleanupGuards = pendingCleanupGuards<CountedFamilyEndpoint>(inventory.families, plan.endpoints);
    if (failed(cleanupGuards)) { return std::nullopt; }
    detail::CountedCleanupPredicate cleanupView(construction);
    for (const auto& family : inventory.families) {
        if (family.sourceChoices && family.sourceChoices->pendingMaterialization()) { return std::nullopt; }
        if (!family.targetChoices || !family.targetChoices->pendingMaterialization()) { continue; }
        auto cut = family.targetChoices->cuts().front().before;
        auto predicate = cleanupGuards->at({family.members.front().record, family.targetCut.before});
        if (!detail::singletonCleanupConsumer(cut, cleanupView, predicate, cut->getParentOfType<func::FuncOp>())) {
            return std::nullopt;
        }
    }
    SmallVector<detail::ScalarToken> selectorEvidence;
    // The materializer qualifies pending cleanup against these original WAIT
    // guards before endpoint compaction. Later guards cannot replace the proof
    // owner's retained Value, even when the old guard also has command users.
    for (const auto& entry : *cleanupGuards) { selectorEvidence.push_back(entry.second); }
    for (const auto& family : inventory.families) {
        for (const auto& choices : {family.sourceChoices, family.targetChoices}) {
            if (choices && choices->consumerPredicate()) {
                selectorEvidence.push_back(construction.input(choices->consumerPredicate()));
            }
        }
    }
    const auto aliases = detail::deduplicateScalarInstructions(construction.trace());
    FamilyNormalizationCounter normalizer(construction, aliases);
    for (auto& endpoint : plan.endpoints) {
        endpoint.guard = normalizer.canonical(endpoint.guard);
        endpoint.identity = normalizer.canonical(endpoint.identity);
        for (auto& member : endpoint.memberCoordinates) { member = normalizer.canonical(member); }
        for (auto& box : endpoint.domain) {
            for (auto& constraint : box) { constraint.coordinate = normalizer.canonical(constraint.coordinate); }
        }
    }
    if (failed(detail::coalesceEndpointAlternatives(plan, normalizer))) { return std::nullopt; }
    if (failed(detail::coalesceDomainAlternatives(plan, normalizer, true)) || !construction.validTrace()) {
        return std::nullopt;
    }
    if (sharedSelectors) {
        countSharedSelectorGuards(construction, aliases, plan.endpoints, selectorEvidence);
    }
    std::map<Operation*, SmallVector<const CountedFamilyEndpoint*>> cuts;
    for (const auto& endpoint : plan.endpoints) { cuts[endpoint.before].push_back(&endpoint); }
    for (const auto& [cut, endpoints] : cuts) {
        construction.setInsertionCut(cut);
        (void)detail::combineBarrierGuards<CountedFamilyEndpoint>(endpoints,
            [&](auto a, auto b) { return normalizer.lor(a, b); });
    }
    uint64_t mapsCost = 0;
    for (const auto& endpoint : plan.endpoints) {
        if (!endpoint.physicalMapping) { continue; }
        const auto cost = physicalMapConstructionOperations(*endpoint.physicalMapping);
        if (!cost || *cost > uint64_t(INT64_MAX) - mapsCost) { return std::nullopt; }
        mapsCost += *cost;
    }
    // Deduplication and dead-root removal do not refund operations already
    // constructed. Compaction adds its actual schedule; only retained command
    // sites subsequently construct physical maps.
    const auto upstream = construction.operations();
    if (!upstream || *upstream > uint64_t(INT64_MAX) - mapsCost) { return std::nullopt; }
    return *upstream + mapsCost;
}
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
llvm::json::Array profileFamilyControlRoots(const FamilyEndpointRecipe& recipe)
{
    SmallVector<FamilyPart> leaves;
    SmallVector<FamilyContextPath> contexts;
    if (failed(collectFamilyParts(recipe, leaves, contexts))) { return {}; }
    struct Roots {
        SmallVector<RegionExpressions::Id> guards, selectors;
        std::set<std::pair<const FamilyEndpointRecipe*, std::size_t>> environments;
        std::set<Operation*> cuts;
        struct Origin { Operation* cut; EndpointKind kind; std::size_t context, piece; };
        std::map<RegionExpressions::Id, Origin> origins;
    };
    std::map<const RegionExpressions*, Roots> groups;
    for (const auto& part : leaves) {
        const auto& leaf = *part.recipe;
        if (!leaf.arena) { continue; }
        auto& group = groups[leaf.arena.get()];
        group.environments.emplace(&leaf, part.contextPath);
        for (auto [index, binding] : llvm::enumerate(leaf.bindings)) {
            const auto pieceIndex = index;
            const auto& piece = leaf.pieces[index];
            auto remember = [&](const auto& scalar) {
                if (scalar) { group.origins.try_emplace(cutExpressionRoot(*scalar),
                    Roots::Origin{piece.cut.before, piece.kind, part.contextPath, pieceIndex}); }
            };
            if (binding.activity) { group.guards.push_back(cutExpressionRoot(*binding.activity)); }
            remember(binding.activity);
            for (const auto* scalar : {&binding.ordinal, &binding.residueCoordinate}) {
                if (*scalar) { group.selectors.push_back(cutExpressionRoot(**scalar)); remember(*scalar); }
            }
            for (const auto& scalar : binding.matchingCoordinates) {
                if (scalar) { group.selectors.push_back(cutExpressionRoot(*scalar)); remember(scalar); }
            }
        }
        for (const auto& cut : leaf.cuts) { group.cuts.insert(cut.cut.before); }
    }
    for (const auto& context : contexts) {
        if (!context.recipe->arena) { continue; }
        auto& group = groups[context.recipe->arena.get()];
        for (const auto& entry : context.recipe->cuts) {
            group.cuts.insert(entry.first);
            auto& roots = context.recipe->kind == FamilyContextKind::Activity ? group.guards : group.selectors;
            roots.push_back(cutExpressionRoot(*entry.second));
        }
    }
    llvm::json::Array result;
    for (const auto& [arena, group] : groups) {
        auto report = detail::RegionConstructionAccess::profileRoots(*arena, group.guards, group.selectors);
        report["arena_nodes"] = arena->size();
        report["binding_environments"] = group.environments.size(); report["original_cuts"] = group.cuts.size();
        report["scope"] = "structural-arena-sharing-not-placement";
        llvm::json::Array origins;
        if (auto* largest = report.getArray("largest_expression_ids")) {
            for (const auto& entry : *largest) {
                const auto id = entry.getAsInteger();
                if (!id || !group.origins.count(*id)) { origins.push_back(nullptr); continue; }
                const auto& origin = group.origins.at(*id);
                std::string location;
                llvm::raw_string_ostream stream(location);
                if (origin.cut) { origin.cut->getLoc().print(stream); }
                llvm::json::Object item{{"expression", *id}, {"location", stream.str()},
                    {"cut_operation", origin.cut ? origin.cut->getName().getStringRef() : StringRef("block-end")},
                    {"command_kind", origin.kind == EndpointKind::Set ? "set" :
                        (origin.kind == EndpointKind::Wait ? "wait" : "barrier")},
                    {"terminal_cut", origin.cut && isa<func::ReturnOp>(origin.cut)}, {"piece", origin.piece}};
                if (origin.context != SIZE_MAX) { item["context_path"] = origin.context; }
                origins.push_back(std::move(item));
            }
        }
        report["largest_root_origins"] = std::move(origins);
        result.push_back(std::move(report));
    }
    return result;
}
FailureOr<std::map<uint32_t, FamilyMatchingShape>> familyMatchingShapes(const FamilyEndpointRecipe& recipe)
{
    SmallVector<FamilyPart> leaves;
    SmallVector<FamilyContextPath> contexts;
    if (failed(collectFamilyParts(recipe, leaves, contexts))) { return failure(); }
    std::map<uint32_t, FamilyMatchingShape> result;
    for (const auto& part : leaves) {
        const uint64_t depth = part.contextPath == SIZE_MAX ? 0 : contexts[part.contextPath].invocationDepth;
        if (depth == UINT64_MAX) { return failure(); }
        for (auto [position, piece] : llvm::enumerate(part.recipe->pieces)) {
            const auto extra = part.recipe->arena ? part.recipe->bindings[position].matchingCoordinates.size() : 0;
            if (extra > UINT64_MAX - depth - 1) { return failure(); }
            const auto arity = depth + 1 + extra;
            for (const auto& member : piece.members) {
                if (member.record > UINT32_MAX - part.recordBase) { return failure(); }
                auto& shape = result[member.record + part.recordBase];
                if (piece.kind == EndpointKind::Barrier) { shape.hasLocal = true; continue; }
                if ((shape.hasSet || shape.hasWait) && shape.coordinateCount != arity) { return failure(); }
                shape.coordinateCount = arity;
                shape.hasSet |= piece.kind == EndpointKind::Set;
                shape.hasWait |= piece.kind == EndpointKind::Wait;
            }
        }
    }
    return result;
}
FailureOr<std::map<uint32_t, uint64_t>> familyMatchingArities(const FamilyEndpointRecipe& recipe)
{
    auto shapes = familyMatchingShapes(recipe);
    if (failed(shapes)) { return failure(); }
    std::map<uint32_t, uint64_t> result;
    for (const auto& [record, shape] : *shapes) {
        if (shape.hasSet || shape.hasWait) { result.emplace(record, shape.coordinateCount); }
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
            key.push_back(binding.matchingCoordinates.size());
            for (const auto& coordinate : binding.matchingCoordinates) {
                appendCutExpressionKey(*coordinate, key);
            }
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
    inventory->families = plan.families;
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
    // Original records define namespaces for direct producers as well as
    // composed leaves; recipe-local labels are not separate protocol domains.
    inventory->translateNamespaces = bool(original->child) || !plan.families.empty();
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
FailureOr<NumericalFamilyRecipe> numericalFamilyRecipe(PreparedLogicalPlan& plan)
{
    if (plan.privateState || plan.completionOwner || plan.physicalOutput || plan.executedOutput ||
        plan.nestedIdentities ||
        !plan.endpoints.empty() || !plan.preparation.empty() || !plan.familyRecipe ||
        failed(normalizeLogicalFamilies(plan))) { return failure(); }
    const auto& inventory = plan.familyRecipe->normalized;
    if (!inventory || !inventory->contexts.empty() || inventory->stream.size() != 1 ||
        !inventory->stream.front().leaf) { return failure(); }
    const auto& contribution = inventory->stream.front();
    const auto& leaf = *contribution.leaf;
    if (leaf.arena || !leaf.shared.empty() || !numericalLeaf(leaf)) { return failure(); }
    NumericalFamilyRecipe result{leaf.outer, leaf.pieces};
    const auto namespaces = inventory->translateNamespaces ? familyRecordNamespaces(plan.families) :
        std::map<uint32_t, uint32_t>{};
    for (auto& piece : result.pieces) {
        piece.namespaceId += contribution.recordBase;
        for (auto& member : piece.members) { member.record += contribution.recordBase; }
        if (!namespaces.empty()) { piece.namespaceId = namespaces.at(piece.members.front().record); }
    }
    return result;
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> restoreNumericalFamilyRecipe(
    const NumericalFamilyRecipe& source)
{
    if (source.pieces.empty()) { return failure(); }
    auto recipe = std::make_shared<FamilyEndpointRecipe>();
    recipe->outer = source.outer;
    recipe->pieces = source.pieces;
    recipe->pieceCount = source.pieces.size();
    std::map<Operation*, std::size_t> cuts;
    for (const auto& piece : source.pieces) {
        if (!piece.cut.before || piece.cut.block != piece.cut.before->getBlock() || piece.members.empty() ||
            (!source.outer && piece.displacement)) { return failure(); }
        auto [cut, added] = cuts.emplace(piece.cut.before, recipe->cuts.size());
        if (added) { recipe->cuts.push_back({piece.cut, {}}); }
        recipe->cuts[cut->second].pieces.push_back(recipe->selectors.size());
        auto selection = buildPieceSelector(piece);
        if (!selection.error.empty()) { return failure(); }
        recipe->selectors.push_back(std::move(selection.recipe));
        recipe->maximumRecord = std::max(recipe->maximumRecord, piece.namespaceId);
        for (const auto& member : piece.members) {
            recipe->maximumRecord = std::max(recipe->maximumRecord, member.record);
        }
    }
    for (auto& cut : recipe->cuts) {
        llvm::stable_sort(cut.pieces, [&](auto a, auto b) {
            return recipe->pieces[a].order < recipe->pieces[b].order;
        });
    }
    return std::shared_ptr<const FamilyEndpointRecipe>(std::move(recipe));
}
FailureOr<std::vector<DeferredFamilyNode>> deferredFamilyRecipe(PreparedLogicalPlan& plan)
{
    if (plan.privateState || plan.completionOwner || plan.physicalOutput || plan.executedOutput ||
        !plan.endpoints.empty() || !plan.preparation.empty() || !plan.familyRecipe ||
        failed(normalizeLogicalFamilies(plan))) { return failure(); }
    auto original = familyRecipeForComposition(plan.familyRecipe);
    std::vector<DeferredFamilyNode> result;
    std::map<const FamilyEndpointRecipe*, int64_t> positions;
    SmallVector<std::pair<const FamilyEndpointRecipe*, bool>> pending{{original.get(), false}};
    while (!pending.empty()) {
        auto [recipe, ready] = pending.pop_back_val();
        if (positions.count(recipe)) { continue; }
        if (!ready) {
            pending.push_back({recipe, true});
            if (recipe->previous) { pending.push_back({recipe->previous.get(), false}); }
            if (recipe->child) { pending.push_back({recipe->child.get(), false}); }
            continue;
        }
        if (recipe->normalized || recipe->endpointIndex) { return failure(); }
        DeferredFamilyNode node;
        if (recipe->child) {
            node.child = positions.at(recipe->child.get());
            if (recipe->context) {
                if (recipe->previous || recipe->recordBase) { return failure(); }
                node.kind = recipe->context->kind == FamilyContextKind::Invocation ?
                    DeferredFamilyNode::Kind::Invocation : DeferredFamilyNode::Kind::Activity;
                node.arena = recipe->context->arena;
                for (const auto& [cut, binding] : recipe->context->cuts) {
                    node.contextCuts.emplace(cut, cutExpressionRoot(*binding));
                }
            } else {
                node.kind = DeferredFamilyNode::Kind::Append;
                node.recordBase = recipe->recordBase;
                if (recipe->previous) { node.previous = positions.at(recipe->previous.get()); }
            }
        } else {
            if (recipe->previous || recipe->pieces.empty()) { return failure(); }
            node.arena = recipe->arena;
            node.outer = recipe->outer;
            node.pieces = recipe->pieces;
            for (const auto& shared : recipe->shared) { node.shared.push_back(shared.expression); }
            for (std::size_t i = 0; i < recipe->pieces.size(); ++i) {
                const bool numerical = !recipe->arena ||
                    (i < recipe->selectors.size() && bool(recipe->selectors[i]));
                node.numerical.push_back(numerical);
                ContextualEndpointFamily family;
                family.piece = recipe->pieces[i];
                if (!numerical) {
                    if (i >= recipe->bindings.size()) { return failure(); }
                    const auto& binding = recipe->bindings[i];
                    if (!binding.activity) { return failure(); }
                    family.activity = cutExpressionRoot(*binding.activity);
                    if (binding.ordinal) { family.ordinal = cutExpressionRoot(*binding.ordinal); }
                    if (binding.residueCoordinate) {
                        family.residueCoordinate = cutExpressionRoot(*binding.residueCoordinate);
                        family.modulus = binding.modulus; family.residue = binding.residue;
                    }
                    for (const auto& coordinate : binding.matchingCoordinates) {
                        family.matchingCoordinates.push_back(cutExpressionRoot(*coordinate));
                    }
                }
                node.bindings.push_back(std::move(family));
            }
        }
        positions.emplace(recipe, result.size());
        result.push_back(std::move(node));
    }
    return result;
}
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> restoreDeferredFamilyRecipe(
    ArrayRef<DeferredFamilyNode> nodes)
{
    std::vector<std::shared_ptr<const FamilyEndpointRecipe>> restored;
    for (const auto& node : nodes) {
        auto reference = [&](int64_t id) -> std::shared_ptr<const FamilyEndpointRecipe> {
            return id >= 0 && static_cast<uint64_t>(id) < restored.size() ? restored[id] : nullptr;
        };
        if (node.kind != DeferredFamilyNode::Kind::Leaf) {
            auto child = reference(node.child);
            if (!child || node.outer || !node.pieces.empty() || !node.bindings.empty() ||
                !node.numerical.empty() || !node.shared.empty()) { return failure(); }
            if (node.kind == DeferredFamilyNode::Kind::Append) {
                auto previous = reference(node.previous);
                if (node.previous != -1 && !previous) { return failure(); }
                if (node.arena || !node.contextCuts.empty()) { return failure(); }
                auto joined = appendFamilyEndpointRecipe(previous, child, node.recordBase, 0);
                if (failed(joined)) { return failure(); }
                restored.push_back(*joined);
            } else {
                if (!node.arena || node.previous != -1 || node.recordBase) { return failure(); }
                auto context = std::make_shared<FamilyContextRecipe>();
                context->kind = node.kind == DeferredFamilyNode::Kind::Invocation ?
                    FamilyContextKind::Invocation : FamilyContextKind::Activity;
                context->arena = node.arena;
                auto cuts = originalFamilyCuts(*child, *context);
                std::size_t count = 0;
                for (const auto& [cut, nonlocal] : *cuts) {
                    if (context->kind == FamilyContextKind::Invocation && !nonlocal) { continue; }
                    auto found = node.contextCuts.find(cut);
                    if (found == node.contextCuts.end() || found->second >= node.arena->size() ||
                        node.arena->isBoolean(found->second) != (context->kind == FamilyContextKind::Activity)) {
                        return failure();
                    }
                    RegionExpressions::CutEmission emission;
                    auto binding = node.arena->prepareContextual(found->second, cut, emission);
                    if (failed(binding)) { return failure(); }
                    context->cuts.emplace(cut, std::move(*binding));
                    ++count;
                }
                if (count != node.contextCuts.size()) { return failure(); }
                auto recipe = std::make_shared<FamilyEndpointRecipe>();
                recipe->maximumRecord = child->maximumRecord;
                recipe->maximumEndpointIndex = child->maximumEndpointIndex;
                recipe->pieceCount = child->pieceCount;
                recipe->originalCuts = std::move(cuts);
                recipe->child = std::move(child);
                recipe->context = std::move(context);
                restored.push_back(std::move(recipe));
            }
            continue;
        }
        if (node.child != -1 || node.previous != -1 || node.recordBase || !node.contextCuts.empty() ||
            node.pieces.empty() || node.bindings.size() != node.pieces.size() ||
            node.numerical.size() != node.pieces.size()) { return failure(); }
        if (!node.arena) {
            if (!node.shared.empty() || llvm::any_of(node.numerical, [](uint8_t value) { return value != 1; })) {
                return failure();
            }
            auto recipe = restoreNumericalFamilyRecipe({node.outer, node.pieces});
            if (failed(recipe)) { return failure(); }
            restored.push_back(*recipe);
            continue;
        }
        std::vector<ContextualEndpointFamily> residual;
        for (std::size_t i = 0; i < node.pieces.size(); ++i) {
            if (node.numerical[i] > 1) { return failure(); }
            if (!node.numerical[i]) {
                auto binding = node.bindings[i]; binding.piece = node.pieces[i];
                residual.push_back(std::move(binding));
            }
        }
        std::shared_ptr<const FamilyEndpointRecipe> contextual;
        if (!residual.empty()) {
            auto prepared = buildContextualFamilyEndpointRecipe(node.arena, residual, node.shared);
            if (failed(prepared)) { return failure(); }
            contextual = *prepared;
        } else if (!node.shared.empty()) { return failure(); }
        auto recipe = std::make_shared<FamilyEndpointRecipe>();
        recipe->arena = node.arena; recipe->outer = node.outer; recipe->pieces = node.pieces;
        recipe->pieceCount = node.pieces.size();
        if (contextual) { recipe->shared = contextual->shared; }
        std::size_t residualIndex = 0;
        for (std::size_t i = 0; i < node.pieces.size(); ++i) {
            const auto& piece = node.pieces[i];
            if (node.numerical[i]) {
                if (!node.outer && piece.displacement) { return failure(); }
                auto selection = buildPieceSelector(piece);
                if (!selection.error.empty()) { return failure(); }
                recipe->selectors.push_back(std::move(selection.recipe)); recipe->bindings.emplace_back();
            } else {
                recipe->selectors.push_back(nullptr);
                recipe->bindings.push_back(contextual->bindings[residualIndex++]);
            }
            recipe->cuts.push_back({piece.cut, {}});
            recipe->maximumRecord = std::max(recipe->maximumRecord, piece.namespaceId);
            for (const auto& member : piece.members) {
                recipe->maximumRecord = std::max(recipe->maximumRecord, member.record);
            }
        }
        if (llvm::none_of(node.numerical, [](uint8_t value) { return value != 0; })) {
            recipe->selectors.clear();
        }
        restored.push_back(std::move(recipe));
    }
    if (restored.empty()) { return failure(); }
    return restored.back();
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
namespace {
bool emptyPhysicalPlan(const PreparedLogicalPlan& plan)
{
    return !plan.completeInvocation && !plan.familyRecipe && !plan.familyProtocol &&
        !plan.privateState && !plan.completionOwner && !plan.executedOutput && !plan.allocationPreparation &&
        !plan.allocationProofs && !plan.regionalAllocation && !plan.compactBoundingOwner &&
        plan.preparation.empty() && plan.endpoints.empty() && plan.families.empty();
}
} // namespace
bool isCertifiedEmptyFamilyPlan(func::FuncOp function, const PreparedLogicalPlan& plan)
{
    return function && !plan.physicalOutput && emptyPhysicalPlan(plan) &&
        plan.allocationCertificate == explicitAllocationCertificate(ExplicitAnalysis{}, plan.planId,
            function.getContext());
}
bool validFamilyEndpointProtocol(const PreparedLogicalPlan& plan)
{
    if (!plan.familyProtocol) {
        return !plan.executedOutput && (!plan.physicalOutput ||
            (plan.physicalOutput->emptyInvocation && plan.physicalOutput->emptyPlanId == plan.planId &&
             !plan.deferredOutput && emptyPhysicalPlan(plan)));
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
    if (plan.completionOwner) {
        if (plan.deferredOutput || plan.physicalOutput || plan.executedOutput || plan.completionPhysicalIds) {
            return function.emitError("completion physical output has incompatible request state");
        }
        std::set<int64_t> pool;
        for (auto id : eligibleIds) {
            if (id < 0 || id > 5 || !pool.insert(id).second) {
                return function.emitError("invalid or duplicate completion allocation ID");
            }
        }
        plan.completionPhysicalIds.emplace(eligibleIds.begin(), eligibleIds.end());
        return success();
    }
    if (plan.privateState) {
        return plan.privateState->requestPhysical(function, plan, eligibleIds);
    }
    // The producer may certify a whole invocation that executes no events.
    // Accept only the existing exact empty finite encoding and an empty plan;
    // a missing family recipe alone is never evidence of zero resource demand.
    if (!plan.deferredOutput && isCertifiedEmptyFamilyPlan(function, plan)) {
        if (failed(preflightPhysicalOutput(function, plan.allocationCertificate))) { return failure(); }
        auto decoded = decodePhysicalAllocationCertificate(function, plan.allocationCertificate, eligibleIds);
        if (failed(decoded) || decoded->planId != plan.planId || !decoded->records.empty()) { return failure(); }
        auto output = std::make_shared<PhysicalFamilyOutput>();
        output->emptyInvocation = true;
        output->emptyPlanId = plan.planId;
        output->certificate = plan.allocationCertificate;
        output->eligibleIds.assign(eligibleIds.begin(), eligibleIds.end());
        plan.physicalOutput = std::move(output);
        return success();
    }
    auto strategy = plan.allocationCertificate ?
        plan.allocationCertificate.getAs<StringAttr>("strategy") : StringAttr{};
    if (strategy && strategy.getValue() == "executed-family-counters") {
        if (!function || !plan.familyRecipe || failed(normalizeFamilyEndpointPlan(plan))) { return failure(); }
        // This route currently admits the arithmetic producer's contextual
        // leaves. No numerical selector or invocation wrapper is approximated.
        for (const auto& contribution : plan.familyRecipe->normalized->stream) {
            const auto* leaf = contribution.leaf;
            if (!leaf || contribution.contextPath != SIZE_MAX || !leaf->arena || !leaf->shared.empty() ||
                !leaf->selectors.empty() || llvm::any_of(leaf->pieces, [](const auto& piece) {
                    return piece.members.size() != 1;
                })) {
                return function.emitError("executed output requires contextual singleton arithmetic recipes");
            }
        }
        return planExecutedFamilyOutput(function, plan, eligibleIds);
    }
    auto refuse = [&](StringRef stage, Operation* cut = nullptr) -> InFlightDiagnostic {
        auto diagnostic = emitError(cut ? cut->getLoc() : function.getLoc());
        diagnostic << "combined physical mapping: " << stage;
        return diagnostic;
    };
    if (plan.physicalOutput || plan.completionOwner || !plan.familyRecipe || !plan.allocationCertificate) {
        return refuse("unsupported plan state") << " (physical=" << bool(plan.physicalOutput)
            << ", completion=" << bool(plan.completionOwner) << ", recipe=" << bool(plan.familyRecipe)
            << ", certificate=" << bool(plan.allocationCertificate) << ")";
    }
    if (failed(preflightPhysicalOutput(function, plan.allocationCertificate))) { return failure(); }
    if (failed(normalizeFamilyEndpointPlan(plan))) { return refuse("protocol normalization failed"); }
    std::set<int64_t> pool;
    for (auto id : eligibleIds) {
        if (id < 0 || id > 5 || !pool.insert(id).second) {
            return refuse("invalid or duplicate eligible ID") << ": " << id;
        }
    }
    if (pool.empty()) { return refuse("empty eligible ID pool"); }
    auto decoded = decodePhysicalAllocationCertificate(function, plan.allocationCertificate, eligibleIds);
    if (failed(decoded)) { return refuse("allocation certificate decoding failed"); }
    if (decoded->planId != plan.planId) {
        return refuse("allocation plan identity mismatch") << ": certificate=" << decoded->planId
            << ", prepared=" << plan.planId;
    }
    auto arities = familyMatchingArities(*plan.familyRecipe);
    if (failed(arities)) { return refuse("inconsistent deferred matching arities"); }
    for (const auto& endpoint : plan.endpoints) {
        if (endpoint.kind == LogicalCommandKind::Barrier) { continue; }
        if (endpoint.memberCoordinates.empty()) {
            return refuse("concrete endpoint has no matching tuple", endpoint.before)
                << " (record=" << endpoint.record << ")";
        }
        for (auto record : endpoint.records) {
            auto [found, added] = arities->emplace(record, endpoint.memberCoordinates.size());
            if (!added && found->second != endpoint.memberCoordinates.size()) {
                return refuse("concrete/deferred matching arity mismatch", endpoint.before)
                    << " (record=" << record << ", retained=" << found->second
                    << ", concrete=" << endpoint.memberCoordinates.size() << ")";
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
                refuse("invalid or duplicate allocation record") << ": " << record.record;
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
                if (found == records.end() || shape == arities->end()) {
                    refuse("missing allocation record or matching arity", cut) << " (record=" << label
                        << ", allocation=" << (found != records.end()) << ", arity=" << (shape != arities->end())
                        << ")";
                    return failure();
                }
                if (found->second->sourcePipe != source || found->second->targetPipe != target) {
                    refuse("allocation direction mismatch", cut) << " (record=" << label
                        << ", endpoint=" << source << "->" << target << ", allocation="
                        << found->second->sourcePipe << "->" << found->second->targetPipe << ")";
                    return failure();
                }
                if (arity && *arity != shape->second) {
                    refuse("endpoint members have different matching arities", cut)
                        << " (record=" << label << ", first=" << *arity << ", member=" << shape->second << ")";
                    return failure();
                }
                arity = shape->second;
                members.push_back(*found->second);
            }
            if (!arity) {
                refuse("endpoint has no original records", cut);
                return failure();
            }
            auto planned = planPhysicalEndpointMap(cut, members, points, *arity, plan.nestedIdentities, dominance);
            if (failed(planned)) {
                refuse("endpoint map construction failed", cut) << " (first record=" << labels.front()
                    << ", members=" << labels.size() << ", arity=" << *arity
                    << ", nested=" << plan.nestedIdentities << ")";
                return failure();
            }
            for (auto record : labels) {
                if (!result->bindings
                         .emplace(
                             std::make_tuple(cut, kind, record),
                             PhysicalFamilyOutput::Binding{source, target, *planned})
                         .second) {
                    refuse("duplicate original-record binding at command cut", cut)
                        << " (record=" << record << ", kind=" << static_cast<unsigned>(kind) << ")";
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
                    refuse("concrete endpoint lacks family provenance", endpoint.before)
                        << " (record=" << record << ")";
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
    auto compare = [&](const PhysicalFamilyOutput& before, const PhysicalFamilyOutput& after)
        -> std::optional<std::pair<uint64_t, uint64_t>> {
        // The fallback suffix comparison requires unchanged command sites.
        // Owned complete footprints below share the actual normalization and
        // account for allocation-dependent compaction before counting maps.
        auto uniqueSites = [](const PhysicalFamilyOutput& output) {
            std::map<std::pair<Operation*, LogicalCommandKind>, const PhysicalEndpointMap*> sites;
            for (const auto& [key, binding] : output.bindings) {
                auto [entry, added] = sites.emplace(
                    std::make_pair(std::get<0>(key), std::get<1>(key)), binding.mapping.get());
                if (!added && entry->second != binding.mapping.get()) { return false; }
            }
            return true;
        };
        if (before.protocol != after.protocol || before.concrete.size() != after.concrete.size() ||
            before.deferred.size() != after.deferred.size()) { return std::nullopt; }
        const bool ownedPreparation = !plan.privateState && !plan.executedOutput &&
            plan.preparation.empty() && plan.endpoints.empty();
        const bool sharedSelectors = ownedPreparation && plan.familyRecipe && !plan.allocationPreparation;
        const auto oldComplete = ownedPreparation ?
            ownedFamilyConstruction(before, sharedSelectors) : std::nullopt;
        const auto newComplete = ownedPreparation ?
            ownedFamilyConstruction(after, sharedSelectors) : std::nullopt;
        if (oldComplete && newComplete) {
            accumulateCost(plan.mappingWork.completeFootprints, 1);
            return std::make_pair(*oldComplete, *newComplete);
        }
        accumulateCost(plan.mappingWork.unavailablePreparation, 1);
        if (!uniqueSites(before) || !uniqueSites(after)) {
            accumulateCost(plan.mappingWork.unknownCoalescing, 1);
            return std::nullopt;
        }
        uint64_t oldCost = 0, newCost = 0;
        auto add = [&](const auto& a, const auto& b) {
            if (!a || !b) { return !a && !b; }
            PhysicalMapComparisonKind kind;
            auto costs = comparePhysicalMapConstruction(*a, *b, &kind);
            if (kind == PhysicalMapComparisonKind::ExactFootprint) {
                accumulateCost(plan.mappingWork.exactFootprints, 1);
            } else if (kind == PhysicalMapComparisonKind::InputDominance) {
                // Fewer consumers can split a shared guard union into several
                // groups. Only the complete schedule accounts for the new OR
                // and fallback work; a map suffix alone cannot cancel it.
                if (sharedSelectors) {
                    accumulateCost(plan.mappingWork.unknownInputs, 1);
                    return false;
                }
                accumulateCost(plan.mappingWork.inputDominance, 1);
            } else { accumulateCost(plan.mappingWork.unknownInputs, 1); }
            if (!costs || costs->first > uint64_t(INT64_MAX) - oldCost ||
                costs->second > uint64_t(INT64_MAX) - newCost) { return false; }
            oldCost += costs->first;
            newCost += costs->second;
            return true;
        };
        if (before.protocol != after.protocol || before.concrete.size() != after.concrete.size() ||
            before.deferred.size() != after.deferred.size()) { return std::nullopt; }
        for (auto [a, b] : llvm::zip(before.concrete, after.concrete)) {
            if (!add(a, b)) { return std::nullopt; }
        }
        for (const auto& [contribution, maps] : before.deferred) {
            auto found = after.deferred.find(contribution);
            if (found == after.deferred.end() || maps.size() != found->second.size()) { return std::nullopt; }
            for (auto [a, b] : llvm::zip(maps, found->second)) {
                if (!add(a, b)) { return std::nullopt; }
            }
        }
        // Identical live input masks cancel the exact frozen guard, wrapper,
        // shared-root, cofactor, replay and integer-expression preparation. Map
        // selectors/dispatch use their isolated common construction schedule.
        // The residual dominance subclass may only remove upstream inputs and
        // does not count assumed exclusive savings. Multiple contributions at
        // one physical site remain unknown because they may coalesce differently.
        return std::make_pair(oldCost, newCost);
    };
    // Every alternative already carries a completed parent certificate. This
    // comparison requests neither summary construction nor reuse proofs.
    auto alternatives = certifiedAllocationCandidates(plan.allocationCertificate);
    if (failed(alternatives)) { return refuse("malformed completed allocation candidate inventory"); }
    for (auto alternative : *alternatives) {
        FailureOr<std::shared_ptr<PhysicalFamilyOutput>> other = failure();
        {
            ScopedDiagnosticHandler optionalDiagnostics(function.getContext(), [](Diagnostic&) { return success(); });
            auto candidate = decodePhysicalAllocationCertificate(function, alternative, eligibleIds);
            if (succeeded(candidate) && candidate->planId == plan.planId) { other = build(*candidate); }
        }
        if (succeeded(other)) {
            accumulateCost(plan.mappingWork.candidates, 1);
            const auto costs = compare(**incumbent, **other);
            if (costs && costs->second < costs->first) {
                accumulateCost(plan.mappingWork.selected, 1);
                incumbent = std::move(other);
            }
        }
    }
    plan.physicalOutput = std::move(*incumbent);
    return success();
}
std::optional<uint64_t> physicalFamilyConstructionOperations(const PreparedLogicalPlan& plan)
{
    if (!validFamilyEndpointProtocol(plan) || !validPhysicalFamilyOutput(plan) || !plan.physicalOutput ||
        plan.executedOutput || plan.privateState || plan.completionOwner || !plan.preparation.empty() ||
        !plan.endpoints.empty()) { return std::nullopt; }
    return ownedFamilyConstruction(*plan.physicalOutput, plan.familyRecipe && !plan.allocationPreparation);
}
std::optional<uint64_t> logicalFamilyConstructionOperations(PreparedLogicalPlan& plan)
{
    if (!plan.familyRecipe || plan.privateState || plan.completionOwner || plan.physicalOutput ||
        plan.executedOutput || !plan.preparation.empty() || !plan.endpoints.empty()) {
        return std::nullopt;
    }
    if (failed(normalizeLogicalFamilies(plan))) { return std::nullopt; }
    PhysicalFamilyOutput logical;
    logical.protocol = plan.familyRecipe;
    return ownedFamilyConstruction(logical, !plan.allocationPreparation, true);
}
LogicalResult materializeFamilyEndpointCode(const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared)
{
    if (!validFamilyEndpointProtocol(prepared) || !validPhysicalFamilyOutput(prepared) ||
        !validExecutedFamilyOutput(prepared) || !recipe.normalized ||
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
    // Original-record namespace normalization applies to direct and composed
    // recipes alike. Validate provenance before allocating any operations.
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
                                                 ArrayRef<std::shared_ptr<const PhysicalEndpointMap>>{},
                       contribution.contextPath == SIZE_MAX ? 0 : contexts[contribution.contextPath].invocationDepth,
                       bool(prepared.executedOutput))
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
    auto pendingGuards = pendingCleanupGuards<PreparedLogicalEndpoint>(prepared.families, lowered.endpoints);
    if (failed(pendingGuards)) { return failure(); }
    const auto& primaryGuards = *pendingGuards;
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
