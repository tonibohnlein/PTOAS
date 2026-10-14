// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Retain numerical and contextual family recipes through composition; lower at insertion.
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
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
};
class FamilyEndpointRecipe {
public:
    // A leaf owns numerical selectors or contextual bindings. A composition references earlier
    // recipes and records namespace/placement transforms; append is constant work.
    std::shared_ptr<const FamilyEndpointRecipe> previous, child;
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
    scf::ForOp outer;
    std::vector<EndpointPiece> pieces;
    std::vector<std::shared_ptr<const FamilyExpressionRecipe>> selectors;
    struct Cut { TemplateEndpointCut cut; SmallVector<std::size_t> pieces; };
    std::vector<Cut> cuts;
};
namespace {
class FamilyPreparer {
public:
    FamilyPreparer(const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared, uint32_t recordBase,
                   const std::map<uint32_t, uint32_t>& namespaces)
        : recipe(recipe), prepared(prepared), recordBase(recordBase), namespaces(namespaces),
          builder(recipe.cuts.front().cut.before->getContext()) {}
    LogicalResult run()
    {
        if (recipe.arena) { return prepareContextual(); }
        if (recipe.outer) { initializeOrdinals(); }
        for (const auto& cut : recipe.cuts) {
            if (!cut.pieces.empty()) { prepareCut(cut.cut, cut.pieces); }
        }
        return success();
    }
private:
    const FamilyEndpointRecipe& recipe;
    PreparedLogicalPlan& prepared;
    uint32_t recordBase;
    const std::map<uint32_t, uint32_t>& namespaces;
    OpBuilder builder;
    Value trips;
    Value ordinal;
    Value zero;
    Value one;
    LogicalResult prepareContextual()
    {
        std::map<Operation*, RegionExpressions::CutEmission> memos;
        std::map<Operation*, RegionExpressions::CutEmission> sharedMemos;
        for (const auto& shared : recipe.shared) {
            auto* cut = shared.expression.cut;
            builder.setInsertionPointToEnd(&prepared.addPreparation(cut));
            auto value = recipe.arena->emitPreparedContextual(*shared.binding, builder, cut, sharedMemos[cut]);
            if (failed(value)) { return failure(); }
            sharedMemos[cut].values[shared.expression.expression] = *value;
        }
        DominanceInfo dominance(recipe.cuts.front().cut.before->getParentOfType<func::FuncOp>());
        std::set<Operation*> seeded;
        for (std::size_t i = 0; i < recipe.pieces.size(); ++i) {
            const auto& piece = recipe.pieces[i];
            const auto& binding = recipe.bindings[i];
            auto* cut = piece.cut.before;
            if (seeded.insert(cut).second) {
                for (const auto& shared : sharedMemos) {
                    if (shared.first != cut && !dominance.properlyDominates(shared.first, cut)) { continue; }
                    for (const auto& value : shared.second.values) {
                        memos[cut].values.try_emplace(value.first, value.second);
                    }
                    for (const auto& input : shared.second.inputs) {
                        memos[cut].inputs.try_emplace(input.first, input.second);
                    }
                }
            }
            builder.setInsertionPointToEnd(&prepared.addPreparation(cut));
            auto emit = [&](const std::shared_ptr<const CutExpressionRecipe>& value) {
                return recipe.arena->emitPreparedContextual(*value, builder, cut, memos[cut]);
            };
            auto guard = emit(binding.activity);
            if (failed(guard)) { return failure(); }
            const bool local = piece.kind == EndpointKind::Barrier;
            auto kind = local ? LogicalCommandKind::Barrier :
                (piece.kind == EndpointKind::Set ? LogicalCommandKind::Set : LogicalCommandKind::Wait);
            const auto record = piece.members.front().record + recordBase;
            PreparedLogicalEndpoint endpoint{cut, kind, piece.sourcePipe, piece.targetPipe,
                namespaces.empty() ? piece.namespaceId + recordBase : namespaces.at(record), *guard, {}};
            endpoint.records = {record};
            retainEndpointCoordinates(endpoint, piece.members.front().coordinates);
            if (binding.residueCoordinate) {
                auto coordinate = emit(binding.residueCoordinate);
                if (failed(coordinate)) { return failure(); }
                if (endpoint.domain.empty()) { endpoint.domain.emplace_back(); }
                for (auto& box : endpoint.domain) {
                    box.push_back({*coordinate, binding.modulus, static_cast<int64_t>(binding.residue)});
                }
            }
            if (!local) {
                auto ordinal = emit(binding.ordinal);
                if (failed(ordinal)) { return failure(); }
                endpoint.identity = *ordinal;
                endpoint.memberCoordinates.push_back(number(record, cut->getLoc()));
            }
            endpoint.piece = prepared.endpoints.size();
            prepared.endpoints.push_back(std::move(endpoint));
        }
        return success();
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
        auto selector = emitFamilyExpressions(builder, location, *recipe.selectors[index]);
        Value guard = builder.create<arith::AndIOp>(location, selector.present, partnerPresent);
        Value identity;
        auto kind = LogicalCommandKind::Barrier;
        const bool local = piece.kind == EndpointKind::Barrier;
        if (!local) {
            kind = piece.kind == EndpointKind::Set ? LogicalCommandKind::Set : LogicalCommandKind::Wait;
            identity = ordinal;
            if (piece.kind == EndpointKind::Wait && piece.displacement) {
                identity = builder.create<arith::SubIOp>(location, ordinal, number(piece.displacement, location));
            }
        }
        PreparedLogicalEndpoint endpoint{before, kind, piece.sourcePipe, piece.targetPipe,
                                         namespaces.empty() ? piece.namespaceId + recordBase :
                                             namespaces.at(piece.members.front().record + recordBase),
                                         guard, identity, {}};
        if (!local) {
            Value member = selector.member;
            if (recordBase) {
                member = builder.create<arith::AddIOp>(location, member, number(recordBase, location));
            }
            endpoint.memberCoordinates.push_back(member);
        }
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
        if (!recipe.outer) { ordinal = number(0, cut.before->getLoc()); }
        std::map<std::pair<bool, uint64_t>, Value> predicates;
        for (auto index : selected) {
            const auto& piece = recipe.pieces[index];
            auto [entry, inserted] = predicates.try_emplace(
                std::make_pair(piece.kind == EndpointKind::Set, piece.displacement));
            if (inserted) {
                entry->second = boundary(piece, cut.before->getLoc());
            }
            preparePiece(cut.before, index, entry->second);
        }
    }
};
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildRecipe(const NumericTemplateEndpoints& plan)
{
    if (!plan.logical.error.empty()) { return failure(); }
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
        SmallVector<SmallVector<TemplateCoordinate>> coordinates;
        SmallVector<int64_t> labels;
        for (const auto& member : piece.members) {
            coordinates.push_back(member.coordinates);
            labels.push_back(member.record);
        }
        auto selection = buildFamilyExpressions(coordinates, labels);
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
    RegionExpressions::Id expression, Operation* cut)
{
    if (!owner) { return failure(); }
    const auto key = std::make_pair(expression, cut);
    auto found = recipes.find(key);
    if (found != recipes.end()) { return found->second; }
    auto recipe = owner->prepareContextual(expression, cut, contexts[cut]);
    if (failed(recipe)) { return failure(); }
    recipes.emplace(key, *recipe);
    return *recipe;
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
    for (const auto& expression : shared) {
        if (!expression.cut || expression.cut->getParentOfType<func::FuncOp>() != function) { return failure(); }
        auto planned = expressions.prepare(expression.expression, expression.cut);
        if (failed(planned)) { return failure(); }
        recipe->shared.push_back({expression, std::move(*planned)});
    }
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
            return expressions.prepare(expression, cut);
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
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> bindFamilyContext(
    std::shared_ptr<const FamilyEndpointRecipe> child, std::shared_ptr<RegionExpressions> arena,
    RegionExpressions::Id expression, FamilyContextKind kind)
{
    if (!child || !arena || expression >= arena->size() ||
        arena->isBoolean(expression) != (kind == FamilyContextKind::Activity)) { return failure(); }
    auto context = std::make_shared<FamilyContextRecipe>();
    context->kind = kind;
    context->arena = std::move(arena);
    SmallVector<const FamilyEndpointRecipe*> pending{child.get()};
    while (!pending.empty()) {
        const auto* current = pending.pop_back_val();
        if (current->child) { pending.push_back(current->child.get()); }
        if (current->previous) { pending.push_back(current->previous.get()); }
        for (const auto& piece : current->pieces) {
            if ((kind == FamilyContextKind::Invocation && piece.sourcePipe == piece.targetPipe) ||
                context->cuts.count(piece.cut.before)) { continue; }
            RegionExpressions::CutEmission emission;
            auto prepared = context->arena->prepareContextual(expression, piece.cut.before, emission);
            if (failed(prepared)) { return failure(); }
            context->cuts.emplace(piece.cut.before, std::move(*prepared));
        }
    }
    auto recipe = std::make_shared<FamilyEndpointRecipe>();
    recipe->maximumRecord = child->maximumRecord;
    recipe->maximumEndpointIndex = child->maximumEndpointIndex;
    recipe->pieceCount = child->pieceCount;
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
LogicalResult collectFamilyParts(const FamilyEndpointRecipe& recipe, SmallVectorImpl<FamilyPart>& leaves,
                                 SmallVectorImpl<FamilyContextPath>& contexts)
{
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
    explicit FamilyContextLowering(PreparedLogicalPlan& plan) : plan(plan) {}
    LogicalResult apply(PreparedLogicalEndpoint& endpoint, std::size_t path, ArrayRef<FamilyContextPath> paths)
    {
        // The innermost binding is visited first. Inserting at member one then
        // leaves enclosing invocation coordinates before their descendants.
        while (path != SIZE_MAX) {
            const auto& item = paths[path];
            path = item.parent;
            if (item.recipe->kind == FamilyContextKind::Invocation &&
                endpoint.kind == LogicalCommandKind::Barrier) { continue; }
            auto value = lower(*item.recipe, endpoint.before);
            if (failed(value)) { return failure(); }
            if (item.recipe->kind == FamilyContextKind::Invocation) {
                if (endpoint.memberCoordinates.empty()) { return failure(); }
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
    std::map<Operation*, Block*> blocks;
    std::map<std::pair<RegionExpressions*, Operation*>, RegionExpressions::CutEmission> contexts;
    FailureOr<Value> lower(const FamilyContextRecipe& recipe, Operation* cut)
    {
        auto found = recipe.cuts.find(cut);
        if (found == recipe.cuts.end()) { return failure(); }
        auto& block = blocks[cut];
        if (!block) { block = &plan.addPreparation(cut); }
        OpBuilder builder(cut->getContext());
        builder.setInsertionPointToEnd(block);
        return recipe.arena->emitPreparedContextual(*found->second, builder, cut,
                                                   contexts[{recipe.arena.get(), cut}]);
    }
};
} // namespace
std::size_t familyEndpointPieceCount(const FamilyEndpointRecipe& recipe) { return recipe.pieceCount; }
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
LogicalResult materializeFamilyEndpointCode(const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared)
{
    if (!recipe.child && (!prepared.preparation.empty() || !prepared.endpoints.empty())) { return failure(); }
    if (prepared.endpoints.size() > static_cast<uint64_t>(INT64_MAX) ||
        recipe.maximumEndpointIndex > prepared.endpoints.size() ||
        recipe.pieceCount > static_cast<uint64_t>(INT64_MAX) - prepared.endpoints.size()) { return failure(); }
    SmallVector<FamilyPart> leaves;
    SmallVector<FamilyContextPath> contexts;
    if (failed(collectFamilyParts(recipe, leaves, contexts))) { return failure(); }
    // Sequence namespace normalization applies to delayed and concrete records
    // alike. Validate the translated provenance before allocating any operations.
    auto namespaces = recipe.child ? familyRecordNamespaces(prepared.families) : std::map<uint32_t, uint32_t>{};
    if (recipe.child) {
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
    FamilyContextLowering contextLowering(lowered);
    std::size_t next = 0;
    for (const auto& part : leaves) {
        while (next < part.endpointIndex) { lowered.endpoints.push_back(prepared.endpoints[next++]); }
        const auto begin = lowered.endpoints.size();
        if (!part.recipe->pieces.empty()) {
            if (failed(FamilyPreparer(*part.recipe, lowered, part.recordBase, namespaces).run())) {
                return failure();
            }
        }
        for (std::size_t i = begin; i < lowered.endpoints.size(); ++i) {
            if (failed(contextLowering.apply(lowered.endpoints[i], part.contextPath, contexts))) {
                return failure();
            }
        }
    }
    while (next < prepared.endpoints.size()) { lowered.endpoints.push_back(prepared.endpoints[next++]); }
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
    prepared.endpoints = std::move(lowered.endpoints);
    for (auto& stage : lowered.preparation) { prepared.preparation.push_back(std::move(stage)); }
    prepared.groupedFamilies = true;
    prepared.independentPieces = true;
    return success();
}
} // namespace mlir::pto::frontiersynch
