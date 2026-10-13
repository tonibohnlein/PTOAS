// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Retain numerical family recipes through sequence composition; lower at insertion.
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
#include "PTO/Transforms/FrontierSynch/Shared/EndpointPieces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>
#include <limits>
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
struct FamilyInvocationRecipe {
    std::shared_ptr<RegionExpressions> arena;
    std::map<Operation*, std::shared_ptr<const CutExpressionRecipe>> cuts;
};
class FamilyEndpointRecipe {
public:
    // A leaf owns a numerical endpoint plan. A composition references earlier
    // recipes and records namespace/placement transforms; append is constant work.
    std::shared_ptr<const FamilyEndpointRecipe> previous, child;
    std::shared_ptr<const FamilyInvocationRecipe> invocation;
    uint32_t recordBase = 0, maximumRecord = 0;
    std::size_t endpointIndex = 0, maximumEndpointIndex = 0, pieceCount = 0;
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
    void run()
    {
        if (recipe.outer) { initializeOrdinals(); }
        for (const auto& cut : recipe.cuts) {
            if (!cut.pieces.empty()) { prepareCut(cut.cut, cut.pieces); }
        }
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
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> prefixFamilyInvocation(
    std::shared_ptr<const FamilyEndpointRecipe> child, std::shared_ptr<RegionExpressions> arena,
    RegionExpressions::Id visit)
{
    if (!child || !arena || visit >= arena->size() || arena->isBoolean(visit)) { return failure(); }
    auto invocation = std::make_shared<FamilyInvocationRecipe>();
    invocation->arena = std::move(arena);
    SmallVector<const FamilyEndpointRecipe*> pending{child.get()};
    while (!pending.empty()) {
        const auto* current = pending.pop_back_val();
        if (current->child) { pending.push_back(current->child.get()); }
        if (current->previous) { pending.push_back(current->previous.get()); }
        for (const auto& piece : current->pieces) {
            if (piece.sourcePipe == piece.targetPipe || invocation->cuts.count(piece.cut.before)) { continue; }
            RegionExpressions::CutEmission context;
            auto prepared = invocation->arena->prepareContextual(visit, piece.cut.before, context);
            if (failed(prepared)) { return failure(); }
            invocation->cuts.emplace(piece.cut.before, std::move(*prepared));
        }
    }
    auto recipe = std::make_shared<FamilyEndpointRecipe>();
    recipe->maximumRecord = child->maximumRecord;
    recipe->maximumEndpointIndex = child->maximumEndpointIndex;
    recipe->pieceCount = child->pieceCount;
    recipe->child = std::move(child);
    recipe->invocation = std::move(invocation);
    return std::shared_ptr<const FamilyEndpointRecipe>(std::move(recipe));
}
namespace {
struct InvocationPath {
    const FamilyInvocationRecipe* recipe;
    std::size_t parent;
};
class InvocationLowering {
public:
    explicit InvocationLowering(PreparedLogicalPlan& plan) : plan(plan) {}
    LogicalResult prefix(PreparedLogicalEndpoint& endpoint, std::size_t path, ArrayRef<InvocationPath> paths)
    {
        if (endpoint.kind == LogicalCommandKind::Barrier) { return success(); }
        if (endpoint.memberCoordinates.empty()) { return failure(); }
        // The innermost binding is visited first. Inserting at member one then
        // leaves enclosing invocation coordinates before their descendants.
        while (path != SIZE_MAX) {
            const auto& item = paths[path];
            auto value = lower(*item.recipe, endpoint.before);
            if (failed(value)) { return failure(); }
            endpoint.memberCoordinates.insert(endpoint.memberCoordinates.begin() + 1, *value);
            path = item.parent;
        }
        return success();
    }
private:
    PreparedLogicalPlan& plan;
    std::map<Operation*, Block*> blocks;
    std::map<std::pair<RegionExpressions*, Operation*>, RegionExpressions::CutEmission> contexts;
    FailureOr<Value> lower(const FamilyInvocationRecipe& recipe, Operation* cut)
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
LogicalResult materializeFamilyEndpointCode(const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared)
{
    if (!recipe.child && (!prepared.preparation.empty() || !prepared.endpoints.empty())) { return failure(); }
    if (prepared.endpoints.size() > static_cast<uint64_t>(INT64_MAX) ||
        recipe.maximumEndpointIndex > prepared.endpoints.size() ||
        recipe.pieceCount > static_cast<uint64_t>(INT64_MAX) - prepared.endpoints.size()) { return failure(); }
    struct Part {
        const FamilyEndpointRecipe* recipe;
        uint32_t recordBase;
        std::size_t endpointIndex, invocation;
    };
    SmallVector<Part> pending{{&recipe, 0, 0, SIZE_MAX}}, leaves;
    SmallVector<InvocationPath> invocations;
    // Preserve the original endpoint inventory order, including concrete
    // crossings between recipe contributions. No recursive host call stack.
    while (!pending.empty()) {
        auto part = pending.pop_back_val();
        if (part.recipe->child) {
            auto path = part.invocation;
            if (part.recipe->invocation) {
                path = invocations.size();
                invocations.push_back({part.recipe->invocation.get(), part.invocation});
            }
            pending.push_back({part.recipe->child.get(), part.recordBase + part.recipe->recordBase,
                               part.endpointIndex + part.recipe->endpointIndex, path});
            if (part.recipe->previous) {
                pending.push_back({part.recipe->previous.get(), part.recordBase, part.endpointIndex, part.invocation});
            }
        } else {
            if (!leaves.empty() && leaves.back().endpointIndex > part.endpointIndex) { return failure(); }
            leaves.push_back(part);
        }
    }
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
    // Stage all lowering independently: unavailable invocation bindings must
    // not consume the incumbent concrete inventory or partially lower a recipe.
    PreparedLogicalPlan lowered(prepared.planId);
    InvocationLowering invocationLowering(lowered);
    std::size_t next = 0;
    for (const auto& part : leaves) {
        while (next < part.endpointIndex) { lowered.endpoints.push_back(prepared.endpoints[next++]); }
        const auto begin = lowered.endpoints.size();
        if (!part.recipe->pieces.empty()) {
            FamilyPreparer(*part.recipe, lowered, part.recordBase, namespaces).run();
        }
        for (std::size_t i = begin; i < lowered.endpoints.size(); ++i) {
            if (failed(invocationLowering.prefix(lowered.endpoints[i], part.invocation, invocations))) {
                return failure();
            }
        }
    }
    while (next < prepared.endpoints.size()) { lowered.endpoints.push_back(prepared.endpoints[next++]); }
    for (std::size_t i = 0; i < lowered.endpoints.size(); ++i) {
        lowered.endpoints[i].piece = static_cast<int64_t>(i);
    }
    prepared.endpoints = std::move(lowered.endpoints);
    for (auto& stage : lowered.preparation) { prepared.preparation.push_back(std::move(stage)); }
    prepared.groupedFamilies = true;
    prepared.independentPieces = true;
    return success();
}
} // namespace mlir::pto::frontiersynch
