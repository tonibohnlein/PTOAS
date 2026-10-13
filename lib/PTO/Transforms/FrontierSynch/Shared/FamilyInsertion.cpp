// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Materialize family domains and matching identities before logical insertion.
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
#include "PTO/Transforms/FrontierSynch/Shared/EndpointPieces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>
#include <map>
namespace mlir::pto::frontiersynch {
class FamilyEndpointRecipe {
public:
    scf::ForOp outer;
    std::vector<EndpointPiece> pieces;
    std::vector<std::shared_ptr<const FamilyExpressionRecipe>> selectors;
    struct Cut { TemplateEndpointCut cut; SmallVector<std::size_t> pieces; };
    std::vector<Cut> cuts;
};
namespace {
class FamilyPreparer {
public:
    FamilyPreparer(const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared)
        : recipe(recipe), prepared(prepared), builder(recipe.outer->getContext()) {}
    void run()
    {
        initializeOrdinals();
        for (const auto& cut : recipe.cuts) {
            if (!cut.pieces.empty()) { prepareCut(cut.cut, cut.pieces); }
        }
    }
private:
    const FamilyEndpointRecipe& recipe;
    PreparedLogicalPlan& prepared;
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
                                         piece.namespaceId, guard, identity, {}};
        if (!local) {
            endpoint.memberCoordinates.push_back(selector.member);
        }
        for (const auto& member : piece.members) {
            retainEndpointCoordinates(endpoint, member.coordinates);
            endpoint.records.push_back(member.record);
        }
        endpoint.piece = static_cast<int64_t>(prepared.endpoints.size());
        prepared.endpoints.push_back(std::move(endpoint));
    }
    void prepareCut(TemplateEndpointCut cut, ArrayRef<std::size_t> selected)
    {
        builder.setInsertionPointToEnd(&prepared.addPreparation(cut.before));
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
} // namespace
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildFamilyEndpointRecipe(const NumericTemplateEndpoints& plan)
{
    if (!plan.outer || !plan.logical.error.empty()) { return failure(); }
    auto pieces = buildEndpointPieces(plan);
    if (!pieces.error.empty()) { return plan.outer->emitError(pieces.error), failure(); }
    auto recipe = std::make_shared<FamilyEndpointRecipe>();
    recipe->outer = plan.outer;
    recipe->pieces = std::move(pieces.pieces);
    std::map<Operation*, std::size_t> cuts;
    for (const auto& group : plan.groups) {
        if (!cuts.emplace(group.cut.before, recipe->cuts.size()).second) { return failure(); }
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
        if (!selection.error.empty()) { return plan.outer->emitError(selection.error), failure(); }
        recipe->selectors.push_back(std::move(selection.recipe));
    }
    for (auto& cut : recipe->cuts) {
        llvm::sort(cut.pieces, [&](std::size_t a, std::size_t b) {
            return recipe->pieces[a].order < recipe->pieces[b].order;
        });
    }
    return std::shared_ptr<const FamilyEndpointRecipe>(std::move(recipe));
}
std::size_t familyEndpointPieceCount(const FamilyEndpointRecipe& recipe) { return recipe.pieces.size(); }
void materializeFamilyEndpointCode(const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared)
{
    if (!recipe.pieces.empty()) { FamilyPreparer(recipe, prepared).run(); }
    prepared.groupedFamilies = true;
    prepared.independentPieces = true;
}
} // namespace mlir::pto::frontiersynch
