// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Materialize family domains and matching identities before logical insertion.
#include "PTO/Transforms/FrontierSynch/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/FamilyExpressions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
struct Piece {
    const EndpointFamily* family = nullptr;
    bool publish = false;
    std::size_t order = 0;
};
bool sameCut(TemplateEndpointCut a, TemplateEndpointCut b)
{
    return a.block == b.block && a.before == b.before;
}
SmallVector<Piece> collectPieces(TemplateEndpointCut cut, ArrayRef<EndpointFamily> families)
{
    SmallVector<Piece> result;
    for (const auto& family : families) {
        if (!family.local && sameCut(cut, family.sourceCut)) {
            result.push_back({&family, true, family.sourceOrder});
        }
        if (sameCut(cut, family.targetCut)) {
            result.push_back({&family, false, family.targetOrder});
        }
    }
    std::sort(result.begin(), result.end(), [](const Piece& a, const Piece& b) { return a.order < b.order; });
    return result;
}
class FamilyPreparer {
public:
    FamilyPreparer(const NumericTemplateEndpoints& plan, PreparedLogicalPlan& prepared)
        : plan(plan), prepared(prepared), builder(plan.outer->getContext()) {}
    LogicalResult run()
    {
        initializeOrdinals();
        for (const auto& group : plan.groups) {
            auto pieces = collectPieces(group.cut, prepared.families);
            if (!pieces.empty() && failed(prepareCut(group.cut, pieces))) {
                return failure();
            }
        }
        return success();
    }
private:
    const NumericTemplateEndpoints& plan;
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
        auto loop = plan.outer;
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
    Value boundary(const Piece& piece, Location location)
    {
        const auto displacement = piece.family->displacement;
        if (!displacement) {
            return builder.create<arith::ConstantIntOp>(location, 1, 1);
        }
        auto delay = number(displacement, location);
        if (!piece.publish) {
            return compare(arith::CmpIPredicate::uge, ordinal, delay, location);
        }
        auto remaining = builder.create<arith::SubIOp>(location, trips, delay);
        auto hasTarget = compare(arith::CmpIPredicate::ult, delay, trips, location);
        auto beforeEnd = compare(arith::CmpIPredicate::ult, ordinal, remaining, location);
        return builder.create<arith::AndIOp>(location, hasTarget, beforeEnd);
    }
    LogicalResult preparePiece(Operation* before, const Piece& piece, Value partnerPresent)
    {
        auto location = before->getLoc();
        const auto& family = *piece.family;
        SmallVector<SmallVector<TemplateCoordinate>> coordinates;
        for (const auto& member : family.members) {
            coordinates.push_back(piece.publish ? member.sourceCoordinates : member.targetCoordinates);
        }
        auto selector = emitFamilyExpressions(builder, location, coordinates);
        if (!selector.error.empty()) {
            return before->emitError(selector.error);
        }
        Value guard = builder.create<arith::AndIOp>(location, selector.present, partnerPresent);
        Value identity;
        auto kind = LogicalCommandKind::Barrier;
        if (!family.local) {
            kind = piece.publish ? LogicalCommandKind::Set : LogicalCommandKind::Wait;
            identity = ordinal;
            if (!piece.publish && family.displacement) {
                identity = builder.create<arith::SubIOp>(location, ordinal, number(family.displacement, location));
            }
        }
        PreparedLogicalEndpoint endpoint{before, kind, family.sourcePipe, family.targetPipe,
                                         family.id, guard, identity, {}};
        if (!family.local && family.members.size() > 1) {
            endpoint.memberCoordinates.push_back(selector.member);
        }
        prepared.endpoints.push_back(std::move(endpoint));
        return success();
    }
    LogicalResult prepareCut(TemplateEndpointCut cut, ArrayRef<Piece> pieces)
    {
        builder.setInsertionPointToEnd(&prepared.addPreparation(cut.before));
        std::map<std::pair<bool, uint64_t>, Value> predicates;
        for (const auto& piece : pieces) {
            auto [entry, inserted] = predicates.try_emplace({piece.publish, piece.family->displacement});
            if (inserted) {
                entry->second = boundary(piece, cut.before->getLoc());
            }
            if (failed(preparePiece(cut.before, piece, entry->second))) {
                return failure();
            }
        }
        return success();
    }
};
} // namespace
LogicalResult prepareFamilyEndpointCode(const NumericTemplateEndpoints& plan, PreparedLogicalPlan& prepared)
{
    if (!plan.outer || !plan.logical.error.empty() ||
        (plan.logical.recipes.empty() != prepared.families.empty())) {
        return failure();
    }
    if (prepared.families.empty()) {
        prepared.groupedFamilies = true;
        return success();
    }
    if (failed(FamilyPreparer(plan, prepared).run())) {
        return failure();
    }
    prepared.groupedFamilies = true;
    return success();
}
} // namespace mlir::pto::frontiersynch
