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
#include "PTO/Transforms/FrontierSynch/EndpointPieces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
bool sameCut(TemplateEndpointCut a, TemplateEndpointCut b)
{
    return a.block == b.block && a.before == b.before;
}
SmallVector<const EndpointPiece*> collectPieces(TemplateEndpointCut cut, ArrayRef<EndpointPiece> pieces)
{
    SmallVector<const EndpointPiece*> result;
    for (const auto& piece : pieces) {
        if (sameCut(cut, piece.cut)) {
            result.push_back(&piece);
        }
    }
    std::sort(result.begin(), result.end(), [](const EndpointPiece* a, const EndpointPiece* b) {
        return a->order < b->order;
    });
    return result;
}
class FamilyPreparer {
public:
    FamilyPreparer(const NumericTemplateEndpoints& plan, PreparedLogicalPlan& prepared,
                   ArrayRef<EndpointPiece> pieces)
        : plan(plan), prepared(prepared), pieces(pieces), builder(plan.outer->getContext()) {}
    LogicalResult run()
    {
        initializeOrdinals();
        for (const auto& group : plan.groups) {
            auto selected = collectPieces(group.cut, pieces);
            if (!selected.empty() && failed(prepareCut(group.cut, selected))) {
                return failure();
            }
        }
        return success();
    }
private:
    const NumericTemplateEndpoints& plan;
    PreparedLogicalPlan& prepared;
    ArrayRef<EndpointPiece> pieces;
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
    LogicalResult preparePiece(Operation* before, const EndpointPiece& piece, Value partnerPresent)
    {
        auto location = before->getLoc();
        SmallVector<SmallVector<TemplateCoordinate>> coordinates;
        SmallVector<int64_t> labels;
        for (const auto& member : piece.members) {
            coordinates.push_back(member.coordinates);
            labels.push_back(member.record);
        }
        auto selector = emitFamilyExpressions(builder, location, coordinates, labels);
        if (!selector.error.empty()) {
            return before->emitError(selector.error);
        }
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
            endpoint.records.push_back(member.record);
        }
        endpoint.piece = static_cast<int64_t>(prepared.endpoints.size());
        prepared.endpoints.push_back(std::move(endpoint));
        return success();
    }
    LogicalResult prepareCut(TemplateEndpointCut cut, ArrayRef<const EndpointPiece*> selected)
    {
        builder.setInsertionPointToEnd(&prepared.addPreparation(cut.before));
        std::map<std::pair<bool, uint64_t>, Value> predicates;
        for (const auto* piece : selected) {
            auto [entry, inserted] = predicates.try_emplace(
                std::make_pair(piece->kind == EndpointKind::Set, piece->displacement));
            if (inserted) {
                entry->second = boundary(*piece, cut.before->getLoc());
            }
            if (failed(preparePiece(cut.before, *piece, entry->second))) {
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
    auto pieces = buildEndpointPieces(plan);
    if (!pieces.error.empty()) {
        return plan.outer->emitError(pieces.error);
    }
    if (!pieces.pieces.empty() && failed(FamilyPreparer(plan, prepared, pieces.pieces).run())) {
        return failure();
    }
    prepared.groupedFamilies = true;
    prepared.independentPieces = true;
    return success();
}
} // namespace mlir::pto::frontiersynch
