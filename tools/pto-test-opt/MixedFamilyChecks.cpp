// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent unfolded-order checks for mixed numerical/guarded inventories.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/MixedEndpointPieces.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/MixedFamilyInsertion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include <set>
namespace {
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
constexpr uint32_t residualRecord = 100000;
bool active(const fs::EndpointPieceMember& member, int64_t visit, bool guard)
{
    if (member.record == residualRecord && !guard) { return false; }
    return llvm::all_of(member.coordinates, [&](const auto& point) { return point.induction == visit; });
}
bool checkOrder(ArrayRef<fs::EndpointPiece> input, fs::EndpointPieces grouped, unsigned count)
{
    if (!grouped.error.empty()) { return false; }
    llvm::sort(grouped.pieces, [](const auto& a, const auto& b) { return a.order < b.order; });
    for (unsigned visit = 0; visit < count; ++visit) {
        for (bool guard : {false, true}) {
            SmallVector<uint32_t> expected, actual;
            for (const auto& point : input) {
                if (active(point.members.front(), visit, guard)) { expected.push_back(point.members.front().record); }
            }
            for (const auto& piece : grouped.pieces) {
                unsigned executed = 0;
                for (const auto& member : piece.members) {
                    if (active(member, visit, guard)) { actual.push_back(member.record); ++executed; }
                }
                if (executed > 1) { return false; }
            }
            if (actual != expected) { return false; }
        }
    }
    return true;
}
bool checkLowering(ArrayRef<fs::EndpointPiece> input, ArrayRef<uint8_t> numerical,
                   scf::ForOp loop, Value guard)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    SmallVector<fs::ContextualEndpointFamily> families;
    for (const auto& point : input) {
        fs::ContextualEndpointFamily family;
        family.piece = point;
        family.activity = point.members.front().record == residualRecord ? arena->input(guard) :
            arena->eq(arena->input(loop.getInductionVar()),
                      arena->constant(point.members.front().coordinates.front().induction));
        family.ordinal = arena->constant(0);
        families.push_back(std::move(family));
    }
    fs::ContextualFamilyExpressions expressions(arena);
    auto recipe = fs::buildMixedFamilyEndpointRecipe(expressions, families, numerical);
    if (failed(recipe) || fs::familyEndpointPieceCount(**recipe) > 3) { return false; }
    fs::PreparedLogicalPlan plan(0);
    plan.familyRecipe = *recipe;
    auto shapes = fs::familyMatchingArities(**recipe);
    if (failed(shapes) || shapes->size() != input.size() || !plan.preparation.empty() ||
        failed(fs::materializeLogicalFamilies(plan)) || plan.endpoints.size() > 3) { return false; }
    std::set<uint32_t> records;
    for (const auto& endpoint : plan.endpoints) {
        records.insert(endpoint.records.begin(), endpoint.records.end());
        if (endpoint.records.front() == residualRecord && endpoint.guard != guard) { return false; }
    }
    return records.size() == input.size();
}
} // namespace
bool runMixedFamilyChecks(mlir::MLIRContext* context)
{
    OpBuilder builder(context);
    auto location = builder.getUnknownLoc();
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(location, "mixed_family_order",
        builder.getFunctionType({builder.getI1Type()}, {})));
    auto* body = function->addEntryBlock();
    builder.setInsertionPointToEnd(body);
    auto zero = builder.create<arith::ConstantIndexOp>(location, 0);
    auto one = builder.create<arith::ConstantIndexOp>(location, 1);
    auto end = builder.create<arith::ConstantIndexOp>(location, 128);
    auto loop = builder.create<scf::ForOp>(location, zero, end, one);
    builder.create<func::ReturnOp>(location);
    builder.setInsertionPointToStart(loop.getBody());
    auto* cut = builder.create<arith::ConstantIndexOp>(location, 17).getOperation();
    fs::EndpointPiece residual;
    residual.sourcePipe = 1; residual.targetPipe = 2;
    residual.cut = {cut->getBlock(), cut};
    residual.members.push_back({residualRecord, {}});
    for (unsigned count : {8U, 32U, 128U}) {
        for (unsigned position : {0U, count / 2, count}) {
            for (bool reverseLabels : {false, true}) {
                SmallVector<fs::EndpointPiece> points;
                SmallVector<uint8_t> numerical;
                for (unsigned i = 0; i <= count; ++i) {
                    if (i == position) { points.push_back(residual); numerical.push_back(0); }
                    if (i == count) { continue; }
                    auto point = residual;
                    point.members = {{reverseLabels ? count - i : i, {{loop, int64_t(i)}}}};
                    points.push_back(point); numerical.push_back(1);
                }
                auto grouped = fs::buildMixedEndpointPieces(points, numerical);
                const unsigned expected = position == 0 || position == count ? 2 : 3;
                if (grouped.pieces.size() != expected || !checkOrder(points, grouped, count)) { return false; }
                if (!checkLowering(points, numerical, loop, body->getArgument(0))) { return false; }
                // A residual exclusive with most points does not impose their
                // order. The oracle checks both its executed and skipped cases.
                points[position].members.front().coordinates.push_back({loop, 0});
                grouped = fs::buildMixedEndpointPieces(points, numerical);
                if (grouped.pieces.size() > 3 || !checkOrder(points, grouped, count)) { return false; }
            }
        }
    }
    auto a = residual, b = residual;
    a.members = {{7, {{loop, 0}}}}; b.members = {{8, {{loop, 0}}}};
    auto simultaneous = fs::buildMixedEndpointPieces({a, b}, {1, 1});
    if (simultaneous.pieces.size() != 2 || !checkOrder({a, b}, simultaneous, 1)) { return false; }
    if (fs::buildMixedEndpointPieces({a}, {}).error.empty() ||
        fs::buildMixedEndpointPieces({a, a}, {1, 0}).error.empty()) { return false; }
    // A few numerical candidates among many opaque commands must not require
    // a dense opaque-pair graph. All opaque commands execute together here.
    SmallVector<fs::EndpointPiece> sparse{a};
    SmallVector<uint8_t> flags{1};
    for (unsigned i = 0; i < 4096; ++i) {
        auto point = residual;
        point.members.front().record = 1000 + i;
        sparse.push_back(point); flags.push_back(0);
    }
    b.members.front().coordinates.front().induction = 1;
    sparse.push_back(b); flags.push_back(1);
    auto sparseGroups = fs::buildMixedEndpointPieces(sparse, flags);
    return sparseGroups.pieces.size() == sparse.size() && checkOrder(sparse, sparseGroups, 2);
}
