// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent unfolding of frozen family contributions, before and after SSA.
#ifndef PTO_FAMILY_CONSTRUCTION_CHECKS_H
#define PTO_FAMILY_CONSTRUCTION_CHECKS_H
#include "PTO/Transforms/FrontierSynch/Shared/DeferredFamily.h"
namespace {
bool checkNumericalConstructionSchedules(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    // Supplied maps isolate lowering/count parity. These fixtures do not claim
    // to derive reuse certificates from payload dependence analysis.
    for (unsigned mode : {0U, 1U, 2U}) {
        OpBuilder b(context);
        const auto loc = b.getUnknownLoc();
        OwningOpRef<func::FuncOp> f(func::FuncOp::create(loc, "numerical_construction",
            b.getFunctionType({b.getIndexType(), b.getI1Type()}, {})));
        auto* body = f->addEntryBlock(); b.setInsertionPointToEnd(body);
        auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
        auto one = b.create<arith::ConstantIndexOp>(loc, 1);
        auto two = b.create<arith::ConstantIndexOp>(loc, 2);
        auto three = b.create<arith::ConstantIndexOp>(loc, 3);
        auto loop = b.create<scf::ForOp>(loc, one, body->getArgument(0), two);
        b.create<func::ReturnOp>(loc);
        b.setInsertionPointToStart(loop.getBody());
        auto inner = b.create<scf::ForOp>(loc, zero, three, one);
        b.setInsertionPointToStart(inner.getBody());
        auto* setCut = b.create<arith::ConstantIndexOp>(loc, 7).getOperation();
        auto* waitCut = b.create<arith::ConstantIndexOp>(loc, 8).getOperation();
        auto arena = std::make_shared<fs::RegionExpressions>();
        const auto point = arena->input(inner.getInductionVar());
        const auto activity = arena->land(arena->input(body->getArgument(1)), arena->lt(point, arena->constant(2)));
        std::vector<fs::EndpointPiece> pieces;
        SmallVector<fs::ContextualEndpointFamily> mixed;
        for (auto kind : {fs::EndpointKind::Set, fs::EndpointKind::Wait}) {
            fs::EndpointPiece piece;
            piece.kind = kind; piece.sourcePipe = 1; piece.targetPipe = 2; piece.displacement = 1;
            piece.cut = {inner.getBody(), kind == fs::EndpointKind::Set ? setCut : waitCut};
            piece.order = pieces.size(); piece.members.push_back({0, {{inner, 1}}});
            pieces.push_back(piece);
            fs::ContextualEndpointFamily endpoint;
            endpoint.piece = piece; endpoint.activity = arena->eq(point, arena->constant(1));
            mixed.push_back(endpoint);
        }
        auto recipe = fs::restoreNumericalFamilyRecipe({loop, pieces});
        if (mode == 2) {
            fs::ContextualFamilyExpressions expressions(arena);
            recipe = fs::buildMixedFamilyEndpointRecipe(expressions, mixed, {1, 1}, loop);
        }
        if (failed(recipe)) { return false; }
        if (mode) {
            recipe = fs::prefixFamilyInvocation(*recipe, arena, arena->input(body->getArgument(0)));
            if (failed(recipe)) { return false; }
            recipe = fs::prefixFamilyInvocation(*recipe, arena, arena->constant(7));
            if (failed(recipe)) { return false; }
        }
        // Repeated cached wrapper roots still construct separate false/select
        // masks, and this entire leaf finishes before the later barrier leaf.
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            recipe = fs::maskFamilyActivity(*recipe, arena, activity);
            if (failed(recipe)) { return false; }
        }
        recipe = fs::appendFamilyEndpointRecipe({}, *recipe, 7, 0);
        if (failed(recipe)) { return false; }
        fs::ContextualEndpointFamily barrier;
        barrier.piece.kind = fs::EndpointKind::Barrier;
        barrier.piece.cut = {inner.getBody(), setCut};
        barrier.piece.sourcePipe = barrier.piece.targetPipe = 1;
        barrier.piece.members.push_back({8, {}}); barrier.activity = activity;
        auto barrierLeaf = fs::buildContextualFamilyEndpointRecipe(arena, {barrier});
        if (failed(barrierLeaf)) { return false; }
        recipe = fs::appendFamilyEndpointRecipe(*recipe, *barrierLeaf, 0, 0);
        if (failed(recipe)) { return false; }
        fs::PreparedLogicalPlan plan(0);
        plan.familyRecipe = *recipe; plan.nestedIdentities = bool(mode);
        fs::EndpointFamily family;
        family.id = 7; family.sourcePipe = 1; family.targetPipe = 2; family.displacement = 1;
        family.sourceCut = {inner.getBody(), setCut}; family.targetCut = {inner.getBody(), waitCut};
        family.members.push_back({7, 0, 1, {{inner, 1}}, {{inner, 1}}});
        plan.families.push_back(family);
        fs::EndpointFamily local;
        local.id = 8; local.sourcePipe = local.targetPipe = 1; local.local = true;
        local.sourceCut = local.targetCut = {inner.getBody(), setCut};
        local.members.push_back({8, 0, 0, {}, {}}); plan.families.push_back(local);
        auto direction = b.getDictionaryAttr({b.getNamedAttr("source", b.getI64IntegerAttr(1)),
            b.getNamedAttr("target", b.getI64IntegerAttr(2)),
            b.getNamedAttr("budget", b.getI64IntegerAttr(mode == 1 ? 1 : 2)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr({7}))});
        NamedAttrList certificate;
        certificate.set("version", b.getI64IntegerAttr(2)); certificate.set("plan", b.getI64IntegerAttr(0));
        if (mode == 2) {
            NamedAttrList group(direction);
            group.set("strides", b.getDenseI64ArrayAttr({0})); group.set("phases", b.getDenseI64ArrayAttr({0}));
            group.set("conflicts", b.getDenseI64ArrayAttr({}));
            auto rule = b.getDictionaryAttr({b.getNamedAttr("coordinate_count", b.getI64IntegerAttr(3)),
                b.getNamedAttr("base", b.getI64IntegerAttr(0)),
                b.getNamedAttr("terms", b.getArrayAttr({b.getDenseI64ArrayAttr({2, 1, 0, 2, 1})}))});
            group.set("tuple_rules", b.getArrayAttr({rule}));
            certificate.set("strategy", b.getStringAttr("regional-palettes"));
            certificate.set("groups", b.getArrayAttr({group.getDictionary(context)}));
        } else { certificate.set("directions", b.getArrayAttr({direction})); }
        plan.allocationCertificate = certificate.getDictionary(context);
        if (failed(fs::planPhysicalFamilyOutput(*f, plan, {0, 1})) || !checkCountedPhysicalFamily(plan, b)) {
            return false;
        }
        // Even when the map drops ordinal, displacement must retain trip and
        // ordinal arithmetic. The independent oracle covers partial nonunit
        // visits and both activity valuations; zero trips have no visits.
        unsigned divisions = 0;
        for (const auto& stage : plan.preparation) {
            for (auto& operation : *stage.code) { divisions += isa<arith::DivUIOp>(operation); }
        }
        if (divisions < 2) { return false; }
        for (uint64_t n : {0U, 1U, 2U, 5U, 8U}) {
            const uint64_t trips = n > 1 ? (n - 1 + 1) / 2 : 0;
            for (uint64_t iv = 1; iv < n; iv += 2) {
                for (uint64_t j = 0; j < 3; ++j) {
                    for (bool gate : {false, true}) {
                        DenseMap<Value, uint64_t> values;
                        values[body->getArgument(0)] = n; values[body->getArgument(1)] = gate;
                        values[loop.getInductionVar()] = iv; values[inner.getInductionVar()] = j;
                        f->walk([&](arith::ConstantOp constant) {
                            values[constant.getResult()] = cast<IntegerAttr>(constant.getValue()).getInt();
                        });
                        for (const auto& stage : plan.preparation) {
                            if (!evaluateProtocolBlock(*stage.code, values)) { return false; }
                        }
                        const uint64_t ordinal = (iv - 1) / 2;
                        for (const auto& endpoint : plan.endpoints) {
                            const bool local = endpoint.kind == fs::LogicalCommandKind::Barrier;
                            const bool publish = endpoint.kind == fs::LogicalCommandKind::Set;
                            const bool expected = gate && j < 2 && (local ||
                                (j == 1 && (publish ? ordinal + 1 < trips : ordinal >= 1)));
                            if (values.lookup(endpoint.guard) != expected) { return false; }
                            if (local || !expected) { continue; }
                            const auto sourceOrdinal = publish ? ordinal : ordinal - 1;
                            if (endpoint.identity && values.lookup(endpoint.identity) != sourceOrdinal) {
                                return false;
                            }
                            if (mode && endpoint.memberCoordinates.size() != 3) { return false; }
                            Block code; b.setInsertionPointToEnd(&code);
                            SmallVector<Value> tuple{endpoint.identity};
                            tuple.append(endpoint.memberCoordinates.begin() + 1, endpoint.memberCoordinates.end());
                            auto id = fs::emitPhysicalEndpointMap(b, loc, *endpoint.physicalMapping,
                                tuple, endpoint.memberCoordinates.front());
                            if (!evaluateProtocolBlock(code, values) || values.lookup(id) !=
                                (mode == 1 ? 0 : mode == 2 ? n % 2 : sourceOrdinal % 2)) { return false; }
                        }
                    }
                }
            }
        }
    }
    return true;
}
} // namespace
#endif
