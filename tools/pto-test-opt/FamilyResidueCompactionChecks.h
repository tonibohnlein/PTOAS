// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent unfolding of frozen family contributions, before and after SSA.
#ifndef PTO_FAMILY_RESIDUE_COMPACTION_CHECKS_H
#define PTO_FAMILY_RESIDUE_COMPACTION_CHECKS_H
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include <optional>
#include <vector>
namespace {
std::optional<uint64_t> residueScalar(mlir::Value value, mlir::Value argument, bool selected)
{
    using namespace mlir;
    if (value == argument) { return selected; }
    auto* operation = value.getDefiningOp();
    if (!operation) { return std::nullopt; }
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        return cast<IntegerAttr>(constant.getValue()).getValue().getZExtValue();
    }
    SmallVector<uint64_t> inputs;
    for (auto operand : operation->getOperands()) {
        auto result = residueScalar(operand, argument, selected);
        if (!result) { return std::nullopt; }
        inputs.push_back(*result);
    }
    if (isa<arith::SelectOp>(operation)) { return inputs[0] ? inputs[1] : inputs[2]; }
    if (isa<arith::OrIOp>(operation)) { return inputs[0] | inputs[1]; }
    if (isa<arith::XOrIOp>(operation)) { return inputs[0] ^ inputs[1]; }
    if (isa<arith::AndIOp>(operation)) { return inputs[0] & inputs[1]; }
    return std::nullopt;
}
bool checkResidueCommands(mlir::func::FuncOp function, unsigned mode)
{
    using namespace mlir;
    unsigned setSites = 0, waitSites = 0;
    function.walk([&](pto::LogicalSetOp) { ++setSites; });
    function.walk([&](pto::LogicalWaitOp) { ++waitSites; });
    if (setSites != (mode == 5 ? 3U : mode ? 2U : 1U) || waitSites != (mode == 5 ? 3U : 2U)) {
        return false;
    }
    for (bool selected : {false, true}) {
        std::vector<std::vector<uint64_t>> sets, waits;
        bool valid = true;
        function.walk([&](Operation* operation) {
            if (!isa<pto::LogicalSetOp, pto::LogicalWaitOp>(operation)) { return; }
            auto branch = operation->getParentOfType<scf::IfOp>();
            auto active = branch ? residueScalar(branch.getCondition(), function.getArgument(0), selected) :
                std::optional<uint64_t>{};
            if (!active) { valid = false; return; }
            if (!*active) { return; }
            std::vector<uint64_t> tuple;
            for (auto operand : operation->getOperands()) {
                auto value = residueScalar(operand, function.getArgument(0), selected);
                if (!value) { valid = false; return; }
                tuple.push_back(*value);
            }
            (isa<pto::LogicalSetOp>(operation) ? sets : waits).push_back(std::move(tuple));
        });
        if (mode == 5) {
            const std::vector<std::vector<uint64_t>> expected = selected ?
                std::vector<std::vector<uint64_t>>{{0, 0}, {0, 2}} :
                std::vector<std::vector<uint64_t>>{{0, 2}, {0, 1}};
            if (sets != expected) { return false; }
            llvm::sort(sets); llvm::sort(waits);
        }
        if (!valid || sets != waits || sets.size() != (mode == 1 || mode == 5 ? 2U : 1U)) { return false; }
        for (const auto& tuple : sets) {
            if (tuple.size() < 2 || tuple[1] > (mode == 5 ? 2U : 1U) ||
                (mode != 1 && mode != 5 && tuple[1] != (selected ? 0U : 1U))) {
                return false;
            }
        }
    }
    return true;
}
bool checkResidueCompaction(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    // Shared residue evidence admits exclusive opaque guards only. Coactive
    // records, absent evidence, different ordinals and different tuples survive.
    for (unsigned mode = 0; mode < 6; ++mode) {
        OpBuilder builder(context);
        auto location = builder.getUnknownLoc();
        OwningOpRef<func::FuncOp> function(func::FuncOp::create(location, "residue_compaction",
            builder.getFunctionType({builder.getI1Type()}, {})));
        auto* body = function->addEntryBlock();
        builder.setInsertionPointToEnd(body);
        auto* source = builder.create<arith::ConstantIndexOp>(location, 11).getOperation();
        SmallVector<Operation*> targets;
        for (unsigned i = 0; i < (mode == 5 ? 3U : 2U); ++i) {
            targets.push_back(builder.create<arith::ConstantIndexOp>(location, 12 + i).getOperation());
        }
        builder.create<func::ReturnOp>(location);
        auto arena = std::make_shared<fs::RegionExpressions>();
        const auto guard = arena->input(body->getArgument(0)), zero = arena->constant(0);
        const auto tag = arena->select(guard, zero, arena->constant(1));
        fs::PreparedLogicalPlan plan(0);
        plan.groupedFamilies = plan.independentPieces = true;
        plan.nestedIdentities = mode == 4;
        SmallVector<fs::ContextualEndpointFamily> endpoints;
        for (unsigned record : mode == 5 ? SmallVector<unsigned>{0, 2, 1} : SmallVector<unsigned>{0, 1}) {
            const bool intervening = record == 2;
            fs::EndpointFamily family;
            family.id = record; family.sourcePipe = 4; family.targetPipe = intervening ? 2 : 3;
            family.sourceCut = {body, source}; family.targetCut = {body, targets[record]};
            family.members.push_back({record, 0, record + 1, {}, {}}); plan.families.push_back(family);
            for (bool publish : {true, false}) {
                fs::ContextualEndpointFamily endpoint;
                endpoint.piece.kind = publish ? fs::EndpointKind::Set : fs::EndpointKind::Wait;
                endpoint.piece.sourcePipe = 4; endpoint.piece.targetPipe = family.targetPipe;
                endpoint.piece.namespaceId = intervening ? 2 : 0;
                endpoint.piece.cut = publish ? family.sourceCut : family.targetCut;
                endpoint.piece.order = endpoints.size(); endpoint.piece.members.push_back({record, {}});
                endpoint.activity = mode == 1 || intervening ? arena->boolean(true) :
                    record ? arena->lnot(guard) : guard;
                endpoint.ordinal = mode == 3 ? arena->constant(record) : zero;
                if (mode != 2 && !intervening) {
                    endpoint.residueCoordinate = mode == 1 ? zero : tag;
                    endpoint.modulus = 2; endpoint.residue = mode == 1 ? 0 : record;
                }
                if (mode == 4) { endpoint.matchingCoordinates.push_back(arena->constant(record)); }
                endpoints.push_back(std::move(endpoint));
            }
        }
        auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints);
        if (failed(recipe)) { return false; }
        plan.familyRecipe = *recipe;
        if (failed(fs::insertLogicalSynchronization(*function, plan)) ||
            failed(verify(*function)) || !checkResidueCommands(*function, mode)) { return false; }
    }
    return true;
}
} // namespace
#endif
