// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Fixed-coordinate evaluation shared by normalization and finite adapters.
#include "NormalizedControl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "../InsertSync/SyncScalarEvolution.h"
#include "../InsertSync/SyncScalarReplay.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
namespace mlir::pto::frontiersynch::detail {
namespace {
// This memo belongs to exactly one fixed-coordinate environment. Dialect
// folders operate on detached copies and never rewrite the shared input IR.
class FixedCondition {
public:
    FixedCondition(const ArithmeticSite& site, uint64_t& work) : site(site), work(work) {}
    std::optional<bool> evaluate(Value value)
    {
        auto result = dyn_cast_or_null<IntegerAttr>(fold(value, 0));
        if (!result || !result.getType().isInteger(1)) { return std::nullopt; }
        return result.getValue().isOne();
    }
private:
    const ArithmeticSite& site;
    uint64_t& work;
    DenseMap<Value, Attribute> memo;
    Attribute fold(Value value, unsigned depth)
    {
        if (!value || depth >= 64) { return {}; }
        auto inserted = memo.try_emplace(value, Attribute{});
        if (!inserted.second) { return inserted.first->second; }
        auto remember = [&](Attribute result) { memo[value] = result; return result; };
        for (auto coordinate : site.fixedCoordinates) {
            if (value == coordinate.loop.getInductionVar()) {
                return remember(IntegerAttr::get(value.getType(), coordinate.induction));
            }
        }
        Attribute literal;
        if (matchPattern(value, m_Constant(&literal))) {
            // Poison/undefined fold results are not concrete scalar values.
            if (isa<IntegerAttr, FloatAttr>(literal)) { return remember(literal); }
            return {};
        }
        auto* operation = value.getDefiningOp();
        const bool replay = mlir::pto::detail::canReplayScalar(operation);
        const bool arithmetic = replay && isa<arith::ArithDialect>(operation->getDialect());
        const bool singleResult = arithmetic && operation->getNumResults() == 1;
        if (!singleResult) {
            return {};
        }
        // LLVM folders may ignore poison-producing overflow/fast-math
        // promises. Until separately proved, those operations stay symbolic.
        if (auto flags = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(operation)) {
            const bool unflagged = flags.getOverflowAttr().getValue() == arith::IntegerOverflowFlags::none;
            if (!unflagged) { return {}; }
        }
        if (auto flags = dyn_cast<arith::ArithFastMathInterface>(operation)) {
            const bool unflagged = flags.getFastMathFlagsAttr().getValue() == arith::FastMathFlags::none;
            if (!unflagged) { return {}; }
        }
        const bool usesIndex = value.getType().isIndex() ||
            llvm::any_of(operation->getOperandTypes(), [](Type type) { return type.isIndex(); });
        if (usesIndex) {
            const auto width = DataLayout::closest(operation).getTypeSizeInBits(IndexType::get(value.getContext()));
            const bool supportedWidth = !width.isScalable() &&
                width.getFixedValue() == IndexType::kInternalStorageBitWidth;
            if (!supportedWidth) { return {}; }
        }
        SmallVector<Attribute> operands;
        for (Value operand : operation->getOperands()) {
            auto constant = fold(operand, depth + 1);
            if (!constant) { return {}; }
            operands.push_back(constant);
        }
        if (work != UINT64_MAX) { ++work; }
        OwningOpRef<Operation*> copy(operation->cloneWithoutRegions());
        SmallVector<OpFoldResult> results;
        const bool folded = succeeded(copy->fold(operands, results));
        const bool singleFold = folded && results.size() == 1;
        if (!singleFold) { return {}; }
        auto result = dyn_cast<Attribute>(results.front());
        auto typed = dyn_cast_or_null<TypedAttr>(result);
        const bool scalar = typed && isa<IntegerAttr, FloatAttr>(result);
        const bool sameType = scalar && typed.getType() == value.getType();
        if (!sameType) { return {}; }
        return remember(result);
    }
};
} // namespace
std::optional<int64_t> normalizedInteger(Value value, ArrayRef<FixedLoopCoordinate> coordinates, MLIRContext* context)
{
    mlir::pto::detail::ScalarEvolution evolution(context, value.getDefiningOp() ? value.getDefiningOp() :
        value.getParentRegion()->getParentOp());
    auto expression = evolution.value(value, [&](Value current) -> AffineExpr {
        for (auto fixed : coordinates) {
            auto loop = fixed.loop;
            if (current == loop.getInductionVar()) {
                return getAffineConstantExpr(fixed.induction, context);
            }
        }
        return {};
    });
    auto constant = dyn_cast_or_null<AffineConstantExpr>(expression);
    return constant ? std::optional<int64_t>(constant.getValue()) : std::nullopt;
}
std::optional<bool> normalizedCondition(Value value, const ArithmeticSite& site, uint64_t& work)
{
    return FixedCondition(site, work).evaluate(value);
}
} // namespace mlir::pto::frontiersynch::detail
