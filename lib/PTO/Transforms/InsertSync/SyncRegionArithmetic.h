// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// LLVM 19 affine simplification folds nested int64 coefficients unchecked.
// Preflight coefficient magnitudes before constructing or substituting maps.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCREGIONARITHMETIC_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCREGIONARITHMETIC_H
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"
#include <optional>
namespace mlir::pto::detail {
inline std::optional<int64_t> magnitude(AffineExpr expression, llvm::ArrayRef<int64_t> coordinates = {})
{
    if (!expression) {
        return std::nullopt;
    }
    if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
        auto value = constant.getValue();
        return value == INT64_MIN ? std::nullopt : std::optional<int64_t>(value < 0 ? -value : value);
    }
    if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
        return coordinates.empty() ? std::optional<int64_t>(1) :
            (dimension.getPosition() < coordinates.size() ?
             std::optional<int64_t>(coordinates[dimension.getPosition()]) : std::nullopt);
    }
    if (isa<AffineSymbolExpr>(expression)) {
        return 1;
    }
    auto binary = cast<AffineBinaryOpExpr>(expression);
    auto left = magnitude(binary.getLHS(), coordinates), right = magnitude(binary.getRHS(), coordinates);
    int64_t result = 0;
    if (!left || !right) {
        return std::nullopt;
    }
    if (expression.getKind() == AffineExprKind::Add) {
        return llvm::AddOverflow(*left, *right, result) ? std::nullopt : std::optional<int64_t>(result);
    }
    if (expression.getKind() == AffineExprKind::Mul) {
        return llvm::MulOverflow(*left, *right, result) ? std::nullopt : std::optional<int64_t>(result);
    }
    // Division/modulo coefficients are not multiplied. Conservatively keep
    // both operands' magnitudes; this is a safety preflight, not an address bound.
    return std::max(*left, *right);
}
inline AffineExpr checkedAdd(AffineExpr a, AffineExpr b)
{
    auto x = magnitude(a), y = magnitude(b);
    int64_t result = 0;
    return x && y && !llvm::AddOverflow(*x, *y, result) ? a + b : AffineExpr{};
}
inline AffineExpr checkedMul(AffineExpr a, AffineExpr b)
{
    auto x = magnitude(a), y = magnitude(b);
    int64_t result = 0;
    return x && y && !llvm::MulOverflow(*x, *y, result) ? a * b : AffineExpr{};
}
inline AffineExpr substitute(AffineExpr expression, ArrayRef<AffineExpr> dimensions,
                             ArrayRef<AffineExpr> symbols, unsigned depth = 0)
{
    if (!expression || depth > 64) {
        return {};
    }
    if (auto d = dyn_cast<AffineDimExpr>(expression)) {
        return d.getPosition() < dimensions.size() ? dimensions[d.getPosition()] : AffineExpr{};
    }
    if (auto s = dyn_cast<AffineSymbolExpr>(expression)) {
        return s.getPosition() < symbols.size() ? symbols[s.getPosition()] : AffineExpr{};
    }
    if (isa<AffineConstantExpr>(expression)) {
        return expression;
    }
    auto binary = cast<AffineBinaryOpExpr>(expression);
    auto a = substitute(binary.getLHS(), dimensions, symbols, depth + 1);
    auto b = substitute(binary.getRHS(), dimensions, symbols, depth + 1);
    if (!a || !b) {
        return {};
    }
    if (expression.getKind() == AffineExprKind::Add) {
        return checkedAdd(a, b);
    }
    if (expression.getKind() == AffineExprKind::Mul) {
        return checkedMul(a, b);
    }
    auto divisor = dyn_cast<AffineConstantExpr>(b);
    if (!divisor || divisor.getValue() <= 0) {
        return {};
    }
    switch (expression.getKind()) {
    case AffineExprKind::Mod: return a % divisor.getValue();
    case AffineExprKind::FloorDiv: return a.floorDiv(divisor.getValue());
    case AffineExprKind::CeilDiv: return a.ceilDiv(divisor.getValue());
    default: return {};
    }
}
} // namespace mlir::pto::detail
#endif
