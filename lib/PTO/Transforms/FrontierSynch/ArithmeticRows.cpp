// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Integer row normalization preserves inequalities by floor division; an
// indivisible equality makes its conjunction empty rather than being rounded.
#include "ArithmeticRows.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

namespace mlir::pto::frontiersynch::detail {
namespace {
using Form = SmallVector<int64_t>;
std::optional<ArithmeticIssue> combine(AffineBinaryOpExpr expr, const Form& left, const Form& right, Form& result)
{
    if (expr.getKind() == AffineExprKind::Add) {
        for (std::size_t i = 0; i < left.size(); ++i) {
            if (llvm::AddOverflow(left[i], right[i], result[i])) {
                return ArithmeticIssue::CoefficientOverflow;
            }
        }
        return std::nullopt;
    }
    if (expr.getKind() != AffineExprKind::Mul) {
        return ArithmeticIssue::UnsupportedExpression;
    }
    auto constant = [](const Form& form) {
        return llvm::all_of(ArrayRef<int64_t>(form).drop_back(), [](int64_t x) { return x == 0; });
    };
    const bool leftConstant = constant(left);
    if (!leftConstant && !constant(right)) {
        return ArithmeticIssue::UnsupportedExpression;
    }
    const auto scale = leftConstant ? left.back() : right.back();
    const auto& variable = leftConstant ? right : left;
    for (std::size_t i = 0; i < variable.size(); ++i) {
        if (llvm::MulOverflow(variable[i], scale, result[i])) {
            return ArithmeticIssue::CoefficientOverflow;
        }
    }
    return std::nullopt;
}
}
std::optional<ArithmeticIssue> collectRow(AffineExpr expression, unsigned dimensions, unsigned symbols, LinearRow& row)
{
    const std::size_t count = static_cast<std::size_t>(dimensions) + symbols;
    DenseMap<AffineExpr, Form> forms;
    SmallVector<std::pair<AffineExpr, bool>> stack{{expression, false}};
    while (!stack.empty()) {
        auto [expr, finish] = stack.pop_back_val();
        if (forms.count(expr)) {
            continue;
        }
        auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
        if (binary && !finish) {
            stack.push_back({expr, true});
            stack.push_back({binary.getRHS(), false});
            stack.push_back({binary.getLHS(), false});
            continue;
        }
        Form form(count + 1, 0);
        if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
            if (dim.getPosition() >= dimensions) {
                return ArithmeticIssue::InvalidCoordinate;
            }
            form[dim.getPosition()] = 1;
        } else if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) {
            if (symbol.getPosition() >= symbols) {
                return ArithmeticIssue::InvalidCoordinate;
            }
            form[dimensions + symbol.getPosition()] = 1;
        } else if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
            form.back() = constant.getValue();
        } else if (binary) {
            if (auto issue = combine(binary, forms.find(binary.getLHS())->second,
                                     forms.find(binary.getRHS())->second, form)) {
                return issue;
            }
        } else {
            return ArithmeticIssue::UnsupportedExpression;
        }
        forms.try_emplace(expr, std::move(form));
    }
    auto form = forms.find(expression);
    row.coefficients.assign(form->second.begin(), form->second.end() - 1);
    row.constant = form->second.back();
    return std::nullopt;
}

uint64_t magnitude(int64_t value)
{
    return value < 0 ? static_cast<uint64_t>(-(value + 1)) + 1 : static_cast<uint64_t>(value);
}
bool normalizeRow(LinearRow& row)
{
    uint64_t divisor = 0;
    for (auto coefficient : row.coefficients) {
        divisor = std::gcd(divisor, magnitude(coefficient));
    }
    if (!divisor) {
        return row.equality ? row.constant != 0 : row.constant < 0;
    }
    // Widen first: |INT64_MIN| need not fit a signed machine coefficient.
    APInt denominator(128, divisor);
    APInt constant(128, static_cast<uint64_t>(row.constant), true);
    const auto remainder = constant.srem(denominator);
    if (row.equality && !remainder.isZero()) {
        return true;
    }
    auto quotient = constant.sdiv(denominator);
    if (!row.equality && remainder.isNegative()) {
        --quotient;
    }
    row.constant = quotient.getSExtValue();
    for (auto& coefficient : row.coefficients) {
        coefficient = APInt(128, static_cast<uint64_t>(coefficient), true).sdiv(denominator).getSExtValue();
    }
    return false;
}
ArithmeticClass classifyRow(const LinearRow& row)
{
    unsigned nonzero = 0;
    int sum = 0;
    for (auto coefficient : row.coefficients) {
        if (!coefficient) {
            continue;
        }
        if (magnitude(coefficient) != 1 || ++nonzero > 2) {
            return ArithmeticClass::BoundedCoefficients;
        }
        sum += static_cast<int>(coefficient);
    }
    return nonzero < 2 || sum == 0 ? ArithmeticClass::Differences : ArithmeticClass::Octagons;
}
} // namespace mlir::pto::frontiersynch::detail
