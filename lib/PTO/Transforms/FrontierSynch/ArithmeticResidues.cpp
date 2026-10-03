// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact substitution x = P*q+r, with checked fixed-width coefficients.
#include "ArithmeticProgramInternal.h"
#include "ArithmeticRows.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/MathExtras.h"
namespace mlir::pto::frontiersynch::detail {
PrimitiveRelation ProgramBuilder::relation(PrimitiveKind kind, unsigned dimensions) const
{
    PrimitiveRelation result;
    result.kind = kind;
    result.dimensions = dimensions;
    for (unsigned i = 0; i < dimensions; ++i) {
        result.coordinates.push_back({"i" + std::to_string(i), CoordinateKind::Occurrence});
    }
    for (const auto& name : output.primitives.parameters) {
        result.coordinates.push_back({name, CoordinateKind::Parameter});
    }
    return result;
}
AffineExpr ProgramBuilder::value(Value input, const ArithmeticSite& site, unsigned offset) const
{
    APInt number;
    if (matchPattern(input, m_ConstantInt(&number)) && number.isSignedIntN(64)) {
        return getAffineConstantExpr(number.getSExtValue(), context);
    }
    auto parameter = parameterIds.find(input);
    if (parameter != parameterIds.end()) {
        return getAffineSymbolExpr(parameter->second, context);
    }
    for (auto [id, storedLoop] : llvm::enumerate(site.loops)) {
        auto loop = storedLoop;
        if (input == loop.getInductionVar()) {
            return getAffineDimExpr(offset + id, context);
        }
    }
    return {};
}
SmallVector<AffineExpr> ProgramBuilder::domain(const ArithmeticSite& site, unsigned offset) const
{
    SmallVector<AffineExpr> rows;
    for (auto [id, storedLoop] : llvm::enumerate(site.loops)) {
        auto loop = storedLoop;
        const auto iv = getAffineDimExpr(offset + id, context);
        rows.push_back(iv);
        auto upper = value(loop.getUpperBound(), site, offset);
        auto constant = dyn_cast<AffineConstantExpr>(upper);
        // Avoid constructing INT64_MIN-1 for an already empty domain.
        rows.push_back(constant && constant.getValue() <= 0 ? getAffineConstantExpr(-1, context) : upper - iv - 1);
    }
    return rows;
}
namespace {
std::optional<AffineExpr> substitute(const LinearRow& row, const PrimitiveRelation& relation,
                                    ArrayRef<uint64_t> residues, uint64_t period, MLIRContext* context)
{
    int64_t constant = row.constant;
    auto expression = getAffineConstantExpr(0, context);
    for (auto [id, coefficient] : llvm::enumerate(row.coefficients)) {
        int64_t shift = 0, scale = 0;
        if (llvm::MulOverflow(coefficient, static_cast<int64_t>(residues[id]), shift) ||
            llvm::AddOverflow(constant, shift, constant) ||
            llvm::MulOverflow(coefficient, static_cast<int64_t>(period), scale)) {
            return std::nullopt;
        }
        auto coordinate = id < relation.dimensions ? getAffineDimExpr(id, context) :
                          getAffineSymbolExpr(id - relation.dimensions, context);
        expression = expression + coordinate * scale;
    }
    return expression + constant;
}
}
void ProgramBuilder::emit(PrimitiveRelation& target, ArrayRef<AffineExpr> rows,
                         std::optional<std::pair<unsigned, uint64_t>> filter, uint64_t modulus)
{
    const auto count = target.coordinates.size();
    if (count > limits.dimensions) {
        output.extraction.note(RecognitionIssue::ArithmeticDimension, nullptr, true);
        return;
    }
    const auto period = limits.period;
    std::size_t combinations = 1;
    for (std::size_t i = 0; i < count; ++i) {
        if (combinations > SIZE_MAX / period) {
            output.extraction.note(RecognitionIssue::ArithmeticDimension, nullptr, true);
            return;
        }
        combinations *= period;
    }
    SmallVector<LinearRow> collected;
    for (auto expression : rows) {
        LinearRow row;
        if (collectRow(expression, target.dimensions, count - target.dimensions, row)) {
            output.extraction.note(RecognitionIssue::IndexArithmetic, nullptr);
            return;
        }
        collected.push_back(std::move(row));
    }
    // Iterative mixed-radix enumeration: fixed P and D bound its expansion.
    SmallVector<uint64_t> residues(count, 0);
    bool more = true;
    while (more) {
        if (!filter || residues[filter->first] % modulus == filter->second) {
            SmallVector<AffineExpr> constraints;
            for (const auto& row : collected) {
                auto expression = substitute(row, target, residues, period, context);
                if (!expression) {
                    output.extraction.note(RecognitionIssue::IndexArithmetic, nullptr);
                    return;
                }
                constraints.push_back(*expression);
            }
            if (constraints.empty()) {
                constraints.push_back(getAffineConstantExpr(0, context));
            }
            SmallVector<bool> equalities(constraints.size(), false);
            target.pieces.push_back({IntegerSet::get(target.dimensions, count - target.dimensions,
                                                    constraints, equalities), residues});
        }
        more = false;
        for (auto& residue : residues) {
            if (++residue < period) {
                more = true;
                break;
            }
            residue = 0;
        }
    }
}
} // namespace mlir::pto::frontiersynch::detail
