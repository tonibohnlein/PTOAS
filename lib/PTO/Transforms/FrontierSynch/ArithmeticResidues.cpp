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
#include "../InsertSync/SyncScalarEvolution.h"
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
namespace {
AffineExpr normalizeValue(Value input, const ArithmeticSite& site, unsigned offset,
                          MLIRContext* context, llvm::function_ref<AffineExpr(Value)> parameter)
{
    mlir::pto::detail::ScalarEvolution evolution(context, input.getDefiningOp() ? input.getDefiningOp() :
        input.getParentRegion()->getParentOp());
    return evolution.value(input, [&](Value value) -> AffineExpr {
        for (auto [id, storedLoop] : llvm::enumerate(site.loops)) {
            auto loop = storedLoop;
            if (value == loop.getInductionVar()) {
                return getAffineDimExpr(offset + id, context);
            }
        }
        return parameter(value);
    });
}
}
bool ProgramBuilder::entryParameter(Value value) const
{
    if (!value.getType().isIndex() && !value.getType().isInteger(1)) {
        return false;
    }
    auto argument = dyn_cast<BlockArgument>(value);
    if (argument && argument.getOwner() == &output.context.function.front()) {
        return true;
    }
    auto* root = output.context.root;
    return root != output.context.function.getOperation() && index.valueAvailable(value, root, Boundary::Before);
}
AffineExpr ProgramBuilder::registerParameter(Value value)
{
    auto inserted = parameterIds.try_emplace(value, output.parameters.size());
    if (inserted.second) {
        output.parameters.push_back(value);
        output.primitives.parameters.push_back("p" + std::to_string(inserted.first->second));
    }
    return getAffineSymbolExpr(inserted.first->second, context);
}
bool ProgramBuilder::prepareValue(Value input, const ArithmeticSite& site)
{
    auto expression = normalizeValue(input, site, 0, context, [&](Value value) -> AffineExpr {
        return entryParameter(value) ? registerParameter(value) : AffineExpr{};
    });
    return static_cast<bool>(expression);
}
AffineExpr ProgramBuilder::value(Value input, const ArithmeticSite& site, unsigned offset) const
{
    return normalizeValue(input, site, offset, context, [&](Value input) -> AffineExpr {
        auto parameter = parameterIds.find(input);
        return parameter == parameterIds.end() ? AffineExpr{} : getAffineSymbolExpr(parameter->second, context);
    });
}
bool ProgramBuilder::staticallyEmpty(const ArithmeticSite& site) const
{
    for (auto storedLoop : site.loops) {
        auto loop = storedLoop;
        APInt lower, upper;
        const bool known = matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
                           matchPattern(loop.getUpperBound(), m_ConstantInt(&upper));
        if (known && lower.sge(upper)) {
            return true;
        }
    }
    return false;
}
SmallVector<AffineExpr> ProgramBuilder::domain(const ArithmeticSite& site, unsigned offset) const
{
    SmallVector<AffineExpr> rows;
    if (staticallyEmpty(site)) {
        rows.push_back(getAffineConstantExpr(-1, context));
        return rows;
    }
    for (auto [id, storedLoop] : llvm::enumerate(site.loops)) {
        auto loop = storedLoop;
        const auto iv = getAffineDimExpr(offset + id, context);
        auto lower = value(loop.getLowerBound(), site, offset);
        auto upper = value(loop.getUpperBound(), site, offset);
        auto step = cast<AffineConstantExpr>(value(loop.getStep(), site, offset)).getValue();
        auto distance = mlir::pto::detail::checkedAdd(iv, mlir::pto::detail::checkedMul(
            lower, getAffineConstantExpr(-1, context)));
        rows.push_back(distance);
        auto constant = dyn_cast<AffineConstantExpr>(upper);
        // Avoid constructing INT64_MIN-1 for an already empty domain.
        auto bound = constant && constant.getValue() <= 0 ? getAffineConstantExpr(-1, context) :
            mlir::pto::detail::checkedAdd(mlir::pto::detail::checkedAdd(upper, -iv),
                                        getAffineConstantExpr(-1, context));
        rows.push_back(bound);
        if (step != 1) {
            rows.push_back(distance ? -(distance % step) : AffineExpr{});
        }
    }
    return rows;
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
    // Entry i1 values have the mathematical representation 0 or 1. Keep this
    // context in every relation, including unconditional sites outside a branch.
    SmallVector<AffineExpr> constrainedRows(rows);
    for (auto [id, parameter] : llvm::enumerate(output.parameters)) {
        if (parameter.getType().isInteger(1)) {
            auto value = getAffineSymbolExpr(id, context);
            constrainedRows.push_back(value);
            constrainedRows.push_back(1 - value);
        }
    }
    // Iterative mixed-radix enumeration: fixed P and D bound its expansion.
    SmallVector<uint64_t> residues(count, 0);
    bool more = true;
    while (more) {
        if (!filter || residues[filter->first] % modulus == filter->second) {
            SmallVector<AffineExpr> constraints;
            SmallVector<AffineExpr> dimensions, symbols;
            for (auto [id, residue] : llvm::enumerate(residues)) {
                auto coordinate = id < target.dimensions ? getAffineDimExpr(id, context) :
                    getAffineSymbolExpr(id - target.dimensions, context);
                auto replacement = mlir::pto::detail::checkedAdd(
                    mlir::pto::detail::checkedMul(coordinate, getAffineConstantExpr(period, context)),
                    getAffineConstantExpr(residue, context));
                (id < target.dimensions ? dimensions : symbols).push_back(replacement);
            }
            for (auto row : constrainedRows) {
                auto expression = mlir::pto::detail::substitute(row, dimensions, symbols);
                if (expression) {
                    expression = simplifyAffineExpr(expression, target.dimensions, count - target.dimensions);
                }
                LinearRow checked;
                if (!expression || collectRow(expression, target.dimensions, count - target.dimensions, checked)) {
                    output.extraction.note(RecognitionIssue::IndexArithmetic, nullptr);
                    return;
                }
                constraints.push_back(expression);
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
