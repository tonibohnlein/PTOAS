// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Normalize rotating selectors with the shared scalar semantics. Unsigned
// power-of-two remainders also permit wrapping arithmetic at the same width.
#include "RotationPattern.h"
#include "RecognitionInternal.h"
#include "ArithmeticRows.h"
#include "../InsertSync/SyncScalarEvolution.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::pto::frontiersynch::detail {
namespace {
std::optional<int64_t> integer(Value value)
{
    APInt number;
    const bool known = matchPattern(value, m_ConstantInt(&number)) && number.isSignedIntN(64);
    return known ? std::optional<int64_t>(number.getSExtValue()) : std::nullopt;
}
uint64_t residue(int64_t n, uint64_t count)
{
    auto rem = APInt(128, static_cast<uint64_t>(n), true).srem(APInt(128, count));
    return (rem.isNegative() ? rem + APInt(128, count) : rem).getZExtValue();
}
std::optional<SlotPattern> fromExpression(AffineExpr expression, ArrayRef<Value> symbols,
                                        scf::ForOp loop, uint64_t count,
                                        const PhaseIndex& index, bool allowParameters)
{
    LinearRow row;
    if (collectRow(expression, 0, symbols.size(), row)) {
        return std::nullopt;
    }
    SlotPattern result;
    result.offset = residue(row.constant, count);
    auto offset = getAffineConstantExpr(result.offset, loop.getContext());
    SmallVector<Operation*> recipe;
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        auto coefficient = residue(row.coefficients[i], count);
        if (!coefficient) {
            continue;
        }
        if (symbols[i] == loop.getInductionVar()) {
            result.stride = coefficient;
            continue;
        }
        if (!allowParameters || !entryExpression(symbols[i], loop, index, recipe)) {
            return std::nullopt;
        }
        offset = offset + getAffineSymbolExpr(result.parameters.size(), loop.getContext()) *
                          static_cast<int64_t>(coefficient);
        result.parameters.push_back(symbols[i]);
    }
    if (!result.parameters.empty()) {
        result.parameterOffset = offset % static_cast<int64_t>(count);
    }
    return result;
}
struct Form {
    llvm::MapVector<Value, uint64_t> coefficients;
    uint64_t constant = 0;
};
std::optional<Form> wrappingForm(Value root, scf::ForOp loop, uint64_t count,
                                 const PhaseIndex& index, bool allowParameters)
{
    DenseMap<Value, Form> forms;
    SmallVector<std::pair<Value, bool>> stack{{root, false}};
    SmallVector<Operation*> recipe;
    auto add = [count](uint64_t a, uint64_t b) { return (a + b) % count; };
    auto multiply = [count](uint64_t a, uint64_t b) {
        return (APInt(128, a) * APInt(128, b)).urem(APInt(128, count)).getZExtValue();
    };
    while (!stack.empty()) {
        auto [value, finish] = stack.pop_back_val();
        if (forms.count(value)) {
            continue;
        }
        const bool sameWidth = value.getType() == root.getType();
        if (!sameWidth) {
            return std::nullopt;
        }
        if (auto n = integer(value)) {
            forms[value].constant = residue(*n, count);
            continue;
        }
        if (value == loop.getInductionVar()) {
            forms[value].coefficients[value] = 1 % count;
            continue;
        }
        Operation* op = value.getDefiningOp();
        const bool arithmetic = op && isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op);
        if (!arithmetic) {
            if (!allowParameters || !entryExpression(value, loop, index, recipe)) {
                return std::nullopt;
            }
            forms[value].coefficients[value] = 1 % count;
            continue;
        }
        auto left = op->getOperand(0), right = op->getOperand(1);
        if (!finish) {
            stack.push_back({value, true});
            stack.push_back({right, false});
            stack.push_back({left, false});
            continue;
        }
        Form a = forms[left], b = forms[right];
        if (isa<arith::MulIOp>(op)) {
            const bool nonlinear = !a.coefficients.empty() && !b.coefficients.empty();
            if (nonlinear) {
                return std::nullopt;
            }
            if (!b.coefficients.empty()) {
                std::swap(a, b);
            }
            for (auto& entry : a.coefficients) {
                entry.second = multiply(entry.second, b.constant);
            }
            a.constant = multiply(a.constant, b.constant);
        } else {
            const bool subtract = isa<arith::SubIOp>(op);
            auto signedResidue = [count, subtract](uint64_t n) {
                return subtract && n ? count - n : n;
            };
            a.constant = add(a.constant, signedResidue(b.constant));
            for (const auto& entry : b.coefficients) {
                a.coefficients[entry.first] = add(a.coefficients[entry.first], signedResidue(entry.second));
            }
        }
        forms[value] = std::move(a);
    }
    return forms[root];
}
} // namespace

std::optional<SlotPattern> matchRotatingSlot(Value slot, scf::ForOp loop, uint64_t count,
                                            const PhaseIndex& index, bool allowParameters)
{
    if (!count || count > static_cast<uint64_t>(INT64_MAX)) {
        return std::nullopt;
    }
    SmallVector<Value> symbols;
    mlir::pto::detail::ScalarEvolution evolution(loop.getContext());
    auto symbol = [&](Value v) {
        auto found = llvm::find(symbols, v);
        unsigned position = std::distance(symbols.begin(), found);
        if (found == symbols.end()) {
            symbols.push_back(v);
        }
        return getAffineSymbolExpr(position, loop.getContext());
    };
    auto expression = evolution.value(slot, symbol);
    auto mod = dyn_cast<AffineBinaryOpExpr>(expression);
    const bool modulo = mod && mod.getKind() == AffineExprKind::Mod;
    if (modulo) {
        auto divisor = dyn_cast<AffineConstantExpr>(mod.getRHS());
        const bool matches = divisor && divisor.getValue() == static_cast<int64_t>(count);
        if (matches) {
            if (auto result = fromExpression(mod.getLHS(), symbols, loop, count, index, allowParameters)) {
                return result;
            }
        }
    }
    auto rem = slot.getDefiningOp<arith::RemUIOp>();
    unsigned width = slot.getType().isIndex() ? 32 :
                     (isa<IntegerType>(slot.getType()) ? cast<IntegerType>(slot.getType()).getWidth() : 0);
    const bool compatible = llvm::isPowerOf2_64(count) && width && llvm::Log2_64(count) < width;
    const bool matches = rem && compatible && integer(rem.getRhs()) == static_cast<int64_t>(count);
    if (matches) {
        if (auto form = wrappingForm(rem.getLhs(), loop, count, index, allowParameters)) {
            symbols.clear();
            auto affine = getAffineConstantExpr(form->constant, loop.getContext());
            for (const auto& entry : form->coefficients) {
                affine = affine + symbol(entry.first) * static_cast<int64_t>(entry.second);
            }
            if (auto result = fromExpression(affine, symbols, loop, count, index, allowParameters)) {
                return result;
            }
        }
    }
    return matchSlot(slot, loop.getInductionVar(), count);
}
} // namespace mlir::pto::frontiersynch::detail
