// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Normalize scalar address arithmetic without changing the IR. A range proves
// machine operations do not wrap; a matched modular recurrence supplies the
// value of its block argument at the start of each loop iteration. Unproved
// values remain original SSA symbols, including loop results and branch merges.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCSCALAREVOLUTION_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCSCALAREVOLUTION_H
#include "SyncRegionArithmetic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <algorithm>
#include <iterator>
namespace mlir::pto::detail {
class ScalarEvolution {
    struct Range { int64_t lower, upper; };
    struct Result {
        AffineExpr expression;
        std::optional<Range> range;
    };
    using Symbol = llvm::function_ref<AffineExpr(Value)>;
    MLIRContext* context;
    DenseMap<Value, Result> cache;

    AffineExpr number(int64_t n) { return getAffineConstantExpr(n, context); }
    static std::optional<int64_t> constant(Value v)
    {
        APInt n;
        if (v && matchPattern(v, m_ConstantInt(&n)) && n.isSignedIntN(64)) {
            return n.getSExtValue();
        }
        return std::nullopt;
    }
    // Index width is target-dependent. Prove unflagged arithmetic for both
    // supported widths by requiring a signed 32-bit result. Integer types use
    // their declared width; wider-than-64 values stay opaque.
    static unsigned width(Type type)
    {
        return type.isIndex() ? 32 : (isa<IntegerType>(type) ? cast<IntegerType>(type).getWidth() : 0);
    }
    static bool fits(Range r, Type type)
    {
        unsigned bits = width(type);
        return bits && bits <= 64 && APInt(64, r.lower, true).isSignedIntN(bits) &&
               APInt(64, r.upper, true).isSignedIntN(bits);
    }
    // Congruence is preserved by addition and multiplication. Remove nested
    // compatible remainders only inside an outer remainder, never through a
    // division. This turns ((i + 1) mod 2 + 1) mod 2 into i mod 2.
    static AffineExpr stripModulo(AffineExpr expression, int64_t modulus, unsigned depth = 0)
    {
        auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
        if (!binary || depth >= 64) {
            return expression;
        }
        if (binary.getKind() == AffineExprKind::Mod) {
            auto inner = dyn_cast<AffineConstantExpr>(binary.getRHS());
            return inner && inner.getValue() > 0 && inner.getValue() % modulus == 0 ?
                stripModulo(binary.getLHS(), modulus, depth + 1) : expression;
        }
        if (binary.getKind() != AffineExprKind::Add && binary.getKind() != AffineExprKind::Mul) {
            return expression;
        }
        auto left = stripModulo(binary.getLHS(), modulus, depth + 1);
        auto right = stripModulo(binary.getRHS(), modulus, depth + 1);
        return binary.getKind() == AffineExprKind::Add ? checkedAdd(left, right) : checkedMul(left, right);
    }
    static AffineExpr modulo(AffineExpr expression, int64_t modulus)
    {
        auto stripped = stripModulo(expression, modulus);
        return stripped ? stripped % modulus : AffineExpr{};
    }
    Result argument(BlockArgument arg, Symbol symbol);
    Result recurrence(BlockArgument arg, scf::ForOp loop, Symbol symbol);
    Result operation(Value v, Symbol symbol, unsigned depth);
    Result binary(Operation* op, Result a, Result b);
    static std::optional<Range> arithmeticRange(Operation* op, Range a, Range b);
    Result resolve(Value v, Symbol symbol, unsigned depth)
    {
        if (auto n = constant(v)) {
            return {number(*n), Range{*n, *n}};
        }
        auto found = cache.find(v);
        if (found != cache.end()) {
            return found->second;
        }
        if (depth >= 64) {
            return {symbol(v), std::nullopt};
        }
        Result result = isa<BlockArgument>(v) ? argument(cast<BlockArgument>(v), symbol) :
                                             operation(v, symbol, depth + 1);
        if (!result.expression) {
            result.expression = symbol(v);
        }
        cache[v] = result;
        return result;
    }
public:
    explicit ScalarEvolution(MLIRContext* context) : context(context) {}
    AffineExpr value(Value v, Symbol symbol)
    {
        return v ? resolve(v, symbol, 0).expression : AffineExpr{};
    }
};

inline ScalarEvolution::Result ScalarEvolution::argument(BlockArgument arg, Symbol symbol)
{
    auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
    if (!loop) {
        return {};
    }
    if (arg != loop.getInductionVar()) {
        return recurrence(arg, loop, symbol);
    }
    auto lower = constant(loop.getLowerBound()), upper = constant(loop.getUpperBound());
    auto step = constant(loop.getStep());
    if (!lower || !step || *step <= 0 || (upper && *upper <= *lower)) {
        return {};
    }
    // scf.for uses signed bounds. For a symbolic upper bound, this range
    // supplies nonnegativity but cannot justify unflagged index arithmetic.
    return {symbol(arg), Range{*lower, upper ? *upper - 1 : INT64_MAX}};
}

inline ScalarEvolution::Result ScalarEvolution::recurrence(BlockArgument arg, scf::ForOp loop, Symbol symbol)
{
    const unsigned id = arg.getArgNumber() - 1;
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    auto lower = constant(loop.getLowerBound()), step = constant(loop.getStep());
    if (!yield || id >= yield.getNumOperands() || id >= loop.getInitArgs().size() ||
        !lower || *lower == INT64_MIN || !step || *step <= 0) {
        return {};
    }
    auto seed = constant(loop.getInitArgs()[id]);
    auto* remainder = yield.getOperand(id).getDefiningOp();
    if (!seed || !remainder || !isa<arith::RemUIOp, arith::RemSIOp>(remainder)) {
        return {};
    }
    auto modulus = constant(remainder->getOperand(1));
    auto update = remainder->getOperand(0).getDefiningOp<arith::AddIOp>();
    if (!modulus || *modulus <= 0 || *seed < 0 || *seed >= *modulus || !update) {
        return {};
    }
    Value increment = update.getLhs() == arg ? update.getRhs() :
                      (update.getRhs() == arg ? update.getLhs() : Value{});
    auto stride = constant(increment);
    int64_t maximum;
    if (!stride || *stride < 0 || llvm::AddOverflow(*modulus - 1, *stride, maximum) ||
        !fits(Range{0, maximum}, arg.getType())) {
        return {};
    }
    auto ordinal = checkedAdd(symbol(loop.getInductionVar()), number(-*lower));
    auto shifted = ordinal ? checkedMul(ordinal.floorDiv(*step), number(*stride)) : AffineExpr{};
    auto expression = shifted ? checkedAdd(number(*seed), shifted) : AffineExpr{};
    return expression ? Result{modulo(expression, *modulus), Range{0, *modulus - 1}} : Result{};
}

inline std::optional<ScalarEvolution::Range> ScalarEvolution::arithmeticRange(Operation* op, Range a, Range b)
{
    int64_t lower, upper;
    if (isa<arith::AddIOp>(op)) {
        if (!llvm::AddOverflow(a.lower, b.lower, lower) && !llvm::AddOverflow(a.upper, b.upper, upper)) {
            return Range{lower, upper};
        }
    } else if (isa<arith::SubIOp>(op)) {
        if (!llvm::SubOverflow(a.lower, b.upper, lower) && !llvm::SubOverflow(a.upper, b.lower, upper)) {
            return Range{lower, upper};
        }
    } else if (isa<arith::MulIOp>(op)) {
        int64_t products[4];
        if (!llvm::MulOverflow(a.lower, b.lower, products[0]) &&
            !llvm::MulOverflow(a.lower, b.upper, products[1]) &&
            !llvm::MulOverflow(a.upper, b.lower, products[2]) &&
            !llvm::MulOverflow(a.upper, b.upper, products[3])) {
            auto bounds = std::minmax_element(std::begin(products), std::end(products));
            return Range{*bounds.first, *bounds.second};
        }
    }
    return std::nullopt;
}

inline ScalarEvolution::Result ScalarEvolution::binary(Operation* op, Result a, Result b)
{
    auto range = a.range && b.range ? arithmeticRange(op, *a.range, *b.range) : std::nullopt;
    auto flags = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(op);
    const bool proven = range && fits(*range, op->getResult(0).getType());
    if (!proven && (!flags || !flags.hasNoSignedWrap())) {
        return {};
    }
    AffineExpr expression;
    if (isa<arith::AddIOp>(op)) {
        expression = checkedAdd(a.expression, b.expression);
    } else if (isa<arith::SubIOp>(op)) {
        expression = checkedAdd(a.expression, checkedMul(b.expression, number(-1)));
    } else if (isa<arith::MulIOp>(op) &&
               (isa<AffineConstantExpr>(a.expression) || isa<AffineConstantExpr>(b.expression))) {
        expression = checkedMul(a.expression, b.expression);
    }
    return {expression, proven ? range : std::nullopt};
}

inline ScalarEvolution::Result ScalarEvolution::operation(Value v, Symbol symbol, unsigned depth)
{
    auto* op = v.getDefiningOp();
    if (!op || width(v.getType()) == 0 || width(v.getType()) > 64) {
        return {};
    }
    if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
        auto input = resolve(op->getOperand(0), symbol, depth);
        if (input.range && fits(*input.range, v.getType()) &&
            (!isa<arith::ExtUIOp>(op) || input.range->lower >= 0)) {
            return input;
        }
        return {};
    }
    if (op->getNumOperands() != 2) {
        return {};
    }
    auto a = resolve(op->getOperand(0), symbol, depth), b = resolve(op->getOperand(1), symbol, depth);
    // A consumer may deliberately reject unknown SSA symbols. A range alone
    // does not supply an expression for arithmetic or division on that value.
    if (!a.expression || !b.expression) {
        return {};
    }
    if (isa<arith::AndIOp>(op)) {
        auto mask = constant(op->getOperand(1));
        auto input = a.expression;
        if (!mask) {
            mask = constant(op->getOperand(0));
            input = b.expression;
        }
        // Low-bit masks compute Euclidean modulo even for negative operands.
        // Keep the modulus representable by the signed affine-expression API.
        if (mask && *mask >= 0 && *mask < INT64_MAX &&
            llvm::isPowerOf2_64(static_cast<uint64_t>(*mask) + 1)) {
            return {modulo(input, *mask + 1), Range{0, *mask}};
        }
        return {};
    }
    if (isa<arith::RemUIOp, arith::RemSIOp, arith::DivUIOp, arith::DivSIOp>(op)) {
        auto divisor = constant(op->getOperand(1));
        if (!divisor || *divisor <= 0 || !a.range || a.range->lower < 0) {
            return {};
        }
        if (isa<arith::RemUIOp, arith::RemSIOp>(op)) {
            return {modulo(a.expression, *divisor), Range{0, *divisor - 1}};
        }
        return {a.expression.floorDiv(*divisor), Range{a.range->lower / *divisor, a.range->upper / *divisor}};
    }
    return binary(op, a, b);
}
} // namespace mlir::pto::detail
#endif
