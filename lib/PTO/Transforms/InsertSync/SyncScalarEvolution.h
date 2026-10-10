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
#include "SyncScalarConstraints.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <algorithm>
#include <iterator>
#include <memory>
namespace mlir::pto::detail {
class ScalarEvolution {
    struct Range { int64_t lower, upper; };
    struct Result {
        AffineExpr expression;
        std::optional<Range> range;
    };
    using Symbol = llvm::function_ref<AffineExpr(Value)>;
    MLIRContext* context;
    unsigned indexBits = 32;
    DenseMap<Value, Result> cache;
    std::shared_ptr<const ScalarConstraints> constraints;
    struct RangeQueries {
        DenseMap<Value, std::optional<Range>> values;
        unsigned depth = 0;
    };
    std::shared_ptr<RangeQueries> rangeQueries;

    AffineExpr number(int64_t n) { return getAffineConstantExpr(n, context); }
    static std::optional<int64_t> constant(Value v)
    {
        APInt n;
        if (v && matchPattern(v, m_ConstantInt(&n)) && n.isSignedIntN(64)) {
            return n.getSExtValue();
        }
        return std::nullopt;
    }
    // Use the supplied IR data layout. Standalone clients without an anchor
    // retain the portable 32-bit proof; wider-than-64 values stay opaque.
    unsigned width(Type type) const
    {
        return type.isIndex() ? indexBits : (isa<IntegerType>(type) ? cast<IntegerType>(type).getWidth() : 0);
    }
    bool fits(Range r, Type type) const
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
    Result folded(Value v, Symbol symbol, unsigned depth);
    Result operation(Value v, Symbol symbol, unsigned depth);
    Result binary(Operation* op, Result a, Result b);
    static std::optional<Range> arithmeticRange(Operation* op, Range a, Range b);
    Result resolve(Value v, Symbol symbol, unsigned depth)
    {
        if (auto n = constant(v); n && fits(Range{*n, *n}, v.getType())) {
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
        // Range inference is shared operation semantics, not an address-op
        // whitelist. Keep unsupported expressions symbolic while using their
        // declared result bounds to prove subsequent arithmetic cannot wrap.
        if (!result.range && v.getDefiningOp() && v.getType().isIntOrIndex() &&
            width(v.getType()) && width(v.getType()) <= 64) {
            auto* operation = v.getDefiningOp();
            const bool hasIndex =
                llvm::any_of(operation->getOperandTypes(), [](Type type) { return type.isIndex(); }) ||
                llvm::any_of(operation->getResultTypes(), [](Type type) { return type.isIndex(); });
            // LLVM's generic index range interface uses its internal storage
            // width; it cannot model narrowing casts for a different layout.
            if (auto infer = dyn_cast<InferIntRangeInterface>(operation);
                infer && (!hasIndex || indexBits == IndexType::kInternalStorageBitWidth)) {
                SmallVector<ConstantIntRanges> operands;
                bool supported = true;
                for (Value operand : v.getDefiningOp()->getOperands()) {
                    unsigned bits = operand.getType().isIntOrIndex() ? width(operand.getType()) : 0;
                    if (!bits || bits > 64) {
                        supported = false;
                        break;
                    }
                    auto known = resolve(operand, symbol, depth + 1).range;
                    operands.push_back(known ? ConstantIntRanges::fromSigned(
                        APInt(bits, known->lower, true), APInt(bits, known->upper, true)) :
                        ConstantIntRanges::maxRange(bits));
                }
                if (supported) {
                    infer.inferResultRanges(operands, [&](Value value, const ConstantIntRanges& range) {
                        if (value == v && range.smin().isSignedIntN(64) && range.smax().isSignedIntN(64)) {
                            Range inferred{range.smin().getSExtValue(), range.smax().getSExtValue()};
                            if (fits(inferred, v.getType())) { result.range = inferred; }
                        }
                    });
                }
            }
        }
        // A scalar load or opaque integer operation still has its declared
        // signed bit range. This supplies no expression or alias fact, but it
        // proves widening index casts and later bounded arithmetic exact.
        if (!result.range && v.getType().isIntOrIndex() && width(v.getType()) && width(v.getType()) <= 64) {
            const unsigned bits = width(v.getType());
            result.range = Range{APInt::getSignedMinValue(bits).getSExtValue(),
                                 APInt::getSignedMaxValue(bits).getSExtValue()};
        }
        // Refine the result bit-pattern range only after operation admission.
        // A guard on x+1 does not prove that computing x+1 was nonwrapping.
        if (constraints && result.range) {
            auto fact = constraints->find(v);
            if (fact != constraints->end()) {
                Range refined{std::max(result.range->lower, fact->second.first),
                              std::min(result.range->upper, fact->second.second)};
                if (refined.lower <= refined.upper) { result.range = refined; }
            }
        }
        cache[v] = result;
        return result;
    }
public:
    struct ModularRecurrence {
        int64_t seed, stride, modulus, lower, step;
    };
    // Shared machine-semantics proof; consumers may use its closed form or
    // evaluate a representative with wide modular arithmetic.
    std::optional<ModularRecurrence> modularRecurrence(BlockArgument argument);
    explicit ScalarEvolution(MLIRContext* context, Operation* anchor = nullptr) : context(context)
    {
        if (anchor) {
            auto bits = DataLayout::closest(anchor).getTypeSizeInBits(IndexType::get(context));
            indexBits = !bits.isScalable() && bits.getFixedValue() <= 64 ? bits.getFixedValue() : 0;
            constraints = std::make_shared<const ScalarConstraints>(enclosingScalarConstraints(anchor, indexBits));
        }
    }
    // Range-only query uses independent state: it must not publish placeholder
    // symbols into a caller's address-expression cache.
    std::optional<std::pair<int64_t, int64_t>> signedRange(Value value)
    {
        if (!rangeQueries) { rangeQueries = std::make_shared<RangeQueries>(); }
        auto& memo = rangeQueries->values;
        auto found = memo.find(value);
        if (found != memo.end()) {
            return found->second ? std::optional<std::pair<int64_t, int64_t>>(
                std::make_pair(found->second->lower, found->second->upper)) : std::nullopt;
        }
        // Mark before traversal and share the recursion bound between fresh
        // expression contexts. Repeated lower/upper dependencies are visited once.
        memo.try_emplace(value, std::nullopt);
        if (rangeQueries->depth >= 64) { return std::nullopt; }
        ++rangeQueries->depth;
        ScalarEvolution query(context);
        query.indexBits = indexBits;
        query.rangeQueries = rangeQueries;
        query.constraints = constraints;
        auto range = query.resolve(value, [&](Value) { return getAffineSymbolExpr(0, context); }, 0).range;
        --rangeQueries->depth;
        const unsigned bits = width(value.getType());
        if (!range && bits && bits <= 64) {
            range = Range{APInt::getSignedMinValue(bits).getSExtValue(),
                          APInt::getSignedMaxValue(bits).getSExtValue()};
        }
        memo[value] = range;
        return range ? std::optional<std::pair<int64_t, int64_t>>(
            std::make_pair(range->lower, range->upper)) : std::nullopt;
    }
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
    auto lower = signedRange(loop.getLowerBound()), upper = signedRange(loop.getUpperBound());
    auto step = constant(loop.getStep());
    if (!lower || !upper || !step || *step <= 0 || upper->second <= lower->first) { return {}; }
    // On an executed visit lower <= iv < upper. Runtime origins retain their
    // independently proved range; no relation between unrelated values is assumed.
    const unsigned bits = width(arg.getType());
    if (!bits || bits > 64) { return {}; }
    const int64_t last = upper->second - 1;
    return {symbol(arg), Range{lower->first, last}};
}

inline std::optional<ScalarEvolution::ModularRecurrence> ScalarEvolution::modularRecurrence(BlockArgument arg)
{
    auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
    if (!loop || !arg.getArgNumber()) { return std::nullopt; }
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
    return ModularRecurrence{*seed, *stride, *modulus, *lower, *step};
}

inline ScalarEvolution::Result ScalarEvolution::recurrence(BlockArgument arg, scf::ForOp loop, Symbol symbol)
{
    auto proof = modularRecurrence(arg);
    if (!proof) { return {}; }
    auto ordinal = checkedAdd(symbol(loop.getInductionVar()), number(-proof->lower));
    auto shifted = ordinal ? checkedMul(ordinal.floorDiv(proof->step), number(proof->stride)) : AffineExpr{};
    auto expression = shifted ? checkedAdd(number(proof->seed), shifted) : AffineExpr{};
    return expression ? Result{modulo(expression, proof->modulus), Range{0, proof->modulus - 1}} : Result{};
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

// Use dialect folding for identities and machine-width constant arithmetic.
// Folders may mutate their operation: fold a detached scalar copy and accept
// only a constant or an existing SSA value. Never rewrite the analyzed program.
inline ScalarEvolution::Result ScalarEvolution::folded(Value v, Symbol symbol, unsigned depth)
{
    auto* op = v.getDefiningOp();
    if (op->getNumResults() != 1 || op->getNumRegions() || op->getNumSuccessors() ||
        !isMemoryEffectFree(op)) {
        return {};
    }
    // Arith scalar folders are context independent; an arbitrary dialect's
    // folder may require ancestors that a detached copy deliberately lacks.
    if (!isa<arith::ArithDialect>(op->getDialect())) {
        return {};
    }
    // Index attributes use the internal storage width, not the target width.
    // Even representable inputs/results are insufficient (unsigned -1 % 7
    // differs at 32 and 64 bits). Keep checked symbolic reasoning below for
    // narrow/unknown index layouts instead of accepting host-width folds.
    const bool usesIndex = v.getType().isIndex() ||
        llvm::any_of(op->getOperandTypes(), [](Type type) { return type.isIndex(); });
    if (usesIndex && indexBits != IndexType::kInternalStorageBitWidth) {
        return {};
    }
    OwningOpRef<Operation*> copy(op->cloneWithoutRegions());
    SmallVector<OpFoldResult> results;
    if (failed(copy->fold(results)) || results.size() != 1) {
        return {};
    }
    if (auto value = dyn_cast<Value>(results.front())) {
        return value != v && value.getDefiningOp() != copy.get() && value.getType() == v.getType() ?
            resolve(value, symbol, depth) : Result{};
    }
    auto integer = dyn_cast<IntegerAttr>(dyn_cast<Attribute>(results.front()));
    if (!integer || integer.getType() != v.getType() || !integer.getValue().isSignedIntN(64)) {
        return {};
    }
    int64_t n = integer.getValue().getSExtValue();
    return fits(Range{n, n}, v.getType()) ? Result{number(n), Range{n, n}} : Result{};
}

inline ScalarEvolution::Result ScalarEvolution::operation(Value v, Symbol symbol, unsigned depth)
{
    auto* op = v.getDefiningOp();
    if (!op || width(v.getType()) == 0 || width(v.getType()) > 64) {
        return {};
    }
    auto fold = folded(v, symbol, depth);
    if (fold.expression) {
        return fold;
    }
    if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
        auto input = resolve(op->getOperand(0), symbol, depth);
        if (input.range && fits(*input.range, v.getType()) &&
            (!isa<arith::ExtUIOp>(op) || input.range->lower >= 0)) {
            return input;
        }
        return {};
    }
    if (op->getNumOperands() != 2 ||
        !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::MaxSIOp, arith::MinSIOp,
             arith::AndIOp, arith::RemUIOp, arith::RemSIOp, arith::DivUIOp, arith::DivSIOp>(op)) {
        // Unsupported scalar expressions remain one original SSA symbol. Do
        // not register irrelevant operands (e.g. a load's pointer) in the
        // caller's affine map merely to discover that normalization stops here.
        return {};
    }
    auto a = resolve(op->getOperand(0), symbol, depth), b = resolve(op->getOperand(1), symbol, depth);
    // A consumer may deliberately reject unknown SSA symbols. A range alone
    // does not supply an expression for arithmetic or division on that value.
    if (!a.expression || !b.expression) {
        return {};
    }
    // A signed min/max can select one affine operand when its entire range
    // lies on the appropriate side of the other. Overlapping ranges remain
    // opaque; in particular an overflowing multiply cannot justify a clamp.
    if ((isa<arith::MaxSIOp>(op) || isa<arith::MinSIOp>(op)) && a.range && b.range) {
        const bool maximum = isa<arith::MaxSIOp>(op);
        if (a.range->lower >= b.range->upper) {
            return maximum ? a : b;
        }
        if (b.range->lower >= a.range->upper) {
            return maximum ? b : a;
        }
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
        if (!divisor || *divisor <= 0 || !fits(Range{0, *divisor}, v.getType())) {
            return {};
        }
        // Unsigned remainder by 2^j selects low bits. Signed and unsigned
        // interpretations differ by 2^w, divisible by 2^j for j < w.
        // Non-power-of-two divisors and signed remainders still need the
        // nonnegative proof; signed remainder of a negative input is not mod.
        if (isa<arith::RemUIOp>(op) && llvm::isPowerOf2_64(static_cast<uint64_t>(*divisor))) {
            return {modulo(a.expression, *divisor), Range{0, *divisor - 1}};
        }
        if (!a.range || a.range->lower < 0) {
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
