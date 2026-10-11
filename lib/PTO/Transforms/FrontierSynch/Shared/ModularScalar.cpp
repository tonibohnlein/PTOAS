// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve machine casts, signedness and modular arithmetic before relational
// lowering. The shared arena owns all circuits; this evaluator owns one fixed
// leaf-binding context and never modifies source operations.
#include "PTO/Transforms/FrontierSynch/Shared/ModularScalar.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
namespace mlir::pto::frontiersynch {
ModularScalar::ModularScalar(RegionExpressions& arena, Operation* anchor) : e(arena)
{
    if (!anchor) { return; }
    auto bits = DataLayout::closest(anchor).getTypeSizeInBits(IndexType::get(anchor->getContext()));
    if (!bits.isScalable() && bits.getFixedValue() <= 64) { indexBits = bits.getFixedValue(); }
}
unsigned ModularScalar::width(Type type) const
{
    return type.isIndex() ? indexBits : (isa<IntegerType>(type) ? cast<IntegerType>(type).getWidth() : 0);
}
std::optional<std::pair<int64_t, int64_t>> ModularScalar::inputRange(
    Value input, std::pair<int64_t, int64_t> range) const
{
    const auto bits = input ? width(input.getType()) : 0;
    if (!bits || bits > 64 || range.first > range.second) { return std::nullopt; }
    if (bits == 64 || input.getType().isIndex()) { return range; }
    const auto modulus = uint64_t(1) << bits;
    const auto low = -static_cast<int64_t>(modulus / 2), high = static_cast<int64_t>(modulus / 2 - 1);
    if (range.first < low || range.second > high || (range.first < 0 && range.second >= 0)) {
        return std::make_pair(int64_t(0), static_cast<int64_t>(modulus - 1));
    }
    if (range.second < 0) {
        return std::make_pair(static_cast<int64_t>(modulus + range.first),
                              static_cast<int64_t>(modulus + range.second));
    }
    return range;
}
ModularScalar::Id ModularScalar::truncate(Id input, unsigned bits)
{
    return bits == 64 ? input : e.rem(input, e.constant(uint64_t(1) << bits));
}
ModularScalar::Id ModularScalar::signedBits(Id input, unsigned bits)
{
    if (bits == 64) { return input; }
    auto sign = e.constant(uint64_t(1) << (bits - 1));
    return e.select(e.lt(input, sign), input, e.sub(input, e.constant(uint64_t(1) << bits)));
}
ModularScalar::Id ModularScalar::multiply(Id input, uint64_t factor)
{
    auto result = e.constant(0);
    while (factor) {
        if (factor & 1) { result = e.add(result, input); }
        factor >>= 1;
        if (factor) { input = e.add(input, input); }
    }
    return result;
}
std::optional<ModularScalar::Id> ModularScalar::value(Value input, Binding binding)
{
    return resolve(input, binding, 0);
}
std::optional<ModularScalar::Id> ModularScalar::signedValue(Value input, Binding binding)
{
    const auto bits = input ? width(input.getType()) : 0;
    auto result = value(input, binding);
    return result && bits > 1 ? std::optional<Id>(signedBits(*result, bits)) : result;
}
std::optional<ModularScalar::Id> ModularScalar::resolve(Value input, Binding binding, unsigned depth)
{
    if (!input || !width(input.getType()) || width(input.getType()) > 64 || depth >= 64) { return std::nullopt; }
    if (auto found = memo.find(input); found != memo.end()) { return found->second; }
    memo.try_emplace(input, std::nullopt);
    APInt constant;
    std::optional<Id> result;
    if (matchPattern(input, m_ConstantInt(&constant))) {
        const auto bits = width(input.getType());
        result = bits == 1 ? e.boolean(!constant.isZero()) : e.constant(constant.zextOrTrunc(bits).getZExtValue());
    } else if (auto leaf = binding(input)) {
        const auto bits = width(input.getType());
        if (*leaf < e.size() && e.isBoolean(*leaf) == (bits == 1)) {
            result = bits == 1 ? *leaf : truncate(*leaf, bits);
        }
    } else {
        result = operation(input, binding, depth + 1);
    }
    memo[input] = result;
    return result;
}
std::optional<ModularScalar::Id> ModularScalar::operation(Value input, Binding binding, unsigned depth)
{
    auto* op = input.getDefiningOp();
    if (!op || op->getNumResults() != 1 || op->getNumRegions()) { return std::nullopt; }
    auto operand = [&](unsigned i) { return resolve(op->getOperand(i), binding, depth); };
    auto cast = isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op);
    const auto bits = width(input.getType());
    if (cast) {
        auto source = operand(0);
        const auto from = width(op->getOperand(0).getType());
        if (!source || from <= 1 || bits <= 1) { return std::nullopt; }
        const bool sign = isa<arith::IndexCastOp, arith::ExtSIOp>(op);
        return truncate(sign && bits > from ? signedBits(*source, from) : *source, bits);
    }
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
        auto condition = operand(0), yes = operand(1), no = operand(2);
        if (!condition || !yes || !no) { return std::nullopt; }
        return e.select(*condition, *yes, *no);
    }
    if (op->getNumOperands() != 2) { return std::nullopt; }
    const bool supported = isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivUIOp,
        arith::RemUIOp, arith::DivSIOp, arith::RemSIOp, arith::CmpIOp, arith::AndIOp,
        arith::OrIOp, arith::XOrIOp>(op);
    if (!supported) { return std::nullopt; }
    auto a = operand(0), b = operand(1);
    if (!a || !b) { return std::nullopt; }
    if (auto compare = dyn_cast<arith::CmpIOp>(op)) {
        const auto sourceBits = width(compare.getLhs().getType());
        if (sourceBits <= 1) { return std::nullopt; }
        auto sa = signedBits(*a, sourceBits), sb = signedBits(*b, sourceBits);
        switch (compare.getPredicate()) {
        case arith::CmpIPredicate::eq: return e.eq(*a, *b);
        case arith::CmpIPredicate::ne: return e.lnot(e.eq(*a, *b));
        case arith::CmpIPredicate::slt: return e.slt(sa, sb);
        case arith::CmpIPredicate::sle: return e.sle(sa, sb);
        case arith::CmpIPredicate::sgt: return e.slt(sb, sa);
        case arith::CmpIPredicate::sge: return e.sle(sb, sa);
        case arith::CmpIPredicate::ult: return e.lt(*a, *b);
        case arith::CmpIPredicate::ule: return e.le(*a, *b);
        case arith::CmpIPredicate::ugt: return e.lt(*b, *a);
        case arith::CmpIPredicate::uge: return e.le(*b, *a);
        }
        return std::nullopt;
    }
    if (bits == 1) {
        if (isa<arith::AndIOp>(op)) { return e.land(*a, *b); }
        if (isa<arith::OrIOp>(op)) { return e.lor(*a, *b); }
        if (isa<arith::XOrIOp>(op)) { return e.lor(e.land(*a, e.lnot(*b)), e.land(e.lnot(*a), *b)); }
        return std::nullopt;
    }
    if (isa<arith::AddIOp>(op)) { return truncate(e.add(*a, *b), bits); }
    if (isa<arith::SubIOp>(op)) { return truncate(e.sub(*a, *b), bits); }
    if (isa<arith::MulIOp>(op)) {
        auto factor = e.constantValue(*b);
        auto other = a;
        if (!factor) { factor = e.constantValue(*a); other = b; }
        return factor ? std::optional<Id>(truncate(multiply(*other, *factor), bits)) : std::nullopt;
    }
    if (!isa<arith::DivUIOp, arith::RemUIOp, arith::DivSIOp, arith::RemSIOp>(op)) { return std::nullopt; }
    auto divisor = e.constantValue(*b);
    if (!divisor || !*divisor) { return std::nullopt; }
    if (isa<arith::DivUIOp>(op)) { return e.div(*a, *b); }
    if (isa<arith::RemUIOp>(op)) { return e.rem(*a, *b); }
    // Positive signed constant divisors exclude MIN/-1 poison. Compute the
    // magnitude in unsigned bits, which also handles signed MIN exactly.
    if (*divisor >= (uint64_t(1) << (bits - 1))) { return std::nullopt; }
    auto negative = e.slt(signedBits(*a, bits), e.constant(0));
    if (isa<arith::DivSIOp>(op)) {
        // Truncation toward zero equals floor after the sign-dependent bias.
        // Form the numerator mathematically: adding the bias in source bits
        // would wrap at the positive signed endpoint.
        IntegerAffine numerator{{BoundInteger(1), BoundInteger(*divisor - 1)}, BoundInteger(0)};
        auto result = e.integerFloor(numerator, BoundInteger(*divisor),
            {signedBits(*a, bits), negative}, 1, {0, 0});
        return truncate(result, bits);
    }
    auto magnitude = e.select(negative, truncate(e.sub(e.constant(0), *a), bits), *a);
    auto quotient = isa<arith::DivSIOp>(op) ? e.div(magnitude, *b) : e.rem(magnitude, *b);
    return truncate(e.select(negative, e.sub(e.constant(0), quotient), quotient), bits);
}
} // namespace mlir::pto::frontiersynch
