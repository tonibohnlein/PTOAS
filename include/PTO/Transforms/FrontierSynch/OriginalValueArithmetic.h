// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALVALUEARITHMETIC_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALVALUEARITHMETIC_H

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::pto::frontiersynch::value_arithmetic {

// A value is represented by its low width bits. Signedness belongs to the
// operation interpreting those bits, not to an implicitly signed host integer.
// These helpers neither execute original IR nor use wrapping arithmetic as a
// proof of mathematical equality. They are shared with the endpoint qualifier.
struct Integer {
    unsigned width = 0;
    bool isUnsigned = false;

    bool valid() const { return width > 0 && width <= 64; }
    uint64_t mask() const
    {
        if (!valid()) {
            return 0;
        }
        return width == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t(1) << width) - 1;
    }
    uint64_t sign() const { return valid() ? uint64_t(1) << (width - 1) : 0; }
    uint64_t minimum() const { return isUnsigned ? 0 : sign(); }
    uint64_t maximum() const { return isUnsigned ? mask() : sign() - 1; }
    bool fits(uint64_t bits) const { return valid() && (bits & ~mask()) == 0; }
    // Biasing the sign bit maps signed order to unsigned order, including i64.
    uint64_t rank(uint64_t bits) const { return bits ^ (isUnsigned ? 0 : sign()); }
    bool less(uint64_t a, uint64_t b) const { return rank(a) < rank(b); }
};

enum class Binary { Add, Subtract, Multiply, Minimum, Maximum, Remainder };

inline std::optional<uint64_t> checked(Binary op, Integer type, uint64_t a, uint64_t b)
{
    if (!type.fits(a) || !type.fits(b)) {
        return {};
    }
    if (op == Binary::Minimum || op == Binary::Maximum) {
        const bool takeA = op == Binary::Minimum ? type.less(a, b) : type.less(b, a);
        return takeA ? a : b;
    }
    const auto mask = type.mask();
    if (op == Binary::Add) {
        const auto result = (a + b) & mask; // unsigned host overflow is defined
        const bool overflow =
            type.isUnsigned ? a > mask - b : ((a ^ b) & type.sign()) == 0 && ((a ^ result) & type.sign()) != 0;
        return overflow ? std::optional<uint64_t>{} : result;
    }
    if (op == Binary::Subtract) {
        const auto result = (a - b) & mask;
        const bool overflow =
            type.isUnsigned ? a < b : ((a ^ b) & type.sign()) != 0 && ((a ^ result) & type.sign()) != 0;
        return overflow ? std::optional<uint64_t>{} : result;
    }
    const bool negativeA = !type.isUnsigned && (a & type.sign());
    const bool negativeB = !type.isUnsigned && (b & type.sign());
    const auto magnitudeA = negativeA ? ((~a + 1) & mask) : a;
    const auto magnitudeB = negativeB ? ((~b + 1) & mask) : b;
    if (op == Binary::Multiply) {
        const bool negative = negativeA != negativeB;
        const auto limit = type.isUnsigned ? mask : (negative ? type.sign() : type.sign() - 1);
        if (magnitudeB && magnitudeA > limit / magnitudeB) {
            return {};
        }
        const auto product = magnitudeA * magnitudeB;
        return negative ? ((~product + 1) & mask) : product;
    }
    if (!magnitudeB || (!type.isUnsigned && a == type.minimum() && b == mask)) {
        return {}; // do not admit a signed min/-1 lowering or division by zero
    }
    const auto remainder = magnitudeA % magnitudeB;
    return negativeA ? ((~remainder + 1) & mask) : remainder;
}

// Check the *original* positive-step counted loop, including its final update.
// The step is a positive mathematical integer, not an unsigned reinterpretation
// of a negative step. A zero-trip loop executes no increment.
inline bool countedLoop(Integer type, uint64_t lower, uint64_t upper, uint64_t step)
{
    if (!type.fits(lower) || !type.fits(upper) || !step || step > type.maximum()) {
        return false;
    }
    if (!type.less(lower, upper)) {
        return true;
    }
    const auto remaining = (upper - lower) & type.mask();
    const auto lastDistance = ((remaining - 1) / step) * step;
    const auto last = (lower + lastDistance) & type.mask();
    return bool(checked(Binary::Add, type, last, step));
}

// Recipes used by LoopHasPrevious/LoopHasNext. On a qualified body visit,
// subtraction is an UNSIGNED distance, even for a signed original loop. This
// avoids speculative iv + k*step and its overflow on the final iteration.
inline std::optional<bool> hasPrevious(
    Integer type, uint64_t lower, uint64_t upper, uint64_t step, uint64_t iv, uint64_t distance)
{
    if (!countedLoop(type, lower, upper, step) || !type.fits(iv) || type.less(iv, lower) || !type.less(iv, upper)) {
        return {};
    }
    const auto elapsed = (iv - lower) & type.mask();
    if (elapsed % step) {
        return {};
    }
    return elapsed / step >= distance;
}
inline std::optional<bool> hasNext(
    Integer type, uint64_t lower, uint64_t upper, uint64_t step, uint64_t iv, uint64_t distance)
{
    if (!hasPrevious(type, lower, upper, step, iv, 0)) {
        return {};
    }
    const auto remaining = (upper - iv) & type.mask();
    return (remaining - 1) / step >= distance;
}

struct IntervalExtrema {
    bool nonempty = false;
    uint64_t first = 0, last = 0; // last is meaningful only when nonempty
};
// This conditional is part of the executable recipe, not merely a proof guard.
// In particular, unsigned zero and signed MIN are never decremented on an empty
// interval. The bounds may already be shared min/max expressions.
inline std::optional<IntervalExtrema> interval(Integer type, uint64_t lower, uint64_t upper)
{
    if (!type.fits(lower) || !type.fits(upper)) {
        return {};
    }
    if (!type.less(lower, upper)) {
        return IntervalExtrema{};
    }
    return IntervalExtrema{true, lower, (upper - 1) & type.mask()};
}

} // namespace mlir::pto::frontiersynch::value_arithmetic
#endif
