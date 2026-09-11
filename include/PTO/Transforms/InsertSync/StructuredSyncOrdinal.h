// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCORDINAL_H
#define PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCORDINAL_H

#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>

namespace mlir::pto::structured_sync {

// Original signed-index loop i = lower + step*k. The production adapter
// requires constant nonnegative lower and constant positive step. Dynamic
// upper bounds may be negative: that is the empty loop, not wrapped arithmetic.
// This is analysis-only normalization. No payload loop or induction use changes.
struct LoopOrdinal {
    int64_t lower = 0;
    int64_t step = 1;
    bool valid() const { return lower >= 0 && step > 0; }
    std::optional<uint64_t> count(int64_t upper) const {
        if (!valid()) return {};
        if (upper <= lower) return uint64_t(0);
        const auto extent = uint64_t(upper) - uint64_t(lower);
        const auto stride = uint64_t(step);
        return extent / stride + (extent % stride != 0);
    }
    std::optional<int64_t> distance(uint64_t ordinals) const {
        if (!valid() || ordinals > uint64_t(INT64_MAX) / uint64_t(step)) return {};
        return int64_t(ordinals * uint64_t(step));
    }
    std::optional<int64_t> value(uint64_t ordinal) const {
        if (!valid() || ordinal > (uint64_t(INT64_MAX) - uint64_t(lower)) / uint64_t(step)) return {};
        return lower + int64_t(ordinal * uint64_t(step));
    }
    std::optional<uint64_t> index(int64_t raw) const {
        if (!valid() || raw < lower) return {};
        const auto delta = uint64_t(raw) - uint64_t(lower);
        if (delta % uint64_t(step)) return {};
        return delta / uint64_t(step);
    }
};

inline std::optional<int64_t> ordinalAdd(int64_t a, int64_t b) {
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return {};
    return a + b;
}
inline std::optional<int64_t> ordinalNegate(int64_t a) {
    if (a == INT64_MIN) return {};
    return -a;
}
inline std::optional<int64_t> ordinalMultiply(int64_t a, int64_t b) {
    if (!a || !b) return int64_t(0);
    if ((a == -1 && b == INT64_MIN) || (b == -1 && a == INT64_MIN)) return {};
    if (a > 0 ? (b > 0 ? a > INT64_MAX / b : b < INT64_MIN / a)
              : (b > 0 ? a < INT64_MIN / b : a < INT64_MAX / b)) return {};
    return a * b;
}

// All operands are residues. Addition and multiplication stay in range even
// near UINT64_MAX. Multiplication takes at most 64 bit steps, not modulus steps.
inline uint64_t ordinalModAdd(uint64_t a, uint64_t b, uint64_t m) {
    return a >= m - b ? a - (m - b) : a + b;
}
inline uint64_t ordinalModMultiply(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t result = 0;
    while (b) {
        if (b & 1) result = ordinalModAdd(result, a, m);
        b >>= 1;
        if (b) a = ordinalModAdd(a, a, m);
    }
    return result;
}
struct OrdinalResidue {
    uint64_t modulus = 1, base = 0, stride = 0, period = 1;
    static std::optional<OrdinalResidue> get(const LoopOrdinal &loop, uint64_t modulus,
                                            uint64_t offset = 0) {
        auto first = loop.value(offset);
        if (!first || !modulus || modulus > uint64_t(INT64_MAX)) return {};
        return OrdinalResidue{modulus, uint64_t(*first) % modulus,
            uint64_t(loop.step) % modulus, modulus / std::gcd(uint64_t(loop.step), modulus)};
    }
    uint64_t at(uint64_t ordinal) const {
        return ordinalModAdd(base, ordinalModMultiply(stride, ordinal % modulus, modulus), modulus);
    }
    // Exact preimage of one raw selector value. Euclid decreases the remainder
    // each step. An absent result means the raw value is never selected.
    std::optional<uint64_t> preimage(uint64_t selected) const {
        if (selected >= modulus) return {};
        const uint64_t g = modulus / period;
        const uint64_t difference = selected >= base ? selected - base : modulus - (base - selected);
        if (difference % g) return {};
        if (period == 1) return uint64_t(0);
        uint64_t oldR = period, r = stride / g, oldT = 0, t = 1;
        while (r) {
            const uint64_t q = oldR / r, nextR = oldR % r;
            const uint64_t product = ordinalModMultiply(q % period, t, period);
            const uint64_t nextT = oldT >= product ? oldT - product : period - (product - oldT);
            oldR = r; r = nextR; oldT = t; t = nextT;
        }
        if (oldR != 1) return {};
        return ordinalModMultiply(oldT, (difference / g) % period, period);
    }
};

enum class OrdinalCompare : uint8_t { EQ, NE, SLT, SLE, SGT, SGE, ULT, ULE, UGT, UGE };
inline bool ordinalCompare(OrdinalCompare predicate, int64_t a, int64_t b, bool booleanOperands) {
    // i1 true has unsigned value 1 and signed value -1. Keep the bit pattern
    // for equality/unsigned comparisons; reinterpret only signed ordering.
    const int64_t sa = booleanOperands ? -(a & 1) : a;
    const int64_t sb = booleanOperands ? -(b & 1) : b;
    switch (predicate) {
    case OrdinalCompare::EQ: return a == b;
    case OrdinalCompare::NE: return a != b;
    case OrdinalCompare::SLT: return sa < sb;
    case OrdinalCompare::SLE: return sa <= sb;
    case OrdinalCompare::SGT: return sa > sb;
    case OrdinalCompare::SGE: return sa >= sb;
    case OrdinalCompare::ULT: return uint64_t(a) < uint64_t(b);
    case OrdinalCompare::ULE: return uint64_t(a) <= uint64_t(b);
    case OrdinalCompare::UGT: return uint64_t(a) > uint64_t(b);
    case OrdinalCompare::UGE: return uint64_t(a) >= uint64_t(b);
    }
    return false;
}
} // namespace mlir::pto::structured_sync
#endif
