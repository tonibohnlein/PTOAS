// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_LOGICALSYNCCOMPACTFORMS_H
#define PTO_TRANSFORMS_INSERTSYNC_LOGICALSYNCCOMPACTFORMS_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace mlir::pto::logical_sync::compact {

// These helpers recognize syntax; they are NOT an integer-set solver. The
// native caller still proves the complete proposed guard equivalent to the
// requested occurrence domain and qualifies every emitted arithmetic operation.
struct LinearForm {
    std::vector<int64_t> coefficients;
    int64_t constant = 0;
};
enum class Comparison { Equal, AtLeast, AtMost };
struct Bound {
    Comparison comparison;
    int64_t value;
};
inline std::optional<int64_t> checkedAdd(int64_t a, int64_t b)
{
    if ((b > 0 && a > std::numeric_limits<int64_t>::max() - b) ||
        (b < 0 && a < std::numeric_limits<int64_t>::min() - b)) return {};
    return a + b;
}
inline std::optional<int64_t> checkedSubtract(int64_t a, int64_t b)
{
    if ((b > 0 && a < std::numeric_limits<int64_t>::min() + b) ||
        (b < 0 && a > std::numeric_limits<int64_t>::max() + b)) return {};
    return a - b;
}

// Recognize row(x) >= 0 (or == 0) as expression(x) CMP bound.
// Only equal/opposite coefficient vectors are recognized. No scaling,
// projection, rational relaxation, or assumed relation between coordinates.
inline std::optional<Bound> matchBound(const LinearForm& row, bool equality,
                                      const LinearForm& expression)
{
    if (row.coefficients.size() != expression.coefficients.size()) return {};
    bool positive = true, negative = true, nonconstant = false;
    for (size_t i = 0; i < row.coefficients.size(); ++i) {
        const int64_t a = row.coefficients[i], b = expression.coefficients[i];
        nonconstant |= b != 0;
        positive &= a == b;
        negative &= b != std::numeric_limits<int64_t>::min() && a == -b;
    }
    if (!nonconstant) return {};
    auto bound = positive ? checkedSubtract(expression.constant, row.constant) :
                 negative ? checkedAdd(expression.constant, row.constant) : std::nullopt;
    if (!bound) return {};
    return Bound{equality ? Comparison::Equal :
                 positive ? Comparison::AtLeast : Comparison::AtMost, *bound};
}

struct Residue {
    unsigned coordinate;
    int64_t modulus;
    int64_t value;
};
// Recognize a unit scalar equality x +/- m*q + c == 0. q must be the ONLY
// mentioned local and x the ONLY mentioned public coordinate. Constraints on
// q are deliberately NOT dropped by this helper: this is a proposal, and the
// native whole-domain equivalence check must still establish their effects.
inline std::optional<Residue> matchResidue(const std::vector<int64_t>& row,
                                          unsigned publicCoordinates)
{
    if (row.size() <= size_t(publicCoordinates) + 1) return {};
    std::optional<unsigned> scalar, local;
    for (unsigned i = 0; i + 1 < row.size(); ++i) {
        if (!row[i]) continue;
        if (i < publicCoordinates) {
            if (scalar || (row[i] != 1 && row[i] != -1)) return {};
            scalar = i;
        } else {
            if (local) return {};
            local = i;
        }
    }
    if (!scalar || !local || row[*local] == std::numeric_limits<int64_t>::min()) return {};
    int64_t modulus = row[*local] < 0 ? -row[*local] : row[*local];
    if (modulus <= 1) return {};
    int64_t remainder = row.back() % modulus;
    // Reduce first: never negate INT64_MIN or compute a potentially overflowing
    // constant + modulus. A negative unit coefficient reverses the sign again.
    int64_t value = row[*scalar] == 1 ?
        (remainder > 0 ? modulus - remainder : -remainder) :
        (remainder < 0 ? remainder + modulus : remainder);
    return Residue{*scalar, modulus, value};
}

// Preserve deterministic key order, but do not build a larger event family
// merely to save a key when an unused one exists. Occupancy is a snapshot for
// one immutable assignment step; the caller must still prove recurrence.
template <typename Occupied>
std::vector<unsigned> unusedFirstKeys(unsigned capacity, Occupied occupied)
{
    std::vector<unsigned> free, used;
    free.reserve(capacity);
    used.reserve(capacity);
    for (unsigned k = 0; k < capacity; ++k)
        (occupied(k) ? used : free).push_back(k);
    free.insert(free.end(), used.begin(), used.end());
    return free;
}

// Preserve the existing sharing-first search in a genuinely over-capacity
// domain. Otherwise early dedicated assignments could occupy all keys and
// prevent a later stream from using a key that earlier sharing would free.
template <typename Occupied>
std::vector<unsigned> eventKeyTrialOrder(unsigned capacity, size_t streamCount, Occupied occupied)
{
    if (streamCount <= capacity) return unusedFirstKeys(capacity, occupied);
    std::vector<unsigned> keys;
    keys.reserve(capacity);
    for (unsigned k = 0; k < capacity; ++k) keys.push_back(k);
    return keys;
}

} // namespace mlir::pto::logical_sync::compact
#endif
