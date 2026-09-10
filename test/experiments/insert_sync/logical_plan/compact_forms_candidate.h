// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

#ifndef PTO_TEST_LOGICAL_SYNC_COMPACT_FORMS_CANDIDATE_H
#define PTO_TEST_LOGICAL_SYNC_COMPACT_FORMS_CANDIDATE_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace mlir::pto::logical_sync::compact_candidate {

// Candidate-only affine/residue recognizers. The native lowering currently
// uses its own qualified direct-row implementation and does not call these.
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
    int64_t value = row[*scalar] == 1 ?
        (remainder > 0 ? modulus - remainder : -remainder) :
        (remainder < 0 ? remainder + modulus : remainder);
    return Residue{*scalar, modulus, value};
}

} // namespace mlir::pto::logical_sync::compact_candidate
#endif
