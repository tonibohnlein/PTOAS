// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Estimates rank implemented choices; unknown work never excludes a method.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ANALYSISCOST_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ANALYSISCOST_H
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
namespace mlir::pto::frontiersynch {
using EstimatedCount = std::optional<uint64_t>;
inline EstimatedCount estimatedAdd(EstimatedCount a, EstimatedCount b)
{
    if (!a || !b || *b > UINT64_MAX - *a) { return {}; }
    return *a + *b;
}
inline EstimatedCount estimatedMultiply(EstimatedCount a, EstimatedCount b)
{
    const bool zero = (a && !*a) || (b && !*b);
    if (zero) { return 0; }
    if (!a || !b || *b > UINT64_MAX / *a) { return {}; }
    return *a * *b;
}
inline EstimatedCount estimatedPower(EstimatedCount base, EstimatedCount exponent)
{
    if (exponent && !*exponent) { return 1; }
    if (!base || !exponent) { return {}; }
    EstimatedCount result = 1;
    for (uint64_t count = *exponent; count; count >>= 1) {
        if (count & 1) { result = estimatedMultiply(result, base); }
        if (count > 1) { base = estimatedMultiply(base, base); }
    }
    return result;
}
inline void accumulateCost(uint64_t& target, uint64_t amount)
{
    target = amount > UINT64_MAX - target ? UINT64_MAX : target + amount;
}
struct AnalysisCostEstimate {
    EstimatedCount work;
    EstimatedCount representation;
    EstimatedCount generatorPieces;
    EstimatedCount ports;
    EstimatedCount numericalWindow;
    EstimatedCount circuitNodes;
    EstimatedCount relationConversion;
};
inline bool estimatedCostLess(const AnalysisCostEstimate& a, const AnalysisCostEstimate& b)
{
    auto compare = [](EstimatedCount x, EstimatedCount y) {
        const bool differentAvailability = x.has_value() != y.has_value();
        if (differentAvailability) { return x ? -1 : 1; }
        if (!x || *x == *y) { return 0; }
        return *x < *y ? -1 : 1;
    };
    const auto work = compare(a.work, b.work);
    return work ? work < 0 : compare(a.representation, b.representation) < 0;
}
struct AnalysisCostRecord {
    std::size_t region = 0;
    // Query, selector, synchronization and fallback bits, respectively.
    uint8_t request = 0;
    std::string method;
    AnalysisCostEstimate estimate;
    uint64_t attemptConstructions = 0;
};
} // namespace mlir::pto::frontiersynch
#endif
