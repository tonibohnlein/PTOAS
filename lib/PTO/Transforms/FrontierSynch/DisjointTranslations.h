// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_DISJOINTTRANSLATIONS_H
#define PTO_FRONTIERSYNCH_DISJOINTTRANSLATIONS_H
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
namespace mlir::pto::frontiersynch::detail {
// Exact finite union test for copies U + i*S, 0 <= i < trips. All ranges
// share one physical origin; stride is its nonnegative magnitude in i128.
// O(F^2) arithmetic operations, independent of trip count and byte distances.
inline bool disjointTranslations(ArrayRef<SyncStorageCell> ranges, const APInt& stride, uint64_t trips)
{
    if (trips <= 1) { return true; }
    if (stride.isZero()) { return ranges.empty(); }
    uint64_t begin = UINT64_MAX, end = 0;
    for (const auto& range : ranges) {
        if (range.begin >= range.end) { continue; }
        begin = std::min(begin, range.begin); end = std::max(end, range.end);
    }
    if (begin == UINT64_MAX || stride.uge(APInt(128, end - begin))) { return true; }
    const APInt one(128, 1), last(128, trips - 1);
    for (const auto& a : ranges) {
        if (a.begin >= a.end) { continue; }
        for (const auto& b : ranges) {
            if (b.begin >= b.end) { continue; }
            auto low = APInt(128, a.begin) - APInt(128, b.end) + one;
            auto high = APInt(128, a.end) - APInt(128, b.begin) - one;
            if (high.slt(stride)) { continue; }
            auto first = low.sle(stride) ? one : (low + stride - one).udiv(stride);
            auto final = high.udiv(stride);
            if (first.ule(final) && first.ule(last)) { return false; }
        }
    }
    return true;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
