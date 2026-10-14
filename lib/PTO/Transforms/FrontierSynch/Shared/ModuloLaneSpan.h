// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Congruence-based span extraction for an already certified allocation rule.
#ifndef PTO_FRONTIERSYNCH_MODULO_LANE_SPAN_H
#define PTO_FRONTIERSYNCH_MODULO_LANE_SPAN_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "ModularArithmetic.h"
#include <numeric>
#include <optional>
namespace mlir::pto::frontiersynch::detail {
struct ModuloLaneSpan {
    RegionExpressions::Id present, first, last;
};
class ModuloLaneDomain {
public:
    ModuloLaneDomain(uint64_t stride, uint64_t phase, uint64_t width)
        : width(width), phase(width ? phase % width : 0), divisor(std::gcd(stride, width)),
          period(width && divisor ? width / divisor : 0)
    {
        if (period > 1) {
            inverse = modularInverse(stride / divisor, period);
        }
    }
    std::optional<uint64_t> residue(uint64_t lane) const
    {
        if (!period || lane >= width) { return std::nullopt; }
        const auto delta = lane >= phase ? lane - phase : width - (phase - lane);
        if (delta % divisor) { return std::nullopt; }
        if (period == 1) { return 0; }
        return (llvm::APInt(128, delta / divisor) * llvm::APInt(128, inverse))
            .urem(llvm::APInt(128, period)).getZExtValue();
    }
    // Inclusive nonnegative-ordinal interval [lower, upper], under exact activity.
    // Coordinates are meaningful only under returned presence. Check remaining
    // room before adding the residue shift, including near UINT64_MAX.
    ModuloLaneSpan span(RegionExpressions& e, RegionExpressions::Id lower,
        RegionExpressions::Id upper, RegionExpressions::Id active, uint64_t lane) const
    {
        const auto selected = residue(lane);
        if (!selected) { return {e.boolean(false), lower, lower}; }
        const auto modulus = e.constant(period), r = e.constant(*selected);
        const auto rem = e.rem(lower, modulus);
        const auto shift = e.select(e.le(rem, r), e.sub(r, rem), e.sub(modulus, e.sub(rem, r)));
        const auto ordered = e.le(lower, upper);
        const auto room = e.select(ordered, e.sub(upper, lower), e.constant(0));
        const auto present = e.land(active, e.land(ordered, e.le(shift, room)));
        auto first = e.add(lower, shift);
        const auto fixedLower = e.constantValue(lower), fixedShift = e.constantValue(shift);
        // Preserve supplied fixed head coordinates even on an absent phase;
        // replacing them by a guarded ordinal would complicate child queries.
        if (!fixedLower || !fixedShift || *fixedShift > UINT64_MAX - *fixedLower) {
            first = e.select(present, first, lower);
        }
        const auto safeUpper = e.select(present, upper, first);
        const auto last = e.sub(safeUpper, e.rem(e.sub(safeUpper, first), modulus));
        return {present, first, last};
    }
private:
    uint64_t width, phase, divisor, period, inverse = 0;
};
} // namespace mlir::pto::frontiersynch::detail
#endif
