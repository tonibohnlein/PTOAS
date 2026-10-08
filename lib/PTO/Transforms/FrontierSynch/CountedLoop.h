// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_COUNTEDLOOP_H
#define PTO_FRONTIERSYNCH_COUNTEDLOOP_H
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "../InsertSync/SyncScalarEvolution.h"
namespace mlir::pto::frontiersynch {
// Original signed index coordinates remain in the IR. Regional interfaces use
// nonnegative ordinals. Bounds prove the ordinal fits signed i64, as required
// by the shared exact integer circuits; no runtime iteration is enumerated.
struct CountedLoop {
    scf::ForOp loop;
    int64_t step;
    uint64_t maximumOrdinal;
    static std::optional<CountedLoop> get(scf::ForOp loop)
    {
        if (!loop || !loop.getInductionVar().getType().isIndex()) { return std::nullopt; }
        APInt step;
        auto bits = DataLayout::closest(loop).getTypeSizeInBits(loop.getInductionVar().getType());
        if (bits.isScalable() || bits.getFixedValue() != 64 ||
            !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
            !step.isSignedIntN(64) || step.getSExtValue() <= 0) { return std::nullopt; }
        mlir::pto::detail::ScalarEvolution range(loop.getContext(), loop);
        auto lower = range.signedRange(loop.getLowerBound()), upper = range.signedRange(loop.getUpperBound());
        if (!lower || !upper) { return std::nullopt; }
        auto distance = APInt(128, upper->second, true) - APInt(128, lower->first, true);
        auto count = distance.isStrictlyPositive() ?
            (distance - 1).udiv(step.sextOrTrunc(128)) + 1 : APInt(128, 0);
        if (!count.isSignedIntN(64)) { return std::nullopt; }
        return CountedLoop{loop, step.getSExtValue(), count.isZero() ? 0 : count.getZExtValue() - 1};
    }
    // At an executed cut IV >= lower in signed order. Their difference is
    // the exact unsigned distance even across zero; division yields the
    // original visit ordinal without reconstructing or changing the IV.
    RegionExpressions::Id ordinal(RegionExpressions& arena) const
    {
        auto current = loop;
        return arena.div(arena.sub(arena.input(current.getInductionVar()), arena.input(current.getLowerBound())),
            arena.constant(step));
    }
    RegionExpressions::Id trips(RegionExpressions& arena) const
    {
        auto zero = arena.constant(0), one = arena.constant(1), stride = arena.constant(step);
        auto current = loop;
        auto lower = arena.input(current.getLowerBound()), upper = arena.input(current.getUpperBound());
        // In the nonempty branch unsigned subtraction is the exact distance,
        // including a signed interval crossing zero. Divide before rounding up.
        auto distance = arena.select(arena.slt(lower, upper), arena.sub(upper, lower), zero);
        return arena.add(arena.div(distance, stride),
            arena.select(arena.eq(arena.rem(distance, stride), zero), zero, one));
    }
};
} // namespace mlir::pto::frontiersynch
#endif
