// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Prefix budgets and widened modulo offsets, independent of hardware mapping.
#include "PTO/Transforms/FrontierSynch/PeriodicAllocation.h"
#include "llvm/ADT/APInt.h"
#include <algorithm>
#include <limits>
namespace mlir::pto::frontiersynch {
AllocationNumber DirectedAllocation::handoffCount(uint64_t length) const
{
    if (!handoffs.empty() && !payloadTypes) {
        return {AllocationError::InvalidInput, uint64_t(0)};
    }
    uint64_t total = 0;
    for (const auto& handoff : handoffs) {
        if (handoff.target >= payloadTypes) {
            return {AllocationError::InvalidInput, uint64_t(0)};
        }
        if (length <= handoff.target) {
            continue;
        }
        const auto lastPeriod = (length - 1 - handoff.target) / payloadTypes;
        if (lastPeriod < handoff.displacement) {
            continue;
        }
        // lastPeriod <= length-1, so each addend fits even at UINT64_MAX.
        const auto count = lastPeriod - handoff.displacement + 1;
        if (count > UINT64_MAX - total) {
            return {AllocationError::Overflow, uint64_t(0)};
        }
        total += count;
    }
    return {AllocationError::None, total};
}
AllocationNumber DirectedAllocation::finiteBudget(uint64_t length) const
{
    auto total = handoffCount(length);
    if (total.error != AllocationError::None) {
        return total;
    }
    uint64_t budget = 0;
    const uint64_t phases = std::min<uint64_t>(handoffs.size(), *total.value);
    for (uint64_t r = 0; r < phases; ++r) {
        const auto remaining = *total.value - r;
        const auto reuse = handoffs[r].firstReuse;
        budget = std::max(budget, reuse ? std::min(*reuse, remaining) : remaining);
    }
    return {AllocationError::None, budget};
}
AllocationDecision DirectedAllocation::capacitySuffices(uint64_t capacity, std::optional<uint64_t> length) const
{
    const auto bound = length ? finiteBudget(*length) : AllocationNumber{AllocationError::None, uniformBudget};
    return {bound.error, bound.error == AllocationError::None && bound.value && capacity >= *bound.value};
}
AllocationNumber DirectedAllocation::localOffset(uint32_t phase, uint64_t sourcePeriod, uint64_t capacity) const
{
    if (!capacity || phase >= handoffs.size() || handoffs.size() > UINT32_MAX) {
        return {AllocationError::InvalidInput, uint64_t(0)};
    }
    const auto ordinal = llvm::APInt(128, handoffs.size()) * llvm::APInt(128, sourcePeriod) + llvm::APInt(128, phase);
    return {AllocationError::None, ordinal.urem(llvm::APInt(128, capacity)).getZExtValue()};
}
} // namespace mlir::pto::frontiersynch
