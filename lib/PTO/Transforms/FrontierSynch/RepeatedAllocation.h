// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_REPEATEDALLOCATION_H
#define PTO_FRONTIERSYNCH_REPEATEDALLOCATION_H
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
struct RepeatedRegionState;
using RepeatedAllocationExtrema = std::pair<std::vector<RegionalSelector>, std::vector<RegionalSelector>>;
std::optional<RepeatedAllocationExtrema> repeatedAllocationExtrema(
    const RegionalAnalysis& body, const RegionalAllocationGroup& group);
// Restricting a finite invocation must preserve the complete internal chain,
// or the member must certify one handoff whose paired endpoints are clipped.
bool canRestrictRepeatedAllocationMember(RepeatedRegionState& state, const RegionalAllocationMember& member);
bool liftRepeatedAllocationEnvelope(RepeatedRegionState& state, RegionalAllocationMember& member, uint64_t delay);
// Returns unit lanes only for structurally constant certified offsets. Dynamic
// groups are returned unchanged, so callers explicitly check the resulting width.
std::vector<RegionalAllocationGroup> constantAllocationLanes(const RegionalAllocationGroup& group);
// Child palettes already certify internal reuse. Crossings identify newly
// inserted records and their index in the unchanged repeated crossing list.
// Failure means this sufficient palette construction is unavailable, not that
// the logical plan requires more physical IDs.
std::shared_ptr<RegionalAllocationSummary> repeatedRegionalAllocation(
    RepeatedRegionState& state, const RegionalAllocationSummary& child,
    ArrayRef<std::pair<uint32_t, std::size_t>> crossings);
} // namespace mlir::pto::frontiersynch
#endif
