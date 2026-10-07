// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALLANEEXPORTS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALLANEEXPORTS_H
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include <utility>
namespace mlir::pto::frontiersynch {
std::pair<uint32_t, uint32_t> regionalAllocationDirection(
    const RegionalAllocationGroup& group, const RegionalAllocationMember& member);
// Materialize only bounded physical lanes (at most six), never loop visits.
// Failure leaves the group unchanged. Existing complete metadata is preserved.
bool exportConstantRegionalLanes(RegionalAllocationGroup& group);
// The original region owns the expression arena and must outlive the summary.
std::shared_ptr<RegionalAllocationSummary> exportPeriodicSharedLanes(
    const RegionalAnalysis& region, const PeriodicAnalysis& periodic, RegionExpressions::Id trips);
} // namespace mlir::pto::frontiersynch
#endif
