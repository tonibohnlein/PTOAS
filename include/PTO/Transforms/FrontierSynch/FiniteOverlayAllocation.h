// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FINITEOVERLAYALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FINITEOVERLAYALLOCATION_H
#include "PTO/Transforms/FrontierSynch/FiniteOverlay.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
// Overlay IDs are original provenance records, never endpoint namespace IDs.
// Filtering removes uses without invalidating the base's conservative extrema.
// Null means the supplied base has no complete composable allocation proof.
std::shared_ptr<RegionalAllocationSummary> finiteOverlayAllocation(
    const FiniteOverlayAnalysis& analysis, const PreparedLogicalPlan& plan,
    ArrayRef<std::optional<uint32_t>> overlayRecords);
} // namespace mlir::pto::frontiersynch
#endif
