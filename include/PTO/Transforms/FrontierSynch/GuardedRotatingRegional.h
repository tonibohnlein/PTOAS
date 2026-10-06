// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDROTATINGREGIONAL_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDROTATINGREGIONAL_H
#include "PTO/Transforms/FrontierSynch/GuardedRotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
struct PeriodicSlice {
    RegionExpressions::Id begin, end; // Original iteration ordinals, half-open.
};
// Export one counted immutable region on its existing expression arena. This
// materializes at most maxRegionalSlotVisits physical slot visits per child,
// charged separately from compact periodic analysis. Larger families require a
// compact-selector exporter. No runtime iterations, offsets or guards are expanded.
// The returned callbacks own the analysis and internal endpoint recipes. Original
// IR and SyncInput phase anchors must remain alive and unchanged, as for all
// regional exports. Preparation is detached and adds no invocation completion.
FailureOr<RegionalAnalysis> guardedRotatingRegionalResult(func::FuncOp function, const SyncInput& input,
    const GuardedRotatingAnalysis& analysis, std::string& error,
    std::optional<PeriodicSlice> slice = std::nullopt);
// A slice must satisfy 0 <= begin <= end <= trip count. The caller proves that
// its selected skeleton describes precisely those whole iterations. Original
// ordinal identities and insertion cuts are retained; there is no IR cloning.
} // namespace mlir::pto::frontiersynch
#endif
