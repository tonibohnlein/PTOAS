// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_REGIONALCONSTANTPALETTE_H
#define PTO_FRONTIERSYNCH_REGIONALCONSTANTPALETTE_H
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
// Fold certified constant lanes by pairwise lifetime compatibility. Dynamic
// coordinate formulas keep their original palettes and lane selectors.
std::shared_ptr<RegionalAllocationSummary> coalesceConstantRegionalAllocation(
    const RegionalAnalysis& region, const RegionalAllocationSummary& input);
} // namespace mlir::pto::frontiersynch
#endif
