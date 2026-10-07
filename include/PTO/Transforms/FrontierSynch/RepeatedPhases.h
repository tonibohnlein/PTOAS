// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REPEATEDPHASES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REPEATEDPHASES_H
#include "PTO/Transforms/FrontierSynch/RepeatedRegion.h"
namespace mlir::pto::frontiersynch {
// Explicit phase views are charged descriptions, not unfolded loop iterations.
// Callers certify period-invariant control/effects and internal prerequisites.
// Each view retains original cuts. The result's first enclosing coordinate is
// the period ordinal; the payload type identifies its static phase.
// Optional begin restricts the original ordinal interval to [begin, trips).
// Endpoint identities remain absolute; both ends of each handoff are filtered.
// maximumLength, when supplied, is a caller-certified bound on the interval's
// executed visits. A bound <=1 removes the inter-period query/recipe graph.
RepeatedRegionAnalysis repeatPhasedRegions(func::FuncOp function, scf::ForOp loop,
    std::vector<RegionalAnalysis> phases, RegionExpressions::Id trips,
    ArrayRef<scf::ForOp> enclosing = {},
    RegionExpressions::Id begin = RegionExpressions::invalid,
    std::optional<uint64_t> maximumLength = std::nullopt);
} // namespace mlir::pto::frontiersynch
#endif
