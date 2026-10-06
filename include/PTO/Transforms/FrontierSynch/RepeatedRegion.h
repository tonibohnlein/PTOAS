// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Common exact regional contract. All expressions belong to the supplied arena.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REPEATEDREGION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REPEATEDREGION_H
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
struct RepeatedCrossing {
    RegionalEvent source, target; // Body coordinates; target is in the next visit.
    RegionExpressions::Id guard = RegionExpressions::invalid;
    bool native = false;
};
struct RepeatedRegionState;
struct RepeatedRegionAnalysis {
    std::string error;
    RegionalAnalysis regional;
    std::shared_ptr<RepeatedRegionState> state;
};
// Construct q=1 repetition of an invariant body. The caller certifies invariant
// control, physical byte maps, query/selector guards and per-pipe presence,
// complete storage/native selectors and native start/completion order in Q,
// and no extra loop-carried
// prerequisites. The body may itself contain compact repeated regions. This
// operation does not establish invariance from IR or allocate physical IDs.
// Inputs must satisfy the automatic-sync front-end contract (in particular,
// manual descriptor rebinding is excluded by validateTAssignConfiguration).
RepeatedRegionAnalysis repeatInvariantRegion(func::FuncOp function, scf::ForOp loop,
    RegionalAnalysis body, RegionExpressions::Id trips);
} // namespace mlir::pto::frontiersynch
#endif
