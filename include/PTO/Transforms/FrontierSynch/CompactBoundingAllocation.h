// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTBOUNDINGALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTBOUNDINGALLOCATION_H
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
// Attach the existing periodic shared-cycle proof to an exactly paired compact
// plan on this selected upper graph. Every abstract slot executes once per
// iteration (a balanced slot may choose a different original cut each time).
// Original canonical generator IDs are unchanged in family/member metadata.
// The producer certifies those execution/graph premises. This adapter never
// creates or weakens demands. Existing physical hidden-macro reservation gates
// remain applicable; this adapter does not supply a new reservation proof.
// Empty plans export a zero-budget certificate bound to the original invocation.
// True means an exported certificate, not that hardware capacity
// suffices. Per-direction numeric capacity is still checked by the physical decoder;
// missing reuse proof and scarcity never trigger repairs or discard the plan.
// Typed lane summaries are supplied when the existing regional exporter supports
// the graph/frame. No minimum for arbitrary plans or actual excess is claimed.
bool attachCompactBoundingAllocation(const RegionalAnalysis& region, const PeriodicAnalysis& selected,
    RegionExpressions::Id trips, PreparedLogicalPlan& plan, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
