// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Composable allocation lifetimes. Queries and selectors borrow one expression arena.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALALLOCATION_H
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
namespace mlir::pto::frontiersynch {
struct RegionalAllocationMember {
    uint32_t record = 0;
    uint64_t stride = 0, phase = 0;
    RegionalEvent firstSource, lastTarget;
    RegionExpressions::Id active = 0;
};
// Internal reuse within a palette is already certified. Different palettes may
// share IDs only when every pair of their active lifetime envelopes is ordered.
struct RegionalAllocationGroup {
    uint32_t sourcePipe = 0, targetPipe = 0;
    uint64_t budget = 0;
    std::vector<RegionalAllocationMember> members;
};
struct RegionalAllocationSummary { std::vector<RegionalAllocationGroup> groups; };
// For an unconditional fixed skeleton: every type is present in each period.
// Guarded periodic producers must supply their own guarded lifetime selectors.
std::shared_ptr<RegionalAllocationSummary> periodicRegionalAllocation(
    const RegionalAnalysis& region, const PeriodicAnalysis& periodic, RegionExpressions::Id trips);
std::shared_ptr<RegionalAllocationSummary> finiteRegionalAllocation(
    const RegionalAnalysis& region, const PreparedLogicalPlan& plan);
DictionaryAttr regionalAllocationCertificate(const RegionalAnalysis& region, const PreparedLogicalPlan& plan);
FailureOr<PhysicalAllocationPlan> decodeRegionalAllocation(func::FuncOp function,
    DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds);
} // namespace mlir::pto::frontiersynch
#endif
