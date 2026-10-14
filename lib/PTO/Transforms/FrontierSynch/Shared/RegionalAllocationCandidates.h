// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Select a certified candidate only for an explicit whole-plan capacity request.
#ifndef PTO_FRONTIERSYNCH_REGIONALALLOCATIONCANDIDATES_H
#define PTO_FRONTIERSYNCH_REGIONALALLOCATIONCANDIDATES_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
// Optional lane sharing is a separate upgrade of the already-certified palette
// graph. The existing split export still requests both stages.
DictionaryAttr regionalBaseAllocationCertificate(const RegionalAnalysis& region,
    const PreparedLogicalPlan& plan, AllocationProfile* profile = nullptr);
DictionaryAttr strengthenRegionalAllocationCertificate(const RegionalAnalysis& region,
    const PreparedLogicalPlan& plan, DictionaryAttr base, AllocationProfile* profile = nullptr);
// The caller owns the unchanged regional command plan and supplies the actual
// eligible pool. Success freezes a self-contained certificate; failure retains
// the incumbent plan. No source IR or endpoint arithmetic is materialized.
LogicalResult selectRegionalAllocationCandidate(func::FuncOp function,
    const RegionalAnalysis& region, PreparedLogicalPlan& plan, ArrayRef<int64_t> eligibleIds,
    AllocationProfile* profile = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
