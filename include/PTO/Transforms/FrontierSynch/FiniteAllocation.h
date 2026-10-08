// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Allocation-only lowering of a certified logical plan. No scarcity repair.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FINITEALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FINITEALLOCATION_H
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/ExplicitAnalysis.h"
namespace mlir::pto::frontiersynch {
inline constexpr llvm::StringLiteral FiniteAllocationAttr = "pto.finite_allocation";
// Version-two evidence uses one shared numeric-ID pool. Explicit reuse-order
// exports are exact for the fixed plan; guarded compatibility is sufficient.
// Records denote at most one handoff per invocation, never a repeated family.
// Explicit certificates serialize O(kh) source rank profiles, not h² reuse
// pairs. Decoder greedy uses at most E queries per handoff for the supplied
// pool; only its failure materializes the exact relation for matching.
DictionaryAttr explicitAllocationCertificate(const ExplicitAnalysis& analysis, int64_t plan, MLIRContext* context);
// Null means the regional finite interface is unavailable. No IR mutation.
// Uses original cuts to identify occurrences; it never reads emitted guards.
DictionaryAttr finiteRegionalAllocationCertificate(const RegionalAnalysis& region, const PreparedLogicalPlan& plan);
FailureOr<PhysicalAllocationPlan> decodeFiniteAllocation(
    func::FuncOp function, DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds);
} // namespace mlir::pto::frontiersynch
#endif
