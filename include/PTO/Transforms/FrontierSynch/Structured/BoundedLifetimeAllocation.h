// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Uniform reuse through guaranteed occurrences; no guard valuations are enumerated.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDEDLIFETIMEALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDEDLIFETIMEALLOCATION_H
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetime.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>
namespace mlir::pto::frontiersynch {
struct RegionalAllocationSummary;
// The supplied window comes from one common storage-renaming template. The
// unconditional flags certify that those sites execute in every iteration;
// numeric access modes, cell identities and operation identities repeat under
// that renaming. Exact sourceDemands have at most one active partner per source
// occurrence and destination pipe. Record numbers follow sourceDemands order
// unless originalRecords supplies their stable enclosing-plan IDs.
// Uses only guaranteed intermediate occurrences to certify optional endpoints.
// Applies to one whole-loop invocation. Repeated invocation requires a separate
// visit-qualified lifetime proof; this certificate is not a regional export.
// Empty means no complete uniform proof, not ID exhaustion. Successful palettes
// are sufficient and may exceed the minimum required hardware capacity.
DictionaryAttr boundedLifetimeAllocationCertificate(
    func::FuncOp function, RegionExpressions& expressions, const LifetimeWindowInput& window,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> sourceDemands, int64_t plan,
    llvm::ArrayRef<uint32_t> originalRecords = {});
// Ordinary palettes only: retain the same mandatory-return reuse assignment
// and one-coordinate source-ordinal rules as the whole-invocation proof. Every
// optional endpoint requires guaranteed native-pipe enclosing occurrences.
// Coarse guarded envelopes are safe but may prevent palette sharing. No storage
// lane coordinate is added; storage-backed regional allocation is unsupported.
// Window guards belong to producerExpressions; trips and every returned event,
// activity predicate and selector belong to outputExpressions. The numerical
// assignment carries no expression IDs across this explicit arena boundary.
// Empty means no supported regional proof; demand/endpoint results stay valid.
std::shared_ptr<RegionalAllocationSummary> boundedLifetimeRegionalAllocation(
    RegionExpressions& producerExpressions, RegionExpressions& outputExpressions,
    const LifetimeWindowInput& window,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> sourceDemands,
    RegionExpressions::Id trips);
} // namespace mlir::pto::frontiersynch
#endif
