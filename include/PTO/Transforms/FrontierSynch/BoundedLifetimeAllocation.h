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
#include "PTO/Transforms/FrontierSynch/BoundedLifetime.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pto::frontiersynch {
// The supplied window comes from one common storage-renaming template. The
// unconditional flags certify that those sites execute in every iteration;
// numeric access modes, cell identities and operation identities repeat under
// that renaming. Exact sourceDemands have at most one active partner per source
// occurrence and destination pipe. Record numbers follow sourceDemands order.
// Uses only guaranteed intermediate occurrences to certify optional endpoints.
// Applies to one whole-loop invocation. Repeated invocation requires a separate
// visit-qualified lifetime proof; this certificate is not a regional export.
// Empty means no complete uniform proof, not ID exhaustion. Successful palettes
// are sufficient and may exceed the minimum required hardware capacity.
DictionaryAttr boundedLifetimeAllocationCertificate(
    func::FuncOp function, RegionExpressions& expressions, const LifetimeWindowInput& window,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> sourceDemands, int64_t plan);
} // namespace mlir::pto::frontiersynch
#endif
