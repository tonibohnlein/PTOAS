// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One immutable may-overlap partition for all scopes of a source invocation.
#ifndef PTO_COMPLETION_STORAGE_H
#define PTO_COMPLETION_STORAGE_H
#include "PTO/Transforms/FrontierSynch/Shared/CompletionTransfer.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "llvm/ADT/DenseMap.h"
namespace mlir::pto::frontiersynch {
class PhaseIndex;
class CompletionStorage {
public:
    explicit CompletionStorage(ArrayRef<const CompoundInstanceElement*> phases);
    std::size_t size() const { return classCount; }
    CompletionFacts effects(ArrayRef<const CompoundInstanceElement*> phases) const;
    CompletionSummary summarize(ArrayRef<Operation*> roots, const PhaseIndex& index) const;
    // A coarse call's class numbering is never a caller reservation identity.
    CompletionSummary liftCoarse(const CompletionSummary& summary) const;
private:
    std::size_t classCount = 1;
    struct Membership {
        // Unknown effects name a contiguous class range without copying it.
        std::size_t first = 0, count = 0;
        SmallVector<std::size_t> ids;
    };
    DenseMap<const BaseMemInfo*, Membership> memberships;
};
CompletionSummary collapseCompletionSummary(const CompletionSummary& summary);
} // namespace mlir::pto::frontiersynch
#endif
