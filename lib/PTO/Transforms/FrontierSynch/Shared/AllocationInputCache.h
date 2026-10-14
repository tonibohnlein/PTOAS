// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One producer's unchanged command graph, arena, remapping and input bindings.
// Borrowed summaries are immutable: equal owned input identities permit reuse,
// including failure. No structural guard comparison or global memo is involved.
#ifndef PTO_FRONTIERSYNCH_ALLOCATIONINPUTCACHE_H
#define PTO_FRONTIERSYNCH_ALLOCATIONINPUTCACHE_H
#include "PTO/Transforms/FrontierSynch/Shared/AllocationPreparation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <vector>
namespace mlir::pto::frontiersynch {
class AllocationInputCache {
public:
    using Input = std::shared_ptr<const RegionalAllocationSummary>;
    using Summary = AllocationPreparation::Summary;
    Summary request(llvm::ArrayRef<Input> inputs, llvm::function_ref<Summary()> build, bool* reused = nullptr)
    {
        const bool hit = ready && inputs == llvm::ArrayRef<Input>(previous);
        if (reused) { *reused = hit; }
        if (hit) { return result; }
        previous.assign(inputs.begin(), inputs.end());
        result = build();
        ready = true;
        return result;
    }
private:
    std::vector<Input> previous;
    Summary result;
    bool ready = false;
};
} // namespace mlir::pto::frontiersynch
#endif
