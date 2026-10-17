// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Retained finite implementations with no notification resources. This is a
// trusted producer capability; mathematical exactness alone cannot import events.
#ifndef PTO_COMPLETION_INLINE_CHILD_H
#define PTO_COMPLETION_INLINE_CHILD_H
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionTransfer.h"
#include "mlir/IR/IRMapping.h"
namespace mlir::pto::frontiersynch {
class CompletionSourceSnapshot;
class CompletionStorage;
class ClosedCompletionChild;
class CompletionInlineChild {
public:
    static std::shared_ptr<const CompletionInlineChild> qualify(std::shared_ptr<FiniteAnalysis> finite);
    static std::shared_ptr<const CompletionInlineChild> qualify(std::shared_ptr<const CompletionRegion> completion);
    const CompletionSummary& summary() const { return effects; }
    CompletionSummary classifiedSummary(const CompletionStorage& storage, const PhaseIndex& index) const;
    ArrayRef<Operation*> roots() const;
    const SyncInput* input() const;
    LogicalResult append(PreparedLogicalPlan& destination, IRMapping& mapping,
                         SmallVectorImpl<Operation*>& completionCuts,
                         DenseSet<const CompletionSourceSnapshot*>& checkedSources,
                         DenseSet<const ClosedCompletionChild*>& checkedCalls) const;
private:
    std::shared_ptr<FiniteAnalysis> finite;
    std::shared_ptr<const CompletionRegion> completion;
    CompletionSummary effects;
    std::shared_ptr<const FamilyEndpointRecipe> recipe;
    std::vector<EndpointFamily> families;
    int64_t planId = 0;
    bool nested = false, independent = false, grouped = false;
};
} // namespace mlir::pto::frontiersynch
#endif
