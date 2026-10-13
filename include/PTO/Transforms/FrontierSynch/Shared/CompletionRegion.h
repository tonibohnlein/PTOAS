// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owned conservative completion evidence; independent of exact demands/queries.
#ifndef PTO_FRONTIERSYNCH_COMPLETIONREGION_H
#define PTO_FRONTIERSYNCH_COMPLETIONREGION_H
#include "PTO/Transforms/FrontierSynch/Shared/CompletionTransfer.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include <memory>
namespace mlir::pto::frontiersynch {
class PhaseIndex;
struct PreparedLogicalPlan;
class CompletionRegion {
public:
    static std::shared_ptr<const CompletionRegion> build(func::FuncOp function,
        std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error);
    const CompletionTransferResult& transfer() const { return result; }
    std::shared_ptr<const SyncInput> inputOwner() const { return input; }
    // No event publications: matching is closed and physical IDs are unnecessary.
    LogicalResult insert(func::FuncOp function, PreparedLogicalPlan& plan) const;
private:
    func::FuncOp original;
    std::shared_ptr<const SyncInput> input;
    CompletionTransferResult result;
    std::vector<Operation*> cuts, originalOperations;
    std::string originalText;
    DictionaryAttr moduleAttributes;
};
} // namespace mlir::pto::frontiersynch
#endif
