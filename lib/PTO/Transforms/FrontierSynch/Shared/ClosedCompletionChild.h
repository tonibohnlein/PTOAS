// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Closed executable children have a smaller interface than selective queries.
#ifndef PTO_CLOSED_COMPLETION_CHILD_H
#define PTO_CLOSED_COMPLETION_CHILD_H
#include "PTO/Transforms/FrontierSynch/Shared/CompletionTransfer.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include <optional>
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
struct PreparedLogicalPlan;
class CompletionRegion;
class ClosedCompletionChild;
// One read-only qualification request owns recursion detection and successful
// DAG reuse. Source functions must remain unchanged throughout this request.
struct ClosedCompletionImportContext {
    DenseSet<Operation*> active;
    DenseMap<Operation*, std::shared_ptr<const ClosedCompletionChild>> retained;
};
std::optional<CompletionSummary> closedCompletionSummary(const PreparedLogicalPlan& plan);
// Trusted producer only: closed matching and the existing final ALL are already
// established by the complete logical plan. Never re-certifies arbitrary flags.
LogicalResult emitClosedCompletionExport(func::FuncOp function, const CompletionSummary& summary);
FailureOr<CompletionSummary> decodeClosedCompletionExport(func::FuncOp function, std::string& error);
class ClosedCompletionChild {
public:
    enum class ResourceClosure { Empty, CompletedExit, ClosedDescendants };
    static std::shared_ptr<const ClosedCompletionChild> import(func::CallOp call, std::string& error);
    static std::shared_ptr<const ClosedCompletionChild> import(
        func::CallOp call, std::string& error, ClosedCompletionImportContext& context);
    const CompletionSummary& summary() const { return effects; }
    ResourceClosure resourceClosure() const { return resources; }
    bool unchanged() const;
    bool unchanged(DenseSet<const ClosedCompletionChild*>& checked) const;

private:
    func::FuncOp callee;
    Operation* module = nullptr;
    std::string seal;
    CompletionSummary effects;
    ResourceClosure resources = ResourceClosure::Empty;
    // Own descendant proofs: a local body seal does not cover callee bodies.
    std::shared_ptr<const CompletionRegion> qualification;
};
} // namespace mlir::pto::frontiersynch
#endif
