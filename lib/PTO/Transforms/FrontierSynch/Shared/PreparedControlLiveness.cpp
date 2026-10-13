// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared liveness boundary for generated control and non-SSA plan evidence.
#include "PreparedControlLiveness.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
namespace mlir::pto::frontiersynch {
llvm::DenseSet<Operation*> livePreparedControl(const PreparedLogicalPlan& plan, bool includeEvidence)
{
    llvm::DenseSet<Operation*> operations, live;
    SmallVector<Operation*> pending;
    for (const auto& stage : plan.preparation) {
        for (auto& operation : *stage.code) { operations.insert(&operation); }
    }
    if (includeEvidence && plan.allocationPreparation) { return operations; }
    for (auto* operation : operations) {
        if (!isMemoryEffectFree(operation) ||
            (includeEvidence && operation->getName().getDialectNamespace() != "arith")) {
            pending.push_back(operation);
        }
        if (includeEvidence && llvm::any_of(operation->getUsers(), [&](Operation* user) {
            return !operations.contains(user);
        })) { pending.push_back(operation); }
    }
    auto root = [&](Value value) {
        if (auto* operation = value ? value.getDefiningOp() : nullptr) { pending.push_back(operation); }
    };
    for (const auto& endpoint : plan.endpoints) {
        root(endpoint.guard); root(endpoint.identity);
        for (auto value : endpoint.memberCoordinates) { root(value); }
        if (includeEvidence) {
            for (const auto& box : endpoint.domain) {
                for (const auto& constraint : box) { root(constraint.coordinate); }
            }
        }
    }
    if (includeEvidence) {
        for (const auto& family : plan.families) {
            for (const auto& choices : {family.sourceChoices, family.targetChoices}) {
                if (choices) { root(choices->consumerPredicate()); }
            }
        }
    }
    while (!pending.empty()) {
        auto* operation = pending.pop_back_val();
        if (!operations.contains(operation) || !live.insert(operation).second) { continue; }
        for (auto value : operation->getOperands()) { root(value); }
    }
    return live;
}
} // namespace mlir::pto::frontiersynch
