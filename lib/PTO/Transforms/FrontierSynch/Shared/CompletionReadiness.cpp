// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "CompletionReadiness.h"
namespace mlir::pto::frontiersynch {
FailureOr<CompletionReadiness> CompletionReadiness::qualify(ArrayRef<Operation*> roots, const PhaseIndex& index)
{
    CompletionReadiness result;
    DenseSet<Operation*> scope;
    for (auto* root : roots) {
        root->walk([&](Operation* op) { ++result.inspectedOperations; scope.insert(op); });
    }
    DenseSet<const CompoundInstanceElement*> producers;
    for (auto* op : scope) {
        ++result.inspectedOperations;
        for (const auto* phase : index.phasesFor(op)) {
            for (Value value : op->getResults()) {
                if (resultAvailability(*phase, value) != SyncResultAvailability::RequiresCompletion) { continue; }
                Operation* first = nullptr;
                for (Operation* user : value.getUsers()) {
                    ++result.inspectedUses;
                    if (!scope.contains(user) || user->getBlock() != op->getBlock() ||
                        !op->isBeforeInBlock(user) || user->getNumRegions() ||
                        user->hasTrait<OpTrait::IsTerminator>()) { return failure(); }
                    if (!first || user->isBeforeInBlock(first)) { first = user; }
                }
                if (first) { result.cuts.insert(first); }
                producers.insert(phase);
            }
        }
    }
    for (auto* op : scope) {
        ++result.inspectedOperations;
        // Unmapped control, carried and interface obligations remain unavailable.
        if (index.needsValuePrerequisite(op)) { return failure(); }
        for (const auto& prerequisite : index.prerequisitesFor(op)) {
            if (prerequisite.native) { continue; }
            auto* source = prerequisite.producer ? prerequisite.producer->elementOp : nullptr;
            if (!producers.contains(prerequisite.producer) || !prerequisite.directSSA || !source ||
                source->getBlock() != op->getBlock() || !source->isBeforeInBlock(op)) { return failure(); }
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
