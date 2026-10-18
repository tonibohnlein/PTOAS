// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "CompletionReadiness.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
namespace mlir::pto::frontiersynch {
FailureOr<CompletionReadiness> CompletionReadiness::qualify(ArrayRef<Operation*> roots, const PhaseIndex& index)
{
    CompletionReadiness result;
    DenseSet<Operation*> scope;
    for (auto* root : roots) {
        root->walk([&](Operation* op) { ++result.inspectedOperations; scope.insert(op); });
    }
    DenseMap<const CompoundInstanceElement*, Operation*> producers;
    DominanceInfo dominance;
    for (auto* op : scope) {
        ++result.inspectedOperations;
        for (const auto* phase : index.phasesFor(op)) {
            for (Value value : op->getResults()) {
                if (resultAvailability(*phase, value) != SyncResultAvailability::RequiresCompletion) { continue; }
                Operation* first = nullptr;
                for (Operation* user : value.getUsers()) {
                    ++result.inspectedUses;
                    if (!scope.contains(user) || user->getBlock() != op->getBlock() ||
                        !op->isBeforeInBlock(user) || (user->getNumRegions() && !isa<scf::IfOp>(user)) ||
                        user->hasTrait<OpTrait::IsTerminator>()) { return failure(); }
                    if (!first || user->isBeforeInBlock(first)) { first = user; }
                }
                if (first) { result.cuts.insert(first); }
                if (first) {
                    auto inserted = producers.try_emplace(phase, first);
                    if (!inserted.second && first->isBeforeInBlock(inserted.first->second)) {
                        inserted.first->second = first;
                    }
                }
            }
        }
    }
    for (auto* op : scope) {
        ++result.inspectedOperations;
        // An indirect branch condition is ready once its actual scalar first
        // use is protected. Carried/interface obligations without such explicit
        // prerequisite provenance remain unavailable.
        auto prerequisites = index.prerequisitesFor(op);
        if (index.needsValuePrerequisite(op) && (!isa<scf::IfOp>(op) || prerequisites.empty())) {
            return failure();
        }
        for (const auto& prerequisite : prerequisites) {
            if (prerequisite.native) { continue; }
            auto* source = prerequisite.producer ? prerequisite.producer->elementOp : nullptr;
            auto found = producers.find(prerequisite.producer);
            if (!source || found == producers.end() ||
                (found->second != op && !dominance.properlyDominates(found->second, op))) { return failure(); }
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
