// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve branch predicates and arm identity in a shared guard DAG. A guarded
// loop requires predicates defined before that loop, so participation repeats.
#include "RecognitionInternal.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::pto::frontiersynch {
namespace {
void collect(Region& root, Operation* entry, const PhaseIndex& index, GuardedRecognition& output,
             bool requireInvariant)
{
    // An explicit stack avoids host recursion on deeply nested if/else trees.
    struct Work {
        Operation* operation = nullptr;
        std::optional<std::size_t> guard;
    };
    SmallVector<Work> stack;
    auto append = [&](Region& region, std::optional<std::size_t> guard) {
        if (region.empty()) {
            return;
        }
        if (!region.hasOneBlock()) {
            output.result.note(RecognitionIssue::UnsupportedControl, region.getParentOp(), true);
            return;
        }
        for (Operation& op : llvm::reverse(region.front())) {
            stack.push_back({&op, guard});
        }
    };
    append(root, std::nullopt);
    while (!stack.empty()) {
        const auto work = stack.pop_back_val();
        Operation& op = *work.operation;
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
            const bool available = entry && index.valueAvailable(branch.getCondition(), entry, Boundary::Before);
            output.entryGuardsAvailable &= available;
            if (requireInvariant && !available) {
                output.result.note(RecognitionIssue::GuardInvariance, &op);
            }
            const auto thenGuard = output.guards.size();
            output.guards.push_back({work.guard, branch.getCondition(), true});
            const auto elseGuard = output.guards.size();
            output.guards.push_back({work.guard, branch.getCondition(), false});
            append(branch.getElseRegion(), elseGuard);
            append(branch.getThenRegion(), thenGuard);
            continue;
        }
        if (op.getNumRegions()) {
            output.result.note(RecognitionIssue::UnsupportedControl, &op, true);
            continue;
        }
        detail::inspectLeaf(op, index, output.result);
        for (const auto* phase : index.phasesFor(&op)) {
            output.phases.push_back({phase, work.guard});
        }
    }
}
} // namespace

GuardedRecognition recognizeFiniteGuarded(Region& region, const PhaseIndex& index,
                                          const SyncStorageEffects& effects)
{
    GuardedRecognition output;
    Operation* entry = region.hasOneBlock() && !region.front().empty() ? &region.front().front() : nullptr;
    collect(region, entry, index, output, false);
    for (const auto& item : output.phases) {
        for (auto id : effects.effectsFor(item.phase)) {
            if (effects.effects()[id].precision != SyncAccessPrecision::Exact) {
                output.result.note(RecognitionIssue::InexactFootprint, item.phase->elementOp);
            }
        }
    }
    return output;
}

GuardedRecognition recognizeGuardedRotating(scf::ForOp loop, const PhaseIndex& index,
                                            const SyncInput& input, const SyncStorageEffects& effects)
{
    GuardedRecognition output;
    if (!detail::checkRotatingDomain(loop, output.result)) {
        return output;
    }
    collect(loop.getRegion(), loop, index, output, true);
    SmallVector<const CompoundInstanceElement*> phases;
    DenseMap<const CompoundInstanceElement*, std::optional<std::size_t>> guards;
    for (const auto& item : output.phases) {
        phases.push_back(item.phase);
        guards[item.phase] = item.guard;
    }
    detail::inspectRotatingPhases(loop, phases, input, effects, output.result);
    for (auto& access : output.result.accesses) {
        access.guard = guards.lookup(effects.effects()[access.effect].phase);
    }
    return output;
}
} // namespace mlir::pto::frontiersynch
