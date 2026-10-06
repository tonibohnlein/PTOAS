// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

namespace mlir::pto::frontiersynch {
LogicalResult PhaseIndex::build(func::FuncOp source, const SyncInput& input)
{
    return build(source, input.instructions());
}

LogicalResult PhaseIndex::build(func::FuncOp source, ArrayRef<const CompoundInstanceElement*> phases)
{
    anchorPhases.clear();
    valuePrerequisites.clear();
    prerequisites.clear();
    dominance.invalidate();
    function = {};
    if (!source) {
        return failure();
    }
    DenseMap<Operation*, SmallVector<const CompoundInstanceElement*>> pending;
    for (const auto* phase : phases) {
        Operation* anchor = phase->elementOp;
        if (!anchor || !anchor->getBlock() || !source->isProperAncestor(anchor)) {
            return source.emitError("shared synchronization phase has no anchor in this function");
        }
        pending[anchor].push_back(phase);
    }
    anchorPhases = std::move(pending);
    function = source;
    for (const auto& entry : anchorPhases) {
        if (entry.second.size() != 1) {
            continue;
        }
        for (Value value : entry.first->getResults()) {
            traceResult(entry.second.front(), value);
        }
    }
    return success();
}

void PhaseIndex::traceResult(const CompoundInstanceElement* producer, Value result)
{
    const auto availability = resultAvailability(*producer, result);
    if (availability == SyncResultAvailability::NonScalar) {
        return;
    }
    const bool native = availability == SyncResultAvailability::SynchronousScalar;
    SmallVector<Value> work{result};
    DenseSet<Value> seen;
    auto record = [&](Operation* consumer) {
        auto& entries = prerequisites[consumer];
        if (llvm::none_of(entries, [producer](const auto& entry) { return entry.producer == producer; })) {
            entries.push_back({producer, consumer, native});
        }
    };
    while (!work.empty()) {
        Value value = work.pop_back_val();
        if (!seen.insert(value).second) {
            continue;
        }
        for (OpOperand& use : value.getUses()) {
            Operation* user = use.getOwner();
            if (!phasesFor(user).empty()) {
                record(user);
            } else if (auto branch = dyn_cast<scf::IfOp>(user)) {
                if (!native) {
                    valuePrerequisites.insert(user);
                }
                record(user);
                branch.walk([&](Operation* nested) {
                    if (!phasesFor(nested).empty()) {
                        record(nested);
                    }
                });
            } else if (isa<scf::YieldOp>(user) && isa<scf::IfOp>(user->getParentOp())) {
                work.push_back(user->getParentOp()->getResult(use.getOperandNumber()));
            } else if (auto loop = dyn_cast<scf::ForOp>(user); loop && use.getOperandNumber() < 3 && native) {
                // Native scalar bounds are available before loop execution.
                // They order body users without becoming storage demands.
                record(user);
                loop.walk([&](Operation* nested) {
                    if (!phasesFor(nested).empty()) { record(nested); }
                });
            } else if (user->getNumRegions() || isa<scf::YieldOp>(user)) {
                // A carried value needs an occurrence map, not a static SSA edge.
                valuePrerequisites.insert(user->getNumRegions() ? user : user->getParentOp());
            } else if (isa<func::ReturnOp>(user)) {
                if (!native) {
                    valuePrerequisites.insert(user);
                }
            } else if (isMemoryEffectFree(user)) {
                llvm::append_range(work, user->getResults());
            } else {
                valuePrerequisites.insert(user);
            }
        }
    }
}

ArrayRef<ValuePrerequisite> PhaseIndex::prerequisitesFor(Operation* operation) const
{
    auto found = prerequisites.find(operation);
    return found == prerequisites.end() ? ArrayRef<ValuePrerequisite>{} : found->second;
}

PhasePrerequisiteEdges PhaseIndex::mapPrerequisites(ArrayRef<const CompoundInstanceElement*> phases) const
{
    PhasePrerequisiteEdges output;
    DenseMap<const CompoundInstanceElement*, uint32_t> positions;
    if (phases.size() > UINT32_MAX) {
        output.error = "value prerequisite identity overflow";
        return output;
    }
    for (uint32_t i = 0; i < phases.size(); ++i) {
        if (!positions.try_emplace(phases[i], i).second) {
            output.error = "repeated scalar producer needs an occurrence mapping";
            return output;
        }
    }
    for (uint32_t target = 0; target < phases.size(); ++target) {
        for (const auto& requirement : prerequisitesFor(phases[target]->elementOp)) {
            auto source = positions.find(requirement.producer);
            if (source == positions.end()) {
                continue;
            }
            if (source->second >= target) {
                output.error = "value prerequisite needs a forward occurrence mapping";
                return output;
            }
            auto& edges = requirement.native ? output.native : output.demands;
            edges.push_back({source->second, target});
        }
    }
    return output;
}

bool PhaseIndex::needsValuePrerequisite(Operation* operation) const
{
    return valuePrerequisites.contains(operation);
}

bool PhaseIndex::contains(Operation* operation) const
{
    return function && operation && function->isProperAncestor(operation);
}

ArrayRef<const CompoundInstanceElement*> PhaseIndex::phasesFor(Operation* anchor) const
{
    auto found = anchorPhases.find(anchor);
    if (found == anchorPhases.end()) {
        return {};
    }
    return found->second;
}

SmallVector<Region*> PhaseIndex::controlPath(const CompoundInstanceElement& phase) const
{
    SmallVector<Region*> path;
    if (!contains(phase.elementOp) || !llvm::is_contained(phasesFor(phase.elementOp), &phase)) {
        return path;
    }
    for (Region* region = phase.elementOp->getParentRegion(); region != &function->getRegion(0);
         region = region->getParentRegion()) {
        if (isa<RegionBranchOpInterface>(region->getParentOp())) {
            path.push_back(region);
        }
    }
    std::reverse(path.begin(), path.end());
    return path;
}

FailureOr<SmallVector<const CompoundInstanceElement*>> PhaseIndex::explicitSequence(Block& block) const
{
    Region* region = block.getParent();
    if (!function || !region || !function->getRegion(0).isAncestor(region) || !region->hasOneBlock() ||
        !dominance.hasSSADominance(&block)) {
        return failure();
    }
    SmallVector<const CompoundInstanceElement*> sequence;
    for (Operation& operation : block) {
        auto phases = phasesFor(&operation);
        if (operation.getNumRegions() != 0 || phases.size() > 1) {
            return failure();
        }
        llvm::append_range(sequence, phases);
    }
    return sequence;
}

bool PhaseIndex::valueAvailable(Value value, Operation* anchor, Boundary boundary) const
{
    if (!contains(anchor) || !value || !value.getParentRegion() ||
        !function->getRegion(0).isAncestor(value.getParentRegion()) ||
        !dominance.hasSSADominance(anchor->getBlock())) {
        return false;
    }
    if (boundary == Boundary::After) {
        if (anchor->hasTrait<OpTrait::IsTerminator>()) {
            return false;
        }
        anchor = anchor->getNextNode();
    }
    return anchor && dominance.properlyDominates(value, anchor);
}
} // namespace mlir::pto::frontiersynch
