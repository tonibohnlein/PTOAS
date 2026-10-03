// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

namespace mlir::pto::frontiersynch {
LogicalResult PhaseIndex::build(func::FuncOp source, const SyncInput& input)
{
    anchorPhases.clear();
    dominance.invalidate();
    function = {};
    if (!source) {
        return failure();
    }
    DenseMap<Operation*, SmallVector<const CompoundInstanceElement*>> pending;
    for (const auto* phase : input.instructions()) {
        Operation* anchor = phase->elementOp;
        if (!anchor || !anchor->getBlock() || !source->isProperAncestor(anchor)) {
            return source.emitError("shared synchronization phase has no anchor in this function");
        }
        pending[anchor].push_back(phase);
    }
    anchorPhases = std::move(pending);
    function = source;
    return success();
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
