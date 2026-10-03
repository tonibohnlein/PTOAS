// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Borrowed links between shared phases and source MLIR. Control remains in the
// source regions; dominance is used only for SSA availability, not event order.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PHASEINDEX_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PHASEINDEX_H

#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::pto::frontiersynch {
enum class Boundary { Before, After };

class PhaseIndex {
public:
    // Function, SyncInput and records must outlive the index. Preserve original
    // payload anchors and region/control structure; rebuild if those change.
    // Inserting commands into existing blocks does not change phase membership.
    // Build validates every anchor and publishes no partial index on failure.
    LogicalResult build(func::FuncOp source, const SyncInput& input);
    ArrayRef<const CompoundInstanceElement*> phasesFor(Operation* anchor) const;

    // Outer-to-inner region choices for one invocation of each enclosing
    // RegionBranchOpInterface. Repeated invocations require occurrence identities.
    SmallVector<Region*> controlPath(const CompoundInstanceElement& phase) const;

    // A sequence within one invocation of a single-block SSA region. Nested
    // regions and multi-phase anchors require structured/target semantics and
    // produce failure, rather than being silently flattened into a trace.
    FailureOr<SmallVector<const CompoundInstanceElement*>> explicitSequence(Block& block) const;

    // SSA availability at an operation boundary. This says nothing about a
    // legal internal macro cut, pipe completion or synchronization visibility.
    bool valueAvailable(Value value, Operation* anchor, Boundary boundary) const;

private:
    bool contains(Operation* operation) const;
    func::FuncOp function;
    DominanceInfo dominance;
    DenseMap<Operation*, SmallVector<const CompoundInstanceElement*>> anchorPhases;
};
} // namespace mlir::pto::frontiersynch
#endif
