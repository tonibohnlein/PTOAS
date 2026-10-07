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
#include "PTO/Transforms/FrontierSynch/LifetimeScan.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::pto::frontiersynch {
enum class Boundary { Before, After };
struct ValuePrerequisite {
    const CompoundInstanceElement* producer = nullptr;
    Operation* consumer = nullptr;
    bool native = false;
};
struct PhasePrerequisiteEdges {
    std::vector<StorageGenerator> native, demands;
    std::string error;
};

class PhaseIndex {
public:
    // Function, SyncInput and records must outlive the index. Preserve original
    // payload anchors and region/control structure; rebuild if those change.
    // Phase membership survives insertion; rebuild prerequisite/availability
    // information whenever operations or their SSA uses change.
    // Build validates every anchor and publishes no partial index on failure.
    LogicalResult build(func::FuncOp source, const SyncInput& input);
    LogicalResult build(func::FuncOp source, ArrayRef<const CompoundInstanceElement*> phases);
    ArrayRef<const CompoundInstanceElement*> phasesFor(Operation* anchor) const;

    // Outer-to-inner region choices for one invocation of each enclosing
    // RegionBranchOpInterface. Repeated invocations require occurrence identities.
    SmallVector<Region*> controlPath(const CompoundInstanceElement& phase) const;

    // A sequence within one invocation of a single-block SSA region. Nested
    // regions and multi-phase anchors require structured/target semantics and
    // produce failure, rather than being silently flattened into a trace.
    FailureOr<SmallVector<const CompoundInstanceElement*>> explicitSequence(Block& block) const;

    // A payload result flows into a payload/control/interface operation. These
    // routes must account for its completion prerequisite, beyond storage.
    bool needsValuePrerequisite(Operation* operation) const;
    ArrayRef<ValuePrerequisite> prerequisitesFor(Operation* operation) const;
    // One occurrence per phase. External producers belong to composition.
    PhasePrerequisiteEdges mapPrerequisites(ArrayRef<const CompoundInstanceElement*> phases) const;

    // Backward relevance to payload operands, effects, control and returned
    // values. Computed once, including zero-trip and loop-carried SSA edges.
    bool isRelevant(Value value) const { return relevantValues.contains(value); }
    bool hasRelevantCarriedState(Operation* loop) const;
    bool hasRelevantResults(Operation* operation) const;

    // SSA availability at an operation boundary. This says nothing about a
    // legal internal macro cut, pipe completion or synchronization visibility.
    bool valueAvailable(Value value, Operation* anchor, Boundary boundary) const;

private:
    bool contains(Operation* operation) const;
    func::FuncOp function;
    DominanceInfo dominance;
    DenseMap<Operation*, SmallVector<const CompoundInstanceElement*>> anchorPhases;
    DenseSet<Operation*> valuePrerequisites;
    DenseSet<Value> relevantValues;
    void computeRelevance();
    DenseMap<Operation*, SmallVector<ValuePrerequisite>> prerequisites;
    void traceResult(const CompoundInstanceElement* producer, Value result,
                     DenseMap<Operation*, bool>& carriedRelevance);
};
} // namespace mlir::pto::frontiersynch
#endif
