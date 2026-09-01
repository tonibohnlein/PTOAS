// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

constexpr StringLiteral kGeneratedAttr = "pto.canonical_sync.generated";
constexpr StringLiteral kMechanismAttr = "pto.canonical_sync.mechanism";

struct EventAction {
  CanonicalMechanismId mechanism = kInvalidCanonicalSyncId;
  PIPE source = PIPE::PIPE_UNASSIGNED;
  PIPE target = PIPE::PIPE_UNASSIGNED;
  unsigned eventId = 0;
};

void tagGenerated(Operation *operation, OpBuilder &builder,
                  CanonicalMechanismId mechanism) {
  operation->setAttr(kGeneratedAttr, builder.getUnitAttr());
  operation->setAttr(kMechanismAttr, builder.getI32IntegerAttr(
                                         static_cast<int32_t>(mechanism)));
}

LogicalResult collectActions(
    const CanonicalSyncProgram &program, const IRMapping &mapping,
    DenseMap<Operation *, SmallVector<EventAction, 2>> &sets,
    DenseMap<Operation *, SmallVector<EventAction, 2>> &waits,
    DenseMap<Operation *, SmallVector<CanonicalMechanismId, 2>> &barriers) {
  const CanonicalSetCoverSolution &solution = *program.getSetCoverSolution();
  for (CanonicalMechanismId mechanismId : solution.mechanisms) {
    const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
    if (mechanism.kind == CanonicalMechanismKind::IntrinsicOrder ||
        mechanism.kind == CanonicalMechanismKind::FixedFence ||
        mechanism.kind == CanonicalMechanismKind::TailBarrier) {
      continue;
    }
    Operation *waitAnchor =
        mapping.lookupOrNull(mechanism.targetPoint.operation);
    if (!waitAnchor) {
      return program.getFunction().emitError(
          "canonical sync failed to map a materialization anchor into its "
          "clone");
    }
    if (mechanism.kind == CanonicalMechanismKind::PipeBarrier) {
      barriers[waitAnchor].push_back(mechanism.id);
      continue;
    }
    Operation *setAnchor =
        mapping.lookupOrNull(mechanism.sourcePoint.operation);
    const std::optional<unsigned> eventId = mechanism.eventId;
    if (!setAnchor || !eventId) {
      return program.getFunction().emitError(
          "canonical sync event is missing a cloned anchor or allocated ID");
    }
    EventAction action{mechanism.id, mechanism.source.pipe,
                       mechanism.target.pipe, *eventId};
    sets[setAnchor].push_back(action);
    waits[waitAnchor].push_back(action);
  }
  return success();
}

void sortActions(SmallVectorImpl<EventAction> &actions) {
  llvm::sort(actions, [](const EventAction &first, const EventAction &second) {
    return std::tie(first.source, first.target, first.eventId,
                    first.mechanism) < std::tie(second.source, second.target,
                                                second.eventId,
                                                second.mechanism);
  });
}

void emitActions(
    func::FuncOp clone,
    DenseMap<Operation *, SmallVector<EventAction, 2>> &sets,
    DenseMap<Operation *, SmallVector<EventAction, 2>> &waits,
    DenseMap<Operation *, SmallVector<CanonicalMechanismId, 2>> &barriers,
    const CanonicalSyncProgram &program) {
  OpBuilder builder(clone.getContext());
  for (auto &entry : sets) {
    sortActions(entry.second);
    builder.setInsertionPointAfter(entry.first);
    for (const EventAction &action : entry.second) {
      auto operation = builder.create<SetFlagOp>(
          entry.first->getLoc(),
          PipeAttr::get(clone.getContext(), action.source),
          PipeAttr::get(clone.getContext(), action.target),
          EventAttr::get(clone.getContext(),
                         static_cast<EVENT>(action.eventId)));
      tagGenerated(operation, builder, action.mechanism);
    }
  }
  for (auto &entry : waits) {
    sortActions(entry.second);
    builder.setInsertionPoint(entry.first);
    for (const EventAction &action : entry.second) {
      auto operation = builder.create<WaitFlagOp>(
          entry.first->getLoc(),
          PipeAttr::get(clone.getContext(), action.source),
          PipeAttr::get(clone.getContext(), action.target),
          EventAttr::get(clone.getContext(),
                         static_cast<EVENT>(action.eventId)));
      tagGenerated(operation, builder, action.mechanism);
    }
  }
  for (auto &entry : barriers) {
    llvm::sort(entry.second);
    builder.setInsertionPoint(entry.first);
    for (CanonicalMechanismId mechanismId : entry.second) {
      const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
      auto operation = builder.create<BarrierOp>(
          entry.first->getLoc(),
          PipeAttr::get(clone.getContext(), mechanism.source.pipe));
      tagGenerated(operation, builder, mechanismId);
    }
  }
}

void emitTailBarriers(func::FuncOp clone, const CanonicalSyncProgram &program) {
  CanonicalMechanismId tail = kInvalidCanonicalSyncId;
  const CanonicalSetCoverSolution &solution = *program.getSetCoverSolution();
  for (CanonicalMechanismId mechanismId : solution.mechanisms) {
    const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
    if (mechanism.kind == CanonicalMechanismKind::TailBarrier) {
      tail = mechanism.id;
      break;
    }
  }
  OpBuilder builder(clone.getContext());
  clone.walk([&](func::ReturnOp operation) {
    builder.setInsertionPoint(operation);
    auto barrier = builder.create<BarrierOp>(
        operation.getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
    tagGenerated(barrier, builder, tail);
    barrier->setAttr("pto.auto_sync_tail_barrier", builder.getUnitAttr());
    barrier->setAttr("pto.auto_sync_tail_hint",
                     builder.getStringAttr("barrier-all"));
  });
}

} // namespace

LogicalResult
mlir::pto::materializeAndVerifyCanonicalSync(CanonicalSyncProgram &program) {
  const bool planReady = program.isFrozen() && program.getSetCoverSolution();
  if (!planReady) {
    return program.getFunction().emitError(
        "canonical sync materialization requires a frozen verified plan");
  }
  func::FuncOp function = program.getFunction();
  IRMapping mapping;
  OwningOpRef<Operation *> clonedOperation(function->clone(mapping));
  func::FuncOp clone = cast<func::FuncOp>(*clonedOperation);
  DenseMap<Operation *, SmallVector<EventAction, 2>> sets;
  DenseMap<Operation *, SmallVector<EventAction, 2>> waits;
  DenseMap<Operation *, SmallVector<CanonicalMechanismId, 2>> barriers;
  if (failed(collectActions(program, mapping, sets, waits, barriers))) {
    return failure();
  }
  emitActions(clone, sets, waits, barriers, program);
  emitTailBarriers(clone, program);
  if (failed(verifyMaterializedCanonicalSync(clone))) {
    function.emitError("canonical sync rejected its staged materialization; "
                       "original IR is unchanged");
    return failure();
  }
  if (failed(mlir::verify(clone))) {
    function.emitError("canonical sync rejected its staged materialization; "
                       "original IR is unchanged");
    return failure();
  }
  function.getBody().takeBody(clone.getBody());
  return success();
}
