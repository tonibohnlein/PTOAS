// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

PipeAttr getPipeAttr(Builder &builder, PipelineType pipe) {
  return PipeAttr::get(builder.getContext(), static_cast<PIPE>(pipe));
}

EventAttr getEventAttr(Builder &builder, unsigned eventId) {
  return EventAttr::get(builder.getContext(), static_cast<EVENT>(eventId));
}

void setInsertionPoint(IRRewriter &rewriter, const CanonicalAnchor &anchor) {
  if (anchor.before || anchor.operation->hasTrait<OpTrait::IsTerminator>()) {
    rewriter.setInsertionPoint(anchor.operation);
  } else {
    rewriter.setInsertionPointAfter(anchor.operation);
  }
}

void markGenerated(Operation *operation, Builder &builder) {
  operation->setAttr("pto.canonical_sync", builder.getUnitAttr());
}

void createStaticFlag(IRRewriter &rewriter, Location location,
                      PipelineType source, PipelineType target,
                      unsigned eventId, bool set) {
  PipeAttr sourceAttr = getPipeAttr(rewriter, source);
  PipeAttr targetAttr = getPipeAttr(rewriter, target);
  EventAttr eventAttr = getEventAttr(rewriter, eventId);
  Operation *created =
      set ? rewriter
                .create<SetFlagOp>(location, sourceAttr, targetAttr, eventAttr)
                .getOperation()
          : rewriter
                .create<WaitFlagOp>(location, sourceAttr, targetAttr, eventAttr)
                .getOperation();
  markGenerated(created, rewriter);
}

Value selectDynamicEventId(IRRewriter &rewriter, Location location, Value slot,
                           ArrayRef<unsigned> eventIds) {
  Value indexSlot = slot;
  if (!indexSlot.getType().isIndex()) {
    indexSlot = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getIndexType(), indexSlot);
  }
  Value width =
      rewriter.create<arith::ConstantIndexOp>(location, eventIds.size());
  Value slotModulo =
      rewriter.create<arith::RemUIOp>(location, indexSlot, width);
  Value selected =
      rewriter.create<arith::ConstantIndexOp>(location, eventIds.front());
  for (std::size_t lane = 1; lane < eventIds.size(); ++lane) {
    Value laneValue = rewriter.create<arith::ConstantIndexOp>(location, lane);
    Value matches = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::eq, slotModulo, laneValue);
    Value event =
        rewriter.create<arith::ConstantIndexOp>(location, eventIds[lane]);
    selected =
        rewriter.create<arith::SelectOp>(location, matches, event, selected);
  }
  return selected;
}

void createDynamicFlag(IRRewriter &rewriter, Location location,
                       PipelineType source, PipelineType target, Value slot,
                       ArrayRef<unsigned> eventIds, bool set) {
  PipeAttr sourceAttr = getPipeAttr(rewriter, source);
  PipeAttr targetAttr = getPipeAttr(rewriter, target);
  Value selected = selectDynamicEventId(rewriter, location, slot, eventIds);
  Operation *created = set ? rewriter
                                 .create<SetFlagDynOp>(location, sourceAttr,
                                                       targetAttr, selected)
                                 .getOperation()
                           : rewriter
                                 .create<WaitFlagDynOp>(location, sourceAttr,
                                                        targetAttr, selected)
                                 .getOperation();
  markGenerated(created, rewriter);
}

void emitEventAction(IRRewriter &rewriter, const CanonicalEvent &event,
                     const CanonicalEventAction &action) {
  setInsertionPoint(rewriter, action.anchor);
  const bool set = action.kind == CanonicalEventActionKind::Set;
  const Location location = action.anchor.operation->getLoc();
  if (action.nonEmptyLoopGuard) {
    auto loop = cast<scf::ForOp>(action.nonEmptyLoopGuard);
    Value condition = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::slt, loop.getLowerBound(),
        loop.getUpperBound());
    auto guard = rewriter.create<scf::IfOp>(
        location, TypeRange{}, condition, /*withElseRegion=*/false);
    markGenerated(guard, rewriter);
    rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
  }
  if (action.lane.kind == CanonicalEventLaneKind::All) {
    for (unsigned eventId : event.eventIds) {
      createStaticFlag(rewriter, location, event.sourcePipe, event.targetPipe,
                       eventId, set);
    }
    return;
  }
  if (action.lane.kind == CanonicalEventLaneKind::Dynamic) {
    createDynamicFlag(rewriter, location, event.sourcePipe, event.targetPipe,
                      action.lane.selector, event.eventIds, set);
    return;
  }
  createStaticFlag(rewriter, location, event.sourcePipe, event.targetPipe,
                   event.eventIds[action.lane.index], set);
}

} // namespace

LogicalResult mlir::pto::emitCanonicalSyncPlan(func::FuncOp func,
                                               const CanonicalSyncPlan &plan) {
  if (plan.getNodes().empty()) {
    return success();
  }
  IRRewriter rewriter(func.getContext());
  for (const CanonicalBarrier &barrier : plan.getBarriers()) {
    setInsertionPoint(rewriter, barrier.anchor);
    auto created =
        rewriter.create<BarrierOp>(barrier.anchor.operation->getLoc(),
                                   getPipeAttr(rewriter, barrier.pipe));
    markGenerated(created.getOperation(), rewriter);
  }
  for (const CanonicalEvent &event : plan.getEvents()) {
    for (const CanonicalEventAction &action : event.actions) {
      emitEventAction(rewriter, event, action);
    }
  }
  SmallVector<func::ReturnOp, 4> returns;
  func.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
  for (func::ReturnOp returnOp : returns) {
    rewriter.setInsertionPoint(returnOp);
    auto barrier = rewriter.create<BarrierOp>(
        returnOp.getLoc(), getPipeAttr(rewriter, PipelineType::PIPE_ALL));
    barrier->setAttr("pto.auto_sync_tail_barrier", rewriter.getUnitAttr());
    barrier->setAttr("pto.canonical_sync", rewriter.getUnitAttr());
    if (auto hint =
            func->getAttrOfType<StringAttr>("pto.auto_sync_tail_hint")) {
      barrier->setAttr("pto.auto_sync_tail_hint", hint);
    }
  }
  return success();
}
