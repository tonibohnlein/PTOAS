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

#include "mlir/Dialect/SCF/IR/SCF.h"
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
constexpr StringLiteral kProtocolRoleAttr = "pto.canonical_sync.protocol_role";
constexpr StringLiteral kReleaseOwnerAttr = "pto.canonical_sync.release_owner";

struct EventAction {
  CanonicalMechanismId mechanism = kInvalidCanonicalSyncId;
  PIPE source = PIPE::PIPE_UNASSIGNED;
  PIPE target = PIPE::PIPE_UNASSIGNED;
  unsigned eventId = 0;
  bool crossCore = false;
};

struct RecurringEventAction {
  CanonicalMechanismId mechanism = kInvalidCanonicalSyncId;
  PIPE source = PIPE::PIPE_UNASSIGNED;
  PIPE target = PIPE::PIPE_UNASSIGNED;
  unsigned readyEventId = 0;
  unsigned releaseEventId = 0;
  Operation *loop = nullptr;
  Operation *sourceAnchor = nullptr;
  Operation *targetAnchor = nullptr;
  bool boundary = false;
};

void tagGenerated(Operation *operation, OpBuilder &builder,
                  CanonicalMechanismId mechanism) {
  operation->setAttr(kGeneratedAttr, builder.getUnitAttr());
  operation->setAttr(kMechanismAttr, builder.getI32IntegerAttr(
                                         static_cast<int32_t>(mechanism)));
}

void tagProtocol(Operation *operation, OpBuilder &builder,
                 CanonicalMechanismId mechanism, StringRef role) {
  tagGenerated(operation, builder, mechanism);
  operation->setAttr(kProtocolRoleAttr, builder.getStringAttr(role));
}

void tagReadyProtocol(Operation *operation, OpBuilder &builder,
                      CanonicalMechanismId mechanism,
                      CanonicalMechanismId releaseOwner, StringRef role) {
  tagProtocol(operation, builder, mechanism, role);
  operation->setAttr(
      kReleaseOwnerAttr,
      builder.getI32IntegerAttr(static_cast<int32_t>(releaseOwner)));
}

LogicalResult collectActions(
    const CanonicalSyncProgram &program, const IRMapping &mapping,
    DenseMap<Operation *, SmallVector<EventAction, 2>> &sets,
    DenseMap<Operation *, SmallVector<EventAction, 2>> &waits,
    DenseMap<Operation *, SmallVector<CanonicalMechanismId, 2>> &barriers,
    DenseMap<Operation *, SmallVector<CanonicalMechanismId, 2>>
        &visibilityFences,
    SmallVectorImpl<RecurringEventAction> &protocols) {
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
    if (mechanism.kind == CanonicalMechanismKind::VisibilityFence) {
      visibilityFences[waitAnchor].push_back(mechanism.id);
      continue;
    }
    Operation *setAnchor =
        mapping.lookupOrNull(mechanism.sourcePoint.operation);
    const std::optional<unsigned> eventId = mechanism.eventId;
    if (!setAnchor || !eventId) {
      return program.getFunction().emitError(
          "canonical sync event is missing a cloned anchor or allocated ID");
    }
    if (mechanism.kind == CanonicalMechanismKind::RecurringEvent) {
      if (!mechanism.recurrenceLoop || !mechanism.releaseEventId) {
        return program.getFunction().emitError(
            "canonical sync recurring event lacks a loop or release ID");
      }
      Operation *loop = mapping.lookupOrNull(
          program.getRegion(*mechanism.recurrenceLoop).operation);
      if (!loop || !isa<scf::ForOp>(loop)) {
        return program.getFunction().emitError(
            "canonical sync failed to map a recurring event loop");
      }
      protocols.push_back({mechanism.id, mechanism.source.pipe,
                           mechanism.target.pipe, *eventId,
                           *mechanism.releaseEventId, loop, setAnchor,
                           waitAnchor, mechanism.boundaryRecurring});
      continue;
    }
    EventAction action{
        mechanism.id, mechanism.source.pipe, mechanism.target.pipe, *eventId,
        mechanism.kind == CanonicalMechanismKind::CrossCoreEvent};
    sets[setAnchor].push_back(action);
    waits[waitAnchor].push_back(action);
  }
  return success();
}

template <typename OpTy>
Operation *createProtocolEvent(OpBuilder &builder, Location location,
                               func::FuncOp clone, PIPE source, PIPE target,
                               unsigned eventId, CanonicalMechanismId mechanism,
                               StringRef role) {
  auto operation = builder.create<OpTy>(
      location, PipeAttr::get(clone.getContext(), source),
      PipeAttr::get(clone.getContext(), target),
      EventAttr::get(clone.getContext(), static_cast<EVENT>(eventId)));
  tagProtocol(operation, builder, mechanism, role);
  return operation.getOperation();
}

void emitRecurringProtocols(func::FuncOp clone,
                            SmallVectorImpl<RecurringEventAction> &protocols) {
  llvm::sort(protocols, [](const RecurringEventAction &first,
                           const RecurringEventAction &second) {
    return first.mechanism < second.mechanism;
  });
  OpBuilder builder(clone.getContext());
  for (auto [index, protocol] : llvm::enumerate(protocols)) {
    const ArrayRef<RecurringEventAction> preceding =
        ArrayRef<RecurringEventAction>(protocols).take_front(index);
    const auto precedingOwner =
        llvm::find_if(preceding, [&](const RecurringEventAction &candidate) {
          return candidate.loop == protocol.loop &&
                 candidate.source == protocol.source &&
                 candidate.target == protocol.target &&
                 candidate.releaseEventId == protocol.releaseEventId;
        });
    const RecurringEventAction &releaseOwner =
        precedingOwner == preceding.end() ? protocol : *precedingOwner;
    const bool ownsRelease = releaseOwner.mechanism == protocol.mechanism;
    const Location location = protocol.loop->getLoc();
    auto loop = cast<scf::ForOp>(protocol.loop);
    if (ownsRelease) {
      builder.setInsertionPoint(protocol.loop);
      createProtocolEvent<SetFlagOp>(builder, location, clone, protocol.target,
                                     protocol.source, protocol.releaseEventId,
                                     protocol.mechanism, "release-prime-set");
    }
    if (protocol.boundary) {
      builder.setInsertionPoint(protocol.loop);
      Operation *readyPrime = createProtocolEvent<SetFlagOp>(
          builder, location, clone, protocol.source, protocol.target,
          protocol.readyEventId, protocol.mechanism, "ready-prime-set");
      tagReadyProtocol(readyPrime, builder, protocol.mechanism,
                       releaseOwner.mechanism, "ready-prime-set");
    }

    // The reverse release channel is the loop-carried ownership token.  Its
    // wait and set live at the loop header/latch so they cover reuse across
    // different choice arms.  The forward ready pair remains at the precise
    // branch-local producer/consumer cuts.
    if (ownsRelease) {
      builder.setInsertionPointToStart(loop.getBody());
      createProtocolEvent<WaitFlagOp>(builder, location, clone, protocol.target,
                                      protocol.source, protocol.releaseEventId,
                                      protocol.mechanism, "release-body-wait");
    }
    if (protocol.boundary) {
      builder.setInsertionPointToStart(loop.getBody());
      Operation *readyWait = createProtocolEvent<WaitFlagOp>(
          builder, location, clone, protocol.source, protocol.target,
          protocol.readyEventId, protocol.mechanism, "ready-body-wait");
      tagReadyProtocol(readyWait, builder, protocol.mechanism,
                       releaseOwner.mechanism, "ready-body-wait");

      builder.setInsertionPoint(loop.getBody()->getTerminator());
      Operation *readySet = createProtocolEvent<SetFlagOp>(
          builder, location, clone, protocol.source, protocol.target,
          protocol.readyEventId, protocol.mechanism, "ready-body-set");
      tagReadyProtocol(readySet, builder, protocol.mechanism,
                       releaseOwner.mechanism, "ready-body-set");
    } else {
      builder.setInsertionPointAfter(protocol.sourceAnchor);
      Operation *readySet = createProtocolEvent<SetFlagOp>(
          builder, location, clone, protocol.source, protocol.target,
          protocol.readyEventId, protocol.mechanism, "ready-body-set");
      tagReadyProtocol(readySet, builder, protocol.mechanism,
                       releaseOwner.mechanism, "ready-body-set");

      builder.setInsertionPoint(protocol.targetAnchor);
      Operation *readyWait = createProtocolEvent<WaitFlagOp>(
          builder, location, clone, protocol.source, protocol.target,
          protocol.readyEventId, protocol.mechanism, "ready-body-wait");
      tagReadyProtocol(readyWait, builder, protocol.mechanism,
                       releaseOwner.mechanism, "ready-body-wait");
    }
    if (ownsRelease) {
      builder.setInsertionPoint(loop.getBody()->getTerminator());
      createProtocolEvent<SetFlagOp>(builder, location, clone, protocol.target,
                                     protocol.source, protocol.releaseEventId,
                                     protocol.mechanism, "release-body-set");
    }

    if (ownsRelease) {
      builder.setInsertionPointAfter(protocol.loop);
      createProtocolEvent<WaitFlagOp>(builder, location, clone, protocol.target,
                                      protocol.source, protocol.releaseEventId,
                                      protocol.mechanism, "release-drain-wait");
    }
    if (protocol.boundary) {
      builder.setInsertionPointAfter(protocol.loop);
      Operation *readyDrain = createProtocolEvent<WaitFlagOp>(
          builder, location, clone, protocol.source, protocol.target,
          protocol.readyEventId, protocol.mechanism, "ready-drain-wait");
      tagReadyProtocol(readyDrain, builder, protocol.mechanism,
                       releaseOwner.mechanism, "ready-drain-wait");
    }
  }
}

void sortActions(SmallVectorImpl<EventAction> &actions) {
  llvm::sort(actions, [](const EventAction &first, const EventAction &second) {
    return std::tie(first.source, first.target, first.eventId, first.crossCore,
                    first.mechanism) <
           std::tie(second.source, second.target, second.eventId,
                    second.crossCore, second.mechanism);
  });
}

void emitActions(
    func::FuncOp clone,
    DenseMap<Operation *, SmallVector<EventAction, 2>> &sets,
    DenseMap<Operation *, SmallVector<EventAction, 2>> &waits,
    DenseMap<Operation *, SmallVector<CanonicalMechanismId, 2>> &barriers,
    DenseMap<Operation *, SmallVector<CanonicalMechanismId, 2>>
        &visibilityFences,
    const CanonicalSyncProgram &program) {
  OpBuilder builder(clone.getContext());
  for (auto &entry : sets) {
    sortActions(entry.second);
    builder.setInsertionPointAfter(entry.first);
    for (const EventAction &action : entry.second) {
      if (action.crossCore) {
        auto operation = builder.create<SyncSetOp>(
            entry.first->getLoc(),
            PipeAttr::get(clone.getContext(), action.source),
            builder.getI32IntegerAttr(action.eventId),
            builder.getI32IntegerAttr(2), Value());
        tagGenerated(operation, builder, action.mechanism);
        continue;
      }
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
      if (action.crossCore) {
        auto operation = builder.create<SyncWaitOp>(
            entry.first->getLoc(),
            PipeAttr::get(clone.getContext(), action.target),
            builder.getI32IntegerAttr(action.eventId),
            builder.getI32IntegerAttr(2), Value());
        tagGenerated(operation, builder, action.mechanism);
        continue;
      }
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
  for (auto &entry : visibilityFences) {
    llvm::sort(entry.second);
    builder.setInsertionPoint(entry.first);
    // All visibility mechanisms at one physical cut share the same GM fence.
    // Source-side cleans must precede publication, and target-side
    // invalidations must follow it. Emitting each mechanism as an independent
    // CMO/fence pair can put a later fence after an acquire invalidation and
    // thereby invalidate that acquire protocol.
    for (CanonicalMechanismId mechanismId : entry.second) {
      const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
      if (*mechanism.generatedCacheMaintenance ==
          CanonicalCacheMaintenance::CleanSource) {
        auto operation = builder.create<CmoCacheInvalidOp>(
            entry.first->getLoc(), Value(),
            AddressSpaceAttr::get(clone.getContext(), AddressSpace::GM));
        tagGenerated(operation, builder, mechanismId);
      }
    }
    auto fence = builder.create<FenceBarrierAllOp>(
        entry.first->getLoc(),
        FenceScopeAttr::get(clone.getContext(), FenceScope::GM));
    tagGenerated(fence, builder, entry.second.front());
    for (CanonicalMechanismId mechanismId : entry.second) {
      const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
      if (*mechanism.generatedCacheMaintenance ==
          CanonicalCacheMaintenance::InvalidateTarget) {
        auto operation = builder.create<CmoCacheInvalidOp>(
            entry.first->getLoc(), Value(),
            AddressSpaceAttr::get(clone.getContext(), AddressSpace::GM));
        tagGenerated(operation, builder, mechanismId);
      }
    }
  }
}

void tagTailBarrier(BarrierOp barrier, OpBuilder &builder,
                    CanonicalMechanismId mechanism) {
  tagGenerated(barrier, builder, mechanism);
  barrier->setAttr("pto.auto_sync_tail_barrier", builder.getUnitAttr());
  barrier->setAttr("pto.auto_sync_tail_hint",
                   builder.getStringAttr("barrier-all"));
}

LogicalResult emitTailBarriers(func::FuncOp clone,
                               const CanonicalSyncProgram &program,
                               const IRMapping &mapping) {
  CanonicalMechanismId functionTail = kInvalidCanonicalSyncId;
  const CanonicalSetCoverSolution &solution = *program.getSetCoverSolution();
  OpBuilder builder(clone.getContext());
  for (CanonicalMechanismId mechanismId : solution.mechanisms) {
    const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
    if (mechanism.kind != CanonicalMechanismKind::TailBarrier) {
      continue;
    }
    if (!mechanism.targetPoint.operation) {
      functionTail = mechanism.id;
      continue;
    }
    Operation *anchor = mapping.lookupOrNull(mechanism.targetPoint.operation);
    if (!anchor || mechanism.targetPoint.position !=
                       CanonicalProgramPointPosition::After) {
      return clone.emitError(
          "canonical sync failed to map a section tail barrier anchor");
    }
    builder.setInsertionPointToEnd(anchor->getBlock());
    auto barrier = builder.create<BarrierOp>(
        anchor->getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
    // The auto_sync_tail attribute is a code-generation request to coalesce a
    // barrier into the function return helper. A section-local drain must stay
    // at its physical section boundary instead.
    tagGenerated(barrier, builder, mechanism.id);
  }
  if (functionTail == kInvalidCanonicalSyncId) {
    return success();
  }
  clone.walk([&](func::ReturnOp operation) {
    builder.setInsertionPoint(operation);
    auto barrier = builder.create<BarrierOp>(
        operation.getLoc(), PipeAttr::get(clone.getContext(), PIPE::PIPE_ALL));
    tagTailBarrier(barrier, builder, functionTail);
  });
  return success();
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
  // Verify the candidate inside a scratch symbol table. A detached function
  // cannot resolve the private external declarations that define exact
  // normalized runtime-adapter contracts, which would make the independent
  // verifier reject a contract already validated on the original IR.
  OwningOpRef<ModuleOp> stagingModule = ModuleOp::create(function.getLoc());
  func::FuncOp clone = cast<func::FuncOp>(function->clone(mapping));
  stagingModule->push_back(clone);
  if (ModuleOp sourceModule = function->getParentOfType<ModuleOp>()) {
    for (func::FuncOp declaration : sourceModule.getOps<func::FuncOp>()) {
      if (declaration != function && declaration.isDeclaration()) {
        stagingModule->push_back(cast<func::FuncOp>(declaration->clone()));
      }
    }
  }
  DenseMap<Operation *, SmallVector<EventAction, 2>> sets;
  DenseMap<Operation *, SmallVector<EventAction, 2>> waits;
  DenseMap<Operation *, SmallVector<CanonicalMechanismId, 2>> barriers;
  DenseMap<Operation *, SmallVector<CanonicalMechanismId, 2>> visibilityFences;
  SmallVector<RecurringEventAction, 2> protocols;
  if (failed(collectActions(program, mapping, sets, waits, barriers,
                            visibilityFences, protocols))) {
    return failure();
  }
  emitActions(clone, sets, waits, barriers, visibilityFences, program);
  emitRecurringProtocols(clone, protocols);
  if (failed(emitTailBarriers(clone, program, mapping))) {
    return failure();
  }
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
