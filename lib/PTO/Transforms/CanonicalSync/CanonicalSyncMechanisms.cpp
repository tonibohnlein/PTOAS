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
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

Block *findCommonBlock(Operation *first, Operation *second) {
  SmallVector<Block *, 8> firstBlocks;
  for (Operation *current = first; current; current = current->getParentOp()) {
    if (current->getBlock()) {
      firstBlocks.push_back(current->getBlock());
    }
  }
  for (Operation *current = second; current; current = current->getParentOp()) {
    Block *currentBlock = current->getBlock();
    if (currentBlock && llvm::is_contained(firstBlocks, currentBlock)) {
      return currentBlock;
    }
  }
  return nullptr;
}

Operation *liftToBlock(Operation *operation, Block *block) {
  while (operation) {
    Block *operationBlock = operation->getBlock();
    if (operationBlock == block) {
      break;
    }
    operation = operation->getParentOp();
  }
  return operation;
}

bool isRepeatedBlock(Block *block) {
  Operation *blockParent = block ? block->getParentOp() : nullptr;
  for (Operation *parent = blockParent; parent;
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp>(parent)) {
      return true;
    }
  }
  return false;
}

bool hasPositiveDistance(const CanonicalDemand &demand) {
  return llvm::is_contained(demand.iterationDistance,
                            kCanonicalAnyPositiveDistance);
}

SmallVector<CanonicalControlAtom, 2>
commonGuard(ArrayRef<CanonicalControlAtom> first,
            ArrayRef<CanonicalControlAtom> second) {
  SmallVector<CanonicalControlAtom, 2> result;
  for (const CanonicalControlAtom &atom : first) {
    if (llvm::is_contained(second, atom)) {
      result.push_back(atom);
    }
  }
  return result;
}

Operation *findFixedVisibilityFence(Operation *source, Operation *target) {
  Block *block = findCommonBlock(source, target);
  Operation *sourceAnchor = liftToBlock(source, block);
  Operation *targetAnchor = liftToBlock(target, block);
  if (!block || !sourceAnchor || !targetAnchor ||
      sourceAnchor == targetAnchor ||
      !sourceAnchor->isBeforeInBlock(targetAnchor)) {
    return nullptr;
  }
  for (Operation *current = sourceAnchor->getNextNode();
       current && current != targetAnchor; current = current->getNextNode()) {
    if (isa<FenceBarrierAllOp>(current)) {
      return current;
    }
  }
  return nullptr;
}

bool sameMechanism(const CanonicalMechanism &left,
                   const CanonicalMechanism &right) {
  const bool sameSource = left.source == right.source;
  const bool sameTarget = left.target == right.target;
  if (left.kind != right.kind || !sameSource || !sameTarget ||
      left.guard != right.guard) {
    return false;
  }
  switch (left.kind) {
  case CanonicalMechanismKind::PipeBarrier:
    return left.waitBefore == right.waitBefore;
  case CanonicalMechanismKind::Event:
    return left.setAfter == right.setAfter &&
           left.waitBefore == right.waitBefore;
  case CanonicalMechanismKind::FixedFence:
    return left.waitBefore == right.waitBefore;
  case CanonicalMechanismKind::IntrinsicOrder:
  case CanonicalMechanismKind::TailBarrier:
    return true;
  }
  llvm_unreachable("unknown canonical mechanism kind");
}

CanonicalMechanismId internMechanism(CanonicalSyncProgram &program,
                                     CanonicalMechanism mechanism) {
  for (const CanonicalMechanism &existing : program.getMechanisms()) {
    if (sameMechanism(existing, mechanism)) {
      return existing.id;
    }
  }
  return program.appendMechanism(std::move(mechanism));
}

FailureOr<CanonicalMechanism> buildEventMechanism(
    const CanonicalSyncProgram &program, const CanonicalDemand &demand,
    CanonicalPhysicalResource source, CanonicalPhysicalResource target) {
  const CanonicalPhase &sourcePhase = program.getPhase(demand.source);
  const CanonicalPhase &targetPhase = program.getPhase(demand.target);
  Block *actionBlock =
      findCommonBlock(sourcePhase.operation, targetPhase.operation);
  Operation *setAfter = liftToBlock(sourcePhase.operation, actionBlock);
  Operation *waitBefore = liftToBlock(targetPhase.operation, actionBlock);
  const bool invalidOrder = !actionBlock || !setAfter || !waitBefore ||
                            setAfter == waitBefore ||
                            !setAfter->isBeforeInBlock(waitBefore);
  if (invalidOrder) {
    targetPhase.operation->emitError("canonical sync cannot place a matched "
                                     "event pair on this structured path")
        << "; demand d" << demand.id << " source p" << demand.source
        << " target p" << demand.target;
    return failure();
  }
  if (isRepeatedBlock(actionBlock)) {
    targetPhase.operation->emitError("canonical sync rejects an event whose "
                                     "set/wait lifecycle repeats inside a loop")
        << "; demand d" << demand.id
        << " requires a separately proven recurrence protocol";
    return failure();
  }
  CanonicalMechanism mechanism;
  mechanism.kind = CanonicalMechanismKind::Event;
  mechanism.source = source;
  mechanism.target = target;
  mechanism.sourceCut = demand.source;
  mechanism.targetCut = demand.target;
  mechanism.setAfter = setAfter;
  mechanism.waitBefore = waitBefore;
  mechanism.actionRegion =
      findRegionLca(program, sourcePhase.region, targetPhase.region);
  mechanism.guard =
      commonGuard(sourcePhase.controlPath, targetPhase.controlPath);
  return mechanism;
}

FailureOr<CanonicalMechanism>
buildDirectMechanism(const CanonicalSyncProgram &program,
                     const CanonicalSyncTarget &target,
                     const CanonicalDemand &demand) {
  if (demand.kind == CanonicalDemandKind::ExitCompletion) {
    CanonicalMechanism tail;
    tail.kind = CanonicalMechanismKind::TailBarrier;
    tail.source = {CanonicalCore::AIV, PIPE::PIPE_ALL};
    tail.target = tail.source;
    tail.sourceCut = kInvalidCanonicalSyncId;
    tail.targetCut = kInvalidCanonicalSyncId;
    tail.actionRegion = 0;
    return tail;
  }
  const CanonicalPhase &sourcePhase = program.getPhase(demand.source);
  const CanonicalPhase &targetPhase = program.getPhase(demand.target);
  const CanonicalPhysicalResource source = sourcePhase.resource;
  const CanonicalPhysicalResource destination = targetPhase.resource;
  if (demand.requirement == CanonicalRequirement::Visibility) {
    Operation *fence =
        findFixedVisibilityFence(sourcePhase.operation, targetPhase.operation);
    if (!fence) {
      targetPhase.operation->emitError(
          "canonical sync cannot satisfy a GM cache-visibility demand with "
          "SetFlag/WaitFlag")
          << "; demand d" << demand.id
          << " requires an existing explicit visibility fence";
      return failure();
    }
    CanonicalMechanism mechanism;
    mechanism.kind = CanonicalMechanismKind::FixedFence;
    mechanism.source = source;
    mechanism.target = destination;
    mechanism.sourceCut = demand.source;
    mechanism.targetCut = demand.target;
    mechanism.waitBefore = fence;
    mechanism.actionRegion = demand.owner;
    mechanism.guard = demand.guard;
    return mechanism;
  }
  if (source == destination) {
    CanonicalMechanism mechanism;
    mechanism.source = source;
    mechanism.target = destination;
    mechanism.sourceCut = demand.source;
    mechanism.targetCut = demand.target;
    mechanism.waitBefore = targetPhase.operation;
    mechanism.actionRegion = targetPhase.region;
    mechanism.guard = targetPhase.controlPath;
    if (target.hasIntrinsicCompletion(source)) {
      mechanism.kind = CanonicalMechanismKind::IntrinsicOrder;
      return mechanism;
    }
    if (!target.supportsPipeBarrier(source)) {
      targetPhase.operation->emitError("canonical sync target has no same-pipe "
                                       "completion mechanism for demand")
          << " d" << demand.id;
      return failure();
    }
    mechanism.kind = CanonicalMechanismKind::PipeBarrier;
    return mechanism;
  }
  if (source.core != destination.core) {
    targetPhase.operation->emitError(
        "canonical sync cannot satisfy a cross-core memory demand with an "
        "intra-core event")
        << "; demand d" << demand.id << " crosses "
        << stringifyCanonicalCore(source.core) << " to "
        << stringifyCanonicalCore(destination.core);
    return failure();
  }
  if (!target.supportsEvent(source, destination)) {
    targetPhase.operation->emitError(
        "canonical sync target table forbids the required directed event")
        << "; demand d" << demand.id << " requires "
        << stringifyPIPE(source.pipe) << " -> "
        << stringifyPIPE(destination.pipe);
    return failure();
  }
  if (hasPositiveDistance(demand)) {
    targetPhase.operation->emitError(
        "canonical sync rejects cross-pipe loop recurrence events in v1")
        << "; demand d" << demand.id << " has a positive iteration distance";
    return failure();
  }
  return buildEventMechanism(program, demand, source, destination);
}

} // namespace

LogicalResult
mlir::pto::buildCanonicalDirectMechanisms(CanonicalSyncProgram &program) {
  const bool invalidState = !program.isGraphFrozen() || program.isFrozen();
  if (invalidState) {
    return program.getFunction().emitError(
        "canonical sync direct mechanisms require a frozen demand graph");
  }
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(program.getFunction());
  if (failed(target)) {
    return failure();
  }
  for (const CanonicalDemand &demand : program.getDemands()) {
    FailureOr<CanonicalMechanism> mechanism =
        buildDirectMechanism(program, *target, demand);
    if (failed(mechanism)) {
      return failure();
    }
    const CanonicalMechanismId id =
        internMechanism(program, std::move(*mechanism));
    program.setDirectMechanism(demand.id, id);
  }
  if (program.getDemands().empty()) {
    CanonicalMechanism tail;
    tail.kind = CanonicalMechanismKind::TailBarrier;
    tail.source = {CanonicalCore::AIV, PIPE::PIPE_ALL};
    tail.target = tail.source;
    tail.sourceCut = kInvalidCanonicalSyncId;
    tail.targetCut = kInvalidCanonicalSyncId;
    tail.actionRegion = 0;
    program.appendMechanism(std::move(tail));
  }
  return success();
}
