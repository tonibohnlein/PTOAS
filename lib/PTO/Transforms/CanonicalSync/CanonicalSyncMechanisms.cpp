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

#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

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
  return llvm::any_of(
      demand.iterationDistance, [](const CanonicalLoopDistance &distance) {
        return distance.relation == CanonicalIterationRelation::AnyPositive;
      });
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

bool guardImplies(ArrayRef<CanonicalControlAtom> execution,
                  ArrayRef<CanonicalControlAtom> required) {
  return llvm::all_of(required, [execution](const CanonicalControlAtom &atom) {
    return llvm::is_contained(execution, atom);
  });
}

bool executionLoopsImpliedByPhase(ArrayRef<CanonicalRegionId> executionLoops,
                                  const CanonicalPhase &phase) {
  return llvm::all_of(executionLoops, [&](CanonicalRegionId loop) {
    return llvm::is_contained(phase.loopPath, loop);
  });
}

const CanonicalFenceEffect *
findFixedRecurrenceFence(const CanonicalSyncProgram &program,
                         const CanonicalDemand &demand) {
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &target = program.getPhase(demand.target);
  if (source.resource.core != target.resource.core) {
    return nullptr;
  }
  const auto carrying = llvm::find_if(
      demand.iterationDistance, [](const CanonicalLoopDistance &distance) {
        return distance.relation == CanonicalIterationRelation::AnyPositive;
      });
  if (carrying == demand.iterationDistance.end()) {
    return nullptr;
  }
  for (const CanonicalFenceEffect &effect : program.getFenceEffects()) {
    if (!llvm::is_contained(effect.drainedResources, source.resource)) {
      continue;
    }
    const CanonicalProgramPoint point{effect.operation,
                                      CanonicalProgramPointPosition::After};
    if (!llvm::is_contained(effect.loopPath, carrying->loop)) {
      continue;
    }
    const bool completesAfterSource =
        executionLoopsImpliedByPhase(effect.loopPath, source) &&
        phaseMayPrecedePoint(source, point) &&
        guardImplies(demand.sourceGuard, effect.guard);
    const bool completesBeforeTarget =
        executionLoopsImpliedByPhase(effect.loopPath, target) &&
        pointMustPrecedePhase(point, target) &&
        guardImplies(demand.targetGuard, effect.guard);
    if (completesAfterSource || completesBeforeTarget) {
      return &effect;
    }
  }
  return nullptr;
}

CanonicalMechanism
buildFixedFenceMechanism(const CanonicalFenceEffect &effect) {
  CanonicalMechanism mechanism;
  mechanism.kind = CanonicalMechanismKind::FixedFence;
  mechanism.fenceEffect = effect.id;
  mechanism.sourcePoint = {effect.operation,
                           CanonicalProgramPointPosition::After};
  mechanism.targetPoint = mechanism.sourcePoint;
  mechanism.actionRegion = effect.region;
  mechanism.guard = effect.guard;
  return mechanism;
}

FailureOr<CanonicalMechanism>
buildTailMechanism(const CanonicalSyncProgram &program) {
  const std::optional<bool> vectorExecution =
      resolvePTOExecutionVector(program.getFunction());
  if (!vectorExecution) {
    program.getFunction().emitError(
        "canonical sync cannot resolve the physical core for the "
        "PIPE_ALL epilogue");
    return failure();
  }
  CanonicalMechanism tail;
  tail.kind = CanonicalMechanismKind::TailBarrier;
  tail.source = {*vectorExecution ? CanonicalCore::AIV : CanonicalCore::AIC,
                 PIPE::PIPE_ALL};
  tail.target = tail.source;
  tail.actionRegion = 0;
  return tail;
}

bool fenceCoversScope(FenceBarrierAllOp fence, FenceScope required) {
  const FenceScope actual = fence.getScope().getScope();
  return actual == FenceScope::All || actual == required;
}

Operation *findFixedVisibilityFence(Operation *source, Operation *target,
                                    FenceScope scope) {
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
    auto fence = dyn_cast<FenceBarrierAllOp>(current);
    if (fence && fenceCoversScope(fence, scope)) {
      return current;
    }
  }
  return nullptr;
}

bool isGmCmo(CmoCacheInvalidOp operation) {
  const AddressSpace space = operation.getSpace().getAddressSpace();
  return space == AddressSpace::GM || space == AddressSpace::Zero;
}

bool cmoCoversAccess(CmoCacheInvalidOp operation,
                     const CanonicalAccess &access) {
  if (!isGmCmo(operation)) {
    return false;
  }
  Value address = operation.getAddr();
  if (!address) {
    return true;
  }
  // SINGLE_CACHE_LINE covers only the line containing the CMO address. The
  // canonical model does not yet carry GM pointer alignment or cache-line
  // geometry, so only a one-byte access at that exact address is guaranteed
  // not to cross into another line. Wider accesses require whole-GM CMO.
  const bool provenSingleLineAccess =
      access.addressByteOffset && *access.addressByteOffset == 0 &&
      access.addressByteSize && *access.addressByteSize == 1;
  return address == access.value && provenSingleLineAccess;
}

Operation *findFixedCmo(Operation *begin, Operation *end,
                        const CanonicalAccess &access) {
  Operation *first = begin ? begin->getNextNode() : nullptr;
  for (Operation *current = first; current && current != end;
       current = current->getNextNode()) {
    if (auto cmo = dyn_cast<CmoCacheInvalidOp>(current);
        cmo && cmoCoversAccess(cmo, access)) {
      return current;
    }
  }
  return nullptr;
}

FailureOr<SmallVector<Operation *, 2>>
findFixedCacheMaintenance(const CanonicalSyncProgram &program,
                          const CanonicalDemand &demand, Operation *fence) {
  SmallVector<Operation *, 2> result;
  if (!demand.visibility ||
      demand.visibility->cacheMaintenance == CanonicalCacheMaintenance::None) {
    return result;
  }
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &target = program.getPhase(demand.target);
  Block *block = fence->getBlock();
  Operation *sourceAnchor = liftToBlock(source.operation, block);
  Operation *targetAnchor = liftToBlock(target.operation, block);
  const bool cleanSource = demand.visibility->cacheMaintenance ==
                           CanonicalCacheMaintenance::CleanSource;
  Operation *begin = cleanSource ? sourceAnchor : fence;
  Operation *end = cleanSource ? fence : targetAnchor;
  for (const CanonicalDemandCause &cause : demand.causes) {
    const CanonicalAccessId accessId =
        cleanSource ? cause.sourceAccess : cause.targetAccess;
    if (accessId == kInvalidCanonicalSyncId) {
      continue;
    }
    Operation *cmo = findFixedCmo(begin, end, program.getAccess(accessId));
    if (!cmo) {
      return failure();
    }
    if (!llvm::is_contained(result, cmo)) {
      result.push_back(cmo);
    }
  }
  return result;
}

bool sameMechanism(const CanonicalMechanism &left,
                   const CanonicalMechanism &right) {
  if (left.kind == CanonicalMechanismKind::FixedFence ||
      right.kind == CanonicalMechanismKind::FixedFence) {
    return left.kind == right.kind && left.fenceEffect == right.fenceEffect;
  }
  const bool sameSource = left.source == right.source;
  const bool sameTarget = left.target == right.target;
  if (left.kind != right.kind || !sameSource || !sameTarget ||
      left.guard != right.guard) {
    return false;
  }
  switch (left.kind) {
  case CanonicalMechanismKind::PipeBarrier:
    return left.targetPoint == right.targetPoint;
  case CanonicalMechanismKind::Event:
    return left.sourcePoint == right.sourcePoint &&
           left.targetPoint == right.targetPoint;
  case CanonicalMechanismKind::FixedFence:
    llvm_unreachable("fixed fences are compared by physical effect");
  case CanonicalMechanismKind::IntrinsicOrder:
  case CanonicalMechanismKind::TailBarrier:
    return true;
  }
  llvm_unreachable("unknown canonical mechanism kind");
}

CanonicalMechanismId internMechanism(CanonicalSyncProgram &program,
                                     CanonicalMechanism mechanism,
                                     CanonicalDemandId origin) {
  for (const CanonicalMechanism &existing : program.getMechanisms()) {
    if (sameMechanism(existing, mechanism)) {
      program.appendMechanismOrigin(existing.id, origin);
      program.appendMechanismCacheMaintenance(existing.id,
                                              mechanism.cacheMaintenance);
      return existing.id;
    }
  }
  mechanism.origins.push_back(origin);
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
  mechanism.sourcePoint = {setAfter, CanonicalProgramPointPosition::After};
  mechanism.targetPoint = {waitBefore, CanonicalProgramPointPosition::Before};
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
    FailureOr<CanonicalMechanism> tail = buildTailMechanism(program);
    if (failed(tail)) {
      return failure();
    }
    const CanonicalPhase &source = program.getPhase(demand.source);
    if (tail->source.core != source.resource.core) {
      source.operation->emitError(
          "canonical sync cannot use a function-core PIPE_ALL epilogue to "
          "complete an opposite-core section phase");
      return failure();
    }
    return tail;
  }
  const CanonicalPhase &sourcePhase = program.getPhase(demand.source);
  const CanonicalPhase &targetPhase = program.getPhase(demand.target);
  const CanonicalPhysicalResource source = sourcePhase.resource;
  const CanonicalPhysicalResource destination = targetPhase.resource;
  if (demand.requirement == CanonicalRequirement::Visibility) {
    if (!demand.visibility) {
      targetPhase.operation->emitError(
          "canonical sync visibility demand lacks explicit requirements");
      return failure();
    }
    if (demand.visibility->direction ==
        CanonicalVisibilityDirection::Mte3ToMte2Gm) {
      targetPhase.operation->emitError(
          "canonical sync has no device-proven MTE3-to-MTE2 GM visibility "
          "primitive")
          << "; demand d" << demand.id << " fails closed";
      return failure();
    }
    Operation *fence = findFixedVisibilityFence(
        sourcePhase.operation, targetPhase.operation, demand.visibility->scope);
    if (!fence) {
      targetPhase.operation->emitError(
          "canonical sync cannot satisfy a GM cache-visibility demand with "
          "SetFlag/WaitFlag")
          << "; demand d" << demand.id
          << " requires an existing explicit visibility fence";
      return failure();
    }
    FailureOr<SmallVector<Operation *, 2>> cacheMaintenance =
        findFixedCacheMaintenance(program, demand, fence);
    if (failed(cacheMaintenance)) {
      targetPhase.operation->emitError(
          "canonical sync visibility demand is missing required GM cache "
          "maintenance")
          << "; demand d" << demand.id << " requires "
          << stringifyCanonicalCacheMaintenance(
                 demand.visibility->cacheMaintenance);
      return failure();
    }
    auto effect = llvm::find_if(program.getFenceEffects(),
                                [fence](const CanonicalFenceEffect &item) {
                                  return item.operation == fence;
                                });
    if (effect == program.getFenceEffects().end()) {
      targetPhase.operation->emitError(
          "canonical sync visibility fence lacks a physical graph effect");
      return failure();
    }
    CanonicalMechanism mechanism = buildFixedFenceMechanism(*effect);
    mechanism.cacheMaintenance = std::move(*cacheMaintenance);
    return mechanism;
  }
  if (source == destination) {
    CanonicalMechanism mechanism;
    mechanism.source = source;
    mechanism.target = destination;
    mechanism.sourcePoint = {targetPhase.operation,
                             CanonicalProgramPointPosition::Before};
    mechanism.targetPoint = mechanism.sourcePoint;
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
  if (hasPositiveDistance(demand)) {
    if (const CanonicalFenceEffect *effect =
            findFixedRecurrenceFence(program, demand)) {
      return buildFixedFenceMechanism(*effect);
    }
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
  const bool invalidState = !program.isGraphFrozen() || program.isFrozen() ||
                            program.getSetCoverInstance().has_value() ||
                            program.buildingMechanisms ||
                            program.mechanismCatalogComplete;
  if (invalidState) {
    return program.getFunction().emitError(
        "canonical sync direct mechanisms require a frozen demand graph");
  }
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(program.getFunction());
  if (failed(target)) {
    return failure();
  }
  program.buildingMechanisms = true;
  const auto finishBuilding = llvm::make_scope_exit(
      [&program]() { program.buildingMechanisms = false; });
  for (const CanonicalFenceEffect &effect : program.getFenceEffects()) {
    CanonicalMechanism mechanism;
    mechanism.kind = CanonicalMechanismKind::FixedFence;
    mechanism.fenceEffect = effect.id;
    mechanism.sourcePoint = {effect.operation,
                             CanonicalProgramPointPosition::After};
    mechanism.targetPoint = mechanism.sourcePoint;
    mechanism.actionRegion = effect.region;
    mechanism.guard = effect.guard;
    program.appendMechanism(std::move(mechanism));
  }
  for (const CanonicalDemand &demand : program.getDemands()) {
    FailureOr<CanonicalMechanism> mechanism =
        buildDirectMechanism(program, *target, demand);
    if (failed(mechanism)) {
      return failure();
    }
    const CanonicalMechanismId id =
        internMechanism(program, std::move(*mechanism), demand.id);
    program.setDirectMechanism(demand.id, id);
  }
  if (program.getDemands().empty()) {
    if (resolvePTOExecutionVector(program.getFunction())) {
      FailureOr<CanonicalMechanism> tail = buildTailMechanism(program);
      if (failed(tail)) {
        return failure();
      }
      program.appendMechanism(std::move(*tail));
    }
  }
  program.mechanismCatalogComplete = true;
  return success();
}
