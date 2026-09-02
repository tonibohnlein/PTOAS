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
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <array>
#include <iterator>
#include <unordered_map>

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

std::optional<CanonicalRegionId>
findLoopRegion(const CanonicalSyncProgram &program, scf::ForOp loop) {
  auto found =
      llvm::find_if(program.getRegions(), [&](const CanonicalRegion &region) {
        return region.kind == CanonicalRegionKind::Loop &&
               region.operation == loop.getOperation();
      });
  return found == program.getRegions().end()
             ? std::nullopt
             : std::optional<CanonicalRegionId>(found->id);
}

bool hasEnclosingLoop(const CanonicalSyncProgram &program,
                      CanonicalRegionId region) {
  for (CanonicalRegionId current = program.getRegion(region).parent;
       current != kInvalidCanonicalSyncId;
       current = program.getRegion(current).parent) {
    if (program.getRegion(current).kind == CanonicalRegionKind::Loop) {
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

std::optional<CanonicalRegionId> carryingLoop(const CanonicalDemand &demand) {
  auto found = llvm::find_if(
      demand.iterationDistance, [](const CanonicalLoopDistance &distance) {
        return distance.relation == CanonicalIterationRelation::AnyPositive;
      });
  return found == demand.iterationDistance.end()
             ? std::nullopt
             : std::optional<CanonicalRegionId>(found->loop);
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

Operation *findEnclosingPhysicalSection(Operation *operation) {
  Operation *parent = operation ? operation->getParentOp() : nullptr;
  while (parent) {
    if (isa<SectionCubeOp, SectionVectorOp>(parent)) {
      return parent;
    }
    parent = parent->getParentOp();
  }
  return nullptr;
}

FailureOr<CanonicalMechanism>
buildTailMechanism(const CanonicalSyncProgram &program,
                   const CanonicalPhase &sourcePhase) {
  if (Operation *section =
          findEnclosingPhysicalSection(sourcePhase.operation)) {
    auto sectionRegion =
        llvm::find_if(program.getRegions(), [&](const CanonicalRegion &region) {
          return region.kind == CanonicalRegionKind::Transparent &&
                 region.operation == section;
        });
    if (sectionRegion == program.getRegions().end()) {
      return section->emitError(
          "canonical sync cannot resolve a physical section region");
    }
    CanonicalMechanism tail;
    tail.kind = CanonicalMechanismKind::TailBarrier;
    tail.source = {sourcePhase.resource.core, PIPE::PIPE_ALL};
    tail.target = tail.source;
    Operation *last = &section->getRegion(0).front().back();
    tail.sourcePoint = {last, CanonicalProgramPointPosition::After};
    tail.targetPoint = tail.sourcePoint;
    tail.actionRegion = sectionRegion->id;
    for (const CanonicalControlAtom &atom : sourcePhase.controlPath) {
      Operation *choice = program.getRegion(atom.choice).operation;
      if (!section->isProperAncestor(choice)) {
        tail.guard.push_back(atom);
      }
    }
    return tail;
  }
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
  if (tail.source.core != sourcePhase.resource.core) {
    sourcePhase.operation->emitError(
        "canonical sync cannot place a PIPE_ALL exit barrier for this "
        "physical core");
    return failure();
  }
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
  case CanonicalMechanismKind::CrossCoreEvent:
    return left.sourcePoint == right.sourcePoint &&
           left.targetPoint == right.targetPoint;
  case CanonicalMechanismKind::RecurringEvent:
    return left.sourcePoint == right.sourcePoint &&
           left.targetPoint == right.targetPoint &&
           left.recurrenceLoop == right.recurrenceLoop &&
           left.boundaryRecurring == right.boundaryRecurring;
  case CanonicalMechanismKind::VisibilityFence:
    return left.targetPoint == right.targetPoint &&
           left.generatedCacheMaintenance == right.generatedCacheMaintenance;
  case CanonicalMechanismKind::FixedFence:
    llvm_unreachable("fixed fences are compared by physical effect");
  case CanonicalMechanismKind::IntrinsicOrder:
    return true;
  case CanonicalMechanismKind::TailBarrier:
    return left.sourcePoint == right.sourcePoint &&
           left.targetPoint == right.targetPoint;
  }
  llvm_unreachable("unknown canonical mechanism kind");
}

std::size_t hashMechanism(const CanonicalMechanism &mechanism) {
  if (mechanism.kind == CanonicalMechanismKind::FixedFence) {
    return static_cast<std::size_t>(llvm::hash_combine(
        static_cast<unsigned>(mechanism.kind),
        mechanism.fenceEffect.value_or(kInvalidCanonicalSyncId)));
  }
  llvm::hash_code hash = llvm::hash_combine(
      static_cast<unsigned>(mechanism.kind),
      static_cast<unsigned>(mechanism.source.core),
      static_cast<unsigned>(mechanism.source.pipe),
      static_cast<unsigned>(mechanism.target.core),
      static_cast<unsigned>(mechanism.target.pipe));
  for (const CanonicalControlAtom &atom : mechanism.guard) {
    hash = llvm::hash_combine(hash, atom.choice, atom.arm);
  }
  switch (mechanism.kind) {
  case CanonicalMechanismKind::PipeBarrier:
    hash = llvm::hash_combine(
        hash, mechanism.targetPoint.operation,
        static_cast<unsigned>(mechanism.targetPoint.position));
    break;
  case CanonicalMechanismKind::Event:
  case CanonicalMechanismKind::CrossCoreEvent:
    hash = llvm::hash_combine(
        hash, mechanism.sourcePoint.operation,
        static_cast<unsigned>(mechanism.sourcePoint.position),
        mechanism.targetPoint.operation,
        static_cast<unsigned>(mechanism.targetPoint.position));
    break;
  case CanonicalMechanismKind::RecurringEvent:
    hash = llvm::hash_combine(
        hash, mechanism.sourcePoint.operation,
        static_cast<unsigned>(mechanism.sourcePoint.position),
        mechanism.targetPoint.operation,
        static_cast<unsigned>(mechanism.targetPoint.position),
        mechanism.recurrenceLoop.value_or(kInvalidCanonicalSyncId),
        mechanism.boundaryRecurring);
    break;
  case CanonicalMechanismKind::VisibilityFence:
    hash = llvm::hash_combine(
        hash, mechanism.targetPoint.operation,
        static_cast<unsigned>(mechanism.targetPoint.position),
        mechanism.generatedCacheMaintenance.has_value(),
        mechanism.generatedCacheMaintenance
            ? static_cast<unsigned>(*mechanism.generatedCacheMaintenance)
            : 0U);
    break;
  case CanonicalMechanismKind::TailBarrier:
    hash = llvm::hash_combine(
        hash, mechanism.sourcePoint.operation,
        static_cast<unsigned>(mechanism.sourcePoint.position),
        mechanism.targetPoint.operation,
        static_cast<unsigned>(mechanism.targetPoint.position));
    break;
  case CanonicalMechanismKind::IntrinsicOrder:
  case CanonicalMechanismKind::FixedFence:
    break;
  }
  return static_cast<std::size_t>(hash);
}

class MechanismInterner {
public:
  explicit MechanismInterner(const CanonicalSyncProgram &program)
      : statistics(program.getStatistics()) {
    for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
      record(mechanism);
    }
  }

  std::optional<CanonicalMechanismId>
  find(const CanonicalSyncProgram &program,
       const CanonicalMechanism &mechanism) const {
    const auto bucket = mechanismsByHash.find(hashMechanism(mechanism));
    if (bucket == mechanismsByHash.end()) {
      return std::nullopt;
    }
    for (CanonicalMechanismId id : bucket->second) {
      if (statistics) {
        ++statistics->mechanismInternKeyTests;
      }
      if (sameMechanism(program.getMechanism(id), mechanism)) {
        return id;
      }
    }
    return std::nullopt;
  }

  void record(const CanonicalMechanism &mechanism) {
    mechanismsByHash[hashMechanism(mechanism)].push_back(mechanism.id);
  }

private:
  std::unordered_map<std::size_t, SmallVector<CanonicalMechanismId, 2>>
      mechanismsByHash;
  CanonicalSyncStatistics *statistics = nullptr;
};

CanonicalMechanismId internMechanism(CanonicalSyncProgram &program,
                                     MechanismInterner &interner,
                                     CanonicalMechanism mechanism,
                                     CanonicalDemandId origin) {
  if (std::optional<CanonicalMechanismId> existing =
          interner.find(program, mechanism)) {
    program.appendMechanismOrigin(*existing, origin);
    program.appendMechanismCacheMaintenance(*existing,
                                            mechanism.cacheMaintenance);
    return *existing;
  }
  mechanism.origins.push_back(origin);
  const CanonicalMechanismId id =
      program.appendMechanism(std::move(mechanism));
  interner.record(program.getMechanism(id));
  return id;
}

CanonicalMechanismId internBaselineMechanism(CanonicalSyncProgram &program,
                                             MechanismInterner &interner,
                                             CanonicalMechanism mechanism) {
  if (std::optional<CanonicalMechanismId> existing =
          interner.find(program, mechanism)) {
    return *existing;
  }
  const CanonicalMechanismId id =
      program.appendMechanism(std::move(mechanism));
  interner.record(program.getMechanism(id));
  return id;
}

FailureOr<CanonicalMechanism> buildEventMechanism(
    const CanonicalSyncProgram &program, const CanonicalSyncTarget &targetModel,
    const CanonicalDemand &demand, CanonicalPhysicalResource source,
    CanonicalPhysicalResource target) {
  const CanonicalPhase &sourcePhase = program.getPhase(demand.source);
  const CanonicalPhase &targetPhase = program.getPhase(demand.target);
  Block *actionBlock =
      findCommonBlock(sourcePhase.operation, targetPhase.operation);
  Operation *setAfter = liftToBlock(sourcePhase.operation, actionBlock);
  Operation *waitBefore = liftToBlock(targetPhase.operation, actionBlock);
  const bool unresolvedAnchors = !actionBlock || !setAfter || !waitBefore;
  const bool invalidOrder =
      unresolvedAnchors || setAfter == waitBefore ||
      (setAfter && waitBefore && !setAfter->isBeforeInBlock(waitBefore));
  if (invalidOrder) {
    const std::optional<CanonicalRegionId> loopRegion = carryingLoop(demand);
    scf::ForOp carryingFor;
    if (loopRegion) {
      carryingFor = dyn_cast_or_null<scf::ForOp>(
          program.getRegion(*loopRegion).operation);
    }
    const bool containsEndpoints =
        carryingFor && carryingFor->isAncestor(sourcePhase.operation) &&
        carryingFor->isAncestor(targetPhase.operation);
    const bool boundaryProtocol = containsEndpoints &&
                                  !carryingFor.getBody()->empty() &&
                                  targetModel.supportsEvent(target, source);
    if (boundaryProtocol) {
      // Boundary recurrence circulates ownership in both directions.  The
      // forward ready lane has its own prime and drain around this loop, so a
      // surrounding loop would repeat that lifecycle and could reset an
      // unconsumed token.  Release pooling closes only the reverse lane.
      // Reject until both directions can be lifted over the outer lifecycle.
      if (hasEnclosingLoop(program, *loopRegion)) {
        targetPhase.operation->emitError(
            "canonical sync cannot construct a boundary recurrence protocol "
            "inside a repeating outer loop");
        return failure();
      }
      // Opposite choice arms cannot host a same-iteration Set/Wait pair. Use
      // two loop-carried ownership lanes instead: both directions are primed,
      // waited at the header, set at the latch, and drained after the loop.
      // This conservatively serializes the two physical pipelines across loop
      // iterations and establishes source_i -> target_{i+1} without pretending
      // the mutually exclusive endpoints coexecute.
      CanonicalMechanism mechanism;
      mechanism.kind = CanonicalMechanismKind::RecurringEvent;
      mechanism.source = source;
      mechanism.target = target;
      mechanism.sourcePoint = {carryingFor.getBody()->getTerminator(),
                               CanonicalProgramPointPosition::Before};
      mechanism.targetPoint = {&carryingFor.getBody()->front(),
                               CanonicalProgramPointPosition::Before};
      mechanism.actionRegion = *loopRegion;
      mechanism.guard =
          commonGuard(sourcePhase.controlPath, targetPhase.controlPath);
      mechanism.recurrenceLoop = *loopRegion;
      mechanism.boundaryRecurring = true;
      return mechanism;
    }
    targetPhase.operation->emitError("canonical sync cannot place a matched "
                                     "event pair on this structured path")
        << "; demand d" << demand.id << " source p" << demand.source
        << " target p" << demand.target;
    return failure();
  }
  if (isRepeatedBlock(actionBlock)) {
    auto loop = setAfter->getParentOfType<scf::ForOp>();
    auto targetLoop = waitBefore->getParentOfType<scf::ForOp>();
    const std::optional<CanonicalRegionId> loopRegion =
        loop && loop == targetLoop ? findLoopRegion(program, loop)
                                   : std::nullopt;
    // The reverse release token is consumed and returned once on every loop
    // iteration. The forward ready handoff must therefore also execute on
    // every iteration. A lifted wait before a nested choice remains legal
    // because both physical anchors then live directly in the loop body; a
    // pair confined to one choice arm does not.
    const bool unconditionalBodyActions =
        loop && setAfter->getBlock() == loop.getBody() &&
        waitBefore->getBlock() == loop.getBody();
    if (!loopRegion || !unconditionalBodyActions ||
        !targetModel.supportsEvent(target, source)) {
      targetPhase.operation->emitError(
          "canonical sync cannot construct a single-lane recurring event "
          "protocol on this loop")
          << "; demand d" << demand.id << " p" << demand.source << " ("
          << stringifyPIPE(source.pipe) << ") -> p" << demand.target << " ("
          << stringifyPIPE(target.pipe) << ")"
          << " requires loop-body physical anchors and a reverse release "
             "event";
      return failure();
    }
    CanonicalMechanism mechanism;
    mechanism.kind = CanonicalMechanismKind::RecurringEvent;
    mechanism.source = source;
    mechanism.target = target;
    mechanism.sourcePoint = {setAfter, CanonicalProgramPointPosition::After};
    mechanism.targetPoint = {waitBefore, CanonicalProgramPointPosition::Before};
    mechanism.actionRegion = *loopRegion;
    mechanism.guard =
        commonGuard(sourcePhase.controlPath, targetPhase.controlPath);
    mechanism.recurrenceLoop = *loopRegion;
    return mechanism;
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
    const CanonicalPhase &source = program.getPhase(demand.source);
    return buildTailMechanism(program, source);
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
          "canonical sync has no device-proven MTE3-to-MTE2 GM "
          "publication primitive")
          << "; demand d" << demand.id << " fails closed";
      return failure();
    }
    if (source.core != destination.core) {
      targetPhase.operation->emitError(
          "canonical sync has no proven cross-core GM visibility protocol")
          << "; demand d" << demand.id
          << " requires source-core publication, a cross-core completion "
             "transfer, and target-core acquisition";
      return failure();
    }
    if (hasPositiveDistance(demand)) {
      const std::optional<CanonicalRegionId> loopRegion = carryingLoop(demand);
      scf::ForOp carryingFor;
      if (loopRegion) {
        carryingFor = dyn_cast_or_null<scf::ForOp>(
            program.getRegion(*loopRegion).operation);
      }
      const bool containsEndpoints =
          carryingFor && carryingFor->isAncestor(sourcePhase.operation) &&
          carryingFor->isAncestor(targetPhase.operation);
      if (!containsEndpoints) {
        targetPhase.operation->emitError(
            "canonical sync has no proven loop-carried GM visibility "
            "protocol at a matching physical-core latch")
            << "; demand d" << demand.id;
        return failure();
      }
      Operation *latch = carryingFor.getBody()->getTerminator();
      FailureOr<CanonicalPhysicalResource> latchScalar =
          resolvePhysicalResource(program.getFunction(), latch, PIPE::PIPE_S);
      if (failed(latchScalar)) {
        return failure();
      }
      const bool matchingCore = latchScalar->core == source.core &&
                                latchScalar->core == destination.core;
      FailureOr<SmallVector<CanonicalPhysicalResource, 8>> drained =
          target.getFenceDrainedResources(latch);
      const bool sourceCompletesAtFence =
          succeeded(drained) &&
          (getVPTOSchedulingSemantics(sourcePhase.operation)
               .completionIsSynchronous ||
           llvm::is_contained(*drained, source));
      if (!matchingCore || !sourceCompletesAtFence) {
        targetPhase.operation->emitError(
            "canonical sync has no proven loop-carried GM visibility "
            "protocol at a matching physical-core latch")
            << "; demand d" << demand.id;
        return failure();
      }

      // The carrying-loop latch executes after every possible source in
      // iteration i and before scalar issue reaches iteration i+1.  At that
      // physical cut, the target-defined GM fence drains the documented
      // source pipes (or follows an already synchronous scalar store), while
      // the direction-specific DCache operation publishes dirty scalar data
      // or invalidates a stale scalar copy.  This is the documented
      // DCCI+DSB ordering protocol repeated once per carrying-loop iteration;
      // a bare SetFlag/WaitFlag is deliberately not credited with visibility.
      CanonicalMechanism mechanism;
      mechanism.kind = CanonicalMechanismKind::VisibilityFence;
      mechanism.source = source;
      mechanism.target = destination;
      mechanism.sourcePoint = {latch, CanonicalProgramPointPosition::Before};
      mechanism.targetPoint = mechanism.sourcePoint;
      mechanism.actionRegion = *loopRegion;
      mechanism.guard =
          commonGuard(sourcePhase.controlPath, targetPhase.controlPath);
      mechanism.generatedCacheMaintenance = demand.visibility->cacheMaintenance;
      return mechanism;
    }
    Operation *fence = findFixedVisibilityFence(
        sourcePhase.operation, targetPhase.operation, demand.visibility->scope);
    if (!fence) {
      CanonicalMechanism mechanism;
      mechanism.kind = CanonicalMechanismKind::VisibilityFence;
      mechanism.source = source;
      mechanism.target = destination;
      mechanism.sourcePoint = {targetPhase.operation,
                               CanonicalProgramPointPosition::Before};
      mechanism.targetPoint = mechanism.sourcePoint;
      mechanism.actionRegion = targetPhase.region;
      mechanism.guard = targetPhase.controlPath;
      mechanism.generatedCacheMaintenance = demand.visibility->cacheMaintenance;
      return mechanism;
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
  // A scalar instruction is complete before the scalar control stream can
  // issue an operation that consumes its result.  This is a completion fact,
  // not an AIC PIPE_S event (the documented AIC matrix has no such event
  // direction).  Scalar/non-scalar memory visibility remains handled by the
  // stronger visibility branch above.
  const bool synchronousCompletion =
      getVPTOSchedulingSemantics(sourcePhase.operation).completionIsSynchronous;
  if (synchronousCompletion && source.core == destination.core) {
    CanonicalMechanism mechanism;
    mechanism.kind = CanonicalMechanismKind::IntrinsicOrder;
    mechanism.source = source;
    mechanism.target = destination;
    mechanism.sourcePoint = {sourcePhase.operation,
                             CanonicalProgramPointPosition::After};
    mechanism.targetPoint = {targetPhase.operation,
                             CanonicalProgramPointPosition::Before};
    mechanism.actionRegion = demand.owner;
    mechanism.guard =
        commonGuard(sourcePhase.controlPath, targetPhase.controlPath);
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
        "canonical sync cannot generate a cross-core event until collective "
        "AIC/AIV participation is represented and proven")
        << "; demand d" << demand.id << " crosses "
        << stringifyCanonicalCore(source.core) << ':'
        << stringifyPIPE(source.pipe) << " to "
        << stringifyCanonicalCore(destination.core) << ':'
        << stringifyPIPE(destination.pipe);
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
    // A recurrence-only dependence can use the same documented ready/release
    // protocol as a same-iteration dependence.  The ready half deliberately
    // orders source_i before target_i (possibly stronger than the demand), and
    // the reverse release half orders target_i before source_{i+1}.  This is a
    // complete single-lane lifecycle with priming and draining, not bare event
    // reuse across iterations.
    return buildEventMechanism(program, target, demand, source, destination);
  }
  return buildEventMechanism(program, target, demand, source, destination);
}

bool demandIsMapped(const CanonicalSyncProgram &program,
                    CanonicalDemandId demand) {
  return program.getDirectMechanisms().size() == program.getDemands().size() &&
         program.getDirectMechanisms()[demand] != kInvalidCanonicalSyncId;
}

bool recurringMechanismMatches(const CanonicalSyncProgram &program,
                               const CanonicalMechanism &mechanism,
                               const CanonicalDemand &demand) {
  if (demand.requirement != CanonicalRequirement::Completion ||
      mechanism.kind != CanonicalMechanismKind::RecurringEvent ||
      mechanism.origins.empty() ||
      carryingLoop(demand) != mechanism.recurrenceLoop) {
    return false;
  }
  // A physical channel lifted around structured control can be the direct
  // mechanism for several branch-local demands.  Recurrence matching must
  // therefore consider every authenticated origin, not just the first demand
  // that happened to intern the shared channel.
  return llvm::any_of(mechanism.origins, [&](CanonicalDemandId origin) {
    const CanonicalDemand &forward = program.getDemand(origin);
    if (forward.requirement != CanonicalRequirement::Completion) {
      return false;
    }
    const bool sameEndpoints =
        demand.source == forward.source && demand.target == forward.target;
    const bool reverseEndpoints =
        demand.source == forward.target && demand.target == forward.source;
    if (sameEndpoints || reverseEndpoints) {
      return true;
    }

    // A branch-local forward channel also establishes completion on its
    // target pipeline.  FIFO issue order on that physical pipeline carries
    // the fact to a later target in another arm/iteration.  The reverse
    // release channel supplies the symmetric reuse relation.  Keep one
    // endpoint exact so this is a direct consequence of the documented
    // ready/release protocol rather than a general reachability guess.
    const CanonicalPhysicalResource demandSource =
        program.getPhase(demand.source).resource;
    const CanonicalPhysicalResource demandTarget =
        program.getPhase(demand.target).resource;
    const CanonicalPhysicalResource forwardSource =
        program.getPhase(forward.source).resource;
    const CanonicalPhysicalResource forwardTarget =
        program.getPhase(forward.target).resource;
    const bool forwardTargetPipeline =
        demand.source == forward.source && demandTarget == forwardTarget;
    const bool reverseTargetPipeline =
        demandSource == forwardTarget && demandTarget == forwardSource;
    return forwardTargetPipeline || reverseTargetPipeline;
  });
}

bool recurringMechanismHasExactOrigin(const CanonicalSyncProgram &program,
                                      const CanonicalMechanism &mechanism,
                                      const CanonicalDemand &demand) {
  return llvm::any_of(mechanism.origins, [&](CanonicalDemandId origin) {
    const CanonicalDemand &forward = program.getDemand(origin);
    return (demand.source == forward.source &&
            demand.target == forward.target) ||
           (demand.source == forward.target && demand.target == forward.source);
  });
}

std::optional<CanonicalMechanismId>
findRecurringMechanism(const CanonicalSyncProgram &program,
                       const CanonicalDemand &demand,
                       ArrayRef<CanonicalMechanismId> candidates) {
  auto found = llvm::find_if(candidates, [&](CanonicalMechanismId id) {
    return recurringMechanismMatches(program, program.getMechanism(id),
                                     demand);
  });
  return found == candidates.end()
             ? std::nullopt
             : std::optional<CanonicalMechanismId>(*found);
}

const CanonicalDemand *
findRecurringForwardDemand(const CanonicalSyncProgram &program,
                           const CanonicalDemand &recurrence,
                           ArrayRef<CanonicalDemandId> candidates) {
  const std::optional<CanonicalRegionId> loop = carryingLoop(recurrence);
  if (!loop) {
    return nullptr;
  }
  auto found = llvm::find_if(candidates, [&](CanonicalDemandId candidateId) {
    const CanonicalDemand &candidate = program.getDemand(candidateId);
    const bool invalidEndpoint =
        candidate.source >= program.getPhases().size() ||
        candidate.target >= program.getPhases().size();
    if (invalidEndpoint) {
      return false;
    }
    const bool matchingEndpoints = (candidate.source == recurrence.source &&
                                    candidate.target == recurrence.target) ||
                                   (candidate.source == recurrence.target &&
                                    candidate.target == recurrence.source);
    const CanonicalPhysicalResource recurrenceTarget =
        program.getPhase(recurrence.target).resource;
    const CanonicalPhysicalResource recurrenceSource =
        program.getPhase(recurrence.source).resource;
    const CanonicalPhysicalResource candidateSource =
        program.getPhase(candidate.source).resource;
    const CanonicalPhysicalResource candidateTarget =
        program.getPhase(candidate.target).resource;
    const bool matchingForwardPipeline =
        candidate.source == recurrence.source &&
        candidateTarget == recurrenceTarget;
    const bool matchingReleasePipeline = candidateTarget == recurrenceSource &&
                                         candidateSource == recurrenceTarget;
    const bool sameIteration = !hasPositiveDistance(candidate);
    const CanonicalPhase &source = program.getPhase(candidate.source);
    const CanonicalPhase &target = program.getPhase(candidate.target);
    return candidate.requirement == CanonicalRequirement::Completion &&
           (matchingEndpoints || matchingForwardPipeline ||
            matchingReleasePipeline) &&
           sameIteration && llvm::is_contained(source.loopPath, *loop) &&
           llvm::is_contained(target.loopPath, *loop);
  });
  return found == candidates.end() ? nullptr : &program.getDemand(*found);
}

SmallVector<SmallVector<CanonicalDemandId, 8>, 0>
buildRecurringForwardIndex(const CanonicalSyncProgram &program) {
  SmallVector<SmallVector<CanonicalDemandId, 8>, 0> byLoop(
      program.getRegions().size());
  for (const CanonicalDemand &demand : program.getDemands()) {
    const bool invalidEndpoint = demand.source >= program.getPhases().size() ||
                                 demand.target >= program.getPhases().size();
    if (invalidEndpoint ||
        demand.requirement != CanonicalRequirement::Completion ||
        hasPositiveDistance(demand)) {
      continue;
    }
    const CanonicalPhase &source = program.getPhase(demand.source);
    const CanonicalPhase &target = program.getPhase(demand.target);
    for (CanonicalRegionId loop : source.loopPath) {
      if (llvm::is_contained(target.loopPath, loop)) {
        byLoop[loop].push_back(demand.id);
      }
    }
  }
  return byLoop;
}

bool intrinsicBaselineCovers(const CanonicalSyncProgram &program,
                             const CanonicalSyncTarget &target,
                             const CanonicalDemand &demand) {
  if (demand.requirement != CanonicalRequirement::Completion ||
      demand.target >= program.getPhases().size()) {
    return false;
  }
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &destination = program.getPhase(demand.target);
  const bool synchronous =
      getVPTOSchedulingSemantics(source.operation).completionIsSynchronous;
  return (synchronous && source.resource.core == destination.resource.core) ||
         (source.resource == destination.resource &&
          target.hasIntrinsicCompletion(source.resource));
}

using FixedGuard = SmallVector<CanonicalControlAtom, 2>;

bool guardsCoverAllExecutions(ArrayRef<FixedGuard> guards) {
  if (llvm::any_of(guards, [](const FixedGuard &guard) {
        return guard.empty();
      })) {
    return true;
  }
  if (guards.empty()) {
    return false;
  }
  CanonicalRegionId choice = kInvalidCanonicalSyncId;
  for (const FixedGuard &guard : guards) {
    for (const CanonicalControlAtom &atom : guard) {
      choice = std::min(choice, atom.choice);
    }
  }
  if (choice == kInvalidCanonicalSyncId) {
    return false;
  }

  std::array<SmallVector<FixedGuard, 4>, 2> armGuards;
  for (const FixedGuard &guard : guards) {
    const auto atom = llvm::find_if(
        guard, [choice](const CanonicalControlAtom &candidate) {
          return candidate.choice == choice;
        });
    for (unsigned arm = 0; arm < armGuards.size(); ++arm) {
      const bool excludesArm = atom != guard.end() && atom->arm != arm;
      if (excludesArm) {
        continue;
      }
      FixedGuard reduced;
      llvm::copy_if(guard, std::back_inserter(reduced),
                    [choice](const CanonicalControlAtom &candidate) {
                      return candidate.choice != choice;
                    });
      armGuards[arm].push_back(std::move(reduced));
    }
  }
  return guardsCoverAllExecutions(armGuards[0]) &&
         guardsCoverAllExecutions(armGuards[1]);
}

bool fixedFenceCoversCompletion(const CanonicalSyncProgram &program,
                                const CanonicalDemand &demand) {
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &target = program.getPhase(demand.target);
  if (!controlsCanCoexecute(demand.sourceGuard, demand.targetGuard)) {
    return false;
  }
  const SmallVector<CanonicalControlAtom, 2> executionGuard =
      conjoinCompatibleControlPaths(demand.sourceGuard, demand.targetGuard);
  struct GuardGroup {
    SmallVector<CanonicalRegionId, 2> loops;
    SmallVector<FixedGuard, 4> guards;
  };
  SmallVector<GuardGroup, 2> groups;
  for (const CanonicalFenceEffect &effect : program.getFenceEffects()) {
    const bool requiredExecution =
        llvm::all_of(effect.loopPath, [&](CanonicalRegionId loop) {
          return llvm::is_contained(source.loopPath, loop) ||
                 llvm::is_contained(target.loopPath, loop);
        });
    const bool inapplicable =
        !llvm::is_contained(effect.drainedResources, source.resource) ||
        !requiredExecution ||
        !controlsCanCoexecute(executionGuard, effect.guard);
    if (inapplicable) {
      continue;
    }
    const CanonicalProgramPoint point{effect.operation,
                                      CanonicalProgramPointPosition::After};
    const bool ordered = phaseMayPrecedePoint(source, point) &&
                         pointMustPrecedePhase(point, target);
    if (!ordered) {
      continue;
    }
    FixedGuard residualGuard;
    llvm::copy_if(effect.guard, std::back_inserter(residualGuard),
                  [&](const CanonicalControlAtom &atom) {
                    return !llvm::is_contained(executionGuard, atom);
                  });
    auto group = llvm::find_if(groups, [&](const GuardGroup &candidate) {
      return candidate.loops == effect.loopPath;
    });
    if (group == groups.end()) {
      groups.push_back({effect.loopPath, {}});
      group = std::prev(groups.end());
    }
    group->guards.push_back(std::move(residualGuard));
  }
  return llvm::any_of(groups, [](const GuardGroup &group) {
    return guardsCoverAllExecutions(group.guards);
  });
}

bool fixedBaselineCovers(const CanonicalSyncProgram &program,
                         const CanonicalSyncTarget &target,
                         const CanonicalDemand &demand) {
  if (demand.kind == CanonicalDemandKind::ExitCompletion ||
      intrinsicBaselineCovers(program, target, demand)) {
    return true;
  }
  if (demand.requirement == CanonicalRequirement::Visibility) {
    const bool unsupportedVisibility =
        !demand.visibility || hasPositiveDistance(demand) ||
        demand.visibility->direction ==
            CanonicalVisibilityDirection::Mte3ToMte2Gm;
    if (unsupportedVisibility) {
      return false;
    }
    const CanonicalPhase &source = program.getPhase(demand.source);
    const CanonicalPhase &destination = program.getPhase(demand.target);
    Operation *fence = findFixedVisibilityFence(
        source.operation, destination.operation, demand.visibility->scope);
    return fence &&
           succeeded(findFixedCacheMaintenance(program, demand, fence));
  }
  if (hasPositiveDistance(demand)) {
    return findFixedRecurrenceFence(program, demand) != nullptr;
  }
  return fixedFenceCoversCompletion(program, demand);
}

} // namespace

LogicalResult mlir::pto::canonical_sync_detail::integrateCanonicalFixedBaseline(
    CanonicalSyncProgram &program, const CanonicalSyncTarget &target) {
  const bool frozen = program.isGraphFrozen() || program.isFrozen();
  if (frozen) {
    return program.getFunction().emitError(
        "canonical sync cannot integrate fixed supply into a frozen graph");
  }
  llvm::BitVector retained(program.getDemands().size(), true);
  std::uint64_t fixedCovered = 0;
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (fixedBaselineCovers(program, target, demand)) {
      retained.reset(demand.id);
      ++fixedCovered;
    }
  }
  program.retainDemands(retained);
  if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
    statistics->fixedCoveredDemands = fixedCovered;
    statistics->demands = program.getDemands().size();
  }
  return success();
}

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
  MechanismInterner interner(program);
  SmallVector<SmallVector<CanonicalMechanismId, 2>, 0> recurringByLoop(
      program.getRegions().size());
  const auto recordRecurring = [&](CanonicalMechanismId id) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (mechanism.kind != CanonicalMechanismKind::RecurringEvent ||
        !mechanism.recurrenceLoop) {
      return;
    }
    SmallVectorImpl<CanonicalMechanismId> &bucket =
        recurringByLoop[*mechanism.recurrenceLoop];
    if (!llvm::is_contained(bucket, id)) {
      bucket.push_back(id);
    }
  };
  const SmallVector<SmallVector<CanonicalDemandId, 8>, 0>
      recurringForwardByLoop = buildRecurringForwardIndex(program);
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (demandIsMapped(program, demand.id)) {
      continue;
    }
    if (demand.requirement == CanonicalRequirement::Completion &&
        hasPositiveDistance(demand)) {
      const std::optional<CanonicalRegionId> loop = carryingLoop(demand);
      std::optional<CanonicalMechanismId> recurring =
          loop ? findRecurringMechanism(program, demand, recurringByLoop[*loop])
               : std::nullopt;
      if (!recurring) {
        const CanonicalDemand *forward =
            loop ? findRecurringForwardDemand(program, demand,
                                              recurringForwardByLoop[*loop])
                 : nullptr;
        if (forward && !demandIsMapped(program, forward->id)) {
          FailureOr<CanonicalMechanism> forwardMechanism =
              buildDirectMechanism(program, *target, *forward);
          const bool recurringConstructed =
              succeeded(forwardMechanism) &&
              forwardMechanism->kind == CanonicalMechanismKind::RecurringEvent;
          if (recurringConstructed) {
            const CanonicalMechanismId forwardId = internMechanism(
                program, interner, std::move(*forwardMechanism), forward->id);
            recordRecurring(forwardId);
            program.setDirectMechanism(forward->id, forwardId);
            recurring = forwardId;
          }
        }
      }
      if (recurring) {
        // A recurrence that reaches another phase on the same physical target
        // pipeline is additional coverage, not another direct origin of the
        // channel.  Preserve that distinction in the set-cover column.
        if (recurringMechanismHasExactOrigin(
                program, program.getMechanism(*recurring), demand)) {
          program.appendMechanismOrigin(*recurring, demand.id);
        }
        program.setDirectMechanism(demand.id, *recurring);
        continue;
      }
    }
    FailureOr<CanonicalMechanism> mechanism =
        buildDirectMechanism(program, *target, demand);
    if (failed(mechanism)) {
      return failure();
    }
    const CanonicalMechanismId id =
        internMechanism(program, interner, std::move(*mechanism), demand.id);
    recordRecurring(id);
    program.setDirectMechanism(demand.id, id);
  }
  for (const CanonicalPhase &phase : program.getPhases()) {
    FailureOr<CanonicalMechanism> tail = buildTailMechanism(program, phase);
    if (failed(tail)) {
      return failure();
    }
    internBaselineMechanism(program, interner, std::move(*tail));
  }
  if (program.getPhases().empty()) {
    const std::optional<bool> vectorExecution =
        resolvePTOExecutionVector(program.getFunction());
    if (vectorExecution) {
      CanonicalMechanism tail;
      tail.kind = CanonicalMechanismKind::TailBarrier;
      tail.source = {*vectorExecution ? CanonicalCore::AIV : CanonicalCore::AIC,
                     PIPE::PIPE_ALL};
      tail.target = tail.source;
      tail.actionRegion = 0;
      internBaselineMechanism(program, interner, std::move(tail));
    }
  }
  program.mechanismCatalogComplete = true;
  return success();
}
