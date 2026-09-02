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

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

enum class AllocationUnitKind : std::uint8_t {
  Ready,
  RecurringRelease,
  PooledRelease,
  SerializedReady,
  SerializedRelease,
  CrossCore,
};

struct AllocationUnit {
  AllocationUnitKind kind = AllocationUnitKind::Ready;
  SmallVector<CanonicalMechanismId, 2> mechanisms;
  std::optional<unsigned> serializedGroup;
  std::optional<unsigned> releasePool;
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  SmallVector<CanonicalControlAtom, 2> guard;
  std::optional<CanonicalRegionId> recurrenceLoop;
  unsigned eventId = 0;
  SmallVector<unsigned, 2> releaseGroups;
  bool boundaryRecurring = false;
};

bool recurringReadyReuseIsOrdered(const CanonicalSyncProgram &program,
                                  const AllocationUnit &first,
                                  const AllocationUnit &second);

struct AllocationFailure {
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  Operation *witness = nullptr;
  bool crossCore = false;
};

bool unitsInterfere(const CanonicalSyncProgram &program,
                    const AllocationUnit &first, const AllocationUnit &second) {
  return !recurringReadyReuseIsOrdered(program, first, second) &&
         controlsCanCoexecute(first.guard, second.guard);
}

bool regionHasLoopAncestor(const CanonicalSyncProgram &program,
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

SmallVector<CanonicalControlAtom, 2>
getOnceOnlyControlPath(const CanonicalSyncProgram &program,
                       ArrayRef<CanonicalControlAtom> path) {
  SmallVector<CanonicalControlAtom, 2> result;
  for (const CanonicalControlAtom &atom : path) {
    if (!regionHasLoopAncestor(program, atom.choice)) {
      result.push_back(atom);
    }
  }
  return result;
}

SmallVector<CanonicalControlAtom, 2>
getRegionControlPath(const CanonicalSyncProgram &program,
                     CanonicalRegionId region) {
  SmallVector<CanonicalControlAtom, 2> result;
  for (CanonicalRegionId current = region;
       current != kInvalidCanonicalSyncId;) {
    const CanonicalRegion &currentRegion = program.getRegion(current);
    const CanonicalRegionId parent = currentRegion.parent;
    if (parent == kInvalidCanonicalSyncId) {
      break;
    }
    if (program.getRegion(parent).kind == CanonicalRegionKind::Choice &&
        !regionHasLoopAncestor(program, parent)) {
      result.push_back({parent, currentRegion.arm});
    }
    current = parent;
  }
  llvm::sort(result, [](const CanonicalControlAtom &left,
                        const CanonicalControlAtom &right) {
    return std::tie(left.choice, left.arm) <
           std::tie(right.choice, right.arm);
  });
  return result;
}

bool idAvailable(const CanonicalSyncProgram &program,
                 const AllocationUnit &candidate, unsigned eventId,
                 ArrayRef<AllocationUnit> assigned) {
  for (const AllocationUnit &existing : assigned) {
    const bool sameCrossCoreKey =
        candidate.kind == AllocationUnitKind::CrossCore &&
        existing.kind == AllocationUnitKind::CrossCore &&
        existing.eventId == eventId;
    const bool sameIntraCoreKey =
        candidate.kind != AllocationUnitKind::CrossCore &&
        existing.kind != AllocationUnitKind::CrossCore &&
        candidate.source == existing.source &&
        candidate.target == existing.target && existing.eventId == eventId;
    const bool collides = (sameCrossCoreKey || sameIntraCoreKey) &&
                          unitsInterfere(program, candidate, existing);
    if (collides) {
      return false;
    }
  }
  return true;
}

bool isScarcityMember(ArrayRef<CanonicalScarcityEventGroup> groups,
                      CanonicalMechanismId mechanism) {
  return llvm::any_of(groups, [&](const CanonicalScarcityEventGroup &group) {
    return llvm::is_contained(group.members, mechanism);
  });
}

SmallVector<AllocationUnit, 8>
buildAllocationUnits(const CanonicalSyncProgram &program,
                     ArrayRef<CanonicalScarcityEventGroup> groups,
                     ArrayRef<CanonicalRecurringReleasePool> releasePools) {
  const CanonicalSetCoverSolution &solution = *program.getSetCoverSolution();
  SmallVector<AllocationUnit, 8> units;
  for (CanonicalMechanismId mechanismId : solution.mechanisms) {
    const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
    if (mechanism.kind == CanonicalMechanismKind::Event) {
      if (!isScarcityMember(groups, mechanism.id)) {
        units.push_back({AllocationUnitKind::Ready,
                         {mechanism.id},
                         std::nullopt,
                         std::nullopt,
                         mechanism.source,
                         mechanism.target,
                         getOnceOnlyControlPath(program, mechanism.guard),
                         std::nullopt,
                         0});
      }
      continue;
    }
    if (mechanism.kind == CanonicalMechanismKind::CrossCoreEvent) {
      units.push_back({AllocationUnitKind::CrossCore,
                       {mechanism.id},
                       std::nullopt,
                       std::nullopt,
                       mechanism.source,
                       mechanism.target,
                       getOnceOnlyControlPath(program, mechanism.guard),
                       std::nullopt,
                       0});
      continue;
    }
    if (mechanism.kind != CanonicalMechanismKind::RecurringEvent) {
      continue;
    }
    if (isScarcityMember(groups, mechanism.id)) {
      continue;
    }
    units.push_back({AllocationUnitKind::Ready,
                     {mechanism.id},
                     std::nullopt,
                     std::nullopt,
                     mechanism.source,
                     mechanism.target,
                     getOnceOnlyControlPath(program, mechanism.guard),
                     mechanism.recurrenceLoop,
                     0});
    units.back().boundaryRecurring = mechanism.boundaryRecurring;
    auto release = llvm::find_if(units, [&](const AllocationUnit &unit) {
      return unit.kind == AllocationUnitKind::RecurringRelease &&
             unit.source == mechanism.target &&
             unit.target == mechanism.source &&
             unit.recurrenceLoop == mechanism.recurrenceLoop;
    });
    if (release == units.end()) {
      units.push_back({AllocationUnitKind::RecurringRelease,
                       {mechanism.id},
                       std::nullopt,
                       std::nullopt,
                       mechanism.target,
                       mechanism.source,
                       getRegionControlPath(program,
                                            *mechanism.recurrenceLoop),
                       mechanism.recurrenceLoop,
                       0});
      units.back().boundaryRecurring = mechanism.boundaryRecurring;
    } else {
      release->mechanisms.push_back(mechanism.id);
      release->boundaryRecurring |= mechanism.boundaryRecurring;
    }
  }
  for (auto [index, group] : llvm::enumerate(groups)) {
    const bool boundaryRecurring =
        llvm::any_of(group.members, [&](CanonicalMechanismId mechanism) {
          return program.getMechanism(mechanism).boundaryRecurring;
        });
    units.push_back({AllocationUnitKind::SerializedReady,
                     {},
                     static_cast<unsigned>(index),
                     std::nullopt,
                     group.source,
                     group.target,
                     getOnceOnlyControlPath(program, group.guard),
                     group.recurrenceLoop,
                     0});
    units.back().boundaryRecurring = boundaryRecurring;
    if (group.kind == CanonicalScarcityEventKind::Serialized) {
      units.push_back({AllocationUnitKind::SerializedRelease,
                       {},
                       static_cast<unsigned>(index),
                       std::nullopt,
                       group.target,
                       group.source,
                       getOnceOnlyControlPath(program, group.guard),
                       std::nullopt,
                       0});
      continue;
    }
    if (!group.recurrenceLoop) {
      continue;
    }
    auto release = llvm::find_if(units, [&](const AllocationUnit &unit) {
      return unit.kind == AllocationUnitKind::RecurringRelease &&
             unit.source == group.target && unit.target == group.source &&
             unit.recurrenceLoop == group.recurrenceLoop;
    });
    if (release == units.end()) {
      AllocationUnit unit{AllocationUnitKind::RecurringRelease,
                          {},
                          std::nullopt,
                          std::nullopt,
                          group.target,
                          group.source,
                          getRegionControlPath(program,
                                               *group.recurrenceLoop),
                          group.recurrenceLoop,
                          0};
      unit.releaseGroups.push_back(static_cast<unsigned>(index));
      unit.boundaryRecurring = boundaryRecurring;
      units.push_back(std::move(unit));
    } else {
      release->releaseGroups.push_back(static_cast<unsigned>(index));
      release->boundaryRecurring |= boundaryRecurring;
    }
  }
  for (auto [poolIndex, pool] : llvm::enumerate(releasePools)) {
    for (AllocationUnit &unit : units) {
      if ((unit.kind == AllocationUnitKind::Ready ||
           unit.kind == AllocationUnitKind::SerializedReady) &&
          unit.recurrenceLoop &&
          unit.target == pool.releaseSource &&
          unit.source == pool.releaseTarget &&
          llvm::is_contained(pool.recurrenceLoops, *unit.recurrenceLoop)) {
        unit.releasePool = static_cast<unsigned>(poolIndex);
      }
    }
    SmallVector<unsigned, 4> members;
    for (auto [unitIndex, unit] : llvm::enumerate(units)) {
      if (unit.kind == AllocationUnitKind::RecurringRelease &&
          unit.source == pool.releaseSource &&
          unit.target == pool.releaseTarget && unit.recurrenceLoop &&
          llvm::is_contained(pool.recurrenceLoops, *unit.recurrenceLoop)) {
        members.push_back(static_cast<unsigned>(unitIndex));
      }
    }
    const bool lostAllocationUnit =
        members.size() != pool.recurrenceLoops.size();
    if (lostAllocationUnit) {
      llvm_unreachable("recurring release pool lost an allocation unit");
    }
    AllocationUnit pooled = units[members.front()];
    pooled.kind = AllocationUnitKind::PooledRelease;
    pooled.releasePool = static_cast<unsigned>(poolIndex);
    pooled.recurrenceLoop.reset();
    pooled.guard.clear();
    for (unsigned member : ArrayRef<unsigned>(members).drop_front()) {
      llvm::append_range(pooled.mechanisms, units[member].mechanisms);
      llvm::append_range(pooled.releaseGroups, units[member].releaseGroups);
    }
    for (unsigned member : llvm::reverse(members)) {
      units.erase(units.begin() + member);
    }
    units.push_back(std::move(pooled));
  }
  return units;
}

bool allocateUnits(const CanonicalSyncProgram &program,
                   const CanonicalSyncTarget &target,
                   SmallVectorImpl<AllocationUnit> &units,
                   AllocationFailure &failure) {
  SmallVector<AllocationUnit, 8> assigned;
  for (AllocationUnit &unit : units) {
    ArrayRef<unsigned> ids = unit.kind == AllocationUnitKind::CrossCore
                                 ? target.getCompilerCrossCoreEventIds()
                                 : target.getCompilerEventIds();
    SmallVector<unsigned, 6> reserved;
    if (unit.kind != AllocationUnitKind::CrossCore) {
      reserved =
          reservedEventIds(program.getFunction(), unit.source, unit.target);
    }
    const auto available = llvm::find_if(ids, [&](unsigned eventId) {
      return !llvm::is_contained(reserved, eventId) &&
             idAvailable(program, unit, eventId, assigned);
    });
    if (available == ids.end()) {
      failure.source = unit.source;
      failure.target = unit.target;
      failure.crossCore = unit.kind == AllocationUnitKind::CrossCore;
      const CanonicalMechanismId witnessId = unit.mechanisms.empty()
                                                 ? kInvalidCanonicalSyncId
                                                 : unit.mechanisms.front();
      failure.witness =
          witnessId == kInvalidCanonicalSyncId
              ? program.getFunction().getOperation()
              : program.getMechanism(witnessId).targetPoint.operation;
      return false;
    }
    unit.eventId = *available;
    assigned.push_back(unit);
  }
  return true;
}

bool sameChainClass(const CanonicalMechanism &first,
                    const CanonicalMechanism &second) {
  return first.source == second.source && first.target == second.target &&
         first.guard == second.guard && first.sourcePoint.operation &&
         first.targetPoint.operation && second.sourcePoint.operation &&
         second.targetPoint.operation &&
         first.sourcePoint.operation->getBlock() ==
             first.targetPoint.operation->getBlock() &&
         first.sourcePoint.operation->getBlock() ==
             second.sourcePoint.operation->getBlock() &&
         first.sourcePoint.operation->getBlock() ==
             second.targetPoint.operation->getBlock();
}

bool sourcePointLessInBlock(const CanonicalMechanism *first,
                            const CanonicalMechanism *second) {
  Operation *left = first->sourcePoint.operation;
  Operation *right = second->sourcePoint.operation;
  const bool sameBlock = left->getBlock() == right->getBlock();
  if (!sameBlock) {
    llvm_unreachable("cannot order canonical event cuts across blocks");
  }
  if (left == right) {
    return first->id < second->id;
  }
  return left->isBeforeInBlock(right);
}

bool sameCoalescingClass(const CanonicalMechanism &first,
                         const CanonicalMechanism &second) {
  return first.kind == second.kind && first.source == second.source &&
         first.target == second.target && first.guard == second.guard &&
         first.recurrenceLoop == second.recurrenceLoop &&
         sameChainClass(first, second);
}

bool appendCoalescedFallback(
    const CanonicalSyncProgram &program, const CanonicalSyncTarget &target,
    const AllocationFailure &failure,
    SmallVectorImpl<CanonicalScarcityEventGroup> &groups) {
  if (failure.crossCore) {
    return false;
  }
  const CanonicalSetCoverSolution &solution = *program.getSetCoverSolution();
  SmallVector<const CanonicalMechanism *, 16> candidates;
  for (CanonicalMechanismId id : solution.mechanisms) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    const bool supportedKind =
        mechanism.kind == CanonicalMechanismKind::Event ||
        (mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
         !mechanism.boundaryRecurring);
    const bool recurringHasRelease =
        mechanism.kind != CanonicalMechanismKind::RecurringEvent ||
        target.supportsEvent(failure.target, failure.source);
    if (supportedKind && recurringHasRelease &&
        mechanism.source == failure.source &&
        mechanism.target == failure.target &&
        !isScarcityMember(groups, mechanism.id) &&
        mechanism.sourcePoint.position ==
            CanonicalProgramPointPosition::After &&
        mechanism.targetPoint.position ==
            CanonicalProgramPointPosition::Before &&
        mechanism.sourcePoint.operation && mechanism.targetPoint.operation &&
        mechanism.sourcePoint.operation->getBlock() ==
            mechanism.targetPoint.operation->getBlock() &&
        programPointMustPrecede(mechanism.sourcePoint, mechanism.targetPoint)) {
      candidates.push_back(&mechanism);
    }
  }

  SmallVector<const CanonicalMechanism *, 8> best;
  CanonicalProgramPoint bestSource;
  CanonicalProgramPoint bestTarget;
  for (const CanonicalMechanism *pivot : candidates) {
    SmallVector<const CanonicalMechanism *, 8> group;
    CanonicalProgramPoint latestSource = pivot->sourcePoint;
    for (const CanonicalMechanism *candidate : candidates) {
      const bool sameClass = sameCoalescingClass(*pivot, *candidate);
      const bool startsBeforeTarget =
          programPointMustPrecede(candidate->sourcePoint, pivot->targetPoint);
      const bool targetCoversCandidate =
          programPointMustPrecede(pivot->targetPoint, candidate->targetPoint);
      if (!sameClass || !startsBeforeTarget || !targetCoversCandidate) {
        continue;
      }
      group.push_back(candidate);
      if (programPointMustPrecede(latestSource, candidate->sourcePoint)) {
        latestSource = candidate->sourcePoint;
      }
    }
    llvm::stable_sort(group, sourcePointLessInBlock);
    const bool improvesBest = group.size() > best.size();
    if (improvesBest) {
      best = std::move(group);
      bestSource = latestSource;
      bestTarget = pivot->targetPoint;
    }
  }
  const bool validBest =
      best.size() >= 2 && programPointMustPrecede(bestSource, bestTarget);
  if (!validBest) {
    return false;
  }

  CanonicalScarcityEventGroup group;
  group.kind = CanonicalScarcityEventKind::Coalesced;
  group.source = failure.source;
  group.target = failure.target;
  group.sourcePoint = bestSource;
  group.targetPoint = bestTarget;
  group.guard = best.front()->guard;
  group.recurrenceLoop = best.front()->recurrenceLoop;
  for (const CanonicalMechanism *mechanism : best) {
    group.members.push_back(mechanism->id);
  }
  groups.push_back(std::move(group));
  return true;
}

bool appendSerializedFallback(
    const CanonicalSyncProgram &program, const CanonicalSyncTarget &target,
    const AllocationFailure &failure,
    SmallVectorImpl<CanonicalScarcityEventGroup> &groups) {
  if (failure.crossCore ||
      !target.supportsEvent(failure.target, failure.source)) {
    return false;
  }
  const CanonicalSetCoverSolution &solution = *program.getSetCoverSolution();
  SmallVector<const CanonicalMechanism *, 16> candidates;
  for (CanonicalMechanismId id : solution.mechanisms) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (mechanism.kind == CanonicalMechanismKind::Event &&
        mechanism.source == failure.source &&
        mechanism.target == failure.target &&
        !isScarcityMember(groups, mechanism.id) &&
        mechanism.sourcePoint.position ==
            CanonicalProgramPointPosition::After &&
        mechanism.targetPoint.position ==
            CanonicalProgramPointPosition::Before &&
        mechanism.sourcePoint.operation && mechanism.targetPoint.operation &&
        mechanism.sourcePoint.operation->getBlock() ==
            mechanism.targetPoint.operation->getBlock()) {
      candidates.push_back(&mechanism);
    }
  }
  SmallVector<const CanonicalMechanism *, 8> best;
  for (const CanonicalMechanism *seed : candidates) {
    SmallVector<const CanonicalMechanism *, 8> chainClass;
    for (const CanonicalMechanism *candidate : candidates) {
      if (sameChainClass(*seed, *candidate)) {
        chainClass.push_back(candidate);
      }
    }
    llvm::stable_sort(chainClass, sourcePointLessInBlock);
    const auto seedPosition = llvm::find(chainClass, seed);
    const std::size_t firstCandidate =
        static_cast<std::size_t>(seedPosition - chainClass.begin() + 1);
    SmallVector<const CanonicalMechanism *, 8> chain{seed};
    const CanonicalMechanism *previous = seed;
    for (const CanonicalMechanism *candidate :
         ArrayRef<const CanonicalMechanism *>(chainClass)
             .drop_front(firstCandidate)) {
      const bool distinctOperations =
          previous->targetPoint.operation != candidate->sourcePoint.operation;
      const bool ordered = distinctOperations &&
                           previous->targetPoint.operation->isBeforeInBlock(
                               candidate->sourcePoint.operation);
      if (!ordered) {
        continue;
      }
      chain.push_back(candidate);
      previous = candidate;
    }
    const bool improvesBest = chain.size() > best.size();
    if (improvesBest) {
      best = std::move(chain);
    }
  }
  const bool validBest = best.size() >= 2;
  if (!validBest) {
    return false;
  }

  CanonicalScarcityEventGroup group;
  group.kind = CanonicalScarcityEventKind::Serialized;
  group.source = failure.source;
  group.target = failure.target;
  group.guard = best.front()->guard;
  for (const CanonicalMechanism *mechanism : best) {
    group.members.push_back(mechanism->id);
  }
  groups.push_back(std::move(group));
  return true;
}

Block *findCommonBlock(Operation *first, Operation *second) {
  SmallVector<Block *, 8> firstBlocks;
  for (Operation *current = first; current; current = current->getParentOp()) {
    if (current->getBlock()) {
      firstBlocks.push_back(current->getBlock());
    }
  }
  for (Operation *current = second; current; current = current->getParentOp()) {
    Block *block = current->getBlock();
    if (block && llvm::is_contained(firstBlocks, block)) {
      return block;
    }
  }
  return nullptr;
}

Operation *liftToBlock(Operation *operation, Block *block) {
  while (operation) {
    Block *currentBlock = operation->getBlock();
    if (currentBlock == block) {
      return operation;
    }
    operation = operation->getParentOp();
  }
  return nullptr;
}

bool regionContains(const CanonicalSyncProgram &program,
                    CanonicalRegionId ancestor, CanonicalRegionId region) {
  for (CanonicalRegionId current = region;
       current != kInvalidCanonicalSyncId;
       current = program.getRegion(current).parent) {
    if (current == ancestor) {
      return true;
    }
  }
  return false;
}

bool appendRecurringReleasePool(
    const CanonicalSyncProgram &program, CanonicalPhysicalResource source,
    CanonicalPhysicalResource target, bool nestedOnly,
    std::size_t minimumSize,
    ArrayRef<CanonicalScarcityEventGroup> groups,
    SmallVectorImpl<CanonicalRecurringReleasePool> &releasePools) {
  const SmallVector<AllocationUnit, 8> units =
      buildAllocationUnits(program, groups, releasePools);
  Block &entry = program.getFunction().getBody().front();
  struct Candidate {
    CanonicalRegionId loop = kInvalidCanonicalSyncId;
    Operation *frontier = nullptr;
  };
  SmallVector<Candidate, 8> candidates;
  for (const AllocationUnit &unit : units) {
    if (unit.kind != AllocationUnitKind::RecurringRelease ||
        unit.source != source || unit.target != target ||
        !unit.recurrenceLoop || unit.boundaryRecurring ||
        (nestedOnly &&
         !regionHasLoopAncestor(program, *unit.recurrenceLoop))) {
      continue;
    }
    Operation *loop = program.getRegion(*unit.recurrenceLoop).operation;
    Operation *frontier = liftToBlock(loop, &entry);
    if (frontier) {
      candidates.push_back({*unit.recurrenceLoop, frontier});
    }
  }
  llvm::stable_sort(candidates, [](const Candidate &left,
                                   const Candidate &right) {
    if (left.frontier == right.frontier) {
      return left.loop < right.loop;
    }
    return left.frontier->isBeforeInBlock(right.frontier);
  });

  SmallVector<Candidate, 8> poolMembers;
  for (const Candidate &candidate : candidates) {
    const bool nested = llvm::any_of(poolMembers, [&](const Candidate &member) {
      return regionContains(program, member.loop, candidate.loop) ||
             regionContains(program, candidate.loop, member.loop);
    });
    if (!nested) {
      poolMembers.push_back(candidate);
    }
  }
  const bool insufficientMembers = poolMembers.size() < minimumSize;
  if (insufficientMembers) {
    return false;
  }

  CanonicalRecurringReleasePool pool;
  pool.releaseSource = source;
  pool.releaseTarget = target;
  pool.primePoint = {poolMembers.front().frontier,
                     CanonicalProgramPointPosition::Before};
  pool.drainPoint = {poolMembers.back().frontier,
                     CanonicalProgramPointPosition::After};
  for (const Candidate &member : poolMembers) {
    pool.recurrenceLoops.push_back(member.loop);
  }
  releasePools.push_back(std::move(pool));
  return true;
}

bool appendRequiredNestedReleasePool(
    const CanonicalSyncProgram &program,
    ArrayRef<CanonicalScarcityEventGroup> groups,
    SmallVectorImpl<CanonicalRecurringReleasePool> &releasePools) {
  const SmallVector<AllocationUnit, 8> units =
      buildAllocationUnits(program, groups, releasePools);
  auto candidate = llvm::find_if(units, [&](const AllocationUnit &unit) {
    return unit.kind == AllocationUnitKind::RecurringRelease &&
           unit.recurrenceLoop &&
           regionHasLoopAncestor(program, *unit.recurrenceLoop);
  });
  return candidate != units.end() &&
         appendRecurringReleasePool(program, candidate->source,
                                    candidate->target, true, 1, groups,
                                    releasePools);
}

bool appendRecurringReleasePoolFallback(
    const CanonicalSyncProgram &program, const AllocationFailure &failure,
    ArrayRef<CanonicalScarcityEventGroup> groups,
    SmallVectorImpl<CanonicalRecurringReleasePool> &releasePools) {
  return !failure.crossCore &&
         appendRecurringReleasePool(program, failure.source, failure.target,
                                    false, 2, groups, releasePools);
}

bool onceOnlyLoopPrecedes(const CanonicalSyncProgram &program,
                          CanonicalRegionId previousLoop,
                          CanonicalRegionId nextLoop) {
  Operation *previous = program.getRegion(previousLoop).operation;
  Operation *next = program.getRegion(nextLoop).operation;
  Block *block = findCommonBlock(previous, next);
  // A previous release drain can certify ready-lane reuse only across a
  // once-only outer issue frontier. Repeating outer frontiers would require a
  // separate lifecycle proof.
  if (!block || !isa<func::FuncOp>(block->getParentOp())) {
    return false;
  }
  Operation *previousAnchor = liftToBlock(previous, block);
  Operation *nextAnchor = liftToBlock(next, block);
  if (!previousAnchor || !nextAnchor || previousAnchor == nextAnchor ||
      !previousAnchor->isBeforeInBlock(nextAnchor)) {
    return false;
  }
  return true;
}

bool recurringReadyReuseIsOrdered(const CanonicalSyncProgram &program,
                                  const AllocationUnit &first,
                                  const AllocationUnit &second) {
  const auto isEligible = [](const AllocationUnit &unit) {
    const bool readyKind = unit.kind == AllocationUnitKind::Ready ||
                           unit.kind == AllocationUnitKind::SerializedReady;
    return readyKind && unit.recurrenceLoop && !unit.boundaryRecurring;
  };
  const bool eligible = isEligible(first) && isEligible(second) &&
                        first.source == second.source &&
                        first.target == second.target;
  if (!eligible) {
    return false;
  }
  if (first.releasePool && first.releasePool == second.releasePool &&
      first.recurrenceLoop != second.recurrenceLoop) {
    return true;
  }
  return onceOnlyLoopPrecedes(program, *first.recurrenceLoop,
                              *second.recurrenceLoop) ||
         onceOnlyLoopPrecedes(program, *second.recurrenceLoop,
                              *first.recurrenceLoop);
}

void commitAllocation(CanonicalSyncProgram &program,
                      ArrayRef<AllocationUnit> units,
                      SmallVector<CanonicalScarcityEventGroup, 2> groups,
                      SmallVector<CanonicalRecurringReleasePool, 2>
                          releasePools) {
  for (const AllocationUnit &unit : units) {
    if (unit.kind == AllocationUnitKind::SerializedReady) {
      groups[*unit.serializedGroup].eventId = unit.eventId;
      continue;
    }
    if (unit.kind == AllocationUnitKind::SerializedRelease) {
      groups[*unit.serializedGroup].releaseEventId = unit.eventId;
      continue;
    }
    if (unit.kind == AllocationUnitKind::PooledRelease) {
      releasePools[*unit.releasePool].releaseEventId = unit.eventId;
    }
    for (unsigned group : unit.releaseGroups) {
      groups[group].releaseEventId = unit.eventId;
    }
    for (CanonicalMechanismId mechanism : unit.mechanisms) {
      if (unit.kind == AllocationUnitKind::RecurringRelease ||
          unit.kind == AllocationUnitKind::PooledRelease) {
        program.setMechanismReleaseEventId(mechanism, unit.eventId);
      } else {
        program.setMechanismEventId(mechanism, unit.eventId);
      }
    }
  }
  program.setScarcityEventPlan(std::move(groups), std::move(releasePools));
}

LogicalResult
emitAllocationFailure(const CanonicalSyncProgram &program,
                      const AllocationFailure &allocationFailure) {
  Operation *witness = allocationFailure.witness
                           ? allocationFailure.witness
                           : program.getFunction().getOperation();
  if (allocationFailure.crossCore) {
    return witness->emitError(
        "canonical sync exhausted cross-core counter IDs");
  }
  witness->emitError("canonical sync exhausted compiler event IDs after "
                     "hidden macro reservations, cut coalescing, serialized "
                     "ready/release fallback, and certified recurring-ready "
                     "or pooled-release reuse")
      << "; domain " << stringifyCanonicalCore(allocationFailure.source.core)
      << ':' << stringifyPIPE(allocationFailure.source.pipe) << " -> "
      << stringifyPIPE(allocationFailure.target.pipe);
  return failure();
}

} // namespace

LogicalResult
mlir::pto::allocateCanonicalSyncEvents(CanonicalSyncProgram &program) {
  if (!program.getSetCoverSolution()) {
    return program.getFunction().emitError(
        "canonical sync event allocation requires a selected cover");
  }
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(program.getFunction());
  if (failed(target)) {
    return failure();
  }

  SmallVector<CanonicalScarcityEventGroup, 2> groups;
  SmallVector<CanonicalRecurringReleasePool, 2> releasePools;
  while (appendRequiredNestedReleasePool(program, groups, releasePools)) {
  }
  const std::size_t retryLimit =
      program.getSetCoverSolution()->mechanisms.size() + 1;
  for (std::size_t attempt = 0; attempt < retryLimit; ++attempt) {
    SmallVector<AllocationUnit, 8> units =
        buildAllocationUnits(program, groups, releasePools);
    AllocationFailure allocationFailure;
    if (allocateUnits(program, *target, units, allocationFailure)) {
      commitAllocation(program, units, std::move(groups),
                       std::move(releasePools));
      return success();
    }
    bool repaired =
        appendCoalescedFallback(program, *target, allocationFailure, groups);
    if (!repaired) {
      repaired =
          appendSerializedFallback(program, *target, allocationFailure, groups);
    }
    if (!repaired) {
      repaired = appendRecurringReleasePoolFallback(
          program, allocationFailure, groups, releasePools);
    }
    if (!repaired) {
      return emitAllocationFailure(program, allocationFailure);
    }
  }
  return program.getFunction().emitError(
      "canonical sync event-scarcity repair did not converge");
}
