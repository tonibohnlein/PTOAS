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

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

enum class AllocationUnitKind : std::uint8_t {
  Ready,
  RecurringRelease,
  SerializedReady,
  SerializedRelease,
  CrossCore,
};

struct AllocationUnit {
  AllocationUnitKind kind = AllocationUnitKind::Ready;
  SmallVector<CanonicalMechanismId, 2> mechanisms;
  std::optional<unsigned> serializedGroup;
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  SmallVector<CanonicalControlAtom, 2> guard;
  std::optional<CanonicalRegionId> recurrenceLoop;
  unsigned eventId = 0;
  SmallVector<unsigned, 2> releaseGroups;
};

struct AllocationFailure {
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  Operation *witness = nullptr;
  bool crossCore = false;
};

bool isUnconditionalLifecycle(const AllocationUnit &unit) {
  return unit.kind == AllocationUnitKind::RecurringRelease;
}

bool unitsInterfere(const AllocationUnit &first, const AllocationUnit &second) {
  return isUnconditionalLifecycle(first) || isUnconditionalLifecycle(second) ||
         controlsCanCoexecute(first.guard, second.guard);
}

bool idAvailable(const AllocationUnit &candidate, unsigned eventId,
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
                          unitsInterfere(candidate, existing);
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
                     ArrayRef<CanonicalScarcityEventGroup> groups) {
  const CanonicalSetCoverSolution &solution = *program.getSetCoverSolution();
  SmallVector<AllocationUnit, 8> units;
  for (CanonicalMechanismId mechanismId : solution.mechanisms) {
    const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
    if (mechanism.kind == CanonicalMechanismKind::Event) {
      if (!isScarcityMember(groups, mechanism.id)) {
        units.push_back({AllocationUnitKind::Ready,
                         {mechanism.id},
                         std::nullopt,
                         mechanism.source,
                         mechanism.target,
                         mechanism.guard,
                         std::nullopt,
                         0});
      }
      continue;
    }
    if (mechanism.kind == CanonicalMechanismKind::CrossCoreEvent) {
      units.push_back({AllocationUnitKind::CrossCore,
                       {mechanism.id},
                       std::nullopt,
                       mechanism.source,
                       mechanism.target,
                       mechanism.guard,
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
                     mechanism.source,
                     mechanism.target,
                     mechanism.guard,
                     mechanism.recurrenceLoop,
                     0});
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
                       mechanism.target,
                       mechanism.source,
                       {},
                       mechanism.recurrenceLoop,
                       0});
    } else {
      release->mechanisms.push_back(mechanism.id);
    }
  }
  for (auto [index, group] : llvm::enumerate(groups)) {
    units.push_back({AllocationUnitKind::SerializedReady,
                     {},
                     static_cast<unsigned>(index),
                     group.source,
                     group.target,
                     group.guard,
                     std::nullopt,
                     0});
    if (group.kind == CanonicalScarcityEventKind::Serialized) {
      units.push_back({AllocationUnitKind::SerializedRelease,
                       {},
                       static_cast<unsigned>(index),
                       group.target,
                       group.source,
                       group.guard,
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
                          group.target,
                          group.source,
                          {},
                          group.recurrenceLoop,
                          0};
      unit.releaseGroups.push_back(static_cast<unsigned>(index));
      units.push_back(std::move(unit));
    } else {
      release->releaseGroups.push_back(static_cast<unsigned>(index));
    }
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
             idAvailable(unit, eventId, assigned);
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

void commitAllocation(CanonicalSyncProgram &program,
                      ArrayRef<AllocationUnit> units,
                      SmallVector<CanonicalScarcityEventGroup, 2> groups) {
  for (const AllocationUnit &unit : units) {
    if (unit.kind == AllocationUnitKind::SerializedReady) {
      groups[*unit.serializedGroup].eventId = unit.eventId;
      continue;
    }
    if (unit.kind == AllocationUnitKind::SerializedRelease) {
      groups[*unit.serializedGroup].releaseEventId = unit.eventId;
      continue;
    }
    for (unsigned group : unit.releaseGroups) {
      groups[group].releaseEventId = unit.eventId;
    }
    for (CanonicalMechanismId mechanism : unit.mechanisms) {
      if (unit.kind == AllocationUnitKind::RecurringRelease) {
        program.setMechanismReleaseEventId(mechanism, unit.eventId);
      } else {
        program.setMechanismEventId(mechanism, unit.eventId);
      }
    }
  }
  program.setScarcityEventGroups(std::move(groups));
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
                     "hidden macro reservations, cut coalescing, and "
                     "serialized ready/release fallback")
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
  const std::size_t retryLimit =
      program.getSetCoverSolution()->mechanisms.size() + 1;
  for (std::size_t attempt = 0; attempt < retryLimit; ++attempt) {
    SmallVector<AllocationUnit, 8> units =
        buildAllocationUnits(program, groups);
    AllocationFailure allocationFailure;
    if (allocateUnits(program, *target, units, allocationFailure)) {
      commitAllocation(program, units, std::move(groups));
      return success();
    }
    bool repaired =
        appendCoalescedFallback(program, *target, allocationFailure, groups);
    if (!repaired) {
      repaired =
          appendSerializedFallback(program, *target, allocationFailure, groups);
    }
    if (!repaired) {
      return emitAllocationFailure(program, allocationFailure);
    }
  }
  return program.getFunction().emitError(
      "canonical sync event-scarcity repair did not converge");
}
