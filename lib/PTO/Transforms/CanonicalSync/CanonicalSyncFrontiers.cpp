// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Event-frontier discovery is shared by ordinary candidate synthesis and the
// post-selection scarcity coalescer. It proposes only concrete physical cuts;
// the coverage engine remains responsible for authenticating every ordinary
// candidate before selection.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"

#include <set>
#include <tuple>
#include <unordered_map>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

struct EventFrontierClass {
  CanonicalMechanismKind kind = CanonicalMechanismKind::Event;
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  CanonicalRegionId actionRegion = kInvalidCanonicalSyncId;
  SmallVector<CanonicalControlAtom, 2> guard;
  std::optional<CanonicalRegionId> recurrenceLoop;
  Block *block = nullptr;
  SmallVector<CanonicalMechanismId, 8> members;
};

bool isEligible(const CanonicalMechanism &mechanism) {
  const bool supportedKind =
      mechanism.kind == CanonicalMechanismKind::Event ||
      (mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
       !mechanism.boundaryRecurring);
  if (!supportedKind || !mechanism.sourcePoint.operation ||
      !mechanism.targetPoint.operation ||
      mechanism.sourcePoint.position != CanonicalProgramPointPosition::After ||
      mechanism.targetPoint.position != CanonicalProgramPointPosition::Before) {
    return false;
  }
  return mechanism.sourcePoint.operation->getBlock() ==
             mechanism.targetPoint.operation->getBlock() &&
         programPointMustPrecede(mechanism.sourcePoint, mechanism.targetPoint);
}

std::size_t hashClass(const CanonicalMechanism &mechanism) {
  llvm::hash_code hash = llvm::hash_combine(
      static_cast<unsigned>(mechanism.kind),
      static_cast<unsigned>(mechanism.source.core),
      static_cast<unsigned>(mechanism.source.pipe),
      static_cast<unsigned>(mechanism.target.core),
      static_cast<unsigned>(mechanism.target.pipe), mechanism.actionRegion,
      mechanism.recurrenceLoop.value_or(kInvalidCanonicalSyncId),
      mechanism.sourcePoint.operation->getBlock());
  for (const CanonicalControlAtom &atom : mechanism.guard) {
    hash = llvm::hash_combine(hash, atom.choice, atom.arm);
  }
  return static_cast<std::size_t>(hash);
}

bool sameClass(const EventFrontierClass &frontierClass,
               const CanonicalMechanism &mechanism) {
  return frontierClass.kind == mechanism.kind &&
         frontierClass.source == mechanism.source &&
         frontierClass.target == mechanism.target &&
         frontierClass.actionRegion == mechanism.actionRegion &&
         frontierClass.guard == mechanism.guard &&
         frontierClass.recurrenceLoop == mechanism.recurrenceLoop &&
         frontierClass.block == mechanism.sourcePoint.operation->getBlock();
}

SmallVector<EventFrontierClass, 4>
buildClasses(const CanonicalSyncProgram &program,
             ArrayRef<CanonicalMechanismId> mechanismIds) {
  SmallVector<EventFrontierClass, 4> classes;
  std::unordered_map<std::size_t, SmallVector<std::size_t, 2>> classesByHash;
  for (CanonicalMechanismId mechanismId : mechanismIds) {
    const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
    if (!isEligible(mechanism)) {
      continue;
    }
    SmallVectorImpl<std::size_t> &bucket = classesByHash[hashClass(mechanism)];
    auto existing = llvm::find_if(bucket, [&](std::size_t index) {
      return sameClass(classes[index], mechanism);
    });
    if (existing != bucket.end()) {
      classes[*existing].members.push_back(mechanism.id);
      continue;
    }
    EventFrontierClass frontierClass;
    frontierClass.kind = mechanism.kind;
    frontierClass.source = mechanism.source;
    frontierClass.target = mechanism.target;
    frontierClass.actionRegion = mechanism.actionRegion;
    frontierClass.guard = mechanism.guard;
    frontierClass.recurrenceLoop = mechanism.recurrenceLoop;
    frontierClass.block = mechanism.sourcePoint.operation->getBlock();
    frontierClass.members.push_back(mechanism.id);
    const std::size_t classIndex = classes.size();
    classes.push_back(std::move(frontierClass));
    bucket.push_back(classIndex);
  }
  return classes;
}

void discoverClassFrontiers(
    const CanonicalSyncProgram &program,
    const EventFrontierClass &frontierClass,
    SmallVectorImpl<CanonicalEventFrontier> &frontiers) {
  struct EventWindow {
    CanonicalMechanismId mechanism = kInvalidCanonicalSyncId;
    std::size_t source = 0;
    std::size_t target = 0;
  };

  DenseMap<Operation *, std::size_t> operationOrder;
  std::size_t nextOrder = 0;
  for (Operation &operation : *frontierClass.block) {
    operationOrder[&operation] = nextOrder++;
  }
  SmallVector<EventWindow, 8> bySource;
  SmallVector<EventWindow, 8> byTarget;
  for (CanonicalMechanismId mechanismId : frontierClass.members) {
    const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
    const EventWindow window{mechanism.id,
                             operationOrder[mechanism.sourcePoint.operation],
                             operationOrder[mechanism.targetPoint.operation]};
    bySource.push_back(window);
    byTarget.push_back(window);
  }
  llvm::sort(bySource, [](const EventWindow &left, const EventWindow &right) {
    return std::tie(left.source, left.mechanism) <
           std::tie(right.source, right.mechanism);
  });
  llvm::sort(byTarget, [](const EventWindow &left, const EventWindow &right) {
    return std::tie(left.target, left.mechanism) <
           std::tie(right.target, right.mechanism);
  });

  std::set<std::pair<std::size_t, CanonicalMechanismId>> active;
  std::size_t sourceCursor = 0;
  std::size_t expiredCursor = 0;
  const std::size_t sourceCount = bySource.size();
  const std::size_t targetCount = byTarget.size();
  for (std::size_t pivotCursor = 0; pivotCursor < targetCount;) {
    const std::size_t pivotTarget = byTarget[pivotCursor].target;
    while (sourceCursor < sourceCount &&
           bySource[sourceCursor].source < pivotTarget) {
      active.emplace(bySource[sourceCursor].source,
                     bySource[sourceCursor].mechanism);
      ++sourceCursor;
    }
    while (expiredCursor < targetCount &&
           byTarget[expiredCursor].target < pivotTarget) {
      active.erase(
          {byTarget[expiredCursor].source, byTarget[expiredCursor].mechanism});
      ++expiredCursor;
    }

    const CanonicalMechanism &pivot =
        program.getMechanism(byTarget[pivotCursor].mechanism);
    CanonicalEventFrontier frontier;
    frontier.source = frontierClass.source;
    frontier.target = frontierClass.target;
    frontier.targetPoint = pivot.targetPoint;
    frontier.actionRegion = frontierClass.actionRegion;
    frontier.guard = frontierClass.guard;
    frontier.recurrenceLoop = frontierClass.recurrenceLoop;

    if (!active.empty()) {
      frontier.sourcePoint =
          program.getMechanism(active.rbegin()->second).sourcePoint;
      for (const auto &entry : active) {
        frontier.members.push_back(entry.second);
      }
    }
    const bool useful =
        frontier.members.size() >= 2 &&
        programPointMustPrecede(frontier.sourcePoint, frontier.targetPoint);
    if (useful) {
      frontiers.push_back(std::move(frontier));
    }
    while (pivotCursor < targetCount &&
           byTarget[pivotCursor].target == pivotTarget) {
      ++pivotCursor;
    }
  }
}

} // namespace

SmallVector<CanonicalEventFrontier, 4>
mlir::pto::canonical_sync_detail::discoverCanonicalEventFrontiers(
    const CanonicalSyncProgram &program,
    ArrayRef<CanonicalMechanismId> mechanisms) {
  SmallVector<CanonicalEventFrontier, 4> frontiers;
  // Classes contain every frontier identity component, and each class emits
  // at most one frontier per concrete target point, so no global deduplication
  // pass is required.
  for (const EventFrontierClass &frontierClass :
       buildClasses(program, mechanisms)) {
    discoverClassFrontiers(program, frontierClass, frontiers);
  }
  return frontiers;
}
