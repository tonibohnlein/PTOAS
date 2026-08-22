// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

bool anchorsEqual(const CanonicalAnchor &first,
                  const CanonicalAnchor &second) {
  return first.operation == second.operation && first.before == second.before;
}

bool matchesAction(const CanonicalEventAction &action,
                   CanonicalEventActionKind kind,
                   CanonicalEventActionPhase phase,
                   const CanonicalAnchor &anchor,
                   CanonicalEventLaneKind laneKind, unsigned lane = 0) {
  return action.kind == kind && action.phase == phase &&
         anchorsEqual(action.anchor, anchor) && action.lane.kind == laneKind &&
         (laneKind != CanonicalEventLaneKind::Static ||
          action.lane.index == lane) &&
         !action.lane.selector;
}

bool hasReadyLifecycle(const CanonicalEvent &event,
                       const CanonicalOwnershipCycle &cycle) {
  std::size_t actionIndex = 0;
  for (const CanonicalOwnershipPath &path : cycle.paths) {
    for (const CanonicalOwnershipUse &use : path.uses) {
      if (actionIndex + 2 > event.actions.size()) {
        return false;
      }
      const bool validSet = matchesAction(
          event.actions[actionIndex], CanonicalEventActionKind::Set,
          CanonicalEventActionPhase::Straight, use.readyAnchor,
          CanonicalEventLaneKind::Static, use.lane);
      const bool validWait = matchesAction(
          event.actions[actionIndex + 1], CanonicalEventActionKind::Wait,
          CanonicalEventActionPhase::Straight, use.readAcquireAnchor,
          CanonicalEventLaneKind::Static, use.lane);
      if (!validSet || !validWait) {
        return false;
      }
      actionIndex += 2;
    }
  }
  return actionIndex == event.actions.size();
}

bool hasReleaseLifecycle(const CanonicalEvent &event,
                         const CanonicalOwnershipCycle &cycle) {
  const std::size_t minimumActions = 2;
  const bool hasMinimumActions = event.actions.size() >= minimumActions;
  if (!hasMinimumActions) {
    return false;
  }
  const bool validPrime = matchesAction(
      event.actions.front(), CanonicalEventActionKind::Set,
      CanonicalEventActionPhase::Prime, {cycle.loop, true},
      CanonicalEventLaneKind::All);
  if (!validPrime) {
    return false;
  }
  std::size_t actionIndex = 1;
  for (const CanonicalOwnershipPath &path : cycle.paths) {
    for (const CanonicalOwnershipUse &use : path.uses) {
      if (actionIndex + 2 >= event.actions.size()) {
        return false;
      }
      const bool validWait = matchesAction(
          event.actions[actionIndex], CanonicalEventActionKind::Wait,
          CanonicalEventActionPhase::Body, use.writeAcquireAnchor,
          CanonicalEventLaneKind::Static, use.lane);
      const bool validSet = matchesAction(
          event.actions[actionIndex + 1], CanonicalEventActionKind::Set,
          CanonicalEventActionPhase::Body, use.releaseAnchor,
          CanonicalEventLaneKind::Static, use.lane);
      if (!validWait || !validSet) {
        return false;
      }
      actionIndex += 2;
    }
  }
  const bool hasOneDrain = actionIndex + 1 == event.actions.size();
  if (!hasOneDrain) {
    return false;
  }
  return matchesAction(event.actions.back(), CanonicalEventActionKind::Wait,
                       CanonicalEventActionPhase::Drain, {cycle.loop, false},
                       CanonicalEventLaneKind::All);
}

bool cycleContainsNode(const CanonicalOwnershipCycle &cycle, std::size_t node,
                       bool producer, unsigned lane) {
  return llvm::any_of(cycle.paths, [&](const CanonicalOwnershipPath &path) {
    return llvm::any_of(path.uses, [&](const CanonicalOwnershipUse &use) {
      if (use.lane != lane) {
        return false;
      }
      const ArrayRef<std::size_t> nodes =
          producer ? ArrayRef<std::size_t>(use.producers)
                   : ArrayRef<std::size_t>(use.consumers);
      return llvm::is_contained(nodes, node);
    });
  });
}

bool readyCompletionMatchesUse(const CanonicalOwnershipCycle &cycle,
                               const CanonicalEventCompletion &completion,
                               unsigned lane) {
  return llvm::any_of(cycle.paths, [&](const CanonicalOwnershipPath &path) {
    return llvm::any_of(path.uses, [&](const CanonicalOwnershipUse &use) {
      return use.lane == lane &&
             llvm::is_contained(use.producers, completion.source) &&
             llvm::is_contained(use.consumers, completion.target);
    });
  });
}

bool hasValidCompletions(const CanonicalEvent &event,
                         const CanonicalOwnershipCycle &cycle) {
  const bool ready =
      event.ownershipRole == CanonicalOwnershipEventRole::Ready;
  bool hasRecurrence = false;
  for (const CanonicalEventCompletion &completion : event.completions) {
    const bool validActionIndices =
        completion.setAction < event.actions.size() &&
        completion.waitAction < event.actions.size();
    if (!validActionIndices) {
      return false;
    }
    const CanonicalEventAction &set = event.actions[completion.setAction];
    const CanonicalEventAction &wait = event.actions[completion.waitAction];
    const bool hasStaticLane =
        set.lane.kind == CanonicalEventLaneKind::Static &&
        wait.lane.kind == CanonicalEventLaneKind::Static &&
        set.lane.index == wait.lane.index;
    if (!hasStaticLane) {
      return false;
    }
    const unsigned lane = set.lane.index;
    if (ready) {
      const bool matchesUse =
          readyCompletionMatchesUse(cycle, completion, lane);
      if (!matchesUse || completion.iterationDistance != 0 ||
          completion.recurrenceLoop) {
        return false;
      }
      continue;
    }
    const bool validSource =
        cycleContainsNode(cycle, completion.source, false, lane);
    const bool validTarget =
        cycleContainsNode(cycle, completion.target, true, lane);
    if (!validSource || !validTarget) {
      return false;
    }
    const bool sameIteration = completion.iterationDistance == 0 &&
                               !completion.recurrenceLoop;
    const bool nextIteration = completion.iterationDistance == 1 &&
                               completion.recurrenceLoop == cycle.loop;
    if (!sameIteration && !nextIteration) {
      return false;
    }
    hasRecurrence = hasRecurrence || nextIteration;
  }
  return ready || hasRecurrence;
}

bool matchesCycle(const CanonicalEvent &event,
                  const CanonicalOwnershipCycle &cycle) {
  const bool commonShape = event.ownershipProtocol &&
                           event.ownershipCycle == cycle.id &&
                           event.scopeLoop == cycle.loop &&
                           event.width == cycle.lanes.size() &&
                           !event.forwardDrainLoop;
  if (!commonShape) {
    return false;
  }
  if (event.ownershipRole == CanonicalOwnershipEventRole::Ready) {
    return event.sourcePipe == cycle.producerPipe &&
           event.targetPipe == cycle.consumerPipe && !event.recurrenceLoop &&
           event.iterationDistance == 0 && hasReadyLifecycle(event, cycle) &&
           hasValidCompletions(event, cycle);
  }
  if (event.ownershipRole == CanonicalOwnershipEventRole::Release) {
    return event.sourcePipe == cycle.consumerPipe &&
           event.targetPipe == cycle.producerPipe &&
           event.recurrenceLoop == cycle.loop &&
           event.iterationDistance == 1 &&
           hasReleaseLifecycle(event, cycle) &&
           hasValidCompletions(event, cycle);
  }
  return false;
}

} // namespace

bool mlir::pto::verifyCanonicalOwnershipEventPair(
    const CanonicalOwnershipCycle &cycle,
    ArrayRef<const CanonicalEvent *> events) {
  const bool hasPair = events.size() == 2;
  if (!hasPair) {
    return false;
  }
  const CanonicalEvent *ready = nullptr;
  const CanonicalEvent *release = nullptr;
  for (const CanonicalEvent *event : events) {
    if (!event) {
      return false;
    }
    if (event->ownershipRole == CanonicalOwnershipEventRole::Ready) {
      if (ready) {
        return false;
      }
      ready = event;
    } else if (event->ownershipRole ==
               CanonicalOwnershipEventRole::Release) {
      if (release) {
        return false;
      }
      release = event;
    }
  }
  return ready && release && matchesCycle(*ready, cycle) &&
         matchesCycle(*release, cycle);
}
