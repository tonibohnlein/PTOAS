// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <vector>

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
         !action.lane.selector && !action.nonEmptyLoopGuard;
}

bool isGuardedAction(const CanonicalEventAction &action,
                     CanonicalEventActionKind kind,
                     CanonicalEventActionPhase phase, Operation *loop,
                     CanonicalEventLaneKind laneKind, unsigned lane = 0) {
  return action.kind == kind && action.phase == phase &&
         action.nonEmptyLoopGuard == loop &&
         action.anchor.operation == loop &&
         action.anchor.before == (phase == CanonicalEventActionPhase::Prime) &&
         action.lane.kind == laneKind &&
         (laneKind != CanonicalEventLaneKind::Static ||
          action.lane.index == lane) &&
         !action.lane.selector;
}

bool hasValidAlternatingLifecycle(const CanonicalOwnershipCycle &cycle,
                                  const CanonicalEvent &ready,
                                  const CanonicalEvent &release) {
  const bool commonShape =
      cycle.loop && isa<scf::ForOp>(cycle.loop) && cycle.lanes.size() == 2 &&
      cycle.paths.size() == 2 && !cycle.initialProducers.empty() &&
      cycle.initialReadyAnchor.operation == cycle.loop &&
      cycle.initialReadyAnchor.before && cycle.initiallyFreeLanes.size() == 1 &&
      llvm::all_of(cycle.paths, [](const CanonicalOwnershipPath &path) {
        return path.region && path.uses.size() == 1;
      });
  if (!commonShape) {
    return false;
  }
  const CanonicalOwnershipUse &first = cycle.paths[0].uses.front();
  const CanonicalOwnershipUse &second = cycle.paths[1].uses.front();
  const bool alternates =
      first.lane == cycle.initialReadyLane &&
      cycle.initiallyFreeLanes.front() == first.producerLane &&
      first.lane != first.producerLane && second.lane == first.producerLane &&
      second.producerLane == first.lane;
  const bool hasExpectedActionCounts =
      ready.actions.size() == 5 && release.actions.size() == 6;
  if (!alternates || !hasExpectedActionCounts) {
    return false;
  }
  const bool validReadyPrime = isGuardedAction(
      ready.actions[0], CanonicalEventActionKind::Set,
      CanonicalEventActionPhase::Prime, cycle.loop,
      CanonicalEventLaneKind::Static, cycle.initialReadyLane);
  const bool validReleasePrime = isGuardedAction(
      release.actions[0], CanonicalEventActionKind::Set,
      CanonicalEventActionPhase::Prime, cycle.loop,
      CanonicalEventLaneKind::Static, cycle.initiallyFreeLanes.front());
  const bool validDrain = isGuardedAction(
      release.actions.back(), CanonicalEventActionKind::Wait,
      CanonicalEventActionPhase::Drain, cycle.loop,
      CanonicalEventLaneKind::All);
  if (!validReadyPrime || !validReleasePrime || !validDrain) {
    return false;
  }
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalOwnershipUse &use = path.uses.front();
    const unsigned readyAction = 1 + pathIndex * 2;
    const unsigned releaseAction = 1 + pathIndex * 2;
    const bool validReadyBody =
        matchesAction(ready.actions[readyAction],
                      CanonicalEventActionKind::Wait,
                      CanonicalEventActionPhase::Body, use.readAcquireAnchor,
                      CanonicalEventLaneKind::Static, use.lane) &&
        matchesAction(ready.actions[readyAction + 1],
                      CanonicalEventActionKind::Set,
                      CanonicalEventActionPhase::Body, use.readyAnchor,
                      CanonicalEventLaneKind::Static, use.producerLane);
    const bool validReleaseBody =
        matchesAction(release.actions[releaseAction],
                      CanonicalEventActionKind::Set,
                      CanonicalEventActionPhase::Body, use.releaseAnchor,
                      CanonicalEventLaneKind::Static, use.lane) &&
        matchesAction(release.actions[releaseAction + 1],
                      CanonicalEventActionKind::Wait,
                      CanonicalEventActionPhase::Body,
                      use.writeAcquireAnchor,
                      CanonicalEventLaneKind::Static, use.producerLane);
    if (!validReadyBody || !validReleaseBody) {
      return false;
    }
  }
  return true;
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

bool completionsEqual(ArrayRef<CanonicalEventCompletion> actual,
                      ArrayRef<CanonicalEventCompletion> expected) {
  const bool sameSize = actual.size() == expected.size();
  if (!sameSize) {
    return false;
  }
  SmallVector<bool, 16> matched(actual.size(), false);
  auto indexedActual = llvm::enumerate(actual);
  for (const CanonicalEventCompletion &candidate : expected) {
    auto match = llvm::find_if(indexedActual, [&](const auto &entry) {
      const CanonicalEventCompletion &completion = entry.value();
      return !matched[entry.index()] &&
             completion.source == candidate.source &&
             completion.target == candidate.target &&
             completion.iterationDistance == candidate.iterationDistance &&
             completion.recurrenceLoop == candidate.recurrenceLoop &&
             completion.setAction == candidate.setAction &&
             completion.waitAction == candidate.waitAction;
    });
    if (match == indexedActual.end()) {
      return false;
    }
    matched[(*match).index()] = true;
  }
  return true;
}

SmallVector<CanonicalEventCompletion, 16>
getExpectedReadyCompletions(const CanonicalOwnershipCycle &cycle) {
  SmallVector<CanonicalEventCompletion, 16> expected;
  unsigned action = 0;
  for (const CanonicalOwnershipPath &path : cycle.paths) {
    for (const CanonicalOwnershipUse &use : path.uses) {
      for (std::size_t producer : use.producers) {
        for (std::size_t consumer : use.consumers) {
          expected.push_back({producer, consumer, 0, nullptr, action,
                              action + 1});
        }
      }
      action += 2;
    }
  }
  return expected;
}

SmallVector<CanonicalEventCompletion, 16>
getExpectedReleaseCompletions(const CanonicalOwnershipCycle &cycle) {
  SmallVector<CanonicalEventCompletion, 16> expected;
  SmallVector<SmallVector<unsigned, 8>, 2> waitActions;
  SmallVector<SmallVector<unsigned, 8>, 2> setActions;
  unsigned action = 1;
  for (const CanonicalOwnershipPath &path : cycle.paths) {
    SmallVector<unsigned, 8> waits;
    SmallVector<unsigned, 8> sets;
    std::vector<std::optional<std::size_t>> previousUse(cycle.lanes.size());
    for (auto [useIndex, use] : llvm::enumerate(path.uses)) {
      const unsigned wait = action++;
      const unsigned set = action++;
      waits.push_back(wait);
      sets.push_back(set);
      if (previousUse[use.lane]) {
        const std::size_t previousIndex = *previousUse[use.lane];
        const CanonicalOwnershipUse &previous = path.uses[previousIndex];
        for (std::size_t consumer : previous.consumers) {
          for (std::size_t producer : use.producers) {
            expected.push_back({consumer, producer, 0, nullptr,
                                sets[previousIndex], wait});
          }
        }
      }
      previousUse[use.lane] = useIndex;
    }
    waitActions.push_back(std::move(waits));
    setActions.push_back(std::move(sets));
  }

  for (auto [sourcePathIndex, sourcePath] : llvm::enumerate(cycle.paths)) {
    for (auto [targetPathIndex, targetPath] : llvm::enumerate(cycle.paths)) {
      for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
        std::size_t sourceUse = sourcePath.uses.size();
        std::size_t targetUse = targetPath.uses.size();
        for (std::size_t index = 0; index < sourcePath.uses.size(); ++index) {
          if (sourcePath.uses[index].lane == lane) {
            sourceUse = index;
          }
        }
        for (std::size_t index = 0; index < targetPath.uses.size(); ++index) {
          if (targetPath.uses[index].lane == lane) {
            targetUse = index;
            break;
          }
        }
        const bool foundUses = sourceUse != sourcePath.uses.size() &&
                               targetUse != targetPath.uses.size();
        if (!foundUses) {
          continue;
        }
        const CanonicalOwnershipUse &source = sourcePath.uses[sourceUse];
        const CanonicalOwnershipUse &target = targetPath.uses[targetUse];
        for (std::size_t consumer : source.consumers) {
          for (std::size_t producer : target.producers) {
            expected.push_back({consumer, producer, 1, cycle.loop,
                                setActions[sourcePathIndex][sourceUse],
                                waitActions[targetPathIndex][targetUse]});
          }
        }
      }
    }
  }
  return expected;
}

bool hasValidCompletions(const CanonicalEvent &event,
                         const CanonicalOwnershipCycle &cycle) {
  if (event.ownershipRole == CanonicalOwnershipEventRole::Ready) {
    return completionsEqual(event.completions,
                            getExpectedReadyCompletions(cycle));
  }
  return completionsEqual(event.completions,
                          getExpectedReleaseCompletions(cycle));
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

bool lanesEqual(const CanonicalEventLane &first,
                const CanonicalEventLane &second) {
  return first.kind == second.kind && first.index == second.index &&
         first.selector == second.selector;
}

bool actionsEqual(const CanonicalEventAction &first,
                  const CanonicalEventAction &second) {
  return first.kind == second.kind && first.phase == second.phase &&
         anchorsEqual(first.anchor, second.anchor) &&
         lanesEqual(first.lane, second.lane) &&
         first.nonEmptyLoopGuard == second.nonEmptyLoopGuard;
}

bool completionsEqual(const CanonicalEventCompletion &first,
                      const CanonicalEventCompletion &second) {
  return first.source == second.source && first.target == second.target &&
         first.iterationDistance == second.iterationDistance &&
         first.recurrenceLoop == second.recurrenceLoop &&
         first.setAction == second.setAction &&
         first.waitAction == second.waitAction;
}

bool tracesEqual(const CanonicalEventTrace &first,
                 const CanonicalEventTrace &second) {
  return first.kind == second.kind && first.actions == second.actions &&
         first.controlRegion == second.controlRegion &&
         first.hasExplicitTokenState == second.hasExplicitTokenState &&
         first.initialTokens == second.initialTokens &&
         first.expectedTokens == second.expectedTokens;
}

bool eventsEqual(const CanonicalEvent &actual,
                 const CanonicalEvent &expected) {
  const bool common =
      actual.source == expected.source && actual.target == expected.target &&
      actual.sourcePipe == expected.sourcePipe &&
      actual.targetPipe == expected.targetPipe &&
      anchorsEqual(actual.setAnchor, expected.setAnchor) &&
      anchorsEqual(actual.waitAnchor, expected.waitAnchor) &&
      actual.recurrenceLoop == expected.recurrenceLoop &&
      actual.forwardDrainLoop == expected.forwardDrainLoop &&
      actual.scopeLoop == expected.scopeLoop &&
      actual.iterationDistance == expected.iterationDistance &&
      actual.width == expected.width &&
      actual.protocolBundle == expected.protocolBundle &&
      actual.ownershipCycle == expected.ownershipCycle &&
      actual.ownershipRole == expected.ownershipRole &&
      actual.ownershipProtocol == expected.ownershipProtocol &&
      actual.actions.size() == expected.actions.size() &&
      actual.completions.size() == expected.completions.size() &&
      actual.traces.size() == expected.traces.size();
  if (!common) {
    return false;
  }
  return llvm::all_of(llvm::zip(actual.actions, expected.actions),
                      [](const auto &entry) {
                        return actionsEqual(std::get<0>(entry),
                                            std::get<1>(entry));
                      }) &&
         llvm::all_of(llvm::zip(actual.completions, expected.completions),
                      [](const auto &entry) {
                        return completionsEqual(std::get<0>(entry),
                                                std::get<1>(entry));
                      }) &&
         llvm::all_of(llvm::zip(actual.traces, expected.traces),
                      [](const auto &entry) {
                        return tracesEqual(std::get<0>(entry),
                                           std::get<1>(entry));
                      });
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
  if (!ready || !release) {
    return false;
  }
  if (cycle.protocol == CanonicalOwnershipProtocolKind::RoundTrip &&
      (!matchesCycle(*ready, cycle) || !matchesCycle(*release, cycle))) {
    return false;
  }
  if (cycle.protocol ==
          CanonicalOwnershipProtocolKind::AlternatingPrefetch &&
      !hasValidAlternatingLifecycle(cycle, *ready, *release)) {
    return false;
  }
  auto [expectedReady, expectedRelease] =
      buildCanonicalOwnershipProtocols(cycle);
  return eventsEqual(*ready, expectedReady) &&
         eventsEqual(*release, expectedRelease);
}
