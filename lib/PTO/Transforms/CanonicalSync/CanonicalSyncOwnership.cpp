// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

unsigned addAction(CanonicalEvent &event, CanonicalEventActionKind kind,
                   CanonicalEventActionPhase phase, CanonicalAnchor anchor,
                   CanonicalEventLane lane) {
  event.actions.push_back({kind, phase, anchor, lane});
  return event.actions.size() - 1;
}

CanonicalEventLane staticLane(unsigned lane) {
  CanonicalEventLane result;
  result.kind = CanonicalEventLaneKind::Static;
  result.index = lane;
  return result;
}

CanonicalEventLane allLanes() {
  CanonicalEventLane result;
  result.kind = CanonicalEventLaneKind::All;
  return result;
}

void addTrace(CanonicalEvent &event, CanonicalEventTraceKind kind,
              ArrayRef<unsigned> actions, Region *region = nullptr) {
  CanonicalEventTrace trace;
  trace.kind = kind;
  trace.actions.append(actions.begin(), actions.end());
  trace.controlRegion = region;
  event.traces.push_back(std::move(trace));
}

void addStateTrace(CanonicalEvent &event, CanonicalEventTraceKind kind,
                   ArrayRef<unsigned> actions, ArrayRef<unsigned> initial,
                   ArrayRef<unsigned> expected, Region *region = nullptr) {
  CanonicalEventTrace trace;
  trace.kind = kind;
  trace.actions.append(actions.begin(), actions.end());
  trace.controlRegion = region;
  trace.hasExplicitTokenState = true;
  trace.initialTokens.append(initial.begin(), initial.end());
  trace.expectedTokens.append(expected.begin(), expected.end());
  event.traces.push_back(std::move(trace));
}

struct OwnershipActionUse {
  unsigned wait = 0;
  unsigned set = 0;
};

std::size_t firstNodePosition(ArrayRef<std::size_t> nodes) {
  return *llvm::min_element(nodes) * 2;
}

std::size_t lastNodePosition(ArrayRef<std::size_t> nodes) {
  return *llvm::max_element(nodes) * 2 + 1;
}

void addOrderedTrace(CanonicalEvent &event, CanonicalEventTraceKind kind,
                     SmallVectorImpl<std::pair<std::size_t, unsigned>> &actions,
                     Region *region = nullptr) {
  llvm::stable_sort(actions, [](const auto &first, const auto &second) {
    return first.first < second.first;
  });
  SmallVector<unsigned, 16> ordered;
  llvm::transform(actions, std::back_inserter(ordered),
                  [](const auto &entry) { return entry.second; });
  addTrace(event, kind, ordered, region);
}

std::pair<CanonicalEvent, CanonicalEvent>
buildRoundTripOwnershipProtocols(const CanonicalOwnershipCycle &cycle) {
  CanonicalEvent ready;
  ready.sourcePipe = cycle.producerPipe;
  ready.targetPipe = cycle.consumerPipe;
  ready.scopeLoop = cycle.loop;
  ready.width = cycle.lanes.size();
  ready.ownershipCycle = cycle.id;
  ready.ownershipRole = CanonicalOwnershipEventRole::Ready;
  ready.ownershipProtocol = true;

  for (const CanonicalOwnershipPath &path : cycle.paths) {
    SmallVector<std::pair<std::size_t, unsigned>, 16> trace;
    for (const CanonicalOwnershipUse &use : path.uses) {
      const unsigned set = addAction(ready, CanonicalEventActionKind::Set,
                                     CanonicalEventActionPhase::Straight,
                                     use.readyAnchor, staticLane(use.lane));
      const unsigned wait =
          addAction(ready, CanonicalEventActionKind::Wait,
                    CanonicalEventActionPhase::Straight, use.readAcquireAnchor,
                    staticLane(use.lane));
      trace.push_back({lastNodePosition(use.producers), set});
      trace.push_back({firstNodePosition(use.consumers), wait});
      for (std::size_t producer : use.producers) {
        for (std::size_t consumer : use.consumers) {
          ready.completions.push_back(
              {producer, consumer, 0, nullptr, set, wait});
        }
      }
    }
    addOrderedTrace(ready, CanonicalEventTraceKind::Straight, trace,
                    path.region);
  }

  CanonicalEvent release;
  release.sourcePipe = cycle.consumerPipe;
  release.targetPipe = cycle.producerPipe;
  release.recurrenceLoop = cycle.loop;
  release.scopeLoop = cycle.loop;
  release.iterationDistance = 1;
  release.width = cycle.lanes.size();
  release.ownershipCycle = cycle.id;
  release.ownershipRole = CanonicalOwnershipEventRole::Release;
  release.ownershipProtocol = true;
  const unsigned prime = addAction(release, CanonicalEventActionKind::Set,
                                   CanonicalEventActionPhase::Prime,
                                   {cycle.loop, true}, allLanes());
  addTrace(release, CanonicalEventTraceKind::Prime, {prime});

  SmallVector<SmallVector<OwnershipActionUse, 8>, 2> actionPaths;
  for (const CanonicalOwnershipPath &path : cycle.paths) {
    SmallVector<std::pair<std::size_t, unsigned>, 16> trace;
    SmallVector<OwnershipActionUse, 8> actionUses;
    std::vector<std::optional<std::size_t>> previousUse(cycle.lanes.size());
    for (auto [useIndex, use] : llvm::enumerate(path.uses)) {
      const unsigned wait =
          addAction(release, CanonicalEventActionKind::Wait,
                    CanonicalEventActionPhase::Body, use.writeAcquireAnchor,
                    staticLane(use.lane));
      const unsigned set = addAction(release, CanonicalEventActionKind::Set,
                                     CanonicalEventActionPhase::Body,
                                     use.releaseAnchor, staticLane(use.lane));
      trace.push_back({firstNodePosition(use.producers), wait});
      trace.push_back({lastNodePosition(use.consumers), set});
      actionUses.push_back({wait, set});
      if (previousUse[use.lane]) {
        const CanonicalOwnershipUse &previous =
            path.uses[*previousUse[use.lane]];
        const unsigned previousSet = actionUses[*previousUse[use.lane]].set;
        for (std::size_t consumer : previous.consumers) {
          for (std::size_t producer : use.producers) {
            release.completions.push_back(
                {consumer, producer, 0, nullptr, previousSet, wait});
          }
        }
      }
      previousUse[use.lane] = useIndex;
    }
    addOrderedTrace(release, CanonicalEventTraceKind::Cycle, trace,
                    path.region);
    actionPaths.push_back(std::move(actionUses));
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
        if (sourceUse == sourcePath.uses.size() ||
            targetUse == targetPath.uses.size()) {
          continue;
        }
        const CanonicalOwnershipUse &source = sourcePath.uses[sourceUse];
        const CanonicalOwnershipUse &target = targetPath.uses[targetUse];
        const unsigned set = actionPaths[sourcePathIndex][sourceUse].set;
        const unsigned wait = actionPaths[targetPathIndex][targetUse].wait;
        for (std::size_t consumer : source.consumers) {
          for (std::size_t producer : target.producers) {
            release.completions.push_back(
                {consumer, producer, 1, cycle.loop, set, wait});
          }
        }
      }
    }
  }
  const unsigned drain = addAction(release, CanonicalEventActionKind::Wait,
                                   CanonicalEventActionPhase::Drain,
                                   {cycle.loop, false}, allLanes());
  addTrace(release, CanonicalEventTraceKind::Final, {drain});

  const auto setRepresentative =
      [](CanonicalEvent &event, const CanonicalEventCompletion &completion) {
        event.source = completion.source;
        event.target = completion.target;
        event.setAnchor = event.actions[completion.setAction].anchor;
        event.waitAnchor = event.actions[completion.waitAction].anchor;
      };
  setRepresentative(ready, ready.completions.front());
  auto recurrence = llvm::find_if(
      release.completions, [](const CanonicalEventCompletion &completion) {
        return completion.iterationDistance != 0;
      });
  setRepresentative(release, *recurrence);
  return {std::move(ready), std::move(release)};
}

std::pair<CanonicalEvent, CanonicalEvent>
buildAlternatingPrefetchProtocols(const CanonicalOwnershipCycle &cycle) {
  CanonicalEvent ready;
  CanonicalEvent release;
  const bool validShape = cycle.loop && cycle.lanes.size() == 2 &&
                          cycle.paths.size() == 2 &&
                          !cycle.initialProducers.empty() &&
                          cycle.initialReadyAnchor.operation &&
                          cycle.initiallyFreeLanes.size() == 1 &&
                          llvm::all_of(cycle.paths, [](const auto &path) {
                            return path.region && path.uses.size() == 1;
                          });
  if (!validShape) {
    return {std::move(ready), std::move(release)};
  }

  ready.sourcePipe = cycle.producerPipe;
  ready.targetPipe = cycle.consumerPipe;
  ready.recurrenceLoop = cycle.loop;
  ready.scopeLoop = cycle.loop;
  ready.iterationDistance = 1;
  ready.width = cycle.lanes.size();
  ready.ownershipCycle = cycle.id;
  ready.ownershipRole = CanonicalOwnershipEventRole::Ready;
  ready.ownershipProtocol = true;

  const unsigned initialSet = addAction(
      ready, CanonicalEventActionKind::Set, CanonicalEventActionPhase::Prime,
      cycle.initialReadyAnchor, staticLane(cycle.initialReadyLane));
  ready.actions[initialSet].nonEmptyLoopGuard = cycle.loop;
  addStateTrace(ready, CanonicalEventTraceKind::Prime, {initialSet}, {},
                {cycle.initialReadyLane});

  SmallVector<unsigned, 2> readyWaits;
  SmallVector<unsigned, 2> readySets;
  for (const CanonicalOwnershipPath &path : cycle.paths) {
    const CanonicalOwnershipUse &use = path.uses.front();
    const unsigned wait = addAction(
        ready, CanonicalEventActionKind::Wait,
        CanonicalEventActionPhase::Body, use.readAcquireAnchor,
        staticLane(use.lane));
    const unsigned set = addAction(
        ready, CanonicalEventActionKind::Set,
        CanonicalEventActionPhase::Body, use.readyAnchor,
        staticLane(use.producerLane));
    readyWaits.push_back(wait);
    readySets.push_back(set);
    addStateTrace(ready, CanonicalEventTraceKind::Cycle, {wait, set},
                  {use.lane}, {use.producerLane}, path.region);
    addStateTrace(ready, CanonicalEventTraceKind::Final, {wait}, {use.lane},
                  {}, path.region);
  }

  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalOwnershipUse &use = path.uses.front();
    if (use.lane == cycle.initialReadyLane) {
      for (std::size_t producer : cycle.initialProducers) {
        for (std::size_t consumer : use.consumers) {
          ready.completions.push_back(
              {producer, consumer, 0, nullptr, initialSet,
               readyWaits[pathIndex]});
        }
      }
    }
    for (auto [targetIndex, targetPath] : llvm::enumerate(cycle.paths)) {
      const CanonicalOwnershipUse &target = targetPath.uses.front();
      if (target.lane != use.producerLane) {
        continue;
      }
      for (std::size_t producer : use.producers) {
        for (std::size_t consumer : target.consumers) {
          ready.completions.push_back(
              {producer, consumer, 1, cycle.loop, readySets[pathIndex],
               readyWaits[targetIndex]});
        }
      }
    }
  }

  release.sourcePipe = cycle.consumerPipe;
  release.targetPipe = cycle.producerPipe;
  release.recurrenceLoop = cycle.loop;
  release.scopeLoop = cycle.loop;
  release.iterationDistance = 1;
  release.width = cycle.lanes.size();
  release.ownershipCycle = cycle.id;
  release.ownershipRole = CanonicalOwnershipEventRole::Release;
  release.ownershipProtocol = true;

  SmallVector<unsigned, 2> primeActions;
  for (unsigned lane : cycle.initiallyFreeLanes) {
    const unsigned prime = addAction(
        release, CanonicalEventActionKind::Set,
        CanonicalEventActionPhase::Prime, {cycle.loop, true},
        staticLane(lane));
    release.actions[prime].nonEmptyLoopGuard = cycle.loop;
    primeActions.push_back(prime);
  }
  addStateTrace(release, CanonicalEventTraceKind::Prime, primeActions, {},
                cycle.initiallyFreeLanes);

  SmallVector<unsigned, 2> releaseSets;
  SmallVector<unsigned, 2> releaseWaits;
  for (const CanonicalOwnershipPath &path : cycle.paths) {
    const CanonicalOwnershipUse &use = path.uses.front();
    const unsigned set = addAction(
        release, CanonicalEventActionKind::Set,
        CanonicalEventActionPhase::Body, use.releaseAnchor,
        staticLane(use.lane));
    const unsigned wait = addAction(
        release, CanonicalEventActionKind::Wait,
        CanonicalEventActionPhase::Body, use.writeAcquireAnchor,
        staticLane(use.producerLane));
    releaseSets.push_back(set);
    releaseWaits.push_back(wait);
    addStateTrace(release, CanonicalEventTraceKind::Cycle, {set, wait},
                  {use.producerLane}, {use.lane}, path.region);
  }
  const unsigned drain = addAction(
      release, CanonicalEventActionKind::Wait,
      CanonicalEventActionPhase::Drain, {cycle.loop, false}, allLanes());
  release.actions[drain].nonEmptyLoopGuard = cycle.loop;
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalOwnershipUse &use = path.uses.front();
    addStateTrace(release, CanonicalEventTraceKind::Final,
                  {releaseSets[pathIndex], drain}, {use.producerLane}, {});
    for (auto [targetIndex, targetPath] : llvm::enumerate(cycle.paths)) {
      const CanonicalOwnershipUse &target = targetPath.uses.front();
      if (target.producerLane != use.lane) {
        continue;
      }
      for (std::size_t consumer : use.consumers) {
        for (std::size_t producer : target.producers) {
          release.completions.push_back(
              {consumer, producer, 1, cycle.loop, releaseSets[pathIndex],
               releaseWaits[targetIndex]});
        }
      }
    }
  }

  const auto setRepresentative = [](CanonicalEvent &event) {
    auto representative = llvm::find_if(
        event.completions, [&](const CanonicalEventCompletion &completion) {
          return completion.iterationDistance == event.iterationDistance &&
                 completion.recurrenceLoop == event.recurrenceLoop;
        });
    if (representative == event.completions.end()) {
      return;
    }
    const CanonicalEventCompletion &completion = *representative;
    event.source = completion.source;
    event.target = completion.target;
    event.setAnchor = event.actions[completion.setAction].anchor;
    event.waitAnchor = event.actions[completion.waitAction].anchor;
  };
  setRepresentative(ready);
  setRepresentative(release);
  return {std::move(ready), std::move(release)};
}

} // namespace

std::pair<CanonicalEvent, CanonicalEvent>
mlir::pto::buildCanonicalOwnershipProtocols(
    const CanonicalOwnershipCycle &cycle) {
  if (cycle.protocol ==
      CanonicalOwnershipProtocolKind::AlternatingPrefetch) {
    return buildAlternatingPrefetchProtocols(cycle);
  }
  return buildRoundTripOwnershipProtocols(cycle);
}

bool mlir::pto::tryCommitCanonicalOwnershipCandidate(
    std::vector<CanonicalEvent> &acceptedOwnership,
    std::vector<CanonicalBarrier> &currentBarriers,
    std::vector<CanonicalEvent> &currentEvents, CanonicalEvent ready,
    CanonicalEvent release, llvm::function_ref<bool()> evaluate) {
  const std::size_t previousOwnershipSize = acceptedOwnership.size();
  const std::vector<CanonicalBarrier> previousBarriers = currentBarriers;
  const std::vector<CanonicalEvent> previousEvents = currentEvents;
  acceptedOwnership.push_back(std::move(ready));
  acceptedOwnership.push_back(std::move(release));
  if (evaluate()) {
    return true;
  }
  acceptedOwnership.resize(previousOwnershipSize);
  currentBarriers = previousBarriers;
  currentEvents = previousEvents;
  return false;
}

void CanonicalSyncPlanBuilder::synthesizeOwnershipProtocols() {
  const std::vector<CanonicalBarrier> baselineBarriers = plan_.barriers_;
  const std::vector<CanonicalEvent> baselineEvents = plan_.events_;
  std::vector<CanonicalEvent> acceptedOwnership;

  optimizeBarriers();

  for (const CanonicalOwnershipCycle &cycle : plan_.ownershipCycles_) {
    const bool supported = cycle.kind == CanonicalOwnershipKind::L0Operand ||
                           cycle.kind == CanonicalOwnershipKind::L1Tile ||
                           cycle.kind ==
                               CanonicalOwnershipKind::L0Accumulator;
    if (!supported) {
      continue;
    }
    auto [ready, release] = buildCanonicalOwnershipProtocols(cycle);
    const auto sortTraceActions = [&](CanonicalEvent &event) {
      for (CanonicalEventTrace &trace : event.traces) {
        llvm::stable_sort(trace.actions, [&](unsigned first, unsigned second) {
          const std::size_t firstPosition =
              getAnchorPosition(event.actions[first].anchor);
          const std::size_t secondPosition =
              getAnchorPosition(event.actions[second].anchor);
          return firstPosition < secondPosition;
        });
      }
    };
    sortTraceActions(ready);
    sortTraceActions(release);
    deriveEventInterval(ready);
    deriveEventInterval(release);

    if (failed(verifyEventProtocols({ready, release},
                                    /*requireAllocation=*/false,
                                    /*diagnose=*/false))) {
      continue;
    }

    tryCommitCanonicalOwnershipCandidate(
        acceptedOwnership, plan_.barriers_, plan_.events_, std::move(ready),
        std::move(release), [&]() {
          plan_.barriers_ = baselineBarriers;
          plan_.events_ = baselineEvents;
          plan_.events_.insert(plan_.events_.end(), acceptedOwnership.begin(),
                               acceptedOwnership.end());
          optimizeBarriers();
          return eventsFitBudget(plan_.events_) &&
                 succeeded(verifyEventProtocols(
                     plan_.events_, /*requireAllocation=*/false,
                     /*diagnose=*/false)) &&
                 planCoversRequirements(plan_.barriers_, plan_.events_);
        });
  }
}
