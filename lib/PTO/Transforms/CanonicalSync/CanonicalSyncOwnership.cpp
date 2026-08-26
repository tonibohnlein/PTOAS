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

unsigned addBoundaryAction(CanonicalEvent &event,
                           CanonicalEventActionKind kind,
                           CanonicalAnchor anchor, CanonicalEventLane lane,
                           CanonicalEventExecutionGuardKind guard) {
  const unsigned action =
      addAction(event, kind, CanonicalEventActionPhase::Body, anchor, lane);
  event.actions[action].guard = {guard, event.recurrenceLoop};
  return action;
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
                   ArrayRef<unsigned> expected, Region *region = nullptr,
                   CanonicalEventExecutionGuard guard = {}) {
  CanonicalEventTrace trace;
  trace.kind = kind;
  trace.actions.append(actions.begin(), actions.end());
  trace.controlRegion = region;
  trace.guard = guard;
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
  ready.ownershipProtocolKind = cycle.protocol;
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
  release.ownershipProtocolKind = cycle.protocol;
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
buildBoundaryGuardedRoundTripProtocols(
    const CanonicalOwnershipCycle &cycle) {
  CanonicalEvent ready;
  CanonicalEvent release;
  auto loop = dyn_cast_or_null<scf::ForOp>(cycle.loop);
  const bool validShape = cycle.kind == CanonicalOwnershipKind::L0Accumulator &&
                          loop &&
                          !cycle.lanes.empty() && cycle.paths.size() == 1 &&
                          cycle.paths.front().region == &loop.getRegion() &&
                          cycle.paths.front().uses.size() ==
                              cycle.lanes.size();
  if (!validShape) {
    return {std::move(ready), std::move(release)};
  }

  auto roundTrip = buildRoundTripOwnershipProtocols(cycle);
  ready = std::move(roundTrip.first);

  release.sourcePipe = cycle.consumerPipe;
  release.targetPipe = cycle.producerPipe;
  release.recurrenceLoop = cycle.loop;
  release.scopeLoop = cycle.loop;
  release.iterationDistance = 1;
  release.width = cycle.lanes.size();
  release.ownershipCycle = cycle.id;
  release.ownershipProtocolKind = cycle.protocol;
  release.ownershipRole = CanonicalOwnershipEventRole::Release;
  release.ownershipProtocol = true;

  SmallVector<std::pair<std::size_t, unsigned>, 16> firstActions;
  SmallVector<std::pair<std::size_t, unsigned>, 16> middleActions;
  SmallVector<std::pair<std::size_t, unsigned>, 16> finalActions;
  SmallVector<unsigned, 8> lanes;
  for (const CanonicalOwnershipUse &use : cycle.paths.front().uses) {
    const bool invalidUse =
        use.lane >= cycle.lanes.size() || use.producers.empty() ||
        use.consumers.empty() || llvm::is_contained(lanes, use.lane);
    if (invalidUse) {
      return {CanonicalEvent{}, CanonicalEvent{}};
    }
    lanes.push_back(use.lane);
    const unsigned wait = addBoundaryAction(
        release, CanonicalEventActionKind::Wait, use.writeAcquireAnchor,
        staticLane(use.lane),
        CanonicalEventExecutionGuardKind::NotFirstIteration);
    const unsigned set = addBoundaryAction(
        release, CanonicalEventActionKind::Set, use.releaseAnchor,
        staticLane(use.lane),
        CanonicalEventExecutionGuardKind::HasSuccessor);
    firstActions.push_back({lastNodePosition(use.consumers), set});
    middleActions.push_back({firstNodePosition(use.producers), wait});
    middleActions.push_back({lastNodePosition(use.consumers), set});
    finalActions.push_back({firstNodePosition(use.producers), wait});
    for (std::size_t consumer : use.consumers) {
      for (std::size_t producer : use.producers) {
        release.completions.push_back(
            {consumer, producer, 1, cycle.loop, set, wait});
      }
    }
  }
  llvm::sort(lanes);
  const bool hasEveryLane =
      lanes.size() == cycle.lanes.size() &&
      llvm::all_of(llvm::enumerate(lanes), [](const auto &entry) {
        return entry.index() == entry.value();
      });
  if (!hasEveryLane) {
    return {CanonicalEvent{}, CanonicalEvent{}};
  }

  const auto orderActions = [](auto &actions) {
    llvm::stable_sort(actions, [](const auto &first, const auto &second) {
      return first.first < second.first;
    });
    SmallVector<unsigned, 16> ordered;
    llvm::transform(actions, std::back_inserter(ordered),
                    [](const auto &entry) { return entry.second; });
    return ordered;
  };
  const SmallVector<unsigned, 16> first = orderActions(firstActions);
  const SmallVector<unsigned, 16> middle = orderActions(middleActions);
  const SmallVector<unsigned, 16> final = orderActions(finalActions);
  addStateTrace(release, CanonicalEventTraceKind::Prime, {}, {}, {});
  addStateTrace(release, CanonicalEventTraceKind::Cycle, first, {}, lanes,
                cycle.paths.front().region);
  addStateTrace(release, CanonicalEventTraceKind::Cycle, middle, lanes, lanes,
                cycle.paths.front().region);
  addStateTrace(release, CanonicalEventTraceKind::Final, final, lanes, {},
                cycle.paths.front().region);

  const CanonicalEventCompletion &representative =
      release.completions.front();
  release.source = representative.source;
  release.target = representative.target;
  release.setAnchor = release.actions[representative.setAction].anchor;
  release.waitAnchor = release.actions[representative.waitAction].anchor;
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
  ready.ownershipProtocolKind = cycle.protocol;
  ready.ownershipRole = CanonicalOwnershipEventRole::Ready;
  ready.ownershipProtocol = true;

  const unsigned initialSet = addAction(
      ready, CanonicalEventActionKind::Set, CanonicalEventActionPhase::Prime,
      cycle.initialReadyAnchor, staticLane(cycle.initialReadyLane));
  ready.actions[initialSet].guard =
      {CanonicalEventExecutionGuardKind::LoopNonEmpty, cycle.loop};
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
  release.ownershipProtocolKind = cycle.protocol;
  release.ownershipRole = CanonicalOwnershipEventRole::Release;
  release.ownershipProtocol = true;

  SmallVector<unsigned, 2> primeActions;
  for (unsigned lane : cycle.initiallyFreeLanes) {
    const unsigned prime = addAction(
        release, CanonicalEventActionKind::Set,
        CanonicalEventActionPhase::Prime, {cycle.loop, true},
        staticLane(lane));
    release.actions[prime].guard =
        {CanonicalEventExecutionGuardKind::LoopNonEmpty, cycle.loop};
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
  release.actions[drain].guard =
      {CanonicalEventExecutionGuardKind::LoopNonEmpty, cycle.loop};
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

scf::ForOp getOuterLifecycleLoop(const CanonicalOwnershipCycle &cycle) {
  auto inner = dyn_cast_or_null<scf::ForOp>(cycle.loop);
  return inner ? inner->getParentOfType<scf::ForOp>() : scf::ForOp{};
}

void stampHierarchicalProtocol(CanonicalEvent &event) {
  event.ownershipProtocolKind =
      CanonicalOwnershipProtocolKind::HierarchicalOuterCarry;
}

std::pair<CanonicalEvent, CanonicalEvent>
buildHierarchicalStableProtocols(const CanonicalOwnershipCycle &cycle,
                                 scf::ForOp outer) {
  CanonicalOwnershipCycle baseCycle = cycle;
  baseCycle.protocol = CanonicalOwnershipProtocolKind::RoundTrip;
  auto [ready, release] = buildRoundTripOwnershipProtocols(baseCycle);
  stampHierarchicalProtocol(ready);
  stampHierarchicalProtocol(release);
  release.scopeLoop = outer;
  release.resourceScopeLoop = outer;
  release.actions.front().anchor = {outer, true};
  release.actions.back().anchor = {outer, false};

  SmallVector<SmallVector<unsigned, 8>, 2> waits;
  SmallVector<SmallVector<unsigned, 8>, 2> sets;
  unsigned action = 1;
  for (const CanonicalOwnershipPath &path : cycle.paths) {
    SmallVector<unsigned, 8> pathWaits;
    SmallVector<unsigned, 8> pathSets;
    for (std::size_t useIndex = 0; useIndex < path.uses.size(); ++useIndex) {
      pathWaits.push_back(action++);
      pathSets.push_back(action++);
    }
    waits.push_back(std::move(pathWaits));
    sets.push_back(std::move(pathSets));
  }
  if (cycle.paths.empty()) {
    return {CanonicalEvent{}, CanonicalEvent{}};
  }
  const CanonicalOwnershipPath &firstPath = cycle.paths.front();
  for (auto [sourcePathIndex, sourcePath] : llvm::enumerate(cycle.paths)) {
    for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
      std::optional<std::size_t> sourceUse;
      std::optional<std::size_t> targetUse;
      for (auto [useIndex, use] : llvm::enumerate(sourcePath.uses)) {
        if (use.lane == lane) {
          sourceUse = useIndex;
        }
      }
      for (auto [useIndex, use] : llvm::enumerate(firstPath.uses)) {
        if (use.lane == lane) {
          targetUse = useIndex;
          break;
        }
      }
      if (!sourceUse || !targetUse) {
        return {CanonicalEvent{}, CanonicalEvent{}};
      }
      const CanonicalOwnershipUse &source = sourcePath.uses[*sourceUse];
      const CanonicalOwnershipUse &target = firstPath.uses[*targetUse];
      for (std::size_t consumer : source.consumers) {
        for (std::size_t producer : target.producers) {
          release.completions.push_back(
              {consumer, producer, 1, outer,
               sets[sourcePathIndex][*sourceUse], waits.front()[*targetUse]});
        }
      }
    }
  }
  return {std::move(ready), std::move(release)};
}

std::pair<CanonicalEvent, CanonicalEvent>
buildHierarchicalAlternatingProtocols(const CanonicalOwnershipCycle &cycle,
                                      scf::ForOp outer) {
  CanonicalOwnershipCycle baseCycle = cycle;
  baseCycle.protocol = CanonicalOwnershipProtocolKind::AlternatingPrefetch;
  auto baseProtocols = buildAlternatingPrefetchProtocols(baseCycle);
  CanonicalEvent ready = std::move(baseProtocols.first);
  const bool invalidReady =
      ready.actions.empty() || !cycle.initialWriteAcquireAnchor.operation ||
      cycle.initialReadyLane >= cycle.lanes.size();
  if (invalidReady) {
    return {CanonicalEvent{}, CanonicalEvent{}};
  }
  stampHierarchicalProtocol(ready);
  ready.resourceScopeLoop = outer;
  ready.actions.front().guard = {};
  const unsigned emptyReadyWait = addAction(
      ready, CanonicalEventActionKind::Wait,
      CanonicalEventActionPhase::Body, {cycle.loop, true},
      staticLane(cycle.initialReadyLane));
  ready.actions[emptyReadyWait].guard =
      {CanonicalEventExecutionGuardKind::LoopEmpty, cycle.loop};
  addStateTrace(ready, CanonicalEventTraceKind::Cycle, {emptyReadyWait},
                {cycle.initialReadyLane}, {}, nullptr,
                {CanonicalEventExecutionGuardKind::LoopEmpty, cycle.loop});

  CanonicalEvent release;
  release.sourcePipe = cycle.consumerPipe;
  release.targetPipe = cycle.producerPipe;
  release.recurrenceLoop = cycle.loop;
  release.scopeLoop = outer;
  release.resourceScopeLoop = outer;
  release.iterationDistance = 1;
  release.width = cycle.lanes.size();
  release.ownershipCycle = cycle.id;
  release.ownershipProtocolKind =
      CanonicalOwnershipProtocolKind::HierarchicalOuterCarry;
  release.ownershipRole = CanonicalOwnershipEventRole::Release;
  release.ownershipProtocol = true;

  const unsigned prime = addAction(
      release, CanonicalEventActionKind::Set,
      CanonicalEventActionPhase::Prime, {outer, true}, allLanes());
  const unsigned initialWait = addAction(
      release, CanonicalEventActionKind::Wait,
      CanonicalEventActionPhase::Body, cycle.initialWriteAcquireAnchor,
      staticLane(cycle.initialReadyLane));
  SmallVector<unsigned, 8> all;
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    all.push_back(lane);
  }
  addStateTrace(release, CanonicalEventTraceKind::Prime, {prime}, {}, all);
  addStateTrace(release, CanonicalEventTraceKind::Cycle, {initialWait}, all,
                cycle.initiallyFreeLanes, nullptr,
                {CanonicalEventExecutionGuardKind::LoopNonEmpty, cycle.loop});

  SmallVector<unsigned, 2> sets;
  SmallVector<unsigned, 2> waits;
  for (const CanonicalOwnershipPath &path : cycle.paths) {
    const bool invalidPath = path.uses.size() != 1;
    if (invalidPath) {
      return {CanonicalEvent{}, CanonicalEvent{}};
    }
    const CanonicalOwnershipUse &use = path.uses.front();
    const unsigned set = addAction(
        release, CanonicalEventActionKind::Set,
        CanonicalEventActionPhase::Body, use.releaseAnchor,
        staticLane(use.lane));
    const unsigned wait = addAction(
        release, CanonicalEventActionKind::Wait,
        CanonicalEventActionPhase::Body, use.writeAcquireAnchor,
        staticLane(use.producerLane));
    sets.push_back(set);
    waits.push_back(wait);
    addStateTrace(release, CanonicalEventTraceKind::Cycle, {set, wait},
                  {use.producerLane}, {use.lane}, path.region);
  }
  const unsigned emptyReleaseSet = addAction(
      release, CanonicalEventActionKind::Set,
      CanonicalEventActionPhase::Body, {cycle.loop, false},
      staticLane(cycle.initialReadyLane));
  release.actions[emptyReleaseSet].guard =
      {CanonicalEventExecutionGuardKind::LoopEmpty, cycle.loop};
  addStateTrace(release, CanonicalEventTraceKind::Cycle,
                {initialWait, emptyReleaseSet}, all, all, nullptr,
                {CanonicalEventExecutionGuardKind::LoopEmpty, cycle.loop});
  const unsigned drain = addAction(
      release, CanonicalEventActionKind::Wait,
      CanonicalEventActionPhase::Drain, {outer, false}, allLanes());
  addStateTrace(release, CanonicalEventTraceKind::Final, {drain}, all, {});
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalOwnershipUse &use = path.uses.front();
    addStateTrace(release, CanonicalEventTraceKind::Final,
                  {sets[pathIndex], drain}, {use.producerLane}, {});
  }

  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalOwnershipUse &use = path.uses.front();
    for (auto [targetIndex, targetPath] : llvm::enumerate(cycle.paths)) {
      const CanonicalOwnershipUse &target = targetPath.uses.front();
      if (target.producerLane != use.lane) {
        continue;
      }
      for (std::size_t consumer : use.consumers) {
        for (std::size_t producer : target.producers) {
          release.completions.push_back(
              {consumer, producer, 1, cycle.loop, sets[pathIndex],
               waits[targetIndex]});
        }
      }
    }
  }
  auto initialLanePath = llvm::find_if(
      cycle.paths, [&](const CanonicalOwnershipPath &path) {
        return path.uses.front().lane == cycle.initialReadyLane;
      });
  if (initialLanePath == cycle.paths.end()) {
    return {CanonicalEvent{}, CanonicalEvent{}};
  }
  const std::size_t sourcePath =
      static_cast<std::size_t>(initialLanePath - cycle.paths.begin());
  for (std::size_t consumer : initialLanePath->uses.front().consumers) {
    for (std::size_t producer : cycle.initialProducers) {
      release.completions.push_back(
          {consumer, producer, 1, outer, sets[sourcePath], initialWait});
    }
  }

  auto representative = llvm::find_if(
      release.completions, [&](const CanonicalEventCompletion &completion) {
        return completion.recurrenceLoop == cycle.loop;
      });
  if (representative == release.completions.end()) {
    return {CanonicalEvent{}, CanonicalEvent{}};
  }
  release.source = representative->source;
  release.target = representative->target;
  release.setAnchor = release.actions[representative->setAction].anchor;
  release.waitAnchor = release.actions[representative->waitAction].anchor;
  return {std::move(ready), std::move(release)};
}

std::pair<CanonicalEvent, CanonicalEvent>
buildHierarchicalOuterCarryProtocols(const CanonicalOwnershipCycle &cycle) {
  const scf::ForOp outer = getOuterLifecycleLoop(cycle);
  const bool alternating = !cycle.initialProducers.empty();
  CanonicalOwnershipCycle baseCycle = cycle;
  baseCycle.protocol = alternating
                           ? CanonicalOwnershipProtocolKind::AlternatingPrefetch
                           : CanonicalOwnershipProtocolKind::RoundTrip;
  const bool validShape = cycle.kind == CanonicalOwnershipKind::L1Tile &&
                          outer &&
                          (!alternating ||
                           verifyCanonicalAlternatingPathMapping(baseCycle));
  if (!validShape) {
    return {CanonicalEvent{}, CanonicalEvent{}};
  }
  if (alternating) {
    return buildHierarchicalAlternatingProtocols(cycle, outer);
  }
  return buildHierarchicalStableProtocols(cycle, outer);
}

} // namespace

std::pair<CanonicalEvent, CanonicalEvent>
mlir::pto::buildCanonicalOwnershipProtocols(
    const CanonicalOwnershipCycle &cycle) {
  if (cycle.protocol ==
      CanonicalOwnershipProtocolKind::BoundaryGuardedRoundTrip) {
    return buildBoundaryGuardedRoundTripProtocols(cycle);
  }
  if (cycle.protocol ==
      CanonicalOwnershipProtocolKind::AlternatingPrefetch) {
    return buildAlternatingPrefetchProtocols(cycle);
  }
  if (cycle.protocol ==
      CanonicalOwnershipProtocolKind::HierarchicalOuterCarry) {
    return buildHierarchicalOuterCarryProtocols(cycle);
  }
  return buildRoundTripOwnershipProtocols(cycle);
}

std::optional<CanonicalEventBundleCandidate>
CanonicalSyncPlanBuilder::buildOwnershipEventBundle(
    const CanonicalOwnershipCycle &cycle,
    CanonicalOwnershipProtocolKind protocol) {
  const bool supported = cycle.kind == CanonicalOwnershipKind::L0Operand ||
                         cycle.kind == CanonicalOwnershipKind::L1Tile ||
                         cycle.kind ==
                             CanonicalOwnershipKind::L0Accumulator;
  if (!supported) {
    return std::nullopt;
  }
  CanonicalOwnershipCycle protocolCycle = cycle;
  protocolCycle.protocol = protocol;
  auto [ready, release] = buildCanonicalOwnershipProtocols(protocolCycle);
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
    return std::nullopt;
  }

  CanonicalEventBundleCandidate bundle;
  bundle.kind = CanonicalEventBundleKind::Ownership;
  bundle.protocolIdentity = cycle.id;
  bundle.ownershipProtocol = protocol;
  bundle.events.push_back(std::move(ready));
  bundle.events.push_back(std::move(release));
  return bundle;
}

std::optional<CanonicalEventBundleCandidate>
CanonicalSyncPlanBuilder::buildCompositeOwnershipEventBundle() {
  std::optional<CanonicalEventBundleCandidate> result;
  for (const CanonicalOwnershipCycle &stable : plan_.ownershipCycles_) {
    const bool stableL1 =
        stable.kind == CanonicalOwnershipKind::L1Tile &&
        stable.protocol == CanonicalOwnershipProtocolKind::RoundTrip &&
        stable.initialProducers.empty();
    if (!stableL1) {
      continue;
    }
    for (const CanonicalOwnershipCycle &alternating :
         plan_.ownershipCycles_) {
      const bool alternatingL1 =
          alternating.kind == CanonicalOwnershipKind::L1Tile &&
          alternating.protocol ==
              CanonicalOwnershipProtocolKind::AlternatingPrefetch &&
          alternating.loop == stable.loop;
      if (!alternatingL1) {
        continue;
      }
      const scf::ForOp outer = getOuterLifecycleLoop(stable);
      if (!outer) {
        continue;
      }
      for (const CanonicalOwnershipCycle &accumulator :
           plan_.ownershipCycles_) {
        const bool matchingAccumulator =
            outer &&
            accumulator.kind == CanonicalOwnershipKind::L0Accumulator &&
            accumulator.protocol ==
                CanonicalOwnershipProtocolKind::RoundTrip &&
            accumulator.loop == outer;
        if (!matchingAccumulator) {
          continue;
        }
        auto stableBundle = buildOwnershipEventBundle(
            stable,
            CanonicalOwnershipProtocolKind::HierarchicalOuterCarry);
        auto alternatingBundle = buildOwnershipEventBundle(
            alternating,
            CanonicalOwnershipProtocolKind::HierarchicalOuterCarry);
        auto accumulatorBundle = buildOwnershipEventBundle(
            accumulator,
            CanonicalOwnershipProtocolKind::BoundaryGuardedRoundTrip);
        if (!stableBundle || !alternatingBundle || !accumulatorBundle) {
          continue;
        }
        CanonicalEventBundleCandidate composite;
        composite.kind = CanonicalEventBundleKind::CompositeOwnership;
        composite.events.append(stableBundle->events.begin(),
                                stableBundle->events.end());
        composite.events.append(alternatingBundle->events.begin(),
                                alternatingBundle->events.end());
        composite.events.append(accumulatorBundle->events.begin(),
                                accumulatorBundle->events.end());
        if (!verifyCanonicalCompositeOwnershipBundle(
                composite, plan_.ownershipCycles_, plan_.nodes_)) {
          continue;
        }
        if (result) {
          return std::nullopt;
        }
        result = std::move(composite);
      }
    }
  }
  return result;
}
