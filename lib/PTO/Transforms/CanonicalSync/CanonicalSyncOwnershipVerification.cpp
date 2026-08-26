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

#include <map>
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
         !action.lane.selector &&
         action.guard.kind == CanonicalEventExecutionGuardKind::None &&
         !action.guard.loop;
}

bool isGuardedAction(const CanonicalEventAction &action,
                     CanonicalEventActionKind kind,
                     CanonicalEventActionPhase phase, Operation *loop,
                     CanonicalEventLaneKind laneKind, unsigned lane = 0) {
  return action.kind == kind && action.phase == phase &&
         action.guard.kind ==
             CanonicalEventExecutionGuardKind::LoopNonEmpty &&
         action.guard.loop == loop &&
         action.anchor.operation == loop &&
         action.anchor.before == (phase == CanonicalEventActionPhase::Prime) &&
         action.lane.kind == laneKind &&
         (laneKind != CanonicalEventLaneKind::Static ||
          action.lane.index == lane) &&
         !action.lane.selector;
}

bool matchesBoundaryAction(const CanonicalEventAction &action,
                           CanonicalEventActionKind kind,
                           const CanonicalAnchor &anchor, unsigned lane,
                           CanonicalEventExecutionGuardKind guard) {
  return action.kind == kind &&
         action.phase == CanonicalEventActionPhase::Body &&
         anchorsEqual(action.anchor, anchor) &&
         action.lane.kind == CanonicalEventLaneKind::Static &&
         action.lane.index == lane && !action.lane.selector &&
         action.guard.kind == guard && action.guard.loop != nullptr;
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
                           event.ownershipProtocolKind == cycle.protocol &&
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

bool hasValidBoundaryGuardedLifecycle(
    const CanonicalOwnershipCycle &cycle, const CanonicalEvent &ready,
    const CanonicalEvent &release) {
  auto loop = dyn_cast_or_null<scf::ForOp>(cycle.loop);
  const bool validCycle =
      cycle.kind == CanonicalOwnershipKind::L0Accumulator && loop &&
      !cycle.lanes.empty() &&
      cycle.paths.size() == 1 &&
      cycle.paths.front().region == &loop.getRegion() &&
      cycle.paths.front().uses.size() == cycle.lanes.size();
  const bool validReleaseShape =
      release.ownershipProtocol && release.ownershipCycle == cycle.id &&
      release.ownershipProtocolKind == cycle.protocol &&
      release.ownershipRole == CanonicalOwnershipEventRole::Release &&
      release.sourcePipe == cycle.consumerPipe &&
      release.targetPipe == cycle.producerPipe &&
      release.recurrenceLoop == cycle.loop && release.scopeLoop == cycle.loop &&
      release.iterationDistance == 1 &&
      release.width == cycle.lanes.size() && !release.forwardDrainLoop &&
      release.actions.size() == cycle.lanes.size() * 2;
  const bool invalidLifecycle =
      !validCycle || !matchesCycle(ready, cycle) || !validReleaseShape;
  if (invalidLifecycle) {
    return false;
  }

  SmallVector<CanonicalEventCompletion, 8> expectedCompletions;
  SmallVector<bool, 8> seenLanes(cycle.lanes.size(), false);
  for (auto [useIndex, use] :
       llvm::enumerate(cycle.paths.front().uses)) {
    const bool invalidUse =
        use.lane >= cycle.lanes.size() || seenLanes[use.lane];
    if (invalidUse) {
      return false;
    }
    seenLanes[use.lane] = true;
    const unsigned wait = useIndex * 2;
    const unsigned set = wait + 1;
    const bool validActions =
        matchesBoundaryAction(
            release.actions[wait], CanonicalEventActionKind::Wait,
            use.writeAcquireAnchor, use.lane,
            CanonicalEventExecutionGuardKind::NotFirstIteration) &&
        matchesBoundaryAction(
            release.actions[set], CanonicalEventActionKind::Set,
            use.releaseAnchor, use.lane,
            CanonicalEventExecutionGuardKind::HasSuccessor);
    if (!validActions) {
      return false;
    }
    for (std::size_t consumer : use.consumers) {
      for (std::size_t producer : use.producers) {
        expectedCompletions.push_back(
            {consumer, producer, 1, cycle.loop, set, wait});
      }
    }
  }
  return llvm::all_of(seenLanes, [](bool seen) { return seen; }) &&
         completionsEqual(release.completions, expectedCompletions);
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
         first.guard.kind == second.guard.kind &&
         first.guard.loop == second.guard.loop;
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
         first.guard.kind == second.guard.kind &&
         first.guard.loop == second.guard.loop &&
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
      actual.resourceScopeLoop == expected.resourceScopeLoop &&
      actual.iterationDistance == expected.iterationDistance &&
      actual.width == expected.width &&
      actual.ownershipCycle == expected.ownershipCycle &&
      actual.ownershipProtocolKind == expected.ownershipProtocolKind &&
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

bool slotsEqual(const CanonicalPhysicalSlot &first,
                const CanonicalPhysicalSlot &second) {
  return first.space == second.space && first.address == second.address &&
         first.size == second.size;
}

bool rangesOverlap(std::uint64_t address, std::uint64_t size,
                   const CanonicalPhysicalSlot &slot) {
  if (size == 0 || slot.size == 0) {
    return false;
  }
  if (address <= slot.address) {
    return slot.address - address < size;
  }
  return address - slot.address < slot.size;
}

struct ExpectedManagedAccess {
  CanonicalPhysicalSlot slot;
  bool reads = false;
  bool writes = false;
};

bool ownsEveryManagedL1Access(
    ArrayRef<const CanonicalOwnershipCycle *> cycles, scf::ForOp outer,
    ArrayRef<CanonicalSyncNode> nodes) {
  SmallVector<CanonicalPhysicalSlot, 8> managedSlots;
  std::map<std::size_t, SmallVector<ExpectedManagedAccess, 4>> expected;
  const auto addExpected = [&](const CanonicalOwnershipCycle &cycle,
                               unsigned lane,
                               ArrayRef<std::size_t> nodeIds, bool reads,
                               bool writes) {
    if (lane >= cycle.lanes.size()) {
      return false;
    }
    for (std::size_t node : nodeIds) {
      SmallVector<ExpectedManagedAccess, 4> &accesses = expected[node];
      for (const CanonicalPhysicalSlot &slot : cycle.lanes[lane].slots) {
        if (slot.space != AddressSpace::MAT || slot.size == 0) {
          return false;
        }
        auto known = llvm::find_if(accesses, [&](const auto &access) {
          return slotsEqual(access.slot, slot);
        });
        if (known == accesses.end()) {
          accesses.push_back({slot, reads, writes});
        } else {
          known->reads |= reads;
          known->writes |= writes;
        }
      }
    }
    return true;
  };

  for (const CanonicalOwnershipCycle *cycle : cycles) {
    if (!cycle) {
      return false;
    }
    for (const CanonicalOwnershipLane &lane : cycle->lanes) {
      managedSlots.append(lane.slots.begin(), lane.slots.end());
    }
    const bool invalidInitial =
        !cycle->initialProducers.empty() &&
        !addExpected(*cycle, cycle->initialReadyLane,
                     cycle->initialProducers, /*reads=*/false,
                     /*writes=*/true);
    if (invalidInitial) {
      return false;
    }
    for (const CanonicalOwnershipPath &path : cycle->paths) {
      for (const CanonicalOwnershipUse &use : path.uses) {
        const bool invalidUse =
            !addExpected(*cycle, use.producerLane, use.producers,
                         /*reads=*/false, /*writes=*/true) ||
            !addExpected(*cycle, use.lane, use.consumers,
                         /*reads=*/true, /*writes=*/false);
        if (invalidUse) {
          return false;
        }
      }
    }
  }

  std::map<std::size_t, SmallVector<bool, 4>> observed;
  for (const auto &[node, accesses] : expected) {
    const bool invalidExpectedNode =
        node >= nodes.size() || accesses.empty();
    if (invalidExpectedNode) {
      return false;
    }
    observed[node].resize(accesses.size(), false);
  }

  for (const CanonicalSyncNode &node : nodes) {
    if (!node.operation || !outer->isAncestor(node.operation)) {
      continue;
    }
    for (const CanonicalMemoryAccess &access : node.accesses) {
      if (access.space != AddressSpace::MAT) {
        continue;
      }
      if (!access.knownPhysical || access.unknownRange || access.size == 0 ||
          access.addresses.empty()) {
        return false;
      }
      for (std::uint64_t address : access.addresses) {
        const bool touchesManaged =
            llvm::any_of(managedSlots, [&](const auto &slot) {
              return rangesOverlap(address, access.size, slot);
            });
        if (!touchesManaged) {
          continue;
        }
        auto expectedNode = expected.find(node.id);
        if (expectedNode == expected.end()) {
          return false;
        }
        auto expectedAccess =
            llvm::find_if(expectedNode->second, [&](const auto &entry) {
              return entry.slot.space == access.space &&
                     entry.slot.address == address &&
                     entry.slot.size == access.size &&
                     (!entry.reads || access.reads) &&
                     (!entry.writes || access.writes);
        });
        if (expectedAccess == expectedNode->second.end()) {
          return false;
        }
        const std::size_t index =
            static_cast<std::size_t>(expectedAccess -
                                     expectedNode->second.begin());
        observed[node.id][index] = true;
      }
    }
  }
  return llvm::all_of(observed, [](const auto &entry) {
    return llvm::all_of(entry.second, [](bool value) { return value; });
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
          CanonicalOwnershipProtocolKind::BoundaryGuardedRoundTrip &&
      !hasValidBoundaryGuardedLifecycle(cycle, *ready, *release)) {
    return false;
  }
  if (cycle.protocol ==
          CanonicalOwnershipProtocolKind::AlternatingPrefetch &&
      !hasValidAlternatingLifecycle(cycle, *ready, *release)) {
    return false;
  }
  if (cycle.protocol ==
          CanonicalOwnershipProtocolKind::HierarchicalOuterCarry &&
      cycle.kind != CanonicalOwnershipKind::L1Tile) {
    return false;
  }
  auto [expectedReady, expectedRelease] =
      buildCanonicalOwnershipProtocols(cycle);
  return eventsEqual(*ready, expectedReady) &&
         eventsEqual(*release, expectedRelease);
}

bool mlir::pto::verifyCanonicalCompositeOwnershipBundle(
    const CanonicalEventBundleCandidate &bundle,
    ArrayRef<CanonicalOwnershipCycle> cycles,
    ArrayRef<CanonicalSyncNode> nodes) {
  const bool invalidBundle =
      bundle.kind != CanonicalEventBundleKind::CompositeOwnership ||
      bundle.events.size() != 6;
  if (invalidBundle) {
    return false;
  }
  std::map<std::size_t, SmallVector<const CanonicalEvent *, 2>> grouped;
  for (const CanonicalEvent &event : bundle.events) {
    if (!event.ownershipProtocol || event.ownershipCycle == 0) {
      return false;
    }
    grouped[event.ownershipCycle].push_back(&event);
  }
  const bool wrongGroupCount = grouped.size() != 3;
  if (wrongGroupCount) {
    return false;
  }

  SmallVector<const CanonicalOwnershipCycle *, 2> l1Cycles;
  const CanonicalOwnershipCycle *accumulator = nullptr;
  for (const auto &[cycleId, events] : grouped) {
    auto cycle = llvm::find_if(cycles, [&](const auto &candidate) {
      return candidate.id == cycleId;
    });
    const bool invalidCycle =
        cycle == cycles.end() || events.size() != 2 ||
        events[0]->ownershipProtocolKind !=
            events[1]->ownershipProtocolKind;
    if (invalidCycle) {
      return false;
    }
    CanonicalOwnershipCycle protocolCycle = *cycle;
    protocolCycle.protocol = events.front()->ownershipProtocolKind;
    if (!verifyCanonicalOwnershipEventPair(protocolCycle, events)) {
      return false;
    }
    if (cycle->kind == CanonicalOwnershipKind::L1Tile &&
        protocolCycle.protocol ==
            CanonicalOwnershipProtocolKind::HierarchicalOuterCarry) {
      l1Cycles.push_back(&*cycle);
      continue;
    }
    if (cycle->kind == CanonicalOwnershipKind::L0Accumulator &&
        protocolCycle.protocol ==
            CanonicalOwnershipProtocolKind::BoundaryGuardedRoundTrip &&
        !accumulator) {
      accumulator = &*cycle;
      continue;
    }
    return false;
  }
  const bool invalidComposition =
      l1Cycles.size() != 2 || !accumulator ||
      l1Cycles[0]->loop != l1Cycles[1]->loop;
  if (invalidComposition) {
    return false;
  }
  auto inner = dyn_cast_or_null<scf::ForOp>(l1Cycles[0]->loop);
  auto outer = inner ? inner->getParentOfType<scf::ForOp>() : scf::ForOp{};
  const bool oneAlternating =
      l1Cycles[0]->initialProducers.empty() !=
      l1Cycles[1]->initialProducers.empty();
  if (!outer || accumulator->loop != outer || !oneAlternating) {
    return false;
  }
  if (!ownsEveryManagedL1Access(l1Cycles, outer, nodes)) {
    return false;
  }

  const auto slotsOverlap = [](const CanonicalPhysicalSlot &first,
                               const CanonicalPhysicalSlot &second) {
    if (first.space != second.space || first.size == 0 || second.size == 0) {
      return false;
    }
    if (first.address <= second.address) {
      return second.address - first.address < first.size;
    }
    return first.address - second.address < second.size;
  };
  for (const CanonicalOwnershipLane &first : l1Cycles[0]->lanes) {
    for (const CanonicalOwnershipLane &second : l1Cycles[1]->lanes) {
      for (const CanonicalPhysicalSlot &firstSlot : first.slots) {
        for (const CanonicalPhysicalSlot &secondSlot : second.slots) {
          if (slotsOverlap(firstSlot, secondSlot)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}
