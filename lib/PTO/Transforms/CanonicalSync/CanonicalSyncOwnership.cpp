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

struct OwnershipActionUse {
  unsigned wait = 0;
  unsigned set = 0;
};

std::pair<CanonicalEvent, CanonicalEvent>
buildOwnershipProtocols(const CanonicalOwnershipCycle &cycle) {
  CanonicalEvent ready;
  ready.sourcePipe = cycle.producerPipe;
  ready.targetPipe = cycle.consumerPipe;
  ready.scopeLoop = cycle.loop;
  ready.width = cycle.lanes.size();
  ready.ownershipCycle = cycle.id;
  ready.ownershipProtocol = true;

  for (const CanonicalOwnershipPath &path : cycle.paths) {
    SmallVector<unsigned, 16> trace;
    for (const CanonicalOwnershipUse &use : path.uses) {
      const unsigned set = addAction(ready, CanonicalEventActionKind::Set,
                                     CanonicalEventActionPhase::Straight,
                                     use.readyAnchor, staticLane(use.lane));
      const unsigned wait =
          addAction(ready, CanonicalEventActionKind::Wait,
                    CanonicalEventActionPhase::Straight, use.readAcquireAnchor,
                    staticLane(use.lane));
      trace.push_back(set);
      trace.push_back(wait);
      for (std::size_t producer : use.producers) {
        for (std::size_t consumer : use.consumers) {
          ready.completions.push_back(
              {producer, consumer, 0, nullptr, set, wait});
        }
      }
    }
    addTrace(ready, CanonicalEventTraceKind::Straight, trace, path.region);
  }

  CanonicalEvent release;
  release.sourcePipe = cycle.consumerPipe;
  release.targetPipe = cycle.producerPipe;
  release.recurrenceLoop = cycle.loop;
  release.scopeLoop = cycle.loop;
  release.iterationDistance = 1;
  release.width = cycle.lanes.size();
  release.ownershipCycle = cycle.id;
  release.ownershipProtocol = true;
  const unsigned prime = addAction(release, CanonicalEventActionKind::Set,
                                   CanonicalEventActionPhase::Prime,
                                   {cycle.loop, true}, allLanes());
  addTrace(release, CanonicalEventTraceKind::Prime, {prime});

  SmallVector<SmallVector<OwnershipActionUse, 8>, 2> actionPaths;
  for (const CanonicalOwnershipPath &path : cycle.paths) {
    SmallVector<unsigned, 16> trace;
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
      trace.push_back(wait);
      trace.push_back(set);
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
    addTrace(release, CanonicalEventTraceKind::Cycle, trace, path.region);
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

} // namespace

void CanonicalSyncPlanBuilder::synthesizeOwnershipProtocols() {
  for (const CanonicalOwnershipCycle &cycle : plan_.ownershipCycles_) {
    if (cycle.kind != CanonicalOwnershipKind::L0Operand) {
      continue;
    }
    auto [ready, release] = buildOwnershipProtocols(cycle);
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

    std::vector<CanonicalEvent> trial = plan_.events_;
    trial.push_back(ready);
    trial.push_back(release);
    if (failed(verifyEventProtocols({ready, release},
                                    /*requireAllocation=*/false,
                                    /*diagnose=*/false))) {
      continue;
    }
    std::vector<CanonicalEvent> selected =
        selectRequiredEvents(plan_.barriers_, trial);
    if (!eventsFitBudget(selected)) {
      continue;
    }
    plan_.events_.push_back(std::move(ready));
    plan_.events_.push_back(std::move(release));
  }
}
