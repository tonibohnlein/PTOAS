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
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr std::size_t kBarrierBeamWidth = 8;
constexpr std::size_t kBarrierMaxBundleSize = 3;

struct BarrierSearchState {
  SmallVector<std::size_t, 3> additions;
  std::size_t uncovered = std::numeric_limits<std::size_t>::max();
  std::size_t intervalSpan = 0;
};

struct BarrierEventBundle {
  SmallVector<CanonicalEvent, 2> events;
  std::size_t id = 0;
};

using OwnershipEventPairs =
    std::map<std::size_t, SmallVector<const CanonicalEvent *, 2>>;

OwnershipEventPairs buildOwnershipEventPairs(ArrayRef<CanonicalEvent> events) {
  OwnershipEventPairs pairs;
  for (const CanonicalEvent &event : events) {
    if (event.ownershipProtocol) {
      pairs[event.ownershipCycle].push_back(&event);
    }
  }
  return pairs;
}

const CanonicalEvent *
findOwnershipEvent(ArrayRef<const CanonicalEvent *> events,
                   CanonicalOwnershipEventRole role) {
  auto match = llvm::find_if(events, [&](const CanonicalEvent *event) {
    return event && event->ownershipRole == role;
  });
  return match == events.end() ? nullptr : *match;
}

bool isInRegion(Operation *operation, Region *region) {
  if (!operation || !region) {
    return false;
  }
  return operation->getParentRegion() == region ||
         llvm::any_of(region->getOps(), [&](Operation &candidate) {
           return candidate.isAncestor(operation);
         });
}

bool ownershipCycleContainsNode(const CanonicalOwnershipCycle &cycle,
                                std::size_t node) {
  return llvm::any_of(cycle.paths, [&](const CanonicalOwnershipPath &path) {
    return llvm::any_of(path.uses, [&](const CanonicalOwnershipUse &use) {
      return llvm::is_contained(use.producers, node) ||
             llvm::is_contained(use.consumers, node);
    });
  });
}

bool ownershipCycleContainsAccess(const CanonicalOwnershipCycle &cycle,
                                  const CanonicalMemoryAccess &access) {
  if (!access.knownPhysical || access.unknownRange || access.size == 0 ||
      access.addresses.size() != 1) {
    return false;
  }
  const CanonicalPhysicalSlot slot{access.space, access.addresses.front(),
                                   access.size};
  return llvm::any_of(cycle.lanes, [&](const CanonicalOwnershipLane &lane) {
    return llvm::is_contained(lane.slots, slot);
  });
}

bool formsMemoryHazard(CanonicalDependencyKind kind,
                       const CanonicalMemoryAccess &source,
                       const CanonicalMemoryAccess &target) {
  switch (kind) {
  case CanonicalDependencyKind::MemoryRAW:
    return source.writes && target.reads;
  case CanonicalDependencyKind::MemoryWAR:
    return source.reads && target.writes;
  case CanonicalDependencyKind::MemoryWAW:
    return source.writes && target.writes;
  case CanonicalDependencyKind::SSA:
  case CanonicalDependencyKind::LoopCarriedSSA:
    return false;
  }
  return false;
}

SmallVector<const CanonicalOwnershipCycle *, 2>
getVerifiedAlternatingCycles(ArrayRef<CanonicalOwnershipCycle> cycles,
                             ArrayRef<CanonicalEvent> events,
                             Operation *loop = nullptr) {
  const OwnershipEventPairs pairs = buildOwnershipEventPairs(events);
  SmallVector<const CanonicalOwnershipCycle *, 2> verified;
  for (const CanonicalOwnershipCycle &cycle : cycles) {
    if (cycle.protocol != CanonicalOwnershipProtocolKind::AlternatingPrefetch ||
        (loop && cycle.loop != loop) ||
        !verifyCanonicalAlternatingPathMapping(cycle)) {
      continue;
    }
    auto pair = pairs.find(cycle.id);
    const bool validPair =
        pair != pairs.end() &&
        verifyCanonicalOwnershipEventPair(cycle, pair->second);
    if (validPair) {
      verified.push_back(&cycle);
    }
  }
  return verified;
}

void appendAlternatingOwnershipForwardEdges(
    ArrayRef<CanonicalOwnershipCycle> cycles, ArrayRef<CanonicalEvent> events,
    std::size_t nodeCount, std::set<std::pair<std::size_t, std::size_t>> &seen,
    std::vector<SyncGraphEdge> &edges) {
  const OwnershipEventPairs pairs = buildOwnershipEventPairs(events);
  for (const CanonicalOwnershipCycle *cycle :
       getVerifiedAlternatingCycles(cycles, events)) {
    auto pair = pairs.find(cycle->id);
    if (pair == pairs.end()) {
      continue;
    }
    const CanonicalEvent *ready =
        findOwnershipEvent(pair->second, CanonicalOwnershipEventRole::Ready);
    const CanonicalEvent *release =
        findOwnershipEvent(pair->second, CanonicalOwnershipEventRole::Release);
    if (!ready || !release) {
      continue;
    }

    // Compose the initial ready handoff with the first release handoff. This
    // summarizes a preheader producer -> first physical-slot reuse path that
    // spans two event domains and therefore is not represented by either
    // event's individual completion edge.
    for (const CanonicalEventCompletion &initial : ready->completions) {
      if (initial.iterationDistance != 0 || initial.recurrenceLoop ||
          !llvm::is_contained(cycle->initialProducers, initial.source)) {
        continue;
      }
      for (const CanonicalEventCompletion &reuse : release->completions) {
        if (reuse.iterationDistance != 1 ||
            reuse.recurrenceLoop != cycle->loop ||
            initial.target != reuse.source || initial.source >= reuse.target ||
            reuse.target >= nodeCount) {
          continue;
        }
        if (seen.emplace(initial.source, reuse.target).second) {
          edges.push_back({initial.source, reuse.target,
                           SyncGraphEdgeKind::HardwareCompletion});
        }
      }
    }
  }
}

bool sameEventProtocol(const CanonicalEvent &first,
                       const CanonicalEvent &second) {
  if (first.sourcePipe != second.sourcePipe ||
      first.targetPipe != second.targetPipe || first.width != second.width ||
      first.scopeLoop != second.scopeLoop ||
      first.resourceScopeLoop != second.resourceScopeLoop ||
      first.ownershipCycle != second.ownershipCycle ||
      first.ownershipProtocolKind != second.ownershipProtocolKind ||
      first.ownershipRole != second.ownershipRole ||
      first.actions.size() != second.actions.size() ||
      first.completions.size() != second.completions.size() ||
      first.traces.size() != second.traces.size()) {
    return false;
  }
  for (auto [left, right] : llvm::zip(first.actions, second.actions)) {
    if (left.kind != right.kind || left.phase != right.phase ||
        left.anchor.operation != right.anchor.operation ||
        left.anchor.before != right.anchor.before ||
        left.lane.kind != right.lane.kind ||
        left.lane.index != right.lane.index ||
        left.lane.selector != right.lane.selector ||
        left.guard.kind != right.guard.kind ||
        left.guard.loop != right.guard.loop) {
      return false;
    }
  }
  for (auto [left, right] : llvm::zip(first.completions, second.completions)) {
    if (left.source != right.source || left.target != right.target ||
        left.iterationDistance != right.iterationDistance ||
        left.recurrenceLoop != right.recurrenceLoop ||
        left.setAction != right.setAction ||
        left.waitAction != right.waitAction) {
      return false;
    }
  }
  for (auto [left, right] : llvm::zip(first.traces, second.traces)) {
    if (left.kind != right.kind || left.actions != right.actions ||
        left.controlRegion != right.controlRegion ||
        left.guard.kind != right.guard.kind ||
        left.guard.loop != right.guard.loop) {
      return false;
    }
  }
  return true;
}

std::size_t getIntervalSpan(const CanonicalEvent &event) {
  return event.intervalEnd >= event.intervalBegin
             ? event.intervalEnd - event.intervalBegin
             : event.intervalBegin - event.intervalEnd;
}

bool searchStateLess(const BarrierSearchState &first,
                     const BarrierSearchState &second) {
  return std::tie(first.uncovered, first.intervalSpan, first.additions) <
         std::tie(second.uncovered, second.intervalSpan, second.additions);
}

void appendOwnershipRoundTripEdges(ArrayRef<CanonicalOwnershipCycle> cycles,
                                   ArrayRef<CanonicalEvent> events,
                                   Operation *loop, std::size_t nodeCount,
                                   std::size_t occurrenceCount,
                                   std::vector<SyncGraphEdge> &edges) {
  const OwnershipEventPairs eventPairs = buildOwnershipEventPairs(events);

  for (const CanonicalOwnershipCycle &cycle : cycles) {
    if (cycle.loop != loop ||
        cycle.protocol != CanonicalOwnershipProtocolKind::RoundTrip) {
      continue;
    }
    auto pair = eventPairs.find(cycle.id);
    const bool validPair =
        pair != eventPairs.end() &&
        verifyCanonicalOwnershipEventPair(cycle, pair->second);
    if (!validPair) {
      continue;
    }
    // A complete ready/release pair orders every endpoint in the lane's last
    // use before every endpoint in its first use in the next iteration.
    for (std::size_t occurrence = 0; occurrence + 1 < occurrenceCount;
         ++occurrence) {
      const std::size_t sourceOffset = occurrence * nodeCount;
      const std::size_t targetOffset = sourceOffset + nodeCount;
      for (const CanonicalOwnershipPath &sourcePath : cycle.paths) {
        for (const CanonicalOwnershipPath &targetPath : cycle.paths) {
          for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
            const CanonicalOwnershipUse *sourceUse = nullptr;
            const CanonicalOwnershipUse *targetUse = nullptr;
            for (const CanonicalOwnershipUse &use : sourcePath.uses) {
              if (use.lane == lane) {
                sourceUse = &use;
              }
            }
            for (const CanonicalOwnershipUse &use : targetPath.uses) {
              if (use.lane == lane) {
                targetUse = &use;
                break;
              }
            }
            if (!sourceUse || !targetUse) {
              continue;
            }
            SmallVector<std::size_t, 4> sourceNodes(
                sourceUse->producers.begin(), sourceUse->producers.end());
            sourceNodes.append(sourceUse->consumers.begin(),
                               sourceUse->consumers.end());
            SmallVector<std::size_t, 4> targetNodes(
                targetUse->producers.begin(), targetUse->producers.end());
            targetNodes.append(targetUse->consumers.begin(),
                               targetUse->consumers.end());
            for (std::size_t source : sourceNodes) {
              for (std::size_t target : targetNodes) {
                if (source >= nodeCount || target >= nodeCount) {
                  continue;
                }
                edges.push_back({sourceOffset + source, targetOffset + target,
                                 SyncGraphEdgeKind::HardwareCompletion});
              }
            }
          }
        }
      }
    }
  }
}

} // namespace

bool CanonicalSyncPlanBuilder::isVacuousOwnedAlternatingRecurrence(
    ArrayRef<const CanonicalOwnershipCycle *> cycles,
    const CanonicalDependency &requirement) const {
  if (!requirement.recurrenceLoop || requirement.iterationDistance == 0 ||
      requirement.source >= plan_.nodes_.size() ||
      requirement.target >= plan_.nodes_.size()) {
    return false;
  }
  for (const CanonicalOwnershipCycle *cycle : cycles) {
    if (!cycle || cycle->loop != requirement.recurrenceLoop ||
        cycle->paths.size() != 2) {
      continue;
    }
    std::optional<unsigned> sourcePath;
    std::optional<unsigned> targetPath;
    for (auto [pathIndex, path] : llvm::enumerate(cycle->paths)) {
      if (isInRegion(plan_.nodes_[requirement.source].operation, path.region)) {
        sourcePath = pathIndex;
      }
      if (isInRegion(plan_.nodes_[requirement.target].operation, path.region)) {
        targetPath = pathIndex;
      }
    }
    if (!sourcePath || !targetPath) {
      continue;
    }
    // Path zero is the even arm and path one is the odd arm. Unit-step
    // execution flips the selected arm once per iteration.
    const unsigned expectedTarget =
        (*sourcePath + requirement.iterationDistance) % cycle->paths.size();
    if (*targetPath == expectedTarget ||
        !ownershipCycleContainsNode(*cycle, requirement.source) ||
        !ownershipCycleContainsNode(*cycle, requirement.target)) {
      continue;
    }

    bool foundManagedHazard = false;
    const CanonicalSyncNode &sourceNode = plan_.nodes_[requirement.source];
    const CanonicalSyncNode &targetNode = plan_.nodes_[requirement.target];
    for (const CanonicalMemoryAccess &sourceAccess : sourceNode.accesses) {
      for (const CanonicalMemoryAccess &targetAccess : targetNode.accesses) {
        const bool aliases = memoryAliasesAcrossIterations(
            sourceAccess, targetAccess, requirement.recurrenceLoop,
            requirement.iterationDistance);
        const bool isHazard =
            formsMemoryHazard(requirement.kind, sourceAccess, targetAccess);
        if (!isHazard || !aliases) {
          continue;
        }
        const bool sourceIsOwned =
            ownershipCycleContainsAccess(*cycle, sourceAccess);
        const bool targetIsOwned =
            ownershipCycleContainsAccess(*cycle, targetAccess);
        if (!sourceIsOwned || !targetIsOwned) {
          return false;
        }
        foundManagedHazard = true;
      }
    }
    if (foundManagedHazard) {
      return true;
    }
  }
  return false;
}

std::vector<SyncGraphEdge> CanonicalSyncPlanBuilder::buildEventCompletionEdges(
    ArrayRef<CanonicalEvent> events) const {
  std::vector<SyncGraphEdge> edges;
  std::set<std::pair<std::size_t, std::size_t>> seen;
  for (const CanonicalEvent &event : events) {
    for (const CanonicalEventCompletion &completion : event.completions) {
      if (completion.iterationDistance == 0 &&
          completion.source < plan_.nodes_.size() &&
          completion.target < plan_.nodes_.size() &&
          seen.emplace(completion.source, completion.target).second) {
        edges.push_back({completion.source, completion.target,
                         SyncGraphEdgeKind::HardwareCompletion});
      }
    }
  }
  appendAlternatingOwnershipForwardEdges(plan_.ownershipCycles_, events,
                                         plan_.nodes_.size(), seen, edges);
  return edges;
}

std::size_t CanonicalSyncPlanBuilder::countUncoveredRecurrenceRequirements(
    ArrayRef<CanonicalBarrier> barriers, ArrayRef<CanonicalEvent> events,
    ArrayRef<CanonicalDependency> requirements, bool diagnose,
    SmallVectorImpl<std::size_t> *uncoveredRequirements,
    bool usePositiveTripFacts) const {
  const SmallVector<const CanonicalOwnershipCycle *, 2> alternatingCycles =
      getVerifiedAlternatingCycles(plan_.ownershipCycles_, events);
  std::map<Operation *, std::map<unsigned, SmallVector<std::size_t, 8>>,
           std::less<Operation *>>
      groups;
  for (auto [index, requirement] : llvm::enumerate(requirements)) {
    if (requirement.iterationDistance != 0 && requirement.recurrenceLoop &&
        !hasIntrinsicMmadAccumulatorOrdering(requirement) &&
        !isVacuousOwnedAlternatingRecurrence(alternatingCycles, requirement)) {
      groups[requirement.recurrenceLoop][requirement.iterationDistance]
          .push_back(index);
    }
  }

  std::vector<SyncGraphEdge> forwardEdges = plan_.fixedEdges_;
  std::vector<SyncGraphEdge> barrierEdges =
      buildBarrierCompletionEdges(barriers);
  std::vector<SyncGraphEdge> eventEdges = buildEventCompletionEdges(events);
  forwardEdges.insert(forwardEdges.end(), barrierEdges.begin(),
                      barrierEdges.end());
  forwardEdges.insert(forwardEdges.end(), eventEdges.begin(), eventEdges.end());

  std::size_t uncoveredCount = 0;
  const std::size_t nodeCount = plan_.nodes_.size();
  for (const auto &loopEntry : groups) {
    Operation *loop = loopEntry.first;
    for (const auto &distanceEntry : loopEntry.second) {
      const unsigned distance = distanceEntry.first;
      const std::size_t occurrenceCount =
          static_cast<std::size_t>(distance) + 1;
      if (nodeCount == 0 ||
          occurrenceCount >
              std::numeric_limits<std::size_t>::max() / nodeCount) {
        uncoveredCount += distanceEntry.second.size();
        if (uncoveredRequirements) {
          uncoveredRequirements->append(distanceEntry.second.begin(),
                                        distanceEntry.second.end());
        }
        continue;
      }
      const std::size_t expandedNodeCount = nodeCount * occurrenceCount;
      std::vector<SyncGraphEdge> expandedEdges;
      for (const SyncGraphEdge &edge : forwardEdges) {
        if (edge.source >= nodeCount || edge.target >= nodeCount ||
            !loop->isAncestor(plan_.nodes_[edge.source].operation) ||
            !loop->isAncestor(plan_.nodes_[edge.target].operation)) {
          continue;
        }
        for (std::size_t occurrence = 0; occurrence < occurrenceCount;
             ++occurrence) {
          const std::size_t offset = occurrence * nodeCount;
          expandedEdges.push_back(
              {offset + edge.source, offset + edge.target, edge.kind});
        }
      }

      for (std::size_t occurrence = 0; occurrence + 1 < occurrenceCount;
           ++occurrence) {
        const std::size_t sourceOffset = occurrence * nodeCount;
        const std::size_t targetOffset = sourceOffset + nodeCount;
        for (const CanonicalSyncNode &source : plan_.nodes_) {
          if (!loop->isAncestor(source.operation)) {
            continue;
          }
          for (const CanonicalSyncNode &target : plan_.nodes_) {
            if (source.pipe != target.pipe ||
                !loop->isAncestor(target.operation)) {
              continue;
            }
            expandedEdges.push_back({sourceOffset + source.id,
                                     targetOffset + target.id,
                                     hasHardwareCompletion(source.pipe)
                                         ? SyncGraphEdgeKind::HardwareCompletion
                                         : SyncGraphEdgeKind::IssueOrder});
          }
        }
      }

      for (const CanonicalEvent &event : events) {
        for (const CanonicalEventCompletion &completion : event.completions) {
          if (completion.iterationDistance == 0 ||
              completion.iterationDistance > distance ||
              completion.recurrenceLoop != loop ||
              completion.source >= nodeCount ||
              completion.target >= nodeCount) {
            continue;
          }
          for (std::size_t occurrence = 0;
               occurrence + completion.iterationDistance < occurrenceCount;
               ++occurrence) {
            const std::size_t sourceOffset = occurrence * nodeCount;
            const std::size_t targetOffset =
                (occurrence + completion.iterationDistance) * nodeCount;
            expandedEdges.push_back({sourceOffset + completion.source,
                                     targetOffset + completion.target,
                                     SyncGraphEdgeKind::HardwareCompletion});
          }
        }
      }
      appendOwnershipRoundTripEdges(plan_.ownershipCycles_, events, loop,
                                    nodeCount, occurrenceCount, expandedEdges);

      for (const CanonicalDependency &dependency : plan_.dependencies_) {
        if (!dependency.retained || dependency.iterationDistance == 0 ||
            dependency.recurrenceLoop != loop ||
            dependency.iterationDistance > distance ||
            dependency.source >= nodeCount || dependency.target >= nodeCount) {
          continue;
        }
        const CanonicalSyncNode &source = plan_.nodes_[dependency.source];
        const CanonicalSyncNode &target = plan_.nodes_[dependency.target];
        const bool hasIntrinsicCompletion =
            source.pipe == target.pipe && hasHardwareCompletion(source.pipe);
        const bool hasBarrier =
            source.pipe == target.pipe && !hasHardwareCompletion(source.pipe) &&
            llvm::any_of(barriers, [&](const CanonicalBarrier &barrier) {
              return barrier.pipe == source.pipe &&
                     barrier.recurrenceLoop == loop &&
                     barrier.anchor.operation == target.operation &&
                     barrier.anchor.before;
            });
        if (!hasIntrinsicCompletion && !hasBarrier) {
          continue;
        }
        for (std::size_t occurrence = 0;
             occurrence + dependency.iterationDistance < occurrenceCount;
             ++occurrence) {
          expandedEdges.push_back(
              {occurrence * nodeCount + dependency.source,
               (occurrence + dependency.iterationDistance) * nodeCount +
                   dependency.target,
               SyncGraphEdgeKind::HardwareCompletion});
        }
      }

      std::vector<CompletionRequirement> expandedRequirements;
      for (std::size_t index : distanceEntry.second) {
        const CanonicalDependency &requirement = requirements[index];
        expandedRequirements.push_back(
            {requirement.source, distance * nodeCount + requirement.target});
      }
      const auto isVertexAvailable = [&](std::size_t requirement,
                                         std::size_t vertex) {
        return requirement < distanceEntry.second.size() &&
               isRecurrenceVertexAvailable(
                   requirements[distanceEntry.second[requirement]], vertex,
                   nodeCount, usePositiveTripFacts);
      };
      const std::vector<bool> covered = getCompletionRequirementCoverage(
          expandedNodeCount, expandedEdges, expandedRequirements,
          isVertexAvailable);
      for (std::size_t index = 0; index < covered.size(); ++index) {
        if (covered[index]) {
          continue;
        }
        ++uncoveredCount;
        if (uncoveredRequirements) {
          uncoveredRequirements->push_back(distanceEntry.second[index]);
        }
        if (diagnose) {
          const CanonicalDependency &requirement =
              requirements[distanceEntry.second[index]];
          llvm::errs() << "uncovered canonical recurrence requirement "
                       << requirement.source << " -> " << requirement.target
                       << " distance=" << requirement.iterationDistance << '\n';
        }
      }
    }
  }
  return uncoveredCount;
}

std::vector<CanonicalEvent> CanonicalSyncPlanBuilder::selectRequiredEvents(
    ArrayRef<CanonicalBarrier> barriers,
    ArrayRef<CanonicalEvent> candidates) const {
  std::vector<SyncGraphEdge> fixedEdges = plan_.fixedEdges_;
  std::vector<SyncGraphEdge> barrierEdges =
      buildBarrierCompletionEdges(barriers);
  fixedEdges.insert(fixedEdges.end(), barrierEdges.begin(), barrierEdges.end());
  SmallVector<CanonicalEvent, 4> ownershipEvents;
  llvm::copy_if(
      candidates, std::back_inserter(ownershipEvents),
      [](const CanonicalEvent &event) { return event.ownershipProtocol; });
  std::vector<SyncGraphEdge> ownershipEdges =
      buildEventCompletionEdges(ownershipEvents);
  fixedEdges.insert(fixedEdges.end(), ownershipEdges.begin(),
                    ownershipEdges.end());

  std::vector<CompletionRequirement> requirements;
  SmallVector<std::size_t, 16> forwardCandidates;
  for (auto [index, event] : llvm::enumerate(candidates)) {
    if (!event.recurrenceLoop && !event.ownershipProtocol &&
        event.protocolBundle == 0) {
      requirements.push_back({event.source, event.target});
      forwardCandidates.push_back(index);
    }
  }
  const auto isVertexAvailable = [&](std::size_t requirement,
                                     std::size_t vertex) {
    if (requirement >= forwardCandidates.size()) {
      return false;
    }
    const CanonicalEvent &event = candidates[forwardCandidates[requirement]];
    CanonicalDependency dependency;
    dependency.source = event.source;
    dependency.target = event.target;
    return isForwardVertexAvailable(dependency, vertex);
  };
  const std::vector<bool> keep = reduceCompletionRequirements(
      plan_.nodes_.size(), fixedEdges, requirements, isVertexAvailable);

  std::vector<CanonicalEvent> selected;
  selected.reserve(candidates.size());
  std::size_t forward = 0;
  for (const CanonicalEvent &event : candidates) {
    if (event.recurrenceLoop || event.ownershipProtocol ||
        event.protocolBundle != 0 || (forward < keep.size() && keep[forward])) {
      selected.push_back(event);
    }
    if (!event.recurrenceLoop && !event.ownershipProtocol &&
        event.protocolBundle == 0) {
      ++forward;
    }
  }

  SmallVector<CanonicalDependency, 32> compactRequirements;
  for (const CanonicalDependency &dependency : plan_.dependencies_) {
    if (dependency.retained) {
      compactRequirements.push_back(dependency);
    }
  }

  const auto getOverBudgetDomains = [&]() {
    std::vector<CanonicalEventDomainKey> overBudget;
    const std::optional<CanonicalEventColorPressureMap> pressure =
        evaluateCanonicalEventColorPressure(selected, eventIdMax_,
                                            reservedIds_);
    if (!pressure) {
      for (const CanonicalEvent &event : selected) {
        overBudget.push_back({event.sourcePipe, event.targetPipe});
      }
      llvm::sort(overBudget);
      overBudget.erase(std::unique(overBudget.begin(), overBudget.end()),
                       overBudget.end());
      return overBudget;
    }
    for (const auto &entry : *pressure) {
      if (entry.second.overflow != 0) {
        overBudget.push_back(entry.first);
      }
    }
    return overBudget;
  };

  for (std::size_t index = 0; index < selected.size();) {
    if (!selected[index].recurrenceLoop || selected[index].ownershipProtocol) {
      ++index;
      continue;
    }
    std::vector<CanonicalEvent> reduced = selected;
    reduced.erase(reduced.begin() + index);
    if (planCoversRequirements(barriers, reduced)) {
      selected = std::move(reduced);
      continue;
    }
    ++index;
  }

  while (true) {
    bool removed = false;
    for (const CanonicalEventDomainKey &domain : getOverBudgetDomains()) {
      for (std::size_t index = 0; index < selected.size(); ++index) {
        const CanonicalEvent &event = selected[index];
        if (!event.recurrenceLoop || event.ownershipProtocol ||
            event.sourcePipe != domain.source ||
            event.targetPipe != domain.target) {
          continue;
        }
        std::vector<CanonicalEvent> reduced = selected;
        reduced.erase(reduced.begin() + index);
        if (countUncoveredRequirements(barriers, reduced,
                                       compactRequirements) != 0) {
          continue;
        }
        selected = std::move(reduced);
        removed = true;
        break;
      }
      if (removed) {
        break;
      }
    }
    if (!removed) {
      break;
    }
  }
  if (!planCoversRequirements(barriers, selected)) {
    return std::vector<CanonicalEvent>(candidates.begin(), candidates.end());
  }
  return selected;
}

bool CanonicalSyncPlanBuilder::planCoversRequirements(
    ArrayRef<CanonicalBarrier> barriers, ArrayRef<CanonicalEvent> events,
    bool diagnose, bool usePositiveTripFacts) const {
  return countUncoveredRequirements(
             barriers, events, plan_.completionRequirements_, diagnose,
             nullptr, usePositiveTripFacts) == 0;
}

std::size_t CanonicalSyncPlanBuilder::countUncoveredRequirements(
    ArrayRef<CanonicalBarrier> barriers, ArrayRef<CanonicalEvent> events,
    ArrayRef<CanonicalDependency> requirements, bool diagnose,
    SmallVectorImpl<std::size_t> *uncoveredRequirements,
    bool usePositiveTripFacts) const {
  std::vector<SyncGraphEdge> edges = plan_.fixedEdges_;
  std::vector<SyncGraphEdge> barrierEdges =
      buildBarrierCompletionEdges(barriers);
  std::vector<SyncGraphEdge> eventEdges = buildEventCompletionEdges(events);
  edges.insert(edges.end(), barrierEdges.begin(), barrierEdges.end());
  edges.insert(edges.end(), eventEdges.begin(), eventEdges.end());

  std::vector<CompletionRequirement> forwardRequirements;
  SmallVector<std::size_t, 16> forwardIndices;
  SmallVector<SmallVector<std::size_t, 2>, 16> forwardEquivalentIndices;
  std::size_t uncoveredCount = countUncoveredRecurrenceRequirements(
      barriers, events, requirements, diagnose, uncoveredRequirements,
      usePositiveTripFacts);
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> forwardGroups;
  for (auto [index, requirement] : llvm::enumerate(requirements)) {
    if (requirement.iterationDistance == 0 &&
        !hasIntrinsicMmadAccumulatorOrdering(requirement)) {
      const auto key = std::make_pair(requirement.source, requirement.target);
      auto [group, inserted] =
          forwardGroups.emplace(key, forwardRequirements.size());
      if (inserted) {
        forwardRequirements.push_back({requirement.source, requirement.target});
        forwardIndices.push_back(index);
        forwardEquivalentIndices.emplace_back();
      }
      forwardEquivalentIndices[group->second].push_back(index);
    }
  }
  const auto isVertexAvailable = [&](std::size_t requirement,
                                     std::size_t vertex) {
    return requirement < forwardIndices.size() &&
           isForwardVertexAvailable(requirements[forwardIndices[requirement]],
                                    vertex);
  };
  const std::vector<bool> covered = getCompletionRequirementCoverage(
      plan_.nodes_.size(), edges, forwardRequirements, isVertexAvailable);
  if (diagnose) {
    for (std::size_t index = 0; index < covered.size(); ++index) {
      if (!covered[index]) {
        const CanonicalDependency &requirement =
            requirements[forwardIndices[index]];
        llvm::errs() << "uncovered canonical forward requirement "
                     << requirement.source << " -> " << requirement.target
                     << '\n';
      }
    }
  }
  if (uncoveredRequirements) {
    for (std::size_t index = 0; index < covered.size(); ++index) {
      if (!covered[index]) {
        uncoveredRequirements->append(forwardEquivalentIndices[index].begin(),
                                      forwardEquivalentIndices[index].end());
      }
    }
  }
  uncoveredCount += llvm::count(covered, false);
  return uncoveredCount;
}

bool CanonicalSyncPlanBuilder::eventsFitBudget(
    ArrayRef<CanonicalEvent> events) const {
  for (const CanonicalEvent &event : events) {
    if (event.width == 0 || event.width > kMaxMultiBufferCount) {
      return false;
    }
  }
  const std::optional<CanonicalEventColorPressureMap> pressure =
      evaluateCanonicalEventColorPressure(events, eventIdMax_, reservedIds_);
  return pressure && llvm::all_of(*pressure, [](const auto &entry) {
           return entry.second.overflow == 0;
         });
}

void CanonicalSyncPlanBuilder::optimizeBarriers() {
  std::vector<CanonicalBarrier> barriers = plan_.barriers_;
  std::vector<CanonicalEvent> events =
      selectRequiredEvents(barriers, plan_.events_);

  SmallVector<CanonicalDependency, 32> compactRequirements;
  for (const CanonicalDependency &dependency : plan_.dependencies_) {
    if (dependency.retained) {
      compactRequirements.push_back(dependency);
    }
  }
  std::size_t nextProtocolBundle = 1;

  for (std::size_t index = 0; index < barriers.size();) {
    std::vector<CanonicalBarrier> candidateBarriers = barriers;
    candidateBarriers.erase(candidateBarriers.begin() + index);

    if (eventsFitBudget(events) &&
        countUncoveredRequirements(candidateBarriers, events,
                                   compactRequirements) == 0 &&
        planCoversRequirements(candidateBarriers, events)) {
      barriers = std::move(candidateBarriers);
      continue;
    }
    if (barriers[index].recurrenceLoop) {
      ++index;
      continue;
    }

    const CanonicalBarrier &removedBarrier = barriers[index];
    const std::size_t barrierPosition =
        getAnchorPosition(removedBarrier.anchor);
    SmallVector<BarrierEventBundle, 16> alternatives;
    const auto addAlternative = [&](ArrayRef<CanonicalEvent> candidates) {
      BarrierEventBundle bundle;
      bool needsAddition = false;
      for (const CanonicalEvent &candidate : candidates) {
        if (candidate.recurrenceLoop ||
            (candidate.sourcePipe != removedBarrier.pipe &&
             candidate.targetPipe != removedBarrier.pipe) ||
            getAnchorPosition(candidate.setAnchor) > barrierPosition ||
            getAnchorPosition(candidate.waitAnchor) > barrierPosition) {
          return;
        }
        if (llvm::any_of(bundle.events, [&](const CanonicalEvent &event) {
              return sameEventProtocol(event, candidate);
            })) {
          continue;
        }
        auto selected = llvm::find_if(events, [&](const CanonicalEvent &event) {
          return sameEventProtocol(event, candidate);
        });
        if (candidates.size() > 1 && selected != events.end() &&
            selected->protocolBundle != 0) {
          return;
        }
        needsAddition |= selected == events.end();
        bundle.events.push_back(candidate);
      }
      if (bundle.events.empty() || !needsAddition) {
        return;
      }
      if (bundle.events.size() > 1) {
        bundle.id = nextProtocolBundle++;
      }
      const bool duplicate =
          llvm::any_of(alternatives, [&](const BarrierEventBundle &other) {
            if (other.events.size() != bundle.events.size()) {
              return false;
            }
            for (auto [left, right] : llvm::zip(other.events, bundle.events)) {
              if (!sameEventProtocol(left, right)) {
                return false;
              }
            }
            return true;
          });
      if (!duplicate) {
        alternatives.push_back(std::move(bundle));
      }
    };
    for (const CanonicalEvent &candidate : eventCandidates_) {
      addAlternative(ArrayRef<CanonicalEvent>(&candidate, 1));
    }

    // Complete a same-pipe requirement through an existing incoming ready
    // event. The new reverse event intentionally delays that producer until
    // the earlier same-pipe operation completes.
    for (const CanonicalDependency &requirement : compactRequirements) {
      if (requirement.iterationDistance != 0 ||
          requirement.source >= plan_.nodes_.size() ||
          requirement.target >= plan_.nodes_.size()) {
        continue;
      }
      const CanonicalSyncNode &source = plan_.nodes_[requirement.source];
      const CanonicalSyncNode &target = plan_.nodes_[requirement.target];
      if (source.pipe != removedBarrier.pipe ||
          target.pipe != removedBarrier.pipe) {
        continue;
      }
      SmallVector<CanonicalEvent, 32> incomingCandidates(events.begin(),
                                                         events.end());
      incomingCandidates.append(eventCandidates_.begin(),
                                eventCandidates_.end());
      for (const CanonicalEvent &incoming : incomingCandidates) {
        if (incoming.recurrenceLoop ||
            incoming.targetPipe != removedBarrier.pipe ||
            incoming.target != requirement.target ||
            requirement.source >= incoming.source ||
            incoming.source >= requirement.target) {
          continue;
        }
        CanonicalEvent roundTrip =
            makeForwardEvent(requirement.source, incoming.source);
        if (roundTrip.sourcePipe == roundTrip.targetPipe) {
          continue;
        }
        const CanonicalEvent bundle[] = {roundTrip, incoming};
        addAlternative(bundle);
      }
    }

    const auto buildEvents = [&](const BarrierSearchState &state,
                                 bool tagBundles = false) {
      std::vector<CanonicalEvent> candidateEvents = events;
      for (std::size_t alternative : state.additions) {
        const BarrierEventBundle &bundle = alternatives[alternative];
        for (const CanonicalEvent &candidate : bundle.events) {
          auto existing =
              llvm::find_if(candidateEvents, [&](const CanonicalEvent &event) {
                return sameEventProtocol(event, candidate);
              });
          if (existing == candidateEvents.end()) {
            candidateEvents.push_back(candidate);
            existing = std::prev(candidateEvents.end());
          }
          if (tagBundles && bundle.id != 0) {
            existing->protocolBundle = bundle.id;
          }
        }
      }
      return candidateEvents;
    };

    std::optional<std::vector<CanonicalEvent>> replacement;
    std::vector<BarrierSearchState> beam(1);
    for (std::size_t depth = 0;
         depth < kBarrierMaxBundleSize && !beam.empty() && !replacement;
         ++depth) {
      std::vector<BarrierSearchState> next;
      std::optional<BarrierSearchState> feasible;
      for (const BarrierSearchState &state : beam) {
        const std::size_t begin =
            state.additions.empty() ? 0 : state.additions.back() + 1;
        for (std::size_t alternative = begin; alternative < alternatives.size();
             ++alternative) {
          const bool overlapsAtomicBundle =
              alternatives[alternative].id != 0 &&
              llvm::any_of(state.additions, [&](std::size_t selected) {
                if (alternatives[selected].id == 0) {
                  return false;
                }
                return llvm::any_of(alternatives[alternative].events,
                                    [&](const CanonicalEvent &candidate) {
                                      return llvm::any_of(
                                          alternatives[selected].events,
                                          [&](const CanonicalEvent &existing) {
                                            return sameEventProtocol(candidate,
                                                                     existing);
                                          });
                                    });
              });
          if (overlapsAtomicBundle) {
            continue;
          }
          BarrierSearchState candidateState = state;
          candidateState.additions.push_back(alternative);
          for (const CanonicalEvent &added : alternatives[alternative].events) {
            candidateState.intervalSpan += getIntervalSpan(added);
          }
          std::vector<CanonicalEvent> candidateEvents =
              buildEvents(candidateState);
          if (!eventsFitBudget(candidateEvents)) {
            continue;
          }
          if (failed(verifyEventProtocols(candidateEvents,
                                          /*requireAllocation=*/false,
                                          /*diagnose=*/false))) {
            continue;
          }
          candidateState.uncovered = countUncoveredRequirements(
              candidateBarriers, candidateEvents, compactRequirements);
          if (candidateState.uncovered == 0 &&
              planCoversRequirements(candidateBarriers, candidateEvents)) {
            if (!feasible || searchStateLess(candidateState, *feasible)) {
              feasible = std::move(candidateState);
            }
            continue;
          }
          next.push_back(std::move(candidateState));
        }
      }
      if (feasible) {
        replacement = buildEvents(*feasible, /*tagBundles=*/true);
        break;
      }
      llvm::stable_sort(next, searchStateLess);
      if (next.size() > kBarrierBeamWidth) {
        next.resize(kBarrierBeamWidth);
      }
      beam = std::move(next);
    }

    if (!replacement) {
      ++index;
      continue;
    }
    barriers = std::move(candidateBarriers);
    events = std::move(*replacement);
    index = 0;
  }
  events = selectRequiredEvents(barriers, events);
  plan_.barriers_ = std::move(barriers);
  plan_.events_ = std::move(events);
  removeRedundantMechanisms();
}

LogicalResult CanonicalSyncPlanBuilder::verifyFinalPlan() {
  if (!canonicalEventBundleProjectionMatches(selectedEventBundles_,
                                             plan_.events_)) {
    return func_.emitError(
        "internal error: canonical event bundle projection is stale");
  }
  if (!isCandidatePlanFeasible(plan_.barriers_, selectedEventBundles_,
                               plan_.completionRequirements_,
                               /*diagnose=*/true)) {
    return func_.emitError(
        "internal error: canonical synchronization candidate plan is "
        "infeasible");
  }
  if (failed(verifyEventProtocols(plan_.events_, /*requireAllocation=*/false,
                                  /*diagnose=*/true))) {
    return func_.emitError(
        "internal error: canonical event protocol verification failed");
  }
  if (!planCoversRequirements(plan_.barriers_, plan_.events_,
                              /*diagnose=*/true)) {
    return func_.emitError(
        "internal error: canonical synchronization plan does not cover every "
        "preserved completion requirement");
  }
  if (!eventsFitBudget(plan_.events_)) {
    return func_.emitError(
        "internal error: canonical synchronization plan exceeds the event-id "
        "budget after scarcity repair");
  }
  return success();
}
