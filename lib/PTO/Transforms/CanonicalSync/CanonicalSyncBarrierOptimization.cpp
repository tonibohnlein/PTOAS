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

bool sameEventProtocol(const CanonicalEvent &first,
                       const CanonicalEvent &second) {
  return first.sourcePipe == second.sourcePipe &&
         first.targetPipe == second.targetPipe &&
         first.setAnchor.operation == second.setAnchor.operation &&
         first.setAnchor.before == second.setAnchor.before &&
         first.waitAnchor.operation == second.waitAnchor.operation &&
         first.waitAnchor.before == second.waitAnchor.before &&
         first.recurrenceLoop == second.recurrenceLoop &&
         first.forwardDrainLoop == second.forwardDrainLoop &&
         first.setSlot == second.setSlot && first.waitSlot == second.waitSlot &&
         first.width == second.width &&
         first.iterationDistance == second.iterationDistance;
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

} // namespace

std::vector<SyncGraphEdge> CanonicalSyncPlanBuilder::buildEventCompletionEdges(
    ArrayRef<CanonicalEvent> events) const {
  std::vector<SyncGraphEdge> edges;
  std::set<std::pair<std::size_t, std::size_t>> seen;
  const auto addEdge = [&](std::size_t source, std::size_t target) {
    if (seen.emplace(source, target).second) {
      edges.push_back({source, target, SyncGraphEdgeKind::HardwareCompletion});
    }
  };
  for (const CanonicalEvent &event : events) {
    if (event.recurrenceLoop) {
      continue;
    }
    if (event.source >= plan_.nodes_.size() ||
        event.target >= plan_.nodes_.size()) {
      continue;
    }
    const CanonicalSyncNode &eventSource = plan_.nodes_[event.source];
    const CanonicalSyncNode &eventTarget = plan_.nodes_[event.target];
    if (event.setAnchor.operation == eventSource.operation &&
        !event.setAnchor.before &&
        event.waitAnchor.operation == eventTarget.operation &&
        event.waitAnchor.before) {
      addEdge(event.source, event.target);
      continue;
    }
    const std::size_t setPosition = getAnchorPosition(event.setAnchor);
    const std::size_t waitPosition = getAnchorPosition(event.waitAnchor);
    for (const CanonicalSyncNode &source : plan_.nodes_) {
      if (source.pipe != event.sourcePipe ||
          source.order * 2 + 1 > setPosition) {
        continue;
      }
      for (const CanonicalSyncNode &target : plan_.nodes_) {
        if (target.pipe != event.targetPipe || source.id >= target.id ||
            target.order * 2 < waitPosition ||
            !mayExecuteTogether(source.operation, target.operation) ||
            !isAnchorGuaranteedForRequirement(event.setAnchor, source.id,
                                              target.id) ||
            !isAnchorGuaranteedForRequirement(event.waitAnchor, source.id,
                                              target.id)) {
          continue;
        }
        addEdge(source.id, target.id);
      }
    }
  }
  return edges;
}

std::size_t CanonicalSyncPlanBuilder::countUncoveredRecurrenceRequirements(
    ArrayRef<CanonicalBarrier> barriers, ArrayRef<CanonicalEvent> events,
    ArrayRef<CanonicalDependency> requirements, bool diagnose) const {
  std::map<Operation *, std::map<unsigned, SmallVector<std::size_t, 8>>,
           std::less<Operation *>>
      groups;
  for (auto [index, requirement] : llvm::enumerate(requirements)) {
    if (requirement.iterationDistance != 0 && requirement.recurrenceLoop) {
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

      for (const CanonicalDependency &dependency : plan_.dependencies_) {
        if (!dependency.retained || dependency.iterationDistance == 0 ||
            dependency.recurrenceLoop != loop ||
            dependency.iterationDistance > distance ||
            dependency.source >= nodeCount || dependency.target >= nodeCount) {
          continue;
        }
        const CanonicalSyncNode &source = plan_.nodes_[dependency.source];
        const CanonicalSyncNode &target = plan_.nodes_[dependency.target];
        bool hasProtocol =
            hasHardwareCompletion(source.pipe) && source.pipe == target.pipe;
        if (source.pipe == target.pipe && !hasProtocol) {
          hasProtocol =
              llvm::any_of(barriers, [&](const CanonicalBarrier &barrier) {
                return barrier.pipe == source.pipe &&
                       barrier.recurrenceLoop == loop &&
                       barrier.anchor.operation == target.operation &&
                       barrier.anchor.before;
              });
        } else if (source.pipe != target.pipe) {
          const CanonicalEvent expected = makeRecurrenceEvent(dependency);
          hasProtocol = llvm::any_of(events, [&](const CanonicalEvent &event) {
            return event.source == expected.source &&
                   event.target == expected.target &&
                   sameEventProtocol(event, expected);
          });
        }
        if (!hasProtocol) {
          continue;
        }
        for (std::size_t occurrence = 0;
             occurrence + dependency.iterationDistance < occurrenceCount;
             ++occurrence) {
          const std::size_t sourceOffset = occurrence * nodeCount;
          const std::size_t targetOffset =
              (occurrence + dependency.iterationDistance) * nodeCount;
          expandedEdges.push_back({sourceOffset + dependency.source,
                                   targetOffset + dependency.target,
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
                   nodeCount);
      };
      const std::vector<bool> covered = getCompletionRequirementCoverage(
          expandedNodeCount, expandedEdges, expandedRequirements,
          isVertexAvailable);
      for (std::size_t index = 0; index < covered.size(); ++index) {
        if (covered[index]) {
          continue;
        }
        ++uncoveredCount;
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

  std::vector<CompletionRequirement> requirements;
  SmallVector<std::size_t, 16> forwardCandidates;
  for (auto [index, event] : llvm::enumerate(candidates)) {
    if (!event.recurrenceLoop) {
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
    if (event.recurrenceLoop || (forward < keep.size() && keep[forward])) {
      selected.push_back(event);
    }
    if (!event.recurrenceLoop) {
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
    std::map<CanonicalEventDomainKey, std::vector<SyncInterval>> intervals;
    std::vector<CanonicalEventDomainKey> overBudget;
    for (const CanonicalEvent &event : selected) {
      auto &domain = intervals[{event.sourcePipe, event.targetPipe}];
      for (unsigned lane = 0; lane < event.width; ++lane) {
        domain.push_back({event.intervalBegin, event.intervalEnd});
      }
    }
    for (const auto &entry : intervals) {
      unsigned reserved = 0;
      auto reservedIt = reservedIds_.find(entry.first);
      if (reservedIt != reservedIds_.end()) {
        for (unsigned eventId : reservedIt->second) {
          reserved += eventId < eventIdMax_ ? 1U : 0U;
        }
      }
      if (colorSyncIntervals(entry.second).colorCount >
          eventIdMax_ - reserved) {
        overBudget.push_back(entry.first);
      }
    }
    return overBudget;
  };

  while (true) {
    bool removed = false;
    for (const CanonicalEventDomainKey &domain : getOverBudgetDomains()) {
      for (std::size_t index = 0; index < selected.size(); ++index) {
        const CanonicalEvent &event = selected[index];
        if (!event.recurrenceLoop || event.sourcePipe != domain.source ||
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
    bool diagnose) const {
  return countUncoveredRequirements(
             barriers, events, plan_.completionRequirements_, diagnose) == 0;
}

std::size_t CanonicalSyncPlanBuilder::countUncoveredRequirements(
    ArrayRef<CanonicalBarrier> barriers, ArrayRef<CanonicalEvent> events,
    ArrayRef<CanonicalDependency> requirements, bool diagnose) const {
  std::vector<SyncGraphEdge> edges = plan_.fixedEdges_;
  std::vector<SyncGraphEdge> barrierEdges =
      buildBarrierCompletionEdges(barriers);
  std::vector<SyncGraphEdge> eventEdges = buildEventCompletionEdges(events);
  edges.insert(edges.end(), barrierEdges.begin(), barrierEdges.end());
  edges.insert(edges.end(), eventEdges.begin(), eventEdges.end());

  std::vector<CompletionRequirement> forwardRequirements;
  SmallVector<std::size_t, 16> forwardIndices;
  std::size_t uncoveredCount = countUncoveredRecurrenceRequirements(
      barriers, events, requirements, diagnose);
  std::set<std::pair<std::size_t, std::size_t>> seenForward;
  for (auto [index, requirement] : llvm::enumerate(requirements)) {
    if (requirement.iterationDistance == 0) {
      if (!seenForward.emplace(requirement.source, requirement.target).second) {
        continue;
      }
      forwardRequirements.push_back({requirement.source, requirement.target});
      forwardIndices.push_back(index);
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
  uncoveredCount += llvm::count(covered, false);
  return uncoveredCount;
}

bool CanonicalSyncPlanBuilder::eventsFitBudget(
    ArrayRef<CanonicalEvent> events) const {
  std::map<CanonicalEventDomainKey, std::vector<SyncInterval>> intervals;
  for (const CanonicalEvent &event : events) {
    if (event.width == 0 || event.width > kMaxMultiBufferCount) {
      return false;
    }
    auto &domain = intervals[{event.sourcePipe, event.targetPipe}];
    for (unsigned lane = 0; lane < event.width; ++lane) {
      domain.push_back({event.intervalBegin, event.intervalEnd});
    }
  }
  for (const auto &entry : intervals) {
    unsigned reserved = 0;
    auto reservedIt = reservedIds_.find(entry.first);
    if (reservedIt != reservedIds_.end()) {
      for (unsigned eventId : reservedIt->second) {
        reserved += eventId < eventIdMax_ ? 1U : 0U;
      }
    }
    const unsigned available = eventIdMax_ - reserved;
    if (colorSyncIntervals(entry.second).colorCount > available) {
      return false;
    }
  }
  return true;
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

  for (std::size_t index = 0; index < barriers.size();) {
    if (barriers[index].recurrenceLoop) {
      ++index;
      continue;
    }
    std::vector<CanonicalBarrier> candidateBarriers = barriers;
    candidateBarriers.erase(candidateBarriers.begin() + index);

    if (eventsFitBudget(events) &&
        countUncoveredRequirements(candidateBarriers, events,
                                   compactRequirements) == 0 &&
        planCoversRequirements(candidateBarriers, events)) {
      barriers = std::move(candidateBarriers);
      continue;
    }

    const CanonicalBarrier &removedBarrier = barriers[index];
    const std::size_t barrierPosition =
        getAnchorPosition(removedBarrier.anchor);
    SmallVector<CanonicalEvent, 16> alternatives;
    const auto addAlternative = [&](const CanonicalEvent &candidate) {
      if (candidate.recurrenceLoop ||
          candidate.sourcePipe != removedBarrier.pipe ||
          getAnchorPosition(candidate.setAnchor) > barrierPosition ||
          getAnchorPosition(candidate.waitAnchor) > barrierPosition ||
          llvm::any_of(events,
                       [&](const CanonicalEvent &event) {
                         return sameEventProtocol(event, candidate);
                       }) ||
          llvm::any_of(alternatives, [&](const CanonicalEvent &event) {
            return sameEventProtocol(event, candidate);
          })) {
        return;
      }
      alternatives.push_back(candidate);
    };
    for (const CanonicalEvent &candidate : eventCandidates_) {
      addAlternative(candidate);
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
      for (const CanonicalEvent &incoming : events) {
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
        addAlternative(roundTrip);
      }
    }

    const auto buildEvents = [&](const BarrierSearchState &state) {
      std::vector<CanonicalEvent> candidateEvents = events;
      for (std::size_t alternative : state.additions) {
        const CanonicalEvent &candidate = alternatives[alternative];
        const bool duplicate =
            llvm::any_of(candidateEvents, [&](const CanonicalEvent &event) {
              return sameEventProtocol(event, candidate);
            });
        if (!duplicate) {
          candidateEvents.push_back(candidate);
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
          BarrierSearchState candidateState = state;
          candidateState.additions.push_back(alternative);
          const CanonicalEvent &added = alternatives[alternative];
          candidateState.intervalSpan += getIntervalSpan(added);
          std::vector<CanonicalEvent> candidateEvents =
              buildEvents(candidateState);
          if (!eventsFitBudget(candidateEvents)) {
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
        replacement = buildEvents(*feasible);
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
  plan_.barriers_ = std::move(barriers);
  plan_.events_ = std::move(events);
}

LogicalResult CanonicalSyncPlanBuilder::verifyFinalPlan() {
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
