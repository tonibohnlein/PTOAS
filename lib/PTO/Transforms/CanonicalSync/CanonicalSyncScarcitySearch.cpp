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
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr std::size_t kScarcityBeamWidth = 16;

struct EventGroup {
  SmallVector<std::size_t, 4> members;
  CanonicalEvent event;
};

struct ScarcityState {
  std::vector<EventGroup> groups;
  SmallVector<std::size_t, 16> hotspotGroups;
  std::size_t serializationCost = 0;
  std::uint64_t criticalPathWeight = 0;
  unsigned colorCount = 0;
};

using StateSignature = std::vector<std::vector<std::size_t>>;

std::vector<std::size_t> getBlockPipeRanks(ArrayRef<CanonicalSyncNode> nodes) {
  std::map<Block *, std::map<PipelineType, std::vector<std::size_t>>> groups;
  for (auto [index, node] : llvm::enumerate(nodes)) {
    if (node.operation) {
      groups[node.operation->getBlock()][node.pipe].push_back(index);
    }
  }

  std::vector<std::size_t> ranks(nodes.size(), 0);
  for (auto &blockEntry : groups) {
    for (auto &pipeEntry : blockEntry.second) {
      std::vector<std::size_t> &members = pipeEntry.second;
      llvm::stable_sort(members, [&](std::size_t first, std::size_t second) {
        return std::tie(nodes[first].order, first) <
               std::tie(nodes[second].order, second);
      });
      for (auto [rank, member] : llvm::enumerate(members)) {
        ranks[member] = rank;
      }
    }
  }
  return ranks;
}

bool isInDomain(const CanonicalEvent &event,
                const CanonicalEventDomainKey &key) {
  return event.sourcePipe == key.source && event.targetPipe == key.target;
}

StateSignature getSignature(const ScarcityState &state) {
  StateSignature signature;
  for (const EventGroup &group : state.groups) {
    signature.emplace_back(group.members.begin(), group.members.end());
  }
  return signature;
}

void sortGroups(ScarcityState &state) {
  llvm::stable_sort(state.groups,
                    [](const EventGroup &first, const EventGroup &second) {
                      return std::lexicographical_compare(
                          first.members.begin(), first.members.end(),
                          second.members.begin(), second.members.end());
                    });
}

std::size_t saturatingAddDistance(std::size_t value, std::size_t increment) {
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  return increment > maximum - value ? maximum : value + increment;
}

std::size_t calculateCost(const ScarcityState &state,
                          ArrayRef<CanonicalEvent> originals,
                          ArrayRef<std::size_t> blockPipeRanks) {
  std::size_t cost = 0;
  for (const EventGroup &group : state.groups) {
    const std::size_t mergedSourceRank = blockPipeRanks[group.event.source];
    const std::size_t mergedTargetRank = blockPipeRanks[group.event.target];
    for (std::size_t member : group.members) {
      const CanonicalEvent &original = originals[member];
      cost = saturatingAddDistance(cost, mergedSourceRank -
                                             blockPipeRanks[original.source]);
      cost = saturatingAddDistance(cost, blockPipeRanks[original.target] -
                                             mergedTargetRank);
    }
  }
  return cost;
}

void evaluateState(ScarcityState &state, ArrayRef<CanonicalEvent> originals,
                   ArrayRef<std::size_t> blockPipeRanks,
                   const CanonicalSyncLatencyContext &latencyContext) {
  std::vector<SyncInterval> intervals;
  SmallVector<std::size_t, 16> laneOwners;
  for (auto [groupIndex, group] : llvm::enumerate(state.groups)) {
    for (unsigned lane = 0; lane < group.event.width; ++lane) {
      intervals.push_back({group.event.intervalBegin, group.event.intervalEnd});
      laneOwners.push_back(groupIndex);
    }
  }
  state.colorCount = colorSyncIntervals(intervals).colorCount;
  std::set<std::size_t> hotspot;
  for (std::size_t lane : findMaximumIntervalClique(intervals)) {
    hotspot.insert(laneOwners[lane]);
  }
  state.hotspotGroups.assign(hotspot.begin(), hotspot.end());
  state.serializationCost = calculateCost(state, originals, blockPipeRanks);
  SmallVector<CanonicalEvent, 16> events;
  for (const EventGroup &group : state.groups) {
    events.push_back(group.event);
  }
  state.criticalPathWeight = latencyContext.calculateCriticalPathWeight(events);
}

bool stateLess(const ScarcityState &first, const ScarcityState &second,
               unsigned availableIds) {
  const unsigned firstOverflow =
      first.colorCount > availableIds ? first.colorCount - availableIds : 0;
  const unsigned secondOverflow =
      second.colorCount > availableIds ? second.colorCount - availableIds : 0;
  return std::make_tuple(firstOverflow, first.criticalPathWeight,
                         first.serializationCost, first.groups.size(),
                         getSignature(first)) <
         std::make_tuple(secondOverflow, second.criticalPathWeight,
                         second.serializationCost, second.groups.size(),
                         getSignature(second));
}

std::vector<CanonicalEvent> collectEvents(const ScarcityState &state) {
  std::vector<CanonicalEvent> events;
  for (const EventGroup &group : state.groups) {
    events.push_back(group.event);
  }
  return events;
}

} // namespace

LogicalResult
CanonicalSyncPlanBuilder::repairEventDomain(const CanonicalEventDomainKey &key,
                                            unsigned availableIds) {
  std::vector<std::size_t> domainIndices;
  std::vector<CanonicalEvent> originals;
  for (auto [index, event] : llvm::enumerate(plan_.events_)) {
    if (isInDomain(event, key)) {
      domainIndices.push_back(index);
      originals.push_back(event);
    }
  }

  ScarcityState initial;
  for (auto [index, event] : llvm::enumerate(originals)) {
    initial.groups.push_back({{index}, event});
  }
  const std::vector<std::size_t> blockPipeRanks =
      getBlockPipeRanks(plan_.nodes_);
  const CanonicalSyncLatencyContext latencyContext(plan_, originals, key);
  evaluateState(initial, originals, blockPipeRanks, latencyContext);
  CanonicalScarcityStats &stats = scarcityStats_[key];
  stats.originalEventCount = originals.size();
  stats.originalColorCount = initial.colorCount;
  stats.originalCriticalPathWeight = initial.criticalPathWeight;
  stats.criticalPathWeight = initial.criticalPathWeight;
  if (initial.colorCount <= availableIds) {
    return success();
  }

  std::vector<ScarcityState> frontier{initial};
  std::set<StateSignature> seen{getSignature(initial)};
  for (std::size_t depth = 0; depth + 1 < originals.size(); ++depth) {
    std::vector<ScarcityState> candidates;
    for (const ScarcityState &state : frontier) {
      for (std::size_t hotspot : state.hotspotGroups) {
        for (std::size_t other = 0; other < state.groups.size(); ++other) {
          if (hotspot == other) {
            continue;
          }
          const std::size_t left = std::min(hotspot, other);
          const std::size_t right = std::max(hotspot, other);
          ScarcityState candidate = state;
          EventGroup merged = candidate.groups[left];
          merged.members.append(candidate.groups[right].members.begin(),
                                candidate.groups[right].members.end());
          llvm::sort(merged.members);
          SmallVector<CanonicalEvent, 4> mergedOriginals;
          for (std::size_t member : merged.members) {
            mergedOriginals.push_back(originals[member]);
          }
          std::optional<CanonicalEvent> event =
              coalesceForwardEvents(mergedOriginals);
          if (!event || !coversCoalescedEvents(*event, mergedOriginals)) {
            continue;
          }
          merged.event = *event;
          candidate.groups[left] = std::move(merged);
          candidate.groups.erase(candidate.groups.begin() + right);
          sortGroups(candidate);
          if (!seen.insert(getSignature(candidate)).second) {
            continue;
          }
          evaluateState(candidate, originals, blockPipeRanks, latencyContext);
          candidates.push_back(std::move(candidate));
        }
      }
    }
    llvm::stable_sort(candidates, [&](const ScarcityState &first,
                                      const ScarcityState &second) {
      return stateLess(first, second, availableIds);
    });
    const bool hasCandidate = !candidates.empty();
    if (hasCandidate && candidates.front().colorCount <= availableIds) {
      const ScarcityState &solution = candidates.front();
      stats.serializationCost = solution.serializationCost;
      stats.criticalPathWeight = solution.criticalPathWeight;
      const std::vector<CanonicalEvent> repaired = collectEvents(solution);
      std::vector<CanonicalEvent> updated;
      bool inserted = false;
      for (auto [index, event] : llvm::enumerate(plan_.events_)) {
        if (llvm::is_contained(domainIndices, index)) {
          if (!inserted) {
            updated.insert(updated.end(), repaired.begin(), repaired.end());
            inserted = true;
          }
        } else {
          updated.push_back(event);
        }
      }
      plan_.events_ = std::move(updated);
      return success();
    }
    const std::size_t candidateCount = candidates.size();
    if (candidateCount > kScarcityBeamWidth) {
      candidates.resize(kScarcityBeamWidth);
    }
    frontier = std::move(candidates);
    if (frontier.empty()) {
      break;
    }
  }

  return func_.emitError()
         << "PTOCanonicalSync cannot repair event-id scarcity in domain "
         << stringifyPIPE(static_cast<PIPE>(key.source)) << " -> "
         << stringifyPIPE(static_cast<PIPE>(key.target)) << ": "
         << initial.colorCount << " colors requested but only " << availableIds
         << " event IDs are available; bounded structural coalescing found no "
            "completion-covered plan";
}
