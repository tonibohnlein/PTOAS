// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverMechanism.h"

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"

#include <algorithm>
#include <optional>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::pto;

namespace {

struct OwnedInterval {
  SyncWeightedInterval interval;
  SyncCoverResourceWitnessUse owner;
};

std::optional<SyncInterval>
getResourceLifetime(const SyncCoverGraph &graph,
                    const SyncCoverMechanism &mechanism,
                    const SyncCoverResourceUse &use) {
  if (use.distance != 0) {
    if (use.scope >= graph.getScopes().size()) {
      return std::nullopt;
    }
    const std::optional<SyncCoverTimelineInterval> &timeline =
        graph.getScopes()[use.scope].timeline;
    if (!timeline) {
      return std::nullopt;
    }
    return SyncInterval{timeline->begin, timeline->end};
  }

  std::optional<SyncCoverTimelinePosition> begin;
  std::optional<SyncCoverTimelinePosition> end;
  for (std::size_t actionIndex : use.actions) {
    if (actionIndex >= mechanism.actions.size()) {
      return std::nullopt;
    }
    const std::optional<SyncCoverTimelinePosition> position =
        resolveSyncCoverAnchor(graph, mechanism.actions[actionIndex].anchor);
    if (!position) {
      return std::nullopt;
    }
    begin = begin ? std::min(*begin, *position) : *position;
    end = end ? std::max(*end, *position) : *position;
  }
  if (!begin || !end) {
    return std::nullopt;
  }
  return SyncInterval{*begin, *end};
}

SyncCoverResourceSelection
makeSelectionError(SyncCoverResourceSelectionError error) {
  SyncCoverResourceSelection result;
  result.error = error;
  return result;
}

std::optional<std::vector<SyncCoverResourceAllocation>>
allocateResourceIds(const std::vector<OwnedInterval> &intervals,
                    const SyncCoverResourceDomain &domain) {
  std::set<unsigned> reusable;
  unsigned nextFresh = 0;
  std::vector<std::size_t> order(intervals.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t first, std::size_t second) {
                     return std::tie(intervals[first].interval.interval.begin,
                                     intervals[first].interval.interval.end,
                                     intervals[first].owner.mechanism,
                                     intervals[first].owner.resourceUse) <
                            std::tie(intervals[second].interval.interval.begin,
                                     intervals[second].interval.interval.end,
                                     intervals[second].owner.mechanism,
                                     intervals[second].owner.resourceUse);
                   });

  using ActiveUse = std::pair<std::size_t, std::size_t>;
  std::priority_queue<ActiveUse, std::vector<ActiveUse>,
                      std::greater<ActiveUse>>
      active;
  std::vector<SyncCoverResourceAllocation> assignments(intervals.size());
  for (std::size_t interval : order) {
    const std::size_t begin = intervals[interval].interval.interval.begin;
    while (true) {
      const bool hasExpired = !active.empty() && active.top().first < begin;
      if (!hasExpired) {
        break;
      }
      const std::size_t expired = active.top().second;
      active.pop();
      reusable.insert(assignments[expired].ids.begin(),
                      assignments[expired].ids.end());
    }
    const std::size_t width = intervals[interval].interval.width;
    assignments[interval].owner = intervals[interval].owner;
    for (std::size_t lane = 0; lane < width; ++lane) {
      if (!reusable.empty()) {
        const auto id = reusable.begin();
        assignments[interval].ids.push_back(*id);
        reusable.erase(id);
        continue;
      }
      while (nextFresh < domain.budget &&
             std::binary_search(domain.reservedIds.begin(),
                                domain.reservedIds.end(), nextFresh)) {
        ++nextFresh;
      }
      if (nextFresh >= domain.budget) {
        return std::nullopt;
      }
      assignments[interval].ids.push_back(nextFresh++);
    }
    active.emplace(intervals[interval].interval.interval.end, interval);
  }
  std::sort(assignments.begin(), assignments.end(),
            [](const auto &first, const auto &second) {
              return std::tie(first.owner.mechanism, first.owner.resourceUse) <
                     std::tie(second.owner.mechanism, second.owner.resourceUse);
            });
  return assignments;
}

} // namespace

bool SyncCoverResourceWitnessUse::operator==(
    const SyncCoverResourceWitnessUse &other) const {
  return mechanism == other.mechanism && resourceUse == other.resourceUse &&
         width == other.width;
}

bool SyncCoverResourceAllocation::operator==(
    const SyncCoverResourceAllocation &other) const {
  return owner == other.owner && ids == other.ids;
}

SyncCoverResourceSelection
SyncCoverMechanismUniverse::evaluateResourceSelection(
    const std::vector<SyncCoverMechanismId> &selected) const {
  if (!validate()) {
    return makeSelectionError(SyncCoverResourceSelectionError::InvalidUniverse);
  }

  std::vector<SyncCoverMechanismId> normalized = selected;
  std::sort(normalized.begin(), normalized.end());
  const bool hasDuplicate =
      std::adjacent_find(normalized.begin(), normalized.end()) !=
      normalized.end();
  if (hasDuplicate) {
    return makeSelectionError(
        SyncCoverResourceSelectionError::InvalidSelection);
  }
  const bool hasUnknown =
      !normalized.empty() && normalized.back() >= mechanisms_.size();
  if (hasUnknown) {
    return makeSelectionError(
        SyncCoverResourceSelectionError::InvalidSelection);
  }

  for (SyncCoverMechanismId mechanism : normalized) {
    for (SyncCoverMechanismId conflict : mechanisms_[mechanism].conflicts) {
      if (mechanism < conflict &&
          std::binary_search(normalized.begin(), normalized.end(), conflict)) {
        SyncCoverResourceSelection result =
            makeSelectionError(SyncCoverResourceSelectionError::Conflict);
        result.firstConflict = mechanism;
        result.secondConflict = conflict;
        return result;
      }
    }
  }

  std::vector<std::vector<OwnedInterval>> intervals(domains_.size());
  for (SyncCoverMechanismId mechanismId : normalized) {
    const SyncCoverMechanism &mechanism = mechanisms_[mechanismId];
    for (std::size_t useId = 0; useId < mechanism.resourceUses.size();
         ++useId) {
      const SyncCoverResourceUse &use = mechanism.resourceUses[useId];
      if (use.domain >= domains_.size()) {
        return makeSelectionError(
            SyncCoverResourceSelectionError::InvalidUniverse);
      }
      const std::optional<SyncInterval> lifetime =
          getResourceLifetime(graph_, mechanism, use);
      if (!lifetime) {
        return makeSelectionError(
            SyncCoverResourceSelectionError::InvalidUniverse);
      }
      intervals[use.domain].push_back(
          {{*lifetime, use.width}, {mechanismId, useId, use.width}});
    }
  }

  SyncCoverResourceSelection result;
  result.resourceFeasible = true;
  for (const SyncCoverResourceDomain &domain : domains_) {
    std::vector<SyncWeightedInterval> weighted;
    weighted.reserve(intervals[domain.id].size());
    for (const OwnedInterval &interval : intervals[domain.id]) {
      weighted.push_back(interval.interval);
    }
    const SyncIntervalPressure pressure = evaluateSyncIntervalPressure(
        weighted, domain.budget, domain.reservedIds);
    if (!pressure) {
      return makeSelectionError(
          pressure.error == SyncIntervalPressureError::ArithmeticOverflow
              ? SyncCoverResourceSelectionError::ArithmeticOverflow
              : SyncCoverResourceSelectionError::InvalidUniverse);
    }

    SyncCoverDomainFeasibility feasibility;
    feasibility.domain = domain.id;
    feasibility.required = pressure.required;
    feasibility.available = pressure.available;
    feasibility.overflow = pressure.overflow;
    feasibility.maximumPoint = pressure.maximumPoint;
    for (std::size_t interval : pressure.maximumClique) {
      feasibility.maximumClique.push_back(intervals[domain.id][interval].owner);
    }
    if (pressure.overflow == 0) {
      const auto allocations =
          allocateResourceIds(intervals[domain.id], domain);
      if (!allocations) {
        return makeSelectionError(
            SyncCoverResourceSelectionError::InvalidUniverse);
      }
      feasibility.allocations = *allocations;
    }
    result.resourceFeasible &= feasibility.overflow == 0;
    result.domains.push_back(std::move(feasibility));
  }
  return result;
}
