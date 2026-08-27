// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include <algorithm>
#include <queue>
#include <set>
#include <tuple>

using namespace mlir::pto;

namespace {

struct IntervalUse {
  CanonicalSyncMechanismId mechanism = 0;
  std::size_t eventUse = 0;
  CanonicalSyncEventLifetime lifetime;
  std::size_t width = 1;
};

struct ActiveUse {
  SyncCoverTimelinePosition end = 0;
  std::size_t interval = 0;

  bool operator>(const ActiveUse &other) const {
    return std::tie(end, interval) > std::tie(other.end, other.interval);
  }
};

bool normalizedSelection(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected) {
  return std::is_sorted(selected.begin(), selected.end()) &&
         std::adjacent_find(selected.begin(), selected.end()) ==
             selected.end() &&
         std::all_of(selected.begin(), selected.end(), [&](auto mechanism) {
           return mechanism < problem.getMechanisms().size();
         });
}

bool conflictFree(const CanonicalSyncPatternProblem &problem,
                  const std::vector<CanonicalSyncMechanismId> &selected) {
  for (CanonicalSyncMechanismId mechanism : selected) {
    const auto &conflicts = problem.getMechanisms()[mechanism].conflicts;
    const bool hasConflict =
        std::any_of(conflicts.begin(), conflicts.end(), [&](auto conflict) {
          return std::binary_search(selected.begin(), selected.end(), conflict);
        });
    if (hasConflict) {
      return false;
    }
  }
  return true;
}

std::size_t availableIds(const CanonicalSyncEventDomain &domain) {
  return domain.budget -
         static_cast<std::size_t>(
             std::count_if(domain.reservedIds.begin(), domain.reservedIds.end(),
                           [&](unsigned id) { return id < domain.budget; }));
}

std::optional<std::vector<CanonicalSyncEventAllocation>>
allocateDomain(const CanonicalSyncEventDomain &domain,
               std::vector<IntervalUse> intervals, std::size_t &required) {
  std::stable_sort(intervals.begin(), intervals.end(),
                   [](const IntervalUse &first, const IntervalUse &second) {
                     return std::tie(first.lifetime.begin, first.lifetime.end,
                                     first.mechanism, first.eventUse) <
                            std::tie(second.lifetime.begin, second.lifetime.end,
                                     second.mechanism, second.eventUse);
                   });
  std::priority_queue<ActiveUse, std::vector<ActiveUse>,
                      std::greater<ActiveUse>>
      active;
  std::set<unsigned> reusable;
  unsigned nextFresh = 0;
  std::size_t activeWidth = 0;
  required = 0;
  std::vector<CanonicalSyncEventAllocation> allocations(intervals.size());

  for (std::size_t interval = 0; interval < intervals.size(); ++interval) {
    while (!active.empty()) {
      const bool hasExpired =
          active.top().end < intervals[interval].lifetime.begin;
      if (!hasExpired) {
        break;
      }
      const std::size_t expiredInterval = active.top().interval;
      active.pop();
      activeWidth -= allocations[expiredInterval].ids.size();
      reusable.insert(allocations[expiredInterval].ids.begin(),
                      allocations[expiredInterval].ids.end());
    }
    allocations[interval].mechanism = intervals[interval].mechanism;
    allocations[interval].eventUse = intervals[interval].eventUse;
    for (std::size_t lane = 0; lane < intervals[interval].width; ++lane) {
      if (!reusable.empty()) {
        const auto id = reusable.begin();
        allocations[interval].ids.push_back(*id);
        reusable.erase(id);
        continue;
      }
      while (nextFresh < domain.budget &&
             std::binary_search(domain.reservedIds.begin(),
                                domain.reservedIds.end(), nextFresh)) {
        ++nextFresh;
      }
      if (nextFresh >= domain.budget) {
        required = std::max(required, activeWidth + intervals[interval].width);
        return std::nullopt;
      }
      allocations[interval].ids.push_back(nextFresh++);
    }
    activeWidth += intervals[interval].width;
    required = std::max(required, activeWidth);
    active.push({intervals[interval].lifetime.end, interval});
  }
  std::sort(allocations.begin(), allocations.end(),
            [](const auto &first, const auto &second) {
              return std::tie(first.mechanism, first.eventUse) <
                     std::tie(second.mechanism, second.eventUse);
            });
  return allocations;
}

} // namespace

CanonicalSyncResourceAllocation mlir::pto::allocateCanonicalSyncEvents(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected) {
  CanonicalSyncResourceAllocation result;
  const bool invalidSelection = !normalizedSelection(problem, selected) ||
                                !conflictFree(problem, selected);
  if (invalidSelection) {
    return result;
  }
  result.valid = true;
  result.feasible = true;
  std::vector<std::vector<IntervalUse>> intervals(problem.getDomains().size());
  for (CanonicalSyncMechanismId mechanismId : selected) {
    const CanonicalSyncMechanism &mechanism =
        problem.getMechanisms()[mechanismId];
    for (std::size_t use = 0; use < mechanism.descriptor.eventUses.size();
         ++use) {
      const CanonicalSyncEventUse &eventUse =
          mechanism.descriptor.eventUses[use];
      const bool invalidUse = eventUse.domain >= intervals.size() ||
                              use >= mechanism.eventLifetimes.size();
      if (invalidUse) {
        result.valid = false;
        result.feasible = false;
        return result;
      }
      const CanonicalSyncEventLifetime &lifetime =
          mechanism.eventLifetimes[use];
      if (lifetime.begin > lifetime.end) {
        result.valid = false;
        result.feasible = false;
        return result;
      }
      intervals[eventUse.domain].push_back(
          {mechanismId, use, lifetime, eventUse.width});
    }
  }

  for (const CanonicalSyncEventDomain &domain : problem.getDomains()) {
    CanonicalSyncDomainAllocation allocation;
    allocation.domain = domain.id;
    allocation.available = availableIds(domain);
    const auto assigned = allocateDomain(
        domain, std::move(intervals[domain.id]), allocation.required);
    if (!assigned) {
      result.feasible = false;
    } else {
      allocation.uses = *assigned;
    }
    result.domains.push_back(std::move(allocation));
  }
  return result;
}
