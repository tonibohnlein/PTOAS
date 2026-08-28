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
#include <limits>
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

struct PressureEvent {
  SyncCoverTimelinePosition position = 0;
  bool begins = false;
  std::size_t interval = 0;
};

bool consumeWork(SyncCoverCoverageWorkBudget *budget, std::size_t amount = 1) {
  return !budget || budget->consume(amount);
}

template <typename T, typename Compare>
bool meteredStableSort(std::vector<T> &values, Compare compare,
                       SyncCoverCoverageWorkBudget *budget) {
  if (!budget) {
    std::stable_sort(values.begin(), values.end(), compare);
    return true;
  }
  for (std::size_t index = 1; index < values.size(); ++index) {
    T value = std::move(values[index]);
    std::size_t position = index;
    while (position != 0) {
      if (!consumeWork(budget)) {
        return false;
      }
      if (!compare(value, values[position - 1])) {
        break;
      }
      values[position] = std::move(values[position - 1]);
      --position;
    }
    values[position] = std::move(value);
  }
  return true;
}

template <typename T>
bool meteredBinarySearch(const std::vector<T> &values, const T &value,
                         SyncCoverCoverageWorkBudget *budget) {
  std::size_t first = 0;
  std::size_t last = values.size();
  while (first < last) {
    if (!consumeWork(budget)) {
      return false;
    }
    const std::size_t middle = first + (last - first) / 2;
    if (values[middle] < value) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  return first != values.size() && values[first] == value;
}

bool normalizedSelection(const CanonicalSyncPatternProblem &problem,
                         const std::vector<CanonicalSyncMechanismId> &selected,
                         SyncCoverCoverageWorkBudget *budget) {
  for (std::size_t index = 0; index < selected.size(); ++index) {
    if (!consumeWork(budget)) {
      return false;
    }
    const bool invalid = selected[index] >= problem.getMechanisms().size() ||
                         (index != 0 && selected[index - 1] >= selected[index]);
    if (invalid) {
      return false;
    }
  }
  return true;
}

bool conflictFree(const CanonicalSyncPatternProblem &problem,
                  const std::vector<CanonicalSyncMechanismId> &selected,
                  SyncCoverCoverageWorkBudget *budget) {
  for (CanonicalSyncMechanismId mechanism : selected) {
    const auto &conflicts = problem.getMechanisms()[mechanism].conflicts;
    for (CanonicalSyncMechanismId conflict : conflicts) {
      if (!consumeWork(budget)) {
        return false;
      }
      if (meteredBinarySearch(selected, conflict, budget)) {
        return false;
      }
      if (budget && budget->exhausted) {
        return false;
      }
    }
  }
  return true;
}

std::optional<std::size_t> availableIds(const CanonicalSyncEventDomain &domain,
                                        SyncCoverCoverageWorkBudget *budget) {
  std::size_t reserved = 0;
  for (unsigned id : domain.reservedIds) {
    if (!consumeWork(budget)) {
      return std::nullopt;
    }
    reserved += id < domain.budget;
  }
  return domain.budget - reserved;
}

bool measureDomainPressure(
    const std::vector<IntervalUse> &intervals, std::size_t &required,
    std::optional<SyncCoverTimelinePosition> &maximumPoint,
    std::vector<CanonicalSyncMechanismId> &liveMechanisms,
    SyncCoverCoverageWorkBudget *budget) {
  std::vector<PressureEvent> events;
  const bool eventCountOverflows =
      intervals.size() > std::numeric_limits<std::size_t>::max() / 2;
  if (eventCountOverflows) {
    if (budget) {
      budget->exhausted = true;
    }
    return false;
  }
  const std::size_t eventCount = intervals.size() * 2;
  if (!consumeWork(budget, eventCount == 0 ? 1 : eventCount)) {
    return false;
  }
  events.reserve(eventCount);
  for (std::size_t interval = 0; interval < intervals.size(); ++interval) {
    if (!consumeWork(budget, 2)) {
      return false;
    }
    events.push_back({intervals[interval].lifetime.begin, true, interval});
    events.push_back({intervals[interval].lifetime.end, false, interval});
  }
  if (!meteredStableSort(
          events,
          [](const auto &first, const auto &second) {
            return std::tie(first.position, first.begins, first.interval) <
                   std::tie(second.position, second.begins, second.interval);
          },
          budget)) {
    return false;
  }

  std::set<std::size_t> active;
  std::size_t activeWidth = 0;
  for (std::size_t event = 0; event < events.size();) {
    const SyncCoverTimelinePosition position = events[event].position;
    std::size_t next = event;
    while (next < events.size()) {
      if (!consumeWork(budget)) {
        return false;
      }
      if (events[next].position != position) {
        break;
      }
      if (events[next].begins) {
        const std::size_t width = intervals[events[next].interval].width;
        const bool pressureOverflows =
            width > std::numeric_limits<std::size_t>::max() - activeWidth;
        if (pressureOverflows) {
          return false;
        }
        activeWidth += width;
        const bool activeInsertWorkUnavailable =
            !consumeWork(budget, active.size() + 1);
        if (activeInsertWorkUnavailable) {
          return false;
        }
        active.insert(events[next].interval);
      }
      ++next;
    }
    if (activeWidth > required) {
      required = activeWidth;
      maximumPoint = position;
      liveMechanisms.clear();
      for (std::size_t interval : active) {
        if (!consumeWork(budget)) {
          return false;
        }
        liveMechanisms.push_back(intervals[interval].mechanism);
      }
      if (!meteredStableSort(liveMechanisms, std::less<>(), budget)) {
        return false;
      }
      if (!consumeWork(budget,
                       liveMechanisms.empty() ? 1 : liveMechanisms.size())) {
        return false;
      }
      liveMechanisms.erase(
          std::unique(liveMechanisms.begin(), liveMechanisms.end()),
          liveMechanisms.end());
    }
    for (std::size_t current = event; current < next; ++current) {
      if (!events[current].begins) {
        const bool activeEraseWorkUnavailable =
            !consumeWork(budget, active.size() + 1);
        if (activeEraseWorkUnavailable) {
          return false;
        }
        const std::size_t interval = events[current].interval;
        activeWidth -= intervals[interval].width;
        active.erase(interval);
      }
    }
    event = next;
  }
  return true;
}

std::optional<std::vector<CanonicalSyncEventAllocation>>
allocateDomain(const CanonicalSyncEventDomain &domain,
               const std::vector<IntervalUse> &inputIntervals,
               SyncCoverCoverageWorkBudget *budget) {
  if (!consumeWork(budget,
                   inputIntervals.empty() ? 1 : inputIntervals.size())) {
    return std::nullopt;
  }
  std::vector<IntervalUse> intervals = inputIntervals;
  if (!meteredStableSort(
          intervals,
          [](const IntervalUse &first, const IntervalUse &second) {
            return std::tie(first.lifetime.begin, first.lifetime.end,
                            first.mechanism, first.eventUse) <
                   std::tie(second.lifetime.begin, second.lifetime.end,
                            second.mechanism, second.eventUse);
          },
          budget)) {
    return std::nullopt;
  }
  std::priority_queue<ActiveUse, std::vector<ActiveUse>,
                      std::greater<ActiveUse>>
      active;
  std::set<unsigned> reusable;
  unsigned nextFresh = 0;
  const bool allocationWorkspaceUnavailable =
      !consumeWork(budget, intervals.empty() ? 1 : intervals.size());
  if (allocationWorkspaceUnavailable) {
    return std::nullopt;
  }
  std::vector<CanonicalSyncEventAllocation> allocations(intervals.size());

  for (std::size_t interval = 0; interval < intervals.size(); ++interval) {
    while (!active.empty()) {
      if (!consumeWork(budget, active.size())) {
        return std::nullopt;
      }
      const bool hasExpired =
          active.top().end < intervals[interval].lifetime.begin;
      if (!hasExpired) {
        break;
      }
      const std::size_t expiredInterval = active.top().interval;
      active.pop();
      for (unsigned id : allocations[expiredInterval].ids) {
        const bool reusableInsertWorkUnavailable =
            !consumeWork(budget, reusable.size() + 1);
        if (reusableInsertWorkUnavailable) {
          return std::nullopt;
        }
        reusable.insert(id);
      }
    }
    allocations[interval].mechanism = intervals[interval].mechanism;
    allocations[interval].eventUse = intervals[interval].eventUse;
    for (std::size_t lane = 0; lane < intervals[interval].width; ++lane) {
      const bool laneWorkUnavailable =
          !consumeWork(budget, reusable.size() + 1);
      if (laneWorkUnavailable) {
        return std::nullopt;
      }
      if (!reusable.empty()) {
        const auto id = reusable.begin();
        allocations[interval].ids.push_back(*id);
        reusable.erase(id);
        continue;
      }
      while (nextFresh < domain.budget &&
             meteredBinarySearch(domain.reservedIds, nextFresh, budget)) {
        ++nextFresh;
      }
      if (budget && budget->exhausted) {
        return std::nullopt;
      }
      if (nextFresh >= domain.budget) {
        return std::nullopt;
      }
      allocations[interval].ids.push_back(nextFresh++);
    }
    const bool activePushWorkUnavailable =
        !consumeWork(budget, active.size() + 1);
    if (activePushWorkUnavailable) {
      return std::nullopt;
    }
    active.push({intervals[interval].lifetime.end, interval});
  }
  if (!meteredStableSort(
          allocations,
          [](const auto &first, const auto &second) {
            return std::tie(first.mechanism, first.eventUse) <
                   std::tie(second.mechanism, second.eventUse);
          },
          budget)) {
    return std::nullopt;
  }
  return allocations;
}

} // namespace

CanonicalSyncResourceAllocation mlir::pto::allocateCanonicalSyncEvents(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected,
    SyncCoverCoverageWorkBudget *workBudget) {
  CanonicalSyncResourceAllocation result;
  const bool invalidSelection =
      !normalizedSelection(problem, selected, workBudget) ||
      !conflictFree(problem, selected, workBudget);
  if (invalidSelection) {
    return result;
  }
  result.valid = true;
  result.feasible = true;
  const bool domainWorkspaceUnavailable = !consumeWork(
      workBudget,
      problem.getDomains().empty() ? 1 : problem.getDomains().size());
  if (domainWorkspaceUnavailable) {
    return result;
  }
  std::vector<std::vector<IntervalUse>> intervals(problem.getDomains().size());
  for (CanonicalSyncMechanismId mechanismId : selected) {
    const CanonicalSyncMechanism &mechanism =
        problem.getMechanisms()[mechanismId];
    for (std::size_t use = 0; use < mechanism.descriptor.eventUses.size();
         ++use) {
      const CanonicalSyncEventUse &eventUse =
          mechanism.descriptor.eventUses[use];
      if (!consumeWork(workBudget)) {
        return result;
      }
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
    const std::optional<std::size_t> available =
        availableIds(domain, workBudget);
    if (!available) {
      return result;
    }
    allocation.available = *available;
    const bool pressureValid = measureDomainPressure(
        intervals[domain.id], allocation.required,
        allocation.maximumPressurePoint, allocation.liveMechanisms, workBudget);
    if (!pressureValid) {
      result.valid = false;
      result.feasible = false;
      return result;
    }
    const bool feasible = allocation.required <= allocation.available;
    auto assigned =
        feasible ? allocateDomain(domain, intervals[domain.id], workBudget)
                 : std::nullopt;
    if (!assigned) {
      result.feasible = false;
    } else {
      allocation.uses = std::move(*assigned);
    }
    if (!consumeWork(workBudget)) {
      return result;
    }
    result.domains.push_back(std::move(allocation));
  }
  return result;
}
