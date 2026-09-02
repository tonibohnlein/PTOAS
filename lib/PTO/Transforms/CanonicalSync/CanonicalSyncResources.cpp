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
#include <map>
#include <set>
#include <tuple>

using namespace mlir::pto;

namespace {

struct IntervalUse {
  CanonicalSyncMechanismId mechanism = 0;
  std::size_t eventUse = 0;
  CanonicalSyncEventLifetime lifetime;
  std::size_t width = 1;
  bool quiescentAtLifetimeBoundaries = false;
};

struct SweepEvent {
  SyncCoverTimelinePosition position = 0;
  bool start = false;
  CanonicalSyncMechanismId mechanism = 0;
  std::size_t width = 1;
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
  if (values.size() < 2) {
    return consumeWork(budget);
  }
  if (!consumeWork(budget, values.size())) {
    return false;
  }
  std::vector<T> source = std::move(values);
  std::vector<T> target;
  target.reserve(source.size());
  for (std::size_t width = 1; width < source.size();) {
    target.clear();
    for (std::size_t begin = 0; begin < source.size();) {
      const std::size_t middle = std::min(source.size(), begin + width);
      const std::size_t end = std::min(source.size(), middle + width);
      std::size_t left = begin;
      std::size_t right = middle;
      while (left < middle && right < end) {
        if (!consumeWork(budget, 2)) {
          return false;
        }
        if (compare(source[right], source[left])) {
          target.push_back(std::move(source[right++]));
        } else {
          target.push_back(std::move(source[left++]));
        }
      }
      while (left < middle) {
        if (!consumeWork(budget)) {
          return false;
        }
        target.push_back(std::move(source[left++]));
      }
      while (right < end) {
        if (!consumeWork(budget)) {
          return false;
        }
        target.push_back(std::move(source[right++]));
      }
      begin = end;
    }
    source.swap(target);
    if (width > source.size() / 2) {
      break;
    }
    width *= 2;
  }
  values = std::move(source);
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
  // An ordinary event use owns its hardware channel for the whole plan: a
  // lexically later source Set can overtake an earlier target Wait. A deep
  // lifecycle certificate is different. Its two-way prime/body/drain
  // automaton proves that every token is consumed at the lifetime exit, so
  // two such uses may share only when their inclusive lifetimes do not
  // overlap.
  required = 0;
  liveMechanisms.clear();
  std::map<CanonicalSyncMechanismId, std::size_t> persistentLive;
  std::vector<SweepEvent> events;
  for (const IntervalUse &interval : intervals) {
    if (!consumeWork(budget, 2)) {
      return false;
    }
    if (interval.quiescentAtLifetimeBoundaries) {
      events.push_back(
          {interval.lifetime.begin, true, interval.mechanism, interval.width});
      events.push_back(
          {interval.lifetime.end, false, interval.mechanism, interval.width});
      continue;
    }
    if (interval.width > std::numeric_limits<std::size_t>::max() - required ||
        interval.width > std::numeric_limits<std::size_t>::max() -
                             persistentLive[interval.mechanism]) {
      return false;
    }
    required += interval.width;
    persistentLive[interval.mechanism] += interval.width;
    if (!maximumPoint || interval.lifetime.begin < *maximumPoint) {
      maximumPoint = interval.lifetime.begin;
    }
  }
  std::map<CanonicalSyncMechanismId, std::size_t> live = persistentLive;
  const auto retainLive = [&]() {
    liveMechanisms.clear();
    for (const auto &[mechanism, count] : live) {
      if (count != 0) {
        liveMechanisms.push_back(mechanism);
      }
    }
  };
  retainLive();
  if (!meteredStableSort(
          events,
          [](const SweepEvent &left, const SweepEvent &right) {
            // Inclusive lifetimes overlap at a shared endpoint, so starts are
            // applied before pressure is measured and ends are removed after.
            return std::tie(left.position, left.start, left.mechanism) <
                   std::tie(right.position, right.start, right.mechanism);
          },
          budget)) {
    return false;
  }
  std::size_t current = required;
  for (std::size_t begin = 0; begin < events.size();) {
    std::size_t end = begin;
    while (end < events.size() &&
           events[end].position == events[begin].position) {
      ++end;
    }
    for (std::size_t index = begin; index < end; ++index) {
      const SweepEvent &event = events[index];
      if (!event.start) {
        continue;
      }
      if (!consumeWork(budget) ||
          event.width > std::numeric_limits<std::size_t>::max() - current ||
          event.width >
              std::numeric_limits<std::size_t>::max() - live[event.mechanism]) {
        return false;
      }
      current += event.width;
      live[event.mechanism] += event.width;
    }
    if (current > required) {
      required = current;
      maximumPoint = events[begin].position;
      retainLive();
    }
    for (std::size_t index = begin; index < end; ++index) {
      const SweepEvent &event = events[index];
      if (event.start) {
        continue;
      }
      auto found = live.find(event.mechanism);
      if (!consumeWork(budget) || event.width > current ||
          found == live.end() || event.width > found->second) {
        return false;
      }
      current -= event.width;
      found->second -= event.width;
    }
    begin = end;
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
  unsigned nextFresh = 0;
  const bool allocationWorkspaceUnavailable =
      !consumeWork(budget, intervals.empty() ? 1 : intervals.size());
  if (allocationWorkspaceUnavailable) {
    return std::nullopt;
  }
  std::vector<CanonicalSyncEventAllocation> allocations;
  allocations.reserve(intervals.size());
  const auto takeFresh = [&]() -> std::optional<unsigned> {
    while (nextFresh < domain.budget &&
           meteredBinarySearch(domain.reservedIds, nextFresh, budget)) {
      ++nextFresh;
    }
    if ((budget && budget->exhausted) || nextFresh >= domain.budget) {
      return std::nullopt;
    }
    return nextFresh++;
  };
  struct ActiveId {
    SyncCoverTimelinePosition end = 0;
    unsigned id = 0;
  };
  std::vector<ActiveId> active;
  std::set<unsigned> reusable;

  // Persistent uses can never share, even with certified lifetimes.
  for (const IntervalUse &interval : intervals) {
    if (interval.quiescentAtLifetimeBoundaries) {
      continue;
    }
    CanonicalSyncEventAllocation allocation;
    allocation.mechanism = interval.mechanism;
    allocation.eventUse = interval.eventUse;
    for (std::size_t lane = 0; lane < interval.width; ++lane) {
      if (!consumeWork(budget)) {
        return std::nullopt;
      }
      const std::optional<unsigned> id = takeFresh();
      if (!id) {
        return std::nullopt;
      }
      allocation.ids.push_back(*id);
    }
    allocations.push_back(std::move(allocation));
  }

  for (const IntervalUse &interval : intervals) {
    if (!interval.quiescentAtLifetimeBoundaries) {
      continue;
    }
    std::vector<ActiveId> retained;
    retained.reserve(active.size());
    for (const ActiveId &entry : active) {
      if (!consumeWork(budget)) {
        return std::nullopt;
      }
      if (entry.end < interval.lifetime.begin) {
        reusable.insert(entry.id);
      } else {
        retained.push_back(entry);
      }
    }
    active.swap(retained);
    CanonicalSyncEventAllocation allocation;
    allocation.mechanism = interval.mechanism;
    allocation.eventUse = interval.eventUse;
    for (std::size_t lane = 0; lane < interval.width; ++lane) {
      if (!consumeWork(budget)) {
        return std::nullopt;
      }
      unsigned id = 0;
      if (!reusable.empty()) {
        const auto first = reusable.begin();
        id = *first;
        reusable.erase(first);
      } else {
        const std::optional<unsigned> fresh = takeFresh();
        if (!fresh) {
          return std::nullopt;
        }
        id = *fresh;
      }
      allocation.ids.push_back(id);
      active.push_back({interval.lifetime.end, id});
    }
    allocations.push_back(std::move(allocation));
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
          {mechanismId, use, lifetime, eventUse.width,
           eventUse.quiescentAtLifetimeBoundaries});
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
