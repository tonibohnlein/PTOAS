// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <queue>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir::pto;

SyncColoring
mlir::pto::colorSyncIntervals(const std::vector<SyncInterval> &intervals) {
  SyncColoring result;
  result.colors.resize(intervals.size());

  std::vector<std::size_t> order(intervals.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t first, std::size_t second) {
                     if (intervals[first].begin != intervals[second].begin) {
                       return intervals[first].begin < intervals[second].begin;
                     }
                     if (intervals[first].end != intervals[second].end) {
                       return intervals[first].end < intervals[second].end;
                     }
                     return first < second;
                   });

  using ActiveColor = std::pair<std::size_t, unsigned>;
  std::priority_queue<ActiveColor, std::vector<ActiveColor>,
                      std::greater<ActiveColor>>
      active;
  std::priority_queue<unsigned, std::vector<unsigned>, std::greater<unsigned>>
      available;
  for (std::size_t intervalIndex : order) {
    const SyncInterval &interval = intervals[intervalIndex];
    while (!active.empty() && active.top().first < interval.begin) {
      available.push(active.top().second);
      active.pop();
    }
    unsigned color = 0;
    if (available.empty()) {
      color = result.colorCount++;
    } else {
      color = available.top();
      available.pop();
    }
    result.colors[intervalIndex] = color;
    active.emplace(interval.end, color);
  }
  return result;
}

bool mlir::pto::verifySyncIntervalAllocation(
    unsigned eventIdMax, const std::vector<unsigned> &reservedIds,
    const std::vector<SyncAllocatedInterval> &allocation) {
  const std::set<unsigned> reserved(reservedIds.begin(), reservedIds.end());
  for (std::size_t index = 0; index < allocation.size(); ++index) {
    const SyncAllocatedInterval &current = allocation[index];
    if (current.eventId >= eventIdMax || reserved.count(current.eventId) != 0) {
      return false;
    }
    for (std::size_t other = 0; other < index; ++other) {
      const SyncAllocatedInterval &previous = allocation[other];
      const bool overlaps = current.interval.end >= previous.interval.begin &&
                            previous.interval.end >= current.interval.begin;
      if (overlaps && current.eventId == previous.eventId) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::size_t> mlir::pto::findMaximumIntervalClique(
    const std::vector<SyncInterval> &intervals) {
  std::vector<std::size_t> order(intervals.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
      order.begin(), order.end(), [&](std::size_t first, std::size_t second) {
        return std::tie(intervals[first].begin, intervals[first].end, first) <
               std::tie(intervals[second].begin, intervals[second].end, second);
      });

  std::set<std::pair<std::size_t, std::size_t>> activeByEnd;
  std::set<std::size_t> activeIndices;
  std::vector<std::size_t> maximum;
  for (std::size_t position = 0; position < order.size();) {
    const std::size_t point = intervals[order[position]].begin;
    while (!activeByEnd.empty()) {
      if (activeByEnd.begin()->first >= point) {
        break;
      }
      activeIndices.erase(activeByEnd.begin()->second);
      activeByEnd.erase(activeByEnd.begin());
    }
    while (position < order.size()) {
      if (intervals[order[position]].begin != point) {
        break;
      }
      const std::size_t interval = order[position++];
      activeByEnd.emplace(intervals[interval].end, interval);
      activeIndices.insert(interval);
    }
    std::vector<std::size_t> candidate(activeIndices.begin(),
                                       activeIndices.end());
    const bool isLarger = candidate.size() > maximum.size();
    const bool isSameSize = candidate.size() == maximum.size();
    if (isLarger || (isSameSize && candidate < maximum)) {
      maximum = std::move(candidate);
    }
  }
  return maximum;
}
