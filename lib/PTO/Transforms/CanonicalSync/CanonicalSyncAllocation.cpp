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
#include <map>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct EventLane {
  std::size_t event = 0;
  unsigned lane = 0;
};

std::vector<unsigned> getAvailableIds(unsigned maximum,
                                      const std::set<unsigned> &reserved) {
  std::vector<unsigned> available;
  for (unsigned id = 0; id < maximum; ++id) {
    if (!reserved.count(id)) {
      available.push_back(id);
    }
  }
  return available;
}

} // namespace

LogicalResult CanonicalSyncPlanBuilder::allocateEvents() {
  std::map<CanonicalEventDomainKey, SmallVector<std::size_t, 8>> eventsByDomain;
  for (auto [index, event] : llvm::enumerate(plan_.events_)) {
    eventsByDomain[{event.sourcePipe, event.targetPipe}].push_back(index);
  }

  for (const auto &entry : eventsByDomain) {
    const CanonicalEventDomainKey key = entry.first;
    const SmallVector<std::size_t, 8> &eventIndices = entry.second;
    std::vector<EventLane> lanes;
    std::vector<SyncInterval> intervals;
    for (std::size_t eventIndex : eventIndices) {
      CanonicalEvent &event = plan_.events_[eventIndex];
      if (event.width == 0 || event.width > kMaxMultiBufferCount) {
        return func_.emitError()
               << "canonical sync event width " << event.width
               << " is outside the supported multi-buffer range";
      }
      for (unsigned lane = 0; lane < event.width; ++lane) {
        lanes.push_back({eventIndex, lane});
        intervals.push_back({event.intervalBegin, event.intervalEnd});
      }
    }

    const SyncColoring coloring = colorSyncIntervals(intervals);
    const std::set<unsigned> &reserved = reservedIds_[key];
    const std::vector<unsigned> available =
        getAvailableIds(eventIdMax_, reserved);

    CanonicalEventDomain domain;
    domain.sourcePipe = key.source;
    domain.targetPipe = key.target;
    domain.eventCount = eventIndices.size();
    domain.availableIds = available.size();
    domain.colorCount = coloring.colorCount;
    domain.reservedIds.append(reserved.begin(), reserved.end());
    plan_.domains_.push_back(domain);

    if (coloring.colorCount > available.size()) {
      func_.emitError() << "PTOCanonicalSync cannot allocate domain "
                        << stringifyPIPE(static_cast<PIPE>(key.source))
                        << " -> "
                        << stringifyPIPE(static_cast<PIPE>(key.target)) << ": "
                        << coloring.colorCount << " colors requested but only "
                        << available.size()
                        << " event IDs are available; the interval-domain "
                           "color count is exact";
      return failure();
    }
    for (std::size_t vertex = 0; vertex < lanes.size(); ++vertex) {
      CanonicalEvent &event = plan_.events_[lanes[vertex].event];
      if (event.eventIds.empty()) {
        event.eventIds.resize(event.width);
      }
      event.eventIds[lanes[vertex].lane] = available[coloring.colors[vertex]];
    }
  }
  return success();
}
