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

#include <limits>
#include <map>
#include <optional>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

std::optional<CanonicalEventColorPressureMap>
mlir::pto::evaluateCanonicalEventColorPressure(
    ArrayRef<CanonicalEvent> events, unsigned eventIdMax,
    const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds) {
  std::map<CanonicalEventDomainKey, std::vector<SyncWeightedInterval>>
      intervals;
  std::map<CanonicalEventDomainKey, std::vector<std::size_t>> owners;
  for (auto [eventIndex, event] : llvm::enumerate(events)) {
    const CanonicalEventDomainKey domain{event.sourcePipe, event.targetPipe};
    intervals[domain].push_back(
        {{event.intervalBegin, event.intervalEnd}, event.width});
    owners[domain].push_back(eventIndex);
  }

  CanonicalEventColorPressureMap result;
  for (const auto &entry : intervals) {
    std::vector<unsigned> reserved;
    auto reservation = reservedIds.find(entry.first);
    if (reservation != reservedIds.end()) {
      reserved.assign(reservation->second.begin(), reservation->second.end());
    }
    const SyncIntervalPressure pressure =
        evaluateSyncIntervalPressure(entry.second, eventIdMax, reserved);
    if (!pressure) {
      return std::nullopt;
    }

    CanonicalEventColorPressure domain;
    domain.required = pressure.required;
    domain.available = pressure.available;
    domain.overflow = pressure.overflow;
    domain.maximumPoint = pressure.maximumPoint;
    for (std::size_t interval : pressure.maximumClique) {
      domain.maximumCliqueEvents.push_back(owners[entry.first][interval]);
    }
    result.emplace(entry.first, std::move(domain));
  }
  return result;
}

std::size_t mlir::pto::calculateCanonicalEventColorOverflow(
    ArrayRef<CanonicalEvent> events, unsigned eventIdMax,
    const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds) {
  const std::optional<CanonicalEventColorPressureMap> pressure =
      evaluateCanonicalEventColorPressure(events, eventIdMax, reservedIds);
  if (!pressure) {
    return std::numeric_limits<std::size_t>::max();
  }
  std::size_t overflow = 0;
  for (const auto &entry : *pressure) {
    if (overflow >
        std::numeric_limits<std::size_t>::max() - entry.second.overflow) {
      return std::numeric_limits<std::size_t>::max();
    }
    overflow += entry.second.overflow;
  }
  return overflow;
}
