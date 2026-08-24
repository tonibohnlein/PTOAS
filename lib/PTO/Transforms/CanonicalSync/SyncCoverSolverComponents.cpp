// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "SyncCoverSolverInternal.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_internal;

namespace {

class UnionFind {
public:
  explicit UnionFind(std::size_t size) : parents_(size), ranks_(size, 0) {
    std::iota(parents_.begin(), parents_.end(), 0);
  }

  std::size_t find(std::size_t value) {
    if (parents_[value] != value) {
      parents_[value] = find(parents_[value]);
    }
    return parents_[value];
  }

  void unite(std::size_t first, std::size_t second) {
    first = find(first);
    second = find(second);
    if (first == second) {
      return;
    }
    if (ranks_[first] < ranks_[second]) {
      std::swap(first, second);
    }
    parents_[second] = first;
    if (ranks_[first] == ranks_[second]) {
      ++ranks_[first];
    }
  }

private:
  std::vector<std::size_t> parents_;
  std::vector<unsigned> ranks_;
};

struct DomainInterval {
  SyncCoverResourceDomainId domain = 0;
  SyncCoverTimelinePosition begin = 0;
  SyncCoverTimelinePosition end = 0;
  SyncCoverMechanismId mechanism = 0;
};

} // namespace

ComponentBuildResult mlir::pto::sync_cover_internal::buildComponents(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    SyncCoverCoverageOracle &oracle, std::size_t exactThreshold) {
  ComponentBuildResult result;
  const std::size_t demandCount = activeDemands.size();
  const std::size_t mechanismCount = universe.getMechanisms().size();
  UnionFind sets(demandCount + mechanismCount);
  std::vector<bool> potential(mechanismCount, false);

  for (std::size_t localDemand = 0; localDemand < demandCount; ++localDemand) {
    const SyncCoverDemandTopologyResult topology =
        oracle.getDemandTopology(activeDemands[localDemand]);
    if (!topology) {
      result.valid = false;
      return result;
    }
    for (SyncCoverMechanismId mechanism : topology.potentialMechanisms) {
      if (mechanism >= mechanismCount) {
        result.valid = false;
        return result;
      }
      potential[mechanism] = true;
      sets.unite(localDemand, demandCount + mechanism);
    }
  }

  std::vector<DomainInterval> resourceIntervals;
  for (SyncCoverMechanismId mechanism = 0; mechanism < mechanismCount;
       ++mechanism) {
    if (!potential[mechanism]) {
      continue;
    }
    for (const SyncCoverResourceUse &use :
         universe.getMechanisms()[mechanism].resourceUses) {
      const std::optional<SyncCoverTimelineInterval> lifetime =
          getSyncCoverResourceLifetime(
              universe.getGraph(), universe.getMechanisms()[mechanism], use);
      if (!lifetime) {
        result.valid = false;
        return result;
      }
      resourceIntervals.push_back(
          {use.domain, lifetime->begin, lifetime->end, mechanism});
    }
    for (SyncCoverMechanismId conflict :
         universe.getMechanisms()[mechanism].conflicts) {
      if (conflict < mechanismCount && potential[conflict]) {
        sets.unite(demandCount + mechanism, demandCount + conflict);
      }
    }
  }

  std::sort(resourceIntervals.begin(), resourceIntervals.end(),
            [](const DomainInterval &first, const DomainInterval &second) {
              return std::tie(first.domain, first.begin, first.end,
                              first.mechanism) <
                     std::tie(second.domain, second.begin, second.end,
                              second.mechanism);
            });
  std::optional<DomainInterval> active;
  for (const DomainInterval &interval : resourceIntervals) {
    const bool startsComponent = !active || active->domain != interval.domain ||
                                 active->end < interval.begin;
    if (startsComponent) {
      active = interval;
      continue;
    }
    sets.unite(demandCount + active->mechanism,
               demandCount + interval.mechanism);
    active->end = std::max(active->end, interval.end);
  }

  std::map<std::size_t, SyncCoverSelectionComponent> byRoot;
  for (std::size_t localDemand = 0; localDemand < demandCount; ++localDemand) {
    byRoot[sets.find(localDemand)].demands.push_back(
        activeDemands[localDemand]);
  }
  for (SyncCoverMechanismId mechanism = 0; mechanism < mechanismCount;
       ++mechanism) {
    if (potential[mechanism]) {
      byRoot[sets.find(demandCount + mechanism)].mechanisms.push_back(
          mechanism);
    }
  }
  for (auto &entry : byRoot) {
    SyncCoverSelectionComponent component = std::move(entry.second);
    component.exact = component.mechanisms.size() <= exactThreshold;
    result.components.push_back(std::move(component));
  }
  std::sort(result.components.begin(), result.components.end(),
            [](const auto &first, const auto &second) {
              return std::tie(first.demands, first.mechanisms) <
                     std::tie(second.demands, second.mechanisms);
            });
  for (std::size_t index = 0; index < result.components.size(); ++index) {
    result.components[index].id = index;
  }
  return result;
}

std::vector<std::vector<SyncCoverMechanismId>>
mlir::pto::sync_cover_internal::getComponentSeeds(
    const SyncCoverSelectionComponent &component,
    const std::vector<SyncCoverSelectionSeed> &seeds) {
  std::vector<std::vector<SyncCoverMechanismId>> result;
  for (const SyncCoverSelectionSeed &seed : seeds) {
    std::vector<SyncCoverMechanismId> selected;
    std::set_intersection(seed.mechanisms.begin(), seed.mechanisms.end(),
                          component.mechanisms.begin(),
                          component.mechanisms.end(),
                          std::back_inserter(selected));
    result.push_back(std::move(selected));
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}
