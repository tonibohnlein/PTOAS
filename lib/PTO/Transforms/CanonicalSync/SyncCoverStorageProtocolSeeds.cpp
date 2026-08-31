// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolSeeds.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

using ComponentKey =
    std::pair<SyncCoverStorageAccessFamilyId, SyncCoverScopeId>;
using SlotKey =
    std::tuple<SyncCoverStorageDomainId, std::uint64_t, std::uint64_t>;

class WorkBudget {
public:
  WorkBudget(std::size_t maximum, std::size_t &used)
      : maximum_(maximum), used_(used) {}

  bool consume(std::size_t amount = 1) {
    if (amount > maximum_ - used_) {
      return false;
    }
    used_ += amount;
    return true;
  }

private:
  std::size_t maximum_ = 0;
  std::size_t &used_;
};

bool consumeOrderedOperation(WorkBudget &budget, std::size_t elementCount) {
  if (!budget.consume(elementCount)) {
    return false;
  }
  return budget.consume();
}

bool consumeScopeLcaWork(WorkBudget &budget, std::size_t scopeCount) {
  for (unsigned walk = 0; walk < 4; ++walk) {
    if (!budget.consume(scopeCount)) {
      return false;
    }
  }
  return budget.consume(4);
}

std::optional<std::size_t> findRoot(std::vector<std::size_t> &parents,
                                    std::size_t value, WorkBudget &budget) {
  std::size_t root = value;
  while (parents[root] != root) {
    if (!budget.consume()) {
      return std::nullopt;
    }
    root = parents[root];
  }
  while (parents[value] != value) {
    if (!budget.consume()) {
      return std::nullopt;
    }
    const std::size_t next = parents[value];
    parents[value] = root;
    value = next;
  }
  return root;
}

bool unite(std::vector<std::size_t> &parents, std::size_t first,
           std::size_t second, WorkBudget &budget) {
  const std::optional<std::size_t> firstRoot =
      findRoot(parents, first, budget);
  const std::optional<std::size_t> secondRoot =
      findRoot(parents, second, budget);
  if (!firstRoot || !secondRoot) {
    return false;
  }
  first = *firstRoot;
  second = *secondRoot;
  if (first == second) {
    return true;
  }
  if (!budget.consume()) {
    return false;
  }
  if (second < first) {
    std::swap(first, second);
  }
  parents[second] = first;
  return true;
}

SyncCoverStorageLifecycleEdgeKindMask readyReleaseKinds() {
  return syncCoverStorageLifecycleEdgeKindBit(
             SyncCoverStorageLifecycleEdgeKind::Ready) |
         syncCoverStorageLifecycleEdgeKindBit(
             SyncCoverStorageLifecycleEdgeKind::Release);
}

} // namespace

SyncCoverStorageProtocolSeedIndex
mlir::pto::buildSyncCoverStorageProtocolSeedIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolSeedLimits &limits) {
  SyncCoverStorageProtocolSeedIndex result;
  const auto fail = [&](SyncCoverStorageProtocolSeedStatistics statistics,
                        SyncCoverStorageProtocolSeedError error) {
    result.seeds_.clear();
    statistics.seeds = 0;
    statistics.readyReleaseSeeds = 0;
    statistics.componentIncidences = 0;
    statistics.slotIncidences = 0;
    statistics.sccIncidences = 0;
    statistics.demandIncidences = 0;
    statistics.maximumSeedComponents = 0;
    statistics.maximumSeedSlots = 0;
    statistics.maximumSeedSccs = 0;
    statistics.truncated =
        error == SyncCoverStorageProtocolSeedError::LimitExceeded;
    result.statistics_ = statistics;
    result.error_ = error;
    return std::move(result);
  };
  const bool invalidLimit =
      limits.maximumWorkUnits == 0 || limits.maximumSeeds == 0 ||
      limits.maximumComponentIncidences == 0 ||
      limits.maximumSlotIncidences == 0 || limits.maximumSccIncidences == 0 ||
      limits.maximumDemandIncidences == 0;
  if (invalidLimit) {
    return fail({}, SyncCoverStorageProtocolSeedError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return fail({}, SyncCoverStorageProtocolSeedError::InvalidGraph);
  }
  if (!lifecycleIndex.isComplete()) {
    return fail({},
                SyncCoverStorageProtocolSeedError::IncompleteLifecycleIndex);
  }

  const std::vector<SyncCoverStorageLifecycleComponent> &components =
      lifecycleIndex.getComponents();
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  SyncCoverStorageProtocolSeedStatistics statistics;
  WorkBudget budget(limits.maximumWorkUnits, statistics.workUnits);
  const bool componentLimitExceeded =
      components.size() > limits.maximumComponentIncidences;
  if (componentLimitExceeded) {
    return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
  }
  if (!budget.consume(components.size())) {
    return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
  }
  statistics.componentIncidences = components.size();
  std::map<ComponentKey, SyncCoverStorageLifecycleComponentId> byOwner;
  std::vector<std::size_t> parents;
  parents.reserve(components.size());
  for (std::size_t componentId = 0; componentId < components.size();
       ++componentId) {
    parents.push_back(componentId);
  }

  for (SyncCoverStorageLifecycleComponentId componentId = 0;
       componentId < components.size(); ++componentId) {
    const SyncCoverStorageLifecycleComponent &component =
        components[componentId];
    const bool invalidComponent =
        component.id != componentId || component.owningScope >= scopes.size();
    if (invalidComponent) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::InvalidGraph);
    }
    if (!consumeOrderedOperation(budget, byOwner.size())) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
    }
    const auto inserted = byOwner.emplace(
        ComponentKey{component.family, component.owningScope}, componentId);
    if (!inserted.second) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::InvalidGraph);
    }
  }

  for (SyncCoverStorageLifecycleComponentId componentId = 0;
       componentId < components.size(); ++componentId) {
    const SyncCoverStorageLifecycleComponent &component =
        components[componentId];
    SyncCoverScopeId scope = component.owningScope;
    for (std::size_t depth = 0; depth < scopes.size(); ++depth) {
      if (!budget.consume()) {
        return fail(statistics,
                    SyncCoverStorageProtocolSeedError::LimitExceeded);
      }
      if (!consumeOrderedOperation(budget, byOwner.size())) {
        return fail(statistics,
                    SyncCoverStorageProtocolSeedError::LimitExceeded);
      }
      const auto ancestor = byOwner.find({component.family, scope});
      const bool hasDifferentAncestor =
          ancestor != byOwner.end() && ancestor->second != componentId;
      if (hasDifferentAncestor) {
        if (!unite(parents, componentId, ancestor->second, budget)) {
          return fail(statistics,
                      SyncCoverStorageProtocolSeedError::LimitExceeded);
        }
        break;
      }
      if (scope == 0) {
        break;
      }
      scope = scopes[scope].parent;
      if (scope >= scopes.size()) {
        return fail(statistics,
                    SyncCoverStorageProtocolSeedError::InvalidGraph);
      }
    }
  }

  std::map<std::size_t, std::vector<SyncCoverStorageLifecycleComponentId>>
      groups;
  for (SyncCoverStorageLifecycleComponentId componentId = 0;
       componentId < components.size(); ++componentId) {
    const std::optional<std::size_t> root =
        findRoot(parents, componentId, budget);
    if (!root) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
    }
    if (!consumeOrderedOperation(budget, groups.size())) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
    }
    auto group = groups.find(*root);
    if (group == groups.end()) {
      const bool seedLimitReached = groups.size() >= limits.maximumSeeds;
      if (seedLimitReached) {
        return fail(statistics,
                    SyncCoverStorageProtocolSeedError::LimitExceeded);
      }
      if (!consumeOrderedOperation(budget, groups.size())) {
        return fail(statistics,
                    SyncCoverStorageProtocolSeedError::LimitExceeded);
      }
      group = groups.try_emplace(*root).first;
    }
    if (!budget.consume()) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
    }
    group->second.push_back(componentId);
  }

  if (!budget.consume(groups.size())) {
    return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
  }
  result.seeds_.reserve(groups.size());
  for (const auto &[root, componentIds] : groups) {
    (void)root;
    if (componentIds.empty()) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::InvalidGraph);
    }
    if (!budget.consume(componentIds.size())) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
    }
    SyncCoverStorageProtocolSeed seed;
    seed.id = result.seeds_.size();
    seed.family = components[componentIds.front()].family;
    seed.owningScope = components[componentIds.front()].owningScope;
    std::map<SlotKey, SyncCoverStorageProtocolSlotRef> slots;
    std::set<SyncCoverDemandId> demands;
    for (SyncCoverStorageLifecycleComponentId componentId : componentIds) {
      const SyncCoverStorageLifecycleComponent &component =
          components[componentId];
      if (component.family != seed.family) {
        return fail(statistics,
                    SyncCoverStorageProtocolSeedError::InvalidGraph);
      }
      // getLowestCommonScope measures both depths, aligns them, and ascends
      // both chains. Reserve its complete portable bound before querying.
      if (!consumeScopeLcaWork(budget, scopes.size())) {
        return fail(statistics,
                    SyncCoverStorageProtocolSeedError::LimitExceeded);
      }
      const std::optional<SyncCoverScopeId> common =
          graph.getLowestCommonScope(seed.owningScope, component.owningScope);
      if (!common) {
        return fail(statistics,
                    SyncCoverStorageProtocolSeedError::InvalidGraph);
      }
      seed.owningScope = *common;
      seed.components.push_back(componentId);
      for (std::size_t slotId = 0; slotId < component.slots.size(); ++slotId) {
        const SyncCoverStorageLifecycleSlot &slot = component.slots[slotId];
        const bool invalidSlot =
            slot.id != slotId || slot.family != seed.family;
        if (invalidSlot) {
          return fail(statistics,
                      SyncCoverStorageProtocolSeedError::InvalidGraph);
        }
        if (!consumeOrderedOperation(budget, slots.size())) {
          return fail(statistics,
                      SyncCoverStorageProtocolSeedError::LimitExceeded);
        }
        const SlotKey slotKey{slot.domain, slot.extent.begin, slot.extent.end};
        const bool isNewSlot = slots.find(slotKey) == slots.end();
        if (isNewSlot) {
          const bool slotLimitReached =
              statistics.slotIncidences >= limits.maximumSlotIncidences;
          if (slotLimitReached) {
            return fail(statistics,
                        SyncCoverStorageProtocolSeedError::LimitExceeded);
          }
          if (!consumeOrderedOperation(budget, slots.size())) {
            return fail(statistics,
                        SyncCoverStorageProtocolSeedError::LimitExceeded);
          }
          slots.emplace(slotKey,
                        SyncCoverStorageProtocolSlotRef{componentId, slot.id});
          ++statistics.slotIncidences;
        }
      }
      for (std::size_t sccId = 0; sccId < component.sccs.size(); ++sccId) {
        const SyncCoverStorageLifecycleScc &scc = component.sccs[sccId];
        const bool invalidScc = scc.id != sccId;
        if (invalidScc) {
          return fail(statistics,
                      SyncCoverStorageProtocolSeedError::InvalidGraph);
        }
        if (!budget.consume()) {
          return fail(statistics,
                      SyncCoverStorageProtocolSeedError::LimitExceeded);
        }
        const bool isReadyReleaseScc =
            scc.cyclic && (scc.kinds & readyReleaseKinds()) ==
                              readyReleaseKinds();
        if (isReadyReleaseScc) {
          const bool sccLimitReached =
              statistics.sccIncidences >= limits.maximumSccIncidences;
          if (sccLimitReached) {
            return fail(statistics,
                        SyncCoverStorageProtocolSeedError::LimitExceeded);
          }
          if (!budget.consume()) {
            return fail(statistics,
                        SyncCoverStorageProtocolSeedError::LimitExceeded);
          }
          seed.readyReleaseSccs.push_back({componentId, scc.id});
          ++statistics.sccIncidences;
        }
      }
      for (std::size_t edgeId = 0; edgeId < component.edges.size();
           ++edgeId) {
        const SyncCoverStorageLifecycleEdge &edge = component.edges[edgeId];
        const bool invalidEdge = edge.id != edgeId;
        if (invalidEdge) {
          return fail(statistics,
                      SyncCoverStorageProtocolSeedError::InvalidGraph);
        }
        if (!budget.consume()) {
          return fail(statistics,
                      SyncCoverStorageProtocolSeedError::LimitExceeded);
        }
        seed.kinds |= edge.kinds;
        seed.maximumDistance = std::max(seed.maximumDistance, edge.distance);
      }
      for (SyncCoverDemandId demand : component.demands) {
        if (demand >= graph.getDemands().size()) {
          return fail(statistics,
                      SyncCoverStorageProtocolSeedError::InvalidGraph);
        }
        if (!consumeOrderedOperation(budget, demands.size())) {
          return fail(statistics,
                      SyncCoverStorageProtocolSeedError::LimitExceeded);
        }
        const bool isNewDemand = demands.find(demand) == demands.end();
        if (isNewDemand) {
          const bool demandLimitReached =
              statistics.demandIncidences >=
              limits.maximumDemandIncidences;
          if (demandLimitReached) {
            return fail(statistics,
                        SyncCoverStorageProtocolSeedError::LimitExceeded);
          }
          if (!consumeOrderedOperation(budget, demands.size())) {
            return fail(statistics,
                        SyncCoverStorageProtocolSeedError::LimitExceeded);
          }
          demands.insert(demand);
          ++statistics.demandIncidences;
        }
      }
    }
    if (!budget.consume(slots.size())) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
    }
    seed.slots.reserve(slots.size());
    for (const auto &[key, slot] : slots) {
      (void)key;
      seed.slots.push_back(slot);
    }
    if (!budget.consume(demands.size())) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
    }
    seed.demands.assign(demands.begin(), demands.end());
    const bool isReadyReleaseSeed = !seed.readyReleaseSccs.empty() &&
                                    (seed.kinds & readyReleaseKinds()) ==
                                        readyReleaseKinds();
    if (isReadyReleaseSeed) {
      ++statistics.readyReleaseSeeds;
    }
    statistics.maximumSeedComponents =
        std::max(statistics.maximumSeedComponents, seed.components.size());
    statistics.maximumSeedSlots =
        std::max(statistics.maximumSeedSlots, seed.slots.size());
    statistics.maximumSeedSccs =
        std::max(statistics.maximumSeedSccs, seed.readyReleaseSccs.size());
    if (!budget.consume()) {
      return fail(statistics, SyncCoverStorageProtocolSeedError::LimitExceeded);
    }
    result.seeds_.push_back(std::move(seed));
    statistics.seeds = result.seeds_.size();
  }

  result.statistics_ = statistics;
  return result;
}
