// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageLifecycle.h"

#include <algorithm>
#include <limits>
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

struct PendingComponent {
  SyncCoverStorageLifecycleComponent component;
  std::map<SlotKey, SyncCoverStorageLifecycleSlotId> slots;
  std::map<SyncCoverStorageAccessId, SyncCoverStorageLifecycleEpochId> epochs;
  std::set<SyncCoverDemandId> demands;
};

class LifecycleWorkBudget {
public:
  LifecycleWorkBudget(std::size_t limit, std::size_t &used)
      : limit_(limit), used_(used) {}

  bool consume(std::size_t amount = 1) {
    if (amount > limit_ - used_) {
      failed_ = true;
      return false;
    }
    used_ += amount;
    return true;
  }

  bool failed() const { return failed_; }

private:
  std::size_t limit_;
  std::size_t &used_;
  bool failed_ = false;
};

/// Conservatively charge an ordered-container operation as a linear search.
/// This intentionally over-approximates std::map/std::set logarithmic work.
bool consumeOrderedOperation(LifecycleWorkBudget &budget,
                             std::size_t elementCount) {
  return budget.consume(elementCount) && budget.consume();
}

/// The graph LCA walks at most four full parent chains; finding the nearest
/// enclosing loop walks at most one more. Reserve their complete portable
/// upper bound before calling either graph query.
bool consumeScopeOwnerWork(LifecycleWorkBudget &budget,
                           std::size_t scopeCount) {
  for (unsigned walk = 0; walk < 5; ++walk) {
    if (!budget.consume(scopeCount)) {
      return false;
    }
  }
  return budget.consume(2);
}

bool checkedIncrement(std::size_t &value, std::size_t limit) {
  const bool exhausted = value == std::numeric_limits<std::size_t>::max() ||
                         value >= limit;
  if (exhausted) {
    return false;
  }
  ++value;
  return true;
}

bool intervalEquals(const SyncCoverStorageInterval &first,
                    const SyncCoverStorageInterval &second) {
  return first.begin == second.begin && first.end == second.end;
}

bool modeReads(SyncCoverStorageAccessMode mode) {
  return (static_cast<unsigned>(mode) &
          static_cast<unsigned>(SyncCoverStorageAccessMode::Read)) != 0;
}

bool modeWrites(SyncCoverStorageAccessMode mode) {
  return (static_cast<unsigned>(mode) &
          static_cast<unsigned>(SyncCoverStorageAccessMode::Write)) != 0;
}

SyncCoverStorageLifecycleEdgeKindMask
getCompatibleKinds(const SyncCoverDemand &demand,
                   const SyncCoverStorageAccess &source,
                   const SyncCoverStorageAccess &target) {
  SyncCoverStorageLifecycleEdgeKindMask result = 0;
  for (SyncCoverDemandKind kind : demand.provenanceKinds) {
    switch (kind) {
    case SyncCoverDemandKind::SSA:
      break;
    case SyncCoverDemandKind::MemoryRAW: {
      const bool compatible =
          modeWrites(source.mode) && modeReads(target.mode);
      if (compatible) {
        result |= syncCoverStorageLifecycleEdgeKindBit(
            SyncCoverStorageLifecycleEdgeKind::Ready);
      }
      break;
    }
    case SyncCoverDemandKind::MemoryWAR: {
      const bool compatible =
          modeReads(source.mode) && modeWrites(target.mode);
      if (compatible) {
        result |= syncCoverStorageLifecycleEdgeKindBit(
            SyncCoverStorageLifecycleEdgeKind::Release);
      }
      break;
    }
    case SyncCoverDemandKind::MemoryWAW: {
      const bool compatible =
          modeWrites(source.mode) && modeWrites(target.mode);
      if (compatible) {
        result |= syncCoverStorageLifecycleEdgeKindBit(
            SyncCoverStorageLifecycleEdgeKind::Exclusion);
      }
      break;
    }
    }
  }
  return result;
}

std::optional<SyncCoverScopeId>
getLifecycleOwner(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                  const SyncCoverStorageAccess &source,
                  const SyncCoverStorageAccess &target,
                  LifecycleWorkBudget &budget) {
  if (demand.distance != 0) {
    return budget.consume() ? std::optional<SyncCoverScopeId>(demand.scope)
                            : std::nullopt;
  }
  if (!consumeScopeOwnerWork(budget, graph.getScopes().size())) {
    return std::nullopt;
  }
  const std::optional<SyncCoverScopeId> common = graph.getLowestCommonScope(
      graph.getNodes()[source.node].scope, graph.getNodes()[target.node].scope);
  if (!common) {
    return std::nullopt;
  }
  return graph.getNearestEnclosingLoop(*common, true).value_or(0);
}

} // namespace

SyncCoverStorageLifecycleIndex mlir::pto::buildSyncCoverStorageLifecycleIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleLimits &limits) {
  SyncCoverStorageLifecycleIndex result;
  const auto fail = [&](SyncCoverStorageLifecycleStatistics statistics,
                        SyncCoverStorageLifecycleError error) {
    result.components_.clear();
    statistics.components = 0;
    statistics.slots = 0;
    statistics.epochs = 0;
    statistics.edges = 0;
    statistics.demandIncidences = 0;
    statistics.truncated =
        error == SyncCoverStorageLifecycleError::LimitExceeded;
    result.statistics_ = statistics;
    result.error_ = error;
    return std::move(result);
  };
  const bool invalidLimit =
      limits.maximumWorkUnits == 0 || limits.maximumComponents == 0 ||
      limits.maximumSlots == 0 || limits.maximumEpochs == 0 ||
      limits.maximumEdges == 0 || limits.maximumDemandIncidences == 0;
  if (invalidLimit) {
    return fail({}, SyncCoverStorageLifecycleError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return fail({}, SyncCoverStorageLifecycleError::InvalidGraph);
  }

  const std::vector<SyncCoverNode> &nodes = graph.getNodes();
  const std::vector<SyncCoverStorageAccess> &accesses =
      graph.getStorageAccesses();
  const std::vector<SyncCoverStorageWitness> &witnesses =
      graph.getStorageWitnesses();
  const std::vector<SyncCoverDemand> &demands = graph.getDemands();
  std::map<ComponentKey, PendingComponent> pending;
  SyncCoverStorageLifecycleStatistics statistics;
  LifecycleWorkBudget workBudget(limits.maximumWorkUnits,
                                 statistics.workUnits);
  std::size_t totalSlots = 0;
  std::size_t totalEpochs = 0;
  std::size_t totalEdges = 0;
  std::size_t totalDemandIncidences = 0;

  for (SyncCoverDemandId demandId = 0; demandId < demands.size(); ++demandId) {
    if (!workBudget.consume()) {
      return fail(statistics, SyncCoverStorageLifecycleError::LimitExceeded);
    }
    const SyncCoverDemand &demand = demands[demandId];
    for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
      const bool witnessWorkAvailable =
          workBudget.consume() &&
          workBudget.consume(demand.provenanceKinds.size());
      if (!witnessWorkAvailable) {
        return fail(statistics, SyncCoverStorageLifecycleError::LimitExceeded);
      }
      if (witnessId >= witnesses.size()) {
        return fail(statistics, SyncCoverStorageLifecycleError::InvalidGraph);
      }
      const SyncCoverStorageWitness &witness = witnesses[witnessId];
      const bool invalidWitnessAccess =
          witness.sourceAccess >= accesses.size() ||
          witness.targetAccess >= accesses.size();
      if (invalidWitnessAccess) {
        return fail(statistics, SyncCoverStorageLifecycleError::InvalidGraph);
      }
      const SyncCoverStorageAccess &source = accesses[witness.sourceAccess];
      const SyncCoverStorageAccess &target = accesses[witness.targetAccess];
      const bool invalidAccessNode =
          source.node >= nodes.size() || target.node >= nodes.size();
      if (invalidAccessNode) {
        return fail(statistics, SyncCoverStorageLifecycleError::InvalidGraph);
      }
      const SyncCoverStorageLifecycleEdgeKindMask kinds =
          getCompatibleKinds(demand, source, target);
      const bool exactWholeSlot =
          source.exactPhysical && target.exactPhysical &&
          source.domain == target.domain && source.family == target.family &&
          intervalEquals(source.extent, target.extent) &&
          intervalEquals(source.extent, witness.overlap);
      if (!exactWholeSlot || kinds == 0) {
        if (!checkedIncrement(statistics.ineligibleWitnesses,
                              std::numeric_limits<std::size_t>::max())) {
          return fail(statistics,
                      SyncCoverStorageLifecycleError::ArithmeticOverflow);
        }
        continue;
      }
      const std::optional<SyncCoverScopeId> owner =
          getLifecycleOwner(graph, demand, source, target, workBudget);
      if (workBudget.failed()) {
        return fail(statistics, SyncCoverStorageLifecycleError::LimitExceeded);
      }
      if (!owner || *owner >= graph.getScopes().size()) {
        return fail(statistics, SyncCoverStorageLifecycleError::InvalidGraph);
      }
      if (!checkedIncrement(statistics.eligibleWitnesses,
                            std::numeric_limits<std::size_t>::max())) {
        return fail(statistics,
                    SyncCoverStorageLifecycleError::ArithmeticOverflow);
      }

      const ComponentKey componentKey{source.family, *owner};
      if (!consumeOrderedOperation(workBudget, pending.size())) {
        return fail(statistics, SyncCoverStorageLifecycleError::LimitExceeded);
      }
      auto componentPosition = pending.find(componentKey);
      if (componentPosition == pending.end()) {
        const bool componentLimitReached =
            pending.size() >= limits.maximumComponents;
        if (componentLimitReached) {
          return fail(statistics,
                      SyncCoverStorageLifecycleError::LimitExceeded);
        }
        if (!consumeOrderedOperation(workBudget, pending.size())) {
          return fail(statistics,
                      SyncCoverStorageLifecycleError::LimitExceeded);
        }
        componentPosition = pending.try_emplace(componentKey).first;
        componentPosition->second.component.family = source.family;
        componentPosition->second.component.owningScope = *owner;
      }
      PendingComponent &pendingComponent = componentPosition->second;
      const SlotKey slotKey{source.domain, source.extent.begin,
                            source.extent.end};
      if (!consumeOrderedOperation(workBudget,
                                   pendingComponent.slots.size())) {
        return fail(statistics, SyncCoverStorageLifecycleError::LimitExceeded);
      }
      auto slotPosition = pendingComponent.slots.find(slotKey);
      if (slotPosition == pendingComponent.slots.end()) {
        if (!checkedIncrement(totalSlots, limits.maximumSlots)) {
          return fail(statistics,
                      SyncCoverStorageLifecycleError::LimitExceeded);
        }
        if (!consumeOrderedOperation(workBudget,
                                     pendingComponent.slots.size())) {
          return fail(statistics,
                      SyncCoverStorageLifecycleError::LimitExceeded);
        }
        slotPosition =
            pendingComponent.slots
                .try_emplace(slotKey, pendingComponent.component.slots.size())
                .first;
        pendingComponent.component.slots.push_back({slotPosition->second,
                                                    source.domain,
                                                    source.family,
                                                    source.extent,
                                                    {}});
      }
      const SyncCoverStorageLifecycleSlotId slot = slotPosition->second;

      const auto addEpoch = [&](const SyncCoverStorageAccess &access)
          -> std::optional<SyncCoverStorageLifecycleEpochId> {
        if (!consumeOrderedOperation(workBudget,
                                     pendingComponent.epochs.size())) {
          return std::nullopt;
        }
        auto position = pendingComponent.epochs.find(access.id);
        if (position == pendingComponent.epochs.end()) {
          if (!checkedIncrement(totalEpochs, limits.maximumEpochs)) {
            return std::nullopt;
          }
          if (!consumeOrderedOperation(workBudget,
                                       pendingComponent.epochs.size())) {
            return std::nullopt;
          }
          position = pendingComponent.epochs
                         .try_emplace(access.id,
                                      pendingComponent.component.epochs.size())
                         .first;
          const SyncCoverNode &node = nodes[access.node];
          pendingComponent.component.epochs.push_back(
              {position->second, access.id, slot, access.node, access.mode,
               node.resource, node.scope});
          pendingComponent.component.slots[slot].accesses.push_back(access.id);
        }
        return position->second;
      };
      const std::optional<SyncCoverStorageLifecycleEpochId> sourceEpoch =
          addEpoch(source);
      const std::optional<SyncCoverStorageLifecycleEpochId> targetEpoch =
          addEpoch(target);
      if (!sourceEpoch || !targetEpoch) {
        return fail(statistics, SyncCoverStorageLifecycleError::LimitExceeded);
      }
      if (!checkedIncrement(totalEdges, limits.maximumEdges)) {
        return fail(statistics, SyncCoverStorageLifecycleError::LimitExceeded);
      }
      pendingComponent.component.edges.push_back(
          {pendingComponent.component.edges.size(), demandId, witnessId,
           *sourceEpoch, *targetEpoch, kinds, demand.scope, demand.distance});
      if (!consumeOrderedOperation(workBudget,
                                   pendingComponent.demands.size())) {
        return fail(statistics, SyncCoverStorageLifecycleError::LimitExceeded);
      }
      const bool newDemand = pendingComponent.demands.find(demandId) ==
                             pendingComponent.demands.end();
      if (newDemand) {
        if (!checkedIncrement(totalDemandIncidences,
                              limits.maximumDemandIncidences)) {
          return fail(statistics,
                      SyncCoverStorageLifecycleError::LimitExceeded);
        }
        if (!consumeOrderedOperation(workBudget,
                                     pendingComponent.demands.size())) {
          return fail(statistics,
                      SyncCoverStorageLifecycleError::LimitExceeded);
        }
        pendingComponent.demands.insert(demandId);
      }
    }
  }

  result.components_.reserve(pending.size());
  for (auto &[key, entry] : pending) {
    const bool publicationWorkAvailable =
        workBudget.consume() && workBudget.consume(entry.demands.size());
    if (!publicationWorkAvailable) {
      return fail(statistics, SyncCoverStorageLifecycleError::LimitExceeded);
    }
    (void)key;
    entry.component.id = result.components_.size();
    entry.component.demands.assign(entry.demands.begin(), entry.demands.end());
    result.components_.push_back(std::move(entry.component));
  }
  statistics.components = result.components_.size();
  statistics.slots = totalSlots;
  statistics.epochs = totalEpochs;
  statistics.edges = totalEdges;
  statistics.demandIncidences = totalDemandIncidences;
  result.statistics_ = statistics;
  return result;
}
