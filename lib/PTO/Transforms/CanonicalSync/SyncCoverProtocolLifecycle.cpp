// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "SyncCoverProtocolInternal.h"

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageLifecycle.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolGroups.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::sync_cover_protocol_detail;

namespace {

struct LifecycleEdge {
  SyncCoverDemandId demand = 0;
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageAccessId sourceAccess = 0;
  SyncCoverStorageAccessId targetAccess = 0;
  SyncCoverStorageWitnessId witness = 0;
  unsigned distance = 0;
};

/// Schedule facts derived only from one exact storage-lifecycle SCC.  These
/// deliberately do not reuse SyncCoverBasicOwnershipCertificate: the latter
/// is legacy recognizer input stored on the graph, whereas generic lifecycle
/// synthesis must remain a function of raw nodes, accesses, witnesses, and
/// demands.
struct DerivedLifecycleSlot {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval extent;
  std::vector<SyncCoverStorageAccessId> accesses;
};

struct DerivedLifecycleLane {
  std::size_t id = 0;
  std::vector<DerivedLifecycleSlot> slots;
};

struct DerivedLifecycleUse {
  std::size_t lane = 0;
  std::size_t producerLane = 0;
  std::vector<SyncCoverNodeId> producers;
  std::vector<SyncCoverNodeId> consumers;
  SyncCoverAnchor writeAcquireAnchor;
  SyncCoverAnchor readyAnchor;
  SyncCoverAnchor readAcquireAnchor;
  SyncCoverAnchor releaseAnchor;
};

struct DerivedLifecyclePath {
  SyncCoverScopeId scope = 0;
  std::vector<DerivedLifecycleUse> uses;
};

struct DerivedLifecycleCertificate {
  std::size_t id = 0;
  bool alternating = false;
  SyncCoverScopeId loopScope = 0;
  std::uint32_t producerResource = 0;
  std::uint32_t consumerResource = 0;
  std::vector<DerivedLifecycleLane> lanes;
  std::vector<DerivedLifecyclePath> paths;
  std::optional<SyncCoverControlId> periodicControl;
  std::vector<SyncCoverNodeId> initialProducers;
  SyncCoverAnchor initialWriteAcquireAnchor;
  SyncCoverAnchor initialReadyAnchor;
  std::size_t initialReadyLane = 0;
  std::vector<std::size_t> initiallyFreeLanes;
};

bool containsKind(const SyncCoverDemand &demand, SyncCoverDemandKind kind) {
  return std::find(demand.provenanceKinds.begin(), demand.provenanceKinds.end(),
                   kind) != demand.provenanceKinds.end();
}

bool checkedAccumulateSize(std::size_t &total, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - total) {
    return false;
  }
  total += amount;
  return true;
}

std::optional<std::size_t> comparisonWork(std::size_t count,
                                          std::size_t multiplier = 1) {
  std::size_t levels = 1;
  for (std::size_t remaining = count; remaining > 1;
       remaining = (remaining + 1) / 2) {
    ++levels;
  }
  if (count != 0 && levels > std::numeric_limits<std::size_t>::max() / count) {
    return std::nullopt;
  }
  std::size_t work = count * levels;
  if (multiplier != 0 &&
      work > std::numeric_limits<std::size_t>::max() / multiplier) {
    return std::nullopt;
  }
  return work * multiplier;
}

std::size_t lookupWork(std::size_t count) {
  std::size_t work = 1;
  for (std::size_t remaining = count; remaining > 1;
       remaining = (remaining + 1) / 2) {
    ++work;
  }
  return work;
}

std::optional<bool>
scopeContainsWithWork(const SyncCoverGraph &graph, SyncCoverScopeId ancestor,
                      SyncCoverScopeId descendant,
                      SyncCoverCoverageWorkBudget *workBudget) {
  if (!consumeWork(workBudget, graph.getScopes().size())) {
    return std::nullopt;
  }
  return graph.scopeContains(ancestor, descendant);
}

std::optional<SyncCoverScopeId>
nearestSharedLifecycleLoop(const SyncCoverGraph &graph, SyncCoverScopeId first,
                           SyncCoverScopeId second,
                           SyncCoverCoverageWorkBudget *workBudget) {
  const auto &scopes = graph.getScopes();
  if (first >= scopes.size() || second >= scopes.size()) {
    return std::nullopt;
  }
  for (SyncCoverScopeId candidate = first;;
       candidate = scopes[candidate].parent) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    const std::optional<bool> contains =
        scopeContainsWithWork(graph, candidate, second, workBudget);
    if (!contains) {
      return std::nullopt;
    }
    if (scopes[candidate].isLoop && *contains) {
      return candidate;
    }
    if (candidate == 0) {
      return std::nullopt;
    }
  }
}

bool finishOrder(const std::vector<std::vector<std::size_t>> &adjacency,
                 std::vector<std::size_t> &order,
                 SyncCoverCoverageWorkBudget *workBudget) {
  std::vector<bool> visited(adjacency.size());
  for (std::size_t root = 0; root < adjacency.size(); ++root) {
    if (visited[root]) {
      continue;
    }
    std::vector<std::pair<std::size_t, std::size_t>> stack{{root, 0}};
    visited[root] = true;
    while (!stack.empty()) {
      if (!consumeWork(workBudget)) {
        return false;
      }
      std::size_t node = stack.back().first;
      std::size_t &next = stack.back().second;
      if (next < adjacency[node].size()) {
        const std::size_t successor = adjacency[node][next++];
        if (!visited[successor]) {
          visited[successor] = true;
          stack.push_back({successor, 0});
        }
        continue;
      }
      order.push_back(node);
      stack.pop_back();
    }
  }
  return true;
}

bool collectComponent(const std::vector<std::vector<std::size_t>> &reverse,
                      std::size_t root, std::vector<bool> &visited,
                      std::vector<std::size_t> &component,
                      SyncCoverCoverageWorkBudget *workBudget) {
  std::vector<std::size_t> pending{root};
  visited[root] = true;
  while (!pending.empty()) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    const std::size_t node = pending.back();
    pending.pop_back();
    component.push_back(node);
    for (std::size_t predecessor : reverse[node]) {
      if (!consumeWork(workBudget)) {
        return false;
      }
      if (!visited[predecessor]) {
        visited[predecessor] = true;
        pending.push_back(predecessor);
      }
    }
  }
  return true;
}

} // namespace

SyncCoverLifecycleSccResult mlir::pto::discoverSyncCoverLifecycleSccs(
    const SyncCoverGraph &graph, SyncCoverProtocolLimits limits,
    SyncCoverCoverageWorkBudget *workBudget) {
  SyncCoverLifecycleSccResult result;
  if (!graphFitsProtocolLimits(graph, limits)) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  if (!graph.isStructureFrozen() || !graph.validate()) {
    result.error = SyncCoverProtocolError::InvalidGraph;
    return result;
  }
  if (graph.getNodes().size() > limits.maximumLifecycleVertices ||
      graph.getDemands().size() > limits.maximumResultRows ||
      graph.getStorageDomains().size() >
          limits.maximumLifecycleDomainIncidences) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  const auto normalizeIds = [&](auto &values) {
    const std::optional<std::size_t> sortWork = comparisonWork(values.size());
    if (!sortWork || !consumeWork(workBudget, *sortWork)) {
      return false;
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return true;
  };

  std::map<SyncCoverScopeId, std::vector<LifecycleEdge>> edgesByLoop;
  std::vector<SyncCoverLifecycleScc> pendingComponents;
  std::size_t witnessIncidences = 0;
  std::size_t edgeIncidences = 0;
  std::size_t lifecycleAccessIncidences = 0;
  using ExactAccessKey =
      std::tuple<SyncCoverStorageDomainId, std::uint64_t, std::uint64_t>;
  std::vector<std::pair<ExactAccessKey, SyncCoverStorageAccessId>>
      exactAccessIndex;
  exactAccessIndex.reserve(graph.getStorageAccesses().size());
  for (const SyncCoverStorageAccess &access : graph.getStorageAccesses()) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = access.id;
      return result;
    }
    if (!access.exactPhysical ||
        access.path != SyncCoverStorageAccessPath::PhysicalPipeline) {
      continue;
    }
    if (lifecycleAccessIncidences == limits.maximumLifecycleAccessIncidences) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = access.id;
      return result;
    }
    exactAccessIndex.push_back(
        {{access.domain, access.extent.begin, access.extent.end}, access.id});
    ++lifecycleAccessIncidences;
  }
  const std::optional<std::size_t> accessSortWork =
      comparisonWork(exactAccessIndex.size());
  if (!accessSortWork || !consumeWork(workBudget, *accessSortWork)) {
    result.error = SyncCoverProtocolError::WorkLimitExceeded;
    return result;
  }
  std::sort(exactAccessIndex.begin(), exactAccessIndex.end());
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = demandId;
      return result;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    // A ready edge may cross an iteration boundary: alternating prefetches
    // write the next lane near the tail of iteration i and consume it in
    // iteration i + d.  Omitting those positive-distance RAW edges loses one
    // half of the physical ownership cycle.  Conversely, a WAR participates
    // in lifecycle discovery only when it is a genuine reuse recurrence.
    const bool ready = containsKind(demand, SyncCoverDemandKind::MemoryRAW);
    const bool release = demand.distance != 0 &&
                         containsKind(demand, SyncCoverDemandKind::MemoryWAR);
    if (!ready && !release) {
      continue;
    }
    for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
      if (witnessIncidences == limits.maximumStorageWitnessIncidences ||
          !consumeWork(workBudget)) {
        result.error = workBudget && workBudget->exhausted
                           ? SyncCoverProtocolError::WorkLimitExceeded
                           : SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = demandId;
        return result;
      }
      ++witnessIncidences;
      const SyncCoverStorageWitness &witness =
          graph.getStorageWitnesses()[witnessId];
      const SyncCoverStorageAccess &source =
          graph.getStorageAccesses()[witness.sourceAccess];
      const SyncCoverStorageAccess &target =
          graph.getStorageAccesses()[witness.targetAccess];
      const bool exactSlot =
          source.exactPhysical && target.exactPhysical &&
          source.domain == target.domain &&
          source.extent.begin == target.extent.begin &&
          source.extent.end == target.extent.end &&
          source.path == SyncCoverStorageAccessPath::PhysicalPipeline &&
          target.path == SyncCoverStorageAccessPath::PhysicalPipeline;
      const bool rolesMatch = ready
                                  ? syncCoverStorageModeWrites(source.mode) &&
                                        syncCoverStorageModeReads(target.mode)
                                  : syncCoverStorageModeReads(source.mode) &&
                                        syncCoverStorageModeWrites(target.mode);
      if (!exactSlot || !rolesMatch) {
        continue;
      }
      const SyncCoverNode &sourceNode = graph.getNodes()[source.node];
      const SyncCoverNode &targetNode = graph.getNodes()[target.node];
      if (sourceNode.resource == targetNode.resource) {
        continue;
      }
      const bool recurrence = demand.distance != 0;
      const std::optional<SyncCoverScopeId> innermostLoop =
          recurrence ? std::optional<SyncCoverScopeId>(demand.scope)
                     : nearestSharedLifecycleLoop(graph, sourceNode.scope,
                                                  targetNode.scope, workBudget);
      if (!innermostLoop) {
        if (workBudget && workBudget->exhausted) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          result.invalidIndex = demandId;
          return result;
        }
        continue;
      }
      SyncCoverScopeId loop = *innermostLoop;
      while (true) {
        if (!consumeWork(workBudget)) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          result.invalidIndex = demandId;
          return result;
        }
        if (graph.getScopes()[loop].isLoop) {
          if (edgeIncidences == limits.maximumLifecycleEdges) {
            result.error = SyncCoverProtocolError::LimitExceeded;
            result.invalidIndex = demandId;
            return result;
          }
          edgesByLoop[loop].push_back(
              {demandId, demand.source, demand.target, sourceNode.resource,
               targetNode.resource, source.domain, witness.sourceAccess,
               witness.targetAccess, witnessId, demand.distance});
          ++edgeIncidences;
        }
        if (recurrence || loop == 0) {
          break;
        }
        loop = graph.getScopes()[loop].parent;
      }
    }
  }

  for (const auto &[loop, edges] : edgesByLoop) {
    std::vector<std::uint32_t> resources;
    if (edges.size() > std::numeric_limits<std::size_t>::max() / 2) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = loop;
      return result;
    }
    const std::optional<std::size_t> resourceBuildWork =
        comparisonWork(edges.size(), 2);
    if (!resourceBuildWork || !consumeWork(workBudget, *resourceBuildWork)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = loop;
      return result;
    }
    resources.reserve(edges.size() * 2);
    for (const LifecycleEdge &edge : edges) {
      resources.push_back(edge.sourceResource);
      resources.push_back(edge.targetResource);
    }
    if (!normalizeIds(resources) ||
        resources.size() > limits.maximumLifecycleVertices ||
        !consumeWork(workBudget, resources.size() + edges.size())) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = loop;
      return result;
    }
    struct IndexedEdge {
      const LifecycleEdge *edge = nullptr;
      std::size_t source = 0;
      std::size_t target = 0;
    };
    std::vector<IndexedEdge> indexedEdges;
    indexedEdges.reserve(edges.size());
    std::vector<std::vector<std::size_t>> adjacency(resources.size());
    std::vector<std::vector<std::size_t>> reverse(resources.size());
    for (const LifecycleEdge &edge : edges) {
      const std::optional<std::size_t> lookupWork =
          comparisonWork(resources.size(), 2);
      if (!lookupWork || !consumeWork(workBudget, *lookupWork)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = loop;
        return result;
      }
      const std::size_t source = static_cast<std::size_t>(
          std::lower_bound(resources.begin(), resources.end(),
                           edge.sourceResource) -
          resources.begin());
      const std::size_t target = static_cast<std::size_t>(
          std::lower_bound(resources.begin(), resources.end(),
                           edge.targetResource) -
          resources.begin());
      adjacency[source].push_back(target);
      reverse[target].push_back(source);
      indexedEdges.push_back({&edge, source, target});
    }
    std::vector<std::size_t> order;
    if (!finishOrder(adjacency, order, workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = loop;
      return result;
    }
    std::vector<bool> visited(resources.size());
    std::vector<std::size_t> componentOf(
        resources.size(), std::numeric_limits<std::size_t>::max());
    std::size_t componentCount = 0;
    while (!order.empty()) {
      const std::size_t root = order.back();
      order.pop_back();
      if (visited[root]) {
        continue;
      }
      std::vector<std::size_t> component;
      if (!collectComponent(reverse, root, visited, component, workBudget)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = loop;
        return result;
      }
      for (std::size_t vertex : component) {
        componentOf[vertex] = componentCount;
      }
      ++componentCount;
    }
    std::vector<std::vector<std::uint32_t>> resourcesByComponent(
        componentCount);
    for (std::size_t vertex = 0; vertex < resources.size(); ++vertex) {
      if (!consumeWork(workBudget)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = loop;
        return result;
      }
      resourcesByComponent[componentOf[vertex]].push_back(resources[vertex]);
    }
    std::vector<std::vector<const LifecycleEdge *>> edgesByComponent(
        componentCount);
    for (const IndexedEdge &indexed : indexedEdges) {
      if (!consumeWork(workBudget)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = loop;
        return result;
      }
      if (componentOf[indexed.source] == componentOf[indexed.target]) {
        edgesByComponent[componentOf[indexed.source]].push_back(indexed.edge);
      }
    }
    for (std::size_t component = 0; component < componentCount; ++component) {
      SyncCoverLifecycleScc lifecycle;
      lifecycle.loopScope = loop;
      lifecycle.resources = resourcesByComponent[component];
      for (const LifecycleEdge *edgePointer : edgesByComponent[component]) {
        if (!consumeWork(workBudget)) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          result.invalidIndex = loop;
          return result;
        }
        const LifecycleEdge &edge = *edgePointer;
        lifecycle.nodes.push_back(edge.source);
        lifecycle.nodes.push_back(edge.target);
        lifecycle.demands.push_back(edge.demand);
        lifecycle.storageDomains.push_back(edge.domain);
        lifecycle.storageAccesses.push_back(edge.sourceAccess);
        lifecycle.storageAccesses.push_back(edge.targetAccess);
        lifecycle.storageWitnesses.push_back(edge.witness);
        lifecycle.maximumDistance =
            std::max(lifecycle.maximumDistance, edge.distance);
      }
      if (!normalizeIds(lifecycle.nodes) || !normalizeIds(lifecycle.demands) ||
          !normalizeIds(lifecycle.storageDomains) ||
          !normalizeIds(lifecycle.storageAccesses) ||
          !normalizeIds(lifecycle.storageWitnesses)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = loop;
        pendingComponents.clear();
        return result;
      }
      std::vector<ExactAccessKey> componentSlots;
      if (!consumeWork(workBudget, lifecycle.storageAccesses.size())) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = loop;
        pendingComponents.clear();
        return result;
      }
      componentSlots.reserve(lifecycle.storageAccesses.size());
      for (SyncCoverStorageAccessId accessId : lifecycle.storageAccesses) {
        const SyncCoverStorageAccess &access =
            graph.getStorageAccesses()[accessId];
        componentSlots.emplace_back(access.domain, access.extent.begin,
                                    access.extent.end);
      }
      if (!normalizeIds(componentSlots)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = loop;
        pendingComponents.clear();
        return result;
      }
      for (const ExactAccessKey &slot : componentSlots) {
        const std::optional<std::size_t> lookupWork =
            comparisonWork(exactAccessIndex.size(), 2);
        if (!lookupWork || !consumeWork(workBudget, *lookupWork)) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          result.invalidIndex = loop;
          pendingComponents.clear();
          return result;
        }
        const auto range = std::equal_range(
            exactAccessIndex.begin(), exactAccessIndex.end(),
            std::pair<ExactAccessKey, SyncCoverStorageAccessId>{slot, 0},
            [](const auto &left, const auto &right) {
              return left.first < right.first;
            });
        for (auto accessEntry = range.first; accessEntry != range.second;
             ++accessEntry) {
          if (lifecycleAccessIncidences ==
                  limits.maximumLifecycleAccessIncidences ||
              !consumeWork(workBudget)) {
            result.error = workBudget && workBudget->exhausted
                               ? SyncCoverProtocolError::WorkLimitExceeded
                               : SyncCoverProtocolError::LimitExceeded;
            result.invalidIndex = loop;
            pendingComponents.clear();
            return result;
          }
          ++lifecycleAccessIncidences;
          const SyncCoverStorageAccess &access =
              graph.getStorageAccesses()[accessEntry->second];
          const std::optional<bool> contained = scopeContainsWithWork(
              graph, loop, graph.getNodes()[access.node].scope, workBudget);
          if (!contained) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            result.invalidIndex = loop;
            pendingComponents.clear();
            return result;
          }
          if (!*contained) {
            continue;
          }
          lifecycle.storageAccesses.push_back(access.id);
          lifecycle.nodes.push_back(access.node);
        }
      }
      if (!normalizeIds(lifecycle.storageAccesses) ||
          !normalizeIds(lifecycle.nodes)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = loop;
        pendingComponents.clear();
        return result;
      }
      if (lifecycle.resources.empty() ||
          lifecycle.resources.size() > limits.maximumLifecycleLanes ||
          lifecycle.nodes.size() > limits.maximumLifecycleNodeReferences ||
          lifecycle.storageAccesses.size() >
              limits.maximumLifecycleAccessIncidences ||
          lifecycle.storageWitnesses.size() >
              limits.maximumStorageWitnessIncidences) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = loop;
        pendingComponents.clear();
        return result;
      }
      if (lifecycle.maximumDistance == 0 || lifecycle.demands.empty()) {
        continue;
      }
      if (pendingComponents.size() == limits.maximumLifecycleSccs) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = loop;
        pendingComponents.clear();
        return result;
      }
      pendingComponents.push_back(std::move(lifecycle));
    }
  }
  result.components = std::move(pendingComponents);
  result.lifecycleAccessIncidences = lifecycleAccessIncidences;
  return result;
}

namespace {

struct LifecycleSlotKey {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval extent;

  bool operator<(const LifecycleSlotKey &other) const {
    return std::tie(domain, extent.begin, extent.end) <
           std::tie(other.domain, other.extent.begin, other.extent.end);
  }

  bool operator==(const LifecycleSlotKey &other) const {
    return domain == other.domain && extent.begin == other.extent.begin &&
           extent.end == other.extent.end;
  }
};

using LifecycleSlotBundle = std::vector<LifecycleSlotKey>;

struct LifecyclePairKey {
  SyncCoverScopeId loop = 0;
  std::uint32_t producer = 0;
  std::uint32_t consumer = 0;

  bool operator<(const LifecyclePairKey &other) const {
    return std::tie(loop, producer, consumer) <
           std::tie(other.loop, other.producer, other.consumer);
  }
};

struct LifecyclePairFacts {
  std::set<LifecycleSlotKey> slots;
};

struct CrossChildRoundTripKey {
  LifecyclePairKey pair;
  LifecycleSlotKey slot;
  SyncCoverNodeId producer = 0;
  SyncCoverNodeId consumer = 0;

  bool operator<(const CrossChildRoundTripKey &other) const {
    return std::tie(pair.loop, pair.producer, pair.consumer, slot.domain,
                    slot.extent.begin, slot.extent.end, producer, consumer) <
           std::tie(other.pair.loop, other.pair.producer, other.pair.consumer,
                    other.slot.domain, other.slot.extent.begin,
                    other.slot.extent.end, other.producer, other.consumer);
  }
};

struct ExactRoundTripCutKey {
  LifecyclePairKey pair;
  SyncCoverAnchor writeAcquire;
  SyncCoverAnchor ready;
  SyncCoverAnchor readAcquire;
  SyncCoverAnchor release;

  bool operator<(const ExactRoundTripCutKey &other) const {
    return std::tie(pair.loop, pair.producer, pair.consumer, writeAcquire.kind,
                    writeAcquire.node, writeAcquire.scope,
                    writeAcquire.position, ready.kind, ready.node, ready.scope,
                    ready.position, readAcquire.kind, readAcquire.node,
                    readAcquire.scope, readAcquire.position, release.kind,
                    release.node, release.scope, release.position) <
           std::tie(other.pair.loop, other.pair.producer, other.pair.consumer,
                    other.writeAcquire.kind, other.writeAcquire.node,
                    other.writeAcquire.scope, other.writeAcquire.position,
                    other.ready.kind, other.ready.node, other.ready.scope,
                    other.ready.position, other.readAcquire.kind,
                    other.readAcquire.node, other.readAcquire.scope,
                    other.readAcquire.position, other.release.kind,
                    other.release.node, other.release.scope,
                    other.release.position);
  }
};

struct LifecycleNodeFacts {
  SyncCoverNodeId node = 0;
  LifecycleSlotBundle produced;
  LifecycleSlotBundle consumed;
};

bool intervalEqual(const SyncCoverStorageInterval &left,
                   const SyncCoverStorageInterval &right) {
  return left.begin == right.begin && left.end == right.end;
}

bool normalizeBundle(LifecycleSlotBundle &bundle,
                     SyncCoverCoverageWorkBudget *workBudget) {
  const std::optional<std::size_t> sortWork = comparisonWork(bundle.size());
  if (!sortWork || !consumeWork(workBudget, *sortWork)) {
    return false;
  }
  std::sort(bundle.begin(), bundle.end());
  bundle.erase(std::unique(bundle.begin(), bundle.end()), bundle.end());
  return true;
}

std::optional<SyncCoverScopeId>
nearestCommonLoop(const SyncCoverGraph &graph, SyncCoverScopeId first,
                  SyncCoverScopeId second,
                  SyncCoverCoverageWorkBudget *workBudget) {
  const auto &scopes = graph.getScopes();
  if (first >= scopes.size() || second >= scopes.size()) {
    return std::nullopt;
  }
  for (SyncCoverScopeId candidate = first;;
       candidate = scopes[candidate].parent) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    const std::optional<bool> contains =
        scopeContainsWithWork(graph, candidate, second, workBudget);
    if (!contains) {
      return std::nullopt;
    }
    if (scopes[candidate].isLoop && *contains) {
      return candidate;
    }
    if (candidate == 0) {
      break;
    }
  }
  return std::nullopt;
}

std::optional<SyncCoverScopeId>
directChildScope(const SyncCoverGraph &graph, SyncCoverScopeId loop,
                 SyncCoverScopeId descendant,
                 SyncCoverCoverageWorkBudget *workBudget) {
  const auto &scopes = graph.getScopes();
  if (loop >= scopes.size() || descendant >= scopes.size() ||
      loop == descendant) {
    return std::nullopt;
  }
  const std::optional<bool> contained =
      scopeContainsWithWork(graph, loop, descendant, workBudget);
  if (!contained || !*contained) {
    return std::nullopt;
  }
  while (scopes[descendant].parent != loop) {
    if (!consumeWork(workBudget) || descendant == 0) {
      return std::nullopt;
    }
    descendant = scopes[descendant].parent;
  }
  return descendant;
}

struct LoopPhaseControlQuery {
  std::optional<SyncCoverControlId> control;
  bool ambiguous = false;
  bool workUnavailable = false;
};

LoopPhaseControlQuery
findLoopPhaseControl(const SyncCoverGraph &graph, SyncCoverScopeId loop,
                     SyncCoverCoverageWorkBudget *workBudget) {
  LoopPhaseControlQuery result;
  for (const SyncCoverControl &control : graph.getControls()) {
    if (!consumeWork(workBudget)) {
      result.workUnavailable = true;
      return result;
    }
    if (!control.phaseRelation || control.phaseRelation->loopScope != loop) {
      continue;
    }
    if (result.control) {
      result.ambiguous = true;
      result.control.reset();
      return result;
    }
    result.control = control.id;
  }
  return result;
}

bool guardHasLiteral(const SyncCoverGuard &guard, SyncCoverControlId control,
                     unsigned alternative) {
  return std::binary_search(guard.literals.begin(), guard.literals.end(),
                            SyncCoverGuardLiteral{control, alternative});
}

std::vector<unsigned>
reachableAlternatives(const SyncCoverControl &control,
                      SyncCoverCoverageWorkBudget *workBudget) {
  std::vector<unsigned> result;
  if (!control.phaseRelation || control.phaseRelation->nextPhase.empty()) {
    return result;
  }
  const SyncCoverControlPhaseRelation &relation = *control.phaseRelation;
  std::vector<bool> visited(relation.nextPhase.size());
  std::size_t phase = relation.initialPhase;
  while (phase < relation.nextPhase.size() && !visited[phase]) {
    if (!consumeWork(workBudget)) {
      return {};
    }
    visited[phase] = true;
    result.push_back(relation.activeAlternative[phase]);
    phase = relation.nextPhase[phase];
  }
  const std::optional<std::size_t> sortWork = comparisonWork(result.size());
  if (!sortWork || !consumeWork(workBudget, *sortWork)) {
    return {};
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

std::optional<unsigned> phaseDistanceBetweenPaths(
    const SyncCoverGraph &graph, SyncCoverControlId controlId,
    SyncCoverScopeId sourcePath, SyncCoverScopeId targetPath,
    SyncCoverCoverageWorkBudget *workBudget) {
  if (controlId >= graph.getControls().size() ||
      sourcePath >= graph.getScopes().size() ||
      targetPath >= graph.getScopes().size()) {
    return std::nullopt;
  }
  const std::optional<SyncCoverControlPhaseRelation> &relation =
      graph.getControls()[controlId].phaseRelation;
  if (!relation || relation->nextPhase.empty() ||
      relation->nextPhase.size() != relation->activeAlternative.size()) {
    return std::nullopt;
  }
  const auto alternativeFor = [&](SyncCoverScopeId path) {
    std::optional<unsigned> alternative;
    for (const SyncCoverGuardLiteral &literal :
         graph.getScopes()[path].guard.literals) {
      if (!consumeWork(workBudget)) {
        return std::optional<unsigned>{};
      }
      if (literal.control == controlId) {
        if (alternative) {
          return std::optional<unsigned>{};
        }
        alternative = literal.alternative;
      }
    }
    return alternative;
  };
  const std::optional<unsigned> sourceAlternative = alternativeFor(sourcePath);
  const std::optional<unsigned> targetAlternative = alternativeFor(targetPath);
  if (!sourceAlternative || !targetAlternative) {
    return std::nullopt;
  }
  for (std::size_t sourcePhase = 0;
       sourcePhase < relation->activeAlternative.size(); ++sourcePhase) {
    if (relation->activeAlternative[sourcePhase] != *sourceAlternative) {
      continue;
    }
    std::size_t phase = sourcePhase;
    for (unsigned distance = 1; distance <= relation->nextPhase.size();
         ++distance) {
      if (!consumeWork(workBudget) || phase >= relation->nextPhase.size()) {
        return std::nullopt;
      }
      phase = relation->nextPhase[phase];
      if (phase >= relation->activeAlternative.size()) {
        return std::nullopt;
      }
      if (relation->activeAlternative[phase] == *targetAlternative) {
        return distance;
      }
    }
  }
  return std::nullopt;
}

std::optional<SyncCoverRegionId>
lowestCommonRegion(const SyncCoverGraph &graph,
                   const std::vector<SyncCoverNodeId> &nodes,
                   SyncCoverCoverageWorkBudget *workBudget) {
  if (nodes.empty()) {
    return std::nullopt;
  }
  const auto &regions = graph.getRegions();
  std::vector<SyncCoverRegionId> ancestors;
  for (SyncCoverRegionId region = graph.getNodes()[nodes.front()].region;;
       region = regions[region].parent) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    ancestors.push_back(region);
    if (region == 0) {
      break;
    }
  }
  for (SyncCoverRegionId candidate : ancestors) {
    bool containsAll = true;
    for (SyncCoverNodeId node : nodes) {
      SyncCoverRegionId region = graph.getNodes()[node].region;
      while (region != candidate && region != 0) {
        if (!consumeWork(workBudget)) {
          return std::nullopt;
        }
        region = regions[region].parent;
      }
      if (region != candidate) {
        containsAll = false;
        break;
      }
    }
    if (containsAll) {
      return candidate;
    }
  }
  return std::nullopt;
}

std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>> groupAnchors(
    const SyncCoverGraph &graph, const std::vector<SyncCoverNodeId> &nodes,
    SyncCoverScopeId lifecycleLoop, SyncCoverCoverageWorkBudget *workBudget,
    bool preservePeriodicAlternatives = false) {
  const std::optional<SyncCoverRegionId> common =
      lowestCommonRegion(graph, nodes, workBudget);
  if (!common) {
    return std::nullopt;
  }
  const SyncCoverRegion &region = graph.getRegions()[*common];
  const SyncCoverRegionId lifecycleRegion =
      lifecycleLoop < graph.getScopes().size()
          ? graph.getScopes()[lifecycleLoop].region
          : 0;
  const auto isLifecycleControl = [&](SyncCoverControlId control) {
    if (!preservePeriodicAlternatives ||
        control >= graph.getControls().size()) {
      return false;
    }
    const SyncCoverControl &candidate = graph.getControls()[control];
    return (candidate.phaseRelation &&
            candidate.phaseRelation->loopScope == lifecycleLoop) ||
           (candidate.successorRelation &&
            candidate.successorRelation->loopScope == lifecycleLoop);
  };
  SyncCoverScopeId commonScope = graph.getNodes()[nodes.front()].scope;
  for (SyncCoverNodeId node : nodes) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    const std::optional<SyncCoverScopeId> lowest =
        graph.getLowestCommonScope(commonScope, graph.getNodes()[node].scope);
    if (!lowest) {
      return std::nullopt;
    }
    commonScope = *lowest;
  }
  std::optional<SyncCoverScopeId> nestedLoop =
      graph.getNearestEnclosingLoop(commonScope);
  while (nestedLoop && *nestedLoop != lifecycleLoop) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    const SyncCoverScopeId parent = graph.getScopes()[*nestedLoop].parent;
    const std::optional<SyncCoverScopeId> outerLoop =
        graph.getNearestEnclosingLoop(parent);
    if (!outerLoop || *outerLoop == lifecycleLoop) {
      break;
    }
    if (*outerLoop == *nestedLoop) {
      return std::nullopt;
    }
    nestedLoop = outerLoop;
  }
  std::optional<bool> containsNested = false;
  if (nestedLoop && *nestedLoop != lifecycleLoop) {
    containsNested =
        scopeContainsWithWork(graph, lifecycleLoop, *nestedLoop, workBudget);
    if (!containsNested) {
      return std::nullopt;
    }
  }
  if (containsNested.value_or(false)) {
    // A nested loop boundary is an exact scalar cut even for a zero-trip
    // loop, but not when the entire loop belongs to one optional branch.
    // Lift such a boundary to the enclosing choice so the protocol action is
    // issued once on every lifecycle-loop iteration, independently of which
    // alternative executes.
    const SyncCoverRegionId nestedRegion =
        graph.getScopes()[*nestedLoop].region;
    for (SyncCoverRegionId ancestor = nestedRegion;
         ancestor != 0 && ancestor != lifecycleRegion;
         ancestor = graph.getRegions()[ancestor].parent) {
      if (!consumeWork(workBudget)) {
        return std::nullopt;
      }
      const SyncCoverRegion &candidate = graph.getRegions()[ancestor];
      if (candidate.kind != SyncCoverRegionKind::Alternative ||
          candidate.parent >= graph.getRegions().size()) {
        continue;
      }
      const SyncCoverRegion &choice = graph.getRegions()[candidate.parent];
      if (choice.kind == SyncCoverRegionKind::Choice && choice.control) {
        if (isLifecycleControl(*choice.control)) {
          continue;
        }
        return std::make_pair(SyncCoverAnchor{SyncCoverAnchorKind::ControlEntry,
                                              *choice.control, choice.scope, 0},
                              SyncCoverAnchor{SyncCoverAnchorKind::ControlExit,
                                              *choice.control, choice.scope,
                                              0});
      }
    }
    return std::make_pair(
        SyncCoverAnchor{SyncCoverAnchorKind::ScopeEntry, 0, *nestedLoop, 0},
        SyncCoverAnchor{SyncCoverAnchorKind::ScopeExit, 0, *nestedLoop, 0});
  }
  for (SyncCoverRegionId ancestor = *common;
       ancestor != 0 && ancestor != lifecycleRegion;
       ancestor = graph.getRegions()[ancestor].parent) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    const SyncCoverRegion &candidate = graph.getRegions()[ancestor];
    if (candidate.kind != SyncCoverRegionKind::Alternative ||
        candidate.parent >= graph.getRegions().size()) {
      continue;
    }
    const SyncCoverRegion &choice = graph.getRegions()[candidate.parent];
    if (choice.kind == SyncCoverRegionKind::Choice && choice.control) {
      if (isLifecycleControl(*choice.control)) {
        continue;
      }
      return std::make_pair(SyncCoverAnchor{SyncCoverAnchorKind::ControlEntry,
                                            *choice.control, choice.scope, 0},
                            SyncCoverAnchor{SyncCoverAnchorKind::ControlExit,
                                            *choice.control, choice.scope, 0});
    }
  }
  if (region.kind == SyncCoverRegionKind::Loop && region.scope != 0 &&
      region.scope != lifecycleLoop) {
    return std::make_pair(
        SyncCoverAnchor{SyncCoverAnchorKind::ScopeEntry, 0, region.scope, 0},
        SyncCoverAnchor{SyncCoverAnchorKind::ScopeExit, 0, region.scope, 0});
  }
  if (nodes.size() > 1 && region.kind == SyncCoverRegionKind::Sequence &&
      region.parent < graph.getRegions().size()) {
    const SyncCoverRegion &parent = graph.getRegions()[region.parent];
    if (parent.kind == SyncCoverRegionKind::Loop && parent.scope != 0 &&
        parent.scope != lifecycleLoop) {
      return std::make_pair(
          SyncCoverAnchor{SyncCoverAnchorKind::ScopeEntry, 0, parent.scope, 0},
          SyncCoverAnchor{SyncCoverAnchorKind::ScopeExit, 0, parent.scope, 0});
    }
  }
  if (region.kind == SyncCoverRegionKind::Alternative &&
      region.parent < graph.getRegions().size()) {
    const SyncCoverRegion &choice = graph.getRegions()[region.parent];
    if (choice.kind == SyncCoverRegionKind::Choice && choice.control) {
      if (isLifecycleControl(*choice.control)) {
        // Periodic alternatives are part of the lifecycle automaton.  Keep
        // both endpoints inside the same alternative so the phase guard can
        // balance the transaction; lifting either endpoint to the enclosing
        // choice would move a ready Set after its guarded Wait.
      } else {
        // A demand endpoint in one optional branch cannot own a one-sided event
        // action. Lift the acquire/release cuts to the opaque choice boundary
        // so both actions execute once regardless of the selected alternative.
        return std::make_pair(SyncCoverAnchor{SyncCoverAnchorKind::ControlEntry,
                                              *choice.control, choice.scope, 0},
                              SyncCoverAnchor{SyncCoverAnchorKind::ControlExit,
                                              *choice.control, choice.scope,
                                              0});
      }
    }
  }
  if (region.kind == SyncCoverRegionKind::Choice && region.control) {
    if (!isLifecycleControl(*region.control)) {
      return std::make_pair(SyncCoverAnchor{SyncCoverAnchorKind::ControlEntry,
                                            *region.control, region.scope, 0},
                            SyncCoverAnchor{SyncCoverAnchorKind::ControlExit,
                                            *region.control, region.scope, 0});
    }
  }
  const auto byOrder = [&](SyncCoverNodeId left, SyncCoverNodeId right) {
    return graph.getNodes()[left].order < graph.getNodes()[right].order;
  };
  const auto first = std::min_element(nodes.begin(), nodes.end(), byOrder);
  const auto last = std::max_element(nodes.begin(), nodes.end(), byOrder);
  const auto liftGuardedBoundary = [&](SyncCoverNodeId node,
                                       bool entry) -> SyncCoverAnchor {
    SyncCoverAnchor best{entry ? SyncCoverAnchorKind::BeforeNode
                               : SyncCoverAnchorKind::AfterNode,
                         node, 0, 0};
    std::optional<SyncCoverTimelinePosition> bestPosition =
        resolveSyncCoverAnchor(graph, best);
    for (const SyncCoverGuardLiteral &literal :
         graph.getNodes()[node].guard.literals) {
      if (!consumeWork(workBudget) ||
          literal.control >= graph.getControls().size()) {
        return best;
      }
      const SyncCoverControl &control = graph.getControls()[literal.control];
      const bool loopPhase = control.phaseRelation &&
                             control.phaseRelation->loopScope == lifecycleLoop;
      const bool successor =
          control.successorRelation &&
          control.successorRelation->loopScope == lifecycleLoop;
      if (loopPhase || successor) {
        continue;
      }
      SyncCoverAnchor candidate{entry ? SyncCoverAnchorKind::ControlEntry
                                      : SyncCoverAnchorKind::ControlExit,
                                literal.control, control.scope, 0};
      const std::optional<SyncCoverTimelinePosition> position =
          resolveSyncCoverAnchor(graph, candidate);
      if (!position || (bestPosition && (entry ? *position > *bestPosition
                                               : *position < *bestPosition))) {
        continue;
      }
      best = candidate;
      bestPosition = position;
    }
    return best;
  };
  return std::make_pair(liftGuardedBoundary(*first, true),
                        liftGuardedBoundary(*last, false));
}

std::vector<LifecycleNodeFacts> collectLifecycleNodes(
    const SyncCoverGraph &graph, SyncCoverScopeId scope, std::uint32_t producer,
    std::uint32_t consumer, const std::set<LifecycleSlotKey> &slots,
    SyncCoverProtocolLimits limits, std::size_t &accessIncidences,
    SyncCoverCoverageWorkBudget *workBudget, SyncCoverProtocolError &error) {
  std::vector<std::vector<const SyncCoverStorageAccess *>> accessesByNode(
      graph.getNodes().size());
  if (!consumeWork(workBudget, graph.getNodes().size())) {
    error = SyncCoverProtocolError::WorkLimitExceeded;
    return {};
  }
  for (const SyncCoverStorageAccess &access : graph.getStorageAccesses()) {
    if (!consumeWork(workBudget)) {
      error = SyncCoverProtocolError::WorkLimitExceeded;
      return {};
    }
    ++accessIncidences;
    if (accessIncidences > limits.maximumLifecycleAccessIncidences) {
      error = SyncCoverProtocolError::LimitExceeded;
      return {};
    }
    accessesByNode[access.node].push_back(&access);
  }
  std::vector<LifecycleNodeFacts> result;
  for (const SyncCoverNode &node : graph.getNodes()) {
    const std::optional<bool> contained =
        scopeContainsWithWork(graph, scope, node.scope, workBudget);
    if (!contained) {
      error = SyncCoverProtocolError::WorkLimitExceeded;
      return {};
    }
    if (!*contained) {
      continue;
    }
    LifecycleNodeFacts facts;
    facts.node = node.id;
    for (const SyncCoverStorageAccess *access : accessesByNode[node.id]) {
      if (!consumeWork(workBudget)) {
        error = SyncCoverProtocolError::WorkLimitExceeded;
        return {};
      }
      const LifecycleSlotKey slot{access->domain, access->extent};
      if (!access->exactPhysical) {
        continue;
      }
      if (!consumeWork(workBudget, lookupWork(slots.size() + 1))) {
        error = SyncCoverProtocolError::WorkLimitExceeded;
        return {};
      }
      if (slots.count(slot) == 0) {
        continue;
      }
      if (node.resource == producer &&
          syncCoverStorageModeWrites(access->mode)) {
        facts.produced.push_back(slot);
      }
      if (node.resource == consumer &&
          syncCoverStorageModeReads(access->mode)) {
        facts.consumed.push_back(slot);
      }
    }
    if (!normalizeBundle(facts.produced, workBudget) ||
        !normalizeBundle(facts.consumed, workBudget)) {
      error = SyncCoverProtocolError::WorkLimitExceeded;
      return {};
    }
    if (!facts.produced.empty() || !facts.consumed.empty()) {
      result.push_back(std::move(facts));
    }
  }
  const std::optional<std::size_t> resultSortWork =
      comparisonWork(result.size());
  if (!resultSortWork || !consumeWork(workBudget, *resultSortWork)) {
    error = SyncCoverProtocolError::WorkLimitExceeded;
    return {};
  }
  std::sort(
      result.begin(), result.end(),
      [&](const LifecycleNodeFacts &left, const LifecycleNodeFacts &right) {
        return std::tie(graph.getNodes()[left.node].order, left.node) <
               std::tie(graph.getNodes()[right.node].order, right.node);
      });
  return result;
}

std::optional<SyncCoverScopeId>
pathScopeForAlternative(const SyncCoverGraph &graph, SyncCoverScopeId loop,
                        SyncCoverControlId control, unsigned alternative,
                        const std::vector<LifecycleNodeFacts> &nodes,
                        SyncCoverCoverageWorkBudget *workBudget) {
  std::optional<SyncCoverScopeId> result;
  for (const LifecycleNodeFacts &facts : nodes) {
    const SyncCoverNode &node = graph.getNodes()[facts.node];
    if (!consumeWork(workBudget,
                     1 + lookupWork(node.guard.literals.size() + 1))) {
      return std::nullopt;
    }
    if (!guardHasLiteral(node.guard, control, alternative)) {
      continue;
    }
    const std::optional<SyncCoverScopeId> child =
        directChildScope(graph, loop, node.scope, workBudget);
    if (!child || (result && *result != *child)) {
      return std::nullopt;
    }
    result = child;
  }
  return result;
}

struct SlotPathPresence {
  bool produced = false;
  bool consumed = false;
  std::size_t firstProducer = std::numeric_limits<std::size_t>::max();
  std::size_t firstConsumer = std::numeric_limits<std::size_t>::max();
};

std::optional<SlotPathPresence>
slotPresence(const SyncCoverGraph &graph, SyncCoverScopeId path,
             const LifecycleSlotKey &slot,
             const std::vector<LifecycleNodeFacts> &nodes,
             SyncCoverCoverageWorkBudget *workBudget) {
  SlotPathPresence result;
  for (const LifecycleNodeFacts &facts : nodes) {
    const std::optional<bool> contained = scopeContainsWithWork(
        graph, path, graph.getNodes()[facts.node].scope, workBudget);
    if (!contained) {
      return std::nullopt;
    }
    if (!*contained) {
      continue;
    }
    if (!consumeWork(workBudget, lookupWork(facts.produced.size() + 1) +
                                     lookupWork(facts.consumed.size() + 1))) {
      return std::nullopt;
    }
    if (std::binary_search(facts.produced.begin(), facts.produced.end(),
                           slot)) {
      result.produced = true;
      result.firstProducer =
          std::min(result.firstProducer, graph.getNodes()[facts.node].order);
    }
    if (std::binary_search(facts.consumed.begin(), facts.consumed.end(),
                           slot)) {
      result.consumed = true;
      result.firstConsumer =
          std::min(result.firstConsumer, graph.getNodes()[facts.node].order);
    }
  }
  return result;
}

bool populateLifecycleSlotAccesses(const SyncCoverGraph &graph,
                                   const SyncCoverLifecycleScc &component,
                                   DerivedLifecycleCertificate &certificate,
                                   SyncCoverCoverageWorkBudget *workBudget) {
  for (DerivedLifecycleLane &lane : certificate.lanes) {
    for (DerivedLifecycleSlot &slot : lane.slots) {
      for (const SyncCoverStorageAccess &access : graph.getStorageAccesses()) {
        if (!consumeWork(workBudget,
                         1 + lookupWork(component.nodes.size() + 1) +
                             certificate.initialProducers.size())) {
          return false;
        }
        const bool componentNode = std::binary_search(
            component.nodes.begin(), component.nodes.end(), access.node);
        const bool initialProducer =
            std::find(certificate.initialProducers.begin(),
                      certificate.initialProducers.end(),
                      access.node) != certificate.initialProducers.end();
        if (!componentNode && !initialProducer) {
          continue;
        }
        const std::optional<bool> contained = scopeContainsWithWork(
            graph, certificate.loopScope, graph.getNodes()[access.node].scope,
            workBudget);
        if (!contained) {
          return false;
        }
        const bool relevant = *contained || initialProducer;
        if (relevant && access.exactPhysical && access.domain == slot.domain &&
            intervalEqual(access.extent, slot.extent)) {
          slot.accesses.push_back(access.id);
        }
      }
      if (slot.accesses.empty()) {
        return false;
      }
    }
  }
  return true;
}

bool hasMatchingReleaseDemands(const SyncCoverGraph &graph,
                               const SyncCoverLifecycleScc &component,
                               const DerivedLifecycleCertificate &certificate,
                               SyncCoverCoverageWorkBudget *workBudget) {
  if (certificate.lanes.empty()) {
    return false;
  }
  for (const DerivedLifecycleLane &lane : certificate.lanes) {
    std::vector<SyncCoverNodeId> sourceConsumers;
    std::vector<SyncCoverNodeId> targetProducers;
    std::optional<unsigned> reuseDistance;
    const DerivedLifecyclePath *sourcePath = nullptr;
    const DerivedLifecyclePath *targetPath = nullptr;
    for (const DerivedLifecyclePath &path : certificate.paths) {
      for (const DerivedLifecycleUse &use : path.uses) {
        if (!consumeWork(workBudget,
                         1 + use.consumers.size() + use.producers.size())) {
          return false;
        }
        if (use.lane == lane.id) {
          if (certificate.alternating) {
            if (sourcePath) {
              return false;
            }
            sourcePath = &path;
          }
          sourceConsumers.insert(sourceConsumers.end(), use.consumers.begin(),
                                 use.consumers.end());
        }
        if (use.producerLane == lane.id) {
          if (certificate.alternating) {
            if (targetPath) {
              return false;
            }
            targetPath = &path;
          }
          targetProducers.insert(targetProducers.end(), use.producers.begin(),
                                 use.producers.end());
        }
      }
    }
    const std::optional<std::size_t> sourceSortWork =
        comparisonWork(sourceConsumers.size());
    const std::optional<std::size_t> targetSortWork =
        comparisonWork(targetProducers.size());
    if (!sourceSortWork || !targetSortWork ||
        !consumeWork(workBudget, *sourceSortWork + *targetSortWork)) {
      return false;
    }
    std::sort(sourceConsumers.begin(), sourceConsumers.end());
    sourceConsumers.erase(
        std::unique(sourceConsumers.begin(), sourceConsumers.end()),
        sourceConsumers.end());
    std::sort(targetProducers.begin(), targetProducers.end());
    targetProducers.erase(
        std::unique(targetProducers.begin(), targetProducers.end()),
        targetProducers.end());
    if (certificate.alternating) {
      if (!certificate.periodicControl || !sourcePath || !targetPath) {
        return false;
      }
      reuseDistance = phaseDistanceBetweenPaths(
          graph, *certificate.periodicControl, sourcePath->scope,
          targetPath->scope, workBudget);
    } else {
      reuseDistance = 1;
    }
    if (!reuseDistance || *reuseDistance == 0 || sourceConsumers.empty() ||
        targetProducers.empty()) {
      return false;
    }
    for (const DerivedLifecycleSlot &slot : lane.slots) {
      bool found = false;
      for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
           ++demandId) {
        if (!consumeWork(workBudget,
                         1 + lookupWork(component.demands.size() + 1) +
                             lookupWork(sourceConsumers.size() + 1) +
                             lookupWork(targetProducers.size() + 1))) {
          return false;
        }
        if (!std::binary_search(component.demands.begin(),
                                component.demands.end(), demandId)) {
          continue;
        }
        const SyncCoverDemand &demand = graph.getDemands()[demandId];
        if (demand.scope != certificate.loopScope ||
            demand.distance != *reuseDistance ||
            !std::binary_search(sourceConsumers.begin(), sourceConsumers.end(),
                                demand.source) ||
            !std::binary_search(targetProducers.begin(), targetProducers.end(),
                                demand.target)) {
          continue;
        }
        for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
          if (!consumeWork(workBudget)) {
            return false;
          }
          const SyncCoverStorageWitness &witness =
              graph.getStorageWitnesses()[witnessId];
          const SyncCoverStorageAccess &source =
              graph.getStorageAccesses()[witness.sourceAccess];
          const SyncCoverStorageAccess &target =
              graph.getStorageAccesses()[witness.targetAccess];
          const bool matches = source.exactPhysical && target.exactPhysical &&
                               source.domain == slot.domain &&
                               target.domain == slot.domain &&
                               intervalEqual(source.extent, slot.extent) &&
                               intervalEqual(target.extent, slot.extent) &&
                               syncCoverStorageModeReads(source.mode) &&
                               syncCoverStorageModeWrites(target.mode) &&
                               graph.getNodes()[source.node].resource ==
                                   certificate.consumerResource &&
                               graph.getNodes()[target.node].resource ==
                                   certificate.producerResource;
          if (matches) {
            found = true;
            break;
          }
        }
        if (found) {
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }
  return true;
}

std::optional<DerivedLifecycleCertificate>
buildCrossChildRoundTripCertificate(const SyncCoverGraph &graph,
                                    const SyncCoverLifecycleScc &component,
                                    const CrossChildRoundTripKey &key,
                                    SyncCoverCoverageWorkBudget *workBudget) {
  if (key.pair.loop == 0 || key.pair.loop >= graph.getScopes().size() ||
      key.producer >= graph.getNodes().size() ||
      key.consumer >= graph.getNodes().size()) {
    return std::nullopt;
  }
  const std::vector<SyncCoverNodeId> producers{key.producer};
  const std::vector<SyncCoverNodeId> consumers{key.consumer};
  const SyncCoverNode &producer = graph.getNodes()[key.producer];
  const SyncCoverNode &consumer = graph.getNodes()[key.consumer];
  const std::optional<SyncCoverScopeId> producerLoop =
      graph.getNearestEnclosingLoop(producer.scope, true);
  const std::optional<SyncCoverScopeId> consumerLoop =
      graph.getNearestEnclosingLoop(consumer.scope, true);
  const bool oneInvocationGuardedTransfer =
      producerLoop && consumerLoop && *producerLoop == key.pair.loop &&
      *consumerLoop == key.pair.loop &&
      producer.guard.literals == consumer.guard.literals;
  const std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>>
      producerAnchors =
          oneInvocationGuardedTransfer
              ? std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>>(
                    std::make_pair(
                        SyncCoverAnchor{SyncCoverAnchorKind::BeforeNode,
                                        key.producer, 0, 0},
                        SyncCoverAnchor{SyncCoverAnchorKind::AfterNode,
                                        key.producer, 0, 0}))
              : groupAnchors(graph, producers, key.pair.loop, workBudget);
  const std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>>
      consumerAnchors =
          oneInvocationGuardedTransfer
              ? std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>>(
                    std::make_pair(
                        SyncCoverAnchor{SyncCoverAnchorKind::BeforeNode,
                                        key.consumer, 0, 0},
                        SyncCoverAnchor{SyncCoverAnchorKind::AfterNode,
                                        key.consumer, 0, 0}))
              : groupAnchors(graph, consumers, key.pair.loop, workBudget);
  if (!producerAnchors || !consumerAnchors) {
    return std::nullopt;
  }
  const std::optional<SyncCoverTimelinePosition> readyPosition =
      resolveSyncCoverAnchor(graph, producerAnchors->second);
  const std::optional<SyncCoverTimelinePosition> acquirePosition =
      resolveSyncCoverAnchor(graph, consumerAnchors->first);
  if (!readyPosition || !acquirePosition ||
      *readyPosition >= *acquirePosition) {
    return std::nullopt;
  }

  DerivedLifecycleCertificate certificate;
  certificate.loopScope = key.pair.loop;
  certificate.producerResource = key.pair.producer;
  certificate.consumerResource = key.pair.consumer;
  certificate.lanes.push_back({0, {{key.slot.domain, key.slot.extent, {}}}});
  DerivedLifecyclePath path;
  path.scope = key.pair.loop;
  path.uses.push_back({0, 0, producers, consumers, producerAnchors->first,
                       producerAnchors->second, consumerAnchors->first,
                       consumerAnchors->second});
  certificate.paths.push_back(std::move(path));
  return populateLifecycleSlotAccesses(graph, component, certificate,
                                       workBudget)
             ? std::optional<DerivedLifecycleCertificate>(
                   std::move(certificate))
             : std::nullopt;
}

std::optional<DerivedLifecycleCertificate>
buildBalancedDirectCertificate(const SyncCoverGraph &graph,
                               SyncCoverDemandId demandId,
                               SyncCoverCoverageWorkBudget *workBudget) {
  if (demandId >= graph.getDemands().size()) {
    return std::nullopt;
  }
  const SyncCoverDemand &demand = graph.getDemands()[demandId];
  if (demand.source >= graph.getNodes().size() ||
      demand.target >= graph.getNodes().size()) {
    return std::nullopt;
  }
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  if (source.resource == target.resource) {
    return std::nullopt;
  }
  std::optional<SyncCoverScopeId> loop =
      demand.distance != 0 && demand.scope < graph.getScopes().size() &&
              graph.getScopes()[demand.scope].isLoop
          ? std::optional<SyncCoverScopeId>(demand.scope)
          : nearestSharedLifecycleLoop(graph, source.scope, target.scope,
                                       workBudget);
  if (!loop) {
    return std::nullopt;
  }
  const std::optional<SyncCoverScopeId> sourceLoop =
      graph.getNearestEnclosingLoop(source.scope, true);
  const std::optional<SyncCoverScopeId> targetLoop =
      graph.getNearestEnclosingLoop(target.scope, true);
  if (demand.distance == 0 && source.guard.literals != target.guard.literals) {
    if (syncCoverGuardImplies(target.guard, source.guard) && targetLoop) {
      loop = targetLoop;
    } else if (syncCoverGuardImplies(source.guard, target.guard) &&
               sourceLoop) {
      loop = sourceLoop;
    }
  }
  if (!loop) {
    return std::nullopt;
  }
  const bool oneInvocationGuardedTransfer =
      sourceLoop && targetLoop && *sourceLoop == *loop &&
      *targetLoop == *loop && source.guard.literals == target.guard.literals;
  std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>> sourceAnchors =
      oneInvocationGuardedTransfer
          ? std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>>(
                std::make_pair(SyncCoverAnchor{SyncCoverAnchorKind::BeforeNode,
                                               source.id, 0, 0},
                               SyncCoverAnchor{SyncCoverAnchorKind::AfterNode,
                                               source.id, 0, 0}))
          : groupAnchors(graph, {source.id}, *loop, workBudget);
  std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>> targetAnchors =
      oneInvocationGuardedTransfer
          ? std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>>(
                std::make_pair(SyncCoverAnchor{SyncCoverAnchorKind::BeforeNode,
                                               target.id, 0, 0},
                               SyncCoverAnchor{SyncCoverAnchorKind::AfterNode,
                                               target.id, 0, 0}))
          : groupAnchors(graph, {target.id}, *loop, workBudget);
  if (!sourceAnchors || !targetAnchors) {
    return std::nullopt;
  }

  if (source.guard.literals != target.guard.literals) {
    const SyncCoverAnchor beforeSource{SyncCoverAnchorKind::BeforeNode,
                                       source.id, 0, 0};
    const SyncCoverAnchor afterSource{SyncCoverAnchorKind::AfterNode, source.id,
                                      0, 0};
    const SyncCoverAnchor beforeTarget{SyncCoverAnchorKind::BeforeNode,
                                       target.id, 0, 0};
    const SyncCoverAnchor afterTarget{SyncCoverAnchorKind::AfterNode, target.id,
                                      0, 0};
    if (syncCoverGuardImplies(target.guard, source.guard)) {
      // The target is the narrower occurrence. Issue the source-prefix Set
      // only when that target executes, and circulate the release token
      // between consecutive executions of the same guarded transaction.
      sourceAnchors = std::make_pair(beforeTarget, beforeTarget);
      targetAnchors = std::make_pair(beforeTarget, afterTarget);
    } else if (syncCoverGuardImplies(source.guard, target.guard)) {
      // The source is the narrower occurrence. Consume the ready token on the
      // target pipe immediately after the guarded source; physical target-pipe
      // issue order then carries that completion fact to the later target.
      sourceAnchors = std::make_pair(beforeSource, afterSource);
      targetAnchors = std::make_pair(afterSource, afterSource);
    } else {
      return std::nullopt;
    }
  }
  const std::optional<SyncCoverTimelinePosition> setPosition =
      resolveSyncCoverAnchor(graph, sourceAnchors->second);
  const std::optional<SyncCoverTimelinePosition> waitPosition =
      resolveSyncCoverAnchor(graph, targetAnchors->first);
  if (!setPosition || !waitPosition || *setPosition > *waitPosition) {
    return std::nullopt;
  }

  DerivedLifecycleCertificate certificate;
  certificate.loopScope = *loop;
  certificate.producerResource = source.resource;
  certificate.consumerResource = target.resource;
  certificate.lanes.push_back({0, {}});
  DerivedLifecyclePath path;
  path.scope = *loop;
  path.uses.push_back({0,
                       0,
                       {source.id},
                       {target.id},
                       sourceAnchors->first,
                       sourceAnchors->second,
                       targetAnchors->first,
                       targetAnchors->second});
  certificate.paths.push_back(std::move(path));
  return certificate;
}

SyncCoverProtocolError
checkLifecycleCertificateBounds(const DerivedLifecycleCertificate &certificate,
                                SyncCoverProtocolLimits limits,
                                SyncCoverCoverageWorkBudget *workBudget) {
  if (certificate.lanes.size() > limits.maximumLifecycleLanes ||
      certificate.paths.size() > limits.maximumLifecyclePaths ||
      certificate.initiallyFreeLanes.size() > limits.maximumLifecycleLanes ||
      certificate.initialProducers.size() >
          limits.maximumLifecycleNodeReferences) {
    return SyncCoverProtocolError::LimitExceeded;
  }
  std::size_t slots = 0;
  std::size_t accessIncidences = 0;
  std::size_t transfers = 0;
  std::size_t nodeReferences = certificate.initialProducers.size();
  for (const DerivedLifecycleLane &lane : certificate.lanes) {
    if (!consumeWork(workBudget)) {
      return SyncCoverProtocolError::WorkLimitExceeded;
    }
    if (!checkedAccumulateSize(slots, lane.slots.size()) ||
        slots > limits.maximumLifecycleSlots) {
      return SyncCoverProtocolError::LimitExceeded;
    }
    for (const DerivedLifecycleSlot &slot : lane.slots) {
      if (!consumeWork(workBudget)) {
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
      if (!checkedAccumulateSize(accessIncidences, slot.accesses.size()) ||
          accessIncidences > limits.maximumLifecycleAccessIncidences) {
        return SyncCoverProtocolError::LimitExceeded;
      }
    }
  }
  for (const DerivedLifecyclePath &path : certificate.paths) {
    if (!consumeWork(workBudget)) {
      return SyncCoverProtocolError::WorkLimitExceeded;
    }
    if (!checkedAccumulateSize(transfers, path.uses.size()) ||
        transfers > limits.maximumLifecycleTransfers / 2) {
      return SyncCoverProtocolError::LimitExceeded;
    }
    for (const DerivedLifecycleUse &use : path.uses) {
      if (!consumeWork(workBudget)) {
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
      std::size_t useReferences = 0;
      if (!checkedAccumulateSize(useReferences, use.producers.size()) ||
          !checkedAccumulateSize(useReferences, use.consumers.size()) ||
          !checkedAccumulateSize(nodeReferences, useReferences) ||
          nodeReferences > limits.maximumLifecycleNodeReferences) {
        return SyncCoverProtocolError::LimitExceeded;
      }
    }
  }
  return SyncCoverProtocolError::None;
}

std::optional<DerivedLifecycleCertificate> buildStableLifecycleCertificate(
    const SyncCoverGraph &graph, const SyncCoverLifecycleScc &component,
    const LifecyclePairKey &key, const std::vector<SyncCoverScopeId> &paths,
    const std::vector<LifecycleNodeFacts> &nodes,
    const std::set<LifecycleSlotKey> &stableSlots,
    SyncCoverCoverageWorkBudget *workBudget) {
  if (stableSlots.empty() || paths.empty()) {
    return std::nullopt;
  }
  DerivedLifecycleCertificate certificate;
  certificate.loopScope = key.loop;
  certificate.producerResource = key.producer;
  certificate.consumerResource = key.consumer;
  std::map<LifecycleSlotBundle, std::size_t> laneByBundle;
  for (SyncCoverScopeId pathScope : paths) {
    DerivedLifecyclePath path;
    path.scope = pathScope;
    std::vector<SyncCoverNodeId> producers;
    std::vector<SyncCoverNodeId> consumers;
    const auto finalize = [&]() -> bool {
      if (producers.empty() || consumers.empty()) {
        return false;
      }
      LifecycleSlotBundle produced;
      LifecycleSlotBundle consumed;
      for (SyncCoverNodeId node : producers) {
        if (!consumeWork(workBudget, nodes.size())) {
          return false;
        }
        const auto found = std::find_if(nodes.begin(), nodes.end(),
                                        [&](const LifecycleNodeFacts &facts) {
                                          return facts.node == node;
                                        });
        if (found != nodes.end()) {
          for (const LifecycleSlotKey &slot : found->produced) {
            if (!consumeWork(workBudget, lookupWork(stableSlots.size() + 1))) {
              return false;
            }
            if (stableSlots.count(slot) != 0) {
              produced.push_back(slot);
            }
          }
        }
      }
      for (SyncCoverNodeId node : consumers) {
        if (!consumeWork(workBudget, nodes.size())) {
          return false;
        }
        const auto found = std::find_if(nodes.begin(), nodes.end(),
                                        [&](const LifecycleNodeFacts &facts) {
                                          return facts.node == node;
                                        });
        if (found != nodes.end()) {
          for (const LifecycleSlotKey &slot : found->consumed) {
            if (!consumeWork(workBudget, lookupWork(stableSlots.size() + 1))) {
              return false;
            }
            if (stableSlots.count(slot) != 0) {
              consumed.push_back(slot);
            }
          }
        }
      }
      if (!normalizeBundle(produced, workBudget) ||
          !normalizeBundle(consumed, workBudget)) {
        return false;
      }
      if (produced.empty() || produced != consumed) {
        return false;
      }
      const std::optional<std::size_t> laneLookupWork =
          comparisonWork(laneByBundle.size() + 1, produced.size() + 1);
      if (!laneLookupWork || !consumeWork(workBudget, *laneLookupWork)) {
        return false;
      }
      auto lane = laneByBundle.find(produced);
      if (lane == laneByBundle.end()) {
        lane = laneByBundle.emplace(produced, laneByBundle.size()).first;
      }
      const auto producerAnchors =
          groupAnchors(graph, producers, key.loop, workBudget,
                       /*preservePeriodicAlternatives=*/true);
      const auto consumerAnchors =
          groupAnchors(graph, consumers, key.loop, workBudget,
                       /*preservePeriodicAlternatives=*/true);
      if (!producerAnchors || !consumerAnchors) {
        return false;
      }
      path.uses.push_back({lane->second, lane->second, producers, consumers,
                           producerAnchors->first, producerAnchors->second,
                           consumerAnchors->first, consumerAnchors->second});
      producers.clear();
      consumers.clear();
      return true;
    };
    for (const LifecycleNodeFacts &facts : nodes) {
      const std::optional<bool> contained = scopeContainsWithWork(
          graph, pathScope, graph.getNodes()[facts.node].scope, workBudget);
      if (!contained) {
        return std::nullopt;
      }
      if (!*contained) {
        continue;
      }
      const std::size_t stableLookupWork = lookupWork(stableSlots.size() + 1);
      if ((facts.produced.size() != 0 &&
           stableLookupWork > std::numeric_limits<std::size_t>::max() /
                                  facts.produced.size()) ||
          (facts.consumed.size() != 0 &&
           stableLookupWork > std::numeric_limits<std::size_t>::max() /
                                  facts.consumed.size())) {
        return std::nullopt;
      }
      const std::size_t producedWork = facts.produced.size() * stableLookupWork;
      const std::size_t consumedWork = facts.consumed.size() * stableLookupWork;
      if (consumedWork >
              std::numeric_limits<std::size_t>::max() - producedWork ||
          !consumeWork(workBudget, producedWork + consumedWork)) {
        return std::nullopt;
      }
      const bool produces =
          std::any_of(facts.produced.begin(), facts.produced.end(),
                      [&](const LifecycleSlotKey &slot) {
                        return stableSlots.count(slot) != 0;
                      });
      const bool consumes =
          std::any_of(facts.consumed.begin(), facts.consumed.end(),
                      [&](const LifecycleSlotKey &slot) {
                        return stableSlots.count(slot) != 0;
                      });
      if (produces) {
        if (!consumers.empty() && !finalize()) {
          return std::nullopt;
        }
        producers.push_back(facts.node);
      }
      if (consumes) {
        if (producers.empty()) {
          return std::nullopt;
        }
        consumers.push_back(facts.node);
      }
    }
    if ((!producers.empty() || !consumers.empty()) && !finalize()) {
      return std::nullopt;
    }
    if (path.uses.empty()) {
      return std::nullopt;
    }
    certificate.paths.push_back(std::move(path));
  }
  certificate.lanes.resize(laneByBundle.size());
  for (const auto &[bundle, lane] : laneByBundle) {
    certificate.lanes[lane].id = lane;
    for (const LifecycleSlotKey &slot : bundle) {
      certificate.lanes[lane].slots.push_back({slot.domain, slot.extent, {}});
    }
  }
  // Exact bundle equality is too weak for ownership lanes: two observed
  // bundles such as {A, B} and {A} still share one physical token because an
  // access to A cannot belong to two independently circulating lanes. Merge
  // the connected components induced by shared exact slots before assigning
  // protocol lane identities.
  std::vector<std::size_t> laneParent(certificate.lanes.size());
  for (std::size_t lane = 0; lane < laneParent.size(); ++lane) {
    laneParent[lane] = lane;
  }
  const auto findLane = [&](std::size_t lane) {
    while (laneParent[lane] != lane) {
      lane = laneParent[lane];
    }
    return lane;
  };
  std::map<LifecycleSlotKey, std::size_t> slotOwner;
  for (const DerivedLifecycleLane &lane : certificate.lanes) {
    for (const DerivedLifecycleSlot &slot : lane.slots) {
      if (!consumeWork(workBudget, lookupWork(slotOwner.size() + 1))) {
        return std::nullopt;
      }
      const LifecycleSlotKey key{slot.domain, slot.extent};
      const auto existing = slotOwner.find(key);
      if (existing == slotOwner.end()) {
        slotOwner.emplace(key, lane.id);
        continue;
      }
      const std::size_t left = findLane(existing->second);
      const std::size_t right = findLane(lane.id);
      if (left != right) {
        laneParent[right] = left;
      }
    }
  }
  std::map<std::size_t, std::size_t> compactLane;
  std::vector<DerivedLifecycleLane> mergedLanes;
  std::vector<std::size_t> laneRemap(certificate.lanes.size());
  for (const DerivedLifecycleLane &lane : certificate.lanes) {
    const std::size_t root = findLane(lane.id);
    auto inserted = compactLane.emplace(root, compactLane.size());
    const std::size_t merged = inserted.first->second;
    if (inserted.second) {
      mergedLanes.push_back({merged, {}});
    }
    laneRemap[lane.id] = merged;
    mergedLanes[merged].slots.insert(mergedLanes[merged].slots.end(),
                                     lane.slots.begin(), lane.slots.end());
  }
  for (DerivedLifecycleLane &lane : mergedLanes) {
    const std::optional<std::size_t> sortWork =
        comparisonWork(lane.slots.size());
    if (!sortWork || !consumeWork(workBudget, *sortWork)) {
      return std::nullopt;
    }
    std::sort(
        lane.slots.begin(), lane.slots.end(),
        [](const DerivedLifecycleSlot &left,
           const DerivedLifecycleSlot &right) {
          return std::tie(left.domain, left.extent.begin, left.extent.end) <
                 std::tie(right.domain, right.extent.begin, right.extent.end);
        });
    lane.slots.erase(std::unique(lane.slots.begin(), lane.slots.end(),
                                 [](const DerivedLifecycleSlot &left,
                                    const DerivedLifecycleSlot &right) {
                                   return left.domain == right.domain &&
                                          intervalEqual(left.extent,
                                                        right.extent);
                                 }),
                     lane.slots.end());
  }
  certificate.lanes = std::move(mergedLanes);
  for (DerivedLifecyclePath &path : certificate.paths) {
    for (DerivedLifecycleUse &use : path.uses) {
      use.lane = laneRemap[use.lane];
      use.producerLane = laneRemap[use.producerLane];
    }
  }
  for (const DerivedLifecyclePath &path : certificate.paths) {
    std::set<std::size_t> used;
    for (const DerivedLifecycleUse &use : path.uses) {
      if (!consumeWork(workBudget, lookupWork(used.size() + 1))) {
        return std::nullopt;
      }
      used.insert(use.lane);
    }
    if (used.size() != certificate.lanes.size()) {
      return std::nullopt;
    }
  }
  return populateLifecycleSlotAccesses(graph, component, certificate,
                                       workBudget)
             ? std::optional<DerivedLifecycleCertificate>(
                   std::move(certificate))
             : std::nullopt;
}

bool hasSuccessorGuard(const SyncCoverGraph &graph, SyncCoverScopeId loop,
                       const std::vector<SyncCoverNodeId> &nodes,
                       SyncCoverCoverageWorkBudget *workBudget) {
  return std::all_of(nodes.begin(), nodes.end(), [&](SyncCoverNodeId node) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    return std::any_of(
        graph.getNodes()[node].guard.literals.begin(),
        graph.getNodes()[node].guard.literals.end(),
        [&](const SyncCoverGuardLiteral &literal) {
          if (!consumeWork(workBudget)) {
            return false;
          }
          if (literal.control >= graph.getControls().size()) {
            return false;
          }
          const auto &relation =
              graph.getControls()[literal.control].successorRelation;
          return relation && relation->loopScope == loop &&
                 relation->hasSuccessorAlternative == literal.alternative;
        });
  });
}

std::optional<DerivedLifecycleCertificate> buildAlternatingLifecycleCertificate(
    const SyncCoverGraph &graph, const SyncCoverLifecycleScc &component,
    const LifecyclePairKey &key, SyncCoverControlId phaseControl,
    const std::vector<SyncCoverScopeId> &paths,
    const std::vector<LifecycleNodeFacts> &nodes,
    const std::set<LifecycleSlotKey> &alternatingSlots,
    SyncCoverCoverageWorkBudget *workBudget) {
  if (paths.size() < 2 || alternatingSlots.size() < 2 ||
      phaseControl >= graph.getControls().size()) {
    return std::nullopt;
  }
  const std::optional<SyncCoverControlPhaseRelation> &phaseRelation =
      graph.getControls()[phaseControl].phaseRelation;
  if (!phaseRelation || phaseRelation->nextPhase.empty() ||
      phaseRelation->nextPhase.size() !=
          phaseRelation->activeAlternative.size() ||
      phaseRelation->initialPhase >= phaseRelation->nextPhase.size()) {
    return std::nullopt;
  }
  DerivedLifecycleCertificate certificate;
  certificate.alternating = true;
  certificate.loopScope = key.loop;
  certificate.producerResource = key.producer;
  certificate.consumerResource = key.consumer;
  certificate.periodicControl = phaseControl;
  std::map<LifecycleSlotKey, std::size_t> laneBySlot;
  for (const LifecycleSlotKey &slot : alternatingSlots) {
    if (!consumeWork(workBudget, lookupWork(laneBySlot.size() + 1))) {
      return std::nullopt;
    }
    const std::size_t lane = laneBySlot.size();
    laneBySlot.emplace(slot, lane);
    certificate.lanes.push_back({lane, {{slot.domain, slot.extent, {}}}});
  }
  std::map<unsigned, std::size_t> pathByAlternative;
  for (SyncCoverScopeId pathScope : paths) {
    if (!consumeWork(workBudget,
                     graph.getScopes()[pathScope].guard.literals.size())) {
      return std::nullopt;
    }
    const auto phaseLiteral =
        std::find_if(graph.getScopes()[pathScope].guard.literals.begin(),
                     graph.getScopes()[pathScope].guard.literals.end(),
                     [&](const SyncCoverGuardLiteral &literal) {
                       return literal.control == phaseControl;
                     });
    if (!consumeWork(workBudget, lookupWork(pathByAlternative.size() + 1))) {
      return std::nullopt;
    }
    if (phaseLiteral == graph.getScopes()[pathScope].guard.literals.end()) {
      return std::nullopt;
    }
    const auto pathInsertionPoint =
        pathByAlternative.lower_bound(phaseLiteral->alternative);
    if (pathInsertionPoint != pathByAlternative.end() &&
        pathInsertionPoint->first == phaseLiteral->alternative) {
      return std::nullopt;
    }
    LifecycleSlotBundle produced;
    LifecycleSlotBundle consumed;
    std::vector<SyncCoverNodeId> producerNodes;
    std::vector<SyncCoverNodeId> consumerNodes;
    for (const LifecycleNodeFacts &facts : nodes) {
      const std::optional<bool> contained = scopeContainsWithWork(
          graph, pathScope, graph.getNodes()[facts.node].scope, workBudget);
      if (!contained) {
        return std::nullopt;
      }
      if (!*contained) {
        continue;
      }
      for (const LifecycleSlotKey &slot : facts.produced) {
        if (!consumeWork(workBudget, lookupWork(alternatingSlots.size() + 1))) {
          return std::nullopt;
        }
        if (alternatingSlots.count(slot) != 0) {
          produced.push_back(slot);
          producerNodes.push_back(facts.node);
        }
      }
      for (const LifecycleSlotKey &slot : facts.consumed) {
        if (!consumeWork(workBudget, lookupWork(alternatingSlots.size() + 1))) {
          return std::nullopt;
        }
        if (alternatingSlots.count(slot) != 0) {
          consumed.push_back(slot);
          consumerNodes.push_back(facts.node);
        }
      }
    }
    if (!normalizeBundle(produced, workBudget) ||
        !normalizeBundle(consumed, workBudget)) {
      return std::nullopt;
    }
    const std::optional<std::size_t> producerSortWork =
        comparisonWork(producerNodes.size());
    const std::optional<std::size_t> consumerSortWork =
        comparisonWork(consumerNodes.size());
    if (!producerSortWork || !consumerSortWork ||
        !consumeWork(workBudget, *producerSortWork + *consumerSortWork)) {
      return std::nullopt;
    }
    std::sort(producerNodes.begin(), producerNodes.end());
    producerNodes.erase(std::unique(producerNodes.begin(), producerNodes.end()),
                        producerNodes.end());
    std::sort(consumerNodes.begin(), consumerNodes.end());
    consumerNodes.erase(std::unique(consumerNodes.begin(), consumerNodes.end()),
                        consumerNodes.end());
    if (produced.size() != 1 || consumed.size() != 1 ||
        produced.front() == consumed.front() || producerNodes.empty() ||
        consumerNodes.empty() ||
        !hasSuccessorGuard(graph, key.loop, producerNodes, workBudget)) {
      return std::nullopt;
    }
    const auto producerAnchors =
        groupAnchors(graph, producerNodes, key.loop, workBudget,
                     /*preservePeriodicAlternatives=*/true);
    if (!producerAnchors) {
      return std::nullopt;
    }
    const std::pair<SyncCoverAnchor, SyncCoverAnchor> consumerAnchors = {
        {SyncCoverAnchorKind::ScopeEntry, 0, pathScope, 0},
        {SyncCoverAnchorKind::ScopeExit, 0, pathScope, 0}};
    if (!consumeWork(workBudget, lookupWork(laneBySlot.size()) * 2)) {
      return std::nullopt;
    }
    DerivedLifecyclePath path;
    path.scope = pathScope;
    path.uses.push_back({laneBySlot.at(consumed.front()),
                         laneBySlot.at(produced.front()), producerNodes,
                         consumerNodes, producerAnchors->first,
                         producerAnchors->second, consumerAnchors.first,
                         consumerAnchors.second});
    pathByAlternative.emplace_hint(pathInsertionPoint,
                                   phaseLiteral->alternative,
                                   certificate.paths.size());
    certificate.paths.push_back(std::move(path));
  }

  std::vector<bool> visitedPhases(phaseRelation->nextPhase.size());
  std::vector<bool> visitedPaths(certificate.paths.size());
  std::set<std::size_t> consumedLanes;
  std::set<std::size_t> producedLanes;
  for (const DerivedLifecyclePath &path : certificate.paths) {
    if (!consumeWork(workBudget, lookupWork(consumedLanes.size() + 1) +
                                     lookupWork(producedLanes.size() + 1))) {
      return std::nullopt;
    }
    consumedLanes.insert(path.uses.front().lane);
    producedLanes.insert(path.uses.front().producerLane);
  }
  if (consumedLanes.size() != certificate.lanes.size() ||
      producedLanes != consumedLanes) {
    return std::nullopt;
  }
  std::size_t phase = phaseRelation->initialPhase;
  while (!visitedPhases[phase]) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    visitedPhases[phase] = true;
    const std::size_t nextPhase = phaseRelation->nextPhase[phase];
    if (nextPhase >= phaseRelation->nextPhase.size()) {
      return std::nullopt;
    }
    if (!consumeWork(workBudget,
                     lookupWork(pathByAlternative.size() + 1) * 2)) {
      return std::nullopt;
    }
    const auto sourcePath =
        pathByAlternative.find(phaseRelation->activeAlternative[phase]);
    const auto targetPath =
        pathByAlternative.find(phaseRelation->activeAlternative[nextPhase]);
    if (sourcePath == pathByAlternative.end() ||
        targetPath == pathByAlternative.end()) {
      return std::nullopt;
    }
    visitedPaths[sourcePath->second] = true;
    const DerivedLifecycleUse &sourceUse =
        certificate.paths[sourcePath->second].uses.front();
    const DerivedLifecycleUse &targetUse =
        certificate.paths[targetPath->second].uses.front();
    if (sourceUse.producerLane != targetUse.lane) {
      return std::nullopt;
    }
    phase = nextPhase;
  }
  if (!consumeWork(workBudget, visitedPaths.size())) {
    return std::nullopt;
  }
  if (phase != phaseRelation->initialPhase ||
      std::find(visitedPaths.begin(), visitedPaths.end(), false) !=
          visitedPaths.end()) {
    return std::nullopt;
  }
  if (!consumeWork(workBudget, lookupWork(pathByAlternative.size() + 1))) {
    return std::nullopt;
  }
  const auto initialPath = pathByAlternative.find(
      phaseRelation->activeAlternative[phaseRelation->initialPhase]);
  if (initialPath == pathByAlternative.end()) {
    return std::nullopt;
  }
  certificate.initialReadyLane =
      certificate.paths[initialPath->second].uses.front().lane;
  for (std::size_t lane = 0; lane < certificate.lanes.size(); ++lane) {
    if (lane != certificate.initialReadyLane) {
      certificate.initiallyFreeLanes.push_back(lane);
    }
  }
  const DerivedLifecycleSlot &initialSlot =
      certificate.lanes[certificate.initialReadyLane].slots.front();
  const auto &loopTimeline = graph.getScopes()[key.loop].timeline;
  if (!loopTimeline) {
    return std::nullopt;
  }
  for (const SyncCoverStorageAccess &access : graph.getStorageAccesses()) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    const SyncCoverNode &node = graph.getNodes()[access.node];
    const std::optional<SyncCoverTimelinePosition> afterNode =
        resolveSyncCoverAnchor(graph,
                               {SyncCoverAnchorKind::AfterNode, node.id, 0, 0});
    if (node.resource == key.producer && access.exactPhysical && afterNode &&
        access.domain == initialSlot.domain &&
        intervalEqual(access.extent, initialSlot.extent) &&
        syncCoverStorageModeWrites(access.mode) &&
        *afterNode < loopTimeline->begin) {
      certificate.initialProducers.push_back(node.id);
    }
  }
  if (certificate.initialProducers.empty()) {
    return std::nullopt;
  }
  const auto initialAnchors =
      groupAnchors(graph, certificate.initialProducers, key.loop, workBudget,
                   /*preservePeriodicAlternatives=*/true);
  if (!initialAnchors) {
    return std::nullopt;
  }
  certificate.initialWriteAcquireAnchor = initialAnchors->first;
  certificate.initialReadyAnchor = {SyncCoverAnchorKind::ScopeEntry, 0,
                                    key.loop, 0};
  return populateLifecycleSlotAccesses(graph, component, certificate,
                                       workBudget)
             ? std::optional<DerivedLifecycleCertificate>(
                   std::move(certificate))
             : std::nullopt;
}

} // namespace

namespace {

struct DerivedLifecycleSchedules {
  SyncCoverProtocolError error = SyncCoverProtocolError::None;
  std::vector<DerivedLifecycleCertificate> certificates;
  std::size_t inspectedDemands = 0;
  std::size_t inspectedAccesses = 0;
  std::size_t lifecycleAccessIncidences = 0;
  std::size_t candidateResourcePairs = 0;
  std::optional<std::size_t> invalidIndex;
};

DerivedLifecycleSchedules deriveLifecycleSchedules(
    const SyncCoverGraph &graph, const SyncCoverLifecycleScc &component,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  DerivedLifecycleSchedules result;
  if (!graphFitsProtocolLimits(graph, limits)) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  if (!graph.isStructureFrozen() || !graph.validate()) {
    result.error = SyncCoverProtocolError::InvalidGraph;
    return result;
  }
  std::map<LifecyclePairKey, LifecyclePairFacts> pairs;
  std::map<LifecyclePairKey, std::set<LifecycleSlotKey>> releaseSlots;
  std::map<CrossChildRoundTripKey, LifecycleEdge> crossChildReady;
  std::map<CrossChildRoundTripKey, LifecycleEdge> crossChildRelease;
  std::size_t proposalSlotIncidences = 0;
  // The storage-lifecycle index groups ordinary components by frontend access
  // family.  Exact physical planning can nevertheless place accesses from two
  // families on the same interval.  Hardware ownership follows that physical
  // interval, so discover a direct/reverse round trip across the complete
  // graph and assign it to this component's owning loop.
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    if (!std::binary_search(component.demands.begin(), component.demands.end(),
                            demandId)) {
      continue;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (demand.distance > 1) {
      continue;
    }
    for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
      if (!consumeWork(workBudget) ||
          witnessId >= graph.getStorageWitnesses().size()) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        return result;
      }
      const SyncCoverStorageWitness &witness =
          graph.getStorageWitnesses()[witnessId];
      if (witness.sourceAccess >= graph.getStorageAccesses().size() ||
          witness.targetAccess >= graph.getStorageAccesses().size()) {
        result.error = SyncCoverProtocolError::InvalidGraph;
        return result;
      }
      const SyncCoverStorageAccess &source =
          graph.getStorageAccesses()[witness.sourceAccess];
      const SyncCoverStorageAccess &target =
          graph.getStorageAccesses()[witness.targetAccess];
      const SyncCoverNode &sourceNode = graph.getNodes()[source.node];
      const SyncCoverNode &targetNode = graph.getNodes()[target.node];
      const bool exactCrossResource =
          source.exactPhysical && target.exactPhysical &&
          source.domain == target.domain &&
          intervalEqual(source.extent, target.extent) &&
          sourceNode.resource != targetNode.resource;
      if (!exactCrossResource) {
        continue;
      }
      std::optional<SyncCoverScopeId> loop;
      if (demand.distance == 0) {
        loop = nearestCommonLoop(graph, sourceNode.scope, targetNode.scope,
                                 workBudget);
      } else {
        loop = demand.scope;
      }
      if (workBudget && workBudget->exhausted) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        return result;
      }
      if (!loop || *loop != component.loopScope) {
        continue;
      }
      const CrossChildRoundTripKey roundTrip{
          demand.distance == 0 ? LifecyclePairKey{*loop, sourceNode.resource,
                                                  targetNode.resource}
                               : LifecyclePairKey{*loop, targetNode.resource,
                                                  sourceNode.resource},
          {source.domain, source.extent},
          demand.distance == 0 ? source.node : target.node,
          demand.distance == 0 ? target.node : source.node};
      auto &index = demand.distance == 0 ? crossChildReady : crossChildRelease;
      if (!consumeWork(workBudget, lookupWork(index.size() + 1))) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        return result;
      }
      index.try_emplace(roundTrip,
                        LifecycleEdge{demandId, source.node, target.node,
                                      sourceNode.resource, targetNode.resource,
                                      source.domain, witness.sourceAccess,
                                      witness.targetAccess, witnessId,
                                      demand.distance});
      if (index.size() > limits.maximumLifecycleEdges) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        return result;
      }
    }
  }
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    ++result.inspectedDemands;
    if (!consumeWork(workBudget, lookupWork(component.demands.size() + 1))) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    if (!std::binary_search(component.demands.begin(), component.demands.end(),
                            demandId)) {
      continue;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
      if (!consumeWork(workBudget)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        return result;
      }
      const SyncCoverStorageWitness &witness =
          graph.getStorageWitnesses()[witnessId];
      const SyncCoverStorageAccess &source =
          graph.getStorageAccesses()[witness.sourceAccess];
      const SyncCoverStorageAccess &target =
          graph.getStorageAccesses()[witness.targetAccess];
      ++result.inspectedAccesses;
      const SyncCoverNode &sourceNode = graph.getNodes()[source.node];
      const SyncCoverNode &targetNode = graph.getNodes()[target.node];
      const bool exactSameSlot = source.exactPhysical && target.exactPhysical &&
                                 source.domain == target.domain &&
                                 intervalEqual(source.extent, target.extent);
      const bool readyAccess = exactSameSlot &&
                               syncCoverStorageModeWrites(source.mode) &&
                               syncCoverStorageModeReads(target.mode) &&
                               sourceNode.resource != targetNode.resource;
      if (readyAccess) {
        const std::optional<SyncCoverScopeId> loop =
            demand.distance != 0
                ? std::optional<SyncCoverScopeId>(demand.scope)
                : nearestCommonLoop(graph, sourceNode.scope, targetNode.scope,
                                    workBudget);
        if (!loop) {
          if (workBudget && workBudget->exhausted) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            return result;
          }
        } else {
          const LifecyclePairKey key{*loop, sourceNode.resource,
                                     targetNode.resource};
          if (!consumeWork(workBudget, lookupWork(pairs.size() + 1))) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            return result;
          }
          auto [pair, insertedPair] = pairs.try_emplace(key);
          if (insertedPair && pairs.size() > limits.maximumLifecycleSccs) {
            result.error = SyncCoverProtocolError::LimitExceeded;
            return result;
          }
          if (!consumeWork(workBudget,
                           lookupWork(pair->second.slots.size() + 1))) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            return result;
          }
          const bool insertedSlot =
              pair->second.slots.insert({source.domain, source.extent}).second;
          if (insertedSlot) {
            if (proposalSlotIncidences ==
                limits.maximumLifecycleDomainIncidences) {
              result.error = SyncCoverProtocolError::LimitExceeded;
              return result;
            }
            ++proposalSlotIncidences;
          }
        }
      }
      const bool releaseAccess = demand.distance != 0 && exactSameSlot &&
                                 syncCoverStorageModeReads(source.mode) &&
                                 syncCoverStorageModeWrites(target.mode) &&
                                 sourceNode.resource != targetNode.resource;
      if (releaseAccess) {
        const LifecyclePairKey key{demand.scope, targetNode.resource,
                                   sourceNode.resource};
        if (!consumeWork(workBudget, lookupWork(releaseSlots.size() + 1))) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          return result;
        }
        auto [pair, insertedPair] = releaseSlots.try_emplace(key);
        if (insertedPair && releaseSlots.size() > limits.maximumLifecycleSccs) {
          result.error = SyncCoverProtocolError::LimitExceeded;
          return result;
        }
        if (!consumeWork(workBudget, lookupWork(pair->second.size() + 1))) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          return result;
        }
        const bool insertedSlot =
            pair->second.insert({source.domain, source.extent}).second;
        if (insertedSlot) {
          if (proposalSlotIncidences ==
              limits.maximumLifecycleDomainIncidences) {
            result.error = SyncCoverProtocolError::LimitExceeded;
            return result;
          }
          ++proposalSlotIncidences;
        }
      }
    }
  }
  result.candidateResourcePairs = pairs.size();
  if (pairs.size() > limits.maximumLifecycleSccs) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }

  std::map<ExactRoundTripCutKey, std::size_t> groupedRoundTrips;
  std::vector<DerivedLifecycleCertificate> roundTripCertificates;
  for (const auto &entry : crossChildReady) {
    const CrossChildRoundTripKey &key = entry.first;
    if (!consumeWork(workBudget, lookupWork(crossChildRelease.size() + 1))) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    if (crossChildRelease.find(key) == crossChildRelease.end()) {
      continue;
    }
    std::optional<DerivedLifecycleCertificate> certificate =
        buildCrossChildRoundTripCertificate(graph, component, key, workBudget);
    if (workBudget && workBudget->exhausted) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    if (!certificate) {
      continue;
    }
    const DerivedLifecycleUse &use = certificate->paths.front().uses.front();
    const ExactRoundTripCutKey cutKey{key.pair, use.writeAcquireAnchor,
                                      use.readyAnchor, use.readAcquireAnchor,
                                      use.releaseAnchor};
    if (!consumeWork(workBudget, lookupWork(groupedRoundTrips.size() + 1))) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    const auto insertion =
        groupedRoundTrips.emplace(cutKey, roundTripCertificates.size());
    if (insertion.second) {
      roundTripCertificates.push_back(std::move(*certificate));
      continue;
    }
    DerivedLifecycleCertificate &grouped =
        roundTripCertificates[insertion.first->second];
    DerivedLifecycleUse &groupedUse = grouped.paths.front().uses.front();
    const DerivedLifecycleUse &candidateUse =
        certificate->paths.front().uses.front();
    if (!consumeWork(workBudget, candidateUse.producers.size() +
                                     candidateUse.consumers.size())) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    groupedUse.producers.insert(groupedUse.producers.end(),
                                candidateUse.producers.begin(),
                                candidateUse.producers.end());
    groupedUse.consumers.insert(groupedUse.consumers.end(),
                                candidateUse.consumers.begin(),
                                candidateUse.consumers.end());
    for (const DerivedLifecycleSlot &slot : certificate->lanes.front().slots) {
      auto existing =
          std::find_if(grouped.lanes.front().slots.begin(),
                       grouped.lanes.front().slots.end(),
                       [&](const DerivedLifecycleSlot &candidate) {
                         return candidate.domain == slot.domain &&
                                intervalEqual(candidate.extent, slot.extent);
                       });
      if (!consumeWork(workBudget, grouped.lanes.front().slots.size() + 1)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        return result;
      }
      if (existing == grouped.lanes.front().slots.end()) {
        grouped.lanes.front().slots.push_back(slot);
      } else {
        existing->accesses.insert(existing->accesses.end(),
                                  slot.accesses.begin(), slot.accesses.end());
      }
    }
  }
  for (DerivedLifecycleCertificate &certificate : roundTripCertificates) {
    DerivedLifecycleUse &use = certificate.paths.front().uses.front();
    const auto normalizeNodeIds = [&](std::vector<SyncCoverNodeId> &nodes) {
      const std::optional<std::size_t> work = comparisonWork(nodes.size());
      if (!work || !consumeWork(workBudget, *work)) {
        return false;
      }
      std::sort(nodes.begin(), nodes.end());
      nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
      return true;
    };
    if (!normalizeNodeIds(use.producers) || !normalizeNodeIds(use.consumers)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    for (DerivedLifecycleSlot &slot : certificate.lanes.front().slots) {
      const std::optional<std::size_t> work =
          comparisonWork(slot.accesses.size());
      if (!work || !consumeWork(workBudget, *work)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        return result;
      }
      std::sort(slot.accesses.begin(), slot.accesses.end());
      slot.accesses.erase(
          std::unique(slot.accesses.begin(), slot.accesses.end()),
          slot.accesses.end());
    }
    const SyncCoverProtocolError bounds =
        checkLifecycleCertificateBounds(certificate, limits, workBudget);
    if (bounds != SyncCoverProtocolError::None) {
      result.error = bounds;
      return result;
    }
    if (result.certificates.size() == limits.maximumLifecycleSccs) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.certificates.clear();
      return result;
    }
    certificate.id = result.certificates.size();
    result.certificates.push_back(std::move(certificate));
  }

  std::size_t accessIncidences = 0;
  for (const auto &[key, facts] : pairs) {
    if (!consumeWork(workBudget, lookupWork(releaseSlots.size() + 1))) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    const auto releases = releaseSlots.find(key);
    if (releases == releaseSlots.end()) {
      continue;
    }
    std::set<LifecycleSlotKey> completeSlots;
    auto readySlot = facts.slots.begin();
    auto releaseSlot = releases->second.begin();
    while (readySlot != facts.slots.end() &&
           releaseSlot != releases->second.end()) {
      if (!consumeWork(workBudget)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        return result;
      }
      if (*readySlot < *releaseSlot) {
        ++readySlot;
      } else if (*releaseSlot < *readySlot) {
        ++releaseSlot;
      } else {
        completeSlots.insert(completeSlots.end(), *readySlot);
        ++readySlot;
        ++releaseSlot;
      }
    }
    if (completeSlots.empty()) {
      continue;
    }
    SyncCoverProtocolError collectError = SyncCoverProtocolError::None;
    std::vector<LifecycleNodeFacts> nodes = collectLifecycleNodes(
        graph, key.loop, key.producer, key.consumer, completeSlots, limits,
        accessIncidences, workBudget, collectError);
    if (collectError != SyncCoverProtocolError::None) {
      result.error = collectError;
      return result;
    }
    if (nodes.empty()) {
      continue;
    }
    const std::size_t componentLookupWork =
        lookupWork(component.nodes.size() + 1);
    if (nodes.size() != 0 &&
        componentLookupWork >
            std::numeric_limits<std::size_t>::max() / nodes.size()) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    if (!consumeWork(workBudget, nodes.size() * componentLookupWork)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                               [&](const auto &node) {
                                 return !std::binary_search(
                                     component.nodes.begin(),
                                     component.nodes.end(), node.node);
                               }),
                nodes.end());
    if (nodes.empty()) {
      continue;
    }
    const LoopPhaseControlQuery phaseQuery =
        findLoopPhaseControl(graph, key.loop, workBudget);
    if (phaseQuery.workUnavailable || (workBudget && workBudget->exhausted)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    if (phaseQuery.ambiguous) {
      continue;
    }
    const std::optional<SyncCoverControlId> phaseControl = phaseQuery.control;
    std::vector<SyncCoverScopeId> paths;
    if (phaseControl) {
      const std::vector<unsigned> alternatives =
          reachableAlternatives(graph.getControls()[*phaseControl], workBudget);
      for (unsigned alternative : alternatives) {
        const std::optional<SyncCoverScopeId> path = pathScopeForAlternative(
            graph, key.loop, *phaseControl, alternative, nodes, workBudget);
        if (!path) {
          if (workBudget && workBudget->exhausted) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            return result;
          }
          paths.clear();
          break;
        }
        paths.push_back(*path);
      }
    } else {
      paths.push_back(key.loop);
    }
    if (paths.empty()) {
      continue;
    }

    std::set<LifecycleSlotKey> stableSlots;
    std::set<LifecycleSlotKey> alternatingSlots;
    for (const LifecycleSlotKey &slot : completeSlots) {
      std::vector<SlotPathPresence> presence;
      for (SyncCoverScopeId path : paths) {
        const std::optional<SlotPathPresence> item =
            slotPresence(graph, path, slot, nodes, workBudget);
        if (!item) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          return result;
        }
        presence.push_back(*item);
      }
      const bool stable = std::all_of(
          presence.begin(), presence.end(), [](const SlotPathPresence &item) {
            return item.produced && item.consumed &&
                   item.firstProducer < item.firstConsumer;
          });
      const bool alternating =
          paths.size() > 1 &&
          std::none_of(presence.begin(), presence.end(),
                       [](const SlotPathPresence &item) {
                         return item.produced && item.consumed;
                       }) &&
          std::any_of(
              presence.begin(), presence.end(),
              [](const SlotPathPresence &item) { return item.produced; }) &&
          std::any_of(
              presence.begin(), presence.end(),
              [](const SlotPathPresence &item) { return item.consumed; });
      if (stable) {
        if (!consumeWork(workBudget, lookupWork(stableSlots.size() + 1))) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          return result;
        }
        stableSlots.insert(slot);
      } else if (alternating) {
        if (!consumeWork(workBudget, lookupWork(alternatingSlots.size() + 1))) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          return result;
        }
        alternatingSlots.insert(slot);
      }
    }
    if (std::optional<DerivedLifecycleCertificate> stable =
            buildStableLifecycleCertificate(graph, component, key, paths, nodes,
                                            stableSlots, workBudget)) {
      const SyncCoverProtocolError bounds =
          checkLifecycleCertificateBounds(*stable, limits, workBudget);
      if (bounds != SyncCoverProtocolError::None) {
        result.error = bounds;
        return result;
      }
      if (hasMatchingReleaseDemands(graph, component, *stable, workBudget)) {
        stable->id = result.certificates.size();
        result.certificates.push_back(std::move(*stable));
      } else if (workBudget && workBudget->exhausted) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        return result;
      }
    }
    if (phaseControl) {
      if (std::optional<DerivedLifecycleCertificate> alternating =
              buildAlternatingLifecycleCertificate(
                  graph, component, key, *phaseControl, paths, nodes,
                  alternatingSlots, workBudget)) {
        const SyncCoverProtocolError bounds =
            checkLifecycleCertificateBounds(*alternating, limits, workBudget);
        if (bounds != SyncCoverProtocolError::None) {
          result.error = bounds;
          return result;
        }
        if (hasMatchingReleaseDemands(graph, component, *alternating,
                                      workBudget)) {
          alternating->id = result.certificates.size();
          result.certificates.push_back(std::move(*alternating));
        } else if (workBudget && workBudget->exhausted) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          return result;
        }
      }
    }
    if (result.certificates.size() > limits.maximumLifecycleSccs) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.certificates.clear();
      return result;
    }
  }
  if (workBudget && workBudget->exhausted) {
    result.error = SyncCoverProtocolError::WorkLimitExceeded;
    result.certificates.clear();
  }
  result.lifecycleAccessIncidences = accessIncidences;
  return result;
}

SyncCoverCutPoint lifecyclePoint(SyncCoverCutPointKind kind,
                                 std::uint32_t resource,
                                 const SyncCoverAnchor &anchor) {
  return {kind, resource, anchor, {}, 0};
}

std::optional<std::vector<std::size_t>>
protocolPhysicalKey(const SyncCoverEventProtocol &protocol,
                    SyncCoverProtocolLimits limits,
                    SyncCoverCoverageWorkBudget *workBudget) {
  std::vector<std::size_t> key;
  const auto append = [&](std::size_t value) {
    if (key.size() == limits.maximumLifecycleCatalogPayloadIncidences) {
      return false;
    }
    key.push_back(value);
    return true;
  };
  const auto appendOptional = [&](std::optional<std::size_t> value) {
    return append(value.has_value()) && (!value || append(*value));
  };
  const auto appendValues = [&](const std::vector<std::size_t> &values) {
    if (!append(values.size())) {
      return false;
    }
    for (std::size_t value : values) {
      if (!append(value)) {
        return false;
      }
    }
    return true;
  };
  const auto appendAnchor = [&](const SyncCoverAnchor &anchor) {
    return append(static_cast<std::size_t>(anchor.kind)) &&
           append(anchor.node) && append(anchor.scope) &&
           append(anchor.position);
  };
  const auto appendGuard = [&](const SyncCoverGuard &guard) {
    if (!append(guard.literals.size())) {
      return false;
    }
    for (const SyncCoverGuardLiteral &literal : guard.literals) {
      if (!append(literal.control) || !append(literal.alternative)) {
        return false;
      }
    }
    return true;
  };
  const auto appendPoint = [&](const SyncCoverCutPoint &point) {
    return append(static_cast<std::size_t>(point.kind)) &&
           append(point.resource) && appendAnchor(point.anchor) &&
           appendGuard(point.guard) && append(point.ordinal);
  };

  if (!append(static_cast<std::size_t>(protocol.kind)) ||
      !append(protocol.loop.has_value())) {
    return std::nullopt;
  }
  if (protocol.loop && (!append(protocol.loop->scope) ||
                        !append(protocol.loop->mayExecuteZeroTimes) ||
                        !appendOptional(protocol.loop->phaseControl) ||
                        !appendValues(protocol.loop->laneByPhase))) {
    return std::nullopt;
  }
  if (!appendOptional(protocol.lifetimeScope) ||
      !append(protocol.lifetimeMayExecuteZeroTimes) ||
      !append(protocol.channels.size())) {
    return std::nullopt;
  }
  for (const SyncCoverEventChannel &channel : protocol.channels) {
    if (!append(channel.id) ||
        !append(static_cast<std::size_t>(channel.flow)) ||
        !appendPoint(channel.set) || !appendPoint(channel.wait) ||
        !append(channel.width) || !append(channel.distance) ||
        !append(channel.suppliedRequirements) ||
        !appendValues(channel.activePhases) ||
        !append(channel.exportsCompletionAtExit) ||
        !append(channel.transfers.size())) {
      return std::nullopt;
    }
    for (const SyncCoverEventTransfer &transfer : channel.transfers) {
      if (!append(transfer.id) || !appendPoint(transfer.set) ||
          !appendPoint(transfer.wait) || !append(transfer.setLane) ||
          !append(transfer.waitLane) || !appendValues(transfer.activePhases)) {
        return std::nullopt;
      }
    }
    if (!append(channel.actions.size())) {
      return std::nullopt;
    }
    for (const SyncCoverProtocolAction &action : channel.actions) {
      if (!append(action.id) ||
          !append(static_cast<std::size_t>(action.kind)) ||
          !append(static_cast<std::size_t>(action.segment)) ||
          !appendPoint(action.point) || !append(action.lane) ||
          !append(static_cast<std::size_t>(action.guard)) ||
          !appendValues(action.activePhases)) {
        return std::nullopt;
      }
    }
    if (!append(channel.supplies.size())) {
      return std::nullopt;
    }
    for (const SyncCoverProtocolSupply &supply : channel.supplies) {
      if (!append(supply.setAction) || !append(supply.waitAction) ||
          !append(supply.distance) || !appendOptional(supply.distanceScope) ||
          !append(static_cast<std::size_t>(supply.kind))) {
        return std::nullopt;
      }
    }
  }
  if (!append(protocol.rearmProofs.size())) {
    return std::nullopt;
  }
  for (const SyncCoverProtocolRearmProof &proof : protocol.rearmProofs) {
    if (!append(proof.fromWaitChannel) || !append(proof.toSetChannel) ||
        !append(proof.iterationDistance) || !append(proof.evidence)) {
      return std::nullopt;
    }
  }
  if (!consumeWork(workBudget, key.size())) {
    return std::nullopt;
  }
  return key;
}

std::optional<std::vector<std::size_t>> activePhasesForTransfer(
    const SyncCoverGraph &graph, const SyncCoverProtocolLoopSchedule &schedule,
    const SyncCoverCutPoint &set, const SyncCoverCutPoint &wait,
    SyncCoverCoverageWorkBudget *workBudget) {
  const std::optional<SyncCoverGuard> setGuard =
      effectivePointGuard(graph, set, workBudget);
  const std::optional<SyncCoverGuard> waitGuard =
      effectivePointGuard(graph, wait, workBudget);
  if (!setGuard || !waitGuard || (workBudget && workBudget->exhausted)) {
    return std::nullopt;
  }
  // A conditional lifecycle transfer is admissible only as one balanced
  // transaction: both ends execute under the same structured-control guard.
  // Omitting the whole transaction then leaves the one-bit event state
  // unchanged.  The protocol verifier explores the resulting guard worlds.
  if (setGuard->literals != waitGuard->literals) {
    return std::nullopt;
  }
  const SyncCoverGuard &loopGuard = graph.getScopes()[schedule.scope].guard;
  if (!schedule.phaseControl) {
    if (!syncCoverGuardsCompatible(loopGuard, *setGuard)) {
      return std::nullopt;
    }
    return std::vector<std::size_t>{};
  }
  const SyncCoverControl &control = graph.getControls()[*schedule.phaseControl];
  if (!control.phaseRelation) {
    return std::nullopt;
  }
  std::vector<std::size_t> active;
  for (std::size_t phase = 0;
       phase < control.phaseRelation->activeAlternative.size(); ++phase) {
    if (!consumeWork(workBudget, loopGuard.literals.size() + 2)) {
      return std::nullopt;
    }
    SyncCoverGuard condition = loopGuard;
    condition.literals.push_back(
        {*schedule.phaseControl,
         control.phaseRelation->activeAlternative[phase]});
    if (!normalizeSyncCoverGuard(condition)) {
      return std::nullopt;
    }
    if (syncCoverGuardsCompatible(condition, *setGuard)) {
      active.push_back(phase);
    }
  }
  if (active.empty()) {
    return std::nullopt;
  }
  return active;
}

std::optional<std::vector<std::size_t>> activePhasesForPoint(
    const SyncCoverGraph &graph, const SyncCoverProtocolLoopSchedule &schedule,
    const SyncCoverCutPoint &point, SyncCoverCoverageWorkBudget *workBudget) {
  return activePhasesForTransfer(graph, schedule, point, point, workBudget);
}

std::optional<SyncCoverProtocolActionGuard> temporalGuardForPoint(
    const SyncCoverGraph &graph, const SyncCoverProtocolLoopSchedule &schedule,
    const SyncCoverCutPoint &point, SyncCoverCoverageWorkBudget *workBudget) {
  const std::optional<SyncCoverGuard> guard =
      effectivePointGuard(graph, point, workBudget);
  if (!guard) {
    return std::nullopt;
  }
  SyncCoverProtocolActionGuard temporal = SyncCoverProtocolActionGuard::Always;
  for (const SyncCoverGuardLiteral &literal : guard->literals) {
    if (!consumeWork(workBudget) ||
        literal.control >= graph.getControls().size()) {
      return std::nullopt;
    }
    const auto &relation =
        graph.getControls()[literal.control].firstIterationRelation;
    if (!relation || relation->loopScope != schedule.scope) {
      continue;
    }
    const SyncCoverProtocolActionGuard candidate =
        literal.alternative == relation->firstIterationAlternative
            ? SyncCoverProtocolActionGuard::FirstIteration
            : SyncCoverProtocolActionGuard::NotFirstIteration;
    if (temporal != SyncCoverProtocolActionGuard::Always &&
        temporal != candidate) {
      return std::nullopt;
    }
    temporal = candidate;
  }
  return temporal;
}

std::size_t appendLifecycleAction(SyncCoverEventChannel &channel,
                                  SyncCoverProtocolActionKind kind,
                                  SyncCoverProtocolActionSegment segment,
                                  SyncCoverCutPoint point, std::size_t lane,
                                  SyncCoverProtocolActionGuard guard,
                                  std::vector<std::size_t> activePhases = {}) {
  const std::size_t id = channel.actions.size();
  channel.actions.push_back({id, kind, segment, std::move(point), lane, guard,
                             std::move(activePhases)});
  return id;
}

bool appendParentInvocationCompletionExports(
    SyncCoverEventChannel &channel, SyncCoverScopeId lifetimeScope,
    SyncCoverCoverageWorkBudget *workBudget) {
  const std::size_t localSupplyCount = channel.supplies.size();
  for (std::size_t supplyIndex = 0; supplyIndex < localSupplyCount;
       ++supplyIndex) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    const SyncCoverProtocolSupply &supply = channel.supplies[supplyIndex];
    if (supply.kind != SyncCoverProtocolSupplyKind::TokenPair ||
        supply.distanceScope || supply.setAction >= channel.actions.size() ||
        supply.waitAction >= channel.actions.size()) {
      continue;
    }
    const SyncCoverProtocolAction &set = channel.actions[supply.setAction];
    const SyncCoverProtocolAction &wait = channel.actions[supply.waitAction];
    if (set.segment != SyncCoverProtocolActionSegment::Body ||
        wait.segment != SyncCoverProtocolActionSegment::Body) {
      continue;
    }
    bool alreadyPresent = false;
    for (const SyncCoverProtocolSupply &candidate : channel.supplies) {
      if (!consumeWork(workBudget)) {
        return false;
      }
      alreadyPresent =
          candidate.kind == SyncCoverProtocolSupplyKind::CompletionExport &&
          candidate.setAction == supply.setAction &&
          candidate.waitAction == supply.waitAction &&
          candidate.distance == 0 && candidate.distanceScope == lifetimeScope;
      if (alreadyPresent) {
        break;
      }
    }
    if (!alreadyPresent) {
      channel.supplies.push_back(
          {supply.setAction, supply.waitAction, 0, lifetimeScope,
           SyncCoverProtocolSupplyKind::CompletionExport});
    }
  }
  return true;
}

std::optional<SyncCoverOrderingRequirementMask>
lifecycleEventRequirements(const SyncCoverProtocolTargetContract &target,
                           std::uint32_t source, std::uint32_t destination) {
  const auto capability = std::lower_bound(
      target.eventCapabilities.begin(), target.eventCapabilities.end(),
      SyncCoverProtocolTargetContract::EventCapability{source, destination, 0});
  if (capability == target.eventCapabilities.end() ||
      capability->sourceResource != source ||
      capability->targetResource != destination) {
    return std::nullopt;
  }
  return capability->suppliedRequirements;
}

/// Construct the conservative direct basis for a distance-one recurrence.
/// The ready token is consumed at the next loop-entry cut and the reverse
/// acknowledgement prevents the source from setting the one-bit ready event
/// again before that consume has completed.  Both cuts are unconditional
/// within the loop invocation, so a guarded or zero-trip nested target cannot
/// unbalance the hardware token state.
std::optional<SyncCoverEventProtocol> makeDirectRecurrenceRoundTripProtocol(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    SyncCoverDemandId demandId, SyncCoverMechanismId mechanism,
    SyncCoverCoverageWorkBudget *workBudget) {
  if (demandId >= graph.getDemands().size()) {
    return std::nullopt;
  }
  const SyncCoverDemand &demand = graph.getDemands()[demandId];
  if (demand.distance != 1 || demand.scope == 0 ||
      demand.scope >= graph.getScopes().size() ||
      !graph.getScopes()[demand.scope].isLoop ||
      demand.source >= graph.getNodes().size() ||
      demand.target >= graph.getNodes().size()) {
    return std::nullopt;
  }
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &destination = graph.getNodes()[demand.target];
  if (source.resource == destination.resource) {
    return std::nullopt;
  }
  const std::optional<SyncCoverOrderingRequirementMask> readyRequirements =
      lifecycleEventRequirements(target, source.resource, destination.resource);
  const std::optional<SyncCoverOrderingRequirementMask> releaseRequirements =
      lifecycleEventRequirements(target, destination.resource, source.resource);
  if (!readyRequirements || !releaseRequirements ||
      (*readyRequirements & demand.orderingRequirements) !=
          demand.orderingRequirements) {
    return std::nullopt;
  }
  const SyncCoverRegionId loopRegion = graph.getScopes()[demand.scope].region;
  if (loopRegion >= graph.getRegions().size()) {
    return std::nullopt;
  }

  SyncCoverEventProtocol protocol;
  protocol.mechanism = mechanism;
  protocol.kind = SyncCoverEventProtocolKind::LifecycleNetwork;
  protocol.loop = SyncCoverProtocolLoopSchedule{
      demand.scope,
      graph.getRegions()[loopRegion].cardinality ==
          SyncCoverRegionCardinality::ZeroOrMore,
      std::nullopt,
      {0}};

  const SyncCoverAnchor entry{SyncCoverAnchorKind::ScopeEntry, 0, demand.scope,
                              0};
  const SyncCoverAnchor exit{SyncCoverAnchorKind::ScopeExit, 0, demand.scope,
                             0};
  const std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>>
      sourceAnchors =
          groupAnchors(graph, {source.id}, demand.scope, workBudget);
  const std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>>
      targetAnchors =
          groupAnchors(graph, {destination.id}, demand.scope, workBudget);
  if (!sourceAnchors || !targetAnchors) {
    return std::nullopt;
  }
  // The ready Wait and its acknowledgement must execute before both the
  // protected target and this iteration's source-side ready Set.  A loop may
  // contain no scheduled operation outside an opaque choice, so use the
  // graph-owned entry cuts of both endpoint groups rather than requiring an
  // unconditional node. groupAnchors lifts guarded endpoints to the enclosing
  // ControlEntry, which executes exactly once for every loop iteration.
  const std::optional<SyncCoverTimelinePosition> sourceEntry =
      resolveSyncCoverAnchor(graph, sourceAnchors->first);
  const std::optional<SyncCoverTimelinePosition> targetEntry =
      resolveSyncCoverAnchor(graph, targetAnchors->first);
  if (!sourceEntry || !targetEntry) {
    return std::nullopt;
  }
  const SyncCoverAnchor bodyEntry = *sourceEntry <= *targetEntry
                                        ? sourceAnchors->first
                                        : targetAnchors->first;
  const SyncCoverAnchor sourceCompletion = sourceAnchors->second;

  SyncCoverEventChannel ready;
  ready.id = 0;
  ready.flow = SyncCoverEventChannelFlow::LoopCarry;
  ready.width = 1;
  ready.distance = 1;
  ready.suppliedRequirements = *readyRequirements;
  appendLifecycleAction(
      ready, SyncCoverProtocolActionKind::Set,
      SyncCoverProtocolActionSegment::Entry,
      lifecyclePoint(SyncCoverCutPointKind::EventSet, source.resource, entry),
      0, SyncCoverProtocolActionGuard::LoopNonEmpty);
  const std::size_t readyWait =
      appendLifecycleAction(ready, SyncCoverProtocolActionKind::Wait,
                            SyncCoverProtocolActionSegment::Body,
                            lifecyclePoint(SyncCoverCutPointKind::EventWait,
                                           destination.resource, bodyEntry),
                            0, SyncCoverProtocolActionGuard::Always);
  const std::size_t readySet =
      appendLifecycleAction(ready, SyncCoverProtocolActionKind::Set,
                            SyncCoverProtocolActionSegment::Body,
                            lifecyclePoint(SyncCoverCutPointKind::EventSet,
                                           source.resource, sourceCompletion),
                            0, SyncCoverProtocolActionGuard::Always);
  appendLifecycleAction(ready, SyncCoverProtocolActionKind::Wait,
                        SyncCoverProtocolActionSegment::Exit,
                        lifecyclePoint(SyncCoverCutPointKind::EventWait,
                                       destination.resource, exit),
                        0, SyncCoverProtocolActionGuard::LoopNonEmpty);
  ready.supplies.push_back({readySet, readyWait, 1});
  ready.set = ready.actions[readySet].point;
  ready.wait = ready.actions[readyWait].point;

  SyncCoverEventChannel release;
  release.id = 1;
  release.flow = SyncCoverEventChannelFlow::SameIteration;
  release.width = 1;
  release.distance = 0;
  release.suppliedRequirements = *releaseRequirements;
  SyncCoverCutPoint releaseSetPoint = lifecyclePoint(
      SyncCoverCutPointKind::EventSet, destination.resource, bodyEntry);
  SyncCoverCutPoint releaseWaitPoint = lifecyclePoint(
      SyncCoverCutPointKind::EventWait, source.resource, bodyEntry);
  // The acknowledgement is issued after the ready Wait at the same lifted
  // body-entry boundary.  Its source-side Wait likewise follows any ready
  // priming Set at a coincident loop-entry boundary.  These explicit
  // ordinals are part of the token rearm proof; treating all four actions as
  // ordinal zero leaves the two same-pipeline orderings unspecified.
  releaseSetPoint.ordinal = 1;
  releaseWaitPoint.ordinal = 1;
  const std::size_t releaseSet = appendLifecycleAction(
      release, SyncCoverProtocolActionKind::Set,
      SyncCoverProtocolActionSegment::Body, std::move(releaseSetPoint), 0,
      SyncCoverProtocolActionGuard::Always);
  const std::size_t releaseWait = appendLifecycleAction(
      release, SyncCoverProtocolActionKind::Wait,
      SyncCoverProtocolActionSegment::Body, std::move(releaseWaitPoint), 0,
      SyncCoverProtocolActionGuard::Always);
  release.supplies.push_back({releaseSet, releaseWait, 0});
  release.set = release.actions[releaseSet].point;
  release.wait = release.actions[releaseWait].point;

  protocol.channels = {std::move(ready), std::move(release)};
  return protocol;
}

std::optional<SyncCoverEventProtocol> makeStructuralDirectProtocol(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    SyncCoverDemandId demandId, SyncCoverMechanismId mechanism,
    SyncCoverCoverageWorkBudget *workBudget) {
  if (demandId >= graph.getDemands().size()) {
    return std::nullopt;
  }
  const SyncCoverDemand &demand = graph.getDemands()[demandId];
  if (demand.distance != 0 || demand.source >= graph.getNodes().size() ||
      demand.target >= graph.getNodes().size()) {
    return std::nullopt;
  }
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &destination = graph.getNodes()[demand.target];
  if (source.resource == destination.resource ||
      source.guard.literals != destination.guard.literals) {
    return std::nullopt;
  }
  const std::optional<SyncCoverOrderingRequirementMask> requirements =
      lifecycleEventRequirements(target, source.resource, destination.resource);
  if (!requirements || (*requirements & demand.orderingRequirements) !=
                           demand.orderingRequirements) {
    return std::nullopt;
  }
  const std::optional<SyncCoverScopeId> sourceLoop =
      graph.getNearestEnclosingLoop(source.scope, true);
  const std::optional<SyncCoverScopeId> targetLoop =
      graph.getNearestEnclosingLoop(destination.scope, true);

  SyncCoverAnchor setAnchor;
  SyncCoverAnchor waitAnchor;
  if (targetLoop && !sourceLoop && source.scope != destination.scope &&
      graph.scopeContains(source.scope, destination.scope) &&
      graph.scopeContains(source.scope, *targetLoop)) {
    setAnchor = {SyncCoverAnchorKind::AfterNode, source.id, 0, 0};
    waitAnchor = {SyncCoverAnchorKind::ScopeEntry, 0, *targetLoop, 0};
  } else if (sourceLoop && !targetLoop && source.scope != destination.scope &&
             graph.scopeContains(destination.scope, source.scope) &&
             graph.scopeContains(destination.scope, *sourceLoop)) {
    setAnchor = {SyncCoverAnchorKind::ScopeExit, 0, *sourceLoop, 0};
    waitAnchor = {SyncCoverAnchorKind::BeforeNode, destination.id, 0, 0};
  } else if (sourceLoop && targetLoop && *sourceLoop != *targetLoop) {
    const std::optional<SyncCoverScopeId> sharedLoop =
        nearestSharedLifecycleLoop(graph, source.scope, destination.scope,
                                   workBudget);
    if (sharedLoop || (workBudget && workBudget->exhausted)) {
      return std::nullopt;
    }
    setAnchor = {SyncCoverAnchorKind::ScopeExit, 0, *sourceLoop, 0};
    waitAnchor = {SyncCoverAnchorKind::ScopeEntry, 0, *targetLoop, 0};
  } else {
    return std::nullopt;
  }
  if (source.order >= destination.order) {
    return std::nullopt;
  }

  SyncCoverEventProtocol protocol;
  protocol.mechanism = mechanism;
  protocol.kind = SyncCoverEventProtocolKind::SingleShot;

  SyncCoverEventChannel channel;
  channel.id = 0;
  channel.flow = SyncCoverEventChannelFlow::SingleShot;
  channel.width = 1;
  channel.suppliedRequirements = *requirements;
  channel.set = lifecyclePoint(SyncCoverCutPointKind::EventSet, source.resource,
                               setAnchor);
  channel.wait = lifecyclePoint(SyncCoverCutPointKind::EventWait,
                                destination.resource, waitAnchor);
  channel.transfers.push_back({0, channel.set, channel.wait, 0, 0, {}});
  protocol.channels.push_back(std::move(channel));
  return protocol;
}

struct SccCycleEdge {
  SyncCoverDemandId demand = 0;
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  SyncCoverStorageAccessId sourceAccess = 0;
  SyncCoverStorageAccessId targetAccess = 0;
  unsigned distance = 0;
};

struct WholeSccCycleProtocol {
  SyncCoverEventProtocol protocol;
  DerivedLifecycleSlot slot;
};

std::vector<WholeSccCycleProtocol> makeWholeSccCycleProtocols(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverLifecycleScc &component, SyncCoverProtocolLimits limits,
    SyncCoverCoverageWorkBudget *workBudget) {
  std::map<LifecycleSlotKey, std::vector<SccCycleEdge>> edgesBySlot;
  std::set<std::tuple<LifecycleSlotKey, SyncCoverDemandId>> seen;
  std::set<LifecycleSlotKey> oversizedSlots;
  std::map<LifecycleSlotKey, std::size_t> actionsBySlot;
  for (SyncCoverDemandId demandId : component.demands) {
    if (!consumeWork(workBudget) || demandId >= graph.getDemands().size()) {
      return {};
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (demand.distance > 1) {
      continue;
    }
    const SyncCoverNode &sourceNode = graph.getNodes()[demand.source];
    const SyncCoverNode &targetNode = graph.getNodes()[demand.target];
    if (sourceNode.resource == targetNode.resource) {
      continue;
    }
    for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
      if (!consumeWork(workBudget) ||
          witnessId >= graph.getStorageWitnesses().size()) {
        return {};
      }
      const SyncCoverStorageWitness &witness =
          graph.getStorageWitnesses()[witnessId];
      const SyncCoverStorageAccess &source =
          graph.getStorageAccesses()[witness.sourceAccess];
      const SyncCoverStorageAccess &targetAccess =
          graph.getStorageAccesses()[witness.targetAccess];
      const bool exactSlot =
          source.exactPhysical && targetAccess.exactPhysical &&
          source.path == SyncCoverStorageAccessPath::PhysicalPipeline &&
          targetAccess.path == SyncCoverStorageAccessPath::PhysicalPipeline &&
          source.domain == targetAccess.domain &&
          intervalEqual(source.extent, targetAccess.extent);
      if (!exactSlot) {
        continue;
      }
      const LifecycleSlotKey slot{source.domain, source.extent};
      if (!consumeWork(workBudget, lookupWork(seen.size() + 1))) {
        return {};
      }
      if (seen.emplace(slot, demandId).second) {
        if (!consumeWork(workBudget, lookupWork(oversizedSlots.size() + 1))) {
          return {};
        }
        if (oversizedSlots.count(slot) != 0) {
          continue;
        }
        if (!consumeWork(workBudget,
                         lookupWork(edgesBySlot.size() + 1) +
                             lookupWork(actionsBySlot.size() + 1))) {
          return {};
        }
        std::vector<SccCycleEdge> &slotEdges = edgesBySlot[slot];
        std::size_t &slotActions = actionsBySlot[slot];
        const std::size_t addedActions = demand.distance == 0 ? 2 : 4;
        if (slotEdges.size() >= limits.maximumChannels ||
            slotEdges.size() >= limits.maximumChannelLaneIncidences ||
            slotEdges.size() >= limits.maximumLifecycleTransfers ||
            addedActions >
                limits.maximumDynamicActions -
                    std::min(slotActions, limits.maximumDynamicActions)) {
          oversizedSlots.insert(slot);
          slotEdges.clear();
          actionsBySlot.erase(slot);
          continue;
        }
        slotEdges.push_back({demandId, demand.source, demand.target,
                             sourceNode.resource, targetNode.resource,
                             witness.sourceAccess, witness.targetAccess,
                             demand.distance});
        slotActions += addedActions;
      }
    }
  }

  std::vector<WholeSccCycleProtocol> protocols;
  std::size_t retainedChannels = 0;
  std::size_t retainedActions = 0;
  std::size_t retainedPhaseIncidences = 0;
  std::size_t retainedAccessIncidences = 0;
  for (const auto &[slot, edges] : edgesBySlot) {
    if (oversizedSlots.count(slot) != 0 || edges.size() < 3 ||
        protocols.size() >= limits.maximumProtocols ||
        protocols.size() >= limits.maximumLifecycleProtocols) {
      continue;
    }
    std::map<std::uint32_t, const SccCycleEdge *> outgoing;
    std::map<std::uint32_t, const SccCycleEdge *> incoming;
    bool uniqueCycle = true;
    unsigned totalDistance = 0;
    for (const SccCycleEdge &edge : edges) {
      if (!consumeWork(workBudget, lookupWork(outgoing.size() + 1) +
                                       lookupWork(incoming.size() + 1) + 1) ||
          !outgoing.emplace(edge.sourceResource, &edge).second ||
          !incoming.emplace(edge.targetResource, &edge).second) {
        uniqueCycle = false;
        break;
      }
      totalDistance += edge.distance;
    }
    if (!uniqueCycle || totalDistance == 0 || outgoing.size() != edges.size() ||
        incoming.size() != edges.size() || outgoing.size() != incoming.size()) {
      continue;
    }
    std::set<std::uint32_t> visited;
    std::uint32_t resource = outgoing.begin()->first;
    while (true) {
      if (!consumeWork(workBudget, lookupWork(visited.size() + 1))) {
        return {};
      }
      if (!visited.insert(resource).second) {
        break;
      }
      if (!consumeWork(workBudget, lookupWork(outgoing.size() + 1))) {
        return {};
      }
      const auto edge = outgoing.find(resource);
      if (edge == outgoing.end()) {
        uniqueCycle = false;
        break;
      }
      resource = edge->second->targetResource;
    }
    if (!uniqueCycle || resource != outgoing.begin()->first ||
        visited.size() != outgoing.size()) {
      continue;
    }

    const SyncCoverRegionId loopRegion =
        graph.getScopes()[component.loopScope].region;
    if (loopRegion >= graph.getRegions().size()) {
      continue;
    }
    SyncCoverProtocolLoopSchedule schedule;
    schedule.scope = component.loopScope;
    schedule.mayExecuteZeroTimes = graph.getRegions()[loopRegion].cardinality ==
                                   SyncCoverRegionCardinality::ZeroOrMore;
    const LoopPhaseControlQuery phaseQuery =
        findLoopPhaseControl(graph, component.loopScope, workBudget);
    if (phaseQuery.workUnavailable || phaseQuery.ambiguous) {
      return {};
    }
    schedule.phaseControl = phaseQuery.control;
    std::size_t phaseIncidences = 1;
    if (schedule.phaseControl) {
      const auto &relation =
          graph.getControls()[*schedule.phaseControl].phaseRelation;
      if (!relation) {
        continue;
      }
      phaseIncidences = relation->nextPhase.size();
      std::size_t protocolPhaseIncidences = 0;
      if (!checkedAccumulateSize(protocolPhaseIncidences, phaseIncidences) ||
          (edges.size() != 0 &&
           phaseIncidences >
               std::numeric_limits<std::size_t>::max() / edges.size()) ||
          !checkedAccumulateSize(protocolPhaseIncidences,
                                 phaseIncidences * edges.size())) {
        continue;
      }
      phaseIncidences = protocolPhaseIncidences;
      if (phaseIncidences > limits.maximumPhaseIncidences -
                                std::min(retainedPhaseIncidences,
                                         limits.maximumPhaseIncidences)) {
        continue;
      }
      schedule.laneByPhase.assign(relation->nextPhase.size(), 0);
    } else {
      if (edges.size() == std::numeric_limits<std::size_t>::max()) {
        continue;
      }
      phaseIncidences = edges.size() + 1;
      if (phaseIncidences > limits.maximumPhaseIncidences -
                                std::min(retainedPhaseIncidences,
                                         limits.maximumPhaseIncidences)) {
        continue;
      }
      schedule.laneByPhase = {0};
    }

    if (!consumeWork(workBudget, lookupWork(actionsBySlot.size() + 1))) {
      return {};
    }
    const std::size_t protocolActions = actionsBySlot.at(slot);
    if (edges.size() > limits.maximumChannelLaneIncidences -
                           std::min(retainedChannels,
                                    limits.maximumChannelLaneIncidences) ||
        protocolActions >
            limits.maximumTotalDynamicActions -
                std::min(retainedActions, limits.maximumTotalDynamicActions)) {
      continue;
    }

    SyncCoverEventProtocol protocol;
    protocol.mechanism = protocols.size();
    protocol.kind = SyncCoverEventProtocolKind::LifecycleNetwork;
    protocol.loop = schedule;
    bool valid = true;
    for (const SccCycleEdge &edge : edges) {
      const std::optional<SyncCoverOrderingRequirementMask> requirements =
          lifecycleEventRequirements(target, edge.sourceResource,
                                     edge.targetResource);
      if (!requirements) {
        valid = false;
        break;
      }
      SyncCoverEventChannel channel;
      channel.id = protocol.channels.size();
      channel.flow = edge.distance == 0
                         ? SyncCoverEventChannelFlow::SameIteration
                         : SyncCoverEventChannelFlow::LoopCarry;
      channel.width = 1;
      channel.distance = edge.distance;
      channel.suppliedRequirements = *requirements;
      const SyncCoverCutPoint set =
          lifecyclePoint(SyncCoverCutPointKind::EventSet, edge.sourceResource,
                         {SyncCoverAnchorKind::AfterNode, edge.source, 0, 0});
      const SyncCoverCutPoint wait =
          lifecyclePoint(SyncCoverCutPointKind::EventWait, edge.targetResource,
                         {SyncCoverAnchorKind::BeforeNode, edge.target, 0, 0});
      const std::optional<std::vector<std::size_t>> activePhases =
          activePhasesForTransfer(graph, schedule, set, wait, workBudget);
      if (!activePhases) {
        valid = false;
        break;
      }
      if (edge.distance != 0) {
        appendLifecycleAction(channel, SyncCoverProtocolActionKind::Set,
                              SyncCoverProtocolActionSegment::Entry,
                              lifecyclePoint(SyncCoverCutPointKind::EventSet,
                                             edge.sourceResource,
                                             {SyncCoverAnchorKind::ScopeEntry,
                                              0, component.loopScope, 0}),
                              0, SyncCoverProtocolActionGuard::LoopNonEmpty);
      }
      const std::size_t waitAction = appendLifecycleAction(
          channel, SyncCoverProtocolActionKind::Wait,
          SyncCoverProtocolActionSegment::Body, wait, 0,
          SyncCoverProtocolActionGuard::Always, *activePhases);
      const std::size_t setAction = appendLifecycleAction(
          channel, SyncCoverProtocolActionKind::Set,
          SyncCoverProtocolActionSegment::Body, set, 0,
          SyncCoverProtocolActionGuard::Always, *activePhases);
      if (edge.distance == 0) {
        std::swap(channel.actions[waitAction], channel.actions[setAction]);
        channel.actions[0].id = 0;
        channel.actions[1].id = 1;
        channel.supplies.push_back({0, 1, 0});
      } else {
        channel.supplies.push_back({setAction, waitAction, edge.distance});
        appendLifecycleAction(channel, SyncCoverProtocolActionKind::Wait,
                              SyncCoverProtocolActionSegment::Exit,
                              lifecyclePoint(SyncCoverCutPointKind::EventWait,
                                             edge.targetResource,
                                             {SyncCoverAnchorKind::ScopeExit, 0,
                                              component.loopScope, 0}),
                              0, SyncCoverProtocolActionGuard::LoopNonEmpty);
      }
      channel.set = set;
      channel.wait = wait;
      protocol.channels.push_back(std::move(channel));
    }
    if (!valid || protocol.channels.size() < 3) {
      continue;
    }
    const SyncCoverProtocolVerificationResult verified =
        verifySyncCoverEventProtocol(graph, target, protocol, limits,
                                     workBudget);
    if (!verified) {
      if (workBudget && workBudget->exhausted) {
        return {};
      }
      continue;
    }
    std::vector<SyncCoverStorageAccessId> accesses;
    if (edges.size() > std::numeric_limits<std::size_t>::max() / 2 ||
        edges.size() * 2 >
            limits.maximumLifecycleAccessIncidences -
                std::min(retainedAccessIncidences,
                         limits.maximumLifecycleAccessIncidences) ||
        !consumeWork(workBudget, edges.size() * 2)) {
      if (workBudget && workBudget->exhausted) {
        return {};
      }
      continue;
    }
    accesses.reserve(edges.size() * 2);
    for (const SccCycleEdge &edge : edges) {
      accesses.push_back(edge.sourceAccess);
      accesses.push_back(edge.targetAccess);
    }
    const std::optional<std::size_t> accessSortWork =
        comparisonWork(accesses.size());
    if (!accessSortWork || !consumeWork(workBudget, *accessSortWork)) {
      return {};
    }
    std::sort(accesses.begin(), accesses.end());
    accesses.erase(std::unique(accesses.begin(), accesses.end()),
                   accesses.end());
    if (accesses.size() >
        limits.maximumLifecycleAccessIncidences -
            std::min(retainedAccessIncidences,
                     limits.maximumLifecycleAccessIncidences)) {
      continue;
    }
    protocol.mechanism = protocols.size();
    retainedChannels += edges.size();
    retainedActions += protocolActions;
    retainedPhaseIncidences += phaseIncidences;
    retainedAccessIncidences += accesses.size();
    protocols.push_back(
        {std::move(protocol), {slot.domain, slot.extent, std::move(accesses)}});
  }
  return protocols;
}

bool witnessMatchesLifecycleSlot(const SyncCoverGraph &graph,
                                 SyncCoverStorageWitnessId witnessId,
                                 const DerivedLifecycleSlot &slot,
                                 SyncCoverDemandKind demandKind) {
  if (witnessId >= graph.getStorageWitnesses().size()) {
    return false;
  }
  const SyncCoverStorageWitness &witness =
      graph.getStorageWitnesses()[witnessId];
  if (witness.sourceAccess >= graph.getStorageAccesses().size() ||
      witness.targetAccess >= graph.getStorageAccesses().size()) {
    return false;
  }
  const SyncCoverStorageAccess &source =
      graph.getStorageAccesses()[witness.sourceAccess];
  const SyncCoverStorageAccess &target =
      graph.getStorageAccesses()[witness.targetAccess];
  bool modesMatch = false;
  switch (demandKind) {
  case SyncCoverDemandKind::SSA:
    return false;
  case SyncCoverDemandKind::MemoryRAW:
    modesMatch = syncCoverStorageModeWrites(source.mode) &&
                 syncCoverStorageModeReads(target.mode);
    break;
  case SyncCoverDemandKind::MemoryWAR:
    modesMatch = syncCoverStorageModeReads(source.mode) &&
                 syncCoverStorageModeWrites(target.mode);
    break;
  case SyncCoverDemandKind::MemoryWAW:
    modesMatch = syncCoverStorageModeWrites(source.mode) &&
                 syncCoverStorageModeWrites(target.mode);
    break;
  case SyncCoverDemandKind::HardwareAccRAR:
    modesMatch = syncCoverStorageModeReads(source.mode) &&
                 syncCoverStorageModeReads(target.mode);
    break;
  }
  return source.exactPhysical && target.exactPhysical && modesMatch &&
         source.domain == slot.domain && target.domain == slot.domain &&
         intervalEqual(source.extent, slot.extent) &&
         intervalEqual(target.extent, slot.extent);
}

bool loopMayExecuteZeroTimes(const SyncCoverGraph &graph,
                             SyncCoverScopeId loop) {
  if (loop >= graph.getScopes().size()) {
    return true;
  }
  const SyncCoverRegionId region = graph.getScopes()[loop].region;
  return region >= graph.getRegions().size() ||
         graph.getRegions()[region].cardinality ==
             SyncCoverRegionCardinality::ZeroOrMore;
}

std::optional<std::size_t>
findInitialLifecyclePath(const SyncCoverGraph &graph,
                         const DerivedLifecycleCertificate &certificate,
                         const SyncCoverProtocolLoopSchedule &schedule) {
  if (certificate.paths.size() == 1) {
    return 0;
  }
  if (!schedule.phaseControl ||
      *schedule.phaseControl >= graph.getControls().size()) {
    return std::nullopt;
  }
  const std::optional<SyncCoverControlPhaseRelation> &relation =
      graph.getControls()[*schedule.phaseControl].phaseRelation;
  if (!relation ||
      relation->initialPhase >= relation->activeAlternative.size()) {
    return std::nullopt;
  }
  const unsigned initialAlternative =
      relation->activeAlternative[relation->initialPhase];
  std::optional<std::size_t> result;
  for (std::size_t path = 0; path < certificate.paths.size(); ++path) {
    const SyncCoverScopeId scope = certificate.paths[path].scope;
    if (scope >= graph.getScopes().size() ||
        !guardHasLiteral(graph.getScopes()[scope].guard, *schedule.phaseControl,
                         initialAlternative)) {
      continue;
    }
    if (result) {
      return std::nullopt;
    }
    result = path;
  }
  return result;
}

bool certificateContainsSlotWitness(
    const SyncCoverGraph &graph, const DerivedLifecycleCertificate &certificate,
    SyncCoverStorageWitnessId witness) {
  for (const DerivedLifecycleLane &lane : certificate.lanes) {
    for (const DerivedLifecycleSlot &slot : lane.slots) {
      if (witnessMatchesLifecycleSlot(graph, witness, slot,
                                      SyncCoverDemandKind::MemoryWAR)) {
        return true;
      }
    }
  }
  return false;
}

std::optional<SyncCoverGuard>
guardAtLifecycleAnchor(const SyncCoverGraph &graph,
                       const SyncCoverAnchor &anchor) {
  switch (anchor.kind) {
  case SyncCoverAnchorKind::BeforeNode:
  case SyncCoverAnchorKind::AfterNode:
    if (anchor.node >= graph.getNodes().size()) {
      return std::nullopt;
    }
    return graph.getNodes()[anchor.node].guard;
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit:
  case SyncCoverAnchorKind::LoopBodyEntry:
  case SyncCoverAnchorKind::LoopBodyExit:
  case SyncCoverAnchorKind::ControlEntry:
  case SyncCoverAnchorKind::ControlExit:
    if (anchor.scope >= graph.getScopes().size()) {
      return std::nullopt;
    }
    return graph.getScopes()[anchor.scope].guard;
  case SyncCoverAnchorKind::TimelinePoint:
    return SyncCoverGuard{};
  }
  return std::nullopt;
}

bool demandEndpointExecutesAction(
    const SyncCoverGraph &graph, const DerivedLifecycleCertificate &certificate,
    std::optional<SyncCoverControlId> phaseControl,
    const SyncCoverGuard &endpointGuard, const SyncCoverProtocolAction &action,
    bool allowEarlierPhase, SyncCoverCoverageWorkBudget *workBudget) {
  const std::optional<SyncCoverGuard> actionGuard =
      guardAtLifecycleAnchor(graph, action.point.anchor);
  if (!actionGuard) {
    return false;
  }
  std::size_t guardWork = 0;
  if (endpointGuard.literals.size() >
      std::numeric_limits<std::size_t>::max() - actionGuard->literals.size()) {
    return false;
  }
  guardWork = endpointGuard.literals.size() + actionGuard->literals.size();
  if (!consumeWork(workBudget, guardWork)) {
    return false;
  }
  const bool directGuardImplication =
      syncCoverGuardImplies(endpointGuard, *actionGuard);
  bool earlierPhaseExecutes = false;
  if (allowEarlierPhase && phaseControl &&
      *phaseControl < graph.getControls().size()) {
    const SyncCoverControl &control = graph.getControls()[*phaseControl];
    const auto endpointLiteral = std::find_if(
        endpointGuard.literals.begin(), endpointGuard.literals.end(),
        [&](const SyncCoverGuardLiteral &candidate) {
          return candidate.control == *phaseControl;
        });
    bool nonPhaseGuardImplied = true;
    for (const SyncCoverGuardLiteral &required : actionGuard->literals) {
      if (!consumeWork(workBudget)) {
        return false;
      }
      if (required.control != *phaseControl &&
          !std::binary_search(endpointGuard.literals.begin(),
                              endpointGuard.literals.end(), required)) {
        nonPhaseGuardImplied = false;
        break;
      }
    }
    if (control.phaseRelation && nonPhaseGuardImplied &&
        endpointLiteral != endpointGuard.literals.end()) {
      const SyncCoverControlPhaseRelation &relation = *control.phaseRelation;
      std::vector<bool> visited(relation.nextPhase.size());
      std::size_t phase = relation.initialPhase;
      bool actionOccurred = false;
      while (phase < relation.nextPhase.size() && !visited[phase]) {
        if (!consumeWork(workBudget)) {
          return false;
        }
        visited[phase] = true;
        const bool actionActive =
            action.activePhases.empty() ||
            std::binary_search(action.activePhases.begin(),
                               action.activePhases.end(), phase);
        actionOccurred = actionOccurred || actionActive;
        if (phase < relation.activeAlternative.size() &&
            relation.activeAlternative[phase] == endpointLiteral->alternative) {
          earlierPhaseExecutes = actionOccurred;
          break;
        }
        phase = relation.nextPhase[phase];
      }
    }
  }
  if (!directGuardImplication && !earlierPhaseExecutes) {
    return false;
  }
  if (!action.activePhases.empty() && !earlierPhaseExecutes) {
    if (!phaseControl || *phaseControl >= graph.getControls().size()) {
      return false;
    }
    const SyncCoverControl &control = graph.getControls()[*phaseControl];
    if (!control.phaseRelation) {
      return false;
    }
    const auto literal = std::find_if(
        endpointGuard.literals.begin(), endpointGuard.literals.end(),
        [&](const SyncCoverGuardLiteral &candidate) {
          return candidate.control == *phaseControl;
        });
    if (literal == endpointGuard.literals.end()) {
      return false;
    }
    bool phaseMatches = false;
    for (std::size_t phase : action.activePhases) {
      if (!consumeWork(workBudget)) {
        return false;
      }
      if (phase < control.phaseRelation->activeAlternative.size() &&
          control.phaseRelation->activeAlternative[phase] ==
              literal->alternative) {
        phaseMatches = true;
        break;
      }
    }
    if (!phaseMatches) {
      return false;
    }
  }
  switch (action.guard) {
  case SyncCoverProtocolActionGuard::Always:
    return true;
  case SyncCoverProtocolActionGuard::HasSuccessor:
    if (earlierPhaseExecutes) {
      return true;
    }
    return std::any_of(
        endpointGuard.literals.begin(), endpointGuard.literals.end(),
        [&](const SyncCoverGuardLiteral &literal) {
          if (!consumeWork(workBudget) ||
              literal.control >= graph.getControls().size()) {
            return false;
          }
          const auto &relation =
              graph.getControls()[literal.control].successorRelation;
          return relation && relation->loopScope == certificate.loopScope &&
                 relation->hasSuccessorAlternative == literal.alternative;
        });
  case SyncCoverProtocolActionGuard::LoopNonEmpty:
  case SyncCoverProtocolActionGuard::LoopEmpty:
  case SyncCoverProtocolActionGuard::FirstIteration:
  case SyncCoverProtocolActionGuard::NotFirstIteration:
    // Parent-distance summaries currently use only ordinary body cuts.  Entry,
    // exit, and iteration-special actions need their own dynamic proof.
    return false;
  }
  return false;
}

bool demandEndpointsStraddleActions(
    const SyncCoverGraph &graph, const DerivedLifecycleCertificate &certificate,
    std::optional<SyncCoverControlId> phaseControl,
    const SyncCoverDemand &demand, const SyncCoverProtocolAction &setAction,
    const SyncCoverProtocolAction &waitAction,
    SyncCoverCoverageWorkBudget *workBudget) {
  const std::optional<SyncCoverTimelinePosition> sourcePosition =
      resolveSyncCoverAnchor(graph,
                             {SyncCoverAnchorKind::AfterNode, demand.source,
                              graph.getNodes()[demand.source].scope, 0});
  const std::optional<SyncCoverTimelinePosition> setPosition =
      resolveSyncCoverAnchor(graph, setAction.point.anchor);
  const std::optional<SyncCoverTimelinePosition> waitPosition =
      resolveSyncCoverAnchor(graph, waitAction.point.anchor);
  const std::optional<SyncCoverTimelinePosition> targetPosition =
      resolveSyncCoverAnchor(graph,
                             {SyncCoverAnchorKind::BeforeNode, demand.target,
                              graph.getNodes()[demand.target].scope, 0});
  return sourcePosition && setPosition && waitPosition && targetPosition &&
         *sourcePosition <= *setPosition && *waitPosition <= *targetPosition &&
         demandEndpointExecutesAction(graph, certificate, phaseControl,
                                      demand.sourceGuard, setAction, false,
                                      workBudget) &&
         demandEndpointExecutesAction(graph, certificate, phaseControl,
                                      demand.targetGuard, waitAction, true,
                                      workBudget);
}

/// Returns whether the frozen demand graph contains at least one physical
/// parent-loop obligation compatible with this prospective completion export.
/// This is only an admission witness for generating the summary: no demand ID
/// is stored in the protocol and exact-world grounding remains the proof.
std::optional<bool> hasParentLifecycleDemandWitness(
    const SyncCoverGraph &graph, const DerivedLifecycleCertificate &certificate,
    const std::vector<SyncCoverNodeId> &sourceNodes, std::size_t lane,
    SyncCoverScopeId parentScope,
    std::optional<SyncCoverControlId> phaseControl,
    SyncCoverDemandKind demandKind, std::uint32_t sourceResource,
    std::uint32_t targetResource, const SyncCoverProtocolAction &setAction,
    const SyncCoverProtocolAction &waitAction, bool targetMayBeOutsideChild,
    SyncCoverCoverageWorkBudget *workBudget) {
  if (lane >= certificate.lanes.size()) {
    return std::nullopt;
  }
  std::vector<SyncCoverNodeId> sortedSources(sourceNodes.begin(),
                                             sourceNodes.end());
  const std::optional<std::size_t> sourceSortWork =
      comparisonWork(sortedSources.size());
  if (!sourceSortWork || !consumeWork(workBudget, *sourceSortWork)) {
    return std::nullopt;
  }
  std::sort(sortedSources.begin(), sortedSources.end());
  sortedSources.erase(std::unique(sortedSources.begin(), sortedSources.end()),
                      sortedSources.end());
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const std::optional<bool> targetInScope = scopeContainsWithWork(
        graph, targetMayBeOutsideChild ? parentScope : certificate.loopScope,
        graph.getNodes()[demand.target].scope, workBudget);
    if (!targetInScope) {
      return std::nullopt;
    }
    if (demand.scope != parentScope || demand.distance != 1 ||
        !containsKind(demand, demandKind) ||
        !std::binary_search(sortedSources.begin(), sortedSources.end(),
                            demand.source) ||
        graph.getNodes()[demand.source].resource != sourceResource ||
        graph.getNodes()[demand.target].resource != targetResource ||
        !*targetInScope ||
        !demandEndpointsStraddleActions(graph, certificate, phaseControl,
                                        demand, setAction, waitAction,
                                        workBudget)) {
      continue;
    }
    bool matchesLane = false;
    for (SyncCoverStorageWitnessId witness : demand.storageWitnesses) {
      if (!consumeWork(workBudget)) {
        return std::nullopt;
      }
      for (const DerivedLifecycleSlot &slot : certificate.lanes[lane].slots) {
        if (!consumeWork(workBudget)) {
          return std::nullopt;
        }
        if (witnessMatchesLifecycleSlot(graph, witness, slot, demandKind)) {
          matchesLane = true;
          break;
        }
      }
      if (matchesLane) {
        break;
      }
    }
    if (matchesLane) {
      return true;
    }
  }
  return false;
}

std::optional<SyncCoverScopeId>
findRoundTripLifetimeScope(const SyncCoverGraph &graph,
                           const DerivedLifecycleCertificate &certificate,
                           const SyncCoverProtocolLoopSchedule &schedule,
                           SyncCoverCoverageWorkBudget *workBudget) {
  if (certificate.loopScope == 0 ||
      certificate.loopScope >= graph.getScopes().size()) {
    return std::nullopt;
  }
  const SyncCoverScopeId parent =
      graph.getScopes()[certificate.loopScope].parent;
  const std::optional<std::size_t> initialPath =
      findInitialLifecyclePath(graph, certificate, schedule);
  if (parent == 0 || parent >= graph.getScopes().size() ||
      !graph.getScopes()[parent].isLoop || !initialPath) {
    return std::nullopt;
  }
  std::vector<SyncCoverNodeId> sources;
  std::vector<SyncCoverNodeId> targets;
  for (const DerivedLifecyclePath &path : certificate.paths) {
    for (const DerivedLifecycleUse &use : path.uses) {
      sources.insert(sources.end(), use.consumers.begin(), use.consumers.end());
    }
  }
  for (const DerivedLifecycleUse &use : certificate.paths[*initialPath].uses) {
    targets.insert(targets.end(), use.producers.begin(), use.producers.end());
  }
  const std::optional<std::size_t> sourceSortWork =
      comparisonWork(sources.size());
  const std::optional<std::size_t> targetSortWork =
      comparisonWork(targets.size());
  if (!sourceSortWork || !targetSortWork ||
      !consumeWork(workBudget, *sourceSortWork + *targetSortWork)) {
    return std::nullopt;
  }
  std::sort(sources.begin(), sources.end());
  sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
  std::sort(targets.begin(), targets.end());
  targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
  for (const SyncCoverDemand &demand : graph.getDemands()) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    if (demand.scope != parent || demand.distance != 1 ||
        !containsKind(demand, SyncCoverDemandKind::MemoryWAR) ||
        !std::binary_search(sources.begin(), sources.end(), demand.source) ||
        !std::binary_search(targets.begin(), targets.end(), demand.target)) {
      continue;
    }
    for (SyncCoverStorageWitnessId witness : demand.storageWitnesses) {
      if (!consumeWork(workBudget)) {
        return std::nullopt;
      }
      if (certificateContainsSlotWitness(graph, certificate, witness)) {
        return parent;
      }
    }
  }
  return std::nullopt;
}

std::optional<SyncCoverEventProtocol> makeAlternatingLifecycleProtocol(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const DerivedLifecycleCertificate &certificate,
    SyncCoverMechanismId mechanism, SyncCoverProtocolLoopSchedule schedule,
    SyncCoverCoverageWorkBudget *workBudget) {
  if (certificate.lanes.size() < 2 || certificate.paths.size() < 2 ||
      certificate.lanes.size() != certificate.paths.size() ||
      certificate.initiallyFreeLanes.empty()) {
    return std::nullopt;
  }
  for (const DerivedLifecyclePath &path : certificate.paths) {
    if (path.uses.size() != 1) {
      return std::nullopt;
    }
  }

  const std::optional<SyncCoverOrderingRequirementMask> readyRequirements =
      lifecycleEventRequirements(target, certificate.producerResource,
                                 certificate.consumerResource);
  const std::optional<SyncCoverOrderingRequirementMask> releaseRequirements =
      lifecycleEventRequirements(target, certificate.consumerResource,
                                 certificate.producerResource);
  if (!readyRequirements || !releaseRequirements) {
    return std::nullopt;
  }
  SyncCoverEventProtocol protocol;
  protocol.mechanism = mechanism;
  protocol.kind = SyncCoverEventProtocolKind::LifecycleNetwork;
  protocol.loop = std::move(schedule);
  // The physical recipe is local to each child invocation: loop-nonempty
  // guards prime it and child-exit waits drain its release channel.  The
  // verifier still expands two parent invocations when an exact outer reuse
  // witness exists, so completion exports are summaries of the local recipe
  // rather than extra boundary actions.
  const std::optional<SyncCoverScopeId> lifetimeScope =
      findRoundTripLifetimeScope(graph, certificate, *protocol.loop,
                                 workBudget);
  if (workBudget && workBudget->exhausted) {
    return std::nullopt;
  }
  if (lifetimeScope) {
    protocol.lifetimeScope = lifetimeScope;
    protocol.lifetimeMayExecuteZeroTimes =
        loopMayExecuteZeroTimes(graph, *lifetimeScope);
  }

  SyncCoverEventChannel ready;
  ready.id = 0;
  ready.flow = SyncCoverEventChannelFlow::LoopCarry;
  ready.width = certificate.lanes.size();
  ready.distance = 1;
  ready.suppliedRequirements = *readyRequirements;
  const std::size_t initialReadySet = appendLifecycleAction(
      ready, SyncCoverProtocolActionKind::Set,
      SyncCoverProtocolActionSegment::Entry,
      lifecyclePoint(SyncCoverCutPointKind::EventSet,
                     certificate.producerResource,
                     certificate.initialReadyAnchor),
      certificate.initialReadyLane, SyncCoverProtocolActionGuard::LoopNonEmpty);
  std::vector<std::size_t> readyWaits(certificate.paths.size());
  std::vector<std::size_t> readySets(certificate.paths.size());
  for (std::size_t pathIndex = 0; pathIndex < certificate.paths.size();
       ++pathIndex) {
    const DerivedLifecycleUse &use = certificate.paths[pathIndex].uses.front();
    const SyncCoverCutPoint wait =
        lifecyclePoint(SyncCoverCutPointKind::EventWait,
                       certificate.consumerResource, use.readAcquireAnchor);
    const SyncCoverCutPoint set =
        lifecyclePoint(SyncCoverCutPointKind::EventSet,
                       certificate.producerResource, use.readyAnchor);
    const SyncCoverCutPoint phaseProbe = lifecyclePoint(
        SyncCoverCutPointKind::EventWait, certificate.consumerResource,
        {SyncCoverAnchorKind::ScopeEntry, 0, certificate.paths[pathIndex].scope,
         0});
    const std::optional<std::vector<std::size_t>> waitPhases =
        activePhasesForPoint(graph, *protocol.loop, phaseProbe, workBudget);
    const std::optional<std::vector<std::size_t>> setPhases =
        activePhasesForPoint(graph, *protocol.loop, phaseProbe, workBudget);
    if (!waitPhases || !setPhases) {
      return std::nullopt;
    }
    readyWaits[pathIndex] = appendLifecycleAction(
        ready, SyncCoverProtocolActionKind::Wait,
        SyncCoverProtocolActionSegment::Body, wait, use.lane,
        SyncCoverProtocolActionGuard::Always, *waitPhases);
    readySets[pathIndex] = appendLifecycleAction(
        ready, SyncCoverProtocolActionKind::Set,
        SyncCoverProtocolActionSegment::Body, set, use.producerLane,
        SyncCoverProtocolActionGuard::HasSuccessor, *setPhases);
  }
  bool initialReadySupplied = false;
  for (std::size_t pathIndex = 0; pathIndex < certificate.paths.size();
       ++pathIndex) {
    const DerivedLifecycleUse &use = certificate.paths[pathIndex].uses.front();
    if (use.lane == certificate.initialReadyLane) {
      ready.supplies.push_back({initialReadySet, readyWaits[pathIndex], 0});
      initialReadySupplied = true;
    }
    for (std::size_t targetIndex = 0; targetIndex < certificate.paths.size();
         ++targetIndex) {
      const DerivedLifecycleUse &target =
          certificate.paths[targetIndex].uses.front();
      if (target.lane == use.producerLane) {
        ready.supplies.push_back(
            {readySets[pathIndex], readyWaits[targetIndex], 1});
      }
    }
  }
  if (lifetimeScope) {
    for (std::size_t sourceIndex = 0; sourceIndex < certificate.paths.size();
         ++sourceIndex) {
      const DerivedLifecycleUse &source =
          certificate.paths[sourceIndex].uses.front();
      for (std::size_t targetIndex = 0; targetIndex < certificate.paths.size();
           ++targetIndex) {
        const DerivedLifecycleUse &targetUse =
            certificate.paths[targetIndex].uses.front();
        if (targetUse.lane != source.producerLane) {
          continue;
        }
        const std::optional<bool> parentWitness =
            hasParentLifecycleDemandWitness(
                graph, certificate, source.producers, source.producerLane,
                *lifetimeScope, protocol.loop->phaseControl,
                SyncCoverDemandKind::MemoryRAW, certificate.producerResource,
                certificate.consumerResource,
                ready.actions[readySets[sourceIndex]],
                ready.actions[readyWaits[targetIndex]], true, workBudget);
        if (!parentWitness) {
          return std::nullopt;
        }
        if (*parentWitness) {
          ready.supplies.push_back(
              {readySets[sourceIndex], readyWaits[targetIndex], 1,
               *lifetimeScope, SyncCoverProtocolSupplyKind::CompletionExport});
        }
      }
    }
  }
  if (!initialReadySupplied) {
    return std::nullopt;
  }
  if (lifetimeScope) {
    for (std::size_t pathIndex = 0; pathIndex < certificate.paths.size();
         ++pathIndex) {
      const DerivedLifecycleUse &use =
          certificate.paths[pathIndex].uses.front();
      if (use.lane == certificate.initialReadyLane) {
        // The entry Set is a physical source-prefix cut for initial producers
        // outside the inner schedule. It orders those producers with the
        // first matching inner Wait in the same owning-loop invocation. The
        // release channel supplies the positive-distance return edge.
        ready.supplies.push_back(
            {initialReadySet, readyWaits[pathIndex], 0, *lifetimeScope,
             SyncCoverProtocolSupplyKind::CompletionExport});
      }
    }
  }
  ready.set = ready.actions[initialReadySet].point;
  ready.wait = ready.actions[readyWaits.front()].point;

  SyncCoverEventChannel release;
  release.id = 1;
  release.flow = SyncCoverEventChannelFlow::LoopCarry;
  release.width = certificate.lanes.size();
  release.distance = 1;
  release.suppliedRequirements = *releaseRequirements;
  std::vector<std::size_t> initialReleaseLanes = certificate.initiallyFreeLanes;
  const std::optional<std::size_t> initialPath =
      findInitialLifecyclePath(graph, certificate, *protocol.loop);
  if (!initialPath) {
    return std::nullopt;
  }
  const DerivedLifecycleUse &initialUse =
      certificate.paths[*initialPath].uses.front();
  SyncCoverCutPoint releasePrime = lifecyclePoint(
      SyncCoverCutPointKind::EventSet, certificate.consumerResource,
      initialUse.readAcquireAnchor);
  releasePrime.ordinal = 1;
  const std::optional<std::vector<std::size_t>> releasePrimePhases =
      activePhasesForPoint(graph, *protocol.loop, releasePrime, workBudget);
  if (!releasePrimePhases) {
    return std::nullopt;
  }
  for (std::size_t lane : initialReleaseLanes) {
    if (lane >= release.width) {
      return std::nullopt;
    }
    releasePrime.ordinal = lane + 1;
    appendLifecycleAction(release, SyncCoverProtocolActionKind::Set,
                          SyncCoverProtocolActionSegment::Body, releasePrime,
                          lane, SyncCoverProtocolActionGuard::FirstIteration,
                          *releasePrimePhases);
  }
  std::vector<std::size_t> releaseSets(certificate.paths.size());
  std::vector<std::size_t> releaseWaits(certificate.paths.size());
  for (std::size_t pathIndex = 0; pathIndex < certificate.paths.size();
       ++pathIndex) {
    const DerivedLifecycleUse &use = certificate.paths[pathIndex].uses.front();
    const SyncCoverCutPoint set =
        lifecyclePoint(SyncCoverCutPointKind::EventSet,
                       certificate.consumerResource, use.releaseAnchor);
    const SyncCoverCutPoint wait =
        lifecyclePoint(SyncCoverCutPointKind::EventWait,
                       certificate.producerResource, use.writeAcquireAnchor);
    const SyncCoverCutPoint phaseProbe = lifecyclePoint(
        SyncCoverCutPointKind::EventWait, certificate.consumerResource,
        {SyncCoverAnchorKind::ScopeEntry, 0, certificate.paths[pathIndex].scope,
         0});
    const std::optional<std::vector<std::size_t>> setPhases =
        activePhasesForPoint(graph, *protocol.loop, phaseProbe, workBudget);
    const std::optional<std::vector<std::size_t>> waitPhases =
        activePhasesForPoint(graph, *protocol.loop, phaseProbe, workBudget);
    if (!setPhases || !waitPhases) {
      return std::nullopt;
    }
    releaseSets[pathIndex] = appendLifecycleAction(
        release, SyncCoverProtocolActionKind::Set,
        SyncCoverProtocolActionSegment::Body, set, use.lane,
        SyncCoverProtocolActionGuard::Always, *setPhases);
    releaseWaits[pathIndex] = appendLifecycleAction(
        release, SyncCoverProtocolActionKind::Wait,
        SyncCoverProtocolActionSegment::Body, wait, use.producerLane,
        SyncCoverProtocolActionGuard::HasSuccessor, *waitPhases);
  }
  for (std::size_t sourceIndex = 0; sourceIndex < certificate.paths.size();
       ++sourceIndex) {
    const DerivedLifecycleUse &source =
        certificate.paths[sourceIndex].uses.front();
    for (std::size_t targetIndex = 0; targetIndex < certificate.paths.size();
         ++targetIndex) {
      const DerivedLifecycleUse &target =
          certificate.paths[targetIndex].uses.front();
      if (target.producerLane == source.lane) {
        if (!protocol.loop->phaseControl) {
          return std::nullopt;
        }
        const std::optional<unsigned> distance = phaseDistanceBetweenPaths(
            graph, *protocol.loop->phaseControl,
            certificate.paths[sourceIndex].scope,
            certificate.paths[targetIndex].scope, workBudget);
        if (!distance || *distance == 0) {
          return std::nullopt;
        }
        release.distance = std::max(release.distance, *distance);
        release.supplies.push_back(
            {releaseSets[sourceIndex], releaseWaits[targetIndex], *distance});
      }
    }
  }
  std::vector<std::size_t> releaseExitWaits(release.width);
  for (std::size_t lane = 0; lane < release.width; ++lane) {
    releaseExitWaits[lane] = appendLifecycleAction(
        release, SyncCoverProtocolActionKind::Wait,
        SyncCoverProtocolActionSegment::Exit,
        lifecyclePoint(
            SyncCoverCutPointKind::EventWait, certificate.producerResource,
            {SyncCoverAnchorKind::ScopeExit, 0, certificate.loopScope, 0}),
        lane, SyncCoverProtocolActionGuard::LoopNonEmpty);
  }
  if (lifetimeScope) {
    // The final child iteration has no body successor and therefore no body
    // release Wait. Its per-lane exit drain consumes the last release token
    // and blocks the producer pipe before the enclosing-loop reuse. Bind the
    // parent completion summary to that physical drain, not to a body Wait
    // guarded by HasSuccessor.
    for (std::size_t sourceIndex = 0; sourceIndex < certificate.paths.size();
         ++sourceIndex) {
      const DerivedLifecycleUse &source =
          certificate.paths[sourceIndex].uses.front();
      for (std::size_t targetIndex = 0; targetIndex < certificate.paths.size();
           ++targetIndex) {
        const DerivedLifecycleUse &target =
            certificate.paths[targetIndex].uses.front();
        if (target.producerLane != source.lane) {
          continue;
        }
        const std::optional<bool> parentWitness =
            hasParentLifecycleDemandWitness(
                graph, certificate, source.consumers, source.lane,
                *lifetimeScope, protocol.loop->phaseControl,
                SyncCoverDemandKind::MemoryWAR, certificate.consumerResource,
                certificate.producerResource,
                release.actions[releaseSets[sourceIndex]],
                release.actions[releaseWaits[targetIndex]], true, workBudget);
        if (!parentWitness) {
          return std::nullopt;
        }
        if (!*parentWitness) {
          continue;
        }
        if (source.lane >= releaseExitWaits.size()) {
          return std::nullopt;
        }
        release.supplies.push_back(
            {releaseSets[sourceIndex], releaseExitWaits[source.lane], 0,
             *lifetimeScope, SyncCoverProtocolSupplyKind::CompletionExport});
        break;
      }
    }
  }
  if (ready.supplies.empty() || release.supplies.empty()) {
    return std::nullopt;
  }
  if (lifetimeScope) {
    if (!appendParentInvocationCompletionExports(ready, *lifetimeScope,
                                                 workBudget) ||
        !appendParentInvocationCompletionExports(release, *lifetimeScope,
                                                 workBudget)) {
      return std::nullopt;
    }
  }
  release.set = release.actions[releaseSets.front()].point;
  release.wait = release.actions[releaseWaits.front()].point;
  protocol.channels = {std::move(ready), std::move(release)};
  return protocol;
}

std::optional<SyncCoverEventProtocol> makeLifecycleProtocol(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const DerivedLifecycleCertificate &certificate,
    SyncCoverMechanismId mechanism, SyncCoverCoverageWorkBudget *workBudget) {
  const auto rejectProtocol = [](const char *reason) {
    (void)reason;
    return std::optional<SyncCoverEventProtocol>{};
  };
  if (certificate.loopScope == 0 ||
      certificate.loopScope >= graph.getScopes().size() ||
      certificate.lanes.empty() || certificate.paths.empty()) {
    return rejectProtocol("header");
  }
  const SyncCoverRegionId loopRegion =
      graph.getScopes()[certificate.loopScope].region;
  if (loopRegion >= graph.getRegions().size()) {
    return rejectProtocol("loop-region");
  }
  SyncCoverProtocolLoopSchedule schedule;
  schedule.scope = certificate.loopScope;
  schedule.mayExecuteZeroTimes = graph.getRegions()[loopRegion].cardinality ==
                                 SyncCoverRegionCardinality::ZeroOrMore;
  schedule.phaseControl = certificate.periodicControl;
  if (!schedule.phaseControl) {
    const LoopPhaseControlQuery phaseQuery =
        findLoopPhaseControl(graph, certificate.loopScope, workBudget);
    if (phaseQuery.workUnavailable || phaseQuery.ambiguous) {
      return rejectProtocol("phase-query");
    }
    schedule.phaseControl = phaseQuery.control;
  }
  if (schedule.phaseControl) {
    const SyncCoverControl &control =
        graph.getControls()[*schedule.phaseControl];
    if (!control.phaseRelation) {
      return rejectProtocol("phase-relation");
    }
    schedule.laneByPhase.assign(control.phaseRelation->nextPhase.size(), 0);
  } else {
    schedule.laneByPhase = {0};
  }

  SyncCoverEventProtocol protocol;
  if (certificate.alternating) {
    std::optional<SyncCoverEventProtocol> alternating =
        makeAlternatingLifecycleProtocol(graph, target, certificate, mechanism,
                                         std::move(schedule), workBudget);
    return alternating ? std::move(alternating)
                       : rejectProtocol("alternating-construction");
  }
  protocol.mechanism = mechanism;
  protocol.kind = SyncCoverEventProtocolKind::LifecycleNetwork;
  protocol.loop = schedule;
  const bool ownsStorageLifecycle = std::any_of(
      certificate.lanes.begin(), certificate.lanes.end(),
      [](const DerivedLifecycleLane &lane) { return !lane.slots.empty(); });
  const std::optional<SyncCoverScopeId> lifetimeScope =
      ownsStorageLifecycle
          ? findRoundTripLifetimeScope(graph, certificate, schedule, workBudget)
          : std::nullopt;
  if (workBudget && workBudget->exhausted) {
    return std::nullopt;
  }
  if (lifetimeScope) {
    protocol.lifetimeScope = lifetimeScope;
    protocol.lifetimeMayExecuteZeroTimes =
        loopMayExecuteZeroTimes(graph, *lifetimeScope);
  }
  SyncCoverEventChannel ready;
  ready.id = 0;
  ready.flow = SyncCoverEventChannelFlow::SameIteration;
  ready.width = certificate.lanes.size();
  const std::optional<SyncCoverOrderingRequirementMask> readyRequirements =
      lifecycleEventRequirements(target, certificate.producerResource,
                                 certificate.consumerResource);
  const std::optional<SyncCoverOrderingRequirementMask> releaseRequirements =
      lifecycleEventRequirements(target, certificate.consumerResource,
                                 certificate.producerResource);
  if (!readyRequirements || !releaseRequirements) {
    return rejectProtocol("event-requirements");
  }
  ready.suppliedRequirements = *readyRequirements;
  SyncCoverEventChannel release;
  release.id = 1;
  release.flow = SyncCoverEventChannelFlow::LoopCarry;
  release.width = certificate.lanes.size();
  release.distance = 1;
  release.suppliedRequirements = *releaseRequirements;
  release.exportsCompletionAtExit = false;
  for (std::size_t lane = 0; lane < release.width; ++lane) {
    appendLifecycleAction(
        release, SyncCoverProtocolActionKind::Set,
        SyncCoverProtocolActionSegment::Entry,
        lifecyclePoint(SyncCoverCutPointKind::EventSet,
                       certificate.consumerResource,
                       {SyncCoverAnchorKind::ScopeEntry, 0,
                        lifetimeScope.value_or(certificate.loopScope), 0}),
        lane,
        lifetimeScope ? SyncCoverProtocolActionGuard::Always
                      : SyncCoverProtocolActionGuard::LoopNonEmpty);
  }
  struct ReleaseUse {
    std::size_t path = 0;
    std::size_t setAction = 0;
    std::size_t waitAction = 0;
    std::size_t lane = 0;
    std::size_t producerLane = 0;
    SyncCoverTimelinePosition setPosition = 0;
    SyncCoverTimelinePosition waitPosition = 0;
    std::vector<std::size_t> activePhases;
    SyncCoverProtocolActionGuard temporalGuard =
        SyncCoverProtocolActionGuard::Always;
  };
  struct ReadyUse {
    std::size_t path = 0;
    std::size_t setAction = 0;
    std::size_t waitAction = 0;
    std::size_t lane = 0;
    std::vector<SyncCoverNodeId> producers;
  };
  std::vector<ReadyUse> readyUses;
  std::vector<ReleaseUse> releaseUses;
  for (std::size_t pathIndex = 0; pathIndex < certificate.paths.size();
       ++pathIndex) {
    const DerivedLifecyclePath &path = certificate.paths[pathIndex];
    for (const DerivedLifecycleUse &use : path.uses) {
      if (use.lane >= certificate.lanes.size() ||
          use.producerLane >= certificate.lanes.size()) {
        return std::nullopt;
      }
      SyncCoverEventTransfer readyTransfer;
      readyTransfer.id = ready.transfers.size();
      readyTransfer.set =
          lifecyclePoint(SyncCoverCutPointKind::EventSet,
                         certificate.producerResource, use.readyAnchor);
      readyTransfer.wait =
          lifecyclePoint(SyncCoverCutPointKind::EventWait,
                         certificate.consumerResource, use.readAcquireAnchor);
      readyTransfer.setLane = use.lane;
      readyTransfer.waitLane = use.lane;
      const std::optional<std::vector<std::size_t>> readyPhases =
          activePhasesForTransfer(graph, schedule, readyTransfer.set,
                                  readyTransfer.wait, workBudget);
      if (!readyPhases) {
        return rejectProtocol("ready-phases");
      }
      const std::optional<SyncCoverProtocolActionGuard> readyTemporalGuard =
          temporalGuardForPoint(graph, schedule, readyTransfer.set, workBudget);
      const std::optional<SyncCoverProtocolActionGuard> readyWaitTemporalGuard =
          temporalGuardForPoint(graph, schedule, readyTransfer.wait,
                                workBudget);
      if (!readyTemporalGuard || !readyWaitTemporalGuard ||
          *readyTemporalGuard != *readyWaitTemporalGuard) {
        return rejectProtocol("ready-temporal-guard");
      }
      readyTransfer.activePhases = *readyPhases;
      const std::size_t readySetAction = appendLifecycleAction(
          ready, SyncCoverProtocolActionKind::Set,
          SyncCoverProtocolActionSegment::Body, readyTransfer.set, use.lane,
          *readyTemporalGuard, *readyPhases);
      const std::size_t readyWaitAction = appendLifecycleAction(
          ready, SyncCoverProtocolActionKind::Wait,
          SyncCoverProtocolActionSegment::Body, readyTransfer.wait, use.lane,
          *readyTemporalGuard, *readyPhases);
      ready.supplies.push_back({readySetAction, readyWaitAction, 0});
      readyUses.push_back({pathIndex, readySetAction, readyWaitAction, use.lane,
                           use.producers});
      SyncCoverEventTransfer releaseTransfer;
      releaseTransfer.id = release.transfers.size();
      releaseTransfer.set =
          lifecyclePoint(SyncCoverCutPointKind::EventSet,
                         certificate.consumerResource, use.releaseAnchor);
      releaseTransfer.wait =
          lifecyclePoint(SyncCoverCutPointKind::EventWait,
                         certificate.producerResource, use.writeAcquireAnchor);
      releaseTransfer.setLane = use.lane;
      releaseTransfer.waitLane = use.producerLane;
      const std::optional<std::vector<std::size_t>> releasePhases =
          activePhasesForTransfer(graph, schedule, releaseTransfer.set,
                                  releaseTransfer.wait, workBudget);
      if (!releasePhases) {
        return rejectProtocol("release-phases");
      }
      const std::optional<SyncCoverProtocolActionGuard> releaseTemporalGuard =
          temporalGuardForPoint(graph, schedule, releaseTransfer.set,
                                workBudget);
      const std::optional<SyncCoverProtocolActionGuard>
          releaseWaitTemporalGuard = temporalGuardForPoint(
              graph, schedule, releaseTransfer.wait, workBudget);
      if (!releaseTemporalGuard || !releaseWaitTemporalGuard ||
          *releaseTemporalGuard != *releaseWaitTemporalGuard) {
        return rejectProtocol("release-temporal-guard");
      }
      releaseTransfer.activePhases = *releasePhases;
      const std::optional<SyncCoverTimelinePosition> setPosition =
          resolveSyncCoverAnchor(graph, releaseTransfer.set.anchor);
      const std::optional<SyncCoverTimelinePosition> waitPosition =
          resolveSyncCoverAnchor(graph, releaseTransfer.wait.anchor);
      if (!setPosition || !waitPosition) {
        return rejectProtocol("release-positions");
      }
      const std::size_t waitAction = appendLifecycleAction(
          release, SyncCoverProtocolActionKind::Wait,
          SyncCoverProtocolActionSegment::Body, releaseTransfer.wait,
          use.producerLane, *releaseTemporalGuard, *releasePhases);
      const std::size_t setAction = appendLifecycleAction(
          release, SyncCoverProtocolActionKind::Set,
          SyncCoverProtocolActionSegment::Body, releaseTransfer.set, use.lane,
          *releaseTemporalGuard, *releasePhases);
      releaseUses.push_back({pathIndex, setAction, waitAction, use.lane,
                             use.producerLane, *setPosition, *waitPosition,
                             *releasePhases, *releaseTemporalGuard});
    }
  }
  if (lifetimeScope) {
    for (const ReadyUse &source : readyUses) {
      for (const ReadyUse &targetUse : readyUses) {
        if (targetUse.lane != source.lane) {
          continue;
        }
        const std::optional<bool> parentWitness =
            hasParentLifecycleDemandWitness(
                graph, certificate, source.producers, source.lane,
                *lifetimeScope, protocol.loop->phaseControl,
                SyncCoverDemandKind::MemoryRAW, certificate.producerResource,
                certificate.consumerResource, ready.actions[source.setAction],
                ready.actions[targetUse.waitAction], true, workBudget);
        if (!parentWitness) {
          return std::nullopt;
        }
        if (*parentWitness) {
          ready.supplies.push_back(
              {source.setAction, targetUse.waitAction, 1, *lifetimeScope,
               SyncCoverProtocolSupplyKind::CompletionExport});
        }
      }
    }
  }
  std::vector<std::size_t> releaseExitWaits(release.width);
  for (std::size_t lane = 0; lane < release.width; ++lane) {
    releaseExitWaits[lane] = appendLifecycleAction(
        release, SyncCoverProtocolActionKind::Wait,
        SyncCoverProtocolActionSegment::Exit,
        lifecyclePoint(SyncCoverCutPointKind::EventWait,
                       certificate.producerResource,
                       {SyncCoverAnchorKind::ScopeExit, 0,
                        lifetimeScope.value_or(certificate.loopScope), 0}),
        lane,
        lifetimeScope ? SyncCoverProtocolActionGuard::Always
                      : SyncCoverProtocolActionGuard::LoopNonEmpty);
  }
  const auto pathDistance =
      [&](const ReleaseUse &source,
          const ReleaseUse &target) -> std::optional<unsigned> {
    if (!schedule.phaseControl) {
      return source.path == target.path ? std::optional<unsigned>(1)
                                        : std::nullopt;
    }
    const SyncCoverControlPhaseRelation &relation =
        *graph.getControls()[*schedule.phaseControl].phaseRelation;
    if (source.activePhases.empty() || target.activePhases.empty()) {
      return std::nullopt;
    }
    std::optional<unsigned> uniformDistance;
    for (std::size_t sourcePhase : source.activePhases) {
      if (sourcePhase >= relation.nextPhase.size()) {
        return std::nullopt;
      }
      std::size_t phase = sourcePhase;
      std::optional<unsigned> nextDistance;
      for (unsigned distance = 1; distance <= relation.nextPhase.size();
           ++distance) {
        if (!consumeWork(workBudget) || phase >= relation.nextPhase.size()) {
          return std::nullopt;
        }
        phase = relation.nextPhase[phase];
        if (phase >= relation.nextPhase.size()) {
          return std::nullopt;
        }
        if (std::binary_search(target.activePhases.begin(),
                               target.activePhases.end(), phase)) {
          nextDistance = distance;
          break;
        }
      }
      if (!nextDistance ||
          (uniformDistance && *uniformDistance != *nextDistance)) {
        return std::nullopt;
      }
      uniformDistance = nextDistance;
    }
    return uniformDistance;
  };
  for (const ReleaseUse &source : releaseUses) {
    if (source.temporalGuard == SyncCoverProtocolActionGuard::FirstIteration) {
      if (source.lane >= releaseExitWaits.size()) {
        return rejectProtocol("release-exit-lane");
      }
      // This guarded occurrence executes at most once in the loop. Its body
      // Set is consumed by the explicit loop-exit drain; there is no later
      // body Wait to which a positive recurrence distance could be assigned.
      release.supplies.push_back(
          {source.setAction, releaseExitWaits[source.lane], 0});
      continue;
    }
    const ReleaseUse *best = nullptr;
    unsigned bestDistance = 0;
    for (const ReleaseUse &target : releaseUses) {
      if (target.producerLane != source.lane) {
        continue;
      }
      const bool sameIteration = source.path == target.path &&
                                 source.setPosition < target.waitPosition;
      const std::optional<unsigned> nextDistance =
          sameIteration ? std::optional<unsigned>(0)
                        : pathDistance(source, target);
      if (!nextDistance) {
        continue;
      }
      const unsigned distance = *nextDistance;
      if (!best || std::make_tuple(distance, target.waitPosition) <
                       std::make_tuple(bestDistance, best->waitPosition)) {
        best = &target;
        bestDistance = distance;
      }
    }
    if (!best) {
      return rejectProtocol("release-successor");
    }
    release.distance = std::max(release.distance, bestDistance);
    release.supplies.push_back(
        {source.setAction, best->waitAction, bestDistance});
  }
  if (lifetimeScope) {
    const std::size_t localSupplyCount = release.supplies.size();
    if (localSupplyCount != releaseUses.size()) {
      return std::nullopt;
    }
    for (std::size_t sourceIndex = 0; sourceIndex < releaseUses.size();
         ++sourceIndex) {
      const ReleaseUse &source = releaseUses[sourceIndex];
      std::vector<SyncCoverNodeId> sourceNodes;
      for (const DerivedLifecycleUse &use :
           certificate.paths[source.path].uses) {
        if (use.lane == source.lane) {
          sourceNodes.insert(sourceNodes.end(), use.consumers.begin(),
                             use.consumers.end());
        }
      }
      const SyncCoverProtocolSupply &localSupply =
          release.supplies[sourceIndex];
      const std::optional<bool> parentWitness = hasParentLifecycleDemandWitness(
          graph, certificate, sourceNodes, source.lane, *lifetimeScope,
          protocol.loop->phaseControl, SyncCoverDemandKind::MemoryWAR,
          certificate.consumerResource, certificate.producerResource,
          release.actions[localSupply.setAction],
          release.actions[localSupply.waitAction], true, workBudget);
      if (!parentWitness) {
        return std::nullopt;
      }
      if (*parentWitness) {
        release.supplies.push_back(
            {localSupply.setAction, localSupply.waitAction, 1, *lifetimeScope,
             SyncCoverProtocolSupplyKind::CompletionExport});
      }
    }
    const std::optional<std::size_t> initialPath =
        findInitialLifecyclePath(graph, certificate, schedule);
    if (!initialPath) {
      return rejectProtocol("initial-path");
    }
    for (std::size_t sourcePath = 0; sourcePath < certificate.paths.size();
         ++sourcePath) {
      for (std::size_t lane = 0; lane < release.width; ++lane) {
        const ReleaseUse *source = nullptr;
        const ReleaseUse *target = nullptr;
        for (const ReleaseUse &use : releaseUses) {
          if (use.path == sourcePath && use.lane == lane &&
              (!source || source->setPosition < use.setPosition)) {
            source = &use;
          }
          if (use.path == *initialPath && use.producerLane == lane &&
              (!target || use.waitPosition < target->waitPosition)) {
            target = &use;
          }
        }
        if (!source || !target) {
          return std::nullopt;
        }
        std::vector<SyncCoverNodeId> sourceNodes;
        for (const DerivedLifecycleUse &use :
             certificate.paths[sourcePath].uses) {
          if (use.lane == lane) {
            sourceNodes.insert(sourceNodes.end(), use.consumers.begin(),
                               use.consumers.end());
          }
        }
        const std::optional<bool> parentWitness =
            hasParentLifecycleDemandWitness(
                graph, certificate, sourceNodes, lane, *lifetimeScope,
                protocol.loop->phaseControl, SyncCoverDemandKind::MemoryWAR,
                certificate.consumerResource, certificate.producerResource,
                release.actions[source->setAction],
                release.actions[target->waitAction], true, workBudget);
        if (!parentWitness || !*parentWitness) {
          continue;
        }
        release.supplies.push_back(
            {source->setAction, target->waitAction, 1, *lifetimeScope,
             SyncCoverProtocolSupplyKind::CompletionExport});
      }
    }
  }
  if (ready.actions.empty() || ready.supplies.empty() ||
      release.actions.empty() || release.supplies.empty()) {
    return rejectProtocol("empty-recipe");
  }
  if (lifetimeScope) {
    if (!appendParentInvocationCompletionExports(ready, *lifetimeScope,
                                                 workBudget) ||
        !appendParentInvocationCompletionExports(release, *lifetimeScope,
                                                 workBudget)) {
      return std::nullopt;
    }
  }
  ready.set = ready.actions[readyUses.front().setAction].point;
  ready.wait = ready.actions[readyUses.front().waitAction].point;
  release.set = release.actions[releaseUses.front().setAction].point;
  release.wait = release.actions[releaseUses.front().waitAction].point;
  const auto sameAnchor = [](const SyncCoverAnchor &left,
                             const SyncCoverAnchor &right) {
    return std::tie(left.kind, left.node, left.scope, left.position) ==
           std::tie(right.kind, right.node, right.scope, right.position);
  };
  bool releaseMustPrecedeReady = false;
  bool readyMustPrecedeRelease = false;
  for (const ReleaseUse &releaseUse : releaseUses) {
    for (const ReadyUse &readyUse : readyUses) {
      releaseMustPrecedeReady |=
          sameAnchor(release.actions[releaseUse.waitAction].point.anchor,
                     ready.actions[readyUse.setAction].point.anchor);
      readyMustPrecedeRelease |=
          sameAnchor(ready.actions[readyUse.waitAction].point.anchor,
                     release.actions[releaseUse.setAction].point.anchor);
    }
  }
  if (releaseMustPrecedeReady && readyMustPrecedeRelease) {
    return rejectProtocol("coincident-action-cycle");
  }
  if (releaseMustPrecedeReady) {
    release.id = 0;
    ready.id = 1;
    protocol.channels = {std::move(release), std::move(ready)};
  } else {
    ready.id = 0;
    release.id = 1;
    protocol.channels = {std::move(ready), std::move(release)};
  }
  return protocol;
}

SyncCoverProtocolError
addProposalSlot(SyncCoverLifecycleProposal &proposal,
                std::map<LifecycleSlotKey, std::size_t> &slotIndex,
                const DerivedLifecycleSlot &slot,
                SyncCoverProtocolLimits limits,
                SyncCoverCoverageWorkBudget *workBudget) {
  if (!consumeWork(workBudget, lookupWork(slotIndex.size() + 1))) {
    return SyncCoverProtocolError::WorkLimitExceeded;
  }
  const LifecycleSlotKey key{slot.domain, slot.extent};
  const auto existing = slotIndex.find(key);
  if (existing != slotIndex.end()) {
    std::vector<SyncCoverStorageAccessId> merged =
        proposal.slots[existing->second].accesses;
    if (slot.accesses.size() >
            limits.maximumLifecycleAccessIncidences -
                std::min(merged.size(),
                         limits.maximumLifecycleAccessIncidences) ||
        !consumeWork(workBudget, slot.accesses.size())) {
      return workBudget && workBudget->exhausted
                 ? SyncCoverProtocolError::WorkLimitExceeded
                 : SyncCoverProtocolError::LimitExceeded;
    }
    merged.insert(merged.end(), slot.accesses.begin(), slot.accesses.end());
    const std::optional<std::size_t> sortWork = comparisonWork(merged.size());
    if (!sortWork || !consumeWork(workBudget, *sortWork)) {
      return SyncCoverProtocolError::WorkLimitExceeded;
    }
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    proposal.slots[existing->second].accesses = std::move(merged);
    return SyncCoverProtocolError::None;
  }
  if (proposal.slots.size() == limits.maximumLifecycleSlots) {
    return SyncCoverProtocolError::LimitExceeded;
  }
  if (slot.accesses.size() > limits.maximumLifecycleAccessIncidences ||
      !consumeWork(workBudget, slot.accesses.size())) {
    return workBudget && workBudget->exhausted
               ? SyncCoverProtocolError::WorkLimitExceeded
               : SyncCoverProtocolError::LimitExceeded;
  }
  slotIndex.emplace(key, proposal.slots.size());
  proposal.slots.push_back({slot.domain, slot.extent, slot.accesses});
  return SyncCoverProtocolError::None;
}

struct LifecycleCatalogFootprint {
  std::size_t protocols = 0;
  std::size_t lanes = 0;
  std::size_t slots = 0;
  std::size_t transfers = 0;
  std::size_t nodeReferences = 0;
  std::size_t accessIncidences = 0;
  std::size_t payloadIncidences = 0;
};

std::optional<LifecycleCatalogFootprint>
getLifecycleCatalogFootprint(const SyncCoverLifecycleProposal &proposal) {
  LifecycleCatalogFootprint result;
  result.protocols = proposal.protocols.size();
  result.slots = proposal.slots.size();
  const auto add = [&](std::size_t &field, std::size_t amount) {
    return checkedAccumulateSize(field, amount) &&
           checkedAccumulateSize(result.payloadIncidences, amount);
  };
  if (!add(result.nodeReferences, proposal.seedDemands.size()) ||
      !checkedAccumulateSize(result.payloadIncidences,
                             proposal.exactCoverage.getWords().size()) ||
      !checkedAccumulateSize(result.payloadIncidences,
                             proposal.protocols.size()) ||
      !checkedAccumulateSize(result.payloadIncidences, proposal.slots.size())) {
    return std::nullopt;
  }
  for (const SyncCoverLifecycleSlot &slot : proposal.slots) {
    if (!add(result.accessIncidences, slot.accesses.size())) {
      return std::nullopt;
    }
  }
  for (const SyncCoverEventProtocol &protocol : proposal.protocols) {
    if (!add(result.nodeReferences, protocol.rearmProofs.size()) ||
        !checkedAccumulateSize(result.payloadIncidences,
                               protocol.channels.size()) ||
        !checkedAccumulateSize(result.payloadIncidences,
                               protocol.loop ? protocol.loop->laneByPhase.size()
                                             : 0)) {
      return std::nullopt;
    }
    for (const SyncCoverEventChannel &channel : protocol.channels) {
      if (!add(result.lanes, channel.width) ||
          !add(result.transfers, channel.transfers.size()) ||
          !add(result.transfers, channel.actions.size()) ||
          !add(result.transfers, channel.supplies.size()) ||
          !checkedAccumulateSize(result.payloadIncidences,
                                 channel.activePhases.size()) ||
          !checkedAccumulateSize(result.payloadIncidences,
                                 channel.set.guard.literals.size()) ||
          !checkedAccumulateSize(result.payloadIncidences,
                                 channel.wait.guard.literals.size())) {
        return std::nullopt;
      }
      for (const SyncCoverEventTransfer &transfer : channel.transfers) {
        if (!checkedAccumulateSize(result.payloadIncidences,
                                   transfer.activePhases.size()) ||
            !checkedAccumulateSize(result.payloadIncidences,
                                   transfer.set.guard.literals.size()) ||
            !checkedAccumulateSize(result.payloadIncidences,
                                   transfer.wait.guard.literals.size())) {
          return std::nullopt;
        }
      }
      for (const SyncCoverProtocolAction &action : channel.actions) {
        if (!checkedAccumulateSize(result.payloadIncidences,
                                   action.activePhases.size()) ||
            !checkedAccumulateSize(result.payloadIncidences,
                                   action.point.guard.literals.size())) {
          return std::nullopt;
        }
        if (action.point.anchor.kind == SyncCoverAnchorKind::BeforeNode ||
            action.point.anchor.kind == SyncCoverAnchorKind::AfterNode) {
          if (!add(result.nodeReferences, 1)) {
            return std::nullopt;
          }
        }
      }
    }
  }
  return result;
}

SyncCoverProtocolError verifyAndPruneLifecycleCompletionExports(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    SyncCoverEventProtocol &protocol, SyncCoverProtocolLimits limits,
    SyncCoverCoverageWorkBudget *workBudget,
    std::optional<std::size_t> *invalidIndex = nullptr) {
  SyncCoverProtocolVerificationResult verified =
      verifySyncCoverEventProtocol(graph, target, protocol, limits, workBudget);
  if (verified) {
    return SyncCoverProtocolError::None;
  }
  if (verified.error != SyncCoverProtocolError::InvalidTokenLifecycle ||
      verified.supplyWitnesses.size() != protocol.channels.size()) {
    if (invalidIndex) {
      *invalidIndex = verified.invalidIndex;
    }
    return verified.error;
  }
  for (std::size_t channelIndex = 0; channelIndex < protocol.channels.size();
       ++channelIndex) {
    SyncCoverEventChannel &channel = protocol.channels[channelIndex];
    if (verified.supplyWitnesses[channelIndex].size() !=
        channel.supplies.size()) {
      if (invalidIndex) {
        *invalidIndex = channelIndex;
      }
      return SyncCoverProtocolError::InvalidTokenLifecycle;
    }
    std::vector<SyncCoverProtocolSupply> retained;
    retained.reserve(channel.supplies.size());
    for (std::size_t supplyIndex = 0; supplyIndex < channel.supplies.size();
         ++supplyIndex) {
      const SyncCoverProtocolSupply &supply = channel.supplies[supplyIndex];
      if (supply.kind == SyncCoverProtocolSupplyKind::TokenPair ||
          verified.supplyWitnesses[channelIndex][supplyIndex]) {
        retained.push_back(supply);
      }
    }
    channel.supplies = std::move(retained);
  }
  verified =
      verifySyncCoverEventProtocol(graph, target, protocol, limits, workBudget);
  if (!verified && invalidIndex) {
    *invalidIndex = verified.invalidIndex;
  }
  return verified.error;
}

std::optional<SyncCoverDemandSet>
certifyLifecycleStorageReuse(const SyncCoverGraph &graph,
                             const SyncCoverProtocolTargetContract &target,
                             const DerivedLifecycleCertificate &certificate,
                             const SyncCoverEventProtocol &protocol,
                             SyncCoverCoverageWorkBudget *workBudget) {
  SyncCoverDemandSet covered(graph.getDemands().size());
  const auto rejectStorage = [&](const char *reason) {
    (void)reason;
    return std::optional<SyncCoverDemandSet>{};
  };
  if (!target.directEventCompletesSourcePrefix) {
    return covered;
  }
  std::vector<std::optional<std::size_t>> accessLane(
      graph.getStorageAccesses().size());
  std::vector<std::vector<SyncCoverNodeId>> producers(certificate.lanes.size());
  std::vector<std::vector<SyncCoverNodeId>> consumers(certificate.lanes.size());
  std::vector<std::vector<std::size_t>> producerRegions(
      graph.getNodes().size());
  for (const DerivedLifecycleLane &lane : certificate.lanes) {
    if (lane.id >= producers.size()) {
      return rejectStorage("lane-id");
    }
    for (const DerivedLifecycleSlot &slot : lane.slots) {
      for (SyncCoverStorageAccessId access : slot.accesses) {
        if (!consumeWork(workBudget) || access >= accessLane.size() ||
            (accessLane[access] && *accessLane[access] != lane.id)) {
          return rejectStorage("access-lane");
        }
        accessLane[access] = lane.id;
      }
    }
  }
  std::size_t nextProducerRegion = 0;
  for (const DerivedLifecyclePath &path : certificate.paths) {
    for (const DerivedLifecycleUse &use : path.uses) {
      if (use.producerLane >= producers.size()) {
        return rejectStorage("producer-lane");
      }
      if (!consumeWork(workBudget, use.producers.size())) {
        return std::nullopt;
      }
      producers[use.producerLane].insert(producers[use.producerLane].end(),
                                         use.producers.begin(),
                                         use.producers.end());
      if (use.lane >= consumers.size() ||
          !consumeWork(workBudget, use.consumers.size())) {
        return rejectStorage("consumer-lane");
      }
      consumers[use.lane].insert(consumers[use.lane].end(),
                                 use.consumers.begin(), use.consumers.end());
      for (SyncCoverNodeId producer : use.producers) {
        if (producer >= producerRegions.size()) {
          return rejectStorage("producer-node");
        }
        producerRegions[producer].push_back(nextProducerRegion);
      }
      ++nextProducerRegion;
    }
  }
  if (certificate.initialReadyLane >= producers.size() ||
      !consumeWork(workBudget, certificate.initialProducers.size())) {
    return rejectStorage("initial-ready-lane");
  }
  producers[certificate.initialReadyLane].insert(
      producers[certificate.initialReadyLane].end(),
      certificate.initialProducers.begin(), certificate.initialProducers.end());
  for (SyncCoverNodeId producer : certificate.initialProducers) {
    if (producer >= producerRegions.size()) {
      return rejectStorage("initial-producer-node");
    }
    producerRegions[producer].push_back(nextProducerRegion);
  }
  const auto normalizeNodes = [&](std::vector<SyncCoverNodeId> &nodes) {
    const std::optional<std::size_t> work = comparisonWork(nodes.size());
    if (!work || !consumeWork(workBudget, *work)) {
      return false;
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return true;
  };
  for (std::size_t lane = 0; lane < producers.size(); ++lane) {
    if (!normalizeNodes(producers[lane]) || !normalizeNodes(consumers[lane])) {
      return std::nullopt;
    }
  }
  for (std::vector<std::size_t> &regions : producerRegions) {
    const std::optional<std::size_t> work = comparisonWork(regions.size());
    if (!work || !consumeWork(workBudget, *work)) {
      return std::nullopt;
    }
    std::sort(regions.begin(), regions.end());
    regions.erase(std::unique(regions.begin(), regions.end()), regions.end());
  }
  const auto producersShareRegion =
      [&](SyncCoverNodeId first,
          SyncCoverNodeId second) -> std::optional<bool> {
    if (first >= producerRegions.size() || second >= producerRegions.size()) {
      return std::nullopt;
    }
    std::size_t left = 0;
    std::size_t right = 0;
    while (left < producerRegions[first].size() &&
           right < producerRegions[second].size()) {
      if (!consumeWork(workBudget)) {
        return std::nullopt;
      }
      if (producerRegions[first][left] == producerRegions[second][right]) {
        return true;
      }
      if (producerRegions[first][left] < producerRegions[second][right]) {
        ++left;
      } else {
        ++right;
      }
    }
    return false;
  };

  const SyncCoverEventChannel *readyChannel = nullptr;
  const SyncCoverEventChannel *releaseChannel = nullptr;
  for (const SyncCoverEventChannel &channel : protocol.channels) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    if (channel.set.resource == certificate.producerResource &&
        channel.wait.resource == certificate.consumerResource) {
      if (readyChannel) {
        return rejectStorage("duplicate-ready-channel");
      }
      readyChannel = &channel;
    }
    if (channel.set.resource == certificate.consumerResource &&
        channel.wait.resource == certificate.producerResource) {
      if (releaseChannel) {
        return rejectStorage("duplicate-release-channel");
      }
      releaseChannel = &channel;
    }
  }
  if (!readyChannel || !releaseChannel ||
      readyChannel->width < certificate.lanes.size() ||
      releaseChannel->width < certificate.lanes.size()) {
    return covered;
  }
  std::vector<bool> drainedReleaseLanes(releaseChannel->width);
  for (const SyncCoverProtocolAction &action : releaseChannel->actions) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    const bool childOrLifetimeExit =
        action.point.anchor.kind == SyncCoverAnchorKind::ScopeExit &&
        (action.point.anchor.scope == certificate.loopScope ||
         (protocol.lifetimeScope &&
          action.point.anchor.scope == *protocol.lifetimeScope));
    const bool activeOnNonempty =
        action.guard == SyncCoverProtocolActionGuard::Always ||
        action.guard == SyncCoverProtocolActionGuard::LoopNonEmpty;
    if (action.kind == SyncCoverProtocolActionKind::Wait &&
        action.segment == SyncCoverProtocolActionSegment::Exit &&
        childOrLifetimeExit && activeOnNonempty &&
        action.lane < drainedReleaseLanes.size()) {
      drainedReleaseLanes[action.lane] = true;
    }
  }
  if (!consumeWork(workBudget, drainedReleaseLanes.size())) {
    return std::nullopt;
  }
  if (std::find(drainedReleaseLanes.begin(), drainedReleaseLanes.end(),
                false) != drainedReleaseLanes.end()) {
    return covered;
  }
  const auto containsNode = [&](const std::vector<SyncCoverNodeId> &nodes,
                                SyncCoverNodeId node) -> std::optional<bool> {
    if (!consumeWork(workBudget, lookupWork(nodes.size()))) {
      return std::nullopt;
    }
    return std::binary_search(nodes.begin(), nodes.end(), node);
  };

  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const bool writeReuseKind =
        containsKind(demand, SyncCoverDemandKind::MemoryWAW);
    bool sharedProducerRegion = false;
    if (writeReuseKind && demand.distance == 0) {
      const std::optional<bool> shared =
          producersShareRegion(demand.source, demand.target);
      if (!shared) {
        return std::nullopt;
      }
      sharedProducerRegion = *shared;
    }
    const bool writeReuse =
        writeReuseKind && (demand.distance != 0 || !sharedProducerRegion);
    const bool readRelease =
        containsKind(demand, SyncCoverDemandKind::MemoryWAR);
    const bool localScope = demand.scope == certificate.loopScope;
    const bool lifetimeScope =
        protocol.lifetimeScope && demand.scope == *protocol.lifetimeScope;
    if ((!writeReuse && !readRelease) || (!localScope && !lifetimeScope) ||
        demand.storageWitnesses.empty()) {
      continue;
    }
    const bool releaseRequirementsSupplied =
        (releaseChannel->suppliedRequirements & demand.orderingRequirements) ==
        demand.orderingRequirements;
    const bool readyRequirementsSupplied =
        (readyChannel->suppliedRequirements & demand.orderingRequirements) ==
        demand.orderingRequirements;
    const bool requirementsSupplied =
        releaseRequirementsSupplied &&
        (readRelease || readyRequirementsSupplied);
    if (!requirementsSupplied) {
      continue;
    }
    bool valid = true;
    for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
      if (!consumeWork(workBudget) ||
          witnessId >= graph.getStorageWitnesses().size()) {
        return std::nullopt;
      }
      const SyncCoverStorageWitness &witness =
          graph.getStorageWitnesses()[witnessId];
      if (witness.sourceAccess >= accessLane.size() ||
          witness.targetAccess >= accessLane.size() ||
          !accessLane[witness.sourceAccess] ||
          accessLane[witness.sourceAccess] !=
              accessLane[witness.targetAccess]) {
        valid = false;
        break;
      }
      const std::size_t lane = *accessLane[witness.sourceAccess];
      if (lane >= producers.size() || lane >= consumers.size()) {
        return rejectStorage("witness-lane");
      }
      const SyncCoverStorageAccess &source =
          graph.getStorageAccesses()[witness.sourceAccess];
      const SyncCoverStorageAccess &target =
          graph.getStorageAccesses()[witness.targetAccess];
      const bool modesMatch =
          (writeReuse && syncCoverStorageModeWrites(source.mode) &&
           syncCoverStorageModeWrites(target.mode)) ||
          (readRelease && syncCoverStorageModeReads(source.mode) &&
           syncCoverStorageModeWrites(target.mode));
      const std::optional<bool> sourceProducer =
          containsNode(producers[lane], source.node);
      const std::optional<bool> sourceConsumer =
          containsNode(consumers[lane], source.node);
      const std::optional<bool> targetProducer =
          containsNode(producers[lane], target.node);
      if (!sourceProducer || !sourceConsumer || !targetProducer) {
        return std::nullopt;
      }
      if (!source.exactPhysical || !target.exactPhysical ||
          source.node != demand.source || target.node != demand.target ||
          !modesMatch || (writeReuse && !*sourceProducer) ||
          (readRelease && !*sourceConsumer) || !*targetProducer) {
        valid = false;
        break;
      }
    }
    if (valid) {
      covered.insert(demandId);
    }
  }
  return covered;
}

} // namespace

namespace {

SyncCoverLifecycleSccResult
importStorageLifecycleSccs(const SyncCoverGraph &graph,
                           const SyncCoverStorageLifecycleIndex &lifecycleIndex,
                           SyncCoverProtocolLimits limits,
                           SyncCoverCoverageWorkBudget *workBudget) {
  SyncCoverLifecycleSccResult result;
  if (!graph.isStructureFrozen() || !graph.validate() ||
      !lifecycleIndex.isComplete()) {
    result.error = lifecycleIndex.isComplete()
                       ? SyncCoverProtocolError::InvalidGraph
                       : SyncCoverProtocolError::LimitExceeded;
    return result;
  }

  const auto normalizeIds = [&](auto &values) {
    const std::optional<std::size_t> sortWork = comparisonWork(values.size());
    if (!sortWork || !consumeWork(workBudget, *sortWork)) {
      return false;
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return true;
  };
  const SyncCoverStorageLifecycleEdgeKindMask readyRelease =
      syncCoverStorageLifecycleEdgeKindBit(
          SyncCoverStorageLifecycleEdgeKind::Ready) |
      syncCoverStorageLifecycleEdgeKindBit(
          SyncCoverStorageLifecycleEdgeKind::Release);

  for (const SyncCoverStorageLifecycleComponent &component :
       lifecycleIndex.getComponents()) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = component.id;
      return result;
    }
    const bool hasClosedLifecycle = std::any_of(
        component.sccs.begin(), component.sccs.end(),
        [&](const SyncCoverStorageLifecycleScc &scc) {
          return scc.cyclic && (scc.kinds & readyRelease) == readyRelease;
        });
    if (!hasClosedLifecycle) {
      continue;
    }
    if (component.owningScope >= graph.getScopes().size() ||
        !graph.getScopes()[component.owningScope].isLoop) {
      result.error = SyncCoverProtocolError::InvalidGraph;
      result.invalidIndex = component.id;
      return result;
    }
    if (result.components.size() == limits.maximumLifecycleSccs) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = component.id;
      return result;
    }

    SyncCoverLifecycleScc imported;
    imported.loopScope = component.owningScope;
    imported.demands = component.demands;
    for (const SyncCoverStorageLifecycleEpoch &epoch : component.epochs) {
      if (!consumeWork(workBudget, 3)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = component.id;
        return result;
      }
      imported.resources.push_back(epoch.resource);
      imported.nodes.push_back(epoch.node);
      imported.storageAccesses.push_back(epoch.access);
    }
    for (const SyncCoverStorageLifecycleSlot &slot : component.slots) {
      if (!consumeWork(workBudget, slot.accesses.size() + 1)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = component.id;
        return result;
      }
      imported.storageDomains.push_back(slot.domain);
      imported.storageAccesses.insert(imported.storageAccesses.end(),
                                      slot.accesses.begin(),
                                      slot.accesses.end());
    }
    for (const SyncCoverStorageLifecycleEdge &edge : component.edges) {
      if (!consumeWork(workBudget, 2)) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = component.id;
        return result;
      }
      imported.storageWitnesses.push_back(edge.witness);
      imported.maximumDistance =
          std::max(imported.maximumDistance, edge.distance);
    }
    if (!normalizeIds(imported.resources) || !normalizeIds(imported.nodes) ||
        !normalizeIds(imported.demands) ||
        !normalizeIds(imported.storageDomains) ||
        !normalizeIds(imported.storageAccesses) ||
        !normalizeIds(imported.storageWitnesses)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = component.id;
      return result;
    }
    if (imported.nodes.size() > limits.maximumLifecycleVertices ||
        imported.demands.size() > limits.maximumLifecycleEdges ||
        imported.storageDomains.size() >
            limits.maximumLifecycleDomainIncidences ||
        imported.storageAccesses.size() >
            limits.maximumLifecycleAccessIncidences ||
        imported.storageWitnesses.size() >
            limits.maximumStorageWitnessIncidences) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = component.id;
      return result;
    }
    if (!checkedAccumulateSize(result.lifecycleAccessIncidences,
                               imported.storageAccesses.size()) ||
        result.lifecycleAccessIncidences >
            limits.maximumLifecycleAccessIncidences) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = component.id;
      return result;
    }
    result.components.push_back(std::move(imported));
  }
  return result;
}

SyncCoverLifecycleSccResult importStorageLifecycleGroups(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolSeedIndex &seedIndex,
    const SyncCoverStorageProtocolGroupIndex &groupIndex,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  SyncCoverLifecycleSccResult result;
  if (!graph.isStructureFrozen() || !graph.validate() ||
      !lifecycleIndex.isComplete() || !seedIndex.isComplete() ||
      !groupIndex.isComplete()) {
    result.error = graph.isStructureFrozen() && graph.validate()
                       ? SyncCoverProtocolError::LimitExceeded
                       : SyncCoverProtocolError::InvalidGraph;
    return result;
  }

  const std::vector<SyncCoverStorageLifecycleComponent> &components =
      lifecycleIndex.getComponents();
  const std::vector<SyncCoverStorageProtocolSeed> &seeds = seedIndex.getSeeds();
  const auto normalizeIds = [&](auto &values) {
    const std::optional<std::size_t> sortWork = comparisonWork(values.size());
    if (!sortWork || !consumeWork(workBudget, *sortWork)) {
      return false;
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return true;
  };

  for (const SyncCoverStorageProtocolGroup &group : groupIndex.getGroups()) {
    if (!consumeWork(workBudget, group.seeds.size() + 1)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = group.id;
      return result;
    }
    if (group.owningScope >= graph.getScopes().size() ||
        !graph.getScopes()[group.owningScope].isLoop || group.demands.empty()) {
      result.error = SyncCoverProtocolError::InvalidGraph;
      result.invalidIndex = group.id;
      return result;
    }
    if (result.components.size() == limits.maximumLifecycleSccs) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = group.id;
      return result;
    }

    std::set<SyncCoverStorageLifecycleComponentId> componentIds;
    for (SyncCoverStorageProtocolSeedId seedId : group.seeds) {
      if (seedId >= seeds.size() || seeds[seedId].id != seedId ||
          !consumeWork(workBudget, seeds[seedId].components.size() +
                                       lookupWork(componentIds.size() + 1))) {
        result.error = workBudget && workBudget->exhausted
                           ? SyncCoverProtocolError::WorkLimitExceeded
                           : SyncCoverProtocolError::InvalidGraph;
        result.invalidIndex = group.id;
        return result;
      }
      componentIds.insert(seeds[seedId].components.begin(),
                          seeds[seedId].components.end());
    }

    SyncCoverLifecycleScc imported;
    imported.loopScope = group.owningScope;
    imported.demands = group.demands;
    imported.maximumDistance = group.maximumDistance;
    for (SyncCoverStorageLifecycleComponentId componentId : componentIds) {
      if (componentId >= components.size() ||
          components[componentId].id != componentId) {
        result.error = SyncCoverProtocolError::InvalidGraph;
        result.invalidIndex = group.id;
        return result;
      }
      const SyncCoverStorageLifecycleComponent &component =
          components[componentId];
      for (const SyncCoverStorageLifecycleEpoch &epoch : component.epochs) {
        if (!consumeWork(workBudget, 3)) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          result.invalidIndex = group.id;
          return result;
        }
        imported.resources.push_back(epoch.resource);
        imported.nodes.push_back(epoch.node);
        imported.storageAccesses.push_back(epoch.access);
      }
      for (const SyncCoverStorageLifecycleSlot &slot : component.slots) {
        if (!consumeWork(workBudget, slot.accesses.size() + 1)) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          result.invalidIndex = group.id;
          return result;
        }
        imported.storageDomains.push_back(slot.domain);
        imported.storageAccesses.insert(imported.storageAccesses.end(),
                                        slot.accesses.begin(),
                                        slot.accesses.end());
      }
      for (const SyncCoverStorageLifecycleEdge &edge : component.edges) {
        if (!consumeWork(workBudget, 2)) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          result.invalidIndex = group.id;
          return result;
        }
        imported.storageWitnesses.push_back(edge.witness);
        imported.maximumDistance =
            std::max(imported.maximumDistance, edge.distance);
      }
    }
    if (!normalizeIds(imported.resources) || !normalizeIds(imported.nodes) ||
        !normalizeIds(imported.demands) ||
        !normalizeIds(imported.storageDomains) ||
        !normalizeIds(imported.storageAccesses) ||
        !normalizeIds(imported.storageWitnesses)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = group.id;
      return result;
    }
    if (imported.nodes.size() > limits.maximumLifecycleVertices ||
        imported.demands.size() > limits.maximumLifecycleEdges ||
        imported.storageDomains.size() >
            limits.maximumLifecycleDomainIncidences ||
        imported.storageAccesses.size() >
            limits.maximumLifecycleAccessIncidences ||
        imported.storageWitnesses.size() >
            limits.maximumStorageWitnessIncidences ||
        !checkedAccumulateSize(result.lifecycleAccessIncidences,
                               imported.storageAccesses.size()) ||
        result.lifecycleAccessIncidences >
            limits.maximumLifecycleAccessIncidences) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = group.id;
      return result;
    }
    result.components.push_back(std::move(imported));
  }
  return result;
}

SyncCoverLifecycleSynthesisResult synthesizeLifecycleCertificatesFromSccs(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverLifecycleSccResult &components,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {

  SyncCoverLifecycleSynthesisResult result;
  const auto fail = [&](SyncCoverProtocolError error,
                        std::optional<std::size_t> index = std::nullopt) {
    result.error = error;
    result.invalidIndex = index;
    result.proposals.clear();
    return result;
  };
  LifecycleCatalogFootprint retainedCatalog;
  const auto retainProposal = [&](const SyncCoverLifecycleProposal &proposal) {
    const std::optional<LifecycleCatalogFootprint> footprint =
        getLifecycleCatalogFootprint(proposal);
    if (!footprint) {
      return SyncCoverProtocolError::LimitExceeded;
    }
    LifecycleCatalogFootprint next = retainedCatalog;
    if (!checkedAccumulateSize(next.protocols, footprint->protocols) ||
        !checkedAccumulateSize(next.lanes, footprint->lanes) ||
        !checkedAccumulateSize(next.slots, footprint->slots) ||
        !checkedAccumulateSize(next.transfers, footprint->transfers) ||
        !checkedAccumulateSize(next.nodeReferences,
                               footprint->nodeReferences) ||
        !checkedAccumulateSize(next.accessIncidences,
                               footprint->accessIncidences) ||
        !checkedAccumulateSize(next.payloadIncidences,
                               footprint->payloadIncidences) ||
        next.protocols > limits.maximumLifecycleProtocols ||
        next.lanes > limits.maximumLifecycleLanes ||
        next.slots > limits.maximumLifecycleSlots ||
        next.transfers > limits.maximumLifecycleTransfers ||
        next.nodeReferences > limits.maximumLifecycleNodeReferences ||
        next.accessIncidences > limits.maximumLifecycleAccessIncidences ||
        next.payloadIncidences >
            limits.maximumLifecycleCatalogPayloadIncidences) {
      return SyncCoverProtocolError::LimitExceeded;
    }
    if (!consumeWork(workBudget, footprint->payloadIncidences)) {
      return SyncCoverProtocolError::WorkLimitExceeded;
    }
    retainedCatalog = next;
    return SyncCoverProtocolError::None;
  };
  result.lifecycleAccessIncidences = components.lifecycleAccessIncidences;
  result.lifecycleSccs = components.components.size();
  for (std::size_t componentIndex = 0;
       componentIndex < components.components.size(); ++componentIndex) {
    if (!consumeWork(workBudget)) {
      return fail(SyncCoverProtocolError::WorkLimitExceeded, componentIndex);
    }
    SyncCoverLifecycleProposal proposal;
    proposal.id = result.proposals.size();
    proposal.lifecycleScc = componentIndex;
    proposal.seedDemands = components.components[componentIndex].demands;
    const std::size_t certificatesBefore = result.derivedLifecycleCertificates;
    const std::size_t constructionRejectsBefore =
        result.rejectedLifecycleConstructions;
    const std::size_t verificationRejectsBefore =
        result.rejectedLifecycleVerifications;
    const std::size_t storageRejectsBefore =
        result.rejectedLifecycleStorageCertificates;
    std::map<LifecycleSlotKey, std::size_t> proposalSlotIndex;
    std::vector<SyncCoverDemandSet> certifiedStorageReuseByProtocol;
    DerivedLifecycleSchedules schedules = deriveLifecycleSchedules(
        graph, components.components[componentIndex], limits, workBudget);
    if (schedules.error != SyncCoverProtocolError::None) {
      return fail(schedules.error, schedules.invalidIndex);
    }
    if (!checkedAccumulateSize(result.inspectedDemands,
                               schedules.inspectedDemands) ||
        !checkedAccumulateSize(result.inspectedAccesses,
                               schedules.inspectedAccesses) ||
        !checkedAccumulateSize(result.lifecycleAccessIncidences,
                               schedules.lifecycleAccessIncidences) ||
        !checkedAccumulateSize(result.candidateResourcePairs,
                               schedules.candidateResourcePairs)) {
      return fail(SyncCoverProtocolError::LimitExceeded, componentIndex);
    }
    if (result.lifecycleAccessIncidences >
        limits.maximumLifecycleAccessIncidences) {
      return fail(SyncCoverProtocolError::LimitExceeded, componentIndex);
    }
    if (!checkedAccumulateSize(result.derivedLifecycleCertificates,
                               schedules.certificates.size())) {
      return fail(SyncCoverProtocolError::LimitExceeded, componentIndex);
    }
    for (const DerivedLifecycleCertificate &certificate :
         schedules.certificates) {
      if (proposal.protocols.size() == limits.maximumLifecycleProtocols) {
        return fail(SyncCoverProtocolError::LimitExceeded, componentIndex);
      }
      std::optional<SyncCoverEventProtocol> protocol = makeLifecycleProtocol(
          graph, target, certificate, proposal.protocols.size(), workBudget);
      if (workBudget && workBudget->exhausted) {
        return fail(SyncCoverProtocolError::WorkLimitExceeded, componentIndex);
      }
      if (!protocol) {
        ++result.rejectedLifecycleConstructions;
        continue;
      }
      std::optional<std::size_t> verificationRejectionIndex;
      const SyncCoverProtocolError protocolError =
          verifyAndPruneLifecycleCompletionExports(graph, target, *protocol,
                                                   limits, workBudget,
                                                   &verificationRejectionIndex);
      if (protocolError == SyncCoverProtocolError::WorkLimitExceeded) {
        return fail(protocolError, componentIndex);
      }
      if (protocolError != SyncCoverProtocolError::None) {
        if (!result.firstLifecycleVerificationRejection) {
          result.firstLifecycleVerificationRejection = protocolError;
          result.firstLifecycleVerificationRejectionIndex =
              verificationRejectionIndex;
        }
        ++result.rejectedLifecycleVerifications;
        continue;
      }
      const std::optional<SyncCoverDemandSet> storageReuse =
          certifyLifecycleStorageReuse(graph, target, certificate, *protocol,
                                       workBudget);
      if (!storageReuse) {
        if (workBudget && workBudget->exhausted) {
          return fail(SyncCoverProtocolError::WorkLimitExceeded,
                      componentIndex);
        }
        // Lifecycle SCC discovery deliberately over-approximates proposal
        // opportunities.  A derived schedule that cannot be certified against
        // the exact storage lanes is simply not an admissible candidate; it is
        // not evidence that the immutable graph or target contract is invalid.
        // Keep candidate discovery conservative and let the ordinary direct
        // mechanisms cover the component instead of rejecting the function.
        ++result.rejectedLifecycleStorageCertificates;
        continue;
      }
      certifiedStorageReuseByProtocol.push_back(*storageReuse);
      proposal.protocols.push_back(*protocol);
      for (const DerivedLifecycleLane &lane : certificate.lanes) {
        for (const DerivedLifecycleSlot &slot : lane.slots) {
          const SyncCoverProtocolError slotError = addProposalSlot(
              proposal, proposalSlotIndex, slot, limits, workBudget);
          if (slotError != SyncCoverProtocolError::None) {
            return fail(slotError, componentIndex);
          }
        }
      }
    }
    std::vector<WholeSccCycleProtocol> cycleProtocols =
        makeWholeSccCycleProtocols(graph, target,
                                   components.components[componentIndex],
                                   limits, workBudget);
    if (workBudget && workBudget->exhausted) {
      return fail(SyncCoverProtocolError::WorkLimitExceeded, componentIndex);
    }
    for (WholeSccCycleProtocol &cycle : cycleProtocols) {
      if (proposal.protocols.size() == limits.maximumLifecycleProtocols) {
        return fail(SyncCoverProtocolError::LimitExceeded, componentIndex);
      }
      SyncCoverEventProtocol &protocol = cycle.protocol;
      protocol.mechanism = proposal.protocols.size();
      proposal.protocols.push_back(std::move(protocol));
      certifiedStorageReuseByProtocol.emplace_back(graph.getDemands().size());
      const SyncCoverProtocolError slotError = addProposalSlot(
          proposal, proposalSlotIndex, cycle.slot, limits, workBudget);
      if (slotError != SyncCoverProtocolError::None) {
        return fail(slotError, componentIndex);
      }
    }
    if (proposal.protocols.empty()) {
      ++result.lifecycleSccsWithoutProtocols;
      if (!components.components[componentIndex].demands.empty()) {
        const SyncCoverDemandId first =
            components.components[componentIndex].demands.front();
        if (!result.firstUnproposedLifecycleDemand ||
            first < *result.firstUnproposedLifecycleDemand) {
          result.firstUnproposedLifecycleDemand = first;
          result.firstUnproposedLifecycleCertificates =
              result.derivedLifecycleCertificates - certificatesBefore;
          result.firstUnproposedLifecycleConstructionRejects =
              result.rejectedLifecycleConstructions - constructionRejectsBefore;
          result.firstUnproposedLifecycleVerificationRejects =
              result.rejectedLifecycleVerifications - verificationRejectsBefore;
          result.firstUnproposedLifecycleStorageRejects =
              result.rejectedLifecycleStorageCertificates -
              storageRejectsBefore;
        }
      }
      continue;
    }
    if (certifiedStorageReuseByProtocol.size() != proposal.protocols.size()) {
      return fail(SyncCoverProtocolError::InvalidProtocol, componentIndex);
    }
    std::map<std::vector<std::size_t>, std::size_t> physicalProtocols;
    std::vector<SyncCoverEventProtocol> uniqueProtocols;
    std::vector<SyncCoverDemandSet> uniqueStorageReuse;
    uniqueProtocols.reserve(proposal.protocols.size());
    uniqueStorageReuse.reserve(certifiedStorageReuseByProtocol.size());
    for (std::size_t protocolIndex = 0;
         protocolIndex < proposal.protocols.size(); ++protocolIndex) {
      std::optional<std::vector<std::size_t>> physicalKey = protocolPhysicalKey(
          proposal.protocols[protocolIndex], limits, workBudget);
      if (!physicalKey) {
        return fail(workBudget && workBudget->exhausted
                        ? SyncCoverProtocolError::WorkLimitExceeded
                        : SyncCoverProtocolError::LimitExceeded,
                    componentIndex);
      }
      const std::size_t lookup = lookupWork(physicalProtocols.size() + 1);
      if ((!physicalKey->empty() &&
           lookup >
               std::numeric_limits<std::size_t>::max() / physicalKey->size()) ||
          !consumeWork(workBudget, lookup * physicalKey->size())) {
        return fail(SyncCoverProtocolError::WorkLimitExceeded, componentIndex);
      }
      const auto duplicate = physicalProtocols.find(*physicalKey);
      if (duplicate != physicalProtocols.end()) {
        if (!consumeWork(workBudget,
                         certifiedStorageReuseByProtocol[protocolIndex]
                             .getWords()
                             .size())) {
          return fail(SyncCoverProtocolError::WorkLimitExceeded,
                      componentIndex);
        }
        uniqueStorageReuse[duplicate->second].unite(
            certifiedStorageReuseByProtocol[protocolIndex]);
        continue;
      }
      const std::size_t retained = uniqueProtocols.size();
      physicalProtocols.emplace(std::move(*physicalKey), retained);
      proposal.protocols[protocolIndex].mechanism = retained;
      uniqueProtocols.push_back(std::move(proposal.protocols[protocolIndex]));
      uniqueStorageReuse.push_back(
          std::move(certifiedStorageReuseByProtocol[protocolIndex]));
    }
    proposal.protocols = std::move(uniqueProtocols);
    certifiedStorageReuseByProtocol = std::move(uniqueStorageReuse);
    SyncCoverExactWorld proposalWorld;
    proposalWorld.enabledMechanisms.reserve(proposal.protocols.size());
    for (const SyncCoverEventProtocol &protocol : proposal.protocols) {
      proposalWorld.enabledMechanisms.push_back(protocol.mechanism);
    }
    const SyncCoverProtocolCoverageResult directCoverage =
        computeSyncCoverProtocolDirectWorlds(graph, target, proposal.protocols,
                                             {proposalWorld}, limits,
                                             workBudget);
    if (!directCoverage) {
      return fail(directCoverage.error, directCoverage.invalidIndex);
    }
    const std::size_t demandWords =
        directCoverage.coveredByWorld.front().getWords().size();
    if (demandWords > std::numeric_limits<std::size_t>::max() / 12 ||
        !consumeWork(workBudget, demandWords * 12)) {
      return fail(SyncCoverProtocolError::WorkLimitExceeded, componentIndex);
    }
    SyncCoverDemandSet directUnion = directCoverage.coveredByWorld.front();
    proposal.singletonUnionCoverageRows = directUnion.count();
    const SyncCoverProtocolCoverageResult connectorCoverage =
        computeSyncCoverProtocolConnectorClosure(graph, directUnion, limits,
                                                 workBudget);
    if (!connectorCoverage) {
      return fail(connectorCoverage.error, connectorCoverage.invalidIndex);
    }
    SyncCoverDemandSet residualQueries =
        connectorCoverage.coveredByWorld.front();
    for (const SyncCoverDemandSet &certified :
         certifiedStorageReuseByProtocol) {
      residualQueries.unite(certified);
    }
    residualQueries.subtract(directUnion);
    proposal.exactCoverage = directUnion;
    if (!residualQueries.empty()) {
      const SyncCoverProtocolCoverageResult exactCoverage =
          computeSyncCoverProtocolExactWorldsForDemands(
              graph, target, proposal.protocols, {proposalWorld},
              residualQueries, limits, workBudget);
      if (!exactCoverage) {
        return fail(exactCoverage.error, exactCoverage.invalidIndex);
      }
      if (!consumeWork(
              workBudget,
              exactCoverage.coveredByWorld.front().getWords().size())) {
        return fail(SyncCoverProtocolError::WorkLimitExceeded, componentIndex);
      }
      proposal.exactCoverage.unite(exactCoverage.coveredByWorld.front());
    }
    // Storage reuse is proven independently from ordinary graph reachability:
    // every witness must name one exact physical lane, both ready/release
    // channels must supply the required hardware ordering, the relevant
    // producer/consumer occurrences must belong to that lane, and every
    // release lane must be drained.  Parent-loop WAW obligations may cross a
    // nested lifecycle whose compact arena intentionally does not flatten the
    // child protocol into the parent graph.  Preserve those certified rows in
    // the immutable column instead of treating them only as reachability
    // queries; the event-token automaton and materialized-plan verifier still
    // validate the complete physical recipe.
    for (const SyncCoverDemandSet &certified :
         certifiedStorageReuseByProtocol) {
      if (!consumeWork(workBudget, certified.getWords().size())) {
        return fail(SyncCoverProtocolError::WorkLimitExceeded, componentIndex);
      }
      proposal.exactCoverage.unite(certified);
    }
    SyncCoverDemandSet extra = proposal.exactCoverage;
    extra.subtract(directUnion);
    proposal.extraCoverageRows = extra.count();
    if (proposal.exactCoverage.empty()) {
      ++result.lifecycleSccsWithoutCoverage;
      continue;
    }
    if (result.proposals.size() == limits.maximumLifecycleProposals) {
      return fail(SyncCoverProtocolError::LimitExceeded, componentIndex);
    }
    const SyncCoverProtocolError retained = retainProposal(proposal);
    if (retained != SyncCoverProtocolError::None) {
      return fail(retained, componentIndex);
    }
    result.proposals.push_back(std::move(proposal));
  }
  struct ConnectorKey {
    SyncCoverRegionId ownerRegion = 0;
    SyncCoverScopeId scope = 0;
    unsigned distance = 0;
    SyncCoverRegionId region = 0;
    SyncCoverNodeId node = 0;
    std::uint32_t resource = 0;
    SyncCoverOrderingRequirementMask capability = 0;
    std::size_t guardClass = 0;

    bool operator<(const ConnectorKey &other) const {
      return std::tie(ownerRegion, scope, distance, region, node, resource,
                      capability, guardClass) <
             std::tie(other.ownerRegion, other.scope, other.distance,
                      other.region, other.node, other.resource,
                      other.capability, other.guardClass);
    }
  };
  struct ConnectorBucket {
    std::set<std::size_t> endingProposals;
    std::set<std::size_t> startingProposals;
  };
  const std::size_t baseProposalCount = result.proposals.size();
  std::map<ConnectorKey, ConnectorBucket> connectorIndex;
  std::map<std::vector<SyncCoverGuardLiteral>, std::size_t> guardClasses;
  std::size_t connectorGuardIncidences = 0;
  const auto internGuardClass =
      [&](const std::vector<SyncCoverGuardLiteral> &guard)
      -> std::optional<std::size_t> {
    const std::optional<std::size_t> comparisons =
        comparisonWork(guard.size(), guardClasses.size() + 1);
    if (!comparisons || !consumeWork(workBudget, *comparisons + 1)) {
      return std::nullopt;
    }
    const auto existing = guardClasses.find(guard);
    if (existing != guardClasses.end()) {
      return existing->second;
    }
    if (guard.size() >
        limits.maximumLifecycleConnectorGuardIncidences -
            std::min(connectorGuardIncidences,
                     limits.maximumLifecycleConnectorGuardIncidences)) {
      return std::nullopt;
    }
    const std::size_t id = guardClasses.size();
    guardClasses.emplace(guard, id);
    connectorGuardIncidences += guard.size();
    return id;
  };
  for (std::size_t proposalIndex = 0; proposalIndex < baseProposalCount;
       ++proposalIndex) {
    const SyncCoverLifecycleProposal &proposal =
        result.proposals[proposalIndex];
    for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
         ++demandId) {
      if (!consumeWork(workBudget)) {
        return fail(SyncCoverProtocolError::WorkLimitExceeded, proposalIndex);
      }
      if (!proposal.exactCoverage.contains(demandId)) {
        continue;
      }
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      // Group closure is merely a proposal accelerator. Restrict it to one
      // exact zero-distance coordinate; positive-distance composition belongs
      // to the lifecycle automaton and must never be flattened here.
      if (demand.distance != 0) {
        continue;
      }
      const SyncCoverNode &source = graph.getNodes()[demand.source];
      const SyncCoverNode &targetNode = graph.getNodes()[demand.target];
      const std::optional<std::size_t> sourceGuard =
          internGuardClass(demand.sourceGuard.literals);
      const std::optional<std::size_t> targetGuard =
          internGuardClass(demand.targetGuard.literals);
      if (!sourceGuard || !targetGuard) {
        return fail(workBudget && workBudget->exhausted
                        ? SyncCoverProtocolError::WorkLimitExceeded
                        : SyncCoverProtocolError::LimitExceeded,
                    proposalIndex);
      }
      const ConnectorKey start{demand.ownerRegion,
                               demand.scope,
                               demand.distance,
                               source.region,
                               demand.source,
                               source.resource,
                               demand.orderingRequirements,
                               *sourceGuard};
      const ConnectorKey end{
          demand.ownerRegion,          demand.scope,  demand.distance,
          targetNode.region,           demand.target, targetNode.resource,
          demand.orderingRequirements, *targetGuard};
      const auto getBucket = [&](const ConnectorKey &key) -> ConnectorBucket * {
        if (!consumeWork(workBudget, lookupWork(connectorIndex.size() + 1))) {
          return nullptr;
        }
        return &connectorIndex.try_emplace(key).first->second;
      };
      ConnectorBucket *startBucket = getBucket(start);
      ConnectorBucket *endBucket = getBucket(end);
      if (!startBucket || !endBucket ||
          !consumeWork(workBudget,
                       lookupWork(startBucket->startingProposals.size() + 1) +
                           lookupWork(endBucket->endingProposals.size() + 1))) {
        return fail(SyncCoverProtocolError::WorkLimitExceeded, proposalIndex);
      }
      if (startBucket->startingProposals.insert(proposalIndex).second) {
        ++result.lifecycleConnectorIncidences;
      }
      if (endBucket->endingProposals.insert(proposalIndex).second) {
        ++result.lifecycleConnectorIncidences;
      }
      if (result.lifecycleConnectorIncidences >
              limits.maximumLifecycleConnectorIncidences ||
          !consumeWork(workBudget, 2)) {
        return fail(workBudget && workBudget->exhausted
                        ? SyncCoverProtocolError::WorkLimitExceeded
                        : SyncCoverProtocolError::LimitExceeded,
                    proposalIndex);
      }
    }
  }

  if (!consumeWork(workBudget, baseProposalCount)) {
    return fail(SyncCoverProtocolError::WorkLimitExceeded);
  }
  std::vector<std::size_t> parents(baseProposalCount);
  for (std::size_t index = 0; index < parents.size(); ++index) {
    parents[index] = index;
  }
  const auto findRoot = [&](std::size_t node) -> std::optional<std::size_t> {
    while (parents[node] != node) {
      if (!consumeWork(workBudget)) {
        return std::nullopt;
      }
      parents[node] = parents[parents[node]];
      node = parents[node];
    }
    return node;
  };
  std::size_t connectorInspections = 0;
  std::set<std::pair<std::size_t, std::size_t>> connectorPairs;
  for (const auto &[key, bucket] : connectorIndex) {
    (void)key;
    for (std::size_t ending : bucket.endingProposals) {
      for (std::size_t starting : bucket.startingProposals) {
        if (++connectorInspections >
                limits.maximumLifecycleConnectorIncidences ||
            !consumeWork(workBudget)) {
          return fail(workBudget && workBudget->exhausted
                          ? SyncCoverProtocolError::WorkLimitExceeded
                          : SyncCoverProtocolError::LimitExceeded,
                      ending);
        }
        const std::optional<std::size_t> endingRoot = findRoot(ending);
        const std::optional<std::size_t> startingRoot = findRoot(starting);
        if (!endingRoot || !startingRoot) {
          return fail(SyncCoverProtocolError::WorkLimitExceeded, ending);
        }
        if (*endingRoot != *startingRoot) {
          parents[*startingRoot] = *endingRoot;
        }
        if (ending != starting) {
          if (!consumeWork(workBudget, lookupWork(connectorPairs.size() + 1))) {
            return fail(SyncCoverProtocolError::WorkLimitExceeded, ending);
          }
          connectorPairs.emplace(std::min(ending, starting),
                                 std::max(ending, starting));
          if (connectorPairs.size() >
              limits.maximumLifecycleConnectorIncidences) {
            return fail(SyncCoverProtocolError::LimitExceeded, ending);
          }
        }
      }
    }
  }

  std::map<std::size_t, std::vector<std::size_t>> connectorGroups;
  for (std::size_t proposalIndex = 0; proposalIndex < baseProposalCount;
       ++proposalIndex) {
    if (!consumeWork(workBudget, lookupWork(connectorGroups.size() + 1))) {
      return fail(SyncCoverProtocolError::WorkLimitExceeded, proposalIndex);
    }
    const std::optional<std::size_t> root = findRoot(proposalIndex);
    if (!root) {
      return fail(SyncCoverProtocolError::WorkLimitExceeded, proposalIndex);
    }
    connectorGroups[*root].push_back(proposalIndex);
  }
  std::set<std::vector<std::size_t>> candidateGroups;
  std::size_t groupIncidences = 0;
  const auto addCandidateGroup = [&](std::vector<std::size_t> group) {
    const std::optional<std::size_t> lookup =
        comparisonWork(candidateGroups.size() + 1, group.size() + 1);
    if (!lookup || !consumeWork(workBudget, *lookup)) {
      return false;
    }
    if (candidateGroups.count(group) != 0) {
      return true;
    }
    if (group.size() > limits.maximumLifecycleGroupIncidences -
                           std::min(groupIncidences,
                                    limits.maximumLifecycleGroupIncidences) ||
        !consumeWork(workBudget, group.size())) {
      return false;
    }
    groupIncidences += group.size();
    candidateGroups.insert(std::move(group));
    return true;
  };
  for (const auto &pair : connectorPairs) {
    if (!addCandidateGroup({pair.first, pair.second})) {
      return fail(workBudget && workBudget->exhausted
                      ? SyncCoverProtocolError::WorkLimitExceeded
                      : SyncCoverProtocolError::LimitExceeded,
                  pair.first);
    }
  }
  for (const auto &[root, group] : connectorGroups) {
    (void)root;
    if (group.size() >= 2 && !addCandidateGroup(group)) {
      return fail(workBudget && workBudget->exhausted
                      ? SyncCoverProtocolError::WorkLimitExceeded
                      : SyncCoverProtocolError::LimitExceeded,
                  root);
    }
  }
  for (const std::vector<std::size_t> &group : candidateGroups) {
    if (result.proposals.size() == limits.maximumLifecycleProposals ||
        !consumeWork(workBudget, group.size())) {
      return fail(workBudget && workBudget->exhausted
                      ? SyncCoverProtocolError::WorkLimitExceeded
                      : SyncCoverProtocolError::LimitExceeded,
                  result.proposals.size());
    }
    std::size_t copyPayload = 0;
    for (std::size_t proposalIndex : group) {
      const std::optional<LifecycleCatalogFootprint> footprint =
          getLifecycleCatalogFootprint(result.proposals[proposalIndex]);
      if (!footprint ||
          !checkedAccumulateSize(copyPayload, footprint->payloadIncidences)) {
        return fail(SyncCoverProtocolError::LimitExceeded, proposalIndex);
      }
    }
    if (copyPayload > limits.maximumLifecycleCatalogPayloadIncidences ||
        !consumeWork(workBudget, copyPayload)) {
      return fail(workBudget && workBudget->exhausted
                      ? SyncCoverProtocolError::WorkLimitExceeded
                      : SyncCoverProtocolError::LimitExceeded,
                  group.front());
    }
    SyncCoverLifecycleProposal combined;
    combined.id = result.proposals.size();
    combined.lifecycleScc = components.components.size();
    std::set<LifecycleSlotKey> retainedSlots;
    for (std::size_t proposalIndex : group) {
      const SyncCoverLifecycleProposal &proposal =
          result.proposals[proposalIndex];
      combined.seedDemands.insert(combined.seedDemands.end(),
                                  proposal.seedDemands.begin(),
                                  proposal.seedDemands.end());
      for (const SyncCoverLifecycleSlot &slot : proposal.slots) {
        if (!consumeWork(workBudget, lookupWork(retainedSlots.size() + 1))) {
          return fail(SyncCoverProtocolError::WorkLimitExceeded, combined.id);
        }
        if (retainedSlots.insert({slot.domain, slot.extent}).second) {
          if (combined.slots.size() == limits.maximumLifecycleSlots) {
            return fail(SyncCoverProtocolError::LimitExceeded, combined.id);
          }
          combined.slots.push_back(slot);
        }
      }
      for (const SyncCoverEventProtocol &source : proposal.protocols) {
        if (combined.protocols.size() == limits.maximumLifecycleProtocols) {
          return fail(SyncCoverProtocolError::LimitExceeded, combined.id);
        }
        SyncCoverEventProtocol protocol = source;
        protocol.mechanism = combined.protocols.size();
        combined.protocols.push_back(std::move(protocol));
      }
    }
    const std::optional<std::size_t> seedSortWork =
        comparisonWork(combined.seedDemands.size());
    if (!seedSortWork || !consumeWork(workBudget, *seedSortWork)) {
      return fail(SyncCoverProtocolError::WorkLimitExceeded, combined.id);
    }
    std::sort(combined.seedDemands.begin(), combined.seedDemands.end());
    combined.seedDemands.erase(
        std::unique(combined.seedDemands.begin(), combined.seedDemands.end()),
        combined.seedDemands.end());
    SyncCoverDemandSet directUnion(graph.getDemands().size());
    for (std::size_t proposalIndex : group) {
      if (!consumeWork(workBudget, result.proposals[proposalIndex]
                                       .exactCoverage.getWords()
                                       .size())) {
        return fail(SyncCoverProtocolError::WorkLimitExceeded, combined.id);
      }
      directUnion.unite(result.proposals[proposalIndex].exactCoverage);
    }
    SyncCoverExactWorld combinedWorld;
    combinedWorld.enabledMechanisms.reserve(combined.protocols.size());
    for (const SyncCoverEventProtocol &protocol : combined.protocols) {
      combinedWorld.enabledMechanisms.push_back(protocol.mechanism);
    }
    const SyncCoverProtocolCoverageResult connectorCoverage =
        computeSyncCoverProtocolConnectorClosure(graph, directUnion, limits,
                                                 workBudget);
    if (!connectorCoverage) {
      return fail(connectorCoverage.error, connectorCoverage.invalidIndex);
    }
    SyncCoverDemandSet residualQueries =
        connectorCoverage.coveredByWorld.front();
    if (!consumeWork(workBudget, residualQueries.getWords().size())) {
      return fail(SyncCoverProtocolError::WorkLimitExceeded, combined.id);
    }
    residualQueries.subtract(directUnion);
    combined.exactCoverage = directUnion;
    if (!residualQueries.empty()) {
      const SyncCoverProtocolCoverageResult exactCoverage =
          computeSyncCoverProtocolExactWorldsForDemands(
              graph, target, combined.protocols, {combinedWorld},
              residualQueries, limits, workBudget);
      if (!exactCoverage) {
        return fail(exactCoverage.error, exactCoverage.invalidIndex);
      }
      if (!consumeWork(
              workBudget,
              exactCoverage.coveredByWorld.front().getWords().size())) {
        return fail(SyncCoverProtocolError::WorkLimitExceeded, combined.id);
      }
      combined.exactCoverage.unite(exactCoverage.coveredByWorld.front());
    }
    SyncCoverDemandSet combinedExtra = combined.exactCoverage;
    if (!consumeWork(workBudget, combinedExtra.getWords().size() * 3)) {
      return fail(SyncCoverProtocolError::WorkLimitExceeded, combined.id);
    }
    combinedExtra.subtract(directUnion);
    combined.singletonUnionCoverageRows = directUnion.count();
    combined.extraCoverageRows = combinedExtra.count();
    if (combined.extraCoverageRows == 0) {
      continue;
    }
    if (result.proposals.size() == limits.maximumLifecycleProposals) {
      return fail(SyncCoverProtocolError::LimitExceeded, combined.id);
    }
    const SyncCoverProtocolError retained = retainProposal(combined);
    if (retained != SyncCoverProtocolError::None) {
      return fail(retained, combined.id);
    }
    result.proposals.push_back(std::move(combined));
  }
  return result;
}

} // namespace

SyncCoverLifecycleSynthesisResult
mlir::pto::synthesizeSyncCoverBalancedDirectProtocols(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget,
    bool enableCutFusion) {
  SyncCoverLifecycleSynthesisResult result;
  if (!graphFitsProtocolLimits(graph, limits) || !graph.isStructureFrozen() ||
      !graph.validate()) {
    result.error = graphFitsProtocolLimits(graph, limits)
                       ? SyncCoverProtocolError::InvalidGraph
                       : SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  LifecycleCatalogFootprint retainedCatalog;
  std::map<std::vector<std::size_t>, std::size_t> retainedProtocolKeys;
  std::vector<std::vector<std::size_t>> retainedProtocolPhysicalKeys;
  const auto fail = [&](SyncCoverProtocolError error,
                        std::optional<std::size_t> index) {
    result.error = error;
    result.invalidIndex = index;
    result.proposals.clear();
    return false;
  };
  const auto retainVerifiedProtocol = [&](SyncCoverEventProtocol protocol,
                                          std::vector<SyncCoverDemandId>
                                              seedDemands,
                                          std::size_t diagnosticIndex) {
    std::optional<std::vector<std::size_t>> physicalKey =
        protocolPhysicalKey(protocol, limits, workBudget);
    if (!physicalKey) {
      return fail(workBudget && workBudget->exhausted
                      ? SyncCoverProtocolError::WorkLimitExceeded
                      : SyncCoverProtocolError::LimitExceeded,
                  diagnosticIndex);
    }
    const std::size_t lookup = lookupWork(retainedProtocolKeys.size() + 1);
    if ((!physicalKey->empty() &&
         lookup >
             std::numeric_limits<std::size_t>::max() / physicalKey->size()) ||
        !consumeWork(workBudget, lookup * physicalKey->size())) {
      return fail(SyncCoverProtocolError::WorkLimitExceeded, diagnosticIndex);
    }
    const auto retained = retainedProtocolKeys.find(*physicalKey);
    if (retained != retainedProtocolKeys.end()) {
      if (retained->second >= result.proposals.size()) {
        return fail(SyncCoverProtocolError::InvalidProtocol, diagnosticIndex);
      }
      for (SyncCoverDemandId seed : seedDemands) {
        if (!result.proposals[retained->second].exactCoverage.contains(seed)) {
          ++result.lifecycleSccsWithoutCoverage;
        }
      }
      return true;
    }

    std::optional<std::size_t> invalidIndex;
    const SyncCoverProtocolError verification =
        verifyAndPruneLifecycleCompletionExports(
            graph, target, protocol, limits, workBudget, &invalidIndex);
    if (verification == SyncCoverProtocolError::WorkLimitExceeded) {
      return fail(verification, diagnosticIndex);
    }
    if (verification != SyncCoverProtocolError::None) {
      ++result.rejectedLifecycleVerifications;
      if (!result.firstLifecycleVerificationRejection) {
        result.firstLifecycleVerificationRejection = verification;
        result.firstLifecycleVerificationRejectionIndex = invalidIndex;
      }
      return true;
    }

    SyncCoverLifecycleProposal proposal;
    proposal.id = result.proposals.size();
    proposal.lifecycleScc = graph.getDemands().size() + diagnosticIndex;
    // This is a fused multi-cut optimization derived from the singleton
    // direct basis below.  Keep it out of the resource-fallback basis even
    // though all of its parents are direct mechanisms.
    proposal.directBasis = false;
    proposal.seedDemands = std::move(seedDemands);
    proposal.protocols.push_back(std::move(protocol));
    const SyncCoverExactWorld world{{0}};
    const SyncCoverProtocolCoverageResult direct =
        computeSyncCoverProtocolDirectWorlds(graph, target, proposal.protocols,
                                             {world}, limits, workBudget);
    if (!direct) {
      return fail(direct.error, direct.invalidIndex);
    }
    proposal.singletonUnionCoverageRows = direct.coveredByWorld.front().count();
    proposal.exactCoverage = direct.coveredByWorld.front();
    const SyncCoverProtocolCoverageResult connector =
        computeSyncCoverProtocolConnectorClosure(graph, proposal.exactCoverage,
                                                 limits, workBudget);
    if (!connector) {
      return fail(connector.error, connector.invalidIndex);
    }
    SyncCoverDemandSet residualQueries = connector.coveredByWorld.front();
    residualQueries.subtract(proposal.exactCoverage);
    if (!residualQueries.empty()) {
      const SyncCoverProtocolCoverageResult exact =
          computeSyncCoverProtocolExactWorldsForDemands(
              graph, target, proposal.protocols, {world}, residualQueries,
              limits, workBudget);
      if (!exact) {
        return fail(exact.error, exact.invalidIndex);
      }
      proposal.exactCoverage.unite(exact.coveredByWorld.front());
    }
    for (SyncCoverDemandId seed : proposal.seedDemands) {
      if (!proposal.exactCoverage.contains(seed)) {
        ++result.lifecycleSccsWithoutCoverage;
        return true;
      }
    }
    SyncCoverDemandSet extra = proposal.exactCoverage;
    extra.subtract(direct.coveredByWorld.front());
    proposal.extraCoverageRows = extra.count();
    if (result.proposals.size() == limits.maximumLifecycleProposals) {
      return fail(SyncCoverProtocolError::LimitExceeded, diagnosticIndex);
    }
    const std::optional<LifecycleCatalogFootprint> footprint =
        getLifecycleCatalogFootprint(proposal);
    if (!footprint) {
      return fail(SyncCoverProtocolError::LimitExceeded, diagnosticIndex);
    }
    LifecycleCatalogFootprint next = retainedCatalog;
    const bool fits =
        checkedAccumulateSize(next.protocols, footprint->protocols) &&
        checkedAccumulateSize(next.lanes, footprint->lanes) &&
        checkedAccumulateSize(next.slots, footprint->slots) &&
        checkedAccumulateSize(next.transfers, footprint->transfers) &&
        checkedAccumulateSize(next.nodeReferences, footprint->nodeReferences) &&
        checkedAccumulateSize(next.accessIncidences,
                              footprint->accessIncidences) &&
        checkedAccumulateSize(next.payloadIncidences,
                              footprint->payloadIncidences) &&
        next.protocols <= limits.maximumLifecycleProtocols &&
        next.lanes <= limits.maximumLifecycleLanes &&
        next.slots <= limits.maximumLifecycleSlots &&
        next.transfers <= limits.maximumLifecycleTransfers &&
        next.nodeReferences <= limits.maximumLifecycleNodeReferences &&
        next.accessIncidences <= limits.maximumLifecycleAccessIncidences &&
        next.payloadIncidences <=
            limits.maximumLifecycleCatalogPayloadIncidences;
    if (!fits) {
      return fail(SyncCoverProtocolError::LimitExceeded, diagnosticIndex);
    }
    if (!consumeWork(workBudget, footprint->payloadIncidences)) {
      return fail(SyncCoverProtocolError::WorkLimitExceeded, diagnosticIndex);
    }
    retainedCatalog = next;
    retainedProtocolKeys.emplace(std::move(*physicalKey),
                                 result.proposals.size());
    result.proposals.push_back(std::move(proposal));
    return true;
  };
  std::map<
      LifecyclePairKey,
      std::vector<std::pair<SyncCoverDemandId, DerivedLifecycleCertificate>>>
      groupedCertificates;
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = demandId;
      result.proposals.clear();
      return result;
    }
    std::optional<SyncCoverEventProtocol> protocol =
        makeDirectRecurrenceRoundTripProtocol(graph, target, demandId, 0,
                                              workBudget);
    if (!protocol) {
      protocol =
          makeStructuralDirectProtocol(graph, target, demandId, 0, workBudget);
    }
    std::optional<DerivedLifecycleCertificate> certificate;
    if (!protocol) {
      certificate = buildBalancedDirectCertificate(graph, demandId, workBudget);
    }
    if (workBudget && workBudget->exhausted) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = demandId;
      result.proposals.clear();
      return result;
    }
    if (!protocol && !certificate) {
      continue;
    }
    ++result.derivedLifecycleCertificates;
    if (certificate) {
      groupedCertificates[{certificate->loopScope,
                           certificate->producerResource,
                           certificate->consumerResource}]
          .emplace_back(demandId, *certificate);
      const SyncCoverProtocolError bounds =
          checkLifecycleCertificateBounds(*certificate, limits, workBudget);
      if (bounds != SyncCoverProtocolError::None) {
        result.error = bounds;
        result.invalidIndex = demandId;
        result.proposals.clear();
        return result;
      }
      protocol =
          makeLifecycleProtocol(graph, target, *certificate, 0, workBudget);
    }
    if (workBudget && workBudget->exhausted) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = demandId;
      result.proposals.clear();
      return result;
    }
    if (!protocol) {
      ++result.rejectedLifecycleConstructions;
      continue;
    }

    // Canonicalize the unverified physical recipe before running the graph
    // automaton or grounding its rectangles. Construction and verification
    // are deterministic functions of this complete recipe and the frozen
    // graph, so an identical key must produce the same verified protocol and
    // exact coverage. This prevents access-witness and phase-expanded demand
    // rows from repeatedly paying the full protocol-analysis cost.
    std::optional<std::vector<std::size_t>> physicalKey =
        protocolPhysicalKey(*protocol, limits, workBudget);
    if (!physicalKey) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = demandId;
      result.proposals.clear();
      return result;
    }
    const std::size_t lookup = lookupWork(retainedProtocolKeys.size() + 1);
    if ((!physicalKey->empty() &&
         lookup >
             std::numeric_limits<std::size_t>::max() / physicalKey->size()) ||
        !consumeWork(workBudget, lookup * physicalKey->size())) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = demandId;
      result.proposals.clear();
      return result;
    }
    const auto retained = retainedProtocolKeys.find(*physicalKey);
    if (retained != retainedProtocolKeys.end()) {
      if (retained->second >= result.proposals.size()) {
        result.error = SyncCoverProtocolError::InvalidProtocol;
        result.invalidIndex = demandId;
        result.proposals.clear();
        return result;
      }
      result.proposals[retained->second].seedDemands.push_back(demandId);
      continue;
    }

    std::optional<std::size_t> invalidIndex;
    const SyncCoverProtocolError verification =
        verifyAndPruneLifecycleCompletionExports(
            graph, target, *protocol, limits, workBudget, &invalidIndex);
    if (verification == SyncCoverProtocolError::WorkLimitExceeded) {
      result.error = verification;
      result.invalidIndex = demandId;
      result.proposals.clear();
      return result;
    }
    if (verification != SyncCoverProtocolError::None) {
      ++result.rejectedLifecycleVerifications;
      if (!result.firstLifecycleVerificationRejection) {
        result.firstLifecycleVerificationRejection = verification;
        result.firstLifecycleVerificationRejectionIndex = invalidIndex;
      }
      continue;
    }
    protocol->mechanism = 0;

    SyncCoverLifecycleProposal proposal;
    proposal.id = result.proposals.size();
    proposal.lifecycleScc = graph.getDemands().size() + demandId;
    proposal.directBasis = true;
    proposal.seedDemands = {demandId};
    proposal.protocols.push_back(std::move(*protocol));
    if (result.proposals.size() == limits.maximumLifecycleProposals) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = demandId;
      result.proposals.clear();
      return result;
    }
    retainedProtocolPhysicalKeys.push_back(*physicalKey);
    retainedProtocolKeys.emplace(std::move(*physicalKey),
                                 result.proposals.size());
    result.proposals.push_back(std::move(proposal));
  }

  // Ground independent direct protocols in bounded multi-world batches.  A
  // world still enables exactly one physical recipe, so this is semantically
  // identical to one call per proposal.  Sharing target validation, graph
  // indices, recurrence expansion, and protocol preparation avoids repeating
  // graph-wide setup for every demand-derived candidate.
  std::vector<SyncCoverDemandSet> directCoverage;
  directCoverage.reserve(result.proposals.size());
  for (std::size_t index = 0; index < result.proposals.size(); ++index) {
    directCoverage.emplace_back(graph.getDemands().size());
  }
  for (std::size_t begin = 0; begin < result.proposals.size();) {
    const std::size_t count =
        std::min(limits.maximumWorlds, result.proposals.size() - begin);
    if (count == 0) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.proposals.clear();
      return result;
    }
    std::vector<SyncCoverEventProtocol> protocols;
    std::vector<SyncCoverExactWorld> worlds;
    protocols.reserve(count);
    worlds.reserve(count);
    for (std::size_t offset = 0; offset < count; ++offset) {
      const SyncCoverLifecycleProposal &proposal =
          result.proposals[begin + offset];
      if (proposal.protocols.size() != 1) {
        result.error = SyncCoverProtocolError::InvalidProtocol;
        result.invalidIndex = begin + offset;
        result.proposals.clear();
        return result;
      }
      protocols.push_back(proposal.protocols.front());
      protocols.back().mechanism = offset;
      worlds.push_back({{offset}});
    }
    const SyncCoverProtocolCoverageResult direct =
        computeSyncCoverProtocolDirectWorlds(graph, target, protocols, worlds,
                                             limits, workBudget);
    if (!direct || direct.coveredByWorld.size() != count) {
      result.error =
          direct ? SyncCoverProtocolError::InvalidProtocol : direct.error;
      result.invalidIndex = direct.invalidIndex;
      result.proposals.clear();
      return result;
    }
    for (std::size_t offset = 0; offset < count; ++offset) {
      directCoverage[begin + offset] = direct.coveredByWorld[offset];
    }
    begin += count;
  }

  std::vector<SyncCoverLifecycleProposal> groundedProposals;
  std::vector<std::vector<std::size_t>> groundedPhysicalKeys;
  groundedProposals.reserve(result.proposals.size());
  groundedPhysicalKeys.reserve(result.proposals.size());
  retainedCatalog = {};
  for (std::size_t proposalIndex = 0; proposalIndex < result.proposals.size();
       ++proposalIndex) {
    SyncCoverLifecycleProposal proposal =
        std::move(result.proposals[proposalIndex]);
    proposal.singletonUnionCoverageRows = directCoverage[proposalIndex].count();
    proposal.exactCoverage = directCoverage[proposalIndex];
    if (enableCutFusion) {
      const SyncCoverProtocolCoverageResult connector =
          computeSyncCoverProtocolConnectorClosure(
              graph, proposal.exactCoverage, limits, workBudget);
      if (!connector) {
        result.error = connector.error;
        result.invalidIndex = connector.invalidIndex;
        result.proposals.clear();
        return result;
      }
      SyncCoverDemandSet residualQueries = connector.coveredByWorld.front();
      residualQueries.subtract(proposal.exactCoverage);
      if (!residualQueries.empty()) {
        const SyncCoverExactWorld world{{0}};
        const SyncCoverProtocolCoverageResult exact =
            computeSyncCoverProtocolExactWorldsForDemands(
                graph, target, proposal.protocols, {world}, residualQueries,
                limits, workBudget);
        if (!exact) {
          result.error = exact.error;
          result.invalidIndex = exact.invalidIndex;
          result.proposals.clear();
          return result;
        }
        proposal.exactCoverage.unite(exact.coveredByWorld.front());
      }
    }
    SyncCoverDemandSet missingSeeds(graph.getDemands().size());
    for (SyncCoverDemandId seed : proposal.seedDemands) {
      if (!proposal.exactCoverage.contains(seed)) {
        missingSeeds.insert(seed);
      }
    }
    if (!missingSeeds.empty()) {
      const SyncCoverExactWorld world{{0}};
      const SyncCoverProtocolCoverageResult exact =
          computeSyncCoverProtocolExactWorldsForDemands(
              graph, target, proposal.protocols, {world}, missingSeeds, limits,
              workBudget);
      if (!exact) {
        result.error = exact.error;
        result.invalidIndex = exact.invalidIndex;
        result.proposals.clear();
        return result;
      }
      proposal.exactCoverage.unite(exact.coveredByWorld.front());
    }
    std::size_t coveredSeeds = 0;
    for (SyncCoverDemandId seed : proposal.seedDemands) {
      if (proposal.exactCoverage.contains(seed)) {
        ++coveredSeeds;
      } else {
        ++result.lifecycleSccsWithoutCoverage;
      }
    }
    if (coveredSeeds == 0) {
      continue;
    }
    SyncCoverDemandSet extra = proposal.exactCoverage;
    extra.subtract(directCoverage[proposalIndex]);
    proposal.extraCoverageRows = extra.count();
    const std::optional<LifecycleCatalogFootprint> footprint =
        getLifecycleCatalogFootprint(proposal);
    if (!footprint) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = proposalIndex;
      result.proposals.clear();
      return result;
    }
    LifecycleCatalogFootprint next = retainedCatalog;
    const bool fits =
        checkedAccumulateSize(next.protocols, footprint->protocols) &&
        checkedAccumulateSize(next.lanes, footprint->lanes) &&
        checkedAccumulateSize(next.slots, footprint->slots) &&
        checkedAccumulateSize(next.transfers, footprint->transfers) &&
        checkedAccumulateSize(next.nodeReferences, footprint->nodeReferences) &&
        checkedAccumulateSize(next.accessIncidences,
                              footprint->accessIncidences) &&
        checkedAccumulateSize(next.payloadIncidences,
                              footprint->payloadIncidences) &&
        next.protocols <= limits.maximumLifecycleProtocols &&
        next.lanes <= limits.maximumLifecycleLanes &&
        next.slots <= limits.maximumLifecycleSlots &&
        next.transfers <= limits.maximumLifecycleTransfers &&
        next.nodeReferences <= limits.maximumLifecycleNodeReferences &&
        next.accessIncidences <= limits.maximumLifecycleAccessIncidences &&
        next.payloadIncidences <=
            limits.maximumLifecycleCatalogPayloadIncidences;
    if (!fits || !consumeWork(workBudget, footprint->payloadIncidences)) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = proposalIndex;
      result.proposals.clear();
      return result;
    }
    retainedCatalog = next;
    proposal.id = groundedProposals.size();
    groundedPhysicalKeys.push_back(
        std::move(retainedProtocolPhysicalKeys[proposalIndex]));
    groundedProposals.push_back(std::move(proposal));
  }
  result.proposals = std::move(groundedProposals);
  retainedProtocolKeys.clear();
  for (std::size_t index = 0; index < groundedPhysicalKeys.size(); ++index) {
    retainedProtocolKeys.emplace(std::move(groundedPhysicalKeys[index]), index);
  }
  if (!enableCutFusion) {
    return result;
  }
  // Direct obligations in one loop and directed resource pair may describe
  // several non-overlapping cuts of the same physical ready/release channel.
  // Fuse the complete group into one candidate recipe and let the token
  // automaton reject any ordering, phase, guard, or rearm conflict. This is a
  // generic cut grouping: singleton direct protocols remain the completeness
  // basis, while an admitted fused protocol consumes only one event use per
  // direction and can cover all of its seed demands.
  for (const auto &[key, entries] : groupedCertificates) {
    if (entries.size() < 2) {
      continue;
    }
    if (!consumeWork(workBudget, entries.size())) {
      fail(SyncCoverProtocolError::WorkLimitExceeded, entries.front().first);
      return result;
    }
    DerivedLifecycleCertificate grouped = entries.front().second;
    grouped.paths.clear();
    DerivedLifecyclePath path;
    path.scope = key.loop;
    for (const auto &[demandId, certificate] : entries) {
      if (certificate.lanes.size() != 1 || certificate.paths.size() != 1 ||
          certificate.paths.front().scope != key.loop ||
          certificate.loopScope != key.loop ||
          certificate.producerResource != key.producer ||
          certificate.consumerResource != key.consumer) {
        path.uses.clear();
        break;
      }
      (void)demandId;
      path.uses.insert(path.uses.end(), certificate.paths.front().uses.begin(),
                       certificate.paths.front().uses.end());
    }
    if (path.uses.size() < 2) {
      continue;
    }
    const std::optional<std::size_t> sortWork =
        comparisonWork(path.uses.size());
    if (!sortWork || !consumeWork(workBudget, *sortWork)) {
      fail(SyncCoverProtocolError::WorkLimitExceeded, entries.front().first);
      return result;
    }
    const auto anchorKey = [](const SyncCoverAnchor &anchor) {
      return std::tie(anchor.kind, anchor.node, anchor.scope, anchor.position);
    };
    const auto samePhysicalUse = [&](const DerivedLifecycleUse &left,
                                     const DerivedLifecycleUse &right) {
      return left.lane == right.lane &&
             left.producerLane == right.producerLane &&
             anchorKey(left.writeAcquireAnchor) ==
                 anchorKey(right.writeAcquireAnchor) &&
             anchorKey(left.readyAnchor) == anchorKey(right.readyAnchor) &&
             anchorKey(left.readAcquireAnchor) ==
                 anchorKey(right.readAcquireAnchor) &&
             anchorKey(left.releaseAnchor) == anchorKey(right.releaseAnchor);
    };
    std::sort(
        path.uses.begin(), path.uses.end(),
        [&](const DerivedLifecycleUse &left, const DerivedLifecycleUse &right) {
          return std::tie(
                     left.lane, left.producerLane, left.writeAcquireAnchor.kind,
                     left.writeAcquireAnchor.node,
                     left.writeAcquireAnchor.scope,
                     left.writeAcquireAnchor.position, left.readyAnchor.kind,
                     left.readyAnchor.node, left.readyAnchor.scope,
                     left.readyAnchor.position, left.readAcquireAnchor.kind,
                     left.readAcquireAnchor.node, left.readAcquireAnchor.scope,
                     left.readAcquireAnchor.position, left.releaseAnchor.kind,
                     left.releaseAnchor.node, left.releaseAnchor.scope,
                     left.releaseAnchor.position, left.producers,
                     left.consumers) <
                 std::tie(
                     right.lane, right.producerLane,
                     right.writeAcquireAnchor.kind,
                     right.writeAcquireAnchor.node,
                     right.writeAcquireAnchor.scope,
                     right.writeAcquireAnchor.position, right.readyAnchor.kind,
                     right.readyAnchor.node, right.readyAnchor.scope,
                     right.readyAnchor.position, right.readAcquireAnchor.kind,
                     right.readAcquireAnchor.node,
                     right.readAcquireAnchor.scope,
                     right.readAcquireAnchor.position, right.releaseAnchor.kind,
                     right.releaseAnchor.node, right.releaseAnchor.scope,
                     right.releaseAnchor.position, right.producers,
                     right.consumers);
        });
    std::vector<DerivedLifecycleUse> uniqueUses;
    uniqueUses.reserve(path.uses.size());
    for (DerivedLifecycleUse &use : path.uses) {
      if (!uniqueUses.empty() && samePhysicalUse(uniqueUses.back(), use)) {
        DerivedLifecycleUse &retained = uniqueUses.back();
        if (!consumeWork(workBudget,
                         use.producers.size() + use.consumers.size())) {
          fail(SyncCoverProtocolError::WorkLimitExceeded,
               entries.front().first);
          return result;
        }
        retained.producers.insert(retained.producers.end(),
                                  use.producers.begin(), use.producers.end());
        retained.consumers.insert(retained.consumers.end(),
                                  use.consumers.begin(), use.consumers.end());
        continue;
      }
      uniqueUses.push_back(std::move(use));
    }
    for (DerivedLifecycleUse &use : uniqueUses) {
      const auto normalizeNodes = [&](std::vector<SyncCoverNodeId> &nodes) {
        const std::optional<std::size_t> work = comparisonWork(nodes.size());
        if (!work || !consumeWork(workBudget, *work)) {
          return false;
        }
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        return true;
      };
      if (!normalizeNodes(use.producers) || !normalizeNodes(use.consumers)) {
        fail(SyncCoverProtocolError::WorkLimitExceeded, entries.front().first);
        return result;
      }
    }
    if (uniqueUses.size() < 2) {
      continue;
    }
    std::vector<std::vector<SyncCoverDemandId>> seedsByUse(uniqueUses.size());
    for (const auto &[demandId, certificate] : entries) {
      for (const DerivedLifecycleUse &use : certificate.paths.front().uses) {
        const auto found =
            std::find_if(uniqueUses.begin(), uniqueUses.end(),
                         [&](const DerivedLifecycleUse &candidate) {
                           return samePhysicalUse(candidate, use);
                         });
        if (!consumeWork(workBudget, uniqueUses.size()) ||
            found == uniqueUses.end()) {
          fail(workBudget && workBudget->exhausted
                   ? SyncCoverProtocolError::WorkLimitExceeded
                   : SyncCoverProtocolError::InvalidProtocol,
               demandId);
          return result;
        }
        seedsByUse[std::distance(uniqueUses.begin(), found)].push_back(
            demandId);
      }
    }
    for (std::vector<SyncCoverDemandId> &useSeeds : seedsByUse) {
      const std::optional<std::size_t> work = comparisonWork(useSeeds.size());
      if (!work || !consumeWork(workBudget, *work)) {
        fail(SyncCoverProtocolError::WorkLimitExceeded, entries.front().first);
        return result;
      }
      std::sort(useSeeds.begin(), useSeeds.end());
      useSeeds.erase(std::unique(useSeeds.begin(), useSeeds.end()),
                     useSeeds.end());
    }

    const auto tryUseGroup = [&](const auto &useIndices) {
      DerivedLifecycleCertificate candidate = grouped;
      candidate.paths.clear();
      DerivedLifecyclePath candidatePath;
      candidatePath.scope = key.loop;
      std::vector<SyncCoverDemandId> candidateSeeds;
      for (std::size_t useIndex : useIndices) {
        if (useIndex >= uniqueUses.size() ||
            !consumeWork(workBudget, seedsByUse[useIndex].size() + 1)) {
          return fail(workBudget && workBudget->exhausted
                          ? SyncCoverProtocolError::WorkLimitExceeded
                          : SyncCoverProtocolError::InvalidProtocol,
                      entries.front().first);
        }
        candidatePath.uses.push_back(uniqueUses[useIndex]);
        candidateSeeds.insert(candidateSeeds.end(),
                              seedsByUse[useIndex].begin(),
                              seedsByUse[useIndex].end());
      }
      candidate.paths.push_back(std::move(candidatePath));
      const SyncCoverProtocolError bounds =
          checkLifecycleCertificateBounds(candidate, limits, workBudget);
      if (bounds != SyncCoverProtocolError::None) {
        return fail(bounds, entries.front().first);
      }
      ++result.derivedLifecycleCertificates;
      std::optional<SyncCoverEventProtocol> protocol =
          makeLifecycleProtocol(graph, target, candidate, 0, workBudget);
      if (workBudget && workBudget->exhausted) {
        return fail(SyncCoverProtocolError::WorkLimitExceeded,
                    entries.front().first);
      }
      if (!protocol) {
        ++result.rejectedLifecycleConstructions;
        return true;
      }
      return retainVerifiedProtocol(std::move(*protocol),
                                    std::move(candidateSeeds),
                                    entries.front().first);
    };

    // The maximal cut group is the only bounded synthesis trial here. Smaller
    // compatible subsets belong in an indexed frontier generator; enumerating
    // all pairs makes direct-catalog construction quadratic on large kernels.
    std::vector<std::size_t> maximalGroup(uniqueUses.size());
    std::iota(maximalGroup.begin(), maximalGroup.end(), 0);
    if (!tryUseGroup(maximalGroup)) {
      return result;
    }
  }
  return result;
}

SyncCoverLifecycleSynthesisResult
mlir::pto::synthesizeSyncCoverLifecycleCertificates(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  const SyncCoverLifecycleSccResult components =
      discoverSyncCoverLifecycleSccs(graph, limits, workBudget);
  if (!components) {
    SyncCoverLifecycleSynthesisResult result;
    result.error = components.error;
    result.invalidIndex = components.invalidIndex;
    return result;
  }
  return synthesizeLifecycleCertificatesFromSccs(graph, target, components,
                                                 limits, workBudget);
}

namespace {

std::optional<std::vector<std::size_t>>
proposalPhysicalKey(const SyncCoverLifecycleProposal &proposal,
                    SyncCoverProtocolLimits limits,
                    SyncCoverCoverageWorkBudget *workBudget) {
  std::vector<std::size_t> key;
  key.push_back(proposal.protocols.size());
  for (const SyncCoverEventProtocol &protocol : proposal.protocols) {
    std::optional<std::vector<std::size_t>> protocolKey =
        protocolPhysicalKey(protocol, limits, workBudget);
    if (!protocolKey ||
        !consumeWork(workBudget, protocolKey->size() + std::size_t{1})) {
      return std::nullopt;
    }
    key.push_back(protocolKey->size());
    key.insert(key.end(), protocolKey->begin(), protocolKey->end());
  }
  return key;
}

bool lifecycleSlotEqual(const SyncCoverLifecycleSlot &left,
                        const SyncCoverLifecycleSlot &right) {
  return left.domain == right.domain &&
         left.extent.begin == right.extent.begin &&
         left.extent.end == right.extent.end && left.accesses == right.accesses;
}

std::optional<bool>
lifecycleProposalEqual(const SyncCoverLifecycleProposal &left,
                       const std::vector<std::size_t> &leftKey,
                       const SyncCoverLifecycleProposal &right,
                       const std::vector<std::size_t> &rightKey,
                       SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t work = 0;
  const std::size_t fixedIncidences[] = {
      leftKey.size(),
      rightKey.size(),
      left.seedDemands.size(),
      right.seedDemands.size(),
      left.slots.size(),
      right.slots.size(),
      left.exactCoverage.getWords().size(),
      right.exactCoverage.getWords().size(),
  };
  for (std::size_t incidences : fixedIncidences) {
    if (!checkedAccumulateSize(work, incidences)) {
      return std::nullopt;
    }
  }
  for (const SyncCoverLifecycleSlot &slot : left.slots) {
    if (!checkedAccumulateSize(work, slot.accesses.size())) {
      return std::nullopt;
    }
  }
  for (const SyncCoverLifecycleSlot &slot : right.slots) {
    if (!checkedAccumulateSize(work, slot.accesses.size())) {
      return std::nullopt;
    }
  }
  if (!consumeWork(workBudget, work)) {
    return std::nullopt;
  }
  return leftKey == rightKey && left.directBasis == right.directBasis &&
         left.seedDemands == right.seedDemands &&
         left.exactCoverage == right.exactCoverage &&
         left.singletonUnionCoverageRows == right.singletonUnionCoverageRows &&
         left.extraCoverageRows == right.extraCoverageRows &&
         left.slots.size() == right.slots.size() &&
         std::equal(left.slots.begin(), left.slots.end(), right.slots.begin(),
                    lifecycleSlotEqual);
}

} // namespace

SyncCoverProtocolError mlir::pto::verifySyncCoverLifecycleProposalsFresh(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverStorageLifecycleIndex *lifecycleIndex,
    const SyncCoverStorageProtocolSeedIndex *seedIndex,
    const SyncCoverStorageProtocolGroupIndex *groupIndex,
    const std::vector<SyncCoverLifecycleProposal> &frozenProposals,
    const std::vector<std::size_t> &selectedProposalIndices,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget,
    bool enableDirectCutFusion) {
  if (!graph.isStructureFrozen() || !graph.validate() ||
      selectedProposalIndices.empty()) {
    return SyncCoverProtocolError::InvalidGraph;
  }
  SyncCoverLifecycleSynthesisResult grouped;
  if (lifecycleIndex && seedIndex && groupIndex &&
      lifecycleIndex->isComplete() && seedIndex->isComplete() &&
      groupIndex->isComplete()) {
    grouped = synthesizeSyncCoverLifecycleCertificates(
        graph, target, *lifecycleIndex, *seedIndex, *groupIndex, limits,
        workBudget);
  } else if (lifecycleIndex && lifecycleIndex->isComplete()) {
    grouped = synthesizeSyncCoverLifecycleCertificates(
        graph, target, *lifecycleIndex, limits, workBudget);
  } else {
    grouped = synthesizeSyncCoverLifecycleCertificates(graph, target, limits,
                                                       workBudget);
  }
  if (!grouped) {
    return grouped.error;
  }
  SyncCoverLifecycleSynthesisResult direct =
      synthesizeSyncCoverBalancedDirectProtocols(graph, target, limits,
                                                 workBudget,
                                                 enableDirectCutFusion);
  if (!direct) {
    return direct.error;
  }

  std::vector<const SyncCoverLifecycleProposal *> rebuilt;
  rebuilt.reserve(grouped.proposals.size() + direct.proposals.size());
  for (const SyncCoverLifecycleProposal &proposal : grouped.proposals) {
    rebuilt.push_back(&proposal);
  }
  for (const SyncCoverLifecycleProposal &proposal : direct.proposals) {
    rebuilt.push_back(&proposal);
  }
  if (!consumeWork(workBudget,
                   rebuilt.size() + selectedProposalIndices.size())) {
    return SyncCoverProtocolError::WorkLimitExceeded;
  }
  std::vector<std::optional<std::vector<std::size_t>>> rebuiltKeys;
  rebuiltKeys.reserve(rebuilt.size());
  for (const SyncCoverLifecycleProposal *proposal : rebuilt) {
    rebuiltKeys.push_back(proposalPhysicalKey(*proposal, limits, workBudget));
    if (!rebuiltKeys.back()) {
      return workBudget && workBudget->exhausted
                 ? SyncCoverProtocolError::WorkLimitExceeded
                 : SyncCoverProtocolError::LimitExceeded;
    }
  }
  std::vector<bool> used(rebuilt.size());
  for (std::size_t frozenIndex : selectedProposalIndices) {
    if (!consumeWork(workBudget) || frozenIndex >= frozenProposals.size()) {
      return workBudget && workBudget->exhausted
                 ? SyncCoverProtocolError::WorkLimitExceeded
                 : SyncCoverProtocolError::InvalidProtocol;
    }
    const SyncCoverLifecycleProposal &frozen = frozenProposals[frozenIndex];
    std::optional<std::vector<std::size_t>> frozenKey =
        proposalPhysicalKey(frozen, limits, workBudget);
    if (!frozenKey) {
      return workBudget && workBudget->exhausted
                 ? SyncCoverProtocolError::WorkLimitExceeded
                 : SyncCoverProtocolError::LimitExceeded;
    }
    bool matched = false;
    for (std::size_t rebuiltIndex = 0; rebuiltIndex < rebuilt.size();
         ++rebuiltIndex) {
      if (!consumeWork(workBudget) || used[rebuiltIndex]) {
        if (workBudget && workBudget->exhausted) {
          return SyncCoverProtocolError::WorkLimitExceeded;
        }
        continue;
      }
      const std::optional<bool> same =
          lifecycleProposalEqual(frozen, *frozenKey, *rebuilt[rebuiltIndex],
                                 *rebuiltKeys[rebuiltIndex], workBudget);
      if (!same) {
        return workBudget && workBudget->exhausted
                   ? SyncCoverProtocolError::WorkLimitExceeded
                   : SyncCoverProtocolError::LimitExceeded;
      }
      if (*same) {
        used[rebuiltIndex] = true;
        matched = true;
        break;
      }
      if (workBudget && workBudget->exhausted) {
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
    }
    if (!matched) {
      return SyncCoverProtocolError::InvalidProtocol;
    }
  }
  return SyncCoverProtocolError::None;
}

SyncCoverLifecycleSynthesisResult
mlir::pto::synthesizeSyncCoverLifecycleCertificates(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolSeedIndex &seedIndex,
    const SyncCoverStorageProtocolGroupIndex &groupIndex,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  const SyncCoverLifecycleSccResult components = importStorageLifecycleGroups(
      graph, lifecycleIndex, seedIndex, groupIndex, limits, workBudget);
  if (!components) {
    SyncCoverLifecycleSynthesisResult result;
    result.error = components.error;
    result.invalidIndex = components.invalidIndex;
    return result;
  }
  return synthesizeLifecycleCertificatesFromSccs(graph, target, components,
                                                 limits, workBudget);
}

SyncCoverLifecycleSynthesisResult
mlir::pto::synthesizeSyncCoverLifecycleCertificates(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  const SyncCoverLifecycleSccResult components =
      importStorageLifecycleSccs(graph, lifecycleIndex, limits, workBudget);
  if (!components) {
    SyncCoverLifecycleSynthesisResult result;
    result.error = components.error;
    result.invalidIndex = components.invalidIndex;
    return result;
  }
  return synthesizeLifecycleCertificatesFromSccs(graph, target, components,
                                                 limits, workBudget);
}
