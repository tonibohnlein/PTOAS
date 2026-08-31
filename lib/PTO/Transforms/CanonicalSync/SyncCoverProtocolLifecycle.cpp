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

#include <algorithm>
#include <limits>
#include <optional>
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
  std::vector<SyncCoverStorageDomainId> domains;
};

std::optional<LifecycleEdge>
makeLifecycleEdge(const SyncCoverGraph &graph, SyncCoverDemandId demandId,
                  std::vector<std::size_t> &marks, std::size_t stamp,
                  SyncCoverProtocolLimits limits,
                  std::size_t &witnessIncidences, std::size_t &domainIncidences,
                  SyncCoverCoverageWorkBudget *workBudget,
                  SyncCoverProtocolError &error) {
  const SyncCoverDemand &demand = graph.getDemands()[demandId];
  std::vector<SyncCoverStorageDomainId> domains;
  for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
    if (witnessIncidences == limits.maximumStorageWitnessIncidences ||
        !consumeWork(workBudget)) {
      error = workBudget && workBudget->exhausted
                  ? SyncCoverProtocolError::WorkLimitExceeded
                  : SyncCoverProtocolError::LimitExceeded;
      return std::nullopt;
    }
    ++witnessIncidences;
    const SyncCoverStorageWitness &witness =
        graph.getStorageWitnesses()[witnessId];
    const SyncCoverStorageAccess &source =
        graph.getStorageAccesses()[witness.sourceAccess];
    const SyncCoverStorageAccess &target =
        graph.getStorageAccesses()[witness.targetAccess];
    if (source.domain == target.domain && source.exactPhysical &&
        target.exactPhysical && marks[source.domain] != stamp) {
      if (domainIncidences == limits.maximumLifecycleDomainIncidences) {
        error = SyncCoverProtocolError::LimitExceeded;
        return std::nullopt;
      }
      marks[source.domain] = stamp;
      domains.push_back(source.domain);
      ++domainIncidences;
    }
  }
  if (domains.empty()) {
    return std::nullopt;
  }
  return LifecycleEdge{demandId, demand.source, demand.target,
                       std::move(domains)};
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
  const bool graphLimitExceeded = !graphFitsProtocolLimits(graph, limits);
  if (graphLimitExceeded) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  const bool invalidGraph = !graph.isStructureFrozen() || !graph.validate();
  if (invalidGraph) {
    result.error = SyncCoverProtocolError::InvalidGraph;
    return result;
  }
  const bool tooManyNodes =
      graph.getNodes().size() > limits.maximumLifecycleVertices;
  const bool tooManyMetadata =
      graph.getDemands().size() > limits.maximumResultRows ||
      graph.getStorageDomains().size() >
          limits.maximumLifecycleDomainIncidences;
  if (tooManyNodes || tooManyMetadata) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }

  std::vector<LifecycleEdge> edges;
  std::vector<std::vector<std::size_t>> adjacency(graph.getNodes().size());
  std::vector<std::vector<std::size_t>> reverse(graph.getNodes().size());
  std::vector<std::size_t> domainMarks(graph.getStorageDomains().size());
  std::size_t witnessIncidences = 0;
  std::size_t domainIncidences = 0;
  for (SyncCoverDemandId demand = 0; demand < graph.getDemands().size();
       ++demand) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = demand;
      return result;
    }
    SyncCoverProtocolError edgeError = SyncCoverProtocolError::None;
    std::optional<LifecycleEdge> edge = makeLifecycleEdge(
        graph, demand, domainMarks, demand + 1, limits, witnessIncidences,
        domainIncidences, workBudget, edgeError);
    if (edgeError != SyncCoverProtocolError::None) {
      result.error = edgeError;
      result.invalidIndex = demand;
      return result;
    }
    if (!edge) {
      continue;
    }
    const bool tooManyEdges = edges.size() == limits.maximumLifecycleEdges;
    if (tooManyEdges) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = demand;
      return result;
    }
    adjacency[edge->source].push_back(edge->target);
    reverse[edge->target].push_back(edge->source);
    edges.push_back(std::move(*edge));
  }

  std::vector<std::size_t> order;
  order.reserve(graph.getNodes().size());
  if (!finishOrder(adjacency, order, workBudget)) {
    result.error = SyncCoverProtocolError::WorkLimitExceeded;
    return result;
  }
  std::vector<bool> visited(graph.getNodes().size());
  std::vector<std::size_t> componentOf(graph.getNodes().size(),
                                       std::numeric_limits<std::size_t>::max());
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
      return result;
    }
    for (std::size_t node : component) {
      componentOf[node] = componentCount;
    }
    ++componentCount;
  }

  struct ComponentFacts {
    SyncCoverLifecycleScc lifecycle;
    std::vector<SyncCoverStorageDomainId> domains;
    std::optional<SyncCoverScopeId> loopScope;
    bool incompatibleLoops = false;
  };
  std::vector<ComponentFacts> facts(componentCount);
  for (SyncCoverNodeId node = 0; node < componentOf.size(); ++node) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    facts[componentOf[node]].lifecycle.nodes.push_back(node);
  }
  std::vector<std::vector<std::size_t>> componentEdges(componentCount);
  for (std::size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    const LifecycleEdge &edge = edges[edgeIndex];
    const std::size_t component = componentOf[edge.source];
    if (component != componentOf[edge.target]) {
      continue;
    }
    componentEdges[component].push_back(edgeIndex);
  }
  std::vector<std::size_t> componentDomainMarks(
      graph.getStorageDomains().size());
  for (std::size_t component = 0; component < componentCount; ++component) {
    ComponentFacts &componentFacts = facts[component];
    for (std::size_t edgeIndex : componentEdges[component]) {
      const LifecycleEdge &edge = edges[edgeIndex];
      const bool workUnavailable =
          !consumeWork(workBudget, edge.domains.size() + 1);
      if (workUnavailable) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        return result;
      }
      componentFacts.lifecycle.demands.push_back(edge.demand);
      for (SyncCoverStorageDomainId domain : edge.domains) {
        if (componentDomainMarks[domain] != component + 1) {
          componentDomainMarks[domain] = component + 1;
          componentFacts.domains.push_back(domain);
        }
      }
      const SyncCoverDemand &demand = graph.getDemands()[edge.demand];
      componentFacts.lifecycle.maximumDistance =
          std::max(componentFacts.lifecycle.maximumDistance, demand.distance);
      if (demand.distance != 0) {
        if (componentFacts.loopScope &&
            *componentFacts.loopScope != demand.scope) {
          componentFacts.incompatibleLoops = true;
        }
        componentFacts.loopScope = demand.scope;
      }
    }
  }
  std::vector<std::optional<std::size_t>> componentAtMinimum(
      graph.getNodes().size());
  for (std::size_t component = 0; component < facts.size(); ++component) {
    if (!facts[component].lifecycle.nodes.empty()) {
      componentAtMinimum[facts[component].lifecycle.nodes.front()] = component;
    }
  }
  for (const std::optional<std::size_t> &component : componentAtMinimum) {
    if (!component) {
      continue;
    }
    ComponentFacts &componentFacts = facts[*component];
    SyncCoverLifecycleScc &lifecycle = componentFacts.lifecycle;
    if (lifecycle.maximumDistance == 0 || !componentFacts.loopScope ||
        componentFacts.incompatibleLoops) {
      continue;
    }
    lifecycle.loopScope = *componentFacts.loopScope;
    lifecycle.storageDomains = std::move(componentFacts.domains);
    const bool tooManyComponents =
        result.components.size() == limits.maximumLifecycleSccs;
    if (tooManyComponents) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      return result;
    }
    result.components.push_back(std::move(lifecycle));
  }
  return result;
}
