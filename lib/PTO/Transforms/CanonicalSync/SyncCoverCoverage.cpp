// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir::pto;

namespace {

constexpr unsigned kStaticCopy = std::numeric_limits<unsigned>::max();
constexpr std::size_t kMaxVirtualNodes = 1U << 20;

struct ContextLiteral {
  SyncCoverControlId control = 0;
  unsigned copy = 0;
  unsigned alternative = 0;

  bool operator<(const ContextLiteral &other) const {
    return std::tie(control, copy, alternative) <
           std::tie(other.control, other.copy, other.alternative);
  }

  bool operator==(const ContextLiteral &other) const {
    return control == other.control && copy == other.copy &&
           alternative == other.alternative;
  }
};

struct VirtualEdge {
  std::size_t source = 0;
  std::size_t target = 0;
  SyncCoverEdgeKind kind = SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  std::optional<SyncCoverMechanismId> mechanism;
};

struct Predecessor {
  std::size_t state = std::numeric_limits<std::size_t>::max();
  std::optional<SyncCoverMechanismId> mechanism;
};

struct PreparedDemand {
  SyncCoverCoverageError error = SyncCoverCoverageError::None;
  std::size_t nodeCount = 0;
  std::size_t stateCount = 0;
  std::size_t start = 0;
  std::size_t target = 0;
  std::vector<VirtualEdge> edges;
  std::vector<std::vector<std::size_t>> outgoing;
  std::vector<SyncCoverMechanismId> potentialMechanisms;
};

unsigned contextCopy(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                     SyncCoverControlId control, unsigned copy) {
  const SyncCoverScopeId controlScope = graph.getControls()[control].scope;
  const bool perIteration =
      demand.distance != 0 && graph.scopeContains(demand.scope, controlScope);
  return perIteration ? copy : kStaticCopy;
}

bool appendGuard(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                 const SyncCoverGuard &guard, unsigned copy,
                 std::vector<ContextLiteral> &context) {
  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    context.push_back({literal.control,
                       contextCopy(graph, demand, literal.control, copy),
                       literal.alternative});
  }
  std::sort(context.begin(), context.end());
  context.erase(std::unique(context.begin(), context.end()), context.end());
  for (std::size_t index = 1; index < context.size(); ++index) {
    const ContextLiteral &previous = context[index - 1];
    const ContextLiteral &current = context[index];
    if (previous.control == current.control && previous.copy == current.copy &&
        previous.alternative != current.alternative) {
      return false;
    }
  }
  return true;
}

bool guardIsImplied(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                    const std::vector<ContextLiteral> &condition,
                    const SyncCoverEdge &edge, unsigned sourceCopy,
                    unsigned targetCopy) {
  std::vector<ContextLiteral> required;
  const bool sourceValid =
      appendGuard(graph, demand, edge.sourceGuard, sourceCopy, required);
  const bool targetValid =
      appendGuard(graph, demand, edge.targetGuard, targetCopy, required);
  if (!sourceValid || !targetValid) {
    return false;
  }
  return std::includes(condition.begin(), condition.end(), required.begin(),
                       required.end());
}

bool nodeInstanceAvailable(const SyncCoverGraph &graph,
                           const SyncCoverDemand &demand, SyncCoverNodeId node,
                           unsigned copy) {
  const bool isSource = node == demand.source && copy == 0;
  const bool isTarget = node == demand.target && copy == demand.distance;
  if (isSource || isTarget) {
    return true;
  }
  const SyncCoverScopeId nodeScope = graph.getNodes()[node].scope;
  if (graph.scopeMustExecuteWithin(demand.scope, nodeScope)) {
    return true;
  }

  // An optional scope is available when execution of either endpoint in the
  // same virtual copy implies that scope. This mirrors the legacy
  // condition-union rule without assuming an optional path executes in an
  // unrelated recurrence copy.
  const SyncCoverScopeId sourceScope = graph.getNodes()[demand.source].scope;
  if (copy == 0 && graph.scopeExecutesWhen(sourceScope, nodeScope)) {
    return true;
  }
  const SyncCoverScopeId targetScope = graph.getNodes()[demand.target].scope;
  return copy == demand.distance &&
         graph.scopeExecutesWhen(targetScope, nodeScope);
}

std::vector<VirtualEdge>
buildVirtualEdges(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                  const std::vector<ContextLiteral> &condition) {
  std::vector<VirtualEdge> result;
  const std::size_t nodeCount = graph.getNodes().size();
  for (const SyncCoverEdge &edge : graph.getEdges()) {
    if (edge.distance > demand.distance) {
      continue;
    }
    if (edge.distance != 0 && edge.scope != demand.scope) {
      continue;
    }
    for (unsigned sourceCopy = 0; sourceCopy + edge.distance <= demand.distance;
         ++sourceCopy) {
      const unsigned targetCopy = sourceCopy + edge.distance;
      const bool sourceAvailable =
          nodeInstanceAvailable(graph, demand, edge.source, sourceCopy);
      const bool targetAvailable =
          nodeInstanceAvailable(graph, demand, edge.target, targetCopy);
      if (!sourceAvailable || !targetAvailable) {
        continue;
      }
      if (!guardIsImplied(graph, demand, condition, edge, sourceCopy,
                          targetCopy)) {
        continue;
      }
      result.push_back({sourceCopy * nodeCount + edge.source,
                        targetCopy * nodeCount + edge.target, edge.kind,
                        edge.mechanism});
    }
  }
  return result;
}

std::size_t stateIndex(std::size_t virtualNode, bool completion) {
  return virtualNode * 2 + static_cast<std::size_t>(completion);
}

std::optional<std::size_t> transition(const VirtualEdge &edge,
                                      std::size_t state) {
  const bool completion = (state % 2) != 0;
  switch (edge.kind) {
  case SyncCoverEdgeKind::CompletionSupply:
    return stateIndex(edge.target, true);
  case SyncCoverEdgeKind::CompletionPreservingIssueOrder:
    return completion
               ? std::optional<std::size_t>(stateIndex(edge.target, true))
               : std::nullopt;
  case SyncCoverEdgeKind::NonCompletionPreservingIssueOrder:
    return std::nullopt;
  }
  return std::nullopt;
}

std::vector<SyncCoverMechanismId>
collectPotentialMechanisms(const PreparedDemand &prepared) {
  const std::size_t virtualNodeCount = prepared.stateCount / 2;
  std::vector<std::vector<std::size_t>> incoming(virtualNodeCount);
  for (std::size_t edge = 0; edge < prepared.edges.size(); ++edge) {
    incoming[prepared.edges[edge].target].push_back(edge);
  }

  std::vector<bool> reachableFromSource(virtualNodeCount, false);
  std::vector<std::size_t> ready{prepared.start / 2};
  reachableFromSource[ready.front()] = true;
  for (std::size_t index = 0; index < ready.size(); ++index) {
    for (std::size_t edge : prepared.outgoing[ready[index]]) {
      const std::size_t target = prepared.edges[edge].target;
      if (!reachableFromSource[target]) {
        reachableFromSource[target] = true;
        ready.push_back(target);
      }
    }
  }

  std::vector<bool> reachesTarget(virtualNodeCount, false);
  ready = {prepared.target / 2};
  reachesTarget[ready.front()] = true;
  for (std::size_t index = 0; index < ready.size(); ++index) {
    for (std::size_t edge : incoming[ready[index]]) {
      const std::size_t source = prepared.edges[edge].source;
      if (!reachesTarget[source]) {
        reachesTarget[source] = true;
        ready.push_back(source);
      }
    }
  }

  std::vector<SyncCoverMechanismId> result;
  for (const VirtualEdge &edge : prepared.edges) {
    if (edge.mechanism && reachableFromSource[edge.source] &&
        reachesTarget[edge.target]) {
      result.push_back(*edge.mechanism);
    }
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

bool mechanismSelected(const std::vector<SyncCoverMechanismId> &selected,
                       const std::optional<SyncCoverMechanismId> &mechanism) {
  return !mechanism ||
         std::binary_search(selected.begin(), selected.end(), *mechanism);
}

SyncCoverCoverageResult makeError(SyncCoverCoverageError error) {
  SyncCoverCoverageResult result;
  result.error = error;
  return result;
}

std::optional<std::size_t> getVirtualNodeCount(const SyncCoverGraph &graph,
                                               const SyncCoverDemand &demand) {
  const std::size_t nodeCount = graph.getNodes().size();
  const std::size_t copyCount = static_cast<std::size_t>(demand.distance) + 1;
  const bool invalidProduct = copyCount == 0 || nodeCount == 0 ||
                              copyCount > kMaxVirtualNodes / nodeCount;
  if (invalidProduct) {
    return std::nullopt;
  }
  const std::size_t virtualNodeCount = copyCount * nodeCount;
  const bool invalidStateCount =
      virtualNodeCount > kMaxVirtualNodes ||
      virtualNodeCount > std::numeric_limits<std::size_t>::max() / 2;
  if (invalidStateCount) {
    return std::nullopt;
  }
  return virtualNodeCount;
}

PreparedDemand prepareDemand(const SyncCoverGraph &graph,
                             SyncCoverDemandId demandId) {
  PreparedDemand prepared;
  const SyncCoverDemand &demand = graph.getDemands()[demandId];
  std::vector<ContextLiteral> condition;
  const bool sourceValid =
      appendGuard(graph, demand, demand.sourceGuard, 0, condition);
  const bool targetValid = appendGuard(graph, demand, demand.targetGuard,
                                       demand.distance, condition);
  if (!sourceValid || !targetValid) {
    prepared.error = SyncCoverCoverageError::InvalidGraph;
    return prepared;
  }

  const std::optional<std::size_t> virtualNodeCount =
      getVirtualNodeCount(graph, demand);
  if (!virtualNodeCount) {
    prepared.error = SyncCoverCoverageError::ExpansionLimitExceeded;
    return prepared;
  }
  prepared.nodeCount = graph.getNodes().size();
  prepared.edges = buildVirtualEdges(graph, demand, condition);
  prepared.outgoing.resize(*virtualNodeCount);
  for (std::size_t index = 0; index < prepared.edges.size(); ++index) {
    const VirtualEdge &edge = prepared.edges[index];
    prepared.outgoing[edge.source].push_back(index);
  }

  prepared.start = stateIndex(demand.source, false);
  prepared.target = stateIndex(static_cast<std::size_t>(demand.distance) *
                                       prepared.nodeCount +
                                   demand.target,
                               true);
  prepared.stateCount = *virtualNodeCount * 2;
  prepared.potentialMechanisms = collectPotentialMechanisms(prepared);
  return prepared;
}

SyncCoverCoverageResult
checkDemandImpl(const PreparedDemand &prepared,
                const std::vector<SyncCoverMechanismId> &selected) {
  if (prepared.error != SyncCoverCoverageError::None) {
    return makeError(prepared.error);
  }
  std::vector<bool> reachable(prepared.stateCount, false);
  std::vector<Predecessor> predecessors(prepared.stateCount);
  std::vector<std::size_t> ready{prepared.start};
  reachable[prepared.start] = true;
  predecessors[prepared.start].state = prepared.start;

  for (std::size_t readyIndex = 0; readyIndex < ready.size(); ++readyIndex) {
    const std::size_t state = ready[readyIndex];
    const std::size_t virtualNode = state / 2;
    for (std::size_t edgeIndex : prepared.outgoing[virtualNode]) {
      const VirtualEdge &edge = prepared.edges[edgeIndex];
      if (!mechanismSelected(selected, edge.mechanism)) {
        continue;
      }
      const std::optional<std::size_t> next = transition(edge, state);
      if (!next || reachable[*next]) {
        continue;
      }
      reachable[*next] = true;
      predecessors[*next] = {state, edge.mechanism};
      ready.push_back(*next);
    }
  }

  SyncCoverCoverageResult result;
  result.covered = reachable[prepared.target];
  for (std::size_t state = 0; state < prepared.stateCount; ++state) {
    if (!reachable[state]) {
      continue;
    }
    const std::size_t virtualNode = state / 2;
    result.reachableStates.push_back(
        {virtualNode % prepared.nodeCount,
         static_cast<unsigned>(virtualNode / prepared.nodeCount),
         (state % 2) != 0});
  }
  std::sort(result.reachableStates.begin(), result.reachableStates.end());

  if (result.covered) {
    for (std::size_t state = prepared.target; state != prepared.start;
         state = predecessors[state].state) {
      if (predecessors[state].mechanism) {
        result.witnessMechanisms.push_back(*predecessors[state].mechanism);
      }
    }
    std::sort(result.witnessMechanisms.begin(), result.witnessMechanisms.end());
    result.witnessMechanisms.erase(std::unique(result.witnessMechanisms.begin(),
                                               result.witnessMechanisms.end()),
                                   result.witnessMechanisms.end());
    return result;
  }

  for (const VirtualEdge &edge : prepared.edges) {
    if (!edge.mechanism || mechanismSelected(selected, edge.mechanism)) {
      continue;
    }
    for (bool completion : {false, true}) {
      const std::size_t sourceState = stateIndex(edge.source, completion);
      if (!reachable[sourceState]) {
        continue;
      }
      const std::optional<std::size_t> next = transition(edge, sourceState);
      if (next && !reachable[*next]) {
        result.cutMechanisms.push_back(*edge.mechanism);
      }
    }
  }
  std::sort(result.cutMechanisms.begin(), result.cutMechanisms.end());
  result.cutMechanisms.erase(
      std::unique(result.cutMechanisms.begin(), result.cutMechanisms.end()),
      result.cutMechanisms.end());
  return result;
}

std::vector<SyncCoverMechanismId>
normalizeSelection(const std::vector<SyncCoverMechanismId> &selected) {
  std::vector<SyncCoverMechanismId> result = selected;
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

} // namespace

struct SyncCoverCoverageOracle::Implementation {
  explicit Implementation(const SyncCoverGraph &source) : graph(source) {
    ++statistics.graphValidations;
    if (!graph.validate()) {
      validationError = SyncCoverCoverageError::InvalidGraph;
      return;
    }
    prepared.resize(graph.getDemands().size());
  }

  const PreparedDemand &getPreparedDemand(SyncCoverDemandId demand) const {
    std::optional<PreparedDemand> &entry = prepared[demand];
    if (!entry) {
      entry = prepareDemand(graph, demand);
      ++statistics.demandPreparations;
    }
    return *entry;
  }

  SyncCoverGraph graph;
  SyncCoverCoverageError validationError = SyncCoverCoverageError::None;
  mutable std::vector<std::optional<PreparedDemand>> prepared;
  mutable SyncCoverCoverageStatistics statistics;
};

bool SyncCoverReachableState::operator<(
    const SyncCoverReachableState &other) const {
  return std::tie(copy, node, hasCompletion) <
         std::tie(other.copy, other.node, other.hasCompletion);
}

bool SyncCoverReachableState::operator==(
    const SyncCoverReachableState &other) const {
  return node == other.node && copy == other.copy &&
         hasCompletion == other.hasCompletion;
}

SyncCoverCoverageOracle::SyncCoverCoverageOracle(const SyncCoverGraph &graph)
    : implementation_(std::make_shared<Implementation>(graph)) {}

SyncCoverCoverageResult SyncCoverCoverageOracle::checkDemand(
    SyncCoverDemandId demand,
    const std::vector<SyncCoverMechanismId> &selected) const {
  return checkDemandCanonicalSelection(demand, normalizeSelection(selected));
}

SyncCoverCoverageResult SyncCoverCoverageOracle::checkDemandCanonicalSelection(
    SyncCoverDemandId demand,
    const std::vector<SyncCoverMechanismId> &selected) const {
  if (implementation_->validationError != SyncCoverCoverageError::None) {
    return makeError(implementation_->validationError);
  }
  if (demand >= implementation_->graph.getDemands().size()) {
    return makeError(SyncCoverCoverageError::InvalidDemand);
  }
  const bool invalidSelection =
      !std::is_sorted(selected.begin(), selected.end()) ||
      std::adjacent_find(selected.begin(), selected.end()) != selected.end();
  if (invalidSelection) {
    return makeError(SyncCoverCoverageError::InvalidSelection);
  }
  ++implementation_->statistics.coverageQueries;
  return checkDemandImpl(implementation_->getPreparedDemand(demand), selected);
}

std::vector<SyncCoverCoverageResult> SyncCoverCoverageOracle::checkAll(
    const std::vector<SyncCoverMechanismId> &selected) const {
  if (implementation_->validationError != SyncCoverCoverageError::None) {
    return {makeError(implementation_->validationError)};
  }
  const std::vector<SyncCoverMechanismId> normalized =
      normalizeSelection(selected);
  std::vector<SyncCoverCoverageResult> results;
  results.reserve(implementation_->graph.getDemands().size());
  for (SyncCoverDemandId demand = 0;
       demand < implementation_->graph.getDemands().size(); ++demand) {
    ++implementation_->statistics.coverageQueries;
    results.push_back(checkDemandImpl(
        implementation_->getPreparedDemand(demand), normalized));
  }
  return results;
}

SyncCoverDemandTopologyResult
SyncCoverCoverageOracle::getDemandTopology(SyncCoverDemandId demand) const {
  SyncCoverDemandTopologyResult result;
  if (implementation_->validationError != SyncCoverCoverageError::None) {
    result.error = implementation_->validationError;
    return result;
  }
  if (demand >= implementation_->graph.getDemands().size()) {
    result.error = SyncCoverCoverageError::InvalidDemand;
    return result;
  }
  const PreparedDemand &prepared = implementation_->getPreparedDemand(demand);
  result.error = prepared.error;
  result.potentialMechanisms = prepared.potentialMechanisms;
  return result;
}

SyncCoverCoverageStatistics SyncCoverCoverageOracle::getStatistics() const {
  return implementation_->statistics;
}
