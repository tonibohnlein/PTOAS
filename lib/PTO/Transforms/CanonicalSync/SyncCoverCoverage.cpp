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

bool scopeContains(const SyncCoverGraph &graph, SyncCoverScopeId ancestor,
                   SyncCoverScopeId descendant) {
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  const bool invalidScope =
      ancestor >= scopes.size() || descendant >= scopes.size();
  if (invalidScope) {
    return false;
  }
  while (descendant != ancestor && descendant != 0) {
    descendant = scopes[descendant].parent;
  }
  return descendant == ancestor;
}

unsigned contextCopy(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                     SyncCoverControlId control, unsigned copy) {
  const SyncCoverScopeId controlScope = graph.getControls()[control].scope;
  const bool perIteration =
      demand.distance != 0 && scopeContains(graph, demand.scope, controlScope);
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

bool scopeMustExecuteWithin(const SyncCoverGraph &graph,
                            SyncCoverScopeId ancestor,
                            SyncCoverScopeId descendant) {
  if (!scopeContains(graph, ancestor, descendant)) {
    return false;
  }
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  while (descendant != ancestor) {
    if (!scopes[descendant].mustExecuteWithinParent) {
      return false;
    }
    descendant = scopes[descendant].parent;
  }
  return true;
}

bool nodeInstanceAvailable(const SyncCoverGraph &graph,
                           const SyncCoverDemand &demand, SyncCoverNodeId node,
                           unsigned copy) {
  const bool isSource = node == demand.source && copy == 0;
  const bool isTarget = node == demand.target && copy == demand.distance;
  if (isSource || isTarget) {
    return true;
  }
  return scopeMustExecuteWithin(graph, demand.scope,
                                graph.getNodes()[node].scope);
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

SyncCoverCoverageResult
checkDemandImpl(const SyncCoverGraph &graph, SyncCoverDemandId demandId,
                const std::vector<SyncCoverMechanismId> &selected) {
  const SyncCoverDemand &demand = graph.getDemands()[demandId];
  std::vector<ContextLiteral> condition;
  const bool sourceValid =
      appendGuard(graph, demand, demand.sourceGuard, 0, condition);
  const bool targetValid = appendGuard(graph, demand, demand.targetGuard,
                                       demand.distance, condition);
  if (!sourceValid || !targetValid) {
    return makeError(SyncCoverCoverageError::InvalidGraph);
  }

  const std::optional<std::size_t> virtualNodeCount =
      getVirtualNodeCount(graph, demand);
  if (!virtualNodeCount) {
    return makeError(SyncCoverCoverageError::ExpansionLimitExceeded);
  }
  const std::size_t nodeCount = graph.getNodes().size();
  const std::vector<VirtualEdge> edges =
      buildVirtualEdges(graph, demand, condition);
  std::vector<std::vector<std::size_t>> outgoing(*virtualNodeCount);
  for (std::size_t index = 0; index < edges.size(); ++index) {
    outgoing[edges[index].source].push_back(index);
  }

  const std::size_t start = stateIndex(demand.source, false);
  const std::size_t target = stateIndex(
      static_cast<std::size_t>(demand.distance) * nodeCount + demand.target,
      true);
  const std::size_t stateCount = *virtualNodeCount * 2;
  std::vector<bool> reachable(stateCount, false);
  std::vector<Predecessor> predecessors(stateCount);
  std::vector<std::size_t> ready{start};
  reachable[start] = true;
  predecessors[start].state = start;

  for (std::size_t readyIndex = 0; readyIndex < ready.size(); ++readyIndex) {
    const std::size_t state = ready[readyIndex];
    const std::size_t virtualNode = state / 2;
    for (std::size_t edgeIndex : outgoing[virtualNode]) {
      const VirtualEdge &edge = edges[edgeIndex];
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
  result.covered = reachable[target];
  for (std::size_t state = 0; state < stateCount; ++state) {
    if (!reachable[state]) {
      continue;
    }
    const std::size_t virtualNode = state / 2;
    result.reachableStates.push_back(
        {virtualNode % nodeCount,
         static_cast<unsigned>(virtualNode / nodeCount), (state % 2) != 0});
  }
  std::sort(result.reachableStates.begin(), result.reachableStates.end());

  if (result.covered) {
    for (std::size_t state = target; state != start;
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

  for (const VirtualEdge &edge : edges) {
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

SyncCoverCoverageResult SyncCoverCoverageOracle::checkDemand(
    SyncCoverDemandId demand,
    const std::vector<SyncCoverMechanismId> &selected) const {
  if (!graph_.validate()) {
    return makeError(SyncCoverCoverageError::InvalidGraph);
  }
  if (demand >= graph_.getDemands().size()) {
    return makeError(SyncCoverCoverageError::InvalidDemand);
  }
  return checkDemandImpl(graph_, demand, normalizeSelection(selected));
}

std::vector<SyncCoverCoverageResult> SyncCoverCoverageOracle::checkAll(
    const std::vector<SyncCoverMechanismId> &selected) const {
  if (!graph_.validate()) {
    return {makeError(SyncCoverCoverageError::InvalidGraph)};
  }
  const std::vector<SyncCoverMechanismId> normalized =
      normalizeSelection(selected);
  std::vector<SyncCoverCoverageResult> results;
  results.reserve(graph_.getDemands().size());
  for (SyncCoverDemandId demand = 0; demand < graph_.getDemands().size();
       ++demand) {
    results.push_back(checkDemandImpl(graph_, demand, normalized));
  }
  return results;
}
