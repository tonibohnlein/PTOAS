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
#include "PTO/Transforms/CanonicalSync/SyncCoverExpansion.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir::pto;

namespace {

constexpr unsigned kStaticCopy = std::numeric_limits<unsigned>::max();
constexpr std::size_t kMaxVirtualNodes = 1U << 20;

void addSaturated(std::size_t value, std::size_t &total) {
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  total = value > maximum - total ? maximum : total + value;
}

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

using MechanismSet = std::vector<SyncCoverMechanismId>;

struct PreparedDemand {
  SyncCoverCoverageError error = SyncCoverCoverageError::None;
  std::size_t nodeCount = 0;
  std::size_t stateCount = 0;
  std::size_t start = 0;
  std::size_t target = 0;
  std::vector<VirtualEdge> edges;
  std::vector<std::vector<std::size_t>> outgoing;
  const SyncCoverGraph *graph = nullptr;
  const SyncCoverExpandedArena *arena = nullptr;
  std::vector<std::uint8_t> structuralEdgeMask;
  std::vector<std::uint8_t> mechanismEdgeMask;
  std::size_t activeEdgeCount = 0;
};

struct CoverageContextKey {
  SyncCoverScopeId recurrenceScope = 0;
  unsigned distance = 0;
  SyncCoverScopeId sourceScope = 0;
  SyncCoverScopeId targetScope = 0;
  std::vector<SyncCoverGuardLiteral> sourceGuard;
  std::vector<SyncCoverGuardLiteral> targetGuard;

  bool operator<(const CoverageContextKey &other) const {
    return std::tie(recurrenceScope, distance, sourceScope, targetScope,
                    sourceGuard, targetGuard) <
           std::tie(other.recurrenceScope, other.distance, other.sourceScope,
                    other.targetScope, other.sourceGuard, other.targetGuard);
  }
};

CoverageContextKey makeCoverageContextKey(const SyncCoverGraph &graph,
                                          const SyncCoverDemand &demand) {
  return {demand.scope, demand.distance,
          graph.getNodes()[demand.source].scope,
          graph.getNodes()[demand.target].scope, demand.sourceGuard.literals,
          demand.targetGuard.literals};
}

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
  const std::size_t copyCount = static_cast<std::size_t>(demand.distance) + 1;
  if (graph.getEdges().size() <=
      std::numeric_limits<std::size_t>::max() / copyCount) {
    result.reserve(graph.getEdges().size() * copyCount);
  }
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

VirtualEdge makeVirtualEdge(const SyncCoverGraph &graph,
                            const SyncCoverExpandedEdge &expanded) {
  const SyncCoverEdge &edge = graph.getEdges()[expanded.graphEdge];
  return {expanded.source, expanded.target, edge.kind, edge.mechanism};
}

template <typename Visitor>
void visitExpandedOutgoing(const PreparedDemand &prepared,
                           const SyncCoverExpandedAdjacency &adjacency,
                           const std::vector<std::uint8_t> &mask,
                           std::size_t virtualNode, Visitor &&visitor) {
  const auto &edges = adjacency.getEdges();
  for (const SyncCoverExpandedEdge &expanded :
       adjacency.getOutgoingEdges(virtualNode)) {
    const std::size_t index =
        static_cast<std::size_t>(&expanded - edges.data());
    if (index < mask.size() && mask[index] != 0) {
      visitor(makeVirtualEdge(*prepared.graph, expanded));
    }
  }
}

template <typename Visitor>
void visitOutgoing(const PreparedDemand &prepared, std::size_t virtualNode,
                   Visitor &&visitor) {
  if (!prepared.arena) {
    for (std::size_t edgeIndex : prepared.outgoing[virtualNode]) {
      visitor(prepared.edges[edgeIndex]);
    }
    return;
  }
  visitExpandedOutgoing(prepared, prepared.arena->getStructuralEdges(),
                        prepared.structuralEdgeMask, virtualNode, visitor);
  visitExpandedOutgoing(prepared, prepared.arena->getMechanismEdges(),
                        prepared.mechanismEdgeMask, virtualNode, visitor);
}

template <typename Visitor>
void visitExpandedEdges(const PreparedDemand &prepared,
                        const SyncCoverExpandedAdjacency &adjacency,
                        const std::vector<std::uint8_t> &mask,
                        Visitor &&visitor) {
  const auto &edges = adjacency.getEdges();
  for (std::size_t index = 0; index < edges.size(); ++index) {
    if (index < mask.size() && mask[index] != 0) {
      visitor(makeVirtualEdge(*prepared.graph, edges[index]));
    }
  }
}

template <typename Visitor>
void visitEdges(const PreparedDemand &prepared, Visitor &&visitor) {
  if (!prepared.arena) {
    for (const VirtualEdge &edge : prepared.edges) {
      visitor(edge);
    }
    return;
  }
  visitExpandedEdges(prepared, prepared.arena->getStructuralEdges(),
                     prepared.structuralEdgeMask, visitor);
  visitExpandedEdges(prepared, prepared.arena->getMechanismEdges(),
                     prepared.mechanismEdgeMask, visitor);
}

std::size_t stateIndex(std::size_t virtualNode, bool completion) {
  return virtualNode * 2 + static_cast<std::size_t>(completion);
}

std::optional<std::size_t> preparedStateIndex(const PreparedDemand &prepared,
                                              SyncCoverNodeId node,
                                              unsigned copy,
                                              bool completion) {
  if (prepared.arena) {
    const std::optional<std::size_t> virtualNode =
        prepared.arena->getVirtualNode(node, copy);
    return virtualNode ? std::optional<std::size_t>(
                             stateIndex(*virtualNode, completion))
                       : std::nullopt;
  }
  if (node >= prepared.nodeCount) {
    return std::nullopt;
  }
  return stateIndex(static_cast<std::size_t>(copy) * prepared.nodeCount + node,
                    completion);
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

std::vector<std::uint8_t> buildExpandedEdgeMask(
    const SyncCoverGraph &graph, const SyncCoverDemand &demand,
    const std::vector<ContextLiteral> &condition,
    const SyncCoverExpandedAdjacency &adjacency) {
  std::vector<std::uint8_t> result(adjacency.getEdges().size(), 0);
  for (std::size_t index = 0; index < adjacency.getEdges().size(); ++index) {
    const SyncCoverExpandedEdge &expanded = adjacency.getEdges()[index];
    const SyncCoverEdge &edge = graph.getEdges()[expanded.graphEdge];
    if (expanded.targetCopy > demand.distance ||
        !nodeInstanceAvailable(graph, demand, edge.source,
                               expanded.sourceCopy) ||
        !nodeInstanceAvailable(graph, demand, edge.target,
                               expanded.targetCopy) ||
        !guardIsImplied(graph, demand, condition, edge, expanded.sourceCopy,
                        expanded.targetCopy)) {
      continue;
    }
    result[index] = 1;
  }
  return result;
}

PreparedDemand prepareDemandShared(const SyncCoverGraph &graph,
                                   const SyncCoverExpandedProgram &expansion,
                                   SyncCoverDemandId demandId) {
  PreparedDemand prepared;
  const SyncCoverDemand &demand = graph.getDemands()[demandId];
  std::vector<ContextLiteral> condition;
  const bool sourceValid =
      appendGuard(graph, demand, demand.sourceGuard, 0, condition);
  const bool targetValid = appendGuard(graph, demand, demand.targetGuard,
                                       demand.distance, condition);
  const SyncCoverExpandedArena *arena = expansion.getArena(demand);
  if (!sourceValid || !targetValid || !arena) {
    prepared.error = !arena ? SyncCoverCoverageError::ExpansionLimitExceeded
                            : SyncCoverCoverageError::InvalidGraph;
    return prepared;
  }

  const std::optional<std::size_t> source =
      arena->getVirtualNode(demand.source, 0);
  const std::optional<std::size_t> target =
      arena->getVirtualNode(demand.target, demand.distance);
  if (!source || !target) {
    prepared.error = SyncCoverCoverageError::InvalidDemand;
    return prepared;
  }
  prepared.graph = &graph;
  prepared.arena = arena;
  prepared.nodeCount = arena->getNodeCount();
  prepared.stateCount =
      (static_cast<std::size_t>(demand.distance) + 1) *
      prepared.nodeCount * 2;
  prepared.start = stateIndex(*source, false);
  prepared.target = stateIndex(*target, true);
  prepared.structuralEdgeMask = buildExpandedEdgeMask(
      graph, demand, condition, arena->getStructuralEdges());
  prepared.mechanismEdgeMask = buildExpandedEdgeMask(
      graph, demand, condition, arena->getMechanismEdges());
  prepared.activeEdgeCount = static_cast<std::size_t>(std::count(
                                 prepared.structuralEdgeMask.begin(),
                                 prepared.structuralEdgeMask.end(), 1)) +
                             static_cast<std::size_t>(std::count(
                                 prepared.mechanismEdgeMask.begin(),
                                 prepared.mechanismEdgeMask.end(), 1));
  return prepared;
}

PreparedDemand prepareDemandLegacy(const SyncCoverGraph &graph,
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
  prepared.activeEdgeCount = prepared.edges.size();
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
    visitOutgoing(prepared, virtualNode, [&](const VirtualEdge &edge) {
      if (!mechanismSelected(selected, edge.mechanism)) {
        return;
      }
      const std::optional<std::size_t> next = transition(edge, state);
      if (!next || reachable[*next]) {
        return;
      }
      reachable[*next] = true;
      predecessors[*next] = {state, edge.mechanism};
      ready.push_back(*next);
    });
  }

  SyncCoverCoverageResult result;
  result.covered = reachable[prepared.target];
  for (std::size_t state = 0; state < prepared.stateCount; ++state) {
    if (!reachable[state]) {
      continue;
    }
    const std::size_t virtualNode = state / 2;
    const std::size_t localNode = virtualNode % prepared.nodeCount;
    const SyncCoverNodeId graphNode =
        prepared.arena ? prepared.arena->getGlobalNodes()[localNode]
                       : localNode;
    result.reachableStates.push_back(
        {graphNode,
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

  visitEdges(prepared, [&](const VirtualEdge &edge) {
    if (!edge.mechanism || mechanismSelected(selected, edge.mechanism)) {
      return;
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
  });
  std::sort(result.cutMechanisms.begin(), result.cutMechanisms.end());
  result.cutMechanisms.erase(
      std::unique(result.cutMechanisms.begin(), result.cutMechanisms.end()),
      result.cutMechanisms.end());
  return result;
}

std::vector<bool>
computeReachableStates(const PreparedDemand &prepared, std::size_t start,
                       const std::vector<SyncCoverMechanismId> &selected) {
  if (prepared.error != SyncCoverCoverageError::None ||
      start >= prepared.stateCount) {
    return {};
  }
  std::vector<bool> reachable(prepared.stateCount, false);
  std::vector<std::size_t> ready{start};
  reachable[start] = true;
  for (std::size_t readyIndex = 0; readyIndex < ready.size(); ++readyIndex) {
    const std::size_t state = ready[readyIndex];
    const std::size_t virtualNode = state / 2;
    visitOutgoing(prepared, virtualNode, [&](const VirtualEdge &edge) {
      if (!mechanismSelected(selected, edge.mechanism)) {
        return;
      }
      const std::optional<std::size_t> next = transition(edge, state);
      if (!next || reachable[*next]) {
        return;
      }
      reachable[*next] = true;
      ready.push_back(*next);
    });
  }
  return reachable;
}

struct SingletonReachability {
  SyncCoverCoverageError error = SyncCoverCoverageError::None;
  std::vector<std::vector<std::uint64_t>> states;
};

SingletonReachability computeSingletonReachability(
    const PreparedDemand &prepared, std::size_t start,
    std::size_t mechanismCount) {
  SingletonReachability result;
  if (prepared.error != SyncCoverCoverageError::None) {
    result.error = prepared.error;
    return result;
  }
  if (start >= prepared.stateCount) {
    result.error = SyncCoverCoverageError::InvalidDemand;
    return result;
  }
  constexpr std::size_t kWordBits = 64;
  const std::size_t wordCount = (mechanismCount + kWordBits - 1) / kWordBits;
  if (wordCount == 0) {
    return result;
  }
  result.states.assign(
      prepared.stateCount, std::vector<std::uint64_t>(wordCount, 0));
  for (std::size_t mechanism = 0; mechanism < mechanismCount; ++mechanism) {
    result.states[start][mechanism / kWordBits] |=
        std::uint64_t{1} << static_cast<unsigned>(mechanism % kWordBits);
  }

  std::vector<std::size_t> ready{start};
  std::vector<bool> queued(prepared.stateCount, false);
  queued[start] = true;
  for (std::size_t index = 0; index < ready.size(); ++index) {
    if (result.error != SyncCoverCoverageError::None) {
      break;
    }
    const std::size_t state = ready[index];
    queued[state] = false;
    const std::size_t virtualNode = state / 2;
    visitOutgoing(prepared, virtualNode, [&](const VirtualEdge &edge) {
      const std::optional<std::size_t> next = transition(edge, state);
      if (!next) {
        return;
      }
      bool changed = false;
      if (edge.mechanism) {
        const std::size_t mechanism = *edge.mechanism;
        if (mechanism >= mechanismCount) {
          result.error = SyncCoverCoverageError::InvalidGraph;
          return;
        }
        const std::size_t word = mechanism / kWordBits;
        const std::uint64_t bit =
            std::uint64_t{1}
            << static_cast<unsigned>(mechanism % kWordBits);
        if ((result.states[state][word] & bit) != 0 &&
            (result.states[*next][word] & bit) == 0) {
          result.states[*next][word] |= bit;
          changed = true;
        }
      } else {
        for (std::size_t word = 0; word < wordCount; ++word) {
          const std::uint64_t additions =
              result.states[state][word] & ~result.states[*next][word];
          if (additions != 0) {
            result.states[*next][word] |= additions;
            changed = true;
          }
        }
      }
      if (changed && !queued[*next]) {
        queued[*next] = true;
        ready.push_back(*next);
      }
    });
  }

  return result;
}

SyncCoverSingletonWitnessResult collectSingletonWitnesses(
    const PreparedDemand &prepared, std::size_t mechanismCount,
    std::optional<std::size_t> queryStart = {},
    std::optional<std::size_t> queryTarget = {}) {
  SyncCoverSingletonWitnessResult result;
  const std::size_t start = queryStart.value_or(prepared.start);
  const std::size_t target = queryTarget.value_or(prepared.target);
  const SingletonReachability reachable =
      computeSingletonReachability(prepared, start, mechanismCount);
  if (reachable.error != SyncCoverCoverageError::None) {
    result.error = reachable.error;
    return result;
  }
  if (mechanismCount == 0) {
    return result;
  }
  if (target >= reachable.states.size()) {
    result.error = SyncCoverCoverageError::InvalidDemand;
    return result;
  }
  constexpr std::size_t kWordBits = 64;
  for (std::size_t mechanism = 0; mechanism < mechanismCount; ++mechanism) {
    const std::size_t word = mechanism / kWordBits;
    const std::uint64_t bit =
        std::uint64_t{1}
        << static_cast<unsigned>(mechanism % kWordBits);
    if (!reachable.states.empty() &&
        (reachable.states[target][word] & bit) != 0) {
      result.mechanisms.push_back(mechanism);
    }
  }
  return result;
}

SyncCoverSelectionWitnessResult collectSelectionWitnesses(
    const PreparedDemand &prepared,
    const std::vector<std::vector<SyncCoverMechanismId>> &selections,
    std::size_t mechanismCount, std::optional<std::size_t> queryStart = {},
    std::optional<std::size_t> queryTarget = {}) {
  SyncCoverSelectionWitnessResult result;
  if (prepared.error != SyncCoverCoverageError::None) {
    result.error = prepared.error;
    return result;
  }
  constexpr std::size_t kWordBits = 64;
  const std::size_t wordCount =
      selections.size() / kWordBits +
      (selections.size() % kWordBits == 0 ? 0 : 1);
  if (wordCount == 0) {
    return result;
  }
  const std::size_t start = queryStart.value_or(prepared.start);
  const std::size_t target = queryTarget.value_or(prepared.target);
  if (start >= prepared.stateCount || target >= prepared.stateCount) {
    result.error = SyncCoverCoverageError::InvalidDemand;
    return result;
  }
  std::vector<std::vector<std::uint64_t>> mechanismSelections(
      mechanismCount, std::vector<std::uint64_t>(wordCount, 0));
  for (std::size_t selection = 0; selection < selections.size(); ++selection) {
    const std::vector<SyncCoverMechanismId> &members = selections[selection];
    if (members.empty() || !std::is_sorted(members.begin(), members.end()) ||
        std::adjacent_find(members.begin(), members.end()) != members.end() ||
        members.back() >= mechanismCount) {
      result.error = SyncCoverCoverageError::InvalidSelection;
      return result;
    }
    const std::size_t word = selection / kWordBits;
    const std::uint64_t bit =
        std::uint64_t{1}
        << static_cast<unsigned>(selection % kWordBits);
    for (SyncCoverMechanismId mechanism : members) {
      mechanismSelections[mechanism][word] |= bit;
    }
  }

  std::vector<std::vector<std::uint64_t>> reachable(
      prepared.stateCount, std::vector<std::uint64_t>(wordCount, 0));
  std::fill(reachable[start].begin(), reachable[start].end(),
            std::numeric_limits<std::uint64_t>::max());
  const unsigned tailBits = static_cast<unsigned>(selections.size() % kWordBits);
  if (tailBits != 0) {
    reachable[start].back() =
        (std::uint64_t{1} << tailBits) - 1;
  }

  std::vector<std::size_t> ready{start};
  std::vector<bool> queued(prepared.stateCount, false);
  queued[start] = true;
  for (std::size_t index = 0; index < ready.size(); ++index) {
    if (result.error != SyncCoverCoverageError::None) {
      break;
    }
    const std::size_t state = ready[index];
    queued[state] = false;
    const std::size_t virtualNode = state / 2;
    visitOutgoing(prepared, virtualNode, [&](const VirtualEdge &edge) {
      const std::optional<std::size_t> next = transition(edge, state);
      if (!next) {
        return;
      }
      bool changed = false;
      for (std::size_t word = 0; word < wordCount; ++word) {
        std::uint64_t additions = reachable[state][word];
        if (edge.mechanism) {
          if (*edge.mechanism >= mechanismCount) {
            result.error = SyncCoverCoverageError::InvalidGraph;
            return;
          }
          additions &= mechanismSelections[*edge.mechanism][word];
        }
        additions &= ~reachable[*next][word];
        if (additions != 0) {
          reachable[*next][word] |= additions;
          changed = true;
        }
      }
      if (changed && !queued[*next]) {
        queued[*next] = true;
        ready.push_back(*next);
      }
    });
  }

  if (result.error != SyncCoverCoverageError::None) {
    return result;
  }

  for (std::size_t selection = 0; selection < selections.size(); ++selection) {
    const std::size_t word = selection / kWordBits;
    const std::uint64_t bit =
        std::uint64_t{1}
        << static_cast<unsigned>(selection % kWordBits);
    if ((reachable[target][word] & bit) != 0) {
      result.selections.push_back(selection);
    }
  }
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
  explicit Implementation(const SyncCoverGraph &source,
                          SyncCoverCoverageBackend requestedBackend)
      : graph(source), backend(requestedBackend) {
    ++statistics.graphValidations;
    if (!graph.validate()) {
      validationError = SyncCoverCoverageError::InvalidGraph;
      return;
    }
    if (backend == SyncCoverCoverageBackend::SharedExpansion) {
      if (!graph.isStructureFrozen()) {
        validationError = SyncCoverCoverageError::InvalidGraph;
        return;
      }
      expansion = std::make_unique<SyncCoverExpandedProgram>(graph);
      if (!*expansion) {
        validationError =
            expansion->getError() ==
                    SyncCoverExpansionError::ExpansionLimitExceeded
                ? SyncCoverCoverageError::ExpansionLimitExceeded
                : SyncCoverCoverageError::InvalidGraph;
      }
      return;
    }
  }

  const PreparedDemand &getPreparedDemand(SyncCoverDemandId demand) const {
    return getPreparedContext(demand);
  }

  /// Demands sharing a CoverageContextKey have identical virtual topologies
  /// (endpoint-specific availability is subsumed by the scope-level rules),
  /// so one prepared product graph serves every demand in the context. The
  /// batch APIs relied on this within a call; caching extends the sharing
  /// across calls on this oracle. The endpoint-derived fields (start, target,
  /// potential mechanisms) are re-derived per demand on access: the oracle
  /// is single-consumer, so fixing up the cached entry in place is safe.
  const PreparedDemand &getPreparedContext(SyncCoverDemandId demandId) const {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const CoverageContextKey key = makeCoverageContextKey(graph, demand);
    auto entry = preparedContexts.find(key);
    if (entry == preparedContexts.end()) {
      PreparedDemand prepared =
          backend == SyncCoverCoverageBackend::SharedExpansion
              ? prepareDemandShared(graph, *expansion, demandId)
              : prepareDemandLegacy(graph, demandId);
      entry = preparedContexts.emplace(key, std::move(prepared)).first;
      recordPreparation(entry->second);
      return entry->second;
    }
    PreparedDemand &prepared = entry->second;
    if (prepared.error == SyncCoverCoverageError::None) {
      std::size_t source = demand.source;
      std::size_t target = static_cast<std::size_t>(demand.distance) *
                               prepared.nodeCount +
                           demand.target;
      if (prepared.arena) {
        const std::optional<std::size_t> localSource =
            prepared.arena->getVirtualNode(demand.source, 0);
        const std::optional<std::size_t> localTarget =
            prepared.arena->getVirtualNode(demand.target, demand.distance);
        if (!localSource || !localTarget) {
          prepared.error = SyncCoverCoverageError::InvalidDemand;
          return prepared;
        }
        source = *localSource;
        target = *localTarget;
      }
      prepared.start = stateIndex(source, false);
      prepared.target = stateIndex(target, true);
    }
    return prepared;
  }

  void recordPreparation(const PreparedDemand &entry) const {
    ++statistics.demandPreparations;
    const std::size_t virtualNodes = entry.stateCount / 2;
    addSaturated(virtualNodes, statistics.preparedVirtualNodes);
    addSaturated(entry.activeEdgeCount, statistics.preparedVirtualEdges);
    statistics.maximumVirtualNodes =
        std::max(statistics.maximumVirtualNodes, virtualNodes);
    statistics.maximumVirtualEdges =
        std::max(statistics.maximumVirtualEdges, entry.activeEdgeCount);
  }

  SyncCoverGraph graph;
  SyncCoverCoverageBackend backend =
      SyncCoverCoverageBackend::LegacyPerContext;
  std::unique_ptr<SyncCoverExpandedProgram> expansion;
  SyncCoverCoverageError validationError = SyncCoverCoverageError::None;
  mutable std::map<CoverageContextKey, PreparedDemand> preparedContexts;
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

SyncCoverCoverageOracle::SyncCoverCoverageOracle(
    const SyncCoverGraph &graph, SyncCoverCoverageBackend backend)
    : implementation_(std::make_unique<Implementation>(graph, backend)) {}

SyncCoverCoverageOracle::~SyncCoverCoverageOracle() = default;

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

std::vector<SyncCoverCoverageResult>
SyncCoverCoverageOracle::checkDemandsCanonicalSelection(
    const std::vector<SyncCoverDemandId> &demands,
    const std::vector<SyncCoverMechanismId> &selected) const {
  std::vector<SyncCoverCoverageResult> results(demands.size());
  if (implementation_->validationError != SyncCoverCoverageError::None) {
    std::fill(results.begin(), results.end(),
              makeError(implementation_->validationError));
    return results;
  }
  const bool invalidSelection =
      !std::is_sorted(selected.begin(), selected.end()) ||
      std::adjacent_find(selected.begin(), selected.end()) != selected.end();
  if (invalidSelection) {
    std::fill(results.begin(), results.end(),
              makeError(SyncCoverCoverageError::InvalidSelection));
    return results;
  }

  using IndexedDemand = std::pair<std::size_t, SyncCoverDemandId>;
  std::map<CoverageContextKey, std::vector<IndexedDemand>> contexts;
  for (std::size_t index = 0; index < demands.size(); ++index) {
    const SyncCoverDemandId demand = demands[index];
    if (demand >= implementation_->graph.getDemands().size()) {
      results[index] = makeError(SyncCoverCoverageError::InvalidDemand);
      continue;
    }
    contexts[makeCoverageContextKey(
                 implementation_->graph,
                 implementation_->graph.getDemands()[demand])]
        .emplace_back(index, demand);
  }

  for (const auto &[key, contextDemands] : contexts) {
    (void)key;
    const PreparedDemand &prepared =
        implementation_->getPreparedContext(contextDemands.front().second);
    if (prepared.error != SyncCoverCoverageError::None) {
      for (const auto &[index, demand] : contextDemands) {
        (void)demand;
        results[index] = makeError(prepared.error);
      }
      continue;
    }
    std::map<SyncCoverNodeId, std::vector<bool>> reachableBySource;
    for (const auto &[index, demandId] : contextDemands) {
      ++implementation_->statistics.coverageQueries;
      const SyncCoverDemand &demand =
          implementation_->graph.getDemands()[demandId];
      const std::optional<std::size_t> sourceState =
          preparedStateIndex(prepared, demand.source, 0, false);
      const std::optional<std::size_t> targetState = preparedStateIndex(
          prepared, demand.target, demand.distance, true);
      if (!sourceState || !targetState) {
        results[index] = makeError(SyncCoverCoverageError::InvalidDemand);
        continue;
      }
      auto reachable = reachableBySource.find(demand.source);
      if (reachable == reachableBySource.end()) {
        reachable = reachableBySource
                        .emplace(demand.source,
                                 computeReachableStates(prepared, *sourceState,
                                                        selected))
                        .first;
      }
      if (reachable->second.empty()) {
        results[index] = makeError(prepared.error);
        continue;
      }
      results[index].covered = *targetState < reachable->second.size() &&
                               reachable->second[*targetState];
    }
  }
  return results;
}

SyncCoverSingletonWitnessResult
SyncCoverCoverageOracle::getSingletonMechanismWitnesses(
    SyncCoverDemandId demand) const {
  if (implementation_->validationError != SyncCoverCoverageError::None) {
    SyncCoverSingletonWitnessResult result;
    result.error = implementation_->validationError;
    return result;
  }
  if (demand >= implementation_->graph.getDemands().size()) {
    SyncCoverSingletonWitnessResult result;
    result.error = SyncCoverCoverageError::InvalidDemand;
    return result;
  }
  ++implementation_->statistics.groundingQueries;
  std::size_t mechanismCount = 0;
  for (const SyncCoverEdge &edge : implementation_->graph.getEdges()) {
    if (edge.mechanism) {
      if (*edge.mechanism == std::numeric_limits<std::size_t>::max()) {
        SyncCoverSingletonWitnessResult result;
        result.error = SyncCoverCoverageError::InvalidGraph;
        return result;
      }
      mechanismCount = std::max(mechanismCount, *edge.mechanism + 1);
    }
  }
  const PreparedDemand &prepared = implementation_->getPreparedContext(demand);
  return collectSingletonWitnesses(prepared, mechanismCount);
}

std::vector<SyncCoverSingletonWitnessResult>
SyncCoverCoverageOracle::getSingletonMechanismWitnessesForDemands(
    const std::vector<SyncCoverDemandId> &demands,
    std::size_t mechanismCount) const {
  std::vector<SyncCoverSingletonWitnessResult> results(demands.size());
  if (implementation_->validationError != SyncCoverCoverageError::None) {
    for (SyncCoverSingletonWitnessResult &result : results) {
      result.error = implementation_->validationError;
    }
    return results;
  }
  using IndexedDemand = std::pair<std::size_t, SyncCoverDemandId>;
  std::map<CoverageContextKey, std::vector<IndexedDemand>> contexts;
  for (std::size_t index = 0; index < demands.size(); ++index) {
    const SyncCoverDemandId demand = demands[index];
    if (demand >= implementation_->graph.getDemands().size()) {
      results[index].error = SyncCoverCoverageError::InvalidDemand;
      continue;
    }
    contexts[makeCoverageContextKey(
                 implementation_->graph,
                 implementation_->graph.getDemands()[demand])]
        .emplace_back(index, demand);
  }
  constexpr std::size_t kWordBits = 64;
  for (const auto &[key, contextDemands] : contexts) {
    (void)key;
    const PreparedDemand &context =
        implementation_->getPreparedContext(contextDemands.front().second);
    std::map<SyncCoverNodeId, SingletonReachability> reachableBySource;
    for (const auto &[index, demandId] : contextDemands) {
      ++implementation_->statistics.groundingQueries;
      if (context.error != SyncCoverCoverageError::None) {
        results[index].error = context.error;
        continue;
      }
      const SyncCoverDemand &demand =
          implementation_->graph.getDemands()[demandId];
      const std::optional<std::size_t> sourceState =
          preparedStateIndex(context, demand.source, 0, false);
      const std::optional<std::size_t> targetState = preparedStateIndex(
          context, demand.target, demand.distance, true);
      if (!sourceState || !targetState) {
        results[index].error = SyncCoverCoverageError::InvalidDemand;
        continue;
      }
      auto reachable = reachableBySource.find(demand.source);
      if (reachable == reachableBySource.end()) {
        reachable = reachableBySource
                        .emplace(demand.source,
                                 computeSingletonReachability(
                                     context, *sourceState, mechanismCount))
                        .first;
      }
      if (reachable->second.error != SyncCoverCoverageError::None) {
        results[index].error = reachable->second.error;
        continue;
      }
      if (mechanismCount == 0) {
        continue;
      }
      if (*targetState >= reachable->second.states.size()) {
        results[index].error = SyncCoverCoverageError::InvalidDemand;
        continue;
      }
      for (std::size_t mechanism = 0; mechanism < mechanismCount;
           ++mechanism) {
        const std::size_t word = mechanism / kWordBits;
        const std::uint64_t bit =
            std::uint64_t{1}
            << static_cast<unsigned>(mechanism % kWordBits);
        if ((reachable->second.states[*targetState][word] & bit) != 0) {
          results[index].mechanisms.push_back(mechanism);
        }
      }
    }
  }
  return results;
}

std::vector<SyncCoverSelectionWitnessResult>
SyncCoverCoverageOracle::getSelectionWitnessesForDemands(
    const std::vector<SyncCoverDemandId> &demands,
    const std::vector<std::vector<SyncCoverMechanismId>> &selections,
    std::size_t mechanismCount) const {
  std::vector<SyncCoverSelectionWitnessResult> results(demands.size());
  if (implementation_->validationError != SyncCoverCoverageError::None) {
    for (SyncCoverSelectionWitnessResult &result : results) {
      result.error = implementation_->validationError;
    }
    return results;
  }
  using IndexedDemand = std::pair<std::size_t, SyncCoverDemandId>;
  std::map<CoverageContextKey, std::vector<IndexedDemand>> contexts;
  for (std::size_t index = 0; index < demands.size(); ++index) {
    const SyncCoverDemandId demand = demands[index];
    if (demand >= implementation_->graph.getDemands().size()) {
      results[index].error = SyncCoverCoverageError::InvalidDemand;
      continue;
    }
    contexts[makeCoverageContextKey(
                 implementation_->graph,
                 implementation_->graph.getDemands()[demand])]
        .emplace_back(index, demand);
  }
  for (const auto &[key, contextDemands] : contexts) {
    (void)key;
    const PreparedDemand &context =
        implementation_->getPreparedContext(contextDemands.front().second);
    for (const auto &[index, demandId] : contextDemands) {
      ++implementation_->statistics.groundingQueries;
      if (context.error != SyncCoverCoverageError::None) {
        results[index].error = context.error;
        continue;
      }
      const SyncCoverDemand &demand =
          implementation_->graph.getDemands()[demandId];
      const std::optional<std::size_t> start =
          preparedStateIndex(context, demand.source, 0, false);
      const std::optional<std::size_t> target = preparedStateIndex(
          context, demand.target, demand.distance, true);
      if (!start || !target) {
        results[index].error = SyncCoverCoverageError::InvalidDemand;
        continue;
      }
      results[index] = collectSelectionWitnesses(
          context, selections, mechanismCount, *start, *target);
    }
  }
  return results;
}


SyncCoverCoverageStatistics SyncCoverCoverageOracle::getStatistics() const {
  return implementation_->statistics;
}
