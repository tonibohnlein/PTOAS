// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverExpansion.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

using namespace mlir::pto;

namespace {

bool checkedAdd(std::size_t left, std::size_t right, std::size_t &result) {
  const bool overflows = right > std::numeric_limits<std::size_t>::max() - left;
  if (overflows) {
    return false;
  }
  result = left + right;
  return true;
}

bool checkedMultiply(std::size_t left, std::size_t right, std::size_t &result) {
  const bool overflows =
      left != 0 && right > std::numeric_limits<std::size_t>::max() / left;
  if (overflows) {
    return false;
  }
  result = left * right;
  return true;
}

std::vector<std::size_t> allDemandIds(const SyncCoverGraph &graph) {
  std::vector<std::size_t> result(graph.getDemands().size());
  for (std::size_t demand = 0; demand < result.size(); ++demand) {
    result[demand] = demand;
  }
  return result;
}

SyncCoverScopeId getArenaScopeForScope(const SyncCoverGraph &graph,
                                       SyncCoverScopeId scope) {
  return graph.getNearestEnclosingLoop(scope, true).value_or(0);
}

std::optional<SyncCoverScopeId>
getImmediateChildLoop(const SyncCoverGraph &graph, SyncCoverScopeId arenaScope,
                      SyncCoverScopeId nodeScope) {
  if (arenaScope != 0 && !graph.scopeContains(arenaScope, nodeScope)) {
    return std::nullopt;
  }
  std::optional<SyncCoverScopeId> current =
      graph.getNearestEnclosingLoop(nodeScope, true);
  if (!current || *current == arenaScope) {
    return std::nullopt;
  }
  while (current) {
    const std::optional<SyncCoverScopeId> parent =
        graph.getNearestEnclosingLoop(graph.getScopes()[*current].parent, true);
    const bool isImmediateChild =
        (arenaScope == 0 && !parent) || (parent && *parent == arenaScope);
    if (isImmediateChild) {
      return current;
    }
    current = parent;
  }
  return std::nullopt;
}

std::vector<std::size_t>
getReachablePhases(const SyncCoverControlPhaseRelation &relation) {
  std::vector<std::size_t> result;
  std::vector<bool> visited(relation.nextPhase.size(), false);
  std::size_t phase = relation.initialPhase;
  while (!visited[phase]) {
    visited[phase] = true;
    result.push_back(phase);
    phase = relation.nextPhase[phase];
  }
  std::sort(result.begin(), result.end());
  return result;
}

} // namespace

struct SyncCoverExpandedProgram::ArenaBuildResult {
  SyncCoverExpandedArena arena;
  std::optional<SyncCoverArenaUnavailableReason> unavailable;
  std::size_t compactCarryEdges = 0;
  std::size_t loopSummaryNodes = 0;
  std::size_t zeroTripEdges = 0;
};

std::optional<std::size_t>
SyncCoverExpandedArena::getVirtualOperation(SyncCoverNodeId node,
                                            unsigned copy) const {
  if (copy > horizon_) {
    return std::nullopt;
  }
  const auto position =
      std::lower_bound(operationNodes_.begin(), operationNodes_.end(), node);
  const bool missing = position == operationNodes_.end() || *position != node;
  if (missing) {
    return std::nullopt;
  }
  const std::size_t local =
      static_cast<std::size_t>(position - operationNodes_.begin());
  return static_cast<std::size_t>(copy) * operationNodes_.size() + local;
}

std::optional<SyncCoverNodeId>
SyncCoverExpandedArena::getOperationForVirtualNode(
    std::size_t virtualNode) const {
  if (virtualNode >= operationVirtualNodeCount_ || operationNodes_.empty()) {
    return std::nullopt;
  }
  return operationNodes_[virtualNode % operationNodes_.size()];
}

std::optional<unsigned>
SyncCoverExpandedArena::getCopyForVirtualNode(std::size_t virtualNode) const {
  if (virtualNode >= virtualNodeCount_) {
    return std::nullopt;
  }
  if (virtualNode < operationVirtualNodeCount_) {
    if (operationNodes_.empty()) {
      return std::nullopt;
    }
    return static_cast<unsigned>(virtualNode / operationNodes_.size());
  }
  const std::size_t summaryEnd =
      operationVirtualNodeCount_ + loopSummaryVirtualNodeCount_;
  if (virtualNode < summaryEnd) {
    const std::size_t nodesPerCopy = loopSummaryResources_.size() * 2;
    if (nodesPerCopy == 0) {
      return std::nullopt;
    }
    return static_cast<unsigned>((virtualNode - operationVirtualNodeCount_) /
                                 nodesPerCopy);
  }
  if (carryResources_.empty()) {
    return std::nullopt;
  }
  return static_cast<unsigned>((virtualNode - summaryEnd) /
                               carryResources_.size());
}

std::optional<std::size_t> SyncCoverExpandedArena::getLoopBoundary(
    SyncCoverScopeId scope, std::uint32_t resource,
    SyncCoverLoopBoundaryKind kind, unsigned copy) const {
  if (copy > horizon_) {
    return std::nullopt;
  }
  const std::pair<SyncCoverScopeId, std::uint32_t> key{scope, resource};
  const auto position = std::lower_bound(loopSummaryResources_.begin(),
                                         loopSummaryResources_.end(), key);
  const bool missing =
      position == loopSummaryResources_.end() || *position != key;
  if (missing) {
    return std::nullopt;
  }
  const std::size_t pair =
      static_cast<std::size_t>(position - loopSummaryResources_.begin());
  const std::size_t nodesPerCopy = loopSummaryResources_.size() * 2;
  const std::size_t boundary = kind == SyncCoverLoopBoundaryKind::Entry ? 0 : 1;
  return operationVirtualNodeCount_ +
         static_cast<std::size_t>(copy) * nodesPerCopy + pair * 2 + boundary;
}

SyncCoverExpandedEdgeRange
SyncCoverExpandedArena::getOutgoingEdges(std::size_t virtualNode) const {
  const bool invalidNode =
      outgoingOffsets_.empty() || virtualNode >= virtualNodeCount_;
  if (invalidNode) {
    return {edges_.end(), edges_.end()};
  }
  const auto begin = edges_.begin() +
                     static_cast<std::ptrdiff_t>(outgoingOffsets_[virtualNode]);
  const auto end = edges_.begin() + static_cast<std::ptrdiff_t>(
                                        outgoingOffsets_[virtualNode + 1]);
  return {begin, end};
}

SyncCoverExpandedProgram::SyncCoverExpandedProgram(
    const SyncCoverGraph &graph, SyncCoverExpansionLimits limits)
    : SyncCoverExpandedProgram(graph, allDemandIds(graph), limits) {}

SyncCoverExpandedProgram::SyncCoverExpandedProgram(
    const SyncCoverGraph &graph, const std::vector<std::size_t> &activeDemands,
    SyncCoverExpansionLimits limits)
    : ownerIdentity_(graph.getIdentity()), limits_(limits) {
  const bool invalidLimits =
      limits.maximumArenaNodes == 0 || limits.maximumArenaEdges == 0 ||
      limits.maximumTotalNodes == 0 || limits.maximumTotalEdges == 0;
  if (invalidLimits) {
    error_ = SyncCoverExpansionError::InvalidLimits;
    return;
  }
  const bool invalidDemandSet =
      !std::is_sorted(activeDemands.begin(), activeDemands.end()) ||
      std::adjacent_find(activeDemands.begin(), activeDemands.end()) !=
          activeDemands.end() ||
      std::any_of(activeDemands.begin(), activeDemands.end(),
                  [&](std::size_t demand) {
                    return demand >= graph.getDemands().size();
                  });
  const bool invalidGraph =
      !graph.isStructureFrozen() || !graph.validate() || invalidDemandSet;
  if (invalidGraph) {
    error_ = SyncCoverExpansionError::InvalidGraph;
    return;
  }

  arenaScopeByScope_.resize(graph.getScopes().size());
  for (const SyncCoverScope &scope : graph.getScopes()) {
    arenaScopeByScope_[scope.id] = getArenaScopeForScope(graph, scope.id);
  }
  std::map<SyncCoverScopeId, unsigned> horizons;
  for (std::size_t demandId : activeDemands) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverScopeId arenaScope =
        demand.distance == 0 ? arenaScopeByScope_[demand.scope] : demand.scope;
    if (arenaScope != 0) {
      horizons[arenaScope] = std::max(horizons[arenaScope], demand.distance);
    }
  }

  std::vector<std::pair<std::size_t, SyncCoverScopeId>> loopOrder;
  for (const SyncCoverScope &scope : graph.getScopes()) {
    if (!scope.isLoop) {
      continue;
    }
    std::size_t depth = 0;
    for (SyncCoverScopeId current = scope.id; current != 0;
         current = graph.getScopes()[current].parent) {
      ++depth;
    }
    loopOrder.push_back({depth, scope.id});
  }
  std::sort(loopOrder.begin(), loopOrder.end(),
            [](const auto &left, const auto &right) {
              return std::make_pair(left.first, left.second) >
                     std::make_pair(right.first, right.second);
            });
  for (const auto &[depth, scopeId] : loopOrder) {
    (void)depth;
    const SyncCoverScope &scope = graph.getScopes()[scopeId];
    SyncCoverLoopSummary summary;
    summary.scope = scopeId;
    summary.parentLoop = graph.getNearestEnclosingLoop(scope.parent, true);
    summary.entry = {SyncCoverAnchorKind::ScopeEntry, 0, scopeId,
                     scope.timeline->begin};
    summary.exit = {SyncCoverAnchorKind::ScopeExit, 0, scopeId,
                    scope.timeline->end};
    summary.zeroTripPossible = !scope.mustExecuteWithinParent;
    loopSummaryIndices_[scopeId] = loopSummaries_.size();
    loopSummaries_.push_back(std::move(summary));
  }
  for (const SyncCoverControl &control : graph.getControls()) {
    if (!control.phaseRelation) {
      continue;
    }
    const auto summary =
        loopSummaryIndices_.find(control.phaseRelation->loopScope);
    if (summary == loopSummaryIndices_.end()) {
      continue;
    }
    loopSummaries_[summary->second].periodicControls.push_back(
        {control.id, control.phaseRelation->initialPhase,
         control.phaseRelation->nextPhase,
         control.phaseRelation->activeAlternative,
         getReachablePhases(*control.phaseRelation)});
  }
  for (const SyncCoverNode &node : graph.getNodes()) {
    const std::optional<SyncCoverScopeId> loop =
        graph.getNearestEnclosingLoop(node.scope, true);
    if (!loop) {
      continue;
    }
    SyncCoverLoopSummary &summary =
        loopSummaries_[loopSummaryIndices_.at(*loop)];
    summary.resources.push_back(node.resource);
    const bool hasCarry =
        graph.getResourceRecurrenceCarryKinds().count(node.resource) != 0;
    if (hasCarry) {
      summary.carryResources.push_back(node.resource);
    }
  }
  for (SyncCoverLoopSummary &summary : loopSummaries_) {
    if (!summary.parentLoop) {
      continue;
    }
    SyncCoverLoopSummary &parent =
        loopSummaries_[loopSummaryIndices_.at(*summary.parentLoop)];
    parent.childLoops.push_back(summary.scope);
    parent.resources.insert(parent.resources.end(), summary.resources.begin(),
                            summary.resources.end());
    parent.carryResources.insert(parent.carryResources.end(),
                                 summary.carryResources.begin(),
                                 summary.carryResources.end());
  }
  for (SyncCoverLoopSummary &summary : loopSummaries_) {
    auto normalize = [](auto &values) {
      std::sort(values.begin(), values.end());
      values.erase(std::unique(values.begin(), values.end()), values.end());
    };
    normalize(summary.childLoops);
    normalize(summary.resources);
    normalize(summary.carryResources);
    for (std::uint32_t resource : summary.resources) {
      summary.completionTransfers.push_back(
          {resource, true, summary.zeroTripPossible});
    }
    std::sort(summary.periodicControls.begin(), summary.periodicControls.end(),
              [](const auto &left, const auto &right) {
                return left.control < right.control;
              });
  }

  ArenaBuildResult base = buildArena(
      graph, 0, 0, std::min(limits.maximumArenaNodes, limits.maximumTotalNodes),
      std::min(limits.maximumArenaEdges, limits.maximumTotalEdges),
      SyncCoverArenaUnavailableReason::NodeLimit,
      SyncCoverArenaUnavailableReason::EdgeLimit);
  const bool invalidBase =
      base.unavailable ||
      base.arena.getVirtualNodeCount() > limits.maximumTotalNodes ||
      base.arena.getEdges().size() > limits.maximumTotalEdges;
  if (invalidBase) {
    error_ = SyncCoverExpansionError::BaseLimitExceeded;
    return;
  }
  baseArena_ = std::move(base.arena);
  statistics_.arenaCount = 1;
  statistics_.virtualNodes = baseArena_.getVirtualNodeCount();
  statistics_.virtualEdges = baseArena_.getEdges().size();
  statistics_.compactCarryEdges = base.compactCarryEdges;
  statistics_.loopSummaryNodes = base.loopSummaryNodes;
  statistics_.zeroTripEdges = base.zeroTripEdges;
  for (const auto &[scope, horizon] : horizons) {
    const std::size_t remainingNodes =
        limits.maximumTotalNodes - statistics_.virtualNodes;
    const std::size_t remainingEdges =
        limits.maximumTotalEdges - statistics_.virtualEdges;
    const std::size_t maximumNodes =
        std::min(limits.maximumArenaNodes, remainingNodes);
    const std::size_t maximumEdges =
        std::min(limits.maximumArenaEdges, remainingEdges);
    const SyncCoverArenaUnavailableReason nodeLimitReason =
        remainingNodes < limits.maximumArenaNodes
            ? SyncCoverArenaUnavailableReason::AggregateNodeLimit
            : SyncCoverArenaUnavailableReason::NodeLimit;
    const SyncCoverArenaUnavailableReason edgeLimitReason =
        remainingEdges < limits.maximumArenaEdges
            ? SyncCoverArenaUnavailableReason::AggregateEdgeLimit
            : SyncCoverArenaUnavailableReason::EdgeLimit;
    ArenaBuildResult result =
        buildArena(graph, scope, horizon, maximumNodes, maximumEdges,
                   nodeLimitReason, edgeLimitReason);
    if (result.unavailable) {
      unavailableArenas_[scope] = *result.unavailable;
      ++statistics_.unavailableArenaCount;
      continue;
    }
    statistics_.virtualNodes += result.arena.getVirtualNodeCount();
    statistics_.virtualEdges += result.arena.getEdges().size();
    statistics_.compactCarryEdges += result.compactCarryEdges;
    statistics_.loopSummaryNodes += result.loopSummaryNodes;
    statistics_.zeroTripEdges += result.zeroTripEdges;
    ++statistics_.arenaCount;
    recurrenceArenas_.emplace(scope, std::move(result.arena));
  }
}

const SyncCoverExpandedArena *
SyncCoverExpandedProgram::getRecurrenceArena(SyncCoverScopeId scope) const {
  const auto arena = recurrenceArenas_.find(scope);
  return arena == recurrenceArenas_.end() ? nullptr : &arena->second;
}

const SyncCoverExpandedArena *
SyncCoverExpandedProgram::getArena(const SyncCoverDemand &demand) const {
  if (!*this) {
    return nullptr;
  }
  const SyncCoverScopeId arenaScope =
      demand.distance == 0 ? arenaScopeByScope_[demand.scope] : demand.scope;
  if (arenaScope == 0) {
    return &baseArena_;
  }
  const SyncCoverExpandedArena *arena = getRecurrenceArena(arenaScope);
  return arena && demand.distance <= arena->getHorizon() ? arena : nullptr;
}

const SyncCoverLoopSummary *
SyncCoverExpandedProgram::getLoopSummary(SyncCoverScopeId scope) const {
  const auto summary = loopSummaryIndices_.find(scope);
  return summary == loopSummaryIndices_.end()
             ? nullptr
             : &loopSummaries_[summary->second];
}

std::optional<std::size_t> SyncCoverExpandedProgram::projectEndpoint(
    const SyncCoverGraph &graph, const SyncCoverExpandedArena &arena,
    SyncCoverNodeId node, SyncCoverEndpointRole role, unsigned copy) const {
  const bool invalidEndpoint = !isForGraph(graph) ||
                               node >= graph.getNodes().size() ||
                               copy > arena.getHorizon();
  if (invalidEndpoint) {
    return std::nullopt;
  }
  if (const std::optional<std::size_t> local =
          arena.getVirtualOperation(node, copy)) {
    return local;
  }
  const SyncCoverNode &operation = graph.getNodes()[node];
  const std::optional<SyncCoverScopeId> child =
      getImmediateChildLoop(graph, arena.getScope(), operation.scope);
  if (!child) {
    return std::nullopt;
  }
  const SyncCoverLoopBoundaryKind boundary =
      role == SyncCoverEndpointRole::Source ? SyncCoverLoopBoundaryKind::Exit
                                            : SyncCoverLoopBoundaryKind::Entry;
  return arena.getLoopBoundary(*child, operation.resource, boundary, copy);
}

std::optional<SyncCoverProjectedCompletion>
SyncCoverExpandedProgram::projectCompletion(
    const SyncCoverGraph &graph, const SyncCoverExpandedArena &arena,
    const SyncCoverEdge &edge, unsigned sourceCopy,
    bool exportsCompletionAtScopeExit) const {
  const bool wrongGraphOrCopy =
      !isForGraph(graph) || sourceCopy > arena.getHorizon();
  if (wrongGraphOrCopy) {
    return std::nullopt;
  }
  if (edge.distance == 0 || edge.scope == arena.getScope()) {
    const bool outOfRange = edge.distance > arena.getHorizon() - sourceCopy;
    if (outOfRange) {
      return std::nullopt;
    }
    const unsigned targetCopy = sourceCopy + edge.distance;
    const std::optional<SyncCoverScopeId> sourceChild = getImmediateChildLoop(
        graph, arena.getScope(), graph.getNodes()[edge.source].scope);
    const std::optional<SyncCoverScopeId> targetChild = getImmediateChildLoop(
        graph, arena.getScope(), graph.getNodes()[edge.target].scope);
    const SyncCoverEndpointRole targetRole =
        edge.distance == 0 && sourceChild && sourceChild == targetChild
            ? SyncCoverEndpointRole::Source
            : SyncCoverEndpointRole::Target;
    const std::optional<std::size_t> source = projectEndpoint(
        graph, arena, edge.source, SyncCoverEndpointRole::Source, sourceCopy);
    const std::optional<std::size_t> target =
        projectEndpoint(graph, arena, edge.target, targetRole, targetCopy);
    return source && target
               ? std::optional<SyncCoverProjectedCompletion>(
                     {{*source, *target, sourceCopy, targetCopy, std::nullopt}})
               : std::nullopt;
  }

  const bool invalidExport =
      !exportsCompletionAtScopeExit || edge.distance == 0 ||
      !edge.sourceGuard.literals.empty() ||
      !edge.targetGuard.literals.empty() || !getLoopSummary(edge.scope) ||
      (arena.getScope() != 0 &&
       !graph.scopeContains(arena.getScope(), edge.scope)) ||
      !graph.scopeContains(edge.scope, graph.getNodes()[edge.source].scope) ||
      !graph.scopeContains(edge.scope, graph.getNodes()[edge.target].scope);
  if (invalidExport) {
    return std::nullopt;
  }
  const std::optional<std::size_t> source = projectEndpoint(
      graph, arena, edge.source, SyncCoverEndpointRole::Source, sourceCopy);
  const std::optional<std::size_t> target = projectEndpoint(
      graph, arena, edge.target, SyncCoverEndpointRole::Source, sourceCopy);
  const std::optional<SyncCoverScopeId> exportedLoop = getImmediateChildLoop(
      graph, arena.getScope(), graph.getNodes()[edge.target].scope);
  return source && target && exportedLoop
             ? std::optional<SyncCoverProjectedCompletion>(
                   {{*source, *target, sourceCopy, sourceCopy, *exportedLoop}})
             : std::nullopt;
}

std::optional<SyncCoverArenaUnavailableReason>
SyncCoverExpandedProgram::getUnavailableReason(SyncCoverScopeId scope) const {
  const auto reason = unavailableArenas_.find(scope);
  return reason == unavailableArenas_.end()
             ? std::nullopt
             : std::optional<SyncCoverArenaUnavailableReason>(reason->second);
}

SyncCoverExpandedProgram::ArenaBuildResult SyncCoverExpandedProgram::buildArena(
    const SyncCoverGraph &graph, SyncCoverScopeId scope, unsigned horizon,
    std::size_t maximumNodes, std::size_t maximumEdges,
    SyncCoverArenaUnavailableReason nodeReason,
    SyncCoverArenaUnavailableReason edgeReason) const {
  ArenaBuildResult result;
  SyncCoverExpandedArena &arena = result.arena;
  arena.scope_ = scope;
  arena.horizon_ = horizon;
  for (const SyncCoverNode &node : graph.getNodes()) {
    const SyncCoverScopeId owner = getArenaScopeForScope(graph, node.scope);
    if (owner == scope) {
      const bool nodeBudgetExhausted =
          arena.operationNodes_.size() >= maximumNodes;
      if (nodeBudgetExhausted) {
        result.unavailable = nodeReason;
        return result;
      }
      arena.operationNodes_.push_back(node.id);
    }
  }

  for (const SyncCoverLoopSummary &summary : loopSummaries_) {
    const bool immediateChild =
        scope == 0 ? !summary.parentLoop
                   : summary.parentLoop && *summary.parentLoop == scope;
    if (!immediateChild) {
      continue;
    }
    for (const SyncCoverLoopCompletionTransfer &transfer :
         summary.completionTransfers) {
      arena.loopSummaryResources_.push_back({summary.scope, transfer.resource});
    }
  }
  std::sort(arena.loopSummaryResources_.begin(),
            arena.loopSummaryResources_.end());
  arena.loopSummaryResources_.erase(
      std::unique(arena.loopSummaryResources_.begin(),
                  arena.loopSummaryResources_.end()),
      arena.loopSummaryResources_.end());

  std::map<std::uint32_t, std::vector<SyncCoverNodeId>> carryNodes;
  if (horizon != 0) {
    for (SyncCoverNodeId node : arena.operationNodes_) {
      const std::uint32_t resource = graph.getNodes()[node].resource;
      const auto kind = graph.getResourceRecurrenceCarryKinds().find(resource);
      if (kind != graph.getResourceRecurrenceCarryKinds().end()) {
        carryNodes[resource].push_back(node);
      }
    }
    for (const auto &[resource, nodes] : carryNodes) {
      (void)nodes;
      arena.carryResources_.push_back(resource);
    }
    for (const auto &[loop, resource] : arena.loopSummaryResources_) {
      (void)loop;
      const bool hasCarry =
          graph.getResourceRecurrenceCarryKinds().count(resource) != 0;
      if (hasCarry) {
        arena.carryResources_.push_back(resource);
      }
    }
    std::sort(arena.carryResources_.begin(), arena.carryResources_.end());
    arena.carryResources_.erase(
        std::unique(arena.carryResources_.begin(), arena.carryResources_.end()),
        arena.carryResources_.end());
  }

  std::size_t copyCount = 0;
  std::size_t summaryNodesPerCopy = 0;
  std::size_t carryBoundaryCount = 0;
  const bool invalidNodeCount =
      !checkedAdd(static_cast<std::size_t>(horizon), 1, copyCount) ||
      !checkedMultiply(arena.operationNodes_.size(), copyCount,
                       arena.operationVirtualNodeCount_) ||
      !checkedMultiply(arena.loopSummaryResources_.size(), 2,
                       summaryNodesPerCopy) ||
      !checkedMultiply(summaryNodesPerCopy, copyCount,
                       arena.loopSummaryVirtualNodeCount_) ||
      !checkedMultiply(arena.carryResources_.size(), horizon,
                       carryBoundaryCount) ||
      !checkedAdd(arena.operationVirtualNodeCount_,
                  arena.loopSummaryVirtualNodeCount_,
                  arena.virtualNodeCount_) ||
      !checkedAdd(arena.virtualNodeCount_, carryBoundaryCount,
                  arena.virtualNodeCount_) ||
      arena.virtualNodeCount_ > maximumNodes;
  if (invalidNodeCount) {
    result.unavailable = nodeReason;
    return result;
  }
  result.loopSummaryNodes = arena.loopSummaryVirtualNodeCount_;

  auto appendEdge = [&](SyncCoverExpandedEdge edge, bool compactCarry,
                        bool zeroTrip) {
    const bool full = arena.edges_.size() >= maximumEdges;
    if (full) {
      return false;
    }
    arena.edges_.push_back(std::move(edge));
    result.compactCarryEdges += compactCarry ? 1 : 0;
    result.zeroTripEdges += zeroTrip ? 1 : 0;
    return true;
  };

  for (const auto &[loop, resource] : arena.loopSummaryResources_) {
    const SyncCoverLoopSummary *summary = getLoopSummary(loop);
    if (!summary) {
      continue;
    }
    for (unsigned copy = 0; copy <= horizon; ++copy) {
      const std::optional<std::size_t> entry = arena.getLoopBoundary(
          loop, resource, SyncCoverLoopBoundaryKind::Entry, copy);
      const std::optional<std::size_t> exit = arena.getLoopBoundary(
          loop, resource, SyncCoverLoopBoundaryKind::Exit, copy);
      const bool appended =
          entry && exit &&
          appendEdge({*entry, *exit,
                      SyncCoverEdgeKind::CompletionPreservingIssueOrder,
                      std::nullopt, copy, copy},
                     false, summary->zeroTripPossible);
      if (!appended) {
        result.unavailable = edgeReason;
        return result;
      }
    }
  }

  for (std::size_t edgeId = 0; edgeId < graph.getEdges().size(); ++edgeId) {
    const SyncCoverEdge &edge = graph.getEdges()[edgeId];
    const std::optional<SyncCoverScopeId> sourceChild = getImmediateChildLoop(
        graph, scope, graph.getNodes()[edge.source].scope);
    const std::optional<SyncCoverScopeId> targetChild = getImmediateChildLoop(
        graph, scope, graph.getNodes()[edge.target].scope);
    const bool delegatedToChild =
        edge.distance == 0 && sourceChild && sourceChild == targetChild &&
        edge.kind != SyncCoverEdgeKind::CompletionSupply;
    if (delegatedToChild || edge.distance > horizon ||
        (edge.distance != 0 && edge.scope != scope)) {
      continue;
    }
    for (std::size_t sourceCopy = 0;
         sourceCopy <= static_cast<std::size_t>(horizon - edge.distance);
         ++sourceCopy) {
      const unsigned sourceCopyValue = static_cast<unsigned>(sourceCopy);
      const unsigned targetCopy = sourceCopyValue + edge.distance;
      const std::optional<std::size_t> source =
          projectEndpoint(graph, arena, edge.source,
                          SyncCoverEndpointRole::Source, sourceCopyValue);
      const SyncCoverEndpointRole targetRole =
          edge.distance == 0 && sourceChild && sourceChild == targetChild
              ? SyncCoverEndpointRole::Source
              : SyncCoverEndpointRole::Target;
      const std::optional<std::size_t> target =
          projectEndpoint(graph, arena, edge.target, targetRole, targetCopy);
      if (!source || !target) {
        continue;
      }
      const bool appended = appendEdge(
          {*source, *target, edge.kind, edgeId, sourceCopyValue, targetCopy},
          false, false);
      if (!appended) {
        result.unavailable = edgeReason;
        return result;
      }
    }
  }

  for (std::size_t resourceIndex = 0;
       resourceIndex < arena.carryResources_.size(); ++resourceIndex) {
    const std::uint32_t resource = arena.carryResources_[resourceIndex];
    const SyncCoverEdgeKind carryKind =
        graph.getResourceRecurrenceCarryKinds().at(resource);
    for (unsigned copy = 0; copy < horizon; ++copy) {
      const std::size_t boundary =
          arena.operationVirtualNodeCount_ +
          arena.loopSummaryVirtualNodeCount_ +
          static_cast<std::size_t>(copy) * arena.carryResources_.size() +
          resourceIndex;
      const auto localCarryNodes = carryNodes.find(resource);
      if (localCarryNodes != carryNodes.end()) {
        for (SyncCoverNodeId node : localCarryNodes->second) {
          const SyncCoverExpandedEdge intoBoundary{
              *arena.getVirtualOperation(node, copy),
              boundary,
              carryKind,
              std::nullopt,
              copy,
              copy};
          const SyncCoverExpandedEdge fromBoundary{
              boundary,  *arena.getVirtualOperation(node, copy + 1),
              carryKind, std::nullopt,
              copy,      static_cast<unsigned>(copy + 1)};
          const bool appended = appendEdge(intoBoundary, true, false) &&
                                appendEdge(fromBoundary, true, false);
          if (!appended) {
            result.unavailable = edgeReason;
            return result;
          }
        }
      }
      for (const auto &[loop, summaryResource] : arena.loopSummaryResources_) {
        if (summaryResource != resource) {
          continue;
        }
        const std::optional<std::size_t> exit = arena.getLoopBoundary(
            loop, resource, SyncCoverLoopBoundaryKind::Exit, copy);
        const std::optional<std::size_t> entry = arena.getLoopBoundary(
            loop, resource, SyncCoverLoopBoundaryKind::Entry, copy + 1);
        const bool appended =
            exit && entry &&
            appendEdge({*exit, boundary, carryKind, std::nullopt, copy, copy},
                       true, false) &&
            appendEdge({boundary, *entry, carryKind, std::nullopt, copy,
                        static_cast<unsigned>(copy + 1)},
                       true, false);
        if (!appended) {
          result.unavailable = edgeReason;
          return result;
        }
      }
    }
  }

  std::sort(
      arena.edges_.begin(), arena.edges_.end(),
      [](const SyncCoverExpandedEdge &left,
         const SyncCoverExpandedEdge &right) {
        return std::tie(left.source, left.target, left.kind, left.graphEdge,
                        left.sourceCopy, left.targetCopy) <
               std::tie(right.source, right.target, right.kind, right.graphEdge,
                        right.sourceCopy, right.targetCopy);
      });
  arena.edges_.erase(
      std::unique(arena.edges_.begin(), arena.edges_.end(),
                  [](const SyncCoverExpandedEdge &left,
                     const SyncCoverExpandedEdge &right) {
                    return std::tie(left.source, left.target, left.kind,
                                    left.graphEdge, left.sourceCopy,
                                    left.targetCopy) ==
                           std::tie(right.source, right.target, right.kind,
                                    right.graphEdge, right.sourceCopy,
                                    right.targetCopy);
                  }),
      arena.edges_.end());
  arena.outgoingOffsets_.assign(arena.virtualNodeCount_ + 1, 0);
  for (const SyncCoverExpandedEdge &edge : arena.edges_) {
    ++arena.outgoingOffsets_[edge.source + 1];
  }
  for (std::size_t index = 1; index < arena.outgoingOffsets_.size(); ++index) {
    arena.outgoingOffsets_[index] += arena.outgoingOffsets_[index - 1];
  }
  return result;
}
