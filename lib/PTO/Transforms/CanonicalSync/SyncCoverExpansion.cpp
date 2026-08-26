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
#include <utility>

using namespace mlir::pto;

namespace {

bool checkedAdd(std::size_t value, std::size_t &total) {
  if (value > std::numeric_limits<std::size_t>::max() - total) {
    return false;
  }
  total += value;
  return true;
}

bool checkedVirtualNodeCount(std::size_t nodeCount, unsigned horizon,
                             std::size_t maximumNodes, std::size_t &result) {
  const std::size_t copyCount = static_cast<std::size_t>(horizon) + 1;
  if (copyCount == 0) {
    return false;
  }
  if (nodeCount == 0) {
    result = 0;
    return true;
  }
  if (copyCount > maximumNodes / nodeCount) {
    return false;
  }
  result = copyCount * nodeCount;
  return result <= maximumNodes;
}

bool edgeBelongsToArena(const SyncCoverEdge &edge,
                        const SyncCoverExpandedArena &arena) {
  if (edge.distance > arena.getHorizon()) {
    return false;
  }
  if (edge.distance != 0 && edge.scope != arena.getScope()) {
    return false;
  }
  return arena.getLocalNode(edge.source).has_value() &&
         arena.getLocalNode(edge.target).has_value();
}

} // namespace

SyncCoverExpandedEdgeRange SyncCoverExpandedAdjacency::getOutgoingEdges(
    std::size_t virtualNode) const {
  if (outgoingOffsets_.empty() ||
      virtualNode >= outgoingOffsets_.size() - 1) {
    return {edges_.end(), edges_.end()};
  }
  return {edges_.begin() +
              static_cast<std::ptrdiff_t>(outgoingOffsets_[virtualNode]),
          edges_.begin() + static_cast<std::ptrdiff_t>(
                               outgoingOffsets_[virtualNode + 1])};
}

std::optional<std::size_t>
SyncCoverExpandedArena::getLocalNode(SyncCoverNodeId globalNode) const {
  const auto position =
      std::lower_bound(globalNodes_.begin(), globalNodes_.end(), globalNode);
  if (position == globalNodes_.end() || *position != globalNode) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(position - globalNodes_.begin());
}

std::optional<std::size_t> SyncCoverExpandedArena::getVirtualNode(
    SyncCoverNodeId globalNode, unsigned copy) const {
  const std::optional<std::size_t> local = getLocalNode(globalNode);
  if (!local || copy > horizon_) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(copy) * globalNodes_.size() + *local;
}

SyncCoverExpandedProgram::SyncCoverExpandedProgram(
    const SyncCoverGraph &graph, SyncCoverExpansionLimits limits)
    : owner_(&graph), limits_(limits), graphGeneration_(graph.getGeneration()),
      structuralGeneration_(graph.getStructuralGeneration()) {
  if (!graph.isStructureFrozen() || !graph.validate()) {
    error_ = SyncCoverExpansionError::InvalidGraph;
    return;
  }

  std::map<SyncCoverScopeId, unsigned> horizons;
  for (const SyncCoverDemand &demand : graph.getDemands()) {
    if (demand.distance != 0) {
      horizons[demand.scope] =
          std::max(horizons[demand.scope], demand.distance);
    }
  }

  std::size_t remainingNodes = limits_.maximumNodes;
  error_ = buildArenaShape(graph, 0, 0, remainingNodes, baseArena_);
  if (error_ != SyncCoverExpansionError::None) {
    return;
  }
  for (const auto &[scope, horizon] : horizons) {
    SyncCoverExpandedArena arena;
    error_ = buildArenaShape(graph, scope, horizon, remainingNodes, arena);
    if (error_ != SyncCoverExpansionError::None) {
      recurrenceArenas_.clear();
      return;
    }
    recurrenceArenas_.emplace(scope, std::move(arena));
  }

  std::size_t remainingEdges = limits_.maximumEdges;
  auto buildStructural = [&](SyncCoverExpandedArena &arena) {
    return buildAdjacency(graph, 0, graph.getStructuralEdgeCount(), arena,
                          remainingEdges, arena.structuralEdges_);
  };
  error_ = buildStructural(baseArena_);
  for (auto &entry : recurrenceArenas_) {
    if (error_ != SyncCoverExpansionError::None) {
      break;
    }
    error_ = buildStructural(entry.second);
  }
  if (error_ != SyncCoverExpansionError::None) {
    recurrenceArenas_.clear();
    return;
  }
  structuralExpandedEdges_ = limits_.maximumEdges - remainingEdges;
  error_ = refreshMechanismOverlay(graph);
}

bool SyncCoverExpandedProgram::isStructuralCurrent(
    const SyncCoverGraph &graph) const {
  return *this && &graph == owner_ && graph.isStructureFrozen() &&
         graph.getStructuralGeneration() == structuralGeneration_;
}

bool SyncCoverExpandedProgram::isCurrent(const SyncCoverGraph &graph) const {
  return isStructuralCurrent(graph) &&
         graph.getGeneration() == graphGeneration_;
}

SyncCoverExpansionError SyncCoverExpandedProgram::refreshMechanismOverlay(
    const SyncCoverGraph &graph) {
  if (!isStructuralCurrent(graph) || !graph.validate() ||
      graph.getStructuralEdgeCount() > graph.getEdges().size()) {
    return SyncCoverExpansionError::InvalidGraph;
  }

  std::size_t remainingEdges = limits_.maximumEdges - structuralExpandedEdges_;
  SyncCoverExpandedAdjacency baseOverlay;
  SyncCoverExpansionError result = buildAdjacency(
      graph, graph.getStructuralEdgeCount(), graph.getEdges().size(),
      baseArena_, remainingEdges, baseOverlay);
  std::map<SyncCoverScopeId, SyncCoverExpandedAdjacency> recurrenceOverlays;
  for (const auto &[scope, arena] : recurrenceArenas_) {
    if (result != SyncCoverExpansionError::None) {
      break;
    }
    SyncCoverExpandedAdjacency overlay;
    result = buildAdjacency(graph, graph.getStructuralEdgeCount(),
                            graph.getEdges().size(), arena, remainingEdges,
                            overlay);
    recurrenceOverlays.emplace(scope, std::move(overlay));
  }
  if (result != SyncCoverExpansionError::None) {
    return result;
  }

  baseArena_.mechanismEdges_ = std::move(baseOverlay);
  for (auto &entry : recurrenceArenas_) {
    entry.second.mechanismEdges_ =
        std::move(recurrenceOverlays.at(entry.first));
  }
  graphGeneration_ = graph.getGeneration();
  return SyncCoverExpansionError::None;
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
  if (demand.distance == 0) {
    return &baseArena_;
  }
  const SyncCoverExpandedArena *arena = getRecurrenceArena(demand.scope);
  return arena && demand.distance <= arena->getHorizon() ? arena : nullptr;
}

SyncCoverExpansionError SyncCoverExpandedProgram::buildArenaShape(
    const SyncCoverGraph &graph, SyncCoverScopeId scope, unsigned horizon,
    std::size_t &remainingNodes, SyncCoverExpandedArena &arena) {
  arena.scope_ = scope;
  arena.horizon_ = horizon;
  for (const SyncCoverNode &node : graph.getNodes()) {
    const bool inRecurrenceSubtree = graph.scopeContains(scope, node.scope);
    const bool inAncestorScope = graph.scopeContains(node.scope, scope);
    if (inRecurrenceSubtree || inAncestorScope) {
      arena.globalNodes_.push_back(node.id);
    }
  }
  if (!std::is_sorted(arena.globalNodes_.begin(), arena.globalNodes_.end())) {
    return SyncCoverExpansionError::InvalidGraph;
  }
  if (!checkedVirtualNodeCount(arena.globalNodes_.size(), horizon,
                               limits_.maximumNodes,
                               arena.virtualNodeCount_) ||
      arena.virtualNodeCount_ > remainingNodes) {
    return SyncCoverExpansionError::ExpansionLimitExceeded;
  }
  remainingNodes -= arena.virtualNodeCount_;
  return SyncCoverExpansionError::None;
}

SyncCoverExpansionError SyncCoverExpandedProgram::buildAdjacency(
    const SyncCoverGraph &graph, std::size_t edgeBegin, std::size_t edgeEnd,
    const SyncCoverExpandedArena &arena, std::size_t &remainingEdges,
    SyncCoverExpandedAdjacency &adjacency) const {
  if (edgeBegin > edgeEnd || edgeEnd > graph.getEdges().size()) {
    return SyncCoverExpansionError::InvalidGraph;
  }

  std::size_t expandedEdgeCount = 0;
  for (std::size_t edgeId = edgeBegin; edgeId < edgeEnd; ++edgeId) {
    const SyncCoverEdge &edge = graph.getEdges()[edgeId];
    if (!edgeBelongsToArena(edge, arena)) {
      continue;
    }
    const std::size_t copies =
        static_cast<std::size_t>(arena.getHorizon() - edge.distance) + 1;
    if (!checkedAdd(copies, expandedEdgeCount) ||
        expandedEdgeCount > remainingEdges) {
      return SyncCoverExpansionError::ExpansionLimitExceeded;
    }
  }

  adjacency.outgoingOffsets_.assign(arena.getVirtualNodeCount() + 1, 0);
  auto visitOccurrences = [&](auto &&visitor) {
    for (std::size_t edgeId = edgeBegin; edgeId < edgeEnd; ++edgeId) {
      const SyncCoverEdge &edge = graph.getEdges()[edgeId];
      if (!edgeBelongsToArena(edge, arena)) {
        continue;
      }
      const std::size_t source = *arena.getLocalNode(edge.source);
      const std::size_t target = *arena.getLocalNode(edge.target);
      for (unsigned sourceCopy = 0;
           sourceCopy + edge.distance <= arena.getHorizon(); ++sourceCopy) {
        const unsigned targetCopy = sourceCopy + edge.distance;
        visitor(SyncCoverExpandedEdge{
            edgeId, sourceCopy * arena.getNodeCount() + source,
            targetCopy * arena.getNodeCount() + target, sourceCopy,
            targetCopy});
      }
    }
  };
  visitOccurrences([&](const SyncCoverExpandedEdge &edge) {
    ++adjacency.outgoingOffsets_[edge.source + 1];
  });
  for (std::size_t node = 1; node < adjacency.outgoingOffsets_.size(); ++node) {
    adjacency.outgoingOffsets_[node] += adjacency.outgoingOffsets_[node - 1];
  }

  adjacency.edges_.resize(expandedEdgeCount);
  std::vector<std::size_t> insertion = adjacency.outgoingOffsets_;
  visitOccurrences([&](const SyncCoverExpandedEdge &edge) {
    adjacency.edges_[insertion[edge.source]++] = edge;
  });
  remainingEdges -= expandedEdgeCount;
  return SyncCoverExpansionError::None;
}
