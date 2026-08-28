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

} // namespace

struct SyncCoverExpandedProgram::ArenaBuildResult {
  SyncCoverExpandedArena arena;
  std::optional<SyncCoverArenaUnavailableReason> unavailable;
  std::size_t compactCarryEdges = 0;
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
  if (carryResources_.empty()) {
    return std::nullopt;
  }
  return static_cast<unsigned>((virtualNode - operationVirtualNodeCount_) /
                               carryResources_.size());
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

  std::map<SyncCoverScopeId, unsigned> horizons;
  for (std::size_t demandId : activeDemands) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (demand.distance != 0) {
      horizons[demand.scope] =
          std::max(horizons[demand.scope], demand.distance);
    }
  }
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
  if (demand.distance == 0) {
    return &baseArena_;
  }
  const SyncCoverExpandedArena *arena = getRecurrenceArena(demand.scope);
  return arena && demand.distance <= arena->getHorizon() ? arena : nullptr;
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
    if (scope == 0 || graph.scopeContains(scope, node.scope)) {
      const bool nodeBudgetExhausted =
          arena.operationNodes_.size() >= maximumNodes;
      if (nodeBudgetExhausted) {
        result.unavailable = nodeReason;
        return result;
      }
      arena.operationNodes_.push_back(node.id);
    }
  }
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
  }

  std::size_t copyCount = 0;
  std::size_t boundaryCount = 0;
  const bool invalidNodeCount =
      !checkedAdd(static_cast<std::size_t>(horizon), 1, copyCount) ||
      !checkedMultiply(arena.operationNodes_.size(), copyCount,
                       arena.operationVirtualNodeCount_) ||
      !checkedMultiply(arena.carryResources_.size(), horizon, boundaryCount) ||
      !checkedAdd(arena.operationVirtualNodeCount_, boundaryCount,
                  arena.virtualNodeCount_) ||
      arena.virtualNodeCount_ > maximumNodes;
  if (invalidNodeCount) {
    result.unavailable = nodeReason;
    return result;
  }

  std::size_t predictedEdgeCount = 0;
  for (const SyncCoverEdge &edge : graph.getEdges()) {
    const bool included = arena.getVirtualOperation(edge.source, 0) &&
                          arena.getVirtualOperation(edge.target, 0) &&
                          edge.distance <= horizon &&
                          (edge.distance == 0 || edge.scope == scope);
    if (!included ||
        !checkedAdd(predictedEdgeCount,
                    static_cast<std::size_t>(horizon - edge.distance) + 1,
                    predictedEdgeCount)) {
      if (included) {
        result.unavailable = edgeReason;
        return result;
      }
    }
  }
  std::size_t carryEdgeCount = 0;
  for (const auto &[resource, nodes] : carryNodes) {
    (void)resource;
    std::size_t resourceEdges = 0;
    const bool invalidCarryCount =
        !checkedMultiply(nodes.size(), static_cast<std::size_t>(horizon),
                         resourceEdges) ||
        !checkedMultiply(resourceEdges, 2, resourceEdges) ||
        !checkedAdd(carryEdgeCount, resourceEdges, carryEdgeCount);
    if (invalidCarryCount) {
      result.unavailable = edgeReason;
      return result;
    }
  }
  const bool invalidEdgeCount =
      !checkedAdd(predictedEdgeCount, carryEdgeCount, predictedEdgeCount) ||
      predictedEdgeCount > maximumEdges;
  if (invalidEdgeCount) {
    result.unavailable = edgeReason;
    return result;
  }
  arena.edges_.reserve(predictedEdgeCount);

  auto appendEdge = [&](SyncCoverExpandedEdge edge, bool compactCarry) {
    const bool full = arena.edges_.size() >= maximumEdges;
    if (full) {
      return false;
    }
    arena.edges_.push_back(std::move(edge));
    result.compactCarryEdges += compactCarry ? 1 : 0;
    return true;
  };
  for (std::size_t edgeId = 0; edgeId < graph.getEdges().size(); ++edgeId) {
    const SyncCoverEdge &edge = graph.getEdges()[edgeId];
    const std::optional<std::size_t> localSource =
        arena.getVirtualOperation(edge.source, 0);
    const std::optional<std::size_t> localTarget =
        arena.getVirtualOperation(edge.target, 0);
    if (!localSource || !localTarget || edge.distance > horizon ||
        (edge.distance != 0 && edge.scope != scope)) {
      continue;
    }
    for (std::size_t sourceCopy = 0;
         sourceCopy <= static_cast<std::size_t>(horizon - edge.distance);
         ++sourceCopy) {
      const unsigned sourceCopyValue = static_cast<unsigned>(sourceCopy);
      const unsigned targetCopy = sourceCopyValue + edge.distance;
      const SyncCoverExpandedEdge expanded{
          *arena.getVirtualOperation(edge.source, sourceCopyValue),
          *arena.getVirtualOperation(edge.target, targetCopy),
          edge.kind,
          edgeId,
          sourceCopyValue,
          targetCopy};
      if (!appendEdge(expanded, false)) {
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
          static_cast<std::size_t>(copy) * arena.carryResources_.size() +
          resourceIndex;
      for (SyncCoverNodeId node : carryNodes.at(resource)) {
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
        const bool appended =
            appendEdge(intoBoundary, true) && appendEdge(fromBoundary, true);
        if (!appended) {
          result.unavailable = edgeReason;
          return result;
        }
      }
    }
  }

  std::sort(arena.edges_.begin(), arena.edges_.end(),
            [](const SyncCoverExpandedEdge &left,
               const SyncCoverExpandedEdge &right) {
              return std::tie(left.source, left.target, left.kind,
                              left.graphEdge) <
                     std::tie(right.source, right.target, right.kind,
                              right.graphEdge);
            });
  arena.outgoingOffsets_.assign(arena.virtualNodeCount_ + 1, 0);
  for (const SyncCoverExpandedEdge &edge : arena.edges_) {
    ++arena.outgoingOffsets_[edge.source + 1];
  }
  for (std::size_t index = 1; index < arena.outgoingOffsets_.size(); ++index) {
    arena.outgoingOffsets_[index] += arena.outgoingOffsets_[index - 1];
  }
  return result;
}
