// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

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

namespace {

constexpr unsigned kHardwareEventIdCount = 8;

using EventDomainKey = std::pair<std::uint32_t, std::uint32_t>;

struct BarrierFallbackGroup {
  std::vector<SyncCoverDemandId> demands;
};

struct DirectEventRecord {
  SyncCoverDemandId demand = 0;
  CanonicalSyncMechanismId mechanism = 0;
  CanonicalSyncEventDomainId domain = 0;
};

/// Exact per-demand event/protocol records retained for physical-slot
/// lifecycle discovery. Unlike DirectEventRecord, recurrence protocols belong
/// here and must not enter allocation-frontier repair generation.
struct ExactEventRecord {
  SyncCoverDemandId demand = 0;
  CanonicalSyncMechanismId mechanism = 0;
  CanonicalSyncEventDomainId domain = 0;
};

SyncCoverEdge getDemandEdge(const SyncCoverDemand &demand) {
  return {
      demand.source,     demand.target,   SyncCoverEdgeKind::CompletionSupply,
      demand.scope,      demand.distance, demand.sourceGuard,
      demand.targetGuard};
}

std::vector<SyncCoverDemandId> getActiveDemands(const SyncCoverGraph &graph) {
  std::vector<SyncCoverDemandId> demands(graph.getDemands().size());
  std::iota(demands.begin(), demands.end(), 0);
  return demands;
}

struct DemandBasisResult {
  std::vector<SyncCoverDemandId> demands;
  bool truncated = false;
};

using DemandBasisGroupKey = SyncCoverScopeId;

std::optional<DemandBasisGroupKey>
getDemandBasisGroup(const SyncCoverGraph &graph,
                    const SyncCoverDemand &demand) {
  if (demand.distance != 0 || demand.storageWitnesses.empty() ||
      demand.sourceGuard.literals.size() != 0 ||
      demand.targetGuard.literals.size() != 0 ||
      llvm::is_contained(demand.provenanceKinds, SyncCoverDemandKind::SSA)) {
    return std::nullopt;
  }
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  if (source.scope != demand.scope || target.scope != demand.scope ||
      source.guard.literals.size() != 0 || target.guard.literals.size() != 0 ||
      source.order >= target.order ||
      demand.scope >= graph.getScopes().size() ||
      graph.getScopes()[demand.scope].guard.literals.size() != 0) {
    return std::nullopt;
  }
  return demand.scope;
}

bool checkedBasisProduct(std::size_t first, std::size_t second,
                         std::size_t &result) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

bool checkedBasisSum(std::size_t first, std::size_t second,
                     std::size_t &result) {
  if (second > std::numeric_limits<std::size_t>::max() - first) {
    return false;
  }
  result = first + second;
  return true;
}

DemandBasisResult
buildDemandSelectionBasis(const SyncCoverGraph &graph,
                          const SyncCoverExpandedProgram &expansion,
                          const CanonicalSyncBuildOptions &options) {
  DemandBasisResult result{getActiveDemands(graph), false};
  if (!options.enableDemandBasisReduction || result.demands.empty()) {
    return result;
  }
  // Dense scope IDs permit direct indexing. Charge every full-universe scan
  // and scope-vector slot before allocating group workspaces. Basis reduction
  // is optional, so an insufficient budget retains every row.
  std::size_t demandScans = 0;
  std::size_t globalWork = 0;
  const bool globalOverflow =
      !checkedBasisProduct(result.demands.size(), 4, demandScans) ||
      !checkedBasisSum(demandScans, graph.getScopes().size(), globalWork);
  if (globalOverflow || globalWork > options.maximumDemandBasisReductionWork) {
    result.truncated = true;
    return result;
  }
  std::vector<std::vector<SyncCoverDemandId>> groups(graph.getScopes().size());
  for (SyncCoverDemandId demandId : result.demands) {
    const auto key = getDemandBasisGroup(graph, graph.getDemands()[demandId]);
    if (key) {
      groups[*key].push_back(demandId);
    }
  }
  std::vector<bool> retained(graph.getDemands().size(), true);
  std::size_t usedWords = 0;
  std::size_t usedWork = globalWork;
  for (auto &demandIds : groups) {
    if (demandIds.size() < 3) {
      continue;
    }
    std::size_t maximumNodes = 0;
    std::size_t wordsPerMaximumRow = 0;
    std::size_t groupWords = 0;
    std::size_t nodeSquare = 0;
    std::size_t edgeSquare = 0;
    std::size_t outgoingLookupWork = 0;
    std::size_t propagationWork = 0;
    std::size_t groupWork = 0;
    bool initialOverflow =
        !checkedBasisProduct(demandIds.size(), 2, maximumNodes);
    if (!initialOverflow) {
      wordsPerMaximumRow = maximumNodes / 64 + (maximumNodes % 64 != 0);
      initialOverflow =
          !checkedBasisProduct(maximumNodes, wordsPerMaximumRow, groupWords) ||
          !checkedBasisProduct(maximumNodes, maximumNodes, nodeSquare) ||
          !checkedBasisProduct(demandIds.size(), demandIds.size(),
                               edgeSquare) ||
          !checkedBasisProduct(demandIds.size(), maximumNodes,
                               outgoingLookupWork) ||
          !checkedBasisProduct(demandIds.size(), wordsPerMaximumRow + 2,
                               propagationWork) ||
          // Two node-quadratic terms bound node sorting and ordered-map
          // indexing. The edge-quadratic term bounds all outgoing-edge sorts.
          !checkedBasisSum(nodeSquare, nodeSquare, groupWork) ||
          !checkedBasisSum(groupWork, edgeSquare, groupWork) ||
          !checkedBasisSum(groupWork, outgoingLookupWork, groupWork) ||
          !checkedBasisSum(groupWork, propagationWork, groupWork) ||
          !checkedBasisSum(groupWork, groupWords, groupWork) ||
          // Linear construction, unique, reverse, and edge scans.
          !checkedBasisSum(groupWork, maximumNodes, groupWork) ||
          !checkedBasisSum(groupWork, maximumNodes, groupWork) ||
          !checkedBasisSum(groupWork, maximumNodes, groupWork) ||
          !checkedBasisSum(groupWork, demandIds.size(), groupWork) ||
          !checkedBasisSum(groupWork, demandIds.size(), groupWork);
    }
    const bool exceedsGroupEdges =
        demandIds.size() > options.maximumDemandBasisGroupEdges;
    const bool exceedsWords =
        initialOverflow ||
        groupWords > options.maximumDemandBasisReachabilityWords ||
        usedWords > options.maximumDemandBasisReachabilityWords - groupWords;
    const bool exceedsWork =
        initialOverflow ||
        groupWork > options.maximumDemandBasisReductionWork ||
        usedWork > options.maximumDemandBasisReductionWork - groupWork;
    if (initialOverflow || exceedsGroupEdges || exceedsWords || exceedsWork) {
      result.truncated = true;
      continue;
    }
    std::vector<SyncCoverNodeId> nodes;
    nodes.reserve(maximumNodes);
    for (SyncCoverDemandId demandId : demandIds) {
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      nodes.push_back(demand.source);
      nodes.push_back(demand.target);
    }
    llvm::sort(nodes, [&](SyncCoverNodeId first, SyncCoverNodeId second) {
      return std::tie(graph.getNodes()[first].order, first) <
             std::tie(graph.getNodes()[second].order, second);
    });
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    usedWords += groupWords;
    usedWork += groupWork;
    std::map<SyncCoverNodeId, std::size_t> nodeIndices;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      nodeIndices.emplace(nodes[index], index);
    }
    std::vector<std::vector<SyncCoverDemandId>> outgoing(nodes.size());
    for (SyncCoverDemandId demandId : demandIds) {
      outgoing[nodeIndices[graph.getDemands()[demandId].source]].push_back(
          demandId);
    }
    for (auto &edges : outgoing) {
      llvm::sort(edges, [&](SyncCoverDemandId first, SyncCoverDemandId second) {
        const SyncCoverNode &firstTarget =
            graph.getNodes()[graph.getDemands()[first].target];
        const SyncCoverNode &secondTarget =
            graph.getNodes()[graph.getDemands()[second].target];
        return std::tie(firstTarget.order, firstTarget.id, first) <
               std::tie(secondTarget.order, secondTarget.id, second);
      });
    }
    std::vector<llvm::BitVector> reachable(
        nodes.size(), llvm::BitVector(nodes.size(), false));
    for (std::size_t reverse = nodes.size(); reverse > 0; --reverse) {
      const std::size_t sourceIndex = reverse - 1;
      for (SyncCoverDemandId demandId : outgoing[sourceIndex]) {
        const std::size_t targetIndex =
            nodeIndices[graph.getDemands()[demandId].target];
        if (reachable[sourceIndex].test(targetIndex)) {
          retained[demandId] = false;
          continue;
        }
        reachable[sourceIndex].set(targetIndex);
        reachable[sourceIndex] |= reachable[targetIndex];
      }
    }
  }

  // Reduce exact recurrence implications only inside one loop-local,
  // unguarded memory context.  A retained obligation is an abstract
  // completion-supply edge in every valid virtual copy.  The virtual-copy
  // coordinate makes path distance exact: d1+d1 may imply d2, but can never
  // imply d1 or d3.  The original obligation universe remains unchanged and
  // is checked again by fresh verification after materialization.
  std::vector<std::vector<SyncCoverDemandId>> recurrenceGroups(
      graph.getScopes().size());
  std::vector<bool> groupHasPositiveDistance(graph.getScopes().size(), false);
  for (SyncCoverDemandId demandId : result.demands) {
    if (!retained[demandId]) {
      continue;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (demand.scope >= graph.getScopes().size() ||
        !graph.getScopes()[demand.scope].isLoop ||
        !graph.getScopes()[demand.scope].guard.literals.empty() ||
        demand.storageWitnesses.empty() ||
        llvm::is_contained(demand.provenanceKinds, SyncCoverDemandKind::SSA) ||
        !demand.sourceGuard.literals.empty() ||
        !demand.targetGuard.literals.empty()) {
      continue;
    }
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    if (source.scope != demand.scope || target.scope != demand.scope ||
        !source.guard.literals.empty() || !target.guard.literals.empty() ||
        (demand.distance == 0 && source.order >= target.order)) {
      continue;
    }
    const SyncCoverExpandedArena *arena = expansion.getArena(demand);
    if (!arena || arena->getScope() != demand.scope ||
        demand.distance > arena->getHorizon()) {
      continue;
    }
    recurrenceGroups[demand.scope].push_back(demandId);
    groupHasPositiveDistance[demand.scope] =
        groupHasPositiveDistance[demand.scope] || demand.distance != 0;
  }

  struct RecurrenceNode {
    unsigned copy = 0;
    SyncCoverNodeId node = 0;
  };
  struct RecurrenceEdge {
    std::size_t target = 0;
    std::optional<SyncCoverDemandId> demand;
  };
  const auto sortFactor = [](std::size_t size) {
    std::size_t factor = 0;
    for (std::size_t remaining = size; remaining > 1;
         remaining = (remaining + 1) / 2) {
      ++factor;
    }
    return factor;
  };
  for (SyncCoverScopeId scope = 0; scope < recurrenceGroups.size(); ++scope) {
    const std::vector<SyncCoverDemandId> &demandIds = recurrenceGroups[scope];
    if (!groupHasPositiveDistance[scope] || demandIds.size() < 2) {
      continue;
    }
    if (demandIds.size() > options.maximumDemandBasisGroupEdges) {
      result.truncated = true;
      continue;
    }
    const SyncCoverExpandedArena *arena = expansion.getRecurrenceArena(scope);
    if (!arena) {
      result.truncated = true;
      continue;
    }

    std::size_t projectedEdges = 0;
    bool overflow = false;
    for (SyncCoverDemandId demandId : demandIds) {
      const unsigned distance = graph.getDemands()[demandId].distance;
      std::size_t instances =
          static_cast<std::size_t>(arena->getHorizon() - distance) + 1;
      overflow = overflow ||
                 !checkedBasisSum(projectedEdges, instances, projectedEdges);
    }
    std::size_t projectedEndpointNodes = 0;
    std::size_t operationInstances = 0;
    std::size_t maximumProjectedNodes = 0;
    std::size_t projectionWork = 0;
    if (!overflow) {
      overflow =
          !checkedBasisProduct(projectedEdges, 2, projectedEndpointNodes) ||
          !checkedBasisProduct(arena->getOperationNodeCount(),
                               static_cast<std::size_t>(arena->getHorizon()) +
                                   1,
                               operationInstances) ||
          !checkedBasisSum(projectedEndpointNodes, operationInstances,
                           maximumProjectedNodes) ||
          !checkedBasisProduct(projectedEdges, 4, projectionWork);
    }
    std::size_t preflightStateNodes = 0;
    std::size_t preflightWordsPerRow = 0;
    std::size_t preflightGroupWords = 0;
    std::size_t preflightAssumptionTransitions = 0;
    std::size_t preflightFixedTransitions = 0;
    std::size_t preflightStateEdges = 0;
    std::size_t preflightNodeSortWork = 0;
    std::size_t preflightEdgeSortWork = 0;
    std::size_t preflightLookupWork = 0;
    std::size_t preflightPropagationWork = 0;
    std::size_t preflightWork = projectionWork;
    if (!overflow) {
      overflow =
          !checkedBasisProduct(operationInstances, 2, preflightStateNodes);
    }
    if (!overflow) {
      preflightWordsPerRow =
          preflightStateNodes / 64 + (preflightStateNodes % 64 != 0);
      overflow =
          !checkedBasisProduct(preflightStateNodes, preflightWordsPerRow,
                               preflightGroupWords) ||
          !checkedBasisProduct(projectedEdges, 2,
                               preflightAssumptionTransitions) ||
          !checkedBasisProduct(arena->getEdges().size(), 2,
                               preflightFixedTransitions) ||
          !checkedBasisSum(preflightAssumptionTransitions,
                           preflightFixedTransitions, preflightStateEdges) ||
          !checkedBasisProduct(maximumProjectedNodes,
                               sortFactor(maximumProjectedNodes),
                               preflightNodeSortWork) ||
          !checkedBasisProduct(preflightStateEdges,
                               sortFactor(preflightStateEdges),
                               preflightEdgeSortWork) ||
          !checkedBasisProduct(preflightStateEdges,
                               2 * sortFactor(operationInstances),
                               preflightLookupWork) ||
          !checkedBasisProduct(preflightStateEdges, preflightWordsPerRow + 3,
                               preflightPropagationWork) ||
          !checkedBasisSum(preflightWork, preflightNodeSortWork,
                           preflightWork) ||
          !checkedBasisSum(preflightWork, preflightEdgeSortWork,
                           preflightWork) ||
          !checkedBasisSum(preflightWork, preflightLookupWork, preflightWork) ||
          !checkedBasisSum(preflightWork, preflightPropagationWork,
                           preflightWork) ||
          !checkedBasisSum(preflightWork, preflightStateNodes, preflightWork) ||
          !checkedBasisSum(preflightWork, demandIds.size(), preflightWork);
    }
    const bool preflightWordsExceeded =
        !overflow &&
        (preflightGroupWords > options.maximumDemandBasisReachabilityWords ||
         usedWords >
             options.maximumDemandBasisReachabilityWords - preflightGroupWords);
    const bool preflightWorkExceeded =
        !overflow &&
        (preflightWork > options.maximumDemandBasisReductionWork ||
         usedWork > options.maximumDemandBasisReductionWork - preflightWork);
    if (overflow || projectedEdges > options.maximumDemandBasisGroupEdges ||
        preflightWordsExceeded || preflightWorkExceeded) {
      result.truncated = true;
      continue;
    }

    std::vector<RecurrenceNode> nodes;
    nodes.reserve(maximumProjectedNodes);
    struct ProjectedRecurrenceEdge {
      RecurrenceNode source;
      RecurrenceNode target;
      SyncCoverDemandId demand = 0;
    };
    std::vector<ProjectedRecurrenceEdge> projected;
    projected.reserve(projectedEdges);
    bool projectionFailed = false;
    for (unsigned copy = 0; copy <= arena->getHorizon(); ++copy) {
      for (SyncCoverNodeId node : arena->getOperationNodes()) {
        nodes.push_back({copy, node});
      }
    }
    for (SyncCoverDemandId demandId : demandIds) {
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      for (unsigned copy = 0; copy <= arena->getHorizon() - demand.distance;
           ++copy) {
        const std::optional<std::size_t> source =
            expansion.projectEndpoint(graph, *arena, demand.source, copy);
        const std::optional<std::size_t> target = expansion.projectEndpoint(
            graph, *arena, demand.target, copy + demand.distance);
        if (!source || !target) {
          projectionFailed = true;
          break;
        }
        (void)source;
        (void)target;
        RecurrenceNode sourceNode{copy, demand.source};
        RecurrenceNode targetNode{copy + demand.distance, demand.target};
        nodes.push_back(sourceNode);
        nodes.push_back(targetNode);
        projected.push_back({sourceNode, targetNode, demandId});
      }
      if (projectionFailed) {
        break;
      }
    }
    if (projectionFailed) {
      result.truncated = true;
      continue;
    }
    const auto nodeLess = [&](const RecurrenceNode &first,
                              const RecurrenceNode &second) {
      return std::tie(first.copy, graph.getNodes()[first.node].order,
                      first.node) <
             std::tie(second.copy, graph.getNodes()[second.node].order,
                      second.node);
    };
    llvm::sort(nodes, nodeLess);
    nodes.erase(std::unique(nodes.begin(), nodes.end(),
                            [](const RecurrenceNode &first,
                               const RecurrenceNode &second) {
                              return first.copy == second.copy &&
                                     first.node == second.node;
                            }),
                nodes.end());
    std::size_t stateNodes = 0;
    std::size_t maximumAssumptionTransitions = 0;
    std::size_t maximumFixedTransitions = 0;
    std::size_t maximumStateEdges = 0;
    overflow =
        !checkedBasisProduct(nodes.size(), 2, stateNodes) ||
        !checkedBasisProduct(projectedEdges, 2, maximumAssumptionTransitions) ||
        !checkedBasisProduct(arena->getEdges().size(), 2,
                             maximumFixedTransitions) ||
        !checkedBasisSum(maximumAssumptionTransitions, maximumFixedTransitions,
                         maximumStateEdges);
    const std::size_t wordsPerRow = stateNodes / 64 + (stateNodes % 64 != 0);
    std::size_t groupWords = 0;
    std::size_t nodeSortWork = 0;
    std::size_t edgeSortWork = 0;
    std::size_t lookupWork = 0;
    std::size_t propagationWork = 0;
    std::size_t groupWork = projectionWork;
    overflow =
        overflow || !checkedBasisProduct(stateNodes, wordsPerRow, groupWords) ||
        !checkedBasisProduct(maximumProjectedNodes,
                             sortFactor(maximumProjectedNodes), nodeSortWork) ||
        !checkedBasisProduct(maximumStateEdges, sortFactor(maximumStateEdges),
                             edgeSortWork) ||
        !checkedBasisProduct(maximumStateEdges, 2 * sortFactor(nodes.size()),
                             lookupWork) ||
        !checkedBasisProduct(maximumStateEdges, wordsPerRow + 3,
                             propagationWork) ||
        !checkedBasisSum(groupWork, nodeSortWork, groupWork) ||
        !checkedBasisSum(groupWork, edgeSortWork, groupWork) ||
        !checkedBasisSum(groupWork, lookupWork, groupWork) ||
        !checkedBasisSum(groupWork, propagationWork, groupWork) ||
        !checkedBasisSum(groupWork, stateNodes, groupWork) ||
        !checkedBasisSum(groupWork, demandIds.size(), groupWork);
    const bool exceedsWords =
        overflow || groupWords > options.maximumDemandBasisReachabilityWords ||
        usedWords > options.maximumDemandBasisReachabilityWords - groupWords;
    const bool exceedsWork =
        overflow || groupWork > options.maximumDemandBasisReductionWork ||
        usedWork > options.maximumDemandBasisReductionWork - groupWork;
    if (overflow || exceedsWords || exceedsWork) {
      result.truncated = true;
      continue;
    }

    std::map<std::pair<unsigned, SyncCoverNodeId>, std::size_t> nodeIndices;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      nodeIndices.emplace(std::make_pair(nodes[index].copy, nodes[index].node),
                          index);
    }
    const auto findNodeIndex =
        [&](unsigned copy, SyncCoverNodeId node) -> std::optional<std::size_t> {
      const auto position = nodeIndices.find({copy, node});
      return position == nodeIndices.end()
                 ? std::nullopt
                 : std::optional<std::size_t>(position->second);
    };
    const auto stateIndex = [](std::size_t node, bool completed) {
      return node * 2 + static_cast<std::size_t>(completed);
    };
    std::vector<std::vector<RecurrenceEdge>> outgoing(stateNodes);
    bool invalidOrder = false;
    for (const ProjectedRecurrenceEdge &edge : projected) {
      const std::optional<std::size_t> source =
          findNodeIndex(edge.source.copy, edge.source.node);
      const std::optional<std::size_t> target =
          findNodeIndex(edge.target.copy, edge.target.node);
      if (!source || !target || *source >= *target) {
        invalidOrder = true;
        break;
      }
      outgoing[stateIndex(*source, false)].push_back(
          {stateIndex(*target, true), edge.demand});
      outgoing[stateIndex(*source, true)].push_back(
          {stateIndex(*target, true), edge.demand});
    }
    for (const SyncCoverExpandedEdge &edge : arena->getEdges()) {
      const std::optional<SyncCoverNodeId> sourceNode =
          arena->getOperationForVirtualNode(edge.source);
      const std::optional<SyncCoverNodeId> targetNode =
          arena->getOperationForVirtualNode(edge.target);
      const std::optional<unsigned> sourceCopy =
          arena->getCopyForVirtualNode(edge.source);
      const std::optional<unsigned> targetCopy =
          arena->getCopyForVirtualNode(edge.target);
      if (!sourceNode || !targetNode || !sourceCopy || !targetCopy) {
        continue;
      }
      const SyncCoverNode &sourceDescription = graph.getNodes()[*sourceNode];
      const SyncCoverNode &targetDescription = graph.getNodes()[*targetNode];
      const bool localUnguarded = sourceDescription.scope == scope &&
                                  targetDescription.scope == scope &&
                                  sourceDescription.guard.literals.empty() &&
                                  targetDescription.guard.literals.empty();
      const bool edgeUnguarded =
          !edge.graphEdge ||
          (graph.getEdges()[*edge.graphEdge].sourceGuard.literals.empty() &&
           graph.getEdges()[*edge.graphEdge].targetGuard.literals.empty());
      if (!localUnguarded || !edgeUnguarded) {
        continue;
      }
      const std::optional<std::size_t> source =
          findNodeIndex(*sourceCopy, *sourceNode);
      const std::optional<std::size_t> target =
          findNodeIndex(*targetCopy, *targetNode);
      if (!source || !target || *source >= *target) {
        invalidOrder = true;
        break;
      }
      switch (edge.kind) {
      case SyncCoverEdgeKind::CertifiedCompletionFrontier:
        outgoing[stateIndex(*source, false)].push_back(
            {stateIndex(*target, false), std::nullopt});
        outgoing[stateIndex(*source, true)].push_back(
            {stateIndex(*target, true), std::nullopt});
        break;
      case SyncCoverEdgeKind::CompletionPreservingIssueOrder:
      case SyncCoverEdgeKind::NonCompletionPreservingIssueOrder:
        outgoing[stateIndex(*source, true)].push_back(
            {stateIndex(*target, true), std::nullopt});
        break;
      case SyncCoverEdgeKind::CompletionSupply:
        outgoing[stateIndex(*source, false)].push_back(
            {stateIndex(*target, true), std::nullopt});
        outgoing[stateIndex(*source, true)].push_back(
            {stateIndex(*target, true), std::nullopt});
        break;
      }
    }
    if (invalidOrder) {
      result.truncated = true;
      continue;
    }
    for (auto &edges : outgoing) {
      llvm::sort(edges,
                 [](const RecurrenceEdge &first, const RecurrenceEdge &second) {
                   return std::tie(first.target, first.demand) <
                          std::tie(second.target, second.demand);
                 });
    }
    std::vector<bool> required(graph.getDemands().size(), false);
    std::vector<llvm::BitVector> reachable(stateNodes,
                                           llvm::BitVector(stateNodes, false));
    for (std::size_t reverse = stateNodes; reverse > 0; --reverse) {
      const std::size_t source = reverse - 1;
      for (const RecurrenceEdge &edge : outgoing[source]) {
        if (reachable[source].test(edge.target)) {
          continue;
        }
        if (edge.demand) {
          required[*edge.demand] = true;
        }
        reachable[source].set(edge.target);
        reachable[source] |= reachable[edge.target];
      }
    }
    for (SyncCoverDemandId demandId : demandIds) {
      if (!required[demandId]) {
        retained[demandId] = false;
      }
    }
    usedWords += groupWords;
    usedWork += groupWork;
  }
  result.demands.erase(std::remove_if(result.demands.begin(),
                                      result.demands.end(),
                                      [&](SyncCoverDemandId demand) {
                                        return !retained[demand];
                                      }),
                       result.demands.end());
  return result;
}

std::vector<std::uint32_t> getIssueResources(const SyncCoverGraph &graph) {
  std::vector<std::uint32_t> resources;
  resources.reserve(graph.getNodes().size());
  for (const SyncCoverNode &node : graph.getNodes()) {
    resources.push_back(node.resource);
  }
  llvm::sort(resources);
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
  return resources;
}

std::vector<unsigned> getReservations(const CanonicalSyncProgram &program,
                                      const EventDomainKey &key) {
  const auto reservation = program.getEventReservations().find(key);
  return reservation == program.getEventReservations().end()
             ? std::vector<unsigned>{}
             : reservation->second;
}

bool canUseDistanceZeroEvent(const SyncCoverGraph &graph,
                             const SyncCoverDemand &demand) {
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const SyncCoverEdge edge = getDemandEdge(demand);
  return demand.distance == 0 && source.resource != target.resource &&
         syncCoverNodeCanProduceCompletion(graph, source.id, target.resource) &&
         syncCoverEndpointsCoExecute(graph, edge);
}

bool canUseTargetLocalPipeDrain(const CanonicalSyncProgram &program,
                                std::uint32_t resource) {
  return program.getGraph().supportsBlockingTargetedBarrier(resource);
}

bool canUseTargetPrefixEvent(const CanonicalSyncProgram &program,
                             const SyncCoverDemand &demand) {
  const SyncCoverGraph &graph = program.getGraph();
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const bool sameMacroOperation =
      demand.source < program.getNodeBindings().size() &&
      demand.target < program.getNodeBindings().size() &&
      program.getNodeBindings()[demand.source].operation ==
          program.getNodeBindings()[demand.target].operation;
  const bool validRecurrence =
      demand.distance == 0 || (demand.scope < graph.getScopes().size() &&
                               graph.getScopes()[demand.scope].isLoop &&
                               graph.getScopes()[demand.scope].timeline);
  const bool prefixCompletion = source.completionSignalCoversIssuedPrefix;
  const bool barrierCompletesPrefix =
      canUseTargetLocalPipeDrain(program, source.resource);
  return validRecurrence && !sameMacroOperation &&
         source.resource != target.resource &&
         (prefixCompletion || barrierCompletesPrefix);
}

bool targetPrefixNeedsBarrier(const CanonicalSyncProgram &program,
                              const SyncCoverDemand &demand) {
  const SyncCoverGraph &graph = program.getGraph();
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  return !source.completionSignalCoversIssuedPrefix;
}

bool canUseRecurrenceEvent(const SyncCoverGraph &graph,
                           const SyncCoverDemand &demand) {
  const bool invalid = demand.distance == 0 ||
                       demand.scope >= graph.getScopes().size() ||
                       !graph.getScopes()[demand.scope].isLoop ||
                       !graph.getScopes()[demand.scope].timeline ||
                       !demand.sourceGuard.literals.empty() ||
                       !demand.targetGuard.literals.empty();
  if (invalid) {
    return false;
  }
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  return source.resource != target.resource &&
         graph.scopeMustExecuteWithin(demand.scope, source.scope) &&
         graph.scopeMustExecuteWithin(demand.scope, target.scope) &&
         syncCoverNodeCanProduceCompletion(graph, source.id, target.resource);
}

bool isReleaseStyleRecurrence(const SyncCoverGraph &graph,
                              const SyncCoverDemand &demand) {
  return graph.getNodes()[demand.target].order <
         graph.getNodes()[demand.source].order;
}

bool canUsePreciseEvent(const SyncCoverGraph &graph,
                        const SyncCoverDemand &demand) {
  return canUseDistanceZeroEvent(graph, demand) ||
         canUseRecurrenceEvent(graph, demand);
}

bool canUseSourceLocalCompletionEvent(const CanonicalSyncProgram &program,
                                      const SyncCoverDemand &demand) {
  const SyncCoverGraph &graph = program.getGraph();
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const bool sameMacroOperation =
      demand.source < program.getNodeBindings().size() &&
      demand.target < program.getNodeBindings().size() &&
      program.getNodeBindings()[demand.source].operation ==
          program.getNodeBindings()[demand.target].operation;
  const bool sourceCanSignal =
      syncCoverNodeCanProduceCompletion(graph, source.id, target.resource);
  const bool canDrainExactSource =
      graph.supportsBlockingTargetedBarrier(source.resource);
  if (sameMacroOperation || source.resource == target.resource ||
      (!sourceCanSignal && !canDrainExactSource) ||
      source.physicalExit >= graph.getNodes().size() ||
      target.physicalAnchor >= graph.getNodes().size()) {
    return false;
  }
  if (demand.distance != 0) {
    return demand.scope < graph.getScopes().size() &&
           graph.getScopes()[demand.scope].isLoop &&
           graph.getScopes()[demand.scope].timeline &&
           graph.scopeContains(demand.scope, source.scope) &&
           graph.scopeContains(demand.scope, target.scope);
  }
  const std::optional<SyncCoverTimelinePosition> fencePosition =
      resolveSyncCoverAnchor(
          graph, {SyncCoverAnchorKind::AfterNode, source.physicalExit, 0, 0});
  const std::optional<SyncCoverTimelinePosition> targetPosition =
      resolveSyncCoverAnchor(graph, {SyncCoverAnchorKind::BeforeNode,
                                     target.physicalAnchor, 0, 0});
  return fencePosition && targetPosition && *fencePosition < *targetPosition &&
         graph.getNodes()[source.physicalExit].order <
             graph.getNodes()[target.physicalAnchor].order;
}

CanonicalSyncMechanismDescriptor
makeSourceLocalCompletionEvent(const SyncCoverGraph &graph,
                               CanonicalSyncEventDomainId domain,
                               ArrayRef<SyncCoverDemandId> demandIds) {
  const SyncCoverDemand &firstDemand = graph.getDemands()[demandIds.front()];
  const SyncCoverNode &source = graph.getNodes()[firstDemand.source];
  const SyncCoverNode &target = graph.getNodes()[firstDemand.target];
  const bool needsBarrier =
      !syncCoverNodeCanProduceCompletion(graph, source.id, target.resource);
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Event;
  descriptor.eventUses.push_back({domain, 1, std::nullopt});
  if (needsBarrier) {
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::Barrier,
         source.resource,
         {SyncCoverAnchorKind::AfterNode, source.physicalExit, 0, 0},
         std::nullopt,
         0,
         {source.resource},
         CanonicalSyncBarrierKind::Targeted});
  }
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventSet,
       source.resource,
       {SyncCoverAnchorKind::AfterNode, source.physicalExit, 0, 0},
       0,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventWait,
       target.resource,
       {SyncCoverAnchorKind::AfterNode, source.physicalExit, 0, 0},
       0,
       0,
       {}});
  for (SyncCoverDemandId demandId : demandIds) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    CanonicalSyncSupplyBinding binding;
    binding.edge = getDemandEdge(demand);
    binding.eventUse = 0;
    binding.proof = CanonicalSyncSupplyProof::SourceLocalCompletionAction;
    binding.attestedDemand = demandId;
    if (demand.distance == 0) {
      binding.applicability = SyncCoverSupplyApplicability::DistanceZeroOnly;
    } else {
      binding.allowedDemands = {demandId};
    }
    descriptor.supplies.push_back(std::move(binding));
  }
  return descriptor;
}

LogicalResult addSourceLocalCompletionEvents(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    ArrayRef<SyncCoverDemandId> demandIds,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds) {
  const SyncCoverGraph &graph = program.getGraph();
  // The set/wait recipe is identified by its directed event domain, complete
  // physical source exit, and whether an exact source drain is required.
  // Multiple semantic macro phases sharing that complete recipe are
  // independently revalidated below, but must not create duplicate actions or
  // event lifetimes.
  using GroupKey =
      std::tuple<CanonicalSyncEventDomainId, SyncCoverNodeId, bool>;
  std::map<GroupKey, std::vector<SyncCoverDemandId>> groups;
  for (SyncCoverDemandId demandId : demandIds) {
    if (demandId >= graph.getDemands().size()) {
      return program.getFunction().emitError(
          "canonical sync source-local event names an invalid demand");
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (!canUseSourceLocalCompletionEvent(program, demand)) {
      continue;
    }
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    const auto domain = domainIds.find({source.resource, target.resource});
    if (domain == domainIds.end()) {
      continue;
    }
    const bool needsBarrier =
        !syncCoverNodeCanProduceCompletion(graph, source.id, target.resource);
    groups[{domain->second, source.physicalExit, needsBarrier}].push_back(
        demandId);
  }
  for (const auto &[key, groupedDemands] : groups) {
    const CanonicalSyncProblemResult added =
        problem.internMechanism(makeSourceLocalCompletionEvent(
            graph, std::get<0>(key), groupedDemands));
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      return program.getFunction().emitError(
          "canonical sync mechanism limit prevents source-local events");
    }
    if (!added) {
      return program.getFunction().emitError(
          "cannot add canonical sync source-local completion event");
    }
  }
  return success();
}

CanonicalSyncMechanismDescriptor
makeDirectEvent(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                CanonicalSyncEventDomainId domain) {
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Event;
  descriptor.eventUses.push_back({domain, 1, std::nullopt});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventSet,
       source.resource,
       {SyncCoverAnchorKind::AfterNode, source.id, 0, 0},
       0,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventWait,
       target.resource,
       {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
       0,
       0,
       {}});
  CanonicalSyncSupplyBinding binding;
  binding.edge = getDemandEdge(demand);
  binding.eventUse = 0;
  descriptor.supplies.push_back(std::move(binding));
  return descriptor;
}

class CompletionFrontierIndex {
public:
  explicit CompletionFrontierIndex(const SyncCoverGraph &graph)
      : graph_(graph), successors_(graph.getNodes().size()),
        visited_(graph.getNodes().size(), 0) {
    for (const SyncCoverNode &frontier : graph.getNodes()) {
      for (SyncCoverNodeId dominated : frontier.completionDominatedSources) {
        successors_[dominated].push_back(frontier.id);
      }
    }
    ready_.reserve(graph.getNodes().size());
  }

  std::optional<SyncCoverNodeId> findLatest(const SyncCoverDemand &demand) {
    if (demand.distance != 0) {
      return std::nullopt;
    }
    ++epoch_;
    if (epoch_ == 0) {
      std::fill(visited_.begin(), visited_.end(), 0);
      ++epoch_;
    }
    ready_.clear();
    ready_.push_back(demand.source);

    const SyncCoverNode &source = graph_.getNodes()[demand.source];
    const SyncCoverNode &target = graph_.getNodes()[demand.target];
    std::optional<SyncCoverNodeId> latest;
    while (!ready_.empty()) {
      const SyncCoverNodeId current = ready_.back();
      ready_.pop_back();
      if (visited_[current] == epoch_) {
        continue;
      }
      visited_[current] = epoch_;
      for (SyncCoverNodeId candidateId : successors_[current]) {
        const SyncCoverNode &candidate = graph_.getNodes()[candidateId];
        if (candidate.order >= target.order) {
          continue;
        }
        ready_.push_back(candidateId);
        const std::optional<SyncCoverScopeId> scope =
            graph_.getLowestCommonScope(candidate.scope, target.scope);
        const std::optional<SyncCoverTimelinePosition> setPosition =
            resolveSyncCoverAnchor(
                graph_, {SyncCoverAnchorKind::AfterNode, candidate.id, 0, 0});
        const std::optional<SyncCoverTimelinePosition> waitPosition =
            resolveSyncCoverAnchor(
                graph_, {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0});
        if (!scope || !setPosition || !waitPosition ||
            *setPosition >= *waitPosition) {
          continue;
        }
        SyncCoverEdge supply{candidate.id,
                             target.id,
                             SyncCoverEdgeKind::CompletionSupply,
                             *scope,
                             0,
                             candidate.guard,
                             demand.targetGuard};
        const bool valid = candidate.resource == source.resource &&
                           candidate.physicalAnchor != target.physicalAnchor &&
                           syncCoverNodeCanProduceCompletion(
                               graph_, candidate.id, target.resource) &&
                           syncCoverEndpointsCoExecute(graph_, supply);
        if (valid &&
            (!latest || graph_.getNodes()[*latest].order < candidate.order)) {
          latest = candidate.id;
        }
      }
    }
    return latest;
  }

private:
  const SyncCoverGraph &graph_;
  std::vector<std::vector<SyncCoverNodeId>> successors_;
  std::vector<std::uint32_t> visited_;
  std::vector<SyncCoverNodeId> ready_;
  std::uint32_t epoch_ = 0;
};

CanonicalSyncMechanismDescriptor makeCompletionFrontierEvent(
    const SyncCoverGraph &graph, const SyncCoverDemand &demand,
    SyncCoverNodeId frontier, CanonicalSyncEventDomainId domain) {
  const SyncCoverNode &completion = graph.getNodes()[frontier];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Event;
  descriptor.eventUses.push_back({domain, 1, std::nullopt});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventSet,
       completion.resource,
       {SyncCoverAnchorKind::AfterNode, completion.id, 0, 0},
       0,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventWait,
       target.resource,
       {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
       0,
       0,
       {}});
  CanonicalSyncSupplyBinding binding;
  binding.edge = {completion.id,
                  target.id,
                  SyncCoverEdgeKind::CompletionSupply,
                  *graph.getLowestCommonScope(completion.scope, target.scope),
                  0,
                  completion.guard,
                  demand.targetGuard};
  binding.eventUse = 0;
  binding.proof = CanonicalSyncSupplyProof::CompletionFrontierAction;
  descriptor.supplies.push_back(std::move(binding));
  return descriptor;
}

bool verifyCompletionFrontierEvent(
    const SyncCoverGraph &graph, const SyncCoverDemand &demand,
    SyncCoverNodeId frontier, CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  const SyncCoverNode &completion = graph.getNodes()[frontier];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const std::optional<SyncCoverTimelinePosition> setPosition =
      resolveSyncCoverAnchor(
          graph, {SyncCoverAnchorKind::AfterNode, completion.id, 0, 0});
  const std::optional<SyncCoverTimelinePosition> waitPosition =
      resolveSyncCoverAnchor(
          graph, {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0});
  if (!graph.completionDominates(frontier, demand.source) ||
      completion.physicalAnchor == target.physicalAnchor || !setPosition ||
      !waitPosition || *setPosition >= *waitPosition ||
      descriptor.kind != CanonicalSyncMechanismKind::Event ||
      descriptor.eventUses.size() != 1 || descriptor.actions.size() != 2 ||
      descriptor.supplies.size() != 1) {
    return false;
  }
  const CanonicalSyncEventUse &use = descriptor.eventUses.front();
  const CanonicalSyncAction &set = descriptor.actions[0];
  const CanonicalSyncAction &wait = descriptor.actions[1];
  const CanonicalSyncSupplyBinding &binding = descriptor.supplies.front();
  const SyncCoverEdge expected{
      completion.id,
      target.id,
      SyncCoverEdgeKind::CompletionSupply,
      *graph.getLowestCommonScope(completion.scope, target.scope),
      0,
      completion.guard,
      demand.targetGuard};
  return use.domain == domain && use.width == 1 && !use.recurrenceScope &&
         !use.lifetimeScope && set.kind == CanonicalSyncActionKind::EventSet &&
         set.resource == completion.resource &&
         set.anchor.kind == SyncCoverAnchorKind::AfterNode &&
         set.anchor.node == completion.id && set.eventUse == 0 &&
         set.eventLane == 0 &&
         wait.kind == CanonicalSyncActionKind::EventWait &&
         wait.resource == target.resource &&
         wait.anchor.kind == SyncCoverAnchorKind::BeforeNode &&
         wait.anchor.node == target.id && wait.eventUse == 0 &&
         wait.eventLane == 0 && binding.edge.source == expected.source &&
         binding.edge.target == expected.target &&
         binding.edge.scope == expected.scope && binding.edge.distance == 0 &&
         binding.edge.sourceGuard.literals == expected.sourceGuard.literals &&
         binding.edge.targetGuard.literals == expected.targetGuard.literals &&
         binding.eventUse == 0 && !binding.barrierAction &&
         !binding.produceAction && !binding.consumeAction &&
         binding.proof == CanonicalSyncSupplyProof::CompletionFrontierAction &&
         binding.allowedDemands.empty();
}

/// Build a completeness mechanism for exact demand rows that have no cover in
/// the normal catalog. A supported source-pipe drain completes the issued
/// source prefix before the subsequent target begins. Completion-ordered
/// resources instead use a set and wait together at the target. Either recipe
/// executes within every target occurrence, so distance remains an
/// independently attested per-binding fact rather than carried protocol state.
CanonicalSyncMechanismDescriptor
makeTargetPrefixEvent(const CanonicalSyncProgram &program,
                      ArrayRef<SyncCoverDemandId> demandIds,
                      CanonicalSyncEventDomainId domain) {
  const SyncCoverGraph &graph = program.getGraph();
  const SyncCoverDemand &demand = graph.getDemands()[demandIds.front()];
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const bool needsBarrier = targetPrefixNeedsBarrier(program, demand);
  CanonicalSyncMechanismDescriptor descriptor;
  if (needsBarrier) {
    descriptor.kind = CanonicalSyncMechanismKind::Barrier;
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::Barrier,
         source.resource,
         {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
         std::nullopt,
         0,
         {source.resource},
         CanonicalSyncBarrierKind::Targeted});
  } else {
    descriptor.kind = CanonicalSyncMechanismKind::Event;
    descriptor.eventUses.push_back({domain, 1, std::nullopt});
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::EventSet,
         source.resource,
         {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
         0,
         0,
         {}});
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::EventWait,
         target.resource,
         {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
         0,
         0,
         {}});
  }
  for (SyncCoverDemandId demandId : demandIds) {
    CanonicalSyncSupplyBinding binding;
    binding.edge = getDemandEdge(graph.getDemands()[demandId]);
    if (needsBarrier) {
      binding.barrierAction = 0;
      binding.proof = CanonicalSyncSupplyProof::TargetLocalPipeDrainAction;
    } else {
      binding.eventUse = 0;
      binding.proof = CanonicalSyncSupplyProof::TargetLocalFenceAction;
    }
    binding.attestedDemand = demandId;
    if (binding.edge.distance == 0) {
      binding.applicability = SyncCoverSupplyApplicability::DistanceZeroOnly;
    } else {
      binding.allowedDemands = {demandId};
    }
    descriptor.supplies.push_back(std::move(binding));
  }
  return descriptor;
}

bool verifyTargetPrefixEvent(
    const CanonicalSyncProgram &program, ArrayRef<SyncCoverDemandId> demandIds,
    CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  const SyncCoverGraph &graph = program.getGraph();
  if (demandIds.empty()) {
    return false;
  }
  const SyncCoverDemand &demand = graph.getDemands()[demandIds.front()];
  const bool needsBarrier = targetPrefixNeedsBarrier(program, demand);
  const bool correctShape =
      needsBarrier
          ? descriptor.kind == CanonicalSyncMechanismKind::Barrier &&
                descriptor.eventUses.empty() && descriptor.actions.size() == 1
          : descriptor.kind == CanonicalSyncMechanismKind::Event &&
                descriptor.eventUses.size() == 1 &&
                descriptor.actions.size() == 2;
  if (!canUseTargetPrefixEvent(program, demand) || !correctShape ||
      descriptor.supplies.size() != demandIds.size()) {
    return false;
  }
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const auto eventActionMatches = [&](const CanonicalSyncAction &action,
                                      CanonicalSyncActionKind kind,
                                      std::uint32_t resource) {
    return action.kind == kind && action.resource == resource &&
           action.anchor.kind == SyncCoverAnchorKind::BeforeNode &&
           action.anchor.node == target.id && action.eventUse == 0 &&
           action.eventLane == 0 && action.drainedResources.empty() &&
           action.guard == CanonicalSyncActionGuardKind::None &&
           !action.guardScope &&
           action.eventLaneKind == CanonicalSyncEventLaneKind::Static &&
           !action.eventLaneScope;
  };
  const auto barrierActionMatches = [&](const CanonicalSyncAction &action) {
    return action.kind == CanonicalSyncActionKind::Barrier &&
           action.resource == source.resource &&
           action.anchor.kind == SyncCoverAnchorKind::BeforeNode &&
           action.anchor.node == target.id && !action.eventUse &&
           action.eventLane == 0 &&
           action.drainedResources ==
               std::vector<std::uint32_t>{source.resource} &&
           action.barrierKind == CanonicalSyncBarrierKind::Targeted &&
           action.guard == CanonicalSyncActionGuardKind::None &&
           !action.guardScope &&
           action.eventLaneKind == CanonicalSyncEventLaneKind::Static &&
           !action.eventLaneScope;
  };
  const bool allDemandsEligible =
      llvm::all_of(demandIds, [&](SyncCoverDemandId demandId) {
        const SyncCoverDemand &candidate = graph.getDemands()[demandId];
        return canUseTargetPrefixEvent(program, candidate) &&
               graph.getNodes()[candidate.target].physicalAnchor ==
                   target.physicalAnchor &&
               targetPrefixNeedsBarrier(program, candidate) == needsBarrier &&
               graph.getNodes()[candidate.source].resource == source.resource &&
               (needsBarrier ||
                graph.getNodes()[candidate.target].resource == target.resource);
      });
  std::vector<SyncCoverDemandId> attestedDemands;
  const bool suppliesMatch = llvm::all_of(
      descriptor.supplies, [&](const CanonicalSyncSupplyBinding &binding) {
        if (!binding.attestedDemand ||
            !llvm::is_contained(demandIds, *binding.attestedDemand)) {
          return false;
        }
        const SyncCoverDemandId demandId = *binding.attestedDemand;
        attestedDemands.push_back(demandId);
        const SyncCoverDemand &candidate = graph.getDemands()[demandId];
        return binding.edge.source == candidate.source &&
               binding.edge.target == candidate.target &&
               binding.edge.scope == candidate.scope &&
               binding.edge.distance == candidate.distance &&
               binding.edge.sourceGuard.literals ==
                   candidate.sourceGuard.literals &&
               binding.edge.targetGuard.literals ==
                   candidate.targetGuard.literals &&
               (needsBarrier
                    ? !binding.eventUse && binding.barrierAction == 0
                    : binding.eventUse == 0 && !binding.barrierAction) &&
               !binding.produceAction && !binding.consumeAction &&
               binding.proof ==
                   (needsBarrier
                        ? CanonicalSyncSupplyProof::TargetLocalPipeDrainAction
                        : CanonicalSyncSupplyProof::TargetLocalFenceAction) &&
               binding.completionExport ==
                   CanonicalSyncSupplyExport::LocalTarget &&
               binding.allowedDemands ==
                   (candidate.distance == 0
                        ? std::vector<SyncCoverDemandId>{}
                        : std::vector<SyncCoverDemandId>{demandId}) &&
               binding.applicability ==
                   (candidate.distance == 0
                        ? SyncCoverSupplyApplicability::DistanceZeroOnly
                        : SyncCoverSupplyApplicability::AllDemands);
      });
  std::vector<SyncCoverDemandId> expectedDemands(demandIds.begin(),
                                                 demandIds.end());
  llvm::sort(attestedDemands);
  llvm::sort(expectedDemands);
  const bool oneToOneAttestation = attestedDemands == expectedDemands;
  if (!allDemandsEligible || !suppliesMatch || !oneToOneAttestation) {
    return false;
  }
  if (needsBarrier) {
    return barrierActionMatches(descriptor.actions.front());
  }
  const CanonicalSyncEventUse &use = descriptor.eventUses.front();
  return use.domain == domain && use.width == 1 && !use.recurrenceScope &&
         !use.lifetimeScope &&
         eventActionMatches(descriptor.actions[0],
                            CanonicalSyncActionKind::EventSet,
                            source.resource) &&
         eventActionMatches(descriptor.actions[1],
                            CanonicalSyncActionKind::EventWait,
                            target.resource);
}

CanonicalSyncMechanismDescriptor
makeTargetPipeDrainCut(const CanonicalSyncProgram &program,
                       ArrayRef<SyncCoverDemandId> demandIds) {
  const SyncCoverGraph &graph = program.getGraph();
  const SyncCoverDemand &firstDemand = graph.getDemands()[demandIds.front()];
  const SyncCoverNode &firstSource = graph.getNodes()[firstDemand.source];
  const SyncCoverNode &firstTarget = graph.getNodes()[firstDemand.target];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Barrier;
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::Barrier,
       firstSource.resource,
       {SyncCoverAnchorKind::BeforeNode, firstTarget.id, 0, 0},
       std::nullopt,
       0,
       {firstSource.resource},
       CanonicalSyncBarrierKind::Targeted});

  std::vector<SyncCoverNodeId> targets;
  for (SyncCoverDemandId demandId : demandIds) {
    targets.push_back(graph.getDemands()[demandId].target);
  }
  llvm::sort(targets);
  targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

  std::set<std::pair<SyncCoverNodeId, SyncCoverNodeId>> cutEdges;
  const auto prefix = graph.getBlockingTargetedBarrierPrefixes().find(
      {firstSource.resource, firstTarget.physicalAnchor});
  for (SyncCoverNodeId targetId : targets) {
    const SyncCoverNode &target = graph.getNodes()[targetId];
    if (prefix == graph.getBlockingTargetedBarrierPrefixes().end()) {
      continue;
    }
    for (SyncCoverNodeId sourceId : prefix->second) {
      const SyncCoverNode &source = graph.getNodes()[sourceId];
      const std::optional<SyncCoverScopeId> scope =
          graph.getLowestCommonScope(source.scope, target.scope);
      if (!scope || source.order >= target.order ||
          !syncCoverGuardsCompatible(source.guard, target.guard)) {
        continue;
      }
      SyncCoverEdge edge{
          source.id,   target.id, SyncCoverEdgeKind::CompletionSupply,
          *scope,      0,         target.guard,
          target.guard};
      if (graph.canonicalizeCompletionEdge(edge) != SyncCoverGraphError::None) {
        continue;
      }
      CanonicalSyncSupplyBinding binding;
      binding.edge = std::move(edge);
      binding.barrierAction = 0;
      binding.proof = CanonicalSyncSupplyProof::DominatingTargetedDrainCut;
      binding.applicability = SyncCoverSupplyApplicability::DistanceZeroOnly;
      descriptor.supplies.push_back(std::move(binding));
      cutEdges.insert({source.id, target.id});
    }
  }

  for (SyncCoverDemandId demandId : demandIds) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const bool suppliedByCut =
        demand.distance == 0 &&
        cutEdges.find({demand.source, demand.target}) != cutEdges.end();
    if (suppliedByCut) {
      continue;
    }
    CanonicalSyncSupplyBinding binding;
    binding.edge = getDemandEdge(demand);
    binding.barrierAction = 0;
    binding.proof = CanonicalSyncSupplyProof::TargetLocalPipeDrainAction;
    binding.attestedDemand = demandId;
    if (demand.distance == 0) {
      binding.applicability = SyncCoverSupplyApplicability::DistanceZeroOnly;
    } else {
      binding.allowedDemands = {demandId};
    }
    descriptor.supplies.push_back(std::move(binding));
  }
  return descriptor;
}

bool verifyTargetPipeDrainCut(
    const CanonicalSyncProgram &program, ArrayRef<SyncCoverDemandId> demandIds,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (demandIds.empty()) {
    return false;
  }
  const SyncCoverGraph &graph = program.getGraph();
  const SyncCoverDemand &firstDemand = graph.getDemands()[demandIds.front()];
  const SyncCoverNode &firstSource = graph.getNodes()[firstDemand.source];
  const SyncCoverNode &firstTarget = graph.getNodes()[firstDemand.target];
  const bool eligible =
      llvm::all_of(demandIds, [&](SyncCoverDemandId demandId) {
        if (demandId >= graph.getDemands().size()) {
          return false;
        }
        const SyncCoverDemand &demand = graph.getDemands()[demandId];
        const SyncCoverNode &source = graph.getNodes()[demand.source];
        const SyncCoverNode &target = graph.getNodes()[demand.target];
        return canUseTargetPrefixEvent(program, demand) &&
               targetPrefixNeedsBarrier(program, demand) &&
               source.resource == firstSource.resource &&
               target.physicalAnchor == firstTarget.physicalAnchor;
      });
  if (!eligible) {
    return false;
  }
  const CanonicalSyncMechanismDescriptor expected =
      makeTargetPipeDrainCut(program, demandIds);
  const auto sameEdge = [](const SyncCoverEdge &left,
                           const SyncCoverEdge &right) {
    return left.source == right.source && left.target == right.target &&
           left.kind == right.kind && left.scope == right.scope &&
           left.distance == right.distance &&
           left.sourceGuard.literals == right.sourceGuard.literals &&
           left.targetGuard.literals == right.targetGuard.literals;
  };
  const auto sameBinding = [&](const CanonicalSyncSupplyBinding &left,
                               const CanonicalSyncSupplyBinding &right) {
    return sameEdge(left.edge, right.edge) && left.eventUse == right.eventUse &&
           left.barrierAction == right.barrierAction &&
           left.produceAction == right.produceAction &&
           left.consumeAction == right.consumeAction &&
           left.proof == right.proof &&
           left.completionExport == right.completionExport &&
           left.allowedDemands == right.allowedDemands &&
           left.attestedDemand == right.attestedDemand &&
           left.applicability == right.applicability;
  };
  const auto sameAction = [](const CanonicalSyncAction &left,
                             const CanonicalSyncAction &right) {
    return left.kind == right.kind && left.resource == right.resource &&
           left.anchor.kind == right.anchor.kind &&
           left.anchor.node == right.anchor.node &&
           left.anchor.scope == right.anchor.scope &&
           left.anchor.position == right.anchor.position &&
           left.eventUse == right.eventUse &&
           left.eventLane == right.eventLane &&
           left.drainedResources == right.drainedResources &&
           left.barrierKind == right.barrierKind && left.guard == right.guard &&
           left.guardScope == right.guardScope &&
           left.eventLaneKind == right.eventLaneKind &&
           left.eventLaneScope == right.eventLaneScope;
  };
  return descriptor.kind == expected.kind && descriptor.eventUses.empty() &&
         expected.eventUses.empty() &&
         descriptor.actions.size() == expected.actions.size() &&
         descriptor.supplies.size() == expected.supplies.size() &&
         std::equal(descriptor.actions.begin(), descriptor.actions.end(),
                    expected.actions.begin(), sameAction) &&
         std::equal(descriptor.supplies.begin(), descriptor.supplies.end(),
                    expected.supplies.begin(), sameBinding);
}

CanonicalSyncMechanismDescriptor
makeRecurrenceEvent(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                    CanonicalSyncEventDomainId domain) {
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  if (!isReleaseStyleRecurrence(graph, demand)) {
    CanonicalSyncMechanismDescriptor descriptor;
    descriptor.kind = CanonicalSyncMechanismKind::Event;
    descriptor.eventUses.push_back({domain, 1, std::nullopt});
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::EventSet,
         source.resource,
         {SyncCoverAnchorKind::AfterNode, source.id, 0, 0},
         0,
         0,
         {}});
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::EventWait,
         target.resource,
         {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
         0,
         0,
         {}});
    SyncCoverEdge edge = getDemandEdge(demand);
    edge.distance = 0;
    edge.scope = *graph.getLowestCommonScope(source.scope, target.scope);
    CanonicalSyncSupplyBinding binding;
    binding.edge = std::move(edge);
    binding.eventUse = 0;
    descriptor.supplies.push_back(std::move(binding));
    return descriptor;
  }

  const std::size_t width = demand.distance;
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Protocol;
  descriptor.eventUses.push_back({domain, width, demand.scope});
  for (std::size_t lane = 0; lane < width; ++lane) {
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::EventSet,
         source.resource,
         {SyncCoverAnchorKind::ScopeEntry, 0, demand.scope},
         0,
         lane,
         {}});
  }
  const std::size_t consumeAction = descriptor.actions.size();
  CanonicalSyncAction bodyWait{CanonicalSyncActionKind::EventWait,
                               target.resource,
                               {SyncCoverAnchorKind::BeforeNode, target.id, 0},
                               0,
                               0,
                               {}};
  if (width > 1) {
    bodyWait.eventLaneKind = CanonicalSyncEventLaneKind::LoopIterationModulo;
    bodyWait.eventLaneScope = demand.scope;
  }
  descriptor.actions.push_back(std::move(bodyWait));
  const std::size_t produceAction = descriptor.actions.size();
  CanonicalSyncAction bodySet{CanonicalSyncActionKind::EventSet,
                              source.resource,
                              {SyncCoverAnchorKind::AfterNode, source.id, 0},
                              0,
                              0,
                              {}};
  if (width > 1) {
    bodySet.eventLaneKind = CanonicalSyncEventLaneKind::LoopIterationModulo;
    bodySet.eventLaneScope = demand.scope;
  }
  descriptor.actions.push_back(std::move(bodySet));
  for (std::size_t lane = 0; lane < width; ++lane) {
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::EventWait,
         target.resource,
         {SyncCoverAnchorKind::ScopeExit, 0, demand.scope},
         0,
         lane,
         {}});
  }
  CanonicalSyncSupplyBinding binding;
  binding.edge = getDemandEdge(demand);
  binding.eventUse = 0;
  binding.produceAction = produceAction;
  binding.consumeAction = consumeAction;
  binding.proof = CanonicalSyncSupplyProof::VerifiedProtocol;
  binding.completionExport = CanonicalSyncSupplyExport::ScopeExitAfterDrain;
  descriptor.supplies.push_back(std::move(binding));
  return descriptor;
}

bool verifyRecurrenceEvent(const SyncCoverGraph &graph,
                           const SyncCoverDemand &demand,
                           CanonicalSyncEventDomainId domain,
                           const CanonicalSyncMechanismDescriptor &descriptor) {
  const bool releaseStyle = isReleaseStyleRecurrence(graph, demand);
  const bool invalid =
      !canUseRecurrenceEvent(graph, demand) ||
      descriptor.kind != (releaseStyle ? CanonicalSyncMechanismKind::Protocol
                                       : CanonicalSyncMechanismKind::Event) ||
      descriptor.eventUses.size() != 1 || descriptor.supplies.size() != 1;
  if (invalid) {
    return false;
  }
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const CanonicalSyncEventUse &use = descriptor.eventUses.front();
  const CanonicalSyncSupplyBinding &binding = descriptor.supplies.front();
  const auto actionMatches =
      [](const CanonicalSyncAction &action, CanonicalSyncActionKind kind,
         std::uint32_t resource, SyncCoverAnchorKind anchorKind,
         SyncCoverNodeId node, SyncCoverScopeId scope, std::size_t lane,
         CanonicalSyncEventLaneKind laneKind,
         std::optional<SyncCoverScopeId> laneScope) {
        return action.kind == kind && action.resource == resource &&
               action.anchor.kind == anchorKind && action.anchor.node == node &&
               action.anchor.scope == scope && action.eventUse == 0 &&
               action.eventLane == lane && action.drainedResources.empty() &&
               action.guard == CanonicalSyncActionGuardKind::None &&
               !action.guardScope && action.eventLaneKind == laneKind &&
               action.eventLaneScope == laneScope;
      };
  if (!releaseStyle) {
    const std::optional<SyncCoverScopeId> common =
        graph.getLowestCommonScope(source.scope, target.scope);
    const std::size_t setAction = 0;
    const std::size_t waitAction = setAction + 1;
    return common && descriptor.actions.size() == 2 && use.domain == domain &&
           use.width == 1 && !use.recurrenceScope && !use.lifetimeScope &&
           binding.edge.source == demand.source &&
           binding.edge.target == demand.target &&
           binding.edge.scope == *common && binding.edge.distance == 0 &&
           binding.eventUse == 0 && !binding.barrierAction &&
           !binding.produceAction && !binding.consumeAction &&
           binding.proof == CanonicalSyncSupplyProof::DirectAction &&
           binding.completionExport == CanonicalSyncSupplyExport::LocalTarget &&
           actionMatches(descriptor.actions[setAction],
                         CanonicalSyncActionKind::EventSet, source.resource,
                         SyncCoverAnchorKind::AfterNode, source.id, 0, 0,
                         CanonicalSyncEventLaneKind::Static, std::nullopt) &&
           actionMatches(descriptor.actions[waitAction],
                         CanonicalSyncActionKind::EventWait, target.resource,
                         SyncCoverAnchorKind::BeforeNode, target.id, 0, 0,
                         CanonicalSyncEventLaneKind::Static, std::nullopt);
  }

  const std::size_t width = demand.distance;
  const std::size_t consumeAction = width;
  const std::size_t produceAction = width + 1;
  const std::size_t drainBegin = produceAction + 1;
  const bool correctUse = use.domain == domain && use.width == width &&
                          use.recurrenceScope == demand.scope &&
                          !use.lifetimeScope;
  const bool correctSupply =
      binding.edge.source == demand.source &&
      binding.edge.target == demand.target &&
      binding.edge.scope == demand.scope &&
      binding.edge.distance == demand.distance && binding.eventUse == 0 &&
      !binding.barrierAction && binding.produceAction == produceAction &&
      binding.consumeAction == consumeAction &&
      binding.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
      binding.completionExport ==
          CanonicalSyncSupplyExport::ScopeExitAfterDrain;
  if (!correctUse || !correctSupply ||
      descriptor.actions.size() != width * 2 + 2) {
    return false;
  }
  for (std::size_t lane = 0; lane < width; ++lane) {
    if (!actionMatches(descriptor.actions[lane],
                       CanonicalSyncActionKind::EventSet, source.resource,
                       SyncCoverAnchorKind::ScopeEntry, 0, demand.scope, lane,
                       CanonicalSyncEventLaneKind::Static, std::nullopt) ||
        !actionMatches(descriptor.actions[drainBegin + lane],
                       CanonicalSyncActionKind::EventWait, target.resource,
                       SyncCoverAnchorKind::ScopeExit, 0, demand.scope, lane,
                       CanonicalSyncEventLaneKind::Static, std::nullopt)) {
      return false;
    }
  }
  const CanonicalSyncEventLaneKind bodyLaneKind =
      width > 1 ? CanonicalSyncEventLaneKind::LoopIterationModulo
                : CanonicalSyncEventLaneKind::Static;
  const std::optional<SyncCoverScopeId> bodyLaneScope =
      width > 1 ? std::optional<SyncCoverScopeId>(demand.scope) : std::nullopt;
  return actionMatches(descriptor.actions[consumeAction],
                       CanonicalSyncActionKind::EventWait, target.resource,
                       SyncCoverAnchorKind::BeforeNode, target.id, 0, 0,
                       bodyLaneKind, bodyLaneScope) &&
         actionMatches(descriptor.actions[produceAction],
                       CanonicalSyncActionKind::EventSet, source.resource,
                       SyncCoverAnchorKind::AfterNode, source.id, 0, 0,
                       bodyLaneKind, bodyLaneScope);
}

CanonicalSyncMechanismDescriptor
makeLoopBoundarySourcePrefixProtocol(const SyncCoverGraph &graph,
                                     CanonicalSyncEventDomainId domain,
                                     SyncCoverScopeId scope, unsigned distance,
                                     ArrayRef<SyncCoverDemandId> demandIds) {
  const SyncCoverDemand &firstDemand = graph.getDemands()[demandIds.front()];
  const SyncCoverNode &source = graph.getNodes()[firstDemand.source];
  const SyncCoverNode &target = graph.getNodes()[firstDemand.target];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Protocol;
  descriptor.eventUses.push_back({domain, distance, scope});
  for (std::size_t lane = 0; lane < distance; ++lane) {
    descriptor.actions.push_back({CanonicalSyncActionKind::EventSet,
                                  source.resource,
                                  {SyncCoverAnchorKind::ScopeEntry, 0, scope},
                                  0,
                                  lane,
                                  {}});
  }
  const std::size_t consumeAction = descriptor.actions.size();
  CanonicalSyncAction bodyWait{CanonicalSyncActionKind::EventWait,
                               target.resource,
                               {SyncCoverAnchorKind::LoopBodyEntry, 0, scope},
                               0,
                               0,
                               {}};
  if (distance > 1) {
    bodyWait.eventLaneKind = CanonicalSyncEventLaneKind::LoopIterationModulo;
    bodyWait.eventLaneScope = scope;
  }
  descriptor.actions.push_back(std::move(bodyWait));
  descriptor.actions.push_back({CanonicalSyncActionKind::Barrier,
                                source.resource,
                                {SyncCoverAnchorKind::LoopBodyExit, 0, scope},
                                std::nullopt,
                                0,
                                {source.resource},
                                CanonicalSyncBarrierKind::Targeted});
  const std::size_t produceAction = descriptor.actions.size();
  CanonicalSyncAction bodySet{CanonicalSyncActionKind::EventSet,
                              source.resource,
                              {SyncCoverAnchorKind::LoopBodyExit, 0, scope},
                              0,
                              0,
                              {}};
  if (distance > 1) {
    bodySet.eventLaneKind = CanonicalSyncEventLaneKind::LoopIterationModulo;
    bodySet.eventLaneScope = scope;
  }
  descriptor.actions.push_back(std::move(bodySet));
  for (std::size_t lane = 0; lane < distance; ++lane) {
    descriptor.actions.push_back({CanonicalSyncActionKind::EventWait,
                                  target.resource,
                                  {SyncCoverAnchorKind::ScopeExit, 0, scope},
                                  0,
                                  lane,
                                  {}});
  }
  for (SyncCoverDemandId demandId : demandIds) {
    CanonicalSyncSupplyBinding binding;
    binding.edge = getDemandEdge(graph.getDemands()[demandId]);
    binding.eventUse = 0;
    binding.produceAction = produceAction;
    binding.consumeAction = consumeAction;
    binding.proof = CanonicalSyncSupplyProof::LoopBoundarySourcePrefixProtocol;
    binding.completionExport = CanonicalSyncSupplyExport::ScopeExitAfterDrain;
    binding.allowedDemands = {demandId};
    binding.attestedDemand = demandId;
    descriptor.supplies.push_back(std::move(binding));
  }
  return descriptor;
}

LogicalResult addLoopBoundarySourcePrefixProtocols(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
    const CanonicalSyncPatternOptions &options) {
  const SyncCoverGraph &graph = program.getGraph();
  const ArrayRef<SyncCoverDemandId> obligations =
      problem.getObligationDemands();
  if (obligations.size() > options.maximumLoopBoundaryProtocolInspections) {
    const CanonicalSyncProblemResult recorded =
        problem.recordLoopBoundaryProtocolGeneration(
            options.maximumLoopBoundaryProtocolInspections, 0, 0, true);
    if (!recorded) {
      return program.getFunction().emitError(
          "cannot record truncated loop-boundary protocol generation");
    }
    return success();
  }
  using GroupKey =
      std::tuple<SyncCoverScopeId, unsigned, std::uint32_t, std::uint32_t>;
  std::map<GroupKey, std::vector<SyncCoverDemandId>> groups;
  std::set<GroupKey> activeGroups;
  for (SyncCoverDemandId demandId : obligations) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    const bool eligible =
        demand.distance != 0 && source.resource != target.resource &&
        demand.scope < graph.getScopes().size() &&
        graph.getScopes()[demand.scope].isLoop &&
        graph.getScopes()[demand.scope].timeline &&
        graph.scopeContains(demand.scope, source.scope) &&
        graph.scopeContains(demand.scope, target.scope) &&
        graph.supportsBlockingTargetedBarrier(source.resource) &&
        domainIds.find({source.resource, target.resource}) != domainIds.end();
    if (!eligible) {
      continue;
    }
    const GroupKey key{demand.scope, demand.distance, source.resource,
                       target.resource};
    groups[key].push_back(demandId);
    if (std::binary_search(problem.getDemands().begin(),
                           problem.getDemands().end(), demandId)) {
      activeGroups.insert(key);
    }
  }
  bool truncated = false;
  std::size_t candidates = 0;
  std::size_t incidences = 0;
  for (const auto &[key, demandIds] : groups) {
    if (activeGroups.find(key) == activeGroups.end() || demandIds.size() < 2) {
      continue;
    }
    const std::size_t distance = std::get<1>(key);
    const std::size_t maximumActions =
        problem.getLimits().maximumActionsPerMechanism;
    const bool actionCountOverflows =
        maximumActions < 3 || distance > (maximumActions - 3) / 2;
    if (actionCountOverflows ||
        demandIds.size() > problem.getLimits().maximumSuppliesPerMechanism) {
      truncated = true;
      continue;
    }
    const bool candidateLimitReached =
        candidates >= options.maximumLoopBoundaryProtocolCandidates;
    const bool incidenceLimitReached =
        incidences > options.maximumLoopBoundaryProtocolIncidences ||
        demandIds.size() >
            options.maximumLoopBoundaryProtocolIncidences - incidences;
    if (candidateLimitReached || incidenceLimitReached) {
      truncated = true;
      break;
    }
    const auto domain = domainIds.find({std::get<2>(key), std::get<3>(key)});
    if (domain == domainIds.end()) {
      continue;
    }
    ++candidates;
    incidences += demandIds.size();
    CanonicalSyncMechanismDescriptor descriptor =
        makeLoopBoundarySourcePrefixProtocol(graph, domain->second,
                                             std::get<0>(key), std::get<1>(key),
                                             demandIds);
    const CanonicalSyncProblemResult added = problem.internVerifiedProtocol(
        std::move(descriptor), [&](const auto &verified) {
          return verified.kind == CanonicalSyncMechanismKind::Protocol &&
                 verified.eventUses.size() == 1 &&
                 verified.supplies.size() == demandIds.size() &&
                 llvm::all_of(
                     verified.supplies,
                     [&](const CanonicalSyncSupplyBinding &binding) {
                       return binding.proof ==
                                  CanonicalSyncSupplyProof::
                                      LoopBoundarySourcePrefixProtocol &&
                              binding.attestedDemand &&
                              llvm::is_contained(demandIds,
                                                 *binding.attestedDemand);
                     });
        });
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      truncated = true;
      break;
    }
    if (!added) {
      return program.getFunction().emitError(
          "cannot add canonical sync loop-boundary source-prefix protocol");
    }
  }
  const CanonicalSyncProblemResult recorded =
      problem.recordLoopBoundaryProtocolGeneration(
          obligations.size(), candidates, incidences, truncated);
  if (!recorded) {
    return program.getFunction().emitError(
        "cannot record truncated loop-boundary protocol generation");
  }
  return success();
}

CanonicalSyncMechanismDescriptor
makeBarrier(const SyncCoverGraph &graph,
            const std::vector<std::uint32_t> &allResources,
            ArrayRef<SyncCoverDemandId> demands, bool broad) {
  const SyncCoverDemand &first = graph.getDemands()[demands.front()];
  const SyncCoverNode &target = graph.getNodes()[first.target];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Barrier;
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::Barrier,
       target.resource,
       {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
       std::nullopt,
       0,
       broad ? allResources : std::vector<std::uint32_t>{target.resource},
       broad ? CanonicalSyncBarrierKind::All
             : CanonicalSyncBarrierKind::Targeted});
  for (SyncCoverDemandId demandId : demands) {
    CanonicalSyncSupplyBinding binding;
    binding.edge = getDemandEdge(graph.getDemands()[demandId]);
    binding.barrierAction = 0;
    descriptor.supplies.push_back(std::move(binding));
  }
  return descriptor;
}

LogicalResult addEventDomains(
    const CanonicalSyncProgram &program, unsigned budget,
    CanonicalSyncPatternProblem &problem, const SyncCoverDemandSet &baseline,
    std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds) {
  const SyncCoverGraph &graph = program.getGraph();
  std::set<EventDomainKey> keys;
  for (SyncCoverDemandId demandId : problem.getDemands()) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const bool needsMechanism = !baseline.contains(demandId);
    const bool supportsDirectEvent =
        canUsePreciseEvent(graph, demand) ||
        canUseSourceLocalCompletionEvent(program, demand) ||
        canUseTargetPrefixEvent(program, demand);
    if (!needsMechanism || !supportsDirectEvent) {
      continue;
    }
    const EventDomainKey key{graph.getNodes()[demand.source].resource,
                             graph.getNodes()[demand.target].resource};
    keys.insert(key);
  }
  for (const EventDomainKey &key : keys) {
    const CanonicalSyncEventDomainId id = domainIds.size();
    CanonicalSyncEventDomain domain{id, key.first, key.second, budget,
                                    getReservations(program, key)};
    const CanonicalSyncProblemResult added =
        problem.addEventDomain(std::move(domain));
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      return program.getFunction().emitError(
          "canonical sync event-domain limit prevents a complete catalog");
    }
    if (!added) {
      return program.getFunction().emitError(
          "cannot add canonical sync event domain");
    }
    domainIds.emplace(key, id);
  }
  return success();
}

LogicalResult addTargetCompletionCertificateEvents(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const SyncCoverDemandSet &baseline,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
    std::vector<DirectEventRecord> &directEvents) {
  const SyncCoverGraph &graph = program.getGraph();
  for (const SyncCoverTargetCompletionCertificate &certificate :
       graph.getTargetCompletionCertificates()) {
    const CanonicalSyncTargetCapabilities &capabilities =
        program.getTargetCapabilities();
    const bool capabilityEnabled =
        (certificate.kind == SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix &&
         capabilities.mte1L0ReadySetCompletesPrefix) ||
        (certificate.kind ==
             SyncCoverTargetCompletionKind::MToFixAccumulatorBoundary &&
         capabilities.mToFixAccumulatorBoundaryCompletes);
    const bool storageSpacesMatch = llvm::all_of(
        certificate.storageDomains, [&](SyncCoverStorageDomainId domain) {
          if (domain >= program.getStorageSpaces().size()) {
            return false;
          }
          const AddressSpace space = program.getStorageSpaces()[domain];
          return certificate.kind ==
                         SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix
                     ? space == AddressSpace::LEFT ||
                           space == AddressSpace::RIGHT
                     : space == AddressSpace::ACC;
        });
    if (!capabilityEnabled || !storageSpacesMatch) {
      return program.getFunction().emitError(
          "canonical sync target certificate is not authorized by the "
          "program target/storage contract");
    }
    const auto domain = domainIds.find(
        {certificate.sourceResource, certificate.targetResource});
    if (domain == domainIds.end()) {
      return program.getFunction().emitError(
          "canonical sync target certificate has no event domain");
    }
    std::vector<SyncCoverDemandId> activeDemands;
    for (SyncCoverDemandId demandId : certificate.demands) {
      if (!baseline.contains(demandId) &&
          llvm::is_contained(problem.getDemands(), demandId)) {
        activeDemands.push_back(demandId);
      }
    }
    if (activeDemands.empty()) {
      continue;
    }
    CanonicalSyncMechanismDescriptor descriptor;
    descriptor.kind = CanonicalSyncMechanismKind::Event;
    descriptor.eventUses.push_back({domain->second, 1, std::nullopt});
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::EventSet,
         certificate.sourceResource,
         {SyncCoverAnchorKind::AfterNode, certificate.completionNode, 0, 0},
         0,
         0,
         {}});
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::EventWait,
         certificate.targetResource,
         {SyncCoverAnchorKind::BeforeNode, certificate.target, 0, 0},
         0,
         0,
         {}});
    for (SyncCoverDemandId demandId : activeDemands) {
      CanonicalSyncSupplyBinding binding;
      binding.edge = getDemandEdge(graph.getDemands()[demandId]);
      binding.eventUse = 0;
      binding.proof =
          CanonicalSyncSupplyProof::TargetCompletionCertificateAction;
      binding.allowedDemands = {demandId};
      binding.attestedDemand = demandId;
      descriptor.supplies.push_back(std::move(binding));
    }
    const CanonicalSyncProblemResult added =
        problem.internMechanism(std::move(descriptor));
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      return program.getFunction().emitError(
          "canonical sync mechanism limit prevents target certificates");
    }
    if (!added || !added.index) {
      return program.getFunction().emitError(
                 "cannot add canonical sync target-certificate event, error=")
             << static_cast<unsigned>(added.error);
    }
    for (SyncCoverDemandId demandId : activeDemands) {
      directEvents.push_back({demandId, *added.index, domain->second});
    }
  }
  return success();
}

LogicalResult addExactEvents(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const SyncCoverDemandSet &baseline,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
    std::vector<DirectEventRecord> &directEvents,
    std::vector<ExactEventRecord> &exactEvents) {
  const SyncCoverGraph &graph = program.getGraph();
  for (SyncCoverDemandId demandId : problem.getDemands()) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const bool needsMechanism = !baseline.contains(demandId);
    if (!needsMechanism || !canUsePreciseEvent(graph, demand)) {
      continue;
    }
    const EventDomainKey key{graph.getNodes()[demand.source].resource,
                             graph.getNodes()[demand.target].resource};
    const auto domain = domainIds.find(key);
    if (domain == domainIds.end()) {
      continue;
    }
    const auto record = [&](CanonicalSyncProblemResult added,
                            bool directEvent) -> LogicalResult {
      if (added.error == CanonicalSyncProblemError::LimitExceeded) {
        return program.getFunction().emitError(
            "canonical sync mechanism limit prevents a complete catalog");
      }
      if (!added || !added.index) {
        return program.getFunction().emitError(
            "cannot add canonical sync precise event");
      }
      exactEvents.push_back({demandId, *added.index, domain->second});
      if (directEvent) {
        directEvents.push_back({demandId, *added.index, domain->second});
      }
      return success();
    };

    if (demand.distance == 0 && canUseDistanceZeroEvent(graph, demand) &&
        failed(record(problem.internMechanism(
                          makeDirectEvent(graph, demand, domain->second)),
                      true))) {
      return failure();
    }
    if (demand.distance != 0 && canUseRecurrenceEvent(graph, demand)) {
      CanonicalSyncMechanismDescriptor descriptor =
          makeRecurrenceEvent(graph, demand, domain->second);
      CanonicalSyncProblemResult added;
      if (descriptor.kind == CanonicalSyncMechanismKind::Protocol) {
        added = problem.internVerifiedProtocol(
            std::move(descriptor),
            [&](const CanonicalSyncMechanismDescriptor &actual) {
              return verifyRecurrenceEvent(graph, demand, domain->second,
                                           actual);
            });
      } else if (verifyRecurrenceEvent(graph, demand, domain->second,
                                       descriptor)) {
        added = problem.internMechanism(std::move(descriptor));
      } else {
        return program.getFunction().emitError(
            "cannot verify canonical sync forward recurrence event");
      }
      if (failed(record(added, false))) {
        return failure();
      }
    }
  }
  return success();
}

bool sameMechanismAction(const CanonicalSyncAction &left,
                         const CanonicalSyncAction &right) {
  return left.kind == right.kind && left.resource == right.resource &&
         left.anchor.kind == right.anchor.kind &&
         left.anchor.node == right.anchor.node &&
         left.anchor.scope == right.anchor.scope &&
         left.anchor.position == right.anchor.position &&
         left.eventUse == right.eventUse && left.eventLane == right.eventLane &&
         left.drainedResources == right.drainedResources &&
         left.barrierKind == right.barrierKind && left.guard == right.guard &&
         left.guardScope == right.guardScope &&
         left.eventLaneKind == right.eventLaneKind &&
         left.eventLaneScope == right.eventLaneScope;
}

bool sameMechanismSupply(const CanonicalSyncSupplyBinding &left,
                         const CanonicalSyncSupplyBinding &right) {
  const SyncCoverEdge &leftEdge = left.edge;
  const SyncCoverEdge &rightEdge = right.edge;
  return leftEdge.source == rightEdge.source &&
         leftEdge.target == rightEdge.target &&
         leftEdge.kind == rightEdge.kind && leftEdge.scope == rightEdge.scope &&
         leftEdge.distance == rightEdge.distance &&
         leftEdge.sourceGuard.literals == rightEdge.sourceGuard.literals &&
         leftEdge.targetGuard.literals == rightEdge.targetGuard.literals &&
         left.eventUse == right.eventUse &&
         left.barrierAction == right.barrierAction &&
         left.produceAction == right.produceAction &&
         left.consumeAction == right.consumeAction &&
         left.proof == right.proof &&
         left.completionExport == right.completionExport &&
         left.allowedDemands == right.allowedDemands &&
         left.attestedDemand == right.attestedDemand &&
         left.applicability == right.applicability;
}

bool sameMechanismDescriptor(const CanonicalSyncMechanismDescriptor &left,
                             const CanonicalSyncMechanismDescriptor &right) {
  const bool sameUses =
      left.eventUses.size() == right.eventUses.size() &&
      std::equal(left.eventUses.begin(), left.eventUses.end(),
                 right.eventUses.begin(),
                 [](const CanonicalSyncEventUse &first,
                    const CanonicalSyncEventUse &second) {
                   return first.domain == second.domain &&
                          first.width == second.width &&
                          first.recurrenceScope == second.recurrenceScope &&
                          first.lifetimeScope == second.lifetimeScope;
                 });
  return left.kind == right.kind && sameUses &&
         left.actions.size() == right.actions.size() &&
         left.supplies.size() == right.supplies.size() &&
         std::equal(left.actions.begin(), left.actions.end(),
                    right.actions.begin(), sameMechanismAction) &&
         std::equal(left.supplies.begin(), left.supplies.end(),
                    right.supplies.begin(), sameMechanismSupply);
}

void appendMechanismDescriptor(CanonicalSyncMechanismDescriptor &destination,
                               const CanonicalSyncMechanismDescriptor &part) {
  const std::size_t eventUseOffset = destination.eventUses.size();
  const std::size_t actionOffset = destination.actions.size();
  destination.eventUses.insert(destination.eventUses.end(),
                               part.eventUses.begin(), part.eventUses.end());
  for (CanonicalSyncAction action : part.actions) {
    if (action.eventUse) {
      *action.eventUse += eventUseOffset;
    }
    destination.actions.push_back(std::move(action));
  }
  for (CanonicalSyncSupplyBinding binding : part.supplies) {
    const std::optional<std::size_t> partEventUse = binding.eventUse;
    if (binding.eventUse) {
      *binding.eventUse += eventUseOffset;
    }
    if (binding.barrierAction) {
      *binding.barrierAction += actionOffset;
    }
    if (binding.produceAction) {
      *binding.produceAction += actionOffset;
    }
    if (binding.consumeAction) {
      *binding.consumeAction += actionOffset;
    }
    if (destination.kind == CanonicalSyncMechanismKind::Protocol &&
        part.kind != CanonicalSyncMechanismKind::Protocol && partEventUse) {
      for (std::size_t index = 0; index < part.actions.size(); ++index) {
        const CanonicalSyncAction &action = part.actions[index];
        if (action.eventUse != partEventUse) {
          continue;
        }
        if (action.kind == CanonicalSyncActionKind::EventSet) {
          binding.produceAction = actionOffset + index;
        } else if (action.kind == CanonicalSyncActionKind::EventWait) {
          binding.consumeAction = actionOffset + index;
        }
      }
      binding.proof = CanonicalSyncSupplyProof::VerifiedProtocol;
    }
    destination.supplies.push_back(std::move(binding));
  }
}

CanonicalSyncMechanismDescriptor
makeExactSlotLifecycleBundle(const CanonicalSyncMechanismDescriptor &ready,
                             const CanonicalSyncMechanismDescriptor &release) {
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Protocol;
  appendMechanismDescriptor(descriptor, ready);
  appendMechanismDescriptor(descriptor, release);
  return descriptor;
}

bool intervalsOverlap(SyncCoverStorageInterval first,
                      SyncCoverStorageInterval second) {
  return first.begin < second.end && second.begin < first.end;
}

bool isExactUnitSlotLifecycle(const SyncCoverGraph &graph,
                              SyncCoverDemandId readyId,
                              SyncCoverStorageWitnessId readyWitnessId,
                              SyncCoverDemandId releaseId,
                              SyncCoverStorageWitnessId releaseWitnessId) {
  if (readyId >= graph.getDemands().size() ||
      releaseId >= graph.getDemands().size() ||
      readyWitnessId >= graph.getStorageWitnesses().size() ||
      releaseWitnessId >= graph.getStorageWitnesses().size()) {
    return false;
  }
  const SyncCoverDemand &ready = graph.getDemands()[readyId];
  const SyncCoverDemand &release = graph.getDemands()[releaseId];
  const bool demandShape = ready.distance == 0 && release.distance == 1 &&
                           release.scope < graph.getScopes().size() &&
                           graph.getScopes()[release.scope].isLoop &&
                           graph.getScopes()[release.scope].timeline &&
                           llvm::is_contained(ready.provenanceKinds,
                                              SyncCoverDemandKind::MemoryRAW) &&
                           llvm::is_contained(release.provenanceKinds,
                                              SyncCoverDemandKind::MemoryWAR) &&
                           ready.source == release.target &&
                           ready.target == release.source &&
                           ready.sourceGuard.literals.empty() &&
                           ready.targetGuard.literals.empty() &&
                           release.sourceGuard.literals.empty() &&
                           release.targetGuard.literals.empty();
  if (!demandShape) {
    return false;
  }
  const SyncCoverStorageWitness &readyWitness =
      graph.getStorageWitnesses()[readyWitnessId];
  const SyncCoverStorageWitness &releaseWitness =
      graph.getStorageWitnesses()[releaseWitnessId];
  const auto &accesses = graph.getStorageAccesses();
  const SyncCoverStorageAccess &producer = accesses[readyWitness.sourceAccess];
  const SyncCoverStorageAccess &consumer = accesses[readyWitness.targetAccess];
  const bool reversedAccesses =
      readyWitness.sourceAccess == releaseWitness.targetAccess &&
      readyWitness.targetAccess == releaseWitness.sourceAccess;
  const bool exactSlot =
      reversedAccesses && producer.exactPhysical && consumer.exactPhysical &&
      producer.domain == consumer.domain &&
      producer.extent.begin == consumer.extent.begin &&
      producer.extent.end == consumer.extent.end &&
      readyWitness.overlap.begin == producer.extent.begin &&
      readyWitness.overlap.end == producer.extent.end &&
      releaseWitness.overlap.begin == producer.extent.begin &&
      releaseWitness.overlap.end == producer.extent.end &&
      producer.mode == SyncCoverStorageAccessMode::Write &&
      consumer.mode == SyncCoverStorageAccessMode::Read;
  if (!exactSlot) {
    return false;
  }
  const SyncCoverNode &producerNode = graph.getNodes()[producer.node];
  const SyncCoverNode &consumerNode = graph.getNodes()[consumer.node];
  const bool nodeShape =
      producer.node == ready.source && consumer.node == ready.target &&
      producerNode.resource != consumerNode.resource &&
      producerNode.order < consumerNode.order &&
      producerNode.guard.literals.empty() &&
      consumerNode.guard.literals.empty() &&
      graph.scopeMustExecuteWithin(release.scope, producerNode.scope) &&
      graph.scopeMustExecuteWithin(release.scope, consumerNode.scope) &&
      syncCoverNodeCanProduceCompletion(graph, producerNode.id,
                                        consumerNode.resource) &&
      syncCoverNodeCanProduceCompletion(graph, consumerNode.id,
                                        producerNode.resource);
  if (!nodeShape) {
    return false;
  }
  for (const SyncCoverStorageAccess &access : accesses) {
    if (access.domain != producer.domain ||
        !intervalsOverlap(access.extent, producer.extent)) {
      continue;
    }
    if (access.id != producer.id && access.id != consumer.id) {
      return false;
    }
  }
  return true;
}

LogicalResult
addExactSlotLifecycleBundles(const CanonicalSyncProgram &program,
                             CanonicalSyncPatternProblem &problem,
                             ArrayRef<ExactEventRecord> exactEvents,
                             const CanonicalSyncPatternOptions &options) {
  const SyncCoverGraph &graph = program.getGraph();
  using LifecycleKey = std::tuple<SyncCoverStorageDomainId, std::uint64_t,
                                  std::uint64_t, std::uint32_t, std::uint32_t>;
  struct Opportunity {
    ExactEventRecord event;
    SyncCoverStorageWitnessId witness = 0;
  };
  std::map<LifecycleKey, std::vector<Opportunity>> readyBySlot;
  std::size_t inspections = 0;
  std::size_t candidates = 0;
  bool truncated = false;
  const auto consumeInspection = [&](std::size_t amount = 1) {
    if (inspections > options.maximumSlotLifecycleInspections ||
        amount > options.maximumSlotLifecycleInspections - inspections) {
      truncated = true;
      return false;
    }
    inspections += amount;
    return true;
  };
  for (const ExactEventRecord &event : exactEvents) {
    if (!consumeInspection()) {
      break;
    }
    const SyncCoverDemand &demand = graph.getDemands()[event.demand];
    if (demand.distance != 0 ||
        !llvm::is_contained(demand.provenanceKinds,
                            SyncCoverDemandKind::MemoryRAW)) {
      continue;
    }
    for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
      if (!consumeInspection()) {
        break;
      }
      const SyncCoverStorageWitness &witness =
          graph.getStorageWitnesses()[witnessId];
      const SyncCoverStorageAccess &source =
          graph.getStorageAccesses()[witness.sourceAccess];
      const SyncCoverStorageAccess &target =
          graph.getStorageAccesses()[witness.targetAccess];
      const bool wholeExactSlot =
          source.exactPhysical && target.exactPhysical &&
          source.mode == SyncCoverStorageAccessMode::Write &&
          target.mode == SyncCoverStorageAccessMode::Read &&
          source.domain == target.domain &&
          source.extent.begin == target.extent.begin &&
          source.extent.end == target.extent.end &&
          witness.overlap.begin == source.extent.begin &&
          witness.overlap.end == source.extent.end;
      if (!wholeExactSlot) {
        continue;
      }
      const SyncCoverNode &sourceNode = graph.getNodes()[demand.source];
      const SyncCoverNode &targetNode = graph.getNodes()[demand.target];
      readyBySlot[{source.domain, source.extent.begin, source.extent.end,
                   sourceNode.resource, targetNode.resource}]
          .push_back({event, witnessId});
    }
    if (truncated) {
      break;
    }
  }

  std::set<std::pair<CanonicalSyncMechanismId, CanonicalSyncMechanismId>>
      admittedPairs;
  std::map<CanonicalSyncMechanismId, std::vector<CanonicalSyncMechanismId>>
      bundlesByComponent;
  for (const ExactEventRecord &releaseEvent : exactEvents) {
    if (truncated || !consumeInspection()) {
      break;
    }
    const SyncCoverDemand &release = graph.getDemands()[releaseEvent.demand];
    if (release.distance != 1 ||
        !llvm::is_contained(release.provenanceKinds,
                            SyncCoverDemandKind::MemoryWAR)) {
      continue;
    }
    for (SyncCoverStorageWitnessId releaseWitnessId :
         release.storageWitnesses) {
      if (!consumeInspection()) {
        break;
      }
      const SyncCoverStorageWitness &releaseWitness =
          graph.getStorageWitnesses()[releaseWitnessId];
      const SyncCoverStorageAccess &producer =
          graph.getStorageAccesses()[releaseWitness.targetAccess];
      const SyncCoverNode &consumerNode = graph.getNodes()[release.source];
      const SyncCoverNode &producerNode = graph.getNodes()[release.target];
      const LifecycleKey key{producer.domain, producer.extent.begin,
                             producer.extent.end, producerNode.resource,
                             consumerNode.resource};
      const auto ready = readyBySlot.find(key);
      if (ready == readyBySlot.end()) {
        continue;
      }
      for (const Opportunity &readyOpportunity : ready->second) {
        if (!consumeInspection()) {
          break;
        }
        const std::pair<CanonicalSyncMechanismId, CanonicalSyncMechanismId>
            pair = std::minmax(readyOpportunity.event.mechanism,
                               releaseEvent.mechanism);
        if (pair.first == pair.second || admittedPairs.count(pair) != 0) {
          continue;
        }
        const std::size_t accessCount = graph.getStorageAccesses().size();
        const bool accessWorkOverflows =
            accessCount > std::numeric_limits<std::size_t>::max() / 2;
        if (accessWorkOverflows || !consumeInspection(accessCount * 2)) {
          break;
        }
        if (!isExactUnitSlotLifecycle(graph, readyOpportunity.event.demand,
                                      readyOpportunity.witness,
                                      releaseEvent.demand, releaseWitnessId)) {
          continue;
        }
        if (candidates >= options.maximumSlotLifecycleCandidates) {
          truncated = true;
          break;
        }
        const CanonicalSyncMechanismDescriptor &readyDescriptor =
            problem.getMechanisms()[readyOpportunity.event.mechanism]
                .descriptor;
        const CanonicalSyncMechanismDescriptor &releaseDescriptor =
            problem.getMechanisms()[releaseEvent.mechanism].descriptor;
        const bool exactComponents =
            sameMechanismDescriptor(
                readyDescriptor,
                makeDirectEvent(
                    graph, graph.getDemands()[readyOpportunity.event.demand],
                    readyOpportunity.event.domain)) &&
            verifyRecurrenceEvent(graph,
                                  graph.getDemands()[releaseEvent.demand],
                                  releaseEvent.domain, releaseDescriptor);
        if (!exactComponents) {
          continue;
        }
        CanonicalSyncMechanismDescriptor descriptor =
            makeExactSlotLifecycleBundle(readyDescriptor, releaseDescriptor);
        const CanonicalSyncMechanismDescriptor expected = descriptor;
        const CanonicalSyncProblemResult added = problem.internVerifiedProtocol(
            std::move(descriptor),
            [&](const CanonicalSyncMechanismDescriptor &actual) {
              return isExactUnitSlotLifecycle(
                         graph, readyOpportunity.event.demand,
                         readyOpportunity.witness, releaseEvent.demand,
                         releaseWitnessId) &&
                     sameMechanismDescriptor(actual, expected);
            });
        if (added.error == CanonicalSyncProblemError::LimitExceeded) {
          truncated = true;
          break;
        }
        if (!added || !added.index) {
          return program.getFunction().emitError(
                     "cannot add verified exact-slot lifecycle bundle, error=")
                 << static_cast<unsigned>(added.error);
        }
        const CanonicalSyncMechanismId bundle = *added.index;
        for (CanonicalSyncMechanismId component :
             {readyOpportunity.event.mechanism, releaseEvent.mechanism}) {
          if (bundle != component && !problem.addConflict(bundle, component)) {
            return program.getFunction().emitError(
                "cannot record exact-slot lifecycle component conflict");
          }
          for (CanonicalSyncMechanismId prior : bundlesByComponent[component]) {
            if (prior != bundle && !problem.addConflict(bundle, prior)) {
              return program.getFunction().emitError(
                  "cannot record overlapping exact-slot lifecycle conflict");
            }
          }
          bundlesByComponent[component].push_back(bundle);
        }
        admittedPairs.insert(pair);
        ++candidates;
      }
      if (truncated) {
        break;
      }
    }
  }
  const CanonicalSyncProblemResult recorded =
      problem.recordSlotLifecycleGeneration(inspections, candidates, truncated);
  if (!recorded) {
    return program.getFunction().emitError(
        "cannot record exact-slot lifecycle generation");
  }
  return success();
}

LogicalResult addCompletionFrontierEvents(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const SyncCoverDemandSet &baseline,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
    std::vector<DirectEventRecord> &directEvents) {
  const SyncCoverGraph &graph = program.getGraph();
  CompletionFrontierIndex frontierIndex(graph);
  for (SyncCoverDemandId demandId : problem.getDemands()) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (baseline.contains(demandId) || demand.distance != 0 ||
        graph.getNodes()[demand.source].resource ==
            graph.getNodes()[demand.target].resource) {
      continue;
    }
    const std::optional<SyncCoverNodeId> frontier =
        frontierIndex.findLatest(demand);
    if (!frontier) {
      continue;
    }
    const EventDomainKey key{graph.getNodes()[*frontier].resource,
                             graph.getNodes()[demand.target].resource};
    const auto domain = domainIds.find(key);
    if (domain == domainIds.end()) {
      continue;
    }
    CanonicalSyncMechanismDescriptor descriptor =
        makeCompletionFrontierEvent(graph, demand, *frontier, domain->second);
    if (!verifyCompletionFrontierEvent(graph, demand, *frontier, domain->second,
                                       descriptor)) {
      return program.getFunction().emitError(
          "cannot verify canonical sync completion-frontier event");
    }
    const CanonicalSyncProblemResult added =
        problem.internMechanism(std::move(descriptor));
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      return program.getFunction().emitError(
          "canonical sync mechanism limit prevents completion frontiers");
    }
    if (!added || !added.index) {
      return program.getFunction().emitError(
          "cannot add canonical sync completion-frontier event");
    }
    directEvents.push_back({demandId, *added.index, domain->second});
  }
  return success();
}

enum class TargetLocalCatalogMode : std::uint8_t {
  EventGroupsOnly,
  PipeDrainsOnly,
};

LogicalResult addTargetLocalFenceEvents(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    ArrayRef<SyncCoverDemandId> demandIds,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
    bool requireDistanceZeroBinding, TargetLocalCatalogMode mode) {
  const SyncCoverGraph &graph = program.getGraph();
  using TargetFenceGroupKey =
      std::pair<CanonicalSyncEventDomainId, SyncCoverNodeId>;
  using TargetPipeDrainGroupKey = std::pair<std::uint32_t, SyncCoverNodeId>;
  std::map<TargetFenceGroupKey, std::vector<SyncCoverDemandId>> eventGroups;
  std::map<TargetPipeDrainGroupKey, std::vector<SyncCoverDemandId>>
      pipeDrainGroups;
  for (SyncCoverDemandId demandId : demandIds) {
    if (demandId >= graph.getDemands().size()) {
      return program.getFunction().emitError(
          "canonical sync target-local fence names an invalid demand");
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (!canUseTargetPrefixEvent(program, demand)) {
      continue;
    }
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    if (targetPrefixNeedsBarrier(program, demand)) {
      if (mode == TargetLocalCatalogMode::PipeDrainsOnly) {
        pipeDrainGroups[{source.resource, target.physicalAnchor}].push_back(
            demandId);
      }
      continue;
    }
    if (mode != TargetLocalCatalogMode::EventGroupsOnly) {
      continue;
    }
    const EventDomainKey key{source.resource, target.resource};
    const auto domain = domainIds.find(key);
    if (domain == domainIds.end()) {
      continue;
    }
    eventGroups[{domain->second, target.physicalAnchor}].push_back(demandId);
  }

  const auto acceptsGroup = [&](ArrayRef<SyncCoverDemandId> groupedDemands) {
    const bool hasDistanceZeroBinding =
        llvm::any_of(groupedDemands, [&](SyncCoverDemandId demandId) {
          return graph.getDemands()[demandId].distance == 0;
        });
    return !requireDistanceZeroBinding || hasDistanceZeroBinding;
  };
  bool optionalCatalogFull = false;
  const auto internDescriptor = [&](CanonicalSyncMechanismDescriptor descriptor,
                                    StringRef role,
                                    bool optional) -> LogicalResult {
    if (optionalCatalogFull) {
      return success();
    }
    const CanonicalSyncProblemResult added =
        problem.internMechanism(std::move(descriptor));
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      if (optional) {
        optionalCatalogFull = problem.getMechanisms().size() >=
                              problem.getLimits().maximumMechanisms;
        problem.markPatternGenerationTruncated();
        return success();
      }
      return program.getFunction().emitError(
          "canonical sync mechanism limit prevents a complete catalog");
    }
    if (!added || !added.index) {
      return program.getFunction().emitError(
                 "cannot add canonical sync target-local ")
             << role << ", error=" << static_cast<unsigned>(added.error);
    }
    return success();
  };
  const auto addEventGroup =
      [&](ArrayRef<SyncCoverDemandId> groupedDemands,
          CanonicalSyncEventDomainId domain) -> LogicalResult {
    if (!acceptsGroup(groupedDemands)) {
      return success();
    }
    CanonicalSyncMechanismDescriptor descriptor =
        makeTargetPrefixEvent(program, groupedDemands, domain);
    if (!verifyTargetPrefixEvent(program, groupedDemands, domain, descriptor)) {
      return program.getFunction().emitError(
          "cannot verify canonical sync target-prefix event group");
    }
    return internDescriptor(std::move(descriptor), "event group", true);
  };
  for (const auto &[key, groupedDemands] : eventGroups) {
    if (failed(addEventGroup(groupedDemands, key.first))) {
      return failure();
    }
  }
  for (const auto &[key, groupedDemands] : pipeDrainGroups) {
    (void)key;
    if (!acceptsGroup(groupedDemands)) {
      continue;
    }
    CanonicalSyncMechanismDescriptor descriptor =
        makeTargetPipeDrainCut(program, groupedDemands);
    if (!verifyTargetPipeDrainCut(program, groupedDemands, descriptor)) {
      return program.getFunction().emitError(
          "cannot verify canonical sync target pipe-drain cut");
    }
    if (failed(
            internDescriptor(std::move(descriptor), "pipe-drain cut", false))) {
      return failure();
    }
  }
  return success();
}

LogicalResult addTargetedBarriers(const CanonicalSyncProgram &program,
                                  CanonicalSyncPatternProblem &problem,
                                  const SyncCoverDemandSet &baseline) {
  const SyncCoverGraph &graph = program.getGraph();
  const std::vector<std::uint32_t> allResources = getIssueResources(graph);
  using TargetedBarrierKey = std::pair<SyncCoverNodeId, std::uint32_t>;
  std::map<TargetedBarrierKey, BarrierFallbackGroup> targetedGroups;
  for (SyncCoverDemandId demandId : problem.getDemands()) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (baseline.contains(demandId)) {
      continue;
    }
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    if (source.resource == target.resource) {
      targetedGroups[{target.physicalAnchor, target.resource}]
          .demands.push_back(demandId);
    }
  }
  const auto addGroups = [&](const auto &groups, bool broad) -> LogicalResult {
    for (const auto &[key, group] : groups) {
      (void)key;
      const bool hasDemands = !group.demands.empty();
      const bool added =
          hasDemands && problem.internMechanism(makeBarrier(
                            graph, allResources, group.demands, broad));
      if (!added) {
        return failure();
      }
    }
    return success();
  };
  if (failed(addGroups(targetedGroups, false))) {
    return program.getFunction().emitError(
        "cannot add canonical sync targeted barrier");
  }
  return success();
}

CanonicalSyncMechanismDescriptor
makeSourceLocalPipeDrain(const SyncCoverGraph &graph,
                         ArrayRef<SyncCoverDemandId> demands) {
  const SyncCoverDemand &firstDemand = graph.getDemands()[demands.front()];
  const SyncCoverNode &source = graph.getNodes()[firstDemand.source];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Barrier;
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::Barrier,
       source.resource,
       {SyncCoverAnchorKind::AfterNode, source.physicalExit, 0, 0},
       std::nullopt,
       0,
       {source.resource},
       CanonicalSyncBarrierKind::Targeted});
  for (SyncCoverDemandId demandId : demands) {
    CanonicalSyncSupplyBinding binding;
    binding.edge = getDemandEdge(graph.getDemands()[demandId]);
    binding.barrierAction = 0;
    binding.proof = CanonicalSyncSupplyProof::SourceLocalPipeDrainAction;
    binding.attestedDemand = demandId;
    if (binding.edge.distance == 0) {
      binding.applicability = SyncCoverSupplyApplicability::DistanceZeroOnly;
    } else {
      binding.allowedDemands = {demandId};
    }
    descriptor.supplies.push_back(std::move(binding));
  }
  return descriptor;
}

LogicalResult addSourceLocalPipeDrains(const CanonicalSyncProgram &program,
                                       CanonicalSyncPatternProblem &problem,
                                       const SyncCoverDemandSet &baseline,
                                       ArrayRef<SyncCoverDemandId> demandIds,
                                       bool crossResource) {
  const SyncCoverGraph &graph = program.getGraph();
  using SourceDrainKey = std::pair<SyncCoverNodeId, std::uint32_t>;
  std::map<SourceDrainKey, std::vector<SyncCoverDemandId>> groups;
  for (SyncCoverDemandId demandId : demandIds) {
    if (baseline.contains(demandId)) {
      continue;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    if ((source.resource != target.resource) != crossResource) {
      continue;
    }
    const bool samePhysicalOperation =
        source.physicalAnchor == target.physicalAnchor;
    const auto sourcePosition = resolveSyncCoverAnchor(
        graph, {SyncCoverAnchorKind::AfterNode, source.physicalExit, 0, 0});
    const auto targetPosition = resolveSyncCoverAnchor(
        graph, {SyncCoverAnchorKind::BeforeNode, target.physicalAnchor, 0, 0});
    const bool invalidDistanceZeroOrder =
        demand.distance == 0 &&
        (samePhysicalOperation ||
         source.physicalExit >= graph.getNodes().size() ||
         target.physicalAnchor >= graph.getNodes().size() ||
         graph.getNodes()[source.physicalExit].order >=
             graph.getNodes()[target.physicalAnchor].order ||
         !sourcePosition || !targetPosition ||
         *sourcePosition >= *targetPosition);
    if (invalidDistanceZeroOrder ||
        !graph.supportsBlockingTargetedBarrier(source.resource)) {
      continue;
    }
    groups[{source.physicalExit, source.resource}].push_back(demandId);
  }
  for (const auto &[key, demands] : groups) {
    (void)key;
    const CanonicalSyncProblemResult added =
        problem.internMechanism(makeSourceLocalPipeDrain(graph, demands));
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      continue;
    }
    if (!added) {
      return program.getFunction().emitError(
          "cannot add canonical sync source-local pipe drain");
    }
  }
  return success();
}

CanonicalSyncMechanismDescriptor
makeSourcePrefixPipeDrain(const SyncCoverGraph &graph, SyncCoverNodeId cut,
                          std::uint32_t resource,
                          ArrayRef<SyncCoverDemandId> demands) {
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Barrier;
  descriptor.actions.push_back({CanonicalSyncActionKind::Barrier,
                                resource,
                                {SyncCoverAnchorKind::AfterNode, cut, 0, 0},
                                std::nullopt,
                                0,
                                {resource},
                                CanonicalSyncBarrierKind::Targeted});
  for (SyncCoverDemandId demandId : demands) {
    CanonicalSyncSupplyBinding binding;
    binding.edge = getDemandEdge(graph.getDemands()[demandId]);
    binding.barrierAction = 0;
    binding.proof = CanonicalSyncSupplyProof::SourcePrefixPipeDrainAction;
    binding.attestedDemand = demandId;
    if (binding.edge.distance == 0) {
      binding.applicability = SyncCoverSupplyApplicability::DistanceZeroOnly;
    } else {
      binding.allowedDemands = {demandId};
    }
    descriptor.supplies.push_back(std::move(binding));
  }
  return descriptor;
}

LogicalResult addSourcePrefixPipeDrains(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const SyncCoverDemandSet &baseline,
    const CanonicalSyncPatternOptions &options,
    ArrayRef<SyncCoverDemandId> demandIds, bool crossResource) {
  const SyncCoverGraph &graph = program.getGraph();
  using SourceDrainKey = std::pair<SyncCoverNodeId, std::uint32_t>;
  using SourceControlKey =
      std::tuple<SyncCoverScopeId, std::vector<SyncCoverGuardLiteral>,
                 std::uint32_t>;
  std::map<SyncCoverNodeId, std::vector<SyncCoverDemandId>> demandsBySource;
  std::map<SourceDrainKey, SyncCoverNodeId> cuts;
  const CanonicalSyncPatternStatistics &priorStatistics =
      problem.getPatternStatistics();
  const std::size_t priorInspections = priorStatistics.sourcePrefixInspections;
  const std::size_t priorCandidates = priorStatistics.sourcePrefixCandidates;
  const std::size_t priorIncidences = priorStatistics.sourcePrefixIncidences;
  std::size_t inspections = 0;
  std::size_t candidates = 0;
  std::size_t incidences = 0;
  bool truncated = false;
  const auto consumeInspections = [&](std::size_t amount = 1) {
    if (priorInspections > options.maximumSourcePrefixInspections ||
        inspections >
            options.maximumSourcePrefixInspections - priorInspections ||
        amount > options.maximumSourcePrefixInspections - priorInspections -
                     inspections) {
      truncated = true;
      return false;
    }
    inspections += amount;
    return true;
  };
  for (SyncCoverDemandId demandId : demandIds) {
    if (!consumeInspections()) {
      break;
    }
    if (baseline.contains(demandId)) {
      continue;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    if ((source.resource != target.resource) != crossResource) {
      continue;
    }
    if (!graph.supportsBlockingTargetedBarrier(source.resource)) {
      continue;
    }
    demandsBySource[source.id].push_back(demandId);
    cuts.try_emplace({source.physicalExit, source.resource}, source.id);
  }

  std::map<SourceControlKey, std::vector<SyncCoverNodeId>> sourcesByControl;
  std::map<SourceControlKey,
           std::vector<std::pair<SourceDrainKey, SyncCoverNodeId>>>
      cutsByControl;
  if (!truncated) {
    for (const auto &[sourceId, demands] : demandsBySource) {
      (void)demands;
      if (!consumeInspections()) {
        break;
      }
      const SyncCoverNode &source = graph.getNodes()[sourceId];
      sourcesByControl[{source.scope, source.guard.literals, source.resource}]
          .push_back(sourceId);
    }
  }
  if (!truncated) {
    for (const auto &[key, representativeId] : cuts) {
      if (!consumeInspections()) {
        break;
      }
      const SyncCoverNode &cut = graph.getNodes()[key.first];
      cutsByControl[{cut.scope, cut.guard.literals, key.second}].push_back(
          {key, representativeId});
    }
  }

  bool stop = truncated;
  for (const auto &[control, indexedCuts] : cutsByControl) {
    if (stop) {
      break;
    }
    const auto indexedSources = sourcesByControl.find(control);
    if (indexedSources == sourcesByControl.end()) {
      continue;
    }
    for (const auto &[key, representativeId] : indexedCuts) {
      if (stop) {
        break;
      }
      const SyncCoverNode &representative = graph.getNodes()[representativeId];
      const SyncCoverNode &cut = graph.getNodes()[key.first];
      const auto issuedPrefix = graph.getBlockingTargetedBarrierPrefixes().find(
          {key.second, representative.physicalAnchor});
      bool expandsLocalCut = false;
      bool groupExceeded = false;
      std::vector<SyncCoverDemandId> groupedDemands;
      for (SyncCoverNodeId sourceId : indexedSources->second) {
        if (!consumeInspections()) {
          stop = true;
          break;
        }
        const SyncCoverNode &source = graph.getNodes()[sourceId];
        const bool samePhysicalCut =
            source.physicalAnchor == representative.physicalAnchor;
        const bool inIssuedPrefix =
            issuedPrefix != graph.getBlockingTargetedBarrierPrefixes().end() &&
            std::binary_search(issuedPrefix->second.begin(),
                               issuedPrefix->second.end(), sourceId);
        if (!samePhysicalCut && !inIssuedPrefix) {
          continue;
        }
        const auto sourceDemands = demandsBySource.find(sourceId);
        if (sourceDemands == demandsBySource.end()) {
          continue;
        }
        for (SyncCoverDemandId demandId : sourceDemands->second) {
          if (!consumeInspections()) {
            stop = true;
            break;
          }
          const SyncCoverDemand &demand = graph.getDemands()[demandId];
          const SyncCoverNode &target = graph.getNodes()[demand.target];
          bool validPlacement = false;
          if (demand.distance == 0) {
            const auto cutPosition = resolveSyncCoverAnchor(
                graph, {SyncCoverAnchorKind::AfterNode, cut.id, 0, 0});
            const auto targetPosition =
                resolveSyncCoverAnchor(graph, {SyncCoverAnchorKind::BeforeNode,
                                               target.physicalAnchor, 0, 0});
            validPlacement =
                cutPosition && targetPosition && *cutPosition < *targetPosition;
          } else {
            const SyncCoverExpandedArena *arena =
                problem.getExpansion().getArena(demand);
            validPlacement =
                arena &&
                problem.getExpansion().projectEndpoint(graph, *arena, cut.id,
                                                       0) &&
                problem.getExpansion().projectEndpoint(
                    graph, *arena, target.physicalAnchor, demand.distance);
          }
          if (!validPlacement) {
            continue;
          }
          if (groupedDemands.size() >=
              problem.getLimits().maximumSuppliesPerMechanism) {
            truncated = true;
            groupExceeded = true;
            break;
          }
          groupedDemands.push_back(demandId);
          expandsLocalCut = expandsLocalCut || !samePhysicalCut;
        }
        if (stop || groupExceeded) {
          break;
        }
      }
      if (stop) {
        break;
      }
      if (groupExceeded || !expandsLocalCut || groupedDemands.empty()) {
        continue;
      }
      llvm::sort(groupedDemands);
      groupedDemands.erase(
          std::unique(groupedDemands.begin(), groupedDemands.end()),
          groupedDemands.end());
      const bool candidateLimitReached =
          priorCandidates > options.maximumSourcePrefixCandidates ||
          candidates >= options.maximumSourcePrefixCandidates - priorCandidates;
      const bool incidenceLimitReached =
          priorIncidences > options.maximumSourcePrefixIncidences ||
          incidences >
              options.maximumSourcePrefixIncidences - priorIncidences ||
          groupedDemands.size() > options.maximumSourcePrefixIncidences -
                                      priorIncidences - incidences;
      if (candidateLimitReached || incidenceLimitReached) {
        truncated = true;
        stop = true;
        break;
      }
      const CanonicalSyncProblemResult added = problem.internMechanism(
          makeSourcePrefixPipeDrain(graph, cut.id, key.second, groupedDemands));
      if (added.error == CanonicalSyncProblemError::LimitExceeded) {
        truncated = true;
        stop = true;
        break;
      }
      if (!added) {
        return program.getFunction().emitError(
            "cannot add canonical sync source-prefix pipe drain");
      }
      ++candidates;
      incidences += groupedDemands.size();
    }
  }
  const CanonicalSyncProblemResult recorded =
      problem.recordSourcePrefixGeneration(inspections, candidates, incidences,
                                           truncated);
  if (!recorded) {
    return program.getFunction().emitError(
        "cannot record canonical sync source-prefix generation");
  }
  return success();
}

CanonicalSyncMechanismDescriptor
makeLoopCarryPipeDrain(const SyncCoverGraph &graph, SyncCoverScopeId scope,
                       std::uint32_t resource,
                       ArrayRef<SyncCoverDemandId> demands) {
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Barrier;
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::Barrier,
       resource,
       {SyncCoverAnchorKind::LoopBodyEntry, 0, scope, 0},
       std::nullopt,
       0,
       {resource},
       CanonicalSyncBarrierKind::Targeted,
       CanonicalSyncActionGuardKind::NotFirstIteration,
       scope});
  for (SyncCoverDemandId demandId : demands) {
    CanonicalSyncSupplyBinding binding;
    binding.edge = getDemandEdge(graph.getDemands()[demandId]);
    binding.barrierAction = 0;
    binding.proof = CanonicalSyncSupplyProof::LoopCarryPipeDrain;
    binding.attestedDemand = demandId;
    binding.allowedDemands = {demandId};
    descriptor.supplies.push_back(std::move(binding));
  }
  return descriptor;
}

LogicalResult
addLoopCarryPipeDrains(const CanonicalSyncProgram &program,
                       CanonicalSyncPatternProblem &problem,
                       const SyncCoverDemandSet &baseline,
                       const CanonicalSyncPatternOptions &options) {
  const SyncCoverGraph &graph = program.getGraph();
  const ArrayRef<SyncCoverDemandId> demands = problem.getDemands();
  if (demands.size() > options.maximumLoopCarryInspections) {
    const CanonicalSyncProblemResult recorded =
        problem.recordLoopCarryGeneration(options.maximumLoopCarryInspections,
                                          0, 0, true);
    if (!recorded) {
      return program.getFunction().emitError(
          "cannot record truncated canonical sync loop-carry generation");
    }
    return success();
  }
  using LoopCarryKey = std::pair<SyncCoverScopeId, std::uint32_t>;
  std::map<LoopCarryKey, std::vector<SyncCoverDemandId>> groups;
  for (SyncCoverDemandId demandId : demands) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (baseline.contains(demandId) || demand.distance == 0 ||
        demand.scope >= graph.getScopes().size() ||
        !graph.getScopes()[demand.scope].isLoop) {
      continue;
    }
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    if (!graph.supportsBlockingTargetedBarrier(source.resource)) {
      continue;
    }
    groups[{demand.scope, source.resource}].push_back(demandId);
  }
  bool truncated = false;
  std::size_t candidates = 0;
  std::size_t incidences = 0;
  for (const auto &[key, groupDemands] : groups) {
    if (groupDemands.size() > problem.getLimits().maximumSuppliesPerMechanism) {
      truncated = true;
      continue;
    }
    const bool candidateLimitReached =
        candidates >= options.maximumLoopCarryCandidates;
    const bool incidenceLimitReached =
        incidences > options.maximumLoopCarryIncidences ||
        groupDemands.size() > options.maximumLoopCarryIncidences - incidences;
    if (candidateLimitReached || incidenceLimitReached) {
      truncated = true;
      break;
    }
    const CanonicalSyncProblemResult added = problem.internMechanism(
        makeLoopCarryPipeDrain(graph, key.first, key.second, groupDemands));
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      truncated = true;
      break;
    }
    if (!added) {
      return program.getFunction().emitError(
          "cannot add canonical sync loop-carry pipe drain");
    }
    ++candidates;
    incidences += groupDemands.size();
  }
  const CanonicalSyncProblemResult recorded = problem.recordLoopCarryGeneration(
      demands.size(), candidates, incidences, truncated);
  if (!recorded) {
    return program.getFunction().emitError(
        "cannot record canonical sync loop-carry generation");
  }
  return success();
}

LogicalResult addPipeAllBackstop(const CanonicalSyncProgram &program,
                                 CanonicalSyncPatternProblem &problem,
                                 const SyncCoverDemandSet &baseline) {
  const SyncCoverGraph &graph = program.getGraph();
  const std::vector<std::uint32_t> allResources = getIssueResources(graph);
  std::map<SyncCoverNodeId, BarrierFallbackGroup> groups;
  for (SyncCoverDemandId demandId : problem.getDemands()) {
    if (!baseline.contains(demandId)) {
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      groups[demand.target].demands.push_back(demandId);
    }
  }
  for (const auto &[target, group] : groups) {
    (void)target;
    if (!problem.internMechanism(
            makeBarrier(graph, allResources, group.demands, true))) {
      return program.getFunction().emitError(
          "cannot add canonical sync localized PIPE_ALL backstop");
    }
  }
  return success();
}

CanonicalSyncMechanismDescriptor makeRepairBarrier(const SyncCoverGraph &graph,
                                                   const SyncCoverEdge &edge) {
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Barrier;
  const SyncCoverNode &target = graph.getNodes()[edge.target];
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::Barrier,
       target.resource,
       {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
       std::nullopt,
       0,
       {target.resource},
       CanonicalSyncBarrierKind::Targeted});
  CanonicalSyncSupplyBinding supply;
  supply.edge = edge;
  supply.barrierAction = 0;
  descriptor.supplies.push_back(std::move(supply));
  return descriptor;
}

CanonicalSyncMechanismDescriptor
makeRepairEvent(const SyncCoverGraph &graph, const SyncCoverEdge &edge,
                CanonicalSyncEventDomainId domain) {
  const SyncCoverNode &source = graph.getNodes()[edge.source];
  const SyncCoverNode &target = graph.getNodes()[edge.target];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Event;
  descriptor.eventUses.push_back({domain, 1, std::nullopt});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventSet,
       source.resource,
       {SyncCoverAnchorKind::AfterNode, source.id, 0, 0},
       0,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventWait,
       target.resource,
       {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
       0,
       0,
       {}});
  CanonicalSyncSupplyBinding supply;
  supply.edge = edge;
  supply.eventUse = 0;
  descriptor.supplies.push_back(std::move(supply));
  return descriptor;
}

struct RepairFrontierProposal {
  SyncCoverEdge barrier;
  SyncCoverEdge event;
  CanonicalSyncEventDomainId domain = 0;
};

bool frontierProposalLess(const RepairFrontierProposal &left,
                          const RepairFrontierProposal &right) {
  return std::tie(left.domain, left.barrier.source, left.barrier.target,
                  left.event.source, left.event.target) <
         std::tie(right.domain, right.barrier.source, right.barrier.target,
                  right.event.source, right.event.target);
}

enum class RepairFrontierBuildStatus : std::uint8_t {
  Complete,
  Truncated,
  Failed,
};

struct RepairFrontierBuildResult {
  RepairFrontierBuildStatus status = RepairFrontierBuildStatus::Complete;
  std::size_t inspections = 0;
  std::size_t proposals = 0;
};

struct RepairFrontierMetadata {
  std::size_t inspections = 0;
  std::size_t proposals = 0;
  bool truncated = false;
};

RepairFrontierBuildResult
addRepairFrontierPatterns(const CanonicalSyncProgram &program,
                          CanonicalSyncPatternProblem &problem,
                          ArrayRef<DirectEventRecord> directEvents,
                          ArrayRef<CanonicalSyncMechanismId> conflictCore,
                          const CanonicalSyncPatternOptions &options) {
  const SyncCoverGraph &graph = program.getGraph();
  RepairFrontierBuildResult result;
  std::vector<CanonicalSyncMechanismId> sortedCore(conflictCore.begin(),
                                                   conflictCore.end());
  llvm::sort(sortedCore);
  sortedCore.erase(std::unique(sortedCore.begin(), sortedCore.end()),
                   sortedCore.end());
  std::vector<DirectEventRecord> liveEvents;
  llvm::copy_if(directEvents, std::back_inserter(liveEvents),
                [&](const DirectEventRecord &event) {
                  return std::binary_search(sortedCore.begin(),
                                            sortedCore.end(), event.mechanism);
                });
  const auto less = [](const RepairFrontierProposal &left,
                       const RepairFrontierProposal &right) {
    return frontierProposalLess(left, right);
  };
  std::set<RepairFrontierProposal, decltype(less)> proposals(less);
  for (std::size_t first = 0; first < liveEvents.size(); ++first) {
    const SyncCoverDemand &firstDemand =
        graph.getDemands()[liveEvents[first].demand];
    for (std::size_t second = first + 1; second < liveEvents.size(); ++second) {
      if (liveEvents[first].mechanism == liveEvents[second].mechanism) {
        continue;
      }
      if (result.inspections == options.maximumRepairFrontierInspections) {
        result.status = RepairFrontierBuildStatus::Truncated;
        result.proposals = proposals.size();
        return result;
      }
      ++result.inspections;
      const SyncCoverDemand &secondDemand =
          graph.getDemands()[liveEvents[second].demand];
      const bool compatible =
          liveEvents[first].domain == liveEvents[second].domain &&
          firstDemand.scope == secondDemand.scope &&
          firstDemand.sourceGuard.literals.empty() &&
          firstDemand.targetGuard.literals.empty() &&
          secondDemand.sourceGuard.literals.empty() &&
          secondDemand.targetGuard.literals.empty();
      if (!compatible) {
        continue;
      }
      const SyncCoverNode &firstSource = graph.getNodes()[firstDemand.source];
      const SyncCoverNode &secondSource = graph.getNodes()[secondDemand.source];
      const SyncCoverNode &firstTarget = graph.getNodes()[firstDemand.target];
      const SyncCoverNode &secondTarget = graph.getNodes()[secondDemand.target];
      const SyncCoverNode &earlySource =
          firstSource.order < secondSource.order ? firstSource : secondSource;
      const SyncCoverNode &lateSource =
          firstSource.order < secondSource.order ? secondSource : firstSource;
      const SyncCoverNode &earlyTarget =
          firstTarget.order < secondTarget.order ? firstTarget : secondTarget;
      const bool distinctSources = earlySource.id != lateSource.id;
      const bool forwardFrontier = lateSource.order < earlyTarget.order;
      if (!distinctSources || !forwardFrontier) {
        continue;
      }
      SyncCoverEdge barrier{earlySource.id,
                            lateSource.id,
                            SyncCoverEdgeKind::CompletionSupply,
                            firstDemand.scope,
                            0,
                            {},
                            {}};
      SyncCoverEdge event{lateSource.id,
                          earlyTarget.id,
                          SyncCoverEdgeKind::CompletionSupply,
                          firstDemand.scope,
                          0,
                          {},
                          {}};
      const bool invalidFrontier =
          !syncCoverEndpointsCoExecute(graph, barrier) ||
          !syncCoverEndpointsCoExecute(graph, event) ||
          !syncCoverNodeCanProduceCompletion(graph, lateSource.id,
                                             earlyTarget.resource);
      if (invalidFrontier) {
        continue;
      }
      RepairFrontierProposal proposal{barrier, event, liveEvents[first].domain};
      const auto insertion = proposals.lower_bound(proposal);
      const bool duplicate =
          insertion != proposals.end() && !less(proposal, *insertion);
      if (duplicate) {
        continue;
      }
      const bool proposalLimitReached =
          proposals.size() == options.maximumRepairFrontierProposals;
      if (proposalLimitReached) {
        result.status = RepairFrontierBuildStatus::Truncated;
        result.proposals = proposals.size();
        return result;
      }
      proposals.emplace_hint(insertion, std::move(proposal));
    }
  }
  result.proposals = proposals.size();
  for (const RepairFrontierProposal &proposal : proposals) {
    const CanonicalSyncProblemResult barrier =
        problem.internMechanism(makeRepairBarrier(graph, proposal.barrier));
    const CanonicalSyncProblemResult event = problem.internMechanism(
        makeRepairEvent(graph, proposal.event, proposal.domain));
    const bool barrierSemanticFailure =
        !barrier && barrier.error != CanonicalSyncProblemError::LimitExceeded;
    const bool eventSemanticFailure =
        !event && event.error != CanonicalSyncProblemError::LimitExceeded;
    if (barrierSemanticFailure || eventSemanticFailure) {
      result.status = RepairFrontierBuildStatus::Failed;
      program.getFunction().emitError(
          "cannot add canonical sync repair-frontier mechanisms");
      return result;
    }
    const bool limitExceeded =
        barrier.error == CanonicalSyncProblemError::LimitExceeded ||
        event.error == CanonicalSyncProblemError::LimitExceeded;
    if (limitExceeded) {
      result.status = RepairFrontierBuildStatus::Truncated;
      return result;
    }
    if (!barrier || !event || !barrier.index || !event.index) {
      result.status = RepairFrontierBuildStatus::Failed;
      program.getFunction().emitError(
          "cannot add canonical sync repair-frontier mechanisms");
      return result;
    }
    const CanonicalSyncProblemResult pattern = addCanonicalSyncFeasiblePattern(
        problem, {CanonicalSyncPatternKind::RepairFrontier,
                  {*barrier.index, *event.index}});
    if (pattern.error == CanonicalSyncProblemError::LimitExceeded) {
      result.status = RepairFrontierBuildStatus::Truncated;
      return result;
    }
    if (!pattern) {
      result.status = RepairFrontierBuildStatus::Failed;
      program.getFunction().emitError(
          "cannot add canonical sync repair-frontier pattern");
      return result;
    }
  }
  return result;
}

LogicalResult addDirectPairPatterns(const CanonicalSyncProgram &program,
                                    CanonicalSyncPatternProblem &problem,
                                    CanonicalSyncDirectPairOptions options) {
  const CanonicalSyncProblemResult pairs =
      addCanonicalSyncDirectPairPatterns(problem, options);
  if (!pairs) {
    return program.getFunction().emitError(
        "cannot add canonical sync direct-pair patterns");
  }
  return success();
}

struct RepairCriticalDemandResult {
  std::map<CanonicalSyncMechanismId, std::vector<SyncCoverDemandId>> demands;
  bool workExceeded = false;
};

RepairCriticalDemandResult
findRepairCriticalDemands(const CanonicalSyncPatternProblem &preciseProblem,
                          ArrayRef<CanonicalSyncMechanismId> conflictCore,
                          ArrayRef<CanonicalSyncMechanismId> selectedMechanisms,
                          std::size_t maximumWork) {
  RepairCriticalDemandResult result;
  std::size_t work = 0;
  const auto consume = [&](std::size_t amount) {
    if (work > maximumWork || amount > maximumWork - work) {
      result.workExceeded = true;
      return false;
    }
    work += amount;
    return true;
  };
  std::vector<CanonicalSyncMechanismId> selected(selectedMechanisms.begin(),
                                                 selectedMechanisms.end());
  llvm::sort(selected);
  selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
  std::vector<bool> active(preciseProblem.getPatterns().size(), false);
  std::vector<std::size_t> totalCoverage(preciseProblem.getDemands().size(), 0);
  const bool useSelectedPlan = !selected.empty();
  for (const CanonicalSyncPattern &pattern : preciseProblem.getPatterns()) {
    if (!consume(pattern.members.size() + pattern.coverage.getWords().size())) {
      return result;
    }
    const bool patternActive =
        !useSelectedPlan ||
        llvm::all_of(pattern.members, [&](CanonicalSyncMechanismId mechanism) {
          return std::binary_search(selected.begin(), selected.end(),
                                    mechanism);
        });
    if (!patternActive) {
      continue;
    }
    active[pattern.id] = true;
    for (std::size_t demand = 0; demand < preciseProblem.getDemands().size();
         ++demand) {
      if (!pattern.coverage.contains(demand)) {
        continue;
      }
      if (!consume(1)) {
        return result;
      }
      ++totalCoverage[demand];
    }
  }

  std::vector<CanonicalSyncMechanismId> owners(conflictCore.begin(),
                                               conflictCore.end());
  llvm::sort(owners);
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
  for (CanonicalSyncMechanismId owner : owners) {
    if (owner >= preciseProblem.getMechanismPatterns().size()) {
      result.workExceeded = true;
      return result;
    }
    std::vector<std::size_t> removedCoverage(preciseProblem.getDemands().size(),
                                             0);
    for (CanonicalSyncPatternId patternId :
         preciseProblem.getMechanismPatterns()[owner]) {
      if (!consume(1) || patternId >= active.size()) {
        result.workExceeded = true;
        return result;
      }
      if (!active[patternId]) {
        continue;
      }
      const CanonicalSyncPattern &pattern =
          preciseProblem.getPatterns()[patternId];
      if (!consume(pattern.coverage.getWords().size())) {
        return result;
      }
      for (std::size_t demand = 0; demand < preciseProblem.getDemands().size();
           ++demand) {
        if (!pattern.coverage.contains(demand)) {
          continue;
        }
        if (!consume(1)) {
          return result;
        }
        ++removedCoverage[demand];
      }
    }
    std::vector<SyncCoverDemandId> &critical = result.demands[owner];
    for (std::size_t demand = 0; demand < preciseProblem.getDemands().size();
         ++demand) {
      if (!consume(1)) {
        return result;
      }
      const bool fixed = preciseProblem.getBaselineCoverage().contains(demand);
      if (!fixed && removedCoverage[demand] != 0 &&
          removedCoverage[demand] == totalCoverage[demand]) {
        critical.push_back(preciseProblem.getDemands()[demand]);
      }
    }
  }
  return result;
}

enum class CandidateCatalogKind : std::uint8_t {
  Precise,
  ConflictCoreRepair,
  LocalizedPipeAll,
};

CanonicalSyncProblemBuildResult buildCandidateCatalog(
    const CanonicalSyncProgram &program,
    const CanonicalSyncBuildOptions &options, CandidateCatalogKind kind,
    ArrayRef<CanonicalSyncMechanismId> conflictCore = {},
    const CanonicalSyncPatternProblem *preciseProblem = nullptr,
    ArrayRef<CanonicalSyncMechanismId> selectedMechanisms = {},
    std::optional<RepairFrontierMetadata> repairMetadata = std::nullopt) {
  if (options.eventIdBudget > kHardwareEventIdCount) {
    program.getFunction().emitError(
        "canonical sync event-id budget must be in [0, 8]");
    return {nullptr, {CanonicalSyncProblemError::InvalidDomain, std::nullopt}};
  }
  const std::vector<SyncCoverDemandId> obligations =
      getActiveDemands(program.getGraph());
  DemandBasisResult basis;
  {
    const SyncCoverExpandedProgram basisExpansion(
        program.getGraph(), obligations, options.expansionLimits);
    basis =
        buildDemandSelectionBasis(program.getGraph(), basisExpansion, options);
  }
  auto problem = std::make_unique<CanonicalSyncPatternProblem>(
      program.getGraph(), obligations, basis.demands, options.problemLimits,
      options.expansionLimits, basis.truncated);
  std::map<CanonicalSyncMechanismId, std::vector<CanonicalSyncMechanismId>>
      repairMechanismsByOwner;
  std::vector<CanonicalSyncMechanismId> collectiveRepairMechanisms;
  const SyncCoverCoverageResult baseline = computeSyncCoverCoverage(
      program.getGraph(), problem->getExpansion(), {}, problem->getDemands());
  if (!baseline) {
    program.getFunction().emitError(
        "cannot compute canonical sync fixed coverage");
    return {nullptr,
            {CanonicalSyncProblemError::CoverageFailure, std::nullopt}};
  }

  bool failedBuild = false;
  if (kind == CandidateCatalogKind::LocalizedPipeAll) {
    failedBuild =
        failed(addPipeAllBackstop(program, *problem, baseline.covered));
  } else {
    std::map<EventDomainKey, CanonicalSyncEventDomainId> domainIds;
    std::vector<DirectEventRecord> directEvents;
    std::vector<ExactEventRecord> exactEvents;
    std::vector<SyncCoverDemandId> sameResourceObligations;
    std::vector<SyncCoverDemandId> uncoveredBasisDemands;
    for (SyncCoverDemandId demandId : problem->getDemands()) {
      if (!baseline.covered.contains(demandId)) {
        uncoveredBasisDemands.push_back(demandId);
      }
    }
    for (SyncCoverDemandId demandId : problem->getDemands()) {
      const SyncCoverDemand &demand = program.getGraph().getDemands()[demandId];
      if (program.getGraph().getNodes()[demand.source].resource ==
          program.getGraph().getNodes()[demand.target].resource) {
        sameResourceObligations.push_back(demandId);
      }
    }
    failedBuild =
        failed(addTargetedBarriers(program, *problem, baseline.covered)) ||
        // Same-pipeline drains may consolidate precise targeted barriers.
        // Cross-pipeline drains are conflict-core repair mechanisms and must
        // not compete with exact events in the normal frozen catalog.
        failed(addSourceLocalPipeDrains(program, *problem, baseline.covered,
                                        sameResourceObligations, false)) ||
        failed(addSourcePrefixPipeDrains(program, *problem, baseline.covered,
                                         options.patterns,
                                         sameResourceObligations, false)) ||
        failed(addEventDomains(program, options.eventIdBudget, *problem,
                               baseline.covered, domainIds)) ||
        failed(addExactEvents(program, *problem, baseline.covered, domainIds,
                              directEvents, exactEvents)) ||
        failed(addExactSlotLifecycleBundles(program, *problem, exactEvents,
                                            options.patterns)) ||
        failed(addCompletionFrontierEvents(program, *problem, baseline.covered,
                                           domainIds, directEvents)) ||
        failed(addTargetCompletionCertificateEvents(
            program, *problem, baseline.covered, domainIds, directEvents)) ||
        failed(addTargetLocalFenceEvents(
            program, *problem, uncoveredBasisDemands, domainIds, false,
            TargetLocalCatalogMode::EventGroupsOnly));
    std::vector<SyncCoverDemandId> sourceLocalResidual;
    if (!failedBuild) {
      SyncCoverDemandSet preciseCovered;
      const CanonicalSyncProblemResult preview =
          problem->previewCoveredDemands(preciseCovered);
      if (!preview) {
        program.getFunction().emitError(
            "cannot preview canonical sync precise catalog coverage");
        failedBuild = true;
      } else {
        for (SyncCoverDemandId demandId : problem->getDemands()) {
          if (!preciseCovered.contains(demandId)) {
            sourceLocalResidual.push_back(demandId);
          }
        }
      }
    }
    if (!failedBuild) {
      failedBuild = failed(addSourceLocalCompletionEvents(
          program, *problem, sourceLocalResidual, domainIds));
    }
    std::vector<SyncCoverDemandId> singletonResidual;
    if (!failedBuild) {
      SyncCoverDemandSet exactCovered;
      const CanonicalSyncProblemResult preview =
          problem->previewCoveredDemands(exactCovered);
      if (!preview) {
        program.getFunction().emitError(
            "cannot preview canonical sync source-local event coverage");
        failedBuild = true;
      } else {
        for (SyncCoverDemandId demandId : problem->getDemands()) {
          if (!exactCovered.contains(demandId)) {
            singletonResidual.push_back(demandId);
          }
        }
      }
    }
    if (!failedBuild && !singletonResidual.empty()) {
      program.getFunction().emitError(
          "canonical sync required singleton catalog is incomplete");
      failedBuild = true;
    }
    if (!failedBuild) {
      failedBuild = failed(addLoopCarryPipeDrains(
          program, *problem, baseline.covered, options.patterns));
    }
    if (!failedBuild) {
      failedBuild = failed(addLoopBoundarySourcePrefixProtocols(
          program, *problem, domainIds, options.patterns));
    }
    if (!failedBuild && options.patterns.enableDirectPairs) {
      failedBuild =
          failed(addDirectPairPatterns(program, *problem, options.directPairs));
    }
    if (!failedBuild && kind == CandidateCatalogKind::ConflictCoreRepair) {
      const bool invalidOwner =
          !preciseProblem || !preciseProblem->isFrozen() ||
          !problem->hasSameCandidatePrefix(*preciseProblem) ||
          llvm::any_of(conflictCore, [&](CanonicalSyncMechanismId mechanism) {
            return mechanism >= problem->getMechanisms().size();
          });
      if (invalidOwner) {
        program.getFunction().emitError(
            "canonical sync repair core does not match the precise catalog");
        return {nullptr,
                {CanonicalSyncProblemError::InvalidPattern, std::nullopt}};
      }
      std::vector<CanonicalSyncMechanismId> sortedCore(conflictCore.begin(),
                                                       conflictCore.end());
      llvm::sort(sortedCore);
      sortedCore.erase(std::unique(sortedCore.begin(), sortedCore.end()),
                       sortedCore.end());
      const RepairCriticalDemandResult critical = findRepairCriticalDemands(
          *preciseProblem, sortedCore, selectedMechanisms,
          options.maximumRepairWorkUnits);
      if (critical.workExceeded) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
      const std::size_t preciseMechanismCount =
          preciseProblem->getMechanisms().size();
      for (const auto &[owner, criticalDemands] : critical.demands) {
        if (failed(addSourceLocalPipeDrains(program, *problem, baseline.covered,
                                            criticalDemands, true)) ||
            failed(addSourcePrefixPipeDrains(program, *problem,
                                             baseline.covered, options.patterns,
                                             criticalDemands, true)) ||
            failed(addTargetLocalFenceEvents(
                program, *problem, criticalDemands, domainIds, false,
                TargetLocalCatalogMode::PipeDrainsOnly))) {
          return {nullptr,
                  {CanonicalSyncProblemError::InvalidMechanism, std::nullopt}};
        }
        std::vector<SyncCoverDemandId> sortedDemands(criticalDemands.begin(),
                                                     criticalDemands.end());
        llvm::sort(sortedDemands);
        for (CanonicalSyncMechanismId mechanism = preciseMechanismCount;
             mechanism < problem->getMechanisms().size(); ++mechanism) {
          const bool owned = llvm::any_of(
              problem->getMechanisms()[mechanism].descriptor.supplies,
              [&](const CanonicalSyncSupplyBinding &binding) {
                return binding.attestedDemand &&
                       std::binary_search(sortedDemands.begin(),
                                          sortedDemands.end(),
                                          *binding.attestedDemand);
              });
          if (owned) {
            repairMechanismsByOwner[owner].push_back(mechanism);
          }
        }
      }
      for (auto &[owner, mechanisms] : repairMechanismsByOwner) {
        (void)owner;
        llvm::sort(mechanisms);
        mechanisms.erase(std::unique(mechanisms.begin(), mechanisms.end()),
                         mechanisms.end());
      }
      const std::size_t frontierMechanismBegin =
          problem->getMechanisms().size();
      const RepairFrontierBuildResult repair = addRepairFrontierPatterns(
          program, *problem, directEvents, conflictCore, options.patterns);
      if (repair.status == RepairFrontierBuildStatus::Failed) {
        return {nullptr,
                {CanonicalSyncProblemError::InvalidMechanism, std::nullopt}};
      }
      if (repair.status == RepairFrontierBuildStatus::Truncated) {
        return buildCandidateCatalog(
            program, options, CandidateCatalogKind::Precise, {}, nullptr, {},
            RepairFrontierMetadata{repair.inspections, repair.proposals, true});
      }
      for (CanonicalSyncMechanismId mechanism = frontierMechanismBegin;
           mechanism < problem->getMechanisms().size(); ++mechanism) {
        collectiveRepairMechanisms.push_back(mechanism);
      }
      const CanonicalSyncProblemResult recorded =
          problem->recordRepairFrontierGeneration(repair.inspections,
                                                  repair.proposals, false);
      if (!recorded) {
        return {nullptr, recorded};
      }
    }
  }
  if (failedBuild) {
    return {nullptr,
            {CanonicalSyncProblemError::InvalidMechanism, std::nullopt}};
  }
  if (problem->wasPatternGenerationTruncated()) {
    program.getFunction().emitRemark(
        "canonical sync pattern generation reached its bounded proposal "
        "limit; singleton candidates remain available");
  }
  if (repairMetadata) {
    const CanonicalSyncProblemResult recorded =
        problem->recordRepairFrontierGeneration(repairMetadata->inspections,
                                                repairMetadata->proposals,
                                                repairMetadata->truncated);
    if (!recorded) {
      return {nullptr, recorded};
    }
  }
  const CanonicalSyncProblemResult frozen = problem->freeze();
  return {std::move(problem), frozen, std::move(repairMechanismsByOwner),
          std::move(collectiveRepairMechanisms)};
}

} // namespace

CanonicalSyncProblemBuildResult mlir::pto::buildCanonicalSyncPreciseProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncBuildOptions &options) {
  return buildCandidateCatalog(program, options, CandidateCatalogKind::Precise);
}

CanonicalSyncProblemBuildResult mlir::pto::buildCanonicalSyncRepairProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncPatternProblem &preciseProblem,
    const CanonicalSyncBuildOptions &options,
    const std::vector<CanonicalSyncMechanismId> &conflictCore,
    const std::vector<CanonicalSyncMechanismId> &selectedMechanisms) {
  return buildCandidateCatalog(
      program, options, CandidateCatalogKind::ConflictCoreRepair, conflictCore,
      &preciseProblem, selectedMechanisms);
}

CanonicalSyncProblemBuildResult mlir::pto::buildCanonicalSyncPipeAllProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncBuildOptions &options) {
  return buildCandidateCatalog(program, options,
                               CandidateCatalogKind::LocalizedPipeAll);
}

FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>>
mlir::pto::buildCanonicalSyncSingletonProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncBuildOptions &options) {
  CanonicalSyncProblemBuildResult built =
      buildCanonicalSyncPreciseProblem(program, options);
  if (!built) {
    program.getFunction().emitError()
        << "cannot freeze canonical sync precise problem, error="
        << static_cast<unsigned>(built.status.error);
    return failure();
  }
  return std::move(built.problem);
}
