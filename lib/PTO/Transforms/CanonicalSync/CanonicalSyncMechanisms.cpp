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
#include <array>
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

constexpr unsigned kCompilerUsableEventIdCount = 6;

constexpr std::uint64_t kConfigurationHashOffset = 1469598103934665603ULL;
constexpr std::uint64_t kConfigurationHashPrime = 1099511628211ULL;

bool consumeProtocolWork(SyncCoverCoverageWorkBudget &budget,
                         std::size_t amount = 1) {
  return budget.consume(amount);
}

bool consumeProtocolProduct(SyncCoverCoverageWorkBudget &budget,
                            std::size_t left, std::size_t right) {
  const bool productOverflows =
      left != 0 && right > std::numeric_limits<std::size_t>::max() / left;
  if (productOverflows) {
    budget.exhausted = true;
    return false;
  }
  return consumeProtocolWork(budget, left * right);
}

bool checkedProtocolAdd(std::size_t &total, std::size_t amount,
                        SyncCoverCoverageWorkBudget &budget) {
  const bool sumOverflows =
      amount > std::numeric_limits<std::size_t>::max() - total;
  if (sumOverflows) {
    budget.exhausted = true;
    return false;
  }
  total += amount;
  return true;
}

bool consumeProtocolTripleProduct(SyncCoverCoverageWorkBudget &budget,
                                  std::size_t first, std::size_t second,
                                  std::size_t third) {
  const bool partialProductOverflows =
      first != 0 && second > std::numeric_limits<std::size_t>::max() / first;
  if (partialProductOverflows) {
    budget.exhausted = true;
    return false;
  }
  return consumeProtocolProduct(budget, first * second, third);
}

CanonicalSyncProblemError
protocolVerificationResult(SyncCoverCoverageWorkBudget &budget, bool verified) {
  if (budget.exhausted) {
    return CanonicalSyncProblemError::LimitExceeded;
  }
  return verified ? CanonicalSyncProblemError::None
                  : CanonicalSyncProblemError::UnverifiedProtocol;
}

void hashConfigurationValue(std::uint64_t &hash, std::uint64_t value) {
  for (unsigned byte = 0; byte < sizeof(value); ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffU;
    hash *= kConfigurationHashPrime;
  }
}

std::uint64_t
getCandidateConfigurationSignature(const CanonicalSyncBuildOptions &options) {
  constexpr CanonicalSyncMechanismFamilyMask preciseFamilies =
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::CompletionFrontier) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::TargetCompletionCertificate) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::TargetLocalFence) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::SourceLocalCompletion) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::SourceLocalDrain) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::SourcePrefixDrain) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::LoopCarryDrain) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::LoopBoundaryProtocol) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::L0OperandOwnership) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::BasicOwnership) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::BoundaryOwnership) |
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::HierarchicalOwnership);
  std::uint64_t hash = kConfigurationHashOffset;
  const auto add = [&](std::size_t value) {
    hashConfigurationValue(hash, static_cast<std::uint64_t>(value));
  };
  add(options.eventIdBudget);
  const CanonicalSyncCatalogMode candidateCatalogMode =
      options.patterns.catalogMode == CanonicalSyncCatalogMode::Standard
          ? CanonicalSyncCatalogMode::Standard
          : CanonicalSyncCatalogMode::StrictMinimalDirect;
  add(static_cast<std::size_t>(candidateCatalogMode));
  add(options.patterns.enabledMechanismFamilies & preciseFamilies);
  add(options.patterns.enableDirectPairs);
  add(options.patterns.maximumSourcePrefixInspections);
  add(options.patterns.maximumSourcePrefixCandidates);
  add(options.patterns.maximumSourcePrefixIncidences);
  add(options.patterns.maximumLoopCarryInspections);
  add(options.patterns.maximumLoopCarryCandidates);
  add(options.patterns.maximumLoopCarryIncidences);
  add(options.patterns.maximumLoopBoundaryProtocolInspections);
  add(options.patterns.maximumLoopBoundaryProtocolCandidates);
  add(options.patterns.maximumLoopBoundaryProtocolIncidences);
  add(options.directPairs.maximumEvaluationsPerScope);
  add(options.directPairs.maximumConnectorIndexEntries);
  add(options.directPairs.maximumConnectorInspections);
  add(options.directPairs.pairCoverageLimits.maximumWorkspaceWords);
  add(options.directPairs.pairCoverageLimits.maximumResultWords);
  add(options.directPairs.pairCoverageLimits.maximumResultRows);
  add(options.directPairs.pairCoverageLimits.maximumMechanismRows);
  add(options.directPairs.pairCoverageLimits.maximumTotalWords);
  add(options.directPairs.maximumPreparationWords);
  add(options.problemLimits.maximumDomains);
  add(options.problemLimits.maximumEventBudget);
  add(options.problemLimits.maximumReservedEventIds);
  add(options.problemLimits.maximumMechanisms);
  add(options.problemLimits.maximumPatterns);
  add(options.problemLimits.maximumPatternProposals);
  add(options.problemLimits.maximumActionsPerMechanism);
  add(options.problemLimits.maximumDrainedResourcesPerBarrier);
  add(options.problemLimits.maximumEventUsesPerMechanism);
  add(options.problemLimits.maximumSuppliesPerMechanism);
  add(options.problemLimits.maximumTotalActions);
  add(options.problemLimits.maximumTotalEventUses);
  add(options.problemLimits.maximumTotalSupplies);
  add(options.problemLimits.maximumMembersPerPattern);
  add(options.problemLimits.maximumIncidences);
  add(options.problemLimits.maximumSingletonCoverageWords);
  add(options.problemLimits.maximumCoverageWords);
  add(options.expansionLimits.maximumArenaNodes);
  add(options.expansionLimits.maximumArenaEdges);
  add(options.expansionLimits.maximumTotalNodes);
  add(options.expansionLimits.maximumTotalEdges);
  add(options.enableDemandBasisReduction);
  add(options.maximumDemandBasisGroupEdges);
  add(options.maximumDemandBasisReachabilityWords);
  add(options.maximumDemandBasisReductionWork);
  return hash;
}

using EventDomainKey = std::pair<std::uint32_t, std::uint32_t>;

struct BarrierFallbackGroup {
  std::vector<SyncCoverDemandId> demands;
};

struct DirectEventRecord {
  SyncCoverDemandId demand = 0;
  CanonicalSyncMechanismId mechanism = 0;
  CanonicalSyncEventDomainId domain = 0;
};

LogicalResult recordDirectOnlyDemand(const CanonicalSyncProgram &program,
                                     SyncCoverDemandId demand,
                                     SyncCoverDemandSet *admittedDemands,
                                     SyncCoverCoverageWorkBudget *workBudget) {
  if (!admittedDemands) {
    return success();
  }
  if (!workBudget || !workBudget->consume() ||
      !admittedDemands->insert(demand)) {
    return program.getFunction().emitError(
        "canonical sync direct-only admission work limit exceeded");
  }
  return success();
}

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

bool checkedOrderedLookupWork(std::size_t entries, std::size_t &result) {
  std::size_t levels = 1;
  for (std::size_t remaining = entries; remaining > 1;
       remaining = remaining / 2 + remaining % 2) {
    ++levels;
  }
  return checkedBasisProduct(levels, 2, result) &&
         checkedBasisSum(result, 2, result);
}

template <typename T>
bool stableSortAndUniqueRepairValues(std::vector<T> &values,
                                     SyncCoverCoverageWorkBudget *workBudget) {
  const bool needsSorting = values.size() > 1;
  if (needsSorting) {
    if (workBudget && !workBudget->consume(values.size())) {
      return false;
    }
    std::vector<T> scratch(values.size());
    for (std::size_t width = 1; width < values.size();) {
      if (workBudget && !workBudget->consume(values.size())) {
        return false;
      }
      for (std::size_t begin = 0; begin < values.size();) {
        const std::size_t middle =
            begin + std::min(width, values.size() - begin);
        const std::size_t end =
            middle + std::min(width, values.size() - middle);
        std::size_t left = begin;
        std::size_t right = middle;
        std::size_t output = begin;
        while (left < middle && right < end) {
          if (workBudget && !workBudget->consume()) {
            return false;
          }
          if (values[right] < values[left]) {
            scratch[output++] = values[right++];
          } else {
            scratch[output++] = values[left++];
          }
        }
        while (left < middle) {
          scratch[output++] = values[left++];
        }
        while (right < end) {
          scratch[output++] = values[right++];
        }
        begin = end;
      }
      values.swap(scratch);
      const std::size_t remainingWidth = values.size() - width;
      if (width >= remainingWidth) {
        break;
      }
      width *= 2;
    }
  }
  if (workBudget && !workBudget->consume(values.size())) {
    return false;
  }
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return true;
}

template <typename T>
std::optional<bool>
meteredRepairContains(ArrayRef<T> values, const T &value,
                      SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t begin = 0;
  std::size_t end = values.size();
  while (begin < end) {
    if (workBudget && !workBudget->consume()) {
      return std::nullopt;
    }
    const std::size_t middle = begin + (end - begin) / 2;
    if (values[middle] < value) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (workBudget && !workBudget->consume()) {
    return std::nullopt;
  }
  return begin != values.size() && values[begin] == value;
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

bool canUseStandaloneTargetedBarrier(const SyncCoverGraph &graph,
                                     std::uint32_t sourceResource,
                                     std::uint32_t targetResource) {
  if (!graph.supportsBlockingTargetedBarrier(sourceResource)) {
    return false;
  }
  return sourceResource == targetResource ||
         graph.supportsCrossResourceTargetedBarrier(sourceResource,
                                                    targetResource);
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
      graph.supportsBlockingTargetedBarrier(source.resource);
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
    const CanonicalSyncProblemResult added = problem.internMechanism(
        makeSourceLocalCompletionEvent(graph, std::get<0>(key), groupedDemands),
        CanonicalSyncMechanismOrigin::SourceLocalCompletionEvent);
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
makeDirectEvent(const SyncCoverGraph &graph, SyncCoverDemandId demandId,
                CanonicalSyncEventDomainId domain, bool attestDemand) {
  const SyncCoverDemand &demand = graph.getDemands()[demandId];
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
  if (attestDemand) {
    binding.attestedDemand = demandId;
  }
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
  descriptor.kind = CanonicalSyncMechanismKind::Event;
  descriptor.eventUses.push_back({domain, 1, std::nullopt});
  if (needsBarrier) {
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::Barrier,
         source.resource,
         {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
         std::nullopt,
         0,
         {source.resource},
         CanonicalSyncBarrierKind::Targeted});
  }
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
  for (SyncCoverDemandId demandId : demandIds) {
    CanonicalSyncSupplyBinding binding;
    binding.edge = getDemandEdge(graph.getDemands()[demandId]);
    binding.eventUse = 0;
    binding.proof = CanonicalSyncSupplyProof::TargetLocalFenceAction;
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
      descriptor.kind == CanonicalSyncMechanismKind::Event &&
      descriptor.eventUses.size() == 1 &&
      descriptor.actions.size() == (needsBarrier ? 3 : 2);
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
               graph.getNodes()[candidate.target].resource == target.resource;
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
               binding.eventUse == 0 && !binding.barrierAction &&
               !binding.produceAction && !binding.consumeAction &&
               binding.proof ==
                   CanonicalSyncSupplyProof::TargetLocalFenceAction &&
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
  const CanonicalSyncEventUse &use = descriptor.eventUses.front();
  const std::size_t setAction = needsBarrier ? 1 : 0;
  const std::size_t waitAction = needsBarrier ? 2 : 1;
  return use.domain == domain && use.width == 1 && !use.recurrenceScope &&
         !use.lifetimeScope &&
         (!needsBarrier || barrierActionMatches(descriptor.actions[0])) &&
         eventActionMatches(descriptor.actions[setAction],
                            CanonicalSyncActionKind::EventSet,
                            source.resource) &&
         eventActionMatches(descriptor.actions[waitAction],
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
        std::move(descriptor),
        [demandIds](const auto &verified, SyncCoverCoverageWorkBudget &work) {
          const bool workAvailable =
              consumeProtocolWork(work, 3) &&
              consumeProtocolWork(work, verified.supplies.size()) &&
              consumeProtocolProduct(work, verified.supplies.size(),
                                     demandIds.size());
          if (!workAvailable) {
            return CanonicalSyncProblemError::LimitExceeded;
          }
          const bool valid =
              verified.kind == CanonicalSyncMechanismKind::Protocol &&
              verified.eventUses.size() == 1 &&
              verified.supplies.size() == demandIds.size() &&
              llvm::all_of(verified.supplies,
                           [&](const CanonicalSyncSupplyBinding &binding) {
                             return binding.proof ==
                                        CanonicalSyncSupplyProof::
                                            LoopBoundarySourcePrefixProtocol &&
                                    binding.attestedDemand &&
                                    llvm::is_contained(demandIds,
                                                       *binding.attestedDemand);
                           });
          return protocolVerificationResult(work, valid);
        },
        CanonicalSyncMechanismOrigin::LoopBoundarySourcePrefixProtocol);
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
            ArrayRef<SyncCoverDemandId> demands, bool broad,
            bool attestDemands = false) {
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
    if (attestDemands) {
      binding.attestedDemand = demandId;
    }
    descriptor.supplies.push_back(std::move(binding));
  }
  return descriptor;
}

bool sameOwnershipEdge(const SyncCoverEdge &left, const SyncCoverEdge &right) {
  return left.source == right.source && left.target == right.target &&
         left.kind == right.kind && left.scope == right.scope &&
         left.distance == right.distance &&
         left.sourceGuard.literals == right.sourceGuard.literals &&
         left.targetGuard.literals == right.targetGuard.literals;
}

CanonicalSyncAction makeOwnershipAction(
    CanonicalSyncActionKind kind, std::uint32_t resource,
    SyncCoverAnchor anchor, std::size_t eventUse, std::size_t lane,
    CanonicalSyncActionGuardKind guard = CanonicalSyncActionGuardKind::None,
    std::optional<SyncCoverScopeId> guardScope = std::nullopt) {
  CanonicalSyncAction result;
  result.kind = kind;
  result.resource = resource;
  result.anchor = anchor;
  result.eventUse = eventUse;
  result.eventLane = lane;
  result.guard = guard;
  result.guardScope = guardScope;
  return result;
}

bool ownershipActionEqual(const CanonicalSyncAction &left,
                          const CanonicalSyncAction &right) {
  return std::tie(left.kind, left.resource, left.anchor.kind, left.anchor.node,
                  left.anchor.scope, left.anchor.position, left.eventUse,
                  left.eventLane, left.drainedResources, left.barrierKind,
                  left.guard, left.guardScope, left.eventLaneKind,
                  left.eventLaneScope) ==
         std::tie(right.kind, right.resource, right.anchor.kind,
                  right.anchor.node, right.anchor.scope, right.anchor.position,
                  right.eventUse, right.eventLane, right.drainedResources,
                  right.barrierKind, right.guard, right.guardScope,
                  right.eventLaneKind, right.eventLaneScope);
}

bool ownershipBindingEqual(const CanonicalSyncSupplyBinding &left,
                           const CanonicalSyncSupplyBinding &right) {
  return sameOwnershipEdge(left.edge, right.edge) &&
         std::tie(left.eventUse, left.barrierAction, left.produceAction,
                  left.consumeAction, left.proof, left.completionExport,
                  left.allowedDemands, left.attestedDemand,
                  left.applicability) ==
             std::tie(right.eventUse, right.barrierAction, right.produceAction,
                      right.consumeAction, right.proof, right.completionExport,
                      right.allowedDemands, right.attestedDemand,
                      right.applicability);
}

bool ownershipDescriptorEqual(const CanonicalSyncMechanismDescriptor &left,
                              const CanonicalSyncMechanismDescriptor &right,
                              SyncCoverCoverageWorkBudget &work) {
  const bool initialWorkUnavailable = !work.consume(left.eventUses.size()) ||
                                      !work.consume(left.actions.size()) ||
                                      !work.consume(left.supplies.size());
  if (initialWorkUnavailable) {
    return false;
  }
  if (left.kind != right.kind ||
      left.eventUses.size() != right.eventUses.size() ||
      left.actions.size() != right.actions.size() ||
      left.supplies.size() != right.supplies.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.eventUses.size(); ++index) {
    const CanonicalSyncEventUse &leftUse = left.eventUses[index];
    const CanonicalSyncEventUse &rightUse = right.eventUses[index];
    if (std::tie(leftUse.domain, leftUse.width, leftUse.recurrenceScope,
                 leftUse.lifetimeScope) !=
        std::tie(rightUse.domain, rightUse.width, rightUse.recurrenceScope,
                 rightUse.lifetimeScope)) {
      return false;
    }
  }
  for (std::size_t index = 0; index < left.actions.size(); ++index) {
    const bool actionWorkUnavailable =
        !work.consume(left.actions[index].drainedResources.size()) ||
        !work.consume(right.actions[index].drainedResources.size());
    if (actionWorkUnavailable) {
      return false;
    }
    if (!ownershipActionEqual(left.actions[index], right.actions[index])) {
      return false;
    }
  }
  std::vector<bool> matched(right.supplies.size(), false);
  for (const CanonicalSyncSupplyBinding &binding : left.supplies) {
    std::optional<std::size_t> found;
    for (std::size_t index = 0; index < right.supplies.size(); ++index) {
      const CanonicalSyncSupplyBinding &candidate = right.supplies[index];
      std::size_t comparisonWork = 1;
      const bool workOverflows =
          !checkedProtocolAdd(comparisonWork,
                              binding.edge.sourceGuard.literals.size(), work) ||
          !checkedProtocolAdd(comparisonWork,
                              binding.edge.targetGuard.literals.size(), work) ||
          !checkedProtocolAdd(comparisonWork, binding.allowedDemands.size(),
                              work) ||
          !checkedProtocolAdd(comparisonWork,
                              candidate.edge.sourceGuard.literals.size(),
                              work) ||
          !checkedProtocolAdd(comparisonWork,
                              candidate.edge.targetGuard.literals.size(),
                              work) ||
          !checkedProtocolAdd(comparisonWork, candidate.allowedDemands.size(),
                              work);
      if (workOverflows || !work.consume(comparisonWork)) {
        return false;
      }
      if (!matched[index] && ownershipBindingEqual(binding, candidate)) {
        found = index;
        break;
      }
    }
    if (!found) {
      return false;
    }
    matched[*found] = true;
  }
  return true;
}

bool reserveOwnershipFactoryWork(
    const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    SyncCoverCoverageWorkBudget &work) {
  // buildOwnershipDemandIndex initializes one entry for every graph access and
  // node, then performs a second all-node pass to normalize ready regions.
  // Reserve these independent dimensions before either allocation occurs.
  const bool graphWorkAvailable =
      consumeProtocolWork(work, graph.getStorageAccesses().size()) &&
      consumeProtocolWork(work, graph.getNodes().size()) &&
      consumeProtocolWork(work, graph.getNodes().size());
  if (!graphWorkAvailable) {
    return false;
  }
  std::size_t certificateIncidences = 1;
  const auto addIncidences = [&](std::size_t amount) {
    return checkedProtocolAdd(certificateIncidences, amount, work) &&
           consumeProtocolWork(work, amount == 0 ? 1 : amount);
  };
  const bool headerWorkUnavailable =
      !addIncidences(certificate.lanes.size()) ||
      !addIncidences(certificate.paths.size()) ||
      !addIncidences(certificate.initialProducers.size()) ||
      !addIncidences(certificate.initiallyFreeLanes.size());
  if (headerWorkUnavailable) {
    return false;
  }
  for (const SyncCoverBasicOwnershipLane &lane : certificate.lanes) {
    if (!addIncidences(lane.slots.size())) {
      return false;
    }
    for (const SyncCoverBasicOwnershipSlot &slot : lane.slots) {
      if (!addIncidences(slot.accesses.size())) {
        return false;
      }
    }
  }
  for (const SyncCoverBasicOwnershipPath &path : certificate.paths) {
    const std::size_t guardIncidences =
        path.scope < graph.getScopes().size()
            ? graph.getScopes()[path.scope].guard.literals.size()
            : 0;
    const bool pathWorkAvailable =
        addIncidences(path.uses.size()) && addIncidences(guardIncidences);
    if (!pathWorkAvailable) {
      return false;
    }
    for (const SyncCoverBasicOwnershipUse &use : path.uses) {
      const bool useWorkUnavailable = !addIncidences(use.producers.size()) ||
                                      !addIncidences(use.consumers.size());
      if (useWorkUnavailable) {
        return false;
      }
    }
  }

  std::size_t useCount = 0;
  for (const SyncCoverBasicOwnershipPath &path : certificate.paths) {
    if (!checkedProtocolAdd(useCount, path.uses.size(), work)) {
      return false;
    }
  }
  // Certificate-local setup consists of container insertion/sorting bounded
  // by the squared incidence census, plus path/lane/use scans. Producer and
  // consumer cross products end in addOwnershipActionBindings and are charged
  // by that function, including their complete demand and supply scans.
  for (unsigned factor = 0; factor < 8; ++factor) {
    if (!consumeProtocolProduct(work, certificateIncidences,
                                certificateIncidences) ||
        !consumeProtocolTripleProduct(work, certificate.paths.size(),
                                      certificate.lanes.size(), useCount)) {
      return false;
    }
  }
  std::size_t witnessIncidences = 0;
  if (!consumeProtocolWork(work, graph.getDemands().size())) {
    return false;
  }
  for (const SyncCoverDemand &demand : graph.getDemands()) {
    if (!checkedProtocolAdd(witnessIncidences, demand.storageWitnesses.size(),
                            work)) {
      return false;
    }
  }
  std::size_t perDemand = 0;
  const bool demandWorkOverflows =
      !checkedProtocolAdd(perDemand, graph.getScopes().size(), work) ||
      !checkedProtocolAdd(perDemand, 1, work) ||
      !checkedProtocolAdd(perDemand, certificateIncidences, work);
  if (demandWorkOverflows) {
    return false;
  }
  return consumeProtocolProduct(work, graph.getDemands().size(), perDemand) &&
         consumeProtocolProduct(work, witnessIncidences, perDemand);
}

struct BasicOwnershipDemandIndex {
  std::vector<std::optional<std::size_t>> accessLane;
  std::vector<std::set<SyncCoverNodeId>> producers;
  std::vector<std::set<SyncCoverNodeId>> consumers;
  std::vector<std::vector<std::size_t>> producerReadyRegions;
};

BasicOwnershipDemandIndex buildOwnershipDemandIndex(
    const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate) {
  BasicOwnershipDemandIndex result;
  result.accessLane.resize(graph.getStorageAccesses().size());
  result.producers.resize(certificate.lanes.size());
  result.consumers.resize(certificate.lanes.size());
  result.producerReadyRegions.resize(graph.getNodes().size());
  for (const SyncCoverBasicOwnershipLane &lane : certificate.lanes) {
    for (const SyncCoverBasicOwnershipSlot &slot : lane.slots) {
      for (SyncCoverStorageAccessId access : slot.accesses) {
        if (access < result.accessLane.size()) {
          result.accessLane[access] = lane.id;
        }
      }
    }
  }
  std::size_t nextReadyRegion = 0;
  const auto recordReadyRegion = [&](ArrayRef<SyncCoverNodeId> producers) {
    const std::size_t region = nextReadyRegion++;
    for (SyncCoverNodeId producer : producers) {
      if (producer < result.producerReadyRegions.size()) {
        result.producerReadyRegions[producer].push_back(region);
      }
    }
  };
  for (const SyncCoverBasicOwnershipPath &path : certificate.paths) {
    for (const SyncCoverBasicOwnershipUse &use : path.uses) {
      result.producers[use.producerLane].insert(use.producers.begin(),
                                                use.producers.end());
      result.consumers[use.lane].insert(use.consumers.begin(),
                                        use.consumers.end());
      recordReadyRegion(use.producers);
    }
  }
  result.producers[certificate.initialReadyLane].insert(
      certificate.initialProducers.begin(), certificate.initialProducers.end());
  recordReadyRegion(certificate.initialProducers);
  for (std::vector<std::size_t> &regions : result.producerReadyRegions) {
    llvm::sort(regions);
    regions.erase(std::unique(regions.begin(), regions.end()), regions.end());
  }
  return result;
}

bool isOwnershipMemoryDemand(const SyncCoverDemand &demand) {
  // The target's ACC read/read exception requires the same completed-before
  // relation as the exact ownership lifecycle.  It is not an ordinary memory
  // dependence, but a verified ownership protocol may discharge it when the
  // overlap witness belongs to the protocol's physical lane.
  return !demand.provenanceKinds.empty() &&
         llvm::all_of(demand.provenanceKinds, [](SyncCoverDemandKind kind) {
           return kind == SyncCoverDemandKind::MemoryRAW ||
                  kind == SyncCoverDemandKind::MemoryWAR ||
                  kind == SyncCoverDemandKind::MemoryWAW ||
                  kind == SyncCoverDemandKind::HardwareAccRAR;
         });
}

bool nodeOwnsLane(const BasicOwnershipDemandIndex &index, SyncCoverNodeId node,
                  std::size_t lane, bool producer) {
  const auto &nodes = producer ? index.producers : index.consumers;
  return lane < nodes.size() && nodes[lane].count(node) != 0;
}

bool producersShareReadyRegion(const BasicOwnershipDemandIndex &index,
                               SyncCoverNodeId first, SyncCoverNodeId second) {
  if (first >= index.producerReadyRegions.size() ||
      second >= index.producerReadyRegions.size()) {
    return false;
  }
  const std::vector<std::size_t> &left = index.producerReadyRegions[first];
  const std::vector<std::size_t> &right = index.producerReadyRegions[second];
  auto leftIt = left.begin();
  auto rightIt = right.begin();
  while (leftIt != left.end() && rightIt != right.end()) {
    if (*leftIt == *rightIt) {
      return true;
    }
    if (*leftIt < *rightIt) {
      ++leftIt;
    } else {
      ++rightIt;
    }
  }
  return false;
}

bool demandBelongsToOwnership(
    const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    const BasicOwnershipDemandIndex &index, const SyncCoverDemand &demand) {
  if (!isOwnershipMemoryDemand(demand) || demand.storageWitnesses.empty()) {
    return false;
  }
  // The lifecycle is local, but its balanced per-lane scope-exit drains form a
  // completion summary for enclosing loops. No sibling or descendant arena
  // may consume that summary.
  if (demand.distance != 0 && demand.scope != certificate.loopScope &&
      !graph.scopeContains(demand.scope, certificate.loopScope)) {
    return false;
  }
  return llvm::all_of(
      demand.storageWitnesses, [&](SyncCoverStorageWitnessId witnessId) {
        if (witnessId >= graph.getStorageWitnesses().size()) {
          return false;
        }
        const SyncCoverStorageWitness &witness =
            graph.getStorageWitnesses()[witnessId];
        if (witness.sourceAccess >= index.accessLane.size() ||
            witness.targetAccess >= index.accessLane.size() ||
            !index.accessLane[witness.sourceAccess] ||
            !index.accessLane[witness.targetAccess] ||
            index.accessLane[witness.sourceAccess] !=
                index.accessLane[witness.targetAccess]) {
          return false;
        }
        const std::size_t lane = *index.accessLane[witness.sourceAccess];
        const SyncCoverStorageAccess &source =
            graph.getStorageAccesses()[witness.sourceAccess];
        const SyncCoverStorageAccess &target =
            graph.getStorageAccesses()[witness.targetAccess];
        if (!source.exactPhysical || !target.exactPhysical ||
            source.node != demand.source || target.node != demand.target) {
          return false;
        }
        const bool sourceProducer =
            syncCoverStorageModeWrites(source.mode) &&
            nodeOwnsLane(index, source.node, lane, true);
        const bool sourceConsumer =
            syncCoverStorageModeReads(source.mode) &&
            nodeOwnsLane(index, source.node, lane, false);
        const bool targetProducer =
            syncCoverStorageModeWrites(target.mode) &&
            nodeOwnsLane(index, target.node, lane, true);
        const bool targetConsumer =
            syncCoverStorageModeReads(target.mode) &&
            nodeOwnsLane(index, target.node, lane, false);
        // The ownership token orders producers in distinct lifecycle uses via
        // an intervening ready/consume/release transfer. It does not order two
        // distance-zero writes grouped behind the same ready set, which is
        // emitted only after both producers have issued. A positive-distance
        // transition crosses the consumer release and is lifecycle-backed.
        return (sourceProducer && targetConsumer) ||
               (sourceConsumer && targetProducer) ||
               (sourceProducer && targetProducer &&
                (demand.distance != 0 ||
                 !producersShareReadyRegion(index, source.node, target.node)));
      });
}

std::vector<SyncCoverDemandId>
getOwnershipDemands(const SyncCoverGraph &graph,
                    const SyncCoverBasicOwnershipCertificate &certificate) {
  const BasicOwnershipDemandIndex index =
      buildOwnershipDemandIndex(graph, certificate);
  std::vector<SyncCoverDemandId> result;
  for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
    if (demandBelongsToOwnership(graph, certificate, index, demand)) {
      result.push_back(demandId);
    }
  }
  return result;
}

using OwnershipBindingKey =
    std::tuple<SyncCoverNodeId, SyncCoverNodeId, unsigned>;

struct OwnershipBindingBucket {
  OwnershipBindingKey key;
  std::vector<SyncCoverDemandId> demands;
};

struct OwnershipBindingIndex {
  std::vector<SyncCoverDemandId> demands;
  std::vector<OwnershipBindingBucket> byEndpoints;
  std::vector<SyncCoverEdge> suppliedEdges;
};

struct OwnershipBindingEntry {
  OwnershipBindingKey key;
  SyncCoverDemandId demand = 0;
};

bool stableSortOwnershipEntries(std::vector<OwnershipBindingEntry> &entries,
                                SyncCoverCoverageWorkBudget *workBudget) {
  const std::size_t entryCount = entries.size();
  if (entryCount < 2) {
    return true;
  }
  if (workBudget && !workBudget->consume(entries.size())) {
    return false;
  }
  std::vector<OwnershipBindingEntry> scratch(entries.size());
  for (std::size_t width = 1; width < entries.size();) {
    // Every entry is moved exactly once in this merge pass. Comparisons are
    // metered immediately below and stop before the next comparison.
    if (workBudget && !workBudget->consume(entries.size())) {
      return false;
    }
    for (std::size_t begin = 0; begin < entries.size();) {
      const std::size_t middle =
          begin + std::min(width, entries.size() - begin);
      const std::size_t end = middle + std::min(width, entries.size() - middle);
      std::size_t left = begin;
      std::size_t right = middle;
      std::size_t output = begin;
      while (left < middle && right < end) {
        if (workBudget && !workBudget->consume()) {
          return false;
        }
        if (entries[right].key < entries[left].key) {
          scratch[output++] = entries[right++];
        } else {
          scratch[output++] = entries[left++];
        }
      }
      while (left < middle) {
        scratch[output++] = entries[left++];
      }
      while (right < end) {
        scratch[output++] = entries[right++];
      }
      begin = end;
    }
    entries.swap(scratch);
    const std::size_t remainingWidth = entries.size() - width;
    if (width >= remainingWidth) {
      break;
    }
    width *= 2;
  }
  return true;
}

std::optional<OwnershipBindingIndex>
buildOwnershipBindingIndex(const SyncCoverGraph &graph,
                           std::vector<SyncCoverDemandId> demands,
                           SyncCoverCoverageWorkBudget *workBudget) {
  if (workBudget && !workBudget->consume(demands.size())) {
    return std::nullopt;
  }
  OwnershipBindingIndex result;
  result.demands = std::move(demands);
  std::vector<OwnershipBindingEntry> entries;
  entries.reserve(result.demands.size());
  for (SyncCoverDemandId demandId : result.demands) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    entries.push_back(
        {{demand.source, demand.target, demand.distance}, demandId});
  }
  const bool indexWorkAvailable =
      stableSortOwnershipEntries(entries, workBudget) &&
      (!workBudget || workBudget->consume(entries.size()));
  if (!indexWorkAvailable) {
    return std::nullopt;
  }
  result.byEndpoints.reserve(entries.size());
  for (const OwnershipBindingEntry &entry : entries) {
    const bool needsBucket = result.byEndpoints.empty() ||
                             result.byEndpoints.back().key != entry.key;
    if (needsBucket) {
      result.byEndpoints.push_back({entry.key, {}});
    }
    result.byEndpoints.back().demands.push_back(entry.demand);
  }
  return result;
}

std::optional<std::size_t>
findOwnershipBindingBucket(const OwnershipBindingIndex &index,
                           const OwnershipBindingKey &key,
                           SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t begin = 0;
  std::size_t end = index.byEndpoints.size();
  while (begin < end) {
    if (workBudget && !workBudget->consume()) {
      return std::nullopt;
    }
    const std::size_t middle = begin + (end - begin) / 2;
    if (index.byEndpoints[middle].key < key) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (begin == index.byEndpoints.size()) {
    return std::nullopt;
  }
  if (workBudget && !workBudget->consume()) {
    return std::nullopt;
  }
  const bool missingBucket = index.byEndpoints[begin].key != key;
  if (missingBucket) {
    return std::nullopt;
  }
  return begin;
}

bool addOwnershipActionBindings(CanonicalSyncMechanismDescriptor &descriptor,
                                const SyncCoverGraph &graph,
                                OwnershipBindingIndex &bindingIndex,
                                SyncCoverNodeId source, SyncCoverNodeId target,
                                unsigned distance, std::size_t eventUse,
                                std::size_t produceAction,
                                std::size_t consumeAction,
                                SyncCoverCoverageWorkBudget *workBudget) {
  const std::optional<std::size_t> bucket = findOwnershipBindingBucket(
      bindingIndex, {source, target, distance}, workBudget);
  if (workBudget && workBudget->exhausted) {
    return false;
  }
  if (!bucket) {
    return true;
  }
  const std::vector<SyncCoverDemandId> &candidates =
      bindingIndex.byEndpoints[*bucket].demands;
  if (workBudget && !workBudget->consume(candidates.size())) {
    return false;
  }
  for (SyncCoverDemandId demandId : candidates) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (demand.distance != 0 &&
        descriptor.eventUses[eventUse].recurrenceScope != demand.scope) {
      const std::optional<SyncCoverScopeId> lifetimeScope =
          descriptor.eventUses[eventUse].lifetimeScope;
      if (!lifetimeScope || *lifetimeScope != demand.scope) {
        continue;
      }
    }
    const SyncCoverEdge edge = getDemandEdge(demand);
    bool alreadySupplied = false;
    for (const SyncCoverEdge &supplied : bindingIndex.suppliedEdges) {
      if (workBudget) {
        std::size_t comparisonWork = 1;
        const bool comparisonOverflows =
            !checkedProtocolAdd(comparisonWork,
                                supplied.sourceGuard.literals.size(),
                                *workBudget) ||
            !checkedProtocolAdd(comparisonWork,
                                supplied.targetGuard.literals.size(),
                                *workBudget) ||
            !checkedProtocolAdd(comparisonWork,
                                edge.sourceGuard.literals.size(),
                                *workBudget) ||
            !checkedProtocolAdd(comparisonWork,
                                edge.targetGuard.literals.size(), *workBudget);
        if (comparisonOverflows || !workBudget->consume(comparisonWork)) {
          return false;
        }
      }
      if (sameOwnershipEdge(supplied, edge)) {
        alreadySupplied = true;
        break;
      }
    }
    if (alreadySupplied) {
      continue;
    }
    CanonicalSyncSupplyBinding binding;
    binding.edge = edge;
    binding.eventUse = eventUse;
    binding.produceAction = produceAction;
    binding.consumeAction = consumeAction;
    binding.proof = CanonicalSyncSupplyProof::VerifiedBasicOwnershipProtocol;
    binding.allowedDemands = {demandId};
    descriptor.supplies.push_back(std::move(binding));
    bindingIndex.suppliedEdges.push_back(edge);
  }
  return true;
}

bool addOwnershipCompositeBindings(CanonicalSyncMechanismDescriptor &descriptor,
                                   const SyncCoverGraph &graph,
                                   ArrayRef<SyncCoverDemandId> ownedDemands,
                                   SyncCoverCoverageWorkBudget *workBudget) {
  if (workBudget && !workBudget->consume(ownedDemands.size())) {
    return false;
  }
  for (SyncCoverDemandId demandId : ownedDemands) {
    const SyncCoverEdge edge = getDemandEdge(graph.getDemands()[demandId]);
    bool alreadySupplied = false;
    for (const CanonicalSyncSupplyBinding &binding : descriptor.supplies) {
      if (workBudget) {
        std::size_t comparisonWork = 1;
        const bool comparisonOverflows =
            !checkedProtocolAdd(comparisonWork,
                                binding.edge.sourceGuard.literals.size(),
                                *workBudget) ||
            !checkedProtocolAdd(comparisonWork,
                                binding.edge.targetGuard.literals.size(),
                                *workBudget) ||
            !checkedProtocolAdd(comparisonWork,
                                edge.sourceGuard.literals.size(),
                                *workBudget) ||
            !checkedProtocolAdd(comparisonWork,
                                edge.targetGuard.literals.size(), *workBudget);
        if (comparisonOverflows || !workBudget->consume(comparisonWork)) {
          return false;
        }
      }
      if (sameOwnershipEdge(binding.edge, edge)) {
        alreadySupplied = true;
        break;
      }
    }
    if (alreadySupplied) {
      continue;
    }
    CanonicalSyncSupplyBinding binding;
    binding.edge = edge;
    binding.proof = CanonicalSyncSupplyProof::VerifiedBasicOwnershipComposite;
    binding.allowedDemands = {demandId};
    descriptor.supplies.push_back(std::move(binding));
  }
  return true;
}

struct OwnershipUseActions {
  std::size_t releaseWait = 0;
  std::vector<std::pair<SyncCoverNodeId, std::size_t>> releaseSets;
};

void buildRoundTripReady(CanonicalSyncMechanismDescriptor &descriptor,
                         const SyncCoverGraph &graph,
                         const SyncCoverBasicOwnershipCertificate &certificate,
                         OwnershipBindingIndex &bindingIndex,
                         std::size_t eventUse,
                         SyncCoverCoverageWorkBudget *workBudget) {
  for (const SyncCoverBasicOwnershipPath &path : certificate.paths) {
    for (const SyncCoverBasicOwnershipUse &use : path.uses) {
      const std::size_t set = descriptor.actions.size();
      descriptor.actions.push_back(makeOwnershipAction(
          CanonicalSyncActionKind::EventSet, certificate.producerResource,
          use.readyAnchor, eventUse, use.lane));
      const std::size_t wait = descriptor.actions.size();
      descriptor.actions.push_back(makeOwnershipAction(
          CanonicalSyncActionKind::EventWait, certificate.consumerResource,
          use.readAcquireAnchor, eventUse, use.lane));
      for (SyncCoverNodeId producer : use.producers) {
        for (SyncCoverNodeId consumer : use.consumers) {
          if (!addOwnershipActionBindings(descriptor, graph, bindingIndex,
                                          producer, consumer, 0, eventUse, set,
                                          wait, workBudget)) {
            return;
          }
        }
      }
    }
  }
}

void buildRoundTripRelease(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    OwnershipBindingIndex &bindingIndex, std::size_t eventUse,
    SyncCoverCoverageWorkBudget *workBudget) {
  for (std::size_t lane = 0; lane < certificate.lanes.size(); ++lane) {
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventSet, certificate.consumerResource,
        {SyncCoverAnchorKind::ScopeEntry, 0, certificate.loopScope, 0},
        eventUse, lane));
  }

  std::vector<std::vector<OwnershipUseActions>> actions(
      certificate.paths.size());
  for (std::size_t pathIndex = 0; pathIndex < certificate.paths.size();
       ++pathIndex) {
    const SyncCoverBasicOwnershipPath &path = certificate.paths[pathIndex];
    std::map<std::size_t, std::size_t> previous;
    for (const SyncCoverBasicOwnershipUse &use : path.uses) {
      OwnershipUseActions useActions;
      useActions.releaseWait = descriptor.actions.size();
      descriptor.actions.push_back(makeOwnershipAction(
          CanonicalSyncActionKind::EventWait, certificate.producerResource,
          use.writeAcquireAnchor, eventUse, use.producerLane));
      const std::size_t set = descriptor.actions.size();
      descriptor.actions.push_back(makeOwnershipAction(
          CanonicalSyncActionKind::EventSet, certificate.consumerResource,
          use.releaseAnchor, eventUse, use.lane));
      for (SyncCoverNodeId consumer : use.consumers) {
        useActions.releaseSets.push_back({consumer, set});
      }
      actions[pathIndex].push_back(std::move(useActions));
      const auto prior = previous.find(use.lane);
      if (prior != previous.end()) {
        const OwnershipUseActions &sourceActions =
            actions[pathIndex][prior->second];
        for (const auto &[consumer, set] : sourceActions.releaseSets) {
          for (SyncCoverNodeId producer : use.producers) {
            if (!addOwnershipActionBindings(
                    descriptor, graph, bindingIndex, consumer, producer, 0,
                    eventUse, set, actions[pathIndex].back().releaseWait,
                    workBudget)) {
              return;
            }
          }
        }
      }
      previous[use.lane] = actions[pathIndex].size() - 1;
    }
  }

  for (std::size_t sourcePath = 0; sourcePath < certificate.paths.size();
       ++sourcePath) {
    for (std::size_t targetPath = 0; targetPath < certificate.paths.size();
         ++targetPath) {
      for (std::size_t lane = 0; lane < certificate.lanes.size(); ++lane) {
        std::optional<std::size_t> sourceUse;
        std::optional<std::size_t> targetUse;
        for (std::size_t index = 0;
             index < certificate.paths[sourcePath].uses.size(); ++index) {
          if (certificate.paths[sourcePath].uses[index].lane == lane) {
            sourceUse = index;
          }
        }
        for (std::size_t index = 0;
             index < certificate.paths[targetPath].uses.size(); ++index) {
          if (certificate.paths[targetPath].uses[index].producerLane == lane) {
            targetUse = index;
            break;
          }
        }
        if (!sourceUse || !targetUse) {
          continue;
        }
        const SyncCoverBasicOwnershipUse &target =
            certificate.paths[targetPath].uses[*targetUse];
        for (const auto &[consumer, set] :
             actions[sourcePath][*sourceUse].releaseSets) {
          for (SyncCoverNodeId producer : target.producers) {
            if (!addOwnershipActionBindings(
                    descriptor, graph, bindingIndex, consumer, producer, 1,
                    eventUse, set, actions[targetPath][*targetUse].releaseWait,
                    workBudget)) {
              return;
            }
          }
        }
      }
    }
  }
  for (std::size_t lane = 0; lane < certificate.lanes.size(); ++lane) {
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventWait, certificate.producerResource,
        {SyncCoverAnchorKind::ScopeExit, 0, certificate.loopScope, 0}, eventUse,
        lane));
  }
}

bool buildBoundaryGuardedRelease(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    OwnershipBindingIndex &bindingIndex, std::size_t eventUse,
    SyncCoverCoverageWorkBudget *workBudget) {
  const bool validShape =
      certificate.kind == SyncCoverBasicOwnershipKind::L0Accumulator &&
      certificate.protocol == SyncCoverBasicOwnershipProtocolKind::RoundTrip &&
      certificate.paths.size() == 1 &&
      certificate.paths.front().scope == certificate.loopScope &&
      certificate.paths.front().uses.size() == certificate.lanes.size() &&
      graph.getScopes()[certificate.loopScope].isLoop;
  if (!validShape) {
    return false;
  }

  std::vector<bool> seen(certificate.lanes.size(), false);
  for (const SyncCoverBasicOwnershipUse &use : certificate.paths.front().uses) {
    if (use.lane >= seen.size() || use.producerLane != use.lane ||
        seen[use.lane]) {
      return false;
    }
    seen[use.lane] = true;
    const std::size_t wait = descriptor.actions.size();
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventWait, certificate.producerResource,
        use.writeAcquireAnchor, eventUse, use.lane,
        CanonicalSyncActionGuardKind::NotFirstIteration,
        certificate.loopScope));
    const std::size_t set = descriptor.actions.size();
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventSet, certificate.consumerResource,
        use.releaseAnchor, eventUse, use.lane,
        CanonicalSyncActionGuardKind::HasSuccessor, certificate.loopScope));
    for (SyncCoverNodeId consumer : use.consumers) {
      for (SyncCoverNodeId producer : use.producers) {
        if (!addOwnershipActionBindings(descriptor, graph, bindingIndex,
                                        consumer, producer, 1, eventUse, set,
                                        wait, workBudget)) {
          return false;
        }
      }
    }
  }
  return llvm::is_contained(seen, false) == false;
}

std::optional<std::size_t> findInitialOwnershipPath(
    const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &alternating) {
  if (!alternating.periodicControl ||
      *alternating.periodicControl >= graph.getControls().size()) {
    return std::nullopt;
  }
  const SyncCoverControl &control =
      graph.getControls()[*alternating.periodicControl];
  if (!control.phaseRelation ||
      control.phaseRelation->initialPhase >=
          control.phaseRelation->activeAlternative.size()) {
    return std::nullopt;
  }
  const unsigned initialAlternative =
      control.phaseRelation
          ->activeAlternative[control.phaseRelation->initialPhase];
  std::optional<std::size_t> result;
  for (auto [pathIndex, path] : llvm::enumerate(alternating.paths)) {
    const SyncCoverGuard &guard = graph.getScopes()[path.scope].guard;
    const bool initial =
        llvm::any_of(guard.literals, [&](const SyncCoverGuardLiteral &literal) {
          return literal.control == *alternating.periodicControl &&
                 literal.alternative == initialAlternative;
        });
    if (initial) {
      if (result) {
        return std::nullopt;
      }
      result = pathIndex;
    }
  }
  return result;
}

bool buildHierarchicalRoundTripRelease(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    OwnershipBindingIndex &bindingIndex, std::size_t eventUse,
    SyncCoverScopeId outerScope, SyncCoverScopeId initialPathScope,
    SyncCoverCoverageWorkBudget *workBudget) {
  if (certificate.protocol != SyncCoverBasicOwnershipProtocolKind::RoundTrip ||
      !graph.getScopes()[outerScope].isLoop ||
      graph.getScopes()[certificate.loopScope].parent != outerScope) {
    return false;
  }
  descriptor.eventUses[eventUse].lifetimeScope = outerScope;
  const std::size_t releaseBegin = descriptor.actions.size();
  buildRoundTripRelease(descriptor, graph, certificate, bindingIndex, eventUse,
                        workBudget);
  if (workBudget && workBudget->exhausted) {
    return false;
  }

  const std::size_t laneCount = certificate.lanes.size();
  if (descriptor.actions.size() - releaseBegin < laneCount * 2) {
    return false;
  }
  for (std::size_t lane = 0; lane < laneCount; ++lane) {
    descriptor.actions[releaseBegin + lane].anchor = {
        SyncCoverAnchorKind::ScopeEntry, 0, outerScope, 0};
    descriptor.actions[descriptor.actions.size() - laneCount + lane].anchor = {
        SyncCoverAnchorKind::ScopeExit, 0, outerScope, 0};
  }

  std::vector<std::vector<OwnershipUseActions>> actions(
      certificate.paths.size());
  std::size_t action = releaseBegin + laneCount;
  std::optional<std::size_t> initialPath;
  for (auto [pathIndex, path] : llvm::enumerate(certificate.paths)) {
    if (path.scope == initialPathScope) {
      if (initialPath) {
        return false;
      }
      initialPath = pathIndex;
    }
    for (const SyncCoverBasicOwnershipUse &use : path.uses) {
      if (action + 1 >= descriptor.actions.size() - laneCount) {
        return false;
      }
      OwnershipUseActions useActions;
      useActions.releaseWait = action++;
      const std::size_t set = action++;
      for (SyncCoverNodeId consumer : use.consumers) {
        useActions.releaseSets.push_back({consumer, set});
      }
      actions[pathIndex].push_back(std::move(useActions));
    }
  }
  if (!initialPath || action != descriptor.actions.size() - laneCount) {
    return false;
  }

  const SyncCoverBasicOwnershipPath &targetPath =
      certificate.paths[*initialPath];
  for (auto [sourcePathIndex, sourcePath] :
       llvm::enumerate(certificate.paths)) {
    for (std::size_t lane = 0; lane < laneCount; ++lane) {
      std::optional<std::size_t> sourceUse;
      std::optional<std::size_t> targetUse;
      for (auto [useIndex, use] : llvm::enumerate(sourcePath.uses)) {
        if (use.lane == lane) {
          sourceUse = useIndex;
        }
      }
      for (auto [useIndex, use] : llvm::enumerate(targetPath.uses)) {
        if (use.producerLane == lane) {
          targetUse = useIndex;
          break;
        }
      }
      if (!sourceUse || !targetUse) {
        return false;
      }
      const SyncCoverBasicOwnershipUse &source = sourcePath.uses[*sourceUse];
      const SyncCoverBasicOwnershipUse &target = targetPath.uses[*targetUse];
      const std::size_t set =
          actions[sourcePathIndex][*sourceUse].releaseSets.front().second;
      const std::size_t wait = actions[*initialPath][*targetUse].releaseWait;
      for (SyncCoverNodeId consumer : source.consumers) {
        for (SyncCoverNodeId producer : target.producers) {
          if (!addOwnershipActionBindings(descriptor, graph, bindingIndex,
                                          consumer, producer, 1, eventUse, set,
                                          wait, workBudget)) {
            return false;
          }
        }
      }
    }
  }
  // The wider lifetime is required even when the flat demand adapter has no
  // separate outer-distance row: the exact ownership certificate proves that
  // these are the complete accesses to the managed slots, and the body token
  // is carried from the final inner invocation to the next outer invocation.
  return true;
}

void buildAlternatingReady(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    OwnershipBindingIndex &bindingIndex, std::size_t eventUse,
    SyncCoverCoverageWorkBudget *workBudget) {
  const std::size_t initialSet = descriptor.actions.size();
  descriptor.actions.push_back(makeOwnershipAction(
      CanonicalSyncActionKind::EventSet, certificate.producerResource,
      certificate.initialReadyAnchor, eventUse, certificate.initialReadyLane,
      CanonicalSyncActionGuardKind::LoopNonEmpty, certificate.loopScope));
  std::vector<std::size_t> waits(certificate.paths.size());
  std::vector<std::size_t> sets(certificate.paths.size());
  for (auto [pathIndex, path] : llvm::enumerate(certificate.paths)) {
    const SyncCoverBasicOwnershipUse &use = path.uses.front();
    waits[pathIndex] = descriptor.actions.size();
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventWait, certificate.consumerResource,
        use.readAcquireAnchor, eventUse, use.lane));
    sets[pathIndex] = descriptor.actions.size();
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventSet, certificate.producerResource,
        use.readyAnchor, eventUse, use.producerLane));
  }
  for (auto [pathIndex, path] : llvm::enumerate(certificate.paths)) {
    const SyncCoverBasicOwnershipUse &use = path.uses.front();
    if (use.lane == certificate.initialReadyLane) {
      for (SyncCoverNodeId producer : certificate.initialProducers) {
        for (SyncCoverNodeId consumer : use.consumers) {
          if (!addOwnershipActionBindings(
                  descriptor, graph, bindingIndex, producer, consumer, 0,
                  eventUse, initialSet, waits[pathIndex], workBudget)) {
            return;
          }
        }
      }
    }
    for (auto [targetIndex, targetPath] : llvm::enumerate(certificate.paths)) {
      const SyncCoverBasicOwnershipUse &target = targetPath.uses.front();
      if (target.lane != use.producerLane) {
        continue;
      }
      for (SyncCoverNodeId producer : use.producers) {
        for (SyncCoverNodeId consumer : target.consumers) {
          if (!addOwnershipActionBindings(
                  descriptor, graph, bindingIndex, producer, consumer, 1,
                  eventUse, sets[pathIndex], waits[targetIndex], workBudget)) {
            return;
          }
        }
      }
    }
  }
}

void buildAlternatingRelease(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    OwnershipBindingIndex &bindingIndex, std::size_t eventUse,
    SyncCoverCoverageWorkBudget *workBudget) {
  for (std::size_t lane : certificate.initiallyFreeLanes) {
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventSet, certificate.consumerResource,
        {SyncCoverAnchorKind::ScopeEntry, 0, certificate.loopScope, 0},
        eventUse, lane, CanonicalSyncActionGuardKind::LoopNonEmpty,
        certificate.loopScope));
  }
  std::vector<std::size_t> sets(certificate.paths.size());
  std::vector<std::size_t> waits(certificate.paths.size());
  for (auto [pathIndex, path] : llvm::enumerate(certificate.paths)) {
    const SyncCoverBasicOwnershipUse &use = path.uses.front();
    sets[pathIndex] = descriptor.actions.size();
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventSet, certificate.consumerResource,
        use.releaseAnchor, eventUse, use.lane));
    waits[pathIndex] = descriptor.actions.size();
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventWait, certificate.producerResource,
        use.writeAcquireAnchor, eventUse, use.producerLane));
  }
  for (auto [pathIndex, path] : llvm::enumerate(certificate.paths)) {
    const SyncCoverBasicOwnershipUse &use = path.uses.front();
    for (auto [targetIndex, targetPath] : llvm::enumerate(certificate.paths)) {
      const SyncCoverBasicOwnershipUse &target = targetPath.uses.front();
      if (target.producerLane != use.lane) {
        continue;
      }
      for (SyncCoverNodeId consumer : use.consumers) {
        for (SyncCoverNodeId producer : target.producers) {
          if (!addOwnershipActionBindings(
                  descriptor, graph, bindingIndex, consumer, producer, 1,
                  eventUse, sets[pathIndex], waits[targetIndex], workBudget)) {
            return;
          }
        }
      }
    }
  }
  for (std::size_t lane = 0; lane < certificate.lanes.size(); ++lane) {
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventWait, certificate.producerResource,
        {SyncCoverAnchorKind::ScopeExit, 0, certificate.loopScope, 0}, eventUse,
        lane, CanonicalSyncActionGuardKind::LoopNonEmpty,
        certificate.loopScope));
  }
}

bool buildHierarchicalAlternatingReady(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    OwnershipBindingIndex &bindingIndex, std::size_t eventUse,
    SyncCoverScopeId outerScope, SyncCoverCoverageWorkBudget *workBudget) {
  if (certificate.protocol !=
          SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch ||
      certificate.lanes.size() != 2 || certificate.paths.size() != 2 ||
      graph.getScopes()[certificate.loopScope].parent != outerScope ||
      !graph.getScopes()[outerScope].isLoop) {
    return false;
  }
  descriptor.eventUses[eventUse].lifetimeScope = outerScope;
  buildAlternatingReady(descriptor, graph, certificate, bindingIndex, eventUse,
                        workBudget);
  if (workBudget && workBudget->exhausted) {
    return false;
  }
  if (descriptor.actions.empty()) {
    return false;
  }
  CanonicalSyncAction &initialSet = descriptor.actions.front();
  if (initialSet.guard != CanonicalSyncActionGuardKind::LoopNonEmpty ||
      initialSet.guardScope != certificate.loopScope) {
    return false;
  }
  initialSet.guard = CanonicalSyncActionGuardKind::None;
  initialSet.guardScope.reset();
  descriptor.actions.push_back(makeOwnershipAction(
      CanonicalSyncActionKind::EventWait, certificate.consumerResource,
      {SyncCoverAnchorKind::ScopeEntry, 0, certificate.loopScope, 0}, eventUse,
      certificate.initialReadyLane, CanonicalSyncActionGuardKind::LoopEmpty,
      certificate.loopScope));
  return true;
}

bool buildHierarchicalAlternatingRelease(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    OwnershipBindingIndex &bindingIndex, std::size_t eventUse,
    SyncCoverScopeId outerScope, SyncCoverCoverageWorkBudget *workBudget) {
  if (certificate.protocol !=
          SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch ||
      certificate.lanes.size() != 2 || certificate.paths.size() != 2 ||
      certificate.initialProducers.empty() ||
      graph.getScopes()[certificate.loopScope].parent != outerScope ||
      !graph.getScopes()[outerScope].isLoop ||
      !findInitialOwnershipPath(graph, certificate)) {
    return false;
  }
  descriptor.eventUses[eventUse].lifetimeScope = outerScope;
  for (std::size_t lane = 0; lane < certificate.lanes.size(); ++lane) {
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventSet, certificate.consumerResource,
        {SyncCoverAnchorKind::ScopeEntry, 0, outerScope, 0}, eventUse, lane));
  }
  const std::size_t initialWait = descriptor.actions.size();
  descriptor.actions.push_back(makeOwnershipAction(
      CanonicalSyncActionKind::EventWait, certificate.producerResource,
      certificate.initialWriteAcquireAnchor, eventUse,
      certificate.initialReadyLane));

  std::vector<std::size_t> sets(certificate.paths.size());
  std::vector<std::size_t> waits(certificate.paths.size());
  for (auto [pathIndex, path] : llvm::enumerate(certificate.paths)) {
    if (path.uses.size() != 1) {
      return false;
    }
    const SyncCoverBasicOwnershipUse &use = path.uses.front();
    sets[pathIndex] = descriptor.actions.size();
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventSet, certificate.consumerResource,
        use.releaseAnchor, eventUse, use.lane));
    waits[pathIndex] = descriptor.actions.size();
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventWait, certificate.producerResource,
        use.writeAcquireAnchor, eventUse, use.producerLane));
  }
  descriptor.actions.push_back(makeOwnershipAction(
      CanonicalSyncActionKind::EventSet, certificate.consumerResource,
      {SyncCoverAnchorKind::ScopeExit, 0, certificate.loopScope, 0}, eventUse,
      certificate.initialReadyLane, CanonicalSyncActionGuardKind::LoopEmpty,
      certificate.loopScope));
  for (std::size_t lane = 0; lane < certificate.lanes.size(); ++lane) {
    descriptor.actions.push_back(makeOwnershipAction(
        CanonicalSyncActionKind::EventWait, certificate.producerResource,
        {SyncCoverAnchorKind::ScopeExit, 0, outerScope, 0}, eventUse, lane));
  }

  for (auto [pathIndex, path] : llvm::enumerate(certificate.paths)) {
    const SyncCoverBasicOwnershipUse &use = path.uses.front();
    for (auto [targetIndex, targetPath] : llvm::enumerate(certificate.paths)) {
      const SyncCoverBasicOwnershipUse &target = targetPath.uses.front();
      if (target.producerLane != use.lane) {
        continue;
      }
      for (SyncCoverNodeId consumer : use.consumers) {
        for (SyncCoverNodeId producer : target.producers) {
          if (!addOwnershipActionBindings(
                  descriptor, graph, bindingIndex, consumer, producer, 1,
                  eventUse, sets[pathIndex], waits[targetIndex], workBudget)) {
            return false;
          }
        }
      }
    }
  }

  const std::size_t suppliesBefore = descriptor.supplies.size();
  const auto initialPath = llvm::find_if(
      certificate.paths, [&](const SyncCoverBasicOwnershipPath &path) {
        return path.uses.front().lane == certificate.initialReadyLane;
      });
  if (initialPath == certificate.paths.end()) {
    return false;
  }
  const std::size_t sourcePath = static_cast<std::size_t>(
      std::distance(certificate.paths.begin(), initialPath));
  for (SyncCoverNodeId consumer : initialPath->uses.front().consumers) {
    for (SyncCoverNodeId producer : certificate.initialProducers) {
      if (!addOwnershipActionBindings(descriptor, graph, bindingIndex, consumer,
                                      producer, 1, eventUse, sets[sourcePath],
                                      initialWait, workBudget)) {
        return false;
      }
    }
  }
  return descriptor.supplies.size() != suppliesBefore;
}

enum class OwnershipRecipe : std::uint8_t {
  Basic,
  BoundaryGuarded,
  Hierarchical,
};

std::optional<CanonicalSyncMechanismDescriptor> makeOwnershipProtocol(
    const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain, OwnershipRecipe recipe,
    std::optional<SyncCoverScopeId> outerScope = std::nullopt,
    std::optional<SyncCoverScopeId> initialPathScope = std::nullopt,
    SyncCoverCoverageWorkBudget *workBudget = nullptr) {
  if (workBudget &&
      !reserveOwnershipFactoryWork(graph, certificate, *workBudget)) {
    return std::nullopt;
  }
  std::optional<OwnershipBindingIndex> bindingIndex =
      buildOwnershipBindingIndex(graph, getOwnershipDemands(graph, certificate),
                                 workBudget);
  if (!bindingIndex || bindingIndex->demands.empty()) {
    return std::nullopt;
  }
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Protocol;
  descriptor.eventUses.push_back(
      {readyDomain, certificate.lanes.size(), certificate.loopScope});
  descriptor.eventUses.push_back(
      {releaseDomain, certificate.lanes.size(), certificate.loopScope});
  if (recipe == OwnershipRecipe::BoundaryGuarded) {
    buildRoundTripReady(descriptor, graph, certificate, *bindingIndex, 0,
                        workBudget);
    if (!buildBoundaryGuardedRelease(descriptor, graph, certificate,
                                     *bindingIndex, 1, workBudget)) {
      return std::nullopt;
    }
  } else if (recipe == OwnershipRecipe::Hierarchical) {
    if (!outerScope) {
      return std::nullopt;
    }
    if (certificate.protocol ==
        SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch) {
      if (!buildHierarchicalAlternatingReady(descriptor, graph, certificate,
                                             *bindingIndex, 0, *outerScope,
                                             workBudget) ||
          !buildHierarchicalAlternatingRelease(descriptor, graph, certificate,
                                               *bindingIndex, 1, *outerScope,
                                               workBudget)) {
        return std::nullopt;
      }
    } else {
      if (!initialPathScope) {
        return std::nullopt;
      }
      buildRoundTripReady(descriptor, graph, certificate, *bindingIndex, 0,
                          workBudget);
      if (!buildHierarchicalRoundTripRelease(descriptor, graph, certificate,
                                             *bindingIndex, 1, *outerScope,
                                             *initialPathScope, workBudget)) {
        return std::nullopt;
      }
    }
  } else if (certificate.protocol ==
             SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch) {
    buildAlternatingReady(descriptor, graph, certificate, *bindingIndex, 0,
                          workBudget);
    buildAlternatingRelease(descriptor, graph, certificate, *bindingIndex, 1,
                            workBudget);
  } else {
    buildRoundTripReady(descriptor, graph, certificate, *bindingIndex, 0,
                        workBudget);
    buildRoundTripRelease(descriptor, graph, certificate, *bindingIndex, 1,
                          workBudget);
  }
  const bool bindingWorkUnavailable =
      (workBudget && workBudget->exhausted) ||
      !addOwnershipCompositeBindings(descriptor, graph, bindingIndex->demands,
                                     workBudget);
  if (bindingWorkUnavailable) {
    return std::nullopt;
  }
  std::vector<bool> supplied(descriptor.eventUses.size(), false);
  for (const CanonicalSyncSupplyBinding &binding : descriptor.supplies) {
    if (binding.eventUse && *binding.eventUse < supplied.size()) {
      supplied[*binding.eventUse] = true;
    }
  }
  if (llvm::is_contained(supplied, false)) {
    return std::nullopt;
  }
  return descriptor;
}

CanonicalSyncMechanismOrigin
getBasicOwnershipOrigin(const SyncCoverBasicOwnershipCertificate &certificate) {
  if (certificate.kind == SyncCoverBasicOwnershipKind::L0Operand) {
    return CanonicalSyncMechanismOrigin::BasicOwnershipL0OperandProtocol;
  }
  if (certificate.kind == SyncCoverBasicOwnershipKind::L0Accumulator) {
    return CanonicalSyncMechanismOrigin::BasicOwnershipAccumulatorProtocol;
  }
  return certificate.protocol ==
                 SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch
             ? CanonicalSyncMechanismOrigin::BasicOwnershipAlternatingL1Protocol
             : CanonicalSyncMechanismOrigin::BasicOwnershipStableL1Protocol;
}

CanonicalSyncProblemResult internOwnershipProtocol(
    CanonicalSyncPatternProblem &problem, const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain, OwnershipRecipe recipe,
    CanonicalSyncMechanismOrigin origin,
    std::optional<SyncCoverScopeId> outerScope = std::nullopt,
    std::optional<SyncCoverScopeId> initialPathScope = std::nullopt) {
  std::optional<CanonicalSyncMechanismDescriptor> descriptor =
      makeOwnershipProtocol(graph, certificate, readyDomain, releaseDomain,
                            recipe, outerScope, initialPathScope);
  if (!descriptor) {
    return {CanonicalSyncProblemError::None, std::nullopt};
  }
  const SyncCoverBasicOwnershipCertificateId certificateId = certificate.id;
  return problem.internVerifiedProtocol(
      std::move(*descriptor),
      [&graph, certificateId, readyDomain, releaseDomain, recipe, outerScope,
       initialPathScope](const CanonicalSyncMechanismDescriptor &actual,
                         SyncCoverCoverageWorkBudget &work) {
        if (certificateId >= graph.getBasicOwnershipCertificates().size()) {
          return CanonicalSyncProblemError::UnverifiedProtocol;
        }
        const SyncCoverBasicOwnershipCertificate &storedCertificate =
            graph.getBasicOwnershipCertificates()[certificateId];
        const std::optional<CanonicalSyncMechanismDescriptor> expected =
            makeOwnershipProtocol(graph, storedCertificate, readyDomain,
                                  releaseDomain, recipe, outerScope,
                                  initialPathScope, &work);
        return protocolVerificationResult(
            work,
            expected && ownershipDescriptorEqual(actual, *expected, work));
      },
      origin);
}

struct HierarchicalOwnershipRecipeInfo {
  SyncCoverScopeId outerScope = 0;
  std::optional<SyncCoverScopeId> initialPathScope;
  SyncCoverBasicOwnershipCertificateId stable = 0;
  SyncCoverBasicOwnershipCertificateId alternating = 0;
  SyncCoverBasicOwnershipCertificateId accumulator = 0;
};

std::optional<HierarchicalOwnershipRecipeInfo>
getHierarchicalOwnershipRecipeInfo(
    const SyncCoverGraph &graph,
    const SyncCoverBasicOwnershipCertificate &certificate) {
  const bool l1 = certificate.kind == SyncCoverBasicOwnershipKind::L1Tile;
  if (!l1 || certificate.loopScope >= graph.getScopes().size()) {
    return std::nullopt;
  }
  const SyncCoverScopeId outerScope =
      graph.getScopes()[certificate.loopScope].parent;
  if (outerScope == 0 || outerScope >= graph.getScopes().size() ||
      !graph.getScopes()[outerScope].isLoop) {
    return std::nullopt;
  }

  const SyncCoverBasicOwnershipCertificate *stable = nullptr;
  const SyncCoverBasicOwnershipCertificate *alternating = nullptr;
  const SyncCoverBasicOwnershipCertificate *accumulator = nullptr;
  for (const SyncCoverBasicOwnershipCertificate &candidate :
       graph.getBasicOwnershipCertificates()) {
    if (candidate.kind == SyncCoverBasicOwnershipKind::L1Tile &&
        candidate.loopScope == certificate.loopScope) {
      const bool isAlternating =
          candidate.protocol ==
          SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch;
      const SyncCoverBasicOwnershipCertificate *&slot =
          isAlternating ? alternating : stable;
      if (slot) {
        return std::nullopt;
      }
      slot = &candidate;
      continue;
    }
    if (candidate.kind == SyncCoverBasicOwnershipKind::L0Accumulator &&
        candidate.loopScope == outerScope) {
      if (accumulator) {
        return std::nullopt;
      }
      accumulator = &candidate;
    }
  }
  if (!stable || !alternating || !accumulator ||
      stable->producerResource != alternating->producerResource ||
      stable->consumerResource != alternating->consumerResource ||
      stable->paths.size() != alternating->paths.size()) {
    return std::nullopt;
  }
  std::vector<SyncCoverScopeId> stableScopes;
  std::vector<SyncCoverScopeId> alternatingScopes;
  llvm::transform(
      stable->paths, std::back_inserter(stableScopes),
      [](const SyncCoverBasicOwnershipPath &path) { return path.scope; });
  llvm::transform(
      alternating->paths, std::back_inserter(alternatingScopes),
      [](const SyncCoverBasicOwnershipPath &path) { return path.scope; });
  llvm::sort(stableScopes);
  llvm::sort(alternatingScopes);
  if (stableScopes != alternatingScopes) {
    return std::nullopt;
  }

  HierarchicalOwnershipRecipeInfo result;
  result.outerScope = outerScope;
  result.stable = stable->id;
  result.alternating = alternating->id;
  result.accumulator = accumulator->id;
  if (certificate.protocol == SyncCoverBasicOwnershipProtocolKind::RoundTrip) {
    const std::optional<std::size_t> initialPath =
        findInitialOwnershipPath(graph, *alternating);
    if (!initialPath) {
      return std::nullopt;
    }
    result.initialPathScope = alternating->paths[*initialPath].scope;
  }
  return result;
}

bool ownershipSlotsOverlap(const SyncCoverBasicOwnershipSlot &left,
                           const SyncCoverBasicOwnershipSlot &right) {
  return left.domain == right.domain && left.extent.begin < right.extent.end &&
         right.extent.begin < left.extent.end;
}

bool verifyCompositeOwnershipAccessClosure(
    const SyncCoverGraph &graph, const HierarchicalOwnershipRecipeInfo &info,
    SyncCoverCoverageWorkBudget *workBudget = nullptr) {
  const auto &certificates = graph.getBasicOwnershipCertificates();
  if (info.stable >= certificates.size() ||
      info.alternating >= certificates.size() ||
      info.accumulator >= certificates.size() ||
      info.outerScope >= graph.getScopes().size()) {
    return false;
  }
  const SyncCoverBasicOwnershipCertificate &stable = certificates[info.stable];
  const SyncCoverBasicOwnershipCertificate &alternating =
      certificates[info.alternating];
  const SyncCoverBasicOwnershipCertificate &accumulator =
      certificates[info.accumulator];
  const bool invalidShape =
      stable.kind != SyncCoverBasicOwnershipKind::L1Tile ||
      stable.protocol != SyncCoverBasicOwnershipProtocolKind::RoundTrip ||
      alternating.kind != SyncCoverBasicOwnershipKind::L1Tile ||
      alternating.protocol !=
          SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch ||
      stable.loopScope != alternating.loopScope ||
      graph.getScopes()[stable.loopScope].parent != info.outerScope ||
      accumulator.kind != SyncCoverBasicOwnershipKind::L0Accumulator ||
      accumulator.loopScope != info.outerScope;
  if (invalidShape) {
    return false;
  }

  if (workBudget) {
    std::size_t slots = 0;
    std::size_t accessReferences = 0;
    for (const SyncCoverBasicOwnershipCertificate *certificate :
         {&stable, &alternating}) {
      if (!workBudget->consume(certificate->lanes.size())) {
        return false;
      }
      for (const SyncCoverBasicOwnershipLane &lane : certificate->lanes) {
        const bool laneWorkUnavailable =
            !checkedProtocolAdd(slots, lane.slots.size(), *workBudget) ||
            !workBudget->consume(lane.slots.size());
        if (laneWorkUnavailable) {
          return false;
        }
        for (const SyncCoverBasicOwnershipSlot &slot : lane.slots) {
          if (!checkedProtocolAdd(accessReferences, slot.accesses.size(),
                                  *workBudget) ||
              !workBudget->consume(slot.accesses.size())) {
            return false;
          }
        }
      }
    }
    std::size_t perAccess = 1;
    const bool accessWorkUnavailable =
        !checkedProtocolAdd(perAccess, slots, *workBudget) ||
        !checkedProtocolAdd(perAccess, graph.getScopes().size(), *workBudget) ||
        !consumeProtocolProduct(*workBudget, slots, slots) ||
        !consumeProtocolProduct(*workBudget, accessReferences,
                                accessReferences) ||
        !consumeProtocolProduct(*workBudget, graph.getStorageAccesses().size(),
                                perAccess);
    if (accessWorkUnavailable) {
      return false;
    }
  }

  std::vector<const SyncCoverBasicOwnershipSlot *> managedSlots;
  std::set<SyncCoverStorageAccessId> namedAccesses;
  for (const SyncCoverBasicOwnershipCertificate *certificate :
       {&stable, &alternating}) {
    for (const SyncCoverBasicOwnershipLane &lane : certificate->lanes) {
      for (const SyncCoverBasicOwnershipSlot &slot : lane.slots) {
        if (llvm::any_of(managedSlots, [&](const auto *existing) {
              return ownershipSlotsOverlap(*existing, slot);
            })) {
          return false;
        }
        managedSlots.push_back(&slot);
        namedAccesses.insert(slot.accesses.begin(), slot.accesses.end());
      }
    }
  }
  if (managedSlots.empty() || namedAccesses.empty()) {
    return false;
  }
  for (SyncCoverStorageAccessId accessId : namedAccesses) {
    if (accessId >= graph.getStorageAccesses().size()) {
      return false;
    }
    const SyncCoverStorageAccess &access = graph.getStorageAccesses()[accessId];
    if (!graph.scopeContains(info.outerScope,
                             graph.getNodes()[access.node].scope)) {
      return false;
    }
  }
  for (const SyncCoverStorageAccess &access : graph.getStorageAccesses()) {
    if (!graph.scopeContains(info.outerScope,
                             graph.getNodes()[access.node].scope)) {
      continue;
    }
    const bool touchesManagedSlot =
        llvm::any_of(managedSlots, [&](const auto *slot) {
          return access.domain == slot->domain &&
                 access.extent.begin < slot->extent.end &&
                 slot->extent.begin < access.extent.end;
        });
    if (touchesManagedSlot && !namedAccesses.count(access.id)) {
      return false;
    }
  }
  return true;
}

std::optional<HierarchicalOwnershipRecipeInfo>
getCompositeOwnershipRecipeInfo(const SyncCoverGraph &graph) {
  std::optional<HierarchicalOwnershipRecipeInfo> result;
  for (const SyncCoverBasicOwnershipCertificate &certificate :
       graph.getBasicOwnershipCertificates()) {
    if (certificate.kind != SyncCoverBasicOwnershipKind::L1Tile ||
        certificate.protocol !=
            SyncCoverBasicOwnershipProtocolKind::RoundTrip) {
      continue;
    }
    const std::optional<HierarchicalOwnershipRecipeInfo> candidate =
        getHierarchicalOwnershipRecipeInfo(graph, certificate);
    if (!candidate ||
        !verifyCompositeOwnershipAccessClosure(graph, *candidate)) {
      continue;
    }
    if (result) {
      return std::nullopt;
    }
    result = candidate;
  }
  return result;
}

void appendOwnershipDescriptor(CanonicalSyncMechanismDescriptor &result,
                               CanonicalSyncMechanismDescriptor component) {
  const std::size_t eventUseOffset = result.eventUses.size();
  const std::size_t actionOffset = result.actions.size();
  for (CanonicalSyncAction &action : component.actions) {
    if (action.eventUse) {
      *action.eventUse += eventUseOffset;
    }
  }
  for (CanonicalSyncSupplyBinding &binding : component.supplies) {
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
  }
  result.eventUses.insert(result.eventUses.end(),
                          std::make_move_iterator(component.eventUses.begin()),
                          std::make_move_iterator(component.eventUses.end()));
  result.actions.insert(result.actions.end(),
                        std::make_move_iterator(component.actions.begin()),
                        std::make_move_iterator(component.actions.end()));
  result.supplies.insert(result.supplies.end(),
                         std::make_move_iterator(component.supplies.begin()),
                         std::make_move_iterator(component.supplies.end()));
}

std::optional<CanonicalSyncMechanismDescriptor> makeCompositeOwnershipProtocol(
    const SyncCoverGraph &graph, const HierarchicalOwnershipRecipeInfo &info,
    CanonicalSyncEventDomainId l1ReadyDomain,
    CanonicalSyncEventDomainId l1ReleaseDomain,
    CanonicalSyncEventDomainId accumulatorReadyDomain,
    CanonicalSyncEventDomainId accumulatorReleaseDomain,
    SyncCoverCoverageWorkBudget *workBudget = nullptr) {
  if (!verifyCompositeOwnershipAccessClosure(graph, info, workBudget)) {
    return std::nullopt;
  }
  const auto &certificates = graph.getBasicOwnershipCertificates();
  std::optional<CanonicalSyncMechanismDescriptor> stable =
      makeOwnershipProtocol(graph, certificates[info.stable], l1ReadyDomain,
                            l1ReleaseDomain, OwnershipRecipe::Hierarchical,
                            info.outerScope, info.initialPathScope, workBudget);
  std::optional<CanonicalSyncMechanismDescriptor> alternating =
      makeOwnershipProtocol(graph, certificates[info.alternating],
                            l1ReadyDomain, l1ReleaseDomain,
                            OwnershipRecipe::Hierarchical, info.outerScope,
                            std::nullopt, workBudget);
  std::optional<CanonicalSyncMechanismDescriptor> accumulator =
      makeOwnershipProtocol(graph, certificates[info.accumulator],
                            accumulatorReadyDomain, accumulatorReleaseDomain,
                            OwnershipRecipe::BoundaryGuarded, std::nullopt,
                            std::nullopt, workBudget);
  if (!stable || !alternating || !accumulator) {
    return std::nullopt;
  }
  if (workBudget) {
    std::size_t appendWork = 0;
    for (const CanonicalSyncMechanismDescriptor *component :
         {&*stable, &*alternating, &*accumulator}) {
      if (!checkedProtocolAdd(appendWork, component->eventUses.size(),
                              *workBudget) ||
          !checkedProtocolAdd(appendWork, component->actions.size(),
                              *workBudget) ||
          !checkedProtocolAdd(appendWork, component->supplies.size(),
                              *workBudget)) {
        return std::nullopt;
      }
    }
    if (!workBudget->consume(appendWork)) {
      return std::nullopt;
    }
  }
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Protocol;
  appendOwnershipDescriptor(result, std::move(*stable));
  appendOwnershipDescriptor(result, std::move(*alternating));
  appendOwnershipDescriptor(result, std::move(*accumulator));
  return result;
}

LogicalResult addOwnershipProtocols(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
    bool includeL0Operand, bool includeBasic, bool includeBoundary,
    bool includeHierarchical) {
  const SyncCoverGraph &graph = program.getGraph();
  std::map<SyncCoverBasicOwnershipCertificateId,
           std::vector<CanonicalSyncMechanismId>>
      alternativesByCertificate;
  for (const SyncCoverBasicOwnershipCertificate &certificate :
       graph.getBasicOwnershipCertificates()) {
    const auto ready = domainIds.find(
        {certificate.producerResource, certificate.consumerResource});
    const auto release = domainIds.find(
        {certificate.consumerResource, certificate.producerResource});
    if (ready == domainIds.end() || release == domainIds.end()) {
      return program.getFunction().emitError(
          "canonical sync basic ownership certificate has no event domain");
    }
    std::vector<CanonicalSyncMechanismId> &alternatives =
        alternativesByCertificate[certificate.id];
    const auto addRecipe = [&](OwnershipRecipe recipe,
                               CanonicalSyncMechanismOrigin origin) {
      const CanonicalSyncProblemResult added =
          internOwnershipProtocol(problem, graph, certificate, ready->second,
                                  release->second, recipe, origin);
      if (!added || !added.index) {
        return added;
      }
      alternatives.push_back(*added.index);
      return added;
    };
    if (includeBasic ||
        (includeL0Operand &&
         certificate.kind == SyncCoverBasicOwnershipKind::L0Operand)) {
      const CanonicalSyncProblemResult added = addRecipe(
          OwnershipRecipe::Basic, getBasicOwnershipOrigin(certificate));
      if (!added) {
        return program.getFunction().emitError(
                   "cannot add canonical sync basic ownership protocol ")
               << certificate.id
               << ", error=" << static_cast<unsigned>(added.error);
      }
    }
    if (includeBoundary &&
        certificate.kind == SyncCoverBasicOwnershipKind::L0Accumulator) {
      const CanonicalSyncProblemResult added = addRecipe(
          OwnershipRecipe::BoundaryGuarded,
          CanonicalSyncMechanismOrigin::BoundaryGuardedAccumulatorProtocol);
      if (!added && added.error != CanonicalSyncProblemError::None) {
        return program.getFunction().emitError(
                   "cannot add boundary-guarded accumulator protocol ")
               << certificate.id
               << ", error=" << static_cast<unsigned>(added.error);
      }
    }
    for (std::size_t first = 0; first < alternatives.size(); ++first) {
      for (std::size_t second = first + 1; second < alternatives.size();
           ++second) {
        const CanonicalSyncProblemResult conflict =
            problem.addConflict(alternatives[first], alternatives[second]);
        if (!conflict) {
          return program.getFunction().emitError(
              "cannot add canonical sync ownership-alternative conflict");
        }
      }
    }
  }
  if (!includeHierarchical) {
    return success();
  }

  const std::optional<HierarchicalOwnershipRecipeInfo> composite =
      getCompositeOwnershipRecipeInfo(graph);
  if (!composite) {
    return success();
  }
  const auto &certificates = graph.getBasicOwnershipCertificates();
  const SyncCoverBasicOwnershipCertificate &stable =
      certificates[composite->stable];
  const SyncCoverBasicOwnershipCertificate &accumulator =
      certificates[composite->accumulator];
  const auto l1Ready =
      domainIds.find({stable.producerResource, stable.consumerResource});
  const auto l1Release =
      domainIds.find({stable.consumerResource, stable.producerResource});
  const auto accumulatorReady = domainIds.find(
      {accumulator.producerResource, accumulator.consumerResource});
  const auto accumulatorRelease = domainIds.find(
      {accumulator.consumerResource, accumulator.producerResource});
  if (l1Ready == domainIds.end() || l1Release == domainIds.end() ||
      accumulatorReady == domainIds.end() ||
      accumulatorRelease == domainIds.end()) {
    return program.getFunction().emitError(
        "canonical sync composite ownership has no event domain");
  }
  const CanonicalSyncEventDomainId l1ReadyDomain = l1Ready->second;
  const CanonicalSyncEventDomainId l1ReleaseDomain = l1Release->second;
  const CanonicalSyncEventDomainId accumulatorReadyDomain =
      accumulatorReady->second;
  const CanonicalSyncEventDomainId accumulatorReleaseDomain =
      accumulatorRelease->second;
  std::optional<CanonicalSyncMechanismDescriptor> descriptor =
      makeCompositeOwnershipProtocol(graph, *composite, l1ReadyDomain,
                                     l1ReleaseDomain, accumulatorReadyDomain,
                                     accumulatorReleaseDomain);
  if (!descriptor) {
    return program.getFunction().emitError(
        "cannot construct canonical sync composite ownership protocol");
  }
  const HierarchicalOwnershipRecipeInfo compositeInfo = *composite;
  const CanonicalSyncProblemResult added = problem.internVerifiedProtocol(
      std::move(*descriptor),
      [&graph, compositeInfo, l1ReadyDomain, l1ReleaseDomain,
       accumulatorReadyDomain,
       accumulatorReleaseDomain](const CanonicalSyncMechanismDescriptor &actual,
                                 SyncCoverCoverageWorkBudget &work) {
        const std::optional<CanonicalSyncMechanismDescriptor> expected =
            makeCompositeOwnershipProtocol(
                graph, compositeInfo, l1ReadyDomain, l1ReleaseDomain,
                accumulatorReadyDomain, accumulatorReleaseDomain, &work);
        return protocolVerificationResult(
            work,
            expected && ownershipDescriptorEqual(actual, *expected, work));
      },
      CanonicalSyncMechanismOrigin::CompositeOwnershipProtocol);
  if (!added || !added.index) {
    return program.getFunction().emitError(
               "cannot add canonical sync composite ownership protocol, error=")
           << static_cast<unsigned>(added.error);
  }
  for (SyncCoverBasicOwnershipCertificateId certificate :
       {composite->stable, composite->alternating, composite->accumulator}) {
    for (CanonicalSyncMechanismId alternative :
         alternativesByCertificate[certificate]) {
      const CanonicalSyncProblemResult conflict =
          problem.addConflict(*added.index, alternative);
      if (!conflict) {
        return program.getFunction().emitError(
            "cannot add composite ownership-alternative conflict");
      }
    }
  }
  return success();
}

LogicalResult
addEventDomains(const CanonicalSyncProgram &program, unsigned budget,
                CanonicalSyncPatternProblem &problem,
                const SyncCoverDemandSet &baseline,
                std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
                bool includeBasicOwnership) {
  const SyncCoverGraph &graph = program.getGraph();
  const CanonicalSyncDirectedResourceCapability &directEvents =
      program.getTargetCapabilities().directEventCompletion;
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
    if (directEvents.supports(key.first, key.second)) {
      keys.insert(key);
    }
  }
  if (includeBasicOwnership) {
    for (const SyncCoverBasicOwnershipCertificate &certificate :
         graph.getBasicOwnershipCertificates()) {
      const EventDomainKey ready{certificate.producerResource,
                                 certificate.consumerResource};
      const EventDomainKey release{certificate.consumerResource,
                                   certificate.producerResource};
      if (directEvents.supports(ready.first, ready.second)) {
        keys.insert(ready);
      }
      if (directEvents.supports(release.first, release.second)) {
        keys.insert(release);
      }
    }
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
         capabilities.mte1L0ReadySetCompletesPrefix.isEnabled()) ||
        (certificate.kind ==
             SyncCoverTargetCompletionKind::MToFixAccumulatorBoundary &&
         capabilities.mToFixAccumulatorBoundaryCompletes.isEnabled());
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
    const CanonicalSyncProblemResult added = problem.internMechanism(
        std::move(descriptor),
        CanonicalSyncMechanismOrigin::TargetCompletionCertificateEvent);
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
    bool requireIndependentEventProtocol = false,
    SyncCoverDemandSet *admittedDemands = nullptr,
    SyncCoverCoverageWorkBudget *admissionWork = nullptr) {
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
      if (directEvent) {
        directEvents.push_back({demandId, *added.index, domain->second});
      }
      return recordDirectOnlyDemand(program, demandId, admittedDemands,
                                    admissionWork);
    };

    const bool repeatedDistanceZeroEvent =
        demand.distance == 0 &&
        (graph.getNearestEnclosingLoop(graph.getNodes()[demand.source].scope) ||
         graph.getNearestEnclosingLoop(graph.getNodes()[demand.target].scope));
    if (requireIndependentEventProtocol && repeatedDistanceZeroEvent) {
      return program.getFunction().emitError(
                 "canonical sync direct-only mode cannot admit a "
                 "loop-repeated one-bit event without an independent token "
                 "lifecycle protocol for demand ")
             << demandId;
    }
    if (demand.distance == 0 && canUseDistanceZeroEvent(graph, demand) &&
        failed(
            record(problem.internMechanism(
                       makeDirectEvent(graph, demandId, domain->second,
                                       requireIndependentEventProtocol),
                       CanonicalSyncMechanismOrigin::DirectDistanceZeroEvent),
                   true))) {
      return failure();
    }
    if (demand.distance != 0 && canUseRecurrenceEvent(graph, demand)) {
      CanonicalSyncMechanismDescriptor descriptor =
          makeRecurrenceEvent(graph, demand, domain->second);
      if (requireIndependentEventProtocol &&
          descriptor.kind != CanonicalSyncMechanismKind::Protocol) {
        return program.getFunction().emitError(
                   "canonical sync direct-only mode cannot admit a "
                   "loop-repeated event without an independent token "
                   "lifecycle protocol for demand ")
               << demandId;
      }
      CanonicalSyncProblemResult added;
      if (descriptor.kind == CanonicalSyncMechanismKind::Protocol) {
        const CanonicalSyncEventDomainId recurrenceDomain = domain->second;
        added = problem.internVerifiedProtocol(
            std::move(descriptor),
            [&graph, demandId,
             recurrenceDomain](const CanonicalSyncMechanismDescriptor &actual,
                               SyncCoverCoverageWorkBudget &work) {
              const bool validDemand = demandId < graph.getDemands().size();
              const SyncCoverNodeId source =
                  validDemand ? graph.getDemands()[demandId].source : 0;
              const std::size_t completionLookupWork =
                  validDemand && source < graph.getNodes().size()
                      ? graph.getNodes()[source].completionTargets.size()
                      : 0;
              const bool workAvailable =
                  consumeProtocolProduct(work, graph.getScopes().size(), 4) &&
                  consumeProtocolWork(work, completionLookupWork) &&
                  consumeProtocolWork(work, actual.actions.size()) &&
                  consumeProtocolWork(work, actual.eventUses.size()) &&
                  consumeProtocolWork(work, actual.supplies.size()) &&
                  consumeProtocolWork(work);
              if (!workAvailable) {
                return CanonicalSyncProblemError::LimitExceeded;
              }
              const bool valid =
                  validDemand &&
                  verifyRecurrenceEvent(graph, graph.getDemands()[demandId],
                                        recurrenceDomain, actual);
              return protocolVerificationResult(work, valid);
            },
            CanonicalSyncMechanismOrigin::DirectReleaseRecurrenceProtocol);
      } else if (verifyRecurrenceEvent(graph, demand, domain->second,
                                       descriptor)) {
        added = problem.internMechanism(
            std::move(descriptor),
            CanonicalSyncMechanismOrigin::DirectForwardRecurrenceEvent);
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
    const CanonicalSyncProblemResult added = problem.internMechanism(
        std::move(descriptor),
        CanonicalSyncMechanismOrigin::CompletionFrontierEvent);
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
    bool requireDistanceZeroBinding, TargetLocalCatalogMode mode,
    SyncCoverCoverageWorkBudget *workBudget = nullptr) {
  const SyncCoverGraph &graph = program.getGraph();
  using TargetFenceGroupKey =
      std::pair<CanonicalSyncEventDomainId, SyncCoverNodeId>;
  using TargetPipeDrainGroupKey = std::pair<std::uint32_t, SyncCoverNodeId>;
  std::map<TargetFenceGroupKey, std::vector<SyncCoverDemandId>> eventGroups;
  std::map<TargetPipeDrainGroupKey, std::vector<SyncCoverDemandId>>
      pipeDrainGroups;
  for (SyncCoverDemandId demandId : demandIds) {
    if (workBudget && !workBudget->consume()) {
      return failure();
    }
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
        if (canUseStandaloneTargetedBarrier(graph, source.resource,
                                            target.resource)) {
          pipeDrainGroups[{source.resource, target.physicalAnchor}].push_back(
              demandId);
        }
        continue;
      }
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
  const auto internDescriptor =
      [&](CanonicalSyncMechanismDescriptor descriptor, StringRef role,
          CanonicalSyncMechanismOrigin origin, bool optional) -> LogicalResult {
    if (optionalCatalogFull) {
      return success();
    }
    const CanonicalSyncProblemResult added =
        problem.internMechanism(std::move(descriptor), origin);
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
    return internDescriptor(std::move(descriptor), "event group",
                            CanonicalSyncMechanismOrigin::TargetLocalFenceEvent,
                            true);
  };
  for (const auto &[key, groupedDemands] : eventGroups) {
    if (workBudget && !workBudget->consume(groupedDemands.size())) {
      return failure();
    }
    if (failed(addEventGroup(groupedDemands, key.first))) {
      return failure();
    }
  }
  for (const auto &[key, groupedDemands] : pipeDrainGroups) {
    if (workBudget && !workBudget->consume(groupedDemands.size())) {
      return failure();
    }
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
    if (failed(internDescriptor(
            std::move(descriptor), "pipe-drain cut",
            CanonicalSyncMechanismOrigin::RepairTargetLocalPipeDrain, false))) {
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
          hasDemands &&
          problem.internMechanism(
              makeBarrier(graph, allResources, group.demands, broad),
              CanonicalSyncMechanismOrigin::DirectTargetedBarrier);
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

/// Add the correctness basis for same-resource rows without pre-grouping rows
/// into a synthesized cut. Grounding may still prove that one such direct
/// barrier covers additional rows.
LogicalResult addDirectTargetedBarriers(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const SyncCoverDemandSet &baseline,
    SyncCoverDemandSet *admittedDemands = nullptr,
    SyncCoverCoverageWorkBudget *admissionWork = nullptr) {
  const SyncCoverGraph &graph = program.getGraph();
  const std::vector<std::uint32_t> allResources = getIssueResources(graph);
  for (SyncCoverDemandId demandId : problem.getDemands()) {
    if (baseline.contains(demandId)) {
      continue;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    if (source.resource != target.resource) {
      continue;
    }
    if (!canUseStandaloneTargetedBarrier(graph, target.resource,
                                         target.resource)) {
      return program.getFunction().emitError(
                 "canonical sync direct catalog cannot place a targeted "
                 "barrier for demand ")
             << demandId;
    }
    const std::array<SyncCoverDemandId, 1> directDemand{demandId};
    const CanonicalSyncProblemResult added = problem.internMechanism(
        makeBarrier(graph, allResources, directDemand, false,
                    /*attestDemands=*/true),
        CanonicalSyncMechanismOrigin::DirectTargetedBarrier);
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      return program.getFunction().emitError(
          "canonical sync mechanism limit prevents a complete direct "
          "catalog");
    }
    if (!added || !added.index) {
      return program.getFunction().emitError(
                 "cannot add canonical sync direct targeted barrier, error=")
             << static_cast<unsigned>(added.error);
    }
    if (failed(recordDirectOnlyDemand(program, demandId, admittedDemands,
                                      admissionWork))) {
      return failure();
    }
  }
  return success();
}

LogicalResult
requireCompleteDirectOnlyCatalog(const CanonicalSyncProgram &program,
                                 CanonicalSyncPatternProblem &problem,
                                 const SyncCoverDemandSet &baseline,
                                 const SyncCoverDemandSet &admittedDemands,
                                 SyncCoverCoverageWorkBudget &workBudget) {
  if (!consumeProtocolProduct(workBudget, problem.getDemands().size(), 2)) {
    return program.getFunction().emitError(
        "canonical sync direct-only completeness work limit exceeded");
  }
  for (SyncCoverDemandId demandId : problem.getDemands()) {
    if (baseline.contains(demandId) || admittedDemands.contains(demandId)) {
      continue;
    }
    const SyncCoverDemand &demand = program.getGraph().getDemands()[demandId];
    const SyncCoverNode &source = program.getGraph().getNodes()[demand.source];
    const SyncCoverNode &target = program.getGraph().getNodes()[demand.target];
    return program.getFunction().emitError(
               "canonical sync direct-only catalog has no "
               "independently attested recipe for demand ")
           << demandId << ", source-resource=" << source.resource
           << ", target-resource=" << target.resource
           << ", scope=" << demand.scope << ", distance=" << demand.distance;
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

LogicalResult addSourceLocalPipeDrains(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const SyncCoverDemandSet &baseline, ArrayRef<SyncCoverDemandId> demandIds,
    bool crossResource, SyncCoverCoverageWorkBudget *workBudget = nullptr) {
  const SyncCoverGraph &graph = program.getGraph();
  using SourceDrainKey = std::pair<SyncCoverNodeId, std::uint32_t>;
  std::map<SourceDrainKey, std::vector<SyncCoverDemandId>> groups;
  for (SyncCoverDemandId demandId : demandIds) {
    if (workBudget && !workBudget->consume()) {
      return failure();
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
        !canUseStandaloneTargetedBarrier(graph, source.resource,
                                         target.resource)) {
      continue;
    }
    groups[{source.physicalExit, source.resource}].push_back(demandId);
  }
  for (const auto &[key, demands] : groups) {
    if (workBudget && !workBudget->consume(demands.size())) {
      return failure();
    }
    (void)key;
    const CanonicalSyncMechanismOrigin origin =
        crossResource ? CanonicalSyncMechanismOrigin::RepairSourceLocalPipeDrain
                      : CanonicalSyncMechanismOrigin::SourceLocalPipeDrain;
    const CanonicalSyncProblemResult added = problem.internMechanism(
        makeSourceLocalPipeDrain(graph, demands), origin);
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
    ArrayRef<SyncCoverDemandId> demandIds, bool crossResource,
    SyncCoverCoverageWorkBudget *workBudget = nullptr) {
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
    if (workBudget && !workBudget->consume(amount)) {
      truncated = true;
      return false;
    }
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
    if (!canUseStandaloneTargetedBarrier(graph, source.resource,
                                         target.resource)) {
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
      const CanonicalSyncMechanismOrigin origin =
          crossResource
              ? CanonicalSyncMechanismOrigin::RepairSourcePrefixPipeDrain
              : CanonicalSyncMechanismOrigin::SourcePrefixPipeDrain;
      const CanonicalSyncProblemResult added = problem.internMechanism(
          makeSourcePrefixPipeDrain(graph, cut.id, key.second, groupedDemands),
          origin);
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
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    if (!canUseStandaloneTargetedBarrier(graph, source.resource,
                                         target.resource)) {
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
        makeLoopCarryPipeDrain(graph, key.first, key.second, groupDemands),
        CanonicalSyncMechanismOrigin::LoopCarryPipeDrain);
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
            makeBarrier(graph, allResources, group.demands, true),
            CanonicalSyncMechanismOrigin::LocalizedPipeAll)) {
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

RepairFrontierBuildResult
addRepairFrontierPatterns(const CanonicalSyncProgram &program,
                          CanonicalSyncPatternProblem &problem,
                          ArrayRef<DirectEventRecord> directEvents,
                          ArrayRef<CanonicalSyncMechanismId> conflictCore,
                          const CanonicalSyncPatternOptions &options,
                          SyncCoverCoverageWorkBudget *workBudget = nullptr) {
  const SyncCoverGraph &graph = program.getGraph();
  RepairFrontierBuildResult result;
  std::vector<CanonicalSyncMechanismId> sortedCore(conflictCore.begin(),
                                                   conflictCore.end());
  if (!stableSortAndUniqueRepairValues(sortedCore, workBudget)) {
    result.status = RepairFrontierBuildStatus::Truncated;
    return result;
  }
  std::vector<DirectEventRecord> liveEvents;
  if (workBudget && !workBudget->consume(directEvents.size())) {
    result.status = RepairFrontierBuildStatus::Truncated;
    return result;
  }
  for (const DirectEventRecord &event : directEvents) {
    const std::optional<bool> live =
        meteredRepairContains(ArrayRef<CanonicalSyncMechanismId>(sortedCore),
                              event.mechanism, workBudget);
    if (!live) {
      result.status = RepairFrontierBuildStatus::Truncated;
      return result;
    }
    if (*live) {
      liveEvents.push_back(event);
    }
  }
  const auto less = [](const RepairFrontierProposal &left,
                       const RepairFrontierProposal &right) {
    return frontierProposalLess(left, right);
  };
  std::set<RepairFrontierProposal, decltype(less)> proposals(less);
  const std::size_t proposalLimit =
      std::min(options.maximumRepairFrontierProposals,
               problem.getLimits().maximumPatternProposals);
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
      if (workBudget && !workBudget->consume()) {
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
      std::size_t proposalLookupWork = 0;
      const bool proposalLookupAvailable =
          checkedOrderedLookupWork(proposals.size() + 1, proposalLookupWork) &&
          (!workBudget || workBudget->consume(proposalLookupWork));
      if (!proposalLookupAvailable) {
        result.status = RepairFrontierBuildStatus::Truncated;
        result.proposals = proposals.size();
        return result;
      }
      const auto insertion = proposals.lower_bound(proposal);
      const bool duplicate =
          insertion != proposals.end() && !less(proposal, *insertion);
      if (duplicate) {
        continue;
      }
      const bool proposalLimitReached = proposals.size() == proposalLimit;
      if (proposalLimitReached) {
        result.status = RepairFrontierBuildStatus::Truncated;
        result.proposals = proposals.size();
        return result;
      }
      proposals.emplace_hint(insertion, std::move(proposal));
    }
  }
  result.proposals = proposals.size();
  constexpr std::size_t stagingEntriesPerProposal = 8;
  const bool stagingSizeOverflows =
      proposals.size() >
      std::numeric_limits<std::size_t>::max() / stagingEntriesPerProposal;
  const std::size_t stagingEntries =
      stagingSizeOverflows ? 0 : proposals.size() * stagingEntriesPerProposal;
  if (stagingSizeOverflows ||
      (workBudget && !workBudget->consume(stagingEntries))) {
    result.status = RepairFrontierBuildStatus::Truncated;
    return result;
  }
  std::vector<CanonicalSyncRepairFrontierBatchEntry> batch;
  batch.reserve(proposals.size());
  while (!proposals.empty()) {
    const auto proposal = proposals.begin();
    batch.push_back(
        {makeRepairBarrier(graph, proposal->barrier),
         makeRepairEvent(graph, proposal->event, proposal->domain)});
    proposals.erase(proposal);
  }
  const CanonicalSyncProblemResult committed =
      problem.addRepairFrontierBatch(std::move(batch));
  if (committed.error == CanonicalSyncProblemError::LimitExceeded) {
    result.status = RepairFrontierBuildStatus::Truncated;
    return result;
  }
  if (!committed) {
    result.status = RepairFrontierBuildStatus::Failed;
    program.getFunction().emitError(
        "cannot add canonical sync repair-frontier batch");
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
  std::size_t incidences = 0;
  bool workExceeded = false;
};

RepairCriticalDemandResult
findRepairCriticalDemands(const CanonicalSyncPatternProblem &preciseProblem,
                          ArrayRef<CanonicalSyncMechanismId> conflictCore,
                          ArrayRef<CanonicalSyncMechanismId> selectedMechanisms,
                          std::size_t maximumIncidences,
                          SyncCoverCoverageWorkBudget *workBudget) {
  RepairCriticalDemandResult result;
  const auto consume = [&](std::size_t amount) {
    if (workBudget && !workBudget->consume(amount)) {
      result.workExceeded = true;
      return false;
    }
    return true;
  };
  if (!consume(selectedMechanisms.size())) {
    return result;
  }
  std::vector<CanonicalSyncMechanismId> selected(selectedMechanisms.begin(),
                                                 selectedMechanisms.end());
  if (!stableSortAndUniqueRepairValues(selected, workBudget)) {
    result.workExceeded = true;
    return result;
  }
  std::size_t coverageWorkspaceWork = 0;
  const bool coverageWorkspaceAvailable =
      checkedBasisSum(preciseProblem.getPatterns().size(),
                      preciseProblem.getDemands().size(),
                      coverageWorkspaceWork) &&
      consume(coverageWorkspaceWork);
  if (!coverageWorkspaceAvailable) {
    return result;
  }
  std::vector<bool> active(preciseProblem.getPatterns().size(), false);
  std::vector<std::size_t> totalCoverage(preciseProblem.getDemands().size(), 0);
  const bool useSelectedPlan = !selected.empty();
  for (const CanonicalSyncPattern &pattern : preciseProblem.getPatterns()) {
    bool patternActive = !useSelectedPlan;
    if (useSelectedPlan) {
      patternActive = true;
      for (CanonicalSyncMechanismId mechanism : pattern.members) {
        const std::optional<bool> selectedMember =
            meteredRepairContains(ArrayRef<CanonicalSyncMechanismId>(selected),
                                  mechanism, workBudget);
        if (!selectedMember) {
          result.workExceeded = true;
          return result;
        }
        if (!*selectedMember) {
          patternActive = false;
          break;
        }
      }
    }
    if (!patternActive) {
      continue;
    }
    active[pattern.id] = true;
    if (!consume(preciseProblem.getDemands().size())) {
      return result;
    }
    for (std::size_t demand = 0; demand < preciseProblem.getDemands().size();
         ++demand) {
      if (!pattern.coverage.contains(demand)) {
        continue;
      }
      ++totalCoverage[demand];
    }
  }

  if (!consume(conflictCore.size())) {
    return result;
  }
  std::vector<CanonicalSyncMechanismId> owners(conflictCore.begin(),
                                               conflictCore.end());
  if (!stableSortAndUniqueRepairValues(owners, workBudget)) {
    result.workExceeded = true;
    return result;
  }
  for (CanonicalSyncMechanismId owner : owners) {
    if (owner >= preciseProblem.getMechanismPatterns().size()) {
      result.workExceeded = true;
      return result;
    }
    if (!consume(preciseProblem.getDemands().size())) {
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
      if (!consume(preciseProblem.getDemands().size())) {
        return result;
      }
      for (std::size_t demand = 0; demand < preciseProblem.getDemands().size();
           ++demand) {
        if (!pattern.coverage.contains(demand)) {
          continue;
        }
        ++removedCoverage[demand];
      }
    }
    std::size_t criticalMapEntries = 0;
    std::size_t criticalMapWork = 0;
    const bool criticalMapWorkAvailable =
        checkedBasisSum(result.demands.size(), 1, criticalMapEntries) &&
        checkedOrderedLookupWork(criticalMapEntries, criticalMapWork) &&
        consume(criticalMapWork);
    if (!criticalMapWorkAvailable) {
      return result;
    }
    auto [criticalPosition, inserted] = result.demands.try_emplace(owner);
    (void)inserted;
    std::vector<SyncCoverDemandId> &critical = criticalPosition->second;
    if (!consume(preciseProblem.getDemands().size())) {
      return result;
    }
    for (std::size_t demand = 0; demand < preciseProblem.getDemands().size();
         ++demand) {
      const bool fixed = preciseProblem.getBaselineCoverage().contains(demand);
      if (!fixed && removedCoverage[demand] != 0 &&
          removedCoverage[demand] == totalCoverage[demand]) {
        if (result.incidences == maximumIncidences) {
          result.workExceeded = true;
          return result;
        }
        critical.push_back(preciseProblem.getDemands()[demand]);
        ++result.incidences;
      }
    }
  }
  return result;
}

enum class CandidateCatalogKind : std::uint8_t {
  Precise,
  LocalizedPipeAll,
};

CanonicalSyncProblemBuildResult
buildCandidateCatalog(const CanonicalSyncProgram &program,
                      const CanonicalSyncBuildOptions &options,
                      CandidateCatalogKind kind) {
  if (options.eventIdBudget > kCompilerUsableEventIdCount) {
    program.getFunction().emitError(
        "canonical sync event-id budget must be in [0, 6]");
    return {nullptr, {CanonicalSyncProblemError::InvalidDomain, std::nullopt}};
  }
  const CanonicalSyncMechanismFamilyMask familyMask =
      options.patterns.enabledMechanismFamilies;
  if ((familyMask & ~kAllCanonicalSyncMechanismFamilies) != 0) {
    program.getFunction().emitError(
        "canonical sync mechanism-family mask contains an invalid bit");
    return {nullptr,
            {CanonicalSyncProblemError::InvalidMechanism, std::nullopt}};
  }
  const bool strictMinimalDirect =
      options.patterns.catalogMode ==
      CanonicalSyncCatalogMode::StrictMinimalDirect;
  const bool mechanicalDirect = options.patterns.catalogMode ==
                                CanonicalSyncCatalogMode::MechanicalDirect;
  const bool directOnlyCatalog = strictMinimalDirect || mechanicalDirect;
  const bool invalidStrictConfiguration =
      directOnlyCatalog &&
      (familyMask != 0 || options.patterns.enableConflictCoreRepair ||
       kind != CandidateCatalogKind::Precise ||
       options.patterns.enableDirectPairs);
  if (invalidStrictConfiguration) {
    program.getFunction().emitError(
        "canonical sync direct-only catalog requires a core family mask, "
        "disabled repair, precise construction, and direct-only patterns");
    return {nullptr,
            {CanonicalSyncProblemError::InvalidMechanism, std::nullopt}};
  }
  const auto familyEnabled = [&](CanonicalSyncMechanismFamily family) {
    return canonicalSyncMechanismFamilyEnabled(familyMask, family);
  };
  const bool minimalDirectCatalog =
      kind == CandidateCatalogKind::Precise && directOnlyCatalog;
  const std::vector<SyncCoverDemandId> obligations =
      getActiveDemands(program.getGraph());
  DemandBasisResult basis;
  if (minimalDirectCatalog) {
    // In the minimal correctness mode, the set-cover rows are the complete
    // hazard basis. Any reduction must come from grounded direct mechanisms
    // covering more than one row, not from a separate demand pre-reduction.
    basis.demands = obligations;
    basis.truncated = false;
  } else {
    const SyncCoverExpandedProgram basisExpansion(
        program.getGraph(), obligations, options.expansionLimits);
    basis =
        buildDemandSelectionBasis(program.getGraph(), basisExpansion, options);
  }
  auto problem = std::make_unique<CanonicalSyncPatternProblem>(
      program.getGraph(), obligations, basis.demands, options.problemLimits,
      options.expansionLimits, basis.truncated);
  const CanonicalSyncProblemResult configured =
      problem->setCandidateConfigurationSignature(
          getCandidateConfigurationSignature(options));
  if (!configured) {
    return {nullptr, configured};
  }
  if (directOnlyCatalog) {
    const CanonicalSyncTargetCapabilities &target =
        program.getTargetCapabilities();
    const CanonicalSyncProblemResult exactGrounding =
        problem->enableExactDirectCutGrounding(
            target.directEventOrderingRequirements,
            target.pipeBarrierOrderingRequirements,
            target.directEventCompletesSourcePrefix.isEnabled());
    if (!exactGrounding) {
      return {nullptr, exactGrounding};
    }
  }
  const SyncCoverCoverageResult baseline = computeSyncCoverCoverage(
      program.getGraph(), problem->getExpansion(), {}, problem->getDemands());
  if (!baseline) {
    program.getFunction().emitError(
        "cannot compute canonical sync fixed coverage");
    return {nullptr,
            {CanonicalSyncProblemError::CoverageFailure, std::nullopt}};
  }

  SyncCoverCoverageWorkBudget directAdmissionWork(
      options.selection.maximumWorkUnits);
  std::optional<SyncCoverDemandSet> directlyAdmittedDemands;
  if (directOnlyCatalog) {
    const std::size_t demandCount = program.getGraph().getDemands().size();
    const std::size_t demandWords =
        demandCount / 64 + (demandCount % 64 != 0 ? 1 : 0);
    if (!directAdmissionWork.consume(demandWords)) {
      program.getFunction().emitError(
          "canonical sync direct-only admission work limit exceeded");
      return {nullptr,
              {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
    }
    directlyAdmittedDemands.emplace(demandCount);
  }

  bool failedBuild = false;
  if (kind == CandidateCatalogKind::LocalizedPipeAll) {
    failedBuild =
        failed(addPipeAllBackstop(program, *problem, baseline.covered));
  } else if (minimalDirectCatalog) {
    std::map<EventDomainKey, CanonicalSyncEventDomainId> domainIds;
    std::vector<DirectEventRecord> directEvents;
    failedBuild =
        failed(addEventDomains(program, options.eventIdBudget, *problem,
                               baseline.covered, domainIds,
                               /*includeBasicOwnership=*/false)) ||
        failed(addDirectTargetedBarriers(
            program, *problem, baseline.covered,
            directlyAdmittedDemands ? &*directlyAdmittedDemands : nullptr,
            &directAdmissionWork)) ||
        failed(addExactEvents(
            program, *problem, baseline.covered, domainIds, directEvents,
            /*requireIndependentEventProtocol=*/true,
            directlyAdmittedDemands ? &*directlyAdmittedDemands : nullptr,
            &directAdmissionWork)) ||
        failed(requireCompleteDirectOnlyCatalog(
            program, *problem, baseline.covered, *directlyAdmittedDemands,
            directAdmissionWork));
  } else {
    std::map<EventDomainKey, CanonicalSyncEventDomainId> domainIds;
    std::vector<DirectEventRecord> directEvents;
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
        (familyEnabled(CanonicalSyncMechanismFamily::SourceLocalDrain) &&
         failed(addSourceLocalPipeDrains(program, *problem, baseline.covered,
                                         sameResourceObligations, false))) ||
        (familyEnabled(CanonicalSyncMechanismFamily::SourcePrefixDrain) &&
         failed(addSourcePrefixPipeDrains(program, *problem, baseline.covered,
                                          options.patterns,
                                          sameResourceObligations, false))) ||
        failed(addEventDomains(
            program, options.eventIdBudget, *problem, baseline.covered,
            domainIds,
            familyEnabled(CanonicalSyncMechanismFamily::L0OperandOwnership) ||
                familyEnabled(CanonicalSyncMechanismFamily::BasicOwnership) ||
                familyEnabled(
                    CanonicalSyncMechanismFamily::BoundaryOwnership) ||
                familyEnabled(
                    CanonicalSyncMechanismFamily::HierarchicalOwnership))) ||
        failed(addExactEvents(program, *problem, baseline.covered, domainIds,
                              directEvents)) ||
        ((familyEnabled(CanonicalSyncMechanismFamily::L0OperandOwnership) ||
          familyEnabled(CanonicalSyncMechanismFamily::BasicOwnership) ||
          familyEnabled(CanonicalSyncMechanismFamily::BoundaryOwnership) ||
          familyEnabled(CanonicalSyncMechanismFamily::HierarchicalOwnership)) &&
         failed(addOwnershipProtocols(
             program, *problem, domainIds,
             familyEnabled(CanonicalSyncMechanismFamily::L0OperandOwnership),
             familyEnabled(CanonicalSyncMechanismFamily::BasicOwnership),
             familyEnabled(CanonicalSyncMechanismFamily::BoundaryOwnership),
             familyEnabled(
                 CanonicalSyncMechanismFamily::HierarchicalOwnership)))) ||
        (familyEnabled(CanonicalSyncMechanismFamily::CompletionFrontier) &&
         failed(addCompletionFrontierEvents(program, *problem, baseline.covered,
                                            domainIds, directEvents))) ||
        (familyEnabled(
             CanonicalSyncMechanismFamily::TargetCompletionCertificate) &&
         failed(addTargetCompletionCertificateEvents(
             program, *problem, baseline.covered, domainIds, directEvents))) ||
        (familyEnabled(CanonicalSyncMechanismFamily::TargetLocalFence) &&
         failed(addTargetLocalFenceEvents(
             program, *problem, uncoveredBasisDemands, domainIds, false,
             TargetLocalCatalogMode::EventGroupsOnly)));
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
    if (!failedBuild &&
        familyEnabled(CanonicalSyncMechanismFamily::SourceLocalCompletion)) {
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
      const SyncCoverDemand &first =
          program.getGraph().getDemands()[singletonResidual.front()];
      std::array<std::size_t, 4> ownershipCounts{};
      for (const SyncCoverBasicOwnershipCertificate &certificate :
           program.getGraph().getBasicOwnershipCertificates()) {
        const std::size_t index =
            certificate.protocol ==
                    SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch
                ? 3
                : static_cast<std::size_t>(certificate.kind);
        ++ownershipCounts[index];
      }
      program.getFunction().emitError(
          "canonical sync required singleton catalog is incomplete: residual=")
          << singletonResidual.size()
          << ", first-demand=" << singletonResidual.front()
          << ", source-resource="
          << program.getGraph().getNodes()[first.source].resource
          << ", target-resource="
          << program.getGraph().getNodes()[first.target].resource
          << ", scope=" << first.scope << ", distance=" << first.distance
          << ", ownership-l0=" << ownershipCounts[0]
          << ", ownership-l1=" << ownershipCounts[1]
          << ", ownership-acc=" << ownershipCounts[2]
          << ", ownership-alternating=" << ownershipCounts[3];
      failedBuild = true;
    }
    if (!failedBuild &&
        familyEnabled(CanonicalSyncMechanismFamily::LoopCarryDrain)) {
      failedBuild = failed(addLoopCarryPipeDrains(
          program, *problem, baseline.covered, options.patterns));
    }
    if (!failedBuild &&
        familyEnabled(CanonicalSyncMechanismFamily::LoopBoundaryProtocol)) {
      failedBuild = failed(addLoopBoundarySourcePrefixProtocols(
          program, *problem, domainIds, options.patterns));
    }
    if (!failedBuild && options.patterns.enableDirectPairs) {
      failedBuild =
          failed(addDirectPairPatterns(program, *problem, options.directPairs));
    }
    if (!failedBuild) {
      for (const DirectEventRecord &event : directEvents) {
        const CanonicalSyncProblemResult recorded =
            problem->recordRepairEventSeed(
                {event.demand, event.mechanism, event.domain});
        if (!recorded) {
          failedBuild = true;
          break;
        }
      }
    }
  }
  if (failedBuild) {
    if (directOnlyCatalog && directAdmissionWork.exhausted) {
      return {nullptr,
              {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
    }
    return {nullptr,
            {CanonicalSyncProblemError::InvalidMechanism, std::nullopt}};
  }
  const CanonicalSyncMechanismOriginMask unclassified =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::Unclassified);
  const bool hasUnclassifiedMechanism = llvm::any_of(
      problem->getMechanisms(), [&](const CanonicalSyncMechanism &mechanism) {
        return (mechanism.originMask & unclassified) != 0;
      });
  if (hasUnclassifiedMechanism) {
    program.getFunction().emitError(
        "canonical sync production catalog contains unclassified mechanism");
    return {nullptr,
            {CanonicalSyncProblemError::InvalidMechanism, std::nullopt}};
  }
  if (problem->wasPatternGenerationTruncated()) {
    program.getFunction().emitRemark(
        "canonical sync pattern generation reached its bounded proposal "
        "limit; singleton candidates remain available");
  }
  const CanonicalSyncProblemResult frozen =
      problem->freeze(directOnlyCatalog ? &directAdmissionWork : nullptr);
  return {std::move(problem), frozen, {}, {}, {}};
}

CanonicalSyncProblemBuildResult buildRepairCatalogFromPrefix(
    const CanonicalSyncProgram &program,
    const CanonicalSyncPatternProblem &preciseProblem,
    const CanonicalSyncBuildOptions &options,
    ArrayRef<CanonicalSyncMechanismId> conflictCore,
    ArrayRef<CanonicalSyncMechanismId> selectedMechanisms,
    SyncCoverCoverageWorkBudget *workBudget,
    ArrayRef<CanonicalSyncRepairCriticalDemandSeed> retainedCriticalDemands) {
  const CanonicalSyncMechanismFamilyMask familyMask =
      options.patterns.enabledMechanismFamilies;
  const bool invalidConfiguration =
      options.eventIdBudget > kCompilerUsableEventIdCount ||
      (familyMask & ~kAllCanonicalSyncMechanismFamilies) != 0 ||
      options.patterns.catalogMode ==
          CanonicalSyncCatalogMode::StrictMinimalDirect ||
      options.patterns.catalogMode ==
          CanonicalSyncCatalogMode::MechanicalDirect ||
      !preciseProblem.isFrozen() ||
      preciseProblem.getCandidateConfigurationSignature() !=
          getCandidateConfigurationSignature(options);
  if (invalidConfiguration) {
    program.getFunction().emitError(
        "canonical sync repair core does not match the precise catalog");
    return {nullptr, {CanonicalSyncProblemError::InvalidPattern, std::nullopt}};
  }
  std::size_t validationWork = 0;
  const bool validationWorkAvailable =
      checkedBasisSum(conflictCore.size(), selectedMechanisms.size(),
                      validationWork) &&
      (!workBudget || workBudget->consume(validationWork));
  if (!validationWorkAvailable) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  const bool invalidOwner =
      llvm::any_of(conflictCore, [&](CanonicalSyncMechanismId mechanism) {
        return mechanism >= preciseProblem.getMechanisms().size();
      });
  const bool invalidSelectedMechanism =
      llvm::any_of(selectedMechanisms, [&](CanonicalSyncMechanismId mechanism) {
        return mechanism >= preciseProblem.getMechanisms().size();
      });
  if (invalidOwner || invalidSelectedMechanism) {
    program.getFunction().emitError(
        "canonical sync repair core does not match the precise catalog");
    return {nullptr, {CanonicalSyncProblemError::InvalidPattern, std::nullopt}};
  }
  if (workBudget && !workBudget->consume(conflictCore.size())) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  std::vector<CanonicalSyncMechanismId> sortedCore(conflictCore.begin(),
                                                   conflictCore.end());
  if (!stableSortAndUniqueRepairValues(sortedCore, workBudget)) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  if (sortedCore.empty()) {
    return {nullptr, {CanonicalSyncProblemError::InvalidPattern, std::nullopt}};
  }
  bool invalidRetainedProvenance = false;
  if (!retainedCriticalDemands.empty()) {
    std::optional<CanonicalSyncMechanismId> previousOwner;
    for (const CanonicalSyncRepairCriticalDemandSeed &seed :
         retainedCriticalDemands) {
      if (workBudget && !workBudget->consume()) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
      const CanonicalSyncMechanismId owner = seed.owner;
      const ArrayRef<SyncCoverDemandId> retainedDemands = seed.demands;
      const bool retainedIncidencesWithinLimit =
          retainedDemands.size() <=
          preciseProblem.getLimits().maximumIncidences;
      if (!retainedIncidencesWithinLimit) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
      const std::optional<bool> ownerInCore = meteredRepairContains(
          ArrayRef<CanonicalSyncMechanismId>(sortedCore), owner, workBudget);
      if (!ownerInCore) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
      bool invalidDemandOrder = false;
      bool invalidDemand = false;
      std::optional<SyncCoverDemandId> previousDemand;
      for (SyncCoverDemandId demand : retainedDemands) {
        if (workBudget && !workBudget->consume()) {
          return {nullptr,
                  {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
        }
        invalidDemandOrder |= previousDemand && *previousDemand >= demand;
        const std::optional<bool> demandIsActive = meteredRepairContains(
            ArrayRef<SyncCoverDemandId>(preciseProblem.getDemands()), demand,
            workBudget);
        if (!demandIsActive) {
          return {nullptr,
                  {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
        }
        invalidDemand |= !*demandIsActive;
        previousDemand = demand;
      }
      const bool invalidOwner =
          !*ownerInCore || (previousOwner && *previousOwner >= owner);
      invalidRetainedProvenance |=
          invalidOwner || invalidDemandOrder || invalidDemand;
      previousOwner = owner;
    }
  }
  if (invalidRetainedProvenance) {
    program.getFunction().emitError(
        "canonical sync retained repair provenance does not match the repair "
        "core");
    return {nullptr, {CanonicalSyncProblemError::InvalidPattern, std::nullopt}};
  }

  auto problem = preciseProblem.cloneMutableRepairPrefix(workBudget);
  if (!problem) {
    return {nullptr,
            {workBudget && workBudget->exhausted
                 ? CanonicalSyncProblemError::LimitExceeded
                 : CanonicalSyncProblemError::InvalidPattern,
             std::nullopt}};
  }
  const auto familyEnabled = [&](CanonicalSyncMechanismFamily family) {
    return canonicalSyncMechanismFamilyEnabled(familyMask, family);
  };
  const SyncCoverGraph &graph = program.getGraph();
  const std::size_t baselineWords =
      graph.getDemands().size() / 64 +
      static_cast<std::size_t>(graph.getDemands().size() % 64 != 0);
  if (workBudget && !workBudget->consume(baselineWords)) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  SyncCoverDemandSet baseline(graph.getDemands().size());
  for (std::size_t demand = 0; demand < preciseProblem.getDemands().size();
       ++demand) {
    if (workBudget && !workBudget->consume()) {
      return {nullptr,
              {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
    }
    if (preciseProblem.getBaselineCoverage().contains(demand)) {
      baseline.insert(preciseProblem.getDemands()[demand]);
    }
  }
  std::map<EventDomainKey, CanonicalSyncEventDomainId> domainIds;
  for (const CanonicalSyncEventDomain &domain : problem->getDomains()) {
    std::size_t domainMapEntries = 0;
    std::size_t domainMapWork = 0;
    const bool domainMapWorkAvailable =
        checkedBasisSum(domainIds.size(), 1, domainMapEntries) &&
        checkedOrderedLookupWork(domainMapEntries, domainMapWork) &&
        (!workBudget || workBudget->consume(domainMapWork));
    if (!domainMapWorkAvailable) {
      return {nullptr,
              {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
    }
    domainIds.emplace(
        EventDomainKey{domain.sourceResource, domain.targetResource},
        domain.id);
  }
  std::vector<DirectEventRecord> directEvents;
  if (workBudget &&
      !workBudget->consume(problem->getRepairEventSeeds().size())) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  directEvents.reserve(problem->getRepairEventSeeds().size());
  for (const CanonicalSyncRepairEventSeed &seed :
       problem->getRepairEventSeeds()) {
    if (workBudget && !workBudget->consume()) {
      return {nullptr,
              {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
    }
    directEvents.push_back({seed.demand, seed.mechanism, seed.domain});
  }

  RepairCriticalDemandResult critical = findRepairCriticalDemands(
      preciseProblem, sortedCore, selectedMechanisms,
      preciseProblem.getLimits().maximumIncidences, workBudget);
  if (critical.workExceeded) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  if (!retainedCriticalDemands.empty()) {
    for (const CanonicalSyncRepairCriticalDemandSeed &seed :
         retainedCriticalDemands) {
      if (workBudget && !workBudget->consume()) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
      const CanonicalSyncMechanismId owner = seed.owner;
      const ArrayRef<SyncCoverDemandId> retainedDemands = seed.demands;
      std::size_t criticalMapEntries = 0;
      std::size_t criticalMapWork = 0;
      const bool criticalMapWorkAvailable =
          checkedBasisSum(critical.demands.size(), 1, criticalMapEntries) &&
          checkedOrderedLookupWork(criticalMapEntries, criticalMapWork) &&
          (!workBudget || workBudget->consume(criticalMapWork));
      if (!criticalMapWorkAvailable) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
      auto [criticalPosition, inserted] = critical.demands.try_emplace(owner);
      (void)inserted;
      std::vector<SyncCoverDemandId> &demands = criticalPosition->second;
      std::size_t maximumUnionSize = 0;
      if (!checkedBasisSum(demands.size(), retainedDemands.size(),
                           maximumUnionSize)) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
      std::size_t current = 0;
      std::size_t retained = 0;
      std::size_t unionSize = 0;
      bool hasPendingDemand =
          current < demands.size() || retained < retainedDemands.size();
      while (hasPendingDemand) {
        if (workBudget && !workBudget->consume()) {
          return {nullptr,
                  {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
        }
        const bool takeCurrent = retained == retainedDemands.size() ||
                                 (current < demands.size() &&
                                  demands[current] < retainedDemands[retained]);
        const bool takeRetained =
            !takeCurrent && (current == demands.size() ||
                             retainedDemands[retained] < demands[current]);
        if (takeCurrent) {
          ++current;
        } else if (takeRetained) {
          ++retained;
        } else {
          ++current;
          ++retained;
        }
        ++unionSize;
        hasPendingDemand =
            current < demands.size() || retained < retainedDemands.size();
      }
      if (critical.incidences < demands.size()) {
        return {nullptr,
                {CanonicalSyncProblemError::InvalidPattern, std::nullopt}};
      }
      const std::size_t otherIncidences = critical.incidences - demands.size();
      if (otherIncidences > preciseProblem.getLimits().maximumIncidences ||
          unionSize >
              preciseProblem.getLimits().maximumIncidences - otherIncidences) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
      if (workBudget && !workBudget->consume(unionSize)) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
      std::vector<SyncCoverDemandId> merged;
      merged.reserve(unionSize);
      current = 0;
      retained = 0;
      hasPendingDemand =
          current < demands.size() || retained < retainedDemands.size();
      while (hasPendingDemand) {
        if (workBudget && !workBudget->consume()) {
          return {nullptr,
                  {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
        }
        const bool takeCurrent = retained == retainedDemands.size() ||
                                 (current < demands.size() &&
                                  demands[current] < retainedDemands[retained]);
        const bool takeRetained =
            !takeCurrent && (current == demands.size() ||
                             retainedDemands[retained] < demands[current]);
        if (takeCurrent) {
          merged.push_back(demands[current++]);
        } else if (takeRetained) {
          merged.push_back(retainedDemands[retained++]);
        } else {
          merged.push_back(demands[current]);
          ++current;
          ++retained;
        }
        hasPendingDemand =
            current < demands.size() || retained < retainedDemands.size();
      }
      demands = std::move(merged);
      critical.incidences = otherIncidences + unionSize;
    }
  }
  std::map<CanonicalSyncMechanismId, std::vector<CanonicalSyncMechanismId>>
      repairMechanismsByOwner;
  const std::size_t preciseMechanismCount =
      preciseProblem.getMechanisms().size();
  for (const auto &[owner, criticalDemands] : critical.demands) {
    (void)owner;
    if (workBudget && !workBudget->consume()) {
      return {nullptr,
              {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
    }
    const bool failedRepairMechanism =
        (familyEnabled(CanonicalSyncMechanismFamily::RepairSourceLocalDrain) &&
         failed(addSourceLocalPipeDrains(program, *problem, baseline,
                                         criticalDemands, true, workBudget))) ||
        (familyEnabled(CanonicalSyncMechanismFamily::RepairSourcePrefixDrain) &&
         failed(addSourcePrefixPipeDrains(program, *problem, baseline,
                                          options.patterns, criticalDemands,
                                          true, workBudget))) ||
        (familyEnabled(CanonicalSyncMechanismFamily::RepairTargetLocalDrain) &&
         failed(addTargetLocalFenceEvents(
             program, *problem, criticalDemands, domainIds, false,
             TargetLocalCatalogMode::PipeDrainsOnly, workBudget)));
    if (failedRepairMechanism) {
      return {nullptr,
              {workBudget && workBudget->exhausted
                   ? CanonicalSyncProblemError::LimitExceeded
                   : CanonicalSyncProblemError::InvalidMechanism,
               std::nullopt}};
    }
  }

  std::size_t demandIndexLookupWork = 0;
  std::size_t ownerMapLookupWork = 0;
  std::size_t demandIndexEntryWork = 0;
  std::size_t ownerMapEntryWork = 0;
  std::size_t criticalIndexWork = 0;
  const bool indexWorkAvailable =
      checkedOrderedLookupWork(critical.incidences, demandIndexLookupWork) &&
      checkedOrderedLookupWork(critical.demands.size(), ownerMapLookupWork) &&
      checkedBasisSum(demandIndexLookupWork, 1, demandIndexLookupWork) &&
      checkedBasisSum(ownerMapLookupWork, 1, ownerMapLookupWork) &&
      checkedBasisProduct(critical.incidences, demandIndexLookupWork,
                          demandIndexEntryWork) &&
      checkedBasisProduct(critical.demands.size(), ownerMapLookupWork,
                          ownerMapEntryWork) &&
      checkedBasisSum(demandIndexEntryWork, ownerMapEntryWork,
                      criticalIndexWork) &&
      checkedBasisSum(criticalIndexWork, preciseMechanismCount,
                      criticalIndexWork);
  if (!indexWorkAvailable ||
      (workBudget && !workBudget->consume(criticalIndexWork))) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  std::map<SyncCoverDemandId, std::vector<CanonicalSyncMechanismId>>
      ownersByDemand;
  std::vector<std::vector<CanonicalSyncMechanismId> *> ownerMechanisms(
      preciseMechanismCount, nullptr);
  for (const auto &[owner, criticalDemands] : critical.demands) {
    auto [position, inserted] = repairMechanismsByOwner.try_emplace(owner);
    (void)inserted;
    ownerMechanisms[owner] = &position->second;
    for (SyncCoverDemandId demand : criticalDemands) {
      ownersByDemand[demand].push_back(owner);
    }
  }

  const std::size_t repairMechanismCount =
      problem->getMechanisms().size() - preciseMechanismCount;
  if (workBudget && !workBudget->consume(repairMechanismCount)) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  std::size_t repairSupplyCount = 0;
  std::size_t attestedSupplyCount = 0;
  for (CanonicalSyncMechanismId mechanism = preciseMechanismCount;
       mechanism < problem->getMechanisms().size(); ++mechanism) {
    const auto &supplies =
        problem->getMechanisms()[mechanism].descriptor.supplies;
    if (workBudget && !workBudget->consume(supplies.size())) {
      return {nullptr,
              {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
    }
    if (!checkedBasisSum(repairSupplyCount, supplies.size(),
                         repairSupplyCount)) {
      return {nullptr,
              {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
    }
    for (const CanonicalSyncSupplyBinding &binding : supplies) {
      if (binding.attestedDemand &&
          !checkedBasisSum(attestedSupplyCount, 1, attestedSupplyCount)) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
    }
  }
  std::size_t demandLookupWork = 0;
  std::size_t attestedLookupWork = 0;
  std::size_t classificationWork = 0;
  const bool classificationWorkAvailable =
      checkedOrderedLookupWork(ownersByDemand.size(), demandLookupWork) &&
      checkedBasisProduct(attestedSupplyCount, demandLookupWork,
                          attestedLookupWork) &&
      checkedBasisSum(repairMechanismCount, repairSupplyCount,
                      classificationWork) &&
      checkedBasisSum(classificationWork, attestedLookupWork,
                      classificationWork) &&
      checkedBasisSum(classificationWork, critical.demands.size(),
                      classificationWork);
  if (!classificationWorkAvailable ||
      (workBudget && !workBudget->consume(classificationWork))) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  std::size_t ownerAssociations = 0;
  for (CanonicalSyncMechanismId mechanism = preciseMechanismCount;
       mechanism < problem->getMechanisms().size(); ++mechanism) {
    for (const CanonicalSyncSupplyBinding &binding :
         problem->getMechanisms()[mechanism].descriptor.supplies) {
      if (!binding.attestedDemand) {
        continue;
      }
      const auto owners = ownersByDemand.find(*binding.attestedDemand);
      if (owners == ownersByDemand.end()) {
        continue;
      }
      const std::size_t ownerCount = owners->second.size();
      const std::size_t maximumAssociations =
          preciseProblem.getLimits().maximumIncidences;
      if (ownerAssociations > maximumAssociations ||
          ownerCount > maximumAssociations - ownerAssociations ||
          (workBudget && !workBudget->consume(ownerCount))) {
        return {nullptr,
                {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
      }
      ownerAssociations += ownerCount;
      for (CanonicalSyncMechanismId owner : owners->second) {
        std::vector<CanonicalSyncMechanismId> &mechanisms =
            *ownerMechanisms[owner];
        const bool newOwnerMechanism =
            mechanisms.empty() || mechanisms.back() != mechanism;
        if (newOwnerMechanism) {
          mechanisms.push_back(mechanism);
        }
      }
    }
  }
  for (auto position = repairMechanismsByOwner.begin();
       position != repairMechanismsByOwner.end();) {
    if (position->second.empty()) {
      position = repairMechanismsByOwner.erase(position);
    } else {
      ++position;
    }
  }

  const std::size_t frontierMechanismBegin = problem->getMechanisms().size();
  RepairFrontierBuildResult frontier;
  if (familyEnabled(CanonicalSyncMechanismFamily::RepairFrontier)) {
    frontier =
        addRepairFrontierPatterns(program, *problem, directEvents, sortedCore,
                                  options.patterns, workBudget);
  }
  if (workBudget && workBudget->exhausted) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  if (frontier.status == RepairFrontierBuildStatus::Failed) {
    return {nullptr,
            {CanonicalSyncProblemError::InvalidMechanism, std::nullopt}};
  }
  if (frontier.status == RepairFrontierBuildStatus::Truncated) {
    const CanonicalSyncProblemResult recorded =
        problem->recordRepairFrontierGeneration(frontier.inspections,
                                                frontier.proposals, true);
    if (!recorded) {
      return {nullptr, recorded};
    }
    const CanonicalSyncProblemResult frozen = problem->freeze();
    return {std::move(problem),
            frozen,
            std::move(repairMechanismsByOwner),
            std::move(critical.demands),
            {}};
  }
  std::vector<CanonicalSyncMechanismId> collectiveRepairMechanisms;
  const std::size_t collectiveMechanismCount =
      problem->getMechanisms().size() - frontierMechanismBegin;
  if (workBudget && !workBudget->consume(collectiveMechanismCount)) {
    return {nullptr, {CanonicalSyncProblemError::LimitExceeded, std::nullopt}};
  }
  collectiveRepairMechanisms.reserve(collectiveMechanismCount);
  for (CanonicalSyncMechanismId mechanism = frontierMechanismBegin;
       mechanism < problem->getMechanisms().size(); ++mechanism) {
    collectiveRepairMechanisms.push_back(mechanism);
  }
  const CanonicalSyncProblemResult recorded =
      problem->recordRepairFrontierGeneration(frontier.inspections,
                                              frontier.proposals, false);
  if (!recorded) {
    return {nullptr, recorded};
  }
  const CanonicalSyncProblemResult frozen = problem->freeze();
  return {std::move(problem), frozen, std::move(repairMechanismsByOwner),
          std::move(critical.demands), std::move(collectiveRepairMechanisms)};
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
    const std::vector<CanonicalSyncMechanismId> &selectedMechanisms,
    SyncCoverCoverageWorkBudget *workBudget,
    ArrayRef<CanonicalSyncRepairCriticalDemandSeed> retainedCriticalDemands) {
  return buildRepairCatalogFromPrefix(program, preciseProblem, options,
                                      conflictCore, selectedMechanisms,
                                      workBudget, retainedCriticalDemands);
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
