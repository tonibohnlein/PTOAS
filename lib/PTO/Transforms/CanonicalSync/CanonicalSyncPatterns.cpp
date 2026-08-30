// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::pto;

namespace {

constexpr std::size_t kBitsPerCoverageWord = 64;
constexpr std::size_t kPairBatchScratchRows = 5;

bool checkedAdd(std::size_t first, std::size_t second, std::size_t &result) {
  const bool sumOverflows =
      second > std::numeric_limits<std::size_t>::max() - first;
  if (sumOverflows) {
    return false;
  }
  result = first + second;
  return true;
}

bool checkedMultiply(std::size_t first, std::size_t second,
                     std::size_t &result) {
  const bool productOverflows =
      first != 0 && second > std::numeric_limits<std::size_t>::max() / first;
  if (productOverflows) {
    return false;
  }
  result = first * second;
  return true;
}

std::size_t coverageWords(std::size_t bits) {
  return bits / kBitsPerCoverageWord +
         (bits % kBitsPerCoverageWord != 0 ? 1 : 0);
}

bool countSingletonCoverageWords(
    const SyncCoverSingletonCoverageResult &singleton, std::size_t &words) {
  words = singleton.baseline.getWords().size();
  for (const SyncCoverDemandSet &coverage : singleton.mechanisms) {
    if (!checkedAdd(words, coverage.getWords().size(), words)) {
      return false;
    }
  }
  return true;
}

bool isPairCapableBinding(const CanonicalSyncSupplyBinding &binding) {
  return binding.allowedDemands.empty();
}

bool isDirectPairMember(const CanonicalSyncMechanism &mechanism) {
  return std::any_of(mechanism.descriptor.supplies.begin(),
                     mechanism.descriptor.supplies.end(), isPairCapableBinding);
}

std::optional<SyncCoverScopeId>
getMechanismOwner(const SyncCoverGraph &graph,
                  const CanonicalSyncMechanism &mechanism) {
  std::optional<SyncCoverScopeId> owner;
  for (const CanonicalSyncSupplyBinding &binding :
       mechanism.descriptor.supplies) {
    if (!isPairCapableBinding(binding)) {
      continue;
    }
    if (!owner) {
      owner = binding.edge.scope;
      continue;
    }
    owner = graph.getLowestCommonScope(*owner, binding.edge.scope);
    if (!owner) {
      return std::nullopt;
    }
  }
  return owner;
}

struct ConnectorEndpoint {
  CanonicalSyncMechanismId mechanism = 0;
  SyncCoverNodeId node = 0;
  SyncCoverNodeId outerNode = 0;
  std::size_t distance = 0;
  std::uint32_t outerResource = 0;
  CanonicalSyncMechanismKind kind = CanonicalSyncMechanismKind::Event;
  CanonicalSyncSupplyProof proof = CanonicalSyncSupplyProof::DirectAction;
  const SyncCoverGuard *guard = nullptr;
};

using ConnectorOwnerIndex =
    std::map<SyncCoverScopeId, std::vector<ConnectorEndpoint>>;
using ConnectorNodeIndex = std::map<SyncCoverNodeId, ConnectorOwnerIndex>;
using FixedConnectorState = std::size_t;

struct FixedConnectorAutomaton {
  std::size_t nodes = 0;
  std::vector<std::vector<FixedConnectorState>> successors;
};
using PlausibleDemandEndpoint =
    std::tuple<SyncCoverNodeId, SyncCoverNodeId, std::size_t>;

struct MechanismPairLess {
  bool operator()(const SyncCoverMechanismPair &first,
                  const SyncCoverMechanismPair &second) const {
    return std::tie(first.first, first.second) <
           std::tie(second.first, second.second);
  }
};

struct OwnedPairProposals {
  std::set<SyncCoverMechanismPair, MechanismPairLess> pairs;
  std::size_t proposalCount = 0;
  bool truncated = false;
};

bool mechanismsConflict(
    const std::vector<CanonicalSyncMechanism> &mechanisms,
    const SyncCoverMechanismPair &pair) {
  if (pair.first >= mechanisms.size() || pair.second >= mechanisms.size()) {
    return true;
  }
  const auto hasConflict = [&](CanonicalSyncMechanismId first,
                               CanonicalSyncMechanismId second) {
    const auto &conflicts = mechanisms[first].conflicts;
    return std::binary_search(conflicts.begin(), conflicts.end(), second);
  };
  return hasConflict(pair.first, pair.second) ||
         hasConflict(pair.second, pair.first);
}

bool indexMechanismConnectors(
    const SyncCoverGraph &graph, const CanonicalSyncMechanism &mechanism,
    SyncCoverScopeId owner, std::size_t maximumEntries, std::size_t &entryCount,
    ConnectorNodeIndex &sources, ConnectorNodeIndex &targets) {
  for (const CanonicalSyncSupplyBinding &binding :
       mechanism.descriptor.supplies) {
    if (!isPairCapableBinding(binding)) {
      continue;
    }
    const SyncCoverEdge &edge = binding.edge;
    const bool invalidEndpoint = edge.source >= graph.getNodes().size() ||
                                 edge.target >= graph.getNodes().size();
    if (invalidEndpoint) {
      continue;
    }
    const bool entryLimitExceeded =
        entryCount > maximumEntries || maximumEntries - entryCount < 2;
    if (entryLimitExceeded) {
      return false;
    }
    sources[edge.source][owner].push_back(
        {mechanism.id, edge.source, edge.target, edge.distance,
         graph.getNodes()[edge.target].resource, mechanism.descriptor.kind,
         binding.proof, &edge.sourceGuard});
    targets[edge.target][owner].push_back(
        {mechanism.id, edge.target, edge.source, edge.distance,
         graph.getNodes()[edge.source].resource, mechanism.descriptor.kind,
         binding.proof, &edge.targetGuard});
    entryCount += 2;
  }
  return true;
}

bool endpointLess(const ConnectorEndpoint &first,
                  const ConnectorEndpoint &second) {
  return std::tie(first.mechanism, first.node, first.outerNode, first.distance,
                  first.outerResource, first.kind, first.proof,
                  first.guard->literals) <
         std::tie(second.mechanism, second.node, second.outerNode,
                  second.distance, second.outerResource, second.kind,
                  second.proof, second.guard->literals);
}

bool endpointEqual(const ConnectorEndpoint &first,
                   const ConnectorEndpoint &second) {
  return first.mechanism == second.mechanism && first.node == second.node &&
         first.outerNode == second.outerNode &&
         first.distance == second.distance &&
         first.outerResource == second.outerResource &&
         first.kind == second.kind && first.proof == second.proof &&
         first.guard->literals == second.guard->literals;
}

void canonicalizeConnectorIndex(ConnectorNodeIndex &index) {
  for (auto &[node, owners] : index) {
    (void)node;
    for (auto &[owner, endpoints] : owners) {
      (void)owner;
      std::sort(endpoints.begin(), endpoints.end(), endpointLess);
      endpoints.erase(
          std::unique(endpoints.begin(), endpoints.end(), endpointEqual),
          endpoints.end());
    }
  }
}

bool consumeConnectorInspection(std::size_t maximum, std::size_t &inspections) {
  if (inspections == maximum) {
    return false;
  }
  ++inspections;
  return true;
}

FixedConnectorState fixedConnectorState(SyncCoverNodeId node, bool completed) {
  return static_cast<std::size_t>(node) * 2 + (completed ? 1 : 0);
}

SyncCoverNodeId fixedConnectorNode(FixedConnectorState state) {
  return static_cast<SyncCoverNodeId>(state / 2);
}

bool fixedConnectorCompleted(FixedConnectorState state) {
  return state % 2 != 0;
}

FixedConnectorAutomaton
buildFixedConnectorAutomaton(const SyncCoverGraph &graph) {
  FixedConnectorAutomaton result;
  result.nodes = graph.getNodes().size();
  if (result.nodes > std::numeric_limits<std::size_t>::max() / 2) {
    return result;
  }
  result.successors.resize(result.nodes * 2);
  const auto addTransition = [&](SyncCoverNodeId source, bool sourceCompleted,
                                 SyncCoverNodeId target, bool targetCompleted) {
    result.successors[fixedConnectorState(source, sourceCompleted)].push_back(
        fixedConnectorState(target, targetCompleted));
  };
  for (const SyncCoverEdge &edge : graph.getEdges()) {
    if (edge.distance != 0 || edge.source >= result.nodes ||
        edge.target >= result.nodes) {
      continue;
    }
    switch (edge.kind) {
    case SyncCoverEdgeKind::CertifiedCompletionFrontier:
      addTransition(edge.source, false, edge.target, false);
      addTransition(edge.source, true, edge.target, true);
      break;
    case SyncCoverEdgeKind::CompletionPreservingIssueOrder:
    case SyncCoverEdgeKind::NonCompletionPreservingIssueOrder:
      addTransition(edge.source, true, edge.target, true);
      break;
    case SyncCoverEdgeKind::CompletionSupply:
      addTransition(edge.source, false, edge.target, true);
      addTransition(edge.source, true, edge.target, true);
      break;
    }
  }
  for (std::vector<FixedConnectorState> &successors : result.successors) {
    std::sort(successors.begin(), successors.end());
    successors.erase(std::unique(successors.begin(), successors.end()),
                     successors.end());
  }
  return result;
}

bool findFixedConnectorStates(const FixedConnectorAutomaton &automaton,
                              SyncCoverNodeId start, bool startsCompleted,
                              std::size_t maximumInspections,
                              std::size_t &inspections,
                              std::vector<FixedConnectorState> &reachable) {
  reachable.clear();
  if (start >= automaton.nodes) {
    return true;
  }
  std::vector<std::uint8_t> visited(automaton.successors.size(), 0);
  std::vector<FixedConnectorState> ready{
      fixedConnectorState(start, startsCompleted)};
  while (!ready.empty()) {
    const FixedConnectorState current = ready.back();
    ready.pop_back();
    if (visited[current] != 0) {
      continue;
    }
    visited[current] = 1;
    reachable.push_back(current);
    for (FixedConnectorState successor : automaton.successors[current]) {
      if (!consumeConnectorInspection(maximumInspections, inspections)) {
        return false;
      }
      if (visited[successor] == 0) {
        ready.push_back(successor);
      }
    }
  }
  std::sort(reachable.begin(), reachable.end());
  return true;
}

FixedConnectorAutomaton
reverseFixedConnectorAutomaton(const FixedConnectorAutomaton &automaton) {
  FixedConnectorAutomaton result;
  result.nodes = automaton.nodes;
  result.successors.resize(automaton.successors.size());
  for (FixedConnectorState source = 0; source < automaton.successors.size();
       ++source) {
    for (FixedConnectorState target : automaton.successors[source]) {
      result.successors[target].push_back(source);
    }
  }
  for (std::vector<FixedConnectorState> &predecessors : result.successors) {
    std::sort(predecessors.begin(), predecessors.end());
    predecessors.erase(std::unique(predecessors.begin(), predecessors.end()),
                       predecessors.end());
  }
  return result;
}

bool buildPlausibleDemandEndpoints(
    const FixedConnectorAutomaton &automaton, const ConnectorNodeIndex &sources,
    const ConnectorNodeIndex &targets,
    const std::set<PlausibleDemandEndpoint> &activeDemands,
    std::size_t maximumInspections, std::size_t &inspections,
    std::set<PlausibleDemandEndpoint> &plausible) {
  if (activeDemands.empty()) {
    return true;
  }
  std::set<SyncCoverNodeId> outerSources;
  std::set<SyncCoverNodeId> outerTargets;
  for (const auto &[node, owners] : targets) {
    (void)node;
    for (const auto &[owner, endpoints] : owners) {
      (void)owner;
      for (const ConnectorEndpoint &endpoint : endpoints) {
        outerSources.insert(endpoint.outerNode);
      }
    }
  }
  for (const auto &[node, owners] : sources) {
    (void)node;
    for (const auto &[owner, endpoints] : owners) {
      (void)owner;
      for (const ConnectorEndpoint &endpoint : endpoints) {
        outerTargets.insert(endpoint.outerNode);
      }
    }
  }
  const FixedConnectorAutomaton reverse =
      reverseFixedConnectorAutomaton(automaton);
  std::map<SyncCoverNodeId, std::vector<SyncCoverNodeId>> forwardCache;
  std::map<SyncCoverNodeId, std::vector<SyncCoverNodeId>> backwardCache;
  std::vector<FixedConnectorState> reachable;
  for (const auto &[source, target, distance] : activeDemands) {
    auto forward = forwardCache.find(source);
    if (forward == forwardCache.end()) {
      if (!findFixedConnectorStates(automaton, source, false,
                                    maximumInspections, inspections,
                                    reachable)) {
        return false;
      }
      std::vector<SyncCoverNodeId> filtered;
      for (FixedConnectorState state : reachable) {
        const SyncCoverNodeId node = fixedConnectorNode(state);
        if (outerSources.find(node) != outerSources.end()) {
          filtered.push_back(node);
        }
      }
      std::sort(filtered.begin(), filtered.end());
      filtered.erase(std::unique(filtered.begin(), filtered.end()),
                     filtered.end());
      forward = forwardCache.emplace(source, std::move(filtered)).first;
    }
    auto backward = backwardCache.find(target);
    if (backward == backwardCache.end()) {
      if (!findFixedConnectorStates(reverse, target, true, maximumInspections,
                                    inspections, reachable)) {
        return false;
      }
      std::vector<SyncCoverNodeId> filtered;
      for (FixedConnectorState state : reachable) {
        const SyncCoverNodeId node = fixedConnectorNode(state);
        if (fixedConnectorCompleted(state) &&
            outerTargets.find(node) != outerTargets.end()) {
          filtered.push_back(node);
        }
      }
      std::sort(filtered.begin(), filtered.end());
      filtered.erase(std::unique(filtered.begin(), filtered.end()),
                     filtered.end());
      backward = backwardCache.emplace(target, std::move(filtered)).first;
    }
    for (SyncCoverNodeId outerSource : forward->second) {
      for (SyncCoverNodeId outerTarget : backward->second) {
        if (!consumeConnectorInspection(maximumInspections, inspections)) {
          return false;
        }
        plausible.insert({outerSource, outerTarget, distance});
      }
    }
  }
  return true;
}

bool addConnectorGroup(
    const std::vector<CanonicalSyncMechanism> &mechanisms,
    const std::vector<ConnectorEndpoint> &targets,
    const std::vector<ConnectorEndpoint> &sources,
    const std::set<std::pair<std::uint32_t, std::uint32_t>>
        &activeDemandResourcePairs,
    const std::set<PlausibleDemandEndpoint> &activeDemandEndpoints,
    std::size_t maximumProposals, std::size_t maximumInspections,
    std::size_t &inspections, OwnedPairProposals &proposals) {
  if (proposals.truncated) {
    return true;
  }
  for (const ConnectorEndpoint &target : targets) {
    for (const ConnectorEndpoint &source : sources) {
      if (!consumeConnectorInspection(maximumInspections, inspections)) {
        return false;
      }
      const bool plausibleResourceChain =
          activeDemandResourcePairs.empty() ||
          activeDemandResourcePairs.find(
              {target.outerResource, source.outerResource}) !=
              activeDemandResourcePairs.end();
      const bool distanceOverflows =
          target.distance >
          std::numeric_limits<std::size_t>::max() - source.distance;
      const std::size_t combinedDistance =
          distanceOverflows ? 0 : target.distance + source.distance;
      const bool plausibleEndpoints =
          activeDemandEndpoints.empty() ||
          (!distanceOverflows &&
           activeDemandEndpoints.find(
               {target.outerNode, source.outerNode, combinedDistance}) !=
               activeDemandEndpoints.end());
      if (!plausibleResourceChain || !plausibleEndpoints ||
          target.mechanism == source.mechanism ||
          !syncCoverGuardsCompatible(*target.guard, *source.guard)) {
        continue;
      }
      const auto members = std::minmax(target.mechanism, source.mechanism);
      const SyncCoverMechanismPair pair{members.first, members.second};
      if (mechanismsConflict(mechanisms, pair)) {
        continue;
      }
      proposals.pairs.insert(pair);
      if (proposals.pairs.size() > maximumProposals) {
        proposals.proposalCount =
            maximumProposals == std::numeric_limits<std::size_t>::max()
                ? maximumProposals
                : maximumProposals + 1;
        proposals.pairs.clear();
        proposals.truncated = true;
      } else {
        proposals.proposalCount = proposals.pairs.size();
      }
    }
  }
  return true;
}

} // namespace

CanonicalSyncProblemResult mlir::pto::addCanonicalSyncDirectPairPatterns(
    CanonicalSyncPatternProblem &problem,
    CanonicalSyncDirectPairOptions options) {
  if (problem.isFrozen()) {
    return {CanonicalSyncProblemError::Frozen, std::nullopt};
  }
  const SyncCoverGraph &graph = problem.getGraph();
  std::set<std::pair<std::uint32_t, std::uint32_t>> activeDemandResourcePairs;
  std::set<PlausibleDemandEndpoint> activeDemandRows;
  for (SyncCoverDemandId demandId : problem.getDemands()) {
    if (demandId >= graph.getDemands().size()) {
      return {CanonicalSyncProblemError::InvalidGraph, std::nullopt};
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    activeDemandResourcePairs.insert(
        {graph.getNodes()[demand.source].resource,
         graph.getNodes()[demand.target].resource});
    activeDemandRows.insert({demand.source, demand.target, demand.distance});
  }
  ConnectorNodeIndex sources;
  ConnectorNodeIndex targets;
  std::size_t connectorIndexEntries = 0;
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    if (!isDirectPairMember(mechanism)) {
      continue;
    }
    const std::optional<SyncCoverScopeId> owner =
        getMechanismOwner(graph, mechanism);
    const bool indexed = !owner || indexMechanismConnectors(
                                       graph, mechanism, *owner,
                                       options.maximumConnectorIndexEntries,
                                       connectorIndexEntries, sources, targets);
    if (!indexed) {
      problem.markPatternGenerationTruncated();
      problem.recordDirectPairGeneration(0, 0, 0);
      return {CanonicalSyncProblemError::None, 0};
    }
  }
  canonicalizeConnectorIndex(sources);
  canonicalizeConnectorIndex(targets);
  const FixedConnectorAutomaton fixedConnectors =
      buildFixedConnectorAutomaton(graph);
  std::map<SyncCoverScopeId, OwnedPairProposals> indexedProposals;
  std::size_t connectorInspections = 0;
  std::set<PlausibleDemandEndpoint> activeDemandEndpoints;
  if (!buildPlausibleDemandEndpoints(
          fixedConnectors, sources, targets, activeDemandRows,
          options.maximumConnectorInspections, connectorInspections,
          activeDemandEndpoints)) {
    problem.markPatternGenerationTruncated();
    problem.recordDirectPairGeneration(0, 0, connectorInspections);
    return {CanonicalSyncProblemError::None, 0};
  }
  bool connectorLimitExceeded = false;
  std::vector<FixedConnectorState> reachable;
  for (const auto &[targetNode, targetOwners] : targets) {
    if (!findFixedConnectorStates(fixedConnectors, targetNode, true,
                                  options.maximumConnectorInspections,
                                  connectorInspections, reachable)) {
      connectorLimitExceeded = true;
      break;
    }
    for (FixedConnectorState state : reachable) {
      if (!fixedConnectorCompleted(state)) {
        continue;
      }
      const SyncCoverNodeId sourceNode = fixedConnectorNode(state);
      const auto sourcePosition = sources.find(sourceNode);
      if (sourcePosition == sources.end()) {
        continue;
      }
      for (const auto &[targetOwner, targetEndpoints] : targetOwners) {
        for (const auto &[sourceOwner, sourceEndpoints] :
             sourcePosition->second) {
          if (!consumeConnectorInspection(options.maximumConnectorInspections,
                                          connectorInspections)) {
            connectorLimitExceeded = true;
            break;
          }
          const std::optional<SyncCoverScopeId> pairOwner =
              graph.getLowestCommonScope(targetOwner, sourceOwner);
          if (!pairOwner) {
            continue;
          }
          OwnedPairProposals &proposals = indexedProposals[*pairOwner];
          const bool inspected = addConnectorGroup(
              problem.getMechanisms(), targetEndpoints, sourceEndpoints,
              activeDemandResourcePairs, activeDemandEndpoints,
              options.maximumEvaluationsPerScope,
              options.maximumConnectorInspections, connectorInspections,
              proposals);
          if (!inspected) {
            connectorLimitExceeded = true;
            break;
          }
        }
        if (connectorLimitExceeded) {
          break;
        }
      }
      if (connectorLimitExceeded) {
        break;
      }
    }
  }
  if (connectorLimitExceeded) {
    problem.markPatternGenerationTruncated();
    problem.recordDirectPairGeneration(0, 0, connectorInspections);
    return {CanonicalSyncProblemError::None, 0};
  }

  std::map<SyncCoverScopeId, std::vector<SyncCoverMechanismPair>> byOwner;
  std::size_t proposalCount = 0;
  for (auto &[owner, proposals] : indexedProposals) {
    const bool proposalCountOverflows =
        proposals.proposalCount >
        std::numeric_limits<std::size_t>::max() - proposalCount;
    if (proposalCountOverflows) {
      problem.markPatternGenerationTruncated();
      proposalCount = std::numeric_limits<std::size_t>::max();
    } else {
      proposalCount += proposals.proposalCount;
    }
    if (proposals.truncated) {
      problem.markPatternGenerationTruncated();
      byOwner.try_emplace(owner);
      continue;
    }
    std::vector<SyncCoverMechanismPair> conflictFree;
    conflictFree.reserve(proposals.pairs.size());
    for (const SyncCoverMechanismPair &pair : proposals.pairs) {
      if (!mechanismsConflict(problem.getMechanisms(), pair)) {
        conflictFree.push_back(pair);
      }
    }
    byOwner.emplace(owner, std::move(conflictFree));
  }

  std::vector<SyncCoverCompletionSupply> supplies;
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    for (const CanonicalSyncSupplyBinding &binding :
         mechanism.descriptor.supplies) {
      supplies.push_back({mechanism.id, binding.edge, binding.allowedDemands,
                          binding.completionExport ==
                              CanonicalSyncSupplyExport::ScopeExitAfterDrain,
                          binding.applicability});
    }
  }
  SyncCoverCoverageLimits singletonLimits;
  singletonLimits.maximumTotalWords = std::min(
      singletonLimits.maximumTotalWords, options.maximumPreparationWords);
  const SyncCoverSingletonCoverageResult singleton =
      computeSyncCoverSingletonCoverage(
          graph, problem.getExpansion(), problem.getMechanisms().size(),
          supplies, problem.getDemands(), singletonLimits);
  if (singleton.error == SyncCoverCoverageError::LimitExceeded) {
    problem.markPatternGenerationTruncated();
    problem.recordDirectPairGeneration(proposalCount, 0, connectorInspections);
    return {CanonicalSyncProblemError::None, 0};
  }
  if (!singleton) {
    return {CanonicalSyncProblemError::CoverageFailure, std::nullopt};
  }

  std::size_t singletonWords = 0;
  std::size_t batchScratchWords = 0;
  std::size_t retainedPreparationWords = 0;
  const bool preparationLimitExceeded =
      !countSingletonCoverageWords(singleton, singletonWords) ||
      !checkedMultiply(coverageWords(problem.getDemands().size()),
                       kPairBatchScratchRows, batchScratchWords) ||
      !checkedAdd(singletonWords, batchScratchWords,
                  retainedPreparationWords) ||
      retainedPreparationWords > options.maximumPreparationWords;
  if (preparationLimitExceeded) {
    problem.markPatternGenerationTruncated();
    problem.recordDirectPairGeneration(proposalCount, 0, connectorInspections);
    return {CanonicalSyncProblemError::None, 0};
  }
  const std::size_t availablePairWords =
      options.maximumPreparationWords - retainedPreparationWords;

  std::size_t addedCount = 0;
  std::size_t evaluationCount = 0;
  for (const auto &[owner, owned] : byOwner) {
    (void)owner;
    if (owned.empty()) {
      continue;
    }
    SyncCoverCoverageLimits pairLimits = options.pairCoverageLimits;
    pairLimits.maximumTotalWords =
        std::min(pairLimits.maximumTotalWords, availablePairWords);
    const SyncCoverPairCoverageResult joint = computeSyncCoverPairCoverage(
        graph, problem.getExpansion(), problem.getMechanisms().size(), supplies,
        owned, problem.getDemands(), pairLimits);
    if (joint.error == SyncCoverCoverageError::LimitExceeded) {
      problem.markPatternGenerationTruncated();
      continue;
    }
    if (!joint) {
      return {CanonicalSyncProblemError::CoverageFailure, std::nullopt};
    }
    const bool evaluationCountOverflows =
        owned.size() >
        std::numeric_limits<std::size_t>::max() - evaluationCount;
    if (evaluationCountOverflows) {
      evaluationCount = std::numeric_limits<std::size_t>::max();
    } else {
      evaluationCount += owned.size();
    }
    const CanonicalSyncProblemResult added =
        problem.addDirectPairBatch(owned, joint.pairs, singleton.mechanisms);
    if (!added) {
      return {added.error, addedCount};
    }
    if (added.index) {
      const bool addedCountOverflows =
          *added.index > std::numeric_limits<std::size_t>::max() - addedCount;
      if (addedCountOverflows) {
        return {CanonicalSyncProblemError::ArithmeticOverflow, addedCount};
      }
      addedCount += *added.index;
    }
  }
  problem.recordDirectPairGeneration(proposalCount, evaluationCount,
                                     connectorInspections);
  return {CanonicalSyncProblemError::None, addedCount};
}

CanonicalSyncProblemResult
mlir::pto::addCanonicalSyncFeasiblePattern(CanonicalSyncPatternProblem &problem,
                                           CanonicalSyncPatternSpec pattern) {
  if (problem.isFrozen()) {
    return {CanonicalSyncProblemError::Frozen, std::nullopt};
  }
  std::sort(pattern.members.begin(), pattern.members.end());
  pattern.members.erase(
      std::unique(pattern.members.begin(), pattern.members.end()),
      pattern.members.end());
  const bool hasCompositeMembers = pattern.members.size() >= 2;
  if (!hasCompositeMembers) {
    return {CanonicalSyncProblemError::InvalidPattern, std::nullopt};
  }
  const bool withinMemberLimit =
      pattern.members.size() <= problem.getLimits().maximumMembersPerPattern;
  if (!withinMemberLimit) {
    return {CanonicalSyncProblemError::None, std::nullopt};
  }
  if (std::any_of(pattern.members.begin(), pattern.members.end(),
                  [&](CanonicalSyncMechanismId member) {
                    return member >= problem.getMechanisms().size();
                  })) {
    return {CanonicalSyncProblemError::InvalidPattern, std::nullopt};
  }
  return problem.addPattern(std::move(pattern));
}
