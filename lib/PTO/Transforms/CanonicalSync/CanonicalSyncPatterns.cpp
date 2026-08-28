// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

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

bool isDirectPairMember(const CanonicalSyncMechanism &mechanism) {
  return mechanism.descriptor.selectionTier ==
             CanonicalSyncSelectionTier::Precise &&
         !mechanism.descriptor.supplies.empty() &&
         std::all_of(mechanism.descriptor.supplies.begin(),
                     mechanism.descriptor.supplies.end(), [](const auto &item) {
                       return item.allowedDemands.empty();
                     });
}

std::optional<SyncCoverScopeId>
getMechanismOwner(const SyncCoverGraph &graph,
                  const CanonicalSyncMechanism &mechanism) {
  std::optional<SyncCoverScopeId> owner;
  for (const CanonicalSyncSupplyBinding &binding :
       mechanism.descriptor.supplies) {
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
  SyncCoverGuard guard;
};

using ConnectorOwnerIndex =
    std::map<SyncCoverScopeId, std::vector<ConnectorEndpoint>>;
using ConnectorResourceIndex = std::map<std::uint32_t, ConnectorOwnerIndex>;

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

void indexMechanismConnectors(const SyncCoverGraph &graph,
                              const CanonicalSyncMechanism &mechanism,
                              SyncCoverScopeId owner,
                              ConnectorResourceIndex &sources,
                              ConnectorResourceIndex &targets) {
  for (const CanonicalSyncSupplyBinding &binding :
       mechanism.descriptor.supplies) {
    const SyncCoverEdge &edge = binding.edge;
    const bool invalidEndpoint = edge.source >= graph.getNodes().size() ||
                                 edge.target >= graph.getNodes().size();
    if (invalidEndpoint) {
      continue;
    }
    const SyncCoverNode &source = graph.getNodes()[edge.source];
    const SyncCoverNode &target = graph.getNodes()[edge.target];
    sources[source.resource][owner].push_back({mechanism.id, edge.sourceGuard});
    targets[target.resource][owner].push_back({mechanism.id, edge.targetGuard});
  }
}

void addConnectorGroup(const std::vector<ConnectorEndpoint> &targets,
                       const std::vector<ConnectorEndpoint> &sources,
                       std::size_t maximumProposals,
                       OwnedPairProposals &proposals) {
  if (proposals.truncated) {
    return;
  }
  for (const ConnectorEndpoint &target : targets) {
    for (const ConnectorEndpoint &source : sources) {
      if (target.mechanism == source.mechanism ||
          !syncCoverGuardsCompatible(target.guard, source.guard)) {
        continue;
      }
      const auto members = std::minmax(target.mechanism, source.mechanism);
      const bool inserted =
          proposals.pairs.insert({members.first, members.second}).second;
      if (!inserted) {
        continue;
      }
      proposals.proposalCount = proposals.pairs.size();
      const bool capacityExceeded = proposals.pairs.size() > maximumProposals;
      if (capacityExceeded) {
        proposals.pairs.clear();
        proposals.truncated = true;
        return;
      }
    }
  }
}

} // namespace

CanonicalSyncProblemResult mlir::pto::addCanonicalSyncDirectPairPatterns(
    CanonicalSyncPatternProblem &problem,
    CanonicalSyncDirectPairOptions options) {
  if (problem.isFrozen()) {
    return {CanonicalSyncProblemError::Frozen, std::nullopt};
  }
  const SyncCoverGraph &graph = problem.getGraph();
  ConnectorResourceIndex sources;
  ConnectorResourceIndex targets;
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    if (!isDirectPairMember(mechanism)) {
      continue;
    }
    const std::optional<SyncCoverScopeId> owner =
        getMechanismOwner(graph, mechanism);
    if (owner) {
      indexMechanismConnectors(graph, mechanism, *owner, sources, targets);
    }
  }
  std::map<SyncCoverScopeId, OwnedPairProposals> indexedProposals;
  for (const auto &[resource, targetOwners] : targets) {
    const auto sourcePosition = sources.find(resource);
    if (sourcePosition == sources.end()) {
      continue;
    }
    for (const auto &[targetOwner, targetEndpoints] : targetOwners) {
      for (const auto &[sourceOwner, sourceEndpoints] :
           sourcePosition->second) {
        const std::optional<SyncCoverScopeId> pairOwner =
            graph.getLowestCommonScope(targetOwner, sourceOwner);
        if (!pairOwner) {
          continue;
        }
        OwnedPairProposals &proposals = indexedProposals[*pairOwner];
        if (proposals.truncated) {
          continue;
        }
        addConnectorGroup(targetEndpoints, sourceEndpoints,
                          options.maximumEvaluationsPerScope, proposals);
        if (proposals.truncated) {
          problem.markPatternGenerationTruncated();
        }
      }
    }
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
      byOwner.try_emplace(owner);
      continue;
    }
    byOwner.emplace(owner, std::vector<SyncCoverMechanismPair>(
                               proposals.pairs.begin(), proposals.pairs.end()));
  }

  std::vector<SyncCoverCompletionSupply> supplies;
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    for (const CanonicalSyncSupplyBinding &binding :
         mechanism.descriptor.supplies) {
      supplies.push_back({mechanism.id, binding.edge, binding.allowedDemands,
                          binding.completionExport ==
                              CanonicalSyncSupplyExport::ScopeExitAfterDrain});
    }
  }
  const SyncCoverSingletonCoverageResult singleton =
      computeSyncCoverSingletonCoverage(graph, problem.getExpansion(),
                                        problem.getMechanisms().size(),
                                        supplies, problem.getDemands());
  if (!singleton) {
    return {CanonicalSyncProblemError::CoverageFailure, std::nullopt};
  }

  std::size_t addedCount = 0;
  std::size_t evaluationCount = 0;
  for (const auto &[owner, owned] : byOwner) {
    (void)owner;
    if (owned.empty()) {
      continue;
    }
    const SyncCoverPairCoverageResult joint = computeSyncCoverPairCoverage(
        graph, problem.getExpansion(), problem.getMechanisms().size(), supplies,
        owned, problem.getDemands(), options.pairCoverageLimits);
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
  problem.recordDirectPairGeneration(proposalCount, evaluationCount);
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
