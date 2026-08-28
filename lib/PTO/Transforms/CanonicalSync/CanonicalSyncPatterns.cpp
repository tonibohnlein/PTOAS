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

bool suppliesMayCompose(const SyncCoverGraph &graph,
                        const CanonicalSyncSupplyBinding &first,
                        const CanonicalSyncSupplyBinding &second) {
  const SyncCoverEdge &firstEdge = first.edge;
  const SyncCoverEdge &secondEdge = second.edge;
  const std::size_t nodeCount = graph.getNodes().size();
  const bool invalidEndpoint =
      firstEdge.source >= nodeCount || firstEdge.target >= nodeCount ||
      secondEdge.source >= nodeCount || secondEdge.target >= nodeCount;
  if (invalidEndpoint) {
    return false;
  }
  const SyncCoverNode &firstSource = graph.getNodes()[firstEdge.source];
  const SyncCoverNode &firstTarget = graph.getNodes()[firstEdge.target];
  const SyncCoverNode &secondSource = graph.getNodes()[secondEdge.source];
  const SyncCoverNode &secondTarget = graph.getNodes()[secondEdge.target];
  const bool forwardChain =
      firstTarget.resource == secondSource.resource &&
      syncCoverGuardsCompatible(firstEdge.targetGuard, secondEdge.sourceGuard);
  const bool reverseChain =
      secondTarget.resource == firstSource.resource &&
      syncCoverGuardsCompatible(secondEdge.targetGuard, firstEdge.sourceGuard);
  return forwardChain || reverseChain;
}

bool mechanismsMayCompose(const SyncCoverGraph &graph,
                          const CanonicalSyncMechanism &first,
                          const CanonicalSyncMechanism &second) {
  return std::any_of(
      first.descriptor.supplies.begin(), first.descriptor.supplies.end(),
      [&](const CanonicalSyncSupplyBinding &firstSupply) {
        return std::any_of(second.descriptor.supplies.begin(),
                           second.descriptor.supplies.end(),
                           [&](const CanonicalSyncSupplyBinding &secondSupply) {
                             return suppliesMayCompose(graph, firstSupply,
                                                       secondSupply);
                           });
      });
}

} // namespace

CanonicalSyncProblemResult mlir::pto::addCanonicalSyncDirectPairPatterns(
    CanonicalSyncPatternProblem &problem,
    CanonicalSyncDirectPairOptions options) {
  if (problem.isFrozen()) {
    return {CanonicalSyncProblemError::Frozen, std::nullopt};
  }
  const SyncCoverGraph &graph = problem.getGraph();
  std::vector<CanonicalSyncMechanismId> eligible;
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    if (isDirectPairMember(mechanism)) {
      eligible.push_back(mechanism.id);
    }
  }
  std::map<SyncCoverScopeId, std::vector<SyncCoverMechanismPair>> byOwner;
  std::vector<SyncCoverMechanismPair> proposals;
  for (std::size_t first = 0; first < eligible.size(); ++first) {
    for (std::size_t second = first + 1; second < eligible.size(); ++second) {
      const CanonicalSyncMechanism &firstMechanism =
          problem.getMechanisms()[eligible[first]];
      const CanonicalSyncMechanism &secondMechanism =
          problem.getMechanisms()[eligible[second]];
      if (!mechanismsMayCompose(graph, firstMechanism, secondMechanism)) {
        continue;
      }
      const SyncCoverScopeId firstScope =
          firstMechanism.descriptor.supplies.front().edge.scope;
      const SyncCoverScopeId secondScope =
          secondMechanism.descriptor.supplies.front().edge.scope;
      const std::optional<SyncCoverScopeId> owner =
          graph.getLowestCommonScope(firstScope, secondScope);
      if (!owner) {
        continue;
      }
      const auto members = std::minmax(eligible[first], eligible[second]);
      byOwner[*owner].push_back({members.first, members.second});
    }
  }
  std::size_t proposalCount = 0;
  for (auto &[owner, owned] : byOwner) {
    (void)owner;
    const bool proposalCountOverflows =
        owned.size() > std::numeric_limits<std::size_t>::max() - proposalCount;
    if (proposalCountOverflows) {
      problem.markPatternGenerationTruncated();
      proposalCount = std::numeric_limits<std::size_t>::max();
    } else {
      proposalCount += owned.size();
    }
    const bool evaluationLimitExceeded =
        owned.size() > options.maximumEvaluationsPerScope;
    if (evaluationLimitExceeded) {
      if (!owned.empty()) {
        problem.markPatternGenerationTruncated();
      }
      continue;
    }
    proposals.insert(proposals.end(), owned.begin(), owned.end());
  }
  std::sort(proposals.begin(), proposals.end(),
            [](const auto &first, const auto &second) {
              return std::tie(first.first, first.second) <
                     std::tie(second.first, second.second);
            });
  proposals.erase(std::unique(proposals.begin(), proposals.end(),
                              [](const auto &first, const auto &second) {
                                return first.first == second.first &&
                                       first.second == second.second;
                              }),
                  proposals.end());
  problem.recordDirectPairGeneration(proposalCount, proposals.size());

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
  const SyncCoverPairCoverageResult joint = computeSyncCoverPairCoverage(
      graph, problem.getExpansion(), problem.getMechanisms().size(), supplies,
      proposals, problem.getDemands());
  if (!singleton || !joint) {
    return {CanonicalSyncProblemError::CoverageFailure, std::nullopt};
  }

  std::size_t addedCount = 0;
  for (std::size_t proposal = 0; proposal < proposals.size(); ++proposal) {
    const SyncCoverMechanismPair &members = proposals[proposal];
    const std::vector<CanonicalSyncMechanismId> selection{members.first,
                                                          members.second};
    SyncCoverDemandSet singletonUnion = singleton.mechanisms[members.first];
    singletonUnion.unite(singleton.mechanisms[members.second]);
    const CanonicalSyncProblemResult added =
        problem.addPattern({CanonicalSyncPatternKind::DirectPair, selection},
                           joint.pairs[proposal], singletonUnion);
    if (!added) {
      return {added.error, addedCount};
    }
    addedCount += added.index.has_value();
  }
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
