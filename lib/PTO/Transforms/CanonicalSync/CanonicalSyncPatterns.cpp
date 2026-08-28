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
  for (std::size_t first = 0; first < eligible.size(); ++first) {
    for (std::size_t second = first + 1; second < eligible.size(); ++second) {
      const CanonicalSyncMechanism &firstMechanism =
          problem.getMechanisms()[eligible[first]];
      const CanonicalSyncMechanism &secondMechanism =
          problem.getMechanisms()[eligible[second]];
      if (!mechanismsMayCompose(graph, firstMechanism, secondMechanism)) {
        continue;
      }
      const std::optional<SyncCoverScopeId> firstOwner =
          getMechanismOwner(graph, firstMechanism);
      const std::optional<SyncCoverScopeId> secondOwner =
          getMechanismOwner(graph, secondMechanism);
      const std::optional<SyncCoverScopeId> pairOwner =
          firstOwner && secondOwner
              ? graph.getLowestCommonScope(*firstOwner, *secondOwner)
              : std::nullopt;
      if (!pairOwner) {
        continue;
      }
      const auto members = std::minmax(eligible[first], eligible[second]);
      byOwner[*pairOwner].push_back({members.first, members.second});
    }
  }
  std::size_t proposalCount = 0;
  for (auto &[owner, owned] : byOwner) {
    (void)owner;
    std::sort(owned.begin(), owned.end(),
              [](const auto &first, const auto &second) {
                return std::tie(first.first, first.second) <
                       std::tie(second.first, second.second);
              });
    owned.erase(std::unique(owned.begin(), owned.end(),
                            [](const auto &first, const auto &second) {
                              return first.first == second.first &&
                                     first.second == second.second;
                            }),
                owned.end());
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
      owned.clear();
    }
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
