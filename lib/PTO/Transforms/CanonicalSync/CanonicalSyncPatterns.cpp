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
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::pto;

namespace {

struct SupplyEntry {
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  unsigned distance = 0;
  CanonicalSyncMechanismId mechanism = 0;
};

bool isDirectPairMember(const CanonicalSyncMechanism &mechanism) {
  if (mechanism.descriptor.selectionTier !=
          CanonicalSyncSelectionTier::Precise ||
      mechanism.descriptor.kind == CanonicalSyncMechanismKind::Barrier ||
      mechanism.descriptor.supplies.size() != 1 ||
      mechanism.descriptor.eventUses.size() != 1 ||
      !mechanism.descriptor.supplies.front().allowedDemands.empty()) {
    return false;
  }
  return true;
}

std::size_t scopeDepth(const SyncCoverGraph &graph, SyncCoverScopeId scope) {
  std::size_t depth = 0;
  while (scope != 0) {
    ++depth;
    scope = graph.getScopes()[scope].parent;
  }
  return depth;
}

bool supplyLess(const SupplyEntry &left, const SupplyEntry &right) {
  return std::tie(left.distance, left.source, left.target, left.mechanism) <
         std::tie(right.distance, right.source, right.target, right.mechanism);
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
  std::stable_sort(
      eligible.begin(), eligible.end(), [&](auto first, auto second) {
        const SyncCoverScopeId firstScope = problem.getMechanisms()[first]
                                                .descriptor.supplies.front()
                                                .edge.scope;
        const SyncCoverScopeId secondScope = problem.getMechanisms()[second]
                                                 .descriptor.supplies.front()
                                                 .edge.scope;
        return std::make_pair(scopeDepth(graph, firstScope), first) >
               std::make_pair(scopeDepth(graph, secondScope), second);
      });

  std::vector<SyncCoverMechanismPair> proposals;
  for (std::size_t first = 0; first < eligible.size(); ++first) {
    for (std::size_t second = first + 1; second < eligible.size(); ++second) {
      const SyncCoverScopeId firstScope =
          problem.getMechanisms()[eligible[first]]
              .descriptor.supplies.front()
              .edge.scope;
      const SyncCoverScopeId secondScope =
          problem.getMechanisms()[eligible[second]]
              .descriptor.supplies.front()
              .edge.scope;
      const bool relatedScopes = graph.scopeContains(firstScope, secondScope) ||
                                 graph.scopeContains(secondScope, firstScope);
      if (!relatedScopes) {
        continue;
      }
      const auto members = std::minmax(eligible[first], eligible[second]);
      proposals.push_back({members.first, members.second});
      const bool evaluationLimitExceeded =
          proposals.size() > options.maximumEvaluations;
      if (evaluationLimitExceeded) {
        problem.markPatternGenerationTruncated();
        return {CanonicalSyncProblemError::None, 0};
      }
    }
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

  std::vector<SyncCoverCompletionSupply> supplies;
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    for (const CanonicalSyncSupplyBinding &binding :
         mechanism.descriptor.supplies) {
      supplies.push_back({mechanism.id, binding.edge, binding.allowedDemands});
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
    const CanonicalSyncResourceAllocation allocation =
        allocateCanonicalSyncEvents(problem, selection);
    if (!allocation.valid || !allocation.feasible) {
      continue;
    }
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
  const CanonicalSyncResourceAllocation allocation =
      allocateCanonicalSyncEvents(problem, pattern.members);
  if (!allocation.valid || !allocation.feasible) {
    return {CanonicalSyncProblemError::None, std::nullopt};
  }
  return problem.addPattern(std::move(pattern));
}

CanonicalSyncProblemResult mlir::pto::addCanonicalSyncRoundTripPatterns(
    CanonicalSyncPatternProblem &problem,
    CanonicalSyncRoundTripOptions options) {
  if (problem.isFrozen()) {
    return {CanonicalSyncProblemError::Frozen, std::nullopt};
  }
  const SyncCoverGraph &graph = problem.getGraph();
  std::set<unsigned> activeRecurrenceDistances;
  for (SyncCoverDemandId demandId : problem.getDemands()) {
    if (demandId >= graph.getDemands().size()) {
      return {CanonicalSyncProblemError::InvalidGraph, std::nullopt};
    }
    const unsigned distance = graph.getDemands()[demandId].distance;
    if (distance > 0) {
      activeRecurrenceDistances.insert(distance);
    }
  }

  std::vector<SupplyEntry> carried;
  std::map<std::pair<SyncCoverNodeId, SyncCoverNodeId>,
           std::vector<CanonicalSyncMechanismId>>
      directByEndpoints;
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    if (mechanism.descriptor.selectionTier !=
            CanonicalSyncSelectionTier::Precise ||
        mechanism.descriptor.kind == CanonicalSyncMechanismKind::Barrier) {
      continue;
    }
    for (const CanonicalSyncSupplyBinding &binding :
         mechanism.descriptor.supplies) {
      const SyncCoverEdge &edge = binding.edge;
      if (edge.distance == 0) {
        directByEndpoints[{edge.source, edge.target}].push_back(mechanism.id);
      } else if (activeRecurrenceDistances.find(edge.distance) !=
                 activeRecurrenceDistances.end()) {
        carried.push_back(
            {edge.source, edge.target, edge.distance, mechanism.id});
      }
    }
  }
  std::sort(carried.begin(), carried.end(), supplyLess);
  for (auto &[endpoints, mechanisms] : directByEndpoints) {
    (void)endpoints;
    std::sort(mechanisms.begin(), mechanisms.end());
    mechanisms.erase(std::unique(mechanisms.begin(), mechanisms.end()),
                     mechanisms.end());
  }

  std::set<std::pair<CanonicalSyncMechanismId, CanonicalSyncMechanismId>>
      proposed;
  std::size_t addedCount = 0;
  std::size_t evaluationCount = 0;
  for (const SupplyEntry &ring : carried) {
    const auto reversing = directByEndpoints.find({ring.target, ring.source});
    if (reversing == directByEndpoints.end()) {
      continue;
    }
    for (CanonicalSyncMechanismId closing : reversing->second) {
      if (ring.mechanism == closing) {
        continue;
      }
      const auto members = std::minmax(ring.mechanism, closing);
      if (!proposed.insert(members).second) {
        continue;
      }
      if (addedCount == options.maximumPatterns ||
          evaluationCount == options.maximumEvaluations) {
        return {CanonicalSyncProblemError::None, addedCount};
      }
      ++evaluationCount;
      const std::vector<CanonicalSyncMechanismId> selection{members.first,
                                                            members.second};
      const CanonicalSyncProblemResult added = addCanonicalSyncFeasiblePattern(
          problem, {CanonicalSyncPatternKind::RoundTrip, selection});
      if (!added) {
        return {added.error, addedCount};
      }
      addedCount += added.index.has_value();
    }
  }
  return {CanonicalSyncProblemError::None, addedCount};
}
