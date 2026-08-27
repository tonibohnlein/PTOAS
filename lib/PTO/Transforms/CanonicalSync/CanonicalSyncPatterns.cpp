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

bool supplyLess(const SupplyEntry &left, const SupplyEntry &right) {
  return std::tie(left.distance, left.source, left.target, left.mechanism) <
         std::tie(right.distance, right.source, right.target, right.mechanism);
}

} // namespace

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
  const CanonicalSyncProblemResult added =
      problem.addPattern(std::move(pattern));
  return added.error == CanonicalSyncProblemError::LimitExceeded
             ? CanonicalSyncProblemResult{}
             : added;
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
    if (mechanism.descriptor.kind == CanonicalSyncMechanismKind::Barrier) {
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
