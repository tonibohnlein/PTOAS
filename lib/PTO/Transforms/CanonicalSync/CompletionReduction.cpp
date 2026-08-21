// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"

#include <algorithm>
#include <deque>
#include <numeric>
#include <tuple>

using namespace mlir::pto;

namespace {

struct ReachabilityState {
  std::size_t vertex = 0;
  bool hasCompletion = false;
};

bool isValidRequirement(std::size_t vertexCount,
                        const CompletionRequirement &requirement) {
  return requirement.source < vertexCount && requirement.target < vertexCount &&
         requirement.source < requirement.target;
}

bool isAvailable(const CompletionVertexFilter &isVertexAvailable,
                 std::size_t requirement, std::size_t vertex) {
  return !isVertexAvailable || isVertexAvailable(requirement, vertex);
}

bool hasQualifiedPath(std::size_t vertexCount,
                      const std::vector<SyncGraphEdge> &fixedEdges,
                      const std::vector<CompletionRequirement> &requirements,
                      const std::vector<bool> &keep, std::size_t skipped,
                      std::size_t source, std::size_t target,
                      const CompletionVertexFilter &isVertexAvailable) {
  std::vector<std::vector<SyncGraphEdge>> outgoing(vertexCount);
  for (const SyncGraphEdge &edge : fixedEdges) {
    if (edge.source < vertexCount && edge.target < vertexCount &&
        isAvailable(isVertexAvailable, skipped, edge.source) &&
        isAvailable(isVertexAvailable, skipped, edge.target)) {
      outgoing[edge.source].push_back(edge);
    }
  }
  for (std::size_t index = 0; index < requirements.size(); ++index) {
    if (index != skipped && keep[index]) {
      const CompletionRequirement &requirement = requirements[index];
      const bool isUnavailable =
          !isValidRequirement(vertexCount, requirement) ||
          !isAvailable(isVertexAvailable, skipped, requirement.source) ||
          !isAvailable(isVertexAvailable, skipped, requirement.target);
      if (isUnavailable) {
        continue;
      }
      outgoing[requirement.source].push_back(
          {requirement.source, requirement.target,
           SyncGraphEdgeKind::HardwareCompletion});
    }
  }

  std::vector<std::vector<bool>> visited(vertexCount,
                                         std::vector<bool>(2, false));
  std::deque<ReachabilityState> ready{{source, false}};
  visited[source][0] = true;
  while (!ready.empty()) {
    const ReachabilityState state = ready.front();
    ready.pop_front();
    for (const SyncGraphEdge &edge : outgoing[state.vertex]) {
      const bool hasCompletion =
          state.hasCompletion ||
          edge.kind == SyncGraphEdgeKind::HardwareCompletion;
      if (edge.target == target && hasCompletion) {
        return true;
      }
      const unsigned stateIndex = hasCompletion ? 1U : 0U;
      if (!visited[edge.target][stateIndex]) {
        visited[edge.target][stateIndex] = true;
        ready.push_back({edge.target, hasCompletion});
      }
    }
  }
  return false;
}

} // namespace

std::vector<bool> mlir::pto::reduceCompletionRequirements(
    std::size_t vertexCount, const std::vector<SyncGraphEdge> &fixedEdges,
    const std::vector<CompletionRequirement> &requirements,
    const CompletionVertexFilter &isVertexAvailable) {
  std::vector<bool> keep(requirements.size(), true);
  for (std::size_t index = 0; index < requirements.size(); ++index) {
    if (!isValidRequirement(vertexCount, requirements[index])) {
      keep[index] = false;
    }
  }
  std::vector<std::size_t> order(requirements.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
      order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        const CompletionRequirement &left = requirements[lhs];
        const CompletionRequirement &right = requirements[rhs];
        const std::size_t leftSpan =
            left.target >= left.source ? left.target - left.source : 0;
        const std::size_t rightSpan =
            right.target >= right.source ? right.target - right.source : 0;
        if (leftSpan != rightSpan) {
          return leftSpan > rightSpan;
        }
        return std::tie(left.source, left.target, lhs) <
               std::tie(right.source, right.target, rhs);
      });

  for (std::size_t candidate : order) {
    const CompletionRequirement &requirement = requirements[candidate];
    if (!keep[candidate]) {
      continue;
    }
    keep[candidate] = false;
    if (!hasQualifiedPath(vertexCount, fixedEdges, requirements, keep,
                          candidate, requirement.source, requirement.target,
                          isVertexAvailable)) {
      keep[candidate] = true;
    }
  }
  return keep;
}

std::vector<bool> mlir::pto::getCompletionRequirementCoverage(
    std::size_t vertexCount, const std::vector<SyncGraphEdge> &edges,
    const std::vector<CompletionRequirement> &requirements,
    const CompletionVertexFilter &isVertexAvailable) {
  std::vector<bool> covered(requirements.size(), false);
  const std::vector<CompletionRequirement> noAdditionalRequirements;
  const std::vector<bool> noRetainedRequirements;
  for (std::size_t index = 0; index < requirements.size(); ++index) {
    const CompletionRequirement &requirement = requirements[index];
    const bool valid = isValidRequirement(vertexCount, requirement);
    const bool sourceAvailable =
        isAvailable(isVertexAvailable, index, requirement.source);
    const bool targetAvailable =
        isAvailable(isVertexAvailable, index, requirement.target);
    if (!valid || !sourceAvailable || !targetAvailable) {
      continue;
    }
    covered[index] = hasQualifiedPath(
        vertexCount, edges, noAdditionalRequirements, noRetainedRequirements,
        index, requirement.source, requirement.target, isVertexAvailable);
  }
  return covered;
}
