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
#include <limits>

using namespace mlir;
using namespace mlir::pto;

namespace {

std::uint64_t saturatingAdd(std::uint64_t value, std::uint64_t increment) {
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  return increment > maximum - value ? maximum : value + increment;
}

} // namespace

std::optional<std::uint64_t> mlir::pto::calculateWeightedCriticalPath(
    const std::vector<std::uint64_t> &vertexWeights,
    const std::vector<SyncGraphEdge> &edges) {
  const std::size_t vertexCount = vertexWeights.size();
  std::vector<std::vector<std::size_t>> children(vertexCount);
  std::vector<std::size_t> indegrees(vertexCount, 0);
  for (const SyncGraphEdge &edge : edges) {
    if (edge.source >= vertexCount || edge.target >= vertexCount ||
        edge.source == edge.target) {
      return std::nullopt;
    }
    children[edge.source].push_back(edge.target);
  }
  for (std::vector<std::size_t> &successors : children) {
    std::sort(successors.begin(), successors.end());
    successors.erase(std::unique(successors.begin(), successors.end()),
                     successors.end());
    for (std::size_t target : successors) {
      ++indegrees[target];
    }
  }

  std::vector<std::size_t> ready;
  ready.reserve(vertexCount);
  for (std::size_t vertex = 0; vertex < vertexCount; ++vertex) {
    if (indegrees[vertex] == 0) {
      ready.push_back(vertex);
    }
  }

  std::vector<std::uint64_t> distances(vertexWeights);
  std::uint64_t criticalPath = 0;
  for (std::size_t index = 0; index < ready.size(); ++index) {
    const std::size_t vertex = ready[index];
    criticalPath = std::max(criticalPath, distances[vertex]);
    for (std::size_t child : children[vertex]) {
      distances[child] =
          std::max(distances[child],
                   saturatingAdd(distances[vertex], vertexWeights[child]));
      if (--indegrees[child] == 0) {
        ready.push_back(child);
      }
    }
  }
  const bool hasCycle = ready.size() != vertexCount;
  if (hasCycle) {
    return std::nullopt;
  }
  return criticalPath;
}
