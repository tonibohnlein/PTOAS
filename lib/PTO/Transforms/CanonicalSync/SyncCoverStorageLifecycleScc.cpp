// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "SyncCoverStorageLifecycleInternal.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct CompactAdjacency {
  std::vector<std::size_t> offsets;
  std::vector<SyncCoverStorageLifecycleEpochId> nodes;
};

struct DfsFrame {
  SyncCoverStorageLifecycleEpochId epoch = 0;
  std::size_t next = 0;
  std::size_t end = 0;
};

SyncCoverStorageLifecycleError buildAdjacency(
    const SyncCoverStorageLifecycleComponent &component,
    SyncCoverStorageLifecycleWorkBudget &budget, CompactAdjacency &forward,
    CompactAdjacency &reverse) {
  const std::size_t epochCount = component.epochs.size();
  const std::size_t edgeCount = component.edges.size();
  const bool initializationAvailable =
      budget.consume(epochCount + 1) && budget.consume(epochCount + 1);
  if (!initializationAvailable) {
    return SyncCoverStorageLifecycleError::LimitExceeded;
  }
  forward.offsets.assign(epochCount + 1, 0);
  reverse.offsets.assign(epochCount + 1, 0);
  for (const SyncCoverStorageLifecycleEdge &edge : component.edges) {
    if (!budget.consume(3)) {
      return SyncCoverStorageLifecycleError::LimitExceeded;
    }
    if (edge.source >= epochCount || edge.target >= epochCount) {
      return SyncCoverStorageLifecycleError::InvalidGraph;
    }
    ++forward.offsets[edge.source + 1];
    ++reverse.offsets[edge.target + 1];
  }
  for (std::size_t epoch = 0; epoch < epochCount; ++epoch) {
    if (!budget.consume(2)) {
      return SyncCoverStorageLifecycleError::LimitExceeded;
    }
    forward.offsets[epoch + 1] += forward.offsets[epoch];
    reverse.offsets[epoch + 1] += reverse.offsets[epoch];
  }
  const bool nodeStorageWorkAvailable =
      budget.consume(edgeCount) && budget.consume(edgeCount);
  if (!nodeStorageWorkAvailable) {
    return SyncCoverStorageLifecycleError::LimitExceeded;
  }
  forward.nodes.resize(edgeCount);
  reverse.nodes.resize(edgeCount);
  const bool cursorWorkAvailable =
      budget.consume(forward.offsets.size()) &&
      budget.consume(reverse.offsets.size());
  if (!cursorWorkAvailable) {
    return SyncCoverStorageLifecycleError::LimitExceeded;
  }
  std::vector<std::size_t> forwardCursor = forward.offsets;
  std::vector<std::size_t> reverseCursor = reverse.offsets;
  for (const SyncCoverStorageLifecycleEdge &edge : component.edges) {
    if (!budget.consume(3)) {
      return SyncCoverStorageLifecycleError::LimitExceeded;
    }
    forward.nodes[forwardCursor[edge.source]++] = edge.target;
    reverse.nodes[reverseCursor[edge.target]++] = edge.source;
  }
  return SyncCoverStorageLifecycleError::None;
}

SyncCoverStorageLifecycleError computeFinishOrder(
    const CompactAdjacency &forward,
    SyncCoverStorageLifecycleWorkBudget &budget,
    std::vector<SyncCoverStorageLifecycleEpochId> &finishOrder) {
  const std::size_t epochCount = forward.offsets.size() - 1;
  if (!budget.consume(epochCount)) {
    return SyncCoverStorageLifecycleError::LimitExceeded;
  }
  std::vector<std::uint8_t> visited(epochCount, 0);
  std::vector<DfsFrame> stack;
  stack.reserve(epochCount);
  finishOrder.reserve(epochCount);
  for (SyncCoverStorageLifecycleEpochId root = 0; root < epochCount; ++root) {
    if (!budget.consume()) {
      return SyncCoverStorageLifecycleError::LimitExceeded;
    }
    if (visited[root] != 0) {
      continue;
    }
    visited[root] = 1;
    stack.push_back({root, forward.offsets[root], forward.offsets[root + 1]});
    while (!stack.empty()) {
      if (!budget.consume()) {
        return SyncCoverStorageLifecycleError::LimitExceeded;
      }
      DfsFrame &frame = stack.back();
      if (frame.next == frame.end) {
        finishOrder.push_back(frame.epoch);
        stack.pop_back();
        continue;
      }
      const SyncCoverStorageLifecycleEpochId successor =
          forward.nodes[frame.next++];
      if (visited[successor] != 0) {
        continue;
      }
      visited[successor] = 1;
      stack.push_back({successor, forward.offsets[successor],
                       forward.offsets[successor + 1]});
    }
  }
  return SyncCoverStorageLifecycleError::None;
}

SyncCoverStorageLifecycleError assignSccs(
    const CompactAdjacency &reverse,
    const std::vector<SyncCoverStorageLifecycleEpochId> &finishOrder,
    const SyncCoverStorageLifecycleLimits &limits,
    SyncCoverStorageLifecycleStatistics &statistics,
    SyncCoverStorageLifecycleWorkBudget &budget,
    SyncCoverStorageLifecycleComponent &component) {
  const SyncCoverStorageLifecycleSccId unassigned =
      std::numeric_limits<SyncCoverStorageLifecycleSccId>::max();
  const std::size_t epochCount = component.epochs.size();
  if (!budget.consume(epochCount)) {
    return SyncCoverStorageLifecycleError::LimitExceeded;
  }
  component.epochSccs.assign(epochCount, unassigned);
  std::vector<SyncCoverStorageLifecycleEpochId> stack;
  stack.reserve(epochCount);
  for (auto position = finishOrder.rbegin(); position != finishOrder.rend();
       ++position) {
    if (!budget.consume()) {
      return SyncCoverStorageLifecycleError::LimitExceeded;
    }
    const SyncCoverStorageLifecycleEpochId root = *position;
    if (component.epochSccs[root] != unassigned) {
      continue;
    }
    if (statistics.sccs >= limits.maximumSccs) {
      return SyncCoverStorageLifecycleError::LimitExceeded;
    }
    const SyncCoverStorageLifecycleSccId scc = component.sccs.size();
    component.sccs.push_back({scc});
    ++statistics.sccs;
    component.epochSccs[root] = scc;
    stack.push_back(root);
    while (!stack.empty()) {
      if (!budget.consume()) {
        return SyncCoverStorageLifecycleError::LimitExceeded;
      }
      const SyncCoverStorageLifecycleEpochId epoch = stack.back();
      stack.pop_back();
      component.sccs[scc].epochs.push_back(epoch);
      const std::size_t begin = reverse.offsets[epoch];
      const std::size_t end = reverse.offsets[epoch + 1];
      for (std::size_t edge = begin; edge < end; ++edge) {
        if (!budget.consume()) {
          return SyncCoverStorageLifecycleError::LimitExceeded;
        }
        const SyncCoverStorageLifecycleEpochId predecessor =
            reverse.nodes[edge];
        if (component.epochSccs[predecessor] != unassigned) {
          continue;
        }
        component.epochSccs[predecessor] = scc;
        stack.push_back(predecessor);
      }
    }
  }
  return SyncCoverStorageLifecycleError::None;
}

SyncCoverStorageLifecycleError classifySccEdges(
    SyncCoverStorageLifecycleComponent &component,
    SyncCoverStorageLifecycleStatistics &statistics,
    SyncCoverStorageLifecycleWorkBudget &budget) {
  for (const SyncCoverStorageLifecycleEdge &edge : component.edges) {
    if (!budget.consume()) {
      return SyncCoverStorageLifecycleError::LimitExceeded;
    }
    const SyncCoverStorageLifecycleSccId source =
        component.epochSccs[edge.source];
    const SyncCoverStorageLifecycleSccId target =
        component.epochSccs[edge.target];
    if (source != target) {
      component.sccTransfers.push_back({edge.id, source, target});
      ++statistics.sccTransfers;
      continue;
    }
    SyncCoverStorageLifecycleScc &scc = component.sccs[source];
    scc.internalEdges.push_back(edge.id);
    scc.kinds |= edge.kinds;
    scc.maximumDistance = std::max(scc.maximumDistance, edge.distance);
    scc.cyclic = scc.cyclic || edge.source == edge.target;
  }
  const SyncCoverStorageLifecycleEdgeKindMask readyRelease =
      syncCoverStorageLifecycleEdgeKindBit(
          SyncCoverStorageLifecycleEdgeKind::Ready) |
      syncCoverStorageLifecycleEdgeKindBit(
          SyncCoverStorageLifecycleEdgeKind::Release);
  for (SyncCoverStorageLifecycleScc &scc : component.sccs) {
    if (!budget.consume()) {
      return SyncCoverStorageLifecycleError::LimitExceeded;
    }
    scc.cyclic = scc.cyclic || scc.epochs.size() > 1;
    statistics.maximumSccEpochs =
        std::max(statistics.maximumSccEpochs, scc.epochs.size());
    statistics.cyclicSccs += scc.cyclic ? 1 : 0;
    statistics.readyReleaseSccs +=
        scc.cyclic && (scc.kinds & readyRelease) == readyRelease ? 1 : 0;
  }
  return SyncCoverStorageLifecycleError::None;
}

} // namespace

SyncCoverStorageLifecycleError mlir::pto::buildSyncCoverStorageLifecycleSccs(
    SyncCoverStorageLifecycleComponent &component,
    const SyncCoverStorageLifecycleLimits &limits,
    SyncCoverStorageLifecycleStatistics &statistics,
    SyncCoverStorageLifecycleWorkBudget &workBudget) {
  CompactAdjacency forward;
  CompactAdjacency reverse;
  SyncCoverStorageLifecycleError error =
      buildAdjacency(component, workBudget, forward, reverse);
  if (error != SyncCoverStorageLifecycleError::None) {
    return error;
  }
  std::vector<SyncCoverStorageLifecycleEpochId> finishOrder;
  error = computeFinishOrder(forward, workBudget, finishOrder);
  if (error != SyncCoverStorageLifecycleError::None) {
    return error;
  }
  error = assignSccs(reverse, finishOrder, limits, statistics, workBudget,
                     component);
  if (error != SyncCoverStorageLifecycleError::None) {
    return error;
  }
  return classifySccEdges(component, statistics, workBudget);
}
