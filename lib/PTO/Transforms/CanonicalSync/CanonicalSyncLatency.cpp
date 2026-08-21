// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>

using namespace mlir;
using namespace mlir::pto;

namespace {

std::uint64_t saturatingAdd(std::uint64_t value, std::uint64_t increment) {
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  return increment > maximum - value ? maximum : value + increment;
}

bool isInDomain(const CanonicalEvent &event,
                const CanonicalEventDomainKey &domain) {
  return event.sourcePipe == domain.source && event.targetPipe == domain.target;
}

std::set<Block *> getAffectedBlocks(const CanonicalSyncPlan &plan,
                                    ArrayRef<CanonicalEvent> events) {
  std::set<Block *> affectedBlocks;
  for (const CanonicalEvent &event : events) {
    const bool validEndpoints = event.source < plan.getNodes().size() &&
                                event.target < plan.getNodes().size();
    if (!validEndpoints) {
      continue;
    }
    Operation *source = plan.getNodes()[event.source].operation;
    Operation *target = plan.getNodes()[event.target].operation;
    const bool sameBlock =
        source && target && source->getBlock() == target->getBlock();
    if (sameBlock) {
      affectedBlocks.insert(source->getBlock());
    }
  }
  return affectedBlocks;
}

} // namespace

CanonicalSyncLatencyContext::CanonicalSyncLatencyContext(
    const CanonicalSyncPlan &plan, ArrayRef<CanonicalEvent> domainEvents,
    const CanonicalEventDomainKey &domain) {
  addNodes(plan, getAffectedBlocks(plan, domainEvents));
  for (const SyncGraphEdge &edge : plan.getFixedEdges()) {
    addBaseEdge(edge.source, edge.target);
  }
  for (const CanonicalEvent &event : plan.getEvents()) {
    if (!isInDomain(event, domain)) {
      addBaseEdge(event.source, event.target);
    }
  }
}

void CanonicalSyncLatencyContext::addNodes(
    const CanonicalSyncPlan &plan, const std::set<Block *> &affectedBlocks) {
  std::map<Block *, std::size_t> blockIndices;
  for (auto [nodeIndex, node] : llvm::enumerate(plan.getNodes())) {
    if (!node.operation || !affectedBlocks.count(node.operation->getBlock())) {
      continue;
    }
    const auto inserted =
        blockIndices.try_emplace(node.operation->getBlock(), blocks_.size());
    if (inserted.second) {
      blocks_.emplace_back();
    }
    const std::size_t blockIndex = inserted.first->second;
    BlockGraph &block = blocks_[blockIndex];
    const std::size_t localIndex = block.weights.size();
    block.weights.push_back(
        saturatingAdd(node.computeWeight, node.transferWeight));
    nodeLocations_.emplace(nodeIndex, std::make_pair(blockIndex, localIndex));
  }
}

std::optional<std::pair<std::size_t, SyncGraphEdge>>
CanonicalSyncLatencyContext::localizeEdge(std::size_t source,
                                          std::size_t target) const {
  auto sourceLocation = nodeLocations_.find(source);
  auto targetLocation = nodeLocations_.find(target);
  const bool invalidLocation = sourceLocation == nodeLocations_.end() ||
                               targetLocation == nodeLocations_.end();
  if (invalidLocation) {
    return std::nullopt;
  }
  const bool invalidEdge =
      sourceLocation->second.first != targetLocation->second.first ||
      sourceLocation->second.second >= targetLocation->second.second;
  if (invalidEdge) {
    return std::nullopt;
  }
  return std::make_pair(sourceLocation->second.first,
                        SyncGraphEdge{sourceLocation->second.second,
                                      targetLocation->second.second,
                                      SyncGraphEdgeKind::HardwareCompletion});
}

void CanonicalSyncLatencyContext::addBaseEdge(std::size_t source,
                                              std::size_t target) {
  std::optional<std::pair<std::size_t, SyncGraphEdge>> localized =
      localizeEdge(source, target);
  if (localized) {
    blocks_[localized->first].baseEdges.push_back(localized->second);
  }
}

std::uint64_t CanonicalSyncLatencyContext::calculateCriticalPathWeight(
    ArrayRef<CanonicalEvent> domainEvents) const {
  std::vector<std::vector<SyncGraphEdge>> edges;
  edges.reserve(blocks_.size());
  for (const BlockGraph &block : blocks_) {
    edges.push_back(block.baseEdges);
  }
  for (const CanonicalEvent &event : domainEvents) {
    std::optional<std::pair<std::size_t, SyncGraphEdge>> localized =
        localizeEdge(event.source, event.target);
    if (localized) {
      edges[localized->first].push_back(localized->second);
    }
  }

  std::uint64_t total = 0;
  for (std::size_t block = 0; block < blocks_.size(); ++block) {
    std::optional<std::uint64_t> criticalPath =
        calculateWeightedCriticalPath(blocks_[block].weights, edges[block]);
    if (!criticalPath) {
      return std::numeric_limits<std::uint64_t>::max();
    }
    total = saturatingAdd(total, *criticalPath);
  }
  return total;
}
