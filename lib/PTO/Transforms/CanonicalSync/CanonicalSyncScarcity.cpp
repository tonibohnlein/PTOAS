// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// This file repairs event-id scarcity without changing operation order. It
// searches deterministic coalescings of forward events and validates every
// replacement against completion-qualified reachability before accepting it.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <utility>

using namespace mlir;
using namespace mlir::pto;

LogicalResult CanonicalSyncPlanBuilder::repairEventScarcity() {
  std::set<CanonicalEventDomainKey> domains;
  for (const CanonicalEvent &event : plan_.events_) {
    domains.insert({event.sourcePipe, event.targetPipe});
  }
  for (const CanonicalEventDomainKey &key : domains) {
    unsigned reserved = 0;
    for (unsigned eventId : reservedIds_[key]) {
      reserved += eventId < eventIdMax_ ? 1U : 0U;
    }
    const unsigned available = eventIdMax_ - reserved;
    if (failed(repairEventDomain(key, available))) {
      return failure();
    }
  }
  return success();
}

std::optional<CanonicalEvent> CanonicalSyncPlanBuilder::coalesceForwardEvents(
    ArrayRef<CanonicalEvent> events) const {
  const std::size_t eventCount = events.size();
  if (eventCount < 2) {
    return std::nullopt;
  }
  Block *block = nullptr;
  std::size_t source = events.front().source;
  std::size_t target = events.front().target;
  for (const CanonicalEvent &event : events) {
    if (event.width != 1 || event.recurrenceLoop || event.forwardDrainLoop ||
        event.source >= plan_.nodes_.size() ||
        event.target >= plan_.nodes_.size()) {
      return std::nullopt;
    }
    Operation *sourceOp = plan_.nodes_[event.source].operation;
    Operation *targetOp = plan_.nodes_[event.target].operation;
    if (!sourceOp || !targetOp ||
        sourceOp->getBlock() != targetOp->getBlock() ||
        (block && block != sourceOp->getBlock())) {
      return std::nullopt;
    }
    block = sourceOp->getBlock();
    const auto currentSource =
        std::make_pair(plan_.nodes_[source].order, source);
    const auto candidateSource =
        std::make_pair(plan_.nodes_[event.source].order, event.source);
    if (currentSource < candidateSource) {
      source = event.source;
    }
    const auto candidateTarget =
        std::make_pair(plan_.nodes_[event.target].order, event.target);
    const auto currentTarget =
        std::make_pair(plan_.nodes_[target].order, target);
    if (candidateTarget < currentTarget) {
      target = event.target;
    }
  }
  if (source >= target) {
    return std::nullopt;
  }

  CanonicalEvent result = events.front();
  result.source = source;
  result.target = target;
  result.setAnchor = getSetAnchor(plan_.nodes_[source].operation,
                                  plan_.nodes_[target].operation);
  result.waitAnchor = getWaitAnchor(plan_.nodes_[source].operation,
                                    plan_.nodes_[target].operation);
  result.intervalBegin = getAnchorPosition(result.setAnchor);
  result.intervalEnd = getAnchorPosition(result.waitAnchor);
  result.eventIds.clear();
  if (result.intervalBegin > result.intervalEnd) {
    return std::nullopt;
  }
  return result;
}

bool CanonicalSyncPlanBuilder::coversCoalescedEvents(
    const CanonicalEvent &candidate, ArrayRef<CanonicalEvent> originals) const {
  Block *block = plan_.nodes_[candidate.source].operation->getBlock();
  std::vector<SyncGraphEdge> edges;
  const std::size_t nodeCount = plan_.nodes_.size();
  for (const SyncGraphEdge &edge : plan_.fixedEdges_) {
    if (edge.source < nodeCount && edge.target < nodeCount &&
        plan_.nodes_[edge.source].operation->getBlock() == block &&
        plan_.nodes_[edge.target].operation->getBlock() == block) {
      edges.push_back(edge);
    }
  }
  edges.push_back({candidate.source, candidate.target,
                   SyncGraphEdgeKind::HardwareCompletion});

  std::vector<CompletionRequirement> requirements;
  for (const CanonicalEvent &event : originals) {
    requirements.push_back({event.source, event.target});
  }
  const auto inBlock = [&](std::size_t, std::size_t vertex) {
    return vertex < plan_.nodes_.size() &&
           plan_.nodes_[vertex].operation->getBlock() == block;
  };
  const std::vector<bool> covered = getCompletionRequirementCoverage(
      plan_.nodes_.size(), edges, requirements, inBlock);
  return llvm::all_of(covered, [](bool value) { return value; });
}
