// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverExpansion.h - Bounded periodic graph expansion -*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVEREXPANSION_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVEREXPANSION_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

enum class SyncCoverExpansionError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidLimits,
  BaseLimitExceeded,
};

enum class SyncCoverArenaUnavailableReason : std::uint8_t {
  NodeLimit,
  EdgeLimit,
  AggregateNodeLimit,
  AggregateEdgeLimit,
};

struct SyncCoverExpansionLimits {
  std::size_t maximumArenaNodes = 1U << 20;
  std::size_t maximumArenaEdges = 1U << 22;
  std::size_t maximumTotalNodes = 1U << 21;
  std::size_t maximumTotalEdges = 1U << 23;
};

struct SyncCoverExpandedEdge {
  std::size_t source = 0;
  std::size_t target = 0;
  SyncCoverEdgeKind kind = SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  std::optional<std::size_t> graphEdge;
  unsigned sourceCopy = 0;
  unsigned targetCopy = 0;
};

class SyncCoverExpandedEdgeRange {
public:
  using const_iterator = std::vector<SyncCoverExpandedEdge>::const_iterator;

  const_iterator begin() const { return begin_; }
  const_iterator end() const { return end_; }

private:
  friend class SyncCoverExpandedArena;

  SyncCoverExpandedEdgeRange(const_iterator begin, const_iterator end)
      : begin_(begin), end_(end) {}

  const_iterator begin_;
  const_iterator end_;
};

/// One immutable expansion is built for the distance-zero graph and one for
/// each recurrence loop at that loop's maximum demanded distance. Compact
/// per-resource boundary nodes encode cross-iteration issue order in O(N)
/// edges per copy instead of an O(N^2) all-pairs carry relation.
class SyncCoverExpandedArena {
public:
  SyncCoverScopeId getScope() const { return scope_; }
  unsigned getHorizon() const { return horizon_; }
  std::size_t getOperationNodeCount() const { return operationNodes_.size(); }
  std::size_t getVirtualNodeCount() const { return virtualNodeCount_; }
  const std::vector<SyncCoverNodeId> &getOperationNodes() const {
    return operationNodes_;
  }
  const std::vector<SyncCoverExpandedEdge> &getEdges() const { return edges_; }

  std::optional<std::size_t> getVirtualOperation(SyncCoverNodeId node,
                                                 unsigned copy) const;
  std::optional<SyncCoverNodeId>
  getOperationForVirtualNode(std::size_t virtualNode) const;
  std::optional<unsigned> getCopyForVirtualNode(std::size_t virtualNode) const;
  SyncCoverExpandedEdgeRange getOutgoingEdges(std::size_t virtualNode) const;

private:
  friend class SyncCoverExpandedProgram;

  SyncCoverScopeId scope_ = 0;
  unsigned horizon_ = 0;
  std::size_t operationVirtualNodeCount_ = 0;
  std::size_t virtualNodeCount_ = 0;
  std::vector<SyncCoverNodeId> operationNodes_;
  std::vector<std::uint32_t> carryResources_;
  std::vector<SyncCoverExpandedEdge> edges_;
  std::vector<std::size_t> outgoingOffsets_;
};

struct SyncCoverExpansionStatistics {
  std::size_t arenaCount = 0;
  std::size_t unavailableArenaCount = 0;
  std::size_t virtualNodes = 0;
  std::size_t virtualEdges = 0;
  std::size_t compactCarryEdges = 0;
};

class SyncCoverExpandedProgram {
public:
  explicit SyncCoverExpandedProgram(const SyncCoverGraph &graph,
                                    SyncCoverExpansionLimits limits = {});

  explicit operator bool() const {
    return error_ == SyncCoverExpansionError::None;
  }
  SyncCoverExpansionError getError() const { return error_; }
  bool isForGraph(const SyncCoverGraph &graph) const {
    return ownerIdentity_ && ownerIdentity_ == graph.getIdentity() &&
           graph.isStructureFrozen();
  }
  const SyncCoverExpandedArena &getBaseArena() const { return baseArena_; }
  const SyncCoverExpandedArena *
  getRecurrenceArena(SyncCoverScopeId scope) const;
  const SyncCoverExpandedArena *getArena(const SyncCoverDemand &demand) const;
  std::optional<SyncCoverArenaUnavailableReason>
  getUnavailableReason(SyncCoverScopeId scope) const;
  SyncCoverExpansionStatistics getStatistics() const { return statistics_; }

private:
  struct ArenaBuildResult;

  ArenaBuildResult
  buildArena(const SyncCoverGraph &graph, SyncCoverScopeId scope,
             unsigned horizon, std::size_t maximumNodes,
             std::size_t maximumEdges,
             SyncCoverArenaUnavailableReason nodeLimitReason,
             SyncCoverArenaUnavailableReason edgeLimitReason) const;

  SyncCoverExpansionError error_ = SyncCoverExpansionError::None;
  std::shared_ptr<const std::uint8_t> ownerIdentity_;
  SyncCoverExpansionLimits limits_;
  SyncCoverExpandedArena baseArena_;
  std::map<SyncCoverScopeId, SyncCoverExpandedArena> recurrenceArenas_;
  std::map<SyncCoverScopeId, SyncCoverArenaUnavailableReason>
      unavailableArenas_;
  SyncCoverExpansionStatistics statistics_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVEREXPANSION_H
