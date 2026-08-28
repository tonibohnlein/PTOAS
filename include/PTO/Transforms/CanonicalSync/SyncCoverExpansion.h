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

enum class SyncCoverLoopBoundaryKind : std::uint8_t {
  Entry,
  Exit,
};

struct SyncCoverLoopPeriodicSummary {
  SyncCoverControlId control = 0;
  std::size_t initialPhase = 0;
  std::vector<std::size_t> nextPhase;
  std::vector<unsigned> activeAlternative;
  std::vector<std::size_t> reachablePhases;
};

/// A phase-independent completion transfer exposed by a child loop. The
/// transfer preserves an established completion fact from child entry to
/// child exit on one issue resource. A zero-trip path is recorded separately
/// for diagnostics, but both paths have the same completion consequence.
struct SyncCoverLoopCompletionTransfer {
  std::uint32_t resource = 0;
  bool hasNonZeroPath = true;
  bool hasZeroTripPath = true;
};

/// One bottom-up semantic summary of a loop-local DAG. Resources include
/// nested summaries so a parent can route completion through child exits.
struct SyncCoverLoopSummary {
  SyncCoverScopeId scope = 0;
  std::optional<SyncCoverScopeId> parentLoop;
  SyncCoverAnchor entry;
  SyncCoverAnchor exit;
  bool zeroTripPossible = true;
  std::vector<SyncCoverScopeId> childLoops;
  std::vector<std::uint32_t> resources;
  std::vector<std::uint32_t> carryResources;
  std::vector<SyncCoverLoopPeriodicSummary> periodicControls;
  std::vector<SyncCoverLoopCompletionTransfer> completionTransfers;
  /// Locally owned operation identities that may be exposed through an
  /// ancestor interface. Descendant ports remain hierarchical references and
  /// are expanded only while charging an arena's node budget.
  std::vector<SyncCoverNodeId> completionPorts;
};

struct SyncCoverProjectedCompletion {
  std::size_t source = 0;
  std::size_t target = 0;
  unsigned sourceCopy = 0;
  unsigned targetCopy = 0;
  std::optional<SyncCoverScopeId> exportedLoop;
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

/// One immutable root arena and one local arena for each active loop scope are
/// built. An arena contains only locally owned operation nodes plus the
/// transfer interfaces of its immediate child loops. Recurrence arenas use
/// the scope's maximum demanded distance. Compact per-resource boundary nodes
/// encode cross-iteration issue order in O(N) edges per copy instead of an
/// O(N^2) all-pairs carry relation.
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
  std::optional<std::size_t> getLoopBoundary(SyncCoverScopeId scope,
                                             std::uint32_t resource,
                                             SyncCoverLoopBoundaryKind kind,
                                             unsigned copy) const;
  std::optional<std::size_t> getLoopPort(SyncCoverScopeId scope,
                                         SyncCoverNodeId node,
                                         unsigned copy) const;
  SyncCoverExpandedEdgeRange getOutgoingEdges(std::size_t virtualNode) const;

private:
  friend class SyncCoverExpandedProgram;

  SyncCoverScopeId scope_ = 0;
  unsigned horizon_ = 0;
  std::size_t operationVirtualNodeCount_ = 0;
  std::size_t loopSummaryVirtualNodeCount_ = 0;
  std::size_t loopPortVirtualNodeCount_ = 0;
  std::size_t virtualNodeCount_ = 0;
  std::vector<SyncCoverNodeId> operationNodes_;
  std::vector<std::pair<SyncCoverScopeId, std::uint32_t>> loopSummaryResources_;
  std::vector<std::pair<SyncCoverScopeId, SyncCoverNodeId>> loopSummaryPorts_;
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
  std::size_t loopSummaryNodes = 0;
  std::size_t zeroTripEdges = 0;
};

class SyncCoverExpandedProgram {
public:
  explicit SyncCoverExpandedProgram(const SyncCoverGraph &graph,
                                    SyncCoverExpansionLimits limits = {});
  SyncCoverExpandedProgram(const SyncCoverGraph &graph,
                           const std::vector<std::size_t> &activeDemands,
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
  const std::vector<SyncCoverLoopSummary> &getLoopSummaries() const {
    return loopSummaries_;
  }
  const SyncCoverLoopSummary *getLoopSummary(SyncCoverScopeId scope) const;
  std::optional<std::size_t>
  projectEndpoint(const SyncCoverGraph &graph,
                  const SyncCoverExpandedArena &arena, SyncCoverNodeId node,
                  unsigned copy) const;
  std::optional<SyncCoverProjectedCompletion>
  projectCompletion(const SyncCoverGraph &graph,
                    const SyncCoverExpandedArena &arena,
                    const SyncCoverEdge &edge, unsigned sourceCopy,
                    bool exportsCompletionAtScopeExit) const;
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
  std::vector<SyncCoverLoopSummary> loopSummaries_;
  std::map<SyncCoverScopeId, std::size_t> loopSummaryIndices_;
  std::vector<SyncCoverScopeId> arenaScopeByScope_;
  SyncCoverExpansionStatistics statistics_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVEREXPANSION_H
