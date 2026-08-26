// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverExpansion.h - Shared finite loop expansion ----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVEREXPANSION_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVEREXPANSION_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

enum class SyncCoverExpansionError : std::uint8_t {
  None,
  InvalidGraph,
  ExpansionLimitExceeded,
};

struct SyncCoverExpansionLimits {
  std::size_t maximumNodes = 1U << 20;
  std::size_t maximumEdges = 1U << 22;
};

/// One graph edge instantiated between two occurrences in a finite recurrence
/// window. Guards and optional-scope availability remain per-query masks.
struct SyncCoverExpandedEdge {
  std::size_t graphEdge = 0;
  std::size_t source = 0;
  std::size_t target = 0;
  unsigned sourceCopy = 0;
  unsigned targetCopy = 0;
};

class SyncCoverExpandedEdgeRange {
public:
  using const_iterator = std::vector<SyncCoverExpandedEdge>::const_iterator;

  const_iterator begin() const { return begin_; }
  const_iterator end() const { return end_; }

private:
  friend class SyncCoverExpandedAdjacency;

  SyncCoverExpandedEdgeRange(const_iterator begin, const_iterator end)
      : begin_(begin), end_(end) {}

  const_iterator begin_;
  const_iterator end_;
};

/// Source-sorted CSR edges. Structural edges and mechanism-supplied edges use
/// separate instances so adding opportunities never invalidates fixed
/// topology.
class SyncCoverExpandedAdjacency {
public:
  const std::vector<SyncCoverExpandedEdge> &getEdges() const { return edges_; }
  SyncCoverExpandedEdgeRange getOutgoingEdges(std::size_t virtualNode) const;

private:
  friend class SyncCoverExpandedProgram;

  std::vector<SyncCoverExpandedEdge> edges_;
  std::vector<std::size_t> outgoingOffsets_;
};

/// CSR topology for either the distance-zero program or one exact recurrence
/// scope. Nested loops are not multiplied into a Cartesian iteration space;
/// an arena contains its scope subtree plus ancestor-scope boundary nodes and
/// copies 0..horizon. Per-context masks restrict boundary nodes to endpoint-
/// implied copies.
class SyncCoverExpandedArena {
public:
  SyncCoverScopeId getScope() const { return scope_; }
  unsigned getHorizon() const { return horizon_; }
  std::size_t getNodeCount() const { return globalNodes_.size(); }
  std::size_t getVirtualNodeCount() const { return virtualNodeCount_; }
  const std::vector<SyncCoverNodeId> &getGlobalNodes() const {
    return globalNodes_;
  }
  std::optional<std::size_t> getLocalNode(SyncCoverNodeId globalNode) const;
  std::optional<std::size_t> getVirtualNode(SyncCoverNodeId globalNode,
                                            unsigned copy) const;
  const SyncCoverExpandedAdjacency &getStructuralEdges() const {
    return structuralEdges_;
  }
  const SyncCoverExpandedAdjacency &getMechanismEdges() const {
    return mechanismEdges_;
  }

private:
  friend class SyncCoverExpandedProgram;

  SyncCoverScopeId scope_ = 0;
  unsigned horizon_ = 0;
  std::size_t virtualNodeCount_ = 0;
  std::vector<SyncCoverNodeId> globalNodes_;
  SyncCoverExpandedAdjacency structuralEdges_;
  SyncCoverExpandedAdjacency mechanismEdges_;
};

/// Immutable structural expansion shared by grounding and final verification,
/// with an explicitly versioned mechanism overlay. The graph structure must be
/// frozen before construction. A positive-distance demand uses the arena for
/// its exact recurrence scope and addresses target copy `distance`.
class SyncCoverExpandedProgram {
public:
  explicit SyncCoverExpandedProgram(
      const SyncCoverGraph &graph, SyncCoverExpansionLimits limits = {});

  explicit operator bool() const {
    return error_ == SyncCoverExpansionError::None;
  }
  SyncCoverExpansionError getError() const { return error_; }
  std::size_t getGraphGeneration() const { return graphGeneration_; }
  std::size_t getStructuralGeneration() const {
    return structuralGeneration_;
  }
  bool isStructuralCurrent(const SyncCoverGraph &graph) const;
  bool isCurrent(const SyncCoverGraph &graph) const;
  SyncCoverExpansionError refreshMechanismOverlay(const SyncCoverGraph &graph);
  const SyncCoverExpandedArena &getBaseArena() const { return baseArena_; }
  const SyncCoverExpandedArena *getRecurrenceArena(
      SyncCoverScopeId scope) const;
  const SyncCoverExpandedArena *getArena(const SyncCoverDemand &demand) const;
  std::size_t getArenaCount() const { return recurrenceArenas_.size() + 1; }

private:
  SyncCoverExpansionError buildArenaShape(const SyncCoverGraph &graph,
                                          SyncCoverScopeId scope,
                                          unsigned horizon,
                                          std::size_t &remainingNodes,
                                          SyncCoverExpandedArena &arena);
  SyncCoverExpansionError buildAdjacency(
      const SyncCoverGraph &graph, std::size_t edgeBegin, std::size_t edgeEnd,
      const SyncCoverExpandedArena &arena, std::size_t &remainingEdges,
      SyncCoverExpandedAdjacency &adjacency) const;

  SyncCoverExpansionError error_ = SyncCoverExpansionError::None;
  const SyncCoverGraph *owner_ = nullptr;
  SyncCoverExpansionLimits limits_;
  std::size_t graphGeneration_ = 0;
  std::size_t structuralGeneration_ = 0;
  std::size_t structuralExpandedEdges_ = 0;
  SyncCoverExpandedArena baseArena_;
  std::map<SyncCoverScopeId, SyncCoverExpandedArena> recurrenceArenas_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVEREXPANSION_H
