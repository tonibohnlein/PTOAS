// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverGraph.h - Synchronization covering graph -------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERGRAPH_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERGRAPH_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverNodeId = std::size_t;
using SyncCoverScopeId = std::size_t;
using SyncCoverControlId = std::size_t;
using SyncCoverMechanismId = std::size_t;
using SyncCoverTimelinePosition = std::size_t;

struct SyncCoverTimelineInterval {
  SyncCoverTimelinePosition begin = 0;
  SyncCoverTimelinePosition end = 0;
};

enum class SyncCoverAnchorKind : std::uint8_t {
  BeforeNode,
  AfterNode,
  ScopeEntry,
  ScopeExit,
};

struct SyncCoverAnchor {
  SyncCoverAnchorKind kind = SyncCoverAnchorKind::BeforeNode;
  SyncCoverNodeId node = 0;
  SyncCoverScopeId scope = 0;
};

struct SyncCoverGuardLiteral {
  SyncCoverControlId control = 0;
  unsigned alternative = 0;

  bool operator<(const SyncCoverGuardLiteral &other) const;
  bool operator==(const SyncCoverGuardLiteral &other) const;
};

/// A conjunction of structured-control alternatives. Literals are normalized
/// by SyncCoverGraph when a node, edge, or demand is added.
struct SyncCoverGuard {
  std::vector<SyncCoverGuardLiteral> literals;
};

enum class SyncCoverEdgeKind : std::uint8_t {
  /// Preserves an already-established completion fact but does not establish
  /// one. Its availability is an explicit operation/architecture capability.
  CompletionPreservingIssueOrder,
  /// Pure issue order that cannot carry a completion fact.
  NonCompletionPreservingIssueOrder,
  /// Establishes completion at the target.
  CompletionSupply,
};

enum class SyncCoverDemandKind : std::uint8_t {
  SSA,
  MemoryRAW,
  MemoryWAR,
  MemoryWAW,
};

struct SyncCoverNode {
  SyncCoverNodeId id = 0;
  std::uint32_t resource = 0;
  std::uint64_t weight = 1;
  SyncCoverScopeId scope = 0;
  std::size_t order = 0;
  SyncCoverGuard guard;
  /// Event destinations for which a Produce immediately after this node
  /// observes this node's completion. The source resource is node.resource.
  std::vector<std::uint32_t> completionTargets;
};

struct SyncCoverEdge {
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  SyncCoverEdgeKind kind = SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  SyncCoverGuard sourceGuard;
  SyncCoverGuard targetGuard;
  std::optional<SyncCoverMechanismId> mechanism;
};

struct SyncCoverDemand {
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  SyncCoverDemandKind kind = SyncCoverDemandKind::MemoryRAW;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  SyncCoverGuard sourceGuard;
  SyncCoverGuard targetGuard;
};

struct SyncCoverScope {
  SyncCoverScopeId id = 0;
  SyncCoverScopeId parent = 0;
  bool mustExecuteWithinParent = false;
  std::optional<SyncCoverTimelineInterval> timeline;
  bool isLoop = false;
};

struct SyncCoverControl {
  SyncCoverControlId id = 0;
  unsigned alternatives = 0;
  SyncCoverScopeId scope = 0;
};

enum class SyncCoverGraphError : std::uint8_t {
  None,
  InvalidNode,
  InvalidScope,
  InvalidControl,
  InvalidGuard,
  InvalidDistance,
  InvalidOrder,
  InvalidTimeline,
  IncompatibleEndpoints,
  ZeroDistanceSelfEdge,
  ZeroDistanceSelfDemand,
  ZeroDistanceCycle,
};

/// A successful mutation contains the inserted object's index. Failed
/// mutations leave the graph unchanged. validate() reports the index of the
/// first invalid object when one exists.
struct SyncCoverGraphResult {
  SyncCoverGraphError error = SyncCoverGraphError::None;
  std::optional<std::size_t> index;

  explicit operator bool() const { return error == SyncCoverGraphError::None; }
};

class SyncCoverGraph {
public:
  SyncCoverGraphResult
  addScope(SyncCoverScopeId parent = 0, bool mustExecuteWithinParent = false,
           std::optional<SyncCoverTimelineInterval> timeline = std::nullopt,
           bool isLoop = false);
  SyncCoverGraphResult addControl(unsigned alternatives,
                                  SyncCoverScopeId scope = 0);
  SyncCoverGraphResult
  addNode(std::uint32_t resource, std::uint64_t weight, SyncCoverScopeId scope,
          std::size_t order, SyncCoverGuard guard = {},
          std::vector<std::uint32_t> completionTargets = {});
  SyncCoverGraphResult addEdge(SyncCoverEdge edge);
  SyncCoverGraphResult addDemand(SyncCoverDemand demand);

  const std::vector<SyncCoverNode> &getNodes() const { return nodes_; }
  const std::vector<SyncCoverEdge> &getEdges() const { return edges_; }
  const std::vector<SyncCoverDemand> &getDemands() const { return demands_; }
  const std::vector<SyncCoverScope> &getScopes() const { return scopes_; }
  const std::vector<SyncCoverControl> &getControls() const { return controls_; }

  SyncCoverGraphResult validate() const;

private:
  bool hasValidScope(SyncCoverScopeId scope) const;
  bool isScopeAncestor(SyncCoverScopeId ancestor,
                       SyncCoverScopeId descendant) const;
  SyncCoverGraphError
  normalizeAndValidateGuard(SyncCoverGuard &guard,
                            SyncCoverScopeId occurrenceScope) const;
  SyncCoverGraphError completeEndpointGuards(SyncCoverNodeId source,
                                             SyncCoverNodeId target,
                                             SyncCoverScopeId recurrenceScope,
                                             unsigned distance,
                                             SyncCoverGuard &sourceGuard,
                                             SyncCoverGuard &targetGuard) const;

  std::vector<SyncCoverNode> nodes_;
  std::vector<SyncCoverEdge> edges_;
  std::vector<SyncCoverDemand> demands_;
  std::vector<SyncCoverScope> scopes_{
      {0, 0, true,
       SyncCoverTimelineInterval{0, std::numeric_limits<std::size_t>::max()},
       false}};
  std::vector<SyncCoverControl> controls_;
};

bool normalizeSyncCoverGuard(SyncCoverGuard &guard);
bool syncCoverGuardImplies(const SyncCoverGuard &condition,
                           const SyncCoverGuard &required);
bool syncCoverGuardsCompatible(const SyncCoverGuard &first,
                               const SyncCoverGuard &second);
std::optional<SyncCoverTimelinePosition>
resolveSyncCoverAnchor(const SyncCoverGraph &graph,
                       const SyncCoverAnchor &anchor);
bool syncCoverNodeCanProduceCompletion(const SyncCoverGraph &graph,
                                       SyncCoverNodeId node,
                                       std::uint32_t targetResource);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERGRAPH_H
