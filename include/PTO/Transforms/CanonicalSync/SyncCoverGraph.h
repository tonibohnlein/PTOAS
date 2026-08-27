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
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace mlir {
namespace pto {

class SyncCoverExpandedProgram;

using SyncCoverNodeId = std::size_t;
using SyncCoverScopeId = std::size_t;
using SyncCoverControlId = std::size_t;
using SyncCoverStorageDomainId = std::size_t;
using SyncCoverStorageAccessId = std::size_t;
using SyncCoverStorageWitnessId = std::size_t;
using SyncCoverStorageAccessFamilyId = std::uint64_t;
using SyncCoverTimelinePosition = std::size_t;

/// Inclusive anchor coordinates. A node with static order n occupies
/// [2*n, 2*n+1], leaving distinct before/after positions.
struct SyncCoverTimelineInterval {
  SyncCoverTimelinePosition begin = 0;
  SyncCoverTimelinePosition end = 0;
};

enum class SyncCoverAnchorKind : std::uint8_t {
  BeforeNode,
  AfterNode,
  ScopeEntry,
  ScopeExit,
  TimelinePoint,
};

struct SyncCoverAnchor {
  SyncCoverAnchorKind kind = SyncCoverAnchorKind::BeforeNode;
  SyncCoverNodeId node = 0;
  SyncCoverScopeId scope = 0;
  SyncCoverTimelinePosition position = 0;
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
  /// one. This models in-order issue on a completion-ordered resource.
  CompletionPreservingIssueOrder,
  /// Pure issue order. It preserves a completion fact that was established
  /// earlier, but cannot establish or aggregate one itself.
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
  std::vector<std::uint32_t> completionTargets;
  /// Earlier same-resource nodes whose completion is implied when this node
  /// completes. These are explicit target facts, not generic issue order.
  std::vector<SyncCoverNodeId> completionDominatedSources;
};

struct SyncCoverEdge {
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  SyncCoverEdgeKind kind = SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  SyncCoverGuard sourceGuard;
  SyncCoverGuard targetGuard;
};

struct SyncCoverDemand {
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  SyncCoverGuard sourceGuard;
  SyncCoverGuard targetGuard;
  /// One completion obligation may have several SSA or memory-hazard causes.
  /// Memory causes require role-compatible overlap witnesses below.
  std::vector<SyncCoverDemandKind> provenanceKinds{SyncCoverDemandKind::SSA};
  std::vector<SyncCoverStorageWitnessId> storageWitnesses;
};

struct SyncCoverScope {
  SyncCoverScopeId id = 0;
  SyncCoverScopeId parent = 0;
  bool mustExecuteWithinParent = false;
  std::optional<SyncCoverTimelineInterval> timeline;
  /// Loop scopes always own an explicit timeline. Positive-distance rows must
  /// name such a loop as their recurrence scope.
  bool isLoop = false;
  /// Structured-control condition under which this region is entered.
  SyncCoverGuard guard;
};

/// A frontend-proven periodic relation for one structured control inside a
/// loop. Each phase names the active alternative and its successor phase.
/// Conservative v1 coverage stores this evidence for later path-sensitive
/// patterns but still contextualizes loop-local controls independently in each
/// virtual copy. Controls without this evidence remain nondeterministic across
/// iterations.
struct SyncCoverControlPhaseRelation {
  SyncCoverScopeId loopScope = 0;
  std::size_t initialPhase = 0;
  std::vector<std::size_t> nextPhase;
  std::vector<unsigned> activeAlternative;
};

struct SyncCoverControl {
  SyncCoverControlId id = 0;
  unsigned alternatives = 0;
  SyncCoverScopeId scope = 0;
  std::optional<SyncCoverControlPhaseRelation> phaseRelation;
};

enum class SyncCoverStorageAccessMode : std::uint8_t {
  Read = 1,
  Write = 2,
  ReadWrite = 3,
};

struct SyncCoverStorageInterval {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

struct SyncCoverStorageDomain {
  SyncCoverStorageDomainId id = 0;
};

struct SyncCoverStorageAccess {
  SyncCoverStorageAccessId id = 0;
  SyncCoverNodeId node = 0;
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageAccessFamilyId family = 0;
  SyncCoverStorageInterval extent;
  SyncCoverStorageAccessMode mode = SyncCoverStorageAccessMode::Read;
  std::optional<unsigned> addressOrdinal;
  /// True only when the interval denotes an exact physical slot rather than a
  /// conservative may-alias range.
  bool exactPhysical = false;
};

struct SyncCoverStorageWitness {
  SyncCoverStorageWitnessId id = 0;
  SyncCoverStorageAccessId sourceAccess = 0;
  SyncCoverStorageAccessId targetAccess = 0;
  SyncCoverStorageInterval overlap;
};

enum class SyncCoverGraphError : std::uint8_t {
  None,
  InvalidNode,
  InvalidScope,
  InvalidControl,
  InvalidGuard,
  InvalidEdgeKind,
  InvalidDemandKind,
  InvalidDistance,
  InvalidOrder,
  InvalidTimeline,
  InvalidCompletionTargets,
  InvalidStorageDomain,
  InvalidStorageAccess,
  InvalidStorageWitness,
  InvalidStorageProvenance,
  DuplicateEdge,
  DuplicateDemand,
  StructureFrozen,
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
  SyncCoverGraph() = default;
  SyncCoverGraph(const SyncCoverGraph &) = delete;
  SyncCoverGraph(SyncCoverGraph &&) = default;
  SyncCoverGraph &operator=(const SyncCoverGraph &) = delete;
  SyncCoverGraph &operator=(SyncCoverGraph &&) = default;

  SyncCoverGraphResult
  addScope(SyncCoverScopeId parent = 0, bool mustExecuteWithinParent = false,
           std::optional<SyncCoverTimelineInterval> timeline = std::nullopt,
           bool isLoop = false, SyncCoverGuard guard = {});
  SyncCoverGraphResult addControl(unsigned alternatives,
                                  SyncCoverScopeId scope = 0);
  SyncCoverGraphResult
  setControlPhaseRelation(SyncCoverControlId control,
                          SyncCoverControlPhaseRelation relation);
  SyncCoverGraphResult setScopeTimeline(SyncCoverScopeId scope,
                                        SyncCoverTimelineInterval timeline);
  SyncCoverGraphResult
  addNode(std::uint32_t resource, std::uint64_t weight, SyncCoverScopeId scope,
          std::size_t order, SyncCoverGuard guard = {},
          std::vector<std::uint32_t> completionTargets = {});
  SyncCoverGraphResult addEdge(SyncCoverEdge edge);
  SyncCoverGraphResult addDemand(SyncCoverDemand demand);
  SyncCoverGraphResult setResourceRecurrenceCarryKind(std::uint32_t resource,
                                                      SyncCoverEdgeKind kind);
  SyncCoverGraphResult addCompletionDominance(SyncCoverNodeId source,
                                              SyncCoverNodeId completionNode);
  SyncCoverGraphError canonicalizeCompletionEdge(SyncCoverEdge &edge) const;
  SyncCoverGraphResult addStorageDomain();
  SyncCoverGraphResult
  addStorageAccess(SyncCoverNodeId node, SyncCoverStorageDomainId domain,
                   SyncCoverStorageAccessFamilyId family,
                   SyncCoverStorageInterval extent,
                   SyncCoverStorageAccessMode mode,
                   std::optional<unsigned> addressOrdinal = std::nullopt,
                   bool exactPhysical = false);
  SyncCoverGraphResult addStorageWitness(SyncCoverStorageAccessId sourceAccess,
                                         SyncCoverStorageAccessId targetAccess);
  SyncCoverGraphResult freezeStructure();

  const std::vector<SyncCoverNode> &getNodes() const { return nodes_; }
  const std::vector<SyncCoverEdge> &getEdges() const { return edges_; }
  const std::vector<SyncCoverDemand> &getDemands() const { return demands_; }
  const std::vector<SyncCoverScope> &getScopes() const { return scopes_; }
  const std::vector<SyncCoverControl> &getControls() const { return controls_; }
  const std::vector<SyncCoverStorageDomain> &getStorageDomains() const {
    return storageDomains_;
  }
  const std::vector<SyncCoverStorageAccess> &getStorageAccesses() const {
    return storageAccesses_;
  }
  const std::vector<SyncCoverStorageWitness> &getStorageWitnesses() const {
    return storageWitnesses_;
  }
  const std::map<std::uint32_t, SyncCoverEdgeKind> &
  getResourceRecurrenceCarryKinds() const {
    return resourceRecurrenceCarryKinds_;
  }

  bool scopeContains(SyncCoverScopeId ancestor,
                     SyncCoverScopeId descendant) const;
  bool scopeMustExecuteWithin(SyncCoverScopeId ancestor,
                              SyncCoverScopeId descendant) const;
  bool completionDominates(SyncCoverNodeId completionNode,
                           SyncCoverNodeId source) const;
  std::optional<SyncCoverScopeId>
  getLowestCommonScope(SyncCoverScopeId first, SyncCoverScopeId second) const;
  std::optional<std::size_t> getScopeLoopDepth(SyncCoverScopeId scope,
                                               bool includeScope = true) const;
  std::optional<SyncCoverScopeId>
  getNearestEnclosingLoop(SyncCoverScopeId scope,
                          bool includeScope = true) const;
  std::optional<SyncCoverScopeId>
  getOwningTimelineScope(SyncCoverScopeId scope) const;
  bool isStructureFrozen() const { return structureFrozen_; }

  SyncCoverGraphResult validate() const;

private:
  friend class SyncCoverExpandedProgram;

  const std::shared_ptr<const std::uint8_t> &getIdentity() const {
    return identity_;
  }
  bool hasValidScope(SyncCoverScopeId scope) const;
  bool canMutateStructure() const { return !structureFrozen_; }
  SyncCoverGraphError
  normalizeAndValidateGuard(SyncCoverGuard &guard,
                            SyncCoverScopeId occurrenceScope) const;
  SyncCoverGraphError completeEndpointGuards(SyncCoverNodeId source,
                                             SyncCoverNodeId target,
                                             SyncCoverScopeId recurrenceScope,
                                             unsigned distance,
                                             SyncCoverGuard &sourceGuard,
                                             SyncCoverGuard &targetGuard) const;
  SyncCoverGraphResult validateScopesControlsAndNodes() const;
  SyncCoverGraphResult validateDemands() const;
  SyncCoverGraphResult validateEdges() const;
  SyncCoverGraphResult validateStorage() const;

  std::vector<SyncCoverNode> nodes_;
  std::vector<SyncCoverEdge> edges_;
  std::vector<SyncCoverDemand> demands_;
  std::vector<SyncCoverStorageDomain> storageDomains_;
  std::vector<SyncCoverStorageAccess> storageAccesses_;
  std::vector<SyncCoverStorageWitness> storageWitnesses_;
  std::map<std::uint32_t, SyncCoverEdgeKind> resourceRecurrenceCarryKinds_;
  using StorageAccessKey =
      std::tuple<SyncCoverNodeId, SyncCoverStorageDomainId,
                 SyncCoverStorageAccessFamilyId, std::uint64_t, std::uint64_t,
                 std::optional<unsigned>>;
  std::map<StorageAccessKey, SyncCoverStorageAccessId> storageAccessIds_;
  std::map<std::pair<SyncCoverStorageAccessId, SyncCoverStorageAccessId>,
           SyncCoverStorageWitnessId>
      storageWitnessIds_;
  using EdgeKey = std::tuple<SyncCoverNodeId, SyncCoverNodeId, SyncCoverScopeId,
                             unsigned, std::vector<SyncCoverGuardLiteral>,
                             std::vector<SyncCoverGuardLiteral>>;
  std::map<EdgeKey, std::size_t> edgeIds_;
  using DemandKey =
      std::tuple<SyncCoverNodeId, SyncCoverNodeId, SyncCoverScopeId, unsigned,
                 std::vector<SyncCoverGuardLiteral>,
                 std::vector<SyncCoverGuardLiteral>>;
  std::map<DemandKey, std::size_t> demandIds_;
  std::vector<SyncCoverScope> scopes_{
      {0, 0, true,
       SyncCoverTimelineInterval{0, std::numeric_limits<std::size_t>::max()},
       false, {}}};
  std::vector<SyncCoverControl> controls_;
  std::shared_ptr<const std::uint8_t> identity_ =
      std::make_shared<const std::uint8_t>(0);
  bool structureFrozen_ = false;
};

bool syncCoverStorageModeReads(SyncCoverStorageAccessMode mode);
bool syncCoverStorageModeWrites(SyncCoverStorageAccessMode mode);

bool normalizeSyncCoverGuard(SyncCoverGuard &guard);
bool syncCoverGuardImplies(const SyncCoverGuard &condition,
                           const SyncCoverGuard &required);
bool syncCoverGuardsCompatible(const SyncCoverGuard &first,
                               const SyncCoverGuard &second);
bool syncCoverEndpointsCoExecute(const SyncCoverGraph &graph,
                                 const SyncCoverEdge &edge);
std::optional<SyncCoverTimelinePosition>
resolveSyncCoverAnchor(const SyncCoverGraph &graph,
                       const SyncCoverAnchor &anchor);
bool syncCoverNodeCanProduceCompletion(const SyncCoverGraph &graph,
                                       SyncCoverNodeId node,
                                       std::uint32_t targetResource);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERGRAPH_H
