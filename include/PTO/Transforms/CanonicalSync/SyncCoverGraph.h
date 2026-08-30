// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

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
using SyncCoverTargetCompletionCertificateId = std::size_t;
using SyncCoverBasicOwnershipCertificateId = std::size_t;
using SyncCoverStorageAccessFamilyId = std::uint64_t;
using SyncCoverTimelinePosition = std::size_t;
using SyncCoverDemandId = std::size_t;

/// Inclusive anchor coordinates. A node with static order n occupies
/// [2*n, 2*n+1], leaving distinct before/after positions.
struct SyncCoverTimelineInterval {
  SyncCoverTimelinePosition begin = 0;
  SyncCoverTimelinePosition end = 0;
};

enum class SyncCoverAnchorKind : std::uint8_t {
  BeforeNode,
  AfterNode,
  /// Immediately before one structured control selects an alternative, or
  /// after all alternatives rejoin. The anchor's node field stores the
  /// SyncCoverControlId and scope stores the control's owning scope.
  ControlEntry,
  ControlExit,
  ScopeEntry,
  ScopeExit,
  /// The first insertion point inside one loop body. Unlike ScopeEntry this
  /// anchor executes once per loop iteration rather than before the loop op.
  LoopBodyEntry,
  /// The insertion point immediately before the loop-body terminator. Unlike
  /// ScopeExit this anchor executes once per loop iteration rather than after
  /// the loop op.
  LoopBodyExit,
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
  /// Fixed issue order on a resource for which completing the target is an
  /// authoritative certificate that every represented source prefix has also
  /// completed. Unlike ordinary issue order, this edge may be traversed before
  /// a completion supply is acquired; it still does not establish completion.
  CertifiedCompletionFrontier,
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
  /// Canonical node whose MLIR operation is the physical insertion anchor.
  /// Multi-phase macro nodes share the representative of their operation.
  SyncCoverNodeId physicalAnchor = 0;
  SyncCoverNodeId physicalExit = 0;
  /// Target-provided contract: a synchronization set issued after this node
  /// certifies completion of the previously issued prefix on its resource.
  bool completionSignalCoversIssuedPrefix = false;
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
  /// Number of original obligations interned into this canonical row.
  std::size_t originalDemandCount = 1;
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
/// Recurrence demand construction follows this automaton exactly. Coverage
/// retains the guards on resulting rows and contextualizes loop-local controls
/// independently in each virtual copy; the demand basis has already removed
/// unreachable phase/distance combinations. Controls without this evidence
/// remain nondeterministic across iterations.
struct SyncCoverControlPhaseRelation {
  SyncCoverScopeId loopScope = 0;
  std::size_t initialPhase = 0;
  std::vector<std::size_t> nextPhase;
  std::vector<unsigned> activeAlternative;
};

/// Exact frontend evidence that one branch alternative means that the nearest
/// enclosing loop has a successor iteration. The MLIR adapter only records
/// this relation for `iv + step < upperBound`.
struct SyncCoverControlSuccessorRelation {
  SyncCoverScopeId loopScope = 0;
  unsigned hasSuccessorAlternative = 0;
};

struct SyncCoverControl {
  SyncCoverControlId id = 0;
  unsigned alternatives = 0;
  SyncCoverScopeId scope = 0;
  std::optional<SyncCoverControlPhaseRelation> phaseRelation;
  std::optional<SyncCoverControlSuccessorRelation> successorRelation;
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

enum class SyncCoverStorageDomainRole : std::uint8_t {
  Unspecified,
  Other,
  L1Tile,
  L0Left,
  L0Right,
  Accumulator,
};

struct SyncCoverStorageDomain {
  SyncCoverStorageDomainId id = 0;
  SyncCoverStorageDomainRole role = SyncCoverStorageDomainRole::Unspecified;
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

/// Target-qualified completion facts admitted only after frontend analysis has
/// proved the corresponding physical-storage lifecycle.  These facts do not
/// change generic resource completion or issue-order semantics.
enum class SyncCoverTargetCompletionKind : std::uint8_t {
  Mte1L0ReadyPrefix,
  MToFixAccumulatorBoundary,
};

struct SyncCoverTargetCompletionCertificate {
  SyncCoverTargetCompletionCertificateId id = 0;
  SyncCoverTargetCompletionKind kind =
      SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix;
  /// Node after which the target-qualified set is issued.
  SyncCoverNodeId completionNode = 0;
  /// Physical node before which the matching wait is issued.
  SyncCoverNodeId target = 0;
  /// Authoritative event-domain resources. They are independent of the
  /// resources of multiphase physical anchor nodes.
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  /// Exact storage domains participating in the certified lifecycle.
  std::vector<SyncCoverStorageDomainId> storageDomains;
  /// Exact distance-zero demand rows discharged by this certificate.
  std::vector<SyncCoverDemandId> demands;
};

/// Opaque graph-resource IDs corresponding to the target-specific certificate
/// vocabulary. The MLIR adapter configures these once; the graph core remains
/// independent of PTO pipeline enum headers.
struct SyncCoverTargetCompletionResources {
  std::uint32_t mte1 = 0;
  std::uint32_t matrix = 0;
  std::uint32_t fix = 0;
};

/// Basic exact-slot ownership protocols. Composite and hierarchical ownership
/// are intentionally not part of this certificate vocabulary.
enum class SyncCoverBasicOwnershipKind : std::uint8_t {
  L0Operand,
  L1Tile,
  L0Accumulator,
};

enum class SyncCoverBasicOwnershipProtocolKind : std::uint8_t {
  RoundTrip,
  AlternatingPrefetch,
};

struct SyncCoverBasicOwnershipSlot {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval extent;
  /// Complete set of exact physical accesses to this slot that can execute in
  /// the certified loop. Initial producers outside the loop are also named.
  std::vector<SyncCoverStorageAccessId> accesses;
};

struct SyncCoverBasicOwnershipLane {
  std::size_t id = 0;
  std::vector<SyncCoverBasicOwnershipSlot> slots;
};

struct SyncCoverBasicOwnershipUse {
  /// Lane consumed by this iteration/path and lane produced for a future use.
  std::size_t lane = 0;
  std::size_t producerLane = 0;
  std::vector<SyncCoverNodeId> producers;
  std::vector<SyncCoverNodeId> consumers;
  SyncCoverAnchor writeAcquireAnchor;
  SyncCoverAnchor readyAnchor;
  SyncCoverAnchor readAcquireAnchor;
  SyncCoverAnchor releaseAnchor;
};

struct SyncCoverBasicOwnershipPath {
  SyncCoverScopeId scope = 0;
  std::vector<SyncCoverBasicOwnershipUse> uses;
};

struct SyncCoverBasicOwnershipCertificate {
  SyncCoverBasicOwnershipCertificateId id = 0;
  SyncCoverBasicOwnershipKind kind = SyncCoverBasicOwnershipKind::L0Operand;
  SyncCoverBasicOwnershipProtocolKind protocol =
      SyncCoverBasicOwnershipProtocolKind::RoundTrip;
  SyncCoverScopeId loopScope = 0;
  std::uint32_t producerResource = 0;
  std::uint32_t consumerResource = 0;
  std::vector<SyncCoverBasicOwnershipLane> lanes;
  std::vector<SyncCoverBasicOwnershipPath> paths;

  /// Alternating-prefetch-only state. The periodic control selects the current
  /// lane. Initial producers establish the initially ready lane before entry;
  /// the other named lanes start free for a guarded successor prefetch.
  std::optional<SyncCoverControlId> periodicControl;
  std::vector<SyncCoverNodeId> initialProducers;
  SyncCoverAnchor initialWriteAcquireAnchor;
  SyncCoverAnchor initialReadyAnchor;
  std::size_t initialReadyLane = 0;
  std::vector<std::size_t> initiallyFreeLanes;
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
  InvalidTargetCompletionCertificate,
  InvalidBasicOwnershipCertificate,
  ArithmeticOverflow,
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
  SyncCoverGraphResult
  setControlSuccessorRelation(SyncCoverControlId control,
                              SyncCoverControlSuccessorRelation relation);
  SyncCoverGraphResult setScopeTimeline(SyncCoverScopeId scope,
                                        SyncCoverTimelineInterval timeline);
  SyncCoverGraphResult
  addNode(std::uint32_t resource, std::uint64_t weight, SyncCoverScopeId scope,
          std::size_t order, SyncCoverGuard guard = {},
          std::vector<std::uint32_t> completionTargets = {},
          std::optional<SyncCoverNodeId> physicalAnchor = std::nullopt,
          bool completionSignalCoversIssuedPrefix = false);
  SyncCoverGraphResult addEdge(SyncCoverEdge edge);
  SyncCoverGraphResult addDemand(SyncCoverDemand demand);
  SyncCoverGraphResult setResourceRecurrenceCarryKind(std::uint32_t resource,
                                                      SyncCoverEdgeKind kind);
  SyncCoverGraphResult
  setBlockingTargetedBarrierResources(std::vector<std::uint32_t> resources);
  SyncCoverGraphResult
  setTargetCompletionResources(SyncCoverTargetCompletionResources resources);
  bool supportsBlockingTargetedBarrier(std::uint32_t resource) const;
  SyncCoverGraphResult
  setBlockingTargetedBarrierPrefix(std::uint32_t resource,
                                   SyncCoverNodeId physicalTarget,
                                   std::vector<SyncCoverNodeId> issuedSources);
  const std::map<std::pair<std::uint32_t, SyncCoverNodeId>,
                 std::vector<SyncCoverNodeId>> &
  getBlockingTargetedBarrierPrefixes() const {
    return blockingTargetedBarrierPrefixes_;
  }
  SyncCoverGraphResult addCompletionDominance(SyncCoverNodeId source,
                                              SyncCoverNodeId completionNode);
  SyncCoverGraphResult setPhysicalExit(SyncCoverNodeId node,
                                       SyncCoverNodeId physicalExit);
  SyncCoverGraphError canonicalizeCompletionEdge(SyncCoverEdge &edge) const;
  SyncCoverGraphResult
  addStorageDomain(SyncCoverStorageDomainRole role =
                       SyncCoverStorageDomainRole::Unspecified);
  SyncCoverGraphResult
  addStorageAccess(SyncCoverNodeId node, SyncCoverStorageDomainId domain,
                   SyncCoverStorageAccessFamilyId family,
                   SyncCoverStorageInterval extent,
                   SyncCoverStorageAccessMode mode,
                   std::optional<unsigned> addressOrdinal = std::nullopt,
                   bool exactPhysical = false);
  SyncCoverGraphResult addStorageWitness(SyncCoverStorageAccessId sourceAccess,
                                         SyncCoverStorageAccessId targetAccess);
  SyncCoverGraphResult addTargetCompletionCertificate(
      SyncCoverTargetCompletionKind kind, SyncCoverNodeId completionNode,
      SyncCoverNodeId target, std::uint32_t sourceResource,
      std::uint32_t targetResource,
      std::vector<SyncCoverStorageDomainId> storageDomains,
      std::vector<SyncCoverDemandId> demands);
  SyncCoverGraphResult
  addBasicOwnershipCertificate(SyncCoverBasicOwnershipCertificate certificate);
  SyncCoverGraphResult addBasicOwnershipCertificates(
      std::vector<SyncCoverBasicOwnershipCertificate> certificates);
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
  const std::vector<SyncCoverTargetCompletionCertificate> &
  getTargetCompletionCertificates() const {
    return targetCompletionCertificates_;
  }
  const std::vector<SyncCoverBasicOwnershipCertificate> &
  getBasicOwnershipCertificates() const {
    return basicOwnershipCertificates_;
  }
  bool hasTargetCompletionCertificate(SyncCoverTargetCompletionKind kind,
                                      SyncCoverNodeId completionNode,
                                      SyncCoverNodeId target,
                                      std::uint32_t sourceResource,
                                      std::uint32_t targetResource,
                                      SyncCoverDemandId demand) const;
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
  SyncCoverGraphResult validateTargetCompletionCertificates() const;
  SyncCoverGraphResult validateBasicOwnershipCertificates() const;

  std::vector<SyncCoverNode> nodes_;
  std::vector<SyncCoverEdge> edges_;
  std::vector<SyncCoverDemand> demands_;
  std::vector<SyncCoverStorageDomain> storageDomains_;
  std::vector<SyncCoverStorageAccess> storageAccesses_;
  std::vector<SyncCoverStorageWitness> storageWitnesses_;
  std::vector<SyncCoverTargetCompletionCertificate>
      targetCompletionCertificates_;
  std::vector<SyncCoverBasicOwnershipCertificate> basicOwnershipCertificates_;
  std::optional<SyncCoverTargetCompletionResources> targetCompletionResources_;
  std::map<std::uint32_t, SyncCoverEdgeKind> resourceRecurrenceCarryKinds_;
  std::vector<std::uint32_t> blockingTargetedBarrierResources_;
  std::map<std::pair<std::uint32_t, SyncCoverNodeId>,
           std::vector<SyncCoverNodeId>>
      blockingTargetedBarrierPrefixes_;
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
      {0,
       0,
       true,
       SyncCoverTimelineInterval{0, std::numeric_limits<std::size_t>::max()},
       false,
       {}}};
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
