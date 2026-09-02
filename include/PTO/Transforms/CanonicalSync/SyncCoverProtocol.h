// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverProtocol.h - Event lifecycle certificates -----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERPROTOCOL_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERPROTOCOL_H

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {

class SyncCoverStorageLifecycleIndex;
class SyncCoverStorageProtocolGroupIndex;
class SyncCoverStorageProtocolSeedIndex;

using SyncCoverProtocolChannelId = std::size_t;

/// Target-owned facts consumed by the target-neutral lifecycle verifier. The
/// provider must populate the exact directed HardEvent table and the usable ID
/// set from one versioned device specification.
struct SyncCoverProtocolTargetContract {
  struct EventCapability {
    std::uint32_t sourceResource = 0;
    std::uint32_t targetResource = 0;
    SyncCoverOrderingRequirementMask suppliedRequirements = 0;

    bool operator<(const EventCapability &other) const;
    bool operator==(const EventCapability &other) const;
  };

  struct RearmFact {
    std::uint32_t evidence = 0;
    std::uint32_t fromWaitResource = 0;
    SyncCoverAnchor fromWaitAnchor;
    SyncCoverGuard fromWaitGuard;
    std::uint32_t toSetResource = 0;
    SyncCoverAnchor toSetAnchor;
    SyncCoverGuard toSetGuard;
    SyncCoverScopeId loopScope = 0;
    unsigned iterationDistance = 0;
    std::size_t width = 0;

    bool operator<(const RearmFact &other) const;
    bool operator==(const RearmFact &other) const;
  };

  /// Every entry is copied from one versioned, core-specific device profile.
  /// The requirement mask prevents an ordinary HardEvent from being promoted
  /// into cache visibility or a hardware-special ordering certificate.
  std::vector<EventCapability> eventCapabilities;
  std::vector<unsigned> compilerUsableEventIds;
  /// A legal directed event widens to the issued source-pipeline prefix only
  /// when the versioned target contract explicitly authorizes that effect.
  bool directEventCompletesSourcePrefix = false;
  /// Exact provider-owned Wait-to-Set facts. An evidence number alone never
  /// certifies a different resource, guard, anchor, scope, width, or distance.
  std::vector<RearmFact> certifiedRearmFacts;

  bool supportsEvent(std::uint32_t source, std::uint32_t target,
                     SyncCoverOrderingRequirementMask requirements) const;
  const RearmFact *findRearmFact(std::uint32_t evidence) const;
};

enum class SyncCoverEventProtocolKind : std::uint8_t {
  SingleShot,
  ProvenNoOverlap,
  RoundTrip,
  RotatingLanes,
  /// A target-typed collection of lifecycle channels derived from one storage
  /// SCC.  Its legality is established by the token automaton rather than by a
  /// fixed two-channel recipe shape.
  LifecycleNetwork,
};

/// SameIteration pairs one body Set with one body Wait in the same dynamic
/// iteration. LoopCarry primes each lane at entry, consumes before reuse,
/// produces after release, and drains every lane at exit.
enum class SyncCoverEventChannelFlow : std::uint8_t {
  SingleShot,
  SameIteration,
  LoopCarry,
};

struct SyncCoverProtocolLoopSchedule {
  SyncCoverScopeId scope = 0;
  bool mayExecuteZeroTimes = true;
  /// If present, phases are the graph-owned reachable phases of this control.
  /// If absent, the loop has one unconditional phase. Callers cannot invent a
  /// transition relation or phase guards.
  std::optional<SyncCoverControlId> phaseControl;
  /// Lane selected by each declared graph phase. The vector has one element
  /// for an unconditional loop or exactly matches the authoritative relation.
  std::vector<std::size_t> laneByPhase;
};

/// One balanced body transfer on a logical event channel.  A channel may
/// contain several transfers when one physical lane pool is reused at
/// multiple ordered cuts in the same iteration.  Empty activePhases means
/// every reachable phase.  Explicit lane numbers are used by storage
/// lifecycle protocols; legacy direct protocols leave transfers empty and
/// continue to select their lane from the loop schedule.
struct SyncCoverEventTransfer {
  std::size_t id = 0;
  SyncCoverCutPoint set;
  SyncCoverCutPoint wait;
  std::size_t setLane = 0;
  std::size_t waitLane = 0;
  std::vector<std::size_t> activePhases;
};

enum class SyncCoverProtocolActionKind : std::uint8_t { Set, Wait };

enum class SyncCoverProtocolActionSegment : std::uint8_t {
  Entry,
  Body,
  Exit,
};

enum class SyncCoverProtocolActionGuard : std::uint8_t {
  Always,
  LoopNonEmpty,
  LoopEmpty,
  /// Execute at a body cut only when the current induction value is the loop
  /// lower bound. Zero-trip paths execute no such action.
  FirstIteration,
  NotFirstIteration,
  HasSuccessor,
};

/// One explicit action in a lifecycle channel.  Explicit recipes are used for
/// schedules whose token circulation cannot be described by the legacy
/// prime-all/body/drain-all channel shape.  The temporal guard is checked by
/// the protocol automaton and is not treated as an arbitrary graph predicate.
struct SyncCoverProtocolAction {
  std::size_t id = 0;
  SyncCoverProtocolActionKind kind = SyncCoverProtocolActionKind::Set;
  SyncCoverProtocolActionSegment segment = SyncCoverProtocolActionSegment::Body;
  SyncCoverCutPoint point;
  std::size_t lane = 0;
  SyncCoverProtocolActionGuard guard = SyncCoverProtocolActionGuard::Always;
  std::vector<std::size_t> activePhases;
};

enum class SyncCoverProtocolSupplyKind : std::uint8_t {
  /// The named Set token is consumed directly by the named Wait at the
  /// declared loop displacement.
  TokenPair,
  /// A locally verified token pair establishes a completion fact that is
  /// retained through the child-loop exit and consumed in the enclosing
  /// lifetime-loop coordinate. The protocol automaton must witness the exact
  /// named Set as the physical mate of the exact named Wait at the declared
  /// invocation displacement.
  CompletionExport,
};

/// One completion rectangle backed by two actions in the same channel.
/// `distance` is the exact dynamic iteration displacement from the Set to the
/// consuming Wait.  The verifier rejects references that do not name a legal,
/// dynamically balanced Set/Wait token pair.
struct SyncCoverProtocolSupply {
  std::size_t setAction = 0;
  std::size_t waitAction = 0;
  unsigned distance = 0;
  /// Loop whose dynamic iterations the distance counts. Empty means the
  /// protocol's principal loop. Hierarchical protocols name their enclosing
  /// lifetime loop explicitly so an inner carry is never reinterpreted as a
  /// parent-loop transfer. A zero-distance CompletionExport summarizes a
  /// relation wholly contained in one invocation of the child loop.
  std::optional<SyncCoverScopeId> distanceScope;
  SyncCoverProtocolSupplyKind kind = SyncCoverProtocolSupplyKind::TokenPair;
};

struct SyncCoverEventChannel {
  SyncCoverProtocolChannelId id = 0;
  SyncCoverEventChannelFlow flow = SyncCoverEventChannelFlow::SingleShot;
  SyncCoverCutPoint set;
  SyncCoverCutPoint wait;
  std::size_t width = 1;
  /// SameIteration and SingleShot use zero. LoopCarry uses the exact dynamic
  /// iteration displacement between a body Set and its consuming body Wait.
  unsigned distance = 0;
  SyncCoverOrderingRequirementMask suppliedRequirements = 0;
  /// Empty means every reachable phase. Otherwise the sorted unique phase set
  /// controls both body actions.
  std::vector<std::size_t> activePhases;
  bool exportsCompletionAtExit = false;
  /// When nonempty, these transfers are the complete body recipe for this
  /// channel.  The legacy set/wait fields retain the canonical resource pair
  /// and the first transfer for stable hashing and compatibility.
  std::vector<SyncCoverEventTransfer> transfers;
  /// Explicit token recipe and its certified completion rectangles.  These
  /// vectors are either both empty or both nonempty, and are mutually
  /// exclusive with `transfers`.
  std::vector<SyncCoverProtocolAction> actions;
  std::vector<SyncCoverProtocolSupply> supplies;
};

/// An externally certified order can rearm a one-way channel. It is not
/// inferred from a demand that still needs synchronization. Nonzero evidence
/// identifies the target/provider contract that established the relation.
struct SyncCoverProtocolRearmProof {
  SyncCoverProtocolChannelId fromWaitChannel = 0;
  SyncCoverProtocolChannelId toSetChannel = 0;
  unsigned iterationDistance = 0;
  std::uint32_t evidence = 0;
};

struct SyncCoverEventProtocol {
  SyncCoverMechanismId mechanism = 0;
  SyncCoverEventProtocolKind kind = SyncCoverEventProtocolKind::SingleShot;
  std::optional<SyncCoverProtocolLoopSchedule> loop;
  std::vector<SyncCoverEventChannel> channels;
  std::vector<SyncCoverProtocolRearmProof> rearmProofs;
  /// Optional enclosing loop that owns event allocation, priming, and
  /// draining.  When it differs from `loop->scope`, the inner schedule is a
  /// repeatable invocation transformer: outer entry/exit actions execute once
  /// while inner entry/body/exit actions execute for every outer iteration.
  std::optional<SyncCoverScopeId> lifetimeScope;
  bool lifetimeMayExecuteZeroTimes = true;
};

struct SyncCoverProtocolLimits {
  std::size_t maximumGraphNodes = 1U << 16;
  std::size_t maximumGraphEdges = 1U << 18;
  std::size_t maximumGraphDemands = 1U << 16;
  std::size_t maximumGraphScopes = 1U << 16;
  std::size_t maximumGraphRegions = 1U << 17;
  std::size_t maximumGraphControls = 1U << 16;
  std::size_t maximumGraphStorageDomains = 1U << 16;
  std::size_t maximumGraphStorageAccesses = 1U << 18;
  std::size_t maximumGraphStorageWitnesses = 1U << 18;
  std::size_t maximumTargetCapabilities = 256;
  std::size_t maximumTargetEventIds = 64;
  std::size_t maximumTargetRearmFacts = 1U << 12;
  std::size_t maximumChannels = 256;
  std::size_t maximumChannelLaneIncidences = 1U << 16;
  std::size_t maximumProtocols = 1U << 14;
  std::size_t maximumWorlds = 256;
  std::size_t maximumWorldMechanismIncidences = 1U << 18;
  std::size_t maximumResultRows = 1U << 16;
  std::size_t maximumResultWords = 1U << 22;
  std::size_t maximumGuardLiterals = 1U << 18;
  std::size_t maximumPhaseIncidences = 1U << 18;
  std::size_t maximumReachablePhases = 1U << 12;
  std::size_t maximumTripCounts = 1U << 12;
  std::size_t maximumInvocationSequences = 1U << 12;
  std::size_t maximumDynamicActions = 1U << 14;
  std::size_t maximumTotalDynamicActions = 1U << 20;
  std::size_t maximumLaneInitializationWork = 1U << 20;
  std::size_t maximumAutomatonEdges = 1U << 18;
  std::size_t maximumRearmProofs = 1U << 12;
  std::size_t maximumRearmProofLaneIncidences = 1U << 16;
  std::size_t maximumRearmLookupWork = 1U << 20;
  std::size_t maximumRearmQueries = 1U << 18;
  std::size_t maximumCompletionExportQueries = 1U << 18;
  std::size_t maximumReachabilityWords = 1U << 22;
  std::size_t maximumReachabilityWork = 1U << 24;
  std::size_t maximumCoverageStates = 1U << 22;
  std::size_t maximumCoverageTransitions = 1U << 24;
  std::size_t maximumExitExports = 1U << 16;
  std::size_t maximumExitExportGuardLiterals = 1U << 18;
  std::size_t maximumLifecycleSccs = 1U << 12;
  std::size_t maximumLifecycleVertices = 1U << 16;
  std::size_t maximumLifecycleEdges = 1U << 18;
  std::size_t maximumLifecycleAccessIncidences = 1U << 20;
  std::size_t maximumLifecycleProposals = 1U << 12;
  std::size_t maximumLifecycleProtocols = 1U << 14;
  std::size_t maximumLifecycleLanes = 1U << 16;
  std::size_t maximumLifecycleSlots = 1U << 16;
  std::size_t maximumLifecyclePaths = 1U << 16;
  std::size_t maximumLifecycleTransfers = 1U << 18;
  std::size_t maximumLifecycleNodeReferences = 1U << 20;
  std::size_t maximumLifecycleConnectorIncidences = 1U << 20;
  std::size_t maximumLifecycleConnectorGuardIncidences = 1U << 20;
  std::size_t maximumLifecycleGroupIncidences = 1U << 20;
  std::size_t maximumLifecycleCatalogPayloadIncidences = 1U << 22;
  std::size_t maximumStorageWitnessIncidences = 1U << 20;
  std::size_t maximumLifecycleDomainIncidences = 1U << 20;
  std::size_t maximumReservations = 1U << 14;
  std::size_t maximumReservationDomains = 1U << 12;
  std::size_t maximumReservationIdIncidences = 1U << 16;
  std::size_t maximumChannelRequests = 1U << 16;
  std::size_t maximumAllocatedEventIds = 1U << 16;
};

enum class SyncCoverProtocolError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidTargetContract,
  InvalidProtocol,
  InvalidTokenLifecycle,
  LimitExceeded,
  WorkLimitExceeded,
  ResourceInfeasible,
};

struct SyncCoverProtocolExitExport {
  SyncCoverProtocolChannelId channel = 0;
  SyncCoverGuard guard;
  bool availableOnZeroTrip = false;
  bool availableOnNonzeroTrip = false;
};

struct SyncCoverProtocolStatistics {
  std::size_t reachablePhases = 0;
  std::size_t tripCountsChecked = 0;
  std::size_t maximumDynamicActions = 0;
  std::size_t automatonEdges = 0;
  std::size_t rearmQueries = 0;
  std::size_t completionExportQueries = 0;
  std::size_t rearmProofLaneIncidences = 0;
  std::size_t rearmLookupWork = 0;
  std::size_t totalDynamicActions = 0;
  std::size_t laneInitializationWork = 0;
  std::size_t reachabilityWork = 0;
  std::size_t coverageTransitions = 0;
};

struct SyncCoverProtocolVerificationResult {
  SyncCoverProtocolError error = SyncCoverProtocolError::None;
  SyncCoverProtocolStatistics statistics;
  std::vector<SyncCoverProtocolExitExport> exitExports;
  /// One bit per declared channel supply. The token automaton records the bit
  /// only after observing the exact named Set consumed by the exact named
  /// Wait at the declared local-iteration or lifetime-invocation distance.
  /// The matrix is retained on InvalidTokenLifecycle so bounded synthesis can
  /// discard unproved optional completion exports without weakening the
  /// verifier's fail-closed result.
  std::vector<std::vector<bool>> supplyWitnesses;
  std::optional<std::size_t> invalidIndex;

  explicit operator bool() const {
    return error == SyncCoverProtocolError::None;
  }
};

SyncCoverProtocolVerificationResult verifySyncCoverEventProtocol(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverEventProtocol &protocol, SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

/// One exact-world coverage result for lifecycle-verified direct protocol
/// rectangles. This does not trust or import legacy supply bindings.
struct SyncCoverProtocolCoverageResult {
  SyncCoverProtocolError error = SyncCoverProtocolError::None;
  std::vector<SyncCoverDemandSet> coveredByWorld;
  SyncCoverProtocolStatistics statistics;
  std::optional<std::size_t> invalidIndex;

  explicit operator bool() const {
    return error == SyncCoverProtocolError::None;
  }
};

SyncCoverProtocolCoverageResult computeSyncCoverProtocolExactWorlds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverExactWorld> &worlds,
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

/// Exact-world grounding restricted to a caller-supplied subset of demand
/// rows. The result bitsets retain the full graph demand width and contain no
/// facts for unqueried rows. This is used after a direct union is already
/// known, so optional group closure pays graph-search work only for possible
/// additional coverage.
SyncCoverProtocolCoverageResult computeSyncCoverProtocolExactWorldsForDemands(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverExactWorld> &worlds,
    const SyncCoverDemandSet &queriedDemands,
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

/// Conservatively grounds only completion rectangles supplied directly by
/// each enabled protocol.  Unlike `computeSyncCoverProtocolExactWorlds`, this
/// routine does not search for additional transitive composition.  It is used
/// while preparing the bounded singleton lifecycle catalog; group composition
/// is added separately through typed connector proposals.
SyncCoverProtocolCoverageResult computeSyncCoverProtocolDirectWorlds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverExactWorld> &worlds,
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

/// Computes a bounded transitive closure of already certified direct demand
/// relations. Connector paths retain their full guard condition; a demand is
/// proposed only when the union of reaching path conditions covers every
/// feasible alternative not fixed by the demand itself. This is an optional
/// proposal accelerator, not correctness proof. Every retained group must be
/// re-evaluated by computeSyncCoverProtocolExactWorlds.
SyncCoverProtocolCoverageResult computeSyncCoverProtocolConnectorClosure(
    const SyncCoverGraph &graph, const SyncCoverDemandSet &directCoverage,
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

/// A generic storage-lifecycle SCC. It is a proposal/certificate input, not a
/// selectable mechanism: later synthesis must still build and verify a full
/// physical event recipe.
struct SyncCoverLifecycleScc {
  SyncCoverScopeId loopScope = 0;
  /// Exact physical pipeline resources in this strongly connected ownership
  /// component.  Schedule synthesis must not infer ownership from resources
  /// outside this set, even when they access an adjacent storage interval.
  std::vector<std::uint32_t> resources;
  std::vector<SyncCoverNodeId> nodes;
  std::vector<SyncCoverDemandId> demands;
  std::vector<SyncCoverStorageDomainId> storageDomains;
  std::vector<SyncCoverStorageAccessId> storageAccesses;
  std::vector<SyncCoverStorageWitnessId> storageWitnesses;
  unsigned maximumDistance = 0;
};

struct SyncCoverLifecycleSccResult {
  SyncCoverProtocolError error = SyncCoverProtocolError::None;
  std::vector<SyncCoverLifecycleScc> components;
  std::size_t lifecycleAccessIncidences = 0;
  std::optional<std::size_t> invalidIndex;

  explicit operator bool() const {
    return error == SyncCoverProtocolError::None;
  }
};

SyncCoverLifecycleSccResult discoverSyncCoverLifecycleSccs(
    const SyncCoverGraph &graph, SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

struct SyncCoverLifecycleSlot {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval extent;
  std::vector<SyncCoverStorageAccessId> accesses;
};

/// One immutable, target-verified proposal derived from a storage-lifecycle
/// SCC. Protocols are independently token-safe physical recipes. Direct
/// rectangle grounding is composed only through a bounded, guard-aware
/// connector closure, and exactCoverage is never caller-authored proof.
struct SyncCoverLifecycleProposal {
  std::size_t id = 0;
  std::size_t lifecycleScc = 0;
  /// True only for the independently verified per-demand correctness basis.
  /// Grouped lifecycle proposals are optional cover improvements and leave
  /// this false so event-scarcity repair can deterministically fall back to
  /// the complete direct catalog.
  bool directBasis = false;
  std::vector<SyncCoverEventProtocol> protocols;
  std::vector<SyncCoverLifecycleSlot> slots;
  std::vector<SyncCoverDemandId> seedDemands;
  SyncCoverDemandSet exactCoverage;
  std::size_t singletonUnionCoverageRows = 0;
  std::size_t extraCoverageRows = 0;
};

/// Storage-derived lifecycle groups reconstructed exclusively from the frozen
/// physical graph and one versioned target contract.  No operation name,
/// legacy ownership certificate, or GEMM-specific recognizer participates in
/// discovery or admission.
struct SyncCoverLifecycleSynthesisResult {
  SyncCoverProtocolError error = SyncCoverProtocolError::None;
  std::vector<SyncCoverLifecycleProposal> proposals;
  std::size_t lifecycleSccs = 0;
  std::size_t derivedLifecycleCertificates = 0;
  std::size_t rejectedLifecycleConstructions = 0;
  std::size_t rejectedLifecycleVerifications = 0;
  std::size_t rejectedLifecycleStorageCertificates = 0;
  std::optional<SyncCoverProtocolError> firstLifecycleVerificationRejection;
  std::optional<std::size_t> firstLifecycleVerificationRejectionIndex;
  std::size_t lifecycleSccsWithoutProtocols = 0;
  std::optional<SyncCoverDemandId> firstUnproposedLifecycleDemand;
  std::size_t firstUnproposedLifecycleCertificates = 0;
  std::size_t firstUnproposedLifecycleConstructionRejects = 0;
  std::size_t firstUnproposedLifecycleVerificationRejects = 0;
  std::size_t firstUnproposedLifecycleStorageRejects = 0;
  std::size_t lifecycleSccsWithoutCoverage = 0;
  std::size_t inspectedDemands = 0;
  std::size_t inspectedAccesses = 0;
  std::size_t lifecycleAccessIncidences = 0;
  std::size_t lifecycleConnectorIncidences = 0;
  std::size_t candidateResourcePairs = 0;
  std::optional<std::size_t> invalidIndex;

  explicit operator bool() const {
    return error == SyncCoverProtocolError::None;
  }
};

SyncCoverLifecycleSynthesisResult synthesizeSyncCoverLifecycleCertificates(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

/// Constructs the conservative direct-event basis for demands repeated by a
/// loop. Each returned column is one exact forward cut plus an independently
/// verified reverse acknowledgement channel. The acknowledgement primes the
/// token at loop entry, prevents Set(i + 1) from overtaking Wait(i), and drains
/// the final token at loop exit. No storage-lifecycle recognizer participates.
SyncCoverLifecycleSynthesisResult synthesizeSyncCoverBalancedDirectProtocols(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr,
    bool enableCutFusion = true);

/// Alternative synthesis entry point for the target-neutral exact-storage
/// lifecycle index. It remains an analysis/test seam until its proposal set is
/// proven equivalent to the independently verified graph-owned recipe
/// synthesis used by the opt-in catalog.
SyncCoverLifecycleSynthesisResult synthesizeSyncCoverLifecycleCertificates(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

/// Rebuilds the graph-derived grouped and balanced-direct lifecycle catalogs
/// and verifies that every selected frozen proposal is reproduced exactly.
/// This is the materialization-boundary proof for storage-certified coverage:
/// it reruns target legality, token automata, exact graph grounding, and exact
/// physical-lane reuse certification instead of trusting admission bitsets.
SyncCoverProtocolError verifySyncCoverLifecycleProposalsFresh(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverStorageLifecycleIndex *lifecycleIndex,
    const SyncCoverStorageProtocolSeedIndex *seedIndex,
    const SyncCoverStorageProtocolGroupIndex *groupIndex,
    const std::vector<SyncCoverLifecycleProposal> &frozenProposals,
    const std::vector<std::size_t> &selectedProposalIndices,
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr,
    bool enableDirectCutFusion = true);

/// Alternative synthesis entry point for protocol groups proven by the
/// target-neutral storage-lifecycle analysis. Each group is imported as one
/// lifecycle SCC so disjoint exact-storage seeds with the same directed
/// ready/release behavior can be verified as one shared physical recipe.
SyncCoverLifecycleSynthesisResult synthesizeSyncCoverLifecycleCertificates(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolSeedIndex &seedIndex,
    const SyncCoverStorageProtocolGroupIndex &groupIndex,
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

struct SyncCoverProtocolEventReservation {
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  std::vector<unsigned> eventIds;
};

struct SyncCoverProtocolChannelAllocation {
  SyncCoverMechanismId mechanism = 0;
  SyncCoverProtocolChannelId channel = 0;
  std::vector<unsigned> eventIds;
};

struct SyncCoverProtocolAllocationResult {
  SyncCoverProtocolError error = SyncCoverProtocolError::None;
  std::vector<SyncCoverProtocolChannelAllocation> channels;
  std::optional<std::size_t> invalidIndex;

  explicit operator bool() const {
    return error == SyncCoverProtocolError::None;
  }
};

SyncCoverProtocolAllocationResult allocateSyncCoverProtocolEventIds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverProtocolEventReservation> &reservations = {},
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERPROTOCOL_H
