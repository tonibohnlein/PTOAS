// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CanonicalSyncSelection.h - Bounded pattern cover -------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCSELECTION_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCSELECTION_H

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

using CanonicalSyncMechanismId = std::size_t;
using CanonicalSyncPatternId = std::size_t;
using CanonicalSyncEventDomainId = std::size_t;

/// The three bounded greedy policies evaluated over one frozen problem.
enum class CanonicalSyncSelectionStrategy : std::uint8_t {
  FixedCover,
  ActionAwareSingleton,
  PairLookahead,
  /// Reporting identity for the non-optimizing select-all policy. This value
  /// is internal and is not accepted by the greedy-selector CLI option.
  MechanicalAll,
};

/// Ordering of the two principal structural-cost coordinates. Both modes use
/// the same frozen mechanisms and coverage; this is an objective ablation, not
/// a semantic change.
enum class CanonicalSyncSelectionObjective : std::uint8_t {
  ActionFirst,
  SerializationFirst,
};

enum class CanonicalSyncMechanismKind : std::uint8_t {
  Event,
  Barrier,
  Protocol,
};

/// Bounded diagnostic provenance for one physical mechanism recipe. A recipe
/// may be discovered by more than one generator; interning merges the
/// corresponding bits without making provenance part of mechanism identity.
enum class CanonicalSyncMechanismOrigin : std::uint8_t {
  Unclassified,
  DirectTargetedBarrier,
  DirectDistanceZeroEvent,
  DirectForwardRecurrenceEvent,
  DirectReleaseRecurrenceProtocol,
  CompletionFrontierEvent,
  TargetCompletionCertificateEvent,
  TargetLocalFenceEvent,
  SourceLocalCompletionEvent,
  SourceLocalPipeDrain,
  SourcePrefixPipeDrain,
  LoopCarryPipeDrain,
  LoopBoundarySourcePrefixProtocol,
  GenericLifecycleProtocol,
  BasicOwnershipL0OperandProtocol,
  BasicOwnershipStableL1Protocol,
  BasicOwnershipAlternatingL1Protocol,
  BasicOwnershipAccumulatorProtocol,
  BoundaryGuardedAccumulatorProtocol,
  HierarchicalStableL1Protocol,
  HierarchicalAlternatingL1Protocol,
  CompositeOwnershipProtocol,
  RepairTargetLocalPipeDrain,
  RepairSourceLocalPipeDrain,
  RepairSourcePrefixPipeDrain,
  RepairFrontierBarrier,
  RepairFrontierEvent,
  LocalizedPipeAll,
  /// Appended to preserve the numeric origin-mask encoding of schema v1.
  DirectBalancedTargetFenceEvent,
  Count,
};

using CanonicalSyncMechanismOriginMask = std::uint32_t;

constexpr std::size_t kCanonicalSyncMechanismOriginCount =
    static_cast<std::size_t>(CanonicalSyncMechanismOrigin::Count);
static_assert(kCanonicalSyncMechanismOriginCount <=
                  sizeof(CanonicalSyncMechanismOriginMask) * 8,
              "canonical sync mechanism origins must fit their mask");

constexpr CanonicalSyncMechanismOriginMask
canonicalSyncMechanismOriginBit(CanonicalSyncMechanismOrigin origin) {
  return CanonicalSyncMechanismOriginMask{1} << static_cast<unsigned>(origin);
}

enum class CanonicalSyncActionKind : std::uint8_t {
  EventSet,
  EventWait,
  Barrier,
};

enum class CanonicalSyncActionGuardKind : std::uint8_t {
  None,
  LoopNonEmpty,
  LoopEmpty,
  FirstIteration,
  NotFirstIteration,
  HasSuccessor,
};

enum class CanonicalSyncEventLaneKind : std::uint8_t {
  Static,
  /// Select eventIds[iterationOrdinal % width] in the named loop. This keeps
  /// one recurrence channel materializable without expanding one body action
  /// per static lane.
  LoopIterationModulo,
};

enum class CanonicalSyncBarrierKind : std::uint8_t {
  Targeted,
  All,
};

struct CanonicalSyncEventDomain {
  CanonicalSyncEventDomainId id = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  unsigned budget = 0;
  std::vector<unsigned> reservedIds;
};

struct CanonicalSyncEventUse {
  CanonicalSyncEventDomainId domain = 0;
  std::size_t width = 1;
  /// Principal recurrence scope used by ordinary event protocols.
  std::optional<SyncCoverScopeId> recurrenceScope;
  /// Optional wider resource lifetime for a verified hierarchical protocol.
  /// Supply edges retain their own recurrence scopes inside this loop.
  std::optional<SyncCoverScopeId> lifetimeScope;
};

struct CanonicalSyncAction {
  CanonicalSyncActionKind kind = CanonicalSyncActionKind::EventSet;
  std::uint32_t resource = 0;
  SyncCoverAnchor anchor;
  std::optional<std::size_t> eventUse;
  std::size_t eventLane = 0;
  /// Nonempty only for barriers. This is both the physical drain contract and
  /// the single authoritative barrier cost weight.
  std::vector<std::uint32_t> drainedResources;
  CanonicalSyncBarrierKind barrierKind = CanonicalSyncBarrierKind::Targeted;
  CanonicalSyncActionGuardKind guard = CanonicalSyncActionGuardKind::None;
  std::optional<SyncCoverScopeId> guardScope;
  CanonicalSyncEventLaneKind eventLaneKind = CanonicalSyncEventLaneKind::Static;
  std::optional<SyncCoverScopeId> eventLaneScope;
};

enum class CanonicalSyncSupplyProof : std::uint8_t {
  DirectAction,
  /// A direct set/wait whose producer is a certified later completion
  /// frontier. The graph's typed frontier edges, rather than a merged supply
  /// declaration, account for completion of earlier source nodes.
  CompletionFrontierAction,
  /// A direct event whose completion semantics are authorized by one
  /// immutable target/storage lifecycle certificate in SyncCoverGraph.  The
  /// binding remains restricted to its exact attested demand.
  TargetCompletionCertificateAction,
  /// A balanced, source-controlled event fence. The set is issued on the
  /// source pipe immediately after the complete physical source operation;
  /// the matching wait is issued on the target pipe at the same dynamic
  /// source occurrence. Target-pipe issue order carries that completed wait
  /// to later target occurrences without requiring source/target coexecution.
  /// Positive-distance bindings remain restricted to their attested demand.
  SourceLocalCompletionAction,
  /// A width-one, locally rearmed set/wait pair issued at one target anchor.
  /// Every bound demand is independently certified in its d+1 arena and is
  /// restricted through allowedDemands. Multiple bindings are action
  /// deduplication only when their complete physical recipes are identical.
  TargetLocalFenceAction,
  /// A targeted source-pipe drain issued immediately before the physical
  /// target. The ISA guarantees that every previously issued operation on
  /// that source pipe completes before the subsequent target begins. Each
  /// binding remains independently attested and distance-qualified.
  TargetLocalPipeDrainAction,
  /// A targeted source-pipe drain before one physical target. The immutable
  /// certificate names resource-matching, guard-compatible source-prefix
  /// nodes issued before that cut. Each edge is owned by the source/target LCA.
  /// Its supplies are distance-zero-only and may therefore persist through
  /// later fixed issue order without leaking into recurrence.
  DominatingTargetedDrainCut,
  /// A source-pipe drain at the beginning of every non-first loop iteration.
  /// Each positive-distance supply is separately attested and restricted to
  /// its original demand; the shared action only certifies that the complete
  /// prior-iteration source prefix has finished before the current body.
  LoopCarryPipeDrain,
  /// A named-pipe drain after the complete physical source operation. Every
  /// binding is exact-demand attested; sharing is physical-action
  /// deduplication across later targets, never unrestricted prefix export.
  SourceLocalPipeDrainAction,
  /// A named-pipe drain after a later same-scope, same-guard cut. The immutable
  /// issued-prefix certificate proves that earlier source operations on the
  /// named pipe have issued before the cut; every supplied demand remains
  /// independently attested and distance-qualified.
  SourcePrefixPipeDrainAction,
  /// One lifecycle-complete recurrence channel shared by exact demands with
  /// the same loop, distance, and directed pipe domain. Entry priming and
  /// scope-exit draining balance the channel. A targeted source-prefix drain
  /// followed by the body set at LoopBodyExit completes every admitted source
  /// before the next-copy wait at LoopBodyEntry.
  LoopBoundarySourcePrefixProtocol,
  /// One action-backed edge of a graph-certified basic ownership lifecycle.
  /// The complete descriptor is independently reconstructed and verified
  /// against its immutable exact-slot certificate before interning.
  VerifiedBasicOwnershipProtocol,
  /// A completion consequence of the complete ready/release ownership token
  /// path. It owns no single action; it is admitted only by the same exact
  /// descriptor verifier and remains restricted to named demand rows.
  VerifiedBasicOwnershipComposite,
  /// Action-backed transfer admitted only after the target-typed lifecycle
  /// protocol and its exact world have been independently verified.
  VerifiedGenericLifecycleProtocol,
  /// An actionless transitive consequence of a complete verified lifecycle
  /// proposal.  The binding is always restricted to one exact demand row.
  VerifiedGenericLifecycleComposite,
  VerifiedProtocol,
};

enum class CanonicalSyncSupplyExport : std::uint8_t {
  LocalTarget,
  ScopeExitAfterDrain,
};

struct CanonicalSyncSupplyBinding {
  SyncCoverEdge edge;
  std::optional<std::size_t> eventUse;
  std::optional<std::size_t> barrierAction;
  std::optional<std::size_t> produceAction;
  std::optional<std::size_t> consumeAction;
  CanonicalSyncSupplyProof proof = CanonicalSyncSupplyProof::DirectAction;
  /// Scope-exit export is a separate certificate from protocol admission. It
  /// is accepted only when common validation proves balanced priming, body
  /// lane use, and one scope-exit drain for every recurrence lane.
  CanonicalSyncSupplyExport completionExport =
      CanonicalSyncSupplyExport::LocalTarget;
  /// Empty means this completion consequence is generally applicable. A
  /// verified ownership release names only the exact storage demands for
  /// which issue, rather than full operation completion, is sufficient.
  std::vector<SyncCoverDemandId> allowedDemands;
  /// Original row whose exact dynamic recipe was checked. This is validation
  /// provenance and does not itself restrict graph propagation.
  std::optional<SyncCoverDemandId> attestedDemand;
  SyncCoverSupplyApplicability applicability =
      SyncCoverSupplyApplicability::AllDemands;
};

/// One atomic synchronization unit. This descriptor is also its emission
/// recipe; no provider or selected-mechanism representation is reconstructed
/// after selection.
struct CanonicalSyncMechanismDescriptor {
  CanonicalSyncMechanismKind kind = CanonicalSyncMechanismKind::Event;
  /// LCA of every concrete action region. This is derived by the problem and
  /// is never accepted as caller-authored proof.
  SyncCoverRegionId ownerRegion = 0;
  std::vector<CanonicalSyncSupplyBinding> supplies;
  std::vector<CanonicalSyncEventUse> eventUses;
  std::vector<CanonicalSyncAction> actions;
};

struct CanonicalSyncEventLifetime {
  SyncCoverTimelinePosition begin = 0;
  SyncCoverTimelinePosition end = 0;
};

struct CanonicalSyncMechanismCost {
  /// Natural loop depth: index zero is one-shot function scope.
  std::vector<std::uint64_t> barrierActions;
  std::vector<std::uint64_t> eventActions;
  std::uint64_t serializationBreadth = 0;
};

struct CanonicalSyncMechanism {
  CanonicalSyncMechanismId id = 0;
  CanonicalSyncMechanismDescriptor descriptor;
  std::vector<CanonicalSyncEventLifetime> eventLifetimes;
  CanonicalSyncMechanismCost cost;
  std::vector<CanonicalSyncMechanismId> conflicts;
  CanonicalSyncMechanismOriginMask originMask = canonicalSyncMechanismOriginBit(
      CanonicalSyncMechanismOrigin::Unclassified);
};

enum class CanonicalSyncPatternKind : std::uint8_t {
  Singleton,
  DirectPair,
  RepairFrontier,
};

constexpr std::size_t kCanonicalSyncPatternKindCount =
    static_cast<std::size_t>(CanonicalSyncPatternKind::RepairFrontier) + 1;

struct CanonicalSyncPatternSpec {
  CanonicalSyncPatternKind kind = CanonicalSyncPatternKind::Singleton;
  std::vector<CanonicalSyncMechanismId> members;
};

/// One independently valid barrier/event recipe proposed by conflict-core
/// repair. A batch is admitted transactionally so optional truncation cannot
/// leak a partial frontier into the owner-repair catalog.
struct CanonicalSyncRepairFrontierBatchEntry {
  CanonicalSyncMechanismDescriptor barrier;
  CanonicalSyncMechanismDescriptor event;
};

struct CanonicalSyncPattern {
  CanonicalSyncPatternId id = 0;
  CanonicalSyncPatternKind kind = CanonicalSyncPatternKind::Singleton;
  std::vector<CanonicalSyncMechanismId> members;
  /// Singleton patterns store their complete non-baseline coverage. Composite
  /// patterns store only coverage unavailable from their member singletons.
  SyncCoverDemandSet coverage;
  /// Exact counts retained so a frozen precise catalog can be cloned into a
  /// mutable repair extension without rerunning its semantic preparation.
  std::size_t jointCoverageCount = 0;
  std::size_t singletonCoverageCount = 0;
  /// Number of covered demands not available from the union of the members'
  /// singleton coverage. Zero means the pattern is greedy packaging, not a
  /// semantic synchronization composition. Detailed extra bits are transient
  /// during construction so diagnostics do not double pattern storage.
  std::size_t extraCoverageCount = 0;
};

/// Minimal immutable seed needed to generate conflict-core frontier recipes
/// from an already frozen precise catalog. The seed records only independently
/// verified event mechanisms; it is not itself a synchronization candidate.
struct CanonicalSyncRepairEventSeed {
  SyncCoverDemandId demand = 0;
  CanonicalSyncMechanismId mechanism = 0;
  CanonicalSyncEventDomainId domain = 0;
};

struct CanonicalSyncPatternKindStatistics {
  std::size_t patterns = 0;
  std::size_t jointCoverageIncidences = 0;
  std::size_t singletonCoverageIncidences = 0;
  std::size_t extraCoverageIncidences = 0;
  std::size_t patternsWithExtraCoverage = 0;
};

struct CanonicalSyncPatternStatistics {
  std::array<CanonicalSyncPatternKindStatistics, kCanonicalSyncPatternKindCount>
      kinds;
  std::size_t directPairProposals = 0;
  std::size_t directPairEvaluations = 0;
  std::size_t directPairConnectorInspections = 0;
  std::size_t repairFrontierInspections = 0;
  std::size_t repairFrontierProposals = 0;
  bool repairFrontierTruncated = false;
  std::size_t sourcePrefixInspections = 0;
  std::size_t sourcePrefixCandidates = 0;
  std::size_t sourcePrefixIncidences = 0;
  bool sourcePrefixGenerationTruncated = false;
  std::size_t loopCarryInspections = 0;
  std::size_t loopCarryCandidates = 0;
  std::size_t loopCarryIncidences = 0;
  bool loopCarryGenerationTruncated = false;
  std::size_t loopBoundaryProtocolInspections = 0;
  std::size_t loopBoundaryProtocolCandidates = 0;
  std::size_t loopBoundaryProtocolIncidences = 0;
  bool loopBoundaryProtocolGenerationTruncated = false;
  std::size_t genericLifecycleSynthesisWorkUnits = 0;
  bool genericLifecycleGenerationTruncated = false;

  const CanonicalSyncPatternKindStatistics &
  get(CanonicalSyncPatternKind kind) const {
    return kinds[static_cast<std::size_t>(kind)];
  }
};

enum class CanonicalSyncProblemError : std::uint8_t {
  None,
  Frozen,
  InvalidGraph,
  InvalidDomain,
  InvalidMechanism,
  UnverifiedProtocol,
  InvalidPattern,
  UncoverableDemand,
  LimitExceeded,
  CoverageFailure,
  ArithmeticOverflow,
};

struct CanonicalSyncProblemResult {
  CanonicalSyncProblemError error = CanonicalSyncProblemError::None;
  std::optional<std::size_t> index;

  explicit operator bool() const {
    return error == CanonicalSyncProblemError::None;
  }
};

/// Persistent verifier for one admitted protocol mechanism. The callback must
/// charge the supplied budget for all protocol-specific work before performing
/// it and may return only None, UnverifiedProtocol, or LimitExceeded.
using CanonicalSyncProtocolVerifier = std::function<CanonicalSyncProblemError(
    const CanonicalSyncMechanismDescriptor &, SyncCoverCoverageWorkBudget &)>;

/// Immutable view passed to a whole-plan protocol verifier. The descriptor is
/// owned by the frozen problem and remains valid for the duration of the call.
struct CanonicalSyncMaterializedMechanismView {
  CanonicalSyncMechanismId mechanism = 0;
  CanonicalSyncMechanismOriginMask originMask = 0;
  const CanonicalSyncMechanismDescriptor *descriptor = nullptr;
};

/// Deep verifier for a family of selected physical protocols. Unlike an
/// admission verifier, this callback runs only at the final materialization
/// boundary and may rebuild graph-wide certificates and token automata once
/// for the complete selected family.
using CanonicalSyncMaterializedPlanVerifier =
    std::function<CanonicalSyncProblemError(
        const std::vector<CanonicalSyncMaterializedMechanismView> &,
        SyncCoverCoverageWorkBudget &)>;

class CanonicalSyncPatternProblem {
public:
  struct Limits {
    std::size_t maximumDomains = 64;
    std::size_t maximumEventBudget = 64;
    std::size_t maximumReservedEventIds = 64;
    std::size_t maximumMechanisms = 1U << 16;
    std::size_t maximumPatterns = 1U << 16;
    std::size_t maximumPatternProposals = 1U << 17;
    std::size_t maximumActionsPerMechanism = 1U << 12;
    std::size_t maximumDrainedResourcesPerBarrier = 64;
    std::size_t maximumEventUsesPerMechanism = 1U << 10;
    std::size_t maximumSuppliesPerMechanism = 1U << 12;
    std::size_t maximumTotalActions = 1U << 20;
    std::size_t maximumTotalEventUses = 1U << 18;
    std::size_t maximumTotalSupplies = 1U << 20;
    std::size_t maximumMembersPerPattern = 64;
    std::size_t maximumIncidences = 1U << 22;
    /// Aggregate dense words retained by singleton coverage preparation,
    /// including the fixed baseline and temporary propagation workspace.
    std::size_t maximumSingletonCoverageWords = 1U << 22;
    /// Aggregate dense coverage words retained by optional patterns.
    std::size_t maximumCoverageWords = 1U << 22;
  };

  CanonicalSyncPatternProblem(const SyncCoverGraph &graph,
                              std::vector<SyncCoverDemandId> activeDemands);
  CanonicalSyncPatternProblem(const SyncCoverGraph &graph,
                              std::vector<SyncCoverDemandId> activeDemands,
                              Limits limits,
                              SyncCoverExpansionLimits expansionLimits = {});
  CanonicalSyncPatternProblem(const SyncCoverGraph &graph,
                              std::vector<SyncCoverDemandId> obligationDemands,
                              std::vector<SyncCoverDemandId> selectionDemands,
                              Limits limits,
                              SyncCoverExpansionLimits expansionLimits = {},
                              bool basisReductionTruncated = false);
  CanonicalSyncPatternProblem(const CanonicalSyncPatternProblem &) = delete;
  CanonicalSyncPatternProblem(CanonicalSyncPatternProblem &&) = default;
  CanonicalSyncPatternProblem &
  operator=(const CanonicalSyncPatternProblem &) = delete;
  CanonicalSyncPatternProblem &
  operator=(CanonicalSyncPatternProblem &&) = delete;

  CanonicalSyncProblemResult addEventDomain(CanonicalSyncEventDomain domain);
  CanonicalSyncProblemResult
  internMechanism(CanonicalSyncMechanismDescriptor descriptor,
                  CanonicalSyncMechanismOrigin origin =
                      CanonicalSyncMechanismOrigin::Unclassified);
  CanonicalSyncProblemResult
  internVerifiedProtocol(CanonicalSyncMechanismDescriptor descriptor,
                         CanonicalSyncProtocolVerifier verifier,
                         CanonicalSyncMechanismOrigin origin =
                             CanonicalSyncMechanismOrigin::Unclassified);
  CanonicalSyncProblemResult
  addMaterializedPlanVerifier(CanonicalSyncMechanismOriginMask origins,
                              CanonicalSyncMaterializedPlanVerifier verifier);
  CanonicalSyncProblemResult addConflict(CanonicalSyncMechanismId first,
                                         CanonicalSyncMechanismId second);
  CanonicalSyncProblemResult addPattern(CanonicalSyncPatternSpec pattern);
  /// Request physical-cut singleton grounding. This mode is used only by the
  /// strict direct catalog; recurrence protocols continue through the
  /// lifecycle-aware completion-supply engine until they can be represented
  /// as complete physical cut groups.
  CanonicalSyncProblemResult enableExactDirectCutGrounding(
      SyncCoverOrderingRequirementMask eventRequirements,
      SyncCoverOrderingRequirementMask barrierRequirements,
      bool eventCompletesSourcePrefix);
  /// Classify and atomically commit one owner scope's exact direct-pair rows.
  /// Joint, singleton, and baseline rows use graph-global demand IDs. The
  /// result index is the number of retained extra-coverage patterns committed
  /// by the batch.
  CanonicalSyncProblemResult addDirectPairBatch(
      const std::vector<SyncCoverMechanismPair> &pairs,
      const std::vector<SyncCoverDemandSet> &jointCoverage,
      const std::vector<SyncCoverDemandSet> &singletonMechanismCoverage,
      const SyncCoverDemandSet &baselineCoverage);
  /// Atomically admit a bounded batch of repair-frontier recipes. Limit or
  /// semantic failure restores the exact mutable catalog prefix.
  CanonicalSyncProblemResult addRepairFrontierBatch(
      std::vector<CanonicalSyncRepairFrontierBatchEntry> entries);
  CanonicalSyncProblemResult
  freeze(SyncCoverCoverageWorkBudget *workBudget = nullptr);

  /// Reconstruct and evaluate exact physical cuts for the requested worlds.
  /// Every referenced mechanism must be a token-independent event or targeted
  /// pipe barrier. Results use graph-global demand IDs.
  SyncCoverRegionWorldResult computeExactDirectCutWorlds(
      const std::vector<SyncCoverExactWorld> &worlds,
      SyncCoverRegionWorldLimits limits = {},
      SyncCoverCoverageWorkBudget *workBudget = nullptr) const;

  /// Copy one immutable precise catalog into a mutable repair extension. The
  /// grounded prefix remains byte-for-byte identical; only newly appended
  /// repair mechanisms and patterns are semantically evaluated. Copy work and
  /// all subsequent construction work use the supplied shared budget.
  std::unique_ptr<CanonicalSyncPatternProblem>
  cloneMutableRepairPrefix(SyncCoverCoverageWorkBudget *workBudget) const;

  bool isFrozen() const { return frozen_; }
  /// Direct-pair preparation owns one aggregate dense-memory reservation and
  /// therefore starts only before any per-pattern construction cache exists.
  bool canPrepareDirectPairPatterns() const;
  const SyncCoverGraph &getGraph() const { return graph_; }
  const SyncCoverExpandedProgram &getExpansion() const { return *expansion_; }
  const std::vector<SyncCoverDemandId> &getDemands() const {
    return activeDemands_;
  }
  const std::vector<SyncCoverDemandId> &getObligationDemands() const {
    return obligationDemands_;
  }
  bool wasBasisReductionTruncated() const { return basisReductionTruncated_; }
  const std::vector<CanonicalSyncEventDomain> &getDomains() const {
    return domains_;
  }
  const std::vector<CanonicalSyncMechanism> &getMechanisms() const {
    return mechanisms_;
  }
  const std::vector<CanonicalSyncPattern> &getPatterns() const {
    return patterns_;
  }
  const std::vector<std::vector<CanonicalSyncPatternId>> &
  getDemandPatterns() const {
    return demandPatterns_;
  }
  const std::vector<std::vector<CanonicalSyncPatternId>> &
  getMechanismPatterns() const {
    return mechanismPatterns_;
  }
  const CanonicalSyncPatternStatistics &getPatternStatistics() const {
    return patternStatistics_;
  }
  bool wasPatternGenerationTruncated() const {
    return patternGenerationTruncated_;
  }
  CanonicalSyncProblemResult markPatternGenerationTruncated() {
    if (frozen_) {
      return {CanonicalSyncProblemError::Frozen, std::nullopt};
    }
    patternGenerationTruncated_ = true;
    return {};
  }
  CanonicalSyncProblemResult
  recordDirectPairGeneration(std::size_t proposals, std::size_t evaluations,
                             std::size_t connectorInspections) {
    if (frozen_) {
      return {CanonicalSyncProblemError::Frozen, std::nullopt};
    }
    patternStatistics_.directPairProposals = proposals;
    patternStatistics_.directPairEvaluations = evaluations;
    patternStatistics_.directPairConnectorInspections = connectorInspections;
    return {};
  }
  CanonicalSyncProblemResult
  recordRepairFrontierGeneration(std::size_t inspections, std::size_t proposals,
                                 bool truncated) {
    if (frozen_) {
      return {CanonicalSyncProblemError::Frozen, std::nullopt};
    }
    patternStatistics_.repairFrontierInspections = inspections;
    patternStatistics_.repairFrontierProposals = proposals;
    patternStatistics_.repairFrontierTruncated = truncated;
    return {};
  }
  CanonicalSyncProblemResult
  recordSourcePrefixGeneration(std::size_t inspections, std::size_t candidates,
                               std::size_t incidences, bool truncated) {
    if (frozen_) {
      return {CanonicalSyncProblemError::Frozen, std::nullopt};
    }
    const bool overflow =
        inspections > std::numeric_limits<std::size_t>::max() -
                          patternStatistics_.sourcePrefixInspections ||
        candidates > std::numeric_limits<std::size_t>::max() -
                         patternStatistics_.sourcePrefixCandidates ||
        incidences > std::numeric_limits<std::size_t>::max() -
                         patternStatistics_.sourcePrefixIncidences;
    if (overflow) {
      return {CanonicalSyncProblemError::ArithmeticOverflow, std::nullopt};
    }
    patternStatistics_.sourcePrefixInspections += inspections;
    patternStatistics_.sourcePrefixCandidates += candidates;
    patternStatistics_.sourcePrefixIncidences += incidences;
    patternStatistics_.sourcePrefixGenerationTruncated |= truncated;
    return {};
  }
  CanonicalSyncProblemResult recordLoopCarryGeneration(std::size_t inspections,
                                                       std::size_t candidates,
                                                       std::size_t incidences,
                                                       bool truncated) {
    if (frozen_) {
      return {CanonicalSyncProblemError::Frozen, std::nullopt};
    }
    patternStatistics_.loopCarryInspections = inspections;
    patternStatistics_.loopCarryCandidates = candidates;
    patternStatistics_.loopCarryIncidences = incidences;
    patternStatistics_.loopCarryGenerationTruncated = truncated;
    return {};
  }
  CanonicalSyncProblemResult
  recordLoopBoundaryProtocolGeneration(std::size_t inspections,
                                       std::size_t candidates,
                                       std::size_t incidences, bool truncated) {
    if (frozen_) {
      return {CanonicalSyncProblemError::Frozen, std::nullopt};
    }
    const bool overflow =
        inspections > std::numeric_limits<std::size_t>::max() -
                          patternStatistics_.loopBoundaryProtocolInspections ||
        candidates > std::numeric_limits<std::size_t>::max() -
                         patternStatistics_.loopBoundaryProtocolCandidates ||
        incidences > std::numeric_limits<std::size_t>::max() -
                         patternStatistics_.loopBoundaryProtocolIncidences;
    if (overflow) {
      return {CanonicalSyncProblemError::ArithmeticOverflow, std::nullopt};
    }
    patternStatistics_.loopBoundaryProtocolInspections += inspections;
    patternStatistics_.loopBoundaryProtocolCandidates += candidates;
    patternStatistics_.loopBoundaryProtocolIncidences += incidences;
    patternStatistics_.loopBoundaryProtocolGenerationTruncated |= truncated;
    return {};
  }
  CanonicalSyncProblemResult markGenericLifecycleGenerationTruncated() {
    if (frozen_) {
      return {CanonicalSyncProblemError::Frozen, std::nullopt};
    }
    patternStatistics_.genericLifecycleGenerationTruncated = true;
    return {};
  }
  CanonicalSyncProblemResult
  recordGenericLifecycleSynthesisWorkUnits(std::size_t workUnits) {
    if (frozen_) {
      return {CanonicalSyncProblemError::Frozen, std::nullopt};
    }
    const bool overflow =
        workUnits > std::numeric_limits<std::size_t>::max() -
                        patternStatistics_.genericLifecycleSynthesisWorkUnits;
    if (overflow) {
      return {CanonicalSyncProblemError::ArithmeticOverflow, std::nullopt};
    }
    patternStatistics_.genericLifecycleSynthesisWorkUnits += workUnits;
    return {};
  }
  bool hasSameCandidatePrefix(const CanonicalSyncPatternProblem &other) const;
  /// Revalidate one immutable mechanism and its derived lifetime/cost data.
  /// Protocols rerun the common balanced-action and supply-export contract.
  CanonicalSyncProblemResult
  verifyMechanism(CanonicalSyncMechanismId mechanism,
                  SyncCoverCoverageWorkBudget *workBudget = nullptr) const;
  /// Revalidate a candidate recipe against one admitted mechanism's retained
  /// certificate. This is non-mutating and is used by repair diagnostics and
  /// adversarial certificate tests.
  CanonicalSyncProblemResult verifyMechanismDescriptor(
      CanonicalSyncMechanismId mechanism,
      CanonicalSyncMechanismDescriptor descriptor,
      SyncCoverCoverageWorkBudget *workBudget = nullptr) const;
  /// Run graph-wide protocol-family verification once for the complete
  /// selected mechanism set. Candidate coverage and admission-time callbacks
  /// are not accepted as proof at this boundary.
  CanonicalSyncProblemResult verifyMaterializedPlanMechanisms(
      const std::vector<CanonicalSyncMechanismId> &mechanisms,
      SyncCoverCoverageWorkBudget *workBudget = nullptr,
      std::size_t *verifiersRun = nullptr,
      CanonicalSyncMechanismOriginMask *verifiedOrigins = nullptr) const;
  std::uint64_t getMechanismSignature(CanonicalSyncMechanismId mechanism) const;
  /// Preview the union of baseline, singleton, and retained optional-pattern
  /// coverage without freezing the catalog. Used to ground completeness
  /// mechanisms only for rows absent from the exact catalog.
  CanonicalSyncProblemResult
  previewCoveredDemands(SyncCoverDemandSet &covered) const;
  const SyncCoverDemandSet &getBaselineCoverage() const {
    return baselineCoverage_;
  }
  const Limits &getLimits() const { return limits_; }
  CanonicalSyncProblemResult
  recordRepairEventSeed(CanonicalSyncRepairEventSeed seed);
  const std::vector<CanonicalSyncRepairEventSeed> &getRepairEventSeeds() const {
    return repairEventSeeds_;
  }
  CanonicalSyncProblemResult
  setCandidateConfigurationSignature(std::uint64_t signature);
  std::uint64_t getCandidateConfigurationSignature() const {
    return candidateConfigurationSignature_;
  }
  bool usesExactDirectCutGrounding() const { return exactDirectCutGrounding_; }

private:
  struct MutableBatchJournal;
  struct PendingPattern {
    CanonicalSyncPatternSpec spec;
    /// Sparse active-demand IDs. Dense rows are materialized only during
    /// freeze, under coverageWordCount_'s aggregate reservation.
    std::vector<SyncCoverDemandId> coverage;
    std::size_t jointCoverageCount = 0;
    std::size_t singletonCoverageCount = 0;
    std::size_t extraCoverageCount = 0;
  };

  CanonicalSyncProblemResult
  internMechanismImpl(CanonicalSyncMechanismDescriptor descriptor,
                      bool protocolVerified,
                      CanonicalSyncProtocolVerifier verifier = {},
                      CanonicalSyncMechanismOrigin origin =
                          CanonicalSyncMechanismOrigin::Unclassified,
                      MutableBatchJournal *journal = nullptr);
  CanonicalSyncProblemResult addPatternImpl(CanonicalSyncPatternSpec pattern,
                                            MutableBatchJournal *journal);
  void rollbackMutableBatch(MutableBatchJournal &journal);
  CanonicalSyncProblemResult
  checkDenseConstructionEnvelope(std::size_t additionalMechanisms,
                                 std::size_t additionalPatternCacheCandidates,
                                 std::size_t additionalRetainedPatterns,
                                 bool includeGroundingPeak);
  CanonicalSyncProblemResult validateAndCostMechanism(
      CanonicalSyncMechanismDescriptor &descriptor,
      std::vector<CanonicalSyncEventLifetime> &lifetimes,
      CanonicalSyncMechanismCost &cost, bool protocolVerified,
      SyncCoverCoverageWorkBudget *workBudget = nullptr) const;
  CanonicalSyncProblemResult
  buildPatterns(std::vector<CanonicalSyncPattern> &patterns,
                CanonicalSyncPatternStatistics &statistics,
                SyncCoverDemandSet &baselineCoverage) const;
  CanonicalSyncProblemResult
  buildExactDirectCutPatterns(std::vector<CanonicalSyncPattern> &patterns,
                              CanonicalSyncPatternStatistics &statistics,
                              SyncCoverDemandSet &baselineCoverage) const;
  CanonicalSyncProblemResult
  buildIncrementalPatterns(std::vector<CanonicalSyncPattern> &patterns,
                           CanonicalSyncPatternStatistics &statistics,
                           SyncCoverDemandSet &baselineCoverage);
  CanonicalSyncPatternProblem(
      const CanonicalSyncPatternProblem &preciseProblem,
      SyncCoverCoverageWorkBudget *constructionWorkBudget);

  const SyncCoverGraph &graph_;
  std::shared_ptr<const SyncCoverExpandedProgram> expansion_;
  Limits limits_;
  std::vector<std::uint32_t> issueResources_;
  bool graphValid_ = false;
  bool frozen_ = false;
  std::size_t incidenceCount_ = 0;
  std::size_t actionCount_ = 0;
  std::size_t eventUseCount_ = 0;
  std::size_t supplyCount_ = 0;
  std::vector<SyncCoverDemandId> obligationDemands_;
  std::vector<SyncCoverDemandId> activeDemands_;
  bool basisReductionTruncated_ = false;
  std::vector<CanonicalSyncEventDomain> domains_;
  std::vector<CanonicalSyncMechanism> mechanisms_;
  /// Admission verifiers for protocol mechanisms, aligned with mechanisms_.
  /// They are retained so fresh verification re-runs the protocol-specific
  /// factory/certificate contract instead of trusting the admission result.
  /// Shared immutable callbacks keep repair-prefix cloning proportional to the
  /// number of mechanisms rather than opaque verifier-capture sizes.
  std::vector<std::shared_ptr<const CanonicalSyncProtocolVerifier>>
      protocolVerifiers_;
  struct MaterializedPlanVerifierEntry {
    CanonicalSyncMechanismOriginMask origins = 0;
    std::shared_ptr<const CanonicalSyncMaterializedPlanVerifier> verifier;
  };
  std::vector<MaterializedPlanVerifierEntry> materializedPlanVerifiers_;
  std::vector<PendingPattern> patternSpecs_;
  std::optional<SyncCoverDemandSet> constructionBaselineCoverage_;
  std::vector<std::optional<SyncCoverDemandSet>> constructionSingletonCoverage_;
  std::size_t retainedPatternCount_ = 0;
  std::size_t coverageWordCount_ = 0;
  std::size_t pendingCoverageIncidenceCount_ = 0;
  bool patternGenerationTruncated_ = false;
  std::vector<CanonicalSyncPattern> patterns_;
  CanonicalSyncPatternStatistics patternStatistics_;
  SyncCoverDemandSet baselineCoverage_;
  std::vector<std::vector<CanonicalSyncPatternId>> demandPatterns_;
  std::vector<std::vector<CanonicalSyncPatternId>> mechanismPatterns_;
  std::map<std::uint64_t, std::vector<CanonicalSyncMechanismId>>
      mechanismBuckets_;
  std::vector<CanonicalSyncRepairEventSeed> repairEventSeeds_;
  std::uint64_t candidateConfigurationSignature_ = 0;
  std::size_t frozenPrefixMechanismCount_ = 0;
  SyncCoverCoverageWorkBudget *constructionWorkBudget_ = nullptr;
  bool exactDirectCutGrounding_ = false;
  SyncCoverOrderingRequirementMask exactDirectEventRequirements_ = 0;
  SyncCoverOrderingRequirementMask exactDirectBarrierRequirements_ = 0;
  bool exactDirectEventCompletesSourcePrefix_ = false;
};

struct CanonicalSyncDirectPairOptions {
  /// Proposals are owned by the LCA of their mechanism scopes. An oversized
  /// owner batch is skipped atomically; no batch is cut by mechanism ID order.
  std::size_t maximumEvaluationsPerScope = 1U << 12;
  /// Total source/target endpoint entries retained by the connector index.
  std::size_t maximumConnectorIndexEntries = 1U << 20;
  /// Global deterministic bound on owner-group and endpoint comparisons.
  std::size_t maximumConnectorInspections = 1U << 24;
  /// Pair preparation is optional. A scope whose exact pair matrices exceed
  /// these limits is skipped without weakening singleton correctness.
  SyncCoverCoverageLimits pairCoverageLimits;
  /// Aggregate dense words simultaneously retained by optional pair
  /// preparation: singleton rows, one owner pair query, and batch scratch.
  std::size_t maximumPreparationWords = 1U << 22;
};

CanonicalSyncProblemResult
addCanonicalSyncDirectPairPatterns(CanonicalSyncPatternProblem &problem,
                                   CanonicalSyncDirectPairOptions options = {});

/// Add an optional repair pattern only when its complete member set is within
/// construction limits and conflict-free. Event feasibility is deliberately
/// deferred until after cover selection and reverse deletion.
CanonicalSyncProblemResult
addCanonicalSyncFeasiblePattern(CanonicalSyncPatternProblem &problem,
                                CanonicalSyncPatternSpec pattern);

struct CanonicalSyncEventAllocation {
  CanonicalSyncMechanismId mechanism = 0;
  std::size_t eventUse = 0;
  std::vector<unsigned> ids;
};

struct CanonicalSyncDomainAllocation {
  CanonicalSyncEventDomainId domain = 0;
  std::size_t required = 0;
  std::size_t available = 0;
  std::optional<SyncCoverTimelinePosition> maximumPressurePoint;
  std::vector<CanonicalSyncMechanismId> liveMechanisms;
  std::vector<CanonicalSyncEventAllocation> uses;
};

struct CanonicalSyncResourceAllocation {
  bool valid = false;
  bool feasible = false;
  std::vector<CanonicalSyncDomainAllocation> domains;
};

CanonicalSyncResourceAllocation allocateCanonicalSyncEvents(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected,
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

struct CanonicalSyncGreedyStatistics {
  std::size_t patternEvaluations = 0;
  std::size_t deletionEvaluations = 0;
  /// Aggregate bound over incidence visits, bitset words, selected mechanisms,
  /// and resource evaluations performed by greedy selection and deletion.
  std::size_t workUnits = 0;
  /// Structural-cost construction failed instead of collapsing distinct
  /// candidates onto a saturated sentinel value.
  bool arithmeticOverflow = false;
};

/// Calibration-free static cost. Action profiles are indexed by natural loop
/// depth, with deeper entries compared before shallower entries.
struct CanonicalSyncStructuralCost {
  /// Natural loop-depth profiles reported separately for diagnostics. The
  /// production selector continues to compare their aggregate action count.
  std::vector<std::uint64_t> barrierActionProfile;
  std::vector<std::uint64_t> eventActionProfile;
  /// Aggregate physical action count retained for reporting compatibility.
  std::vector<std::uint64_t> actionProfile;
  std::uint64_t serializationBreadth = 0;
  std::uint64_t eventLifetimeArea = 0;
  std::size_t mechanismCount = 0;
};

enum class CanonicalSyncSelectionError : std::uint8_t {
  None,
  InvalidProblem,
  NoCoveringPattern,
  InvalidAllocation,
  ResourceInfeasible,
  WorkLimitExceeded,
  ArithmeticOverflow,
  FinalValidationFailed,
};

struct CanonicalSyncGreedyOptions {
  std::size_t maximumWorkUnits = 1U << 27;
  CanonicalSyncSelectionStrategy strategy =
      CanonicalSyncSelectionStrategy::PairLookahead;
  CanonicalSyncSelectionObjective objective =
      CanonicalSyncSelectionObjective::ActionFirst;
  /// Mechanisms disabled by one bounded resource-repair trial.
  std::vector<CanonicalSyncMechanismId> forbiddenMechanisms;
};

struct CanonicalSyncSelection {
  CanonicalSyncSelectionError error = CanonicalSyncSelectionError::None;
  std::vector<CanonicalSyncMechanismId> mechanisms;
  SyncCoverDemandSet covered;
  CanonicalSyncResourceAllocation allocation;
  CanonicalSyncStructuralCost cost;
  std::vector<CanonicalSyncMechanismId> selectionOrder;
  CanonicalSyncGreedyStatistics statistics;

  explicit operator bool() const {
    return error == CanonicalSyncSelectionError::None;
  }
};

CanonicalSyncSelection
selectCanonicalSyncPatterns(const CanonicalSyncPatternProblem &problem,
                            CanonicalSyncGreedyOptions options = {});

/// Select every mechanism in a frozen, conflict-free singleton problem. This
/// generic select-all primitive performs no set-cover reduction, pair
/// composition, or reverse deletion. Callers remain responsible for
/// validating the catalog policy at their boundary.
CanonicalSyncSelection selectAllCanonicalSyncSingletonMechanisms(
    const CanonicalSyncPatternProblem &problem,
    std::size_t maximumWorkUnits = 1U << 27);

std::optional<CanonicalSyncStructuralCost> computeCanonicalSyncStructuralCost(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected);

/// Compare two complete structural costs with the production objective. Equal
/// costs compare false in both directions; callers supply their own
/// deterministic identity tie-breaker when one is required.
bool canonicalSyncStructuralCostLess(const CanonicalSyncStructuralCost &first,
                                     const CanonicalSyncStructuralCost &second,
                                     CanonicalSyncSelectionObjective objective);

/// Streaming, objective-aware ranking for one repair round. Verification and
/// resource diagnosis stay with the repair driver; this object only retains
/// the stable identity of the cheapest verified trial and the best strictly
/// pressure-improving trial.
class CanonicalSyncRepairRoundRanker {
public:
  enum class Decision : std::uint8_t {
    Discard,
    ReplaceBestVerified,
    ReplaceBestPressure,
    WorkLimitExceeded,
  };

  CanonicalSyncRepairRoundRanker(CanonicalSyncSelectionObjective objective,
                                 std::size_t baselineResourceOverflow)
      : objective_(objective),
        baselineResourceOverflow_(baselineResourceOverflow) {}

  /// `diagnosedResourceOverflow` is supplied by the metered repair driver and
  /// ignored for freshly verified trials.
  Decision consider(const CanonicalSyncSelection &trial, bool freshlyVerified,
                    std::size_t diagnosedResourceOverflow,
                    SyncCoverCoverageWorkBudget *workBudget = nullptr);

  std::optional<std::vector<CanonicalSyncMechanismId>>
  getBestVerifiedMechanisms() const;
  std::optional<std::vector<CanonicalSyncMechanismId>>
  getBestPressureMechanisms() const;

private:
  struct RankKey {
    std::size_t resourceOverflow = 0;
    CanonicalSyncStructuralCost cost;
    std::vector<CanonicalSyncMechanismId> mechanisms;
  };

  CanonicalSyncSelectionObjective objective_;
  std::size_t baselineResourceOverflow_ = 0;
  std::optional<RankKey> bestVerified_;
  std::optional<RankKey> bestPressure_;
};

/// Opaque immutable capability created only after the exact selected mechanism
/// set, allocation, graph, physical actions, and deep protocol proofs have been
/// bound together. Its definition is private to materialization.
class CanonicalSyncMaterializationToken;

/// The only result future materialization may consume. Finalization recomputes
/// coverage from the frozen problem and exact event allocation from the
/// selected atomic mechanisms; it does not trust the solver's mutable coverage
/// state.
struct CanonicalSyncVerifiedPlan {
  CanonicalSyncSelectionError error = CanonicalSyncSelectionError::None;
  std::vector<CanonicalSyncMechanismId> mechanisms;
  CanonicalSyncResourceAllocation allocation;
  std::optional<SyncCoverDemandId> firstUncoveredDemand;
  /// Set only after fresh whole-selected-world semantic verification, exact
  /// allocation validation, combined protocol verification, and physical
  /// action reconstruction. Semantic selection alone never sets this bit.
  bool wholePlanWorldVerified = false;
  std::shared_ptr<const CanonicalSyncMaterializationToken> materializationToken;

  explicit operator bool() const {
    return error == CanonicalSyncSelectionError::None;
  }
};

CanonicalSyncVerifiedPlan verifyCanonicalSyncSelection(
    const CanonicalSyncPatternProblem &problem,
    const CanonicalSyncSelection &selection,
    SyncCoverCoverageWorkBudget *coverageWork = nullptr);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCSELECTION_H
