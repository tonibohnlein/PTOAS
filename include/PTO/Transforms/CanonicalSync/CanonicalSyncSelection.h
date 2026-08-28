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
};

enum class CanonicalSyncMechanismKind : std::uint8_t {
  Event,
  Barrier,
  Protocol,
};

enum class CanonicalSyncSelectionTier : std::uint8_t {
  Precise,
  ScarcityFrontier,
  PipeAllRescue,
};

enum class CanonicalSyncActionKind : std::uint8_t {
  EventSet,
  EventWait,
  Barrier,
};

enum class CanonicalSyncActionGuardKind : std::uint8_t {
  None,
  LoopNonEmpty,
  LoopEmpty,
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
};

/// One atomic synchronization unit. This descriptor is also its emission
/// recipe; no provider or selected-mechanism representation is reconstructed
/// after selection.
struct CanonicalSyncMechanismDescriptor {
  CanonicalSyncMechanismKind kind = CanonicalSyncMechanismKind::Event;
  CanonicalSyncSelectionTier selectionTier =
      CanonicalSyncSelectionTier::Precise;
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
};

struct CanonicalSyncMechanism {
  CanonicalSyncMechanismId id = 0;
  CanonicalSyncMechanismDescriptor descriptor;
  std::vector<CanonicalSyncEventLifetime> eventLifetimes;
  CanonicalSyncMechanismCost cost;
  std::vector<CanonicalSyncMechanismId> conflicts;
};

enum class CanonicalSyncPatternKind : std::uint8_t {
  Singleton,
  DirectPair,
  ScarcityFrontier,
};

constexpr std::size_t kCanonicalSyncPatternKindCount =
    static_cast<std::size_t>(CanonicalSyncPatternKind::ScarcityFrontier) + 1;

struct CanonicalSyncPatternSpec {
  CanonicalSyncPatternKind kind = CanonicalSyncPatternKind::Singleton;
  std::vector<CanonicalSyncMechanismId> members;
};

struct CanonicalSyncPattern {
  CanonicalSyncPatternId id = 0;
  CanonicalSyncPatternKind kind = CanonicalSyncPatternKind::Singleton;
  std::vector<CanonicalSyncMechanismId> members;
  /// Singleton patterns store their complete non-baseline coverage. Composite
  /// patterns store only coverage unavailable from their member singletons.
  SyncCoverDemandSet coverage;
  /// Number of covered demands not available from the union of the members'
  /// singleton coverage. Zero means the pattern is greedy packaging, not a
  /// semantic synchronization composition. Detailed extra bits are transient
  /// during construction so diagnostics do not double pattern storage.
  std::size_t extraCoverageCount = 0;
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
  };

  CanonicalSyncPatternProblem(const SyncCoverGraph &graph,
                              std::vector<SyncCoverDemandId> activeDemands);
  CanonicalSyncPatternProblem(const SyncCoverGraph &graph,
                              std::vector<SyncCoverDemandId> activeDemands,
                              Limits limits,
                              SyncCoverExpansionLimits expansionLimits = {});
  CanonicalSyncPatternProblem(const CanonicalSyncPatternProblem &) = delete;
  CanonicalSyncPatternProblem(CanonicalSyncPatternProblem &&) = default;
  CanonicalSyncPatternProblem &
  operator=(const CanonicalSyncPatternProblem &) = delete;
  CanonicalSyncPatternProblem &
  operator=(CanonicalSyncPatternProblem &&) = delete;

  CanonicalSyncProblemResult addEventDomain(CanonicalSyncEventDomain domain);
  CanonicalSyncProblemResult
  internMechanism(CanonicalSyncMechanismDescriptor descriptor);
  CanonicalSyncProblemResult internVerifiedProtocol(
      CanonicalSyncMechanismDescriptor descriptor,
      const std::function<bool(const CanonicalSyncMechanismDescriptor &)>
          &verifier);
  CanonicalSyncProblemResult addConflict(CanonicalSyncMechanismId first,
                                         CanonicalSyncMechanismId second);
  CanonicalSyncProblemResult addPattern(CanonicalSyncPatternSpec pattern);
  /// Add a pattern whose exact joint and singleton-union coverage were
  /// computed by the shared batched coverage engine. Inputs use graph-global
  /// demand IDs and are projected onto the active problem rows here.
  CanonicalSyncProblemResult
  addPattern(CanonicalSyncPatternSpec pattern,
             const SyncCoverDemandSet &jointCoverage,
             const SyncCoverDemandSet &singletonCoverage);
  CanonicalSyncProblemResult freeze();

  bool isFrozen() const { return frozen_; }
  const SyncCoverGraph &getGraph() const { return graph_; }
  const SyncCoverExpandedProgram &getExpansion() const { return expansion_; }
  const std::vector<SyncCoverDemandId> &getDemands() const {
    return activeDemands_;
  }
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
  void markPatternGenerationTruncated() { patternGenerationTruncated_ = true; }
  void recordDirectPairGeneration(std::size_t proposals,
                                  std::size_t evaluations) {
    patternStatistics_.directPairProposals = proposals;
    patternStatistics_.directPairEvaluations = evaluations;
  }
  const SyncCoverDemandSet &getBaselineCoverage() const {
    return baselineCoverage_;
  }
  const Limits &getLimits() const { return limits_; }

private:
  struct PendingPattern {
    CanonicalSyncPatternSpec spec;
    SyncCoverDemandSet coverage;
    std::size_t jointCoverageCount = 0;
    std::size_t singletonCoverageCount = 0;
    std::size_t extraCoverageCount = 0;
  };

  CanonicalSyncProblemResult internMechanismImpl(
      CanonicalSyncMechanismDescriptor descriptor, bool protocolVerified,
      const std::function<bool(const CanonicalSyncMechanismDescriptor &)>
          &verifier = {});
  CanonicalSyncProblemResult
  validateAndCostMechanism(CanonicalSyncMechanismDescriptor &descriptor,
                           std::vector<CanonicalSyncEventLifetime> &lifetimes,
                           CanonicalSyncMechanismCost &cost,
                           bool protocolVerified) const;
  CanonicalSyncProblemResult buildPatterns();

  const SyncCoverGraph &graph_;
  SyncCoverExpandedProgram expansion_;
  Limits limits_;
  std::vector<std::uint32_t> issueResources_;
  bool graphValid_ = false;
  bool frozen_ = false;
  std::size_t incidenceCount_ = 0;
  std::size_t actionCount_ = 0;
  std::size_t eventUseCount_ = 0;
  std::size_t supplyCount_ = 0;
  std::vector<SyncCoverDemandId> activeDemands_;
  std::vector<CanonicalSyncEventDomain> domains_;
  std::vector<CanonicalSyncMechanism> mechanisms_;
  std::vector<PendingPattern> patternSpecs_;
  std::optional<SyncCoverDemandSet> constructionBaselineCoverage_;
  std::vector<std::optional<SyncCoverDemandSet>> constructionSingletonCoverage_;
  std::size_t retainedPatternCount_ = 0;
  bool patternGenerationTruncated_ = false;
  std::vector<CanonicalSyncPattern> patterns_;
  CanonicalSyncPatternStatistics patternStatistics_;
  SyncCoverDemandSet baselineCoverage_;
  std::vector<std::vector<CanonicalSyncPatternId>> demandPatterns_;
  std::vector<std::vector<CanonicalSyncPatternId>> mechanismPatterns_;
  std::map<std::uint64_t, std::vector<CanonicalSyncMechanismId>>
      mechanismBuckets_;
};

struct CanonicalSyncDirectPairOptions {
  /// Proposals are owned by the LCA of their mechanism scopes. An oversized
  /// scope is skipped as a whole so truncation never depends on ID order.
  std::size_t maximumEvaluationsPerScope = 1U << 12;
  /// Pair preparation is optional. A scope whose exact pair matrices exceed
  /// these limits is skipped without weakening singleton correctness.
  SyncCoverCoverageLimits pairCoverageLimits;
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
    const std::vector<CanonicalSyncMechanismId> &selected);

struct CanonicalSyncGreedyStatistics {
  std::size_t patternEvaluations = 0;
  std::size_t deletionEvaluations = 0;
  /// Aggregate bound over incidence visits, bitset words, selected mechanisms,
  /// and resource evaluations performed by greedy selection and deletion.
  std::size_t workUnits = 0;
};

/// Calibration-free static cost. Action profiles are indexed by natural loop
/// depth, with deeper entries compared before shallower entries.
struct CanonicalSyncStructuralCost {
  std::vector<std::uint64_t> actionProfile;
  std::uint64_t serializationBreadth = 0;
  std::uint64_t eventLifetimeArea = 0;
  std::size_t mechanismCount = 0;
};

enum class CanonicalSyncSelectionError : std::uint8_t {
  None,
  InvalidProblem,
  NoCoveringPattern,
  ResourceInfeasible,
  WorkLimitExceeded,
  FinalValidationFailed,
};

struct CanonicalSyncGreedyOptions {
  std::size_t maximumWorkUnits = 1U << 27;
  CanonicalSyncSelectionStrategy strategy =
      CanonicalSyncSelectionStrategy::PairLookahead;
  CanonicalSyncSelectionTier maximumTier = CanonicalSyncSelectionTier::Precise;
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

CanonicalSyncStructuralCost computeCanonicalSyncStructuralCost(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected);

/// The only result future materialization may consume. Finalization recomputes
/// coverage from the frozen pattern table and exact event allocation from the
/// selected atomic mechanisms; it does not trust the solver's mutable coverage
/// state.
struct CanonicalSyncVerifiedPlan {
  CanonicalSyncSelectionError error = CanonicalSyncSelectionError::None;
  std::vector<CanonicalSyncMechanismId> mechanisms;
  CanonicalSyncResourceAllocation allocation;
  std::optional<SyncCoverDemandId> firstUncoveredDemand;

  explicit operator bool() const {
    return error == CanonicalSyncSelectionError::None;
  }
};

CanonicalSyncVerifiedPlan
verifyCanonicalSyncSelection(const CanonicalSyncPatternProblem &problem,
                             const CanonicalSyncSelection &selection);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCSELECTION_H
