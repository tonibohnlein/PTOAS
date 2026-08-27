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

enum class CanonicalSyncMechanismKind : std::uint8_t {
  Event,
  Barrier,
  Protocol,
};

enum class CanonicalSyncActionKind : std::uint8_t {
  EventSet,
  EventWait,
  Barrier,
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
  /// Positive-distance protocols conservatively own the complete recurrence
  /// timeline in version one.
  std::optional<SyncCoverScopeId> recurrenceScope;
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
};

enum class CanonicalSyncSupplyProof : std::uint8_t {
  DirectAction,
  VerifiedProtocol,
};

struct CanonicalSyncSupplyBinding {
  SyncCoverEdge edge;
  std::optional<std::size_t> eventUse;
  std::optional<std::size_t> barrierAction;
  std::optional<std::size_t> produceAction;
  std::optional<std::size_t> consumeAction;
  CanonicalSyncSupplyProof proof = CanonicalSyncSupplyProof::DirectAction;
};

/// One atomic synchronization unit. This descriptor is also its emission
/// recipe; no provider or selected-mechanism representation is reconstructed
/// after selection.
struct CanonicalSyncMechanismDescriptor {
  CanonicalSyncMechanismKind kind = CanonicalSyncMechanismKind::Event;
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
  SlotLifecycle,
  PipelineScope,
  RoundTrip,
};

struct CanonicalSyncPatternSpec {
  CanonicalSyncPatternKind kind = CanonicalSyncPatternKind::Singleton;
  std::vector<CanonicalSyncMechanismId> members;
};

struct CanonicalSyncPattern {
  CanonicalSyncPatternId id = 0;
  CanonicalSyncPatternKind kind = CanonicalSyncPatternKind::Singleton;
  std::vector<CanonicalSyncMechanismId> members;
  SyncCoverDemandSet coverage;
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
  const SyncCoverDemandSet &getBaselineCoverage() const {
    return baselineCoverage_;
  }

private:
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
  bool graphValid_ = false;
  bool frozen_ = false;
  std::size_t incidenceCount_ = 0;
  std::size_t actionCount_ = 0;
  std::size_t eventUseCount_ = 0;
  std::size_t supplyCount_ = 0;
  std::vector<SyncCoverDemandId> activeDemands_;
  std::vector<CanonicalSyncEventDomain> domains_;
  std::vector<CanonicalSyncMechanism> mechanisms_;
  std::vector<CanonicalSyncPatternSpec> patternSpecs_;
  std::vector<CanonicalSyncPattern> patterns_;
  SyncCoverDemandSet baselineCoverage_;
  std::vector<std::vector<CanonicalSyncPatternId>> demandPatterns_;
  std::vector<std::vector<CanonicalSyncPatternId>> mechanismPatterns_;
  std::map<std::uint64_t, std::vector<CanonicalSyncMechanismId>>
      mechanismBuckets_;
};

struct CanonicalSyncEventAllocation {
  CanonicalSyncMechanismId mechanism = 0;
  std::size_t eventUse = 0;
  std::vector<unsigned> ids;
};

struct CanonicalSyncDomainAllocation {
  CanonicalSyncEventDomainId domain = 0;
  std::size_t required = 0;
  std::size_t available = 0;
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
};

struct CanonicalSyncSelection {
  CanonicalSyncSelectionError error = CanonicalSyncSelectionError::None;
  std::vector<CanonicalSyncMechanismId> mechanisms;
  SyncCoverDemandSet covered;
  CanonicalSyncResourceAllocation allocation;
  CanonicalSyncGreedyStatistics statistics;

  explicit operator bool() const {
    return error == CanonicalSyncSelectionError::None;
  }
};

CanonicalSyncSelection
selectCanonicalSyncPatterns(const CanonicalSyncPatternProblem &problem,
                            CanonicalSyncGreedyOptions options = {});

/// The only result future materialization may consume. Pattern bitsets are not
/// part of this trust boundary: finalization recomputes semantic coverage and
/// exact event allocation from the selected atomic mechanisms.
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
