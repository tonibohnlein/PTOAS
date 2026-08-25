// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverMechanism.h - Atomic synchronization supply ----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERMECHANISM_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERMECHANISM_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

/// Unified public API for mechanism construction and selection evaluation.
/// Resource feasibility and structural cost remain here because both are
/// derived from one immutable mechanism-universe selection.
using SyncCoverResourceDomainId = std::size_t;

enum class SyncCoverMechanismKind : std::uint8_t {
  EventBundle,
  Barrier,
  VerifiedProtocol,
};

enum class SyncCoverResourceKind : std::uint8_t {
  EventId,
  BufferToken,
};

struct SyncCoverResourceDomain {
  SyncCoverResourceDomainId id = 0;
  SyncCoverResourceKind kind = SyncCoverResourceKind::EventId;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  /// BufferToken domains require one globally unique nonzero identity. Shared
  /// physical pools are rejected until feasibility and component splitting can
  /// reason about them jointly. EventId domains always use zero.
  std::uint64_t poolIdentity = 0;
  unsigned budget = 0;
  /// Sorted unique reservations. IDs outside [0, budget) remain visible for
  /// diagnostics but do not reduce availability.
  std::vector<unsigned> reservedIds;
};

enum class SyncCoverResourceActionKind : std::uint8_t {
  Produce,
  Consume,
};

/// One physical synchronization action. A logical EventId use and a logical
/// BufferToken use may share this action without duplicating emitted cost.
struct SyncCoverResourceAction {
  SyncCoverResourceActionKind kind = SyncCoverResourceActionKind::Produce;
  std::uint32_t resource = 0;
  SyncCoverAnchor anchor;
};

/// One periodic resource lifetime. Straight lifetimes enclose every bound
/// action. Positive distances conservatively occupy the complete explicit
/// recurrence-scope timeline in version one.
struct SyncCoverResourceUse {
  SyncCoverResourceDomainId domain = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  std::size_t width = 1;
  std::vector<std::size_t> actions;
  /// Descriptor indices are local to supplyEdges. Stored mechanism indices
  /// are rewritten to graph edge indices by the universe.
  std::vector<std::size_t> supplyEdges;
};

/// Associates one completion-supply edge with the physical actions that
/// implement it. Action and resource-use indices remain mechanism-local.
struct SyncCoverSupplyBinding {
  std::size_t supplyEdge = 0;
  std::size_t resourceUse = 0;
  std::size_t produceAction = 0;
  std::size_t consumeAction = 0;
};

struct SyncCoverBarrierPlacement {
  std::uint32_t resource = 0;
  SyncCoverAnchor anchor;
  SyncCoverScopeId scope = 0;

  SyncCoverBarrierPlacement() = default;
  SyncCoverBarrierPlacement(std::uint32_t resource, SyncCoverNodeId anchor,
                            SyncCoverScopeId scope)
      : resource(resource),
        anchor{SyncCoverAnchorKind::BeforeNode, anchor, 0, 0}, scope(scope) {}
  SyncCoverBarrierPlacement(std::uint32_t resource, SyncCoverAnchor anchor,
                            SyncCoverScopeId scope)
      : resource(resource), anchor(anchor), scope(scope) {}
};

struct SyncCoverMechanismDescriptor {
  SyncCoverMechanismKind kind = SyncCoverMechanismKind::EventBundle;
  std::uint64_t providerIdentity = 0;
  std::optional<SyncCoverBarrierPlacement> barrier;
  std::vector<SyncCoverEdge> supplyEdges;
  std::vector<SyncCoverResourceAction> actions;
  std::vector<SyncCoverResourceUse> resourceUses;
  std::vector<SyncCoverSupplyBinding> supplyBindings;
};

struct SyncCoverMechanism {
  SyncCoverMechanismId id = 0;
  SyncCoverMechanismKind kind = SyncCoverMechanismKind::EventBundle;
  std::uint64_t providerIdentity = 0;
  bool protocolVerified = false;
  std::optional<SyncCoverBarrierPlacement> barrier;
  std::vector<std::size_t> supplyEdges;
  std::vector<SyncCoverResourceAction> actions;
  std::vector<SyncCoverResourceUse> resourceUses;
  std::vector<SyncCoverSupplyBinding> supplyBindings;
  std::vector<SyncCoverMechanismId> conflicts;
};

enum class SyncCoverMechanismError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidDomain,
  InvalidAction,
  InvalidResourceUse,
  InvalidBinding,
  EmptySupply,
  InvalidSupply,
  UnverifiedProtocol,
  InvalidMechanism,
  InvalidConflict,
};

enum class SyncCoverResourceSelectionError : std::uint8_t {
  None,
  InvalidUniverse,
  InvalidSelection,
  Conflict,
  ArithmeticOverflow,
};

struct SyncCoverResourceWitnessUse {
  SyncCoverMechanismId mechanism = 0;
  std::size_t resourceUse = 0;
  std::size_t width = 0;

  bool operator==(const SyncCoverResourceWitnessUse &other) const;
};

struct SyncCoverResourceAllocation {
  SyncCoverResourceWitnessUse owner;
  std::vector<unsigned> ids;

  bool operator==(const SyncCoverResourceAllocation &other) const;
};

struct SyncCoverDomainFeasibility {
  SyncCoverResourceDomainId domain = 0;
  std::size_t required = 0;
  std::size_t available = 0;
  std::size_t overflow = 0;
  std::optional<SyncCoverTimelinePosition> maximumPoint;
  std::vector<SyncCoverResourceWitnessUse> maximumClique;
  /// Deterministic physical IDs for every selected use when overflow is zero.
  std::vector<SyncCoverResourceAllocation> allocations;
};

struct SyncCoverResourceSelection {
  SyncCoverResourceSelectionError error = SyncCoverResourceSelectionError::None;
  bool resourceFeasible = false;
  std::optional<SyncCoverMechanismId> firstConflict;
  std::optional<SyncCoverMechanismId> secondConflict;
  std::vector<SyncCoverDomainFeasibility> domains;

  bool isValid() const {
    return error == SyncCoverResourceSelectionError::None;
  }
  explicit operator bool() const { return isValid() && resourceFeasible; }
};

enum class SyncCoverStructuralCostError : std::uint8_t {
  None,
  NotEvaluated,
  InvalidUniverse,
  InvalidSelection,
  Conflict,
  ResourceInfeasible,
  ArithmeticOverflow,
};

/// Symbolic static cost. Profiles are ordered from deepest loop nesting to
/// one-shot function scope. Physical actions are counted once even when they
/// carry both EventId and BufferToken uses.
struct SyncCoverStructuralCost {
  SyncCoverStructuralCostError error =
      SyncCoverStructuralCostError::NotEvaluated;
  std::vector<std::size_t> actionProfile;
  std::vector<std::size_t> barrierActionProfile;
  std::size_t peakEventPressure = 0;
  std::size_t totalEventPressure = 0;
  std::size_t minimumEventHeadroom = 0;
  std::size_t eventDomainCount = 0;
  std::size_t mechanismCount = 0;
  std::vector<SyncCoverMechanismId> signature;

  explicit operator bool() const {
    return error == SyncCoverStructuralCostError::None;
  }
};

bool syncCoverStructuralCostLess(const SyncCoverStructuralCost &first,
                                 const SyncCoverStructuralCost &second);

struct SyncCoverSelectionEvaluation {
  SyncCoverResourceSelection resources;
  SyncCoverStructuralCost cost;

  explicit operator bool() const {
    return static_cast<bool>(resources) && static_cast<bool>(cost);
  }
};

struct SyncCoverMechanismResult {
  SyncCoverMechanismError error = SyncCoverMechanismError::None;
  /// On success, identifies the inserted or existing object. On failure,
  /// identifies an existing invalid object when one is known; candidate
  /// insertion positions are never reported as object identities.
  std::optional<std::size_t> index;

  explicit operator bool() const {
    return error == SyncCoverMechanismError::None;
  }
};

struct SyncCoverUniverseStatistics {
  std::size_t fullValidations = 0;
};

/// Owns mechanism identity and atomically attaches supplied completion edges
/// to a SyncCoverGraph. Failed additions leave both objects unchanged. This
/// mutable construction object is not thread-safe.
class SyncCoverMechanismUniverse {
public:
  explicit SyncCoverMechanismUniverse(SyncCoverGraph &graph);
  SyncCoverMechanismUniverse(const SyncCoverMechanismUniverse &) = delete;
  SyncCoverMechanismUniverse(SyncCoverMechanismUniverse &&) = delete;
  SyncCoverMechanismUniverse &
  operator=(const SyncCoverMechanismUniverse &) = delete;
  SyncCoverMechanismUniverse &operator=(SyncCoverMechanismUniverse &&) =
      delete;

  SyncCoverMechanismResult
  addResourceDomain(SyncCoverResourceKind kind, std::uint32_t sourceResource,
                    std::uint32_t targetResource, unsigned budget,
                    std::uint64_t poolIdentity = 0,
                    std::vector<unsigned> reservedIds = {});
  SyncCoverMechanismResult
  addMechanism(const SyncCoverMechanismDescriptor &descriptor);
  /// The verifier is trusted, side-effect-free compiler infrastructure. It
  /// runs after non-mutating structural validation and before the graph commit.
  /// Mutating the graph or universe from the callback violates this contract.
  SyncCoverMechanismResult addVerifiedProtocol(
      const SyncCoverMechanismDescriptor &descriptor,
      const std::function<bool(const SyncCoverMechanismDescriptor &)> &verify);
  SyncCoverMechanismResult addConflict(SyncCoverMechanismId first,
                                       SyncCoverMechanismId second);
  SyncCoverMechanismResult validate() const;
  SyncCoverResourceSelection evaluateResourceSelection(
      const std::vector<SyncCoverMechanismId> &selected) const;
  SyncCoverStructuralCost evaluateStructuralCost(
      const std::vector<SyncCoverMechanismId> &selected) const;

  const std::vector<SyncCoverResourceDomain> &getResourceDomains() const {
    return domains_;
  }
  const std::vector<SyncCoverMechanism> &getMechanisms() const {
    return mechanisms_;
  }
  const SyncCoverGraph &getGraph() const { return graph_; }
  std::size_t getVersion() const { return version_; }
  SyncCoverGraphResult getInitializationResult() const {
    return initializationResult_;
  }
  SyncCoverUniverseStatistics getStatistics() const {
    return {fullValidationCount_};
  }

private:
  friend class SyncCoverSelectionEvaluator;

  SyncCoverMechanismResult addMechanismImpl(
      const SyncCoverMechanismDescriptor &descriptor, bool protocolVerified,
      const std::function<bool(const SyncCoverMechanismDescriptor &)> &verify =
          {});
  bool ensureConstructionState();
  void noteSuccessfulMutation();
  SyncCoverMechanismResult validateUncached() const;
  SyncCoverMechanismError validateResourceUse(
      SyncCoverMechanismKind kind, const SyncCoverResourceUse &use,
      const std::vector<SyncCoverEdge> &edges,
      const std::vector<SyncCoverResourceAction> &actions) const;
  SyncCoverMechanismError validateSupplyBindings(
      SyncCoverMechanismKind kind, const std::vector<SyncCoverEdge> &edges,
      const std::vector<std::size_t> &supplyEdges,
      const std::vector<SyncCoverResourceAction> &actions,
      const std::vector<SyncCoverResourceUse> &uses,
      const std::vector<SyncCoverSupplyBinding> &bindings) const;
  SyncCoverMechanismError
  validateBarrier(const SyncCoverMechanismDescriptor &descriptor) const;
  SyncCoverResourceSelection evaluateResourceSelectionImpl(
      const std::vector<SyncCoverMechanismId> &selected,
      bool validateUniverse) const;
  SyncCoverStructuralCost
  evaluateStructuralCostImpl(const std::vector<SyncCoverMechanismId> &selected,
                             const SyncCoverResourceSelection &resources) const;

  SyncCoverGraph &graph_;
  SyncCoverGraphResult initializationResult_;
  std::vector<SyncCoverResourceDomain> domains_;
  std::vector<SyncCoverMechanism> mechanisms_;
  std::size_t version_ = 0;
  std::size_t knownGraphGeneration_ = 0;
  bool constructionValidated_ = false;
  mutable std::optional<std::size_t> cachedValidationVersion_;
  mutable std::size_t cachedGraphGeneration_ = 0;
  mutable SyncCoverMechanismResult cachedValidation_;
  mutable std::size_t fullValidationCount_ = 0;
};

/// Immutable phase-boundary evaluator. The universe is validated once; each
/// selection is colored once and its structural cost is derived from that
/// authoritative resource result.
class SyncCoverSelectionEvaluator {
public:
  explicit SyncCoverSelectionEvaluator(
      const SyncCoverMechanismUniverse &universe);

  explicit operator bool() const {
    return valid_ && universe_.version_ == version_ &&
           universe_.graph_.getGeneration() == graphGeneration_;
  }

  SyncCoverSelectionEvaluation
  evaluate(const std::vector<SyncCoverMechanismId> &selected) const;

private:
  const SyncCoverMechanismUniverse &universe_;
  std::size_t version_ = 0;
  std::size_t graphGeneration_ = 0;
  bool valid_ = false;
};

std::optional<SyncCoverTimelineInterval>
getSyncCoverResourceLifetime(const SyncCoverGraph &graph,
                             const SyncCoverMechanism &mechanism,
                             const SyncCoverResourceUse &use);

/// Returns whether a barrier at the exact placement can supply the completion
/// edge without relying on a control path on which the barrier may not run.
bool syncCoverBarrierCanSupply(const SyncCoverGraph &graph,
                               const SyncCoverBarrierPlacement &placement,
                               const SyncCoverEdge &edge);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERMECHANISM_H
