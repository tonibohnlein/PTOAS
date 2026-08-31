// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverStorageProtocolFrontiers.h - Lifecycle cuts -*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLFRONTIERS_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLFRONTIERS_H

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolAutomata.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverStorageProtocolFrontierId = std::size_t;
using SyncCoverStorageProtocolFrontierPlanId = std::size_t;

enum class SyncCoverStorageProtocolFrontierKind : std::uint8_t {
  Ready,
  Reuse,
};

/// One target-authorized endpoint transfer retained under the lifecycle
/// automaton that proves its reachable guard states. Unlike an ordinary direct
/// rectangle, this frontier is not independently balanced. A later lifecycle
/// certificate must prove priming, steady-state circulation, and draining
/// before it can become a physical event recipe.
struct SyncCoverStorageProtocolFrontier {
  SyncCoverStorageProtocolFrontierId id = 0;
  SyncCoverStorageProtocolAutomatonId automaton = 0;
  std::optional<SyncCoverStorageProtocolTransferId> transfer;
  SyncCoverStorageProtocolFrontierKind kind =
      SyncCoverStorageProtocolFrontierKind::Ready;
  std::optional<SyncCoverStorageLifecycleEdgeRef> edge;
  SyncCoverAnchor completionAnchor;
  SyncCoverAnchor acquisitionAnchor;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  std::optional<SyncCoverCompletionCutFactId> completionCutFact;
  std::optional<SyncCoverTargetCompletionCertificateId> completionCertificate;
};

/// Bounded endpoint-frontier candidate pool for one finite-state storage
/// lifecycle. A later rectangle grounding and lifecycle certificate must prove
/// that a selected subset covers every transfer; the pool is not itself a
/// complete synchronization plan. Same-resource reuse transfers remain
/// implicit issue-order obligations.
struct SyncCoverStorageProtocolFrontierPlan {
  SyncCoverStorageProtocolFrontierPlanId id = 0;
  SyncCoverStorageProtocolAutomatonId automaton = 0;
  SyncCoverStorageProtocolGroupId group = 0;
  SyncCoverScopeId owningScope = 0;
  std::size_t laneCount = 0;
  std::vector<SyncCoverStorageProtocolFrontierId> frontiers;
  std::size_t readyFrontiers = 0;
  std::size_t reuseFrontiers = 0;
  std::size_t directFrontiers = 0;
  std::size_t completionCutFactFrontiers = 0;
  std::size_t certificateFrontiers = 0;
  std::size_t sameResourceRecurrenceReuses = 0;
};

struct SyncCoverStorageProtocolFrontierLimits {
  std::size_t maximumWorkUnits = 1U << 24;
  std::size_t maximumPlans = 1U << 14;
  std::size_t maximumFrontiers = 1U << 21;
  std::size_t maximumTransferInspections = 1U << 20;
  std::size_t maximumStatePairInspections = 1U << 22;
  std::size_t maximumPlanFrontierIncidences = 1U << 21;
  std::size_t maximumCertificateDemandIncidences = 1U << 20;
  std::size_t maximumCompletionCutFactDemandIncidences = 1U << 20;
};

struct SyncCoverStorageProtocolFrontierStatistics {
  std::size_t workUnits = 0;
  std::size_t eligibleAutomata = 0;
  std::size_t ineligibleAutomata = 0;
  std::size_t missingReadyAutomata = 0;
  std::size_t missingRecurrenceReuseAutomata = 0;
  std::size_t missingCompletionFrontierAutomata = 0;
  std::size_t plans = 0;
  std::size_t frontiers = 0;
  std::size_t readyFrontiers = 0;
  std::size_t reuseFrontiers = 0;
  std::size_t directFrontiers = 0;
  std::size_t completionCutFactFrontiers = 0;
  std::size_t certificateFrontiers = 0;
  std::size_t sameResourceRecurrenceReuses = 0;
  std::size_t certificateDemandIncidences = 0;
  std::size_t completionCutFactDemandIncidences = 0;
  std::size_t transferInspections = 0;
  std::size_t statePairInspections = 0;
  std::size_t planFrontierIncidences = 0;
  std::size_t maximumPlanFrontiers = 0;
  bool truncated = false;
};

enum class SyncCoverStorageProtocolFrontierError : std::uint8_t {
  None,
  InvalidGraph,
  IncompleteLifecycleIndex,
  IncompleteAutomatonIndex,
  InvalidLimit,
  LimitExceeded,
  ArithmeticOverflow,
};

class SyncCoverStorageProtocolFrontierIndex {
public:
  const std::vector<SyncCoverStorageProtocolFrontier> &getFrontiers() const {
    return frontiers_;
  }
  const std::vector<SyncCoverStorageProtocolFrontierPlan> &getPlans() const {
    return plans_;
  }
  const SyncCoverStorageProtocolFrontierStatistics &getStatistics() const {
    return statistics_;
  }
  SyncCoverStorageProtocolFrontierError getError() const { return error_; }
  bool isComplete() const {
    return error_ == SyncCoverStorageProtocolFrontierError::None &&
           !statistics_.truncated;
  }
  bool isForGraph(const SyncCoverGraph &graph) const {
    return ownerIdentity_ && ownerIdentity_ == graph.getIdentity() &&
           graph.isStructureFrozen();
  }

private:
  friend SyncCoverStorageProtocolFrontierIndex
  buildSyncCoverStorageProtocolFrontierIndex(
      const SyncCoverGraph &, const SyncCoverStorageLifecycleIndex &,
      const SyncCoverStorageProtocolAutomatonIndex &,
      const SyncCoverStorageProtocolFrontierLimits &);

  void bindToGraph(const SyncCoverGraph &graph) {
    ownerIdentity_ = graph.getIdentity();
  }

  std::shared_ptr<const std::uint8_t> ownerIdentity_;
  std::vector<SyncCoverStorageProtocolFrontier> frontiers_;
  std::vector<SyncCoverStorageProtocolFrontierPlan> plans_;
  SyncCoverStorageProtocolFrontierStatistics statistics_;
  SyncCoverStorageProtocolFrontierError error_ =
      SyncCoverStorageProtocolFrontierError::None;
};

/// Enumerate lifecycle-local endpoint frontiers without requiring each
/// set/wait pair to coexecute independently. An automaton is retained when the
/// pool contains at least one ready frontier and one recurrence-reuse path;
/// transfers without an endpoint frontier are reported but do not prematurely
/// reject cut factoring. Reachable automaton state pairs remain authoritative;
/// this index does not prove coverage or a balanced protocol and does not
/// affect selection, verification, or materialization. Structural and bound
/// failures are transactional.
SyncCoverStorageProtocolFrontierIndex
buildSyncCoverStorageProtocolFrontierIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolAutomatonIndex &automatonIndex,
    const SyncCoverStorageProtocolFrontierLimits &limits = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLFRONTIERS_H
