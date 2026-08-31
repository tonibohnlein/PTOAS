// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverStorageProtocolAutomata.h - Lifecycle automata -*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLAUTOMATA_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLAUTOMATA_H

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolGroups.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverStorageProtocolAutomatonId = std::size_t;
using SyncCoverStorageProtocolTransferId = std::size_t;

struct SyncCoverStorageProtocolStatePair {
  std::size_t source = 0;
  std::size_t target = 0;

  bool operator==(const SyncCoverStorageProtocolStatePair &other) const {
    return source == other.source && target == other.target;
  }
};

/// One original lifecycle obligation projected onto the reachable periodic
/// states in which its source and distance-shifted target can execute.
struct SyncCoverStorageProtocolTransfer {
  SyncCoverStorageProtocolTransferId id = 0;
  SyncCoverStorageLifecycleEdgeRef edge;
  SyncCoverDemandId demand = 0;
  SyncCoverStorageLifecycleEdgeKindMask kinds = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  /// Reachable endpoint-state pairs. Same-loop recurrences advance by their
  /// distance, recurrences nested under the phase loop retain its state, and
  /// recurrences enclosing the phase loop may cross independent invocations.
  std::vector<SyncCoverStorageProtocolStatePair> activeStatePairs;
};

/// Target-neutral finite-state projection of one protocol-group proposal.
/// This is not a lifecycle certificate, completion supply, or physical recipe.
struct SyncCoverStorageProtocolAutomaton {
  SyncCoverStorageProtocolAutomatonId id = 0;
  SyncCoverStorageProtocolGroupId group = 0;
  SyncCoverScopeId owningScope = 0;
  std::size_t stateCount = 0;
  std::vector<SyncCoverStorageProtocolTransfer> transfers;
  std::size_t statePairIncidences = 0;
  unsigned maximumDistance = 0;
};

struct SyncCoverStorageProtocolAutomatonLimits {
  std::size_t maximumWorkUnits = 1U << 24;
  std::size_t maximumAutomata = 1U << 14;
  std::size_t maximumStates = 1U << 16;
  std::size_t maximumTransfers = 1U << 20;
  std::size_t maximumStatePairIncidences = 1U << 22;
  std::size_t maximumLanes = 8;
};

struct SyncCoverStorageProtocolAutomatonStatistics {
  std::size_t workUnits = 0;
  std::size_t eligibleGroups = 0;
  std::size_t ineligibleGroups = 0;
  std::size_t laneLimitedGroups = 0;
  std::size_t scopeRejectedGroups = 0;
  std::size_t membershipRejectedGroups = 0;
  std::size_t directionRejectedGroups = 0;
  std::size_t unreachableTransferGroups = 0;
  std::size_t unreachableReadyTransferGroups = 0;
  std::size_t unreachableReleaseTransferGroups = 0;
  std::size_t unreachableExclusionTransferGroups = 0;
  std::size_t demandSetMismatchGroups = 0;
  std::size_t distanceMismatchGroups = 0;
  std::size_t automata = 0;
  std::size_t states = 0;
  std::size_t transfers = 0;
  std::size_t statePairIncidences = 0;
  std::size_t maximumAutomatonTransfers = 0;
  std::size_t maximumTransferStatePairs = 0;
  bool truncated = false;
};

enum class SyncCoverStorageProtocolAutomatonError : std::uint8_t {
  None,
  InvalidGraph,
  IncompleteLifecycleIndex,
  IncompleteSeedIndex,
  IncompleteGroupIndex,
  InvalidLimit,
  LimitExceeded,
  ArithmeticOverflow,
};

class SyncCoverStorageProtocolAutomatonIndex {
public:
  const std::vector<SyncCoverStorageProtocolAutomaton> &getAutomata() const {
    return automata_;
  }
  const SyncCoverStorageProtocolAutomatonStatistics &getStatistics() const {
    return statistics_;
  }
  SyncCoverStorageProtocolAutomatonError getError() const { return error_; }
  bool isComplete() const {
    return error_ == SyncCoverStorageProtocolAutomatonError::None &&
           !statistics_.truncated;
  }

private:
  friend SyncCoverStorageProtocolAutomatonIndex
  buildSyncCoverStorageProtocolAutomatonIndex(
      const SyncCoverGraph &, const SyncCoverStorageLifecycleIndex &,
      const SyncCoverStorageProtocolSeedIndex &,
      const SyncCoverStorageProtocolGroupIndex &,
      const SyncCoverStorageProtocolAutomatonLimits &);

  std::vector<SyncCoverStorageProtocolAutomaton> automata_;
  SyncCoverStorageProtocolAutomatonStatistics statistics_;
  SyncCoverStorageProtocolAutomatonError error_ =
      SyncCoverStorageProtocolAutomatonError::None;
};

/// Project each complete protocol group onto reachable periodic states. A
/// positive-distance edge advances exactly `distance` states only when its
/// recurrence scope owns the periodic relation. Enclosing recurrences match
/// independently reachable child-loop endpoint phases; nested recurrences
/// retain their enclosing phase. Groups with unreachable edges or more lanes
/// than the configured synthesis bound are omitted fail-closed. Resource and
/// work exhaustion is transactional.
SyncCoverStorageProtocolAutomatonIndex
buildSyncCoverStorageProtocolAutomatonIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolSeedIndex &seedIndex,
    const SyncCoverStorageProtocolGroupIndex &groupIndex,
    const SyncCoverStorageProtocolAutomatonLimits &limits = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLAUTOMATA_H
