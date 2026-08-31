// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverStorageProtocolGroups.h - Lifecycle groups -*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLGROUPS_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLGROUPS_H

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolSeeds.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverStorageProtocolGroupId = std::size_t;

enum class SyncCoverStorageProtocolBehavior : std::uint8_t {
  /// Every constituent ready/release SCC has an in-iteration ready transfer in
  /// every reachable joint periodic state.
  StableRoundTrip,
  /// At least one constituent SCC obtains readiness from another iteration in
  /// a reachable joint state. A later certificate must prove the lane
  /// permutation.
  PhaseRotatingRoundTrip,
};

/// Target-neutral proposal for seeds that may share one finite-state token
/// protocol. A group is not a certificate and has no physical recipe.
struct SyncCoverStorageProtocolGroup {
  SyncCoverStorageProtocolGroupId id = 0;
  SyncCoverScopeId owningScope = 0;
  SyncCoverStorageProtocolBehavior behavior =
      SyncCoverStorageProtocolBehavior::StableRoundTrip;
  std::uint32_t readySourceResource = 0;
  std::uint32_t readyTargetResource = 0;
  /// Canonical joint-period and per-SCC readiness masks. Rotationally
  /// equivalent lane behaviors have the same signature.
  std::vector<std::uint64_t> behaviorSignature;
  std::vector<SyncCoverStorageProtocolSeedId> seeds;
  std::vector<SyncCoverControlId> periodicControls;
  std::vector<SyncCoverDemandId> demands;
  unsigned maximumDistance = 0;
};

struct SyncCoverStorageProtocolGroupLimits {
  std::size_t maximumWorkUnits = 1U << 22;
  std::size_t maximumGroups = 1U << 14;
  std::size_t maximumSeedIncidences = 1U << 20;
  std::size_t maximumControlIncidences = 1U << 20;
  std::size_t maximumDemandIncidences = 1U << 20;
  std::size_t maximumSlotIncidences = 1U << 20;
  std::size_t maximumJointStateIncidences = 1U << 16;
  std::size_t maximumReachablePhases = 16;
};

struct SyncCoverStorageProtocolGroupStatistics {
  std::size_t workUnits = 0;
  std::size_t eligibleSeeds = 0;
  std::size_t ineligibleSeeds = 0;
  std::size_t stableSeeds = 0;
  std::size_t phaseRotatingSeeds = 0;
  std::size_t groups = 0;
  std::size_t seedIncidences = 0;
  std::size_t controlIncidences = 0;
  std::size_t demandIncidences = 0;
  std::size_t slotIncidences = 0;
  std::size_t jointStateIncidences = 0;
  std::size_t maximumGroupSeeds = 0;
  bool truncated = false;
};

enum class SyncCoverStorageProtocolGroupError : std::uint8_t {
  None,
  InvalidGraph,
  IncompleteLifecycleIndex,
  IncompleteSeedIndex,
  InvalidLimit,
  LimitExceeded,
  ArithmeticOverflow,
};

class SyncCoverStorageProtocolGroupIndex {
public:
  const std::vector<SyncCoverStorageProtocolGroup> &getGroups() const {
    return groups_;
  }
  const SyncCoverStorageProtocolGroupStatistics &getStatistics() const {
    return statistics_;
  }
  SyncCoverStorageProtocolGroupError getError() const { return error_; }
  bool isComplete() const {
    return error_ == SyncCoverStorageProtocolGroupError::None &&
           !statistics_.truncated;
  }

private:
  friend SyncCoverStorageProtocolGroupIndex
  buildSyncCoverStorageProtocolGroupIndex(
      const SyncCoverGraph &, const SyncCoverStorageLifecycleIndex &,
      const SyncCoverStorageProtocolSeedIndex &,
      const SyncCoverStorageProtocolGroupLimits &);

  std::vector<SyncCoverStorageProtocolGroup> groups_;
  SyncCoverStorageProtocolGroupStatistics statistics_;
  SyncCoverStorageProtocolGroupError error_ =
      SyncCoverStorageProtocolGroupError::None;
};

struct SyncCoverStorageProtocolGroupPrefix {
  std::size_t retainedGroups = 0;
  std::size_t retainedSeedIncidences = 0;
  std::size_t retainedBehaviorSignatureEntries = 0;
  bool truncated = false;
};

/// Select a complete-group prefix for bounded diagnostics. No group is
/// partially copied when either aggregate detail bound is reached.
SyncCoverStorageProtocolGroupPrefix boundedSyncCoverStorageProtocolGroupPrefix(
    const std::vector<SyncCoverStorageProtocolGroup> &groups,
    std::size_t maximumGroups, std::size_t maximumSeedIncidences,
    std::size_t maximumBehaviorSignatureEntries);

/// Partition ready/release seeds by their directed resource cycle and by
/// reachable periodic behavior. Exact slots within a group must be disjoint.
/// Limit exhaustion is transactional and publishes no partial group index.
SyncCoverStorageProtocolGroupIndex buildSyncCoverStorageProtocolGroupIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolSeedIndex &seedIndex,
    const SyncCoverStorageProtocolGroupLimits &limits = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLGROUPS_H
