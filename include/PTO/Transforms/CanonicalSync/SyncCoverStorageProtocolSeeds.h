// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncCoverStorageProtocolSeeds.h - Lifecycle owner seeds -*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLSEEDS_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLSEEDS_H

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageLifecycle.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverStorageProtocolSeedId = std::size_t;

struct SyncCoverStorageProtocolSlotRef {
  SyncCoverStorageLifecycleComponentId component = 0;
  SyncCoverStorageLifecycleSlotId slot = 0;
};

struct SyncCoverStorageProtocolSccRef {
  SyncCoverStorageLifecycleComponentId component = 0;
  SyncCoverStorageLifecycleSccId scc = 0;
};

/// One exact-storage owner collected across nested lifecycle scopes. This is
/// target-neutral input to later lane grouping and finite-state protocol
/// synthesis. It is not by itself a certificate or physical recipe.
struct SyncCoverStorageProtocolSeed {
  SyncCoverStorageProtocolSeedId id = 0;
  SyncCoverStorageAccessFamilyId family = 0;
  SyncCoverScopeId owningScope = 0;
  std::vector<SyncCoverStorageLifecycleComponentId> components;
  std::vector<SyncCoverStorageProtocolSlotRef> slots;
  std::vector<SyncCoverStorageProtocolSccRef> readyReleaseSccs;
  std::vector<SyncCoverDemandId> demands;
  SyncCoverStorageLifecycleEdgeKindMask kinds = 0;
  unsigned maximumDistance = 0;
};

struct SyncCoverStorageProtocolSeedLimits {
  std::size_t maximumWorkUnits = 1U << 22;
  std::size_t maximumSeeds = 1U << 14;
  std::size_t maximumComponentIncidences = 1U << 20;
  std::size_t maximumSlotIncidences = 1U << 20;
  std::size_t maximumSccIncidences = 1U << 20;
  std::size_t maximumDemandIncidences = 1U << 20;
};

struct SyncCoverStorageProtocolSeedStatistics {
  std::size_t workUnits = 0;
  std::size_t seeds = 0;
  std::size_t readyReleaseSeeds = 0;
  std::size_t componentIncidences = 0;
  std::size_t slotIncidences = 0;
  std::size_t sccIncidences = 0;
  std::size_t demandIncidences = 0;
  std::size_t maximumSeedComponents = 0;
  std::size_t maximumSeedSlots = 0;
  std::size_t maximumSeedSccs = 0;
  bool truncated = false;
};

enum class SyncCoverStorageProtocolSeedError : std::uint8_t {
  None,
  InvalidGraph,
  IncompleteLifecycleIndex,
  InvalidLimit,
  LimitExceeded,
  ArithmeticOverflow,
};

class SyncCoverStorageProtocolSeedIndex {
public:
  const std::vector<SyncCoverStorageProtocolSeed> &getSeeds() const {
    return seeds_;
  }
  const SyncCoverStorageProtocolSeedStatistics &getStatistics() const {
    return statistics_;
  }
  SyncCoverStorageProtocolSeedError getError() const { return error_; }
  bool isComplete() const {
    return error_ == SyncCoverStorageProtocolSeedError::None &&
           !statistics_.truncated;
  }

private:
  friend SyncCoverStorageProtocolSeedIndex
  buildSyncCoverStorageProtocolSeedIndex(
      const SyncCoverGraph &, const SyncCoverStorageLifecycleIndex &,
      const SyncCoverStorageProtocolSeedLimits &);

  std::vector<SyncCoverStorageProtocolSeed> seeds_;
  SyncCoverStorageProtocolSeedStatistics statistics_;
  SyncCoverStorageProtocolSeedError error_ =
      SyncCoverStorageProtocolSeedError::None;
};

/// Merge exact-storage lifecycle components with the same provenance family
/// across nested scopes. A bound failure is transactional and publishes no
/// partial seed index.
SyncCoverStorageProtocolSeedIndex buildSyncCoverStorageProtocolSeedIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolSeedLimits &limits = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLSEEDS_H
