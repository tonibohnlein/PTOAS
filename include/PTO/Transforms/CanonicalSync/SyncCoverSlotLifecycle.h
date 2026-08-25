// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverSlotLifecycle.h - Physical-slot lifecycles -----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSLOTLIFECYCLE_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSLOTLIFECYCLE_H

#include "PTO/Transforms/CanonicalSync/SyncCoverCandidateIndex.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverSlotLifecycleId = std::size_t;

enum class SyncCoverSlotLifecycleError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidCandidateIndex,
};

struct SyncCoverPhysicalSlot {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval extent;
};

struct SyncCoverSlotLifecycleOptions {
  std::size_t maximumLifecycles = 4096;
};

/// One candidate producer/consumer ownership round trip for one exact access
/// extent. Partial and partitioned views are deliberately rejected in version
/// one. This is discovery evidence, not a verified or emittable protocol.
/// Protocol factories must independently prove access closure over
/// `managedAccesses`, anchors, guards, and token balance.
struct SyncCoverSlotLifecycle {
  SyncCoverSlotLifecycleId id = 0;
  SyncCoverPhysicalSlot slot;
  std::uint32_t producerResource = 0;
  std::uint32_t consumerResource = 0;
  SyncCoverScopeId recurrenceScope = 0;
  unsigned distance = 0;
  std::vector<SyncCoverCandidateOpportunityId> ready;
  std::vector<SyncCoverCandidateOpportunityId> release;
  std::vector<SyncCoverStorageAccessId> managedAccesses;
  bool hasUnrepresentedAccesses = false;
  bool requiresPathSensitiveProof = false;
};

struct SyncCoverSlotLifecycleResult {
  SyncCoverSlotLifecycleError error = SyncCoverSlotLifecycleError::None;
  std::vector<SyncCoverSlotLifecycle> lifecycles;
  std::size_t partialSlotOpportunities = 0;
  bool truncated = false;

  explicit operator bool() const {
    return error == SyncCoverSlotLifecycleError::None;
  }
};

SyncCoverSlotLifecycleResult discoverSyncCoverSlotLifecycles(
    const SyncCoverGraph &graph, const SyncCoverCandidateIndex &index,
    const SyncCoverSlotLifecycleOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSLOTLIFECYCLE_H
