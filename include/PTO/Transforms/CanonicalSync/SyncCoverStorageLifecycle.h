// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncCoverStorageLifecycle.h - Exact storage lifecycles --*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGELIFECYCLE_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGELIFECYCLE_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverStorageLifecycleComponentId = std::size_t;
using SyncCoverStorageLifecycleSlotId = std::size_t;
using SyncCoverStorageLifecycleEpochId = std::size_t;
using SyncCoverStorageLifecycleEdgeId = std::size_t;
using SyncCoverStorageLifecycleSccId = std::size_t;

enum class SyncCoverStorageLifecycleEdgeKind : std::uint8_t {
  Ready = 1,
  Release = 2,
  Exclusion = 4,
};

using SyncCoverStorageLifecycleEdgeKindMask = std::uint8_t;

constexpr SyncCoverStorageLifecycleEdgeKindMask
syncCoverStorageLifecycleEdgeKindBit(SyncCoverStorageLifecycleEdgeKind kind) {
  return static_cast<SyncCoverStorageLifecycleEdgeKindMask>(kind);
}

struct SyncCoverStorageLifecycleSlot {
  SyncCoverStorageLifecycleSlotId id = 0;
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageAccessFamilyId family = 0;
  SyncCoverStorageInterval extent;
  std::vector<SyncCoverStorageAccessId> accesses;
};

/// One exact physical access occurrence in one lifecycle owner. An access can
/// appear in more than one component only when distinct enclosing scopes own
/// independent hazards involving the same physical slot.
struct SyncCoverStorageLifecycleEpoch {
  SyncCoverStorageLifecycleEpochId id = 0;
  SyncCoverStorageAccessId access = 0;
  SyncCoverStorageLifecycleSlotId slot = 0;
  SyncCoverNodeId node = 0;
  SyncCoverStorageAccessMode mode = SyncCoverStorageAccessMode::Read;
  std::uint32_t resource = 0;
  SyncCoverScopeId scope = 0;
};

/// Original semantic obligation retained as a storage-local lifecycle edge.
/// A canonical demand can carry more than one compatible hazard kind.
struct SyncCoverStorageLifecycleEdge {
  SyncCoverStorageLifecycleEdgeId id = 0;
  SyncCoverDemandId demand = 0;
  SyncCoverStorageWitnessId witness = 0;
  SyncCoverStorageLifecycleEpochId source = 0;
  SyncCoverStorageLifecycleEpochId target = 0;
  SyncCoverStorageLifecycleEdgeKindMask kinds = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
};

/// One strongly connected storage-lifecycle region. A cyclic SCC containing
/// both ready and release edges is the neutral input to later protocol
/// synthesis; it is not itself evidence that a physical recipe is legal.
struct SyncCoverStorageLifecycleScc {
  SyncCoverStorageLifecycleSccId id = 0;
  std::vector<SyncCoverStorageLifecycleEpochId> epochs;
  std::vector<SyncCoverStorageLifecycleEdgeId> internalEdges;
  SyncCoverStorageLifecycleEdgeKindMask kinds = 0;
  unsigned maximumDistance = 0;
  bool cyclic = false;
};

/// One obligation crossing between SCCs in the component condensation DAG.
struct SyncCoverStorageLifecycleSccTransfer {
  SyncCoverStorageLifecycleEdgeId edge = 0;
  SyncCoverStorageLifecycleSccId source = 0;
  SyncCoverStorageLifecycleSccId target = 0;
};

/// Target-neutral exact-storage component. The family is the frontend storage
/// root; multiple exact slots from that root can participate in one component.
/// Target recipe policy and ownership vocabulary are deliberately absent.
struct SyncCoverStorageLifecycleComponent {
  SyncCoverStorageLifecycleComponentId id = 0;
  SyncCoverStorageAccessFamilyId family = 0;
  /// Nearest loop owning all represented obligations, or root scope zero.
  SyncCoverScopeId owningScope = 0;
  std::vector<SyncCoverStorageLifecycleSlot> slots;
  std::vector<SyncCoverStorageLifecycleEpoch> epochs;
  std::vector<SyncCoverStorageLifecycleEdge> edges;
  std::vector<SyncCoverDemandId> demands;
  std::vector<SyncCoverStorageLifecycleScc> sccs;
  std::vector<SyncCoverStorageLifecycleSccId> epochSccs;
  std::vector<SyncCoverStorageLifecycleSccTransfer> sccTransfers;
};

struct SyncCoverStorageLifecycleLimits {
  /// Shared upper bound for demand/witness visits, provenance scans,
  /// scope-ancestry queries, ordered-container operations, and publication.
  std::size_t maximumWorkUnits = 1U << 22;
  std::size_t maximumComponents = 1U << 14;
  std::size_t maximumSlots = 1U << 16;
  std::size_t maximumEpochs = 1U << 20;
  std::size_t maximumEdges = 1U << 20;
  std::size_t maximumDemandIncidences = 1U << 20;
  std::size_t maximumSccs = 1U << 20;
};

struct SyncCoverStorageLifecycleStatistics {
  std::size_t workUnits = 0;
  std::size_t eligibleWitnesses = 0;
  std::size_t ineligibleWitnesses = 0;
  std::size_t components = 0;
  std::size_t slots = 0;
  std::size_t epochs = 0;
  std::size_t edges = 0;
  std::size_t demandIncidences = 0;
  std::size_t sccs = 0;
  std::size_t cyclicSccs = 0;
  std::size_t readyReleaseSccs = 0;
  std::size_t sccTransfers = 0;
  std::size_t maximumSccEpochs = 0;
  bool truncated = false;
};

enum class SyncCoverStorageLifecycleError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidLimit,
  LimitExceeded,
  ArithmeticOverflow,
};

class SyncCoverStorageLifecycleIndex {
public:
  const std::vector<SyncCoverStorageLifecycleComponent> &getComponents() const {
    return components_;
  }
  const SyncCoverStorageLifecycleStatistics &getStatistics() const {
    return statistics_;
  }
  SyncCoverStorageLifecycleError getError() const { return error_; }
  bool isComplete() const {
    return error_ == SyncCoverStorageLifecycleError::None &&
           !statistics_.truncated;
  }

private:
  friend SyncCoverStorageLifecycleIndex
  buildSyncCoverStorageLifecycleIndex(const SyncCoverGraph &,
                                      const SyncCoverStorageLifecycleLimits &);

  std::vector<SyncCoverStorageLifecycleComponent> components_;
  SyncCoverStorageLifecycleStatistics statistics_;
  SyncCoverStorageLifecycleError error_ = SyncCoverStorageLifecycleError::None;
};

/// Build a deterministic index from exact, whole-slot memory witnesses. A
/// bound failure is transactional: statistics record truncation, but no
/// partial component is retained for later synthesis.
SyncCoverStorageLifecycleIndex buildSyncCoverStorageLifecycleIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleLimits &limits = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGELIFECYCLE_H
