// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_LIB_TRANSFORMS_CANONICALSYNC_SYNCCOVERGRAPHINTERNAL_H
#define PTO_LIB_TRANSFORMS_CANONICALSYNC_SYNCCOVERGRAPHINTERNAL_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include <algorithm>
#include <limits>

namespace mlir {
namespace pto {
namespace sync_cover_detail {

inline bool isValidEdgeKind(SyncCoverEdgeKind kind) {
  switch (kind) {
  case SyncCoverEdgeKind::CertifiedCompletionFrontier:
  case SyncCoverEdgeKind::CompletionPreservingIssueOrder:
  case SyncCoverEdgeKind::NonCompletionPreservingIssueOrder:
  case SyncCoverEdgeKind::CompletionSupply:
    return true;
  }
  return false;
}

inline bool isValidDemandKind(SyncCoverDemandKind kind) {
  switch (kind) {
  case SyncCoverDemandKind::SSA:
  case SyncCoverDemandKind::MemoryRAW:
  case SyncCoverDemandKind::MemoryWAR:
  case SyncCoverDemandKind::MemoryWAW:
    return true;
  }
  return false;
}

inline bool
isValidOrderingRequirements(SyncCoverOrderingRequirementMask requirements) {
  return requirements != 0 &&
         (requirements & ~kAllSyncCoverOrderingRequirements) == 0;
}

inline bool isValidAccessMode(SyncCoverStorageAccessMode mode) {
  switch (mode) {
  case SyncCoverStorageAccessMode::Read:
  case SyncCoverStorageAccessMode::Write:
  case SyncCoverStorageAccessMode::ReadWrite:
    return true;
  }
  return false;
}

inline bool isValidAccessPath(SyncCoverStorageAccessPath path) {
  switch (path) {
  case SyncCoverStorageAccessPath::Unknown:
  case SyncCoverStorageAccessPath::PhysicalPipeline:
  case SyncCoverStorageAccessPath::ScalarDCache:
    return true;
  }
  return false;
}

inline unsigned edgeStrength(SyncCoverEdgeKind kind) {
  switch (kind) {
  case SyncCoverEdgeKind::NonCompletionPreservingIssueOrder:
    return 0;
  case SyncCoverEdgeKind::CompletionPreservingIssueOrder:
    return 1;
  case SyncCoverEdgeKind::CertifiedCompletionFrontier:
    return 2;
  case SyncCoverEdgeKind::CompletionSupply:
    return 3;
  }
  return 0;
}

inline bool accessModesMatchDemand(SyncCoverDemandKind kind,
                                   SyncCoverStorageAccessMode source,
                                   SyncCoverStorageAccessMode target) {
  switch (kind) {
  case SyncCoverDemandKind::SSA:
    return false;
  case SyncCoverDemandKind::MemoryRAW:
    return syncCoverStorageModeWrites(source) &&
           syncCoverStorageModeReads(target);
  case SyncCoverDemandKind::MemoryWAR:
    return syncCoverStorageModeReads(source) &&
           syncCoverStorageModeWrites(target);
  case SyncCoverDemandKind::MemoryWAW:
    return syncCoverStorageModeWrites(source) &&
           syncCoverStorageModeWrites(target);
  }
  return false;
}

inline std::optional<SyncCoverTimelineInterval>
getNodeAnchorInterval(std::size_t order) {
  const bool overflows =
      order > (std::numeric_limits<std::size_t>::max() - 1) / 2;
  if (overflows) {
    return std::nullopt;
  }
  return SyncCoverTimelineInterval{order * 2, order * 2 + 1};
}

inline SyncCoverGraphError
validateDemandStorage(const SyncCoverGraph &graph,
                      const SyncCoverDemand &demand) {
  const bool hasMemoryKind =
      std::any_of(demand.provenanceKinds.begin(), demand.provenanceKinds.end(),
                  [](SyncCoverDemandKind kind) {
                    return kind != SyncCoverDemandKind::SSA;
                  });
  if (hasMemoryKind != !demand.storageWitnesses.empty()) {
    return SyncCoverGraphError::InvalidStorageProvenance;
  }
  const auto &witnesses = graph.getStorageWitnesses();
  const auto &accesses = graph.getStorageAccesses();
  for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
    if (witnessId >= witnesses.size()) {
      return SyncCoverGraphError::InvalidStorageWitness;
    }
    const SyncCoverStorageWitness &witness = witnesses[witnessId];
    const bool invalidAccess = witness.sourceAccess >= accesses.size() ||
                               witness.targetAccess >= accesses.size();
    if (invalidAccess) {
      return SyncCoverGraphError::InvalidStorageWitness;
    }
    const SyncCoverStorageAccess &source = accesses[witness.sourceAccess];
    const SyncCoverStorageAccess &target = accesses[witness.targetAccess];
    const bool matchingKind = std::any_of(
        demand.provenanceKinds.begin(), demand.provenanceKinds.end(),
        [&](SyncCoverDemandKind kind) {
          return accessModesMatchDemand(kind, source.mode, target.mode);
        });
    const bool wrongEndpoints =
        source.node != demand.source || target.node != demand.target;
    if (wrongEndpoints || !matchingKind) {
      return SyncCoverGraphError::InvalidStorageProvenance;
    }
  }
  for (SyncCoverDemandKind kind : demand.provenanceKinds) {
    if (kind == SyncCoverDemandKind::SSA) {
      continue;
    }
    const bool kindHasWitness = std::any_of(
        demand.storageWitnesses.begin(), demand.storageWitnesses.end(),
        [&](SyncCoverStorageWitnessId witnessId) {
          const SyncCoverStorageWitness &witness = witnesses[witnessId];
          return accessModesMatchDemand(kind,
                                        accesses[witness.sourceAccess].mode,
                                        accesses[witness.targetAccess].mode);
        });
    if (!kindHasWitness) {
      return SyncCoverGraphError::InvalidStorageProvenance;
    }
  }
  return SyncCoverGraphError::None;
}

} // namespace sync_cover_detail
} // namespace pto
} // namespace mlir

#endif // PTO_LIB_TRANSFORMS_CANONICALSYNC_SYNCCOVERGRAPHINTERNAL_H
