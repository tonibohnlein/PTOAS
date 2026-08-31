// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncCoverStorageCuts.h - Exact storage cut index -------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGECUTS_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGECUTS_H

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageLifecycle.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverStorageCutId = std::size_t;
using SyncCoverStorageRectangleId = std::size_t;

struct SyncCoverStorageLifecycleEdgeRef {
  SyncCoverStorageLifecycleComponentId component = 0;
  SyncCoverStorageLifecycleEdgeId edge = 0;
};

enum class SyncCoverStorageCutKind : std::uint8_t {
  Completion,
  Acquisition,
};

/// A direct, target-authorized event frontier shared by exact-storage
/// obligations. This is a semantic cut candidate, not yet a selected recipe.
struct SyncCoverStorageCut {
  SyncCoverStorageCutId id = 0;
  SyncCoverStorageCutKind kind = SyncCoverStorageCutKind::Completion;
  SyncCoverAnchor anchor;
  std::uint32_t resource = 0;
  std::uint32_t peerResource = 0;
  SyncCoverScopeId scope = 0;
  SyncCoverGuard guard;
  std::vector<SyncCoverStorageLifecycleEdgeRef> edges;
};

/// Compact incidence between one completion cut and one acquisition cut.
/// Coverage remains represented by original lifecycle edges rather than by a
/// materialized producer-by-consumer Cartesian product.
struct SyncCoverStorageRectangle {
  SyncCoverStorageRectangleId id = 0;
  SyncCoverStorageCutId completionCut = 0;
  SyncCoverStorageCutId acquisitionCut = 0;
  SyncCoverStorageLifecycleEdgeKindMask kinds = 0;
  std::vector<SyncCoverStorageLifecycleEdgeRef> edges;
};

struct SyncCoverStorageCutLimits {
  std::size_t maximumWorkUnits = 1U << 22;
  std::size_t maximumCuts = 1U << 20;
  std::size_t maximumRectangles = 1U << 20;
  std::size_t maximumIncidences = 1U << 22;
  std::size_t maximumGuardLiterals = 1U << 20;
};

struct SyncCoverStorageCutStatistics {
  std::size_t workUnits = 0;
  std::size_t eligibleEdges = 0;
  std::size_t ineligibleEdges = 0;
  std::size_t completionCuts = 0;
  std::size_t acquisitionCuts = 0;
  std::size_t rectangles = 0;
  std::size_t incidences = 0;
  std::size_t guardLiterals = 0;
  std::size_t maximumRectangleEdges = 0;
  bool truncated = false;
};

enum class SyncCoverStorageCutError : std::uint8_t {
  None,
  InvalidGraph,
  IncompleteLifecycleIndex,
  InvalidLimit,
  LimitExceeded,
  ArithmeticOverflow,
};

class SyncCoverStorageCutIndex {
public:
  const std::vector<SyncCoverStorageCut> &getCuts() const { return cuts_; }
  const std::vector<SyncCoverStorageRectangle> &getRectangles() const {
    return rectangles_;
  }
  const SyncCoverStorageCutStatistics &getStatistics() const {
    return statistics_;
  }
  SyncCoverStorageCutError getError() const { return error_; }
  bool isComplete() const {
    return error_ == SyncCoverStorageCutError::None && !statistics_.truncated;
  }

private:
  friend SyncCoverStorageCutIndex buildSyncCoverStorageCutIndex(
      const SyncCoverGraph &, const SyncCoverStorageLifecycleIndex &,
      const SyncCoverStorageCutLimits &);

  std::vector<SyncCoverStorageCut> cuts_;
  std::vector<SyncCoverStorageRectangle> rectangles_;
  SyncCoverStorageCutStatistics statistics_;
  SyncCoverStorageCutError error_ = SyncCoverStorageCutError::None;
};

/// Enumerate exact distance-zero source and target event cuts. A bound failure
/// is transactional and publishes no partial cut or rectangle catalog.
SyncCoverStorageCutIndex buildSyncCoverStorageCutIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageCutLimits &limits = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGECUTS_H
