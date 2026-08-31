// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncCoverStorageRectangles.h - Factored event cuts -----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGERECTANGLES_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGERECTANGLES_H

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageCuts.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverStorageFactoredRectangleId = std::size_t;

/// One balanced, distance-zero event transfer between two certified cuts.
/// Coverage is deliberately not stored here: it is grounded later by the
/// ordinary completion oracle over this compact supply edge.
struct SyncCoverStorageFactoredRectangle {
  SyncCoverStorageFactoredRectangleId id = 0;
  SyncCoverStorageCutId completionCut = 0;
  SyncCoverStorageCutId acquisitionCut = 0;
  SyncCoverScopeId scope = 0;
  SyncCoverGuard guard;
  std::optional<SyncCoverStorageRectangleId> directRectangle;
};

struct SyncCoverStorageFactoredRectangleLimits {
  std::size_t maximumWorkUnits = 1U << 24;
  std::size_t maximumInspections = 1U << 20;
  std::size_t maximumRectangles = 1U << 14;
  std::size_t maximumGuardLiterals = 1U << 20;
};

struct SyncCoverStorageFactoredRectangleStatistics {
  std::size_t workUnits = 0;
  std::size_t inspections = 0;
  std::size_t rectangles = 0;
  std::size_t directRectangles = 0;
  std::size_t syntheticRectangles = 0;
  std::size_t guardLiterals = 0;
  bool truncated = false;
};

enum class SyncCoverStorageFactoredRectangleError : std::uint8_t {
  None,
  InvalidGraph,
  IncompleteCutIndex,
  InvalidLimit,
  LimitExceeded,
  ArithmeticOverflow,
};

class SyncCoverStorageFactoredRectangleIndex {
public:
  const std::vector<SyncCoverStorageFactoredRectangle> &getRectangles() const {
    return rectangles_;
  }
  const SyncCoverStorageFactoredRectangleStatistics &getStatistics() const {
    return statistics_;
  }
  SyncCoverStorageFactoredRectangleError getError() const { return error_; }
  bool isComplete() const {
    return error_ == SyncCoverStorageFactoredRectangleError::None &&
           !statistics_.truncated;
  }

private:
  friend SyncCoverStorageFactoredRectangleIndex
  buildSyncCoverStorageFactoredRectangleIndex(
      const SyncCoverGraph &, const SyncCoverStorageCutIndex &,
      const SyncCoverStorageFactoredRectangleLimits &);

  std::vector<SyncCoverStorageFactoredRectangle> rectangles_;
  SyncCoverStorageFactoredRectangleStatistics statistics_;
  SyncCoverStorageFactoredRectangleError error_ =
      SyncCoverStorageFactoredRectangleError::None;
};

/// Enumerate balanced event transfers at existing schedule change points.
/// A bound failure is transactional and publishes no partial rectangle list.
SyncCoverStorageFactoredRectangleIndex
buildSyncCoverStorageFactoredRectangleIndex(
    const SyncCoverGraph &graph, const SyncCoverStorageCutIndex &cutIndex,
    const SyncCoverStorageFactoredRectangleLimits &limits = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGERECTANGLES_H
