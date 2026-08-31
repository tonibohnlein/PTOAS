// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncCoverStorageRectangleGrounding.h - Cut coverage ----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGERECTANGLEGROUNDING_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGERECTANGLEGROUNDING_H

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverStorageRectangles.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

struct SyncCoverSyntheticRectangleGroundingDetail {
  SyncCoverStorageFactoredRectangleId rectangle = 0;
  SyncCoverStorageCutId completionCut = 0;
  SyncCoverStorageCutId acquisitionCut = 0;
  std::size_t coverageRows = 0;
};

struct SyncCoverSyntheticRectangleGroundingLimits {
  std::size_t maximumWorkUnits = 1U << 28;
  std::size_t maximumDetails = 64;
};

struct SyncCoverSyntheticRectangleGroundingStatistics {
  std::size_t workUnits = 0;
  std::size_t evaluatedSyntheticRectangles = 0;
  std::size_t syntheticRectanglesWithCoverage = 0;
  std::size_t syntheticRectanglesCoveringMultipleRows = 0;
  std::size_t maximumCoverageRows = 0;
  std::size_t totalCoverageRows = 0;
  bool detailsTruncated = false;
  bool truncated = false;
};

enum class SyncCoverSyntheticRectangleGroundingError : std::uint8_t {
  None,
  InvalidGraph,
  IncompleteRectangleIndex,
  InvalidLimit,
  WorkLimitExceeded,
  CoverageFailure,
  ArithmeticOverflow,
};

class SyncCoverSyntheticRectangleGrounding {
public:
  const SyncCoverSyntheticRectangleGroundingStatistics &
  getStatistics() const {
    return statistics_;
  }
  const std::vector<SyncCoverSyntheticRectangleGroundingDetail> &
  getDetails() const {
    return details_;
  }
  SyncCoverSyntheticRectangleGroundingError getError() const { return error_; }
  bool isComplete() const {
    return error_ == SyncCoverSyntheticRectangleGroundingError::None &&
           !statistics_.truncated;
  }

private:
  friend SyncCoverSyntheticRectangleGrounding
  groundSyncCoverSyntheticStorageRectangles(
      const SyncCoverGraph &, const SyncCoverExpandedProgram &,
      const SyncCoverStorageCutIndex &,
      const SyncCoverStorageFactoredRectangleIndex &,
      const std::vector<SyncCoverDemandId> &,
      const SyncCoverSyntheticRectangleGroundingLimits &);

  SyncCoverSyntheticRectangleGroundingStatistics statistics_;
  std::vector<SyncCoverSyntheticRectangleGroundingDetail> details_;
  SyncCoverSyntheticRectangleGroundingError error_ =
      SyncCoverSyntheticRectangleGroundingError::None;
};

/// Ground each synthetic compact event transfer independently through the
/// ordinary completion oracle. Exact direct rectangles already have frozen
/// direct-mechanism coverage. Results are diagnostic only and are streamed so
/// the analysis never retains a candidate-by-demand matrix.
SyncCoverSyntheticRectangleGrounding
groundSyncCoverSyntheticStorageRectangles(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    const SyncCoverStorageCutIndex &cutIndex,
    const SyncCoverStorageFactoredRectangleIndex &rectangleIndex,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const SyncCoverSyntheticRectangleGroundingLimits &limits = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGERECTANGLEGROUNDING_H
