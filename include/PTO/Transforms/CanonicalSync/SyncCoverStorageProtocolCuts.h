// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverStorageProtocolCuts.h - Protocol cut plans ----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLCUTS_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLCUTS_H

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageCuts.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolAutomata.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverStorageProtocolCutPlanId = std::size_t;

/// Target-neutral association between one lifecycle automaton and the exact
/// direct rectangles that can implement its distance-zero ready transfers.
/// This remains a proposal: it does not prove token balance, select a recipe,
/// establish coverage, or create a set-cover column.
struct SyncCoverStorageProtocolCutPlan {
  SyncCoverStorageProtocolCutPlanId id = 0;
  SyncCoverStorageProtocolAutomatonId automaton = 0;
  SyncCoverStorageProtocolGroupId group = 0;
  SyncCoverScopeId owningScope = 0;
  std::size_t laneCount = 0;
  std::size_t directReadyTransfers = 0;
  std::size_t recurrenceReleaseTransfers = 0;
  std::vector<SyncCoverStorageRectangleId> readyRectangles;
};

struct SyncCoverStorageProtocolCutPlanLimits {
  std::size_t maximumWorkUnits = 1U << 24;
  std::size_t maximumPlans = 1U << 14;
  std::size_t maximumRectangleEdgeIncidences = 1U << 20;
  std::size_t maximumTransferInspections = 1U << 20;
  std::size_t maximumReadyRectangleIncidences = 1U << 20;
};

struct SyncCoverStorageProtocolCutPlanStatistics {
  std::size_t workUnits = 0;
  std::size_t eligibleAutomata = 0;
  std::size_t ineligibleAutomata = 0;
  std::size_t missingReadyCutAutomata = 0;
  std::size_t missingReleaseAutomata = 0;
  std::size_t plans = 0;
  std::size_t transferInspections = 0;
  std::size_t directReadyTransfers = 0;
  std::size_t recurrenceReleaseTransfers = 0;
  std::size_t readyRectangleIncidences = 0;
  std::size_t maximumPlanReadyRectangles = 0;
  bool truncated = false;
};

enum class SyncCoverStorageProtocolCutPlanError : std::uint8_t {
  None,
  InvalidGraph,
  IncompleteLifecycleIndex,
  IncompleteAutomatonIndex,
  IncompleteCutIndex,
  InvalidLimit,
  LimitExceeded,
  ArithmeticOverflow,
};

class SyncCoverStorageProtocolCutPlanIndex {
public:
  const std::vector<SyncCoverStorageProtocolCutPlan> &getPlans() const {
    return plans_;
  }
  const SyncCoverStorageProtocolCutPlanStatistics &getStatistics() const {
    return statistics_;
  }
  SyncCoverStorageProtocolCutPlanError getError() const { return error_; }
  bool isComplete() const {
    return error_ == SyncCoverStorageProtocolCutPlanError::None &&
           !statistics_.truncated;
  }

private:
  friend SyncCoverStorageProtocolCutPlanIndex
  buildSyncCoverStorageProtocolCutPlanIndex(
      const SyncCoverGraph &, const SyncCoverStorageLifecycleIndex &,
      const SyncCoverStorageProtocolAutomatonIndex &,
      const SyncCoverStorageCutIndex &,
      const SyncCoverStorageProtocolCutPlanLimits &);

  std::vector<SyncCoverStorageProtocolCutPlan> plans_;
  SyncCoverStorageProtocolCutPlanStatistics statistics_;
  SyncCoverStorageProtocolCutPlanError error_ =
      SyncCoverStorageProtocolCutPlanError::None;
};

/// Attach direct ready transfers to their exact source/target cut rectangles.
/// Missing cuts or release transfers reject only that automaton proposal.
/// Structural or resource-limit failure publishes no partial plan index.
SyncCoverStorageProtocolCutPlanIndex buildSyncCoverStorageProtocolCutPlanIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolAutomatonIndex &automatonIndex,
    const SyncCoverStorageCutIndex &cutIndex,
    const SyncCoverStorageProtocolCutPlanLimits &limits = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLCUTS_H
