// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverSolver.h - Direct synchronization covering ------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVER_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVER_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGrounded.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

struct SyncCoverSolverOptions {
  /// Upper bound on post-search oracle-checked redundancy deletions. The
  /// search itself never queries the oracle; this bounds the one polish pass
  /// over the final selection so its cost stays proportional to the emitted
  /// plan, not to the candidate universe.
  std::size_t oracleRedundancyLimit = 32;
};

enum class SyncCoverSelectionError : std::uint8_t {
  None,
  InvalidUniverse,
  InvalidDemand,
  SearchIncomplete,
  FinalVerificationFailed,
  GroundingFailed,
};

struct SyncCoverSelectionResult {
  SyncCoverSelectionError error = SyncCoverSelectionError::None;
  std::vector<SyncCoverMechanismId> mechanisms;
  SyncCoverStructuralCost cost;
  std::size_t evaluations = 0;
  std::size_t redundancyEvaluations = 0;
  /// Whole-demand oracle checks used by the bounded post-search polish.
  std::size_t oracleRedundancyChecks = 0;
  std::vector<SyncCoverDemandId> missingFactoryDemands;
  std::vector<SyncCoverDemandId> demandsWithoutEventColumn;
  std::optional<SyncCoverDemandId> failedFinalDemand;
  SyncCoverResourceSelection resources;
  SyncCoverCoverageStatistics coverageStatistics;
  SyncCoverCoverageStatistics finalVerificationStatistics;

  explicit operator bool() const {
    return error == SyncCoverSelectionError::None;
  }
};

/// Select an atomic synchronization cover for the active immutable demands.
/// Selection is one deterministic greedy pass over the grounded coverage
/// bitsets, polished by single-deletion and grounded-column exchange. The
/// selector is untrusted: every returned result passes fresh graph,
/// protocol, conflict, coloring, and structural-cost validation plus one
/// exact coverage traversal over the immutable prepared demand topology.
SyncCoverSelectionResult
solveSyncCoverSelection(const SyncCoverMechanismUniverse &universe,
                        const std::vector<SyncCoverDemandId> &activeDemands,
                        const SyncCoverSolverOptions &options = {},
                        const std::vector<SyncCoverVerifiedFactoryColumn>
                            &factoryColumns = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVER_H
