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

using SyncCoverComponentId = std::size_t;

struct SyncCoverSelectionComponent {
  SyncCoverComponentId id = 0;
  std::vector<SyncCoverDemandId> demands;
  std::vector<SyncCoverGroundedColumnId> columns;
  std::vector<SyncCoverMechanismId> mechanisms;
  bool exact = false;
};

struct SyncCoverSolverOptions {
  static constexpr std::size_t maximumExactMechanismThreshold = 24;

  std::size_t exactMechanismThreshold = 12;
  /// Per-component bounded-state evaluation limit for exact and greedy
  /// search. Truncation falls back to all-barrier coverage.
  std::size_t evaluationLimit = 512;
  /// Upper bound on post-search oracle-checked redundancy deletions. The
  /// search itself never queries the oracle; this bounds the one polish pass
  /// over the final selection so its cost stays proportional to the emitted
  /// plan, not to the candidate universe.
  std::size_t oracleRedundancyLimit = 32;
  /// Lazy composition pricing runs only for demands whose every selected
  /// cover uses a barrier: at most this many demands are priced (deepest
  /// loops first), trying at most pricingCoverLimit priced covers each.
  std::size_t pricingDemandLimit = 32;
  std::size_t pricingCoverLimit = 4;
  /// Candidate pair budget handed to the frontier witness search per priced
  /// demand.
  std::size_t pricingPairLimit = 32;
};

enum class SyncCoverSelectionError : std::uint8_t {
  None,
  InvalidUniverse,
  InvalidDemand,
  InvalidOptions,
  SearchIncomplete,
  FinalVerificationFailed,
  GroundingFailed,
};

struct SyncCoverSearchTruncation {
  bool evaluationLimit = false;

  explicit operator bool() const { return evaluationLimit; }
};

struct SyncCoverSelectionResult {
  SyncCoverSelectionError error = SyncCoverSelectionError::None;
  std::vector<SyncCoverMechanismId> mechanisms;
  SyncCoverStructuralCost cost;
  std::vector<SyncCoverSelectionComponent> components;
  std::size_t evaluations = 0;
  std::size_t redundancyEvaluations = 0;
  /// Whole-demand oracle checks used by the bounded post-search polish.
  std::size_t oracleRedundancyChecks = 0;
  /// Components whose truncated search was rescued by the all-barrier
  /// fallback instead of a searched selection.
  std::size_t rescuedComponents = 0;
  /// Demands priced lazily because their selected cover used barriers, and
  /// how many of those pricings improved the plan.
  std::size_t pricedDemands = 0;
  std::size_t pricedImprovements = 0;
  SyncCoverSearchTruncation truncation;
  bool optimalityProven = false;
  std::vector<SyncCoverDemandId> missingFactoryDemands;
  std::vector<SyncCoverDemandId> demandsWithoutEventColumn;
  std::optional<SyncCoverComponentId> failedComponent;
  std::optional<SyncCoverDemandId> failedFinalDemand;
  SyncCoverResourceSelection resources;
  SyncCoverCoverageStatistics coverageStatistics;
  SyncCoverCoverageStatistics finalVerificationStatistics;

  explicit operator bool() const {
    return error == SyncCoverSelectionError::None;
  }
};

/// Select an atomic synchronization cover for the active immutable demands.
/// Components conservatively include conflict and shared-resource coupling.
/// Exact search is column-guided; larger components use deterministic greedy
/// selection and local deletion. Every returned result passes fresh graph,
/// protocol, conflict, coloring, and structural-cost validation plus one exact
/// coverage traversal over the immutable prepared demand topology.
SyncCoverSelectionResult
solveSyncCoverSelection(const SyncCoverMechanismUniverse &universe,
                        const std::vector<SyncCoverDemandId> &activeDemands,
                        const SyncCoverSolverOptions &options = {},
                        const std::vector<SyncCoverVerifiedFactoryColumn>
                            &factoryColumns = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVER_H
