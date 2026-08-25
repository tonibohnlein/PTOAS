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

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverMechanism.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverComponentId = std::size_t;

struct SyncCoverSelectionSeed {
  std::uint64_t identity = 0;
  std::vector<SyncCoverMechanismId> mechanisms;
};

struct SyncCoverSelectionComponent {
  SyncCoverComponentId id = 0;
  std::vector<SyncCoverDemandId> demands;
  std::vector<SyncCoverMechanismId> mechanisms;
  bool exact = false;
};

struct SyncCoverSolverOptions {
  static constexpr std::size_t maximumExactMechanismThreshold = 24;

  std::size_t exactMechanismThreshold = 18;
  std::size_t beamWidth = 16;
  std::size_t beamDepth = 64;
  /// Per-component bounded-state evaluation limit for exact and beam search.
  /// Seed evaluation is outside this limit so a valid incumbent survives
  /// truncation.
  std::size_t evaluationLimit = 4096;
};

enum class SyncCoverSelectionError : std::uint8_t {
  None,
  InvalidUniverse,
  InvalidDemand,
  InvalidSeed,
  InvalidOptions,
  ProvenInfeasible,
  SearchIncomplete,
  FinalVerificationFailed,
};

struct SyncCoverSearchTruncation {
  bool beamWidth = false;
  bool beamDepth = false;
  bool evaluationLimit = false;

  explicit operator bool() const {
    return beamWidth || beamDepth || evaluationLimit;
  }
};

struct SyncCoverSelectionResult {
  SyncCoverSelectionError error = SyncCoverSelectionError::None;
  std::vector<SyncCoverMechanismId> mechanisms;
  SyncCoverStructuralCost cost;
  std::vector<SyncCoverSelectionComponent> components;
  std::size_t evaluations = 0;
  std::size_t redundancyEvaluations = 0;
  SyncCoverSearchTruncation truncation;
  bool optimalityProven = false;
  SyncCoverResourceSelection resources;
  SyncCoverCoverageStatistics coverageStatistics;
  SyncCoverCoverageStatistics finalVerificationStatistics;

  explicit operator bool() const {
    return error == SyncCoverSelectionError::None;
  }
};

/// Select an atomic synchronization cover for the active immutable demands.
/// Components conservatively include conflict and shared-resource coupling.
/// Exact search is cut-guided; larger components use a bounded deterministic
/// beam. Every returned result passes fresh graph, protocol, conflict,
/// coloring, and structural-cost validation plus a second exact coverage
/// traversal over the immutable prepared demand topology.
SyncCoverSelectionResult
solveSyncCoverSelection(const SyncCoverMechanismUniverse &universe,
                        const std::vector<SyncCoverDemandId> &activeDemands,
                        const std::vector<SyncCoverSelectionSeed> &seeds = {},
                        const SyncCoverSolverOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVER_H
