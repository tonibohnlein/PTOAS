// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CanonicalSync.h - Bounded pattern synchronization ------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAnalysis.h"
#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace mlir {
namespace pto {

/// Internal ablation controls for the restricted direct-mechanism catalog.
/// Scarcity frontiers are grounded only for bounded repair and are excluded
/// from ordinary precise selection.
struct CanonicalSyncPatternOptions {
  bool enableDirectPairs = true;
  bool enableScarcityFrontiers = true;
};

struct CanonicalSyncStrategyReport {
  CanonicalSyncSelectionStrategy strategy =
      CanonicalSyncSelectionStrategy::PairLookahead;
  CanonicalSyncSelectionError error = CanonicalSyncSelectionError::None;
  bool verified = false;
  bool usedLocalizedPipeAll = false;
  std::size_t repairRounds = 0;
  std::size_t selectedEvents = 0;
  std::size_t selectedTargetedBarriers = 0;
  std::size_t selectedPipeAllBarriers = 0;
  CanonicalSyncStructuralCost cost;
  CanonicalSyncGreedyStatistics search;
  CanonicalSyncResourceAllocation allocation;
};

struct CanonicalSyncComparisonReport {
  std::size_t demands = 0;
  std::size_t directMechanisms = 0;
  std::size_t directPairProposals = 0;
  std::size_t directPairEvaluations = 0;
  std::size_t synergisticPairs = 0;
  bool pairGenerationTruncated = false;
  std::vector<CanonicalSyncStrategyReport> strategies;
};

struct CanonicalSyncBuildOptions {
  unsigned eventIdBudget = 8;
  CanonicalSyncAnalysisOptions analysis;
  CanonicalSyncPatternOptions patterns;
  CanonicalSyncPatternProblem::Limits problemLimits;
  SyncCoverExpansionLimits expansionLimits;
  CanonicalSyncGreedyOptions selection;
  std::size_t maximumRepairRounds = 8;
  bool analysisOnly = false;
  bool compareSelectionStrategies = false;
  std::function<LogicalResult(const CanonicalSyncComparisonReport &)>
      reportCallback;
};

/// Build and freeze the singleton candidate problem. The returned problem
/// retains a non-owning reference to program and must not outlive or move it.
FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>>
buildCanonicalSyncSingletonProblem(const CanonicalSyncProgram &program,
                                   const CanonicalSyncBuildOptions &options);

/// Validate every concrete anchor and allocation before modifying the IR, then
/// emit each selected atomic recipe exactly once.
LogicalResult
materializeCanonicalSyncPlan(const CanonicalSyncProgram &program,
                             const CanonicalSyncPatternProblem &problem,
                             const CanonicalSyncVerifiedPlan &plan);

/// Run analysis, bounded-pattern selection, bitset-based finalization, and
/// materialization while all referenced graph storage remains alive.
LogicalResult runCanonicalSync(func::FuncOp function,
                               const CanonicalSyncBuildOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
