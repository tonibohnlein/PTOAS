// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVERINTERNAL_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVERINTERNAL_H

#include "PTO/Transforms/CanonicalSync/SyncCoverSolver.h"

#include <map>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {
namespace sync_cover_internal {

struct CoverageEvaluation {
  bool valid = true;
  std::vector<SyncCoverDemandId> uncovered;
  std::map<SyncCoverDemandId, std::vector<SyncCoverMechanismId>> cuts;
  std::map<SyncCoverDemandId, std::vector<SyncCoverReachableState>>
      reachableStates;
};

class CoverageEvaluator {
public:
  explicit CoverageEvaluator(SyncCoverCoverageOracle &oracle)
      : oracle_(oracle) {}

  CoverageEvaluation
  evaluate(const std::vector<SyncCoverDemandId> &demands,
           const std::vector<SyncCoverMechanismId> &selected);

private:
  SyncCoverCoverageOracle &oracle_;
  std::map<SyncCoverDemandId, std::vector<std::vector<SyncCoverMechanismId>>>
      witnesses_;
};

struct ComponentBuildResult {
  bool valid = true;
  std::vector<SyncCoverSelectionComponent> components;
};

struct ComponentSearchResult {
  std::optional<std::vector<SyncCoverMechanismId>> selected;
  std::optional<SyncCoverStructuralCost> cost;
  std::size_t evaluations = 0;
  SyncCoverSearchTruncation truncation;
  bool optimalityProven = false;
};

struct AffectedSliceExchangeResult {
  std::vector<SyncCoverMechanismId> selected;
  SyncCoverStructuralCost cost;
  SyncCoverExchangeStatistics statistics;
  bool candidateLimitReached = false;
  bool evaluationLimitReached = false;
  bool roundLimitReached = false;
};

ComponentBuildResult
buildComponents(const SyncCoverMechanismUniverse &universe,
                const std::vector<SyncCoverDemandId> &activeDemands,
                SyncCoverCoverageOracle &oracle, std::size_t exactThreshold);

std::vector<std::vector<SyncCoverMechanismId>>
getComponentSeeds(const SyncCoverSelectionComponent &component,
                  const std::vector<SyncCoverSelectionSeed> &seeds);

ComponentSearchResult searchExact(
    const SyncCoverSelectionEvaluator &selectionEvaluator,
    CoverageEvaluator &coverage, const SyncCoverSelectionComponent &component,
    const std::vector<std::vector<SyncCoverMechanismId>> &seedSelections,
    const SyncCoverSolverOptions &options);

ComponentSearchResult
searchBeam(const SyncCoverSelectionEvaluator &selectionEvaluator,
           CoverageEvaluator &coverage,
           const SyncCoverSelectionComponent &component,
           const std::vector<std::vector<SyncCoverMechanismId>> &seedSelections,
           const SyncCoverSolverOptions &options);

bool evaluateCompleteSelection(
    const SyncCoverSelectionEvaluator &selectionEvaluator,
    CoverageEvaluator &coverage,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &selected,
    SyncCoverStructuralCost &cost);

void removeRedundantMechanisms(
    const SyncCoverSelectionEvaluator &selectionEvaluator,
    CoverageEvaluator &coverage,
    const std::vector<SyncCoverDemandId> &activeDemands,
    std::vector<SyncCoverMechanismId> &selected, SyncCoverStructuralCost &cost,
    std::size_t &evaluations);

bool independentlyVerifySelection(
    const SyncCoverMechanismUniverse &universe,
    SyncCoverCoverageOracle &coverage,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &selected,
    SyncCoverStructuralCost &cost, SyncCoverResourceSelection &resources,
    SyncCoverCoverageStatistics &statistics);

AffectedSliceExchangeResult improveByAffectedSliceExchange(
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverSelectionEvaluator &selectionEvaluator,
    SyncCoverCoverageOracle &oracle,
    CoverageEvaluator &coverage,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &incumbent,
    const SyncCoverStructuralCost &incumbentCost,
    const SyncCoverSolverOptions &options,
    std::size_t &redundancyEvaluations);

} // namespace sync_cover_internal
} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVERINTERNAL_H
