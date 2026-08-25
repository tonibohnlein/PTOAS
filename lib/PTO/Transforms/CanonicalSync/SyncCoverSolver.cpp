// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "SyncCoverSolverInternal.h"

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_internal;

namespace {

bool normalizeUnique(std::vector<std::size_t> &values) {
  std::sort(values.begin(), values.end());
  return std::adjacent_find(values.begin(), values.end()) == values.end();
}

SyncCoverSelectionResult makeError(SyncCoverSelectionError error) {
  SyncCoverSelectionResult result;
  result.error = error;
  return result;
}

} // namespace

SyncCoverSelectionResult mlir::pto::solveSyncCoverSelection(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverSelectionSeed> &inputSeeds,
    const SyncCoverSolverOptions &options) {
  const SyncCoverSelectionEvaluator selectionEvaluator(universe);
  if (!selectionEvaluator) {
    return makeError(SyncCoverSelectionError::InvalidUniverse);
  }
  const bool invalidOptions =
      options.beamWidth == 0 || options.beamDepth == 0 ||
      options.evaluationLimit == 0 ||
      options.exactMechanismThreshold >
          SyncCoverSolverOptions::maximumExactMechanismThreshold;
  if (invalidOptions) {
    return makeError(SyncCoverSelectionError::InvalidOptions);
  }

  std::vector<SyncCoverDemandId> demands = activeDemands;
  const bool invalidDemands =
      !normalizeUnique(demands) ||
      (!demands.empty() &&
       demands.back() >= universe.getGraph().getDemands().size());
  if (invalidDemands) {
    return makeError(SyncCoverSelectionError::InvalidDemand);
  }

  std::vector<SyncCoverSelectionSeed> seeds = inputSeeds;
  for (SyncCoverSelectionSeed &seed : seeds) {
    const bool invalidIdentity =
        !normalizeUnique(seed.mechanisms) ||
        (!seed.mechanisms.empty() &&
         seed.mechanisms.back() >= universe.getMechanisms().size());
    if (invalidIdentity) {
      return makeError(SyncCoverSelectionError::InvalidSeed);
    }
    const SyncCoverResourceSelection seedResources =
        selectionEvaluator.evaluate(seed.mechanisms).resources;
    if (!seedResources.isValid()) {
      return makeError(SyncCoverSelectionError::InvalidSeed);
    }
  }
  std::sort(seeds.begin(), seeds.end(),
            [](const auto &first, const auto &second) {
              return std::tie(first.identity, first.mechanisms) <
                     std::tie(second.identity, second.mechanisms);
            });

  SyncCoverCoverageOracle oracle(universe.getGraph());
  CoverageEvaluator coverage(oracle);
  const ComponentBuildResult decomposition = buildComponents(
      universe, demands, oracle, options.exactMechanismThreshold);
  if (!decomposition.valid) {
    return makeError(SyncCoverSelectionError::InvalidUniverse);
  }

  SyncCoverSelectionResult result;
  result.components = decomposition.components;
  result.optimalityProven = true;
  std::vector<SyncCoverMechanismId> selected;
  for (const SyncCoverSelectionComponent &component : result.components) {
    const std::vector<std::vector<SyncCoverMechanismId>> componentSeeds =
        getComponentSeeds(component, seeds);
    ComponentSearchResult componentResult =
        component.exact ? searchExact(selectionEvaluator, coverage, component,
                                      componentSeeds, options)
                        : searchBeam(selectionEvaluator, coverage, component,
                                     componentSeeds, options);
    result.evaluations += componentResult.evaluations;
    result.truncation.beamWidth |= componentResult.truncation.beamWidth;
    result.truncation.beamDepth |= componentResult.truncation.beamDepth;
    result.truncation.evaluationLimit |=
        componentResult.truncation.evaluationLimit;
    result.optimalityProven &= componentResult.optimalityProven;
    if (!componentResult.selected) {
      result.error = componentResult.optimalityProven
                         ? SyncCoverSelectionError::ProvenInfeasible
                         : SyncCoverSelectionError::SearchIncomplete;
      result.coverageStatistics = oracle.getStatistics();
      return result;
    }
    selected.insert(selected.end(), componentResult.selected->begin(),
                    componentResult.selected->end());
  }
  std::sort(selected.begin(), selected.end());

  SyncCoverStructuralCost selectedCost;
  if (!evaluateCompleteSelection(selectionEvaluator, coverage, demands,
                                 selected, selectedCost)) {
    result.error = result.truncation
                       ? SyncCoverSelectionError::SearchIncomplete
                       : SyncCoverSelectionError::FinalVerificationFailed;
    result.optimalityProven = false;
    result.coverageStatistics = oracle.getStatistics();
    return result;
  }
  removeRedundantMechanisms(selectionEvaluator, coverage, demands, selected,
                            selectedCost, result.redundancyEvaluations);

  for (const SyncCoverSelectionSeed &seed : seeds) {
    SyncCoverStructuralCost seedCost;
    const bool feasibleSeed = evaluateCompleteSelection(
        selectionEvaluator, coverage, demands, seed.mechanisms, seedCost);
    if (feasibleSeed && syncCoverStructuralCostLess(seedCost, selectedCost)) {
      selected = seed.mechanisms;
      selectedCost = std::move(seedCost);
    }
  }

  const SyncCoverCoverageStatistics searchCoverageStatistics =
      oracle.getStatistics();
  if (!independentlyVerifySelection(
          universe, oracle, demands, selected, selectedCost, result.resources,
          result.finalVerificationStatistics)) {
    result.error = SyncCoverSelectionError::FinalVerificationFailed;
    result.optimalityProven = false;
    result.coverageStatistics = searchCoverageStatistics;
    return result;
  }

  result.mechanisms = std::move(selected);
  result.cost = std::move(selectedCost);
  result.coverageStatistics = searchCoverageStatistics;
  return result;
}
