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
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_internal;

namespace {

constexpr std::size_t kMaximumCachedWitnessesPerDemand = 4;

bool isSubset(const std::vector<SyncCoverMechanismId> &subset,
              const std::vector<SyncCoverMechanismId> &superset) {
  return std::includes(superset.begin(), superset.end(), subset.begin(),
                       subset.end());
}

template <typename T>
bool isCanonicalIdentitySet(const std::vector<T> &values) {
  return std::is_sorted(values.begin(), values.end()) &&
         std::adjacent_find(values.begin(), values.end()) == values.end();
}

} // namespace

SyncCoverMembershipResult mlir::pto::evaluateSyncCoverMembership(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &selected) {
  SyncCoverMembershipResult result;
  const SyncCoverSelectionEvaluator selectionEvaluator(universe);
  if (!selectionEvaluator) {
    result.error = SyncCoverMembershipError::InvalidUniverse;
    return result;
  }
  const bool invalidDemands =
      !isCanonicalIdentitySet(activeDemands) ||
      (!activeDemands.empty() &&
       activeDemands.back() >= universe.getGraph().getDemands().size());
  if (invalidDemands) {
    result.error = SyncCoverMembershipError::InvalidDemand;
    return result;
  }
  const bool invalidSelection =
      !isCanonicalIdentitySet(selected) ||
      (!selected.empty() &&
       selected.back() >= universe.getMechanisms().size());
  if (invalidSelection) {
    result.error = SyncCoverMembershipError::InvalidSelection;
    return result;
  }

  const SyncCoverSelectionEvaluation evaluation =
      selectionEvaluator.evaluate(selected);
  result.resources = evaluation.resources;
  result.cost = evaluation.cost;
  if (!result.resources.isValid() || !result.resources.resourceFeasible) {
    return result;
  }

  SyncCoverCoverageOracle oracle(universe.getGraph());
  for (SyncCoverDemandId demand : activeDemands) {
    const SyncCoverCoverageResult coverage =
        oracle.checkDemandCanonicalSelection(demand, selected);
    if (!coverage) {
      result.error = SyncCoverMembershipError::CoverageFailure;
      return result;
    }
    if (!coverage.covered) {
      result.uncoveredDemands.push_back({demand, coverage.cutMechanisms});
    }
  }
  result.coverageComplete = result.uncoveredDemands.empty();
  result.coverageStatistics = oracle.getStatistics();
  return result;
}

SyncCoverBarrierFreeCensusResult
mlir::pto::evaluateSyncCoverBarrierFreeCensus(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands) {
  SyncCoverBarrierFreeCensusResult result;
  if (!universe.validate()) {
    result.error = SyncCoverBarrierFreeCensusError::InvalidUniverse;
    return result;
  }
  const bool invalidDemands =
      !isCanonicalIdentitySet(activeDemands) ||
      (!activeDemands.empty() &&
       activeDemands.back() >= universe.getGraph().getDemands().size());
  if (invalidDemands) {
    result.error = SyncCoverBarrierFreeCensusError::InvalidDemand;
    return result;
  }

  std::vector<SyncCoverMechanismId> barrierFree;
  for (const SyncCoverMechanism &mechanism : universe.getMechanisms()) {
    if (mechanism.kind != SyncCoverMechanismKind::Barrier) {
      barrierFree.push_back(mechanism.id);
    }
  }

  SyncCoverCoverageOracle oracle(universe.getGraph());
  result.entries.reserve(activeDemands.size());
  for (SyncCoverDemandId demand : activeDemands) {
    const SyncCoverCoverageResult coverage =
        oracle.checkDemandCanonicalSelection(demand, barrierFree);
    if (!coverage) {
      result.error = SyncCoverBarrierFreeCensusError::CoverageFailure;
      result.failedDemand = demand;
      result.coverageError = coverage.error;
      result.coverageStatistics = oracle.getStatistics();
      return result;
    }

    SyncCoverBarrierFreeCensusEntry entry;
    entry.demand = demand;
    entry.reachableStates = coverage.reachableStates;
    if (!coverage.covered) {
      result.entries.push_back(std::move(entry));
      continue;
    }

    entry.witnessMechanisms = coverage.witnessMechanisms;
    entry.witnessResources =
        universe.evaluateResourceSelection(entry.witnessMechanisms);
    switch (entry.witnessResources.error) {
    case SyncCoverResourceSelectionError::None:
    case SyncCoverResourceSelectionError::Conflict:
      break;
    case SyncCoverResourceSelectionError::InvalidUniverse:
      result.error = SyncCoverBarrierFreeCensusError::InvalidUniverse;
      return result;
    case SyncCoverResourceSelectionError::InvalidSelection:
      result.error = SyncCoverBarrierFreeCensusError::InvalidWitness;
      return result;
    case SyncCoverResourceSelectionError::ArithmeticOverflow:
      result.error =
          SyncCoverBarrierFreeCensusError::ResourceEvaluationFailed;
      return result;
    }
    entry.status = entry.witnessResources
                       ? SyncCoverBarrierFreeCensusStatus::FeasibleWitness
                       : SyncCoverBarrierFreeCensusStatus::InfeasibleWitness;
    result.entries.push_back(std::move(entry));
  }
  result.coverageStatistics = oracle.getStatistics();
  return result;
}

CoverageEvaluation
CoverageEvaluator::evaluate(const std::vector<SyncCoverDemandId> &demands,
                            const std::vector<SyncCoverMechanismId> &selected) {
  CoverageEvaluation evaluation;
  for (SyncCoverDemandId demand : demands) {
    bool coveredByCache = false;
    for (const auto &witness : witnesses_[demand]) {
      if (isSubset(witness, selected)) {
        coveredByCache = true;
        break;
      }
    }
    if (coveredByCache) {
      continue;
    }

    const SyncCoverCoverageResult result =
        oracle_.checkDemandCanonicalSelection(demand, selected);
    if (!result) {
      evaluation.valid = false;
      return evaluation;
    }
    if (result.covered) {
      auto &known = witnesses_[demand];
      const bool impliedByKnown =
          std::any_of(known.begin(), known.end(), [&](const auto &witness) {
            return isSubset(witness, result.witnessMechanisms);
          });
      if (!impliedByKnown) {
        known.erase(std::remove_if(known.begin(), known.end(),
                                   [&](const auto &witness) {
                                     return isSubset(result.witnessMechanisms,
                                                     witness);
                                   }),
                    known.end());
        known.push_back(result.witnessMechanisms);
        std::sort(known.begin(), known.end(),
                  [](const auto &first, const auto &second) {
                    return std::make_tuple(first.size(), first) <
                           std::make_tuple(second.size(), second);
                  });
        const bool exceedsWitnessLimit =
            known.size() > kMaximumCachedWitnessesPerDemand;
        if (exceedsWitnessLimit) {
          known.resize(kMaximumCachedWitnessesPerDemand);
        }
      }
      continue;
    }
    evaluation.uncovered.push_back(demand);
    evaluation.cuts.emplace(demand, result.cutMechanisms);
    evaluation.reachableStates.emplace(demand, result.reachableStates);
  }
  return evaluation;
}

bool mlir::pto::sync_cover_internal::evaluateCompleteSelection(
    const SyncCoverSelectionEvaluator &selectionEvaluator,
    CoverageEvaluator &coverage,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &selected,
    SyncCoverStructuralCost &cost) {
  const SyncCoverSelectionEvaluation selection =
      selectionEvaluator.evaluate(selected);
  if (!selection || !selection.resources.resourceFeasible) {
    return false;
  }
  const CoverageEvaluation result = coverage.evaluate(activeDemands, selected);
  if (!result.valid || !result.uncovered.empty()) {
    return false;
  }
  cost = selection.cost;
  return true;
}

void mlir::pto::sync_cover_internal::removeRedundantMechanisms(
    const SyncCoverSelectionEvaluator &selectionEvaluator,
    CoverageEvaluator &coverage,
    const std::vector<SyncCoverDemandId> &activeDemands,
    std::vector<SyncCoverMechanismId> &selected,
    SyncCoverStructuralCost &cost, std::size_t &evaluations) {
  while (!selected.empty()) {
    std::optional<std::vector<SyncCoverMechanismId>> bestSelection;
    std::optional<SyncCoverStructuralCost> bestCost;
    for (std::size_t index = 0; index < selected.size(); ++index) {
      std::vector<SyncCoverMechanismId> candidate = selected;
      candidate.erase(candidate.begin() + index);
      SyncCoverStructuralCost candidateCost;
      ++evaluations;
      const bool complete =
          evaluateCompleteSelection(selectionEvaluator, coverage, activeDemands,
                                    candidate, candidateCost);
      if (!complete || !syncCoverStructuralCostLess(candidateCost, cost)) {
        continue;
      }
      if (!bestCost || syncCoverStructuralCostLess(candidateCost, *bestCost)) {
        bestSelection = std::move(candidate);
        bestCost = std::move(candidateCost);
      }
    }
    if (!bestSelection) {
      return;
    }
    selected = std::move(*bestSelection);
    cost = std::move(*bestCost);
  }
}

bool mlir::pto::sync_cover_internal::independentlyVerifySelection(
    const SyncCoverMechanismUniverse &universe,
    SyncCoverCoverageOracle &coverage,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &selected,
    SyncCoverStructuralCost &cost, SyncCoverResourceSelection &resources,
    SyncCoverCoverageStatistics &statistics) {
  const SyncCoverGraphResult graphValidation = universe.getGraph().validate();
  const SyncCoverSelectionEvaluator selectionEvaluator(universe);
  if (!graphValidation || !selectionEvaluator) {
    return false;
  }
  const SyncCoverSelectionEvaluation selection =
      selectionEvaluator.evaluate(selected);
  if (!selection || !selection.resources.resourceFeasible) {
    return false;
  }

  const SyncCoverCoverageStatistics before = coverage.getStatistics();
  for (SyncCoverDemandId demand : activeDemands) {
    const SyncCoverCoverageResult result =
        coverage.checkDemandCanonicalSelection(demand, selected);
    if (!result || !result.covered) {
      return false;
    }
  }
  const SyncCoverCoverageStatistics after = coverage.getStatistics();
  statistics.graphValidations = 1;
  statistics.coverageQueries =
      after.coverageQueries - before.coverageQueries;
  cost = selection.cost;
  resources = selection.resources;
  return true;
}
