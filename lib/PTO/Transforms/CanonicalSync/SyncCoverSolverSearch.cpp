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
#include <set>
#include <utility>
#include <vector>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_internal;

namespace {

struct SearchEvaluation {
  CoverageEvaluation coverage;
  SyncCoverStructuralCost cost;
  std::vector<SyncCoverMechanismId> branch;
};

std::optional<SearchEvaluation>
evaluateState(const SyncCoverSelectionEvaluator &selectionEvaluator,
              CoverageEvaluator &coverage,
              const SyncCoverSelectionComponent &component,
              const std::vector<SyncCoverMechanismId> &selected) {
  const SyncCoverSelectionEvaluation selection =
      selectionEvaluator.evaluate(selected);
  if (!selection || !selection.resources.resourceFeasible) {
    return std::nullopt;
  }
  SearchEvaluation result;
  result.coverage = coverage.evaluate(component.demands, selected);
  if (!result.coverage.valid) {
    return std::nullopt;
  }
  result.cost = selection.cost;

  std::optional<std::pair<std::size_t, SyncCoverDemandId>> bestCut;
  for (SyncCoverDemandId demand : result.coverage.uncovered) {
    std::vector<SyncCoverMechanismId> candidates;
    const auto cut = result.coverage.cuts.find(demand);
    if (cut != result.coverage.cuts.end()) {
      std::set_intersection(
          cut->second.begin(), cut->second.end(), component.mechanisms.begin(),
          component.mechanisms.end(), std::back_inserter(candidates));
    }
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [&](SyncCoverMechanismId mechanism) {
                                      return std::binary_search(
                                          selected.begin(), selected.end(),
                                          mechanism);
                                    }),
                     candidates.end());
    const auto candidateKey = std::make_pair(candidates.size(), demand);
    if (!bestCut || candidateKey < *bestCut) {
      bestCut = candidateKey;
      result.branch = std::move(candidates);
    }
  }
  return result;
}

using SearchState =
    std::pair<std::vector<SyncCoverMechanismId>, SearchEvaluation>;

bool stateLess(const SearchState &first, const SearchState &second) {
  const std::size_t firstUncovered = first.second.coverage.uncovered.size();
  const std::size_t secondUncovered = second.second.coverage.uncovered.size();
  if (firstUncovered != secondUncovered) {
    return firstUncovered < secondUncovered;
  }
  const bool differentBranchCounts =
      first.second.branch.size() != second.second.branch.size();
  if (differentBranchCounts) {
    return first.second.branch.size() < second.second.branch.size();
  }
  if (syncCoverStructuralCostLess(first.second.cost, second.second.cost)) {
    return true;
  }
  if (syncCoverStructuralCostLess(second.second.cost, first.second.cost)) {
    return false;
  }
  return first.first < second.first;
}

void considerComplete(
    const std::vector<SyncCoverMechanismId> &selected,
    const SyncCoverStructuralCost &cost,
    std::optional<std::vector<SyncCoverMechanismId>> &bestSelection,
    std::optional<SyncCoverStructuralCost> &bestCost) {
  if (!bestCost || syncCoverStructuralCostLess(cost, *bestCost)) {
    bestSelection = selected;
    bestCost = cost;
  }
}

void exactSearch(const SyncCoverSelectionEvaluator &selectionEvaluator,
                 CoverageEvaluator &coverage,
                 const SyncCoverSelectionComponent &component,
                 std::vector<SyncCoverMechanismId> selected,
                 std::set<std::vector<SyncCoverMechanismId>> &seen,
                 ComponentSearchResult &result) {
  if (!seen.insert(selected).second) {
    return;
  }
  ++result.evaluations;
  const std::optional<SearchEvaluation> evaluation =
      evaluateState(selectionEvaluator, coverage, component, selected);
  if (!evaluation) {
    return;
  }
  if (evaluation->coverage.uncovered.empty()) {
    considerComplete(selected, evaluation->cost, result.selected, result.cost);
    return;
  }
  const bool dominated = result.cost && !syncCoverStructuralCostLess(
                                            evaluation->cost, *result.cost);
  const bool cannotImprove = evaluation->branch.empty() || dominated;
  if (cannotImprove) {
    return;
  }
  for (SyncCoverMechanismId mechanism : evaluation->branch) {
    std::vector<SyncCoverMechanismId> successor = selected;
    successor.insert(
        std::lower_bound(successor.begin(), successor.end(), mechanism),
        mechanism);
    exactSearch(selectionEvaluator, coverage, component, std::move(successor),
                seen, result);
  }
}

void evaluateInitialState(const SyncCoverSelectionEvaluator &selectionEvaluator,
                          CoverageEvaluator &coverage,
                          const SyncCoverSelectionComponent &component,
                          const std::vector<SyncCoverMechanismId> &selected,
                          ComponentSearchResult &result,
                          std::vector<SearchState> *incomplete) {
  ++result.evaluations;
  const auto evaluation =
      evaluateState(selectionEvaluator, coverage, component, selected);
  if (!evaluation) {
    return;
  }
  if (evaluation->coverage.uncovered.empty()) {
    considerComplete(selected, evaluation->cost, result.selected, result.cost);
  } else if (incomplete != nullptr && !evaluation->branch.empty()) {
    incomplete->emplace_back(selected, *evaluation);
  }
}

} // namespace

ComponentSearchResult mlir::pto::sync_cover_internal::searchExact(
    const SyncCoverSelectionEvaluator &selectionEvaluator,
    CoverageEvaluator &coverage, const SyncCoverSelectionComponent &component,
    const std::vector<std::vector<SyncCoverMechanismId>> &seedSelections) {
  ComponentSearchResult result;
  for (const auto &seed : seedSelections) {
    evaluateInitialState(selectionEvaluator, coverage, component, seed, result,
                         nullptr);
  }
  std::set<std::vector<SyncCoverMechanismId>> seen;
  exactSearch(selectionEvaluator, coverage, component, {}, seen, result);
  result.optimalityProven = true;
  return result;
}

ComponentSearchResult mlir::pto::sync_cover_internal::searchBeam(
    const SyncCoverSelectionEvaluator &selectionEvaluator,
    CoverageEvaluator &coverage, const SyncCoverSelectionComponent &component,
    const std::vector<std::vector<SyncCoverMechanismId>> &seedSelections,
    const SyncCoverSolverOptions &options) {
  ComponentSearchResult result;
  std::vector<SearchState> beam;
  std::set<std::vector<SyncCoverMechanismId>> seen;
  std::vector<std::vector<SyncCoverMechanismId>> seeds = seedSelections;
  std::sort(seeds.begin(), seeds.end());
  seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
  for (const auto &selected : seeds) {
    seen.insert(selected);
    evaluateInitialState(selectionEvaluator, coverage, component, selected,
                         result, &beam);
  }

  std::size_t boundedEvaluations = 0;
  const std::vector<SyncCoverMechanismId> empty;
  if (seen.insert(empty).second) {
    ++boundedEvaluations;
    evaluateInitialState(selectionEvaluator, coverage, component, empty, result,
                         &beam);
  }

  const std::size_t depthLimit =
      std::min(options.beamDepth, component.mechanisms.size());
  std::size_t depth = 0;
  for (; depth < depthLimit && !beam.empty() &&
         !result.truncation.evaluationLimit;
       ++depth) {
    std::vector<SearchState> next;
    for (const SearchState &state : beam) {
      for (SyncCoverMechanismId mechanism : state.second.branch) {
        std::vector<SyncCoverMechanismId> successor = state.first;
        successor.insert(
            std::lower_bound(successor.begin(), successor.end(), mechanism),
            mechanism);
        if (!seen.insert(successor).second) {
          continue;
        }
        if (boundedEvaluations == options.evaluationLimit) {
          result.truncation.evaluationLimit = true;
          break;
        }
        ++boundedEvaluations;
        evaluateInitialState(selectionEvaluator, coverage, component, successor,
                             result, &next);
      }
      if (result.truncation.evaluationLimit) {
        break;
      }
    }
    std::stable_sort(next.begin(), next.end(), stateLess);
    const bool exceedsBeamWidth = next.size() > options.beamWidth;
    if (exceedsBeamWidth) {
      result.truncation.beamWidth = true;
      next.resize(options.beamWidth);
    }
    beam = std::move(next);
  }
  const bool stoppedAtDepth = !beam.empty() && depth == depthLimit &&
                              !result.truncation.evaluationLimit;
  result.truncation.beamDepth = stoppedAtDepth;
  result.optimalityProven = !result.truncation;
  return result;
}
