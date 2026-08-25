// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverSolver.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::pto;

namespace {

class UnionFind {
public:
  explicit UnionFind(std::size_t size) : parents_(size), ranks_(size, 0) {
    std::iota(parents_.begin(), parents_.end(), 0);
  }

  std::size_t find(std::size_t value) {
    if (parents_[value] != value) {
      parents_[value] = find(parents_[value]);
    }
    return parents_[value];
  }

  void unite(std::size_t first, std::size_t second) {
    first = find(first);
    second = find(second);
    if (first == second) {
      return;
    }
    if (ranks_[first] < ranks_[second]) {
      std::swap(first, second);
    }
    parents_[second] = first;
    if (ranks_[first] == ranks_[second]) {
      ++ranks_[first];
    }
  }

private:
  std::vector<std::size_t> parents_;
  std::vector<unsigned> ranks_;
};

struct ResourceInterval {
  SyncCoverResourceDomainId domain = 0;
  SyncCoverTimelinePosition begin = 0;
  SyncCoverTimelinePosition end = 0;
  SyncCoverMechanismId mechanism = 0;
};

struct SearchEvaluation {
  SyncCoverDemandSet covered;
  SyncCoverStructuralCost cost;
};

struct ComponentSearchResult {
  std::optional<std::vector<SyncCoverMechanismId>> selected;
  std::optional<SyncCoverStructuralCost> cost;
  std::size_t evaluations = 0;
  std::size_t boundedEvaluations = 0;
  SyncCoverSearchTruncation truncation;
  bool optimalityProven = false;
};

template <typename T> bool normalizeUnique(std::vector<T> &values) {
  std::sort(values.begin(), values.end());
  const auto duplicate = std::adjacent_find(values.begin(), values.end());
  return duplicate == values.end();
}

SyncCoverSelectionResult makeError(SyncCoverSelectionError error) {
  SyncCoverSelectionResult result;
  result.error = error;
  return result;
}

std::size_t localDemand(const SyncCoverGroundedInstance &instance,
                        SyncCoverDemandId demand) {
  return static_cast<std::size_t>(
      std::lower_bound(instance.demands.begin(), instance.demands.end(),
                       demand) -
      instance.demands.begin());
}

std::vector<SyncCoverMechanismId>
addMembers(const std::vector<SyncCoverMechanismId> &selected,
           const std::vector<SyncCoverMechanismId> &members) {
  std::vector<SyncCoverMechanismId> result;
  result.reserve(selected.size() + members.size());
  std::set_union(selected.begin(), selected.end(), members.begin(),
                 members.end(), std::back_inserter(result));
  return result;
}

bool componentCovered(const SyncCoverGroundedInstance &instance,
                      const SyncCoverSelectionComponent &component,
                      const SyncCoverDemandSet &covered) {
  return std::all_of(component.demands.begin(), component.demands.end(),
                     [&](SyncCoverDemandId demand) {
                       return covered.contains(localDemand(instance, demand));
                     });
}

std::size_t componentUncoveredCount(
    const SyncCoverGroundedInstance &instance,
    const SyncCoverSelectionComponent &component,
    const SyncCoverDemandSet &covered) {
  return static_cast<std::size_t>(std::count_if(
      component.demands.begin(), component.demands.end(),
      [&](SyncCoverDemandId demand) {
        return !covered.contains(localDemand(instance, demand));
      }));
}

std::optional<std::size_t>
firstUncoveredDemand(const SyncCoverGroundedInstance &instance,
                     const SyncCoverSelectionComponent &component,
                     const SyncCoverDemandSet &covered) {
  for (SyncCoverDemandId demand : component.demands) {
    const std::size_t local = localDemand(instance, demand);
    if (!covered.contains(local)) {
      return local;
    }
  }
  return std::nullopt;
}

std::optional<SearchEvaluation>
evaluateSelection(const SyncCoverSelectionEvaluator &evaluator,
                  const SyncCoverGroundedInstance &instance,
                  const std::vector<SyncCoverMechanismId> &selected) {
  const SyncCoverSelectionEvaluation evaluation = evaluator.evaluate(selected);
  if (!evaluation || !evaluation.resources.resourceFeasible) {
    return std::nullopt;
  }
  return SearchEvaluation{instance.coveredBy(selected), evaluation.cost};
}

void considerComplete(
    const std::vector<SyncCoverMechanismId> &selected,
    const SyncCoverStructuralCost &cost,
    std::optional<std::vector<SyncCoverMechanismId>> &best,
    std::optional<SyncCoverStructuralCost> &bestCost) {
  if (!bestCost || syncCoverStructuralCostLess(cost, *bestCost)) {
    best = selected;
    bestCost = cost;
  }
}

std::vector<SyncCoverSelectionComponent> buildComponents(
    const SyncCoverGroundedInstance &instance, std::size_t exactThreshold) {
  const std::size_t demandCount = instance.demands.size();
  const std::size_t mechanismCount = instance.mechanisms.size();
  UnionFind sets(demandCount + mechanismCount);
  std::vector<bool> relevant(mechanismCount, false);

  for (const SyncCoverGroundedColumn &column : instance.columns) {
    for (SyncCoverMechanismId mechanism : column.members) {
      relevant[mechanism] = true;
    }
    for (std::size_t demand = 0; demand < demandCount; ++demand) {
      if (!column.coverage.contains(demand)) {
        continue;
      }
      for (SyncCoverMechanismId mechanism : column.members) {
        sets.unite(demandCount + mechanism, demand);
      }
    }
  }

  std::vector<ResourceInterval> intervals;
  for (const SyncCoverGroundedMechanism &mechanism : instance.mechanisms) {
    if (!relevant[mechanism.id]) {
      continue;
    }
    for (const SyncCoverGroundedResourceUse &use : mechanism.resourceUses) {
      intervals.push_back({use.domain, use.lifetime.begin, use.lifetime.end,
                           mechanism.id});
    }
    for (SyncCoverMechanismId conflict : mechanism.conflicts) {
      const bool relevantConflict =
          conflict < relevant.size() && relevant[conflict];
      if (relevantConflict) {
        sets.unite(demandCount + mechanism.id, demandCount + conflict);
      }
    }
  }
  std::sort(intervals.begin(), intervals.end(),
            [](const ResourceInterval &first,
               const ResourceInterval &second) {
              return std::tie(first.domain, first.begin, first.end,
                              first.mechanism) <
                     std::tie(second.domain, second.begin, second.end,
                              second.mechanism);
            });
  std::optional<ResourceInterval> active;
  for (const ResourceInterval &interval : intervals) {
    const bool startsGroup = !active || active->domain != interval.domain ||
                             active->end < interval.begin;
    if (startsGroup) {
      active = interval;
      continue;
    }
    sets.unite(demandCount + active->mechanism,
               demandCount + interval.mechanism);
    active->end = std::max(active->end, interval.end);
  }

  std::map<std::size_t, SyncCoverSelectionComponent> byRoot;
  for (std::size_t demand = 0; demand < demandCount; ++demand) {
    byRoot[sets.find(demand)].demands.push_back(instance.demands[demand]);
  }
  for (SyncCoverMechanismId mechanism = 0; mechanism < mechanismCount;
       ++mechanism) {
    if (relevant[mechanism]) {
      byRoot[sets.find(demandCount + mechanism)].mechanisms.push_back(
          mechanism);
    }
  }
  for (const SyncCoverGroundedColumn &column : instance.columns) {
    std::optional<std::size_t> root;
    if (!column.members.empty()) {
      root = sets.find(demandCount + column.members.front());
    } else {
      for (std::size_t demand = 0; demand < demandCount; ++demand) {
        if (column.coverage.contains(demand)) {
          root = sets.find(demand);
          break;
        }
      }
    }
    if (root) {
      byRoot[*root].columns.push_back(column.id);
    }
  }

  std::vector<SyncCoverSelectionComponent> result;
  for (auto &entry : byRoot) {
    SyncCoverSelectionComponent component = std::move(entry.second);
    component.exact = component.mechanisms.size() <= exactThreshold;
    result.push_back(std::move(component));
  }
  std::sort(result.begin(), result.end(), [](const auto &first,
                                             const auto &second) {
    return std::tie(first.demands, first.mechanisms) <
           std::tie(second.demands, second.mechanisms);
  });
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index].id = index;
  }
  return result;
}

void exactSearch(const SyncCoverSelectionEvaluator &evaluator,
                 const SyncCoverGroundedInstance &instance,
                 const SyncCoverSelectionComponent &component,
                 const SyncCoverSolverOptions &options,
                 std::vector<SyncCoverMechanismId> selected,
                 std::set<std::vector<SyncCoverMechanismId>> &seen,
                 ComponentSearchResult &result) {
  if (!seen.insert(selected).second) {
    return;
  }
  if (result.boundedEvaluations >= options.evaluationLimit) {
    result.truncation.evaluationLimit = true;
    return;
  }
  ++result.boundedEvaluations;
  ++result.evaluations;
  const std::optional<SearchEvaluation> evaluation =
      evaluateSelection(evaluator, instance, selected);
  if (!evaluation) {
    return;
  }
  if (componentCovered(instance, component, evaluation->covered)) {
    considerComplete(selected, evaluation->cost, result.selected, result.cost);
    return;
  }
  if (result.cost &&
      !syncCoverStructuralCostLess(evaluation->cost, *result.cost)) {
    return;
  }
  const std::optional<std::size_t> demand =
      firstUncoveredDemand(instance, component, evaluation->covered);
  if (!demand) {
    return;
  }
  for (SyncCoverGroundedColumnId columnId :
       instance.demandColumns[*demand]) {
    const std::vector<SyncCoverMechanismId> successor =
        addMembers(selected, instance.columns[columnId].members);
    if (successor == selected) {
      continue;
    }
    exactSearch(evaluator, instance, component, options, successor, seen,
                result);
    if (result.truncation.evaluationLimit) {
      return;
    }
  }
}

std::optional<std::pair<std::vector<SyncCoverMechanismId>,
                        SyncCoverStructuralCost>>
greedySearch(const SyncCoverSelectionEvaluator &evaluator,
             const SyncCoverGroundedInstance &instance,
             const SyncCoverSelectionComponent &component,
             const SyncCoverSolverOptions &options,
             std::vector<SyncCoverMechanismId> selected,
             std::size_t &evaluations, std::size_t &boundedEvaluations,
             SyncCoverSearchTruncation &truncation) {
  while (true) {
    if (boundedEvaluations >= options.evaluationLimit) {
      truncation.evaluationLimit = true;
      return std::nullopt;
    }
    ++boundedEvaluations;
    ++evaluations;
    const std::optional<SearchEvaluation> current =
        evaluateSelection(evaluator, instance, selected);
    if (!current) {
      return std::nullopt;
    }
    if (componentCovered(instance, component, current->covered)) {
      return std::make_pair(selected, current->cost);
    }
    const std::optional<std::size_t> demand =
        firstUncoveredDemand(instance, component, current->covered);
    if (!demand) {
      return std::nullopt;
    }

    std::optional<std::vector<SyncCoverMechanismId>> best;
    std::optional<SearchEvaluation> bestEvaluation;
    for (SyncCoverGroundedColumnId columnId :
         instance.demandColumns[*demand]) {
      const std::vector<SyncCoverMechanismId> successor =
          addMembers(selected, instance.columns[columnId].members);
      if (successor == selected) {
        continue;
      }
      if (boundedEvaluations >= options.evaluationLimit) {
        truncation.evaluationLimit = true;
        break;
      }
      ++boundedEvaluations;
      ++evaluations;
      const std::optional<SearchEvaluation> candidate =
          evaluateSelection(evaluator, instance, successor);
      if (!candidate) {
        continue;
      }
      const std::size_t candidateUncovered = componentUncoveredCount(
          instance, component, candidate->covered);
      const std::size_t bestUncovered =
          bestEvaluation
              ? componentUncoveredCount(instance, component,
                                        bestEvaluation->covered)
              : std::numeric_limits<std::size_t>::max();
      const bool betterCoverage = candidateUncovered < bestUncovered;
      const bool equalCoverage = candidateUncovered == bestUncovered;
      const bool betterCost =
          !bestEvaluation ||
          syncCoverStructuralCostLess(candidate->cost, bestEvaluation->cost);
      if (betterCoverage || (equalCoverage && betterCost)) {
        best = successor;
        bestEvaluation = candidate;
      }
    }
    if (!best || truncation.evaluationLimit) {
      return std::nullopt;
    }
    selected = std::move(*best);
  }
}

std::vector<std::vector<SyncCoverMechanismId>> componentSeeds(
    const SyncCoverSelectionComponent &component,
    const std::vector<SyncCoverSelectionSeed> &seeds) {
  std::vector<std::vector<SyncCoverMechanismId>> result{{}};
  for (const SyncCoverSelectionSeed &seed : seeds) {
    std::vector<SyncCoverMechanismId> selected;
    std::set_intersection(seed.mechanisms.begin(), seed.mechanisms.end(),
                          component.mechanisms.begin(),
                          component.mechanisms.end(),
                          std::back_inserter(selected));
    result.push_back(std::move(selected));
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

ComponentSearchResult searchComponent(
    const SyncCoverSelectionEvaluator &evaluator,
    const SyncCoverGroundedInstance &instance,
    const SyncCoverSelectionComponent &component,
    const std::vector<SyncCoverSelectionSeed> &seeds,
    const SyncCoverSolverOptions &options) {
  ComponentSearchResult result;
  const auto starts = componentSeeds(component, seeds);
  for (const auto &start : starts) {
    const std::optional<SearchEvaluation> evaluation =
        evaluateSelection(evaluator, instance, start);
    ++result.evaluations;
    if (evaluation && componentCovered(instance, component,
                                       evaluation->covered)) {
      considerComplete(start, evaluation->cost, result.selected, result.cost);
    }
  }
  if (component.exact) {
    std::set<std::vector<SyncCoverMechanismId>> seen;
    exactSearch(evaluator, instance, component, options, {}, seen, result);
    result.optimalityProven = !result.truncation;
    return result;
  }

  for (const auto &start : starts) {
    const auto candidate =
        greedySearch(evaluator, instance, component, options, start,
                     result.evaluations, result.boundedEvaluations,
                     result.truncation);
    if (candidate) {
      considerComplete(candidate->first, candidate->second, result.selected,
                       result.cost);
    }
    if (result.truncation.evaluationLimit) {
      break;
    }
  }
  result.optimalityProven = false;
  return result;
}

void removeRedundant(const SyncCoverSelectionEvaluator &evaluator,
                     const SyncCoverGroundedInstance &instance,
                     std::vector<SyncCoverMechanismId> &selected,
                     SyncCoverStructuralCost &cost,
                     std::size_t &evaluations) {
  for (std::size_t index = selected.size(); index > 0; --index) {
    std::vector<SyncCoverMechanismId> candidate = selected;
    candidate.erase(candidate.begin() + static_cast<std::ptrdiff_t>(index - 1));
    ++evaluations;
    const std::optional<SearchEvaluation> evaluation =
        evaluateSelection(evaluator, instance, candidate);
    const bool improves = evaluation && instance.coversAll(candidate) &&
                          syncCoverStructuralCostLess(evaluation->cost, cost);
    if (improves) {
      selected = std::move(candidate);
      cost = evaluation->cost;
    }
  }
}

bool finalVerify(const SyncCoverMechanismUniverse &universe,
                 const std::vector<SyncCoverDemandId> &demands,
                 const std::vector<SyncCoverMechanismId> &selected,
                 SyncCoverResourceSelection &resources,
                 SyncCoverStructuralCost &cost,
                 SyncCoverCoverageStatistics &statistics) {
  const SyncCoverSelectionEvaluator evaluator(universe);
  const SyncCoverSelectionEvaluation evaluation = evaluator.evaluate(selected);
  if (!evaluation) {
    return false;
  }
  SyncCoverCoverageOracle oracle(universe.getGraph());
  for (SyncCoverDemandId demand : demands) {
    const SyncCoverCoverageResult coverage =
        oracle.checkDemandCanonicalSelection(demand, selected);
    if (!coverage || !coverage.covered) {
      statistics = oracle.getStatistics();
      return false;
    }
  }
  resources = evaluation.resources;
  cost = evaluation.cost;
  statistics = oracle.getStatistics();
  return true;
}

} // namespace

SyncCoverSelectionResult mlir::pto::solveSyncCoverSelection(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverSelectionSeed> &inputSeeds,
    const SyncCoverSolverOptions &options) {
  const bool invalidOptions =
      options.evaluationLimit == 0 ||
      options.exactMechanismThreshold >
          SyncCoverSolverOptions::maximumExactMechanismThreshold;
  if (invalidOptions) {
    return makeError(SyncCoverSelectionError::InvalidOptions);
  }
  std::vector<SyncCoverDemandId> demands = activeDemands;
  const bool invalidDemandOrder = !normalizeUnique(demands);
  const bool invalidDemandId =
      !demands.empty() &&
      demands.back() >= universe.getGraph().getDemands().size();
  if (invalidDemandOrder || invalidDemandId) {
    return makeError(SyncCoverSelectionError::InvalidDemand);
  }
  std::vector<SyncCoverSelectionSeed> seeds = inputSeeds;
  for (SyncCoverSelectionSeed &seed : seeds) {
    const bool invalidSeedOrder = !normalizeUnique(seed.mechanisms);
    const bool invalidMechanism =
        !seed.mechanisms.empty() &&
        seed.mechanisms.back() >= universe.getMechanisms().size();
    if (invalidSeedOrder || invalidMechanism) {
      return makeError(SyncCoverSelectionError::InvalidSeed);
    }
  }
  std::sort(seeds.begin(), seeds.end(), [](const auto &first,
                                           const auto &second) {
    return std::tie(first.identity, first.mechanisms) <
           std::tie(second.identity, second.mechanisms);
  });

  std::vector<std::vector<SyncCoverMechanismId>> seedColumns;
  seedColumns.reserve(seeds.size());
  for (const SyncCoverSelectionSeed &seed : seeds) {
    seedColumns.push_back(seed.mechanisms);
  }
  const SyncCoverGroundingResult grounding =
      groundSyncCoverInstance(universe, demands, seedColumns);
  if (!grounding) {
    SyncCoverSelectionResult result =
        makeError(SyncCoverSelectionError::InvalidUniverse);
    result.coverageStatistics = grounding.statistics;
    return result;
  }
  const SyncCoverGroundedInstance &instance = grounding.instance;
  const SyncCoverSelectionEvaluator evaluator(universe);
  if (!evaluator || !instance.isCurrent(universe)) {
    return makeError(SyncCoverSelectionError::InvalidUniverse);
  }

  SyncCoverSelectionResult result;
  result.coverageStatistics = grounding.statistics;
  if (!instance.provenUncoverableDemands.empty()) {
    result.error = SyncCoverSelectionError::ProvenInfeasible;
    result.optimalityProven = true;
    return result;
  }
  result.components =
      buildComponents(instance, options.exactMechanismThreshold);
  result.optimalityProven = instance.demandsNeedingPricing.empty();
  std::vector<SyncCoverMechanismId> selected;
  for (const SyncCoverSelectionComponent &component : result.components) {
    ComponentSearchResult componentResult =
        searchComponent(evaluator, instance, component, seeds, options);
    result.evaluations += componentResult.evaluations;
    result.truncation.evaluationLimit |=
        componentResult.truncation.evaluationLimit;
    result.optimalityProven &= componentResult.optimalityProven;
    if (!componentResult.selected) {
      result.error = SyncCoverSelectionError::SearchIncomplete;
      result.optimalityProven = false;
      return result;
    }
    selected.insert(selected.end(), componentResult.selected->begin(),
                    componentResult.selected->end());
  }
  std::sort(selected.begin(), selected.end());
  selected.erase(std::unique(selected.begin(), selected.end()),
                 selected.end());

  const std::optional<SearchEvaluation> initial =
      evaluateSelection(evaluator, instance, selected);
  if (!initial || !instance.coversAll(selected)) {
    result.error = SyncCoverSelectionError::SearchIncomplete;
    result.optimalityProven = false;
    return result;
  }
  SyncCoverStructuralCost selectedCost = initial->cost;
  removeRedundant(evaluator, instance, selected, selectedCost,
                  result.redundancyEvaluations);
  for (const SyncCoverSelectionSeed &seed : seeds) {
    const std::optional<SearchEvaluation> seedEvaluation =
        evaluateSelection(evaluator, instance, seed.mechanisms);
    const bool improves =
        seedEvaluation && instance.coversAll(seed.mechanisms) &&
        syncCoverStructuralCostLess(seedEvaluation->cost, selectedCost);
    if (improves) {
      selected = seed.mechanisms;
      selectedCost = seedEvaluation->cost;
    }
  }

  if (!finalVerify(universe, demands, selected, result.resources,
                   selectedCost, result.finalVerificationStatistics)) {
    result.error = SyncCoverSelectionError::FinalVerificationFailed;
    result.optimalityProven = false;
    return result;
  }
  result.mechanisms = std::move(selected);
  result.cost = std::move(selectedCost);
  return result;
}
