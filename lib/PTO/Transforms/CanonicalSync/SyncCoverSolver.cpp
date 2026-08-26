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

struct DemandCoverageKey {
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  std::vector<SyncCoverGuardLiteral> sourceGuard;
  std::vector<SyncCoverGuardLiteral> targetGuard;

  bool operator<(const DemandCoverageKey &other) const {
    return std::tie(source, target, scope, distance, sourceGuard,
                    targetGuard) <
           std::tie(other.source, other.target, other.scope, other.distance,
                    other.sourceGuard, other.targetGuard);
  }
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

std::optional<std::size_t>
mostConstrainedUncoveredDemand(const SyncCoverGroundedInstance &instance,
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

struct GreedyCandidate {
  SyncCoverGroundedColumnId column = 0;
  std::vector<SyncCoverMechanismId> successor;
  std::vector<std::size_t> barrierActions;
  std::vector<std::size_t> eventActions;
  std::size_t addedMechanisms = 0;
  std::size_t newCoverage = 0;
};

std::size_t countNewCoverage(const SyncCoverGroundedInstance &instance,
                             const SyncCoverSelectionComponent &component,
                             const SyncCoverDemandSet &covered,
                             const SyncCoverDemandSet &candidate) {
  return static_cast<std::size_t>(std::count_if(
      component.demands.begin(), component.demands.end(),
      [&](SyncCoverDemandId demand) {
        const std::size_t local = localDemand(instance, demand);
        return !covered.contains(local) && candidate.contains(local);
      }));
}

GreedyCandidate makeGreedyCandidate(
    const SyncCoverGroundedInstance &instance,
    const SyncCoverSelectionComponent &component,
    const std::vector<SyncCoverMechanismId> &selected,
    const SyncCoverDemandSet &covered, SyncCoverGroundedColumnId columnId) {
  const SyncCoverGroundedColumn &column = instance.columns[columnId];
  GreedyCandidate result;
  result.column = columnId;
  result.successor = addMembers(selected, column.members);
  result.newCoverage =
      countNewCoverage(instance, component, covered, column.coverage);
  const std::size_t profileSize = instance.mechanisms.empty()
                                      ? 0
                                      : instance.mechanisms.front()
                                            .actionProfile.size();
  result.barrierActions.assign(profileSize, 0);
  result.eventActions.assign(profileSize, 0);
  for (SyncCoverMechanismId member : column.members) {
    if (std::binary_search(selected.begin(), selected.end(), member)) {
      continue;
    }
    ++result.addedMechanisms;
    const SyncCoverGroundedMechanism &mechanism =
        instance.mechanisms[member];
    for (std::size_t index = 0; index < profileSize; ++index) {
      result.barrierActions[index] += mechanism.barrierActionProfile[index];
      result.eventActions[index] += mechanism.actionProfile[index];
    }
  }
  return result;
}

bool greedyCandidateLess(const GreedyCandidate &first,
                         const GreedyCandidate &second) {
  const auto ratioLess = [&](std::size_t firstValue,
                             std::size_t secondValue) {
    return static_cast<long double>(firstValue) /
               static_cast<long double>(first.newCoverage) <
           static_cast<long double>(secondValue) /
               static_cast<long double>(second.newCoverage);
  };
  for (std::size_t index = 0; index < first.barrierActions.size(); ++index) {
    if (ratioLess(first.barrierActions[index],
                  second.barrierActions[index])) {
      return true;
    }
    if (ratioLess(second.barrierActions[index],
                  first.barrierActions[index])) {
      return false;
    }
    if (ratioLess(first.eventActions[index], second.eventActions[index])) {
      return true;
    }
    if (ratioLess(second.eventActions[index], first.eventActions[index])) {
      return false;
    }
  }
  if (ratioLess(first.addedMechanisms, second.addedMechanisms)) {
    return true;
  }
  if (ratioLess(second.addedMechanisms, first.addedMechanisms)) {
    return false;
  }
  return first.column < second.column;
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

void greedySearch(const SyncCoverSelectionEvaluator &evaluator,
                  const SyncCoverGroundedInstance &instance,
                  const SyncCoverSelectionComponent &component,
                  const SyncCoverSolverOptions &options,
                  ComponentSearchResult &result) {
  std::vector<SyncCoverMechanismId> selected;
  std::optional<SearchEvaluation> evaluation =
      evaluateSelection(evaluator, instance, selected);
  ++result.evaluations;
  if (!evaluation) {
    return;
  }
  while (!componentCovered(instance, component, evaluation->covered)) {
    const std::optional<std::size_t> demand = mostConstrainedUncoveredDemand(
        instance, component, evaluation->covered);
    if (!demand) {
      return;
    }
    std::vector<GreedyCandidate> candidates;
    for (SyncCoverGroundedColumnId columnId :
         instance.demandColumns[*demand]) {
      GreedyCandidate candidate = makeGreedyCandidate(
          instance, component, selected, evaluation->covered, columnId);
      if (candidate.successor == selected || candidate.newCoverage == 0) {
        continue;
      }
      candidates.push_back(std::move(candidate));
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     greedyCandidateLess);
    bool advanced = false;
    for (GreedyCandidate &candidate : candidates) {
      if (result.boundedEvaluations >= options.evaluationLimit) {
        result.truncation.evaluationLimit = true;
        return;
      }
      ++result.boundedEvaluations;
      ++result.evaluations;
      std::optional<SearchEvaluation> successorEvaluation =
          evaluateSelection(evaluator, instance, candidate.successor);
      if (!successorEvaluation) {
        continue;
      }
      selected = std::move(candidate.successor);
      evaluation = std::move(successorEvaluation);
      advanced = true;
      break;
    }
    if (!advanced) {
      return;
    }
  }
  considerComplete(selected, evaluation->cost, result.selected, result.cost);
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
  const std::optional<std::size_t> demand = mostConstrainedUncoveredDemand(
      instance, component, evaluation->covered);
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
  greedySearch(evaluator, instance, component, options, result);
  if (component.exact) {
    std::set<std::vector<SyncCoverMechanismId>> seen;
    exactSearch(evaluator, instance, component, options, {}, seen, result);
    result.optimalityProven = !result.truncation;
    return result;
  }

  // Large components deliberately stop after deterministic greedy selection
  // and local deletion. Search operates only on grounded incidence and never
  // expands a beam of full resource evaluations.
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

std::vector<SyncCoverMechanismId> removalOrder(
    const SyncCoverGroundedInstance &instance,
    const std::vector<SyncCoverMechanismId> &selected) {
  std::vector<SyncCoverMechanismId> result = selected;
  std::stable_sort(result.begin(), result.end(), [&](SyncCoverMechanismId first,
                                                     SyncCoverMechanismId second) {
    const SyncCoverGroundedMechanism &firstMechanism =
        instance.mechanisms[first];
    const SyncCoverGroundedMechanism &secondMechanism =
        instance.mechanisms[second];
    if (firstMechanism.barrierActionProfile !=
        secondMechanism.barrierActionProfile) {
      return firstMechanism.barrierActionProfile >
             secondMechanism.barrierActionProfile;
    }
    if (firstMechanism.actionProfile != secondMechanism.actionProfile) {
      return firstMechanism.actionProfile > secondMechanism.actionProfile;
    }
    return first < second;
  });
  return result;
}

void addColumnCoverage(const SyncCoverGroundedColumn &column,
                       std::vector<std::size_t> &counts) {
  const std::vector<std::uint64_t> &words = column.coverage.getWords();
  for (std::size_t word = 0; word < words.size(); ++word) {
    std::uint64_t remaining = words[word];
    while (remaining != 0) {
      const unsigned bit = static_cast<unsigned>(__builtin_ctzll(remaining));
      const std::size_t demand = word * 64 + bit;
      if (demand < counts.size()) {
        ++counts[demand];
      }
      remaining &= remaining - 1;
    }
  }
}

bool canDisableColumns(const SyncCoverGroundedInstance &instance,
                       const std::vector<SyncCoverGroundedColumnId> &columns,
                       const std::vector<std::size_t> &coverageCounts,
                       std::vector<std::size_t> &removedCounts) {
  std::fill(removedCounts.begin(), removedCounts.end(), 0);
  for (SyncCoverGroundedColumnId columnId : columns) {
    const std::vector<std::uint64_t> &words =
        instance.columns[columnId].coverage.getWords();
    for (std::size_t word = 0; word < words.size(); ++word) {
      std::uint64_t remaining = words[word];
      while (remaining != 0) {
        const unsigned bit =
            static_cast<unsigned>(__builtin_ctzll(remaining));
        const std::size_t demand = word * 64 + bit;
        if (demand < removedCounts.size()) {
          ++removedCounts[demand];
        }
        remaining &= remaining - 1;
      }
    }
  }
  for (std::size_t demand = 0; demand < coverageCounts.size(); ++demand) {
    if (coverageCounts[demand] <= removedCounts[demand]) {
      return false;
    }
  }
  return true;
}

bool canImproveStructuralProfile(
    const SyncCoverGroundedInstance &instance,
    const std::vector<SyncCoverMechanismId> &candidate,
    const SyncCoverStructuralCost &incumbent) {
  std::vector<std::size_t> barrierActions(incumbent.barrierActionProfile.size(),
                                          0);
  std::vector<std::size_t> eventActions(incumbent.actionProfile.size(), 0);
  for (SyncCoverMechanismId mechanismId : candidate) {
    const SyncCoverGroundedMechanism &mechanism =
        instance.mechanisms[mechanismId];
    for (std::size_t index = 0; index < barrierActions.size(); ++index) {
      barrierActions[index] += mechanism.barrierActionProfile[index];
      eventActions[index] += mechanism.actionProfile[index];
    }
  }
  for (std::size_t index = 0; index < barrierActions.size(); ++index) {
    const auto candidateDepth =
        std::make_pair(barrierActions[index], eventActions[index]);
    const auto incumbentDepth =
        std::make_pair(incumbent.barrierActionProfile[index],
                       incumbent.actionProfile[index]);
    if (candidateDepth != incumbentDepth) {
      return candidateDepth < incumbentDepth;
    }
  }
  return candidate.size() < incumbent.mechanismCount;
}

void improveWithGroundedColumns(
    const SyncCoverSelectionEvaluator &evaluator,
    const SyncCoverGroundedInstance &instance,
    std::vector<SyncCoverMechanismId> &selected,
    SyncCoverStructuralCost &selectedCost, std::size_t &evaluations) {
  std::vector<std::vector<SyncCoverGroundedColumnId>> mechanismColumns(
      instance.mechanisms.size());
  for (const SyncCoverGroundedColumn &column : instance.columns) {
    for (SyncCoverMechanismId member : column.members) {
      mechanismColumns[member].push_back(column.id);
    }
  }

  std::set<std::vector<SyncCoverMechanismId>> memberSets;
  for (const SyncCoverGroundedColumn &column : instance.columns) {
    if (!column.members.empty()) {
      memberSets.insert(column.members);
    }
  }
  for (const std::vector<SyncCoverMechanismId> &members : memberSets) {
    std::vector<SyncCoverMechanismId> candidate =
        addMembers(selected, members);
    if (candidate == selected) {
      continue;
    }
    std::vector<bool> memberSelected(instance.mechanisms.size(), false);
    for (SyncCoverMechanismId mechanism : candidate) {
      memberSelected[mechanism] = true;
    }
    std::vector<bool> activeColumn(instance.columns.size(), false);
    std::vector<std::size_t> coverageCounts(instance.demands.size(), 0);
    for (const SyncCoverGroundedColumn &column : instance.columns) {
      const bool active = std::all_of(
          column.members.begin(), column.members.end(),
          [&](SyncCoverMechanismId member) { return memberSelected[member]; });
      activeColumn[column.id] = active;
      if (active) {
        addColumnCoverage(column, coverageCounts);
      }
    }
    if (std::any_of(coverageCounts.begin(), coverageCounts.end(),
                    [](std::size_t count) { return count == 0; })) {
      continue;
    }

    std::vector<std::size_t> removedCounts(instance.demands.size(), 0);
    for (SyncCoverMechanismId mechanism : removalOrder(instance, candidate)) {
      std::vector<SyncCoverGroundedColumnId> disabled;
      for (SyncCoverGroundedColumnId columnId :
           mechanismColumns[mechanism]) {
        if (activeColumn[columnId]) {
          disabled.push_back(columnId);
        }
      }
      if (!canDisableColumns(instance, disabled, coverageCounts,
                             removedCounts)) {
        continue;
      }
      memberSelected[mechanism] = false;
      candidate.erase(std::lower_bound(candidate.begin(), candidate.end(),
                                       mechanism));
      for (SyncCoverGroundedColumnId columnId : disabled) {
        activeColumn[columnId] = false;
      }
      for (std::size_t demand = 0; demand < coverageCounts.size(); ++demand) {
        coverageCounts[demand] -= removedCounts[demand];
      }
    }

    if (!canImproveStructuralProfile(instance, candidate, selectedCost)) {
      continue;
    }
    ++evaluations;
    const std::optional<SearchEvaluation> evaluation =
        evaluateSelection(evaluator, instance, candidate);
    if (evaluation && instance.coversAll(candidate) &&
        syncCoverStructuralCostLess(evaluation->cost, selectedCost)) {
      selected = std::move(candidate);
      selectedCost = evaluation->cost;
    }
  }
}

bool finalVerify(const SyncCoverMechanismUniverse &universe,
                 const std::vector<SyncCoverDemandId> &demands,
                 const std::vector<SyncCoverMechanismId> &selected,
                 SyncCoverResourceSelection &resources,
                 SyncCoverStructuralCost &cost,
                 SyncCoverCoverageStatistics &statistics,
                 std::optional<SyncCoverDemandId> &failedDemand) {
  const SyncCoverSelectionEvaluator evaluator(universe);
  const SyncCoverSelectionEvaluation evaluation = evaluator.evaluate(selected);
  if (!evaluation) {
    return false;
  }
  SyncCoverCoverageOracle oracle(universe.getGraph());
  std::set<DemandCoverageKey> verified;
  std::vector<SyncCoverDemandId> uniqueDemands;
  uniqueDemands.reserve(demands.size());
  for (SyncCoverDemandId demand : demands) {
    const SyncCoverDemand &requirement =
        universe.getGraph().getDemands()[demand];
    DemandCoverageKey key{requirement.source,
                          requirement.target,
                          requirement.scope,
                          requirement.distance,
                          requirement.sourceGuard.literals,
                          requirement.targetGuard.literals};
    if (!verified.insert(std::move(key)).second) {
      continue;
    }
    uniqueDemands.push_back(demand);
  }
  const std::vector<SyncCoverCoverageResult> coverages =
      oracle.checkDemandsCanonicalSelection(uniqueDemands, selected);
  if (coverages.size() != uniqueDemands.size()) {
    statistics = oracle.getStatistics();
    return false;
  }
  for (std::size_t index = 0; index < coverages.size(); ++index) {
    const SyncCoverCoverageResult &coverage = coverages[index];
    if (!coverage || !coverage.covered) {
      failedDemand = uniqueDemands[index];
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
    const SyncCoverSolverOptions &options,
    const std::vector<SyncCoverVerifiedFactoryColumn> &factoryColumns) {
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

  const SyncCoverGroundingResult grounding =
      groundSyncCoverInstance(universe, demands, factoryColumns);
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
  result.missingFactoryDemands = instance.demandsNeedingPricing;
  for (std::size_t local = 0; local < instance.demands.size(); ++local) {
    const bool eventCoverExists = std::any_of(
        instance.demandColumns[local].begin(),
        instance.demandColumns[local].end(),
        [&](SyncCoverGroundedColumnId columnId) {
          return std::all_of(
              instance.columns[columnId].members.begin(),
              instance.columns[columnId].members.end(),
              [&](SyncCoverMechanismId mechanism) {
                return instance.mechanisms[mechanism].kind !=
                       SyncCoverMechanismKind::Barrier;
              });
        });
    if (!eventCoverExists) {
      result.demandsWithoutEventColumn.push_back(instance.demands[local]);
    }
  }
  if (!result.missingFactoryDemands.empty()) {
    result.error = SyncCoverSelectionError::SearchIncomplete;
    result.optimalityProven = false;
    return result;
  }
  if (!instance.provenUncoverableDemands.empty()) {
    result.error = SyncCoverSelectionError::ProvenInfeasible;
    result.optimalityProven = true;
    return result;
  }
  result.components =
      buildComponents(instance, options.exactMechanismThreshold);
  result.optimalityProven = instance.demandsNeedingPricing.empty() &&
                            !instance.columnsTruncated;
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
      result.failedComponent = component.id;
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
  improveWithGroundedColumns(evaluator, instance, selected, selectedCost,
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

  result.mechanisms = selected;
  if (!finalVerify(universe, demands, result.mechanisms, result.resources,
                   selectedCost, result.finalVerificationStatistics,
                   result.failedFinalDemand)) {
    result.error = SyncCoverSelectionError::FinalVerificationFailed;
    result.optimalityProven = false;
    return result;
  }
  result.cost = std::move(selectedCost);
  return result;
}
