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

std::optional<std::size_t>
mostConstrainedUncoveredDemand(const SyncCoverGroundedInstance &instance,
                               const SyncCoverSelectionComponent &component,
                               const SyncCoverDemandSet &covered) {
  std::optional<std::size_t> best;
  std::size_t bestColumns = 0;
  for (SyncCoverDemandId demand : component.demands) {
    const std::size_t local = localDemand(instance, demand);
    if (covered.contains(local)) {
      continue;
    }
    const std::size_t columns = instance.demandColumns[local].size();
    if (!best || columns < bestColumns) {
      best = local;
      bestColumns = columns;
    }
  }
  return best;
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
    // Anchor each round on the first uncovered demand in component order.
    // Anchoring on the most-constrained demand front-loads the demands
    // whose only columns are fat fallback barriers and shadows cheaper
    // shared covers; ranking every uncovered demand's columns instead
    // degenerates under the per-depth ratio order (top-level one-demand
    // columns always beat in-loop bundles) and exhausts the evaluation
    // budget. A principled global-density greedy needs a scalar density.
    const std::optional<std::size_t> demand =
        firstUncoveredDemand(instance, component, evaluation->covered);
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

ComponentSearchResult searchComponent(
    const SyncCoverSelectionEvaluator &evaluator,
    const SyncCoverGroundedInstance &instance,
    const SyncCoverSelectionComponent &component,
    const SyncCoverSolverOptions &options) {
  ComponentSearchResult result;
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

std::vector<SyncCoverDemandId>
uniqueCoverageDemands(const SyncCoverMechanismUniverse &universe,
                      const std::vector<SyncCoverDemandId> &demands) {
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
    if (verified.insert(std::move(key)).second) {
      uniqueDemands.push_back(demand);
    }
  }
  return uniqueDemands;
}

bool oracleCoversAll(const SyncCoverCoverageOracle &oracle,
                     const std::vector<SyncCoverDemandId> &uniqueDemands,
                     const std::vector<SyncCoverMechanismId> &selected,
                     std::optional<SyncCoverDemandId> *failedDemand = nullptr) {
  const std::vector<SyncCoverCoverageResult> coverages =
      oracle.checkDemandsCanonicalSelection(uniqueDemands, selected);
  if (coverages.size() != uniqueDemands.size()) {
    return false;
  }
  for (std::size_t index = 0; index < coverages.size(); ++index) {
    const SyncCoverCoverageResult &coverage = coverages[index];
    if (!coverage || !coverage.covered) {
      if (failedDemand) {
        *failedDemand = uniqueDemands[index];
      }
      return false;
    }
  }
  return true;
}

/// Grounded bitsets under-approximate coverage: transitive covers absent
/// from the column universe are invisible to removeRedundant. Re-check the
/// rejected deletions of the final, small selection against the oracle so a
/// mechanism whose demands are covered through paths outside the grounded
/// columns can still be dropped.
void oracleRemoveRedundant(const SyncCoverSelectionEvaluator &evaluator,
                           const SyncCoverGroundedInstance &instance,
                           const SyncCoverCoverageOracle &oracle,
                           const std::vector<SyncCoverDemandId> &uniqueDemands,
                           const SyncCoverSolverOptions &options,
                           std::vector<SyncCoverMechanismId> &selected,
                           SyncCoverStructuralCost &cost,
                           std::size_t &evaluations,
                           std::size_t &oracleChecks) {
  for (std::size_t index = selected.size(); index > 0; --index) {
    std::vector<SyncCoverMechanismId> candidate = selected;
    candidate.erase(candidate.begin() + static_cast<std::ptrdiff_t>(index - 1));
    ++evaluations;
    const std::optional<SearchEvaluation> evaluation =
        evaluateSelection(evaluator, instance, candidate);
    const bool improves =
        evaluation && syncCoverStructuralCostLess(evaluation->cost, cost);
    if (!improves) {
      continue;
    }
    if (instance.coversAll(candidate)) {
      selected = std::move(candidate);
      cost = evaluation->cost;
      continue;
    }
    // The grounded bitsets rejected the deletion; one bounded oracle pass
    // can still prove it covered through paths outside the column universe.
    if (oracleChecks == options.oracleRedundancyLimit) {
      continue;
    }
    ++oracleChecks;
    if (oracleCoversAll(oracle, uniqueDemands, candidate)) {
      selected = std::move(candidate);
      cost = evaluation->cost;
    }
  }
}

/// Composition columns are not enumerated at grounding. Price them lazily,
/// only for demands whose every selected cover uses a barrier (the cost
/// order already treats barriers as the last resort): search frontier
/// witnesses for such a demand, then exchange the superseded barriers for
/// the witness cover. Every drop is justified constructively — each demand
/// must keep a selected grounded column or an independently verified
/// witness cover — so the exchange never queries the oracle; the frontier
/// search is the only oracle work, bounded by the pricing options. Pricing
/// never fails the solve (an unpriceable demand just keeps its barrier).
void lazyPriceBarrierCovered(const SyncCoverSelectionEvaluator &evaluator,
                             const SyncCoverGroundedInstance &instance,
                             const SyncCoverMechanismUniverse &universe,
                             const SyncCoverCoverageOracle &oracle,
                             const SyncCoverSolverOptions &options,
                             std::vector<SyncCoverMechanismId> &selected,
                             SyncCoverStructuralCost &cost,
                             SyncCoverSelectionResult &result) {
  const auto isBarrier = [&](SyncCoverMechanismId mechanism) {
    return universe.getMechanisms()[mechanism].kind ==
           SyncCoverMechanismKind::Barrier;
  };
  const auto barrierOnlyDemands = [&]() {
    SyncCoverDemandSet coveredBarrierFree(instance.demands.size());
    for (const SyncCoverGroundedColumn &column : instance.columns) {
      const bool columnSelected = std::all_of(
          column.members.begin(), column.members.end(),
          [&](SyncCoverMechanismId member) {
            return std::binary_search(selected.begin(), selected.end(),
                                      member);
          });
      if (!columnSelected) {
        continue;
      }
      const bool usesBarrier =
          std::any_of(column.members.begin(), column.members.end(), isBarrier);
      if (!usesBarrier) {
        coveredBarrierFree.unite(column.coverage);
      }
    }
    std::vector<std::size_t> demandsOnBarriers;
    for (std::size_t local = 0; local < instance.demands.size(); ++local) {
      if (!coveredBarrierFree.contains(local)) {
        demandsOnBarriers.push_back(local);
      }
    }
    return demandsOnBarriers;
  };

  std::vector<std::size_t> candidates = barrierOnlyDemands();
  if (candidates.empty()) {
    return;
  }
  const SyncCoverGraph &graph = universe.getGraph();
  std::stable_sort(candidates.begin(), candidates.end(),
                   [&](std::size_t first, std::size_t second) {
                     const auto depth = [&](std::size_t local) {
                       return graph
                           .getScopeLoopDepth(
                               graph.getDemands()[instance.demands[local]]
                                   .scope)
                           .value_or(0);
                     };
                     return depth(first) > depth(second);
                   });

  // Every witness the frontier search returns is an independently verified
  // coverage fact for its demand; the pool persists across exchanges so a
  // later exchange can lean on an earlier demand's witnesses.
  struct VerifiedCover {
    std::size_t local = 0;
    std::vector<SyncCoverMechanismId> members;
  };
  std::vector<VerifiedCover> verified;

  struct ExchangeOutcome {
    std::vector<SyncCoverMechanismId> selection;
    SyncCoverStructuralCost cost;
  };
  // Add the cover to the selection, then drop every superseded barrier whose
  // removal keeps each demand justified by a selected grounded column or a
  // verified witness cover — pure bitset-and-count arithmetic, no oracle.
  const auto tryExchange = [&](const std::vector<SyncCoverMechanismId> &cover,
                               const std::vector<std::size_t> &pricedLocals)
      -> std::optional<ExchangeOutcome> {
    std::vector<SyncCoverMechanismId> candidate = selected;
    candidate.insert(candidate.end(), cover.begin(), cover.end());
    std::sort(candidate.begin(), candidate.end());
    candidate.erase(std::unique(candidate.begin(), candidate.end()),
                    candidate.end());
    const auto inCandidate = [&](SyncCoverMechanismId mechanism) {
      return std::binary_search(candidate.begin(), candidate.end(),
                                mechanism);
    };
    // Justification counts per demand over the columns and witnesses whose
    // members all sit inside the candidate.
    std::vector<std::size_t> support(instance.demands.size(), 0);
    std::vector<signed char> columnIncluded(instance.columns.size(), -1);
    std::vector<SyncCoverGroundedColumnId> includedColumns;
    std::vector<std::vector<std::size_t>> columnDemands(
        instance.columns.size());
    for (std::size_t local = 0; local < instance.demands.size(); ++local) {
      for (SyncCoverGroundedColumnId columnId :
           instance.demandColumns[local]) {
        signed char &memo = columnIncluded[columnId];
        if (memo < 0) {
          const auto &members = instance.columns[columnId].members;
          memo = std::all_of(members.begin(), members.end(), inCandidate)
                     ? 1
                     : 0;
          if (memo == 1) {
            includedColumns.push_back(columnId);
          }
        }
        if (memo == 1) {
          ++support[local];
          columnDemands[columnId].push_back(local);
        }
      }
    }
    std::vector<std::size_t> includedWitnesses;
    for (std::size_t index = 0; index < verified.size(); ++index) {
      const VerifiedCover &witness = verified[index];
      if (std::all_of(witness.members.begin(), witness.members.end(),
                      inCandidate)) {
        ++support[witness.local];
        includedWitnesses.push_back(index);
      }
    }

    // Only barriers appearing in the priced demands' selected covers are
    // superseded; other barriers belong to the redundancy passes.
    std::vector<SyncCoverMechanismId> drops;
    for (std::size_t local : pricedLocals) {
      for (SyncCoverGroundedColumnId columnId :
           instance.demandColumns[local]) {
        if (columnIncluded[columnId] != 1) {
          continue;
        }
        for (SyncCoverMechanismId member :
             instance.columns[columnId].members) {
          if (isBarrier(member)) {
            drops.push_back(member);
          }
        }
      }
    }
    std::sort(drops.begin(), drops.end());
    drops.erase(std::unique(drops.begin(), drops.end()), drops.end());

    std::vector<signed char> columnDropped(instance.columns.size(), 0);
    std::vector<signed char> witnessDropped(verified.size(), 0);
    bool dropped = false;
    for (SyncCoverMechanismId barrier : drops) {
      std::vector<std::size_t> decremented;
      std::vector<SyncCoverGroundedColumnId> droppedColumns;
      std::vector<std::size_t> droppedWitnesses;
      for (SyncCoverGroundedColumnId columnId : includedColumns) {
        if (columnDropped[columnId]) {
          continue;
        }
        const auto &members = instance.columns[columnId].members;
        if (std::find(members.begin(), members.end(), barrier) ==
            members.end()) {
          continue;
        }
        droppedColumns.push_back(columnId);
        for (std::size_t local : columnDemands[columnId]) {
          --support[local];
          decremented.push_back(local);
        }
      }
      for (std::size_t index : includedWitnesses) {
        if (witnessDropped[index]) {
          continue;
        }
        const VerifiedCover &witness = verified[index];
        if (std::find(witness.members.begin(), witness.members.end(),
                      barrier) == witness.members.end()) {
          continue;
        }
        droppedWitnesses.push_back(index);
        --support[witness.local];
        decremented.push_back(witness.local);
      }
      const bool justified =
          std::all_of(decremented.begin(), decremented.end(),
                      [&](std::size_t local) { return support[local] != 0; });
      if (justified) {
        candidate.erase(std::lower_bound(candidate.begin(), candidate.end(),
                                         barrier));
        for (SyncCoverGroundedColumnId columnId : droppedColumns) {
          columnDropped[columnId] = 1;
        }
        for (std::size_t index : droppedWitnesses) {
          witnessDropped[index] = 1;
        }
        dropped = true;
      } else {
        for (std::size_t local : decremented) {
          ++support[local];
        }
      }
    }

    if (!dropped && candidate == selected) {
      return std::nullopt;
    }
    const std::optional<SearchEvaluation> evaluation =
        evaluateSelection(evaluator, instance, candidate);
    ++result.redundancyEvaluations;
    if (!evaluation) {
      return std::nullopt;
    }
    return ExchangeOutcome{std::move(candidate), evaluation->cost};
  };

  std::vector<SyncCoverMechanismId> combinedCover;
  std::vector<std::size_t> combinedLocals;
  std::size_t priced = 0;
  for (std::size_t local : candidates) {
    if (priced == options.pricingDemandLimit) {
      break;
    }
    ++priced;
    ++result.pricedDemands;
    const SyncCoverFactoryWitnessResult witnesses =
        oracle.getFactoryMechanismWitnesses(instance.demands[local],
                                            universe.getMechanisms().size(),
                                            options.pricingPairLimit);
    if (!witnesses) {
      continue;
    }
    // A cover leaning on a selected barrier re-justifies the demand through
    // that barrier and can never help drop it; only barrier-free covers and
    // narrower unselected barriers are exchange candidates. Every witness
    // still enters the verified pool.
    const auto reusesSelectedBarrier =
        [&](const std::vector<SyncCoverMechanismId> &members) {
          return std::any_of(members.begin(), members.end(),
                             [&](SyncCoverMechanismId member) {
                               return isBarrier(member) &&
                                      std::binary_search(selected.begin(),
                                                         selected.end(),
                                                         member);
                             });
        };
    std::vector<std::vector<SyncCoverMechanismId>> covers;
    for (SyncCoverMechanismId singleton : witnesses.singletons) {
      verified.push_back({local, {singleton}});
      if (!reusesSelectedBarrier({singleton})) {
        covers.push_back({singleton});
      }
    }
    for (const std::vector<SyncCoverMechanismId> &pair : witnesses.pairs) {
      verified.push_back({local, pair});
      if (!reusesSelectedBarrier(pair)) {
        covers.push_back(pair);
      }
    }
    // Covers reusing already-selected mechanisms come first: they add the
    // fewest actions for the same coverage.
    const auto addedMembers = [&](const std::vector<SyncCoverMechanismId>
                                      &members) {
      return std::count_if(members.begin(), members.end(),
                           [&](SyncCoverMechanismId member) {
                             return !std::binary_search(selected.begin(),
                                                        selected.end(),
                                                        member);
                           });
    };
    std::stable_sort(covers.begin(), covers.end(),
                     [&](const std::vector<SyncCoverMechanismId> &first,
                         const std::vector<SyncCoverMechanismId> &second) {
                       return addedMembers(first) < addedMembers(second);
                     });
    std::optional<ExchangeOutcome> best;
    std::size_t attempted = 0;
    for (const std::vector<SyncCoverMechanismId> &cover : covers) {
      if (attempted == options.pricingCoverLimit) {
        break;
      }
      ++attempted;
      std::optional<ExchangeOutcome> outcome = tryExchange(cover, {local});
      if (outcome &&
          syncCoverStructuralCostLess(outcome->cost,
                                      best ? best->cost : cost)) {
        best = std::move(outcome);
      }
    }
    if (best) {
      selected = std::move(best->selection);
      cost = best->cost;
      ++result.pricedImprovements;
    } else if (!covers.empty()) {
      // A barrier covering several priced demands cannot be dropped by any
      // single demand's exchange; remember the cheapest cover for one
      // combined attempt after the loop.
      combinedCover.insert(combinedCover.end(), covers.front().begin(),
                           covers.front().end());
      combinedLocals.push_back(local);
    }
  }
  if (!combinedLocals.empty()) {
    std::optional<ExchangeOutcome> outcome =
        tryExchange(combinedCover, combinedLocals);
    if (outcome && syncCoverStructuralCostLess(outcome->cost, cost)) {
      selected = std::move(outcome->selection);
      cost = outcome->cost;
      ++result.pricedImprovements;
    }
  }

  // A verified barrier-free witness settles the demand's event coverage the
  // same way an eager composite column would have.
  if (!result.demandsWithoutEventColumn.empty() && !verified.empty()) {
    const auto eventWitnessed = [&](SyncCoverDemandId demand) {
      return std::any_of(
          verified.begin(), verified.end(), [&](const VerifiedCover &witness) {
            return instance.demands[witness.local] == demand &&
                   std::none_of(witness.members.begin(),
                                witness.members.end(), isBarrier);
          });
    };
    auto &uncovered = result.demandsWithoutEventColumn;
    uncovered.erase(
        std::remove_if(uncovered.begin(), uncovered.end(), eventWitnessed),
        uncovered.end());
  }
}

bool finalVerify(const SyncCoverMechanismUniverse &universe,
                 const SyncCoverCoverageOracle &oracle,
                 const std::vector<SyncCoverDemandId> &uniqueDemands,
                 const std::vector<SyncCoverMechanismId> &selected,
                 SyncCoverResourceSelection &resources,
                 SyncCoverStructuralCost &cost,
                 std::optional<SyncCoverDemandId> &failedDemand) {
  const SyncCoverSelectionEvaluator evaluator(universe);
  const SyncCoverSelectionEvaluation evaluation = evaluator.evaluate(selected);
  if (!evaluation) {
    return false;
  }
  if (!oracleCoversAll(oracle, uniqueDemands, selected, &failedDemand)) {
    return false;
  }
  resources = evaluation.resources;
  cost = evaluation.cost;
  return true;
}

} // namespace

SyncCoverSelectionResult mlir::pto::solveSyncCoverSelection(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
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
  const SyncCoverGroundingResult grounding =
      groundSyncCoverInstance(universe, demands, factoryColumns);
  if (!grounding) {
    SyncCoverSelectionResult result =
        makeError(SyncCoverSelectionError::GroundingFailed);
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
  result.components =
      buildComponents(instance, options.exactMechanismThreshold);
  result.optimalityProven = instance.demandsNeedingPricing.empty() &&
                            !instance.columnsTruncated &&
                            !instance.pricingRestricted;
  std::vector<SyncCoverMechanismId> selected;
  for (const SyncCoverSelectionComponent &component : result.components) {
    ComponentSearchResult componentResult =
        searchComponent(evaluator, instance, component, options);
    result.evaluations += componentResult.evaluations;
    result.truncation.evaluationLimit |=
        componentResult.truncation.evaluationLimit;
    result.optimalityProven &= componentResult.optimalityProven;
    if (!componentResult.selected) {
      // Safety net for evaluation-budget truncation ONLY: cover every demand
      // in the component with an all-barrier column (barriers hold no event
      // IDs, so this is always resource-feasible). A component search that
      // fails without hitting its budget indicates a solver defect and must
      // stay a hard failure rather than be masked by a barrier plan.
      if (!componentResult.truncation.evaluationLimit) {
        result.error = SyncCoverSelectionError::SearchIncomplete;
        result.optimalityProven = false;
        result.failedComponent = component.id;
        return result;
      }
      std::vector<SyncCoverMechanismId> fallback;
      bool rescued = true;
      for (SyncCoverDemandId demand : component.demands) {
        const std::size_t local = localDemand(instance, demand);
        const auto barrierOnly = [&](SyncCoverGroundedColumnId columnId) {
          const SyncCoverGroundedColumn &column = instance.columns[columnId];
          return std::all_of(column.members.begin(), column.members.end(),
                             [&](SyncCoverMechanismId member) {
                               return instance.mechanisms[member].kind ==
                                      SyncCoverMechanismKind::Barrier;
                             });
        };
        const auto column =
            std::find_if(instance.demandColumns[local].begin(),
                         instance.demandColumns[local].end(), barrierOnly);
        if (column == instance.demandColumns[local].end()) {
          rescued = false;
          break;
        }
        const SyncCoverGroundedColumn &members = instance.columns[*column];
        fallback.insert(fallback.end(), members.members.begin(),
                        members.members.end());
      }
      if (!rescued) {
        result.error = SyncCoverSelectionError::SearchIncomplete;
        result.optimalityProven = false;
        result.failedComponent = component.id;
        return result;
      }
      result.optimalityProven = false;
      ++result.rescuedComponents;
      selected.insert(selected.end(), fallback.begin(), fallback.end());
      continue;
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
  // The post-search oracle is prepared fresh from the graph, independent of
  // the grounded bitsets the search consumed: both the oracle-checked
  // redundancy pass and the final whole-plan verification see topologies the
  // search never touched.
  SyncCoverCoverageOracle finalOracle(universe.getGraph());
  const std::vector<SyncCoverDemandId> uniqueDemands =
      uniqueCoverageDemands(universe, demands);
  lazyPriceBarrierCovered(evaluator, instance, universe, finalOracle, options,
                          selected, selectedCost, result);
  oracleRemoveRedundant(evaluator, instance, finalOracle, uniqueDemands,
                        options, selected, selectedCost,
                        result.redundancyEvaluations,
                        result.oracleRedundancyChecks);

  result.mechanisms = selected;
  if (!finalVerify(universe, finalOracle, uniqueDemands, result.mechanisms,
                   result.resources, selectedCost,
                   result.failedFinalDemand)) {
    result.finalVerificationStatistics = finalOracle.getStatistics();
    result.error = SyncCoverSelectionError::FinalVerificationFailed;
    result.optimalityProven = false;
    return result;
  }
  result.finalVerificationStatistics = finalOracle.getStatistics();
  result.cost = std::move(selectedCost);
  return result;
}
