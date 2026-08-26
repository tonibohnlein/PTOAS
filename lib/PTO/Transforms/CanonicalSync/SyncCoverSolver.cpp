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

struct SearchEvaluation {
  SyncCoverDemandSet covered;
  SyncCoverStructuralCost cost;
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

std::vector<SyncCoverMechanismId>
addMembers(const std::vector<SyncCoverMechanismId> &selected,
           const std::vector<SyncCoverMechanismId> &members) {
  std::vector<SyncCoverMechanismId> result;
  result.reserve(selected.size() + members.size());
  std::set_union(selected.begin(), selected.end(), members.begin(),
                 members.end(), std::back_inserter(result));
  return result;
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

std::size_t countNewCoverage(const SyncCoverDemandSet &covered,
                             const SyncCoverDemandSet &candidate) {
  const std::vector<std::uint64_t> &coveredWords = covered.getWords();
  const std::vector<std::uint64_t> &candidateWords = candidate.getWords();
  std::size_t count = 0;
  for (std::size_t word = 0; word < candidateWords.size(); ++word) {
    const std::uint64_t already =
        word < coveredWords.size() ? coveredWords[word] : 0;
    count += static_cast<std::size_t>(
        __builtin_popcountll(candidateWords[word] & ~already));
  }
  return count;
}

GreedyCandidate makeGreedyCandidate(
    const SyncCoverGroundedInstance &instance,
    const std::vector<SyncCoverMechanismId> &selected,
    const SyncCoverDemandSet &covered,
    const SyncCoverDemandSet &eventCoverable,
    SyncCoverGroundedColumnId columnId) {
  const SyncCoverGroundedColumn &column = instance.columns[columnId];
  GreedyCandidate result;
  result.column = columnId;
  result.successor = addMembers(selected, column.members);
  const bool usesBarrier = std::any_of(
      column.members.begin(), column.members.end(),
      [&](SyncCoverMechanismId member) {
        return instance.mechanisms[member].kind ==
               SyncCoverMechanismKind::Barrier;
      });
  if (usesBarrier) {
    // A barrier's breadth only counts demands no event column can take;
    // coverage that events can supply must not justify a broad drain.
    SyncCoverDemandSet forced = column.coverage;
    forced.subtract(eventCoverable);
    result.newCoverage = countNewCoverage(covered, forced);
    if (result.newCoverage == 0) {
      result.newCoverage = 1;
    }
  } else {
    result.newCoverage = countNewCoverage(covered, column.coverage);
  }
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

struct GreedySelection {
  std::vector<SyncCoverMechanismId> selected;
  SyncCoverStructuralCost cost;
};

/// One deterministic greedy pass over the grounded coverage bitsets: anchor
/// each round on the first uncovered demand in instance order and adopt the
/// cheapest feasible column that covers it. Anchoring on the
/// most-constrained demand front-loads the demands whose only columns are
/// fat fallback barriers and shadows cheaper shared covers; ranking every
/// uncovered demand's columns instead degenerates under the per-depth ratio
/// order. Fallback barrier columns hold no event resources, so the pass
/// completes whenever every demand has a column.
std::optional<GreedySelection>
greedySelect(const SyncCoverSelectionEvaluator &evaluator,
             const SyncCoverGroundedInstance &instance,
             std::size_t &evaluations) {
  std::vector<SyncCoverMechanismId> selected;
  ++evaluations;
  std::optional<SearchEvaluation> evaluation =
      evaluateSelection(evaluator, instance, selected);
  if (!evaluation) {
    return std::nullopt;
  }
  SyncCoverDemandSet eventCoverable(instance.demands.size());
  for (const SyncCoverGroundedColumn &column : instance.columns) {
    const bool barrierFree = std::none_of(
        column.members.begin(), column.members.end(),
        [&](SyncCoverMechanismId member) {
          return instance.mechanisms[member].kind ==
                 SyncCoverMechanismKind::Barrier;
        });
    if (barrierFree) {
      eventCoverable.unite(column.coverage);
    }
  }
  while (evaluation->covered.count() != instance.demands.size()) {
    std::optional<std::size_t> demand;
    for (std::size_t local = 0; local < instance.demands.size(); ++local) {
      if (!evaluation->covered.contains(local)) {
        demand = local;
        break;
      }
    }
    if (!demand) {
      return std::nullopt;
    }
    std::vector<GreedyCandidate> candidates;
    for (SyncCoverGroundedColumnId columnId :
         instance.demandColumns[*demand]) {
      GreedyCandidate candidate = makeGreedyCandidate(
          instance, selected, evaluation->covered, eventCoverable, columnId);
      if (candidate.successor == selected || candidate.newCoverage == 0) {
        continue;
      }
      candidates.push_back(std::move(candidate));
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     greedyCandidateLess);
    bool advanced = false;
    for (GreedyCandidate &candidate : candidates) {
      ++evaluations;
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
      return std::nullopt;
    }
  }
  return GreedySelection{std::move(selected), evaluation->cost};
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
    const bool improves =
        evaluation && evaluation->covered.count() == instance.demands.size() &&
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
    if (evaluation &&
        evaluation->covered.count() == instance.demands.size() &&
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
    if (evaluation->covered.count() == instance.demands.size()) {
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
    return result;
  }
  std::optional<GreedySelection> greedy =
      greedySelect(evaluator, instance, result.evaluations);
  if (!greedy) {
    result.error = SyncCoverSelectionError::SearchIncomplete;
    return result;
  }
  std::vector<SyncCoverMechanismId> selected = std::move(greedy->selected);
  SyncCoverStructuralCost selectedCost = greedy->cost;
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
    return result;
  }
  result.finalVerificationStatistics = finalOracle.getStatistics();
  result.cost = std::move(selectedCost);
  return result;
}
