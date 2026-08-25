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
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_internal;

namespace {

struct EvictionPlan {
  std::vector<SyncCoverMechanismId> evicted;
  std::vector<SyncCoverMechanismId> retained;
  SyncCoverStructuralCost retainedCost;
};

struct ExchangeCandidate {
  std::vector<SyncCoverMechanismId> selected;
  SyncCoverStructuralCost cost;
};

struct ExchangeCandidatePool {
  bool valid = true;
  bool truncated = false;
  std::vector<SyncCoverMechanismId> mechanisms;
};

struct RankedExchangeMechanism {
  SyncCoverMechanismId mechanism = 0;
  std::size_t cutSupport = 0;
  std::size_t topologySupport = 0;
  SyncCoverStructuralCost standaloneCost;
};

std::vector<SyncCoverMechanismId> subtractSorted(
    const std::vector<SyncCoverMechanismId> &selected,
    const std::vector<SyncCoverMechanismId> &removed) {
  std::vector<SyncCoverMechanismId> result;
  std::set_difference(selected.begin(), selected.end(), removed.begin(),
                      removed.end(), std::back_inserter(result));
  return result;
}

std::vector<SyncCoverMechanismId> mergeSorted(
    const std::vector<SyncCoverMechanismId> &retained,
    const std::vector<SyncCoverMechanismId> &added) {
  std::vector<SyncCoverMechanismId> result;
  std::set_union(retained.begin(), retained.end(), added.begin(), added.end(),
                 std::back_inserter(result));
  return result;
}

bool candidateLess(const ExchangeCandidate &first,
                   const ExchangeCandidate &second) {
  if (syncCoverStructuralCostLess(first.cost, second.cost)) {
    return true;
  }
  if (syncCoverStructuralCostLess(second.cost, first.cost)) {
    return false;
  }
  return first.selected < second.selected;
}

bool evictionLess(const EvictionPlan &first, const EvictionPlan &second) {
  if (syncCoverStructuralCostLess(first.retainedCost, second.retainedCost)) {
    return true;
  }
  if (syncCoverStructuralCostLess(second.retainedCost, first.retainedCost)) {
    return false;
  }
  return first.evicted < second.evicted;
}

bool rankedMechanismLess(const RankedExchangeMechanism &first,
                         const RankedExchangeMechanism &second) {
  if (first.cutSupport != second.cutSupport) {
    return first.cutSupport > second.cutSupport;
  }
  if (first.topologySupport != second.topologySupport) {
    return first.topologySupport > second.topologySupport;
  }
  if (syncCoverStructuralCostLess(first.standaloneCost,
                                  second.standaloneCost)) {
    return true;
  }
  if (syncCoverStructuralCostLess(second.standaloneCost,
                                  first.standaloneCost)) {
    return false;
  }
  return first.mechanism < second.mechanism;
}

ExchangeCandidatePool buildCandidatePool(
    const SyncCoverMechanismUniverse &universe,
    SyncCoverCoverageOracle &oracle,
    const CoverageEvaluation &affected,
    const std::vector<SyncCoverMechanismId> &retained,
    const std::vector<SyncCoverStructuralCost> &standaloneCosts,
    std::size_t candidateLimit) {
  ExchangeCandidatePool result;
  std::map<SyncCoverMechanismId, std::pair<std::size_t, std::size_t>> support;
  for (const auto &[demand, cut] : affected.cuts) {
    (void)demand;
    for (SyncCoverMechanismId mechanism : cut) {
      ++support[mechanism].first;
    }
  }
  for (SyncCoverDemandId demand : affected.uncovered) {
    const SyncCoverDemandTopologyResult topology =
        oracle.getDemandTopology(demand);
    if (!topology) {
      result.valid = false;
      return result;
    }
    for (SyncCoverMechanismId mechanism : topology.potentialMechanisms) {
      ++support[mechanism].second;
    }
  }

  std::vector<RankedExchangeMechanism> ranked;
  ranked.reserve(support.size());
  for (const auto &[mechanism, counts] : support) {
    if (mechanism >= universe.getMechanisms().size() ||
        std::binary_search(retained.begin(), retained.end(), mechanism)) {
      continue;
    }
    ranked.push_back(
        {mechanism, counts.first, counts.second, standaloneCosts[mechanism]});
  }
  std::sort(ranked.begin(), ranked.end(), rankedMechanismLess);

  std::set<SyncCoverDemandId> uncoveredCuts;
  for (const auto &[demand, cut] : affected.cuts) {
    if (!cut.empty()) {
      uncoveredCuts.insert(demand);
    }
  }
  std::set<SyncCoverMechanismId> selectedPool;
  while (!uncoveredCuts.empty() && result.mechanisms.size() < candidateLimit) {
    std::optional<SyncCoverMechanismId> best;
    std::size_t bestCoverage = 0;
    for (const RankedExchangeMechanism &entry : ranked) {
      if (selectedPool.count(entry.mechanism) != 0) {
        continue;
      }
      std::size_t coverage = 0;
      for (SyncCoverDemandId demand : uncoveredCuts) {
        const auto cut = affected.cuts.find(demand);
        if (cut != affected.cuts.end() &&
            std::binary_search(cut->second.begin(), cut->second.end(),
                               entry.mechanism)) {
          ++coverage;
        }
      }
      if (coverage > bestCoverage) {
        best = entry.mechanism;
        bestCoverage = coverage;
      }
    }
    if (!best || bestCoverage == 0) {
      break;
    }
    selectedPool.insert(*best);
    result.mechanisms.push_back(*best);
    for (auto demand = uncoveredCuts.begin(); demand != uncoveredCuts.end();) {
      const auto cut = affected.cuts.find(*demand);
      if (cut != affected.cuts.end() &&
          std::binary_search(cut->second.begin(), cut->second.end(), *best)) {
        demand = uncoveredCuts.erase(demand);
      } else {
        ++demand;
      }
    }
  }
  for (const RankedExchangeMechanism &entry : ranked) {
    if (result.mechanisms.size() == candidateLimit) {
      break;
    }
    if (selectedPool.insert(entry.mechanism).second) {
      result.mechanisms.push_back(entry.mechanism);
    }
  }
  result.truncated = ranked.size() > result.mechanisms.size() ||
                     !uncoveredCuts.empty();
  return result;
}

class ExchangeSearch {
public:
  ExchangeSearch(const SyncCoverSelectionEvaluator &selectionEvaluator,
                 CoverageEvaluator &coverage,
                 const std::vector<SyncCoverDemandId> &activeDemands,
                 const SyncCoverStructuralCost &incumbentCost,
                 const SyncCoverSolverOptions &options,
                 SyncCoverExchangeStatistics &statistics,
                 std::size_t &redundancyEvaluations,
                 bool &evaluationLimitReached)
      : selectionEvaluator_(selectionEvaluator), coverage_(coverage),
        activeDemands_(activeDemands), incumbentCost_(incumbentCost),
        options_(options), statistics_(statistics),
        redundancyEvaluations_(redundancyEvaluations),
        evaluationLimitReached_(evaluationLimitReached) {}

  std::optional<ExchangeCandidate>
  run(const EvictionPlan &eviction,
      const std::vector<SyncCoverDemandId> &affectedDemands,
      const std::vector<SyncCoverMechanismId> &candidatePool,
      std::size_t evaluationAllowance) {
    retained_ = &eviction.retained;
    affectedDemands_ = &affectedDemands;
    candidatePool_ = &candidatePool;
    seen_.clear();
    best_.reset();
    localEvaluationLimit_ =
        std::min(options_.exchangeEvaluationLimit,
                 statistics_.evaluations + evaluationAllowance);
    localLimitReached_ = false;
    search({});
    evaluationLimitReached_ |= localLimitReached_;
    return best_;
  }

private:
  void reduceCompleteSelection(std::vector<SyncCoverMechanismId> &selected,
                               SyncCoverStructuralCost &cost) {
    while (!selected.empty() && !localLimitReached_) {
      std::optional<std::vector<SyncCoverMechanismId>> bestSelection;
      std::optional<SyncCoverStructuralCost> bestCost;
      for (std::size_t index = 0; index < selected.size(); ++index) {
        if (statistics_.evaluations == localEvaluationLimit_) {
          localLimitReached_ = true;
          break;
        }
        std::vector<SyncCoverMechanismId> candidate = selected;
        candidate.erase(candidate.begin() + index);
        ++statistics_.evaluations;
        ++redundancyEvaluations_;
        SyncCoverStructuralCost candidateCost;
        const bool complete = evaluateCompleteSelection(
            selectionEvaluator_, coverage_, activeDemands_, candidate,
            candidateCost);
        if (!complete ||
            !syncCoverStructuralCostLess(candidateCost, cost)) {
          continue;
        }
        if (!bestCost ||
            syncCoverStructuralCostLess(candidateCost, *bestCost)) {
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

  void search(std::vector<SyncCoverMechanismId> added) {
    if (localLimitReached_ || !seen_.insert(added).second) {
      return;
    }
    if (statistics_.evaluations == localEvaluationLimit_) {
      localLimitReached_ = true;
      return;
    }
    ++statistics_.evaluations;
    const std::vector<SyncCoverMechanismId> selected =
        mergeSorted(*retained_, added);
    const SyncCoverSelectionEvaluation selection =
        selectionEvaluator_.evaluate(selected);
    if (!selection || !selection.resources.resourceFeasible ||
        !syncCoverStructuralCostLess(selection.cost, incumbentCost_)) {
      return;
    }

    const CoverageEvaluation evaluation =
        coverage_.evaluate(*affectedDemands_, selected);
    if (!evaluation.valid) {
      return;
    }
    if (evaluation.uncovered.empty()) {
      if (statistics_.evaluations == localEvaluationLimit_) {
        localLimitReached_ = true;
        return;
      }
      ++statistics_.evaluations;
      std::vector<SyncCoverMechanismId> completeSelection = selected;
      SyncCoverStructuralCost completeCost;
      const bool complete = evaluateCompleteSelection(
          selectionEvaluator_, coverage_, activeDemands_, completeSelection,
          completeCost);
      if (!complete) {
        return;
      }
      reduceCompleteSelection(completeSelection, completeCost);
      ExchangeCandidate candidate{std::move(completeSelection),
                                  std::move(completeCost)};
      if (syncCoverStructuralCostLess(candidate.cost, incumbentCost_) &&
          (!best_ || candidateLess(candidate, *best_))) {
        best_ = std::move(candidate);
      }
      return;
    }

    std::optional<std::pair<std::size_t, SyncCoverDemandId>> bestCut;
    std::vector<SyncCoverMechanismId> branch;
    for (SyncCoverDemandId demand : evaluation.uncovered) {
      auto cut = evaluation.cuts.find(demand);
      if (cut == evaluation.cuts.end()) {
        return;
      }
      std::vector<SyncCoverMechanismId> candidates;
      for (SyncCoverMechanismId mechanism : cut->second) {
        if (std::find(candidatePool_->begin(), candidatePool_->end(),
                      mechanism) != candidatePool_->end() &&
            !std::binary_search(selected.begin(), selected.end(), mechanism)) {
          candidates.push_back(mechanism);
        }
      }
      std::stable_sort(candidates.begin(), candidates.end(),
                       [&](SyncCoverMechanismId first,
                           SyncCoverMechanismId second) {
                         return std::find(candidatePool_->begin(),
                                          candidatePool_->end(), first) <
                                std::find(candidatePool_->begin(),
                                          candidatePool_->end(), second);
                       });
      const auto key = std::make_pair(candidates.size(), demand);
      if (!bestCut || key < *bestCut) {
        bestCut = key;
        branch = std::move(candidates);
      }
    }
    if (!bestCut || branch.empty()) {
      return;
    }
    for (SyncCoverMechanismId mechanism : branch) {
      std::vector<SyncCoverMechanismId> successor = added;
      successor.insert(
          std::lower_bound(successor.begin(), successor.end(), mechanism),
          mechanism);
      search(std::move(successor));
      if (localLimitReached_) {
        return;
      }
    }
  }

  const SyncCoverSelectionEvaluator &selectionEvaluator_;
  CoverageEvaluator &coverage_;
  const std::vector<SyncCoverDemandId> &activeDemands_;
  const SyncCoverStructuralCost &incumbentCost_;
  const SyncCoverSolverOptions &options_;
  SyncCoverExchangeStatistics &statistics_;
  std::size_t &redundancyEvaluations_;
  bool &evaluationLimitReached_;
  const std::vector<SyncCoverMechanismId> *retained_ = nullptr;
  const std::vector<SyncCoverDemandId> *affectedDemands_ = nullptr;
  const std::vector<SyncCoverMechanismId> *candidatePool_ = nullptr;
  std::size_t localEvaluationLimit_ = 0;
  bool localLimitReached_ = false;
  std::set<std::vector<SyncCoverMechanismId>> seen_;
  std::optional<ExchangeCandidate> best_;
};

std::vector<EvictionPlan> buildEvictionPlans(
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverSelectionEvaluator &selectionEvaluator,
    const std::vector<SyncCoverMechanismId> &selected,
    const SyncCoverStructuralCost &incumbentCost,
    const SyncCoverSolverOptions &options, bool &candidateLimitReached) {
  struct RankedMechanism {
    SyncCoverMechanismId mechanism = 0;
    SyncCoverStructuralCost retainedCost;
  };
  std::vector<RankedMechanism> ranked;
  for (SyncCoverMechanismId mechanism : selected) {
    const std::vector<SyncCoverMechanismId> retained =
        subtractSorted(selected, {mechanism});
    const SyncCoverSelectionEvaluation evaluation =
        selectionEvaluator.evaluate(retained);
    if (evaluation && evaluation.resources.resourceFeasible &&
        syncCoverStructuralCostLess(evaluation.cost, incumbentCost)) {
      ranked.push_back({mechanism, evaluation.cost});
    }
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto &first,
                                             const auto &second) {
    if (syncCoverStructuralCostLess(first.retainedCost,
                                    second.retainedCost)) {
      return true;
    }
    if (syncCoverStructuralCostLess(second.retainedCost,
                                    first.retainedCost)) {
      return false;
    }
    return first.mechanism < second.mechanism;
  });
  if (ranked.size() > options.exchangeEvictionCandidateLimit) {
    ranked.resize(options.exchangeEvictionCandidateLimit);
    candidateLimitReached = true;
  }

  std::vector<std::vector<SyncCoverMechanismId>> evictionSets;
  for (const RankedMechanism &entry : ranked) {
    evictionSets.push_back({entry.mechanism});
  }
  for (std::size_t first = 0; first < ranked.size(); ++first) {
    for (std::size_t second = first + 1; second < ranked.size(); ++second) {
      std::vector<SyncCoverMechanismId> pair = {
          ranked[first].mechanism, ranked[second].mechanism};
      std::sort(pair.begin(), pair.end());
      evictionSets.push_back(std::move(pair));
    }
  }

  std::vector<SyncCoverMechanismId> selectedBarriers;
  for (SyncCoverMechanismId mechanism : selected) {
    if (universe.getMechanisms()[mechanism].barrier) {
      selectedBarriers.push_back(mechanism);
    }
  }
  if (!selectedBarriers.empty()) {
    evictionSets.push_back(selectedBarriers);
  }
  std::sort(evictionSets.begin(), evictionSets.end());
  evictionSets.erase(std::unique(evictionSets.begin(), evictionSets.end()),
                     evictionSets.end());

  std::vector<EvictionPlan> plans;
  for (std::vector<SyncCoverMechanismId> evicted : evictionSets) {
    std::vector<SyncCoverMechanismId> retained =
        subtractSorted(selected, evicted);
    const SyncCoverSelectionEvaluation evaluation =
        selectionEvaluator.evaluate(retained);
    if (!evaluation || !evaluation.resources.resourceFeasible ||
        !syncCoverStructuralCostLess(evaluation.cost, incumbentCost)) {
      continue;
    }
    plans.push_back(
        {std::move(evicted), std::move(retained), evaluation.cost});
  }
  std::sort(plans.begin(), plans.end(), evictionLess);
  return plans;
}

} // namespace

AffectedSliceExchangeResult
mlir::pto::sync_cover_internal::improveByAffectedSliceExchange(
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverSelectionEvaluator &selectionEvaluator,
    SyncCoverCoverageOracle &oracle,
    CoverageEvaluator &coverage,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &incumbent,
    const SyncCoverStructuralCost &incumbentCost,
    const SyncCoverSolverOptions &options,
    std::size_t &redundancyEvaluations) {
  AffectedSliceExchangeResult result;
  result.selected = incumbent;
  result.cost = incumbentCost;
  std::vector<SyncCoverStructuralCost> standaloneCosts;
  standaloneCosts.reserve(universe.getMechanisms().size());
  for (SyncCoverMechanismId mechanism = 0;
       mechanism < universe.getMechanisms().size(); ++mechanism) {
    standaloneCosts.push_back(
        universe.evaluateStructuralCost({mechanism}));
  }
  for (std::size_t round = 0; round < options.exchangeRoundLimit; ++round) {
    ++result.statistics.rounds;
    const std::vector<EvictionPlan> plans = buildEvictionPlans(
        universe, selectionEvaluator, result.selected, result.cost, options,
        result.candidateLimitReached);
    std::optional<ExchangeCandidate> best;
    ExchangeSearch search(selectionEvaluator, coverage, activeDemands,
                          result.cost, options, result.statistics,
                          redundancyEvaluations,
                          result.evaluationLimitReached);
    for (std::size_t planIndex = 0; planIndex < plans.size(); ++planIndex) {
      if (result.statistics.evaluations ==
          options.exchangeEvaluationLimit) {
        result.evaluationLimitReached = true;
        break;
      }
      const EvictionPlan &plan = plans[planIndex];
      ++result.statistics.evictionSets;
      const CoverageEvaluation affected =
          coverage.evaluate(activeDemands, plan.retained);
      if (!affected.valid) {
        continue;
      }
      const ExchangeCandidatePool pool = buildCandidatePool(
          universe, oracle, affected, plan.retained, standaloneCosts,
          options.exchangeCandidateLimit);
      if (!pool.valid) {
        continue;
      }
      result.candidateLimitReached |= pool.truncated;
      const std::size_t remainingPlans = plans.size() - planIndex;
      const std::size_t remainingEvaluations =
          options.exchangeEvaluationLimit - result.statistics.evaluations;
      const std::size_t allowance =
          std::max<std::size_t>(1, remainingEvaluations / remainingPlans);
      const std::optional<ExchangeCandidate> candidate =
          search.run(plan, affected.uncovered, pool.mechanisms, allowance);
      if (candidate && (!best || candidateLess(*candidate, *best))) {
        best = candidate;
      }
    }
    if (!best) {
      break;
    }
    result.selected = std::move(best->selected);
    result.cost = std::move(best->cost);
    ++result.statistics.accepted;
    removeRedundantMechanisms(selectionEvaluator, coverage, activeDemands,
                              result.selected, result.cost,
                              redundancyEvaluations);
    if (result.statistics.evaluations ==
        options.exchangeEvaluationLimit) {
      break;
    }
  }
  result.roundLimitReached =
      result.statistics.accepted == options.exchangeRoundLimit;
  return result;
}
