// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir::pto;

namespace {

struct StructuralCost {
  std::vector<std::uint64_t> actions;
  std::size_t mechanisms = 0;
  std::vector<CanonicalSyncMechanismId> signature;
};

bool consumeWork(std::size_t amount, std::size_t limit, std::size_t &used) {
  const bool exceeded = used > limit || amount > limit - used;
  if (exceeded) {
    return false;
  }
  used += amount;
  return true;
}

std::optional<std::size_t> checkedWorkProduct(std::size_t first,
                                              std::size_t second) {
  const bool overflows =
      first != 0 && second > std::numeric_limits<std::size_t>::max() / first;
  if (overflows) {
    return std::nullopt;
  }
  return first * second;
}

bool addWork(std::size_t amount, std::size_t &total) {
  const bool overflows =
      amount > std::numeric_limits<std::size_t>::max() - total;
  if (overflows) {
    return false;
  }
  total += amount;
  return true;
}

std::size_t logarithmicFactor(std::size_t count) {
  std::size_t factor = 1;
  while (count > 1) {
    count = (count + 1) / 2;
    ++factor;
  }
  return factor;
}

std::optional<std::size_t>
getAllocationWork(const CanonicalSyncPatternProblem &problem,
                  const std::vector<CanonicalSyncMechanismId> &selected) {
  std::size_t work = problem.getDomains().size();
  if (!addWork(selected.size(), work)) {
    return std::nullopt;
  }
  std::vector<std::size_t> intervals(problem.getDomains().size(), 0);
  for (CanonicalSyncMechanismId mechanism : selected) {
    const CanonicalSyncMechanism &item = problem.getMechanisms()[mechanism];
    if (!addWork(item.conflicts.size(), work)) {
      return std::nullopt;
    }
    for (const CanonicalSyncEventUse &use : item.descriptor.eventUses) {
      std::size_t useWork = 1;
      const bool invalidUse = use.domain >= intervals.size() ||
                              !addWork(use.width, useWork) ||
                              !addWork(useWork, work);
      if (invalidUse) {
        return std::nullopt;
      }
      ++intervals[use.domain];
    }
  }
  for (std::size_t domain = 0; domain < intervals.size(); ++domain) {
    const std::size_t factor = logarithmicFactor(intervals[domain]);
    const std::optional<std::size_t> sortWork =
        checkedWorkProduct(intervals[domain], factor * factor);
    if (!sortWork ||
        !addWork(problem.getDomains()[domain].reservedIds.size(), work) ||
        !addWork(*sortWork, work)) {
      return std::nullopt;
    }
  }
  return work;
}

std::vector<CanonicalSyncMechanismId>
addMembers(const std::vector<CanonicalSyncMechanismId> &selected,
           const std::vector<CanonicalSyncMechanismId> &members) {
  std::vector<CanonicalSyncMechanismId> result;
  result.reserve(selected.size() + members.size());
  std::set_union(selected.begin(), selected.end(), members.begin(),
                 members.end(), std::back_inserter(result));
  return result;
}

std::vector<CanonicalSyncMechanismId>
missingMembers(const std::vector<CanonicalSyncMechanismId> &selected,
               const std::vector<CanonicalSyncMechanismId> &members) {
  std::vector<CanonicalSyncMechanismId> result;
  std::set_difference(members.begin(), members.end(), selected.begin(),
                      selected.end(), std::back_inserter(result));
  return result;
}

std::optional<std::size_t> countNewCoverage(const SyncCoverDemandSet &covered,
                                            const SyncCoverDemandSet &candidate,
                                            std::size_t limit,
                                            std::size_t &work) {
  if (!consumeWork(candidate.getWords().size(), limit, work)) {
    return std::nullopt;
  }
  std::size_t result = 0;
  const auto &coveredWords = covered.getWords();
  const auto &candidateWords = candidate.getWords();
  for (std::size_t word = 0; word < candidateWords.size(); ++word) {
    const std::uint64_t existing =
        word < coveredWords.size() ? coveredWords[word] : 0;
    result += static_cast<std::size_t>(
        __builtin_popcountll(candidateWords[word] & ~existing));
  }
  return result;
}

bool patternActive(const CanonicalSyncPattern &pattern,
                   const std::vector<CanonicalSyncMechanismId> &selected) {
  return std::includes(selected.begin(), selected.end(),
                       pattern.members.begin(), pattern.members.end());
}

std::optional<SyncCoverDemandSet>
coveredBy(const CanonicalSyncPatternProblem &problem,
          const std::vector<CanonicalSyncMechanismId> &selected,
          std::size_t limit, std::size_t &work) {
  SyncCoverDemandSet result = problem.getBaselineCoverage();
  for (const CanonicalSyncPattern &pattern : problem.getPatterns()) {
    std::size_t patternWork = 1;
    const bool workAvailable = addWork(pattern.members.size(), patternWork) &&
                               addWork(selected.size(), patternWork) &&
                               consumeWork(patternWork, limit, work);
    if (!workAvailable) {
      return std::nullopt;
    }
    if (patternActive(pattern, selected)) {
      if (!consumeWork(pattern.coverage.getWords().size(), limit, work)) {
        return std::nullopt;
      }
      result.unite(pattern.coverage);
    }
  }
  return result;
}

std::optional<SyncCoverDemandSet>
coverageAfterAdding(const CanonicalSyncPatternProblem &problem,
                    const std::vector<CanonicalSyncMechanismId> &selected,
                    const SyncCoverDemandSet &covered,
                    const std::vector<CanonicalSyncMechanismId> &added,
                    std::size_t maximumWork, std::size_t &work) {
  const std::vector<CanonicalSyncMechanismId> successor =
      addMembers(selected, added);
  std::vector<CanonicalSyncPatternId> affected;
  for (CanonicalSyncMechanismId mechanism : added) {
    const auto &patterns = problem.getMechanismPatterns()[mechanism];
    if (!consumeWork(patterns.size(), maximumWork, work)) {
      return std::nullopt;
    }
    affected.insert(affected.end(), patterns.begin(), patterns.end());
  }
  const std::size_t sortFactor = logarithmicFactor(affected.size());
  const std::optional<std::size_t> sortWork =
      checkedWorkProduct(affected.size(), sortFactor);
  if (!sortWork || !consumeWork(*sortWork, maximumWork, work)) {
    return std::nullopt;
  }
  std::sort(affected.begin(), affected.end());
  affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
  SyncCoverDemandSet result = covered;
  for (CanonicalSyncPatternId pattern : affected) {
    const CanonicalSyncPattern &candidate = problem.getPatterns()[pattern];
    std::size_t patternWork = candidate.members.size();
    const bool workAvailable = addWork(successor.size(), patternWork) &&
                               consumeWork(patternWork, maximumWork, work);
    if (!workAvailable) {
      return std::nullopt;
    }
    if (patternActive(candidate, successor)) {
      if (!consumeWork(candidate.coverage.getWords().size(), maximumWork,
                       work)) {
        return std::nullopt;
      }
      result.unite(candidate.coverage);
    }
  }
  return result;
}

std::optional<StructuralCost>
getCost(const CanonicalSyncPatternProblem &problem,
        const std::vector<CanonicalSyncMechanismId> &selected,
        std::size_t limit, std::size_t &work) {
  if (!consumeWork(selected.size(), limit, work)) {
    return std::nullopt;
  }
  StructuralCost result;
  result.mechanisms = selected.size();
  result.signature = selected;
  for (CanonicalSyncMechanismId mechanism : selected) {
    const CanonicalSyncMechanismCost &cost =
        problem.getMechanisms()[mechanism].cost;
    std::size_t profileWork = cost.barrierActions.size();
    const bool workAvailable = addWork(cost.eventActions.size(), profileWork) &&
                               consumeWork(profileWork, limit, work);
    if (!workAvailable) {
      return std::nullopt;
    }
    result.actions.resize(
        std::max({result.actions.size(), cost.barrierActions.size(),
                  cost.eventActions.size()}),
        0);
    for (std::size_t depth = 0; depth < cost.barrierActions.size(); ++depth) {
      result.actions[depth] += cost.barrierActions[depth];
    }
    for (std::size_t depth = 0; depth < cost.eventActions.size(); ++depth) {
      result.actions[depth] += cost.eventActions[depth];
    }
  }
  return result;
}

std::uint64_t profileValue(const std::vector<std::uint64_t> &profile,
                           std::size_t depth) {
  return depth < profile.size() ? profile[depth] : 0;
}

bool structuralCostLess(const StructuralCost &first,
                        const StructuralCost &second) {
  const std::size_t depthCount =
      std::max(first.actions.size(), second.actions.size());
  for (std::size_t reverse = depthCount; reverse > 0; --reverse) {
    const std::size_t depth = reverse - 1;
    const std::uint64_t firstDepth = profileValue(first.actions, depth);
    const std::uint64_t secondDepth = profileValue(second.actions, depth);
    if (firstDepth != secondDepth) {
      return firstDepth < secondDepth;
    }
  }
  return std::tie(first.mechanisms, first.signature) <
         std::tie(second.mechanisms, second.signature);
}

struct GreedyCandidate {
  CanonicalSyncPatternId pattern = 0;
  CanonicalSyncSelectionTier tier = CanonicalSyncSelectionTier::Precise;
  std::vector<CanonicalSyncMechanismId> additions;
  SyncCoverDemandSet coverage;
  std::vector<std::uint64_t> actions;
  std::size_t gain = 0;
  std::size_t pressureNumerator = 0;
  std::size_t pressureDenominator = 1;
  std::uint64_t eventLifetimeSpan = 0;
};

CanonicalSyncSelectionTier
getPatternTier(const CanonicalSyncPatternProblem &problem,
               const CanonicalSyncPattern &pattern) {
  CanonicalSyncSelectionTier tier = CanonicalSyncSelectionTier::Precise;
  for (CanonicalSyncMechanismId mechanism : pattern.members) {
    tier = std::max(
        tier, problem.getMechanisms()[mechanism].descriptor.selectionTier);
  }
  return tier;
}

std::optional<std::size_t>
getCandidateSortWork(const std::vector<GreedyCandidate> &candidates) {
  const std::size_t factor = logarithmicFactor(candidates.size());
  std::size_t work = 0;
  for (const GreedyCandidate &candidate : candidates) {
    std::size_t profile = candidate.actions.size();
    const bool profileValid = addWork(2, profile);
    if (!profileValid) {
      return std::nullopt;
    }
    const std::optional<std::size_t> candidateWork =
        checkedWorkProduct(profile, factor);
    if (!candidateWork || !addWork(*candidateWork, work)) {
      return std::nullopt;
    }
  }
  return work;
}

bool ratioLess(std::uint64_t first, std::size_t firstGain, std::uint64_t second,
               std::size_t secondGain) {
  const unsigned __int128 left =
      static_cast<unsigned __int128>(first) * secondGain;
  const unsigned __int128 right =
      static_cast<unsigned __int128>(second) * firstGain;
  return left < right;
}

int compareRatio(std::uint64_t first, std::size_t firstGain,
                 std::uint64_t second, std::size_t secondGain) {
  if (ratioLess(first, firstGain, second, secondGain)) {
    return -1;
  }
  if (ratioLess(second, secondGain, first, firstGain)) {
    return 1;
  }
  return 0;
}

bool pressureLess(const GreedyCandidate &first, const GreedyCandidate &second) {
  const unsigned __int128 left =
      static_cast<unsigned __int128>(first.pressureNumerator) *
      second.pressureDenominator;
  const unsigned __int128 right =
      static_cast<unsigned __int128>(second.pressureNumerator) *
      first.pressureDenominator;
  return left < right;
}

bool greedyLess(const GreedyCandidate &first, const GreedyCandidate &second,
                bool preferEventHeadroom) {
  const bool firstPipeAll =
      first.tier == CanonicalSyncSelectionTier::PipeAllRescue;
  const bool secondPipeAll =
      second.tier == CanonicalSyncSelectionTier::PipeAllRescue;
  if (firstPipeAll != secondPipeAll) {
    return !firstPipeAll;
  }
  if (preferEventHeadroom) {
    if (pressureLess(first, second)) {
      return true;
    }
    if (pressureLess(second, first)) {
      return false;
    }
    if (first.eventLifetimeSpan != second.eventLifetimeSpan) {
      return first.eventLifetimeSpan < second.eventLifetimeSpan;
    }
  }
  const std::size_t depths =
      std::max(first.actions.size(), second.actions.size());
  for (std::size_t reverse = depths; reverse > 0; --reverse) {
    const std::size_t depth = reverse - 1;
    const int actions =
        compareRatio(profileValue(first.actions, depth), first.gain,
                     profileValue(second.actions, depth), second.gain);
    if (actions != 0) {
      return actions < 0;
    }
  }
  const int mechanisms = compareRatio(first.additions.size(), first.gain,
                                      second.additions.size(), second.gain);
  return mechanisms != 0 ? mechanisms < 0 : first.pattern < second.pattern;
}

void setCandidatePressure(const CanonicalSyncResourceAllocation &allocation,
                          GreedyCandidate &candidate) {
  for (const CanonicalSyncDomainAllocation &domain : allocation.domains) {
    if (domain.available == 0) {
      continue;
    }
    const unsigned __int128 current =
        static_cast<unsigned __int128>(candidate.pressureNumerator) *
        domain.available;
    const unsigned __int128 proposed =
        static_cast<unsigned __int128>(domain.required) *
        candidate.pressureDenominator;
    if (proposed > current) {
      candidate.pressureNumerator = domain.required;
      candidate.pressureDenominator = domain.available;
    }
  }
}

std::optional<GreedyCandidate>
makeCandidate(const CanonicalSyncPatternProblem &problem,
              const SyncCoverDemandSet &covered,
              CanonicalSyncPatternId patternId,
              std::vector<CanonicalSyncMechanismId> additions,
              SyncCoverDemandSet successorCoverage, std::size_t maximumWork,
              std::size_t &work) {
  GreedyCandidate result;
  result.pattern = patternId;
  result.tier = getPatternTier(problem, problem.getPatterns()[patternId]);
  result.additions = std::move(additions);
  result.coverage = std::move(successorCoverage);
  for (CanonicalSyncMechanismId mechanism : result.additions) {
    const CanonicalSyncMechanism &item = problem.getMechanisms()[mechanism];
    std::size_t profileWork = item.cost.barrierActions.size();
    const bool workAvailable =
        addWork(item.cost.eventActions.size(), profileWork) &&
        addWork(1, profileWork) && consumeWork(profileWork, maximumWork, work);
    if (!workAvailable) {
      return std::nullopt;
    }
    result.actions.resize(
        std::max({result.actions.size(), item.cost.barrierActions.size(),
                  item.cost.eventActions.size()}),
        0);
    for (std::size_t depth = 0; depth < item.cost.barrierActions.size();
         ++depth) {
      result.actions[depth] += item.cost.barrierActions[depth];
    }
    for (std::size_t depth = 0; depth < item.cost.eventActions.size();
         ++depth) {
      result.actions[depth] += item.cost.eventActions[depth];
    }
    for (const CanonicalSyncEventLifetime &lifetime : item.eventLifetimes) {
      const std::uint64_t span = lifetime.end - lifetime.begin;
      const std::uint64_t remainingSpan =
          std::numeric_limits<std::uint64_t>::max() - result.eventLifetimeSpan;
      if (span > remainingSpan) {
        result.eventLifetimeSpan = std::numeric_limits<std::uint64_t>::max();
      } else {
        result.eventLifetimeSpan += span;
      }
    }
  }
  const std::optional<std::size_t> gain =
      countNewCoverage(covered, result.coverage, maximumWork, work);
  if (!gain) {
    return std::nullopt;
  }
  result.gain = *gain;
  return result;
}

} // namespace

CanonicalSyncSelection mlir::pto::selectCanonicalSyncPatterns(
    const CanonicalSyncPatternProblem &problem,
    CanonicalSyncGreedyOptions options) {
  CanonicalSyncSelection result;
  result.covered = problem.getBaselineCoverage();
  const bool invalidProblem =
      !problem.isFrozen() || options.maximumWorkUnits == 0;
  if (invalidProblem) {
    result.error = CanonicalSyncSelectionError::InvalidProblem;
    return result;
  }

  std::vector<CanonicalSyncMechanismId> selected;
  std::vector<CanonicalSyncMechanismId> selectionOrder;
  while (true) {
    std::size_t completionWork = result.covered.getWords().size();
    const bool workAvailable =
        addWork(problem.getDemands().size(), completionWork) &&
        consumeWork(completionWork, options.maximumWorkUnits,
                    result.statistics.workUnits);
    if (!workAvailable) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    const bool complete = result.covered.count() == problem.getDemands().size();
    if (complete) {
      break;
    }
    std::optional<std::size_t> anchor;
    for (std::size_t demand = 0; demand < problem.getDemands().size();
         ++demand) {
      if (!result.covered.contains(demand)) {
        anchor = demand;
        break;
      }
    }
    if (!anchor) {
      result.error = CanonicalSyncSelectionError::NoCoveringPattern;
      return result;
    }

    std::vector<GreedyCandidate> candidates;
    for (CanonicalSyncPatternId pattern :
         problem.getDemandPatterns()[*anchor]) {
      const CanonicalSyncPattern &patternDetails =
          problem.getPatterns()[pattern];
      const bool tierDisabled =
          getPatternTier(problem, patternDetails) > options.maximumTier;
      if (tierDisabled) {
        continue;
      }
      std::size_t memberWork = selected.size();
      const bool workAvailable =
          addWork(patternDetails.members.size(), memberWork) &&
          consumeWork(memberWork, options.maximumWorkUnits,
                      result.statistics.workUnits);
      if (!workAvailable) {
        result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
        return result;
      }
      const auto additions = missingMembers(selected, patternDetails.members);
      if (additions.empty()) {
        continue;
      }
      const std::optional<SyncCoverDemandSet> coverage = coverageAfterAdding(
          problem, selected, result.covered, additions,
          options.maximumWorkUnits, result.statistics.workUnits);
      if (!coverage) {
        result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
        return result;
      }
      std::optional<GreedyCandidate> candidate =
          makeCandidate(problem, result.covered, pattern, additions, *coverage,
                        options.maximumWorkUnits, result.statistics.workUnits);
      if (!candidate) {
        result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
        return result;
      }
      if (candidate->gain != 0) {
        if (options.preferEventHeadroom) {
          const std::vector<CanonicalSyncMechanismId> successor =
              addMembers(selected, candidate->additions);
          const std::optional<std::size_t> allocationWork =
              getAllocationWork(problem, successor);
          const bool pressureWorkAvailable =
              allocationWork &&
              consumeWork(*allocationWork, options.maximumWorkUnits,
                          result.statistics.workUnits);
          if (!pressureWorkAvailable) {
            result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
            return result;
          }
          const CanonicalSyncResourceAllocation allocation =
              allocateCanonicalSyncEvents(problem, successor);
          if (!allocation.valid || !allocation.feasible) {
            continue;
          }
          setCandidatePressure(allocation, *candidate);
        }
        candidates.push_back(std::move(*candidate));
      }
    }
    const std::optional<std::size_t> sortWork =
        getCandidateSortWork(candidates);
    if (!sortWork || !consumeWork(*sortWork, options.maximumWorkUnits,
                                  result.statistics.workUnits)) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [&](const auto &first, const auto &second) {
                       return greedyLess(first, second,
                                         options.preferEventHeadroom);
                     });

    bool advanced = false;
    for (GreedyCandidate &candidate : candidates) {
      ++result.statistics.patternEvaluations;
      const std::vector<CanonicalSyncMechanismId> successor =
          addMembers(selected, candidate.additions);
      const std::optional<std::size_t> allocationWork =
          getAllocationWork(problem, successor);
      const bool workAvailable =
          allocationWork &&
          consumeWork(*allocationWork, options.maximumWorkUnits,
                      result.statistics.workUnits);
      if (!workAvailable) {
        result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
        return result;
      }
      const CanonicalSyncResourceAllocation allocation =
          allocateCanonicalSyncEvents(problem, successor);
      if (!allocation.valid || !allocation.feasible) {
        continue;
      }
      for (CanonicalSyncMechanismId mechanism : candidate.additions) {
        if (!std::binary_search(selected.begin(), selected.end(), mechanism)) {
          selectionOrder.push_back(mechanism);
        }
      }
      selected = successor;
      result.covered = std::move(candidate.coverage);
      advanced = true;
      break;
    }
    if (!advanced) {
      result.error = candidates.empty()
                         ? CanonicalSyncSelectionError::NoCoveringPattern
                         : CanonicalSyncSelectionError::ResourceInfeasible;
      return result;
    }
  }

  std::optional<StructuralCost> cost = getCost(
      problem, selected, options.maximumWorkUnits, result.statistics.workUnits);
  if (!cost) {
    result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
    return result;
  }
  for (auto position = selectionOrder.rbegin();
       position != selectionOrder.rend(); ++position) {
    std::vector<CanonicalSyncMechanismId> candidate = selected;
    const auto found =
        std::lower_bound(candidate.begin(), candidate.end(), *position);
    const bool absent = found == candidate.end() || *found != *position;
    if (absent) {
      continue;
    }
    candidate.erase(found);
    ++result.statistics.deletionEvaluations;
    const std::optional<std::size_t> allocationWork =
        getAllocationWork(problem, candidate);
    const bool workAvailable =
        allocationWork && consumeWork(*allocationWork, options.maximumWorkUnits,
                                      result.statistics.workUnits);
    if (!workAvailable) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    const CanonicalSyncResourceAllocation allocation =
        allocateCanonicalSyncEvents(problem, candidate);
    if (!allocation.valid || !allocation.feasible) {
      continue;
    }
    const std::optional<SyncCoverDemandSet> coverage =
        coveredBy(problem, candidate, options.maximumWorkUnits,
                  result.statistics.workUnits);
    const std::optional<StructuralCost> candidateCost =
        getCost(problem, candidate, options.maximumWorkUnits,
                result.statistics.workUnits);
    if (!coverage || !candidateCost) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    if (!consumeWork(coverage->getWords().size(), options.maximumWorkUnits,
                     result.statistics.workUnits)) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    const bool improves = coverage->count() == problem.getDemands().size() &&
                          structuralCostLess(*candidateCost, *cost);
    if (improves) {
      selected = std::move(candidate);
      result.covered = *coverage;
      cost = *candidateCost;
    }
  }

  result.mechanisms = std::move(selected);
  const std::optional<std::size_t> finalAllocationWork =
      getAllocationWork(problem, result.mechanisms);
  if (!finalAllocationWork ||
      !consumeWork(*finalAllocationWork, options.maximumWorkUnits,
                   result.statistics.workUnits)) {
    result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
    return result;
  }
  result.allocation = allocateCanonicalSyncEvents(problem, result.mechanisms);
  if (!result.allocation.valid || !result.allocation.feasible) {
    result.error = CanonicalSyncSelectionError::ResourceInfeasible;
  }
  return result;
}

CanonicalSyncVerifiedPlan mlir::pto::verifyCanonicalSyncSelection(
    const CanonicalSyncPatternProblem &problem,
    const CanonicalSyncSelection &selection) {
  CanonicalSyncVerifiedPlan result;
  const bool invalid =
      !problem.isFrozen() || !selection ||
      !std::is_sorted(selection.mechanisms.begin(),
                      selection.mechanisms.end()) ||
      std::adjacent_find(selection.mechanisms.begin(),
                         selection.mechanisms.end()) !=
          selection.mechanisms.end() ||
      std::any_of(selection.mechanisms.begin(), selection.mechanisms.end(),
                  [&](CanonicalSyncMechanismId mechanism) {
                    return mechanism >= problem.getMechanisms().size();
                  });
  if (invalid) {
    result.error = CanonicalSyncSelectionError::FinalValidationFailed;
    return result;
  }
  result.mechanisms = selection.mechanisms;
  result.allocation = allocateCanonicalSyncEvents(problem, result.mechanisms);
  if (!result.allocation.valid || !result.allocation.feasible) {
    result.error = CanonicalSyncSelectionError::ResourceInfeasible;
    return result;
  }
  std::size_t coverageWork = 0;
  const std::optional<SyncCoverDemandSet> coverage =
      coveredBy(problem, result.mechanisms,
                std::numeric_limits<std::size_t>::max(), coverageWork);
  if (!coverage) {
    result.error = CanonicalSyncSelectionError::FinalValidationFailed;
    return result;
  }
  for (std::size_t demand = 0; demand < problem.getDemands().size(); ++demand) {
    if (!coverage->contains(demand)) {
      result.error = CanonicalSyncSelectionError::FinalValidationFailed;
      result.firstUncoveredDemand = problem.getDemands()[demand];
      return result;
    }
  }
  return result;
}
