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
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir::pto;

namespace {

struct CandidateCost {
  CanonicalSyncStructuralCost value;
  std::vector<CanonicalSyncMechanismId> signature;
};

struct GreedyCandidate {
  CanonicalSyncPatternId pattern = 0;
  std::vector<CanonicalSyncMechanismId> additions;
  SyncCoverDemandSet coverage;
  CandidateCost cost;
  std::size_t gain = 0;
};

bool consumeWork(std::size_t amount, std::size_t limit, std::size_t &used) {
  if (used > limit || amount > limit - used) {
    return false;
  }
  used += amount;
  return true;
}

std::uint64_t saturatedAdd(std::uint64_t first, std::uint64_t second) {
  const bool overflows =
      second > std::numeric_limits<std::uint64_t>::max() - first;
  if (overflows) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return first + second;
}

std::uint64_t saturatedMultiply(std::uint64_t first, std::uint64_t second) {
  if (first != 0 &&
      second > std::numeric_limits<std::uint64_t>::max() / first) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return first * second;
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

bool containsForbidden(const std::vector<CanonicalSyncMechanismId> &members,
                       const std::vector<CanonicalSyncMechanismId> &forbidden) {
  return std::any_of(members.begin(), members.end(), [&](auto mechanism) {
    return std::binary_search(forbidden.begin(), forbidden.end(), mechanism);
  });
}

bool conflictFree(const CanonicalSyncPatternProblem &problem,
                  const std::vector<CanonicalSyncMechanismId> &selected) {
  for (CanonicalSyncMechanismId mechanism : selected) {
    const auto &conflicts = problem.getMechanisms()[mechanism].conflicts;
    if (std::any_of(conflicts.begin(), conflicts.end(), [&](auto conflict) {
          return std::binary_search(selected.begin(), selected.end(), conflict);
        })) {
      return false;
    }
  }
  return true;
}

bool patternActive(const CanonicalSyncPattern &pattern,
                   const std::vector<CanonicalSyncMechanismId> &selected) {
  return std::includes(selected.begin(), selected.end(),
                       pattern.members.begin(), pattern.members.end());
}

std::optional<SyncCoverDemandSet>
coveredBy(const CanonicalSyncPatternProblem &problem,
          const std::vector<CanonicalSyncMechanismId> &selected,
          std::size_t maximumWork, std::size_t &work) {
  SyncCoverDemandSet result = problem.getBaselineCoverage();
  for (const CanonicalSyncPattern &pattern : problem.getPatterns()) {
    const bool workAvailable = consumeWork(
        pattern.members.size() + selected.size() + 1, maximumWork, work);
    if (!workAvailable) {
      return std::nullopt;
    }
    if (patternActive(pattern, selected)) {
      if (!consumeWork(pattern.coverage.getWords().size(), maximumWork, work)) {
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
                    const std::vector<CanonicalSyncMechanismId> &additions,
                    std::size_t maximumWork, std::size_t &work) {
  const std::vector<CanonicalSyncMechanismId> successor =
      addMembers(selected, additions);
  std::vector<CanonicalSyncPatternId> affected;
  for (CanonicalSyncMechanismId mechanism : additions) {
    const auto &patterns = problem.getMechanismPatterns()[mechanism];
    if (!consumeWork(patterns.size(), maximumWork, work)) {
      return std::nullopt;
    }
    affected.insert(affected.end(), patterns.begin(), patterns.end());
  }
  std::sort(affected.begin(), affected.end());
  affected.erase(std::unique(affected.begin(), affected.end()), affected.end());

  SyncCoverDemandSet result = covered;
  for (CanonicalSyncPatternId patternId : affected) {
    const CanonicalSyncPattern &pattern = problem.getPatterns()[patternId];
    const bool workAvailable = consumeWork(
        pattern.members.size() + successor.size(), maximumWork, work);
    if (!workAvailable) {
      return std::nullopt;
    }
    if (patternActive(pattern, successor)) {
      result.unite(pattern.coverage);
    }
  }
  return result;
}

std::optional<std::size_t> countNewCoverage(const SyncCoverDemandSet &covered,
                                            const SyncCoverDemandSet &candidate,
                                            std::size_t maximumWork,
                                            std::size_t &work) {
  if (!consumeWork(candidate.getWords().size(), maximumWork, work)) {
    return std::nullopt;
  }
  std::size_t result = 0;
  for (std::size_t word = 0; word < candidate.getWords().size(); ++word) {
    const std::uint64_t existing =
        word < covered.getWords().size() ? covered.getWords()[word] : 0;
    result += static_cast<std::size_t>(
        __builtin_popcountll(candidate.getWords()[word] & ~existing));
  }
  return result;
}

CandidateCost getCost(const CanonicalSyncPatternProblem &problem,
                      const std::vector<CanonicalSyncMechanismId> &members) {
  CandidateCost result;
  result.signature = members;
  result.value.mechanismCount = members.size();
  for (CanonicalSyncMechanismId mechanismId : members) {
    const CanonicalSyncMechanism &mechanism =
        problem.getMechanisms()[mechanismId];
    const CanonicalSyncMechanismCost &cost = mechanism.cost;
    result.value.actionProfile.resize(
        std::max({result.value.actionProfile.size(), cost.barrierActions.size(),
                  cost.eventActions.size()}),
        0);
    for (std::size_t depth = 0; depth < cost.barrierActions.size(); ++depth) {
      result.value.actionProfile[depth] = saturatedAdd(
          result.value.actionProfile[depth], cost.barrierActions[depth]);
    }
    for (std::size_t depth = 0; depth < cost.eventActions.size(); ++depth) {
      result.value.actionProfile[depth] = saturatedAdd(
          result.value.actionProfile[depth], cost.eventActions[depth]);
    }
    result.value.serializationBreadth = saturatedAdd(
        result.value.serializationBreadth, cost.serializationBreadth);
    for (std::size_t use = 0; use < mechanism.eventLifetimes.size(); ++use) {
      const CanonicalSyncEventLifetime &lifetime =
          mechanism.eventLifetimes[use];
      const std::uint64_t span =
          static_cast<std::uint64_t>(lifetime.end - lifetime.begin) + 1;
      const std::uint64_t width =
          use < mechanism.descriptor.eventUses.size()
              ? mechanism.descriptor.eventUses[use].width
              : 0;
      result.value.eventLifetimeArea = saturatedAdd(
          result.value.eventLifetimeArea, saturatedMultiply(span, width));
    }
  }
  return result;
}

std::uint64_t profileValue(const std::vector<std::uint64_t> &profile,
                           std::size_t depth) {
  return depth < profile.size() ? profile[depth] : 0;
}

int compareRatio(std::uint64_t first, std::size_t firstGain,
                 std::uint64_t second, std::size_t secondGain) {
  const unsigned __int128 left =
      static_cast<unsigned __int128>(first) * secondGain;
  const unsigned __int128 right =
      static_cast<unsigned __int128>(second) * firstGain;
  if (left < right) {
    return -1;
  }
  if (right < left) {
    return 1;
  }
  return 0;
}

bool candidateLess(const GreedyCandidate &first,
                   const GreedyCandidate &second) {
  const std::size_t depths = std::max(first.cost.value.actionProfile.size(),
                                      second.cost.value.actionProfile.size());
  for (std::size_t reverse = depths; reverse > 0; --reverse) {
    const std::size_t depth = reverse - 1;
    const int order = compareRatio(
        profileValue(first.cost.value.actionProfile, depth), first.gain,
        profileValue(second.cost.value.actionProfile, depth), second.gain);
    if (order != 0) {
      return order < 0;
    }
  }
  const int serialization =
      compareRatio(first.cost.value.serializationBreadth, first.gain,
                   second.cost.value.serializationBreadth, second.gain);
  if (serialization != 0) {
    return serialization < 0;
  }
  const int lifetime =
      compareRatio(first.cost.value.eventLifetimeArea, first.gain,
                   second.cost.value.eventLifetimeArea, second.gain);
  if (lifetime != 0) {
    return lifetime < 0;
  }
  const int mechanisms = compareRatio(first.additions.size(), first.gain,
                                      second.additions.size(), second.gain);
  if (mechanisms != 0) {
    return mechanisms < 0;
  }
  return std::tie(first.additions, first.pattern) <
         std::tie(second.additions, second.pattern);
}

bool costLess(const CandidateCost &first, const CandidateCost &second) {
  const std::size_t depths = std::max(first.value.actionProfile.size(),
                                      second.value.actionProfile.size());
  for (std::size_t reverse = depths; reverse > 0; --reverse) {
    const std::size_t depth = reverse - 1;
    const std::uint64_t firstValue =
        profileValue(first.value.actionProfile, depth);
    const std::uint64_t secondValue =
        profileValue(second.value.actionProfile, depth);
    if (firstValue != secondValue) {
      return firstValue < secondValue;
    }
  }
  return std::tie(first.value.serializationBreadth,
                  first.value.eventLifetimeArea, first.value.mechanismCount,
                  first.signature) < std::tie(second.value.serializationBreadth,
                                              second.value.eventLifetimeArea,
                                              second.value.mechanismCount,
                                              second.signature);
}

bool addCandidate(const CanonicalSyncPatternProblem &problem,
                  const std::vector<CanonicalSyncMechanismId> &selected,
                  const SyncCoverDemandSet &covered,
                  CanonicalSyncPatternId pattern,
                  std::vector<CanonicalSyncMechanismId> additions,
                  const std::vector<CanonicalSyncMechanismId> &costMembers,
                  SyncCoverDemandSet successorCoverage,
                  const CanonicalSyncGreedyOptions &options,
                  CanonicalSyncGreedyStatistics &statistics,
                  std::vector<GreedyCandidate> &candidates) {
  const bool workAvailable =
      consumeWork(additions.size() + costMembers.size() + 1,
                  options.maximumWorkUnits, statistics.workUnits);
  if (!workAvailable) {
    return false;
  }
  const std::vector<CanonicalSyncMechanismId> successor =
      addMembers(selected, additions);
  if (!conflictFree(problem, successor)) {
    return true;
  }
  const std::optional<std::size_t> gain =
      countNewCoverage(covered, successorCoverage, options.maximumWorkUnits,
                       statistics.workUnits);
  if (!gain) {
    return false;
  }
  if (*gain == 0) {
    return true;
  }
  candidates.push_back({pattern, std::move(additions),
                        std::move(successorCoverage),
                        getCost(problem, costMembers), *gain});
  return true;
}

bool buildFixedCandidates(const CanonicalSyncPatternProblem &problem,
                          const std::vector<CanonicalSyncMechanismId> &selected,
                          const SyncCoverDemandSet &covered,
                          const CanonicalSyncGreedyOptions &options,
                          CanonicalSyncGreedyStatistics &statistics,
                          std::vector<GreedyCandidate> &candidates) {
  for (const CanonicalSyncPattern &pattern : problem.getPatterns()) {
    const bool forbidden =
        containsForbidden(pattern.members, options.forbiddenMechanisms);
    if (forbidden) {
      continue;
    }
    std::vector<CanonicalSyncMechanismId> additions =
        missingMembers(selected, pattern.members);
    if (additions.empty()) {
      continue;
    }
    const std::optional<SyncCoverDemandSet> columnCoverage =
        coveredBy(problem, pattern.members, options.maximumWorkUnits,
                  statistics.workUnits);
    if (!columnCoverage) {
      return false;
    }
    SyncCoverDemandSet successorCoverage = covered;
    successorCoverage.unite(*columnCoverage);
    if (!addCandidate(problem, selected, covered, pattern.id,
                      std::move(additions), pattern.members,
                      std::move(successorCoverage), options, statistics,
                      candidates)) {
      return false;
    }
  }
  return true;
}

bool buildActionAwareCandidate(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected,
    const SyncCoverDemandSet &covered,
    std::vector<CanonicalSyncMechanismId> additions,
    CanonicalSyncPatternId pattern, const CanonicalSyncGreedyOptions &options,
    CanonicalSyncGreedyStatistics &statistics,
    std::vector<GreedyCandidate> &candidates) {
  const std::optional<SyncCoverDemandSet> successorCoverage =
      coverageAfterAdding(problem, selected, covered, additions,
                          options.maximumWorkUnits, statistics.workUnits);
  if (!successorCoverage) {
    return false;
  }
  return addCandidate(problem, selected, covered, pattern, additions, additions,
                      *successorCoverage, options, statistics, candidates);
}

bool buildActionAwareCandidates(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected,
    const SyncCoverDemandSet &covered,
    const CanonicalSyncGreedyOptions &options,
    CanonicalSyncGreedyStatistics &statistics,
    std::vector<GreedyCandidate> &candidates) {
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    const bool alreadySelected =
        std::binary_search(selected.begin(), selected.end(), mechanism.id);
    const bool forbidden =
        std::binary_search(options.forbiddenMechanisms.begin(),
                           options.forbiddenMechanisms.end(), mechanism.id);
    if (alreadySelected || forbidden) {
      continue;
    }
    if (!buildActionAwareCandidate(problem, selected, covered, {mechanism.id},
                                   mechanism.id, options, statistics,
                                   candidates)) {
      return false;
    }
  }
  if (options.strategy != CanonicalSyncSelectionStrategy::PairLookahead) {
    return true;
  }
  for (const CanonicalSyncPattern &pattern : problem.getPatterns()) {
    const bool isPair = pattern.members.size() == 2;
    const bool forbidden =
        containsForbidden(pattern.members, options.forbiddenMechanisms);
    if (!isPair || forbidden) {
      continue;
    }
    std::vector<CanonicalSyncMechanismId> additions =
        missingMembers(selected, pattern.members);
    const bool addsBothMembers = additions.size() == 2;
    if (!addsBothMembers) {
      continue;
    }
    if (!buildActionAwareCandidate(problem, selected, covered,
                                   std::move(additions), pattern.id, options,
                                   statistics, candidates)) {
      return false;
    }
  }
  return true;
}

bool complete(const CanonicalSyncPatternProblem &problem,
              const SyncCoverDemandSet &covered) {
  return covered.count() == problem.getDemands().size();
}

bool consumeMechanismVerificationWork(
    const CanonicalSyncMechanism &mechanism,
    SyncCoverCoverageWorkBudget *coverageWork) {
  if (!coverageWork) {
    return true;
  }
  const CanonicalSyncMechanismDescriptor &descriptor = mechanism.descriptor;
  const bool mechanismWorkUnavailable =
      !coverageWork->consume(1) ||
      !coverageWork->consume(descriptor.supplies.size()) ||
      !coverageWork->consume(descriptor.eventUses.size()) ||
      !coverageWork->consume(descriptor.actions.size()) ||
      !coverageWork->consume(mechanism.eventLifetimes.size()) ||
      !coverageWork->consume(mechanism.cost.barrierActions.size()) ||
      !coverageWork->consume(mechanism.cost.eventActions.size()) ||
      !coverageWork->consume(mechanism.conflicts.size());
  if (mechanismWorkUnavailable) {
    return false;
  }
  for (const CanonicalSyncSupplyBinding &binding : descriptor.supplies) {
    const bool bindingWorkUnavailable =
        !coverageWork->consume(binding.edge.sourceGuard.literals.size()) ||
        !coverageWork->consume(binding.edge.targetGuard.literals.size()) ||
        !coverageWork->consume(binding.allowedDemands.size());
    if (bindingWorkUnavailable) {
      return false;
    }
  }
  for (const CanonicalSyncAction &action : descriptor.actions) {
    if (!coverageWork->consume(action.drainedResources.size())) {
      return false;
    }
  }
  return true;
}

} // namespace

CanonicalSyncStructuralCost mlir::pto::computeCanonicalSyncStructuralCost(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected) {
  return getCost(problem, selected).value;
}

CanonicalSyncSelection mlir::pto::selectCanonicalSyncPatterns(
    const CanonicalSyncPatternProblem &problem,
    CanonicalSyncGreedyOptions options) {
  CanonicalSyncSelection result;
  result.covered = problem.getBaselineCoverage();
  std::sort(options.forbiddenMechanisms.begin(),
            options.forbiddenMechanisms.end());
  options.forbiddenMechanisms.erase(
      std::unique(options.forbiddenMechanisms.begin(),
                  options.forbiddenMechanisms.end()),
      options.forbiddenMechanisms.end());
  const bool invalidForbidden =
      !options.forbiddenMechanisms.empty() &&
      options.forbiddenMechanisms.back() >= problem.getMechanisms().size();
  const bool invalidProblem =
      !problem.isFrozen() || options.maximumWorkUnits == 0 || invalidForbidden;
  if (invalidProblem) {
    result.error = CanonicalSyncSelectionError::InvalidProblem;
    return result;
  }

  std::vector<CanonicalSyncMechanismId> selected;
  while (!complete(problem, result.covered)) {
    std::vector<GreedyCandidate> candidates;
    const bool built =
        options.strategy == CanonicalSyncSelectionStrategy::FixedCover
            ? buildFixedCandidates(problem, selected, result.covered, options,
                                   result.statistics, candidates)
            : buildActionAwareCandidates(problem, selected, result.covered,
                                         options, result.statistics,
                                         candidates);
    if (!built) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    if (candidates.empty()) {
      result.error = CanonicalSyncSelectionError::NoCoveringPattern;
      return result;
    }
    if (!consumeWork(candidates.size(), options.maximumWorkUnits,
                     result.statistics.workUnits)) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    const bool evaluationCountOverflows =
        candidates.size() > std::numeric_limits<std::size_t>::max() -
                                result.statistics.patternEvaluations;
    if (evaluationCountOverflows) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    result.statistics.patternEvaluations += candidates.size();
    std::stable_sort(candidates.begin(), candidates.end(), candidateLess);
    GreedyCandidate chosen = std::move(candidates.front());
    for (CanonicalSyncMechanismId mechanism : chosen.additions) {
      if (!std::binary_search(selected.begin(), selected.end(), mechanism)) {
        result.selectionOrder.push_back(mechanism);
      }
    }
    selected = addMembers(selected, chosen.additions);
    result.covered = std::move(chosen.coverage);
  }

  CandidateCost selectedCost = getCost(problem, selected);
  for (auto position = result.selectionOrder.rbegin();
       position != result.selectionOrder.rend(); ++position) {
    std::vector<CanonicalSyncMechanismId> trial = selected;
    const auto found = std::lower_bound(trial.begin(), trial.end(), *position);
    const bool mechanismMissing = found == trial.end() || *found != *position;
    if (mechanismMissing) {
      continue;
    }
    trial.erase(found);
    ++result.statistics.deletionEvaluations;
    const std::optional<SyncCoverDemandSet> trialCoverage = coveredBy(
        problem, trial, options.maximumWorkUnits, result.statistics.workUnits);
    if (!trialCoverage) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    const CandidateCost trialCost = getCost(problem, trial);
    const bool remainsComplete = complete(problem, *trialCoverage);
    const bool reducesCost = costLess(trialCost, selectedCost);
    if (remainsComplete && reducesCost) {
      selected = std::move(trial);
      result.covered = *trialCoverage;
      selectedCost = trialCost;
    }
  }

  result.mechanisms = std::move(selected);
  result.cost = selectedCost.value;
  result.allocation = allocateCanonicalSyncEvents(problem, result.mechanisms);
  if (!result.allocation.valid) {
    result.error = CanonicalSyncSelectionError::InvalidAllocation;
  } else if (!result.allocation.feasible) {
    result.error = CanonicalSyncSelectionError::ResourceInfeasible;
  }
  return result;
}

CanonicalSyncVerifiedPlan mlir::pto::verifyCanonicalSyncSelection(
    const CanonicalSyncPatternProblem &problem,
    const CanonicalSyncSelection &selection,
    SyncCoverCoverageWorkBudget *coverageWork) {
  CanonicalSyncVerifiedPlan result;
  bool invalid = !problem.isFrozen() ||
                 selection.error != CanonicalSyncSelectionError::None;
  for (std::size_t index = 0; index < selection.mechanisms.size(); ++index) {
    const CanonicalSyncMechanismId mechanism = selection.mechanisms[index];
    if (mechanism >= problem.getMechanisms().size()) {
      invalid = true;
      continue;
    }
    if (!consumeMechanismVerificationWork(problem.getMechanisms()[mechanism],
                                          coverageWork)) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    const bool invalidMechanism =
        index != 0 && selection.mechanisms[index - 1] >= mechanism;
    const CanonicalSyncProblemResult verified =
        problem.verifyMechanism(mechanism, coverageWork);
    if ((coverageWork && coverageWork->exhausted) ||
        verified.error == CanonicalSyncProblemError::LimitExceeded) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    invalid = invalid || invalidMechanism || !verified;
  }
  if (invalid) {
    result.error = CanonicalSyncSelectionError::FinalValidationFailed;
    return result;
  }
  for (CanonicalSyncMechanismId mechanism : selection.mechanisms) {
    for (CanonicalSyncMechanismId conflict :
         problem.getMechanisms()[mechanism].conflicts) {
      if (std::binary_search(selection.mechanisms.begin(),
                             selection.mechanisms.end(), conflict)) {
        result.error = CanonicalSyncSelectionError::FinalValidationFailed;
        return result;
      }
    }
  }

  if (coverageWork &&
      !coverageWork->consume(
          selection.mechanisms.empty() ? 1 : selection.mechanisms.size())) {
    result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
    return result;
  }
  result.mechanisms = selection.mechanisms;
  result.allocation =
      allocateCanonicalSyncEvents(problem, result.mechanisms, coverageWork);
  if (coverageWork && coverageWork->exhausted) {
    result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
    return result;
  }
  if (!result.allocation.valid) {
    result.error = CanonicalSyncSelectionError::InvalidAllocation;
    return result;
  }
  if (!result.allocation.feasible) {
    result.error = CanonicalSyncSelectionError::ResourceInfeasible;
    return result;
  }

  std::vector<SyncCoverCompletionSupply> supplies;
  for (CanonicalSyncMechanismId mechanism : result.mechanisms) {
    if (coverageWork && !coverageWork->consume()) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    for (const CanonicalSyncSupplyBinding &binding :
         problem.getMechanisms()[mechanism].descriptor.supplies) {
      const std::size_t sourceLiterals =
          binding.edge.sourceGuard.literals.size();
      const std::size_t targetLiterals =
          binding.edge.targetGuard.literals.size();
      const std::size_t allowedDemands = binding.allowedDemands.size();
      const bool metadataOverflows =
          targetLiterals >
              std::numeric_limits<std::size_t>::max() - sourceLiterals ||
          allowedDemands > std::numeric_limits<std::size_t>::max() -
                               sourceLiterals - targetLiterals;
      if (metadataOverflows) {
        if (coverageWork) {
          coverageWork->exhausted = true;
        }
        result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
        return result;
      }
      const std::size_t metadata =
          sourceLiterals + targetLiterals + allowedDemands;
      if (coverageWork &&
          (!coverageWork->consume() || !coverageWork->consume(metadata))) {
        result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
        return result;
      }
      supplies.push_back({mechanism, binding.edge, binding.allowedDemands,
                          binding.completionExport ==
                              CanonicalSyncSupplyExport::ScopeExitAfterDrain,
                          binding.applicability});
    }
  }
  const auto verifyCovered = [&](const std::vector<SyncCoverDemandId> &demands,
                                 const SyncCoverCoverageResult &coverage) {
    for (SyncCoverDemandId graphDemand : demands) {
      if (coverageWork && !coverageWork->consume()) {
        result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
        return false;
      }
      if (!coverage.covered.contains(graphDemand)) {
        result.error = CanonicalSyncSelectionError::FinalValidationFailed;
        result.firstUncoveredDemand = graphDemand;
        return false;
      }
    }
    return true;
  };
  const SyncCoverCoverageResult basisCoverage = computeSyncCoverCoverage(
      problem.getGraph(), problem.getExpansion(), supplies,
      problem.getDemands(), coverageWork);
  if (!basisCoverage) {
    result.error =
        basisCoverage.error == SyncCoverCoverageError::WorkLimitExceeded
            ? CanonicalSyncSelectionError::WorkLimitExceeded
            : CanonicalSyncSelectionError::FinalValidationFailed;
    return result;
  }
  if (!verifyCovered(problem.getDemands(), basisCoverage)) {
    return result;
  }

  // A basis demand that has just been proved is now a completion lemma.  Its
  // canonical edge is unrestricted even when the concrete mechanism used to
  // establish it carried a row-local ownership-release filter.  Verify the
  // complete original universe from the concrete supplies plus these proved
  // lemmas; reduced rows never bypass fresh semantic verification.
  const std::size_t lemmaMechanismBegin = problem.getMechanisms().size();
  if (problem.getDemands().size() >
          std::numeric_limits<std::size_t>::max() - lemmaMechanismBegin ||
      problem.getDemands().size() >
          std::numeric_limits<std::size_t>::max() - supplies.size()) {
    result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
    return result;
  }
  const std::size_t combinedSupplyCount =
      supplies.size() + problem.getDemands().size();
  if (coverageWork && !coverageWork->consume(combinedSupplyCount)) {
    result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
    return result;
  }
  supplies.reserve(combinedSupplyCount);
  for (std::size_t index = 0; index < problem.getDemands().size(); ++index) {
    const SyncCoverDemand &demand = problem.getGraph().getDemands()[
        problem.getDemands()[index]];
    const bool guardMetadataOverflows =
        demand.targetGuard.literals.size() >
        std::numeric_limits<std::size_t>::max() -
            demand.sourceGuard.literals.size();
    const std::size_t guardMetadata =
        guardMetadataOverflows
            ? 0
            : demand.sourceGuard.literals.size() +
                  demand.targetGuard.literals.size();
    if (guardMetadataOverflows ||
        (coverageWork &&
         (!coverageWork->consume() ||
          !coverageWork->consume(guardMetadata)))) {
      if (coverageWork && guardMetadataOverflows) {
        coverageWork->exhausted = true;
      }
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    SyncCoverCompletionSupply lemma;
    lemma.mechanism = lemmaMechanismBegin + index;
    lemma.edge = {demand.source, demand.target,
                  SyncCoverEdgeKind::CompletionSupply, demand.scope,
                  demand.distance, demand.sourceGuard, demand.targetGuard};
    lemma.applicability = SyncCoverSupplyApplicability::AllDemands;
    supplies.push_back(std::move(lemma));
  }
  const SyncCoverCoverageResult obligationCoverage = computeSyncCoverCoverage(
      problem.getGraph(), problem.getExpansion(), supplies,
      problem.getObligationDemands(), coverageWork);
  if (!obligationCoverage) {
    result.error =
        obligationCoverage.error == SyncCoverCoverageError::WorkLimitExceeded
            ? CanonicalSyncSelectionError::WorkLimitExceeded
            : CanonicalSyncSelectionError::FinalValidationFailed;
    return result;
  }
  for (SyncCoverDemandId graphDemand : problem.getObligationDemands()) {
    if (coverageWork && !coverageWork->consume()) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    if (!obligationCoverage.covered.contains(graphDemand)) {
      result.error = CanonicalSyncSelectionError::FinalValidationFailed;
      result.firstUncoveredDemand = graphDemand;
      return result;
    }
  }
  return result;
}
