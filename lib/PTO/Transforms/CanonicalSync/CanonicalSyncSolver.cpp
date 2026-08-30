// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

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

enum class CostComputationError : std::uint8_t {
  None,
  WorkLimitExceeded,
  ArithmeticOverflow,
};

struct CostComputation {
  CostComputationError error = CostComputationError::None;
  CandidateCost cost;

  explicit operator bool() const { return error == CostComputationError::None; }
};

struct GreedyCandidate {
  CanonicalSyncPatternId pattern = 0;
  std::vector<CanonicalSyncMechanismId> additions;
  CandidateCost cost;
  std::size_t gain = 0;
};

struct GreedyCandidateScan {
  explicit GreedyCandidateScan(std::size_t demandCount)
      : scratchCoverage(demandCount), bestCoverage(demandCount) {}

  void reset() { best.reset(); }

  std::optional<GreedyCandidate> best;
  SyncCoverDemandSet scratchCoverage;
  SyncCoverDemandSet bestCoverage;
};

bool consumeWork(std::size_t amount, std::size_t limit, std::size_t &used) {
  if (used > limit || amount > limit - used) {
    return false;
  }
  used += amount;
  return true;
}

bool checkedWorkSum(std::size_t first, std::size_t second,
                    std::size_t &result) {
  const bool valid = second <= std::numeric_limits<std::size_t>::max() - first;
  result = valid ? first + second : 0;
  return valid;
}

bool checkedWorkProduct(std::size_t first, std::size_t second,
                        std::size_t &result) {
  const bool valid =
      first == 0 || second <= std::numeric_limits<std::size_t>::max() / first;
  result = valid ? first * second : 0;
  return valid;
}

bool consumeSortWork(std::size_t count, std::size_t limit, std::size_t &used) {
  if (count < 2) {
    return consumeWork(1, limit, used);
  }
  std::size_t levels = 0;
  for (std::size_t remaining = count - 1; remaining != 0; remaining >>= 1) {
    ++levels;
  }
  if (levels > std::numeric_limits<std::size_t>::max() / count) {
    return false;
  }
  return consumeWork(count * levels, limit, used);
}

bool checkedCostAdd(std::uint64_t first, std::uint64_t second,
                    std::uint64_t &result) {
  if (second > std::numeric_limits<std::uint64_t>::max() - first) {
    return false;
  }
  result = first + second;
  return true;
}

bool checkedCostMultiply(std::uint64_t first, std::uint64_t second,
                         std::uint64_t &result) {
  if (first != 0 &&
      second > std::numeric_limits<std::uint64_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
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

std::optional<std::vector<CanonicalSyncMechanismId>>
missingMembers(const std::vector<CanonicalSyncMechanismId> &selected,
               const std::vector<CanonicalSyncMechanismId> &members,
               std::size_t maximumWork, std::size_t &work) {
  if (!consumeWork(selected.size() + members.size(), maximumWork, work)) {
    return std::nullopt;
  }
  std::vector<CanonicalSyncMechanismId> result;
  std::set_difference(members.begin(), members.end(), selected.begin(),
                      selected.end(), std::back_inserter(result));
  return result;
}

std::optional<bool>
containsSorted(const std::vector<CanonicalSyncMechanismId> &values,
               CanonicalSyncMechanismId value, std::size_t maximumWork,
               std::size_t &work) {
  std::size_t first = 0;
  std::size_t last = values.size();
  while (first < last) {
    if (!consumeWork(1, maximumWork, work)) {
      return std::nullopt;
    }
    const std::size_t middle = first + (last - first) / 2;
    if (values[middle] < value) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  return first != values.size() && values[first] == value;
}

std::optional<bool>
containsForbidden(const std::vector<CanonicalSyncMechanismId> &members,
                  const std::vector<CanonicalSyncMechanismId> &forbidden,
                  std::size_t maximumWork, std::size_t &work) {
  for (CanonicalSyncMechanismId mechanism : members) {
    const std::optional<bool> forbiddenMember =
        containsSorted(forbidden, mechanism, maximumWork, work);
    if (!forbiddenMember) {
      return std::nullopt;
    }
    if (*forbiddenMember) {
      return true;
    }
  }
  return false;
}

std::optional<bool>
additionsConflictFree(const CanonicalSyncPatternProblem &problem,
                      const std::vector<CanonicalSyncMechanismId> &selected,
                      const std::vector<CanonicalSyncMechanismId> &additions,
                      std::size_t maximumWork, std::size_t &work) {
  for (CanonicalSyncMechanismId mechanism : additions) {
    const auto &conflicts = problem.getMechanisms()[mechanism].conflicts;
    for (CanonicalSyncMechanismId conflict : conflicts) {
      const std::optional<bool> selectedConflict =
          containsSorted(selected, conflict, maximumWork, work);
      const std::optional<bool> additionConflict =
          containsSorted(additions, conflict, maximumWork, work);
      if (!selectedConflict || !additionConflict) {
        return std::nullopt;
      }
      if (*selectedConflict || *additionConflict) {
        return false;
      }
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

bool coverageAfterAdding(const CanonicalSyncPatternProblem &problem,
                         const std::vector<CanonicalSyncMechanismId> &selected,
                         const SyncCoverDemandSet &covered,
                         const std::vector<CanonicalSyncMechanismId> &additions,
                         std::size_t maximumWork, std::size_t &work,
                         SyncCoverDemandSet &result) {
  if (!consumeWork(selected.size() + additions.size(), maximumWork, work)) {
    return false;
  }
  const std::vector<CanonicalSyncMechanismId> successor =
      addMembers(selected, additions);
  std::vector<CanonicalSyncPatternId> affected;
  for (CanonicalSyncMechanismId mechanism : additions) {
    const auto &patterns = problem.getMechanismPatterns()[mechanism];
    if (!consumeWork(patterns.size(), maximumWork, work)) {
      return false;
    }
    affected.insert(affected.end(), patterns.begin(), patterns.end());
  }
  if (!consumeSortWork(affected.size(), maximumWork, work)) {
    return false;
  }
  std::sort(affected.begin(), affected.end());
  affected.erase(std::unique(affected.begin(), affected.end()), affected.end());

  if (!consumeWork(covered.getWords().size(), maximumWork, work)) {
    return false;
  }
  result = covered;
  for (CanonicalSyncPatternId patternId : affected) {
    const CanonicalSyncPattern &pattern = problem.getPatterns()[patternId];
    const bool workAvailable = consumeWork(
        pattern.members.size() + successor.size(), maximumWork, work);
    if (!workAvailable) {
      return false;
    }
    if (patternActive(pattern, successor)) {
      if (!consumeWork(pattern.coverage.getWords().size(), maximumWork, work)) {
        return false;
      }
      result.unite(pattern.coverage);
    }
  }
  return true;
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

CostComputation getCost(const CanonicalSyncPatternProblem &problem,
                        const std::vector<CanonicalSyncMechanismId> &members) {
  CostComputation result;
  result.cost.signature = members;
  result.cost.value.mechanismCount = members.size();
  for (CanonicalSyncMechanismId mechanismId : members) {
    const CanonicalSyncMechanism &mechanism =
        problem.getMechanisms()[mechanismId];
    const CanonicalSyncMechanismCost &cost = mechanism.cost;
    const std::size_t profileSize =
        std::max(cost.barrierActions.size(), cost.eventActions.size());
    result.cost.value.barrierActionProfile.resize(
        std::max(result.cost.value.barrierActionProfile.size(), profileSize),
        0);
    result.cost.value.eventActionProfile.resize(
        std::max(result.cost.value.eventActionProfile.size(), profileSize), 0);
    result.cost.value.actionProfile.resize(
        std::max(result.cost.value.actionProfile.size(), profileSize), 0);
    for (std::size_t depth = 0; depth < cost.barrierActions.size(); ++depth) {
      std::uint64_t nextBarrier = 0;
      std::uint64_t nextTotal = 0;
      if (!checkedCostAdd(result.cost.value.barrierActionProfile[depth],
                          cost.barrierActions[depth], nextBarrier) ||
          !checkedCostAdd(result.cost.value.actionProfile[depth],
                          cost.barrierActions[depth], nextTotal)) {
        result.error = CostComputationError::ArithmeticOverflow;
        return result;
      }
      result.cost.value.barrierActionProfile[depth] = nextBarrier;
      result.cost.value.actionProfile[depth] = nextTotal;
    }
    for (std::size_t depth = 0; depth < cost.eventActions.size(); ++depth) {
      std::uint64_t nextEvent = 0;
      std::uint64_t nextTotal = 0;
      if (!checkedCostAdd(result.cost.value.eventActionProfile[depth],
                          cost.eventActions[depth], nextEvent) ||
          !checkedCostAdd(result.cost.value.actionProfile[depth],
                          cost.eventActions[depth], nextTotal)) {
        result.error = CostComputationError::ArithmeticOverflow;
        return result;
      }
      result.cost.value.eventActionProfile[depth] = nextEvent;
      result.cost.value.actionProfile[depth] = nextTotal;
    }
    std::uint64_t nextSerialization = 0;
    if (!checkedCostAdd(result.cost.value.serializationBreadth,
                        cost.serializationBreadth, nextSerialization)) {
      result.error = CostComputationError::ArithmeticOverflow;
      return result;
    }
    result.cost.value.serializationBreadth = nextSerialization;
    for (std::size_t use = 0; use < mechanism.eventLifetimes.size(); ++use) {
      const CanonicalSyncEventLifetime &lifetime =
          mechanism.eventLifetimes[use];
      if (lifetime.end < lifetime.begin) {
        result.error = CostComputationError::ArithmeticOverflow;
        return result;
      }
      std::uint64_t span = 0;
      if (!checkedCostAdd(
              static_cast<std::uint64_t>(lifetime.end - lifetime.begin), 1,
              span)) {
        result.error = CostComputationError::ArithmeticOverflow;
        return result;
      }
      const std::uint64_t width =
          use < mechanism.descriptor.eventUses.size()
              ? static_cast<std::uint64_t>(
                    mechanism.descriptor.eventUses[use].width)
              : 0;
      std::uint64_t area = 0;
      std::uint64_t nextArea = 0;
      if (!checkedCostMultiply(span, width, area) ||
          !checkedCostAdd(result.cost.value.eventLifetimeArea, area,
                          nextArea)) {
        result.error = CostComputationError::ArithmeticOverflow;
        return result;
      }
      result.cost.value.eventLifetimeArea = nextArea;
    }
  }
  return result;
}

CostComputation
getCostMetered(const CanonicalSyncPatternProblem &problem,
               const std::vector<CanonicalSyncMechanismId> &members,
               std::size_t maximumWork, std::size_t &work) {
  std::size_t costWork = members.size();
  for (CanonicalSyncMechanismId mechanismId : members) {
    const CanonicalSyncMechanism &mechanism =
        problem.getMechanisms()[mechanismId];
    const std::size_t profiles = mechanism.cost.barrierActions.size() +
                                 mechanism.cost.eventActions.size();
    const std::size_t lifetimes = mechanism.eventLifetimes.size();
    if (profiles > std::numeric_limits<std::size_t>::max() - costWork ||
        lifetimes >
            std::numeric_limits<std::size_t>::max() - costWork - profiles) {
      return {CostComputationError::WorkLimitExceeded, {}};
    }
    costWork += profiles + lifetimes;
  }
  if (!consumeWork(costWork, maximumWork, work)) {
    return {CostComputationError::WorkLimitExceeded, {}};
  }
  return getCost(problem, members);
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

int compareActionProfileRatio(const GreedyCandidate &first,
                              const GreedyCandidate &second) {
  const std::size_t depths = std::max(first.cost.value.actionProfile.size(),
                                      second.cost.value.actionProfile.size());
  for (std::size_t reverse = depths; reverse > 0; --reverse) {
    const std::size_t depth = reverse - 1;
    const int order = compareRatio(
        profileValue(first.cost.value.actionProfile, depth), first.gain,
        profileValue(second.cost.value.actionProfile, depth), second.gain);
    if (order != 0) {
      return order;
    }
  }
  return 0;
}

bool candidateLess(const GreedyCandidate &first, const GreedyCandidate &second,
                   CanonicalSyncSelectionObjective objective) {
  const int serialization =
      compareRatio(first.cost.value.serializationBreadth, first.gain,
                   second.cost.value.serializationBreadth, second.gain);
  const int actions = compareActionProfileRatio(first, second);
  if (objective == CanonicalSyncSelectionObjective::SerializationFirst) {
    if (serialization != 0) {
      return serialization < 0;
    }
    if (actions != 0) {
      return actions < 0;
    }
  } else {
    if (actions != 0) {
      return actions < 0;
    }
    if (serialization != 0) {
      return serialization < 0;
    }
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

int compareActionProfile(const CanonicalSyncStructuralCost &first,
                         const CanonicalSyncStructuralCost &second) {
  const std::size_t depths =
      std::max(first.actionProfile.size(), second.actionProfile.size());
  for (std::size_t reverse = depths; reverse > 0; --reverse) {
    const std::size_t depth = reverse - 1;
    const std::uint64_t firstValue = profileValue(first.actionProfile, depth);
    const std::uint64_t secondValue = profileValue(second.actionProfile, depth);
    if (firstValue != secondValue) {
      return firstValue < secondValue ? -1 : 1;
    }
  }
  return 0;
}

bool costLess(const CandidateCost &first, const CandidateCost &second,
              CanonicalSyncSelectionObjective objective) {
  if (canonicalSyncStructuralCostLess(first.value, second.value, objective)) {
    return true;
  }
  if (canonicalSyncStructuralCostLess(second.value, first.value, objective)) {
    return false;
  }
  return first.signature < second.signature;
}

bool considerCandidate(const CanonicalSyncPatternProblem &problem,
                       const std::vector<CanonicalSyncMechanismId> &selected,
                       const SyncCoverDemandSet &covered,
                       CanonicalSyncPatternId pattern,
                       const std::vector<CanonicalSyncMechanismId> &additions,
                       const std::vector<CanonicalSyncMechanismId> &costMembers,
                       const SyncCoverDemandSet &successorCoverage,
                       const CanonicalSyncGreedyOptions &options,
                       CanonicalSyncGreedyStatistics &statistics,
                       GreedyCandidateScan &scan) {
  const bool workAvailable =
      consumeWork(additions.size() + costMembers.size() + 1,
                  options.maximumWorkUnits, statistics.workUnits);
  if (!workAvailable) {
    return false;
  }
  const std::optional<bool> conflictFree =
      additionsConflictFree(problem, selected, additions,
                            options.maximumWorkUnits, statistics.workUnits);
  if (!conflictFree) {
    return false;
  }
  if (!*conflictFree) {
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
  const bool evaluationCountOverflows =
      statistics.patternEvaluations == std::numeric_limits<std::size_t>::max();
  if (evaluationCountOverflows ||
      !consumeWork(1, options.maximumWorkUnits, statistics.workUnits)) {
    return false;
  }
  ++statistics.patternEvaluations;

  const CostComputation cost = getCostMetered(
      problem, costMembers, options.maximumWorkUnits, statistics.workUnits);
  if (!cost) {
    statistics.arithmeticOverflow =
        cost.error == CostComputationError::ArithmeticOverflow;
    return false;
  }
  GreedyCandidate candidate{pattern, additions, cost.cost, *gain};
  // candidateLess includes the stable identity tie-break. A streaming minimum
  // therefore chooses exactly the same move as sorting the complete catalog.
  const std::size_t comparisonWork =
      !scan.best
          ? 1
          : std::max(candidate.cost.value.actionProfile.size(),
                     scan.best->cost.value.actionProfile.size()) +
                candidate.additions.size() + scan.best->additions.size() + 4;
  if (!consumeWork(comparisonWork, options.maximumWorkUnits,
                   statistics.workUnits)) {
    return false;
  }
  const bool improvesBest =
      !scan.best || candidateLess(candidate, *scan.best, options.objective);
  if (!improvesBest) {
    return true;
  }
  if (!consumeWork(successorCoverage.getWords().size(),
                   options.maximumWorkUnits, statistics.workUnits)) {
    return false;
  }
  scan.best = std::move(candidate);
  scan.bestCoverage = successorCoverage;
  return true;
}

bool prepareFixedColumnCoverage(const CanonicalSyncPatternProblem &problem,
                                const CanonicalSyncPattern &pattern,
                                const SyncCoverDemandSet &covered,
                                std::size_t maximumWork, std::size_t &work,
                                SyncCoverDemandSet &result) {
  if (!consumeWork(covered.getWords().size(), maximumWork, work)) {
    return false;
  }
  result = covered;

  // Freeze stores a composite's complete joint coverage unavailable from its
  // member singletons. Its fixed column is consequently the union of those
  // singleton columns and its own extra bits. Reconstructing it this way avoids
  // rescanning the complete pattern catalog for every candidate and round.
  for (CanonicalSyncMechanismId member : pattern.members) {
    if (member >= problem.getPatterns().size()) {
      return false;
    }
    const CanonicalSyncPattern &singleton = problem.getPatterns()[member];
    const bool invalidSingleton =
        singleton.kind != CanonicalSyncPatternKind::Singleton ||
        singleton.members.size() != 1 || singleton.members.front() != member;
    if (invalidSingleton ||
        !consumeWork(singleton.coverage.getWords().size(), maximumWork, work)) {
      return false;
    }
    result.unite(singleton.coverage);
  }
  if (pattern.kind != CanonicalSyncPatternKind::Singleton) {
    if (!consumeWork(pattern.coverage.getWords().size(), maximumWork, work)) {
      return false;
    }
    result.unite(pattern.coverage);
  }
  return true;
}

bool buildFixedCandidates(const CanonicalSyncPatternProblem &problem,
                          const std::vector<CanonicalSyncMechanismId> &selected,
                          const SyncCoverDemandSet &covered,
                          const CanonicalSyncGreedyOptions &options,
                          CanonicalSyncGreedyStatistics &statistics,
                          GreedyCandidateScan &scan) {
  for (const CanonicalSyncPattern &pattern : problem.getPatterns()) {
    const std::optional<bool> forbidden =
        containsForbidden(pattern.members, options.forbiddenMechanisms,
                          options.maximumWorkUnits, statistics.workUnits);
    if (!forbidden) {
      return false;
    }
    if (*forbidden) {
      continue;
    }
    std::optional<std::vector<CanonicalSyncMechanismId>> additions =
        missingMembers(selected, pattern.members, options.maximumWorkUnits,
                       statistics.workUnits);
    if (!additions) {
      return false;
    }
    if (additions->empty()) {
      continue;
    }
    if (!prepareFixedColumnCoverage(
            problem, pattern, covered, options.maximumWorkUnits,
            statistics.workUnits, scan.scratchCoverage)) {
      return false;
    }
    if (!considerCandidate(problem, selected, covered, pattern.id, *additions,
                           pattern.members, scan.scratchCoverage, options,
                           statistics, scan)) {
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
    CanonicalSyncGreedyStatistics &statistics, GreedyCandidateScan &scan) {
  if (!coverageAfterAdding(problem, selected, covered, additions,
                           options.maximumWorkUnits, statistics.workUnits,
                           scan.scratchCoverage)) {
    return false;
  }
  return considerCandidate(problem, selected, covered, pattern, additions,
                           additions, scan.scratchCoverage, options, statistics,
                           scan);
}

bool buildActionAwareCandidates(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected,
    const SyncCoverDemandSet &covered,
    const CanonicalSyncGreedyOptions &options,
    CanonicalSyncGreedyStatistics &statistics, GreedyCandidateScan &scan) {
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    const std::optional<bool> alreadySelected = containsSorted(
        selected, mechanism.id, options.maximumWorkUnits, statistics.workUnits);
    const std::optional<bool> forbidden =
        containsSorted(options.forbiddenMechanisms, mechanism.id,
                       options.maximumWorkUnits, statistics.workUnits);
    if (!alreadySelected || !forbidden) {
      return false;
    }
    if (*alreadySelected || *forbidden) {
      continue;
    }
    if (!buildActionAwareCandidate(problem, selected, covered, {mechanism.id},
                                   mechanism.id, options, statistics, scan)) {
      return false;
    }
  }
  if (options.strategy != CanonicalSyncSelectionStrategy::PairLookahead) {
    return true;
  }
  for (const CanonicalSyncPattern &pattern : problem.getPatterns()) {
    const bool isPair = pattern.members.size() == 2;
    const std::optional<bool> forbidden =
        containsForbidden(pattern.members, options.forbiddenMechanisms,
                          options.maximumWorkUnits, statistics.workUnits);
    if (!forbidden) {
      return false;
    }
    if (!isPair || *forbidden) {
      continue;
    }
    std::optional<std::vector<CanonicalSyncMechanismId>> additions =
        missingMembers(selected, pattern.members, options.maximumWorkUnits,
                       statistics.workUnits);
    if (!additions) {
      return false;
    }
    const bool addsBothMembers = additions->size() == 2;
    if (!addsBothMembers) {
      continue;
    }
    if (!buildActionAwareCandidate(problem, selected, covered,
                                   std::move(*additions), pattern.id, options,
                                   statistics, scan)) {
      return false;
    }
  }
  return true;
}

std::optional<bool> complete(const CanonicalSyncPatternProblem &problem,
                             const SyncCoverDemandSet &covered,
                             std::size_t maximumWork, std::size_t &work) {
  if (!consumeWork(covered.getWords().size(), maximumWork, work)) {
    return std::nullopt;
  }
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

bool mlir::pto::canonicalSyncStructuralCostLess(
    const CanonicalSyncStructuralCost &first,
    const CanonicalSyncStructuralCost &second,
    CanonicalSyncSelectionObjective objective) {
  const int actions = compareActionProfile(first, second);
  if (objective == CanonicalSyncSelectionObjective::SerializationFirst) {
    if (first.serializationBreadth != second.serializationBreadth) {
      return first.serializationBreadth < second.serializationBreadth;
    }
    if (actions != 0) {
      return actions < 0;
    }
  } else {
    if (actions != 0) {
      return actions < 0;
    }
    if (first.serializationBreadth != second.serializationBreadth) {
      return first.serializationBreadth < second.serializationBreadth;
    }
  }
  return std::tie(first.eventLifetimeArea, first.mechanismCount) <
         std::tie(second.eventLifetimeArea, second.mechanismCount);
}

CanonicalSyncRepairRoundRanker::Decision
CanonicalSyncRepairRoundRanker::consider(
    const CanonicalSyncSelection &trial, bool freshlyVerified,
    std::size_t diagnosedResourceOverflow,
    SyncCoverCoverageWorkBudget *workBudget) {
  const auto rankLess = [&](const RankKey &first, const RankKey &second,
                            bool comparePressure) {
    if (comparePressure && first.resourceOverflow != second.resourceOverflow) {
      return first.resourceOverflow < second.resourceOverflow;
    }
    if (canonicalSyncStructuralCostLess(first.cost, second.cost, objective_)) {
      return true;
    }
    if (canonicalSyncStructuralCostLess(second.cost, first.cost, objective_)) {
      return false;
    }
    return first.mechanisms < second.mechanisms;
  };
  const bool verifiedCandidate =
      freshlyVerified && trial.error == CanonicalSyncSelectionError::None;
  const bool pressureCandidate =
      !freshlyVerified &&
      trial.error == CanonicalSyncSelectionError::ResourceInfeasible &&
      trial.allocation.valid && !trial.allocation.feasible;
  if (!verifiedCandidate && !pressureCandidate) {
    return Decision::Discard;
  }
  if (pressureCandidate &&
      diagnosedResourceOverflow >= baselineResourceOverflow_) {
    return Decision::Discard;
  }
  const std::optional<RankKey> &incumbent =
      verifiedCandidate ? bestVerified_ : bestPressure_;
  std::size_t copiedProfiles = 0;
  std::size_t comparedProfiles = 0;
  std::size_t mechanismWork = 0;
  std::size_t rankWork = 0;
  const bool profileCopiesAvailable =
      checkedWorkSum(trial.cost.barrierActionProfile.size(),
                     trial.cost.eventActionProfile.size(), copiedProfiles) &&
      checkedWorkSum(copiedProfiles, trial.cost.actionProfile.size(),
                     copiedProfiles);
  bool comparisonWorkAvailable = true;
  mechanismWork = trial.mechanisms.size();
  if (incumbent) {
    const std::size_t comparedDepths = std::max(
        trial.cost.actionProfile.size(), incumbent->cost.actionProfile.size());
    comparisonWorkAvailable =
        checkedWorkProduct(comparedDepths, 2, comparedProfiles) &&
        checkedWorkSum(mechanismWork, incumbent->mechanisms.size(),
                       mechanismWork);
  }
  const bool rankWorkAvailable =
      profileCopiesAvailable && comparisonWorkAvailable &&
      checkedWorkSum(copiedProfiles, comparedProfiles, rankWork) &&
      checkedWorkSum(rankWork, mechanismWork, rankWork) &&
      checkedWorkSum(rankWork, 4, rankWork) &&
      (!workBudget || workBudget->consume(rankWork));
  if (!rankWorkAvailable) {
    return Decision::WorkLimitExceeded;
  }
  RankKey candidate;
  candidate.cost = trial.cost;
  candidate.mechanisms = trial.mechanisms;
  if (verifiedCandidate) {
    if (bestVerified_ && !rankLess(candidate, *bestVerified_, false)) {
      return Decision::Discard;
    }
    bestVerified_ = std::move(candidate);
    return Decision::ReplaceBestVerified;
  }
  candidate.resourceOverflow = diagnosedResourceOverflow;
  if (bestPressure_ && !rankLess(candidate, *bestPressure_, true)) {
    return Decision::Discard;
  }
  bestPressure_ = std::move(candidate);
  return Decision::ReplaceBestPressure;
}

std::optional<std::vector<CanonicalSyncMechanismId>>
CanonicalSyncRepairRoundRanker::getBestVerifiedMechanisms() const {
  return bestVerified_ ? std::optional<std::vector<CanonicalSyncMechanismId>>(
                             bestVerified_->mechanisms)
                       : std::nullopt;
}

std::optional<std::vector<CanonicalSyncMechanismId>>
CanonicalSyncRepairRoundRanker::getBestPressureMechanisms() const {
  return bestPressure_ ? std::optional<std::vector<CanonicalSyncMechanismId>>(
                             bestPressure_->mechanisms)
                       : std::nullopt;
}

std::optional<CanonicalSyncStructuralCost>
mlir::pto::computeCanonicalSyncStructuralCost(
    const CanonicalSyncPatternProblem &problem,
    const std::vector<CanonicalSyncMechanismId> &selected) {
  CostComputation result = getCost(problem, selected);
  if (!result) {
    return std::nullopt;
  }
  return std::move(result.cost.value);
}

CanonicalSyncSelection mlir::pto::selectCanonicalSyncPatterns(
    const CanonicalSyncPatternProblem &problem,
    CanonicalSyncGreedyOptions options) {
  CanonicalSyncSelection result;
  result.covered = problem.getBaselineCoverage();
  if (!problem.isFrozen() || options.maximumWorkUnits == 0) {
    result.error = CanonicalSyncSelectionError::InvalidProblem;
    return result;
  }
  if (!consumeSortWork(options.forbiddenMechanisms.size(),
                       options.maximumWorkUnits, result.statistics.workUnits)) {
    result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
    return result;
  }
  std::sort(options.forbiddenMechanisms.begin(),
            options.forbiddenMechanisms.end());
  options.forbiddenMechanisms.erase(
      std::unique(options.forbiddenMechanisms.begin(),
                  options.forbiddenMechanisms.end()),
      options.forbiddenMechanisms.end());
  const bool invalidForbidden =
      !options.forbiddenMechanisms.empty() &&
      options.forbiddenMechanisms.back() >= problem.getMechanisms().size();
  if (invalidForbidden) {
    result.error = CanonicalSyncSelectionError::InvalidProblem;
    return result;
  }

  std::vector<CanonicalSyncMechanismId> selected;
  GreedyCandidateScan candidateScan(problem.getDemands().size());
  for (;;) {
    const std::optional<bool> isComplete =
        complete(problem, result.covered, options.maximumWorkUnits,
                 result.statistics.workUnits);
    if (!isComplete) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    if (*isComplete) {
      break;
    }
    candidateScan.reset();
    const bool built =
        options.strategy == CanonicalSyncSelectionStrategy::FixedCover
            ? buildFixedCandidates(problem, selected, result.covered, options,
                                   result.statistics, candidateScan)
            : buildActionAwareCandidates(problem, selected, result.covered,
                                         options, result.statistics,
                                         candidateScan);
    if (!built) {
      result.error = result.statistics.arithmeticOverflow
                         ? CanonicalSyncSelectionError::ArithmeticOverflow
                         : CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    if (!candidateScan.best) {
      result.error = CanonicalSyncSelectionError::NoCoveringPattern;
      return result;
    }
    GreedyCandidate chosen = std::move(*candidateScan.best);
    if (!consumeWork(selected.size() + chosen.additions.size(),
                     options.maximumWorkUnits, result.statistics.workUnits)) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    result.selectionOrder.insert(result.selectionOrder.end(),
                                 chosen.additions.begin(),
                                 chosen.additions.end());
    selected = addMembers(selected, chosen.additions);
    std::swap(result.covered, candidateScan.bestCoverage);
  }

  CostComputation selectedCost = getCostMetered(
      problem, selected, options.maximumWorkUnits, result.statistics.workUnits);
  if (!selectedCost) {
    result.statistics.arithmeticOverflow =
        selectedCost.error == CostComputationError::ArithmeticOverflow;
    result.error = result.statistics.arithmeticOverflow
                       ? CanonicalSyncSelectionError::ArithmeticOverflow
                       : CanonicalSyncSelectionError::WorkLimitExceeded;
    return result;
  }
  for (auto position = result.selectionOrder.rbegin();
       position != result.selectionOrder.rend(); ++position) {
    if (!consumeWork(selected.size(), options.maximumWorkUnits,
                     result.statistics.workUnits)) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
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
    const CostComputation trialCost = getCostMetered(
        problem, trial, options.maximumWorkUnits, result.statistics.workUnits);
    if (!trialCost) {
      result.statistics.arithmeticOverflow =
          trialCost.error == CostComputationError::ArithmeticOverflow;
      result.error = result.statistics.arithmeticOverflow
                         ? CanonicalSyncSelectionError::ArithmeticOverflow
                         : CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    const std::optional<bool> remainsComplete =
        complete(problem, *trialCoverage, options.maximumWorkUnits,
                 result.statistics.workUnits);
    if (!remainsComplete) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    const std::size_t comparisonWork =
        std::max(trialCost.cost.value.actionProfile.size(),
                 selectedCost.cost.value.actionProfile.size()) +
        4;
    if (!consumeWork(comparisonWork, options.maximumWorkUnits,
                     result.statistics.workUnits)) {
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    const bool reducesCost =
        costLess(trialCost.cost, selectedCost.cost, options.objective);
    if (*remainsComplete && reducesCost) {
      selected = std::move(trial);
      result.covered = *trialCoverage;
      selectedCost = trialCost;
    }
  }

  result.mechanisms = std::move(selected);
  result.cost = selectedCost.cost.value;
  const std::size_t remainingWork =
      options.maximumWorkUnits - result.statistics.workUnits;
  if (remainingWork == 0) {
    result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
    return result;
  }
  SyncCoverCoverageWorkBudget allocationWork(remainingWork);
  result.allocation =
      allocateCanonicalSyncEvents(problem, result.mechanisms, &allocationWork);
  result.statistics.workUnits += allocationWork.workUnits;
  if (allocationWork.exhausted) {
    result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
    return result;
  }
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
  const SyncCoverCoverageResult basisCoverage =
      computeSyncCoverCoverage(problem.getGraph(), problem.getExpansion(),
                               supplies, problem.getDemands(), coverageWork);
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
    const SyncCoverDemand &demand =
        problem.getGraph().getDemands()[problem.getDemands()[index]];
    const bool guardMetadataOverflows =
        demand.targetGuard.literals.size() >
        std::numeric_limits<std::size_t>::max() -
            demand.sourceGuard.literals.size();
    const std::size_t guardMetadata =
        guardMetadataOverflows ? 0
                               : demand.sourceGuard.literals.size() +
                                     demand.targetGuard.literals.size();
    if (guardMetadataOverflows ||
        (coverageWork &&
         (!coverageWork->consume() || !coverageWork->consume(guardMetadata)))) {
      if (coverageWork && guardMetadataOverflows) {
        coverageWork->exhausted = true;
      }
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    SyncCoverCompletionSupply lemma;
    lemma.mechanism = lemmaMechanismBegin + index;
    lemma.edge = {
        demand.source,     demand.target,   SyncCoverEdgeKind::CompletionSupply,
        demand.scope,      demand.distance, demand.sourceGuard,
        demand.targetGuard};
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
