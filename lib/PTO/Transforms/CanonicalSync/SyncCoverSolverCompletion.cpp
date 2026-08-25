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

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_internal;

namespace {

struct CompletionState {
  std::vector<SyncCoverMechanismId> selected;
  SyncCoverStructuralCost cost;
  CoverageEvaluation coverage;
};

struct RankedCandidates {
  bool blocked = false;
  bool clipped = false;
  std::vector<SyncCoverMechanismId> mechanisms;
  std::optional<SyncCoverDemandId> blockedDemand;
  std::vector<SyncCoverMechanismId> blockedCut;
  std::vector<SyncCoverReachableState> blockedReachableStates;
};

struct CompletionAttempt {
  std::optional<CompletionState> state;
  std::optional<SyncCoverCompletionRejection> rejection;
};

constexpr std::size_t kMaximumRecordedRejections = 64;

bool stateLess(const CompletionState &first, const CompletionState &second) {
  if (first.coverage.uncovered.size() != second.coverage.uncovered.size()) {
    return first.coverage.uncovered.size() < second.coverage.uncovered.size();
  }
  if (syncCoverStructuralCostLess(first.cost, second.cost)) {
    return true;
  }
  if (syncCoverStructuralCostLess(second.cost, first.cost)) {
    return false;
  }
  return first.selected < second.selected;
}

bool isCanonicalSet(const std::vector<SyncCoverMechanismId> &values) {
  return std::is_sorted(values.begin(), values.end()) &&
         std::adjacent_find(values.begin(), values.end()) == values.end();
}

std::vector<SyncCoverMechanismId>
addMechanism(const std::vector<SyncCoverMechanismId> &selected,
             SyncCoverMechanismId mechanism) {
  std::vector<SyncCoverMechanismId> result = selected;
  result.insert(std::lower_bound(result.begin(), result.end(), mechanism),
                mechanism);
  return result;
}

SyncCoverCompletionRejection makeResourceRejection(
    SyncCoverMechanismId mechanism,
    const SyncCoverResourceSelection &resources) {
  SyncCoverCompletionRejection rejection;
  rejection.mechanism = mechanism;
  rejection.kind = resources.error == SyncCoverResourceSelectionError::None
                       ? SyncCoverCompletionRejectionKind::ResourceInfeasible
                       : SyncCoverCompletionRejectionKind::InvalidSelection;
  rejection.resourceError = resources.error;
  rejection.firstConflict = resources.firstConflict;
  rejection.secondConflict = resources.secondConflict;
  for (const SyncCoverDomainFeasibility &domain : resources.domains) {
    if (domain.overflow == 0) {
      continue;
    }
    if (!rejection.domain || domain.overflow >
                                 rejection.required - rejection.available) {
      rejection.domain = domain.domain;
      rejection.required = domain.required;
      rejection.available = domain.available;
    }
  }
  return rejection;
}

void recordRejection(SyncCoverCompletionResult &result,
                     SyncCoverCompletionRejection rejection) {
  if (result.rejections.size() == kMaximumRecordedRejections) {
    result.rejectionDiagnosticsTruncated = true;
    return;
  }
  result.rejections.push_back(std::move(rejection));
}

void recordBlockedCut(SyncCoverCompletionResult &result,
                      const RankedCandidates &candidates,
                      const std::vector<SyncCoverMechanismId> &selected) {
  if (!candidates.blockedDemand) {
    return;
  }
  const SyncCoverCompletionBlockedCut blocked{
      *candidates.blockedDemand, selected, candidates.blockedCut,
      candidates.blockedReachableStates};
  if (std::find_if(result.blockedCuts.begin(), result.blockedCuts.end(),
                   [&](const auto &known) {
                     return known.demand == blocked.demand &&
                            known.selected == blocked.selected &&
                            known.mechanisms == blocked.mechanisms;
                   }) != result.blockedCuts.end()) {
    return;
  }
  if (result.blockedCuts.size() == kMaximumRecordedRejections) {
    result.blockedCutDiagnosticsTruncated = true;
    return;
  }
  result.blockedCuts.push_back(blocked);
}

RankedCandidates rankCandidates(
    const CompletionState &state,
    const std::vector<SyncCoverMechanismId> &allowed,
    const SyncCoverMechanismUniverse &universe, std::size_t candidateLimit) {
  struct Ranked {
    SyncCoverMechanismId mechanism = 0;
    std::size_t support = 0;
    SyncCoverStructuralCost cost;
  };
  RankedCandidates result;
  std::map<SyncCoverMechanismId, std::size_t> support;
  for (SyncCoverDemandId demand : state.coverage.uncovered) {
    auto cut = state.coverage.cuts.find(demand);
    if (cut == state.coverage.cuts.end()) {
      continue;
    }
    for (SyncCoverMechanismId mechanism : cut->second) {
      const bool available =
          std::binary_search(allowed.begin(), allowed.end(), mechanism) &&
          !std::binary_search(state.selected.begin(), state.selected.end(),
                              mechanism);
      if (available) {
        ++support[mechanism];
      }
    }
  }
  std::optional<std::vector<SyncCoverMechanismId>> branch;
  for (SyncCoverDemandId demand : state.coverage.uncovered) {
    auto cut = state.coverage.cuts.find(demand);
    if (cut == state.coverage.cuts.end()) {
      result.blocked = true;
      result.blockedDemand = demand;
      auto reachable = state.coverage.reachableStates.find(demand);
      if (reachable != state.coverage.reachableStates.end()) {
        result.blockedReachableStates = reachable->second;
      }
      return result;
    }
    std::vector<SyncCoverMechanismId> candidates;
    for (SyncCoverMechanismId mechanism : cut->second) {
      const bool available =
          std::binary_search(allowed.begin(), allowed.end(), mechanism) &&
          !std::binary_search(state.selected.begin(), state.selected.end(),
                              mechanism);
      if (available) {
        candidates.push_back(mechanism);
      }
    }
    if (candidates.empty()) {
      result.blocked = true;
      result.blockedDemand = demand;
      result.blockedCut = cut->second;
      auto reachable = state.coverage.reachableStates.find(demand);
      if (reachable != state.coverage.reachableStates.end()) {
        result.blockedReachableStates = reachable->second;
      }
      return result;
    }
    if (!branch || candidates.size() < branch->size() ||
        (candidates.size() == branch->size() && candidates < *branch)) {
      branch = std::move(candidates);
    }
  }
  if (!branch) {
    return result;
  }
  std::vector<Ranked> ranked;
  ranked.reserve(branch->size());
  for (SyncCoverMechanismId mechanism : *branch) {
    ranked.push_back(
        {mechanism, support[mechanism],
         universe.evaluateStructuralCost({mechanism})});
  }
  std::sort(ranked.begin(), ranked.end(), [](const Ranked &first,
                                             const Ranked &second) {
    if (first.support != second.support) {
      return first.support > second.support;
    }
    if (syncCoverStructuralCostLess(first.cost, second.cost)) {
      return true;
    }
    if (syncCoverStructuralCostLess(second.cost, first.cost)) {
      return false;
    }
    return first.mechanism < second.mechanism;
  });
  if (ranked.size() > candidateLimit) {
    ranked.resize(candidateLimit);
    result.clipped = true;
  }
  result.mechanisms.reserve(ranked.size());
  for (const Ranked &entry : ranked) {
    result.mechanisms.push_back(entry.mechanism);
  }
  return result;
}

} // namespace

SyncCoverCompletionResult mlir::pto::completeSyncCoverMembership(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &fixed,
    const std::vector<SyncCoverMechanismId> &allowed,
    const SyncCoverCompletionOptions &options) {
  SyncCoverCompletionResult result;
  const bool invalidOptions =
      options.beamWidth == 0 || options.depthLimit == 0 ||
      options.candidateLimit == 0 || options.evaluationLimit == 0;
  const bool invalidSelection =
      !isCanonicalSet(fixed) || !isCanonicalSet(allowed) ||
      (!fixed.empty() && fixed.back() >= universe.getMechanisms().size()) ||
      (!allowed.empty() &&
       allowed.back() >= universe.getMechanisms().size()) ||
      !std::includes(allowed.begin(), allowed.end(), fixed.begin(),
                     fixed.end());
  if (invalidOptions || invalidSelection) {
    result.error = SyncCoverMembershipError::InvalidSelection;
    return result;
  }

  const SyncCoverSelectionEvaluator selectionEvaluator(universe);
  if (!selectionEvaluator) {
    result.error = SyncCoverMembershipError::InvalidUniverse;
    return result;
  }
  SyncCoverCoverageOracle oracle(universe.getGraph());
  CoverageEvaluator coverage(oracle);
  const auto evaluate = [&](const std::vector<SyncCoverMechanismId> &selected,
                            std::optional<SyncCoverMechanismId> added)
      -> CompletionAttempt {
    const SyncCoverSelectionEvaluation selection =
        selectionEvaluator.evaluate(selected);
    if (!selection || !selection.resources.resourceFeasible) {
      CompletionAttempt attempt;
      if (added) {
        attempt.rejection =
            makeResourceRejection(*added, selection.resources);
      }
      return attempt;
    }
    CoverageEvaluation completion = coverage.evaluate(activeDemands, selected);
    if (!completion.valid) {
      CompletionAttempt attempt;
      if (added) {
        attempt.rejection = SyncCoverCompletionRejection{
            *added, SyncCoverCompletionRejectionKind::CoverageFailure};
      }
      return attempt;
    }
    return CompletionAttempt{
        CompletionState{selected, selection.cost, std::move(completion)},
        std::nullopt};
  };

  CompletionAttempt initialAttempt = evaluate(fixed, std::nullopt);
  ++result.evaluations;
  if (!initialAttempt.state) {
    result.error = SyncCoverMembershipError::InvalidSelection;
    return result;
  }
  std::vector<CompletionState> frontier = {
      std::move(*initialAttempt.state)};
  std::set<std::vector<SyncCoverMechanismId>> seen = {fixed};
  std::optional<CompletionState> best;
  for (std::size_t depth = 0; depth <= options.depthLimit; ++depth) {
    std::vector<CompletionState> successors;
    for (CompletionState &state : frontier) {
      if (state.coverage.uncovered.empty()) {
        if (!best || stateLess(state, *best)) {
          best = state;
        }
        continue;
      }
      if (depth == options.depthLimit) {
        result.truncated = true;
        continue;
      }
      const RankedCandidates candidates = rankCandidates(
          state, allowed, universe, options.candidateLimit);
      result.truncated |= candidates.clipped;
      if (candidates.blocked || candidates.mechanisms.empty()) {
        recordBlockedCut(result, candidates, state.selected);
        continue;
      }
      for (SyncCoverMechanismId mechanism : candidates.mechanisms) {
        if (result.evaluations == options.evaluationLimit) {
          result.truncated = true;
          break;
        }
        std::vector<SyncCoverMechanismId> selected =
            addMechanism(state.selected, mechanism);
        if (!seen.insert(selected).second) {
          continue;
        }
        ++result.evaluations;
        CompletionAttempt attempt = evaluate(selected, mechanism);
        if (!attempt.state) {
          if (attempt.rejection) {
            recordRejection(result, std::move(*attempt.rejection));
          }
          continue;
        }
        std::optional<CompletionState> successor =
            std::move(attempt.state);
        if (successor->coverage.uncovered.empty()) {
          if (!best || stateLess(*successor, *best)) {
            best = std::move(*successor);
          }
        } else {
          successors.push_back(std::move(*successor));
        }
      }
      if (result.evaluations == options.evaluationLimit) {
        break;
      }
    }
    if (best || successors.empty() ||
        result.evaluations == options.evaluationLimit) {
      result.truncated |=
          result.evaluations == options.evaluationLimit;
      break;
    }
    std::sort(successors.begin(), successors.end(), stateLess);
    if (successors.size() > options.beamWidth) {
      successors.resize(options.beamWidth);
      result.truncated = true;
    }
    frontier = std::move(successors);
  }
  if (!best) {
    return result;
  }

  result.membership = evaluateSyncCoverMembership(
      universe, activeDemands, best->selected);
  if (!result.membership || !result.membership.coverageComplete ||
      !result.membership.resources.resourceFeasible) {
    result.error = result.membership.error == SyncCoverMembershipError::None
                       ? SyncCoverMembershipError::CoverageFailure
                       : result.membership.error;
    return result;
  }
  result.complete = true;
  result.mechanisms = std::move(best->selected);
  return result;
}
