// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverMechanism.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

using namespace mlir::pto;

namespace {

SyncCoverStructuralCost makeCostError(SyncCoverStructuralCostError error) {
  SyncCoverStructuralCost result;
  result.error = error;
  return result;
}

std::optional<std::size_t>
getActionLoopDepth(const SyncCoverGraph &graph,
                   const SyncCoverResourceAction &action) {
  switch (action.anchor.kind) {
  case SyncCoverAnchorKind::BeforeNode:
  case SyncCoverAnchorKind::AfterNode:
    if (action.anchor.node >= graph.getNodes().size()) {
      return std::nullopt;
    }
    return graph.getScopeLoopDepth(graph.getNodes()[action.anchor.node].scope);
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit:
  case SyncCoverAnchorKind::TimelinePoint:
    if (action.anchor.scope >= graph.getScopes().size()) {
      return std::nullopt;
    }
    return graph.getScopeLoopDepth(action.anchor.scope, false);
  }
  return std::nullopt;
}

bool incrementProfile(std::vector<std::size_t> &profile,
                      std::size_t maximumDepth, std::size_t depth,
                      std::size_t amount = 1) {
  if (depth > maximumDepth) {
    return false;
  }
  std::size_t &count = profile[maximumDepth - depth];
  if (amount > std::numeric_limits<std::size_t>::max() - count) {
    return false;
  }
  count += amount;
  return true;
}

SyncCoverStructuralCostError
mapResourceError(SyncCoverResourceSelectionError error) {
  switch (error) {
  case SyncCoverResourceSelectionError::None:
    return SyncCoverStructuralCostError::None;
  case SyncCoverResourceSelectionError::InvalidUniverse:
    return SyncCoverStructuralCostError::InvalidUniverse;
  case SyncCoverResourceSelectionError::InvalidSelection:
    return SyncCoverStructuralCostError::InvalidSelection;
  case SyncCoverResourceSelectionError::Conflict:
    return SyncCoverStructuralCostError::Conflict;
  case SyncCoverResourceSelectionError::ArithmeticOverflow:
    return SyncCoverStructuralCostError::ArithmeticOverflow;
  }
  return SyncCoverStructuralCostError::InvalidUniverse;
}

int compareLoopProfiles(const SyncCoverStructuralCost &first,
                        const SyncCoverStructuralCost &second) {
  const std::size_t profileSize =
      std::max({first.actionProfile.size(), first.barrierActionProfile.size(),
                second.actionProfile.size(),
                second.barrierActionProfile.size()});
  const auto get = [](const std::vector<std::size_t> &profile,
                      std::size_t index) {
    return index < profile.size() ? profile[index] : 0;
  };
  for (std::size_t index = 0; index < profileSize; ++index) {
    const auto firstDepth =
        std::make_pair(get(first.barrierActionProfile, index),
                       get(first.actionProfile, index));
    const auto secondDepth =
        std::make_pair(get(second.barrierActionProfile, index),
                       get(second.actionProfile, index));
    if (firstDepth < secondDepth) {
      return -1;
    }
    if (secondDepth < firstDepth) {
      return 1;
    }
  }
  return 0;
}

} // namespace

bool mlir::pto::syncCoverStructuralCostLess(
    const SyncCoverStructuralCost &first,
    const SyncCoverStructuralCost &second) {
  if (first.error != second.error) {
    return first.error < second.error;
  }
  const int profileOrder = compareLoopProfiles(first, second);
  if (profileOrder != 0) {
    return profileOrder < 0;
  }
  return std::tie(first.mechanismCount, first.signature) <
         std::tie(second.mechanismCount, second.signature);
}

SyncCoverStructuralCost SyncCoverMechanismUniverse::evaluateStructuralCost(
    const std::vector<SyncCoverMechanismId> &selected) const {
  const SyncCoverResourceSelection resources =
      evaluateResourceSelection(selected);
  return evaluateStructuralCostImpl(selected, resources);
}

SyncCoverStructuralCost SyncCoverMechanismUniverse::evaluateStructuralCostImpl(
    const std::vector<SyncCoverMechanismId> &selected,
    const SyncCoverResourceSelection &resources) const {
  if (!resources.isValid()) {
    return makeCostError(mapResourceError(resources.error));
  }
  if (!resources.resourceFeasible) {
    return makeCostError(SyncCoverStructuralCostError::ResourceInfeasible);
  }

  SyncCoverStructuralCost result;
  result.error = SyncCoverStructuralCostError::None;
  result.signature = selected;
  std::sort(result.signature.begin(), result.signature.end());
  result.mechanismCount = result.signature.size();
  std::size_t maximumDepth = 0;
  for (const SyncCoverScope &scope : graph_.getScopes()) {
    const std::optional<std::size_t> depth =
        graph_.getScopeLoopDepth(scope.id);
    if (!depth) {
      return makeCostError(SyncCoverStructuralCostError::InvalidUniverse);
    }
    maximumDepth = std::max(maximumDepth, *depth);
  }
  result.actionProfile.assign(maximumDepth + 1, 0);
  result.barrierActionProfile.assign(maximumDepth + 1, 0);

  std::set<std::uint32_t> issueResources;
  for (const SyncCoverNode &node : graph_.getNodes()) {
    issueResources.insert(node.resource);
  }
  const std::size_t allResourceBarrierWeight =
      std::max<std::size_t>(issueResources.size(), 2);

  for (SyncCoverMechanismId mechanismId : result.signature) {
    const SyncCoverMechanism &mechanism = mechanisms_[mechanismId];
    for (const SyncCoverResourceAction &action : mechanism.actions) {
      const std::optional<std::size_t> depth =
          getActionLoopDepth(graph_, action);
      if (!depth ||
          !incrementProfile(result.actionProfile, maximumDepth, *depth)) {
        return makeCostError(SyncCoverStructuralCostError::ArithmeticOverflow);
      }
    }
    if (mechanism.barrier) {
      const std::optional<std::size_t> depth =
          graph_.getScopeLoopDepth(mechanism.barrier->scope);
      // A barrier that drains every issue resource stalls all pipes, not
      // one; weight it by the number of distinct issue resources so the
      // solver never prefers it over a same-depth targeted barrier.
      const std::size_t weight = mechanism.barrier->drainsAllResources
                                     ? allResourceBarrierWeight
                                     : 1;
      if (!depth || !incrementProfile(result.barrierActionProfile,
                                      maximumDepth, *depth, weight)) {
        return makeCostError(SyncCoverStructuralCostError::ArithmeticOverflow);
      }
    }
  }

  result.minimumEventHeadroom = std::numeric_limits<std::size_t>::max();
  for (const SyncCoverDomainFeasibility &domain : resources.domains) {
    if (domains_[domain.domain].kind != SyncCoverResourceKind::EventId ||
        domain.required == 0) {
      continue;
    }
    ++result.eventDomainCount;
    result.peakEventPressure =
        std::max(result.peakEventPressure, domain.required);
    if (result.totalEventPressure >
        std::numeric_limits<std::size_t>::max() - domain.required) {
      return makeCostError(SyncCoverStructuralCostError::ArithmeticOverflow);
    }
    result.totalEventPressure += domain.required;
    const std::size_t headroom = domain.required < domain.available
                                     ? domain.available - domain.required
                                     : 0;
    result.minimumEventHeadroom =
        std::min(result.minimumEventHeadroom, headroom);
  }
  if (result.eventDomainCount == 0) {
    result.minimumEventHeadroom = 0;
  }
  return result;
}

SyncCoverSelectionEvaluation SyncCoverSelectionEvaluator::evaluate(
    const std::vector<SyncCoverMechanismId> &selected) const {
  SyncCoverSelectionEvaluation result;
  if (!valid_ || universe_.version_ != version_ ||
      universe_.graph_.getGeneration() != graphGeneration_) {
    result.resources.error = SyncCoverResourceSelectionError::InvalidUniverse;
    result.cost.error = SyncCoverStructuralCostError::InvalidUniverse;
    return result;
  }
  result.resources = universe_.evaluateResourceSelectionImpl(selected, false);
  result.cost =
      universe_.evaluateStructuralCostImpl(selected, result.resources);
  return result;
}
