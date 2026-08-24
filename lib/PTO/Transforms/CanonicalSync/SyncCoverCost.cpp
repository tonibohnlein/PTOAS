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
#include <tuple>
#include <vector>

using namespace mlir::pto;

namespace {

SyncCoverStructuralCost makeCostError(SyncCoverStructuralCostError error) {
  SyncCoverStructuralCost result;
  result.error = error;
  return result;
}

std::size_t getLoopDepth(const SyncCoverGraph &graph, SyncCoverScopeId scope,
                         bool includeScope) {
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  std::size_t depth = 0;
  bool first = true;
  while (scope != 0) {
    if (scopes[scope].isLoop && (includeScope || !first)) {
      ++depth;
    }
    first = false;
    scope = scopes[scope].parent;
  }
  return depth;
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
    return getLoopDepth(graph, graph.getNodes()[action.anchor.node].scope,
                        true);
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit:
    if (action.anchor.scope >= graph.getScopes().size()) {
      return std::nullopt;
    }
    return getLoopDepth(graph, action.anchor.scope, false);
  }
  return std::nullopt;
}

bool incrementProfile(std::vector<std::size_t> &profile,
                      std::size_t maximumDepth, std::size_t depth) {
  if (depth > maximumDepth) {
    return false;
  }
  std::size_t &count = profile[maximumDepth - depth];
  if (count == std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  ++count;
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

} // namespace

bool mlir::pto::syncCoverStructuralCostLess(
    const SyncCoverStructuralCost &first,
    const SyncCoverStructuralCost &second) {
  if (first.error != second.error) {
    return first.error < second.error;
  }
  return std::tie(first.actionProfile, first.barrierActionProfile,
                  first.mechanismCount, first.signature) <
         std::tie(second.actionProfile, second.barrierActionProfile,
                  second.mechanismCount, second.signature);
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
  if (!resources) {
    return makeCostError(mapResourceError(resources.error));
  }
  if (!resources.resourceFeasible) {
    return makeCostError(SyncCoverStructuralCostError::ResourceInfeasible);
  }

  SyncCoverStructuralCost result;
  result.signature = selected;
  std::sort(result.signature.begin(), result.signature.end());
  result.mechanismCount = result.signature.size();
  std::size_t maximumDepth = 0;
  for (const SyncCoverScope &scope : graph_.getScopes()) {
    maximumDepth = std::max(maximumDepth, getLoopDepth(graph_, scope.id, true));
  }
  result.actionProfile.assign(maximumDepth + 1, 0);
  result.barrierActionProfile.assign(maximumDepth + 1, 0);

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
      const SyncCoverNodeId anchor = mechanism.barrier->anchor;
      if (anchor >= graph_.getNodes().size()) {
        return makeCostError(SyncCoverStructuralCostError::InvalidUniverse);
      }
      const std::size_t depth =
          getLoopDepth(graph_, graph_.getNodes()[anchor].scope, true);
      if (!incrementProfile(result.barrierActionProfile, maximumDepth, depth)) {
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
  if (!valid_ || universe_.version_ != version_) {
    result.resources.error = SyncCoverResourceSelectionError::InvalidUniverse;
    result.cost.error = SyncCoverStructuralCostError::InvalidUniverse;
    return result;
  }
  result.resources = universe_.evaluateResourceSelectionImpl(selected, false);
  result.cost =
      universe_.evaluateStructuralCostImpl(selected, result.resources);
  return result;
}
