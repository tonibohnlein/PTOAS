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

using namespace mlir::pto;

namespace {

template <typename T>
bool canonicalIdentitySet(const std::vector<T> &values) {
  return std::is_sorted(values.begin(), values.end()) &&
         std::adjacent_find(values.begin(), values.end()) == values.end();
}

} // namespace

SyncCoverMembershipResult mlir::pto::evaluateSyncCoverMembership(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &selected) {
  SyncCoverMembershipResult result;
  const SyncCoverSelectionEvaluator selectionEvaluator(universe);
  if (!selectionEvaluator) {
    result.error = SyncCoverMembershipError::InvalidUniverse;
    return result;
  }
  const bool invalidDemands =
      !canonicalIdentitySet(activeDemands) ||
      (!activeDemands.empty() &&
       activeDemands.back() >= universe.getGraph().getDemands().size());
  if (invalidDemands) {
    result.error = SyncCoverMembershipError::InvalidDemand;
    return result;
  }
  const bool invalidSelection =
      !canonicalIdentitySet(selected) ||
      (!selected.empty() &&
       selected.back() >= universe.getMechanisms().size());
  if (invalidSelection) {
    result.error = SyncCoverMembershipError::InvalidSelection;
    return result;
  }

  const SyncCoverSelectionEvaluation evaluation =
      selectionEvaluator.evaluate(selected);
  result.resources = evaluation.resources;
  result.cost = evaluation.cost;
  if (!result.resources.isValid() || !result.resources.resourceFeasible) {
    return result;
  }

  SyncCoverCoverageOracle oracle(universe.getGraph());
  for (SyncCoverDemandId demand : activeDemands) {
    const SyncCoverCoverageResult coverage =
        oracle.checkDemandCanonicalSelection(demand, selected);
    if (!coverage) {
      result.error = SyncCoverMembershipError::CoverageFailure;
      return result;
    }
    if (!coverage.covered) {
      result.uncoveredDemands.push_back({demand, coverage.cutMechanisms});
    }
  }
  result.coverageComplete = result.uncoveredDemands.empty();
  result.coverageStatistics = oracle.getStatistics();
  return result;
}
