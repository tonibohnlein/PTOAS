// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageRectangleGrounding.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

bool checkedAdd(std::size_t left, std::size_t right, std::size_t &result) {
  const bool overflow =
      left > std::numeric_limits<std::size_t>::max() - right;
  if (overflow) {
    return false;
  }
  result = left + right;
  return true;
}

bool detailLess(const SyncCoverSyntheticRectangleGroundingDetail &left,
                const SyncCoverSyntheticRectangleGroundingDetail &right) {
  if (left.coverageRows != right.coverageRows) {
    return left.coverageRows > right.coverageRows;
  }
  return left.rectangle < right.rectangle;
}

} // namespace

SyncCoverSyntheticRectangleGrounding
mlir::pto::groundSyncCoverSyntheticStorageRectangles(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    const SyncCoverStorageCutIndex &cutIndex,
    const SyncCoverStorageFactoredRectangleIndex &rectangleIndex,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const SyncCoverSyntheticRectangleGroundingLimits &limits) {
  SyncCoverSyntheticRectangleGrounding result;
  const auto finish = [&](SyncCoverSyntheticRectangleGroundingError error) {
    result.statistics_.workUnits = 0;
    result.error_ = error;
    return std::move(result);
  };
  if (limits.maximumWorkUnits == 0 || limits.maximumDetails == 0) {
    return finish(SyncCoverSyntheticRectangleGroundingError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return finish(SyncCoverSyntheticRectangleGroundingError::InvalidGraph);
  }
  const bool incompleteIndex =
      !cutIndex.isComplete() || !rectangleIndex.isComplete();
  if (incompleteIndex) {
    return finish(
        SyncCoverSyntheticRectangleGroundingError::IncompleteRectangleIndex);
  }

  SyncCoverCoverageWorkBudget budget(limits.maximumWorkUnits);
  const auto stop = [&](SyncCoverSyntheticRectangleGroundingError error,
                        bool truncated) {
    result.statistics_.workUnits = budget.workUnits;
    result.statistics_.truncated = truncated;
    result.error_ = error;
    return std::move(result);
  };
  result.details_.reserve(std::min(limits.maximumDetails,
                                   rectangleIndex.getRectangles().size()));
  const SyncCoverCoverageResult baseline = computeSyncCoverCoverage(
      graph, expansion, {}, activeDemands, &budget);
  if (baseline.error == SyncCoverCoverageError::WorkLimitExceeded) {
    return stop(SyncCoverSyntheticRectangleGroundingError::WorkLimitExceeded,
                true);
  }
  if (!baseline) {
    return stop(SyncCoverSyntheticRectangleGroundingError::CoverageFailure,
                false);
  }
  for (std::size_t rectangleId = 0;
       rectangleId < rectangleIndex.getRectangles().size(); ++rectangleId) {
    if (!budget.consume()) {
      return stop(SyncCoverSyntheticRectangleGroundingError::WorkLimitExceeded,
                  true);
    }
    const SyncCoverStorageFactoredRectangle &rectangle =
        rectangleIndex.getRectangles()[rectangleId];
    const bool invalidRectangle =
        rectangle.id != rectangleId ||
        rectangle.completionCut >= cutIndex.getCuts().size() ||
        rectangle.acquisitionCut >= cutIndex.getCuts().size();
    if (invalidRectangle) {
      return stop(SyncCoverSyntheticRectangleGroundingError::InvalidGraph,
                  false);
    }
    if (rectangle.directRectangle.has_value()) {
      continue;
    }
    const SyncCoverStorageCut &completion =
        cutIndex.getCuts()[rectangle.completionCut];
    const SyncCoverStorageCut &acquisition =
        cutIndex.getCuts()[rectangle.acquisitionCut];
    const std::size_t guardLiterals = rectangle.guard.literals.size();
    const bool validationWorkAvailable =
        budget.consume(guardLiterals) && budget.consume(guardLiterals) &&
        budget.consume(guardLiterals) && budget.consume(guardLiterals) &&
        budget.consume(8);
    if (!validationWorkAvailable) {
      return stop(SyncCoverSyntheticRectangleGroundingError::WorkLimitExceeded,
                  true);
    }
    const bool invalidCuts =
        completion.kind != SyncCoverStorageCutKind::Completion ||
        acquisition.kind != SyncCoverStorageCutKind::Acquisition ||
        completion.anchor.node >= graph.getNodes().size() ||
        acquisition.anchor.node >= graph.getNodes().size() ||
        completion.guard.literals != rectangle.guard.literals ||
        acquisition.guard.literals != rectangle.guard.literals;
    if (invalidCuts) {
      return stop(SyncCoverSyntheticRectangleGroundingError::InvalidGraph,
                  false);
    }

    SyncCoverCompletionSupply supply;
    supply.edge = {completion.anchor.node,
                   acquisition.anchor.node,
                   SyncCoverEdgeKind::CompletionSupply,
                   rectangle.scope,
                   0,
                   rectangle.guard,
                   rectangle.guard};
    // This is the same balanced per-iteration event recipe as a direct event.
    // Its distance-zero supply may therefore contribute transitively inside a
    // recurrence arena; it does not itself span iterations.
    std::vector<SyncCoverCompletionSupply> supplies;
    supplies.reserve(1);
    supplies.push_back(std::move(supply));
    const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
        graph, expansion, supplies, activeDemands, &budget);
    if (coverage.error == SyncCoverCoverageError::WorkLimitExceeded) {
      return stop(SyncCoverSyntheticRectangleGroundingError::WorkLimitExceeded,
                  true);
    }
    if (!coverage) {
      return stop(SyncCoverSyntheticRectangleGroundingError::CoverageFailure,
                  false);
    }
    const std::size_t coverageWords = coverage.covered.getWords().size();
    const bool coverageWorkAvailable =
        budget.consume(coverageWords) && budget.consume(coverageWords) &&
        budget.consume(coverageWords);
    if (!coverageWorkAvailable) {
      return stop(SyncCoverSyntheticRectangleGroundingError::WorkLimitExceeded,
                  true);
    }
    SyncCoverDemandSet incrementalCoverage = coverage.covered;
    incrementalCoverage.subtract(baseline.covered);
    const std::size_t coverageRows = incrementalCoverage.count();
    std::size_t evaluatedSyntheticRectangles = 0;
    std::size_t syntheticRectanglesWithCoverage = 0;
    std::size_t syntheticRectanglesCoveringMultipleRows = 0;
    const bool statisticsValid =
        checkedAdd(result.statistics_.evaluatedSyntheticRectangles, 1,
                   evaluatedSyntheticRectangles) &&
        checkedAdd(result.statistics_.syntheticRectanglesWithCoverage,
                   coverageRows != 0 ? 1 : 0,
                   syntheticRectanglesWithCoverage) &&
        checkedAdd(result.statistics_.syntheticRectanglesCoveringMultipleRows,
                   coverageRows > 1 ? 1 : 0,
                   syntheticRectanglesCoveringMultipleRows);
    if (!statisticsValid) {
      return stop(
          SyncCoverSyntheticRectangleGroundingError::ArithmeticOverflow,
          false);
    }
    result.statistics_.evaluatedSyntheticRectangles =
        evaluatedSyntheticRectangles;
    result.statistics_.syntheticRectanglesWithCoverage =
        syntheticRectanglesWithCoverage;
    result.statistics_.syntheticRectanglesCoveringMultipleRows =
        syntheticRectanglesCoveringMultipleRows;
    result.statistics_.maximumCoverageRows =
        std::max(result.statistics_.maximumCoverageRows, coverageRows);
    std::size_t totalCoverageRows = 0;
    if (!checkedAdd(result.statistics_.totalCoverageRows, coverageRows,
                    totalCoverageRows)) {
      return stop(
          SyncCoverSyntheticRectangleGroundingError::ArithmeticOverflow,
          false);
    }
    result.statistics_.totalCoverageRows = totalCoverageRows;
    if (coverageRows == 0) {
      continue;
    }

    const SyncCoverSyntheticRectangleGroundingDetail detail{
        rectangle.id, rectangle.completionCut, rectangle.acquisitionCut,
        coverageRows};
    const bool detailWorkAvailable =
        budget.consume(result.details_.size() + 1);
    if (!detailWorkAvailable) {
      return stop(SyncCoverSyntheticRectangleGroundingError::WorkLimitExceeded,
                  true);
    }
    const auto position =
        std::lower_bound(result.details_.begin(), result.details_.end(), detail,
                         detailLess);
    const bool hasDetailCapacity =
        result.details_.size() < limits.maximumDetails;
    if (hasDetailCapacity) {
      result.details_.insert(position, detail);
    } else {
      result.statistics_.detailsTruncated = true;
      if (position != result.details_.end()) {
        result.details_.insert(position, detail);
        result.details_.pop_back();
      }
    }
  }
  result.statistics_.workUnits = budget.workUnits;
  return result;
}
