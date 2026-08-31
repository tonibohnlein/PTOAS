// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageRectangles.h"

#include <limits>
#include <map>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

using RectangleKey = std::pair<SyncCoverStorageCutId, SyncCoverStorageCutId>;

class WorkBudget {
public:
  WorkBudget(std::size_t limit, std::size_t &used)
      : limit_(limit), used_(used) {}

  bool consume(std::size_t amount = 1) {
    if (failed_ || used_ > limit_ || amount > limit_ - used_) {
      failed_ = true;
      return false;
    }
    used_ += amount;
    return true;
  }

private:
  std::size_t limit_ = 0;
  std::size_t &used_;
  bool failed_ = false;
};

bool consumeOrderedOperation(WorkBudget &budget, std::size_t elementCount) {
  return budget.consume(elementCount) && budget.consume();
}

bool consumeStructuralWork(WorkBudget &budget, const SyncCoverGraph &graph) {
  for (unsigned walk = 0; walk < 10; ++walk) {
    if (!budget.consume(graph.getScopes().size())) {
      return false;
    }
  }
  return budget.consume(16);
}

bool checkedIncrement(std::size_t &value, std::size_t limit) {
  const bool exhausted =
      value == std::numeric_limits<std::size_t>::max() || value >= limit;
  if (exhausted) {
    return false;
  }
  ++value;
  return true;
}

bool validateCut(const SyncCoverGraph &graph, const SyncCoverStorageCut &cut,
                 SyncCoverStorageCutId expectedId) {
  const bool invalidNode = cut.anchor.node >= graph.getNodes().size();
  if (cut.id != expectedId || invalidNode || cut.resource == cut.peerResource) {
    return false;
  }
  const SyncCoverNode &node = graph.getNodes()[cut.anchor.node];
  const SyncCoverAnchorKind expectedAnchor =
      cut.kind == SyncCoverStorageCutKind::Completion
          ? SyncCoverAnchorKind::AfterNode
          : SyncCoverAnchorKind::BeforeNode;
  return cut.anchor.kind == expectedAnchor && cut.anchor.scope == 0 &&
         cut.anchor.position == 0 && cut.resource == node.resource &&
         cut.scope == node.scope && cut.guard.literals == node.guard.literals;
}

} // namespace

SyncCoverStorageFactoredRectangleIndex
mlir::pto::buildSyncCoverStorageFactoredRectangleIndex(
    const SyncCoverGraph &graph, const SyncCoverStorageCutIndex &cutIndex,
    const SyncCoverStorageFactoredRectangleLimits &limits) {
  SyncCoverStorageFactoredRectangleIndex result;
  const auto fail = [&](SyncCoverStorageFactoredRectangleStatistics statistics,
                        SyncCoverStorageFactoredRectangleError error) {
    result.rectangles_.clear();
    statistics.rectangles = 0;
    statistics.directRectangles = 0;
    statistics.syntheticRectangles = 0;
    statistics.guardLiterals = 0;
    statistics.truncated =
        error == SyncCoverStorageFactoredRectangleError::LimitExceeded;
    result.statistics_ = statistics;
    result.error_ = error;
    return std::move(result);
  };
  const bool invalidLimit =
      limits.maximumWorkUnits == 0 || limits.maximumInspections == 0 ||
      limits.maximumRectangles == 0 || limits.maximumGuardLiterals == 0;
  if (invalidLimit) {
    return fail({}, SyncCoverStorageFactoredRectangleError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return fail({}, SyncCoverStorageFactoredRectangleError::InvalidGraph);
  }
  if (!cutIndex.isComplete()) {
    return fail({},
                SyncCoverStorageFactoredRectangleError::IncompleteCutIndex);
  }

  SyncCoverStorageFactoredRectangleStatistics statistics;
  WorkBudget budget(limits.maximumWorkUnits, statistics.workUnits);
  std::map<RectangleKey, SyncCoverStorageRectangleId> directRectangles;
  for (std::size_t rectangleId = 0;
       rectangleId < cutIndex.getRectangles().size(); ++rectangleId) {
    const SyncCoverStorageRectangle &rectangle =
        cutIndex.getRectangles()[rectangleId];
    const bool invalidRectangle =
        rectangle.id != rectangleId ||
        rectangle.completionCut >= cutIndex.getCuts().size() ||
        rectangle.acquisitionCut >= cutIndex.getCuts().size() ||
        cutIndex.getCuts()[rectangle.completionCut].kind !=
            SyncCoverStorageCutKind::Completion ||
        cutIndex.getCuts()[rectangle.acquisitionCut].kind !=
            SyncCoverStorageCutKind::Acquisition;
    if (invalidRectangle) {
      return fail(statistics,
                  SyncCoverStorageFactoredRectangleError::InvalidGraph);
    }
    if (!consumeOrderedOperation(budget, directRectangles.size())) {
      return fail(statistics,
                  SyncCoverStorageFactoredRectangleError::LimitExceeded);
    }
    const auto [position, inserted] = directRectangles.emplace(
        RectangleKey{rectangle.completionCut, rectangle.acquisitionCut},
        rectangle.id);
    if (!inserted) {
      (void)position;
      return fail(statistics,
                  SyncCoverStorageFactoredRectangleError::InvalidGraph);
    }
  }

  for (std::size_t cutId = 0; cutId < cutIndex.getCuts().size(); ++cutId) {
    const SyncCoverStorageCut &cut = cutIndex.getCuts()[cutId];
    const bool validationWorkAvailable =
        budget.consume() && budget.consume(cut.guard.literals.size());
    if (!validationWorkAvailable) {
      return fail(statistics,
                  SyncCoverStorageFactoredRectangleError::LimitExceeded);
    }
    if (!validateCut(graph, cut, cutId)) {
      return fail(statistics,
                  SyncCoverStorageFactoredRectangleError::InvalidGraph);
    }
  }

  std::size_t guardLiterals = 0;
  for (SyncCoverStorageCutId completionId = 0;
       completionId < cutIndex.getCuts().size(); ++completionId) {
    const SyncCoverStorageCut &completion = cutIndex.getCuts()[completionId];
    if (completion.kind != SyncCoverStorageCutKind::Completion) {
      continue;
    }
    for (SyncCoverStorageCutId acquisitionId = 0;
         acquisitionId < cutIndex.getCuts().size(); ++acquisitionId) {
      if (!checkedIncrement(statistics.inspections,
                            limits.maximumInspections) ||
          !budget.consume()) {
        return fail(statistics,
                    SyncCoverStorageFactoredRectangleError::LimitExceeded);
      }
      const SyncCoverStorageCut &acquisition =
          cutIndex.getCuts()[acquisitionId];
      if (acquisition.kind != SyncCoverStorageCutKind::Acquisition) {
        continue;
      }
      const bool matchingDomain =
          completion.resource == acquisition.peerResource &&
          completion.peerResource == acquisition.resource;
      if (!matchingDomain) {
        continue;
      }
      const std::size_t completionGuardLiterals =
          completion.guard.literals.size();
      const std::size_t acquisitionGuardLiterals =
          acquisition.guard.literals.size();
      const bool guardWorkAvailable =
          budget.consume(completionGuardLiterals) &&
          budget.consume(acquisitionGuardLiterals);
      if (!guardWorkAvailable) {
        return fail(statistics,
                    SyncCoverStorageFactoredRectangleError::LimitExceeded);
      }
      if (completion.guard.literals != acquisition.guard.literals) {
        continue;
      }
      if (!consumeStructuralWork(budget, graph)) {
        return fail(statistics,
                    SyncCoverStorageFactoredRectangleError::LimitExceeded);
      }
      const SyncCoverNode &source =
          graph.getNodes()[completion.anchor.node];
      const SyncCoverNode &target =
          graph.getNodes()[acquisition.anchor.node];
      const std::optional<SyncCoverScopeId> scope =
          graph.getLowestCommonScope(completion.scope, acquisition.scope);
      const std::optional<SyncCoverTimelinePosition> setPosition =
          resolveSyncCoverAnchor(graph, completion.anchor);
      const std::optional<SyncCoverTimelinePosition> waitPosition =
          resolveSyncCoverAnchor(graph, acquisition.anchor);
      const bool balanced =
          scope && graph.scopeMustExecuteWithin(*scope, completion.scope) &&
          graph.scopeMustExecuteWithin(*scope, acquisition.scope);
      if (!budget.consume(source.completionTargets.size())) {
        return fail(statistics,
                    SyncCoverStorageFactoredRectangleError::LimitExceeded);
      }
      const bool legal =
          balanced && setPosition && waitPosition &&
          *setPosition < *waitPosition &&
          source.physicalAnchor != target.physicalAnchor &&
          syncCoverNodeCanProduceCompletion(graph, source.id,
                                            acquisition.resource);
      if (!legal) {
        continue;
      }
      const bool rectangleLimitReached =
          result.rectangles_.size() >= limits.maximumRectangles;
      const bool guardLimitReached =
          guardLiterals > limits.maximumGuardLiterals ||
          completionGuardLiterals >
              limits.maximumGuardLiterals - guardLiterals;
      if (rectangleLimitReached || guardLimitReached ||
          !budget.consume(completionGuardLiterals) || !budget.consume()) {
        return fail(statistics,
                    SyncCoverStorageFactoredRectangleError::LimitExceeded);
      }
      const RectangleKey key{completionId, acquisitionId};
      if (!consumeOrderedOperation(budget, directRectangles.size())) {
        return fail(statistics,
                    SyncCoverStorageFactoredRectangleError::LimitExceeded);
      }
      const auto direct = directRectangles.find(key);
      const std::optional<SyncCoverStorageRectangleId> directRectangle =
          direct == directRectangles.end()
              ? std::nullopt
              : std::optional<SyncCoverStorageRectangleId>(direct->second);
      const SyncCoverStorageFactoredRectangleId rectangle =
          result.rectangles_.size();
      result.rectangles_.push_back({rectangle, completionId, acquisitionId,
                                    *scope, completion.guard,
                                    directRectangle});
      guardLiterals += completionGuardLiterals;
      if (directRectangle) {
        if (!checkedIncrement(statistics.directRectangles,
                              std::numeric_limits<std::size_t>::max())) {
          return fail(
              statistics,
              SyncCoverStorageFactoredRectangleError::ArithmeticOverflow);
        }
      } else if (!checkedIncrement(
                     statistics.syntheticRectangles,
                     std::numeric_limits<std::size_t>::max())) {
        return fail(statistics,
                    SyncCoverStorageFactoredRectangleError::ArithmeticOverflow);
      }
    }
  }

  statistics.rectangles = result.rectangles_.size();
  statistics.guardLiterals = guardLiterals;
  result.statistics_ = statistics;
  return result;
}
