// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageCuts.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

using CutKey = std::pair<SyncCoverNodeId, std::uint32_t>;
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

bool consumeEligibilityWork(WorkBudget &budget, const SyncCoverGraph &graph,
                            const SyncCoverDemand &demand,
                            const SyncCoverNode &source) {
  if (!budget.consume(source.completionTargets.size())) {
    return false;
  }
  for (unsigned walk = 0; walk < 10; ++walk) {
    if (!budget.consume(graph.getScopes().size())) {
      return false;
    }
  }
  for (unsigned guardOperation = 0; guardOperation < 8; ++guardOperation) {
    const bool guardWorkAvailable =
        budget.consume(demand.sourceGuard.literals.size()) &&
        budget.consume(demand.targetGuard.literals.size());
    if (!guardWorkAvailable) {
      return false;
    }
  }
  return budget.consume(16);
}

bool frozenDemandEndpointsCoExecute(const SyncCoverGraph &graph,
                                    const SyncCoverDemand &demand) {
  // Frozen demands already contain normalized, scope-complete endpoint
  // guards. Mutual implication is therefore exact vector equality and can be
  // checked without the copies and sorts used by the general query helper.
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const std::optional<SyncCoverScopeId> common =
      graph.getLowestCommonScope(source.scope, target.scope);
  return common && graph.scopeMustExecuteWithin(*common, source.scope) &&
         graph.scopeMustExecuteWithin(*common, target.scope) &&
         demand.sourceGuard.literals == demand.targetGuard.literals;
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

} // namespace

SyncCoverStorageCutIndex mlir::pto::buildSyncCoverStorageCutIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageCutLimits &limits) {
  SyncCoverStorageCutIndex result;
  const auto fail = [&](SyncCoverStorageCutStatistics statistics,
                        SyncCoverStorageCutError error) {
    result.cuts_.clear();
    result.rectangles_.clear();
    statistics.completionCuts = 0;
    statistics.acquisitionCuts = 0;
    statistics.rectangles = 0;
    statistics.incidences = 0;
    statistics.guardLiterals = 0;
    statistics.maximumRectangleEdges = 0;
    statistics.truncated = error == SyncCoverStorageCutError::LimitExceeded;
    result.statistics_ = statistics;
    result.error_ = error;
    return std::move(result);
  };
  const bool invalidLimit =
      limits.maximumWorkUnits == 0 || limits.maximumCuts == 0 ||
      limits.maximumRectangles == 0 || limits.maximumIncidences == 0 ||
      limits.maximumGuardLiterals == 0;
  if (invalidLimit) {
    return fail({}, SyncCoverStorageCutError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return fail({}, SyncCoverStorageCutError::InvalidGraph);
  }
  if (!lifecycleIndex.isComplete()) {
    return fail({}, SyncCoverStorageCutError::IncompleteLifecycleIndex);
  }

  SyncCoverStorageCutStatistics statistics;
  WorkBudget budget(limits.maximumWorkUnits, statistics.workUnits);
  std::map<CutKey, SyncCoverStorageCutId> completionCuts;
  std::map<CutKey, SyncCoverStorageCutId> acquisitionCuts;
  std::map<RectangleKey, SyncCoverStorageRectangleId> rectangles;
  std::size_t totalIncidences = 0;
  std::size_t totalGuardLiterals = 0;

  const auto addCut = [&](std::map<CutKey, SyncCoverStorageCutId> &index,
                          SyncCoverStorageCutKind kind, SyncCoverNodeId node,
                          std::uint32_t peerResource)
      -> std::optional<SyncCoverStorageCutId> {
    if (!consumeOrderedOperation(budget, index.size())) {
      return std::nullopt;
    }
    const CutKey key{node, peerResource};
    auto position = index.find(key);
    if (position != index.end()) {
      return position->second;
    }
    const bool invalidCut = result.cuts_.size() >= limits.maximumCuts ||
                            node >= graph.getNodes().size();
    if (invalidCut) {
      return std::nullopt;
    }
    const SyncCoverNode &description = graph.getNodes()[node];
    const std::size_t guardLiterals = description.guard.literals.size();
    const bool guardLimitReached =
        totalGuardLiterals > limits.maximumGuardLiterals ||
        guardLiterals > limits.maximumGuardLiterals - totalGuardLiterals;
    const bool cutWorkAvailable =
        budget.consume(guardLiterals) &&
        consumeOrderedOperation(budget, index.size());
    if (guardLimitReached || !cutWorkAvailable) {
      return std::nullopt;
    }
    const SyncCoverStorageCutId cut = result.cuts_.size();
    const SyncCoverAnchorKind anchorKind =
        kind == SyncCoverStorageCutKind::Completion
            ? SyncCoverAnchorKind::AfterNode
            : SyncCoverAnchorKind::BeforeNode;
    result.cuts_.push_back({cut,
                            kind,
                            {anchorKind, node, 0, 0},
                            description.resource,
                            peerResource,
                            description.scope,
                            description.guard,
                            {}});
    index.emplace(key, cut);
    totalGuardLiterals += guardLiterals;
    return cut;
  };

  const auto addIncidence = [&](std::vector<SyncCoverStorageLifecycleEdgeRef>
                                    &incidences,
                                SyncCoverStorageLifecycleEdgeRef edge) {
    const bool incidenceAvailable =
        checkedIncrement(totalIncidences, limits.maximumIncidences) &&
        budget.consume();
    if (!incidenceAvailable) {
      return false;
    }
    incidences.push_back(edge);
    return true;
  };

  for (const SyncCoverStorageLifecycleComponent &component :
       lifecycleIndex.getComponents()) {
    if (!budget.consume()) {
      return fail(statistics, SyncCoverStorageCutError::LimitExceeded);
    }
    for (const SyncCoverStorageLifecycleEdge &edge : component.edges) {
      if (!budget.consume()) {
        return fail(statistics, SyncCoverStorageCutError::LimitExceeded);
      }
      const bool invalidEdge =
          edge.id >= component.edges.size() ||
          edge.source >= component.epochs.size() ||
          edge.target >= component.epochs.size() ||
          edge.demand >= graph.getDemands().size();
      if (invalidEdge) {
        return fail(statistics, SyncCoverStorageCutError::InvalidGraph);
      }
      const SyncCoverStorageLifecycleEpoch &sourceEpoch =
          component.epochs[edge.source];
      const SyncCoverStorageLifecycleEpoch &targetEpoch =
          component.epochs[edge.target];
      const SyncCoverDemand &demand = graph.getDemands()[edge.demand];
      const bool invalidProvenance = sourceEpoch.node != demand.source ||
                                     targetEpoch.node != demand.target;
      if (invalidProvenance) {
        return fail(statistics, SyncCoverStorageCutError::InvalidGraph);
      }
      const bool structurallyEligible =
          edge.distance == 0 && sourceEpoch.resource != targetEpoch.resource;
      if (structurallyEligible &&
          !consumeEligibilityWork(budget, graph, demand,
                                  graph.getNodes()[sourceEpoch.node])) {
        return fail(statistics, SyncCoverStorageCutError::LimitExceeded);
      }
      const bool eligible =
          structurallyEligible &&
          syncCoverNodeCanProduceCompletion(graph, sourceEpoch.node,
                                            targetEpoch.resource) &&
          frozenDemandEndpointsCoExecute(graph, demand);
      const SyncCoverAnchor completionAnchor{SyncCoverAnchorKind::AfterNode,
                                             sourceEpoch.node, 0, 0};
      const SyncCoverAnchor acquisitionAnchor{SyncCoverAnchorKind::BeforeNode,
                                              targetEpoch.node, 0, 0};
      const std::optional<SyncCoverTimelinePosition> completionPosition =
          eligible ? resolveSyncCoverAnchor(graph, completionAnchor)
                   : std::nullopt;
      const std::optional<SyncCoverTimelinePosition> acquisitionPosition =
          eligible ? resolveSyncCoverAnchor(graph, acquisitionAnchor)
                   : std::nullopt;
      const bool ordered = completionPosition && acquisitionPosition &&
                           *completionPosition < *acquisitionPosition;
      if (!eligible || !ordered) {
        if (!checkedIncrement(statistics.ineligibleEdges,
                              std::numeric_limits<std::size_t>::max())) {
          return fail(statistics, SyncCoverStorageCutError::ArithmeticOverflow);
        }
        continue;
      }
      if (!checkedIncrement(statistics.eligibleEdges,
                            std::numeric_limits<std::size_t>::max())) {
        return fail(statistics, SyncCoverStorageCutError::ArithmeticOverflow);
      }
      const std::optional<SyncCoverStorageCutId> completion =
          addCut(completionCuts, SyncCoverStorageCutKind::Completion,
                 sourceEpoch.node, targetEpoch.resource);
      const std::optional<SyncCoverStorageCutId> acquisition =
          addCut(acquisitionCuts, SyncCoverStorageCutKind::Acquisition,
                 targetEpoch.node, sourceEpoch.resource);
      if (!completion || !acquisition) {
        return fail(statistics, SyncCoverStorageCutError::LimitExceeded);
      }
      const SyncCoverStorageLifecycleEdgeRef reference{component.id, edge.id};
      const bool cutIncidencesAvailable =
          addIncidence(result.cuts_[*completion].edges, reference) &&
          addIncidence(result.cuts_[*acquisition].edges, reference);
      if (!cutIncidencesAvailable) {
        return fail(statistics, SyncCoverStorageCutError::LimitExceeded);
      }
      if (!consumeOrderedOperation(budget, rectangles.size())) {
        return fail(statistics, SyncCoverStorageCutError::LimitExceeded);
      }
      const RectangleKey rectangleKey{*completion, *acquisition};
      auto rectanglePosition = rectangles.find(rectangleKey);
      if (rectanglePosition == rectangles.end()) {
        const bool rectangleLimitReached =
            result.rectangles_.size() >= limits.maximumRectangles;
        const bool rectangleWorkAvailable =
            consumeOrderedOperation(budget, rectangles.size());
        if (rectangleLimitReached || !rectangleWorkAvailable) {
          return fail(statistics, SyncCoverStorageCutError::LimitExceeded);
        }
        const SyncCoverStorageRectangleId rectangle =
            result.rectangles_.size();
        result.rectangles_.push_back(
            {rectangle, *completion, *acquisition, 0, {}});
        rectanglePosition = rectangles.emplace(rectangleKey, rectangle).first;
      }
      SyncCoverStorageRectangle &rectangle =
          result.rectangles_[rectanglePosition->second];
      rectangle.kinds |= edge.kinds;
      if (!addIncidence(rectangle.edges, reference)) {
        return fail(statistics, SyncCoverStorageCutError::LimitExceeded);
      }
      statistics.maximumRectangleEdges =
          std::max(statistics.maximumRectangleEdges, rectangle.edges.size());
    }
  }

  statistics.completionCuts = completionCuts.size();
  statistics.acquisitionCuts = acquisitionCuts.size();
  statistics.rectangles = result.rectangles_.size();
  statistics.incidences = totalIncidences;
  statistics.guardLiterals = totalGuardLiterals;
  result.statistics_ = statistics;
  return result;
}
