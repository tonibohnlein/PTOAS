// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolCuts.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

using EdgeRef = SyncCoverStorageLifecycleEdgeRef;

struct RectangleEdgeEntry {
  EdgeRef edge;
  SyncCoverStorageRectangleId rectangle = 0;
};

bool edgeRefLess(const EdgeRef &left, const EdgeRef &right) {
  return std::tie(left.component, left.edge) <
         std::tie(right.component, right.edge);
}

bool entryLess(const RectangleEdgeEntry &left,
               const RectangleEdgeEntry &right) {
  return edgeRefLess(left.edge, right.edge);
}

class WorkBudget {
public:
  WorkBudget(std::size_t maximum, std::size_t &used)
      : maximum_(maximum), used_(used) {}

  bool consume(std::size_t amount = 1) {
    if (used_ > maximum_ || amount > maximum_ - used_) {
      exhausted_ = true;
      return false;
    }
    used_ += amount;
    return true;
  }

  bool exhausted() const { return exhausted_; }

private:
  std::size_t maximum_ = 0;
  std::size_t &used_;
  bool exhausted_ = false;
};

bool consumeSortWork(WorkBudget &budget, std::size_t elementCount) {
  if (elementCount < 2) {
    return true;
  }
  std::size_t levels = 0;
  for (std::size_t covered = 1; covered < elementCount;) {
    ++levels;
    const bool coveredWouldOverflow =
        covered > std::numeric_limits<std::size_t>::max() / 2;
    if (coveredWouldOverflow) {
      return false;
    }
    covered *= 2;
  }
  constexpr std::size_t operationsPerLevel = 4;
  const bool levelWorkWouldOverflow =
      levels > std::numeric_limits<std::size_t>::max() / operationsPerLevel;
  if (levelWorkWouldOverflow) {
    return false;
  }
  const std::size_t workPerElement = levels * operationsPerLevel;
  return elementCount <=
             std::numeric_limits<std::size_t>::max() / workPerElement &&
         budget.consume(elementCount * workPerElement);
}

std::optional<SyncCoverStorageRectangleId>
findRectangle(const std::vector<RectangleEdgeEntry> &entries,
              const EdgeRef &edge, WorkBudget &budget) {
  std::size_t begin = 0;
  std::size_t end = entries.size();
  while (begin < end) {
    if (!budget.consume()) {
      return std::nullopt;
    }
    const std::size_t middle = begin + (end - begin) / 2;
    if (edgeRefLess(entries[middle].edge, edge)) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  const bool missing =
      begin == entries.size() || edgeRefLess(edge, entries[begin].edge);
  if (missing) {
    return std::nullopt;
  }
  return entries[begin].rectangle;
}

constexpr SyncCoverStorageLifecycleEdgeKindMask readyBit() {
  return syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Ready);
}

constexpr SyncCoverStorageLifecycleEdgeKindMask releaseBit() {
  return syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Release);
}

} // namespace

SyncCoverStorageProtocolCutPlanIndex
mlir::pto::buildSyncCoverStorageProtocolCutPlanIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolAutomatonIndex &automatonIndex,
    const SyncCoverStorageCutIndex &cutIndex,
    const SyncCoverStorageProtocolCutPlanLimits &limits) {
  SyncCoverStorageProtocolCutPlanIndex result;
  const auto fail = [&](SyncCoverStorageProtocolCutPlanStatistics statistics,
                        SyncCoverStorageProtocolCutPlanError error) {
    result.plans_.clear();
    statistics.plans = 0;
    statistics.directReadyTransfers = 0;
    statistics.recurrenceReleaseTransfers = 0;
    statistics.readyRectangleIncidences = 0;
    statistics.maximumPlanReadyRectangles = 0;
    statistics.truncated =
        error == SyncCoverStorageProtocolCutPlanError::LimitExceeded;
    result.statistics_ = statistics;
    result.error_ = error;
    return std::move(result);
  };
  const bool invalidLimit = limits.maximumWorkUnits == 0 ||
                            limits.maximumPlans == 0 ||
                            limits.maximumRectangleEdgeIncidences == 0 ||
                            limits.maximumTransferInspections == 0 ||
                            limits.maximumReadyRectangleIncidences == 0;
  if (invalidLimit) {
    return fail({}, SyncCoverStorageProtocolCutPlanError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return fail({}, SyncCoverStorageProtocolCutPlanError::InvalidGraph);
  }
  if (!lifecycleIndex.isComplete()) {
    return fail({},
                SyncCoverStorageProtocolCutPlanError::IncompleteLifecycleIndex);
  }
  if (!automatonIndex.isComplete()) {
    return fail({},
                SyncCoverStorageProtocolCutPlanError::IncompleteAutomatonIndex);
  }
  if (!cutIndex.isComplete()) {
    return fail({}, SyncCoverStorageProtocolCutPlanError::IncompleteCutIndex);
  }

  const std::vector<SyncCoverStorageLifecycleComponent> &components =
      lifecycleIndex.getComponents();
  const std::vector<SyncCoverStorageRectangle> &rectangles =
      cutIndex.getRectangles();
  const std::vector<SyncCoverStorageProtocolAutomaton> &automata =
      automatonIndex.getAutomata();
  SyncCoverStorageProtocolCutPlanStatistics statistics;
  WorkBudget budget(limits.maximumWorkUnits, statistics.workUnits);

  std::vector<RectangleEdgeEntry> rectangleEdges;
  for (std::size_t rectanglePosition = 0; rectanglePosition < rectangles.size();
       ++rectanglePosition) {
    const SyncCoverStorageRectangle &rectangle = rectangles[rectanglePosition];
    const bool invalidRectangle =
        rectangle.id != rectanglePosition || rectangle.edges.empty();
    if (invalidRectangle) {
      return fail(statistics,
                  SyncCoverStorageProtocolCutPlanError::InvalidGraph);
    }
    const bool incidenceLimitReached =
        rectangleEdges.size() > limits.maximumRectangleEdgeIncidences ||
        rectangle.edges.size() >
            limits.maximumRectangleEdgeIncidences - rectangleEdges.size();
    if (incidenceLimitReached || !budget.consume(rectangle.edges.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolCutPlanError::LimitExceeded);
    }
    for (const EdgeRef &edge : rectangle.edges) {
      const bool invalidEdge =
          edge.component >= components.size() ||
          edge.edge >= components[edge.component].edges.size();
      if (invalidEdge) {
        return fail(statistics,
                    SyncCoverStorageProtocolCutPlanError::InvalidGraph);
      }
      rectangleEdges.push_back({edge, rectangle.id});
    }
  }
  if (!consumeSortWork(budget, rectangleEdges.size())) {
    return fail(statistics,
                SyncCoverStorageProtocolCutPlanError::LimitExceeded);
  }
  std::sort(rectangleEdges.begin(), rectangleEdges.end(), entryLess);
  for (std::size_t edge = 1; edge < rectangleEdges.size(); ++edge) {
    if (!budget.consume()) {
      return fail(statistics,
                  SyncCoverStorageProtocolCutPlanError::LimitExceeded);
    }
    const bool duplicate =
        !edgeRefLess(rectangleEdges[edge - 1].edge, rectangleEdges[edge].edge);
    if (duplicate) {
      return fail(statistics,
                  SyncCoverStorageProtocolCutPlanError::InvalidGraph);
    }
  }

  result.plans_.reserve(std::min(automata.size(), limits.maximumPlans));
  for (std::size_t automatonPosition = 0; automatonPosition < automata.size();
       ++automatonPosition) {
    const SyncCoverStorageProtocolAutomaton &automaton =
        automata[automatonPosition];
    const bool invalidAutomaton = automaton.id != automatonPosition;
    if (invalidAutomaton) {
      return fail(statistics,
                  SyncCoverStorageProtocolCutPlanError::InvalidGraph);
    }
    SyncCoverStorageProtocolCutPlan plan;
    plan.id = result.plans_.size();
    plan.automaton = automaton.id;
    plan.group = automaton.group;
    plan.owningScope = automaton.owningScope;
    plan.laneCount = std::max<std::size_t>(automaton.maximumDistance, 1);
    std::size_t missingReadyCuts = 0;
    for (std::size_t transferPosition = 0;
         transferPosition < automaton.transfers.size(); ++transferPosition) {
      const SyncCoverStorageProtocolTransfer &transfer =
          automaton.transfers[transferPosition];
      if (statistics.transferInspections >= limits.maximumTransferInspections ||
          !budget.consume()) {
        return fail(statistics,
                    SyncCoverStorageProtocolCutPlanError::LimitExceeded);
      }
      ++statistics.transferInspections;
      const bool invalidTransferRef =
          transfer.id != transferPosition ||
          transfer.edge.component >= components.size() ||
          transfer.edge.edge >=
              components[transfer.edge.component].edges.size();
      if (invalidTransferRef) {
        return fail(statistics,
                    SyncCoverStorageProtocolCutPlanError::InvalidGraph);
      }
      const SyncCoverStorageLifecycleComponent &component =
          components[transfer.edge.component];
      const SyncCoverStorageLifecycleEdge &edge =
          component.edges[transfer.edge.edge];
      const bool invalidLifecycleRef = edge.id != transfer.edge.edge ||
                                       edge.source >= component.epochs.size() ||
                                       edge.target >= component.epochs.size();
      if (invalidLifecycleRef) {
        return fail(statistics,
                    SyncCoverStorageProtocolCutPlanError::InvalidGraph);
      }
      const SyncCoverStorageLifecycleEpoch &source =
          component.epochs[edge.source];
      const SyncCoverStorageLifecycleEpoch &target =
          component.epochs[edge.target];
      const bool inconsistentTransfer =
          transfer.demand != edge.demand || transfer.kinds != edge.kinds ||
          transfer.scope != edge.scope || transfer.distance != edge.distance ||
          transfer.sourceResource != source.resource ||
          transfer.targetResource != target.resource;
      if (inconsistentTransfer) {
        return fail(statistics,
                    SyncCoverStorageProtocolCutPlanError::InvalidGraph);
      }
      const bool directReady =
          transfer.distance == 0 &&
          transfer.sourceResource != transfer.targetResource &&
          (transfer.kinds & readyBit()) != 0;
      if (directReady) {
        ++plan.directReadyTransfers;
        const std::optional<SyncCoverStorageRectangleId> rectangle =
            findRectangle(rectangleEdges, transfer.edge, budget);
        if (budget.exhausted()) {
          return fail(statistics,
                      SyncCoverStorageProtocolCutPlanError::LimitExceeded);
        }
        if (!rectangle) {
          ++missingReadyCuts;
        } else {
          plan.readyRectangles.push_back(*rectangle);
        }
      }
      const bool release =
          transfer.distance != 0 && (transfer.kinds & releaseBit()) != 0;
      if (release) {
        ++plan.recurrenceReleaseTransfers;
      }
    }
    if (!consumeSortWork(budget, plan.readyRectangles.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolCutPlanError::LimitExceeded);
    }
    std::sort(plan.readyRectangles.begin(), plan.readyRectangles.end());
    plan.readyRectangles.erase(
        std::unique(plan.readyRectangles.begin(), plan.readyRectangles.end()),
        plan.readyRectangles.end());
    const bool missingReady =
        plan.directReadyTransfers == 0 || missingReadyCuts != 0;
    const bool missingRelease = plan.recurrenceReleaseTransfers == 0;
    if (missingReady || missingRelease) {
      ++statistics.ineligibleAutomata;
      statistics.missingReadyCutAutomata += missingReady;
      statistics.missingReleaseAutomata += missingRelease;
      continue;
    }
    const bool planLimitReached = result.plans_.size() >= limits.maximumPlans;
    const bool incidenceLimitReached =
        statistics.readyRectangleIncidences >
            limits.maximumReadyRectangleIncidences ||
        plan.readyRectangles.size() > limits.maximumReadyRectangleIncidences -
                                          statistics.readyRectangleIncidences;
    const bool publicationWorkAvailable =
        budget.consume(plan.readyRectangles.size());
    if (planLimitReached || incidenceLimitReached ||
        !publicationWorkAvailable) {
      return fail(statistics,
                  SyncCoverStorageProtocolCutPlanError::LimitExceeded);
    }
    statistics.directReadyTransfers += plan.directReadyTransfers;
    statistics.recurrenceReleaseTransfers += plan.recurrenceReleaseTransfers;
    statistics.readyRectangleIncidences += plan.readyRectangles.size();
    statistics.maximumPlanReadyRectangles = std::max(
        statistics.maximumPlanReadyRectangles, plan.readyRectangles.size());
    result.plans_.push_back(std::move(plan));
    ++statistics.eligibleAutomata;
  }
  statistics.plans = result.plans_.size();
  result.statistics_ = statistics;
  return result;
}
