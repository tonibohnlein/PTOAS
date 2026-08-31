// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolRectangles.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

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

bool checkedAdd(std::size_t left, std::size_t right, std::size_t &result) {
  const bool overflow = left > std::numeric_limits<std::size_t>::max() - right;
  if (overflow) {
    return false;
  }
  result = left + right;
  return true;
}

bool checkedIncrement(std::size_t &value) {
  if (value == std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  ++value;
  return true;
}

bool consumeSortWork(WorkBudget &budget, std::size_t count) {
  if (count < 2) {
    return true;
  }
  std::size_t levels = 0;
  for (std::size_t covered = 1; covered < count;) {
    ++levels;
    const bool coveredWouldOverflow =
        covered > std::numeric_limits<std::size_t>::max() / 2;
    if (coveredWouldOverflow) {
      return false;
    }
    covered *= 2;
  }
  constexpr std::size_t kOperationsPerLevel = 4;
  const bool levelOverflow =
      levels > std::numeric_limits<std::size_t>::max() / kOperationsPerLevel;
  const std::size_t workPerElement =
      levelOverflow ? 0 : levels * kOperationsPerLevel;
  const bool overflow =
      levelOverflow ||
      count > std::numeric_limits<std::size_t>::max() / workPerElement;
  if (overflow) {
    return false;
  }
  return budget.consume(count * workPerElement);
}

bool consumeSortWork(SyncCoverCoverageWorkBudget &budget, std::size_t count) {
  if (count < 2) {
    return true;
  }
  std::size_t levels = 0;
  for (std::size_t covered = 1; covered < count;) {
    ++levels;
    const bool coveredWouldOverflow =
        covered > std::numeric_limits<std::size_t>::max() / 2;
    if (coveredWouldOverflow) {
      budget.exhausted = true;
      return false;
    }
    covered *= 2;
  }
  constexpr std::size_t kOperationsPerLevel = 4;
  const bool levelOverflow =
      levels > std::numeric_limits<std::size_t>::max() / kOperationsPerLevel;
  const std::size_t workPerElement =
      levelOverflow ? 0 : levels * kOperationsPerLevel;
  const bool overflow =
      levelOverflow ||
      count > std::numeric_limits<std::size_t>::max() / workPerElement;
  if (overflow) {
    budget.exhausted = true;
    return false;
  }
  return budget.consume(count * workPerElement);
}

auto anchorKey(const SyncCoverAnchor &anchor) {
  return std::tie(anchor.kind, anchor.node, anchor.scope, anchor.position);
}

bool frontierLess(const SyncCoverStorageProtocolFrontier &left,
                  const SyncCoverStorageProtocolFrontier &right) {
  return std::tie(left.kind, left.completionAnchor.kind,
                  left.completionAnchor.node, left.completionAnchor.scope,
                  left.completionAnchor.position, left.acquisitionAnchor.kind,
                  left.acquisitionAnchor.node, left.acquisitionAnchor.scope,
                  left.acquisitionAnchor.position, left.scope, left.distance,
                  left.sourceResource, left.targetResource, left.id) <
         std::tie(right.kind, right.completionAnchor.kind,
                  right.completionAnchor.node, right.completionAnchor.scope,
                  right.completionAnchor.position, right.acquisitionAnchor.kind,
                  right.acquisitionAnchor.node, right.acquisitionAnchor.scope,
                  right.acquisitionAnchor.position, right.scope, right.distance,
                  right.sourceResource, right.targetResource, right.id);
}

bool sameRectangleKey(const SyncCoverStorageProtocolFrontier &left,
                      const SyncCoverStorageProtocolFrontier &right) {
  return left.kind == right.kind &&
         anchorKey(left.completionAnchor) ==
             anchorKey(right.completionAnchor) &&
         anchorKey(left.acquisitionAnchor) ==
             anchorKey(right.acquisitionAnchor) &&
         left.scope == right.scope && left.distance == right.distance &&
         left.sourceResource == right.sourceResource &&
         left.targetResource == right.targetResource;
}

bool rectangleMatchesFrontier(
    const SyncCoverStorageProtocolRectangle &rectangle,
    const SyncCoverStorageProtocolFrontier &frontier) {
  return rectangle.automaton == frontier.automaton &&
         rectangle.kind == frontier.kind &&
         anchorKey(rectangle.completionAnchor) ==
             anchorKey(frontier.completionAnchor) &&
         anchorKey(rectangle.acquisitionAnchor) ==
             anchorKey(frontier.acquisitionAnchor) &&
         rectangle.scope == frontier.scope &&
         rectangle.distance == frontier.distance &&
         rectangle.sourceResource == frontier.sourceResource &&
         rectangle.targetResource == frontier.targetResource;
}

bool edgeRefsEqual(const SyncCoverStorageLifecycleEdgeRef &left,
                   const SyncCoverStorageLifecycleEdgeRef &right) {
  return left.component == right.component && left.edge == right.edge;
}

bool hasKind(SyncCoverStorageLifecycleEdgeKindMask kinds,
             SyncCoverStorageLifecycleEdgeKind kind) {
  return (kinds & syncCoverStorageLifecycleEdgeKindBit(kind)) != 0;
}

template <typename BudgetT>
std::optional<bool>
meteredContains(const std::vector<SyncCoverDemandId> &demands,
                SyncCoverDemandId demand, BudgetT &budget) {
  std::size_t begin = 0;
  std::size_t end = demands.size();
  while (begin < end) {
    if (!budget.consume()) {
      return std::nullopt;
    }
    const std::size_t middle = begin + (end - begin) / 2;
    if (demands[middle] < demand) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (!budget.consume()) {
    return std::nullopt;
  }
  return begin != demands.size() && demands[begin] == demand;
}

template <typename BudgetT>
std::optional<bool>
planContainsFrontier(const SyncCoverStorageProtocolFrontierPlan &plan,
                     SyncCoverStorageProtocolFrontierId frontier,
                     BudgetT &budget) {
  if (!budget.consume(3)) {
    return std::nullopt;
  }
  const bool precedesPlan =
      plan.frontiers.empty() || frontier < plan.frontiers.front();
  if (precedesPlan) {
    return false;
  }
  const std::size_t offset = frontier - plan.frontiers.front();
  return offset < plan.frontiers.size() && plan.frontiers[offset] == frontier;
}

template <typename BudgetT>
std::optional<bool>
validateFrontier(const SyncCoverGraph &graph,
                 const SyncCoverStorageProtocolAutomaton &automaton,
                 const SyncCoverStorageProtocolFrontier &frontier,
                 SyncCoverStorageProtocolFrontierId expectedId,
                 BudgetT &budget) {
  const bool invalidHeader =
      frontier.id != expectedId || frontier.automaton != automaton.id ||
      frontier.completionAnchor.kind != SyncCoverAnchorKind::AfterNode ||
      frontier.acquisitionAnchor.kind != SyncCoverAnchorKind::BeforeNode ||
      frontier.completionAnchor.scope != 0 ||
      frontier.completionAnchor.position != 0 ||
      frontier.acquisitionAnchor.scope != 0 ||
      frontier.acquisitionAnchor.position != 0 ||
      frontier.completionAnchor.node >= graph.getNodes().size() ||
      frontier.acquisitionAnchor.node >= graph.getNodes().size() ||
      frontier.sourceResource == frontier.targetResource;
  if (invalidHeader) {
    return false;
  }
  const SyncCoverNode &completion =
      graph.getNodes()[frontier.completionAnchor.node];
  const SyncCoverNode &acquisition =
      graph.getNodes()[frontier.acquisitionAnchor.node];
  if (completion.resource != frontier.sourceResource ||
      acquisition.resource != frontier.targetResource) {
    return false;
  }

  if (frontier.transfer) {
    const bool invalidTransfer =
        *frontier.transfer >= automaton.transfers.size() || !frontier.edge;
    if (invalidTransfer) {
      return false;
    }
    const SyncCoverStorageProtocolTransfer &transfer =
        automaton.transfers[*frontier.transfer];
    const bool kindMatches =
        frontier.kind == SyncCoverStorageProtocolFrontierKind::Ready
            ? hasKind(transfer.kinds, SyncCoverStorageLifecycleEdgeKind::Ready)
            : transfer.distance != 0 &&
                  (hasKind(transfer.kinds,
                           SyncCoverStorageLifecycleEdgeKind::Release) ||
                   hasKind(transfer.kinds,
                           SyncCoverStorageLifecycleEdgeKind::Exclusion));
    if (transfer.id != *frontier.transfer ||
        !edgeRefsEqual(transfer.edge, *frontier.edge) || !kindMatches ||
        frontier.scope != transfer.scope ||
        frontier.distance != transfer.distance ||
        frontier.sourceResource != transfer.sourceResource ||
        frontier.targetResource != transfer.targetResource) {
      return false;
    }
  } else if (frontier.edge) {
    return false;
  }

  const bool direct =
      !frontier.completionCutFact && !frontier.completionCertificate;
  if (direct) {
    return frontier.transfer.has_value();
  }
  if (frontier.completionCutFact && frontier.completionCertificate) {
    return false;
  }
  if (frontier.completionCutFact) {
    if (!frontier.transfer ||
        *frontier.completionCutFact >= graph.getCompletionCutFacts().size()) {
      return false;
    }
    const SyncCoverCompletionCutFact &fact =
        graph.getCompletionCutFacts()[*frontier.completionCutFact];
    const SyncCoverDemandId demand =
        automaton.transfers[*frontier.transfer].demand;
    const bool factMatches =
        fact.id == *frontier.completionCutFact &&
        fact.completionNode == frontier.completionAnchor.node &&
        fact.sourceResource == frontier.sourceResource &&
        fact.targetResource == frontier.targetResource;
    if (!factMatches) {
      return false;
    }
    return meteredContains(fact.demands, demand, budget);
  }
  if (!frontier.completionCertificate || frontier.transfer || frontier.edge ||
      frontier.kind != SyncCoverStorageProtocolFrontierKind::Ready ||
      frontier.distance != 0 ||
      *frontier.completionCertificate >=
          graph.getTargetCompletionCertificates().size()) {
    return false;
  }
  const SyncCoverTargetCompletionCertificate &certificate =
      graph.getTargetCompletionCertificates()[*frontier.completionCertificate];
  return certificate.id == *frontier.completionCertificate &&
         certificate.completionNode == frontier.completionAnchor.node &&
         certificate.target == frontier.acquisitionAnchor.node &&
         certificate.sourceResource == frontier.sourceResource &&
         certificate.targetResource == frontier.targetResource;
}

bool detailLess(const SyncCoverStorageProtocolRectangleGroundingDetail &left,
                const SyncCoverStorageProtocolRectangleGroundingDetail &right) {
  if (left.coverageRows != right.coverageRows) {
    return left.coverageRows > right.coverageRows;
  }
  if (left.frontierCount != right.frontierCount) {
    return left.frontierCount > right.frontierCount;
  }
  return left.rectangle < right.rectangle;
}

bool validCoverageLimits(const SyncCoverCoverageLimits &limits) {
  return limits.maximumWorkspaceWords != 0 && limits.maximumResultWords != 0 &&
         limits.maximumTotalWords != 0 && limits.maximumResultRows != 0 &&
         limits.maximumMechanismRows != 0;
}

std::optional<bool>
containsDomain(const std::vector<SyncCoverStorageDomainId> &domains,
               SyncCoverStorageDomainId domain,
               SyncCoverCoverageWorkBudget &budget) {
  std::size_t begin = 0;
  std::size_t end = domains.size();
  while (begin < end) {
    if (!budget.consume()) {
      return std::nullopt;
    }
    const std::size_t middle = begin + (end - begin) / 2;
    if (domains[middle] < domain) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (!budget.consume()) {
    return std::nullopt;
  }
  return begin != domains.size() && domains[begin] == domain;
}

std::optional<bool>
admitsWholeDemand(const SyncCoverGraph &graph, SyncCoverDemandId demandId,
                  const std::vector<SyncCoverStorageDomainId> &domains,
                  SyncCoverCoverageWorkBudget &budget) {
  if (demandId >= graph.getDemands().size()) {
    return false;
  }
  const SyncCoverDemand &demand = graph.getDemands()[demandId];
  if (demand.storageWitnesses.empty()) {
    return false;
  }
  for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
    if (!budget.consume()) {
      return std::nullopt;
    }
    if (witnessId >= graph.getStorageWitnesses().size()) {
      return false;
    }
    const SyncCoverStorageWitness &witness =
        graph.getStorageWitnesses()[witnessId];
    if (witness.sourceAccess >= graph.getStorageAccesses().size()) {
      return false;
    }
    const SyncCoverStorageDomainId domain =
        graph.getStorageAccesses()[witness.sourceAccess].domain;
    const std::optional<bool> admitted =
        containsDomain(domains, domain, budget);
    if (!admitted || !*admitted) {
      return admitted;
    }
  }
  return true;
}

} // namespace

SyncCoverStorageProtocolRectangleIndex
mlir::pto::buildSyncCoverStorageProtocolRectangleIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageProtocolAutomatonIndex &automatonIndex,
    const SyncCoverStorageProtocolFrontierIndex &frontierIndex,
    const SyncCoverStorageProtocolRectangleLimits &limits) {
  SyncCoverStorageProtocolRectangleIndex result;
  const auto fail = [&](SyncCoverStorageProtocolRectangleStatistics statistics,
                        SyncCoverStorageProtocolRectangleError error) {
    result.rectangles_.clear();
    result.frontierIncidences_.clear();
    statistics.plans = 0;
    statistics.rectangles = 0;
    statistics.readyRectangles = 0;
    statistics.reuseRectangles = 0;
    statistics.mergedRectangles = 0;
    statistics.frontierIncidences = 0;
    statistics.maximumRectangleFrontiers = 0;
    statistics.truncated =
        error == SyncCoverStorageProtocolRectangleError::LimitExceeded;
    result.statistics_ = statistics;
    result.error_ = error;
    return std::move(result);
  };
  const bool invalidLimit =
      limits.maximumWorkUnits == 0 || limits.maximumFrontierInspections == 0 ||
      limits.maximumRectangles == 0 || limits.maximumFrontierIncidences == 0;
  if (invalidLimit) {
    return fail({}, SyncCoverStorageProtocolRectangleError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return fail({}, SyncCoverStorageProtocolRectangleError::InvalidGraph);
  }
  result.bindToGraph(graph);
  if (!automatonIndex.isComplete()) {
    return fail(
        {}, SyncCoverStorageProtocolRectangleError::IncompleteAutomatonIndex);
  }
  if (!automatonIndex.isForGraph(graph)) {
    return fail({}, SyncCoverStorageProtocolRectangleError::InvalidGraph);
  }
  if (!frontierIndex.isComplete()) {
    return fail(
        {}, SyncCoverStorageProtocolRectangleError::IncompleteFrontierIndex);
  }
  if (!frontierIndex.isForGraph(graph)) {
    return fail({}, SyncCoverStorageProtocolRectangleError::InvalidGraph);
  }

  SyncCoverStorageProtocolRectangleStatistics statistics;
  WorkBudget budget(limits.maximumWorkUnits, statistics.workUnits);
  const std::vector<SyncCoverStorageProtocolAutomaton> &automata =
      automatonIndex.getAutomata();
  const std::vector<SyncCoverStorageProtocolFrontier> &frontiers =
      frontierIndex.getFrontiers();
  const std::vector<SyncCoverStorageProtocolFrontierPlan> &plans =
      frontierIndex.getPlans();
  result.rectangles_.reserve(
      std::min(limits.maximumRectangles, frontiers.size()));
  result.frontierIncidences_.reserve(
      std::min(limits.maximumFrontierIncidences, frontiers.size()));

  for (std::size_t planPosition = 0; planPosition < plans.size();
       ++planPosition) {
    const SyncCoverStorageProtocolFrontierPlan &plan = plans[planPosition];
    const bool invalidPlan =
        plan.id != planPosition || plan.automaton >= automata.size() ||
        automata[plan.automaton].id != plan.automaton ||
        plan.group != automata[plan.automaton].group ||
        plan.owningScope != automata[plan.automaton].owningScope ||
        plan.frontiers.empty();
    if (invalidPlan) {
      return fail(statistics,
                  SyncCoverStorageProtocolRectangleError::InvalidGraph);
    }
    const bool inspectionLimitReached =
        statistics.frontierInspections > limits.maximumFrontierInspections ||
        plan.frontiers.size() >
            limits.maximumFrontierInspections - statistics.frontierInspections;
    const bool incidenceLimitReached =
        result.frontierIncidences_.size() > limits.maximumFrontierIncidences ||
        plan.frontiers.size() > limits.maximumFrontierIncidences -
                                    result.frontierIncidences_.size();
    if (inspectionLimitReached || incidenceLimitReached ||
        !budget.consume(plan.frontiers.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolRectangleError::LimitExceeded);
    }
    statistics.frontierInspections += plan.frontiers.size();
    std::vector<SyncCoverStorageProtocolFrontierId> ordered = plan.frontiers;
    for (SyncCoverStorageProtocolFrontierId frontierId : ordered) {
      if (frontierId >= frontiers.size()) {
        return fail(statistics,
                    SyncCoverStorageProtocolRectangleError::InvalidGraph);
      }
      const std::optional<bool> validFrontier =
          validateFrontier(graph, automata[plan.automaton],
                           frontiers[frontierId], frontierId, budget);
      if (!validFrontier) {
        return fail(statistics,
                    SyncCoverStorageProtocolRectangleError::LimitExceeded);
      }
      if (!*validFrontier) {
        return fail(statistics,
                    SyncCoverStorageProtocolRectangleError::InvalidGraph);
      }
    }
    if (!consumeSortWork(budget, ordered.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolRectangleError::LimitExceeded);
    }
    std::sort(ordered.begin(), ordered.end(),
              [&](SyncCoverStorageProtocolFrontierId left,
                  SyncCoverStorageProtocolFrontierId right) {
                return frontierLess(frontiers[left], frontiers[right]);
              });
    const std::size_t orderedSize = ordered.size();
    for (std::size_t begin = 0; begin < orderedSize;) {
      std::size_t end = begin + 1;
      while (end < orderedSize) {
        const bool same = sameRectangleKey(frontiers[ordered[begin]],
                                           frontiers[ordered[end]]);
        if (!same) {
          break;
        }
        if (!budget.consume()) {
          return fail(statistics,
                      SyncCoverStorageProtocolRectangleError::LimitExceeded);
        }
        ++end;
      }
      const bool rectangleUnavailable =
          result.rectangles_.size() >= limits.maximumRectangles ||
          !budget.consume(end - begin + 1);
      if (rectangleUnavailable) {
        return fail(statistics,
                    SyncCoverStorageProtocolRectangleError::LimitExceeded);
      }
      const SyncCoverStorageProtocolFrontier &representative =
          frontiers[ordered[begin]];
      const SyncCoverStorageProtocolRectangleId rectangleId =
          result.rectangles_.size();
      result.rectangles_.push_back(
          {rectangleId, plan.id, plan.automaton, representative.kind,
           representative.completionAnchor, representative.acquisitionAnchor,
           representative.scope, representative.distance,
           representative.sourceResource, representative.targetResource,
           result.frontierIncidences_.size(), end - begin});
      result.frontierIncidences_.insert(result.frontierIncidences_.end(),
                                        ordered.begin() + begin,
                                        ordered.begin() + end);
      if (representative.kind == SyncCoverStorageProtocolFrontierKind::Ready) {
        if (!checkedIncrement(statistics.readyRectangles)) {
          return fail(
              statistics,
              SyncCoverStorageProtocolRectangleError::ArithmeticOverflow);
        }
      } else if (!checkedIncrement(statistics.reuseRectangles)) {
        return fail(statistics,
                    SyncCoverStorageProtocolRectangleError::ArithmeticOverflow);
      }
      if (end - begin > 1 && !checkedIncrement(statistics.mergedRectangles)) {
        return fail(statistics,
                    SyncCoverStorageProtocolRectangleError::ArithmeticOverflow);
      }
      statistics.maximumRectangleFrontiers =
          std::max(statistics.maximumRectangleFrontiers, end - begin);
      begin = end;
    }
    if (!checkedIncrement(statistics.plans)) {
      return fail(statistics,
                  SyncCoverStorageProtocolRectangleError::ArithmeticOverflow);
    }
  }
  statistics.rectangles = result.rectangles_.size();
  statistics.frontierIncidences = result.frontierIncidences_.size();
  result.statistics_ = statistics;
  return result;
}

SyncCoverStorageProtocolRectangleGrounding
mlir::pto::groundSyncCoverStorageProtocolRectangles(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    const SyncCoverStorageProtocolAutomatonIndex &automatonIndex,
    const SyncCoverStorageProtocolFrontierIndex &frontierIndex,
    const SyncCoverStorageProtocolRectangleIndex &rectangleIndex,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const SyncCoverStorageProtocolRectangleGroundingLimits &limits) {
  SyncCoverStorageProtocolRectangleGrounding result;
  const auto finish =
      [&](SyncCoverStorageProtocolRectangleGroundingError error) {
        result.error_ = error;
        return std::move(result);
      };
  const bool invalidLimit = limits.maximumWorkUnits == 0 ||
                            limits.maximumAdmittedDemandIncidences == 0 ||
                            limits.maximumBatchRectangles == 0 ||
                            limits.maximumCoverageBatches == 0 ||
                            limits.maximumDetails == 0 ||
                            !validCoverageLimits(limits.coverageLimits);
  if (invalidLimit) {
    return finish(
        SyncCoverStorageProtocolRectangleGroundingError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return finish(
        SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph);
  }
  if (!automatonIndex.isComplete()) {
    return finish(SyncCoverStorageProtocolRectangleGroundingError::
                      IncompleteAutomatonIndex);
  }
  if (!automatonIndex.isForGraph(graph)) {
    return finish(
        SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph);
  }
  if (!frontierIndex.isComplete()) {
    return finish(SyncCoverStorageProtocolRectangleGroundingError::
                      IncompleteFrontierIndex);
  }
  if (!frontierIndex.isForGraph(graph)) {
    return finish(
        SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph);
  }
  if (!rectangleIndex.isComplete()) {
    return finish(SyncCoverStorageProtocolRectangleGroundingError::
                      IncompleteRectangleIndex);
  }
  if (!rectangleIndex.isForGraph(graph)) {
    return finish(
        SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph);
  }

  SyncCoverCoverageWorkBudget budget(limits.maximumWorkUnits);
  const auto stop = [&](SyncCoverStorageProtocolRectangleGroundingError error,
                        bool truncated) {
    result.statistics_.workUnits = budget.workUnits;
    result.statistics_.truncated = truncated;
    result.error_ = error;
    return std::move(result);
  };
  const std::vector<SyncCoverStorageProtocolAutomaton> &automata =
      automatonIndex.getAutomata();
  const std::vector<SyncCoverStorageProtocolFrontier> &frontiers =
      frontierIndex.getFrontiers();
  const std::vector<SyncCoverStorageProtocolFrontierPlan> &plans =
      frontierIndex.getPlans();
  const std::vector<SyncCoverStorageProtocolFrontierId> &incidences =
      rectangleIndex.getFrontierIncidences();
  const std::vector<SyncCoverCompletionCutFact> &facts =
      graph.getCompletionCutFacts();
  const std::vector<SyncCoverTargetCompletionCertificate> &certificates =
      graph.getTargetCompletionCertificates();
  result.details_.reserve(
      std::min(limits.maximumDetails, rectangleIndex.getRectangles().size()));
  const std::size_t coverageBatchSize = std::min(
      {limits.maximumBatchRectangles, limits.coverageLimits.maximumResultRows,
       limits.coverageLimits.maximumMechanismRows});
  if (coverageBatchSize == 0) {
    return stop(SyncCoverStorageProtocolRectangleGroundingError::InvalidLimit,
                false);
  }
  const auto recordCoverage =
      [&](const SyncCoverStorageProtocolRectangle &rectangle,
          std::size_t admittedDemandCount, std::size_t coverageRows)
      -> SyncCoverStorageProtocolRectangleGroundingError {
    std::size_t next = 0;
    if (!checkedAdd(result.statistics_.evaluatedRectangles, 1, next)) {
      return SyncCoverStorageProtocolRectangleGroundingError::
          ArithmeticOverflow;
    }
    result.statistics_.evaluatedRectangles = next;
    if (coverageRows != 0) {
      if (!checkedAdd(result.statistics_.rectanglesWithCoverage, 1, next)) {
        return SyncCoverStorageProtocolRectangleGroundingError::
            ArithmeticOverflow;
      }
      result.statistics_.rectanglesWithCoverage = next;
    }
    if (coverageRows > 1) {
      if (!checkedAdd(result.statistics_.rectanglesCoveringMultipleRows, 1,
                      next)) {
        return SyncCoverStorageProtocolRectangleGroundingError::
            ArithmeticOverflow;
      }
      result.statistics_.rectanglesCoveringMultipleRows = next;
    }
    if (!checkedAdd(result.statistics_.totalCoverageRows, coverageRows, next)) {
      return SyncCoverStorageProtocolRectangleGroundingError::
          ArithmeticOverflow;
    }
    result.statistics_.totalCoverageRows = next;
    result.statistics_.maximumCoverageRows =
        std::max(result.statistics_.maximumCoverageRows, coverageRows);
    if (coverageRows == 0) {
      return SyncCoverStorageProtocolRectangleGroundingError::None;
    }
    const SyncCoverStorageProtocolRectangleGroundingDetail detail{
        rectangle.id, rectangle.automaton, rectangle.frontierCount,
        admittedDemandCount, coverageRows};
    const bool detailWorkUnavailable =
        !budget.consume(result.details_.size() + 1);
    if (detailWorkUnavailable) {
      return SyncCoverStorageProtocolRectangleGroundingError::WorkLimitExceeded;
    }
    const auto position = std::lower_bound(
        result.details_.begin(), result.details_.end(), detail, detailLess);
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
    return SyncCoverStorageProtocolRectangleGroundingError::None;
  };

  for (std::size_t batchBegin = 0;
       batchBegin < rectangleIndex.getRectangles().size();) {
    if (result.statistics_.coverageBatches >= limits.maximumCoverageBatches) {
      return stop(SyncCoverStorageProtocolRectangleGroundingError::
                      CoverageLimitExceeded,
                  true);
    }
    const std::size_t batchCount = std::min(
        coverageBatchSize, rectangleIndex.getRectangles().size() - batchBegin);
    std::vector<SyncCoverCompletionSupply> supplies;
    supplies.reserve(batchCount);
    std::vector<const SyncCoverStorageProtocolRectangle *> preparedRectangles;
    preparedRectangles.reserve(batchCount);
    std::vector<std::size_t> admittedCounts;
    admittedCounts.reserve(batchCount);

    for (std::size_t local = 0; local < batchCount; ++local) {
      const std::size_t rectanglePosition = batchBegin + local;
      if (!budget.consume()) {
        return stop(
            SyncCoverStorageProtocolRectangleGroundingError::WorkLimitExceeded,
            true);
      }
      const SyncCoverStorageProtocolRectangle &rectangle =
          rectangleIndex.getRectangles()[rectanglePosition];
      const bool invalidRange =
          rectangle.id != rectanglePosition || rectangle.plan >= plans.size() ||
          rectangle.automaton >= automata.size() ||
          rectangle.frontierCount == 0 ||
          rectangle.frontierBegin > incidences.size() ||
          rectangle.frontierCount >
              incidences.size() - rectangle.frontierBegin ||
          rectangle.completionAnchor.node >= graph.getNodes().size() ||
          rectangle.acquisitionAnchor.node >= graph.getNodes().size();
      if (invalidRange) {
        return stop(
            SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph,
            false);
      }
      const SyncCoverStorageProtocolAutomaton &automaton =
          automata[rectangle.automaton];
      const SyncCoverStorageProtocolFrontierPlan &plan = plans[rectangle.plan];
      const bool invalidPlan = plan.id != rectangle.plan ||
                               plan.automaton != rectangle.automaton ||
                               plan.group != automaton.group ||
                               plan.owningScope != automaton.owningScope;
      if (invalidPlan) {
        return stop(
            SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph,
            false);
      }
      std::vector<SyncCoverDemandId> admittedDemands;
      const auto appendAdmittedDemand =
          [&](SyncCoverDemandId demand,
              const std::vector<SyncCoverStorageDomainId> *domains)
          -> SyncCoverStorageProtocolRectangleGroundingError {
        if (demand >= graph.getDemands().size()) {
          return SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph;
        }
        if (domains) {
          const std::optional<bool> admitted =
              admitsWholeDemand(graph, demand, *domains, budget);
          if (!admitted) {
            return SyncCoverStorageProtocolRectangleGroundingError::
                WorkLimitExceeded;
          }
          if (!*admitted) {
            return SyncCoverStorageProtocolRectangleGroundingError::None;
          }
        }
        const bool incidenceLimitReached =
            result.statistics_.admittedDemandIncidences >=
            limits.maximumAdmittedDemandIncidences;
        if (incidenceLimitReached) {
          return SyncCoverStorageProtocolRectangleGroundingError::
              IncidenceLimitExceeded;
        }
        if (!budget.consume()) {
          return SyncCoverStorageProtocolRectangleGroundingError::
              WorkLimitExceeded;
        }
        ++result.statistics_.admittedDemandIncidences;
        admittedDemands.push_back(demand);
        return SyncCoverStorageProtocolRectangleGroundingError::None;
      };
      for (std::size_t offset = 0; offset < rectangle.frontierCount; ++offset) {
        if (!budget.consume()) {
          return stop(SyncCoverStorageProtocolRectangleGroundingError::
                          WorkLimitExceeded,
                      true);
        }
        const SyncCoverStorageProtocolFrontierId frontierId =
            incidences[rectangle.frontierBegin + offset];
        if (frontierId >= frontiers.size()) {
          return stop(
              SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph,
              false);
        }
        const SyncCoverStorageProtocolFrontier &frontier =
            frontiers[frontierId];
        const std::optional<bool> validFrontier =
            validateFrontier(graph, automaton, frontier, frontierId, budget);
        if (!validFrontier) {
          return stop(SyncCoverStorageProtocolRectangleGroundingError::
                          WorkLimitExceeded,
                      true);
        }
        const std::optional<bool> frontierInPlan =
            planContainsFrontier(plan, frontierId, budget);
        if (!frontierInPlan) {
          return stop(SyncCoverStorageProtocolRectangleGroundingError::
                          WorkLimitExceeded,
                      true);
        }
        const bool invalidFrontier =
            !*validFrontier || !*frontierInPlan ||
            !rectangleMatchesFrontier(rectangle, frontier) ||
            !sameRectangleKey(frontiers[incidences[rectangle.frontierBegin]],
                              frontier);
        if (invalidFrontier) {
          return stop(
              SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph,
              false);
        }
        if (frontier.completionCutFact) {
          if (*frontier.completionCutFact >= facts.size()) {
            return stop(
                SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph,
                false);
          }
          if (!frontier.transfer) {
            return stop(
                SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph,
                false);
          }
          const SyncCoverCompletionCutFact &fact =
              facts[*frontier.completionCutFact];
          for (SyncCoverDemandId demand : fact.demands) {
            const SyncCoverStorageProtocolRectangleGroundingError error =
                appendAdmittedDemand(demand, &fact.storageDomains);
            if (error !=
                SyncCoverStorageProtocolRectangleGroundingError::None) {
              return stop(
                  error,
                  error == SyncCoverStorageProtocolRectangleGroundingError::
                               WorkLimitExceeded ||
                      error == SyncCoverStorageProtocolRectangleGroundingError::
                                   IncidenceLimitExceeded);
            }
          }
        } else if (frontier.completionCertificate) {
          if (*frontier.completionCertificate >= certificates.size()) {
            return stop(
                SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph,
                false);
          }
          const SyncCoverTargetCompletionCertificate &certificate =
              certificates[*frontier.completionCertificate];
          for (SyncCoverDemandId demand : certificate.demands) {
            const SyncCoverStorageProtocolRectangleGroundingError error =
                appendAdmittedDemand(demand, &certificate.storageDomains);
            if (error !=
                SyncCoverStorageProtocolRectangleGroundingError::None) {
              return stop(
                  error,
                  error == SyncCoverStorageProtocolRectangleGroundingError::
                               WorkLimitExceeded ||
                      error == SyncCoverStorageProtocolRectangleGroundingError::
                                   IncidenceLimitExceeded);
            }
          }
        } else {
          if (!frontier.transfer ||
              *frontier.transfer >= automaton.transfers.size()) {
            return stop(
                SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph,
                false);
          }
          const SyncCoverStorageProtocolRectangleGroundingError error =
              appendAdmittedDemand(
                  automaton.transfers[*frontier.transfer].demand, nullptr);
          if (error != SyncCoverStorageProtocolRectangleGroundingError::None) {
            return stop(
                error,
                error == SyncCoverStorageProtocolRectangleGroundingError::
                             WorkLimitExceeded ||
                    error == SyncCoverStorageProtocolRectangleGroundingError::
                                 IncidenceLimitExceeded);
          }
        }
      }
      if (!consumeSortWork(budget, admittedDemands.size())) {
        return stop(
            SyncCoverStorageProtocolRectangleGroundingError::WorkLimitExceeded,
            true);
      }
      std::sort(admittedDemands.begin(), admittedDemands.end());
      if (!budget.consume(admittedDemands.size())) {
        return stop(
            SyncCoverStorageProtocolRectangleGroundingError::WorkLimitExceeded,
            true);
      }
      admittedDemands.erase(
          std::unique(admittedDemands.begin(), admittedDemands.end()),
          admittedDemands.end());
      const bool admittedWorkUnavailable = !budget.consume();
      if (admittedWorkUnavailable) {
        return stop(
            SyncCoverStorageProtocolRectangleGroundingError::WorkLimitExceeded,
            true);
      }
      if (admittedDemands.empty()) {
        const SyncCoverStorageProtocolRectangleGroundingError recordError =
            recordCoverage(rectangle, 0, 0);
        if (recordError !=
            SyncCoverStorageProtocolRectangleGroundingError::None) {
          return stop(recordError,
                      recordError ==
                          SyncCoverStorageProtocolRectangleGroundingError::
                              WorkLimitExceeded);
        }
        continue;
      }
      const SyncCoverNode &completion =
          graph.getNodes()[rectangle.completionAnchor.node];
      const SyncCoverNode &acquisition =
          graph.getNodes()[rectangle.acquisitionAnchor.node];
      SyncCoverCompletionSupply supply;
      supply.mechanism = supplies.size();
      supply.edge = {rectangle.completionAnchor.node,
                     rectangle.acquisitionAnchor.node,
                     SyncCoverEdgeKind::CompletionSupply,
                     rectangle.scope,
                     rectangle.distance,
                     completion.guard,
                     acquisition.guard};
      admittedCounts.push_back(admittedDemands.size());
      preparedRectangles.push_back(&rectangle);
      supply.allowedDemands = std::move(admittedDemands);
      supplies.push_back(std::move(supply));
    }

    if (supplies.empty()) {
      batchBegin += batchCount;
      continue;
    }
    const SyncCoverSingletonCoverageResult coverage =
        computeSyncCoverSingletonCoverage(graph, expansion, supplies.size(),
                                          supplies, activeDemands,
                                          limits.coverageLimits, &budget);
    if (coverage.error == SyncCoverCoverageError::WorkLimitExceeded) {
      return stop(
          SyncCoverStorageProtocolRectangleGroundingError::WorkLimitExceeded,
          true);
    }
    if (coverage.error == SyncCoverCoverageError::LimitExceeded) {
      return stop(SyncCoverStorageProtocolRectangleGroundingError::
                      CoverageLimitExceeded,
                  true);
    }
    if (!coverage) {
      return stop(
          SyncCoverStorageProtocolRectangleGroundingError::CoverageFailure,
          false);
    }
    if (!checkedIncrement(result.statistics_.coverageBatches)) {
      return stop(
          SyncCoverStorageProtocolRectangleGroundingError::ArithmeticOverflow,
          false);
    }

    for (std::size_t local = 0; local < supplies.size(); ++local) {
      const SyncCoverStorageProtocolRectangle &rectangle =
          *preparedRectangles[local];
      if (local >= coverage.mechanisms.size()) {
        return stop(
            SyncCoverStorageProtocolRectangleGroundingError::CoverageFailure,
            false);
      }
      const std::size_t coverageWords =
          coverage.mechanisms[local].getWords().size();
      const bool coverageWordOverflow =
          coverageWords > std::numeric_limits<std::size_t>::max() / 3;
      if (coverageWordOverflow) {
        return stop(
            SyncCoverStorageProtocolRectangleGroundingError::ArithmeticOverflow,
            false);
      }
      if (!budget.consume(coverageWords * 3)) {
        return stop(
            SyncCoverStorageProtocolRectangleGroundingError::WorkLimitExceeded,
            true);
      }
      SyncCoverDemandSet incrementalCoverage = coverage.mechanisms[local];
      incrementalCoverage.subtract(coverage.baseline);
      const std::size_t coverageRows = incrementalCoverage.count();
      const SyncCoverStorageProtocolRectangleGroundingError recordError =
          recordCoverage(rectangle, admittedCounts[local], coverageRows);
      if (recordError !=
          SyncCoverStorageProtocolRectangleGroundingError::None) {
        return stop(recordError,
                    recordError ==
                        SyncCoverStorageProtocolRectangleGroundingError::
                            WorkLimitExceeded);
      }
    }
    batchBegin += batchCount;
  }
  result.statistics_.workUnits = budget.workUnits;
  return result;
}
