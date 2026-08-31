// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolFrontiers.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

class WorkBudget {
public:
  WorkBudget(std::size_t maximum, std::size_t &used)
      : maximum_(maximum), used_(used) {}

  bool consume(std::size_t amount = 1) {
    if (used_ > maximum_ || amount > maximum_ - used_) {
      return false;
    }
    used_ += amount;
    return true;
  }

private:
  std::size_t maximum_ = 0;
  std::size_t &used_;
};

constexpr SyncCoverStorageLifecycleEdgeKindMask readyBit() {
  return syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Ready);
}

constexpr SyncCoverStorageLifecycleEdgeKindMask releaseBit() {
  return syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Release);
}

constexpr SyncCoverStorageLifecycleEdgeKindMask exclusionBit() {
  return syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Exclusion);
}

bool consumeSortWork(WorkBudget &budget, std::size_t elementCount) {
  if (elementCount < 2) {
    return true;
  }
  std::size_t levels = 0;
  constexpr std::size_t kMaximumSize = std::numeric_limits<std::size_t>::max();
  for (std::size_t covered = 1; covered < elementCount;) {
    ++levels;
    if (covered > kMaximumSize / 2) {
      return false;
    }
    covered *= 2;
  }
  constexpr std::size_t operationsPerLevel = 4;
  const bool levelWorkOverflows =
      levels > std::numeric_limits<std::size_t>::max() / operationsPerLevel;
  if (levelWorkOverflows) {
    return false;
  }
  const std::size_t workPerElement = levels * operationsPerLevel;
  return elementCount <=
             std::numeric_limits<std::size_t>::max() / workPerElement &&
         budget.consume(elementCount * workPerElement);
}

bool statePairLess(const SyncCoverStorageProtocolStatePair &left,
                   const SyncCoverStorageProtocolStatePair &right) {
  return std::tie(left.source, left.target) <
         std::tie(right.source, right.target);
}

bool checkedAdd(std::size_t &value, std::size_t amount) {
  constexpr std::size_t kMaximumSize = std::numeric_limits<std::size_t>::max();
  if (amount > kMaximumSize - value) {
    return false;
  }
  value += amount;
  return true;
}

struct TransferDescription {
  const SyncCoverStorageLifecycleEdge *edge = nullptr;
  const SyncCoverStorageLifecycleEpoch *source = nullptr;
  const SyncCoverStorageLifecycleEpoch *target = nullptr;
};

struct CertificateEntry {
  SyncCoverDemandId demand = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  SyncCoverTargetCompletionCertificateId certificate = 0;
  SyncCoverNodeId completionNode = 0;
  SyncCoverNodeId target = 0;
};

struct CompletionCutFactEntry {
  SyncCoverDemandId demand = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  SyncCoverCompletionCutFactId fact = 0;
  SyncCoverNodeId completionNode = 0;
};

bool completionCutFactLess(const CompletionCutFactEntry &left,
                           const CompletionCutFactEntry &right) {
  return std::tie(left.demand, left.sourceResource, left.targetResource,
                  left.fact, left.completionNode) <
         std::tie(right.demand, right.sourceResource, right.targetResource,
                  right.fact, right.completionNode);
}

std::optional<std::size_t> lowerBoundCompletionCutFact(
    const std::vector<CompletionCutFactEntry> &facts,
    const CompletionCutFactEntry &key, WorkBudget &budget,
    bool &limitExceeded) {
  std::size_t begin = 0;
  std::size_t end = facts.size();
  while (begin < end) {
    if (!budget.consume()) {
      limitExceeded = true;
      return std::nullopt;
    }
    const std::size_t middle = begin + (end - begin) / 2;
    const CompletionCutFactEntry &entry = facts[middle];
    const bool before =
        std::tie(entry.demand, entry.sourceResource, entry.targetResource) <
        std::tie(key.demand, key.sourceResource, key.targetResource);
    if (before) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

std::optional<bool> containsStorageDomain(
    const std::vector<SyncCoverStorageDomainId> &domains,
    SyncCoverStorageDomainId domain, WorkBudget &budget) {
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
  return begin < domains.size() && domains[begin] == domain;
}

bool certificateLess(const CertificateEntry &left,
                     const CertificateEntry &right) {
  return std::tie(left.demand, left.sourceResource, left.targetResource,
                  left.certificate, left.completionNode, left.target) <
         std::tie(right.demand, right.sourceResource, right.targetResource,
                  right.certificate, right.completionNode, right.target);
}

std::optional<bool>
sortedIntersects(const std::vector<SyncCoverDemandId> &left,
                 const std::vector<SyncCoverDemandId> &right,
                 WorkBudget &budget) {
  std::size_t leftPosition = 0;
  std::size_t rightPosition = 0;
  const std::size_t leftSize = left.size();
  const std::size_t rightSize = right.size();
  while (leftPosition < leftSize && rightPosition < rightSize) {
    if (!budget.consume()) {
      return std::nullopt;
    }
    if (left[leftPosition] == right[rightPosition]) {
      return true;
    }
    if (left[leftPosition] < right[rightPosition]) {
      ++leftPosition;
    } else {
      ++rightPosition;
    }
  }
  return false;
}

std::optional<TransferDescription> validateTransfer(
    const SyncCoverGraph &graph,
    const std::vector<SyncCoverStorageLifecycleComponent> &components,
    const SyncCoverStorageProtocolAutomaton &automaton,
    const SyncCoverStorageProtocolTransfer &transfer,
    std::size_t transferPosition,
    const SyncCoverStorageProtocolFrontierLimits &limits,
    SyncCoverStorageProtocolFrontierStatistics &statistics, WorkBudget &budget,
    bool &limitExceeded) {
  const bool invalidReference =
      transfer.id != transferPosition ||
      transfer.edge.component >= components.size() ||
      transfer.edge.edge >= components[transfer.edge.component].edges.size();
  if (invalidReference) {
    return std::nullopt;
  }
  const SyncCoverStorageLifecycleComponent &component =
      components[transfer.edge.component];
  const SyncCoverStorageLifecycleEdge &edge =
      component.edges[transfer.edge.edge];
  const bool invalidEdge = component.id != transfer.edge.component ||
                           edge.id != transfer.edge.edge ||
                           edge.source >= component.epochs.size() ||
                           edge.target >= component.epochs.size() ||
                           edge.demand >= graph.getDemands().size();
  if (invalidEdge) {
    return std::nullopt;
  }
  const SyncCoverStorageLifecycleEpoch &source = component.epochs[edge.source];
  const SyncCoverStorageLifecycleEpoch &target = component.epochs[edge.target];
  const SyncCoverDemand &demand = graph.getDemands()[edge.demand];
  const bool inconsistentTransfer =
      source.id != edge.source || target.id != edge.target ||
      source.node != demand.source || target.node != demand.target ||
      transfer.demand != edge.demand || transfer.kinds != edge.kinds ||
      transfer.scope != edge.scope || transfer.distance != edge.distance ||
      transfer.sourceResource != source.resource ||
      transfer.targetResource != target.resource || automaton.stateCount == 0 ||
      transfer.activeStatePairs.empty();
  if (inconsistentTransfer) {
    return std::nullopt;
  }
  for (std::size_t pairPosition = 0;
       pairPosition < transfer.activeStatePairs.size(); ++pairPosition) {
    if (statistics.statePairInspections >= limits.maximumStatePairInspections ||
        !budget.consume()) {
      limitExceeded = true;
      return std::nullopt;
    }
    ++statistics.statePairInspections;
    const SyncCoverStorageProtocolStatePair &pair =
        transfer.activeStatePairs[pairPosition];
    const bool invalidPair =
        pair.source >= automaton.stateCount ||
        pair.target >= automaton.stateCount ||
        (pairPosition != 0 &&
         !statePairLess(transfer.activeStatePairs[pairPosition - 1], pair));
    if (invalidPair) {
      return std::nullopt;
    }
  }
  return TransferDescription{&edge, &source, &target};
}

struct EndpointFrontierLookup {
  std::optional<SyncCoverNodeId> completionNode;
  bool limitExceeded = false;
};

bool consumeEndpointSemanticWork(const SyncCoverGraph &graph,
                                 const SyncCoverNode &candidate,
                                 const SyncCoverNode &source,
                                 WorkBudget &budget) {
  constexpr std::size_t kScopeWalks = 4;
  constexpr std::size_t kMaximumSize = std::numeric_limits<std::size_t>::max();
  const std::size_t scopeCount = graph.getScopes().size();
  if (scopeCount > kMaximumSize / kScopeWalks) {
    return false;
  }
  std::size_t work = kScopeWalks * scopeCount;
  return checkedAdd(work, candidate.guard.literals.size()) &&
         checkedAdd(work, source.guard.literals.size()) && budget.consume(work);
}

EndpointFrontierLookup findEndpointFrontier(
    const SyncCoverGraph &graph, const SyncCoverStorageLifecycleEdge &edge,
    const SyncCoverStorageLifecycleEpoch &source,
    const SyncCoverStorageLifecycleEpoch &target, WorkBudget &budget) {
  EndpointFrontierLookup result;
  const std::size_t nodeCount = graph.getNodes().size();
  if (source.node >= nodeCount || target.node >= nodeCount) {
    return result;
  }
  const SyncCoverNode &sourceNode = graph.getNodes()[source.node];
  const SyncCoverNode &targetNode = graph.getNodes()[target.node];
  const bool invalidPhysicalExit =
      sourceNode.physicalExit >= graph.getNodes().size();
  if (invalidPhysicalExit) {
    return result;
  }
  const SyncCoverNodeId candidates[] = {source.node, sourceNode.physicalExit};
  for (SyncCoverNodeId candidateId : candidates) {
    if (result.completionNode && *result.completionNode == candidateId) {
      continue;
    }
    const SyncCoverNode &candidate = graph.getNodes()[candidateId];
    if (!consumeEndpointSemanticWork(graph, candidate, sourceNode, budget)) {
      result.limitExceeded = true;
      return result;
    }
    const bool completionTargetsFit =
        budget.consume(candidate.completionTargets.size());
    const bool dominatedSourcesFit =
        budget.consume(candidate.completionDominatedSources.size());
    if (!completionTargetsFit || !dominatedSourcesFit) {
      result.limitExceeded = true;
      return result;
    }
    const bool completionDominatesSource =
        candidateId == source.node ||
        std::binary_search(candidate.completionDominatedSources.begin(),
                           candidate.completionDominatedSources.end(),
                           source.node);
    const bool structurallyLegal =
        source.resource != target.resource &&
        candidate.resource == source.resource &&
        candidate.physicalAnchor != targetNode.physicalAnchor &&
        candidate.scope == sourceNode.scope &&
        candidate.guard.literals == sourceNode.guard.literals &&
        graph.scopeContains(edge.scope, candidate.scope) &&
        graph.scopeContains(edge.scope, targetNode.scope) &&
        completionDominatesSource &&
        syncCoverNodeCanProduceCompletion(graph, candidateId, target.resource);
    if (!structurallyLegal) {
      continue;
    }
    if (edge.distance == 0) {
      const SyncCoverAnchor completion{SyncCoverAnchorKind::AfterNode,
                                       candidateId, 0, 0};
      const SyncCoverAnchor acquisition{SyncCoverAnchorKind::BeforeNode,
                                        target.node, 0, 0};
      const std::optional<SyncCoverTimelinePosition> completionPosition =
          resolveSyncCoverAnchor(graph, completion);
      const std::optional<SyncCoverTimelinePosition> acquisitionPosition =
          resolveSyncCoverAnchor(graph, acquisition);
      if (!completionPosition || !acquisitionPosition ||
          *completionPosition >= *acquisitionPosition) {
        continue;
      }
    }
    if (!result.completionNode ||
        graph.getNodes()[*result.completionNode].order < candidate.order) {
      result.completionNode = candidateId;
    }
  }
  return result;
}

} // namespace

SyncCoverStorageProtocolFrontierIndex
mlir::pto::buildSyncCoverStorageProtocolFrontierIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolAutomatonIndex &automatonIndex,
    const SyncCoverStorageProtocolFrontierLimits &limits) {
  SyncCoverStorageProtocolFrontierIndex result;
  const auto fail = [&](SyncCoverStorageProtocolFrontierStatistics statistics,
                        SyncCoverStorageProtocolFrontierError error) {
    result.frontiers_.clear();
    result.plans_.clear();
    statistics.eligibleAutomata = 0;
    statistics.plans = 0;
    statistics.frontiers = 0;
    statistics.readyFrontiers = 0;
    statistics.reuseFrontiers = 0;
    statistics.directFrontiers = 0;
    statistics.completionCutFactFrontiers = 0;
    statistics.certificateFrontiers = 0;
    statistics.sameResourceRecurrenceReuses = 0;
    statistics.planFrontierIncidences = 0;
    statistics.maximumPlanFrontiers = 0;
    statistics.truncated =
        error == SyncCoverStorageProtocolFrontierError::LimitExceeded;
    result.statistics_ = statistics;
    result.error_ = error;
    return std::move(result);
  };
  const bool invalidLimit =
      limits.maximumWorkUnits == 0 || limits.maximumPlans == 0 ||
      limits.maximumFrontiers == 0 || limits.maximumTransferInspections == 0 ||
      limits.maximumStatePairInspections == 0 ||
      limits.maximumPlanFrontierIncidences == 0 ||
      limits.maximumCertificateDemandIncidences == 0 ||
      limits.maximumCompletionCutFactDemandIncidences == 0;
  if (invalidLimit) {
    return fail({}, SyncCoverStorageProtocolFrontierError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return fail({}, SyncCoverStorageProtocolFrontierError::InvalidGraph);
  }
  result.bindToGraph(graph);
  if (!lifecycleIndex.isComplete()) {
    return fail(
        {}, SyncCoverStorageProtocolFrontierError::IncompleteLifecycleIndex);
  }
  if (!automatonIndex.isComplete()) {
    return fail(
        {}, SyncCoverStorageProtocolFrontierError::IncompleteAutomatonIndex);
  }
  if (!automatonIndex.isForGraph(graph)) {
    return fail({}, SyncCoverStorageProtocolFrontierError::InvalidGraph);
  }

  SyncCoverStorageProtocolFrontierStatistics statistics;
  WorkBudget budget(limits.maximumWorkUnits, statistics.workUnits);
  std::vector<CompletionCutFactEntry> completionCutFacts;
  const std::vector<SyncCoverCompletionCutFact> &graphCompletionCutFacts =
      graph.getCompletionCutFacts();
  for (std::size_t factPosition = 0;
       factPosition < graphCompletionCutFacts.size(); ++factPosition) {
    const SyncCoverCompletionCutFact &fact =
        graphCompletionCutFacts[factPosition];
    const bool invalidFact = fact.id != factPosition ||
                             fact.completionNode >= graph.getNodes().size() ||
                             fact.sourceResource == fact.targetResource ||
                             fact.demands.empty();
    if (invalidFact) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::InvalidGraph);
    }
    const bool incidenceLimitReached =
        statistics.completionCutFactDemandIncidences >
            limits.maximumCompletionCutFactDemandIncidences ||
        fact.demands.size() > limits.maximumCompletionCutFactDemandIncidences -
                                  statistics.completionCutFactDemandIncidences;
    if (incidenceLimitReached ||
        fact.demands.size() == std::numeric_limits<std::size_t>::max()) {
      return fail(
          statistics,
          fact.demands.size() == std::numeric_limits<std::size_t>::max()
              ? SyncCoverStorageProtocolFrontierError::ArithmeticOverflow
              : SyncCoverStorageProtocolFrontierError::LimitExceeded);
    }
    const std::size_t factDemandWork = fact.demands.size() + 1;
    if (!budget.consume(factDemandWork)) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::LimitExceeded);
    }
    SyncCoverDemandId previousDemand = 0;
    bool havePreviousDemand = false;
    for (SyncCoverDemandId demandId : fact.demands) {
      const bool invalidDemandId =
          demandId >= graph.getDemands().size() ||
          (havePreviousDemand && demandId <= previousDemand);
      if (invalidDemandId) {
        return fail(statistics,
                    SyncCoverStorageProtocolFrontierError::InvalidGraph);
      }
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      const bool invalidDemand =
          demand.source >= graph.getNodes().size() ||
          demand.target >= graph.getNodes().size() || demand.distance != 0 ||
          graph.getNodes()[demand.source].resource != fact.sourceResource ||
          graph.getNodes()[demand.target].resource != fact.targetResource;
      if (invalidDemand) {
        return fail(statistics,
                    SyncCoverStorageProtocolFrontierError::InvalidGraph);
      }
      completionCutFacts.push_back({demandId, fact.sourceResource,
                                    fact.targetResource, fact.id,
                                    fact.completionNode});
      previousDemand = demandId;
      havePreviousDemand = true;
      ++statistics.completionCutFactDemandIncidences;
    }
  }
  if (!consumeSortWork(budget, completionCutFacts.size())) {
    return fail(statistics,
                SyncCoverStorageProtocolFrontierError::LimitExceeded);
  }
  std::sort(completionCutFacts.begin(), completionCutFacts.end(),
            completionCutFactLess);
  for (std::size_t fact = 1; fact < completionCutFacts.size(); ++fact) {
    if (!budget.consume()) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::LimitExceeded);
    }
    if (!completionCutFactLess(completionCutFacts[fact - 1],
                               completionCutFacts[fact])) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::InvalidGraph);
    }
  }
  std::vector<CertificateEntry> certificates;
  const std::vector<SyncCoverTargetCompletionCertificate> &graphCertificates =
      graph.getTargetCompletionCertificates();
  for (std::size_t certificatePosition = 0;
       certificatePosition < graphCertificates.size(); ++certificatePosition) {
    const SyncCoverTargetCompletionCertificate &certificate =
        graphCertificates[certificatePosition];
    const bool invalidCertificate =
        certificate.id != certificatePosition ||
        certificate.completionNode >= graph.getNodes().size() ||
        certificate.target >= graph.getNodes().size() ||
        certificate.sourceResource == certificate.targetResource ||
        certificate.demands.empty();
    if (invalidCertificate) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::InvalidGraph);
    }
    const bool incidenceLimitReached =
        statistics.certificateDemandIncidences >
            limits.maximumCertificateDemandIncidences ||
        certificate.demands.size() > limits.maximumCertificateDemandIncidences -
                                         statistics.certificateDemandIncidences;
    if (incidenceLimitReached ||
        certificate.demands.size() == std::numeric_limits<std::size_t>::max()) {
      return fail(
          statistics,
          certificate.demands.size() == std::numeric_limits<std::size_t>::max()
              ? SyncCoverStorageProtocolFrontierError::ArithmeticOverflow
              : SyncCoverStorageProtocolFrontierError::LimitExceeded);
    }
    const std::size_t certificateWork = certificate.demands.size() + 1;
    if (!budget.consume(certificateWork)) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::LimitExceeded);
    }
    SyncCoverDemandId previousDemand = 0;
    bool havePreviousDemand = false;
    for (SyncCoverDemandId demandId : certificate.demands) {
      const bool invalidDemandId =
          demandId >= graph.getDemands().size() ||
          (havePreviousDemand && demandId <= previousDemand);
      if (invalidDemandId) {
        return fail(statistics,
                    SyncCoverStorageProtocolFrontierError::InvalidGraph);
      }
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      const bool invalidDemand =
          demand.source >= graph.getNodes().size() ||
          demand.target >= graph.getNodes().size() || demand.distance != 0 ||
          graph.getNodes()[demand.source].resource !=
              certificate.sourceResource ||
          graph.getNodes()[demand.target].resource !=
              certificate.targetResource ||
          graph.getNodes()[demand.target].physicalAnchor != certificate.target;
      if (invalidDemand) {
        return fail(statistics,
                    SyncCoverStorageProtocolFrontierError::InvalidGraph);
      }
      certificates.push_back({demandId, certificate.sourceResource,
                              certificate.targetResource, certificate.id,
                              certificate.completionNode, certificate.target});
      previousDemand = demandId;
      havePreviousDemand = true;
      ++statistics.certificateDemandIncidences;
    }
  }
  if (!consumeSortWork(budget, certificates.size())) {
    return fail(statistics,
                SyncCoverStorageProtocolFrontierError::LimitExceeded);
  }
  std::sort(certificates.begin(), certificates.end(), certificateLess);
  for (std::size_t certificate = 1; certificate < certificates.size();
       ++certificate) {
    if (!budget.consume()) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::LimitExceeded);
    }
    if (!certificateLess(certificates[certificate - 1],
                         certificates[certificate])) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::InvalidGraph);
    }
  }
  const std::vector<SyncCoverStorageLifecycleComponent> &components =
      lifecycleIndex.getComponents();
  const std::vector<SyncCoverStorageProtocolAutomaton> &automata =
      automatonIndex.getAutomata();

  result.plans_.reserve(std::min(automata.size(), limits.maximumPlans));
  for (std::size_t automatonPosition = 0; automatonPosition < automata.size();
       ++automatonPosition) {
    const SyncCoverStorageProtocolAutomaton &automaton =
        automata[automatonPosition];
    if (automaton.id != automatonPosition) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::InvalidGraph);
    }
    SyncCoverStorageProtocolFrontierPlan plan;
    plan.id = result.plans_.size();
    plan.automaton = automaton.id;
    plan.group = automaton.group;
    plan.owningScope = automaton.owningScope;
    plan.laneCount = std::max<std::size_t>(automaton.maximumDistance, 1);
    std::vector<SyncCoverStorageProtocolFrontier> pending;
    std::vector<SyncCoverDemandId> automatonDemands;
    bool missingCompletion = false;
    bool sawReady = false;
    bool sawRecurrenceReuse = false;
    const auto appendPending =
        [&](SyncCoverStorageProtocolFrontier frontier) -> bool {
      const bool frontierCapacityAvailable =
          result.frontiers_.size() <= limits.maximumFrontiers &&
          pending.size() < limits.maximumFrontiers - result.frontiers_.size();
      const bool incidenceCapacityAvailable =
          statistics.planFrontierIncidences <=
              limits.maximumPlanFrontierIncidences &&
          pending.size() < limits.maximumPlanFrontierIncidences -
                               statistics.planFrontierIncidences;
      if (!frontierCapacityAvailable || !incidenceCapacityAvailable ||
          !budget.consume()) {
        return false;
      }
      frontier.id = pending.size();
      pending.push_back(std::move(frontier));
      return true;
    };

    for (std::size_t transferPosition = 0;
         transferPosition < automaton.transfers.size(); ++transferPosition) {
      if (statistics.transferInspections >= limits.maximumTransferInspections ||
          !budget.consume()) {
        return fail(statistics,
                    SyncCoverStorageProtocolFrontierError::LimitExceeded);
      }
      ++statistics.transferInspections;
      const SyncCoverStorageProtocolTransfer &transfer =
          automaton.transfers[transferPosition];
      bool validationLimitExceeded = false;
      const std::optional<TransferDescription> description = validateTransfer(
          graph, components, automaton, transfer, transferPosition, limits,
          statistics, budget, validationLimitExceeded);
      if (!description) {
        return fail(statistics,
                    validationLimitExceeded
                        ? SyncCoverStorageProtocolFrontierError::LimitExceeded
                        : SyncCoverStorageProtocolFrontierError::InvalidGraph);
      }
      const bool ready = (transfer.kinds & readyBit()) != 0;
      const bool recurrenceReuse =
          transfer.distance != 0 &&
          (transfer.kinds & (releaseBit() | exclusionBit())) != 0;
      sawReady |= ready;
      sawRecurrenceReuse |= recurrenceReuse;
      const bool crossResource =
          transfer.sourceResource != transfer.targetResource;
      automatonDemands.push_back(transfer.demand);
      if (recurrenceReuse && !crossResource) {
        ++plan.sameResourceRecurrenceReuses;
      }
      if (!crossResource || (!ready && !recurrenceReuse)) {
        continue;
      }
      const auto appendFrontier =
          [&](SyncCoverStorageProtocolFrontierKind kind,
              SyncCoverNodeId completionNode, SyncCoverNodeId targetNode,
              std::optional<SyncCoverCompletionCutFactId> completionCutFact,
              std::optional<SyncCoverTargetCompletionCertificateId> certificate)
          -> bool {
        SyncCoverStorageProtocolFrontier frontier;
        frontier.automaton = automaton.id;
        frontier.transfer = transfer.id;
        frontier.kind = kind;
        frontier.edge = transfer.edge;
        frontier.completionAnchor = {SyncCoverAnchorKind::AfterNode,
                                     completionNode, 0, 0};
        frontier.acquisitionAnchor = {SyncCoverAnchorKind::BeforeNode,
                                      targetNode, 0, 0};
        frontier.scope = transfer.scope;
        frontier.distance = transfer.distance;
        frontier.sourceResource = transfer.sourceResource;
        frontier.targetResource = transfer.targetResource;
        frontier.completionCutFact = completionCutFact;
        frontier.completionCertificate = certificate;
        return appendPending(std::move(frontier));
      };
      if (ready) {
        const EndpointFrontierLookup direct = findEndpointFrontier(
            graph, *description->edge, *description->source,
            *description->target, budget);
        if (direct.limitExceeded) {
          return fail(statistics,
                      SyncCoverStorageProtocolFrontierError::LimitExceeded);
        }
        if (direct.completionNode) {
          if (!appendFrontier(SyncCoverStorageProtocolFrontierKind::Ready,
                              *direct.completionNode, description->target->node,
                              std::nullopt, std::nullopt)) {
            return fail(statistics,
                        SyncCoverStorageProtocolFrontierError::LimitExceeded);
          }
          ++plan.readyFrontiers;
          ++plan.directFrontiers;
        } else {
          missingCompletion = true;
        }
        const CompletionCutFactEntry lookup{transfer.demand,
                                            transfer.sourceResource,
                                            transfer.targetResource, 0, 0};
        bool factLookupLimitExceeded = false;
        const std::optional<std::size_t> factPosition =
            lowerBoundCompletionCutFact(completionCutFacts, lookup, budget,
                                        factLookupLimitExceeded);
        if (factLookupLimitExceeded || !factPosition) {
          return fail(statistics,
                      SyncCoverStorageProtocolFrontierError::LimitExceeded);
        }
        auto fact = completionCutFacts.begin() + *factPosition;
        for (; fact != completionCutFacts.end() &&
               fact->demand == transfer.demand &&
               fact->sourceResource == transfer.sourceResource &&
               fact->targetResource == transfer.targetResource;
             ++fact) {
          if (!budget.consume()) {
            return fail(statistics,
                        SyncCoverStorageProtocolFrontierError::LimitExceeded);
          }
          if (description->edge->witness >=
                  graph.getStorageWitnesses().size() ||
              fact->fact >= graphCompletionCutFacts.size()) {
            return fail(statistics,
                        SyncCoverStorageProtocolFrontierError::InvalidGraph);
          }
          const SyncCoverStorageWitness &witness =
              graph.getStorageWitnesses()[description->edge->witness];
          if (witness.sourceAccess >= graph.getStorageAccesses().size()) {
            return fail(statistics,
                        SyncCoverStorageProtocolFrontierError::InvalidGraph);
          }
          const SyncCoverStorageDomainId domain =
              graph.getStorageAccesses()[witness.sourceAccess].domain;
          const std::optional<bool> admitted = containsStorageDomain(
              graphCompletionCutFacts[fact->fact].storageDomains, domain,
              budget);
          if (!admitted) {
            return fail(statistics,
                        SyncCoverStorageProtocolFrontierError::LimitExceeded);
          }
          if (!*admitted) {
            continue;
          }
          if (!appendFrontier(SyncCoverStorageProtocolFrontierKind::Ready,
                              fact->completionNode, description->target->node,
                              fact->fact, std::nullopt)) {
            return fail(statistics,
                        SyncCoverStorageProtocolFrontierError::LimitExceeded);
          }
          ++plan.readyFrontiers;
          ++plan.completionCutFactFrontiers;
        }
      }
      if (recurrenceReuse) {
        const EndpointFrontierLookup direct = findEndpointFrontier(
            graph, *description->edge, *description->source,
            *description->target, budget);
        if (direct.limitExceeded) {
          return fail(statistics,
                      SyncCoverStorageProtocolFrontierError::LimitExceeded);
        }
        if (direct.completionNode) {
          if (!appendFrontier(SyncCoverStorageProtocolFrontierKind::Reuse,
                              *direct.completionNode, description->target->node,
                              std::nullopt, std::nullopt)) {
            return fail(statistics,
                        SyncCoverStorageProtocolFrontierError::LimitExceeded);
          }
          ++plan.reuseFrontiers;
          ++plan.directFrontiers;
        } else {
          missingCompletion = true;
        }
      }
    }

    if (!consumeSortWork(budget, automatonDemands.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::LimitExceeded);
    }
    std::sort(automatonDemands.begin(), automatonDemands.end());
    automatonDemands.erase(
        std::unique(automatonDemands.begin(), automatonDemands.end()),
        automatonDemands.end());
    for (const SyncCoverTargetCompletionCertificate &certificate :
         graphCertificates) {
      if (!budget.consume()) {
        return fail(statistics,
                    SyncCoverStorageProtocolFrontierError::LimitExceeded);
      }
      const std::optional<bool> intersects =
          sortedIntersects(automatonDemands, certificate.demands, budget);
      if (!intersects) {
        return fail(statistics,
                    SyncCoverStorageProtocolFrontierError::LimitExceeded);
      }
      if (!*intersects) {
        continue;
      }
      SyncCoverStorageProtocolFrontier frontier;
      frontier.automaton = automaton.id;
      frontier.kind = SyncCoverStorageProtocolFrontierKind::Ready;
      frontier.completionAnchor = {SyncCoverAnchorKind::AfterNode,
                                   certificate.completionNode, 0, 0};
      frontier.acquisitionAnchor = {SyncCoverAnchorKind::BeforeNode,
                                    certificate.target, 0, 0};
      frontier.scope = automaton.owningScope;
      frontier.sourceResource = certificate.sourceResource;
      frontier.targetResource = certificate.targetResource;
      frontier.completionCutFact = std::nullopt;
      frontier.completionCertificate = certificate.id;
      if (!appendPending(std::move(frontier))) {
        return fail(statistics,
                    SyncCoverStorageProtocolFrontierError::LimitExceeded);
      }
      ++plan.readyFrontiers;
      ++plan.certificateFrontiers;
    }

    const bool missingReady = !sawReady || plan.readyFrontiers == 0;
    const bool missingReuse =
        !sawRecurrenceReuse ||
        (plan.reuseFrontiers == 0 && plan.sameResourceRecurrenceReuses == 0);
    statistics.missingCompletionFrontierAutomata += missingCompletion;
    if (missingReady || missingReuse) {
      ++statistics.ineligibleAutomata;
      statistics.missingReadyAutomata += missingReady;
      statistics.missingRecurrenceReuseAutomata += missingReuse;
      continue;
    }

    const bool frontierLimitReached =
        result.frontiers_.size() > limits.maximumFrontiers ||
        pending.size() > limits.maximumFrontiers - result.frontiers_.size();
    const bool incidenceLimitReached =
        statistics.planFrontierIncidences >
            limits.maximumPlanFrontierIncidences ||
        pending.size() > limits.maximumPlanFrontierIncidences -
                             statistics.planFrontierIncidences;
    const bool planLimitReached = result.plans_.size() >= limits.maximumPlans;
    const bool publicationWorkOverflows =
        pending.size() == std::numeric_limits<std::size_t>::max();
    if (publicationWorkOverflows) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::ArithmeticOverflow);
    }
    const std::size_t publicationWork = pending.size() + 1;
    if (frontierLimitReached || incidenceLimitReached || planLimitReached ||
        !budget.consume(publicationWork)) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::LimitExceeded);
    }
    plan.frontiers.reserve(pending.size());
    for (SyncCoverStorageProtocolFrontier &frontier : pending) {
      frontier.id = result.frontiers_.size();
      plan.frontiers.push_back(frontier.id);
      result.frontiers_.push_back(std::move(frontier));
    }
    const bool readyCountFits =
        checkedAdd(statistics.readyFrontiers, plan.readyFrontiers);
    const bool reuseCountFits =
        checkedAdd(statistics.reuseFrontiers, plan.reuseFrontiers);
    const bool directCountFits =
        checkedAdd(statistics.directFrontiers, plan.directFrontiers);
    const bool completionCutFactCountFits = checkedAdd(
        statistics.completionCutFactFrontiers, plan.completionCutFactFrontiers);
    const bool certificateCountFits =
        checkedAdd(statistics.certificateFrontiers, plan.certificateFrontiers);
    const bool sameResourceCountFits =
        checkedAdd(statistics.sameResourceRecurrenceReuses,
                   plan.sameResourceRecurrenceReuses);
    const bool incidenceCountFits =
        checkedAdd(statistics.planFrontierIncidences, plan.frontiers.size());
    if (!readyCountFits || !reuseCountFits || !directCountFits ||
        !completionCutFactCountFits || !certificateCountFits ||
        !sameResourceCountFits || !incidenceCountFits) {
      return fail(statistics,
                  SyncCoverStorageProtocolFrontierError::ArithmeticOverflow);
    }
    statistics.maximumPlanFrontiers =
        std::max(statistics.maximumPlanFrontiers, plan.frontiers.size());
    result.plans_.push_back(std::move(plan));
    ++statistics.eligibleAutomata;
  }

  statistics.plans = result.plans_.size();
  statistics.frontiers = result.frontiers_.size();
  result.statistics_ = statistics;
  return result;
}
