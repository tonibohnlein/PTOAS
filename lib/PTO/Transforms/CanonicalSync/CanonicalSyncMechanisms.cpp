// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kHardwareEventIdCount = 8;

using EventDomainKey = std::pair<std::uint32_t, std::uint32_t>;

struct BarrierFallbackGroup {
  bool broad = false;
  std::vector<SyncCoverDemandId> demands;
};

SyncCoverEdge getDemandEdge(const SyncCoverDemand &demand) {
  return {
      demand.source,     demand.target,   SyncCoverEdgeKind::CompletionSupply,
      demand.scope,      demand.distance, demand.sourceGuard,
      demand.targetGuard};
}

std::vector<SyncCoverDemandId> getActiveDemands(const SyncCoverGraph &graph) {
  std::vector<SyncCoverDemandId> demands(graph.getDemands().size());
  std::iota(demands.begin(), demands.end(), 0);
  return demands;
}

std::vector<std::uint32_t> getIssueResources(const SyncCoverGraph &graph) {
  std::vector<std::uint32_t> resources;
  resources.reserve(graph.getNodes().size());
  for (const SyncCoverNode &node : graph.getNodes()) {
    resources.push_back(node.resource);
  }
  llvm::sort(resources);
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
  return resources;
}

std::vector<unsigned> getReservations(const CanonicalSyncProgram &program,
                                      const EventDomainKey &key) {
  const auto reservation = program.getEventReservations().find(key);
  return reservation == program.getEventReservations().end()
             ? std::vector<unsigned>{}
             : reservation->second;
}

bool hasAvailableEventId(const CanonicalSyncProgram &program,
                         const EventDomainKey &key, unsigned budget) {
  const std::vector<unsigned> reserved = getReservations(program, key);
  const std::size_t unavailable = static_cast<std::size_t>(
      std::count_if(reserved.begin(), reserved.end(),
                    [&](unsigned eventId) { return eventId < budget; }));
  return unavailable < budget;
}

bool canUseDirectEvent(const SyncCoverGraph &graph,
                       const SyncCoverDemand &demand) {
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const SyncCoverEdge edge = getDemandEdge(demand);
  return demand.distance == 0 && source.resource != target.resource &&
         syncCoverNodeCanProduceCompletion(graph, source.id, target.resource) &&
         syncCoverEndpointsCoExecute(graph, edge);
}

CanonicalSyncMechanismDescriptor
makeDirectEvent(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                CanonicalSyncEventDomainId domain) {
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Event;
  descriptor.eventUses.push_back({domain, 1, std::nullopt});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventSet,
       source.resource,
       {SyncCoverAnchorKind::AfterNode, source.id, 0, 0},
       0,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventWait,
       target.resource,
       {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
       0,
       0,
       {}});
  CanonicalSyncSupplyBinding binding;
  binding.edge = getDemandEdge(demand);
  binding.eventUse = 0;
  descriptor.supplies.push_back(std::move(binding));
  return descriptor;
}

CanonicalSyncMechanismDescriptor
makeBarrier(const SyncCoverGraph &graph,
            const std::vector<std::uint32_t> &allResources,
            ArrayRef<SyncCoverDemandId> demands, bool broad) {
  const SyncCoverDemand &first = graph.getDemands()[demands.front()];
  const SyncCoverNode &target = graph.getNodes()[first.target];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Barrier;
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::Barrier,
       target.resource,
       {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
       std::nullopt,
       0,
       broad ? allResources : std::vector<std::uint32_t>{target.resource},
       broad ? CanonicalSyncBarrierKind::All
             : CanonicalSyncBarrierKind::Targeted});
  for (SyncCoverDemandId demandId : demands) {
    CanonicalSyncSupplyBinding binding;
    binding.edge = getDemandEdge(graph.getDemands()[demandId]);
    binding.barrierAction = 0;
    descriptor.supplies.push_back(std::move(binding));
  }
  return descriptor;
}

LogicalResult addEventDomains(
    const CanonicalSyncProgram &program, unsigned budget,
    CanonicalSyncPatternProblem &problem, const SyncCoverDemandSet &baseline,
    std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds) {
  const SyncCoverGraph &graph = program.getGraph();
  std::set<EventDomainKey> keys;
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const bool needsMechanism = !baseline.contains(demandId);
    const bool supportsDirectEvent = canUseDirectEvent(graph, demand);
    if (!needsMechanism || !supportsDirectEvent) {
      continue;
    }
    const EventDomainKey key{graph.getNodes()[demand.source].resource,
                             graph.getNodes()[demand.target].resource};
    if (hasAvailableEventId(program, key, budget)) {
      keys.insert(key);
    }
  }
  for (const EventDomainKey &key : keys) {
    const CanonicalSyncEventDomainId id = domainIds.size();
    CanonicalSyncEventDomain domain{id, key.first, key.second, budget,
                                    getReservations(program, key)};
    const CanonicalSyncProblemResult added =
        problem.addEventDomain(std::move(domain));
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      break;
    }
    if (!added) {
      return program.getFunction().emitError(
          "cannot add canonical sync event domain");
    }
    domainIds.emplace(key, id);
  }
  return success();
}

LogicalResult addDirectEvents(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const SyncCoverDemandSet &baseline,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds) {
  const SyncCoverGraph &graph = program.getGraph();
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const bool needsMechanism = !baseline.contains(demandId);
    const bool supportsDirectEvent = canUseDirectEvent(graph, demand);
    if (!needsMechanism || !supportsDirectEvent) {
      continue;
    }
    const EventDomainKey key{graph.getNodes()[demand.source].resource,
                             graph.getNodes()[demand.target].resource};
    const auto domain = domainIds.find(key);
    if (domain == domainIds.end()) {
      continue;
    }
    const CanonicalSyncProblemResult added =
        problem.internMechanism(makeDirectEvent(graph, demand, domain->second));
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      return success();
    }
    if (!added) {
      return program.getFunction().emitError(
          "cannot add canonical sync direct event");
    }
  }
  return success();
}

LogicalResult addBarrierFallbacks(const CanonicalSyncProgram &program,
                                  CanonicalSyncPatternProblem &problem,
                                  const SyncCoverDemandSet &baseline) {
  const SyncCoverGraph &graph = program.getGraph();
  const std::vector<std::uint32_t> allResources = getIssueResources(graph);
  std::map<SyncCoverNodeId, BarrierFallbackGroup> groups;
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (baseline.contains(demandId)) {
      continue;
    }
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    BarrierFallbackGroup &group = groups[target.id];
    group.broad = group.broad || source.resource != target.resource;
    group.demands.push_back(demandId);
  }
  for (const auto &[target, group] : groups) {
    (void)target;
    const bool hasDemands = !group.demands.empty();
    const bool added =
        hasDemands && problem.internMechanism(makeBarrier(
                          graph, allResources, group.demands, group.broad));
    if (!added) {
      return program.getFunction().emitError(
          "cannot add canonical sync barrier fallback");
    }
  }
  return success();
}

} // namespace

FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>>
mlir::pto::buildCanonicalSyncSingletonProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncBuildOptions &options) {
  if (options.eventIdBudget == 0 ||
      options.eventIdBudget > kHardwareEventIdCount) {
    program.getFunction().emitError(
        "canonical sync event-id budget must be in [1, 8]");
    return failure();
  }
  auto problem = std::make_unique<CanonicalSyncPatternProblem>(
      program.getGraph(), getActiveDemands(program.getGraph()),
      options.problemLimits, options.expansionLimits);
  const SyncCoverCoverageResult baseline = computeSyncCoverCoverage(
      program.getGraph(), problem->getExpansion(), {}, problem->getDemands());
  if (!baseline) {
    program.getFunction().emitError(
        "cannot compute canonical sync fixed coverage");
    return failure();
  }
  std::map<EventDomainKey, CanonicalSyncEventDomainId> domainIds;
  const bool failedBuild =
      failed(addBarrierFallbacks(program, *problem, baseline.covered)) ||
      failed(addEventDomains(program, options.eventIdBudget, *problem,
                             baseline.covered, domainIds)) ||
      failed(addDirectEvents(program, *problem, baseline.covered, domainIds));
  if (failedBuild) {
    return failure();
  }
  const CanonicalSyncProblemResult frozen = problem->freeze();
  if (!frozen) {
    program.getFunction().emitError()
        << "cannot freeze canonical sync singleton problem, error="
        << static_cast<unsigned>(frozen.error);
    return failure();
  }
  return problem;
}
