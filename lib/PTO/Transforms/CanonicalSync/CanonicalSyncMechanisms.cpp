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
#include <iterator>
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
  std::vector<SyncCoverDemandId> demands;
};

struct DirectEventRecord {
  SyncCoverDemandId demand = 0;
  CanonicalSyncMechanismId mechanism = 0;
  CanonicalSyncEventDomainId domain = 0;
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

std::size_t availableEventIds(const CanonicalSyncEventDomain &domain) {
  const std::size_t reserved = static_cast<std::size_t>(
      std::count_if(domain.reservedIds.begin(), domain.reservedIds.end(),
                    [&](unsigned eventId) { return eventId < domain.budget; }));
  return domain.budget - std::min<std::size_t>(domain.budget, reserved);
}

bool canUseDistanceZeroEvent(const SyncCoverGraph &graph,
                             const SyncCoverDemand &demand) {
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const SyncCoverEdge edge = getDemandEdge(demand);
  return demand.distance == 0 && source.resource != target.resource &&
         syncCoverNodeCanProduceCompletion(graph, source.id, target.resource) &&
         syncCoverEndpointsCoExecute(graph, edge);
}

bool canUseRecurrenceEvent(const SyncCoverGraph &graph,
                           const SyncCoverDemand &demand) {
  const bool invalid = demand.distance == 0 ||
                       demand.scope >= graph.getScopes().size() ||
                       !graph.getScopes()[demand.scope].isLoop ||
                       !graph.getScopes()[demand.scope].timeline ||
                       !demand.sourceGuard.literals.empty() ||
                       !demand.targetGuard.literals.empty();
  if (invalid) {
    return false;
  }
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  return source.resource != target.resource &&
         graph.scopeMustExecuteWithin(demand.scope, source.scope) &&
         graph.scopeMustExecuteWithin(demand.scope, target.scope) &&
         syncCoverNodeCanProduceCompletion(graph, source.id, target.resource);
}

bool isReleaseStyleRecurrence(const SyncCoverGraph &graph,
                              const SyncCoverDemand &demand) {
  return graph.getNodes()[demand.target].order <
         graph.getNodes()[demand.source].order;
}

bool canUsePreciseEvent(const SyncCoverGraph &graph,
                        const SyncCoverDemand &demand) {
  return canUseDistanceZeroEvent(graph, demand) ||
         canUseRecurrenceEvent(graph, demand);
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
makeRecurrenceEvent(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                    CanonicalSyncEventDomainId domain) {
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  if (!isReleaseStyleRecurrence(graph, demand)) {
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
    SyncCoverEdge edge = getDemandEdge(demand);
    edge.distance = 0;
    edge.scope = *graph.getLowestCommonScope(source.scope, target.scope);
    CanonicalSyncSupplyBinding binding;
    binding.edge = std::move(edge);
    binding.eventUse = 0;
    descriptor.supplies.push_back(std::move(binding));
    return descriptor;
  }

  const std::size_t width = demand.distance;
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Protocol;
  descriptor.eventUses.push_back({domain, width, demand.scope});
  for (std::size_t lane = 0; lane < width; ++lane) {
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::EventSet,
         source.resource,
         {SyncCoverAnchorKind::ScopeEntry, 0, demand.scope},
         0,
         lane,
         {}});
  }
  const std::size_t consumeAction = descriptor.actions.size();
  CanonicalSyncAction bodyWait{CanonicalSyncActionKind::EventWait,
                               target.resource,
                               {SyncCoverAnchorKind::BeforeNode, target.id, 0},
                               0,
                               0,
                               {}};
  if (width > 1) {
    bodyWait.eventLaneKind = CanonicalSyncEventLaneKind::LoopIterationModulo;
    bodyWait.eventLaneScope = demand.scope;
  }
  descriptor.actions.push_back(std::move(bodyWait));
  const std::size_t produceAction = descriptor.actions.size();
  CanonicalSyncAction bodySet{CanonicalSyncActionKind::EventSet,
                              source.resource,
                              {SyncCoverAnchorKind::AfterNode, source.id, 0},
                              0,
                              0,
                              {}};
  if (width > 1) {
    bodySet.eventLaneKind = CanonicalSyncEventLaneKind::LoopIterationModulo;
    bodySet.eventLaneScope = demand.scope;
  }
  descriptor.actions.push_back(std::move(bodySet));
  for (std::size_t lane = 0; lane < width; ++lane) {
    descriptor.actions.push_back(
        {CanonicalSyncActionKind::EventWait,
         target.resource,
         {SyncCoverAnchorKind::ScopeExit, 0, demand.scope},
         0,
         lane,
         {}});
  }
  CanonicalSyncSupplyBinding binding;
  binding.edge = getDemandEdge(demand);
  binding.eventUse = 0;
  binding.produceAction = produceAction;
  binding.consumeAction = consumeAction;
  binding.proof = CanonicalSyncSupplyProof::VerifiedProtocol;
  binding.completionExport = CanonicalSyncSupplyExport::ScopeExitAfterDrain;
  descriptor.supplies.push_back(std::move(binding));
  return descriptor;
}

bool verifyRecurrenceEvent(const SyncCoverGraph &graph,
                           const SyncCoverDemand &demand,
                           CanonicalSyncEventDomainId domain,
                           const CanonicalSyncMechanismDescriptor &descriptor) {
  const bool releaseStyle = isReleaseStyleRecurrence(graph, demand);
  const bool invalid =
      !canUseRecurrenceEvent(graph, demand) ||
      descriptor.kind != (releaseStyle ? CanonicalSyncMechanismKind::Protocol
                                       : CanonicalSyncMechanismKind::Event) ||
      descriptor.eventUses.size() != 1 || descriptor.supplies.size() != 1;
  if (invalid) {
    return false;
  }
  const SyncCoverNode &source = graph.getNodes()[demand.source];
  const SyncCoverNode &target = graph.getNodes()[demand.target];
  const CanonicalSyncEventUse &use = descriptor.eventUses.front();
  const CanonicalSyncSupplyBinding &binding = descriptor.supplies.front();
  const auto actionMatches =
      [](const CanonicalSyncAction &action, CanonicalSyncActionKind kind,
         std::uint32_t resource, SyncCoverAnchorKind anchorKind,
         SyncCoverNodeId node, SyncCoverScopeId scope, std::size_t lane,
         CanonicalSyncEventLaneKind laneKind,
         std::optional<SyncCoverScopeId> laneScope) {
        return action.kind == kind && action.resource == resource &&
               action.anchor.kind == anchorKind && action.anchor.node == node &&
               action.anchor.scope == scope && action.eventUse == 0 &&
               action.eventLane == lane && action.drainedResources.empty() &&
               action.guard == CanonicalSyncActionGuardKind::None &&
               !action.guardScope && action.eventLaneKind == laneKind &&
               action.eventLaneScope == laneScope;
      };
  if (!releaseStyle) {
    const std::optional<SyncCoverScopeId> common =
        graph.getLowestCommonScope(source.scope, target.scope);
    return common && descriptor.actions.size() == 2 && use.domain == domain &&
           use.width == 1 && !use.recurrenceScope && !use.lifetimeScope &&
           binding.edge.source == demand.source &&
           binding.edge.target == demand.target &&
           binding.edge.scope == *common && binding.edge.distance == 0 &&
           binding.eventUse == 0 && !binding.barrierAction &&
           !binding.produceAction && !binding.consumeAction &&
           binding.proof == CanonicalSyncSupplyProof::DirectAction &&
           binding.completionExport == CanonicalSyncSupplyExport::LocalTarget &&
           actionMatches(descriptor.actions[0],
                         CanonicalSyncActionKind::EventSet, source.resource,
                         SyncCoverAnchorKind::AfterNode, source.id, 0, 0,
                         CanonicalSyncEventLaneKind::Static, std::nullopt) &&
           actionMatches(descriptor.actions[1],
                         CanonicalSyncActionKind::EventWait, target.resource,
                         SyncCoverAnchorKind::BeforeNode, target.id, 0, 0,
                         CanonicalSyncEventLaneKind::Static, std::nullopt);
  }

  const std::size_t width = demand.distance;
  const std::size_t consumeAction = width;
  const std::size_t produceAction = width + 1;
  const bool correctUse = use.domain == domain && use.width == width &&
                          use.recurrenceScope == demand.scope &&
                          !use.lifetimeScope;
  const bool correctSupply =
      binding.edge.source == demand.source &&
      binding.edge.target == demand.target &&
      binding.edge.scope == demand.scope &&
      binding.edge.distance == demand.distance && binding.eventUse == 0 &&
      !binding.barrierAction && binding.produceAction == produceAction &&
      binding.consumeAction == consumeAction &&
      binding.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
      binding.completionExport ==
          CanonicalSyncSupplyExport::ScopeExitAfterDrain;
  if (!correctUse || !correctSupply ||
      descriptor.actions.size() != width * 2 + 2) {
    return false;
  }
  for (std::size_t lane = 0; lane < width; ++lane) {
    if (!actionMatches(descriptor.actions[lane],
                       CanonicalSyncActionKind::EventSet, source.resource,
                       SyncCoverAnchorKind::ScopeEntry, 0, demand.scope, lane,
                       CanonicalSyncEventLaneKind::Static, std::nullopt) ||
        !actionMatches(descriptor.actions[width + 2 + lane],
                       CanonicalSyncActionKind::EventWait, target.resource,
                       SyncCoverAnchorKind::ScopeExit, 0, demand.scope, lane,
                       CanonicalSyncEventLaneKind::Static, std::nullopt)) {
      return false;
    }
  }
  const CanonicalSyncEventLaneKind bodyLaneKind =
      width > 1 ? CanonicalSyncEventLaneKind::LoopIterationModulo
                : CanonicalSyncEventLaneKind::Static;
  const std::optional<SyncCoverScopeId> bodyLaneScope =
      width > 1 ? std::optional<SyncCoverScopeId>(demand.scope) : std::nullopt;
  return actionMatches(descriptor.actions[consumeAction],
                       CanonicalSyncActionKind::EventWait, target.resource,
                       SyncCoverAnchorKind::BeforeNode, target.id, 0, 0,
                       bodyLaneKind, bodyLaneScope) &&
         actionMatches(descriptor.actions[produceAction],
                       CanonicalSyncActionKind::EventSet, source.resource,
                       SyncCoverAnchorKind::AfterNode, source.id, 0, 0,
                       bodyLaneKind, bodyLaneScope);
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
    const bool supportsDirectEvent = canUsePreciseEvent(graph, demand);
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
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
    std::vector<DirectEventRecord> &directEvents) {
  const SyncCoverGraph &graph = program.getGraph();
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const bool needsMechanism = !baseline.contains(demandId);
    const bool supportsDirectEvent = canUsePreciseEvent(graph, demand);
    if (!needsMechanism || !supportsDirectEvent) {
      continue;
    }
    const EventDomainKey key{graph.getNodes()[demand.source].resource,
                             graph.getNodes()[demand.target].resource};
    const auto domain = domainIds.find(key);
    if (domain == domainIds.end()) {
      continue;
    }
    CanonicalSyncProblemResult added;
    if (demand.distance == 0) {
      added = problem.internMechanism(
          makeDirectEvent(graph, demand, domain->second));
    } else {
      const CanonicalSyncEventDomain &eventDomain =
          problem.getDomains()[domain->second];
      const bool releaseStyle = isReleaseStyleRecurrence(graph, demand);
      if (releaseStyle && demand.distance > availableEventIds(eventDomain)) {
        continue;
      }
      CanonicalSyncMechanismDescriptor descriptor =
          makeRecurrenceEvent(graph, demand, domain->second);
      if (descriptor.kind == CanonicalSyncMechanismKind::Protocol) {
        added = problem.internVerifiedProtocol(
            std::move(descriptor),
            [&](const CanonicalSyncMechanismDescriptor &actual) {
              return verifyRecurrenceEvent(graph, demand, domain->second,
                                           actual);
            });
      } else if (verifyRecurrenceEvent(graph, demand, domain->second,
                                       descriptor)) {
        added = problem.internMechanism(std::move(descriptor));
      } else {
        return program.getFunction().emitError(
            "cannot verify canonical sync forward recurrence event");
      }
    }
    if (added.error == CanonicalSyncProblemError::LimitExceeded) {
      return success();
    }
    if (!added || !added.index) {
      return program.getFunction().emitError(
          "cannot add canonical sync direct event");
    }
    if (demand.distance == 0) {
      directEvents.push_back({demandId, *added.index, domain->second});
    }
  }
  return success();
}

LogicalResult addTargetedBarriers(const CanonicalSyncProgram &program,
                                  CanonicalSyncPatternProblem &problem,
                                  const SyncCoverDemandSet &baseline) {
  const SyncCoverGraph &graph = program.getGraph();
  const std::vector<std::uint32_t> allResources = getIssueResources(graph);
  std::map<SyncCoverNodeId, BarrierFallbackGroup> targetedGroups;
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (baseline.contains(demandId)) {
      continue;
    }
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    if (source.resource == target.resource) {
      targetedGroups[target.id].demands.push_back(demandId);
    }
  }
  const auto addGroups = [&](const auto &groups, bool broad) -> LogicalResult {
    for (const auto &[target, group] : groups) {
      (void)target;
      const bool hasDemands = !group.demands.empty();
      const bool added =
          hasDemands && problem.internMechanism(makeBarrier(
                            graph, allResources, group.demands, broad));
      if (!added) {
        return failure();
      }
    }
    return success();
  };
  if (failed(addGroups(targetedGroups, false))) {
    return program.getFunction().emitError(
        "cannot add canonical sync targeted barrier");
  }
  return success();
}

LogicalResult addPipeAllBackstop(const CanonicalSyncProgram &program,
                                 CanonicalSyncPatternProblem &problem,
                                 const SyncCoverDemandSet &baseline) {
  const SyncCoverGraph &graph = program.getGraph();
  const std::vector<std::uint32_t> allResources = getIssueResources(graph);
  std::map<SyncCoverNodeId, BarrierFallbackGroup> groups;
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    if (!baseline.contains(demandId)) {
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      groups[demand.target].demands.push_back(demandId);
    }
  }
  for (const auto &[target, group] : groups) {
    (void)target;
    if (!problem.internMechanism(
            makeBarrier(graph, allResources, group.demands, true))) {
      return program.getFunction().emitError(
          "cannot add canonical sync localized PIPE_ALL backstop");
    }
  }
  return success();
}

CanonicalSyncMechanismDescriptor makeRepairBarrier(const SyncCoverGraph &graph,
                                                   const SyncCoverEdge &edge) {
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Barrier;
  const SyncCoverNode &target = graph.getNodes()[edge.target];
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::Barrier,
       target.resource,
       {SyncCoverAnchorKind::BeforeNode, target.id, 0, 0},
       std::nullopt,
       0,
       {target.resource},
       CanonicalSyncBarrierKind::Targeted});
  CanonicalSyncSupplyBinding supply;
  supply.edge = edge;
  supply.barrierAction = 0;
  descriptor.supplies.push_back(std::move(supply));
  return descriptor;
}

CanonicalSyncMechanismDescriptor
makeRepairEvent(const SyncCoverGraph &graph, const SyncCoverEdge &edge,
                CanonicalSyncEventDomainId domain) {
  const SyncCoverNode &source = graph.getNodes()[edge.source];
  const SyncCoverNode &target = graph.getNodes()[edge.target];
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
  CanonicalSyncSupplyBinding supply;
  supply.edge = edge;
  supply.eventUse = 0;
  descriptor.supplies.push_back(std::move(supply));
  return descriptor;
}

struct RepairFrontierProposal {
  SyncCoverEdge barrier;
  SyncCoverEdge event;
  CanonicalSyncEventDomainId domain = 0;
};

bool frontierProposalLess(const RepairFrontierProposal &left,
                          const RepairFrontierProposal &right) {
  return std::tie(left.domain, left.barrier.source, left.barrier.target,
                  left.event.source, left.event.target) <
         std::tie(right.domain, right.barrier.source, right.barrier.target,
                  right.event.source, right.event.target);
}

bool frontierProposalEqual(const RepairFrontierProposal &left,
                           const RepairFrontierProposal &right) {
  return !frontierProposalLess(left, right) &&
         !frontierProposalLess(right, left);
}

LogicalResult
addRepairFrontierPatterns(const CanonicalSyncProgram &program,
                          CanonicalSyncPatternProblem &problem,
                          ArrayRef<DirectEventRecord> directEvents,
                          ArrayRef<CanonicalSyncMechanismId> conflictCore) {
  constexpr std::size_t kMaximumFrontierProposals = 4096;
  const SyncCoverGraph &graph = program.getGraph();
  std::vector<CanonicalSyncMechanismId> sortedCore(conflictCore.begin(),
                                                   conflictCore.end());
  llvm::sort(sortedCore);
  sortedCore.erase(std::unique(sortedCore.begin(), sortedCore.end()),
                   sortedCore.end());
  std::vector<DirectEventRecord> liveEvents;
  llvm::copy_if(directEvents, std::back_inserter(liveEvents),
                [&](const DirectEventRecord &event) {
                  return std::binary_search(sortedCore.begin(),
                                            sortedCore.end(), event.mechanism);
                });
  std::vector<RepairFrontierProposal> proposals;
  for (std::size_t first = 0; first < liveEvents.size(); ++first) {
    const SyncCoverDemand &firstDemand =
        graph.getDemands()[liveEvents[first].demand];
    for (std::size_t second = first + 1; second < liveEvents.size(); ++second) {
      const SyncCoverDemand &secondDemand =
          graph.getDemands()[liveEvents[second].demand];
      const bool compatible =
          liveEvents[first].domain == liveEvents[second].domain &&
          firstDemand.scope == secondDemand.scope &&
          firstDemand.sourceGuard.literals.empty() &&
          firstDemand.targetGuard.literals.empty() &&
          secondDemand.sourceGuard.literals.empty() &&
          secondDemand.targetGuard.literals.empty();
      if (!compatible) {
        continue;
      }
      const SyncCoverNode &firstSource = graph.getNodes()[firstDemand.source];
      const SyncCoverNode &secondSource = graph.getNodes()[secondDemand.source];
      const SyncCoverNode &firstTarget = graph.getNodes()[firstDemand.target];
      const SyncCoverNode &secondTarget = graph.getNodes()[secondDemand.target];
      const SyncCoverNode &earlySource =
          firstSource.order < secondSource.order ? firstSource : secondSource;
      const SyncCoverNode &lateSource =
          firstSource.order < secondSource.order ? secondSource : firstSource;
      const SyncCoverNode &earlyTarget =
          firstTarget.order < secondTarget.order ? firstTarget : secondTarget;
      const bool distinctSources = earlySource.id != lateSource.id;
      const bool forwardFrontier = lateSource.order < earlyTarget.order;
      if (!distinctSources || !forwardFrontier) {
        continue;
      }
      SyncCoverEdge barrier{earlySource.id,
                            lateSource.id,
                            SyncCoverEdgeKind::CompletionSupply,
                            firstDemand.scope,
                            0,
                            {},
                            {}};
      SyncCoverEdge event{lateSource.id,
                          earlyTarget.id,
                          SyncCoverEdgeKind::CompletionSupply,
                          firstDemand.scope,
                          0,
                          {},
                          {}};
      const bool invalidFrontier =
          !syncCoverEndpointsCoExecute(graph, barrier) ||
          !syncCoverEndpointsCoExecute(graph, event) ||
          !syncCoverNodeCanProduceCompletion(graph, lateSource.id,
                                             earlyTarget.resource);
      if (invalidFrontier) {
        continue;
      }
      proposals.push_back({barrier, event, liveEvents[first].domain});
    }
  }
  llvm::sort(proposals, frontierProposalLess);
  proposals.erase(
      std::unique(proposals.begin(), proposals.end(), frontierProposalEqual),
      proposals.end());
  const bool proposalLimitExceeded =
      proposals.size() > kMaximumFrontierProposals;
  if (proposalLimitExceeded) {
    problem.markPatternGenerationTruncated();
    return success();
  }
  for (const RepairFrontierProposal &proposal : proposals) {
    const CanonicalSyncProblemResult barrier =
        problem.internMechanism(makeRepairBarrier(graph, proposal.barrier));
    const CanonicalSyncProblemResult event = problem.internMechanism(
        makeRepairEvent(graph, proposal.event, proposal.domain));
    if (!barrier || !event || !barrier.index || !event.index) {
      return program.getFunction().emitError(
          "cannot add canonical sync repair-frontier mechanisms");
    }
    const CanonicalSyncProblemResult pattern = addCanonicalSyncFeasiblePattern(
        problem, {CanonicalSyncPatternKind::RepairFrontier,
                  {*barrier.index, *event.index}});
    if (!pattern) {
      return program.getFunction().emitError(
          "cannot add canonical sync repair-frontier pattern");
    }
  }
  return success();
}

LogicalResult addDirectPairPatterns(const CanonicalSyncProgram &program,
                                    CanonicalSyncPatternProblem &problem) {
  const CanonicalSyncProblemResult pairs =
      addCanonicalSyncDirectPairPatterns(problem);
  if (!pairs) {
    return program.getFunction().emitError(
        "cannot add canonical sync direct-pair patterns");
  }
  return success();
}

enum class CandidateCatalogKind : std::uint8_t {
  Precise,
  ConflictCoreRepair,
  LocalizedPipeAll,
};

CanonicalSyncProblemBuildResult
buildCandidateCatalog(const CanonicalSyncProgram &program,
                      const CanonicalSyncBuildOptions &options,
                      CandidateCatalogKind kind,
                      ArrayRef<CanonicalSyncMechanismId> conflictCore = {}) {
  if (options.eventIdBudget == 0 ||
      options.eventIdBudget > kHardwareEventIdCount) {
    program.getFunction().emitError(
        "canonical sync event-id budget must be in [1, 8]");
    return {nullptr, {CanonicalSyncProblemError::InvalidDomain, std::nullopt}};
  }
  auto problem = std::make_unique<CanonicalSyncPatternProblem>(
      program.getGraph(), getActiveDemands(program.getGraph()),
      options.problemLimits, options.expansionLimits);
  const SyncCoverCoverageResult baseline = computeSyncCoverCoverage(
      program.getGraph(), problem->getExpansion(), {}, problem->getDemands());
  if (!baseline) {
    program.getFunction().emitError(
        "cannot compute canonical sync fixed coverage");
    return {nullptr,
            {CanonicalSyncProblemError::CoverageFailure, std::nullopt}};
  }

  bool failedBuild = false;
  if (kind == CandidateCatalogKind::LocalizedPipeAll) {
    failedBuild =
        failed(addPipeAllBackstop(program, *problem, baseline.covered));
  } else {
    std::map<EventDomainKey, CanonicalSyncEventDomainId> domainIds;
    std::vector<DirectEventRecord> directEvents;
    failedBuild =
        failed(addTargetedBarriers(program, *problem, baseline.covered)) ||
        failed(addEventDomains(program, options.eventIdBudget, *problem,
                               baseline.covered, domainIds)) ||
        failed(addDirectEvents(program, *problem, baseline.covered, domainIds,
                               directEvents)) ||
        (options.patterns.enableDirectPairs &&
         failed(addDirectPairPatterns(program, *problem))) ||
        (kind == CandidateCatalogKind::ConflictCoreRepair &&
         failed(addRepairFrontierPatterns(program, *problem, directEvents,
                                          conflictCore)));
  }
  if (failedBuild) {
    return {nullptr,
            {CanonicalSyncProblemError::InvalidMechanism, std::nullopt}};
  }
  if (problem->wasPatternGenerationTruncated()) {
    program.getFunction().emitRemark(
        "canonical sync pattern generation reached its bounded proposal "
        "limit; singleton candidates remain available");
  }
  const CanonicalSyncProblemResult frozen = problem->freeze();
  return {std::move(problem), frozen};
}

} // namespace

CanonicalSyncProblemBuildResult mlir::pto::buildCanonicalSyncPreciseProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncBuildOptions &options) {
  return buildCandidateCatalog(program, options, CandidateCatalogKind::Precise);
}

CanonicalSyncProblemBuildResult mlir::pto::buildCanonicalSyncRepairProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncBuildOptions &options,
    const std::vector<CanonicalSyncMechanismId> &conflictCore) {
  return buildCandidateCatalog(
      program, options, CandidateCatalogKind::ConflictCoreRepair, conflictCore);
}

CanonicalSyncProblemBuildResult mlir::pto::buildCanonicalSyncPipeAllProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncBuildOptions &options) {
  return buildCandidateCatalog(program, options,
                               CandidateCatalogKind::LocalizedPipeAll);
}

FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>>
mlir::pto::buildCanonicalSyncSingletonProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncBuildOptions &options) {
  CanonicalSyncProblemBuildResult built =
      buildCanonicalSyncPreciseProblem(program, options);
  if (!built) {
    program.getFunction().emitError()
        << "cannot freeze canonical sync precise problem, error="
        << static_cast<unsigned>(built.status.error);
    return failure();
  }
  return std::move(built.problem);
}
