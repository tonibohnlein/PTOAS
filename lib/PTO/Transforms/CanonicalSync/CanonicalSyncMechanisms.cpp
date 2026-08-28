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
#include "PTO/Transforms/CanonicalSync/CanonicalSyncLifecycle.h"
#include "PTO/Transforms/CanonicalSync/CanonicalSyncOwnership.h"

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
  std::vector<SyncCoverDemandId> demands;
};

struct DirectEventRecord {
  SyncCoverDemandId demand = 0;
  CanonicalSyncMechanismId mechanism = 0;
  CanonicalSyncEventDomainId domain = 0;
};

struct OwnershipMechanismRecord {
  const CanonicalSyncOwnershipCycle *cycle = nullptr;
  CanonicalSyncMechanismId mechanism = 0;
};

struct L0OwnershipMechanismRecord {
  const CanonicalSyncOwnershipCycle *cycle = nullptr;
  CanonicalSyncMechanismId mechanism = 0;
};

struct OwnershipPipelineMembers {
  std::vector<OwnershipMechanismRecord> stableL1;
  std::vector<OwnershipMechanismRecord> alternatingL1;
  std::vector<L0OwnershipMechanismRecord> l0Operands;
};

std::optional<SyncCoverScopeId>
getEnclosingLoopScope(const SyncCoverGraph &graph, SyncCoverScopeId scope) {
  if (scope >= graph.getScopes().size()) {
    return std::nullopt;
  }
  const SyncCoverScopeId parent = graph.getScopes()[scope].parent;
  const bool parentIsLoop =
      parent < graph.getScopes().size() && graph.getScopes()[parent].isLoop;
  if (parentIsLoop) {
    return parent;
  }
  return std::nullopt;
}

bool ownershipSlotsAreDisjoint(const CanonicalSyncOwnershipCycle &first,
                               const CanonicalSyncOwnershipCycle &second) {
  return llvm::none_of(first.lanes, [&](const auto &firstLane) {
    return llvm::any_of(firstLane.slots, [&](const auto &firstSlot) {
      return llvm::any_of(second.lanes, [&](const auto &secondLane) {
        return llvm::any_of(secondLane.slots, [&](const auto &secondSlot) {
          return firstSlot.domain == secondSlot.domain &&
                 firstSlot.extent.begin < secondSlot.extent.end &&
                 secondSlot.extent.begin < firstSlot.extent.end;
        });
      });
    });
  });
}

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
      descriptor.selectionTier != CanonicalSyncSelectionTier::Precise ||
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
      binding.proof == CanonicalSyncSupplyProof::VerifiedProtocol;
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
  descriptor.selectionTier = broad ? CanonicalSyncSelectionTier::PipeAllRescue
                                   : CanonicalSyncSelectionTier::Precise;
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
    const CanonicalSyncLifecycleResult &lifecycles,
    const CanonicalSyncOwnershipResult &ownership,
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
  for (const CanonicalSyncUnitSlotLifecycle &lifecycle :
       lifecycles.lifecycles) {
    for (const EventDomainKey &key :
         {EventDomainKey{lifecycle.producerResource,
                         lifecycle.consumerResource},
          EventDomainKey{lifecycle.consumerResource,
                         lifecycle.producerResource}}) {
      if (hasAvailableEventId(program, key, budget)) {
        keys.insert(key);
      }
    }
  }
  for (const CanonicalSyncOwnershipCycle &cycle : ownership.cycles) {
    for (const EventDomainKey &key :
         {EventDomainKey{cycle.producerResource, cycle.consumerResource},
          EventDomainKey{cycle.consumerResource, cycle.producerResource}}) {
      if (hasAvailableEventId(program, key, budget)) {
        keys.insert(key);
      }
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

LogicalResult addUnitSlotLifecycles(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const CanonicalSyncLifecycleResult &lifecycles,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
    const CanonicalSyncPatternOptions &patternOptions) {
  const SyncCoverGraph &graph = program.getGraph();
  std::map<SyncCoverScopeId, std::vector<CanonicalSyncMechanismId>>
      pipelineMembers;
  for (const CanonicalSyncUnitSlotLifecycle &lifecycle :
       lifecycles.lifecycles) {
    const auto readyDomain = domainIds.find(
        {lifecycle.producerResource, lifecycle.consumerResource});
    const auto releaseDomain = domainIds.find(
        {lifecycle.consumerResource, lifecycle.producerResource});
    const bool lacksEventDomain =
        readyDomain == domainIds.end() || releaseDomain == domainIds.end();
    if (lacksEventDomain) {
      continue;
    }
    std::optional<CanonicalSyncMechanismDescriptor> releaseDescriptor =
        makeCanonicalSyncUnitSlotProtocol(graph, lifecycle,
                                          releaseDomain->second);
    if (!releaseDescriptor) {
      return program.getFunction().emitError(
          "cannot build canonical sync unit-slot release protocol");
    }
    const CanonicalSyncProblemResult release = problem.internVerifiedProtocol(
        std::move(*releaseDescriptor), [&](const auto &candidate) {
          return verifyCanonicalSyncUnitSlotProtocol(
              graph, lifecycle, releaseDomain->second, candidate);
        });
    if (release.error == CanonicalSyncProblemError::LimitExceeded) {
      return success();
    }
    if (!release || !release.index) {
      return program.getFunction().emitError(
          "cannot admit canonical sync unit-slot release protocol");
    }
    std::optional<CanonicalSyncMechanismDescriptor> descriptor =
        makeCanonicalSyncAtomicUnitSlotProtocol(
            graph, lifecycle, readyDomain->second, releaseDomain->second);
    if (!descriptor) {
      return program.getFunction().emitError(
          "cannot build canonical sync atomic unit-slot protocol");
    }
    const CanonicalSyncProblemResult protocol = problem.internVerifiedProtocol(
        std::move(*descriptor), [&](const auto &candidate) {
          return verifyCanonicalSyncAtomicUnitSlotProtocol(
              graph, lifecycle, readyDomain->second, releaseDomain->second,
              candidate);
        });
    if (protocol.error == CanonicalSyncProblemError::LimitExceeded) {
      continue;
    }
    if (!protocol || !protocol.index) {
      return program.getFunction().emitError(
          "cannot admit canonical sync atomic unit-slot protocol");
    }
    if (problem.addConflict(*release.index, *protocol.index).error !=
        CanonicalSyncProblemError::None) {
      return program.getFunction().emitError(
          "cannot register canonical sync unit-slot protocol conflict");
    }
    std::vector<CanonicalSyncMechanismId> &pipeline =
        pipelineMembers[lifecycle.recurrenceScope];
    pipeline.push_back(*protocol.index);
  }
  if (!patternOptions.enablePipelineScope) {
    return success();
  }
  for (auto &[scope, members] : pipelineMembers) {
    (void)scope;
    std::sort(members.begin(), members.end());
    members.erase(std::unique(members.begin(), members.end()), members.end());
    const bool completePipeline = members.size() >= 2;
    if (!completePipeline) {
      continue;
    }
    const CanonicalSyncProblemResult pattern = addCanonicalSyncFeasiblePattern(
        problem, {CanonicalSyncPatternKind::PipelineScope, members});
    if (!pattern) {
      return program.getFunction().emitError(
          "cannot add canonical sync pipeline-scope pattern");
    }
  }
  return success();
}

LogicalResult addOwnershipCycles(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const CanonicalSyncOwnershipResult &ownership,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
    const CanonicalSyncPatternOptions &patternOptions) {
  std::map<SyncCoverScopeId, OwnershipPipelineMembers> l1Pipelines;
  std::map<SyncCoverScopeId, std::vector<OwnershipMechanismRecord>>
      accumulators;
  for (const CanonicalSyncOwnershipCycle &cycle : ownership.cycles) {
    const auto readyDomain =
        domainIds.find({cycle.producerResource, cycle.consumerResource});
    const auto releaseDomain =
        domainIds.find({cycle.consumerResource, cycle.producerResource});
    if (readyDomain == domainIds.end() || releaseDomain == domainIds.end()) {
      continue;
    }
    const CanonicalSyncEventDomain &readyResource =
        problem.getDomains()[readyDomain->second];
    const CanonicalSyncEventDomain &releaseResource =
        problem.getDomains()[releaseDomain->second];
    const std::size_t producerCount =
        cycle.paths.front().uses.front().producers.size();
    const std::size_t readyIds = availableEventIds(readyResource);
    const bool hasPrefixReady =
        cycle.kind == CanonicalSyncOwnershipKind::L0Operand &&
        program.getTargetCapabilities().mte1L0ReadySetCompletesPrefix;
    const bool readyOverflow =
        cycle.kind == CanonicalSyncOwnershipKind::L0Operand &&
        !hasPrefixReady && cycle.lanes.size() != 0 &&
        producerCount > readyIds / cycle.lanes.size();
    if (readyOverflow || cycle.lanes.size() > readyIds ||
        cycle.lanes.size() > availableEventIds(releaseResource)) {
      continue;
    }
    if (cycle.kind != CanonicalSyncOwnershipKind::L0Operand) {
      if (cycle.kind == CanonicalSyncOwnershipKind::L0Accumulator &&
          !program.getTargetCapabilities().mToFixAccumulatorBoundaryCompletes) {
        continue;
      }
      const bool needsScopePrefix =
          llvm::any_of(cycle.paths, [](const auto &path) {
            return llvm::any_of(path.uses, [](const auto &use) {
              return use.release.kind == SyncCoverAnchorKind::ScopeExit;
            });
          });
      if (needsScopePrefix &&
          !program.getTargetCapabilities().mte1ScopeExitSetCompletesPrefix) {
        continue;
      }
      std::optional<CanonicalSyncMechanismDescriptor> descriptor =
          makeCanonicalSyncAtomicOwnershipProtocol(
              program, cycle, readyDomain->second, releaseDomain->second);
      if (!descriptor) {
        return program.getFunction().emitError(
            "cannot build canonical sync atomic ownership protocol");
      }
      const CanonicalSyncProblemResult mechanism =
          problem.internVerifiedProtocol(
              std::move(*descriptor),
              [&](const CanonicalSyncMechanismDescriptor &candidate) {
                return verifyCanonicalSyncAtomicOwnershipProtocol(
                    program, cycle, readyDomain->second, releaseDomain->second,
                    candidate);
              });
      if (mechanism.error == CanonicalSyncProblemError::LimitExceeded) {
        continue;
      }
      if (!mechanism || !mechanism.index) {
        return program.getFunction().emitError(
                   "cannot admit canonical sync atomic ownership protocol")
               << " (kind=" << static_cast<unsigned>(cycle.kind)
               << ", protocol=" << static_cast<unsigned>(cycle.protocol)
               << ", error=" << static_cast<unsigned>(mechanism.error) << ')';
      }
      if (cycle.kind == CanonicalSyncOwnershipKind::L1Tile) {
        OwnershipPipelineMembers &pipeline = l1Pipelines[cycle.recurrenceScope];
        std::vector<OwnershipMechanismRecord> &members =
            cycle.protocol ==
                    CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch
                ? pipeline.alternatingL1
                : pipeline.stableL1;
        members.push_back({&cycle, *mechanism.index});
      } else if (cycle.kind == CanonicalSyncOwnershipKind::L0Accumulator) {
        accumulators[cycle.recurrenceScope].push_back(
            {&cycle, *mechanism.index});
      }
      continue;
    }
    std::optional<CanonicalSyncOwnershipProtocol> splitProtocol =
        makeCanonicalSyncOwnershipProtocol(program, cycle, readyDomain->second,
                                           releaseDomain->second);
    if (!splitProtocol || !verifyCanonicalSyncOwnershipProtocol(
                              program, cycle, readyDomain->second,
                              releaseDomain->second, *splitProtocol)) {
      return program.getFunction().emitError(
          "cannot build canonical sync split L0 ownership protocol");
    }
    const CanonicalSyncOwnershipProtocol reference = *splitProtocol;
    const CanonicalSyncProblemResult ready = problem.internVerifiedProtocol(
        std::move(splitProtocol->ready), [&](const auto &candidate) {
          CanonicalSyncOwnershipProtocol checked = reference;
          checked.ready = candidate;
          return verifyCanonicalSyncOwnershipProtocol(
              program, cycle, readyDomain->second, releaseDomain->second,
              checked);
        });
    const CanonicalSyncProblemResult release = problem.internVerifiedProtocol(
        std::move(splitProtocol->release), [&](const auto &candidate) {
          CanonicalSyncOwnershipProtocol checked = reference;
          checked.release = candidate;
          return verifyCanonicalSyncOwnershipProtocol(
              program, cycle, readyDomain->second, releaseDomain->second,
              checked);
        });
    const bool invalidReady =
        ready.error != CanonicalSyncProblemError::LimitExceeded &&
        (!ready || !ready.index);
    const bool invalidRelease =
        release.error != CanonicalSyncProblemError::LimitExceeded &&
        (!release || !release.index);
    if (invalidReady || invalidRelease) {
      return program.getFunction().emitError(
          "cannot admit canonical sync split L0 ownership protocol");
    }
    const bool completeSplit = ready && ready.index && release && release.index;
    if (completeSplit && patternOptions.enableRoundTrip) {
      const CanonicalSyncProblemResult splitRoundTrip =
          addCanonicalSyncFeasiblePattern(problem,
                                          {CanonicalSyncPatternKind::RoundTrip,
                                           {*ready.index, *release.index}});
      if (!splitRoundTrip) {
        return program.getFunction().emitError(
            "cannot add canonical sync split L0 round-trip pattern");
      }
    }
    std::optional<CanonicalSyncMechanismDescriptor> atomicDescriptor =
        makeCanonicalSyncAtomicOwnershipProtocol(
            program, cycle, readyDomain->second, releaseDomain->second);
    if (!atomicDescriptor) {
      return program.getFunction().emitError(
          "cannot build canonical sync atomic L0 ownership protocol");
    }
    const CanonicalSyncProblemResult atomic = problem.internVerifiedProtocol(
        std::move(*atomicDescriptor), [&](const auto &candidate) {
          return verifyCanonicalSyncAtomicOwnershipProtocol(
              program, cycle, readyDomain->second, releaseDomain->second,
              candidate);
        });
    if (atomic.error == CanonicalSyncProblemError::LimitExceeded) {
      continue;
    }
    if (!atomic || !atomic.index) {
      return program.getFunction().emitError(
          "cannot admit canonical sync atomic L0 ownership protocol");
    }
    const bool readyConflictFailed =
        ready && ready.index &&
        problem.addConflict(*ready.index, *atomic.index).error !=
            CanonicalSyncProblemError::None;
    const bool releaseConflictFailed =
        release && release.index &&
        problem.addConflict(*release.index, *atomic.index).error !=
            CanonicalSyncProblemError::None;
    if (readyConflictFailed || releaseConflictFailed) {
      return program.getFunction().emitError(
          "cannot register canonical sync L0 protocol conflicts");
    }
    l1Pipelines[cycle.recurrenceScope].l0Operands.push_back(
        {&cycle, *atomic.index});
  }
  for (const auto &[scope, pipeline] : l1Pipelines) {
    const std::optional<SyncCoverScopeId> outer =
        getEnclosingLoopScope(problem.getGraph(), scope);
    const bool hasUniquePipeline = outer && pipeline.stableL1.size() == 1 &&
                                   pipeline.alternatingL1.size() == 1 &&
                                   pipeline.l0Operands.size() == 1;
    if (!hasUniquePipeline) {
      continue;
    }
    const auto accumulator = accumulators.find(*outer);
    const bool hasUniqueAccumulator =
        accumulator != accumulators.end() && accumulator->second.size() == 1;
    if (!hasUniqueAccumulator) {
      continue;
    }
    const OwnershipMechanismRecord &stable = pipeline.stableL1.front();
    const OwnershipMechanismRecord &alternating =
        pipeline.alternatingL1.front();
    const OwnershipMechanismRecord &accumulatorMember =
        accumulator->second.front();
    const L0OwnershipMechanismRecord &l0 = pipeline.l0Operands.front();
    if (!stable.cycle || !alternating.cycle || !accumulatorMember.cycle ||
        !l0.cycle ||
        !ownershipSlotsAreDisjoint(*stable.cycle, *alternating.cycle)) {
      continue;
    }
    const auto stableReady = domainIds.find(
        {stable.cycle->producerResource, stable.cycle->consumerResource});
    const auto stableRelease = domainIds.find(
        {stable.cycle->consumerResource, stable.cycle->producerResource});
    const auto alternatingReady =
        domainIds.find({alternating.cycle->producerResource,
                        alternating.cycle->consumerResource});
    const auto alternatingRelease =
        domainIds.find({alternating.cycle->consumerResource,
                        alternating.cycle->producerResource});
    const bool hasRequiredDomains = stableReady != domainIds.end() &&
                                    stableRelease != domainIds.end() &&
                                    alternatingReady != domainIds.end() &&
                                    alternatingRelease != domainIds.end();
    if (!hasRequiredDomains) {
      continue;
    }
    std::optional<CanonicalSyncMechanismDescriptor> stableDescriptor =
        makeCanonicalSyncHierarchicalL1Protocol(program, *stable.cycle, *outer,
                                                stableReady->second,
                                                stableRelease->second);
    std::optional<CanonicalSyncMechanismDescriptor> alternatingDescriptor =
        makeCanonicalSyncHierarchicalL1Protocol(
            program, *alternating.cycle, *outer, alternatingReady->second,
            alternatingRelease->second);
    if (!stableDescriptor || !alternatingDescriptor) {
      continue;
    }
    const CanonicalSyncProblemResult hierarchicalStable =
        problem.internVerifiedProtocol(
            std::move(*stableDescriptor), [&](const auto &candidate) {
              return verifyCanonicalSyncHierarchicalL1Protocol(
                  program, *stable.cycle, *outer, stableReady->second,
                  stableRelease->second, candidate);
            });
    const CanonicalSyncProblemResult hierarchicalAlternating =
        problem.internVerifiedProtocol(
            std::move(*alternatingDescriptor), [&](const auto &candidate) {
              return verifyCanonicalSyncHierarchicalL1Protocol(
                  program, *alternating.cycle, *outer, alternatingReady->second,
                  alternatingRelease->second, candidate);
            });
    if (hierarchicalStable && hierarchicalStable.index) {
      const CanonicalSyncProblemResult conflict =
          problem.addConflict(stable.mechanism, *hierarchicalStable.index);
      if (!conflict) {
        return program.getFunction().emitError(
            "cannot register canonical sync stable hierarchical conflict");
      }
    }
    if (hierarchicalAlternating && hierarchicalAlternating.index) {
      const CanonicalSyncProblemResult conflict = problem.addConflict(
          alternating.mechanism, *hierarchicalAlternating.index);
      if (!conflict) {
        return program.getFunction().emitError(
            "cannot register canonical sync alternating hierarchical "
            "conflict");
      }
    }
    const bool optionalLimitReached =
        hierarchicalStable.error == CanonicalSyncProblemError::LimitExceeded ||
        hierarchicalAlternating.error ==
            CanonicalSyncProblemError::LimitExceeded;
    if (optionalLimitReached) {
      continue;
    }
    if (!hierarchicalStable || !hierarchicalAlternating ||
        !hierarchicalStable.index || !hierarchicalAlternating.index) {
      return program.getFunction().emitError(
                 "cannot admit canonical sync hierarchical L1 protocols")
             << " (stable=" << static_cast<unsigned>(hierarchicalStable.error)
             << ", alternating="
             << static_cast<unsigned>(hierarchicalAlternating.error) << ')';
    }
    if (patternOptions.enablePipelineScope) {
      const CanonicalSyncProblemResult pattern =
          addCanonicalSyncFeasiblePattern(
              problem,
              {CanonicalSyncPatternKind::PipelineScope,
               {*hierarchicalStable.index, *hierarchicalAlternating.index,
                accumulatorMember.mechanism, l0.mechanism}});
      if (!pattern) {
        return program.getFunction().emitError(
            "cannot add canonical sync ownership-pipeline pattern");
      }
    }
  }
  return success();
}

LogicalResult addBarrierFallbacks(const CanonicalSyncProgram &program,
                                  CanonicalSyncPatternProblem &problem,
                                  const SyncCoverDemandSet &baseline) {
  const SyncCoverGraph &graph = program.getGraph();
  const std::vector<std::uint32_t> allResources = getIssueResources(graph);
  std::map<SyncCoverNodeId, BarrierFallbackGroup> targetedGroups;
  std::map<SyncCoverNodeId, BarrierFallbackGroup> rescueGroups;
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (baseline.contains(demandId)) {
      continue;
    }
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    BarrierFallbackGroup &group = source.resource == target.resource
                                      ? targetedGroups[target.id]
                                      : rescueGroups[target.id];
    group.demands.push_back(demandId);
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
  const bool failedGroups = failed(addGroups(targetedGroups, false)) ||
                            failed(addGroups(rescueGroups, true));
  if (failedGroups) {
    return program.getFunction().emitError(
        "cannot add canonical sync barrier fallback");
  }
  return success();
}

CanonicalSyncMechanismDescriptor
makeScarcityBarrier(const SyncCoverGraph &graph, const SyncCoverEdge &edge) {
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Barrier;
  descriptor.selectionTier = CanonicalSyncSelectionTier::ScarcityFrontier;
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
makeScarcityEvent(const SyncCoverGraph &graph, const SyncCoverEdge &edge,
                  CanonicalSyncEventDomainId domain) {
  const SyncCoverNode &source = graph.getNodes()[edge.source];
  const SyncCoverNode &target = graph.getNodes()[edge.target];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Event;
  descriptor.selectionTier = CanonicalSyncSelectionTier::ScarcityFrontier;
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

struct ScarcityFrontierProposal {
  SyncCoverEdge barrier;
  SyncCoverEdge event;
  CanonicalSyncEventDomainId domain = 0;
};

bool frontierProposalLess(const ScarcityFrontierProposal &left,
                          const ScarcityFrontierProposal &right) {
  return std::tie(left.domain, left.barrier.source, left.barrier.target,
                  left.event.source, left.event.target) <
         std::tie(right.domain, right.barrier.source, right.barrier.target,
                  right.event.source, right.event.target);
}

bool frontierProposalEqual(const ScarcityFrontierProposal &left,
                           const ScarcityFrontierProposal &right) {
  return !frontierProposalLess(left, right) &&
         !frontierProposalLess(right, left);
}

LogicalResult
addScarcityFrontierPatterns(const CanonicalSyncProgram &program,
                            CanonicalSyncPatternProblem &problem,
                            ArrayRef<DirectEventRecord> directEvents) {
  constexpr std::size_t kMaximumFrontierProposals = 4096;
  const SyncCoverGraph &graph = program.getGraph();
  std::vector<ScarcityFrontierProposal> proposals;
  for (std::size_t first = 0; first < directEvents.size(); ++first) {
    const SyncCoverDemand &firstDemand =
        graph.getDemands()[directEvents[first].demand];
    for (std::size_t second = first + 1; second < directEvents.size();
         ++second) {
      const SyncCoverDemand &secondDemand =
          graph.getDemands()[directEvents[second].demand];
      const bool compatible =
          directEvents[first].domain == directEvents[second].domain &&
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
      proposals.push_back({barrier, event, directEvents[first].domain});
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
  for (const ScarcityFrontierProposal &proposal : proposals) {
    const CanonicalSyncProblemResult barrier =
        problem.internMechanism(makeScarcityBarrier(graph, proposal.barrier));
    const CanonicalSyncProblemResult event = problem.internMechanism(
        makeScarcityEvent(graph, proposal.event, proposal.domain));
    if (!barrier || !event || !barrier.index || !event.index) {
      return program.getFunction().emitError(
          "cannot add canonical sync scarcity-frontier mechanisms");
    }
    const CanonicalSyncProblemResult pattern = addCanonicalSyncFeasiblePattern(
        problem, {CanonicalSyncPatternKind::ScarcityFrontier,
                  {*barrier.index, *event.index}});
    if (!pattern) {
      return program.getFunction().emitError(
          "cannot add canonical sync scarcity-frontier pattern");
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
  CanonicalSyncLifecycleResult lifecycles;
  if (options.patterns.enableSpecializedOwnership) {
    lifecycles = discoverCanonicalSyncUnitSlotLifecycles(program.getGraph(),
                                                         problem->getDemands());
  }
  if (!lifecycles) {
    program.getFunction().emitError(
        "cannot discover canonical sync unit-slot lifecycles");
    return failure();
  }
  CanonicalSyncOwnershipResult ownership;
  if (options.patterns.enableSpecializedOwnership) {
    ownership = discoverCanonicalSyncOwnershipCycles(program);
  }
  if (!ownership) {
    program.getFunction().emitError(
        "cannot discover canonical sync ownership cycles");
    return failure();
  }
  if (ownership.truncated) {
    program.getFunction().emitRemark(
        "canonical sync ownership discovery reached its bounded analysis "
        "limit; conservative event and barrier fallbacks remain available");
  }
  std::map<EventDomainKey, CanonicalSyncEventDomainId> domainIds;
  std::vector<DirectEventRecord> directEvents;
  const bool failedBuild =
      failed(addBarrierFallbacks(program, *problem, baseline.covered)) ||
      failed(addEventDomains(program, options.eventIdBudget, *problem,
                             baseline.covered, lifecycles, ownership,
                             domainIds)) ||
      failed(addDirectEvents(program, *problem, baseline.covered, domainIds,
                             directEvents)) ||
      (options.patterns.enableDirectPairs &&
       failed(addDirectPairPatterns(program, *problem))) ||
      (options.patterns.enableScarcityFrontiers &&
       failed(addScarcityFrontierPatterns(program, *problem, directEvents))) ||
      (options.patterns.enableSpecializedOwnership &&
       failed(addOwnershipCycles(program, *problem, ownership, domainIds,
                                 options.patterns))) ||
      (options.patterns.enableSpecializedOwnership &&
       failed(addUnitSlotLifecycles(program, *problem, lifecycles, domainIds,
                                    options.patterns)));
  if (failedBuild) {
    return failure();
  }
  if (options.patterns.enableRoundTrip) {
    const CanonicalSyncProblemResult roundTrips =
        addCanonicalSyncRoundTripPatterns(*problem);
    if (!roundTrips) {
      program.getFunction().emitError(
          "cannot add canonical sync round-trip patterns");
      return failure();
    }
  }
  if (problem->wasPatternGenerationTruncated()) {
    program.getFunction().emitRemark(
        "canonical sync pattern generation reached its bounded proposal "
        "limit; singleton and barrier fallbacks remain available");
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
