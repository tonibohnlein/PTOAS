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
using DemandMechanismMap =
    std::map<SyncCoverDemandId, CanonicalSyncMechanismId>;

struct BarrierFallbackGroup {
  bool broad = false;
  std::vector<SyncCoverDemandId> demands;
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
    const CanonicalSyncLifecycleResult &lifecycles,
    const CanonicalSyncOwnershipResult &ownership,
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
  for (const CanonicalSyncUnitSlotLifecycle &lifecycle :
       lifecycles.lifecycles) {
    const EventDomainKey key{lifecycle.consumerResource,
                             lifecycle.producerResource};
    if (hasAvailableEventId(program, key, budget)) {
      keys.insert(key);
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
    DemandMechanismMap &mechanismsByDemand) {
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
    if (!added.index) {
      return program.getFunction().emitError(
          "canonical sync direct event has no mechanism identity");
    }
    mechanismsByDemand.emplace(demandId, *added.index);
  }
  return success();
}

LogicalResult addUnitSlotLifecycles(
    const CanonicalSyncProgram &program, CanonicalSyncPatternProblem &problem,
    const CanonicalSyncLifecycleResult &lifecycles,
    const std::map<EventDomainKey, CanonicalSyncEventDomainId> &domainIds,
    const DemandMechanismMap &mechanismsByDemand,
    const CanonicalSyncPatternOptions &patternOptions) {
  const SyncCoverGraph &graph = program.getGraph();
  std::map<SyncCoverScopeId, std::vector<CanonicalSyncMechanismId>>
      pipelineMembers;
  for (const CanonicalSyncUnitSlotLifecycle &lifecycle :
       lifecycles.lifecycles) {
    const EventDomainKey key{lifecycle.consumerResource,
                             lifecycle.producerResource};
    const auto domain = domainIds.find(key);
    if (domain == domainIds.end()) {
      continue;
    }
    std::vector<CanonicalSyncMechanismId> members;
    bool hasEveryReady = true;
    for (SyncCoverDemandId ready : lifecycle.readyDemands) {
      const auto mechanism = mechanismsByDemand.find(ready);
      if (mechanism == mechanismsByDemand.end()) {
        hasEveryReady = false;
        break;
      }
      members.push_back(mechanism->second);
    }
    if (!hasEveryReady) {
      continue;
    }
    std::optional<CanonicalSyncMechanismDescriptor> descriptor =
        makeCanonicalSyncUnitSlotProtocol(graph, lifecycle, domain->second);
    if (!descriptor) {
      return program.getFunction().emitError(
          "cannot build canonical sync unit-slot protocol");
    }
    const CanonicalSyncProblemResult protocol = problem.internVerifiedProtocol(
        std::move(*descriptor), [&](const auto &candidate) {
          return verifyCanonicalSyncUnitSlotProtocol(graph, lifecycle,
                                                     domain->second, candidate);
        });
    if (protocol.error == CanonicalSyncProblemError::LimitExceeded) {
      return success();
    }
    if (!protocol || !protocol.index) {
      return program.getFunction().emitError(
          "cannot admit canonical sync unit-slot protocol");
    }
    members.push_back(*protocol.index);
    std::sort(members.begin(), members.end());
    members.erase(std::unique(members.begin(), members.end()), members.end());
    const bool completeLifecycle = members.size() >= 2;
    if (!completeLifecycle) {
      continue;
    }
    if (!patternOptions.enableSlotLifecycle) {
      continue;
    }
    const CanonicalSyncProblemResult pattern = addCanonicalSyncFeasiblePattern(
        problem, {CanonicalSyncPatternKind::SlotLifecycle, members});
    if (!pattern) {
      return program.getFunction().emitError(
          "cannot add canonical sync slot-lifecycle pattern");
    }
    if (!pattern.index) {
      continue;
    }
    std::vector<CanonicalSyncMechanismId> &pipeline =
        pipelineMembers[lifecycle.recurrenceScope];
    pipeline.insert(pipeline.end(), members.begin(), members.end());
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
    const bool readyOverflow =
        cycle.kind == CanonicalSyncOwnershipKind::L0Operand &&
        cycle.lanes.size() != 0 &&
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
    std::optional<CanonicalSyncOwnershipProtocol> protocol =
        makeCanonicalSyncOwnershipProtocol(program, cycle, readyDomain->second,
                                           releaseDomain->second);
    if (!protocol || !verifyCanonicalSyncOwnershipProtocol(
                         program, cycle, readyDomain->second,
                         releaseDomain->second, *protocol)) {
      return program.getFunction().emitError(
          "cannot build canonical sync ownership protocol");
    }
    const CanonicalSyncOwnershipProtocol reference = *protocol;
    const CanonicalSyncProblemResult ready = problem.internVerifiedProtocol(
        std::move(protocol->ready),
        [&](const CanonicalSyncMechanismDescriptor &candidate) {
          CanonicalSyncOwnershipProtocol checked = reference;
          checked.ready = candidate;
          return verifyCanonicalSyncOwnershipProtocol(
              program, cycle, readyDomain->second, releaseDomain->second,
              checked);
        });
    const CanonicalSyncProblemResult release = problem.internVerifiedProtocol(
        std::move(protocol->release),
        [&](const CanonicalSyncMechanismDescriptor &candidate) {
          CanonicalSyncOwnershipProtocol checked = reference;
          checked.release = candidate;
          return verifyCanonicalSyncOwnershipProtocol(
              program, cycle, readyDomain->second, releaseDomain->second,
              checked);
        });
    if (ready.error == CanonicalSyncProblemError::LimitExceeded ||
        release.error == CanonicalSyncProblemError::LimitExceeded) {
      continue;
    }
    if (!ready || !release || !ready.index || !release.index) {
      return program.getFunction().emitError(
                 "cannot admit canonical sync ownership protocol")
             << " (kind=" << static_cast<unsigned>(cycle.kind)
             << ", ready-error=" << static_cast<unsigned>(ready.error)
             << ", release-error=" << static_cast<unsigned>(release.error)
             << ")";
    }
    if (patternOptions.enableOwnershipCycle) {
      const CanonicalSyncProblemResult pattern =
          addCanonicalSyncFeasiblePattern(
              problem, {CanonicalSyncPatternKind::OwnershipCycle,
                        {*ready.index, *release.index}});
      if (!pattern) {
        return program.getFunction().emitError(
            "cannot add canonical sync ownership-cycle pattern");
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
    if (problem.addConflict(*ready.index, *atomic.index).error !=
            CanonicalSyncProblemError::None ||
        problem.addConflict(*release.index, *atomic.index).error !=
            CanonicalSyncProblemError::None) {
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
      const CanonicalSyncProblemResult conflict = problem.addConflict(
          stable.mechanism, *hierarchicalStable.index);
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
  const CanonicalSyncLifecycleResult lifecycles =
      discoverCanonicalSyncUnitSlotLifecycles(program.getGraph(),
                                              problem->getDemands());
  if (!lifecycles) {
    program.getFunction().emitError(
        "cannot discover canonical sync unit-slot lifecycles");
    return failure();
  }
  const CanonicalSyncOwnershipResult ownership =
      discoverCanonicalSyncOwnershipCycles(program);
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
  DemandMechanismMap mechanismsByDemand;
  const bool failedBuild =
      failed(addBarrierFallbacks(program, *problem, baseline.covered)) ||
      failed(addEventDomains(program, options.eventIdBudget, *problem,
                             baseline.covered, lifecycles, ownership,
                             domainIds)) ||
      failed(addDirectEvents(program, *problem, baseline.covered, domainIds,
                             mechanismsByDemand)) ||
      failed(addOwnershipCycles(program, *problem, ownership, domainIds,
                                options.patterns)) ||
      failed(addUnitSlotLifecycles(program, *problem, lifecycles, domainIds,
                                   mechanismsByDemand, options.patterns));
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
  const CanonicalSyncProblemResult frozen = problem->freeze();
  if (!frozen) {
    program.getFunction().emitError()
        << "cannot freeze canonical sync singleton problem, error="
        << static_cast<unsigned>(frozen.error);
    return failure();
  }
  return problem;
}
