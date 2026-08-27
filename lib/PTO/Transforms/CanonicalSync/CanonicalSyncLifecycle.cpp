// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncLifecycle.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir::pto;

namespace {

using AccessPair =
    std::pair<SyncCoverStorageAccessId, SyncCoverStorageAccessId>;

bool intervalEqual(SyncCoverStorageInterval first,
                   SyncCoverStorageInterval second) {
  return first.begin == second.begin && first.end == second.end;
}

bool overlaps(SyncCoverStorageInterval first, SyncCoverStorageInterval second) {
  return first.begin < second.end && second.begin < first.end;
}

bool hasKind(const SyncCoverDemand &demand, SyncCoverDemandKind kind) {
  return std::binary_search(demand.provenanceKinds.begin(),
                            demand.provenanceKinds.end(), kind);
}

std::optional<AccessPair>
getExactAccessPair(const SyncCoverGraph &graph,
                   SyncCoverStorageWitnessId witnessId) {
  if (witnessId >= graph.getStorageWitnesses().size()) {
    return std::nullopt;
  }
  const SyncCoverStorageWitness &witness =
      graph.getStorageWitnesses()[witnessId];
  const bool hasSource =
      witness.sourceAccess < graph.getStorageAccesses().size();
  const bool hasTarget =
      witness.targetAccess < graph.getStorageAccesses().size();
  if (!hasSource || !hasTarget) {
    return std::nullopt;
  }
  const SyncCoverStorageAccess &source =
      graph.getStorageAccesses()[witness.sourceAccess];
  const SyncCoverStorageAccess &target =
      graph.getStorageAccesses()[witness.targetAccess];
  const bool exact = source.exactPhysical && target.exactPhysical &&
                     source.domain == target.domain &&
                     intervalEqual(source.extent, target.extent) &&
                     intervalEqual(source.extent, witness.overlap);
  return exact ? std::optional<AccessPair>(
                     {witness.sourceAccess, witness.targetAccess})
               : std::nullopt;
}

bool demandHasAccessPair(const SyncCoverGraph &graph,
                         const SyncCoverDemand &demand, AccessPair expected) {
  return std::any_of(demand.storageWitnesses.begin(),
                     demand.storageWitnesses.end(),
                     [&](SyncCoverStorageWitnessId witness) {
                       const std::optional<AccessPair> pair =
                           getExactAccessPair(graph, witness);
                       return pair && *pair == expected;
                     });
}

bool isPathInsensitiveEndpoint(const SyncCoverGraph &graph,
                               SyncCoverScopeId recurrenceScope,
                               SyncCoverNodeId node) {
  if (node >= graph.getNodes().size()) {
    return false;
  }
  const SyncCoverNode &endpoint = graph.getNodes()[node];
  const std::optional<std::size_t> recurrenceDepth =
      graph.getScopeLoopDepth(recurrenceScope);
  const std::optional<std::size_t> endpointDepth =
      graph.getScopeLoopDepth(endpoint.scope);
  return endpoint.guard.literals.empty() && recurrenceDepth && endpointDepth &&
         *recurrenceDepth == *endpointDepth &&
         graph.scopeMustExecuteWithin(recurrenceScope, endpoint.scope);
}

SyncCoverEdge releaseEdge(const SyncCoverGraph &graph,
                          const CanonicalSyncUnitSlotLifecycle &lifecycle) {
  const SyncCoverDemand &release = graph.getDemands()[lifecycle.releaseDemand];
  return {
      release.source,     release.target,   SyncCoverEdgeKind::CompletionSupply,
      release.scope,      release.distance, release.sourceGuard,
      release.targetGuard};
}

SyncCoverEdge readyEdge(const SyncCoverGraph &graph,
                        const CanonicalSyncUnitSlotLifecycle &lifecycle) {
  const SyncCoverDemand &ready =
      graph.getDemands()[lifecycle.readyDemands.front()];
  return {ready.source,     ready.target,   SyncCoverEdgeKind::CompletionSupply,
          ready.scope,      ready.distance, ready.sourceGuard,
          ready.targetGuard};
}

bool edgeEqual(const SyncCoverEdge &left, const SyncCoverEdge &right) {
  return std::tie(left.source, left.target, left.kind, left.scope,
                  left.distance, left.sourceGuard.literals,
                  left.targetGuard.literals) ==
         std::tie(right.source, right.target, right.kind, right.scope,
                  right.distance, right.sourceGuard.literals,
                  right.targetGuard.literals);
}

bool actionMatches(const CanonicalSyncAction &action,
                   CanonicalSyncActionKind kind, std::uint32_t resource,
                   SyncCoverAnchorKind anchorKind, SyncCoverNodeId node,
                   SyncCoverScopeId scope, std::size_t eventUse = 0) {
  return action.kind == kind && action.resource == resource &&
         action.anchor.kind == anchorKind && action.anchor.node == node &&
         action.anchor.scope == scope && action.anchor.position == 0 &&
         action.eventUse == eventUse && action.eventLane == 0 &&
         action.drainedResources.empty() &&
         action.barrierKind == CanonicalSyncBarrierKind::Targeted &&
         action.guard == CanonicalSyncActionGuardKind::None &&
         !action.guardScope;
}

} // namespace

bool mlir::pto::verifyCanonicalSyncUnitSlotLifecycle(
    const SyncCoverGraph &graph,
    const CanonicalSyncUnitSlotLifecycle &lifecycle) {
  const bool invalidId =
      lifecycle.releaseDemand >= graph.getDemands().size() ||
      lifecycle.producerAccess >= graph.getStorageAccesses().size() ||
      lifecycle.consumerAccess >= graph.getStorageAccesses().size() ||
      lifecycle.recurrenceScope >= graph.getScopes().size();
  if (invalidId || !graph.getScopes()[lifecycle.recurrenceScope].isLoop ||
      !graph.getScopes()[lifecycle.recurrenceScope].timeline) {
    return false;
  }
  const SyncCoverStorageAccess &producer =
      graph.getStorageAccesses()[lifecycle.producerAccess];
  const SyncCoverStorageAccess &consumer =
      graph.getStorageAccesses()[lifecycle.consumerAccess];
  const SyncCoverDemand &release = graph.getDemands()[lifecycle.releaseDemand];
  const bool wrongStorage =
      producer.domain != lifecycle.domain ||
      consumer.domain != lifecycle.domain || !producer.exactPhysical ||
      !consumer.exactPhysical ||
      !intervalEqual(producer.extent, lifecycle.extent) ||
      !intervalEqual(consumer.extent, lifecycle.extent) ||
      producer.mode != SyncCoverStorageAccessMode::Write ||
      consumer.mode != SyncCoverStorageAccessMode::Read;
  const bool wrongRelease =
      release.distance != 1 || release.scope != lifecycle.recurrenceScope ||
      !release.sourceGuard.literals.empty() ||
      !release.targetGuard.literals.empty() ||
      !hasKind(release, SyncCoverDemandKind::MemoryWAR) ||
      release.source != consumer.node || release.target != producer.node ||
      !demandHasAccessPair(
          graph, release, {lifecycle.consumerAccess, lifecycle.producerAccess});
  const bool wrongResources =
      graph.getNodes()[producer.node].resource != lifecycle.producerResource ||
      graph.getNodes()[consumer.node].resource != lifecycle.consumerResource ||
      lifecycle.producerResource == lifecycle.consumerResource;
  if (wrongStorage || wrongRelease || wrongResources ||
      !isPathInsensitiveEndpoint(graph, lifecycle.recurrenceScope,
                                 producer.node) ||
      !isPathInsensitiveEndpoint(graph, lifecycle.recurrenceScope,
                                 consumer.node) ||
      !syncCoverNodeCanProduceCompletion(graph, consumer.node,
                                         lifecycle.producerResource)) {
    return false;
  }
  for (SyncCoverDemandId readyId : lifecycle.readyDemands) {
    if (readyId >= graph.getDemands().size()) {
      return false;
    }
    const SyncCoverDemand &ready = graph.getDemands()[readyId];
    const bool wrongReady =
        ready.distance != 0 || !ready.sourceGuard.literals.empty() ||
        !ready.targetGuard.literals.empty() ||
        !hasKind(ready, SyncCoverDemandKind::MemoryRAW) ||
        ready.source != producer.node || ready.target != consumer.node ||
        !demandHasAccessPair(
            graph, ready, {lifecycle.producerAccess, lifecycle.consumerAccess});
    if (wrongReady) {
      return false;
    }
  }
  const bool hasReadyDemand = !lifecycle.readyDemands.empty();
  const bool producerCanComplete = syncCoverNodeCanProduceCompletion(
      graph, producer.node, lifecycle.consumerResource);
  if (!hasReadyDemand || !producerCanComplete) {
    return false;
  }
  for (const SyncCoverStorageAccess &access : graph.getStorageAccesses()) {
    const bool managed = access.domain == lifecycle.domain &&
                         overlaps(access.extent, lifecycle.extent);
    if (managed && access.id != lifecycle.producerAccess &&
        access.id != lifecycle.consumerAccess) {
      return false;
    }
  }
  return true;
}

CanonicalSyncLifecycleResult mlir::pto::discoverCanonicalSyncUnitSlotLifecycles(
    const SyncCoverGraph &graph,
    const std::vector<SyncCoverDemandId> &activeDemands,
    CanonicalSyncLifecycleOptions options) {
  CanonicalSyncLifecycleResult result;
  const bool validGraph = graph.isStructureFrozen() && graph.validate();
  if (!validGraph) {
    result.error = CanonicalSyncLifecycleError::InvalidGraph;
    return result;
  }
  std::map<AccessPair, std::vector<SyncCoverDemandId>> readyByAccesses;
  struct ReleaseOpportunity {
    SyncCoverDemandId demand = 0;
    AccessPair accesses;
  };
  std::vector<ReleaseOpportunity> releases;
  for (SyncCoverDemandId demandId : activeDemands) {
    if (demandId >= graph.getDemands().size()) {
      result.error = CanonicalSyncLifecycleError::InvalidGraph;
      return result;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    for (SyncCoverStorageWitnessId witness : demand.storageWitnesses) {
      const std::optional<AccessPair> accesses =
          getExactAccessPair(graph, witness);
      if (!accesses) {
        continue;
      }
      const SyncCoverStorageAccess &source =
          graph.getStorageAccesses()[accesses->first];
      const SyncCoverStorageAccess &target =
          graph.getStorageAccesses()[accesses->second];
      if (demand.distance == 0 &&
          hasKind(demand, SyncCoverDemandKind::MemoryRAW) &&
          source.mode == SyncCoverStorageAccessMode::Write &&
          target.mode == SyncCoverStorageAccessMode::Read) {
        readyByAccesses[*accesses].push_back(demandId);
      }
      if (demand.distance == 1 &&
          hasKind(demand, SyncCoverDemandKind::MemoryWAR) &&
          source.mode == SyncCoverStorageAccessMode::Read &&
          target.mode == SyncCoverStorageAccessMode::Write) {
        releases.push_back({demandId, *accesses});
      }
    }
  }
  for (auto &[accesses, demands] : readyByAccesses) {
    (void)accesses;
    std::sort(demands.begin(), demands.end());
    demands.erase(std::unique(demands.begin(), demands.end()), demands.end());
  }
  std::sort(releases.begin(), releases.end(),
            [](const auto &left, const auto &right) {
              return std::tie(left.demand, left.accesses) <
                     std::tie(right.demand, right.accesses);
            });
  std::set<std::tuple<SyncCoverDemandId, SyncCoverStorageAccessId,
                      SyncCoverStorageAccessId>>
      proposed;
  for (const ReleaseOpportunity &release : releases) {
    const AccessPair readyAccesses{release.accesses.second,
                                   release.accesses.first};
    const auto ready = readyByAccesses.find(readyAccesses);
    if (ready == readyByAccesses.end()) {
      continue;
    }
    const auto key = std::make_tuple(release.demand, readyAccesses.first,
                                     readyAccesses.second);
    if (!proposed.insert(key).second) {
      continue;
    }
    const bool lifecycleCapReached =
        result.lifecycles.size() == options.maximumLifecycles;
    if (lifecycleCapReached) {
      result.truncated = true;
      return result;
    }
    const SyncCoverStorageAccess &producer =
        graph.getStorageAccesses()[readyAccesses.first];
    const SyncCoverStorageAccess &consumer =
        graph.getStorageAccesses()[readyAccesses.second];
    const std::size_t inspection = graph.getStorageAccesses().size();
    if (result.evaluations > options.maximumEvaluations ||
        inspection > options.maximumEvaluations - result.evaluations) {
      result.truncated = true;
      return result;
    }
    result.evaluations += inspection;
    CanonicalSyncUnitSlotLifecycle lifecycle;
    lifecycle.id = result.lifecycles.size();
    lifecycle.domain = producer.domain;
    lifecycle.extent = producer.extent;
    lifecycle.producerAccess = producer.id;
    lifecycle.consumerAccess = consumer.id;
    lifecycle.producerResource = graph.getNodes()[producer.node].resource;
    lifecycle.consumerResource = graph.getNodes()[consumer.node].resource;
    lifecycle.recurrenceScope = graph.getDemands()[release.demand].scope;
    lifecycle.releaseDemand = release.demand;
    lifecycle.readyDemands = ready->second;
    if (verifyCanonicalSyncUnitSlotLifecycle(graph, lifecycle)) {
      result.lifecycles.push_back(std::move(lifecycle));
    }
  }
  return result;
}

std::optional<CanonicalSyncMechanismDescriptor>
mlir::pto::makeCanonicalSyncUnitSlotProtocol(
    const SyncCoverGraph &graph,
    const CanonicalSyncUnitSlotLifecycle &lifecycle,
    CanonicalSyncEventDomainId domain) {
  if (!verifyCanonicalSyncUnitSlotLifecycle(graph, lifecycle)) {
    return std::nullopt;
  }
  const SyncCoverStorageAccess &producer =
      graph.getStorageAccesses()[lifecycle.producerAccess];
  const SyncCoverStorageAccess &consumer =
      graph.getStorageAccesses()[lifecycle.consumerAccess];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Protocol;
  descriptor.eventUses.push_back({domain, 1, lifecycle.recurrenceScope});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventSet,
       lifecycle.consumerResource,
       {SyncCoverAnchorKind::ScopeEntry, 0, lifecycle.recurrenceScope},
       0,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventWait,
       lifecycle.producerResource,
       {SyncCoverAnchorKind::BeforeNode, producer.node, 0},
       0,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventSet,
       lifecycle.consumerResource,
       {SyncCoverAnchorKind::AfterNode, consumer.node, 0},
       0,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventWait,
       lifecycle.producerResource,
       {SyncCoverAnchorKind::ScopeExit, 0, lifecycle.recurrenceScope},
       0,
       0,
       {}});
  descriptor.supplies.push_back({releaseEdge(graph, lifecycle), 0, std::nullopt,
                                 2, 1,
                                 CanonicalSyncSupplyProof::VerifiedProtocol});
  return descriptor;
}

bool mlir::pto::verifyCanonicalSyncUnitSlotProtocol(
    const SyncCoverGraph &graph,
    const CanonicalSyncUnitSlotLifecycle &lifecycle,
    CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  const bool validLifecycle =
      verifyCanonicalSyncUnitSlotLifecycle(graph, lifecycle);
  const bool validCardinality =
      descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
      descriptor.eventUses.size() == 1 && descriptor.actions.size() == 4 &&
      descriptor.supplies.size() == 1;
  if (!validLifecycle || !validCardinality) {
    return false;
  }
  const CanonicalSyncEventUse &use = descriptor.eventUses.front();
  const CanonicalSyncSupplyBinding &binding = descriptor.supplies.front();
  const SyncCoverStorageAccess &producer =
      graph.getStorageAccesses()[lifecycle.producerAccess];
  const SyncCoverStorageAccess &consumer =
      graph.getStorageAccesses()[lifecycle.consumerAccess];
  const bool correctUse = use.domain == domain && use.width == 1 &&
                          use.recurrenceScope == lifecycle.recurrenceScope;
  const bool correctBinding =
      binding.edge.source == consumer.node &&
      binding.edge.target == producer.node && binding.edge.distance == 1 &&
      binding.edge.scope == lifecycle.recurrenceScope &&
      binding.edge.kind == SyncCoverEdgeKind::CompletionSupply &&
      binding.eventUse == 0 && !binding.barrierAction &&
      binding.produceAction == 2 && binding.consumeAction == 1 &&
      binding.proof == CanonicalSyncSupplyProof::VerifiedProtocol;
  return correctUse && correctBinding &&
         actionMatches(descriptor.actions[0], CanonicalSyncActionKind::EventSet,
                       lifecycle.consumerResource,
                       SyncCoverAnchorKind::ScopeEntry, 0,
                       lifecycle.recurrenceScope) &&
         actionMatches(descriptor.actions[1],
                       CanonicalSyncActionKind::EventWait,
                       lifecycle.producerResource,
                       SyncCoverAnchorKind::BeforeNode, producer.node, 0) &&
         actionMatches(descriptor.actions[2], CanonicalSyncActionKind::EventSet,
                       lifecycle.consumerResource,
                       SyncCoverAnchorKind::AfterNode, consumer.node, 0) &&
         actionMatches(
             descriptor.actions[3], CanonicalSyncActionKind::EventWait,
             lifecycle.producerResource, SyncCoverAnchorKind::ScopeExit, 0,
             lifecycle.recurrenceScope);
}

std::optional<CanonicalSyncMechanismDescriptor>
mlir::pto::makeCanonicalSyncAtomicUnitSlotProtocol(
    const SyncCoverGraph &graph,
    const CanonicalSyncUnitSlotLifecycle &lifecycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain) {
  if (!verifyCanonicalSyncUnitSlotLifecycle(graph, lifecycle)) {
    return std::nullopt;
  }
  const SyncCoverStorageAccess &producer =
      graph.getStorageAccesses()[lifecycle.producerAccess];
  const SyncCoverStorageAccess &consumer =
      graph.getStorageAccesses()[lifecycle.consumerAccess];
  CanonicalSyncMechanismDescriptor descriptor;
  descriptor.kind = CanonicalSyncMechanismKind::Protocol;
  descriptor.eventUses.push_back({readyDomain, 1, lifecycle.recurrenceScope});
  descriptor.eventUses.push_back({releaseDomain, 1, lifecycle.recurrenceScope});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventSet,
       lifecycle.producerResource,
       {SyncCoverAnchorKind::AfterNode, producer.node, 0},
       0,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventWait,
       lifecycle.consumerResource,
       {SyncCoverAnchorKind::BeforeNode, consumer.node, 0},
       0,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventSet,
       lifecycle.consumerResource,
       {SyncCoverAnchorKind::ScopeEntry, 0, lifecycle.recurrenceScope},
       1,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventWait,
       lifecycle.producerResource,
       {SyncCoverAnchorKind::BeforeNode, producer.node, 0},
       1,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventSet,
       lifecycle.consumerResource,
       {SyncCoverAnchorKind::AfterNode, consumer.node, 0},
       1,
       0,
       {}});
  descriptor.actions.push_back(
      {CanonicalSyncActionKind::EventWait,
       lifecycle.producerResource,
       {SyncCoverAnchorKind::ScopeExit, 0, lifecycle.recurrenceScope},
       1,
       0,
       {}});
  descriptor.supplies.push_back({readyEdge(graph, lifecycle), 0, std::nullopt,
                                 0, 1,
                                 CanonicalSyncSupplyProof::VerifiedProtocol});
  descriptor.supplies.push_back({releaseEdge(graph, lifecycle), 1, std::nullopt,
                                 4, 3,
                                 CanonicalSyncSupplyProof::VerifiedProtocol});
  return descriptor;
}

bool mlir::pto::verifyCanonicalSyncAtomicUnitSlotProtocol(
    const SyncCoverGraph &graph,
    const CanonicalSyncUnitSlotLifecycle &lifecycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  const bool validLifecycle =
      verifyCanonicalSyncUnitSlotLifecycle(graph, lifecycle);
  const bool validCardinality =
      descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
      descriptor.eventUses.size() == 2 && descriptor.actions.size() == 6 &&
      descriptor.supplies.size() == 2;
  if (!validLifecycle || !validCardinality) {
    return false;
  }
  const CanonicalSyncEventUse &readyUse = descriptor.eventUses[0];
  const CanonicalSyncEventUse &releaseUse = descriptor.eventUses[1];
  const SyncCoverStorageAccess &producer =
      graph.getStorageAccesses()[lifecycle.producerAccess];
  const SyncCoverStorageAccess &consumer =
      graph.getStorageAccesses()[lifecycle.consumerAccess];
  const auto readyBinding =
      std::find_if(descriptor.supplies.begin(), descriptor.supplies.end(),
                   [](const CanonicalSyncSupplyBinding &binding) {
                     return binding.eventUse == 0;
                   });
  const auto releaseBinding =
      std::find_if(descriptor.supplies.begin(), descriptor.supplies.end(),
                   [](const CanonicalSyncSupplyBinding &binding) {
                     return binding.eventUse == 1;
                   });
  const bool correctUses =
      readyUse.domain == readyDomain && readyUse.width == 1 &&
      readyUse.recurrenceScope == lifecycle.recurrenceScope &&
      !readyUse.lifetimeScope && releaseUse.domain == releaseDomain &&
      releaseUse.width == 1 &&
      releaseUse.recurrenceScope == lifecycle.recurrenceScope &&
      !releaseUse.lifetimeScope;
  const bool correctBindings =
      readyBinding != descriptor.supplies.end() &&
      releaseBinding != descriptor.supplies.end() &&
      edgeEqual(readyBinding->edge, readyEdge(graph, lifecycle)) &&
      !readyBinding->barrierAction && readyBinding->produceAction == 0 &&
      readyBinding->consumeAction == 1 &&
      readyBinding->proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
      readyBinding->allowedDemands.empty() &&
      edgeEqual(releaseBinding->edge, releaseEdge(graph, lifecycle)) &&
      !releaseBinding->barrierAction && releaseBinding->produceAction == 4 &&
      releaseBinding->consumeAction == 3 &&
      releaseBinding->proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
      releaseBinding->allowedDemands.empty();
  return correctUses && correctBindings &&
         actionMatches(descriptor.actions[0], CanonicalSyncActionKind::EventSet,
                       lifecycle.producerResource,
                       SyncCoverAnchorKind::AfterNode, producer.node, 0, 0) &&
         actionMatches(descriptor.actions[1],
                       CanonicalSyncActionKind::EventWait,
                       lifecycle.consumerResource,
                       SyncCoverAnchorKind::BeforeNode, consumer.node, 0, 0) &&
         actionMatches(descriptor.actions[2], CanonicalSyncActionKind::EventSet,
                       lifecycle.consumerResource,
                       SyncCoverAnchorKind::ScopeEntry, 0,
                       lifecycle.recurrenceScope, 1) &&
         actionMatches(descriptor.actions[3],
                       CanonicalSyncActionKind::EventWait,
                       lifecycle.producerResource,
                       SyncCoverAnchorKind::BeforeNode, producer.node, 0, 1) &&
         actionMatches(descriptor.actions[4], CanonicalSyncActionKind::EventSet,
                       lifecycle.consumerResource,
                       SyncCoverAnchorKind::AfterNode, consumer.node, 0, 1) &&
         actionMatches(
             descriptor.actions[5], CanonicalSyncActionKind::EventWait,
             lifecycle.producerResource, SyncCoverAnchorKind::ScopeExit, 0,
             lifecycle.recurrenceScope, 1);
}
