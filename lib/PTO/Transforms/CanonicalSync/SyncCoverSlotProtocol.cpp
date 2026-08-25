// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverSlotProtocol.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>

using namespace mlir::pto;

namespace {

struct SlotKey {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval extent;

  bool operator<(const SlotKey &other) const {
    return std::tie(domain, extent.begin, extent.end) <
           std::tie(other.domain, other.extent.begin, other.extent.end);
  }
};

bool overlaps(SyncCoverStorageInterval first,
              SyncCoverStorageInterval second) {
  return first.begin < second.end && second.begin < first.end;
}

bool hasAnchor(const SyncCoverResourceAction &action,
               SyncCoverResourceActionKind kind, std::uint32_t resource,
               SyncCoverAnchorKind anchorKind, SyncCoverNodeId node,
               SyncCoverScopeId scope) {
  return action.kind == kind && action.resource == resource &&
         action.anchor.kind == anchorKind && action.anchor.node == node &&
         action.anchor.scope == scope;
}

bool isExactSlotOpportunity(
    const SyncCoverCandidateOpportunity &opportunity,
    const SyncCoverPhysicalSlot &slot) {
  return opportunity.storageProvenance ==
             SyncCoverStorageProvenance::Complete &&
         opportunity.slot && opportunity.slot->domain == slot.domain &&
         opportunity.slot->overlap == slot.extent &&
         opportunity.slot->sourceExtent == slot.extent &&
         opportunity.slot->targetExtent == slot.extent;
}

bool appendRepresentedAccesses(
    const std::vector<SyncCoverCandidateOpportunity> &opportunities,
    const std::vector<SyncCoverCandidateOpportunityId> &ids,
    const SyncCoverPhysicalSlot &slot,
    std::set<SyncCoverStorageAccessId> &represented) {
  for (SyncCoverCandidateOpportunityId id : ids) {
    const bool invalidOpportunity =
        id >= opportunities.size() ||
        !isExactSlotOpportunity(opportunities[id], slot);
    if (invalidOpportunity) {
      return false;
    }
    represented.insert(opportunities[id].slot->sourceAccess);
    represented.insert(opportunities[id].slot->targetAccess);
  }
  return true;
}

bool isPathInsensitive(const SyncCoverGraph &graph,
                       const SyncCoverCandidateOpportunity &opportunity,
                       SyncCoverScopeId recurrenceScope) {
  const bool guarded = !opportunity.sourceGuard.literals.empty() ||
                       !opportunity.targetGuard.literals.empty();
  if (guarded) {
    return false;
  }
  const std::optional<std::size_t> recurrenceDepth =
      graph.getScopeLoopDepth(recurrenceScope);
  const SyncCoverNodeId endpoints[] = {opportunity.source,
                                       opportunity.target};
  for (SyncCoverNodeId endpoint : endpoints) {
    const SyncCoverScopeId scope = graph.getNodes()[endpoint].scope;
    const std::optional<std::size_t> depth = graph.getScopeLoopDepth(scope);
    if (!recurrenceDepth || !depth || *depth != *recurrenceDepth ||
        !graph.scopeMustExecuteWithin(recurrenceScope, scope)) {
      return false;
    }
  }
  return true;
}

bool hasExactLifecycleRoles(
    const SyncCoverGraph &graph,
    const std::vector<SyncCoverCandidateOpportunity> &opportunities,
    const SyncCoverSlotLifecycle &lifecycle, bool requirePathInsensitive) {
  for (SyncCoverCandidateOpportunityId id : lifecycle.ready) {
    if (id >= opportunities.size()) {
      return false;
    }
    const SyncCoverCandidateOpportunity &ready = opportunities[id];
    const bool valid =
        isExactSlotOpportunity(ready, lifecycle.slot) &&
        ready.kind == SyncCoverDemandKind::MemoryRAW && ready.distance == 0 &&
        ready.sourceResource == lifecycle.producerResource &&
        ready.targetResource == lifecycle.consumerResource && ready.slot &&
        syncCoverStorageModeWrites(ready.slot->sourceMode) &&
        ready.slot->targetMode == SyncCoverStorageAccessMode::Read &&
        (!requirePathInsensitive ||
         isPathInsensitive(graph, ready, lifecycle.recurrenceScope));
    if (!valid) {
      return false;
    }
  }
  for (SyncCoverCandidateOpportunityId id : lifecycle.release) {
    if (id >= opportunities.size()) {
      return false;
    }
    const SyncCoverCandidateOpportunity &release = opportunities[id];
    const bool valid =
        isExactSlotOpportunity(release, lifecycle.slot) &&
        release.kind == SyncCoverDemandKind::MemoryWAR &&
        release.distance != 0 && release.distance == lifecycle.distance &&
        release.scope == lifecycle.recurrenceScope &&
        release.sourceResource == lifecycle.consumerResource &&
        release.targetResource == lifecycle.producerResource && release.slot &&
        release.slot->sourceMode == SyncCoverStorageAccessMode::Read &&
        syncCoverStorageModeWrites(release.slot->targetMode) &&
        (!requirePathInsensitive ||
         isPathInsensitive(graph, release, lifecycle.recurrenceScope));
    if (!valid) {
      return false;
    }
  }
  return !lifecycle.ready.empty() && !lifecycle.release.empty();
}

bool matchesReleaseCandidate(
    const std::vector<SyncCoverCandidateOpportunity> &opportunities,
    const SyncCoverSlotLifecycle &lifecycle,
    const SyncCoverSlotProtocolCandidate &candidate) {
  const bool invalid =
      candidate.kind != SyncCoverSlotProtocolKind::UnitRelease ||
      candidate.lifecycle != lifecycle.id ||
      candidate.releases.size() != 1 || candidate.targets.size() != 1 ||
      candidate.releases.front() >= opportunities.size() ||
      std::find(lifecycle.release.begin(), lifecycle.release.end(),
                candidate.releases.front()) == lifecycle.release.end();
  if (invalid) {
    return false;
  }
  const SyncCoverCandidateOpportunity &release =
      opportunities[candidate.releases.front()];
  const bool exactRelease =
      isExactSlotOpportunity(release, lifecycle.slot) &&
      release.kind == SyncCoverDemandKind::MemoryWAR &&
      release.distance == 1 && release.scope == lifecycle.recurrenceScope &&
      release.sourceResource == lifecycle.consumerResource &&
      release.targetResource == lifecycle.producerResource;
  const bool exactCandidate =
      candidate.source == release.source &&
      candidate.targets.front() == release.target &&
      candidate.sourceResource == release.sourceResource &&
      candidate.targetResource == release.targetResource &&
      candidate.recurrenceScope == release.scope &&
      candidate.distance == release.distance;
  return exactRelease && exactCandidate;
}

std::optional<std::vector<SyncCoverStorageAccessId>> getManagedAccesses(
    const SyncCoverGraph &graph, const SyncCoverCandidateIndex &index,
    const SyncCoverPhysicalSlot &slot) {
  const auto domainAccesses = index.getDomainAccesses(slot.domain);
  if (!domainAccesses) {
    return std::nullopt;
  }
  std::vector<SyncCoverStorageAccessId> managed;
  for (SyncCoverStorageAccessId accessId : *domainAccesses.value) {
    if (accessId >= graph.getStorageAccesses().size()) {
      return std::nullopt;
    }
    if (overlaps(graph.getStorageAccesses()[accessId].extent, slot.extent)) {
      managed.push_back(accessId);
    }
  }
  return managed;
}

bool verifyAccessClosure(
    const SyncCoverCandidateIndex &index,
    const SyncCoverSlotLifecycle &lifecycle,
    const std::vector<SyncCoverStorageAccessId> &managed) {
  const auto opportunities = index.getOpportunities();
  if (!opportunities) {
    return false;
  }
  std::set<SyncCoverStorageAccessId> represented;
  if (!appendRepresentedAccesses(*opportunities.value, lifecycle.ready,
                                 lifecycle.slot, represented) ||
      !appendRepresentedAccesses(*opportunities.value, lifecycle.release,
                                 lifecycle.slot, represented)) {
    return false;
  }
  return managed == lifecycle.managedAccesses &&
         std::all_of(managed.begin(), managed.end(), [&](std::size_t access) {
           return represented.find(access) != represented.end();
         });
}

bool hasCompletionTarget(const SyncCoverNode &node, std::uint32_t target) {
  return std::binary_search(node.completionTargets.begin(),
                            node.completionTargets.end(), target);
}

bool verifyStockPrerequisites(
    const SyncCoverGraph &graph,
    const SyncCoverSlotProtocolCandidate &candidate) {
  const bool invalid = candidate.targets.size() != 1 || candidate.targetLoop ||
                       candidate.source >= graph.getNodes().size() ||
                       candidate.targets.front() >= graph.getNodes().size() ||
                       candidate.sourceResource == candidate.targetResource;
  if (invalid) {
    return false;
  }
  const SyncCoverNode &source = graph.getNodes()[candidate.source];
  const SyncCoverNode &target = graph.getNodes()[candidate.targets.front()];
  return source.resource == candidate.sourceResource &&
         target.resource == candidate.targetResource &&
         target.order < source.order &&
         hasCompletionTarget(source, candidate.targetResource);
}

bool verifyCandidateBoundaryClosure(
    const SyncCoverGraph &graph,
    const std::vector<SyncCoverStorageAccessId> &managed,
    const SyncCoverSlotProtocolCandidate &candidate) {
  const SyncCoverNode &releaseSource = graph.getNodes()[candidate.source];
  const SyncCoverNode &releaseTarget =
      graph.getNodes()[candidate.targets.front()];
  for (SyncCoverStorageAccessId accessId : managed) {
    const SyncCoverStorageAccess &access =
        graph.getStorageAccesses()[accessId];
    const SyncCoverNode &node = graph.getNodes()[access.node];
    if (access.mode == SyncCoverStorageAccessMode::Read) {
      if (node.resource != candidate.sourceResource ||
          node.order > releaseSource.order) {
        return false;
      }
      continue;
    }
    if (access.mode == SyncCoverStorageAccessMode::Write) {
      if (node.resource != candidate.targetResource ||
          node.order < releaseTarget.order) {
        return false;
      }
      continue;
    }
    return false;
  }
  return true;
}

std::optional<SyncCoverScopeId> findCommonNestedTargetLoop(
    const SyncCoverGraph &graph, SyncCoverScopeId recurrenceScope,
    const std::vector<SyncCoverNodeId> &targets) {
  std::optional<SyncCoverScopeId> best;
  std::size_t bestDepth = 0;
  for (const SyncCoverScope &scope : graph.getScopes()) {
    if (!scope.isLoop || scope.id == recurrenceScope ||
        !graph.scopeContains(recurrenceScope, scope.id)) {
      continue;
    }
    const bool containsEveryTarget =
        std::all_of(targets.begin(), targets.end(), [&](SyncCoverNodeId target) {
          return target < graph.getNodes().size() &&
                 graph.scopeContains(scope.id,
                                     graph.getNodes()[target].scope);
        });
    const std::optional<std::size_t> depth =
        graph.getScopeLoopDepth(scope.id);
    if (containsEveryTarget && depth && (!best || *depth > bestDepth)) {
      best = scope.id;
      bestDepth = *depth;
    }
  }
  return best;
}

bool verifyHierarchicalAccessBoundary(
    const SyncCoverGraph &graph,
    const std::vector<SyncCoverStorageAccessId> &managed,
    const SyncCoverSlotLifecycle &lifecycle,
    const SyncCoverSlotProtocolCandidate &candidate) {
  const bool invalidCandidate =
      candidate.source >= graph.getNodes().size() ||
      !candidate.targetLoop ||
      *candidate.targetLoop >= graph.getScopes().size();
  if (invalidCandidate) {
    return false;
  }
  const SyncCoverNode &releaseSource = graph.getNodes()[candidate.source];
  if (releaseSource.resource != lifecycle.consumerResource ||
      !releaseSource.guard.literals.empty()) {
    return false;
  }
  std::set<SyncCoverNodeId> producerNodes;
  for (SyncCoverStorageAccessId accessId : managed) {
    if (accessId >= graph.getStorageAccesses().size()) {
      return false;
    }
    const SyncCoverStorageAccess &access =
        graph.getStorageAccesses()[accessId];
    const SyncCoverNode &node = graph.getNodes()[access.node];
    if (node.resource == lifecycle.producerResource) {
      const bool invalidProducerAccess =
          !syncCoverStorageModeWrites(access.mode) ||
          !graph.scopeContains(*candidate.targetLoop, node.scope);
      if (invalidProducerAccess) {
        return false;
      }
      producerNodes.insert(access.node);
      continue;
    }
    if (node.resource != lifecycle.consumerResource ||
        access.mode != SyncCoverStorageAccessMode::Read ||
        access.node != candidate.source) {
      return false;
    }
  }
  const std::set<SyncCoverNodeId> targets(candidate.targets.begin(),
                                          candidate.targets.end());
  return !producerNodes.empty() && producerNodes == targets;
}

bool verifyHierarchicalCandidateShape(
    const SyncCoverGraph &graph,
    const std::vector<SyncCoverCandidateOpportunity> &opportunities,
    const SyncCoverSlotLifecycle &lifecycle,
    const SyncCoverSlotProtocolCandidate &candidate) {
  const bool invalid =
      candidate.kind != SyncCoverSlotProtocolKind::HierarchicalRelease ||
      candidate.lifecycle != lifecycle.id || candidate.distance != 1 ||
      lifecycle.distance != 1 || candidate.releases.empty() ||
      candidate.targets.empty() ||
      candidate.releases.size() != candidate.targets.size() ||
      candidate.source >= graph.getNodes().size() ||
      candidate.sourceResource != lifecycle.consumerResource ||
      candidate.targetResource != lifecycle.producerResource ||
      candidate.recurrenceScope != lifecycle.recurrenceScope ||
      !candidate.targetLoop;
  const std::set<SyncCoverCandidateOpportunityId> distinctReleases(
      candidate.releases.begin(), candidate.releases.end());
  const std::set<SyncCoverNodeId> distinctTargets(candidate.targets.begin(),
                                                   candidate.targets.end());
  const bool invalidIdentity =
      invalid || distinctReleases.size() != candidate.releases.size() ||
      distinctTargets.size() != candidate.targets.size();
  if (invalidIdentity) {
    return false;
  }
  const std::optional<SyncCoverScopeId> targetLoop =
      findCommonNestedTargetLoop(graph, lifecycle.recurrenceScope,
                                 candidate.targets);
  // ScopeEntry is emitted before the nested loop. Its parent must execute on
  // every outer iteration even though the nested loop itself may be zero-trip.
  const bool invalidTargetLoop =
      !targetLoop || !candidate.targetLoop ||
      *targetLoop != *candidate.targetLoop ||
      !graph.getScopes()[*candidate.targetLoop].isLoop ||
      !graph.scopeMustExecuteWithin(
          lifecycle.recurrenceScope,
          graph.getScopes()[*candidate.targetLoop].parent);
  if (invalidTargetLoop) {
    return false;
  }
  for (std::size_t index = 0; index < candidate.releases.size(); ++index) {
    const SyncCoverCandidateOpportunityId releaseId =
        candidate.releases[index];
    const bool missingRelease =
        releaseId >= opportunities.size() ||
        std::find(lifecycle.release.begin(), lifecycle.release.end(),
                  releaseId) == lifecycle.release.end();
    if (missingRelease) {
      return false;
    }
    const SyncCoverCandidateOpportunity &release = opportunities[releaseId];
    const bool mismatch =
        !isExactSlotOpportunity(release, lifecycle.slot) ||
        release.kind != SyncCoverDemandKind::MemoryWAR ||
        release.distance != 1 || release.scope != lifecycle.recurrenceScope ||
        release.source != candidate.source ||
        release.target != candidate.targets[index] || !release.slot ||
        release.slot->sourceMode != SyncCoverStorageAccessMode::Read ||
        !syncCoverStorageModeWrites(release.slot->targetMode);
    if (mismatch) {
      return false;
    }
  }
  const SyncCoverNode &source = graph.getNodes()[candidate.source];
  const std::optional<std::size_t> recurrenceDepth =
      graph.getScopeLoopDepth(candidate.recurrenceScope);
  const std::optional<std::size_t> sourceDepth =
      graph.getScopeLoopDepth(source.scope);
  const std::optional<std::size_t> targetDepth =
      graph.getScopeLoopDepth(*candidate.targetLoop);
  const std::optional<SyncCoverTimelinePosition> sourceAfter =
      resolveSyncCoverAnchor(
          graph, {SyncCoverAnchorKind::AfterNode, candidate.source, 0});
  const std::optional<SyncCoverTimelineInterval> targetTimeline =
      graph.getScopes()[*candidate.targetLoop].timeline;
  return recurrenceDepth && sourceDepth == recurrenceDepth && targetDepth &&
         *targetDepth > *recurrenceDepth &&
         *targetDepth - *recurrenceDepth == 1 && sourceAfter &&
         targetTimeline && targetTimeline->end < *sourceAfter &&
         graph.scopeMustExecuteWithin(candidate.recurrenceScope,
                                      source.scope) &&
         hasCompletionTarget(source, candidate.targetResource);
}

std::optional<SyncCoverSlotProtocolCandidate>
buildHierarchicalCandidate(
    const SyncCoverGraph &graph,
    const std::vector<SyncCoverCandidateOpportunity> &opportunities,
    const SyncCoverSlotLifecycle &lifecycle,
    const std::vector<SyncCoverStorageAccessId> &managed,
    SyncCoverSlotProtocolCandidateId id) {
  if (lifecycle.distance != 1 || lifecycle.release.empty()) {
    return std::nullopt;
  }
  std::map<SyncCoverNodeId, SyncCoverCandidateOpportunityId> targetReleases;
  std::optional<SyncCoverNodeId> source;
  for (SyncCoverCandidateOpportunityId releaseId : lifecycle.release) {
    if (releaseId >= opportunities.size()) {
      return std::nullopt;
    }
    const SyncCoverCandidateOpportunity &release = opportunities[releaseId];
    const bool invalid =
        !isExactSlotOpportunity(release, lifecycle.slot) ||
        release.kind != SyncCoverDemandKind::MemoryWAR ||
        release.distance != 1 || release.scope != lifecycle.recurrenceScope ||
        release.sourceResource != lifecycle.consumerResource ||
        release.targetResource != lifecycle.producerResource || !release.slot ||
        release.slot->sourceMode != SyncCoverStorageAccessMode::Read ||
        !syncCoverStorageModeWrites(release.slot->targetMode);
    if (invalid || (source && *source != release.source)) {
      return std::nullopt;
    }
    source = release.source;
    targetReleases.emplace(release.target, releaseId);
  }
  if (!source || targetReleases.empty()) {
    return std::nullopt;
  }
  std::vector<SyncCoverNodeId> targets;
  std::vector<SyncCoverCandidateOpportunityId> releases;
  targets.reserve(targetReleases.size());
  releases.reserve(targetReleases.size());
  for (const auto &[target, release] : targetReleases) {
    targets.push_back(target);
    releases.push_back(release);
  }
  const std::optional<SyncCoverScopeId> targetLoop =
      findCommonNestedTargetLoop(graph, lifecycle.recurrenceScope, targets);
  if (!targetLoop) {
    return std::nullopt;
  }
  SyncCoverSlotProtocolCandidate candidate;
  candidate.id = id;
  candidate.kind = SyncCoverSlotProtocolKind::HierarchicalRelease;
  candidate.lifecycle = lifecycle.id;
  candidate.releases = std::move(releases);
  candidate.source = *source;
  candidate.targets = std::move(targets);
  candidate.sourceResource = lifecycle.consumerResource;
  candidate.targetResource = lifecycle.producerResource;
  candidate.recurrenceScope = lifecycle.recurrenceScope;
  candidate.targetLoop = *targetLoop;
  candidate.distance = 1;
  if (!verifyHierarchicalCandidateShape(graph, opportunities, lifecycle,
                                        candidate) ||
      !verifyHierarchicalAccessBoundary(graph, managed, lifecycle, candidate)) {
    return std::nullopt;
  }
  return candidate;
}

bool addInspectionUnits(std::size_t count, std::size_t multiplier,
                        std::size_t &total) {
  const bool productOverflows =
      multiplier != 0 && count > std::numeric_limits<std::size_t>::max() /
                                     multiplier;
  if (productOverflows) {
    return false;
  }
  const std::size_t increment = count * multiplier;
  const bool sumOverflows =
      increment > std::numeric_limits<std::size_t>::max() - total;
  if (sumOverflows) {
    return false;
  }
  total += increment;
  return true;
}

std::optional<std::size_t> getInspectionUnits(
    const SyncCoverSlotLifecycle &lifecycle, std::size_t domainAccesses) {
  std::size_t total = 1;
  const bool commonFits =
      addInspectionUnits(lifecycle.ready.size(), 2, total) &&
      addInspectionUnits(lifecycle.release.size(), 4, total) &&
      addInspectionUnits(domainAccesses, 1, total) &&
      addInspectionUnits(lifecycle.managedAccesses.size(), 3, total);
  if (!commonFits) {
    return std::nullopt;
  }
  if (!lifecycle.requiresPathSensitiveProof &&
      !addInspectionUnits(lifecycle.managedAccesses.size(),
                          lifecycle.release.size(), total)) {
    return std::nullopt;
  }
  return total;
}

} // namespace

bool mlir::pto::verifySyncCoverSlotProtocolCandidate(
    const SyncCoverGraph &graph, const SyncCoverCandidateIndex &index,
    const SyncCoverSlotLifecycle &lifecycle,
    const SyncCoverSlotProtocolCandidate &candidate) {
  const auto opportunities = index.getOpportunities();
  const bool invalidContext =
      !graph.isStructureFrozen() || !graph.validate() ||
      !index.isCurrentFor(graph) || !opportunities;
  if (invalidContext) {
    return false;
  }
  const auto managed = getManagedAccesses(graph, index, lifecycle.slot);
  if (!managed) {
    return false;
  }
  if (lifecycle.hasUnrepresentedAccesses ||
      !verifyAccessClosure(index, lifecycle, *managed)) {
    return false;
  }
  if (candidate.kind == SyncCoverSlotProtocolKind::HierarchicalRelease) {
    return hasExactLifecycleRoles(graph, *opportunities.value, lifecycle,
                                  false) &&
           verifyHierarchicalCandidateShape(
               graph, *opportunities.value, lifecycle, candidate) &&
           verifyHierarchicalAccessBoundary(graph, *managed, lifecycle,
                                            candidate);
  }
  return matchesReleaseCandidate(*opportunities.value, lifecycle, candidate) &&
         lifecycle.distance == 1 &&
         !lifecycle.requiresPathSensitiveProof &&
         hasExactLifecycleRoles(graph, *opportunities.value, lifecycle, true) &&
         verifyStockPrerequisites(graph, candidate) &&
         verifyCandidateBoundaryClosure(graph, *managed, candidate);
}

std::optional<SyncCoverMechanismDescriptor>
mlir::pto::makeSyncCoverSlotProtocolDescriptor(
    const SyncCoverResourceDomain &domain,
    const SyncCoverSlotProtocolCandidate &candidate,
    std::uint64_t providerIdentity) {
  const bool invalidUnit =
      candidate.kind == SyncCoverSlotProtocolKind::UnitRelease &&
      candidate.targets.size() != 1;
  if (invalidUnit) {
    return std::nullopt;
  }
  if (candidate.kind == SyncCoverSlotProtocolKind::UnitRelease) {
    return makeSyncCoverUnitRecurrenceEvent(
        domain, candidate.source, candidate.targets.front(),
        candidate.recurrenceScope, providerIdentity);
  }
  if (candidate.kind != SyncCoverSlotProtocolKind::HierarchicalRelease ||
      domain.kind != SyncCoverResourceKind::EventId ||
      domain.sourceResource != candidate.sourceResource ||
      domain.targetResource != candidate.targetResource ||
      candidate.releases.empty() || candidate.targets.empty() ||
      candidate.releases.size() != candidate.targets.size() ||
      !candidate.targetLoop) {
    return std::nullopt;
  }
  SyncCoverMechanismDescriptorBuilder builder(
      SyncCoverMechanismKind::VerifiedProtocol, providerIdentity);
  const SyncCoverDescriptorActionRef prime = builder.addAction(
      SyncCoverResourceActionKind::Produce, domain.sourceResource,
      {SyncCoverAnchorKind::ScopeEntry, 0, candidate.recurrenceScope});
  const SyncCoverDescriptorActionRef bodyWait = builder.addAction(
      SyncCoverResourceActionKind::Consume, domain.targetResource,
      {SyncCoverAnchorKind::ScopeEntry, 0, *candidate.targetLoop});
  const SyncCoverDescriptorActionRef bodySet = builder.addAction(
      SyncCoverResourceActionKind::Produce, domain.sourceResource,
      {SyncCoverAnchorKind::AfterNode, candidate.source, 0});
  const SyncCoverDescriptorActionRef drain = builder.addAction(
      SyncCoverResourceActionKind::Consume, domain.targetResource,
      {SyncCoverAnchorKind::ScopeExit, 0, candidate.recurrenceScope});
  std::vector<SyncCoverProtocolSupply> supplies;
  supplies.reserve(candidate.targets.size());
  for (SyncCoverNodeId target : candidate.targets) {
    SyncCoverEdge edge;
    edge.source = candidate.source;
    edge.target = target;
    edge.kind = SyncCoverEdgeKind::CompletionSupply;
    edge.scope = candidate.recurrenceScope;
    edge.distance = 1;
    supplies.push_back({edge, bodySet, bodyWait});
  }
  if (!builder.addProtocolLane(domain, candidate.recurrenceScope, 1, 1,
                               {prime, bodyWait, bodySet, drain},
                               std::move(supplies))) {
    return std::nullopt;
  }
  return std::move(builder).takeDescriptor();
}

bool mlir::pto::verifySyncCoverSlotProtocol(
    const SyncCoverCandidateIndex &index,
    const SyncCoverSlotLifecycle &lifecycle,
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverSlotProtocolCandidate &candidate,
    const SyncCoverMechanismDescriptor &descriptor) {
  const SyncCoverGraph &graph = universe.getGraph();
  if (!verifySyncCoverSlotProtocolCandidate(graph, index, lifecycle,
                                            candidate)) {
    return false;
  }
  if (candidate.kind == SyncCoverSlotProtocolKind::UnitRelease) {
    return verifySyncCoverUnitRecurrenceEvent(universe, descriptor);
  }
  if (candidate.kind != SyncCoverSlotProtocolKind::HierarchicalRelease) {
    return false;
  }
  const bool invalidCardinality =
      descriptor.kind != SyncCoverMechanismKind::VerifiedProtocol ||
      descriptor.barrier || descriptor.actions.size() != 4 ||
      descriptor.resourceUses.size() != 1 ||
      descriptor.supplyEdges.size() != candidate.targets.size() ||
      descriptor.supplyBindings.size() != candidate.targets.size();
  if (invalidCardinality) {
    return false;
  }
  const SyncCoverResourceUse &use = descriptor.resourceUses.front();
  if (use.domain >= universe.getResourceDomains().size()) {
    return false;
  }
  const SyncCoverResourceDomain &domain =
      universe.getResourceDomains()[use.domain];
  const bool invalidUse =
      domain.kind != SyncCoverResourceKind::EventId ||
      domain.sourceResource != candidate.sourceResource ||
      domain.targetResource != candidate.targetResource ||
      use.scope != candidate.recurrenceScope || use.distance != 1 ||
      use.width != 1 || use.actions != std::vector<std::size_t>({0, 1, 2, 3});
  const bool invalidResourceUse =
      invalidUse || use.supplyEdges.size() != candidate.targets.size();
  if (invalidResourceUse) {
    return false;
  }
  for (std::size_t index = 0; index < candidate.targets.size(); ++index) {
    const bool invalidEdgeIndex = use.supplyEdges[index] != index;
    const SyncCoverEdge &edge = descriptor.supplyEdges[index];
    const SyncCoverSupplyBinding &binding = descriptor.supplyBindings[index];
    const bool invalidEdge =
        edge.source != candidate.source ||
        edge.target != candidate.targets[index] ||
        edge.kind != SyncCoverEdgeKind::CompletionSupply || edge.mechanism ||
        edge.scope != candidate.recurrenceScope || edge.distance != 1;
    const bool invalidBinding =
        binding.supplyEdge != index || binding.resourceUse != 0 ||
        binding.produceAction != 2 || binding.consumeAction != 1;
    if (invalidEdgeIndex || invalidEdge || invalidBinding) {
      return false;
    }
  }
  const auto &actions = descriptor.actions;
  return hasAnchor(actions[0], SyncCoverResourceActionKind::Produce,
                   candidate.sourceResource, SyncCoverAnchorKind::ScopeEntry,
                   0, candidate.recurrenceScope) &&
         hasAnchor(actions[1], SyncCoverResourceActionKind::Consume,
                   candidate.targetResource, SyncCoverAnchorKind::ScopeEntry,
                   0, *candidate.targetLoop) &&
         hasAnchor(actions[2], SyncCoverResourceActionKind::Produce,
                   candidate.sourceResource, SyncCoverAnchorKind::AfterNode,
                   candidate.source, 0) &&
         hasAnchor(actions[3], SyncCoverResourceActionKind::Consume,
                   candidate.targetResource, SyncCoverAnchorKind::ScopeExit,
                   0, candidate.recurrenceScope);
}

SyncCoverSlotProtocolResult mlir::pto::buildSyncCoverSlotProtocolCandidates(
    const SyncCoverGraph &graph, const SyncCoverCandidateIndex &index,
    const SyncCoverSlotLifecycleResult &lifecycles,
    const SyncCoverSlotProtocolOptions &options) {
  SyncCoverSlotProtocolResult result;
  const bool invalidGraph = !graph.isStructureFrozen() || !graph.validate();
  if (invalidGraph) {
    result.error = SyncCoverSlotProtocolError::InvalidGraph;
    return result;
  }
  const auto opportunities = index.getOpportunities();
  const bool invalidIndex = !index.isCurrentFor(graph) || !opportunities;
  if (invalidIndex) {
    result.error = SyncCoverSlotProtocolError::InvalidCandidateIndex;
    return result;
  }
  if (!lifecycles) {
    result.error = SyncCoverSlotProtocolError::InvalidLifecycle;
    return result;
  }
  result.truncated = lifecycles.truncated;
  result.partialSlotOpportunities = lifecycles.partialSlotOpportunities;
  std::map<SlotKey, std::vector<SyncCoverStorageAccessId>> managedCache;
  for (std::size_t lifecycleIndex = 0;
       lifecycleIndex < lifecycles.lifecycles.size(); ++lifecycleIndex) {
    const bool candidateCapReached =
        result.candidates.size() == options.maximumCandidates;
    if (candidateCapReached) {
      result.truncated = true;
      return result;
    }
    const SyncCoverSlotLifecycle &lifecycle =
        lifecycles.lifecycles[lifecycleIndex];
    if (lifecycle.id != lifecycleIndex) {
      result.error = SyncCoverSlotProtocolError::InvalidLifecycle;
      return result;
    }
    const auto domainAccesses =
        index.getDomainAccesses(lifecycle.slot.domain);
    if (!domainAccesses) {
      result.error = SyncCoverSlotProtocolError::InvalidCandidateIndex;
      return result;
    }
    const std::optional<std::size_t> inspectionUnits = getInspectionUnits(
        lifecycle, domainAccesses.value->size());
    const std::size_t remainingEvaluations =
        options.maximumEvaluations - result.evaluations;
    if (!inspectionUnits || *inspectionUnits > remainingEvaluations) {
      result.truncated = true;
      return result;
    }
    result.evaluations += *inspectionUnits;
    if (!hasExactLifecycleRoles(graph, *opportunities.value, lifecycle,
                                !lifecycle.requiresPathSensitiveProof)) {
      ++result.unsupportedEffectLifecycles;
      continue;
    }
    const SlotKey slotKey{lifecycle.slot.domain, lifecycle.slot.extent};
    auto managed = managedCache.find(slotKey);
    if (managed == managedCache.end()) {
      const auto discovered = getManagedAccesses(graph, index, lifecycle.slot);
      if (!discovered) {
        result.error = SyncCoverSlotProtocolError::InvalidCandidateIndex;
        return result;
      }
      managed = managedCache.emplace(slotKey, *discovered).first;
    }
    const bool accessOpen =
        lifecycle.hasUnrepresentedAccesses ||
        !verifyAccessClosure(index, lifecycle, managed->second);
    if (accessOpen) {
      ++result.accessOpenLifecycles;
      continue;
    }
    if (lifecycle.requiresPathSensitiveProof) {
      const std::optional<SyncCoverSlotProtocolCandidate> hierarchical =
          buildHierarchicalCandidate(
              graph, *opportunities.value, lifecycle, managed->second,
              result.candidates.size());
      if (!hierarchical) {
        ++result.pathSensitiveLifecycles;
        continue;
      }
      const bool capReached =
          result.candidates.size() == options.maximumCandidates;
      if (capReached) {
        result.truncated = true;
        return result;
      }
      result.candidates.push_back(*hierarchical);
      continue;
    }
    for (SyncCoverCandidateOpportunityId releaseId : lifecycle.release) {
      if (releaseId >= opportunities.value->size()) {
        result.error = SyncCoverSlotProtocolError::InvalidLifecycle;
        return result;
      }
      const SyncCoverCandidateOpportunity &release =
          (*opportunities.value)[releaseId];
      if (release.distance != 1) {
        ++result.unsupportedDistanceReleases;
        continue;
      }
      const bool capReached =
          result.candidates.size() == options.maximumCandidates;
      if (capReached) {
        result.truncated = true;
        return result;
      }
      SyncCoverSlotProtocolCandidate candidate;
      candidate.id = result.candidates.size();
      candidate.kind = SyncCoverSlotProtocolKind::UnitRelease;
      candidate.lifecycle = lifecycle.id;
      candidate.releases = {releaseId};
      candidate.source = release.source;
      candidate.targets = {release.target};
      candidate.sourceResource = release.sourceResource;
      candidate.targetResource = release.targetResource;
      candidate.recurrenceScope = release.scope;
      candidate.distance = release.distance;
      if (!matchesReleaseCandidate(*opportunities.value, lifecycle,
                                   candidate)) {
        result.error = SyncCoverSlotProtocolError::InvalidLifecycle;
        return result;
      }
      const bool completeBoundary =
          verifyStockPrerequisites(graph, candidate) &&
          verifyCandidateBoundaryClosure(graph, managed->second, candidate);
      if (!completeBoundary) {
        ++result.nonBoundaryReleases;
        continue;
      }
      result.candidates.push_back(candidate);
    }
  }
  return result;
}
