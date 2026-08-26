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
      candidate.sources != std::vector<SyncCoverNodeId>({candidate.source}) ||
      candidate.completionEdges !=
          std::vector<std::pair<SyncCoverNodeId, SyncCoverNodeId>>(
              {{candidate.source, candidate.targets.front()}}) ||
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

bool guardMatches(const SyncCoverGuard &guard,
                  const std::map<SyncCoverControlId, unsigned> &assignment) {
  return std::all_of(
      guard.literals.begin(), guard.literals.end(),
      [&](const SyncCoverGuardLiteral &literal) {
        auto value = assignment.find(literal.control);
        return value != assignment.end() && value->second == literal.alternative;
      });
}

bool verifyStockPrerequisites(
    const SyncCoverGraph &graph,
    const SyncCoverSlotProtocolCandidate &candidate) {
  const bool invalid = candidate.targets.size() != 1 ||
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

struct GuardedLifecyclePath {
  std::vector<SyncCoverNodeId> sources;
  SyncCoverNodeId waitTarget = 0;
};

struct GuardedLifecycleLayout {
  unsigned width = 0;
  std::map<SyncCoverNodeId, unsigned> sourceLanes;
  std::map<SyncCoverNodeId, SyncCoverNodeId> targetWaits;
  std::vector<GuardedLifecyclePath> paths;
};

std::optional<SyncCoverScopeId> findCommonNestedTargetLoop(
    const SyncCoverGraph &graph, SyncCoverScopeId recurrenceScope,
    const std::vector<SyncCoverNodeId> &targets) {
  std::optional<SyncCoverScopeId> result;
  std::size_t resultDepth = 0;
  for (const SyncCoverScope &scope : graph.getScopes()) {
    if (!scope.isLoop || scope.id == recurrenceScope ||
        !graph.scopeContains(recurrenceScope, scope.id)) {
      continue;
    }
    const bool containsEveryTarget = std::all_of(
        targets.begin(), targets.end(), [&](SyncCoverNodeId target) {
          return target < graph.getNodes().size() &&
                 graph.scopeContains(scope.id, graph.getNodes()[target].scope);
        });
    const std::optional<std::size_t> depth =
        graph.getScopeLoopDepth(scope.id);
    if (containsEveryTarget && depth && (!result || *depth > resultDepth)) {
      result = scope.id;
      resultDepth = *depth;
    }
  }
  return result;
}

std::optional<GuardedLifecycleLayout> deriveGuardedLifecycleLayout(
    const SyncCoverGraph &graph, SyncCoverScopeId recurrenceScope,
    const std::vector<SyncCoverNodeId> &sources,
    const std::vector<SyncCoverNodeId> &targets) {
  if (sources.empty() || targets.empty()) {
    return std::nullopt;
  }
  const std::optional<std::size_t> recurrenceDepth =
      graph.getScopeLoopDepth(recurrenceScope);
  if (!recurrenceDepth) {
    return std::nullopt;
  }
  std::vector<SyncCoverNodeId> endpoints = sources;
  endpoints.insert(endpoints.end(), targets.begin(), targets.end());
  std::vector<SyncCoverGuard> guards;
  guards.reserve(endpoints.size());
  for (SyncCoverNodeId endpoint : endpoints) {
    if (endpoint >= graph.getNodes().size()) {
      return std::nullopt;
    }
    guards.push_back(graph.getNodes()[endpoint].guard);
  }

  std::vector<SyncCoverGuardLiteral> common = guards.front().literals;
  for (const SyncCoverGuard &guard : guards) {
    std::vector<SyncCoverGuardLiteral> intersection;
    std::set_intersection(common.begin(), common.end(),
                          guard.literals.begin(), guard.literals.end(),
                          std::back_inserter(intersection));
    common = std::move(intersection);
  }
  std::vector<SyncCoverControlId> controls;
  for (SyncCoverGuard &guard : guards) {
    std::vector<SyncCoverGuardLiteral> reduced;
    std::set_difference(guard.literals.begin(), guard.literals.end(),
                        common.begin(), common.end(),
                        std::back_inserter(reduced));
    guard.literals = std::move(reduced);
    for (const SyncCoverGuardLiteral &literal : guard.literals) {
      controls.push_back(literal.control);
    }
  }
  std::sort(controls.begin(), controls.end());
  controls.erase(std::unique(controls.begin(), controls.end()), controls.end());

  constexpr std::size_t kMaximumAssignments = 4096;
  std::size_t assignmentCount = 1;
  for (SyncCoverControlId control : controls) {
    if (control >= graph.getControls().size()) {
      return std::nullopt;
    }
    const unsigned alternatives = graph.getControls()[control].alternatives;
    if (alternatives == 0 ||
        assignmentCount > kMaximumAssignments / alternatives) {
      return std::nullopt;
    }
    assignmentCount *= alternatives;
  }

  GuardedLifecycleLayout result;
  std::set<std::pair<std::vector<SyncCoverNodeId>, SyncCoverNodeId>>
      uniquePaths;
  for (std::size_t ordinal = 0; ordinal < assignmentCount; ++ordinal) {
    std::size_t remaining = ordinal;
    std::map<SyncCoverControlId, unsigned> assignment;
    for (SyncCoverControlId control : controls) {
      const unsigned alternatives = graph.getControls()[control].alternatives;
      assignment.emplace(control,
                         static_cast<unsigned>(remaining % alternatives));
      remaining /= alternatives;
    }
    std::vector<SyncCoverNodeId> pathSources;
    std::vector<SyncCoverNodeId> pathTargets;
    for (std::size_t index = 0; index < sources.size(); ++index) {
      if (guardMatches(guards[index], assignment)) {
        pathSources.push_back(sources[index]);
      }
    }
    for (std::size_t index = 0; index < targets.size(); ++index) {
      if (guardMatches(guards[sources.size() + index], assignment)) {
        pathTargets.push_back(targets[index]);
      }
    }
    if (pathSources.empty() || pathTargets.empty()) {
      return std::nullopt;
    }
    std::sort(pathSources.begin(), pathSources.end(),
              [&](SyncCoverNodeId first, SyncCoverNodeId second) {
                return std::tie(graph.getNodes()[first].order, first) <
                       std::tie(graph.getNodes()[second].order, second);
              });
    const bool endpointsAreLocal =
        std::all_of(pathSources.begin(), pathSources.end(),
                    [&](SyncCoverNodeId source) {
                      return graph.getScopeLoopDepth(
                                 graph.getNodes()[source].scope) ==
                                 recurrenceDepth &&
                             graph.scopeContains(
                                 recurrenceScope,
                                 graph.getNodes()[source].scope);
                    }) &&
        std::all_of(pathTargets.begin(), pathTargets.end(),
                    [&](SyncCoverNodeId target) {
                      return graph.getScopeLoopDepth(
                                 graph.getNodes()[target].scope) ==
                                 recurrenceDepth &&
                             graph.scopeContains(
                                 recurrenceScope,
                                 graph.getNodes()[target].scope);
                    });
    if (!endpointsAreLocal) {
      return std::nullopt;
    }
    std::sort(pathTargets.begin(), pathTargets.end(),
              [&](SyncCoverNodeId first, SyncCoverNodeId second) {
                return std::tie(graph.getNodes()[first].order, first) <
                       std::tie(graph.getNodes()[second].order, second);
              });
    const SyncCoverNodeId waitTarget = pathTargets.front();
    const bool waitPrecedesReleases = std::all_of(
        pathSources.begin(), pathSources.end(), [&](SyncCoverNodeId source) {
          return graph.getNodes()[waitTarget].order <
                 graph.getNodes()[source].order;
        });
    if (!waitPrecedesReleases) {
      return std::nullopt;
    }
    if (result.width == 0) {
      result.width = static_cast<unsigned>(pathSources.size());
    }
    if (pathSources.size() != result.width) {
      return std::nullopt;
    }
    for (std::size_t lane = 0; lane < pathSources.size(); ++lane) {
      auto [entry, inserted] = result.sourceLanes.emplace(
          pathSources[lane], static_cast<unsigned>(lane));
      if (!inserted && entry->second != lane) {
        return std::nullopt;
      }
    }
    for (SyncCoverNodeId target : pathTargets) {
      auto [entry, inserted] = result.targetWaits.emplace(target, waitTarget);
      if (!inserted && entry->second != waitTarget) {
        return std::nullopt;
      }
    }
    if (uniquePaths.emplace(pathSources, waitTarget).second) {
      result.paths.push_back({std::move(pathSources), waitTarget});
    }
  }
  if (result.width == 0 || result.sourceLanes.size() != sources.size() ||
      result.targetWaits.size() != targets.size()) {
    return std::nullopt;
  }
  return result;
}

bool verifyHierarchicalAccessBoundary(
    const SyncCoverGraph &graph,
    const std::vector<SyncCoverStorageAccessId> &managed,
    const SyncCoverSlotLifecycle &lifecycle,
    const SyncCoverSlotProtocolCandidate &candidate) {
  const bool invalidCandidate = candidate.sources.empty() ||
      candidate.source != candidate.sources.front() ||
      candidate.sourceLanes.size() != candidate.sources.size() ||
      candidate.targetWaits.size() != candidate.targets.size();
  if (invalidCandidate) {
    return false;
  }
  const std::set<SyncCoverNodeId> sourceNodes(candidate.sources.begin(),
                                              candidate.sources.end());
  std::set<SyncCoverNodeId> producerNodes;
  for (SyncCoverStorageAccessId accessId : managed) {
    if (accessId >= graph.getStorageAccesses().size()) {
      return false;
    }
    const SyncCoverStorageAccess &access =
        graph.getStorageAccesses()[accessId];
    const SyncCoverNode &node = graph.getNodes()[access.node];
    if (node.resource == lifecycle.producerResource) {
      const auto target = std::lower_bound(candidate.targets.begin(),
                                           candidate.targets.end(), access.node);
      const std::size_t targetIndex =
          static_cast<std::size_t>(target - candidate.targets.begin());
      bool waitPrecedesAccess = false;
      if (candidate.waitScope) {
        waitPrecedesAccess =
            *candidate.waitScope < graph.getScopes().size() &&
            graph.scopeContains(*candidate.waitScope, node.scope);
      } else if (targetIndex < candidate.targetWaits.size() &&
                 candidate.targetWaits[targetIndex] < graph.getNodes().size()) {
        waitPrecedesAccess =
            graph.getNodes()[candidate.targetWaits[targetIndex]].order <=
            node.order;
      }
      const bool invalidProducerAccess =
          !syncCoverStorageModeWrites(access.mode) ||
          target == candidate.targets.end() || *target != access.node ||
          !waitPrecedesAccess;
      if (invalidProducerAccess) {
        return false;
      }
      producerNodes.insert(access.node);
      continue;
    }
    const bool isDeclaredConsumer =
        sourceNodes.find(access.node) != sourceNodes.end();
    if (node.resource != lifecycle.consumerResource ||
        access.mode != SyncCoverStorageAccessMode::Read ||
        !isDeclaredConsumer) {
      return false;
    }
  }
  const std::set<SyncCoverNodeId> targets(candidate.targets.begin(),
                                          candidate.targets.end());
  if (producerNodes.empty() || producerNodes != targets) {
    return false;
  }
  if (candidate.waitScope) {
    return candidate.sources.size() == 1 &&
           graph.getNodes()[candidate.source].guard.literals.empty();
  }
  return deriveGuardedLifecycleLayout(graph, candidate.recurrenceScope,
                                      candidate.sources, candidate.targets)
      .has_value();
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
      candidate.sources.empty() || candidate.targets.empty() ||
      candidate.source != candidate.sources.front() ||
      candidate.sourceLanes.size() != candidate.sources.size() ||
      candidate.targetWaits.size() != candidate.targets.size() ||
      candidate.paths.empty() ||
      candidate.sourceResource != lifecycle.consumerResource ||
      candidate.targetResource != lifecycle.producerResource ||
      candidate.recurrenceScope != lifecycle.recurrenceScope ||
      candidate.width == 0;
  const std::set<SyncCoverCandidateOpportunityId> distinctReleases(
      candidate.releases.begin(), candidate.releases.end());
  const std::set<SyncCoverNodeId> distinctSources(candidate.sources.begin(),
                                                   candidate.sources.end());
  const std::set<SyncCoverNodeId> distinctTargets(candidate.targets.begin(),
                                                   candidate.targets.end());
  const std::set<std::pair<SyncCoverNodeId, SyncCoverNodeId>>
      declaredCompletions(candidate.completionEdges.begin(),
                          candidate.completionEdges.end());
  const bool invalidIdentity =
      invalid || distinctReleases.size() != candidate.releases.size() ||
      distinctSources.size() != candidate.sources.size() ||
      distinctTargets.size() != candidate.targets.size() ||
      declaredCompletions.size() != candidate.completionEdges.size();
  if (invalidIdentity) {
    return false;
  }
  std::optional<GuardedLifecycleLayout> layout;
  if (candidate.waitScope) {
    const std::optional<SyncCoverScopeId> targetLoop =
        findCommonNestedTargetLoop(graph, lifecycle.recurrenceScope,
                                   candidate.targets);
    const bool invalidNestedLayout =
        !targetLoop || *targetLoop != *candidate.waitScope ||
        candidate.sources.size() != 1 || candidate.width != 1 ||
        candidate.sourceLanes != std::vector<unsigned>{0} ||
        candidate.paths.size() != 1 ||
        candidate.paths.front().sources != candidate.sources ||
        candidate.targetWaits !=
            std::vector<SyncCoverNodeId>(candidate.targets.size(),
                                         candidate.targets.front()) ||
        candidate.paths.front().waitTarget != candidate.targets.front();
    if (invalidNestedLayout) {
      return false;
    }
  } else {
    layout = deriveGuardedLifecycleLayout(graph, lifecycle.recurrenceScope,
                                          candidate.sources,
                                          candidate.targets);
    if (!layout || layout->width != candidate.width ||
        candidate.paths.size() != layout->paths.size()) {
      return false;
    }
    for (std::size_t index = 0; index < candidate.paths.size(); ++index) {
      if (candidate.paths[index].sources != layout->paths[index].sources ||
          candidate.paths[index].waitTarget !=
              layout->paths[index].waitTarget) {
        return false;
      }
    }
    for (std::size_t index = 0; index < candidate.sources.size(); ++index) {
      const auto lane = layout->sourceLanes.find(candidate.sources[index]);
      if (lane == layout->sourceLanes.end() ||
          lane->second != candidate.sourceLanes[index]) {
        return false;
      }
    }
    for (std::size_t index = 0; index < candidate.targets.size(); ++index) {
      const auto wait = layout->targetWaits.find(candidate.targets[index]);
      if (wait == layout->targetWaits.end() ||
          wait->second != candidate.targetWaits[index]) {
        return false;
      }
    }
  }
  std::set<std::pair<SyncCoverNodeId, SyncCoverNodeId>> releasePairs;
  for (SyncCoverCandidateOpportunityId releaseId : candidate.releases) {
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
        distinctSources.find(release.source) == distinctSources.end() ||
        distinctTargets.find(release.target) == distinctTargets.end() ||
        !release.slot ||
        release.slot->sourceMode != SyncCoverStorageAccessMode::Read ||
        !syncCoverStorageModeWrites(release.slot->targetMode);
    if (mismatch) {
      return false;
    }
    releasePairs.emplace(release.source, release.target);
  }
  const std::optional<std::size_t> recurrenceDepth =
      graph.getScopeLoopDepth(candidate.recurrenceScope);
  if (!recurrenceDepth) {
    return false;
  }
  for (SyncCoverNodeId sourceId : candidate.sources) {
    if (sourceId >= graph.getNodes().size()) {
      return false;
    }
    const SyncCoverNode &source = graph.getNodes()[sourceId];
    const std::optional<std::size_t> sourceDepth =
        graph.getScopeLoopDepth(source.scope);
    const std::optional<SyncCoverTimelinePosition> sourceAfter =
        resolveSyncCoverAnchor(
            graph, {SyncCoverAnchorKind::AfterNode, sourceId, 0});
    if (!sourceAfter) {
      return false;
    }
    bool followsOneWait = false;
    if (candidate.waitScope) {
      const SyncCoverScope &waitScope = graph.getScopes()[*candidate.waitScope];
      const std::optional<std::size_t> waitDepth =
          graph.getScopeLoopDepth(*candidate.waitScope);
      followsOneWait = waitScope.timeline && waitDepth && recurrenceDepth &&
                       *waitDepth == *recurrenceDepth + 1 &&
                       waitScope.timeline->end < *sourceAfter &&
                       graph.scopeMustExecuteWithin(
                           candidate.recurrenceScope, waitScope.parent);
    } else {
      followsOneWait = std::any_of(
          layout->paths.begin(), layout->paths.end(),
          [&](const GuardedLifecyclePath &path) {
            if (std::find(path.sources.begin(), path.sources.end(), sourceId) ==
                path.sources.end()) {
              return false;
            }
            const std::optional<SyncCoverTimelinePosition> waitBefore =
                resolveSyncCoverAnchor(
                    graph,
                    {SyncCoverAnchorKind::BeforeNode, path.waitTarget, 0});
            return waitBefore && *waitBefore < *sourceAfter;
          });
    }
    if (source.resource != lifecycle.consumerResource ||
        sourceDepth != recurrenceDepth || !followsOneWait ||
        !hasCompletionTarget(source, candidate.targetResource)) {
      return false;
    }
  }
  return releasePairs.size() == candidate.releases.size() &&
         releasePairs == declaredCompletions;
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
  std::map<std::pair<SyncCoverNodeId, SyncCoverNodeId>,
           SyncCoverCandidateOpportunityId>
      releasePairs;
  std::set<SyncCoverNodeId> sourceSet;
  std::set<SyncCoverNodeId> targetSet;
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
    if (invalid) {
      return std::nullopt;
    }
    sourceSet.insert(release.source);
    targetSet.insert(release.target);
    releasePairs.emplace(std::make_pair(release.source, release.target),
                         releaseId);
  }
  if (sourceSet.empty() || targetSet.empty() || releasePairs.empty()) {
    return std::nullopt;
  }
  std::vector<SyncCoverNodeId> sources(sourceSet.begin(), sourceSet.end());
  std::vector<SyncCoverNodeId> targets(targetSet.begin(), targetSet.end());
  std::vector<SyncCoverCandidateOpportunityId> releases;
  releases.reserve(releasePairs.size());
  for (const auto &[endpoints, release] : releasePairs) {
    (void)endpoints;
    releases.push_back(release);
  }
  SyncCoverSlotProtocolCandidate candidate;
  candidate.id = id;
  candidate.kind = SyncCoverSlotProtocolKind::HierarchicalRelease;
  candidate.lifecycle = lifecycle.id;
  candidate.releases = std::move(releases);
  for (const auto &[endpoints, release] : releasePairs) {
    (void)release;
    candidate.completionEdges.push_back(endpoints);
  }
  candidate.sources = std::move(sources);
  candidate.source = candidate.sources.front();
  candidate.targets = std::move(targets);
  const std::optional<SyncCoverScopeId> targetLoop =
      findCommonNestedTargetLoop(graph, lifecycle.recurrenceScope,
                                 candidate.targets);
  if (targetLoop && candidate.sources.size() == 1) {
    candidate.waitScope = *targetLoop;
    candidate.sourceLanes = {0};
    candidate.targetWaits.assign(candidate.targets.size(),
                                 candidate.targets.front());
    candidate.paths.push_back(
        {candidate.sources, candidate.targets.front()});
    candidate.width = 1;
  } else {
    const std::optional<GuardedLifecycleLayout> layout =
        deriveGuardedLifecycleLayout(graph, lifecycle.recurrenceScope,
                                     candidate.sources, candidate.targets);
    if (!layout) {
      return std::nullopt;
    }
    for (SyncCoverNodeId source : candidate.sources) {
      candidate.sourceLanes.push_back(layout->sourceLanes.at(source));
    }
    for (SyncCoverNodeId target : candidate.targets) {
      candidate.targetWaits.push_back(layout->targetWaits.at(target));
    }
    for (const GuardedLifecyclePath &path : layout->paths) {
      candidate.paths.push_back({path.sources, path.waitTarget});
    }
    candidate.width = layout->width;
  }
  candidate.sourceResource = lifecycle.consumerResource;
  candidate.targetResource = lifecycle.producerResource;
  candidate.recurrenceScope = lifecycle.recurrenceScope;
  candidate.distance = 1;
  if (!verifyHierarchicalCandidateShape(graph, opportunities, lifecycle,
                                        candidate)) {
    return std::nullopt;
  }
  if (!verifyHierarchicalAccessBoundary(graph, managed, lifecycle, candidate)) {
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
      candidate.releases.empty() || candidate.sources.empty() ||
      candidate.targets.empty() || candidate.completionEdges.empty() ||
      candidate.sourceLanes.size() != candidate.sources.size() ||
      candidate.targetWaits.size() != candidate.targets.size() ||
      candidate.width == 0) {
    return std::nullopt;
  }
  SyncCoverMechanismDescriptorBuilder builder(
      SyncCoverMechanismKind::VerifiedProtocol, providerIdentity);
  const SyncCoverDescriptorActionRef prime = builder.addAction(
      SyncCoverResourceActionKind::Produce, domain.sourceResource,
      {SyncCoverAnchorKind::ScopeEntry, 0, candidate.recurrenceScope});
  std::set<SyncCoverNodeId> uniqueWaitTargets(candidate.targetWaits.begin(),
                                              candidate.targetWaits.end());
  std::map<std::pair<SyncCoverNodeId, unsigned>,
           SyncCoverDescriptorActionRef>
      bodyWaits;
  for (SyncCoverNodeId waitTarget : uniqueWaitTargets) {
    for (unsigned lane = 0; lane < candidate.width; ++lane) {
      const SyncCoverAnchor anchor = candidate.waitScope
                                         ? SyncCoverAnchor{
                                               SyncCoverAnchorKind::ScopeEntry,
                                               0, *candidate.waitScope}
                                         : SyncCoverAnchor{
                                               SyncCoverAnchorKind::BeforeNode,
                                               waitTarget, 0};
      bodyWaits.emplace(
          std::make_pair(waitTarget, lane),
          builder.addAction(SyncCoverResourceActionKind::Consume,
                            domain.targetResource, anchor));
    }
  }
  std::map<SyncCoverNodeId, SyncCoverDescriptorActionRef> bodySets;
  for (SyncCoverNodeId source : candidate.sources) {
    bodySets.emplace(
        source,
        builder.addAction(SyncCoverResourceActionKind::Produce,
                          domain.sourceResource,
                          {SyncCoverAnchorKind::AfterNode, source, 0}));
  }
  const SyncCoverDescriptorActionRef drain = builder.addAction(
      SyncCoverResourceActionKind::Consume, domain.targetResource,
      {SyncCoverAnchorKind::ScopeExit, 0, candidate.recurrenceScope});
  std::vector<SyncCoverProtocolSupply> supplies;
  supplies.reserve(candidate.completionEdges.size());
  for (const auto &[source, target] : candidate.completionEdges) {
    auto bodySet = bodySets.find(source);
    auto sourcePosition = std::lower_bound(candidate.sources.begin(),
                                           candidate.sources.end(), source);
    auto targetPosition = std::lower_bound(candidate.targets.begin(),
                                           candidate.targets.end(), target);
    if (bodySet == bodySets.end() ||
        sourcePosition == candidate.sources.end() ||
        *sourcePosition != source || targetPosition == candidate.targets.end() ||
        *targetPosition != target) {
      return std::nullopt;
    }
    const std::size_t sourceIndex = static_cast<std::size_t>(
        sourcePosition - candidate.sources.begin());
    const std::size_t targetIndex = static_cast<std::size_t>(
        targetPosition - candidate.targets.begin());
    const auto bodyWait = bodyWaits.find(
        {candidate.targetWaits[targetIndex], candidate.sourceLanes[sourceIndex]});
    if (bodyWait == bodyWaits.end()) {
      return std::nullopt;
    }
    SyncCoverEdge edge;
    edge.source = source;
    edge.target = target;
    edge.kind = SyncCoverEdgeKind::CompletionSupply;
    edge.scope = candidate.recurrenceScope;
    edge.distance = 1;
    supplies.push_back({edge, bodySet->second, bodyWait->second});
  }
  std::vector<SyncCoverDescriptorActionRef> actions{prime};
  for (const auto &[key, action] : bodyWaits) {
    (void)key;
    actions.push_back(action);
  }
  for (const auto &[source, action] : bodySets) {
    (void)source;
    actions.push_back(action);
  }
  actions.push_back(drain);
  if (!builder.addProtocolLane(domain, candidate.recurrenceScope, 1,
                               candidate.width,
                               std::move(actions),
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
  const std::set<SyncCoverNodeId> uniqueWaitTargets(
      candidate.targetWaits.begin(), candidate.targetWaits.end());
  const std::size_t waitCount = uniqueWaitTargets.size() * candidate.width;
  const std::size_t firstSetAction = 1 + waitCount;
  const std::size_t drainAction = firstSetAction + candidate.sources.size();
  const bool invalidCardinality =
      descriptor.kind != SyncCoverMechanismKind::VerifiedProtocol ||
      descriptor.barrier || descriptor.actions.size() != drainAction + 1 ||
      descriptor.resourceUses.size() != 1 ||
      descriptor.supplyEdges.size() != candidate.completionEdges.size() ||
      descriptor.supplyBindings.size() != candidate.completionEdges.size();
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
      use.width != candidate.width ||
      use.actions.size() != descriptor.actions.size();
  const bool invalidResourceUse =
      invalidUse || use.supplyEdges.size() != candidate.completionEdges.size();
  if (invalidResourceUse) {
    return false;
  }
  for (std::size_t action = 0; action < use.actions.size(); ++action) {
    if (use.actions[action] != action) {
      return false;
    }
  }
  std::map<std::pair<SyncCoverNodeId, unsigned>, std::size_t> waitActions;
  std::size_t nextWait = 1;
  for (SyncCoverNodeId waitTarget : uniqueWaitTargets) {
    for (unsigned lane = 0; lane < candidate.width; ++lane) {
      waitActions.emplace(std::make_pair(waitTarget, lane), nextWait++);
    }
  }
  std::map<SyncCoverNodeId, std::size_t> setActions;
  for (std::size_t source = 0; source < candidate.sources.size(); ++source) {
    setActions.emplace(candidate.sources[source], firstSetAction + source);
  }
  for (std::size_t index = 0; index < candidate.completionEdges.size();
       ++index) {
    const bool invalidEdgeIndex = use.supplyEdges[index] != index;
    const SyncCoverEdge &edge = descriptor.supplyEdges[index];
    const SyncCoverSupplyBinding &binding = descriptor.supplyBindings[index];
    const auto &[source, target] = candidate.completionEdges[index];
    auto setAction = setActions.find(source);
    auto sourcePosition = std::lower_bound(candidate.sources.begin(),
                                           candidate.sources.end(), source);
    auto targetPosition = std::lower_bound(candidate.targets.begin(),
                                           candidate.targets.end(), target);
    if (sourcePosition == candidate.sources.end() ||
        *sourcePosition != source || targetPosition == candidate.targets.end() ||
        *targetPosition != target) {
      return false;
    }
    const std::size_t sourceIndex = static_cast<std::size_t>(
        sourcePosition - candidate.sources.begin());
    const std::size_t targetIndex = static_cast<std::size_t>(
        targetPosition - candidate.targets.begin());
    auto waitAction = waitActions.find(
        {candidate.targetWaits[targetIndex], candidate.sourceLanes[sourceIndex]});
    const bool invalidEdge =
        edge.source != source || edge.target != target ||
        edge.kind != SyncCoverEdgeKind::CompletionSupply || edge.mechanism ||
        edge.scope != candidate.recurrenceScope || edge.distance != 1;
    const bool invalidBinding =
        binding.supplyEdge != index || binding.resourceUse != 0 ||
        setAction == setActions.end() || waitAction == waitActions.end() ||
        binding.produceAction != setAction->second ||
        binding.consumeAction != waitAction->second;
    if (invalidEdgeIndex || invalidEdge || invalidBinding) {
      return false;
    }
  }
  const auto &actions = descriptor.actions;
  if (!hasAnchor(actions[0], SyncCoverResourceActionKind::Produce,
                 candidate.sourceResource, SyncCoverAnchorKind::ScopeEntry, 0,
                 candidate.recurrenceScope) ||
      !hasAnchor(actions[drainAction], SyncCoverResourceActionKind::Consume,
                 candidate.targetResource, SyncCoverAnchorKind::ScopeExit, 0,
                 candidate.recurrenceScope)) {
    return false;
  }
  for (const auto &[key, action] : waitActions) {
    const SyncCoverAnchorKind anchorKind =
        candidate.waitScope ? SyncCoverAnchorKind::ScopeEntry
                            : SyncCoverAnchorKind::BeforeNode;
    const SyncCoverNodeId anchorNode = candidate.waitScope ? 0 : key.first;
    const SyncCoverScopeId anchorScope = candidate.waitScope
                                             ? *candidate.waitScope
                                             : SyncCoverScopeId{0};
    if (!hasAnchor(actions[action], SyncCoverResourceActionKind::Consume,
                   candidate.targetResource, anchorKind, anchorNode,
                   anchorScope)) {
      return false;
    }
  }
  for (std::size_t index = 0; index < candidate.sources.size(); ++index) {
    if (!hasAnchor(actions[firstSetAction + index],
                   SyncCoverResourceActionKind::Produce,
                   candidate.sourceResource, SyncCoverAnchorKind::AfterNode,
                   candidate.sources[index], 0)) {
      return false;
    }
  }
  return true;
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
      candidate.completionEdges = {{release.source, release.target}};
      candidate.sources = {release.source};
      candidate.sourceLanes = {0};
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
