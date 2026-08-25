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
    const SyncCoverSlotLifecycle &lifecycle) {
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
        ready.slot->sourceMode == SyncCoverStorageAccessMode::Write &&
        ready.slot->targetMode == SyncCoverStorageAccessMode::Read &&
        isPathInsensitive(graph, ready, lifecycle.recurrenceScope);
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
        release.slot->targetMode == SyncCoverStorageAccessMode::Write &&
        isPathInsensitive(graph, release, lifecycle.recurrenceScope);
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
      candidate.lifecycle != lifecycle.id ||
      candidate.release >= opportunities.size() ||
      std::find(lifecycle.release.begin(), lifecycle.release.end(),
                candidate.release) == lifecycle.release.end();
  if (invalid) {
    return false;
  }
  const SyncCoverCandidateOpportunity &release =
      opportunities[candidate.release];
  const bool exactRelease =
      isExactSlotOpportunity(release, lifecycle.slot) &&
      release.kind == SyncCoverDemandKind::MemoryWAR &&
      release.distance == 1 && release.scope == lifecycle.recurrenceScope &&
      release.sourceResource == lifecycle.consumerResource &&
      release.targetResource == lifecycle.producerResource;
  const bool exactCandidate =
      candidate.source == release.source && candidate.target == release.target &&
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
  const bool invalid = candidate.source >= graph.getNodes().size() ||
                       candidate.target >= graph.getNodes().size() ||
                       candidate.sourceResource == candidate.targetResource;
  if (invalid) {
    return false;
  }
  const SyncCoverNode &source = graph.getNodes()[candidate.source];
  const SyncCoverNode &target = graph.getNodes()[candidate.target];
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
  const SyncCoverNode &releaseTarget = graph.getNodes()[candidate.target];
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
  return matchesReleaseCandidate(*opportunities.value, lifecycle, candidate) &&
         lifecycle.distance == 1 &&
         !lifecycle.requiresPathSensitiveProof &&
         !lifecycle.hasUnrepresentedAccesses &&
         hasExactLifecycleRoles(graph, *opportunities.value, lifecycle) &&
         verifyAccessClosure(index, lifecycle, *managed) &&
         verifyStockPrerequisites(graph, candidate) &&
         verifyCandidateBoundaryClosure(graph, *managed, candidate);
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
    const bool evaluationCapReached =
        result.evaluations == options.maximumEvaluations;
    if (candidateCapReached || evaluationCapReached) {
      result.truncated = true;
      return result;
    }
    ++result.evaluations;
    const SyncCoverSlotLifecycle &lifecycle =
        lifecycles.lifecycles[lifecycleIndex];
    if (lifecycle.id != lifecycleIndex) {
      result.error = SyncCoverSlotProtocolError::InvalidLifecycle;
      return result;
    }
    if (lifecycle.requiresPathSensitiveProof) {
      ++result.pathSensitiveLifecycles;
      continue;
    }
    if (!hasExactLifecycleRoles(graph, *opportunities.value, lifecycle)) {
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
      const bool evaluationLimit =
          result.evaluations == options.maximumEvaluations;
      if (capReached || evaluationLimit) {
        result.truncated = true;
        return result;
      }
      ++result.evaluations;
      SyncCoverSlotProtocolCandidate candidate;
      candidate.id = result.candidates.size();
      candidate.lifecycle = lifecycle.id;
      candidate.release = releaseId;
      candidate.source = release.source;
      candidate.target = release.target;
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
