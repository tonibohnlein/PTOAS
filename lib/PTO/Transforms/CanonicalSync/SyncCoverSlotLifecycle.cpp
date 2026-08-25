// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverSlotLifecycle.h"

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

struct ReadyKey {
  SlotKey slot;
  std::uint32_t producerResource = 0;
  std::uint32_t consumerResource = 0;

  bool operator<(const ReadyKey &other) const {
    return std::tie(slot, producerResource, consumerResource) <
           std::tie(other.slot, other.producerResource,
                    other.consumerResource);
  }
};

struct LifecycleKey {
  ReadyKey ready;
  SyncCoverScopeId recurrenceScope = 0;
  unsigned distance = 0;

  bool operator<(const LifecycleKey &other) const {
    return std::tie(ready, recurrenceScope, distance) <
           std::tie(other.ready, other.recurrenceScope, other.distance);
  }
};

struct ReadyScopeKey {
  ReadyKey ready;
  SyncCoverScopeId recurrenceScope = 0;

  bool operator<(const ReadyScopeKey &other) const {
    return std::tie(ready, recurrenceScope) <
           std::tie(other.ready, other.recurrenceScope);
  }
};

bool reads(SyncCoverStorageAccessMode mode) {
  return (static_cast<unsigned>(mode) &
          static_cast<unsigned>(SyncCoverStorageAccessMode::Read)) != 0;
}

bool writes(SyncCoverStorageAccessMode mode) {
  return (static_cast<unsigned>(mode) &
          static_cast<unsigned>(SyncCoverStorageAccessMode::Write)) != 0;
}

bool isWholeSlot(const SyncCoverCandidateSlot &slot) {
  return slot.sourceExtent == slot.targetExtent &&
         slot.overlap == slot.sourceExtent;
}

bool overlaps(SyncCoverStorageInterval first,
              SyncCoverStorageInterval second) {
  return first.begin < second.end && second.begin < first.end;
}

bool isReady(const SyncCoverCandidateOpportunity &opportunity) {
  return opportunity.kind == SyncCoverDemandKind::MemoryRAW &&
         opportunity.distance == 0 && opportunity.slot &&
         writes(opportunity.slot->sourceMode) &&
         reads(opportunity.slot->targetMode) &&
         opportunity.sourceResource != opportunity.targetResource;
}

bool isRelease(const SyncCoverGraph &graph,
               const SyncCoverCandidateOpportunity &opportunity) {
  const bool validScope = opportunity.scope < graph.getScopes().size() &&
                          graph.getScopes()[opportunity.scope].isLoop;
  return opportunity.kind == SyncCoverDemandKind::MemoryWAR &&
         opportunity.distance != 0 && validScope && opportunity.slot &&
         reads(opportunity.slot->sourceMode) &&
         writes(opportunity.slot->targetMode) &&
         opportunity.sourceResource != opportunity.targetResource;
}

bool belongsToRecurrence(const SyncCoverGraph &graph,
                         const SyncCoverCandidateOpportunity &ready,
                         SyncCoverScopeId recurrenceScope) {
  return graph.scopeContains(recurrenceScope,
                             graph.getNodes()[ready.source].scope) &&
         graph.scopeContains(recurrenceScope,
                             graph.getNodes()[ready.target].scope);
}

bool requiresPathSensitiveProof(
    const SyncCoverGraph &graph,
    const SyncCoverCandidateOpportunity &opportunity,
    SyncCoverScopeId recurrenceScope) {
  const bool hasGuard = !opportunity.sourceGuard.literals.empty() ||
                        !opportunity.targetGuard.literals.empty();
  if (hasGuard) {
    return true;
  }
  const std::optional<std::size_t> recurrenceDepth =
      graph.getScopeLoopDepth(recurrenceScope);
  const SyncCoverNodeId endpoints[] = {opportunity.source,
                                       opportunity.target};
  for (SyncCoverNodeId endpoint : endpoints) {
    const SyncCoverScopeId endpointScope = graph.getNodes()[endpoint].scope;
    const std::optional<std::size_t> endpointDepth =
        graph.getScopeLoopDepth(endpointScope);
    const bool optionalScope =
        !graph.scopeMustExecuteWithin(recurrenceScope, endpointScope);
    const bool nestedLoop = !recurrenceDepth || !endpointDepth ||
                            *endpointDepth > *recurrenceDepth;
    if (optionalScope || nestedLoop) {
      return true;
    }
  }
  return false;
}

} // namespace

SyncCoverSlotLifecycleResult mlir::pto::discoverSyncCoverSlotLifecycles(
    const SyncCoverGraph &graph, const SyncCoverCandidateIndex &index,
    const SyncCoverSlotLifecycleOptions &options) {
  SyncCoverSlotLifecycleResult result;
  const bool invalidGraph = !graph.isStructureFrozen() || !graph.validate();
  if (invalidGraph) {
    result.error = SyncCoverSlotLifecycleError::InvalidGraph;
    return result;
  }
  const auto opportunities = index.getOpportunities();
  const bool invalidIndex = !index.isCurrentFor(graph) || !opportunities;
  if (invalidIndex) {
    result.error = SyncCoverSlotLifecycleError::InvalidCandidateIndex;
    return result;
  }

  std::map<ReadyKey, std::vector<const SyncCoverCandidateOpportunity *>> ready;
  std::map<LifecycleKey, std::vector<const SyncCoverCandidateOpportunity *>>
      release;
  for (const SyncCoverCandidateOpportunity &opportunity :
       *opportunities.value) {
    if (!opportunity.slot) {
      continue;
    }
    if (!isWholeSlot(*opportunity.slot)) {
      ++result.partialSlotOpportunities;
      continue;
    }
    const SlotKey slot{opportunity.slot->domain,
                       opportunity.slot->sourceExtent};
    if (isReady(opportunity)) {
      ready[{slot, opportunity.sourceResource, opportunity.targetResource}]
          .push_back(&opportunity);
    }
    if (isRelease(graph, opportunity)) {
      release[{{slot, opportunity.targetResource,
                opportunity.sourceResource},
               opportunity.scope, opportunity.distance}]
          .push_back(&opportunity);
    }
  }

  std::map<ReadyScopeKey,
           std::vector<const SyncCoverCandidateOpportunity *>>
      matchingReadyCache;
  for (const auto &releaseEntry : release) {
    auto readyEntry = ready.find(releaseEntry.first.ready);
    if (readyEntry == ready.end()) {
      continue;
    }
    const bool capReached =
        result.lifecycles.size() == options.maximumLifecycles;
    if (capReached) {
      result.truncated = true;
      break;
    }
    const ReadyScopeKey readyScope{releaseEntry.first.ready,
                                   releaseEntry.first.recurrenceScope};
    auto [cached, inserted] = matchingReadyCache.emplace(
        readyScope,
        std::vector<const SyncCoverCandidateOpportunity *>{});
    if (inserted) {
      for (const SyncCoverCandidateOpportunity *opportunity :
           readyEntry->second) {
        if (belongsToRecurrence(graph, *opportunity,
                                releaseEntry.first.recurrenceScope)) {
          cached->second.push_back(opportunity);
        }
      }
    }
    const auto &matchingReady = cached->second;
    if (matchingReady.empty()) {
      continue;
    }

    SyncCoverSlotLifecycle lifecycle;
    lifecycle.id = result.lifecycles.size();
    lifecycle.slot = {releaseEntry.first.ready.slot.domain,
                      releaseEntry.first.ready.slot.extent};
    lifecycle.producerResource = releaseEntry.first.ready.producerResource;
    lifecycle.consumerResource = releaseEntry.first.ready.consumerResource;
    lifecycle.recurrenceScope = releaseEntry.first.recurrenceScope;
    lifecycle.distance = releaseEntry.first.distance;
    std::set<SyncCoverStorageAccessId> represented;
    for (const SyncCoverCandidateOpportunity *opportunity : matchingReady) {
      lifecycle.ready.push_back(opportunity->id);
      represented.insert(opportunity->slot->sourceAccess);
      represented.insert(opportunity->slot->targetAccess);
      lifecycle.requiresPathSensitiveProof |= requiresPathSensitiveProof(
          graph, *opportunity, releaseEntry.first.recurrenceScope);
    }
    for (const SyncCoverCandidateOpportunity *opportunity :
         releaseEntry.second) {
      lifecycle.release.push_back(opportunity->id);
      represented.insert(opportunity->slot->sourceAccess);
      represented.insert(opportunity->slot->targetAccess);
      lifecycle.requiresPathSensitiveProof |= requiresPathSensitiveProof(
          graph, *opportunity, releaseEntry.first.recurrenceScope);
    }
    const auto domainAccesses =
        index.getDomainAccesses(lifecycle.slot.domain);
    if (!domainAccesses) {
      result.error = SyncCoverSlotLifecycleError::InvalidCandidateIndex;
      return result;
    }
    for (SyncCoverStorageAccessId accessId : *domainAccesses.value) {
      const SyncCoverStorageAccess &access =
          graph.getStorageAccesses()[accessId];
      if (overlaps(access.extent, lifecycle.slot.extent)) {
        lifecycle.managedAccesses.push_back(accessId);
        lifecycle.hasUnrepresentedAccesses |=
            represented.find(accessId) == represented.end();
      }
    }
    result.lifecycles.push_back(std::move(lifecycle));
  }
  return result;
}
