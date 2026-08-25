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

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::pto;

namespace {

using ResourceOwner = std::pair<SyncCoverMechanismId, std::size_t>;

CanonicalSyncCoveringAllocationValidation makeError(
    CanonicalSyncCoveringAllocationError error,
    std::optional<SyncCoverMechanismId> mechanism = std::nullopt,
    std::optional<std::size_t> resourceUse = std::nullopt,
    std::optional<SyncCoverResourceDomainId> domain = std::nullopt,
    std::optional<unsigned> physicalId = std::nullopt) {
  return {error, mechanism, resourceUse, domain, physicalId};
}

bool sameProvider(const CanonicalSelectionMechanismRef &first,
                  const CanonicalSelectionMechanismRef &second) {
  return first == second;
}

bool containsReservedId(const SyncCoverResourceDomain &domain, unsigned id) {
  return std::binary_search(domain.reservedIds.begin(),
                            domain.reservedIds.end(), id);
}

} // namespace

bool mlir::pto::canonicalSyncCoveringResourceUseMatches(
    const CanonicalSyncCoveringSelectedResourceUse &selected,
    const SyncCoverResourceUse &live,
    const SyncCoverTimelineInterval &liveLifetime) {
  return live.domain == selected.domain && live.scope == selected.scope &&
         live.distance == selected.distance && live.width == selected.width &&
         liveLifetime.begin == selected.lifetime.begin &&
         liveLifetime.end == selected.lifetime.end;
}

CanonicalSyncCoveringAllocationValidation
mlir::pto::validateCanonicalSyncCoveringAllocation(
    const CanonicalSyncCoveringShadowSnapshot &snapshot) {
  const bool selectionReady =
      snapshot.selectionAttempted &&
      snapshot.selectionError == SyncCoverSelectionError::None &&
      static_cast<bool>(snapshot.selectedResources);
  if (!selectionReady) {
    return makeError(
        CanonicalSyncCoveringAllocationError::SelectionNotReady);
  }

  std::map<SyncCoverMechanismId, CanonicalSelectionMechanismRef> providers;
  std::set<CanonicalSelectionMechanismRef> providerIdentities;
  for (const CanonicalSyncCoveringSelectedProvider &selected :
       snapshot.selectedProviders) {
    const bool uniqueMechanism =
        providers.emplace(selected.mechanism, selected.provider).second;
    const bool uniqueProvider =
        providerIdentities.insert(selected.provider).second;
    if (!uniqueMechanism || !uniqueProvider) {
      return makeError(CanonicalSyncCoveringAllocationError::InvalidProvider,
                       selected.mechanism);
    }
  }
  const bool providerCountMismatch =
      providers.size() != snapshot.selectedMechanisms;
  if (providerCountMismatch) {
    return makeError(CanonicalSyncCoveringAllocationError::InvalidProvider);
  }

  std::map<SyncCoverResourceDomainId, const SyncCoverResourceDomain *> domains;
  for (const SyncCoverResourceDomain &domain :
       snapshot.resourceDomainDetails) {
    const bool uniqueDomain = domains.emplace(domain.id, &domain).second;
    const bool reservationsValid =
        std::is_sorted(domain.reservedIds.begin(), domain.reservedIds.end()) &&
        std::adjacent_find(domain.reservedIds.begin(),
                           domain.reservedIds.end()) ==
            domain.reservedIds.end();
    if (!uniqueDomain || !reservationsValid) {
      return makeError(CanonicalSyncCoveringAllocationError::InvalidDomain,
                       std::nullopt, std::nullopt, domain.id);
    }
  }
  const bool domainCountMismatch =
      domains.size() != snapshot.resourceDomainCount;
  if (domainCountMismatch) {
    return makeError(CanonicalSyncCoveringAllocationError::InvalidDomain);
  }

  std::map<SyncCoverResourceDomainId, const SyncCoverDomainFeasibility *>
      feasibility;
  for (const SyncCoverDomainFeasibility &entry :
       snapshot.selectedResources.domains) {
    const bool uniqueFeasibility =
        feasibility.emplace(entry.domain, &entry).second;
    const bool domainKnown = domains.count(entry.domain) != 0;
    if (!uniqueFeasibility || !domainKnown || entry.overflow != 0) {
      return makeError(CanonicalSyncCoveringAllocationError::InvalidDomain,
                       std::nullopt, std::nullopt, entry.domain);
    }
  }
  const bool allDomainsHaveFeasibility =
      feasibility.size() == domains.size();
  if (!allDomainsHaveFeasibility) {
    return makeError(CanonicalSyncCoveringAllocationError::InvalidDomain);
  }

  std::map<ResourceOwner, const CanonicalSyncCoveringSelectedResourceUse *> uses;
  std::map<SyncCoverResourceDomainId, std::vector<SyncWeightedInterval>>
      intervals;
  for (const CanonicalSyncCoveringSelectedResourceUse &use :
       snapshot.selectedResourceUses) {
    const ResourceOwner owner{use.mechanism, use.resourceUse};
    auto provider = providers.find(use.mechanism);
    auto domain = domains.find(use.domain);
    const bool providerMatches =
        provider != providers.end() && sameProvider(provider->second, use.provider);
    const bool domainMatches =
        domain != domains.end() && domain->second->kind == use.kind &&
        domain->second->sourceResource == use.sourceResource &&
        domain->second->targetResource == use.targetResource &&
        domain->second->poolIdentity == use.poolIdentity;
    const bool validLifetime = use.lifetime.begin <= use.lifetime.end;
    const bool uniqueUse = uses.emplace(owner, &use).second;
    if (!providerMatches || !domainMatches || !validLifetime || use.width == 0 ||
        !uniqueUse) {
      return makeError(
          CanonicalSyncCoveringAllocationError::InvalidResourceUse,
          use.mechanism, use.resourceUse, use.domain);
    }
    if (use.kind != SyncCoverResourceKind::EventId || use.poolIdentity != 0 ||
        !use.eventIndex) {
      return makeError(
          CanonicalSyncCoveringAllocationError::UnsupportedResourceKind,
          use.mechanism, use.resourceUse, use.domain);
    }
    intervals[use.domain].push_back(
        {{use.lifetime.begin, use.lifetime.end}, use.width});
  }

  struct AuthoritativeAllocation {
    SyncCoverResourceDomainId domain = 0;
    std::size_t width = 0;
    const std::vector<unsigned> *ids = nullptr;
  };
  std::map<ResourceOwner, AuthoritativeAllocation> authoritative;
  for (const auto &[domainId, entry] : feasibility) {
    for (const SyncCoverResourceAllocation &allocation : entry->allocations) {
      const ResourceOwner owner{allocation.owner.mechanism,
                                allocation.owner.resourceUse};
      const bool uniqueAllocation =
          authoritative
              .emplace(owner, AuthoritativeAllocation{
                                  domainId, allocation.owner.width,
                                  &allocation.ids})
              .second;
      if (!uniqueAllocation) {
        return makeError(CanonicalSyncCoveringAllocationError::InvalidAllocation,
                         owner.first, owner.second, domainId);
      }
    }
  }

  std::map<ResourceOwner,
           const CanonicalSyncCoveringResourceAllocation *> allocations;
  std::map<SyncCoverResourceDomainId,
           std::map<unsigned, std::vector<SyncCoverTimelineInterval>>>
      assignedLifetimes;
  for (const CanonicalSyncCoveringResourceAllocation &allocation :
       snapshot.selectedAllocations) {
    const ResourceOwner owner{allocation.mechanism, allocation.resourceUse};
    auto use = uses.find(owner);
    auto domain = domains.find(allocation.domain);
    auto finalAllocation = authoritative.find(owner);
    const bool uniqueAllocation = allocations.emplace(owner, &allocation).second;
    const bool referencesKnownObjects =
        use != uses.end() && domain != domains.end() &&
        finalAllocation != authoritative.end();
    if (!uniqueAllocation || !referencesKnownObjects) {
      return makeError(CanonicalSyncCoveringAllocationError::InvalidAllocation,
                       allocation.mechanism, allocation.resourceUse,
                       allocation.domain);
    }
    const bool ownerMatches =
        sameProvider(use->second->provider, allocation.provider);
    const bool metadataMatches =
        ownerMatches && allocation.domain == use->second->domain &&
        allocation.kind == use->second->kind &&
        allocation.sourceResource == use->second->sourceResource &&
        allocation.targetResource == use->second->targetResource;
    const bool authoritativeMatches =
        finalAllocation->second.domain == allocation.domain &&
        finalAllocation->second.width == use->second->width &&
        *finalAllocation->second.ids == allocation.ids;
    const bool widthMatches = allocation.ids.size() == use->second->width;
    if (!metadataMatches || !authoritativeMatches || !widthMatches) {
      return makeError(CanonicalSyncCoveringAllocationError::InvalidAllocation,
                       allocation.mechanism, allocation.resourceUse,
                       allocation.domain);
    }
    std::set<unsigned> laneIds;
    for (unsigned id : allocation.ids) {
      const bool idValid = id < domain->second->budget &&
                           !containsReservedId(*domain->second, id) &&
                           laneIds.insert(id).second;
      if (!idValid) {
        return makeError(
            CanonicalSyncCoveringAllocationError::InvalidAllocation,
            allocation.mechanism, allocation.resourceUse, allocation.domain,
            id);
      }
      assignedLifetimes[allocation.domain][id].push_back(use->second->lifetime);
    }
  }
  const bool allUsesAllocated = allocations.size() == uses.size();
  const bool allAuthoritativeUsesAllocated =
      authoritative.size() == uses.size();
  if (!allUsesAllocated || !allAuthoritativeUsesAllocated) {
    return makeError(CanonicalSyncCoveringAllocationError::InvalidAllocation);
  }

  for (const auto &[domainId, domain] : domains) {
    const SyncIntervalPressure pressure = evaluateSyncIntervalPressure(
        intervals[domainId], domain->budget, domain->reservedIds);
    const SyncCoverDomainFeasibility &expected = *feasibility.at(domainId);
    const bool pressureMatches =
        static_cast<bool>(pressure) && pressure.overflow == 0 &&
        pressure.required == expected.required &&
        pressure.available == expected.available && expected.overflow == 0;
    const bool unusedDomainValid =
        !intervals.count(domainId)
            ? expected.required == 0 && expected.allocations.empty()
            : true;
    if (!pressureMatches || !unusedDomainValid) {
      return makeError(CanonicalSyncCoveringAllocationError::InvalidPressure,
                       std::nullopt, std::nullopt, domainId);
    }
  }

  for (auto &[domainId, byId] : assignedLifetimes) {
    for (auto &[id, lifetimes] : byId) {
      std::sort(lifetimes.begin(), lifetimes.end(), [](const auto &first,
                                                       const auto &second) {
        return std::tie(first.begin, first.end) <
               std::tie(second.begin, second.end);
      });
      for (std::size_t index = 1; index < lifetimes.size(); ++index) {
        if (lifetimes[index - 1].end >= lifetimes[index].begin) {
          return makeError(
              CanonicalSyncCoveringAllocationError::ConflictingAssignment,
              std::nullopt, std::nullopt, domainId, id);
        }
      }
    }
  }
  return {};
}
