// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

using ResourceOwner = std::pair<SyncCoverMechanismId, std::size_t>;

const CanonicalEventBundleCandidate *findEventBundleProvider(
    const CanonicalMechanismUniverse &universe,
    ArrayRef<CanonicalEventBundleCandidate> selected, std::size_t id) {
  auto candidate = llvm::find_if(
      universe.eventBundles,
      [&](const CanonicalEventBundleCandidate &bundle) { return bundle.id == id; });
  if (candidate != universe.eventBundles.end()) {
    return &*candidate;
  }
  auto fallback = llvm::find_if(
      selected,
      [&](const CanonicalEventBundleCandidate &bundle) { return bundle.id == id; });
  return fallback == selected.end() ? nullptr : &*fallback;
}

const CanonicalBarrierCandidate *findBarrierProvider(
    const CanonicalMechanismUniverse &universe, std::size_t id) {
  auto candidate = llvm::find_if(
      universe.barriers,
      [&](const CanonicalBarrierCandidate &barrier) { return barrier.id == id; });
  return candidate == universe.barriers.end() ? nullptr : &*candidate;
}

} // namespace

LogicalResult CanonicalSyncPlanBuilder::materializeCoveringSelection() {
  if (!plan_.coveringShadowSnapshot_) {
    return func_.emitError(
        "internal error: covering emission has no selected snapshot");
  }
  const CanonicalSyncCoveringShadowSnapshot &snapshot =
      *plan_.coveringShadowSnapshot_;
  const CanonicalSyncCoveringAllocationValidation allocationValidation =
      validateCanonicalSyncCoveringAllocation(snapshot);
  if (!allocationValidation) {
    InFlightDiagnostic diagnostic = func_.emitError()
                                    << "internal error: covering emission "
                                       "allocation is invalid: "
                                    << static_cast<unsigned>(
                                           allocationValidation.error);
    if (allocationValidation.mechanism) {
      diagnostic << " mechanism=" << *allocationValidation.mechanism;
    }
    if (allocationValidation.resourceUse) {
      diagnostic << " use=" << *allocationValidation.resourceUse;
    }
    if (allocationValidation.domain) {
      diagnostic << " domain=" << *allocationValidation.domain;
    }
    if (allocationValidation.physicalId) {
      diagnostic << " physical-id=" << *allocationValidation.physicalId;
    }
    return failure();
  }

  std::map<ResourceOwner, const CanonicalSyncCoveringResourceAllocation *>
      allocations;
  for (const CanonicalSyncCoveringResourceAllocation &allocation :
       snapshot.selectedAllocations) {
    allocations.emplace(ResourceOwner{allocation.mechanism,
                                      allocation.resourceUse},
                        &allocation);
  }
  std::map<CanonicalSelectionMechanismRef,
           std::vector<const CanonicalSyncCoveringSelectedResourceUse *>>
      usesByProvider;
  for (const CanonicalSyncCoveringSelectedResourceUse &use :
       snapshot.selectedResourceUses) {
    usesByProvider[use.provider].push_back(&use);
  }

  std::vector<CanonicalBarrier> barriers;
  std::vector<CanonicalEventBundleCandidate> bundles;
  for (const CanonicalSyncCoveringSelectedProvider &selected :
       snapshot.selectedProviders) {
    auto uses = usesByProvider.find(selected.provider);
    const bool hasUses = uses != usesByProvider.end() && !uses->second.empty();
    if (selected.provider.kind == CanonicalSelectionMechanismKind::Barrier) {
      const CanonicalBarrierCandidate *candidate =
          findBarrierProvider(mechanismUniverse_, selected.provider.id);
      if (!candidate || hasUses) {
        return func_.emitError(
            "internal error: covering barrier provider is invalid");
      }
      barriers.push_back(candidate->barrier);
      continue;
    }

    const CanonicalEventBundleCandidate *candidate = findEventBundleProvider(
        mechanismUniverse_, selectedEventBundles_, selected.provider.id);
    if (!candidate || !hasUses) {
      return func_.emitError(
          "internal error: covering event provider is invalid");
    }
    CanonicalEventBundleCandidate bundle = *candidate;
    std::vector<bool> assigned(bundle.events.size(), false);
    for (CanonicalEvent &event : bundle.events) {
      event.eventIds.clear();
    }
    for (const CanonicalSyncCoveringSelectedResourceUse *use : uses->second) {
      const bool eventInvalid =
          use->kind != SyncCoverResourceKind::EventId || !use->eventIndex ||
          *use->eventIndex >= bundle.events.size() ||
          assigned[*use->eventIndex];
      if (eventInvalid) {
        return func_.emitError(
            "internal error: covering event/use mapping is invalid");
      }
      CanonicalEvent &event = bundle.events[*use->eventIndex];
      const bool eventShapeMismatch =
          static_cast<std::uint32_t>(event.sourcePipe) != use->sourceResource ||
          static_cast<std::uint32_t>(event.targetPipe) != use->targetResource ||
          event.width != use->width;
      auto allocation =
          allocations.find(ResourceOwner{use->mechanism, use->resourceUse});
      if (eventShapeMismatch || allocation == allocations.end()) {
        return func_.emitError(
            "internal error: covering event allocation does not match its "
            "provider");
      }
      event.eventIds.assign(allocation->second->ids.begin(),
                            allocation->second->ids.end());
      assigned[*use->eventIndex] = true;
    }
    const bool hasUnassignedEvent =
        llvm::any_of(assigned, [](bool value) { return !value; });
    if (hasUnassignedEvent) {
      return func_.emitError(
          "internal error: covering event provider has an unallocated event");
    }
    bundles.push_back(std::move(bundle));
  }

  std::vector<CanonicalEvent> events = flattenCanonicalEventBundles(bundles);
  if (!canonicalEventBundleProjectionMatches(bundles, events)) {
    return func_.emitError(
        "internal error: covering event-bundle projection is stale");
  }
  if (!isCandidatePlanWellFormed(barriers, bundles,
                                 plan_.completionRequirements_,
                                 /*diagnose=*/true)) {
    return func_.emitError(
        "internal error: covering synchronization providers are malformed");
  }
  if (failed(verifyEventProtocols(events, /*requireAllocation=*/true,
                                  /*diagnose=*/true))) {
    return func_.emitError(
        "internal error: covering synchronization protocol is invalid");
  }
  if (!planCoversRequirements(barriers, events, /*diagnose=*/true,
                              /*usePositiveTripFacts=*/true)) {
    return func_.emitError(
        "internal error: covering synchronization plan does not cover every "
        "active completion requirement");
  }

  std::map<CanonicalEventDomainKey, std::size_t> eventCounts;
  for (const CanonicalEvent &event : events) {
    ++eventCounts[{event.sourcePipe, event.targetPipe}];
  }
  std::map<SyncCoverResourceDomainId, const SyncCoverDomainFeasibility *>
      feasibility;
  for (const SyncCoverDomainFeasibility &domain :
       snapshot.selectedResources.domains) {
    feasibility.emplace(domain.domain, &domain);
  }
  std::vector<CanonicalEventDomain> domains;
  for (const auto &[key, eventCount] : eventCounts) {
    const SyncCoverResourceDomain *resourceDomain = nullptr;
    for (const SyncCoverResourceDomain &candidate :
         snapshot.resourceDomainDetails) {
      const bool matches =
          candidate.kind == SyncCoverResourceKind::EventId &&
          candidate.sourceResource == static_cast<std::uint32_t>(key.source) &&
          candidate.targetResource == static_cast<std::uint32_t>(key.target);
      if (!matches) {
        continue;
      }
      if (resourceDomain) {
        return func_.emitError(
            "internal error: covering event domain is ambiguous");
      }
      resourceDomain = &candidate;
    }
    if (!resourceDomain) {
      return func_.emitError(
          "internal error: covering event domain is missing");
    }
    auto pressure = feasibility.find(resourceDomain->id);
    const bool countOverflow =
        eventCount > std::numeric_limits<unsigned>::max();
    const bool pressureOverflow =
        pressure == feasibility.end() ||
        pressure->second->required > std::numeric_limits<unsigned>::max() ||
        pressure->second->available > std::numeric_limits<unsigned>::max();
    if (countOverflow || pressureOverflow) {
      return func_.emitError(
          "internal error: covering event domain statistics overflow");
    }
    CanonicalEventDomain domain;
    domain.sourcePipe = key.source;
    domain.targetPipe = key.target;
    domain.originalEventCount = eventCount;
    domain.eventCount = static_cast<unsigned>(eventCount);
    domain.availableIds =
        static_cast<unsigned>(pressure->second->available);
    domain.originalColorCount =
        static_cast<unsigned>(pressure->second->required);
    domain.colorCount = static_cast<unsigned>(pressure->second->required);
    domain.reservedIds.append(resourceDomain->reservedIds.begin(),
                              resourceDomain->reservedIds.end());
    domains.push_back(std::move(domain));
  }

  plan_.barriers_ = std::move(barriers);
  plan_.events_ = std::move(events);
  plan_.domains_ = std::move(domains);
  selectedEventBundles_ = std::move(bundles);
  return success();
}
