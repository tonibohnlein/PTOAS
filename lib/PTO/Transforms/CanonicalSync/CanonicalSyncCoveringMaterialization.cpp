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
#include "CanonicalSyncCoveringSlotRecipe.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

using ResourceOwner = std::pair<SyncCoverMechanismId, std::size_t>;

std::optional<std::size_t>
getFirstSyntheticBundleId(const CanonicalMechanismUniverse &universe,
                          ArrayRef<CanonicalSyncCoveringSelectedEventBundle>
                              selected) {
  std::size_t next = 0;
  const auto account = [&](const CanonicalEventBundleCandidate &bundle) {
    if (bundle.id == std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    next = std::max(next, bundle.id + 1);
    return true;
  };
  const bool identityOverflow =
      !llvm::all_of(universe.eventBundles, account) ||
      !llvm::all_of(selected, [&](const auto &recipe) {
        if (recipe.bundleId == std::numeric_limits<std::size_t>::max()) {
          return false;
        }
        next = std::max(next, recipe.bundleId + 1);
        return true;
      });
  if (identityOverflow) {
    return std::nullopt;
  }
  return next;
}

} // namespace

LogicalResult CanonicalSyncPlanBuilder::materializeCoveringSelection() {
  if (!plan_.coveringSnapshot_) {
    return func_.emitError(
        "internal error: covering emission has no selected snapshot");
  }
  const CanonicalSyncCoveringSnapshot &snapshot =
      *plan_.coveringSnapshot_;
  const CanonicalSyncCoveringAllocationValidation allocationValidation =
      validateCanonicalSyncCoveringAllocation(snapshot);
  if (!allocationValidation) {
    InFlightDiagnostic diagnostic =
        func_.emitError() << "internal error: covering emission "
                             "allocation is invalid: "
                          << static_cast<unsigned>(allocationValidation.error);
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
    allocations.emplace(
        ResourceOwner{allocation.mechanism, allocation.resourceUse},
        &allocation);
  }
  std::map<CanonicalSelectionMechanismRef,
           std::vector<const CanonicalSyncCoveringSelectedResourceUse *>>
      usesByProvider;
  for (const CanonicalSyncCoveringSelectedResourceUse &use :
       snapshot.selectedResourceUses) {
    usesByProvider[use.provider].push_back(&use);
  }
  std::map<CanonicalSelectionMechanismRef,
           const CanonicalSyncCoveringSelectedSlotProtocol *>
      slotProtocols;
  for (const CanonicalSyncCoveringSelectedSlotProtocol &recipe :
       snapshot.selectedSlotProtocols) {
    if (!slotProtocols.emplace(recipe.provider, &recipe).second) {
      return func_.emitError(
          "internal error: covering slot protocol recipe is duplicated");
    }
  }
  std::map<CanonicalSelectionMechanismRef,
           const CanonicalSyncCoveringSelectedBarrier *>
      selectedBarriers;
  for (const CanonicalSyncCoveringSelectedBarrier &recipe :
       snapshot.selectedBarriers) {
    if (!selectedBarriers.emplace(recipe.provider, &recipe).second) {
      return func_.emitError(
          "internal error: covering barrier recipe is duplicated");
    }
  }
  std::map<CanonicalSelectionMechanismRef,
           const CanonicalSyncCoveringSelectedEventBundle *>
      selectedEvents;
  for (const CanonicalSyncCoveringSelectedEventBundle &recipe :
       snapshot.selectedEventBundles) {
    if (!selectedEvents.emplace(recipe.provider, &recipe).second) {
      return func_.emitError(
          "internal error: covering event recipe is duplicated");
    }
  }
  std::optional<std::size_t> nextSyntheticBundle =
      getFirstSyntheticBundleId(mechanismUniverse_,
                                snapshot.selectedEventBundles);
  if (!nextSyntheticBundle) {
    return func_.emitError(
        "internal error: covering slot protocol bundle identity overflow");
  }

  std::vector<CanonicalBarrier> barriers;
  std::vector<CanonicalEventBundleCandidate> bundles;
  for (const CanonicalSyncCoveringSelectedProvider &selected :
       snapshot.selectedProviders) {
    auto uses = usesByProvider.find(selected.provider);
    const bool hasUses = uses != usesByProvider.end() && !uses->second.empty();
    if (selected.provider.kind == CanonicalSelectionMechanismKind::Barrier) {
      auto recipe = selectedBarriers.find(selected.provider);
      const bool invalidRecipe =
          recipe == selectedBarriers.end() || hasUses ||
          recipe->second->mechanism != selected.mechanism ||
          !(recipe->second->provider == selected.provider);
      if (invalidRecipe) {
        return func_.emitError(
            "internal error: covering barrier provider is invalid");
      }
      barriers.push_back(recipe->second->barrier);
      continue;
    }
    if (selected.provider.kind ==
        CanonicalSelectionMechanismKind::SlotProtocol) {
      auto recipe = slotProtocols.find(selected.provider);
      const bool invalidRecipe =
          recipe == slotProtocols.end() || !hasUses ||
          uses->second.size() != 1 ||
          recipe->second->mechanism != selected.mechanism ||
          !(recipe->second->provider == selected.provider) ||
          recipe->second->resourceUse != uses->second.front()->resourceUse;
      if (invalidRecipe) {
        return func_.emitError(
            "internal error: covering slot protocol recipe is invalid");
      }
      const CanonicalSyncCoveringSelectedResourceUse &use =
          *uses->second.front();
      auto allocation =
          allocations.find(ResourceOwner{use.mechanism, use.resourceUse});
      if (allocation == allocations.end()) {
        return func_.emitError(
            "internal error: covering slot protocol allocation is invalid");
      }
      std::optional<CanonicalEventBundleCandidate> bundle =
          canonical_sync_covering::materializeSlotProtocolBundle(
              *recipe->second, use, *allocation->second, *nextSyntheticBundle);
      if (!bundle) {
        return func_.emitError(
            "internal error: covering slot protocol allocation is invalid");
      }
      ++*nextSyntheticBundle;
      bundles.push_back(std::move(*bundle));
      continue;
    }

    auto recipe = selectedEvents.find(selected.provider);
    if (recipe == selectedEvents.end() ||
        recipe->second->mechanism != selected.mechanism ||
        !(recipe->second->provider == selected.provider)) {
      return func_.emitError()
             << "internal error: covering event provider has no emission "
                "recipe: event-bundle["
             << selected.provider.id << "]";
    }
    if (!hasUses) {
      return func_.emitError()
             << "internal error: covering event provider has no resource "
                "use: event-bundle["
             << selected.provider.id << "]";
    }
    CanonicalEventBundleCandidate bundle;
    bundle.id = recipe->second->bundleId;
    bundle.kind = recipe->second->kind;
    bundle.events.append(recipe->second->events.begin(),
                         recipe->second->events.end());
    std::vector<bool> assigned(bundle.events.size(), false);
    for (CanonicalEvent &event : bundle.events) {
      event.eventIds.clear();
    }
    for (const CanonicalSyncCoveringSelectedResourceUse *use : uses->second) {
      const bool eventInvalid =
          use->kind != SyncCoverResourceKind::EventId ||
          !use->materializationEventIndex ||
          *use->materializationEventIndex >= bundle.events.size() ||
          assigned[*use->materializationEventIndex];
      if (eventInvalid) {
        return func_.emitError(
            "internal error: covering event/use mapping is invalid");
      }
      CanonicalEvent &event = bundle.events[*use->materializationEventIndex];
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
      assigned[*use->materializationEventIndex] = true;
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
  if (failed(verifyEventProtocols(events, /*requireAllocation=*/true,
                                  /*diagnose=*/true))) {
    return func_.emitError(
        "internal error: covering synchronization protocol is invalid");
  }
  // The direct-cover solver has already run its independent whole-plan graph
  // oracle over every active demand. Re-running the legacy verifier here is
  // both quadratic and incomplete for verifier-proved hierarchical ownership
  // consequences, which span more than one structured recurrence scope.

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
    domain.availableIds = static_cast<unsigned>(pressure->second->available);
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
