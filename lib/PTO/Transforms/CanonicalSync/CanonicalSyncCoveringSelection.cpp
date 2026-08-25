// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Runs direct-cover selection over the translated universe without changing
// CanonicalSync emission.

#include "CanonicalSyncCoveringSelection.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_covering;

LogicalResult MechanismAdapter::buildLegacySeed() {
  for (const CanonicalBarrier &barrier : plan_.getBarriers()) {
    auto candidate =
        llvm::find_if(legacyUniverse_.barriers, [&](const auto &entry) {
          return barriersEquivalent(barrier, entry.barrier);
        });
    if (candidate == legacyUniverse_.barriers.end()) {
      return func_.emitError(
          "internal error: selected canonical barrier has no covering "
          "provider");
    }
    const CanonicalSelectionMechanismRef provider{
        CanonicalSelectionMechanismKind::Barrier, candidate->id};
    auto mechanism = providers_.find(provider);
    if (mechanism == providers_.end()) {
      return func_.emitError(
          "internal error: selected canonical barrier was not adapted");
    }
    legacySeed_.push_back(mechanism->second);
  }
  for (const CanonicalEventBundleCandidate &bundle : selectedEventBundles_) {
    const CanonicalSelectionMechanismRef provider{
        CanonicalSelectionMechanismKind::EventBundle, bundle.id};
    auto mechanism = providers_.find(provider);
    if (mechanism == providers_.end()) {
      return func_.emitError(
          "internal error: selected canonical event bundle was not adapted");
    }
    legacySeed_.push_back(mechanism->second);
  }
  llvm::sort(legacySeed_);
  legacySeed_.erase(std::unique(legacySeed_.begin(), legacySeed_.end()),
                    legacySeed_.end());
  return success();
}

LogicalResult
MechanismAdapter::solve(CanonicalSyncCoveringShadowSnapshot &snapshot) {
  snapshot.selectionAttempted = true;
  const std::vector<SyncCoverDemandId> demands(activeDemands_.begin(),
                                               activeDemands_.end());
  std::vector<SyncCoverSelectionSeed> seeds = {{1, legacySeed_}};
  const SyncCoverSelectionResult result =
      solveSyncCoverSelection(universe_, demands, seeds);
  snapshot.selectionError = result.error;
  snapshot.searchTruncated = static_cast<bool>(result.truncation);
  const bool candidateUniverseComplete =
      !slotLifecycles_.truncated && !slotProtocols_.truncated &&
      unmaterializableSlotProtocols_ == 0;
  snapshot.optimalityProven =
      result.optimalityProven && candidateUniverseComplete;
  snapshot.solverComponents = result.components.size();
  snapshot.solverEvaluations = result.evaluations;
  snapshot.redundancyEvaluations = result.redundancyEvaluations;
  snapshot.coverageStatistics = result.coverageStatistics;
  snapshot.finalVerificationStatistics = result.finalVerificationStatistics;
  if (!result) {
    return func_.emitError()
           << "canonical covering shadow selection failed with error "
           << static_cast<unsigned>(result.error);
  }
  snapshot.selectedMechanisms = result.mechanisms.size();
  snapshot.actionProfile = result.cost.actionProfile;
  snapshot.barrierActionProfile = result.cost.barrierActionProfile;
  snapshot.selectedResources = result.resources;
  std::map<SyncCoverMechanismId, CanonicalSelectionMechanismRef>
      providerForMechanism;
  for (const auto &[provider, mechanism] : providers_) {
    const bool unique =
        providerForMechanism.emplace(mechanism, provider).second;
    if (!unique) {
      return func_.emitError(
          "internal error: covering mechanism has multiple providers");
    }
  }
  for (SyncCoverMechanismId mechanism : result.mechanisms) {
    auto provider = providerForMechanism.find(mechanism);
    if (provider == providerForMechanism.end()) {
      return func_.emitError(
          "internal error: selected covering mechanism has no provider");
    }
    snapshot.selectedProviders.push_back({mechanism, provider->second});
    if (provider->second.kind ==
        CanonicalSelectionMechanismKind::SlotProtocol) {
      auto recipe = slotProtocolRecipes_.find(provider->second);
      if (recipe == slotProtocolRecipes_.end()) {
        return func_.emitError(
            "internal error: selected slot protocol has no emission recipe");
      }
      snapshot.selectedSlotProtocols.push_back({mechanism, provider->second,
                                                recipe->second.resourceUse,
                                                recipe->second.event});
    }
    const SyncCoverMechanism &selected = universe_.getMechanisms()[mechanism];
    for (auto [resourceUse, use] : llvm::enumerate(selected.resourceUses)) {
      if (use.domain >= universe_.getResourceDomains().size()) {
        return func_.emitError(
            "internal error: selected covering resource use has invalid "
            "domain");
      }
      const SyncCoverResourceDomain &domain =
          universe_.getResourceDomains()[use.domain];
      const std::optional<SyncCoverTimelineInterval> lifetime =
          getSyncCoverResourceLifetime(universe_.getGraph(), selected, use);
      if (!lifetime) {
        return func_.emitError(
            "internal error: selected covering resource use has invalid "
            "lifetime");
      }
      const auto event =
          eventResourceUses_.find(std::make_pair(mechanism, resourceUse));
      const bool materializesAsEvent =
          provider->second.kind == CanonicalSelectionMechanismKind::EventBundle;
      const bool generatedProtocol =
          provider->second.kind ==
          CanonicalSelectionMechanismKind::SlotProtocol;
      const bool hasEventMapping = event != eventResourceUses_.end();
      const bool eventMappingValid =
          materializesAsEvent ? hasEventMapping
                              : generatedProtocol && !hasEventMapping;
      const bool eventMappingInvalid =
          domain.kind == SyncCoverResourceKind::EventId ? !eventMappingValid
                                                        : hasEventMapping;
      if (eventMappingInvalid) {
        return func_.emitError(
            "internal error: selected covering resource use has invalid "
            "event mapping");
      }
      std::optional<std::size_t> materializationEventIndex;
      if (event != eventResourceUses_.end()) {
        materializationEventIndex = event->second;
      }
      snapshot.selectedResourceUses.push_back(
          {mechanism, provider->second, resourceUse, use.domain, domain.kind,
           domain.sourceResource, domain.targetResource, domain.poolIdentity,
           use.scope, use.distance, use.width, *lifetime,
           materializationEventIndex});
    }
  }
  for (const SyncCoverDomainFeasibility &feasibility :
       result.resources.domains) {
    if (feasibility.domain >= universe_.getResourceDomains().size()) {
      return func_.emitError(
          "internal error: selected covering resource domain is invalid");
    }
    const SyncCoverResourceDomain &domain =
        universe_.getResourceDomains()[feasibility.domain];
    for (const SyncCoverResourceAllocation &allocation :
         feasibility.allocations) {
      auto provider = providerForMechanism.find(allocation.owner.mechanism);
      const bool providerMissing = provider == providerForMechanism.end();
      const bool mechanismMissing =
          allocation.owner.mechanism >= universe_.getMechanisms().size();
      if (providerMissing || mechanismMissing) {
        return func_.emitError(
            "internal error: covering allocation has no provider");
      }
      const SyncCoverMechanism &mechanism =
          universe_.getMechanisms()[allocation.owner.mechanism];
      const bool useMissing =
          allocation.owner.resourceUse >= mechanism.resourceUses.size();
      if (useMissing ||
          mechanism.resourceUses[allocation.owner.resourceUse].domain !=
              feasibility.domain) {
        return func_.emitError(
            "internal error: covering allocation has no matching resource "
            "use");
      }
      snapshot.selectedAllocations.push_back(
          {allocation.owner.mechanism, provider->second,
           allocation.owner.resourceUse, feasibility.domain, domain.kind,
           domain.sourceResource, domain.targetResource, allocation.ids});
    }
  }
  if (failed(validateSelectedResourceUsesAgainstUniverse(snapshot))) {
    return failure();
  }
  const CanonicalSyncCoveringAllocationValidation allocationValidation =
      validateCanonicalSyncCoveringAllocation(snapshot);
  if (!allocationValidation) {
    return func_.emitError()
           << "internal error: selected covering allocation is invalid: "
           << static_cast<unsigned>(allocationValidation.error);
  }
  return success();
}

LogicalResult MechanismAdapter::validateSelectedResourceUsesAgainstUniverse(
    const CanonicalSyncCoveringShadowSnapshot &snapshot) {
  for (const CanonicalSyncCoveringSelectedResourceUse &selected :
       snapshot.selectedResourceUses) {
    const bool mechanismMissing =
        selected.mechanism >= universe_.getMechanisms().size();
    if (mechanismMissing) {
      return func_.emitError(
          "internal error: selected covering resource use has no mechanism");
    }
    const SyncCoverMechanism &mechanism =
        universe_.getMechanisms()[selected.mechanism];
    const bool useMissing =
        selected.resourceUse >= mechanism.resourceUses.size();
    if (useMissing) {
      return func_.emitError(
          "internal error: selected covering resource use is out of range");
    }
    const SyncCoverResourceUse &use =
        mechanism.resourceUses[selected.resourceUse];
    const std::optional<SyncCoverTimelineInterval> lifetime =
        getSyncCoverResourceLifetime(universe_.getGraph(), mechanism, use);
    const bool useMismatch =
        !lifetime ||
        !canonicalSyncCoveringResourceUseMatches(selected, use, *lifetime);
    if (useMismatch) {
      return func_.emitError(
          "internal error: selected covering resource use disagrees with "
          "the live mechanism universe");
    }
  }
  return success();
}

LogicalResult
MechanismAdapter::emitMechanismError(StringRef context,
                                     const SyncCoverMechanismResult &result) {
  InFlightDiagnostic diagnostic = func_.emitError()
                                  << "canonical covering " << context
                                  << " failed with mechanism error "
                                  << static_cast<unsigned>(result.error);
  if (result.index) {
    diagnostic << " at index " << *result.index;
  }
  return failure();
}

LogicalResult mlir::pto::runCanonicalSyncCoveringShadowSelection(
    func::FuncOp func, const CanonicalSyncPlan &plan,
    const CanonicalMechanismUniverse &legacyUniverse,
    ArrayRef<CanonicalEventBundleCandidate> selectedEventBundles,
    unsigned eventIdMax,
    const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds,
    SyncCoverGraph &graph, const SyncCoverCandidateIndex &candidateIndex,
    const SyncCoverSlotLifecycleResult &slotLifecycles,
    const SyncCoverSlotProtocolResult &slotProtocols,
    ArrayRef<SyncCoverDemandId> activeDemands,
    const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
        &regionScopes,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    std::function<std::size_t(const CanonicalAnchor &)> getAnchorPosition,
    std::function<std::vector<SyncGraphEdge>(const CanonicalBarrier &)>
        getBarrierCompletionEdges,
    std::function<bool(ArrayRef<CanonicalEvent>)> verifyEventProtocols,
    CanonicalSyncCoveringShadowSnapshot &snapshot) {
  MechanismAdapter adapter(
      func, plan, legacyUniverse, selectedEventBundles, eventIdMax, reservedIds,
      graph, candidateIndex, slotLifecycles, slotProtocols, activeDemands,
      regionScopes, loopScopes,
      std::move(getAnchorPosition), std::move(getBarrierCompletionEdges),
      std::move(verifyEventProtocols));
  return adapter.build(snapshot);
}
