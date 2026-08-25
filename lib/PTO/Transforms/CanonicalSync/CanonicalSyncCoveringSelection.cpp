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

BarrierFreeOwnershipProviderSet
mlir::pto::canonical_sync_covering::buildBarrierFreeOwnershipProviderSet(
    ArrayRef<CanonicalOwnershipCycle> cycles,
    ArrayRef<CanonicalEventBundleCandidate> bundles) {
  BarrierFreeOwnershipProviderSet result;
  std::vector<const CanonicalEventBundleCandidate *> composites;
  for (const CanonicalEventBundleCandidate &bundle : bundles) {
    if (bundle.kind == CanonicalEventBundleKind::CompositeOwnership) {
      composites.push_back(&bundle);
    }
  }
  if (composites.empty()) {
    return result;
  }
  result.applicable = true;
  if (composites.size() != 1) {
    return result;
  }

  std::set<std::size_t> compositeCycles;
  for (const CanonicalEvent &event : composites.front()->events) {
    if (event.ownershipProtocol) {
      compositeCycles.insert(event.ownershipCycle);
    }
  }
  result.providers.push_back({CanonicalSelectionMechanismKind::EventBundle,
                              composites.front()->id});
  for (const CanonicalOwnershipCycle &cycle : cycles) {
    if (compositeCycles.count(cycle.id) != 0) {
      continue;
    }
    std::vector<const CanonicalEventBundleCandidate *> matches;
    for (const CanonicalEventBundleCandidate &bundle : bundles) {
      const bool nativeProtocol =
          bundle.kind == CanonicalEventBundleKind::Ownership &&
          bundle.protocolIdentity == cycle.id &&
          bundle.ownershipProtocol == cycle.protocol;
      if (nativeProtocol) {
        matches.push_back(&bundle);
      }
    }
    if (matches.size() != 1) {
      result.providers.clear();
      return result;
    }
    result.providers.push_back({CanonicalSelectionMechanismKind::EventBundle,
                                matches.front()->id});
  }
  result.complete = true;
  return result;
}

LogicalResult MechanismAdapter::buildLegacySeed() {
  for (const CanonicalBarrier &barrier : plan_.getBarriers()) {
    auto candidate = llvm::find_if(
        legacyUniverse_.barriers, [&](const auto &entry) {
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

LogicalResult MechanismAdapter::evaluateOwnershipMembership(
    CanonicalSyncCoveringMembershipSnapshot &snapshot) {
  const BarrierFreeOwnershipProviderSet providerSet =
      buildBarrierFreeOwnershipProviderSet(plan_.getOwnershipCycles(),
                                           eventBundles_);
  if (!providerSet.applicable) {
    return success();
  }

  snapshot.attempted = true;
  if (!providerSet.complete) {
    return success();
  }

  std::vector<SyncCoverMechanismId> selected;
  for (const CanonicalSelectionMechanismRef &provider :
       providerSet.providers) {
    auto mechanism = providers_.find(provider);
    if (mechanism == providers_.end()) {
      return func_.emitError(
          "internal error: ownership membership provider was not adapted");
    }
    selected.push_back(mechanism->second);
    snapshot.selectedProviders.push_back({mechanism->second, provider});
  }
  llvm::sort(selected);
  if (std::adjacent_find(selected.begin(), selected.end()) != selected.end()) {
    return func_.emitError(
        "internal error: ownership membership provider is duplicated");
  }
  llvm::sort(snapshot.selectedProviders, [](const auto &first,
                                            const auto &second) {
    return first.mechanism < second.mechanism;
  });
  snapshot.candidateSetComplete = true;
  snapshot.selectedMechanisms = selected.size();

  const SyncCoverMembershipResult membership =
      evaluateSyncCoverMembership(universe_, activeDemands_, selected);
  if (!membership) {
    return func_.emitError()
           << "internal error: ownership membership evaluation failed with "
              "error "
           << static_cast<unsigned>(membership.error);
  }
  snapshot.resourceError = membership.resources.error;
  snapshot.resourceFeasible = membership.resources.resourceFeasible;
  if (membership.cost) {
    snapshot.actionProfile = membership.cost.actionProfile;
    snapshot.barrierActionProfile = membership.cost.barrierActionProfile;
  }
  if (!membership.resources.isValid() ||
      !membership.resources.resourceFeasible) {
    return success();
  }

  std::map<SyncCoverMechanismId, CanonicalSelectionMechanismRef>
      providerForMechanism;
  for (const auto &[provider, mechanism] : providers_) {
    if (!providerForMechanism.emplace(mechanism, provider).second) {
      return func_.emitError(
          "internal error: covering mechanism has multiple providers");
    }
  }
  for (const SyncCoverMembershipDemand &uncovered :
       membership.uncoveredDemands) {
    CanonicalSyncCoveringMembershipCut cut;
    cut.demand = uncovered.demand;
    for (SyncCoverMechanismId mechanism : uncovered.cutMechanisms) {
      auto provider = providerForMechanism.find(mechanism);
      if (provider == providerForMechanism.end()) {
        return func_.emitError(
            "internal error: ownership membership cut has no provider");
      }
      cut.candidates.push_back({mechanism, provider->second});
    }
    snapshot.uncoveredDemands.push_back(std::move(cut));
  }
  snapshot.coverageComplete = membership.coverageComplete;

  const auto findProvider = [&](SyncCoverMechanismId mechanism)
      -> std::optional<CanonicalSyncCoveringSelectedProvider> {
    auto provider = providerForMechanism.find(mechanism);
    if (provider == providerForMechanism.end()) {
      return std::nullopt;
    }
    return CanonicalSyncCoveringSelectedProvider{mechanism,
                                                  provider->second};
  };

  if (plan_.getGMAliasPolicy() !=
      CanonicalGMAliasPolicy::DistinctArgumentsNoAlias) {
    return success();
  }

  if (snapshot.requested) {
    snapshot.barrierFreeCensusAttempted = true;
    std::vector<SyncCoverDemandId> censusDemands;
    censusDemands.reserve(membership.uncoveredDemands.size());
    for (const SyncCoverMembershipDemand &uncovered :
         membership.uncoveredDemands) {
      censusDemands.push_back(uncovered.demand);
    }
    const SyncCoverBarrierFreeCensusResult census =
        evaluateSyncCoverBarrierFreeCensus(universe_, censusDemands);
    if (!census) {
      return func_.emitError()
             << "internal error: barrier-free covering census failed with "
                "error "
             << static_cast<unsigned>(census.error)
             << " coverage-error "
             << static_cast<unsigned>(census.coverageError);
    }
    snapshot.barrierFreeCensusStatistics = census.coverageStatistics;
    for (const SyncCoverBarrierFreeCensusEntry &entry : census.entries) {
      CanonicalSyncCoveringBarrierFreeCensusEntry translated;
      translated.demand = entry.demand;
      translated.status = entry.status;
      translated.reachableStates = entry.reachableStates;
      translated.witnessResources = entry.witnessResources;
      for (SyncCoverMechanismId mechanism : entry.witnessMechanisms) {
        const auto provider = findProvider(mechanism);
        if (!provider) {
          return func_.emitError(
              "internal error: barrier-free census witness has no provider");
        }
        translated.witness.push_back(*provider);
      }
      snapshot.barrierFreeCensus.push_back(std::move(translated));
    }
  }

  snapshot.completionAttempted = true;
  std::vector<SyncCoverMechanismId> allowed = selected;
  for (const CanonicalEventBundleCandidate &bundle : eventBundles_) {
    const CanonicalSelectionMechanismRef provider{
        CanonicalSelectionMechanismKind::EventBundle, bundle.id};
    auto mechanism = providers_.find(provider);
    if (mechanism == providers_.end()) {
      return func_.emitError(
          "internal error: ownership completion provider was not adapted");
    }
    allowed.push_back(mechanism->second);
  }
  llvm::sort(allowed);
  allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
  const SyncCoverCompletionResult completion = completeSyncCoverMembership(
      universe_, activeDemands_, selected, allowed);
  if (!completion) {
    return func_.emitError()
           << "internal error: ownership completion failed with error "
           << static_cast<unsigned>(completion.error);
  }
  snapshot.completionFound = completion.complete;
  snapshot.completionTruncated = completion.truncated;
  snapshot.completionEvaluations = completion.evaluations;
  snapshot.completionRejectionsTruncated =
      completion.rejectionDiagnosticsTruncated;
  snapshot.completionBlockedCutsTruncated =
      completion.blockedCutDiagnosticsTruncated;
  for (const SyncCoverCompletionRejection &rejection :
       completion.rejections) {
    const auto candidate = findProvider(rejection.mechanism);
    if (!candidate) {
      return func_.emitError(
          "internal error: ownership completion rejection has no provider");
    }
    CanonicalSyncCoveringCompletionRejection translated;
    translated.candidate = *candidate;
    translated.kind = rejection.kind;
    translated.resourceError = rejection.resourceError;
    translated.domain = rejection.domain;
    translated.required = rejection.required;
    translated.available = rejection.available;
    if (rejection.firstConflict) {
      translated.firstConflict = findProvider(*rejection.firstConflict);
      if (!translated.firstConflict) {
        return func_.emitError(
            "internal error: ownership completion conflict has no provider");
      }
    }
    if (rejection.secondConflict) {
      translated.secondConflict = findProvider(*rejection.secondConflict);
      if (!translated.secondConflict) {
        return func_.emitError(
            "internal error: ownership completion conflict has no provider");
      }
    }
    snapshot.completionRejections.push_back(std::move(translated));
  }
  for (const SyncCoverCompletionBlockedCut &blocked :
       completion.blockedCuts) {
    CanonicalSyncCoveringCompletionBlockedCut translated;
    translated.demand = blocked.demand;
    translated.reachableStates = blocked.reachableStates;
    for (SyncCoverMechanismId mechanism : blocked.selected) {
      const auto provider = findProvider(mechanism);
      if (!provider) {
        return func_.emitError(
            "internal error: ownership completion blocked selection has no "
            "provider");
      }
      translated.selected.push_back(*provider);
    }
    for (SyncCoverMechanismId mechanism : blocked.mechanisms) {
      const auto provider = findProvider(mechanism);
      if (!provider) {
        return func_.emitError(
            "internal error: ownership completion blocked cut has no "
            "provider");
      }
      translated.candidates.push_back(*provider);
    }
    snapshot.completionBlockedCuts.push_back(std::move(translated));
  }
  if (!completion.complete) {
    return success();
  }
  ownershipCompletionSeed_ = completion.mechanisms;
  snapshot.completionActionProfile = completion.membership.cost.actionProfile;
  snapshot.completionBarrierActionProfile =
      completion.membership.cost.barrierActionProfile;
  for (SyncCoverMechanismId mechanism : completion.mechanisms) {
    auto provider = providerForMechanism.find(mechanism);
    if (provider == providerForMechanism.end()) {
      return func_.emitError(
          "internal error: ownership completion has no provider");
    }
    snapshot.completionProviders.push_back({mechanism, provider->second});
  }
  return success();
}

LogicalResult
MechanismAdapter::solve(CanonicalSyncCoveringShadowSnapshot &snapshot) {
  snapshot.selectionAttempted = true;
  const std::vector<SyncCoverDemandId> demands(activeDemands_.begin(),
                                                activeDemands_.end());
  std::vector<SyncCoverSelectionSeed> seeds = {{1, legacySeed_}};
  if (!ownershipCompletionSeed_.empty()) {
    seeds.push_back({2, ownershipCompletionSeed_});
  }
  const SyncCoverSelectionResult result =
      solveSyncCoverSelection(universe_, demands, seeds);
  snapshot.selectionError = result.error;
  snapshot.searchTruncated = static_cast<bool>(result.truncation);
  snapshot.optimalityProven = result.optimalityProven;
  snapshot.solverComponents = result.components.size();
  snapshot.solverEvaluations = result.evaluations;
  snapshot.redundancyEvaluations = result.redundancyEvaluations;
  snapshot.exchangeStatistics = result.exchangeStatistics;
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
      const bool eventMappingInvalid =
          domain.kind == SyncCoverResourceKind::EventId
              ? event == eventResourceUses_.end()
              : event != eventResourceUses_.end();
      if (eventMappingInvalid) {
        return func_.emitError(
            "internal error: selected covering resource use has invalid "
            "event mapping");
      }
      std::optional<std::size_t> eventIndex;
      if (event != eventResourceUses_.end()) {
        eventIndex = event->second;
      }
      snapshot.selectedResourceUses.push_back(
          {mechanism, provider->second, resourceUse, use.domain, domain.kind,
           domain.sourceResource, domain.targetResource, domain.poolIdentity,
           use.scope, use.distance, use.width, *lifetime, eventIndex});
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

LogicalResult MechanismAdapter::emitMechanismError(
    StringRef context, const SyncCoverMechanismResult &result) {
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
    SyncCoverGraph &graph, ArrayRef<SyncCoverDemandId> activeDemands,
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
      graph, activeDemands, regionScopes, loopScopes,
      std::move(getAnchorPosition), std::move(getBarrierCompletionEdges),
      std::move(verifyEventProtocols));
  return adapter.build(snapshot);
}
