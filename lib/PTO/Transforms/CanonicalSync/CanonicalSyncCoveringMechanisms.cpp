// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncCoveringSelection.h"

#include "PTO/Transforms/CanonicalSync/SyncCoverDescriptorBuilder.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_covering;

MechanismAdapter::MechanismAdapter(
    func::FuncOp func, const CanonicalSyncPlan &plan,
    const CanonicalMechanismUniverse &legacyUniverse,
    ArrayRef<CanonicalEventBundleCandidate> selectedEventBundles,
    unsigned eventIdMax,
    const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds,
    SyncCoverGraph &graph, const SyncCoverCandidateIndex &candidateIndex,
    ArrayRef<SyncCoverDemandId> activeDemands,
    const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
        &regionScopes,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    std::function<std::size_t(const CanonicalAnchor &)> getAnchorPosition,
    std::function<std::vector<SyncGraphEdge>(const CanonicalBarrier &)>
        getBarrierCompletionEdges,
    std::function<bool(ArrayRef<CanonicalEvent>)> verifyEventProtocols)
    : func_(func), plan_(plan), legacyUniverse_(legacyUniverse),
      selectedEventBundles_(selectedEventBundles), eventIdMax_(eventIdMax),
      reservedIds_(reservedIds), universe_(graph),
      candidateIndex_(candidateIndex),
      regionScopes_(regionScopes), loopScopes_(loopScopes),
      getAnchorPosition_(std::move(getAnchorPosition)),
      getBarrierCompletionEdges_(std::move(getBarrierCompletionEdges)),
      verifyEventProtocols_(std::move(verifyEventProtocols)),
      activeDemands_(activeDemands) {}

LogicalResult
MechanismAdapter::build(CanonicalSyncCoveringShadowSnapshot &snapshot) {
  const bool buildFailed =
      !candidateIndex_ ||
      failed(collectEventBundles()) || failed(addEventDomains()) ||
      failed(addBarriers()) || failed(addEventBundles()) ||
      failed(addConflicts()) || failed(buildLegacySeed());
  if (buildFailed) {
    return failure();
  }
  const SyncCoverMechanismResult validation = universe_.validate();
  if (!validation) {
    return emitMechanismError("universe validation", validation);
  }
  snapshot.resourceDomainCount = universe_.getResourceDomains().size();
  snapshot.resourceDomainDetails = universe_.getResourceDomains();
  snapshot.barrierCandidates = legacyUniverse_.barriers.size();
  snapshot.eventBundleCandidates = eventBundles_.size();
  snapshot.candidateMechanisms = universe_.getMechanisms().size();
  snapshot.legacySeedMechanisms = legacySeed_.size();
  return solve(snapshot);
}

LogicalResult MechanismAdapter::collectEventBundles() {
  eventBundles_ = legacyUniverse_.eventBundles;
  for (const CanonicalEventBundleCandidate &selected :
       selectedEventBundles_) {
    const bool known = llvm::any_of(eventBundles_, [&](const auto &candidate) {
      return candidate.id == selected.id;
    });
    if (!known) {
      eventBundles_.push_back(selected);
    }
  }
  llvm::sort(eventBundles_, [](const auto &first, const auto &second) {
    return first.id < second.id;
  });
  const auto duplicate = std::adjacent_find(
      eventBundles_.begin(), eventBundles_.end(),
      [](const auto &first, const auto &second) {
        return first.id == second.id;
      });
  if (duplicate != eventBundles_.end()) {
    return func_.emitError(
        "internal error: duplicate canonical covering event provider");
  }
  return success();
}

LogicalResult MechanismAdapter::addEventDomains() {
  std::set<CanonicalEventDomainKey> keys;
  for (const CanonicalEventBundleCandidate &bundle : eventBundles_) {
    for (const CanonicalEvent &event : bundle.events) {
      keys.insert({event.sourcePipe, event.targetPipe});
    }
  }
  for (const CanonicalEventDomainKey &key : keys) {
    std::vector<unsigned> reserved;
    auto hidden = reservedIds_.find(key);
    if (hidden != reservedIds_.end()) {
      reserved.assign(hidden->second.begin(), hidden->second.end());
    }
    const SyncCoverMechanismResult result = universe_.addResourceDomain(
        SyncCoverResourceKind::EventId,
        static_cast<std::uint32_t>(key.source),
        static_cast<std::uint32_t>(key.target), eventIdMax_, 0,
        std::move(reserved));
    if (!result || !result.index) {
      return emitMechanismError("event-domain insertion", result);
    }
    domains_.emplace(key, *result.index);
  }
  return success();
}

std::optional<SyncCoverEdge>
MechanismAdapter::translateBarrierEdge(const SyncGraphEdge &legacy) const {
  const std::optional<SyncCoverScopeId> scope = getEndpointScope(
      universe_.getGraph(), legacy.source, legacy.target);
  if (!scope) {
    return std::nullopt;
  }
  SyncCoverEdge edge;
  edge.source = legacy.source;
  edge.target = legacy.target;
  edge.kind = SyncCoverEdgeKind::CompletionSupply;
  edge.scope = *scope;
  return edge;
}

LogicalResult MechanismAdapter::addBarriers() {
  for (const CanonicalBarrierCandidate &candidate :
       legacyUniverse_.barriers) {
    const CanonicalSelectionMechanismRef provider{
        CanonicalSelectionMechanismKind::Barrier, candidate.id};
    const std::optional<std::uint64_t> identity =
        encodeProviderIdentity(provider.kind, provider.id);
    const std::optional<SyncCoverScopeId> anchorScope =
        getAnchorOccurrenceScope(candidate.barrier.anchor, regionScopes_);
    if (!identity || !anchorScope) {
      return func_.emitError(
          "internal error: canonical covering barrier identity or scope is "
          "invalid");
    }
    SyncCoverMechanismDescriptor descriptor;
    descriptor.kind = SyncCoverMechanismKind::Barrier;
    descriptor.providerIdentity = *identity;
    const SyncCoverAnchor anchor{
        SyncCoverAnchorKind::TimelinePoint, 0, *anchorScope,
        getAnchorPosition_(candidate.barrier.anchor)};
    descriptor.barrier = SyncCoverBarrierPlacement{
        static_cast<std::uint32_t>(candidate.barrier.pipe), anchor,
        *anchorScope};
    std::set<std::tuple<std::size_t, std::size_t, unsigned,
                        SyncCoverScopeId>>
        seen;
    for (const SyncGraphEdge &legacy :
         getBarrierCompletionEdges_(candidate.barrier)) {
      const std::optional<SyncCoverEdge> edge = translateBarrierEdge(legacy);
      const bool supplies = edge && syncCoverBarrierCanSupply(
                                        universe_.getGraph(),
                                        *descriptor.barrier, *edge);
      if (supplies &&
          seen.emplace(edge->source, edge->target, 0, edge->scope).second) {
        descriptor.supplyEdges.push_back(*edge);
      }
    }
    for (std::size_t requirement : candidate.barrier.requirements) {
      if (requirement >= universe_.getGraph().getDemands().size()) {
        return func_.emitError(
            "internal error: canonical covering barrier requirement is out "
            "of range");
      }
      const SyncCoverDemand &demand =
          universe_.getGraph().getDemands()[requirement];
      if (!seen.emplace(demand.source, demand.target, demand.distance,
                        demand.scope)
               .second) {
        continue;
      }
      SyncCoverEdge edge;
      edge.source = demand.source;
      edge.target = demand.target;
      edge.kind = SyncCoverEdgeKind::CompletionSupply;
      edge.scope = demand.scope;
      edge.distance = demand.distance;
      if (!syncCoverBarrierCanSupply(universe_.getGraph(),
                                     *descriptor.barrier, edge)) {
        return func_.emitError(
            "internal error: canonical covering barrier cannot supply its "
            "declared requirement");
      }
      descriptor.supplyEdges.push_back(std::move(edge));
    }
    const SyncCoverMechanismResult result =
        universe_.addMechanism(descriptor);
    if (!result || !result.index) {
      return emitMechanismError("barrier insertion", result);
    }
    providers_.emplace(provider, *result.index);
  }
  return success();
}

LogicalResult MechanismAdapter::addEventBundles() {
  for (const CanonicalEventBundleCandidate &bundle : eventBundles_) {
    const CanonicalSelectionMechanismRef provider{
        CanonicalSelectionMechanismKind::EventBundle, bundle.id};
    const std::optional<std::uint64_t> identity =
        encodeProviderIdentity(provider.kind, provider.id);
    if (!identity) {
      return func_.emitError(
          "internal error: canonical covering event provider is out of "
          "range");
    }

    SyncCoverMechanismResult result;
    std::vector<std::size_t> eventResourceUses;
    if (isCanonicalForwardEvent(bundle, universe_.getGraph(),
                                getAnchorPosition_)) {
      const CanonicalEvent &event = bundle.events.front();
      const CanonicalEventDomainKey key{event.sourcePipe, event.targetPipe};
      const std::optional<SyncCoverScopeId> scope = getEndpointScope(
          universe_.getGraph(), event.source, event.target);
      auto domain = domains_.find(key);
      if (!scope || domain == domains_.end()) {
        return func_.emitError(
            "internal error: canonical covering standalone event is "
            "unmappable");
      }
      const auto descriptor = makeSyncCoverCanonicalEvent(
          universe_.getResourceDomains()[domain->second], event.source,
          event.target, *scope, event.width, *identity);
      if (!descriptor) {
        return func_.emitError(
            "internal error: canonical covering standalone event is "
            "invalid");
      }
      eventResourceUses.push_back(0);
      result = universe_.addMechanism(*descriptor);
    } else {
      const auto translated = translateVerifiedEventBundle(
          bundle, *identity, domains_, universe_, regionScopes_, loopScopes_,
          getAnchorPosition_);
      if (!translated) {
        return func_.emitError(
            "internal error: canonical covering event protocol is "
            "unmappable");
      }
      const TranslatedEventBundleMechanism expected = *translated;
      eventResourceUses = expected.eventResourceUses;
      result = universe_.addVerifiedProtocol(
          expected.descriptor, [&, expected](const auto &actual) {
            return verifyTranslatedEventBundleCorrespondence(
                       bundle, expected, actual, universe_, domains_,
                       regionScopes_, loopScopes_, getAnchorPosition_) &&
                   verifyBundleShape(bundle, plan_.getOwnershipCycles(),
                                     plan_.getNodes()) &&
                   verifyEventProtocols_(bundle.events);
          });
    }
    if (!result || !result.index) {
      InFlightDiagnostic diagnostic = func_.emitError()
                                      << "canonical covering event-bundle["
                                      << bundle.id
                                      << "] insertion failed with mechanism "
                                         "error "
                                      << static_cast<unsigned>(result.error);
      if (result.index) {
        diagnostic << " at index " << *result.index;
      }
      diagnostic << "; kind=" << static_cast<unsigned>(bundle.kind)
                 << " events=" << bundle.events.size();
      for (auto [index, event] : llvm::enumerate(bundle.events)) {
        diagnostic << "; event[" << index << "]=" << event.source << "->"
                   << event.target << " width=" << event.width
                   << " distance=" << event.iterationDistance
                   << " scope-loop=" << static_cast<bool>(event.scopeLoop)
                   << " resource-scope-loop="
                   << static_cast<bool>(event.resourceScopeLoop)
                   << " recurrence-loop="
                   << static_cast<bool>(event.recurrenceLoop)
                   << " drain-loop="
                   << static_cast<bool>(event.forwardDrainLoop)
                   << " completions=" << event.completions.size();
        for (const CanonicalEventCompletion &completion : event.completions) {
          diagnostic << ',' << completion.iterationDistance;
        }
      }
      return failure();
    }
    const bool eventMappingComplete =
        eventResourceUses.size() == bundle.events.size();
    if (!eventMappingComplete) {
      return func_.emitError(
          "internal error: canonical covering event/use mapping is incomplete");
    }
    for (auto [eventIndex, resourceUse] :
         llvm::enumerate(eventResourceUses)) {
      const bool unique =
          eventResourceUses_
              .emplace(std::make_pair(*result.index, resourceUse), eventIndex)
              .second;
      if (!unique) {
        return func_.emitError(
            "internal error: duplicate canonical covering event/use mapping");
      }
    }
    providers_.emplace(provider, *result.index);
  }
  return success();
}

LogicalResult MechanismAdapter::addConflicts() {
  for (const CanonicalEventBundleCandidate &bundle : eventBundles_) {
    const CanonicalSelectionMechanismRef firstProvider{
        CanonicalSelectionMechanismKind::EventBundle, bundle.id};
    auto first = providers_.find(firstProvider);
    if (first == providers_.end()) {
      return func_.emitError(
          "internal error: canonical covering provider map is incomplete");
    }
    for (std::size_t conflict : bundle.conflicts) {
      if (bundle.id >= conflict) {
        continue;
      }
      const CanonicalSelectionMechanismRef secondProvider{
          CanonicalSelectionMechanismKind::EventBundle, conflict};
      auto second = providers_.find(secondProvider);
      if (second == providers_.end()) {
        continue;
      }
      const SyncCoverMechanismResult result =
          universe_.addConflict(first->second, second->second);
      if (!result) {
        return emitMechanismError("conflict insertion", result);
      }
    }
  }
  return success();
}
