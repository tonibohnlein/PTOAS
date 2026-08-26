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

namespace {

bool edgeMatchesDemand(const SyncCoverEdge &edge,
                       const SyncCoverDemand &demand) {
  return edge.source == demand.source && edge.target == demand.target &&
         edge.scope == demand.scope && edge.distance == demand.distance &&
         edge.sourceGuard.literals == demand.sourceGuard.literals &&
         edge.targetGuard.literals == demand.targetGuard.literals;
}

std::vector<SyncCoverMechanismId>
mergeMembers(const std::vector<SyncCoverMechanismId> &first,
             const std::vector<SyncCoverMechanismId> &second) {
  std::vector<SyncCoverMechanismId> result;
  result.reserve(first.size() + second.size());
  std::set_union(first.begin(), first.end(), second.begin(), second.end(),
                 std::back_inserter(result));
  return result;
}

std::vector<SyncCoverDemandId>
collectSlotLifecycleCoverage(const SyncCoverGraph &graph,
                             const SyncCoverSlotLifecycle &lifecycle,
                             ArrayRef<SyncCoverDemandId> activeDemands) {
  const std::set<SyncCoverStorageAccessId> managed(
      lifecycle.managedAccesses.begin(), lifecycle.managedAccesses.end());
  std::vector<SyncCoverDemandId> result;
  for (SyncCoverDemandId demandId : activeDemands) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (demand.distance == 0 || demand.distance > lifecycle.distance ||
        demand.scope != lifecycle.recurrenceScope ||
        demand.storageProvenance != SyncCoverStorageProvenance::Complete) {
      continue;
    }
    const bool exactManagedWitness = llvm::any_of(
        demand.storageWitnesses, [&](SyncCoverStorageWitnessId witnessId) {
          if (witnessId >= graph.getStorageWitnesses().size()) {
            return false;
          }
          const SyncCoverStorageWitness &witness =
              graph.getStorageWitnesses()[witnessId];
          return witness.overlap == lifecycle.slot.extent &&
                 managed.count(witness.sourceAccess) != 0 &&
                 managed.count(witness.targetAccess) != 0;
        });
    if (exactManagedWitness) {
      result.push_back(demandId);
    }
  }
  return result;
}

void mergeColumn(SyncCoverVerifiedFactoryColumn &target,
                 const SyncCoverVerifiedFactoryColumn &source) {
  target.members = mergeMembers(target.members, source.members);
  std::vector<SyncCoverDemandId> demands;
  demands.reserve(target.demands.size() + source.demands.size());
  std::set_union(target.demands.begin(), target.demands.end(),
                 source.demands.begin(), source.demands.end(),
                 std::back_inserter(demands));
  target.demands = std::move(demands);
}

} // namespace

LogicalResult
MechanismAdapter::solve(CanonicalSyncCoveringSnapshot &snapshot) {
  snapshot.selectionAttempted = true;
  const std::vector<SyncCoverDemandId> demands(activeDemands_.begin(),
                                               activeDemands_.end());
  std::vector<SyncCoverMechanismId> protocolMechanisms;
  for (const CanonicalEventBundleCandidate &bundle : eventBundles_) {
    const bool ownershipProtocol =
        bundle.kind == CanonicalEventBundleKind::Ownership ||
        bundle.kind == CanonicalEventBundleKind::CompositeOwnership;
    if (!ownershipProtocol) {
      continue;
    }
    auto provider = providers_.find(
        {CanonicalSelectionMechanismKind::EventBundle, bundle.id});
    if (provider != providers_.end()) {
      protocolMechanisms.push_back(provider->second);
    }
  }
  for (const SyncCoverSlotProtocolCandidate &protocol :
       slotProtocols_.candidates) {
    auto provider = providers_.find(
        {CanonicalSelectionMechanismKind::SlotProtocol, protocol.id});
    if (provider != providers_.end()) {
      protocolMechanisms.push_back(provider->second);
    }
  }
  llvm::sort(protocolMechanisms);
  protocolMechanisms.erase(
      std::unique(protocolMechanisms.begin(), protocolMechanisms.end()),
      protocolMechanisms.end());
  std::vector<SyncCoverVerifiedFactoryColumn> protocolColumns;
  for (std::size_t first = 0; first < protocolMechanisms.size(); ++first) {
    for (std::size_t second = first + 1;
         second < protocolMechanisms.size(); ++second) {
      std::vector<SyncCoverMechanismId> column{
          protocolMechanisms[first], protocolMechanisms[second]};
      if (universe_.evaluateResourceSelection(column)) {
        protocolColumns.push_back({std::move(column), {}});
      }
    }
  }
  const auto opportunities = candidateIndex_.getOpportunities();
  if (!opportunities) {
    return func_.emitError(
        "internal error: canonical covering candidate index is stale");
  }
  std::map<SyncCoverMechanismId, CanonicalSelectionMechanismRef>
      providerForMechanism;
  for (const auto &[provider, mechanism] : providers_) {
    if (!providerForMechanism.emplace(mechanism, provider).second) {
      return func_.emitError(
          "internal error: covering mechanism has multiple providers");
    }
  }
  std::map<SyncCoverScopeId, SyncCoverVerifiedFactoryColumn> pipelineColumns;
  for (const SyncCoverSlotProtocolCandidate &protocol :
       slotProtocols_.candidates) {
    if (protocol.lifecycle >= slotLifecycles_.lifecycles.size()) {
      return func_.emitError(
          "internal error: slot protocol references an invalid lifecycle");
    }
    auto releaseProvider = providers_.find(
        {CanonicalSelectionMechanismKind::SlotProtocol, protocol.id});
    if (releaseProvider == providers_.end()) {
      continue;
    }
    std::set<SyncCoverDemandId> uncovered;
    for (SyncCoverCandidateOpportunityId readyId :
         slotLifecycles_.lifecycles[protocol.lifecycle].ready) {
      if (readyId >= opportunities.value->size()) {
        return func_.emitError(
            "internal error: slot lifecycle has an invalid opportunity");
      }
      uncovered.insert((*opportunities.value)[readyId].demand);
    }
    std::vector<SyncCoverMechanismId> lifecycleColumn{
        releaseProvider->second};
    while (!uncovered.empty()) {
      std::optional<SyncCoverMechanismId> best;
      std::size_t bestCoverage = 0;
      for (const SyncCoverMechanism &mechanism : universe_.getMechanisms()) {
        auto provider = providerForMechanism.find(mechanism.id);
        if (provider == providerForMechanism.end() ||
            provider->second.kind !=
                CanonicalSelectionMechanismKind::EventBundle ||
            std::binary_search(lifecycleColumn.begin(), lifecycleColumn.end(),
                               mechanism.id)) {
          continue;
        }
        std::size_t coverage = 0;
        for (SyncCoverDemandId demand : uncovered) {
          if (demand >= universe_.getGraph().getDemands().size()) {
            return func_.emitError(
                "internal error: slot lifecycle demand is out of range");
          }
          const SyncCoverDemand &requirement =
              universe_.getGraph().getDemands()[demand];
          const bool supplies = llvm::any_of(
              mechanism.supplyEdges, [&](std::size_t edge) {
                return edge < universe_.getGraph().getEdges().size() &&
                       edgeMatchesDemand(
                           universe_.getGraph().getEdges()[edge], requirement);
              });
          coverage += static_cast<std::size_t>(supplies);
        }
        if (coverage > bestCoverage ||
            (coverage == bestCoverage && coverage != 0 &&
             (!best || mechanism.id < *best))) {
          best = mechanism.id;
          bestCoverage = coverage;
        }
      }
      if (!best) {
        lifecycleColumn.clear();
        break;
      }
      lifecycleColumn.insert(
          std::lower_bound(lifecycleColumn.begin(), lifecycleColumn.end(),
                           *best),
          *best);
      for (auto demand = uncovered.begin(); demand != uncovered.end();) {
        const SyncCoverDemand &requirement =
            universe_.getGraph().getDemands()[*demand];
        const bool supplied = llvm::any_of(
            universe_.getMechanisms()[*best].supplyEdges,
            [&](std::size_t edge) {
              return edge < universe_.getGraph().getEdges().size() &&
                     edgeMatchesDemand(universe_.getGraph().getEdges()[edge],
                                       requirement);
            });
        demand = supplied ? uncovered.erase(demand) : std::next(demand);
      }
    }
    if (lifecycleColumn.size() > 1 &&
        universe_.evaluateResourceSelection(lifecycleColumn)) {
      SyncCoverVerifiedFactoryColumn column{
          std::move(lifecycleColumn),
          collectSlotLifecycleCoverage(
              universe_.getGraph(),
              slotLifecycles_.lifecycles[protocol.lifecycle], demands)};
      mergeColumn(pipelineColumns[protocol.recurrenceScope], column);
      protocolColumns.push_back(std::move(column));
    }
  }
  for (auto &[scope, pipeline] : pipelineColumns) {
    (void)scope;
    if (pipeline.members.size() > 1 &&
        universe_.evaluateResourceSelection(pipeline.members)) {
      protocolColumns.push_back(std::move(pipeline));
    }
  }
  // Round-trip composition columns: a recurrence demand can be covered by a
  // carried supply mechanism composed with a reverse distance-zero event
  // (the classic cross-pipe round trip), joined to the demand endpoints by
  // fixed issue-order paths. Enumerate those shapes structurally; the claims
  // are not trusted here: grounding proves every claimed demand with an
  // oracle witness and keeps only what it proves, so an unsound proposal
  // costs one grounding query.
  {
    const SyncCoverGraph &graph = universe_.getGraph();
    const std::size_t nodeCount = graph.getNodes().size();
    // Forward reachability over distance-zero fixed edges.
    std::vector<llvm::SmallVector<SyncCoverNodeId, 8>> successors(nodeCount);
    for (const SyncCoverEdge &edge : graph.getEdges()) {
      if (!edge.mechanism && edge.distance == 0 && edge.source < nodeCount &&
          edge.target < nodeCount) {
        successors[edge.source].push_back(edge.target);
      }
    }
    const std::size_t words = (nodeCount + 63) / 64;
    std::vector<std::uint64_t> reachable(nodeCount * words, 0);
    const auto markReachable = [&](std::size_t from, std::size_t to) {
      reachable[from * words + to / 64] |= std::uint64_t{1} << (to % 64);
    };
    const auto reaches = [&](std::size_t from, std::size_t to) {
      return from == to ||
             (reachable[from * words + to / 64] &
              (std::uint64_t{1} << (to % 64))) != 0;
    };
    for (std::size_t node = 0; node < nodeCount; ++node) {
      llvm::SmallVector<SyncCoverNodeId, 16> stack(successors[node].begin(),
                                                   successors[node].end());
      while (!stack.empty()) {
        const SyncCoverNodeId next = stack.pop_back_val();
        if (reaches(node, next)) {
          continue;
        }
        markReachable(node, next);
        stack.append(successors[next].begin(), successors[next].end());
      }
    }
    struct SupplyEntry {
      SyncCoverNodeId source = 0;
      SyncCoverNodeId target = 0;
      SyncCoverMechanismId mechanism = 0;
    };
    std::map<unsigned, std::vector<SupplyEntry>> suppliesByDistance;
    std::map<std::pair<SyncCoverNodeId, SyncCoverNodeId>,
             llvm::SmallVector<SyncCoverMechanismId, 2>>
        forwardByEndpoints;
    for (const SyncCoverMechanism &mechanism : universe_.getMechanisms()) {
      if (mechanism.kind == SyncCoverMechanismKind::Barrier) {
        continue;
      }
      for (std::size_t edgeIndex : mechanism.supplyEdges) {
        if (edgeIndex >= graph.getEdges().size()) {
          continue;
        }
        const SyncCoverEdge &edge = graph.getEdges()[edgeIndex];
        suppliesByDistance[edge.distance].push_back(
            {edge.source, edge.target, mechanism.id});
        if (edge.distance == 0) {
          forwardByEndpoints[{edge.source, edge.target}].push_back(
              mechanism.id);
        }
      }
    }
    constexpr std::size_t kMaximumRoundTripColumns = 512;
    constexpr std::size_t kMaximumPairsPerDemand = 4;
    constexpr std::size_t kMaximumRoundTripClaims = 16384;
    std::size_t totalClaims = 0;
    std::map<std::vector<SyncCoverMechanismId>, std::vector<SyncCoverDemandId>>
        roundTrips;
    std::set<SyncCoverDemandCoverageKey> proposedKeys;
    for (SyncCoverDemandId demand : demands) {
      const SyncCoverDemand &requirement = graph.getDemands()[demand];
      if (requirement.distance == 0 ||
          totalClaims == kMaximumRoundTripClaims) {
        continue;
      }
      // One proposal round per coverage key: duplicate demands share the
      // grounded row, so claiming the first is claiming them all.
      if (!proposedKeys
               .insert(makeSyncCoverDemandCoverageKey(graph, demand))
               .second) {
        continue;
      }
      const auto carried = suppliesByDistance.find(requirement.distance);
      if (carried == suppliesByDistance.end()) {
        continue;
      }
      // Tight anchors first: a ring sharing the demand's own endpoints is
      // the shape the composition was observed to need; transitively placed
      // rings only fill the remaining proposal budget.
      llvm::SmallVector<std::tuple<unsigned, std::size_t, std::size_t>, 8>
          rankedRings;
      for (std::size_t index = 0; index < carried->second.size(); ++index) {
        const SupplyEntry &ring = carried->second[index];
        // The demand source must fixed-reach the ring source; speculative
        // entries through the forward event traded one recovered plan for
        // two regressions, so composite shapes beyond this stay with the
        // sharing-aware follow-up.
        if (!reaches(requirement.source, ring.source)) {
          continue;
        }
        // Tight anchors and specific mechanisms first: a single-supply ring
        // on the demand's own endpoints is the observed composition shape;
        // broad carried mechanisms only fill the remaining budget.
        const unsigned rank =
            (ring.source == requirement.source ? 0u : 1u) +
            (ring.target == requirement.target ? 0u : 1u);
        const std::size_t breadth =
            universe_.getMechanisms()[ring.mechanism].supplyEdges.size();
        rankedRings.push_back({rank, breadth, index});
      }
      llvm::stable_sort(rankedRings);
      llvm::SmallVector<std::vector<SyncCoverMechanismId>, 4> proposed;
      for (const auto &ranked : rankedRings) {
        if (proposed.size() == kMaximumPairsPerDemand ||
            totalClaims == kMaximumRoundTripClaims) {
          break;
        }
        const SupplyEntry &ring = carried->second[std::get<2>(ranked)];
        const auto reversing =
            forwardByEndpoints.find({ring.target, ring.source});
        if (reversing == forwardByEndpoints.end()) {
          continue;
        }
        const bool reachesTarget =
            reaches(ring.target, requirement.target) ||
            reaches(ring.source, requirement.target);
        if (!reachesTarget) {
          continue;
        }
        for (SyncCoverMechanismId forwardMechanism : reversing->second) {
          if (proposed.size() == kMaximumPairsPerDemand ||
              totalClaims == kMaximumRoundTripClaims) {
            break;
          }
          if (forwardMechanism == ring.mechanism) {
            continue;
          }
          std::vector<SyncCoverMechanismId> members{
              std::min(ring.mechanism, forwardMechanism),
              std::max(ring.mechanism, forwardMechanism)};
          if (llvm::is_contained(proposed, members)) {
            continue;
          }
          const auto existing = roundTrips.find(members);
          if (existing != roundTrips.end()) {
            existing->second.push_back(demand);
            ++totalClaims;
            proposed.push_back(std::move(members));
            continue;
          }
          if (roundTrips.size() == kMaximumRoundTripColumns) {
            continue;
          }
          ++totalClaims;
          proposed.push_back(members);
          roundTrips.emplace(std::move(members),
                             std::vector<SyncCoverDemandId>{demand});
        }
      }
    }
    for (auto &entry : roundTrips) {
      std::vector<SyncCoverDemandId> claimed = std::move(entry.second);
      std::sort(claimed.begin(), claimed.end());
      claimed.erase(std::unique(claimed.begin(), claimed.end()),
                    claimed.end());
      protocolColumns.push_back({entry.first, std::move(claimed)});
    }
  }
  llvm::sort(protocolColumns, [](const auto &first, const auto &second) {
    return std::tie(first.members, first.demands) <
           std::tie(second.members, second.demands);
  });
  protocolColumns.erase(
      std::unique(protocolColumns.begin(), protocolColumns.end(),
                  [](const auto &first, const auto &second) {
                    return first.members == second.members &&
                           first.demands == second.demands;
                  }),
      protocolColumns.end());
  const SyncCoverSelectionResult result =
      solveSyncCoverSelection(universe_, demands, {}, protocolColumns);
  snapshot.selectionError = result.error;
  snapshot.solverEvaluations = result.evaluations;
  snapshot.redundancyEvaluations = result.redundancyEvaluations;
  snapshot.oracleRedundancyChecks = result.oracleRedundancyChecks;
  snapshot.demandsWithoutEventColumn = result.demandsWithoutEventColumn;
  snapshot.coverageStatistics = result.coverageStatistics;
  snapshot.finalVerificationStatistics = result.finalVerificationStatistics;
  if (!result) {
    InFlightDiagnostic diagnostic =
        func_.emitError() << "canonical covering selection failed with error "
                          << static_cast<unsigned>(result.error)
                          << "; evaluations=" << result.evaluations
                          << "; missing-factory-demands="
                          << result.missingFactoryDemands.size();
    constexpr std::size_t kDiagnosticLimit = 16;
    for (SyncCoverDemandId demand :
         ArrayRef<SyncCoverDemandId>(result.missingFactoryDemands)
             .take_front(kDiagnosticLimit)) {
      diagnostic << ' ' << demand;
    }
    diagnostic << "; slot-lifecycles=" << slotLifecycles_.lifecycles.size()
               << "; slot-protocols=" << slotProtocols_.candidates.size()
               << "; slot-path-sensitive="
               << slotProtocols_.pathSensitiveLifecycles
               << "; slot-access-open=" << slotProtocols_.accessOpenLifecycles
               << "; slot-unsupported-effects="
               << slotProtocols_.unsupportedEffectLifecycles
               << "; slot-unsupported-distance="
               << slotProtocols_.unsupportedDistanceReleases
               << "; slot-non-boundary="
               << slotProtocols_.nonBoundaryReleases;
    if (result.failedFinalDemand) {
      diagnostic << "; failed-final-demand=" << *result.failedFinalDemand;
      if (*result.failedFinalDemand < universe_.getGraph().getDemands().size()) {
        const SyncCoverDemand &failed =
            universe_.getGraph().getDemands()[*result.failedFinalDemand];
        diagnostic << '(' << failed.source << "->" << failed.target
                   << ",scope=" << failed.scope
                   << ",distance=" << failed.distance << ')';
      }
    }
    diagnostic << "; selected=";
    for (SyncCoverMechanismId mechanism : result.mechanisms) {
      diagnostic << ' ' << mechanism;
      auto provider = providerForMechanism.find(mechanism);
      if (provider != providerForMechanism.end()) {
        diagnostic << ':'
                   << stringifyCanonicalSelectionMechanismKind(
                          provider->second.kind)
                   << '[' << provider->second.id << ']';
      }
    }
    return failure();
  }
  snapshot.selectedMechanisms = result.mechanisms.size();
  snapshot.actionProfile = result.cost.actionProfile;
  snapshot.barrierActionProfile = result.cost.barrierActionProfile;
  snapshot.selectedResources = result.resources;
  for (SyncCoverMechanismId mechanism : result.mechanisms) {
    auto provider = providerForMechanism.find(mechanism);
    if (provider == providerForMechanism.end()) {
      return func_.emitError(
          "internal error: selected covering mechanism has no provider");
    }
    snapshot.selectedProviders.push_back({mechanism, provider->second});
    if (provider->second.kind ==
        CanonicalSelectionMechanismKind::Barrier) {
      auto recipe = barrierRecipes_.find(provider->second);
      if (recipe == barrierRecipes_.end()) {
        return func_.emitError(
            "internal error: selected barrier has no emission recipe");
      }
      snapshot.selectedBarriers.push_back(
          {mechanism, provider->second, recipe->second});
    }
    if (provider->second.kind ==
        CanonicalSelectionMechanismKind::EventBundle) {
      auto recipe = llvm::find_if(eventBundles_, [&](const auto &bundle) {
        return bundle.id == provider->second.id;
      });
      if (recipe == eventBundles_.end()) {
        return func_.emitError(
            "internal error: selected event bundle has no emission recipe");
      }
      CanonicalSyncCoveringSelectedEventBundle selectedRecipe;
      selectedRecipe.mechanism = mechanism;
      selectedRecipe.provider = provider->second;
      selectedRecipe.bundleId = recipe->id;
      selectedRecipe.kind = recipe->kind;
      selectedRecipe.events.assign(recipe->events.begin(),
                                   recipe->events.end());
      snapshot.selectedEventBundles.push_back(std::move(selectedRecipe));
    }
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
    const CanonicalSyncCoveringSnapshot &snapshot) {
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

LogicalResult mlir::pto::runCanonicalSyncCoveringSelection(
    func::FuncOp func, const CanonicalSyncPlan &plan,
    const CanonicalMechanismUniverse &candidateUniverse,
    ArrayRef<CanonicalEventBundleCandidate> selectedEventBundles,
    unsigned eventIdMax,
    const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds,
    SyncCoverGraph &graph, const SyncCoverCandidateIndex &candidateIndex,
    const SyncCoverSlotLifecycleResult &slotLifecycles,
    const SyncCoverSlotProtocolResult &slotProtocols,
    SyncCoverTargetCapabilities target,
    ArrayRef<SyncCoverDemandId> activeDemands,
    const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
        &regionScopes,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    std::function<std::size_t(const CanonicalAnchor &)> getAnchorPosition,
    std::function<std::vector<SyncGraphEdge>(const CanonicalBarrier &)>
        getBarrierCompletionEdges,
    std::function<bool(ArrayRef<CanonicalEvent>)> verifyEventProtocols,
    CanonicalSyncCoveringSnapshot &snapshot) {
  MechanismAdapter adapter(
      func, plan, candidateUniverse, selectedEventBundles, eventIdMax,
      reservedIds,
      graph, candidateIndex, slotLifecycles, slotProtocols, std::move(target),
      activeDemands,
      regionScopes, loopScopes,
      std::move(getAnchorPosition), std::move(getBarrierCompletionEdges),
      std::move(verifyEventProtocols));
  return adapter.build(snapshot);
}
