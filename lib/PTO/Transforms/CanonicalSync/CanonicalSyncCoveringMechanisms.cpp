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

#include "PTO/Transforms/CanonicalSync/SyncCoverColumnGeneration.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverDescriptorBuilder.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_covering;

namespace {

struct OwnershipOccurrence {
  unsigned lane = 0;
  std::size_t path = 0;
  std::size_t use = 0;
  bool producer = false;
  bool initial = false;
};

bool intervalContains(const SyncCoverStorageInterval &outer,
                      const SyncCoverStorageInterval &inner) {
  return outer.begin <= inner.begin && inner.end <= outer.end;
}

bool accessContains(const CanonicalMemoryAccess &access,
                    const CanonicalPhysicalSlot &slot,
                    const SyncCoverStorageInterval &overlap) {
  if (access.space != slot.space || !access.knownPhysical ||
      access.unknownRange || access.size == 0 || slot.size == 0 ||
      slot.address > std::numeric_limits<std::uint64_t>::max() - slot.size) {
    return false;
  }
  const SyncCoverStorageInterval slotRange{slot.address,
                                           slot.address + slot.size};
  if (!intervalContains(slotRange, overlap)) {
    return false;
  }
  return llvm::any_of(access.addresses, [&](std::uint64_t address) {
    if (address > std::numeric_limits<std::uint64_t>::max() - access.size) {
      return false;
    }
    return intervalContains({address, address + access.size}, overlap);
  });
}

bool nodeContainsSlotOverlap(const CanonicalSyncNode &node,
                             const CanonicalPhysicalSlot &slot,
                             const SyncCoverStorageInterval &overlap) {
  return llvm::any_of(node.accesses, [&](const CanonicalMemoryAccess &access) {
    return accessContains(access, slot, overlap);
  });
}

std::map<std::size_t, SmallVector<OwnershipOccurrence, 2>>
collectOwnershipOccurrences(const CanonicalOwnershipCycle &cycle) {
  std::map<std::size_t, SmallVector<OwnershipOccurrence, 2>> result;
  for (std::size_t node : cycle.initialProducers) {
    result[node].push_back({cycle.initialReadyLane, 0, 0,
                            /*producer=*/true, /*initial=*/true});
  }
  for (auto indexedPath : llvm::enumerate(cycle.paths)) {
    for (auto indexedUse : llvm::enumerate(indexedPath.value().uses)) {
      const CanonicalOwnershipUse &use = indexedUse.value();
      for (std::size_t node : use.producers) {
        result[node].push_back({use.producerLane, indexedPath.index(),
                               indexedUse.index(), /*producer=*/true,
                               /*initial=*/false});
      }
      for (std::size_t node : use.consumers) {
        result[node].push_back({use.lane, indexedPath.index(),
                               indexedUse.index(), /*producer=*/false,
                               /*initial=*/false});
      }
    }
  }
  return result;
}

bool protocolOrdersAccesses(
    const CanonicalOwnershipCycle &cycle,
    const std::map<std::size_t, SmallVector<OwnershipOccurrence, 2>>
        &occurrences,
    const SyncCoverDemand &demand) {
  auto source = occurrences.find(demand.source);
  auto target = occurrences.find(demand.target);
  if (source == occurrences.end() || target == occurrences.end()) {
    return false;
  }
  for (const OwnershipOccurrence &sourceOccurrence : source->second) {
    for (const OwnershipOccurrence &targetOccurrence : target->second) {
      if (sourceOccurrence.lane != targetOccurrence.lane ||
          sourceOccurrence.lane >= cycle.lanes.size()) {
        continue;
      }
      if (demand.distance != 0) {
        return true;
      }
      if (sourceOccurrence.initial && !targetOccurrence.initial) {
        return true;
      }
      if (!sourceOccurrence.initial && !targetOccurrence.initial &&
          sourceOccurrence.path == targetOccurrence.path) {
        const std::size_t sourcePhase =
            sourceOccurrence.use * 2 + !sourceOccurrence.producer;
        const std::size_t targetPhase =
            targetOccurrence.use * 2 + !targetOccurrence.producer;
        if (sourcePhase < targetPhase) {
          return true;
        }
      }
    }
  }
  return false;
}

bool demandTouchesOwnedSlot(
    const CanonicalOwnershipCycle &cycle, const SyncCoverDemand &demand,
    const SyncCoverGraph &graph, ArrayRef<CanonicalSyncNode> nodes) {
  if (demand.source >= nodes.size() || demand.target >= nodes.size() ||
      demand.storageProvenance != SyncCoverStorageProvenance::Complete ||
      demand.storageWitnesses.empty()) {
    return false;
  }
  const auto &witnesses = graph.getStorageWitnesses();
  for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
    if (witnessId >= witnesses.size()) {
      return false;
    }
    const SyncCoverStorageInterval &overlap = witnesses[witnessId].overlap;
    for (const CanonicalOwnershipLane &lane : cycle.lanes) {
      for (const CanonicalPhysicalSlot &slot : lane.slots) {
        if (nodeContainsSlotOverlap(nodes[demand.source], slot, overlap) &&
            nodeContainsSlotOverlap(nodes[demand.target], slot, overlap)) {
          return true;
        }
      }
    }
  }
  return false;
}

SmallVector<const CanonicalOwnershipCycle *, 3> getBundleOwnershipCycles(
    const CanonicalEventBundleCandidate &bundle,
    ArrayRef<CanonicalOwnershipCycle> cycles) {
  std::set<std::size_t> identities;
  for (const CanonicalEvent &event : bundle.events) {
    if (event.ownershipProtocol && event.ownershipCycle != 0) {
      identities.insert(event.ownershipCycle);
    }
  }
  SmallVector<const CanonicalOwnershipCycle *, 3> result;
  for (std::size_t identity : identities) {
    auto cycle = llvm::find_if(cycles, [&](const CanonicalOwnershipCycle &item) {
      return item.id == identity;
    });
    if (cycle == cycles.end()) {
      return {};
    }
    result.push_back(&*cycle);
  }
  return result;
}

std::vector<SyncCoverEdge> deriveOwnershipCoverageEdges(
    const CanonicalEventBundleCandidate &bundle,
    ArrayRef<CanonicalOwnershipCycle> cycles,
    ArrayRef<CanonicalSyncNode> nodes, const SyncCoverGraph &graph,
    ArrayRef<SyncCoverDemandId> activeDemands) {
  if (bundle.kind != CanonicalEventBundleKind::Ownership &&
      bundle.kind != CanonicalEventBundleKind::CompositeOwnership) {
    return {};
  }
  const SmallVector<const CanonicalOwnershipCycle *, 3> bundleCycles =
      getBundleOwnershipCycles(bundle, cycles);
  if (bundleCycles.empty()) {
    return {};
  }
  std::vector<SyncCoverEdge> result;
  using EdgeKey =
      std::tuple<SyncCoverNodeId, SyncCoverNodeId, SyncCoverScopeId, unsigned,
                 std::vector<SyncCoverGuardLiteral>,
                 std::vector<SyncCoverGuardLiteral>>;
  std::set<EdgeKey> seen;
  for (const CanonicalOwnershipCycle *cycle : bundleCycles) {
    const auto occurrences = collectOwnershipOccurrences(*cycle);
    for (SyncCoverDemandId demandId : activeDemands) {
      if (demandId >= graph.getDemands().size()) {
        return {};
      }
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      const bool relevant =
          demand.source < graph.getNodes().size() &&
          demand.target < graph.getNodes().size() &&
          protocolOrdersAccesses(*cycle, occurrences, demand) &&
          demandTouchesOwnedSlot(*cycle, demand, graph, nodes);
      if (!relevant ||
          !seen.emplace(demand.source, demand.target, demand.scope,
                        demand.distance, demand.sourceGuard.literals,
                        demand.targetGuard.literals)
               .second) {
        continue;
      }
      result.push_back({demand.source, demand.target,
                        SyncCoverEdgeKind::CompletionSupply, demand.scope,
                        demand.distance, demand.sourceGuard,
                        demand.targetGuard, std::nullopt});
    }
  }
  return result;
}

bool sameCoverageEdge(const SyncCoverEdge &first,
                      const SyncCoverEdge &second) {
  return first.source == second.source && first.target == second.target &&
         first.kind == second.kind && first.scope == second.scope &&
         first.distance == second.distance &&
         first.sourceGuard.literals == second.sourceGuard.literals &&
         first.targetGuard.literals == second.targetGuard.literals;
}

std::optional<CanonicalAnchor>
getCanonicalAnchor(const SyncCoverAnchor &anchor,
                   ArrayRef<CanonicalSyncNode> nodes) {
  const bool supported = anchor.node < nodes.size() &&
                         (anchor.kind == SyncCoverAnchorKind::BeforeNode ||
                          anchor.kind == SyncCoverAnchorKind::AfterNode);
  if (!supported) {
    return std::nullopt;
  }
  return CanonicalAnchor{nodes[anchor.node].operation,
                         anchor.kind == SyncCoverAnchorKind::BeforeNode};
}

} // namespace

MechanismAdapter::MechanismAdapter(
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
    std::function<bool(ArrayRef<CanonicalEvent>)> verifyEventProtocols)
    : func_(func), plan_(plan), candidateUniverse_(candidateUniverse),
      selectedEventBundles_(selectedEventBundles), eventIdMax_(eventIdMax),
      reservedIds_(reservedIds), universe_(graph),
      candidateIndex_(candidateIndex), slotLifecycles_(slotLifecycles),
      slotProtocols_(slotProtocols), target_(std::move(target)),
      regionScopes_(regionScopes), loopScopes_(loopScopes),
      getAnchorPosition_(std::move(getAnchorPosition)),
      getBarrierCompletionEdges_(std::move(getBarrierCompletionEdges)),
      verifyEventProtocols_(std::move(verifyEventProtocols)),
      activeDemands_(activeDemands) {}

LogicalResult
MechanismAdapter::build(CanonicalSyncCoveringSnapshot &snapshot) {
  const bool buildFailed = !candidateIndex_ || failed(collectEventBundles()) ||
                           failed(addEventDomains()) || failed(addBarriers()) ||
                           failed(addSlotProtocols()) ||
                           failed(addEventBundles()) ||
                           failed(addGeneratedColumns(snapshot)) ||
                           failed(addConflicts());
  if (buildFailed) {
    return failure();
  }
  const SyncCoverMechanismResult validation = universe_.validate();
  if (!validation) {
    return emitMechanismError("universe validation", validation);
  }
  snapshot.resourceDomainCount = universe_.getResourceDomains().size();
  snapshot.resourceDomainDetails = universe_.getResourceDomains();
  snapshot.barrierCandidates = barrierRecipes_.size();
  snapshot.eventBundleCandidates = eventBundles_.size();
  snapshot.slotProtocolMechanismCandidates =
      llvm::count_if(providers_, [](const auto &entry) {
        return entry.first.kind ==
               CanonicalSelectionMechanismKind::SlotProtocol;
      });
  snapshot.unmaterializableSlotProtocolCandidates =
      unmaterializableSlotProtocols_;
  snapshot.candidateMechanisms = universe_.getMechanisms().size();
  return solve(snapshot);
}

LogicalResult MechanismAdapter::collectEventBundles() {
  eventBundles_ = candidateUniverse_.eventBundles;
  for (const CanonicalEventBundleCandidate &selected : selectedEventBundles_) {
    const bool known = llvm::any_of(eventBundles_, [&](const auto &candidate) {
      return candidate.id == selected.id;
    });
    if (!known) {
      eventBundles_.push_back(selected);
    }
  }
  std::set<std::pair<SyncCoverNodeId, SyncCoverNodeId>> standaloneEvents;
  std::size_t nextId = 0;
  for (const CanonicalEventBundleCandidate &bundle : eventBundles_) {
    if (bundle.id == std::numeric_limits<std::size_t>::max()) {
      return func_.emitError(
          "internal error: canonical covering event identity overflow");
    }
    nextId = std::max(nextId, bundle.id + 1);
    if (bundle.kind == CanonicalEventBundleKind::Standalone &&
        bundle.events.size() == 1 &&
        bundle.events.front().iterationDistance == 0) {
      standaloneEvents.emplace(bundle.events.front().source,
                               bundle.events.front().target);
    }
  }
  const SyncCoverGraph &graph = universe_.getGraph();
  for (SyncCoverDemandId demandId : activeDemands_) {
    if (demandId >= graph.getDemands().size()) {
      return func_.emitError(
          "internal error: canonical covering active demand is out of range");
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (demand.distance != 0 || demand.source >= graph.getNodes().size() ||
        demand.target >= graph.getNodes().size()) {
      continue;
    }
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    const bool canSignal =
        source.resource != target.resource &&
        std::binary_search(source.completionTargets.begin(),
                           source.completionTargets.end(), target.resource);
    if (!canSignal ||
        !standaloneEvents.emplace(demand.source, demand.target).second) {
      continue;
    }
    if (nextId == std::numeric_limits<std::size_t>::max()) {
      return func_.emitError(
          "internal error: canonical covering event identity overflow");
    }
    CanonicalEvent event;
    event.source = demand.source;
    event.target = demand.target;
    event.sourcePipe = static_cast<PipelineType>(source.resource);
    event.targetPipe = static_cast<PipelineType>(target.resource);
    event.setAnchor = {plan_.getNodes()[demand.source].operation, false};
    event.waitAnchor = {plan_.getNodes()[demand.target].operation, true};
    event.actions = {
        {CanonicalEventActionKind::Set, CanonicalEventActionPhase::Straight,
         event.setAnchor, {}},
        {CanonicalEventActionKind::Wait, CanonicalEventActionPhase::Straight,
         event.waitAnchor, {}}};
    event.completions = {{demand.source, demand.target, 0, nullptr, 0, 1}};
    event.intervalBegin = getAnchorPosition_(event.setAnchor);
    event.intervalEnd = getAnchorPosition_(event.waitAnchor);
    CanonicalEventBundleCandidate bundle;
    bundle.id = nextId++;
    bundle.kind = CanonicalEventBundleKind::Standalone;
    bundle.events.push_back(std::move(event));
    eventBundles_.push_back(std::move(bundle));
  }
  llvm::sort(eventBundles_, [](const auto &first, const auto &second) {
    return first.id < second.id;
  });
  const auto duplicate =
      std::adjacent_find(eventBundles_.begin(), eventBundles_.end(),
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
  for (const SyncCoverSlotProtocolCandidate &candidate :
       slotProtocols_.candidates) {
    if (!canBuildSlotProtocolRecipe(candidate, loopScopes_)) {
      continue;
    }
    keys.insert({static_cast<PipelineType>(candidate.sourceResource),
                 static_cast<PipelineType>(candidate.targetResource)});
  }
  for (const CanonicalEventDomainKey &key : keys) {
    std::vector<unsigned> reserved;
    auto hidden = reservedIds_.find(key);
    if (hidden != reservedIds_.end()) {
      reserved.assign(hidden->second.begin(), hidden->second.end());
    }
    const SyncCoverMechanismResult result = universe_.addResourceDomain(
        SyncCoverResourceKind::EventId, static_cast<std::uint32_t>(key.source),
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
  const std::optional<SyncCoverScopeId> scope =
      getEndpointScope(universe_.getGraph(), legacy.source, legacy.target);
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
  std::size_t nextBarrierId = 0;
  for (const CanonicalBarrierCandidate &candidate : candidateUniverse_.barriers) {
    if (candidate.id == std::numeric_limits<std::size_t>::max()) {
      return func_.emitError(
          "internal error: canonical covering barrier identity overflow");
    }
    nextBarrierId = std::max(nextBarrierId, candidate.id + 1);
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
    const SyncCoverAnchor anchor{SyncCoverAnchorKind::TimelinePoint, 0,
                                 *anchorScope,
                                 getAnchorPosition_(candidate.barrier.anchor)};
    descriptor.barrier = SyncCoverBarrierPlacement{
        static_cast<std::uint32_t>(candidate.barrier.pipe), anchor,
        *anchorScope};
    std::set<std::tuple<std::size_t, std::size_t, unsigned, SyncCoverScopeId>>
        seen;
    for (const SyncGraphEdge &legacy :
         getBarrierCompletionEdges_(candidate.barrier)) {
      const std::optional<SyncCoverEdge> edge = translateBarrierEdge(legacy);
      const bool supplies =
          edge && syncCoverBarrierCanSupply(universe_.getGraph(),
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
      if (!syncCoverBarrierCanSupply(universe_.getGraph(), *descriptor.barrier,
                                     edge)) {
        return func_.emitError(
            "internal error: canonical covering barrier cannot supply its "
            "declared requirement");
      }
      descriptor.supplyEdges.push_back(std::move(edge));
    }
    const SyncCoverMechanismResult result = universe_.addMechanism(descriptor);
    if (!result || !result.index) {
      return emitMechanismError("barrier insertion", result);
    }
    const bool providerUnique = providers_.emplace(provider, *result.index).second;
    const bool recipeUnique =
        barrierRecipes_.emplace(provider, candidate.barrier).second;
    if (!providerUnique || !recipeUnique) {
      return func_.emitError(
          "internal error: duplicate canonical covering barrier provider");
    }
  }

  const SyncCoverGraph &graph = universe_.getGraph();
  using BarrierGroupKey =
      std::tuple<SyncCoverNodeId, SyncCoverScopeId, unsigned>;
  std::set<BarrierGroupKey> groups;
  for (SyncCoverDemandId demandId : activeDemands_) {
    if (demandId >= graph.getDemands().size()) {
      return func_.emitError(
          "internal error: canonical covering fallback demand is out of range");
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    groups.emplace(demand.target, demand.scope, demand.distance);
  }

  struct FallbackBarrierCandidate {
    SyncCoverMechanismDescriptor descriptor;
    CanonicalBarrier recipe;
    std::vector<std::uint64_t> coverage;
    std::size_t loopDepth = 0;
  };
  std::vector<FallbackBarrierCandidate> fallbacks;
  const std::size_t coverageWords =
      activeDemands_.size() / 64 +
      (activeDemands_.size() % 64 == 0 ? 0 : 1);
  for (const BarrierGroupKey &key : groups) {
    const auto [targetId, recurrenceScope, distance] = key;
    if (targetId >= graph.getNodes().size() ||
        targetId >= plan_.getNodes().size()) {
      return func_.emitError(
          "internal error: canonical covering fallback target is invalid");
    }
    const SyncCoverNode &target = graph.getNodes()[targetId];
    const CanonicalSyncNode &canonicalTarget = plan_.getNodes()[targetId];
    const std::optional<std::size_t> loopDepth =
        graph.getScopeLoopDepth(target.scope);
    if (!loopDepth) {
      return func_.emitError(
          "internal error: canonical covering fallback depth is invalid");
    }

    SyncCoverMechanismDescriptor descriptor;
    descriptor.kind = SyncCoverMechanismKind::Barrier;
    descriptor.barrier = SyncCoverBarrierPlacement{
        static_cast<std::uint32_t>(PipelineType::PIPE_ALL),
        {SyncCoverAnchorKind::BeforeNode, targetId, 0, 0},
        target.scope, true};
    CanonicalBarrier recipe;
    recipe.pipe = PipelineType::PIPE_ALL;
    recipe.anchor = {canonicalTarget.operation, true};
    recipe.anchorNodes.push_back(targetId);
    recipe.recurrenceScope = recurrenceScope;
    if (distance != 0) {
      for (const auto &entry : loopScopes_) {
        if (entry.second == recurrenceScope) {
          if (recipe.recurrenceLoop && recipe.recurrenceLoop != entry.first) {
            return func_.emitError(
                "internal error: canonical covering recurrence scope is "
                "ambiguous");
          }
          recipe.recurrenceLoop = entry.first;
        }
      }
      if (!recipe.recurrenceLoop) {
        return func_.emitError(
            "internal error: canonical covering recurrence loop is missing");
      }
    }
    std::vector<std::uint64_t> coverage(coverageWords, 0);
    using SupplyKey =
        std::tuple<SyncCoverNodeId, SyncCoverNodeId, SyncCoverScopeId, unsigned,
                   std::vector<SyncCoverGuardLiteral>,
                   std::vector<SyncCoverGuardLiteral>>;
    std::set<SupplyKey> seen;
    for (auto [localDemand, demandId] : llvm::enumerate(activeDemands_)) {
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      SyncCoverEdge edge;
      edge.source = demand.source;
      edge.target = demand.target;
      edge.kind = SyncCoverEdgeKind::CompletionSupply;
      edge.scope = demand.scope;
      edge.distance = demand.distance;
      edge.sourceGuard = demand.sourceGuard;
      edge.targetGuard = demand.targetGuard;
      if (!syncCoverBarrierCanSupply(graph, *descriptor.barrier, edge)) {
        continue;
      }
      coverage[localDemand / 64] |=
          std::uint64_t{1} << static_cast<unsigned>(localDemand % 64);
      if (seen.emplace(edge.source, edge.target, edge.scope, edge.distance,
                       edge.sourceGuard.literals,
                       edge.targetGuard.literals)
              .second) {
        descriptor.supplyEdges.push_back(std::move(edge));
      }
    }
    if (descriptor.supplyEdges.empty()) {
      continue;
    }
    for (auto indexedRequirement :
         llvm::enumerate(plan_.getCompletionRequirements())) {
      const CanonicalDependency &requirement = indexedRequirement.value();
      if (requirement.target == targetId &&
          requirement.iterationDistance == distance &&
          requirement.recurrenceLoop == recipe.recurrenceLoop) {
        recipe.requirements.push_back(indexedRequirement.index());
      }
    }
    fallbacks.push_back({std::move(descriptor), std::move(recipe),
                         std::move(coverage), *loopDepth});
  }

  const auto coverageSubset = [](ArrayRef<std::uint64_t> first,
                                 ArrayRef<std::uint64_t> second) {
    return llvm::all_of(llvm::zip(first, second), [](const auto &words) {
      return (std::get<0>(words) & ~std::get<1>(words)) == 0;
    });
  };
  std::vector<bool> dominated(fallbacks.size(), false);
  for (std::size_t first = 0; first < fallbacks.size(); ++first) {
    for (std::size_t second = 0; second < fallbacks.size(); ++second) {
      if (first == second ||
          fallbacks[first].loopDepth != fallbacks[second].loopDepth ||
          !coverageSubset(fallbacks[first].coverage,
                          fallbacks[second].coverage)) {
        continue;
      }
      const bool equal = coverageSubset(fallbacks[second].coverage,
                                        fallbacks[first].coverage);
      if (!equal || second < first) {
        dominated[first] = true;
        break;
      }
    }
  }

  for (auto [index, fallback] : llvm::enumerate(fallbacks)) {
    if (dominated[index]) {
      continue;
    }
    if (nextBarrierId == std::numeric_limits<std::size_t>::max()) {
      return func_.emitError(
          "internal error: canonical covering barrier identity overflow");
    }
    const CanonicalSelectionMechanismRef provider{
        CanonicalSelectionMechanismKind::Barrier, nextBarrierId++};
    const std::optional<std::uint64_t> identity =
        encodeProviderIdentity(provider.kind, provider.id);
    if (!identity) {
      return func_.emitError(
          "internal error: canonical covering fallback identity is invalid");
    }
    fallback.descriptor.providerIdentity = *identity;
    fallback.recipe.id = provider.id;
    const SyncCoverMechanismResult insertion =
        universe_.addMechanism(fallback.descriptor);
    if (!insertion || !insertion.index) {
      return emitMechanismError("PIPE_ALL fallback insertion", insertion);
    }
    const bool providerUnique =
        providers_.emplace(provider, *insertion.index).second;
    const bool recipeUnique =
        barrierRecipes_.emplace(provider, std::move(fallback.recipe)).second;
    if (!providerUnique || !recipeUnique) {
      return func_.emitError(
          "internal error: duplicate canonical covering PIPE_ALL fallback");
    }
  }
  return success();
}

LogicalResult MechanismAdapter::addGeneratedColumns(
    CanonicalSyncCoveringSnapshot &snapshot) {
  SyncCoverTargetCapabilities target = target_;
  target.eventIdBudget = eventIdMax_;
  const SyncCoverColumnGenerationContext context{target, activeDemands_};
  const std::size_t oldMechanismCount = universe_.getMechanisms().size();
  // Production generation is restricted to mechanisms with complete emission
  // recipes. Recurrence rings and buffer-token protocols remain available to
  // analysis clients, but must not enter this universe until their physical
  // actions can be materialized and independently verified here.
  std::vector<std::unique_ptr<SyncCoverColumnGenerator>> generators;
  generators.push_back(makeSyncCoverCanonicalEventGenerator());
  generators.push_back(makeSyncCoverMergedPrefixEventGenerator());
  generators.push_back(makeSyncCoverPiercedBarrierGenerator());
  const SyncCoverColumnGenerationResult generated =
      runSyncCoverColumnGenerators(context, universe_, generators);
  if (!generated) {
    return func_.emitError(
        "internal error: canonical synchronization column generation failed");
  }
  for (const SyncCoverColumnGeneratorReport &report : generated.reports) {
    snapshot.generatedColumnCandidates += report.candidates;
    snapshot.generatedColumns += report.admitted;
    snapshot.columnGenerationTruncated |= report.truncated;
  }

  std::size_t nextBarrier = 0;
  for (const auto &[provider, recipe] : barrierRecipes_) {
    static_cast<void>(recipe);
    if (provider.kind == CanonicalSelectionMechanismKind::Barrier) {
      nextBarrier = std::max(nextBarrier, provider.id + 1);
    }
  }
  std::size_t nextBundle = 0;
  for (const CanonicalEventBundleCandidate &bundle : eventBundles_) {
    nextBundle = std::max(nextBundle, bundle.id + 1);
  }

  const SyncCoverGraph &graph = universe_.getGraph();
  const auto &mechanisms = universe_.getMechanisms();
  const auto &edges = graph.getEdges();
  const auto &domains = universe_.getResourceDomains();
  for (std::size_t mechanismId = oldMechanismCount;
       mechanismId < mechanisms.size(); ++mechanismId) {
    const SyncCoverMechanism &mechanism = mechanisms[mechanismId];
    if (mechanism.kind == SyncCoverMechanismKind::Barrier) {
      if (!mechanism.barrier || nextBarrier ==
                                    std::numeric_limits<std::size_t>::max()) {
        return func_.emitError(
            "internal error: generated barrier recipe is invalid");
      }
      const std::optional<CanonicalAnchor> anchor =
          getCanonicalAnchor(mechanism.barrier->anchor, plan_.getNodes());
      if (!anchor) {
        return func_.emitError(
            "internal error: generated barrier anchor is not materializable");
      }
      CanonicalBarrier recipe;
      recipe.id = nextBarrier;
      recipe.pipe =
          static_cast<PipelineType>(mechanism.barrier->resource);
      recipe.anchor = *anchor;
      recipe.anchorNodes.push_back(mechanism.barrier->anchor.node);
      const CanonicalSelectionMechanismRef provider{
          CanonicalSelectionMechanismKind::Barrier, nextBarrier++};
      providers_.emplace(provider, mechanism.id);
      barrierRecipes_.emplace(provider, std::move(recipe));
      continue;
    }

    if (mechanism.resourceUses.size() != 1 || mechanism.supplyEdges.empty() ||
        nextBundle == std::numeric_limits<std::size_t>::max()) {
      return func_.emitError(
          "internal error: generated event recipe is invalid");
    }
    const std::size_t useId = 0;
    const SyncCoverResourceUse &use = mechanism.resourceUses.front();
    if (use.domain >= domains.size() || mechanism.actions.size() != 2) {
      return func_.emitError(
          "internal error: generated event resources are invalid");
    }
    const SyncCoverResourceDomain &domain = domains[use.domain];
    const std::optional<CanonicalAnchor> setAnchor =
        getCanonicalAnchor(mechanism.actions[0].anchor, plan_.getNodes());
    const std::optional<CanonicalAnchor> waitAnchor =
        getCanonicalAnchor(mechanism.actions[1].anchor, plan_.getNodes());
    if (!setAnchor || !waitAnchor) {
      return func_.emitError(
          "internal error: generated event anchors are not materializable");
    }
    CanonicalEvent event;
    event.sourcePipe = static_cast<PipelineType>(domain.sourceResource);
    event.targetPipe = static_cast<PipelineType>(domain.targetResource);
    event.setAnchor = *setAnchor;
    event.waitAnchor = *waitAnchor;
    event.width = use.width;
    event.intervalBegin = getAnchorPosition_(*setAnchor);
    event.intervalEnd = getAnchorPosition_(*waitAnchor);
    event.actions = {
        {CanonicalEventActionKind::Set, CanonicalEventActionPhase::Straight,
         *setAnchor, {}},
        {CanonicalEventActionKind::Wait, CanonicalEventActionPhase::Straight,
         *waitAnchor, {}}};
    for (std::size_t edgeId : mechanism.supplyEdges) {
      if (edgeId >= edges.size()) {
        return func_.emitError(
            "internal error: generated event supply edge is invalid");
      }
      const SyncCoverEdge &edge = edges[edgeId];
      event.completions.push_back(
          {edge.source, edge.target, edge.distance, nullptr, 0, 1});
    }
    event.source = event.completions.front().source;
    event.target = event.completions.front().target;
    event.traces.push_back(
        {CanonicalEventTraceKind::Straight, {0, 1}});
    if (!verifyEventProtocols_({event})) {
      return func_.emitError(
          "internal error: generated event recipe failed verification");
    }
    CanonicalEventBundleCandidate bundle;
    bundle.id = nextBundle;
    bundle.kind = CanonicalEventBundleKind::Standalone;
    bundle.events.push_back(std::move(event));
    const CanonicalSelectionMechanismRef provider{
        CanonicalSelectionMechanismKind::EventBundle, nextBundle++};
    providers_.emplace(provider, mechanism.id);
    eventResourceUses_.emplace(std::make_pair(mechanism.id, useId), 0);
    eventBundles_.push_back(std::move(bundle));
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
      const std::optional<SyncCoverScopeId> scope =
          getEndpointScope(universe_.getGraph(), event.source, event.target);
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
      TranslatedEventBundleMechanism expected = *translated;
      expected.descriptor.verifiedCoverageEdges =
          deriveOwnershipCoverageEdges(
              bundle, plan_.getOwnershipCycles(), plan_.getNodes(),
              universe_.getGraph(), activeDemands_);
      eventResourceUses = expected.eventResourceUses;
      result = universe_.addVerifiedProtocol(
          expected.descriptor, [&, expected](const auto &actual) {
            const std::vector<SyncCoverEdge> verifiedCoverage =
                deriveOwnershipCoverageEdges(
                    bundle, plan_.getOwnershipCycles(), plan_.getNodes(),
                    universe_.getGraph(), activeDemands_);
            return verifyTranslatedEventBundleCorrespondence(
                       bundle, expected, actual, universe_, domains_,
                       regionScopes_, loopScopes_, getAnchorPosition_) &&
                   llvm::equal(actual.verifiedCoverageEdges,
                               verifiedCoverage, sameCoverageEdge) &&
                   verifyBundleShape(bundle, plan_.getOwnershipCycles(),
                                     plan_.getNodes()) &&
                   verifyEventProtocols_(bundle.events);
          });
    }
    if (!result || !result.index) {
      InFlightDiagnostic diagnostic =
          func_.emitError() << "canonical covering event-bundle[" << bundle.id
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
                   << static_cast<bool>(event.recurrenceLoop) << " drain-loop="
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
    for (auto [eventIndex, resourceUse] : llvm::enumerate(eventResourceUses)) {
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

LogicalResult MechanismAdapter::addSlotProtocols() {
  if (!slotLifecycles_ || !slotProtocols_) {
    return func_.emitError(
        "internal error: canonical covering slot protocol universe is "
        "incomplete");
  }
  for (auto indexedCandidate : llvm::enumerate(slotProtocols_.candidates)) {
    const SyncCoverSlotProtocolCandidate &candidate = indexedCandidate.value();
    if (!canBuildSlotProtocolRecipe(candidate, loopScopes_)) {
      ++unmaterializableSlotProtocols_;
      continue;
    }
    const bool invalidCandidate =
        candidate.id != indexedCandidate.index() ||
        candidate.lifecycle >= slotLifecycles_.lifecycles.size();
    if (invalidCandidate) {
      return func_.emitError(
          "internal error: canonical covering slot protocol identity is "
          "invalid");
    }
    const CanonicalSelectionMechanismRef provider{
        CanonicalSelectionMechanismKind::SlotProtocol, candidate.id};
    const std::optional<std::uint64_t> identity =
        encodeProviderIdentity(provider.kind, provider.id);
    const CanonicalEventDomainKey key{
        static_cast<PipelineType>(candidate.sourceResource),
        static_cast<PipelineType>(candidate.targetResource)};
    auto domain = domains_.find(key);
    if (!identity || domain == domains_.end()) {
      return func_.emitError(
          "internal error: canonical covering slot protocol domain is "
          "invalid");
    }
    const auto descriptor = makeSyncCoverSlotProtocolDescriptor(
        universe_.getResourceDomains()[domain->second], candidate, *identity);
    if (!descriptor) {
      return func_.emitError(
          "internal error: canonical covering slot protocol descriptor is "
          "invalid");
    }
    const SyncCoverSlotLifecycle &lifecycle =
        slotLifecycles_.lifecycles[candidate.lifecycle];
    const SyncCoverMechanismResult result =
        universe_.addVerifiedProtocol(*descriptor, [&](const auto &actual) {
          return verifySyncCoverSlotProtocol(candidateIndex_, lifecycle,
                                             universe_, candidate, actual);
        });
    if (!result || !result.index) {
      return emitMechanismError("slot-protocol insertion", result);
    }
    const SyncCoverMechanism &mechanism =
        universe_.getMechanisms()[*result.index];
    const std::optional<SlotProtocolRecipe> recipe =
        buildSlotProtocolRecipe(plan_.getNodes(), universe_, candidate,
                                mechanism, loopScopes_, getAnchorPosition_);
    if (!recipe ||
        !verifySlotProtocolRecipeCorrespondence(
            plan_.getNodes(), universe_, candidate, mechanism, loopScopes_,
            getAnchorPosition_, *recipe) ||
        !verifyEventProtocols_({recipe->event})) {
      return func_.emitError(
          "internal error: canonical covering slot protocol has no verified "
          "emission recipe");
    }
    const bool providerUnique =
        providers_.emplace(provider, *result.index).second;
    const bool recipeUnique =
        slotProtocolRecipes_.emplace(provider, *recipe).second;
    if (!providerUnique || !recipeUnique) {
      return func_.emitError(
          "internal error: duplicate canonical covering slot protocol");
    }
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
