// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncCoveringSlotRecipe.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_covering;

namespace {

Operation *
findUniqueLoop(SyncCoverScopeId scope,
               const DenseMap<Operation *, SyncCoverScopeId> &loopScopes) {
  Operation *loop = nullptr;
  for (const auto &entry : loopScopes) {
    if (entry.second != scope) {
      continue;
    }
    if (loop) {
      return nullptr;
    }
    loop = entry.first;
  }
  return loop && isa<scf::ForOp>(loop) ? loop : nullptr;
}

std::optional<CanonicalAnchor>
translateAnchor(ArrayRef<CanonicalSyncNode> nodes,
                const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
                const SyncCoverAnchor &anchor) {
  if (anchor.kind == SyncCoverAnchorKind::BeforeNode ||
      anchor.kind == SyncCoverAnchorKind::AfterNode) {
    const bool invalidNode =
        anchor.node >= nodes.size() || !nodes[anchor.node].operation;
    if (invalidNode) {
      return std::nullopt;
    }
    return CanonicalAnchor{nodes[anchor.node].operation,
                           anchor.kind == SyncCoverAnchorKind::BeforeNode};
  }
  if (anchor.kind != SyncCoverAnchorKind::ScopeEntry &&
      anchor.kind != SyncCoverAnchorKind::ScopeExit) {
    return std::nullopt;
  }
  Operation *loop = findUniqueLoop(anchor.scope, loopScopes);
  if (!loop) {
    return std::nullopt;
  }
  return CanonicalAnchor{loop, anchor.kind == SyncCoverAnchorKind::ScopeEntry};
}

bool sameAnchor(const CanonicalAnchor &first, const CanonicalAnchor &second) {
  return first.operation == second.operation && first.before == second.before;
}

bool sameResourceAction(const SyncCoverResourceAction &actual,
                        SyncCoverResourceActionKind kind,
                        std::uint32_t resource,
                        const SyncCoverAnchor &anchor) {
  return actual.kind == kind && actual.resource == resource &&
         actual.anchor.kind == anchor.kind &&
         actual.anchor.node == anchor.node &&
         actual.anchor.scope == anchor.scope &&
         actual.anchor.position == anchor.position;
}

bool sameLane(const CanonicalEventLane &first,
              const CanonicalEventLane &second) {
  return first.kind == second.kind && first.index == second.index &&
         first.selector == second.selector;
}

bool sameAction(const CanonicalEventAction &first,
                const CanonicalEventAction &second) {
  return first.kind == second.kind && first.phase == second.phase &&
         sameAnchor(first.anchor, second.anchor) &&
         sameLane(first.lane, second.lane) &&
         first.guard.kind == second.guard.kind &&
         first.guard.loop == second.guard.loop;
}

bool sameCompletion(const CanonicalEventCompletion &first,
                    const CanonicalEventCompletion &second) {
  return first.source == second.source && first.target == second.target &&
         first.iterationDistance == second.iterationDistance &&
         first.recurrenceLoop == second.recurrenceLoop &&
         first.setAction == second.setAction &&
         first.waitAction == second.waitAction;
}

bool sameTrace(const CanonicalEventTrace &first,
               const CanonicalEventTrace &second) {
  return first.kind == second.kind && first.actions == second.actions &&
         first.controlRegion == second.controlRegion &&
         first.guard.kind == second.guard.kind &&
         first.guard.loop == second.guard.loop &&
         first.hasExplicitTokenState == second.hasExplicitTokenState &&
         first.initialTokens == second.initialTokens &&
         first.expectedTokens == second.expectedTokens;
}

CanonicalEventTrace makeTrace(CanonicalEventTraceKind kind,
                              std::initializer_list<unsigned> actions) {
  CanonicalEventTrace trace;
  trace.kind = kind;
  trace.actions.append(actions.begin(), actions.end());
  return trace;
}

} // namespace

bool mlir::pto::canonical_sync_covering::canBuildSlotProtocolRecipe(
    const SyncCoverSlotProtocolCandidate &candidate,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes) {
  const bool unsupported =
      candidate.distance != 1 || candidate.targets.empty() ||
      !findUniqueLoop(candidate.recurrenceScope, loopScopes);
  if (unsupported) {
    return false;
  }
  return candidate.kind == SyncCoverSlotProtocolKind::UnitRelease ||
         (candidate.kind == SyncCoverSlotProtocolKind::HierarchicalRelease &&
          candidate.targetLoop &&
          findUniqueLoop(*candidate.targetLoop, loopScopes));
}

std::optional<SlotProtocolRecipe>
mlir::pto::canonical_sync_covering::buildSlotProtocolRecipe(
    ArrayRef<CanonicalSyncNode> nodes,
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverSlotProtocolCandidate &candidate,
    const SyncCoverMechanism &mechanism,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    llvm::function_ref<std::size_t(const CanonicalAnchor &)>
        getAnchorPosition) {
  const bool invalidShape =
      mechanism.kind != SyncCoverMechanismKind::VerifiedProtocol ||
      !mechanism.protocolVerified ||
      !canBuildSlotProtocolRecipe(candidate, loopScopes) ||
      mechanism.actions.size() != 4 ||
      mechanism.resourceUses.size() != 1 ||
      mechanism.supplyEdges.size() != candidate.targets.size() ||
      mechanism.supplyBindings.size() != candidate.targets.size();
  if (invalidShape) {
    return std::nullopt;
  }
  const SyncCoverResourceUse &use = mechanism.resourceUses.front();
  const bool invalidUse =
      use.domain >= universe.getResourceDomains().size() ||
      use.scope != candidate.recurrenceScope || use.distance != 1 ||
      use.width != 1 || use.actions != std::vector<std::size_t>({0, 1, 2, 3}) ||
      use.supplyEdges != mechanism.supplyEdges;
  if (invalidUse) {
    return std::nullopt;
  }
  const SyncCoverResourceDomain &domain =
      universe.getResourceDomains()[use.domain];
  if (domain.kind != SyncCoverResourceKind::EventId ||
      domain.sourceResource != candidate.sourceResource ||
      domain.targetResource != candidate.targetResource) {
    return std::nullopt;
  }

  Operation *outerLoop = findUniqueLoop(candidate.recurrenceScope, loopScopes);
  std::optional<CanonicalAnchor> prime =
      translateAnchor(nodes, loopScopes, mechanism.actions[0].anchor);
  std::optional<CanonicalAnchor> wait =
      translateAnchor(nodes, loopScopes, mechanism.actions[1].anchor);
  std::optional<CanonicalAnchor> set =
      translateAnchor(nodes, loopScopes, mechanism.actions[2].anchor);
  std::optional<CanonicalAnchor> drain =
      translateAnchor(nodes, loopScopes, mechanism.actions[3].anchor);
  if (!outerLoop || !prime || !wait || !set || !drain) {
    return std::nullopt;
  }

  CanonicalEvent event;
  event.source = candidate.source;
  event.target = candidate.targets.front();
  event.sourcePipe = static_cast<PipelineType>(domain.sourceResource);
  event.targetPipe = static_cast<PipelineType>(domain.targetResource);
  event.setAnchor = *set;
  event.waitAnchor = *wait;
  event.recurrenceLoop = outerLoop;
  event.scopeLoop = outerLoop;
  event.iterationDistance = 1;
  event.width = 1;

  CanonicalEventLane all;
  all.kind = CanonicalEventLaneKind::All;
  event.actions = {{CanonicalEventActionKind::Set,
                    CanonicalEventActionPhase::Prime, *prime, all},
                   {CanonicalEventActionKind::Wait,
                    CanonicalEventActionPhase::Body,
                    *wait,
                    {}},
                   {CanonicalEventActionKind::Set,
                    CanonicalEventActionPhase::Body,
                    *set,
                    {}},
                   {CanonicalEventActionKind::Wait,
                    CanonicalEventActionPhase::Drain, *drain, all}};
  event.completions.reserve(candidate.targets.size());
  for (std::size_t index = 0; index < candidate.targets.size(); ++index) {
    const std::size_t graphEdge = mechanism.supplyEdges[index];
    if (graphEdge >= universe.getGraph().getEdges().size()) {
      return std::nullopt;
    }
    const SyncCoverEdge &edge = universe.getGraph().getEdges()[graphEdge];
    const SyncCoverSupplyBinding &binding = mechanism.supplyBindings[index];
    if (binding.supplyEdge != graphEdge || binding.resourceUse != 0 ||
        binding.produceAction != 2 || binding.consumeAction != 1 ||
        edge.source != candidate.source ||
        edge.target != candidate.targets[index] || edge.distance != 1 ||
        edge.scope != candidate.recurrenceScope ||
        edge.kind != SyncCoverEdgeKind::CompletionSupply ||
        edge.mechanism != mechanism.id) {
      return std::nullopt;
    }
    event.completions.push_back({edge.source, edge.target, 1, outerLoop, 2, 1});
  }
  event.traces.push_back(makeTrace(CanonicalEventTraceKind::Prime, {0}));
  event.traces.push_back(makeTrace(CanonicalEventTraceKind::Cycle, {1, 2}));
  event.traces.push_back(makeTrace(CanonicalEventTraceKind::Final, {3}));

  event.intervalBegin = std::numeric_limits<std::size_t>::max();
  event.intervalEnd = 0;
  for (const CanonicalEventAction &action : event.actions) {
    const std::size_t position = getAnchorPosition(action.anchor);
    event.intervalBegin = std::min(event.intervalBegin, position);
    event.intervalEnd = std::max(event.intervalEnd, position);
  }
  return SlotProtocolRecipe{0, std::move(event)};
}

bool mlir::pto::canonical_sync_covering::verifySlotProtocolRecipeCorrespondence(
    ArrayRef<CanonicalSyncNode> nodes,
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverSlotProtocolCandidate &candidate,
    const SyncCoverMechanism &mechanism,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    llvm::function_ref<std::size_t(const CanonicalAnchor &)> getAnchorPosition,
    const SlotProtocolRecipe &recipe) {
  const bool invalidShape =
      !canBuildSlotProtocolRecipe(candidate, loopScopes) ||
      mechanism.kind != SyncCoverMechanismKind::VerifiedProtocol ||
      !mechanism.protocolVerified || mechanism.actions.size() != 4 ||
      mechanism.resourceUses.size() != 1 || recipe.resourceUse != 0 ||
      mechanism.supplyEdges.size() != candidate.targets.size() ||
      mechanism.supplyBindings.size() != candidate.targets.size();
  if (invalidShape) {
    return false;
  }
  const SyncCoverResourceUse &use = mechanism.resourceUses.front();
  if (use.domain >= universe.getResourceDomains().size()) {
    return false;
  }
  const SyncCoverResourceDomain &domain =
      universe.getResourceDomains()[use.domain];
  const SyncCoverAnchor expectedWait =
      candidate.kind == SyncCoverSlotProtocolKind::UnitRelease
          ? SyncCoverAnchor{SyncCoverAnchorKind::BeforeNode,
                            candidate.targets.front(), 0}
          : SyncCoverAnchor{SyncCoverAnchorKind::ScopeEntry, 0,
                            *candidate.targetLoop};
  const bool invalidActions =
      !sameResourceAction(
          mechanism.actions[0], SyncCoverResourceActionKind::Produce,
          candidate.sourceResource,
          {SyncCoverAnchorKind::ScopeEntry, 0, candidate.recurrenceScope}) ||
      !sameResourceAction(mechanism.actions[1],
                          SyncCoverResourceActionKind::Consume,
                          candidate.targetResource, expectedWait) ||
      !sameResourceAction(
          mechanism.actions[2], SyncCoverResourceActionKind::Produce,
          candidate.sourceResource,
          {SyncCoverAnchorKind::AfterNode, candidate.source, 0}) ||
      !sameResourceAction(
          mechanism.actions[3], SyncCoverResourceActionKind::Consume,
          candidate.targetResource,
          {SyncCoverAnchorKind::ScopeExit, 0, candidate.recurrenceScope});
  Operation *outerLoop = findUniqueLoop(candidate.recurrenceScope, loopScopes);
  const std::optional<CanonicalAnchor> prime =
      translateAnchor(nodes, loopScopes, mechanism.actions[0].anchor);
  const std::optional<CanonicalAnchor> wait =
      translateAnchor(nodes, loopScopes, mechanism.actions[1].anchor);
  const std::optional<CanonicalAnchor> set =
      translateAnchor(nodes, loopScopes, mechanism.actions[2].anchor);
  const std::optional<CanonicalAnchor> drain =
      translateAnchor(nodes, loopScopes, mechanism.actions[3].anchor);
  const CanonicalEvent &event = recipe.event;
  const bool invalidMetadata =
      use.scope != candidate.recurrenceScope || use.distance != 1 ||
      use.width != 1 ||
      use.actions != std::vector<std::size_t>({0, 1, 2, 3}) ||
      use.supplyEdges != mechanism.supplyEdges ||
      domain.kind != SyncCoverResourceKind::EventId ||
      domain.sourceResource != candidate.sourceResource ||
      domain.targetResource != candidate.targetResource || invalidActions ||
      !outerLoop || !prime || !wait || !set || !drain ||
      event.source != candidate.source ||
      event.target != candidate.targets.front() ||
      event.sourcePipe != static_cast<PipelineType>(domain.sourceResource) ||
      event.targetPipe != static_cast<PipelineType>(domain.targetResource) ||
      !sameAnchor(event.setAnchor, *set) ||
      !sameAnchor(event.waitAnchor, *wait) ||
      event.recurrenceLoop != outerLoop || event.forwardDrainLoop ||
      event.scopeLoop != outerLoop || event.resourceScopeLoop ||
      event.iterationDistance != 1 || event.width != 1 ||
      !event.eventIds.empty() || event.protocolBundle != 0 ||
      event.ownershipCycle != 0 || event.ownershipProtocol;
  if (invalidMetadata) {
    return false;
  }

  CanonicalEventLane all;
  all.kind = CanonicalEventLaneKind::All;
  const std::vector<CanonicalEventAction> expectedActions = {
      {CanonicalEventActionKind::Set, CanonicalEventActionPhase::Prime,
       *prime, all},
      {CanonicalEventActionKind::Wait, CanonicalEventActionPhase::Body,
       *wait, {}},
      {CanonicalEventActionKind::Set, CanonicalEventActionPhase::Body,
       *set, {}},
      {CanonicalEventActionKind::Wait, CanonicalEventActionPhase::Drain,
       *drain, all}};
  const std::vector<CanonicalEventTrace> expectedTraces = {
      makeTrace(CanonicalEventTraceKind::Prime, {0}),
      makeTrace(CanonicalEventTraceKind::Cycle, {1, 2}),
      makeTrace(CanonicalEventTraceKind::Final, {3})};
  const bool invalidProtocol =
      !llvm::equal(event.actions, expectedActions, sameAction) ||
      !llvm::equal(event.traces, expectedTraces, sameTrace) ||
      event.completions.size() != candidate.targets.size();
  if (invalidProtocol) {
    return false;
  }
  for (std::size_t index = 0; index < candidate.targets.size(); ++index) {
    const std::size_t graphEdge = mechanism.supplyEdges[index];
    if (graphEdge >= universe.getGraph().getEdges().size()) {
      return false;
    }
    const SyncCoverEdge &edge = universe.getGraph().getEdges()[graphEdge];
    const SyncCoverSupplyBinding &binding = mechanism.supplyBindings[index];
    const CanonicalEventCompletion expected{
        candidate.source, candidate.targets[index], 1, outerLoop, 2, 1};
    const bool invalidSupply =
        edge.source != expected.source || edge.target != expected.target ||
        edge.scope != candidate.recurrenceScope || edge.distance != 1 ||
        edge.kind != SyncCoverEdgeKind::CompletionSupply ||
        edge.mechanism != mechanism.id ||
        binding.supplyEdge != graphEdge || binding.resourceUse != 0 ||
        binding.produceAction != 2 || binding.consumeAction != 1 ||
        !sameCompletion(event.completions[index], expected);
    if (invalidSupply) {
      return false;
    }
  }
  std::size_t intervalBegin = std::numeric_limits<std::size_t>::max();
  std::size_t intervalEnd = 0;
  for (const CanonicalEventAction &action : event.actions) {
    const std::size_t position = getAnchorPosition(action.anchor);
    intervalBegin = std::min(intervalBegin, position);
    intervalEnd = std::max(intervalEnd, position);
  }
  return event.intervalBegin == intervalBegin &&
         event.intervalEnd == intervalEnd;
}

std::optional<CanonicalEventBundleCandidate>
mlir::pto::canonical_sync_covering::materializeSlotProtocolBundle(
    const CanonicalSyncCoveringSelectedSlotProtocol &recipe,
    const CanonicalSyncCoveringSelectedResourceUse &use,
    const CanonicalSyncCoveringResourceAllocation &allocation,
    std::size_t bundleId) {
  const bool invalidIdentity =
      recipe.provider.kind != CanonicalSelectionMechanismKind::SlotProtocol ||
      recipe.mechanism != use.mechanism ||
      recipe.mechanism != allocation.mechanism ||
      !(recipe.provider == use.provider) ||
      !(recipe.provider == allocation.provider) ||
      recipe.resourceUse != use.resourceUse ||
      recipe.resourceUse != allocation.resourceUse;
  const bool invalidResource =
      use.kind != SyncCoverResourceKind::EventId || use.width != 1 ||
      use.materializationEventIndex ||
      allocation.kind != SyncCoverResourceKind::EventId ||
      allocation.domain != use.domain ||
      allocation.sourceResource != use.sourceResource ||
      allocation.targetResource != use.targetResource ||
      allocation.ids.size() != 1;
  CanonicalEvent event = recipe.event;
  const bool invalidEvent =
      !event.eventIds.empty() || event.width != 1 ||
      static_cast<std::uint32_t>(event.sourcePipe) != use.sourceResource ||
      static_cast<std::uint32_t>(event.targetPipe) != use.targetResource ||
      bundleId == std::numeric_limits<std::size_t>::max();
  if (invalidIdentity || invalidResource || invalidEvent) {
    return std::nullopt;
  }
  event.eventIds.assign(allocation.ids.begin(), allocation.ids.end());
  CanonicalEventBundleCandidate bundle;
  bundle.id = bundleId;
  bundle.kind = CanonicalEventBundleKind::Standalone;
  bundle.applicability = CanonicalEventBundleApplicability::CoveringOnly;
  bundle.events.push_back(std::move(event));
  return bundle;
}
