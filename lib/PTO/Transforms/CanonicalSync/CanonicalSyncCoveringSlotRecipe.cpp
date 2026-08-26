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
#include <set>

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
      candidate.distance != 1 || candidate.sources.empty() ||
      candidate.targets.empty() || candidate.completionEdges.empty() ||
      !findUniqueLoop(candidate.recurrenceScope, loopScopes);
  if (unsupported) {
    return false;
  }
  if (candidate.kind == SyncCoverSlotProtocolKind::UnitRelease) {
    return true;
  }
  return candidate.kind == SyncCoverSlotProtocolKind::HierarchicalRelease &&
         candidate.targetWaits.size() == candidate.targets.size() &&
         (!candidate.waitScope ||
          findUniqueLoop(*candidate.waitScope, loopScopes));
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
  const std::size_t setCount = candidate.sources.size();
  const std::set<SyncCoverNodeId> uniqueWaitTargets(
      candidate.targetWaits.begin(), candidate.targetWaits.end());
  const std::size_t waitCount =
      candidate.kind == SyncCoverSlotProtocolKind::UnitRelease
          ? 1
          : uniqueWaitTargets.size() * candidate.width;
  const std::size_t firstSetAction = 1 + waitCount;
  const std::size_t drainAction = firstSetAction + setCount;
  const bool invalidShape =
      mechanism.kind != SyncCoverMechanismKind::VerifiedProtocol ||
      !mechanism.protocolVerified ||
      !canBuildSlotProtocolRecipe(candidate, loopScopes) ||
      setCount == 0 || mechanism.actions.size() != drainAction + 1 ||
      mechanism.resourceUses.size() != 1 ||
      mechanism.supplyEdges.size() != candidate.completionEdges.size() ||
      mechanism.supplyBindings.size() != candidate.completionEdges.size();
  if (invalidShape) {
    return std::nullopt;
  }
  const SyncCoverResourceUse &use = mechanism.resourceUses.front();
  const bool invalidUse =
      use.domain >= universe.getResourceDomains().size() ||
      use.scope != candidate.recurrenceScope || use.distance != 1 ||
      use.width != candidate.width ||
      use.actions.size() != mechanism.actions.size() ||
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
  std::vector<CanonicalAnchor> waits;
  waits.reserve(waitCount);
  for (std::size_t index = 0; index < waitCount; ++index) {
    std::optional<CanonicalAnchor> wait =
        translateAnchor(nodes, loopScopes, mechanism.actions[index + 1].anchor);
    if (!wait) {
      return std::nullopt;
    }
    waits.push_back(*wait);
  }
  std::vector<CanonicalAnchor> sets;
  sets.reserve(setCount);
  for (std::size_t index = 0; index < setCount; ++index) {
    std::optional<CanonicalAnchor> set =
        translateAnchor(nodes, loopScopes,
                        mechanism.actions[firstSetAction + index].anchor);
    if (!set) {
      return std::nullopt;
    }
    sets.push_back(*set);
  }
  std::optional<CanonicalAnchor> drain =
      translateAnchor(nodes, loopScopes, mechanism.actions[drainAction].anchor);
  if (!outerLoop || !prime || waits.empty() || sets.empty() || !drain) {
    return std::nullopt;
  }

  CanonicalEvent event;
  event.source = candidate.source;
  event.target = candidate.targets.front();
  event.sourcePipe = static_cast<PipelineType>(domain.sourceResource);
  event.targetPipe = static_cast<PipelineType>(domain.targetResource);
  event.setAnchor = sets.front();
  event.waitAnchor = waits.front();
  event.recurrenceLoop = outerLoop;
  event.scopeLoop = outerLoop;
  event.iterationDistance = 1;
  event.width = candidate.width;

  CanonicalEventLane all;
  all.kind = CanonicalEventLaneKind::All;
  event.actions = {{CanonicalEventActionKind::Set,
                    CanonicalEventActionPhase::Prime, *prime, all}};
  for (std::size_t index = 0; index < waits.size(); ++index) {
    CanonicalEventLane lane;
    lane.index = candidate.kind == SyncCoverSlotProtocolKind::UnitRelease
                     ? 0
                     : static_cast<unsigned>(index % candidate.width);
    event.actions.push_back({CanonicalEventActionKind::Wait,
                             CanonicalEventActionPhase::Body, waits[index],
                             lane});
  }
  for (std::size_t index = 0; index < sets.size(); ++index) {
    CanonicalEventLane lane;
    lane.index = candidate.sourceLanes[index];
    event.actions.push_back({CanonicalEventActionKind::Set,
                             CanonicalEventActionPhase::Body, sets[index],
                             lane});
  }
  event.actions.push_back({CanonicalEventActionKind::Wait,
                           CanonicalEventActionPhase::Drain, *drain, all});
  event.completions.reserve(candidate.completionEdges.size());
  for (std::size_t index = 0; index < candidate.completionEdges.size();
       ++index) {
    const std::size_t graphEdge = mechanism.supplyEdges[index];
    if (graphEdge >= universe.getGraph().getEdges().size()) {
      return std::nullopt;
    }
    const SyncCoverEdge &edge = universe.getGraph().getEdges()[graphEdge];
    const SyncCoverSupplyBinding &binding = mechanism.supplyBindings[index];
    const auto &[source, target] = candidate.completionEdges[index];
    const auto sourcePosition = std::lower_bound(candidate.sources.begin(),
                                                 candidate.sources.end(), source);
    const bool sourceMissing = sourcePosition == candidate.sources.end() ||
                               *sourcePosition != source;
    if (sourceMissing) {
      return std::nullopt;
    }
    const std::size_t sourceIndex = static_cast<std::size_t>(
        sourcePosition - candidate.sources.begin());
    const std::size_t setAction = firstSetAction + sourceIndex;
    std::size_t waitAction = 1;
    if (candidate.kind == SyncCoverSlotProtocolKind::HierarchicalRelease) {
      const auto targetPosition = std::lower_bound(
          candidate.targets.begin(), candidate.targets.end(), target);
      if (targetPosition == candidate.targets.end() ||
          *targetPosition != target) {
        return std::nullopt;
      }
      const std::size_t targetIndex = static_cast<std::size_t>(
          targetPosition - candidate.targets.begin());
      const auto waitPosition = uniqueWaitTargets.find(
          candidate.targetWaits[targetIndex]);
      if (waitPosition == uniqueWaitTargets.end()) {
        return std::nullopt;
      }
      const std::size_t waitIndex = static_cast<std::size_t>(
          std::distance(uniqueWaitTargets.begin(), waitPosition));
      waitAction = 1 + waitIndex * candidate.width +
                   candidate.sourceLanes[sourceIndex];
    }
    if (binding.supplyEdge != graphEdge || binding.resourceUse != 0 ||
        binding.produceAction != setAction ||
        binding.consumeAction != waitAction || edge.source != source ||
        edge.target != target || edge.distance != 1 ||
        edge.scope != candidate.recurrenceScope ||
        edge.kind != SyncCoverEdgeKind::CompletionSupply ||
        edge.mechanism != mechanism.id) {
      return std::nullopt;
    }
    event.completions.push_back(
        {edge.source, edge.target, 1, outerLoop,
         static_cast<unsigned>(setAction), 1});
    event.completions.back().waitAction =
        static_cast<unsigned>(waitAction);
  }
  event.traces.push_back(makeTrace(CanonicalEventTraceKind::Prime, {0}));
  if (candidate.kind == SyncCoverSlotProtocolKind::UnitRelease) {
    event.traces.push_back(makeTrace(CanonicalEventTraceKind::Cycle,
                                     {1, static_cast<unsigned>(firstSetAction)}));
  } else {
    for (const SyncCoverSlotProtocolPath &path : candidate.paths) {
      auto waitPosition = uniqueWaitTargets.find(path.waitTarget);
      if (waitPosition == uniqueWaitTargets.end()) {
        return std::nullopt;
      }
      const std::size_t waitIndex = static_cast<std::size_t>(
          std::distance(uniqueWaitTargets.begin(), waitPosition));
      CanonicalEventTrace trace;
      trace.kind = CanonicalEventTraceKind::Cycle;
      for (unsigned lane = 0; lane < candidate.width; ++lane) {
        trace.actions.push_back(static_cast<unsigned>(
            1 + waitIndex * candidate.width + lane));
      }
      for (SyncCoverNodeId source : path.sources) {
        auto sourcePosition = std::lower_bound(candidate.sources.begin(),
                                               candidate.sources.end(), source);
        if (sourcePosition == candidate.sources.end() ||
            *sourcePosition != source) {
          return std::nullopt;
        }
        trace.actions.push_back(static_cast<unsigned>(
            firstSetAction +
            std::distance(candidate.sources.begin(), sourcePosition)));
      }
      event.traces.push_back(std::move(trace));
    }
  }
  event.traces.push_back(makeTrace(CanonicalEventTraceKind::Final,
                                   {static_cast<unsigned>(drainAction)}));

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
  const std::size_t setCount = candidate.sources.size();
  const std::set<SyncCoverNodeId> uniqueWaitTargets(
      candidate.targetWaits.begin(), candidate.targetWaits.end());
  const std::size_t waitCount =
      candidate.kind == SyncCoverSlotProtocolKind::UnitRelease
          ? 1
          : uniqueWaitTargets.size() * candidate.width;
  const std::size_t firstSetAction = 1 + waitCount;
  const std::size_t drainAction = firstSetAction + setCount;
  const bool invalidShape =
      !canBuildSlotProtocolRecipe(candidate, loopScopes) ||
      mechanism.kind != SyncCoverMechanismKind::VerifiedProtocol ||
      !mechanism.protocolVerified || setCount == 0 || waitCount == 0 ||
      candidate.sourceLanes.size() != setCount || candidate.width == 0 ||
      mechanism.actions.size() != drainAction + 1 ||
      mechanism.resourceUses.size() != 1 || recipe.resourceUse != 0 ||
      mechanism.supplyEdges.size() != candidate.completionEdges.size() ||
      mechanism.supplyBindings.size() != candidate.completionEdges.size();
  if (invalidShape) {
    return false;
  }
  const SyncCoverResourceUse &use = mechanism.resourceUses.front();
  if (use.domain >= universe.getResourceDomains().size()) {
    return false;
  }
  const SyncCoverResourceDomain &domain =
      universe.getResourceDomains()[use.domain];
  bool invalidActions =
      !sameResourceAction(
          mechanism.actions[0], SyncCoverResourceActionKind::Produce,
          candidate.sourceResource,
          {SyncCoverAnchorKind::ScopeEntry, 0, candidate.recurrenceScope}) ||
      !sameResourceAction(
          mechanism.actions[drainAction], SyncCoverResourceActionKind::Consume,
          candidate.targetResource,
          {SyncCoverAnchorKind::ScopeExit, 0, candidate.recurrenceScope});
  if (candidate.kind == SyncCoverSlotProtocolKind::UnitRelease) {
    invalidActions |= !sameResourceAction(
        mechanism.actions[1], SyncCoverResourceActionKind::Consume,
        candidate.targetResource,
        {SyncCoverAnchorKind::BeforeNode, candidate.targets.front(), 0});
  } else {
    std::size_t action = 1;
    for (SyncCoverNodeId waitTarget : uniqueWaitTargets) {
      for (unsigned lane = 0; lane < candidate.width; ++lane) {
        (void)lane;
        const SyncCoverAnchor expectedAnchor =
            candidate.waitScope
                ? SyncCoverAnchor{SyncCoverAnchorKind::ScopeEntry, 0,
                                  *candidate.waitScope}
                : SyncCoverAnchor{SyncCoverAnchorKind::BeforeNode, waitTarget,
                                  0};
        invalidActions |= !sameResourceAction(
            mechanism.actions[action++], SyncCoverResourceActionKind::Consume,
            candidate.targetResource, expectedAnchor);
      }
    }
  }
  for (std::size_t index = 0; index < setCount; ++index) {
    invalidActions |= !sameResourceAction(
        mechanism.actions[firstSetAction + index],
        SyncCoverResourceActionKind::Produce,
        candidate.sourceResource,
        {SyncCoverAnchorKind::AfterNode, candidate.sources[index], 0});
  }
  Operation *outerLoop = findUniqueLoop(candidate.recurrenceScope, loopScopes);
  const std::optional<CanonicalAnchor> prime =
      translateAnchor(nodes, loopScopes, mechanism.actions[0].anchor);
  std::vector<CanonicalAnchor> waits;
  waits.reserve(waitCount);
  for (std::size_t index = 0; index < waitCount; ++index) {
    const std::optional<CanonicalAnchor> wait = translateAnchor(
        nodes, loopScopes, mechanism.actions[index + 1].anchor);
    if (!wait) {
      return false;
    }
    waits.push_back(*wait);
  }
  std::vector<CanonicalAnchor> sets;
  sets.reserve(setCount);
  for (std::size_t index = 0; index < setCount; ++index) {
    const std::optional<CanonicalAnchor> set = translateAnchor(
        nodes, loopScopes, mechanism.actions[firstSetAction + index].anchor);
    if (!set) {
      return false;
    }
    sets.push_back(*set);
  }
  const std::optional<CanonicalAnchor> drain =
      translateAnchor(nodes, loopScopes, mechanism.actions[drainAction].anchor);
  const CanonicalEvent &event = recipe.event;
  const bool invalidMetadata =
      use.scope != candidate.recurrenceScope || use.distance != 1 ||
      use.width != candidate.width ||
      use.actions.size() != mechanism.actions.size() ||
      use.supplyEdges != mechanism.supplyEdges ||
      domain.kind != SyncCoverResourceKind::EventId ||
      domain.sourceResource != candidate.sourceResource ||
      domain.targetResource != candidate.targetResource || invalidActions ||
      !outerLoop || !prime || waits.empty() || sets.empty() || !drain ||
      event.source != candidate.source ||
      event.target != candidate.targets.front() ||
      event.sourcePipe != static_cast<PipelineType>(domain.sourceResource) ||
      event.targetPipe != static_cast<PipelineType>(domain.targetResource) ||
      !sameAnchor(event.setAnchor, sets.front()) ||
      !sameAnchor(event.waitAnchor, waits.front()) ||
      event.recurrenceLoop != outerLoop || event.forwardDrainLoop ||
      event.scopeLoop != outerLoop || event.resourceScopeLoop ||
      event.iterationDistance != 1 || event.width != candidate.width ||
      !event.eventIds.empty() ||
      event.ownershipCycle != 0 || event.ownershipProtocol;
  if (invalidMetadata) {
    return false;
  }
  for (std::size_t index = 0; index < use.actions.size(); ++index) {
    if (use.actions[index] != index) {
      return false;
    }
  }

  CanonicalEventLane all;
  all.kind = CanonicalEventLaneKind::All;
  std::vector<CanonicalEventAction> expectedActions = {
      {CanonicalEventActionKind::Set, CanonicalEventActionPhase::Prime,
       *prime, all}};
  for (std::size_t index = 0; index < waits.size(); ++index) {
    CanonicalEventLane lane;
    lane.index = candidate.kind == SyncCoverSlotProtocolKind::UnitRelease
                     ? 0
                     : static_cast<unsigned>(index % candidate.width);
    expectedActions.push_back({CanonicalEventActionKind::Wait,
                               CanonicalEventActionPhase::Body, waits[index],
                               lane});
  }
  for (std::size_t index = 0; index < sets.size(); ++index) {
    CanonicalEventLane lane;
    lane.index = candidate.sourceLanes[index];
    expectedActions.push_back({CanonicalEventActionKind::Set,
                               CanonicalEventActionPhase::Body, sets[index],
                               lane});
  }
  expectedActions.push_back({CanonicalEventActionKind::Wait,
                             CanonicalEventActionPhase::Drain, *drain, all});
  std::vector<CanonicalEventTrace> expectedTraces{
      makeTrace(CanonicalEventTraceKind::Prime, {0})};
  if (candidate.kind == SyncCoverSlotProtocolKind::UnitRelease) {
    CanonicalEventTrace trace;
    trace.kind = CanonicalEventTraceKind::Cycle;
    trace.actions = {1, static_cast<unsigned>(firstSetAction)};
    expectedTraces.push_back(std::move(trace));
  } else {
    for (const SyncCoverSlotProtocolPath &path : candidate.paths) {
      const auto waitPosition = uniqueWaitTargets.find(path.waitTarget);
      if (waitPosition == uniqueWaitTargets.end()) {
        return false;
      }
      const std::size_t waitIndex = static_cast<std::size_t>(
          std::distance(uniqueWaitTargets.begin(), waitPosition));
      CanonicalEventTrace trace;
      trace.kind = CanonicalEventTraceKind::Cycle;
      for (unsigned lane = 0; lane < candidate.width; ++lane) {
        trace.actions.push_back(static_cast<unsigned>(
            1 + waitIndex * candidate.width + lane));
      }
      for (SyncCoverNodeId source : path.sources) {
        const auto sourcePosition = std::lower_bound(
            candidate.sources.begin(), candidate.sources.end(), source);
        if (sourcePosition == candidate.sources.end() ||
            *sourcePosition != source) {
          return false;
        }
        trace.actions.push_back(static_cast<unsigned>(
            firstSetAction +
            std::distance(candidate.sources.begin(), sourcePosition)));
      }
      expectedTraces.push_back(std::move(trace));
    }
  }
  expectedTraces.push_back(makeTrace(
      CanonicalEventTraceKind::Final,
      {static_cast<unsigned>(drainAction)}));
  const bool invalidProtocol =
      !llvm::equal(event.actions, expectedActions, sameAction) ||
      !llvm::equal(event.traces, expectedTraces, sameTrace) ||
      event.completions.size() != candidate.completionEdges.size();
  if (invalidProtocol) {
    return false;
  }
  for (std::size_t index = 0; index < candidate.completionEdges.size();
       ++index) {
    const std::size_t graphEdge = mechanism.supplyEdges[index];
    if (graphEdge >= universe.getGraph().getEdges().size()) {
      return false;
    }
    const SyncCoverEdge &edge = universe.getGraph().getEdges()[graphEdge];
    const SyncCoverSupplyBinding &binding = mechanism.supplyBindings[index];
    const auto &[source, target] = candidate.completionEdges[index];
    const auto sourcePosition = std::lower_bound(candidate.sources.begin(),
                                                 candidate.sources.end(), source);
    if (sourcePosition == candidate.sources.end() ||
        *sourcePosition != source) {
      return false;
    }
    const std::size_t sourceIndex = static_cast<std::size_t>(
        sourcePosition - candidate.sources.begin());
    unsigned waitAction = 1;
    if (candidate.kind == SyncCoverSlotProtocolKind::HierarchicalRelease) {
      const auto targetPosition = std::lower_bound(
          candidate.targets.begin(), candidate.targets.end(), target);
      if (targetPosition == candidate.targets.end() ||
          *targetPosition != target) {
        return false;
      }
      const std::size_t targetIndex = static_cast<std::size_t>(
          targetPosition - candidate.targets.begin());
      const auto waitPosition = uniqueWaitTargets.find(
          candidate.targetWaits[targetIndex]);
      if (waitPosition == uniqueWaitTargets.end()) {
        return false;
      }
      const std::size_t waitIndex = static_cast<std::size_t>(
          std::distance(uniqueWaitTargets.begin(), waitPosition));
      waitAction = static_cast<unsigned>(
          1 + waitIndex * candidate.width + candidate.sourceLanes[sourceIndex]);
    }
    const unsigned setAction =
        static_cast<unsigned>(firstSetAction + sourceIndex);
    const CanonicalEventCompletion expected{source, target, 1, outerLoop,
                                            setAction, waitAction};
    const bool invalidSupply =
        edge.source != expected.source || edge.target != expected.target ||
        edge.scope != candidate.recurrenceScope || edge.distance != 1 ||
        edge.kind != SyncCoverEdgeKind::CompletionSupply ||
        edge.mechanism != mechanism.id ||
        binding.supplyEdge != graphEdge || binding.resourceUse != 0 ||
        binding.produceAction != setAction ||
        binding.consumeAction != waitAction ||
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
      use.kind != SyncCoverResourceKind::EventId || use.width == 0 ||
      use.materializationEventIndex ||
      allocation.kind != SyncCoverResourceKind::EventId ||
      allocation.domain != use.domain ||
      allocation.sourceResource != use.sourceResource ||
      allocation.targetResource != use.targetResource ||
      allocation.ids.size() != use.width;
  CanonicalEvent event = recipe.event;
  const bool invalidEvent =
      !event.eventIds.empty() || event.width != use.width ||
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
  bundle.events.push_back(std::move(event));
  return bundle;
}
