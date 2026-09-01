// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "SyncCoverCoverageInternal.h"
#include "SyncCoverProtocolInternal.h"

#include "PTO/Transforms/CanonicalSync/SyncCoverExpansion.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::sync_cover_protocol_detail;

namespace {

bool checkedAdd(std::size_t left, std::size_t right, std::size_t &result) {
  const bool overflow = right > std::numeric_limits<std::size_t>::max() - left;
  if (overflow) {
    return false;
  }
  result = left + right;
  return true;
}

bool checkedProduct(std::size_t left, std::size_t right, std::size_t &result) {
  const bool overflow =
      left != 0 && right > std::numeric_limits<std::size_t>::max() / left;
  if (overflow) {
    return false;
  }
  result = left * right;
  return true;
}

bool checkedAccumulate(std::size_t &target, std::size_t amount) {
  std::size_t result = 0;
  if (!checkedAdd(target, amount, result)) {
    return false;
  }
  target = result;
  return true;
}

std::size_t logarithmicLookupWork(std::size_t size) {
  std::size_t work = 1;
  for (std::size_t value = size; value > 1; value = (value + 1) / 2) {
    ++work;
  }
  return work;
}

bool supportsEventDomain(const SyncCoverProtocolTargetContract &target,
                         std::uint32_t source, std::uint32_t destination) {
  using Capability = SyncCoverProtocolTargetContract::EventCapability;
  const auto found = std::lower_bound(target.eventCapabilities.begin(),
                                      target.eventCapabilities.end(),
                                      Capability{source, destination, 0});
  return found != target.eventCapabilities.end() &&
         found->sourceResource == source &&
         found->targetResource == destination;
}

bool scopeContains(const SyncCoverGraph &graph, SyncCoverScopeId ancestor,
                   SyncCoverScopeId descendant,
                   SyncCoverCoverageWorkBudget *workBudget) {
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  const bool invalidScope =
      ancestor >= scopes.size() || descendant >= scopes.size();
  if (invalidScope) {
    return false;
  }
  while (true) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    if (descendant == ancestor) {
      return true;
    }
    if (descendant == 0) {
      return false;
    }
    descendant = scopes[descendant].parent;
  }
}

bool regionContains(const SyncCoverGraph &graph, SyncCoverRegionId ancestor,
                    SyncCoverRegionId descendant,
                    SyncCoverCoverageWorkBudget *workBudget) {
  const std::vector<SyncCoverRegion> &regions = graph.getRegions();
  const bool invalidRegion =
      ancestor >= regions.size() || descendant >= regions.size();
  if (invalidRegion) {
    return false;
  }
  while (true) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    if (descendant == ancestor) {
      return true;
    }
    if (descendant == 0) {
      return false;
    }
    descendant = regions[descendant].parent;
  }
}

std::optional<SyncCoverScopeId>
nearestEnclosingLoop(const SyncCoverGraph &graph, SyncCoverScopeId scope,
                     SyncCoverCoverageWorkBudget *workBudget) {
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  if (scope >= scopes.size()) {
    return std::nullopt;
  }
  while (true) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    if (scopes[scope].isLoop) {
      return scope;
    }
    if (scope == 0) {
      return std::nullopt;
    }
    scope = scopes[scope].parent;
  }
}

bool nodeOccursInProtocolIteration(const SyncCoverGraph &graph,
                                   const SyncCoverEventProtocol &protocol,
                                   SyncCoverNodeId node,
                                   SyncCoverCoverageWorkBudget *workBudget) {
  if (protocol.loop) {
    return scopeContains(graph, protocol.loop->scope,
                         graph.getNodes()[node].scope, workBudget);
  }
  return !nearestEnclosingLoop(graph, graph.getNodes()[node].scope, workBudget);
}

bool demandForcesRegionAtCopy(const SyncCoverGraph &graph,
                              const SyncCoverDemand &demand,
                              SyncCoverRegionId region, unsigned copy,
                              SyncCoverCoverageWorkBudget *workBudget) {
  const bool sourceForces =
      copy == 0 &&
      regionContains(graph, region, graph.getNodes()[demand.source].region,
                     workBudget);
  const bool targetForces =
      copy == demand.distance &&
      regionContains(graph, region, graph.getNodes()[demand.target].region,
                     workBudget);
  return sourceForces || targetForces;
}

bool guardIsCompatibleWithDemandContext(
    const SyncCoverGraph &graph, const SyncCoverDemand &demand,
    const sync_cover_detail::DemandContext &context,
    const SyncCoverGuard &guard, unsigned copy,
    SyncCoverCoverageWorkBudget *workBudget) {
  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    const sync_cover_detail::ContextLiteral candidate{
        literal.control,
        sync_cover_detail::contextCopy(graph, demand, literal.control, copy,
                                       workBudget),
        literal.alternative};
    for (const sync_cover_detail::ContextLiteral &fixed : context.condition) {
      if (!consumeWork(workBudget)) {
        return false;
      }
      if (fixed.control == candidate.control && fixed.copy == candidate.copy) {
        if (fixed.alternative != candidate.alternative) {
          return false;
        }
        break;
      }
      if (candidate < fixed) {
        break;
      }
    }
  }
  return true;
}

std::optional<unsigned>
getNodeAlternative(const SyncCoverGraph &graph, SyncCoverNodeId node,
                   SyncCoverControlId control,
                   SyncCoverCoverageWorkBudget *workBudget) {
  if (node >= graph.getNodes().size() ||
      control >= graph.getControls().size()) {
    return std::nullopt;
  }
  for (const SyncCoverGuardLiteral &literal :
       graph.getNodes()[node].guard.literals) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    if (literal.control == control) {
      return literal.alternative;
    }
  }
  return std::nullopt;
}

bool controlAlternativeIsRequired(
    const SyncCoverGraph &graph, const SyncCoverDemand &demand,
    const sync_cover_detail::DemandContext &context, SyncCoverControlId control,
    unsigned alternative, unsigned copy,
    SyncCoverCoverageWorkBudget *workBudget) {
  const unsigned occurrence =
      sync_cover_detail::contextCopy(graph, demand, control, copy, workBudget);
  for (const sync_cover_detail::ContextLiteral &fixed : context.condition) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    if (fixed.control == control && fixed.copy == occurrence) {
      return fixed.alternative == alternative;
    }
  }
  return true;
}

bool nodeInstanceMustExecute(const SyncCoverGraph &graph,
                             const SyncCoverDemand &demand,
                             const sync_cover_detail::DemandContext &context,
                             SyncCoverNodeId node, unsigned copy,
                             SyncCoverCoverageWorkBudget *workBudget) {
  bool available = sync_cover_detail::nodeInstanceAvailable(graph, demand, node,
                                                            copy, workBudget);
  if (workBudget && workBudget->exhausted) {
    return false;
  }
  available |= scopeContains(graph, demand.scope, graph.getNodes()[node].scope,
                             workBudget);
  const bool guardIsProven = sync_cover_detail::guardIsImplied(
      graph, demand, context, graph.getNodes()[node].guard, copy, workBudget);
  if (!available || !guardIsProven) {
    return false;
  }
  const bool isDemandEndpoint =
      (node == demand.source && copy == 0) ||
      (node == demand.target && copy == demand.distance);
  if (isDemandEndpoint) {
    return true;
  }
  SyncCoverRegionId region = graph.getNodes()[node].region;
  const SyncCoverRegionId owner = demand.ownerRegion;
  while (region != owner) {
    const bool invalidRegion = !consumeWork(workBudget) || region == 0 ||
                               region >= graph.getRegions().size();
    if (invalidRegion) {
      return false;
    }
    const SyncCoverRegion &description = graph.getRegions()[region];
    const bool forced =
        demandForcesRegionAtCopy(graph, demand, region, copy, workBudget);
    if (workBudget && workBudget->exhausted) {
      return false;
    }
    const bool repeated =
        description.kind == SyncCoverRegionKind::Loop ||
        description.cardinality == SyncCoverRegionCardinality::ZeroOrMore ||
        description.cardinality == SyncCoverRegionCardinality::OneOrMore;
    const bool guardIsPresent = !description.guard.literals.empty();
    const bool guardIsProven =
        guardIsPresent &&
        sync_cover_detail::guardIsImplied(graph, demand, context,
                                          description.guard, copy, workBudget);
    const bool unprovenOptional =
        !forced &&
        description.cardinality == SyncCoverRegionCardinality::ZeroOrOne &&
        (!guardIsPresent || !guardIsProven);
    const bool unprovenGuard = !forced && guardIsPresent && !guardIsProven;
    // A repeated region is not globally must-execute, but a demand endpoint
    // inside that region proves the corresponding dynamic occurrence.  Nodes
    // on its forced straight-line path may therefore participate in the same
    // cut instance.  Reject only repeated regions whose occurrence is not
    // forced by either endpoint.
    if ((!forced && repeated) || unprovenOptional || unprovenGuard) {
      return false;
    }
    region = description.parent;
  }
  return true;
}

bool pointExecutesAtCopy(const SyncCoverGraph &graph,
                         const SyncCoverDemand &demand,
                         const sync_cover_detail::DemandContext &context,
                         const SyncCoverCutPoint &point, unsigned copy,
                         const std::vector<std::uint8_t> *mustExecute,
                         SyncCoverCoverageWorkBudget *workBudget) {
  const std::optional<SyncCoverGuard> guard =
      effectivePointGuard(graph, point, workBudget);
  if (!guard || !sync_cover_detail::guardIsImplied(graph, demand, context,
                                                   *guard, copy, workBudget)) {
    return false;
  }
  const bool nodeAnchor =
      point.anchor.kind == SyncCoverAnchorKind::BeforeNode ||
      point.anchor.kind == SyncCoverAnchorKind::AfterNode;
  if (nodeAnchor) {
    return mustExecute
               ? (*mustExecute)[static_cast<std::size_t>(copy) *
                                    graph.getNodes().size() +
                                point.anchor.node] != 0
               : nodeInstanceMustExecute(graph, demand, context,
                                         point.anchor.node, copy, workBudget);
  }
  const bool scopeBoundary =
      point.anchor.kind == SyncCoverAnchorKind::ScopeEntry ||
      point.anchor.kind == SyncCoverAnchorKind::ScopeExit ||
      point.anchor.kind == SyncCoverAnchorKind::LoopBodyEntry ||
      point.anchor.kind == SyncCoverAnchorKind::LoopBodyExit;
  const bool controlBoundary =
      point.anchor.kind == SyncCoverAnchorKind::ControlEntry ||
      point.anchor.kind == SyncCoverAnchorKind::ControlExit;
  if ((!scopeBoundary && !controlBoundary) ||
      point.anchor.scope >= graph.getScopes().size()) {
    return false;
  }
  const SyncCoverScopeId pointScope = point.anchor.scope;
  const bool localOccurrence =
      scopeContains(graph, demand.scope, pointScope, workBudget);
  const bool forcedBySource =
      copy == 0 &&
      scopeContains(graph, pointScope, graph.getNodes()[demand.source].scope,
                    workBudget);
  const bool forcedByTarget =
      copy == demand.distance &&
      scopeContains(graph, pointScope, graph.getNodes()[demand.target].scope,
                    workBudget);
  return localOccurrence || forcedBySource || forcedByTarget;
}

bool protocolActionExecutesAtCopy(
    const SyncCoverGraph &graph, const SyncCoverEventProtocol &protocol,
    const SyncCoverProtocolAction *action, const SyncCoverDemand &demand,
    const sync_cover_detail::DemandContext &context, unsigned copy,
    SyncCoverCoverageWorkBudget *workBudget) {
  if (!action || !protocol.loop || !protocol.loop->phaseControl ||
      action->segment != SyncCoverProtocolActionSegment::Body ||
      demand.scope != protocol.loop->scope ||
      *protocol.loop->phaseControl >= graph.getControls().size()) {
    return false;
  }
  const SyncCoverControlId phaseControl = *protocol.loop->phaseControl;
  const SyncCoverControl &control = graph.getControls()[phaseControl];
  if (!control.phaseRelation) {
    return false;
  }
  const auto sourceLiteral = std::find_if(
      demand.sourceGuard.literals.begin(), demand.sourceGuard.literals.end(),
      [&](const SyncCoverGuardLiteral &literal) {
        return literal.control == phaseControl;
      });
  if (sourceLiteral == demand.sourceGuard.literals.end()) {
    return false;
  }
  const std::optional<SyncCoverGuard> pointGuard =
      effectivePointGuard(graph, action->point, workBudget);
  if (!pointGuard) {
    return false;
  }
  SyncCoverGuard nonPhaseGuard = *pointGuard;
  nonPhaseGuard.literals.erase(
      std::remove_if(nonPhaseGuard.literals.begin(),
                     nonPhaseGuard.literals.end(),
                     [&](const SyncCoverGuardLiteral &literal) {
                       return literal.control == phaseControl;
                     }),
      nonPhaseGuard.literals.end());
  if (!sync_cover_detail::guardIsImplied(graph, demand, context, nonPhaseGuard,
                                         copy, workBudget)) {
    return false;
  }
  const SyncCoverControlPhaseRelation &relation = *control.phaseRelation;
  bool foundSourcePhase = false;
  for (std::size_t initial = 0; initial < relation.activeAlternative.size();
       ++initial) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    if (relation.activeAlternative[initial] != sourceLiteral->alternative) {
      continue;
    }
    foundSourcePhase = true;
    std::size_t phase = initial;
    for (unsigned step = 0; step < copy; ++step) {
      if (!consumeWork(workBudget) || phase >= relation.nextPhase.size()) {
        return false;
      }
      phase = relation.nextPhase[phase];
    }
    if (phase >= relation.nextPhase.size() ||
        (!action->activePhases.empty() &&
         !std::binary_search(action->activePhases.begin(),
                             action->activePhases.end(), phase))) {
      return false;
    }
  }
  if (!foundSourcePhase) {
    return false;
  }
  switch (action->guard) {
  case SyncCoverProtocolActionGuard::Always:
  case SyncCoverProtocolActionGuard::LoopNonEmpty:
    return true;
  case SyncCoverProtocolActionGuard::FirstIteration:
    return copy == 0;
  case SyncCoverProtocolActionGuard::NotFirstIteration:
    return copy != 0;
  case SyncCoverProtocolActionGuard::HasSuccessor:
    if (copy < demand.distance) {
      return true;
    }
    return std::any_of(
        demand.targetGuard.literals.begin(), demand.targetGuard.literals.end(),
        [&](const SyncCoverGuardLiteral &literal) {
          if (!consumeWork(workBudget) ||
              literal.control >= graph.getControls().size()) {
            return false;
          }
          const auto &successor =
              graph.getControls()[literal.control].successorRelation;
          return successor && successor->loopScope == protocol.loop->scope &&
                 successor->hasSuccessorAlternative == literal.alternative;
        });
  case SyncCoverProtocolActionGuard::LoopEmpty:
    return false;
  }
  return false;
}

bool nodeInSourcePrefix(
    const SyncCoverGraph &graph, const SyncCoverEventProtocol &protocol,
    const SyncCoverProtocolTargetContract &target,
    const SyncCoverEventTransfer &transfer, const SyncCoverDemand &demand,
    const sync_cover_detail::DemandContext &context, SyncCoverNodeId node,
    unsigned nodeCopy, unsigned setCopy,
    const SyncCoverProtocolAction *setAction, bool allowExternalSource,
    const std::vector<std::uint8_t> *mustExecute,
    SyncCoverCoverageWorkBudget *workBudget) {
  const bool sourceInProtocolScope =
      protocol.loop
          ? scopeContains(graph, protocol.loop->scope,
                          graph.getNodes()[node].scope, workBudget)
          : nodeOccursInProtocolIteration(graph, protocol, node, workBudget);
  bool sourceMustExecute =
      mustExecute ? (*mustExecute)[static_cast<std::size_t>(nodeCopy) *
                                       graph.getNodes().size() +
                                   node] != 0
                  : nodeInstanceMustExecute(graph, demand, context, node,
                                            nodeCopy, workBudget);
  if (!sourceMustExecute &&
      transfer.set.anchor.kind == SyncCoverAnchorKind::ControlExit) {
    const std::optional<unsigned> alternative =
        getNodeAlternative(graph, node, transfer.set.anchor.node, workBudget);
    sourceMustExecute =
        alternative && guardIsCompatibleWithDemandContext(
                           graph, demand, context, graph.getNodes()[node].guard,
                           nodeCopy, workBudget);
  }
  const bool setExecutes =
      pointExecutesAtCopy(graph, demand, context, transfer.set, setCopy,
                          mustExecute, workBudget) ||
      protocolActionExecutesAtCopy(graph, protocol, setAction, demand, context,
                                   setCopy, workBudget);
  if (graph.getNodes()[node].resource != transfer.set.resource ||
      (!allowExternalSource && !sourceInProtocolScope) || !sourceMustExecute ||
      !setExecutes || setCopy < nodeCopy) {
    return false;
  }
  if (!target.directEventCompletesSourcePrefix) {
    return transfer.set.anchor.kind == SyncCoverAnchorKind::AfterNode &&
           transfer.set.anchor.node == node && nodeCopy == setCopy;
  }
  // A target-certified SetFlag waits for every preceding source-pipeline
  // access. A repeated cut in a later iteration therefore completes the
  // source prefix accumulated by earlier iterations; no same-pipe
  // completion-order assumption is needed.
  if (nodeCopy < setCopy) {
    return true;
  }
  const std::optional<SyncCoverTimelinePosition> set =
      resolveSyncCoverAnchor(graph, transfer.set.anchor);
  const std::optional<SyncCoverTimelinePosition> position =
      resolveSyncCoverAnchor(graph, {SyncCoverAnchorKind::AfterNode, node,
                                     graph.getNodes()[node].scope, 0});
  return set && position && *position <= *set;
}

bool nodeInTargetSuffix(
    const SyncCoverGraph &graph, const SyncCoverEventProtocol &protocol,
    const SyncCoverEventTransfer &transfer, const SyncCoverDemand &demand,
    const sync_cover_detail::DemandContext &context, SyncCoverNodeId node,
    unsigned nodeCopy, unsigned waitCopy, bool exitExport,
    const SyncCoverProtocolAction *waitAction, bool sourceProvesNonzero,
    bool loopSummary, bool allowExternalTarget,
    const std::vector<std::uint8_t> *mustExecute,
    SyncCoverCoverageWorkBudget *workBudget) {
  bool targetMustExecute =
      mustExecute ? (*mustExecute)[static_cast<std::size_t>(nodeCopy) *
                                       graph.getNodes().size() +
                                   node] != 0
                  : nodeInstanceMustExecute(graph, demand, context, node,
                                            nodeCopy, workBudget);
  if (!targetMustExecute &&
      transfer.wait.anchor.kind == SyncCoverAnchorKind::ControlEntry) {
    const std::optional<unsigned> alternative =
        getNodeAlternative(graph, node, transfer.wait.anchor.node, workBudget);
    targetMustExecute =
        alternative && guardIsCompatibleWithDemandContext(
                           graph, demand, context, graph.getNodes()[node].guard,
                           nodeCopy, workBudget);
  }
  if (graph.getNodes()[node].resource != transfer.wait.resource ||
      !targetMustExecute) {
    return false;
  }
  const std::optional<SyncCoverTimelinePosition> wait =
      resolveSyncCoverAnchor(graph, transfer.wait.anchor);
  const std::optional<SyncCoverTimelinePosition> position =
      resolveSyncCoverAnchor(graph, {SyncCoverAnchorKind::BeforeNode, node,
                                     graph.getNodes()[node].scope, 0});
  if (!wait || !position) {
    return false;
  }
  if (nodeCopy < waitCopy) {
    return false;
  }
  const bool waitExecutes =
      pointExecutesAtCopy(graph, demand, context, transfer.wait, waitCopy,
                          mustExecute, workBudget) ||
      protocolActionExecutesAtCopy(graph, protocol, waitAction, demand, context,
                                   waitCopy, workBudget);
  if (nodeOccursInProtocolIteration(graph, protocol, node, workBudget)) {
    const bool ordinarySuffix =
        waitExecutes && (waitCopy < nodeCopy || *wait <= *position);
    return ordinarySuffix || loopSummary;
  }
  if (allowExternalTarget) {
    return waitExecutes && (waitCopy < nodeCopy || *wait <= *position);
  }
  if (!protocol.loop || !exitExport ||
      (protocol.loop->mayExecuteZeroTimes && !sourceProvesNonzero)) {
    return false;
  }
  const std::optional<SyncCoverTimelineInterval> &timeline =
      graph.getScopes()[protocol.loop->scope].timeline;
  return timeline && waitCopy <= nodeCopy &&
         (waitCopy < nodeCopy || *position > timeline->end) &&
         scopeContains(graph, graph.getNodes()[node].scope,
                       protocol.loop->scope, workBudget);
}

std::vector<SyncCoverEventTransfer>
getChannelTransfers(const SyncCoverEventChannel &channel) {
  if (!channel.transfers.empty()) {
    return channel.transfers;
  }
  SyncCoverEventTransfer transfer;
  transfer.set = channel.set;
  transfer.wait = channel.wait;
  transfer.activePhases = channel.activePhases;
  return {std::move(transfer)};
}

struct PreparedProtocolRectangle {
  std::size_t activationId = 0;
  SyncCoverEventTransfer transfer;
  unsigned distance = 0;
  std::optional<SyncCoverScopeId> distanceScope;
  bool externalSource = false;
  bool externalTarget = false;
  bool loopSummary = false;
  bool completionExport = false;
  const SyncCoverProtocolAction *setAction = nullptr;
  const SyncCoverProtocolAction *waitAction = nullptr;
};

struct PreparedProtocolChannel {
  std::size_t protocolIndex = 0;
  const SyncCoverEventChannel *channel = nullptr;
  std::vector<PreparedProtocolRectangle> rectangles;
};

PreparedProtocolChannel
prepareProtocolChannel(const SyncCoverGraph &graph,
                       const std::vector<SyncCoverEventProtocol> &protocols,
                       std::size_t protocolIndex,
                       const SyncCoverEventChannel &channel,
                       std::size_t &nextActivationId) {
  PreparedProtocolChannel prepared;
  prepared.protocolIndex = protocolIndex;
  prepared.channel = &channel;
  const SyncCoverEventProtocol &protocol = protocols[protocolIndex];
  if (!channel.actions.empty()) {
    prepared.rectangles.reserve(channel.supplies.size());
    for (const SyncCoverProtocolSupply &supply : channel.supplies) {
      const SyncCoverProtocolAction &set = channel.actions[supply.setAction];
      const SyncCoverProtocolAction &wait = channel.actions[supply.waitAction];
      PreparedProtocolRectangle rectangle;
      rectangle.activationId = nextActivationId++;
      rectangle.transfer.set = set.point;
      rectangle.transfer.wait = wait.point;
      rectangle.distance = supply.distance;
      rectangle.distanceScope = supply.distanceScope;
      rectangle.completionExport =
          supply.kind == SyncCoverProtocolSupplyKind::CompletionExport;
      rectangle.setAction = &set;
      rectangle.waitAction = &wait;
      rectangle.externalSource =
          set.segment == SyncCoverProtocolActionSegment::Entry;
      rectangle.externalTarget = supply.distanceScope.has_value();
      const std::size_t initialPhase =
          protocol.loop && protocol.loop->phaseControl
              ? graph.getControls()[*protocol.loop->phaseControl]
                    .phaseRelation->initialPhase
              : 0;
      const bool waitStartsInvocation =
          wait.segment == SyncCoverProtocolActionSegment::Body &&
          (wait.activePhases.empty() ||
           std::binary_search(wait.activePhases.begin(),
                              wait.activePhases.end(), initialPhase));
      const bool parentInvocationCarry =
          rectangle.completionExport && protocol.loop && supply.distanceScope &&
          *supply.distanceScope != protocol.loop->scope;
      // A verified hierarchical lifecycle keeps its token live across child
      // invocations.  A Wait that is guaranteed in the authoritative initial
      // child phase establishes completion for the remainder of that child
      // invocation, even when the parent demand names a later child phase.
      // The protocol automaton validates the cross-invocation token pairing;
      // exact-world grounding still checks the physical source prefix and the
      // parent-loop displacement.
      rectangle.loopSummary =
          waitStartsInvocation &&
          ((rectangle.externalSource && supply.distance == 0) ||
           parentInvocationCarry);
      prepared.rectangles.push_back(std::move(rectangle));
    }
    return prepared;
  }
  for (SyncCoverEventTransfer transfer : getChannelTransfers(channel)) {
    const unsigned distance =
        channel.flow == SyncCoverEventChannelFlow::LoopCarry ? channel.distance
                                                             : 0;
    prepared.rectangles.push_back({nextActivationId++, std::move(transfer),
                                   distance, std::nullopt, false, false, false,
                                   false, nullptr, nullptr});
  }
  return prepared;
}

} // namespace

static SyncCoverProtocolCoverageResult computeProtocolWorlds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverExactWorld> &worlds,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget,
    bool includeTransitiveComposition,
    const SyncCoverDemandSet *queriedDemands = nullptr) {
  SyncCoverProtocolCoverageResult result;
  const bool graphLimitExceeded = !graphFitsProtocolLimits(graph, limits) ||
                                  !targetFitsProtocolLimits(target, limits);
  if (graphLimitExceeded) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  const bool invalidGraph =
      !graph.isStructureFrozen() || !graph.validate() ||
      (queriedDemands && queriedDemands->size() != graph.getDemands().size());
  if (invalidGraph) {
    result.error = SyncCoverProtocolError::InvalidGraph;
    return result;
  }
  const SyncCoverProtocolError targetError =
      validateProtocolTargetContract(target, limits, workBudget);
  if (targetError != SyncCoverProtocolError::None) {
    result.error = targetError;
    return result;
  }
  const bool rowLimitExceeded = protocols.size() > limits.maximumProtocols ||
                                worlds.size() > limits.maximumWorlds ||
                                worlds.size() > limits.maximumResultRows;
  const std::size_t wordsPerWorld = (graph.getDemands().size() + 63) / 64;
  std::size_t resultWords = 0;
  const bool wordLimitExceeded =
      !checkedProduct(wordsPerWorld, worlds.size(), resultWords) ||
      resultWords > limits.maximumResultWords;
  if (rowLimitExceeded || wordLimitExceeded) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  if (!consumeWork(workBudget, worlds.size())) {
    result.error = SyncCoverProtocolError::WorkLimitExceeded;
    return result;
  }
  std::size_t worldIncidences = 0;
  for (const SyncCoverExactWorld &world : worlds) {
    const bool worldIncidenceLimitExceeded =
        !checkedAccumulate(worldIncidences, world.enabledMechanisms.size()) ||
        worldIncidences > limits.maximumWorldMechanismIncidences;
    if (worldIncidenceLimitExceeded) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      return result;
    }
  }
  std::size_t protocolIndexWork = 0;
  std::size_t worldIncidenceWork = 0;
  std::size_t worldWorkPerIncidence = 0;
  const bool indexWorkOverflow =
      !checkedProduct(protocols.size(),
                      logarithmicLookupWork(protocols.size()) + 1,
                      protocolIndexWork) ||
      !checkedAdd(logarithmicLookupWork(protocols.size()), 3,
                  worldWorkPerIncidence) ||
      !checkedProduct(worldIncidences, worldWorkPerIncidence,
                      worldIncidenceWork);
  std::size_t preparationWork = 0;
  const bool preparationFailed =
      indexWorkOverflow ||
      !checkedAdd(resultWords, protocolIndexWork, preparationWork) ||
      !checkedAdd(preparationWork, worldIncidenceWork, preparationWork) ||
      !consumeWork(workBudget, preparationWork);
  if (preparationFailed) {
    result.error = SyncCoverProtocolError::WorkLimitExceeded;
    return result;
  }
  result.coveredByWorld.reserve(worlds.size());
  std::map<SyncCoverMechanismId, std::size_t> protocolByMechanism;
  std::vector<std::vector<bool>> exitExportsByProtocol;
  exitExportsByProtocol.reserve(protocols.size());
  std::size_t rearmProofs = 0;
  std::size_t exitExports = 0;
  std::size_t exitExportGuardLiterals = 0;
  for (std::size_t index = 0; index < protocols.size(); ++index) {
    const bool proofLimitExceeded =
        !checkedAccumulate(rearmProofs, protocols[index].rearmProofs.size()) ||
        rearmProofs > limits.maximumRearmProofs;
    if (proofLimitExceeded) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    const bool duplicateMechanism =
        !protocolByMechanism.emplace(protocols[index].mechanism, index).second;
    if (duplicateMechanism) {
      result.error = SyncCoverProtocolError::InvalidProtocol;
      result.invalidIndex = index;
      return result;
    }
    SyncCoverProtocolLimits remaining = limits;
    remaining.maximumTotalDynamicActions -=
        result.statistics.totalDynamicActions;
    remaining.maximumAutomatonEdges -= result.statistics.automatonEdges;
    remaining.maximumRearmQueries -= result.statistics.rearmQueries;
    remaining.maximumRearmLookupWork -= result.statistics.rearmLookupWork;
    remaining.maximumReachabilityWork -= result.statistics.reachabilityWork;
    remaining.maximumRearmProofLaneIncidences -=
        result.statistics.rearmProofLaneIncidences;
    remaining.maximumLaneInitializationWork -=
        result.statistics.laneInitializationWork;
    remaining.maximumExitExports -= exitExports;
    remaining.maximumExitExportGuardLiterals -= exitExportGuardLiterals;
    SyncCoverProtocolVerificationResult verification =
        verifyProtocolAssumingValidGraph(graph, target, protocols[index],
                                         remaining, workBudget, false);
    if (!verification) {
      result.error = verification.error;
      result.invalidIndex = index;
      return result;
    }
    if (!consumeWork(workBudget, verification.exitExports.size())) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    std::vector<bool> exported(protocols[index].channels.size());
    for (const SyncCoverProtocolExitExport &candidate :
         verification.exitExports) {
      if (candidate.channel >= exported.size()) {
        result.error = SyncCoverProtocolError::InvalidProtocol;
        result.invalidIndex = index;
        return result;
      }
      exported[candidate.channel] = candidate.availableOnNonzeroTrip;
      const bool exportOverflow =
          !checkedAccumulate(exitExports, 1) ||
          !checkedAccumulate(exitExportGuardLiterals,
                             candidate.guard.literals.size());
      if (exportOverflow) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = index;
        return result;
      }
    }
    exitExportsByProtocol.push_back(std::move(exported));
    const bool statisticsOverflow =
        !checkedAccumulate(result.statistics.reachablePhases,
                           verification.statistics.reachablePhases) ||
        !checkedAccumulate(result.statistics.tripCountsChecked,
                           verification.statistics.tripCountsChecked) ||
        !checkedAccumulate(result.statistics.automatonEdges,
                           verification.statistics.automatonEdges) ||
        !checkedAccumulate(result.statistics.rearmQueries,
                           verification.statistics.rearmQueries) ||
        !checkedAccumulate(result.statistics.rearmProofLaneIncidences,
                           verification.statistics.rearmProofLaneIncidences) ||
        !checkedAccumulate(result.statistics.rearmLookupWork,
                           verification.statistics.rearmLookupWork) ||
        !checkedAccumulate(result.statistics.totalDynamicActions,
                           verification.statistics.totalDynamicActions) ||
        !checkedAccumulate(result.statistics.laneInitializationWork,
                           verification.statistics.laneInitializationWork) ||
        !checkedAccumulate(result.statistics.reachabilityWork,
                           verification.statistics.reachabilityWork);
    if (statisticsOverflow ||
        result.statistics.totalDynamicActions >
            limits.maximumTotalDynamicActions ||
        result.statistics.automatonEdges > limits.maximumAutomatonEdges ||
        result.statistics.rearmQueries > limits.maximumRearmQueries ||
        result.statistics.rearmLookupWork > limits.maximumRearmLookupWork ||
        result.statistics.rearmProofLaneIncidences >
            limits.maximumRearmProofLaneIncidences ||
        result.statistics.laneInitializationWork >
            limits.maximumLaneInitializationWork ||
        result.statistics.reachabilityWork > limits.maximumReachabilityWork ||
        exitExports > limits.maximumExitExports ||
        exitExportGuardLiterals > limits.maximumExitExportGuardLiterals) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    result.statistics.maximumDynamicActions =
        std::max(result.statistics.maximumDynamicActions,
                 verification.statistics.maximumDynamicActions);
  }

  std::vector<std::vector<const SyncCoverEdge *>> outgoingEdges(
      graph.getNodes().size());
  std::map<std::uint32_t, std::vector<SyncCoverNodeId>> nodesByResource;
  std::size_t indexWork = 0;
  if (!checkedAdd(graph.getNodes().size(), graph.getEdges().size(),
                  indexWork) ||
      !checkedAdd(indexWork, graph.getNodes().size(), indexWork) ||
      !consumeWork(workBudget, indexWork)) {
    result.error = workBudget && workBudget->exhausted
                       ? SyncCoverProtocolError::WorkLimitExceeded
                       : SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  for (const SyncCoverEdge &edge : graph.getEdges()) {
    if (edge.source >= outgoingEdges.size()) {
      result.error = SyncCoverProtocolError::InvalidGraph;
      return result;
    }
    outgoingEdges[edge.source].push_back(&edge);
  }
  for (const SyncCoverNode &node : graph.getNodes()) {
    nodesByResource[node.resource].push_back(node.id);
  }
  for (auto &[resource, nodes] : nodesByResource) {
    (void)resource;
    std::sort(nodes.begin(), nodes.end(),
              [&](SyncCoverNodeId left, SyncCoverNodeId right) {
                return std::tie(graph.getNodes()[left].order, left) <
                       std::tie(graph.getNodes()[right].order, right);
              });
  }

  SyncCoverExpansionLimits expansionLimits;
  expansionLimits.maximumArenaNodes =
      std::max<std::size_t>(1, limits.maximumCoverageStates / 16);
  expansionLimits.maximumTotalNodes = expansionLimits.maximumArenaNodes;
  expansionLimits.maximumArenaEdges = limits.maximumCoverageTransitions;
  expansionLimits.maximumTotalEdges = expansionLimits.maximumArenaEdges;
  const SyncCoverExpandedProgram expansion(graph, expansionLimits);
  if (!expansion) {
    result.error = expansion.getError() == SyncCoverExpansionError::InvalidGraph
                       ? SyncCoverProtocolError::InvalidGraph
                       : SyncCoverProtocolError::LimitExceeded;
    return result;
  }

  std::size_t coverageTransitions = 0;
  for (std::size_t worldIndex = 0; worldIndex < worlds.size(); ++worldIndex) {
    const SyncCoverExactWorld &world = worlds[worldIndex];
    if (!std::is_sorted(world.enabledMechanisms.begin(),
                        world.enabledMechanisms.end()) ||
        std::adjacent_find(world.enabledMechanisms.begin(),
                           world.enabledMechanisms.end()) !=
            world.enabledMechanisms.end()) {
      result.error = SyncCoverProtocolError::InvalidProtocol;
      result.invalidIndex = worldIndex;
      return result;
    }
    SyncCoverDemandSet covered(graph.getDemands().size());
    std::vector<std::size_t> enabledProtocols;
    enabledProtocols.reserve(world.enabledMechanisms.size());
    for (SyncCoverMechanismId mechanism : world.enabledMechanisms) {
      const auto found = protocolByMechanism.find(mechanism);
      if (found == protocolByMechanism.end()) {
        result.error = SyncCoverProtocolError::InvalidProtocol;
        result.invalidIndex = worldIndex;
        return result;
      }
      enabledProtocols.push_back(found->second);
    }
    std::map<std::uint32_t, std::vector<PreparedProtocolChannel>>
        channelsBySourceResource;
    std::size_t nextActivationId = 0;
    std::size_t channelIndexIncidences = 0;
    for (std::size_t protocolIndex : enabledProtocols) {
      for (const SyncCoverEventChannel &channel :
           protocols[protocolIndex].channels) {
        if (!checkedAccumulate(channelIndexIncidences, 1) ||
            channelIndexIncidences >
                limits.maximumLifecycleConnectorIncidences ||
            !consumeWork(workBudget)) {
          result.error = workBudget && workBudget->exhausted
                             ? SyncCoverProtocolError::WorkLimitExceeded
                             : SyncCoverProtocolError::LimitExceeded;
          result.invalidIndex = worldIndex;
          return result;
        }
        PreparedProtocolChannel prepared = prepareProtocolChannel(
            graph, protocols, protocolIndex, channel, nextActivationId);
        const std::size_t rectangleCount = prepared.rectangles.size();
        if (!checkedAccumulate(channelIndexIncidences, rectangleCount) ||
            channelIndexIncidences >
                limits.maximumLifecycleConnectorIncidences ||
            !consumeWork(workBudget, rectangleCount)) {
          result.error = workBudget && workBudget->exhausted
                             ? SyncCoverProtocolError::WorkLimitExceeded
                             : SyncCoverProtocolError::LimitExceeded;
          result.invalidIndex = worldIndex;
          return result;
        }
        channelsBySourceResource[channel.set.resource].push_back(
            std::move(prepared));
      }
    }
    for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
         ++demandId) {
      if (queriedDemands && !queriedDemands->contains(demandId)) {
        continue;
      }
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      const std::optional<SyncCoverScopeId> demandSourceLoop =
          nearestEnclosingLoop(graph, graph.getNodes()[demand.source].scope,
                               workBudget);
      const std::optional<SyncCoverScopeId> demandTargetLoop =
          nearestEnclosingLoop(graph, graph.getNodes()[demand.target].scope,
                               workBudget);
      const sync_cover_detail::DemandContext context =
          sync_cover_detail::makeDemandContext(graph, demand, workBudget);
      if (!context.valid) {
        result.error = workBudget && workBudget->exhausted
                           ? SyncCoverProtocolError::WorkLimitExceeded
                           : SyncCoverProtocolError::InvalidGraph;
        result.invalidIndex = demandId;
        return result;
      }
      bool directlyCovered = false;
      const auto matchingChannels = channelsBySourceResource.find(
          graph.getNodes()[demand.source].resource);
      if (matchingChannels != channelsBySourceResource.end()) {
        for (const PreparedProtocolChannel &prepared :
             matchingChannels->second) {
          const std::size_t protocolIndex = prepared.protocolIndex;
          const SyncCoverEventProtocol &protocol = protocols[protocolIndex];
          const SyncCoverEventChannel &channel = *prepared.channel;
          const bool localProtocol =
              protocol.loop
                  ? (demand.distance == 0
                         ? scopeContains(graph, protocol.loop->scope,
                                         graph.getNodes()[demand.source].scope,
                                         workBudget) &&
                               scopeContains(
                                   graph, protocol.loop->scope,
                                   graph.getNodes()[demand.target].scope,
                                   workBudget)
                         : demand.scope == protocol.loop->scope)
                  : demand.distance == 0 && !demandSourceLoop &&
                        !demandTargetLoop;
          const bool nestedExport =
              protocol.loop && protocol.loop->scope != demand.scope &&
              scopeContains(graph, demand.scope, protocol.loop->scope,
                            workBudget);
          if (workBudget && workBudget->exhausted) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            result.invalidIndex = demandId;
            return result;
          }
          if (!localProtocol && !nestedExport) {
            continue;
          }
          if ((channel.suppliedRequirements & demand.orderingRequirements) !=
              demand.orderingRequirements) {
            continue;
          }
          for (const PreparedProtocolRectangle &rectangle :
               prepared.rectangles) {
            if (coverageTransitions == limits.maximumCoverageTransitions ||
                !consumeWork(workBudget)) {
              result.error = workBudget && workBudget->exhausted
                                 ? SyncCoverProtocolError::WorkLimitExceeded
                                 : SyncCoverProtocolError::LimitExceeded;
              result.invalidIndex = demandId;
              return result;
            }
            ++coverageTransitions;
            const bool rectangleDistanceApplies =
                rectangle.distance != 0 &&
                (rectangle.distanceScope
                     ? *rectangle.distanceScope == demand.scope
                     : localProtocol);
            const unsigned effectiveDistance =
                rectangleDistanceApplies ? rectangle.distance : 0;
            const bool nestedProtocol =
                protocol.loop && demand.scope != protocol.loop->scope &&
                graph.scopeContains(demand.scope, protocol.loop->scope);
            const bool explicitParentSummary =
                rectangle.distanceScope &&
                *rectangle.distanceScope == demand.scope;
            // A child-loop body cut repeats in the child's iteration space.
            // It cannot be reinterpreted as a parent-loop recurrence merely
            // because the static endpoint nodes are shared. Only an explicit,
            // verified parent-distance supply may cross that boundary.
            if (nestedProtocol && demand.distance != 0 &&
                !explicitParentSummary) {
              continue;
            }
            bool rectangleCovers = false;
            // Legacy transfers are body cuts repeated by the protocol loop.
            // A Set in a later demand copy drains the source-pipeline prefix
            // accumulated by preceding copies, and its Wait establishes a
            // completion fact that remains valid for the target suffix in
            // later copies. Explicit entry/exit recipes are not shifted here;
            // their parent-distance effects require a verified summary.
            const unsigned lastSetCopy =
                channel.actions.empty() ? demand.distance : 0;
            for (unsigned setCopy = 0;
                 !rectangleCovers && setCopy <= lastSetCopy; ++setCopy) {
              if (effectiveDistance > demand.distance - setCopy) {
                continue;
              }
              const unsigned waitCopy = setCopy + effectiveDistance;
              const bool sourcePrefix = nodeInSourcePrefix(
                  graph, protocol, target, rectangle.transfer, demand, context,
                  demand.source, 0, setCopy, rectangle.setAction,
                  rectangle.externalSource, nullptr, workBudget);
              const bool targetSuffix = nodeInTargetSuffix(
                  graph, protocol, rectangle.transfer, demand, context,
                  demand.target, demand.distance, waitCopy,
                  channel.id < exitExportsByProtocol[protocolIndex].size() &&
                      exitExportsByProtocol[protocolIndex][channel.id],
                  rectangle.waitAction,
                  protocol.loop && demandTargetLoop == protocol.loop->scope,
                  rectangle.loopSummary,
                  rectangle.externalTarget && rectangle.distanceScope &&
                      *rectangle.distanceScope == demand.scope,
                  nullptr, workBudget);
              rectangleCovers = sourcePrefix && targetSuffix;
              if (workBudget && workBudget->exhausted) {
                result.error = SyncCoverProtocolError::WorkLimitExceeded;
                result.invalidIndex = demandId;
                return result;
              }
            }
            if (!rectangleCovers) {
              continue;
            }
            directlyCovered = true;
            break;
          }
          if (directlyCovered) {
            break;
          }
        }
      }
      if (directlyCovered) {
        covered.insert(demandId);
        continue;
      }
      if (!includeTransitiveComposition) {
        continue;
      }
      const SyncCoverExpandedArena *arena = expansion.getArena(demand);
      const bool stateOverflow =
          !arena ||
          arena->getVirtualNodeCount() > limits.maximumCoverageStates / 16;
      if (stateOverflow) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = demandId;
        return result;
      }
      std::vector<std::uint16_t> reached(arena->getVirtualNodeCount());
      std::size_t distanceCount = 0;
      std::size_t graphInstances = 0;
      if (!checkedAdd(static_cast<std::size_t>(demand.distance), 1,
                      distanceCount) ||
          !checkedProduct(distanceCount, graph.getNodes().size(),
                          graphInstances)) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = demandId;
        return result;
      }
      std::size_t activationEntries = 0;
      std::size_t activationEntriesWithTripState = 0;
      if (!checkedProduct(nextActivationId, distanceCount, activationEntries) ||
          !checkedProduct(activationEntries, 2,
                          activationEntriesWithTripState) ||
          activationEntriesWithTripState > limits.maximumCoverageStates) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = demandId;
        return result;
      }
      std::vector<std::uint16_t> activatedRectangles(
          activationEntriesWithTripState);
      // A ControlExit Set executes after every feasible alternative rejoins.
      // Completion may therefore flow through that cut only after the same
      // incoming capability has reached a qualifying source-pipeline point in
      // every alternative.  Treating the first reached alternative as enough
      // is an existential-path bug and is unsound for unconditional demands.
      std::map<std::pair<std::size_t, SyncCoverOrderingRequirementMask>,
               std::vector<std::uint8_t>>
          controlExitAlternatives;
      std::vector<std::uint8_t> mustExecute(graphInstances);
      for (unsigned copy = 0; copy <= demand.distance; ++copy) {
        for (const SyncCoverNode &node : graph.getNodes()) {
          if (!consumeWork(workBudget)) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            result.invalidIndex = demandId;
            return result;
          }
          mustExecute[static_cast<std::size_t>(copy) * graph.getNodes().size() +
                      node.id] =
              nodeInstanceMustExecute(graph, demand, context, node.id, copy,
                                      workBudget)
                  ? 1
                  : 0;
          if (workBudget && workBudget->exhausted) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            result.invalidIndex = demandId;
            return result;
          }
        }
      }
      struct State {
        std::size_t virtualNode = 0;
        SyncCoverOrderingRequirementMask capabilities = 0;
      };
      std::deque<State> pending;
      const auto addState = [&](std::size_t virtualNode,
                                SyncCoverOrderingRequirementMask capabilities) {
        if (virtualNode >= arena->getVirtualNodeCount() ||
            capabilities > kAllSyncCoverOrderingRequirements) {
          return;
        }
        const std::uint16_t bit =
            static_cast<std::uint16_t>(std::uint16_t{1} << capabilities);
        const bool newState = (reached[virtualNode] & bit) == 0;
        if (newState) {
          reached[virtualNode] |= bit;
          pending.push_back({virtualNode, capabilities});
        }
      };
      const auto chargeTransition = [&]() {
        if (coverageTransitions == limits.maximumCoverageTransitions ||
            !consumeWork(workBudget)) {
          return false;
        }
        ++coverageTransitions;
        return true;
      };
      const auto activateChannel = [&](const PreparedProtocolChannel &prepared,
                                       std::size_t sourceVirtual,
                                       SyncCoverOrderingRequirementMask
                                           incomingCapabilities) {
        const std::size_t protocolIndex = prepared.protocolIndex;
        const SyncCoverEventChannel &channel = *prepared.channel;
        const std::optional<SyncCoverNodeId> source =
            arena->getOperationForVirtualNode(sourceVirtual);
        const std::optional<unsigned> sourceCopy =
            arena->getCopyForVirtualNode(sourceVirtual);
        if (!source || !sourceCopy) {
          return true;
        }
        if (graph.getNodes()[*source].resource != channel.set.resource) {
          return true;
        }
        const SyncCoverEventProtocol &protocol = protocols[protocolIndex];
        const bool localProtocol =
            protocol.loop
                ? (demand.distance == 0
                       ? scopeContains(graph, protocol.loop->scope,
                                       graph.getNodes()[demand.source].scope,
                                       workBudget) &&
                             scopeContains(
                                 graph, protocol.loop->scope,
                                 graph.getNodes()[demand.target].scope,
                                 workBudget)
                       : demand.scope == protocol.loop->scope)
                : demand.distance == 0 && !demandSourceLoop &&
                      !demandTargetLoop;
        const bool nestedExport =
            protocol.loop && protocol.loop->scope != demand.scope &&
            scopeContains(graph, demand.scope, protocol.loop->scope,
                          workBudget);
        if (workBudget && workBudget->exhausted) {
          return false;
        }
        const bool sourceOutsideProtocolWorld = !localProtocol && !nestedExport;
        if (sourceOutsideProtocolWorld) {
          return true;
        }
        const SyncCoverOrderingRequirementMask transferred =
            incomingCapabilities == 0
                ? channel.suppliedRequirements
                : static_cast<SyncCoverOrderingRequirementMask>(
                      incomingCapabilities & channel.suppliedRequirements);
        if (transferred == 0) {
          return true;
        }
        const bool exitExport =
            channel.id < exitExportsByProtocol[protocolIndex].size() &&
            exitExportsByProtocol[protocolIndex][channel.id];
        const bool sourceProvesNonzero =
            protocol.loop &&
            nearestEnclosingLoop(graph, graph.getNodes()[*source].scope,
                                 workBudget) ==
                std::optional<SyncCoverScopeId>(protocol.loop->scope);
        if (workBudget && workBudget->exhausted) {
          return false;
        }
        for (const PreparedProtocolRectangle &rectangle : prepared.rectangles) {
          const SyncCoverEventTransfer &transfer = rectangle.transfer;
          const bool explicitParentSummary =
              rectangle.distanceScope &&
              *rectangle.distanceScope == demand.scope;
          if (nestedExport && demand.distance != 0 && !explicitParentSummary) {
            continue;
          }
          if (nestedExport && rectangle.distance != 0 &&
              rectangle.distanceScope &&
              *rectangle.distanceScope != demand.scope) {
            // A child-loop recurrence is local to one invocation.  It may
            // contribute to a parent arena only through a separately
            // verified child summary; its distance must never be
            // reinterpreted as a parent-loop distance.
            continue;
          }
          const bool distanceApplies =
              rectangle.distance != 0 &&
              (rectangle.distanceScope
                   ? *rectangle.distanceScope == demand.scope
                   : localProtocol);
          const unsigned appliedDistance =
              distanceApplies ? rectangle.distance : 0;
          const unsigned lastSetCopy =
              channel.actions.empty() ? demand.distance : *sourceCopy;
          for (unsigned setCopy = *sourceCopy; setCopy <= lastSetCopy;
               ++setCopy) {
            const bool sourceInPrefix = nodeInSourcePrefix(
                graph, protocol, target, transfer, demand, context, *source,
                *sourceCopy, setCopy, rectangle.setAction,
                rectangle.externalSource, &mustExecute, workBudget);
            if (workBudget && workBudget->exhausted) {
              return false;
            }
            if (!sourceInPrefix ||
                appliedDistance > demand.distance - setCopy) {
              continue;
            }
            const std::size_t activationEntry =
                (rectangle.activationId * distanceCount + setCopy) * 2 +
                (sourceProvesNonzero ? 1 : 0);
            if (transfer.set.anchor.kind == SyncCoverAnchorKind::ControlExit) {
              const SyncCoverControlId control = transfer.set.anchor.node;
              const std::optional<unsigned> alternative =
                  getNodeAlternative(graph, *source, control, workBudget);
              if (workBudget && workBudget->exhausted) {
                return false;
              }
              if (alternative) {
                if (control >= graph.getControls().size() ||
                    *alternative >= graph.getControls()[control].alternatives) {
                  return false;
                }
                auto &reachedAlternatives =
                    controlExitAlternatives[{activationEntry, transferred}];
                if (reachedAlternatives.empty()) {
                  reachedAlternatives.resize(
                      graph.getControls()[control].alternatives);
                }
                reachedAlternatives[*alternative] = 1;
                bool allRequiredAlternativesReached = true;
                for (unsigned candidate = 0;
                     candidate < reachedAlternatives.size(); ++candidate) {
                  const bool required = controlAlternativeIsRequired(
                      graph, demand, context, control, candidate, setCopy,
                      workBudget);
                  if (workBudget && workBudget->exhausted) {
                    return false;
                  }
                  allRequiredAlternativesReached &=
                      !required || reachedAlternatives[candidate] != 0;
                }
                if (!allRequiredAlternativesReached) {
                  continue;
                }
              }
            }
            const std::uint16_t activationBit =
                static_cast<std::uint16_t>(std::uint16_t{1} << transferred);
            if ((activatedRectangles[activationEntry] & activationBit) != 0) {
              continue;
            }
            activatedRectangles[activationEntry] |= activationBit;
            const auto targets = nodesByResource.find(transfer.wait.resource);
            if (targets == nodesByResource.end()) {
              continue;
            }
            const unsigned waitCopy = setCopy + appliedDistance;
            for (unsigned targetCopy = waitCopy; targetCopy <= demand.distance;
                 ++targetCopy) {
              for (SyncCoverNodeId targetNodeId : targets->second) {
                const SyncCoverNode &targetNode =
                    graph.getNodes()[targetNodeId];
                if (!chargeTransition()) {
                  return false;
                }
                const bool targetIsLocal = nodeOccursInProtocolIteration(
                    graph, protocol, targetNode.id, workBudget);
                if (workBudget && workBudget->exhausted) {
                  return false;
                }
                const bool targetOutsideProtocolWorld =
                    targetIsLocal ? (!localProtocol && !nestedExport)
                                  : !nestedExport;
                if (targetOutsideProtocolWorld) {
                  continue;
                }
                const bool inTargetSuffix = nodeInTargetSuffix(
                    graph, protocol, transfer, demand, context, targetNode.id,
                    targetCopy, waitCopy, exitExport, rectangle.waitAction,
                    sourceProvesNonzero, rectangle.loopSummary,
                    rectangle.externalTarget && rectangle.distanceScope &&
                        *rectangle.distanceScope == demand.scope,
                    &mustExecute, workBudget);
                if (workBudget && workBudget->exhausted) {
                  return false;
                }
                if (!inTargetSuffix) {
                  continue;
                }
                const std::optional<std::size_t> targetVirtual =
                    expansion.projectEndpoint(graph, *arena, targetNode.id,
                                              targetCopy);
                if (!targetVirtual) {
                  continue;
                }
                for (SyncCoverOrderingRequirementMask subset = transferred;
                     subset != 0;
                     subset = static_cast<SyncCoverOrderingRequirementMask>(
                         (subset - 1) & transferred)) {
                  addState(*targetVirtual, subset);
                }
              }
            }
          }
        }
        return !workBudget || !workBudget->exhausted;
      };

      const std::optional<std::size_t> sourceVirtual =
          expansion.projectEndpoint(graph, *arena, demand.source, 0);
      const std::optional<std::size_t> targetVirtual =
          expansion.projectEndpoint(graph, *arena, demand.target,
                                    demand.distance);
      if (!sourceVirtual || !targetVirtual) {
        result.error = SyncCoverProtocolError::InvalidGraph;
        result.invalidIndex = demandId;
        return result;
      }
      addState(*sourceVirtual, 0);
      while (!pending.empty()) {
        const State state = pending.front();
        pending.pop_front();
        for (const SyncCoverExpandedEdge &edge :
             arena->getOutgoingEdges(state.virtualNode)) {
          if (!chargeTransition()) {
            result.error = workBudget && workBudget->exhausted
                               ? SyncCoverProtocolError::WorkLimitExceeded
                               : SyncCoverProtocolError::LimitExceeded;
            result.invalidIndex = demandId;
            return result;
          }
          const auto virtualNodeActive = [&](std::size_t virtualNode) {
            const std::optional<SyncCoverNodeId> operation =
                arena->getOperationForVirtualNode(virtualNode);
            const std::optional<unsigned> copy =
                arena->getCopyForVirtualNode(virtualNode);
            return !operation || (copy && *copy <= demand.distance &&
                                  mustExecute[static_cast<std::size_t>(*copy) *
                                                  graph.getNodes().size() +
                                              *operation] != 0);
          };
          bool edgeActive = edge.targetCopy <= demand.distance &&
                            virtualNodeActive(edge.source) &&
                            virtualNodeActive(edge.target);
          if (edgeActive && edge.graphEdge) {
            edgeActive = sync_cover_detail::edgeGuardsActive(
                graph, demand, context, graph.getEdges()[*edge.graphEdge],
                edge.sourceCopy, edge.targetCopy, workBudget);
          }
          if (workBudget && workBudget->exhausted) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            result.invalidIndex = demandId;
            return result;
          }
          if (!edgeActive) {
            continue;
          }
          if (edge.kind == SyncCoverEdgeKind::CompletionSupply) {
            const SyncCoverOrderingRequirementMask supplied =
                edge.graphEdge
                    ? graph.getEdges()[*edge.graphEdge].suppliedRequirements
                    : syncCoverOrderingRequirementBit(
                          SyncCoverOrderingRequirement::
                              PipelineCompletionBeforeAccess);
            addState(edge.target, static_cast<SyncCoverOrderingRequirementMask>(
                                      state.capabilities | supplied));
          } else {
            const std::optional<std::uint32_t> sourceResource =
                arena->getResourceForVirtualNode(graph, edge.source);
            const std::optional<std::uint32_t> targetResource =
                arena->getResourceForVirtualNode(graph, edge.target);
            const std::uint32_t demandSourceResource =
                graph.getNodes()[demand.source].resource;
            const bool preservesIssuedPrefix =
                target.directEventCompletesSourcePrefix &&
                state.capabilities == 0 && sourceResource && targetResource &&
                *sourceResource == demandSourceResource &&
                *targetResource == demandSourceResource;
            // Capability zero is the distinguished "source has issued" state.
            // It may follow only same-resource issue order so a later Set can
            // certify the complete issued prefix.  It never satisfies a goal
            // by itself.  Once a completion capability is established, the
            // graph edge kinds preserve it exactly as in ordinary coverage.
            if (state.capabilities != 0 || preservesIssuedPrefix) {
              addState(edge.target, state.capabilities);
            }
          }
        }
        const std::optional<SyncCoverNodeId> stateOperation =
            arena->getOperationForVirtualNode(state.virtualNode);
        if (!stateOperation) {
          continue;
        }
        const auto channels = channelsBySourceResource.find(
            graph.getNodes()[*stateOperation].resource);
        if (channels == channelsBySourceResource.end()) {
          continue;
        }
        for (const PreparedProtocolChannel &prepared : channels->second) {
          if (!chargeTransition()) {
            result.error = workBudget && workBudget->exhausted
                               ? SyncCoverProtocolError::WorkLimitExceeded
                               : SyncCoverProtocolError::LimitExceeded;
            result.invalidIndex = demandId;
            return result;
          }
          if (!activateChannel(prepared, state.virtualNode,
                               state.capabilities)) {
            result.error = workBudget && workBudget->exhausted
                               ? SyncCoverProtocolError::WorkLimitExceeded
                               : SyncCoverProtocolError::LimitExceeded;
            result.invalidIndex = demandId;
            return result;
          }
        }
      }
      bool demandCovered = false;
      for (SyncCoverOrderingRequirementMask mask = 1;
           mask <= kAllSyncCoverOrderingRequirements; ++mask) {
        const std::uint16_t bit =
            static_cast<std::uint16_t>(std::uint16_t{1} << mask);
        demandCovered |=
            (reached[*targetVirtual] & bit) != 0 &&
            (mask & demand.orderingRequirements) == demand.orderingRequirements;
      }
      if (demandCovered) {
        covered.insert(demandId);
      }
    }
    if (workBudget && workBudget->exhausted) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = worldIndex;
      return result;
    }
    result.coveredByWorld.push_back(std::move(covered));
  }
  result.statistics.coverageTransitions = coverageTransitions;
  return result;
}

SyncCoverProtocolCoverageResult mlir::pto::computeSyncCoverProtocolExactWorlds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverExactWorld> &worlds,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  return computeProtocolWorlds(graph, target, protocols, worlds, limits,
                               workBudget, true);
}

SyncCoverProtocolCoverageResult
mlir::pto::computeSyncCoverProtocolExactWorldsForDemands(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverExactWorld> &worlds,
    const SyncCoverDemandSet &queriedDemands, SyncCoverProtocolLimits limits,
    SyncCoverCoverageWorkBudget *workBudget) {
  return computeProtocolWorlds(graph, target, protocols, worlds, limits,
                               workBudget, true, &queriedDemands);
}

SyncCoverProtocolCoverageResult mlir::pto::computeSyncCoverProtocolDirectWorlds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverExactWorld> &worlds,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  return computeProtocolWorlds(graph, target, protocols, worlds, limits,
                               workBudget, false);
}

SyncCoverProtocolCoverageResult
mlir::pto::computeSyncCoverProtocolConnectorClosure(
    const SyncCoverGraph &graph, const SyncCoverDemandSet &directCoverage,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  using ContextLiteral = sync_cover_detail::ContextLiteral;
  SyncCoverProtocolCoverageResult result;
  if (!graph.isStructureFrozen() ||
      directCoverage.size() != graph.getDemands().size()) {
    result.error = SyncCoverProtocolError::InvalidGraph;
    return result;
  }
  if (!consumeWork(workBudget, graph.getNodes().size() +
                                   directCoverage.getWords().size() * 2)) {
    result.error = SyncCoverProtocolError::WorkLimitExceeded;
    return result;
  }

  std::vector<std::vector<SyncCoverDemandId>> outgoing(graph.getNodes().size());
  std::size_t connectorIncidences = 0;
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = demandId;
      return result;
    }
    if (!directCoverage.contains(demandId)) {
      continue;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    if (!checkedAccumulate(connectorIncidences, 1) ||
        connectorIncidences > limits.maximumLifecycleConnectorIncidences ||
        !consumeWork(workBudget)) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = demandId;
      return result;
    }
    outgoing[demand.source].push_back(demandId);
  }

  const auto appendAbsoluteGuard = [&](const SyncCoverDemand &goal,
                                       const SyncCoverGuard &guard,
                                       unsigned copy,
                                       std::vector<ContextLiteral> &condition) {
    for (const SyncCoverGuardLiteral &literal : guard.literals) {
      if (!consumeWork(workBudget)) {
        return false;
      }
      const ContextLiteral item{
          literal.control,
          sync_cover_detail::contextCopy(graph, goal, literal.control, copy,
                                         workBudget),
          literal.alternative};
      if (!consumeWork(workBudget, condition.size() + 1)) {
        return false;
      }
      auto position =
          std::lower_bound(condition.begin(), condition.end(), item);
      if (position != condition.end() && position->control == item.control &&
          position->copy == item.copy) {
        if (position->alternative != item.alternative) {
          return false;
        }
        continue;
      }
      if (condition.size() == limits.maximumGuardLiterals) {
        return false;
      }
      condition.insert(position, item);
    }
    return true;
  };

  struct ConnectorState {
    SyncCoverNodeId node = 0;
    unsigned copy = 0;
    unsigned depth = 0;
    std::size_t condition = 0;

    bool operator<(const ConnectorState &other) const {
      return std::tie(node, copy, condition) <
             std::tie(other.node, other.copy, other.condition);
    }
  };

  SyncCoverDemandSet closure = directCoverage;
  std::size_t transitions = 0;
  std::size_t retainedConditionLiterals = 0;
  std::map<std::vector<ContextLiteral>, std::size_t> conditionIds;
  std::vector<const std::vector<ContextLiteral> *> conditions;
  const auto internCondition =
      [&](std::vector<ContextLiteral> condition) -> std::optional<std::size_t> {
    std::size_t comparisonUnits = 0;
    if (!checkedProduct(condition.size(), conditionIds.size() + 1,
                        comparisonUnits) ||
        !consumeWork(workBudget, comparisonUnits + 1)) {
      return std::nullopt;
    }
    const auto existing = conditionIds.find(condition);
    if (existing != conditionIds.end()) {
      return existing->second;
    }
    if (condition.size() >
        limits.maximumLifecycleConnectorGuardIncidences -
            std::min(retainedConditionLiterals,
                     limits.maximumLifecycleConnectorGuardIncidences)) {
      return std::nullopt;
    }
    const std::size_t id = conditions.size();
    auto inserted = conditionIds.emplace(std::move(condition), id);
    retainedConditionLiterals += inserted.first->first.size();
    conditions.push_back(&inserted.first->first);
    return id;
  };
  for (SyncCoverDemandId goalId = 0; goalId < graph.getDemands().size();
       ++goalId) {
    if (closure.contains(goalId)) {
      continue;
    }
    const SyncCoverDemand &goal = graph.getDemands()[goalId];
    if (goal.storageWitnesses.empty()) {
      continue;
    }
    const auto sharesGoalStorage = [&](const SyncCoverDemand &edge) {
      for (SyncCoverStorageWitnessId goalWitnessId : goal.storageWitnesses) {
        if (!consumeWork(workBudget) ||
            goalWitnessId >= graph.getStorageWitnesses().size()) {
          return false;
        }
        const SyncCoverStorageWitness &goalWitness =
            graph.getStorageWitnesses()[goalWitnessId];
        if (goalWitness.sourceAccess >= graph.getStorageAccesses().size()) {
          return false;
        }
        const SyncCoverStorageAccess &goalAccess =
            graph.getStorageAccesses()[goalWitness.sourceAccess];
        for (SyncCoverStorageWitnessId edgeWitnessId : edge.storageWitnesses) {
          if (!consumeWork(workBudget) ||
              edgeWitnessId >= graph.getStorageWitnesses().size()) {
            return false;
          }
          const SyncCoverStorageWitness &edgeWitness =
              graph.getStorageWitnesses()[edgeWitnessId];
          if (edgeWitness.sourceAccess >= graph.getStorageAccesses().size()) {
            return false;
          }
          const SyncCoverStorageAccess &edgeAccess =
              graph.getStorageAccesses()[edgeWitness.sourceAccess];
          if (goalAccess.domain == edgeAccess.domain &&
              goalAccess.extent.begin < edgeAccess.extent.end &&
              edgeAccess.extent.begin < goalAccess.extent.end) {
            return true;
          }
        }
      }
      return false;
    };
    const sync_cover_detail::DemandContext base =
        sync_cover_detail::makeDemandContext(graph, goal, workBudget);
    if (!base.valid) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::InvalidGraph;
      result.invalidIndex = goalId;
      return result;
    }

    std::deque<ConnectorState> pending;
    std::set<ConnectorState> reached;
    const std::optional<std::size_t> initialCondition =
        internCondition(base.condition);
    if (!initialCondition) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = goalId;
      return result;
    }
    ConnectorState initial{goal.source, 0, 0, *initialCondition};
    if (!consumeWork(workBudget, logarithmicLookupWork(reached.size() + 1))) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = goalId;
      return result;
    }
    reached.insert(initial);
    pending.push_back(std::move(initial));
    std::vector<std::size_t> reachingConditions;
    while (!pending.empty()) {
      ConnectorState state = std::move(pending.front());
      pending.pop_front();
      if (state.node == goal.target && state.copy == goal.distance) {
        reachingConditions.push_back(state.condition);
        continue;
      }
      constexpr unsigned kMaximumConnectorDepth = 8;
      if (state.depth == kMaximumConnectorDepth) {
        continue;
      }
      for (SyncCoverDemandId edgeId : outgoing[state.node]) {
        if (transitions == limits.maximumCoverageTransitions ||
            !consumeWork(workBudget)) {
          result.error = workBudget && workBudget->exhausted
                             ? SyncCoverProtocolError::WorkLimitExceeded
                             : SyncCoverProtocolError::LimitExceeded;
          result.invalidIndex = goalId;
          return result;
        }
        ++transitions;
        const SyncCoverDemand &edge = graph.getDemands()[edgeId];
        if ((edge.orderingRequirements & goal.orderingRequirements) !=
                goal.orderingRequirements ||
            edge.scope != goal.scope || !sharesGoalStorage(edge)) {
          if (workBudget && workBudget->exhausted) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            result.invalidIndex = goalId;
            return result;
          }
          continue;
        }
        const unsigned displacement = edge.distance;
        if (displacement > goal.distance - state.copy) {
          continue;
        }
        const unsigned targetCopy = state.copy + displacement;
        const std::vector<ContextLiteral> &stateCondition =
            *conditions[state.condition];
        if (!consumeWork(workBudget, stateCondition.size())) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          result.invalidIndex = goalId;
          return result;
        }
        std::vector<ContextLiteral> condition = stateCondition;
        const unsigned edgeTargetCopy = targetCopy;
        if (!appendAbsoluteGuard(goal, edge.sourceGuard, state.copy,
                                 condition) ||
            !appendAbsoluteGuard(goal, edge.targetGuard, edgeTargetCopy,
                                 condition)) {
          if (workBudget && workBudget->exhausted) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            result.invalidIndex = goalId;
            return result;
          }
          continue;
        }
        const std::optional<std::size_t> conditionId =
            internCondition(std::move(condition));
        if (!conditionId) {
          result.error = workBudget && workBudget->exhausted
                             ? SyncCoverProtocolError::WorkLimitExceeded
                             : SyncCoverProtocolError::LimitExceeded;
          result.invalidIndex = goalId;
          return result;
        }
        ConnectorState next{edge.target, targetCopy, state.depth + 1,
                            *conditionId};
        if (!consumeWork(workBudget,
                         logarithmicLookupWork(reached.size() + 1))) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          result.invalidIndex = goalId;
          return result;
        }
        const auto insertionPoint = reached.lower_bound(next);
        const bool isNewState =
            insertionPoint == reached.end() || next < *insertionPoint;
        if (isNewState) {
          if (reached.size() == limits.maximumCoverageStates) {
            result.error = SyncCoverProtocolError::LimitExceeded;
            result.invalidIndex = goalId;
            return result;
          }
          reached.emplace_hint(insertionPoint, next);
          pending.push_back(std::move(next));
        }
      }
    }
    if (reachingConditions.empty()) {
      continue;
    }

    using ControlOccurrence = std::pair<SyncCoverControlId, unsigned>;
    std::vector<ControlOccurrence> variables;
    for (std::size_t conditionId : reachingConditions) {
      const std::vector<ContextLiteral> &condition = *conditions[conditionId];
      for (const ContextLiteral &literal : condition) {
        if (!consumeWork(workBudget,
                         base.condition.size() + variables.size() + 1)) {
          result.error = SyncCoverProtocolError::WorkLimitExceeded;
          result.invalidIndex = goalId;
          return result;
        }
        const bool fixed =
            std::any_of(base.condition.begin(), base.condition.end(),
                        [&](const ContextLiteral &candidate) {
                          return candidate.control == literal.control &&
                                 candidate.copy == literal.copy;
                        });
        const ControlOccurrence occurrence{literal.control, literal.copy};
        if (!fixed && !std::binary_search(variables.begin(), variables.end(),
                                          occurrence)) {
          variables.insert(
              std::lower_bound(variables.begin(), variables.end(), occurrence),
              occurrence);
        }
      }
    }

    std::vector<std::vector<unsigned>> alternativesByVariable;
    alternativesByVariable.reserve(variables.size());
    std::size_t invocationSequences = 1;
    std::size_t alternativeIncidences = 0;
    bool truncated = false;
    for (const ControlOccurrence &occurrence : variables) {
      if (occurrence.first >= graph.getControls().size()) {
        truncated = true;
        break;
      }
      const SyncCoverControl &control = graph.getControls()[occurrence.first];
      std::optional<unsigned> fixedAlternative;
      if (control.phaseRelation &&
          occurrence.second != sync_cover_detail::kStaticControlCopy) {
        std::size_t phase = control.phaseRelation->initialPhase;
        for (unsigned step = 0; step < occurrence.second; ++step) {
          if (phase >= control.phaseRelation->nextPhase.size()) {
            truncated = true;
            break;
          }
          phase = control.phaseRelation->nextPhase[phase];
        }
        if (truncated ||
            phase >= control.phaseRelation->activeAlternative.size()) {
          truncated = true;
          break;
        }
        fixedAlternative = control.phaseRelation->activeAlternative[phase];
      }
      const std::size_t alternativeCount =
          fixedAlternative ? 1 : control.alternatives;
      std::size_t nextCount = 0;
      if (alternativeCount == 0 ||
          !checkedProduct(invocationSequences, alternativeCount, nextCount) ||
          nextCount > limits.maximumInvocationSequences ||
          !checkedAdd(alternativeIncidences, alternativeCount,
                      alternativeIncidences) ||
          alternativeIncidences > limits.maximumPhaseIncidences ||
          !consumeWork(workBudget, alternativeCount + 1)) {
        truncated = true;
        break;
      }
      invocationSequences = nextCount;
      std::vector<unsigned> alternatives;
      alternatives.reserve(alternativeCount);
      if (fixedAlternative) {
        alternatives.push_back(*fixedAlternative);
      } else {
        for (unsigned alternative = 0; alternative < control.alternatives;
             ++alternative) {
          alternatives.push_back(alternative);
        }
      }
      alternativesByVariable.push_back(std::move(alternatives));
    }
    if (truncated) {
      if (workBudget && workBudget->exhausted) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = goalId;
        return result;
      }
      continue;
    }

    std::vector<ContextLiteral> assignment;
    if (variables.size() >
            limits.maximumGuardLiterals -
                std::min(base.condition.size(), limits.maximumGuardLiterals) ||
        !consumeWork(workBudget, base.condition.size() + variables.size())) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = goalId;
      return result;
    }
    assignment.reserve(base.condition.size() + variables.size());
    bool allAssignmentsCovered = true;
    bool assignmentWorkUnavailable = false;
    std::size_t assignmentsVisited = 0;
    std::vector<std::size_t> alternativeOrdinals(variables.size());
    while (assignmentsVisited < invocationSequences && allAssignmentsCovered &&
           !assignmentWorkUnavailable) {
      assignment.assign(base.condition.begin(), base.condition.end());
      for (std::size_t variable = 0; variable < variables.size(); ++variable) {
        if (!consumeWork(workBudget, assignment.size() + 1)) {
          assignmentWorkUnavailable = true;
          break;
        }
        const ControlOccurrence occurrence = variables[variable];
        const ContextLiteral literal{
            occurrence.first, occurrence.second,
            alternativesByVariable[variable][alternativeOrdinals[variable]]};
        assignment.insert(
            std::lower_bound(assignment.begin(), assignment.end(), literal),
            literal);
      }
      if (assignmentWorkUnavailable) {
        break;
      }
      ++assignmentsVisited;
      bool covered = false;
      for (std::size_t conditionId : reachingConditions) {
        const std::vector<ContextLiteral> &condition = *conditions[conditionId];
        if (!consumeWork(workBudget,
                         assignment.size() + condition.size() + 1)) {
          assignmentWorkUnavailable = true;
          break;
        }
        if (std::includes(assignment.begin(), assignment.end(),
                          condition.begin(), condition.end())) {
          covered = true;
          break;
        }
      }
      allAssignmentsCovered = covered;
      for (std::size_t variable = variables.size(); variable != 0;) {
        --variable;
        if (++alternativeOrdinals[variable] <
            alternativesByVariable[variable].size()) {
          break;
        }
        alternativeOrdinals[variable] = 0;
      }
    }
    if (assignmentWorkUnavailable ||
        assignmentsVisited > limits.maximumInvocationSequences) {
      result.error = assignmentWorkUnavailable
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = goalId;
      return result;
    }
    if (allAssignmentsCovered) {
      closure.insert(goalId);
    }
  }
  result.statistics.coverageTransitions = transitions;
  result.coveredByWorld.push_back(std::move(closure));
  return result;
}

SyncCoverProtocolAllocationResult mlir::pto::allocateSyncCoverProtocolEventIds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverProtocolEventReservation> &reservations,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  SyncCoverProtocolAllocationResult result;
  const bool graphLimitExceeded = !graphFitsProtocolLimits(graph, limits) ||
                                  !targetFitsProtocolLimits(target, limits);
  if (graphLimitExceeded) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  const bool invalidGraph = !graph.isStructureFrozen() || !graph.validate();
  if (invalidGraph) {
    result.error = SyncCoverProtocolError::InvalidGraph;
    return result;
  }
  const SyncCoverProtocolError targetError =
      validateProtocolTargetContract(target, limits, workBudget);
  if (targetError != SyncCoverProtocolError::None) {
    result.error = targetError;
    return result;
  }
  const bool catalogLimitExceeded =
      protocols.size() > limits.maximumProtocols ||
      reservations.size() > limits.maximumReservations;
  if (catalogLimitExceeded) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }

  using Domain = std::pair<std::uint32_t, std::uint32_t>;
  std::map<Domain, std::set<unsigned>> unavailable;
  std::size_t reservationIdIncidences = 0;
  for (std::size_t index = 0; index < reservations.size(); ++index) {
    const SyncCoverProtocolEventReservation &reservation = reservations[index];
    const bool incidenceLimitExceeded =
        !checkedAccumulate(reservationIdIncidences,
                           reservation.eventIds.size()) ||
        reservationIdIncidences > limits.maximumReservationIdIncidences;
    if (incidenceLimitExceeded) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    std::size_t workPerId = 0;
    std::size_t idWork = 0;
    std::size_t reservationWork = 0;
    const std::size_t domainLookupWork =
        logarithmicLookupWork(unavailable.size() + 1);
    const bool reservationWorkOverflow =
        !checkedAdd(logarithmicLookupWork(target.compilerUsableEventIds.size()),
                    logarithmicLookupWork(reservation.eventIds.size() + 1),
                    workPerId) ||
        !checkedAdd(workPerId, 3, workPerId) ||
        !checkedProduct(reservation.eventIds.size(), workPerId, idWork) ||
        !checkedAdd(domainLookupWork, domainLookupWork, reservationWork) ||
        !checkedAdd(reservationWork,
                    logarithmicLookupWork(target.eventCapabilities.size()),
                    reservationWork) ||
        !checkedAdd(reservationWork, idWork, reservationWork) ||
        !consumeWork(workBudget, reservationWork);
    if (reservationWorkOverflow) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    const Domain domain{reservation.sourceResource, reservation.targetResource};
    const bool duplicateDomain = unavailable.count(domain) != 0;
    const bool supportedDomain =
        supportsEventDomain(target, domain.first, domain.second);
    if (duplicateDomain || !supportedDomain ||
        !std::is_sorted(reservation.eventIds.begin(),
                        reservation.eventIds.end()) ||
        std::adjacent_find(reservation.eventIds.begin(),
                           reservation.eventIds.end()) !=
            reservation.eventIds.end() ||
        std::any_of(reservation.eventIds.begin(), reservation.eventIds.end(),
                    [&](unsigned id) {
                      return !std::binary_search(
                          target.compilerUsableEventIds.begin(),
                          target.compilerUsableEventIds.end(), id);
                    })) {
      result.error = SyncCoverProtocolError::InvalidProtocol;
      result.invalidIndex = index;
      return result;
    }
    const bool domainLimitReached =
        unavailable.size() == limits.maximumReservationDomains;
    if (domainLimitReached) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    unavailable[domain].insert(reservation.eventIds.begin(),
                               reservation.eventIds.end());
  }

  struct Request {
    Domain domain;
    SyncCoverMechanismId mechanism = 0;
    SyncCoverProtocolChannelId channel = 0;
    std::size_t width = 0;
    std::size_t protocolIndex = 0;
  };
  std::vector<Request> requests;
  std::set<SyncCoverMechanismId> mechanisms;
  std::size_t requestedIds = 0;
  std::size_t verifiedActions = 0;
  std::size_t verifiedAutomatonEdges = 0;
  std::size_t verifiedQueries = 0;
  std::size_t verifiedRearmLookupWork = 0;
  std::size_t verifiedReachabilityWork = 0;
  std::size_t verifiedProofLaneIncidences = 0;
  std::size_t verifiedLaneInitializationWork = 0;
  std::size_t verifiedExitExports = 0;
  std::size_t verifiedExitExportGuardLiterals = 0;
  std::size_t rearmProofs = 0;
  std::size_t channelIncidences = 0;
  for (const SyncCoverEventProtocol &protocol : protocols) {
    const bool channelLimitExceeded =
        !checkedAccumulate(channelIncidences, protocol.channels.size()) ||
        channelIncidences > limits.maximumChannelRequests;
    if (channelLimitExceeded) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      return result;
    }
  }
  std::size_t catalogWork = 0;
  const bool catalogWorkOverflow =
      !checkedProduct(protocols.size(),
                      logarithmicLookupWork(protocols.size()) + 1,
                      catalogWork) ||
      !checkedAdd(catalogWork, channelIncidences, catalogWork) ||
      !consumeWork(workBudget, catalogWork);
  if (catalogWorkOverflow) {
    result.error = workBudget && workBudget->exhausted
                       ? SyncCoverProtocolError::WorkLimitExceeded
                       : SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  requests.reserve(channelIncidences);
  for (std::size_t protocolIndex = 0; protocolIndex < protocols.size();
       ++protocolIndex) {
    const SyncCoverEventProtocol &protocol = protocols[protocolIndex];
    const bool proofLimitExceeded =
        !checkedAccumulate(rearmProofs, protocol.rearmProofs.size()) ||
        rearmProofs > limits.maximumRearmProofs;
    if (proofLimitExceeded) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = protocolIndex;
      return result;
    }
    if (!mechanisms.insert(protocol.mechanism).second) {
      result.error = SyncCoverProtocolError::InvalidProtocol;
      result.invalidIndex = protocolIndex;
      return result;
    }
    SyncCoverProtocolLimits remaining = limits;
    remaining.maximumTotalDynamicActions -= verifiedActions;
    remaining.maximumAutomatonEdges -= verifiedAutomatonEdges;
    remaining.maximumRearmQueries -= verifiedQueries;
    remaining.maximumRearmLookupWork -= verifiedRearmLookupWork;
    remaining.maximumReachabilityWork -= verifiedReachabilityWork;
    remaining.maximumRearmProofLaneIncidences -= verifiedProofLaneIncidences;
    remaining.maximumLaneInitializationWork -= verifiedLaneInitializationWork;
    remaining.maximumExitExports -= verifiedExitExports;
    remaining.maximumExitExportGuardLiterals -= verifiedExitExportGuardLiterals;
    const SyncCoverProtocolVerificationResult verification =
        verifyProtocolAssumingValidGraph(graph, target, protocol, remaining,
                                         workBudget, false);
    if (!verification) {
      result.error = verification.error;
      result.invalidIndex = protocolIndex;
      return result;
    }
    if (!consumeWork(workBudget, verification.exitExports.size())) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = protocolIndex;
      return result;
    }
    if (!checkedAccumulate(verifiedActions,
                           verification.statistics.totalDynamicActions) ||
        !checkedAccumulate(verifiedAutomatonEdges,
                           verification.statistics.automatonEdges) ||
        !checkedAccumulate(verifiedQueries,
                           verification.statistics.rearmQueries) ||
        !checkedAccumulate(verifiedRearmLookupWork,
                           verification.statistics.rearmLookupWork) ||
        !checkedAccumulate(verifiedProofLaneIncidences,
                           verification.statistics.rearmProofLaneIncidences) ||
        !checkedAccumulate(verifiedLaneInitializationWork,
                           verification.statistics.laneInitializationWork) ||
        !checkedAccumulate(verifiedReachabilityWork,
                           verification.statistics.reachabilityWork)) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = protocolIndex;
      return result;
    }
    for (const SyncCoverProtocolExitExport &candidate :
         verification.exitExports) {
      const bool exportOverflow =
          !checkedAccumulate(verifiedExitExports, 1) ||
          !checkedAccumulate(verifiedExitExportGuardLiterals,
                             candidate.guard.literals.size());
      if (exportOverflow) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = protocolIndex;
        return result;
      }
    }
    for (const SyncCoverEventChannel &channel : protocol.channels) {
      const Domain domain{channel.set.resource, channel.wait.resource};
      const bool requestLimitExceeded =
          requests.size() == limits.maximumChannelRequests ||
          !checkedAccumulate(requestedIds, channel.width) ||
          requestedIds > limits.maximumAllocatedEventIds;
      if (requestLimitExceeded) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = protocolIndex;
        return result;
      }
      requests.push_back({domain, protocol.mechanism, channel.id, channel.width,
                          protocolIndex});
    }
  }
  std::size_t logarithm = 0;
  for (std::size_t value = requests.size(); value > 1;
       value = (value + 1) / 2) {
    ++logarithm;
  }
  std::size_t sortWork = 0;
  std::size_t sortWorkPerRequest = 0;
  const bool sortFailed =
      !checkedAdd(logarithm, 1, sortWorkPerRequest) ||
      !checkedProduct(requests.size(), sortWorkPerRequest, sortWork) ||
      !checkedProduct(sortWork, 3, sortWork) ||
      !consumeWork(workBudget, sortWork);
  if (sortFailed) {
    result.error = workBudget && workBudget->exhausted
                       ? SyncCoverProtocolError::WorkLimitExceeded
                       : SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  std::sort(requests.begin(), requests.end(),
            [](const Request &left, const Request &right) {
              return std::tie(left.domain, left.mechanism, left.channel) <
                     std::tie(right.domain, right.mechanism, right.channel);
            });
  const bool duplicateRequest =
      std::adjacent_find(requests.begin(), requests.end(),
                         [](const Request &left, const Request &right) {
                           return left.mechanism == right.mechanism &&
                                  left.channel == right.channel;
                         }) != requests.end();
  if (duplicateRequest) {
    result.error = SyncCoverProtocolError::InvalidProtocol;
    return result;
  }

  result.channels.reserve(requests.size());
  for (const Request &request : requests) {
    const std::size_t domainLookupWork =
        logarithmicLookupWork(unavailable.size() + 1);
    std::size_t unavailableEntries = 0;
    const bool unavailableOverflow =
        !checkedAdd(limits.maximumReservationIdIncidences,
                    limits.maximumAllocatedEventIds, unavailableEntries) ||
        !checkedAdd(unavailableEntries, 1, unavailableEntries);
    const std::size_t eventLookupWork =
        unavailableOverflow ? 0 : logarithmicLookupWork(unavailableEntries);
    std::size_t workPerEventId = 0;
    std::size_t eventIdWork = 0;
    std::size_t allocationWork = 0;
    const bool allocationWorkOverflow =
        unavailableOverflow ||
        !checkedAdd(domainLookupWork, domainLookupWork, workPerEventId) ||
        !checkedAdd(workPerEventId, eventLookupWork, workPerEventId) ||
        !checkedAdd(workPerEventId, eventLookupWork, workPerEventId) ||
        !checkedAdd(workPerEventId, 1, workPerEventId) ||
        !checkedProduct(target.compilerUsableEventIds.size(), workPerEventId,
                        eventIdWork) ||
        !checkedAdd(domainLookupWork, eventIdWork, allocationWork) ||
        !consumeWork(workBudget, allocationWork);
    if (allocationWorkOverflow) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = request.protocolIndex;
      return result;
    }
    const bool domainLimitReached =
        unavailable.count(request.domain) == 0 &&
        unavailable.size() == limits.maximumReservationDomains;
    if (domainLimitReached) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = request.protocolIndex;
      return result;
    }
    SyncCoverProtocolChannelAllocation allocation;
    allocation.mechanism = request.mechanism;
    allocation.channel = request.channel;
    for (unsigned id : target.compilerUsableEventIds) {
      const bool available = unavailable[request.domain].count(id) == 0;
      if (available) {
        allocation.eventIds.push_back(id);
        unavailable[request.domain].insert(id);
        const bool complete = allocation.eventIds.size() == request.width;
        if (complete) {
          break;
        }
      }
    }
    const bool scarce = allocation.eventIds.size() != request.width;
    if (scarce) {
      result.error = SyncCoverProtocolError::ResourceInfeasible;
      result.invalidIndex = request.protocolIndex;
      return result;
    }
    result.channels.push_back(std::move(allocation));
  }
  return result;
}
