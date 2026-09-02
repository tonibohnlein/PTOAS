// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "SyncCoverProtocolInternal.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::sync_cover_protocol_detail;

namespace {

constexpr unsigned kMaximumCompilerEventId = 5;

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

bool reserveSortWork(std::size_t size, SyncCoverCoverageWorkBudget *budget) {
  std::size_t logarithm = 0;
  for (std::size_t value = size; value > 1; value = (value + 1) / 2) {
    ++logarithm;
  }
  std::size_t perElement = 0;
  std::size_t work = 0;
  return checkedAdd(logarithm, 1, perElement) &&
         checkedProduct(size, perElement, work) &&
         checkedProduct(work, 3, work) && consumeWork(budget, work);
}

std::size_t logarithmicLookupWork(std::size_t size) {
  std::size_t work = 1;
  for (std::size_t value = size; value > 1; value = (value + 1) / 2) {
    ++work;
  }
  return work;
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

bool normalizeGuard(SyncCoverGuard &guard,
                    SyncCoverCoverageWorkBudget *workBudget) {
  const bool unavailable =
      !reserveSortWork(guard.literals.size(), workBudget) ||
      !consumeWork(workBudget, guard.literals.size());
  if (unavailable) {
    return false;
  }
  return normalizeSyncCoverGuard(guard);
}

bool guardImplies(const SyncCoverGuard &condition,
                  const SyncCoverGuard &required,
                  SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t literalWork = 0;
  const bool workOverflow = !checkedAdd(condition.literals.size(),
                                        required.literals.size(), literalWork);
  const bool unavailable =
      workOverflow || !consumeWork(workBudget, literalWork) ||
      !reserveSortWork(condition.literals.size(), workBudget) ||
      !reserveSortWork(required.literals.size(), workBudget) ||
      !consumeWork(workBudget, literalWork);
  if (unavailable) {
    return false;
  }
  return syncCoverGuardImplies(condition, required);
}

auto anchorKey(const SyncCoverAnchor &anchor) {
  return std::tie(anchor.kind, anchor.node, anchor.scope, anchor.position);
}

bool anchorsEqual(const SyncCoverAnchor &left, const SyncCoverAnchor &right) {
  return anchorKey(left) == anchorKey(right);
}

bool guardsEqual(const SyncCoverGuard &left, const SyncCoverGuard &right) {
  return left.literals == right.literals;
}

bool guardsLess(const SyncCoverGuard &left, const SyncCoverGuard &right) {
  return left.literals < right.literals;
}

SyncCoverProtocolError
buildBodyGuardWorlds(const SyncCoverGraph &graph,
                     const SyncCoverEventProtocol &protocol,
                     ResolvedProtocol &resolved, SyncCoverProtocolLimits limits,
                     SyncCoverCoverageWorkBudget *workBudget) {
  resolved.bodyGuardWorlds = {SyncCoverGuard{}};
  if (!protocol.loop) {
    return SyncCoverProtocolError::None;
  }
  for (const ResolvedChannel &channel : resolved.channels) {
    for (const ResolvedAction &action : channel.actions) {
      if (action.description.segment != SyncCoverProtocolActionSegment::Body) {
        continue;
      }
      SyncCoverGuard residual = action.guard;
      residual.literals.erase(
          std::remove_if(
              residual.literals.begin(), residual.literals.end(),
              [&](const SyncCoverGuardLiteral &literal) {
                const bool fixedByLoop = std::binary_search(
                    resolved.loopGuard.literals.begin(),
                    resolved.loopGuard.literals.end(), literal);
                const SyncCoverControl &control =
                    graph.getControls()[literal.control];
                const bool fixedByFirstIteration =
                    control.firstIterationRelation &&
                    control.firstIterationRelation->loopScope ==
                        protocol.loop->scope &&
                    ((action.description.guard ==
                          SyncCoverProtocolActionGuard::FirstIteration &&
                      literal.alternative == control.firstIterationRelation
                                                 ->firstIterationAlternative) ||
                     (action.description.guard ==
                          SyncCoverProtocolActionGuard::NotFirstIteration &&
                      literal.alternative != control.firstIterationRelation
                                                 ->firstIterationAlternative));
                return fixedByLoop ||
                       (protocol.loop->phaseControl &&
                        literal.control == *protocol.loop->phaseControl) ||
                       fixedByFirstIteration;
              }),
          residual.literals.end());
      if (residual.literals.empty()) {
        continue;
      }
      // Periodic schedules already have an authoritative state transition.
      // Mixing an unrelated data-dependent guard into that transition needs a
      // product automaton and remains fail-closed for now.
      if (protocol.loop->phaseControl) {
        return SyncCoverProtocolError::InvalidProtocol;
      }
      const std::size_t existingWorlds = resolved.bodyGuardWorlds.size();
      for (std::size_t worldIndex = 0; worldIndex < existingWorlds;
           ++worldIndex) {
        const SyncCoverGuard &world = resolved.bodyGuardWorlds[worldIndex];
        if (!consumeWork(workBudget, world.literals.size() +
                                         residual.literals.size() + 1)) {
          return SyncCoverProtocolError::WorkLimitExceeded;
        }
        if (!syncCoverGuardsCompatible(world, residual)) {
          continue;
        }
        SyncCoverGuard combined = world;
        combined.literals.insert(combined.literals.end(),
                                 residual.literals.begin(),
                                 residual.literals.end());
        if (!normalizeGuard(combined, workBudget)) {
          return workBudget && workBudget->exhausted
                     ? SyncCoverProtocolError::WorkLimitExceeded
                     : SyncCoverProtocolError::InvalidProtocol;
        }
        const bool duplicate = std::any_of(resolved.bodyGuardWorlds.begin(),
                                           resolved.bodyGuardWorlds.end(),
                                           [&](const SyncCoverGuard &item) {
                                             return guardsEqual(item, combined);
                                           });
        if (duplicate) {
          continue;
        }
        if (resolved.bodyGuardWorlds.size() == limits.maximumWorlds) {
          return SyncCoverProtocolError::LimitExceeded;
        }
        resolved.bodyGuardWorlds.push_back(std::move(combined));
      }
    }
  }
  return SyncCoverProtocolError::None;
}

bool validTargetContract(const SyncCoverProtocolTargetContract &target) {
  const bool invalidCapabilities =
      target.eventCapabilities.empty() ||
      !std::is_sorted(target.eventCapabilities.begin(),
                      target.eventCapabilities.end()) ||
      std::adjacent_find(target.eventCapabilities.begin(),
                         target.eventCapabilities.end()) !=
          target.eventCapabilities.end() ||
      std::any_of(target.eventCapabilities.begin(),
                  target.eventCapabilities.end(), [](const auto &capability) {
                    return capability.sourceResource ==
                               capability.targetResource ||
                           capability.suppliedRequirements == 0 ||
                           (capability.suppliedRequirements &
                            ~kAllSyncCoverOrderingRequirements) != 0;
                  });
  const bool invalidIds =
      target.compilerUsableEventIds.empty() ||
      !std::is_sorted(target.compilerUsableEventIds.begin(),
                      target.compilerUsableEventIds.end()) ||
      std::adjacent_find(target.compilerUsableEventIds.begin(),
                         target.compilerUsableEventIds.end()) !=
          target.compilerUsableEventIds.end() ||
      target.compilerUsableEventIds.back() > kMaximumCompilerEventId;
  const bool invalidRearmFacts =
      !std::is_sorted(target.certifiedRearmFacts.begin(),
                      target.certifiedRearmFacts.end()) ||
      std::adjacent_find(target.certifiedRearmFacts.begin(),
                         target.certifiedRearmFacts.end(),
                         [](const auto &left, const auto &right) {
                           return left.evidence == right.evidence;
                         }) != target.certifiedRearmFacts.end() ||
      std::any_of(target.certifiedRearmFacts.begin(),
                  target.certifiedRearmFacts.end(), [](const auto &fact) {
                    return fact.evidence == 0 || fact.loopScope == 0 ||
                           fact.iterationDistance == 0 || fact.width == 0;
                  });
  return !invalidCapabilities && !invalidIds && !invalidRearmFacts;
}

bool validGuardForScope(const SyncCoverGraph &graph,
                        const SyncCoverGuard &guard,
                        SyncCoverScopeId occurrenceScope,
                        SyncCoverCoverageWorkBudget *workBudget) {
  if (!consumeWork(workBudget, guard.literals.size())) {
    return false;
  }
  SyncCoverGuard normalized = guard;
  if (!normalizeGuard(normalized, workBudget)) {
    return false;
  }
  for (const SyncCoverGuardLiteral &literal : normalized.literals) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    const bool invalidLiteral =
        literal.control >= graph.getControls().size() ||
        literal.alternative >=
            graph.getControls()[literal.control].alternatives;
    if (invalidLiteral) {
      return false;
    }
    if (!scopeContains(graph, graph.getControls()[literal.control].scope,
                       occurrenceScope, workBudget)) {
      return false;
    }
  }
  return true;
}

bool validPoint(const SyncCoverGraph &graph, const SyncCoverCutPoint &point,
                SyncCoverCutPointKind expected,
                std::optional<SyncCoverScopeId> loopScope,
                SyncCoverCoverageWorkBudget *workBudget) {
  const bool nodeAnchor =
      point.anchor.kind == SyncCoverAnchorKind::BeforeNode ||
      point.anchor.kind == SyncCoverAnchorKind::AfterNode;
  const bool scopeBoundary =
      point.anchor.kind == SyncCoverAnchorKind::ScopeEntry ||
      point.anchor.kind == SyncCoverAnchorKind::ScopeExit;
  const bool controlBoundary =
      point.anchor.kind == SyncCoverAnchorKind::ControlEntry ||
      point.anchor.kind == SyncCoverAnchorKind::ControlExit;
  const bool validAnchor =
      // SetFlag/WaitFlag are scalar-issued commands for their named physical
      // pipelines. Their program anchor need not itself be an operation on
      // that pipeline: a source-prefix Set can be issued at a later target
      // cut, and a target Wait can be issued immediately after a guarded
      // source. Exact grounding separately checks resource, guard, and
      // timeline semantics.
      (nodeAnchor && point.anchor.node < graph.getNodes().size()) ||
      (scopeBoundary && point.anchor.scope < graph.getScopes().size()) ||
      (controlBoundary && point.anchor.node < graph.getControls().size() &&
       point.anchor.scope == graph.getControls()[point.anchor.node].scope);
  const bool invalid = point.kind != expected || !validAnchor;
  if (invalid) {
    return false;
  }
  const SyncCoverScopeId occurrenceScope =
      nodeAnchor ? graph.getNodes()[point.anchor.node].scope
                 : point.anchor.scope;
  const std::optional<SyncCoverGuard> effective =
      effectivePointGuard(graph, point, workBudget);
  const bool invalidSemantics =
      !resolveSyncCoverAnchor(graph, point.anchor) ||
      !validGuardForScope(graph, point.guard, occurrenceScope, workBudget) ||
      !point.guard.literals.empty() || !effective;
  if (invalidSemantics) {
    return false;
  }
  return !loopScope ||
         scopeContains(graph, *loopScope, occurrenceScope, workBudget);
}

bool pointHasExactProtocolCardinality(const SyncCoverGraph &graph,
                                      const SyncCoverCutPoint &point,
                                      std::optional<SyncCoverScopeId> loopScope,
                                      SyncCoverCoverageWorkBudget *workBudget) {
  const bool nodeAnchor =
      point.anchor.kind == SyncCoverAnchorKind::BeforeNode ||
      point.anchor.kind == SyncCoverAnchorKind::AfterNode;
  const bool scopeBoundary =
      point.anchor.kind == SyncCoverAnchorKind::ScopeEntry ||
      point.anchor.kind == SyncCoverAnchorKind::ScopeExit;
  const bool controlBoundary =
      point.anchor.kind == SyncCoverAnchorKind::ControlEntry ||
      point.anchor.kind == SyncCoverAnchorKind::ControlExit;
  if (!nodeAnchor && !scopeBoundary && !controlBoundary) {
    return false;
  }
  const SyncCoverScopeId occurrenceScope =
      nodeAnchor ? graph.getNodes()[point.anchor.node].scope
                 : point.anchor.scope;
  const SyncCoverScopeId loopQueryScope =
      scopeBoundary && graph.getScopes()[occurrenceScope].isLoop
          ? graph.getScopes()[occurrenceScope].parent
          : occurrenceScope;
  const std::optional<SyncCoverScopeId> nearestLoop =
      nearestEnclosingLoop(graph, loopQueryScope, workBudget);
  const auto regionOutsideLoopBoundary = [&](SyncCoverRegionId region) {
    if (!scopeBoundary || !graph.getScopes()[occurrenceScope].isLoop) {
      return region;
    }
    while (region != 0) {
      const SyncCoverRegion &description = graph.getRegions()[region];
      if (description.kind == SyncCoverRegionKind::Loop &&
          description.scope == occurrenceScope) {
        return description.parent;
      }
      region = description.parent;
    }
    return region;
  };
  if (!loopScope) {
    if (nearestLoop) {
      return false;
    }
    SyncCoverRegionId region = nodeAnchor
                                   ? graph.getNodes()[point.anchor.node].region
                                   : graph.getScopes()[occurrenceScope].region;
    region = regionOutsideLoopBoundary(region);
    while (region != 0) {
      if (!consumeWork(workBudget)) {
        return false;
      }
      const SyncCoverRegion &description = graph.getRegions()[region];
      if (description.kind == SyncCoverRegionKind::Loop ||
          description.cardinality != SyncCoverRegionCardinality::ExactlyOnce) {
        return false;
      }
      region = description.parent;
    }
    return true;
  }
  if (!nearestLoop || *nearestLoop != *loopScope) {
    return false;
  }
  SyncCoverRegionId region = nodeAnchor
                                 ? graph.getNodes()[point.anchor.node].region
                                 : graph.getScopes()[occurrenceScope].region;
  region = regionOutsideLoopBoundary(region);
  const SyncCoverRegionId loopRegion = graph.getScopes()[*loopScope].region;
  while (region != loopRegion) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    const bool invalidRegion =
        region == 0 || region >= graph.getRegions().size();
    if (invalidRegion) {
      return false;
    }
    const SyncCoverRegion &description = graph.getRegions()[region];
    const bool repeated =
        description.kind == SyncCoverRegionKind::Loop ||
        description.cardinality == SyncCoverRegionCardinality::ZeroOrMore ||
        description.cardinality == SyncCoverRegionCardinality::OneOrMore;
    const bool guardedAlternative =
        description.cardinality == SyncCoverRegionCardinality::ZeroOrOne &&
        !description.guard.literals.empty();
    if (repeated ||
        (description.cardinality != SyncCoverRegionCardinality::ExactlyOnce &&
         !guardedAlternative)) {
      return false;
    }
    region = description.parent;
  }
  return true;
}

bool phaseIsActive(const SyncCoverEventTransfer &transfer, std::size_t phase) {
  return transfer.activePhases.empty() ||
         std::binary_search(transfer.activePhases.begin(),
                            transfer.activePhases.end(), phase);
}

bool phaseIsActive(const SyncCoverProtocolAction &action, std::size_t phase) {
  return action.activePhases.empty() ||
         std::binary_search(action.activePhases.begin(),
                            action.activePhases.end(), phase);
}

std::optional<SyncCoverGuard>
actionPhaseGuard(const SyncCoverGraph &graph,
                 const SyncCoverEventProtocol &protocol,
                 const SyncCoverProtocolAction &action,
                 SyncCoverCoverageWorkBudget *workBudget) {
  std::optional<SyncCoverGuard> guard =
      effectivePointGuard(graph, action.point, workBudget);
  if (!guard || !protocol.loop) {
    return guard;
  }
  std::optional<std::size_t> successorLiteral;
  for (std::size_t index = 0; index < guard->literals.size(); ++index) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    const SyncCoverGuardLiteral &literal = guard->literals[index];
    if (literal.control >= graph.getControls().size()) {
      return std::nullopt;
    }
    const SyncCoverControl &control = graph.getControls()[literal.control];
    const bool matches =
        control.successorRelation &&
        control.successorRelation->loopScope == protocol.loop->scope &&
        literal.alternative ==
            control.successorRelation->hasSuccessorAlternative;
    if (!matches) {
      continue;
    }
    if (successorLiteral) {
      return std::nullopt;
    }
    successorLiteral = index;
  }
  if (successorLiteral &&
      (action.guard == SyncCoverProtocolActionGuard::HasSuccessor ||
       action.guard == SyncCoverProtocolActionGuard::Always)) {
    guard->literals.erase(guard->literals.begin() + *successorLiteral);
  }
  guard->literals.erase(
      std::remove_if(
          guard->literals.begin(), guard->literals.end(),
          [&](const SyncCoverGuardLiteral &literal) {
            const SyncCoverControl &control =
                graph.getControls()[literal.control];
            if (!control.firstIterationRelation ||
                control.firstIterationRelation->loopScope !=
                    protocol.loop->scope) {
              return false;
            }
            const bool first =
                literal.alternative ==
                control.firstIterationRelation->firstIterationAlternative;
            return (first &&
                    action.guard ==
                        SyncCoverProtocolActionGuard::FirstIteration) ||
                   (!first &&
                    action.guard ==
                        SyncCoverProtocolActionGuard::NotFirstIteration);
          }),
      guard->literals.end());
  return guard;
}

bool actionPointHasValidSegment(const SyncCoverGraph &graph,
                                const SyncCoverEventProtocol &protocol,
                                const SyncCoverProtocolAction &action,
                                SyncCoverTimelinePosition position,
                                SyncCoverCoverageWorkBudget *workBudget) {
  if (!protocol.loop || protocol.loop->scope >= graph.getScopes().size()) {
    return action.segment == SyncCoverProtocolActionSegment::Body;
  }
  const std::optional<SyncCoverTimelineInterval> &timeline =
      graph.getScopes()[protocol.loop->scope].timeline;
  if (!timeline) {
    return false;
  }
  if (action.segment == SyncCoverProtocolActionSegment::Body) {
    const bool loopScopeBoundary =
        (action.point.anchor.kind == SyncCoverAnchorKind::ScopeEntry ||
         action.point.anchor.kind == SyncCoverAnchorKind::ScopeExit) &&
        action.point.anchor.scope < graph.getScopes().size() &&
        graph.getScopes()[action.point.anchor.scope].isLoop;
    if (loopScopeBoundary) {
      const std::optional<SyncCoverScopeId> parentLoop = nearestEnclosingLoop(
          graph, graph.getScopes()[action.point.anchor.scope].parent,
          workBudget);
      if (parentLoop && *parentLoop == protocol.loop->scope) {
        return true;
      }
    }
    return pointHasExactProtocolCardinality(graph, action.point,
                                            protocol.loop->scope, workBudget);
  }
  if (!consumeWork(workBudget)) {
    return false;
  }
  if (action.segment == SyncCoverProtocolActionSegment::Entry) {
    return position <= timeline->begin;
  }
  return position >= timeline->end;
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

SyncCoverProtocolError
validateProtocolShape(const SyncCoverEventProtocol &protocol,
                      SyncCoverCoverageWorkBudget *workBudget) {
  switch (protocol.kind) {
  case SyncCoverEventProtocolKind::SingleShot:
    return !protocol.loop && protocol.rearmProofs.empty() &&
                   protocol.channels.size() == 1 &&
                   protocol.channels.front().flow ==
                       SyncCoverEventChannelFlow::SingleShot
               ? SyncCoverProtocolError::None
               : SyncCoverProtocolError::InvalidProtocol;
  case SyncCoverEventProtocolKind::ProvenNoOverlap:
    if (!protocol.loop || protocol.rearmProofs.empty()) {
      return SyncCoverProtocolError::InvalidProtocol;
    }
    for (const SyncCoverEventChannel &channel : protocol.channels) {
      if (!consumeWork(workBudget)) {
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
      if (channel.flow == SyncCoverEventChannelFlow::SingleShot) {
        return SyncCoverProtocolError::InvalidProtocol;
      }
    }
    return SyncCoverProtocolError::None;
  case SyncCoverEventProtocolKind::RoundTrip:
  case SyncCoverEventProtocolKind::RotatingLanes: {
    using ChannelKey = std::tuple<SyncCoverEventChannelFlow, std::uint32_t,
                                  std::uint32_t, std::size_t>;
    const bool rotating =
        protocol.kind == SyncCoverEventProtocolKind::RotatingLanes;
    if (!protocol.loop || !protocol.rearmProofs.empty()) {
      return SyncCoverProtocolError::InvalidProtocol;
    }
    std::vector<ChannelKey> keys;
    keys.reserve(protocol.channels.size());
    for (const SyncCoverEventChannel &channel : protocol.channels) {
      if (!consumeWork(workBudget)) {
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
      keys.emplace_back(channel.flow, channel.set.resource,
                        channel.wait.resource, channel.width);
    }
    if (!reserveSortWork(keys.size(), workBudget)) {
      return workBudget && workBudget->exhausted
                 ? SyncCoverProtocolError::WorkLimitExceeded
                 : SyncCoverProtocolError::LimitExceeded;
    }
    std::sort(keys.begin(), keys.end());
    std::size_t lookupWork = 1;
    for (std::size_t value = keys.size(); value > 1; value = (value + 1) / 2) {
      ++lookupWork;
    }
    for (const SyncCoverEventChannel &channel : protocol.channels) {
      if (!consumeWork(workBudget, lookupWork)) {
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
      const bool wideEnough = !rotating || channel.width > 1;
      const SyncCoverEventChannelFlow reverseFlow =
          channel.flow == SyncCoverEventChannelFlow::SameIteration
              ? SyncCoverEventChannelFlow::LoopCarry
              : SyncCoverEventChannelFlow::SameIteration;
      const ChannelKey reverse{reverseFlow, channel.wait.resource,
                               channel.set.resource, channel.width};
      if (!wideEnough ||
          channel.flow == SyncCoverEventChannelFlow::SingleShot ||
          !std::binary_search(keys.begin(), keys.end(), reverse)) {
        return SyncCoverProtocolError::InvalidProtocol;
      }
    }
    return SyncCoverProtocolError::None;
  }
  case SyncCoverEventProtocolKind::LifecycleNetwork:
    if (!protocol.loop || !protocol.rearmProofs.empty() ||
        protocol.channels.size() < 2) {
      return SyncCoverProtocolError::InvalidProtocol;
    }
    for (const SyncCoverEventChannel &channel : protocol.channels) {
      if (!consumeWork(workBudget)) {
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
      if (channel.flow == SyncCoverEventChannelFlow::SingleShot) {
        return SyncCoverProtocolError::InvalidProtocol;
      }
    }
    return SyncCoverProtocolError::None;
  }
  return SyncCoverProtocolError::InvalidProtocol;
}

SyncCoverProtocolError resolveLoop(const SyncCoverGraph &graph,
                                   const SyncCoverEventProtocol &protocol,
                                   SyncCoverProtocolLimits limits,
                                   ResolvedProtocol &resolved,
                                   SyncCoverCoverageWorkBudget *workBudget) {
  if (!protocol.loop) {
    if (!consumeWork(workBudget, 3)) {
      return SyncCoverProtocolError::WorkLimitExceeded;
    }
    resolved.reachablePhases = {0};
    resolved.nextPhase = {0};
    resolved.loopGuard = {};
    resolved.guardByPhase = {SyncCoverGuard{}};
    return SyncCoverProtocolError::None;
  }
  const SyncCoverProtocolLoopSchedule &loop = *protocol.loop;
  const SyncCoverRegionId loopRegionId =
      loop.scope < graph.getScopes().size()
          ? graph.getScopes()[loop.scope].region
          : 0;
  const bool validLoopRegion =
      loopRegionId < graph.getRegions().size() &&
      graph.getRegions()[loopRegionId].kind == SyncCoverRegionKind::Loop;
  const bool regionMayExecuteZeroTimes =
      validLoopRegion && graph.getRegions()[loopRegionId].cardinality ==
                             SyncCoverRegionCardinality::ZeroOrMore;
  const bool validLoopCardinality =
      validLoopRegion && (graph.getRegions()[loopRegionId].cardinality ==
                              SyncCoverRegionCardinality::ZeroOrMore ||
                          graph.getRegions()[loopRegionId].cardinality ==
                              SyncCoverRegionCardinality::OneOrMore);
  const bool invalidHeader =
      loop.scope == 0 || loop.scope >= graph.getScopes().size() ||
      !graph.getScopes()[loop.scope].isLoop || !validLoopCardinality ||
      regionMayExecuteZeroTimes != loop.mayExecuteZeroTimes ||
      loop.laneByPhase.empty();
  if (invalidHeader) {
    return SyncCoverProtocolError::InvalidProtocol;
  }
  resolved.loopGuard = graph.getScopes()[loop.scope].guard;
  if (!loop.phaseControl) {
    const bool invalidUnconditionalSchedule = loop.laneByPhase.size() != 1;
    if (invalidUnconditionalSchedule) {
      return SyncCoverProtocolError::InvalidProtocol;
    }
    resolved.initialPhase = 0;
    resolved.nextPhase = {0};
    resolved.guardByPhase = {SyncCoverGuard{}};
  } else {
    if (*loop.phaseControl >= graph.getControls().size()) {
      return SyncCoverProtocolError::InvalidProtocol;
    }
    const SyncCoverControl &control = graph.getControls()[*loop.phaseControl];
    if (!control.phaseRelation ||
        control.phaseRelation->loopScope != loop.scope ||
        control.phaseRelation->nextPhase.size() != loop.laneByPhase.size()) {
      return SyncCoverProtocolError::InvalidProtocol;
    }
    const bool phaseLimitExceeded =
        control.phaseRelation->nextPhase.size() >
            limits.maximumReachablePhases ||
        control.phaseRelation->nextPhase.size() > limits.maximumPhaseIncidences;
    if (phaseLimitExceeded) {
      return SyncCoverProtocolError::LimitExceeded;
    }
    std::size_t phaseCopyWork = 0;
    const bool phaseWorkOverflow = !checkedAdd(
        control.phaseRelation->nextPhase.size(),
        control.phaseRelation->activeAlternative.size(), phaseCopyWork);
    if (phaseWorkOverflow || !consumeWork(workBudget, phaseCopyWork)) {
      return workBudget && workBudget->exhausted
                 ? SyncCoverProtocolError::WorkLimitExceeded
                 : SyncCoverProtocolError::LimitExceeded;
    }
    resolved.initialPhase = control.phaseRelation->initialPhase;
    resolved.nextPhase = control.phaseRelation->nextPhase;
    resolved.guardByPhase.reserve(
        control.phaseRelation->activeAlternative.size());
    for (unsigned alternative : control.phaseRelation->activeAlternative) {
      resolved.guardByPhase.push_back(
          SyncCoverGuard{{{*loop.phaseControl, alternative}}});
    }
  }

  const bool phaseLimitExceeded =
      resolved.nextPhase.size() > limits.maximumReachablePhases ||
      resolved.nextPhase.size() > limits.maximumPhaseIncidences;
  if (phaseLimitExceeded) {
    return SyncCoverProtocolError::LimitExceeded;
  }

  if (!consumeWork(workBudget, resolved.nextPhase.size())) {
    return SyncCoverProtocolError::WorkLimitExceeded;
  }
  std::vector<std::size_t> firstVisit(resolved.nextPhase.size(),
                                      std::numeric_limits<std::size_t>::max());
  std::size_t phase = resolved.initialPhase;
  while (firstVisit[phase] == std::numeric_limits<std::size_t>::max()) {
    const bool workUnavailable = !consumeWork(workBudget);
    const bool phaseLimitReached =
        resolved.reachablePhases.size() == limits.maximumReachablePhases;
    if (workUnavailable || phaseLimitReached) {
      return workBudget && workBudget->exhausted
                 ? SyncCoverProtocolError::WorkLimitExceeded
                 : SyncCoverProtocolError::LimitExceeded;
    }
    firstVisit[phase] = resolved.reachablePhases.size();
    resolved.reachablePhases.push_back(phase);
    phase = resolved.nextPhase[phase];
  }
  resolved.phasePreperiod = firstVisit[phase];
  resolved.phasePeriod =
      resolved.reachablePhases.size() - resolved.phasePreperiod;
  return resolved.phasePeriod == 0 ? SyncCoverProtocolError::InvalidProtocol
                                   : SyncCoverProtocolError::None;
}

SyncCoverProtocolError
validatePhaseDistances(const SyncCoverEventProtocol &protocol,
                       const ResolvedProtocol &resolved,
                       const ResolvedChannel &resolvedChannel,
                       SyncCoverCoverageWorkBudget *workBudget) {
  const SyncCoverEventChannel &channel = *resolvedChannel.description;
  if (!protocol.loop) {
    return SyncCoverProtocolError::None;
  }
  if (!consumeWork(workBudget, channel.width)) {
    return SyncCoverProtocolError::WorkLimitExceeded;
  }
  const SyncCoverProtocolLoopSchedule &loop = *protocol.loop;
  struct PreviousSet {
    std::size_t iteration = 0;
    std::size_t transfer = 0;
  };
  std::vector<std::optional<PreviousSet>> previous(channel.width);
  bool observed = false;
  std::size_t phase = resolved.initialPhase;
  for (std::size_t iteration = 0; iteration < resolved.verificationHorizon;
       ++iteration) {
    for (std::size_t transferIndex = 0;
         transferIndex < resolvedChannel.transfers.size(); ++transferIndex) {
      const ResolvedTransfer &resolvedTransfer =
          resolvedChannel.transfers[transferIndex];
      const SyncCoverEventTransfer &transfer = resolvedTransfer.description;
      if (!consumeWork(workBudget,
                       logarithmicLookupWork(transfer.activePhases.size()) +
                           2)) {
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
      if (!phaseIsActive(transfer, phase)) {
        continue;
      }
      const std::size_t lane = channel.transfers.empty()
                                   ? loop.laneByPhase[phase]
                                   : transfer.waitLane;
      if (lane >= channel.width) {
        return SyncCoverProtocolError::InvalidProtocol;
      }
      if (channel.flow == SyncCoverEventChannelFlow::LoopCarry &&
          previous[lane]) {
        const PreviousSet prior = *previous[lane];
        const bool sameIteration = prior.iteration == iteration;
        const bool orderedWithinIteration =
            sameIteration && prior.transfer < transferIndex;
        const bool expectedCarry =
            !sameIteration && iteration >= prior.iteration &&
            iteration - prior.iteration == channel.distance;
        if (!orderedWithinIteration && !expectedCarry) {
          return SyncCoverProtocolError::InvalidProtocol;
        }
      }
      previous[channel.transfers.empty() ? lane : transfer.setLane] =
          PreviousSet{iteration, transferIndex};
      observed = true;
    }
    phase = resolved.nextPhase[phase];
  }
  return observed ? SyncCoverProtocolError::None
                  : SyncCoverProtocolError::InvalidProtocol;
}

} // namespace

bool SyncCoverProtocolTargetContract::EventCapability::operator<(
    const EventCapability &other) const {
  return std::tie(sourceResource, targetResource, suppliedRequirements) <
         std::tie(other.sourceResource, other.targetResource,
                  other.suppliedRequirements);
}

bool SyncCoverProtocolTargetContract::EventCapability::operator==(
    const EventCapability &other) const {
  return sourceResource == other.sourceResource &&
         targetResource == other.targetResource &&
         suppliedRequirements == other.suppliedRequirements;
}

bool SyncCoverProtocolTargetContract::RearmFact::operator<(
    const RearmFact &other) const {
  if (evidence != other.evidence) {
    return evidence < other.evidence;
  }
  if (fromWaitResource != other.fromWaitResource) {
    return fromWaitResource < other.fromWaitResource;
  }
  const bool differentWaitAnchors =
      anchorKey(fromWaitAnchor) != anchorKey(other.fromWaitAnchor);
  if (differentWaitAnchors) {
    return anchorKey(fromWaitAnchor) < anchorKey(other.fromWaitAnchor);
  }
  if (!guardsEqual(fromWaitGuard, other.fromWaitGuard)) {
    return guardsLess(fromWaitGuard, other.fromWaitGuard);
  }
  if (toSetResource != other.toSetResource) {
    return toSetResource < other.toSetResource;
  }
  const bool differentSetAnchors =
      anchorKey(toSetAnchor) != anchorKey(other.toSetAnchor);
  if (differentSetAnchors) {
    return anchorKey(toSetAnchor) < anchorKey(other.toSetAnchor);
  }
  if (!guardsEqual(toSetGuard, other.toSetGuard)) {
    return guardsLess(toSetGuard, other.toSetGuard);
  }
  return std::tie(loopScope, iterationDistance, width) <
         std::tie(other.loopScope, other.iterationDistance, other.width);
}

bool SyncCoverProtocolTargetContract::RearmFact::operator==(
    const RearmFact &other) const {
  return evidence == other.evidence &&
         fromWaitResource == other.fromWaitResource &&
         anchorsEqual(fromWaitAnchor, other.fromWaitAnchor) &&
         guardsEqual(fromWaitGuard, other.fromWaitGuard) &&
         toSetResource == other.toSetResource &&
         anchorsEqual(toSetAnchor, other.toSetAnchor) &&
         guardsEqual(toSetGuard, other.toSetGuard) &&
         loopScope == other.loopScope &&
         iterationDistance == other.iterationDistance && width == other.width;
}

bool SyncCoverProtocolTargetContract::supportsEvent(
    std::uint32_t source, std::uint32_t target,
    SyncCoverOrderingRequirementMask requirements) const {
  const EventCapability lowerBound{source, target, 0};
  auto capability = std::lower_bound(eventCapabilities.begin(),
                                     eventCapabilities.end(), lowerBound);
  SyncCoverOrderingRequirementMask supplied = 0;
  while (capability != eventCapabilities.end()) {
    if (capability->sourceResource != source ||
        capability->targetResource != target) {
      break;
    }
    supplied = static_cast<SyncCoverOrderingRequirementMask>(
        supplied | capability->suppliedRequirements);
    ++capability;
  }
  return (supplied & requirements) == requirements;
}

const SyncCoverProtocolTargetContract::RearmFact *
SyncCoverProtocolTargetContract::findRearmFact(std::uint32_t evidence) const {
  const auto found = std::lower_bound(
      certifiedRearmFacts.begin(), certifiedRearmFacts.end(), evidence,
      [](const RearmFact &fact, std::uint32_t value) {
        return fact.evidence < value;
      });
  return found != certifiedRearmFacts.end() && found->evidence == evidence
             ? &*found
             : nullptr;
}

bool mlir::pto::sync_cover_protocol_detail::consumeWork(
    SyncCoverCoverageWorkBudget *budget, std::size_t amount) {
  return !budget || budget->consume(amount);
}

bool mlir::pto::sync_cover_protocol_detail::graphFitsProtocolLimits(
    const SyncCoverGraph &graph, SyncCoverProtocolLimits limits) {
  return graph.getNodes().size() <= limits.maximumGraphNodes &&
         graph.getEdges().size() <= limits.maximumGraphEdges &&
         graph.getDemands().size() <= limits.maximumGraphDemands &&
         graph.getScopes().size() <= limits.maximumGraphScopes &&
         graph.getRegions().size() <= limits.maximumGraphRegions &&
         graph.getControls().size() <= limits.maximumGraphControls &&
         graph.getStorageDomains().size() <=
             limits.maximumGraphStorageDomains &&
         graph.getStorageAccesses().size() <=
             limits.maximumGraphStorageAccesses &&
         graph.getStorageWitnesses().size() <=
             limits.maximumGraphStorageWitnesses;
}

bool mlir::pto::sync_cover_protocol_detail::targetFitsProtocolLimits(
    const SyncCoverProtocolTargetContract &target,
    SyncCoverProtocolLimits limits) {
  return target.eventCapabilities.size() <= limits.maximumTargetCapabilities &&
         target.certifiedRearmFacts.size() <= limits.maximumTargetRearmFacts &&
         target.compilerUsableEventIds.size() <= limits.maximumTargetEventIds;
}

SyncCoverProtocolError
mlir::pto::sync_cover_protocol_detail::validateProtocolTargetContract(
    const SyncCoverProtocolTargetContract &target,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  if (!targetFitsProtocolLimits(target, limits)) {
    return SyncCoverProtocolError::LimitExceeded;
  }
  std::size_t entries = 0;
  std::size_t guardLiterals = 0;
  const bool countOverflow =
      !checkedAdd(target.eventCapabilities.size(),
                  target.compilerUsableEventIds.size(), entries) ||
      !checkedAdd(entries, target.certifiedRearmFacts.size(), entries);
  for (const SyncCoverProtocolTargetContract::RearmFact &fact :
       target.certifiedRearmFacts) {
    if (!checkedAdd(guardLiterals, fact.fromWaitGuard.literals.size(),
                    guardLiterals) ||
        !checkedAdd(guardLiterals, fact.toSetGuard.literals.size(),
                    guardLiterals)) {
      return SyncCoverProtocolError::LimitExceeded;
    }
  }
  std::size_t entryWork = 0;
  std::size_t guardWork = 0;
  const bool workOverflow = countOverflow ||
                            guardLiterals > limits.maximumGuardLiterals ||
                            !checkedProduct(entries, 4, entryWork) ||
                            !checkedProduct(guardLiterals, 6, guardWork) ||
                            !checkedAdd(entryWork, guardWork, entryWork);
  if (workOverflow) {
    return SyncCoverProtocolError::LimitExceeded;
  }
  if (!consumeWork(workBudget, entryWork)) {
    return SyncCoverProtocolError::WorkLimitExceeded;
  }
  return validTargetContract(target)
             ? SyncCoverProtocolError::None
             : SyncCoverProtocolError::InvalidTargetContract;
}

std::optional<SyncCoverGuard>
mlir::pto::sync_cover_protocol_detail::effectivePointGuard(
    const SyncCoverGraph &graph, const SyncCoverCutPoint &point,
    SyncCoverCoverageWorkBudget *workBudget) {
  const bool nodeAnchor =
      point.anchor.kind == SyncCoverAnchorKind::BeforeNode ||
      point.anchor.kind == SyncCoverAnchorKind::AfterNode;
  const bool scopeBoundary =
      point.anchor.kind == SyncCoverAnchorKind::ScopeEntry ||
      point.anchor.kind == SyncCoverAnchorKind::ScopeExit;
  const bool controlBoundary =
      point.anchor.kind == SyncCoverAnchorKind::ControlEntry ||
      point.anchor.kind == SyncCoverAnchorKind::ControlExit;
  if ((!nodeAnchor && !scopeBoundary && !controlBoundary) ||
      (nodeAnchor && point.anchor.node >= graph.getNodes().size()) ||
      ((scopeBoundary || controlBoundary) &&
       point.anchor.scope >= graph.getScopes().size()) ||
      (controlBoundary &&
       (point.anchor.node >= graph.getControls().size() ||
        point.anchor.scope != graph.getControls()[point.anchor.node].scope))) {
    return std::nullopt;
  }
  const SyncCoverGuard &occurrenceGuard =
      nodeAnchor ? graph.getNodes()[point.anchor.node].guard
                 : graph.getScopes()[point.anchor.scope].guard;
  std::size_t guardLiterals = 0;
  if (!checkedAdd(point.guard.literals.size(), occurrenceGuard.literals.size(),
                  guardLiterals) ||
      !consumeWork(workBudget, guardLiterals)) {
    return std::nullopt;
  }
  SyncCoverGuard result = point.guard;
  result.literals.insert(result.literals.end(),
                         occurrenceGuard.literals.begin(),
                         occurrenceGuard.literals.end());
  return normalizeGuard(result, workBudget)
             ? std::optional<SyncCoverGuard>(std::move(result))
             : std::nullopt;
}

SyncCoverProtocolError mlir::pto::sync_cover_protocol_detail::resolveProtocol(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverEventProtocol &protocol, SyncCoverProtocolLimits limits,
    ResolvedProtocol &resolved, std::optional<std::size_t> &invalidIndex,
    SyncCoverCoverageWorkBudget *workBudget, bool validateGraph,
    bool validateTarget) {
  const bool invalidGraph =
      validateGraph && (!graph.isStructureFrozen() || !graph.validate());
  if (invalidGraph) {
    return SyncCoverProtocolError::InvalidGraph;
  }
  if (validateTarget) {
    const SyncCoverProtocolError targetError =
        validateProtocolTargetContract(target, limits, workBudget);
    if (targetError != SyncCoverProtocolError::None) {
      return targetError;
    }
  }
  const bool invalidChannelCount =
      protocol.channels.empty() ||
      protocol.channels.size() > limits.maximumChannels;
  const bool invalidProofCount =
      protocol.rearmProofs.size() > limits.maximumRearmProofs;
  if (invalidChannelCount || invalidProofCount) {
    return SyncCoverProtocolError::InvalidProtocol;
  }
  const SyncCoverProtocolError shapeError =
      validateProtocolShape(protocol, workBudget);
  if (shapeError != SyncCoverProtocolError::None) {
    return shapeError;
  }
  SyncCoverProtocolError loopError =
      resolveLoop(graph, protocol, limits, resolved, workBudget);
  if (loopError != SyncCoverProtocolError::None) {
    return loopError;
  }
  if (protocol.lifetimeScope) {
    const bool invalidLifetime =
        !protocol.loop || *protocol.lifetimeScope == 0 ||
        *protocol.lifetimeScope >= graph.getScopes().size() ||
        !graph.getScopes()[*protocol.lifetimeScope].isLoop ||
        !graph.scopeContains(*protocol.lifetimeScope, protocol.loop->scope);
    if (invalidLifetime) {
      return SyncCoverProtocolError::InvalidProtocol;
    }
  }

  std::size_t maximumSpan = 1;
  std::size_t laneIncidences = 0;
  std::size_t guardLiterals = 0;
  std::size_t phaseIncidences = resolved.nextPhase.size();
  std::size_t transferIncidences = 0;
  resolved.channels.reserve(protocol.channels.size());
  for (std::size_t index = 0; index < protocol.channels.size(); ++index) {
    const SyncCoverEventChannel &channel = protocol.channels[index];
    const bool explicitRecipe =
        !channel.actions.empty() || !channel.supplies.empty();
    const std::vector<SyncCoverEventTransfer> transfers =
        explicitRecipe ? std::vector<SyncCoverEventTransfer>{}
                       : getChannelTransfers(channel);
    std::size_t phaseValidationWork = 0;
    std::size_t capabilityLookupWork = 0;
    const bool validationWorkOverflow =
        !checkedAdd(channel.actions.size(), channel.supplies.size(),
                    phaseValidationWork) ||
        !checkedAdd(phaseValidationWork, transfers.size(),
                    phaseValidationWork) ||
        !checkedProduct(phaseValidationWork, 3, phaseValidationWork) ||
        !checkedAdd(target.eventCapabilities.size(),
                    logarithmicLookupWork(target.eventCapabilities.size()),
                    capabilityLookupWork) ||
        !checkedAdd(phaseValidationWork, capabilityLookupWork,
                    phaseValidationWork) ||
        !checkedAdd(phaseValidationWork, 1, phaseValidationWork);
    if (validationWorkOverflow ||
        !consumeWork(workBudget, phaseValidationWork)) {
      return workBudget && workBudget->exhausted
                 ? SyncCoverProtocolError::WorkLimitExceeded
                 : SyncCoverProtocolError::LimitExceeded;
    }
    if (workBudget && workBudget->exhausted) {
      return SyncCoverProtocolError::WorkLimitExceeded;
    }
    const bool invalidIdentity = channel.id != index || channel.width == 0;
    const bool laneLimitExceeded =
        channel.width >
        limits.maximumChannelLaneIncidences -
            std::min(laneIncidences, limits.maximumChannelLaneIncidences);
    const bool invalidRequirements = channel.suppliedRequirements == 0 ||
                                     (channel.suppliedRequirements &
                                      ~kAllSyncCoverOrderingRequirements) != 0;
    const bool invalidPhases =
        !std::is_sorted(channel.activePhases.begin(),
                        channel.activePhases.end()) ||
        std::adjacent_find(channel.activePhases.begin(),
                           channel.activePhases.end()) !=
            channel.activePhases.end() ||
        (protocol.loop &&
         std::any_of(channel.activePhases.begin(), channel.activePhases.end(),
                     [&](std::size_t phase) {
                       return phase >= resolved.nextPhase.size();
                     }));
    const bool invalidFlow =
        (channel.flow == SyncCoverEventChannelFlow::SingleShot &&
         (protocol.loop || channel.width != 1 || channel.distance != 0)) ||
        (channel.flow == SyncCoverEventChannelFlow::SameIteration &&
         (!protocol.loop || channel.distance != 0)) ||
        (channel.flow == SyncCoverEventChannelFlow::LoopCarry &&
         (!protocol.loop || channel.distance == 0 ||
          (!explicitRecipe && channel.transfers.empty() &&
           channel.width != channel.distance)));
    const bool invalidRecipe =
        explicitRecipe &&
        (protocol.kind != SyncCoverEventProtocolKind::LifecycleNetwork ||
         channel.actions.empty() || channel.supplies.empty() ||
         !channel.transfers.empty());
    const std::optional<SyncCoverScopeId> loopScope =
        protocol.loop ? std::optional<SyncCoverScopeId>(protocol.loop->scope)
                      : std::nullopt;
    if (workBudget && workBudget->exhausted) {
      invalidIndex = index;
      return SyncCoverProtocolError::WorkLimitExceeded;
    }
    if (laneLimitExceeded) {
      invalidIndex = index;
      return SyncCoverProtocolError::LimitExceeded;
    }
    if (invalidIdentity || invalidRequirements || invalidPhases ||
        invalidFlow || invalidRecipe ||
        (!explicitRecipe && transfers.empty())) {
      invalidIndex = index;
      return SyncCoverProtocolError::InvalidProtocol;
    }
    laneIncidences += channel.width;
    const std::size_t channelIncidences =
        explicitRecipe ? channel.actions.size() : transfers.size();
    if (channelIncidences >
        limits.maximumDynamicActions -
            std::min(transferIncidences, limits.maximumDynamicActions)) {
      invalidIndex = index;
      return SyncCoverProtocolError::LimitExceeded;
    }
    transferIncidences += channelIncidences;
    ResolvedChannel resolvedChannel;
    resolvedChannel.description = &channel;
    if (explicitRecipe) {
      resolvedChannel.actions.reserve(channel.actions.size());
      for (std::size_t actionIndex = 0; actionIndex < channel.actions.size();
           ++actionIndex) {
        const SyncCoverProtocolAction &action = channel.actions[actionIndex];
        const bool expectedSet =
            action.kind == SyncCoverProtocolActionKind::Set;
        const SyncCoverCutPointKind pointKind =
            expectedSet ? SyncCoverCutPointKind::EventSet
                        : SyncCoverCutPointKind::EventWait;
        const bool invalidActionIdentity = action.id != actionIndex;
        const bool invalidActionLane = action.lane >= channel.width;
        const bool invalidActionPhases =
            !std::is_sorted(action.activePhases.begin(),
                            action.activePhases.end()) ||
            std::adjacent_find(action.activePhases.begin(),
                               action.activePhases.end()) !=
                action.activePhases.end() ||
            (action.segment != SyncCoverProtocolActionSegment::Body
                 ? !action.activePhases.empty()
                 : std::any_of(action.activePhases.begin(),
                               action.activePhases.end(),
                               [&](std::size_t phase) {
                                 return phase >= resolved.nextPhase.size();
                               }));
        const bool invalidTemporalGuard =
            (action.segment == SyncCoverProtocolActionSegment::Entry &&
             action.guard != SyncCoverProtocolActionGuard::Always &&
             action.guard != SyncCoverProtocolActionGuard::LoopNonEmpty &&
             action.guard != SyncCoverProtocolActionGuard::LoopEmpty) ||
            (action.segment == SyncCoverProtocolActionSegment::Body &&
             action.guard != SyncCoverProtocolActionGuard::Always &&
             action.guard != SyncCoverProtocolActionGuard::FirstIteration &&
             action.guard != SyncCoverProtocolActionGuard::NotFirstIteration &&
             action.guard != SyncCoverProtocolActionGuard::HasSuccessor) ||
            (action.segment == SyncCoverProtocolActionSegment::Exit &&
             action.guard != SyncCoverProtocolActionGuard::Always &&
             action.guard != SyncCoverProtocolActionGuard::LoopNonEmpty &&
             action.guard != SyncCoverProtocolActionGuard::LoopEmpty);
        const bool invalidPoint =
            !validPoint(graph, action.point, pointKind, std::nullopt,
                        workBudget) ||
            action.point.resource !=
                (expectedSet ? channel.set.resource : channel.wait.resource) ||
            !target.supportsEvent(channel.set.resource, channel.wait.resource,
                                  channel.suppliedRequirements);
        const std::optional<SyncCoverTimelinePosition> position =
            resolveSyncCoverAnchor(graph, action.point.anchor);
        const std::optional<SyncCoverGuard> guard =
            actionPhaseGuard(graph, protocol, action, workBudget);
        if (workBudget && workBudget->exhausted) {
          invalidIndex = index;
          return SyncCoverProtocolError::WorkLimitExceeded;
        }
        if (invalidActionIdentity || invalidActionLane || invalidActionPhases ||
            invalidTemporalGuard || invalidPoint || !position || !guard ||
            !actionPointHasValidSegment(graph, protocol, action, *position,
                                        workBudget)) {
          invalidIndex = index;
          return SyncCoverProtocolError::InvalidProtocol;
        }
        if (action.segment == SyncCoverProtocolActionSegment::Body) {
          for (std::size_t phase : resolved.reachablePhases) {
            SyncCoverGuard phaseCondition =
                graph.getScopes()[protocol.loop->scope].guard;
            phaseCondition.literals.insert(
                phaseCondition.literals.end(),
                resolved.guardByPhase[phase].literals.begin(),
                resolved.guardByPhase[phase].literals.end());
            if (!normalizeGuard(phaseCondition, workBudget)) {
              invalidIndex = index;
              return workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::InvalidProtocol;
            }
            if (!consumeWork(workBudget, phaseCondition.literals.size() +
                                             guard->literals.size())) {
              invalidIndex = index;
              return SyncCoverProtocolError::WorkLimitExceeded;
            }
            const bool mayExecute =
                syncCoverGuardsCompatible(phaseCondition, *guard);
            if (mayExecute != phaseIsActive(action, phase)) {
              invalidIndex = index;
              return SyncCoverProtocolError::InvalidProtocol;
            }
          }
        }
        std::size_t nextGuardLiterals = 0;
        std::size_t nextPhaseIncidences = 0;
        if (!checkedAdd(guardLiterals, guard->literals.size(),
                        nextGuardLiterals) ||
            !checkedAdd(phaseIncidences, action.activePhases.size(),
                        nextPhaseIncidences) ||
            nextGuardLiterals > limits.maximumGuardLiterals ||
            nextPhaseIncidences > limits.maximumPhaseIncidences) {
          invalidIndex = index;
          return SyncCoverProtocolError::LimitExceeded;
        }
        guardLiterals = nextGuardLiterals;
        phaseIncidences = nextPhaseIncidences;
        resolvedChannel.actions.push_back({action, *position, *guard});
      }
      for (const SyncCoverProtocolSupply &supply : channel.supplies) {
        if (!consumeWork(workBudget)) {
          invalidIndex = index;
          return SyncCoverProtocolError::WorkLimitExceeded;
        }
        const bool invalidDistanceScope =
            supply.distanceScope &&
            (!protocol.loop || *supply.distanceScope == 0 ||
             *supply.distanceScope >= graph.getScopes().size() ||
             !graph.getScopes()[*supply.distanceScope].isLoop ||
             !graph.scopeContains(*supply.distanceScope,
                                  protocol.loop->scope) ||
             (protocol.lifetimeScope &&
              !graph.scopeContains(*protocol.lifetimeScope,
                                   *supply.distanceScope)));
        const bool invalidActionReferences =
            supply.setAction >= channel.actions.size() ||
            supply.waitAction >= channel.actions.size();
        const bool completionExport =
            supply.kind == SyncCoverProtocolSupplyKind::CompletionExport;
        const bool completionExportWait =
            !invalidActionReferences &&
            (channel.actions[supply.waitAction].segment ==
                 SyncCoverProtocolActionSegment::Body ||
             (channel.actions[supply.waitAction].segment ==
                  SyncCoverProtocolActionSegment::Exit &&
              supply.distance == 0));
        const bool invalidCompletionExport =
            completionExport &&
            (!protocol.loop || !protocol.lifetimeScope ||
             !supply.distanceScope ||
             *supply.distanceScope != *protocol.lifetimeScope ||
             invalidActionReferences ||
             (channel.actions[supply.setAction].segment !=
                  SyncCoverProtocolActionSegment::Body &&
              channel.actions[supply.setAction].segment !=
                  SyncCoverProtocolActionSegment::Entry) ||
             !completionExportWait);
        const bool invalidSupply = invalidActionReferences ||
                                   channel.actions[supply.setAction].kind !=
                                       SyncCoverProtocolActionKind::Set ||
                                   channel.actions[supply.waitAction].kind !=
                                       SyncCoverProtocolActionKind::Wait ||
                                   (!protocol.loop && supply.distance != 0) ||
                                   invalidDistanceScope ||
                                   invalidCompletionExport;
        if (invalidSupply) {
          invalidIndex = index;
          return SyncCoverProtocolError::InvalidProtocol;
        }
        maximumSpan = std::max<std::size_t>(maximumSpan, supply.distance);
      }
      resolved.channels.push_back(std::move(resolvedChannel));
      continue;
    }
    resolvedChannel.transfers.reserve(transfers.size());
    for (std::size_t transferIndex = 0; transferIndex < transfers.size();
         ++transferIndex) {
      SyncCoverEventTransfer transfer = transfers[transferIndex];
      transfer.id = transferIndex;
      const bool explicitTransfer = !channel.transfers.empty();
      const bool invalidTransferIdentity =
          explicitTransfer &&
          channel.transfers[transferIndex].id != transferIndex;
      const bool invalidTransferPhases =
          !std::is_sorted(transfer.activePhases.begin(),
                          transfer.activePhases.end()) ||
          std::adjacent_find(transfer.activePhases.begin(),
                             transfer.activePhases.end()) !=
              transfer.activePhases.end() ||
          (!protocol.loop && !transfer.activePhases.empty()) ||
          (protocol.loop &&
           std::any_of(transfer.activePhases.begin(),
                       transfer.activePhases.end(), [&](std::size_t phase) {
                         return phase >= resolved.nextPhase.size();
                       }));
      const bool invalidTransferLanes =
          explicitTransfer && (transfer.setLane >= channel.width ||
                               transfer.waitLane >= channel.width);
      const bool invalidPoints =
          !validPoint(graph, transfer.set, SyncCoverCutPointKind::EventSet,
                      loopScope, workBudget) ||
          !validPoint(graph, transfer.wait, SyncCoverCutPointKind::EventWait,
                      loopScope, workBudget) ||
          !pointHasExactProtocolCardinality(graph, transfer.set, loopScope,
                                            workBudget) ||
          !pointHasExactProtocolCardinality(graph, transfer.wait, loopScope,
                                            workBudget) ||
          transfer.set.resource != channel.set.resource ||
          transfer.wait.resource != channel.wait.resource ||
          !target.supportsEvent(transfer.set.resource, transfer.wait.resource,
                                channel.suppliedRequirements);
      const std::optional<SyncCoverTimelinePosition> setPosition =
          resolveSyncCoverAnchor(graph, transfer.set.anchor);
      const std::optional<SyncCoverTimelinePosition> waitPosition =
          resolveSyncCoverAnchor(graph, transfer.wait.anchor);
      const std::optional<SyncCoverGuard> setGuard =
          effectivePointGuard(graph, transfer.set, workBudget);
      const std::optional<SyncCoverGuard> waitGuard =
          effectivePointGuard(graph, transfer.wait, workBudget);
      if (workBudget && workBudget->exhausted) {
        invalidIndex = index;
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
      if (invalidTransferIdentity || invalidTransferPhases ||
          invalidTransferLanes || invalidPoints || !setPosition ||
          !waitPosition || !setGuard || !waitGuard) {
        invalidIndex = index;
        return SyncCoverProtocolError::InvalidProtocol;
      }
      std::size_t nextGuardLiterals = 0;
      const bool guardOverflow =
          !checkedAdd(guardLiterals, setGuard->literals.size(),
                      nextGuardLiterals) ||
          !checkedAdd(nextGuardLiterals, waitGuard->literals.size(),
                      nextGuardLiterals) ||
          nextGuardLiterals > limits.maximumGuardLiterals;
      const bool phaseIncidenceLimitExceeded =
          transfer.activePhases.size() >
          limits.maximumPhaseIncidences -
              std::min(phaseIncidences, limits.maximumPhaseIncidences);
      if (guardOverflow || phaseIncidenceLimitExceeded) {
        invalidIndex = index;
        return SyncCoverProtocolError::LimitExceeded;
      }
      guardLiterals = nextGuardLiterals;
      phaseIncidences += transfer.activePhases.size();
      const bool unbalancedGuards =
          !guardImplies(*setGuard, *waitGuard, workBudget) ||
          !guardImplies(*waitGuard, *setGuard, workBudget);
      if (workBudget && workBudget->exhausted) {
        invalidIndex = index;
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
      if (unbalancedGuards ||
          (channel.flow != SyncCoverEventChannelFlow::LoopCarry &&
           *setPosition > *waitPosition)) {
        invalidIndex = index;
        return SyncCoverProtocolError::InvalidProtocol;
      }
      if (protocol.loop) {
        for (std::size_t phase : resolved.reachablePhases) {
          const SyncCoverGuard &loopGuard =
              graph.getScopes()[protocol.loop->scope].guard;
          const SyncCoverGuard &phaseGuard = resolved.guardByPhase[phase];
          std::size_t phaseGuardLiterals = 0;
          if (!checkedAdd(loopGuard.literals.size(), phaseGuard.literals.size(),
                          phaseGuardLiterals) ||
              !consumeWork(workBudget, phaseGuardLiterals)) {
            invalidIndex = index;
            return workBudget && workBudget->exhausted
                       ? SyncCoverProtocolError::WorkLimitExceeded
                       : SyncCoverProtocolError::LimitExceeded;
          }
          SyncCoverGuard phaseCondition = loopGuard;
          phaseCondition.literals.insert(phaseCondition.literals.end(),
                                         phaseGuard.literals.begin(),
                                         phaseGuard.literals.end());
          if (!normalizeGuard(phaseCondition, workBudget)) {
            invalidIndex = index;
            return workBudget && workBudget->exhausted
                       ? SyncCoverProtocolError::WorkLimitExceeded
                       : SyncCoverProtocolError::InvalidProtocol;
          }
          const bool executes =
              guardImplies(phaseCondition, *setGuard, workBudget) &&
              guardImplies(phaseCondition, *waitGuard, workBudget);
          if (!consumeWork(workBudget, logarithmicLookupWork(
                                           transfer.activePhases.size())) ||
              (workBudget && workBudget->exhausted)) {
            invalidIndex = index;
            return SyncCoverProtocolError::WorkLimitExceeded;
          }
          if (executes != phaseIsActive(transfer, phase)) {
            invalidIndex = index;
            return SyncCoverProtocolError::InvalidProtocol;
          }
        }
      }
      resolvedChannel.transfers.push_back({std::move(transfer), *setPosition,
                                           *waitPosition, *setGuard,
                                           *waitGuard});
    }
    resolved.channels.push_back(std::move(resolvedChannel));
    maximumSpan = std::max<std::size_t>(
        maximumSpan, std::max<std::size_t>(channel.width, channel.distance));
  }

  const SyncCoverProtocolError guardWorldError =
      buildBodyGuardWorlds(graph, protocol, resolved, limits, workBudget);
  if (guardWorldError != SyncCoverProtocolError::None) {
    return guardWorldError;
  }
  if (resolved.bodyGuardWorlds.size() > 1 && maximumSpan > 1) {
    // The bounded checker below proves arbitrary adjacent guard-world
    // transitions. Wider-distance guarded channels require a correspondingly
    // wider product construction and remain fail-closed.
    return SyncCoverProtocolError::InvalidProtocol;
  }
  if (resolved.bodyGuardWorlds.size() > 1 && protocol.lifetimeScope &&
      *protocol.lifetimeScope != protocol.loop->scope) {
    // Guard choices inside a repeatable child invocation need a product with
    // the enclosing invocation sequence. Keep that hierarchy fail-closed
    // until the product is represented explicitly.
    return SyncCoverProtocolError::InvalidProtocol;
  }

  std::size_t doubledPeriod = 0;
  std::size_t doubledSpan = 0;
  std::size_t periodicHorizon = 0;
  const bool horizonOverflow =
      !checkedAdd(resolved.phasePeriod, resolved.phasePeriod, doubledPeriod) ||
      !checkedAdd(maximumSpan, maximumSpan, doubledSpan) ||
      !checkedAdd(resolved.phasePreperiod, doubledPeriod, periodicHorizon) ||
      !checkedAdd(periodicHorizon, doubledSpan, resolved.verificationHorizon);
  if (horizonOverflow ||
      resolved.verificationHorizon > limits.maximumTripCounts) {
    return SyncCoverProtocolError::LimitExceeded;
  }
  for (const ResolvedChannel &channel : resolved.channels) {
    if (!channel.actions.empty()) {
      continue;
    }
    const SyncCoverProtocolError distanceError =
        validatePhaseDistances(protocol, resolved, channel, workBudget);
    if (distanceError != SyncCoverProtocolError::None) {
      invalidIndex = channel.description->id;
      return distanceError;
    }
  }
  for (std::size_t index = 0; index < protocol.rearmProofs.size(); ++index) {
    const SyncCoverProtocolRearmProof &proof = protocol.rearmProofs[index];
    if (!consumeWork(workBudget, logarithmicLookupWork(
                                     target.certifiedRearmFacts.size()))) {
      invalidIndex = index;
      return SyncCoverProtocolError::WorkLimitExceeded;
    }
    const SyncCoverProtocolTargetContract::RearmFact *fact =
        target.findRearmFact(proof.evidence);
    const bool invalidProof =
        proof.evidence == 0 || proof.iterationDistance == 0 ||
        proof.fromWaitChannel >= protocol.channels.size() ||
        proof.toSetChannel >= protocol.channels.size() || !fact;
    if (invalidProof) {
      invalidIndex = index;
      return SyncCoverProtocolError::InvalidProtocol;
    }
    const SyncCoverEventChannel &from =
        protocol.channels[proof.fromWaitChannel];
    const SyncCoverEventChannel &to = protocol.channels[proof.toSetChannel];
    std::size_t factGuardWork = 0;
    const bool factWorkOverflow =
        !checkedAdd(fact->fromWaitGuard.literals.size(),
                    from.wait.guard.literals.size(), factGuardWork) ||
        !checkedAdd(factGuardWork, fact->toSetGuard.literals.size(),
                    factGuardWork) ||
        !checkedAdd(factGuardWork, to.set.guard.literals.size(), factGuardWork);
    if (factWorkOverflow || !consumeWork(workBudget, factGuardWork)) {
      invalidIndex = index;
      return workBudget && workBudget->exhausted
                 ? SyncCoverProtocolError::WorkLimitExceeded
                 : SyncCoverProtocolError::LimitExceeded;
    }
    const bool mismatchedFact =
        fact->fromWaitResource != from.wait.resource ||
        !anchorsEqual(fact->fromWaitAnchor, from.wait.anchor) ||
        !guardsEqual(fact->fromWaitGuard, from.wait.guard) ||
        fact->toSetResource != to.set.resource ||
        !anchorsEqual(fact->toSetAnchor, to.set.anchor) ||
        !guardsEqual(fact->toSetGuard, to.set.guard) || !protocol.loop ||
        fact->loopScope != protocol.loop->scope ||
        fact->iterationDistance != proof.iterationDistance ||
        fact->width != std::min(from.width, to.width);
    if (mismatchedFact) {
      invalidIndex = index;
      return SyncCoverProtocolError::InvalidProtocol;
    }
  }
  return SyncCoverProtocolError::None;
}

SyncCoverProtocolVerificationResult
mlir::pto::sync_cover_protocol_detail::verifyProtocolAssumingValidGraph(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverEventProtocol &protocol, SyncCoverProtocolLimits limits,
    SyncCoverCoverageWorkBudget *workBudget, bool validateTarget) {
  SyncCoverProtocolVerificationResult result;
  ResolvedProtocol resolved;
  result.error =
      resolveProtocol(graph, target, protocol, limits, resolved,
                      result.invalidIndex, workBudget, false, validateTarget);
  if (result.error != SyncCoverProtocolError::None) {
    return result;
  }
  result.statistics.reachablePhases = resolved.reachablePhases.size();
  result.error = verifyResolvedProtocolAutomaton(protocol, resolved, limits,
                                                 result.statistics, workBudget,
                                                 &result.supplyWitnesses);
  if (result.error != SyncCoverProtocolError::None) {
    return result;
  }
  std::size_t exitGuardLiterals = 0;
  for (const ResolvedChannel &resolvedChannel : resolved.channels) {
    const SyncCoverEventChannel &channel = *resolvedChannel.description;
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      return result;
    }
    const bool exportCandidate =
        channel.flow == SyncCoverEventChannelFlow::LoopCarry &&
        channel.exportsCompletionAtExit;
    if (!exportCandidate) {
      continue;
    }
    std::size_t phaseLookupWork = 0;
    const bool phaseWorkOverflow = !checkedProduct(
        resolved.reachablePhases.size(),
        logarithmicLookupWork(channel.activePhases.size()), phaseLookupWork);
    if (phaseWorkOverflow || !consumeWork(workBudget, phaseLookupWork)) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = channel.id;
      return result;
    }
    const bool activeOnEveryPhase = std::all_of(
        resolved.reachablePhases.begin(), resolved.reachablePhases.end(),
        [&](std::size_t phase) {
          return std::any_of(resolvedChannel.transfers.begin(),
                             resolvedChannel.transfers.end(),
                             [&](const ResolvedTransfer &item) {
                               return phaseIsActive(item.description, phase);
                             });
        });
    if (activeOnEveryPhase) {
      const SyncCoverCutPoint &setPoint =
          resolvedChannel.transfers.front().description.set;
      std::optional<SyncCoverGuard> effectiveGuard =
          effectivePointGuard(graph, setPoint, workBudget);
      if (!effectiveGuard) {
        result.error = workBudget && workBudget->exhausted
                           ? SyncCoverProtocolError::WorkLimitExceeded
                           : SyncCoverProtocolError::InvalidProtocol;
        result.invalidIndex = channel.id;
        return result;
      }
      SyncCoverGuard exportGuard = std::move(*effectiveGuard);
      const SyncCoverGuard &loopGuard =
          graph.getScopes()[protocol.loop->scope].guard;
      const bool unconditionalWithinLoop =
          guardImplies(loopGuard, exportGuard, workBudget) &&
          guardImplies(exportGuard, loopGuard, workBudget);
      if (workBudget && workBudget->exhausted) {
        result.error = SyncCoverProtocolError::WorkLimitExceeded;
        result.invalidIndex = channel.id;
        return result;
      }
      if (!unconditionalWithinLoop) {
        continue;
      }
      const bool exportLimitReached =
          result.exitExports.size() == limits.maximumExitExports;
      const bool guardLimitExceeded =
          exportGuard.literals.size() >
          limits.maximumExitExportGuardLiterals -
              std::min(exitGuardLiterals,
                       limits.maximumExitExportGuardLiterals);
      if (exportLimitReached || guardLimitExceeded) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = channel.id;
        return result;
      }
      exitGuardLiterals += exportGuard.literals.size();
      result.exitExports.push_back(
          {channel.id, std::move(exportGuard), false, true});
    }
  }
  return result;
}

SyncCoverProtocolVerificationResult mlir::pto::verifySyncCoverEventProtocol(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverEventProtocol &protocol, SyncCoverProtocolLimits limits,
    SyncCoverCoverageWorkBudget *workBudget) {
  const bool graphLimitExceeded = !graphFitsProtocolLimits(graph, limits);
  if (graphLimitExceeded) {
    SyncCoverProtocolVerificationResult result;
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  const bool invalidGraph = !graph.isStructureFrozen() || !graph.validate();
  if (invalidGraph) {
    SyncCoverProtocolVerificationResult result;
    result.error = SyncCoverProtocolError::InvalidGraph;
    return result;
  }
  return verifyProtocolAssumingValidGraph(graph, target, protocol, limits,
                                          workBudget, true);
}
