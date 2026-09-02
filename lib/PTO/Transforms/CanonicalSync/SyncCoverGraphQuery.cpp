// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include "SyncCoverGraphInternal.h"

#include <algorithm>
#include <tuple>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_detail;

bool SyncCoverGuardLiteral::operator<(
    const SyncCoverGuardLiteral &other) const {
  return std::tie(control, alternative) <
         std::tie(other.control, other.alternative);
}

bool SyncCoverGuardLiteral::operator==(
    const SyncCoverGuardLiteral &other) const {
  return control == other.control && alternative == other.alternative;
}

bool mlir::pto::normalizeSyncCoverGuard(SyncCoverGuard &guard) {
  std::sort(guard.literals.begin(), guard.literals.end());
  guard.literals.erase(
      std::unique(guard.literals.begin(), guard.literals.end()),
      guard.literals.end());
  for (std::size_t index = 1; index < guard.literals.size(); ++index) {
    const SyncCoverGuardLiteral &previous = guard.literals[index - 1];
    const SyncCoverGuardLiteral &current = guard.literals[index];
    if (previous.control == current.control &&
        previous.alternative != current.alternative) {
      return false;
    }
  }
  return true;
}

bool mlir::pto::syncCoverGuardImplies(const SyncCoverGuard &condition,
                                      const SyncCoverGuard &required) {
  SyncCoverGuard normalizedCondition = condition;
  SyncCoverGuard normalizedRequired = required;
  const bool conditionValid = normalizeSyncCoverGuard(normalizedCondition);
  const bool requiredValid = normalizeSyncCoverGuard(normalizedRequired);
  if (!conditionValid || !requiredValid) {
    return false;
  }
  return std::includes(
      normalizedCondition.literals.begin(), normalizedCondition.literals.end(),
      normalizedRequired.literals.begin(), normalizedRequired.literals.end());
}

bool mlir::pto::syncCoverGuardsCompatible(const SyncCoverGuard &first,
                                          const SyncCoverGuard &second) {
  SyncCoverGuard normalizedFirst = first;
  SyncCoverGuard normalizedSecond = second;
  const bool firstValid = normalizeSyncCoverGuard(normalizedFirst);
  const bool secondValid = normalizeSyncCoverGuard(normalizedSecond);
  if (!firstValid || !secondValid) {
    return false;
  }

  std::size_t firstIndex = 0;
  std::size_t secondIndex = 0;
  while (firstIndex < normalizedFirst.literals.size()) {
    if (secondIndex >= normalizedSecond.literals.size()) {
      break;
    }
    const SyncCoverGuardLiteral &firstLiteral =
        normalizedFirst.literals[firstIndex];
    const SyncCoverGuardLiteral &secondLiteral =
        normalizedSecond.literals[secondIndex];
    if (firstLiteral.control < secondLiteral.control) {
      ++firstIndex;
      continue;
    }
    if (secondLiteral.control < firstLiteral.control) {
      ++secondIndex;
      continue;
    }
    if (firstLiteral.alternative != secondLiteral.alternative) {
      return false;
    }
    ++firstIndex;
    ++secondIndex;
  }
  return true;
}

bool SyncCoverGraph::isGuardReachable(const SyncCoverGuard &input) const {
  SyncCoverGuard guard = input;
  if (!normalizeSyncCoverGuard(guard)) {
    return false;
  }
  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    if (literal.control >= controls_.size() ||
        literal.alternative >= controls_[literal.control].alternatives) {
      return false;
    }
  }

  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    const SyncCoverControl &firstControl = controls_[literal.control];
    if (!firstControl.firstIterationRelation ||
        literal.alternative !=
            firstControl.firstIterationRelation->firstIterationAlternative) {
      continue;
    }
    const SyncCoverScopeId loopScope =
        firstControl.firstIterationRelation->loopScope;
    for (const SyncCoverGuardLiteral &phaseLiteral : guard.literals) {
      const SyncCoverControl &phaseControl = controls_[phaseLiteral.control];
      if (!phaseControl.phaseRelation ||
          phaseControl.phaseRelation->loopScope != loopScope) {
        continue;
      }
      const SyncCoverControlPhaseRelation &phase = *phaseControl.phaseRelation;
      if (phase.initialPhase >= phase.activeAlternative.size() ||
          phaseLiteral.alternative !=
              phase.activeAlternative[phase.initialPhase]) {
        return false;
      }
    }
  }
  return true;
}

bool mlir::pto::syncCoverEndpointsCoExecute(const SyncCoverGraph &graph,
                                            const SyncCoverEdge &edge) {
  const bool invalidEndpoint = edge.source >= graph.getNodes().size() ||
                               edge.target >= graph.getNodes().size();
  if (invalidEndpoint) {
    return false;
  }
  const SyncCoverNode &source = graph.getNodes()[edge.source];
  const SyncCoverNode &target = graph.getNodes()[edge.target];
  const std::optional<SyncCoverScopeId> common =
      graph.getLowestCommonScope(source.scope, target.scope);
  return common && graph.scopeMustExecuteWithin(*common, source.scope) &&
         graph.scopeMustExecuteWithin(*common, target.scope) &&
         syncCoverGuardImplies(edge.targetGuard, edge.sourceGuard) &&
         syncCoverGuardImplies(edge.sourceGuard, edge.targetGuard);
}

bool SyncCoverGraph::hasValidScope(SyncCoverScopeId scope) const {
  return scope < scopes_.size() && scopes_[scope].id == scope;
}

bool SyncCoverGraph::hasValidRegion(SyncCoverRegionId region) const {
  return region < regions_.size() && regions_[region].id == region;
}

bool SyncCoverGraph::scopeContains(SyncCoverScopeId ancestor,
                                   SyncCoverScopeId descendant) const {
  const bool invalidScope =
      !hasValidScope(ancestor) || !hasValidScope(descendant);
  if (invalidScope) {
    return false;
  }
  while (descendant != ancestor && descendant != 0) {
    descendant = scopes_[descendant].parent;
  }
  return descendant == ancestor;
}

bool SyncCoverGraph::regionContains(SyncCoverRegionId ancestor,
                                    SyncCoverRegionId descendant) const {
  const bool invalidRegion =
      !hasValidRegion(ancestor) || !hasValidRegion(descendant);
  if (invalidRegion) {
    return false;
  }
  while (descendant != ancestor && descendant != 0) {
    descendant = regions_[descendant].parent;
  }
  return descendant == ancestor;
}

bool SyncCoverGraph::scopeMustExecuteWithin(SyncCoverScopeId ancestor,
                                            SyncCoverScopeId descendant) const {
  if (!scopeContains(ancestor, descendant)) {
    return false;
  }
  while (descendant != ancestor) {
    if (!scopes_[descendant].mustExecuteWithinParent) {
      return false;
    }
    descendant = scopes_[descendant].parent;
  }
  return true;
}

bool SyncCoverGraph::completionDominates(SyncCoverNodeId completionNode,
                                         SyncCoverNodeId source) const {
  const bool invalidNode =
      completionNode >= nodes_.size() || source >= nodes_.size();
  if (invalidNode) {
    return false;
  }
  std::vector<std::uint8_t> visited(nodes_.size(), 0);
  std::vector<SyncCoverNodeId> ready{completionNode};
  while (!ready.empty()) {
    const SyncCoverNodeId current = ready.back();
    ready.pop_back();
    if (current == source) {
      return true;
    }
    if (visited[current] != 0) {
      continue;
    }
    visited[current] = 1;
    const std::vector<SyncCoverNodeId> &predecessors =
        nodes_[current].completionDominatedSources;
    ready.insert(ready.end(), predecessors.begin(), predecessors.end());
  }
  return false;
}

std::optional<SyncCoverScopeId>
SyncCoverGraph::getLowestCommonScope(SyncCoverScopeId first,
                                     SyncCoverScopeId second) const {
  const bool invalidScope = !hasValidScope(first) || !hasValidScope(second);
  if (invalidScope) {
    return std::nullopt;
  }

  // Align the two parent-chain depths and then ascend them together. Each
  // chain is traversed at most twice, so callers can reserve a portable 4*S
  // upper bound instead of relying on the former nested ancestry search.
  const auto getDepth = [&](SyncCoverScopeId scope) {
    std::size_t depth = 0;
    while (scope != 0) {
      ++depth;
      scope = scopes_[scope].parent;
    }
    return depth;
  };
  std::size_t firstDepth = getDepth(first);
  std::size_t secondDepth = getDepth(second);
  while (firstDepth > secondDepth) {
    first = scopes_[first].parent;
    --firstDepth;
  }
  while (secondDepth > firstDepth) {
    second = scopes_[second].parent;
    --secondDepth;
  }
  while (first != second) {
    first = scopes_[first].parent;
    second = scopes_[second].parent;
  }
  return first;
}

std::optional<SyncCoverRegionId>
SyncCoverGraph::getLowestCommonRegion(SyncCoverRegionId first,
                                      SyncCoverRegionId second) const {
  const bool invalidRegion = !hasValidRegion(first) || !hasValidRegion(second);
  if (invalidRegion) {
    return std::nullopt;
  }
  const auto depth = [&](SyncCoverRegionId region) {
    std::size_t result = 0;
    while (region != 0) {
      ++result;
      region = regions_[region].parent;
    }
    return result;
  };
  std::size_t firstDepth = depth(first);
  std::size_t secondDepth = depth(second);
  while (firstDepth > secondDepth) {
    first = regions_[first].parent;
    --firstDepth;
  }
  while (secondDepth > firstDepth) {
    second = regions_[second].parent;
    --secondDepth;
  }
  while (first != second) {
    first = regions_[first].parent;
    second = regions_[second].parent;
  }
  return first;
}

std::optional<std::size_t>
SyncCoverGraph::getScopeLoopDepth(SyncCoverScopeId scope,
                                  bool includeScope) const {
  if (!hasValidScope(scope)) {
    return std::nullopt;
  }
  std::size_t depth = 0;
  if (!includeScope && scope != 0) {
    scope = scopes_[scope].parent;
  }
  while (true) {
    depth += scopes_[scope].isLoop ? 1 : 0;
    if (scope == 0) {
      break;
    }
    scope = scopes_[scope].parent;
  }
  return depth;
}

std::optional<SyncCoverScopeId>
SyncCoverGraph::getNearestEnclosingLoop(SyncCoverScopeId scope,
                                        bool includeScope) const {
  if (!hasValidScope(scope)) {
    return std::nullopt;
  }
  if (!includeScope && scope != 0) {
    scope = scopes_[scope].parent;
  }
  while (scope != 0 && !scopes_[scope].isLoop) {
    scope = scopes_[scope].parent;
  }
  return scopes_[scope].isLoop ? std::optional<SyncCoverScopeId>(scope)
                               : std::nullopt;
}

std::optional<SyncCoverScopeId>
SyncCoverGraph::getOwningTimelineScope(SyncCoverScopeId scope) const {
  if (!hasValidScope(scope)) {
    return std::nullopt;
  }
  while (!scopes_[scope].timeline) {
    if (scope == 0) {
      return std::nullopt;
    }
    scope = scopes_[scope].parent;
  }
  return scope;
}

SyncCoverGraphError SyncCoverGraph::normalizeAndValidateGuard(
    SyncCoverGuard &guard, SyncCoverScopeId occurrenceScope) const {
  if (!normalizeSyncCoverGuard(guard)) {
    return SyncCoverGraphError::InvalidGuard;
  }
  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    const bool invalidLiteral =
        literal.control >= controls_.size() ||
        literal.alternative >= controls_[literal.control].alternatives;
    if (invalidLiteral) {
      return SyncCoverGraphError::InvalidControl;
    }
    if (!scopeContains(controls_[literal.control].scope, occurrenceScope)) {
      return SyncCoverGraphError::InvalidScope;
    }
  }
  return SyncCoverGraphError::None;
}

SyncCoverGraphError SyncCoverGraph::completeEndpointGuards(
    SyncCoverNodeId source, SyncCoverNodeId target,
    SyncCoverScopeId recurrenceScope, unsigned distance,
    SyncCoverGuard &sourceGuard, SyncCoverGuard &targetGuard) const {
  sourceGuard.literals.insert(sourceGuard.literals.end(),
                              nodes_[source].guard.literals.begin(),
                              nodes_[source].guard.literals.end());
  targetGuard.literals.insert(targetGuard.literals.end(),
                              nodes_[target].guard.literals.begin(),
                              nodes_[target].guard.literals.end());
  SyncCoverGraphError error =
      normalizeAndValidateGuard(sourceGuard, nodes_[source].scope);
  if (error != SyncCoverGraphError::None) {
    return error;
  }
  error = normalizeAndValidateGuard(targetGuard, nodes_[target].scope);
  if (error != SyncCoverGraphError::None) {
    return error;
  }

  std::size_t sourceIndex = 0;
  std::size_t targetIndex = 0;
  while (sourceIndex < sourceGuard.literals.size()) {
    if (targetIndex >= targetGuard.literals.size()) {
      break;
    }
    const SyncCoverGuardLiteral &sourceLiteral =
        sourceGuard.literals[sourceIndex];
    const SyncCoverGuardLiteral &targetLiteral =
        targetGuard.literals[targetIndex];
    if (sourceLiteral.control < targetLiteral.control) {
      ++sourceIndex;
      continue;
    }
    if (targetLiteral.control < sourceLiteral.control) {
      ++targetIndex;
      continue;
    }
    const SyncCoverControl &control = controls_[sourceLiteral.control];
    const bool contextualized =
        distance != 0 && scopeContains(recurrenceScope, control.scope);
    if (!contextualized &&
        sourceLiteral.alternative != targetLiteral.alternative) {
      return SyncCoverGraphError::IncompatibleEndpoints;
    }
    ++sourceIndex;
    ++targetIndex;
  }
  return SyncCoverGraphError::None;
}

std::optional<SyncCoverTimelinePosition>
mlir::pto::resolveSyncCoverAnchor(const SyncCoverGraph &graph,
                                  const SyncCoverAnchor &anchor) {
  const auto &nodes = graph.getNodes();
  const auto &scopes = graph.getScopes();
  switch (anchor.kind) {
  case SyncCoverAnchorKind::BeforeNode:
  case SyncCoverAnchorKind::AfterNode: {
    if (anchor.node >= nodes.size()) {
      return std::nullopt;
    }
    const SyncCoverNodeId physicalEntry = nodes[anchor.node].physicalAnchor;
    const SyncCoverNodeId physicalExit = nodes[anchor.node].physicalExit;
    const SyncCoverNodeId physicalNode =
        anchor.kind == SyncCoverAnchorKind::BeforeNode ? physicalEntry
                                                       : physicalExit;
    const std::optional<SyncCoverTimelineInterval> nodeAnchors =
        getNodeAnchorInterval(nodes[physicalNode].order);
    const std::optional<SyncCoverScopeId> timelineScope =
        graph.getOwningTimelineScope(nodes[physicalNode].scope);
    if (!nodeAnchors || !timelineScope) {
      return std::nullopt;
    }
    const SyncCoverTimelinePosition position =
        anchor.kind == SyncCoverAnchorKind::BeforeNode ? nodeAnchors->begin
                                                       : nodeAnchors->end;
    const SyncCoverTimelineInterval &timeline =
        *scopes[*timelineScope].timeline;
    if (position < timeline.begin || position > timeline.end) {
      return std::nullopt;
    }
    return position;
  }
  case SyncCoverAnchorKind::ControlEntry:
  case SyncCoverAnchorKind::ControlExit: {
    if (anchor.node >= graph.getControls().size()) {
      return std::nullopt;
    }
    const SyncCoverControl &control = graph.getControls()[anchor.node];
    if (anchor.scope != control.scope) {
      return std::nullopt;
    }
    std::optional<SyncCoverTimelinePosition> entry;
    std::optional<SyncCoverTimelinePosition> exit;
    for (const SyncCoverNode &node : nodes) {
      const auto literal =
          std::find_if(node.guard.literals.begin(), node.guard.literals.end(),
                       [&](const SyncCoverGuardLiteral &candidate) {
                         return candidate.control == control.id;
                       });
      const bool invalidLiteral = literal == node.guard.literals.end() ||
                                  literal->alternative >= control.alternatives;
      if (invalidLiteral) {
        continue;
      }
      const std::optional<SyncCoverTimelineInterval> interval =
          getNodeAnchorInterval(node.order);
      if (!interval) {
        return std::nullopt;
      }
      entry = entry ? std::min(*entry, interval->begin) : interval->begin;
      exit = exit ? std::max(*exit, interval->end) : interval->end;
    }
    // Empty alternatives contain no scheduled pipeline operation, but the
    // enclosing structured operation still provides a physical before/after
    // insertion boundary. The represented alternatives therefore suffice to
    // define the graph timeline extent; requiring a node in every alternative
    // incorrectly made opaque choice cuts unavailable for an empty `else`.
    return anchor.kind == SyncCoverAnchorKind::ControlEntry ? entry : exit;
  }
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit: {
    if (anchor.scope >= scopes.size()) {
      return std::nullopt;
    }
    if (scopes[anchor.scope].timeline) {
      return anchor.kind == SyncCoverAnchorKind::ScopeEntry
                 ? scopes[anchor.scope].timeline->begin
                 : scopes[anchor.scope].timeline->end;
    }
    const std::optional<SyncCoverScopeId> timelineScope =
        graph.getOwningTimelineScope(anchor.scope);
    if (!timelineScope) {
      return std::nullopt;
    }
    std::optional<SyncCoverTimelinePosition> first;
    std::optional<SyncCoverTimelinePosition> last;
    for (const SyncCoverNode &node : nodes) {
      if (!graph.scopeContains(anchor.scope, node.scope)) {
        continue;
      }
      const std::optional<SyncCoverTimelineInterval> interval =
          getNodeAnchorInterval(node.order);
      if (!interval) {
        return std::nullopt;
      }
      first = first ? std::min(*first, interval->begin) : interval->begin;
      last = last ? std::max(*last, interval->end) : interval->end;
    }
    if (!first || !last) {
      return std::nullopt;
    }
    const SyncCoverTimelineInterval &timeline =
        *scopes[*timelineScope].timeline;
    const SyncCoverTimelinePosition position =
        anchor.kind == SyncCoverAnchorKind::ScopeEntry ? *first : *last;
    return position >= timeline.begin && position <= timeline.end
               ? std::optional<SyncCoverTimelinePosition>(position)
               : std::nullopt;
  }
  case SyncCoverAnchorKind::LoopBodyEntry:
  case SyncCoverAnchorKind::LoopBodyExit:
    if (anchor.scope >= scopes.size() || !scopes[anchor.scope].isLoop ||
        !scopes[anchor.scope].timeline) {
      return std::nullopt;
    }
    return anchor.kind == SyncCoverAnchorKind::LoopBodyEntry
               ? scopes[anchor.scope].timeline->begin
               : scopes[anchor.scope].timeline->end;
  case SyncCoverAnchorKind::TimelinePoint:
    if (anchor.scope >= scopes.size()) {
      return std::nullopt;
    }
    if (const std::optional<SyncCoverScopeId> timelineScope =
            graph.getOwningTimelineScope(anchor.scope)) {
      const SyncCoverTimelineInterval &timeline =
          *scopes[*timelineScope].timeline;
      if (anchor.position >= timeline.begin &&
          anchor.position <= timeline.end) {
        return anchor.position;
      }
    }
    return std::nullopt;
  }
  return std::nullopt;
}

bool mlir::pto::syncCoverNodeCanProduceCompletion(
    const SyncCoverGraph &graph, SyncCoverNodeId node,
    std::uint32_t targetResource) {
  if (node >= graph.getNodes().size()) {
    return false;
  }
  const auto &targets = graph.getNodes()[node].completionTargets;
  return std::binary_search(targets.begin(), targets.end(), targetResource);
}
