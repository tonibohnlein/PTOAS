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

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>

using namespace mlir::pto;

namespace {

std::optional<SyncCoverTimelinePosition>
getNodePosition(const SyncCoverNode &node, bool after) {
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  const std::size_t phase = static_cast<std::size_t>(after);
  const bool orderOverflows = node.order > (maximum - phase) / 2;
  if (orderOverflows) {
    return std::nullopt;
  }
  return node.order * 2 + phase;
}

bool intervalContains(const SyncCoverTimelineInterval &outer,
                      const SyncCoverTimelineInterval &inner) {
  return outer.begin <= inner.begin && inner.end <= outer.end;
}

bool timelineFitsAncestors(const std::vector<SyncCoverScope> &scopes,
                           SyncCoverScopeId parent,
                           const SyncCoverTimelineInterval &timeline) {
  if (timeline.begin > timeline.end || parent >= scopes.size()) {
    return false;
  }
  while (true) {
    if (scopes[parent].timeline) {
      return intervalContains(*scopes[parent].timeline, timeline);
    }
    if (parent == 0) {
      return false;
    }
    parent = scopes[parent].parent;
  }
}

bool nodeFitsScopeTimeline(const std::vector<SyncCoverScope> &scopes,
                           const SyncCoverNode &node) {
  const std::optional<SyncCoverTimelinePosition> begin =
      getNodePosition(node, false);
  const std::optional<SyncCoverTimelinePosition> end =
      getNodePosition(node, true);
  if (!begin || !end || node.scope >= scopes.size()) {
    return false;
  }
  SyncCoverScopeId scope = node.scope;
  while (true) {
    if (scopes[scope].timeline) {
      const SyncCoverTimelineInterval &timeline = *scopes[scope].timeline;
      if (*begin < timeline.begin || timeline.end < *end) {
        return false;
      }
    }
    if (scope == 0) {
      break;
    }
    scope = scopes[scope].parent;
  }
  return true;
}

std::optional<SyncCoverScopeId>
getOwningTimelineScope(const std::vector<SyncCoverScope> &scopes,
                       SyncCoverScopeId scope) {
  if (scope >= scopes.size()) {
    return std::nullopt;
  }
  while (!scopes[scope].timeline) {
    if (scope == 0) {
      return std::nullopt;
    }
    scope = scopes[scope].parent;
  }
  return scope;
}

} // namespace

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

SyncCoverGraphResult
SyncCoverGraph::addScope(SyncCoverScopeId parent, bool mustExecuteWithinParent,
                         std::optional<SyncCoverTimelineInterval> timeline,
                         bool isLoop) {
  if (!hasValidScope(parent)) {
    return {SyncCoverGraphError::InvalidScope, parent};
  }
  const bool invalidTimeline =
      timeline && !timelineFitsAncestors(scopes_, parent, *timeline);
  if (invalidTimeline) {
    return {SyncCoverGraphError::InvalidTimeline, scopes_.size()};
  }
  const SyncCoverScopeId id = scopes_.size();
  scopes_.push_back({id, parent, mustExecuteWithinParent, timeline, isLoop});
  ++generation_;
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult SyncCoverGraph::addControl(unsigned alternatives,
                                                SyncCoverScopeId scope) {
  if (alternatives == 0) {
    return {SyncCoverGraphError::InvalidControl, controls_.size()};
  }
  if (!hasValidScope(scope)) {
    return {SyncCoverGraphError::InvalidScope, scope};
  }
  const SyncCoverControlId id = controls_.size();
  controls_.push_back({id, alternatives, scope});
  ++generation_;
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult
SyncCoverGraph::addNode(std::uint32_t resource, std::uint64_t weight,
                        SyncCoverScopeId scope, std::size_t order,
                        SyncCoverGuard guard,
                        std::vector<std::uint32_t> completionTargets) {
  if (!hasValidScope(scope)) {
    return {SyncCoverGraphError::InvalidScope, scope};
  }
  const SyncCoverGraphError error = normalizeAndValidateGuard(guard, scope);
  if (error != SyncCoverGraphError::None) {
    return {error, nodes_.size()};
  }
  std::sort(completionTargets.begin(), completionTargets.end());
  completionTargets.erase(
      std::unique(completionTargets.begin(), completionTargets.end()),
      completionTargets.end());
  const SyncCoverNodeId id = nodes_.size();
  SyncCoverNode node{id,
                     resource,
                     weight,
                     scope,
                     order,
                     std::move(guard),
                     std::move(completionTargets)};
  if (!nodeFitsScopeTimeline(scopes_, node)) {
    return {SyncCoverGraphError::InvalidTimeline, id};
  }
  const std::optional<SyncCoverScopeId> timeline =
      getOwningTimelineScope(scopes_, scope);
  const bool duplicateOrder =
      timeline && std::any_of(nodes_.begin(), nodes_.end(),
                             [&](const SyncCoverNode &existing) {
                               return getOwningTimelineScope(
                                          scopes_, existing.scope) ==
                                          timeline &&
                                      existing.order == order;
                             });
  if (duplicateOrder) {
    return {SyncCoverGraphError::InvalidOrder, id};
  }
  nodes_.push_back(std::move(node));
  ++generation_;
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult SyncCoverGraph::addEdge(SyncCoverEdge edge) {
  const SyncCoverGraphResult prepared = prepareEdge(edge);
  if (!prepared) {
    return prepared;
  }
  const std::size_t index = edges_.size();
  edges_.push_back(std::move(edge));
  ++generation_;
  return {SyncCoverGraphError::None, index};
}

SyncCoverGraphResult SyncCoverGraph::prepareEdge(SyncCoverEdge &edge) const {
  const bool invalidNode =
      edge.source >= nodes_.size() || edge.target >= nodes_.size();
  if (invalidNode) {
    return {SyncCoverGraphError::InvalidNode, edges_.size()};
  }
  const bool invalidScope =
      !hasValidScope(edge.scope) ||
      !scopeContains(edge.scope, nodes_[edge.source].scope) ||
      !scopeContains(edge.scope, nodes_[edge.target].scope);
  if (invalidScope) {
    return {SyncCoverGraphError::InvalidScope, edges_.size()};
  }
  if (edge.distance != 0 && (edge.scope == 0 || !scopes_[edge.scope].isLoop)) {
    return {SyncCoverGraphError::InvalidDistance, edges_.size()};
  }
  if (edge.distance == 0 && edge.source == edge.target) {
    return {SyncCoverGraphError::ZeroDistanceSelfEdge, edges_.size()};
  }
  if (edge.distance == 0 &&
      nodes_[edge.source].order >= nodes_[edge.target].order) {
    return {SyncCoverGraphError::InvalidOrder, edges_.size()};
  }
  const SyncCoverGraphError error =
      completeEndpointGuards(edge.source, edge.target, edge.scope,
                             edge.distance, edge.sourceGuard, edge.targetGuard);
  if (error != SyncCoverGraphError::None) {
    return {error, edges_.size()};
  }
  return {SyncCoverGraphError::None, edges_.size()};
}

SyncCoverGraphResult SyncCoverGraph::addDemand(SyncCoverDemand demand) {
  const bool invalidNode =
      demand.source >= nodes_.size() || demand.target >= nodes_.size();
  if (invalidNode) {
    return {SyncCoverGraphError::InvalidNode, demands_.size()};
  }
  const bool invalidScope =
      !hasValidScope(demand.scope) ||
      !scopeContains(demand.scope, nodes_[demand.source].scope) ||
      !scopeContains(demand.scope, nodes_[demand.target].scope);
  if (invalidScope) {
    return {SyncCoverGraphError::InvalidScope, demands_.size()};
  }
  if (demand.distance != 0 &&
      (demand.scope == 0 || !scopes_[demand.scope].isLoop)) {
    return {SyncCoverGraphError::InvalidDistance, demands_.size()};
  }
  if (demand.distance == 0 && demand.source == demand.target) {
    return {SyncCoverGraphError::ZeroDistanceSelfDemand, demands_.size()};
  }
  if (demand.distance == 0 &&
      nodes_[demand.source].order >= nodes_[demand.target].order) {
    return {SyncCoverGraphError::InvalidOrder, demands_.size()};
  }
  const SyncCoverGraphError error = completeEndpointGuards(
      demand.source, demand.target, demand.scope, demand.distance,
      demand.sourceGuard, demand.targetGuard);
  if (error != SyncCoverGraphError::None) {
    return {error, demands_.size()};
  }
  const std::size_t index = demands_.size();
  demands_.push_back(std::move(demand));
  ++generation_;
  return {SyncCoverGraphError::None, index};
}

SyncCoverGraphResult SyncCoverGraph::validate() const {
  const bool invalidRoot = scopes_.empty() || scopes_.front().id != 0 ||
                           scopes_.front().parent != 0 ||
                           !scopes_.front().timeline || scopes_.front().isLoop;
  if (invalidRoot) {
    return {SyncCoverGraphError::InvalidScope, 0};
  }
  for (std::size_t index = 1; index < scopes_.size(); ++index) {
    const SyncCoverScope &scope = scopes_[index];
    const bool invalidIdentity = scope.id != index || scope.parent >= index ||
                                 !hasValidScope(scope.parent);
    if (invalidIdentity) {
      return {SyncCoverGraphError::InvalidScope, index};
    }
    const bool invalidTimeline =
        scope.timeline &&
        !timelineFitsAncestors(scopes_, scope.parent, *scope.timeline);
    if (invalidTimeline) {
      return {SyncCoverGraphError::InvalidTimeline, index};
    }
  }
  for (std::size_t index = 0; index < controls_.size(); ++index) {
    const SyncCoverControl &control = controls_[index];
    const bool invalidControl = control.id != index ||
                                control.alternatives == 0 ||
                                !hasValidScope(control.scope);
    if (invalidControl) {
      return {SyncCoverGraphError::InvalidControl, index};
    }
  }
  std::set<std::pair<SyncCoverScopeId, std::size_t>> timelineOrders;
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    const SyncCoverNode &node = nodes_[index];
    if (!hasValidScope(node.scope)) {
      return {SyncCoverGraphError::InvalidScope, index};
    }
    const bool invalidTimeline = !nodeFitsScopeTimeline(scopes_, node);
    const bool invalidTargets =
        !std::is_sorted(node.completionTargets.begin(),
                        node.completionTargets.end()) ||
        std::adjacent_find(node.completionTargets.begin(),
                           node.completionTargets.end()) !=
            node.completionTargets.end();
    if (invalidTimeline) {
      return {SyncCoverGraphError::InvalidTimeline, index};
    }
    if (invalidTargets) {
      return {SyncCoverGraphError::InvalidCompletionTargets, index};
    }
    const std::optional<SyncCoverScopeId> timeline =
        getOwningTimelineScope(scopes_, node.scope);
    if (!timeline || !timelineOrders.insert({*timeline, node.order}).second) {
      return {SyncCoverGraphError::InvalidOrder, index};
    }
    SyncCoverGuard guard = node.guard;
    const SyncCoverGraphError error =
        normalizeAndValidateGuard(guard, node.scope);
    if (error != SyncCoverGraphError::None) {
      return {error, index};
    }
  }
  for (std::size_t index = 0; index < demands_.size(); ++index) {
    const SyncCoverDemand &demand = demands_[index];
    const bool invalidNode =
        demand.source >= nodes_.size() || demand.target >= nodes_.size();
    if (invalidNode) {
      return {SyncCoverGraphError::InvalidNode, index};
    }
    const bool invalidScope =
        !hasValidScope(demand.scope) ||
        !scopeContains(demand.scope, nodes_[demand.source].scope) ||
        !scopeContains(demand.scope, nodes_[demand.target].scope);
    if (invalidScope) {
      return {SyncCoverGraphError::InvalidScope, index};
    }
    if (demand.distance != 0 &&
        (demand.scope == 0 || !scopes_[demand.scope].isLoop)) {
      return {SyncCoverGraphError::InvalidDistance, index};
    }
    if (demand.distance == 0 && demand.source == demand.target) {
      return {SyncCoverGraphError::ZeroDistanceSelfDemand, index};
    }
    if (demand.distance == 0 &&
        nodes_[demand.source].order >= nodes_[demand.target].order) {
      return {SyncCoverGraphError::InvalidOrder, index};
    }
    SyncCoverGuard sourceGuard = demand.sourceGuard;
    SyncCoverGuard targetGuard = demand.targetGuard;
    const SyncCoverGraphError error =
        completeEndpointGuards(demand.source, demand.target, demand.scope,
                               demand.distance, sourceGuard, targetGuard);
    if (error != SyncCoverGraphError::None) {
      return {error, index};
    }
  }

  std::vector<std::vector<SyncCoverNodeId>> children(nodes_.size());
  std::vector<std::size_t> indegrees(nodes_.size(), 0);
  for (std::size_t index = 0; index < edges_.size(); ++index) {
    const SyncCoverEdge &edge = edges_[index];
    const bool invalidNode =
        edge.source >= nodes_.size() || edge.target >= nodes_.size();
    if (invalidNode) {
      return {SyncCoverGraphError::InvalidNode, index};
    }
    const bool invalidScope =
        !hasValidScope(edge.scope) ||
        !scopeContains(edge.scope, nodes_[edge.source].scope) ||
        !scopeContains(edge.scope, nodes_[edge.target].scope);
    if (invalidScope) {
      return {SyncCoverGraphError::InvalidScope, index};
    }
    if (edge.distance != 0 &&
        (edge.scope == 0 || !scopes_[edge.scope].isLoop)) {
      return {SyncCoverGraphError::InvalidDistance, index};
    }
    SyncCoverGuard sourceGuard = edge.sourceGuard;
    SyncCoverGuard targetGuard = edge.targetGuard;
    const SyncCoverGraphError error =
        completeEndpointGuards(edge.source, edge.target, edge.scope,
                               edge.distance, sourceGuard, targetGuard);
    if (error != SyncCoverGraphError::None) {
      return {error, index};
    }
    if (edge.distance != 0) {
      continue;
    }
    if (edge.source == edge.target) {
      return {SyncCoverGraphError::ZeroDistanceSelfEdge, index};
    }
    if (nodes_[edge.source].order >= nodes_[edge.target].order) {
      return {SyncCoverGraphError::InvalidOrder, index};
    }
    children[edge.source].push_back(edge.target);
    ++indegrees[edge.target];
  }

  std::vector<SyncCoverNodeId> ready;
  for (SyncCoverNodeId node = 0; node < nodes_.size(); ++node) {
    if (indegrees[node] == 0) {
      ready.push_back(node);
    }
  }
  std::size_t visited = 0;
  for (std::size_t index = 0; index < ready.size(); ++index) {
    ++visited;
    for (SyncCoverNodeId child : children[ready[index]]) {
      if (--indegrees[child] == 0) {
        ready.push_back(child);
      }
    }
  }
  if (visited != nodes_.size()) {
    return {SyncCoverGraphError::ZeroDistanceCycle, std::nullopt};
  }
  return {SyncCoverGraphError::None, std::nullopt};
}

bool SyncCoverGraph::hasValidScope(SyncCoverScopeId scope) const {
  return scope < scopes_.size() && scopes_[scope].id == scope;
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

bool SyncCoverGraph::scopeExecutesWhen(SyncCoverScopeId conditionScope,
                                       SyncCoverScopeId requiredScope) const {
  if (scopeContains(requiredScope, conditionScope)) {
    return true;
  }
  return scopeMustExecuteWithin(conditionScope, requiredScope);
}

std::optional<SyncCoverScopeId>
SyncCoverGraph::getLowestCommonScope(SyncCoverScopeId first,
                                     SyncCoverScopeId second) const {
  const bool firstValid = hasValidScope(first);
  const bool secondValid = hasValidScope(second);
  if (!firstValid || !secondValid) {
    return std::nullopt;
  }
  while (!scopeContains(first, second)) {
    if (first == 0) {
      return std::nullopt;
    }
    first = scopes_[first].parent;
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
  bool first = true;
  while (scope != 0) {
    if (scopes_[scope].isLoop && (includeScope || !first)) {
      ++depth;
    }
    first = false;
    scope = scopes_[scope].parent;
  }
  return depth;
}

SyncCoverGraph::EdgeCheckpoint SyncCoverGraph::checkpointEdges() const {
  return {edges_.size()};
}

void SyncCoverGraph::rollbackEdges(EdgeCheckpoint checkpoint) {
  edges_.resize(checkpoint.edgeCount);
  ++generation_;
}

bool SyncCoverGraph::reserveAdditionalEdges(std::size_t additionalEdges) {
  const bool overflow = additionalEdges > edges_.max_size() - edges_.size();
  if (overflow) {
    return false;
  }
  edges_.reserve(edges_.size() + additionalEdges);
  return true;
}

SyncCoverGraph::EdgeTransaction::EdgeTransaction(SyncCoverGraph &graph)
    : graph_(graph), checkpoint_(graph.checkpointEdges()) {}

SyncCoverGraph::EdgeTransaction::~EdgeTransaction() {
  if (!committed_) {
    graph_.rollbackEdges(checkpoint_);
  }
}

void SyncCoverGraph::EdgeTransaction::append(SyncCoverEdge edge) {
  static_assert(std::is_nothrow_move_constructible<SyncCoverEdge>::value,
                "reserved edge commit must not throw");
  graph_.edges_.push_back(std::move(edge));
  ++graph_.generation_;
}

void SyncCoverGraph::EdgeTransaction::commit() { committed_ = true; }

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
  const std::vector<SyncCoverNode> &nodes = graph.getNodes();
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  switch (anchor.kind) {
  case SyncCoverAnchorKind::BeforeNode:
  case SyncCoverAnchorKind::AfterNode: {
    const bool invalidNodeAnchor =
        anchor.node >= nodes.size() || anchor.scope != 0 ||
        anchor.position != 0;
    if (invalidNodeAnchor) {
      return std::nullopt;
    }
    return getNodePosition(nodes[anchor.node],
                           anchor.kind == SyncCoverAnchorKind::AfterNode);
  }
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit: {
    const bool invalidScopeAnchor = anchor.scope >= scopes.size() ||
                                    anchor.node != 0 || anchor.position != 0 ||
                                    !scopes[anchor.scope].timeline;
    if (invalidScopeAnchor) {
      return std::nullopt;
    }
    return anchor.kind == SyncCoverAnchorKind::ScopeEntry
               ? scopes[anchor.scope].timeline->begin
               : scopes[anchor.scope].timeline->end;
  }
  case SyncCoverAnchorKind::TimelinePoint: {
    const bool invalidPoint = anchor.scope >= scopes.size() ||
                              anchor.node != 0 ||
                              !scopes[anchor.scope].timeline ||
                              anchor.position <
                                  scopes[anchor.scope].timeline->begin ||
                              anchor.position >
                                  scopes[anchor.scope].timeline->end;
    return invalidPoint
               ? std::nullopt
               : std::optional<SyncCoverTimelinePosition>(anchor.position);
  }
  }
  return std::nullopt;
}

bool mlir::pto::syncCoverNodeCanProduceCompletion(
    const SyncCoverGraph &graph, SyncCoverNodeId node,
    std::uint32_t targetResource) {
  if (node >= graph.getNodes().size()) {
    return false;
  }
  const std::vector<std::uint32_t> &targets =
      graph.getNodes()[node].completionTargets;
  return std::binary_search(targets.begin(), targets.end(), targetResource);
}
