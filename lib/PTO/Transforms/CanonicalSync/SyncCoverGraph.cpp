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
#include <tuple>
#include <utility>

using namespace mlir::pto;

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

SyncCoverGraphResult SyncCoverGraph::addScope(SyncCoverScopeId parent,
                                              bool mustExecuteWithinParent) {
  if (!hasValidScope(parent)) {
    return {SyncCoverGraphError::InvalidScope, parent};
  }
  const SyncCoverScopeId id = scopes_.size();
  scopes_.push_back({id, parent, mustExecuteWithinParent});
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
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult SyncCoverGraph::addNode(std::uint32_t resource,
                                             std::uint64_t weight,
                                             SyncCoverScopeId scope,
                                             std::size_t order,
                                             SyncCoverGuard guard) {
  if (!hasValidScope(scope)) {
    return {SyncCoverGraphError::InvalidScope, scope};
  }
  const SyncCoverGraphError error = normalizeAndValidateGuard(guard, scope);
  if (error != SyncCoverGraphError::None) {
    return {error, nodes_.size()};
  }
  const SyncCoverNodeId id = nodes_.size();
  nodes_.push_back({id, resource, weight, scope, order, std::move(guard)});
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult SyncCoverGraph::addEdge(SyncCoverEdge edge) {
  const bool invalidNode =
      edge.source >= nodes_.size() || edge.target >= nodes_.size();
  if (invalidNode) {
    return {SyncCoverGraphError::InvalidNode, edges_.size()};
  }
  const bool invalidScope =
      !hasValidScope(edge.scope) ||
      !isScopeAncestor(edge.scope, nodes_[edge.source].scope) ||
      !isScopeAncestor(edge.scope, nodes_[edge.target].scope);
  if (invalidScope) {
    return {SyncCoverGraphError::InvalidScope, edges_.size()};
  }
  if (edge.distance != 0 && edge.scope == 0) {
    return {SyncCoverGraphError::InvalidDistance, edges_.size()};
  }
  if (edge.distance == 0 && edge.source == edge.target) {
    return {SyncCoverGraphError::ZeroDistanceSelfEdge, edges_.size()};
  }
  const SyncCoverGraphError error =
      completeEndpointGuards(edge.source, edge.target, edge.scope,
                             edge.distance, edge.sourceGuard, edge.targetGuard);
  if (error != SyncCoverGraphError::None) {
    return {error, edges_.size()};
  }
  const std::size_t index = edges_.size();
  edges_.push_back(std::move(edge));
  return {SyncCoverGraphError::None, index};
}

SyncCoverGraphResult SyncCoverGraph::addDemand(SyncCoverDemand demand) {
  const bool invalidNode =
      demand.source >= nodes_.size() || demand.target >= nodes_.size();
  if (invalidNode) {
    return {SyncCoverGraphError::InvalidNode, demands_.size()};
  }
  const bool invalidScope =
      !hasValidScope(demand.scope) ||
      !isScopeAncestor(demand.scope, nodes_[demand.source].scope) ||
      !isScopeAncestor(demand.scope, nodes_[demand.target].scope);
  if (invalidScope) {
    return {SyncCoverGraphError::InvalidScope, demands_.size()};
  }
  if (demand.distance != 0 && demand.scope == 0) {
    return {SyncCoverGraphError::InvalidDistance, demands_.size()};
  }
  if (demand.distance == 0 && demand.source == demand.target) {
    return {SyncCoverGraphError::ZeroDistanceSelfDemand, demands_.size()};
  }
  const SyncCoverGraphError error = completeEndpointGuards(
      demand.source, demand.target, demand.scope, demand.distance,
      demand.sourceGuard, demand.targetGuard);
  if (error != SyncCoverGraphError::None) {
    return {error, demands_.size()};
  }
  const std::size_t index = demands_.size();
  demands_.push_back(std::move(demand));
  return {SyncCoverGraphError::None, index};
}

SyncCoverGraphResult SyncCoverGraph::validate() const {
  for (std::size_t index = 0; index < controls_.size(); ++index) {
    const SyncCoverControl &control = controls_[index];
    const bool invalidControl = control.id != index ||
                                control.alternatives == 0 ||
                                !hasValidScope(control.scope);
    if (invalidControl) {
      return {SyncCoverGraphError::InvalidControl, index};
    }
  }
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    const SyncCoverNode &node = nodes_[index];
    if (!hasValidScope(node.scope)) {
      return {SyncCoverGraphError::InvalidScope, index};
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
        !isScopeAncestor(demand.scope, nodes_[demand.source].scope) ||
        !isScopeAncestor(demand.scope, nodes_[demand.target].scope);
    if (invalidScope) {
      return {SyncCoverGraphError::InvalidScope, index};
    }
    if (demand.distance != 0 && demand.scope == 0) {
      return {SyncCoverGraphError::InvalidDistance, index};
    }
    if (demand.distance == 0 && demand.source == demand.target) {
      return {SyncCoverGraphError::ZeroDistanceSelfDemand, index};
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
        !isScopeAncestor(edge.scope, nodes_[edge.source].scope) ||
        !isScopeAncestor(edge.scope, nodes_[edge.target].scope);
    if (invalidScope) {
      return {SyncCoverGraphError::InvalidScope, index};
    }
    if (edge.distance != 0 && edge.scope == 0) {
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

bool SyncCoverGraph::isScopeAncestor(SyncCoverScopeId ancestor,
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
    if (!isScopeAncestor(controls_[literal.control].scope, occurrenceScope)) {
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
        distance != 0 && isScopeAncestor(recurrenceScope, control.scope);
    if (!contextualized &&
        sourceLiteral.alternative != targetLiteral.alternative) {
      return SyncCoverGraphError::IncompatibleEndpoints;
    }
    ++sourceIndex;
    ++targetIndex;
  }
  return SyncCoverGraphError::None;
}
