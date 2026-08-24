// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverMechanism.h"

#include <algorithm>
#include <array>
#include <utility>

using namespace mlir::pto;

namespace {

SyncCoverMechanismResult makeResult(SyncCoverMechanismError error,
                                    std::optional<std::size_t> index = {}) {
  SyncCoverMechanismResult result;
  result.error = error;
  result.index = index;
  return result;
}

bool scopeContains(const SyncCoverGraph &graph, SyncCoverScopeId ancestor,
                   SyncCoverScopeId descendant) {
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  const bool invalid = ancestor >= scopes.size() || descendant >= scopes.size();
  if (invalid) {
    return false;
  }
  while (descendant != ancestor && descendant != 0) {
    descendant = scopes[descendant].parent;
  }
  return descendant == ancestor;
}

bool scopeMustExecuteWithin(const SyncCoverGraph &graph,
                            SyncCoverScopeId ancestor,
                            SyncCoverScopeId descendant) {
  if (!scopeContains(graph, ancestor, descendant)) {
    return false;
  }
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  while (descendant != ancestor) {
    if (!scopes[descendant].mustExecuteWithinParent) {
      return false;
    }
    descendant = scopes[descendant].parent;
  }
  return true;
}

bool scopeExecutesWhen(const SyncCoverGraph &graph,
                       SyncCoverScopeId conditionScope,
                       SyncCoverScopeId requiredScope) {
  if (scopeContains(graph, requiredScope, conditionScope)) {
    return true;
  }
  return scopeMustExecuteWithin(graph, conditionScope, requiredScope);
}

bool validDomainShape(SyncCoverResourceKind kind, std::uint32_t source,
                      std::uint32_t target, std::uint64_t poolIdentity,
                      unsigned budget,
                      const std::vector<unsigned> &reservedIds) {
  if (budget == 0) {
    return false;
  }
  if (kind == SyncCoverResourceKind::EventId) {
    return source != target && poolIdentity == 0 &&
           std::is_sorted(reservedIds.begin(), reservedIds.end()) &&
           std::adjacent_find(reservedIds.begin(), reservedIds.end()) ==
               reservedIds.end();
  }
  return reservedIds.empty();
}

bool validResourceComposition(
    const std::vector<SyncCoverResourceDomain> &domains,
    SyncCoverMechanismKind kind,
    const std::vector<SyncCoverResourceAction> &actions,
    const std::vector<SyncCoverResourceUse> &uses) {
  if (kind == SyncCoverMechanismKind::Barrier) {
    return actions.empty() && uses.empty();
  }
  const bool missingResourceModel = actions.empty() || uses.empty();
  if (missingResourceModel) {
    return false;
  }
  if (kind == SyncCoverMechanismKind::EventBundle) {
    return std::all_of(uses.begin(), uses.end(), [&](const auto &use) {
      return use.domain >= domains.size() ||
             domains[use.domain].kind == SyncCoverResourceKind::EventId;
    });
  }
  return true;
}

bool containsIndex(const std::vector<std::size_t> &indices, std::size_t index) {
  return std::find(indices.begin(), indices.end(), index) != indices.end();
}

bool canonicalActionIndices(const std::vector<std::size_t> &indices) {
  return !indices.empty() && std::is_sorted(indices.begin(), indices.end()) &&
         std::adjacent_find(indices.begin(), indices.end()) == indices.end();
}

bool actionOccursWithin(const SyncCoverGraph &graph,
                        const SyncCoverResourceAction &action,
                        SyncCoverScopeId scope) {
  const std::optional<SyncCoverTimelinePosition> position =
      resolveSyncCoverAnchor(graph, action.anchor);
  if (!position || scope >= graph.getScopes().size()) {
    return false;
  }
  SyncCoverScopeId actionScope = 0;
  switch (action.anchor.kind) {
  case SyncCoverAnchorKind::BeforeNode:
  case SyncCoverAnchorKind::AfterNode:
    actionScope = graph.getNodes()[action.anchor.node].scope;
    if (graph.getNodes()[action.anchor.node].resource != action.resource) {
      return false;
    }
    break;
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit:
    actionScope = action.anchor.scope;
    break;
  }
  if (!scopeContains(graph, scope, actionScope)) {
    return false;
  }
  const std::optional<SyncCoverTimelineInterval> &timeline =
      graph.getScopes()[scope].timeline;
  return !timeline ||
         (timeline->begin <= *position && *position <= timeline->end);
}

SyncCoverMechanismError
validateActionAccounting(const std::vector<SyncCoverResourceDomain> &domains,
                         const std::vector<SyncCoverResourceAction> &actions,
                         const std::vector<SyncCoverResourceUse> &uses) {
  std::vector<std::array<unsigned, 2>> claims(actions.size(), {0, 0});
  for (const SyncCoverResourceUse &use : uses) {
    if (use.domain >= domains.size()) {
      return SyncCoverMechanismError::InvalidDomain;
    }
    const std::size_t kind = static_cast<std::size_t>(domains[use.domain].kind);
    for (std::size_t action : use.actions) {
      const bool invalidClaim =
          action >= actions.size() || ++claims[action][kind] > 1;
      if (invalidClaim) {
        return SyncCoverMechanismError::InvalidAction;
      }
    }
  }
  for (const auto &claim : claims) {
    if (claim[0] == 0 && claim[1] == 0) {
      return SyncCoverMechanismError::InvalidAction;
    }
  }
  return SyncCoverMechanismError::None;
}

bool validBarrierEdge(const SyncCoverGraph &graph,
                      const SyncCoverBarrierPlacement &placement,
                      const SyncCoverEdge &edge) {
  const std::vector<SyncCoverNode> &nodes = graph.getNodes();
  const bool invalidNode = placement.anchor >= nodes.size() ||
                           edge.source >= nodes.size() ||
                           edge.target >= nodes.size();
  if (invalidNode) {
    return false;
  }
  const bool sameResource =
      nodes[placement.anchor].resource == placement.resource &&
      nodes[edge.source].resource == placement.resource &&
      nodes[edge.target].resource == placement.resource;
  const SyncCoverNode &anchor = nodes[placement.anchor];
  const SyncCoverNode &target = nodes[edge.target];
  SyncCoverGuard targetGuard = edge.targetGuard;
  targetGuard.literals.insert(targetGuard.literals.end(),
                              target.guard.literals.begin(),
                              target.guard.literals.end());
  const bool wrongScope =
      placement.scope != anchor.scope ||
      !scopeContains(graph, edge.scope, placement.scope) ||
      !scopeExecutesWhen(graph, target.scope, placement.scope);
  if (!sameResource || wrongScope ||
      !syncCoverGuardImplies(targetGuard, anchor.guard)) {
    return false;
  }
  if (nodes[edge.target].order < nodes[placement.anchor].order) {
    return false;
  }
  return edge.distance != 0 ||
         nodes[edge.source].order < nodes[placement.anchor].order;
}

} // namespace

SyncCoverMechanismResult SyncCoverMechanismUniverse::addResourceDomain(
    SyncCoverResourceKind kind, std::uint32_t sourceResource,
    std::uint32_t targetResource, unsigned budget, std::uint64_t poolIdentity,
    std::vector<unsigned> reservedIds) {
  std::sort(reservedIds.begin(), reservedIds.end());
  reservedIds.erase(std::unique(reservedIds.begin(), reservedIds.end()),
                    reservedIds.end());
  if (!validDomainShape(kind, sourceResource, targetResource, poolIdentity,
                        budget, reservedIds)) {
    return makeResult(SyncCoverMechanismError::InvalidDomain, domains_.size());
  }
  const auto duplicate =
      std::find_if(domains_.begin(), domains_.end(),
                   [&](const SyncCoverResourceDomain &old) {
                     return old.kind == kind &&
                            old.sourceResource == sourceResource &&
                            old.targetResource == targetResource &&
                            old.poolIdentity == poolIdentity;
                   });
  if (duplicate != domains_.end()) {
    const std::size_t index =
        static_cast<std::size_t>(std::distance(domains_.begin(), duplicate));
    if (duplicate->budget == budget && duplicate->reservedIds == reservedIds) {
      return makeResult(SyncCoverMechanismError::None, index);
    }
    return makeResult(SyncCoverMechanismError::InvalidDomain, index);
  }
  const SyncCoverResourceDomainId id = domains_.size();
  domains_.push_back({id, kind, sourceResource, targetResource, poolIdentity,
                      budget, std::move(reservedIds)});
  return makeResult(SyncCoverMechanismError::None, id);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::addMechanism(
    const SyncCoverMechanismDescriptor &descriptor) {
  if (descriptor.kind == SyncCoverMechanismKind::VerifiedProtocol) {
    return makeResult(SyncCoverMechanismError::UnverifiedProtocol,
                      mechanisms_.size());
  }
  return addMechanismImpl(descriptor, false);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::addVerifiedProtocol(
    const SyncCoverMechanismDescriptor &descriptor,
    const std::function<bool(const SyncCoverMechanismDescriptor &)> &verify) {
  const bool validKind =
      descriptor.kind == SyncCoverMechanismKind::VerifiedProtocol;
  if (!validKind || !verify) {
    return makeResult(SyncCoverMechanismError::UnverifiedProtocol,
                      mechanisms_.size());
  }
  return addMechanismImpl(descriptor, true, verify);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::addMechanismImpl(
    const SyncCoverMechanismDescriptor &descriptor, bool protocolVerified,
    const std::function<bool(const SyncCoverMechanismDescriptor &)> &verify) {
  if (!validate()) {
    return makeResult(SyncCoverMechanismError::InvalidGraph,
                      mechanisms_.size());
  }
  if (descriptor.supplyEdges.empty()) {
    return makeResult(SyncCoverMechanismError::EmptySupply, mechanisms_.size());
  }
  const bool barrierKind = descriptor.kind == SyncCoverMechanismKind::Barrier;
  if (barrierKind) {
    const SyncCoverMechanismError error = validateBarrier(descriptor);
    if (error != SyncCoverMechanismError::None) {
      return makeResult(error, mechanisms_.size());
    }
  } else if (descriptor.barrier) {
    return makeResult(SyncCoverMechanismError::InvalidMechanism,
                      mechanisms_.size());
  }
  if (!validResourceComposition(domains_, descriptor.kind, descriptor.actions,
                                descriptor.resourceUses)) {
    return makeResult(SyncCoverMechanismError::InvalidResourceUse,
                      mechanisms_.size());
  }

  for (const SyncCoverResourceUse &use : descriptor.resourceUses) {
    const SyncCoverMechanismError error =
        validateResourceUse(use, descriptor.supplyEdges, descriptor.actions);
    if (error != SyncCoverMechanismError::None) {
      return makeResult(error, mechanisms_.size());
    }
  }
  SyncCoverMechanismError descriptorError = validateActionAccounting(
      domains_, descriptor.actions, descriptor.resourceUses);
  if (descriptorError == SyncCoverMechanismError::None) {
    std::vector<std::size_t> descriptorSupplyEdges(
        descriptor.supplyEdges.size());
    for (std::size_t edge = 0; edge < descriptorSupplyEdges.size(); ++edge) {
      descriptorSupplyEdges[edge] = edge;
    }
    descriptorError = validateSupplyBindings(
        descriptor.kind, descriptor.supplyEdges, descriptorSupplyEdges,
        descriptor.actions, descriptor.resourceUses, descriptor.supplyBindings);
  }
  if (descriptorError != SyncCoverMechanismError::None) {
    return makeResult(descriptorError, mechanisms_.size());
  }

  const SyncCoverMechanismId id = mechanisms_.size();
  const std::size_t firstEdge = graph_.getEdges().size();
  SyncCoverGraph staged = graph_;
  for (SyncCoverEdge edge : descriptor.supplyEdges) {
    if (edge.kind != SyncCoverEdgeKind::CompletionSupply || edge.mechanism) {
      return makeResult(SyncCoverMechanismError::InvalidSupply, id);
    }
    edge.mechanism = id;
    if (!staged.addEdge(std::move(edge))) {
      return makeResult(SyncCoverMechanismError::InvalidSupply, id);
    }
  }
  if (!staged.validate()) {
    return makeResult(SyncCoverMechanismError::InvalidSupply, id);
  }
  if (protocolVerified && !verify(descriptor)) {
    return makeResult(SyncCoverMechanismError::UnverifiedProtocol, id);
  }

  SyncCoverMechanism mechanism;
  mechanism.id = id;
  mechanism.kind = descriptor.kind;
  mechanism.providerIdentity = descriptor.providerIdentity;
  mechanism.protocolVerified = protocolVerified;
  mechanism.barrier = descriptor.barrier;
  mechanism.actions = descriptor.actions;
  mechanism.resourceUses = descriptor.resourceUses;
  mechanism.supplyBindings = descriptor.supplyBindings;
  for (SyncCoverResourceUse &use : mechanism.resourceUses) {
    for (std::size_t &edge : use.supplyEdges) {
      edge += firstEdge;
    }
  }
  for (std::size_t edge = 0; edge < descriptor.supplyEdges.size(); ++edge) {
    mechanism.supplyEdges.push_back(firstEdge + edge);
  }
  for (SyncCoverSupplyBinding &binding : mechanism.supplyBindings) {
    binding.supplyEdge += firstEdge;
  }
  graph_ = std::move(staged);
  mechanisms_.push_back(std::move(mechanism));
  return makeResult(SyncCoverMechanismError::None, id);
}

SyncCoverMechanismResult
SyncCoverMechanismUniverse::addConflict(SyncCoverMechanismId first,
                                        SyncCoverMechanismId second) {
  const bool invalid = first >= mechanisms_.size() ||
                       second >= mechanisms_.size() || first == second;
  if (invalid) {
    return makeResult(SyncCoverMechanismError::InvalidConflict,
                      mechanisms_.size());
  }
  auto addOne = [&](SyncCoverMechanismId source, SyncCoverMechanismId target) {
    std::vector<SyncCoverMechanismId> &conflicts =
        mechanisms_[source].conflicts;
    const auto position =
        std::lower_bound(conflicts.begin(), conflicts.end(), target);
    const bool missing = position == conflicts.end() || *position != target;
    if (missing) {
      conflicts.insert(position, target);
    }
  };
  addOne(first, second);
  addOne(second, first);
  return makeResult(SyncCoverMechanismError::None, first);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::validate() const {
  if (!graph_.validate()) {
    return makeResult(SyncCoverMechanismError::InvalidGraph);
  }
  for (std::size_t index = 0; index < domains_.size(); ++index) {
    const SyncCoverResourceDomain &domain = domains_[index];
    const bool invalid =
        domain.id != index ||
        !validDomainShape(domain.kind, domain.sourceResource,
                          domain.targetResource, domain.poolIdentity,
                          domain.budget, domain.reservedIds);
    if (invalid) {
      return makeResult(SyncCoverMechanismError::InvalidDomain, index);
    }
  }

  std::vector<bool> claimedEdges(graph_.getEdges().size(), false);
  for (std::size_t index = 0; index < mechanisms_.size(); ++index) {
    const SyncCoverMechanism &mechanism = mechanisms_[index];
    if (mechanism.id != index || mechanism.supplyEdges.empty()) {
      return makeResult(SyncCoverMechanismError::InvalidMechanism, index);
    }
    const bool barrierKind = mechanism.kind == SyncCoverMechanismKind::Barrier;
    if (barrierKind != mechanism.barrier.has_value()) {
      return makeResult(SyncCoverMechanismError::InvalidMechanism, index);
    }
    if (mechanism.kind == SyncCoverMechanismKind::VerifiedProtocol &&
        !mechanism.protocolVerified) {
      return makeResult(SyncCoverMechanismError::UnverifiedProtocol, index);
    }
    if (!validResourceComposition(domains_, mechanism.kind, mechanism.actions,
                                  mechanism.resourceUses)) {
      return makeResult(SyncCoverMechanismError::InvalidResourceUse, index);
    }

    for (const SyncCoverResourceUse &use : mechanism.resourceUses) {
      const SyncCoverMechanismError error =
          validateResourceUse(use, graph_.getEdges(), mechanism.actions);
      if (error != SyncCoverMechanismError::None) {
        return makeResult(error, index);
      }
    }
    SyncCoverMechanismError mechanismError = validateActionAccounting(
        domains_, mechanism.actions, mechanism.resourceUses);
    if (mechanismError == SyncCoverMechanismError::None) {
      mechanismError = validateSupplyBindings(
          mechanism.kind, graph_.getEdges(), mechanism.supplyEdges,
          mechanism.actions, mechanism.resourceUses, mechanism.supplyBindings);
    }
    if (mechanismError != SyncCoverMechanismError::None) {
      return makeResult(mechanismError, index);
    }
    for (std::size_t edgeIndex : mechanism.supplyEdges) {
      const bool invalidEdge =
          edgeIndex >= graph_.getEdges().size() || claimedEdges[edgeIndex];
      if (invalidEdge) {
        return makeResult(SyncCoverMechanismError::InvalidSupply, index);
      }
      const SyncCoverEdge &edge = graph_.getEdges()[edgeIndex];
      const bool wrongSupply =
          edge.kind != SyncCoverEdgeKind::CompletionSupply ||
          edge.mechanism != mechanism.id;
      if (wrongSupply) {
        return makeResult(SyncCoverMechanismError::InvalidSupply, index);
      }
      if (barrierKind && !validBarrierEdge(graph_, *mechanism.barrier, edge)) {
        return makeResult(SyncCoverMechanismError::InvalidSupply, index);
      }
      claimedEdges[edgeIndex] = true;
    }
    for (SyncCoverMechanismId conflict : mechanism.conflicts) {
      const bool invalidConflict =
          conflict >= mechanisms_.size() || conflict == mechanism.id;
      if (invalidConflict) {
        return makeResult(SyncCoverMechanismError::InvalidConflict, index);
      }
      const std::vector<SyncCoverMechanismId> &reverse =
          mechanisms_[conflict].conflicts;
      if (!std::binary_search(reverse.begin(), reverse.end(), mechanism.id)) {
        return makeResult(SyncCoverMechanismError::InvalidConflict, index);
      }
    }
  }

  for (std::size_t edge = 0; edge < graph_.getEdges().size(); ++edge) {
    if (graph_.getEdges()[edge].mechanism && !claimedEdges[edge]) {
      return makeResult(SyncCoverMechanismError::InvalidSupply, edge);
    }
  }
  return makeResult(SyncCoverMechanismError::None);
}

SyncCoverMechanismError SyncCoverMechanismUniverse::validateResourceUse(
    const SyncCoverResourceUse &use, const std::vector<SyncCoverEdge> &edges,
    const std::vector<SyncCoverResourceAction> &actions) const {
  if (use.domain >= domains_.size()) {
    return SyncCoverMechanismError::InvalidDomain;
  }
  const bool invalidShape = use.width == 0 ||
                            use.scope >= graph_.getScopes().size() ||
                            !canonicalActionIndices(use.actions) ||
                            !canonicalActionIndices(use.supplyEdges);
  if (invalidShape) {
    return SyncCoverMechanismError::InvalidResourceUse;
  }
  if (use.distance != 0 &&
      (use.scope == 0 || !graph_.getScopes()[use.scope].timeline)) {
    return SyncCoverMechanismError::InvalidResourceUse;
  }

  const SyncCoverResourceDomain &domain = domains_[use.domain];
  bool hasProduce = false;
  bool hasConsume = false;
  for (std::size_t actionIndex : use.actions) {
    if (actionIndex >= actions.size()) {
      return SyncCoverMechanismError::InvalidAction;
    }
    const SyncCoverResourceAction &action = actions[actionIndex];
    const bool produce = action.kind == SyncCoverResourceActionKind::Produce;
    const std::uint32_t expectedResource =
        produce ? domain.sourceResource : domain.targetResource;
    if (action.resource != expectedResource ||
        !actionOccursWithin(graph_, action, use.scope)) {
      return SyncCoverMechanismError::InvalidAction;
    }
    hasProduce |= produce;
    hasConsume |= !produce;
  }
  if (!hasProduce || !hasConsume) {
    return SyncCoverMechanismError::InvalidResourceUse;
  }

  for (std::size_t edgeIndex : use.supplyEdges) {
    if (edgeIndex >= edges.size()) {
      return SyncCoverMechanismError::InvalidResourceUse;
    }
    const SyncCoverEdge &edge = edges[edgeIndex];
    const bool wrongLifetime =
        edge.scope != use.scope || edge.distance != use.distance;
    if (wrongLifetime) {
      return SyncCoverMechanismError::InvalidResourceUse;
    }
  }
  return SyncCoverMechanismError::None;
}

SyncCoverMechanismError SyncCoverMechanismUniverse::validateSupplyBindings(
    SyncCoverMechanismKind kind, const std::vector<SyncCoverEdge> &edges,
    const std::vector<std::size_t> &supplyEdges,
    const std::vector<SyncCoverResourceAction> &actions,
    const std::vector<SyncCoverResourceUse> &uses,
    const std::vector<SyncCoverSupplyBinding> &bindings) const {
  if (kind == SyncCoverMechanismKind::Barrier) {
    return bindings.empty() ? SyncCoverMechanismError::None
                            : SyncCoverMechanismError::InvalidBinding;
  }

  std::vector<std::size_t> expectedEdges;
  for (const SyncCoverResourceUse &use : uses) {
    expectedEdges.insert(expectedEdges.end(), use.supplyEdges.begin(),
                         use.supplyEdges.end());
  }
  std::sort(expectedEdges.begin(), expectedEdges.end());
  expectedEdges.erase(std::unique(expectedEdges.begin(), expectedEdges.end()),
                      expectedEdges.end());
  if (expectedEdges != supplyEdges) {
    return SyncCoverMechanismError::InvalidBinding;
  }
  const bool wrongBindingCount = bindings.size() != expectedEdges.size();
  if (wrongBindingCount) {
    return SyncCoverMechanismError::InvalidBinding;
  }

  std::vector<std::size_t> claimedEdges;
  for (const SyncCoverSupplyBinding &binding : bindings) {
    const bool invalidIndex = binding.supplyEdge >= edges.size() ||
                              binding.resourceUse >= uses.size() ||
                              binding.produceAction >= actions.size() ||
                              binding.consumeAction >= actions.size();
    if (invalidIndex) {
      return SyncCoverMechanismError::InvalidBinding;
    }
    const SyncCoverResourceUse &use = uses[binding.resourceUse];
    const bool invalidUse =
        use.domain >= domains_.size() ||
        !containsIndex(use.supplyEdges, binding.supplyEdge) ||
        !containsIndex(use.actions, binding.produceAction) ||
        !containsIndex(use.actions, binding.consumeAction);
    if (invalidUse) {
      return SyncCoverMechanismError::InvalidBinding;
    }
    const SyncCoverEdge &edge = edges[binding.supplyEdge];
    const SyncCoverResourceAction &produce = actions[binding.produceAction];
    const SyncCoverResourceAction &consume = actions[binding.consumeAction];
    const bool wrongKinds =
        produce.kind != SyncCoverResourceActionKind::Produce ||
        consume.kind != SyncCoverResourceActionKind::Consume;
    const bool wrongLifetime =
        edge.scope != use.scope || edge.distance != use.distance;
    if (wrongKinds || wrongLifetime) {
      return SyncCoverMechanismError::InvalidBinding;
    }

    if (kind == SyncCoverMechanismKind::EventBundle) {
      const SyncCoverResourceDomain &domain = domains_[use.domain];
      const bool canonicalAnchors =
          edge.source < graph_.getNodes().size() &&
          edge.target < graph_.getNodes().size() &&
          produce.anchor.kind == SyncCoverAnchorKind::AfterNode &&
          produce.anchor.node == edge.source && produce.anchor.scope == 0 &&
          consume.anchor.kind == SyncCoverAnchorKind::BeforeNode &&
          consume.anchor.node == edge.target && consume.anchor.scope == 0;
      const bool completionQualified =
          canonicalAnchors && syncCoverNodeCanProduceCompletion(
                                  graph_, edge.source, domain.targetResource);
      if (domain.kind != SyncCoverResourceKind::EventId || edge.distance != 0 ||
          use.distance != 0 || !completionQualified) {
        return SyncCoverMechanismError::InvalidBinding;
      }
    }
    claimedEdges.push_back(binding.supplyEdge);
  }
  std::sort(claimedEdges.begin(), claimedEdges.end());
  return claimedEdges == expectedEdges
             ? SyncCoverMechanismError::None
             : SyncCoverMechanismError::InvalidBinding;
}

SyncCoverMechanismError SyncCoverMechanismUniverse::validateBarrier(
    const SyncCoverMechanismDescriptor &descriptor) const {
  if (!descriptor.barrier || !descriptor.resourceUses.empty()) {
    return SyncCoverMechanismError::InvalidMechanism;
  }
  const SyncCoverBarrierPlacement &placement = *descriptor.barrier;
  const bool invalidPlacement =
      placement.anchor >= graph_.getNodes().size() ||
      placement.scope >= graph_.getScopes().size() ||
      !scopeContains(graph_, placement.scope,
                     graph_.getNodes()[placement.anchor].scope);
  if (invalidPlacement) {
    return SyncCoverMechanismError::InvalidMechanism;
  }
  for (const SyncCoverEdge &edge : descriptor.supplyEdges) {
    if (!validBarrierEdge(graph_, placement, edge)) {
      return SyncCoverMechanismError::InvalidSupply;
    }
  }
  return SyncCoverMechanismError::None;
}
