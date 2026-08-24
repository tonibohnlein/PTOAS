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
                      unsigned budget) {
  if (budget == 0) {
    return false;
  }
  if (kind == SyncCoverResourceKind::EventId) {
    return source != target && poolIdentity == 0;
  }
  return true;
}

bool validResourceComposition(
    const std::vector<SyncCoverResourceDomain> &domains,
    SyncCoverMechanismKind kind,
    const std::vector<SyncCoverResourceUse> &uses) {
  if (kind == SyncCoverMechanismKind::Barrier) {
    return uses.empty();
  }
  if (uses.empty()) {
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
    std::uint32_t targetResource, unsigned budget, std::uint64_t poolIdentity) {
  if (!validDomainShape(kind, sourceResource, targetResource, poolIdentity,
                        budget)) {
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
    if (duplicate->budget == budget) {
      return makeResult(SyncCoverMechanismError::None, index);
    }
    return makeResult(SyncCoverMechanismError::InvalidDomain, index);
  }
  const SyncCoverResourceDomainId id = domains_.size();
  domains_.push_back(
      {id, kind, sourceResource, targetResource, poolIdentity, budget});
  return makeResult(SyncCoverMechanismError::None, id);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::addMechanism(
    const SyncCoverMechanismDescriptor &descriptor) {
  if (descriptor.kind == SyncCoverMechanismKind::OwnershipProtocol) {
    return makeResult(SyncCoverMechanismError::UnverifiedProtocol,
                      mechanisms_.size());
  }
  return addMechanismImpl(descriptor, false);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::addVerifiedProtocol(
    const SyncCoverMechanismDescriptor &descriptor,
    const std::function<bool(const SyncCoverMechanismDescriptor &)> &verify) {
  const bool validKind =
      descriptor.kind == SyncCoverMechanismKind::OwnershipProtocol;
  if (!validKind || !verify || !verify(descriptor)) {
    return makeResult(SyncCoverMechanismError::UnverifiedProtocol,
                      mechanisms_.size());
  }
  return addMechanismImpl(descriptor, true);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::addMechanismImpl(
    const SyncCoverMechanismDescriptor &descriptor, bool protocolVerified) {
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
  if (!validResourceComposition(domains_, descriptor.kind,
                                descriptor.resourceUses)) {
    return makeResult(SyncCoverMechanismError::InvalidResourceUse,
                      mechanisms_.size());
  }

  std::vector<bool> boundSupply(descriptor.supplyEdges.size(), false);
  for (const SyncCoverResourceUse &use : descriptor.resourceUses) {
    const SyncCoverMechanismError error =
        validateResourceUse(use, &descriptor.supplyEdges);
    if (error != SyncCoverMechanismError::None) {
      return makeResult(error, mechanisms_.size());
    }
    for (std::size_t edge : use.supplyEdges) {
      boundSupply[edge] = true;
    }
  }
  if (!barrierKind && std::find(boundSupply.begin(), boundSupply.end(),
                                false) != boundSupply.end()) {
    return makeResult(SyncCoverMechanismError::InvalidResourceUse,
                      mechanisms_.size());
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

  SyncCoverMechanism mechanism;
  mechanism.id = id;
  mechanism.kind = descriptor.kind;
  mechanism.providerIdentity = descriptor.providerIdentity;
  mechanism.protocolVerified = protocolVerified;
  mechanism.barrier = descriptor.barrier;
  mechanism.resourceUses = descriptor.resourceUses;
  for (SyncCoverResourceUse &use : mechanism.resourceUses) {
    for (std::size_t &edge : use.supplyEdges) {
      edge += firstEdge;
    }
  }
  for (std::size_t edge = 0; edge < descriptor.supplyEdges.size(); ++edge) {
    mechanism.supplyEdges.push_back(firstEdge + edge);
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
    const bool invalid = domain.id != index ||
                         !validDomainShape(domain.kind, domain.sourceResource,
                                           domain.targetResource,
                                           domain.poolIdentity, domain.budget);
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
    if (mechanism.kind == SyncCoverMechanismKind::OwnershipProtocol &&
        !mechanism.protocolVerified) {
      return makeResult(SyncCoverMechanismError::UnverifiedProtocol, index);
    }
    if (!validResourceComposition(domains_, mechanism.kind,
                                  mechanism.resourceUses)) {
      return makeResult(SyncCoverMechanismError::InvalidResourceUse, index);
    }

    std::vector<bool> boundSupply(graph_.getEdges().size(), false);
    for (const SyncCoverResourceUse &use : mechanism.resourceUses) {
      const SyncCoverMechanismError error = validateResourceUse(use);
      if (error != SyncCoverMechanismError::None) {
        return makeResult(error, index);
      }
      for (std::size_t edge : use.supplyEdges) {
        boundSupply[edge] = true;
      }
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
      if (!barrierKind && !boundSupply[edgeIndex]) {
        return makeResult(SyncCoverMechanismError::InvalidResourceUse, index);
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
    const SyncCoverResourceUse &use,
    const std::vector<SyncCoverEdge> *descriptorEdges) const {
  if (use.domain >= domains_.size()) {
    return SyncCoverMechanismError::InvalidDomain;
  }
  const bool invalidNode = use.begin >= graph_.getNodes().size() ||
                           use.end >= graph_.getNodes().size();
  const bool invalidShape = invalidNode || use.width == 0 ||
                            use.scope >= graph_.getScopes().size() ||
                            use.supplyEdges.empty();
  if (invalidShape) {
    return SyncCoverMechanismError::InvalidResourceUse;
  }
  const SyncCoverResourceDomain &domain = domains_[use.domain];
  const bool wrongResources =
      domain.sourceResource != graph_.getNodes()[use.begin].resource ||
      domain.targetResource != graph_.getNodes()[use.end].resource;
  if (wrongResources) {
    return SyncCoverMechanismError::InvalidResourceUse;
  }
  const bool invalidScope =
      !scopeContains(graph_, use.scope, graph_.getNodes()[use.begin].scope) ||
      !scopeContains(graph_, use.scope, graph_.getNodes()[use.end].scope);
  if (invalidScope || (use.distance != 0 && use.scope == 0)) {
    return SyncCoverMechanismError::InvalidResourceUse;
  }

  const std::vector<SyncCoverEdge> &edges =
      descriptorEdges ? *descriptorEdges : graph_.getEdges();
  for (std::size_t edgeIndex : use.supplyEdges) {
    if (edgeIndex >= edges.size()) {
      return SyncCoverMechanismError::InvalidResourceUse;
    }
    const SyncCoverEdge &edge = edges[edgeIndex];
    const bool wrongLifetime =
        edge.source != use.begin || edge.target != use.end ||
        edge.scope != use.scope || edge.distance != use.distance;
    if (wrongLifetime) {
      return SyncCoverMechanismError::InvalidResourceUse;
    }
  }
  return SyncCoverMechanismError::None;
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
