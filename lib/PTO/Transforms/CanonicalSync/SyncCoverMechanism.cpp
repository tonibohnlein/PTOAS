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
#include <type_traits>
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
  return poolIdentity != 0 && reservedIds.empty();
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

bool validResourceLifetime(const SyncCoverGraph &graph,
                           SyncCoverMechanismKind kind,
                           const SyncCoverResourceUse &use,
                           const SyncCoverEdge &edge) {
  if (edge.scope == use.scope && edge.distance == use.distance) {
    return true;
  }
  if (kind != SyncCoverMechanismKind::VerifiedProtocol || use.distance == 0 ||
      edge.distance > use.distance) {
    return false;
  }
  if (edge.distance != 0) {
    return graph.scopeContains(use.scope, edge.scope);
  }
  return graph.scopeContains(use.scope, edge.scope) ||
         graph.scopeContains(edge.scope, use.scope);
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
  case SyncCoverAnchorKind::TimelinePoint:
    actionScope = action.anchor.scope;
    break;
  }
  if (!graph.scopeContains(scope, actionScope)) {
    return false;
  }
  const std::optional<SyncCoverTimelineInterval> &timeline =
      graph.getScopes()[scope].timeline;
  return !timeline ||
         (timeline->begin <= *position && *position <= timeline->end);
}

std::optional<SyncCoverScopeId>
getAnchorScope(const SyncCoverGraph &graph, const SyncCoverAnchor &anchor) {
  switch (anchor.kind) {
  case SyncCoverAnchorKind::BeforeNode:
  case SyncCoverAnchorKind::AfterNode:
    if (anchor.node >= graph.getNodes().size()) {
      return std::nullopt;
    }
    return graph.getNodes()[anchor.node].scope;
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit:
  case SyncCoverAnchorKind::TimelinePoint:
    if (anchor.scope >= graph.getScopes().size()) {
      return std::nullopt;
    }
    return anchor.scope;
  }
  return std::nullopt;
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
  const bool invalidNode = edge.source >= nodes.size() ||
                           edge.target >= nodes.size();
  if (invalidNode) {
    return false;
  }
  const std::optional<SyncCoverTimelinePosition> position =
      resolveSyncCoverAnchor(graph, placement.anchor);
  if (!position) {
    return false;
  }
  const bool drainsSourceResource =
      placement.drainsAllResources ||
      nodes[edge.source].resource == placement.resource;
  const SyncCoverNode &target = nodes[edge.target];
  SyncCoverGuard targetGuard = edge.targetGuard;
  targetGuard.literals.insert(targetGuard.literals.end(),
                              target.guard.literals.begin(),
                              target.guard.literals.end());
  SyncCoverGuard placementGuard;
  if (placement.anchor.kind == SyncCoverAnchorKind::BeforeNode ||
      placement.anchor.kind == SyncCoverAnchorKind::AfterNode) {
    placementGuard = nodes[placement.anchor.node].guard;
  }
  const bool wrongScope = !graph.scopeContains(edge.scope, placement.scope) ||
      !graph.scopeExecutesWhen(target.scope, placement.scope);
  if (!drainsSourceResource || wrongScope ||
      !syncCoverGuardImplies(targetGuard, placementGuard)) {
    return false;
  }
  const SyncCoverTimelinePosition targetPosition =
      nodes[edge.target].order * 2;
  if (targetPosition < *position) {
    return false;
  }
  const SyncCoverTimelinePosition sourcePosition =
      nodes[edge.source].order * 2 + 1;
  return edge.distance != 0 || sourcePosition <= *position;
}

} // namespace

bool mlir::pto::syncCoverBarrierCanSupply(
    const SyncCoverGraph &graph, const SyncCoverBarrierPlacement &placement,
    const SyncCoverEdge &edge) {
  return validBarrierEdge(graph, placement, edge);
}

SyncCoverMechanismUniverse::SyncCoverMechanismUniverse(SyncCoverGraph &graph)
    : graph_(graph) {
  initializationResult_ = graph_.freezeStructure();
  knownGraphGeneration_ = graph_.getGeneration();
}

bool SyncCoverMechanismUniverse::ensureConstructionState() {
  if (!initializationResult_ || !graph_.isStructureFrozen()) {
    return false;
  }
  if (constructionValidated_ &&
      knownGraphGeneration_ == graph_.getGeneration()) {
    return true;
  }
  if (!validate()) {
    return false;
  }
  knownGraphGeneration_ = graph_.getGeneration();
  constructionValidated_ = true;
  return true;
}

void SyncCoverMechanismUniverse::noteSuccessfulMutation() {
  ++version_;
  knownGraphGeneration_ = graph_.getGeneration();
  constructionValidated_ = true;
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::addResourceDomain(
    SyncCoverResourceKind kind, std::uint32_t sourceResource,
    std::uint32_t targetResource, unsigned budget, std::uint64_t poolIdentity,
    std::vector<unsigned> reservedIds) {
  std::sort(reservedIds.begin(), reservedIds.end());
  reservedIds.erase(std::unique(reservedIds.begin(), reservedIds.end()),
                    reservedIds.end());
  if (!validDomainShape(kind, sourceResource, targetResource, poolIdentity,
                        budget, reservedIds)) {
    return makeResult(SyncCoverMechanismError::InvalidDomain);
  }
  if (!ensureConstructionState()) {
    return makeResult(SyncCoverMechanismError::InvalidGraph);
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
  if (kind == SyncCoverResourceKind::BufferToken) {
    const auto sharedPool =
        std::find_if(domains_.begin(), domains_.end(),
                     [&](const SyncCoverResourceDomain &old) {
                       return old.kind == SyncCoverResourceKind::BufferToken &&
                              old.poolIdentity == poolIdentity;
                     });
    if (sharedPool != domains_.end()) {
      return makeResult(SyncCoverMechanismError::InvalidDomain,
                        static_cast<std::size_t>(
                            std::distance(domains_.begin(), sharedPool)));
    }
  }
  const SyncCoverResourceDomainId id = domains_.size();
  domains_.push_back({id, kind, sourceResource, targetResource, poolIdentity,
                      budget, std::move(reservedIds)});
  noteSuccessfulMutation();
  return makeResult(SyncCoverMechanismError::None, id);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::addMechanism(
    const SyncCoverMechanismDescriptor &descriptor) {
  if (descriptor.kind == SyncCoverMechanismKind::VerifiedProtocol) {
    return makeResult(SyncCoverMechanismError::UnverifiedProtocol);
  }
  return addMechanismImpl(descriptor, false);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::addVerifiedProtocol(
    const SyncCoverMechanismDescriptor &descriptor,
    const std::function<bool(const SyncCoverMechanismDescriptor &)> &verify) {
  const bool validKind =
      descriptor.kind == SyncCoverMechanismKind::VerifiedProtocol;
  if (!validKind || !verify) {
    return makeResult(SyncCoverMechanismError::UnverifiedProtocol);
  }
  return addMechanismImpl(descriptor, true, verify);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::addMechanismImpl(
    const SyncCoverMechanismDescriptor &descriptor, bool protocolVerified,
    const std::function<bool(const SyncCoverMechanismDescriptor &)> &verify) {
  if (!ensureConstructionState()) {
    return makeResult(SyncCoverMechanismError::InvalidGraph);
  }
  const SyncCoverMechanismDescriptor candidate = descriptor;
  if (candidate.supplyEdges.empty() && candidate.verifiedCoverageEdges.empty()) {
    return makeResult(SyncCoverMechanismError::EmptySupply);
  }
  if (!protocolVerified && !candidate.verifiedCoverageEdges.empty()) {
    return makeResult(SyncCoverMechanismError::UnverifiedProtocol);
  }
  const bool barrierKind = candidate.kind == SyncCoverMechanismKind::Barrier;
  if (barrierKind) {
    const SyncCoverMechanismError error = validateBarrier(candidate);
    if (error != SyncCoverMechanismError::None) {
      return makeResult(error);
    }
  } else if (candidate.barrier) {
    return makeResult(SyncCoverMechanismError::InvalidMechanism);
  }
  if (!validResourceComposition(domains_, candidate.kind, candidate.actions,
                                candidate.resourceUses)) {
    return makeResult(SyncCoverMechanismError::InvalidResourceUse);
  }

  for (const SyncCoverResourceUse &use : candidate.resourceUses) {
    const SyncCoverMechanismError error =
        validateResourceUse(candidate.kind, use, candidate.supplyEdges,
                            candidate.actions);
    if (error != SyncCoverMechanismError::None) {
      return makeResult(error);
    }
  }
  SyncCoverMechanismError descriptorError = validateActionAccounting(
      domains_, candidate.actions, candidate.resourceUses);
  if (descriptorError == SyncCoverMechanismError::None) {
    std::vector<std::size_t> descriptorSupplyEdges(
        candidate.supplyEdges.size());
    for (std::size_t edge = 0; edge < descriptorSupplyEdges.size(); ++edge) {
      descriptorSupplyEdges[edge] = edge;
    }
    descriptorError = validateSupplyBindings(
        candidate.kind, candidate.supplyEdges, descriptorSupplyEdges,
        candidate.actions, candidate.resourceUses, candidate.supplyBindings);
  }
  if (descriptorError != SyncCoverMechanismError::None) {
    return makeResult(descriptorError);
  }

  const SyncCoverMechanismId id = mechanisms_.size();
  std::vector<SyncCoverEdge> preparedEdges = candidate.supplyEdges;
  preparedEdges.insert(preparedEdges.end(),
                       candidate.verifiedCoverageEdges.begin(),
                       candidate.verifiedCoverageEdges.end());
  for (SyncCoverEdge &edge : preparedEdges) {
    if (edge.kind != SyncCoverEdgeKind::CompletionSupply || edge.mechanism) {
      return makeResult(SyncCoverMechanismError::InvalidSupply);
    }
    edge.mechanism = id;
    if (!graph_.prepareEdge(edge)) {
      return makeResult(SyncCoverMechanismError::InvalidSupply);
    }
  }

  const std::size_t preVerificationGeneration = graph_.getGeneration();
  const std::size_t preVerificationVersion = version_;
  const bool verified = !protocolVerified || verify(candidate);
  if (version_ != preVerificationVersion) {
    constructionValidated_ = false;
    return makeResult(SyncCoverMechanismError::InvalidMechanism);
  }
  const bool graphMutated =
      graph_.getGeneration() != preVerificationGeneration;
  if (graphMutated) {
    constructionValidated_ = false;
    return makeResult(SyncCoverMechanismError::InvalidGraph);
  }
  if (!verified) {
    return makeResult(SyncCoverMechanismError::UnverifiedProtocol);
  }

  const std::size_t firstEdge = graph_.getEdges().size();
  SyncCoverMechanism mechanism;
  mechanism.id = id;
  mechanism.kind = candidate.kind;
  mechanism.providerIdentity = candidate.providerIdentity;
  mechanism.protocolVerified = protocolVerified;
  mechanism.barrier = candidate.barrier;
  mechanism.actions = candidate.actions;
  mechanism.resourceUses = candidate.resourceUses;
  mechanism.supplyBindings = candidate.supplyBindings;
  for (SyncCoverResourceUse &use : mechanism.resourceUses) {
    for (std::size_t &edge : use.supplyEdges) {
      edge += firstEdge;
    }
  }
  for (std::size_t edge = 0; edge < preparedEdges.size(); ++edge) {
    if (edge < candidate.supplyEdges.size()) {
      mechanism.supplyEdges.push_back(firstEdge + edge);
    } else {
      mechanism.verifiedCoverageEdges.push_back(firstEdge + edge);
    }
  }
  for (SyncCoverSupplyBinding &binding : mechanism.supplyBindings) {
    binding.supplyEdge += firstEdge;
  }

  const bool mechanismCapacityExhausted =
      mechanisms_.size() == mechanisms_.max_size();
  if (mechanismCapacityExhausted) {
    return makeResult(SyncCoverMechanismError::InvalidMechanism);
  }
  const bool edgesReserved =
      graph_.reserveAdditionalEdges(preparedEdges.size());
  if (!edgesReserved) {
    return makeResult(SyncCoverMechanismError::InvalidMechanism);
  }
  if (mechanisms_.size() == mechanisms_.capacity()) {
    // Geometric growth; an exact-size reserve would relocate every
    // mechanism (eight inner vectors each) on every insertion.
    const std::size_t doubled =
        mechanisms_.size() > mechanisms_.max_size() / 2
            ? mechanisms_.max_size()
            : std::max<std::size_t>(16, mechanisms_.size() * 2);
    mechanisms_.reserve(doubled);
  }
  SyncCoverGraph::EdgeTransaction transaction(graph_);
  for (SyncCoverEdge &edge : preparedEdges) {
    transaction.append(std::move(edge));
  }
  static_assert(std::is_nothrow_move_constructible<SyncCoverMechanism>::value,
                "reserved mechanism commit must not throw");
  mechanisms_.push_back(std::move(mechanism));
  noteSuccessfulMutation();
  transaction.commit();
  return makeResult(SyncCoverMechanismError::None, id);
}

SyncCoverMechanismResult
SyncCoverMechanismUniverse::addConflict(SyncCoverMechanismId first,
                                        SyncCoverMechanismId second) {
  const bool invalid = first >= mechanisms_.size() ||
                       second >= mechanisms_.size() || first == second;
  if (invalid) {
    return makeResult(SyncCoverMechanismError::InvalidConflict);
  }
  if (!ensureConstructionState()) {
    return makeResult(SyncCoverMechanismError::InvalidGraph);
  }
  std::vector<SyncCoverMechanismId> &firstConflicts =
      mechanisms_[first].conflicts;
  std::vector<SyncCoverMechanismId> &secondConflicts =
      mechanisms_[second].conflicts;
  const auto firstPosition =
      std::lower_bound(firstConflicts.begin(), firstConflicts.end(), second);
  const auto secondPosition =
      std::lower_bound(secondConflicts.begin(), secondConflicts.end(), first);
  const bool firstMissing =
      firstPosition == firstConflicts.end() || *firstPosition != second;
  const bool secondMissing =
      secondPosition == secondConflicts.end() || *secondPosition != first;
  if (firstMissing != secondMissing) {
    constructionValidated_ = false;
    return makeResult(SyncCoverMechanismError::InvalidConflict);
  }
  if (!firstMissing) {
    return makeResult(SyncCoverMechanismError::None, first);
  }

  const auto growConflicts = [](std::vector<SyncCoverMechanismId> &conflicts) {
    if (conflicts.size() != conflicts.capacity()) {
      return;
    }
    const std::size_t doubled =
        conflicts.size() > conflicts.max_size() / 2
            ? conflicts.max_size()
            : std::max<std::size_t>(4, conflicts.size() * 2);
    conflicts.reserve(doubled);
  };
  growConflicts(firstConflicts);
  growConflicts(secondConflicts);
  const auto refreshedFirst =
      std::lower_bound(firstConflicts.begin(), firstConflicts.end(), second);
  const auto refreshedSecond =
      std::lower_bound(secondConflicts.begin(), secondConflicts.end(), first);
  firstConflicts.insert(refreshedFirst, second);
  secondConflicts.insert(refreshedSecond, first);
  noteSuccessfulMutation();
  return makeResult(SyncCoverMechanismError::None, first);
}

SyncCoverMechanismResult SyncCoverMechanismUniverse::validate() const {
  const std::size_t graphGeneration = graph_.getGeneration();
  if (cachedValidationVersion_ == version_ &&
      cachedGraphGeneration_ == graphGeneration) {
    return cachedValidation_;
  }
  cachedValidation_ = validateUncached();
  cachedValidationVersion_ = version_;
  cachedGraphGeneration_ = graphGeneration;
  return cachedValidation_;
}

SyncCoverMechanismResult
SyncCoverMechanismUniverse::validateUncached() const {
  ++fullValidationCount_;
  const bool invalidGraph = !graph_.isStructureFrozen() || !graph_.validate();
  if (invalidGraph) {
    return makeResult(SyncCoverMechanismError::InvalidGraph);
  }
  std::vector<std::uint64_t> tokenPools;
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
    if (domain.kind == SyncCoverResourceKind::BufferToken) {
      const bool duplicatePool =
          std::find(tokenPools.begin(), tokenPools.end(),
                    domain.poolIdentity) != tokenPools.end();
      if (duplicatePool) {
        return makeResult(SyncCoverMechanismError::InvalidDomain, index);
      }
      tokenPools.push_back(domain.poolIdentity);
    }
  }

  std::vector<bool> claimedEdges(graph_.getEdges().size(), false);
  for (std::size_t index = 0; index < mechanisms_.size(); ++index) {
    const SyncCoverMechanism &mechanism = mechanisms_[index];
    if (mechanism.id != index ||
        (mechanism.supplyEdges.empty() &&
         mechanism.verifiedCoverageEdges.empty())) {
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
    if (mechanism.kind != SyncCoverMechanismKind::VerifiedProtocol &&
        !mechanism.verifiedCoverageEdges.empty()) {
      return makeResult(SyncCoverMechanismError::UnverifiedProtocol, index);
    }
    if (!validResourceComposition(domains_, mechanism.kind, mechanism.actions,
                                  mechanism.resourceUses)) {
      return makeResult(SyncCoverMechanismError::InvalidResourceUse, index);
    }

    for (const SyncCoverResourceUse &use : mechanism.resourceUses) {
      const SyncCoverMechanismError error =
          validateResourceUse(mechanism.kind, use, graph_.getEdges(),
                              mechanism.actions);
      if (error != SyncCoverMechanismError::None) {
        return makeResult(error, index);
      }
    }
    SyncCoverMechanismError mechanismError = validateActionAccounting(
        domains_, mechanism.actions, mechanism.resourceUses);
    if (mechanismError == SyncCoverMechanismError::None) {
      mechanismError = validateSupplyBindings(
          mechanism.kind, graph_.getEdges(), mechanism.supplyEdges,
          mechanism.actions, mechanism.resourceUses,
          mechanism.supplyBindings);
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
    for (std::size_t edgeIndex : mechanism.verifiedCoverageEdges) {
      const bool invalidEdge =
          edgeIndex >= graph_.getEdges().size() || claimedEdges[edgeIndex];
      if (invalidEdge) {
        return makeResult(SyncCoverMechanismError::InvalidSupply, index);
      }
      const SyncCoverEdge &edge = graph_.getEdges()[edgeIndex];
      if (edge.kind != SyncCoverEdgeKind::CompletionSupply ||
          edge.mechanism != mechanism.id) {
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
    SyncCoverMechanismKind kind, const SyncCoverResourceUse &use,
    const std::vector<SyncCoverEdge> &edges,
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
    if (!validResourceLifetime(graph_, kind, use, edge)) {
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
    if (wrongKinds || !validResourceLifetime(graph_, kind, use, edge)) {
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
  const std::optional<SyncCoverScopeId> anchorScope =
      getAnchorScope(graph_, placement.anchor);
  const bool invalidPlacement =
      placement.scope >= graph_.getScopes().size() ||
      !resolveSyncCoverAnchor(graph_, placement.anchor) || !anchorScope ||
      *anchorScope != placement.scope;
  if (invalidPlacement) {
    return SyncCoverMechanismError::InvalidMechanism;
  }
  for (const SyncCoverEdge &edge : descriptor.supplyEdges) {
    if (!syncCoverBarrierCanSupply(graph_, placement, edge)) {
      return SyncCoverMechanismError::InvalidSupply;
    }
  }
  return SyncCoverMechanismError::None;
}
