// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir::pto;

namespace {

constexpr std::uint64_t kHashOffset = 1469598103934665603ULL;
constexpr std::uint64_t kHashPrime = 1099511628211ULL;

void hashValue(std::uint64_t &hash, std::uint64_t value) {
  for (unsigned byte = 0; byte < sizeof(value); ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffU;
    hash *= kHashPrime;
  }
}

void hashGuard(std::uint64_t &hash, const SyncCoverGuard &guard) {
  hashValue(hash, guard.literals.size());
  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    hashValue(hash, literal.control);
    hashValue(hash, literal.alternative);
  }
}

void hashEdge(std::uint64_t &hash, const SyncCoverEdge &edge) {
  hashValue(hash, edge.source);
  hashValue(hash, edge.target);
  hashValue(hash, edge.scope);
  hashValue(hash, edge.distance);
  hashValue(hash, static_cast<std::uint8_t>(edge.kind));
  hashGuard(hash, edge.sourceGuard);
  hashGuard(hash, edge.targetGuard);
}

bool edgeLess(const SyncCoverEdge &left, const SyncCoverEdge &right) {
  return std::tie(left.source, left.target, left.scope, left.distance,
                  left.kind, left.sourceGuard.literals,
                  left.targetGuard.literals) <
         std::tie(right.source, right.target, right.scope, right.distance,
                  right.kind, right.sourceGuard.literals,
                  right.targetGuard.literals);
}

bool edgeEqual(const SyncCoverEdge &left, const SyncCoverEdge &right) {
  return !edgeLess(left, right) && !edgeLess(right, left);
}

bool bindingLess(const CanonicalSyncSupplyBinding &left,
                 const CanonicalSyncSupplyBinding &right) {
  if (edgeLess(left.edge, right.edge)) {
    return true;
  }
  if (edgeLess(right.edge, left.edge)) {
    return false;
  }
  return std::tie(left.allowedDemands, left.eventUse, left.barrierAction,
                  left.produceAction, left.consumeAction, left.proof) <
         std::tie(right.allowedDemands, right.eventUse, right.barrierAction,
                  right.produceAction, right.consumeAction, right.proof);
}

bool bindingEqual(const CanonicalSyncSupplyBinding &left,
                  const CanonicalSyncSupplyBinding &right) {
  return edgeEqual(left.edge, right.edge) &&
         std::tie(left.allowedDemands, left.eventUse, left.barrierAction,
                  left.produceAction, left.consumeAction, left.proof) ==
             std::tie(right.allowedDemands, right.eventUse, right.barrierAction,
                      right.produceAction, right.consumeAction, right.proof);
}

bool actionEqual(const CanonicalSyncAction &left,
                 const CanonicalSyncAction &right) {
  return std::tie(left.kind, left.resource, left.anchor.kind, left.anchor.node,
                  left.anchor.scope, left.anchor.position, left.eventUse,
                  left.eventLane, left.drainedResources, left.barrierKind,
                  left.guard, left.guardScope) ==
         std::tie(right.kind, right.resource, right.anchor.kind,
                  right.anchor.node, right.anchor.scope, right.anchor.position,
                  right.eventUse, right.eventLane, right.drainedResources,
                  right.barrierKind, right.guard, right.guardScope);
}

bool descriptorEqual(const CanonicalSyncMechanismDescriptor &left,
                     const CanonicalSyncMechanismDescriptor &right) {
  if (left.kind != right.kind ||
      left.supplies.size() != right.supplies.size() ||
      left.eventUses.size() != right.eventUses.size() ||
      left.actions.size() != right.actions.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.supplies.size(); ++index) {
    if (!bindingEqual(left.supplies[index], right.supplies[index])) {
      return false;
    }
  }
  for (std::size_t index = 0; index < left.eventUses.size(); ++index) {
    const CanonicalSyncEventUse &first = left.eventUses[index];
    const CanonicalSyncEventUse &second = right.eventUses[index];
    const bool different =
        std::tie(first.domain, first.width, first.recurrenceScope,
                 first.lifetimeScope) != std::tie(second.domain, second.width,
                                                  second.recurrenceScope,
                                                  second.lifetimeScope);
    if (different) {
      return false;
    }
  }
  for (std::size_t index = 0; index < left.actions.size(); ++index) {
    if (!actionEqual(left.actions[index], right.actions[index])) {
      return false;
    }
  }
  return true;
}

std::uint64_t
descriptorHash(const CanonicalSyncMechanismDescriptor &descriptor) {
  std::uint64_t hash = kHashOffset;
  hashValue(hash, static_cast<std::uint8_t>(descriptor.kind));
  for (const CanonicalSyncSupplyBinding &binding : descriptor.supplies) {
    hashEdge(hash, binding.edge);
    for (SyncCoverDemandId demand : binding.allowedDemands) {
      hashValue(hash, demand);
    }
    hashValue(hash, std::numeric_limits<std::size_t>::max());
    hashValue(hash, binding.eventUse.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, binding.barrierAction.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, binding.produceAction.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, binding.consumeAction.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, static_cast<std::uint8_t>(binding.proof));
  }
  for (const CanonicalSyncEventUse &use : descriptor.eventUses) {
    hashValue(hash, use.domain);
    hashValue(hash, use.width);
    hashValue(hash, use.recurrenceScope.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, use.lifetimeScope.value_or(
                        std::numeric_limits<std::size_t>::max()));
  }
  for (const CanonicalSyncAction &action : descriptor.actions) {
    hashValue(hash, static_cast<std::uint8_t>(action.kind));
    hashValue(hash, action.resource);
    hashValue(hash, static_cast<std::uint8_t>(action.anchor.kind));
    hashValue(hash, action.anchor.node);
    hashValue(hash, action.anchor.scope);
    hashValue(hash, action.anchor.position);
    hashValue(hash, action.eventUse.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, action.eventLane);
    for (std::uint32_t resource : action.drainedResources) {
      hashValue(hash, resource);
    }
    hashValue(hash, static_cast<std::uint8_t>(action.barrierKind));
    hashValue(hash, static_cast<std::uint8_t>(action.guard));
    hashValue(hash, action.guardScope.value_or(
                        std::numeric_limits<std::size_t>::max()));
  }
  return hash;
}

std::optional<SyncCoverScopeId>
getActionScope(const SyncCoverGraph &graph, const CanonicalSyncAction &action) {
  switch (action.anchor.kind) {
  case SyncCoverAnchorKind::BeforeNode:
  case SyncCoverAnchorKind::AfterNode:
    if (action.anchor.node < graph.getNodes().size()) {
      return graph.getNodes()[action.anchor.node].scope;
    }
    return std::nullopt;
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit:
  case SyncCoverAnchorKind::TimelinePoint:
    return action.anchor.scope < graph.getScopes().size()
               ? std::optional<SyncCoverScopeId>(action.anchor.scope)
               : std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::size_t> getCostDepth(const SyncCoverGraph &graph,
                                        const CanonicalSyncAction &action,
                                        SyncCoverScopeId scope) {
  const bool outsideScope =
      action.anchor.kind == SyncCoverAnchorKind::ScopeEntry ||
      action.anchor.kind == SyncCoverAnchorKind::ScopeExit;
  return graph.getScopeLoopDepth(scope, !outsideScope);
}

bool checkedAdd(std::size_t first, std::size_t second, std::size_t &result) {
  const bool overflows =
      second > std::numeric_limits<std::size_t>::max() - first;
  if (overflows) {
    return false;
  }
  result = first + second;
  return true;
}

bool checkedIncrement(std::uint64_t &value, std::size_t amount) {
  const bool overflows =
      amount > std::numeric_limits<std::uint64_t>::max() - value;
  if (overflows) {
    return false;
  }
  value += amount;
  return true;
}

SyncCoverDemandSet
projectCoverage(const SyncCoverDemandSet &all,
                const std::vector<SyncCoverDemandId> &activeDemands) {
  SyncCoverDemandSet result(activeDemands.size());
  for (std::size_t local = 0; local < activeDemands.size(); ++local) {
    if (all.contains(activeDemands[local])) {
      result.insert(local);
    }
  }
  return result;
}

std::vector<std::uint32_t> getIssueResources(const SyncCoverGraph &graph) {
  std::vector<std::uint32_t> result;
  result.reserve(graph.getNodes().size());
  for (const SyncCoverNode &node : graph.getNodes()) {
    result.push_back(node.resource);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

unsigned patternPriority(CanonicalSyncPatternKind kind) {
  switch (kind) {
  case CanonicalSyncPatternKind::Singleton:
    return 0;
  case CanonicalSyncPatternKind::OwnershipCycle:
    return 1;
  case CanonicalSyncPatternKind::SlotLifecycle:
    return 2;
  case CanonicalSyncPatternKind::RoundTrip:
    return 3;
  case CanonicalSyncPatternKind::PipelineScope:
    return 4;
  }
  return std::numeric_limits<unsigned>::max();
}

std::vector<SyncCoverCompletionSupply>
getSupplies(const std::vector<CanonicalSyncMechanism> &mechanisms,
            const std::vector<CanonicalSyncMechanismId> &members) {
  std::vector<SyncCoverCompletionSupply> result;
  for (CanonicalSyncMechanismId member : members) {
    for (const CanonicalSyncSupplyBinding &binding :
         mechanisms[member].descriptor.supplies) {
      result.push_back({member, binding.edge, binding.allowedDemands});
    }
  }
  return result;
}

struct MechanismValidationState {
  std::vector<CanonicalSyncEventLifetime> lifetimes;
  CanonicalSyncMechanismCost cost;
  std::vector<std::size_t> setCounts;
  std::vector<std::size_t> waitCounts;
  std::vector<std::optional<std::size_t>> setActions;
  std::vector<std::optional<std::size_t>> waitActions;
  std::vector<std::vector<bool>> setLanes;
  std::vector<std::vector<bool>> waitLanes;
};

CanonicalSyncProblemError
validateSupplyDeclarations(const SyncCoverGraph &graph,
                           CanonicalSyncMechanismDescriptor &descriptor) {
  const bool protocol = descriptor.kind == CanonicalSyncMechanismKind::Protocol;
  for (CanonicalSyncSupplyBinding &binding : descriptor.supplies) {
    const bool invalidDemands =
        !std::is_sorted(binding.allowedDemands.begin(),
                        binding.allowedDemands.end()) ||
        std::adjacent_find(binding.allowedDemands.begin(),
                           binding.allowedDemands.end()) !=
            binding.allowedDemands.end() ||
        std::any_of(binding.allowedDemands.begin(),
                    binding.allowedDemands.end(), [&](SyncCoverDemandId id) {
                      return id >= graph.getDemands().size();
                    });
    const bool direct = binding.proof == CanonicalSyncSupplyProof::DirectAction;
    const bool verified =
        binding.proof == CanonicalSyncSupplyProof::VerifiedProtocol;
    const bool composite =
        binding.proof == CanonicalSyncSupplyProof::VerifiedCompositeProtocol;
    const bool nestedSummary =
        binding.proof ==
        CanonicalSyncSupplyProof::VerifiedNestedRecurrenceSummary;
    const bool ownershipClosure =
        binding.proof == CanonicalSyncSupplyProof::VerifiedOwnershipClosure;
    const bool validOwner =
        protocol ? (verified && binding.eventUse && !binding.barrierAction &&
                    binding.produceAction && binding.consumeAction) ||
                       ((composite || nestedSummary || ownershipClosure) &&
                        !binding.eventUse && !binding.barrierAction &&
                        !binding.produceAction && !binding.consumeAction)
                 : direct &&
                       (binding.eventUse.has_value() !=
                        binding.barrierAction.has_value()) &&
                       !binding.produceAction && !binding.consumeAction;
    const bool invalid = graph.canonicalizeCompletionEdge(binding.edge) !=
                             SyncCoverGraphError::None ||
                         invalidDemands ||
                         (!binding.allowedDemands.empty() && !protocol) ||
                         !validOwner;
    if (invalid) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
  }
  std::sort(descriptor.supplies.begin(), descriptor.supplies.end(),
            bindingLess);
  const bool duplicate =
      std::adjacent_find(descriptor.supplies.begin(), descriptor.supplies.end(),
                         [](const auto &first, const auto &second) {
                           return edgeEqual(first.edge, second.edge);
                         }) != descriptor.supplies.end();
  return duplicate ? CanonicalSyncProblemError::InvalidMechanism
                   : CanonicalSyncProblemError::None;
}

CanonicalSyncProblemError validateEventUseDeclarations(
    const SyncCoverGraph &graph,
    const std::vector<CanonicalSyncEventDomain> &domains,
    const CanonicalSyncMechanismDescriptor &descriptor,
    MechanismValidationState &state) {
  const std::size_t useCount = descriptor.eventUses.size();
  state.setCounts.assign(useCount, 0);
  state.waitCounts.assign(useCount, 0);
  state.setActions.assign(useCount, std::nullopt);
  state.waitActions.assign(useCount, std::nullopt);
  state.lifetimes.assign(
      useCount, {std::numeric_limits<SyncCoverTimelinePosition>::max(), 0});
  for (const CanonicalSyncEventUse &use : descriptor.eventUses) {
    const std::size_t available =
        use.domain < domains.size()
            ? domains[use.domain].budget -
                  static_cast<std::size_t>(std::count_if(
                      domains[use.domain].reservedIds.begin(),
                      domains[use.domain].reservedIds.end(),
                      [&](unsigned id) {
                        return id < domains[use.domain].budget;
                      }))
            : 0;
    const auto validLoopScope = [&](std::optional<SyncCoverScopeId> scope) {
      return !scope || (*scope < graph.getScopes().size() &&
                        graph.getScopes()[*scope].isLoop &&
                        graph.getScopes()[*scope].timeline);
    };
    const bool invalid =
        use.domain >= domains.size() || use.width == 0 ||
        use.width > available ||
        (descriptor.kind == CanonicalSyncMechanismKind::Event &&
         use.width != 1) ||
        !validLoopScope(use.recurrenceScope) ||
        !validLoopScope(use.lifetimeScope) ||
        (use.lifetimeScope &&
         (!use.recurrenceScope ||
          !graph.scopeContains(*use.lifetimeScope, *use.recurrenceScope)));
    if (invalid) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
    state.setLanes.emplace_back(use.width, false);
    state.waitLanes.emplace_back(use.width, false);
  }
  return CanonicalSyncProblemError::None;
}

CanonicalSyncProblemError
validateActions(const SyncCoverGraph &graph,
                const std::vector<CanonicalSyncEventDomain> &domains,
                const CanonicalSyncPatternProblem::Limits &limits,
                const std::vector<std::uint32_t> &issueResources,
                CanonicalSyncMechanismDescriptor &descriptor,
                MechanismValidationState &state) {
  for (std::size_t index = 0; index < descriptor.actions.size(); ++index) {
    CanonicalSyncAction &action = descriptor.actions[index];
    const std::optional<SyncCoverTimelinePosition> position =
        resolveSyncCoverAnchor(graph, action.anchor);
    const std::optional<SyncCoverScopeId> scope = getActionScope(graph, action);
    if (!position || !scope) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
    const std::optional<std::size_t> depth =
        getCostDepth(graph, action, *scope);
    const bool unguarded = action.guard == CanonicalSyncActionGuardKind::None;
    const bool loopGuard = action.guardScope &&
                           *action.guardScope < graph.getScopes().size() &&
                           graph.getScopes()[*action.guardScope].isLoop &&
                           action.kind != CanonicalSyncActionKind::Barrier;
    const bool scopeBoundary =
        action.guardScope &&
        (action.anchor.kind == SyncCoverAnchorKind::ScopeEntry ||
         action.anchor.kind == SyncCoverAnchorKind::ScopeExit) &&
        action.anchor.scope == *action.guardScope;
    const bool nodeBoundary =
        action.guardScope &&
        (action.anchor.kind == SyncCoverAnchorKind::BeforeNode ||
         action.anchor.kind == SyncCoverAnchorKind::AfterNode) &&
        graph.scopeContains(*action.guardScope, *scope);
    const bool nestedScopeBoundary =
        action.guardScope && *scope != *action.guardScope &&
        (action.anchor.kind == SyncCoverAnchorKind::ScopeEntry ||
         action.anchor.kind == SyncCoverAnchorKind::ScopeExit) &&
        graph.scopeContains(*action.guardScope, *scope);
    const bool validGuard =
        unguarded
            ? !action.guardScope
            : loopGuard &&
                  (((action.guard ==
                         CanonicalSyncActionGuardKind::LoopNonEmpty ||
                     action.guard == CanonicalSyncActionGuardKind::LoopEmpty) &&
                    scopeBoundary) ||
                   ((action.guard ==
                         CanonicalSyncActionGuardKind::NotFirstIteration ||
                     action.guard ==
                         CanonicalSyncActionGuardKind::HasSuccessor) &&
                    (nodeBoundary || nestedScopeBoundary)));
    if (!depth || !validGuard) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
    state.cost.barrierActions.resize(
        std::max(state.cost.barrierActions.size(), *depth + 1), 0);
    state.cost.eventActions.resize(
        std::max(state.cost.eventActions.size(), *depth + 1), 0);

    if (action.kind == CanonicalSyncActionKind::Barrier) {
      std::sort(action.drainedResources.begin(), action.drainedResources.end());
      action.drainedResources.erase(std::unique(action.drainedResources.begin(),
                                                action.drainedResources.end()),
                                    action.drainedResources.end());
      const bool targeted =
          action.barrierKind == CanonicalSyncBarrierKind::Targeted;
      const bool all = action.barrierKind == CanonicalSyncBarrierKind::All;
      const bool validDrain =
          targeted ? action.drainedResources.size() == 1 &&
                         action.drainedResources.front() == action.resource
                   : all && action.drainedResources == issueResources;
      const bool invalid = action.eventUse || action.eventLane != 0 ||
                           action.drainedResources.empty() ||
                           action.drainedResources.size() >
                               limits.maximumDrainedResourcesPerBarrier ||
                           !validDrain ||
                           !checkedIncrement(state.cost.barrierActions[*depth],
                                             action.drainedResources.size());
      if (invalid) {
        return CanonicalSyncProblemError::InvalidMechanism;
      }
      continue;
    }

    const bool invalidUse =
        !action.eventUse || *action.eventUse >= descriptor.eventUses.size() ||
        !action.drainedResources.empty() ||
        action.barrierKind != CanonicalSyncBarrierKind::Targeted;
    if (invalidUse) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
    const CanonicalSyncEventUse &use = descriptor.eventUses[*action.eventUse];
    const CanonicalSyncEventDomain &domain = domains[use.domain];
    const bool set = action.kind == CanonicalSyncActionKind::EventSet;
    const bool wait = action.kind == CanonicalSyncActionKind::EventWait;
    const bool invalidAction =
        (!set && !wait) || action.eventLane >= use.width ||
        action.resource !=
            (set ? domain.sourceResource : domain.targetResource) ||
        !checkedIncrement(state.cost.eventActions[*depth], 1);
    if (invalidAction) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
    std::vector<std::size_t> &counts = set ? state.setCounts : state.waitCounts;
    std::vector<std::optional<std::size_t>> &indices =
        set ? state.setActions : state.waitActions;
    ++counts[*action.eventUse];
    indices[*action.eventUse] = index;
    (set ? state.setLanes
         : state.waitLanes)[*action.eventUse][action.eventLane] = true;
    CanonicalSyncEventLifetime &lifetime = state.lifetimes[*action.eventUse];
    lifetime.begin = std::min(lifetime.begin, *position);
    lifetime.end = std::max(lifetime.end, *position);
    const std::optional<SyncCoverScopeId> lifetimeScope =
        use.lifetimeScope ? use.lifetimeScope : use.recurrenceScope;
    if (lifetimeScope && !graph.scopeContains(*lifetimeScope, *scope)) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
  }
  return CanonicalSyncProblemError::None;
}

CanonicalSyncProblemError
validateMechanismShape(const CanonicalSyncMechanismDescriptor &descriptor,
                       const MechanismValidationState &state) {
  const bool barrier = descriptor.kind == CanonicalSyncMechanismKind::Barrier;
  const bool wrongActionKinds = std::any_of(
      descriptor.actions.begin(), descriptor.actions.end(),
      [&](const CanonicalSyncAction &action) {
        return (action.kind == CanonicalSyncActionKind::Barrier) != barrier;
      });
  const bool incompleteEvents =
      std::any_of(state.setCounts.begin(), state.setCounts.end(),
                  [](std::size_t count) { return count != 1; }) ||
      std::any_of(state.waitCounts.begin(), state.waitCounts.end(),
                  [](std::size_t count) { return count != 1; });
  const auto hasMissingLane = [](const auto &lanes) {
    return std::find(lanes.begin(), lanes.end(), false) != lanes.end();
  };
  const bool incompleteProtocolLanes =
      descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
      (std::any_of(state.setLanes.begin(), state.setLanes.end(),
                   hasMissingLane) ||
       std::any_of(state.waitLanes.begin(), state.waitLanes.end(),
                   hasMissingLane));
  const bool invalid =
      barrier != descriptor.eventUses.empty() || wrongActionKinds ||
      (descriptor.kind != CanonicalSyncMechanismKind::Protocol &&
       incompleteEvents) ||
      incompleteProtocolLanes;
  return invalid ? CanonicalSyncProblemError::InvalidMechanism
                 : CanonicalSyncProblemError::None;
}

CanonicalSyncProblemError
validateSupplyBindings(const SyncCoverGraph &graph,
                       const std::vector<CanonicalSyncEventDomain> &domains,
                       const CanonicalSyncMechanismDescriptor &descriptor,
                       const MechanismValidationState &state) {
  const bool barrier = descriptor.kind == CanonicalSyncMechanismKind::Barrier;
  std::vector<std::size_t> supplyCounts(descriptor.eventUses.size(), 0);
  for (const CanonicalSyncSupplyBinding &binding : descriptor.supplies) {
    const SyncCoverEdge &edge = binding.edge;
    if (barrier) {
      if (binding.eventUse || !binding.barrierAction ||
          *binding.barrierAction >= descriptor.actions.size()) {
        return CanonicalSyncProblemError::InvalidMechanism;
      }
      const CanonicalSyncAction &action =
          descriptor.actions[*binding.barrierAction];
      const std::uint32_t sourceResource =
          graph.getNodes()[edge.source].resource;
      const std::uint32_t targetResource =
          graph.getNodes()[edge.target].resource;
      const bool sourceDrained =
          std::binary_search(action.drainedResources.begin(),
                             action.drainedResources.end(), sourceResource);
      const bool crosses =
          action.anchor.kind == SyncCoverAnchorKind::BeforeNode &&
          action.anchor.node == edge.target;
      if (!sourceDrained || !crosses || action.resource != targetResource) {
        return CanonicalSyncProblemError::InvalidMechanism;
      }
      continue;
    }

    const bool protocol =
        descriptor.kind == CanonicalSyncMechanismKind::Protocol;
    const bool trustedProtocolProof =
        binding.proof == CanonicalSyncSupplyProof::VerifiedCompositeProtocol ||
        binding.proof ==
            CanonicalSyncSupplyProof::VerifiedNestedRecurrenceSummary ||
        binding.proof == CanonicalSyncSupplyProof::VerifiedOwnershipClosure;
    if (protocol && trustedProtocolProof) {
      continue;
    }

    if (!binding.eventUse || *binding.eventUse >= descriptor.eventUses.size()) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
    ++supplyCounts[*binding.eventUse];
    const CanonicalSyncEventUse &use = descriptor.eventUses[*binding.eventUse];
    const CanonicalSyncEventDomain &domain = domains[use.domain];
    const std::size_t produce = protocol ? *binding.produceAction
                                         : *state.setActions[*binding.eventUse];
    const std::size_t consume = protocol
                                    ? *binding.consumeAction
                                    : *state.waitActions[*binding.eventUse];
    const bool invalidActionIndex = produce >= descriptor.actions.size() ||
                                    consume >= descriptor.actions.size();
    if (invalidActionIndex) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
    const CanonicalSyncAction &set = descriptor.actions[produce];
    const CanonicalSyncAction &wait = descriptor.actions[consume];
    const bool wrongActions = set.kind != CanonicalSyncActionKind::EventSet ||
                              wait.kind != CanonicalSyncActionKind::EventWait ||
                              set.eventUse != binding.eventUse ||
                              wait.eventUse != binding.eventUse ||
                              set.eventLane != wait.eventLane;
    const bool wrongResources =
        domain.sourceResource != graph.getNodes()[edge.source].resource ||
        domain.targetResource != graph.getNodes()[edge.target].resource;
    if (wrongActions || wrongResources) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
    if (!protocol) {
      const bool wrongAnchors =
          set.anchor.kind != SyncCoverAnchorKind::AfterNode ||
          set.anchor.node != edge.source ||
          wait.anchor.kind != SyncCoverAnchorKind::BeforeNode ||
          wait.anchor.node != edge.target;
      if (edge.distance != 0 || use.recurrenceScope || use.lifetimeScope ||
          wrongAnchors ||
          !syncCoverNodeCanProduceCompletion(graph, edge.source,
                                             domain.targetResource) ||
          !syncCoverEndpointsCoExecute(graph, edge)) {
        return CanonicalSyncProblemError::InvalidMechanism;
      }
    } else if (!use.recurrenceScope) {
      return CanonicalSyncProblemError::InvalidMechanism;
    } else if (edge.distance == 0) {
      const SyncCoverScopeId lifetimeScope =
          use.lifetimeScope.value_or(*use.recurrenceScope);
      const bool relatedScope =
          graph.scopeContains(lifetimeScope, edge.scope) ||
          graph.scopeContains(edge.scope, lifetimeScope);
      if (!relatedScope) {
        return CanonicalSyncProblemError::InvalidMechanism;
      }
    } else if (use.lifetimeScope) {
      const bool invalidLifetimeSupply =
          edge.scope >= graph.getScopes().size() ||
          !graph.getScopes()[edge.scope].isLoop ||
          !graph.scopeContains(*use.lifetimeScope, edge.scope);
      if (invalidLifetimeSupply) {
        return CanonicalSyncProblemError::InvalidMechanism;
      }
    } else if (*use.recurrenceScope != edge.scope) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
  }
  const bool invalidSupplyCount =
      !barrier &&
      std::any_of(supplyCounts.begin(), supplyCounts.end(),
                  [&](std::size_t count) {
                    return descriptor.kind == CanonicalSyncMechanismKind::Event
                               ? count != 1
                               : count == 0;
                  });
  return invalidSupplyCount ? CanonicalSyncProblemError::InvalidMechanism
                            : CanonicalSyncProblemError::None;
}

void setRecurrenceLifetimes(const SyncCoverGraph &graph,
                            const CanonicalSyncMechanismDescriptor &descriptor,
                            MechanismValidationState &state) {
  for (std::size_t use = 0; use < descriptor.eventUses.size(); ++use) {
    const CanonicalSyncEventUse &eventUse = descriptor.eventUses[use];
    const std::optional<SyncCoverScopeId> lifetimeScope =
        eventUse.lifetimeScope ? eventUse.lifetimeScope
                               : eventUse.recurrenceScope;
    if (lifetimeScope) {
      const SyncCoverTimelineInterval &timeline =
          *graph.getScopes()[*lifetimeScope].timeline;
      state.lifetimes[use] = {timeline.begin, timeline.end};
    }
  }
}

} // namespace

CanonicalSyncPatternProblem::CanonicalSyncPatternProblem(
    const SyncCoverGraph &graph, std::vector<SyncCoverDemandId> activeDemands)
    : CanonicalSyncPatternProblem(graph, std::move(activeDemands), Limits{}) {}

CanonicalSyncPatternProblem::CanonicalSyncPatternProblem(
    const SyncCoverGraph &graph, std::vector<SyncCoverDemandId> activeDemands,
    Limits limits, SyncCoverExpansionLimits expansionLimits)
    : graph_(graph), expansion_(graph, activeDemands, expansionLimits),
      limits_(limits), issueResources_(getIssueResources(graph)),
      activeDemands_(std::move(activeDemands)) {
  const bool normalized =
      std::is_sorted(activeDemands_.begin(), activeDemands_.end()) &&
      std::adjacent_find(activeDemands_.begin(), activeDemands_.end()) ==
          activeDemands_.end();
  const bool inRange = std::all_of(activeDemands_.begin(), activeDemands_.end(),
                                   [&](SyncCoverDemandId demand) {
                                     return demand < graph_.getDemands().size();
                                   });
  graphValid_ = graph_.isStructureFrozen() && graph_.validate() &&
                expansion_.isForGraph(graph_) && normalized && inRange;
}

CanonicalSyncProblemResult
CanonicalSyncPatternProblem::addEventDomain(CanonicalSyncEventDomain domain) {
  if (frozen_) {
    return {CanonicalSyncProblemError::Frozen, domains_.size()};
  }
  const bool duplicate =
      std::any_of(domains_.begin(), domains_.end(), [&](const auto &existing) {
        return existing.sourceResource == domain.sourceResource &&
               existing.targetResource == domain.targetResource;
      });
  const bool invalid = !graphValid_ || domain.id != domains_.size() ||
                       domain.budget == 0 || duplicate;
  if (invalid) {
    return {CanonicalSyncProblemError::InvalidDomain, domains_.size()};
  }
  const bool domainLimitExceeded =
      domains_.size() >= limits_.maximumDomains ||
      domain.budget > limits_.maximumEventBudget ||
      domain.reservedIds.size() > limits_.maximumReservedEventIds;
  if (domainLimitExceeded) {
    return {CanonicalSyncProblemError::LimitExceeded, domains_.size()};
  }
  std::sort(domain.reservedIds.begin(), domain.reservedIds.end());
  domain.reservedIds.erase(
      std::unique(domain.reservedIds.begin(), domain.reservedIds.end()),
      domain.reservedIds.end());
  domains_.push_back(std::move(domain));
  return {CanonicalSyncProblemError::None, domains_.size() - 1};
}

CanonicalSyncProblemResult
CanonicalSyncPatternProblem::validateAndCostMechanism(
    CanonicalSyncMechanismDescriptor &descriptor,
    std::vector<CanonicalSyncEventLifetime> &lifetimes,
    CanonicalSyncMechanismCost &cost, bool protocolVerified) const {
  if (!graphValid_) {
    return {CanonicalSyncProblemError::InvalidGraph, std::nullopt};
  }
  if (descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
      !protocolVerified) {
    return {CanonicalSyncProblemError::UnverifiedProtocol, std::nullopt};
  }
  const bool emptyMechanism =
      descriptor.supplies.empty() || descriptor.actions.empty();
  if (emptyMechanism) {
    return {CanonicalSyncProblemError::InvalidMechanism, std::nullopt};
  }
  const bool mechanismLimitExceeded =
      descriptor.supplies.size() > limits_.maximumSuppliesPerMechanism ||
      descriptor.eventUses.size() > limits_.maximumEventUsesPerMechanism ||
      descriptor.actions.size() > limits_.maximumActionsPerMechanism;
  if (mechanismLimitExceeded) {
    return {CanonicalSyncProblemError::LimitExceeded, std::nullopt};
  }

  CanonicalSyncProblemError validation =
      validateSupplyDeclarations(graph_, descriptor);
  if (validation != CanonicalSyncProblemError::None) {
    return {validation, std::nullopt};
  }
  MechanismValidationState state;
  validation =
      validateEventUseDeclarations(graph_, domains_, descriptor, state);
  if (validation != CanonicalSyncProblemError::None) {
    return {validation, std::nullopt};
  }

  validation = validateActions(graph_, domains_, limits_, issueResources_,
                               descriptor, state);
  if (validation != CanonicalSyncProblemError::None) {
    return {validation, std::nullopt};
  }
  validation = validateMechanismShape(descriptor, state);
  if (validation != CanonicalSyncProblemError::None) {
    return {validation, std::nullopt};
  }

  validation = validateSupplyBindings(graph_, domains_, descriptor, state);
  if (validation != CanonicalSyncProblemError::None) {
    return {validation, std::nullopt};
  }
  setRecurrenceLifetimes(graph_, descriptor, state);
  lifetimes = std::move(state.lifetimes);
  cost = std::move(state.cost);
  return {};
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::internMechanism(
    CanonicalSyncMechanismDescriptor descriptor) {
  return internMechanismImpl(std::move(descriptor), false);
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::internVerifiedProtocol(
    CanonicalSyncMechanismDescriptor descriptor,
    const std::function<bool(const CanonicalSyncMechanismDescriptor &)>
        &verifier) {
  return internMechanismImpl(std::move(descriptor), true, verifier);
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::internMechanismImpl(
    CanonicalSyncMechanismDescriptor descriptor, bool protocolVerified,
    const std::function<bool(const CanonicalSyncMechanismDescriptor &)>
        &verifier) {
  if (frozen_) {
    return {CanonicalSyncProblemError::Frozen, mechanisms_.size()};
  }
  CanonicalSyncMechanismCost cost;
  std::vector<CanonicalSyncEventLifetime> lifetimes;
  CanonicalSyncProblemResult validated =
      validateAndCostMechanism(descriptor, lifetimes, cost, protocolVerified);
  if (!validated) {
    return validated;
  }
  if (protocolVerified && (!verifier || !verifier(descriptor))) {
    return {CanonicalSyncProblemError::UnverifiedProtocol, std::nullopt};
  }
  const std::uint64_t hash = descriptorHash(descriptor);
  const auto bucket = mechanismBuckets_.find(hash);
  if (bucket != mechanismBuckets_.end()) {
    for (CanonicalSyncMechanismId mechanism : bucket->second) {
      if (descriptorEqual(mechanisms_[mechanism].descriptor, descriptor)) {
        return {CanonicalSyncProblemError::None, mechanism};
      }
    }
  }
  const bool mechanismLimitReached =
      mechanisms_.size() >= limits_.maximumMechanisms;
  const bool patternLimitReached =
      mechanisms_.size() >= limits_.maximumPatterns ||
      patternSpecs_.size() >= limits_.maximumPatterns - mechanisms_.size();
  if (mechanismLimitReached || patternLimitReached) {
    return {CanonicalSyncProblemError::LimitExceeded, mechanisms_.size()};
  }
  std::size_t nextActions = 0;
  std::size_t nextUses = 0;
  std::size_t nextSupplies = 0;
  const bool aggregateLimitExceeded =
      !checkedAdd(actionCount_, descriptor.actions.size(), nextActions) ||
      !checkedAdd(eventUseCount_, descriptor.eventUses.size(), nextUses) ||
      !checkedAdd(supplyCount_, descriptor.supplies.size(), nextSupplies) ||
      nextActions > limits_.maximumTotalActions ||
      nextUses > limits_.maximumTotalEventUses ||
      nextSupplies > limits_.maximumTotalSupplies;
  if (aggregateLimitExceeded) {
    return {CanonicalSyncProblemError::LimitExceeded, mechanisms_.size()};
  }
  const CanonicalSyncMechanismId id = mechanisms_.size();
  mechanisms_.push_back(
      {id, std::move(descriptor), std::move(lifetimes), std::move(cost), {}});
  const CanonicalSyncResourceAllocation resources =
      allocateCanonicalSyncEvents(*this, {id});
  if (!resources.valid || !resources.feasible) {
    mechanisms_.pop_back();
    return {CanonicalSyncProblemError::InvalidMechanism, id};
  }
  mechanismBuckets_[hash].push_back(id);
  actionCount_ = nextActions;
  eventUseCount_ = nextUses;
  supplyCount_ = nextSupplies;
  return {CanonicalSyncProblemError::None, id};
}

CanonicalSyncProblemResult
CanonicalSyncPatternProblem::addConflict(CanonicalSyncMechanismId first,
                                         CanonicalSyncMechanismId second) {
  if (frozen_) {
    return {CanonicalSyncProblemError::Frozen, std::nullopt};
  }
  const bool invalidConflict = first >= mechanisms_.size() ||
                               second >= mechanisms_.size() || first == second;
  if (invalidConflict) {
    return {CanonicalSyncProblemError::InvalidMechanism, std::nullopt};
  }
  auto add = [&](CanonicalSyncMechanismId owner,
                 CanonicalSyncMechanismId conflict) {
    auto &conflicts = mechanisms_[owner].conflicts;
    const auto position =
        std::lower_bound(conflicts.begin(), conflicts.end(), conflict);
    const bool missing = position == conflicts.end() || *position != conflict;
    if (missing) {
      conflicts.insert(position, conflict);
    }
  };
  add(first, second);
  add(second, first);
  return {CanonicalSyncProblemError::None, first};
}

CanonicalSyncProblemResult
CanonicalSyncPatternProblem::addPattern(CanonicalSyncPatternSpec pattern) {
  if (frozen_) {
    return {CanonicalSyncProblemError::Frozen, patternSpecs_.size()};
  }
  if (pattern.kind == CanonicalSyncPatternKind::Singleton) {
    return {CanonicalSyncProblemError::InvalidPattern, patternSpecs_.size()};
  }
  std::sort(pattern.members.begin(), pattern.members.end());
  pattern.members.erase(
      std::unique(pattern.members.begin(), pattern.members.end()),
      pattern.members.end());
  const bool invalid =
      pattern.members.size() < 2 ||
      pattern.members.size() > limits_.maximumMembersPerPattern ||
      std::any_of(pattern.members.begin(), pattern.members.end(),
                  [&](CanonicalSyncMechanismId member) {
                    return member >= mechanisms_.size();
                  });
  if (invalid) {
    return {CanonicalSyncProblemError::InvalidPattern, patternSpecs_.size()};
  }
  for (std::size_t index = 0; index < pattern.members.size(); ++index) {
    const auto &conflicts = mechanisms_[pattern.members[index]].conflicts;
    for (std::size_t next = index + 1; next < pattern.members.size(); ++next) {
      if (std::binary_search(conflicts.begin(), conflicts.end(),
                             pattern.members[next])) {
        return {CanonicalSyncProblemError::InvalidPattern,
                patternSpecs_.size()};
      }
    }
  }
  auto existing = std::find_if(patternSpecs_.begin(), patternSpecs_.end(),
                               [&](const auto &candidate) {
                                 return candidate.members == pattern.members;
                               });
  if (existing != patternSpecs_.end()) {
    const bool strongerProvenance =
        patternPriority(pattern.kind) < patternPriority(existing->kind);
    if (strongerProvenance) {
      existing->kind = pattern.kind;
    }
    return {CanonicalSyncProblemError::None,
            static_cast<std::size_t>(existing - patternSpecs_.begin())};
  }
  const bool patternLimitReached =
      mechanisms_.size() >= limits_.maximumPatterns ||
      patternSpecs_.size() >= limits_.maximumPatterns - mechanisms_.size();
  if (patternLimitReached) {
    return {CanonicalSyncProblemError::LimitExceeded, patternSpecs_.size()};
  }
  patternSpecs_.push_back(std::move(pattern));
  return {CanonicalSyncProblemError::None, patternSpecs_.size() - 1};
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::buildPatterns() {
  std::vector<SyncCoverCompletionSupply> allSupplies;
  for (const CanonicalSyncMechanism &mechanism : mechanisms_) {
    for (const CanonicalSyncSupplyBinding &binding :
         mechanism.descriptor.supplies) {
      allSupplies.push_back(
          {mechanism.id, binding.edge, binding.allowedDemands});
    }
  }
  const SyncCoverSingletonCoverageResult singletonCoverage =
      computeSyncCoverSingletonCoverage(graph_, expansion_, mechanisms_.size(),
                                        allSupplies, activeDemands_);
  if (!singletonCoverage) {
    return {CanonicalSyncProblemError::CoverageFailure, std::nullopt};
  }
  baselineCoverage_ =
      projectCoverage(singletonCoverage.baseline, activeDemands_);

  patternStatistics_ = {};
  patterns_.clear();
  patterns_.reserve(mechanisms_.size() + patternSpecs_.size());
  for (const CanonicalSyncMechanism &mechanism : mechanisms_) {
    SyncCoverDemandSet coverage =
        projectCoverage(singletonCoverage.mechanisms[mechanism.id],
                        activeDemands_);
    coverage.subtract(baselineCoverage_);
    CanonicalSyncPatternKindStatistics &statistics =
        patternStatistics_.kinds[static_cast<std::size_t>(
            CanonicalSyncPatternKind::Singleton)];
    ++statistics.patterns;
    statistics.jointCoverageIncidences += coverage.count();
    statistics.singletonCoverageIncidences += coverage.count();
    patterns_.push_back(
        {patterns_.size(),
         CanonicalSyncPatternKind::Singleton,
         {mechanism.id},
         std::move(coverage),
         0});
  }
  for (const CanonicalSyncPatternSpec &spec : patternSpecs_) {
    const CanonicalSyncResourceAllocation resources =
        allocateCanonicalSyncEvents(*this, spec.members);
    if (!resources.valid || !resources.feasible) {
      return {CanonicalSyncProblemError::InvalidPattern, patterns_.size()};
    }
    const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
        graph_, expansion_, getSupplies(mechanisms_, spec.members),
        activeDemands_);
    if (!coverage) {
      return {CanonicalSyncProblemError::CoverageFailure, patterns_.size()};
    }
    SyncCoverDemandSet jointCoverage =
        projectCoverage(coverage.covered, activeDemands_);
    jointCoverage.subtract(baselineCoverage_);
    SyncCoverDemandSet singletonCoverage(activeDemands_.size());
    for (CanonicalSyncMechanismId member : spec.members) {
      const bool validSingleton =
          member < patterns_.size() &&
          patterns_[member].kind == CanonicalSyncPatternKind::Singleton &&
          patterns_[member].members.size() == 1 &&
          patterns_[member].members.front() == member;
      if (!validSingleton) {
        return {CanonicalSyncProblemError::InvalidPattern, patterns_.size()};
      }
      singletonCoverage.unite(patterns_[member].coverage);
    }
    SyncCoverDemandSet extraCoverage = jointCoverage;
    extraCoverage.subtract(singletonCoverage);
    CanonicalSyncPatternKindStatistics &statistics =
        patternStatistics_.kinds[static_cast<std::size_t>(spec.kind)];
    ++statistics.patterns;
    statistics.jointCoverageIncidences += jointCoverage.count();
    statistics.singletonCoverageIncidences += singletonCoverage.count();
    statistics.extraCoverageIncidences += extraCoverage.count();
    if (!extraCoverage.empty()) {
      ++statistics.patternsWithExtraCoverage;
    }
    patterns_.push_back({patterns_.size(), spec.kind, spec.members,
                         std::move(jointCoverage),
                         extraCoverage.count()});
  }
  return {};
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::freeze() {
  if (frozen_) {
    return {CanonicalSyncProblemError::Frozen, std::nullopt};
  }
  if (!graphValid_) {
    return {CanonicalSyncProblemError::InvalidGraph, std::nullopt};
  }
  CanonicalSyncProblemResult built = buildPatterns();
  if (!built) {
    return built;
  }
  demandPatterns_.assign(activeDemands_.size(), {});
  mechanismPatterns_.assign(mechanisms_.size(), {});
  for (const CanonicalSyncPattern &pattern : patterns_) {
    for (CanonicalSyncMechanismId member : pattern.members) {
      mechanismPatterns_[member].push_back(pattern.id);
    }
    const auto &words = pattern.coverage.getWords();
    for (std::size_t wordIndex = 0; wordIndex < words.size(); ++wordIndex) {
      std::uint64_t word = words[wordIndex];
      while (word != 0) {
        const unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
        const std::size_t demand = wordIndex * 64 + bit;
        if (demand < demandPatterns_.size()) {
          if (incidenceCount_ >= limits_.maximumIncidences) {
            return {CanonicalSyncProblemError::LimitExceeded,
                    incidenceCount_ + 1};
          }
          demandPatterns_[demand].push_back(pattern.id);
          ++incidenceCount_;
        }
        word &= word - 1;
      }
    }
  }
  std::optional<std::size_t> missing;
  for (std::size_t demand = 0; demand < demandPatterns_.size(); ++demand) {
    const bool lacksCover =
        demandPatterns_[demand].empty() && !baselineCoverage_.contains(demand);
    if (lacksCover) {
      missing = demand;
      break;
    }
  }
  if (missing) {
    return {CanonicalSyncProblemError::UncoverableDemand, *missing};
  }
  mechanismBuckets_.clear();
  patternSpecs_.clear();
  frozen_ = true;
  return {};
}
