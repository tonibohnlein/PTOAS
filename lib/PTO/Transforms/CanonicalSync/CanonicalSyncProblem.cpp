// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir::pto;

namespace {

constexpr std::uint64_t kHashOffset = 1469598103934665603ULL;
constexpr std::uint64_t kHashPrime = 1099511628211ULL;

bool checkedAddCost(std::uint64_t first, std::uint64_t second,
                    std::uint64_t &result) {
  if (second > std::numeric_limits<std::uint64_t>::max() - first) {
    return false;
  }
  result = first + second;
  return true;
}

bool checkedMultiplyCost(std::uint64_t first, std::uint64_t second,
                         std::uint64_t &result) {
  if (first != 0 &&
      second > std::numeric_limits<std::uint64_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

bool consumeSerializationWork(SyncCoverCoverageWorkBudget *workBudget,
                              std::size_t amount = 1) {
  return !workBudget || workBudget->consume(amount);
}

CanonicalSyncProblemResult
computeSerializationBreadth(const SyncCoverGraph &graph,
                            const CanonicalSyncMechanismDescriptor &descriptor,
                            SyncCoverCoverageWorkBudget *workBudget,
                            std::uint64_t &result) {
  result = 0;
  bool sourceDrain = false;
  for (const CanonicalSyncSupplyBinding &binding : descriptor.supplies) {
    if (!consumeSerializationWork(workBudget)) {
      return {CanonicalSyncProblemError::LimitExceeded, std::nullopt};
    }
    const SyncCoverEdge &edge = binding.edge;
    if (edge.source >= graph.getNodes().size() ||
        edge.target >= graph.getNodes().size()) {
      return {CanonicalSyncProblemError::InvalidMechanism, std::nullopt};
    }
    sourceDrain |=
        binding.proof == CanonicalSyncSupplyProof::SourceLocalPipeDrainAction ||
        binding.proof == CanonicalSyncSupplyProof::SourcePrefixPipeDrainAction;
    const SyncCoverNode &source = graph.getNodes()[edge.source];
    const SyncCoverNode &target = graph.getNodes()[edge.target];
    const std::uint64_t span =
        source.order > target.order
            ? static_cast<std::uint64_t>(source.order - target.order) + 1
            : static_cast<std::uint64_t>(target.order - source.order) + 1;
    std::uint64_t endpointWeight = 0;
    std::uint64_t contribution = 0;
    std::uint64_t nextResult = 0;
    if (!checkedAddCost(source.weight, target.weight, endpointWeight) ||
        !checkedMultiplyCost(span, endpointWeight, contribution) ||
        !checkedAddCost(result, contribution, nextResult)) {
      return {CanonicalSyncProblemError::ArithmeticOverflow, std::nullopt};
    }
    result = nextResult;
  }
  if (!sourceDrain || descriptor.actions.size() != 1) {
    return {};
  }
  if (!consumeSerializationWork(workBudget)) {
    return {CanonicalSyncProblemError::LimitExceeded, std::nullopt};
  }
  const CanonicalSyncAction &action = descriptor.actions.front();
  if (action.anchor.kind != SyncCoverAnchorKind::AfterNode ||
      action.anchor.node >= graph.getNodes().size()) {
    return {CanonicalSyncProblemError::InvalidMechanism, std::nullopt};
  }
  const SyncCoverNode &anchor = graph.getNodes()[action.anchor.node];
  std::optional<SyncCoverScopeId> enclosingScope = anchor.scope;
  for (const CanonicalSyncSupplyBinding &binding : descriptor.supplies) {
    if (!consumeSerializationWork(workBudget)) {
      return {CanonicalSyncProblemError::LimitExceeded, std::nullopt};
    }
    const bool bindingUsesSourceDrain =
        binding.proof == CanonicalSyncSupplyProof::SourceLocalPipeDrainAction ||
        binding.proof == CanonicalSyncSupplyProof::SourcePrefixPipeDrainAction;
    if (!bindingUsesSourceDrain ||
        binding.edge.target >= graph.getNodes().size()) {
      continue;
    }
    // getLowestCommonScope may walk both parent chains. Charge a full scope
    // chain for each side before entering the helper.
    if (!consumeSerializationWork(workBudget, graph.getScopes().size()) ||
        !consumeSerializationWork(workBudget) ||
        !consumeSerializationWork(workBudget, graph.getScopes().size()) ||
        !consumeSerializationWork(workBudget)) {
      return {CanonicalSyncProblemError::LimitExceeded, std::nullopt};
    }
    enclosingScope =
        enclosingScope
            ? graph.getLowestCommonScope(
                  *enclosingScope, graph.getNodes()[binding.edge.target].scope)
            : std::nullopt;
    if (!enclosingScope) {
      return {CanonicalSyncProblemError::InvalidMechanism, std::nullopt};
    }
  }
  // The owning-timeline query walks at most one complete parent chain.
  if (!consumeSerializationWork(workBudget, graph.getScopes().size()) ||
      !consumeSerializationWork(workBudget)) {
    return {CanonicalSyncProblemError::LimitExceeded, std::nullopt};
  }
  const std::optional<SyncCoverScopeId> timelineScope =
      graph.getOwningTimelineScope(*enclosingScope);
  if (!timelineScope) {
    return {CanonicalSyncProblemError::InvalidMechanism, std::nullopt};
  }
  std::uint64_t prefixWeight = 0;
  std::uint64_t suffixWeight = 0;
  for (const SyncCoverNode &node : graph.getNodes()) {
    // scopeContains walks at most one complete parent chain; guard
    // compatibility is linear in both normalized literal vectors.
    if (!consumeSerializationWork(workBudget) ||
        !consumeSerializationWork(workBudget, graph.getScopes().size()) ||
        !consumeSerializationWork(workBudget) ||
        !consumeSerializationWork(workBudget, anchor.guard.literals.size()) ||
        !consumeSerializationWork(workBudget, node.guard.literals.size())) {
      return {CanonicalSyncProblemError::LimitExceeded, std::nullopt};
    }
    if (!graph.scopeContains(*timelineScope, node.scope) ||
        !syncCoverGuardsCompatible(anchor.guard, node.guard)) {
      continue;
    }
    if (node.resource == action.resource && node.order <= anchor.order) {
      std::uint64_t nextPrefix = 0;
      if (!checkedAddCost(prefixWeight, node.weight, nextPrefix)) {
        return {CanonicalSyncProblemError::ArithmeticOverflow, std::nullopt};
      }
      prefixWeight = nextPrefix;
    } else if (node.order > anchor.order) {
      std::uint64_t nextSuffix = 0;
      if (!checkedAddCost(suffixWeight, node.weight, nextSuffix)) {
        return {CanonicalSyncProblemError::ArithmeticOverflow, std::nullopt};
      }
      suffixWeight = nextSuffix;
    }
  }
  std::uint64_t rectangle = 0;
  std::uint64_t nextResult = 0;
  if (!checkedMultiplyCost(prefixWeight, suffixWeight, rectangle) ||
      !checkedAddCost(result, rectangle, nextResult)) {
    return {CanonicalSyncProblemError::ArithmeticOverflow, std::nullopt};
  }
  result = nextResult;
  return {};
}

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
  return std::tie(left.allowedDemands, left.attestedDemand, left.applicability,
                  left.eventUse, left.barrierAction, left.produceAction,
                  left.consumeAction, left.proof, left.completionExport) <
         std::tie(right.allowedDemands, right.attestedDemand,
                  right.applicability, right.eventUse, right.barrierAction,
                  right.produceAction, right.consumeAction, right.proof,
                  right.completionExport);
}

bool bindingEqual(const CanonicalSyncSupplyBinding &left,
                  const CanonicalSyncSupplyBinding &right) {
  return edgeEqual(left.edge, right.edge) &&
         std::tie(left.allowedDemands, left.attestedDemand, left.applicability,
                  left.eventUse, left.barrierAction, left.produceAction,
                  left.consumeAction, left.proof, left.completionExport) ==
             std::tie(right.allowedDemands, right.attestedDemand,
                      right.applicability, right.eventUse, right.barrierAction,
                      right.produceAction, right.consumeAction, right.proof,
                      right.completionExport);
}

bool actionEqual(const CanonicalSyncAction &left,
                 const CanonicalSyncAction &right) {
  return std::tie(left.kind, left.resource, left.anchor.kind, left.anchor.node,
                  left.anchor.scope, left.anchor.position, left.eventUse,
                  left.eventLane, left.drainedResources, left.barrierKind,
                  left.guard, left.guardScope, left.eventLaneKind,
                  left.eventLaneScope) ==
         std::tie(right.kind, right.resource, right.anchor.kind,
                  right.anchor.node, right.anchor.scope, right.anchor.position,
                  right.eventUse, right.eventLane, right.drainedResources,
                  right.barrierKind, right.guard, right.guardScope,
                  right.eventLaneKind, right.eventLaneScope);
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
    hashValue(hash, binding.attestedDemand.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, static_cast<std::uint8_t>(binding.applicability));
    hashValue(hash, binding.eventUse.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, binding.barrierAction.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, binding.produceAction.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, binding.consumeAction.value_or(
                        std::numeric_limits<std::size_t>::max()));
    hashValue(hash, static_cast<std::uint8_t>(binding.proof));
    hashValue(hash, static_cast<std::uint8_t>(binding.completionExport));
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
    hashValue(hash, static_cast<std::uint8_t>(action.eventLaneKind));
    hashValue(hash, action.eventLaneScope.value_or(
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
  case SyncCoverAnchorKind::ControlEntry:
  case SyncCoverAnchorKind::ControlExit: {
    const bool validControl =
        action.anchor.node < graph.getControls().size() &&
        action.anchor.scope == graph.getControls()[action.anchor.node].scope;
    if (validControl) {
      return action.anchor.scope;
    }
    return std::nullopt;
  }
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit:
  case SyncCoverAnchorKind::LoopBodyEntry:
  case SyncCoverAnchorKind::LoopBodyExit:
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

std::vector<SyncCoverDemandId>
getCoveredDemandIds(const SyncCoverDemandSet &coverage) {
  std::vector<SyncCoverDemandId> result;
  result.reserve(coverage.count());
  const std::vector<std::uint64_t> &words = coverage.getWords();
  for (std::size_t wordIndex = 0; wordIndex < words.size(); ++wordIndex) {
    std::uint64_t word = words[wordIndex];
    while (word != 0) {
      const unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
      const SyncCoverDemandId demand = wordIndex * 64 + bit;
      if (demand < coverage.size()) {
        result.push_back(demand);
      }
      word &= word - 1;
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
  case CanonicalSyncPatternKind::DirectPair:
    return 1;
  case CanonicalSyncPatternKind::RepairFrontier:
    return 2;
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
      result.push_back({member, binding.edge, binding.allowedDemands,
                        binding.completionExport ==
                            CanonicalSyncSupplyExport::ScopeExitAfterDrain,
                        binding.applicability});
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
    const bool invalidAttestation =
        binding.attestedDemand &&
        *binding.attestedDemand >= graph.getDemands().size();
    const bool invalidApplicability =
        binding.applicability != SyncCoverSupplyApplicability::AllDemands &&
        binding.applicability != SyncCoverSupplyApplicability::DistanceZeroOnly;
    const bool direct = binding.proof == CanonicalSyncSupplyProof::DirectAction;
    const bool completionFrontier =
        binding.proof == CanonicalSyncSupplyProof::CompletionFrontierAction;
    const bool targetCertificate =
        binding.proof ==
        CanonicalSyncSupplyProof::TargetCompletionCertificateAction;
    const bool sourceLocalCompletion =
        binding.proof == CanonicalSyncSupplyProof::SourceLocalCompletionAction;
    const bool targetPrefix =
        binding.proof == CanonicalSyncSupplyProof::TargetLocalFenceAction;
    const bool targetPipeDrain =
        binding.proof == CanonicalSyncSupplyProof::TargetLocalPipeDrainAction;
    const bool dominatingDrainCut =
        binding.proof == CanonicalSyncSupplyProof::DominatingTargetedDrainCut;
    const bool loopCarryDrain =
        binding.proof == CanonicalSyncSupplyProof::LoopCarryPipeDrain;
    const bool sourceLocalDrain =
        binding.proof == CanonicalSyncSupplyProof::SourceLocalPipeDrainAction;
    const bool sourcePrefixDrain =
        binding.proof == CanonicalSyncSupplyProof::SourcePrefixPipeDrainAction;
    const bool loopBoundaryPrefix =
        binding.proof ==
        CanonicalSyncSupplyProof::LoopBoundarySourcePrefixProtocol;
    const bool basicOwnershipAction =
        binding.proof ==
        CanonicalSyncSupplyProof::VerifiedBasicOwnershipProtocol;
    const bool basicOwnershipComposite =
        binding.proof ==
        CanonicalSyncSupplyProof::VerifiedBasicOwnershipComposite;
    const bool targetLocal = targetPrefix || targetPipeDrain || loopCarryDrain;
    const bool restrictedLocal = targetCertificate || sourceLocalCompletion ||
                                 loopBoundaryPrefix || targetLocal ||
                                 sourceLocalDrain || sourcePrefixDrain;
    const bool verified =
        binding.proof == CanonicalSyncSupplyProof::VerifiedProtocol ||
        loopBoundaryPrefix || basicOwnershipAction || basicOwnershipComposite;
    const bool validBasicOwnershipQualifier =
        (!basicOwnershipAction && !basicOwnershipComposite) ||
        (!binding.allowedDemands.empty() && !binding.attestedDemand &&
         binding.applicability == SyncCoverSupplyApplicability::AllDemands);
    const bool validTargetLocalQualifier =
        !targetLocal ||
        (binding.attestedDemand &&
         ((binding.edge.distance == 0 && binding.allowedDemands.empty() &&
           binding.applicability ==
               SyncCoverSupplyApplicability::DistanceZeroOnly) ||
          (binding.edge.distance != 0 &&
           binding.allowedDemands ==
               std::vector<SyncCoverDemandId>{*binding.attestedDemand} &&
           binding.applicability == SyncCoverSupplyApplicability::AllDemands)));
    const bool validSourceLocalCompletionQualifier =
        !sourceLocalCompletion ||
        (binding.attestedDemand &&
         ((binding.edge.distance == 0 && binding.allowedDemands.empty() &&
           binding.applicability ==
               SyncCoverSupplyApplicability::DistanceZeroOnly) ||
          (binding.edge.distance != 0 &&
           binding.allowedDemands ==
               std::vector<SyncCoverDemandId>{*binding.attestedDemand} &&
           binding.applicability == SyncCoverSupplyApplicability::AllDemands)));
    const bool validTargetCertificateQualifier =
        !targetCertificate ||
        (binding.attestedDemand && binding.edge.distance == 0 &&
         binding.allowedDemands ==
             std::vector<SyncCoverDemandId>{*binding.attestedDemand} &&
         binding.applicability == SyncCoverSupplyApplicability::AllDemands);
    const bool validDrainCutQualifier =
        !dominatingDrainCut ||
        (!binding.attestedDemand && binding.edge.distance == 0 &&
         binding.allowedDemands.empty() &&
         binding.applicability ==
             SyncCoverSupplyApplicability::DistanceZeroOnly);
    const bool validLoopCarryQualifier =
        !loopCarryDrain ||
        (binding.attestedDemand && binding.edge.distance != 0 &&
         binding.allowedDemands ==
             std::vector<SyncCoverDemandId>{*binding.attestedDemand} &&
         binding.applicability == SyncCoverSupplyApplicability::AllDemands);
    const bool validLoopBoundaryPrefixQualifier =
        !loopBoundaryPrefix ||
        (binding.attestedDemand && binding.edge.distance != 0 &&
         binding.allowedDemands ==
             std::vector<SyncCoverDemandId>{*binding.attestedDemand} &&
         binding.applicability == SyncCoverSupplyApplicability::AllDemands);
    const bool validSourceLocalQualifier =
        (!sourceLocalDrain && !sourcePrefixDrain) ||
        (binding.attestedDemand &&
         ((binding.edge.distance == 0 && binding.allowedDemands.empty() &&
           binding.applicability ==
               SyncCoverSupplyApplicability::DistanceZeroOnly) ||
          (binding.edge.distance != 0 &&
           binding.allowedDemands ==
               std::vector<SyncCoverDemandId>{*binding.attestedDemand} &&
           binding.applicability == SyncCoverSupplyApplicability::AllDemands)));
    const bool validOrdinaryQualifier =
        restrictedLocal || dominatingDrainCut ||
        (!binding.attestedDemand &&
         binding.applicability == SyncCoverSupplyApplicability::AllDemands);
    const bool validOwner =
        protocol
            ? basicOwnershipComposite
                  ? verified && !binding.eventUse && !binding.barrierAction &&
                        !binding.produceAction && !binding.consumeAction
                  : verified && binding.eventUse && !binding.barrierAction &&
                        binding.produceAction && binding.consumeAction
            : (direct || completionFrontier || targetCertificate ||
               restrictedLocal || dominatingDrainCut) &&
                  (binding.eventUse.has_value() !=
                   binding.barrierAction.has_value()) &&
                  !binding.produceAction && !binding.consumeAction;
    const bool invalid =
        graph.canonicalizeCompletionEdge(binding.edge) !=
            SyncCoverGraphError::None ||
        invalidDemands || invalidAttestation || invalidApplicability ||
        (!binding.allowedDemands.empty() && !protocol && !restrictedLocal) ||
        !validTargetLocalQualifier || !validDrainCutQualifier ||
        !validTargetCertificateQualifier ||
        !validSourceLocalCompletionQualifier || !validLoopCarryQualifier ||
        !validLoopBoundaryPrefixQualifier || !validSourceLocalQualifier ||
        !validBasicOwnershipQualifier || !validOrdinaryQualifier || !validOwner;
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
    const auto validLoopScope = [&](std::optional<SyncCoverScopeId> scope) {
      return !scope || (*scope < graph.getScopes().size() &&
                        graph.getScopes()[*scope].isLoop &&
                        graph.getScopes()[*scope].timeline);
    };
    const bool invalid =
        use.domain >= domains.size() || use.width == 0 ||
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
                           graph.getScopes()[*action.guardScope].isLoop;
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
    const bool loopBodyEntry =
        action.guardScope &&
        action.anchor.kind == SyncCoverAnchorKind::LoopBodyEntry &&
        action.anchor.scope == *action.guardScope;
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
                    (nodeBoundary || nestedScopeBoundary || loopBodyEntry)));
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
      const bool invalid =
          action.eventUse || action.eventLane != 0 ||
          action.eventLaneKind != CanonicalSyncEventLaneKind::Static ||
          action.eventLaneScope || action.drainedResources.empty() ||
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
    const bool dynamicLane =
        action.eventLaneKind == CanonicalSyncEventLaneKind::LoopIterationModulo;
    const bool validDynamicLane =
        dynamicLane && action.eventLane == 0 && action.eventLaneScope &&
        use.width > 1 && use.recurrenceScope == action.eventLaneScope &&
        *action.eventLaneScope < graph.getScopes().size() &&
        graph.getScopes()[*action.eventLaneScope].isLoop;
    const bool set = action.kind == CanonicalSyncActionKind::EventSet;
    const bool wait = action.kind == CanonicalSyncActionKind::EventWait;
    const bool invalidAction =
        (!set && !wait) ||
        (dynamicLane
             ? !validDynamicLane
             : action.eventLane >= use.width || action.eventLaneScope) ||
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
    std::vector<bool> &lanes =
        (set ? state.setLanes : state.waitLanes)[*action.eventUse];
    if (dynamicLane) {
      std::fill(lanes.begin(), lanes.end(), true);
    } else {
      lanes[action.eventLane] = true;
    }
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
        return barrier && action.kind != CanonicalSyncActionKind::Barrier;
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

bool hasValidatedScopeExitExport(
    const SyncCoverGraph &graph,
    const CanonicalSyncMechanismDescriptor &descriptor,
    const CanonicalSyncSupplyBinding &binding,
    const CanonicalSyncEventUse &use) {
  if (binding.completionExport !=
      CanonicalSyncSupplyExport::ScopeExitAfterDrain) {
    return true;
  }
  const SyncCoverEdge &edge = binding.edge;
  if (binding.proof ==
      CanonicalSyncSupplyProof::LoopBoundarySourcePrefixProtocol) {
    const bool invalidHeader =
        descriptor.kind != CanonicalSyncMechanismKind::Protocol ||
        edge.distance == 0 || edge.scope >= graph.getScopes().size() ||
        !graph.getScopes()[edge.scope].isLoop ||
        !graph.getScopes()[edge.scope].timeline || !binding.eventUse ||
        !binding.produceAction || !binding.consumeAction ||
        use.width != edge.distance || use.recurrenceScope != edge.scope ||
        use.lifetimeScope ||
        !graph.scopeContains(edge.scope, graph.getNodes()[edge.source].scope) ||
        !graph.scopeContains(edge.scope, graph.getNodes()[edge.target].scope);
    if (invalidHeader) {
      return false;
    }
    const std::size_t width = edge.distance;
    const std::size_t consumeAction = width;
    const std::size_t barrierAction = consumeAction + 1;
    const std::size_t produceAction = barrierAction + 1;
    const std::size_t drainBegin = produceAction + 1;
    if (*binding.consumeAction != consumeAction ||
        *binding.produceAction != produceAction ||
        descriptor.actions.size() != 2 * width + 3) {
      return false;
    }
    const std::uint32_t sourceResource = graph.getNodes()[edge.source].resource;
    const std::uint32_t targetResource = graph.getNodes()[edge.target].resource;
    const CanonicalSyncEventLaneKind bodyLaneKind =
        width > 1 ? CanonicalSyncEventLaneKind::LoopIterationModulo
                  : CanonicalSyncEventLaneKind::Static;
    const std::optional<SyncCoverScopeId> bodyLaneScope =
        width > 1 ? std::optional<SyncCoverScopeId>(edge.scope) : std::nullopt;
    const auto eventMatches =
        [&](const CanonicalSyncAction &action, CanonicalSyncActionKind kind,
            std::uint32_t resource, SyncCoverAnchorKind anchorKind,
            std::size_t lane, CanonicalSyncEventLaneKind laneKind,
            std::optional<SyncCoverScopeId> laneScope) {
          return action.kind == kind && action.resource == resource &&
                 action.anchor.kind == anchorKind && action.anchor.node == 0 &&
                 action.anchor.scope == edge.scope &&
                 action.eventUse == binding.eventUse &&
                 action.eventLane == lane && action.eventLaneKind == laneKind &&
                 action.eventLaneScope == laneScope &&
                 action.guard == CanonicalSyncActionGuardKind::None &&
                 !action.guardScope && action.drainedResources.empty();
        };
    const CanonicalSyncAction &barrier = descriptor.actions[barrierAction];
    const bool validBarrier =
        graph.supportsBlockingTargetedBarrier(sourceResource) &&
        barrier.kind == CanonicalSyncActionKind::Barrier &&
        barrier.resource == sourceResource &&
        barrier.anchor.kind == SyncCoverAnchorKind::LoopBodyExit &&
        barrier.anchor.node == 0 && barrier.anchor.scope == edge.scope &&
        !barrier.eventUse && barrier.eventLane == 0 &&
        barrier.drainedResources ==
            std::vector<std::uint32_t>{sourceResource} &&
        barrier.barrierKind == CanonicalSyncBarrierKind::Targeted &&
        barrier.guard == CanonicalSyncActionGuardKind::None &&
        !barrier.guardScope;
    if (!validBarrier ||
        !eventMatches(descriptor.actions[consumeAction],
                      CanonicalSyncActionKind::EventWait, targetResource,
                      SyncCoverAnchorKind::LoopBodyEntry, 0, bodyLaneKind,
                      bodyLaneScope) ||
        !eventMatches(descriptor.actions[produceAction],
                      CanonicalSyncActionKind::EventSet, sourceResource,
                      SyncCoverAnchorKind::LoopBodyExit, 0, bodyLaneKind,
                      bodyLaneScope)) {
      return false;
    }
    for (std::size_t lane = 0; lane < width; ++lane) {
      if (!eventMatches(descriptor.actions[lane],
                        CanonicalSyncActionKind::EventSet, sourceResource,
                        SyncCoverAnchorKind::ScopeEntry, lane,
                        CanonicalSyncEventLaneKind::Static, std::nullopt) ||
          !eventMatches(descriptor.actions[drainBegin + lane],
                        CanonicalSyncActionKind::EventWait, targetResource,
                        SyncCoverAnchorKind::ScopeExit, lane,
                        CanonicalSyncEventLaneKind::Static, std::nullopt)) {
        return false;
      }
    }
    return true;
  }
  const bool invalidContract =
      descriptor.kind != CanonicalSyncMechanismKind::Protocol ||
      binding.proof != CanonicalSyncSupplyProof::VerifiedProtocol ||
      edge.distance == 0 || edge.scope >= graph.getScopes().size() ||
      !graph.getScopes()[edge.scope].isLoop || !binding.eventUse ||
      !binding.produceAction || !binding.consumeAction ||
      use.width != edge.distance || use.recurrenceScope != edge.scope ||
      use.lifetimeScope || !edge.sourceGuard.literals.empty() ||
      !edge.targetGuard.literals.empty();
  if (invalidContract) {
    return false;
  }

  const std::uint32_t sourceResource = graph.getNodes()[edge.source].resource;
  const std::uint32_t targetResource = graph.getNodes()[edge.target].resource;
  const CanonicalSyncEventLaneKind bodyLaneKind =
      edge.distance > 1 ? CanonicalSyncEventLaneKind::LoopIterationModulo
                        : CanonicalSyncEventLaneKind::Static;
  const std::optional<SyncCoverScopeId> bodyLaneScope =
      edge.distance > 1 ? std::optional<SyncCoverScopeId>(edge.scope)
                        : std::nullopt;
  const auto actionMatches =
      [&](const CanonicalSyncAction &action, CanonicalSyncActionKind kind,
          std::uint32_t resource, SyncCoverAnchorKind anchorKind,
          SyncCoverNodeId node, std::size_t lane,
          CanonicalSyncEventLaneKind laneKind,
          std::optional<SyncCoverScopeId> laneScope) {
        return action.kind == kind && action.resource == resource &&
               action.anchor.kind == anchorKind && action.anchor.node == node &&
               action.anchor.scope ==
                   (anchorKind == SyncCoverAnchorKind::ScopeEntry ||
                            anchorKind == SyncCoverAnchorKind::ScopeExit
                        ? edge.scope
                        : 0) &&
               action.eventUse == binding.eventUse &&
               action.eventLane == lane && action.eventLaneKind == laneKind &&
               action.eventLaneScope == laneScope &&
               action.guard == CanonicalSyncActionGuardKind::None &&
               !action.guardScope && action.drainedResources.empty();
      };
  const bool invalidBodyActions =
      *binding.produceAction >= descriptor.actions.size() ||
      *binding.consumeAction >= descriptor.actions.size() ||
      !actionMatches(descriptor.actions[*binding.produceAction],
                     CanonicalSyncActionKind::EventSet, sourceResource,
                     SyncCoverAnchorKind::AfterNode, edge.source, 0,
                     bodyLaneKind, bodyLaneScope) ||
      !actionMatches(descriptor.actions[*binding.consumeAction],
                     CanonicalSyncActionKind::EventWait, targetResource,
                     SyncCoverAnchorKind::BeforeNode, edge.target, 0,
                     bodyLaneKind, bodyLaneScope);
  if (invalidBodyActions) {
    return false;
  }

  std::vector<bool> primed(edge.distance, false);
  std::vector<bool> drained(edge.distance, false);
  std::size_t setCount = 0;
  std::size_t waitCount = 0;
  for (const CanonicalSyncAction &action : descriptor.actions) {
    if (action.eventUse != binding.eventUse) {
      continue;
    }
    setCount += action.kind == CanonicalSyncActionKind::EventSet ? 1 : 0;
    waitCount += action.kind == CanonicalSyncActionKind::EventWait ? 1 : 0;
    for (std::size_t lane = 0; lane < edge.distance; ++lane) {
      if (actionMatches(action, CanonicalSyncActionKind::EventSet,
                        sourceResource, SyncCoverAnchorKind::ScopeEntry, 0,
                        lane, CanonicalSyncEventLaneKind::Static,
                        std::nullopt)) {
        if (primed[lane]) {
          return false;
        }
        primed[lane] = true;
      }
      if (actionMatches(action, CanonicalSyncActionKind::EventWait,
                        targetResource, SyncCoverAnchorKind::ScopeExit, 0, lane,
                        CanonicalSyncEventLaneKind::Static, std::nullopt)) {
        if (drained[lane]) {
          return false;
        }
        drained[lane] = true;
      }
    }
  }
  return setCount == edge.distance + 1 && waitCount == edge.distance + 1 &&
         std::find(primed.begin(), primed.end(), false) == primed.end() &&
         std::find(drained.begin(), drained.end(), false) == drained.end();
}

CanonicalSyncProblemError
validateSupplyBindings(const SyncCoverGraph &graph,
                       const SyncCoverExpandedProgram &expansion,
                       const std::vector<CanonicalSyncEventDomain> &domains,
                       const CanonicalSyncMechanismDescriptor &descriptor,
                       const MechanismValidationState &state) {
  const bool barrier = descriptor.kind == CanonicalSyncMechanismKind::Barrier;
  std::vector<std::size_t> supplyCounts(descriptor.eventUses.size(), 0);
  for (const CanonicalSyncSupplyBinding &binding : descriptor.supplies) {
    const SyncCoverEdge &edge = binding.edge;
    const bool basicOwnershipComposite =
        binding.proof ==
        CanonicalSyncSupplyProof::VerifiedBasicOwnershipComposite;
    const bool namesExactDemand =
        binding.attestedDemand &&
        *binding.attestedDemand < graph.getDemands().size() &&
        edgeEqual(edge,
                  SyncCoverEdge{
                      graph.getDemands()[*binding.attestedDemand].source,
                      graph.getDemands()[*binding.attestedDemand].target,
                      SyncCoverEdgeKind::CompletionSupply,
                      graph.getDemands()[*binding.attestedDemand].scope,
                      graph.getDemands()[*binding.attestedDemand].distance,
                      graph.getDemands()[*binding.attestedDemand].sourceGuard,
                      graph.getDemands()[*binding.attestedDemand].targetGuard});
    if (basicOwnershipComposite) {
      const bool exactRestrictedEdge =
          binding.allowedDemands.size() == 1 &&
          edgeEqual(
              edge,
              SyncCoverEdge{
                  graph.getDemands()[binding.allowedDemands.front()].source,
                  graph.getDemands()[binding.allowedDemands.front()].target,
                  SyncCoverEdgeKind::CompletionSupply,
                  graph.getDemands()[binding.allowedDemands.front()].scope,
                  graph.getDemands()[binding.allowedDemands.front()].distance,
                  graph.getDemands()[binding.allowedDemands.front()]
                      .sourceGuard,
                  graph.getDemands()[binding.allowedDemands.front()]
                      .targetGuard});
      if (!exactRestrictedEdge) {
        return CanonicalSyncProblemError::InvalidMechanism;
      }
      continue;
    }
    const auto targetLocalFencePrecedes = [&] {
      if (!namesExactDemand) {
        return false;
      }
      const SyncCoverDemand &demand =
          graph.getDemands()[*binding.attestedDemand];
      const SyncCoverExpandedArena *arena = expansion.getArena(demand);
      if (!arena || demand.distance > arena->getHorizon()) {
        return false;
      }
      const auto source =
          expansion.projectEndpoint(graph, *arena, demand.source, 0);
      const auto target = expansion.projectEndpoint(
          graph, *arena, demand.target, demand.distance);
      const bool validRecurrence =
          demand.distance == 0 || (demand.scope < graph.getScopes().size() &&
                                   graph.getScopes()[demand.scope].isLoop &&
                                   graph.getScopes()[demand.scope].timeline);
      const bool validSameCopyOrder =
          demand.distance != 0 || graph.getNodes()[demand.source].order <
                                      graph.getNodes()[demand.target].order;
      return source && target && validRecurrence && validSameCopyOrder;
    }();
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
          graph.getNodes()[action.anchor.node].physicalAnchor ==
              graph.getNodes()[edge.target].physicalAnchor;
      const bool targetPipeDrain =
          binding.proof == CanonicalSyncSupplyProof::TargetLocalPipeDrainAction;
      const bool dominatingDrainCut =
          binding.proof == CanonicalSyncSupplyProof::DominatingTargetedDrainCut;
      const bool loopCarryDrain =
          binding.proof == CanonicalSyncSupplyProof::LoopCarryPipeDrain;
      const bool sourceLocalDrain =
          binding.proof == CanonicalSyncSupplyProof::SourceLocalPipeDrainAction;
      const bool sourcePrefixDrain =
          binding.proof ==
          CanonicalSyncSupplyProof::SourcePrefixPipeDrainAction;
      const bool validPipeDrain =
          targetPipeDrain && descriptor.actions.size() == 1 &&
          *binding.barrierAction == 0 && targetLocalFencePrecedes &&
          graph.supportsBlockingTargetedBarrier(sourceResource) &&
          action.resource == sourceResource &&
          action.drainedResources ==
              std::vector<std::uint32_t>{sourceResource} &&
          action.barrierKind == CanonicalSyncBarrierKind::Targeted &&
          action.guard == CanonicalSyncActionGuardKind::None &&
          !action.guardScope;
      const SyncCoverNode &sourceNode = graph.getNodes()[edge.source];
      const SyncCoverNode &targetNode = graph.getNodes()[edge.target];
      const auto afterSourceFencePrecedes = [&] {
        if (!targetLocalFencePrecedes) {
          return false;
        }
        const SyncCoverDemand &demand =
            graph.getDemands()[*binding.attestedDemand];
        const SyncCoverExpandedArena *arena = expansion.getArena(demand);
        if (!arena) {
          return false;
        }
        const auto projectedSourceExit =
            expansion.projectEndpoint(graph, *arena, action.anchor.node, 0);
        const auto projectedTargetEntry = expansion.projectEndpoint(
            graph, *arena, targetNode.physicalAnchor, demand.distance);
        if (!projectedSourceExit || !projectedTargetEntry) {
          return false;
        }
        if (demand.distance != 0) {
          const std::optional<unsigned> sourceCopy =
              arena->getCopyForVirtualNode(*projectedSourceExit);
          const std::optional<unsigned> targetCopy =
              arena->getCopyForVirtualNode(*projectedTargetEntry);
          return sourceCopy && targetCopy && *targetCopy > *sourceCopy &&
                 *targetCopy - *sourceCopy == demand.distance;
        }
        const auto sourcePosition =
            resolveSyncCoverAnchor(graph, action.anchor);
        const auto targetPosition =
            resolveSyncCoverAnchor(graph, {SyncCoverAnchorKind::BeforeNode,
                                           targetNode.physicalAnchor, 0, 0});
        return action.anchor.node < graph.getNodes().size() &&
               targetNode.physicalAnchor < graph.getNodes().size() &&
               graph.getNodes()[action.anchor.node].order <
                   graph.getNodes()[targetNode.physicalAnchor].order &&
               sourcePosition && targetPosition &&
               *sourcePosition < *targetPosition;
      }();
      const auto certifiedPrefix =
          graph.getBlockingTargetedBarrierPrefixes().find(
              {sourceResource, targetNode.physicalAnchor});
      const std::optional<SyncCoverScopeId> cutScope =
          graph.getLowestCommonScope(sourceNode.scope, targetNode.scope);
      const bool validDominatingDrainCut =
          dominatingDrainCut && descriptor.actions.size() == 1 &&
          *binding.barrierAction == 0 && edge.distance == 0 &&
          certifiedPrefix != graph.getBlockingTargetedBarrierPrefixes().end() &&
          std::binary_search(certifiedPrefix->second.begin(),
                             certifiedPrefix->second.end(), edge.source) &&
          cutScope && edge.scope == *cutScope &&
          sourceNode.order < targetNode.order &&
          sourceNode.physicalAnchor != targetNode.physicalAnchor &&
          syncCoverGuardsCompatible(sourceNode.guard, targetNode.guard) &&
          graph.supportsBlockingTargetedBarrier(sourceResource) &&
          action.resource == sourceResource &&
          action.drainedResources ==
              std::vector<std::uint32_t>{sourceResource} &&
          action.barrierKind == CanonicalSyncBarrierKind::Targeted &&
          action.guard == CanonicalSyncActionGuardKind::None &&
          !action.guardScope;
      const bool validLoopCarryDrain =
          loopCarryDrain && namesExactDemand && targetLocalFencePrecedes &&
          descriptor.actions.size() == 1 && *binding.barrierAction == 0 &&
          edge.distance != 0 && edge.scope < graph.getScopes().size() &&
          graph.getScopes()[edge.scope].isLoop &&
          graph.scopeContains(edge.scope, sourceNode.scope) &&
          graph.scopeContains(edge.scope, targetNode.scope) &&
          graph.supportsBlockingTargetedBarrier(sourceResource) &&
          action.resource == sourceResource &&
          action.anchor.kind == SyncCoverAnchorKind::LoopBodyEntry &&
          action.anchor.scope == edge.scope &&
          action.drainedResources ==
              std::vector<std::uint32_t>{sourceResource} &&
          action.barrierKind == CanonicalSyncBarrierKind::Targeted &&
          action.guard == CanonicalSyncActionGuardKind::NotFirstIteration &&
          action.guardScope == std::optional<SyncCoverScopeId>(edge.scope);
      const bool validSourceLocalDrain =
          sourceLocalDrain && namesExactDemand && afterSourceFencePrecedes &&
          descriptor.actions.size() == 1 && *binding.barrierAction == 0 &&
          graph.supportsBlockingTargetedBarrier(sourceResource) &&
          action.resource == sourceResource &&
          action.anchor.kind == SyncCoverAnchorKind::AfterNode &&
          action.anchor.node == sourceNode.physicalExit &&
          action.drainedResources ==
              std::vector<std::uint32_t>{sourceResource} &&
          action.barrierKind == CanonicalSyncBarrierKind::Targeted &&
          action.guard == CanonicalSyncActionGuardKind::None &&
          !action.guardScope;
      const SyncCoverNode &sourcePrefixCut =
          graph.getNodes()[action.anchor.node];
      const auto issuedPrefix = graph.getBlockingTargetedBarrierPrefixes().find(
          {sourceResource, sourcePrefixCut.physicalAnchor});
      const bool sourceIssuedBeforeCut =
          sourceNode.physicalAnchor == sourcePrefixCut.physicalAnchor ||
          (issuedPrefix != graph.getBlockingTargetedBarrierPrefixes().end() &&
           std::binary_search(issuedPrefix->second.begin(),
                              issuedPrefix->second.end(), edge.source));
      const bool validSourcePrefixDrain =
          sourcePrefixDrain && namesExactDemand && afterSourceFencePrecedes &&
          descriptor.actions.size() == 1 && *binding.barrierAction == 0 &&
          graph.supportsBlockingTargetedBarrier(sourceResource) &&
          action.resource == sourceResource &&
          action.anchor.kind == SyncCoverAnchorKind::AfterNode &&
          action.anchor.node == sourcePrefixCut.physicalExit &&
          sourceIssuedBeforeCut && sourceNode.scope == sourcePrefixCut.scope &&
          sourceNode.guard.literals == sourcePrefixCut.guard.literals &&
          action.drainedResources ==
              std::vector<std::uint32_t>{sourceResource} &&
          action.barrierKind == CanonicalSyncBarrierKind::Targeted &&
          action.guard == CanonicalSyncActionGuardKind::None &&
          !action.guardScope;
      const bool validOrdinaryBarrier =
          !targetPipeDrain && !dominatingDrainCut && !loopCarryDrain &&
          !sourceLocalDrain && !sourcePrefixDrain &&
          action.resource == targetResource;
      const bool validAnchor =
          loopCarryDrain || sourceLocalDrain || sourcePrefixDrain || crosses;
      if (!sourceDrained || !validAnchor ||
          (!validPipeDrain && !validDominatingDrainCut &&
           !validLoopCarryDrain && !validSourceLocalDrain &&
           !validSourcePrefixDrain && !validOrdinaryBarrier)) {
        return CanonicalSyncProblemError::InvalidMechanism;
      }
      continue;
    }

    const bool protocol =
        descriptor.kind == CanonicalSyncMechanismKind::Protocol;

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
                              set.eventLane != wait.eventLane ||
                              set.eventLaneKind != wait.eventLaneKind ||
                              set.eventLaneScope != wait.eventLaneScope;
    const bool wrongResources =
        domain.sourceResource != graph.getNodes()[edge.source].resource ||
        domain.targetResource != graph.getNodes()[edge.target].resource;
    if (wrongActions || wrongResources) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
    if (!hasValidatedScopeExitExport(graph, descriptor, binding, use)) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
    const bool sourceCompletesDirectly = syncCoverNodeCanProduceCompletion(
        graph, edge.source, domain.targetResource);
    const bool sourceCompletesPrefix =
        graph.getNodes()[edge.source].completionSignalCoversIssuedPrefix;
    const auto isExactSourceBarrier = [&](std::size_t actionIndex) {
      if (actionIndex >= descriptor.actions.size()) {
        return false;
      }
      const CanonicalSyncAction &action = descriptor.actions[actionIndex];
      return graph.supportsBlockingTargetedBarrier(domain.sourceResource) &&
             action.kind == CanonicalSyncActionKind::Barrier &&
             action.resource == domain.sourceResource &&
             action.anchor.kind == SyncCoverAnchorKind::AfterNode &&
             graph.getNodes()[action.anchor.node].physicalAnchor ==
                 graph.getNodes()[edge.source].physicalAnchor &&
             !action.eventUse && action.eventLane == 0 &&
             action.drainedResources ==
                 std::vector<std::uint32_t>{domain.sourceResource} &&
             action.barrierKind == CanonicalSyncBarrierKind::Targeted &&
             action.guard == CanonicalSyncActionGuardKind::None &&
             !action.guardScope;
    };
    const auto isTargetPrefixBarrier = [&](std::size_t actionIndex) {
      if (actionIndex >= descriptor.actions.size()) {
        return false;
      }
      const CanonicalSyncAction &action = descriptor.actions[actionIndex];
      return action.kind == CanonicalSyncActionKind::Barrier &&
             action.resource == domain.sourceResource &&
             action.anchor.kind == SyncCoverAnchorKind::BeforeNode &&
             graph.getNodes()[action.anchor.node].physicalAnchor ==
                 graph.getNodes()[edge.target].physicalAnchor &&
             !action.eventUse && action.eventLane == 0 &&
             action.drainedResources ==
                 std::vector<std::uint32_t>{domain.sourceResource} &&
             action.barrierKind == CanonicalSyncBarrierKind::Targeted &&
             action.guard == CanonicalSyncActionGuardKind::None &&
             !action.guardScope;
    };
    const bool correctTargetPrefixBarrier =
        sourceCompletesPrefix
            ? descriptor.actions.size() == 2
            : produce == 1 && descriptor.actions.size() == 3 &&
                  isTargetPrefixBarrier(0);
    const bool correctExactSourceCompletion =
        sourceCompletesDirectly ||
        (produce != 0 && isExactSourceBarrier(produce - 1));
    if (!protocol) {
      const bool direct =
          binding.proof == CanonicalSyncSupplyProof::DirectAction;
      const bool completionFrontier =
          binding.proof == CanonicalSyncSupplyProof::CompletionFrontierAction;
      const bool targetCertificate =
          binding.proof ==
          CanonicalSyncSupplyProof::TargetCompletionCertificateAction;
      const bool sourceLocalCompletion =
          binding.proof ==
          CanonicalSyncSupplyProof::SourceLocalCompletionAction;
      const bool targetPrefix =
          binding.proof == CanonicalSyncSupplyProof::TargetLocalFenceAction;
      const bool validTargetCertificate =
          targetCertificate && namesExactDemand && produce == 0 &&
          consume == 1 && descriptor.actions.size() == 2 &&
          set.anchor.kind == SyncCoverAnchorKind::AfterNode &&
          wait.anchor.kind == SyncCoverAnchorKind::BeforeNode &&
          (graph.hasTargetCompletionCertificate(
               SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix,
               set.anchor.node, wait.anchor.node, domain.sourceResource,
               domain.targetResource, *binding.attestedDemand) ||
           graph.hasTargetCompletionCertificate(
               SyncCoverTargetCompletionKind::MToFixAccumulatorBoundary,
               set.anchor.node, wait.anchor.node, domain.sourceResource,
               domain.targetResource, *binding.attestedDemand));
      const bool wrongDirectAnchors =
          set.anchor.kind != SyncCoverAnchorKind::AfterNode ||
          set.anchor.node != edge.source ||
          wait.anchor.kind != SyncCoverAnchorKind::BeforeNode ||
          wait.anchor.node != edge.target;
      const bool wrongTargetPrefixAnchors =
          set.anchor.kind != SyncCoverAnchorKind::BeforeNode ||
          graph.getNodes()[set.anchor.node].physicalAnchor !=
              graph.getNodes()[edge.target].physicalAnchor ||
          wait.anchor.kind != SyncCoverAnchorKind::BeforeNode ||
          graph.getNodes()[wait.anchor.node].physicalAnchor !=
              graph.getNodes()[edge.target].physicalAnchor;
      const bool wrongSourceLocalAnchors =
          set.anchor.kind != SyncCoverAnchorKind::AfterNode ||
          wait.anchor.kind != SyncCoverAnchorKind::AfterNode ||
          set.anchor.node != graph.getNodes()[edge.source].physicalExit ||
          wait.anchor.node != set.anchor.node;
      const bool correctSourceLocalCompletion =
          sourceCompletesDirectly
              ? produce == 0 && consume == 1 && descriptor.actions.size() == 2
              : produce == 1 && consume == 2 &&
                    descriptor.actions.size() == 3 && isExactSourceBarrier(0);
      const auto sourceLocalCompletionPrecedes = [&] {
        if (!sourceLocalCompletion || !binding.attestedDemand ||
            !targetLocalFencePrecedes || !correctSourceLocalCompletion) {
          return false;
        }
        const SyncCoverDemand &demand =
            graph.getDemands()[*binding.attestedDemand];
        const SyncCoverExpandedArena *arena = expansion.getArena(demand);
        if (!arena || demand.distance > arena->getHorizon()) {
          return false;
        }
        const auto sourceFence =
            expansion.projectEndpoint(graph, *arena, set.anchor.node, 0);
        const auto targetEntry = expansion.projectEndpoint(
            graph, *arena, graph.getNodes()[edge.target].physicalAnchor,
            demand.distance);
        if (!sourceFence || !targetEntry) {
          return false;
        }
        if (demand.distance != 0) {
          const std::optional<unsigned> sourceCopy =
              arena->getCopyForVirtualNode(*sourceFence);
          const std::optional<unsigned> targetCopy =
              arena->getCopyForVirtualNode(*targetEntry);
          return sourceCopy && targetCopy && *targetCopy > *sourceCopy &&
                 *targetCopy - *sourceCopy == demand.distance;
        }
        const auto fencePosition = resolveSyncCoverAnchor(graph, set.anchor);
        const auto targetPosition = resolveSyncCoverAnchor(
            graph, {SyncCoverAnchorKind::BeforeNode,
                    graph.getNodes()[edge.target].physicalAnchor, 0, 0});
        return fencePosition && targetPosition &&
               *fencePosition < *targetPosition;
      }();
      const bool invalidSourceLocalCompletion =
          sourceLocalCompletion &&
          (wrongSourceLocalAnchors || !correctSourceLocalCompletion ||
           !sourceLocalCompletionPrecedes);
      const bool invalidDirect =
          (direct || completionFrontier) &&
          (wrongDirectAnchors || !correctExactSourceCompletion ||
           !syncCoverEndpointsCoExecute(graph, edge) ||
           (completionFrontier &&
            graph.getNodes()[edge.source].completionDominatedSources.empty()));
      const bool invalidTargetPrefix =
          targetPrefix &&
          (wrongTargetPrefixAnchors || !correctTargetPrefixBarrier ||
           !targetLocalFencePrecedes);
      const bool invalidTargetCertificate =
          targetCertificate && !validTargetCertificate;
      const bool invalidDistance =
          (direct || completionFrontier || targetCertificate) &&
          edge.distance != 0;
      if (invalidDistance || use.recurrenceScope || use.lifetimeScope ||
          (!direct && !completionFrontier && !targetCertificate &&
           !sourceLocalCompletion && !targetPrefix) ||
          invalidDirect || invalidSourceLocalCompletion ||
          invalidTargetPrefix || invalidTargetCertificate) {
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
    if (protocol &&
        binding.proof !=
            CanonicalSyncSupplyProof::LoopBoundarySourcePrefixProtocol &&
        binding.proof !=
            CanonicalSyncSupplyProof::VerifiedBasicOwnershipProtocol &&
        !sourceCompletesDirectly &&
        (produce == 0 || !isExactSourceBarrier(produce - 1))) {
      return CanonicalSyncProblemError::InvalidMechanism;
    }
  }
  bool invalidSupplyCount = false;
  for (std::size_t use = 0; use < supplyCounts.size(); ++use) {
    const bool groupedExactEvent =
        descriptor.kind == CanonicalSyncMechanismKind::Event &&
        supplyCounts[use] != 0 &&
        std::all_of(
            descriptor.supplies.begin(), descriptor.supplies.end(),
            [&](const CanonicalSyncSupplyBinding &binding) {
              return binding.eventUse != use ||
                     binding.proof == CanonicalSyncSupplyProof::
                                          TargetCompletionCertificateAction ||
                     binding.proof ==
                         CanonicalSyncSupplyProof::TargetLocalFenceAction ||
                     binding.proof ==
                         CanonicalSyncSupplyProof::SourceLocalCompletionAction;
            });
    invalidSupplyCount =
        invalidSupplyCount || supplyCounts[use] == 0 ||
        (descriptor.kind == CanonicalSyncMechanismKind::Event &&
         supplyCounts[use] != 1 && !groupedExactEvent);
  }
  invalidSupplyCount = invalidSupplyCount || (barrier && !supplyCounts.empty());
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
    : CanonicalSyncPatternProblem(graph, activeDemands, activeDemands,
                                  Limits{}) {}

CanonicalSyncPatternProblem::CanonicalSyncPatternProblem(
    const SyncCoverGraph &graph, std::vector<SyncCoverDemandId> activeDemands,
    Limits limits, SyncCoverExpansionLimits expansionLimits)
    : CanonicalSyncPatternProblem(graph, activeDemands, activeDemands, limits,
                                  expansionLimits) {}

CanonicalSyncPatternProblem::CanonicalSyncPatternProblem(
    const SyncCoverGraph &graph,
    std::vector<SyncCoverDemandId> obligationDemands,
    std::vector<SyncCoverDemandId> selectionDemands, Limits limits,
    SyncCoverExpansionLimits expansionLimits, bool basisReductionTruncated)
    : graph_(graph), expansion_(std::make_shared<SyncCoverExpandedProgram>(
                         graph, obligationDemands, expansionLimits)),
      limits_(limits), issueResources_(getIssueResources(graph)),
      obligationDemands_(std::move(obligationDemands)),
      activeDemands_(std::move(selectionDemands)),
      basisReductionTruncated_(basisReductionTruncated) {
  const bool obligationsNormalized =
      std::is_sorted(obligationDemands_.begin(), obligationDemands_.end()) &&
      std::adjacent_find(obligationDemands_.begin(),
                         obligationDemands_.end()) == obligationDemands_.end();
  const bool normalized =
      std::is_sorted(activeDemands_.begin(), activeDemands_.end()) &&
      std::adjacent_find(activeDemands_.begin(), activeDemands_.end()) ==
          activeDemands_.end();
  const bool inRange = std::all_of(activeDemands_.begin(), activeDemands_.end(),
                                   [&](SyncCoverDemandId demand) {
                                     return demand < graph_.getDemands().size();
                                   });
  const bool obligationsInRange =
      std::all_of(obligationDemands_.begin(), obligationDemands_.end(),
                  [&](SyncCoverDemandId demand) {
                    return demand < graph_.getDemands().size();
                  });
  const bool selectionIsSubset =
      std::includes(obligationDemands_.begin(), obligationDemands_.end(),
                    activeDemands_.begin(), activeDemands_.end());
  graphValid_ = graph_.isStructureFrozen() && graph_.validate() &&
                expansion_->isForGraph(graph_) && obligationsNormalized &&
                normalized && obligationsInRange && inRange &&
                selectionIsSubset;
}

CanonicalSyncPatternProblem::CanonicalSyncPatternProblem(
    const CanonicalSyncPatternProblem &preciseProblem,
    SyncCoverCoverageWorkBudget *constructionWorkBudget)
    : graph_(preciseProblem.graph_), expansion_(preciseProblem.expansion_),
      limits_(preciseProblem.limits_),
      issueResources_(preciseProblem.issueResources_), graphValid_(true),
      incidenceCount_(preciseProblem.incidenceCount_),
      actionCount_(preciseProblem.actionCount_),
      eventUseCount_(preciseProblem.eventUseCount_),
      supplyCount_(preciseProblem.supplyCount_),
      obligationDemands_(preciseProblem.obligationDemands_),
      activeDemands_(preciseProblem.activeDemands_),
      basisReductionTruncated_(preciseProblem.basisReductionTruncated_),
      domains_(preciseProblem.domains_),
      mechanisms_(preciseProblem.mechanisms_),
      protocolVerifiers_(preciseProblem.protocolVerifiers_),
      patternGenerationTruncated_(preciseProblem.patternGenerationTruncated_),
      patterns_(preciseProblem.patterns_),
      patternStatistics_(preciseProblem.patternStatistics_),
      baselineCoverage_(preciseProblem.baselineCoverage_),
      repairEventSeeds_(preciseProblem.repairEventSeeds_),
      candidateConfigurationSignature_(
          preciseProblem.candidateConfigurationSignature_),
      frozenPrefixMechanismCount_(preciseProblem.mechanisms_.size()),
      constructionWorkBudget_(constructionWorkBudget) {
  constructionBaselineCoverage_ = baselineCoverage_;
  constructionSingletonCoverage_.resize(mechanisms_.size());
  for (const CanonicalSyncPattern &pattern : patterns_) {
    if (pattern.kind == CanonicalSyncPatternKind::Singleton) {
      continue;
    }
    ++retainedPatternCount_;
    coverageWordCount_ += pattern.coverage.getWords().size();
    pendingCoverageIncidenceCount_ += pattern.extraCoverageCount;
  }
  for (const CanonicalSyncMechanism &mechanism : mechanisms_) {
    mechanismBuckets_[descriptorHash(mechanism.descriptor)].push_back(
        mechanism.id);
  }
}

std::unique_ptr<CanonicalSyncPatternProblem>
CanonicalSyncPatternProblem::cloneMutableRepairPrefix(
    SyncCoverCoverageWorkBudget *workBudget) const {
  if (!frozen_ || !graphValid_) {
    return nullptr;
  }
  const auto consume = [&](std::size_t amount) {
    return !workBudget || workBudget->consume(amount);
  };
  if (!consume(issueResources_.size()) || !consume(obligationDemands_.size()) ||
      !consume(activeDemands_.size()) || !consume(domains_.size()) ||
      !consume(mechanisms_.size()) || !consume(patterns_.size()) ||
      !consume(repairEventSeeds_.size()) ||
      !consume(baselineCoverage_.getWords().size())) {
    return nullptr;
  }
  for (const CanonicalSyncEventDomain &domain : domains_) {
    if (!consume(domain.reservedIds.size())) {
      return nullptr;
    }
  }
  for (const CanonicalSyncMechanism &mechanism : mechanisms_) {
    const CanonicalSyncMechanismDescriptor &descriptor = mechanism.descriptor;
    if (!consume(descriptor.actions.size()) ||
        !consume(descriptor.eventUses.size()) ||
        !consume(descriptor.supplies.size()) ||
        !consume(mechanism.eventLifetimes.size()) ||
        !consume(mechanism.cost.barrierActions.size()) ||
        !consume(mechanism.cost.eventActions.size()) ||
        !consume(mechanism.conflicts.size())) {
      return nullptr;
    }
    for (const CanonicalSyncAction &action : descriptor.actions) {
      if (!consume(action.drainedResources.size())) {
        return nullptr;
      }
    }
    for (const CanonicalSyncSupplyBinding &supply : descriptor.supplies) {
      if (!consume(supply.allowedDemands.size())) {
        return nullptr;
      }
    }
  }
  for (const CanonicalSyncPattern &pattern : patterns_) {
    if (!consume(pattern.members.size()) ||
        !consume(pattern.coverage.getWords().size())) {
      return nullptr;
    }
  }
  return std::unique_ptr<CanonicalSyncPatternProblem>(
      new CanonicalSyncPatternProblem(*this, workBudget));
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::recordRepairEventSeed(
    CanonicalSyncRepairEventSeed seed) {
  if (frozen_) {
    return {CanonicalSyncProblemError::Frozen, std::nullopt};
  }
  const bool invalid = seed.demand >= graph_.getDemands().size() ||
                       seed.mechanism >= mechanisms_.size() ||
                       seed.domain >= domains_.size();
  if (invalid) {
    return {CanonicalSyncProblemError::InvalidMechanism, std::nullopt};
  }
  const auto position = std::lower_bound(
      repairEventSeeds_.begin(), repairEventSeeds_.end(), seed,
      [](const CanonicalSyncRepairEventSeed &left,
         const CanonicalSyncRepairEventSeed &right) {
        return std::tie(left.mechanism, left.demand, left.domain) <
               std::tie(right.mechanism, right.demand, right.domain);
      });
  const bool duplicate =
      position != repairEventSeeds_.end() &&
      std::tie(position->mechanism, position->demand, position->domain) ==
          std::tie(seed.mechanism, seed.demand, seed.domain);
  if (!duplicate) {
    repairEventSeeds_.insert(position, seed);
  }
  return {};
}

CanonicalSyncProblemResult
CanonicalSyncPatternProblem::setCandidateConfigurationSignature(
    std::uint64_t signature) {
  if (frozen_) {
    return {CanonicalSyncProblemError::Frozen, std::nullopt};
  }
  candidateConfigurationSignature_ = signature;
  return {};
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
  const bool invalid =
      !graphValid_ || domain.id != domains_.size() || duplicate;
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

bool CanonicalSyncPatternProblem::hasSameCandidatePrefix(
    const CanonicalSyncPatternProblem &other) const {
  const bool differentShape =
      &graph_ != &other.graph_ ||
      obligationDemands_ != other.obligationDemands_ ||
      activeDemands_ != other.activeDemands_ ||
      domains_.size() != other.domains_.size() ||
      mechanisms_.size() < other.mechanisms_.size() ||
      repairEventSeeds_.size() != other.repairEventSeeds_.size() ||
      candidateConfigurationSignature_ !=
          other.candidateConfigurationSignature_;
  if (differentShape) {
    return false;
  }
  for (std::size_t index = 0; index < domains_.size(); ++index) {
    const CanonicalSyncEventDomain &left = domains_[index];
    const CanonicalSyncEventDomain &right = other.domains_[index];
    if (std::tie(left.id, left.sourceResource, left.targetResource, left.budget,
                 left.reservedIds) !=
        std::tie(right.id, right.sourceResource, right.targetResource,
                 right.budget, right.reservedIds)) {
      return false;
    }
  }
  for (std::size_t index = 0; index < other.mechanisms_.size(); ++index) {
    if (!descriptorEqual(mechanisms_[index].descriptor,
                         other.mechanisms_[index].descriptor) ||
        mechanisms_[index].conflicts != other.mechanisms_[index].conflicts ||
        mechanisms_[index].originMask != other.mechanisms_[index].originMask) {
      return false;
    }
  }
  for (std::size_t index = 0; index < repairEventSeeds_.size(); ++index) {
    const CanonicalSyncRepairEventSeed &left = repairEventSeeds_[index];
    const CanonicalSyncRepairEventSeed &right = other.repairEventSeeds_[index];
    if (std::tie(left.demand, left.mechanism, left.domain) !=
        std::tie(right.demand, right.mechanism, right.domain)) {
      return false;
    }
  }
  return true;
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::verifyMechanism(
    CanonicalSyncMechanismId mechanism,
    SyncCoverCoverageWorkBudget *workBudget) const {
  if (mechanism >= mechanisms_.size()) {
    return {CanonicalSyncProblemError::InvalidMechanism, mechanism};
  }
  const CanonicalSyncMechanism &stored = mechanisms_[mechanism];
  CanonicalSyncMechanismDescriptor descriptor = stored.descriptor;
  const bool protocol = descriptor.kind == CanonicalSyncMechanismKind::Protocol;
  if (protocol) {
    const bool missingVerifier = mechanism >= protocolVerifiers_.size() ||
                                 !protocolVerifiers_[mechanism];
    if (missingVerifier) {
      return {CanonicalSyncProblemError::UnverifiedProtocol, mechanism};
    }
    SyncCoverCoverageWorkBudget localWork;
    SyncCoverCoverageWorkBudget &protocolWork =
        workBudget ? *workBudget : localWork;
    const CanonicalSyncProblemError verification =
        protocolVerifiers_[mechanism](descriptor, protocolWork);
    if (verification != CanonicalSyncProblemError::None &&
        verification != CanonicalSyncProblemError::UnverifiedProtocol &&
        verification != CanonicalSyncProblemError::LimitExceeded) {
      return {CanonicalSyncProblemError::UnverifiedProtocol, mechanism};
    }
    if (verification != CanonicalSyncProblemError::None) {
      return {verification, mechanism};
    }
  }
  std::vector<CanonicalSyncEventLifetime> lifetimes;
  CanonicalSyncMechanismCost cost;
  const bool protocolVerified = protocol;
  const CanonicalSyncProblemResult validated = validateAndCostMechanism(
      descriptor, lifetimes, cost, protocolVerified, workBudget);
  if (!validated) {
    return {validated.error, mechanism};
  }
  const bool differentLifetimes =
      lifetimes.size() != stored.eventLifetimes.size() ||
      !std::equal(lifetimes.begin(), lifetimes.end(),
                  stored.eventLifetimes.begin(),
                  [](const CanonicalSyncEventLifetime &left,
                     const CanonicalSyncEventLifetime &right) {
                    return left.begin == right.begin && left.end == right.end;
                  });
  const bool differentCost =
      cost.barrierActions != stored.cost.barrierActions ||
      cost.eventActions != stored.cost.eventActions ||
      cost.serializationBreadth != stored.cost.serializationBreadth;
  const bool derivedDataChanged =
      !descriptorEqual(descriptor, stored.descriptor) || differentLifetimes ||
      differentCost;
  if (derivedDataChanged) {
    return {CanonicalSyncProblemError::InvalidMechanism, mechanism};
  }
  return {};
}

std::uint64_t CanonicalSyncPatternProblem::getMechanismSignature(
    CanonicalSyncMechanismId mechanism) const {
  return mechanism < mechanisms_.size()
             ? descriptorHash(mechanisms_[mechanism].descriptor)
             : 0;
}

CanonicalSyncProblemResult
CanonicalSyncPatternProblem::validateAndCostMechanism(
    CanonicalSyncMechanismDescriptor &descriptor,
    std::vector<CanonicalSyncEventLifetime> &lifetimes,
    CanonicalSyncMechanismCost &cost, bool protocolVerified,
    SyncCoverCoverageWorkBudget *workBudget) const {
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

  validation =
      validateSupplyBindings(graph_, *expansion_, domains_, descriptor, state);
  if (validation != CanonicalSyncProblemError::None) {
    return {validation, std::nullopt};
  }
  setRecurrenceLifetimes(graph_, descriptor, state);
  std::uint64_t serializationBreadth = 0;
  const CanonicalSyncProblemResult serialization = computeSerializationBreadth(
      graph_, descriptor, workBudget, serializationBreadth);
  if (!serialization) {
    return serialization;
  }
  state.cost.serializationBreadth = serializationBreadth;
  lifetimes = std::move(state.lifetimes);
  cost = std::move(state.cost);
  return {};
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::internMechanism(
    CanonicalSyncMechanismDescriptor descriptor,
    CanonicalSyncMechanismOrigin origin) {
  return internMechanismImpl(std::move(descriptor), false, {}, origin);
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::internVerifiedProtocol(
    CanonicalSyncMechanismDescriptor descriptor,
    const CanonicalSyncProtocolVerifier &verifier,
    CanonicalSyncMechanismOrigin origin) {
  return internMechanismImpl(std::move(descriptor), true, verifier, origin);
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::internMechanismImpl(
    CanonicalSyncMechanismDescriptor descriptor, bool protocolVerified,
    const CanonicalSyncProtocolVerifier &verifier,
    CanonicalSyncMechanismOrigin origin) {
  if (frozen_) {
    return {CanonicalSyncProblemError::Frozen, mechanisms_.size()};
  }
  if (static_cast<std::size_t>(origin) >= kCanonicalSyncMechanismOriginCount) {
    return {CanonicalSyncProblemError::InvalidMechanism, std::nullopt};
  }
  CanonicalSyncMechanismCost cost;
  std::vector<CanonicalSyncEventLifetime> lifetimes;
  CanonicalSyncProblemResult validated = validateAndCostMechanism(
      descriptor, lifetimes, cost, protocolVerified, constructionWorkBudget_);
  if (!validated) {
    return validated;
  }
  if (protocolVerified) {
    if (!verifier) {
      return {CanonicalSyncProblemError::UnverifiedProtocol, std::nullopt};
    }
    SyncCoverCoverageWorkBudget localWork;
    SyncCoverCoverageWorkBudget &protocolWork =
        constructionWorkBudget_ ? *constructionWorkBudget_ : localWork;
    const CanonicalSyncProblemError verification =
        verifier(descriptor, protocolWork);
    if (verification != CanonicalSyncProblemError::None &&
        verification != CanonicalSyncProblemError::UnverifiedProtocol &&
        verification != CanonicalSyncProblemError::LimitExceeded) {
      return {CanonicalSyncProblemError::UnverifiedProtocol, std::nullopt};
    }
    if (verification != CanonicalSyncProblemError::None) {
      return {verification, std::nullopt};
    }
  }
  const CanonicalSyncMechanismOriginMask originBit =
      canonicalSyncMechanismOriginBit(origin);
  const std::uint64_t hash = descriptorHash(descriptor);
  const auto bucket = mechanismBuckets_.find(hash);
  if (bucket != mechanismBuckets_.end()) {
    for (CanonicalSyncMechanismId mechanism : bucket->second) {
      if (descriptorEqual(mechanisms_[mechanism].descriptor, descriptor)) {
        if (protocolVerified && (mechanism >= protocolVerifiers_.size() ||
                                 !protocolVerifiers_[mechanism])) {
          return {CanonicalSyncProblemError::UnverifiedProtocol, mechanism};
        }
        if (mechanism >= frozenPrefixMechanismCount_) {
          mechanisms_[mechanism].originMask |= originBit;
        }
        return {CanonicalSyncProblemError::None, mechanism};
      }
    }
  }
  const bool mechanismLimitReached =
      mechanisms_.size() >= limits_.maximumMechanisms;
  const bool patternLimitReached =
      mechanisms_.size() >= limits_.maximumPatterns ||
      retainedPatternCount_ >= limits_.maximumPatterns - mechanisms_.size();
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
  mechanisms_.push_back({id,
                         std::move(descriptor),
                         std::move(lifetimes),
                         std::move(cost),
                         {},
                         originBit});
  protocolVerifiers_.push_back(
      protocolVerified ? verifier : CanonicalSyncProtocolVerifier{});
  mechanismBuckets_[hash].push_back(id);
  constructionSingletonCoverage_.push_back(std::nullopt);
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
  auto existing = std::find_if(
      patternSpecs_.begin(), patternSpecs_.end(), [&](const auto &candidate) {
        return candidate.spec.members == pattern.members;
      });
  const bool frozenPrefixDuplicate = std::any_of(
      patterns_.begin(), patterns_.end(), [&](const auto &candidate) {
        return candidate.kind != CanonicalSyncPatternKind::Singleton &&
               candidate.members == pattern.members;
      });
  if (frozenPrefixDuplicate) {
    return {CanonicalSyncProblemError::None, std::nullopt};
  }
  if (existing != patternSpecs_.end()) {
    const bool strongerProvenance =
        patternPriority(pattern.kind) < patternPriority(existing->spec.kind);
    if (strongerProvenance) {
      existing->spec.kind = pattern.kind;
    }
    const std::optional<std::size_t> retained =
        existing->extraCoverageCount == 0
            ? std::nullopt
            : std::optional<std::size_t>(
                  static_cast<std::size_t>(existing - patternSpecs_.begin()));
    return {CanonicalSyncProblemError::None, retained};
  }
  const bool proposalLimitReached =
      patternSpecs_.size() >= limits_.maximumPatternProposals;
  if (proposalLimitReached) {
    patternGenerationTruncated_ = true;
    return {CanonicalSyncProblemError::None, std::nullopt};
  }
  if (!constructionBaselineCoverage_) {
    const SyncCoverCoverageResult baseline = computeSyncCoverCoverage(
        graph_, *expansion_, {}, activeDemands_, constructionWorkBudget_);
    if (!baseline) {
      return {constructionWorkBudget_ && constructionWorkBudget_->exhausted
                  ? CanonicalSyncProblemError::LimitExceeded
                  : CanonicalSyncProblemError::CoverageFailure,
              std::nullopt};
    }
    constructionBaselineCoverage_ =
        projectCoverage(baseline.covered, activeDemands_);
  }
  SyncCoverDemandSet singletonCoverage(activeDemands_.size());
  for (CanonicalSyncMechanismId member : pattern.members) {
    std::optional<SyncCoverDemandSet> &cached =
        constructionSingletonCoverage_[member];
    if (!cached) {
      const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
          graph_, *expansion_, getSupplies(mechanisms_, {member}),
          activeDemands_, constructionWorkBudget_);
      if (!coverage) {
        return {constructionWorkBudget_ && constructionWorkBudget_->exhausted
                    ? CanonicalSyncProblemError::LimitExceeded
                    : CanonicalSyncProblemError::CoverageFailure,
                std::nullopt};
      }
      cached = projectCoverage(coverage.covered, activeDemands_);
      cached->subtract(*constructionBaselineCoverage_);
    }
    singletonCoverage.unite(*cached);
  }
  const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
      graph_, *expansion_, getSupplies(mechanisms_, pattern.members),
      activeDemands_, constructionWorkBudget_);
  if (!coverage) {
    return {constructionWorkBudget_ && constructionWorkBudget_->exhausted
                ? CanonicalSyncProblemError::LimitExceeded
                : CanonicalSyncProblemError::CoverageFailure,
            std::nullopt};
  }
  SyncCoverDemandSet jointCoverage =
      projectCoverage(coverage.covered, activeDemands_);
  jointCoverage.subtract(*constructionBaselineCoverage_);
  SyncCoverDemandSet extraCoverage = jointCoverage;
  extraCoverage.subtract(singletonCoverage);
  const std::size_t jointCoverageCount = jointCoverage.count();
  const std::size_t extraCoverageCount = extraCoverage.count();
  const bool retainsPattern = !extraCoverage.empty();
  const bool retainedLimitReached =
      mechanisms_.size() >= limits_.maximumPatterns ||
      retainedPatternCount_ >= limits_.maximumPatterns - mechanisms_.size();
  if (retainsPattern && retainedLimitReached) {
    patternGenerationTruncated_ = true;
    return {CanonicalSyncProblemError::None, std::nullopt};
  }
  const std::size_t addedCoverageWords =
      retainsPattern ? extraCoverage.getWords().size() : 0;
  const bool coverageLimitReached =
      coverageWordCount_ > limits_.maximumCoverageWords ||
      addedCoverageWords > limits_.maximumCoverageWords - coverageWordCount_;
  const bool coverageIncidenceLimitReached =
      pendingCoverageIncidenceCount_ > limits_.maximumIncidences ||
      extraCoverageCount >
          limits_.maximumIncidences - pendingCoverageIncidenceCount_;
  if (coverageLimitReached || coverageIncidenceLimitReached) {
    patternGenerationTruncated_ = true;
    return {CanonicalSyncProblemError::None, std::nullopt};
  }
  std::vector<SyncCoverDemandId> storedCoverage =
      retainsPattern ? getCoveredDemandIds(extraCoverage)
                     : std::vector<SyncCoverDemandId>{};
  patternSpecs_.push_back({std::move(pattern), std::move(storedCoverage),
                           jointCoverageCount, singletonCoverage.count(),
                           extraCoverageCount});
  if (!retainsPattern) {
    return {CanonicalSyncProblemError::None, std::nullopt};
  }
  ++retainedPatternCount_;
  coverageWordCount_ += addedCoverageWords;
  pendingCoverageIncidenceCount_ += extraCoverageCount;
  return {CanonicalSyncProblemError::None, patternSpecs_.size() - 1};
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::addDirectPairBatch(
    const std::vector<SyncCoverMechanismPair> &pairs,
    const std::vector<SyncCoverDemandSet> &exactJointCoverage,
    const std::vector<SyncCoverDemandSet> &exactSingletonMechanismCoverage) {
  if (frozen_) {
    return {CanonicalSyncProblemError::Frozen, patternSpecs_.size()};
  }
  const bool invalidBatch =
      pairs.size() != exactJointCoverage.size() ||
      exactSingletonMechanismCoverage.size() != mechanisms_.size() ||
      std::any_of(exactJointCoverage.begin(), exactJointCoverage.end(),
                  [&](const SyncCoverDemandSet &coverage) {
                    return coverage.size() != graph_.getDemands().size();
                  }) ||
      std::any_of(exactSingletonMechanismCoverage.begin(),
                  exactSingletonMechanismCoverage.end(),
                  [&](const SyncCoverDemandSet &coverage) {
                    return coverage.size() != graph_.getDemands().size();
                  });
  if (invalidBatch) {
    return {CanonicalSyncProblemError::InvalidPattern, patternSpecs_.size()};
  }
  const bool proposalCapacityExceeded =
      patternSpecs_.size() > limits_.maximumPatternProposals ||
      pairs.size() > limits_.maximumPatternProposals - patternSpecs_.size();
  if (proposalCapacityExceeded) {
    patternGenerationTruncated_ = true;
    return {CanonicalSyncProblemError::None, 0};
  }
  if (!constructionBaselineCoverage_) {
    const SyncCoverCoverageResult baseline = computeSyncCoverCoverage(
        graph_, *expansion_, {}, activeDemands_, constructionWorkBudget_);
    if (!baseline) {
      return {constructionWorkBudget_ && constructionWorkBudget_->exhausted
                  ? CanonicalSyncProblemError::LimitExceeded
                  : CanonicalSyncProblemError::CoverageFailure,
              std::nullopt};
    }
    constructionBaselineCoverage_ =
        projectCoverage(baseline.covered, activeDemands_);
  }

  std::vector<PendingPattern> pending;
  pending.reserve(pairs.size());
  std::size_t singletonCoverageIncidences = 0;
  for (const SyncCoverDemandSet &exactCoverage :
       exactSingletonMechanismCoverage) {
    SyncCoverDemandSet coverage =
        projectCoverage(exactCoverage, activeDemands_);
    coverage.subtract(*constructionBaselineCoverage_);
    const std::size_t incidences = coverage.count();
    if (singletonCoverageIncidences > limits_.maximumIncidences ||
        incidences > limits_.maximumIncidences - singletonCoverageIncidences) {
      singletonCoverageIncidences = limits_.maximumIncidences;
      break;
    }
    singletonCoverageIncidences += incidences;
  }
  std::size_t addedRetainedPatterns = 0;
  std::size_t addedCoverageWords = 0;
  std::size_t addedCoverageIncidences = 0;
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const SyncCoverMechanismPair &pair = pairs[index];
    const bool invalidPair =
        pair.first >= pair.second || pair.second >= mechanisms_.size() ||
        std::binary_search(mechanisms_[pair.first].conflicts.begin(),
                           mechanisms_[pair.first].conflicts.end(),
                           pair.second);
    if (invalidPair) {
      return {CanonicalSyncProblemError::InvalidPattern, patternSpecs_.size()};
    }
    const std::vector<CanonicalSyncMechanismId> members{pair.first,
                                                        pair.second};
    const bool duplicate =
        std::any_of(patternSpecs_.begin(), patternSpecs_.end(),
                    [&](const auto &candidate) {
                      return candidate.spec.members == members;
                    }) ||
        std::any_of(pending.begin(), pending.end(), [&](const auto &candidate) {
          return candidate.spec.members == members;
        });
    if (duplicate) {
      return {CanonicalSyncProblemError::InvalidPattern, patternSpecs_.size()};
    }

    SyncCoverDemandSet jointCoverage =
        projectCoverage(exactJointCoverage[index], activeDemands_);
    SyncCoverDemandSet singletonCoverage = projectCoverage(
        exactSingletonMechanismCoverage[pair.first], activeDemands_);
    singletonCoverage.unite(projectCoverage(
        exactSingletonMechanismCoverage[pair.second], activeDemands_));
    jointCoverage.subtract(*constructionBaselineCoverage_);
    singletonCoverage.subtract(*constructionBaselineCoverage_);
    SyncCoverDemandSet extraCoverage = jointCoverage;
    extraCoverage.subtract(singletonCoverage);
    const std::size_t extraCoverageCount = extraCoverage.count();
    std::vector<SyncCoverDemandId> storedCoverage;
    if (extraCoverageCount != 0) {
      const std::size_t words = extraCoverage.getWords().size();
      const std::size_t availableRetainedPatterns =
          mechanisms_.size() >= limits_.maximumPatterns
              ? 0
              : limits_.maximumPatterns - mechanisms_.size();
      const bool retainedCapacityExceeded =
          retainedPatternCount_ > availableRetainedPatterns ||
          addedRetainedPatterns >=
              availableRetainedPatterns - retainedPatternCount_;
      const bool coverageCapacityExceeded =
          coverageWordCount_ > limits_.maximumCoverageWords ||
          addedCoverageWords >
              limits_.maximumCoverageWords - coverageWordCount_ ||
          words > limits_.maximumCoverageWords - coverageWordCount_ -
                      addedCoverageWords;
      const bool incidenceCapacityExceeded =
          singletonCoverageIncidences > limits_.maximumIncidences ||
          pendingCoverageIncidenceCount_ >
              limits_.maximumIncidences - singletonCoverageIncidences ||
          addedCoverageIncidences > limits_.maximumIncidences -
                                        singletonCoverageIncidences -
                                        pendingCoverageIncidenceCount_ ||
          extraCoverageCount >
              limits_.maximumIncidences - singletonCoverageIncidences -
                  pendingCoverageIncidenceCount_ - addedCoverageIncidences;
      if (retainedCapacityExceeded || coverageCapacityExceeded ||
          incidenceCapacityExceeded) {
        patternGenerationTruncated_ = true;
        return {CanonicalSyncProblemError::None, 0};
      }
      addedCoverageWords += words;
      addedCoverageIncidences += extraCoverageCount;
      ++addedRetainedPatterns;
      storedCoverage = getCoveredDemandIds(extraCoverage);
    }
    pending.push_back({{CanonicalSyncPatternKind::DirectPair, members},
                       std::move(storedCoverage),
                       jointCoverage.count(),
                       singletonCoverage.count(),
                       extraCoverageCount});
  }

  patternSpecs_.insert(patternSpecs_.end(),
                       std::make_move_iterator(pending.begin()),
                       std::make_move_iterator(pending.end()));
  retainedPatternCount_ += addedRetainedPatterns;
  coverageWordCount_ += addedCoverageWords;
  pendingCoverageIncidenceCount_ += addedCoverageIncidences;
  return {CanonicalSyncProblemError::None, addedRetainedPatterns};
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::buildPatterns(
    std::vector<CanonicalSyncPattern> &patterns,
    CanonicalSyncPatternStatistics &patternStatistics,
    SyncCoverDemandSet &baselineCoverage) const {
  std::vector<SyncCoverCompletionSupply> allSupplies;
  for (const CanonicalSyncMechanism &mechanism : mechanisms_) {
    for (const CanonicalSyncSupplyBinding &binding :
         mechanism.descriptor.supplies) {
      allSupplies.push_back({mechanism.id, binding.edge, binding.allowedDemands,
                             binding.completionExport ==
                                 CanonicalSyncSupplyExport::ScopeExitAfterDrain,
                             binding.applicability});
    }
  }
  const SyncCoverSingletonCoverageResult singletonCoverage =
      computeSyncCoverSingletonCoverage(graph_, *expansion_, mechanisms_.size(),
                                        allSupplies, activeDemands_);
  if (!singletonCoverage) {
    return {CanonicalSyncProblemError::CoverageFailure, std::nullopt};
  }
  baselineCoverage =
      projectCoverage(singletonCoverage.baseline, activeDemands_);

  const std::size_t directPairProposals =
      patternStatistics_.directPairProposals;
  const std::size_t directPairEvaluations =
      patternStatistics_.directPairEvaluations;
  const std::size_t directPairConnectorInspections =
      patternStatistics_.directPairConnectorInspections;
  const std::size_t repairFrontierInspections =
      patternStatistics_.repairFrontierInspections;
  const std::size_t repairFrontierProposals =
      patternStatistics_.repairFrontierProposals;
  const bool repairFrontierTruncated =
      patternStatistics_.repairFrontierTruncated;
  const std::size_t sourcePrefixInspections =
      patternStatistics_.sourcePrefixInspections;
  const std::size_t sourcePrefixCandidates =
      patternStatistics_.sourcePrefixCandidates;
  const std::size_t sourcePrefixIncidences =
      patternStatistics_.sourcePrefixIncidences;
  const bool sourcePrefixGenerationTruncated =
      patternStatistics_.sourcePrefixGenerationTruncated;
  const std::size_t loopCarryInspections =
      patternStatistics_.loopCarryInspections;
  const std::size_t loopCarryCandidates =
      patternStatistics_.loopCarryCandidates;
  const std::size_t loopCarryIncidences =
      patternStatistics_.loopCarryIncidences;
  const bool loopCarryGenerationTruncated =
      patternStatistics_.loopCarryGenerationTruncated;
  const std::size_t loopBoundaryProtocolInspections =
      patternStatistics_.loopBoundaryProtocolInspections;
  const std::size_t loopBoundaryProtocolCandidates =
      patternStatistics_.loopBoundaryProtocolCandidates;
  const std::size_t loopBoundaryProtocolIncidences =
      patternStatistics_.loopBoundaryProtocolIncidences;
  const bool loopBoundaryProtocolGenerationTruncated =
      patternStatistics_.loopBoundaryProtocolGenerationTruncated;
  patternStatistics = {};
  patternStatistics.directPairProposals = directPairProposals;
  patternStatistics.directPairEvaluations = directPairEvaluations;
  patternStatistics.directPairConnectorInspections =
      directPairConnectorInspections;
  patternStatistics.repairFrontierInspections = repairFrontierInspections;
  patternStatistics.repairFrontierProposals = repairFrontierProposals;
  patternStatistics.repairFrontierTruncated = repairFrontierTruncated;
  patternStatistics.sourcePrefixInspections = sourcePrefixInspections;
  patternStatistics.sourcePrefixCandidates = sourcePrefixCandidates;
  patternStatistics.sourcePrefixIncidences = sourcePrefixIncidences;
  patternStatistics.sourcePrefixGenerationTruncated =
      sourcePrefixGenerationTruncated;
  patternStatistics.loopCarryInspections = loopCarryInspections;
  patternStatistics.loopCarryCandidates = loopCarryCandidates;
  patternStatistics.loopCarryIncidences = loopCarryIncidences;
  patternStatistics.loopCarryGenerationTruncated = loopCarryGenerationTruncated;
  patternStatistics.loopBoundaryProtocolInspections =
      loopBoundaryProtocolInspections;
  patternStatistics.loopBoundaryProtocolCandidates =
      loopBoundaryProtocolCandidates;
  patternStatistics.loopBoundaryProtocolIncidences =
      loopBoundaryProtocolIncidences;
  patternStatistics.loopBoundaryProtocolGenerationTruncated =
      loopBoundaryProtocolGenerationTruncated;
  patterns.clear();
  patterns.reserve(mechanisms_.size() + patternSpecs_.size());
  for (const CanonicalSyncMechanism &mechanism : mechanisms_) {
    SyncCoverDemandSet coverage = projectCoverage(
        singletonCoverage.mechanisms[mechanism.id], activeDemands_);
    coverage.subtract(baselineCoverage);
    const std::size_t coverageCount = coverage.count();
    CanonicalSyncPatternKindStatistics &statistics =
        patternStatistics.kinds[static_cast<std::size_t>(
            CanonicalSyncPatternKind::Singleton)];
    ++statistics.patterns;
    statistics.jointCoverageIncidences += coverageCount;
    statistics.singletonCoverageIncidences += coverageCount;
    patterns.push_back({patterns.size(),
                        CanonicalSyncPatternKind::Singleton,
                        {mechanism.id},
                        std::move(coverage),
                        coverageCount,
                        coverageCount,
                        0});
  }
  for (const PendingPattern &pending : patternSpecs_) {
    const CanonicalSyncPatternSpec &spec = pending.spec;
    CanonicalSyncPatternKindStatistics &statistics =
        patternStatistics.kinds[static_cast<std::size_t>(spec.kind)];
    ++statistics.patterns;
    statistics.jointCoverageIncidences += pending.jointCoverageCount;
    statistics.singletonCoverageIncidences += pending.singletonCoverageCount;
    statistics.extraCoverageIncidences += pending.extraCoverageCount;
    if (pending.extraCoverageCount == 0) {
      continue;
    }
    ++statistics.patternsWithExtraCoverage;
    SyncCoverDemandSet coverage(activeDemands_.size());
    for (SyncCoverDemandId demand : pending.coverage) {
      coverage.insert(demand);
    }
    patterns.push_back({patterns.size(), spec.kind, spec.members,
                        std::move(coverage), pending.jointCoverageCount,
                        pending.singletonCoverageCount,
                        pending.extraCoverageCount});
  }
  return {};
}

CanonicalSyncProblemResult
CanonicalSyncPatternProblem::buildIncrementalPatterns(
    std::vector<CanonicalSyncPattern> &patterns,
    CanonicalSyncPatternStatistics &patternStatistics,
    SyncCoverDemandSet &baselineCoverage) {
  patterns = patterns_;
  patternStatistics = patternStatistics_;
  baselineCoverage = baselineCoverage_;
  for (CanonicalSyncMechanismId mechanism = frozenPrefixMechanismCount_;
       mechanism < mechanisms_.size(); ++mechanism) {
    if (constructionWorkBudget_ &&
        !constructionWorkBudget_->consume(
            mechanisms_[mechanism].descriptor.supplies.size())) {
      return {CanonicalSyncProblemError::LimitExceeded, mechanism};
    }
    std::optional<SyncCoverDemandSet> &cached =
        constructionSingletonCoverage_[mechanism];
    if (!cached) {
      const SyncCoverCoverageResult exact = computeSyncCoverCoverage(
          graph_, *expansion_, getSupplies(mechanisms_, {mechanism}),
          activeDemands_, constructionWorkBudget_);
      if (!exact) {
        return {constructionWorkBudget_ && constructionWorkBudget_->exhausted
                    ? CanonicalSyncProblemError::LimitExceeded
                    : CanonicalSyncProblemError::CoverageFailure,
                mechanism};
      }
      cached = projectCoverage(exact.covered, activeDemands_);
      cached->subtract(baselineCoverage);
    }
    if (constructionWorkBudget_ &&
        !constructionWorkBudget_->consume(cached->getWords().size())) {
      return {CanonicalSyncProblemError::LimitExceeded, mechanism};
    }
    SyncCoverDemandSet coverage = *cached;
    const std::size_t coverageCount = coverage.count();
    CanonicalSyncPatternKindStatistics &statistics =
        patternStatistics.kinds[static_cast<std::size_t>(
            CanonicalSyncPatternKind::Singleton)];
    ++statistics.patterns;
    statistics.jointCoverageIncidences += coverageCount;
    statistics.singletonCoverageIncidences += coverageCount;
    patterns.push_back({patterns.size(),
                        CanonicalSyncPatternKind::Singleton,
                        {mechanism},
                        std::move(coverage),
                        coverageCount,
                        coverageCount,
                        0});
  }
  for (const PendingPattern &pending : patternSpecs_) {
    if (constructionWorkBudget_ &&
        (!constructionWorkBudget_->consume(pending.spec.members.size()) ||
         !constructionWorkBudget_->consume(pending.coverage.size()))) {
      return {CanonicalSyncProblemError::LimitExceeded, std::nullopt};
    }
    CanonicalSyncPatternKindStatistics &statistics =
        patternStatistics.kinds[static_cast<std::size_t>(pending.spec.kind)];
    ++statistics.patterns;
    statistics.jointCoverageIncidences += pending.jointCoverageCount;
    statistics.singletonCoverageIncidences += pending.singletonCoverageCount;
    statistics.extraCoverageIncidences += pending.extraCoverageCount;
    if (pending.extraCoverageCount == 0) {
      continue;
    }
    ++statistics.patternsWithExtraCoverage;
    SyncCoverDemandSet coverage(activeDemands_.size());
    for (SyncCoverDemandId demand : pending.coverage) {
      coverage.insert(demand);
    }
    patterns.push_back(
        {patterns.size(), pending.spec.kind, pending.spec.members,
         std::move(coverage), pending.jointCoverageCount,
         pending.singletonCoverageCount, pending.extraCoverageCount});
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
  std::vector<CanonicalSyncPattern> patterns;
  CanonicalSyncPatternStatistics patternStatistics;
  SyncCoverDemandSet baselineCoverage(activeDemands_.size());
  CanonicalSyncProblemResult built =
      frozenPrefixMechanismCount_ == 0
          ? buildPatterns(patterns, patternStatistics, baselineCoverage)
          : buildIncrementalPatterns(patterns, patternStatistics,
                                     baselineCoverage);
  if (!built) {
    return built;
  }
  std::vector<std::vector<CanonicalSyncPatternId>> demandPatterns(
      activeDemands_.size());
  std::vector<std::vector<CanonicalSyncPatternId>> mechanismPatterns(
      mechanisms_.size());
  std::size_t incidenceCount = 0;
  for (const CanonicalSyncPattern &pattern : patterns) {
    if (constructionWorkBudget_ &&
        (!constructionWorkBudget_->consume(pattern.members.size()) ||
         !constructionWorkBudget_->consume(
             pattern.coverage.getWords().size()))) {
      return {CanonicalSyncProblemError::LimitExceeded, pattern.id};
    }
    for (CanonicalSyncMechanismId member : pattern.members) {
      mechanismPatterns[member].push_back(pattern.id);
    }
    const auto &words = pattern.coverage.getWords();
    for (std::size_t wordIndex = 0; wordIndex < words.size(); ++wordIndex) {
      std::uint64_t word = words[wordIndex];
      while (word != 0) {
        const unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
        const std::size_t demand = wordIndex * 64 + bit;
        if (demand < demandPatterns.size()) {
          if (constructionWorkBudget_ && !constructionWorkBudget_->consume()) {
            return {CanonicalSyncProblemError::LimitExceeded, pattern.id};
          }
          if (incidenceCount >= limits_.maximumIncidences) {
            return {CanonicalSyncProblemError::LimitExceeded,
                    incidenceCount + 1};
          }
          demandPatterns[demand].push_back(pattern.id);
          ++incidenceCount;
        }
        word &= word - 1;
      }
    }
  }
  std::optional<std::size_t> missing;
  for (std::size_t demand = 0; demand < demandPatterns.size(); ++demand) {
    if (constructionWorkBudget_ && !constructionWorkBudget_->consume()) {
      return {CanonicalSyncProblemError::LimitExceeded, demand};
    }
    const bool lacksCover =
        demandPatterns[demand].empty() && !baselineCoverage.contains(demand);
    if (lacksCover) {
      missing = demand;
      break;
    }
  }
  if (missing) {
    return {CanonicalSyncProblemError::UncoverableDemand, *missing};
  }
  patterns_ = std::move(patterns);
  patternStatistics_ = std::move(patternStatistics);
  baselineCoverage_ = std::move(baselineCoverage);
  demandPatterns_ = std::move(demandPatterns);
  mechanismPatterns_ = std::move(mechanismPatterns);
  incidenceCount_ = incidenceCount;
  mechanismBuckets_.clear();
  patternSpecs_.clear();
  coverageWordCount_ = 0;
  pendingCoverageIncidenceCount_ = 0;
  constructionBaselineCoverage_.reset();
  constructionSingletonCoverage_.clear();
  frozenPrefixMechanismCount_ = 0;
  constructionWorkBudget_ = nullptr;
  frozen_ = true;
  return {};
}

CanonicalSyncProblemResult CanonicalSyncPatternProblem::previewCoveredDemands(
    SyncCoverDemandSet &covered) const {
  if (frozen_) {
    return {CanonicalSyncProblemError::Frozen, std::nullopt};
  }
  std::vector<CanonicalSyncPattern> patterns;
  CanonicalSyncPatternStatistics statistics;
  SyncCoverDemandSet baseline(activeDemands_.size());
  const CanonicalSyncProblemResult built =
      buildPatterns(patterns, statistics, baseline);
  if (!built) {
    return built;
  }
  SyncCoverDemandSet activeCovered = std::move(baseline);
  for (const CanonicalSyncPattern &pattern : patterns) {
    activeCovered.unite(pattern.coverage);
  }
  covered = SyncCoverDemandSet(graph_.getDemands().size());
  for (std::size_t active = 0; active < activeDemands_.size(); ++active) {
    if (activeCovered.contains(active)) {
      covered.insert(activeDemands_[active]);
    }
  }
  return {};
}
