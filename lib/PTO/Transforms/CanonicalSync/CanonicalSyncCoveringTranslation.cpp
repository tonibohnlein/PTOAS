// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncCoveringSelection.h"

#include "PTO/Transforms/CanonicalSync/SyncCoverDescriptorBuilder.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_covering;

namespace {

bool sameAnchor(const SyncCoverAnchor &first, const SyncCoverAnchor &second) {
  return first.kind == second.kind && first.node == second.node &&
         first.scope == second.scope && first.position == second.position;
}

bool sameEdge(const SyncCoverEdge &first, const SyncCoverEdge &second) {
  return first.source == second.source && first.target == second.target &&
         first.kind == second.kind && first.scope == second.scope &&
         first.distance == second.distance &&
         first.sourceGuard.literals == second.sourceGuard.literals &&
         first.targetGuard.literals == second.targetGuard.literals &&
         first.mechanism == second.mechanism;
}

bool sameAction(const SyncCoverResourceAction &first,
                const SyncCoverResourceAction &second) {
  return first.kind == second.kind && first.resource == second.resource &&
         sameAnchor(first.anchor, second.anchor);
}

bool sameUse(const SyncCoverResourceUse &first,
             const SyncCoverResourceUse &second) {
  return first.domain == second.domain && first.scope == second.scope &&
         first.distance == second.distance && first.width == second.width &&
         first.actions == second.actions &&
         first.supplyEdges == second.supplyEdges;
}

bool sameBinding(const SyncCoverSupplyBinding &first,
                 const SyncCoverSupplyBinding &second) {
  return first.supplyEdge == second.supplyEdge &&
         first.resourceUse == second.resourceUse &&
         first.produceAction == second.produceAction &&
         first.consumeAction == second.consumeAction;
}

std::optional<SyncCoverAnchor> translateActionAnchor(
    const CanonicalEventAction &action, Operation *protocolLoop,
    const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
        &regionScopes,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    llvm::function_ref<std::size_t(const CanonicalAnchor &)>
        getAnchorPosition) {
  if (protocolLoop && action.phase == CanonicalEventActionPhase::Prime) {
    auto loop = loopScopes.find(protocolLoop);
    if (loop != loopScopes.end()) {
      return SyncCoverAnchor{SyncCoverAnchorKind::ScopeEntry, 0, loop->second,
                             0};
    }
  }
  if (protocolLoop && action.phase == CanonicalEventActionPhase::Drain) {
    auto loop = loopScopes.find(protocolLoop);
    if (loop != loopScopes.end()) {
      return SyncCoverAnchor{SyncCoverAnchorKind::ScopeExit, 0, loop->second,
                             0};
    }
  }
  const std::optional<SyncCoverScopeId> scope =
      getAnchorOccurrenceScope(action.anchor, regionScopes);
  if (!scope) {
    return std::nullopt;
  }
  return SyncCoverAnchor{SyncCoverAnchorKind::TimelinePoint, 0, *scope,
                         getAnchorPosition(action.anchor)};
}

std::optional<SyncCoverEdge> translateCompletion(
    const CanonicalEventCompletion &completion, const SyncCoverGraph &graph,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes) {
  SyncCoverEdge edge;
  edge.source = completion.source;
  edge.target = completion.target;
  edge.kind = SyncCoverEdgeKind::CompletionSupply;
  edge.distance = completion.iterationDistance;
  if (edge.distance != 0) {
    auto loop = loopScopes.find(completion.recurrenceLoop);
    if (loop == loopScopes.end()) {
      return std::nullopt;
    }
    edge.scope = loop->second;
    return edge;
  }
  const std::optional<SyncCoverScopeId> scope =
      getEndpointScope(graph, edge.source, edge.target);
  if (!scope) {
    return std::nullopt;
  }
  edge.scope = *scope;
  return edge;
}

} // namespace

std::optional<std::uint64_t>
mlir::pto::canonical_sync_covering::encodeProviderIdentity(
    CanonicalSelectionMechanismKind kind, std::size_t id) {
  constexpr std::uint64_t kKindCount = 2;
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const bool identityOverflows = id > (maximum - 1) / kKindCount;
  if (identityOverflows) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(id) * kKindCount +
         static_cast<std::uint64_t>(kind) + 1;
}

bool mlir::pto::canonical_sync_covering::sameDescriptor(
    const SyncCoverMechanismDescriptor &first,
    const SyncCoverMechanismDescriptor &second) {
  const bool sameBarrier =
      first.barrier.has_value() == second.barrier.has_value() &&
      (!first.barrier ||
       (first.barrier->resource == second.barrier->resource &&
        first.barrier->scope == second.barrier->scope &&
        sameAnchor(first.barrier->anchor, second.barrier->anchor)));
  return first.kind == second.kind &&
         first.providerIdentity == second.providerIdentity && sameBarrier &&
         llvm::equal(first.supplyEdges, second.supplyEdges, sameEdge) &&
         llvm::equal(first.actions, second.actions, sameAction) &&
         llvm::equal(first.resourceUses, second.resourceUses, sameUse) &&
         llvm::equal(first.supplyBindings, second.supplyBindings, sameBinding);
}

bool mlir::pto::canonical_sync_covering::barriersEquivalent(
    const CanonicalBarrier &first, const CanonicalBarrier &second) {
  return first.pipe == second.pipe &&
         first.anchor.operation == second.anchor.operation &&
         first.anchor.before == second.anchor.before &&
         first.recurrenceLoop == second.recurrenceLoop;
}

std::optional<SyncCoverScopeId>
mlir::pto::canonical_sync_covering::getEndpointScope(
    const SyncCoverGraph &graph, std::size_t source, std::size_t target) {
  const bool endpointMissing = source >= graph.getNodes().size() ||
                               target >= graph.getNodes().size();
  if (endpointMissing) {
    return std::nullopt;
  }
  return graph.getLowestCommonScope(graph.getNodes()[source].scope,
                                    graph.getNodes()[target].scope);
}

std::optional<SyncCoverScopeId>
mlir::pto::canonical_sync_covering::getAnchorOccurrenceScope(
    const CanonicalAnchor &anchor,
    const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
        &regionScopes) {
  if (!anchor.operation) {
    return std::nullopt;
  }
  auto scope = regionScopes.find(anchor.operation->getParentRegion());
  return scope == regionScopes.end()
             ? std::nullopt
             : std::optional<SyncCoverScopeId>(scope->second);
}

bool mlir::pto::canonical_sync_covering::isCanonicalForwardEvent(
    const CanonicalEventBundleCandidate &bundle, const SyncCoverGraph &graph,
    llvm::function_ref<std::size_t(const CanonicalAnchor &)>
        getAnchorPosition) {
  if (bundle.kind != CanonicalEventBundleKind::Standalone ||
      bundle.events.size() != 1) {
    return false;
  }
  const CanonicalEvent &event = bundle.events.front();
  if (event.recurrenceLoop || event.scopeLoop || event.forwardDrainLoop ||
      event.width != 1 || event.actions.size() != 2 ||
      event.completions.size() != 1 ||
      event.source >= graph.getNodes().size() ||
      event.target >= graph.getNodes().size()) {
    return false;
  }
  const CanonicalEventCompletion &completion = event.completions.front();
  return completion.source == event.source &&
         completion.target == event.target &&
         completion.iterationDistance == 0 && completion.setAction == 0 &&
         completion.waitAction == 1 &&
         event.actions[0].kind == CanonicalEventActionKind::Set &&
         event.actions[1].kind == CanonicalEventActionKind::Wait &&
         event.setAnchor.operation && event.waitAnchor.operation &&
         event.setAnchor.operation == event.actions[0].anchor.operation &&
         event.setAnchor.before == event.actions[0].anchor.before &&
         event.waitAnchor.operation == event.actions[1].anchor.operation &&
         event.waitAnchor.before == event.actions[1].anchor.before &&
         getAnchorPosition(event.setAnchor) ==
             graph.getNodes()[event.source].order * 2 + 1 &&
         getAnchorPosition(event.waitAnchor) ==
             graph.getNodes()[event.target].order * 2;
}

std::optional<SyncCoverMechanismDescriptor>
mlir::pto::canonical_sync_covering::translateVerifiedEventBundle(
    const CanonicalEventBundleCandidate &bundle, std::uint64_t provider,
    const DomainMap &domains, const SyncCoverMechanismUniverse &universe,
    const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
        &regionScopes,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    llvm::function_ref<std::size_t(const CanonicalAnchor &)>
        getAnchorPosition) {
  SyncCoverMechanismDescriptorBuilder builder(
      SyncCoverMechanismKind::VerifiedProtocol, provider);
  for (const CanonicalEvent &event : bundle.events) {
    const CanonicalEventDomainKey key{event.sourcePipe, event.targetPipe};
    auto domainId = domains.find(key);
    if (domainId == domains.end()) {
      return std::nullopt;
    }
    const SyncCoverResourceDomain &domain =
        universe.getResourceDomains()[domainId->second];
    Operation *protocolLoop = event.scopeLoop
                                  ? event.scopeLoop
                                  : (event.recurrenceLoop
                                         ? event.recurrenceLoop
                                         : event.forwardDrainLoop);
    SyncCoverScopeId useScope = 0;
    unsigned useDistance = 0;
    if (protocolLoop) {
      auto loop = loopScopes.find(protocolLoop);
      if (loop == loopScopes.end()) {
        return std::nullopt;
      }
      useScope = loop->second;
      useDistance = std::max(1U, event.iterationDistance);
    }

    std::vector<SyncCoverDescriptorActionRef> actions;
    actions.reserve(event.actions.size());
    for (const CanonicalEventAction &action : event.actions) {
      const std::optional<SyncCoverAnchor> anchor = translateActionAnchor(
          action, protocolLoop, regionScopes, loopScopes, getAnchorPosition);
      if (!anchor) {
        return std::nullopt;
      }
      const bool produce = action.kind == CanonicalEventActionKind::Set;
      actions.push_back(builder.addAction(
          produce ? SyncCoverResourceActionKind::Produce
                  : SyncCoverResourceActionKind::Consume,
          produce ? domain.sourceResource : domain.targetResource, *anchor));
    }

    std::vector<SyncCoverProtocolSupply> supplies;
    supplies.reserve(event.completions.size());
    for (const CanonicalEventCompletion &completion : event.completions) {
      const bool actionMissing = completion.setAction >= actions.size() ||
                                 completion.waitAction >= actions.size();
      if (actionMissing) {
        return std::nullopt;
      }
      const std::optional<SyncCoverEdge> edge =
          translateCompletion(completion, universe.getGraph(), loopScopes);
      if (!edge) {
        return std::nullopt;
      }
      useDistance = std::max(useDistance, completion.iterationDistance);
      supplies.push_back(
          {*edge, actions[completion.setAction], actions[completion.waitAction]});
    }
    if (supplies.empty()) {
      return std::nullopt;
    }
    if (!protocolLoop) {
      useScope = supplies.front().edge.scope;
      const bool inconsistentScope = llvm::any_of(
          supplies, [&](const SyncCoverProtocolSupply &supply) {
            return supply.edge.scope != useScope || supply.edge.distance != 0;
          });
      if (inconsistentScope) {
        return std::nullopt;
      }
    }
    if (!builder.addProtocolLane(domain, useScope, useDistance, event.width,
                                 std::move(actions), std::move(supplies))) {
      return std::nullopt;
    }
  }
  return std::move(builder).takeDescriptor();
}

bool mlir::pto::canonical_sync_covering::verifyBundleShape(
    const CanonicalEventBundleCandidate &bundle) {
  SmallVector<const CanonicalEvent *, 2> events;
  for (const CanonicalEvent &event : bundle.events) {
    events.push_back(&event);
  }
  if (bundle.kind != CanonicalEventBundleKind::SyntheticRoundTrip) {
    return true;
  }
  if (!verifyCanonicalSyntheticRoundTripBundle(events)) {
    return false;
  }
  return !bundle.completionWitness || verifyCanonicalSyntheticRoundTripWitness(
                                          events, *bundle.completionWitness);
}
