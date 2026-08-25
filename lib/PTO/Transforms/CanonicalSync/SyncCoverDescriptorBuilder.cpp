// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverDescriptorBuilder.h"

#include <algorithm>
#include <type_traits>
#include <utility>

using namespace mlir::pto;

namespace {

template <typename T>
bool reserveAdditional(std::vector<T> &values, std::size_t additional) {
  const bool overflow = additional > values.max_size() - values.size();
  if (overflow) {
    return false;
  }
  values.reserve(values.size() + additional);
  return true;
}

bool hasAnchor(const SyncCoverResourceAction &action,
               SyncCoverResourceActionKind kind, std::uint32_t resource,
               SyncCoverAnchorKind anchorKind, SyncCoverNodeId node,
               SyncCoverScopeId scope) {
  return action.kind == kind && action.resource == resource &&
         action.anchor.kind == anchorKind && action.anchor.node == node &&
         action.anchor.scope == scope;
}

} // namespace

SyncCoverMechanismDescriptorBuilder::SyncCoverMechanismDescriptorBuilder(
    SyncCoverMechanismKind kind, std::uint64_t providerIdentity) {
  descriptor_.kind = kind;
  descriptor_.providerIdentity = providerIdentity;
}

SyncCoverDescriptorActionRef
SyncCoverMechanismDescriptorBuilder::addAction(
    SyncCoverResourceActionKind kind, std::uint32_t resource,
    SyncCoverAnchor anchor) {
  const std::size_t index = descriptor_.actions.size();
  descriptor_.actions.push_back({kind, resource, anchor});
  return {index};
}

bool SyncCoverMechanismDescriptorBuilder::addCanonicalEvent(
    const SyncCoverResourceDomain &domain, SyncCoverNodeId source,
    SyncCoverNodeId target, SyncCoverScopeId scope, std::size_t width) {
  if (descriptor_.kind != SyncCoverMechanismKind::EventBundle ||
      domain.kind != SyncCoverResourceKind::EventId || width == 0) {
    return false;
  }

  const std::size_t produce = descriptor_.actions.size();
  const std::size_t consume = produce + 1;
  const std::size_t edgeIndex = descriptor_.supplyEdges.size();
  const std::size_t useIndex = descriptor_.resourceUses.size();
  const SyncCoverResourceAction produceAction{
      SyncCoverResourceActionKind::Produce, domain.sourceResource,
      {SyncCoverAnchorKind::AfterNode, source, 0}};
  const SyncCoverResourceAction consumeAction{
      SyncCoverResourceActionKind::Consume, domain.targetResource,
      {SyncCoverAnchorKind::BeforeNode, target, 0}};
  SyncCoverEdge edge;
  edge.source = source;
  edge.target = target;
  edge.kind = SyncCoverEdgeKind::CompletionSupply;
  edge.scope = scope;
  SyncCoverResourceUse use;
  use.domain = domain.id;
  use.scope = scope;
  use.width = width;
  use.actions = {produce, consume};
  use.supplyEdges = {edgeIndex};
  const SyncCoverSupplyBinding binding{edgeIndex, useIndex, produce, consume};

  const bool reserved = reserveAdditional(descriptor_.actions, 2) &&
                        reserveAdditional(descriptor_.supplyEdges, 1) &&
                        reserveAdditional(descriptor_.resourceUses, 1) &&
                        reserveAdditional(descriptor_.supplyBindings, 1);
  if (!reserved) {
    return false;
  }
  static_assert(
      std::is_nothrow_copy_constructible<SyncCoverResourceAction>::value &&
          std::is_nothrow_move_constructible<SyncCoverEdge>::value &&
          std::is_nothrow_move_constructible<SyncCoverResourceUse>::value &&
          std::is_nothrow_copy_constructible<SyncCoverSupplyBinding>::value,
      "reserved descriptor commit must not throw");
  descriptor_.actions.push_back(produceAction);
  descriptor_.actions.push_back(consumeAction);
  descriptor_.supplyEdges.push_back(std::move(edge));
  descriptor_.resourceUses.push_back(std::move(use));
  descriptor_.supplyBindings.push_back(binding);
  return true;
}

bool SyncCoverMechanismDescriptorBuilder::addProtocolLane(
    const SyncCoverResourceDomain &domain, SyncCoverScopeId scope,
    unsigned distance, std::size_t width,
    std::vector<SyncCoverDescriptorActionRef> actions,
    std::vector<SyncCoverProtocolSupply> supplies) {
  if (descriptor_.kind != SyncCoverMechanismKind::VerifiedProtocol) {
    return false;
  }
  return addLane(domain, scope, distance, width, std::move(actions),
                 std::move(supplies));
}

bool SyncCoverMechanismDescriptorBuilder::addLane(
    const SyncCoverResourceDomain &domain, SyncCoverScopeId scope,
    unsigned distance, std::size_t width,
    std::vector<SyncCoverDescriptorActionRef> actions,
    std::vector<SyncCoverProtocolSupply> supplies) {
  const bool emptyLane = width == 0 || actions.empty() || supplies.empty();
  if (emptyLane) {
    return false;
  }
  std::sort(actions.begin(), actions.end(), [](const auto &first,
                                                const auto &second) {
    return first.index < second.index;
  });
  const auto duplicate = std::adjacent_find(
      actions.begin(), actions.end(), [](const auto &first, const auto &second) {
        return first.index == second.index;
      });
  if (duplicate != actions.end()) {
    return false;
  }
  const auto hasAction = [&](SyncCoverDescriptorActionRef action) {
    return std::binary_search(
        actions.begin(), actions.end(), action,
        [](const auto &first, const auto &second) {
          return first.index < second.index;
        });
  };
  for (SyncCoverDescriptorActionRef action : actions) {
    if (action.index >= descriptor_.actions.size()) {
      return false;
    }
    const SyncCoverResourceAction &stored = descriptor_.actions[action.index];
    const std::uint32_t expected =
        stored.kind == SyncCoverResourceActionKind::Produce
            ? domain.sourceResource
            : domain.targetResource;
    if (stored.resource != expected) {
      return false;
    }
  }
  for (const SyncCoverProtocolSupply &supply : supplies) {
    const bool invalidActions =
        !hasAction(supply.produceAction) || !hasAction(supply.consumeAction) ||
        descriptor_.actions[supply.produceAction.index].kind !=
            SyncCoverResourceActionKind::Produce ||
        descriptor_.actions[supply.consumeAction.index].kind !=
            SyncCoverResourceActionKind::Consume;
    const bool straightLifetime = distance == 0;
    const bool invalidLifetime =
        straightLifetime
            ? supply.edge.distance != 0 || supply.edge.scope != scope
            : supply.edge.distance > distance;
    const bool invalidEdge =
        supply.edge.kind != SyncCoverEdgeKind::CompletionSupply ||
        supply.edge.mechanism || invalidLifetime;
    if (invalidActions || invalidEdge) {
      return false;
    }
  }

  const std::size_t firstEdge = descriptor_.supplyEdges.size();
  const std::size_t useIndex = descriptor_.resourceUses.size();
  SyncCoverResourceUse use;
  use.domain = domain.id;
  use.scope = scope;
  use.distance = distance;
  use.width = width;
  use.actions.reserve(actions.size());
  for (SyncCoverDescriptorActionRef action : actions) {
    use.actions.push_back(action.index);
  }
  use.supplyEdges.reserve(supplies.size());
  std::vector<SyncCoverEdge> edges;
  std::vector<SyncCoverSupplyBinding> bindings;
  edges.reserve(supplies.size());
  bindings.reserve(supplies.size());
  for (std::size_t index = 0; index < supplies.size(); ++index) {
    const std::size_t edgeIndex = firstEdge + index;
    use.supplyEdges.push_back(edgeIndex);
    edges.push_back(std::move(supplies[index].edge));
    bindings.push_back(
        {edgeIndex, useIndex, supplies[index].produceAction.index,
         supplies[index].consumeAction.index});
  }

  const bool reserved =
      reserveAdditional(descriptor_.supplyEdges, edges.size()) &&
      reserveAdditional(descriptor_.resourceUses, 1) &&
      reserveAdditional(descriptor_.supplyBindings, bindings.size());
  if (!reserved) {
    return false;
  }
  static_assert(
      std::is_nothrow_move_constructible<SyncCoverEdge>::value &&
          std::is_nothrow_move_constructible<SyncCoverResourceUse>::value &&
          std::is_nothrow_copy_constructible<SyncCoverSupplyBinding>::value,
      "reserved protocol-lane commit must not throw");
  for (SyncCoverEdge &edge : edges) {
    descriptor_.supplyEdges.push_back(std::move(edge));
  }
  descriptor_.resourceUses.push_back(std::move(use));
  for (const SyncCoverSupplyBinding &binding : bindings) {
    descriptor_.supplyBindings.push_back(binding);
  }
  return true;
}

SyncCoverMechanismDescriptor
SyncCoverMechanismDescriptorBuilder::takeDescriptor() && {
  return std::move(descriptor_);
}

std::optional<SyncCoverMechanismDescriptor>
mlir::pto::makeSyncCoverCanonicalEvent(
    const SyncCoverResourceDomain &domain, SyncCoverNodeId source,
    SyncCoverNodeId target, SyncCoverScopeId scope, std::size_t width,
    std::uint64_t providerIdentity) {
  SyncCoverMechanismDescriptorBuilder builder(
      SyncCoverMechanismKind::EventBundle, providerIdentity);
  if (!builder.addCanonicalEvent(domain, source, target, scope, width)) {
    return std::nullopt;
  }
  return std::move(builder).takeDescriptor();
}

std::optional<SyncCoverMechanismDescriptor>
mlir::pto::makeSyncCoverUnitRecurrenceEvent(
    const SyncCoverResourceDomain &domain, SyncCoverNodeId source,
    SyncCoverNodeId target, SyncCoverScopeId loop,
    std::uint64_t providerIdentity) {
  if (domain.kind != SyncCoverResourceKind::EventId) {
    return std::nullopt;
  }
  SyncCoverMechanismDescriptorBuilder builder(
      SyncCoverMechanismKind::VerifiedProtocol, providerIdentity);
  const SyncCoverDescriptorActionRef prime = builder.addAction(
      SyncCoverResourceActionKind::Produce, domain.sourceResource,
      {SyncCoverAnchorKind::ScopeEntry, 0, loop});
  const SyncCoverDescriptorActionRef bodyWait = builder.addAction(
      SyncCoverResourceActionKind::Consume, domain.targetResource,
      {SyncCoverAnchorKind::BeforeNode, target, 0});
  const SyncCoverDescriptorActionRef bodySet = builder.addAction(
      SyncCoverResourceActionKind::Produce, domain.sourceResource,
      {SyncCoverAnchorKind::AfterNode, source, 0});
  const SyncCoverDescriptorActionRef drain = builder.addAction(
      SyncCoverResourceActionKind::Consume, domain.targetResource,
      {SyncCoverAnchorKind::ScopeExit, 0, loop});
  SyncCoverEdge edge;
  edge.source = source;
  edge.target = target;
  edge.kind = SyncCoverEdgeKind::CompletionSupply;
  edge.scope = loop;
  edge.distance = 1;
  if (!builder.addProtocolLane(domain, loop, 1, 1,
                               {prime, bodyWait, bodySet, drain},
                               {{edge, bodySet, bodyWait}})) {
    return std::nullopt;
  }
  return std::move(builder).takeDescriptor();
}

bool mlir::pto::verifySyncCoverUnitRecurrenceEvent(
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverMechanismDescriptor &descriptor) {
  const bool invalidCardinality =
      descriptor.kind != SyncCoverMechanismKind::VerifiedProtocol ||
      descriptor.barrier || descriptor.actions.size() != 4 ||
      descriptor.resourceUses.size() != 1 ||
      descriptor.supplyEdges.size() != 1 ||
      descriptor.supplyBindings.size() != 1;
  if (invalidCardinality) {
    return false;
  }
  const SyncCoverResourceUse &use = descriptor.resourceUses.front();
  if (use.domain >= universe.getResourceDomains().size()) {
    return false;
  }
  const SyncCoverGraph &graph = universe.getGraph();
  const SyncCoverResourceDomain &domain =
      universe.getResourceDomains()[use.domain];
  const SyncCoverEdge &edge = descriptor.supplyEdges.front();
  const SyncCoverSupplyBinding &binding = descriptor.supplyBindings.front();
  const bool invalidIdentity =
      domain.kind != SyncCoverResourceKind::EventId ||
      domain.sourceResource == domain.targetResource ||
      domain.poolIdentity != 0 || domain.budget == 0 ||
      use.domain != domain.id || use.scope >= graph.getScopes().size() ||
      edge.source >= graph.getNodes().size() ||
      edge.target >= graph.getNodes().size() ||
      !graph.getScopes()[use.scope].isLoop || use.distance != 1 ||
      use.width != 1 || use.actions != std::vector<std::size_t>({0, 1, 2, 3}) ||
      use.supplyEdges != std::vector<std::size_t>({0}) ||
      edge.kind != SyncCoverEdgeKind::CompletionSupply || edge.mechanism ||
      edge.scope != use.scope || edge.distance != 1 ||
      binding.supplyEdge != 0 || binding.resourceUse != 0 ||
      binding.produceAction != 2 || binding.consumeAction != 1;
  if (invalidIdentity) {
    return false;
  }
  const SyncCoverNode &source = graph.getNodes()[edge.source];
  const SyncCoverNode &target = graph.getNodes()[edge.target];
  const std::optional<std::size_t> recurrenceDepth =
      graph.getScopeLoopDepth(use.scope);
  const std::optional<std::size_t> sourceDepth =
      graph.getScopeLoopDepth(source.scope);
  const std::optional<std::size_t> targetDepth =
      graph.getScopeLoopDepth(target.scope);
  const bool invalidExecution =
      !graph.scopeMustExecuteWithin(use.scope, source.scope) ||
      !graph.scopeMustExecuteWithin(use.scope, target.scope) ||
      !recurrenceDepth || sourceDepth != recurrenceDepth ||
      targetDepth != recurrenceDepth ||
      source.resource != domain.sourceResource ||
      target.resource != domain.targetResource ||
      !source.guard.literals.empty() || !target.guard.literals.empty() ||
      target.order >= source.order ||
      !syncCoverNodeCanProduceCompletion(graph, edge.source, target.resource);
  if (invalidExecution) {
    return false;
  }
  const auto &actions = descriptor.actions;
  return hasAnchor(actions[0], SyncCoverResourceActionKind::Produce,
                   source.resource, SyncCoverAnchorKind::ScopeEntry, 0,
                   use.scope) &&
         hasAnchor(actions[1], SyncCoverResourceActionKind::Consume,
                   target.resource, SyncCoverAnchorKind::BeforeNode,
                   edge.target, 0) &&
         hasAnchor(actions[2], SyncCoverResourceActionKind::Produce,
                   source.resource, SyncCoverAnchorKind::AfterNode, edge.source,
                   0) &&
         hasAnchor(actions[3], SyncCoverResourceActionKind::Consume,
                   target.resource, SyncCoverAnchorKind::ScopeExit, 0,
                   use.scope);
}
