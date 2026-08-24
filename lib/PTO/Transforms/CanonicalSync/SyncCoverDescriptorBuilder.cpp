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
    const bool invalidEdge =
        supply.edge.kind != SyncCoverEdgeKind::CompletionSupply ||
        supply.edge.mechanism || supply.edge.scope != scope ||
        supply.edge.distance != distance;
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
