// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncOwnership.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct OwnershipSpec {
  CanonicalSyncOwnershipKind kind = CanonicalSyncOwnershipKind::L0Operand;
  std::uint32_t producerResource = 0;
  std::uint32_t consumerResource = 0;
  std::vector<AddressSpace> spaces;
  std::size_t requiredSpaces = 1;
  std::size_t minimumLanes = 1;
};

struct HierarchicalOwnershipSpec {
  CanonicalSyncOwnershipKind kind = CanonicalSyncOwnershipKind::L1Tile;
  std::uint32_t producerResource = 0;
  std::uint32_t consumerResource = 0;
  AddressSpace space = AddressSpace::Zero;
  std::size_t minimumLanes = 1;
  bool allowProducerReads = false;
};

struct OwnershipNode {
  SyncCoverNodeId id = 0;
  Operation *operation = nullptr;
  Operation *topLevel = nullptr;
  Region *path = nullptr;
  std::vector<CanonicalSyncOwnershipSlot> produced;
  std::vector<CanonicalSyncOwnershipSlot> consumed;
};

struct PathItem {
  bool producer = false;
  std::size_t order = 0;
  OwnershipNode *node = nullptr;
  Operation *consumerAnchor = nullptr;
  std::vector<OwnershipNode *> consumers;
};

using SlotBundle = std::vector<CanonicalSyncOwnershipSlot>;
using StorageAccessIndex =
    std::vector<std::vector<const SyncCoverStorageAccess *>>;

struct HierarchicalSlotGroup {
  CanonicalSyncOwnershipSlot slot;
  std::vector<OwnershipNode *> producers;
  std::vector<OwnershipNode *> consumers;
};

struct ParitySlotGroup {
  CanonicalSyncOwnershipSlot slot;
  std::array<std::vector<OwnershipNode *>, 2> producers;
  std::array<std::vector<OwnershipNode *>, 2> consumers;
};

std::optional<StorageAccessIndex>
buildStorageAccessIndex(const SyncCoverGraph &graph) {
  StorageAccessIndex result(graph.getNodes().size());
  for (const SyncCoverStorageAccess &access : graph.getStorageAccesses()) {
    if (access.node >= result.size()) {
      return std::nullopt;
    }
    result[access.node].push_back(&access);
  }
  return result;
}

bool intervalOverlaps(SyncCoverStorageInterval first,
                      SyncCoverStorageInterval second) {
  return first.begin < second.end && second.begin < first.end;
}

bool includesSpace(const OwnershipSpec &spec, AddressSpace space) {
  return llvm::is_contained(spec.spaces, space);
}

std::optional<AddressSpace>
getAccessSpace(const CanonicalSyncProgram &program,
               const SyncCoverStorageAccess &access) {
  return access.domain < program.getStorageSpaces().size()
             ? std::optional<AddressSpace>(
                   program.getStorageSpaces()[access.domain])
             : std::nullopt;
}

Operation *getTopLevelChild(Operation *operation, Operation *loop) {
  Operation *child = operation;
  while (child && child->getParentOp() != loop) {
    child = child->getParentOp();
  }
  return child;
}

Operation *getTopLevelInBlock(Operation *operation, Block *block) {
  Operation *topLevel = operation;
  while (topLevel && topLevel->getBlock() != block) {
    topLevel = topLevel->getParentOp();
  }
  return topLevel;
}

bool isBeforeLoop(Operation *operation, scf::ForOp loop) {
  Operation *topLevel = getTopLevelInBlock(operation, loop->getBlock());
  return topLevel && topLevel != loop && topLevel->isBeforeInBlock(loop);
}

bool hasInterveningResourceNode(const CanonicalSyncProgram &program,
                                SyncCoverNodeId source, scf::ForOp loop,
                                std::uint32_t resource) {
  Operation *sourceOperation = program.getNodeBindings()[source].operation;
  if (!sourceOperation || !isBeforeLoop(sourceOperation, loop)) {
    return true;
  }
  for (const SyncCoverNode &node : program.getGraph().getNodes()) {
    if (node.id == source || node.resource != resource) {
      continue;
    }
    Operation *operation = program.getNodeBindings()[node.id].operation;
    Operation *topLevel = getTopLevelInBlock(operation, loop->getBlock());
    if (topLevel && topLevel != loop &&
        sourceOperation->isBeforeInBlock(topLevel) &&
        topLevel->isBeforeInBlock(loop)) {
      return true;
    }
  }
  return false;
}

Region *getPathRegion(Operation *operation, Operation *branch) {
  Operation *child = operation;
  while (child && child->getParentOp() != branch) {
    child = child->getParentOp();
  }
  return child ? child->getParentRegion() : nullptr;
}

Operation *getPathAnchor(Operation *operation, Region *path) {
  Operation *anchor = operation;
  while (anchor && anchor->getParentRegion() != path) {
    anchor = anchor->getParentOp();
  }
  return anchor;
}

std::optional<SyncCoverScopeId> findScope(const CanonicalSyncProgram &program,
                                          Region *region) {
  for (auto [index, binding] : llvm::enumerate(program.getScopeBindings())) {
    if (binding.region == region) {
      return index;
    }
  }
  return std::nullopt;
}

Operation *getCommonTopLevel(ArrayRef<OwnershipNode *> nodes) {
  if (nodes.empty()) {
    return nullptr;
  }
  Operation *anchor = nodes.front()->topLevel;
  return llvm::all_of(nodes,
                      [&](const OwnershipNode *node) {
                        return node->topLevel == anchor;
                      })
             ? anchor
             : nullptr;
}

std::optional<std::pair<SyncCoverAnchor, SyncCoverAnchor>>
getBoundaryAnchors(const CanonicalSyncProgram &program, Operation *anchor) {
  if (!anchor) {
    return std::nullopt;
  }
  for (auto [node, binding] : llvm::enumerate(program.getNodeBindings())) {
    if (binding.operation == anchor) {
      return std::make_pair(
          SyncCoverAnchor{SyncCoverAnchorKind::BeforeNode, node, 0, 0},
          SyncCoverAnchor{SyncCoverAnchorKind::AfterNode, node, 0, 0});
    }
  }
  auto loop = dyn_cast<scf::ForOp>(anchor);
  if (!loop) {
    return std::nullopt;
  }
  const std::optional<SyncCoverScopeId> scope =
      findScope(program, &loop.getRegion());
  if (!scope) {
    return std::nullopt;
  }
  return std::make_pair(
      SyncCoverAnchor{SyncCoverAnchorKind::ScopeEntry, 0, *scope, 0},
      SyncCoverAnchor{SyncCoverAnchorKind::ScopeExit, 0, *scope, 0});
}

bool slotsAreDisjoint(ArrayRef<CanonicalSyncOwnershipLane> lanes) {
  for (std::size_t first = 0; first < lanes.size(); ++first) {
    for (std::size_t second = first + 1; second < lanes.size(); ++second) {
      for (const CanonicalSyncOwnershipSlot &left : lanes[first].slots) {
        for (const CanonicalSyncOwnershipSlot &right : lanes[second].slots) {
          if (left.domain == right.domain &&
              intervalOverlaps(left.extent, right.extent)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

std::optional<std::vector<OwnershipNode>> collectOwnershipNodes(
    const CanonicalSyncProgram &program, SyncCoverScopeId recurrenceScope,
    const OwnershipSpec &spec,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode) {
  const SyncCoverGraph &graph = program.getGraph();
  if (recurrenceScope >= program.getScopeBindings().size()) {
    return std::nullopt;
  }
  Operation *loop = program.getScopeBindings()[recurrenceScope].owner;
  if (!isa_and_nonnull<scf::ForOp>(loop)) {
    return std::nullopt;
  }
  std::vector<OwnershipNode> result;
  for (const SyncCoverNode &node : graph.getNodes()) {
    if (!graph.scopeContains(recurrenceScope, node.scope)) {
      continue;
    }
    std::vector<const SyncCoverStorageAccess *> accesses;
    for (const SyncCoverStorageAccess *access : accessesByNode[node.id]) {
      const std::optional<AddressSpace> space =
          getAccessSpace(program, *access);
      if (space && includesSpace(spec, *space)) {
        if (!access->exactPhysical) {
          return std::nullopt;
        }
        accesses.push_back(access);
      }
    }
    if (accesses.empty()) {
      continue;
    }
    OwnershipNode ownership;
    ownership.id = node.id;
    ownership.operation = program.getNodeBindings()[node.id].operation;
    ownership.topLevel = getTopLevelChild(ownership.operation, loop);
    if (!ownership.operation || !ownership.topLevel) {
      return std::nullopt;
    }
    if (node.resource == spec.producerResource) {
      for (const SyncCoverStorageAccess *access : accesses) {
        if (access->mode != SyncCoverStorageAccessMode::Write) {
          return std::nullopt;
        }
        ownership.produced.push_back({access->domain, access->extent});
      }
      llvm::sort(ownership.produced);
    } else if (node.resource == spec.consumerResource) {
      if (llvm::any_of(accesses, [](const SyncCoverStorageAccess *access) {
            return access->mode != SyncCoverStorageAccessMode::Read;
          })) {
        return std::nullopt;
      }
      llvm::transform(accesses, std::back_inserter(ownership.consumed),
                      [](const SyncCoverStorageAccess *access) {
                        return CanonicalSyncOwnershipSlot{access->domain,
                                                          access->extent};
                      });
      llvm::sort(ownership.consumed);
      if (ownership.consumed.size() != spec.requiredSpaces) {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
    result.push_back(std::move(ownership));
  }
  return result.empty()
             ? std::nullopt
             : std::optional<std::vector<OwnershipNode>>(std::move(result));
}

bool validateConsumerGroup(Operation *anchor,
                           ArrayRef<OwnershipNode *> consumers) {
  auto branch = dyn_cast_or_null<scf::IfOp>(anchor);
  if (!branch) {
    return consumers.size() == 1 && consumers.front()->operation == anchor;
  }
  if (branch.getElseRegion().empty() || consumers.size() != 2) {
    return false;
  }
  bool thenConsumer = false;
  bool elseConsumer = false;
  for (OwnershipNode *consumer : consumers) {
    Region *region = getPathRegion(consumer->operation, anchor);
    if (region == &branch.getThenRegion()) {
      thenConsumer = !thenConsumer;
    } else if (region == &branch.getElseRegion()) {
      elseConsumer = !elseConsumer;
    } else {
      return false;
    }
  }
  return thenConsumer && elseConsumer;
}

std::optional<CanonicalSyncOwnershipPath> parseOwnershipPath(
    const CanonicalSyncProgram &program, Region *path,
    MutableArrayRef<OwnershipNode> nodes, const OwnershipSpec &spec,
    std::map<SlotBundle, unsigned> &bundleLanes, bool allowNewBundles) {
  const SyncCoverGraph &graph = program.getGraph();
  std::vector<PathItem> items;
  std::map<Operation *, std::vector<OwnershipNode *>, std::less<Operation *>>
      consumerGroups;
  for (OwnershipNode &node : nodes) {
    if (node.path != path) {
      continue;
    }
    if (!node.produced.empty()) {
      if (node.operation->getParentRegion() != path) {
        return std::nullopt;
      }
      items.push_back(
          {true, graph.getNodes()[node.id].order, &node, nullptr, {}});
      continue;
    }
    Operation *anchor = getPathAnchor(node.operation, path);
    if (!anchor || (anchor != node.operation && !isa<scf::IfOp>(anchor))) {
      return std::nullopt;
    }
    consumerGroups[anchor].push_back(&node);
  }
  for (auto &[anchor, consumers] : consumerGroups) {
    if (!validateConsumerGroup(anchor, consumers)) {
      return std::nullopt;
    }
    const SlotBundle &bundle = consumers.front()->consumed;
    if (llvm::any_of(consumers, [&](OwnershipNode *consumer) {
          return consumer->consumed != bundle;
        })) {
      return std::nullopt;
    }
    const auto first = llvm::min_element(
        consumers, [&](OwnershipNode *left, OwnershipNode *right) {
          return graph.getNodes()[left->id].order <
                 graph.getNodes()[right->id].order;
        });
    items.push_back({false, graph.getNodes()[(*first)->id].order, nullptr,
                     anchor, consumers});
  }
  llvm::stable_sort(items, [](const PathItem &left, const PathItem &right) {
    return left.order < right.order;
  });

  const std::optional<SyncCoverScopeId> pathScope = findScope(program, path);
  if (!pathScope) {
    return std::nullopt;
  }
  CanonicalSyncOwnershipPath result;
  result.scope = *pathScope;
  std::vector<OwnershipNode *> producers;
  for (PathItem &item : items) {
    if (item.producer) {
      producers.push_back(item.node);
      continue;
    }
    if (producers.empty()) {
      return std::nullopt;
    }
    SlotBundle produced;
    for (OwnershipNode *producer : producers) {
      produced.insert(produced.end(), producer->produced.begin(),
                      producer->produced.end());
    }
    llvm::sort(produced);
    const SlotBundle consumed = item.consumers.front()->consumed;
    if (produced.size() != spec.requiredSpaces || produced != consumed) {
      return std::nullopt;
    }
    auto lane = bundleLanes.find(produced);
    if (lane == bundleLanes.end()) {
      if (!allowNewBundles) {
        return std::nullopt;
      }
      lane = bundleLanes.emplace(produced, bundleLanes.size()).first;
    }
    llvm::sort(producers, [&](OwnershipNode *left, OwnershipNode *right) {
      return graph.getNodes()[left->id].order <
             graph.getNodes()[right->id].order;
    });
    llvm::sort(item.consumers, [&](OwnershipNode *left, OwnershipNode *right) {
      return graph.getNodes()[left->id].order <
             graph.getNodes()[right->id].order;
    });
    CanonicalSyncOwnershipUse use;
    use.lane = lane->second;
    use.producerLane = use.lane;
    llvm::transform(producers, std::back_inserter(use.producers),
                    [](OwnershipNode *node) { return node->id; });
    llvm::transform(item.consumers, std::back_inserter(use.consumers),
                    [](OwnershipNode *node) { return node->id; });
    use.writeAcquire = {SyncCoverAnchorKind::BeforeNode, producers.front()->id};
    use.ready = {SyncCoverAnchorKind::AfterNode, producers.back()->id};
    use.readAcquire = {SyncCoverAnchorKind::BeforeNode,
                       item.consumers.front()->id};
    use.release = {SyncCoverAnchorKind::AfterNode, item.consumers.back()->id};
    result.uses.push_back(std::move(use));
    producers.clear();
  }
  return producers.empty() && !result.uses.empty()
             ? std::optional<CanonicalSyncOwnershipPath>(std::move(result))
             : std::nullopt;
}

std::optional<CanonicalSyncOwnershipCycle> recognizeL0Operand(
    const CanonicalSyncProgram &program, SyncCoverScopeId recurrenceScope,
    const OwnershipSpec &spec,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode) {
  std::optional<std::vector<OwnershipNode>> nodes =
      collectOwnershipNodes(program, recurrenceScope, spec, accessesByNode);
  if (!nodes) {
    return std::nullopt;
  }
  Operation *loop = program.getScopeBindings()[recurrenceScope].owner;
  Operation *commonChild = nodes->front().topLevel;
  const bool oneChild = llvm::all_of(*nodes, [&](const OwnershipNode &node) {
    return node.topLevel == commonChild;
  });
  scf::IfOp branch =
      oneChild ? dyn_cast_or_null<scf::IfOp>(commonChild) : scf::IfOp{};
  std::vector<Region *> paths;
  if (branch) {
    if (branch.getElseRegion().empty()) {
      return std::nullopt;
    }
    paths = {&branch.getThenRegion(), &branch.getElseRegion()};
    for (OwnershipNode &node : *nodes) {
      node.path = getPathRegion(node.operation, branch);
    }
  } else {
    paths = {&cast<scf::ForOp>(loop).getRegion()};
    for (OwnershipNode &node : *nodes) {
      node.path = paths.front();
    }
  }

  CanonicalSyncOwnershipCycle cycle;
  cycle.kind = spec.kind;
  cycle.recurrenceScope = recurrenceScope;
  cycle.producerResource = spec.producerResource;
  cycle.consumerResource = spec.consumerResource;
  std::map<SlotBundle, unsigned> lanes;
  for (auto [index, path] : llvm::enumerate(paths)) {
    std::optional<CanonicalSyncOwnershipPath> parsed =
        parseOwnershipPath(program, path, *nodes, spec, lanes, index == 0);
    if (!parsed) {
      return std::nullopt;
    }
    cycle.paths.push_back(std::move(*parsed));
  }
  if (lanes.size() < spec.minimumLanes) {
    return std::nullopt;
  }
  cycle.lanes.resize(lanes.size());
  for (const auto &[slots, laneId] : lanes) {
    cycle.lanes[laneId] = {laneId, slots};
  }
  if (!slotsAreDisjoint(cycle.lanes)) {
    return std::nullopt;
  }
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    std::set<unsigned> used;
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      used.insert(use.lane);
    }
    if (used.size() != cycle.lanes.size()) {
      return std::nullopt;
    }
  }
  return cycle;
}

std::optional<std::vector<OwnershipNode>> collectHierarchicalNodes(
    const CanonicalSyncProgram &program, SyncCoverScopeId recurrenceScope,
    const HierarchicalOwnershipSpec &spec,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode) {
  const SyncCoverGraph &graph = program.getGraph();
  if (recurrenceScope >= program.getScopeBindings().size()) {
    return std::nullopt;
  }
  Operation *loop = program.getScopeBindings()[recurrenceScope].owner;
  if (!isa_and_nonnull<scf::ForOp>(loop)) {
    return std::nullopt;
  }
  std::vector<OwnershipNode> result;
  for (const SyncCoverNode &node : graph.getNodes()) {
    if (!graph.scopeContains(recurrenceScope, node.scope)) {
      continue;
    }
    std::vector<const SyncCoverStorageAccess *> accesses;
    for (const SyncCoverStorageAccess *access : accessesByNode[node.id]) {
      const std::optional<AddressSpace> space =
          getAccessSpace(program, *access);
      if (space && *space == spec.space) {
        if (!access->exactPhysical) {
          return std::nullopt;
        }
        accesses.push_back(access);
      }
    }
    if (accesses.empty()) {
      continue;
    }
    if (accesses.size() != 1) {
      return std::nullopt;
    }

    OwnershipNode ownership;
    ownership.id = node.id;
    ownership.operation = program.getNodeBindings()[node.id].operation;
    ownership.topLevel = getTopLevelChild(ownership.operation, loop);
    if (!ownership.operation || !ownership.topLevel) {
      return std::nullopt;
    }
    const SyncCoverStorageAccess &access = *accesses.front();
    CanonicalSyncOwnershipSlot slot{access.domain, access.extent};
    if (node.resource == spec.producerResource) {
      const bool writes = syncCoverStorageModeWrites(access.mode);
      const bool reads = syncCoverStorageModeReads(access.mode);
      if (!writes || (reads && !spec.allowProducerReads)) {
        return std::nullopt;
      }
      ownership.produced.push_back(slot);
    } else if (node.resource == spec.consumerResource) {
      if (access.mode != SyncCoverStorageAccessMode::Read) {
        return std::nullopt;
      }
      ownership.consumed.push_back(slot);
    } else {
      return std::nullopt;
    }
    result.push_back(std::move(ownership));
  }
  return result.empty()
             ? std::nullopt
             : std::optional<std::vector<OwnershipNode>>(std::move(result));
}

bool hierarchicalSlotsAreDisjoint(
    const std::map<CanonicalSyncOwnershipSlot, HierarchicalSlotGroup> &groups) {
  for (auto first = groups.begin(); first != groups.end(); ++first) {
    for (auto second = std::next(first); second != groups.end(); ++second) {
      if (first->first.domain == second->first.domain &&
          intervalOverlaps(first->first.extent, second->first.extent)) {
        return false;
      }
    }
  }
  return true;
}

std::optional<CanonicalSyncOwnershipCycle> recognizeHierarchicalOwnership(
    const CanonicalSyncProgram &program, SyncCoverScopeId recurrenceScope,
    const HierarchicalOwnershipSpec &spec,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode) {
  std::optional<std::vector<OwnershipNode>> nodes =
      collectHierarchicalNodes(program, recurrenceScope, spec, accessesByNode);
  if (!nodes) {
    return std::nullopt;
  }
  auto loop = dyn_cast_or_null<scf::ForOp>(
      program.getScopeBindings()[recurrenceScope].owner);
  if (!loop) {
    return std::nullopt;
  }

  std::map<CanonicalSyncOwnershipSlot, HierarchicalSlotGroup> groups;
  for (OwnershipNode &node : *nodes) {
    const CanonicalSyncOwnershipSlot &slot =
        !node.produced.empty() ? node.produced.front() : node.consumed.front();
    HierarchicalSlotGroup &group = groups[slot];
    group.slot = slot;
    (!node.produced.empty() ? group.producers : group.consumers)
        .push_back(&node);
  }
  if (groups.size() < spec.minimumLanes ||
      !hierarchicalSlotsAreDisjoint(groups)) {
    return std::nullopt;
  }

  std::vector<HierarchicalSlotGroup *> ordered;
  ordered.reserve(groups.size());
  for (auto &[slot, group] : groups) {
    (void)slot;
    const bool validProducers =
        spec.kind == CanonicalSyncOwnershipKind::L1Tile
            ? group.producers.size() == 1 &&
                  group.producers.front()->operation->getParentRegion() ==
                      &loop.getRegion()
            : !group.producers.empty();
    if (!validProducers || group.consumers.empty()) {
      return std::nullopt;
    }
    Operation *producerAnchor = getCommonTopLevel(group.producers);
    Operation *consumerAnchor = getCommonTopLevel(group.consumers);
    const bool orderedInOneBlock =
        consumerAnchor && producerAnchor != consumerAnchor &&
        producerAnchor->getBlock() == consumerAnchor->getBlock() &&
        producerAnchor->isBeforeInBlock(consumerAnchor);
    if (!orderedInOneBlock || !getBoundaryAnchors(program, producerAnchor) ||
        !getBoundaryAnchors(program, consumerAnchor)) {
      return std::nullopt;
    }
    ordered.push_back(&group);
  }
  llvm::stable_sort(ordered, [&](const HierarchicalSlotGroup *left,
                                 const HierarchicalSlotGroup *right) {
    return program.getGraph().getNodes()[left->producers.front()->id].order <
           program.getGraph().getNodes()[right->producers.front()->id].order;
  });

  CanonicalSyncOwnershipCycle cycle;
  cycle.kind = spec.kind;
  cycle.recurrenceScope = recurrenceScope;
  cycle.producerResource = spec.producerResource;
  cycle.consumerResource = spec.consumerResource;
  CanonicalSyncOwnershipPath path;
  path.scope = recurrenceScope;
  for (auto [lane, group] : llvm::enumerate(ordered)) {
    Operation *producerAnchor = getCommonTopLevel(group->producers);
    Operation *consumerAnchor = getCommonTopLevel(group->consumers);
    const auto producerBounds = getBoundaryAnchors(program, producerAnchor);
    const auto consumerBounds = getBoundaryAnchors(program, consumerAnchor);
    if (!producerBounds || !consumerBounds) {
      return std::nullopt;
    }
    CanonicalSyncOwnershipLane ownershipLane;
    ownershipLane.id = lane;
    ownershipLane.slots.push_back(group->slot);
    cycle.lanes.push_back(std::move(ownershipLane));

    CanonicalSyncOwnershipUse use;
    use.lane = lane;
    use.producerLane = lane;
    llvm::sort(group->producers,
               [&](OwnershipNode *left, OwnershipNode *right) {
                 return program.getGraph().getNodes()[left->id].order <
                        program.getGraph().getNodes()[right->id].order;
               });
    llvm::transform(group->producers, std::back_inserter(use.producers),
                    [](OwnershipNode *node) { return node->id; });
    llvm::sort(group->consumers,
               [&](OwnershipNode *left, OwnershipNode *right) {
                 return program.getGraph().getNodes()[left->id].order <
                        program.getGraph().getNodes()[right->id].order;
               });
    llvm::transform(group->consumers, std::back_inserter(use.consumers),
                    [](OwnershipNode *node) { return node->id; });
    use.writeAcquire = producerBounds->first;
    use.ready = producerBounds->second;
    use.readAcquire = consumerBounds->first;
    use.release = consumerBounds->second;
    path.uses.push_back(std::move(use));
  }
  cycle.paths.push_back(std::move(path));
  return cycle;
}

bool paritySlotsAreDisjoint(
    const std::map<CanonicalSyncOwnershipSlot, ParitySlotGroup> &groups) {
  for (auto first = groups.begin(); first != groups.end(); ++first) {
    for (auto second = std::next(first); second != groups.end(); ++second) {
      if (first->first.domain == second->first.domain &&
          intervalOverlaps(first->first.extent, second->first.extent)) {
        return false;
      }
    }
  }
  return true;
}

bool matchNonNegativeConstant(Value value, std::uint64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) &&
         constant.isNonNegative() && constant.getZExtValue() == expected;
}

bool matchAlternatingParityBranch(scf::ForOp loop, scf::IfOp branch) {
  if (!branch || branch.getElseRegion().empty() ||
      !matchNonNegativeConstant(loop.getLowerBound(), 0) ||
      !matchNonNegativeConstant(loop.getStep(), 1)) {
    return false;
  }
  auto compare = branch.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::eq) {
    return false;
  }
  Value remainderValue = compare.getLhs();
  Value zeroValue = compare.getRhs();
  if (!matchNonNegativeConstant(zeroValue, 0)) {
    remainderValue = compare.getRhs();
    zeroValue = compare.getLhs();
  }
  auto remainder = remainderValue.getDefiningOp<arith::RemSIOp>();
  return matchNonNegativeConstant(zeroValue, 0) && remainder &&
         remainder.getLhs() == loop.getInductionVar() &&
         matchNonNegativeConstant(remainder.getRhs(), 2);
}

bool matchNextIteration(Value value, scf::ForOp loop) {
  auto add = value.getDefiningOp<arith::AddIOp>();
  return add && ((add.getLhs() == loop.getInductionVar() &&
                  matchNonNegativeConstant(add.getRhs(), 1)) ||
                 (add.getRhs() == loop.getInductionVar() &&
                  matchNonNegativeConstant(add.getLhs(), 1)));
}

bool hasContinuationGuard(Operation *operation, scf::ForOp loop, Region *path) {
  auto guard = dyn_cast_or_null<scf::IfOp>(operation->getParentOp());
  const bool direct = guard && path && guard->getParentRegion() == path &&
                      operation->getParentRegion() == &guard.getThenRegion() &&
                      guard.getElseRegion().empty();
  if (!direct) {
    return false;
  }
  auto compare = guard.getCondition().getDefiningOp<arith::CmpIOp>();
  return compare && compare.getPredicate() == arith::CmpIPredicate::slt &&
         compare.getRhs() == loop.getUpperBound() &&
         matchNextIteration(compare.getLhs(), loop);
}

bool executeDirectlyIn(ArrayRef<OwnershipNode *> nodes, Region *region) {
  return llvm::all_of(nodes, [&](const OwnershipNode *node) {
    return node->operation->getParentRegion() == region;
  });
}

OwnershipNode *firstByOrder(ArrayRef<OwnershipNode *> nodes,
                            const SyncCoverGraph &graph) {
  return *llvm::min_element(nodes, [&](const OwnershipNode *left,
                                       const OwnershipNode *right) {
    return graph.getNodes()[left->id].order < graph.getNodes()[right->id].order;
  });
}

OwnershipNode *lastByOrder(ArrayRef<OwnershipNode *> nodes,
                           const SyncCoverGraph &graph) {
  return *llvm::max_element(nodes, [&](const OwnershipNode *left,
                                       const OwnershipNode *right) {
    return graph.getNodes()[left->id].order < graph.getNodes()[right->id].order;
  });
}

CanonicalSyncOwnershipUse makeParityUse(unsigned lane, unsigned producerLane,
                                        ArrayRef<OwnershipNode *> producers,
                                        ArrayRef<OwnershipNode *> consumers,
                                        const SyncCoverGraph &graph) {
  CanonicalSyncOwnershipUse use;
  use.lane = lane;
  use.producerLane = producerLane;
  llvm::transform(producers, std::back_inserter(use.producers),
                  [](const OwnershipNode *node) { return node->id; });
  llvm::transform(consumers, std::back_inserter(use.consumers),
                  [](const OwnershipNode *node) { return node->id; });
  OwnershipNode *firstProducer = firstByOrder(producers, graph);
  OwnershipNode *lastProducer = lastByOrder(producers, graph);
  OwnershipNode *firstConsumer = firstByOrder(consumers, graph);
  OwnershipNode *lastConsumer = lastByOrder(consumers, graph);
  use.writeAcquire = {SyncCoverAnchorKind::BeforeNode, firstProducer->id};
  use.ready = {SyncCoverAnchorKind::AfterNode, lastProducer->id};
  use.readAcquire = {SyncCoverAnchorKind::BeforeNode, firstConsumer->id};
  use.release = {SyncCoverAnchorKind::AfterNode, lastConsumer->id};
  return use;
}

std::optional<SyncCoverNodeId> findInitialProducer(
    const CanonicalSyncProgram &program, scf::ForOp loop,
    const CanonicalSyncOwnershipSlot &slot,
    ArrayRef<CanonicalSyncOwnershipSlot> managedSlots,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode) {
  std::optional<SyncCoverNodeId> result;
  for (const SyncCoverNode &node : program.getGraph().getNodes()) {
    Operation *operation = program.getNodeBindings()[node.id].operation;
    Operation *topLevel = getTopLevelInBlock(operation, loop->getBlock());
    if (!topLevel || topLevel == loop || !topLevel->isBeforeInBlock(loop)) {
      continue;
    }
    bool touchesManaged = false;
    bool initialWrite = false;
    bool conflictingAccess = false;
    for (const SyncCoverStorageAccess *access : accessesByNode[node.id]) {
      if (access->domain >= program.getStorageSpaces().size() ||
          program.getStorageSpaces()[access->domain] != AddressSpace::MAT) {
        continue;
      }
      if (!access->exactPhysical) {
        return std::nullopt;
      }
      const CanonicalSyncOwnershipSlot candidate{access->domain,
                                                 access->extent};
      const bool overlapsManaged = llvm::any_of(
          managedSlots, [&](const CanonicalSyncOwnershipSlot &managed) {
            return candidate.domain == managed.domain &&
                   intervalOverlaps(candidate.extent, managed.extent);
          });
      if (!overlapsManaged) {
        continue;
      }
      touchesManaged = true;
      const bool valid = operation == topLevel &&
                         node.resource == static_cast<std::uint32_t>(
                                              PipelineType::PIPE_MTE2) &&
                         access->mode == SyncCoverStorageAccessMode::Write &&
                         candidate == slot;
      conflictingAccess |= initialWrite || !valid;
      initialWrite |= valid;
    }
    if (!touchesManaged) {
      continue;
    }
    if (!initialWrite || conflictingAccess || result) {
      return std::nullopt;
    }
    result = node.id;
  }
  return result;
}

std::vector<CanonicalSyncOwnershipCycle> recognizeParityL1(
    const CanonicalSyncProgram &program, SyncCoverScopeId recurrenceScope,
    const HierarchicalOwnershipSpec &spec,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode) {
  std::vector<CanonicalSyncOwnershipCycle> result;
  std::optional<std::vector<OwnershipNode>> nodes =
      collectHierarchicalNodes(program, recurrenceScope, spec, accessesByNode);
  auto loop = dyn_cast_or_null<scf::ForOp>(
      program.getScopeBindings()[recurrenceScope].owner);
  if (!nodes || !loop) {
    return result;
  }
  Operation *commonChild = nodes->front().topLevel;
  const bool oneChild = llvm::all_of(*nodes, [&](const OwnershipNode &node) {
    return node.topLevel == commonChild;
  });
  scf::IfOp branch =
      oneChild ? dyn_cast_or_null<scf::IfOp>(commonChild) : scf::IfOp{};
  if (!matchAlternatingParityBranch(loop, branch)) {
    return result;
  }
  Region *paths[] = {&branch.getThenRegion(), &branch.getElseRegion()};
  std::map<CanonicalSyncOwnershipSlot, ParitySlotGroup> groups;
  for (OwnershipNode &node : *nodes) {
    node.path = getPathRegion(node.operation, branch);
    const unsigned pathIndex = node.path == paths[0]   ? 0
                               : node.path == paths[1] ? 1
                                                       : 2;
    if (pathIndex == 2) {
      return {};
    }
    const CanonicalSyncOwnershipSlot &slot =
        !node.produced.empty() ? node.produced.front() : node.consumed.front();
    ParitySlotGroup &group = groups[slot];
    group.slot = slot;
    (!node.produced.empty() ? group.producers[pathIndex]
                            : group.consumers[pathIndex])
        .push_back(&node);
  }
  if (!paritySlotsAreDisjoint(groups)) {
    return {};
  }

  std::vector<ParitySlotGroup *> stableGroups;
  std::vector<ParitySlotGroup *> alternatingGroups;
  const SyncCoverGraph &graph = program.getGraph();
  for (auto &[slot, group] : groups) {
    (void)slot;
    const bool stable = llvm::all_of(llvm::seq<unsigned>(0, 2), [&](unsigned
                                                                        path) {
      return group.producers[path].size() == 1 &&
             !group.consumers[path].empty() &&
             executeDirectlyIn(group.producers[path], paths[path]) &&
             executeDirectlyIn(group.consumers[path], paths[path]) &&
             graph.getNodes()[group.producers[path].front()->id].order <
                 graph
                     .getNodes()[firstByOrder(group.consumers[path], graph)->id]
                     .order;
    });
    const bool alternating =
        (group.producers[0].size() == 1 && group.consumers[0].empty() &&
         group.producers[1].empty() && group.consumers[1].size() == 1 &&
         executeDirectlyIn(group.consumers[1], paths[1])) ||
        (group.producers[1].size() == 1 && group.consumers[1].empty() &&
         group.producers[0].empty() && group.consumers[0].size() == 1 &&
         executeDirectlyIn(group.consumers[0], paths[0]));
    if (stable) {
      stableGroups.push_back(&group);
    } else if (alternating) {
      alternatingGroups.push_back(&group);
    } else {
      return {};
    }
  }

  if (stableGroups.size() >= spec.minimumLanes) {
    CanonicalSyncOwnershipCycle stable;
    stable.kind = CanonicalSyncOwnershipKind::L1Tile;
    stable.recurrenceScope = recurrenceScope;
    stable.producerResource = spec.producerResource;
    stable.consumerResource = spec.consumerResource;
    stable.paths.resize(2);
    for (unsigned path = 0; path < 2; ++path) {
      const std::optional<SyncCoverScopeId> scope =
          findScope(program, paths[path]);
      if (!scope) {
        return {};
      }
      stable.paths[path].scope = *scope;
    }
    for (auto [lane, group] : llvm::enumerate(stableGroups)) {
      stable.lanes.push_back({static_cast<unsigned>(lane), {group->slot}});
      for (unsigned path = 0; path < 2; ++path) {
        stable.paths[path].uses.push_back(makeParityUse(
            lane, lane, group->producers[path], group->consumers[path], graph));
      }
    }
    result.push_back(std::move(stable));
  }

  if (alternatingGroups.size() != 2) {
    return result;
  }
  CanonicalSyncOwnershipCycle prefetch;
  prefetch.kind = CanonicalSyncOwnershipKind::L1Tile;
  prefetch.protocol = CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch;
  prefetch.recurrenceScope = recurrenceScope;
  prefetch.producerResource = spec.producerResource;
  prefetch.consumerResource = spec.consumerResource;
  prefetch.paths.resize(2);
  for (unsigned path = 0; path < 2; ++path) {
    const std::optional<SyncCoverScopeId> scope =
        findScope(program, paths[path]);
    if (!scope) {
      return result;
    }
    prefetch.paths[path].scope = *scope;
  }
  for (auto [lane, group] : llvm::enumerate(alternatingGroups)) {
    prefetch.lanes.push_back({static_cast<unsigned>(lane), {group->slot}});
  }
  for (unsigned path = 0; path < 2; ++path) {
    ParitySlotGroup *consumerGroup = nullptr;
    ParitySlotGroup *producerGroup = nullptr;
    unsigned consumerLane = 0;
    unsigned producerLane = 0;
    for (auto [lane, group] : llvm::enumerate(alternatingGroups)) {
      if (!group->consumers[path].empty()) {
        consumerGroup = group;
        consumerLane = lane;
      }
      if (!group->producers[path].empty()) {
        producerGroup = group;
        producerLane = lane;
      }
    }
    if (!consumerGroup || !producerGroup || consumerLane == producerLane ||
        !hasContinuationGuard(producerGroup->producers[path].front()->operation,
                              loop, paths[path])) {
      return result;
    }
    prefetch.paths[path].uses.push_back(makeParityUse(
        consumerLane, producerLane, producerGroup->producers[path],
        consumerGroup->consumers[path], graph));
  }
  const CanonicalSyncOwnershipUse &first = prefetch.paths.front().uses.front();
  std::vector<CanonicalSyncOwnershipSlot> managedSlots;
  for (const CanonicalSyncOwnershipLane &lane : prefetch.lanes) {
    managedSlots.insert(managedSlots.end(), lane.slots.begin(),
                        lane.slots.end());
  }
  const std::optional<SyncCoverNodeId> initial = findInitialProducer(
      program, loop, prefetch.lanes[first.lane].slots.front(), managedSlots,
      accessesByNode);
  if (!initial || hasInterveningResourceNode(program, *initial, loop,
                                             prefetch.producerResource)) {
    return result;
  }
  prefetch.initialProducers.push_back(*initial);
  prefetch.initialWriteAcquire = {SyncCoverAnchorKind::BeforeNode, *initial};
  prefetch.initialReady = {SyncCoverAnchorKind::ScopeEntry, 0, recurrenceScope};
  prefetch.initialReadyLane = first.lane;
  prefetch.initiallyFreeLanes.push_back(first.producerLane);
  result.push_back(std::move(prefetch));
  return result;
}

bool slotsMatch(
    const CanonicalSyncProgram &program,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode,
    SyncCoverNodeId node, ArrayRef<CanonicalSyncOwnershipSlot> expected,
    AddressSpace firstSpace, AddressSpace secondSpace,
    SyncCoverStorageAccessMode firstMode,
    SyncCoverStorageAccessMode secondMode) {
  std::vector<CanonicalSyncOwnershipSlot> actual;
  if (node >= accessesByNode.size()) {
    return false;
  }
  for (const SyncCoverStorageAccess *access : accessesByNode[node]) {
    if (access->domain >= program.getStorageSpaces().size()) {
      continue;
    }
    const AddressSpace space = program.getStorageSpaces()[access->domain];
    if (space != firstSpace && space != secondSpace) {
      continue;
    }
    const SyncCoverStorageAccessMode expectedMode =
        space == firstSpace ? firstMode : secondMode;
    if (!access->exactPhysical || access->mode != expectedMode) {
      return false;
    }
    actual.push_back({access->domain, access->extent});
  }
  llvm::sort(actual);
  return ArrayRef<CanonicalSyncOwnershipSlot>(actual) == expected;
}

bool slotsMatchSpace(
    const CanonicalSyncProgram &program,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode,
    SyncCoverNodeId node, ArrayRef<CanonicalSyncOwnershipSlot> expected,
    AddressSpace space, SyncCoverStorageAccessMode mode) {
  std::vector<CanonicalSyncOwnershipSlot> actual;
  if (node >= accessesByNode.size()) {
    return false;
  }
  for (const SyncCoverStorageAccess *access : accessesByNode[node]) {
    if (access->domain >= program.getStorageSpaces().size() ||
        program.getStorageSpaces()[access->domain] != space) {
      continue;
    }
    if (!access->exactPhysical || access->mode != mode) {
      return false;
    }
    actual.push_back({access->domain, access->extent});
  }
  llvm::sort(actual);
  return ArrayRef<CanonicalSyncOwnershipSlot>(actual) == expected;
}

bool slotsMatchWritingSpace(
    const CanonicalSyncProgram &program,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode,
    SyncCoverNodeId node, ArrayRef<CanonicalSyncOwnershipSlot> expected,
    AddressSpace space) {
  std::vector<CanonicalSyncOwnershipSlot> actual;
  if (node >= accessesByNode.size()) {
    return false;
  }
  for (const SyncCoverStorageAccess *access : accessesByNode[node]) {
    if (access->domain >= program.getStorageSpaces().size() ||
        program.getStorageSpaces()[access->domain] != space) {
      continue;
    }
    if (!access->exactPhysical || !syncCoverStorageModeWrites(access->mode)) {
      return false;
    }
    actual.push_back({access->domain, access->extent});
  }
  llvm::sort(actual);
  return ArrayRef<CanonicalSyncOwnershipSlot>(actual) == expected;
}

bool consumersCoverPath(const SyncCoverGraph &graph,
                        const CanonicalSyncOwnershipPath &path,
                        ArrayRef<SyncCoverNodeId> consumers) {
  if (path.scope >= graph.getScopes().size() || consumers.empty()) {
    return false;
  }
  const SyncCoverGuard &pathGuard = graph.getScopes()[path.scope].guard;
  std::optional<SyncCoverControlId> alternativeControl;
  std::set<unsigned> alternatives;
  for (SyncCoverNodeId consumer : consumers) {
    if (consumer >= graph.getNodes().size()) {
      return false;
    }
    const SyncCoverGuard &guard = graph.getNodes()[consumer].guard;
    if (!syncCoverGuardImplies(guard, pathGuard)) {
      return false;
    }
    std::vector<SyncCoverGuardLiteral> residual;
    std::set_difference(guard.literals.begin(), guard.literals.end(),
                        pathGuard.literals.begin(), pathGuard.literals.end(),
                        std::back_inserter(residual));
    if (residual.empty()) {
      return consumers.size() == 1 &&
             graph.getNodes()[consumer].scope == path.scope;
    }
    if (residual.size() != 1) {
      return false;
    }
    const SyncCoverScope &consumerScope =
        graph.getScopes()[graph.getNodes()[consumer].scope];
    const bool directAlternative =
        consumerScope.parent == path.scope &&
        consumerScope.mustExecuteWithinParent && !consumerScope.isLoop &&
        consumerScope.guard.literals == guard.literals;
    if (!directAlternative) {
      return false;
    }
    if (!alternativeControl) {
      alternativeControl = residual.front().control;
    }
    if (*alternativeControl != residual.front().control ||
        !alternatives.insert(residual.front().alternative).second) {
      return false;
    }
  }
  if (!alternativeControl ||
      *alternativeControl >= graph.getControls().size()) {
    return false;
  }
  const SyncCoverControl &control = graph.getControls()[*alternativeControl];
  return control.scope == path.scope &&
         alternatives.size() == control.alternatives &&
         *alternatives.begin() == 0 &&
         *alternatives.rbegin() + 1 == control.alternatives;
}

bool pathsCoverRecurrence(const SyncCoverGraph &graph,
                          SyncCoverScopeId recurrenceScope,
                          ArrayRef<CanonicalSyncOwnershipPath> paths) {
  if (recurrenceScope >= graph.getScopes().size() || paths.empty()) {
    return false;
  }
  const SyncCoverGuard &recurrenceGuard =
      graph.getScopes()[recurrenceScope].guard;
  std::optional<SyncCoverControlId> alternativeControl;
  std::set<unsigned> alternatives;
  for (const CanonicalSyncOwnershipPath &path : paths) {
    if (path.scope >= graph.getScopes().size()) {
      return false;
    }
    const SyncCoverGuard &guard = graph.getScopes()[path.scope].guard;
    if (!syncCoverGuardImplies(guard, recurrenceGuard)) {
      return false;
    }
    std::vector<SyncCoverGuardLiteral> residual;
    std::set_difference(guard.literals.begin(), guard.literals.end(),
                        recurrenceGuard.literals.begin(),
                        recurrenceGuard.literals.end(),
                        std::back_inserter(residual));
    if (residual.empty()) {
      return paths.size() == 1 && path.scope == recurrenceScope;
    }
    if (residual.size() != 1) {
      return false;
    }
    const SyncCoverScope &pathScope = graph.getScopes()[path.scope];
    const bool directAlternative = pathScope.parent == recurrenceScope &&
                                   pathScope.mustExecuteWithinParent &&
                                   !pathScope.isLoop &&
                                   pathScope.guard.literals == guard.literals;
    if (!directAlternative) {
      return false;
    }
    if (!alternativeControl) {
      alternativeControl = residual.front().control;
    }
    if (*alternativeControl != residual.front().control ||
        !alternatives.insert(residual.front().alternative).second) {
      return false;
    }
  }
  if (!alternativeControl ||
      *alternativeControl >= graph.getControls().size()) {
    return false;
  }
  const SyncCoverControl &control = graph.getControls()[*alternativeControl];
  return control.scope == recurrenceScope &&
         alternatives.size() == control.alternatives &&
         *alternatives.begin() == 0 &&
         *alternatives.rbegin() + 1 == control.alternatives;
}

bool anchorsEqual(const SyncCoverAnchor &left, const SyncCoverAnchor &right) {
  return std::tie(left.kind, left.node, left.scope, left.position) ==
         std::tie(right.kind, right.node, right.scope, right.position);
}

bool accessOverlapsCycle(const CanonicalSyncOwnershipCycle &cycle,
                         const SyncCoverStorageAccess &access) {
  return llvm::any_of(cycle.lanes, [&](const auto &lane) {
    return llvm::any_of(lane.slots, [&](const auto &slot) {
      return access.domain == slot.domain &&
             intervalOverlaps(access.extent, slot.extent);
    });
  });
}

bool verifyHierarchicalUse(const CanonicalSyncProgram &program,
                           const CanonicalSyncOwnershipCycle &cycle,
                           const CanonicalSyncOwnershipPath &path,
                           const CanonicalSyncOwnershipUse &use,
                           std::set<SyncCoverNodeId> &represented) {
  const SyncCoverGraph &graph = program.getGraph();
  const bool accumulator =
      cycle.kind == CanonicalSyncOwnershipKind::L0Accumulator;
  if (use.lane >= cycle.lanes.size() || use.producerLane != use.lane ||
      use.producers.empty() || use.consumers.empty() ||
      (accumulator && use.consumers.size() != 1) ||
      (!accumulator && use.producers.size() != 1) ||
      !graph.scopeContains(cycle.recurrenceScope, path.scope) ||
      cycle.recurrenceScope >= program.getScopeBindings().size()) {
    return false;
  }
  Operation *loop = program.getScopeBindings()[cycle.recurrenceScope].owner;
  Region *pathRegion = program.getScopeBindings()[path.scope].region;
  if (!loop || !pathRegion) {
    return false;
  }
  std::vector<OwnershipNode> producerNodes;
  producerNodes.reserve(use.producers.size());
  for (SyncCoverNodeId producer : use.producers) {
    if (producer >= graph.getNodes().size() ||
        graph.getNodes()[producer].resource != cycle.producerResource ||
        !graph.scopeContains(path.scope, graph.getNodes()[producer].scope)) {
      return false;
    }
    Operation *operation = program.getNodeBindings()[producer].operation;
    Operation *topLevel = getPathAnchor(operation, pathRegion);
    if (!operation || !topLevel) {
      return false;
    }
    producerNodes.push_back({producer, operation, topLevel, nullptr, {}, {}});
  }
  std::vector<OwnershipNode *> producerPointers;
  llvm::transform(producerNodes, std::back_inserter(producerPointers),
                  [](OwnershipNode &node) { return &node; });
  Operation *producerAnchor = getCommonTopLevel(producerPointers);
  if (!producerAnchor ||
      (!accumulator && producerAnchor != producerNodes.front().operation)) {
    return false;
  }
  std::vector<OwnershipNode> consumerNodes;
  consumerNodes.reserve(use.consumers.size());
  for (SyncCoverNodeId consumer : use.consumers) {
    if (consumer >= graph.getNodes().size() ||
        graph.getNodes()[consumer].resource != cycle.consumerResource ||
        !graph.scopeContains(path.scope, graph.getNodes()[consumer].scope)) {
      return false;
    }
    Operation *operation = program.getNodeBindings()[consumer].operation;
    Operation *topLevel = getPathAnchor(operation, pathRegion);
    if (!operation || !topLevel) {
      return false;
    }
    OwnershipNode node;
    node.id = consumer;
    node.operation = operation;
    node.topLevel = topLevel;
    consumerNodes.push_back(std::move(node));
  }
  std::vector<OwnershipNode *> consumerPointers;
  llvm::transform(consumerNodes, std::back_inserter(consumerPointers),
                  [](OwnershipNode &node) { return &node; });
  Operation *consumerAnchor = getCommonTopLevel(consumerPointers);
  if (accumulator && consumerAnchor != consumerNodes.front().operation) {
    return false;
  }
  const bool ordered =
      producerAnchor && consumerAnchor &&
      producerAnchor->getBlock() == consumerAnchor->getBlock() &&
      producerAnchor->isBeforeInBlock(consumerAnchor);
  const auto producerBounds = getBoundaryAnchors(program, producerAnchor);
  const auto consumerBounds = getBoundaryAnchors(program, consumerAnchor);
  if (!ordered || !producerBounds || !consumerBounds ||
      !anchorsEqual(use.writeAcquire, producerBounds->first) ||
      !anchorsEqual(use.ready, producerBounds->second) ||
      !anchorsEqual(use.readAcquire, consumerBounds->first) ||
      !anchorsEqual(use.release, consumerBounds->second)) {
    return false;
  }
  if (!llvm::all_of(use.producers, [&](SyncCoverNodeId producer) {
        return represented.insert(producer).second;
      })) {
    return false;
  }
  return llvm::all_of(use.consumers, [&](SyncCoverNodeId consumer) {
    return represented.insert(consumer).second;
  });
}

bool verifyUse(const CanonicalSyncProgram &program,
               const CanonicalSyncOwnershipCycle &cycle,
               const CanonicalSyncOwnershipPath &path,
               const CanonicalSyncOwnershipUse &use,
               std::set<SyncCoverNodeId> &represented) {
  if (cycle.kind == CanonicalSyncOwnershipKind::L1Tile ||
      cycle.kind == CanonicalSyncOwnershipKind::L0Accumulator) {
    return verifyHierarchicalUse(program, cycle, path, use, represented);
  }
  if (use.lane >= cycle.lanes.size() || use.producerLane != use.lane ||
      use.producers.empty() || use.consumers.empty()) {
    return false;
  }
  const auto resourceMatches = [&](SyncCoverNodeId node,
                                   std::uint32_t resource) {
    return node < program.getGraph().getNodes().size() &&
           program.getGraph().getNodes()[node].resource == resource &&
           program.getGraph().scopeContains(
               cycle.recurrenceScope,
               program.getGraph().getNodes()[node].scope);
  };
  if (llvm::any_of(use.producers,
                   [&](SyncCoverNodeId node) {
                     return !resourceMatches(node, cycle.producerResource) ||
                            program.getGraph().getNodes()[node].scope !=
                                path.scope;
                   }) ||
      llvm::any_of(use.consumers,
                   [&](SyncCoverNodeId node) {
                     return !resourceMatches(node, cycle.consumerResource);
                   }) ||
      !consumersCoverPath(program.getGraph(), path, use.consumers)) {
    return false;
  }
  const auto claim = [&](SyncCoverNodeId node) {
    return represented.insert(node).second;
  };
  if (!llvm::all_of(use.producers, claim) ||
      !llvm::all_of(use.consumers, claim)) {
    return false;
  }
  const auto firstProducer =
      std::min_element(use.producers.begin(), use.producers.end(),
                       [&](SyncCoverNodeId left, SyncCoverNodeId right) {
                         return program.getGraph().getNodes()[left].order <
                                program.getGraph().getNodes()[right].order;
                       });
  const SyncCoverAnchor expectedAcquire{SyncCoverAnchorKind::BeforeNode,
                                        *firstProducer, 0, 0};
  const auto lastProducer =
      std::max_element(use.producers.begin(), use.producers.end(),
                       [&](SyncCoverNodeId left, SyncCoverNodeId right) {
                         return program.getGraph().getNodes()[left].order <
                                program.getGraph().getNodes()[right].order;
                       });
  const auto firstConsumer =
      std::min_element(use.consumers.begin(), use.consumers.end(),
                       [&](SyncCoverNodeId left, SyncCoverNodeId right) {
                         return program.getGraph().getNodes()[left].order <
                                program.getGraph().getNodes()[right].order;
                       });
  const auto lastConsumer =
      std::max_element(use.consumers.begin(), use.consumers.end(),
                       [&](SyncCoverNodeId left, SyncCoverNodeId right) {
                         return program.getGraph().getNodes()[left].order <
                                program.getGraph().getNodes()[right].order;
                       });
  const SyncCoverAnchor expectedReady{SyncCoverAnchorKind::AfterNode,
                                      *lastProducer, 0, 0};
  const SyncCoverAnchor expectedReadAcquire{SyncCoverAnchorKind::BeforeNode,
                                            *firstConsumer, 0, 0};
  const SyncCoverAnchor expectedRelease{SyncCoverAnchorKind::AfterNode,
                                        *lastConsumer, 0, 0};
  if (!anchorsEqual(use.writeAcquire, expectedAcquire) ||
      !anchorsEqual(use.ready, expectedReady) ||
      !anchorsEqual(use.readAcquire, expectedReadAcquire) ||
      !anchorsEqual(use.release, expectedRelease)) {
    return false;
  }
  return true;
}

bool verifyAlternatingCycleImpl(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode,
    bool validateGraph) {
  const SyncCoverGraph &graph = program.getGraph();
  const bool basicShape =
      graph.isStructureFrozen() &&
      (!validateGraph || static_cast<bool>(graph.validate())) &&
      cycle.kind == CanonicalSyncOwnershipKind::L1Tile &&
      cycle.protocol ==
          CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch &&
      cycle.recurrenceScope < graph.getScopes().size() &&
      graph.getScopes()[cycle.recurrenceScope].isLoop &&
      cycle.lanes.size() == 2 && cycle.paths.size() == 2 &&
      cycle.initialProducers.size() == 1 &&
      cycle.initiallyFreeLanes.size() == 1 && slotsAreDisjoint(cycle.lanes) &&
      cycle.initialReadyLane < cycle.lanes.size() &&
      cycle.initiallyFreeLanes.front() < cycle.lanes.size() &&
      accessesByNode.size() == graph.getNodes().size() &&
      program.getNodeBindings().size() == graph.getNodes().size() &&
      program.getScopeBindings().size() == graph.getScopes().size() &&
      pathsCoverRecurrence(graph, cycle.recurrenceScope, cycle.paths) &&
      cycle.paths[0].uses.size() == 1 && cycle.paths[1].uses.size() == 1;
  if (!basicShape) {
    return false;
  }
  for (auto [laneIndex, lane] : llvm::enumerate(cycle.lanes)) {
    if (lane.id != laneIndex || lane.slots.size() != 1) {
      return false;
    }
  }
  auto loop = dyn_cast_or_null<scf::ForOp>(
      program.getScopeBindings()[cycle.recurrenceScope].owner);
  Region *firstRegion = program.getScopeBindings()[cycle.paths[0].scope].region;
  Region *secondRegion =
      program.getScopeBindings()[cycle.paths[1].scope].region;
  auto branch = firstRegion
                    ? dyn_cast_or_null<scf::IfOp>(firstRegion->getParentOp())
                    : scf::IfOp{};
  if (!loop || !branch || branch->getParentOp() != loop.getOperation() ||
      firstRegion != &branch.getThenRegion() ||
      secondRegion != &branch.getElseRegion() ||
      !matchAlternatingParityBranch(loop, branch)) {
    return false;
  }
  const SyncCoverGuard &recurrenceGuard =
      graph.getScopes()[cycle.recurrenceScope].guard;
  std::optional<SyncCoverControlId> pathControl;
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    const SyncCoverGuard &pathGuard = graph.getScopes()[path.scope].guard;
    std::vector<SyncCoverGuardLiteral> residual;
    std::set_difference(pathGuard.literals.begin(), pathGuard.literals.end(),
                        recurrenceGuard.literals.begin(),
                        recurrenceGuard.literals.end(),
                        std::back_inserter(residual));
    if (residual.size() != 1 ||
        (pathControl && *pathControl != residual.front().control)) {
      return false;
    }
    pathControl = residual.front().control;
  }
  const SyncCoverControl *periodicControl =
      pathControl && *pathControl < graph.getControls().size()
          ? &graph.getControls()[*pathControl]
          : nullptr;
  const bool validPhase =
      periodicControl && periodicControl->scope == cycle.recurrenceScope &&
      periodicControl->alternatives == 2 && periodicControl->phaseRelation &&
      periodicControl->phaseRelation->loopScope == cycle.recurrenceScope &&
      periodicControl->phaseRelation->initialPhase == 0 &&
      periodicControl->phaseRelation->nextPhase ==
          std::vector<std::size_t>({1, 0}) &&
      periodicControl->phaseRelation->activeAlternative ==
          std::vector<unsigned>({0, 1});
  if (!validPhase) {
    return false;
  }

  const CanonicalSyncOwnershipUse &first = cycle.paths[0].uses.front();
  const CanonicalSyncOwnershipUse &second = cycle.paths[1].uses.front();
  const bool transitions =
      first.lane == cycle.initialReadyLane &&
      first.producerLane == cycle.initiallyFreeLanes.front() &&
      first.lane != first.producerLane && second.lane == first.producerLane &&
      second.producerLane == first.lane;
  if (!transitions ||
      !anchorsEqual(cycle.initialReady, {SyncCoverAnchorKind::ScopeEntry, 0,
                                         cycle.recurrenceScope, 0})) {
    return false;
  }

  std::set<SyncCoverNodeId> represented;
  const SyncCoverNodeId initial = cycle.initialProducers.front();
  Operation *initialOperation =
      initial < program.getNodeBindings().size()
          ? program.getNodeBindings()[initial].operation
          : nullptr;
  Operation *initialTopLevel = initialOperation;
  while (initialTopLevel && initialTopLevel->getBlock() != loop->getBlock()) {
    initialTopLevel = initialTopLevel->getParentOp();
  }
  const bool validInitial =
      initial < graph.getNodes().size() && initialOperation &&
      initialTopLevel == initialOperation && initialTopLevel != loop &&
      initialTopLevel->isBeforeInBlock(loop) &&
      graph.getNodes()[initial].resource == cycle.producerResource &&
      !hasInterveningResourceNode(program, initial, loop,
                                  cycle.producerResource) &&
      slotsMatchSpace(program, accessesByNode, initial,
                      cycle.lanes[cycle.initialReadyLane].slots,
                      AddressSpace::MAT, SyncCoverStorageAccessMode::Write) &&
      anchorsEqual(cycle.initialWriteAcquire,
                   {SyncCoverAnchorKind::BeforeNode, initial, 0, 0});
  if (!validInitial || !represented.insert(initial).second) {
    return false;
  }

  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    if (path.uses.size() != 1) {
      return false;
    }
    const CanonicalSyncOwnershipUse &use = path.uses.front();
    if (use.lane >= cycle.lanes.size() ||
        use.producerLane >= cycle.lanes.size() || use.producers.size() != 1 ||
        use.consumers.size() != 1) {
      return false;
    }
    const SyncCoverNodeId producer = use.producers.front();
    Operation *producerOperation =
        producer < program.getNodeBindings().size()
            ? program.getNodeBindings()[producer].operation
            : nullptr;
    Region *pathRegion = program.getScopeBindings()[path.scope].region;
    const bool validProducer =
        producer < graph.getNodes().size() && producerOperation &&
        graph.getNodes()[producer].resource == cycle.producerResource &&
        graph.scopeContains(path.scope, graph.getNodes()[producer].scope) &&
        hasContinuationGuard(producerOperation, loop, pathRegion) &&
        slotsMatchSpace(program, accessesByNode, producer,
                        cycle.lanes[use.producerLane].slots, AddressSpace::MAT,
                        SyncCoverStorageAccessMode::Write) &&
        anchorsEqual(use.writeAcquire,
                     {SyncCoverAnchorKind::BeforeNode, producer, 0, 0}) &&
        anchorsEqual(use.ready,
                     {SyncCoverAnchorKind::AfterNode, producer, 0, 0});
    if (!validProducer || !represented.insert(producer).second) {
      return false;
    }
    std::vector<SyncCoverNodeId> orderedConsumers = use.consumers;
    llvm::sort(
        orderedConsumers, [&](SyncCoverNodeId left, SyncCoverNodeId right) {
          return graph.getNodes()[left].order < graph.getNodes()[right].order;
        });
    for (SyncCoverNodeId consumer : orderedConsumers) {
      Operation *consumerOperation =
          consumer < program.getNodeBindings().size()
              ? program.getNodeBindings()[consumer].operation
              : nullptr;
      const bool validConsumer =
          consumer < graph.getNodes().size() && consumerOperation &&
          consumerOperation->getParentRegion() == pathRegion &&
          graph.getNodes()[consumer].resource == cycle.consumerResource &&
          slotsMatchSpace(program, accessesByNode, consumer,
                          cycle.lanes[use.lane].slots, AddressSpace::MAT,
                          SyncCoverStorageAccessMode::Read);
      if (!validConsumer || !represented.insert(consumer).second) {
        return false;
      }
    }
    if (!anchorsEqual(use.readAcquire, {SyncCoverAnchorKind::BeforeNode,
                                        orderedConsumers.front(), 0, 0}) ||
        !anchorsEqual(use.release, {SyncCoverAnchorKind::AfterNode,
                                    orderedConsumers.back(), 0, 0})) {
      return false;
    }
  }

  for (const SyncCoverStorageAccess &access : graph.getStorageAccesses()) {
    Operation *operation = program.getNodeBindings()[access.node].operation;
    const bool inCycle =
        graph.scopeContains(cycle.recurrenceScope,
                            graph.getNodes()[access.node].scope) ||
        isBeforeLoop(operation, loop);
    if (inCycle && access.domain < program.getStorageSpaces().size() &&
        program.getStorageSpaces()[access.domain] == AddressSpace::MAT &&
        accessOverlapsCycle(cycle, access) &&
        represented.count(access.node) == 0) {
      return false;
    }
  }
  return true;
}

bool verifyCycleImpl(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode,
    bool validateGraph) {
  const SyncCoverGraph &graph = program.getGraph();
  if (cycle.protocol ==
      CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch) {
    return verifyAlternatingCycleImpl(program, cycle, accessesByNode,
                                      validateGraph);
  }
  if (!graph.isStructureFrozen() ||
      (validateGraph && !static_cast<bool>(graph.validate())) ||
      accessesByNode.size() != graph.getNodes().size() ||
      cycle.recurrenceScope >= graph.getScopes().size() ||
      !graph.getScopes()[cycle.recurrenceScope].isLoop || cycle.lanes.empty() ||
      cycle.paths.empty() || !slotsAreDisjoint(cycle.lanes) ||
      (cycle.kind == CanonicalSyncOwnershipKind::L0Accumulator
           ? cycle.protocol !=
                 CanonicalSyncOwnershipProtocolKind::BoundaryGuardedRoundTrip
           : cycle.protocol != CanonicalSyncOwnershipProtocolKind::RoundTrip) ||
      (cycle.kind != CanonicalSyncOwnershipKind::L0Operand &&
       cycle.kind != CanonicalSyncOwnershipKind::L1Tile &&
       cycle.kind != CanonicalSyncOwnershipKind::L0Accumulator) ||
      !pathsCoverRecurrence(graph, cycle.recurrenceScope, cycle.paths)) {
    return false;
  }
  for (auto [laneIndex, lane] : llvm::enumerate(cycle.lanes)) {
    if (lane.id != laneIndex || lane.slots.empty() ||
        !std::is_sorted(lane.slots.begin(), lane.slots.end()) ||
        std::adjacent_find(lane.slots.begin(), lane.slots.end()) !=
            lane.slots.end()) {
      return false;
    }
  }
  std::set<SyncCoverNodeId> represented;
  std::optional<std::size_t> producerCount;
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    if (!graph.scopeContains(cycle.recurrenceScope, path.scope) ||
        path.uses.empty()) {
      return false;
    }
    std::set<unsigned> lanes;
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      if (!verifyUse(program, cycle, path, use, represented)) {
        return false;
      }
      if (!producerCount) {
        producerCount = use.producers.size();
      } else if (*producerCount != use.producers.size()) {
        return false;
      }
      lanes.insert(use.lane);
    }
    if (lanes.size() != cycle.lanes.size()) {
      return false;
    }
  }
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      SlotBundle produced;
      for (SyncCoverNodeId producer : use.producers) {
        if (producer >= accessesByNode.size()) {
          return false;
        }
        for (const SyncCoverStorageAccess *access : accessesByNode[producer]) {
          if (access->domain >= program.getStorageSpaces().size()) {
            continue;
          }
          const AddressSpace space = program.getStorageSpaces()[access->domain];
          const bool managed =
              cycle.kind == CanonicalSyncOwnershipKind::L0Operand
                  ? space == AddressSpace::LEFT || space == AddressSpace::RIGHT
              : cycle.kind == CanonicalSyncOwnershipKind::L1Tile
                  ? space == AddressSpace::MAT
                  : space == AddressSpace::ACC;
          if (!managed) {
            continue;
          }
          if (!access->exactPhysical ||
              (cycle.kind == CanonicalSyncOwnershipKind::L0Accumulator
                   ? !syncCoverStorageModeWrites(access->mode)
                   : access->mode != SyncCoverStorageAccessMode::Write)) {
            return false;
          }
          produced.push_back({access->domain, access->extent});
        }
      }
      llvm::sort(produced);
      const bool producerSlotsMatch =
          cycle.kind == CanonicalSyncOwnershipKind::L0Accumulator
              ? llvm::all_of(use.producers,
                             [&](SyncCoverNodeId producer) {
                               return slotsMatchWritingSpace(
                                   program, accessesByNode, producer,
                                   cycle.lanes[use.lane].slots,
                                   AddressSpace::ACC);
                             })
              : produced == cycle.lanes[use.lane].slots;
      if (!producerSlotsMatch) {
        return false;
      }
      for (SyncCoverNodeId consumer : use.consumers) {
        const bool matches =
            cycle.kind == CanonicalSyncOwnershipKind::L0Operand
                ? slotsMatch(program, accessesByNode, consumer,
                             cycle.lanes[use.lane].slots, AddressSpace::LEFT,
                             AddressSpace::RIGHT,
                             SyncCoverStorageAccessMode::Read,
                             SyncCoverStorageAccessMode::Read)
                : slotsMatchSpace(program, accessesByNode, consumer,
                                  cycle.lanes[use.lane].slots,
                                  cycle.kind ==
                                          CanonicalSyncOwnershipKind::L1Tile
                                      ? AddressSpace::MAT
                                      : AddressSpace::ACC,
                                  SyncCoverStorageAccessMode::Read);
        if (!matches) {
          return false;
        }
      }
    }
  }
  for (const SyncCoverStorageAccess &access : graph.getStorageAccesses()) {
    if (!graph.scopeContains(cycle.recurrenceScope,
                             graph.getNodes()[access.node].scope) ||
        access.domain >= program.getStorageSpaces().size()) {
      continue;
    }
    const AddressSpace space = program.getStorageSpaces()[access.domain];
    const bool managed =
        cycle.kind == CanonicalSyncOwnershipKind::L0Operand
            ? space == AddressSpace::LEFT || space == AddressSpace::RIGHT
        : cycle.kind == CanonicalSyncOwnershipKind::L1Tile
            ? space == AddressSpace::MAT
            : space == AddressSpace::ACC;
    if (managed && accessOverlapsCycle(cycle, access) &&
        represented.count(access.node) == 0) {
      return false;
    }
  }
  return !represented.empty();
}

} // namespace

bool CanonicalSyncOwnershipSlot::operator<(
    const CanonicalSyncOwnershipSlot &other) const {
  return std::tie(domain, extent.begin, extent.end) <
         std::tie(other.domain, other.extent.begin, other.extent.end);
}

bool CanonicalSyncOwnershipSlot::operator==(
    const CanonicalSyncOwnershipSlot &other) const {
  return domain == other.domain && extent.begin == other.extent.begin &&
         extent.end == other.extent.end;
}

bool mlir::pto::verifyCanonicalSyncOwnershipCycle(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle) {
  const SyncCoverGraph &graph = program.getGraph();
  const std::optional<StorageAccessIndex> accessesByNode =
      buildStorageAccessIndex(graph);
  if (!accessesByNode) {
    return false;
  }
  return verifyCycleImpl(program, cycle, *accessesByNode, true);
}

CanonicalSyncOwnershipResult mlir::pto::discoverCanonicalSyncOwnershipCycles(
    const CanonicalSyncProgram &program,
    CanonicalSyncOwnershipOptions options) {
  CanonicalSyncOwnershipResult result;
  const SyncCoverGraph &graph = program.getGraph();
  const bool invalid =
      !graph.isStructureFrozen() || !graph.validate() ||
      program.getNodeBindings().size() != graph.getNodes().size() ||
      program.getScopeBindings().size() != graph.getScopes().size() ||
      program.getStorageSpaces().size() != graph.getStorageDomains().size();
  if (invalid) {
    result.error = CanonicalSyncOwnershipError::InvalidProgram;
    return result;
  }
  const OwnershipSpec l0Operand{
      CanonicalSyncOwnershipKind::L0Operand,
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE1),
      static_cast<std::uint32_t>(PipelineType::PIPE_M),
      {AddressSpace::LEFT, AddressSpace::RIGHT},
      2,
      2};
  const HierarchicalOwnershipSpec l1Tile{
      CanonicalSyncOwnershipKind::L1Tile,
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE2),
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE1),
      AddressSpace::MAT,
      2,
      false};
  const HierarchicalOwnershipSpec l0Accumulator{
      CanonicalSyncOwnershipKind::L0Accumulator,
      static_cast<std::uint32_t>(PipelineType::PIPE_M),
      static_cast<std::uint32_t>(PipelineType::PIPE_FIX),
      AddressSpace::ACC,
      1,
      true};
  const std::size_t passCost =
      graph.getNodes().size() + graph.getStorageAccesses().size();
  if (passCost > options.maximumInspections) {
    result.truncated = true;
    return result;
  }
  result.inspections = passCost;
  const std::optional<StorageAccessIndex> accessesByNode =
      buildStorageAccessIndex(graph);
  if (!accessesByNode) {
    result.error = CanonicalSyncOwnershipError::InvalidProgram;
    return result;
  }
  for (const SyncCoverScope &scope : graph.getScopes()) {
    if (!scope.isLoop) {
      continue;
    }
    if (result.inspections > options.maximumInspections ||
        passCost > options.maximumInspections - result.inspections) {
      result.truncated = true;
      return result;
    }
    result.inspections += passCost;
    std::optional<CanonicalSyncOwnershipCycle> cycle =
        recognizeL0Operand(program, scope.id, l0Operand, *accessesByNode);
    if (cycle && verifyCycleImpl(program, *cycle, *accessesByNode, false)) {
      if (result.cycles.size() == options.maximumCycles) {
        result.truncated = true;
        return result;
      }
      cycle->id = result.cycles.size();
      result.cycles.push_back(std::move(*cycle));
    }

    if (result.inspections > options.maximumInspections ||
        passCost > options.maximumInspections - result.inspections) {
      result.truncated = true;
      return result;
    }
    result.inspections += passCost;
    cycle = recognizeHierarchicalOwnership(program, scope.id, l1Tile,
                                           *accessesByNode);
    if (cycle && verifyCycleImpl(program, *cycle, *accessesByNode, false)) {
      if (result.cycles.size() == options.maximumCycles) {
        result.truncated = true;
        return result;
      }
      cycle->id = result.cycles.size();
      result.cycles.push_back(std::move(*cycle));
    }

    if (result.inspections > options.maximumInspections ||
        passCost > options.maximumInspections - result.inspections) {
      result.truncated = true;
      return result;
    }
    result.inspections += passCost;
    for (CanonicalSyncOwnershipCycle parity :
         recognizeParityL1(program, scope.id, l1Tile, *accessesByNode)) {
      if (!verifyCycleImpl(program, parity, *accessesByNode, false)) {
        continue;
      }
      if (result.cycles.size() == options.maximumCycles) {
        result.truncated = true;
        return result;
      }
      parity.id = result.cycles.size();
      result.cycles.push_back(std::move(parity));
    }

    if (result.inspections > options.maximumInspections ||
        passCost > options.maximumInspections - result.inspections) {
      result.truncated = true;
      return result;
    }
    result.inspections += passCost;
    cycle = recognizeHierarchicalOwnership(program, scope.id, l0Accumulator,
                                           *accessesByNode);
    if (cycle) {
      cycle->protocol =
          CanonicalSyncOwnershipProtocolKind::BoundaryGuardedRoundTrip;
    }
    if (cycle && verifyCycleImpl(program, *cycle, *accessesByNode, false)) {
      if (result.cycles.size() == options.maximumCycles) {
        result.truncated = true;
        return result;
      }
      cycle->id = result.cycles.size();
      result.cycles.push_back(std::move(*cycle));
    }
  }
  return result;
}
