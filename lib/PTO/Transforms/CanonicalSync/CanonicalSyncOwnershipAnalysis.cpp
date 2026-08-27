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

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
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

bool verifyUse(const CanonicalSyncProgram &program,
               const CanonicalSyncOwnershipCycle &cycle,
               const CanonicalSyncOwnershipPath &path,
               const CanonicalSyncOwnershipUse &use,
               std::set<SyncCoverNodeId> &represented) {
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
  if (std::tie(use.writeAcquire.kind, use.writeAcquire.node,
               use.writeAcquire.scope, use.writeAcquire.position) !=
      std::tie(expectedAcquire.kind, expectedAcquire.node,
               expectedAcquire.scope, expectedAcquire.position)) {
    return false;
  }
  return true;
}

bool verifyCycleImpl(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    ArrayRef<std::vector<const SyncCoverStorageAccess *>> accessesByNode,
    bool validateGraph) {
  const SyncCoverGraph &graph = program.getGraph();
  if (!graph.isStructureFrozen() ||
      (validateGraph && !static_cast<bool>(graph.validate())) ||
      accessesByNode.size() != graph.getNodes().size() ||
      cycle.recurrenceScope >= graph.getScopes().size() ||
      !graph.getScopes()[cycle.recurrenceScope].isLoop || cycle.lanes.empty() ||
      cycle.paths.empty() || !slotsAreDisjoint(cycle.lanes) ||
      cycle.kind != CanonicalSyncOwnershipKind::L0Operand ||
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
          if (space != AddressSpace::LEFT && space != AddressSpace::RIGHT) {
            continue;
          }
          if (!access->exactPhysical ||
              access->mode != SyncCoverStorageAccessMode::Write) {
            return false;
          }
          produced.push_back({access->domain, access->extent});
        }
      }
      llvm::sort(produced);
      if (produced != cycle.lanes[use.lane].slots) {
        return false;
      }
      for (SyncCoverNodeId consumer : use.consumers) {
        if (!slotsMatch(program, accessesByNode, consumer,
                        cycle.lanes[use.lane].slots, AddressSpace::LEFT,
                        AddressSpace::RIGHT, SyncCoverStorageAccessMode::Read,
                        SyncCoverStorageAccessMode::Read)) {
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
        space == AddressSpace::LEFT || space == AddressSpace::RIGHT;
    if (managed && represented.count(access.node) == 0) {
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
    if (!cycle || !verifyCycleImpl(program, *cycle, *accessesByNode, false)) {
      continue;
    }
    if (result.cycles.size() == options.maximumCycles) {
      result.truncated = true;
      return result;
    }
    cycle->id = result.cycles.size();
    result.cycles.push_back(std::move(*cycle));
  }
  return result;
}
