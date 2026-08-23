// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct OwnershipSpec {
  CanonicalOwnershipKind kind = CanonicalOwnershipKind::L0Operand;
  PipelineType producerPipe = PipelineType::PIPE_UNASSIGNED;
  PipelineType consumerPipe = PipelineType::PIPE_UNASSIGNED;
  SmallVector<AddressSpace, 2> spaces;
  unsigned minimumLanes = 2;
};

struct HierarchicalOwnershipSpec {
  CanonicalOwnershipKind kind = CanonicalOwnershipKind::L1Tile;
  PipelineType producerPipe = PipelineType::PIPE_UNASSIGNED;
  PipelineType consumerPipe = PipelineType::PIPE_UNASSIGNED;
  AddressSpace space = AddressSpace::Zero;
  unsigned minimumLanes = 1;
  bool allowProducerReads = false;
};

struct OwnershipNode {
  std::size_t id = 0;
  Operation *operation = nullptr;
  Operation *topLevelChild = nullptr;
  Region *path = nullptr;
  SmallVector<CanonicalPhysicalSlot, 2> produced;
  SmallVector<CanonicalPhysicalSlot, 2> consumed;
};

struct PathItem {
  bool producer = false;
  std::size_t order = 0;
  OwnershipNode *node = nullptr;
  Operation *consumerAnchor = nullptr;
  SmallVector<OwnershipNode *, 2> consumers;
};

using SlotBundle = std::vector<CanonicalPhysicalSlot>;

bool includesSpace(const OwnershipSpec &spec, AddressSpace space) {
  return llvm::is_contained(spec.spaces, space);
}

bool slotsOverlap(const CanonicalPhysicalSlot &first,
                  const CanonicalPhysicalSlot &second) {
  if (first.space != second.space) {
    return false;
  }
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const bool firstOverflows = first.address > maximum - first.size;
  const bool secondOverflows = second.address > maximum - second.size;
  if (firstOverflows || secondOverflows) {
    return true;
  }
  return first.address < second.address + second.size &&
         second.address < first.address + first.size;
}

std::optional<CanonicalPhysicalSlot>
getExactSlot(const CanonicalMemoryAccess &access,
             ArrayRef<AddressSpace> spaces) {
  const bool exactAddress = access.addresses.size() == 1;
  const bool usable = llvm::is_contained(spaces, access.space) &&
                      access.knownPhysical && !access.unknownRange &&
                      exactAddress && access.size != 0;
  if (!usable) {
    return std::nullopt;
  }
  return CanonicalPhysicalSlot{access.space, access.addresses.front(),
                               access.size};
}

std::optional<CanonicalPhysicalSlot>
getExactSlot(const CanonicalMemoryAccess &access, const OwnershipSpec &spec) {
  return getExactSlot(access, spec.spaces);
}

Operation *getTopLevelChild(Operation *operation, Operation *loop) {
  Operation *child = operation;
  while (child) {
    Operation *parent = child->getParentOp();
    if (parent == loop) {
      break;
    }
    child = child->getParentOp();
  }
  return child;
}

Region *getPathRegion(Operation *operation, Operation *branch) {
  Operation *child = operation;
  while (child) {
    Operation *parent = child->getParentOp();
    if (parent == branch) {
      break;
    }
    child = child->getParentOp();
  }
  return child ? child->getParentRegion() : nullptr;
}

Operation *getPathAnchor(Operation *operation, Region *path) {
  Operation *anchor = operation;
  while (anchor) {
    Region *parent = anchor->getParentRegion();
    if (parent == path) {
      break;
    }
    anchor = anchor->getParentOp();
  }
  return anchor;
}

bool hasRequiredSpaces(ArrayRef<CanonicalPhysicalSlot> slots,
                       const OwnershipSpec &spec) {
  const std::size_t slotCount = slots.size();
  const std::size_t requiredCount = spec.spaces.size();
  if (slotCount != requiredCount) {
    return false;
  }
  return llvm::all_of(spec.spaces, [&](AddressSpace space) {
    return llvm::count_if(slots, [&](const CanonicalPhysicalSlot &slot) {
             return slot.space == space;
           }) == 1;
  });
}

std::optional<SmallVector<OwnershipNode, 16>>
collectOwnershipNodes(scf::ForOp loop, ArrayRef<CanonicalSyncNode> nodes,
                      const OwnershipSpec &spec) {
  SmallVector<OwnershipNode, 16> result;
  for (const CanonicalSyncNode &node : nodes) {
    if (!loop->isAncestor(node.operation)) {
      continue;
    }
    SmallVector<std::pair<CanonicalPhysicalSlot, const CanonicalMemoryAccess *>,
                2>
        accesses;
    for (const CanonicalMemoryAccess &access : node.accesses) {
      if (!includesSpace(spec, access.space)) {
        continue;
      }
      std::optional<CanonicalPhysicalSlot> slot = getExactSlot(access, spec);
      if (!slot) {
        return std::nullopt;
      }
      accesses.push_back({*slot, &access});
    }
    if (accesses.empty()) {
      continue;
    }

    OwnershipNode ownershipNode;
    ownershipNode.id = node.id;
    ownershipNode.operation = node.operation;
    ownershipNode.topLevelChild = getTopLevelChild(node.operation, loop);
    if (!ownershipNode.topLevelChild) {
      return std::nullopt;
    }
    if (node.pipe == spec.producerPipe) {
      if (llvm::any_of(accesses, [](const auto &entry) {
            return !entry.second->writes || entry.second->reads;
          })) {
        return std::nullopt;
      }
      llvm::transform(accesses, std::back_inserter(ownershipNode.produced),
                      [](const auto &entry) { return entry.first; });
      llvm::sort(ownershipNode.produced);
    } else if (node.pipe == spec.consumerPipe) {
      if (llvm::any_of(accesses, [](const auto &entry) {
            return !entry.second->reads || entry.second->writes;
          })) {
        return std::nullopt;
      }
      llvm::transform(accesses, std::back_inserter(ownershipNode.consumed),
                      [](const auto &entry) { return entry.first; });
      llvm::sort(ownershipNode.consumed);
      if (!hasRequiredSpaces(ownershipNode.consumed, spec)) {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
    result.push_back(std::move(ownershipNode));
  }
  return result.empty()
             ? std::nullopt
             : std::optional<SmallVector<OwnershipNode, 16>>(std::move(result));
}

bool validateConsumerGroup(Operation *anchor,
                           ArrayRef<OwnershipNode *> consumers) {
  auto ifOp = dyn_cast<scf::IfOp>(anchor);
  if (!ifOp) {
    return consumers.size() == 1 && consumers.front()->operation == anchor;
  }
  const bool hasElseRegion = !ifOp.getElseRegion().empty();
  const bool hasTwoConsumers = consumers.size() == 2;
  if (!hasElseRegion || !hasTwoConsumers) {
    return false;
  }
  bool hasThen = false;
  bool hasElse = false;
  for (const OwnershipNode *consumer : consumers) {
    Region *region = getPathRegion(consumer->operation, anchor);
    if (region == &ifOp.getThenRegion()) {
      if (hasThen) {
        return false;
      }
      hasThen = true;
    } else if (region == &ifOp.getElseRegion()) {
      if (hasElse) {
        return false;
      }
      hasElse = true;
    } else {
      return false;
    }
  }
  return hasThen && hasElse;
}

std::optional<CanonicalOwnershipPath> parseOwnershipPath(
    Region *path, MutableArrayRef<OwnershipNode> nodes,
    ArrayRef<CanonicalSyncNode> planNodes, const OwnershipSpec &spec,
    std::map<SlotBundle, unsigned> &bundleLanes, bool allowNewBundles) {
  SmallVector<PathItem, 24> items;
  std::map<Operation *, SmallVector<OwnershipNode *, 2>, std::less<Operation *>>
      consumerGroups;
  for (OwnershipNode &node : nodes) {
    if (node.path != path) {
      continue;
    }
    if (!node.produced.empty()) {
      Region *parent = node.operation->getParentRegion();
      if (parent != path) {
        return std::nullopt;
      }
      items.push_back({true, planNodes[node.id].order, &node, nullptr, {}});
      continue;
    }
    Operation *anchor = getPathAnchor(node.operation, path);
    if (!anchor || (anchor != node.operation && !isa<scf::IfOp>(anchor))) {
      return std::nullopt;
    }
    consumerGroups[anchor].push_back(&node);
  }
  for (auto &entry : consumerGroups) {
    if (!validateConsumerGroup(entry.first, entry.second)) {
      return std::nullopt;
    }
    const auto &bundle = entry.second.front()->consumed;
    if (llvm::any_of(entry.second, [&](const OwnershipNode *consumer) {
          return consumer->consumed != bundle;
        })) {
      return std::nullopt;
    }
    const auto minimum =
        llvm::min_element(entry.second, [&](const OwnershipNode *first,
                                            const OwnershipNode *second) {
          return planNodes[first->id].order < planNodes[second->id].order;
        });
    items.push_back({false, planNodes[(*minimum)->id].order, nullptr,
                     entry.first, entry.second});
  }
  llvm::stable_sort(items, [](const PathItem &first, const PathItem &second) {
    return first.order < second.order;
  });

  CanonicalOwnershipPath result;
  result.region = path;
  SmallVector<OwnershipNode *, 4> producers;
  for (PathItem &item : items) {
    if (item.producer) {
      producers.push_back(item.node);
      continue;
    }
    if (producers.empty()) {
      return std::nullopt;
    }
    SlotBundle produced;
    for (const OwnershipNode *producer : producers) {
      produced.insert(produced.end(), producer->produced.begin(),
                      producer->produced.end());
    }
    llvm::sort(produced);
    const SlotBundle consumed(item.consumers.front()->consumed.begin(),
                              item.consumers.front()->consumed.end());
    const bool hasExpectedSpaces = hasRequiredSpaces(produced, spec);
    const bool matchesConsumer = produced == consumed;
    if (!hasExpectedSpaces || !matchesConsumer) {
      return std::nullopt;
    }
    auto laneIt = bundleLanes.find(produced);
    if (laneIt == bundleLanes.end()) {
      if (!allowNewBundles) {
        return std::nullopt;
      }
      laneIt = bundleLanes.emplace(produced, bundleLanes.size()).first;
    }

    const auto producerOrder = [&](const OwnershipNode *producer) {
      return planNodes[producer->id].order;
    };
    auto firstProducer =
        llvm::min_element(producers, [&](const OwnershipNode *first,
                                         const OwnershipNode *second) {
          return producerOrder(first) < producerOrder(second);
        });
    auto lastProducer =
        llvm::max_element(producers, [&](const OwnershipNode *first,
                                         const OwnershipNode *second) {
          return producerOrder(first) < producerOrder(second);
        });

    CanonicalOwnershipUse use;
    use.lane = laneIt->second;
    use.producerLane = use.lane;
    llvm::transform(producers, std::back_inserter(use.producers),
                    [](const OwnershipNode *producer) { return producer->id; });
    llvm::transform(item.consumers, std::back_inserter(use.consumers),
                    [](const OwnershipNode *consumer) { return consumer->id; });
    use.writeAcquireAnchor = {(*firstProducer)->operation, true};
    use.readyAnchor = {(*lastProducer)->operation, false};
    use.readAcquireAnchor = {item.consumerAnchor, true};
    use.releaseAnchor = {item.consumerAnchor, false};
    result.uses.push_back(std::move(use));
    producers.clear();
  }
  const bool hasUnmatchedProducers = !producers.empty();
  const bool hasNoUses = result.uses.empty();
  if (hasUnmatchedProducers || hasNoUses) {
    return std::nullopt;
  }
  return result;
}

bool lanesAreDisjoint(const std::map<SlotBundle, unsigned> &bundleLanes) {
  for (auto first = bundleLanes.begin(); first != bundleLanes.end(); ++first) {
    for (auto second = std::next(first); second != bundleLanes.end();
         ++second) {
      for (const CanonicalPhysicalSlot &firstSlot : first->first) {
        for (const CanonicalPhysicalSlot &secondSlot : second->first) {
          if (slotsOverlap(firstSlot, secondSlot)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

std::optional<CanonicalOwnershipCycle>
recognizeOwnershipCycle(scf::ForOp loop, ArrayRef<CanonicalSyncNode> planNodes,
                        const OwnershipSpec &spec) {
  std::optional<SmallVector<OwnershipNode, 16>> nodes =
      collectOwnershipNodes(loop, planNodes, spec);
  if (!nodes) {
    return std::nullopt;
  }

  Operation *commonChild = nodes->front().topLevelChild;
  const bool oneTopLevelChild = llvm::all_of(*nodes, [&](const auto &node) {
    return node.topLevelChild == commonChild;
  });
  scf::IfOp branch =
      oneTopLevelChild ? dyn_cast<scf::IfOp>(commonChild) : scf::IfOp{};
  SmallVector<Region *, 2> pathRegions;
  if (branch) {
    if (branch.getElseRegion().empty()) {
      return std::nullopt;
    }
    pathRegions.push_back(&branch.getThenRegion());
    pathRegions.push_back(&branch.getElseRegion());
    for (OwnershipNode &node : *nodes) {
      node.path = getPathRegion(node.operation, branch);
    }
  } else {
    pathRegions.push_back(&loop.getRegion());
    for (OwnershipNode &node : *nodes) {
      node.path = &loop.getRegion();
    }
  }

  CanonicalOwnershipCycle result;
  result.kind = spec.kind;
  result.loop = loop;
  result.producerPipe = spec.producerPipe;
  result.consumerPipe = spec.consumerPipe;
  std::map<SlotBundle, unsigned> bundleLanes;
  for (auto [pathIndex, path] : llvm::enumerate(pathRegions)) {
    std::optional<CanonicalOwnershipPath> parsed = parseOwnershipPath(
        path, *nodes, planNodes, spec, bundleLanes, pathIndex == 0);
    if (!parsed) {
      return std::nullopt;
    }
    result.paths.push_back(std::move(*parsed));
  }
  const bool hasEnoughLanes = bundleLanes.size() >= spec.minimumLanes;
  const bool hasDisjointLanes = lanesAreDisjoint(bundleLanes);
  if (!hasEnoughLanes || !hasDisjointLanes) {
    return std::nullopt;
  }
  for (const CanonicalOwnershipPath &path : result.paths) {
    std::set<unsigned> lanes;
    for (const CanonicalOwnershipUse &use : path.uses) {
      lanes.insert(use.lane);
    }
    const std::size_t pathLaneCount = lanes.size();
    const std::size_t cycleLaneCount = bundleLanes.size();
    if (pathLaneCount != cycleLaneCount) {
      return std::nullopt;
    }
  }
  result.lanes.resize(bundleLanes.size());
  for (const auto &entry : bundleLanes) {
    CanonicalOwnershipLane &lane = result.lanes[entry.second];
    lane.id = entry.second;
    lane.slots.append(entry.first.begin(), entry.first.end());
  }
  return result;
}

struct SlotOwnershipGroup {
  CanonicalPhysicalSlot slot;
  SmallVector<OwnershipNode *, 8> producers;
  SmallVector<OwnershipNode *, 8> consumers;
};

std::optional<SmallVector<OwnershipNode, 32>> collectHierarchicalNodes(
    scf::ForOp loop, ArrayRef<CanonicalSyncNode> planNodes,
    const HierarchicalOwnershipSpec &spec) {
  SmallVector<OwnershipNode, 32> result;
  const AddressSpace spaces[] = {spec.space};
  for (const CanonicalSyncNode &node : planNodes) {
    if (!loop->isAncestor(node.operation)) {
      continue;
    }
    SmallVector<std::pair<CanonicalPhysicalSlot, const CanonicalMemoryAccess *>,
                2>
        accesses;
    for (const CanonicalMemoryAccess &access : node.accesses) {
      if (access.space != spec.space) {
        continue;
      }
      std::optional<CanonicalPhysicalSlot> slot = getExactSlot(access, spaces);
      if (!slot) {
        return std::nullopt;
      }
      accesses.push_back({*slot, &access});
    }
    if (accesses.empty()) {
      continue;
    }
    const bool hasSingleAccess = accesses.size() == 1;
    if (!hasSingleAccess) {
      return std::nullopt;
    }

    OwnershipNode ownershipNode;
    ownershipNode.id = node.id;
    ownershipNode.operation = node.operation;
    ownershipNode.topLevelChild = getTopLevelChild(node.operation, loop);
    if (!ownershipNode.topLevelChild) {
      return std::nullopt;
    }
    const CanonicalMemoryAccess &access = *accesses.front().second;
    if (node.pipe == spec.producerPipe) {
      if (!access.writes || (access.reads && !spec.allowProducerReads)) {
        return std::nullopt;
      }
      ownershipNode.produced.push_back(accesses.front().first);
    } else if (node.pipe == spec.consumerPipe) {
      if (!access.reads || access.writes) {
        return std::nullopt;
      }
      ownershipNode.consumed.push_back(accesses.front().first);
    } else {
      return std::nullopt;
    }
    result.push_back(std::move(ownershipNode));
  }
  return result.empty()
             ? std::nullopt
             : std::optional<SmallVector<OwnershipNode, 32>>(std::move(result));
}

Operation *getCommonTopLevelAnchor(ArrayRef<OwnershipNode *> nodes) {
  if (nodes.empty()) {
    return nullptr;
  }
  Operation *anchor = nodes.front()->topLevelChild;
  return llvm::all_of(nodes, [&](const OwnershipNode *node) {
           return node->topLevelChild == anchor;
         })
             ? anchor
             : nullptr;
}

bool hierarchicalSlotsAreDisjoint(
    const std::map<CanonicalPhysicalSlot, SlotOwnershipGroup> &groups) {
  for (auto first = groups.begin(); first != groups.end(); ++first) {
    for (auto second = std::next(first); second != groups.end(); ++second) {
      if (slotsOverlap(first->first, second->first)) {
        return false;
      }
    }
  }
  return true;
}

std::optional<CanonicalOwnershipCycle> recognizeHierarchicalOwnershipCycle(
    scf::ForOp loop, ArrayRef<CanonicalSyncNode> planNodes,
    const HierarchicalOwnershipSpec &spec) {
  std::optional<SmallVector<OwnershipNode, 32>> nodes =
      collectHierarchicalNodes(loop, planNodes, spec);
  if (!nodes) {
    return std::nullopt;
  }

  std::map<CanonicalPhysicalSlot, SlotOwnershipGroup> groups;
  for (OwnershipNode &node : *nodes) {
    const CanonicalPhysicalSlot &slot = !node.produced.empty()
                                            ? node.produced.front()
                                            : node.consumed.front();
    SlotOwnershipGroup &group = groups[slot];
    group.slot = slot;
    (!node.produced.empty() ? group.producers : group.consumers)
        .push_back(&node);
  }
  const bool hasEnoughLanes = groups.size() >= spec.minimumLanes;
  const bool hasDisjointLanes = hierarchicalSlotsAreDisjoint(groups);
  if (!hasEnoughLanes || !hasDisjointLanes) {
    return std::nullopt;
  }

  SmallVector<SlotOwnershipGroup *, 8> orderedGroups;
  for (auto &entry : groups) {
    SlotOwnershipGroup &group = entry.second;
    const bool hasProducer = !group.producers.empty();
    const bool hasConsumer = !group.consumers.empty();
    if (!hasProducer || !hasConsumer) {
      return std::nullopt;
    }
    orderedGroups.push_back(&group);
  }
  llvm::stable_sort(orderedGroups,
                    [&](const SlotOwnershipGroup *first,
                        const SlotOwnershipGroup *second) {
                      return planNodes[first->producers.front()->id].order <
                             planNodes[second->producers.front()->id].order;
                    });

  CanonicalOwnershipCycle result;
  result.kind = spec.kind;
  result.loop = loop;
  result.producerPipe = spec.producerPipe;
  result.consumerPipe = spec.consumerPipe;
  CanonicalOwnershipPath path;
  path.region = &loop.getRegion();
  for (auto [lane, group] : llvm::enumerate(orderedGroups)) {
    Operation *producerAnchor = getCommonTopLevelAnchor(group->producers);
    Operation *consumerAnchor = getCommonTopLevelAnchor(group->consumers);
    if (!producerAnchor || !consumerAnchor ||
        producerAnchor == consumerAnchor ||
        producerAnchor->getBlock() != consumerAnchor->getBlock() ||
        !producerAnchor->isBeforeInBlock(consumerAnchor)) {
      return std::nullopt;
    }

    CanonicalOwnershipLane ownershipLane;
    ownershipLane.id = lane;
    ownershipLane.slots.push_back(group->slot);
    result.lanes.push_back(std::move(ownershipLane));

    CanonicalOwnershipUse use;
    use.lane = lane;
    use.producerLane = use.lane;
    llvm::transform(group->producers, std::back_inserter(use.producers),
                    [](const OwnershipNode *node) { return node->id; });
    llvm::transform(group->consumers, std::back_inserter(use.consumers),
                    [](const OwnershipNode *node) { return node->id; });
    use.writeAcquireAnchor = {producerAnchor, true};
    use.readyAnchor = {producerAnchor, false};
    use.readAcquireAnchor = {consumerAnchor, true};
    use.releaseAnchor = {consumerAnchor, false};
    path.uses.push_back(std::move(use));
  }
  result.paths.push_back(std::move(path));
  return result;
}

struct ParitySlotGroup {
  CanonicalPhysicalSlot slot;
  std::array<SmallVector<OwnershipNode *, 8>, 2> producers;
  std::array<SmallVector<OwnershipNode *, 8>, 2> consumers;
};

bool paritySlotsAreDisjoint(
    const std::map<CanonicalPhysicalSlot, ParitySlotGroup> &groups) {
  for (auto first = groups.begin(); first != groups.end(); ++first) {
    for (auto second = std::next(first); second != groups.end(); ++second) {
      if (slotsOverlap(first->first, second->first)) {
        return false;
      }
    }
  }
  return true;
}

bool matchPositiveConstant(Value value, std::uint64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) &&
         constant.isNonNegative() && constant.getZExtValue() == expected;
}

bool matchAlternatingParityBranch(scf::ForOp loop, scf::IfOp branch) {
  const bool validLoop = branch && !branch.getElseRegion().empty() &&
                         matchPositiveConstant(loop.getLowerBound(), 0) &&
                         matchPositiveConstant(loop.getStep(), 1);
  if (!validLoop) {
    return false;
  }
  auto compare = branch.getCondition().getDefiningOp<arith::CmpIOp>();
  const bool comparesForEquality =
      compare && compare.getPredicate() == arith::CmpIPredicate::eq;
  if (!comparesForEquality) {
    return false;
  }
  Value remainderValue = compare.getLhs();
  Value zeroValue = compare.getRhs();
  if (!matchPositiveConstant(zeroValue, 0)) {
    remainderValue = compare.getRhs();
    zeroValue = compare.getLhs();
  }
  if (!matchPositiveConstant(zeroValue, 0)) {
    return false;
  }
  auto remainder = remainderValue.getDefiningOp<arith::RemSIOp>();
  return remainder && remainder.getLhs() == loop.getInductionVar() &&
         matchPositiveConstant(remainder.getRhs(), 2);
}

bool matchNextIteration(Value value, scf::ForOp loop) {
  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add) {
    return false;
  }
  return (add.getLhs() == loop.getInductionVar() &&
          matchPositiveConstant(add.getRhs(), 1)) ||
         (add.getRhs() == loop.getInductionVar() &&
          matchPositiveConstant(add.getLhs(), 1));
}

bool hasContinuationGuard(Operation *operation, scf::ForOp loop,
                          Region *path) {
  auto guard = dyn_cast_or_null<scf::IfOp>(operation->getParentOp());
  const bool directGuard =
      guard && path && guard->getParentRegion() == path &&
      operation->getParentRegion() == &guard.getThenRegion() &&
      guard.getElseRegion().empty();
  if (!directGuard) {
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
                            ArrayRef<CanonicalSyncNode> planNodes) {
  return *llvm::min_element(nodes, [&](const OwnershipNode *first,
                                      const OwnershipNode *second) {
    return planNodes[first->id].order < planNodes[second->id].order;
  });
}

OwnershipNode *lastByOrder(ArrayRef<OwnershipNode *> nodes,
                           ArrayRef<CanonicalSyncNode> planNodes) {
  return *llvm::max_element(nodes, [&](const OwnershipNode *first,
                                      const OwnershipNode *second) {
    return planNodes[first->id].order < planNodes[second->id].order;
  });
}

CanonicalOwnershipUse makeParityUse(
    unsigned lane, unsigned producerLane, ArrayRef<OwnershipNode *> producers,
    ArrayRef<OwnershipNode *> consumers,
    ArrayRef<CanonicalSyncNode> planNodes) {
  CanonicalOwnershipUse use;
  use.lane = lane;
  use.producerLane = producerLane;
  llvm::transform(producers, std::back_inserter(use.producers),
                  [](const OwnershipNode *node) { return node->id; });
  llvm::transform(consumers, std::back_inserter(use.consumers),
                  [](const OwnershipNode *node) { return node->id; });
  OwnershipNode *firstProducer = firstByOrder(producers, planNodes);
  OwnershipNode *lastProducer = lastByOrder(producers, planNodes);
  OwnershipNode *firstConsumer = firstByOrder(consumers, planNodes);
  OwnershipNode *lastConsumer = lastByOrder(consumers, planNodes);
  use.writeAcquireAnchor = {firstProducer->operation, true};
  use.readyAnchor = {lastProducer->operation, false};
  use.readAcquireAnchor = {firstConsumer->operation, true};
  use.releaseAnchor = {lastConsumer->operation, false};
  return use;
}

std::optional<std::size_t> findInitialProducer(
    scf::ForOp loop, const CanonicalPhysicalSlot &slot,
    ArrayRef<CanonicalPhysicalSlot> managedSlots,
    ArrayRef<CanonicalSyncNode> planNodes) {
  std::optional<std::size_t> result;
  for (const CanonicalSyncNode &node : planNodes) {
    Operation *topLevel = node.operation;
    bool outsideLoopBlock =
        topLevel && topLevel->getBlock() != loop->getBlock();
    while (outsideLoopBlock) {
      topLevel = topLevel->getParentOp();
      outsideLoopBlock =
          topLevel && topLevel->getBlock() != loop->getBlock();
    }
    const bool precedesLoop =
        topLevel && topLevel != loop && topLevel->isBeforeInBlock(loop);
    if (!precedesLoop) {
      continue;
    }
    bool touchesManagedSlot = false;
    bool isInitialWrite = false;
    bool hasOtherManagedAccess = false;
    for (const CanonicalMemoryAccess &access : node.accesses) {
      if (access.space != AddressSpace::MAT) {
        continue;
      }
      const AddressSpace spaces[] = {AddressSpace::MAT};
      std::optional<CanonicalPhysicalSlot> candidate =
          getExactSlot(access, spaces);
      if (!candidate) {
        return std::nullopt;
      }
      const bool overlapsManaged =
          llvm::any_of(managedSlots, [&](const CanonicalPhysicalSlot &managed) {
            return slotsOverlap(*candidate, managed);
          });
      if (!overlapsManaged) {
        continue;
      }
      touchesManagedSlot = true;
      const bool validInitial = node.operation == topLevel &&
                                node.pipe == PipelineType::PIPE_MTE2 &&
                                access.writes && !access.reads &&
                                *candidate == slot;
      hasOtherManagedAccess |= isInitialWrite || !validInitial;
      isInitialWrite |= validInitial;
    }
    if (!touchesManagedSlot) {
      continue;
    }
    if (!isInitialWrite || hasOtherManagedAccess || result) {
      return std::nullopt;
    }
    result = node.id;
  }
  return result;
}

SmallVector<CanonicalOwnershipCycle, 2> recognizeParityL1Cycles(
    scf::ForOp loop, ArrayRef<CanonicalSyncNode> planNodes,
    const HierarchicalOwnershipSpec &spec) {
  SmallVector<CanonicalOwnershipCycle, 2> result;
  std::optional<SmallVector<OwnershipNode, 32>> nodes =
      collectHierarchicalNodes(loop, planNodes, spec);
  if (!nodes) {
    return result;
  }
  Operation *commonChild = nodes->front().topLevelChild;
  const bool oneTopLevelChild = llvm::all_of(*nodes, [&](const auto &node) {
    return node.topLevelChild == commonChild;
  });
  scf::IfOp branch =
      oneTopLevelChild ? dyn_cast<scf::IfOp>(commonChild) : scf::IfOp{};
  if (!matchAlternatingParityBranch(loop, branch)) {
    return result;
  }
  Region *paths[] = {&branch.getThenRegion(), &branch.getElseRegion()};
  std::map<CanonicalPhysicalSlot, ParitySlotGroup> groups;
  for (OwnershipNode &node : *nodes) {
    node.path = getPathRegion(node.operation, branch);
    unsigned pathIndex = node.path == paths[0] ? 0 : node.path == paths[1] ? 1
                                                                         : 2;
    if (pathIndex == 2) {
      return result;
    }
    const CanonicalPhysicalSlot &slot = !node.produced.empty()
                                            ? node.produced.front()
                                            : node.consumed.front();
    ParitySlotGroup &group = groups[slot];
    group.slot = slot;
    (!node.produced.empty() ? group.producers[pathIndex]
                            : group.consumers[pathIndex])
        .push_back(&node);
  }
  if (!paritySlotsAreDisjoint(groups)) {
    return result;
  }

  SmallVector<ParitySlotGroup *, 8> stableGroups;
  SmallVector<ParitySlotGroup *, 2> alternatingGroups;
  for (auto &entry : groups) {
    ParitySlotGroup &group = entry.second;
    const bool stable = llvm::all_of(llvm::seq<unsigned>(0, 2), [&](unsigned p) {
      return group.producers[p].size() == 1 &&
             !group.consumers[p].empty() &&
             executeDirectlyIn(group.producers[p], paths[p]) &&
             executeDirectlyIn(group.consumers[p], paths[p]) &&
             planNodes[group.producers[p].front()->id].order <
                 planNodes[firstByOrder(group.consumers[p], planNodes)->id]
                     .order;
    });
    const bool alternating =
        (group.producers[0].size() == 1 && group.consumers[0].empty() &&
         group.producers[1].empty() && !group.consumers[1].empty() &&
         executeDirectlyIn(group.consumers[1], paths[1])) ||
        (group.producers[1].size() == 1 && group.consumers[1].empty() &&
         group.producers[0].empty() && !group.consumers[0].empty() &&
         executeDirectlyIn(group.consumers[0], paths[0]));
    if (stable) {
      stableGroups.push_back(&group);
    } else if (alternating) {
      alternatingGroups.push_back(&group);
    } else {
      return {};
    }
  }

  std::optional<CanonicalOwnershipCycle> stableCycle;
  const bool hasStableCycle = stableGroups.size() >= spec.minimumLanes &&
                              stableGroups.size() <= kMaxMultiBufferCount;
  if (hasStableCycle) {
    CanonicalOwnershipCycle stable;
    stable.kind = CanonicalOwnershipKind::L1Tile;
    stable.loop = loop;
    stable.producerPipe = spec.producerPipe;
    stable.consumerPipe = spec.consumerPipe;
    stable.paths.resize(2);
    stable.paths[0].region = paths[0];
    stable.paths[1].region = paths[1];
    for (auto [lane, group] : llvm::enumerate(stableGroups)) {
      CanonicalOwnershipLane ownershipLane;
      ownershipLane.id = lane;
      ownershipLane.slots.push_back(group->slot);
      stable.lanes.push_back(std::move(ownershipLane));
      for (unsigned pathIndex = 0; pathIndex < 2; ++pathIndex) {
        stable.paths[pathIndex].uses.push_back(makeParityUse(
            lane, lane, group->producers[pathIndex],
            group->consumers[pathIndex], planNodes));
      }
    }
    stableCycle = std::move(stable);
  }

  const bool hasAlternatingCycle = alternatingGroups.size() == 2;
  if (!hasAlternatingCycle) {
    return {};
  }
  CanonicalOwnershipCycle prefetch;
  prefetch.kind = CanonicalOwnershipKind::L1Tile;
  prefetch.protocol = CanonicalOwnershipProtocolKind::AlternatingPrefetch;
  prefetch.loop = loop;
  prefetch.producerPipe = spec.producerPipe;
  prefetch.consumerPipe = spec.consumerPipe;
  prefetch.paths.resize(2);
  prefetch.paths[0].region = paths[0];
  prefetch.paths[1].region = paths[1];
  for (auto [lane, group] : llvm::enumerate(alternatingGroups)) {
    CanonicalOwnershipLane ownershipLane;
    ownershipLane.id = lane;
    ownershipLane.slots.push_back(group->slot);
    prefetch.lanes.push_back(std::move(ownershipLane));
  }
  for (unsigned pathIndex = 0; pathIndex < 2; ++pathIndex) {
    ParitySlotGroup *consumerGroup = nullptr;
    ParitySlotGroup *producerGroup = nullptr;
    unsigned consumerLane = 0;
    unsigned producerLane = 0;
    for (auto [lane, group] : llvm::enumerate(alternatingGroups)) {
      if (!group->consumers[pathIndex].empty()) {
        consumerGroup = group;
        consumerLane = lane;
      }
      if (!group->producers[pathIndex].empty()) {
        producerGroup = group;
        producerLane = lane;
      }
    }
    if (!consumerGroup || !producerGroup || consumerLane == producerLane ||
        !hasContinuationGuard(producerGroup->producers[pathIndex].front()
                                  ->operation,
                              loop, paths[pathIndex])) {
      return result;
    }
    prefetch.paths[pathIndex].uses.push_back(makeParityUse(
        consumerLane, producerLane, producerGroup->producers[pathIndex],
        consumerGroup->consumers[pathIndex], planNodes));
  }
  const CanonicalOwnershipUse &firstPathUse = prefetch.paths[0].uses.front();
  SmallVector<CanonicalPhysicalSlot, 8> managedSlots;
  llvm::transform(groups, std::back_inserter(managedSlots),
                  [](const auto &entry) { return entry.first; });
  std::optional<std::size_t> initialProducer = findInitialProducer(
      loop, prefetch.lanes[firstPathUse.lane].slots.front(), managedSlots,
      planNodes);
  if (!initialProducer) {
    return result;
  }
  prefetch.initialProducers.push_back(*initialProducer);
  prefetch.initialReadyAnchor = {loop, true};
  prefetch.initialReadyLane = firstPathUse.lane;
  prefetch.initiallyFreeLanes.push_back(firstPathUse.producerLane);
  if (stableCycle) {
    result.push_back(std::move(*stableCycle));
  }
  result.push_back(std::move(prefetch));
  return result;
}

bool ownershipCyclesEqual(const CanonicalOwnershipCycle &first,
                          const CanonicalOwnershipCycle &second) {
  if (first.kind != second.kind || first.protocol != second.protocol ||
      first.loop != second.loop ||
      first.producerPipe != second.producerPipe ||
      first.consumerPipe != second.consumerPipe ||
      first.lanes.size() != second.lanes.size() ||
      first.initialProducers != second.initialProducers ||
      first.initialReadyAnchor.operation !=
          second.initialReadyAnchor.operation ||
      first.initialReadyAnchor.before != second.initialReadyAnchor.before ||
      first.initialReadyLane != second.initialReadyLane ||
      first.initiallyFreeLanes != second.initiallyFreeLanes) {
    return false;
  }
  for (auto [left, right] : llvm::zip(first.lanes, second.lanes)) {
    if (left.slots != right.slots) {
      return false;
    }
  }
  return true;
}

bool ownershipCyclesOverlap(const CanonicalOwnershipCycle &first,
                            const CanonicalOwnershipCycle &second) {
  const bool nestedScopes =
      first.loop == second.loop || first.loop->isAncestor(second.loop) ||
      second.loop->isAncestor(first.loop);
  if (!nestedScopes) {
    return false;
  }
  for (const CanonicalOwnershipLane &firstLane : first.lanes) {
    for (const CanonicalOwnershipLane &secondLane : second.lanes) {
      for (const CanonicalPhysicalSlot &firstSlot : firstLane.slots) {
        for (const CanonicalPhysicalSlot &secondSlot : secondLane.slots) {
          if (slotsOverlap(firstSlot, secondSlot)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void appendUniqueOwnershipCycle(
    SmallVectorImpl<CanonicalOwnershipCycle> &cycles,
    CanonicalOwnershipCycle cycle) {
  if (!llvm::any_of(cycles, [&](const CanonicalOwnershipCycle &existing) {
        return ownershipCyclesEqual(existing, cycle);
      })) {
    cycles.push_back(std::move(cycle));
  }
}

} // namespace

bool mlir::pto::verifyCanonicalAlternatingPathMapping(
    const CanonicalOwnershipCycle &cycle) {
  if (cycle.protocol != CanonicalOwnershipProtocolKind::AlternatingPrefetch ||
      !cycle.loop || cycle.paths.size() != 2 || !cycle.paths[0].region ||
      !cycle.paths[1].region) {
    return false;
  }
  auto loop = dyn_cast<scf::ForOp>(cycle.loop);
  auto branch = dyn_cast_or_null<scf::IfOp>(
      cycle.paths[0].region->getParentOp());
  return loop && branch && branch->getParentOp() == loop.getOperation() &&
         cycle.paths[0].region == &branch.getThenRegion() &&
         cycle.paths[1].region == &branch.getElseRegion() &&
         matchAlternatingParityBranch(loop, branch);
}

void CanonicalSyncPlanBuilder::analyzeOwnershipCycles() {
  const OwnershipSpec l0OperandSpec{
      CanonicalOwnershipKind::L0Operand,
      PipelineType::PIPE_MTE1,
      PipelineType::PIPE_M,
      {AddressSpace::LEFT, AddressSpace::RIGHT},
  };
  const HierarchicalOwnershipSpec l1TileSpec{
      CanonicalOwnershipKind::L1Tile,
      PipelineType::PIPE_MTE2,
      PipelineType::PIPE_MTE1,
      AddressSpace::MAT,
      2,
      false,
  };
  const HierarchicalOwnershipSpec l0AccumulatorSpec{
      CanonicalOwnershipKind::L0Accumulator,
      PipelineType::PIPE_M,
      PipelineType::PIPE_FIX,
      AddressSpace::ACC,
      1,
      true,
  };
  SmallVector<scf::ForOp, 4> loops;
  func_.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  SmallVector<CanonicalOwnershipCycle, 8> candidates;
  for (scf::ForOp loop : loops) {
    std::optional<CanonicalOwnershipCycle> cycle =
        recognizeOwnershipCycle(loop, plan_.nodes_, l0OperandSpec);
    if (cycle) {
      appendUniqueOwnershipCycle(candidates, std::move(*cycle));
    }
    cycle = recognizeHierarchicalOwnershipCycle(loop, plan_.nodes_, l1TileSpec);
    if (cycle) {
      appendUniqueOwnershipCycle(candidates, std::move(*cycle));
    }
    for (CanonicalOwnershipCycle parityCycle :
         recognizeParityL1Cycles(loop, plan_.nodes_, l1TileSpec)) {
      appendUniqueOwnershipCycle(candidates, std::move(parityCycle));
    }
    cycle = recognizeHierarchicalOwnershipCycle(loop, plan_.nodes_,
                                                l0AccumulatorSpec);
    if (cycle) {
      appendUniqueOwnershipCycle(candidates, std::move(*cycle));
    }
  }

  SmallVector<bool, 8> conflicts(candidates.size(), false);
  for (std::size_t first = 0; first < candidates.size(); ++first) {
    for (std::size_t second = first + 1; second < candidates.size(); ++second) {
      if (ownershipCyclesOverlap(candidates[first], candidates[second])) {
        conflicts[first] = true;
        conflicts[second] = true;
      }
    }
  }
  for (auto [index, cycle] : llvm::enumerate(candidates)) {
    if (conflicts[index]) {
      continue;
    }
    cycle.id = plan_.ownershipCycles_.size() + 1;
    plan_.ownershipCycles_.push_back(std::move(cycle));
  }
}
