// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
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
getExactSlot(const CanonicalMemoryAccess &access, const OwnershipSpec &spec) {
  const bool exactAddress = access.addresses.size() == 1;
  const bool usable = includesSpace(spec, access.space) &&
                      access.knownPhysical && !access.unknownRange &&
                      exactAddress && access.size != 0;
  if (!usable) {
    return std::nullopt;
  }
  return CanonicalPhysicalSlot{access.space, access.addresses.front(),
                               access.size};
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
  const bool hasMultipleLanes = bundleLanes.size() >= 2;
  const bool hasDisjointLanes = lanesAreDisjoint(bundleLanes);
  if (!hasMultipleLanes || !hasDisjointLanes) {
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

} // namespace

void CanonicalSyncPlanBuilder::analyzeOwnershipCycles() {
  const OwnershipSpec l0OperandSpec{
      CanonicalOwnershipKind::L0Operand,
      PipelineType::PIPE_MTE1,
      PipelineType::PIPE_M,
      {AddressSpace::LEFT, AddressSpace::RIGHT},
  };
  SmallVector<scf::ForOp, 4> loops;
  func_.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    std::optional<CanonicalOwnershipCycle> cycle =
        recognizeOwnershipCycle(loop, plan_.nodes_, l0OperandSpec);
    if (cycle) {
      plan_.ownershipCycles_.push_back(std::move(*cycle));
    }
  }
}
