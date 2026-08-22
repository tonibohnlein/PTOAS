// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct L0Slot {
  AddressSpace space = AddressSpace::Zero;
  std::uint64_t address = 0;
  std::uint64_t size = 0;

  bool operator<(const L0Slot &other) const {
    return std::tie(space, address, size) <
           std::tie(other.space, other.address, other.size);
  }

  bool operator==(const L0Slot &other) const {
    return space == other.space && address == other.address &&
           size == other.size;
  }
};

using L0SlotPair = std::pair<L0Slot, L0Slot>;

struct L0Node {
  std::size_t id = 0;
  Operation *operation = nullptr;
  Operation *topLevelChild = nullptr;
  Region *path = nullptr;
  std::optional<L0Slot> produced;
  std::optional<L0SlotPair> consumed;
};

struct OwnershipUse {
  unsigned lane = 0;
  std::size_t firstProducer = 0;
  std::size_t secondProducer = 0;
  SmallVector<std::size_t, 2> consumers;
  Operation *consumerAnchor = nullptr;
};

struct OwnershipPath {
  Region *region = nullptr;
  SmallVector<OwnershipUse, 8> uses;
};

struct OwnershipPattern {
  scf::ForOp loop;
  SmallVector<OwnershipPath, 2> paths;
};

bool isL0Space(AddressSpace space) {
  return space == AddressSpace::LEFT || space == AddressSpace::RIGHT;
}

bool slotsOverlap(const L0Slot &first, const L0Slot &second) {
  if (first.space != second.space ||
      first.address > std::numeric_limits<std::uint64_t>::max() - first.size ||
      second.address >
          std::numeric_limits<std::uint64_t>::max() - second.size) {
    return true;
  }
  return first.address < second.address + second.size &&
         second.address < first.address + first.size;
}

std::optional<L0Slot> getExactSlot(const CanonicalMemoryAccess &access) {
  if (!isL0Space(access.space) || !access.knownPhysical ||
      access.unknownRange || access.addresses.size() != 1 || access.size == 0) {
    return std::nullopt;
  }
  return L0Slot{access.space, access.addresses.front(), access.size};
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

std::optional<L0SlotPair> getProducerPair(const L0Node &first,
                                          const L0Node &second) {
  if (!first.produced || !second.produced ||
      first.produced->space == second.produced->space) {
    return std::nullopt;
  }
  const L0Slot &left = first.produced->space == AddressSpace::LEFT
                           ? *first.produced
                           : *second.produced;
  const L0Slot &right = first.produced->space == AddressSpace::RIGHT
                            ? *first.produced
                            : *second.produced;
  return L0SlotPair{left, right};
}

std::optional<SmallVector<L0Node, 16>>
collectL0Nodes(scf::ForOp loop, ArrayRef<CanonicalSyncNode> nodes) {
  SmallVector<L0Node, 16> result;
  for (const CanonicalSyncNode &node : nodes) {
    if (!loop->isAncestor(node.operation)) {
      continue;
    }
    SmallVector<std::pair<L0Slot, const CanonicalMemoryAccess *>, 2> accesses;
    for (const CanonicalMemoryAccess &access : node.accesses) {
      if (!isL0Space(access.space)) {
        continue;
      }
      std::optional<L0Slot> slot = getExactSlot(access);
      if (!slot) {
        return std::nullopt;
      }
      accesses.push_back({*slot, &access});
    }
    if (accesses.empty()) {
      continue;
    }

    L0Node l0Node;
    l0Node.id = node.id;
    l0Node.operation = node.operation;
    l0Node.topLevelChild = getTopLevelChild(node.operation, loop);
    if (!l0Node.topLevelChild) {
      return std::nullopt;
    }
    if (node.pipe == PipelineType::PIPE_MTE1) {
      if (accesses.size() != 1 || !accesses.front().second->writes ||
          accesses.front().second->reads) {
        return std::nullopt;
      }
      l0Node.produced = accesses.front().first;
    } else if (node.pipe == PipelineType::PIPE_M) {
      if (accesses.size() != 2 || llvm::any_of(accesses, [](const auto &entry) {
            return !entry.second->reads || entry.second->writes;
          })) {
        return std::nullopt;
      }
      const L0Slot &first = accesses[0].first;
      const L0Slot &second = accesses[1].first;
      if (first.space == second.space) {
        return std::nullopt;
      }
      l0Node.consumed = first.space == AddressSpace::LEFT
                            ? L0SlotPair{first, second}
                            : L0SlotPair{second, first};
    } else {
      return std::nullopt;
    }
    result.push_back(std::move(l0Node));
  }
  return result.empty() ? std::nullopt
                        : std::optional<SmallVector<L0Node, 16>>(result);
}

bool validateConsumerGroup(Operation *anchor, ArrayRef<L0Node *> consumers) {
  auto ifOp = dyn_cast<scf::IfOp>(anchor);
  if (!ifOp) {
    return consumers.size() == 1 && consumers.front()->operation == anchor;
  }
  if (ifOp.getElseRegion().empty() || consumers.size() != 2) {
    return false;
  }
  bool hasThen = false;
  bool hasElse = false;
  for (const L0Node *consumer : consumers) {
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

struct PathItem {
  bool producer = false;
  std::size_t order = 0;
  L0Node *node = nullptr;
  Operation *consumerAnchor = nullptr;
  SmallVector<L0Node *, 2> consumers;
};

std::optional<OwnershipPath>
parseOwnershipPath(Region *path, MutableArrayRef<L0Node> nodes,
                   ArrayRef<CanonicalSyncNode> planNodes,
                   std::map<L0SlotPair, unsigned> &pairLanes,
                   bool allowNewPairs) {
  SmallVector<PathItem, 24> items;
  std::map<Operation *, SmallVector<L0Node *, 2>, std::less<Operation *>>
      consumerGroups;
  for (L0Node &node : nodes) {
    if (node.path != path) {
      continue;
    }
    if (node.produced) {
      if (node.operation->getParentRegion() != path) {
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
    const L0SlotPair pair = *entry.second.front()->consumed;
    if (llvm::any_of(entry.second, [&](const L0Node *consumer) {
          return !consumer->consumed || *consumer->consumed != pair;
        })) {
      return std::nullopt;
    }
    const auto minimum = llvm::min_element(
        entry.second, [&](const L0Node *first, const L0Node *second) {
          return planNodes[first->id].order < planNodes[second->id].order;
        });
    items.push_back({false, planNodes[(*minimum)->id].order, nullptr,
                     entry.first, entry.second});
  }
  llvm::stable_sort(items, [](const PathItem &first, const PathItem &second) {
    return first.order < second.order;
  });
  if (items.empty() || items.size() % 3 != 0) {
    return std::nullopt;
  }

  OwnershipPath result;
  result.region = path;
  std::optional<unsigned> previousLane;
  for (std::size_t index = 0; index < items.size(); index += 3) {
    PathItem &first = items[index];
    PathItem &second = items[index + 1];
    PathItem &consumer = items[index + 2];
    if (!first.producer || !second.producer || consumer.producer) {
      return std::nullopt;
    }
    std::optional<L0SlotPair> pair = getProducerPair(*first.node, *second.node);
    if (!pair || llvm::any_of(consumer.consumers, [&](const L0Node *node) {
          return !node->consumed || *node->consumed != *pair;
        })) {
      return std::nullopt;
    }
    auto laneIt = pairLanes.find(*pair);
    if (laneIt == pairLanes.end()) {
      if (!allowNewPairs || pairLanes.size() >= 2) {
        return std::nullopt;
      }
      laneIt = pairLanes.emplace(*pair, pairLanes.size()).first;
    }
    const unsigned lane = laneIt->second;
    if (previousLane && *previousLane == lane) {
      return std::nullopt;
    }
    previousLane = lane;
    OwnershipUse use;
    use.lane = lane;
    use.firstProducer = first.node->id;
    use.secondProducer = second.node->id;
    use.consumerAnchor = consumer.consumerAnchor;
    for (const L0Node *node : consumer.consumers) {
      use.consumers.push_back(node->id);
    }
    result.uses.push_back(std::move(use));
  }
  return result;
}

std::optional<OwnershipPattern>
recognizeOwnershipPattern(scf::ForOp loop,
                          ArrayRef<CanonicalSyncNode> planNodes) {
  std::optional<SmallVector<L0Node, 16>> nodes =
      collectL0Nodes(loop, planNodes);
  if (!nodes) {
    return std::nullopt;
  }
  std::set<L0Slot> leftSlots;
  std::set<L0Slot> rightSlots;
  for (const L0Node &node : *nodes) {
    if (node.produced) {
      (node.produced->space == AddressSpace::LEFT ? leftSlots : rightSlots)
          .insert(*node.produced);
    }
  }
  if (leftSlots.size() != 2 || rightSlots.size() != 2) {
    return std::nullopt;
  }
  if (slotsOverlap(*leftSlots.begin(), *std::next(leftSlots.begin())) ||
      slotsOverlap(*rightSlots.begin(), *std::next(rightSlots.begin()))) {
    return std::nullopt;
  }

  Operation *commonChild = nodes->front().topLevelChild;
  const bool oneTopLevelChild = llvm::all_of(*nodes, [&](const L0Node &node) {
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
    for (L0Node &node : *nodes) {
      node.path = getPathRegion(node.operation, branch);
    }
  } else {
    pathRegions.push_back(&loop.getRegion());
    for (L0Node &node : *nodes) {
      node.path = &loop.getRegion();
    }
  }

  OwnershipPattern result;
  result.loop = loop;
  std::map<L0SlotPair, unsigned> pairLanes;
  SmallVector<unsigned, 8> expectedLaneOrder;
  for (auto [pathIndex, path] : llvm::enumerate(pathRegions)) {
    std::optional<OwnershipPath> parsed =
        parseOwnershipPath(path, *nodes, planNodes, pairLanes, pathIndex == 0);
    if (!parsed || parsed->uses.empty()) {
      return std::nullopt;
    }
    SmallVector<unsigned, 8> laneOrder;
    llvm::transform(parsed->uses, std::back_inserter(laneOrder),
                    [](const OwnershipUse &use) { return use.lane; });
    if (pathIndex == 0) {
      expectedLaneOrder = laneOrder;
    } else if (laneOrder != expectedLaneOrder) {
      return std::nullopt;
    }
    result.paths.push_back(std::move(*parsed));
  }
  if (pairLanes.size() != 2 || !llvm::is_contained(expectedLaneOrder, 0U) ||
      !llvm::is_contained(expectedLaneOrder, 1U)) {
    return std::nullopt;
  }
  return result;
}

unsigned addAction(CanonicalEvent &event, CanonicalEventActionKind kind,
                   CanonicalEventActionPhase phase, CanonicalAnchor anchor,
                   CanonicalEventLane lane) {
  event.actions.push_back({kind, phase, anchor, lane});
  return event.actions.size() - 1;
}

CanonicalEventLane staticLane(unsigned lane) {
  CanonicalEventLane result;
  result.kind = CanonicalEventLaneKind::Static;
  result.index = lane;
  return result;
}

CanonicalEventLane allLanes() {
  CanonicalEventLane result;
  result.kind = CanonicalEventLaneKind::All;
  return result;
}

void addTrace(CanonicalEvent &event, CanonicalEventTraceKind kind,
              ArrayRef<unsigned> actions, Region *region = nullptr) {
  CanonicalEventTrace trace;
  trace.kind = kind;
  trace.actions.append(actions.begin(), actions.end());
  trace.controlRegion = region;
  event.traces.push_back(std::move(trace));
}

struct OwnershipActionUse {
  unsigned wait = 0;
  unsigned set = 0;
};

std::pair<CanonicalEvent, CanonicalEvent>
buildOwnershipProtocols(const OwnershipPattern &pattern,
                        ArrayRef<CanonicalSyncNode> nodes) {
  CanonicalEvent ready;
  ready.sourcePipe = PipelineType::PIPE_MTE1;
  ready.targetPipe = PipelineType::PIPE_M;
  ready.scopeLoop = pattern.loop;
  ready.width = 2;
  ready.ownershipProtocol = true;

  for (const OwnershipPath &path : pattern.paths) {
    SmallVector<unsigned, 16> trace;
    for (const OwnershipUse &use : path.uses) {
      const std::size_t lastProducer =
          nodes[use.firstProducer].order > nodes[use.secondProducer].order
              ? use.firstProducer
              : use.secondProducer;
      const unsigned set = addAction(ready, CanonicalEventActionKind::Set,
                                     CanonicalEventActionPhase::Straight,
                                     {nodes[lastProducer].operation, false},
                                     staticLane(use.lane));
      const unsigned wait =
          addAction(ready, CanonicalEventActionKind::Wait,
                    CanonicalEventActionPhase::Straight,
                    {use.consumerAnchor, true}, staticLane(use.lane));
      trace.push_back(set);
      trace.push_back(wait);
      for (std::size_t producer : {use.firstProducer, use.secondProducer}) {
        for (std::size_t consumer : use.consumers) {
          ready.completions.push_back(
              {producer, consumer, 0, nullptr, set, wait});
        }
      }
    }
    addTrace(ready, CanonicalEventTraceKind::Straight, trace, path.region);
  }

  CanonicalEvent release;
  release.sourcePipe = PipelineType::PIPE_M;
  release.targetPipe = PipelineType::PIPE_MTE1;
  release.recurrenceLoop = pattern.loop;
  release.scopeLoop = pattern.loop;
  release.iterationDistance = 1;
  release.width = 2;
  release.ownershipProtocol = true;
  const unsigned prime = addAction(release, CanonicalEventActionKind::Set,
                                   CanonicalEventActionPhase::Prime,
                                   {pattern.loop, true}, allLanes());
  addTrace(release, CanonicalEventTraceKind::Prime, {prime});

  SmallVector<SmallVector<OwnershipActionUse, 8>, 2> actionPaths;
  for (const OwnershipPath &path : pattern.paths) {
    SmallVector<unsigned, 16> trace;
    SmallVector<OwnershipActionUse, 8> actionUses;
    std::optional<std::size_t> previousUse[2];
    for (auto [useIndex, use] : llvm::enumerate(path.uses)) {
      const std::size_t firstProducer =
          nodes[use.firstProducer].order < nodes[use.secondProducer].order
              ? use.firstProducer
              : use.secondProducer;
      const unsigned wait = addAction(release, CanonicalEventActionKind::Wait,
                                      CanonicalEventActionPhase::Body,
                                      {nodes[firstProducer].operation, true},
                                      staticLane(use.lane));
      const unsigned set =
          addAction(release, CanonicalEventActionKind::Set,
                    CanonicalEventActionPhase::Body,
                    {use.consumerAnchor, false}, staticLane(use.lane));
      trace.push_back(wait);
      trace.push_back(set);
      actionUses.push_back({wait, set});
      if (previousUse[use.lane]) {
        const OwnershipUse &previous = path.uses[*previousUse[use.lane]];
        const unsigned previousSet = actionUses[*previousUse[use.lane]].set;
        for (std::size_t consumer : previous.consumers) {
          for (std::size_t producer : {use.firstProducer, use.secondProducer}) {
            release.completions.push_back(
                {consumer, producer, 0, nullptr, previousSet, wait});
          }
        }
      }
      previousUse[use.lane] = useIndex;
    }
    addTrace(release, CanonicalEventTraceKind::Cycle, trace, path.region);
    actionPaths.push_back(std::move(actionUses));
  }

  for (auto [sourcePathIndex, sourcePath] : llvm::enumerate(pattern.paths)) {
    for (auto [targetPathIndex, targetPath] : llvm::enumerate(pattern.paths)) {
      for (unsigned lane = 0; lane < 2; ++lane) {
        std::size_t sourceUse = sourcePath.uses.size();
        std::size_t targetUse = targetPath.uses.size();
        for (std::size_t index = 0; index < sourcePath.uses.size(); ++index) {
          if (sourcePath.uses[index].lane == lane) {
            sourceUse = index;
          }
        }
        for (std::size_t index = 0; index < targetPath.uses.size(); ++index) {
          if (targetPath.uses[index].lane == lane) {
            targetUse = index;
            break;
          }
        }
        if (sourceUse == sourcePath.uses.size() ||
            targetUse == targetPath.uses.size()) {
          continue;
        }
        const OwnershipUse &source = sourcePath.uses[sourceUse];
        const OwnershipUse &target = targetPath.uses[targetUse];
        const unsigned set = actionPaths[sourcePathIndex][sourceUse].set;
        const unsigned wait = actionPaths[targetPathIndex][targetUse].wait;
        for (std::size_t consumer : source.consumers) {
          for (std::size_t producer :
               {target.firstProducer, target.secondProducer}) {
            release.completions.push_back(
                {consumer, producer, 1, pattern.loop, set, wait});
          }
        }
      }
    }
  }
  const unsigned drain = addAction(release, CanonicalEventActionKind::Wait,
                                   CanonicalEventActionPhase::Drain,
                                   {pattern.loop, false}, allLanes());
  addTrace(release, CanonicalEventTraceKind::Final, {drain});

  const auto setRepresentative =
      [](CanonicalEvent &event, const CanonicalEventCompletion &completion) {
        event.source = completion.source;
        event.target = completion.target;
        event.setAnchor = event.actions[completion.setAction].anchor;
        event.waitAnchor = event.actions[completion.waitAction].anchor;
      };
  setRepresentative(ready, ready.completions.front());
  auto recurrence = llvm::find_if(
      release.completions, [](const CanonicalEventCompletion &completion) {
        return completion.iterationDistance != 0;
      });
  setRepresentative(release, *recurrence);
  return {std::move(ready), std::move(release)};
}

} // namespace

void CanonicalSyncPlanBuilder::synthesizeL0OwnershipProtocols() {
  SmallVector<scf::ForOp, 4> loops;
  func_.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    std::optional<OwnershipPattern> pattern =
        recognizeOwnershipPattern(loop, plan_.nodes_);
    if (!pattern) {
      continue;
    }
    auto [ready, release] = buildOwnershipProtocols(*pattern, plan_.nodes_);
    deriveEventInterval(ready);
    deriveEventInterval(release);

    std::vector<CanonicalEvent> trial = plan_.events_;
    trial.push_back(ready);
    trial.push_back(release);
    if (failed(verifyEventProtocols({ready, release},
                                    /*requireAllocation=*/false,
                                    /*diagnose=*/false))) {
      continue;
    }
    std::vector<CanonicalEvent> selected =
        selectRequiredEvents(plan_.barriers_, trial);
    if (!eventsFitBudget(selected)) {
      continue;
    }
    plan_.events_.push_back(std::move(ready));
    plan_.events_.push_back(std::move(release));
  }
}
