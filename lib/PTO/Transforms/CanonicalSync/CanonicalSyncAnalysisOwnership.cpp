// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncAnalysisInternal.h"

#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include <vector>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

struct SlotKey {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval extent;

  friend bool operator<(const SlotKey &left, const SlotKey &right) {
    return std::tie(left.domain, left.extent.begin, left.extent.end) <
           std::tie(right.domain, right.extent.begin, right.extent.end);
  }

  friend bool operator==(const SlotKey &left, const SlotKey &right) {
    return left.domain == right.domain &&
           left.extent.begin == right.extent.begin &&
           left.extent.end == right.extent.end;
  }
};

using SlotBundle = std::vector<SlotKey>;
using StorageAccessIndex =
    std::vector<std::vector<const SyncCoverStorageAccess *>>;

struct OwnershipSpec {
  SyncCoverBasicOwnershipKind kind = SyncCoverBasicOwnershipKind::L0Operand;
  std::uint32_t producerResource = 0;
  std::uint32_t consumerResource = 0;
  std::vector<SyncCoverStorageDomainRole> roles;
  std::size_t requiredSlots = 1;
  std::size_t minimumLanes = 1;
  bool allowProducerReads = false;
};

struct OwnershipNode {
  SyncCoverNodeId id = 0;
  Operation *operation = nullptr;
  Operation *topLevel = nullptr;
  Region *path = nullptr;
  SlotBundle produced;
  SlotBundle consumed;
};

struct PathItem {
  bool producer = false;
  std::size_t order = 0;
  OwnershipNode *node = nullptr;
  Operation *consumerAnchor = nullptr;
  std::vector<OwnershipNode *> consumers;
};

struct SlotGroup {
  SlotKey slot;
  std::vector<OwnershipNode *> producers;
  std::vector<OwnershipNode *> consumers;
};

struct ParitySlotGroup {
  SlotKey slot;
  std::array<std::vector<OwnershipNode *>, 2> producers;
  std::array<std::vector<OwnershipNode *>, 2> consumers;
};

bool intervalsOverlap(SyncCoverStorageInterval first,
                      SyncCoverStorageInterval second) {
  return first.begin < second.end && second.begin < first.end;
}

Operation *getTopLevelChild(Operation *operation, Operation *owner) {
  Operation *child = operation;
  while (child && child->getParentOp() != owner) {
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

bool matchNonNegativeConstant(Value value, std::uint64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) &&
         constant.isNonNegative() && constant.getZExtValue() == expected;
}

class BasicOwnershipDiscovery {
public:
  BasicOwnershipDiscovery(
      SyncCoverGraph &graph,
      const std::vector<CanonicalSyncNodeBinding> &nodeBindings,
      const std::vector<CanonicalSyncScopeBinding> &scopeBindings,
      const std::vector<CanonicalSyncControlBinding> &controlBindings,
      const CanonicalSyncAnalysisOptions &options,
      const CanonicalSyncTargetCapabilities &capabilities)
      : graph_(graph), nodeBindings_(nodeBindings),
        scopeBindings_(scopeBindings), controlBindings_(controlBindings),
        options_(options), capabilities_(capabilities),
        accessesByNode_(graph.getNodes().size()) {
    for (const SyncCoverStorageAccess &access : graph_.getStorageAccesses()) {
      if (access.node < accessesByNode_.size()) {
        accessesByNode_[access.node].push_back(&access);
      }
    }
  }

  void run() {
    const OwnershipSpec l0{SyncCoverBasicOwnershipKind::L0Operand,
                           static_cast<std::uint32_t>(PipelineType::PIPE_MTE1),
                           static_cast<std::uint32_t>(PipelineType::PIPE_M),
                           {SyncCoverStorageDomainRole::L0Left,
                            SyncCoverStorageDomainRole::L0Right},
                           2,
                           2,
                           false};
    const OwnershipSpec l1{SyncCoverBasicOwnershipKind::L1Tile,
                           static_cast<std::uint32_t>(PipelineType::PIPE_MTE2),
                           static_cast<std::uint32_t>(PipelineType::PIPE_MTE1),
                           {SyncCoverStorageDomainRole::L1Tile},
                           1,
                           2,
                           false};
    const OwnershipSpec accumulator{
        SyncCoverBasicOwnershipKind::L0Accumulator,
        static_cast<std::uint32_t>(PipelineType::PIPE_M),
        static_cast<std::uint32_t>(PipelineType::PIPE_FIX),
        {SyncCoverStorageDomainRole::Accumulator},
        1,
        1,
        true};

    const std::size_t passCost =
        graph_.getNodes().size() + graph_.getStorageAccesses().size();
    if (!consume(passCost)) {
      return;
    }
    for (const SyncCoverScope &scope : graph_.getScopes()) {
      if (reachedCertificateLimit()) {
        statistics_.truncated = true;
        break;
      }
      if (!scope.isLoop) {
        continue;
      }
      if (capabilities_.mte1L0ReadySetCompletesPrefix &&
          capabilities_.mL0AlternativeJoinSetCompletes && consume(passCost)) {
        add(recognizeL0(scope.id, l0));
      }
      if (capabilities_.mte1ScopeExitSetCompletesPrefix && consume(passCost)) {
        add(recognizeHierarchical(scope.id, l1));
      }
      if (capabilities_.mte1ScopeExitSetCompletesPrefix && consume(passCost)) {
        for (SyncCoverBasicOwnershipCertificate certificate :
             recognizeParity(scope.id, l1)) {
          add(std::move(certificate));
        }
      }
      if (capabilities_.mToFixAccumulatorBoundaryCompletes &&
          consume(passCost)) {
        add(recognizeHierarchical(scope.id, accumulator));
      }
    }
    statistics_.inspections = inspections_;
  }

  const CanonicalSyncOwnershipDiscoveryStatistics &statistics() const {
    return statistics_;
  }

private:
  bool consume(std::size_t amount) {
    if (amount >
        options_.maximumBasicOwnershipInspections -
            std::min(inspections_, options_.maximumBasicOwnershipInspections)) {
      inspections_ = options_.maximumBasicOwnershipInspections;
      statistics_.inspections = inspections_;
      statistics_.truncated = true;
      return false;
    }
    inspections_ += amount;
    statistics_.inspections = inspections_;
    return true;
  }

  bool reachedCertificateLimit() const {
    return graph_.getBasicOwnershipCertificates().size() >=
           options_.maximumBasicOwnershipCertificates;
  }

  void add(std::optional<SyncCoverBasicOwnershipCertificate> certificate) {
    if (!certificate) {
      return;
    }
    if (reachedCertificateLimit()) {
      statistics_.truncated = true;
      return;
    }
    populateSlotAccesses(*certificate);
    const std::size_t kind = static_cast<std::size_t>(certificate->kind);
    if (graph_.addBasicOwnershipCertificate(std::move(*certificate)) &&
        kind < statistics_.certificatesByKind.size()) {
      ++statistics_.certificatesByKind[kind];
    }
  }

  void add(SyncCoverBasicOwnershipCertificate certificate) {
    add(std::optional<SyncCoverBasicOwnershipCertificate>(
        std::move(certificate)));
  }

  void
  populateSlotAccesses(SyncCoverBasicOwnershipCertificate &certificate) const {
    for (SyncCoverBasicOwnershipLane &lane : certificate.lanes) {
      for (SyncCoverBasicOwnershipSlot &slot : lane.slots) {
        for (const SyncCoverStorageAccess &access :
             graph_.getStorageAccesses()) {
          const bool relevant =
              graph_.scopeContains(certificate.loopScope,
                                   graph_.getNodes()[access.node].scope) ||
              llvm::is_contained(certificate.initialProducers, access.node);
          if (relevant && access.domain == slot.domain &&
              access.extent.begin == slot.extent.begin &&
              access.extent.end == slot.extent.end) {
            slot.accesses.push_back(access.id);
          }
        }
      }
    }
  }

  std::optional<SyncCoverScopeId> findScope(Region *region) const {
    for (auto [index, binding] : llvm::enumerate(scopeBindings_)) {
      if (binding.region == region) {
        return index;
      }
    }
    return std::nullopt;
  }

  std::optional<SyncCoverControlId> findControl(Operation *owner) const {
    for (auto [index, binding] : llvm::enumerate(controlBindings_)) {
      if (binding.owner == owner) {
        return index;
      }
    }
    return std::nullopt;
  }

  bool roleMatches(const OwnershipSpec &spec,
                   const SyncCoverStorageAccess &access) const {
    return access.domain < graph_.getStorageDomains().size() &&
           llvm::is_contained(spec.roles,
                              graph_.getStorageDomains()[access.domain].role);
  }

  std::optional<std::vector<OwnershipNode>>
  collectNodes(SyncCoverScopeId loopScope, const OwnershipSpec &spec,
               bool oneAccessPerNode) const {
    if (loopScope >= scopeBindings_.size() ||
        !isa_and_nonnull<scf::ForOp>(scopeBindings_[loopScope].owner)) {
      return std::nullopt;
    }
    Operation *loop = scopeBindings_[loopScope].owner;
    std::vector<OwnershipNode> result;
    for (const SyncCoverNode &node : graph_.getNodes()) {
      if (!graph_.scopeContains(loopScope, node.scope)) {
        continue;
      }
      std::vector<const SyncCoverStorageAccess *> accesses;
      for (const SyncCoverStorageAccess *access : accessesByNode_[node.id]) {
        if (!roleMatches(spec, *access)) {
          continue;
        }
        if (!access->exactPhysical) {
          return std::nullopt;
        }
        accesses.push_back(access);
      }
      if (accesses.empty()) {
        continue;
      }
      if (oneAccessPerNode && accesses.size() != 1) {
        return std::nullopt;
      }

      OwnershipNode ownership;
      ownership.id = node.id;
      ownership.operation = nodeBindings_[node.id].operation;
      ownership.topLevel = getTopLevelChild(ownership.operation, loop);
      if (!ownership.operation || !ownership.topLevel) {
        return std::nullopt;
      }
      if (node.resource == spec.producerResource) {
        for (const SyncCoverStorageAccess *access : accesses) {
          const bool writes = syncCoverStorageModeWrites(access->mode);
          const bool reads = syncCoverStorageModeReads(access->mode);
          if (!writes || (reads && !spec.allowProducerReads)) {
            return std::nullopt;
          }
          ownership.produced.push_back({access->domain, access->extent});
        }
        llvm::sort(ownership.produced);
      } else if (node.resource == spec.consumerResource) {
        for (const SyncCoverStorageAccess *access : accesses) {
          if (access->mode != SyncCoverStorageAccessMode::Read) {
            return std::nullopt;
          }
          ownership.consumed.push_back({access->domain, access->extent});
        }
        llvm::sort(ownership.consumed);
        if (ownership.consumed.size() != spec.requiredSlots) {
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
                             ArrayRef<OwnershipNode *> consumers) const {
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

  std::optional<SyncCoverBasicOwnershipPath>
  parseL0Path(Region *path, MutableArrayRef<OwnershipNode> nodes,
              const OwnershipSpec &spec,
              std::map<SlotBundle, std::size_t> &bundleLanes,
              bool allowNewBundles) const {
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
            {true, graph_.getNodes()[node.id].order, &node, nullptr, {}});
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
      OwnershipNode *first = *llvm::min_element(
          consumers, [&](OwnershipNode *left, OwnershipNode *right) {
            return graph_.getNodes()[left->id].order <
                   graph_.getNodes()[right->id].order;
          });
      items.push_back({false, graph_.getNodes()[first->id].order, nullptr,
                       anchor, consumers});
    }
    llvm::stable_sort(items, [](const PathItem &left, const PathItem &right) {
      return left.order < right.order;
    });

    const std::optional<SyncCoverScopeId> pathScope = findScope(path);
    if (!pathScope) {
      return std::nullopt;
    }
    SyncCoverBasicOwnershipPath result;
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
      if (produced.size() != spec.requiredSlots ||
          produced != item.consumers.front()->consumed) {
        return std::nullopt;
      }
      auto lane = bundleLanes.find(produced);
      if (lane == bundleLanes.end()) {
        if (!allowNewBundles) {
          return std::nullopt;
        }
        lane = bundleLanes.emplace(produced, bundleLanes.size()).first;
      }
      sortNodes(producers);
      sortNodes(item.consumers);
      SyncCoverBasicOwnershipUse use;
      use.lane = lane->second;
      use.producerLane = use.lane;
      llvm::transform(producers, std::back_inserter(use.producers),
                      [](OwnershipNode *node) { return node->id; });
      llvm::transform(item.consumers, std::back_inserter(use.consumers),
                      [](OwnershipNode *node) { return node->id; });
      use.writeAcquireAnchor = {SyncCoverAnchorKind::BeforeNode,
                                producers.front()->id, 0, 0};
      use.readyAnchor = {SyncCoverAnchorKind::AfterNode, producers.back()->id,
                         0, 0};
      if (auto consumerBranch = dyn_cast<scf::IfOp>(item.consumerAnchor)) {
        const std::optional<SyncCoverControlId> control =
            findControl(consumerBranch.getOperation());
        if (!control || graph_.getControls()[*control].scope != *pathScope) {
          return std::nullopt;
        }
        use.readAcquireAnchor = {SyncCoverAnchorKind::ControlEntry, *control,
                                 *pathScope, 0};
        use.releaseAnchor = {SyncCoverAnchorKind::ControlExit, *control,
                             *pathScope, 0};
      } else {
        use.readAcquireAnchor = {SyncCoverAnchorKind::BeforeNode,
                                 item.consumers.front()->id, 0, 0};
        use.releaseAnchor = {SyncCoverAnchorKind::AfterNode,
                             item.consumers.back()->id, 0, 0};
      }
      result.uses.push_back(std::move(use));
      producers.clear();
    }
    return producers.empty() && !result.uses.empty()
               ? std::optional<SyncCoverBasicOwnershipPath>(std::move(result))
               : std::nullopt;
  }

  std::optional<SyncCoverBasicOwnershipCertificate>
  recognizeL0(SyncCoverScopeId loopScope, const OwnershipSpec &spec) const {
    std::optional<std::vector<OwnershipNode>> nodes =
        collectNodes(loopScope, spec, false);
    if (!nodes) {
      return std::nullopt;
    }
    Operation *loop = scopeBindings_[loopScope].owner;
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

    SyncCoverBasicOwnershipCertificate certificate;
    certificate.kind = spec.kind;
    certificate.loopScope = loopScope;
    certificate.producerResource = spec.producerResource;
    certificate.consumerResource = spec.consumerResource;
    std::map<SlotBundle, std::size_t> lanes;
    for (auto [index, path] : llvm::enumerate(paths)) {
      std::optional<SyncCoverBasicOwnershipPath> parsed =
          parseL0Path(path, *nodes, spec, lanes, index == 0);
      if (!parsed) {
        return std::nullopt;
      }
      certificate.paths.push_back(std::move(*parsed));
    }
    if (lanes.size() < spec.minimumLanes) {
      return std::nullopt;
    }
    certificate.lanes.resize(lanes.size());
    for (const auto &[slots, laneId] : lanes) {
      certificate.lanes[laneId].id = laneId;
      for (const SlotKey &slot : slots) {
        certificate.lanes[laneId].slots.push_back(
            {slot.domain, slot.extent, {}});
      }
    }
    if (!slotsAreDisjoint(certificate.lanes)) {
      return std::nullopt;
    }
    for (const SyncCoverBasicOwnershipPath &path : certificate.paths) {
      std::set<std::size_t> used;
      for (const SyncCoverBasicOwnershipUse &use : path.uses) {
        used.insert(use.lane);
      }
      if (used.size() != certificate.lanes.size()) {
        return std::nullopt;
      }
    }
    return certificate;
  }

  Operation *commonTopLevel(ArrayRef<OwnershipNode *> nodes) const {
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
  boundaryAnchors(Operation *anchor) const {
    if (!anchor) {
      return std::nullopt;
    }
    std::vector<SyncCoverNodeId> nodes;
    for (auto [node, binding] : llvm::enumerate(nodeBindings_)) {
      if (binding.operation == anchor) {
        nodes.push_back(node);
      }
    }
    if (!nodes.empty()) {
      llvm::sort(nodes, [&](SyncCoverNodeId left, SyncCoverNodeId right) {
        return graph_.getNodes()[left].order < graph_.getNodes()[right].order;
      });
      return std::make_pair(
          SyncCoverAnchor{SyncCoverAnchorKind::BeforeNode, nodes.front(), 0, 0},
          SyncCoverAnchor{SyncCoverAnchorKind::AfterNode,
                          graph_.getNodes()[nodes.back()].physicalExit, 0, 0});
    }
    auto loop = dyn_cast<scf::ForOp>(anchor);
    if (!loop) {
      return std::nullopt;
    }
    const std::optional<SyncCoverScopeId> scope = findScope(&loop.getRegion());
    if (!scope) {
      return std::nullopt;
    }
    return std::make_pair(
        SyncCoverAnchor{SyncCoverAnchorKind::ScopeEntry, 0, *scope, 0},
        SyncCoverAnchor{SyncCoverAnchorKind::ScopeExit, 0, *scope, 0});
  }

  std::optional<SyncCoverBasicOwnershipCertificate>
  recognizeHierarchical(SyncCoverScopeId loopScope,
                        const OwnershipSpec &spec) const {
    std::optional<std::vector<OwnershipNode>> nodes =
        collectNodes(loopScope, spec, true);
    auto loop = dyn_cast_or_null<scf::ForOp>(scopeBindings_[loopScope].owner);
    if (!nodes || !loop) {
      return std::nullopt;
    }
    std::map<SlotKey, SlotGroup> groups;
    for (OwnershipNode &node : *nodes) {
      const SlotKey &slot = !node.produced.empty() ? node.produced.front()
                                                   : node.consumed.front();
      SlotGroup &group = groups[slot];
      group.slot = slot;
      (!node.produced.empty() ? group.producers : group.consumers)
          .push_back(&node);
    }
    if (groups.size() < spec.minimumLanes || !groupsAreDisjoint(groups)) {
      return std::nullopt;
    }
    std::vector<SlotGroup *> ordered;
    for (auto &[slot, group] : groups) {
      (void)slot;
      const bool validProducers =
          spec.kind == SyncCoverBasicOwnershipKind::L1Tile
              ? group.producers.size() == 1 &&
                    group.producers.front()->operation->getParentRegion() ==
                        &loop.getRegion()
              : !group.producers.empty();
      if (!validProducers || group.consumers.empty()) {
        return std::nullopt;
      }
      Operation *producerAnchor = commonTopLevel(group.producers);
      Operation *consumerAnchor = commonTopLevel(group.consumers);
      const bool orderedInOneBlock =
          consumerAnchor && producerAnchor != consumerAnchor &&
          producerAnchor->getBlock() == consumerAnchor->getBlock() &&
          producerAnchor->isBeforeInBlock(consumerAnchor);
      if (!orderedInOneBlock || !boundaryAnchors(producerAnchor) ||
          !boundaryAnchors(consumerAnchor)) {
        return std::nullopt;
      }
      ordered.push_back(&group);
    }
    llvm::stable_sort(ordered, [&](SlotGroup *left, SlotGroup *right) {
      return graph_.getNodes()[left->producers.front()->id].order <
             graph_.getNodes()[right->producers.front()->id].order;
    });

    SyncCoverBasicOwnershipCertificate certificate;
    certificate.kind = spec.kind;
    certificate.loopScope = loopScope;
    certificate.producerResource = spec.producerResource;
    certificate.consumerResource = spec.consumerResource;
    SyncCoverBasicOwnershipPath path;
    path.scope = loopScope;
    for (auto [lane, group] : llvm::enumerate(ordered)) {
      Operation *producerAnchor = commonTopLevel(group->producers);
      Operation *consumerAnchor = commonTopLevel(group->consumers);
      const auto producerBounds = boundaryAnchors(producerAnchor);
      const auto consumerBounds = boundaryAnchors(consumerAnchor);
      if (!producerBounds || !consumerBounds) {
        return std::nullopt;
      }
      certificate.lanes.push_back(
          {lane, {{group->slot.domain, group->slot.extent, {}}}});
      sortNodes(group->producers);
      sortNodes(group->consumers);
      SyncCoverBasicOwnershipUse use;
      use.lane = lane;
      use.producerLane = lane;
      llvm::transform(group->producers, std::back_inserter(use.producers),
                      [](OwnershipNode *node) { return node->id; });
      llvm::transform(group->consumers, std::back_inserter(use.consumers),
                      [](OwnershipNode *node) { return node->id; });
      use.writeAcquireAnchor = producerBounds->first;
      use.readyAnchor = producerBounds->second;
      use.readAcquireAnchor = consumerBounds->first;
      use.releaseAnchor = consumerBounds->second;
      path.uses.push_back(std::move(use));
    }
    certificate.paths.push_back(std::move(path));
    return certificate;
  }

  bool matchParityBranch(scf::ForOp loop, scf::IfOp branch) const {
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

  bool matchNextIteration(Value value, scf::ForOp loop) const {
    auto add = value.getDefiningOp<arith::AddIOp>();
    return add && ((add.getLhs() == loop.getInductionVar() &&
                    add.getRhs() == loop.getStep()) ||
                   (add.getRhs() == loop.getInductionVar() &&
                    add.getLhs() == loop.getStep()));
  }

  bool hasContinuationGuard(Operation *operation, scf::ForOp loop,
                            Region *path) const {
    auto guard = dyn_cast_or_null<scf::IfOp>(operation->getParentOp());
    const bool direct =
        guard && path && guard->getParentRegion() == path &&
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

  bool executeDirectlyIn(ArrayRef<OwnershipNode *> nodes,
                         Region *region) const {
    return llvm::all_of(nodes, [&](const OwnershipNode *node) {
      return node->operation->getParentRegion() == region;
    });
  }

  OwnershipNode *firstByOrder(ArrayRef<OwnershipNode *> nodes) const {
    return *llvm::min_element(nodes,
                              [&](OwnershipNode *left, OwnershipNode *right) {
                                return graph_.getNodes()[left->id].order <
                                       graph_.getNodes()[right->id].order;
                              });
  }

  OwnershipNode *lastByOrder(ArrayRef<OwnershipNode *> nodes) const {
    return *llvm::max_element(nodes,
                              [&](OwnershipNode *left, OwnershipNode *right) {
                                return graph_.getNodes()[left->id].order <
                                       graph_.getNodes()[right->id].order;
                              });
  }

  SyncCoverBasicOwnershipUse
  makeParityUse(std::size_t lane, std::size_t producerLane,
                ArrayRef<OwnershipNode *> producers,
                ArrayRef<OwnershipNode *> consumers) const {
    SyncCoverBasicOwnershipUse use;
    use.lane = lane;
    use.producerLane = producerLane;
    llvm::transform(producers, std::back_inserter(use.producers),
                    [](OwnershipNode *node) { return node->id; });
    llvm::transform(consumers, std::back_inserter(use.consumers),
                    [](OwnershipNode *node) { return node->id; });
    OwnershipNode *firstProducer = firstByOrder(producers);
    OwnershipNode *lastProducer = lastByOrder(producers);
    OwnershipNode *firstConsumer = firstByOrder(consumers);
    OwnershipNode *lastConsumer = lastByOrder(consumers);
    use.writeAcquireAnchor = {SyncCoverAnchorKind::BeforeNode,
                              firstProducer->id, 0, 0};
    use.readyAnchor = {SyncCoverAnchorKind::AfterNode, lastProducer->id, 0, 0};
    use.readAcquireAnchor = {SyncCoverAnchorKind::BeforeNode, firstConsumer->id,
                             0, 0};
    use.releaseAnchor = {SyncCoverAnchorKind::AfterNode, lastConsumer->id, 0,
                         0};
    return use;
  }

  bool isBeforeLoop(Operation *operation, scf::ForOp loop) const {
    Operation *topLevel = getTopLevelInBlock(operation, loop->getBlock());
    return topLevel && topLevel != loop && topLevel->isBeforeInBlock(loop);
  }

  bool hasInterveningResourceNode(SyncCoverNodeId source, scf::ForOp loop,
                                  std::uint32_t resource) const {
    Operation *sourceOperation = nodeBindings_[source].operation;
    if (!sourceOperation || !isBeforeLoop(sourceOperation, loop)) {
      return true;
    }
    for (const SyncCoverNode &node : graph_.getNodes()) {
      if (node.id == source || node.resource != resource) {
        continue;
      }
      Operation *operation = nodeBindings_[node.id].operation;
      Operation *topLevel = getTopLevelInBlock(operation, loop->getBlock());
      if (topLevel && topLevel != loop &&
          sourceOperation->isBeforeInBlock(topLevel) &&
          topLevel->isBeforeInBlock(loop)) {
        return true;
      }
    }
    return false;
  }

  std::optional<SyncCoverNodeId>
  findInitialProducer(scf::ForOp loop, const SlotKey &slot,
                      ArrayRef<SlotKey> managedSlots,
                      std::uint32_t producerResource) const {
    std::optional<SyncCoverNodeId> result;
    for (const SyncCoverNode &node : graph_.getNodes()) {
      Operation *operation = nodeBindings_[node.id].operation;
      Operation *topLevel = getTopLevelInBlock(operation, loop->getBlock());
      if (!topLevel || topLevel == loop || !topLevel->isBeforeInBlock(loop)) {
        continue;
      }
      bool touchesManaged = false;
      bool initialWrite = false;
      bool conflicting = false;
      for (const SyncCoverStorageAccess *access : accessesByNode_[node.id]) {
        if (access->domain >= graph_.getStorageDomains().size() ||
            graph_.getStorageDomains()[access->domain].role !=
                SyncCoverStorageDomainRole::L1Tile) {
          continue;
        }
        if (!access->exactPhysical) {
          return std::nullopt;
        }
        const SlotKey candidate{access->domain, access->extent};
        const bool overlapsManaged =
            llvm::any_of(managedSlots, [&](const SlotKey &managed) {
              return candidate.domain == managed.domain &&
                     intervalsOverlap(candidate.extent, managed.extent);
            });
        if (!overlapsManaged) {
          continue;
        }
        touchesManaged = true;
        const bool valid = operation == topLevel &&
                           node.resource == producerResource &&
                           access->mode == SyncCoverStorageAccessMode::Write &&
                           candidate == slot;
        conflicting |= initialWrite || !valid;
        initialWrite |= valid;
      }
      if (!touchesManaged) {
        continue;
      }
      if (!initialWrite || conflicting || result) {
        return std::nullopt;
      }
      result = node.id;
    }
    return result;
  }

  std::vector<SyncCoverBasicOwnershipCertificate>
  recognizeParity(SyncCoverScopeId loopScope, const OwnershipSpec &spec) const {
    std::vector<SyncCoverBasicOwnershipCertificate> result;
    std::optional<std::vector<OwnershipNode>> nodes =
        collectNodes(loopScope, spec, true);
    auto loop = dyn_cast_or_null<scf::ForOp>(scopeBindings_[loopScope].owner);
    if (!nodes || !loop) {
      return result;
    }
    Operation *commonChild = nodes->front().topLevel;
    const bool oneChild = llvm::all_of(*nodes, [&](const OwnershipNode &node) {
      return node.topLevel == commonChild;
    });
    scf::IfOp branch =
        oneChild ? dyn_cast_or_null<scf::IfOp>(commonChild) : scf::IfOp{};
    if (!matchParityBranch(loop, branch)) {
      return result;
    }
    Region *paths[] = {&branch.getThenRegion(), &branch.getElseRegion()};
    std::map<SlotKey, ParitySlotGroup> groups;
    for (OwnershipNode &node : *nodes) {
      node.path = getPathRegion(node.operation, branch);
      const unsigned path = node.path == paths[0]   ? 0
                            : node.path == paths[1] ? 1
                                                    : 2;
      if (path == 2) {
        return {};
      }
      const SlotKey &slot = !node.produced.empty() ? node.produced.front()
                                                   : node.consumed.front();
      ParitySlotGroup &group = groups[slot];
      group.slot = slot;
      (!node.produced.empty() ? group.producers[path] : group.consumers[path])
          .push_back(&node);
    }
    if (!groupsAreDisjoint(groups)) {
      return {};
    }

    std::vector<ParitySlotGroup *> stableGroups;
    std::vector<ParitySlotGroup *> alternatingGroups;
    for (auto &[slot, group] : groups) {
      (void)slot;
      const bool stable =
          llvm::all_of(llvm::seq<unsigned>(0, 2), [&](unsigned path) {
            return group.producers[path].size() == 1 &&
                   !group.consumers[path].empty() &&
                   executeDirectlyIn(group.producers[path], paths[path]) &&
                   executeDirectlyIn(group.consumers[path], paths[path]) &&
                   graph_.getNodes()[group.producers[path].front()->id].order <
                       graph_
                           .getNodes()[firstByOrder(group.consumers[path])->id]
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

    if (stableGroups.size() >= spec.minimumLanes) {
      SyncCoverBasicOwnershipCertificate stable;
      stable.kind = SyncCoverBasicOwnershipKind::L1Tile;
      stable.loopScope = loopScope;
      stable.producerResource = spec.producerResource;
      stable.consumerResource = spec.consumerResource;
      stable.paths.resize(2);
      for (unsigned path = 0; path < 2; ++path) {
        const std::optional<SyncCoverScopeId> scope = findScope(paths[path]);
        if (!scope) {
          return {};
        }
        stable.paths[path].scope = *scope;
      }
      for (auto [lane, group] : llvm::enumerate(stableGroups)) {
        stable.lanes.push_back(
            {lane, {{group->slot.domain, group->slot.extent, {}}}});
        for (unsigned path = 0; path < 2; ++path) {
          SyncCoverBasicOwnershipUse use = makeParityUse(
              lane, lane, group->producers[path], group->consumers[path]);
          stable.paths[path].uses.push_back(std::move(use));
        }
      }
      result.push_back(std::move(stable));
    }

    if (alternatingGroups.size() != 2) {
      return result;
    }
    SyncCoverBasicOwnershipCertificate prefetch;
    prefetch.kind = SyncCoverBasicOwnershipKind::L1Tile;
    prefetch.protocol =
        SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch;
    prefetch.loopScope = loopScope;
    prefetch.producerResource = spec.producerResource;
    prefetch.consumerResource = spec.consumerResource;
    prefetch.periodicControl = findControl(branch.getOperation());
    prefetch.paths.resize(2);
    if (!prefetch.periodicControl) {
      return result;
    }
    for (unsigned path = 0; path < 2; ++path) {
      const std::optional<SyncCoverScopeId> scope = findScope(paths[path]);
      if (!scope) {
        return result;
      }
      prefetch.paths[path].scope = *scope;
    }
    for (auto [lane, group] : llvm::enumerate(alternatingGroups)) {
      prefetch.lanes.push_back(
          {lane, {{group->slot.domain, group->slot.extent, {}}}});
    }
    for (unsigned path = 0; path < 2; ++path) {
      ParitySlotGroup *consumerGroup = nullptr;
      ParitySlotGroup *producerGroup = nullptr;
      std::size_t consumerLane = 0;
      std::size_t producerLane = 0;
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
          !hasContinuationGuard(
              producerGroup->producers[path].front()->operation, loop,
              paths[path])) {
        return result;
      }
      SyncCoverBasicOwnershipUse use = makeParityUse(
          consumerLane, producerLane, producerGroup->producers[path],
          consumerGroup->consumers[path]);
      prefetch.paths[path].uses.push_back(std::move(use));
    }
    const SyncCoverBasicOwnershipUse &first =
        prefetch.paths.front().uses.front();
    std::vector<SlotKey> managedSlots;
    for (const SyncCoverBasicOwnershipLane &lane : prefetch.lanes) {
      for (const SyncCoverBasicOwnershipSlot &slot : lane.slots) {
        managedSlots.push_back({slot.domain, slot.extent});
      }
    }
    const std::optional<SyncCoverNodeId> initial = findInitialProducer(
        loop, managedSlots[first.lane], managedSlots, spec.producerResource);
    if (!initial ||
        hasInterveningResourceNode(*initial, loop, spec.producerResource)) {
      return result;
    }
    prefetch.initialProducers.push_back(*initial);
    prefetch.initialWriteAcquireAnchor = {SyncCoverAnchorKind::BeforeNode,
                                          *initial, 0, 0};
    prefetch.initialReadyAnchor = {SyncCoverAnchorKind::ScopeEntry, 0,
                                   loopScope, 0};
    prefetch.initialReadyLane = first.lane;
    prefetch.initiallyFreeLanes.push_back(first.producerLane);
    result.push_back(std::move(prefetch));
    return result;
  }

  template <typename GroupMap>
  bool groupsAreDisjoint(const GroupMap &groups) const {
    for (auto first = groups.begin(); first != groups.end(); ++first) {
      for (auto second = std::next(first); second != groups.end(); ++second) {
        if (first->first.domain == second->first.domain &&
            intervalsOverlap(first->first.extent, second->first.extent)) {
          return false;
        }
      }
    }
    return true;
  }

  bool slotsAreDisjoint(ArrayRef<SyncCoverBasicOwnershipLane> lanes) const {
    for (std::size_t first = 0; first < lanes.size(); ++first) {
      for (std::size_t second = first + 1; second < lanes.size(); ++second) {
        for (const SyncCoverBasicOwnershipSlot &left : lanes[first].slots) {
          for (const SyncCoverBasicOwnershipSlot &right : lanes[second].slots) {
            if (left.domain == right.domain &&
                intervalsOverlap(left.extent, right.extent)) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }

  void sortNodes(std::vector<OwnershipNode *> &nodes) const {
    llvm::sort(nodes, [&](OwnershipNode *left, OwnershipNode *right) {
      return graph_.getNodes()[left->id].order <
             graph_.getNodes()[right->id].order;
    });
  }

  SyncCoverGraph &graph_;
  const std::vector<CanonicalSyncNodeBinding> &nodeBindings_;
  const std::vector<CanonicalSyncScopeBinding> &scopeBindings_;
  const std::vector<CanonicalSyncControlBinding> &controlBindings_;
  const CanonicalSyncAnalysisOptions &options_;
  const CanonicalSyncTargetCapabilities &capabilities_;
  StorageAccessIndex accessesByNode_;
  std::size_t inspections_ = 0;
  CanonicalSyncOwnershipDiscoveryStatistics statistics_;
};

} // namespace

LogicalResult ProgramBuilder::discoverBasicOwnershipCertificates(
    const CanonicalSyncTargetCapabilities &capabilities) {
  BasicOwnershipDiscovery discovery(graph_, nodeBindings_, scopeBindings_,
                                    controlBindings_, options_, capabilities);
  discovery.run();
  ownershipDiscoveryStatistics_ = discovery.statistics();
  return success();
}
