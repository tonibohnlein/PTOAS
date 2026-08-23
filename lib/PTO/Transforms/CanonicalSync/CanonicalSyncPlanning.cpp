// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct DependencyGroupKey {
  std::size_t source = 0;
  std::size_t target = 0;
  unsigned distance = 0;
  Operation *owner = nullptr;
  std::size_t ownerOrder = 0;

  bool operator<(const DependencyGroupKey &other) const {
    return std::tie(source, target, distance, ownerOrder) <
           std::tie(other.source, other.target, other.distance,
                    other.ownerOrder);
  }
};

std::map<DependencyGroupKey, SmallVector<std::size_t, 3>>
groupDependencies(ArrayRef<CanonicalDependency> dependencies) {
  std::map<DependencyGroupKey, SmallVector<std::size_t, 3>> groups;
  DenseMap<Operation *, std::size_t> ownerOrders;
  std::size_t nextOwnerOrder = 1;
  for (auto [index, dependency] : llvm::enumerate(dependencies)) {
    std::size_t ownerOrder = 0;
    if (dependency.recurrenceLoop) {
      auto [ownerIt, inserted] =
          ownerOrders.try_emplace(dependency.recurrenceLoop, nextOwnerOrder);
      ownerOrder = ownerIt->second;
      if (inserted) {
        ++nextOwnerOrder;
      }
    }
    groups[{dependency.source, dependency.target, dependency.iterationDistance,
            dependency.recurrenceLoop, ownerOrder}]
        .push_back(index);
  }
  return groups;
}

unsigned getLoopDepth(Operation *loop) {
  unsigned depth = 0;
  for (Operation *parent = loop; parent; parent = parent->getParentOp()) {
    if (isa<LoopLikeOpInterface>(parent)) {
      ++depth;
    }
  }
  return depth;
}

bool isStructuralContainer(Operation *op) {
  return isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op) ||
         op->getName().getStringRef().starts_with("pto.section.");
}

Operation *getWhileForwardDrain(Operation *source, Operation *target) {
  for (Operation *parent = source->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto whileOp = dyn_cast<scf::WhileOp>(parent);
    if (!whileOp || !whileOp->isAncestor(target)) {
      continue;
    }
    Region *sourceRegion = nullptr;
    Region *targetRegion = nullptr;
    for (Operation *current = source; current && current != whileOp;
         current = current->getParentOp()) {
      if (current->getParentOp() == whileOp) {
        sourceRegion = current->getParentRegion();
      }
    }
    for (Operation *current = target; current && current != whileOp;
         current = current->getParentOp()) {
      if (current->getParentOp() == whileOp) {
        targetRegion = current->getParentRegion();
      }
    }
    if (sourceRegion == &whileOp.getBefore() &&
        targetRegion == &whileOp.getAfter()) {
      return whileOp;
    }
  }
  return nullptr;
}

struct ExecutionCondition {
  Region *region = nullptr;
  unsigned occurrence = 0;
};

bool hasStaticallyPositiveTripCount(scf::ForOp forOp) {
  APInt lower;
  APInt upper;
  APInt step;
  return matchPattern(forOp.getLowerBound(), m_ConstantInt(&lower)) &&
         matchPattern(forOp.getUpperBound(), m_ConstantInt(&upper)) &&
         matchPattern(forOp.getStep(), m_ConstantInt(&step)) &&
         step.isStrictlyPositive() && lower.slt(upper);
}

std::optional<bool> canRecurrenceOccur(scf::ForOp forOp,
                                       unsigned iterationDistance) {
  APInt lower;
  APInt upper;
  APInt step;
  if (iterationDistance == 0 ||
      !matchPattern(forOp.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(forOp.getUpperBound(), m_ConstantInt(&upper)) ||
      !matchPattern(forOp.getStep(), m_ConstantInt(&step)) ||
      !step.isStrictlyPositive()) {
    return std::nullopt;
  }

  const unsigned operandWidth =
      std::max({lower.getBitWidth(), upper.getBitWidth(), step.getBitWidth()});
  if (operandWidth > std::numeric_limits<unsigned>::max() - 64U) {
    return std::nullopt;
  }
  const unsigned calculationWidth = operandWidth + 64U;
  const APInt extendedLower = lower.sext(calculationWidth);
  const APInt extendedUpper = upper.sext(calculationWidth);
  const APInt extendedStep = step.sext(calculationWidth);
  const APInt distance(calculationWidth, iterationDistance);
  const APInt targetOccurrence = extendedLower + extendedStep * distance;
  if (!targetOccurrence.isSignedIntN(operandWidth)) {
    return std::nullopt;
  }
  return targetOccurrence.slt(extendedUpper);
}

SmallVector<ExecutionCondition, 4>
getExecutionConditions(Operation *operation, unsigned occurrence,
                       Operation *ignoredLoop = nullptr,
                       bool usePositiveTripFacts = false) {
  SmallVector<ExecutionCondition, 4> conditions;
  for (Operation *current = operation; current;
       current = current->getParentOp()) {
    Operation *parent = current->getParentOp();
    if (!parent) {
      break;
    }
    const auto forOp = dyn_cast<scf::ForOp>(parent);
    const bool isGuaranteedLoop =
        usePositiveTripFacts && forOp && hasStaticallyPositiveTripCount(forOp);
    if (parent != ignoredLoop &&
        (isa<scf::IfOp>(parent) ||
         (isa<LoopLikeOpInterface>(parent) && !isGuaranteedLoop))) {
      conditions.push_back({current->getParentRegion(), occurrence});
    }
  }
  return conditions;
}

bool containsCondition(ArrayRef<ExecutionCondition> conditions,
                       const ExecutionCondition &candidate) {
  return llvm::any_of(conditions, [&](const ExecutionCondition &condition) {
    return condition.region == candidate.region &&
           condition.occurrence == candidate.occurrence;
  });
}

bool isGuaranteedInContext(Operation *operation, unsigned occurrence,
                           Operation *source, unsigned sourceOccurrence,
                           Operation *target, unsigned targetOccurrence,
                           Operation *ignoredLoop = nullptr,
                           bool usePositiveTripFacts = false) {
  const auto operationConditions = getExecutionConditions(
      operation, occurrence, ignoredLoop, usePositiveTripFacts);
  const auto sourceConditions = getExecutionConditions(
      source, sourceOccurrence, ignoredLoop, usePositiveTripFacts);
  const auto targetConditions = getExecutionConditions(
      target, targetOccurrence, ignoredLoop, usePositiveTripFacts);
  return llvm::all_of(operationConditions,
                      [&](const ExecutionCondition &condition) {
                        return containsCondition(sourceConditions, condition) ||
                               containsCondition(targetConditions, condition);
                      });
}

CanonicalAnchor getLoopEntryAnchor(Operation *loop) {
  if (loop->getNumRegions() == 0 || loop->getRegion(0).empty() ||
      loop->getRegion(0).front().empty()) {
    return {loop, true};
  }
  return {&loop->getRegion(0).front().front(), true};
}

CanonicalAnchor getLoopRegionExitAnchor(Operation *operation, Operation *loop) {
  Operation *child = operation;
  while (child && child->getParentOp() != loop) {
    child = child->getParentOp();
  }
  if (!child || !child->getBlock() || !child->getBlock()->getTerminator()) {
    return {operation, false};
  }
  return {child->getBlock()->getTerminator(), true};
}

bool formsMemoryHazard(CanonicalDependencyKind kind,
                       const CanonicalMemoryAccess &source,
                       const CanonicalMemoryAccess &target) {
  switch (kind) {
  case CanonicalDependencyKind::MemoryRAW:
    return source.writes && target.reads;
  case CanonicalDependencyKind::MemoryWAR:
    return source.reads && target.writes;
  case CanonicalDependencyKind::MemoryWAW:
    return source.writes && target.writes;
  case CanonicalDependencyKind::SSA:
  case CanonicalDependencyKind::LoopCarriedSSA:
    return false;
  }
  return false;
}

} // namespace

void CanonicalSyncPlanBuilder::discardImpossibleRecurrences() {
  for (CanonicalDependency &dependency : plan_.dependencies_) {
    if (dependency.iterationDistance == 0) {
      continue;
    }
    auto forOp = dyn_cast_or_null<scf::ForOp>(dependency.recurrenceLoop);
    if (!forOp) {
      continue;
    }
    const std::optional<bool> canOccur =
        canRecurrenceOccur(forOp, dependency.iterationDistance);
    if (canOccur && !*canOccur) {
      dependency.retained = false;
    }
  }
}

void CanonicalSyncPlanBuilder::preserveForwardCompletionRequirements() {
  plan_.completionRequirements_.clear();
  for (const CanonicalDependency &dependency : plan_.dependencies_) {
    if (dependency.retained && dependency.iterationDistance == 0) {
      plan_.completionRequirements_.push_back(dependency);
    }
  }
}

void CanonicalSyncPlanBuilder::reduceForwardDependencies() {
  auto groups = groupDependencies(plan_.dependencies_);
  std::map<Block *,
           std::map<std::pair<PipelineType, PipelineType>,
                    std::vector<DependencyGroupKey>>,
           std::less<Block *>>
      blockDomains;
  for (const auto &entry : groups) {
    const CanonicalDependency &dependency =
        plan_.dependencies_[entry.second.front()];
    if (!dependency.retained || entry.first.distance != 0) {
      continue;
    }
    const CanonicalSyncNode &source = plan_.nodes_[entry.first.source];
    const CanonicalSyncNode &target = plan_.nodes_[entry.first.target];
    Block *block = source.operation->getBlock();
    if (!block || block != target.operation->getBlock()) {
      continue;
    }
    blockDomains[block][{source.pipe, target.pipe}].push_back(entry.first);
  }

  for (auto &blockEntry : blockDomains) {
    for (auto &domainEntry : blockEntry.second) {
      std::vector<DependencyGroupKey> &keys = domainEntry.second;
      llvm::stable_sort(keys, [](const DependencyGroupKey &first,
                                 const DependencyGroupKey &second) {
        if (first.source != second.source) {
          return first.source > second.source;
        }
        return first.target < second.target;
      });
      std::optional<std::size_t> earliestTarget;
      for (const DependencyGroupKey &key : keys) {
        if (earliestTarget && key.target >= *earliestTarget) {
          for (std::size_t dependency : groups[key]) {
            plan_.dependencies_[dependency].retained = false;
          }
          continue;
        }
        earliestTarget = key.target;
      }
    }
  }

  std::vector<CompletionRequirement> requirements;
  std::vector<DependencyGroupKey> requirementKeys;
  for (const auto &entry : groups) {
    const bool retained = plan_.dependencies_[entry.second.front()].retained;
    const std::size_t source = entry.first.source;
    const std::size_t target = entry.first.target;
    if (retained && entry.first.distance == 0) {
      requirements.push_back({source, target});
      requirementKeys.push_back(entry.first);
    }
  }
  const auto isForwardVertexAvailable = [&](std::size_t requirement,
                                            std::size_t vertex) {
    const bool isInvalid =
        requirement >= requirementKeys.size() || vertex >= plan_.nodes_.size();
    if (isInvalid) {
      return false;
    }
    const DependencyGroupKey &key = requirementKeys[requirement];
    return isGuaranteedInContext(
        plan_.nodes_[vertex].operation, 0, plan_.nodes_[key.source].operation,
        0, plan_.nodes_[key.target].operation, 0, nullptr, true);
  };
  std::vector<bool> keep =
      reduceCompletionRequirements(plan_.nodes_.size(), plan_.fixedEdges_,
                                   requirements, isForwardVertexAvailable);
  for (std::size_t index = 0; index < keep.size(); ++index) {
    for (std::size_t dependency : groups[requirementKeys[index]]) {
      plan_.dependencies_[dependency].retained = keep[index];
    }
  }

  DenseSet<Operation *> recurrenceOwners;
  for (const auto &entry : groups) {
    Operation *loop = entry.first.owner;
    const bool retained = plan_.dependencies_[entry.second.front()].retained;
    if (retained && entry.first.distance != 0 && loop) {
      recurrenceOwners.insert(loop);
    }
  }
  SmallVector<Operation *, 8> loops;
  func_.walk([&](Operation *op) {
    if (recurrenceOwners.contains(op)) {
      loops.push_back(op);
    }
  });
  llvm::stable_sort(loops, [](Operation *first, Operation *second) {
    return getLoopDepth(first) > getLoopDepth(second);
  });
  for (Operation *loop : loops) {
    const std::size_t nodeCount = plan_.nodes_.size();
    SmallVector<unsigned, 4> distances;
    for (const auto &entry : groups) {
      const bool retained = plan_.dependencies_[entry.second.front()].retained;
      if (!retained || entry.first.distance == 0 || entry.first.owner != loop) {
        continue;
      }
      distances.push_back(entry.first.distance);
    }
    llvm::sort(distances);
    distances.erase(std::unique(distances.begin(), distances.end()),
                    distances.end());

    for (unsigned distance : distances) {
      const std::size_t occurrenceCount =
          static_cast<std::size_t>(distance) + 1;
      if (nodeCount != 0 &&
          occurrenceCount >
              std::numeric_limits<std::size_t>::max() / nodeCount) {
        continue;
      }
      const std::size_t expandedNodeCount = nodeCount * occurrenceCount;
      std::vector<SyncGraphEdge> expandedFixed;
      for (const SyncGraphEdge &edge : plan_.fixedEdges_) {
        if (!loop->isAncestor(plan_.nodes_[edge.source].operation) ||
            !loop->isAncestor(plan_.nodes_[edge.target].operation)) {
          continue;
        }
        for (std::size_t occurrence = 0; occurrence < occurrenceCount;
             ++occurrence) {
          const std::size_t offset = occurrence * nodeCount;
          expandedFixed.push_back(
              {offset + edge.source, offset + edge.target, edge.kind});
        }
      }

      // A nested recurrence primitive has loop-boundary-specific effects. It
      // cannot be summarized as an unconditional completion edge in each copy.
      for (const auto &entry : groups) {
        const std::size_t source = entry.first.source;
        const std::size_t target = entry.first.target;
        const bool retained =
            plan_.dependencies_[entry.second.front()].retained;
        if (!retained || entry.first.distance != 0 ||
            !loop->isAncestor(plan_.nodes_[source].operation) ||
            !loop->isAncestor(plan_.nodes_[target].operation)) {
          continue;
        }
        for (std::size_t occurrence = 0; occurrence < occurrenceCount;
             ++occurrence) {
          const std::size_t offset = occurrence * nodeCount;
          expandedFixed.push_back({offset + source, offset + target,
                                   SyncGraphEdgeKind::HardwareCompletion});
        }
      }

      for (std::size_t occurrence = 0; occurrence + 1 < occurrenceCount;
           ++occurrence) {
        const std::size_t sourceOffset = occurrence * nodeCount;
        const std::size_t targetOffset = sourceOffset + nodeCount;
        for (const CanonicalSyncNode &source : plan_.nodes_) {
          if (!loop->isAncestor(source.operation)) {
            continue;
          }
          for (const CanonicalSyncNode &target : plan_.nodes_) {
            if (source.pipe != target.pipe ||
                !loop->isAncestor(target.operation)) {
              continue;
            }
            expandedFixed.push_back({sourceOffset + source.id,
                                     targetOffset + target.id,
                                     hasHardwareCompletion(source.pipe)
                                         ? SyncGraphEdgeKind::HardwareCompletion
                                         : SyncGraphEdgeKind::IssueOrder});
          }
        }
      }

      std::vector<CompletionRequirement> recurrenceRequirements;
      std::vector<DependencyGroupKey> recurrenceKeys;
      for (const auto &entry : groups) {
        const bool retained =
            plan_.dependencies_[entry.second.front()].retained;
        if (!retained || entry.first.distance != distance ||
            entry.first.owner != loop) {
          continue;
        }
        recurrenceRequirements.push_back(
            {entry.first.source, distance * nodeCount + entry.first.target});
        recurrenceKeys.push_back(entry.first);
      }
      const auto isRecurrenceVertexAvailable = [&](std::size_t requirement,
                                                   std::size_t vertex) {
        const bool isInvalid =
            requirement >= recurrenceKeys.size() || vertex >= expandedNodeCount;
        if (isInvalid) {
          return false;
        }
        const DependencyGroupKey &key = recurrenceKeys[requirement];
        const unsigned occurrence = static_cast<unsigned>(vertex / nodeCount);
        Operation *operation = plan_.nodes_[vertex % nodeCount].operation;
        if (!loop->isAncestor(operation)) {
          return false;
        }
        return isGuaranteedInContext(
            operation, occurrence, plan_.nodes_[key.source].operation, 0,
            plan_.nodes_[key.target].operation, distance, loop);
      };
      std::vector<bool> recurrenceKeep = reduceCompletionRequirements(
          expandedNodeCount, expandedFixed, recurrenceRequirements,
          isRecurrenceVertexAvailable);
      for (std::size_t index = 0; index < recurrenceKeep.size(); ++index) {
        for (std::size_t dependency : groups[recurrenceKeys[index]]) {
          plan_.dependencies_[dependency].retained = recurrenceKeep[index];
        }
      }
    }
  }
}

void CanonicalSyncPlanBuilder::preserveRecurrenceCompletionRequirements() {
  for (const CanonicalDependency &dependency : plan_.dependencies_) {
    if (dependency.retained && dependency.iterationDistance != 0) {
      plan_.completionRequirements_.push_back(dependency);
    }
  }
}

bool CanonicalSyncPlanBuilder::isForwardVertexAvailable(
    const CanonicalDependency &requirement, std::size_t vertex) const {
  if (requirement.iterationDistance != 0 ||
      requirement.source >= plan_.nodes_.size() ||
      requirement.target >= plan_.nodes_.size() ||
      vertex >= plan_.nodes_.size()) {
    return false;
  }
  return isGuaranteedInContext(plan_.nodes_[vertex].operation, 0,
                               plan_.nodes_[requirement.source].operation, 0,
                               plan_.nodes_[requirement.target].operation, 0,
                               nullptr, true);
}

bool CanonicalSyncPlanBuilder::isRecurrenceVertexAvailable(
    const CanonicalDependency &requirement, std::size_t vertex,
    std::size_t nodeCount) const {
  if (requirement.iterationDistance == 0 || !requirement.recurrenceLoop ||
      nodeCount == 0 || requirement.source >= nodeCount ||
      requirement.target >= nodeCount ||
      vertex / nodeCount > requirement.iterationDistance) {
    return false;
  }
  Operation *operation = plan_.nodes_[vertex % nodeCount].operation;
  if (!requirement.recurrenceLoop->isAncestor(operation)) {
    return false;
  }
  const unsigned occurrence = static_cast<unsigned>(vertex / nodeCount);
  return isGuaranteedInContext(
      operation, occurrence, plan_.nodes_[requirement.source].operation, 0,
      plan_.nodes_[requirement.target].operation, requirement.iterationDistance,
      requirement.recurrenceLoop);
}

bool CanonicalSyncPlanBuilder::isAnchorGuaranteedForRequirement(
    const CanonicalAnchor &anchor, std::size_t source,
    std::size_t target) const {
  if (!anchor.operation || source >= plan_.nodes_.size() ||
      target >= plan_.nodes_.size()) {
    return false;
  }
  return isGuaranteedInContext(
      anchor.operation, 0, plan_.nodes_[source].operation, 0,
      plan_.nodes_[target].operation, 0, nullptr, true);
}

bool CanonicalSyncPlanBuilder::isRecurrenceAnchorGuaranteedForEndpoint(
    const CanonicalAnchor &anchor, unsigned anchorOccurrence,
    std::size_t endpoint, unsigned endpointOccurrence, Operation *loop) const {
  if (!anchor.operation || !loop || endpoint >= plan_.nodes_.size()) {
    return false;
  }
  Operation *endpointOperation = plan_.nodes_[endpoint].operation;
  return isGuaranteedInContext(anchor.operation, anchorOccurrence,
                               endpointOperation, endpointOccurrence,
                               endpointOperation, endpointOccurrence, loop);
}

CanonicalAnchor
CanonicalSyncPlanBuilder::getSetAnchor(Operation *source,
                                       Operation *target) const {
  Operation *anchor = source;
  for (Operation *parent = source->getParentOp(); parent && parent != func_;
       parent = parent->getParentOp()) {
    if (!isStructuralContainer(parent)) {
      continue;
    }
    if (parent->isAncestor(target)) {
      break;
    }
    anchor = parent;
  }
  return {anchor, false};
}

CanonicalAnchor
CanonicalSyncPlanBuilder::getWaitAnchor(Operation *source,
                                        Operation *target) const {
  Operation *anchor = target;
  for (Operation *parent = target->getParentOp(); parent && parent != func_;
       parent = parent->getParentOp()) {
    if (!isStructuralContainer(parent)) {
      continue;
    }
    if (parent->isAncestor(source)) {
      break;
    }
    anchor = parent;
  }
  return {anchor, true};
}

std::size_t CanonicalSyncPlanBuilder::getAnchorPosition(
    const CanonicalAnchor &anchor) const {
  std::size_t minimum = std::numeric_limits<std::size_t>::max();
  std::size_t maximum = 0;
  std::size_t timelineEnd = 0;
  bool found = false;
  for (const CanonicalSyncNode &node : plan_.nodes_) {
    timelineEnd = std::max(timelineEnd, node.order * 2 + 1);
    const bool matches = node.operation == anchor.operation ||
                         anchor.operation->isAncestor(node.operation);
    if (!matches) {
      continue;
    }
    found = true;
    minimum = std::min(minimum, node.order * 2);
    maximum = std::max(maximum, node.order * 2 + 1);
  }
  if (!found) {
    Block *anchorBlock = anchor.operation->getBlock();
    std::optional<std::size_t> previous;
    std::optional<std::size_t> next;
    for (const CanonicalSyncNode &node : plan_.nodes_) {
      Operation *child = node.operation;
      while (child && child->getBlock() != anchorBlock) {
        child = child->getParentOp();
      }
      if (!child || child == anchor.operation) {
        continue;
      }
      if (child->isBeforeInBlock(anchor.operation)) {
        previous = std::max(previous.value_or(0), node.order * 2 + 1);
      } else if (anchor.operation->isBeforeInBlock(child)) {
        next = std::min(next.value_or(std::numeric_limits<std::size_t>::max()),
                        node.order * 2);
      }
    }
    if (next) {
      return *next;
    }
    if (previous) {
      return *previous;
    }
    return anchor.before ? 0 : timelineEnd + 1;
  }
  return anchor.before ? minimum : maximum;
}

unsigned CanonicalSyncPlanBuilder::getRecurrenceWidth(
    const CanonicalDependency &dependency, Value &setSlot,
    Value &waitSlot) const {
  const CanonicalSyncNode &source = plan_.nodes_[dependency.source];
  const CanonicalSyncNode &target = plan_.nodes_[dependency.target];
  if (!isa_and_nonnull<scf::ForOp>(dependency.recurrenceLoop) ||
      source.operation->getBlock()->getParentOp() !=
          dependency.recurrenceLoop ||
      target.operation->getBlock()->getParentOp() !=
          dependency.recurrenceLoop) {
    return 1;
  }
  unsigned width = 1;
  for (const CanonicalMemoryAccess &sourceAccess : source.accesses) {
    for (const CanonicalMemoryAccess &targetAccess : target.accesses) {
      if (!formsMemoryHazard(dependency.kind, sourceAccess, targetAccess) ||
          !memoryAliasesAcrossIterations(sourceAccess, targetAccess,
                                         dependency.recurrenceLoop,
                                         dependency.iterationDistance) ||
          compareSlotsAcrossIterations(
              sourceAccess, targetAccess, dependency.recurrenceLoop,
              dependency.iterationDistance) != SlotRelation::kEqual) {
        continue;
      }
      const std::size_t slots = std::max(sourceAccess.addresses.size(),
                                         targetAccess.addresses.size());
      if (slots <= 1 || slots > kMaxMultiBufferCount) {
        continue;
      }
      Value candidateSet = findMultiTileSlotExpr(sourceAccess.base);
      Value candidateWait = findMultiTileSlotExpr(targetAccess.base);
      if (!candidateSet || !candidateWait ||
          (setSlot && setSlot != candidateSet) ||
          (waitSlot && waitSlot != candidateWait) ||
          (width != 1 && width != slots)) {
        setSlot = {};
        waitSlot = {};
        return 1;
      }
      setSlot = candidateSet;
      waitSlot = candidateWait;
      width = static_cast<unsigned>(slots);
    }
  }
  return width;
}

void CanonicalSyncPlanBuilder::materializeSyncRequirements() {
  materializeBarriers();
  materializeEvents();
  synthesizeOwnershipProtocols();
}

void CanonicalSyncPlanBuilder::materializeBarriers() {
  auto groups = groupDependencies(plan_.dependencies_);
  DenseMap<Operation *, std::size_t> operationOrders;
  std::size_t nextOperationOrder = 0;
  func_.walk([&](Operation *operation) {
    operationOrders[operation] = nextOperationOrder++;
  });
  for (const auto &entry : groups) {
    if (!entry.first.owner ||
        llvm::any_of(plan_.recurrenceScopes_, [&](const auto &scope) {
          return scope.operation == entry.first.owner;
        })) {
      continue;
    }
    CanonicalRecurrenceScope scope;
    scope.id = entry.first.ownerOrder;
    scope.operation = entry.first.owner;
    scope.operationOrder = operationOrders.lookup(entry.first.owner);
    plan_.recurrenceScopes_.push_back(scope);
  }
  llvm::sort(plan_.recurrenceScopes_, [](const auto &first,
                                         const auto &second) {
    return first.id < second.id;
  });
  for (CanonicalRecurrenceScope &scope : plan_.recurrenceScopes_) {
    for (Operation *parent = scope.operation->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto parentScope = llvm::find_if(
          plan_.recurrenceScopes_,
          [&](const auto &candidate) { return candidate.operation == parent; });
      if (parentScope != plan_.recurrenceScopes_.end()) {
        scope.parentScope = parentScope->id;
        break;
      }
    }
  }

  std::size_t nextBarrierId = 0;
  for (const auto &entry : groups) {
    const CanonicalDependency &dependency =
        plan_.dependencies_[entry.second.front()];
    if (!dependency.retained) {
      continue;
    }
    const CanonicalSyncNode &source = plan_.nodes_[dependency.source];
    const CanonicalSyncNode &target = plan_.nodes_[dependency.target];
    if (source.pipe != target.pipe || hasHardwareCompletion(source.pipe)) {
      continue;
    }
    CanonicalBarrier barrier;
    barrier.id = nextBarrierId;
    barrier.pipe = source.pipe;
    barrier.anchor = dependency.iterationDistance == 0
                         ? getWaitAnchor(source.operation, target.operation)
                         : CanonicalAnchor{target.operation, true};
    auto anchorNodes = operationNodes_.find(barrier.anchor.operation);
    if (anchorNodes != operationNodes_.end()) {
      barrier.anchorNodes.append(anchorNodes->second.begin(),
                                 anchorNodes->second.end());
    }
    barrier.recurrenceLoop = dependency.recurrenceLoop;
    barrier.recurrenceScope = entry.first.ownerOrder;
    for (auto [requirementId, requirement] :
         llvm::enumerate(plan_.completionRequirements_)) {
      if (requirement.source == entry.first.source &&
          requirement.target == entry.first.target &&
          requirement.iterationDistance == entry.first.distance &&
          requirement.recurrenceLoop == entry.first.owner) {
        barrier.requirements.push_back(requirementId);
      }
    }

    auto duplicate = llvm::find_if(plan_.barriers_, [&](const auto &old) {
      return old.pipe == barrier.pipe &&
             old.anchor.operation == barrier.anchor.operation &&
             old.anchor.before == barrier.anchor.before &&
             old.recurrenceLoop == barrier.recurrenceLoop;
    });
    if (duplicate != plan_.barriers_.end()) {
      for (std::size_t requirement : barrier.requirements) {
        if (!llvm::is_contained(duplicate->requirements, requirement)) {
          duplicate->requirements.push_back(requirement);
        }
      }
      continue;
    }
    plan_.barriers_.push_back(std::move(barrier));
    ++nextBarrierId;
  }
}

std::vector<SyncGraphEdge>
CanonicalSyncPlanBuilder::buildBarrierCompletionEdges(
    ArrayRef<CanonicalBarrier> barriers) const {
  std::vector<SyncGraphEdge> edges;
  std::set<std::pair<std::size_t, std::size_t>> seen;
  for (const CanonicalBarrier &barrier : barriers) {
    if (barrier.recurrenceLoop) {
      continue;
    }
    const std::size_t position = getAnchorPosition(barrier.anchor);
    for (const CanonicalSyncNode &source : plan_.nodes_) {
      if (source.pipe != barrier.pipe || source.order * 2 + 1 > position) {
        continue;
      }
      for (const CanonicalSyncNode &target : plan_.nodes_) {
        if (source.id >= target.id || target.order * 2 < position ||
            !mayExecuteTogether(source.operation, target.operation) ||
            !isGuaranteedInContext(barrier.anchor.operation, 0,
                                   source.operation, 0, target.operation, 0,
                                   nullptr, true)) {
          continue;
        }
        if (seen.emplace(source.id, target.id).second) {
          edges.push_back(
              {source.id, target.id, SyncGraphEdgeKind::HardwareCompletion});
        }
      }
    }
  }
  return edges;
}

void CanonicalSyncPlanBuilder::materializeEvents() {
  materializeEventsFrom(plan_.dependencies_, plan_.events_);
  materializeEventsFrom(plan_.completionRequirements_, eventCandidates_);
}

CanonicalEvent
CanonicalSyncPlanBuilder::makeForwardEvent(std::size_t sourceId,
                                           std::size_t targetId) const {
  const CanonicalSyncNode &source = plan_.nodes_[sourceId];
  const CanonicalSyncNode &target = plan_.nodes_[targetId];
  CanonicalEvent event;
  event.source = sourceId;
  event.target = targetId;
  event.sourcePipe = source.pipe;
  event.targetPipe = target.pipe;
  event.setAnchor = getSetAnchor(source.operation, target.operation);
  event.waitAnchor = getWaitAnchor(source.operation, target.operation);
  event.forwardDrainLoop =
      getWhileForwardDrain(source.operation, target.operation);
  event.intervalBegin = getAnchorPosition(event.setAnchor);
  event.intervalEnd = event.forwardDrainLoop
                          ? getAnchorPosition({event.forwardDrainLoop, false})
                          : getAnchorPosition(event.waitAnchor);
  if (event.intervalBegin > event.intervalEnd) {
    std::swap(event.intervalBegin, event.intervalEnd);
  }
  initializeForwardProtocol(event);
  return event;
}

CanonicalEvent CanonicalSyncPlanBuilder::makeRecurrenceEvent(
    const CanonicalDependency &dependency) const {
  const CanonicalSyncNode &source = plan_.nodes_[dependency.source];
  const CanonicalSyncNode &target = plan_.nodes_[dependency.target];
  CanonicalEvent event;
  event.source = dependency.source;
  event.target = dependency.target;
  event.sourcePipe = source.pipe;
  event.targetPipe = target.pipe;
  event.recurrenceLoop = dependency.recurrenceLoop;
  event.scopeLoop = dependency.recurrenceLoop;
  event.iterationDistance = dependency.iterationDistance;
  event.width = getRecurrenceWidth(dependency, event.setSlot, event.waitSlot);
  event.setAnchor =
      getLoopRegionExitAnchor(source.operation, event.recurrenceLoop);
  event.waitAnchor = event.width == 1 ? getLoopEntryAnchor(event.recurrenceLoop)
                                      : CanonicalAnchor{target.operation, true};
  event.intervalBegin = getAnchorPosition({event.recurrenceLoop, true});
  event.intervalEnd = getAnchorPosition({event.recurrenceLoop, false});
  if (event.intervalBegin > event.intervalEnd) {
    std::swap(event.intervalBegin, event.intervalEnd);
  }
  if (event.width > 1 && compareSlotSSA(event.setSlot, event.waitSlot,
                                        event.width) != SlotRelation::kEqual) {
    event.width = 1;
    event.setSlot = {};
    event.waitSlot = {};
  }
  initializeRecurrenceProtocol(event);
  return event;
}

void CanonicalSyncPlanBuilder::materializeEventsFrom(
    ArrayRef<CanonicalDependency> dependencies,
    std::vector<CanonicalEvent> &events) {
  auto groups = groupDependencies(dependencies);
  for (const auto &entry : groups) {
    const CanonicalDependency &dependency = dependencies[entry.second.front()];
    if (!dependency.retained) {
      continue;
    }
    const CanonicalSyncNode &source = plan_.nodes_[dependency.source];
    const CanonicalSyncNode &target = plan_.nodes_[dependency.target];
    if (source.pipe == target.pipe) {
      continue;
    }

    CanonicalEvent event =
        dependency.iterationDistance == 0
            ? makeForwardEvent(dependency.source, dependency.target)
            : makeRecurrenceEvent(dependency);
    auto existing = llvm::find_if(events, [&](const auto &old) {
      const bool sameProtocol =
          old.sourcePipe == event.sourcePipe &&
          old.targetPipe == event.targetPipe &&
          old.setAnchor.operation == event.setAnchor.operation &&
          old.setAnchor.before == event.setAnchor.before &&
          old.waitAnchor.operation == event.waitAnchor.operation &&
          old.waitAnchor.before == event.waitAnchor.before &&
          old.recurrenceLoop == event.recurrenceLoop &&
          old.forwardDrainLoop == event.forwardDrainLoop &&
          old.setSlot == event.setSlot && old.waitSlot == event.waitSlot &&
          old.width == event.width;
      return sameProtocol &&
             (!event.recurrenceLoop ||
              (old.source == event.source && old.target == event.target &&
               old.iterationDistance == event.iterationDistance));
    });
    if (existing == events.end()) {
      events.push_back(std::move(event));
      continue;
    }
    for (const CanonicalEventCompletion &completion : event.completions) {
      const bool duplicateCompletion = llvm::any_of(
          existing->completions,
          [&](const CanonicalEventCompletion &old) {
            return old.source == completion.source &&
                   old.target == completion.target &&
                   old.iterationDistance == completion.iterationDistance &&
                   old.recurrenceLoop == completion.recurrenceLoop &&
                   old.setAction == completion.setAction &&
                   old.waitAction == completion.waitAction;
          });
      if (!duplicateCompletion) {
        existing->completions.push_back(completion);
      }
    }
  }
}

void CanonicalSyncPlanBuilder::reserveHiddenEventIds() {
  DenseSet<Operation *> visited;
  for (const CanonicalSyncNode &node : plan_.nodes_) {
    if (!visited.insert(node.operation).second) {
      continue;
    }
    std::optional<SyncMacroModel> model = getSyncMacroModel(node.operation);
    if (!model) {
      continue;
    }
    for (const SyncMacroHiddenEvent &hidden : model->hiddenEvents) {
      auto &ids = reservedIds_[{hidden.srcPipe, hidden.dstPipe}];
      ids.insert(hidden.eventIds.begin(), hidden.eventIds.end());
    }
  }
}
