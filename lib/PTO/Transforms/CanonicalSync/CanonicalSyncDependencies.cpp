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

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/Transforms/KernelScheduling/KernelScheduleGraph.h"

#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct MemoryHazardKinds {
  bool raw = false;
  bool war = false;
  bool waw = false;
};

template <typename AliasPredicate>
MemoryHazardKinds collectMemoryHazardKinds(
    const CanonicalSyncNode &source, const CanonicalSyncNode &target,
    AliasPredicate aliases) {
  MemoryHazardKinds hazards;
  for (const CanonicalMemoryAccess &sourceAccess : source.accesses) {
    for (const CanonicalMemoryAccess &targetAccess : target.accesses) {
      if (!aliases(sourceAccess, targetAccess)) {
        continue;
      }
      hazards.raw |= sourceAccess.writes && targetAccess.reads;
      hazards.war |= sourceAccess.reads && targetAccess.writes;
      hazards.waw |= sourceAccess.writes && targetAccess.writes;
    }
  }
  return hazards;
}

unsigned getMaximumRecurrenceDistance(const CanonicalSyncNode &source,
                                      const CanonicalSyncNode &target) {
  unsigned maximumDistance = 1;
  for (const CanonicalMemoryAccess &sourceAccess : source.accesses) {
    for (const CanonicalMemoryAccess &targetAccess : target.accesses) {
      if (sourceAccess.root != targetAccess.root) {
        continue;
      }
      const std::size_t slots = std::max(sourceAccess.addresses.size(),
                                         targetAccess.addresses.size());
      if (slots <= kMaxMultiBufferCount) {
        maximumDistance =
            std::max(maximumDistance, static_cast<unsigned>(slots));
      }
    }
  }
  return maximumDistance;
}

} // namespace

void CanonicalSyncPlanBuilder::addAccessHazards(const CanonicalSyncNode &source,
                                                const CanonicalSyncNode &target,
                                                unsigned iterationDistance,
                                                Operation *loop,
                                                bool compareSlots,
                                                bool honorNoAlias,
                                                bool activeWitness) {
  for (const CanonicalMemoryAccess &sourceAccess : source.accesses) {
    for (const CanonicalMemoryAccess &targetAccess : target.accesses) {
      const bool aliases =
          iterationDistance == 0
              ? memoryAliases(sourceAccess, targetAccess, compareSlots,
                              honorNoAlias)
              : memoryAliasesAcrossIterations(sourceAccess, targetAccess, loop,
                                              iterationDistance,
                                              honorNoAlias);
      if (!aliases) {
        continue;
      }
      if (sourceAccess.writes && targetAccess.reads) {
        addDependency(source.id, target.id, CanonicalDependencyKind::MemoryRAW,
                      iterationDistance, loop, activeWitness);
      }
      if (sourceAccess.reads && targetAccess.writes) {
        addDependency(source.id, target.id, CanonicalDependencyKind::MemoryWAR,
                      iterationDistance, loop, activeWitness);
      }
      if (sourceAccess.writes && targetAccess.writes) {
        addDependency(source.id, target.id, CanonicalDependencyKind::MemoryWAW,
                      iterationDistance, loop, activeWitness);
      }
    }
  }
}

void CanonicalSyncPlanBuilder::addRecurrenceAccessHazards(
    const CanonicalSyncNode &source, const CanonicalSyncNode &target,
    Operation *loop) {
  MemoryHazardKinds activeHazards;
  const unsigned maximumDistance = getMaximumRecurrenceDistance(source, target);
  for (unsigned distance = 1; distance <= maximumDistance; ++distance) {
    const auto collectAtDistance = [&](bool honorNoAlias) {
      return collectMemoryHazardKinds(
          source, target,
          [&](const CanonicalMemoryAccess &sourceAccess,
              const CanonicalMemoryAccess &targetAccess) {
            return memoryAliasesAcrossIterations(
                sourceAccess, targetAccess, loop, distance, honorNoAlias);
          });
    };
    const MemoryHazardKinds conservative = collectAtDistance(false);
    const MemoryHazardKinds active = collectAtDistance(true);

    if (conservative.raw) {
      addDependency(source.id, target.id, CanonicalDependencyKind::MemoryRAW,
                    distance, loop, active.raw && !activeHazards.raw);
    }
    if (conservative.war) {
      addDependency(source.id, target.id, CanonicalDependencyKind::MemoryWAR,
                    distance, loop, active.war && !activeHazards.war);
    }
    if (conservative.waw) {
      addDependency(source.id, target.id, CanonicalDependencyKind::MemoryWAW,
                    distance, loop, active.waw && !activeHazards.waw);
    }
    activeHazards.raw |= active.raw;
    activeHazards.war |= active.war;
    activeHazards.waw |= active.waw;
  }
}

LogicalResult CanonicalSyncPlanBuilder::addDependencies() {
  addIssueOrderEdges();
  addMemoryDependencies();
  return addSSAAndRecurrenceDependencies();
}

void CanonicalSyncPlanBuilder::addFixedEdge(std::size_t source,
                                            std::size_t target,
                                            SyncGraphEdgeKind kind) {
  if (fixedEdgeKeys_.insert({source, target, kind}).second) {
    plan_.fixedEdges_.push_back({source, target, kind});
  }
}

bool CanonicalSyncPlanBuilder::hasHardwareCompletion(PipelineType pipe) const {
  if (pipe == PipelineType::PIPE_S) {
    return true;
  }
  return pipe == PipelineType::PIPE_V && isTargetArchA5(funcOperation_);
}

bool CanonicalSyncPlanBuilder::hasIntrinsicMmadAccumulatorOrdering(
    const CanonicalDependency &dependency) const {
  // A2/A3 orders dependent in-place MMAD accumulation without making the
  // source a transitive completion point for other pipes.
  const bool invalidEndpoint = dependency.source >= plan_.nodes_.size() ||
                               dependency.target >= plan_.nodes_.size();
  if (!isTargetArchA3(funcOperation_)) {
    return false;
  }
  if (invalidEndpoint) {
    return false;
  }
  switch (dependency.kind) {
  case CanonicalDependencyKind::MemoryRAW:
  case CanonicalDependencyKind::MemoryWAR:
  case CanonicalDependencyKind::MemoryWAW:
    break;
  case CanonicalDependencyKind::SSA:
  case CanonicalDependencyKind::LoopCarriedSSA:
    return false;
  }

  const CanonicalSyncNode &source = plan_.nodes_[dependency.source];
  const CanonicalSyncNode &target = plan_.nodes_[dependency.target];
  if (source.pipe != PipelineType::PIPE_M ||
      target.pipe != PipelineType::PIPE_M ||
      !isa<TMatmulOp, TMatmulAccOp>(source.operation) ||
      !isa<TMatmulAccOp>(target.operation)) {
    return false;
  }

  const auto isExactL0C = [](const CanonicalMemoryAccess &access) {
    return access.space == AddressSpace::ACC && access.knownPhysical &&
           !access.unknownRange && access.addresses.size() == 1 &&
           access.size != 0;
  };
  const auto sameSlot = [&](const CanonicalMemoryAccess &first,
                            const CanonicalMemoryAccess &second) {
    return isExactL0C(first) && isExactL0C(second) &&
           first.addresses.front() == second.addresses.front() &&
           first.size == second.size;
  };
  for (const CanonicalMemoryAccess &sourceAccess : source.accesses) {
    if (!sourceAccess.writes || !isExactL0C(sourceAccess)) {
      continue;
    }
    const bool targetReads = llvm::any_of(
        target.accesses, [&](const CanonicalMemoryAccess &targetAccess) {
          return targetAccess.reads && sameSlot(sourceAccess, targetAccess);
        });
    const bool targetWrites = llvm::any_of(
        target.accesses, [&](const CanonicalMemoryAccess &targetAccess) {
          return targetAccess.writes && sameSlot(sourceAccess, targetAccess);
        });
    if (targetReads && targetWrites) {
      return true;
    }
  }
  return false;
}

void CanonicalSyncPlanBuilder::addIssueOrderEdges() {
  for (std::size_t source = 0; source < plan_.nodes_.size(); ++source) {
    for (std::size_t target = source + 1; target < plan_.nodes_.size();
         ++target) {
      const CanonicalSyncNode &sourceNode = plan_.nodes_[source];
      const CanonicalSyncNode &targetNode = plan_.nodes_[target];
      if (sourceNode.pipe != targetNode.pipe ||
          !mayExecuteTogether(sourceNode.operation, targetNode.operation)) {
        continue;
      }
      addFixedEdge(source, target,
                   hasHardwareCompletion(sourceNode.pipe)
                       ? SyncGraphEdgeKind::HardwareCompletion
                       : SyncGraphEdgeKind::IssueOrder);
    }
  }
}

void CanonicalSyncPlanBuilder::addMemoryDependencies() {
  for (std::size_t source = 0; source < plan_.nodes_.size(); ++source) {
    for (std::size_t target = source + 1; target < plan_.nodes_.size();
         ++target) {
      const CanonicalSyncNode &sourceNode = plan_.nodes_[source];
      const CanonicalSyncNode &targetNode = plan_.nodes_[target];
      if (sourceNode.operation == targetNode.operation ||
          !mayExecuteTogether(sourceNode.operation, targetNode.operation)) {
        continue;
      }
      addAccessHazards(sourceNode, targetNode, 0, nullptr,
                       /*compareSlots=*/true, /*honorNoAlias=*/false,
                       /*activeWitness=*/false);
      addAccessHazards(sourceNode, targetNode, 0, nullptr,
                       /*compareSlots=*/true, /*honorNoAlias=*/true,
                       /*activeWitness=*/true);
    }
  }

  SmallVector<Operation *, 8> loops;
  func_.walk([&](Operation *op) {
    if (isa<LoopLikeOpInterface>(op)) {
      loops.push_back(op);
    }
  });
  for (Operation *loop : loops) {
    for (const CanonicalSyncNode &source : plan_.nodes_) {
      if (!loop->isAncestor(source.operation)) {
        continue;
      }
      for (const CanonicalSyncNode &target : plan_.nodes_) {
        if (!loop->isAncestor(target.operation)) {
          continue;
        }
        addRecurrenceAccessHazards(source, target, loop);
      }
    }
  }
}

LogicalResult CanonicalSyncPlanBuilder::addSSAAndRecurrenceDependencies() {
  FailureOr<KernelScheduleGraph> scheduleGraph =
      buildKernelScheduleGraph(func_);
  if (failed(scheduleGraph)) {
    return failure();
  }
  for (const KernelScheduleDependency &dependency :
       scheduleGraph->getDependencies()) {
    const bool isSSA = dependency.kind == ScheduleDependencyKind::SSA;
    const bool isRecurrence =
        dependency.kind == ScheduleDependencyKind::LoopCarriedSSA;
    const bool isControl = dependency.kind == ScheduleDependencyKind::Control;
    if (!isSSA && !isRecurrence && !isControl) {
      continue;
    }
    Operation *sourceOperation =
        scheduleGraph->getNode(dependency.source).operation;
    Operation *targetOperation =
        scheduleGraph->getNode(dependency.target).operation;
    auto sourceIt = operationNodes_.find(sourceOperation);
    auto targetIt = operationNodes_.find(targetOperation);
    if (sourceIt == operationNodes_.end() ||
        targetIt == operationNodes_.end()) {
      continue;
    }
    const std::size_t source = sourceIt->second.back();
    const std::size_t target = targetIt->second.front();
    if (!isRecurrence && source >= target) {
      continue;
    }
    if (isControl) {
      const SyncGraphEdgeKind kind =
          plan_.nodes_[source].pipe == plan_.nodes_[target].pipe
              ? SyncGraphEdgeKind::IssueOrder
              : SyncGraphEdgeKind::NonCompletionPreservingIssueOrder;
      addFixedEdge(source, target, kind);
      continue;
    }
    addDependency(source, target,
                  isRecurrence ? CanonicalDependencyKind::LoopCarriedSSA
                               : CanonicalDependencyKind::SSA,
                  dependency.iterationDistance, dependency.recurrenceLoop);
  }
  return success();
}

void CanonicalSyncPlanBuilder::addDependency(std::size_t source,
                                             std::size_t target,
                                             CanonicalDependencyKind kind,
                                             unsigned iterationDistance,
                                             Operation *recurrenceLoop,
                                             bool activeWitness) {
  const DependencyKey key{source, target, kind, iterationDistance,
                          recurrenceLoop};
  auto [position, inserted] =
      dependencyIndices_.emplace(key, plan_.dependencies_.size());
  if (!inserted) {
    CanonicalDependency &dependency = plan_.dependencies_[position->second];
    dependency.active |= activeWitness;
    dependency.retained |= activeWitness;
    return;
  }
  CanonicalDependency dependency;
  dependency.source = source;
  dependency.target = target;
  dependency.kind = kind;
  dependency.iterationDistance = iterationDistance;
  dependency.recurrenceLoop = recurrenceLoop;
  dependency.active = activeWitness;
  dependency.retained = activeWitness;
  plan_.dependencies_.push_back(std::move(dependency));
}
