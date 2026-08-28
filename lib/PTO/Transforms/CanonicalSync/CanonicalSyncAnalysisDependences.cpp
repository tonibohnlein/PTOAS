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

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

bool executesOnlyOnFirstIteration(Operation *operation, scf::ForOp loop) {
  for (Operation *nested = operation; nested && nested != loop.getOperation();
       nested = nested->getParentOp()) {
    auto conditional = dyn_cast_or_null<scf::IfOp>(nested->getParentOp());
    if (!conditional) {
      continue;
    }
    auto comparison = conditional.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!comparison) {
      continue;
    }
    const bool comparesLoopStart =
        (comparison.getLhs() == loop.getInductionVar() &&
         comparison.getRhs() == loop.getLowerBound()) ||
        (comparison.getRhs() == loop.getInductionVar() &&
         comparison.getLhs() == loop.getLowerBound());
    if (!comparesLoopStart) {
      continue;
    }
    const bool thenAlternative =
        nested->getParentRegion() == &conditional.getThenRegion();
    const bool equality = comparison.getPredicate() == arith::CmpIPredicate::eq;
    const bool inequality =
        comparison.getPredicate() == arith::CmpIPredicate::ne;
    const bool selectsFirstIteration =
        (equality && thenAlternative) || (inequality && !thenAlternative);
    if (selectsFirstIteration) {
      return true;
    }
  }
  return false;
}

} // namespace

bool ProgramBuilder::consumePairInspections(std::size_t amount) {
  if (pairInspections_ > options_.maximumPairInspections ||
      amount > options_.maximumPairInspections - pairInspections_) {
    return false;
  }
  pairInspections_ += amount;
  return true;
}

LogicalResult ProgramBuilder::addFixedIssueOrder() {
  IssueOrderState state;
  return addRegionIssueOrder(function_.getBody(), state);
}

LogicalResult ProgramBuilder::addRegionIssueOrder(Region &region,
                                                  IssueOrderState &state) {
  if (region.empty()) {
    return success();
  }
  for (Operation &operation : region.front()) {
    if (isCanonicalSyncOwned(&operation)) {
      continue;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      IssueOrderState body = state;
      if (failed(addRegionIssueOrder(loop.getRegion(), body))) {
        return failure();
      }
      if (failed(mergeIssueStates(state, body))) {
        return failure();
      }
      continue;
    }
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      IssueOrderState alternatives;
      for (Region &alternative : conditional->getRegions()) {
        IssueOrderState branch = state;
        if (failed(addRegionIssueOrder(alternative, branch))) {
          return failure();
        }
        if (failed(mergeIssueStates(alternatives, branch))) {
          return failure();
        }
      }
      state = std::move(alternatives);
      continue;
    }
    if (isTransparentRegionOperation(&operation)) {
      if (failed(addRegionIssueOrder(operation.getRegion(0), state))) {
        return failure();
      }
      continue;
    }
    if (auto barrier = dyn_cast<BarrierOp>(operation)) {
      if (failed(addFixedBarrier(barrier, state))) {
        return failure();
      }
      continue;
    }
    auto scheduled = operationNodes_.find(&operation);
    if (scheduled == operationNodes_.end()) {
      continue;
    }
    for (SyncCoverNodeId node : scheduled->second) {
      if (failed(addIssueNode(node, state))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult ProgramBuilder::addIssueNode(SyncCoverNodeId target,
                                           IssueOrderState &state) {
  const SyncCoverNode &targetNode = graph_.getNodes()[target];
  if (state.fixedBarriers && !state.fixedBarriers->empty()) {
    bool changesBoundary = false;
    for (const FixedBarrierBoundary &boundary : *state.fixedBarriers) {
      if (!consumePairInspection()) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      changesBoundary |=
          boundary.remainingTargetResources->find(targetNode.resource) !=
          boundary.remainingTargetResources->end();
    }
    if (changesBoundary) {
      if (!consumePairInspections(state.fixedBarriers->size())) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      std::vector<FixedBarrierBoundary> boundaries = *state.fixedBarriers;
      for (FixedBarrierBoundary &boundary : boundaries) {
        auto remaining =
            boundary.remainingTargetResources->find(targetNode.resource);
        if (remaining == boundary.remainingTargetResources->end()) {
          continue;
        }
        if (!consumePairInspections(
                boundary.remainingTargetResources->size())) {
          return function_.emitError(
              "canonical sync pair-inspection limit exceeded");
        }
        std::set<std::uint32_t> remainingTargetResources =
            *boundary.remainingTargetResources;
        for (SyncCoverNodeId source : *boundary.sources) {
          if (!consumePairInspection()) {
            return function_.emitError(
                "canonical sync pair-inspection limit exceeded");
          }
          const SyncCoverNode &sourceNode = graph_.getNodes()[source];
          const bool incompatible =
              sourceNode.order >= targetNode.order ||
              !syncCoverGuardsCompatible(sourceNode.guard, *boundary.guard) ||
              !syncCoverGuardsCompatible(targetNode.guard, *boundary.guard);
          if (incompatible) {
            continue;
          }
          const std::optional<SyncCoverScopeId> scope =
              graph_.getLowestCommonScope(sourceNode.scope, targetNode.scope);
          if (!scope) {
            return function_.emitError(
                "canonical sync fixed-barrier scope failed");
          }
          SyncCoverEdge edge;
          edge.source = source;
          edge.target = target;
          edge.scope = *scope;
          edge.kind = SyncCoverEdgeKind::CompletionSupply;
          const bool guardStorageUnavailable =
              !consumePairInspections(boundary.guard->literals.size()) ||
              !consumePairInspections(boundary.guard->literals.size());
          if (guardStorageUnavailable) {
            return function_.emitError(
                "canonical sync pair-inspection limit exceeded");
          }
          edge.sourceGuard = *boundary.guard;
          edge.targetGuard = *boundary.guard;
          const SyncCoverGraphResult added = graph_.addEdge(std::move(edge));
          const bool duplicate =
              added.error == SyncCoverGraphError::DuplicateEdge;
          if (!added && !duplicate) {
            return function_.emitError()
                   << "cannot construct canonical sync fixed-barrier supply, "
                      "error="
                   << static_cast<unsigned>(added.error);
          }
        }
        remainingTargetResources.erase(targetNode.resource);
        boundary.remainingTargetResources =
            std::make_shared<const std::set<std::uint32_t>>(
                std::move(remainingTargetResources));
      }
      boundaries.erase(
          std::remove_if(boundaries.begin(), boundaries.end(),
                         [](const FixedBarrierBoundary &boundary) {
                           return boundary.remainingTargetResources->empty();
                         }),
          boundaries.end());
      state.fixedBarriers =
          std::make_shared<const std::vector<FixedBarrierBoundary>>(
              std::move(boundaries));
    }
  }

  static const IssueFrontier emptyFrontier;
  const IssueFrontier &currentFrontier =
      state.frontier ? *state.frontier : emptyFrontier;
  const auto currentPredecessors = currentFrontier.find(targetNode.resource);
  const ArrayRef<SyncCoverNodeId> predecessors =
      currentPredecessors == currentFrontier.end()
          ? ArrayRef<SyncCoverNodeId>{}
          : ArrayRef<SyncCoverNodeId>(currentPredecessors->second);
  for (SyncCoverNodeId source : predecessors) {
    if (!consumePairInspection()) {
      return function_.emitError(
          "canonical sync pair-inspection limit exceeded");
    }
    const SyncCoverNode &sourceNode = graph_.getNodes()[source];
    if (!syncCoverGuardsCompatible(sourceNode.guard, targetNode.guard)) {
      continue;
    }
    const std::optional<SyncCoverScopeId> scope =
        graph_.getLowestCommonScope(sourceNode.scope, targetNode.scope);
    if (!scope) {
      return function_.emitError("canonical sync issue-order scope failed");
    }
    SyncCoverEdge edge;
    edge.source = source;
    edge.target = target;
    edge.scope = *scope;
    edge.kind =
        isCompletionOrdered(sourceNode.resource, function_.getOperation())
            ? SyncCoverEdgeKind::CompletionPreservingIssueOrder
            : SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
    if (!graph_.addEdge(std::move(edge))) {
      return function_.emitError(
          "cannot construct canonical sync issue-order edge");
    }
  }
  std::size_t frontierStorage = currentFrontier.size();
  if (frontierStorage == std::numeric_limits<std::size_t>::max()) {
    return function_.emitError("canonical sync pair-inspection limit exceeded");
  }
  ++frontierStorage;
  for (const auto &[resource, nodes] : currentFrontier) {
    (void)resource;
    const bool storageOverflows =
        nodes.size() >
        std::numeric_limits<std::size_t>::max() - frontierStorage;
    if (storageOverflows) {
      return function_.emitError(
          "canonical sync pair-inspection limit exceeded");
    }
    frontierStorage += nodes.size();
  }
  if (!consumePairInspections(frontierStorage)) {
    return function_.emitError("canonical sync pair-inspection limit exceeded");
  }
  IssueFrontier nextFrontier = currentFrontier;
  nextFrontier[targetNode.resource] = {target};
  state.frontier =
      std::make_shared<const IssueFrontier>(std::move(nextFrontier));

  static const IssueHistoryHeads emptyIssued;
  const IssueHistoryHeads &currentIssued =
      state.issued ? *state.issued : emptyIssued;
  const bool issuedStorageUnavailable =
      !consumePairInspections(currentIssued.size()) ||
      !consumePairInspections(2);
  if (issuedStorageUnavailable) {
    return function_.emitError("canonical sync pair-inspection limit exceeded");
  }
  IssueHistoryHeads nextIssued = currentIssued;
  IssueHistory previous;
  const auto previousPosition = currentIssued.find(targetNode.resource);
  if (previousPosition != currentIssued.end()) {
    previous = previousPosition->second;
  }
  nextIssued[targetNode.resource] = std::make_shared<const IssueHistoryNode>(
      IssueHistoryNode{target, std::move(previous), {}});
  state.issued =
      std::make_shared<const IssueHistoryHeads>(std::move(nextIssued));
  return success();
}

LogicalResult
ProgramBuilder::collectIssuedSources(const IssueHistory &history,
                                     const SyncCoverGuard &barrierGuard,
                                     std::vector<SyncCoverNodeId> &sources) {
  llvm::SmallPtrSet<const IssueHistoryNode *, 32> visited;
  SmallVector<IssueHistory, 32> worklist;
  if (history) {
    worklist.push_back(history);
  }
  while (!worklist.empty()) {
    IssueHistory current = std::move(worklist.back());
    worklist.pop_back();
    if (!current || visited.contains(current.get())) {
      continue;
    }
    if (!consumePairInspection()) {
      return function_.emitError(
          "canonical sync pair-inspection limit exceeded");
    }
    visited.insert(current.get());
    if (current->issued &&
        syncCoverGuardsCompatible(graph_.getNodes()[*current->issued].guard,
                                  barrierGuard)) {
      sources.push_back(*current->issued);
    }
    if (current->first) {
      worklist.push_back(current->first);
    }
    if (current->second) {
      worklist.push_back(current->second);
    }
  }
  return success();
}

LogicalResult ProgramBuilder::addFixedBarrier(BarrierOp barrier,
                                              IssueOrderState &state) {
  auto context = contexts_.find(barrier->getParentRegion());
  if (context == contexts_.end()) {
    return barrier.emitError(
        "canonical sync lost fixed-barrier region context");
  }
  FixedBarrierBoundary boundary;
  boundary.operation = barrier.getOperation();
  const SyncCoverGuard &barrierGuard = context->second.guard;
  const std::uint32_t resource =
      static_cast<std::uint32_t>(barrier.getPipe().getPipe());
  std::vector<SyncCoverNodeId> sources;
  std::set<std::uint32_t> remainingTargetResources;
  static const IssueHistoryHeads emptyIssued;
  const IssueHistoryHeads &issued = state.issued ? *state.issued : emptyIssued;
  if (resource == static_cast<std::uint32_t>(PipelineType::PIPE_ALL)) {
    for (const auto &[sourceResource, history] : issued) {
      (void)sourceResource;
      if (failed(collectIssuedSources(history, barrierGuard, sources))) {
        return failure();
      }
    }
    for (const SyncCoverNode &node : graph_.getNodes()) {
      if (!consumePairInspection()) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      remainingTargetResources.insert(node.resource);
    }
  } else {
    const auto history = issued.find(resource);
    if (history != issued.end()) {
      if (failed(
              collectIssuedSources(history->second, barrierGuard, sources))) {
        return failure();
      }
    }
    remainingTargetResources.insert(resource);
  }
  llvm::sort(sources);
  sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
  const bool effectiveBoundary =
      !sources.empty() && !remainingTargetResources.empty();
  if (effectiveBoundary) {
    const std::size_t boundaryCount =
        state.fixedBarriers ? state.fixedBarriers->size() : 0;
    const bool boundaryStorageUnavailable =
        !consumePairInspections(sources.size()) ||
        !consumePairInspections(barrierGuard.literals.size()) ||
        !consumePairInspections(remainingTargetResources.size()) ||
        !consumePairInspections(boundaryCount) || !consumePairInspections(2);
    if (boundaryStorageUnavailable) {
      return function_.emitError(
          "canonical sync pair-inspection limit exceeded");
    }
    boundary.sources = std::make_shared<const std::vector<SyncCoverNodeId>>(
        std::move(sources));
    boundary.guard = std::make_shared<const SyncCoverGuard>(barrierGuard);
    boundary.remainingTargetResources =
        std::make_shared<const std::set<std::uint32_t>>(
            std::move(remainingTargetResources));
    std::vector<FixedBarrierBoundary> boundaries =
        state.fixedBarriers ? *state.fixedBarriers
                            : std::vector<FixedBarrierBoundary>{};
    boundaries.push_back(std::move(boundary));
    state.fixedBarriers =
        std::make_shared<const std::vector<FixedBarrierBoundary>>(
            std::move(boundaries));
  }
  return success();
}

LogicalResult ProgramBuilder::mergeIssueStates(IssueOrderState &target,
                                               const IssueOrderState &source) {
  if (target.frontier != source.frontier) {
    if (!target.frontier) {
      target.frontier = source.frontier;
    } else if (source.frontier) {
      std::size_t storage = target.frontier->size();
      const bool initialStorageOverflows =
          source.frontier->size() >
          std::numeric_limits<std::size_t>::max() - storage;
      if (initialStorageOverflows) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      storage += source.frontier->size();
      for (const auto &[resource, nodes] : *target.frontier) {
        (void)resource;
        const bool storageOverflows =
            nodes.size() > std::numeric_limits<std::size_t>::max() - storage;
        if (storageOverflows) {
          return function_.emitError(
              "canonical sync pair-inspection limit exceeded");
        }
        storage += nodes.size();
      }
      for (const auto &[resource, nodes] : *source.frontier) {
        (void)resource;
        const bool storageOverflows =
            nodes.size() > std::numeric_limits<std::size_t>::max() - storage;
        if (storageOverflows) {
          return function_.emitError(
              "canonical sync pair-inspection limit exceeded");
        }
        storage += nodes.size();
      }
      if (!consumePairInspections(storage)) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      IssueFrontier merged = *target.frontier;
      for (const auto &[resource, nodes] : *source.frontier) {
        std::vector<SyncCoverNodeId> &values = merged[resource];
        values.insert(values.end(), nodes.begin(), nodes.end());
        llvm::sort(values);
        values.erase(std::unique(values.begin(), values.end()), values.end());
      }
      target.frontier =
          std::make_shared<const IssueFrontier>(std::move(merged));
    }
  }

  if (target.issued != source.issued) {
    if (!target.issued) {
      target.issued = source.issued;
    } else if (source.issued) {
      const bool issuedMergeUnavailable =
          !consumePairInspections(target.issued->size()) ||
          !consumePairInspections(source.issued->size());
      if (issuedMergeUnavailable) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      IssueHistoryHeads merged = *target.issued;
      for (const auto &[resource, incoming] : *source.issued) {
        IssueHistory &existing = merged[resource];
        if (!existing) {
          existing = incoming;
        } else if (existing != incoming) {
          if (!consumePairInspection()) {
            return function_.emitError(
                "canonical sync pair-inspection limit exceeded");
          }
          existing = std::make_shared<const IssueHistoryNode>(
              IssueHistoryNode{std::nullopt, existing, incoming});
        }
      }
      target.issued =
          std::make_shared<const IssueHistoryHeads>(std::move(merged));
    }
  }

  if (target.fixedBarriers == source.fixedBarriers) {
    return success();
  }
  if (!target.fixedBarriers) {
    target.fixedBarriers = source.fixedBarriers;
    return success();
  }
  if (!source.fixedBarriers) {
    return success();
  }
  const bool boundaryMergeUnavailable =
      !consumePairInspections(target.fixedBarriers->size()) ||
      !consumePairInspections(source.fixedBarriers->size());
  if (boundaryMergeUnavailable) {
    return function_.emitError("canonical sync pair-inspection limit exceeded");
  }
  std::vector<FixedBarrierBoundary> merged = *target.fixedBarriers;
  for (const FixedBarrierBoundary &incoming : *source.fixedBarriers) {
    auto existing = merged.end();
    for (auto candidate = merged.begin(); candidate != merged.end();
         ++candidate) {
      if (!consumePairInspection()) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      if (candidate->operation == incoming.operation) {
        existing = candidate;
        break;
      }
    }
    if (existing == merged.end()) {
      merged.push_back(incoming);
      continue;
    }
    if (existing->sources != incoming.sources) {
      const bool sourceMergeUnavailable =
          !consumePairInspections(existing->sources->size()) ||
          !consumePairInspections(incoming.sources->size()) ||
          !consumePairInspection();
      if (sourceMergeUnavailable) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      std::vector<SyncCoverNodeId> sources = *existing->sources;
      sources.insert(sources.end(), incoming.sources->begin(),
                     incoming.sources->end());
      llvm::sort(sources);
      sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
      existing->sources = std::make_shared<const std::vector<SyncCoverNodeId>>(
          std::move(sources));
    }
    if (existing->remainingTargetResources !=
        incoming.remainingTargetResources) {
      const bool resourceMergeUnavailable =
          !consumePairInspections(existing->remainingTargetResources->size()) ||
          !consumePairInspections(incoming.remainingTargetResources->size());
      if (resourceMergeUnavailable) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      std::set<std::uint32_t> remainingTargetResources =
          *existing->remainingTargetResources;
      remainingTargetResources.insert(
          incoming.remainingTargetResources->begin(),
          incoming.remainingTargetResources->end());
      existing->remainingTargetResources =
          std::make_shared<const std::set<std::uint32_t>>(
              std::move(remainingTargetResources));
    }
  }
  target.fixedBarriers =
      std::make_shared<const std::vector<FixedBarrierBoundary>>(
          std::move(merged));
  return success();
}

LogicalResult ProgramBuilder::addForwardDependencies() {
  for (SyncCoverNodeId target = 0; target < nodeBindings_.size(); ++target) {
    for (Value operand : nodeBindings_[target].operation->getOperands()) {
      llvm::SetVector<SyncCoverNodeId> producers;
      llvm::DenseSet<Value> visited;
      if (failed(collectScheduledProducers(operand, producers, visited))) {
        return failure();
      }
      for (SyncCoverNodeId source : producers) {
        const bool unavailable =
            source >= target ||
            !syncCoverGuardsCompatible(graph_.getNodes()[source].guard,
                                       graph_.getNodes()[target].guard) ||
            isDemandImplicitlyComplete(source, target);
        if (unavailable) {
          continue;
        }
        if (failed(addDemand(source, target, 0, 0, SyncCoverDemandKind::SSA,
                             {}))) {
          return failure();
        }
      }
    }
  }

  for (SyncCoverNodeId source = 0; source < nodeBindings_.size(); ++source) {
    for (SyncCoverNodeId target : storageConflictPeers_[source]) {
      if (target <= source) {
        continue;
      }
      const bool unavailable =
          nodeBindings_[source].operation == nodeBindings_[target].operation ||
          !syncCoverGuardsCompatible(graph_.getNodes()[source].guard,
                                     graph_.getNodes()[target].guard) ||
          isDemandImplicitlyComplete(source, target);
      if (unavailable) {
        continue;
      }
      if (failed(addMemoryHazards(source, target, nullptr, 0))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult ProgramBuilder::addRecurrenceDependencies() {
  for (const auto &[loopOperation, loopScope] : loopScopes_) {
    auto loop = dyn_cast<scf::ForOp>(loopOperation);
    if (!loop) {
      return loopOperation->emitError(
          "canonical sync supports scf.for recurrences only");
    }
    const auto indexed = loopNodes_.find(loopOperation);
    if (indexed == loopNodes_.end()) {
      continue;
    }
    SmallVector<SyncCoverNodeId, 16> loopNodes = indexed->second;
    llvm::sort(loopNodes);
    for (SyncCoverNodeId source : loopNodes) {
      for (SyncCoverNodeId target : storageConflictPeers_[source]) {
        if (!std::binary_search(loopNodes.begin(), loopNodes.end(), target)) {
          continue;
        }
        if (isDemandImplicitlyComplete(source, target)) {
          continue;
        }
        const unsigned maximumDistance =
            maximumRecurrenceDistance(source, target);
        HazardKinds covered;
        for (unsigned distance = 1; distance <= maximumDistance; ++distance) {
          if (executesOnlyOnFirstIteration(nodeBindings_[target].operation,
                                           loop)) {
            break;
          }
          if (!consumePairInspection()) {
            return function_.emitError(
                "canonical sync pair-inspection limit exceeded");
          }
          FailureOr<HazardKinds> hazards = addMemoryHazards(
              source, target, loopOperation, distance, loopScope, covered);
          if (failed(hazards)) {
            return failure();
          }
          covered.raw |= hazards->raw;
          covered.war |= hazards->war;
          covered.waw |= hazards->waw;
          if (covered.raw && covered.war && covered.waw) {
            break;
          }
        }
      }
    }
  }
  return success();
}

bool ProgramBuilder::isDemandImplicitlyComplete(SyncCoverNodeId source,
                                                SyncCoverNodeId target) {
  const SyncCoverNode &sourceNode = graph_.getNodes()[source];
  const SyncCoverNode &targetNode = graph_.getNodes()[target];
  return sourceNode.resource == targetNode.resource &&
         isCompletionOrdered(sourceNode.resource, function_.getOperation());
}

LogicalResult ProgramBuilder::collectScheduledProducers(
    Value value, llvm::SetVector<SyncCoverNodeId> &producers,
    llvm::DenseSet<Value> &visited) const {
  if (!value || !visited.insert(value).second) {
    return success();
  }
  Operation *definition = value.getDefiningOp();
  if (!definition) {
    return success();
  }
  auto scheduled = operationNodes_.find(definition);
  if (scheduled != operationNodes_.end()) {
    const auto completion = ssaCompletionNodes_.find(value);
    if (completion == ssaCompletionNodes_.end()) {
      return definition->emitError(
          "canonical sync has no completion phase for this SSA result");
    }
    producers.insert(completion->second);
    return success();
  }
  // Storage handles do not represent asynchronously produced data. Treat
  // allocation and declaration operations as explicit provenance roots even
  // though their memory-effect interfaces are not side-effect-free.
  if (isSyncStorageProvenanceRoot(definition)) {
    return success();
  }
  const bool unsupportedDefinition =
      !isMemoryEffectFree(definition) || definition->getNumRegions() != 0;
  if (unsupportedDefinition) {
    return definition->emitError(
        "canonical sync cannot trace SSA provenance through this unscheduled "
        "effectful or region operation");
  }
  for (Value operand : definition->getOperands()) {
    if (failed(collectScheduledProducers(operand, producers, visited))) {
      return failure();
    }
  }
  return success();
}

bool ProgramBuilder::hasIntrinsicMmadAccumulatorOrdering(
    SyncCoverNodeId source, SyncCoverNodeId target,
    const SyncCoverStorageAccess &sourceAccess,
    const SyncCoverStorageAccess &targetAccess) {
  const bool unsupported =
      !isTargetArchA3(function_.getOperation()) ||
      graph_.getNodes()[source].resource !=
          static_cast<std::uint32_t>(PipelineType::PIPE_M) ||
      graph_.getNodes()[target].resource !=
          static_cast<std::uint32_t>(PipelineType::PIPE_M) ||
      !isa<TMatmulOp, TMatmulAccOp>(nodeBindings_[source].operation) ||
      !isa<TMatmulAccOp>(nodeBindings_[target].operation) ||
      sourceAccess.domain != targetAccess.domain ||
      !syncCoverStorageModeWrites(sourceAccess.mode) ||
      !syncCoverStorageModeReads(targetAccess.mode) ||
      !syncCoverStorageModeWrites(targetAccess.mode) ||
      !sourceAccess.exactPhysical || !targetAccess.exactPhysical;
  if (unsupported) {
    return false;
  }
  return sourceAccess.extent.begin == targetAccess.extent.begin &&
         sourceAccess.extent.end == targetAccess.extent.end;
}

unsigned
ProgramBuilder::maximumRecurrenceDistance(SyncCoverNodeId source,
                                          SyncCoverNodeId target) const {
  unsigned maximum = 1;
  for (std::size_t firstIndex : nodeAccessIndices_[source]) {
    const ExtractedAccess &first = extractedAccesses_[firstIndex];
    for (std::size_t secondIndex : nodeAccessIndices_[target]) {
      const ExtractedAccess &second = extractedAccesses_[secondIndex];
      if (first.space != second.space || gmAccessesAreNoAlias(first, second)) {
        continue;
      }
      const std::size_t slots =
          std::max(first.graphAccesses.size(), second.graphAccesses.size());
      if (slots <= kMaximumSlotCount) {
        maximum = std::max(maximum, static_cast<unsigned>(slots));
      }
    }
  }
  return maximum;
}

FailureOr<HazardKinds>
ProgramBuilder::addMemoryHazards(SyncCoverNodeId source, SyncCoverNodeId target,
                                 Operation *loop, unsigned distance,
                                 SyncCoverScopeId recurrenceScope,
                                 HazardKinds alreadyCovered) {
  HazardKinds result;
  for (std::size_t firstIndex : nodeAccessIndices_[source]) {
    const ExtractedAccess &first = extractedAccesses_[firstIndex];
    for (std::size_t secondIndex : nodeAccessIndices_[target]) {
      const ExtractedAccess &second = extractedAccesses_[secondIndex];
      if (first.space != second.space || gmAccessesAreNoAlias(first, second)) {
        continue;
      }
      if (!consumePairInspection()) {
        function_.emitError("canonical sync pair-inspection limit exceeded");
        return failure();
      }
      FailureOr<std::vector<std::pair<unsigned, unsigned>>> ordinalPairs =
          getOrdinalPairs(first, second, loop, distance);
      if (failed(ordinalPairs)) {
        return failure();
      }
      std::vector<SyncCoverStorageWitnessId> raw;
      std::vector<SyncCoverStorageWitnessId> war;
      std::vector<SyncCoverStorageWitnessId> waw;
      for (const auto &[firstOrdinal, secondOrdinal] : *ordinalPairs) {
        if (!consumePairInspection()) {
          function_.emitError("canonical sync pair-inspection limit exceeded");
          return failure();
        }
        const bool invalidOrdinal =
            firstOrdinal >= first.graphAccesses.size() ||
            secondOrdinal >= second.graphAccesses.size();
        if (invalidOrdinal) {
          continue;
        }
        const SyncCoverStorageAccess &firstAccess =
            graph_.getStorageAccesses()[first.graphAccesses[firstOrdinal]];
        const SyncCoverStorageAccess &secondAccess =
            graph_.getStorageAccesses()[second.graphAccesses[secondOrdinal]];
        if (hasIntrinsicMmadAccumulatorOrdering(source, target, firstAccess,
                                                secondAccess)) {
          continue;
        }
        const bool disjoint =
            firstAccess.domain != secondAccess.domain ||
            firstAccess.extent.begin >= secondAccess.extent.end ||
            secondAccess.extent.begin >= firstAccess.extent.end;
        if (disjoint) {
          continue;
        }
        const SyncCoverGraphResult witness =
            graph_.addStorageWitness(firstAccess.id, secondAccess.id);
        if (!witness) {
          return failure();
        }
        const bool rawHazard = syncCoverStorageModeWrites(firstAccess.mode) &&
                               syncCoverStorageModeReads(secondAccess.mode);
        if (rawHazard) {
          raw.push_back(*witness.index);
        }
        const bool warHazard = syncCoverStorageModeReads(firstAccess.mode) &&
                               syncCoverStorageModeWrites(secondAccess.mode);
        if (warHazard) {
          war.push_back(*witness.index);
        }
        const bool wawHazard = syncCoverStorageModeWrites(firstAccess.mode) &&
                               syncCoverStorageModeWrites(secondAccess.mode);
        if (wawHazard) {
          waw.push_back(*witness.index);
        }
      }
      if (!alreadyCovered.raw && !raw.empty()) {
        if (failed(addDemand(source, target, recurrenceScope, distance,
                             SyncCoverDemandKind::MemoryRAW, raw))) {
          return failure();
        }
        result.raw = true;
      }
      if (!alreadyCovered.war && !war.empty()) {
        if (failed(addDemand(source, target, recurrenceScope, distance,
                             SyncCoverDemandKind::MemoryWAR, war))) {
          return failure();
        }
        result.war = true;
      }
      if (!alreadyCovered.waw && !waw.empty()) {
        if (failed(addDemand(source, target, recurrenceScope, distance,
                             SyncCoverDemandKind::MemoryWAW, waw))) {
          return failure();
        }
        result.waw = true;
      }
    }
  }
  return result;
}

LogicalResult
ProgramBuilder::addDemand(SyncCoverNodeId source, SyncCoverNodeId target,
                          SyncCoverScopeId scope, unsigned distance,
                          SyncCoverDemandKind kind,
                          std::vector<SyncCoverStorageWitnessId> witnesses) {
  if (distance == 0) {
    const std::optional<SyncCoverScopeId> common = graph_.getLowestCommonScope(
        graph_.getNodes()[source].scope, graph_.getNodes()[target].scope);
    if (!common) {
      return function_.emitError("canonical sync demand scope failed");
    }
    scope = *common;
  }
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.scope = scope;
  demand.distance = distance;
  demand.provenanceKinds = {kind};
  demand.storageWitnesses = std::move(witnesses);
  const SyncCoverGraphResult added = graph_.addDemand(std::move(demand));
  if (!added) {
    return function_.emitError("cannot construct canonical sync demand")
           << " (error " << static_cast<unsigned>(added.error) << ')';
  }
  return success();
}
