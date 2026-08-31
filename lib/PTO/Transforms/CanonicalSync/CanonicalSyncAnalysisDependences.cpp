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
#include <numeric>
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

std::optional<unsigned> checkedLeastCommonMultiple(unsigned first,
                                                   unsigned second) {
  if (first == 0 || second == 0) {
    return std::nullopt;
  }
  const unsigned divisor = std::gcd(first, second);
  const unsigned reduced = first / divisor;
  const bool overflows =
      reduced > std::numeric_limits<unsigned>::max() / second;
  if (overflows) {
    return std::nullopt;
  }
  return reduced * second;
}

bool checkedAddSize(std::size_t &value, std::size_t amount) {
  const bool overflows =
      amount > std::numeric_limits<std::size_t>::max() - value;
  if (overflows) {
    return false;
  }
  value += amount;
  return true;
}

bool checkedMultiplySize(std::size_t first, std::size_t second,
                         std::size_t &result) {
  const bool overflows =
      first != 0 && second > std::numeric_limits<std::size_t>::max() / first;
  if (overflows) {
    return false;
  }
  result = first * second;
  return true;
}

std::size_t logarithmicWorkBound(std::size_t count) {
  std::size_t result = 1;
  while (count > 1) {
    count = count / 2 + count % 2;
    ++result;
  }
  return result;
}

constexpr std::size_t kMaximumRecurrencePhaseOrbitCacheEntries = 4096;
constexpr std::size_t kMaximumCachedRecurrenceGuardLiterals = 32;

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

LogicalResult ProgramBuilder::addCertifiedCompletionFrontiers() {
  const SyncCoverGraph &graph = graph_;
  const std::vector<SyncCoverEdge> fixedEdges = graph.getEdges();
  for (const SyncCoverEdge &edge : fixedEdges) {
    const bool fixedIssue =
        edge.distance == 0 &&
        (edge.kind == SyncCoverEdgeKind::CompletionPreservingIssueOrder ||
         edge.kind == SyncCoverEdgeKind::NonCompletionPreservingIssueOrder);
    if (!fixedIssue) {
      continue;
    }
    const SyncCoverNode &edgeSource = graph.getNodes()[edge.source];
    const SyncCoverNode &edgeTarget = graph.getNodes()[edge.target];
    const bool certifiedContract =
        canSignalPrefixCompletion(edgeSource.resource, targetCapabilities_);
    const bool sameControlChain =
        edgeSource.scope == edgeTarget.scope &&
        edgeSource.guard.literals == edgeTarget.guard.literals;
    if (!certifiedContract || !sameControlChain ||
        edgeSource.physicalAnchor == edgeTarget.physicalAnchor) {
      continue;
    }

    const SyncCoverNodeId source = edge.source;
    const SyncCoverNodeId target = edge.target;
    const SyncCoverNode &sourceNode = graph.getNodes()[source];
    const SyncCoverNode &targetNode = graph.getNodes()[target];
    if (source == target || sourceNode.order >= targetNode.order ||
        sourceNode.scope != targetNode.scope ||
        sourceNode.guard.literals != targetNode.guard.literals) {
      continue;
    }
    SyncCoverEdge frontier;
    frontier.source = source;
    frontier.target = target;
    frontier.kind = SyncCoverEdgeKind::CertifiedCompletionFrontier;
    frontier.scope = sourceNode.scope;
    frontier.sourceGuard = sourceNode.guard;
    frontier.targetGuard = targetNode.guard;
    const SyncCoverGraphResult added = graph_.addEdge(std::move(frontier));
    if (!added) {
      return function_.emitError(
          "cannot construct canonical sync certified completion frontier");
    }
    if (!added.index || graph_.getEdges()[*added.index].kind !=
                            SyncCoverEdgeKind::CertifiedCompletionFrontier) {
      continue;
    }
    if (!graph_.addCompletionDominance(source, target)) {
      return function_.emitError(
          "cannot register canonical sync completion dominance");
    }
  }
  return success();
}

LogicalResult ProgramBuilder::addTargetCompletionCertificates(
    const CanonicalSyncTargetCapabilities &capabilities) {
  const bool mte1L0ReadyEnabled =
      capabilities.mte1L0ReadySetCompletesPrefix.isEnabled();
  const bool accumulatorBoundaryEnabled =
      capabilities.mToFixAccumulatorBoundaryCompletes.isEnabled();
  if (!mte1L0ReadyEnabled && !accumulatorBoundaryEnabled) {
    return success();
  }
  const auto domainFor =
      [&](AddressSpace space) -> std::optional<SyncCoverStorageDomainId> {
    const auto domain = storageDomains_.find(space);
    return domain == storageDomains_.end()
               ? std::nullopt
               : std::optional<SyncCoverStorageDomainId>(domain->second);
  };
  const std::optional<SyncCoverStorageDomainId> left =
      domainFor(AddressSpace::LEFT);
  const std::optional<SyncCoverStorageDomainId> right =
      domainFor(AddressSpace::RIGHT);
  const std::optional<SyncCoverStorageDomainId> accumulator =
      domainFor(AddressSpace::ACC);
  const SyncCoverGraph &graph = graph_;
  const auto exactRawDomain =
      [&](const SyncCoverDemand &demand,
          ArrayRef<SyncCoverStorageDomainId> admittedDomains)
      -> std::optional<SyncCoverStorageDomainId> {
    if (demand.distance != 0 ||
        !llvm::is_contained(demand.provenanceKinds,
                            SyncCoverDemandKind::MemoryRAW)) {
      return std::nullopt;
    }
    for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
      if (witnessId >= graph.getStorageWitnesses().size()) {
        continue;
      }
      const SyncCoverStorageWitness &witness =
          graph.getStorageWitnesses()[witnessId];
      const SyncCoverStorageAccess &source =
          graph.getStorageAccesses()[witness.sourceAccess];
      const SyncCoverStorageAccess &target =
          graph.getStorageAccesses()[witness.targetAccess];
      if (source.node == demand.source && target.node == demand.target &&
          source.domain == target.domain && source.exactPhysical &&
          target.exactPhysical && syncCoverStorageModeWrites(source.mode) &&
          syncCoverStorageModeReads(target.mode) &&
          llvm::is_contained(admittedDomains, source.domain)) {
        return source.domain;
      }
    }
    return std::nullopt;
  };

  using CertifiedDemand =
      std::pair<SyncCoverDemandId, SyncCoverStorageDomainId>;
  std::map<SyncCoverNodeId, std::vector<CertifiedDemand>> mte1Uses;
  std::map<std::pair<SyncCoverNodeId, SyncCoverNodeId>,
           std::vector<CertifiedDemand>>
      accumulatorUses;
  std::vector<SyncCoverStorageDomainId> l0Domains;
  if (left) {
    l0Domains.push_back(*left);
  }
  if (right) {
    l0Domains.push_back(*right);
  }
  const std::vector<SyncCoverStorageDomainId> accumulatorDomains =
      accumulator ? std::vector<SyncCoverStorageDomainId>{*accumulator}
                  : std::vector<SyncCoverStorageDomainId>{};
  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverNode &source = graph.getNodes()[demand.source];
    const SyncCoverNode &target = graph.getNodes()[demand.target];
    const SyncCoverNode &physicalSource = graph.getNodes()[source.physicalExit];
    const SyncCoverNode &physicalTarget =
        graph.getNodes()[target.physicalAnchor];
    const bool oneControlChain =
        source.scope == target.scope && source.scope == physicalSource.scope &&
        source.scope == physicalTarget.scope &&
        source.guard.literals == target.guard.literals &&
        source.guard.literals == physicalSource.guard.literals &&
        source.guard.literals == physicalTarget.guard.literals &&
        physicalSource.order < physicalTarget.order;
    if (!oneControlChain) {
      continue;
    }
    const bool mte1L0ReadyDemand =
        mte1L0ReadyEnabled && !l0Domains.empty() &&
        source.resource ==
            static_cast<std::uint32_t>(PipelineType::PIPE_MTE1) &&
        target.resource == static_cast<std::uint32_t>(PipelineType::PIPE_M);
    if (mte1L0ReadyDemand) {
      if (const auto domain = exactRawDomain(demand, l0Domains)) {
        mte1Uses[target.physicalAnchor].push_back({demandId, *domain});
      }
    }
    const bool accumulatorBoundaryDemand =
        accumulatorBoundaryEnabled && accumulator &&
        source.resource == static_cast<std::uint32_t>(PipelineType::PIPE_M) &&
        target.resource == static_cast<std::uint32_t>(PipelineType::PIPE_FIX);
    if (accumulatorBoundaryDemand) {
      if (const auto domain = exactRawDomain(demand, accumulatorDomains)) {
        accumulatorUses[{source.physicalExit, target.physicalAnchor}].push_back(
            {demandId, *domain});
      }
    }
  }

  for (const auto &[target, certifiedDemands] : mte1Uses) {
    // A singleton already has the ordinary exact MTE1 event.  The target
    // contract matters only when one final ready set replaces multiple source
    // events belonging to the same exact L0 consumer lifecycle.
    if (certifiedDemands.size() < 2) {
      continue;
    }
    SyncCoverNodeId completionNode = 0;
    std::vector<SyncCoverDemandId> demands;
    std::vector<SyncCoverStorageDomainId> domains;
    for (const auto &[demandId, domain] : certifiedDemands) {
      const SyncCoverNodeId physicalExit =
          graph.getNodes()[graph.getDemands()[demandId].source].physicalExit;
      if (demands.empty() || graph.getNodes()[completionNode].order <
                                 graph.getNodes()[physicalExit].order) {
        completionNode = physicalExit;
      }
      demands.push_back(demandId);
      domains.push_back(domain);
    }
    const SyncCoverGraphResult added = graph_.addTargetCompletionCertificate(
        SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix, completionNode,
        target, static_cast<std::uint32_t>(PipelineType::PIPE_MTE1),
        static_cast<std::uint32_t>(PipelineType::PIPE_M), std::move(domains),
        std::move(demands));
    if (!added) {
      return function_.emitError(
                 "cannot register canonical sync A3 MTE1 L0-ready "
                 "certificate, graph_error=")
             << static_cast<unsigned>(added.error);
    }
  }
  for (const auto &[anchors, certifiedDemands] : accumulatorUses) {
    std::vector<SyncCoverDemandId> demands;
    std::vector<SyncCoverStorageDomainId> domains;
    for (const auto &[demandId, domain] : certifiedDemands) {
      demands.push_back(demandId);
      domains.push_back(domain);
    }
    const SyncCoverGraphResult added = graph_.addTargetCompletionCertificate(
        SyncCoverTargetCompletionKind::MToFixAccumulatorBoundary, anchors.first,
        anchors.second, static_cast<std::uint32_t>(PipelineType::PIPE_M),
        static_cast<std::uint32_t>(PipelineType::PIPE_FIX), std::move(domains),
        std::move(demands));
    if (!added) {
      return function_.emitError(
                 "cannot register canonical sync A3 M-to-FIX certificate, "
                 "graph_error=")
             << static_cast<unsigned>(added.error);
    }
  }
  return success();
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
  if (failed(recordBlockingBarrierPrefixes(target, state))) {
    return failure();
  }
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
          edge.suppliedRequirements =
              targetCapabilities_.pipeBarrierOrderingRequirements;
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
    edge.kind = isCompletionOrdered(sourceNode.resource, targetCapabilities_)
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
ProgramBuilder::recordBlockingBarrierPrefixes(SyncCoverNodeId target,
                                              IssueOrderState &state) {
  const SyncCoverNode &targetNode = graph_.getNodes()[target];
  const SyncCoverNodeId physicalTarget = targetNode.physicalAnchor;
  static const IssueHistoryHeads emptyIssued;
  const IssueHistoryHeads &issued = state.issued ? *state.issued : emptyIssued;
  for (const auto &[resource, carryKind] :
       graph_.getResourceRecurrenceCarryKinds()) {
    (void)carryKind;
    if (!graph_.supportsBlockingTargetedBarrier(resource)) {
      continue;
    }
    const auto key = std::make_pair(resource, physicalTarget);
    if (graph_.getBlockingTargetedBarrierPrefixes().find(key) !=
        graph_.getBlockingTargetedBarrierPrefixes().end()) {
      continue;
    }
    std::vector<SyncCoverNodeId> sources;
    const auto history = issued.find(resource);
    if (history != issued.end() &&
        failed(
            collectIssuedSources(history->second, targetNode.guard, sources))) {
      return failure();
    }
    const SyncCoverGraphResult recorded =
        graph_.setBlockingTargetedBarrierPrefix(resource, physicalTarget,
                                                std::move(sources));
    if (!recorded) {
      return function_.emitError(
          "cannot record canonical sync blocking-barrier prefix");
    }
  }
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
  const bool allPipe =
      resource == static_cast<std::uint32_t>(PipelineType::PIPE_ALL);
  if (allPipe) {
    for (const auto &[sourceResource, history] : issued) {
      (void)sourceResource;
      if (failed(collectIssuedSources(history, barrierGuard, sources))) {
        return failure();
      }
    }
  } else {
    const auto history = issued.find(resource);
    if (history != issued.end()) {
      if (failed(
              collectIssuedSources(history->second, barrierGuard, sources))) {
        return failure();
      }
    }
  }
  if (allPipe) {
    for (const SyncCoverNode &node : graph_.getNodes()) {
      if (!consumePairInspection()) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      remainingTargetResources.insert(node.resource);
    }
  } else {
    if (graph_.supportsBlockingTargetedBarrier(resource)) {
      remainingTargetResources.insert(resource);
    }
    for (const SyncCoverNode &node : graph_.getNodes()) {
      if (!consumePairInspection()) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      if (graph_.supportsCrossResourceTargetedBarrier(resource,
                                                      node.resource)) {
        remainingTargetResources.insert(node.resource);
      }
    }
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
    llvm::SetVector<SyncCoverNodeId> producers;
    llvm::DenseSet<Value> discovered;
    for (Value operand : nodeBindings_[target].ssaOperands) {
      if (failed(walkSsaProvenance(operand, nullptr, &producers, discovered,
                                   nullptr))) {
        return failure();
      }
    }
    for (SyncCoverNodeId source : producers) {
      const bool unavailable =
          source >= target ||
          !syncCoverGuardsCompatible(graph_.getNodes()[source].guard,
                                     graph_.getNodes()[target].guard) ||
          isDemandTargetCertifiedComplete(source, target);
      if (unavailable) {
        continue;
      }
      if (failed(addDemand(source, target, 0, 0, {SyncCoverDemandKind::SSA},
                           {}))) {
        return failure();
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
          isDemandTargetCertifiedComplete(source, target);
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
        if (isDemandTargetCertifiedComplete(source, target)) {
          continue;
        }
        const SyncCoverGuard &sourceGuard = graph_.getNodes()[source].guard;
        const SyncCoverGuard &targetGuard = graph_.getNodes()[target].guard;
        FailureOr<RecurrencePhaseOrbit> orbit =
            buildRecurrencePhaseOrbit(loopScope, sourceGuard, targetGuard);
        if (failed(orbit)) {
          return failure();
        }
        if (!orbit->staticallyReachable) {
          continue;
        }
        FailureOr<unsigned> maximumDistance =
            maximumRecurrenceDistance(source, target, *orbit);
        if (failed(maximumDistance)) {
          return failure();
        }
        HazardWitnesses covered;
        for (unsigned distance = 1; distance <= *maximumDistance; ++distance) {
          if (executesOnlyOnFirstIteration(nodeBindings_[target].operation,
                                           loop)) {
            break;
          }
          if (!consumePairInspection()) {
            return function_.emitError(
                "canonical sync pair-inspection limit exceeded");
          }
          const std::vector<std::size_t> reachableSourcePhases =
              getReachableRecurrenceSourcePhases(*orbit, distance);
          if (reachableSourcePhases.empty()) {
            continue;
          }
          if (failed(addMemoryHazards(source, target, loopOperation, distance,
                                      loopScope, &covered,
                                      &reachableSourcePhases, orbit->period))) {
            return failure();
          }
        }
      }
    }
  }
  return success();
}

bool ProgramBuilder::isDemandTargetCertifiedComplete(SyncCoverNodeId source,
                                                     SyncCoverNodeId target) {
  const SyncCoverNode &sourceNode = graph_.getNodes()[source];
  const SyncCoverNode &targetNode = graph_.getNodes()[target];
  return sourceNode.resource == targetNode.resource &&
         isCompletionOrdered(sourceNode.resource, targetCapabilities_);
}

LogicalResult ProgramBuilder::collectEnclosingLoopControls(
    Operation *operation, llvm::DenseMap<Value, Value> &inductionLowerBounds) {
  for (Operation *parent = operation->getParentOp(); parent != nullptr;
       parent = parent->getParentOp()) {
    if (!consumePairInspection()) {
      return operation->emitError(
          "canonical sync pair-inspection limit exceeded");
    }
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop) {
      continue;
    }
    if (!consumePairInspection()) {
      return operation->emitError(
          "canonical sync pair-inspection limit exceeded");
    }
    inductionLowerBounds.insert({loop.getInductionVar(), loop.getLowerBound()});
  }
  return success();
}

LogicalResult ProgramBuilder::walkSsaProvenance(
    Value seed, const llvm::DenseMap<Value, Value> *trackedLoopInductions,
    llvm::SetVector<SyncCoverNodeId> *producers,
    llvm::DenseSet<Value> &discovered, bool *reachesTrackedLoopInduction) {
  SmallVector<Value, 16> worklist;
  const auto enqueue = [&](Value value) -> LogicalResult {
    // Charge every incoming SSA edge before looking it up. Charge every
    // unique state again before growing either persistent traversal storage.
    if (!value) {
      return success();
    }
    if (!consumePairInspection()) {
      return function_.emitError(
          "canonical sync pair-inspection limit exceeded");
    }
    const bool alreadyDiscovered = discovered.count(value) != 0;
    if (alreadyDiscovered) {
      return success();
    }
    const bool stateLimitReached =
        discovered.size() >= options_.maximumPairInspections ||
        worklist.size() >= worklist.max_size();
    if (stateLimitReached || !consumePairInspection()) {
      return function_.emitError(
          "canonical sync pair-inspection limit exceeded");
    }
    discovered.insert(value);
    worklist.push_back(value);
    return success();
  };
  if (failed(enqueue(seed))) {
    return failure();
  }
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    const bool trackedInduction =
        trackedLoopInductions && trackedLoopInductions->count(current) != 0;
    if (trackedInduction) {
      if (reachesTrackedLoopInduction) {
        *reachesTrackedLoopInduction = true;
      }
      continue;
    }
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      Block *owner = argument.getOwner();
      const bool functionArgument = owner == &function_.getBody().front();
      auto loop = dyn_cast_or_null<scf::ForOp>(owner->getParentOp());
      const bool loopInduction = loop && owner == &loop.getRegion().front() &&
                                 argument == loop.getInductionVar();
      if (functionArgument || loopInduction) {
        continue;
      }
      Operation *parent = owner->getParentOp();
      if (parent) {
        return parent->emitError(
            "canonical sync cannot trace SSA provenance through this block "
            "argument");
      }
      return function_.emitError(
          "canonical sync cannot trace SSA provenance through this block "
          "argument");
    }
    Operation *definition = current.getDefiningOp();
    if (!definition) {
      continue;
    }
    auto scheduled = operationNodes_.find(definition);
    if (scheduled != operationNodes_.end()) {
      if (!producers) {
        continue;
      }
      const auto completion = ssaCompletionNodes_.find(current);
      if (completion == ssaCompletionNodes_.end()) {
        return definition->emitError(
            "canonical sync has no completion phase for this SSA result");
      }
      const bool newProducer = !producers->contains(completion->second);
      if (newProducer && !consumePairInspection()) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      producers->insert(completion->second);
      continue;
    }
    // Storage handles do not represent asynchronously produced data. Treat
    // allocation and declaration operations as explicit provenance roots even
    // though their memory-effect interfaces are not side-effect-free.
    if (isSyncStorageProvenanceRoot(definition)) {
      continue;
    }
    const bool unsupportedDefinition =
        !isMemoryEffectFree(definition) || definition->getNumRegions() != 0;
    if (unsupportedDefinition) {
      return definition->emitError(
          "canonical sync cannot trace SSA provenance through this "
          "unscheduled effectful or region operation");
    }
    for (Value operand : definition->getOperands()) {
      if (failed(enqueue(operand))) {
        return failure();
      }
    }
  }
  return success();
}

bool ProgramBuilder::hasIntrinsicMmadAccumulatorOrdering(
    SyncCoverNodeId source, SyncCoverNodeId target,
    const SyncCoverStorageAccess &sourceAccess,
    const SyncCoverStorageAccess &targetAccess) {
  const bool unsupported =
      !targetCapabilities_.intrinsicMmadAccumulatorOrdering.isEnabled() ||
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

FailureOr<RecurrencePhaseOrbit>
ProgramBuilder::buildRecurrencePhaseOrbit(SyncCoverScopeId loopScope,
                                          const SyncCoverGuard &sourceGuard,
                                          const SyncCoverGuard &targetGuard) {
  std::size_t literalCount = sourceGuard.literals.size();
  if (!checkedAddSize(literalCount, targetGuard.literals.size())) {
    function_.emitError("canonical sync recurrence phase work overflow");
    return failure();
  }
  const bool cacheable = literalCount <= kMaximumCachedRecurrenceGuardLiterals;
  std::size_t literalUnits = literalCount;
  if (!checkedAddSize(literalUnits, 1)) {
    function_.emitError("canonical sync recurrence phase work overflow");
    return failure();
  }
  std::optional<RecurrencePhaseOrbitCacheKey> cacheKey;
  if (cacheable) {
    std::size_t cacheEntries = recurrencePhaseOrbitCache_.size();
    std::size_t lookupWork = 0;
    const bool lookupUnavailable =
        !checkedAddSize(cacheEntries, 1) ||
        !checkedMultiplySize(literalUnits, logarithmicWorkBound(cacheEntries),
                             lookupWork) ||
        !consumePairInspections(lookupWork);
    if (lookupUnavailable) {
      function_.emitError("canonical sync pair-inspection limit exceeded");
      return failure();
    }
    cacheKey.emplace(loopScope, sourceGuard.literals, targetGuard.literals);
    const auto cached = recurrencePhaseOrbitCache_.find(*cacheKey);
    if (cached != recurrencePhaseOrbitCache_.end()) {
      return cached->second;
    }
  }

  std::size_t perLiteralWork =
      8 + 4 * kCanonicalSyncMaximumPeriodicRecurrenceStates;
  std::size_t buildWork = 0;
  const bool buildUnavailable =
      !checkedMultiplySize(literalUnits, perLiteralWork, buildWork) ||
      !consumePairInspections(buildWork);
  if (buildUnavailable) {
    function_.emitError("canonical sync pair-inspection limit exceeded");
    return failure();
  }

  RecurrencePhaseOrbit result;
  const auto retainInCache = [&]() {
    const bool cacheHasRoom =
        cacheKey && recurrencePhaseOrbitCache_.size() <
                        kMaximumRecurrencePhaseOrbitCacheEntries;
    if (cacheHasRoom) {
      recurrencePhaseOrbitCache_.emplace(std::move(*cacheKey), result);
    }
  };
  const std::vector<SyncCoverControl> &controls = graph_.getControls();
  const auto guardIsValid = [&](const SyncCoverGuard &guard) {
    return llvm::all_of(guard.literals,
                        [&](const SyncCoverGuardLiteral &literal) {
                          return literal.control < controls.size();
                        });
  };
  const bool invalidGuard =
      !guardIsValid(sourceGuard) || !guardIsValid(targetGuard);
  if (invalidGuard) {
    function_.emitError("canonical sync recurrence has an invalid guard");
    return failure();
  }
  std::size_t sourceIndex = 0;
  std::size_t targetIndex = 0;
  const std::size_t sourceLiterals = sourceGuard.literals.size();
  const std::size_t targetLiterals = targetGuard.literals.size();
  while (sourceIndex < sourceLiterals && targetIndex < targetLiterals) {
    const SyncCoverGuardLiteral &source = sourceGuard.literals[sourceIndex];
    const SyncCoverGuardLiteral &target = targetGuard.literals[targetIndex];
    if (source.control < target.control) {
      ++sourceIndex;
      continue;
    }
    if (target.control < source.control) {
      ++targetIndex;
      continue;
    }
    const bool iterationLocal =
        graph_.scopeContains(loopScope, controls[source.control].scope);
    if (!iterationLocal && source.alternative != target.alternative) {
      result.staticallyReachable = false;
      result.sourceActivePhases = 0;
      result.targetActivePhases = 0;
      retainInCache();
      return result;
    }
    ++sourceIndex;
    ++targetIndex;
  }

  std::vector<SyncCoverControlId> periodicControls;
  const auto collectPeriodicControls = [&](const SyncCoverGuard &guard) {
    for (const SyncCoverGuardLiteral &literal : guard.literals) {
      const std::optional<SyncCoverControlPhaseRelation> &relation =
          controls[literal.control].phaseRelation;
      if (relation && relation->loopScope == loopScope) {
        periodicControls.push_back(literal.control);
      }
    }
  };
  collectPeriodicControls(sourceGuard);
  collectPeriodicControls(targetGuard);
  llvm::sort(periodicControls);
  periodicControls.erase(
      std::unique(periodicControls.begin(), periodicControls.end()),
      periodicControls.end());
  if (periodicControls.empty()) {
    retainInCache();
    return result;
  }

  unsigned jointPeriod = 1;
  for (SyncCoverControlId control : periodicControls) {
    const SyncCoverControlPhaseRelation &relation =
        *controls[control].phaseRelation;
    const std::size_t phases = relation.nextPhase.size();
    const bool invalidRelation =
        phases == 0 || phases > kCanonicalSyncMaximumPeriodicRecurrenceStates ||
        relation.initialPhase >= phases ||
        relation.activeAlternative.size() != phases ||
        llvm::any_of(relation.nextPhase,
                     [&](std::size_t phase) { return phase >= phases; });
    if (invalidRelation) {
      function_.emitError(
          "canonical sync recurrence has an invalid periodic relation");
      return failure();
    }
    std::vector<bool> visited(phases, false);
    std::size_t state = relation.initialPhase;
    visited[state] = true;
    unsigned controlPeriod = 0;
    while (true) {
      state = relation.nextPhase[state];
      ++controlPeriod;
      if (state == relation.initialPhase) {
        break;
      }
      if (visited[state]) {
        function_.emitError(
            "canonical sync does not support transient periodic recurrence "
            "states");
        return failure();
      }
      if (controlPeriod == options_.maximumPeriodicRecurrenceStates) {
        function_.emitError(
            "canonical sync periodic recurrence state limit exceeded");
        return failure();
      }
      visited[state] = true;
    }
    const std::optional<unsigned> combined =
        checkedLeastCommonMultiple(jointPeriod, controlPeriod);
    if (!combined || *combined > options_.maximumPeriodicRecurrenceStates) {
      function_.emitError(
          "canonical sync periodic recurrence state limit exceeded");
      return failure();
    }
    jointPeriod = *combined;
  }

  result.period = jointPeriod;
  const std::uint32_t allPhases =
      (std::uint32_t{1} << jointPeriod) - std::uint32_t{1};
  const auto activePhasesFor = [&](const SyncCoverGuard &guard) {
    std::uint32_t active = allPhases;
    for (const SyncCoverGuardLiteral &literal : guard.literals) {
      const std::optional<SyncCoverControlPhaseRelation> &optionalRelation =
          controls[literal.control].phaseRelation;
      if (!optionalRelation || optionalRelation->loopScope != loopScope) {
        continue;
      }
      const SyncCoverControlPhaseRelation &relation = *optionalRelation;
      std::size_t state = relation.initialPhase;
      for (unsigned iteration = 0; iteration < jointPeriod; ++iteration) {
        if (relation.activeAlternative[state] != literal.alternative) {
          active &= ~(std::uint32_t{1} << iteration);
        }
        state = relation.nextPhase[state];
      }
    }
    return static_cast<std::uint16_t>(active);
  };
  result.sourceActivePhases = activePhasesFor(sourceGuard);
  result.targetActivePhases = activePhasesFor(targetGuard);
  if (result.sourceActivePhases == 0 || result.targetActivePhases == 0) {
    result.staticallyReachable = false;
  }
  retainInCache();
  return result;
}

std::vector<std::size_t> ProgramBuilder::getReachableRecurrenceSourcePhases(
    const RecurrencePhaseOrbit &orbit, unsigned distance) const {
  std::vector<std::size_t> result;
  if (!orbit.staticallyReachable || orbit.period == 0) {
    return result;
  }
  const std::size_t period = orbit.period;
  const std::size_t offset = static_cast<std::size_t>(distance) % period;
  for (std::size_t sourcePhase = 0; sourcePhase < period; ++sourcePhase) {
    const std::size_t targetPhase = (sourcePhase + offset) % period;
    const bool sourceMatches =
        (orbit.sourceActivePhases & (std::uint16_t{1} << sourcePhase)) != 0;
    const bool targetMatches =
        (orbit.targetActivePhases & (std::uint16_t{1} << targetPhase)) != 0;
    if (sourceMatches && targetMatches) {
      result.push_back(sourcePhase);
    }
  }
  return result;
}

FailureOr<unsigned>
ProgramBuilder::maximumRecurrenceDistance(SyncCoverNodeId source,
                                          SyncCoverNodeId target,
                                          const RecurrencePhaseOrbit &orbit) {
  const bool invalidHorizon =
      orbit.period == 0 || orbit.period > std::numeric_limits<unsigned>::max();
  if (invalidHorizon) {
    function_.emitError("canonical sync recurrence phase horizon overflow");
    return failure();
  }
  const unsigned phasePeriod = static_cast<unsigned>(orbit.period);
  unsigned maximum = phasePeriod;
  std::size_t accessWork = nodeAccessIndices_[source].size();
  const bool accessWorkUnavailable =
      !checkedAddSize(accessWork, nodeAccessIndices_[target].size()) ||
      !checkedAddSize(accessWork, 1) || !consumePairInspections(accessWork);
  if (accessWorkUnavailable) {
    function_.emitError("canonical sync pair-inspection limit exceeded");
    return failure();
  }
  const auto admitAccessPeriods = [&](SyncCoverNodeId node) {
    for (std::size_t accessIndex : nodeAccessIndices_[node]) {
      const std::size_t slots =
          extractedAccesses_[accessIndex].graphAccesses.size();
      if (slots == 0 || slots > kMaximumSlotCount) {
        continue;
      }
      const std::optional<unsigned> period =
          checkedLeastCommonMultiple(phasePeriod, static_cast<unsigned>(slots));
      if (!period) {
        return false;
      }
      maximum = std::max(maximum, *period);
    }
    return true;
  };
  const bool invalidAccessPeriod =
      !admitAccessPeriods(source) || !admitAccessPeriods(target);
  if (invalidAccessPeriod) {
    function_.emitError("canonical sync recurrence horizon overflow");
    return failure();
  }
  return maximum;
}

LogicalResult ProgramBuilder::addMemoryHazards(
    SyncCoverNodeId source, SyncCoverNodeId target, Operation *loop,
    unsigned distance, SyncCoverScopeId recurrenceScope,
    HazardWitnesses *covered,
    const std::vector<std::size_t> *reachableSourcePhases,
    std::size_t phasePeriod) {
  if (reachableSourcePhases) {
    if (phasePeriod == 0 ||
        phasePeriod > kCanonicalSyncMaximumPeriodicRecurrenceStates) {
      return function_.emitError(
          "canonical sync recurrence phase mask is invalid");
    }
    for (std::size_t phase : *reachableSourcePhases) {
      if (phase >= phasePeriod) {
        return function_.emitError(
            "canonical sync recurrence source phase is invalid");
      }
    }
  }
  const auto filterHazard =
      [&](SyncCoverDemandKind kind,
          std::vector<HazardWitnessPhaseCandidate> candidates,
          std::vector<SyncCoverStorageWitnessId> &witnesses) -> LogicalResult {
    std::size_t witnessUnits = candidates.size();
    std::size_t sortWork = 0;
    const bool sortWorkUnavailable =
        !checkedAddSize(witnessUnits, 1) ||
        !checkedMultiplySize(candidates.size(),
                             logarithmicWorkBound(witnessUnits), sortWork) ||
        !checkedAddSize(sortWork, candidates.size()) ||
        !consumePairInspections(sortWork);
    if (sortWorkUnavailable) {
      return function_.emitError(
          "canonical sync pair-inspection limit exceeded");
    }
    llvm::sort(candidates, [](const HazardWitnessPhaseCandidate &lhs,
                              const HazardWitnessPhaseCandidate &rhs) {
      return lhs.witness < rhs.witness;
    });
    std::vector<HazardWitnessPhaseCandidate> merged;
    merged.reserve(candidates.size());
    for (const HazardWitnessPhaseCandidate &candidate : candidates) {
      const bool newWitness =
          merged.empty() || merged.back().witness != candidate.witness;
      if (newWitness) {
        merged.push_back(candidate);
      } else {
        merged.back().sourcePhases |= candidate.sourcePhases;
      }
    }
    witnesses.reserve(merged.size());
    for (const HazardWitnessPhaseCandidate &candidate : merged) {
      if (candidate.sourcePhases == 0) {
        continue;
      }
      if (covered) {
        std::size_t stateUnits = covered->states.size();
        if (!checkedAddSize(stateUnits, 1)) {
          return function_.emitError(
              "canonical sync recurrence witness-state work overflow");
        }
        const std::size_t lookupWork = logarithmicWorkBound(stateUnits);
        if (!consumePairInspections(lookupWork)) {
          return function_.emitError(
              "canonical sync pair-inspection limit exceeded");
        }
        auto position = std::lower_bound(
            covered->states.begin(), covered->states.end(), candidate.witness,
            [](const HazardWitnessPhaseState &state,
               SyncCoverStorageWitnessId candidate) {
              return state.witness < candidate;
            });
        const bool newWitness = position == covered->states.end() ||
                                position->witness != candidate.witness;
        if (newWitness) {
          const bool stateLimitReached =
              covered->states.size() == options_.maximumRecurrenceWitnessStates;
          if (stateLimitReached) {
            return function_.emitError(
                "canonical sync recurrence witness-state limit exceeded");
          }
          const std::size_t positionIndex =
              static_cast<std::size_t>(position - covered->states.begin());
          const std::size_t shiftWork =
              static_cast<std::size_t>(covered->states.end() - position);
          std::size_t insertionWork = shiftWork;
          const bool needsAllocation =
              covered->states.size() == covered->states.capacity();
          if (needsAllocation &&
              !checkedAddSize(insertionWork, covered->states.size())) {
            return function_.emitError(
                "canonical sync recurrence witness-state work overflow");
          }
          const bool insertionWorkUnavailable =
              !checkedAddSize(insertionWork, 1) ||
              !consumePairInspections(insertionWork);
          if (insertionWorkUnavailable) {
            return function_.emitError(
                "canonical sync pair-inspection limit exceeded");
          }
          if (needsAllocation) {
            const std::size_t capacity = covered->states.capacity();
            const std::size_t limit = options_.maximumRecurrenceWitnessStates;
            std::size_t nextCapacity = std::min<std::size_t>(64, limit);
            if (capacity != 0) {
              nextCapacity = capacity <= limit / 2 ? capacity * 2 : limit;
            }
            covered->states.reserve(nextCapacity);
          }
          position = covered->states.insert(
              covered->states.begin() + positionIndex,
              HazardWitnessPhaseState{candidate.witness, 0, 0, 0, 0});
        }
        std::uint16_t *seenPhases = nullptr;
        switch (kind) {
        case SyncCoverDemandKind::MemoryRAW:
          seenPhases = &position->raw;
          break;
        case SyncCoverDemandKind::MemoryWAR:
          seenPhases = &position->war;
          break;
        case SyncCoverDemandKind::MemoryWAW:
          seenPhases = &position->waw;
          break;
        case SyncCoverDemandKind::HardwareAccRAR:
          seenPhases = &position->accRar;
          break;
        case SyncCoverDemandKind::SSA:
          return function_.emitError(
              "canonical sync recurrence hazard kind is invalid");
        }
        const std::uint16_t newPhases =
            candidate.sourcePhases & static_cast<std::uint16_t>(~*seenPhases);
        if (newPhases == 0) {
          continue;
        }
        *seenPhases |= candidate.sourcePhases;
      }
      witnesses.push_back(candidate.witness);
    }
    return success();
  };

  std::vector<HazardWitnessPhaseCandidate> rawCandidates;
  std::vector<HazardWitnessPhaseCandidate> warCandidates;
  std::vector<HazardWitnessPhaseCandidate> wawCandidates;
  std::vector<HazardWitnessPhaseCandidate> accRarCandidates;
  const auto appendCandidate =
      [&](std::vector<HazardWitnessPhaseCandidate> &candidates,
          SyncCoverStorageWitnessId witness,
          std::uint16_t sourcePhases) -> LogicalResult {
    const bool stateLimitReached =
        covered && candidates.size() == options_.maximumRecurrenceWitnessStates;
    if (stateLimitReached) {
      return function_.emitError(
          "canonical sync recurrence witness-state limit exceeded");
    }
    candidates.push_back({witness, sourcePhases});
    return success();
  };
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
      FailureOr<std::vector<OrdinalPairPhaseState>> ordinalPairs =
          getOrdinalPairs(first, second, loop, distance, reachableSourcePhases,
                          phasePeriod);
      if (failed(ordinalPairs)) {
        return failure();
      }
      for (const OrdinalPairPhaseState &ordinalPair : *ordinalPairs) {
        if (!consumePairInspection()) {
          function_.emitError("canonical sync pair-inspection limit exceeded");
          return failure();
        }
        const bool invalidOrdinal =
            ordinalPair.first >= first.graphAccesses.size() ||
            ordinalPair.second >= second.graphAccesses.size();
        if (invalidOrdinal) {
          continue;
        }
        const SyncCoverStorageAccess &firstAccess =
            graph_.getStorageAccesses()[first.graphAccesses[ordinalPair.first]];
        const SyncCoverStorageAccess &secondAccess =
            graph_
                .getStorageAccesses()[second.graphAccesses[ordinalPair.second]];
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
        const bool warHazard = syncCoverStorageModeReads(firstAccess.mode) &&
                               syncCoverStorageModeWrites(secondAccess.mode);
        const bool wawHazard = syncCoverStorageModeWrites(firstAccess.mode) &&
                               syncCoverStorageModeWrites(secondAccess.mode);
        const bool accumulatorReadReadHazard =
            targetCapabilities_.crossPipeAccumulatorReadReadHazard
                .isEnabled() &&
            graph_.getStorageDomains()[firstAccess.domain].role ==
                SyncCoverStorageDomainRole::Accumulator &&
            graph_.getNodes()[source].resource !=
                graph_.getNodes()[target].resource &&
            syncCoverStorageModeReads(firstAccess.mode) &&
            syncCoverStorageModeReads(secondAccess.mode);
        const bool appendFailed =
            (rawHazard && failed(appendCandidate(rawCandidates, *witness.index,
                                                 ordinalPair.sourcePhases))) ||
            (warHazard && failed(appendCandidate(warCandidates, *witness.index,
                                                 ordinalPair.sourcePhases))) ||
            (wawHazard && failed(appendCandidate(wawCandidates, *witness.index,
                                                 ordinalPair.sourcePhases))) ||
            (accumulatorReadReadHazard &&
             failed(appendCandidate(accRarCandidates, *witness.index,
                                    ordinalPair.sourcePhases)));
        if (appendFailed) {
          return failure();
        }
      }
    }
  }

  std::vector<SyncCoverStorageWitnessId> raw;
  std::vector<SyncCoverStorageWitnessId> war;
  std::vector<SyncCoverStorageWitnessId> waw;
  std::vector<SyncCoverStorageWitnessId> accRar;
  const bool hazardFilteringFailed =
      failed(filterHazard(SyncCoverDemandKind::MemoryRAW,
                          std::move(rawCandidates), raw)) ||
      failed(filterHazard(SyncCoverDemandKind::MemoryWAR,
                          std::move(warCandidates), war)) ||
      failed(filterHazard(SyncCoverDemandKind::MemoryWAW,
                          std::move(wawCandidates), waw)) ||
      failed(filterHazard(SyncCoverDemandKind::HardwareAccRAR,
                          std::move(accRarCandidates), accRar));
  if (hazardFilteringFailed) {
    return failure();
  }
  std::vector<SyncCoverDemandKind> kinds;
  std::vector<SyncCoverStorageWitnessId> witnesses;
  std::size_t witnessCount = raw.size();
  const bool witnessCountUnavailable =
      !checkedAddSize(witnessCount, war.size()) ||
      !checkedAddSize(witnessCount, waw.size()) ||
      !checkedAddSize(witnessCount, accRar.size());
  if (witnessCountUnavailable) {
    return function_.emitError("canonical sync demand witness-count overflow");
  }
  witnesses.reserve(witnessCount);
  const auto appendKind =
      [&](SyncCoverDemandKind kind,
          const std::vector<SyncCoverStorageWitnessId> &kindWitnesses) {
        if (kindWitnesses.empty()) {
          return;
        }
        kinds.push_back(kind);
        witnesses.insert(witnesses.end(), kindWitnesses.begin(),
                         kindWitnesses.end());
      };
  appendKind(SyncCoverDemandKind::MemoryRAW, raw);
  appendKind(SyncCoverDemandKind::MemoryWAR, war);
  appendKind(SyncCoverDemandKind::MemoryWAW, waw);
  appendKind(SyncCoverDemandKind::HardwareAccRAR, accRar);
  if (kinds.empty()) {
    return success();
  }
  std::size_t witnessUnits = witnesses.size();
  std::size_t perWitnessWork = 0;
  std::size_t validationWork = 0;
  const bool validationWorkUnavailable =
      !checkedAddSize(witnessUnits, 1) ||
      !checkedMultiplySize(logarithmicWorkBound(witnessUnits), 2,
                           perWitnessWork) ||
      !checkedAddSize(perWitnessWork, 12) ||
      !checkedMultiplySize(witnesses.size(), perWitnessWork, validationWork) ||
      !consumePairInspections(validationWork);
  if (validationWorkUnavailable) {
    return function_.emitError("canonical sync pair-inspection limit exceeded");
  }
  llvm::sort(witnesses);
  witnesses.erase(std::unique(witnesses.begin(), witnesses.end()),
                  witnesses.end());
  const std::size_t originalDemandCount = kinds.size();
  return addDemand(source, target, recurrenceScope, distance, std::move(kinds),
                   std::move(witnesses), originalDemandCount);
}

LogicalResult
ProgramBuilder::addDemand(SyncCoverNodeId source, SyncCoverNodeId target,
                          SyncCoverScopeId scope, unsigned distance,
                          std::vector<SyncCoverDemandKind> kinds,
                          std::vector<SyncCoverStorageWitnessId> witnesses,
                          std::size_t originalDemandCount) {
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
  demand.provenanceKinds = std::move(kinds);
  demand.orderingRequirements = syncCoverOrderingRequirementBit(
      SyncCoverOrderingRequirement::PipelineCompletionBeforeAccess);
  if (llvm::is_contained(demand.provenanceKinds,
                         SyncCoverDemandKind::HardwareAccRAR)) {
    demand.orderingRequirements |= syncCoverOrderingRequirementBit(
        SyncCoverOrderingRequirement::HardwareSpecialOrder);
  }
  const bool hasGlobalMemoryWitness =
      llvm::any_of(witnesses, [&](SyncCoverStorageWitnessId witnessId) {
        const SyncCoverStorageWitness &witness =
            graph_.getStorageWitnesses()[witnessId];
        const SyncCoverStorageAccess &access =
            graph_.getStorageAccesses()[witness.sourceAccess];
        return graph_.getStorageDomains()[access.domain].addressSpace ==
               static_cast<std::uint32_t>(AddressSpace::GM);
      });
  if (hasGlobalMemoryWitness) {
    demand.orderingRequirements |= syncCoverOrderingRequirementBit(
        SyncCoverOrderingRequirement::MemoryOrderBeforeAccess);
  }
  const bool hasScalarDCacheWitness =
      llvm::any_of(witnesses, [&](SyncCoverStorageWitnessId witnessId) {
        const SyncCoverStorageWitness &witness =
            graph_.getStorageWitnesses()[witnessId];
        const SyncCoverStorageAccess &sourceAccess =
            graph_.getStorageAccesses()[witness.sourceAccess];
        const SyncCoverStorageAccess &targetAccess =
            graph_.getStorageAccesses()[witness.targetAccess];
        return sourceAccess.path == SyncCoverStorageAccessPath::ScalarDCache ||
               targetAccess.path == SyncCoverStorageAccessPath::ScalarDCache;
      });
  if (hasScalarDCacheWitness) {
    demand.orderingRequirements |= syncCoverOrderingRequirementBit(
        SyncCoverOrderingRequirement::CacheVisibilityBeforeAccess);
  }
  demand.storageWitnesses = std::move(witnesses);
  demand.originalDemandCount = originalDemandCount;
  const SyncCoverGraphResult added = graph_.addDemand(std::move(demand));
  if (!added) {
    return function_.emitError("cannot construct canonical sync demand")
           << " (error " << static_cast<unsigned>(added.error) << ')';
  }
  return success();
}
