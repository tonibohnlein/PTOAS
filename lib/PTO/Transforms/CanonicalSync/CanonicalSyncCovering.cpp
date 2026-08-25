// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// This file is the MLIR-to-SyncCover trust boundary. The shadow adapter first
// translates only immutable structure and demands. Mechanism translation and
// emission remain separate rollout stages.

#include "CanonicalSyncInternal.h"

#include "PTO/Transforms/CanonicalSync/SyncCoverCandidateIndex.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverSlotLifecycle.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverSlotProtocol.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct RegionContext {
  SyncCoverScopeId scope = 0;
  SyncCoverGuard guard;
};

struct DependencyKey {
  std::size_t source = 0;
  std::size_t target = 0;
  CanonicalDependencyKind kind = CanonicalDependencyKind::SSA;
  unsigned distance = 0;
  Operation *loop = nullptr;

  bool operator<(const DependencyKey &other) const {
    const auto key = std::tie(source, target, kind, distance);
    const auto otherKey =
        std::tie(other.source, other.target, other.kind, other.distance);
    if (key != otherKey) {
      return key < otherKey;
    }
    return std::less<Operation *>{}(loop, other.loop);
  }
};

DependencyKey getDependencyKey(const CanonicalDependency &dependency) {
  return {dependency.source, dependency.target, dependency.kind,
          dependency.iterationDistance, dependency.recurrenceLoop};
}

SyncCoverDemandKind getDemandKind(CanonicalDependencyKind kind) {
  switch (kind) {
  case CanonicalDependencyKind::SSA:
  case CanonicalDependencyKind::LoopCarriedSSA:
    return SyncCoverDemandKind::SSA;
  case CanonicalDependencyKind::MemoryRAW:
    return SyncCoverDemandKind::MemoryRAW;
  case CanonicalDependencyKind::MemoryWAR:
    return SyncCoverDemandKind::MemoryWAR;
  case CanonicalDependencyKind::MemoryWAW:
    return SyncCoverDemandKind::MemoryWAW;
  }
  return SyncCoverDemandKind::SSA;
}

SyncCoverEdgeKind getFixedEdgeKind(SyncGraphEdgeKind kind) {
  switch (kind) {
  case SyncGraphEdgeKind::IssueOrder:
    return SyncCoverEdgeKind::CompletionPreservingIssueOrder;
  case SyncGraphEdgeKind::NonCompletionPreservingIssueOrder:
    return SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  case SyncGraphEdgeKind::HardwareCompletion:
    return SyncCoverEdgeKind::CompletionSupply;
  }
  return SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
}

bool regionContains(const Region *region, Operation *operation) {
  Region *current = nullptr;
  if (operation) {
    current = operation->getParentRegion();
  }
  while (current) {
    if (current == region) {
      return true;
    }
    Operation *parent = current->getParentOp();
    current = parent ? parent->getParentRegion() : nullptr;
  }
  return false;
}

bool hasStaticallyPositiveTripCount(Operation *operation) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(operation);
  if (!forOp) {
    return false;
  }
  APInt lower;
  APInt upper;
  APInt step;
  return matchPattern(forOp.getLowerBound(), m_ConstantInt(&lower)) &&
         matchPattern(forOp.getUpperBound(), m_ConstantInt(&upper)) &&
         matchPattern(forOp.getStep(), m_ConstantInt(&step)) &&
         step.isStrictlyPositive() && lower.slt(upper);
}

class CanonicalSyncCoveringGraphAdapter {
public:
  CanonicalSyncCoveringGraphAdapter(
      func::FuncOp func, const CanonicalSyncPlan &plan,
      const CanonicalMechanismUniverse &legacyUniverse,
      ArrayRef<CanonicalEventBundleCandidate> selectedEventBundles,
      unsigned eventIdMax, bool registerShadowSlotProtocols,
      const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds,
      std::function<bool(PipelineType)> hasHardwareCompletion,
      std::function<bool(const CanonicalDependency &)> hasIntrinsicOrdering,
      std::function<std::size_t(const CanonicalAnchor &)> getAnchorPosition,
      std::function<std::vector<SyncGraphEdge>(const CanonicalBarrier &)>
          getBarrierCompletionEdges,
      std::function<bool(ArrayRef<CanonicalEvent>)> verifyEventProtocols)
      : func_(func), plan_(plan), legacyUniverse_(legacyUniverse),
        selectedEventBundles_(selectedEventBundles), eventIdMax_(eventIdMax),
        registerShadowSlotProtocols_(registerShadowSlotProtocols),
        reservedIds_(reservedIds),
        hasHardwareCompletion_(std::move(hasHardwareCompletion)),
        hasIntrinsicOrdering_(std::move(hasIntrinsicOrdering)),
        getAnchorPosition_(std::move(getAnchorPosition)),
        getBarrierCompletionEdges_(std::move(getBarrierCompletionEdges)),
        verifyEventProtocols_(std::move(verifyEventProtocols)) {}

  LogicalResult build(CanonicalSyncCoveringShadowSnapshot &snapshot) {
    regionContexts_[&func_.getBody()] = {};
    if (failed(buildRegionScopes(func_.getBody(), regionContexts_.at(
                                                     &func_.getBody())))) {
      return failure();
    }
    const bool graphBuilt = succeeded(addNodes()) && succeeded(addFixedEdges()) &&
                            succeeded(addRecurrenceCarryEdges()) &&
                            succeeded(addDemands());
    if (!graphBuilt) {
      return failure();
    }
    const SyncCoverGraphResult frozen = graph_.freezeStructure();
    if (!frozen) {
      return emitGraphError("structural graph freeze", frozen);
    }
    const SyncCoverCandidateIndex candidateIndex(graph_);
    if (!candidateIndex) {
      return func_.emitError()
             << "internal error: canonical covering candidate index failed "
                "with "
             << static_cast<unsigned>(candidateIndex.getError());
    }
    const SyncCoverSlotLifecycleResult lifecycleResult =
        discoverSyncCoverSlotLifecycles(graph_, candidateIndex);
    if (!lifecycleResult) {
      return func_.emitError()
             << "internal error: canonical covering slot-lifecycle discovery "
                "failed with "
             << static_cast<unsigned>(lifecycleResult.error);
    }
    snapshot.slotLifecycleDetails.clear();
    snapshot.slotLifecycleDetails.reserve(lifecycleResult.lifecycles.size());
    for (const SyncCoverSlotLifecycle &lifecycle :
         lifecycleResult.lifecycles) {
      snapshot.slotLifecycleDetails.push_back(
          {lifecycle.id, lifecycle.slot.domain, lifecycle.slot.extent,
           lifecycle.producerResource, lifecycle.consumerResource,
           lifecycle.recurrenceScope, lifecycle.distance, lifecycle.ready,
           lifecycle.release, lifecycle.managedAccesses,
           lifecycle.hasUnrepresentedAccesses,
           lifecycle.requiresPathSensitiveProof});
    }
    snapshot.slotLifecycleCandidates = lifecycleResult.lifecycles.size();
    snapshot.partialSlotOpportunities =
        lifecycleResult.partialSlotOpportunities;
    snapshot.slotLifecycleDiscoveryTruncated = lifecycleResult.truncated;
    snapshot.pathSensitiveSlotLifecycles = static_cast<std::size_t>(
        llvm::count_if(lifecycleResult.lifecycles, [](const auto &lifecycle) {
          return lifecycle.requiresPathSensitiveProof;
        }));
    const SyncCoverSlotProtocolResult protocolResult =
        buildSyncCoverSlotProtocolCandidates(graph_, candidateIndex,
                                             lifecycleResult);
    if (!protocolResult) {
      return func_.emitError()
             << "internal error: canonical covering slot-protocol generation "
                "failed with "
             << static_cast<unsigned>(protocolResult.error);
    }
    snapshot.slotProtocolCandidates = protocolResult.candidates.size();
    snapshot.pathSensitiveSlotProtocolLifecycles =
        protocolResult.pathSensitiveLifecycles;
    snapshot.accessOpenSlotProtocolLifecycles =
        protocolResult.accessOpenLifecycles;
    snapshot.unsupportedEffectSlotProtocolLifecycles =
        protocolResult.unsupportedEffectLifecycles;
    snapshot.unsupportedDistanceSlotProtocolReleases =
        protocolResult.unsupportedDistanceReleases;
    snapshot.nonBoundarySlotProtocolReleases =
        protocolResult.nonBoundaryReleases;
    snapshot.slotProtocolEvaluations = protocolResult.evaluations;
    snapshot.slotProtocolGenerationTruncated = protocolResult.truncated;
    for (const auto &entry : regionContexts_) {
      regionScopes_.emplace(entry.first, entry.second.scope);
    }
    if (failed(runCanonicalSyncCoveringShadowSelection(
        func_, plan_, legacyUniverse_, selectedEventBundles_, eventIdMax_,
            reservedIds_, graph_, candidateIndex, lifecycleResult,
            protocolResult, registerShadowSlotProtocols_, activeDemands_,
            regionScopes_, loopScopes_, getAnchorPosition_,
            getBarrierCompletionEdges_,
            verifyEventProtocols_, snapshot))) {
      return failure();
    }
    snapshot.scopes = graph_.getScopes().size();
    snapshot.controls = graph_.getControls().size();
    snapshot.nodes = graph_.getNodes().size();
    snapshot.storageDomains = graph_.getStorageDomains().size();
    snapshot.storageAccesses = graph_.getStorageAccesses().size();
    snapshot.storageWitnesses = graph_.getStorageWitnesses().size();
    snapshot.fixedEdges = plan_.getFixedEdges().size();
    snapshot.recurrenceCarryEdges = recurrenceCarryEdges_;
    snapshot.conservativeDemands = graph_.getDemands().size();
    snapshot.activeDemands = activeDemands_.size();
    snapshot.intrinsicallySatisfiedDemands = intrinsicallySatisfiedDemands_;
    snapshot.scopeDetails = graph_.getScopes();
    snapshot.controlDetails = graph_.getControls();
    snapshot.nodeDetails = graph_.getNodes();
    snapshot.edgeDetails = graph_.getEdges();
    snapshot.demandDetails = graph_.getDemands();
    snapshot.activeDemandIds = activeDemands_;
    return success();
  }

private:
  std::optional<SyncCoverTimelineInterval>
  getOperationTimeline(Operation *operation) const {
    return getTimeline([&](const CanonicalSyncNode &node) {
      return operation == node.operation || operation->isAncestor(node.operation);
    });
  }

  std::optional<SyncCoverTimelineInterval>
  getRegionTimeline(const Region *region) const {
    return getTimeline([&](const CanonicalSyncNode &node) {
      return regionContains(region, node.operation);
    });
  }

  template <typename Predicate>
  std::optional<SyncCoverTimelineInterval>
  getTimeline(Predicate predicate) const {
    std::size_t begin = std::numeric_limits<std::size_t>::max();
    std::size_t end = 0;
    bool found = false;
    for (const CanonicalSyncNode &node : plan_.getNodes()) {
      const bool selected = predicate(node);
      const bool orderFits =
          node.order <= (std::numeric_limits<std::size_t>::max() - 1) / 2;
      if (!selected || !orderFits) {
        continue;
      }
      found = true;
      begin = std::min(begin, node.order * 2);
      end = std::max(end, node.order * 2 + 1);
    }
    if (!found) {
      return std::nullopt;
    }
    return SyncCoverTimelineInterval{begin, end};
  }

  FailureOr<SyncCoverScopeId>
  addScope(SyncCoverScopeId parent, bool mustExecute,
           std::optional<SyncCoverTimelineInterval> timeline, bool isLoop,
           StringRef context) {
    const SyncCoverGraphResult result =
        graph_.addScope(parent, mustExecute, timeline, isLoop);
    if (!result || !result.index) {
      (void)emitGraphError(context, result);
      return failure();
    }
    return *result.index;
  }

  LogicalResult buildRegionScopes(Region &region,
                                  const RegionContext &context) {
    for (Block &block : region) {
      for (Operation &operation : block) {
        const unsigned regionCount = operation.getNumRegions();
        if (regionCount == 0) {
          continue;
        }
        const bool isLoop = isa<LoopLikeOpInterface>(&operation);
        const bool containerMustExecute =
            !isLoop || hasStaticallyPositiveTripCount(&operation);
        FailureOr<SyncCoverScopeId> containerScope = addScope(
            context.scope, containerMustExecute,
            getOperationTimeline(&operation), isLoop,
            "structured-container scope insertion");
        if (failed(containerScope)) {
          return failure();
        }
        if (isLoop) {
          loopScopes_[&operation] = *containerScope;
        }

        std::optional<SyncCoverControlId> control;
        if (isa<scf::IfOp>(&operation)) {
          const SyncCoverGraphResult result = graph_.addControl(
              operation.getNumRegions(), *containerScope);
          if (!result || !result.index) {
            return emitGraphError("structured-control insertion", result);
          }
          control = *result.index;
        }

        for (auto indexedRegion : llvm::enumerate(operation.getRegions())) {
          Region &child = indexedRegion.value();
          bool regionMustExecute = true;
          if (isa<scf::IfOp>(&operation)) {
            regionMustExecute = false;
          } else if (isa<scf::WhileOp>(&operation) &&
                     indexedRegion.index() != 0) {
            regionMustExecute = false;
          }
          FailureOr<SyncCoverScopeId> childScope = addScope(
              *containerScope, regionMustExecute, getRegionTimeline(&child),
              false, "structured-region scope insertion");
          if (failed(childScope)) {
            return failure();
          }
          RegionContext childContext{*childScope, context.guard};
          if (control) {
            const std::size_t alternative = indexedRegion.index();
            const bool alternativeFits =
                alternative <= std::numeric_limits<unsigned>::max();
            if (!alternativeFits) {
              return operation.emitError(
                  "canonical covering control alternative is out of range");
            }
            childContext.guard.literals.push_back(
                {*control, static_cast<unsigned>(alternative)});
          }
          regionContexts_[&child] = childContext;
          if (failed(buildRegionScopes(child, childContext))) {
            return failure();
          }
        }
      }
    }
    return success();
  }

  LogicalResult addNodes() {
    std::map<std::size_t, std::set<std::uint32_t>> completionTargets;
    const auto collectCompletionTargets = [&](const auto &bundles) {
      for (const CanonicalEventBundleCandidate &bundle : bundles) {
        for (const CanonicalEvent &event : bundle.events) {
          for (const CanonicalEventCompletion &completion :
               event.completions) {
            completionTargets[completion.source].insert(
                static_cast<std::uint32_t>(event.targetPipe));
          }
        }
      }
    };
    collectCompletionTargets(legacyUniverse_.eventBundles);
    collectCompletionTargets(selectedEventBundles_);
    for (const CanonicalSyncNode &node : plan_.getNodes()) {
      auto context = regionContexts_.find(node.operation->getParentRegion());
      const bool wrongIdentity = node.id != graph_.getNodes().size();
      const bool missingContext = context == regionContexts_.end();
      if (wrongIdentity || missingContext) {
        return func_.emitError(
            "internal error: canonical covering node scope is unavailable");
      }
      std::vector<std::uint32_t> targets;
      auto completion = completionTargets.find(node.id);
      if (completion != completionTargets.end()) {
        targets.assign(completion->second.begin(), completion->second.end());
      }
      const SyncCoverGraphResult result = graph_.addNode(
          static_cast<std::uint32_t>(node.pipe), 1, context->second.scope,
          node.order, context->second.guard, std::move(targets));
      if (!result || !result.index || *result.index != node.id) {
        return emitGraphError("node insertion", result);
      }
    }
    return success();
  }

  FailureOr<SyncCoverScopeId> getEndpointScope(std::size_t source,
                                               std::size_t target) const {
    const std::size_t nodeCount = graph_.getNodes().size();
    if (source >= nodeCount || target >= nodeCount) {
      return failure();
    }
    const std::optional<SyncCoverScopeId> scope = graph_.getLowestCommonScope(
        graph_.getNodes()[source].scope, graph_.getNodes()[target].scope);
    if (!scope) {
      return failure();
    }
    return *scope;
  }

  LogicalResult addFixedEdges() {
    for (const SyncGraphEdge &fixed : plan_.getFixedEdges()) {
      FailureOr<SyncCoverScopeId> scope =
          getEndpointScope(fixed.source, fixed.target);
      if (failed(scope)) {
        return func_.emitError(
            "internal error: canonical covering fixed-edge scope is invalid");
      }
      SyncCoverEdge edge;
      edge.source = fixed.source;
      edge.target = fixed.target;
      edge.kind = getFixedEdgeKind(fixed.kind);
      edge.scope = *scope;
      const SyncCoverGraphResult result = graph_.addEdge(std::move(edge));
      if (!result) {
        return emitGraphError("fixed-edge insertion", result);
      }
    }
    return success();
  }

  LogicalResult addRecurrenceCarryEdges() {
    llvm::SmallPtrSet<Operation *, 8> requiredLoops;
    for (const CanonicalDependency &demand :
         plan_.getConservativeCompletionRequirements()) {
      if (demand.iterationDistance != 0 && demand.recurrenceLoop) {
        requiredLoops.insert(demand.recurrenceLoop);
      }
    }
    SmallVector<Operation *, 8> recurrenceLoops;
    func_.walk([&](Operation *operation) {
      if (requiredLoops.contains(operation)) {
        recurrenceLoops.push_back(operation);
      }
    });
    for (Operation *loop : recurrenceLoops) {
      auto scope = loopScopes_.find(loop);
      if (scope == loopScopes_.end()) {
        return loop->emitError(
            "internal error: canonical covering recurrence scope is missing");
      }
      std::map<PipelineType, SmallVector<const CanonicalSyncNode *, 32>>
          loopNodesByPipe;
      for (const CanonicalSyncNode &node : plan_.getNodes()) {
        if (loop->isAncestor(node.operation)) {
          loopNodesByPipe[node.pipe].push_back(&node);
        }
      }
      for (const CanonicalSyncNode &source : plan_.getNodes()) {
        if (!loop->isAncestor(source.operation)) {
          continue;
        }
        for (const CanonicalSyncNode *target : loopNodesByPipe[source.pipe]) {
          SyncCoverEdge edge;
          edge.source = source.id;
          edge.target = target->id;
          edge.kind = hasHardwareCompletion_(source.pipe)
                          ? SyncCoverEdgeKind::CompletionSupply
                          : SyncCoverEdgeKind::CompletionPreservingIssueOrder;
          edge.scope = scope->second;
          edge.distance = 1;
          const SyncCoverGraphResult result = graph_.addEdge(std::move(edge));
          if (!result) {
            return emitGraphError("recurrence carry insertion", result);
          }
          ++recurrenceCarryEdges_;
        }
      }
    }
    return success();
  }

  LogicalResult addDemands() {
    std::map<DependencyKey, SyncCoverDemandId> demandIds;
    for (const CanonicalDependency &requirement :
         plan_.getConservativeCompletionRequirements()) {
      SyncCoverDemand demand;
      demand.source = requirement.source;
      demand.target = requirement.target;
      demand.kind = getDemandKind(requirement.kind);
      demand.distance = requirement.iterationDistance;
      if (requirement.iterationDistance != 0) {
        auto scope = loopScopes_.find(requirement.recurrenceLoop);
        if (scope == loopScopes_.end()) {
          return func_.emitError(
              "internal error: canonical covering demand loop is missing");
        }
        demand.scope = scope->second;
      } else {
        FailureOr<SyncCoverScopeId> scope =
            getEndpointScope(requirement.source, requirement.target);
        if (failed(scope)) {
          return func_.emitError(
              "internal error: canonical covering demand scope is invalid");
        }
        demand.scope = *scope;
      }
      if (failed(addStorageProvenance(requirement, demand))) {
        return failure();
      }
      const SyncCoverGraphResult result = graph_.addDemand(std::move(demand));
      if (!result || !result.index) {
        return emitGraphError("demand insertion", result);
      }
      const auto inserted =
          demandIds.emplace(getDependencyKey(requirement), *result.index);
      if (!inserted.second) {
        return func_.emitError(
            "internal error: duplicate canonical covering demand identity");
      }
    }
    // This list is captured after alias-contract filtering and before the
    // legacy completion-reduction phase mutates the working dependencies.
    for (const CanonicalDependency &requirement :
         plan_.getCompletionRequirements()) {
      if (hasIntrinsicOrdering_(requirement)) {
        ++intrinsicallySatisfiedDemands_;
        continue;
      }
      auto demand = demandIds.find(getDependencyKey(requirement));
      if (demand == demandIds.end()) {
        return func_.emitError(
            "internal error: active canonical covering demand is absent from "
            "the conservative universe");
      }
      activeDemands_.push_back(demand->second);
    }
    llvm::sort(activeDemands_);
    const auto duplicate =
        std::adjacent_find(activeDemands_.begin(), activeDemands_.end());
    if (duplicate != activeDemands_.end()) {
      return func_.emitError(
          "internal error: duplicate active canonical covering demand");
    }
    return success();
  }

  FailureOr<SyncCoverStorageDomainId>
  getStorageDomain(AddressSpace space) {
    auto existing = storageDomains_.find(space);
    if (existing != storageDomains_.end()) {
      return existing->second;
    }
    const SyncCoverGraphResult result = graph_.addStorageDomain();
    if (!result || !result.index) {
      static_cast<void>(emitGraphError("storage-domain insertion", result));
      return failure();
    }
    storageDomains_.emplace(space, *result.index);
    return *result.index;
  }

  FailureOr<SyncCoverStorageAccessId>
  addStorageAccess(const CanonicalSyncNode &node, std::size_t accessIndex,
                   unsigned addressOrdinal) {
    if (accessIndex >= node.accesses.size()) {
      return failure();
    }
    const CanonicalMemoryAccess &access = node.accesses[accessIndex];
    const bool invalidOrdinal =
        addressOrdinal >= access.addresses.size() || access.size == 0;
    if (invalidOrdinal) {
      return failure();
    }
    const std::uint64_t begin = access.addresses[addressOrdinal];
    const bool addressOverflow =
        access.size > std::numeric_limits<std::uint64_t>::max() - begin;
    if (addressOverflow) {
      return failure();
    }
    const std::tuple<std::size_t, std::size_t, unsigned> accessKey{
        node.id, accessIndex, addressOrdinal};
    auto knownAccess = storageAccessIds_.find(accessKey);
    if (knownAccess != storageAccessIds_.end()) {
      return knownAccess->second;
    }
    FailureOr<SyncCoverStorageDomainId> domain = getStorageDomain(access.space);
    if (failed(domain)) {
      return failure();
    }
    const std::pair<std::size_t, std::size_t> familyKey{node.id, accessIndex};
    auto family = storageFamilies_.find(familyKey);
    if (family == storageFamilies_.end()) {
      family = storageFamilies_
                   .emplace(familyKey,
                            static_cast<SyncCoverStorageAccessFamilyId>(
                                storageFamilies_.size()))
                   .first;
    }
    const SyncCoverStorageAccessMode mode =
        access.reads && access.writes
            ? SyncCoverStorageAccessMode::ReadWrite
            : access.writes ? SyncCoverStorageAccessMode::Write
                            : SyncCoverStorageAccessMode::Read;
    const SyncCoverGraphResult result = graph_.addStorageAccess(
        node.id, *domain, family->second, {begin, begin + access.size}, mode,
        addressOrdinal);
    if (!result || !result.index) {
      static_cast<void>(emitGraphError("storage-access insertion", result));
      return failure();
    }
    storageAccessIds_.emplace(accessKey, *result.index);
    return *result.index;
  }

  LogicalResult addStorageProvenance(const CanonicalDependency &requirement,
                                     SyncCoverDemand &demand) {
    if (requirement.kind == CanonicalDependencyKind::SSA ||
        requirement.kind == CanonicalDependencyKind::LoopCarriedSSA) {
      demand.storageProvenance = SyncCoverStorageProvenance::NotApplicable;
      return success();
    }
    demand.storageProvenance =
        requirement.storageProvenance == CanonicalStorageProvenance::Complete
            ? SyncCoverStorageProvenance::Complete
            : SyncCoverStorageProvenance::Incomplete;
    const bool invalidRequirement =
        requirement.source >= plan_.getNodes().size() ||
        requirement.target >= plan_.getNodes().size();
    if (invalidRequirement) {
      return failure();
    }
    const CanonicalSyncNode &source = plan_.getNodes()[requirement.source];
    const CanonicalSyncNode &target = plan_.getNodes()[requirement.target];
    for (const CanonicalMemoryHazardWitness &witness :
         requirement.storageWitnesses) {
      FailureOr<SyncCoverStorageAccessId> sourceAccess = addStorageAccess(
          source, witness.sourceAccess, witness.sourceAddressOrdinal);
      FailureOr<SyncCoverStorageAccessId> targetAccess = addStorageAccess(
          target, witness.targetAccess, witness.targetAddressOrdinal);
      const bool invalidAccess = failed(sourceAccess) || failed(targetAccess);
      if (invalidAccess) {
        return func_.emitError(
            "internal error: canonical covering storage witness access is "
            "invalid");
      }
      const std::pair<SyncCoverStorageAccessId, SyncCoverStorageAccessId>
          witnessKey{*sourceAccess, *targetAccess};
      auto knownWitness = storageWitnessIds_.find(witnessKey);
      SyncCoverGraphResult result;
      if (knownWitness == storageWitnessIds_.end()) {
        result = graph_.addStorageWitness(*sourceAccess, *targetAccess);
        if (result && result.index) {
          storageWitnessIds_.emplace(witnessKey, *result.index);
        }
      } else {
        result = {SyncCoverGraphError::None, knownWitness->second};
      }
      if (!result || !result.index) {
        return emitGraphError("storage-witness insertion", result);
      }
      const SyncCoverStorageWitness &stored =
          graph_.getStorageWitnesses()[*result.index];
      if (stored.overlap.begin != witness.overlapBegin ||
          stored.overlap.end != witness.overlapEnd) {
        return func_.emitError(
            "internal error: canonical covering storage overlap changed "
            "during translation");
      }
      demand.storageWitnesses.push_back(*result.index);
    }
    return success();
  }

  LogicalResult emitGraphError(StringRef context,
                               const SyncCoverGraphResult &result) {
    InFlightDiagnostic diagnostic = func_.emitError()
                                    << "canonical covering " << context
                                    << " failed with graph error "
                                    << static_cast<unsigned>(result.error);
    if (result.index) {
      diagnostic << " at index " << *result.index;
    }
    return failure();
  }

  func::FuncOp func_;
  const CanonicalSyncPlan &plan_;
  const CanonicalMechanismUniverse &legacyUniverse_;
  std::vector<CanonicalEventBundleCandidate> selectedEventBundles_;
  unsigned eventIdMax_ = 0;
  bool registerShadowSlotProtocols_ = false;
  const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds_;
  std::function<bool(PipelineType)> hasHardwareCompletion_;
  std::function<bool(const CanonicalDependency &)> hasIntrinsicOrdering_;
  std::function<std::size_t(const CanonicalAnchor &)> getAnchorPosition_;
  std::function<std::vector<SyncGraphEdge>(const CanonicalBarrier &)>
      getBarrierCompletionEdges_;
  std::function<bool(ArrayRef<CanonicalEvent>)> verifyEventProtocols_;
  SyncCoverGraph graph_;
  std::map<Region *, RegionContext, std::less<Region *>> regionContexts_;
  std::map<Region *, SyncCoverScopeId, std::less<Region *>> regionScopes_;
  DenseMap<Operation *, SyncCoverScopeId> loopScopes_;
  std::map<AddressSpace, SyncCoverStorageDomainId> storageDomains_;
  std::map<std::pair<std::size_t, std::size_t>,
           SyncCoverStorageAccessFamilyId>
      storageFamilies_;
  std::map<std::tuple<std::size_t, std::size_t, unsigned>,
           SyncCoverStorageAccessId>
      storageAccessIds_;
  std::map<std::pair<SyncCoverStorageAccessId, SyncCoverStorageAccessId>,
           SyncCoverStorageWitnessId>
      storageWitnessIds_;
  std::vector<SyncCoverDemandId> activeDemands_;
  std::size_t recurrenceCarryEdges_ = 0;
  std::size_t intrinsicallySatisfiedDemands_ = 0;
};

} // namespace

LogicalResult CanonicalSyncPlanBuilder::buildCoveringShadowGraph() {
  CanonicalSyncCoveringShadowSnapshot snapshot;
  CanonicalSyncCoveringGraphAdapter adapter(
      func_, plan_, mechanismUniverse_, selectedEventBundles_, eventIdMax_,
      !coveringEmissionEnabled_, reservedIds_,
      [&](PipelineType pipe) { return hasHardwareCompletion(pipe); },
      [&](const CanonicalDependency &dependency) {
        return hasIntrinsicMmadAccumulatorOrdering(dependency);
      },
      [&](const CanonicalAnchor &anchor) { return getAnchorPosition(anchor); },
      [&](const CanonicalBarrier &barrier) {
        return buildBarrierCompletionEdges({barrier});
      },
      [&](ArrayRef<CanonicalEvent> events) {
        return succeeded(verifyEventProtocols(events,
                                              /*requireAllocation=*/false,
                                              /*diagnose=*/false));
      });
  if (failed(adapter.build(snapshot))) {
    return failure();
  }
  plan_.coveringShadowSnapshot_ = std::move(snapshot);
  return success();
}
