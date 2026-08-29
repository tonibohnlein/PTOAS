// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include "SyncCoverGraphInternal.h"

#include <algorithm>
#include <set>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_detail;

SyncCoverGraphResult SyncCoverGraph::validate() const {
  for (SyncCoverGraphResult result :
       {validateScopesControlsAndNodes(), validateDemands(), validateEdges(),
        validateStorage(), validateTargetCompletionCertificates()}) {
    if (!result) {
      return result;
    }
  }
  for (const auto &[key, sources] : blockingTargetedBarrierPrefixes_) {
    const auto &[resource, physicalTarget] = key;
    const bool invalidTarget =
        physicalTarget >= nodes_.size() ||
        (physicalTarget < nodes_.size() &&
         nodes_[physicalTarget].physicalAnchor != physicalTarget) ||
        !supportsBlockingTargetedBarrier(resource);
    if (invalidTarget) {
      return {SyncCoverGraphError::InvalidCompletionTargets, physicalTarget};
    }
    const bool invalidSources =
        !std::is_sorted(sources.begin(), sources.end()) ||
        std::adjacent_find(sources.begin(), sources.end()) != sources.end() ||
        std::any_of(sources.begin(), sources.end(), [&](SyncCoverNodeId source) {
          return source >= nodes_.size() ||
                 nodes_[source].resource != resource ||
                 nodes_[source].order >= nodes_[physicalTarget].order ||
                 !syncCoverGuardsCompatible(nodes_[source].guard,
                                            nodes_[physicalTarget].guard);
        });
    if (invalidSources) {
      return {SyncCoverGraphError::InvalidCompletionTargets, physicalTarget};
    }
  }
  return {SyncCoverGraphError::None, std::nullopt};
}

SyncCoverGraphResult SyncCoverGraph::validateScopesControlsAndNodes() const {
  for (std::size_t index = 0; index < scopes_.size(); ++index) {
    const SyncCoverScope &scope = scopes_[index];
    const bool invalidScope = scope.id != index ||
                              !hasValidScope(scope.parent) ||
                              (index != 0 && scope.parent == index);
    if (invalidScope) {
      return {SyncCoverGraphError::InvalidScope, index};
    }
    const bool invalidTimeline =
        (scope.timeline && scope.timeline->begin > scope.timeline->end) ||
        (scope.isLoop && !scope.timeline);
    if (invalidTimeline) {
      return {SyncCoverGraphError::InvalidTimeline, index};
    }
    SyncCoverGuard guard = scope.guard;
    const SyncCoverGraphError guardError =
        normalizeAndValidateGuard(guard, index == 0 ? 0 : scope.parent);
    if (guardError != SyncCoverGraphError::None ||
        guard.literals != scope.guard.literals ||
        (index != 0 &&
         !std::includes(scope.guard.literals.begin(),
                        scope.guard.literals.end(),
                        scopes_[scope.parent].guard.literals.begin(),
                        scopes_[scope.parent].guard.literals.end()))) {
      return {guardError == SyncCoverGraphError::None
                  ? SyncCoverGraphError::InvalidGuard
                  : guardError,
              index};
    }
    if (index != 0 && scope.timeline) {
      const std::optional<SyncCoverScopeId> parentTimelineScope =
          getOwningTimelineScope(scope.parent);
      if (!parentTimelineScope) {
        return {SyncCoverGraphError::InvalidTimeline, index};
      }
      const SyncCoverTimelineInterval &parentTimeline =
          *scopes_[*parentTimelineScope].timeline;
      if (scope.timeline->begin < parentTimeline.begin ||
          scope.timeline->end > parentTimeline.end) {
        return {SyncCoverGraphError::InvalidTimeline, index};
      }
    }
  }
  for (std::size_t index = 0; index < controls_.size(); ++index) {
    const SyncCoverControl &control = controls_[index];
    if (control.id != index || control.alternatives == 0 ||
        !hasValidScope(control.scope)) {
      return {SyncCoverGraphError::InvalidControl, index};
    }
    if (control.phaseRelation) {
      const SyncCoverControlPhaseRelation &relation = *control.phaseRelation;
      const bool invalidRelation =
          relation.loopScope == 0 || !hasValidScope(relation.loopScope) ||
          !scopes_[relation.loopScope].isLoop ||
          getNearestEnclosingLoop(control.scope) != relation.loopScope ||
          relation.nextPhase.empty() ||
          relation.nextPhase.size() != relation.activeAlternative.size() ||
          relation.initialPhase >= relation.nextPhase.size() ||
          std::any_of(relation.nextPhase.begin(), relation.nextPhase.end(),
                      [&](std::size_t phase) {
                        return phase >= relation.nextPhase.size();
                      }) ||
          std::any_of(relation.activeAlternative.begin(),
                      relation.activeAlternative.end(),
                      [&](unsigned alternative) {
                        return alternative >= control.alternatives;
                      });
      if (invalidRelation) {
        return {SyncCoverGraphError::InvalidControl, index};
      }
    }
  }
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    const SyncCoverNode &node = nodes_[index];
    if (node.id != index || !hasValidScope(node.scope)) {
      return {SyncCoverGraphError::InvalidScope, index};
    }
    const std::optional<SyncCoverTimelineInterval> anchors =
        getNodeAnchorInterval(node.order);
    const std::optional<SyncCoverScopeId> timelineScope =
        getOwningTimelineScope(node.scope);
    const bool invalidOrder =
        !anchors || !timelineScope ||
        anchors->begin < scopes_[*timelineScope].timeline->begin ||
        anchors->end > scopes_[*timelineScope].timeline->end ||
        (index != 0 && nodes_[index - 1].order >= node.order);
    if (invalidOrder) {
      return {SyncCoverGraphError::InvalidOrder, index};
    }
    const bool invalidPhysicalAnchor =
        node.physicalAnchor > index ||
        nodes_[node.physicalAnchor].physicalAnchor != node.physicalAnchor;
    const bool invalidPhysicalExit =
        node.physicalExit < index || node.physicalExit >= nodes_.size() ||
        nodes_[node.physicalExit].physicalAnchor != node.physicalAnchor ||
        nodes_[node.physicalExit].physicalExit != node.physicalExit ||
        node.physicalExit != nodes_[node.physicalAnchor].physicalExit;
    if (invalidPhysicalAnchor || invalidPhysicalExit) {
      return {SyncCoverGraphError::InvalidNode, index};
    }
    if (!std::is_sorted(node.completionTargets.begin(),
                        node.completionTargets.end()) ||
        std::adjacent_find(node.completionTargets.begin(),
                           node.completionTargets.end()) !=
            node.completionTargets.end()) {
      return {SyncCoverGraphError::InvalidCompletionTargets, index};
    }
    if (!std::is_sorted(node.completionDominatedSources.begin(),
                        node.completionDominatedSources.end()) ||
        std::adjacent_find(node.completionDominatedSources.begin(),
                           node.completionDominatedSources.end()) !=
            node.completionDominatedSources.end()) {
      return {SyncCoverGraphError::InvalidCompletionTargets, index};
    }
    for (SyncCoverNodeId source : node.completionDominatedSources) {
      const bool invalidDominance =
          source >= nodes_.size() || source == index ||
          nodes_[source].resource != node.resource ||
          nodes_[source].scope != node.scope ||
          nodes_[source].order >= node.order ||
          nodes_[source].guard.literals != node.guard.literals ||
          !node.completionSignalCoversIssuedPrefix ||
          !std::any_of(
              edges_.begin(), edges_.end(), [&](const SyncCoverEdge &edge) {
                return edge.source == source && edge.target == index &&
                       edge.distance == 0 &&
                       edge.kind ==
                           SyncCoverEdgeKind::CertifiedCompletionFrontier;
              });
      if (invalidDominance) {
        return {SyncCoverGraphError::InvalidCompletionTargets, index};
      }
    }
    SyncCoverGuard guard = node.guard;
    const SyncCoverGraphError error =
        normalizeAndValidateGuard(guard, node.scope);
    if (error != SyncCoverGraphError::None) {
      return {error, index};
    }
    if (guard.literals != node.guard.literals ||
        !std::includes(node.guard.literals.begin(), node.guard.literals.end(),
                       scopes_[node.scope].guard.literals.begin(),
                       scopes_[node.scope].guard.literals.end())) {
      return {SyncCoverGraphError::InvalidGuard, index};
    }
  }
  return {SyncCoverGraphError::None, std::nullopt};
}

SyncCoverGraphResult SyncCoverGraph::validateDemands() const {
  std::set<DemandKey> seen;
  for (std::size_t index = 0; index < demands_.size(); ++index) {
    const SyncCoverDemand &demand = demands_[index];
    const bool invalidNode =
        demand.source >= nodes_.size() || demand.target >= nodes_.size();
    if (invalidNode) {
      return {SyncCoverGraphError::InvalidNode, index};
    }
    const bool invalidScope =
        !hasValidScope(demand.scope) ||
        !scopeContains(demand.scope, nodes_[demand.source].scope) ||
        !scopeContains(demand.scope, nodes_[demand.target].scope);
    if (invalidScope) {
      return {SyncCoverGraphError::InvalidScope, index};
    }
    const bool invalidKind =
        demand.originalDemandCount == 0 || demand.provenanceKinds.empty() ||
        !std::is_sorted(demand.provenanceKinds.begin(),
                        demand.provenanceKinds.end()) ||
        std::adjacent_find(demand.provenanceKinds.begin(),
                           demand.provenanceKinds.end()) !=
            demand.provenanceKinds.end() ||
        std::any_of(
            demand.provenanceKinds.begin(), demand.provenanceKinds.end(),
            [](SyncCoverDemandKind kind) { return !isValidDemandKind(kind); });
    if (invalidKind) {
      return {SyncCoverGraphError::InvalidDemandKind, index};
    }
    if (demand.distance != 0 &&
        (demand.scope == 0 || !scopes_[demand.scope].isLoop)) {
      return {SyncCoverGraphError::InvalidDistance, index};
    }
    if (demand.distance == 0 && demand.source == demand.target) {
      return {SyncCoverGraphError::ZeroDistanceSelfDemand, index};
    }
    if (demand.distance == 0 &&
        nodes_[demand.source].order >= nodes_[demand.target].order) {
      return {SyncCoverGraphError::InvalidOrder, index};
    }
    SyncCoverGuard sourceGuard = demand.sourceGuard;
    SyncCoverGuard targetGuard = demand.targetGuard;
    const SyncCoverGraphError guardError =
        completeEndpointGuards(demand.source, demand.target, demand.scope,
                               demand.distance, sourceGuard, targetGuard);
    if (guardError != SyncCoverGraphError::None) {
      return {guardError, index};
    }
    if (sourceGuard.literals != demand.sourceGuard.literals ||
        targetGuard.literals != demand.targetGuard.literals) {
      return {SyncCoverGraphError::InvalidGuard, index};
    }
    if (!std::is_sorted(demand.storageWitnesses.begin(),
                        demand.storageWitnesses.end()) ||
        std::adjacent_find(demand.storageWitnesses.begin(),
                           demand.storageWitnesses.end()) !=
            demand.storageWitnesses.end()) {
      return {SyncCoverGraphError::InvalidStorageProvenance, index};
    }
    const SyncCoverGraphError storageError =
        validateDemandStorage(*this, demand);
    if (storageError != SyncCoverGraphError::None) {
      return {storageError, index};
    }
    const DemandKey key{demand.source,
                        demand.target,
                        demand.scope,
                        demand.distance,
                        demand.sourceGuard.literals,
                        demand.targetGuard.literals};
    if (!seen.insert(key).second) {
      return {SyncCoverGraphError::DuplicateDemand, index};
    }
  }
  return {SyncCoverGraphError::None, std::nullopt};
}

SyncCoverGraphResult SyncCoverGraph::validateEdges() const {
  for (const auto &[resource, kind] : resourceRecurrenceCarryKinds_) {
    (void)resource;
    const bool invalidKind =
        kind != SyncCoverEdgeKind::CompletionPreservingIssueOrder &&
        kind != SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
    if (invalidKind) {
      return {SyncCoverGraphError::InvalidEdgeKind, std::nullopt};
    }
  }
  std::vector<std::vector<SyncCoverNodeId>> children(nodes_.size());
  std::vector<std::size_t> indegrees(nodes_.size(), 0);
  std::set<EdgeKey> seen;
  for (std::size_t index = 0; index < edges_.size(); ++index) {
    const SyncCoverEdge &edge = edges_[index];
    const bool invalidNode =
        edge.source >= nodes_.size() || edge.target >= nodes_.size();
    if (invalidNode) {
      return {SyncCoverGraphError::InvalidNode, index};
    }
    const bool invalidScope =
        !hasValidScope(edge.scope) ||
        !scopeContains(edge.scope, nodes_[edge.source].scope) ||
        !scopeContains(edge.scope, nodes_[edge.target].scope);
    if (invalidScope) {
      return {SyncCoverGraphError::InvalidScope, index};
    }
    if (!isValidEdgeKind(edge.kind)) {
      return {SyncCoverGraphError::InvalidEdgeKind, index};
    }
    const bool issueOrder =
        edge.kind == SyncCoverEdgeKind::CertifiedCompletionFrontier ||
        edge.kind == SyncCoverEdgeKind::CompletionPreservingIssueOrder ||
        edge.kind == SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
    if (issueOrder &&
        nodes_[edge.source].resource != nodes_[edge.target].resource) {
      return {SyncCoverGraphError::InvalidEdgeKind, index};
    }
    if (edge.kind == SyncCoverEdgeKind::CertifiedCompletionFrontier &&
        !nodes_[edge.target].completionSignalCoversIssuedPrefix) {
      return {SyncCoverGraphError::InvalidEdgeKind, index};
    }
    if (edge.distance != 0 &&
        (edge.scope == 0 || !scopes_[edge.scope].isLoop)) {
      return {SyncCoverGraphError::InvalidDistance, index};
    }
    SyncCoverGuard sourceGuard = edge.sourceGuard;
    SyncCoverGuard targetGuard = edge.targetGuard;
    const SyncCoverGraphError guardError =
        completeEndpointGuards(edge.source, edge.target, edge.scope,
                               edge.distance, sourceGuard, targetGuard);
    if (guardError != SyncCoverGraphError::None) {
      return {guardError, index};
    }
    if (sourceGuard.literals != edge.sourceGuard.literals ||
        targetGuard.literals != edge.targetGuard.literals) {
      return {SyncCoverGraphError::InvalidGuard, index};
    }
    const EdgeKey key{edge.source,
                      edge.target,
                      edge.scope,
                      edge.distance,
                      edge.sourceGuard.literals,
                      edge.targetGuard.literals};
    if (!seen.insert(key).second) {
      return {SyncCoverGraphError::DuplicateEdge, index};
    }
    if (edge.distance != 0) {
      continue;
    }
    if (edge.source == edge.target) {
      return {SyncCoverGraphError::ZeroDistanceSelfEdge, index};
    }
    if (nodes_[edge.source].order >= nodes_[edge.target].order) {
      return {SyncCoverGraphError::InvalidOrder, index};
    }
    children[edge.source].push_back(edge.target);
    ++indegrees[edge.target];
  }
  std::vector<SyncCoverNodeId> ready;
  for (SyncCoverNodeId node = 0; node < nodes_.size(); ++node) {
    if (indegrees[node] == 0) {
      ready.push_back(node);
    }
  }
  for (std::size_t index = 0; index < ready.size(); ++index) {
    for (SyncCoverNodeId child : children[ready[index]]) {
      if (--indegrees[child] == 0) {
        ready.push_back(child);
      }
    }
  }
  const bool hasCycle = ready.size() != nodes_.size();
  if (hasCycle) {
    return {SyncCoverGraphError::ZeroDistanceCycle, std::nullopt};
  }
  return {SyncCoverGraphError::None, std::nullopt};
}

SyncCoverGraphResult SyncCoverGraph::validateStorage() const {
  for (std::size_t index = 0; index < storageDomains_.size(); ++index) {
    const SyncCoverStorageDomainRole role = storageDomains_[index].role;
    const bool validRole =
        role == SyncCoverStorageDomainRole::Unspecified ||
        role == SyncCoverStorageDomainRole::Other ||
        role == SyncCoverStorageDomainRole::L0Left ||
        role == SyncCoverStorageDomainRole::L0Right ||
        role == SyncCoverStorageDomainRole::Accumulator;
    if (storageDomains_[index].id != index || !validRole) {
      return {SyncCoverGraphError::InvalidStorageDomain, index};
    }
  }
  for (std::size_t index = 0; index < storageAccesses_.size(); ++index) {
    const SyncCoverStorageAccess &access = storageAccesses_[index];
    const bool invalidAccess = access.id != index ||
                               access.node >= nodes_.size() ||
                               access.domain >= storageDomains_.size() ||
                               access.extent.begin >= access.extent.end ||
                               !isValidAccessMode(access.mode);
    if (invalidAccess) {
      return {SyncCoverGraphError::InvalidStorageAccess, index};
    }
  }
  for (std::size_t index = 0; index < storageWitnesses_.size(); ++index) {
    const SyncCoverStorageWitness &witness = storageWitnesses_[index];
    if (witness.id != index ||
        witness.sourceAccess >= storageAccesses_.size() ||
        witness.targetAccess >= storageAccesses_.size() ||
        witness.overlap.begin >= witness.overlap.end) {
      return {SyncCoverGraphError::InvalidStorageWitness, index};
    }
    const SyncCoverStorageAccess &source =
        storageAccesses_[witness.sourceAccess];
    const SyncCoverStorageAccess &target =
        storageAccesses_[witness.targetAccess];
    const SyncCoverStorageInterval expected{
        std::max(source.extent.begin, target.extent.begin),
        std::min(source.extent.end, target.extent.end)};
    if (source.domain != target.domain || expected.begin >= expected.end ||
        witness.overlap.begin != expected.begin ||
        witness.overlap.end != expected.end) {
      return {SyncCoverGraphError::InvalidStorageWitness, index};
    }
  }
  return {SyncCoverGraphError::None, std::nullopt};
}

SyncCoverGraphResult
SyncCoverGraph::validateTargetCompletionCertificates() const {
  for (std::size_t index = 0; index < targetCompletionCertificates_.size();
       ++index) {
    const SyncCoverTargetCompletionCertificate &certificate =
        targetCompletionCertificates_[index];
    const bool invalidHeader =
        certificate.id != index || certificate.completionNode >= nodes_.size() ||
        certificate.target >= nodes_.size() ||
        certificate.storageDomains.empty() || certificate.demands.empty() ||
        !std::is_sorted(certificate.storageDomains.begin(),
                        certificate.storageDomains.end()) ||
        std::adjacent_find(certificate.storageDomains.begin(),
                           certificate.storageDomains.end()) !=
            certificate.storageDomains.end() ||
        std::any_of(certificate.storageDomains.begin(),
                    certificate.storageDomains.end(),
                    [&](SyncCoverStorageDomainId domain) {
                      return domain >= storageDomains_.size();
                    }) ||
        !std::is_sorted(certificate.demands.begin(),
                        certificate.demands.end()) ||
        std::adjacent_find(certificate.demands.begin(),
                           certificate.demands.end()) !=
            certificate.demands.end() ||
        std::any_of(certificate.demands.begin(), certificate.demands.end(),
                    [&](SyncCoverDemandId demand) {
                      return demand >= demands_.size();
                    });
    if (invalidHeader) {
      return {SyncCoverGraphError::InvalidTargetCompletionCertificate, index};
    }
    const SyncCoverNode &completion = nodes_[certificate.completionNode];
    const SyncCoverNode &physicalTarget = nodes_[certificate.target];
    if (!targetCompletionResources_) {
      return {SyncCoverGraphError::InvalidTargetCompletionCertificate, index};
    }
    const std::uint32_t mte1 = targetCompletionResources_->mte1;
    const std::uint32_t matrix = targetCompletionResources_->matrix;
    const std::uint32_t fix = targetCompletionResources_->fix;
    const bool mte1L0Ready =
        certificate.kind ==
            SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix &&
        certificate.sourceResource == mte1 &&
        certificate.targetResource == matrix &&
        std::all_of(certificate.storageDomains.begin(),
                    certificate.storageDomains.end(),
                    [&](SyncCoverStorageDomainId domain) {
                      const SyncCoverStorageDomainRole role =
                          storageDomains_[domain].role;
                      return role == SyncCoverStorageDomainRole::L0Left ||
                             role == SyncCoverStorageDomainRole::L0Right;
                    });
    const bool mToFix =
        certificate.kind ==
            SyncCoverTargetCompletionKind::MToFixAccumulatorBoundary &&
        certificate.sourceResource == matrix &&
        certificate.targetResource == fix &&
        std::all_of(certificate.storageDomains.begin(),
                    certificate.storageDomains.end(),
                    [&](SyncCoverStorageDomainId domain) {
                      return storageDomains_[domain].role ==
                             SyncCoverStorageDomainRole::Accumulator;
                    });
    if ((!mte1L0Ready && !mToFix) ||
        completion.order >= physicalTarget.order ||
        completion.scope != physicalTarget.scope ||
        completion.guard.literals != physicalTarget.guard.literals ||
        physicalTarget.physicalAnchor != certificate.target) {
      return {SyncCoverGraphError::InvalidTargetCompletionCertificate, index};
    }
    bool completionNamesCertifiedSource = false;
    for (SyncCoverDemandId demandId : certificate.demands) {
      const SyncCoverDemand &demand = demands_[demandId];
      const SyncCoverNode &source = nodes_[demand.source];
      const SyncCoverNode &target = nodes_[demand.target];
      const SyncCoverNode &physicalSource = nodes_[source.physicalExit];
      completionNamesCertifiedSource |=
          source.physicalExit == certificate.completionNode;
      const bool invalidDemand =
          demand.distance != 0 ||
          std::find(demand.provenanceKinds.begin(),
                    demand.provenanceKinds.end(),
                    SyncCoverDemandKind::MemoryRAW) ==
              demand.provenanceKinds.end() ||
          source.resource != certificate.sourceResource ||
          target.resource != certificate.targetResource ||
          target.physicalAnchor != certificate.target ||
          source.scope != completion.scope || target.scope != completion.scope ||
          physicalSource.scope != completion.scope ||
          source.guard.literals != completion.guard.literals ||
          target.guard.literals != completion.guard.literals ||
          physicalSource.guard.literals != completion.guard.literals ||
          source.order > physicalSource.order ||
          physicalSource.order > completion.order;
      if (demand.storageWitnesses.empty() || invalidDemand) {
        return {SyncCoverGraphError::InvalidTargetCompletionCertificate,
                index};
      }
      const bool hasExactRawWitness = std::any_of(
          demand.storageWitnesses.begin(), demand.storageWitnesses.end(),
          [&](SyncCoverStorageWitnessId witnessId) {
            if (witnessId >= storageWitnesses_.size()) {
              return false;
            }
            const SyncCoverStorageWitness &witness =
                storageWitnesses_[witnessId];
            const SyncCoverStorageAccess &sourceAccess =
                storageAccesses_[witness.sourceAccess];
            const SyncCoverStorageAccess &targetAccess =
                storageAccesses_[witness.targetAccess];
            return sourceAccess.node == demand.source &&
                   targetAccess.node == demand.target &&
                   sourceAccess.domain == targetAccess.domain &&
                   std::binary_search(certificate.storageDomains.begin(),
                                      certificate.storageDomains.end(),
                                      sourceAccess.domain) &&
                   sourceAccess.exactPhysical && targetAccess.exactPhysical &&
                   syncCoverStorageModeWrites(sourceAccess.mode) &&
                   syncCoverStorageModeReads(targetAccess.mode);
          });
      if (!hasExactRawWitness) {
        return {SyncCoverGraphError::InvalidTargetCompletionCertificate,
                index};
      }
      if (certificate.kind ==
              SyncCoverTargetCompletionKind::MToFixAccumulatorBoundary &&
          source.physicalExit != certificate.completionNode) {
        return {SyncCoverGraphError::InvalidTargetCompletionCertificate,
                index};
      }
    }
    if (!completionNamesCertifiedSource) {
      return {SyncCoverGraphError::InvalidTargetCompletionCertificate, index};
    }
  }
  return {SyncCoverGraphError::None, std::nullopt};
}
