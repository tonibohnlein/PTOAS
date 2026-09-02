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
#include <limits>
#include <set>
#include <sstream>
#include <utility>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_detail;

SyncCoverGraphResult
SyncCoverGraph::addScope(SyncCoverScopeId parent, bool mustExecuteWithinParent,
                         std::optional<SyncCoverTimelineInterval> timeline,
                         bool isLoop, SyncCoverGuard guard,
                         SyncCoverRegionId region) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, scopes_.size()};
  }
  if (!hasValidScope(parent)) {
    return {SyncCoverGraphError::InvalidScope, parent};
  }
  if (!hasValidRegion(region)) {
    return {SyncCoverGraphError::InvalidRegion, region};
  }
  const SyncCoverGraphError guardError =
      normalizeAndValidateGuard(guard, parent);
  if (guardError != SyncCoverGraphError::None) {
    return {guardError, scopes_.size()};
  }
  if (!std::includes(guard.literals.begin(), guard.literals.end(),
                     scopes_[parent].guard.literals.begin(),
                     scopes_[parent].guard.literals.end())) {
    return {SyncCoverGraphError::InvalidGuard, scopes_.size()};
  }
  const bool invalidTimeline =
      (timeline && timeline->begin > timeline->end) || (isLoop && !timeline);
  if (invalidTimeline) {
    return {SyncCoverGraphError::InvalidTimeline, scopes_.size()};
  }
  const std::optional<SyncCoverScopeId> owningTimeline =
      getOwningTimelineScope(parent);
  if (timeline && owningTimeline) {
    const SyncCoverTimelineInterval &parentTimeline =
        *scopes_[*owningTimeline].timeline;
    if (timeline->begin < parentTimeline.begin ||
        timeline->end > parentTimeline.end) {
      return {SyncCoverGraphError::InvalidTimeline, scopes_.size()};
    }
  }
  const SyncCoverScopeId id = scopes_.size();
  scopes_.push_back({id, parent, region, mustExecuteWithinParent, timeline,
                     isLoop, std::move(guard)});
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult SyncCoverGraph::setScopeRegion(SyncCoverScopeId scope,
                                                    SyncCoverRegionId region) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, scope};
  }
  const bool invalidScope = !hasValidScope(scope) || scope == 0;
  if (invalidScope) {
    return {SyncCoverGraphError::InvalidScope, scope};
  }
  const bool invalidRegion = !hasValidRegion(region) ||
                             (region != 0 && regions_[region].scope != scope);
  if (invalidRegion) {
    return {SyncCoverGraphError::InvalidRegion, region};
  }
  scopes_[scope].region = region;
  return {SyncCoverGraphError::None, scope};
}

SyncCoverGraphResult
SyncCoverGraph::addRegion(SyncCoverRegionId parent, SyncCoverRegionKind kind,
                          SyncCoverRegionCardinality cardinality,
                          SyncCoverScopeId scope, SyncCoverGuard guard,
                          std::optional<SyncCoverControlId> control,
                          std::optional<unsigned> alternative) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, regions_.size()};
  }
  if (!hasValidRegion(parent)) {
    return {SyncCoverGraphError::InvalidRegion, parent};
  }
  if (!hasValidScope(scope)) {
    return {SyncCoverGraphError::InvalidScope, scope};
  }
  const SyncCoverGraphError guardError =
      normalizeAndValidateGuard(guard, scope);
  if (guardError != SyncCoverGraphError::None) {
    return {guardError, regions_.size()};
  }
  if (!std::includes(guard.literals.begin(), guard.literals.end(),
                     regions_[parent].guard.literals.begin(),
                     regions_[parent].guard.literals.end())) {
    return {SyncCoverGraphError::InvalidGuard, regions_.size()};
  }
  const SyncCoverRegionId id = regions_.size();
  regions_.push_back({id,
                      parent,
                      scope,
                      kind,
                      cardinality,
                      control,
                      alternative,
                      std::move(guard),
                      {},
                      {}});
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult SyncCoverGraph::addControl(unsigned alternatives,
                                                SyncCoverScopeId scope) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, controls_.size()};
  }
  if (alternatives == 0) {
    return {SyncCoverGraphError::InvalidControl, controls_.size()};
  }
  if (!hasValidScope(scope)) {
    return {SyncCoverGraphError::InvalidScope, scope};
  }
  const SyncCoverControlId id = controls_.size();
  controls_.push_back({id, alternatives, scope, scopes_[scope].region});
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult
SyncCoverGraph::setControlRegion(SyncCoverControlId control,
                                 SyncCoverRegionId region) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, control};
  }
  if (control >= controls_.size()) {
    return {SyncCoverGraphError::InvalidControl, control};
  }
  const bool invalidRegion =
      !hasValidRegion(region) ||
      regions_[region].scope != controls_[control].scope ||
      regions_[region].kind != SyncCoverRegionKind::Choice ||
      regions_[region].control != control;
  if (invalidRegion) {
    return {SyncCoverGraphError::InvalidRegion, region};
  }
  controls_[control].region = region;
  return {SyncCoverGraphError::None, control};
}

SyncCoverGraphResult SyncCoverGraph::setControlPhaseRelation(
    SyncCoverControlId control, SyncCoverControlPhaseRelation relation) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, control};
  }
  const bool invalidRelation =
      control >= controls_.size() || relation.loopScope == 0 ||
      !hasValidScope(relation.loopScope) ||
      !scopes_[relation.loopScope].isLoop ||
      getNearestEnclosingLoop(controls_[control].scope) != relation.loopScope ||
      relation.nextPhase.empty() ||
      relation.nextPhase.size() != relation.activeAlternative.size() ||
      relation.initialPhase >= relation.nextPhase.size() ||
      std::any_of(relation.nextPhase.begin(), relation.nextPhase.end(),
                  [&](std::size_t phase) {
                    return phase >= relation.nextPhase.size();
                  }) ||
      std::any_of(relation.activeAlternative.begin(),
                  relation.activeAlternative.end(), [&](unsigned alternative) {
                    return alternative >= controls_[control].alternatives;
                  });
  if (invalidRelation) {
    return {SyncCoverGraphError::InvalidControl, control};
  }
  controls_[control].phaseRelation = std::move(relation);
  return {SyncCoverGraphError::None, control};
}

SyncCoverGraphResult SyncCoverGraph::setControlSuccessorRelation(
    SyncCoverControlId control, SyncCoverControlSuccessorRelation relation) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, control};
  }
  const bool invalidRelation =
      control >= controls_.size() || relation.loopScope == 0 ||
      !hasValidScope(relation.loopScope) ||
      !scopes_[relation.loopScope].isLoop ||
      getNearestEnclosingLoop(controls_[control].scope) != relation.loopScope ||
      relation.hasSuccessorAlternative >= controls_[control].alternatives;
  if (invalidRelation) {
    return {SyncCoverGraphError::InvalidControl, control};
  }
  controls_[control].successorRelation = relation;
  return {SyncCoverGraphError::None, control};
}

SyncCoverGraphResult SyncCoverGraph::setControlFirstIterationRelation(
    SyncCoverControlId control,
    SyncCoverControlFirstIterationRelation relation) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, control};
  }
  const bool invalidRelation =
      control >= controls_.size() || relation.loopScope == 0 ||
      !hasValidScope(relation.loopScope) ||
      !scopes_[relation.loopScope].isLoop ||
      getNearestEnclosingLoop(controls_[control].scope) != relation.loopScope ||
      relation.firstIterationAlternative >= controls_[control].alternatives;
  if (invalidRelation) {
    return {SyncCoverGraphError::InvalidControl, control};
  }
  controls_[control].firstIterationRelation = relation;
  return {SyncCoverGraphError::None, control};
}

SyncCoverGraphResult
SyncCoverGraph::setScopeTimeline(SyncCoverScopeId scope,
                                 SyncCoverTimelineInterval timeline) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, scope};
  }
  const bool invalidTimeline =
      scope == 0 || !hasValidScope(scope) || timeline.begin > timeline.end;
  if (invalidTimeline) {
    return {SyncCoverGraphError::InvalidTimeline, scope};
  }
  const std::optional<SyncCoverScopeId> parentTimelineScope =
      getOwningTimelineScope(scopes_[scope].parent);
  if (!parentTimelineScope) {
    return {SyncCoverGraphError::InvalidTimeline, scope};
  }
  const SyncCoverTimelineInterval &parentTimeline =
      *scopes_[*parentTimelineScope].timeline;
  if (timeline.begin < parentTimeline.begin ||
      timeline.end > parentTimeline.end) {
    return {SyncCoverGraphError::InvalidTimeline, scope};
  }
  scopes_[scope].timeline = timeline;
  return {SyncCoverGraphError::None, scope};
}

SyncCoverGraphResult SyncCoverGraph::addNode(
    std::uint32_t resource, std::uint64_t weight, SyncCoverScopeId scope,
    std::size_t order, SyncCoverGuard guard,
    std::vector<std::uint32_t> completionTargets,
    std::optional<SyncCoverNodeId> physicalAnchor,
    bool completionSignalCoversIssuedPrefix, std::size_t physicalOperation,
    int macroPhase, std::vector<unsigned> completedResults,
    SyncCoverRegionId region) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, nodes_.size()};
  }
  if (!hasValidScope(scope)) {
    return {SyncCoverGraphError::InvalidScope, scope};
  }
  const bool invalidRegion = !hasValidRegion(region) ||
                             (region != 0 && regions_[region].scope != scope);
  if (invalidRegion) {
    return {SyncCoverGraphError::InvalidRegion, region};
  }
  const SyncCoverGraphError error = normalizeAndValidateGuard(guard, scope);
  if (error != SyncCoverGraphError::None) {
    return {error, nodes_.size()};
  }
  if (!std::includes(guard.literals.begin(), guard.literals.end(),
                     scopes_[scope].guard.literals.begin(),
                     scopes_[scope].guard.literals.end())) {
    return {SyncCoverGraphError::InvalidGuard, nodes_.size()};
  }
  const std::optional<SyncCoverTimelineInterval> anchors =
      getNodeAnchorInterval(order);
  const std::optional<SyncCoverScopeId> timelineScope =
      getOwningTimelineScope(scope);
  if (!anchors || !timelineScope ||
      anchors->begin < scopes_[*timelineScope].timeline->begin ||
      anchors->end > scopes_[*timelineScope].timeline->end) {
    return {SyncCoverGraphError::InvalidOrder, nodes_.size()};
  }
  const bool nonMonotonicOrder =
      !nodes_.empty() && nodes_.back().order >= order;
  if (nonMonotonicOrder) {
    return {SyncCoverGraphError::InvalidOrder, nodes_.size()};
  }
  std::sort(completionTargets.begin(), completionTargets.end());
  completionTargets.erase(
      std::unique(completionTargets.begin(), completionTargets.end()),
      completionTargets.end());
  std::sort(completedResults.begin(), completedResults.end());
  completedResults.erase(
      std::unique(completedResults.begin(), completedResults.end()),
      completedResults.end());
  const SyncCoverNodeId id = nodes_.size();
  if (physicalOperation == std::numeric_limits<std::size_t>::max()) {
    physicalOperation = id;
  }
  const SyncCoverNodeId anchor = physicalAnchor.value_or(id);
  if (anchor > id || (anchor < id && nodes_[anchor].physicalAnchor != anchor)) {
    return {SyncCoverGraphError::InvalidNode, anchor};
  }
  nodes_.push_back({id,
                    resource,
                    weight,
                    scope,
                    region,
                    order,
                    std::move(guard),
                    std::move(completionTargets),
                    {},
                    anchor,
                    id,
                    completionSignalCoversIssuedPrefix,
                    physicalOperation,
                    macroPhase,
                    std::move(completedResults)});
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult
SyncCoverGraph::setPhysicalExit(SyncCoverNodeId node,
                                SyncCoverNodeId physicalExit) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, node};
  }
  if (node >= nodes_.size() || physicalExit >= nodes_.size() ||
      nodes_[node].physicalAnchor != nodes_[physicalExit].physicalAnchor ||
      nodes_[node].order > nodes_[physicalExit].order) {
    return {SyncCoverGraphError::InvalidNode, node};
  }
  nodes_[node].physicalExit = physicalExit;
  return {SyncCoverGraphError::None, node};
}

SyncCoverGraphResult SyncCoverGraph::addEdge(SyncCoverEdge edge) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, edges_.size()};
  }
  const bool invalidNode =
      edge.source >= nodes_.size() || edge.target >= nodes_.size();
  if (invalidNode) {
    return {SyncCoverGraphError::InvalidNode, edges_.size()};
  }
  const bool invalidScope =
      !hasValidScope(edge.scope) ||
      !scopeContains(edge.scope, nodes_[edge.source].scope) ||
      !scopeContains(edge.scope, nodes_[edge.target].scope);
  if (invalidScope) {
    return {SyncCoverGraphError::InvalidScope, edges_.size()};
  }
  if (!isValidEdgeKind(edge.kind)) {
    return {SyncCoverGraphError::InvalidEdgeKind, edges_.size()};
  }
  if (edge.kind == SyncCoverEdgeKind::CompletionSupply &&
      !isValidOrderingRequirements(edge.suppliedRequirements)) {
    return {SyncCoverGraphError::InvalidOrderingRequirement, edges_.size()};
  }
  const bool issueOrder =
      edge.kind == SyncCoverEdgeKind::CertifiedCompletionFrontier ||
      edge.kind == SyncCoverEdgeKind::CompletionPreservingIssueOrder ||
      edge.kind == SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  if (issueOrder &&
      nodes_[edge.source].resource != nodes_[edge.target].resource) {
    return {SyncCoverGraphError::InvalidEdgeKind, edges_.size()};
  }
  if (edge.kind == SyncCoverEdgeKind::CertifiedCompletionFrontier &&
      !nodes_[edge.target].completionSignalCoversIssuedPrefix) {
    return {SyncCoverGraphError::InvalidEdgeKind, edges_.size()};
  }
  if (edge.distance != 0 && (edge.scope == 0 || !scopes_[edge.scope].isLoop)) {
    return {SyncCoverGraphError::InvalidDistance, edges_.size()};
  }
  if (edge.distance == 0 && edge.source == edge.target) {
    return {SyncCoverGraphError::ZeroDistanceSelfEdge, edges_.size()};
  }
  if (edge.distance == 0 &&
      nodes_[edge.source].order >= nodes_[edge.target].order) {
    return {SyncCoverGraphError::InvalidOrder, edges_.size()};
  }
  const SyncCoverGraphError error =
      completeEndpointGuards(edge.source, edge.target, edge.scope,
                             edge.distance, edge.sourceGuard, edge.targetGuard);
  if (error != SyncCoverGraphError::None) {
    return {error, edges_.size()};
  }
  const EdgeKey key{edge.source,
                    edge.target,
                    edge.scope,
                    edge.distance,
                    edge.sourceGuard.literals,
                    edge.targetGuard.literals};
  auto existing = edgeIds_.find(key);
  if (existing != edgeIds_.end()) {
    SyncCoverEdge &stored = edges_[existing->second];
    const bool stronger = edgeStrength(edge.kind) > edgeStrength(stored.kind);
    if (stronger) {
      stored.kind = edge.kind;
      if (edge.kind == SyncCoverEdgeKind::CompletionSupply) {
        stored.suppliedRequirements = edge.suppliedRequirements;
      }
    } else if (stored.kind == SyncCoverEdgeKind::CompletionSupply &&
               edge.kind == SyncCoverEdgeKind::CompletionSupply) {
      stored.suppliedRequirements |= edge.suppliedRequirements;
    }
    return {SyncCoverGraphError::None, existing->second};
  }
  const std::size_t index = edges_.size();
  edges_.push_back(std::move(edge));
  edgeIds_.emplace(key, index);
  return {SyncCoverGraphError::None, index};
}

SyncCoverGraphResult SyncCoverGraph::addDemand(SyncCoverDemand demand) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, demands_.size()};
  }
  const bool invalidNode =
      demand.source >= nodes_.size() || demand.target >= nodes_.size();
  if (invalidNode) {
    return {SyncCoverGraphError::InvalidNode, demands_.size()};
  }
  const bool invalidScope =
      !hasValidScope(demand.scope) ||
      !scopeContains(demand.scope, nodes_[demand.source].scope) ||
      !scopeContains(demand.scope, nodes_[demand.target].scope);
  if (invalidScope) {
    return {SyncCoverGraphError::InvalidScope, demands_.size()};
  }
  const std::optional<SyncCoverRegionId> owner =
      demand.distance == 0
          ? getLowestCommonRegion(nodes_[demand.source].region,
                                  nodes_[demand.target].region)
          : std::optional<SyncCoverRegionId>(scopes_[demand.scope].region);
  const bool invalidOwner =
      !owner || !regionContains(*owner, nodes_[demand.source].region) ||
      !regionContains(*owner, nodes_[demand.target].region);
  if (invalidOwner) {
    return {SyncCoverGraphError::InvalidRegion, demands_.size()};
  }
  demand.ownerRegion = *owner;
  const bool invalidKind =
      demand.originalDemandCount == 0 || demand.provenanceKinds.empty() ||
      std::any_of(
          demand.provenanceKinds.begin(), demand.provenanceKinds.end(),
          [](SyncCoverDemandKind kind) { return !isValidDemandKind(kind); });
  if (invalidKind) {
    return {SyncCoverGraphError::InvalidDemandKind, demands_.size()};
  }
  if (!isValidOrderingRequirements(demand.orderingRequirements)) {
    return {SyncCoverGraphError::InvalidOrderingRequirement, demands_.size()};
  }
  if (demand.distance != 0 &&
      (demand.scope == 0 || !scopes_[demand.scope].isLoop)) {
    return {SyncCoverGraphError::InvalidDistance, demands_.size()};
  }
  if (demand.distance == 0 && demand.source == demand.target) {
    return {SyncCoverGraphError::ZeroDistanceSelfDemand, demands_.size()};
  }
  if (demand.distance == 0 &&
      nodes_[demand.source].order >= nodes_[demand.target].order) {
    return {SyncCoverGraphError::InvalidOrder, demands_.size()};
  }
  const SyncCoverGraphError guardError = completeEndpointGuards(
      demand.source, demand.target, demand.scope, demand.distance,
      demand.sourceGuard, demand.targetGuard);
  if (guardError != SyncCoverGraphError::None) {
    return {guardError, demands_.size()};
  }
  std::sort(demand.provenanceKinds.begin(), demand.provenanceKinds.end());
  demand.provenanceKinds.erase(
      std::unique(demand.provenanceKinds.begin(), demand.provenanceKinds.end()),
      demand.provenanceKinds.end());
  std::sort(demand.storageWitnesses.begin(), demand.storageWitnesses.end());
  demand.storageWitnesses.erase(std::unique(demand.storageWitnesses.begin(),
                                            demand.storageWitnesses.end()),
                                demand.storageWitnesses.end());
  const SyncCoverGraphError storageError = validateDemandStorage(*this, demand);
  if (storageError != SyncCoverGraphError::None) {
    return {storageError, demands_.size()};
  }

  const DemandKey key{demand.source,
                      demand.target,
                      demand.scope,
                      demand.distance,
                      demand.sourceGuard.literals,
                      demand.targetGuard.literals};
  auto existing = demandIds_.find(key);
  if (existing == demandIds_.end()) {
    const std::size_t index = demands_.size();
    demands_.push_back(std::move(demand));
    demandIds_.emplace(key, index);
    return {SyncCoverGraphError::None, index};
  }

  SyncCoverDemand merged = demands_[existing->second];
  const bool provenanceCountOverflows =
      demand.originalDemandCount >
      std::numeric_limits<std::size_t>::max() - merged.originalDemandCount;
  if (provenanceCountOverflows) {
    return {SyncCoverGraphError::ArithmeticOverflow, existing->second};
  }
  merged.originalDemandCount += demand.originalDemandCount;
  merged.orderingRequirements |= demand.orderingRequirements;
  merged.provenanceKinds.insert(merged.provenanceKinds.end(),
                                demand.provenanceKinds.begin(),
                                demand.provenanceKinds.end());
  std::sort(merged.provenanceKinds.begin(), merged.provenanceKinds.end());
  merged.provenanceKinds.erase(
      std::unique(merged.provenanceKinds.begin(), merged.provenanceKinds.end()),
      merged.provenanceKinds.end());
  merged.storageWitnesses.insert(merged.storageWitnesses.end(),
                                 demand.storageWitnesses.begin(),
                                 demand.storageWitnesses.end());
  std::sort(merged.storageWitnesses.begin(), merged.storageWitnesses.end());
  merged.storageWitnesses.erase(std::unique(merged.storageWitnesses.begin(),
                                            merged.storageWitnesses.end()),
                                merged.storageWitnesses.end());
  const SyncCoverGraphError mergedStorageError =
      validateDemandStorage(*this, merged);
  if (mergedStorageError != SyncCoverGraphError::None) {
    return {mergedStorageError, demands_.size()};
  }
  demands_[existing->second] = std::move(merged);
  return {SyncCoverGraphError::None, existing->second};
}

std::string SyncCoverGraph::getDeterministicRawDump() const {
  std::ostringstream output;
  const auto printGuard = [&](const SyncCoverGuard &guard) {
    output << '[';
    for (std::size_t index = 0; index < guard.literals.size(); ++index) {
      if (index != 0) {
        output << ',';
      }
      output << guard.literals[index].control << ':'
             << guard.literals[index].alternative;
    }
    output << ']';
  };
  for (const SyncCoverScope &scope : scopes_) {
    output << "scope " << scope.id << " parent=" << scope.parent
           << " region=" << scope.region
           << " must=" << scope.mustExecuteWithinParent
           << " loop=" << scope.isLoop << " timeline=";
    if (scope.timeline) {
      output << '[' << scope.timeline->begin << ',' << scope.timeline->end
             << ']';
    } else {
      output << "none";
    }
    output << " guard=";
    printGuard(scope.guard);
    output << '\n';
  }
  for (const SyncCoverRegion &region : regions_) {
    output << "region " << region.id << " parent=" << region.parent
           << " scope=" << region.scope
           << " kind=" << static_cast<unsigned>(region.kind)
           << " cardinality=" << static_cast<unsigned>(region.cardinality)
           << " control=";
    if (region.control) {
      output << *region.control;
    } else {
      output << "none";
    }
    output << " alternative=";
    if (region.alternative) {
      output << *region.alternative;
    } else {
      output << "none";
    }
    output << " guard=";
    printGuard(region.guard);
    output << " elements=[";
    for (std::size_t index = 0; index < region.elements.size(); ++index) {
      if (index != 0) {
        output << ',';
      }
      output << static_cast<unsigned>(region.elements[index].kind) << ':'
             << region.elements[index].value;
    }
    output << "] ports=[";
    for (std::size_t index = 0; index < region.ports.size(); ++index) {
      if (index != 0) {
        output << ',';
      }
      output << region.ports[index];
    }
    output << "]\n";
  }
  for (const SyncCoverRegionPort &port : regionPorts_) {
    output << "region-port " << port.id << " region=" << port.region
           << " kind=" << static_cast<unsigned>(port.kind)
           << " pipe=" << port.resource << " node=";
    if (port.node) {
      output << *port.node;
    } else {
      output << "none";
    }
    output << " demand=";
    if (port.demand) {
      output << *port.demand;
    } else {
      output << "none";
    }
    output << '\n';
  }
  for (const SyncCoverControl &control : controls_) {
    output << "control " << control.id
           << " alternatives=" << control.alternatives
           << " scope=" << control.scope << " region=" << control.region
           << " phase-relation=";
    if (control.phaseRelation) {
      output << "loop:" << control.phaseRelation->loopScope
             << ",initial:" << control.phaseRelation->initialPhase
             << ",states:[";
      for (std::size_t phase = 0;
           phase < control.phaseRelation->nextPhase.size(); ++phase) {
        if (phase != 0) {
          output << ',';
        }
        output << phase << ':'
               << control.phaseRelation->activeAlternative[phase] << "->"
               << control.phaseRelation->nextPhase[phase];
      }
      output << ']';
    } else {
      output << "none";
    }
    output << " successor-relation=";
    if (control.successorRelation) {
      output << "loop:" << control.successorRelation->loopScope
             << ",alternative:"
             << control.successorRelation->hasSuccessorAlternative;
    } else {
      output << "none";
    }
    output << " first-iteration-relation=";
    if (control.firstIterationRelation) {
      output << "loop:" << control.firstIterationRelation->loopScope
             << ",alternative:"
             << control.firstIterationRelation->firstIterationAlternative;
    } else {
      output << "none";
    }
    output << '\n';
  }
  for (const SyncCoverNode &node : nodes_) {
    output << "node " << node.id << " op=" << node.physicalOperation
           << " phase=" << node.macroPhase << " pipe=" << node.resource
           << " scope=" << node.scope << " region=" << node.region
           << " order=" << node.order << " anchor=" << node.physicalAnchor
           << " exit=" << node.physicalExit
           << " prefix-signal=" << node.completionSignalCoversIssuedPrefix
           << " complete-results=";
    output << '[';
    for (std::size_t index = 0; index < node.completedResults.size(); ++index) {
      if (index != 0) {
        output << ',';
      }
      output << node.completedResults[index];
    }
    output << "] completion-targets=[";
    for (std::size_t index = 0; index < node.completionTargets.size();
         ++index) {
      if (index != 0) {
        output << ',';
      }
      output << node.completionTargets[index];
    }
    output << "] guard=";
    printGuard(node.guard);
    output << '\n';
  }
  for (std::size_t edgeId = 0; edgeId < edges_.size(); ++edgeId) {
    const SyncCoverEdge &edge = edges_[edgeId];
    output << "edge " << edgeId << " source=" << edge.source
           << " target=" << edge.target
           << " kind=" << static_cast<unsigned>(edge.kind)
           << " scope=" << edge.scope << " distance=" << edge.distance
           << " supplied-requirements="
           << static_cast<unsigned>(edge.suppliedRequirements)
           << " source-guard=";
    printGuard(edge.sourceGuard);
    output << " target-guard=";
    printGuard(edge.targetGuard);
    output << '\n';
  }
  for (const SyncCoverStorageDomain &domain : storageDomains_) {
    output << "domain " << domain.id
           << " role=" << static_cast<unsigned>(domain.role)
           << " space=" << domain.addressSpace << '\n';
  }
  for (const SyncCoverStorageAccess &access : storageAccesses_) {
    output << "access " << access.id << " node=" << access.node
           << " domain=" << access.domain << " family=" << access.family
           << " range=[" << access.extent.begin << ',' << access.extent.end
           << ") mode=" << static_cast<unsigned>(access.mode)
           << " path=" << static_cast<unsigned>(access.path)
           << " exact=" << access.exactPhysical << " ordinal=";
    if (access.addressOrdinal) {
      output << *access.addressOrdinal;
    } else {
      output << "none";
    }
    output << '\n';
  }
  for (const SyncCoverStorageWitness &witness : storageWitnesses_) {
    output << "witness " << witness.id << " source=" << witness.sourceAccess
           << " target=" << witness.targetAccess << " overlap=["
           << witness.overlap.begin << ',' << witness.overlap.end << ")\n";
  }
  for (SyncCoverDemandId demandId = 0; demandId < demands_.size(); ++demandId) {
    const SyncCoverDemand &demand = demands_[demandId];
    output << "demand " << demandId << " source=" << demand.source
           << " target=" << demand.target << " scope=" << demand.scope
           << " owner-region=" << demand.ownerRegion
           << " distance=" << demand.distance << " requirements="
           << static_cast<unsigned>(demand.orderingRequirements)
           << " original-count=" << demand.originalDemandCount << " kinds=";
    output << '[';
    for (std::size_t index = 0; index < demand.provenanceKinds.size();
         ++index) {
      if (index != 0) {
        output << ',';
      }
      output << static_cast<unsigned>(demand.provenanceKinds[index]);
    }
    output << "] witnesses=[";
    for (std::size_t index = 0; index < demand.storageWitnesses.size();
         ++index) {
      if (index != 0) {
        output << ',';
      }
      output << demand.storageWitnesses[index];
    }
    output << "] source-guard=";
    printGuard(demand.sourceGuard);
    output << " target-guard=";
    printGuard(demand.targetGuard);
    output << '\n';
  }
  return output.str();
}

SyncCoverGraphResult
SyncCoverGraph::setResourceRecurrenceCarryKind(std::uint32_t resource,
                                               SyncCoverEdgeKind kind) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen,
            resourceRecurrenceCarryKinds_.size()};
  }
  const bool invalidKind =
      kind != SyncCoverEdgeKind::CompletionPreservingIssueOrder &&
      kind != SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  if (invalidKind) {
    return {SyncCoverGraphError::InvalidEdgeKind,
            resourceRecurrenceCarryKinds_.size()};
  }
  resourceRecurrenceCarryKinds_[resource] = kind;
  return {SyncCoverGraphError::None, resource};
}

SyncCoverGraphResult SyncCoverGraph::setBlockingTargetedBarrierResources(
    std::vector<std::uint32_t> resources) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen,
            blockingTargetedBarrierResources_.size()};
  }
  std::sort(resources.begin(), resources.end());
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
  blockingTargetedBarrierResources_ = std::move(resources);
  return {SyncCoverGraphError::None, blockingTargetedBarrierResources_.size()};
}

SyncCoverGraphResult SyncCoverGraph::setCrossResourceTargetedBarrierPairs(
    std::vector<std::pair<std::uint32_t, std::uint32_t>> resourcePairs) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen,
            crossResourceTargetedBarrierPairs_.size()};
  }
  std::sort(resourcePairs.begin(), resourcePairs.end());
  resourcePairs.erase(std::unique(resourcePairs.begin(), resourcePairs.end()),
                      resourcePairs.end());
  const bool invalid =
      std::any_of(resourcePairs.begin(), resourcePairs.end(),
                  [](const auto &pair) { return pair.first == pair.second; });
  if (invalid) {
    return {SyncCoverGraphError::InvalidCompletionTargets,
            crossResourceTargetedBarrierPairs_.size()};
  }
  crossResourceTargetedBarrierPairs_ = std::move(resourcePairs);
  return {SyncCoverGraphError::None, crossResourceTargetedBarrierPairs_.size()};
}

SyncCoverGraphResult SyncCoverGraph::setTargetCompletionResources(
    SyncCoverTargetCompletionResources resources) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, std::nullopt};
  }
  const bool invalid = resources.mte1 == resources.matrix ||
                       resources.mte1 == resources.fix ||
                       resources.matrix == resources.fix;
  if (invalid) {
    return {SyncCoverGraphError::InvalidCompletionTargets, std::nullopt};
  }
  targetCompletionResources_ = resources;
  return {SyncCoverGraphError::None, std::nullopt};
}

bool SyncCoverGraph::supportsBlockingTargetedBarrier(
    std::uint32_t resource) const {
  return std::binary_search(blockingTargetedBarrierResources_.begin(),
                            blockingTargetedBarrierResources_.end(), resource);
}

bool SyncCoverGraph::supportsCrossResourceTargetedBarrier(
    std::uint32_t sourceResource, std::uint32_t targetResource) const {
  return sourceResource != targetResource &&
         std::binary_search(crossResourceTargetedBarrierPairs_.begin(),
                            crossResourceTargetedBarrierPairs_.end(),
                            std::make_pair(sourceResource, targetResource));
}

SyncCoverGraphResult SyncCoverGraph::setBlockingTargetedBarrierPrefix(
    std::uint32_t resource, SyncCoverNodeId physicalTarget,
    std::vector<SyncCoverNodeId> issuedSources) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen,
            blockingTargetedBarrierPrefixes_.size()};
  }
  if (physicalTarget >= nodes_.size() ||
      !supportsBlockingTargetedBarrier(resource)) {
    return {SyncCoverGraphError::InvalidNode, physicalTarget};
  }
  std::sort(issuedSources.begin(), issuedSources.end());
  issuedSources.erase(std::unique(issuedSources.begin(), issuedSources.end()),
                      issuedSources.end());
  const SyncCoverNode &target = nodes_[physicalTarget];
  const bool invalidSource = std::any_of(
      issuedSources.begin(), issuedSources.end(), [&](SyncCoverNodeId source) {
        return source >= nodes_.size() || nodes_[source].resource != resource ||
               nodes_[source].order >= target.order ||
               !syncCoverGuardsCompatible(nodes_[source].guard, target.guard);
      });
  if (invalidSource) {
    return {SyncCoverGraphError::InvalidOrder, physicalTarget};
  }
  const auto [position, inserted] = blockingTargetedBarrierPrefixes_.emplace(
      std::make_pair(resource, physicalTarget), std::move(issuedSources));
  if (!inserted) {
    (void)position;
    return {SyncCoverGraphError::DuplicateEdge, physicalTarget};
  }
  return {SyncCoverGraphError::None,
          blockingTargetedBarrierPrefixes_.size() - 1};
}

SyncCoverGraphResult
SyncCoverGraph::addCompletionDominance(SyncCoverNodeId source,
                                       SyncCoverNodeId completionNode) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, completionNode};
  }
  const bool invalidNode =
      source >= nodes_.size() || completionNode >= nodes_.size();
  if (invalidNode) {
    return {SyncCoverGraphError::InvalidNode, completionNode};
  }
  const SyncCoverNode &sourceDescription = nodes_[source];
  SyncCoverNode &completion = nodes_[completionNode];
  const bool hasCertifiedEdge =
      std::any_of(edges_.begin(), edges_.end(), [&](const SyncCoverEdge &edge) {
        return edge.source == source && edge.target == completionNode &&
               edge.distance == 0 &&
               edge.kind == SyncCoverEdgeKind::CertifiedCompletionFrontier;
      });
  const bool invalid =
      source == completionNode ||
      sourceDescription.resource != completion.resource ||
      sourceDescription.scope != completion.scope ||
      sourceDescription.order >= completion.order ||
      sourceDescription.guard.literals != completion.guard.literals ||
      !completion.completionSignalCoversIssuedPrefix || !hasCertifiedEdge;
  if (invalid) {
    return {SyncCoverGraphError::InvalidOrder, completionNode};
  }
  const auto position =
      std::lower_bound(completion.completionDominatedSources.begin(),
                       completion.completionDominatedSources.end(), source);
  const bool missing =
      position == completion.completionDominatedSources.end() ||
      *position != source;
  if (missing) {
    completion.completionDominatedSources.insert(position, source);
  }
  return {SyncCoverGraphError::None, completionNode};
}

SyncCoverGraphError
SyncCoverGraph::canonicalizeCompletionEdge(SyncCoverEdge &edge) const {
  if (edge.kind != SyncCoverEdgeKind::CompletionSupply) {
    return SyncCoverGraphError::InvalidEdgeKind;
  }
  const bool invalidNode =
      edge.source >= nodes_.size() || edge.target >= nodes_.size();
  if (invalidNode) {
    return SyncCoverGraphError::InvalidNode;
  }
  const bool invalidScope =
      !hasValidScope(edge.scope) ||
      !scopeContains(edge.scope, nodes_[edge.source].scope) ||
      !scopeContains(edge.scope, nodes_[edge.target].scope);
  if (invalidScope) {
    return SyncCoverGraphError::InvalidScope;
  }
  if (edge.distance != 0 && (edge.scope == 0 || !scopes_[edge.scope].isLoop)) {
    return SyncCoverGraphError::InvalidDistance;
  }
  if (edge.distance == 0 && edge.source == edge.target) {
    return SyncCoverGraphError::ZeroDistanceSelfEdge;
  }
  if (edge.distance == 0 &&
      nodes_[edge.source].order >= nodes_[edge.target].order) {
    return SyncCoverGraphError::InvalidOrder;
  }
  return completeEndpointGuards(edge.source, edge.target, edge.scope,
                                edge.distance, edge.sourceGuard,
                                edge.targetGuard);
}

SyncCoverGraphResult SyncCoverGraph::rebuildRegionInterfaces() {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, std::nullopt};
  }
  constexpr std::size_t kMaximumRegionPorts = 1U << 20;
  regionInterfacesBuilt_ = false;
  const std::size_t noOrder = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> firstOrder(regions_.size(), noOrder);
  std::vector<std::set<std::uint32_t>> resources(regions_.size());
  for (SyncCoverRegion &region : regions_) {
    region.elements.clear();
    region.ports.clear();
  }
  regionPorts_.clear();

  for (const SyncCoverNode &node : nodes_) {
    if (!hasValidRegion(node.region)) {
      return {SyncCoverGraphError::InvalidRegion, node.id};
    }
    firstOrder[node.region] = std::min(firstOrder[node.region], node.order);
    resources[node.region].insert(node.resource);
    regions_[node.region].elements.push_back(
        {SyncCoverRegionElementKind::Node, node.id});
  }
  for (std::size_t region = regions_.size(); region-- > 1;) {
    const SyncCoverRegionId parent = regions_[region].parent;
    if (!hasValidRegion(parent)) {
      return {SyncCoverGraphError::InvalidRegion, region};
    }
    firstOrder[parent] = std::min(firstOrder[parent], firstOrder[region]);
    resources[parent].insert(resources[region].begin(),
                             resources[region].end());
    regions_[parent].elements.push_back(
        {SyncCoverRegionElementKind::ChildRegion, region});
  }
  for (SyncCoverRegion &region : regions_) {
    const auto elementOrder = [&](const SyncCoverRegionElement &element) {
      return element.kind == SyncCoverRegionElementKind::Node
                 ? nodes_[element.value].order
                 : firstOrder[element.value];
    };
    std::sort(
        region.elements.begin(), region.elements.end(),
        [&](const SyncCoverRegionElement &left,
            const SyncCoverRegionElement &right) {
          const auto leftKey = std::make_tuple(
              elementOrder(left), static_cast<unsigned>(left.kind), left.value);
          const auto rightKey =
              std::make_tuple(elementOrder(right),
                              static_cast<unsigned>(right.kind), right.value);
          return leftKey < rightKey;
        });
  }

  const auto addPort = [&](SyncCoverRegionId region,
                           SyncCoverRegionPortKind kind, std::uint32_t resource,
                           std::optional<SyncCoverNodeId> node,
                           std::optional<SyncCoverDemandId> demand) {
    const bool portLimitReached = regionPorts_.size() >= kMaximumRegionPorts;
    if (portLimitReached) {
      return false;
    }
    const SyncCoverRegionPortId id = regionPorts_.size();
    regionPorts_.push_back({id, region, kind, resource, node, demand});
    regions_[region].ports.push_back(id);
    return true;
  };
  for (SyncCoverRegionId region = 0; region < regions_.size(); ++region) {
    for (std::uint32_t resource : resources[region]) {
      if (!addPort(region, SyncCoverRegionPortKind::Entry, resource,
                   std::nullopt, std::nullopt) ||
          !addPort(region, SyncCoverRegionPortKind::Exit, resource,
                   std::nullopt, std::nullopt)) {
        return {SyncCoverGraphError::InvalidRegionPort, regionPorts_.size()};
      }
    }
  }
  const auto exposeEndpoint =
      [&](SyncCoverDemandId demandId, SyncCoverNodeId endpoint,
          SyncCoverRegionPortKind kind, SyncCoverRegionId owner) {
        SyncCoverRegionId current = nodes_[endpoint].region;
        while (current != owner) {
          const bool invalidExposure =
              current == 0 || !regionContains(owner, current) ||
              !addPort(current, kind, nodes_[endpoint].resource, endpoint,
                       demandId);
          if (invalidExposure) {
            return false;
          }
          current = regions_[current].parent;
        }
        return true;
      };
  for (SyncCoverDemandId demandId = 0; demandId < demands_.size(); ++demandId) {
    const SyncCoverDemand &demand = demands_[demandId];
    if (!exposeEndpoint(demandId, demand.source,
                        SyncCoverRegionPortKind::DemandSource,
                        demand.ownerRegion) ||
        !exposeEndpoint(demandId, demand.target,
                        SyncCoverRegionPortKind::DemandTarget,
                        demand.ownerRegion)) {
      return {SyncCoverGraphError::InvalidRegionPort, demandId};
    }
  }
  regionInterfacesBuilt_ = true;
  return {SyncCoverGraphError::None, std::nullopt};
}

SyncCoverGraphResult SyncCoverGraph::freezeStructure() {
  SyncCoverGraphResult result = rebuildRegionInterfaces();
  if (!result) {
    return result;
  }
  result = validate();
  if (!result) {
    return result;
  }
  structureFrozen_ = true;
  storageAccessIds_.clear();
  storageWitnessIds_.clear();
  edgeIds_.clear();
  demandIds_.clear();
  return {SyncCoverGraphError::None, std::nullopt};
}
