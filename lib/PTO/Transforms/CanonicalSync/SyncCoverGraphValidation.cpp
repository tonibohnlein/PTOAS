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

namespace {

constexpr std::size_t kMaximumBasicOwnershipLanes = 8;

template <typename T> bool isSortedUnique(const std::vector<T> &values) {
  return std::is_sorted(values.begin(), values.end()) &&
         std::adjacent_find(values.begin(), values.end()) == values.end();
}

bool ownershipIntervalsOverlap(const SyncCoverStorageInterval &first,
                               const SyncCoverStorageInterval &second) {
  return first.begin < second.end && second.begin < first.end;
}

bool hasValidOwnershipKind(SyncCoverBasicOwnershipKind kind) {
  switch (kind) {
  case SyncCoverBasicOwnershipKind::L0Operand:
  case SyncCoverBasicOwnershipKind::L1Tile:
  case SyncCoverBasicOwnershipKind::L0Accumulator:
    return true;
  }
  return false;
}

bool hasValidOwnershipProtocol(SyncCoverBasicOwnershipProtocolKind protocol) {
  switch (protocol) {
  case SyncCoverBasicOwnershipProtocolKind::RoundTrip:
  case SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch:
    return true;
  }
  return false;
}

bool hasValidRegionKind(SyncCoverRegionKind kind) {
  switch (kind) {
  case SyncCoverRegionKind::Function:
  case SyncCoverRegionKind::Sequence:
  case SyncCoverRegionKind::Choice:
  case SyncCoverRegionKind::Alternative:
  case SyncCoverRegionKind::Loop:
  case SyncCoverRegionKind::Transparent:
    return true;
  }
  return false;
}

bool hasValidRegionCardinality(SyncCoverRegionCardinality cardinality) {
  switch (cardinality) {
  case SyncCoverRegionCardinality::ExactlyOnce:
  case SyncCoverRegionCardinality::ZeroOrOne:
  case SyncCoverRegionCardinality::ZeroOrMore:
  case SyncCoverRegionCardinality::OneOrMore:
    return true;
  }
  return false;
}

bool ownershipRoleMatches(SyncCoverBasicOwnershipKind kind,
                          SyncCoverStorageDomainRole role) {
  switch (kind) {
  case SyncCoverBasicOwnershipKind::L0Operand:
    return role == SyncCoverStorageDomainRole::L0Left ||
           role == SyncCoverStorageDomainRole::L0Right;
  case SyncCoverBasicOwnershipKind::L1Tile:
    return role == SyncCoverStorageDomainRole::L1Tile;
  case SyncCoverBasicOwnershipKind::L0Accumulator:
    return role == SyncCoverStorageDomainRole::Accumulator;
  }
  return false;
}

bool anchorIsWithinScope(const SyncCoverGraph &graph,
                         const SyncCoverAnchor &anchor,
                         SyncCoverScopeId scope) {
  switch (anchor.kind) {
  case SyncCoverAnchorKind::BeforeNode:
  case SyncCoverAnchorKind::AfterNode:
    return anchor.node < graph.getNodes().size() &&
           graph.scopeContains(scope, graph.getNodes()[anchor.node].scope);
  case SyncCoverAnchorKind::ControlEntry:
  case SyncCoverAnchorKind::ControlExit:
    return anchor.node < graph.getControls().size() &&
           anchor.scope == graph.getControls()[anchor.node].scope &&
           graph.scopeContains(scope, anchor.scope);
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit:
  case SyncCoverAnchorKind::LoopBodyEntry:
  case SyncCoverAnchorKind::LoopBodyExit:
  case SyncCoverAnchorKind::TimelinePoint:
    return anchor.scope < graph.getScopes().size() &&
           graph.scopeContains(scope, anchor.scope);
  }
  return false;
}

} // namespace

SyncCoverGraphResult SyncCoverGraph::validate() const {
  for (SyncCoverGraphResult result :
       {validateScopesControlsAndNodes(), validateRegions(), validateDemands(),
        validateEdges(), validateStorage(), validateCompletionCutFacts(),
        validateTargetCompletionCertificates(),
        validateBasicOwnershipCertificates()}) {
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
        std::any_of(
            sources.begin(), sources.end(), [&](SyncCoverNodeId source) {
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
  for (const auto &[source, target] : crossResourceTargetedBarrierPairs_) {
    if (source == target || !supportsBlockingTargetedBarrier(source)) {
      return {SyncCoverGraphError::InvalidCompletionTargets, std::nullopt};
    }
  }
  return {SyncCoverGraphError::None, std::nullopt};
}

SyncCoverGraphResult SyncCoverGraph::validateScopesControlsAndNodes() const {
  for (std::size_t index = 0; index < scopes_.size(); ++index) {
    const SyncCoverScope &scope = scopes_[index];
    const bool invalidScope =
        scope.id != index || !hasValidScope(scope.parent) ||
        !hasValidRegion(scope.region) || (index != 0 && scope.parent == index);
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
        !hasValidScope(control.scope) || !hasValidRegion(control.region) ||
        (control.region != 0 &&
         (regions_[control.region].kind != SyncCoverRegionKind::Choice ||
          regions_[control.region].control != index))) {
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
    if (control.successorRelation) {
      const SyncCoverControlSuccessorRelation &relation =
          *control.successorRelation;
      const bool invalidRelation =
          relation.loopScope == 0 || !hasValidScope(relation.loopScope) ||
          !scopes_[relation.loopScope].isLoop ||
          getNearestEnclosingLoop(control.scope) != relation.loopScope ||
          relation.hasSuccessorAlternative >= control.alternatives;
      if (invalidRelation) {
        return {SyncCoverGraphError::InvalidControl, index};
      }
    }
    if (control.firstIterationRelation) {
      const SyncCoverControlFirstIterationRelation &relation =
          *control.firstIterationRelation;
      const bool invalidRelation =
          relation.loopScope == 0 || !hasValidScope(relation.loopScope) ||
          !scopes_[relation.loopScope].isLoop ||
          getNearestEnclosingLoop(control.scope) != relation.loopScope ||
          relation.firstIterationAlternative >= control.alternatives;
      if (invalidRelation) {
        return {SyncCoverGraphError::InvalidControl, index};
      }
    }
  }
  std::map<std::size_t, SyncCoverNodeId> firstPhysicalNodes;
  std::map<std::size_t, std::set<int>> physicalOperationPhases;
  std::map<std::size_t, std::set<unsigned>> physicalOperationResults;
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    const SyncCoverNode &node = nodes_[index];
    const bool invalidNodeOwner =
        node.id != index || !hasValidScope(node.scope) ||
        !hasValidRegion(node.region) ||
        (node.region != 0 && regions_[node.region].scope != node.scope);
    if (invalidNodeOwner) {
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
    if (node.macroPhase < -1 ||
        !std::is_sorted(node.completedResults.begin(),
                        node.completedResults.end()) ||
        std::adjacent_find(node.completedResults.begin(),
                           node.completedResults.end()) !=
            node.completedResults.end()) {
      return {SyncCoverGraphError::InvalidNode, index};
    }
    const auto [firstNode, insertedPhysicalOperation] =
        firstPhysicalNodes.try_emplace(node.physicalOperation, index);
    if (!insertedPhysicalOperation &&
        (node.macroPhase < 0 || nodes_[firstNode->second].macroPhase < 0 ||
         node.physicalAnchor != nodes_[firstNode->second].physicalAnchor)) {
      return {SyncCoverGraphError::InvalidNode, index};
    }
    if (node.macroPhase >= 0 && !physicalOperationPhases[node.physicalOperation]
                                     .insert(node.macroPhase)
                                     .second) {
      return {SyncCoverGraphError::InvalidNode, index};
    }
    for (unsigned result : node.completedResults) {
      if (!physicalOperationResults[node.physicalOperation]
               .insert(result)
               .second) {
        return {SyncCoverGraphError::InvalidNode, index};
      }
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

SyncCoverGraphResult SyncCoverGraph::validateRegions() const {
  const bool invalidRoot =
      regions_.empty() || regions_[0].kind != SyncCoverRegionKind::Function ||
      regions_[0].parent != 0 || regions_[0].scope != 0 ||
      regions_[0].cardinality != SyncCoverRegionCardinality::ExactlyOnce;
  if (invalidRoot) {
    return {SyncCoverGraphError::InvalidRegion, 0};
  }
  std::vector<unsigned> nodeOwners(nodes_.size(), 0);
  std::vector<unsigned> childOwners(regions_.size(), 0);
  for (std::size_t index = 0; index < regions_.size(); ++index) {
    const SyncCoverRegion &region = regions_[index];
    const bool invalidIdentity =
        region.id != index || !hasValidRegion(region.parent) ||
        !hasValidScope(region.scope) || !hasValidRegionKind(region.kind) ||
        !hasValidRegionCardinality(region.cardinality) ||
        (index != 0 && region.parent == index);
    if (invalidIdentity) {
      return {SyncCoverGraphError::InvalidRegion, index};
    }
    const bool isChoice = region.kind == SyncCoverRegionKind::Choice;
    const bool isAlternative = region.kind == SyncCoverRegionKind::Alternative;
    const bool invalidControl =
        (isChoice && (!region.control || region.alternative)) ||
        (isAlternative && (!region.control || !region.alternative)) ||
        (!isChoice && !isAlternative &&
         (region.control || region.alternative)) ||
        ((isChoice || isAlternative) &&
         (*region.control >= controls_.size() ||
          controls_[*region.control].scope !=
              (isChoice ? region.scope : regions_[region.parent].scope))) ||
        (isChoice && controls_[*region.control].region != index) ||
        (isAlternative &&
         *region.alternative >= controls_[*region.control].alternatives);
    if (invalidControl) {
      return {SyncCoverGraphError::InvalidRegion, index};
    }
    if (index != 0) {
      const SyncCoverRegionKind parentKind = regions_[region.parent].kind;
      const bool invalidParent =
          (region.kind == SyncCoverRegionKind::Function) ||
          (region.kind == SyncCoverRegionKind::Sequence &&
           parentKind != SyncCoverRegionKind::Function &&
           parentKind != SyncCoverRegionKind::Loop &&
           parentKind != SyncCoverRegionKind::Alternative &&
           parentKind != SyncCoverRegionKind::Transparent) ||
          (region.kind == SyncCoverRegionKind::Choice &&
           parentKind != SyncCoverRegionKind::Sequence) ||
          (region.kind == SyncCoverRegionKind::Alternative &&
           (parentKind != SyncCoverRegionKind::Choice ||
            region.control != regions_[region.parent].control)) ||
          (region.kind == SyncCoverRegionKind::Loop &&
           parentKind != SyncCoverRegionKind::Sequence) ||
          (region.kind == SyncCoverRegionKind::Transparent &&
           parentKind != SyncCoverRegionKind::Sequence) ||
          !scopeContains(regions_[region.parent].scope, region.scope);
      if (invalidParent) {
        return {SyncCoverGraphError::InvalidRegion, index};
      }
    }
    SyncCoverGuard guard = region.guard;
    const SyncCoverGraphError guardError =
        normalizeAndValidateGuard(guard, region.scope);
    if (guardError != SyncCoverGraphError::None ||
        guard.literals != region.guard.literals ||
        !std::includes(region.guard.literals.begin(),
                       region.guard.literals.end(),
                       scopes_[region.scope].guard.literals.begin(),
                       scopes_[region.scope].guard.literals.end()) ||
        (index != 0 &&
         !std::includes(region.guard.literals.begin(),
                        region.guard.literals.end(),
                        regions_[region.parent].guard.literals.begin(),
                        regions_[region.parent].guard.literals.end()))) {
      return {guardError == SyncCoverGraphError::None
                  ? SyncCoverGraphError::InvalidGuard
                  : guardError,
              index};
    }
    if (regionInterfacesBuilt_) {
      std::set<std::pair<SyncCoverRegionElementKind, std::size_t>> seenElements;
      for (const SyncCoverRegionElement &element : region.elements) {
        if (!seenElements.emplace(element.kind, element.value).second) {
          return {SyncCoverGraphError::InvalidRegion, index};
        }
        if (element.kind == SyncCoverRegionElementKind::Node) {
          const bool invalidNode = element.value >= nodes_.size() ||
                                   nodes_[element.value].region != index;
          if (invalidNode) {
            return {SyncCoverGraphError::InvalidRegion, index};
          }
          ++nodeOwners[element.value];
        } else if (element.kind == SyncCoverRegionElementKind::ChildRegion) {
          const bool invalidChild = element.value == 0 ||
                                    element.value >= regions_.size() ||
                                    regions_[element.value].parent != index;
          if (invalidChild) {
            return {SyncCoverGraphError::InvalidRegion, index};
          }
          ++childOwners[element.value];
        } else {
          return {SyncCoverGraphError::InvalidRegion, index};
        }
      }
      if (!std::is_sorted(region.ports.begin(), region.ports.end())) {
        return {SyncCoverGraphError::InvalidRegionPort, index};
      }
      for (SyncCoverRegionPortId portId : region.ports) {
        const bool invalidPort = portId >= regionPorts_.size() ||
                                 regionPorts_[portId].region != index;
        if (invalidPort) {
          return {SyncCoverGraphError::InvalidRegionPort, index};
        }
      }
    }
  }
  if (regionInterfacesBuilt_) {
    if (std::any_of(nodeOwners.begin(), nodeOwners.end(),
                    [](unsigned owners) { return owners != 1; }) ||
        std::any_of(childOwners.begin() + 1, childOwners.end(),
                    [](unsigned owners) { return owners != 1; })) {
      return {SyncCoverGraphError::InvalidRegion, std::nullopt};
    }
    for (std::size_t index = 0; index < regionPorts_.size(); ++index) {
      const SyncCoverRegionPort &port = regionPorts_[index];
      if (port.id != index || !hasValidRegion(port.region)) {
        return {SyncCoverGraphError::InvalidRegionPort, index};
      }
      const bool endpointPort =
          port.kind == SyncCoverRegionPortKind::DemandSource ||
          port.kind == SyncCoverRegionPortKind::DemandTarget;
      if (!endpointPort) {
        if (port.node || port.demand) {
          return {SyncCoverGraphError::InvalidRegionPort, index};
        }
        continue;
      }
      const bool invalidEndpoint =
          !port.node || !port.demand || *port.node >= nodes_.size() ||
          *port.demand >= demands_.size() ||
          !regionContains(port.region, nodes_[*port.node].region) ||
          port.region == demands_[*port.demand].ownerRegion ||
          port.resource != nodes_[*port.node].resource;
      if (invalidEndpoint) {
        return {SyncCoverGraphError::InvalidRegionPort, index};
      }
      const SyncCoverDemand &demand = demands_[*port.demand];
      const SyncCoverNodeId expected =
          port.kind == SyncCoverRegionPortKind::DemandSource ? demand.source
                                                             : demand.target;
      if (*port.node != expected ||
          !regionContains(demand.ownerRegion, port.region)) {
        return {SyncCoverGraphError::InvalidRegionPort, index};
      }
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
    const std::optional<SyncCoverRegionId> expectedOwner =
        demand.distance == 0
            ? getLowestCommonRegion(nodes_[demand.source].region,
                                    nodes_[demand.target].region)
            : std::optional<SyncCoverRegionId>(scopes_[demand.scope].region);
    if (!expectedOwner || demand.ownerRegion != *expectedOwner ||
        !regionContains(demand.ownerRegion, nodes_[demand.source].region) ||
        !regionContains(demand.ownerRegion, nodes_[demand.target].region)) {
      return {SyncCoverGraphError::InvalidRegion, index};
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
    if (!isValidOrderingRequirements(demand.orderingRequirements)) {
      return {SyncCoverGraphError::InvalidOrderingRequirement, index};
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
    if (edge.kind == SyncCoverEdgeKind::CompletionSupply &&
        !isValidOrderingRequirements(edge.suppliedRequirements)) {
      return {SyncCoverGraphError::InvalidOrderingRequirement, index};
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
    const bool validRole = role == SyncCoverStorageDomainRole::Unspecified ||
                           role == SyncCoverStorageDomainRole::Other ||
                           role == SyncCoverStorageDomainRole::L1Tile ||
                           role == SyncCoverStorageDomainRole::L0Left ||
                           role == SyncCoverStorageDomainRole::L0Right ||
                           role == SyncCoverStorageDomainRole::Accumulator;
    if (storageDomains_[index].id != index || !validRole) {
      return {SyncCoverGraphError::InvalidStorageDomain, index};
    }
  }
  for (std::size_t index = 0; index < storageAccesses_.size(); ++index) {
    const SyncCoverStorageAccess &access = storageAccesses_[index];
    const bool invalidAccess =
        access.id != index || access.node >= nodes_.size() ||
        access.domain >= storageDomains_.size() ||
        access.extent.begin >= access.extent.end ||
        !isValidAccessMode(access.mode) || !isValidAccessPath(access.path);
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

SyncCoverGraphResult SyncCoverGraph::validateCompletionCutFacts() const {
  for (std::size_t index = 0; index < completionCutFacts_.size(); ++index) {
    const SyncCoverGraphResult result = validateCompletionCutFact(index);
    if (!result) {
      return result;
    }
  }
  return {SyncCoverGraphError::None, std::nullopt};
}

SyncCoverGraphResult SyncCoverGraph::validateCompletionCutFact(
    SyncCoverCompletionCutFactId index) const {
  if (index >= completionCutFacts_.size()) {
    return {SyncCoverGraphError::InvalidCompletionCutFact, index};
  }
  const SyncCoverCompletionCutFact &fact = completionCutFacts_[index];
  const bool invalidHeader =
      fact.id != index || fact.completionNode >= nodes_.size() ||
      fact.sourceResource == fact.targetResource ||
      fact.storageDomains.empty() || fact.demands.empty() ||
      !isSortedUnique(fact.storageDomains) ||
      std::any_of(fact.storageDomains.begin(), fact.storageDomains.end(),
                  [&](SyncCoverStorageDomainId domain) {
                    return domain >= storageDomains_.size();
                  }) ||
      !isSortedUnique(fact.demands) ||
      std::any_of(
          fact.demands.begin(), fact.demands.end(),
          [&](SyncCoverDemandId demand) { return demand >= demands_.size(); });
  if (invalidHeader) {
    return {SyncCoverGraphError::InvalidCompletionCutFact, index};
  }
  const SyncCoverNode &completion = nodes_[fact.completionNode];
  if (completion.resource != fact.sourceResource) {
    return {SyncCoverGraphError::InvalidCompletionCutFact, index};
  }
  for (SyncCoverDemandId demandId : fact.demands) {
    const SyncCoverDemand &demand = demands_[demandId];
    const SyncCoverNode &source = nodes_[demand.source];
    const SyncCoverNode &target = nodes_[demand.target];
    const SyncCoverNode &physicalSource = nodes_[source.physicalExit];
    const std::optional<SyncCoverTimelinePosition> completionPosition =
        resolveSyncCoverAnchor(
            *this, {SyncCoverAnchorKind::AfterNode, fact.completionNode, 0, 0});
    const std::optional<SyncCoverTimelinePosition> acquisitionPosition =
        resolveSyncCoverAnchor(
            *this, {SyncCoverAnchorKind::BeforeNode, demand.target, 0, 0});
    const bool invalidDemand =
        demand.distance != 0 ||
        std::find(demand.provenanceKinds.begin(), demand.provenanceKinds.end(),
                  SyncCoverDemandKind::MemoryRAW) ==
            demand.provenanceKinds.end() ||
        source.resource != fact.sourceResource ||
        target.resource != fact.targetResource ||
        source.physicalExit != fact.completionNode ||
        source.scope != completion.scope ||
        physicalSource.scope != completion.scope ||
        source.guard.literals != completion.guard.literals ||
        physicalSource.guard.literals != completion.guard.literals ||
        source.order > physicalSource.order || !completionPosition ||
        !acquisitionPosition || *completionPosition >= *acquisitionPosition;
    if (invalidDemand || demand.storageWitnesses.empty()) {
      return {SyncCoverGraphError::InvalidCompletionCutFact, index};
    }
    const bool hasExactRawWitness = std::any_of(
        demand.storageWitnesses.begin(), demand.storageWitnesses.end(),
        [&](SyncCoverStorageWitnessId witnessId) {
          if (witnessId >= storageWitnesses_.size()) {
            return false;
          }
          const SyncCoverStorageWitness &witness = storageWitnesses_[witnessId];
          const SyncCoverStorageAccess &sourceAccess =
              storageAccesses_[witness.sourceAccess];
          const SyncCoverStorageAccess &targetAccess =
              storageAccesses_[witness.targetAccess];
          return sourceAccess.node == demand.source &&
                 targetAccess.node == demand.target &&
                 sourceAccess.domain == targetAccess.domain &&
                 std::binary_search(fact.storageDomains.begin(),
                                    fact.storageDomains.end(),
                                    sourceAccess.domain) &&
                 sourceAccess.exactPhysical && targetAccess.exactPhysical &&
                 syncCoverStorageModeWrites(sourceAccess.mode) &&
                 syncCoverStorageModeReads(targetAccess.mode);
        });
    if (!hasExactRawWitness) {
      return {SyncCoverGraphError::InvalidCompletionCutFact, index};
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
        certificate.id != index ||
        certificate.completionNode >= nodes_.size() ||
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
        certificate.kind == SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix &&
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
    if ((!mte1L0Ready && !mToFix) || completion.order >= physicalTarget.order ||
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
          std::find(
              demand.provenanceKinds.begin(), demand.provenanceKinds.end(),
              SyncCoverDemandKind::MemoryRAW) == demand.provenanceKinds.end() ||
          source.resource != certificate.sourceResource ||
          target.resource != certificate.targetResource ||
          target.physicalAnchor != certificate.target ||
          source.scope != completion.scope ||
          target.scope != completion.scope ||
          physicalSource.scope != completion.scope ||
          source.guard.literals != completion.guard.literals ||
          target.guard.literals != completion.guard.literals ||
          physicalSource.guard.literals != completion.guard.literals ||
          source.order > physicalSource.order ||
          physicalSource.order > completion.order;
      if (demand.storageWitnesses.empty() || invalidDemand) {
        return {SyncCoverGraphError::InvalidTargetCompletionCertificate, index};
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
        return {SyncCoverGraphError::InvalidTargetCompletionCertificate, index};
      }
      if (certificate.kind ==
              SyncCoverTargetCompletionKind::MToFixAccumulatorBoundary &&
          source.physicalExit != certificate.completionNode) {
        return {SyncCoverGraphError::InvalidTargetCompletionCertificate, index};
      }
    }
    if (!completionNamesCertifiedSource) {
      return {SyncCoverGraphError::InvalidTargetCompletionCertificate, index};
    }
  }
  return {SyncCoverGraphError::None, std::nullopt};
}

SyncCoverGraphResult
SyncCoverGraph::validateBasicOwnershipCertificates() const {
  for (std::size_t index = 0; index < basicOwnershipCertificates_.size();
       ++index) {
    const SyncCoverBasicOwnershipCertificate &certificate =
        basicOwnershipCertificates_[index];
    const std::size_t minimumLanes =
        certificate.kind == SyncCoverBasicOwnershipKind::L0Accumulator ? 1 : 2;
    const bool invalidHeader =
        certificate.id != index || !hasValidOwnershipKind(certificate.kind) ||
        !hasValidOwnershipProtocol(certificate.protocol) ||
        certificate.loopScope == 0 || !hasValidScope(certificate.loopScope) ||
        !scopes_[certificate.loopScope].isLoop ||
        certificate.producerResource == certificate.consumerResource ||
        certificate.lanes.size() < minimumLanes ||
        certificate.lanes.size() > kMaximumBasicOwnershipLanes ||
        certificate.paths.empty() ||
        !isSortedUnique(certificate.initialProducers) ||
        !isSortedUnique(certificate.initiallyFreeLanes);
    if (invalidHeader) {
      return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
    }

    std::vector<std::optional<std::size_t>> accessLanes(
        storageAccesses_.size());
    std::vector<std::pair<SyncCoverStorageDomainId, SyncCoverStorageInterval>>
        slots;
    for (std::size_t laneIndex = 0; laneIndex < certificate.lanes.size();
         ++laneIndex) {
      const SyncCoverBasicOwnershipLane &lane = certificate.lanes[laneIndex];
      std::size_t leftSlots = 0;
      std::size_t rightSlots = 0;
      if (lane.id != laneIndex || lane.slots.empty()) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
      for (const SyncCoverBasicOwnershipSlot &slot : lane.slots) {
        const bool invalidSlot =
            slot.domain >= storageDomains_.size() ||
            slot.extent.begin >= slot.extent.end || slot.accesses.empty() ||
            !isSortedUnique(slot.accesses) ||
            (slot.domain < storageDomains_.size() &&
             !ownershipRoleMatches(certificate.kind,
                                   storageDomains_[slot.domain].role));
        if (invalidSlot) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        leftSlots += storageDomains_[slot.domain].role ==
                     SyncCoverStorageDomainRole::L0Left;
        rightSlots += storageDomains_[slot.domain].role ==
                      SyncCoverStorageDomainRole::L0Right;
        for (const auto &[oldDomain, oldExtent] : slots) {
          if (oldDomain == slot.domain &&
              ownershipIntervalsOverlap(oldExtent, slot.extent)) {
            return {SyncCoverGraphError::InvalidBasicOwnershipCertificate,
                    index};
          }
        }
        slots.emplace_back(slot.domain, slot.extent);
        for (SyncCoverStorageAccessId accessId : slot.accesses) {
          if (accessId >= storageAccesses_.size() || accessLanes[accessId]) {
            return {SyncCoverGraphError::InvalidBasicOwnershipCertificate,
                    index};
          }
          const SyncCoverStorageAccess &access = storageAccesses_[accessId];
          const bool relevantOccurrence =
              scopeContains(certificate.loopScope, nodes_[access.node].scope) ||
              std::binary_search(certificate.initialProducers.begin(),
                                 certificate.initialProducers.end(),
                                 access.node);
          if (!relevantOccurrence || !access.exactPhysical ||
              access.domain != slot.domain ||
              access.extent.begin != slot.extent.begin ||
              access.extent.end != slot.extent.end) {
            return {SyncCoverGraphError::InvalidBasicOwnershipCertificate,
                    index};
          }
          accessLanes[accessId] = laneIndex;
        }
        for (const SyncCoverStorageAccess &access : storageAccesses_) {
          const bool relevantOccurrence =
              scopeContains(certificate.loopScope, nodes_[access.node].scope) ||
              std::binary_search(certificate.initialProducers.begin(),
                                 certificate.initialProducers.end(),
                                 access.node);
          if (!relevantOccurrence || access.domain != slot.domain ||
              !ownershipIntervalsOverlap(access.extent, slot.extent)) {
            continue;
          }
          const bool exactMember =
              access.exactPhysical &&
              access.extent.begin == slot.extent.begin &&
              access.extent.end == slot.extent.end &&
              std::binary_search(slot.accesses.begin(), slot.accesses.end(),
                                 access.id);
          if (!exactMember) {
            return {SyncCoverGraphError::InvalidBasicOwnershipCertificate,
                    index};
          }
        }
      }
      const bool invalidLaneSlots =
          (certificate.kind == SyncCoverBasicOwnershipKind::L0Operand &&
           (lane.slots.size() != 2 || leftSlots != 1 || rightSlots != 1)) ||
          (certificate.kind != SyncCoverBasicOwnershipKind::L0Operand &&
           lane.slots.size() != 1);
      if (invalidLaneSlots) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
    }

    const auto nodeHasLaneAccess = [&](SyncCoverNodeId node, std::size_t lane,
                                       bool writes) {
      return std::any_of(
          storageAccesses_.begin(), storageAccesses_.end(),
          [&](const SyncCoverStorageAccess &access) {
            return access.node == node && accessLanes[access.id] == lane &&
                   (writes ? syncCoverStorageModeWrites(access.mode)
                           : syncCoverStorageModeReads(access.mode));
          });
    };
    const auto hasSuccessorGuard = [&](SyncCoverNodeId node) {
      return std::any_of(
          nodes_[node].guard.literals.begin(),
          nodes_[node].guard.literals.end(),
          [&](const SyncCoverGuardLiteral &literal) {
            const SyncCoverControl &control = controls_[literal.control];
            return control.successorRelation &&
                   control.successorRelation->loopScope ==
                       certificate.loopScope &&
                   control.successorRelation->hasSuccessorAlternative ==
                       literal.alternative;
          });
    };

    std::vector<std::set<SyncCoverNodeId>> namedProducers(
        certificate.lanes.size());
    std::vector<std::set<SyncCoverNodeId>> namedConsumers(
        certificate.lanes.size());
    std::set<SyncCoverScopeId> pathScopes;
    for (const SyncCoverBasicOwnershipPath &path : certificate.paths) {
      if (!hasValidScope(path.scope) ||
          !scopeContains(certificate.loopScope, path.scope) ||
          path.uses.empty() || !pathScopes.insert(path.scope).second) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
      std::vector<bool> pathLanes(certificate.lanes.size(), false);
      for (const SyncCoverBasicOwnershipUse &use : path.uses) {
        const bool invalidUseHeader =
            use.lane >= certificate.lanes.size() ||
            use.producerLane >= certificate.lanes.size() ||
            use.producers.empty() || use.consumers.empty() ||
            !isSortedUnique(use.producers) || !isSortedUnique(use.consumers) ||
            !anchorIsWithinScope(*this, use.writeAcquireAnchor, path.scope) ||
            !anchorIsWithinScope(*this, use.readyAnchor, path.scope) ||
            !anchorIsWithinScope(*this, use.readAcquireAnchor, path.scope) ||
            !anchorIsWithinScope(*this, use.releaseAnchor, path.scope);
        if (invalidUseHeader) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        const std::optional<SyncCoverTimelinePosition> writeAcquire =
            resolveSyncCoverAnchor(*this, use.writeAcquireAnchor);
        const std::optional<SyncCoverTimelinePosition> ready =
            resolveSyncCoverAnchor(*this, use.readyAnchor);
        const std::optional<SyncCoverTimelinePosition> readAcquire =
            resolveSyncCoverAnchor(*this, use.readAcquireAnchor);
        const std::optional<SyncCoverTimelinePosition> release =
            resolveSyncCoverAnchor(*this, use.releaseAnchor);
        if (!writeAcquire || !ready || !readAcquire || !release) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        const bool alternatingLifecycle =
            certificate.protocol ==
            SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch;
        const bool validAnchorOrder =
            alternatingLifecycle
                ? *readAcquire <= *release && *writeAcquire <= *ready
                : *writeAcquire <= *ready && *ready < *readAcquire &&
                      *readAcquire <= *release;
        if (!validAnchorOrder) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        SyncCoverTimelinePosition firstProducer =
            std::numeric_limits<SyncCoverTimelinePosition>::max();
        SyncCoverTimelinePosition lastProducer = 0;
        for (SyncCoverNodeId producer : use.producers) {
          if (producer >= nodes_.size() ||
              nodes_[producer].resource != certificate.producerResource ||
              !scopeContains(path.scope, nodes_[producer].scope) ||
              !nodeHasLaneAccess(producer, use.producerLane, true) ||
              (certificate.protocol ==
                   SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch &&
               !hasSuccessorGuard(producer))) {
            return {SyncCoverGraphError::InvalidBasicOwnershipCertificate,
                    index};
          }
          const auto before = resolveSyncCoverAnchor(
              *this, {SyncCoverAnchorKind::BeforeNode, producer, 0, 0});
          const auto after = resolveSyncCoverAnchor(
              *this, {SyncCoverAnchorKind::AfterNode, producer, 0, 0});
          if (!before || !after) {
            return {SyncCoverGraphError::InvalidBasicOwnershipCertificate,
                    index};
          }
          firstProducer = std::min(firstProducer, *before);
          lastProducer = std::max(lastProducer, *after);
          namedProducers[use.producerLane].insert(producer);
        }
        SyncCoverTimelinePosition firstConsumer =
            std::numeric_limits<SyncCoverTimelinePosition>::max();
        SyncCoverTimelinePosition lastConsumer = 0;
        SyncCoverNodeId lastConsumerNode = 0;
        for (SyncCoverNodeId consumer : use.consumers) {
          if (consumer >= nodes_.size() ||
              nodes_[consumer].resource != certificate.consumerResource ||
              !scopeContains(path.scope, nodes_[consumer].scope) ||
              !nodeHasLaneAccess(consumer, use.lane, false)) {
            return {SyncCoverGraphError::InvalidBasicOwnershipCertificate,
                    index};
          }
          const auto before = resolveSyncCoverAnchor(
              *this, {SyncCoverAnchorKind::BeforeNode, consumer, 0, 0});
          const auto after = resolveSyncCoverAnchor(
              *this, {SyncCoverAnchorKind::AfterNode, consumer, 0, 0});
          if (!before || !after) {
            return {SyncCoverGraphError::InvalidBasicOwnershipCertificate,
                    index};
          }
          firstConsumer = std::min(firstConsumer, *before);
          if (*after >= lastConsumer) {
            lastConsumer = *after;
            lastConsumerNode = consumer;
          }
          namedConsumers[use.lane].insert(consumer);
        }
        const bool sharedReleaseNeedsPrefixCompletion =
            use.consumers.size() > 1 &&
            use.releaseAnchor.kind == SyncCoverAnchorKind::AfterNode;
        // A graph-certified L1 ownership cycle is the narrow target contract
        // that permits one MTE1 release set after the final exact-slot
        // consumer to release the whole named consumer prefix. Do not promote
        // this fact to generic MTE1 issue/completion ordering.
        const bool validSharedOwnershipRelease =
            certificate.kind == SyncCoverBasicOwnershipKind::L1Tile &&
            targetCompletionResources_ &&
            certificate.consumerResource == targetCompletionResources_->mte1 &&
            use.releaseAnchor.node == lastConsumerNode;
        if (sharedReleaseNeedsPrefixCompletion &&
            !validSharedOwnershipRelease) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        if (*writeAcquire > firstProducer || *ready < lastProducer ||
            *readAcquire > firstConsumer || *release < lastConsumer) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        pathLanes[use.lane] = true;
      }
      const bool missingRoundTripLane =
          certificate.protocol ==
              SyncCoverBasicOwnershipProtocolKind::RoundTrip &&
          std::any_of(pathLanes.begin(), pathLanes.end(),
                      [](bool present) { return !present; });
      if (missingRoundTripLane) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
    }

    const bool isAlternating =
        certificate.protocol ==
        SyncCoverBasicOwnershipProtocolKind::AlternatingPrefetch;
    if (!isAlternating) {
      const bool invalidRoundTripState =
          certificate.periodicControl ||
          !certificate.initialProducers.empty() ||
          !certificate.initiallyFreeLanes.empty() ||
          std::any_of(certificate.paths.begin(), certificate.paths.end(),
                      [](const SyncCoverBasicOwnershipPath &path) {
                        return std::any_of(
                            path.uses.begin(), path.uses.end(),
                            [](const SyncCoverBasicOwnershipUse &use) {
                              return use.lane != use.producerLane;
                            });
                      });
      if (invalidRoundTripState) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
    } else {
      const bool invalidAlternatingHeader =
          certificate.kind != SyncCoverBasicOwnershipKind::L1Tile ||
          certificate.lanes.size() != 2 || certificate.paths.size() != 2 ||
          !certificate.periodicControl ||
          *certificate.periodicControl >= controls_.size() ||
          certificate.initialProducers.empty() ||
          certificate.initialReadyLane >= certificate.lanes.size() ||
          certificate.initiallyFreeLanes.size() != 1 ||
          certificate.initiallyFreeLanes.front() ==
              certificate.initialReadyLane ||
          std::any_of(certificate.paths.begin(), certificate.paths.end(),
                      [](const SyncCoverBasicOwnershipPath &path) {
                        return path.uses.size() != 1;
                      });
      if (invalidAlternatingHeader) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
      const SyncCoverControl &periodic =
          controls_[*certificate.periodicControl];
      if (!periodic.phaseRelation ||
          periodic.phaseRelation->loopScope != certificate.loopScope ||
          periodic.phaseRelation->nextPhase.size() !=
              certificate.lanes.size() ||
          periodic.phaseRelation->initialPhase !=
              certificate.initialReadyLane) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
      std::vector<bool> seenPhases(certificate.lanes.size(), false);
      for (const SyncCoverBasicOwnershipPath &path : certificate.paths) {
        const auto literal = std::find_if(
            scopes_[path.scope].guard.literals.begin(),
            scopes_[path.scope].guard.literals.end(),
            [&](const SyncCoverGuardLiteral &candidate) {
              return candidate.control == *certificate.periodicControl;
            });
        if (literal == scopes_[path.scope].guard.literals.end()) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        const auto phase =
            std::find(periodic.phaseRelation->activeAlternative.begin(),
                      periodic.phaseRelation->activeAlternative.end(),
                      literal->alternative);
        if (phase == periodic.phaseRelation->activeAlternative.end()) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        const std::size_t phaseIndex = static_cast<std::size_t>(std::distance(
            periodic.phaseRelation->activeAlternative.begin(), phase));
        const SyncCoverBasicOwnershipUse &use = path.uses.front();
        if (phaseIndex >= seenPhases.size() || seenPhases[phaseIndex] ||
            use.lane != phaseIndex ||
            use.producerLane != periodic.phaseRelation->nextPhase[phaseIndex]) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        seenPhases[phaseIndex] = true;
      }
      if (std::any_of(seenPhases.begin(), seenPhases.end(),
                      [](bool seen) { return !seen; })) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
      const std::optional<SyncCoverTimelinePosition> initialWrite =
          resolveSyncCoverAnchor(*this, certificate.initialWriteAcquireAnchor);
      const std::optional<SyncCoverTimelinePosition> initialReady =
          resolveSyncCoverAnchor(*this, certificate.initialReadyAnchor);
      const auto bodyEntry =
          resolveSyncCoverAnchor(*this, {SyncCoverAnchorKind::LoopBodyEntry, 0,
                                         certificate.loopScope, 0});
      if (!initialWrite || !initialReady || !bodyEntry ||
          certificate.initialReadyAnchor.kind !=
              SyncCoverAnchorKind::ScopeEntry ||
          certificate.initialReadyAnchor.scope != certificate.loopScope ||
          *initialWrite > *initialReady || *initialReady != *bodyEntry) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
      SyncCoverTimelinePosition firstInitial =
          std::numeric_limits<SyncCoverTimelinePosition>::max();
      SyncCoverTimelinePosition lastInitial = 0;
      for (SyncCoverNodeId producer : certificate.initialProducers) {
        if (producer >= nodes_.size() ||
            scopeContains(certificate.loopScope, nodes_[producer].scope) ||
            nodes_[producer].resource != certificate.producerResource ||
            !nodeHasLaneAccess(producer, certificate.initialReadyLane, true)) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        const auto before = resolveSyncCoverAnchor(
            *this, {SyncCoverAnchorKind::BeforeNode, producer, 0, 0});
        const auto after = resolveSyncCoverAnchor(
            *this, {SyncCoverAnchorKind::AfterNode, producer, 0, 0});
        if (!before || !after) {
          return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
        }
        firstInitial = std::min(firstInitial, *before);
        lastInitial = std::max(lastInitial, *after);
        namedProducers[certificate.initialReadyLane].insert(producer);
      }
      if (*initialWrite > firstInitial || *initialReady < lastInitial) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
    }

    for (const SyncCoverStorageAccess &access : storageAccesses_) {
      if (!accessLanes[access.id]) {
        continue;
      }
      const std::size_t lane = *accessLanes[access.id];
      const bool producerAccess =
          nodes_[access.node].resource == certificate.producerResource &&
          syncCoverStorageModeWrites(access.mode) &&
          (certificate.kind == SyncCoverBasicOwnershipKind::L0Accumulator ||
           !syncCoverStorageModeReads(access.mode));
      const bool consumerAccess =
          nodes_[access.node].resource == certificate.consumerResource &&
          syncCoverStorageModeReads(access.mode) &&
          !syncCoverStorageModeWrites(access.mode);
      const bool namedProducer =
          producerAccess && namedProducers[lane].count(access.node) != 0;
      const bool namedConsumer =
          consumerAccess && namedConsumers[lane].count(access.node) != 0;
      if (!namedProducer && !namedConsumer) {
        return {SyncCoverGraphError::InvalidBasicOwnershipCertificate, index};
      }
    }
  }

  for (std::size_t first = 0; first < basicOwnershipCertificates_.size();
       ++first) {
    for (std::size_t second = first + 1;
         second < basicOwnershipCertificates_.size(); ++second) {
      const SyncCoverBasicOwnershipCertificate &left =
          basicOwnershipCertificates_[first];
      const SyncCoverBasicOwnershipCertificate &right =
          basicOwnershipCertificates_[second];
      const bool nested = scopeContains(left.loopScope, right.loopScope) ||
                          scopeContains(right.loopScope, left.loopScope);
      if (!nested) {
        continue;
      }
      for (const SyncCoverBasicOwnershipLane &leftLane : left.lanes) {
        for (const SyncCoverBasicOwnershipSlot &leftSlot : leftLane.slots) {
          for (const SyncCoverBasicOwnershipLane &rightLane : right.lanes) {
            for (const SyncCoverBasicOwnershipSlot &rightSlot :
                 rightLane.slots) {
              if (leftSlot.domain == rightSlot.domain &&
                  ownershipIntervalsOverlap(leftSlot.extent,
                                            rightSlot.extent)) {
                return {SyncCoverGraphError::InvalidBasicOwnershipCertificate,
                        second};
              }
            }
          }
        }
      }
    }
  }
  return {SyncCoverGraphError::None, std::nullopt};
}
