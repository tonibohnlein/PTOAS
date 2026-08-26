// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"

#include "PTO/IR/PTO.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto;

StringRef
mlir::pto::stringifyCanonicalDependencyKind(CanonicalDependencyKind kind) {
  switch (kind) {
  case CanonicalDependencyKind::SSA:
    return "ssa";
  case CanonicalDependencyKind::MemoryRAW:
    return "memory-raw";
  case CanonicalDependencyKind::MemoryWAR:
    return "memory-war";
  case CanonicalDependencyKind::MemoryWAW:
    return "memory-waw";
  case CanonicalDependencyKind::LoopCarriedSSA:
    return "loop-carried-ssa";
  }
  return "unknown";
}

StringRef
mlir::pto::stringifyCanonicalGMAliasPolicy(CanonicalGMAliasPolicy policy) {
  switch (policy) {
  case CanonicalGMAliasPolicy::MayAlias:
    return "may-alias";
  case CanonicalGMAliasPolicy::DistinctArgumentsNoAlias:
    return "distinct-args-noalias";
  case CanonicalGMAliasPolicy::AllAccessesNoAlias:
    return "all-accesses-noalias";
  }
  return "unknown";
}

StringRef mlir::pto::stringifyCanonicalSelectionMechanismKind(
    CanonicalSelectionMechanismKind kind) {
  switch (kind) {
  case CanonicalSelectionMechanismKind::Barrier:
    return "barrier";
  case CanonicalSelectionMechanismKind::EventBundle:
    return "event-bundle";
  case CanonicalSelectionMechanismKind::SlotProtocol:
    return "slot-protocol";
  }
  return "unknown";
}

StringRef
mlir::pto::stringifyCanonicalOwnershipKind(CanonicalOwnershipKind kind) {
  switch (kind) {
  case CanonicalOwnershipKind::L0Operand:
    return "l0-operand";
  case CanonicalOwnershipKind::L1Tile:
    return "l1-tile";
  case CanonicalOwnershipKind::L0Accumulator:
    return "l0-accumulator";
  }
  return "unknown";
}

namespace {

bool includesDependencies(StringRef view) {
  return view == "all" || view == "dependencies";
}

bool includesPlan(StringRef view) { return view == "all" || view == "plan"; }

bool includesEvents(StringRef view) {
  return view == "all" || view == "events";
}

bool includesOwnership(StringRef view) {
  return view == "all" || view == "ownership";
}

StringRef getOwnershipProtocolSuffix(CanonicalOwnershipProtocolKind kind) {
  switch (kind) {
  case CanonicalOwnershipProtocolKind::RoundTrip:
    return "";
  case CanonicalOwnershipProtocolKind::AlternatingPrefetch:
    return " protocol=alternating-prefetch";
  case CanonicalOwnershipProtocolKind::BoundaryGuardedRoundTrip:
    return " protocol=boundary-guarded-round-trip";
  case CanonicalOwnershipProtocolKind::HierarchicalOuterCarry:
    return " protocol=hierarchical-outer-carry";
  }
  return " protocol=unknown";
}

bool includesCovering(StringRef view) {
  return view == "all" || view == "covering";
}

StringRef stringifyCoveringResourceKind(SyncCoverResourceKind kind) {
  switch (kind) {
  case SyncCoverResourceKind::EventId:
    return "event-id";
  case SyncCoverResourceKind::BufferToken:
    return "buffer-token";
  }
  return "unknown";
}

StringRef stringifyOwnershipAddressSpace(AddressSpace space) {
  switch (space) {
  case AddressSpace::Zero:
    return "zero";
  case AddressSpace::GM:
    return "gm";
  case AddressSpace::MAT:
    return "mat";
  case AddressSpace::LEFT:
    return "left";
  case AddressSpace::RIGHT:
    return "right";
  case AddressSpace::ACC:
    return "acc";
  case AddressSpace::VEC:
    return "vec";
  case AddressSpace::BIAS:
    return "bias";
  case AddressSpace::SCALING:
    return "scaling";
  }
  return "unknown";
}

StringRef stringifyOwnershipRole(CanonicalOwnershipEventRole role) {
  switch (role) {
  case CanonicalOwnershipEventRole::None:
    return "none";
  case CanonicalOwnershipEventRole::Ready:
    return "ready";
  case CanonicalOwnershipEventRole::Release:
    return "release";
  }
  return "unknown";
}

StringRef stringifyFixedEdgeKind(SyncGraphEdgeKind kind) {
  switch (kind) {
  case SyncGraphEdgeKind::IssueOrder:
    return "issue-order";
  case SyncGraphEdgeKind::NonCompletionPreservingIssueOrder:
    return "cross-pipe-issue-order";
  case SyncGraphEdgeKind::HardwareCompletion:
    return "hardware-completion";
  }
  return "unknown";
}

StringRef stringifyCoveringEdgeKind(SyncCoverEdgeKind kind) {
  switch (kind) {
  case SyncCoverEdgeKind::CompletionPreservingIssueOrder:
    return "completion-preserving-issue-order";
  case SyncCoverEdgeKind::NonCompletionPreservingIssueOrder:
    return "non-completion-preserving-issue-order";
  case SyncCoverEdgeKind::CompletionSupply:
    return "completion-supply";
  }
  return "unknown";
}

StringRef stringifyCoveringDemandKind(SyncCoverDemandKind kind) {
  switch (kind) {
  case SyncCoverDemandKind::SSA:
    return "ssa";
  case SyncCoverDemandKind::MemoryRAW:
    return "memory-raw";
  case SyncCoverDemandKind::MemoryWAR:
    return "memory-war";
  case SyncCoverDemandKind::MemoryWAW:
    return "memory-waw";
  }
  return "unknown";
}

StringRef stringifyCoveringSelectionError(SyncCoverSelectionError error) {
  switch (error) {
  case SyncCoverSelectionError::None:
    return "none";
  case SyncCoverSelectionError::InvalidUniverse:
    return "invalid-universe";
  case SyncCoverSelectionError::InvalidDemand:
    return "invalid-demand";
  case SyncCoverSelectionError::SearchIncomplete:
    return "search-incomplete";
  case SyncCoverSelectionError::FinalVerificationFailed:
    return "final-verification-failed";
  case SyncCoverSelectionError::GroundingFailed:
    return "grounding-failed";
  }
  return "unknown";
}

void printCoveringGuard(llvm::raw_ostream &os,
                        const SyncCoverGuard &guard) {
  os << '[';
  llvm::interleaveComma(guard.literals, os, [&](const auto &literal) {
    os << literal.control << ':' << literal.alternative;
  });
  os << ']';
}

void printIds(llvm::raw_ostream &os, ArrayRef<unsigned> ids) {
  os << '[';
  llvm::interleaveComma(ids, os);
  os << ']';
}

void printNodeIds(llvm::raw_ostream &os, ArrayRef<std::size_t> ids) {
  os << '[';
  llvm::interleaveComma(ids, os);
  os << ']';
}

void printMechanismRef(llvm::raw_ostream &os,
                       const CanonicalSelectionMechanismRef &mechanism) {
  os << stringifyCanonicalSelectionMechanismKind(mechanism.kind) << '['
     << mechanism.id << ']';
}

} // namespace

void mlir::pto::printCanonicalSyncPlan(llvm::raw_ostream &os, func::FuncOp func,
                                       const CanonicalSyncPlan &plan,
                                       StringRef view) {
  os << "PTOCanonicalSyncPlan @" << func.getSymName()
     << " gm-alias-policy="
     << stringifyCanonicalGMAliasPolicy(plan.getGMAliasPolicy())
     << " nodes=" << plan.getNodes().size()
     << " fixed=" << plan.getFixedEdges().size()
     << " dependencies=" << plan.getDependencies().size()
     << " requirements=" << plan.getCompletionRequirements().size()
     << " conservative-requirements="
     << plan.getConservativeCompletionRequirements().size()
     << " barriers=" << plan.getBarriers().size()
     << " events=" << plan.getEvents().size()
     << " ownership-cycles=" << plan.getOwnershipCycles().size() << '\n';
  if (includesDependencies(view)) {
    for (const CanonicalSyncNode &node : plan.getNodes()) {
      os << "  node[" << node.id
         << "] op=" << node.operation->getName().getStringRef()
         << " pipe=" << stringifyPIPE(static_cast<PIPE>(node.pipe))
         << " phase=" << node.macroPhase << " order=" << node.order
         << " accesses=" << node.accesses.size() << '\n';
    }
    for (const SyncGraphEdge &edge : plan.getFixedEdges()) {
      os << "  fixed " << edge.source << " -> " << edge.target
         << " kind=" << stringifyFixedEdgeKind(edge.kind) << '\n';
    }
    for (const CanonicalDependency &dependency : plan.getDependencies()) {
      os << "  dependency " << dependency.source << " -> " << dependency.target
         << " kind=" << stringifyCanonicalDependencyKind(dependency.kind)
         << " distance=" << dependency.iterationDistance
         << " retained=" << (dependency.retained ? "yes" : "no")
         << " active=" << (dependency.active ? "yes" : "no")
         << " possible=" << (dependency.possible ? "yes" : "no")
         << '\n';
    }
  }
  const bool includeScopes = includesDependencies(view) || includesPlan(view);
  if (includeScopes) {
    for (auto [requirementId, requirement] :
         llvm::enumerate(plan.getCompletionRequirements())) {
      os << "  requirement[" << requirementId << "] " << requirement.source
         << " -> " << requirement.target
         << " kind=" << stringifyCanonicalDependencyKind(requirement.kind)
         << " distance=" << requirement.iterationDistance
         << " recurrence=" << (requirement.recurrenceLoop ? "yes" : "no")
         << '\n';
    }
  }
  if (includesPlan(view)) {
    for (const CanonicalBarrier &barrier : plan.getBarriers()) {
      os << "  barrier[" << barrier.id
         << "] pipe=" << stringifyPIPE(static_cast<PIPE>(barrier.pipe))
         << " anchor=" << (barrier.anchor.before ? "before:" : "after:")
         << barrier.anchor.operation->getName().getStringRef()
         << " anchor-nodes=";
      printNodeIds(os, barrier.anchorNodes);
      os << " recurrence=" << (barrier.recurrenceLoop ? "yes" : "no")
         << " scope=" << barrier.recurrenceScope << " requirements=";
      printNodeIds(os, barrier.requirements);
      os << '\n';
    }
  }
  if (includesPlan(view) || includesEvents(view)) {
    for (auto [index, event] : llvm::enumerate(plan.getEvents())) {
      os << "  event[" << index << "] "
         << stringifyPIPE(static_cast<PIPE>(event.sourcePipe)) << " -> "
         << stringifyPIPE(static_cast<PIPE>(event.targetPipe))
         << " source=" << event.source << " target=" << event.target
         << " width=" << event.width << " ids=";
      printIds(os, event.eventIds);
      os << " lifetime=[" << event.intervalBegin << ',' << event.intervalEnd
         << "] recurrence=" << (event.recurrenceLoop ? "yes" : "no")
         << " distance=" << event.iterationDistance
         << " actions=" << event.actions.size()
         << " completions=" << event.completions.size()
         << " traces=" << event.traces.size()
         << " ownership=" << (event.ownershipProtocol ? "yes" : "no")
         << " ownership-cycle=" << event.ownershipCycle
         << " ownership-role=" << stringifyOwnershipRole(event.ownershipRole)
         << '\n';
    }
  }
  if (includesEvents(view)) {
    for (const CanonicalEventDomain &domain : plan.getDomains()) {
      os << "  domain " << stringifyPIPE(static_cast<PIPE>(domain.sourcePipe))
         << " -> " << stringifyPIPE(static_cast<PIPE>(domain.targetPipe))
         << " original-events=" << domain.originalEventCount
         << " events=" << domain.eventCount
         << " available=" << domain.availableIds
         << " original-colors=" << domain.originalColorCount
         << " colors=" << domain.colorCount
         << " serialization-cost=" << domain.serializationCost
         << " original-critical-path-weight="
         << domain.originalCriticalPathWeight
         << " critical-path-weight=" << domain.criticalPathWeight
         << " reserved=";
      printIds(os, domain.reservedIds);
      os << '\n';
    }
  }
  if (includesOwnership(view)) {
    for (auto [cycleIndex, cycle] :
         llvm::enumerate(plan.getOwnershipCycles())) {
      os << "  ownership[" << cycleIndex << "] id=" << cycle.id
         << " kind=" << stringifyCanonicalOwnershipKind(cycle.kind) << ' '
         << stringifyPIPE(static_cast<PIPE>(cycle.producerPipe)) << " -> "
         << stringifyPIPE(static_cast<PIPE>(cycle.consumerPipe))
         << " lanes=" << cycle.lanes.size() << " paths=" << cycle.paths.size()
         << getOwnershipProtocolSuffix(cycle.protocol) << '\n';
      for (const CanonicalOwnershipLane &lane : cycle.lanes) {
        os << "    lane[" << lane.id << "] slots=[";
        llvm::interleaveComma(lane.slots, os,
                              [&](const CanonicalPhysicalSlot &slot) {
                                os << stringifyOwnershipAddressSpace(slot.space)
                                   << '@' << slot.address << '+' << slot.size;
                              });
        os << "]\n";
      }
      for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
        os << "    path[" << pathIndex << "] uses=" << path.uses.size()
           << " lane-order=[";
        llvm::interleaveComma(
            path.uses, os,
            [&](const CanonicalOwnershipUse &use) { os << use.lane; });
        os << "]\n";
        for (auto [useIndex, use] : llvm::enumerate(path.uses)) {
          os << "      use[" << useIndex << "] lane=" << use.lane
             << " producers=";
          printNodeIds(os, use.producers);
          os << " consumers=";
          printNodeIds(os, use.consumers);
          if (use.producerLane != use.lane) {
            os << " producer-lane=" << use.producerLane;
          }
          os << '\n';
        }
      }
    }
  }
  const bool includeCovering = includesCovering(view);
  const bool hasCoveringSnapshot = plan.getCoveringSnapshot().has_value();
  if (includeCovering && hasCoveringSnapshot) {
    const CanonicalSyncCoveringSnapshot &snapshot =
        *plan.getCoveringSnapshot();
    os << "  covering status=graph-ready"
       << " scopes=" << snapshot.scopes
       << " controls=" << snapshot.controls << " nodes=" << snapshot.nodes
       << " fixed-edges=" << snapshot.fixedEdges
       << " recurrence-carries=" << snapshot.recurrenceCarryEdges
       << " conservative-demands=" << snapshot.conservativeDemands
       << " active-demands=" << snapshot.activeDemands
       << " intrinsic-demands=" << snapshot.intrinsicallySatisfiedDemands
       << '\n';
    os << "  covering-slot-lifecycles candidates="
       << snapshot.slotLifecycleCandidates
       << " path-sensitive=" << snapshot.pathSensitiveSlotLifecycles
       << " partial-opportunities=" << snapshot.partialSlotOpportunities
       << " truncated="
       << (snapshot.slotLifecycleDiscoveryTruncated ? "yes" : "no")
       << '\n';
    for (const CanonicalSyncCoveringSlotLifecycle &lifecycle :
         snapshot.slotLifecycleDetails) {
      os << "  covering-slot-lifecycle[" << lifecycle.id << "] domain="
         << lifecycle.domain << " extent=[" << lifecycle.extent.begin << ','
         << lifecycle.extent.end << ") "
         << stringifyPIPE(static_cast<PIPE>(lifecycle.producerResource))
         << "->"
         << stringifyPIPE(static_cast<PIPE>(lifecycle.consumerResource))
         << " scope=" << lifecycle.recurrenceScope
         << " distance=" << lifecycle.distance
         << " ready=";
      printNodeIds(os, lifecycle.ready);
      os << " release=";
      printNodeIds(os, lifecycle.release);
      os << " accesses=";
      printNodeIds(os, lifecycle.managedAccesses);
      os << " extra-accesses="
         << (lifecycle.hasUnrepresentedAccesses ? "yes" : "no")
         << " path-sensitive="
         << (lifecycle.requiresPathSensitiveProof ? "yes" : "no") << '\n';
    }
    os << "  covering-slot-protocols candidates="
       << snapshot.slotProtocolCandidates
       << " path-sensitive-lifecycles="
       << snapshot.pathSensitiveSlotProtocolLifecycles
       << " access-open-lifecycles="
       << snapshot.accessOpenSlotProtocolLifecycles
       << " unsupported-effect-lifecycles="
       << snapshot.unsupportedEffectSlotProtocolLifecycles
       << " unsupported-distance-releases="
       << snapshot.unsupportedDistanceSlotProtocolReleases
       << " non-boundary-releases="
       << snapshot.nonBoundarySlotProtocolReleases
       << " evaluations=" << snapshot.slotProtocolEvaluations
       << " truncated="
       << (snapshot.slotProtocolGenerationTruncated ? "yes" : "no")
       << " unmaterializable-candidates="
       << snapshot.unmaterializableSlotProtocolCandidates << '\n';
    os << "  covering-selection status="
       << (snapshot.selectionAttempted ? "ready" : "not-run")
       << " error=" << stringifyCoveringSelectionError(snapshot.selectionError)
       << " domains=" << snapshot.resourceDomainCount
       << " barrier-candidates=" << snapshot.barrierCandidates
       << " event-bundle-candidates=" << snapshot.eventBundleCandidates
       << " slot-protocol-candidates="
       << snapshot.slotProtocolMechanismCandidates
       << " generated-candidates=" << snapshot.generatedColumnCandidates
       << " generated=" << snapshot.generatedColumns
       << " generation-truncated="
       << (snapshot.columnGenerationTruncated ? "yes" : "no")
       << " mechanisms=" << snapshot.candidateMechanisms
       << " selected=" << snapshot.selectedMechanisms
       << " evaluations=" << snapshot.solverEvaluations
       << " redundancy-evaluations=" << snapshot.redundancyEvaluations
       << " oracle-redundancy-checks=" << snapshot.oracleRedundancyChecks
       << " event-uncovered=" << snapshot.demandsWithoutEventColumn.size()
       << " actions=";
    printNodeIds(os, snapshot.actionProfile);
    os << " barriers=";
    printNodeIds(os, snapshot.barrierActionProfile);
    os << " providers=[";
    llvm::interleaveComma(snapshot.selectedProviders, os,
                          [&](const auto &selected) {
                            os << "mechanism[" << selected.mechanism << "]=";
                            printMechanismRef(os, selected.provider);
                          });
    os << "]\n";
    os << "  covering-event-uncovered demands=";
    printNodeIds(os, snapshot.demandsWithoutEventColumn);
    os << '\n';
    os << "  covering-selection-topology prepared-demands="
       << snapshot.coverageStatistics.demandPreparations
       << " virtual-nodes="
       << snapshot.coverageStatistics.preparedVirtualNodes
       << " virtual-edges="
       << snapshot.coverageStatistics.preparedVirtualEdges
       << " max-virtual-nodes="
       << snapshot.coverageStatistics.maximumVirtualNodes
       << " max-virtual-edges="
       << snapshot.coverageStatistics.maximumVirtualEdges
       << " grounding-queries="
       << snapshot.coverageStatistics.groundingQueries
       << " coverage-queries=" << snapshot.coverageStatistics.coverageQueries
       << " final-prepared-demands="
       << snapshot.finalVerificationStatistics.demandPreparations
       << " final-virtual-nodes="
       << snapshot.finalVerificationStatistics.preparedVirtualNodes
       << " final-virtual-edges="
       << snapshot.finalVerificationStatistics.preparedVirtualEdges
       << " final-validations="
       << snapshot.finalVerificationStatistics.graphValidations
       << " final-coverage-queries="
       << snapshot.finalVerificationStatistics.coverageQueries << '\n';
    for (const CanonicalSyncCoveringSelectedResourceUse &use :
         snapshot.selectedResourceUses) {
      os << "  covering-use mechanism[" << use.mechanism << "]=";
      printMechanismRef(os, use.provider);
      os << " use=" << use.resourceUse << " domain=" << use.domain
         << " kind=" << stringifyCoveringResourceKind(use.kind)
         << " width=" << use.width << " lifetime=[" << use.lifetime.begin
         << ',' << use.lifetime.end << "] event=";
      if (use.materializationEventIndex) {
        os << *use.materializationEventIndex;
      } else {
        os << "none";
      }
      os << '\n';
    }
    for (const CanonicalSyncCoveringResourceAllocation &allocation :
         snapshot.selectedAllocations) {
      os << "  covering-allocation mechanism[" << allocation.mechanism << "]=";
      printMechanismRef(os, allocation.provider);
      os << " use=" << allocation.resourceUse
         << " domain=" << allocation.domain << " kind="
         << stringifyCoveringResourceKind(allocation.kind) << " resources=";
      if (allocation.kind == SyncCoverResourceKind::EventId) {
        os << stringifyPIPE(static_cast<PIPE>(allocation.sourceResource))
           << "->"
           << stringifyPIPE(static_cast<PIPE>(allocation.targetResource));
      } else {
        os << allocation.sourceResource << "->" << allocation.targetResource;
      }
      os << " ids=";
      printIds(os, allocation.ids);
      os << '\n';
    }
    for (const SyncCoverScope &scope : snapshot.scopeDetails) {
      os << "  covering-scope[" << scope.id << "] parent=" << scope.parent
         << " must-execute="
         << (scope.mustExecuteWithinParent ? "yes" : "no")
         << " loop=" << (scope.isLoop ? "yes" : "no") << " timeline=";
      if (scope.timeline) {
        os << '[' << scope.timeline->begin << ',' << scope.timeline->end << ']';
      } else {
        os << "none";
      }
      os << '\n';
    }
    for (const SyncCoverControl &control : snapshot.controlDetails) {
      os << "  covering-control[" << control.id
         << "] alternatives=" << control.alternatives
         << " scope=" << control.scope << '\n';
    }
    for (const SyncCoverNode &node : snapshot.nodeDetails) {
      os << "  covering-node[" << node.id << "] resource="
         << stringifyPIPE(static_cast<PIPE>(node.resource))
         << " scope=" << node.scope << " order=" << node.order
         << " guard=";
      printCoveringGuard(os, node.guard);
      os << " completion-targets=[";
      llvm::interleaveComma(node.completionTargets, os);
      os << "]\n";
    }
    for (auto [edgeId, edge] : llvm::enumerate(snapshot.edgeDetails)) {
      os << "  covering-edge[" << edgeId << "] " << edge.source << " -> "
         << edge.target << " kind=" << stringifyCoveringEdgeKind(edge.kind)
         << " scope=" << edge.scope << " distance=" << edge.distance
         << " origin="
         << (edge.mechanism
                 ? "mechanism["
                 : (edgeId < snapshot.fixedEdges ? "fixed"
                                                 : "recurrence-carry"));
      if (edge.mechanism) {
        os << *edge.mechanism << ']';
      }
      os << " source-guard=";
      printCoveringGuard(os, edge.sourceGuard);
      os << " target-guard=";
      printCoveringGuard(os, edge.targetGuard);
      os << '\n';
    }
    for (auto [demandId, demand] : llvm::enumerate(snapshot.demandDetails)) {
      const bool active =
          std::binary_search(snapshot.activeDemandIds.begin(),
                             snapshot.activeDemandIds.end(), demandId);
      os << "  covering-demand[" << demandId << "] " << demand.source
         << " -> " << demand.target
         << " kind=" << stringifyCoveringDemandKind(demand.kind)
         << " scope=" << demand.scope << " distance=" << demand.distance
         << " active=" << (active ? "yes" : "no") << " source-guard=";
      printCoveringGuard(os, demand.sourceGuard);
      os << " target-guard=";
      printCoveringGuard(os, demand.targetGuard);
      os << '\n';
    }
  }
}
