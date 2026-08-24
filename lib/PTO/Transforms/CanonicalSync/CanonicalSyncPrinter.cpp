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
  }
  return "unknown";
}

StringRef mlir::pto::stringifyCanonicalEventBundleKind(
    CanonicalEventBundleKind kind) {
  switch (kind) {
  case CanonicalEventBundleKind::Standalone:
    return "standalone";
  case CanonicalEventBundleKind::SyntheticRoundTrip:
    return "synthetic-round-trip";
  case CanonicalEventBundleKind::Ownership:
    return "ownership";
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

bool includesSelection(StringRef view) {
  return view == "selection";
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

std::size_t getRecurrenceScopeId(
    Operation *operation, ArrayRef<CanonicalRecurrenceScope> scopes) {
  auto scope = llvm::find_if(scopes, [&](const auto &candidate) {
    return candidate.operation == operation;
  });
  return scope == scopes.end() ? 0 : scope->id;
}

void printQuoted(llvm::raw_ostream &os, StringRef text) {
  os << '"';
  for (char character : text) {
    if (character == '"' || character == '\\') {
      os << '\\';
    }
    if (character == '\n') {
      os << "\\n";
    } else {
      os << character;
    }
  }
  os << '"';
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
     << " ownership-cycles=" << plan.getOwnershipCycles().size()
     << " bootstrap=" << (plan.usedInfeasibleBootstrap() ? "yes" : "no")
     << '\n';
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
    for (const CanonicalRecurrenceScope &scope : plan.getRecurrenceScopes()) {
      os << "  scope[" << scope.id << "] op="
         << scope.operation->getName().getStringRef()
         << " operation-order=" << scope.operationOrder
         << " parent-scope=" << scope.parentScope << '\n';
    }
    for (auto [requirementId, requirement] :
         llvm::enumerate(plan.getCompletionRequirements())) {
      os << "  requirement[" << requirementId << "] " << requirement.source
         << " -> " << requirement.target
         << " kind=" << stringifyCanonicalDependencyKind(requirement.kind)
         << " distance=" << requirement.iterationDistance
         << " recurrence=" << (requirement.recurrenceLoop ? "yes" : "no")
         << " scope="
         << getRecurrenceScopeId(requirement.recurrenceLoop,
                                 plan.getRecurrenceScopes())
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
         << " bundle=" << event.protocolBundle << '\n';
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
         << (cycle.protocol ==
                     CanonicalOwnershipProtocolKind::AlternatingPrefetch
                 ? " protocol=alternating-prefetch"
                 : "")
         << '\n';
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
  if (includesSelection(view)) {
    for (const CanonicalRecurrenceScope &scope : plan.getRecurrenceScopes()) {
      os << "  selection scope[" << scope.id << "] op="
         << scope.operation->getName().getStringRef()
         << " operation-order=" << scope.operationOrder
         << " parent-scope=" << scope.parentScope << '\n';
    }
    for (const CanonicalSelectionDiagnostic &diagnostic :
         plan.getSelectionDiagnostics()) {
      for (const CanonicalSelectionMechanismSummary &mechanism :
           diagnostic.selectedMechanisms) {
        os << "  selection ";
        printMechanismRef(os, mechanism.mechanism);
        if (mechanism.mechanism.kind ==
            CanonicalSelectionMechanismKind::EventBundle) {
          os << " kind="
             << stringifyCanonicalEventBundleKind(
                    mechanism.eventBundleKind)
             << " domains=[";
          llvm::interleaveComma(
              mechanism.eventDomains, os,
              [&](const CanonicalSelectionEventDomain &domain) {
                os << stringifyPIPE(static_cast<PIPE>(domain.sourcePipe))
                   << "->"
                   << stringifyPIPE(static_cast<PIPE>(domain.targetPipe));
              });
          os << "] action-sites=" << mechanism.actionSites
             << " lanes=" << mechanism.eventLanes;
        }
        os << '\n';
      }
      if (diagnostic.evictedMechanisms.empty()) {
        continue;
      }
      os << "  eviction mechanisms=[";
      llvm::interleaveComma(
          diagnostic.evictedMechanisms, os,
          [&](const CanonicalSelectionMechanismRef &mechanism) {
            printMechanismRef(os, mechanism);
          });
      os << "] exclusive-requirements=";
      printNodeIds(os, diagnostic.exclusiveRequirements);
      os << '\n';
      for (const CanonicalSelectionColorPressure &pressure :
           diagnostic.colorPressure) {
        os << "    color "
           << stringifyPIPE(static_cast<PIPE>(pressure.sourcePipe)) << " -> "
           << stringifyPIPE(static_cast<PIPE>(pressure.targetPipe))
           << " before=" << pressure.beforeColors
           << " after=" << pressure.afterColors
           << " available=" << pressure.availableIds << '\n';
      }
      for (const CanonicalSelectionRequirementDiagnostic &entry :
           diagnostic.uncoveredRequirements) {
        if (entry.requirement >= plan.getCompletionRequirements().size()) {
          continue;
        }
        const CanonicalDependency &requirement =
            plan.getCompletionRequirements()[entry.requirement];
        const CanonicalSyncNode &source = plan.getNodes()[requirement.source];
        const CanonicalSyncNode &target = plan.getNodes()[requirement.target];
        os << "    uncovered requirement[" << entry.requirement << "] "
           << requirement.source << " -> " << requirement.target
           << " pipes=" << stringifyPIPE(static_cast<PIPE>(source.pipe))
           << "->" << stringifyPIPE(static_cast<PIPE>(target.pipe))
           << " kind=" << stringifyCanonicalDependencyKind(requirement.kind)
           << " distance=" << requirement.iterationDistance
           << " scope=" << entry.recurrenceScope << " candidates=[";
        llvm::interleaveComma(
            entry.candidates, os,
            [&](const CanonicalSelectionReplacementCandidate &candidate) {
              printMechanismRef(os, candidate.mechanism);
              if (candidate.mechanism.kind ==
                  CanonicalSelectionMechanismKind::Barrier) {
                os << ":pipe="
                   << stringifyPIPE(static_cast<PIPE>(candidate.barrierPipe));
              } else {
                os << ":kind="
                   << stringifyCanonicalEventBundleKind(
                          candidate.eventBundleKind)
                   << ":domains=[";
                llvm::interleaveComma(
                    candidate.eventDomains, os,
                    [&](const CanonicalSelectionEventDomain &domain) {
                      os << stringifyPIPE(
                                static_cast<PIPE>(domain.sourcePipe))
                         << "->"
                         << stringifyPIPE(
                                static_cast<PIPE>(domain.targetPipe));
                    });
                os << ']';
              }
              os << ":overflow=" << candidate.colorOverflow;
            });
        os << "]\n";
      }
    }
  }
}

void mlir::pto::printCanonicalSyncPlanDot(llvm::raw_ostream &os,
                                          func::FuncOp func,
                                          const CanonicalSyncPlan &plan,
                                          StringRef view) {
  os << "digraph ";
  printQuoted(os, func.getSymName());
  os << " {\n  graph [label=";
  printQuoted(os, (llvm::Twine("gm-alias-policy=") +
                   stringifyCanonicalGMAliasPolicy(plan.getGMAliasPolicy()))
                      .str());
  os << ", labelloc=t];\n  rankdir=LR;\n"
        "  node [fontname=\"DejaVu Sans\", fontsize=10, shape=box];\n";
  const bool printNodes = includesDependencies(view) || includesPlan(view) ||
                          includesOwnership(view);
  if (printNodes) {
    for (const CanonicalSyncNode &node : plan.getNodes()) {
      os << "  n" << node.id << " [label=";
      printQuoted(os, (llvm::Twine(node.id) + ": " +
                       node.operation->getName().getStringRef() + "\n" +
                       stringifyPIPE(static_cast<PIPE>(node.pipe)))
                          .str());
      os << "];\n";
    }
    for (const SyncGraphEdge &edge : plan.getFixedEdges()) {
      os << "  n" << edge.source << " -> n" << edge.target << " [label=";
      printQuoted(os, stringifyFixedEdgeKind(edge.kind));
      os << ", color=\"#6B7280\"];\n";
    }
    for (const CanonicalDependency &dependency : plan.getDependencies()) {
      os << "  n" << dependency.source << " -> n" << dependency.target
         << " [label=";
      printQuoted(os, stringifyCanonicalDependencyKind(dependency.kind));
      os << ", style="
         << (dependency.retained
                 ? (dependency.iterationDistance ? "dashed" : "solid")
                 : "dotted")
         << "];\n";
    }
  }
  if (includesPlan(view) || includesEvents(view)) {
    for (auto [index, event] : llvm::enumerate(plan.getEvents())) {
      os << "  e" << index << " [shape=ellipse, label=";
      printQuoted(os,
                  (llvm::Twine("event ") + llvm::Twine(index) + "\n" +
                   stringifyPIPE(static_cast<PIPE>(event.sourcePipe)) + " -> " +
                   stringifyPIPE(static_cast<PIPE>(event.targetPipe)))
                      .str());
      os << "];\n";
    }
    for (std::size_t first = 0; first < plan.getEvents().size(); ++first) {
      for (std::size_t second = first + 1; second < plan.getEvents().size();
           ++second) {
        const CanonicalEvent &left = plan.getEvents()[first];
        const CanonicalEvent &right = plan.getEvents()[second];
        const bool sameDomain = left.sourcePipe == right.sourcePipe &&
                                left.targetPipe == right.targetPipe;
        const bool overlaps = !(left.intervalEnd < right.intervalBegin ||
                                right.intervalEnd < left.intervalBegin);
        if (sameDomain && overlaps) {
          os << "  e" << first << " -> e" << second
             << " [dir=none, color=\"#9C2F45\"];\n";
        }
      }
    }
  }
  if (includesOwnership(view)) {
    for (auto [cycleIndex, cycle] :
         llvm::enumerate(plan.getOwnershipCycles())) {
      os << "  o" << cycleIndex << " [shape=diamond, label=";
      printQuoted(os, (llvm::Twine("ownership ") + llvm::Twine(cycleIndex) +
                       "\n" + stringifyCanonicalOwnershipKind(cycle.kind))
                          .str());
      os << "];\n";
      for (const CanonicalOwnershipPath &path : cycle.paths) {
        for (const CanonicalOwnershipUse &use : path.uses) {
          for (std::size_t producer : use.producers) {
            os << "  n" << producer << " -> o" << cycleIndex
               << " [style=dotted];\n";
          }
          for (std::size_t consumer : use.consumers) {
            os << "  o" << cycleIndex << " -> n" << consumer
               << " [style=dotted];\n";
          }
        }
      }
    }
  }
  os << "}\n";
}
