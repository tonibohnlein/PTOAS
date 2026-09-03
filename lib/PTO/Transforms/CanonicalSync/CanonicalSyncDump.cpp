// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncModel.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

void printResource(raw_ostream &os, CanonicalPhysicalResource resource) {
  os << stringifyCanonicalCore(resource.core) << ':'
     << stringifyPIPE(resource.pipe);
}

void printGuard(raw_ostream &os, ArrayRef<CanonicalControlAtom> guard) {
  os << '[';
  llvm::interleaveComma(guard, os, [&os](const CanonicalControlAtom &atom) {
    os << 'r' << atom.choice << ".a" << atom.arm;
  });
  os << ']';
}

void printPoint(raw_ostream &os, CanonicalProgramPoint point) {
  os << (point.position == CanonicalProgramPointPosition::Before ? "before("
                                                                 : "after(");
  if (point.operation) {
    os << point.operation->getName();
  } else {
    os << '-';
  }
  os << ')';
}

} // namespace

void mlir::pto::printCanonicalSyncProgram(const CanonicalSyncProgram &program,
                                          raw_ostream &os) {
  os << "CANONICAL-SYNC function=" << program.getFunction().getSymName()
     << " gm-alias-policy="
     << stringifyCanonicalGmAliasPolicy(program.getGmAliasPolicy()) << '\n';
  os << "TARGET npu2201-a2a3-v1 ids=0,1,2,3,4,5\n";
  os << "REGIONS " << program.getRegions().size() << '\n';
  for (const CanonicalRegion &region : program.getRegions()) {
    os << "  r" << region.id
       << " kind=" << stringifyCanonicalRegionKind(region.kind) << " parent=";
    if (region.parent == kInvalidCanonicalSyncId) {
      os << '-';
    } else {
      os << 'r' << region.parent;
    }
    os << " depth=" << region.depth << " arm=" << region.arm << '\n';
  }
  os << "PHASES " << program.getPhases().size() << '\n';
  for (const CanonicalPhase &phase : program.getPhases()) {
    os << "  p" << phase.id << " op=" << phase.operation->getName()
       << " resource=";
    printResource(os, phase.resource);
    os << " region=r" << phase.region << " order=" << phase.sourceOrder
       << " guard=";
    printGuard(os, phase.controlPath);
    if (phase.macroPhase) {
      os << " macro-phase=" << *phase.macroPhase;
    }
    os << '\n';
  }
  os << "ACCESSES " << program.getAccesses().size() << '\n';
  for (const CanonicalAccess &access : program.getAccesses()) {
    os << "  a" << access.id << " phase=p" << access.phase
       << " mode=" << stringifyCanonicalAccessMode(access.mode) << " space=";
    if (access.unknownSpace) {
      os << "unknown";
    } else {
      os << stringifyAddressSpace(access.space);
    }
    os << " range=";
    if (access.unknownRange) {
      os << "unknown";
    } else {
      llvm::interleaveComma(
          access.intervals, os, [&os](const CanonicalByteInterval &interval) {
            os << '[' << interval.begin << ',' << *interval.end() << ')';
          });
    }
    os << " ordered=" << (access.ordered ? "yes" : "no")
       << " physical=" << (access.physical ? "yes" : "no")
       << " slot=" << (access.slotExpression ? "explicit" : "root")
       << " provenance=" << access.provenance << '\n';
  }
  os << "FENCE-EFFECTS " << program.getFenceEffects().size() << '\n';
  for (const CanonicalFenceEffect &effect : program.getFenceEffects()) {
    os << "  f" << effect.id << " scope=" << stringifyFenceScope(effect.scope)
       << " point=";
    printPoint(os, {effect.operation, CanonicalProgramPointPosition::After});
    os << " drains=[";
    llvm::interleaveComma(effect.drainedResources, os,
                          [&os](CanonicalPhysicalResource resource) {
                            printResource(os, resource);
                          });
    os << "] guard=";
    printGuard(os, effect.guard);
    os << '\n';
  }
  os << "DEMANDS " << program.getDemands().size() << '\n';
  for (const CanonicalDemand &demand : program.getDemands()) {
    os << "  d" << demand.id
       << " kind=" << stringifyCanonicalDemandKind(demand.kind) << " p"
       << demand.source << "->";
    if (demand.target == kInvalidCanonicalSyncId) {
      os << "exit";
    } else {
      os << 'p' << demand.target;
    }
    os << " owner=r" << demand.owner << " source-guard=";
    printGuard(os, demand.sourceGuard);
    os << " target-guard=";
    printGuard(os, demand.targetGuard);
    os << " distance=[";
    llvm::interleaveComma(
        demand.iterationDistance, os,
        [&os](const CanonicalLoopDistance &distance) {
          os << 'r' << distance.loop << ':';
          if (distance.relation == CanonicalIterationRelation::AnyPositive) {
            os << "+";
          } else if (distance.relation == CanonicalIterationRelation::Any) {
            os << '*';
          } else {
            os << '0';
          }
        });
    os << "] causes=" << demand.causes.size();
    if (demand.visibility) {
      os << " visibility="
         << stringifyCanonicalVisibilityDirection(demand.visibility->direction)
         << ":" << stringifyFenceScope(demand.visibility->scope) << ':'
         << stringifyCanonicalCacheMaintenance(
                demand.visibility->cacheMaintenance);
    }
    if (demand.id < program.getDirectMechanisms().size()) {
      os << " direct=m" << program.getDirectMechanisms()[demand.id];
    }
    os << '\n';
  }
  os << "MECHANISMS " << program.getMechanisms().size() << '\n';
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    os << "  m" << mechanism.id
       << " kind=" << stringifyCanonicalMechanismKind(mechanism.kind) << ' ';
    if (mechanism.kind == CanonicalMechanismKind::FixedFence) {
      os << "fence=f" << *mechanism.fenceEffect;
    } else {
      os << "resource=";
      printResource(os, mechanism.source);
      os << "->";
      printResource(os, mechanism.target);
    }
    if (mechanism.kind == CanonicalMechanismKind::TailBarrier &&
        !mechanism.sourcePoint.operation) {
      os << " points=exit";
    } else {
      os << " points=";
      printPoint(os, mechanism.sourcePoint);
      os << "->";
      printPoint(os, mechanism.targetPoint);
    }
    os << " origins=[";
    llvm::interleaveComma(mechanism.origins, os,
                          [&os](CanonicalDemandId id) { os << 'd' << id; });
    os << ']';
    if (mechanism.recurrenceLoop) {
      os << " recurrence-loop=r" << *mechanism.recurrenceLoop;
      if (mechanism.boundaryRecurring) {
        os << " protocol=boundary-handshake";
      }
    }
    os << '\n';
  }
  os << "STORAGE-GENERATIONS " << program.getStorageGenerations().size()
     << '\n';
  for (const CanonicalStorageGeneration &generation :
       program.getStorageGenerations()) {
    os << "  generation=s" << generation.id << " loop=r"
       << generation.recurrenceLoop << " family={" << generation.familyKey
       << "} depth=" << generation.familyDepth << " slot=" << generation.slot
       << " stage=" << generation.stageOrdinal << " resource=";
    printResource(os, generation.producer);
    os << "->";
    printResource(os, generation.consumer);
    os << " range=[" << generation.interval.begin << ','
       << *generation.interval.end() << ") period=" << generation.period
       << " ready-distance=" << generation.readyDistance
       << " next-overwrite-distance=" << generation.nextOverwriteDistance
       << " initial=" << (generation.initialProducer ? "yes" : "no")
       << " write-acquire=";
    printPoint(os, generation.writeAcquire);
    os << " ready=";
    printPoint(os, generation.ready);
    os << " read-acquire=";
    printPoint(os, generation.readAcquire);
    os << " last-use=";
    printPoint(os, generation.lastUse);
    os << " next-overwrite=";
    printPoint(os, generation.nextOverwrite);
    os << " producers=[";
    llvm::interleaveComma(generation.producers, os,
                          [&os](CanonicalPhaseId id) { os << 'p' << id; });
    os << "] consumers=[";
    llvm::interleaveComma(generation.consumers, os,
                          [&os](CanonicalPhaseId id) { os << 'p' << id; });
    os << "] producer-residues=[";
    llvm::interleaveComma(generation.producerResidues, os);
    os << "] consumer-residues=[";
    llvm::interleaveComma(generation.consumerResidues, os);
    os << "]\n";
  }
  os << "OWNERSHIP-PROTOCOLS " << program.getOwnershipProtocols().size()
     << '\n';
  for (const CanonicalOwnershipProtocol &protocol :
       program.getOwnershipProtocols()) {
    os << "  ownership=o" << protocol.id << " mechanism=m" << protocol.mechanism
       << " owner=r" << protocol.owner << " loop=r" << protocol.recurrenceLoop
       << " resource=";
    printResource(os, protocol.producer);
    os << "->";
    printResource(os, protocol.consumer);
    os << " depth=" << protocol.depth
       << " token-lanes=" << protocol.lanes.size()
       << " period=" << protocol.period
       << " reuse-distance=" << protocol.reuseDistance
       << " witness-horizon=" << protocol.witnessHorizon << " family={"
       << protocol.familyKey << "} parents=[";
    llvm::interleaveComma(protocol.parentMechanisms, os,
                          [&os](CanonicalMechanismId id) { os << 'm' << id; });
    os << "] demands=[";
    llvm::interleaveComma(protocol.witnessDemands, os,
                          [&os](CanonicalDemandId id) { os << 'd' << id; });
    os << "]\n";
    for (auto [laneIndex, lane] : llvm::enumerate(protocol.lanes)) {
      os << "    lane=" << laneIndex;
      if (lane.readyEventId) {
        os << " ready=e" << *lane.readyEventId;
      }
      if (lane.releaseEventId) {
        os << " release=e" << *lane.releaseEventId;
      }
      os << '\n';
    }
    for (auto [slotIndex, slot] : llvm::enumerate(protocol.slots)) {
      os << "    slot=" << slotIndex << " lane=" << slot.lane << " range=["
         << slot.interval.begin << ',' << *slot.interval.end()
         << ") expression=" << (slot.slotExpression ? "explicit" : "root")
         << " reuse-distance=" << slot.reuseDistance << '\n';
    }
    for (auto [stageIndex, stage] : llvm::enumerate(protocol.stages)) {
      os << "    stage=" << stageIndex << " slot=" << stage.slot
         << " lane=" << stage.lane << " generation=s" << stage.generation
         << " initial=" << (stage.initialProducer ? "yes" : "no")
         << " ready-distance=" << stage.readyDistance
         << " release-distance=" << stage.releaseDistance << " write-acquire=";
      printPoint(os, stage.writeAcquire);
      os << " ready=";
      printPoint(os, stage.ready);
      os << " read-acquire=";
      printPoint(os, stage.readAcquire);
      os << " release=";
      printPoint(os, stage.release);
      os << " producers=[";
      llvm::interleaveComma(stage.producers, os,
                            [&os](CanonicalPhaseId id) { os << 'p' << id; });
      os << "] consumers=[";
      llvm::interleaveComma(stage.consumers, os,
                            [&os](CanonicalPhaseId id) { os << 'p' << id; });
      os << "] producer-guard=";
      printGuard(os, stage.producerGuard);
      os << " consumer-guard=";
      printGuard(os, stage.consumerGuard);
      os << '\n';
    }
    for (const CanonicalOwnershipWitnessEdge &edge : protocol.witnessEdges) {
      os << "    witness="
         << (edge.kind == CanonicalOwnershipWitnessKind::Ready ? "ready"
                                                               : "release")
         << " slot=" << edge.slot << " lane=" << edge.lane << " p"
         << edge.source << '@'
         << edge.sourceIteration << "->p" << edge.target << '@'
         << edge.targetIteration << '\n';
    }
  }
  os << "COVERAGE " << program.getCoverageWorlds().size() << '\n';
  for (const CanonicalCoverageWorld &world : program.getCoverageWorlds()) {
    os << "  world=" << world.name << " mechanisms=[";
    llvm::interleaveComma(world.mechanisms, os,
                          [&os](CanonicalMechanismId id) { os << 'm' << id; });
    os << "] covered=[";
    llvm::interleaveComma(world.covered, os,
                          [&os](CanonicalDemandId id) { os << 'd' << id; });
    os << "] summaries=" << world.summaries.size() << " oracle=";
    if (!world.flattenedOracleMatched || !world.unrolledOracleAvailable ||
        (world.unrolledOracleExhaustive && !world.unrolledOracleMatched)) {
      os << "mismatch";
    } else if (!world.unrolledOracleExhaustive) {
      os << "flat-match/unrolled-inconclusive";
    } else {
      os << "match";
    }
    if (world.structuralProposal) {
      os << " proposal=g" << *world.structuralProposal
         << " admitted=" << (world.setCoverCandidate ? "yes" : "no");
    }
    os << '\n';
  }
  os << "STRUCTURAL-PROPOSALS " << program.getStructuralProposals().size()
     << '\n';
  for (const CanonicalStructuralProposal &proposal :
       program.getStructuralProposals()) {
    os << "  proposal=g" << proposal.id
       << " kind=" << stringifyCanonicalStructuralProposalKind(proposal.kind)
       << " owner=r" << proposal.owner << " level=" << proposal.level
       << " semantics={" << proposal.semanticKey << "} mechanisms=[";
    llvm::interleaveComma(proposal.mechanisms, os,
                          [&os](CanonicalMechanismId id) { os << 'm' << id; });
    os << "] crossing=[";
    llvm::interleaveComma(proposal.crossingDemands, os,
                          [&os](CanonicalDemandId id) { os << 'd' << id; });
    os << "] singleton-union=[";
    llvm::interleaveComma(proposal.singletonUnionCoverage, os,
                          [&os](CanonicalDemandId id) { os << 'd' << id; });
    os << "] grounded=[";
    llvm::interleaveComma(proposal.groundedCoverage, os,
                          [&os](CanonicalDemandId id) { os << 'd' << id; });
    os << "] additional=[";
    llvm::interleaveComma(proposal.additionalCoverage, os,
                          [&os](CanonicalDemandId id) { os << 'd' << id; });
    os << "] admitted=" << (proposal.admitted ? "yes" : "no") << '\n';
  }
  if (program.getSetCoverInstance()) {
    const CanonicalSetCoverInstance &instance = *program.getSetCoverInstance();
    const bool grouped = llvm::any_of(
        instance.candidates, [](const CanonicalSetCoverCandidate &candidate) {
          return candidate.structuralProposal.has_value();
        });
    os << "SET-COVER optimization="
       << (grouped ? "grounded-groups" : "singleton") << " baseline=[";
    llvm::interleaveComma(instance.baseline, os,
                          [&os](CanonicalMechanismId id) { os << 'm' << id; });
    os << "] universe=[";
    llvm::interleaveComma(instance.universe, os,
                          [&os](CanonicalDemandId id) { os << 'd' << id; });
    os << "] candidates=" << instance.candidates.size() << '\n';
    for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
      os << "  candidate=c" << candidate.id << " mechanisms=[";
      llvm::interleaveComma(
          candidate.mechanisms, os,
          [&os](CanonicalMechanismId id) { os << 'm' << id; });
      os << "] weight=" << candidate.weight << " direct-origins=[";
      llvm::interleaveComma(candidate.directOrigins, os,
                            [&os](CanonicalDemandId id) { os << 'd' << id; });
      os << "] additional=[";
      llvm::interleaveComma(candidate.additionalCoverage, os,
                            [&os](CanonicalDemandId id) { os << 'd' << id; });
      os << ']';
      if (candidate.structuralProposal) {
        os << " proposal=g" << *candidate.structuralProposal;
      }
      os << '\n';
    }
  }
  if (program.getSetCoverSolution()) {
    const CanonicalSetCoverSolution &solution = *program.getSetCoverSolution();
    const bool grouped =
        program.getSetCoverInstance() &&
        llvm::any_of(program.getSetCoverInstance()->candidates,
                     [](const CanonicalSetCoverCandidate &candidate) {
                       return candidate.structuralProposal.has_value();
                     });
    os << "OPTIMIZATION enabled=yes mode="
       << (grouped ? "grouped-greedy" : "singleton-greedy") << " greedy=[";
    llvm::interleaveComma(
        solution.greedyCandidates, os,
        [&os](CanonicalSetCoverCandidateId id) { os << 'c' << id; });
    os << "] proposed=[";
    llvm::interleaveComma(solution.mechanisms, os,
                          [&os](CanonicalMechanismId id) { os << 'm' << id; });
    os << "] reverse-deleted=[";
    llvm::interleaveComma(solution.reverseDeleted, os,
                          [&os](CanonicalMechanismId id) { os << 'm' << id; });
    os << "] weight=" << solution.weight
       << " coverage-verified=" << (solution.coverageVerified ? "yes" : "no")
       << '\n';
    os << "PLAN selected=[";
    llvm::interleaveComma(solution.mechanisms, os,
                          [&os](CanonicalMechanismId id) { os << 'm' << id; });
    os << "] events=[";
    bool firstEvent = true;
    for (CanonicalMechanismId id : solution.mechanisms) {
      const CanonicalMechanism &mechanism = program.getMechanism(id);
      if (mechanism.kind == CanonicalMechanismKind::PeriodicOwnership &&
          mechanism.ownershipProtocol) {
        const CanonicalOwnershipProtocol &protocol =
            program.getOwnershipProtocol(*mechanism.ownershipProtocol);
        for (auto [laneIndex, lane] : llvm::enumerate(protocol.lanes)) {
          if (!lane.readyEventId || !lane.releaseEventId) {
            continue;
          }
          if (!firstEvent) {
            os << ", ";
          }
          firstEvent = false;
          os << 'm' << id << ".lane" << laneIndex << "=ready-e"
             << *lane.readyEventId << "/release-e" << *lane.releaseEventId;
        }
        continue;
      }
      const std::optional<unsigned> eventId = mechanism.eventId;
      if (!eventId) {
        continue;
      }
      if (!firstEvent) {
        os << ", ";
      }
      firstEvent = false;
      os << 'm' << id << "=e" << *eventId;
      if (mechanism.releaseEventId) {
        os << "/release-e" << *mechanism.releaseEventId;
      }
    }
    os << "] scarcity=[";
    llvm::interleaveComma(
        solution.scarcityEventGroups, os,
        [&os](const CanonicalScarcityEventGroup &group) {
          os << (group.kind == CanonicalScarcityEventKind::Coalesced
                     ? "coalesced{"
                     : "serialized{");
          llvm::interleaveComma(
              group.members, os,
              [&os](CanonicalMechanismId id) { os << 'm' << id; });
          os << "}:e" << group.eventId;
          if (group.releaseEventId) {
            os << "/release-e" << *group.releaseEventId;
          }
        });
    os << "]\n";
  }
}
