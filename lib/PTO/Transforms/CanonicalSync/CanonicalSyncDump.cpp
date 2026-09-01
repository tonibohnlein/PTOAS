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
     << '\n';
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
    }
    os << '\n';
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
    os << '\n';
  }
  if (program.getSetCoverInstance()) {
    const CanonicalSetCoverInstance &instance = *program.getSetCoverInstance();
    os << "SET-COVER optimization=singleton baseline=[";
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
      os << "]\n";
    }
  }
  if (program.getSetCoverSolution()) {
    const CanonicalSetCoverSolution &solution = *program.getSetCoverSolution();
    os << "OPTIMIZATION enabled=yes mode=singleton-greedy"
       << " greedy=[";
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
      const std::optional<unsigned> eventId = program.getMechanism(id).eventId;
      if (!eventId) {
        continue;
      }
      if (!firstEvent) {
        os << ", ";
      }
      firstEvent = false;
      os << 'm' << id << "=e" << *eventId;
      if (program.getMechanism(id).releaseEventId) {
        os << "/release-e" << *program.getMechanism(id).releaseEventId;
      }
    }
    os << "]\n";
  }
}
