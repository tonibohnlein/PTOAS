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
    os << " owner=r" << demand.owner << " guard=";
    printGuard(os, demand.guard);
    os << " distance=[";
    llvm::interleaveComma(demand.iterationDistance, os,
                          [&os](int64_t distance) {
                            if (distance == kCanonicalAnyPositiveDistance) {
                              os << "+";
                            } else {
                              os << distance;
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
       << " kind=" << stringifyCanonicalMechanismKind(mechanism.kind)
       << " resource=";
    printResource(os, mechanism.source);
    os << "->";
    printResource(os, mechanism.target);
    if (mechanism.kind == CanonicalMechanismKind::TailBarrier) {
      os << " cuts=exit";
    } else {
      os << " cuts=p" << mechanism.sourceCut << ":p" << mechanism.targetCut;
    }
    if (mechanism.eventId) {
      os << " event=" << *mechanism.eventId;
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
    os << "]\n";
  }
  os << "PLAN mechanical mechanisms=" << program.getMechanisms().size() << '\n';
}
