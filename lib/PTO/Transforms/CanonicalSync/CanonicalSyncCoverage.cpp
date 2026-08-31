// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

struct CompletionFact {
  CanonicalPhaseId phase = kInvalidCanonicalSyncId;
  CanonicalPhysicalResource resource;
  CanonicalProgramPoint availableAt;
  SmallVector<CanonicalControlAtom, 2> guard;
};

bool guardImplies(ArrayRef<CanonicalControlAtom> execution,
                  ArrayRef<CanonicalControlAtom> required) {
  return llvm::all_of(required, [execution](const CanonicalControlAtom &atom) {
    return llvm::is_contained(execution, atom);
  });
}

SmallVector<CanonicalControlAtom, 2>
combineGuards(ArrayRef<CanonicalControlAtom> first,
              ArrayRef<CanonicalControlAtom> second) {
  if (!controlsCanCoexecute(first, second)) {
    return {};
  }
  return conjoinCompatibleControlPaths(first, second);
}

bool addFact(SmallVectorImpl<CompletionFact> &facts, CompletionFact fact) {
  for (CompletionFact &existing : facts) {
    if (existing.phase == fact.phase && existing.resource == fact.resource &&
        existing.guard == fact.guard) {
      if (existing.availableAt == fact.availableAt ||
          programPointMustPrecede(existing.availableAt, fact.availableAt)) {
        return false;
      }
      if (programPointMustPrecede(fact.availableAt, existing.availableAt)) {
        existing.availableAt = fact.availableAt;
        return true;
      }
    }
  }
  facts.push_back(std::move(fact));
  return true;
}

void applyBarrier(const CanonicalSyncProgram &program,
                  const CanonicalMechanism &mechanism,
                  SmallVectorImpl<CompletionFact> &facts) {
  for (const CanonicalPhase &phase : program.getPhases()) {
    if (phase.resource != mechanism.source ||
        !phaseMayPrecedePoint(phase, mechanism.sourcePoint) ||
        !controlsCanCoexecute(phase.controlPath, mechanism.guard)) {
      continue;
    }
    addFact(facts, {phase.id, mechanism.source, mechanism.targetPoint,
                    combineGuards(phase.controlPath, mechanism.guard)});
  }
}

bool applyEvent(const CanonicalSyncProgram &program,
                const CanonicalMechanism &mechanism,
                SmallVectorImpl<CompletionFact> &facts) {
  bool changed = false;
  for (const CanonicalPhase &phase : program.getPhases()) {
    if (phase.resource != mechanism.source ||
        !phaseMayPrecedePoint(phase, mechanism.sourcePoint) ||
        !controlsCanCoexecute(phase.controlPath, mechanism.guard)) {
      continue;
    }
    changed |=
        addFact(facts, {phase.id, mechanism.target, mechanism.targetPoint,
                        combineGuards(phase.controlPath, mechanism.guard)});
  }
  SmallVector<CompletionFact, 16> snapshot(facts.begin(), facts.end());
  for (const CompletionFact &fact : snapshot) {
    if (fact.resource != mechanism.source ||
        !programPointMustPrecede(fact.availableAt, mechanism.sourcePoint) ||
        !controlsCanCoexecute(fact.guard, mechanism.guard)) {
      continue;
    }
    changed |=
        addFact(facts, {fact.phase, mechanism.target, mechanism.targetPoint,
                        combineGuards(fact.guard, mechanism.guard)});
  }
  return changed;
}

SmallVector<CompletionFact, 32>
evaluateFacts(const CanonicalSyncProgram &program,
              ArrayRef<CanonicalMechanismId> selected) {
  SmallVector<CompletionFact, 32> facts;
  for (CanonicalMechanismId id : selected) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (mechanism.kind == CanonicalMechanismKind::PipeBarrier) {
      applyBarrier(program, mechanism, facts);
    }
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (CanonicalMechanismId id : selected) {
      const CanonicalMechanism &mechanism = program.getMechanism(id);
      if (mechanism.kind == CanonicalMechanismKind::Event) {
        changed |= applyEvent(program, mechanism, facts);
      }
    }
  }
  return facts;
}

bool containsKind(const CanonicalSyncProgram &program,
                  ArrayRef<CanonicalMechanismId> selected,
                  CanonicalMechanismKind kind) {
  return llvm::any_of(selected, [&](CanonicalMechanismId id) {
    return program.getMechanism(id).kind == kind;
  });
}

bool recurrenceCoveredByBarrier(const CanonicalSyncProgram &program,
                                const CanonicalDemand &demand,
                                ArrayRef<CanonicalMechanismId> selected) {
  const bool positive = llvm::any_of(
      demand.iterationDistance, [](const CanonicalLoopDistance &distance) {
        return distance.relation == CanonicalIterationRelation::AnyPositive;
      });
  if (!positive) {
    return false;
  }
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &target = program.getPhase(demand.target);
  return llvm::any_of(selected, [&](CanonicalMechanismId id) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    return mechanism.kind == CanonicalMechanismKind::PipeBarrier &&
           mechanism.source == source.resource &&
           source.resource == target.resource &&
           mechanism.targetPoint ==
               CanonicalProgramPoint{target.operation,
                                     CanonicalProgramPointPosition::Before} &&
           guardImplies(demand.targetGuard, mechanism.guard);
  });
}

bool demandCovered(const CanonicalSyncProgram &program,
                   const CanonicalDemand &demand,
                   ArrayRef<CanonicalMechanismId> selected,
                   ArrayRef<CompletionFact> facts) {
  if (demand.kind == CanonicalDemandKind::ExitCompletion) {
    return containsKind(program, selected, CanonicalMechanismKind::TailBarrier);
  }
  if (demand.requirement == CanonicalRequirement::Visibility) {
    const CanonicalMechanismId direct =
        program.getDirectMechanisms()[demand.id];
    return llvm::is_contained(selected, direct);
  }
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &target = program.getPhase(demand.target);
  if (source.resource == target.resource) {
    FailureOr<CanonicalSyncTarget> model =
        CanonicalSyncTarget::resolve(program.getFunction());
    const bool intrinsicallyComplete =
        succeeded(model) && model->hasIntrinsicCompletion(source.resource);
    if (intrinsicallyComplete) {
      return true;
    }
  }
  if (recurrenceCoveredByBarrier(program, demand, selected)) {
    return true;
  }
  if (!controlsCanCoexecute(demand.sourceGuard, demand.targetGuard)) {
    return false;
  }
  const SmallVector<CanonicalControlAtom, 2> executionGuard =
      conjoinCompatibleControlPaths(demand.sourceGuard, demand.targetGuard);
  return llvm::any_of(facts, [&](const CompletionFact &fact) {
    return fact.phase == demand.source && fact.resource == target.resource &&
           pointMustPrecedePhase(fact.availableAt, target) &&
           guardImplies(executionGuard, fact.guard);
  });
}

CanonicalCoverageWorld evaluateWorld(const CanonicalSyncProgram &program,
                                     StringRef name,
                                     ArrayRef<CanonicalMechanismId> selected) {
  CanonicalCoverageWorld world;
  world.name = name.str();
  world.mechanisms.assign(selected.begin(), selected.end());
  llvm::sort(world.mechanisms);
  world.mechanisms.erase(
      std::unique(world.mechanisms.begin(), world.mechanisms.end()),
      world.mechanisms.end());
  const SmallVector<CompletionFact, 32> facts =
      evaluateFacts(program, world.mechanisms);
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (demandCovered(program, demand, world.mechanisms, facts)) {
      world.covered.push_back(demand.id);
    }
  }
  return world;
}

} // namespace

LogicalResult
mlir::pto::evaluateCanonicalSyncCoverage(CanonicalSyncProgram &program) {
  SmallVector<CanonicalMechanismId, 8> baseline;
  SmallVector<CanonicalMechanismId, 16> all;
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    all.push_back(mechanism.id);
    if (mechanism.kind == CanonicalMechanismKind::IntrinsicOrder ||
        mechanism.kind == CanonicalMechanismKind::FixedFence) {
      baseline.push_back(mechanism.id);
    }
  }
  program.appendCoverageWorld(evaluateWorld(program, "baseline", baseline));
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    if (llvm::is_contained(baseline, mechanism.id)) {
      continue;
    }
    SmallVector<CanonicalMechanismId, 8> singleton = baseline;
    singleton.push_back(mechanism.id);
    program.appendCoverageWorld(evaluateWorld(
        program, (Twine("singleton-m") + Twine(mechanism.id)).str(),
        singleton));
  }
  CanonicalCoverageWorld final = evaluateWorld(program, "mechanical", all);
  const bool incomplete = final.covered.size() != program.getDemands().size();
  if (incomplete) {
    for (const CanonicalDemand &demand : program.getDemands()) {
      if (!llvm::is_contained(final.covered, demand.id)) {
        program.getFunction().emitError(
            "canonical sync mechanical plan does not cover demand d")
            << demand.id << " (" << stringifyCanonicalDemandKind(demand.kind)
            << ')';
        break;
      }
    }
    return failure();
  }
  program.appendCoverageWorld(std::move(final));
  return success();
}
