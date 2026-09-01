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

using CompletionFact = CanonicalCompletionTransfer;

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

template <typename T, typename Range>
void appendUnique(SmallVectorImpl<T> &destination, const Range &source) {
  for (const T &item : source) {
    if (!llvm::is_contained(destination, item)) {
      destination.push_back(item);
    }
  }
}

void canonicalizeLoops(SmallVectorImpl<CanonicalRegionId> &loops) {
  llvm::sort(loops);
  loops.erase(std::unique(loops.begin(), loops.end()), loops.end());
}

bool addFact(SmallVectorImpl<CompletionFact> &facts, CompletionFact fact) {
  canonicalizeLoops(fact.requiredLoops);
  for (CompletionFact &existing : facts) {
    const bool sameKey = existing.phase == fact.phase &&
                         existing.resource == fact.resource &&
                         existing.guard == fact.guard &&
                         existing.requiredLoops == fact.requiredLoops;
    if (!sameKey) {
      continue;
    }
    if (existing.availableAt == fact.availableAt ||
        programPointMustPrecede(existing.availableAt, fact.availableAt)) {
      return false;
    }
    if (programPointMustPrecede(fact.availableAt, existing.availableAt)) {
      existing.availableAt = fact.availableAt;
      return true;
    }
  }
  facts.push_back(std::move(fact));
  return true;
}

SmallVector<CanonicalRegionId, 2>
mechanismExecutionLoops(const CanonicalSyncProgram &program,
                        const CanonicalMechanism &mechanism) {
  SmallVector<CanonicalRegionId, 2> result;
  for (CanonicalRegionId region = mechanism.actionRegion;
       region != kInvalidCanonicalSyncId;
       region = program.getRegion(region).parent) {
    if (program.getRegion(region).kind == CanonicalRegionKind::Loop) {
      result.push_back(region);
    }
  }
  canonicalizeLoops(result);
  return result;
}

void applyBarrier(const CanonicalSyncProgram &program,
                  const CanonicalMechanism &mechanism,
                  SmallVectorImpl<CompletionFact> &facts,
                  ArrayRef<CanonicalRegionId> requiredLoops = {}) {
  for (const CanonicalPhase &phase : program.getPhases()) {
    if (phase.resource != mechanism.source ||
        !phaseMayPrecedePoint(phase, mechanism.sourcePoint) ||
        !controlsCanCoexecute(phase.controlPath, mechanism.guard)) {
      continue;
    }
    addFact(facts, {phase.id, mechanism.target, mechanism.targetPoint,
                    combineGuards(phase.controlPath, mechanism.guard),
                    SmallVector<CanonicalRegionId, 2>(requiredLoops.begin(),
                                                      requiredLoops.end())});
  }
}

bool applyEvent(const CanonicalSyncProgram &program,
                const CanonicalMechanism &mechanism,
                SmallVectorImpl<CompletionFact> &facts,
                ArrayRef<CanonicalRegionId> requiredLoops = {}) {
  bool changed = false;
  for (const CanonicalPhase &phase : program.getPhases()) {
    if (phase.resource != mechanism.source ||
        !phaseMayPrecedePoint(phase, mechanism.sourcePoint) ||
        !controlsCanCoexecute(phase.controlPath, mechanism.guard)) {
      continue;
    }
    changed |=
        addFact(facts, {phase.id, mechanism.target, mechanism.targetPoint,
                        combineGuards(phase.controlPath, mechanism.guard),
                        SmallVector<CanonicalRegionId, 2>(
                            requiredLoops.begin(), requiredLoops.end())});
  }
  const SmallVector<CompletionFact, 16> snapshot(facts.begin(), facts.end());
  for (const CompletionFact &fact : snapshot) {
    if (fact.resource != mechanism.source ||
        !programPointMustPrecede(fact.availableAt, mechanism.sourcePoint) ||
        !controlsCanCoexecute(fact.guard, mechanism.guard)) {
      continue;
    }
    SmallVector<CanonicalRegionId, 2> loops = fact.requiredLoops;
    appendUnique(loops, requiredLoops);
    changed |= addFact(
        facts, {fact.phase, mechanism.target, mechanism.targetPoint,
                combineGuards(fact.guard, mechanism.guard), std::move(loops)});
  }
  return changed;
}

bool sameBoundaryTransfer(const CanonicalBoundaryTransfer &first,
                          const CanonicalBoundaryTransfer &second) {
  return first.source == second.source && first.target == second.target &&
         first.sourcePoint == second.sourcePoint &&
         first.targetPoint == second.targetPoint &&
         first.guard == second.guard &&
         first.requiredLoops == second.requiredLoops;
}

bool addBoundaryTransfer(SmallVectorImpl<CanonicalBoundaryTransfer> &transfers,
                         CanonicalBoundaryTransfer transfer) {
  canonicalizeLoops(transfer.requiredLoops);
  auto existing =
      llvm::find_if(transfers, [&](CanonicalBoundaryTransfer &item) {
        return sameBoundaryTransfer(item, transfer);
      });
  if (existing == transfers.end()) {
    transfers.push_back(std::move(transfer));
    return true;
  }
  return false;
}

SmallVector<CanonicalControlAtom, 2>
withoutChoice(ArrayRef<CanonicalControlAtom> guard, CanonicalRegionId choice) {
  SmallVector<CanonicalControlAtom, 2> result;
  llvm::copy_if(guard, std::back_inserter(result),
                [choice](const CanonicalControlAtom &atom) {
                  return atom.choice != choice;
                });
  return result;
}

std::optional<unsigned> guardArm(ArrayRef<CanonicalControlAtom> guard,
                                 CanonicalRegionId choice) {
  auto atom = llvm::find_if(guard, [choice](const CanonicalControlAtom &item) {
    return item.choice == choice;
  });
  return atom == guard.end() ? std::nullopt
                             : std::optional<unsigned>(atom->arm);
}

bool joinFlattenedChoiceFacts(const CanonicalSyncProgram &program,
                              SmallVectorImpl<CompletionFact> &facts) {
  const SmallVector<CompletionFact, 32> snapshot(facts.begin(), facts.end());
  bool changed = false;
  for (const CanonicalRegion &choice : program.getRegions()) {
    if (choice.kind != CanonicalRegionKind::Choice || !choice.operation) {
      continue;
    }
    const CanonicalProgramPoint afterChoice{
        choice.operation, CanonicalProgramPointPosition::After};
    for (const CompletionFact &first : snapshot) {
      const bool firstIsUnavailable =
          guardArm(first.guard, choice.id) != 0U ||
          !programPointMustPrecede(first.availableAt, afterChoice);
      if (firstIsUnavailable) {
        continue;
      }
      const SmallVector<CanonicalControlAtom, 2> commonGuard =
          withoutChoice(first.guard, choice.id);
      for (const CompletionFact &second : snapshot) {
        const bool secondIsUnavailable =
            guardArm(second.guard, choice.id) != 1U ||
            first.phase != second.phase || first.resource != second.resource ||
            first.requiredLoops != second.requiredLoops ||
            commonGuard != withoutChoice(second.guard, choice.id) ||
            !programPointMustPrecede(second.availableAt, afterChoice);
        if (secondIsUnavailable) {
          continue;
        }
        changed |= addFact(facts, {first.phase, first.resource, afterChoice,
                                   commonGuard, first.requiredLoops});
      }
    }
  }
  return changed;
}

void joinChoiceTransfers(
    const CanonicalSyncProgram &program, const CanonicalRegion &choice,
    ArrayRef<CanonicalRegionSummary> children,
    SmallVectorImpl<CanonicalBoundaryTransfer> &transfers) {
  const CanonicalRegionSummary *arms[2] = {nullptr, nullptr};
  for (const CanonicalRegionSummary &child : children) {
    const unsigned arm = program.getRegion(child.region).arm;
    if (arm < 2U) {
      arms[arm] = &child;
    }
  }
  if (!arms[0] || !arms[1]) {
    return;
  }
  for (const CanonicalBoundaryTransfer &first : arms[0]->transfers) {
    const SmallVector<CanonicalControlAtom, 2> firstGuard =
        withoutChoice(first.guard, choice.id);
    auto second = llvm::find_if(
        arms[1]->transfers, [&](const CanonicalBoundaryTransfer &candidate) {
          return first.source == candidate.source &&
                 first.target == candidate.target &&
                 first.requiredLoops == candidate.requiredLoops &&
                 firstGuard == withoutChoice(candidate.guard, choice.id);
        });
    if (second == arms[1]->transfers.end()) {
      continue;
    }
    CanonicalBoundaryTransfer joined;
    joined.source = first.source;
    joined.target = first.target;
    joined.sourcePoint = {choice.operation,
                          CanonicalProgramPointPosition::Before};
    joined.targetPoint = {choice.operation,
                          CanonicalProgramPointPosition::After};
    joined.guard = firstGuard;
    joined.requiredLoops = first.requiredLoops;
    addBoundaryTransfer(transfers, std::move(joined));
  }
}

void joinChoiceCompletions(const CanonicalRegion &choice,
                           ArrayRef<CanonicalRegionSummary> children,
                           SmallVectorImpl<CompletionFact> &completions,
                           const CanonicalSyncProgram &program) {
  const CanonicalRegionSummary *arms[2] = {nullptr, nullptr};
  for (const CanonicalRegionSummary &child : children) {
    const unsigned arm = program.getRegion(child.region).arm;
    if (arm < 2U) {
      arms[arm] = &child;
    }
  }
  if (!arms[0] || !arms[1]) {
    return;
  }
  const CanonicalProgramPoint afterChoice{choice.operation,
                                          CanonicalProgramPointPosition::After};
  for (const CompletionFact &first : arms[0]->completions) {
    const SmallVector<CanonicalControlAtom, 2> commonGuard =
        withoutChoice(first.guard, choice.id);
    for (const CompletionFact &second : arms[1]->completions) {
      const bool sameCompletion =
          first.phase == second.phase && first.resource == second.resource &&
          first.requiredLoops == second.requiredLoops &&
          commonGuard == withoutChoice(second.guard, choice.id);
      const bool bothAvailable =
          programPointMustPrecede(first.availableAt, afterChoice) &&
          programPointMustPrecede(second.availableAt, afterChoice);
      if (!sameCompletion || !bothAvailable) {
        continue;
      }
      addFact(completions, {first.phase, first.resource, afterChoice,
                            commonGuard, first.requiredLoops});
    }
  }
}

SmallVector<CompletionFact, 32>
evaluateFlattenedFacts(const CanonicalSyncProgram &program,
                       ArrayRef<CanonicalMechanismId> selected) {
  SmallVector<CompletionFact, 32> facts;
  for (CanonicalMechanismId id : selected) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (mechanism.kind == CanonicalMechanismKind::PipeBarrier) {
      applyBarrier(program, mechanism, facts,
                   mechanismExecutionLoops(program, mechanism));
    }
  }
  bool changed = true;
  while (changed) {
    changed = joinFlattenedChoiceFacts(program, facts);
    for (CanonicalMechanismId id : selected) {
      const CanonicalMechanism &mechanism = program.getMechanism(id);
      if (mechanism.kind == CanonicalMechanismKind::Event) {
        changed |= applyEvent(program, mechanism, facts,
                              mechanismExecutionLoops(program, mechanism));
      }
    }
  }
  return facts;
}

bool composeBoundaryTransfers(
    SmallVectorImpl<CanonicalBoundaryTransfer> &transfers) {
  const SmallVector<CanonicalBoundaryTransfer, 16> snapshot(transfers.begin(),
                                                            transfers.end());
  bool changed = false;
  for (const CanonicalBoundaryTransfer &first : snapshot) {
    for (const CanonicalBoundaryTransfer &second : snapshot) {
      if (first.target != second.source ||
          !programPointMustPrecede(first.targetPoint, second.sourcePoint) ||
          !controlsCanCoexecute(first.guard, second.guard)) {
        continue;
      }
      CanonicalBoundaryTransfer composed;
      composed.source = first.source;
      composed.target = second.target;
      composed.sourcePoint = first.sourcePoint;
      composed.targetPoint = second.targetPoint;
      composed.guard = combineGuards(first.guard, second.guard);
      composed.requiredLoops = first.requiredLoops;
      appendUnique(composed.requiredLoops, second.requiredLoops);
      changed |= addBoundaryTransfer(transfers, std::move(composed));
    }
  }
  return changed;
}

bool applyBoundaryTransfer(const CanonicalSyncProgram &program,
                           const CanonicalBoundaryTransfer &transfer,
                           SmallVectorImpl<CompletionFact> &facts) {
  bool changed = false;
  for (const CanonicalPhase &phase : program.getPhases()) {
    if (phase.resource != transfer.source ||
        !phaseMayPrecedePoint(phase, transfer.sourcePoint) ||
        !controlsCanCoexecute(phase.controlPath, transfer.guard)) {
      continue;
    }
    changed |= addFact(facts, {phase.id, transfer.target, transfer.targetPoint,
                               combineGuards(phase.controlPath, transfer.guard),
                               transfer.requiredLoops});
  }
  const SmallVector<CompletionFact, 16> snapshot(facts.begin(), facts.end());
  for (const CompletionFact &fact : snapshot) {
    if (fact.resource != transfer.source ||
        !programPointMustPrecede(fact.availableAt, transfer.sourcePoint) ||
        !controlsCanCoexecute(fact.guard, transfer.guard)) {
      continue;
    }
    SmallVector<CanonicalRegionId, 2> loops = fact.requiredLoops;
    appendUnique(loops, transfer.requiredLoops);
    changed |= addFact(
        facts, {fact.phase, transfer.target, transfer.targetPoint,
                combineGuards(fact.guard, transfer.guard), std::move(loops)});
  }
  return changed;
}

CanonicalRegionSummary
summarizeRegion(const CanonicalSyncProgram &program, CanonicalRegionId region,
                ArrayRef<CanonicalMechanismId> selected,
                SmallVectorImpl<CanonicalRegionSummary> &summaries) {
  CanonicalRegionSummary result;
  result.region = region;
  SmallVector<CanonicalRegionSummary, 4> childSummaries;
  for (const CanonicalRegion &child : program.getRegions()) {
    if (child.parent != region) {
      continue;
    }
    CanonicalRegionSummary childSummary =
        summarizeRegion(program, child.id, selected, summaries);
    result.children.push_back(child.id);
    childSummaries.push_back(childSummary);
    for (const CompletionFact &fact : childSummary.completions) {
      CompletionFact imported = fact;
      if (program.getRegion(region).kind == CanonicalRegionKind::Loop &&
          !llvm::is_contained(imported.requiredLoops, region)) {
        imported.requiredLoops.push_back(region);
      }
      addFact(result.completions, std::move(imported));
    }
    for (const CanonicalBoundaryTransfer &transfer : childSummary.transfers) {
      CanonicalBoundaryTransfer imported = transfer;
      if (program.getRegion(region).kind == CanonicalRegionKind::Loop &&
          !llvm::is_contained(imported.requiredLoops, region)) {
        imported.requiredLoops.push_back(region);
      }
      addBoundaryTransfer(result.transfers, std::move(imported));
    }
  }
  if (program.getRegion(region).kind == CanonicalRegionKind::Choice) {
    joinChoiceCompletions(program.getRegion(region), childSummaries,
                          result.completions, program);
    joinChoiceTransfers(program, program.getRegion(region), childSummaries,
                        result.transfers);
  }
  for (CanonicalMechanismId id : selected) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (mechanism.actionRegion != region) {
      continue;
    }
    if (mechanism.kind == CanonicalMechanismKind::PipeBarrier) {
      applyBarrier(program, mechanism, result.completions,
                   mechanismExecutionLoops(program, mechanism));
    } else if (mechanism.kind == CanonicalMechanismKind::Event) {
      CanonicalBoundaryTransfer transfer;
      transfer.source = mechanism.source;
      transfer.target = mechanism.target;
      transfer.sourcePoint = mechanism.sourcePoint;
      transfer.targetPoint = mechanism.targetPoint;
      transfer.guard = mechanism.guard;
      transfer.requiredLoops = mechanismExecutionLoops(program, mechanism);
      addBoundaryTransfer(result.transfers, std::move(transfer));
    }
  }
  while (composeBoundaryTransfers(result.transfers)) {
  }
  bool factsChanged = true;
  while (factsChanged) {
    factsChanged = false;
    for (const CanonicalBoundaryTransfer &transfer : result.transfers) {
      factsChanged |=
          applyBoundaryTransfer(program, transfer, result.completions);
    }
  }
  summaries.push_back(result);
  return result;
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
  const auto carrying = llvm::find_if(
      demand.iterationDistance, [](const CanonicalLoopDistance &distance) {
        return distance.relation == CanonicalIterationRelation::AnyPositive;
      });
  if (carrying == demand.iterationDistance.end()) {
    return false;
  }
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &target = program.getPhase(demand.target);
  return llvm::any_of(selected, [&](CanonicalMechanismId id) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    const SmallVector<CanonicalRegionId, 2> loops =
        mechanismExecutionLoops(program, mechanism);
    const bool relevant =
        mechanism.kind == CanonicalMechanismKind::PipeBarrier &&
        mechanism.source == source.resource &&
        source.resource == target.resource &&
        llvm::is_contained(loops, carrying->loop);
    if (!relevant) {
      return false;
    }
    // A repeated barrier can close the loop wrap by completing the source
    // later in iteration i, or by draining iteration i before the target in
    // iteration i+1. Either cut orders the carried dependence.
    const bool completesAfterSource =
        phaseMayPrecedePoint(source, mechanism.targetPoint) &&
        guardImplies(demand.sourceGuard, mechanism.guard);
    const bool completesBeforeTarget =
        pointMustPrecedePhase(mechanism.targetPoint, target) &&
        guardImplies(demand.targetGuard, mechanism.guard);
    return completesAfterSource || completesBeforeTarget;
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
    const bool intrinsicCompletion =
        succeeded(model) && model->hasIntrinsicCompletion(source.resource);
    if (intrinsicCompletion) {
      return true;
    }
  }
  if (recurrenceCoveredByBarrier(program, demand, selected)) {
    return true;
  }
  const bool positive = llvm::any_of(
      demand.iterationDistance, [](const CanonicalLoopDistance &distance) {
        return distance.relation == CanonicalIterationRelation::AnyPositive;
      });
  if (positive ||
      !controlsCanCoexecute(demand.sourceGuard, demand.targetGuard)) {
    return false;
  }
  const SmallVector<CanonicalControlAtom, 2> executionGuard =
      conjoinCompatibleControlPaths(demand.sourceGuard, demand.targetGuard);
  return llvm::any_of(facts, [&](const CompletionFact &fact) {
    const bool requiredExecution =
        llvm::all_of(fact.requiredLoops, [&](CanonicalRegionId loop) {
          return llvm::is_contained(source.loopPath, loop) ||
                 llvm::is_contained(target.loopPath, loop);
        });
    return fact.phase == demand.source && fact.resource == target.resource &&
           pointMustPrecedePhase(fact.availableAt, target) &&
           guardImplies(executionGuard, fact.guard) && requiredExecution;
  });
}

SmallVector<CanonicalDemandId, 16>
coveredDemands(const CanonicalSyncProgram &program,
               ArrayRef<CanonicalMechanismId> selected,
               ArrayRef<CompletionFact> facts) {
  SmallVector<CanonicalDemandId, 16> covered;
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (demandCovered(program, demand, selected, facts)) {
      covered.push_back(demand.id);
    }
  }
  return covered;
}

std::optional<CanonicalRegionId>
sharedOppositeChoice(const CanonicalMechanism &first,
                     const CanonicalMechanism &second) {
  for (const CanonicalControlAtom &firstAtom : first.guard) {
    auto secondAtom =
        llvm::find_if(second.guard, [&](const CanonicalControlAtom &atom) {
          return atom.choice == firstAtom.choice && atom.arm != firstAtom.arm;
        });
    const bool incompatible = secondAtom == second.guard.end() ||
                              withoutChoice(first.guard, firstAtom.choice) !=
                                  withoutChoice(second.guard, firstAtom.choice);
    if (incompatible) {
      continue;
    }
    return firstAtom.choice;
  }
  return std::nullopt;
}

struct ChoiceGroupSignature {
  CanonicalRegionId choice = kInvalidCanonicalSyncId;
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  SmallVector<CanonicalControlAtom, 2> residualGuard;

  bool operator==(const ChoiceGroupSignature &other) const {
    return choice == other.choice && source == other.source &&
           target == other.target && residualGuard == other.residualGuard;
  }
};

SmallVector<SmallVector<CanonicalMechanismId, 2>, 4>
discoverChoiceGroups(const CanonicalSyncProgram &program,
                     ArrayRef<CanonicalMechanismId> baseline) {
  SmallVector<SmallVector<CanonicalMechanismId, 2>, 4> groups;
  SmallVector<ChoiceGroupSignature, 4> signatures;
  for (const CanonicalMechanism &first : program.getMechanisms()) {
    if (llvm::is_contained(baseline, first.id)) {
      continue;
    }
    for (const CanonicalMechanism &second :
         program.getMechanisms().drop_front(first.id + 1U)) {
      const std::optional<CanonicalRegionId> choice =
          sharedOppositeChoice(first, second);
      const bool incompatible = llvm::is_contained(baseline, second.id) ||
                                first.source != second.source ||
                                first.target != second.target || !choice;
      if (incompatible) {
        continue;
      }
      ChoiceGroupSignature signature;
      signature.choice = *choice;
      signature.source = first.source;
      signature.target = first.target;
      signature.residualGuard = withoutChoice(first.guard, *choice);
      if (llvm::is_contained(signatures, signature)) {
        continue;
      }
      signatures.push_back(std::move(signature));
      groups.push_back({first.id, second.id});
    }
  }
  return groups;
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

  CanonicalRegionSummary root =
      summarizeRegion(program, 0, world.mechanisms, world.summaries);
  const SmallVector<CanonicalDemandId, 16> summarized =
      coveredDemands(program, world.mechanisms, root.completions);
  world.covered.assign(summarized.begin(), summarized.end());

  const SmallVector<CompletionFact, 32> flattenedFacts =
      evaluateFlattenedFacts(program, world.mechanisms);
  const SmallVector<CanonicalDemandId, 16> flattened =
      coveredDemands(program, world.mechanisms, flattenedFacts);
  world.flattenedOracleMatched = summarized == flattened;

  FailureOr<CanonicalUnrolledCoverageResult> unrolled =
      evaluateCanonicalSyncUnrolledOracle(program, world.mechanisms);
  world.unrolledOracleAvailable = succeeded(unrolled);
  world.unrolledOracleExhaustive = succeeded(unrolled) && unrolled->exhaustive;
  world.unrolledOracleMatched =
      world.unrolledOracleExhaustive && summarized == unrolled->covered;
  if (!world.flattenedOracleMatched ||
      (world.unrolledOracleExhaustive && !world.unrolledOracleMatched)) {
    for (const CanonicalDemand &demand : program.getDemands()) {
      const bool summaryCovers = llvm::is_contained(summarized, demand.id);
      const bool flatCovers = llvm::is_contained(flattened, demand.id);
      const bool unrolledCovers =
          world.unrolledOracleExhaustive &&
          llvm::is_contained(unrolled->covered, demand.id);
      if (summaryCovers != flatCovers ||
          (world.unrolledOracleExhaustive && summaryCovers != unrolledCovers)) {
        world.differentialDisagreements.push_back(demand.id);
      }
    }
  }
  return world;
}

} // namespace

LogicalResult
mlir::pto::evaluateCanonicalSyncCoverage(CanonicalSyncProgram &program) {
  const auto appendChecked = [&program](CanonicalCoverageWorld world) {
    const bool unrolledMismatch =
        world.unrolledOracleExhaustive && !world.unrolledOracleMatched;
    if (!world.flattenedOracleMatched || !world.unrolledOracleAvailable ||
        unrolledMismatch) {
      InFlightDiagnostic diagnostic =
          program.getFunction().emitError(
              "canonical sync region summary disagrees with its differential "
              "coverage oracle in world '")
          << world.name
          << "' (flat=" << (world.flattenedOracleMatched ? "match" : "mismatch")
          << ", unrolled="
          << (!world.unrolledOracleExhaustive
                  ? "inconclusive"
                  : (world.unrolledOracleMatched ? "match" : "mismatch"))
          << ") for demand(s)";
      for (CanonicalDemandId demand : world.differentialDisagreements) {
        const CanonicalDemand &record = program.getDemand(demand);
        diagnostic << " d" << demand << '['
                   << stringifyCanonicalDemandKind(record.kind) << ":p"
                   << record.source << "->";
        if (record.target == kInvalidCanonicalSyncId) {
          diagnostic << "exit";
        } else {
          diagnostic << 'p' << record.target;
        }
        diagnostic << ']';
      }
      return failure();
    }
    program.appendCoverageWorld(std::move(world));
    return success();
  };

  SmallVector<CanonicalMechanismId, 8> baseline;
  SmallVector<CanonicalMechanismId, 16> all;
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    all.push_back(mechanism.id);
    if (mechanism.kind == CanonicalMechanismKind::IntrinsicOrder ||
        mechanism.kind == CanonicalMechanismKind::FixedFence) {
      baseline.push_back(mechanism.id);
    }
  }
  if (failed(appendChecked(evaluateWorld(program, "baseline", baseline)))) {
    return failure();
  }
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    if (llvm::is_contained(baseline, mechanism.id)) {
      continue;
    }
    SmallVector<CanonicalMechanismId, 8> singleton = baseline;
    singleton.push_back(mechanism.id);
    if (failed(appendChecked(evaluateWorld(
            program, (Twine("singleton-m") + Twine(mechanism.id)).str(),
            singleton)))) {
      return failure();
    }
  }

  for (const SmallVector<CanonicalMechanismId, 2> &group :
       discoverChoiceGroups(program, baseline)) {
    SmallVector<CanonicalMechanismId, 8> selected = baseline;
    selected.append(group);
    const std::string name =
        (Twine("choice-group-m") + Twine(group[0]) + "-m" + Twine(group[1]))
            .str();
    if (failed(appendChecked(evaluateWorld(program, name, selected)))) {
      return failure();
    }
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
  return appendChecked(std::move(final));
}
