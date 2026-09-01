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

bool applyFixedFence(const CanonicalSyncProgram &program,
                     const CanonicalMechanism &mechanism,
                     SmallVectorImpl<CompletionFact> &facts,
                     ArrayRef<CanonicalRegionId> requiredLoops = {}) {
  const CanonicalFenceEffect &effect =
      program.getFenceEffect(*mechanism.fenceEffect);
  SmallVector<CanonicalPhysicalResource, 8> destinations;
  for (const CanonicalPhase &phase : program.getPhases()) {
    if (!llvm::is_contained(destinations, phase.resource)) {
      destinations.push_back(phase.resource);
    }
  }
  bool changed = false;
  const auto publish = [&](CanonicalPhaseId phase,
                           ArrayRef<CanonicalControlAtom> guard,
                           ArrayRef<CanonicalRegionId> loops) {
    for (CanonicalPhysicalResource destination : destinations) {
      changed |= addFact(
          facts,
          {phase, destination, mechanism.targetPoint,
           SmallVector<CanonicalControlAtom, 2>(guard.begin(), guard.end()),
           SmallVector<CanonicalRegionId, 2>(loops.begin(), loops.end())});
    }
  };
  for (const CanonicalPhase &phase : program.getPhases()) {
    const bool drained =
        llvm::is_contained(effect.drainedResources, phase.resource);
    const bool precedes = phaseMayPrecedePoint(phase, mechanism.sourcePoint);
    const bool compatible =
        controlsCanCoexecute(phase.controlPath, mechanism.guard);
    if (!drained || !precedes || !compatible) {
      continue;
    }
    publish(phase.id, combineGuards(phase.controlPath, mechanism.guard),
            requiredLoops);
  }
  const SmallVector<CompletionFact, 16> snapshot(facts.begin(), facts.end());
  for (const CompletionFact &fact : snapshot) {
    const bool drained =
        llvm::is_contained(effect.drainedResources, fact.resource);
    const bool precedes =
        programPointMustPrecede(fact.availableAt, mechanism.sourcePoint);
    const bool compatible = controlsCanCoexecute(fact.guard, mechanism.guard);
    if (!drained || !precedes || !compatible) {
      continue;
    }
    SmallVector<CanonicalRegionId, 2> loops = fact.requiredLoops;
    appendUnique(loops, requiredLoops);
    publish(fact.phase, combineGuards(fact.guard, mechanism.guard), loops);
  }
  return changed;
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

void addFixedFenceTransfers(
    const CanonicalSyncProgram &program, const CanonicalMechanism &mechanism,
    SmallVectorImpl<CanonicalBoundaryTransfer> &transfers,
    ArrayRef<CanonicalRegionId> requiredLoops = {}) {
  const CanonicalFenceEffect &effect =
      program.getFenceEffect(*mechanism.fenceEffect);
  SmallVector<CanonicalPhysicalResource, 8> destinations;
  for (const CanonicalPhase &phase : program.getPhases()) {
    if (!llvm::is_contained(destinations, phase.resource)) {
      destinations.push_back(phase.resource);
    }
  }
  for (CanonicalPhysicalResource source : effect.drainedResources) {
    for (CanonicalPhysicalResource target : destinations) {
      CanonicalBoundaryTransfer transfer;
      transfer.source = source;
      transfer.target = target;
      transfer.sourcePoint = mechanism.sourcePoint;
      transfer.targetPoint = mechanism.targetPoint;
      transfer.guard = mechanism.guard;
      transfer.requiredLoops.assign(requiredLoops.begin(), requiredLoops.end());
      addBoundaryTransfer(transfers, std::move(transfer));
    }
  }
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
      if (mechanism.kind == CanonicalMechanismKind::Event ||
          mechanism.kind == CanonicalMechanismKind::CrossCoreEvent ||
          (mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
           !mechanism.boundaryRecurring)) {
        changed |= applyEvent(program, mechanism, facts,
                              mechanismExecutionLoops(program, mechanism));
      } else if (mechanism.kind == CanonicalMechanismKind::FixedFence) {
        changed |= applyFixedFence(program, mechanism, facts,
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
    } else if (mechanism.kind == CanonicalMechanismKind::FixedFence) {
      addFixedFenceTransfers(program, mechanism, result.transfers,
                             mechanismExecutionLoops(program, mechanism));
    } else if (mechanism.kind == CanonicalMechanismKind::Event ||
               mechanism.kind == CanonicalMechanismKind::CrossCoreEvent ||
               (mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
                !mechanism.boundaryRecurring)) {
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

bool executionLoopsImpliedByPhase(ArrayRef<CanonicalRegionId> executionLoops,
                                  const CanonicalPhase &phase) {
  return llvm::all_of(executionLoops, [&](CanonicalRegionId loop) {
    return llvm::is_contained(phase.loopPath, loop);
  });
}

bool recurrenceCoveredByCompletionCut(const CanonicalSyncProgram &program,
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
    const bool recurringEvent =
        mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
        !mechanism.boundaryRecurring &&
        mechanism.recurrenceLoop == carrying->loop &&
        mechanism.source == source.resource &&
        mechanism.target == target.resource;
    if (recurringEvent) {
      // The set in iteration i+1 is ordered after every earlier command on
      // its source pipeline, including commands following the set cut in
      // iteration i.  Its wait therefore imports the complete preceding
      // iteration prefix into the target pipeline.  Alternatively, a source
      // captured by the channel in its own iteration remains complete on the
      // FIFO-ordered target pipeline for a later target iteration.  These are
      // the target-side and source-side loop-wrap transfers respectively.
      const bool sameIterationImport =
          phaseMayPrecedePoint(source, mechanism.sourcePoint) &&
          guardImplies(demand.sourceGuard, mechanism.guard);
      const bool nextIterationImport =
          guardImplies(demand.targetGuard, mechanism.guard) &&
          pointMustPrecedePhase(mechanism.targetPoint, target);
      return executionLoopsImpliedByPhase(loops, source) &&
             executionLoopsImpliedByPhase(loops, target) &&
             (sameIterationImport || nextIterationImport);
    }
    const bool recurringRelease =
        mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
        !mechanism.boundaryRecurring &&
        mechanism.recurrenceLoop == carrying->loop &&
        mechanism.target == source.resource &&
        mechanism.source == target.resource;
    if (recurringRelease) {
      // Release ownership is consumed at the loop header and replenished at
      // the latch, outside any branch-local ready cut.  It therefore orders
      // every source-pipeline prefix against the next iteration's target
      // pipeline, including transitions between opposite choice arms.
      return executionLoopsImpliedByPhase(loops, source) &&
             executionLoopsImpliedByPhase(loops, target);
    }
    const bool pipeBarrier =
        mechanism.kind == CanonicalMechanismKind::PipeBarrier &&
        mechanism.source == source.resource &&
        source.resource == target.resource;
    const bool fixedFence =
        mechanism.kind == CanonicalMechanismKind::FixedFence &&
        llvm::is_contained(
            program.getFenceEffect(*mechanism.fenceEffect).drainedResources,
            source.resource);
    const bool relevant = (pipeBarrier || fixedFence) &&
                          llvm::is_contained(loops, carrying->loop);
    if (!relevant) {
      return false;
    }
    // A repeated completion cut can close the loop wrap by completing the
    // source later in iteration i, or by draining iteration i before the
    // target in iteration i+1. A cut nested in another loop is usable only
    // when that loop also contains the protected endpoint; otherwise the
    // nested loop may execute zero times while the endpoint still executes.
    const bool completesAfterSource =
        executionLoopsImpliedByPhase(loops, source) &&
        phaseMayPrecedePoint(source, mechanism.targetPoint) &&
        guardImplies(demand.sourceGuard, mechanism.guard);
    const bool completesBeforeTarget =
        executionLoopsImpliedByPhase(loops, target) &&
        pointMustPrecedePhase(mechanism.targetPoint, target) &&
        guardImplies(demand.targetGuard, mechanism.guard);
    return completesAfterSource || completesBeforeTarget;
  });
}

bool demandCovered(const CanonicalSyncProgram &program,
                   const CanonicalDemand &demand,
                   ArrayRef<CanonicalMechanismId> selected,
                   ArrayRef<CompletionFact> facts,
                   const CanonicalSyncTarget &targetModel) {
  if (demand.kind == CanonicalDemandKind::ExitCompletion) {
    const CanonicalMechanismId direct =
        program.getDirectMechanisms()[demand.id];
    return llvm::is_contained(selected, direct);
  }
  if (demand.requirement == CanonicalRequirement::Visibility) {
    const CanonicalMechanismId direct =
        program.getDirectMechanisms()[demand.id];
    return llvm::is_contained(selected, direct);
  }
  const CanonicalMechanismId direct = program.getDirectMechanisms()[demand.id];
  const bool recurringSelected = llvm::is_contained(selected, direct) &&
                                 program.getMechanism(direct).kind ==
                                     CanonicalMechanismKind::RecurringEvent;
  if (recurringSelected) {
    return true;
  }
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &target = program.getPhase(demand.target);
  const bool intrinsicCompletion =
      (source.resource.core == target.resource.core &&
       getVPTOSchedulingSemantics(source.operation).completionIsSynchronous) ||
      (source.resource == target.resource &&
       targetModel.hasIntrinsicCompletion(source.resource));
  if (intrinsicCompletion) {
    return true;
  }
  if (recurrenceCoveredByCompletionCut(program, demand, selected)) {
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
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(program.getFunction());
  if (failed(target)) {
    return covered;
  }
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (demandCovered(program, demand, selected, facts, *target)) {
      covered.push_back(demand.id);
    }
  }
  return covered;
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

FailureOr<CanonicalCoverageWorld>
mlir::pto::canonical_sync_detail::evaluateCanonicalSyncGroup(
    const CanonicalSyncProgram &program, StringRef name,
    ArrayRef<CanonicalMechanismId> selected) {
  if (llvm::any_of(selected, [&program](CanonicalMechanismId id) {
        return id >= program.getMechanisms().size();
      })) {
    program.getFunction().emitError(
        "canonical sync coverage group references an invalid mechanism");
    return failure();
  }
  CanonicalCoverageWorld world = evaluateWorld(program, name, selected);
  const bool unrolledMismatch =
      world.unrolledOracleExhaustive && !world.unrolledOracleMatched;
  if (world.flattenedOracleMatched && world.unrolledOracleAvailable &&
      !unrolledMismatch) {
    return world;
  }

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
  FailureOr<CanonicalUnrolledCoverageResult> diagnosticUnrolled =
      evaluateCanonicalSyncUnrolledOracle(program, world.mechanisms);
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
    diagnostic << ",direct=m" << program.getDirectMechanisms()[demand]
               << ",summary="
               << (llvm::is_contained(world.covered, demand) ? "yes" : "no")
               << ",unrolled="
               << (succeeded(diagnosticUnrolled) &&
                           llvm::is_contained(diagnosticUnrolled->covered,
                                              demand)
                       ? "yes"
                       : "no")
               << ']';
    if (!world.summaries.empty()) {
      for (const CompletionFact &fact : world.summaries.back().completions) {
        if (fact.phase == record.source) {
          diagnostic << " fact[p" << fact.phase << '@'
                     << stringifyCanonicalCore(fact.resource.core) << ':'
                     << stringifyPIPE(fact.resource.pipe) << ']';
        }
      }
    }
  }
  for (CanonicalMechanismId mechanismId : world.mechanisms) {
    const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
    diagnostic << " using m" << mechanismId << '['
               << stringifyCanonicalMechanismKind(mechanism.kind) << ':'
               << stringifyCanonicalCore(mechanism.source.core) << ':'
               << stringifyPIPE(mechanism.source.pipe) << "->"
               << stringifyCanonicalCore(mechanism.target.core) << ':'
               << stringifyPIPE(mechanism.target.pipe) << ",origins=";
    for (CanonicalDemandId origin : mechanism.origins) {
      const CanonicalDemand &originDemand = program.getDemand(origin);
      diagnostic << 'd' << origin << ":p" << originDemand.source << "->p"
                 << originDemand.target << ',';
    }
    diagnostic << "captures=";
    for (const CanonicalPhase &phase : program.getPhases()) {
      if (phase.resource == mechanism.source &&
          phaseMayPrecedePoint(phase, mechanism.sourcePoint)) {
        diagnostic << 'p' << phase.id << ',';
      }
    }
    diagnostic << ']';
  }
  return failure();
}

LogicalResult
mlir::pto::evaluateCanonicalSyncCoverage(CanonicalSyncProgram &program) {
  const bool invalidState = !program.isGraphFrozen() || program.isFrozen() ||
                            program.getSetCoverInstance().has_value() ||
                            !program.mechanismCatalogComplete ||
                            program.coverageCatalogComplete;
  if (invalidState) {
    return program.getFunction().emitError(
        "canonical sync coverage requires an unsealed mechanism catalog");
  }
  const auto appendGroup = [&program](StringRef name,
                                      ArrayRef<CanonicalMechanismId> selected) {
    FailureOr<CanonicalCoverageWorld> world =
        canonical_sync_detail::evaluateCanonicalSyncGroup(program, name,
                                                          selected);
    if (failed(world)) {
      return failure();
    }
    program.appendCoverageWorld(std::move(*world));
    return success();
  };

  SmallVector<CanonicalMechanismId, 8> baseline;
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    if (mechanism.kind == CanonicalMechanismKind::IntrinsicOrder ||
        mechanism.kind == CanonicalMechanismKind::FixedFence ||
        mechanism.kind == CanonicalMechanismKind::TailBarrier) {
      baseline.push_back(mechanism.id);
    }
  }
  if (failed(appendGroup("baseline", baseline))) {
    return failure();
  }
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    if (llvm::is_contained(baseline, mechanism.id)) {
      continue;
    }
    SmallVector<CanonicalMechanismId, 8> singleton = baseline;
    singleton.push_back(mechanism.id);
    const std::string name = (Twine("singleton-m") + Twine(mechanism.id)).str();
    if (failed(appendGroup(name, singleton))) {
      return failure();
    }
  }

  program.coverageCatalogComplete = true;
  return success();
}
