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

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"

#include <unordered_map>

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

std::uint64_t resourceKey(CanonicalPhysicalResource resource) {
  return (static_cast<std::uint64_t>(resource.core) << 32U) |
         static_cast<std::uint32_t>(resource.pipe);
}

std::size_t hashFactKey(const CompletionFact &fact) {
  llvm::hash_code hash = llvm::hash_combine(
      fact.phase, static_cast<unsigned>(fact.resource.core),
      static_cast<unsigned>(fact.resource.pipe), fact.guard.size(),
      fact.requiredLoops.size());
  for (const CanonicalControlAtom &atom : fact.guard) {
    hash = llvm::hash_combine(hash, atom.choice, atom.arm);
  }
  for (CanonicalRegionId loop : fact.requiredLoops) {
    hash = llvm::hash_combine(hash, loop);
  }
  return static_cast<std::size_t>(hash);
}

bool sameFactKey(const CompletionFact &first, const CompletionFact &second) {
  return first.phase == second.phase && first.resource == second.resource &&
         first.guard == second.guard &&
         first.requiredLoops == second.requiredLoops;
}

class CompletionFactIndex {
public:
  CompletionFactIndex(SmallVectorImpl<CompletionFact> &facts,
                      CanonicalSyncStatistics *statistics)
      : facts(facts), statistics(statistics) {
    for (std::size_t index = 0; index < facts.size(); ++index) {
      buckets[hashFactKey(facts[index])].push_back(index);
      byResource[resourceKey(facts[index].resource)].push_back(index);
    }
  }

  bool add(CompletionFact fact) {
    canonicalizeLoops(fact.requiredLoops);
    SmallVector<std::size_t, 2> &bucket = buckets[hashFactKey(fact)];
    for (std::size_t index : bucket) {
      if (statistics) {
        ++statistics->coverageFactKeyTests;
      }
      CompletionFact &existing = facts[index];
      if (!sameFactKey(existing, fact)) {
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
    bucket.push_back(facts.size());
    byResource[resourceKey(fact.resource)].push_back(facts.size());
    facts.push_back(std::move(fact));
    return true;
  }

  ArrayRef<CompletionFact> values() const { return facts; }

  SmallVector<CompletionFact, 16>
  snapshot(CanonicalPhysicalResource resource) const {
    SmallVector<CompletionFact, 16> result;
    const auto found = byResource.find(resourceKey(resource));
    if (found == byResource.end()) {
      return result;
    }
    result.reserve(found->second.size());
    for (std::size_t index : found->second) {
      result.push_back(facts[index]);
    }
    return result;
  }

private:
  SmallVectorImpl<CompletionFact> &facts;
  std::unordered_map<std::size_t, SmallVector<std::size_t, 2>> buckets;
  std::unordered_map<std::uint64_t, SmallVector<std::size_t, 8>> byResource;
  CanonicalSyncStatistics *statistics = nullptr;
};

void applyBarrier(const CanonicalSyncProgram &program,
                  const CanonicalMechanism &mechanism,
                  CompletionFactIndex &facts,
                  ArrayRef<CanonicalRegionId> requiredLoops = {}) {
  for (CanonicalPhaseId phaseId :
       program.getMechanismSourcePrefix(mechanism.id)) {
    const CanonicalPhase &phase = program.getPhase(phaseId);
    facts.add({phase.id, mechanism.target, mechanism.targetPoint,
               combineGuards(phase.controlPath, mechanism.guard),
               SmallVector<CanonicalRegionId, 2>(requiredLoops.begin(),
                                                 requiredLoops.end())});
  }
}

bool applyFixedFence(const CanonicalSyncProgram &program,
                     const CanonicalMechanism &mechanism,
                     CompletionFactIndex &facts,
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
      changed |= facts.add(
          {phase, destination, mechanism.targetPoint,
           SmallVector<CanonicalControlAtom, 2>(guard.begin(), guard.end()),
           SmallVector<CanonicalRegionId, 2>(loops.begin(), loops.end())});
    }
  };
  for (CanonicalPhaseId phaseId :
       program.getMechanismSourcePrefix(mechanism.id)) {
    const CanonicalPhase &phase = program.getPhase(phaseId);
    publish(phase.id, combineGuards(phase.controlPath, mechanism.guard),
            requiredLoops);
  }
  for (CanonicalPhysicalResource resource : effect.drainedResources) {
    const SmallVector<CompletionFact, 16> snapshot = facts.snapshot(resource);
    if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
      statistics->coveragePropagationFactTests += snapshot.size();
    }
    for (const CompletionFact &fact : snapshot) {
      const bool precedes =
          programPointMustPrecede(fact.availableAt, mechanism.sourcePoint);
      const bool compatible =
          controlsCanCoexecute(fact.guard, mechanism.guard);
      if (!precedes || !compatible) {
        continue;
      }
      SmallVector<CanonicalRegionId, 2> loops = fact.requiredLoops;
      appendUnique(loops, requiredLoops);
      publish(fact.phase, combineGuards(fact.guard, mechanism.guard), loops);
    }
  }
  return changed;
}

bool applyEvent(const CanonicalSyncProgram &program,
                const CanonicalMechanism &mechanism,
                CompletionFactIndex &facts,
                ArrayRef<CanonicalRegionId> requiredLoops = {}) {
  bool changed = false;
  for (CanonicalPhaseId phaseId :
       program.getMechanismSourcePrefix(mechanism.id)) {
    const CanonicalPhase &phase = program.getPhase(phaseId);
    changed |= facts.add(
        {phase.id, mechanism.target, mechanism.targetPoint,
         combineGuards(phase.controlPath, mechanism.guard),
         SmallVector<CanonicalRegionId, 2>(requiredLoops.begin(),
                                           requiredLoops.end())});
  }
  const SmallVector<CompletionFact, 16> snapshot =
      facts.snapshot(mechanism.source);
  if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
    statistics->coveragePropagationFactTests += snapshot.size();
  }
  for (const CompletionFact &fact : snapshot) {
    if (fact.resource != mechanism.source ||
        !programPointMustPrecede(fact.availableAt, mechanism.sourcePoint) ||
        !controlsCanCoexecute(fact.guard, mechanism.guard)) {
      continue;
    }
    SmallVector<CanonicalRegionId, 2> loops = fact.requiredLoops;
    appendUnique(loops, requiredLoops);
    changed |= facts.add(
        {fact.phase, mechanism.target, mechanism.targetPoint,
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

std::size_t hashBoundaryTransfer(const CanonicalBoundaryTransfer &transfer) {
  llvm::hash_code hash = llvm::hash_combine(
      static_cast<unsigned>(transfer.source.core),
      static_cast<unsigned>(transfer.source.pipe),
      static_cast<unsigned>(transfer.target.core),
      static_cast<unsigned>(transfer.target.pipe),
      transfer.sourcePoint.operation,
      static_cast<unsigned>(transfer.sourcePoint.position),
      transfer.targetPoint.operation,
      static_cast<unsigned>(transfer.targetPoint.position),
      transfer.guard.size(), transfer.requiredLoops.size());
  for (const CanonicalControlAtom &atom : transfer.guard) {
    hash = llvm::hash_combine(hash, atom.choice, atom.arm);
  }
  for (CanonicalRegionId loop : transfer.requiredLoops) {
    hash = llvm::hash_combine(hash, loop);
  }
  return static_cast<std::size_t>(hash);
}

class BoundaryTransferIndex {
public:
  explicit BoundaryTransferIndex(
      SmallVectorImpl<CanonicalBoundaryTransfer> &transfers,
      CanonicalSyncStatistics *statistics)
      : transfers(transfers), statistics(statistics) {
    for (std::size_t index = 0; index < transfers.size(); ++index) {
      record(index);
    }
  }

  std::optional<std::size_t> insert(CanonicalBoundaryTransfer transfer) {
    canonicalizeLoops(transfer.requiredLoops);
    SmallVector<std::size_t, 2> &bucket =
        buckets[hashBoundaryTransfer(transfer)];
    for (std::size_t existing : bucket) {
      if (statistics) {
        ++statistics->coverageTransferKeyTests;
      }
      if (sameBoundaryTransfer(transfers[existing], transfer)) {
        return std::nullopt;
      }
    }
    const std::size_t index = transfers.size();
    bucket.push_back(index);
    transfers.push_back(std::move(transfer));
    bySource[resourceKey(transfers.back().source)].push_back(index);
    byTarget[resourceKey(transfers.back().target)].push_back(index);
    return index;
  }

  bool add(CanonicalBoundaryTransfer transfer) {
    return insert(std::move(transfer)).has_value();
  }

  ArrayRef<CanonicalBoundaryTransfer> values() const { return transfers; }

  void close() {
    SmallVector<std::size_t, 16> worklist;
    worklist.reserve(transfers.size());
    for (std::size_t index = 0; index < transfers.size(); ++index) {
      worklist.push_back(index);
    }
    for (std::size_t next = 0; next < worklist.size(); ++next) {
      const CanonicalBoundaryTransfer current = transfers[worklist[next]];
      composeFrom(current, bySource, current.target, true, worklist);
      composeFrom(current, byTarget, current.source, false, worklist);
    }
  }

private:
  void record(std::size_t index) {
    const CanonicalBoundaryTransfer &transfer = transfers[index];
    buckets[hashBoundaryTransfer(transfer)].push_back(index);
    bySource[resourceKey(transfer.source)].push_back(index);
    byTarget[resourceKey(transfer.target)].push_back(index);
  }

  void composeFrom(
      const CanonicalBoundaryTransfer &current,
      const std::unordered_map<std::uint64_t, SmallVector<std::size_t, 8>> &index,
      CanonicalPhysicalResource connector, bool currentFirst,
      SmallVectorImpl<std::size_t> &worklist) {
    const auto found = index.find(resourceKey(connector));
    if (found == index.end()) {
      return;
    }
    const SmallVector<std::size_t, 8> candidates = found->second;
    for (std::size_t candidateIndex : candidates) {
      if (statistics) {
        ++statistics->coverageTransferComposeTests;
      }
      const CanonicalBoundaryTransfer candidate = transfers[candidateIndex];
      const CanonicalBoundaryTransfer &first =
          currentFirst ? current : candidate;
      const CanonicalBoundaryTransfer &second =
          currentFirst ? candidate : current;
      const bool composablePoints =
          programPointMustPrecede(first.targetPoint, second.sourcePoint);
      const bool composableGuards =
          controlsCanCoexecute(first.guard, second.guard);
      if (!composablePoints || !composableGuards) {
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
      if (std::optional<std::size_t> added = insert(std::move(composed))) {
        worklist.push_back(*added);
      }
    }
  }

  SmallVectorImpl<CanonicalBoundaryTransfer> &transfers;
  std::unordered_map<std::size_t, SmallVector<std::size_t, 2>> buckets;
  std::unordered_map<std::uint64_t, SmallVector<std::size_t, 8>> bySource;
  std::unordered_map<std::uint64_t, SmallVector<std::size_t, 8>> byTarget;
  CanonicalSyncStatistics *statistics = nullptr;
};

void addFixedFenceTransfers(
    const CanonicalSyncProgram &program, const CanonicalMechanism &mechanism,
    BoundaryTransferIndex &transfers,
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
      transfers.add(std::move(transfer));
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
                              CompletionFactIndex &facts) {
  const SmallVector<CompletionFact, 32> snapshot(facts.values().begin(),
                                                  facts.values().end());
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
        changed |= facts.add({first.phase, first.resource, afterChoice,
                              commonGuard, first.requiredLoops});
      }
    }
  }
  return changed;
}

void joinChoiceTransfers(
    const CanonicalSyncProgram &program, const CanonicalRegion &choice,
    ArrayRef<const CanonicalRegionSummary *> children,
    BoundaryTransferIndex &transfers) {
  const CanonicalRegionSummary *arms[2] = {nullptr, nullptr};
  for (const CanonicalRegionSummary *child : children) {
    const unsigned arm = program.getRegion(child->region).arm;
    if (arm < 2U) {
      arms[arm] = child;
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
    transfers.add(std::move(joined));
  }
}

void joinChoiceCompletions(const CanonicalRegion &choice,
                           ArrayRef<const CanonicalRegionSummary *> children,
                           CompletionFactIndex &completions,
                           const CanonicalSyncProgram &program) {
  const CanonicalRegionSummary *arms[2] = {nullptr, nullptr};
  for (const CanonicalRegionSummary *child : children) {
    const unsigned arm = program.getRegion(child->region).arm;
    if (arm < 2U) {
      arms[arm] = child;
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
      completions.add({first.phase, first.resource, afterChoice, commonGuard,
                       first.requiredLoops});
    }
  }
}

void closeFlattenedFacts(const CanonicalSyncProgram &program,
                         ArrayRef<CanonicalMechanismId> selected,
                         CompletionFactIndex &facts) {
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
                              program.getMechanismExecutionLoops(mechanism.id));
      } else if (mechanism.kind == CanonicalMechanismKind::FixedFence) {
        changed |=
            applyFixedFence(program, mechanism, facts,
                            program.getMechanismExecutionLoops(mechanism.id));
      }
    }
  }
}

SmallVector<CompletionFact, 32>
evaluateFlattenedFacts(const CanonicalSyncProgram &program,
                       ArrayRef<CanonicalMechanismId> selected) {
  SmallVector<CompletionFact, 32> facts;
  CompletionFactIndex factIndex(facts, program.getStatistics());
  for (CanonicalMechanismId id : selected) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (mechanism.kind == CanonicalMechanismKind::PipeBarrier) {
      applyBarrier(program, mechanism, factIndex,
                   program.getMechanismExecutionLoops(mechanism.id));
    }
  }
  closeFlattenedFacts(program, selected, factIndex);
  return facts;
}

SmallVector<CompletionFact, 32> extendFlattenedFacts(
    const CanonicalSyncProgram &program,
    ArrayRef<CanonicalMechanismId> selected, CanonicalMechanismId singleton,
    ArrayRef<CompletionFact> baselineFacts) {
  SmallVector<CompletionFact, 32> facts(baselineFacts.begin(),
                                        baselineFacts.end());
  CompletionFactIndex factIndex(facts, program.getStatistics());
  const CanonicalMechanism &mechanism = program.getMechanism(singleton);
  if (mechanism.kind == CanonicalMechanismKind::PipeBarrier) {
    applyBarrier(program, mechanism, factIndex,
                 program.getMechanismExecutionLoops(mechanism.id));
  }
  closeFlattenedFacts(program, selected, factIndex);
  return facts;
}

bool applyBoundaryTransfer(const CanonicalSyncProgram &program,
                           const CanonicalBoundaryTransfer &transfer,
                           CompletionFactIndex &facts) {
  bool changed = false;
  if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
    statistics->coverageBoundaryPhaseTests += program.getPhases().size();
  }
  for (const CanonicalPhase &phase : program.getPhases()) {
    if (phase.resource != transfer.source ||
        !phaseMayPrecedePoint(phase, transfer.sourcePoint) ||
        !controlsCanCoexecute(phase.controlPath, transfer.guard)) {
      continue;
    }
    changed |= facts.add(
        {phase.id, transfer.target, transfer.targetPoint,
         combineGuards(phase.controlPath, transfer.guard),
         transfer.requiredLoops});
  }
  const SmallVector<CompletionFact, 16> snapshot =
      facts.snapshot(transfer.source);
  if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
    statistics->coveragePropagationFactTests += snapshot.size();
  }
  for (const CompletionFact &fact : snapshot) {
    if (fact.resource != transfer.source ||
        !programPointMustPrecede(fact.availableAt, transfer.sourcePoint) ||
        !controlsCanCoexecute(fact.guard, transfer.guard)) {
      continue;
    }
    SmallVector<CanonicalRegionId, 2> loops = fact.requiredLoops;
    appendUnique(loops, transfer.requiredLoops);
    changed |= facts.add(
        {fact.phase, transfer.target, transfer.targetPoint,
         combineGuards(fact.guard, transfer.guard), std::move(loops)});
  }
  return changed;
}

CanonicalRegionSummary summarizeRegionFromChildren(
    const CanonicalSyncProgram &program, CanonicalRegionId region,
    ArrayRef<CanonicalMechanismId> selected,
    ArrayRef<const CanonicalRegionSummary *> childSummaries) {
  CanonicalRegionSummary result;
  result.region = region;
  CompletionFactIndex completionIndex(result.completions,
                                      program.getStatistics());
  BoundaryTransferIndex transferIndex(result.transfers,
                                      program.getStatistics());
  for (const CanonicalRegionSummary *childSummary : childSummaries) {
    result.children.push_back(childSummary->region);
    for (const CompletionFact &fact : childSummary->completions) {
      CompletionFact imported = fact;
      if (program.getRegion(region).kind == CanonicalRegionKind::Loop &&
          !llvm::is_contained(imported.requiredLoops, region)) {
        imported.requiredLoops.push_back(region);
      }
      completionIndex.add(std::move(imported));
    }
    for (const CanonicalBoundaryTransfer &transfer : childSummary->transfers) {
      CanonicalBoundaryTransfer imported = transfer;
      if (program.getRegion(region).kind == CanonicalRegionKind::Loop &&
          !llvm::is_contained(imported.requiredLoops, region)) {
        imported.requiredLoops.push_back(region);
      }
      transferIndex.add(std::move(imported));
    }
  }
  if (program.getRegion(region).kind == CanonicalRegionKind::Choice) {
    joinChoiceCompletions(program.getRegion(region), childSummaries,
                          completionIndex, program);
    joinChoiceTransfers(program, program.getRegion(region), childSummaries,
                        transferIndex);
  }
  for (CanonicalMechanismId id : selected) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (mechanism.actionRegion != region) {
      continue;
    }
    if (mechanism.kind == CanonicalMechanismKind::PipeBarrier) {
      applyBarrier(program, mechanism, completionIndex,
                   program.getMechanismExecutionLoops(mechanism.id));
    } else if (mechanism.kind == CanonicalMechanismKind::FixedFence) {
      addFixedFenceTransfers(program, mechanism, transferIndex,
                             program.getMechanismExecutionLoops(mechanism.id));
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
      transfer.requiredLoops.assign(
          program.getMechanismExecutionLoops(mechanism.id).begin(),
          program.getMechanismExecutionLoops(mechanism.id).end());
      transferIndex.add(std::move(transfer));
    }
  }
  transferIndex.close();
  bool factsChanged = true;
  while (factsChanged) {
    factsChanged = false;
    for (const CanonicalBoundaryTransfer &transfer : result.transfers) {
      factsChanged |=
          applyBoundaryTransfer(program, transfer, completionIndex);
    }
  }
  if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
    ++statistics->coverageRegionSummaries;
    statistics->coverageSummaryFacts += result.completions.size();
    statistics->coverageSummaryTransfers += result.transfers.size();
  }
  return result;
}

std::size_t
summarizeRegion(const CanonicalSyncProgram &program, CanonicalRegionId region,
                ArrayRef<CanonicalMechanismId> selected,
                SmallVectorImpl<CanonicalRegionSummary> &summaries) {
  SmallVector<std::size_t, 4> childSummaryIndices;
  SmallVector<const CanonicalRegionSummary *, 4> childSummaryPointers;
  childSummaryIndices.reserve(program.getRegionChildren(region).size());
  childSummaryPointers.reserve(program.getRegionChildren(region).size());
  for (CanonicalRegionId child : program.getRegionChildren(region)) {
    childSummaryIndices.push_back(
        summarizeRegion(program, child, selected, summaries));
  }
  // Resolve pointers only after all recursive insertions have completed. The
  // summary vector may grow while a later sibling is being summarized.
  for (std::size_t childIndex : childSummaryIndices) {
    childSummaryPointers.push_back(&summaries[childIndex]);
  }
  CanonicalRegionSummary result =
      summarizeRegionFromChildren(program, region, selected,
                                  childSummaryPointers);
  summaries.push_back(std::move(result));
  return summaries.size() - 1U;
}

CanonicalRegionSummary summarizeRegionIncrementally(
    const CanonicalSyncProgram &program, CanonicalRegionId region,
    ArrayRef<CanonicalMechanismId> selected, ArrayRef<uint8_t> dirtyRegions,
    ArrayRef<std::size_t> baselineSummaryIndex,
    ArrayRef<CanonicalRegionSummary> baselineSummaries) {
  SmallVector<CanonicalRegionSummary, 4> changedChildSummaries;
  SmallVector<const CanonicalRegionSummary *, 4> childSummaryPointers;
  changedChildSummaries.reserve(program.getRegionChildren(region).size());
  childSummaryPointers.reserve(program.getRegionChildren(region).size());
  for (CanonicalRegionId child : program.getRegionChildren(region)) {
    if (dirtyRegions[child] == 0U) {
      childSummaryPointers.push_back(
          &baselineSummaries[baselineSummaryIndex[child]]);
      continue;
    }
    changedChildSummaries.push_back(summarizeRegionIncrementally(
        program, child, selected, dirtyRegions, baselineSummaryIndex,
        baselineSummaries));
    childSummaryPointers.push_back(&changedChildSummaries.back());
  }
  return summarizeRegionFromChildren(program, region, selected,
                                     childSummaryPointers);
}

bool executionLoopsImpliedByPhase(ArrayRef<CanonicalRegionId> executionLoops,
                                  const CanonicalPhase &phase) {
  return llvm::all_of(executionLoops, [&](CanonicalRegionId loop) {
    return llvm::is_contained(phase.loopPath, loop);
  });
}

bool recurrenceCoveredByCompletionCut(
    const CanonicalSyncProgram &program, const CanonicalDemand &demand,
    ArrayRef<CanonicalMechanismId> selectedCompletionCuts,
    ArrayRef<SmallVector<CanonicalRegionId, 2>> selectedExecutionLoops) {
  const auto carrying = llvm::find_if(
      demand.iterationDistance, [](const CanonicalLoopDistance &distance) {
        return distance.relation == CanonicalIterationRelation::AnyPositive;
      });
  if (carrying == demand.iterationDistance.end()) {
    return false;
  }
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &target = program.getPhase(demand.target);
  for (std::size_t index = 0; index < selectedCompletionCuts.size(); ++index) {
    const CanonicalMechanismId id = selectedCompletionCuts[index];
    const ArrayRef<CanonicalRegionId> loops = selectedExecutionLoops[index];
    const CanonicalMechanism &mechanism = program.getMechanism(id);
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
      const bool loopTransferApplies =
          executionLoopsImpliedByPhase(loops, source) &&
          executionLoopsImpliedByPhase(loops, target) &&
          (sameIterationImport || nextIterationImport);
      if (loopTransferApplies) {
        return true;
      }
      continue;
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
      const bool releaseApplies = executionLoopsImpliedByPhase(loops, source) &&
                                  executionLoopsImpliedByPhase(loops, target);
      if (releaseApplies) {
        return true;
      }
      continue;
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
      continue;
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
    if (completesAfterSource || completesBeforeTarget) {
      return true;
    }
  }
  return false;
}

bool demandCovered(
    const CanonicalSyncProgram &program, const CanonicalDemand &demand,
    ArrayRef<uint8_t> selectedMask,
    ArrayRef<const CompletionFact *> sourceFacts,
    const CanonicalSyncTarget &targetModel,
    ArrayRef<uint8_t> synchronousCompletion,
    ArrayRef<CanonicalMechanismId> selectedCompletionCuts,
    ArrayRef<SmallVector<CanonicalRegionId, 2>> selectedExecutionLoops) {
  if (demand.kind == CanonicalDemandKind::ExitCompletion) {
    const CanonicalMechanismId direct =
        program.getDirectMechanisms()[demand.id];
    return selectedMask[direct] != 0;
  }
  if (demand.requirement == CanonicalRequirement::Visibility) {
    const CanonicalMechanismId direct =
        program.getDirectMechanisms()[demand.id];
    return selectedMask[direct] != 0;
  }
  const CanonicalMechanismId direct = program.getDirectMechanisms()[demand.id];
  const bool recurringSelected =
      selectedMask[direct] != 0 && program.getMechanism(direct).kind ==
                                       CanonicalMechanismKind::RecurringEvent;
  if (recurringSelected) {
    return true;
  }
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &target = program.getPhase(demand.target);
  const bool intrinsicCompletion =
      (source.resource.core == target.resource.core &&
       synchronousCompletion[source.id] != 0) ||
      (source.resource == target.resource &&
       targetModel.hasIntrinsicCompletion(source.resource));
  if (intrinsicCompletion) {
    return true;
  }
  if (recurrenceCoveredByCompletionCut(program, demand, selectedCompletionCuts,
                                       selectedExecutionLoops)) {
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
  return llvm::any_of(sourceFacts, [&](const CompletionFact *fact) {
    const bool requiredExecution =
        llvm::all_of(fact->requiredLoops, [&](CanonicalRegionId loop) {
          return llvm::is_contained(source.loopPath, loop) ||
                 llvm::is_contained(target.loopPath, loop);
        });
    return fact->resource == target.resource &&
           pointMustPrecedePhase(fact->availableAt, target) &&
           guardImplies(executionGuard, fact->guard) && requiredExecution;
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
  SmallVector<uint8_t, 16> synchronousCompletion(program.getPhases().size(), 0);
  SmallVector<SmallVector<const CompletionFact *, 4>, 0> factsByPhase(
      program.getPhases().size());
  for (const CompletionFact &fact : facts) {
    factsByPhase[fact.phase].push_back(&fact);
  }
  SmallVector<uint8_t, 16> selectedMask(program.getMechanisms().size(), 0);
  SmallVector<CanonicalMechanismId, 8> selectedCompletionCuts;
  SmallVector<SmallVector<CanonicalRegionId, 2>, 8> selectedExecutionLoops;
  for (CanonicalMechanismId id : selected) {
    selectedMask[id] = 1;
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    const bool completionCut =
        mechanism.kind == CanonicalMechanismKind::PipeBarrier ||
        mechanism.kind == CanonicalMechanismKind::FixedFence ||
        mechanism.kind == CanonicalMechanismKind::RecurringEvent;
    if (completionCut) {
      selectedCompletionCuts.push_back(id);
      const ArrayRef<CanonicalRegionId> loops =
          program.getMechanismExecutionLoops(mechanism.id);
      selectedExecutionLoops.emplace_back(loops.begin(), loops.end());
    }
  }
  for (const CanonicalPhase &phase : program.getPhases()) {
    synchronousCompletion[phase.id] =
        getVPTOSchedulingSemantics(phase.operation).completionIsSynchronous;
  }
  for (const CanonicalDemand &demand : program.getDemands()) {
    const ArrayRef<const CompletionFact *> sourceFacts =
        demand.source < factsByPhase.size()
            ? ArrayRef<const CompletionFact *>(factsByPhase[demand.source])
            : ArrayRef<const CompletionFact *>();
    if (demandCovered(program, demand, selectedMask, sourceFacts, *target,
                      synchronousCompletion, selectedCompletionCuts,
                      selectedExecutionLoops)) {
      covered.push_back(demand.id);
    }
  }
  return covered;
}

CanonicalCoverageWorld finishWorldEvaluation(
    const CanonicalSyncProgram &program, CanonicalCoverageWorld world,
    const CanonicalRegionSummary &root,
    ArrayRef<CompletionFact> flattenedFacts) {
  const SmallVector<CanonicalDemandId, 16> summarized =
      coveredDemands(program, world.mechanisms, root.completions);
  world.covered.assign(summarized.begin(), summarized.end());

  const SmallVector<CanonicalDemandId, 16> flattened =
      coveredDemands(program, world.mechanisms, flattenedFacts);
  world.flattenedOracleMatched = summarized == flattened;

  FailureOr<CanonicalUnrolledCoverageResult> unrolled =
      evaluateCanonicalSyncUnrolledOracle(program, world.mechanisms);
  world.unrolledOracleAvailable = succeeded(unrolled);
  world.unrolledOracleExhaustive = succeeded(unrolled) && unrolled->exhaustive;
  // The bounded interpreter is a safety oracle, not a completeness
  // requirement for the compact hierarchical summary. Reject only when the
  // summary claims coverage which an exhaustive unrolling disproves. A
  // conservative summary may omit extra coverage seen by the interpreter.
  world.unrolledOracleMatched =
      world.unrolledOracleExhaustive &&
      llvm::all_of(summarized, [&](CanonicalDemandId demand) {
        return llvm::is_contained(unrolled->covered, demand);
      });
  if (!world.flattenedOracleMatched ||
      (world.unrolledOracleExhaustive && !world.unrolledOracleMatched)) {
    for (const CanonicalDemand &demand : program.getDemands()) {
      const bool summaryCovers = llvm::is_contained(summarized, demand.id);
      const bool flatCovers = llvm::is_contained(flattened, demand.id);
      const bool unrolledCovers =
          world.unrolledOracleExhaustive &&
          llvm::is_contained(unrolled->covered, demand.id);
      const bool unsafeUnrolledOverclaim =
          world.unrolledOracleExhaustive && summaryCovers && !unrolledCovers;
      if (summaryCovers != flatCovers || unsafeUnrolledOverclaim) {
        world.differentialDisagreements.push_back(demand.id);
      }
    }
  }
  return world;
}

CanonicalCoverageWorld evaluateWorld(
    const CanonicalSyncProgram &program, StringRef name,
    ArrayRef<CanonicalMechanismId> selected,
    SmallVectorImpl<CompletionFact> *flattenedOutput = nullptr) {
  CanonicalCoverageWorld world;
  world.name = name.str();
  world.mechanisms.assign(selected.begin(), selected.end());
  llvm::sort(world.mechanisms);
  world.mechanisms.erase(
      std::unique(world.mechanisms.begin(), world.mechanisms.end()),
      world.mechanisms.end());

  const std::size_t rootIndex =
      summarizeRegion(program, 0, world.mechanisms, world.summaries);
  const CanonicalRegionSummary &root = world.summaries[rootIndex];
  const SmallVector<CompletionFact, 32> flattenedFacts =
      evaluateFlattenedFacts(program, world.mechanisms);
  if (flattenedOutput) {
    flattenedOutput->assign(flattenedFacts.begin(), flattenedFacts.end());
  }
  return finishWorldEvaluation(program, std::move(world), root,
                               flattenedFacts);
}

FailureOr<SmallVector<uint8_t, 16>>
findDirtyRegionPath(const CanonicalSyncProgram &program,
                    CanonicalRegionId actionRegion) {
  if (actionRegion >= program.getRegions().size()) {
    program.getFunction().emitError(
        "canonical sync singleton has an invalid action region");
    return failure();
  }

  SmallVector<uint8_t, 16> dirtyRegions(program.getRegions().size(), 0U);
  CanonicalRegionId current = actionRegion;
  while (current != kInvalidCanonicalSyncId) {
    const bool invalidAncestry = current >= program.getRegions().size() ||
                                 dirtyRegions[current] != 0U;
    if (invalidAncestry) {
      program.getFunction().emitError(
          "canonical sync singleton has an invalid region ancestry");
      return failure();
    }
    dirtyRegions[current] = 1U;
    current = program.getRegion(current).parent;
  }
  const bool rootNotReached = dirtyRegions.empty() || dirtyRegions[0] == 0U;
  if (rootNotReached) {
    program.getFunction().emitError(
        "canonical sync singleton action region does not reach the root");
    return failure();
  }
  return dirtyRegions;
}

FailureOr<SmallVector<std::size_t, 16>> indexBaselineSummaries(
    const CanonicalSyncProgram &program,
    ArrayRef<CanonicalRegionSummary> summaries) {
  const std::size_t missingSummary = program.getRegions().size();
  SmallVector<std::size_t, 16> indices(program.getRegions().size(),
                                        missingSummary);
  for (std::size_t index = 0; index < summaries.size(); ++index) {
    const CanonicalRegionId region = summaries[index].region;
    const bool invalidSummary =
        region >= indices.size() || indices[region] != missingSummary;
    if (invalidSummary) {
      program.getFunction().emitError(
          "canonical sync baseline contains invalid region summaries");
      return failure();
    }
    indices[region] = index;
  }
  const bool missingRegion = summaries.size() != program.getRegions().size() ||
                             llvm::is_contained(indices, missingSummary);
  if (missingRegion) {
    program.getFunction().emitError(
        "canonical sync baseline omits a region summary");
    return failure();
  }
  return indices;
}

FailureOr<CanonicalCoverageWorld> evaluateSingletonWorld(
    const CanonicalSyncProgram &program, StringRef name,
    ArrayRef<CanonicalMechanismId> baseline, CanonicalMechanismId singleton,
    const CanonicalCoverageWorld &baselineWorld,
    ArrayRef<CompletionFact> baselineFlattenedFacts,
    ArrayRef<std::size_t> baselineSummaryIndex) {
  const CanonicalMechanism &mechanism = program.getMechanism(singleton);
  CanonicalCoverageWorld world;
  world.name = name.str();
  world.mechanisms.assign(baseline.begin(), baseline.end());
  world.mechanisms.push_back(singleton);
  llvm::sort(world.mechanisms);
  world.mechanisms.erase(
      std::unique(world.mechanisms.begin(), world.mechanisms.end()),
      world.mechanisms.end());

  FailureOr<SmallVector<uint8_t, 16>> dirtyRegions =
      findDirtyRegionPath(program, mechanism.actionRegion);
  if (failed(dirtyRegions)) {
    return failure();
  }

  CanonicalRegionSummary root = summarizeRegionIncrementally(
      program, 0, world.mechanisms, *dirtyRegions, baselineSummaryIndex,
      baselineWorld.summaries);

  const SmallVector<CompletionFact, 32> flattenedFacts = extendFlattenedFacts(
      program, world.mechanisms, singleton, baselineFlattenedFacts);
  world = finishWorldEvaluation(program, std::move(world), root,
                                flattenedFacts);
  world.summaries.push_back(std::move(root));
  return world;
}

FailureOr<CanonicalCoverageWorld>
validateCoverageWorld(const CanonicalSyncProgram &program,
                      CanonicalCoverageWorld world) {
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

void discardDetailedSummaries(CanonicalCoverageWorld &world) {
  SmallVector<CanonicalRegionSummary, 0> empty;
  world.summaries.swap(empty);
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
  return validateCoverageWorld(program, evaluateWorld(program, name, selected));
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
  SmallVector<CanonicalMechanismId, 8> baseline;
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    if (mechanism.kind == CanonicalMechanismKind::IntrinsicOrder ||
        mechanism.kind == CanonicalMechanismKind::FixedFence ||
        mechanism.kind == CanonicalMechanismKind::TailBarrier) {
      baseline.push_back(mechanism.id);
    }
  }
  SmallVector<CompletionFact, 32> baselineFlattenedFacts;
  FailureOr<CanonicalCoverageWorld> baselineWorld = validateCoverageWorld(
      program,
      evaluateWorld(program, "baseline", baseline, &baselineFlattenedFacts));
  if (failed(baselineWorld)) {
    return failure();
  }
  if (!baselineWorld->covered.empty()) {
    InFlightDiagnostic diagnostic = program.getFunction().emitError(
        "canonical sync fixed supply covers residual demand");
    for (CanonicalDemandId demandId : baselineWorld->covered) {
      const CanonicalDemand &demand = program.getDemand(demandId);
      diagnostic << " d" << demandId << "(p" << demand.source << "->p"
                 << demand.target << ')';
    }
    diagnostic << "; baseline integration is incomplete";
    return failure();
  }

  FailureOr<SmallVector<std::size_t, 16>> baselineSummaryIndex =
      indexBaselineSummaries(program, baselineWorld->summaries);
  if (failed(baselineSummaryIndex)) {
    return failure();
  }
  CanonicalCoverageWorld storedBaseline = *baselineWorld;
  discardDetailedSummaries(storedBaseline);
  program.appendCoverageWorld(std::move(storedBaseline));

  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    if (llvm::is_contained(baseline, mechanism.id)) {
      continue;
    }
    const std::string name = (Twine("singleton-m") + Twine(mechanism.id)).str();
    FailureOr<CanonicalCoverageWorld> world = evaluateSingletonWorld(
        program, name, baseline, mechanism.id, *baselineWorld,
        baselineFlattenedFacts, *baselineSummaryIndex);
    if (failed(world)) {
      return failure();
    }
    FailureOr<CanonicalCoverageWorld> checked =
        validateCoverageWorld(program, std::move(*world));
    if (failed(checked)) {
      return failure();
    }
    discardDetailedSummaries(*checked);
    program.appendCoverageWorld(std::move(*checked));
  }

  program.coverageCatalogComplete = true;
  return success();
}
