// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Selectable columns are singleton direct mechanisms or bounded groups of
// those same mechanisms. Every incidence set is grounded before solving.
// Group columns are conjunctions: their coverage becomes active only when all
// member mechanisms are selected.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::pto;

namespace {

bool isBaselineMechanism(const CanonicalMechanism &mechanism) {
  return mechanism.kind == CanonicalMechanismKind::IntrinsicOrder ||
         mechanism.kind == CanonicalMechanismKind::FixedFence ||
         mechanism.kind == CanonicalMechanismKind::TailBarrier;
}

template <typename T> void canonicalize(SmallVectorImpl<T> &items) {
  llvm::sort(items);
  items.erase(std::unique(items.begin(), items.end()), items.end());
}

bool sameMechanisms(const CanonicalCoverageWorld &world,
                    ArrayRef<CanonicalMechanismId> mechanisms) {
  return ArrayRef<CanonicalMechanismId>(world.mechanisms) == mechanisms;
}

const CanonicalCoverageWorld *
findCachedWorld(const CanonicalSyncProgram &program,
                ArrayRef<CanonicalMechanismId> mechanisms) {
  auto found = llvm::find_if(program.getCoverageWorlds(),
                             [mechanisms](const CanonicalCoverageWorld &world) {
                               return sameMechanisms(world, mechanisms);
                             });
  return found == program.getCoverageWorlds().end() ? nullptr : &*found;
}

bool coversUniverse(const BitVector &covered, const BitVector &universe) {
  for (int demand = universe.find_first(); demand >= 0;
       demand = universe.find_next(demand)) {
    if (!covered.test(static_cast<unsigned>(demand))) {
      return false;
    }
  }
  return true;
}

bool candidateIsActive(const CanonicalSetCoverCandidate &candidate,
                       const BitVector &selectedMechanisms) {
  return llvm::all_of(candidate.mechanisms, [&](CanonicalMechanismId id) {
    return selectedMechanisms.test(id);
  });
}

BitVector activeCoverage(const CanonicalSetCoverInstance &instance,
                         const BitVector &selectedMechanisms,
                         std::size_t demandCount) {
  BitVector covered(demandCount);
  for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
    if (candidateIsActive(candidate, selectedMechanisms)) {
      covered |= candidate.incidence;
    }
  }
  return covered;
}

unsigned missingMechanismCount(const CanonicalSetCoverCandidate &candidate,
                               const BitVector &selectedMechanisms) {
  return llvm::count_if(candidate.mechanisms,
                        [&](CanonicalMechanismId mechanism) {
                          return !selectedMechanisms.test(mechanism);
                        });
}

bool betterMove(unsigned gain, unsigned cost,
                const CanonicalSetCoverCandidate &candidate, unsigned bestGain,
                unsigned bestCost, const CanonicalSetCoverCandidate *best) {
  if (!best) {
    return true;
  }
  const std::uint64_t lhs = static_cast<std::uint64_t>(gain) * bestCost;
  const std::uint64_t rhs = static_cast<std::uint64_t>(bestGain) * cost;
  if (lhs != rhs) {
    return lhs > rhs;
  }
  if (gain != bestGain) {
    return gain > bestGain;
  }
  if (cost != bestCost) {
    return cost < bestCost;
  }
  return candidate.id < best->id;
}

LogicalResult appendCandidate(
    CanonicalSyncProgram &program, CanonicalSetCoverInstance &instance,
    const BitVector &baselineCovered, ArrayRef<CanonicalMechanismId> mechanisms,
    ArrayRef<CanonicalDemandId> directOrigins,
    const CanonicalCoverageWorld &world,
    std::optional<CanonicalStructuralProposalId> structuralProposal =
        std::nullopt) {
  const bool candidateIdsExhausted =
      instance.candidates.size() >= kInvalidCanonicalSyncId;
  if (candidateIdsExhausted) {
    return program.getFunction().emitError(
        "canonical sync set-cover candidate ID space is exhausted");
  }

  CanonicalSetCoverCandidate candidate;
  candidate.id =
      static_cast<CanonicalSetCoverCandidateId>(instance.candidates.size());
  candidate.mechanisms.assign(mechanisms.begin(), mechanisms.end());
  canonicalize(candidate.mechanisms);
  candidate.incidence.resize(program.getDemands().size());
  candidate.weight = candidate.mechanisms.size();
  candidate.structuralProposal = structuralProposal;
  for (CanonicalDemandId origin : directOrigins) {
    if (instance.universeIncidence.test(origin)) {
      candidate.directOrigins.push_back(origin);
    }
  }
  canonicalize(candidate.directOrigins);

  BitVector covered(program.getDemands().size());
  for (CanonicalDemandId demand : world.covered) {
    covered.set(demand);
  }
  covered.reset(baselineCovered);
  covered &= instance.universeIncidence;
  if (llvm::any_of(candidate.directOrigins, [&](CanonicalDemandId origin) {
        return !covered.test(origin);
      })) {
    return program.getFunction().emitError(
        "canonical sync singleton does not cover a direct origin");
  }
  candidate.incidence = covered;
  BitVector additional = covered;
  for (CanonicalDemandId origin : candidate.directOrigins) {
    additional.reset(origin);
  }
  for (int demand = additional.find_first(); demand >= 0;
       demand = additional.find_next(demand)) {
    candidate.additionalCoverage.push_back(
        static_cast<CanonicalDemandId>(demand));
  }
  const bool hasCoverage = candidate.incidence.any();
  if (hasCoverage) {
    instance.candidates.push_back(std::move(candidate));
  }
  return success();
}

} // namespace

LogicalResult
mlir::pto::buildCanonicalSyncSetCoverInstance(CanonicalSyncProgram &program) {
  const bool invalidState = !program.isGraphFrozen() || program.isFrozen() ||
                            program.getSetCoverInstance().has_value() ||
                            !program.mechanismCatalogComplete ||
                            !program.structuralProposalCatalogComplete ||
                            !program.coverageCatalogComplete;
  if (invalidState) {
    return program.getFunction().emitError(
        "canonical sync set-cover construction requires one mutable plan");
  }

  CanonicalSetCoverInstance instance;
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    if (isBaselineMechanism(mechanism)) {
      instance.baseline.push_back(mechanism.id);
    }
  }
  canonicalize(instance.baseline);
  const CanonicalCoverageWorld *baseline =
      findCachedWorld(program, instance.baseline);
  if (!baseline ||
      program.getDirectMechanisms().size() != program.getDemands().size()) {
    return program.getFunction().emitError(
        "canonical sync set-cover construction requires checked singleton "
        "coverage and complete direct mechanisms");
  }
  BitVector baselineCovered(program.getDemands().size());
  for (CanonicalDemandId demand : baseline->covered) {
    baselineCovered.set(demand);
  }
  instance.universeIncidence.resize(program.getDemands().size());
  instance.universeIncidence.set();
  instance.universeIncidence.reset(baselineCovered);
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (instance.universeIncidence.test(demand.id)) {
      instance.universe.push_back(demand.id);
    }
  }

  SmallVector<const CanonicalCoverageWorld *, 8> singletonWorlds(
      program.getMechanisms().size(), nullptr);
  BitVector baselineMechanisms(program.getMechanisms().size());
  for (CanonicalMechanismId mechanism : instance.baseline) {
    baselineMechanisms.set(mechanism);
  }
  for (const CanonicalCoverageWorld &world : program.getCoverageWorlds()) {
    CanonicalMechanismId singleton = kInvalidCanonicalSyncId;
    bool missingBaseline = false;
    BitVector present(program.getMechanisms().size());
    for (CanonicalMechanismId mechanism : world.mechanisms) {
      present.set(mechanism);
      if (!baselineMechanisms.test(mechanism)) {
        if (singleton != kInvalidCanonicalSyncId) {
          singleton = kInvalidCanonicalSyncId;
          missingBaseline = true;
          break;
        }
        singleton = mechanism;
      }
    }
    for (CanonicalMechanismId mechanism : instance.baseline) {
      missingBaseline |= !present.test(mechanism);
    }
    if (!missingBaseline && singleton != kInvalidCanonicalSyncId) {
      singletonWorlds[singleton] = &world;
    }
  }

  BitVector providerCoverage(program.getDemands().size());
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    if (isBaselineMechanism(mechanism)) {
      continue;
    }
    const CanonicalCoverageWorld *world = singletonWorlds[mechanism.id];
    if (!world) {
      return program.getFunction().emitError(
          "canonical sync set-cover construction is missing a checked "
          "singleton world");
    }
    const std::size_t oldCandidateCount = instance.candidates.size();
    if (failed(appendCandidate(program, instance, baselineCovered,
                               ArrayRef<CanonicalMechanismId>(mechanism.id),
                               mechanism.origins, *world))) {
      return failure();
    }
    const bool candidateAdded = instance.candidates.size() != oldCandidateCount;
    if (candidateAdded) {
      providerCoverage |= instance.candidates.back().incidence;
    }
  }

  for (const CanonicalStructuralProposal &proposal :
       program.getStructuralProposals()) {
    if (!proposal.admitted) {
      continue;
    }
    const auto world = llvm::find_if(
        program.getCoverageWorlds(), [&](const CanonicalCoverageWorld &item) {
          return item.structuralProposal == proposal.id;
        });
    if (world == program.getCoverageWorlds().end()) {
      return program.getFunction().emitError(
          "canonical sync set-cover construction is missing a grounded "
          "structural world");
    }
    SmallVector<CanonicalDemandId, 8> directOrigins;
    for (CanonicalMechanismId mechanism : proposal.mechanisms) {
      llvm::append_range(directOrigins,
                         program.getMechanism(mechanism).origins);
    }
    canonicalize(directOrigins);
    if (failed(appendCandidate(program, instance, baselineCovered,
                               proposal.mechanisms, directOrigins, *world,
                               proposal.id))) {
      return failure();
    }
  }

  BitVector uncovered = instance.universeIncidence;
  uncovered.reset(providerCoverage);
  if (uncovered.any()) {
    const int demand = uncovered.find_first();
    return program.getFunction().emitError(
               "canonical sync set-cover instance omits demand d")
           << demand;
  }
  program.setSetCoverInstance(std::move(instance));
  return success();
}

LogicalResult
mlir::pto::solveCanonicalSyncSetCover(CanonicalSyncProgram &program) {
  const bool invalidState = !program.isGraphFrozen() || program.isFrozen() ||
                            !program.getSetCoverInstance() ||
                            program.getSetCoverSolution() ||
                            !program.mechanismCatalogComplete ||
                            !program.structuralProposalCatalogComplete ||
                            !program.coverageCatalogComplete;
  if (invalidState) {
    return program.getFunction().emitError(
        "canonical sync set-cover solving requires one complete mutable "
        "instance");
  }

  const CanonicalSetCoverInstance &instance = *program.getSetCoverInstance();
  const BitVector &universe = instance.universeIncidence;
  CanonicalSetCoverSolution solution;
  BitVector selectedMechanisms(program.getMechanisms().size());
  for (CanonicalMechanismId mechanism : instance.baseline) {
    selectedMechanisms.set(mechanism);
  }
  BitVector covered =
      activeCoverage(instance, selectedMechanisms, program.getDemands().size());
  SmallVector<CanonicalMechanismId, 8> additionOrder;

  while (!coversUniverse(covered, universe)) {
    const CanonicalSetCoverCandidate *best = nullptr;
    unsigned bestGain = 0;
    unsigned bestCost = 0;
    for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
      const unsigned cost =
          missingMechanismCount(candidate, selectedMechanisms);
      if (cost == 0U) {
        continue;
      }
      BitVector trialMechanisms = selectedMechanisms;
      for (CanonicalMechanismId mechanism : candidate.mechanisms) {
        trialMechanisms.set(mechanism);
      }
      BitVector gain = activeCoverage(instance, trialMechanisms,
                                      program.getDemands().size());
      gain.reset(covered);
      const unsigned count = gain.count();
      if (count != 0U &&
          betterMove(count, cost, candidate, bestGain, bestCost, best)) {
        best = &candidate;
        bestGain = count;
        bestCost = cost;
      }
    }
    if (!best) {
      return program.getFunction().emitError(
          "canonical sync grouped set-cover solver cannot cover its "
          "universe");
    }
    solution.greedyCandidates.push_back(best->id);
    for (CanonicalMechanismId mechanism : best->mechanisms) {
      if (!selectedMechanisms.test(mechanism)) {
        selectedMechanisms.set(mechanism);
        additionOrder.push_back(mechanism);
      }
    }
    covered = activeCoverage(instance, selectedMechanisms,
                             program.getDemands().size());
  }

  for (CanonicalMechanismId mechanism : llvm::reverse(additionOrder)) {
    selectedMechanisms.reset(mechanism);
    BitVector trialCoverage = activeCoverage(instance, selectedMechanisms,
                                             program.getDemands().size());
    if (!coversUniverse(trialCoverage, universe)) {
      selectedMechanisms.set(mechanism);
      continue;
    }
    solution.reverseDeleted.push_back(mechanism);
    covered = std::move(trialCoverage);
  }

  SmallVector<CanonicalMechanismId, 8> selected;
  for (int mechanism = selectedMechanisms.find_first(); mechanism >= 0;
       mechanism = selectedMechanisms.find_next(mechanism)) {
    selected.push_back(static_cast<CanonicalMechanismId>(mechanism));
  }
  const BitVector finalCoverage =
      activeCoverage(instance, selectedMechanisms, program.getDemands().size());
  if (!coversUniverse(finalCoverage, universe)) {
    return program.getFunction().emitError(
        "canonical sync grouped set-cover solution is incomplete");
  }
  for (CanonicalMechanismId mechanism : instance.baseline) {
    if (!selectedMechanisms.test(mechanism)) {
      return program.getFunction().emitError(
          "canonical sync grouped set-cover removed baseline supply");
    }
  }
  solution.mechanisms = std::move(selected);
  solution.weight = static_cast<std::uint64_t>(
      llvm::count_if(solution.mechanisms, [&](CanonicalMechanismId id) {
        return !llvm::is_contained(instance.baseline, id);
      }));
  solution.coverageVerified = true;
  program.setSetCoverSolution(std::move(solution));
  return success();
}
