// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Each selectable column is one direct physical synchronization mechanism.
// Its coverage is computed once, before solving. Greedy selection and reverse
// deletion consume only those cached incidence sets.

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

LogicalResult appendCandidate(CanonicalSyncProgram &program,
                              CanonicalSetCoverInstance &instance,
                              const BitVector &baselineCovered,
                              const CanonicalMechanism &mechanism,
                              const CanonicalCoverageWorld &world) {
  const bool candidateIdsExhausted =
      instance.candidates.size() >= kInvalidCanonicalSyncId;
  if (candidateIdsExhausted) {
    return program.getFunction().emitError(
        "canonical sync set-cover candidate ID space is exhausted");
  }

  CanonicalSetCoverCandidate candidate;
  candidate.id =
      static_cast<CanonicalSetCoverCandidateId>(instance.candidates.size());
  candidate.mechanisms.push_back(mechanism.id);
  candidate.incidence.resize(program.getDemands().size());
  candidate.weight = 1;
  for (CanonicalDemandId origin : mechanism.origins) {
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
    if (failed(appendCandidate(program, instance, baselineCovered, mechanism,
                               *world))) {
      return failure();
    }
    const bool candidateAdded = instance.candidates.size() != oldCandidateCount;
    if (candidateAdded) {
      providerCoverage |= instance.candidates.back().incidence;
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
  const bool invalidState =
      !program.isGraphFrozen() || program.isFrozen() ||
      !program.getSetCoverInstance() || program.getSetCoverSolution() ||
      !program.mechanismCatalogComplete || !program.coverageCatalogComplete;
  if (invalidState) {
    return program.getFunction().emitError(
        "canonical sync set-cover solving requires one complete mutable "
        "instance");
  }

  const CanonicalSetCoverInstance &instance = *program.getSetCoverInstance();
  const BitVector &universe = instance.universeIncidence;
  BitVector covered(program.getDemands().size());
  CanonicalSetCoverSolution solution;
  SmallVector<CanonicalMechanismId, 8> selected(instance.baseline.begin(),
                                                instance.baseline.end());
  SmallVector<CanonicalMechanismId, 8> additionOrder;
  BitVector selectedCandidates(instance.candidates.size());
  SmallVector<unsigned, 8> coverageCounts(program.getDemands().size(), 0U);

  while (!coversUniverse(covered, universe)) {
    const CanonicalSetCoverCandidate *best = nullptr;
    unsigned bestGain = 0;
    for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
      if (selectedCandidates.test(candidate.id)) {
        continue;
      }
      BitVector gain = candidate.incidence;
      gain.reset(covered);
      const unsigned count = gain.count();
      if (count > bestGain) {
        best = &candidate;
        bestGain = count;
      }
    }
    if (!best) {
      return program.getFunction().emitError(
          "canonical sync singleton set-cover solver cannot cover its "
          "universe");
    }
    solution.greedyCandidates.push_back(best->id);
    selectedCandidates.set(best->id);
    selected.push_back(best->mechanisms.front());
    additionOrder.push_back(best->mechanisms.front());
    covered |= best->incidence;
    for (int demand = best->incidence.find_first(); demand >= 0;
         demand = best->incidence.find_next(demand)) {
      ++coverageCounts[static_cast<unsigned>(demand)];
    }
  }

  for (CanonicalMechanismId mechanism : llvm::reverse(additionOrder)) {
    const CanonicalSetCoverCandidate *candidate =
        llvm::find_if(instance.candidates,
                      [mechanism](const CanonicalSetCoverCandidate &item) {
                        return item.mechanisms.front() == mechanism;
                      });
    const bool removable =
        llvm::all_of(universe.set_bits(), [&](unsigned demand) {
          return !candidate->incidence.test(demand) ||
                 coverageCounts[demand] > 1U;
        });
    if (!removable) {
      continue;
    }
    selected.erase(std::remove(selected.begin(), selected.end(), mechanism),
                   selected.end());
    solution.reverseDeleted.push_back(mechanism);
    for (int demand = candidate->incidence.find_first(); demand >= 0;
         demand = candidate->incidence.find_next(demand)) {
      --coverageCounts[static_cast<unsigned>(demand)];
    }
  }

  canonicalize(selected);
  BitVector finalCoverage(program.getDemands().size());
  for (unsigned demand : universe.set_bits()) {
    if (coverageCounts[demand] != 0U) {
      finalCoverage.set(demand);
    }
  }
  if (!coversUniverse(finalCoverage, universe)) {
    return program.getFunction().emitError(
        "canonical sync singleton set-cover solution is incomplete");
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
