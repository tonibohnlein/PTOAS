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
#include <queue>
#include <vector>

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

LogicalResult appendCandidate(CanonicalSyncProgram &program,
                              CanonicalSetCoverInstance &instance,
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
  candidate.weight = 1;
  for (CanonicalDemandId origin : mechanism.origins) {
    candidate.directOrigins.push_back(origin);
  }
  canonicalize(candidate.directOrigins);

  for (CanonicalDemandId demand : world.covered) {
    candidate.coveredDemands.push_back(demand);
  }
  canonicalize(candidate.coveredDemands);
  if (llvm::any_of(candidate.directOrigins, [&](CanonicalDemandId origin) {
        return !llvm::is_contained(candidate.coveredDemands, origin);
      })) {
    return program.getFunction().emitError(
        "canonical sync singleton does not cover a direct origin");
  }
  for (CanonicalDemandId demand : candidate.coveredDemands) {
    if (!llvm::is_contained(candidate.directOrigins, demand)) {
      candidate.additionalCoverage.push_back(demand);
    }
  }
  if (!candidate.coveredDemands.empty()) {
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
  for (const CanonicalDemand &demand : program.getDemands()) {
    instance.universe.push_back(demand.id);
  }
  instance.providersByDemand.resize(program.getDemands().size());

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
    if (failed(appendCandidate(program, instance, mechanism, *world))) {
      return failure();
    }
    const bool candidateAdded = instance.candidates.size() != oldCandidateCount;
    if (candidateAdded) {
      const CanonicalSetCoverCandidate &candidate = instance.candidates.back();
      for (CanonicalDemandId demand : candidate.coveredDemands) {
        instance.providersByDemand[demand].push_back(candidate.id);
      }
      if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
        statistics->sparseIncidenceEntries += candidate.coveredDemands.size();
      }
    }
  }

  for (CanonicalDemandId demand : instance.universe) {
    if (instance.providersByDemand[demand].empty()) {
      return program.getFunction().emitError(
                 "canonical sync set-cover instance omits demand d")
             << demand;
    }
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
  CanonicalSetCoverSolution solution;
  SmallVector<CanonicalMechanismId, 8> selected(instance.baseline.begin(),
                                                instance.baseline.end());
  SmallVector<CanonicalMechanismId, 8> additionOrder;
  SmallVector<uint8_t, 8> selectedCandidates(instance.candidates.size(), 0U);
  SmallVector<uint8_t, 8> covered(program.getDemands().size(), 0U);
  SmallVector<unsigned, 8> coverageCounts(program.getDemands().size(), 0U);
  unsigned coveredCount = 0;

  struct HeapEntry {
    unsigned gain = 0;
    CanonicalSetCoverCandidateId candidate = kInvalidCanonicalSyncId;
  };
  struct HeapEntryLess {
    bool operator()(const HeapEntry &left, const HeapEntry &right) const {
      if (left.gain != right.gain) {
        return left.gain < right.gain;
      }
      return left.candidate > right.candidate;
    }
  };
  std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryLess> heap;
  for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
    heap.push(
        {static_cast<unsigned>(candidate.coveredDemands.size()), candidate.id});
  }

  while (coveredCount != instance.universe.size()) {
    const CanonicalSetCoverCandidate *best = nullptr;
    while (!heap.empty()) {
      const HeapEntry entry = heap.top();
      heap.pop();
      if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
        ++statistics->greedyHeapPops;
      }
      if (selectedCandidates[entry.candidate] != 0U) {
        continue;
      }
      const CanonicalSetCoverCandidate &candidate =
          instance.candidates[entry.candidate];
      if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
        statistics->greedyIncidenceVisits += candidate.coveredDemands.size();
      }
      const unsigned actualGain = llvm::count_if(
          candidate.coveredDemands,
          [&](CanonicalDemandId demand) { return covered[demand] == 0U; });
      if (actualGain != entry.gain) {
        heap.push({actualGain, candidate.id});
        continue;
      }
      best = actualGain == 0U ? nullptr : &candidate;
      break;
    }
    if (!best) {
      return program.getFunction().emitError(
          "canonical sync singleton set-cover solver cannot cover its "
          "universe");
    }
    solution.greedyCandidates.push_back(best->id);
    selectedCandidates[best->id] = 1U;
    selected.push_back(best->mechanisms.front());
    additionOrder.push_back(best->mechanisms.front());
    for (CanonicalDemandId demand : best->coveredDemands) {
      if (covered[demand] == 0U) {
        covered[demand] = 1U;
        ++coveredCount;
      }
      ++coverageCounts[demand];
    }
  }

  SmallVector<const CanonicalSetCoverCandidate *, 8> candidateByMechanism(
      program.getMechanisms().size(), nullptr);
  for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
    candidateByMechanism[candidate.mechanisms.front()] = &candidate;
  }
  for (CanonicalMechanismId mechanism : llvm::reverse(additionOrder)) {
    const CanonicalSetCoverCandidate *candidate =
        candidateByMechanism[mechanism];
    const bool removable =
        llvm::all_of(candidate->coveredDemands, [&](CanonicalDemandId demand) {
          return coverageCounts[demand] > 1U;
        });
    if (!removable) {
      continue;
    }
    selected.erase(std::remove(selected.begin(), selected.end(), mechanism),
                   selected.end());
    solution.reverseDeleted.push_back(mechanism);
    for (CanonicalDemandId demand : candidate->coveredDemands) {
      --coverageCounts[demand];
    }
  }

  canonicalize(selected);
  if (llvm::any_of(instance.universe, [&](CanonicalDemandId demand) {
        return coverageCounts[demand] == 0U;
      })) {
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
