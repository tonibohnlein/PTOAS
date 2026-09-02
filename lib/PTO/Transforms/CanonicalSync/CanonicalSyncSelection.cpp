// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Singleton columns and bounded structural group columns use only coverage
// grounded before solving. Structural groups are proposals, not proofs: a
// group is admitted only when the ordinary coverage oracle found additional
// coverage beyond the union of its singleton members.

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

template <typename T> void canonicalize(SmallVectorImpl<T> &items) {
  llvm::sort(items);
  items.erase(std::unique(items.begin(), items.end()), items.end());
}

bool coversUniverse(const CanonicalSetCoverInstance &instance,
                    ArrayRef<CanonicalMechanismId> selected) {
  BitVector selectedMechanisms;
  CanonicalMechanismId maximumMechanism = 0;
  for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
    for (CanonicalMechanismId mechanism : candidate.mechanisms) {
      maximumMechanism = std::max(maximumMechanism, mechanism);
    }
  }
  for (CanonicalMechanismId mechanism : selected) {
    maximumMechanism = std::max(maximumMechanism, mechanism);
  }
  selectedMechanisms.resize(static_cast<unsigned>(maximumMechanism) + 1U);
  for (CanonicalMechanismId mechanism : selected) {
    selectedMechanisms.set(mechanism);
  }
  BitVector covered(instance.providersByDemand.size());
  for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
    if (llvm::all_of(candidate.mechanisms, [&](CanonicalMechanismId mechanism) {
          return selectedMechanisms.test(mechanism);
        })) {
      for (CanonicalDemandId demand : candidate.coveredDemands) {
        covered.set(demand);
      }
    }
  }
  return llvm::all_of(instance.universe, [&](CanonicalDemandId demand) {
    return covered.test(demand);
  });
}

struct HeapEntry {
  unsigned gain = 0;
  unsigned cost = 0;
  CanonicalSetCoverCandidateId candidate = kInvalidCanonicalSyncId;
};

struct HeapEntryLess {
  bool operator()(const HeapEntry &left, const HeapEntry &right) const {
    if (left.gain == 0U || right.gain == 0U) {
      if (left.gain != right.gain) {
        return left.gain == 0U;
      }
    }
    if (left.cost == 0U || right.cost == 0U) {
      if (left.cost != right.cost) {
        return left.cost != 0U;
      }
    } else {
      const std::uint64_t leftScore =
          static_cast<std::uint64_t>(left.gain) * right.cost;
      const std::uint64_t rightScore =
          static_cast<std::uint64_t>(right.gain) * left.cost;
      if (leftScore != rightScore) {
        return leftScore < rightScore;
      }
    }
    if (left.gain != right.gain) {
      return left.gain < right.gain;
    }
    if (left.cost != right.cost) {
      return left.cost > right.cost;
    }
    return left.candidate > right.candidate;
  }
};

FailureOr<CanonicalSetCoverSolution>
solveGreedyTrial(CanonicalSyncProgram &program,
                 const CanonicalSetCoverInstance &instance,
                 std::optional<CanonicalSetCoverCandidateId> seed) {
  CanonicalSetCoverSolution solution;
  SmallVector<CanonicalMechanismId, 8> selected(instance.baseline.begin(),
                                                instance.baseline.end());
  SmallVector<CanonicalMechanismId, 8> additionOrder;
  SmallVector<uint8_t, 8> selectedCandidates(instance.candidates.size(), 0U);
  SmallVector<uint8_t, 8> covered(program.getDemands().size(), 0U);
  BitVector selectedMechanisms(program.getMechanisms().size());
  for (CanonicalMechanismId mechanism : instance.baseline) {
    selectedMechanisms.set(mechanism);
  }
  unsigned coveredCount = 0;

  const auto selectCandidate =
      [&](const CanonicalSetCoverCandidate &candidate) {
        solution.greedyCandidates.push_back(candidate.id);
        selectedCandidates[candidate.id] = 1U;
        for (CanonicalMechanismId mechanism : candidate.mechanisms) {
          if (selectedMechanisms.test(mechanism)) {
            continue;
          }
          selectedMechanisms.set(mechanism);
          selected.push_back(mechanism);
          additionOrder.push_back(mechanism);
        }
        for (CanonicalDemandId demand : candidate.coveredDemands) {
          if (covered[demand] == 0U) {
            covered[demand] = 1U;
            ++coveredCount;
          }
        }
      };
  if (seed) {
    selectCandidate(instance.candidates[*seed]);
  }

  const auto candidateCost = [&](const CanonicalSetCoverCandidate &candidate) {
    return static_cast<unsigned>(llvm::count_if(
        candidate.mechanisms, [&](CanonicalMechanismId mechanism) {
          return !selectedMechanisms.test(mechanism);
        }));
  };
  const auto candidateGain = [&](const CanonicalSetCoverCandidate &candidate) {
    return static_cast<unsigned>(
        llvm::count_if(candidate.coveredDemands, [&](CanonicalDemandId demand) {
          return covered[demand] == 0U;
        }));
  };
  SmallVector<SmallVector<CanonicalSetCoverCandidateId, 2>, 0>
      candidatesByMechanism(program.getMechanisms().size());
  for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
    for (CanonicalMechanismId mechanism : candidate.mechanisms) {
      candidatesByMechanism[mechanism].push_back(candidate.id);
    }
  }
  std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryLess> heap;
  for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
    heap.push(
        {candidateGain(candidate), candidateCost(candidate), candidate.id});
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
      const unsigned actualGain = candidateGain(candidate);
      const unsigned actualCost = candidateCost(candidate);
      if (actualGain != entry.gain || actualCost != entry.cost) {
        heap.push({actualGain, actualCost, candidate.id});
        continue;
      }
      best = actualGain == 0U ? nullptr : &candidate;
      break;
    }
    if (!best) {
      return program.getFunction().emitError(
          "canonical sync set-cover solver cannot cover its universe");
    }
    SmallVector<CanonicalMechanismId, 4> newlySelected;
    for (CanonicalMechanismId mechanism : best->mechanisms) {
      if (!selectedMechanisms.test(mechanism)) {
        newlySelected.push_back(mechanism);
      }
    }
    selectCandidate(*best);
    for (CanonicalMechanismId mechanism : newlySelected) {
      for (CanonicalSetCoverCandidateId candidateId :
           candidatesByMechanism[mechanism]) {
        if (selectedCandidates[candidateId] != 0U) {
          continue;
        }
        const CanonicalSetCoverCandidate &candidate =
            instance.candidates[candidateId];
        heap.push(
            {candidateGain(candidate), candidateCost(candidate), candidateId});
      }
    }
  }

  for (CanonicalMechanismId mechanism : llvm::reverse(additionOrder)) {
    SmallVector<CanonicalMechanismId, 8> trial(selected.begin(),
                                               selected.end());
    trial.erase(std::remove(trial.begin(), trial.end(), mechanism),
                trial.end());
    if (!coversUniverse(instance, trial)) {
      continue;
    }
    selected = std::move(trial);
    solution.reverseDeleted.push_back(mechanism);
  }
  canonicalize(selected);
  if (!coversUniverse(instance, selected)) {
    return program.getFunction().emitError(
        "canonical sync set-cover solution is incomplete");
  }
  solution.mechanisms = std::move(selected);
  solution.weight = static_cast<std::uint64_t>(
      llvm::count_if(solution.mechanisms, [&](CanonicalMechanismId id) {
        return !llvm::is_contained(instance.baseline, id);
      }));
  solution.coverageVerified = true;
  return solution;
}

bool isBetterSolution(const CanonicalSetCoverSolution &candidate,
                      const CanonicalSetCoverSolution &current) {
  if (candidate.weight != current.weight) {
    return candidate.weight < current.weight;
  }
  return std::lexicographical_compare(
      candidate.mechanisms.begin(), candidate.mechanisms.end(),
      current.mechanisms.begin(), current.mechanisms.end());
}

} // namespace

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
  FailureOr<CanonicalSetCoverSolution> best =
      solveGreedyTrial(program, instance, std::nullopt);
  if (failed(best)) {
    return failure();
  }
  constexpr unsigned kMaximumStructuralSeedTrials = 32;
  unsigned trialCount = 0;
  for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
    if (!candidate.structuralProposal ||
        trialCount == kMaximumStructuralSeedTrials) {
      continue;
    }
    ++trialCount;
    FailureOr<CanonicalSetCoverSolution> trial =
        solveGreedyTrial(program, instance, candidate.id);
    if (failed(trial)) {
      return failure();
    }
    if (isBetterSolution(*trial, *best)) {
      best = std::move(*trial);
    }
  }
  program.setSetCoverSolution(std::move(*best));
  return success();
}
