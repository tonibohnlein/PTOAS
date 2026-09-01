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
#include <iterator>

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

template <typename T>
SmallVector<T, 8> subtract(ArrayRef<T> values, ArrayRef<T> removed) {
  SmallVector<T, 8> result;
  llvm::copy_if(values, std::back_inserter(result), [removed](T value) {
    return !llvm::is_contained(removed, value);
  });
  canonicalize(result);
  return result;
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

SmallVector<CanonicalDemandId, 8>
candidateCoverage(const CanonicalSetCoverCandidate &candidate) {
  SmallVector<CanonicalDemandId, 8> coverage(candidate.directOrigins.begin(),
                                             candidate.directOrigins.end());
  coverage.append(candidate.additionalCoverage);
  canonicalize(coverage);
  return coverage;
}

BitVector coverageBits(const CanonicalSyncProgram &program,
                       const CanonicalSetCoverCandidate &candidate) {
  BitVector result(program.getDemands().size());
  for (CanonicalDemandId demand : candidateCoverage(candidate)) {
    result.set(demand);
  }
  return result;
}

BitVector universeBits(const CanonicalSyncProgram &program,
                       const CanonicalSetCoverInstance &instance) {
  BitVector result(program.getDemands().size());
  for (CanonicalDemandId demand : instance.universe) {
    result.set(demand);
  }
  return result;
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

BitVector selectedCoverage(const CanonicalSyncProgram &program,
                           const CanonicalSetCoverInstance &instance,
                           ArrayRef<CanonicalMechanismId> selected) {
  BitVector result(program.getDemands().size());
  for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
    if (llvm::is_contained(selected, candidate.mechanisms.front())) {
      result |= coverageBits(program, candidate);
    }
  }
  return result;
}

LogicalResult appendCandidate(CanonicalSyncProgram &program,
                              CanonicalSetCoverInstance &instance,
                              ArrayRef<CanonicalDemandId> baselineCovered,
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
    if (llvm::is_contained(instance.universe, origin)) {
      candidate.directOrigins.push_back(origin);
    }
  }
  canonicalize(candidate.directOrigins);

  const SmallVector<CanonicalDemandId, 8> covered =
      subtract<CanonicalDemandId>(world.covered, baselineCovered);
  if (llvm::any_of(candidate.directOrigins, [&](CanonicalDemandId origin) {
        return !llvm::is_contained(covered, origin);
      })) {
    return program.getFunction().emitError(
        "canonical sync singleton does not cover a direct origin");
  }
  const SmallVector<CanonicalDemandId, 8> additional =
      subtract<CanonicalDemandId>(covered, candidate.directOrigins);
  candidate.additionalCoverage.assign(additional.begin(), additional.end());
  const bool hasCoverage =
      !candidate.directOrigins.empty() || !candidate.additionalCoverage.empty();
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
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (!llvm::is_contained(baseline->covered, demand.id)) {
      instance.universe.push_back(demand.id);
    }
  }

  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    if (isBaselineMechanism(mechanism)) {
      continue;
    }
    SmallVector<CanonicalMechanismId, 8> selected(instance.baseline.begin(),
                                                  instance.baseline.end());
    selected.push_back(mechanism.id);
    canonicalize(selected);
    const CanonicalCoverageWorld *world = findCachedWorld(program, selected);
    if (!world) {
      return program.getFunction().emitError(
          "canonical sync set-cover construction is missing a checked "
          "singleton world");
    }
    if (failed(appendCandidate(program, instance, baseline->covered, mechanism,
                               *world))) {
      return failure();
    }
  }

  for (CanonicalDemandId demand : instance.universe) {
    if (llvm::none_of(instance.candidates,
                      [demand](const CanonicalSetCoverCandidate &candidate) {
                        return llvm::is_contained(candidateCoverage(candidate),
                                                  demand);
                      })) {
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
  const BitVector universe = universeBits(program, instance);
  BitVector covered(program.getDemands().size());
  CanonicalSetCoverSolution solution;
  SmallVector<CanonicalMechanismId, 8> selected(instance.baseline.begin(),
                                                instance.baseline.end());
  SmallVector<CanonicalMechanismId, 8> additionOrder;

  while (!coversUniverse(covered, universe)) {
    const CanonicalSetCoverCandidate *best = nullptr;
    unsigned bestGain = 0;
    for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
      if (llvm::is_contained(solution.greedyCandidates, candidate.id)) {
        continue;
      }
      unsigned count = 0;
      for (CanonicalDemandId demand : candidateCoverage(candidate)) {
        count += covered.test(demand) ? 0U : 1U;
      }
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
    selected.push_back(best->mechanisms.front());
    additionOrder.push_back(best->mechanisms.front());
    canonicalize(selected);
    covered |= coverageBits(program, *best);
  }

  for (CanonicalMechanismId mechanism : llvm::reverse(additionOrder)) {
    SmallVector<CanonicalMechanismId, 8> trial;
    llvm::copy_if(
        selected, std::back_inserter(trial),
        [mechanism](CanonicalMechanismId id) { return id != mechanism; });
    if (coversUniverse(selectedCoverage(program, instance, trial), universe)) {
      selected = std::move(trial);
      solution.reverseDeleted.push_back(mechanism);
    }
  }

  canonicalize(selected);
  if (!coversUniverse(selectedCoverage(program, instance, selected),
                      universe)) {
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
