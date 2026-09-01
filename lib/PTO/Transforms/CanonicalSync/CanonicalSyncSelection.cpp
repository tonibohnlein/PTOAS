// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// This file builds the diagnostic weighted set-cover instance. Candidates are
// concrete mechanism groups: their coverage is evaluated as a group, while
// only coverage beyond the mechanisms' direct demand origins is recorded as
// additional coverage. No selection in this file changes materialization.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

bool isBaselineMechanism(const CanonicalMechanism &mechanism) {
  return mechanism.kind == CanonicalMechanismKind::IntrinsicOrder ||
         mechanism.kind == CanonicalMechanismKind::FixedFence ||
         mechanism.kind == CanonicalMechanismKind::TailBarrier;
}

std::uint64_t mechanismWeight(const CanonicalMechanism &mechanism) {
  switch (mechanism.kind) {
  case CanonicalMechanismKind::IntrinsicOrder:
  case CanonicalMechanismKind::FixedFence:
  case CanonicalMechanismKind::TailBarrier:
    return 0;
  case CanonicalMechanismKind::PipeBarrier:
    return 1;
  case CanonicalMechanismKind::Event:
    return 2;
  }
  llvm_unreachable("unknown canonical synchronization mechanism kind");
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

bool sameGroup(const CanonicalSetCoverCandidate &candidate,
               ArrayRef<CanonicalMechanismId> mechanisms) {
  return ArrayRef<CanonicalMechanismId>(candidate.mechanisms) == mechanisms;
}

bool sameMechanisms(const CanonicalCoverageWorld &world,
                    ArrayRef<CanonicalMechanismId> mechanisms) {
  return ArrayRef<CanonicalMechanismId>(world.mechanisms) == mechanisms;
}

SmallVector<CanonicalDemandId, 8>
candidateCoverage(const CanonicalSetCoverCandidate &candidate) {
  SmallVector<CanonicalDemandId, 8> coverage(candidate.directOrigins.begin(),
                                             candidate.directOrigins.end());
  coverage.append(candidate.additionalCoverage);
  canonicalize(coverage);
  return coverage;
}

bool coversEveryDemand(const CanonicalSyncProgram &program,
                       const CanonicalCoverageWorld &world) {
  return llvm::all_of(program.getDemands(), [&](const CanonicalDemand &demand) {
    return llvm::is_contained(world.covered, demand.id);
  });
}

struct GreedyChoice {
  const CanonicalSetCoverCandidate *candidate = nullptr;
  SmallVector<CanonicalMechanismId, 4> newMechanisms;
  std::uint64_t gain = 0;
  std::uint64_t marginalWeight = 0;
};

bool isBetterChoice(const GreedyChoice &candidate, const GreedyChoice &best) {
  if (!best.candidate) {
    return true;
  }
  if (candidate.marginalWeight == 0 || best.marginalWeight == 0) {
    if (candidate.marginalWeight != best.marginalWeight) {
      return candidate.marginalWeight == 0;
    }
  } else {
    llvm::APInt candidateRatio(128, candidate.gain);
    candidateRatio *= llvm::APInt(128, best.marginalWeight);
    llvm::APInt bestRatio(128, best.gain);
    bestRatio *= llvm::APInt(128, candidate.marginalWeight);
    if (candidateRatio != bestRatio) {
      return candidateRatio.ugt(bestRatio);
    }
  }
  if (candidate.marginalWeight != best.marginalWeight) {
    return candidate.marginalWeight < best.marginalWeight;
  }
  if (candidate.gain != best.gain) {
    return candidate.gain > best.gain;
  }
  const bool differentMechanismCount =
      candidate.newMechanisms.size() != best.newMechanisms.size();
  if (differentMechanismCount) {
    return candidate.newMechanisms.size() < best.newMechanisms.size();
  }
  return candidate.candidate->id < best.candidate->id;
}

LogicalResult computeMarginalWeight(CanonicalSyncProgram &program,
                                    const CanonicalSetCoverCandidate &candidate,
                                    ArrayRef<CanonicalMechanismId> selected,
                                    GreedyChoice &choice) {
  for (CanonicalMechanismId mechanism : candidate.mechanisms) {
    if (llvm::is_contained(selected, mechanism)) {
      continue;
    }
    const std::uint64_t weight =
        mechanismWeight(program.getMechanism(mechanism));
    if (weight >
        std::numeric_limits<std::uint64_t>::max() - choice.marginalWeight) {
      return program.getFunction().emitError(
          "canonical sync set-cover marginal weight overflowed uint64_t");
    }
    choice.marginalWeight += weight;
    choice.newMechanisms.push_back(mechanism);
  }
  canonicalize(choice.newMechanisms);
  return success();
}

FailureOr<std::uint64_t>
computeGroupWeight(const CanonicalSyncProgram &program,
                   ArrayRef<CanonicalMechanismId> mechanisms) {
  std::uint64_t result = 0;
  for (CanonicalMechanismId mechanism : mechanisms) {
    const std::uint64_t weight =
        mechanismWeight(program.getMechanism(mechanism));
    const bool weightOverflow =
        weight > std::numeric_limits<std::uint64_t>::max() - result;
    if (weightOverflow) {
      program.getFunction().emitError(
          "canonical sync set-cover solution weight overflowed uint64_t");
      return failure();
    }
    result += weight;
  }
  return result;
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
                              ArrayRef<CanonicalDemandId> baselineCovered,
                              SmallVector<CanonicalMechanismId, 8> group,
                              const CanonicalCoverageWorld &evaluated) {
  canonicalize(group);
  const bool containsBaseline =
      llvm::any_of(group, [&](CanonicalMechanismId id) {
        return isBaselineMechanism(program.getMechanism(id));
      });
  const bool duplicate = llvm::any_of(
      instance.candidates, [&](const CanonicalSetCoverCandidate &candidate) {
        return sameGroup(candidate, group);
      });
  const bool shouldSkip = group.empty() || containsBaseline || duplicate;
  if (shouldSkip) {
    return success();
  }

  SmallVector<CanonicalMechanismId, 8> selected(instance.baseline.begin(),
                                                instance.baseline.end());
  selected.append(group);
  canonicalize(selected);
  if (!sameMechanisms(evaluated, selected)) {
    return program.getFunction().emitError(
        "canonical sync cached coverage world has the wrong mechanism group");
  }

  CanonicalSetCoverCandidate candidate;
  const bool candidateIdsExhausted =
      instance.candidates.size() >= kInvalidCanonicalSyncId;
  if (candidateIdsExhausted) {
    return program.getFunction().emitError(
        "canonical sync set-cover candidate ID space is exhausted");
  }
  candidate.id =
      static_cast<CanonicalSetCoverCandidateId>(instance.candidates.size());
  candidate.mechanisms.assign(group.begin(), group.end());

  for (CanonicalMechanismId mechanismId : candidate.mechanisms) {
    const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
    const std::uint64_t weight = mechanismWeight(mechanism);
    const bool weightOverflow =
        weight > std::numeric_limits<std::uint64_t>::max() - candidate.weight;
    if (weightOverflow) {
      return program.getFunction().emitError(
          "canonical sync set-cover candidate weight overflowed uint64_t");
    }
    candidate.weight += weight;
    for (CanonicalDemandId origin : mechanism.origins) {
      if (llvm::is_contained(instance.universe, origin)) {
        candidate.directOrigins.push_back(origin);
      }
    }
  }
  canonicalize(candidate.directOrigins);

  SmallVector<CanonicalDemandId, 8> groupCoverage =
      subtract<CanonicalDemandId>(evaluated.covered, baselineCovered);
  if (llvm::any_of(candidate.directOrigins, [&](CanonicalDemandId origin) {
        return !llvm::is_contained(groupCoverage, origin);
      })) {
    return program.getFunction().emitError(
        "canonical sync concrete mechanism group does not cover a direct "
        "origin");
  }
  const SmallVector<CanonicalDemandId, 8> additional =
      subtract<CanonicalDemandId>(groupCoverage, candidate.directOrigins);
  candidate.additionalCoverage.assign(additional.begin(), additional.end());
  const bool noCoverage =
      candidate.directOrigins.empty() && candidate.additionalCoverage.empty();
  if (noCoverage) {
    return success();
  }
  instance.candidates.push_back(std::move(candidate));
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
  const bool directMechanismsComplete =
      program.getDirectMechanisms().size() == program.getDemands().size();
  if (!baseline || !directMechanismsComplete) {
    return program.getFunction().emitError(
        "canonical sync set-cover construction requires checked coverage and "
        "a complete direct-mechanism catalog");
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
    SmallVector<CanonicalMechanismId, 8> singleton{mechanism.id};
    SmallVector<CanonicalMechanismId, 8> selected(instance.baseline.begin(),
                                                  instance.baseline.end());
    selected.push_back(mechanism.id);
    canonicalize(selected);
    const CanonicalCoverageWorld *cached = findCachedWorld(program, selected);
    if (!cached) {
      return program.getFunction().emitError(
          "canonical sync set-cover construction is missing a checked "
          "singleton world");
    }
    if (failed(appendCandidate(program, instance, baseline->covered,
                               std::move(singleton), *cached))) {
      return failure();
    }
  }
  for (const CanonicalCoverageWorld &world : program.getCoverageWorlds()) {
    if (!world.setCoverCandidate) {
      continue;
    }
    SmallVector<CanonicalMechanismId, 8> group =
        subtract<CanonicalMechanismId>(world.mechanisms, instance.baseline);
    const bool groupFailed =
        group.size() > 1U &&
        failed(appendCandidate(program, instance, baseline->covered,
                               std::move(group), world));
    if (groupFailed) {
      return failure();
    }
  }

  SmallVector<CanonicalMechanismId, 8> all;
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    all.push_back(mechanism.id);
  }
  const CanonicalCoverageWorld *mechanical = findCachedWorld(program, all);
  const bool mechanicalComplete =
      mechanical && mechanical->covered.size() == program.getDemands().size();
  if (!mechanicalComplete) {
    return program.getFunction().emitError(
        "canonical sync set-cover construction requires a complete mechanical "
        "coverage world");
  }

  for (CanonicalDemandId demand : instance.universe) {
    const bool represented = llvm::any_of(
        instance.candidates,
        [demand](const CanonicalSetCoverCandidate &candidate) {
          return llvm::is_contained(candidate.directOrigins, demand) ||
                 llvm::is_contained(candidate.additionalCoverage, demand);
        });
    if (!represented) {
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
  SmallVector<CanonicalDemandId, 8> covered;
  SmallVector<CanonicalMechanismId, 8> selected(instance.baseline.begin(),
                                                instance.baseline.end());
  SmallVector<CanonicalMechanismId, 8> additionOrder;

  while (llvm::any_of(instance.universe, [&](CanonicalDemandId demand) {
    return !llvm::is_contained(covered, demand);
  })) {
    GreedyChoice best;
    for (const CanonicalSetCoverCandidate &candidate : instance.candidates) {
      if (llvm::is_contained(solution.greedyCandidates, candidate.id)) {
        continue;
      }
      const SmallVector<CanonicalDemandId, 8> incidence =
          candidateCoverage(candidate);
      GreedyChoice choice;
      choice.candidate = &candidate;
      choice.gain = static_cast<std::uint64_t>(
          llvm::count_if(incidence, [&](CanonicalDemandId demand) {
            return !llvm::is_contained(covered, demand);
          }));
      if (choice.gain == 0) {
        continue;
      }
      if (failed(computeMarginalWeight(program, candidate, selected, choice))) {
        return failure();
      }
      if (isBetterChoice(choice, best)) {
        best = std::move(choice);
      }
    }
    if (!best.candidate) {
      return program.getFunction().emitError(
          "canonical sync weighted set-cover solver cannot cover its "
          "universe");
    }

    solution.greedyCandidates.push_back(best.candidate->id);
    additionOrder.append(best.newMechanisms);
    selected.append(best.newMechanisms);
    canonicalize(selected);
    const SmallVector<CanonicalDemandId, 8> incidence =
        candidateCoverage(*best.candidate);
    covered.append(incidence);
    canonicalize(covered);
  }

  for (CanonicalMechanismId mechanism : llvm::reverse(additionOrder)) {
    SmallVector<CanonicalMechanismId, 8> trial;
    llvm::copy_if(
        selected, std::back_inserter(trial),
        [mechanism](CanonicalMechanismId id) { return id != mechanism; });
    std::string name = "set-cover-reverse-m";
    name += std::to_string(mechanism);
    FailureOr<CanonicalCoverageWorld> evaluated =
        canonical_sync_detail::evaluateCanonicalSyncGroup(program, name, trial);
    if (failed(evaluated)) {
      return failure();
    }
    if (coversEveryDemand(program, *evaluated)) {
      selected = std::move(trial);
      solution.reverseDeleted.push_back(mechanism);
    }
  }

  canonicalize(selected);
  FailureOr<CanonicalCoverageWorld> proposal =
      canonical_sync_detail::evaluateCanonicalSyncGroup(
          program, "set-cover-proposal", selected);
  if (failed(proposal)) {
    return failure();
  }
  if (!coversEveryDemand(program, *proposal)) {
    return program.getFunction().emitError(
        "canonical sync weighted set-cover proposal does not cover every "
        "demand");
  }
  FailureOr<std::uint64_t> weight = computeGroupWeight(program, selected);
  if (failed(weight)) {
    return failure();
  }
  solution.mechanisms = std::move(selected);
  solution.weight = *weight;
  solution.coverageVerified = true;
  program.setSetCoverSolution(std::move(solution));
  return success();
}
