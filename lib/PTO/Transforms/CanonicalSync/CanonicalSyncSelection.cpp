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
         mechanism.kind == CanonicalMechanismKind::FixedFence;
}

std::uint64_t mechanismWeight(const CanonicalMechanism &mechanism) {
  switch (mechanism.kind) {
  case CanonicalMechanismKind::IntrinsicOrder:
  case CanonicalMechanismKind::FixedFence:
    return 0;
  case CanonicalMechanismKind::PipeBarrier:
  case CanonicalMechanismKind::TailBarrier:
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

LogicalResult appendCandidate(CanonicalSyncProgram &program,
                              CanonicalSetCoverInstance &instance,
                              ArrayRef<CanonicalDemandId> baselineCovered,
                              SmallVector<CanonicalMechanismId, 8> group) {
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
  const std::string name =
      (Twine("set-cover-group-c") + Twine(instance.candidates.size())).str();
  FailureOr<CanonicalCoverageWorld> evaluated =
      evaluateCanonicalSyncGroup(program, name, selected);
  if (failed(evaluated)) {
    return failure();
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
      subtract<CanonicalDemandId>(evaluated->covered, baselineCovered);
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
                            program.getSetCoverInstance().has_value();
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
  FailureOr<CanonicalCoverageWorld> baseline = evaluateCanonicalSyncGroup(
      program, "set-cover-baseline", instance.baseline);
  if (failed(baseline)) {
    return failure();
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
    if (failed(appendCandidate(program, instance, baseline->covered,
                               std::move(singleton)))) {
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
                               std::move(group)));
    if (groupFailed) {
      return failure();
    }
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
