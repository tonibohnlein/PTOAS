// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Converts fully grounded singleton and structural-group coverage worlds into
// immutable set-cover columns. No semantic reachability is evaluated here.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

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

LogicalResult appendCandidate(
    CanonicalSyncProgram &program, CanonicalSetCoverInstance &instance,
    ArrayRef<CanonicalMechanismId> mechanisms,
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
  llvm::append_range(candidate.mechanisms, mechanisms);
  canonicalize(candidate.mechanisms);
  candidate.weight = candidate.mechanisms.size();
  candidate.structuralProposal = structuralProposal;
  llvm::append_range(candidate.directOrigins, directOrigins);
  canonicalize(candidate.directOrigins);
  llvm::append_range(candidate.coveredDemands, world.covered);
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

void registerLatestCandidate(CanonicalSyncProgram &program,
                             CanonicalSetCoverInstance &instance) {
  const CanonicalSetCoverCandidate &candidate = instance.candidates.back();
  for (CanonicalDemandId demand : candidate.coveredDemands) {
    instance.providersByDemand[demand].push_back(candidate.id);
  }
  if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
    statistics->sparseIncidenceEntries += candidate.coveredDemands.size();
  }
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
    if (failed(appendCandidate(program, instance,
                               ArrayRef<CanonicalMechanismId>(mechanism.id),
                               mechanism.origins, *world))) {
      return failure();
    }
    const bool candidateAdded =
        instance.candidates.size() != oldCandidateCount;
    if (candidateAdded) {
      registerLatestCandidate(program, instance);
    }
  }

  for (const CanonicalStructuralProposal &proposal :
       program.getStructuralProposals()) {
    if (!proposal.admitted) {
      continue;
    }
    SmallVector<CanonicalMechanismId, 16> selected(instance.baseline.begin(),
                                                   instance.baseline.end());
    llvm::append_range(selected, proposal.mechanisms);
    canonicalize(selected);
    const CanonicalCoverageWorld *world = findCachedWorld(program, selected);
    if (!world || !world->setCoverCandidate ||
        world->structuralProposal != proposal.id) {
      return program.getFunction().emitError(
          "canonical sync set-cover construction is missing a grounded "
          "structural world");
    }
    SmallVector<CanonicalDemandId, 16> directOrigins;
    for (CanonicalMechanismId mechanism : proposal.mechanisms) {
      llvm::append_range(directOrigins,
                         program.getMechanism(mechanism).origins);
    }
    canonicalize(directOrigins);
    const std::size_t oldCandidateCount = instance.candidates.size();
    if (failed(appendCandidate(program, instance, proposal.mechanisms,
                               directOrigins, *world, proposal.id))) {
      return failure();
    }
    const bool candidateAdded =
        instance.candidates.size() != oldCandidateCount;
    if (candidateAdded) {
      registerLatestCandidate(program, instance);
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
