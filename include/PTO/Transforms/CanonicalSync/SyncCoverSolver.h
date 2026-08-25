// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverSolver.h - Direct synchronization covering ------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVER_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVER_H

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverMechanism.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverComponentId = std::size_t;

struct SyncCoverSelectionSeed {
  std::uint64_t identity = 0;
  std::vector<SyncCoverMechanismId> mechanisms;
};

struct SyncCoverSelectionComponent {
  SyncCoverComponentId id = 0;
  std::vector<SyncCoverDemandId> demands;
  std::vector<SyncCoverMechanismId> mechanisms;
  bool exact = false;
};

struct SyncCoverSolverOptions {
  static constexpr std::size_t maximumExactMechanismThreshold = 24;
  static constexpr std::size_t maximumExchangeCandidateLimit = 64;

  std::size_t exactMechanismThreshold = 18;
  std::size_t beamWidth = 16;
  std::size_t beamDepth = 64;
  /// Per-component bounded-state evaluation limit for exact and beam search.
  /// Seed evaluation is outside this limit so a valid incumbent survives
  /// truncation.
  std::size_t evaluationLimit = 4096;
  std::size_t exchangeEvictionCandidateLimit = 16;
  std::size_t exchangeCandidateLimit = 24;
  std::size_t exchangeRoundLimit = 16;
  std::size_t exchangeEvaluationLimit = 4096;
};

enum class SyncCoverSelectionError : std::uint8_t {
  None,
  InvalidUniverse,
  InvalidDemand,
  InvalidSeed,
  InvalidOptions,
  ProvenInfeasible,
  SearchIncomplete,
  FinalVerificationFailed,
};

struct SyncCoverSearchTruncation {
  bool beamWidth = false;
  bool beamDepth = false;
  bool evaluationLimit = false;
  bool exchangeCandidateLimit = false;
  bool exchangeEvaluationLimit = false;
  bool exchangeRoundLimit = false;

  explicit operator bool() const {
    return beamWidth || beamDepth || evaluationLimit ||
           exchangeCandidateLimit || exchangeEvaluationLimit ||
           exchangeRoundLimit;
  }
};

struct SyncCoverExchangeStatistics {
  std::size_t rounds = 0;
  std::size_t evictionSets = 0;
  std::size_t evaluations = 0;
  std::size_t accepted = 0;
};

struct SyncCoverSelectionResult {
  SyncCoverSelectionError error = SyncCoverSelectionError::None;
  std::vector<SyncCoverMechanismId> mechanisms;
  SyncCoverStructuralCost cost;
  std::vector<SyncCoverSelectionComponent> components;
  std::size_t evaluations = 0;
  std::size_t redundancyEvaluations = 0;
  SyncCoverExchangeStatistics exchangeStatistics;
  SyncCoverSearchTruncation truncation;
  bool optimalityProven = false;
  SyncCoverResourceSelection resources;
  SyncCoverCoverageStatistics coverageStatistics;
  SyncCoverCoverageStatistics finalVerificationStatistics;

  explicit operator bool() const {
    return error == SyncCoverSelectionError::None;
  }
};

enum class SyncCoverMembershipError : std::uint8_t {
  None,
  InvalidUniverse,
  InvalidDemand,
  InvalidSelection,
  CoverageFailure,
};

struct SyncCoverMembershipDemand {
  SyncCoverDemandId demand = 0;
  std::vector<SyncCoverMechanismId> cutMechanisms;
};

/// Exact feasibility certificate for one caller-provided mechanism set. This
/// performs no search and reports resource feasibility separately from
/// completion coverage so candidate-generation, coloring, and coverage
/// failures remain distinguishable.
struct SyncCoverMembershipResult {
  SyncCoverMembershipError error = SyncCoverMembershipError::None;
  bool coverageComplete = false;
  SyncCoverResourceSelection resources;
  SyncCoverStructuralCost cost;
  std::vector<SyncCoverMembershipDemand> uncoveredDemands;
  SyncCoverCoverageStatistics coverageStatistics;

  explicit operator bool() const {
    return error == SyncCoverMembershipError::None;
  }
};

SyncCoverMembershipResult evaluateSyncCoverMembership(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &selected);

enum class SyncCoverBarrierFreeCensusStatus : std::uint8_t {
  Uncoverable,
  FeasibleWitness,
  InfeasibleWitness,
};

enum class SyncCoverBarrierFreeCensusError : std::uint8_t {
  None,
  InvalidUniverse,
  InvalidDemand,
  InvalidWitness,
  ResourceEvaluationFailed,
  CoverageFailure,
};

struct SyncCoverBarrierFreeCensusEntry {
  SyncCoverDemandId demand = 0;
  SyncCoverBarrierFreeCensusStatus status =
      SyncCoverBarrierFreeCensusStatus::Uncoverable;
  std::vector<SyncCoverMechanismId> witnessMechanisms;
  std::vector<SyncCoverReachableState> reachableStates;
  SyncCoverResourceSelection witnessResources;
};

/// Classifies demands against every non-barrier mechanism in the completed
/// universe. Uncoverable is a proof because coverage is queried without a
/// search cap. A feasible witness proves barrier-free coverability for one
/// demand in isolation. An infeasible witness describes only the oracle's
/// deterministic first witness; another resource-feasible witness may exist.
struct SyncCoverBarrierFreeCensusResult {
  SyncCoverBarrierFreeCensusError error =
      SyncCoverBarrierFreeCensusError::None;
  std::optional<SyncCoverDemandId> failedDemand;
  SyncCoverCoverageError coverageError = SyncCoverCoverageError::None;
  std::vector<SyncCoverBarrierFreeCensusEntry> entries;
  SyncCoverCoverageStatistics coverageStatistics;

  explicit operator bool() const {
    return error == SyncCoverBarrierFreeCensusError::None;
  }
};

SyncCoverBarrierFreeCensusResult evaluateSyncCoverBarrierFreeCensus(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands);

struct SyncCoverCompletionOptions {
  std::size_t beamWidth = 32;
  std::size_t depthLimit = 16;
  std::size_t candidateLimit = 64;
  std::size_t evaluationLimit = 4096;
};

enum class SyncCoverCompletionRejectionKind : std::uint8_t {
  InvalidSelection,
  ResourceInfeasible,
  CoverageFailure,
};

struct SyncCoverCompletionRejection {
  SyncCoverMechanismId mechanism = 0;
  SyncCoverCompletionRejectionKind kind =
      SyncCoverCompletionRejectionKind::InvalidSelection;
  SyncCoverResourceSelectionError resourceError =
      SyncCoverResourceSelectionError::None;
  std::optional<SyncCoverMechanismId> firstConflict;
  std::optional<SyncCoverMechanismId> secondConflict;
  std::optional<SyncCoverResourceDomainId> domain;
  std::size_t required = 0;
  std::size_t available = 0;
};

struct SyncCoverCompletionBlockedCut {
  SyncCoverDemandId demand = 0;
  std::vector<SyncCoverMechanismId> selected;
  std::vector<SyncCoverMechanismId> mechanisms;
  std::vector<SyncCoverReachableState> reachableStates;
};

struct SyncCoverCompletionResult {
  SyncCoverMembershipError error = SyncCoverMembershipError::None;
  bool complete = false;
  bool truncated = false;
  std::size_t evaluations = 0;
  std::vector<SyncCoverMechanismId> mechanisms;
  SyncCoverMembershipResult membership;
  std::vector<SyncCoverCompletionRejection> rejections;
  bool rejectionDiagnosticsTruncated = false;
  std::vector<SyncCoverCompletionBlockedCut> blockedCuts;
  bool blockedCutDiagnosticsTruncated = false;

  explicit operator bool() const {
    return error == SyncCoverMembershipError::None;
  }
};

/// Complete a fixed mechanism seed using only the explicitly allowed
/// mechanisms. Search is deterministic and bounded; each partial selection is
/// resource-checked, and a returned completion passes fresh whole-plan
/// membership verification.
SyncCoverCompletionResult completeSyncCoverMembership(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverMechanismId> &fixed,
    const std::vector<SyncCoverMechanismId> &allowed,
    const SyncCoverCompletionOptions &options = {});

/// Select an atomic synchronization cover for the active immutable demands.
/// Components conservatively include conflict and shared-resource coupling.
/// Exact search is cut-guided; larger components use a bounded deterministic
/// beam. Every returned result passes fresh graph, protocol, conflict,
/// coloring, and structural-cost validation plus a second exact coverage
/// traversal over the immutable prepared demand topology.
SyncCoverSelectionResult
solveSyncCoverSelection(const SyncCoverMechanismUniverse &universe,
                        const std::vector<SyncCoverDemandId> &activeDemands,
                        const std::vector<SyncCoverSelectionSeed> &seeds = {},
                        const SyncCoverSolverOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSOLVER_H
