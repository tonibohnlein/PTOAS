// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverCoverage.h - Completion coverage oracle --------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGE_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGE_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverDemandId = std::size_t;

struct SyncCoverReachableState {
  SyncCoverNodeId node = 0;
  unsigned copy = 0;
  bool hasCompletion = false;

  bool operator<(const SyncCoverReachableState &other) const;
  bool operator==(const SyncCoverReachableState &other) const;
};

enum class SyncCoverCoverageError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidDemand,
  InvalidSelection,
  InvalidBound,
  ExpansionLimitExceeded,
};

/// The legacy backend is retained only for differential tests while the shared
/// expansion is integrated. Shared expansion requires a frozen structure.
enum class SyncCoverCoverageBackend : std::uint8_t {
  SharedExpansion,
  LegacyPerContext,
};

struct SyncCoverCoverageResult {
  SyncCoverCoverageError error = SyncCoverCoverageError::None;
  bool covered = false;
  std::vector<SyncCoverMechanismId> witnessMechanisms;
  std::vector<SyncCoverReachableState> reachableStates;
  std::vector<SyncCoverMechanismId> cutMechanisms;

  explicit operator bool() const {
    return error == SyncCoverCoverageError::None;
  }
};

/// All individual mechanisms that establish one demand. The implementation
/// propagates mechanism bitsets through the prepared structural topology in
/// one pass; it does not issue one coverage query per mechanism.
struct SyncCoverSingletonWitnessResult {
  SyncCoverCoverageError error = SyncCoverCoverageError::None;
  std::vector<SyncCoverMechanismId> mechanisms;

  explicit operator bool() const {
    return error == SyncCoverCoverageError::None;
  }
};

/// Candidate member sets that establish one demand. All candidates are
/// propagated simultaneously as bitsets through one prepared topology.
struct SyncCoverSelectionWitnessResult {
  SyncCoverCoverageError error = SyncCoverCoverageError::None;
  std::vector<std::size_t> selections;

  explicit operator bool() const {
    return error == SyncCoverCoverageError::None;
  }
};

struct SyncCoverCoverageStatistics {
  std::size_t graphValidations = 0;
  std::size_t demandPreparations = 0;
  std::size_t coverageQueries = 0;
  std::size_t groundingQueries = 0;
  std::size_t preparedVirtualNodes = 0;
  std::size_t preparedVirtualEdges = 0;
  std::size_t maximumVirtualNodes = 0;
  std::size_t maximumVirtualEdges = 0;
};

/// Checks completion-qualified reachability without mutating the graph. A
/// selected mechanism enables every graph edge carrying its ID, so ownership
/// protocols and other multi-edge mechanisms remain atomic. Queries populate
/// mutable caches, so one oracle must not be used concurrently.
class SyncCoverCoverageOracle {
public:
  /// Snapshot and validate the completed graph. Mechanisms added to the source
  /// graph after construction intentionally do not change this oracle epoch.
  explicit SyncCoverCoverageOracle(
      const SyncCoverGraph &graph,
      SyncCoverCoverageBackend backend =
          SyncCoverCoverageBackend::SharedExpansion);
  ~SyncCoverCoverageOracle();
  SyncCoverCoverageOracle(const SyncCoverCoverageOracle &) = delete;
  SyncCoverCoverageOracle(SyncCoverCoverageOracle &&) = delete;
  SyncCoverCoverageOracle &operator=(const SyncCoverCoverageOracle &) = delete;
  SyncCoverCoverageOracle &operator=(SyncCoverCoverageOracle &&) = delete;

  SyncCoverCoverageResult
  checkDemand(SyncCoverDemandId demand,
              const std::vector<SyncCoverMechanismId> &selected) const;

  /// Search fast path. The selection must already be sorted and unique.
  SyncCoverCoverageResult checkDemandCanonicalSelection(
      SyncCoverDemandId demand,
      const std::vector<SyncCoverMechanismId> &selected) const;

  /// Checks a demand subset by sharing immutable virtual topologies between
  /// demands with identical execution contexts. Results correspond to the
  /// input order. The selection must already be sorted and unique.
  std::vector<SyncCoverCoverageResult> checkDemandsCanonicalSelection(
      const std::vector<SyncCoverDemandId> &demands,
      const std::vector<SyncCoverMechanismId> &selected) const;

  SyncCoverSingletonWitnessResult
  getSingletonMechanismWitnesses(SyncCoverDemandId demand) const;

  /// Batched singleton grounding. Demands with a shared execution context and
  /// source reuse one mechanism-bitset propagation.
  std::vector<SyncCoverSingletonWitnessResult>
  getSingletonMechanismWitnessesForDemands(
      const std::vector<SyncCoverDemandId> &demands,
      std::size_t mechanismCount) const;

  /// Batched counterpart to getSelectionWitnesses. Results correspond to the
  /// demand order and share one prepared topology per execution context.
  std::vector<SyncCoverSelectionWitnessResult> getSelectionWitnessesForDemands(
      const std::vector<SyncCoverDemandId> &demands,
      const std::vector<std::vector<SyncCoverMechanismId>> &selections,
      std::size_t mechanismCount) const;

  SyncCoverCoverageStatistics getStatistics() const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGE_H
