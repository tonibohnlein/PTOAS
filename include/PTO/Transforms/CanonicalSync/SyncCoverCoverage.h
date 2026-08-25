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
  ExpansionLimitExceeded,
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

struct SyncCoverDemandTopologyResult {
  SyncCoverCoverageError error = SyncCoverCoverageError::None;
  std::vector<SyncCoverMechanismId> potentialMechanisms;

  explicit operator bool() const {
    return error == SyncCoverCoverageError::None;
  }
};

struct SyncCoverCoverageStatistics {
  std::size_t graphValidations = 0;
  std::size_t demandPreparations = 0;
  std::size_t coverageQueries = 0;
  std::size_t preparedVirtualNodes = 0;
  std::size_t preparedVirtualEdges = 0;
  std::size_t maximumVirtualNodes = 0;
  std::size_t maximumVirtualEdges = 0;
};

/// Checks completion-qualified reachability without mutating the graph. A
/// selected mechanism enables every graph edge carrying its ID, so ownership
/// protocols and other multi-edge mechanisms remain atomic.
class SyncCoverCoverageOracle {
public:
  /// Snapshot and validate the completed graph. Mechanisms added to the source
  /// graph after construction intentionally do not change this oracle epoch.
  explicit SyncCoverCoverageOracle(const SyncCoverGraph &graph);

  SyncCoverCoverageResult
  checkDemand(SyncCoverDemandId demand,
              const std::vector<SyncCoverMechanismId> &selected) const;

  /// Search fast path. The selection must already be sorted and unique.
  SyncCoverCoverageResult checkDemandCanonicalSelection(
      SyncCoverDemandId demand,
      const std::vector<SyncCoverMechanismId> &selected) const;

  std::vector<SyncCoverCoverageResult>
  checkAll(const std::vector<SyncCoverMechanismId> &selected) const;

  SyncCoverDemandTopologyResult
  getDemandTopology(SyncCoverDemandId demand) const;

  SyncCoverCoverageStatistics getStatistics() const;

private:
  struct Implementation;
  std::shared_ptr<Implementation> implementation_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGE_H
