// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverStorageProtocolRectangles.h - Compact cuts -*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLRECTANGLES_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLRECTANGLES_H

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolFrontiers.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverStorageProtocolRectangleId = std::size_t;

/// Exact endpoint-equivalent lifecycle frontiers factored into one compact
/// cut rectangle. The half-open incidence range retains every proof source;
/// it does not expand the producer-by-consumer relation. A rectangle remains
/// an unbalanced semantic proposal until a later lifecycle certificate proves
/// priming, circulation, and draining.
struct SyncCoverStorageProtocolRectangle {
  SyncCoverStorageProtocolRectangleId id = 0;
  SyncCoverStorageProtocolFrontierPlanId plan = 0;
  SyncCoverStorageProtocolAutomatonId automaton = 0;
  SyncCoverStorageProtocolFrontierKind kind =
      SyncCoverStorageProtocolFrontierKind::Ready;
  SyncCoverAnchor completionAnchor;
  SyncCoverAnchor acquisitionAnchor;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  std::size_t frontierBegin = 0;
  std::size_t frontierCount = 0;
};

struct SyncCoverStorageProtocolRectangleLimits {
  std::size_t maximumWorkUnits = 1U << 24;
  std::size_t maximumFrontierInspections = 1U << 21;
  std::size_t maximumRectangles = 1U << 20;
  std::size_t maximumFrontierIncidences = 1U << 21;
};

struct SyncCoverStorageProtocolRectangleStatistics {
  std::size_t workUnits = 0;
  std::size_t plans = 0;
  std::size_t frontierInspections = 0;
  std::size_t rectangles = 0;
  std::size_t readyRectangles = 0;
  std::size_t reuseRectangles = 0;
  std::size_t mergedRectangles = 0;
  std::size_t frontierIncidences = 0;
  std::size_t maximumRectangleFrontiers = 0;
  bool truncated = false;
};

enum class SyncCoverStorageProtocolRectangleError : std::uint8_t {
  None,
  InvalidGraph,
  IncompleteAutomatonIndex,
  IncompleteFrontierIndex,
  InvalidLimit,
  LimitExceeded,
  ArithmeticOverflow,
};

class SyncCoverStorageProtocolRectangleIndex {
public:
  const std::vector<SyncCoverStorageProtocolRectangle> &getRectangles() const {
    return rectangles_;
  }
  const std::vector<SyncCoverStorageProtocolFrontierId> &
  getFrontierIncidences() const {
    return frontierIncidences_;
  }
  const SyncCoverStorageProtocolRectangleStatistics &getStatistics() const {
    return statistics_;
  }
  SyncCoverStorageProtocolRectangleError getError() const { return error_; }
  bool isComplete() const {
    return error_ == SyncCoverStorageProtocolRectangleError::None &&
           !statistics_.truncated;
  }
  bool isForGraph(const SyncCoverGraph &graph) const {
    return ownerIdentity_ && ownerIdentity_ == graph.getIdentity() &&
           graph.isStructureFrozen();
  }

private:
  friend SyncCoverStorageProtocolRectangleIndex
  buildSyncCoverStorageProtocolRectangleIndex(
      const SyncCoverGraph &, const SyncCoverStorageProtocolAutomatonIndex &,
      const SyncCoverStorageProtocolFrontierIndex &,
      const SyncCoverStorageProtocolRectangleLimits &);

  void bindToGraph(const SyncCoverGraph &graph) {
    ownerIdentity_ = graph.getIdentity();
  }

  std::shared_ptr<const std::uint8_t> ownerIdentity_;
  std::vector<SyncCoverStorageProtocolRectangle> rectangles_;
  std::vector<SyncCoverStorageProtocolFrontierId> frontierIncidences_;
  SyncCoverStorageProtocolRectangleStatistics statistics_;
  SyncCoverStorageProtocolRectangleError error_ =
      SyncCoverStorageProtocolRectangleError::None;
};

/// Factor each bounded frontier plan by exact endpoint semantics. Construction
/// is deterministic and transactional: a structural or resource failure
/// publishes neither rectangles nor frontier incidences.
SyncCoverStorageProtocolRectangleIndex
buildSyncCoverStorageProtocolRectangleIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageProtocolAutomatonIndex &automatonIndex,
    const SyncCoverStorageProtocolFrontierIndex &frontierIndex,
    const SyncCoverStorageProtocolRectangleLimits &limits = {});

struct SyncCoverStorageProtocolRectangleGroundingDetail {
  SyncCoverStorageProtocolRectangleId rectangle = 0;
  SyncCoverStorageProtocolAutomatonId automaton = 0;
  std::size_t frontierCount = 0;
  std::size_t admittedDemands = 0;
  std::size_t coverageRows = 0;
};

struct SyncCoverStorageProtocolRectangleGroundingLimits {
  std::size_t maximumWorkUnits = std::size_t{1} << 34;
  std::size_t maximumAdmittedDemandIncidences = 1U << 22;
  std::size_t maximumBatchRectangles = 1U << 12;
  std::size_t maximumCoverageBatches = 1U << 14;
  std::size_t maximumDetails = 64;
  SyncCoverCoverageLimits coverageLimits;
};

struct SyncCoverStorageProtocolRectangleGroundingStatistics {
  std::size_t workUnits = 0;
  std::size_t evaluatedRectangles = 0;
  std::size_t coverageBatches = 0;
  std::size_t rectanglesWithCoverage = 0;
  std::size_t rectanglesCoveringMultipleRows = 0;
  std::size_t admittedDemandIncidences = 0;
  std::size_t maximumCoverageRows = 0;
  std::size_t totalCoverageRows = 0;
  bool detailsTruncated = false;
  bool truncated = false;
};

enum class SyncCoverStorageProtocolRectangleGroundingError : std::uint8_t {
  None,
  InvalidGraph,
  IncompleteAutomatonIndex,
  IncompleteFrontierIndex,
  IncompleteRectangleIndex,
  InvalidLimit,
  WorkLimitExceeded,
  IncidenceLimitExceeded,
  CoverageLimitExceeded,
  CoverageFailure,
  ArithmeticOverflow,
};

class SyncCoverStorageProtocolRectangleGrounding {
public:
  const SyncCoverStorageProtocolRectangleGroundingStatistics &
  getStatistics() const {
    return statistics_;
  }
  const std::vector<SyncCoverStorageProtocolRectangleGroundingDetail> &
  getDetails() const {
    return details_;
  }
  SyncCoverStorageProtocolRectangleGroundingError getError() const {
    return error_;
  }
  bool isComplete() const {
    return error_ == SyncCoverStorageProtocolRectangleGroundingError::None &&
           !statistics_.truncated;
  }

private:
  friend SyncCoverStorageProtocolRectangleGrounding
  groundSyncCoverStorageProtocolRectangles(
      const SyncCoverGraph &, const SyncCoverExpandedProgram &,
      const SyncCoverStorageProtocolAutomatonIndex &,
      const SyncCoverStorageProtocolFrontierIndex &,
      const SyncCoverStorageProtocolRectangleIndex &,
      const std::vector<SyncCoverDemandId> &,
      const SyncCoverStorageProtocolRectangleGroundingLimits &);

  SyncCoverStorageProtocolRectangleGroundingStatistics statistics_;
  std::vector<SyncCoverStorageProtocolRectangleGroundingDetail> details_;
  SyncCoverStorageProtocolRectangleGroundingError error_ =
      SyncCoverStorageProtocolRectangleGroundingError::None;
};

/// Stream exact-demand grounding for every compact protocol rectangle through
/// the ordinary completion oracle. No candidate-by-demand matrix is retained,
/// and results remain diagnostic only.
SyncCoverStorageProtocolRectangleGrounding
groundSyncCoverStorageProtocolRectangles(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    const SyncCoverStorageProtocolAutomatonIndex &automatonIndex,
    const SyncCoverStorageProtocolFrontierIndex &frontierIndex,
    const SyncCoverStorageProtocolRectangleIndex &rectangleIndex,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const SyncCoverStorageProtocolRectangleGroundingLimits &limits = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGEPROTOCOLRECTANGLES_H
