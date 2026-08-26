// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CanonicalSyncAlgorithms.h - Sync graph algorithms --------*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCALGORITHMS_H
#define MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCALGORITHMS_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

struct SyncInterval {
  std::size_t begin = 0;
  std::size_t end = 0;
};

struct SyncAllocatedInterval {
  SyncInterval interval;
  unsigned eventId = 0;
};

struct SyncWeightedInterval {
  SyncInterval interval;
  std::size_t width = 1;
};

enum class SyncIntervalPressureError : std::uint8_t {
  None,
  InvalidInterval,
  ArithmeticOverflow,
};

struct SyncIntervalPressure {
  SyncIntervalPressureError error = SyncIntervalPressureError::None;
  std::size_t required = 0;
  std::size_t available = 0;
  std::size_t overflow = 0;
  std::optional<std::size_t> maximumPoint;
  /// Input indices active at the deterministic maximum-pressure point.
  std::vector<std::size_t> maximumClique;

  explicit operator bool() const {
    return error == SyncIntervalPressureError::None;
  }
};

/// Verify a physical-ID assignment for one pipe domain. Interval endpoints
/// are inclusive, and reserved IDs cannot be assigned.
bool verifySyncIntervalAllocation(
    unsigned eventIdMax, const std::vector<unsigned> &reservedIds,
    const std::vector<SyncAllocatedInterval> &allocation);

/// Return the deterministic maximum clique of an interval graph. The result
/// contains interval indices in ascending order. Inclusive endpoints overlap.
std::vector<std::size_t>
findMaximumIntervalClique(const std::vector<SyncInterval> &intervals);

/// Compute exact pressure for inclusive weighted intervals. In-range reserved
/// IDs reduce availability once; out-of-range IDs remain diagnostic only.
/// Invalid intervals, zero widths, and active-width overflow fail closed.
SyncIntervalPressure
evaluateSyncIntervalPressure(const std::vector<SyncWeightedInterval> &intervals,
                             std::size_t budget,
                             const std::vector<unsigned> &reservedIds = {});

enum class SyncGraphEdgeKind : std::uint8_t {
  /// Same-pipe issue order: completion of the later operation also proves
  /// completion of earlier operations on that pipe.
  IssueOrder,
  /// Cross-pipe sequencer order. It does not propagate completion because a
  /// wait stalls only its destination pipe.
  NonCompletionPreservingIssueOrder,
  HardwareCompletion,
};

struct SyncGraphEdge {
  std::size_t source = 0;
  std::size_t target = 0;
  SyncGraphEdgeKind kind = SyncGraphEdgeKind::IssueOrder;
};

struct CompletionRequirement {
  std::size_t source = 0;
  std::size_t target = 0;
};

enum class SyncTokenActionKind : std::uint8_t { Set, Wait };

struct SyncTokenAction {
  SyncTokenActionKind kind = SyncTokenActionKind::Set;
  unsigned lane = 0;
};

/// Verify one finite event-token trace. Each lane is a one-bit token: set
/// transitions empty to full and wait transitions full to empty. Initial and
/// expected full-lane lists must be duplicate-free and in range.
bool verifySyncTokenTrace(unsigned laneCount,
                          const std::vector<unsigned> &initiallyFullLanes,
                          const std::vector<SyncTokenAction> &actions,
                          const std::vector<unsigned> &expectedFullLanes);

using CompletionVertexFilter =
    std::function<bool(std::size_t requirement, std::size_t vertex)>;

/// Returns one keep bit per completion requirement. A requirement is removed
/// only when another path from its source to target contains a hardware or
/// retained completion edge. Issue-order-only paths never satisfy a hazard.
/// Invalid or non-forward requirements receive a false keep bit. When supplied,
/// `isVertexAvailable` restricts each candidate's proof to vertices guaranteed
/// to execute in that requirement's control-flow context.
std::vector<bool> reduceCompletionRequirements(
    std::size_t vertexCount, const std::vector<SyncGraphEdge> &fixedEdges,
    const std::vector<CompletionRequirement> &requirements,
    const CompletionVertexFilter &isVertexAvailable = {});

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCALGORITHMS_H
