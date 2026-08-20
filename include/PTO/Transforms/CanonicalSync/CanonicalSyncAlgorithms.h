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
#include <vector>

namespace mlir {
namespace pto {

struct SyncInterval {
  std::size_t begin = 0;
  std::size_t end = 0;
};

struct SyncColoring {
  std::vector<unsigned> colors;
  unsigned colorCount = 0;
};

/// Optimally color inclusive intervals. Intervals that share an endpoint
/// overlap and therefore receive different colors.
SyncColoring colorSyncIntervals(const std::vector<SyncInterval> &intervals);

enum class SyncGraphEdgeKind : std::uint8_t {
  IssueOrder,
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

/// Returns one keep bit per completion requirement. A requirement is removed
/// only when another path from its source to target contains a hardware or
/// retained completion edge. Issue-order-only paths never satisfy a hazard.
/// Invalid or non-forward requirements receive a false keep bit.
std::vector<bool> reduceCompletionRequirements(
    std::size_t vertexCount, const std::vector<SyncGraphEdge> &fixedEdges,
    const std::vector<CompletionRequirement> &requirements);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCALGORITHMS_H
