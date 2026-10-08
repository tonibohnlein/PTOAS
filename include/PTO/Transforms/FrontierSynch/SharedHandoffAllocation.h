// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One outstanding notification per numeric ID, shared across all directions.
#ifndef PTO_FRONTIERSYNCH_SHAREDHANDOFFALLOCATION_H
#define PTO_FRONTIERSYNCH_SHAREDHANDOFFALLOCATION_H
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
namespace mlir::pto::frontiersynch {
struct SharedHandoff {
    uint32_t sourcePipe = 0, targetPipe = 0;
};
struct SharedHandoffAllocation {
    std::string error;
    std::vector<uint32_t> lanes; // Input handoff index -> dense global lane index.
    uint64_t budget = 0;
    bool exactMinimum = false;
};
// successors[i] contains exactly the j for which WAIT(i) precedes SET(j).
// This must be an irreflexive transitive relation. It includes command order
// on one pipe; direction labels alone never establish safe reuse. The caller
// supplies a fixed valid plan with matching, fresh entry and closed exit.
// Indices may be in arbitrary order. Duplicate edges are harmless; invalid
// indices, cycles and incomplete transitive closure are rejected.
// Walk a stable topological order, preferring a free lane last used in the
// same direction, then another free lane, then a new lane. If that exceeds
// capacity, maximum bipartite matching computes an exact minimum chain cover.
// Return the assignment and budget even when the exact minimum exceeds
// capacity. No ordering edges, drains, or repairs are introduced. Capacity is
// a count; mapping lanes onto eligible physical IDs belongs to the caller.
SharedHandoffAllocation allocateSharedHandoffs(
    llvm::ArrayRef<SharedHandoff> handoffs, llvm::ArrayRef<std::vector<uint32_t>> successors, uint64_t capacity);
// Certified-query entry point: input indices are already a topological order
// of an exact strict reuse relation. The supplier proves this contract; unlike
// the matrix API, successful greedy allocation does not inspect unused pairs.
// Greedy makes at most capacity tail queries per handoff and retains O(h+E)
// words. On failure only, materialize the forward relation, validate it, and
// compute the exact minimum by matching. An empty callback is rejected.
SharedHandoffAllocation allocateSharedHandoffsByQuery(
    llvm::ArrayRef<SharedHandoff> handoffs,
    const std::function<bool(uint32_t, uint32_t)>& precedes, uint64_t capacity);
} // namespace mlir::pto::frontiersynch
#endif
