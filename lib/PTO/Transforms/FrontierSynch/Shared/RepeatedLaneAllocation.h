// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_REPEATEDLANEALLOCATION_H
#define PTO_FRONTIERSYNCH_REPEATEDLANEALLOCATION_H
#include "RepeatedAllocation.h"
namespace mlir::pto::frontiersynch {
struct RepeatedLaneProofCost {
    RepeatedExtremaCost extrema;
    uint64_t implicationRequests = 0, implicationComputations = 0;
    uint64_t constantRequests = 0, constantComputations = 0;
    uint64_t proofNodeVisits = 0, relativeQueries = 0, relativeComputations = 0;
    uint64_t proofNanoseconds = 0, relativeNanoseconds = 0;
    uint64_t lanes = 0, firstAlternatives = 0, lastAlternatives = 0;
    uint64_t matrixPairs = 0, candidateGaps = 0, boundaryPairs = 0;
    uint64_t partitionAttempts = 0, partitionMatrixEntries = 0;
};
// Allocate certified unit child lanes and boundary handoffs as a periodic word.
// Only proved uniform reuse edges are admitted. Failure is an unavailable
// certificate, never a proof that the physical ID capacity is insufficient.
std::shared_ptr<RegionalAllocationSummary> periodicLaneAllocation(
    RepeatedRegionState& state, const RegionalAllocationSummary& coalesced,
    ArrayRef<std::pair<uint32_t, std::size_t>> crossings, RepeatedLaneProofCost* cost = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
