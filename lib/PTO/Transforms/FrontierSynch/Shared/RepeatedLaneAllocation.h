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
    uint64_t coalescingNanoseconds = 0;
    uint64_t activityRequests = 0, activityComputations = 0, activityNanoseconds = 0;
    uint64_t implicationRequests = 0, implicationComputations = 0;
    uint64_t constantRequests = 0, constantComputations = 0;
    uint64_t proofNodeVisits = 0, relativeQueries = 0, relativeComputations = 0;
    uint64_t proofNanoseconds = 0, relativeNanoseconds = 0;
    uint64_t lanes = 0, firstAlternatives = 0, lastAlternatives = 0;
    uint64_t orderedAttempts = 0, orderedPhases = 0, orderedReuseQueries = 0, orderedAccepted = 0;
    uint64_t orderedOverCapacity = 0, orderedUnresolved = 0, orderedWitnessQueries = 0;
    uint64_t matrixPairs = 0, candidateGaps = 0, boundaryPairs = 0;
    uint64_t partitionAttempts = 0, partitionMatrixEntries = 0;
};
enum class RepeatedLaneAllocationStatus {
    CertifiedAssignment, CertifiedOverCapacity, NotApplicable, UnresolvedQuery
};
// A finite antichain of whole-span reservations under the selected payload
// order. This is not a hardware minimum or a restriction on lending inside a
// span. This is attempt-local diagnostic evidence, not a reusable allocation
// certificate: its arena owner does not own the original repeated-state frame.
// That frame and original IR must remain alive and unchanged while inspecting it.
struct RepeatedLaneCapacityWitness {
    std::shared_ptr<const RegionExpressions> expressions;
    uint32_t sourcePipe = 0, targetPipe = 0;
    uint64_t phasesPerVisit = 0, consumerDelay = 0, admittedTrips = 0;
    struct Span {
        RegionalEvent source, target;
        uint64_t sourceVisit = 0, targetVisit = 0;
        std::vector<uint32_t> records;
    };
    std::vector<Span> spans;
    // Every forward pair is certified absent; reverse pairs are excluded by
    // original reference order. Negative-displacement pairs need no query.
    std::vector<RegionExpressions::Id> negativeReuse;
};
struct RepeatedLaneAllocationResult {
    RepeatedLaneAllocationStatus status = RepeatedLaneAllocationStatus::NotApplicable;
    std::shared_ptr<RegionalAllocationSummary> assignment;
    std::shared_ptr<const RepeatedLaneCapacityWitness> overCapacity;
};
// Allocate certified unit child lanes and boundary handoffs as a periodic word.
// Only proved uniform reuse edges are admitted. Overcapacity applies solely to
// the owned finite whole-span witness; other failures have no capacity meaning.
RepeatedLaneAllocationResult periodicLaneAllocation(
    RepeatedRegionState& state, const RegionalAllocationSummary& coalesced,
    ArrayRef<std::pair<uint32_t, std::size_t>> crossings, RepeatedLaneProofCost* cost = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
