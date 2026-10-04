// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numerical periodic generators and their completion-origin frontier index.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICANALYSIS_H
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
namespace mlir::pto::frontiersynch {
struct PeriodicPayload {
    uint32_t pipe = 0;
};
struct PeriodicRecord {
    uint32_t source = 0;
    uint32_t target = 0;
    uint64_t displacement = 0; // Whole periods, not cycles or payload ranks.
};
enum class PeriodicEventKind { Start, Completion };
struct PeriodicEvent {
    uint32_t type = 0;
    PeriodicEventKind kind = PeriodicEventKind::Start;
};
enum class PeriodicQueryError { None, InvalidInput, Overflow };
struct PeriodicThreshold {
    PeriodicQueryError error = PeriodicQueryError::None;
    std::optional<uint64_t> displacement; // Empty with no error means unreachable.
};
struct PeriodicPredicate {
    PeriodicQueryError error = PeriodicQueryError::None;
    bool value = false;
};
struct PeriodicRank {
    PeriodicQueryError error = PeriodicQueryError::None;
    uint64_t value = 0;
};
struct PipeFrontier {
    uint32_t pipe = 0;
    uint32_t count = 0;
    // Vertex 2*a is I_a, vertex 2*a+1 is C_a. Null is unreachable.
    std::vector<std::optional<uint64_t>> distances;
};
struct PeriodicAnalysis {
    std::string error;
    std::vector<PeriodicPayload> payloads;
    std::vector<PeriodicRecord> generators; // Canonical endpoint/distance deduplication.
    std::vector<uint32_t> retained; // Indices into generators, exactly F* records.
    std::vector<uint32_t> sourceRows;
    std::vector<uint32_t> localRanks;
    std::vector<PipeFrontier> frontiers;
    std::unordered_map<uint32_t, uint32_t> pipeRows; // Expected constant-time pipe lookup.
    uint64_t graphEdges = 0;
    // Present-event queries only. The numerical threshold itself is uniform
    // in the trip count; finite-prefix helpers explicitly filter endpoints.
    PeriodicThreshold completionThreshold(uint32_t source, PeriodicEvent target) const;
    PeriodicPredicate completionPrecedes(uint32_t source, uint64_t sourcePeriod,
                                        PeriodicEvent target, uint64_t targetPeriod,
                                        uint64_t payloadPrefixLength, bool strict = false) const;
    PeriodicRank completionRank(uint32_t pipe, PeriodicEvent target, uint64_t targetPeriod,
                                uint64_t payloadPrefixLength) const;
};
// Types occur in array order within a period. Nonzero-displacement generators
// are forward; zero-displacement generators require source < target. Includes
// native singleton wraps. Exactness relative to effects needs an external
// generator certificate. No start-origin query or physical allocation export.
// On invalid input or checked integer overflow, no partial index is returned.
PeriodicAnalysis analyzePeriodicDemands(llvm::ArrayRef<PeriodicPayload> payloads,
                                       llvm::ArrayRef<PeriodicRecord> records);
} // namespace mlir::pto::frontiersynch
#endif
