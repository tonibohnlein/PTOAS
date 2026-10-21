// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite source-pipe projection of union/GEN/class-wide-KILL transfers.
// No origin identities, distances, readiness, or executable cuts are inferred.
#ifndef PTO_FRONTIERSYNCH_SOURCEPIPESERVICES_H
#define PTO_FRONTIERSYNCH_SOURCEPIPESERVICES_H
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>
namespace mlir::pto::frontiersynch {
struct SourcePipeFact {
    std::size_t cell = 0;
    uint32_t pipe = 0;
    uint8_t mode = 0;
    bool operator==(const SourcePipeFact& b) const { return cell == b.cell && pipe == b.pipe && mode == b.mode; }
};
struct SourcePipeFactHash {
    std::size_t operator()(const SourcePipeFact& fact) const;
};
using SourcePipeFacts = std::unordered_set<SourcePipeFact, SourcePipeFactHash>;
struct SourcePipeSite {
    uint32_t pipe = 0;
    std::vector<uint8_t> modes;
    // Supplied full-class refresh only. Selection reads IN before this KILL.
    std::vector<std::size_t> refreshed;
};
struct SourcePipeNode {
    std::optional<std::size_t> site;
    std::vector<std::size_t> successors;
};
struct SourcePipeWork {
    uint64_t preparedModes = 0, preparedEdges = 0, generated = 0;
    uint64_t incoming = 0, outgoing = 0, transmissions = 0, killed = 0, inspected = 0;
};
struct SourcePipeServices {
    std::vector<SourcePipeFacts> incoming, outgoing;
    std::vector<std::vector<uint32_t>> possible;
    SourcePipeWork work;
    std::string error;
};
// All nodes belong to the supplied possible CFG (including zero-trip paths).
// Each new OUT node/fact is queued once and transmitted once per outgoing edge.
// Expected hash work O((1+hk)(V+E)+descriptor preparation); memory O(hkV+V+E).
// Caller owns access coverage and refresh proofs. Invalid inputs return no states.
SourcePipeServices solveSourcePipeServices(
    const std::vector<SourcePipeNode>& graph, const std::vector<SourcePipeSite>& sites, std::size_t classes);
} // namespace mlir::pto::frontiersynch
#endif
