// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_CONTROLORIGINDISTANCES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_CONTROLORIGINDISTANCES_H
#include "PTO/Transforms/FrontierSynch/OriginDistances.h"
#include "llvm/ADT/ArrayRef.h"
#include <string>
#include <vector>
namespace mlir::pto::frontiersynch {
struct ControlOriginEdge {
    uint32_t source = 0, target = 0;
    uint8_t weight = 0; // Zero within one iteration; one on an iteration advance.
};
struct ControlPayloadCuts { uint32_t input = 0, output = 0; };
struct ControlOriginGraph {
    uint32_t vertices = 0;
    std::vector<ControlPayloadCuts> payloads;
    // Payload input->output through edges are implicit, zero weight, and unique.
    // Control edges may leave outputs and enter inputs, but never bypass a
    // payload through edge by leaving its input or entering its output.
    std::vector<ControlOriginEdge> edges;
};
struct ControlStorageOrigin {
    uint32_t payload = 0, storageClass = 0;
    // Producer-certified definite FULL overwrites of this origin's class.
    // The list is class-specific; partial/may writes must never occur here.
    // Remove only these payloads' through edges. Their input queries remain.
    std::vector<uint32_t> definiteKillPayloads;
};
struct ControlOriginCost {
    uint64_t vertices = 0, edges = 0, origins = 0;
    uint64_t vertexVisits = 0, edgeVisits = 0;
};
struct ControlOriginDistances {
    std::string error;
    // Original origin order, then graph vertex. Query a payload at its input.
    // Rows cover every vertex, including unreachable vertices and the origin's
    // own output (the empty path has weight zero). No dynamic trip expansion.
    std::vector<std::vector<OriginDistanceInterval>> distances;
    ControlOriginCost cost;
};
// Finite control abstraction only; this does not prove executable matching or
// source presence for skipped payloads. Origins begin at their output cut, so
// a killed source still generates fresh state and a killed target is queried
// before its overwrite. Zero-weight returns to the origin's input are rejected:
// revisiting one payload must advance an iteration in this body abstraction.
// Work O(V+E + z*(V+E)), scratch O(V+E), output O(z*V), including per-origin
// kill masks (each contains at most one entry per payload). Uses 0/1 shortest
// paths, reachable SCCs, positive-cycle descendants and DAG longest paths.
// Weights and all finite distances are checked/bounded; no solver, path search
// per query, effect interpretation, guard valuation or iteration unrolling.
ControlOriginDistances computeControlOriginDistances(
    const ControlOriginGraph& graph, llvm::ArrayRef<ControlStorageOrigin> origins);
} // namespace mlir::pto::frontiersynch
#endif
