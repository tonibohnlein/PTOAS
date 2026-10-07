// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_CHAININTERFACE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_CHAININTERFACE_H
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>
namespace mlir::pto::frontiersynch {
struct NumericalChainInterface {
    std::string error;
    std::vector<std::vector<uint32_t>> chains;
    std::vector<uint32_t> chain, rank;
    // First reachable target (chain length means absent), last ancestor+1
    // (zero means absent). Event identity is shared across coincident roles.
    std::vector<std::vector<uint32_t>> forward, reverse;
    uint64_t queries = 0;
    bool reaches(uint32_t source, uint32_t target) const;
};
// Input lists are disjoint, dense IDs, in native chain order. The callback
// supplies exact reflexive reachability; its native-order property justifies
// the monotone sweeps. Sorting and identity normalization belong to the caller.
NumericalChainInterface buildNumericalChainInterface(
    std::vector<std::vector<uint32_t>> chains, const std::function<std::optional<bool>(uint32_t, uint32_t)>& query);
struct NumericalCrossing {
    uint32_t source, target;
};
// Endpoints are local IDs in left/right indices. Crossings must already be
// deduplicated; all fixed native crossings participate in the tests as well.
// Returns cover bits, with O(kP+k^2*r) work and O(P+r) scratch beyond indices.
std::optional<std::vector<bool>> reduceNumericalCrossings(
    const NumericalChainInterface& left, const NumericalChainInterface& right,
    const std::vector<NumericalCrossing>& crossings);
} // namespace mlir::pto::frontiersynch
#endif
