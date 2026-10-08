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
#include <array>
#include <memory>
#include <utility>
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
    uint64_t operations = 0; // Charged numerical sweep/projection word operations.
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
// Output IDs follow this selection; each selected child event occurs once.
struct NumericalChainSelection { uint32_t child = 0, event = 0; };
struct NumericalChainQueryCost { uint64_t leafQueries = 0, indexOperations = 0; };
struct NumericalCrossingGroup {
    // Source rank / suffix minimum target rank; target rank / prefix maximum
    // source rank+1. Lists contain only occupied ranks, in increasing order.
    std::vector<std::pair<uint32_t, uint32_t>> forward, reverse;
};
struct NumericalChainMerge {
    NumericalChainInterface index;
    std::array<std::shared_ptr<const NumericalChainInterface>, 2> children;
    std::array<std::vector<uint32_t>, 2> parentChains;
    std::array<std::vector<std::vector<uint32_t>>, 2> selectedRanks, parentRanks;
    std::vector<std::vector<NumericalCrossingGroup>> crossings;
    std::optional<std::vector<uint32_t>> propagate(uint32_t child,
        const std::vector<uint32_t>& thresholds, bool reverse, NumericalChainQueryCost& cost) const;
    std::optional<bool> crosses(const std::vector<uint32_t>& forward,
        const std::vector<uint32_t>& reverse, NumericalChainQueryCost& cost) const;
};
// Child chain maps assign native chains to a common dense output directory.
// Each child map is injective. Within an output chain all left events precede
// all right events. Supply native links among the deduplicated crossings too.
// Construction uses O(kP+k^2*r) numerical work, no semantic callbacks. It
// retains only reduced links and O(kP) indices; selected IDs need not be sorted.
NumericalChainMerge buildNumericalChainMerge(
    std::shared_ptr<const NumericalChainInterface> left, std::shared_ptr<const NumericalChainInterface> right,
    const std::vector<NumericalCrossing>& crossings,
    const std::vector<NumericalChainSelection>& selection,
    const std::vector<uint32_t>& leftChains, const std::vector<uint32_t>& rightChains);
// Binary-search leaf thresholds. Query receives a port ID and reverse flag:
// false asks event->port; true asks port->event. Null means unavailable.
std::optional<std::vector<uint32_t>> numericalLeafThresholds(const NumericalChainInterface& index,
    const std::function<std::optional<bool>(uint32_t, bool)>& query, bool reverse,
    NumericalChainQueryCost& cost);
// Endpoints are local IDs in left/right indices. Crossings must already be
// deduplicated; all fixed native crossings participate in the tests as well.
// Returns cover bits, with O(kP+k^2*r) work and O(P+r) scratch beyond indices.
std::optional<std::vector<bool>> reduceNumericalCrossings(
    const NumericalChainInterface& left, const NumericalChainInterface& right,
    const std::vector<NumericalCrossing>& crossings);
} // namespace mlir::pto::frontiersynch
#endif
