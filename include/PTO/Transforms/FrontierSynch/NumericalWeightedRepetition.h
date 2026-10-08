// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICALWEIGHTEDREPETITION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICALWEIGHTEDREPETITION_H
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
#include "llvm/ADT/APInt.h"
namespace mlir::pto::frontiersynch {
struct NumericalWeightedCrossing {
    uint32_t source = 0, target = 0;
    uint64_t displacement = 1;
    bool native = false;
};
struct NumericalWeightedDistance {
    bool reachable = false;
    // Absent with reachable=true means a finite distance exceeding uint64.
    std::optional<uint64_t> displacement;
};
struct NumericalWeightedRepetitionCost {
    uint64_t validation = 0, internalEdges = 0, crossingEdges = 0;
    uint64_t heapPushes = 0, heapPops = 0, relaxations = 0;
    uint64_t prefixEntries = 0, coverQueries = 0;
};
struct NumericalWeightedRepetitionIndex {
    NumericalChainInterface body;
    // rho[c][v] = min_{u in c}(h_c-1-rank(u)+h_c*D(u,v)).
    // Distinct unreachable entries stay absent. Arithmetic uses checked 128
    // bits; P<=UINT32_MAX bounds every simple-path scaled sum below 2^128.
    std::vector<std::vector<std::optional<llvm::APInt>>> rho;
    std::optional<NumericalWeightedDistance> distance(uint32_t source, uint32_t target) const;
    std::optional<bool> query(uint32_t source, uint32_t target, uint64_t gap) const;
    // Thresholds come from numericalLeafThresholds(body,...): sourceForward
    // gives first reachable ranks, targetReverse gives last ancestor rank+1.
    // Across visits only; at gap zero ask the original arbitrary-event query.
    // O(c^2) indexed probes, c<=2k; the caller charges its child binary searches.
    std::optional<bool> across(const std::vector<uint32_t>& sourceForward,
        const std::vector<uint32_t>& targetReverse, uint64_t gap, uint64_t& probes) const;
};
struct NumericalWeightedRepetition {
    std::string error;
    std::shared_ptr<const NumericalWeightedRepetitionIndex> index;
    // Sorted by (source,target,displacement). Native duplicates take priority.
    std::vector<NumericalWeightedCrossing> crossings;
    std::vector<uint32_t> originalToCanonical, representatives;
    std::vector<bool> retained; // Native entries remain true regardless of redundancy.
    NumericalWeightedRepetitionCost cost;
};
// The supplier certifies an evaluated, exact reflexive child order with
// distinct present port IDs on native chains. Coincident semantic identities
// must be deduplicated first. Structure, reverse consistency, native order and
// the absence of nontrivial zero cycles are checked; exact child semantics are
// the input contract, not re-proved by a cubic transitive-closure computation.
// Every nonempty chain requires a supplied native last->first displacement-1
// crossing. Additional crossings have positive uint64 weights and guards have
// already been evaluated. Native carries justify padding any reachable gap.
//
// Uses one rank-scaled multisource Dijkstra per chain and shared last-crossing
// prefixes with two distinct record identities. O(c*(cP+r)*log(P+r+2)+1)
// numerical work, O(cP+r+1) words, excluding construction of the supplied child
// interface and integer bit costs. Never expands visits or crossing weights.
// The immutable result owns a copy of the interface. Failure publishes no index
// or partial covers. Empty input succeeds. Counts charge graph/heap/prefix work;
// comparison sorting and binary searches have the stated logarithmic cost.
NumericalWeightedRepetition buildNumericalWeightedRepetition(
    const NumericalChainInterface& body, const std::vector<NumericalWeightedCrossing>& crossings);
} // namespace mlir::pto::frontiersynch
#endif
