// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REPEATEDEXCESS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REPEATEDEXCESS_H
#include "PTO/Transforms/FrontierSynch/BoundaryExcess.h"
#include <optional>
namespace mlir::pto::frontiersynch {
struct ThresholdRankRectangle {
    uint64_t width = 0, height = 0, activation = 1;
};
struct ThresholdRectangleSum {
    std::string error;
    llvm::APInt weighted{256, 0};
    uint64_t comparisons = 0, skylineUpdates = 0;
};
// Sum the active common-corner union over separations s=1..N, weighting
// each area by N-s when fullPeriods=true and by one otherwise. Activation0
// means activation1. A coordinate-compressed ordered skyline inserts/deletes
// each breakpoint at most once: O(r log(r+2)), O(r) scratch, no N-sized loop.
ThresholdRectangleSum sumThresholdRectangles(llvm::ArrayRef<ThresholdRankRectangle> rectangles,
                                             uint64_t completePeriods, bool fullPeriods);
struct RepeatedExcessInput {
    BoundaryExcessRegion period;
    // q phase prefixes: entry h describes the first h phases, 0<=h<q.
    // Prefix0 has zero counts/bound; counts are monotone and <=period counts.
    std::vector<BoundaryExcessRegion> prefixes;
    std::vector<bool> present; // Common evaluated port frame for both signs.
    BoundaryExcessSelectors lower, upper;
};
using RepeatedPortDistances = std::vector<std::vector<std::optional<uint64_t>>>;
struct RepeatedExcessResult {
    std::string error;
    // Entry h counts N complete periods followed by prefix h (T=q*N+h).
    // cross includes full/full and full/prefix pairs. Internal certificates
    // equal to internal differences make exactDifference true, not a claim
    // that the lower graph equals the original input or realized hardware.
    std::vector<BoundaryExcessResult> prefixes;
    uint64_t thresholdEntries = 0, rectangles = 0, comparisons = 0, skylineUpdates = 0;
};
// Evaluated proof boundary, as in BoundaryExcess: exact selected lower/upper
// distances, ranks, counts, presence and internal certificates belong to ONE
// immutable common period domain. Every native/storage/value crossing required
// by each specification is included, including nonadjacent-period demands.
// Lower is contained in upper; arbitrary varying interiors are not admitted by
// this contract. Distances include intra-period closure; null is unreachable
// within uint64 separation (including a certified finite distance >UINT64_MAX).
// This checks representation/containment, not effect coverage or graph synthesis.
// Sweep work after indexing/rank evaluation O(q*k^2*(P+1)^2*log(P+2));
// input-prefix validation O(q*(k+1)); O(P^2+k+1) scratch, plus q output bounds.
// All arithmetic is checked APInt256; excessive supplied bounds return an error.
RepeatedExcessResult countRepeatedExcess(const RepeatedExcessInput& input,
    const RepeatedPortDistances& lower, const RepeatedPortDistances& upper, uint64_t completePeriods);
struct RepeatedChainFrontiers {
    std::vector<std::vector<uint32_t>> chains; // Common native event chains, all ports present.
    // [source chain][target port], rho=h-beta >=0; null is unreachable.
    std::vector<std::vector<std::optional<uint64_t>>> lower, upper;
};
// Same certificate using at most two last-source representatives per
// chain/target. <=2*k native chains, identical to the supplied rank-selector
// order, partition all ports. Sweep O(q*k^3*(P+1)*log(P+2)), validation
// O(q*(k+1)+k*P); O(k*(P+1)+1) scratch, plus q output bounds;
// does not construct all-pairs distances solely for counting.
RepeatedExcessResult countChainRepeatedExcess(const RepeatedExcessInput& input,
    const RepeatedChainFrontiers& frontiers, uint64_t completePeriods);
} // namespace mlir::pto::frontiersynch
#endif
