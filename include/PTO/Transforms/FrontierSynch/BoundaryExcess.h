// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Evaluated cross-region excess without enumerating internal occurrences.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDARYEXCESS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDARYEXCESS_H
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
namespace mlir::pto::frontiersynch {
struct BoundaryExcessRegion {
    std::vector<uint64_t> counts; // One common dense pipe directory.
    llvm::APInt internalBound{256, 0}; // Unsigned supplied certificate; never a sampled estimate.
    bool exactInternalDifference = false;
};
struct BoundaryExcessPort {
    uint32_t region = 0;
    bool present = true; // Same evaluated occurrence domain for BOTH signs.
};
struct BoundaryExcessSelectors {
    // Exact child-internal selectors at each common port and payload pipe.
    // prefix=lambda; suffixExcluded=eta-1. The latter is in [0,n], avoiding
    // overflow of the usual absent sentinel n+1 when n=UINT64_MAX.
    std::vector<std::vector<uint64_t>> prefix, suffixExcluded;
};
struct BoundaryExcessInput {
    uint32_t pipes = 0;
    std::vector<BoundaryExcessRegion> regions;
    std::vector<BoundaryExcessPort> ports;
    BoundaryExcessSelectors lower, upper;
};
struct BoundaryRankRectangle {
    uint64_t sourcePrefix = 0, targetExcluded = 0;
};
struct BoundaryRectangleCount {
    std::string error;
    llvm::APInt pairs{256, 0};
    uint64_t comparisons = 0;
};
// Union of [1,sourcePrefix] x (targetExcluded,consumers]. All endpoints are
// inclusive integer ranks as indicated. O(r log(r+2)) work/O(r) scratch, with
// no arithmetic loop over any rank. Empty/duplicate/nested rectangles are valid.
BoundaryRectangleCount countBoundaryRectangles(llvm::ArrayRef<BoundaryRankRectangle> rectangles,
                                               uint64_t consumers);
struct BoundaryExcessResult {
    std::string error;
    llvm::APInt internal{256, 0}, cross{256, 0}, total{256, 0};
    bool exactDifference = false;
    uint64_t portPairs = 0, rectangles = 0, comparisons = 0, indexOperations = 0;
};
using BoundaryPortClosure = std::vector<std::vector<bool>>;
// Evaluated numerical proof boundary: both exact port closures include their
// own child-internal orders and every required native crossing (even across
// empty regions). Selectors, counts, guards and internal certificates MUST come
// from one immutable execution context and the same lower/upper snapshots.
// The caller certifies the bounding sandwich; this engine checks shape,
// forward-region order and lower containment, not source effects or closure
// synthesis. It never joins incompatible branch valuations. Missing capability
// is an error and does not mutate inputs or discard mathematical records.
// After supplied closures/selectors: O(t+k^2*P^2*log(P+2)) operations. Counts may
// be UINT64_MAX. APInt256 arithmetic is checked, including all internal sums.
BoundaryExcessResult countBoundaryExcess(const BoundaryExcessInput& input,
    const BoundaryPortClosure& lower, const BoundaryPortClosure& upper);
// Binary A;B specialization, with BOTH indices on the SAME complete child-port
// frame, retaining certificate-only ports. All ports must be present in this
// evaluated context. Chains are certified native event chains; A ports precede
// B ports on each chain. At most 2*k chains (start/completion per pipe).
// Reverse thresholds select only the last reaching A port on each chain.
// O(k^3*(P+1)*log(P+2)) work and O(k*P+1) scratch beyond supplied indices/ranks.
BoundaryExcessResult countBinaryBoundaryExcess(const BoundaryExcessInput& input,
    const NumericalChainInterface& lower, const NumericalChainInterface& upper);
} // namespace mlir::pto::frontiersynch
#endif
