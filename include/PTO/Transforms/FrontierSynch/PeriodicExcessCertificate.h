// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact finite-prefix excess counts from immutable periodic frontier snapshots.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICEXCESSCERTIFICATE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICEXCESSCERTIFICATE_H
#include "PTO/Transforms/FrontierSynch/BoundingRegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
#include "llvm/ADT/APInt.h"
namespace mlir::pto::frontiersynch {
class CompactFixedBodyContext;
using CompactFixedBody = std::shared_ptr<const CompactFixedBodyContext>;
class PeriodicExcessGraph;
using PeriodicExcessSnapshot = std::shared_ptr<const PeriodicExcessGraph>;
// Trusted producer boundary: the supplied index is exact for its records on this
// context's fixed body, with every type present in every complete iteration.
// Numerical distances and native prerequisites hold uniformly for ALL prefixes;
// instantiating an unbound parameter at sampled values does not meet this contract.
// Validation checks representation, not graph/index equivalence or input effects.
class PeriodicExcessGraph {
public:
    const OrderContext& context() const;
    const CompactFixedBody& binding() const { return owner; }
    const PeriodicAnalysis& analysis() const { return index; }
    ReductionQuality reduction() const { return quality; }
private:
    PeriodicExcessGraph(CompactFixedBody context, PeriodicAnalysis analysis, ReductionQuality reduction);
    CompactFixedBody owner;
    PeriodicAnalysis index;
    ReductionQuality quality;
    friend PeriodicExcessSnapshot bindPeriodicExcessGraph(
        CompactFixedBody, const PeriodicAnalysis&, ReductionQuality, std::string&);
};
PeriodicExcessSnapshot bindPeriodicExcessGraph(CompactFixedBody context, const PeriodicAnalysis& analysis,
                                             ReductionQuality reduction, std::string& error);
struct PeriodicExcessSummary {
    std::string error;
    uint64_t trips = 0;
    // 256 bits suffice: m,k <= 2^32, T,distances <= 2^64. Even the coarse
    // T*m*m*k*delta bound is <2^224; all Phi/count intermediates are <2^193.
    llvm::APInt excess{256, 0}, quadraticPairs{256, 0};
    std::optional<llvm::APInt> gamma, linearBound, counterpartBound;
    std::optional<uint64_t> counterpartDelta;
    bool sameSupport = false, equalFrontiers = false;
    uint64_t frontierEntries = 0, supportSteps = 0;
};
// Pure numerical graph summary. Both indices have the same fixed word/native
// records and are certified exact by their producer. Equality here says nothing
// about an original shared input, optional occurrences or unbound parameters.
PeriodicExcessSummary analyzePeriodicExcess(const PeriodicAnalysis& upper,
                                           const PeriodicAnalysis& lower, uint64_t trips);
struct PeriodicExcessCertificate : PeriodicExcessSummary {
    PeriodicExcessSnapshot upper, lower;
    bool exactOriginalCovers = false;
    InputOrderGuarantee guarantee = InputOrderGuarantee::InputOrderCovering;
};
// Caller certifies Hlower <= Hinput <= Hupper on the unchanged shared context.
// With no lower snapshot, use the common native graph (valid without storage
// lower facts). Its indexing cost is additional and uses the existing analyzer.
// The result binds the EXACT snapshots; do not transfer promotion to another
// selected graph merely because its context pointer matches. No input is mutated.
// Post-index work O(m*k + k^3 + g+ + g-), space O(k^2), including structure
// validation; no type-pair queries or loops over trips/numerical distances.
// Frontier equality is uniform under the snapshot contract, independently of T.
PeriodicExcessCertificate certifyPeriodicExcess(PeriodicExcessSnapshot upper,
                                                PeriodicExcessSnapshot lower, uint64_t trips);
std::string periodicExcessDecimal(const llvm::APInt& value);
} // namespace mlir::pto::frontiersynch
#endif
