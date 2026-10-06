// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICSELECTORS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICSELECTORS_H
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include <optional>
namespace mlir::pto::frontiersynch {
struct ArithmeticSelectorTerm {
    // Matrix coordinate: zero denotes constant zero; 1..inputDimensions are
    // endpoint quotients, followed by the common parameter quotient tuple.
    unsigned input = 0;
    BoundInteger offset;
};
struct ArithmeticSelectorOutput {
    // The unique output quotient is max(input[term.input] + term.offset).
    std::vector<ArithmeticSelectorTerm> lowerBounds;
    uint64_t residue = 0;
};
struct ArithmeticSelectorPiece {
    // Variables are input endpoint quotients, then all parameter quotients.
    DifferenceBoundSystem domain;
    std::vector<uint64_t> inputResidues, parameterResidues;
    std::size_t outputSite = 0;
    std::vector<ArithmeticSelectorOutput> outputs;
};
struct ArithmeticEndpointSelector {
    std::size_t inputSite = 0;
    uint32_t outputPipe = 0;
    unsigned inputDimensions = 0, parameterCount = 0;
    // Ordered first match: overlapping pieces describe the same tagged output
    // under the certified F* functionality contract. Do not emit each match.
    std::vector<ArithmeticSelectorPiece> pieces;
};
struct ArithmeticSelectors {
    std::string error;
    uint64_t period = 1;
    unsigned parameterCount = 0;
    std::vector<ArithmeticEndpointSelector> forward, inverse;
};
// Consumes a certified exact minimum-demand relation under the core-native
// model. That contract establishes one output per input and destination pipe
// (and dually for inverse selectors); this routine does not re-prove it by
// comparing all pairs of pieces. Schemas and finite coordinate bounds are
// checked. Standalone supplied relations must carry the same certification.
// Original coordinates, including signed parameters, are period*q + residue.
// Closed DBM projection yields each input guard; singleton output intervals
// yield maxima of input-plus-constant lower bounds, including the zero column.
// Initial construction uses O(K*D^3) arbitrary-precision operations for K
// pieces of at most D variables. Exact coalescing tries O(K^2) pairs; each
// hull check uses at most O(D^4) cubic closures, for O(K^2*D^7) total
// operations, plus group lookup. Output remains O(K*D^2). Integer bit costs
// are additional. The greedy sweep need not find the fewest selector pieces.
// No occurrence, parameter-valuation or numeric-distance enumeration; no IR mutation.
ArithmeticSelectors buildArithmeticSelectors(const ArithmeticDemandAnalysis& analysis,
                                             llvm::ArrayRef<uint32_t> sitePipes);
struct ArithmeticSelectedEndpoint {
    std::size_t site = 0;
    std::vector<BoundInteger> coordinates; // Original signed coordinates.
};
// Exact pure evaluation for queries/tests. An empty optional means undefined;
// failure means malformed selector/input schema. Inputs use original signed
// coordinates, not quotients. No fixed-width conversion or truncation occurs.
FailureOr<std::optional<ArithmeticSelectedEndpoint>> evaluateArithmeticSelector(
    const ArithmeticEndpointSelector& selector, uint64_t period,
    llvm::ArrayRef<BoundInteger> coordinates, llvm::ArrayRef<BoundInteger> parameters);
} // namespace mlir::pto::frontiersynch
#endif
