// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GENERALARITHMETICSELECTORS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GENERALARITHMETICSELECTORS_H
#include "PTO/Transforms/FrontierSynch/ArithmeticSelectors.h"
namespace mlir::pto::frontiersynch {
struct GeneralArithmeticSelectorOutput {
    // Coefficients address input endpoint quotients, then every common
    // parameter quotient. The quotient output equals numerator/denominator,
    // with exact integral division on the containing piece's domain.
    IntegerAffine numerator;
    BoundInteger denominator;
    uint64_t residue = 0;
};
struct GeneralArithmeticSelectorPiece {
    IntegerSystem domain; // Input endpoint quotients followed by parameters.
    std::vector<uint64_t> inputResidues, parameterResidues;
    std::size_t outputSite = 0;
    std::vector<GeneralArithmeticSelectorOutput> outputs;
};
struct GeneralArithmeticEndpointSelector {
    std::size_t inputSite = 0;
    uint32_t outputPipe = 0;
    unsigned inputDimensions = 0, parameterCount = 0;
    // Stable first match returns this piece's entire tagged output tuple.
    std::vector<GeneralArithmeticSelectorPiece> pieces;
};
struct GeneralArithmeticSelectors {
    std::string error;
    uint64_t period = 1;
    unsigned parameterCount = 0;
    std::vector<GeneralArithmeticEndpointSelector> forward, inverse;
};
// Requires the same certified exact-F* per-pipe functionality contract as the
// DBM selector builder. No pairwise functionality proof is repeated here.
// Reorder columns to inputs/parameters/outputs, eliminate outputs backwards,
// then substitute each earlier rational witness into later coordinates. Every
// resulting domain therefore selects one consistent complete tuple, retaining
// exact congruence conditions and all original coordinate residues.
// Work and output are polynomial for the fixed arithmetic class and coordinate
// bound, with exact integer-projection branch costs charged explicitly. Neither
// numeric occurrence domains nor parameter valuations are enumerated. No IR mutation.
GeneralArithmeticSelectors buildGeneralArithmeticSelectors(
    const GeneralArithmeticDemandAnalysis& analysis, llvm::ArrayRef<uint32_t> sitePipes);
FailureOr<std::optional<ArithmeticSelectedEndpoint>> evaluateGeneralArithmeticSelector(
    const GeneralArithmeticEndpointSelector& selector, uint64_t period,
    llvm::ArrayRef<BoundInteger> coordinates, llvm::ArrayRef<BoundInteger> parameters);
} // namespace mlir::pto::frontiersynch
#endif
