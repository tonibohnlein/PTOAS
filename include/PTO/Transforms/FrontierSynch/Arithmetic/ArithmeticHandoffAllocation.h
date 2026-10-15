// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact reuse and lifetime selectors for retained arithmetic handoff families.
#ifndef PTO_FRONTIERSYNCH_ARITHMETICHANDOFFALLOCATION_H
#define PTO_FRONTIERSYNCH_ARITHMETICHANDOFFALLOCATION_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/GeneralArithmeticSelectors.h"
#include <optional>
namespace mlir::pto::frontiersynch {
class AllocationProfile;
struct ArithmeticHandoffFamily {
    std::size_t sourceSite = 0, targetSite = 0;
    unsigned width = 0;
    // Parameter-only selectors of the retained handoff relation, not of all
    // payload/storage accesses. Coordinates are original signed loop IVs.
    GeneralArithmeticEndpointSelector firstSource, lastTarget;
};
struct ArithmeticHandoffAllocation {
    uint64_t period = 1;
    unsigned parameterCount = 0;
    std::vector<ArithmeticHandoffFamily> families;
    ArithmeticAnalysisCost cost;
    std::string error;
};
// Separate from executed-rank allocation: the selected ID is the original
// source coordinate modulo width. A missing coordinate means constant ID zero.
// Lifetime selectors bound the whole family, not each individual cyclic lane.
struct ArithmeticCoordinateHandoffFamily {
    ArithmeticHandoffFamily lifetime;
    std::optional<unsigned> sourceCoordinate;
};
struct ArithmeticCoordinateHandoffAllocation {
    std::vector<ArithmeticCoordinateHandoffFamily> families;
    ArithmeticAnalysisCost cost;
    std::string error;
};
// Sufficient, bounded candidate for paired direct communications. Width one
// uses the existing all-later reuse proof. Wider candidates require one
// nonnegative original source IV and prove consumption before EVERY later
// publication whose original-IV distance is at least width. This implies
// legal modulo reuse without constructing executed-rank successors. It is
// neither an exact capacity test nor a certificate for surplus publications.
// The optional profile is borrowed only for this call; no recipe retains it.
ArithmeticCoordinateHandoffAllocation buildArithmeticCoordinateHandoffAllocation(
    const GeneralArithmeticDemandAnalysis& analysis, llvm::ArrayRef<uint32_t> pipes,
    unsigned maxWidth = 6, AllocationProfile* profile = nullptr);
// Requires certified F* functionality and finite invocation domains. For each
// family construct strict order on ACTIVE source tuples, remove its square to
// obtain immediate successor S, and test F* inverse composed with S^E against
// required completion-to-start order, for E=1..maxWidth (at most six).
// Parameters/residues are shared in every join; emptiness is exact. A width E
// assigns executed handoff rank modulo E, never original source ordinal modulo
// E. SET and WAIT counters must advance only when that family executes.
// Work is charged to exact projection/difference output and a fixed number
// of relational compositions (successor powers have exponent at most six).
// Fixed class/coordinate bounds bound elimination depth; no
// occurrence or parameter enumeration and no minimum hardware-pool claim.
ArithmeticHandoffAllocation buildArithmeticHandoffAllocation(
    const GeneralArithmeticDemandAnalysis& analysis, llvm::ArrayRef<uint32_t> pipes,
    unsigned maxWidth = 6);
// Exact DBM-to-integer conversion of the already completed F* and H. No
// recognition or transitive closure is rerun; the same reuse proof is applied.
ArithmeticHandoffAllocation buildArithmeticHandoffAllocation(
    const ArithmeticDemandAnalysis& analysis, llvm::ArrayRef<uint32_t> pipes,
    unsigned maxWidth = 6);
} // namespace mlir::pto::frontiersynch
#endif
