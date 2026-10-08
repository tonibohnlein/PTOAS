// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact distance-interval generators, distinct from availability of periodic
// machine circuits or executable original-cut endpoint recipes.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICPERIODICCONVERSION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICPERIODICCONVERSION_H
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicQuotient.h"
namespace mlir::pto::frontiersynch {
enum class ArithmeticPeriodicStatus { Applicable, NotDistanceIntervals, AdapterUnavailable, InvalidInput };
struct ArithmeticDistanceBound {
    IntegerAffine numerator;
    BoundInteger denominator = BoundInteger(1);
    bool ceiling = false;
};
struct ArithmeticDistanceInterval {
    uint32_t source = 0, target = 0;
    IntegerSystem guard; // Parameter quotient coordinates only.
    std::vector<uint64_t> parameterResidues;
    std::vector<ArithmeticDistanceBound> lower, upper;
    bool native = false;
    uint64_t originalPiece = 0;
};
struct ArithmeticIntervalPiece {
    uint32_t source = 0, target = 0;
    IntegerSystem relation; // source i, target j, shared parameter quotients.
    std::vector<uint64_t> parameterResidues;
    // Supplied common prefix endpoint domains: coordinate i (or j) followed
    // by parameters. Every piece is one alternative of that endpoint domain.
    // Only atoms implied by every alternative may be removed. These domains
    // must describe the same repeating skeleton/prefix, never private cutoffs.
    std::vector<IntegerSystem> sourceDomains, targetDomains;
    bool native = false;
    uint64_t originalPiece = 0;
};
struct ArithmeticPeriodicInput {
    std::shared_ptr<RegionExpressions> expressions;
    std::vector<GuardedPeriodicPayload> payloads;
    std::vector<RegionExpressions::Id> parameters;
    uint64_t parameterPeriod = 1;
    std::vector<ArithmeticIntervalPiece> pieces;
    // TRUSTED generator contract: exact complete storage and supplied C->I
    // prerequisite relations, factored on fixed site/residue tags and restricted
    // only by these common prefix domains and strict reference order. Other
    // native order is the core start/completion pipe order. Parameter inputs are
    // immutable for this invocation. This converter does not establish effects.
};
struct ArithmeticPeriodicConversionCost {
    uint64_t pieces = 0, constraints = 0, domainComparisons = 0, expressionNodes = 0;
};
struct ArithmeticPeriodicConversion {
    ArithmeticPeriodicStatus status = ArithmeticPeriodicStatus::InvalidInput;
    std::string diagnostic, exportError;
    std::shared_ptr<RegionExpressions> expressions;
    std::vector<ArithmeticDistanceInterval> intervals;
    std::vector<GuardedPeriodicPayload> payloads;
    std::vector<GuardedPeriodicRecord> generators, nativePrerequisites;
    std::vector<uint64_t> generatorPieces, nativePieces;
    std::optional<GuardedPeriodicQuotient> guarded;
    std::optional<PeriodicAnalysis> numerical;
    ArithmeticPeriodicConversionCost cost;
};
// After factored row descriptions, normalization is linear in row size plus
// charged domain-atom comparisons. Each interval produces at most one record.
// Parameter-only divisibility is retained; occurrence congruences or absolute
// cutoffs outside common domains decline conversion. Floor/ceil bounds remain
// mathematical even if checked machine/quotient lowering is unavailable.
ArithmeticPeriodicConversion convertArithmeticPeriodicIntervals(const ArithmeticPeriodicInput& input);
struct ArithmeticPeriodicSite {
    uint32_t originalSite = 0;
    uint64_t residue = 0;
};
struct ArithmeticPeriodicProgram {
    ArithmeticPeriodicConversion conversion;
    scf::ForOp loop;
    uint64_t period = 1;
    std::vector<ArithmeticPeriodicSite> sites; // Periodic type -> original cut and IV residue.
    std::vector<const CompoundInstanceElement*> phases;
};
// Original producer adapter; does not complete arithmetic closure/subtraction.
// The stage must belong to this unchanged program. Current prefix binding uses
// one root-relative loop, zero lower bound and unit step; other bindings retain
// the arithmetic stage and explicitly report an adapter gap. Site/residue
// grouping is charged; no fixed period cap, trip or distance expansion.
ArithmeticPeriodicProgram convertArithmeticPeriodicProgram(const ArithmeticProgram& program,
    const GeneralArithmeticGeneratorStage& stage, std::shared_ptr<RegionExpressions> expressions = {});
} // namespace mlir::pto::frontiersynch
#endif
