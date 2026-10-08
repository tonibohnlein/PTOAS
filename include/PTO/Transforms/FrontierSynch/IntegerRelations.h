// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact affine integer conjunctions, residue-preserving projection and witnesses.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_INTEGERRELATIONS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_INTEGERRELATIONS_H
#include "PTO/Transforms/FrontierSynch/DifferenceBoundRelations.h"
namespace mlir::pto::frontiersynch {
struct IntegerConstraint {
    std::vector<BoundInteger> coefficients;
    BoundInteger bound; // sum(coefficients[i]*x[i]) <= bound.
};
struct IntegerCongruence {
    std::vector<BoundInteger> coefficients;
    BoundInteger residue;
    BoundInteger modulus; // Strictly positive; sum(coefficients[i]*x[i]) == residue modulo modulus.
};
struct IntegerAffine {
    std::vector<BoundInteger> coefficients;
    BoundInteger constant;
};
struct IntegerEliminationWitness;
class IntegerSystem {
public:
    // The default is the true conjunction in zero coordinates. All coordinate
    // indices are zero-based; constants have their own field, never a column.
    IntegerSystem() = default;
    static FailureOr<IntegerSystem> create(unsigned dimensions,
        llvm::ArrayRef<IntegerConstraint> constraints, llvm::ArrayRef<IntegerCongruence> congruences = {});
    unsigned dimensions() const { return dimensionCount; }
    const std::vector<IntegerConstraint>& constraints() const { return inequalities; }
    const std::vector<IntegerCongruence>& congruences() const { return divisibilities; }
    // Representation/cache observation only; never invokes an emptiness solver.
    bool isKnownEmpty() const { return contradiction || (emptyCache && *emptyCache); }
    bool isEmpty() const;
    FailureOr<IntegerSystem> intersect(const IntegerSystem& other) const;
    // Duplicates identify old coordinates; fresh coordinates are unconstrained.
    FailureOr<IntegerSystem> remap(unsigned newDimensions, llvm::ArrayRef<unsigned> oldToNew) const;
    // Exact existential projection, preserving keep order. General projection
    // returns a union: divisibility conditions are retained, never rounded away.
    FailureOr<std::vector<IntegerSystem>> project(llvm::ArrayRef<unsigned> keep) const;
    // Columns in each domain/numerator are all original columns except the
    // eliminated one, in original order. Domains cover the exact projection;
    // numerator/denominator is an integral satisfying witness on its domain.
    FailureOr<std::vector<IntegerEliminationWitness>> eliminateWithWitness(unsigned coordinate) const;
    bool isSubsetOf(const IntegerSystem& other) const;
    bool operator==(const IntegerSystem& other) const;
private:
    unsigned dimensionCount = 0;
    std::vector<IntegerConstraint> inequalities;
    std::vector<IntegerCongruence> divisibilities;
    bool contradiction = false;
    mutable std::optional<bool> emptyCache;
    bool isOctagonal() const;
    FailureOr<std::vector<IntegerSystem>> eliminate(unsigned coordinate) const;
};
struct IntegerEliminationWitness {
    IntegerSystem domain;
    IntegerAffine numerator;
    BoundInteger denominator; // Strictly positive.
};
// Exact union difference through a single arrangement of affine thresholds and
// modular residues. No convex joining or successive Boolean piece splitting.
// The direction/modulus count is class-bounded only when callers preserve the
// fixed primitive coefficient bounds and fixed witness/projection depth.
// Malformed schemas fail; no machine coefficient, period or branch-count cap.
FailureOr<std::vector<IntegerSystem>> subtractIntegerUnions(
    unsigned dimensions, llvm::ArrayRef<IntegerSystem> lhs, llvm::ArrayRef<IntegerSystem> rhs);
} // namespace mlir::pto::frontiersynch
#endif
