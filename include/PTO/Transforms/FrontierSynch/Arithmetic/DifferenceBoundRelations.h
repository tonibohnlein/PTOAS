// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact integer difference-bound conjunctions and finite-union subtraction.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_DIFFERENCEBOUNDRELATIONS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_DIFFERENCEBOUNDRELATIONS_H
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DynamicAPInt.h"
#include <optional>
#include <vector>
namespace mlir::pto::frontiersynch {
using BoundInteger = llvm::DynamicAPInt;
struct DifferenceBoundConstraint {
    // Augmented coordinates: 0 is the distinguished constant zero; numeric
    // variable i has index i+1. The atom is x_lhs - x_rhs <= bound.
    unsigned lhs = 0, rhs = 0;
    BoundInteger bound;
};
class DifferenceBoundSystem {
public:
    // Default construction denotes the true, zero-dimensional conjunction.
    DifferenceBoundSystem() = default;
    static FailureOr<DifferenceBoundSystem> create(
        unsigned dimensions, llvm::ArrayRef<DifferenceBoundConstraint> constraints);
    unsigned dimensions() const { return dimensionCount; }
    bool isEmpty() const { return empty; }
    // Finite closed bound, or nullopt for infinity, invalid indices or an empty
    // system. Callers must distinguish the latter through isEmpty().
    std::optional<BoundInteger> bound(unsigned lhs, unsigned rhs) const;
    // Canonical finite off-diagonal atoms; empty is encoded by 0-0 <= -1.
    std::vector<DifferenceBoundConstraint> constraints() const;
    FailureOr<DifferenceBoundSystem> intersect(const DifferenceBoundSystem& other) const;
    // Maps ZERO-BASED numeric variables. Duplicates identify coordinates;
    // new coordinates outside the image are unconstrained. Zero remains zero.
    FailureOr<DifferenceBoundSystem> remap(unsigned newDimensions, llvm::ArrayRef<unsigned> oldToNew) const;
    // Exact existential integer projection of the CLOSED matrix. Keep indices
    // are unique ZERO-BASED numeric variables, ordered as in the output schema.
    FailureOr<DifferenceBoundSystem> project(llvm::ArrayRef<unsigned> keep) const;
    // Exact closed-DBM inclusion. Different dimension counts are incompatible.
    bool isSubsetOf(const DifferenceBoundSystem& other) const;
    bool operator==(const DifferenceBoundSystem& other) const;
private:
    unsigned dimensionCount = 0;
    bool empty = false;
    std::vector<std::optional<BoundInteger>> bounds{BoundInteger(0)};
    std::size_t position(unsigned lhs, unsigned rhs) const;
    void tighten(unsigned lhs, unsigned rhs, const BoundInteger& value);
    void close();
};
// Computes (union lhs) minus (union rhs) with one threshold arrangement over
// all occurring ordered differences. No convex approximation or successive
// exponentially branching piece subtraction. Schemas, tags and residue tuples
// must already agree; their validation/partitioning belongs to the caller.
// With n numeric coordinates fixed, M thresholds produce at most
// (M+1)^(n*(n+1)) arrangement cells. Constants use arbitrary-precision integers.
// Failure means malformed dimensions or coordinates, never "probably empty".
FailureOr<std::vector<DifferenceBoundSystem>> subtractDifferenceBoundUnions(
    unsigned dimensions, llvm::ArrayRef<DifferenceBoundSystem> lhs,
    llvm::ArrayRef<DifferenceBoundSystem> rhs);
} // namespace mlir::pto::frontiersynch
#endif
