// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCOCCURRENCES_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCOCCURRENCES_H
#include "PTO/Transforms/InsertSync/LogicalSyncRelations.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::pto::logical_sync {
struct OccurrencePoint {
    Operation* operation;
    presburger::PresburgerSet domain; // all loop IVs; non-enclosing IVs are zero
    SmallVector<AffineExpr> schedule;
};
// Compact immutable physical/control projection. No completion/selected tokens
// are required to import it. Values identify actual runtime parameters; they
// are never caller promises or independent copies of a repeated expression.
struct SyncOccurrences {
    struct LoopDomain { AffineExpr lower, upper; int64_t step; };
    struct PredicateDomain {
        Value value;
        presburger::PresburgerSet whenTrue;
    };
    SmallVector<scf::ForOp> loops;
    SmallVector<LoopDomain> loopDomains;
    SmallVector<Value> parameters;
    SmallVector<PredicateDomain, 0> predicates;
    // Optional scalar projections use the same original loop and parameter
    // bindings as control import. Absence means unavailable precision, not an
    // impossible value or an empty domain. No integer wrapping is inferred.
    llvm::DenseMap<Value, AffineExpr> scalarExpressions;
    SmallVector<OccurrencePoint, 0> points;
    llvm::DenseMap<Operation*, unsigned> ids;
    unsigned scheduleDimensions = 0;
    std::string reason;
    bool complete = false;
    bool limitExceeded = false;
    uint64_t work = 0;
    static SyncOccurrences build(func::FuncOp function, ArrayRef<Operation*> physicalPoints,
                                 ArrayRef<Value> optionalScalarValues = {});
    // Uniform occurrence tuple: phase ID followed by one IV per original loop.
    unsigned dimensions() const { return 1 + loops.size(); }
    // Queries require complete and valid point IDs. Parameter Values are
    // actual SSA bindings; reconstructors must retain them, not reinterpret
    // unconstrained arithmetic correlations as independent caller promises.
    Relation domain(unsigned point) const;
    RelationResult ordered(unsigned source, unsigned target, bool inclusive = false) const;
    Relation identity(unsigned point) const;
    Relation predicateDomain(unsigned predicate, unsigned point) const;
    // Exact subset where inclusiveLower <= value <= inclusiveUpper. Requires
    // a normalized requested value that dominates the actual queried point.
    // In particular, a loop-local value is not interpreted at an outer point
    // using its zero-filled IV coordinate. i1 uses the existing 0/1 convention.
    // A client must prove full access-domain coverage by [0, count-1] before
    // using ordinary slot-index equality; multi_tile_get does not imply mod N.
    RelationResult scalarDomain(Value value, unsigned point, int64_t inclusiveLower,
                                int64_t inclusiveUpper, RelationQueries& queries) const;
};
} // namespace mlir::pto::logical_sync
#endif
