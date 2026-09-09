// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// See LICENSE for details. Provided AS IS, WITHOUT WARRANTIES OF ANY KIND.
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
    SmallVector<OccurrencePoint, 0> points;
    llvm::DenseMap<Operation*, unsigned> ids;
    unsigned scheduleDimensions = 0;
    std::string reason;
    bool complete = false;
    bool limitExceeded = false;
    uint64_t work = 0;
    static SyncOccurrences build(func::FuncOp function, ArrayRef<Operation*> physicalPoints);
    // Uniform occurrence tuple: phase ID followed by one IV per original loop.
    unsigned dimensions() const { return 1 + loops.size(); }
    // Queries require complete and valid point IDs. Parameter Values are
    // actual SSA bindings; reconstructors must retain them, not reinterpret
    // unconstrained arithmetic correlations as independent caller promises.
    Relation domain(unsigned point) const;
    RelationResult ordered(unsigned source, unsigned target, bool inclusive = false) const;
    Relation identity(unsigned point) const;
    Relation predicateDomain(unsigned predicate, unsigned point) const;
};
} // namespace mlir::pto::logical_sync
#endif
