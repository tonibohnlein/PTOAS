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
    // Exact subset of source/target point domains where the two normalized
    // scalar values are equal. Each expression uses its OWN occurrence's loop
    // coordinates; actual parameter SSA bindings remain shared. Even identical
    // Value handles do not imply equality across different loop occurrences.
    // Both optional projections and dominance at the corresponding points are
    // required. This supplies neither execution order nor physical-slot alias
    // qualification; clients must establish those independently.
    RelationResult equalScalars(Value sourceValue, unsigned sourcePoint,
                                Value targetValue, unsigned targetPoint,
                                RelationQueries& queries) const;
    // Filter an ALREADY QUALIFIED original occurrence relation by the same
    // equality, without adding the ambient-domain product again. Precondition:
    // originalOccurrences is contained in domain(sourcePoint) x
    // domain(targetPoint), with this exact source/range tuple layout and the
    // same immutable SSA parameter bindings. A matching dimensional space alone
    // does not establish that precondition. NativeOrder requirement relations
    // provide it; arbitrary caller-supplied relations do not.
    // Both APIs refuse unavailable normalization/dominance and explicitly
    // report exhausted work. Neither interprets failure as an empty relation.
    RelationResult filterEqualScalars(const Relation& originalOccurrences,
                                     Value sourceValue, unsigned sourcePoint,
                                     Value targetValue, unsigned targetPoint,
                                     RelationQueries& queries) const;
    // Exact successor of an ENTIRE selected publication population, qualified
    // already qualified within point domains of this immutable occurrence
    // universe (matching tuple dimensions alone do not establish provenance).
    // The compact single-loop adapter
    // proves its candidate domain equal to the supplied domain before using
    // commonPeriodSuccessors. Unsupported is permission to use the general
    // exact successor query, never an empty relation or proof of event reuse.
    RelationResult periodicSuccessors(const presburger::PresburgerSet& publications,
                                     RelationQueries& queries) const;
private:
    RelationResult scalarEqualityConstraint(Value sourceValue, unsigned sourcePoint,
                                            Value targetValue, unsigned targetPoint,
                                            RelationQueries& queries) const;
};
// Optional integer-exact normalization of one occurrence conjunct, preserving
// the selected coordinate and all other named coordinates/parameter bindings.
// Unsupported returns no rewrite; it never weakens the input relation.
RelationResult normalizeOccurrenceCell(const presburger::IntegerRelation& cell,
                                       unsigned preservedCoordinate, RelationQueries& queries);
namespace testing {
// Test only: exact local elimination in one conjunct, before the adapter's
// candidate-envelope and whole-population qualification stages.
RelationResult simplifyPeriodicCell(const presburger::IntegerRelation& cell,
                                   unsigned iterationCoordinate, RelationQueries& queries);
}
} // namespace mlir::pto::logical_sync
#endif
