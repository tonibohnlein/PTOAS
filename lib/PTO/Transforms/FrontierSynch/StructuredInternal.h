// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_STRUCTURED_INTERNAL_H
#define PTO_FRONTIERSYNCH_STRUCTURED_INTERNAL_H
#include "PTO/Transforms/FrontierSynch/StructuredInputAdapter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <map>
namespace mlir::pto::frontiersynch::structured {
using Int = llvm::DynamicAPInt;
using Relation = presburger::IntegerRelation;
using Set = presburger::PresburgerRelation;
// Temporary exact row construction, not a retained source expression/control IR.
struct Form {
    std::map<unsigned, Int> terms;
    Int constant = Int(0);
    static Form axis(unsigned column);
    static Form number(const Int& value);
    Form scale(const Int& value) const;
    Form plus(const Form& other) const;
    Form minus(const Form& other) const { return plus(other.scale(Int(-1))); }
};
struct Poly {
    explicit Poly(const presburger::PresburgerSpace& space) : relation(space) {}
    Relation relation;
    // Forms containing existential columns belong to this polyhedron. Copies
    // preserve their defining constraints and column numbers; unrelated scopes
    // never share this cache.
    DenseMap<std::pair<Value, Operation*>, Form> forms;
    void constrain(const Form& value, bool equality = false);
    FailureOr<Form> local();
};
struct Builder {
    func::FuncOp function;
    const StructuredImportOptions& options;
    PhaseIndex index;
    SymbolicSchemaHandle schema;
    ExactStructuredEffects effects;
    SmallVector<StructuredSite> sites;
    DenseMap<const CompoundInstanceElement*, std::size_t> siteTags;
    const DenseMap<const CompoundInstanceElement*, std::size_t>* sharedTags = nullptr;
    SmallVector<Operation*> allLoops;
    StructuredImportIssue issue;
    Builder(func::FuncOp function, const StructuredImportOptions& options) : function(function), options(options) {}
    LogicalResult fail(StructuredImportStatus status, Operation* operation, StringRef reason, Value value = {});
    bool qualify(Operation* operation);
    FailureOr<Form> value(Value value, const StructuredSite& site, Poly& poly);
    FailureOr<Form> evaluateValue(Value value, const StructuredSite& site, Poly& poly);
    FailureOr<SmallVector<Form>> bound(Value value, bool upper, const StructuredSite& site, Poly& poly);
    FailureOr<Form> affine(AffineExpr expr, ArrayRef<Form> operands, unsigned dims, Poly& poly, Operation* op);
    FailureOr<SmallVector<Form>> map(
        AffineMap map, ValueRange values, const StructuredSite& site, Poly& poly, Operation* op);
    FailureOr<SmallVector<Poly, 0>> guard(Value value, bool truth, const StructuredSite& site, const Poly& poly);
    FailureOr<SmallVector<Poly, 0>> affineGuard(
        affine::AffineIfOp op, bool truth, const StructuredSite& site, const Poly& poly);
    FailureOr<SmallVector<Poly, 0>> loop(Operation* op, const StructuredSite& site, const Poly& poly);
    LogicalResult discover(Region& region, SmallVector<Operation*> loops, SmallVector<Region*> path);
    FailureOr<SmallVector<Poly, 0>> presence(std::size_t site);
    FailureOr<SymbolicPrimitive> import(SymbolicTuple source, SymbolicTuple target, const Set& relation);
};
Int integer(const APInt& value, bool isUnsigned = false);
void append(SmallVector<Poly, 0>& target, SmallVector<Poly, 0> source);
} // namespace mlir::pto::frontiersynch::structured
#endif
