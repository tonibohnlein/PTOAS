// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Partition one counted loop at a loop-invariant comparison threshold. Each
// interval has an immutable skeleton and uses the ordinary periodic exporter.
// Their original ordinals/cuts survive; sequence composition supplies all
// cross-interval edges. No payloads or runtime iterations are cloned.
#include "SequenceAnalysisInternal.h"
#include "RecognitionInternal.h"
#include "ArithmeticRows.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingRegional.h"
#include "../InsertSync/SyncScalarEvolution.h"
namespace mlir::pto::frontiersynch {
namespace {
struct Split {
    Expr ordinal;
    bool prefixThen;
};
std::optional<Split> splitCondition(Value condition, scf::ForOp loop,
    const PhaseIndex& index, RegionExpressions& dag, Expr trips)
{
    auto compare = condition.getDefiningOp<arith::CmpIOp>();
    if (!compare || !compare.getLhs().getType().isIndex()) { return std::nullopt; }
    Value varying = compare.getLhs(), bound = compare.getRhs();
    auto predicate = compare.getPredicate();
    SmallVector<Operation*> recipe;
    if (!detail::entryExpression(bound, loop, index, recipe)) { return std::nullopt; }
    SmallVector<Value> symbols;
    mlir::pto::detail::ScalarEvolution evolution(loop.getContext(), loop);
    auto expression = evolution.value(varying, [&](Value value) {
        auto found = llvm::find(symbols, value);
        auto position = found - symbols.begin();
        if (found == symbols.end()) { symbols.push_back(value); }
        return getAffineSymbolExpr(position, loop.getContext());
    });
    LinearRow row;
    if (detail::collectRow(expression, 0, symbols.size(), row) || row.constant < 0) { return std::nullopt; }
    bool foundIV = false;
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        if (symbols[i] == loop.getInductionVar() && row.coefficients[i] == 1) { foundIV = true; }
        else if (row.coefficients[i]) { return std::nullopt; }
    }
    if (!foundIV) { return std::nullopt; }
    bool signedCompare = false, inclusive = false, prefixThen = true;
    switch (predicate) {
    case arith::CmpIPredicate::slt: signedCompare = true; break;
    case arith::CmpIPredicate::ult: break;
    case arith::CmpIPredicate::sle: signedCompare = true; inclusive = true; break;
    case arith::CmpIPredicate::ule: inclusive = true; break;
    case arith::CmpIPredicate::sge: signedCompare = true; prefixThen = false; break;
    case arith::CmpIPredicate::uge: prefixThen = false; break;
    case arith::CmpIPredicate::sgt: signedCompare = true; inclusive = true; prefixThen = false; break;
    case arith::CmpIPredicate::ugt: inclusive = true; prefixThen = false; break;
    default: return std::nullopt;
    }
    // The varying expression is proved iv+c without machine wrap. Clip before
    // adding one, so inclusive comparisons remain total even at UINT64_MAX.
    auto zero = dag.constant(0), offset = dag.constant(row.constant), threshold = dag.input(bound);
    auto below = signedCompare ? dag.slt(threshold, offset) : dag.lt(threshold, offset);
    auto delta = dag.sub(threshold, offset);
    auto clipped = dag.select(dag.lt(delta, trips), delta, trips);
    if (inclusive) { clipped = dag.select(dag.lt(clipped, trips), dag.add(clipped, dag.constant(1)), trips); }
    return Split{dag.select(below, zero, clipped), prefixThen};
}
} // namespace
bool SequenceAnalysisState::boundaryLoop(scf::ForOp loop)
{
    if (sequenceInteger(loop.getLowerBound()) != 0 || sequenceInteger(loop.getStep()) != 1 ||
        loop.getNumRegionIterArgs()) { return false; }
    auto upper = expressions.input(loop.getUpperBound());
    auto trips = expressions.select(expressions.slt(c(0), upper), upper, c(0));
    DenseMap<Value, bool> prefix, suffix;
    std::optional<Expr> cut;
    bool supported = true;
    loop.getBody()->walk([&](scf::IfOp branch) {
        SmallVector<Operation*> recipe;
        if (detail::entryExpression(branch.getCondition(), loop, index, recipe)) { return; }
        auto split = splitCondition(branch.getCondition(), loop, index, expressions, trips);
        if (!split || (cut && *cut != split->ordinal)) { supported = false; return; }
        cut = split->ordinal;
        prefix[branch.getCondition()] = split->prefixThen;
        suffix[branch.getCondition()] = !split->prefixThen;
    });
    if (!supported || !cut) { return false; }
    std::vector<Child> slices;
    for (unsigned part = 0; part < 2; ++part) {
        PeriodicSlice interval{part ? *cut : c(0), part ? trips : *cut};
        if (interval.begin == interval.end) { continue; }
        auto recognized = detail::recognizeRotatingSlice(loop, index, *input, part ? suffix : prefix);
        if (recognized.result.state != RecognitionState::Applicable) { return false; }
        auto analyzed = analyzeGuardedRotating(loop, *input, recognized, arena);
        if (!analyzed.error.empty()) { return false; }
        std::string exportError;
        auto regional = guardedRotatingRegionalResult(function, *input, analyzed, exportError, interval);
        if (failed(regional)) { return false; }
        Child child;
        child.loop = loop;
        child.regional = std::move(*regional);
        child.anchors = child.regional.anchors;
        slices.push_back(std::move(child));
    }
    for (auto& slice : slices) { children.push_back(std::move(slice)); }
    return true;
}
} // namespace mlir::pto::frontiersynch
