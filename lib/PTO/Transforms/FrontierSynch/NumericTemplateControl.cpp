// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact per-visit scalar/guard specialization using the shared recurrence proof.
#include "NumericTemplateInternal.h"
#include "RecognitionInternal.h"
#include "../InsertSync/SyncScalarEvolution.h"
namespace mlir::pto::frontiersynch::detail {
namespace {
// Pure integer arithmetic over values defined outside the retained loop has
// the same value on every visit, even when its computation sits in the body.
// Iteration arguments, loads and region-producing operations are not lifted.
bool invariantScalar(Value value, scf::ForOp loop, DenseMap<Value, bool>& cache, unsigned depth = 0)
{
    if (!value || !value.getType().isIntOrIndex() || depth >= 64) {
        return false;
    }
    if (loop.isDefinedOutsideOfLoop(value)) {
        return true;
    }
    auto found = cache.find(value);
    if (found != cache.end()) {
        return found->second;
    }
    auto* op = value.getDefiningOp();
    const bool invariant = op && !op->getNumRegions() &&
        op->getName().getDialectNamespace() == "arith" && isMemoryEffectFree(op) &&
        llvm::all_of(op->getOperands(), [&](Value operand) {
            return invariantScalar(operand, loop, cache, depth + 1);
        });
    cache.try_emplace(value, invariant);
    return invariant;
}
} // namespace
AffineExpr TemplateBuilder::scalar(Value value, SmallVectorImpl<Value>* invariants, bool control) const
{
    // ScalarEvolution caches by Value: a fresh instance is required for each
    // coordinate environment, rather than carrying the first visit's constants.
    mlir::pto::detail::ScalarEvolution evolution(context(), output.outer);
    DenseMap<Value, bool> invariantCache;
    auto expression = evolution.value(value, [&](Value symbol) -> AffineExpr {
        // Concrete inner occurrences always take precedence over parameter
        // specialization. Control never sees the representative outer phase.
        if (auto found = coordinates.find(symbol); found != coordinates.end()) {
            return getAffineConstantExpr(found->second, context());
        }
        if (!control && geometryConstant) {
            if (auto bound = geometryConstant(symbol)) { return getAffineConstantExpr(*bound, context()); }
        }
        if (symbol == output.outer.getInductionVar()) {
            return mlir::pto::detail::checkedAdd(getAffineConstantExpr(output.lower, context()),
                mlir::pto::detail::checkedMul(getAffineSymbolExpr(0, context()),
                                             getAffineConstantExpr(output.step, context())));
        }
        if (!invariants || !invariantScalar(symbol, output.outer, invariantCache)) {
            return {};
        }
        auto position = llvm::find(*invariants, symbol);
        if (position == invariants->end()) {
            invariants->push_back(symbol);
            position = std::prev(invariants->end());
        }
        return getAffineSymbolExpr(1 + (position - invariants->begin()), context());
    });
    return expression ? simplifyAffineExpr(expression, 0, 1 + (invariants ? invariants->size() : 0)) : AffineExpr{};
}
std::optional<int64_t> TemplateBuilder::integer(Value value) const
{
    auto constant = dyn_cast_or_null<AffineConstantExpr>(scalar(value, nullptr, true));
    return constant ? std::optional<int64_t>(constant.getValue()) : std::nullopt;
}
namespace {
std::optional<bool> evaluateGuard(const TemplateBuilder& builder, Value value, unsigned depth,
                                  DenseMap<Value, std::optional<bool>>& cache);
std::optional<bool> buildGuard(const TemplateBuilder& builder, Value value, unsigned depth,
                               DenseMap<Value, std::optional<bool>>& cache)
{
    if (!value || depth > builder.output.limits.depth) {
        return std::nullopt;
    }
    if (builder.controlConstant) {
        if (auto bound = builder.controlConstant(value)) { return bound; }
    }
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant)) && value.getType().isInteger(1)) {
        return !constant.isZero();
    }
    if (auto cmp = value.getDefiningOp<arith::CmpIOp>()) {
        auto a = builder.integer(cmp.getLhs()), b = builder.integer(cmp.getRhs());
        if (!a || !b) {
            return std::nullopt;
        }
        switch (cmp.getPredicate()) {
            case arith::CmpIPredicate::eq: return *a == *b;
            case arith::CmpIPredicate::ne: return *a != *b;
            case arith::CmpIPredicate::slt: return *a < *b;
            case arith::CmpIPredicate::sle: return *a <= *b;
            case arith::CmpIPredicate::sgt: return *a > *b;
            case arith::CmpIPredicate::sge: return *a >= *b;
            default: return std::nullopt;
        }
    }
    auto* op = value.getDefiningOp();
    if (!op || !isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
        return std::nullopt;
    }
    auto a = evaluateGuard(builder, op->getOperand(0), depth + 1, cache);
    if (a && ((isa<arith::AndIOp>(op) && !*a) || (isa<arith::OrIOp>(op) && *a))) { return a; }
    auto b = evaluateGuard(builder, op->getOperand(1), depth + 1, cache);
    if (b && ((isa<arith::AndIOp>(op) && !*b) || (isa<arith::OrIOp>(op) && *b))) { return b; }
    if (!a || !b) { return std::nullopt; }
    return isa<arith::AndIOp>(op) ? *a && *b : (isa<arith::OrIOp>(op) ? *a || *b : *a != *b);
}
std::optional<bool> evaluateGuard(const TemplateBuilder& builder, Value value, unsigned depth,
                                  DenseMap<Value, std::optional<bool>>& cache)
{
    auto found = cache.find(value);
    if (found != cache.end()) {
        return found->second;
    }
    auto result = buildGuard(builder, value, depth, cache);
    cache.try_emplace(value, result);
    return result;
}
} // namespace
std::optional<bool> TemplateBuilder::guard(Value value, unsigned depth) const
{
    DenseMap<Value, std::optional<bool>> cache;
    return evaluateGuard(*this, value, depth, cache);
}
bool TemplateBuilder::charge(uint64_t count, uint64_t& total, uint64_t limit, Operation* anchor)
{
    if (total > limit || count > limit - total) {
        output.result.note(RecognitionIssue::TemplateExpansionLimit, anchor, true);
        return false;
    }
    total += count;
    return true;
}
bool TemplateBuilder::loop(scf::ForOp nested, bool emit, unsigned depth)
{
    auto lower = integer(nested.getLowerBound()), upper = integer(nested.getUpperBound());
    auto step = integer(nested.getStep());
    const bool valid = lower && upper && step && *step > 0 && *lower >= 0;
    if (!valid) {
        output.result.note(RecognitionIssue::LoopDomain, nested, true);
        return false;
    }
    const auto distance = *upper > *lower ? static_cast<uint64_t>(*upper - *lower) : uint64_t{0};
    const auto count = distance / *step + static_cast<uint64_t>(distance % *step != 0);
    if (count > output.limits.visits - visits || depth > output.limits.depth) {
        output.result.note(RecognitionIssue::TemplateExpansionLimit, nested, true);
        return false;
    }
    int64_t induction = *lower;
    for (uint64_t visit = 0; visit < count; ++visit) {
        coordinates[nested.getInductionVar()] = induction;
        path.push_back({nested, induction});
        for (auto state : nested.getRegionIterArgs()) {
            if (index.isRelevant(state) && !scalar(state)) {
                output.result.note(RecognitionIssue::LoopCarriedState, nested);
                return false;
            }
        }
        if (!block(*nested.getBody(), emit, depth)) {
            return false;
        }
        path.pop_back();
        if (visit + 1 < count && llvm::AddOverflow(induction, *step, induction)) {
            output.result.note(RecognitionIssue::LoopDomain, nested, true);
            return false;
        }
    }
    coordinates.erase(nested.getInductionVar());
    return true;
}
bool TemplateBuilder::block(Block& body, bool emit, unsigned depth)
{
    if (depth > output.limits.depth) {
        output.result.note(RecognitionIssue::TemplateExpansionLimit, body.getParentOp(), true);
        return false;
    }
    for (Operation& op : body) {
        if (!charge(1, visits, output.limits.visits, &op)) {
            return false;
        }
        if (auto nested = dyn_cast<scf::ForOp>(op)) {
            if (!loop(nested, emit, depth + 1)) {
                return false;
            }
        } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
            auto selected = guard(branch.getCondition());
            if (!selected) {
                output.result.note(RecognitionIssue::UnsupportedControl, branch, true);
                return false;
            }
            for (auto [arm, region] : llvm::enumerate(branch->getRegions())) {
                const bool chosen = (arm == 0) == *selected;
                const bool savedPath = activePath;
                activePath = savedPath && chosen;
                const bool walked = (emit && !chosen) || region.empty() || block(region.front(), emit, depth + 1);
                activePath = savedPath;
                if (!walked) { return false; }
            }
        } else {
            auto phaseList = index.phasesFor(&op);
            if (!charge(phaseList.size(), phases, output.limits.payloads, &op)) {
                return false;
            }
            for (const auto* phase : phaseList) {
                if (emit && !payload(phase)) { return false; }
                if (!emit && activePath && plannedPayloads) {
                    TemplatePayload planned;
                    planned.phase = phase; planned.coordinates = path;
                    plannedPayloads->push_back(std::move(planned));
                }
            }
        }
    }
    return true;
}
} // namespace mlir::pto::frontiersynch::detail
