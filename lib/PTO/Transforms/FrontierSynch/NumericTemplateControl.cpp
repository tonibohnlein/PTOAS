// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact per-visit scalar/guard specialization using the shared recurrence proof.
#include "NumericTemplateInternal.h"
#include "NormalizedControl.h"
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
bool TemplateBuilder::normalized(const NormalizedControlDescription& description)
{
    const bool contextMatches = description.index == &index && description.input == &input &&
        description.result.state == RecognitionState::Applicable;
    if (!contextMatches) {
        output.result.note(RecognitionIssue::TemplateContext, output.outer, true); return false;
    }
    auto root = llvm::find_if(description.nodes, [&](const auto& node) {
        return node.original == output.outer.getOperation() && node.fixedCoordinates.empty();
    });
    if (root == description.nodes.end()) {
        output.result.note(RecognitionIssue::TemplateContext, output.outer, true); return false;
    }
    auto bind = [&](ArrayRef<FixedLoopCoordinate> fixed) {
        coordinates.clear(); path.clear();
        for (auto coordinate : fixed) {
            coordinates[coordinate.loop.getInductionVar()] = coordinate.induction;
            path.push_back({coordinate.loop, coordinate.induction});
        }
    };
    std::function<bool(std::size_t, SmallVector<FixedLoopCoordinate>, unsigned)> visit;
    visit = [&](std::size_t id, SmallVector<FixedLoopCoordinate> fixed, unsigned depth) {
        const auto& node = description.nodes[id];
        for (auto coordinate : node.fixedCoordinates) {
            auto found = llvm::find_if(fixed, [&](auto old) { return old.loop == coordinate.loop; });
            if (found == fixed.end()) { fixed.push_back(coordinate); }
            else if (found->induction != coordinate.induction) {
                output.result.note(RecognitionIssue::TemplateContext, node.original, true); return false;
            }
        }
        bind(fixed);
        if (depth > output.limits.depth) {
            output.result.note(RecognitionIssue::TemplateExpansionLimit, node.original, true); return false;
        }
        if (node.original && !charge(1, visits, output.limits.visits, node.original)) { return false; }
        if (node.kind == NormalizedControlKind::Conditional) {
            auto branch = cast<scf::IfOp>(node.original);
            auto selected = guard(branch.getCondition());
            if (!selected) {
                output.result.note(RecognitionIssue::UnsupportedControl, branch, true); return false;
            }
            for (auto [arm, child] : llvm::enumerate(node.children)) {
                const bool inactive = (arm == 0) != *selected;
                if (inactive) { continue; }
                if (!visit(child, fixed, depth + 1)) { return false; }
            }
            return true;
        }
        if (node.kind == NormalizedControlKind::ExpandedLoop ||
            node.kind == NormalizedControlKind::RetainedLoop) {
            auto loop = cast<scf::ForOp>(node.original);
            auto lower = integer(loop.getLowerBound()), upper = integer(loop.getUpperBound());
            auto step = integer(loop.getStep());
            const bool valid = lower && upper && *lower >= 0 && step && *step > 0;
            if (!valid) { output.result.note(RecognitionIssue::LoopDomain, loop, true); return false; }
            auto body = [&](std::size_t child, SmallVector<FixedLoopCoordinate> inherited) {
                // Expanded descendants carry their own binding; retained visits
                // inherit one from this producer's occurrence construction.
                for (auto coordinate : description.nodes[child].fixedCoordinates) {
                    if (llvm::none_of(inherited, [&](auto old) { return old.loop == coordinate.loop; })) {
                        inherited.push_back(coordinate);
                    }
                }
                bind(inherited);
                for (auto state : loop.getRegionIterArgs()) {
                    const bool representable = !index.isRelevant(state) || scalar(state);
                    if (!representable) {
                        output.result.note(RecognitionIssue::LoopCarriedState, loop); return false;
                    }
                }
                return visit(child, std::move(inherited), depth + 1);
            };
            if (node.kind == NormalizedControlKind::ExpandedLoop) {
                for (auto child : node.children) { if (!body(child, fixed)) { return false; } }
                return true;
            }
            const __int128 distance = static_cast<__int128>(*upper) - *lower;
            const __int128 count = distance <= 0 ? 0 : (distance + *step - 1) / *step;
            if (count > output.limits.visits - visits) {
                output.result.note(RecognitionIssue::TemplateExpansionLimit, loop, true); return false;
            }
            for (__int128 iteration = 0; iteration < count; ++iteration) {
                auto inherited = fixed;
                const auto induction = static_cast<__int128>(*lower) + iteration * *step;
                inherited.push_back({loop, static_cast<int64_t>(induction)});
                for (auto child : node.children) { if (!body(child, inherited)) { return false; } }
            }
            return true;
        }
        if (node.kind == NormalizedControlKind::Sequence) {
            for (auto child : node.children) { if (!visit(child, fixed, depth)) { return false; } }
            return true;
        }
        auto phaseList = index.phasesFor(node.original);
        if (!charge(phaseList.size(), phases, output.limits.payloads, node.original)) { return false; }
        if (plannedPayloads) {
            for (auto* phase : phaseList) {
                TemplatePayload planned; planned.phase = phase; planned.coordinates = path;
                plannedPayloads->push_back(std::move(planned));
            }
        }
        return true;
    };
    for (auto child : root->children) { if (!visit(child, {}, 0)) { return false; } }
    return true;
}
} // namespace mlir::pto::frontiersynch::detail
