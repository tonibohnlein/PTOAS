// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact guarded arithmetic domains. Boolean alternatives remain explicit unions;
// conjunction uses their Cartesian product, so costs include the produced pieces.
#include "ArithmeticProgramInternal.h"
#include "../InsertSync/SyncRegionArithmetic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch::detail {
namespace {
using Rows = SmallVector<AffineExpr>;
using Pieces = SmallVector<Rows>;
using GuardCache = DenseMap<std::pair<Value, unsigned>, Pieces>;
constexpr unsigned maxGuardDepth = 64;
// This producer only accepts a bounded explicit guard representation. Exceeding
// it rejects the route; no alternative is dropped or approximated.
constexpr std::size_t maxGuardPieces = 4096;
constexpr std::size_t maxGuardRows = 4096;
std::optional<bool> booleanConstant(Value value)
{
    APInt constant;
    const bool known = value.getType().isInteger(1) && matchPattern(value, m_ConstantInt(&constant));
    if (known) {
        return !constant.isZero();
    }
    return std::nullopt;
}
bool supportedComparison(arith::CmpIOp compare)
{
    if (!compare.getLhs().getType().isIndex()) {
        return false;
    }
    switch (compare.getPredicate()) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne:
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::sge: return true;
    default: return false;
    }
}
std::optional<Pieces> combine(Pieces left, Pieces right, bool conjunction, bool* exceeded = nullptr)
{
    if (!conjunction) {
        for (const auto& piece : right) {
            if (!llvm::is_contained(left, piece)) {
                const bool full = left.size() == maxGuardPieces;
                if (full) {
                    if (exceeded) { *exceeded = true; }
                    return std::nullopt;
                }
                left.push_back(piece);
            }
        }
        return left;
    }
    const bool oversized = !right.empty() && left.size() > maxGuardPieces / right.size();
    if (oversized) {
        if (exceeded) { *exceeded = true; }
        return std::nullopt;
    }
    Pieces result;
    for (const auto& a : left) {
        for (const auto& b : right) {
            Rows rows(a);
            for (auto row : b) {
                if (!llvm::is_contained(rows, row)) {
                    const bool full = rows.size() == maxGuardRows;
                    if (full) {
                        if (exceeded) { *exceeded = true; }
                        return std::nullopt;
                    }
                    rows.push_back(row);
                }
            }
            if (!llvm::is_contained(result, rows)) {
                result.push_back(std::move(rows));
            }
        }
    }
    return result;
}
std::optional<Pieces> comparison(ProgramBuilder& builder, arith::CmpIOp compare, bool truth,
                                  const ArithmeticSite& site, unsigned offset)
{
    auto lhs = builder.value(compare.getLhs(), site, offset);
    auto rhs = builder.value(compare.getRhs(), site, offset);
    auto negative = [&](AffineExpr expression) {
        return mlir::pto::detail::checkedMul(expression, getAffineConstantExpr(-1, builder.context));
    };
    auto difference = mlir::pto::detail::checkedAdd(lhs, negative(rhs));
    auto reverse = negative(difference);
    auto strict = [&](AffineExpr expression) {
        return mlir::pto::detail::checkedAdd(expression, getAffineConstantExpr(-1, builder.context));
    };
    if (!difference || !reverse) {
        return std::nullopt;
    }
    auto predicate = compare.getPredicate();
    if (predicate == arith::CmpIPredicate::eq || predicate == arith::CmpIPredicate::ne) {
        const bool equal = (predicate == arith::CmpIPredicate::eq) == truth;
        return equal ? Pieces{{difference, reverse}} : Pieces{{strict(difference)}, {strict(reverse)}};
    }
    bool reversed = predicate == arith::CmpIPredicate::slt || predicate == arith::CmpIPredicate::sle;
    bool strictBound = predicate == arith::CmpIPredicate::slt || predicate == arith::CmpIPredicate::sgt;
    if (!truth) {
        reversed = !reversed;
        strictBound = !strictBound;
    }
    auto row = reversed ? reverse : difference;
    return Pieces{{strictBound ? strict(row) : row}};
}
std::optional<Pieces> condition(ProgramBuilder& builder, Value value, bool truth,
                               const ArithmeticSite& site, unsigned offset, unsigned depth,
                               GuardCache& cache, bool& exceeded);
std::optional<Pieces> buildCondition(ProgramBuilder& builder, Value value, bool truth,
                               const ArithmeticSite& site, unsigned offset, unsigned depth,
                               GuardCache& cache, bool& exceeded)
{
    if (depth > maxGuardDepth) {
        exceeded = true;
        return std::nullopt;
    }
    if (auto constant = booleanConstant(value)) {
        return *constant == truth ? Pieces{Rows{}} : Pieces{};
    }
    if (auto fixed = builder.constant(value)) {
        return (*fixed != 0) == truth ? Pieces{Rows{}} : Pieces{};
    }
    auto parameter = builder.parameterIds.find(value);
    const bool booleanParameter = parameter != builder.parameterIds.end() && value.getType().isInteger(1);
    if (booleanParameter) {
        auto symbol = getAffineSymbolExpr(parameter->second, builder.context);
        return Pieces{{truth ? symbol - 1 : -symbol}};
    }
    if (auto compare = value.getDefiningOp<arith::CmpIOp>()) {
        return comparison(builder, compare, truth, site, offset);
    }
    auto* operation = value.getDefiningOp();
    if (auto exclusive = dyn_cast_or_null<arith::XOrIOp>(operation)) {
        auto left = booleanConstant(exclusive.getLhs()), right = booleanConstant(exclusive.getRhs());
        if (left || right) {
            return condition(builder, left ? exclusive.getRhs() : exclusive.getLhs(),
                             truth != (left ? *left : *right), site, offset, depth + 1, cache, exceeded);
        }
        return std::nullopt;
    }
    const bool andOr = operation && isa<arith::AndIOp, arith::OrIOp>(operation);
    if (!andOr) {
        return std::nullopt;
    }
    auto left = condition(builder, operation->getOperand(0), truth, site, offset, depth + 1, cache, exceeded);
    auto right = condition(builder, operation->getOperand(1), truth, site, offset, depth + 1, cache, exceeded);
    if (!left || !right) {
        return std::nullopt;
    }
    const bool conjunction = isa<arith::AndIOp>(operation) == truth;
    return combine(std::move(*left), std::move(*right), conjunction, &exceeded);
}
std::optional<Pieces> condition(ProgramBuilder& builder, Value value, bool truth,
                               const ArithmeticSite& site, unsigned offset, unsigned depth,
                               GuardCache& cache, bool& exceeded)
{
    if (depth > maxGuardDepth) {
        exceeded = true;
        return std::nullopt;
    }
    auto key = std::make_pair(value, static_cast<unsigned>(truth));
    auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }
    auto result = buildCondition(builder, value, truth, site, offset, depth, cache, exceeded);
    if (result) {
        cache[key] = *result;
    }
    return result;
}
// Bound alternatives remain explicit relation pieces. Their Cartesian
// products are charged by the emitted primitive-piece count; a representation
// cap is reported as a producer limit, never as a theorem-class violation.
struct BoundPiece {
    AffineExpr value;
    Rows rows;
};
using BoundPieces = SmallVector<BoundPiece>;
std::optional<BoundPieces> boundPieces(ProgramBuilder& builder, Value input, const ArithmeticSite& site,
                                     unsigned offset, unsigned depth, GuardCache& cache, bool& exceeded)
{
    if (depth > maxGuardDepth) { exceeded = true; return std::nullopt; }
    if (auto affine = builder.value(input, site, offset)) { return BoundPieces{{affine, {}}}; }
    auto* op = input.getDefiningOp();
    const bool minmax = op && isa<arith::MinSIOp, arith::MaxSIOp>(op);
    auto select = dyn_cast_or_null<arith::SelectOp>(op);
    if (!input.getType().isIndex() || (!minmax && !select)) { return std::nullopt; }
    Value left = select ? select.getTrueValue() : op->getOperand(0);
    Value right = select ? select.getFalseValue() : op->getOperand(1);
    auto a = boundPieces(builder, left, site, offset, depth + 1, cache, exceeded);
    auto b = boundPieces(builder, right, site, offset, depth + 1, cache, exceeded);
    if (!a || !b) { return std::nullopt; }
    BoundPieces result;
    auto append = [&](AffineExpr value, std::optional<Pieces> predicates) {
        if (!predicates) { return false; }
        if (predicates->size() > maxGuardPieces - result.size()) { exceeded = true; return false; }
        for (auto& rows : *predicates) { result.push_back({value, std::move(rows)}); }
        return true;
    };
    if (select) {
        for (bool truth : {true, false}) {
            auto predicate = condition(builder, select.getCondition(), truth, site, offset,
                                       depth + 1, cache, exceeded);
            if (!predicate) { return std::nullopt; }
            for (const auto& choice : truth ? *a : *b) {
                if (!append(choice.value, combine(Pieces{choice.rows}, *predicate, true, &exceeded))) {
                    return std::nullopt;
                }
            }
        }
    } else {
        for (const auto& x : *a) {
            for (const auto& y : *b) {
                auto rows = combine(Pieces{x.rows}, Pieces{y.rows}, true, &exceeded);
                if (!rows) { return std::nullopt; }
                auto difference = mlir::pto::detail::checkedAdd(y.value,
                    mlir::pto::detail::checkedMul(x.value, getAffineConstantExpr(-1, builder.context)));
                if (!difference) { return std::nullopt; }
                // Tie belongs to the left choice. No fixed-width subtraction
                // is executed: these are mathematical comparison constraints.
                if (isa<arith::MaxSIOp>(op)) {
                    difference = mlir::pto::detail::checkedMul(difference,
                        getAffineConstantExpr(-1, builder.context));
                }
                auto opposite = mlir::pto::detail::checkedAdd(
                    mlir::pto::detail::checkedMul(difference, getAffineConstantExpr(-1, builder.context)),
                    getAffineConstantExpr(-1, builder.context));
                if (!difference || !opposite ||
                    !append(x.value, combine(*rows, Pieces{{difference}}, true, &exceeded)) ||
                    !append(y.value, combine(*rows, Pieces{{opposite}}, true, &exceeded))) { return std::nullopt; }
            }
        }
    }
    return result;
}
std::optional<Pieces> piecewiseDomain(ProgramBuilder& builder, const ArithmeticSite& site,
                                     unsigned offset, GuardCache& cache, bool& exceeded)
{
    Pieces result{Rows{}};
    for (auto [id, storedLoop] : llvm::enumerate(site.loops)) {
        auto loop = storedLoop;
        if (builder.value(loop.getLowerBound(), site, offset) &&
            builder.value(loop.getUpperBound(), site, offset)) { continue; }
        auto lower = boundPieces(builder, loop.getLowerBound(), site, offset, 0, cache, exceeded);
        auto upper = boundPieces(builder, loop.getUpperBound(), site, offset, 0, cache, exceeded);
        if (!lower || !upper) { return std::nullopt; }
        Pieces alternatives;
        auto step = dyn_cast<AffineConstantExpr>(builder.value(loop.getStep(), site, offset));
        if (!step || step.getValue() <= 0) { return std::nullopt; }
        const auto iv = getAffineDimExpr(offset + id, builder.context);
        const auto minusOne = getAffineConstantExpr(-1, builder.context);
        for (const auto& low : *lower) {
            for (const auto& high : *upper) {
                auto distance = mlir::pto::detail::checkedAdd(iv,
                    mlir::pto::detail::checkedMul(low.value, minusOne));
                auto constant = dyn_cast<AffineConstantExpr>(high.value);
                auto bound = constant && constant.getValue() == INT64_MIN ? minusOne :
                    mlir::pto::detail::checkedAdd(mlir::pto::detail::checkedAdd(high.value, -iv), minusOne);
                if (!distance || !bound) { return std::nullopt; }
                Rows rows{distance, bound};
                if (step.getValue() != 1) { rows.push_back(-(distance % step.getValue())); }
                auto selected = combine(Pieces{low.rows}, Pieces{high.rows}, true, &exceeded);
                if (selected) { selected = combine(std::move(*selected), Pieces{rows}, true, &exceeded); }
                auto joined = selected ? combine(std::move(alternatives), std::move(*selected), false, &exceeded) :
                                         std::optional<Pieces>{};
                if (!joined) { return std::nullopt; }
                alternatives = std::move(*joined);
            }
        }
        auto combined = combine(std::move(result), std::move(alternatives), true, &exceeded);
        if (!combined) { return std::nullopt; }
        result = std::move(*combined);
    }
    return result;
}
} // namespace

bool ProgramBuilder::prepareBound(Value input, const ArithmeticSite& site)
{
    SmallVector<std::pair<Value, unsigned>> work{{input, 0}};
    DenseSet<Value> seen;
    while (!work.empty()) {
        auto [value, depth] = work.pop_back_val();
        if (depth > maxGuardDepth) {
            output.extraction.note(RecognitionIssue::TemplateExpansionLimit, value.getDefiningOp());
            return false;
        }
        if (!seen.insert(value).second || prepareValue(value, site)) { continue; }
        auto* op = value.getDefiningOp();
        if (!value.getType().isIndex() || !op) { return false; }
        if (auto select = dyn_cast<arith::SelectOp>(op)) {
            if (!prepareGuard(select.getCondition(), site)) { return false; }
            work.push_back({select.getTrueValue(), depth + 1});
            work.push_back({select.getFalseValue(), depth + 1});
        } else if (isa<arith::MinSIOp, arith::MaxSIOp>(op)) {
            for (auto operand : op->getOperands()) { work.push_back({operand, depth + 1}); }
        } else { return false; }
    }
    return true;
}

bool ProgramBuilder::prepareGuard(Value root, const ArithmeticSite& site)
{
    SmallVector<Value> work{root};
    DenseSet<Value> seen;
    while (!work.empty()) {
        auto value = work.pop_back_val();
        if (!seen.insert(value).second) {
            continue;
        }
        if (!value.getType().isInteger(1)) {
            output.extraction.note(RecognitionIssue::UnsupportedControl, value.getDefiningOp());
            return false;
        }
        if (booleanConstant(value) || constant(value)) {
            continue;
        }
        if (entryParameter(value)) {
            registerParameter(value);
            continue;
        }
        if (auto compare = value.getDefiningOp<arith::CmpIOp>()) {
            if (!supportedComparison(compare)) {
                output.extraction.note(RecognitionIssue::UnsupportedControl, compare);
                return false;
            }
            for (Value operand : compare->getOperands()) {
                if (!prepareValue(operand, site)) {
                    // The predicate kind is supported. Report the scalar that
                    // could not be normalized, rather than rejecting scf.if.
                    output.extraction.note(RecognitionIssue::IndexArithmetic,
                        operand.getDefiningOp() ? operand.getDefiningOp() : compare.getOperation());
                    return false;
                }
            }
            continue;
        }
        auto* operation = value.getDefiningOp();
        const bool andOr = operation && isa<arith::AndIOp, arith::OrIOp>(operation);
        auto exclusive = dyn_cast_or_null<arith::XOrIOp>(operation);
        const bool negation = exclusive && (booleanConstant(exclusive.getLhs()) ||
                                            booleanConstant(exclusive.getRhs()));
        if (!andOr && !negation) {
            output.extraction.note(RecognitionIssue::UnsupportedControl, operation);
            return false;
        }
        llvm::append_range(work, operation->getOperands());
    }
    return true;
}
void ProgramBuilder::emitForSites(PrimitiveRelation& relation, ArrayRef<AffineExpr> rows,
                                 ArrayRef<std::pair<const ArithmeticSite*, unsigned>> endpoints)
{
    if (output.finiteExpansion && output.extraction.state != RecognitionState::Applicable) { return; }
    Pieces pieces{Rows(rows)};
    for (auto [site, offset] : endpoints) {
        GuardCache cache;
        bool exceeded = false;
        auto domain = piecewiseDomain(*this, *site, offset, cache, exceeded);
        auto constrained = domain ? combine(std::move(pieces), std::move(*domain), true, &exceeded) :
                                    std::optional<Pieces>{};
        if (!constrained) {
            output.extraction.note(exceeded ? RecognitionIssue::TemplateExpansionLimit : RecognitionIssue::LoopDomain,
                                   site->phase ? site->phase->elementOp : nullptr);
            return;
        }
        pieces = std::move(*constrained);
        for (auto guard : site->guards) {
            auto branch = guard.branch;
            auto alternatives = condition(*this, branch.getCondition(), guard.takeThen, *site,
                                          offset, 0, cache, exceeded);
            auto combined = alternatives ? combine(std::move(pieces), std::move(*alternatives), true, &exceeded) :
                                          std::optional<Pieces>{};
            if (!combined) {
                output.extraction.note(exceeded ? RecognitionIssue::TemplateExpansionLimit :
                                       RecognitionIssue::UnsupportedControl, branch);
                return;
            }
            pieces = std::move(*combined);
        }
    }
    for (const auto& piece : pieces) {
        emit(relation, piece);
        if (output.finiteExpansion && output.extraction.state != RecognitionState::Applicable) { return; }
    }
}
} // namespace mlir::pto::frontiersynch::detail
