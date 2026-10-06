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
std::optional<Pieces> combine(Pieces left, Pieces right, bool conjunction)
{
    if (!conjunction) {
        for (const auto& piece : right) {
            if (!llvm::is_contained(left, piece)) {
                const bool full = left.size() == maxGuardPieces;
                if (full) {
                    return std::nullopt;
                }
                left.push_back(piece);
            }
        }
        return left;
    }
    const bool oversized = !right.empty() && left.size() > maxGuardPieces / right.size();
    if (oversized) {
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
                               GuardCache& cache);
std::optional<Pieces> buildCondition(ProgramBuilder& builder, Value value, bool truth,
                               const ArithmeticSite& site, unsigned offset, unsigned depth, GuardCache& cache)
{
    if (depth > maxGuardDepth) {
        return std::nullopt;
    }
    if (auto constant = booleanConstant(value)) {
        return *constant == truth ? Pieces{Rows{}} : Pieces{};
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
                             truth != (left ? *left : *right), site, offset, depth + 1, cache);
        }
        return std::nullopt;
    }
    const bool andOr = operation && isa<arith::AndIOp, arith::OrIOp>(operation);
    if (!andOr) {
        return std::nullopt;
    }
    auto left = condition(builder, operation->getOperand(0), truth, site, offset, depth + 1, cache);
    auto right = condition(builder, operation->getOperand(1), truth, site, offset, depth + 1, cache);
    if (!left || !right) {
        return std::nullopt;
    }
    const bool conjunction = isa<arith::AndIOp>(operation) == truth;
    return combine(std::move(*left), std::move(*right), conjunction);
}
std::optional<Pieces> condition(ProgramBuilder& builder, Value value, bool truth,
                               const ArithmeticSite& site, unsigned offset, unsigned depth, GuardCache& cache)
{
    if (depth > maxGuardDepth) {
        return std::nullopt;
    }
    auto key = std::make_pair(value, static_cast<unsigned>(truth));
    auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }
    auto result = buildCondition(builder, value, truth, site, offset, depth, cache);
    if (result) {
        cache[key] = *result;
    }
    return result;
}
} // namespace

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
            return false;
        }
        if (booleanConstant(value)) {
            continue;
        }
        if (entryParameter(value)) {
            registerParameter(value);
            continue;
        }
        if (auto compare = value.getDefiningOp<arith::CmpIOp>()) {
            const bool supported = supportedComparison(compare) && prepareValue(compare.getLhs(), site) &&
                                   prepareValue(compare.getRhs(), site);
            if (!supported) {
                return false;
            }
            continue;
        }
        auto* operation = value.getDefiningOp();
        const bool andOr = operation && isa<arith::AndIOp, arith::OrIOp>(operation);
        auto exclusive = dyn_cast_or_null<arith::XOrIOp>(operation);
        const bool negation = exclusive && (booleanConstant(exclusive.getLhs()) ||
                                            booleanConstant(exclusive.getRhs()));
        if (!andOr && !negation) {
            return false;
        }
        llvm::append_range(work, operation->getOperands());
    }
    return true;
}
void ProgramBuilder::emitForSites(PrimitiveRelation& relation, ArrayRef<AffineExpr> rows,
                                 ArrayRef<std::pair<const ArithmeticSite*, unsigned>> endpoints)
{
    Pieces pieces{Rows(rows)};
    for (auto [site, offset] : endpoints) {
        GuardCache cache;
        for (auto guard : site->guards) {
            auto branch = guard.branch;
            auto alternatives = condition(*this, branch.getCondition(), guard.takeThen, *site, offset, 0, cache);
            auto combined = alternatives ? combine(std::move(pieces), std::move(*alternatives), true) :
                                          std::optional<Pieces>{};
            if (!combined) {
                output.extraction.note(RecognitionIssue::UnsupportedControl, branch);
                return;
            }
            pieces = std::move(*combined);
        }
    }
    for (const auto& piece : pieces) {
        emit(relation, piece);
    }
}
} // namespace mlir::pto::frontiersynch::detail
