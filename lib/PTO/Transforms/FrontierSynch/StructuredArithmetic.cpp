// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "StructuredInternal.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include <limits>
namespace mlir::pto::frontiersynch::structured {
Int integer(const APInt& value, bool isUnsigned)
{
    llvm::SmallString<64> digits;
    value.toString(digits, 10, !isUnsigned);
    StringRef text(digits);
    bool negative = text.starts_with("-");
    if (negative) {
        text = text.drop_front();
    }
    Int result(0);
    for (char digit : text) {
        result = result * 10 + (digit - '0');
    }
    return negative ? -result : result;
}
Form Form::axis(unsigned column)
{
    Form result;
    result.terms.emplace(column, Int(1));
    return result;
}
Form Form::number(const Int& value)
{
    Form result;
    result.constant = value;
    return result;
}
Form Form::scale(const Int& value) const
{
    Form result;
    result.constant = constant * value;
    for (const auto& [column, coefficient] : terms) {
        if (coefficient * value != 0) {
            result.terms.emplace(column, coefficient * value);
        }
    }
    return result;
}
Form Form::plus(const Form& other) const
{
    Form result = *this;
    result.constant += other.constant;
    for (const auto& [column, coefficient] : other.terms) {
        result.terms[column] += coefficient;
        if (result.terms[column] == 0) {
            result.terms.erase(column);
        }
    }
    return result;
}
void Poly::constrain(const Form& value, bool equality)
{
    SmallVector<Int> row(relation.getNumCols(), Int(0));
    for (const auto& [column, coefficient] : value.terms) {
        row[column] = coefficient;
    }
    row.back() = value.constant;
    if (equality) {
        relation.addEquality(row);
    } else {
        relation.addInequality(row);
    }
}
FailureOr<Form> Poly::local()
{
    if (relation.getNumVars() >= std::numeric_limits<unsigned>::max() - 1) {
        return failure();
    }
    unsigned column = relation.getNumVars();
    relation.appendVar(presburger::VarKind::Local);
    return Form::axis(column);
}
void append(SmallVector<Poly, 0>& target, SmallVector<Poly, 0> source)
{
    for (auto& poly : source) {
        target.push_back(std::move(poly));
    }
}
LogicalResult Builder::fail(StructuredImportStatus status, Operation* operation, StringRef reason, Value value)
{
    if (issue.status == StructuredImportStatus::Success) {
        issue = {status, operation, value, reason.str()};
    }
    return failure();
}
bool Builder::qualify(Operation* operation)
{
    if (options.mathematicalArithmetic && options.mathematicalArithmetic(operation, schema, effects.context)) {
        return true;
    }
    (void)fail(
        StructuredImportStatus::MissingQualification, operation,
        "mathematical arithmetic/loop progression requires an admitted-context certificate");
    return false;
}
FailureOr<Form> Builder::affine(AffineExpr expr, ArrayRef<Form> operands, unsigned dims, Poly& poly, Operation* op)
{
    if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
        return Form::number(Int(constant.getValue()));
    }
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
        return operands[dim.getPosition()];
    }
    if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) {
        return operands[dims + symbol.getPosition()];
    }
    auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!binary) {
        return fail(StructuredImportStatus::UnsupportedArithmetic, op, "unknown affine expression");
    }
    auto left = affine(binary.getLHS(), operands, dims, poly, op);
    auto right = affine(binary.getRHS(), operands, dims, poly, op);
    if (failed(left) || failed(right)) {
        return failure();
    }
    if (expr.getKind() == AffineExprKind::Add) {
        return left->plus(*right);
    }
    if (expr.getKind() == AffineExprKind::Mul) {
        if (right->terms.empty()) {
            return left->scale(right->constant);
        }
        if (left->terms.empty()) {
            return right->scale(left->constant);
        }
    } else if (right->terms.empty() && right->constant > 0) {
        // Quotient definitions are total and deterministic. Guard negation below
        // negates only its final predicate, never these existential definitions.
        Form numerator = expr.getKind() == AffineExprKind::CeilDiv ? left->scale(Int(-1)) : *left;
        auto fresh = poly.local();
        if (failed(fresh)) {
            return fail(StructuredImportStatus::RepresentationLimit, op, "existential column count overflow");
        }
        Form quotient = *fresh;
        Form remainder = numerator.minus(quotient.scale(right->constant));
        poly.constrain(remainder);
        poly.constrain(Form::number(right->constant - 1).minus(remainder));
        if (expr.getKind() == AffineExprKind::FloorDiv) {
            return quotient;
        }
        if (expr.getKind() == AffineExprKind::CeilDiv) {
            return quotient.scale(Int(-1));
        }
        if (expr.getKind() == AffineExprKind::Mod) {
            return remainder;
        }
    }
    return fail(
        StructuredImportStatus::UnsupportedArithmetic, op, "nonconstant product/divisor or nonpositive divisor");
}
FailureOr<SmallVector<Form>> Builder::map(
    AffineMap expression, ValueRange operands, const StructuredSite& site, Poly& poly, Operation* op)
{
    if (expression.getNumInputs() != operands.size()) {
        return fail(StructuredImportStatus::InvalidInput, op, "affine operand arity mismatch");
    }
    SmallVector<Form> values, results;
    for (Value operand : operands) {
        auto form = value(operand, site, poly);
        if (failed(form)) {
            return failure();
        }
        values.push_back(*form);
    }
    for (auto expr : expression.getResults()) {
        auto form = affine(expr, values, expression.getNumDims(), poly, op);
        if (failed(form)) {
            return failure();
        }
        results.push_back(*form);
    }
    return results;
}
FailureOr<Form> Builder::value(Value operand, const StructuredSite& site, Poly& poly)
{
    const auto key = std::make_pair(operand, site.anchor);
    auto found = poly.forms.find(key);
    if (found != poly.forms.end()) {
        return found->second;
    }
    auto result = evaluateValue(operand, site, poly);
    if (succeeded(result)) {
        poly.forms.try_emplace(key, *result);
    }
    return result;
}
FailureOr<SmallVector<Form>> Builder::bound(Value operand, bool upper, const StructuredSite& site, Poly& poly)
{
    SmallVector<Value> pending{operand};
    llvm::DenseSet<Value> visited;
    SmallVector<Form> result;
    // Flatten the bound's source DAG once. Shared min/max operands are
    // idempotent; expanding their expression tree would duplicate constraints
    // exponentially. Forms with locals remain scoped to the supplied Poly.
    while (!pending.empty()) {
        Value current = pending.pop_back_val();
        if (!visited.insert(current).second) {
            continue;
        }
        auto* operation = current.getDefiningOp();
        if ((upper && isa_and_nonnull<arith::MinSIOp>(operation)) ||
            (!upper && isa_and_nonnull<arith::MaxSIOp>(operation))) {
            if (!qualify(operation)) {
                return failure();
            }
            llvm::append_range(pending, llvm::reverse(operation->getOperands()));
            continue;
        }
        auto form = value(current, site, poly);
        if (failed(form)) {
            return failure();
        }
        result.push_back(*form);
    }
    return result;
}
FailureOr<Form> Builder::evaluateValue(Value operand, const StructuredSite& site, Poly& poly)
{
    const auto& phases = options.context ? options.context->phases() : index;
    if (!operand || !phases.valueAvailable(operand, site.anchor, Boundary::Before)) {
        return fail(StructuredImportStatus::InvalidInput, site.anchor, "unavailable control operand", operand);
    }
    for (unsigned i = 0; i < site.inductionVariables.size(); ++i) {
        if (operand == site.inductionVariables[i]) {
            return Form::axis(i + 1);
        }
    }
    for (unsigned i = 0; i < schema->parameters().size(); ++i) {
        if (operand == schema->parameters()[i]) {
            return Form::axis(schema->coordinateDepth() + 1 + i);
        }
    }
    auto* op = operand.getDefiningOp();
    if (auto constant = dyn_cast_or_null<arith::ConstantOp>(op)) {
        auto attr = dyn_cast<IntegerAttr>(constant.getValue());
        if (!attr) {
            return fail(StructuredImportStatus::UnsupportedArithmetic, op, "noninteger constant", operand);
        }
        auto type = dyn_cast<IntegerType>(attr.getType());
        return Form::number(integer(attr.getValue(), type && (type.isUnsigned() || type.getWidth() == 1)));
    }
    if (!op) {
        return fail(StructuredImportStatus::UnsupportedArithmetic, site.anchor, "unqualified block argument", operand);
    }
    if (auto apply = dyn_cast<affine::AffineApplyOp>(op)) {
        if (!qualify(op)) {
            return failure();
        }
        auto forms = map(apply.getAffineMap(), apply.getMapOperands(), site, poly, op);
        if (failed(forms)) {
            return failure();
        }
        return forms->front();
    }
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
        if (!qualify(op)) {
            return failure();
        }
        auto left = value(op->getOperand(0), site, poly);
        auto right = value(op->getOperand(1), site, poly);
        if (failed(left) || failed(right)) {
            return failure();
        }
        if (isa<arith::AddIOp>(op)) {
            return left->plus(*right);
        }
        if (isa<arith::SubIOp>(op)) {
            return left->minus(*right);
        }
        if (right->terms.empty()) {
            return left->scale(right->constant);
        }
        if (left->terms.empty()) {
            return right->scale(left->constant);
        }
    }
    if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::TruncIOp>(op)) {
        if (!qualify(op)) {
            return failure();
        }
        // i1 uses Boolean 0/1 in the schema; signed extension interprets true as
        // -1. Decline this case instead of treating it as an identity cast.
        auto sourceType = dyn_cast<IntegerType>(op->getOperand(0).getType());
        if (sourceType && sourceType.getWidth() == 1) {
            return fail(
                StructuredImportStatus::UnsupportedArithmetic, op, "signed i1 cast is not a mathematical identity");
        }
        return value(op->getOperand(0), site, poly);
    }
    return fail(StructuredImportStatus::UnsupportedArithmetic, op, "unsupported control arithmetic", operand);
}
} // namespace mlir::pto::frontiersynch::structured
