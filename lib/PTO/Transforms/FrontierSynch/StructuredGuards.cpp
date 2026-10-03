// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "StructuredInternal.h"
#include "mlir/IR/IntegerSet.h"
namespace mlir::pto::frontiersynch::structured {
namespace {
SmallVector<Poly, 0> predicate(const Poly& poly, const Form& difference, bool equality, bool truth)
{
    SmallVector<Poly, 0> result;
    if (truth) {
        result.push_back(poly);
        result.back().constrain(difference, equality);
    } else {
        result.push_back(poly);
        result.back().constrain(difference.scale(Int(-1)).plus(Form::number(Int(-1))));
        if (equality) {
            result.push_back(poly);
            result.back().constrain(difference.plus(Form::number(Int(-1))));
        }
    }
    return result;
}
} // namespace
FailureOr<SmallVector<Poly, 0>> Builder::guard(
    Value condition, bool truth, const StructuredSite& site, const Poly& input)
{
    if (!condition.getType().isInteger(1)) {
        return fail(StructuredImportStatus::UnsupportedGuard, site.anchor, "non-Boolean condition", condition);
    }
    if (llvm::is_contained(schema->parameters(), condition)) {
        Poly poly = input;
        auto form = value(condition, site, poly);
        if (failed(form)) {
            return failure();
        }
        poly.constrain(form->minus(Form::number(Int(truth ? 1 : 0))), true);
        return SmallVector<Poly, 0>{std::move(poly)};
    }
    auto* op = condition.getDefiningOp();
    if (op && isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
        SmallVector<Poly, 0> result;
        // Negate the Boolean function through truth assignments. Recursive
        // operand definitions stay positive and independently fresh per branch.
        for (bool left : {false, true}) {
            for (bool right : {false, true}) {
                bool value = isa<arith::AndIOp>(op) ? left && right :
                             isa<arith::OrIOp>(op)  ? left || right :
                                                      left != right;
                if (value != truth) {
                    continue;
                }
                auto first = guard(op->getOperand(0), left, site, input);
                if (failed(first)) {
                    return failure();
                }
                for (const auto& branch : *first) {
                    auto second = guard(op->getOperand(1), right, site, branch);
                    if (failed(second)) {
                        return failure();
                    }
                    append(result, std::move(*second));
                }
            }
        }
        return result;
    }
    if (auto compare = dyn_cast_or_null<arith::CmpIOp>(op)) {
        auto comparison = compare.getPredicate();
        using Predicate = arith::CmpIPredicate;
        if (comparison == Predicate::ult || comparison == Predicate::ule || comparison == Predicate::ugt ||
            comparison == Predicate::uge ||
            (compare.getLhs().getType().isInteger(1) && comparison != Predicate::eq && comparison != Predicate::ne)) {
            return fail(StructuredImportStatus::UnsupportedGuard, op, "unsigned or signed relational i1 comparison");
        }
        Poly poly = input;
        auto left = value(compare.getLhs(), site, poly);
        auto right = value(compare.getRhs(), site, poly);
        if (failed(left) || failed(right)) {
            return failure();
        }
        Form difference = left->minus(*right);
        bool equality = comparison == Predicate::eq || comparison == Predicate::ne;
        if (comparison == Predicate::ne) {
            truth = !truth;
        }
        if (comparison == Predicate::slt || comparison == Predicate::sle) {
            difference = difference.scale(Int(-1));
        }
        if (comparison == Predicate::slt || comparison == Predicate::sgt) {
            difference = difference.plus(Form::number(Int(-1)));
        }
        return predicate(poly, difference, equality, truth);
    }
    Poly poly = input;
    auto form = value(condition, site, poly);
    if (failed(form)) {
        return failure();
    }
    poly.constrain(form->minus(Form::number(Int(truth ? 1 : 0))), true);
    return SmallVector<Poly, 0>{std::move(poly)};
}
FailureOr<SmallVector<Poly, 0>> Builder::affineGuard(
    affine::AffineIfOp op, bool truth, const StructuredSite& site, const Poly& input)
{
    if (!qualify(op)) {
        return failure();
    }
    Poly poly = input;
    auto set = op.getIntegerSet();
    SmallVector<Form> operands, conditions;
    for (auto operand : op.getOperands()) {
        auto form = value(operand, site, poly);
        if (failed(form)) {
            return failure();
        }
        operands.push_back(*form);
    }
    for (auto expr : set.getConstraints()) {
        auto form = affine(expr, operands, set.getNumDims(), poly, op);
        if (failed(form)) {
            return failure();
        }
        conditions.push_back(*form);
    }
    SmallVector<Poly, 0> result;
    if (truth) {
        for (unsigned i = 0; i < conditions.size(); ++i) {
            poly.constrain(conditions[i], set.isEq(i));
        }
        result.push_back(std::move(poly));
    } else {
        // The quotient/remainder definitions remain positive. De Morgan applies
        // to the deterministic constraint values, not to existential witnesses.
        for (unsigned i = 0; i < conditions.size(); ++i) {
            append(result, predicate(poly, conditions[i], set.isEq(i), false));
        }
    }
    return result;
}
FailureOr<SmallVector<Poly, 0>> Builder::loop(Operation* operation, const StructuredSite& site, const Poly& input)
{
    if (!qualify(operation)) {
        return failure();
    }
    Poly poly = input;
    Value iv;
    Int step(0);
    SmallVector<Form> lower, upper;
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        iv = loop.getInductionVar();
        auto constant = loop.getStep().getDefiningOp<arith::ConstantOp>();
        auto attr = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
        if (!attr) {
            return fail(StructuredImportStatus::UnsupportedControl, operation, "nonconstant SCF loop step");
        }
        step = integer(attr.getValue());
        auto lb = bound(loop.getLowerBound(), false, site, poly);
        auto ub = bound(loop.getUpperBound(), true, site, poly);
        if (failed(lb) || failed(ub)) {
            return failure();
        }
        lower = std::move(*lb);
        upper = std::move(*ub);
    } else if (auto loop = dyn_cast<affine::AffineForOp>(operation)) {
        iv = loop.getInductionVar();
        step = integer(loop.getStep());
        auto lbs = map(loop.getLowerBoundMap(), loop.getLowerBoundOperands(), site, poly, operation);
        auto ubs = map(loop.getUpperBoundMap(), loop.getUpperBoundOperands(), site, poly, operation);
        if (failed(lbs) || failed(ubs)) {
            return failure();
        }
        lower = std::move(*lbs);
        upper = std::move(*ubs);
    } else {
        return fail(StructuredImportStatus::UnsupportedControl, operation, "unsupported loop");
    }
    if (step <= 0 || lower.empty() || upper.empty()) {
        return fail(StructuredImportStatus::UnsupportedControl, operation, "nonpositive step or empty affine bound");
    }
    auto coordinate = value(iv, site, poly);
    if (failed(coordinate)) {
        return failure();
    }
    SmallVector<Poly, 0> result;
    for (const auto& selected : lower) {
        Poly branch = poly;
        for (const auto& bound : lower) {
            branch.constrain(selected.minus(bound));
        }
        branch.constrain(coordinate->minus(selected));
        for (const auto& bound : upper) {
            branch.constrain(bound.minus(*coordinate).plus(Form::number(Int(-1))));
        }
        // Unit stride admits every integer within the bounds. Avoid a
        // redundant existential trip variable so literal octagonal presence
        // remains available to exact endpoint specialization.
        if (step != 1) {
            auto trip = branch.local();
            if (failed(trip)) {
                return fail(StructuredImportStatus::RepresentationLimit, operation, "loop column count overflow");
            }
            branch.constrain(*trip);
            branch.constrain(coordinate->minus(selected).minus(trip->scale(step)), true);
        }
        result.push_back(std::move(branch));
    }
    return result;
}
FailureOr<SmallVector<Poly, 0>> Builder::presence(std::size_t position)
{
    const auto& site = sites[position];
    Poly initial(*schema->space(SymbolicTuple::Unit, SymbolicTuple::Occurrence));
    auto tag = (sharedTags ? *sharedTags : siteTags).lookup(site.phase);
    initial.constrain(Form::axis(0).minus(Form::number(Int(static_cast<std::int64_t>(tag)))), true);
    for (unsigned i = site.inductionVariables.size(); i < schema->coordinateDepth(); ++i) {
        initial.constrain(Form::axis(i + 1), true);
    }
    SmallVector<Poly, 0> result{std::move(initial)};
    for (Region* region : llvm::drop_begin(site.regions)) {
        Operation* op = region->getParentOp();
        SmallVector<Poly, 0> next;
        for (const auto& poly : result) {
            FailureOr<SmallVector<Poly, 0>> branches = failure();
            if (isa<scf::ForOp, affine::AffineForOp>(op)) {
                branches = loop(op, site, poly);
            } else if (auto condition = dyn_cast<scf::IfOp>(op)) {
                branches = guard(condition.getCondition(), region == &condition.getThenRegion(), site, poly);
            } else if (auto condition = dyn_cast<affine::AffineIfOp>(op)) {
                branches = affineGuard(condition, region == &condition.getThenRegion(), site, poly);
            }
            if (failed(branches)) {
                return failure();
            }
            append(next, std::move(*branches));
        }
        result = std::move(next);
    }
    return result;
}
} // namespace mlir::pto::frontiersynch::structured
