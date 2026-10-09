// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact substitution x = P*q+r, with checked fixed-width coefficients.
#include "ArithmeticProgramInternal.h"
#include "ArithmeticRows.h"
#include "../InsertSync/SyncScalarEvolution.h"
#include "../InsertSync/SyncScalarReplay.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/MathExtras.h"
namespace mlir::pto::frontiersynch::detail {
PrimitiveRelation ProgramBuilder::relation(PrimitiveKind kind, unsigned dimensions) const
{
    PrimitiveRelation result;
    result.kind = kind;
    result.dimensions = dimensions;
    for (unsigned i = 0; i < dimensions; ++i) {
        result.coordinates.push_back({"i" + std::to_string(i), CoordinateKind::Occurrence});
    }
    for (const auto& name : output.primitives.parameters) {
        result.coordinates.push_back({name, CoordinateKind::Parameter});
    }
    return result;
}
namespace {
// Lift constant floor divisions into existential quotient symbols. Original
// coordinate residues remain unchanged; these symbols are projected exactly
// by the relation importer instead of enumerating a joint residue period.
class LocalQuotients {
public:
    LocalQuotients(unsigned dimensions, unsigned symbols, unsigned limit)
        : dimensions(dimensions), symbols(symbols), limit(limit) {}
    SmallVector<AffineExpr> definitions;
    unsigned localCount = 0;
    bool dimensionExceeded = false;
    AffineExpr linearize(AffineExpr expression)
    {
        if (!expression) { return {}; }
        auto found = cache.find(expression);
        if (found != cache.end()) { return found->second; }
        auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
        if (!binary) { return expression; }
        auto left = linearize(binary.getLHS()), right = linearize(binary.getRHS());
        if (!left || !right) { return {}; }
        AffineExpr result;
        const auto kind = binary.getKind();
        if (kind == AffineExprKind::Add) {
            result = mlir::pto::detail::checkedAdd(left, right);
        } else if (kind == AffineExprKind::Mul) {
            if (!isa<AffineConstantExpr>(left) && !isa<AffineConstantExpr>(right)) { return {}; }
            result = mlir::pto::detail::checkedMul(left, right);
        } else {
            auto divisor = dyn_cast<AffineConstantExpr>(right);
            if (!divisor || divisor.getValue() <= 0) { return {}; }
            const auto denominator = divisor.getValue();
            auto* context = expression.getContext();
            const auto negative = getAffineConstantExpr(-1, context);
            if (kind == AffineExprKind::CeilDiv) {
                auto negated = mlir::pto::detail::checkedMul(left, negative);
                auto quotient = negated ? linearize(negated.floorDiv(denominator)) : AffineExpr{};
                result = mlir::pto::detail::checkedMul(quotient, negative);
            } else if (kind == AffineExprKind::FloorDiv || kind == AffineExprKind::Mod) {
                auto quotientForm = left.floorDiv(denominator);
                auto stored = quotients.find(quotientForm);
                AffineExpr quotient;
                if (stored != quotients.end()) {
                    quotient = stored->second;
                } else {
                    if (dimensions + symbols + localCount >= limit) {
                        dimensionExceeded = true;
                        return {};
                    }
                    quotient = getAffineSymbolExpr(symbols + localCount++, context);
                    auto multiple = mlir::pto::detail::checkedMul(quotient, divisor);
                    auto lower = mlir::pto::detail::checkedAdd(left,
                        mlir::pto::detail::checkedMul(multiple, negative));
                    auto upper = mlir::pto::detail::checkedAdd(
                        mlir::pto::detail::checkedMul(lower, negative),
                        getAffineConstantExpr(denominator - 1, context));
                    if (!lower || !upper) { return {}; }
                    definitions.push_back(lower);
                    definitions.push_back(upper);
                    quotients.try_emplace(quotientForm, quotient);
                }
                result = kind == AffineExprKind::FloorDiv ? quotient :
                    mlir::pto::detail::checkedAdd(left, mlir::pto::detail::checkedMul(
                        mlir::pto::detail::checkedMul(quotient, divisor), negative));
            }
        }
        if (result) { cache.try_emplace(expression, result); }
        return result;
    }
private:
    unsigned dimensions, symbols, limit;
    DenseMap<AffineExpr, AffineExpr> cache, quotients;
};
// Shared address normalization may retain an integer SSA value underneath a
// value-preserving index cast. Reuse that existing entry binding rather than
// inventing a new parameter or treating an integer as a pointer identity.
Value indexParameter(const ProgramBuilder& builder, Value input)
{
    if (builder.entryParameter(input)) { return input; }
    if (!isa<IntegerType>(input.getType())) { return {}; }
    for (auto* user : input.getUsers()) {
        auto cast = dyn_cast<arith::IndexCastOp>(user);
        if (!cast || cast.getIn() != input || !cast.getOut().getType().isIndex() ||
            !builder.entryParameter(cast.getOut())) { continue; }
        const auto variable = getAffineSymbolExpr(0, builder.context);
        mlir::pto::detail::ScalarEvolution evolution(builder.context, user);
        auto expression = evolution.value(cast.getOut(), [&](Value value) -> AffineExpr {
            return value == input ? variable : AffineExpr{};
        });
        // ScalarEvolution preserves this expression through the cast only
        // after proving its original signed range fits the index width.
        if (expression == variable) { return cast.getOut(); }
    }
    return {};
}
AffineExpr normalizeValue(Value input, const ArithmeticSite& site, unsigned offset,
                          MLIRContext* context, llvm::function_ref<AffineExpr(Value)> parameter)
{
    mlir::pto::detail::ScalarEvolution evolution(context, input.getDefiningOp() ? input.getDefiningOp() :
        input.getParentRegion()->getParentOp());
    return evolution.value(input, [&](Value value) -> AffineExpr {
        for (auto [id, storedLoop] : llvm::enumerate(site.loops)) {
            auto loop = storedLoop;
            if (value == loop.getInductionVar()) {
                return getAffineDimExpr(offset + id, context);
            }
        }
        return parameter(value);
    });
}
}
bool ProgramBuilder::entryParameter(Value value) const
{
    if (!value.getType().isIndex() && !value.getType().isInteger(1)) {
        return false;
    }
    auto argument = dyn_cast<BlockArgument>(value);
    if (argument && argument.getOwner() == &output.context.function.front()) {
        return true;
    }
    auto* root = output.context.root;
    if (root != output.context.function.getOperation() && index.valueAvailable(value, root, Boundary::Before)) {
        return true;
    }
    // A deterministic expression over invocation arguments is invariant even
    // when its definition is nested. Endpoint emission replays the original
    // operations at cuts where the definition is unavailable; no signed
    // division or wrapping arithmetic is replaced by mathematical arithmetic.
    std::function<bool(Value)> entry = [&](Value current) {
        if (auto found = entryInputs.find(current); found != entryInputs.end()) { return found->second; }
        entryInputs[current] = false;
        if (auto argument = dyn_cast<BlockArgument>(current)) {
            return entryInputs[current] = argument.getOwner() == &output.context.function.front();
        }
        auto* op = current.getDefiningOp();
        if (!mlir::pto::detail::canReplayScalar(op) ||
            !output.context.function->isProperAncestor(op) || !index.phasesFor(op).empty()) { return false; }
        return entryInputs[current] = llvm::all_of(op->getOperands(), entry);
    };
    return entry(value);
}
std::optional<int64_t> ProgramBuilder::constant(Value value) const
{
    // Availability excludes local loop IVs and values produced inside root.
    // Values from preceding payloads can be entry parameters; only the caller's
    // certified phase/interval equality can specialize one to a constant.
    return entryConstant && entryParameter(value) ? entryConstant(value) : std::nullopt;
}
AffineExpr ProgramBuilder::registerParameter(Value value)
{
    if (auto fixed = constant(value)) { return getAffineConstantExpr(*fixed, context); }
    auto inserted = parameterIds.try_emplace(value, output.parameters.size());
    if (inserted.second) {
        output.parameters.push_back(value);
        output.primitives.parameters.push_back("p" + std::to_string(inserted.first->second));
    }
    return getAffineSymbolExpr(inserted.first->second, context);
}
bool ProgramBuilder::prepareValue(Value input, const ArithmeticSite& site)
{
    if (constant(input)) { return true; }
    auto expression = normalizeValue(input, site, 0, context, [&](Value value) -> AffineExpr {
        auto binding = indexParameter(*this, value);
        return binding ? registerParameter(binding) : AffineExpr{};
    });
    return static_cast<bool>(expression);
}
AffineExpr ProgramBuilder::value(Value input, const ArithmeticSite& site, unsigned offset) const
{
    if (auto fixed = constant(input)) { return getAffineConstantExpr(*fixed, context); }
    return normalizeValue(input, site, offset, context, [&](Value input) -> AffineExpr {
        auto binding = indexParameter(*this, input);
        if (binding) {
            if (auto fixed = constant(binding)) { return getAffineConstantExpr(*fixed, context); }
        }
        auto parameter = parameterIds.find(binding);
        return parameter == parameterIds.end() ? AffineExpr{} : getAffineSymbolExpr(parameter->second, context);
    });
}
bool ProgramBuilder::staticallyEmpty(const ArithmeticSite& site) const
{
    for (auto storedLoop : site.loops) {
        auto loop = storedLoop;
        APInt lower, upper;
        const bool known = matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
                           matchPattern(loop.getUpperBound(), m_ConstantInt(&upper));
        if (known && lower.sge(upper)) {
            return true;
        }
    }
    return false;
}
SmallVector<AffineExpr> ProgramBuilder::domain(const ArithmeticSite& site, unsigned offset) const
{
    SmallVector<AffineExpr> rows;
    if (staticallyEmpty(site)) {
        rows.push_back(getAffineConstantExpr(-1, context));
        return rows;
    }
    for (auto [id, storedLoop] : llvm::enumerate(site.loops)) {
        auto loop = storedLoop;
        const auto iv = getAffineDimExpr(offset + id, context);
        auto lower = value(loop.getLowerBound(), site, offset);
        auto upper = value(loop.getUpperBound(), site, offset);
        // Piecewise bounds, including the lower-origin congruence, are emitted
        // together in emitForSites after choosing their exact alternatives.
        if (!lower || !upper) { continue; }
        auto step = cast<AffineConstantExpr>(value(loop.getStep(), site, offset)).getValue();
        auto distance = mlir::pto::detail::checkedAdd(iv, mlir::pto::detail::checkedMul(
            lower, getAffineConstantExpr(-1, context)));
        rows.push_back(distance);
        auto constant = dyn_cast<AffineConstantExpr>(upper);
        // Avoid constructing INT64_MIN-1 for an already empty domain.
        auto bound = constant && constant.getValue() == INT64_MIN ? getAffineConstantExpr(-1, context) :
            mlir::pto::detail::checkedAdd(mlir::pto::detail::checkedAdd(upper, -iv),
                                        getAffineConstantExpr(-1, context));
        rows.push_back(bound);
        if (step != 1) {
            rows.push_back(distance ? -(distance % step) : AffineExpr{});
        }
    }
    return rows;
}
void ProgramBuilder::emit(PrimitiveRelation& target, ArrayRef<AffineExpr> rows,
                         std::optional<std::pair<unsigned, uint64_t>> filter, uint64_t modulus)
{
    const auto count = target.coordinates.size();
    if (count > limits.dimensions) {
        output.extraction.note(RecognitionIssue::ArithmeticDimension, nullptr, true);
        return;
    }
    const auto period = limits.period;
    std::size_t combinations = 1;
    for (std::size_t i = 0; i < count; ++i) {
        if (combinations > SIZE_MAX / period) {
            output.extraction.note(RecognitionIssue::ArithmeticConfiguration, nullptr);
            return;
        }
        combinations *= period;
    }
    // Entry i1 values have the mathematical representation 0 or 1. Keep this
    // context in every relation, including unconditional sites outside a branch.
    SmallVector<AffineExpr> constrainedRows(rows);
    for (auto [id, parameter] : llvm::enumerate(output.parameters)) {
        if (parameter.getType().isInteger(1)) {
            auto value = getAffineSymbolExpr(id, context);
            constrainedRows.push_back(value);
            constrainedRows.push_back(1 - value);
        }
    }
    // Iterative mixed-radix enumeration: fixed P and D bound its expansion.
    SmallVector<uint64_t> residues(count, 0);
    bool more = true;
    while (more) {
        if (!filter || residues[filter->first] % modulus == filter->second) {
            SmallVector<AffineExpr> constraints;
            SmallVector<AffineExpr> dimensions, symbols;
            for (auto [id, residue] : llvm::enumerate(residues)) {
                auto coordinate = id < target.dimensions ? getAffineDimExpr(id, context) :
                    getAffineSymbolExpr(id - target.dimensions, context);
                auto replacement = mlir::pto::detail::checkedAdd(
                    mlir::pto::detail::checkedMul(coordinate, getAffineConstantExpr(period, context)),
                    getAffineConstantExpr(residue, context));
                (id < target.dimensions ? dimensions : symbols).push_back(replacement);
            }
            LocalQuotients locals(target.dimensions, count - target.dimensions, limits.dimensions);
            for (auto row : constrainedRows) {
                auto expression = mlir::pto::detail::substitute(row, dimensions, symbols);
                if (expression) {
                    expression = simplifyAffineExpr(expression, target.dimensions, count - target.dimensions);
                }
                expression = locals.linearize(expression);
                LinearRow checked;
                if (!expression || collectRow(expression, target.dimensions,
                    count - target.dimensions + locals.localCount, checked)) {
                    Operation* anchor = target.sourceSite && *target.sourceSite < output.sites.size() ?
                        output.sites[*target.sourceSite].phase->elementOp : nullptr;
                    output.extraction.note(locals.dimensionExceeded ? RecognitionIssue::ArithmeticDimension :
                        RecognitionIssue::IndexArithmetic, anchor, locals.dimensionExceeded);
                    return;
                }
                constraints.push_back(expression);
            }
            constraints.append(locals.definitions.begin(), locals.definitions.end());
            if (constraints.empty()) {
                constraints.push_back(getAffineConstantExpr(0, context));
            }
            SmallVector<bool> equalities(constraints.size(), false);
            target.pieces.push_back({IntegerSet::get(target.dimensions, count - target.dimensions + locals.localCount,
                                                    constraints, equalities), residues});
        }
        more = false;
        for (auto& residue : residues) {
            if (++residue < period) {
                more = true;
                break;
            }
            residue = 0;
        }
    }
}
} // namespace mlir::pto::frontiersynch::detail
