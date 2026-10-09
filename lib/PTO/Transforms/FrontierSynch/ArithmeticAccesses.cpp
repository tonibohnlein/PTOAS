// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact physical byte relations from shared access maps. A symbolic origin and
// a fixed local byte union stay compact; unsupported local geometry is rejected.
#include "ArithmeticProgramInternal.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "../InsertSync/SyncRegionArithmetic.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/MapVector.h"
#include <algorithm>
namespace mlir::pto::frontiersynch::detail {
namespace {
using Interval = std::pair<int64_t, int64_t>;
struct BoundedExpression {
    AffineExpr expression;
    std::optional<Interval> interval;
};
std::optional<Interval> symbolInterval(Value value)
{
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant)) && constant.isSignedIntN(64)) {
        return Interval{constant.getSExtValue(), constant.getSExtValue()};
    }
    auto argument = dyn_cast<BlockArgument>(value);
    auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp{};
    APInt lower, upper, step;
    if (!loop || argument != loop.getInductionVar() ||
        !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
        !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
        !lower.isSignedIntN(64) || !upper.isSignedIntN(64) || !step.isSignedIntN(64) ||
        step.getSExtValue() <= 0 || lower.getSExtValue() >= upper.getSExtValue()) {
        return std::nullopt;
    }
    return Interval{lower.getSExtValue(), upper.getSExtValue() - 1};
}
int64_t floorQuotient(int64_t numerator, int64_t positiveDenominator)
{
    const auto quotient = numerator / positiveDenominator;
    return numerator % positiveDenominator < 0 ? quotient - 1 : quotient;
}
// Prove simplifications from the actual SSA loop domains, before replacing
// their IVs with arithmetic coordinates. In particular, a uint32_t truncation
// becomes an identity only on a proved [0, 2^32) range. Negative values and
// intervals crossing a wrap point retain their exact modular semantics.
BoundedExpression boundedOrigin(AffineExpr expression, ArrayRef<Value> symbols,
                                DenseMap<AffineExpr, BoundedExpression>& cache)
{
    auto found = cache.find(expression);
    if (found != cache.end()) { return found->second; }
    BoundedExpression result{expression, std::nullopt};
    if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
        result.interval = Interval{constant.getValue(), constant.getValue()};
    } else if (auto symbol = dyn_cast<AffineSymbolExpr>(expression)) {
        if (symbol.getPosition() < symbols.size()) {
            result.interval = symbolInterval(symbols[symbol.getPosition()]);
        }
    } else if (auto binary = dyn_cast<AffineBinaryOpExpr>(expression)) {
        auto left = boundedOrigin(binary.getLHS(), symbols, cache);
        auto right = boundedOrigin(binary.getRHS(), symbols, cache);
        const auto kind = binary.getKind();
        if (kind == AffineExprKind::Add || kind == AffineExprKind::Mul) {
            result.expression = kind == AffineExprKind::Add ?
                mlir::pto::detail::checkedAdd(left.expression, right.expression) :
                mlir::pto::detail::checkedMul(left.expression, right.expression);
            if (left.interval && right.interval) {
                int64_t low, high;
                if (kind == AffineExprKind::Add) {
                    if (!llvm::AddOverflow(left.interval->first, right.interval->first, low) &&
                        !llvm::AddOverflow(left.interval->second, right.interval->second, high)) {
                        result.interval = Interval{low, high};
                    }
                } else {
                    int64_t products[4];
                    if (!llvm::MulOverflow(left.interval->first, right.interval->first, products[0]) &&
                        !llvm::MulOverflow(left.interval->first, right.interval->second, products[1]) &&
                        !llvm::MulOverflow(left.interval->second, right.interval->first, products[2]) &&
                        !llvm::MulOverflow(left.interval->second, right.interval->second, products[3])) {
                        auto bounds = std::minmax_element(products, products + 4);
                        result.interval = Interval{*bounds.first, *bounds.second};
                    }
                }
            }
        } else if (auto divisor = dyn_cast<AffineConstantExpr>(right.expression);
                   divisor && divisor.getValue() > 0 &&
                   (kind == AffineExprKind::Mod || kind == AffineExprKind::FloorDiv)) {
            const int64_t denominator = divisor.getValue();
            result.expression = kind == AffineExprKind::Mod ? left.expression % denominator :
                left.expression.floorDiv(denominator);
            if (kind == AffineExprKind::Mod) { result.interval = Interval{0, denominator - 1}; }
            if (left.interval) {
                const auto low = floorQuotient(left.interval->first, denominator);
                const auto high = floorQuotient(left.interval->second, denominator);
                if (kind == AffineExprKind::FloorDiv) { result.interval = Interval{low, high}; }
                if (low == high) {
                    if (kind == AffineExprKind::FloorDiv) {
                        result.expression = getAffineConstantExpr(low, expression.getContext());
                    } else {
                        int64_t offset, negative;
                        if (!llvm::MulOverflow(low, denominator, offset) &&
                            !llvm::SubOverflow(int64_t{0}, offset, negative)) {
                            result.expression = mlir::pto::detail::checkedAdd(left.expression,
                                getAffineConstantExpr(negative, expression.getContext()));
                            int64_t begin, end;
                            if (!llvm::SubOverflow(left.interval->first, offset, begin) &&
                                !llvm::SubOverflow(left.interval->second, offset, end)) {
                                result.interval = Interval{begin, end};
                            }
                        }
                    }
                }
            }
        }
        // Failed checked reconstruction cannot justify replacing the map.
        if (!result.expression) { result = {expression, std::nullopt}; }
    }
    cache.try_emplace(expression, result);
    return result;
}
AffineExpr add(AffineExpr a, AffineExpr b)
{
    return mlir::pto::detail::checkedAdd(a, b);
}
AffineExpr negate(AffineExpr expression)
{
    return expression ? mlir::pto::detail::checkedMul(expression,
        getAffineConstantExpr(-1, expression.getContext())) : AffineExpr{};
}
void emitRange(ProgramBuilder& builder, std::size_t siteId, const SyncStorageEffect& effect,
               const SyncStorageCell& range, AffineExpr origin, Value base = {})
{
    const bool valid = range.begin < range.end && range.end <= static_cast<uint64_t>(INT64_MAX) && origin;
    if (!valid) {
        builder.output.extraction.note(RecognitionIssue::UnknownGeometry, effect.phase->elementOp);
        return;
    }
    const auto& site = builder.output.sites[siteId];
    const unsigned depth = site.loops.size();
    auto relation = builder.relation(effect.mode == SyncAccessMode::Read ? PrimitiveKind::Reads : PrimitiveKind::Writes,
                                     depth + 1);
    relation.sourceSite = siteId;
    relation.sourceDimensions = depth;
    relation.storageSpace = range.space;
    relation.storageBase = base ? base : range.base;
    relation.coordinates[depth] = {"byte", CoordinateKind::Storage};
    auto rows = builder.domain(site, 0);
    auto byte = getAffineDimExpr(depth, builder.context);
    auto begin = add(origin, getAffineConstantExpr(range.begin, builder.context));
    auto end = add(origin, getAffineConstantExpr(range.end - 1, builder.context));
    rows.push_back(add(byte, negate(begin)));
    rows.push_back(add(end, -byte));
    FiniteAccessRecipe recipe;
    if (builder.output.finiteExpansion && !depth) {
        SmallVector<AffineExpr> zeros(builder.output.parameters.size(), getAffineConstantExpr(0, builder.context));
        auto atZero = mlir::pto::detail::substitute(origin, {}, zeros);
        auto fixed = dyn_cast_or_null<AffineConstantExpr>(atZero ? simplifyAffineExpr(atZero, 0, 0) : AffineExpr{});
        if (fixed) {
            const auto translation = add(origin, negate(fixed));
            if (translation) {
                recipe.translation = simplifyAffineExpr(translation, 0, builder.output.parameters.size());
            }
        }
    }
    builder.emitForSites(relation, rows, {{&site, 0}}, recipe.translation ? &recipe.rows : nullptr);
    if (recipe.translation) {
        recipe.relation = builder.output.primitives.relations.size();
        builder.output.finiteAccessRecipes.push_back(std::move(recipe));
    }
    builder.output.primitives.relations.push_back(std::move(relation));
}
bool symbolicRegion(ProgramBuilder& builder, std::size_t siteId, const SyncStorageEffect& effect,
                    const SyncAccessRegion& region)
{
    if (!region.byteOffset) {
        return false;
    }
    SmallVector<AffineExpr> zeros(region.extents.size(), getAffineConstantExpr(0, builder.context));
    SmallVector<AffineExpr> symbols, occurrenceSymbols;
    for (auto [id, symbol] : llvm::enumerate(region.symbols)) {
        symbols.push_back(getAffineSymbolExpr(id, builder.context));
        auto value = builder.value(symbol, builder.output.sites[siteId], 0);
        if (!value) {
            return false;
        }
        occurrenceSymbols.push_back(value);
    }
    auto localOrigin = mlir::pto::detail::substitute(region.byteOffset, zeros, symbols);
    DenseMap<AffineExpr, BoundedExpression> cache;
    auto bounded = localOrigin ? boundedOrigin(localOrigin, region.symbols, cache).expression : AffineExpr{};
    auto origin = mlir::pto::detail::substitute(bounded, {}, occurrenceSymbols);
    auto offset = add(region.byteOffset, negate(localOrigin));
    if (!origin || !offset) {
        return false;
    }
    SyncAccessRegion local = region;
    // Materialize only the local byte union; preserve the physical base tag on
    // the resulting relations rather than pretending it is an absolute address.
    local.base = {};
    local.byteOffset = simplifyAffineExpr(offset, region.extents.size(), symbols.size());
    for (unsigned i = 0; i < symbols.size(); ++i) {
        if (local.byteOffset.isFunctionOfSymbol(i)) {
            return false;
        }
        for (auto extent : local.extents) {
            if (extent.isFunctionOfSymbol(i)) {
                return false;
            }
        }
    }
    local.symbols.clear();
    SmallVector<SyncStorageCell> pieces;
    if (!mlir::pto::detail::materializeRegion(local, effect.memory->scope, pieces)) {
        return false;
    }
    for (const auto& piece : pieces) {
        emitRange(builder, siteId, effect, piece, origin, region.base);
    }
    return true;
}

}
void extractAccesses(ProgramBuilder& builder, const SyncInput& input, const SyncStorageEffects& effects)
{
    SmallVector<const CompoundInstanceElement*> phases;
    for (const auto& site : builder.output.sites) { phases.push_back(site.phase); }
    if (effects.hasUniformRelationships(phases)) {
        for (uint32_t a = 0; a < builder.output.sites.size(); ++a) {
            for (uint32_t b = a; b < builder.output.sites.size(); ++b) {
                bool conflict = false;
                for (auto x : effects.effectsFor(builder.output.sites[a].phase)) {
                    for (auto y : effects.effectsFor(builder.output.sites[b].phase)) {
                        conflict |= effects.uniformConflict(x, y);
                    }
                }
                if (conflict) { builder.output.uniformConflicts.emplace_back(a, b); }
            }
        }
    }
    for (auto [siteId, site] : llvm::enumerate(builder.output.sites)) {
        if (builder.staticallyEmpty(site)) {
            continue;
        }
        for (auto id : effects.effectsFor(site.phase)) {
            if (llvm::is_contained(builder.output.extraction.dischargedEffects, id)) { continue; }
            const auto& effect = effects.effects()[id];
            if (effect.rangesMaterialized) {
                for (const auto& range : effect.ranges) {
                    emitRange(builder, siteId, effect, range, getAffineConstantExpr(0, builder.context));
                }
                continue;
            }
            for (const auto& region : effect.regions) {
                if (!symbolicRegion(builder, siteId, effect, region)) {
                    builder.output.extraction.note(RecognitionIssue::SymbolicGeometry, site.phase->elementOp);
                }
            }
        }
    }
}
std::optional<ArithmeticProgram> normalizeFiniteDemandAccesses(
    const ArithmeticProgram& program, uint64_t* attemptedFragments)
{
    if (attemptedFragments) { *attemptedFragments = 0; }
    if (!program.finiteExpansion || !program.phaseIndex) { return std::nullopt; }
    using Key = std::pair<AddressSpace, Value>;
    llvm::MapVector<Key, AffineExpr> translations;
    DenseMap<std::size_t, const FiniteAccessRecipe*> recipes;
    for (const auto& recipe : program.finiteAccessRecipes) {
        const bool valid = recipe.relation < program.primitives.relations.size();
        if (!valid) { return std::nullopt; }
        const bool inserted = recipes.try_emplace(recipe.relation, &recipe).second;
        if (!inserted) { return std::nullopt; }
    }
    for (auto [id, relation] : llvm::enumerate(program.primitives.relations)) {
        const bool access = relation.kind == PrimitiveKind::Reads || relation.kind == PrimitiveKind::Writes;
        const bool relevant = access && !relation.pieces.empty() && relation.storageSpace.has_value();
        if (!relevant) { continue; }
        const Key key{*relation.storageSpace, relation.storageBase};
        const auto found = recipes.find(id);
        auto translation = found == recipes.end() ? AffineExpr{} : found->second->translation;
        const bool valid = translation && !relation.sourceDimensions && !relation.targetDimensions &&
            relation.dimensions == 1 && !found->second->rows.empty();
        if (!valid) { translation = {}; }
        auto [entry, inserted] = translations.try_emplace(key, translation);
        if (!inserted && entry->second != translation) { entry->second = {}; }
    }
    const bool useful = llvm::any_of(translations, [](const auto& entry) {
        return entry.second && !isa<AffineConstantExpr>(entry.second);
    });
    if (!useful) { return std::nullopt; }
    auto normalized = program;
    normalized.expandedFragments = 0;
    // These are representation bounds, as in finite expansion, never a new
    // arithmetic class selected from this input's observed coefficients.
    const ArithmeticLimits limits{UINT_MAX, UINT_MAX, program.primitives.period, UINT64_MAX};
    ProgramBuilder builder{normalized, limits, normalized.context.function.getContext(),
                           *program.phaseIndex, DenseMap<Value, unsigned>(), {}};
    for (const auto& [id, recipe] : recipes) {
        auto& relation = normalized.primitives.relations[id];
        if (!relation.storageSpace) { continue; }
        const auto translation = translations[{*relation.storageSpace, relation.storageBase}];
        if (!translation || isa<AffineConstantExpr>(translation)) { continue; }
        relation.pieces.clear();
        SmallVector<AffineExpr> symbols;
        for (unsigned i = 0; i < normalized.parameters.size(); ++i) {
            symbols.push_back(getAffineSymbolExpr(i, builder.context));
        }
        const auto byte = add(getAffineDimExpr(0, builder.context), translation);
        for (const auto& rows : recipe->rows) {
            SmallVector<AffineExpr> shifted;
            for (auto row : rows) { shifted.push_back(mlir::pto::detail::substitute(row, {byte}, symbols)); }
            builder.emit(relation, shifted);
        }
    }
    if (attemptedFragments) { *attemptedFragments = normalized.expandedFragments; }
    normalized.recognition = recognizeArithmetic(normalized.primitives, limits);
    const bool accepted = normalized.extraction.state == RecognitionState::Applicable &&
        normalized.recognition.state == RecognitionState::Applicable;
    return accepted ? std::optional<ArithmeticProgram>(std::move(normalized)) : std::nullopt;
}
} // namespace mlir::pto::frontiersynch::detail
