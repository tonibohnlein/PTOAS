// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Validate byte ownership by exact map reconstruction and finite local ranges.
#include "RepeatedStorageInternal.h"
#include "RepeatedStorageProjection.h"
#include "DisjointTranslations.h"
#include "PhaseNormalization.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "../InsertSync/SyncRegionArithmetic.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
namespace mlir::pto::frontiersynch {
namespace {
using State = RepeatedStorage::State;
using Id = RegionExpressions::Id;
namespace geometry = mlir::pto::detail;
std::optional<int64_t> integer(Value value)
{
    APInt number;
    if (!matchPattern(value, m_ConstantInt(&number)) || !number.isSignedIntN(64)) { return std::nullopt; }
    return number.getSExtValue();
}
bool invariant(Value value, scf::ForOp loop)
{
    if (!value) { return false; }
    auto* owner = value.getDefiningOp();
    if (auto argument = dyn_cast<BlockArgument>(value)) { owner = argument.getOwner()->getParentOp(); }
    return owner && owner != loop && !loop->isProperAncestor(owner);
}
// Owner inversion uses natural byte offsets. Valid pointers alone do not prove
// this premise: an interior canonical base may admit negative relative offsets.
// Prove every origin input nonnegative and every origin operation nonwrapping.
class OriginBounds {
public:
    explicit OriginBounds(Operation* anchor) : layout(DataLayout::closest(anchor)) {}
    std::optional<uint64_t> upper(AffineExpr map, ArrayRef<Value> symbols, unsigned depth = 0)
    {
        if (!map || depth > 64) { return std::nullopt; }
        if (auto constant = dyn_cast<AffineConstantExpr>(map)) {
            return constant.getValue() < 0 ? std::nullopt : std::optional<uint64_t>(constant.getValue());
        }
        if (auto symbol = dyn_cast<AffineSymbolExpr>(map)) {
            if (symbol.getPosition() >= symbols.size()) { return std::nullopt; }
            auto bound = range(symbols[symbol.getPosition()], 0);
            if (!bound || bound->smin().isNegative()) { return std::nullopt; }
            return bound->umax().getZExtValue();
        }
        auto binary = dyn_cast<AffineBinaryOpExpr>(map);
        if (!binary) { return std::nullopt; }
        auto left = upper(binary.getLHS(), symbols, depth + 1);
        auto right = upper(binary.getRHS(), symbols, depth + 1);
        if (!left || !right) { return std::nullopt; }
        if (binary.getKind() == AffineExprKind::Add && *right <= UINT64_MAX - *left) {
            return *left + *right;
        }
        if (binary.getKind() == AffineExprKind::Mul && (!*right || *left <= UINT64_MAX / *right)) {
            return *left * *right;
        }
        auto divisor = dyn_cast<AffineConstantExpr>(binary.getRHS());
        if (divisor && divisor.getValue() > 0) {
            const auto value = uint64_t(divisor.getValue());
            if (binary.getKind() == AffineExprKind::FloorDiv) { return *left / value; }
            if (binary.getKind() == AffineExprKind::Mod) { return std::min(*left, value - 1); }
        }
        return std::nullopt;
    }
private:
    DataLayout layout;
    llvm::DenseMap<Value, ConstantIntRanges> ranges;
    unsigned width(Type type) const
    {
        if (auto integer = dyn_cast<IntegerType>(type)) { return integer.getWidth(); }
        if (!isa<IndexType>(type)) { return 0; }
        auto bits = layout.getTypeSizeInBits(type);
        return bits.isScalable() ? 0 : bits.getFixedValue();
    }
    std::optional<ConstantIntRanges> range(Value value, unsigned depth)
    {
        if (!value || depth > 64) { return std::nullopt; }
        const auto bits = width(value.getType());
        if (!bits || bits > 64) { return std::nullopt; }
        auto found = ranges.find(value);
        if (found != ranges.end()) { return found->second; }
        auto result = ConstantIntRanges::maxRange(bits);
        // Seed unknown before recursion; cyclic SSA cannot manufacture a bound.
        ranges.try_emplace(value, result);
        APInt number;
        if (matchPattern(value, m_ConstantInt(&number))) {
            number = number.sextOrTrunc(bits);
            result = ConstantIntRanges::fromSigned(number, number);
        } else if (auto* operation = value.getDefiningOp()) {
            if (auto infer = dyn_cast<InferIntRangeInterface>(operation)) {
                SmallVector<ConstantIntRanges> operands;
                for (auto operand : operation->getOperands()) {
                    auto known = range(operand, depth + 1);
                    if (!known) { return result; }
                    operands.push_back(*known);
                }
                infer.inferResultRanges(operands, [&](Value output, const ConstantIntRanges& inferred) {
                    if (output == value) { result = inferred; }
                });
            }
        }
        ranges.insert_or_assign(value, result);
        return result;
    }
};
Id scaled(RegionExpressions& e, Id value, uint64_t scale)
{
    auto result = e.constant(0);
    while (scale) {
        if (scale & 1U) { result = e.add(result, value); }
        scale >>= 1U;
        if (scale) { value = e.add(value, value); }
    }
    return result;
}
std::optional<Id> expression(RegionExpressions& e, AffineExpr map, ArrayRef<Value> symbols, unsigned depth = 0)
{
    if (!map || depth > 64) { return std::nullopt; }
    if (auto number = dyn_cast<AffineConstantExpr>(map)) {
        if (number.getValue() < 0) { return std::nullopt; }
        return e.constant(number.getValue());
    }
    if (auto symbol = dyn_cast<AffineSymbolExpr>(map)) {
        if (symbol.getPosition() >= symbols.size()) { return std::nullopt; }
        auto value = symbols[symbol.getPosition()];
        if (!value || !value.getType().isIntOrIndex() || value.getType().isInteger(1) ||
            (isa<IntegerType>(value.getType()) && cast<IntegerType>(value.getType()).getWidth() > 64)) {
            return std::nullopt;
        }
        return e.input(value);
    }
    auto binary = dyn_cast<AffineBinaryOpExpr>(map);
    if (!binary) { return std::nullopt; }
    auto left = expression(e, binary.getLHS(), symbols, depth + 1);
    if (!left) { return std::nullopt; }
    if (binary.getKind() == AffineExprKind::Mul) {
        auto factor = dyn_cast<AffineConstantExpr>(binary.getRHS());
        if (!factor || factor.getValue() < 0) { return std::nullopt; }
        return scaled(e, *left, factor.getValue());
    }
    if (binary.getKind() == AffineExprKind::FloorDiv || binary.getKind() == AffineExprKind::Mod) {
        auto divisor = dyn_cast<AffineConstantExpr>(binary.getRHS());
        if (!divisor || divisor.getValue() <= 0) { return std::nullopt; }
        const auto value = e.constant(uint64_t(divisor.getValue()));
        return binary.getKind() == AffineExprKind::FloorDiv ? e.div(*left, value) : e.rem(*left, value);
    }
    if (binary.getKind() != AffineExprKind::Add) { return std::nullopt; }
    auto right = expression(e, binary.getRHS(), symbols, depth + 1);
    return right ? std::optional<Id>(e.add(*left, *right)) : std::nullopt;
}
struct Context {
    SmallVector<Value> symbols;
    MLIRContext* context;
    AffineExpr map(AffineExpr input, ArrayRef<Value> source, unsigned dimensions)
    {
        SmallVector<AffineExpr> dims, replacements;
        for (unsigned i = 0; i < dimensions; ++i) { dims.push_back(getAffineDimExpr(i, context)); }
        for (auto symbol : source) {
            auto found = llvm::find(symbols, symbol);
            if (found == symbols.end()) { symbols.push_back(symbol); found = std::prev(symbols.end()); }
            replacements.push_back(getAffineSymbolExpr(found - symbols.begin(), context));
        }
        return geometry::substitute(input, dims, replacements);
    }
};
std::optional<SmallVector<SyncStorageCell>> localRanges(const State& state, const State::Family& family,
                                                     const SyncAccessRegion& physical)
{
    if (physical.base != family.spec.origin.base || !physical.byteOffset || !physical.elementBytes) {
        return std::nullopt;
    }
    auto loop = state.loop;
    auto lower = integer(loop.getLowerBound()), step = integer(loop.getStep());
    if (!lower || *lower < 0 || !step || *step <= 0) { return std::nullopt; }
    Context context{{}, loop.getContext()};
    auto map = context.map(physical.byteOffset, physical.symbols, physical.extents.size());
    auto origin = context.map(family.spec.origin.byteOffset, family.spec.origin.symbols, 0);
    if (!map || !origin) { return std::nullopt; }
    auto residual = geometry::checkedAdd(map, geometry::checkedMul(origin, getAffineConstantExpr(-1, context.context)));
    if (!residual) { return std::nullopt; }
    auto varying = llvm::find(context.symbols, loop.getInductionVar());
    if (varying != context.symbols.end()) {
        auto split = geometry::splitTranslation(residual, physical.extents.size(), context.symbols.size(),
                                               varying - context.symbols.begin());
        int64_t stride = 0;
        if (!split || llvm::MulOverflow(split->second, *step, stride) || stride < 0 ||
            uint64_t(stride) != family.spec.stride) { return std::nullopt; }
        auto shift = geometry::checkedMul(getAffineConstantExpr(split->second, context.context),
                                          getAffineConstantExpr(*lower, context.context));
        residual = geometry::checkedAdd(split->first, shift);
    } else if (family.spec.stride) { return std::nullopt; }
    if (!residual) { return std::nullopt; }
    residual = simplifyAffineExpr(residual, physical.extents.size(), context.symbols.size());
    for (unsigned i = 0; i < context.symbols.size(); ++i) {
        if (residual.isFunctionOfSymbol(i)) { return std::nullopt; }
    }
    SyncAccessRegion local;
    local.byteOffset = residual;
    local.elementBytes = physical.elementBytes;
    for (auto extent : physical.extents) {
        auto constant = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(extent, 0, physical.symbols.size()));
        if (!constant || constant.getValue() < 0) { return std::nullopt; }
        local.extents.push_back(constant);
    }
    SmallVector<SyncStorageCell> ranges;
    if (!geometry::materializeRegion(local, family.spec.space, ranges)) { return std::nullopt; }
    return ranges;
}
bool addFamily(State& state, const RepeatedStorageFamily& spec, llvm::DenseSet<std::size_t>& assigned)
{
    if (!spec.origin.byteOffset || !spec.origin.extents.empty() || spec.effects.empty() ||
        spec.stride > INT64_MAX || (spec.kind == RepeatedStorageKind::VisitOwned ? !spec.stride : spec.stride != 0) ||
        (spec.origin.base && (spec.space != AddressSpace::GM || !isCanonicalGMBase(spec.origin.base))) ||
        !llvm::all_of(spec.origin.symbols, [&](Value value) { return invariant(value, state.loop); })) { return false; }
    State::Family family;
    family.spec = spec;
    OriginBounds bounds(state.loop);
    // Factor exact common coefficients before lowering. This exposes scaled
    // remainders to MLIR's affine simplifier without evaluating a negative
    // subtraction in the natural-number owner circuit. Preflight coefficient
    // folding first; failure keeps the original spelling and all obligations.
    auto normalizedOrigin = spec.origin.byteOffset;
    if (geometry::magnitude(normalizedOrigin)) {
        const auto divisor = normalizedOrigin.getLargestKnownDivisor();
        if (divisor > 1) {
            auto normalized = geometry::checkedMul(normalizedOrigin.floorDiv(divisor),
                getAffineConstantExpr(divisor, normalizedOrigin.getContext()));
            if (normalized) { normalizedOrigin = normalized; }
        }
    }
    if (!bounds.upper(normalizedOrigin, spec.origin.symbols)) { return false; }
    auto origin = expression(state.expressions(), normalizedOrigin, spec.origin.symbols);
    if (!origin) { return false; }
    family.origin = *origin;
    const auto effects = state.body.accessModel->effects();
    for (auto id : spec.effects) {
        if (id >= effects.size() || !assigned.insert(id).second) { return false; }
        const auto& effect = effects[id];
        auto selected = llvm::find_if(state.body.accessBoundary,
            [id](const auto& access) { return access.effect == id; });
        if (selected == state.body.accessBoundary.end() || !effect.memory || effect.memory->scope != spec.space ||
            effect.regions.empty() || (spec.kind == RepeatedStorageKind::SharedReadOnly &&
                                     effect.mode != SyncAccessMode::Read)) { return false; }
        if (!validRegionalEvent(state.body, selected->first.event) ||
            !validRegionalEvent(state.body, selected->last.event) ||
            !state.expressions().isBoolean(selected->first.present) ||
            !state.expressions().isBoolean(selected->last.present)) { return false; }
        auto projected = detail::projectAccesses(state.body, state.loop, id);
        if (!projected) { return false; }
        for (const auto& projection : *projected) {
            auto ranges = localRanges(state, family, projection.region);
            if (!ranges) { return false; }
            const auto& access = projection.boundary;
            if (!validRegionalEvent(state.body, access.first.event) ||
                !validRegionalEvent(state.body, access.last.event) ||
                state.body.anchors[access.first.event.type].phase != effect.phase ||
                state.body.anchors[access.last.event.type].phase != effect.phase) { return false; }
            for (auto range : *ranges) {
                if (range.begin > range.end) { return false; }
                uint64_t extent = range.end;
                if (projection.inner) {
                    const auto& inner = *projection.inner;
                    if (!inner.stride || !inner.maxTrips || range.end - range.begin > inner.stride ||
                        inner.maxTrips - 1 > (UINT64_MAX - extent) / inner.stride) { return false; }
                    extent += (inner.maxTrips - 1) * inner.stride;
                    // Dynamic runs use one disjoint reservation per outer visit.
                    // Wider strided reservations need a separate inverse proof.
                    if (spec.kind == RepeatedStorageKind::VisitOwned && extent > spec.stride) { return false; }
                }
                family.extent = std::max(family.extent, extent);
                family.pieces.push_back({range.begin, range.end, access, effect.mode,
                                        static_cast<uint32_t>(effect.phase->kPipeValue), projection.inner});
            }
            state.body.cost.physicalFragments += ranges->size();
        }
        state.effectIds.push_back(id);
    }
    if (spec.kind == RepeatedStorageKind::VisitOwned) {
        SmallVector<SyncStorageCell> ranges;
        for (const auto& piece : family.pieces) {
            const auto end = piece.inner ? piece.end + (piece.inner->maxTrips - 1) * piece.inner->stride : piece.end;
            ranges.push_back({spec.space, piece.begin, end});
        }
        auto trips = state.expressions().constantValue(state.trips).value_or(UINT64_MAX);
        if (!detail::disjointTranslations(ranges, APInt(128, spec.stride), trips)) { return false; }
    }
    state.families.push_back(std::move(family));
    return true;
}
std::optional<int64_t> originDifference(const SyncAccessRegion& a, const SyncAccessRegion& b)
{
    Context context{{}, a.byteOffset.getContext()};
    auto left = context.map(a.byteOffset, a.symbols, 0), right = context.map(b.byteOffset, b.symbols, 0);
    auto delta = geometry::checkedAdd(left, geometry::checkedMul(right, getAffineConstantExpr(-1, context.context)));
    auto constant = delta ? dyn_cast<AffineConstantExpr>(simplifyAffineExpr(delta, 0, context.symbols.size())) :
                           AffineConstantExpr();
    return constant ? std::optional<int64_t>(constant.getValue()) : std::nullopt;
}
std::optional<uint64_t> span(const State& state, const State::Family& family)
{
    if (family.spec.kind == RepeatedStorageKind::SharedReadOnly) { return family.extent; }
    auto trips = state.expressions().constantValue(state.trips);
    if (!trips) { return std::nullopt; }
    if (!*trips) { return 0; }
    auto extent = std::max(family.extent, family.spec.stride);
    if (family.spec.stride && *trips - 1 > (UINT64_MAX - extent) / family.spec.stride) { return std::nullopt; }
    return (*trips - 1) * family.spec.stride + extent;
}
bool separated(const State& state, const State::Family& a, const State::Family& b)
{
    if (a.pieces.empty() || b.pieces.empty() || a.spec.space != b.spec.space) { return true; }
    if (a.spec.origin.base != b.spec.origin.base) {
        SmallVector<SyncStorageCell> domains{{a.spec.space, 0, 1, a.spec.origin.base},
                                             {b.spec.space, 0, 1, b.spec.origin.base}};
        return storageBasesAreComparable(domains, state.body.gmAliasPolicy);
    }
    auto difference = originDifference(a.spec.origin, b.spec.origin);
    if (!difference) { return false; }
    auto firstSpan = span(state, a), secondSpan = span(state, b);
    if (*difference >= 0 && secondSpan && uint64_t(*difference) >= *secondSpan) { return true; }
    if (*difference <= 0 && *difference != INT64_MIN && firstSpan && uint64_t(-*difference) >= *firstSpan) {
        return true;
    }
    return false;
}
bool separation(State& state)
{
    for (std::size_t i = 0; i < state.families.size(); ++i) {
        const auto& family = state.families[i];
        for (std::size_t j = 0; j < i; ++j) {
            if (family.spec.kind == RepeatedStorageKind::SharedReadOnly &&
                state.families[j].spec.kind == RepeatedStorageKind::SharedReadOnly) { continue; }
            if (!separated(state, family, state.families[j])) { return false; }
        }
        for (const auto& profile : state.persistent) {
            if (family.spec.kind == RepeatedStorageKind::SharedReadOnly && profile.readOnly) { continue; }
            const auto& cell = profile.reservation;
            State::Family persistent;
            persistent.spec.space = cell.space;
            persistent.spec.kind = RepeatedStorageKind::SharedReadOnly;
            persistent.spec.origin.base = cell.base;
            persistent.spec.origin.byteOffset = getAffineConstantExpr(cell.begin, state.loop.getContext());
            persistent.extent = cell.end - cell.begin;
            if (persistent.extent) { persistent.pieces.emplace_back(); }
            if (!separated(state, family, persistent)) { return false; }
        }
        for (const auto& cell : state.body.storageBoundary) {
            if (family.spec.kind == RepeatedStorageKind::SharedReadOnly &&
                cell.firstWriters.empty() && cell.lastWriters.empty()) { continue; }
            State::Family persistent;
            persistent.spec.space = cell.cell.space;
            persistent.spec.kind = RepeatedStorageKind::SharedReadOnly;
            persistent.spec.origin.base = cell.cell.base;
            if (cell.cell.begin > INT64_MAX || cell.cell.end < cell.cell.begin) { return false; }
            persistent.spec.origin.byteOffset = getAffineConstantExpr(cell.cell.begin, state.loop.getContext());
            persistent.extent = cell.cell.end - cell.cell.begin;
            if (persistent.extent) { persistent.pieces.emplace_back(); }
            if (!separated(state, family, persistent)) { return false; }
        }
    }
    return true;
}
// Alternatives can use the same reservation in the same visit. Only distinct
// visits must be disjoint, so prove the union rather than requiring disjoint
// families in the two alternative bodies.
bool separateAlternativeFamilies(const State& state, const State::Family& a, const State::Family& b)
{
    if (a.spec.kind == RepeatedStorageKind::SharedReadOnly &&
        b.spec.kind == RepeatedStorageKind::SharedReadOnly) { return true; }
    if (a.spec.kind == RepeatedStorageKind::VisitOwned && b.spec.kind == RepeatedStorageKind::VisitOwned &&
        a.spec.space == b.spec.space && a.spec.origin.base == b.spec.origin.base &&
        a.spec.stride == b.spec.stride && originDifference(a.spec.origin, b.spec.origin) == 0) {
        SmallVector<SyncStorageCell> ranges;
        for (const auto* family : {&a, &b}) {
            for (const auto& piece : family->pieces) {
                const auto end = piece.inner ? piece.end + (piece.inner->maxTrips - 1) * piece.inner->stride :
                                              piece.end;
                ranges.push_back({a.spec.space, piece.begin, end});
            }
        }
        const auto trips = state.expressions().constantValue(state.trips).value_or(UINT64_MAX);
        return detail::disjointTranslations(ranges, APInt(128, a.spec.stride), trips);
    }
    return separated(state, a, b);
}
bool separateTypeResiduals(const State& owner, const State& other)
{
    llvm::DenseSet<std::size_t> omitted(other.effectIds.begin(), other.effectIds.end());
    for (const auto& family : owner.families) {
        for (const auto& cell : other.body.storageBoundary) {
            if (family.spec.kind == RepeatedStorageKind::SharedReadOnly &&
                cell.firstWriters.empty() && cell.lastWriters.empty()) { continue; }
            State::Family persistent;
            persistent.spec.space = cell.cell.space;
            persistent.spec.kind = RepeatedStorageKind::SharedReadOnly;
            persistent.spec.origin.base = cell.cell.base;
            if (cell.cell.begin > INT64_MAX || cell.cell.end < cell.cell.begin) { return false; }
            persistent.spec.origin.byteOffset = getAffineConstantExpr(cell.cell.begin, owner.loop->getContext());
            persistent.extent = cell.cell.end - cell.cell.begin;
            if (persistent.extent) { persistent.pieces.emplace_back(); }
            if (!separated(owner, family, persistent)) { return false; }
        }
        for (const auto& access : other.body.accessBoundary) {
            if (omitted.count(access.effect) || access.representedByCells) { continue; }
            for (auto effect : family.spec.effects) {
                if (owner.body.accessModel->mayOverlap(effect, access.effect)) { return false; }
            }
        }
    }
    return true;
}
bool independent(Value value, scf::ForOp loop, DenseMap<Value, bool>& memo)
{
    if (!value) { return true; }
    if (auto found = memo.find(value); found != memo.end()) { return found->second; }
    memo[value] = false;
    if (invariant(value, loop)) { return memo[value] = true; }
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        auto inner = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
        if (!inner || inner == loop || argument != inner.getInductionVar()) { return false; }
        return memo[value] = independent(inner.getLowerBound(), loop, memo) &&
            independent(inner.getUpperBound(), loop, memo) && independent(inner.getStep(), loop, memo);
    }
    auto* definition = value.getDefiningOp();
    if (!definition || definition->getNumRegions() || !isMemoryEffectFree(definition) ||
        !isSpeculatable(definition)) { return false; }
    auto dialect = definition->getName().getDialectNamespace();
    if (dialect != "arith" && dialect != "index") { return false; }
    return memo[value] = llvm::all_of(definition->getOperands(),
        [&](Value operand) { return independent(operand, loop, memo); });
}
bool invariantEffect(const SyncStorageEffect& effect, scf::ForOp loop)
{
    DenseMap<Value, bool> memo;
    if (effect.selection && !independent(effect.selection->selector, loop, memo)) { return false; }
    if (effect.regions.empty()) { return independent(effect.memory->baseBuffer, loop, memo); }
    for (const auto& region : effect.regions) {
        if (!independent(region.base, loop, memo)) { return false; }
        for (unsigned i = 0; i < region.symbols.size(); ++i) {
            const bool used = region.byteOffset.isFunctionOfSymbol(i) || llvm::any_of(region.extents,
                [i](AffineExpr extent) { return extent.isFunctionOfSymbol(i); });
            if (used && !independent(region.symbols[i], loop, memo)) { return false; }
        }
    }
    return true;
}
bool completeEffects(const State& state, const llvm::DenseSet<std::size_t>& assigned,
                     const PhaseIndex* phaseIndex, uint64_t phasePeriod)
{
    for (const auto& access : state.body.accessBoundary) {
        if (access.effect >= state.body.accessModel->effects().size() ||
            !state.body.accessModel->effects()[access.effect].memory) { return false; }
    }
    for (const auto& anchor : state.body.anchors) {
        for (auto id : state.body.accessModel->effectsFor(anchor.phase)) {
            auto found = llvm::find_if(state.body.accessBoundary,
                [id](const auto& access) { return access.effect == id; });
            if (found == state.body.accessBoundary.end()) { return false; }
            if (assigned.count(id) || llvm::any_of(state.persistent, [id](const auto& profile) {
                    return llvm::is_contained(profile.effects, id);
                })) { continue; }
            const auto& effect = state.body.accessModel->effects()[id];
            if (!invariantEffect(effect, state.loop)) {
                if (!found->representedByCells || !phaseIndex || phasePeriod <= 1 || effect.regions.empty()) {
                    return false;
                }
                PhaseNormalization normalizer(state.loop, *phaseIndex, state.expressions());
                if ((effect.selection && !normalizer.periodic(effect.selection->selector, phasePeriod)) ||
                    llvm::any_of(effect.regions, [&](const auto& region) {
                        return !normalizer.periodic(region, phasePeriod);
                    })) { return false; }
            }
            if (found->representedByCells) { continue; }
            // The projected selector API exports classified residual families
            // only. An unclassified invariant access cannot silently disappear.
            if (!state.persistent.empty()) { return false; }
            for (auto owned : assigned) {
                if (state.body.accessModel->mayOverlap(id, owned)) { return false; }
            }
        }
    }
    return true;
}
} // namespace
static RepeatedStorageResult buildStorage(const RegionalAnalysis& body, scf::ForOp loop,
    RegionExpressions::Id trips, ArrayRef<RepeatedStorageFamily> families,
    const PhaseIndex* phaseIndex, uint64_t phasePeriod, ArrayRef<RepeatedPersistentStorage> persistent)
{
    RepeatedStorageResult result;
    if (!phasePeriod || phasePeriod > maxRegionalSlotVisits || !loop || !body.expressions || !body.accessModel ||
        trips >= body.expressions->size() ||
        body.expressions->isBoolean(trips) ||
        !body.capabilities.exactSelectors || !body.capabilities.exactQueries || !body.presence || !body.reachability ||
        llvm::any_of(body.anchors, [&](const auto& anchor) {
            return !anchor.phase || !anchor.phase->elementOp || !loop->isProperAncestor(anchor.phase->elementOp);
        })) {
        result.error = "evolving storage requires an exact body and an unclassified shared access model";
        return result;
    }
    auto state = std::make_shared<State>();
    state->body = body; state->loop = loop; state->trips = trips;
    state->persistent.assign(persistent.begin(), persistent.end());
    llvm::DenseSet<std::size_t> persistentIds;
    for (const auto& profile : persistent) {
        if (profile.reservation.begin >= profile.reservation.end || profile.reservation.end > INT64_MAX ||
            profile.effects.empty()) {
            result.error = "persistent storage profile has invalid reservation or effect identity"; return result;
        }
        for (auto id : profile.effects) {
            if (id >= body.accessModel->effects().size() || !persistentIds.insert(id).second ||
                llvm::none_of(body.anchors, [&](const auto& anchor) {
                    return anchor.phase == body.accessModel->effects()[id].phase;
                }) ||
                !body.accessModel->effects()[id].memory ||
                body.accessModel->effects()[id].memory->scope != profile.reservation.space ||
                (profile.readOnly && body.accessModel->effects()[id].mode == SyncAccessMode::Write)) {
                result.error = "persistent storage profile has invalid reservation or effect identity"; return result;
            }
        }
    }
    llvm::DenseSet<std::size_t> assigned;
    for (const auto& family : families) {
        if (llvm::any_of(family.effects, [&](auto id) { return persistentIds.count(id); }) ||
            !addFamily(*state, family, assigned)) {
            result.error = "evolving storage lacks complete invariant local byte maps or effect selectors";
            return result;
        }
    }
    if (llvm::any_of(body.symbolicStorageEffects, [&](std::size_t id) {
            return !assigned.count(id) && !persistentIds.count(id);
        }) ||
        !completeEffects(*state, assigned, phaseIndex, phasePeriod)) {
        result.error = "evolving storage has an unexported effect or an unclassified overlapping residual access";
        return result;
    }
    if (!separation(*state)) {
        result.error = "evolving storage reservations are not proved physically disjoint";
        return result;
    }
    result.storage = std::make_shared<RepeatedStorage>(std::move(state));
    return result;
}
RepeatedStorageResult buildRepeatedStorage(const RegionalAnalysis& body, scf::ForOp loop,
    RegionExpressions::Id trips, ArrayRef<RepeatedStorageFamily> families,
    const PhaseIndex* phaseIndex, uint64_t phasePeriod)
{
    return buildStorage(body, loop, trips, families, phaseIndex, phasePeriod, {});
}
RepeatedStorageResult buildRepeatedStorageWithPersistent(const RegionalAnalysis& body, scf::ForOp loop,
    RegionExpressions::Id trips, ArrayRef<RepeatedStorageFamily> families,
    ArrayRef<RepeatedPersistentStorage> persistent)
{
    return buildStorage(body, loop, trips, families, nullptr, 1, persistent);
}
RepeatedStorageTypesResult recognizeRepeatedStorageTypes(ArrayRef<RegionalAnalysis> bodies,
    scf::ForOp loop, RegionExpressions::Id trips, const PhaseIndex* phaseIndex, uint64_t phasePeriod)
{
    RepeatedStorageTypesResult out;
    if (bodies.empty() || !loop) { out.error = "finite visit storage has no original type bodies"; return out; }
    std::vector<std::shared_ptr<State>> states;
    SmallVector<const CompoundInstanceElement*> phases;
    for (const auto& body : bodies) {
        if (!body.expressions || body.expressions != bodies.front().expressions ||
            !body.accessModel || body.accessModel != bodies.front().accessModel ||
            body.gmAliasPolicy != bodies.front().gmAliasPolicy || !body.deferredAccessBoundary.empty()) {
            out.error = "finite visit storage requires common modeled input and complete access boundaries";
            return out;
        }
        auto recovered = recognizeRepeatedStorage(body, loop, trips, phaseIndex, phasePeriod);
        std::shared_ptr<State> state;
        if (recovered.storage) { state = recovered.storage->state; }
        else {
            // No omitted family is needed for an already finite complete type.
            // Reuse the same invariance and effect coverage checks as ownership.
            auto checked = buildRepeatedStorage(body, loop, trips, {}, phaseIndex, phasePeriod);
            if (!checked.storage) { out.error = recovered.error; return out; }
            state = checked.storage->state;
        }
        if (llvm::any_of(body.accessBoundary, [&](const auto& access) {
                return !access.representedByCells && !llvm::is_contained(state->effectIds, access.effect);
            })) {
            out.error = "finite visit projection cannot omit an unclassified residual effect";
            return out;
        }
        for (const auto& anchor : body.anchors) { phases.push_back(anchor.phase); }
        out.storage.push_back(std::move(recovered.storage));
        states.push_back(std::move(state));
    }
    if (bodies.front().accessModel->hasUniformRelationships(phases)) {
        out.error = "finite visit storage projection needs explicit uniform relationship coverage";
        return out;
    }
    for (std::size_t i = 0; i < states.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (!separateTypeResiduals(*states[i], *states[j]) ||
                !separateTypeResiduals(*states[j], *states[i])) {
                out.error = "visit-owned storage can overlap another type's persistent or residual storage";
                return out;
            }
            for (const auto& a : states[i]->families) {
                for (const auto& b : states[j]->families) {
                    if (!separateAlternativeFamilies(*states[i], a, b)) {
                        out.error = "cross-type storage is not proved disjoint across distinct visits";
                        return out;
                    }
                }
            }
        }
    }
    return out;
}
} // namespace mlir::pto::frontiersynch
