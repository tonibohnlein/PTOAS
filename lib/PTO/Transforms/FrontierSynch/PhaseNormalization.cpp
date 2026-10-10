// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PhaseNormalization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/MathExtras.h"
#include <array>
namespace mlir::pto::frontiersynch {
bool PhaseNormalization::independent(Value value)
{
    if (!value) { return false; }
    if (auto found = independence.find(value); found != independence.end()) { return found->second; }
    independence[value] = false;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        auto* owner = argument.getOwner()->getParentOp();
        if (owner == outer) { return false; }
        if (!owner || !outer->isProperAncestor(owner)) { return independence[value] = true; }
        auto loop = dyn_cast<scf::ForOp>(owner);
        return independence[value] = loop &&
            (argument == loop.getInductionVar() || static_cast<bool>(index.carriedOrdinal(argument)));
    }
    auto* op = value.getDefiningOp();
    if (!op) { return false; }
    if (!outer->isProperAncestor(op)) { return independence[value] = true; }
    auto dialect = op->getName().getDialectNamespace();
    if (op->getNumRegions() || !index.phasesFor(op).empty() ||
        (dialect != "arith" && dialect != "index") || !isMemoryEffectFree(op) || !isSpeculatable(op)) { return false; }
    return independence[value] = llvm::all_of(op->getOperands(), [&](Value operand) { return independent(operand); });
}
std::optional<uint64_t> PhaseNormalization::modulus(Value value)
{
    auto* op = value ? value.getDefiningOp() : nullptr;
    if (!op || op->getNumOperands() != 2) { return std::nullopt; }
    APInt number;
    if (isa<arith::RemUIOp, arith::RemSIOp>(op)) {
        if (matchPattern(op->getOperand(1), m_ConstantInt(&number)) && number.isSignedIntN(64) &&
            number.getSExtValue() > 0) { return number.getZExtValue(); }
        return std::nullopt;
    }
    if (!isa<arith::AndIOp>(op) || !value.getType().isIndex()) { return std::nullopt; }
    if (!matchPattern(op->getOperand(1), m_ConstantInt(&number)) &&
        !matchPattern(op->getOperand(0), m_ConstantInt(&number))) { return std::nullopt; }
    if (!number.isSignedIntN(64) || number.isNegative() || number.getZExtValue() == INT64_MAX) { return std::nullopt; }
    const auto candidate = number.getZExtValue() + 1;
    return llvm::isPowerOf2_64(candidate) ? std::optional<uint64_t>(candidate) : std::nullopt;
}
bool PhaseNormalization::periodic(Value value, uint64_t period)
{
    if (!value) { return true; }
    auto& cached = periodicValues[value];
    if (auto found = cached.find(period); found != cached.end()) { return found->second; }
    cached.emplace(period, false);
    auto compute = [&]() -> bool {
        if (!value || independent(value)) { return true; }
        if (auto recurrence = index.carriedOrdinal(value)) {
            return period && period % recurrence->period() == 0;
        }
        auto* op = value.getDefiningOp();
        if (!op || !index.phasesFor(op).empty()) { return false; }
        if (auto divisor = modulus(value)) {
            return period % *divisor == 0 && residue(value, 0, period).has_value();
        }
        if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::IndexCastOp, arith::IndexCastUIOp,
                 arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) { return false; }
        return llvm::all_of(op->getOperands(), [&](Value operand) { return periodic(operand, period); });
    };
    auto result = compute();
    periodicValues[value][period] = result;
    return result;
}
bool PhaseNormalization::periodic(const SyncAccessRegion& region, uint64_t period)
{
    if (region.base && !independent(region.base)) { return false; }
    DenseMap<AffineExpr, std::array<std::optional<bool>, 2>> cache;
    std::function<bool(AffineExpr, bool)> check = [&](AffineExpr expression, bool reduced) {
        if (!expression) { return false; }
        if (auto known = cache[expression][reduced]) { return *known; }
        auto compute = [&]() -> bool {
            if (auto symbol = dyn_cast<AffineSymbolExpr>(expression)) {
                if (symbol.getPosition() >= region.symbols.size()) { return false; }
                auto value = region.symbols[symbol.getPosition()];
                return periodic(value, period) || (reduced && residue(value, 0, period).has_value());
            }
            auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
            if (!binary) { return true; }
            // Layout floor/ceiling divisions are harmless when both complete
            // operands are invariant. Only an evolving congruence needs the
            // restricted modular algebra below.
            if (check(binary.getLHS(), false) && check(binary.getRHS(), false)) { return true; }
            if (binary.getKind() == AffineExprKind::Mod) {
                auto divisor = dyn_cast<AffineConstantExpr>(binary.getRHS());
                return divisor && divisor.getValue() > 0 && period % divisor.getValue() == 0 &&
                    check(binary.getLHS(), true);
            }
            if (binary.getKind() != AffineExprKind::Add && binary.getKind() != AffineExprKind::Mul) { return false; }
            return check(binary.getLHS(), reduced) && check(binary.getRHS(), reduced);
        };
        auto result = compute();
        cache[expression][reduced] = result;
        return result;
    };
    if (!check(region.byteOffset, false)) { return false; }
    return llvm::all_of(region.extents, [&](AffineExpr extent) { return check(extent, false); });
}
std::optional<RegionExpressions::Id> PhaseNormalization::atPhase(Value value, uint64_t phase, uint64_t period)
{
    const auto key = std::make_pair(phase, period);
    auto& cached = phaseValues[value];
    if (auto found = cached.find(key); found != cached.end()) { return found->second; }
    cached.emplace(key, std::nullopt);
    auto compute = [&]() -> std::optional<RegionExpressions::Id> {
        using Id = RegionExpressions::Id;
        if (independent(value)) { return arena.input(value); }
        if (auto recurrence = index.carriedOrdinal(value)) {
            return arena.constant(recurrence->atOrdinal(phase));
        }
        if (value == outer.getInductionVar()) {
            APInt lower, step;
            if (!matchPattern(outer.getLowerBound(), m_ConstantInt(&lower)) ||
                !matchPattern(outer.getStep(), m_ConstantInt(&step)) || !lower.isSignedIntN(64) ||
                !step.isSignedIntN(64) || lower.isNegative() || step.getSExtValue() <= 0) { return std::nullopt; }
            auto number = lower.zextOrTrunc(128) + step.zextOrTrunc(128) * APInt(128, phase);
            if (!number.isSignedIntN(64)) { return std::nullopt; }
            return arena.constant(number.getZExtValue());
        }
        auto* op = value.getDefiningOp();
        if (!op || !index.phasesFor(op).empty()) { return std::nullopt; }
        if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op)) {
            auto width = [&](Type type) {
                auto bits = DataLayout::closest(outer).getTypeSizeInBits(type);
                return !bits.isScalable() && bits.getFixedValue() == 64;
            };
            if (width(value.getType()) && width(op->getOperand(0).getType())) {
                return atPhase(op->getOperand(0), phase, period);
            }
            return std::nullopt;
        }
        if (op->getNumOperands() != 2) { return std::nullopt; }
        // A remainder has its complete value in [0, divisor), so the congruence
        // certificate is sufficient here. It is not sufficient for a raw address.
        if (modulus(value)) { return residue(value, phase, period); }
        auto left = atPhase(op->getOperand(0), phase, period), right = atPhase(op->getOperand(1), phase, period);
        if (!left || !right) { return std::nullopt; }
        if (isa<arith::AddIOp>(op)) { return arena.add(*left, *right); }
        if (isa<arith::SubIOp>(op)) { return arena.sub(*left, *right); }
        if (isa<arith::MinSIOp, arith::MaxSIOp>(op)) {
            const auto bits = DataLayout::closest(outer).getTypeSizeInBits(value.getType());
            const bool signed64 = !bits.isScalable() && bits.getFixedValue() == 64;
            if (!signed64) { return std::nullopt; }
            auto less = arena.slt(*left, *right);
            return isa<arith::MinSIOp>(op) ? arena.select(less, *left, *right) : arena.select(less, *right, *left);
        }
        if (!isa<arith::MulIOp>(op)) { return std::nullopt; }
        auto factor = arena.constantValue(*right);
        if (!factor) { factor = arena.constantValue(*left); std::swap(left, right); }
        if (!factor) { return std::nullopt; }
        Id result = arena.constant(0), term = *left;
        for (uint64_t remaining = *factor; remaining; remaining >>= 1) {
            if (remaining & 1) { result = arena.add(result, term); }
            if (remaining > 1) { term = arena.add(term, term); }
        }
        return result;
    };
    auto result = compute();
    phaseValues[value][key] = result;
    return result;
}
std::optional<RegionExpressions::Id> PhaseNormalization::residue(Value value, uint64_t phase, uint64_t period)
{
    using Id = RegionExpressions::Id;
    if (!value || !period || !value.getType().isIndex()) { return std::nullopt; }
    const auto key = std::make_pair(phase, period);
    auto& values = residues[value];
    if (auto found = values.find(key); found != values.end()) { return found->second; }
    values.emplace(key, std::nullopt);
    auto save = [&](Id result) -> std::optional<Id> {
        if (result == RegionExpressions::invalid) { return std::nullopt; }
        return residues[value][key] = result;
    };
    if (independent(value)) { return save(arena.rem(arena.input(value), arena.constant(period))); }
    if (index.carriedOrdinal(value)) {
        if (!periodic(value, period)) { return std::nullopt; }
        auto concrete = atPhase(value, phase, period);
        return concrete ? save(arena.rem(*concrete, arena.constant(period))) : std::nullopt;
    }
    if (value == outer.getInductionVar()) {
        APInt lower, step;
        if (!matchPattern(outer.getLowerBound(), m_ConstantInt(&lower)) ||
            !matchPattern(outer.getStep(), m_ConstantInt(&step)) || !lower.isSignedIntN(64) ||
            !step.isSignedIntN(64) || lower.isNegative() || step.getSExtValue() <= 0) { return std::nullopt; }
        auto number = lower.zextOrTrunc(128) + step.zextOrTrunc(128) * APInt(128, phase);
        return save(arena.constant(number.urem(APInt(128, period)).getZExtValue()));
    }
    auto* op = value.getDefiningOp();
    if (!op || op->getNumOperands() != 2 || !index.phasesFor(op).empty()) { return std::nullopt; }
    if (isa<arith::RemUIOp, arith::RemSIOp>(op)) {
        APInt divisor;
        if (!matchPattern(op->getOperand(1), m_ConstantInt(&divisor)) || !divisor.isSignedIntN(64) ||
            divisor.getSExtValue() <= 0 || period % divisor.getZExtValue()) { return std::nullopt; }
        // Signed remainder is only normalized for the proven nonnegative IV.
        if (isa<arith::RemSIOp>(op) && op->getOperand(0) != outer.getInductionVar()) { return std::nullopt; }
        auto left = residue(op->getOperand(0), phase, period);
        return left ? save(arena.rem(*left, arena.constant(divisor.getZExtValue()))) : std::nullopt;
    }
    if (isa<arith::AndIOp>(op)) {
        auto divisor = modulus(value);
        if (!divisor || period % *divisor) { return std::nullopt; }
        APInt mask;
        auto input = matchPattern(op->getOperand(1), m_ConstantInt(&mask)) ?
            op->getOperand(0) : op->getOperand(1);
        auto normalized = residue(input, phase, period);
        return normalized ? save(arena.rem(*normalized, arena.constant(*divisor))) : std::nullopt;
    }
    // Low-bit residues commute with fixed-width add/multiply. For other bank
    // counts a separate no-overflow proof is required; do not assume integers.
    if (!llvm::isPowerOf2_64(period) || !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) { return std::nullopt; }
    auto left = residue(op->getOperand(0), phase, period), right = residue(op->getOperand(1), phase, period);
    if (!left || !right) { return std::nullopt; }
    if (isa<arith::AddIOp>(op)) { return save(arena.rem(arena.add(*left, *right), arena.constant(period))); }
    if (isa<arith::SubIOp>(op)) { return save(arena.rem(arena.sub(*left, *right), arena.constant(period))); }
    // Both residues are below the supported explicit phase count. Multiplication by a
    // symbolic residue is a bounded arithmetic circuit, never a trip expansion.
    Id product = arena.constant(0);
    for (uint64_t bit = 1; bit < period; bit <<= 1) {
        auto active = arena.eq(arena.rem(arena.div(*right, arena.constant(bit)), arena.constant(2)), arena.constant(1));
        Id term = *left;
        for (uint64_t scale = 1; scale < bit; scale <<= 1) { term = arena.add(term, term); }
        product = arena.add(product, arena.select(active, term, arena.constant(0)));
    }
    return save(arena.rem(product, arena.constant(period)));
}
} // namespace mlir::pto::frontiersynch
