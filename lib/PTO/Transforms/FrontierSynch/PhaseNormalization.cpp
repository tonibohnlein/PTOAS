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
#include "llvm/Support/MathExtras.h"
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
        return independence[value] = loop && argument == loop.getInductionVar();
    }
    auto* op = value.getDefiningOp();
    if (!op) { return false; }
    if (!outer->isProperAncestor(op)) { return independence[value] = true; }
    auto dialect = op->getName().getDialectNamespace();
    if (op->getNumRegions() || !index.phasesFor(op).empty() ||
        (dialect != "arith" && dialect != "index") || !isMemoryEffectFree(op) || !isSpeculatable(op)) { return false; }
    return independence[value] = llvm::all_of(op->getOperands(), [&](Value operand) { return independent(operand); });
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
    // Low-bit residues commute with fixed-width add/multiply. For other bank
    // counts a separate no-overflow proof is required; do not assume integers.
    if (!llvm::isPowerOf2_64(period) || !isa<arith::AddIOp, arith::MulIOp>(op)) { return std::nullopt; }
    auto left = residue(op->getOperand(0), phase, period), right = residue(op->getOperand(1), phase, period);
    if (!left || !right) { return std::nullopt; }
    if (isa<arith::AddIOp>(op)) { return save(arena.rem(arena.add(*left, *right), arena.constant(period))); }
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
