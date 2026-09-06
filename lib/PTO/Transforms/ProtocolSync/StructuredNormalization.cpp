// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Expand small statically finite inner regions, preserving their exact dynamic
// sequence and SSA transfers. This is not a proof by bounded unrolling: every
// occurrence of the inner loop is represented, while any dynamic outer loop
// still needs the native recurring memory and event-lifetime certificates.
// No synchronization or physical instruction is moved, and no target contract
// is inferred here. Unknown/large loops remain for structured synthesis.
#include "StructuredNormalization.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::pto::protocol_sync;

namespace {
constexpr unsigned MAX_INNER_TRIPS = 8;
constexpr unsigned MAX_EXPANDED_OPERATIONS = 2048;
constexpr unsigned MAX_EXPANDED_LOOPS = 32;

struct FiniteLoop {
    llvm::APInt lower;
    llvm::APInt step;
    unsigned trips;
};

std::optional<FiniteLoop> finiteLoop(scf::ForOp loop)
{
    IntegerAttr lower, upper, step;
    const bool constants = matchPattern(loop.getLowerBound(), m_Constant(&lower)) &&
                           matchPattern(loop.getUpperBound(), m_Constant(&upper)) &&
                           matchPattern(loop.getStep(), m_Constant(&step));
    if (!constants) {
        return std::nullopt;
    }
    const unsigned width = lower.getValue().getBitWidth();
    const bool supported = width <= 64 && upper.getValue().getBitWidth() == width &&
                           step.getValue().getBitWidth() == width && step.getValue().isStrictlyPositive();
    if (!supported) {
        return std::nullopt;
    }
    // Widen before subtraction, ceiling division and the final increment.
    const auto lo = lower.getValue().sext(128);
    const auto hi = upper.getValue().sext(128);
    const auto stride = step.getValue().sext(128);
    const auto count = lo.sge(hi) ? llvm::APInt(128, 0) : (hi - lo + stride - 1).udiv(stride);
    if (count.ugt(MAX_INNER_TRIPS) || !(lo + count * stride).isSignedIntN(width)) {
        return std::nullopt;
    }
    return FiniteLoop{lo, stride, static_cast<unsigned>(count.getZExtValue())};
}

unsigned operationCount(Operation* root)
{
    unsigned count = 0;
    root->walk([&](Operation*) {
        ++count;
        return count > MAX_EXPANDED_OPERATIONS ? WalkResult::interrupt() : WalkResult::advance();
    });
    return count;
}

void expand(scf::ForOp loop, const FiniteLoop& finite)
{
    OpBuilder builder(loop);
    SmallVector<Value> carried(loop.getInitArgs());
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (unsigned iteration = 0; iteration < finite.trips; ++iteration) {
        IRMapping mapping;
        auto value = finite.lower + finite.step * iteration;
        auto type = loop.getInductionVar().getType();
        Value induction =
            builder.create<arith::ConstantOp>(loop.getLoc(), type, builder.getIntegerAttr(type, value.getSExtValue()));
        mapping.map(loop.getInductionVar(), induction);
        mapping.map(loop.getRegionIterArgs(), carried);
        for (Operation& operation : loop.getBody()->without_terminator()) {
            builder.clone(operation, mapping);
        }
        carried.clear();
        for (Value result : yield.getOperands()) {
            carried.push_back(mapping.lookupOrDefault(result));
        }
    }
    for (auto [result, value] : llvm::zip_equal(loop.getResults(), carried)) {
        result.replaceAllUsesWith(value);
    }
    loop.erase();
}

void foldScalarOccurrences(func::FuncOp function)
{
    // Only fold scalar arithmetic. Do not run an unrestricted canonicalizer
    // that could erase/reorder physical operations or change storage placement.
    OperationFolder folder(function.getContext());
    function.walk<WalkOrder::PostOrder>([&](Operation* operation) {
        if (operation->getName().getDialectNamespace() == "arith" && operation->getNumRegions() == 0) {
            (void)folder.tryToFold(operation);
        }
    });
}

bool collectCondition(Value value, func::FuncOp function, llvm::SetVector<Operation*>& definitions, unsigned& budget)
{
    if (budget == 0 || !value.getType().isIntOrIndex()) {
        return false;
    }
    --budget;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        return argument.getOwner() == &function.getBody().front();
    }
    Operation* operation = value.getDefiningOp();
    if (!operation || operation->getNumRegions() != 0 || operation->getName().getDialectNamespace() != "arith" ||
        !isPure(operation)) {
        return false;
    }
    if (definitions.contains(operation)) {
        return true;
    }
    for (Value operand : operation->getOperands()) {
        if (!collectCondition(operand, function, definitions, budget)) {
            return false;
        }
    }
    definitions.insert(operation);
    return true;
}

scf::IfOp guardedBody(scf::ForOp loop)
{
    IntegerAttr step;
    if (loop.getNumResults() != 0 || !matchPattern(loop.getStep(), m_Constant(&step)) || !step.getValue().isOne()) {
        return {};
    }
    scf::IfOp selected;
    for (Operation& operation : loop.getBody()->without_terminator()) {
        if (auto choice = dyn_cast<scf::IfOp>(operation)) {
            if (selected || choice.getNumResults() != 0 ||
                (!choice.getElseRegion().empty() && !choice.getElseRegion().front().without_terminator().empty())) {
                return {};
            }
            selected = choice;
        } else if (operation.getName().getDialectNamespace() != "arith" || !isPure(&operation)) {
            return {};
        }
    }
    return selected;
}

bool restrictGuardedDomain(scf::ForOp loop)
{
    auto choice = guardedBody(loop);
    auto comparison = choice ? choice.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp();
    if (!comparison) {
        return false;
    }
    Value induction = loop.getInductionVar();
    auto predicate = comparison.getPredicate();
    const bool inductionOnLeft = comparison.getLhs() == induction;
    if (!inductionOnLeft && comparison.getRhs() != induction) {
        return false;
    }
    // Only inclusive lower bounds and exclusive upper bounds: no +/-1
    // arithmetic that could overflow, and no alteration of the step lattice.
    const bool lower =
        inductionOnLeft ? predicate == arith::CmpIPredicate::sge : predicate == arith::CmpIPredicate::sle;
    const bool upper =
        inductionOnLeft ? predicate == arith::CmpIPredicate::slt : predicate == arith::CmpIPredicate::sgt;
    if (!lower && !upper) {
        return false;
    }
    Value bound = inductionOnLeft ? comparison.getRhs() : comparison.getLhs();
    llvm::SetVector<Operation*> definitions;
    unsigned budget = 256;
    if (!collectCondition(bound, loop->getParentOfType<func::FuncOp>(), definitions, budget)) {
        return false;
    }
    OpBuilder builder(loop);
    IRMapping mapping;
    for (Operation* operation : definitions) {
        builder.clone(*operation, mapping);
    }
    bound = mapping.lookupOrDefault(bound);
    if (lower) {
        loop.setLowerBound(builder.create<arith::MaxSIOp>(loop.getLoc(), loop.getLowerBound(), bound));
    } else {
        loop.setUpperBound(builder.create<arith::MinSIOp>(loop.getLoc(), loop.getUpperBound(), bound));
    }
    specializeSyncPath(choice, true);
    return true;
}
} // namespace

SyncNormalizationResult mlir::pto::protocol_sync::expandSyncInnerOccurrences(func::FuncOp function)
{
    SyncNormalizationResult result;
    if (operationCount(function) <= MAX_EXPANDED_OPERATIONS) {
        function.walk([&](scf::ForOp loop) { result.restrictedDomains += restrictGuardedDomain(loop); });
    }
    for (unsigned round = 0; round < MAX_EXPANDED_LOOPS; ++round) {
        const unsigned total = operationCount(function);
        if (total > MAX_EXPANDED_OPERATIONS) {
            break;
        }
        scf::ForOp selected;
        std::optional<FiniteLoop> finite;
        function.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
            if (selected || !loop->getParentOfType<scf::ForOp>()) {
                return;
            }
            auto candidate = finiteLoop(loop);
            if (!candidate) {
                return;
            }
            const unsigned body = operationCount(loop);
            const std::uint64_t expansion = static_cast<std::uint64_t>(body + 1) * candidate->trips;
            if (total + expansion > MAX_EXPANDED_OPERATIONS) {
                return;
            }
            selected = loop;
            finite = std::move(candidate);
        });
        if (!selected) {
            break;
        }
        expand(selected, *finite);
        ++result.loops;
        result.iterations += finite->trips;
        foldScalarOccurrences(function);
    }
    return result;
}

scf::IfOp mlir::pto::protocol_sync::findSyncPathChoice(func::FuncOp function)
{
    if (!function.getBody().hasOneBlock() || function.getFunctionType().getNumResults() != 0 ||
        !isa<func::ReturnOp>(function.getBody().front().getTerminator()) ||
        operationCount(function) > MAX_EXPANDED_OPERATIONS) {
        return {};
    }
    scf::IfOp selected;
    unsigned physicalChoices = 0;
    function.walk([&](scf::IfOp choice) {
        bool physical = false;
        choice.walk([&](Operation* operation) {
            physical |= isa<pto::OpPipeInterface, pto::LoadScalarOp, pto::StoreScalarOp>(operation);
        });
        if (physical) {
            ++physicalChoices;
            selected = choice;
        }
    });
    const bool onceOnly =
        physicalChoices == 1 && selected && selected->getParentOp() == function && selected.getNumResults() == 0;
    if (!onceOnly) {
        return {};
    }
    llvm::SetVector<Operation*> definitions;
    unsigned budget = 256;
    return collectCondition(selected.getCondition(), function, definitions, budget) ? selected : scf::IfOp();
}

Value mlir::pto::protocol_sync::materializeSyncPathCondition(OpBuilder& builder, scf::IfOp choice)
{
    llvm::SetVector<Operation*> definitions;
    unsigned budget = 256;
    if (!collectCondition(choice.getCondition(), choice->getParentOfType<func::FuncOp>(), definitions, budget)) {
        return {};
    }
    IRMapping mapping;
    for (Operation* operation : definitions) {
        builder.clone(*operation, mapping);
    }
    return mapping.lookupOrDefault(choice.getCondition());
}

void mlir::pto::protocol_sync::specializeSyncPath(scf::IfOp choice, bool thenPath)
{
    Region& selected = thenPath ? choice.getThenRegion() : choice.getElseRegion();
    if (!selected.empty()) {
        Block& block = selected.front();
        while (&block.front() != block.getTerminator()) {
            block.front().moveBefore(choice);
        }
    }
    choice.erase();
}
