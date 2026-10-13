// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One slot pair per requested evaluation, not per maximum selected ordinal.
#include "InvocationPrefixCollection.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
namespace mlir::pto::frontiersynch {
FailureOr<std::pair<std::size_t, std::size_t>> InvocationPrefixCollection::snapshotPredicate(
    Value value, scf::ForOp loop, std::shared_ptr<RegionExpressions> arena,
    RegionExpressions::Id ordinal, std::string& error)
{
    auto* definition = value ? value.getDefiningOp() : nullptr;
    if (!original || !loop || !definition || !value.getType().isInteger(1) ||
        !original->isProperAncestor(loop) || !loop.getInductionVar().getType().isIndex() ||
        definition->getBlock() != loop.getBody()) {
        error = "historical predicate lacks an unconditional original loop evaluation"; return failure();
    }
    const auto bits = DataLayout::closest(loop).getTypeSizeInBits(loop.getInductionVar().getType());
    if (bits.isScalable() || bits.getFixedValue() != 64) {
        error = "historical predicate requires 64-bit index control"; return failure();
    }
    APInt step;
    if (!arena || ordinal >= arena->size() || arena->isBoolean(ordinal) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) || step != 1) {
        error = "historical predicate requires a unit-step ordinal"; return failure();
    }
    DominanceInfo dominance(original);
    const auto inputs = arena->referencedInputs(ordinal);
    for (auto [id, input] : inputs) {
        if (arena->evaluation(id) || !dominance.properlyDominates(input, loop)) {
            error = "historical predicate ordinal is not available before its invocation"; return failure();
        }
    }
    const auto key = std::make_tuple(value.getAsOpaquePointer(), arena.get(), ordinal);
    auto found = predicateSnapshots.find(key);
    if (found != predicateSnapshots.end()) { return found->second; }
    const auto valueSlot = pending.slots.size(), validSlot = valueSlot + 1;
    auto type = IntegerType::get(original.getContext(), 1);
    auto initial = IntegerAttr::get(type, 0);
    pending.slots.push_back({type, initial}); pending.slots.push_back({type, initial});
    pending.actions.push_back({loop, PrivateStateCut::Before,
        [valueSlot, validSlot](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
            auto no = b.create<arith::ConstantIntOp>(b.getUnknownLoc(), 0, 1);
            state[valueSlot] = no; state[validSlot] = no; return success();
        }});
    pending.actions.push_back({definition, PrivateStateCut::After,
        [value, loop, arena, ordinal, inputs, valueSlot, validSlot](OpBuilder& b, const IRMapping& mapping,
            MutableArrayRef<Value> state, std::string& error) mutable {
            auto actual = mapping.lookupOrNull(value);
            auto visit = mapping.lookupOrNull(loop.getInductionVar());
            if (!actual || !visit) {
                error = "historical predicate lost its original clone identity"; return failure();
            }
            RegionExpressions::CutEmission context;
            for (auto [id, input] : inputs) {
                auto mapped = mapping.lookupOrNull(input);
                if (!mapped) { error = "historical ordinal lost its original input"; return failure(); }
                context.inputs[input] = mapped; context.cofactors[id] = id;
            }
            Block scratch;
            OpBuilder detached(b.getContext()); detached.setInsertionPointToEnd(&scratch);
            auto selected = arena->emitContextual(ordinal, detached, value.getDefiningOp(), context);
            auto lower = mapping.lookupOrNull(loop.getLowerBound());
            if (failed(selected) || !lower || !(*selected).getType().isIndex()) {
                error = "historical ordinal cannot be emitted at its snapshot"; return failure();
            }
            b.getInsertionBlock()->getOperations().splice(b.getInsertionPoint(), scratch.getOperations());
            // The unsigned distance is unique for every executed signed IV,
            // even when lower is negative. Out-of-domain ordinals never match.
            auto current = b.create<arith::SubIOp>(value.getLoc(), visit, lower);
            auto match = b.create<arith::CmpIOp>(value.getLoc(), arith::CmpIPredicate::eq, current, *selected);
            state[valueSlot] = b.create<arith::SelectOp>(value.getLoc(), match, actual, state[valueSlot]);
            state[validSlot] = b.create<arith::OrIOp>(value.getLoc(), match, state[validSlot]);
            return success();
        }});
    auto slots = std::make_pair(valueSlot, validSlot);
    predicateSnapshots.try_emplace(key, slots);
    return slots;
}
} // namespace mlir::pto::frontiersynch
