// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A bounded scalar shift register stores evaluations without replaying their producers.
#include "BoundedPredicateHistory.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <limits>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
struct Entry {
    BoundedPredicateHistoryRequirement requirement;
    std::size_t firstSlot;
    std::size_t visits;
};
bool inVisit(Operation* operation, scf::ForOp loop)
{
    if (!operation || !loop || operation == loop) { return false; }
    for (auto* parent = operation->getParentOp(); parent; parent = parent->getParentOp()) {
        if (parent == loop) { return true; }
        if (!isa<scf::IfOp>(parent)) { return false; }
    }
    return false;
}
bool validRequirement(func::FuncOp function, const BoundedPredicateHistoryRequirement& requirement)
{
    auto* definition = requirement.predicate ? requirement.predicate.getDefiningOp() : nullptr;
    return function && requirement.loop && requirement.loop->getParentOfType<func::FuncOp>() == function &&
        requirement.predicate && requirement.predicate.getType().isInteger(1) && definition &&
        definition->getNumRegions() == 0 && inVisit(definition, requirement.loop);
}
Value falseValue(OpBuilder& builder, Location location)
{
    return builder.create<arith::ConstantIntOp>(location, 0, 1);
}
bool stateExtent(const Entry& entry, llvm::ArrayRef<Value> state, std::string& error)
{
    const bool incomplete = state.size() < entry.firstSlot + 2 * entry.visits;
    if (incomplete) {
        error = "predicate history action has incomplete scalar state";
        return false;
    }
    return true;
}
func::FuncOp valueFunction(Value value)
{
    auto* region = value ? value.getParentRegion() : nullptr;
    auto* owner = region ? region->getParentOp() : nullptr;
    if (!owner) { return {}; }
    if (auto function = dyn_cast<func::FuncOp>(owner)) { return function; }
    return owner->getParentOfType<func::FuncOp>();
}
PrivateStateAction resetAction(const Entry& entry)
{
    return {entry.requirement.loop, PrivateStateCut::Before,
        [entry](OpBuilder& builder, const IRMapping&, llvm::MutableArrayRef<Value> state, std::string& error) {
            if (!stateExtent(entry, state, error)) { return failure(); }
            auto no = falseValue(builder, entry.requirement.loop->getLoc());
            for (std::size_t j = 0; j < 2 * entry.visits; ++j) { state[entry.firstSlot + j] = no; }
            return success();
        }};
}
PrivateStateAction shiftAction(const Entry& entry)
{
    auto loop = entry.requirement.loop;
    return {&loop.getBody()->front(), PrivateStateCut::Before,
        [entry](OpBuilder& builder, const IRMapping&, llvm::MutableArrayRef<Value> state, std::string& error) {
            if (!stateExtent(entry, state, error)) { return failure(); }
            for (std::size_t j = entry.visits - 1; j > 0; --j) {
                state[entry.firstSlot + 2 * j] = state[entry.firstSlot + 2 * (j - 1)];
                state[entry.firstSlot + 2 * j + 1] = state[entry.firstSlot + 2 * (j - 1) + 1];
            }
            auto no = falseValue(builder, entry.requirement.loop->getLoc());
            state[entry.firstSlot] = no;
            state[entry.firstSlot + 1] = no;
            return success();
        }};
}
PrivateStateAction snapshotAction(const Entry& entry)
{
    return {entry.requirement.predicate.getDefiningOp(), PrivateStateCut::After,
        [entry](OpBuilder& builder, const IRMapping& mapping, llvm::MutableArrayRef<Value> state, std::string& error) {
            if (!stateExtent(entry, state, error)) { return failure(); }
            auto actual = mapping.lookupOrNull(entry.requirement.predicate);
            if (!actual || !actual.getType().isInteger(1)) {
                error = "predicate snapshot has no mapped original Boolean evaluation";
                return failure();
            }
            state[entry.firstSlot] = actual;
            state[entry.firstSlot + 1] = builder.create<arith::ConstantIntOp>(actual.getLoc(), 1, 1);
            return success();
        }};
}
} // namespace
struct BoundedPredicateHistory::Layout {
    func::FuncOp original;
    std::size_t firstSlot = 0;
    std::size_t slots = 0;
    std::vector<Entry> entries;
};
FailureOr<BoundedPredicateHistory> BoundedPredicateHistory::append(func::FuncOp original,
    llvm::ArrayRef<BoundedPredicateHistoryRequirement> requirements,
    std::vector<PrivateStateSlot>& slots, std::vector<PrivateStateAction>& actions, std::string& error)
{
    error.clear();
    const bool invalidFrame = !original || original.getBody().empty() || !original.getBody().hasOneBlock() ||
        requirements.size() > (actions.max_size() - actions.size()) / 3;
    if (invalidFrame) {
        error = "predicate history requires one original function body block";
        return failure();
    }
    auto layout = std::make_shared<Layout>();
    layout->original = original;
    layout->firstSlot = slots.size();
    std::set<std::pair<const void*, Operation*>> seen;
    for (const auto& requirement : requirements) {
        auto loop = requirement.loop;
        const bool invalidRequirement = !validRequirement(original, requirement) ||
            !seen.emplace(requirement.predicate.getAsOpaquePointer(), loop.getOperation()).second ||
            requirement.maximumLag >= std::numeric_limits<std::size_t>::max() / 2;
        if (invalidRequirement) {
            error = "predicate history requires distinct scalar evaluations in supported loop frames and bounded slots";
            return failure();
        }
        const auto visits = static_cast<std::size_t>(requirement.maximumLag) + 1;
        const auto count = 2 * visits;
        const bool overflow = count > slots.max_size() - layout->firstSlot - layout->slots;
        if (overflow) {
            error = "predicate history scalar slot extent is not representable";
            return failure();
        }
        layout->entries.push_back({requirement, layout->firstSlot + layout->slots, visits});
        layout->slots += count;
    }
    std::vector<PrivateStateAction> extra;
    for (const auto& entry : layout->entries) {
        extra.push_back(resetAction(entry));
        extra.push_back(shiftAction(entry));
        extra.push_back(snapshotAction(entry));
    }
    auto type = IntegerType::get(original.getContext(), 1);
    auto zero = IntegerAttr::get(type, 0);
    slots.insert(slots.end(), layout->slots, {type, zero});
    actions.insert(actions.end(), extra.begin(), extra.end());
    return BoundedPredicateHistory(std::move(layout));
}
FailureOr<BoundedPredicateHistoryRead> BoundedPredicateHistory::read(std::size_t requirement, uint64_t lag,
    Operation* originalCut, const IRMapping& mapping, llvm::ArrayRef<Value> state, std::string& error) const
{
    error.clear();
    if (!layout || requirement >= layout->entries.size()) {
        error = "predicate history requirement identity is invalid";
        return failure();
    }
    const auto& entry = layout->entries[requirement];
    auto original = layout->original;
    auto* mapped = mapping.lookupOrNull(original.getOperation());
    auto function = dyn_cast_or_null<func::FuncOp>(mapped);
    if (!function || function == layout->original || lag >= entry.visits ||
        !inVisit(originalCut, entry.requirement.loop) || state.size() < layout->firstSlot + layout->slots) {
        error = "predicate history read has foreign clone, cut, lag or scalar state";
        return failure();
    }
    const auto index = entry.firstSlot + 2 * static_cast<std::size_t>(lag);
    auto value = state[index], valid = state[index + 1];
    const bool foreignRead = !value || !valid || !value.getType().isInteger(1) || !valid.getType().isInteger(1) ||
        valueFunction(value) != function || valueFunction(valid) != function;
    if (foreignRead) {
        error = "predicate history scalar values do not belong to the mapped clone";
        return failure();
    }
    return BoundedPredicateHistoryRead{value, valid};
}
std::size_t BoundedPredicateHistory::slotCount() const
{
    return layout ? layout->slots : 0;
}
} // namespace mlir::pto::frontiersynch
