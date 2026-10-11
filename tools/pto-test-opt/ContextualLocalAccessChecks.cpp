// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exercise the IR query with independently supplied dimensions and effect roles.
// These callback facts test contracts; they do not claim descriptor dataflow.
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOAccessRegion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
using namespace mlir::pto;
namespace {
using Effects = SmallVector<MemoryEffects::EffectInstance>;
using Shape = std::optional<std::array<int64_t, 2>>;
std::string printIR(Operation* operation)
{
    std::string result;
    llvm::raw_string_ostream stream(result);
    operation->print(stream);
    return result;
}
bool sameEffects(ArrayRef<MemoryEffects::EffectInstance> a, ArrayRef<MemoryEffects::EffectInstance> b)
{
    if (a.size() != b.size()) {
        return false;
    }
    return llvm::all_of(llvm::zip(a, b), [](const auto& pair) {
        const auto& [x, y] = pair;
        return x.getEffect() == y.getEffect() && x.getValue() == y.getValue() && x.getResource() == y.getResource() &&
               x.getStage() == y.getStage() && x.getEffectOnFullRegion() == y.getEffectOnFullRegion() &&
               x.getParameters() == y.getParameters();
    });
}
Shape fixtureShape(Operation* operation, Value value)
{
    auto shapes = operation->getAttrOfType<DenseI64ArrayAttr>("test.contextual_shapes");
    if (!shapes || shapes.size() % 3) {
        return std::nullopt;
    }
    for (unsigned i = 0; i < shapes.size(); i += 3) {
        if (shapes[i] < 0 || static_cast<uint64_t>(shapes[i]) >= operation->getNumOperands()) {
            return std::nullopt;
        }
        if (operation->getOperand(shapes[i]) == value) {
            return std::array<int64_t, 2>{shapes[i + 1], shapes[i + 2]};
        }
    }
    return std::nullopt;
}
bool checkSelection(Operation* operation, DictionaryAttr contract)
{
    auto coordinates = contract.getAs<AffineMapAttr>("coordinates");
    auto guards = contract.getAs<DenseI64ArrayAttr>("valid_shapes");
    auto capacity = contract.getAs<BoolAttr>("capacity");
    if (!coordinates || !coordinates.getValue().isIdentity() || !guards || guards.size() % 3 || !capacity ||
        capacity.getValue()) {
        return false;
    }
    for (unsigned i = 0; i < guards.size(); i += 3) {
        if (guards[i] < 0 || static_cast<uint64_t>(guards[i]) >= operation->getNumOperands()) {
            return false;
        }
        auto shape = fixtureShape(operation, operation->getOperand(guards[i]));
        if (!shape || (*shape)[0] != guards[i + 1] || (*shape)[1] != guards[i + 2]) {
            return false;
        }
    }
    return true;
}
bool expectedEffects(Operation* operation, ArrayRef<MemoryEffects::EffectInstance> effects)
{
    const bool a3 = getTargetArch(operation) == PTOArch::A3;
    auto expected = operation->getAttrOfType<DenseI64ArrayAttr>(a3 ? "test.effects_a3" : "test.effects_a5");
    if (!expected || expected.size() % 3 || static_cast<uint64_t>(expected.size() / 3) != effects.size()) {
        return false;
    }
    for (unsigned i = 0; i < effects.size(); ++i) {
        auto index = expected[3 * i];
        if (index < 0 || static_cast<uint64_t>(index) >= operation->getNumOperands()) {
            return false;
        }
        const auto& effect = effects[i];
        if (effect.getValue() != operation->getOperand(index) ||
            effect.getEffect() != (expected[3 * i + 1] ?
                                       static_cast<MemoryEffects::Effect*>(MemoryEffects::Write::get()) :
                                       static_cast<MemoryEffects::Effect*>(MemoryEffects::Read::get()))) {
            return false;
        }
        auto contract = dyn_cast_or_null<DictionaryAttr>(effect.getParameters());
        if (bool(contract) != bool(expected[3 * i + 2]) || (contract && !checkSelection(operation, contract))) {
            return false;
        }
    }
    return true;
}
bool checkUnknownQuery(Operation* operation, TileValidShapeQuery query)
{
    Effects effects;
    effects.emplace_back(MemoryEffects::Read::get());
    const Effects original = effects;
    const bool handled = getContextualLocalAccessEffects(operation, effects, query);
    if (!handled) {
        return sameEffects(effects, original);
    }
    if (effects.empty() || !sameEffects(ArrayRef(effects).take_front(), original)) {
        return false;
    }
    // The aligned fallback can still declare a conditional full-capacity
    // selection. Unknown facts cannot become the initial partial valid shape.
    for (const auto& effect : ArrayRef(effects).drop_front()) {
        auto contract = dyn_cast_or_null<DictionaryAttr>(effect.getParameters());
        if (!contract) {
            continue;
        }
        auto guards = contract.getAs<DenseI64ArrayAttr>("valid_shapes");
        if (!guards || guards.size() % 3) {
            return false;
        }
        for (unsigned i = 0; i < guards.size(); i += 3) {
            if (guards[i] < 0 || static_cast<uint64_t>(guards[i]) >= operation->getNumOperands()) {
                return false;
            }
            auto type = dyn_cast<TileBufType>(operation->getOperand(guards[i]).getType());
            if (!type || type.getShape()[0] != guards[i + 1] || type.getShape()[1] != guards[i + 2]) {
                return false;
            }
        }
    }
    return true;
}
bool checkOperation(Operation* operation)
{
    auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!interface) {
        return false;
    }
    Effects ordinary, contextual;
    interface.getEffects(ordinary);
    const bool handled = getContextualLocalAccessEffects(
        operation, contextual, [&](Value value) { return fixtureShape(operation, value); });
    const bool expectedHandled = !isa<TRowSumOp>(operation) || getTargetArch(operation) == PTOArch::A3;
    if (handled != expectedHandled || (!handled && !contextual.empty())) {
        return false;
    }
    if (!handled) {
        contextual = ordinary;
    }
    if (!expectedEffects(operation, contextual) ||
        (operation->hasAttr("test.static_equivalent") && !sameEffects(ordinary, contextual))) {
        return false;
    }
    if (!checkUnknownQuery(operation, [](Value) -> Shape { return std::nullopt; }) ||
        !checkUnknownQuery(operation, [](Value) -> Shape { return std::array<int64_t, 2>{1, INT64_MAX}; })) {
        return false;
    }
    Effects repeated;
    const bool again = getContextualLocalAccessEffects(
        operation, repeated, [&](Value value) { return fixtureShape(operation, value); });
    if (!again) {
        repeated = ordinary;
    }
    return again == handled && sameEffects(repeated, contextual);
}
} // namespace
int runContextualLocalAccessChecks(func::FuncOp function)
{
    const auto before = printIR(function);
    unsigned checked = 0;
    auto result = function.walk([&](Operation* operation) -> WalkResult {
        if (!operation->hasAttr("test.contextual_shapes")) {
            return WalkResult::advance();
        }
        ++checked;
        if (!checkOperation(operation)) {
            operation->emitError("contextual family effect mismatch");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    Effects sentinel;
    sentinel.emplace_back(MemoryEffects::Read::get());
    const Effects original = sentinel;
    if (result.wasInterrupted() || !checked || before != printIR(function) ||
        getContextualLocalAccessEffects(function, sentinel, [](Value) -> Shape { return std::nullopt; }) ||
        !sameEffects(sentinel, original)) {
        return 1;
    }
    llvm::outs() << "contextual local contracts: " << function.getSymName() << " checked=" << checked << "\n";
    return 0;
}
