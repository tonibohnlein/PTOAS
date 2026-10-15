// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_SCALARCONSTRUCTION_H
#define PTO_FRONTIER_SCALARCONSTRUCTION_H
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
#include "ScalarConstructionTrace.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
namespace mlir::pto::frontiersynch::detail {
class ScalarConstruction;
// Tokens model construction identity, not mathematical equivalence. The same
// schedule owns its explicit caches in both modes; there is no speculative SSA
// or extra folding in the counting mode.
struct ScalarToken {
    uint64_t id = 0;
    Value value;
    const ScalarConstruction* owner = nullptr;
    Type type;
    explicit operator bool() const { return id != 0; }
    bool operator==(const ScalarToken& other) const { return id == other.id && owner == other.owner; }
    bool operator!=(const ScalarToken& other) const { return !(*this == other); }
};
class ScalarConstruction {
public:
    ScalarConstruction() = default;
    ScalarConstruction(OpBuilder& builder, Location location) : builder(&builder), location(location) {}
    ScalarConstruction(const ScalarConstruction&) = delete;
    ScalarConstruction& operator=(const ScalarConstruction&) = delete;
    bool owns(ScalarToken value) const { return !value || value.owner == this; }
    void setLocation(Location next) { location = next; }
    void enableTrace(MLIRContext* context) { traceContext = context; traceValid = !serial; }
    void setInsertionCut(Operation* cut) { insertionCut = cut; }
    bool validTrace() const { return traceContext && traceValid; }
    ArrayRef<ScalarInstruction> trace() const { return instructions; }
    ScalarToken lookup(uint64_t id) const { return tokens.lookup(id); }
    ScalarToken input(Value value)
    {
        if (!value) { return {}; }
        auto [entry, added] = inputs.try_emplace(value);
        if (added) { entry->second = token(value); }
        return entry->second;
    }
    // A symbolic external input whose exact upstream construction is common.
    ScalarToken external() { return token({}); }
    std::optional<uint64_t> operations() const { return valid ? std::optional(count) : std::nullopt; }
    ScalarToken number(int64_t value)
    {
        auto type = context() ? Type(IndexType::get(context())) : Type{};
        return constant(type, traceContext ? IntegerAttr::get(type, value) : IntegerAttr{},
            builder ? Value(builder->create<arith::ConstantIndexOp>(*location, value)) : Value{});
    }
    ScalarToken boolean(bool value)
    {
        auto type = context() ? Type(IntegerType::get(context(), 1)) : Type{};
        return constant(type, traceContext ? IntegerAttr::get(type, value) : IntegerAttr{},
            builder ? Value(builder->create<arith::ConstantIntOp>(*location, value, 1)) : Value{});
    }
    ScalarToken indexCast(ScalarToken value)
    {
        auto type = context() ? Type(IndexType::get(context())) : Type{};
        return typedOperation<arith::IndexCastOp>(builder ? Value(builder->create<arith::IndexCastOp>(
            *location, type, value.value)) : Value{}, type, {value});
    }
    ScalarToken integer(const APInt& value)
    {
        auto type = context() ? Type(IntegerType::get(context(), value.getBitWidth())) : Type{};
        auto attribute = type ? IntegerAttr::get(type, value) : IntegerAttr{};
        return constant(type, attribute, builder ? Value(builder->create<arith::ConstantOp>(
            *location, type, attribute)) : Value{});
    }
    template<class Op> ScalarToken cast(Type type, ScalarToken value)
    {
        return typedOperation<Op>(builder ? Value(builder->create<Op>(*location, type, value.value)) : Value{},
            type, {value});
    }
    SmallVector<ScalarToken> replay(Operation& original, ArrayRef<ScalarToken> operands)
    {
        SmallVector<ScalarToken> result;
        if (operands.size() != original.getNumOperands()) { valid = false; return result; }
        Operation* clone = nullptr;
        if (builder) {
            IRMapping mapping;
            for (auto [input, replacement] : llvm::zip(original.getOperands(), operands)) {
                mapping.map(input, replacement.value);
            }
            clone = builder->clone(original, mapping);
        }
        // One replayed operation can define several results. It is constructed
        // once, and every result has a distinct stable token in both modes.
        if (count == uint64_t(INT64_MAX)) { valid = false; }
        else { ++count; }
        for (unsigned i = 0; i < original.getNumResults(); ++i) {
            Value value = clone ? clone->getResult(i) : Value{};
            auto produced = token(value, original.getResult(i).getType());
            if (value) { inputs[value] = produced; }
            result.push_back(produced);
        }
        if (traceContext) {
            llvm::DenseMap<Value, uint64_t> identities;
            for (auto [input, replacement] : llvm::zip(original.getOperands(), operands)) {
                identities[input] = replacement.id;
            }
            for (auto [value, replacement] : llvm::zip(original.getResults(), result)) {
                identities[value] = replacement.id;
            }
            auto described = describeScalarInstruction(original, insertionCut,
                [&](Value value) { return identities.lookup(value); });
            if (!described || !insertionCut) { traceValid = false; }
            else { instructions.push_back(std::move(*described)); }
        }
        return result;
    }
    template<class Op> ScalarToken binary(ScalarToken a, ScalarToken b)
    {
        return typedOperation<Op>(builder ? Value(builder->create<Op>(*location, a.value, b.value)) : Value{},
            a.type, {a, b});
    }
    ScalarToken compare(arith::CmpIPredicate predicate, ScalarToken a, ScalarToken b)
    {
        auto type = context() ? Type(IntegerType::get(context(), 1)) : Type{};
        Attribute properties;
        if (traceContext) {
            arith::CmpIOp::Properties values;
            values.setPredicate(arith::CmpIPredicateAttr::get(traceContext, predicate));
            properties = arith::CmpIOp::getPropertiesAsAttr(traceContext, values);
        }
        return typedOperation<arith::CmpIOp>(builder ? Value(builder->create<arith::CmpIOp>(
            *location, predicate, a.value, b.value)) : Value{}, type, {a, b}, properties);
    }
    ScalarToken select(ScalarToken condition, ScalarToken yes, ScalarToken no)
    {
        return typedOperation<arith::SelectOp>(builder ? Value(builder->create<arith::SelectOp>(
            *location, condition.value, yes.value, no.value)) : Value{}, yes.type, {condition, yes, no});
    }
private:
    OpBuilder* builder = nullptr;
    std::optional<Location> location;
    llvm::DenseMap<Value, ScalarToken> inputs;
    uint64_t serial = 0, count = 0;
    bool valid = true;
    MLIRContext* traceContext = nullptr;
    Operation* insertionCut = nullptr;
    bool traceValid = true;
    SmallVector<ScalarInstruction> instructions;
    llvm::DenseMap<uint64_t, ScalarToken> tokens;
    MLIRContext* context() const { return builder ? builder->getContext() : traceContext; }
    ScalarToken token(Value value, Type type = {})
    {
        if (serial == UINT64_MAX) { valid = false; return {}; }
        ScalarToken result{++serial, value, this, value ? value.getType() : type};
        if (traceContext) { tokens[result.id] = result; }
        return result;
    }
    ScalarToken constant(Type type, IntegerAttr attribute, Value value)
    {
        Attribute properties;
        if (traceContext) {
            arith::ConstantOp::Properties values;
            values.setValue(attribute);
            properties = arith::ConstantOp::getPropertiesAsAttr(traceContext, values);
        }
        return typedOperation<arith::ConstantOp>(value, type, {}, properties);
    }
    template<class Op>
    ScalarToken typedOperation(Value value, Type type, ArrayRef<ScalarToken> operands, Attribute properties = {})
    {
        auto result = operation(value, type);
        if (!traceContext) { return result; }
        if (!insertionCut || !type || llvm::any_of(operands, [&](auto operand) {
                return !operand || !owns(operand) || !operand.type;
            })) { traceValid = false; return result; }
        if constexpr (Op::hasProperties()) {
            if (!properties) {
                typename Op::Properties defaults;
                Op::populateDefaultProperties(OperationName(Op::getOperationName(), traceContext), defaults);
                properties = Op::getPropertiesAsAttr(traceContext, defaults);
            }
        }
        ScalarInstruction instruction;
        instruction.cut = insertionCut; instruction.name = StringAttr::get(traceContext, Op::getOperationName());
        instruction.attributes = DictionaryAttr::get(traceContext); instruction.properties = properties;
        instruction.types.push_back(type); instruction.results.push_back(result.id);
        for (auto operand : operands) { instruction.operands.push_back(operand.id); }
        instruction.commutative = Op::template hasTrait<OpTrait::IsCommutative>();
        instructions.push_back(std::move(instruction));
        return result;
    }
    ScalarToken operation(Value value, Type type)
    {
        if (count == uint64_t(INT64_MAX)) { valid = false; }
        else { ++count; }
        auto result = token(value, type);
        if (value) { inputs[value] = result; }
        return result;
    }
};
struct FamilyScalarTokens { ScalarToken present, member; };
ScalarToken constructFamilyPresence(ScalarConstruction& construction, const FamilyExpressionRecipe& recipe);
FamilyScalarTokens constructFamilyExpressions(ScalarConstruction& construction,
    const FamilyExpressionRecipe& recipe);
} // namespace mlir::pto::frontiersynch::detail
namespace llvm {
template<> struct DenseMapInfo<mlir::pto::frontiersynch::detail::ScalarToken> {
    using Token = mlir::pto::frontiersynch::detail::ScalarToken;
    static Token getEmptyKey() { return {UINT64_MAX, {}, nullptr, {}}; }
    static Token getTombstoneKey() { return {UINT64_MAX - 1, {}, nullptr, {}}; }
    static unsigned getHashValue(Token value)
    {
        return static_cast<unsigned>(hash_combine(value.id, value.owner));
    }
    static bool isEqual(Token a, Token b) { return a == b; }
};
} // namespace llvm
#endif
