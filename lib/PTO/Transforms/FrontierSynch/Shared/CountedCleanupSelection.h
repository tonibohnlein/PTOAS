// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Interpret the same singleton-cut proof over owned counted scalar identities.
#ifndef PTO_FRONTIER_COUNTEDCLEANUPSELECTION_H
#define PTO_FRONTIER_COUNTEDCLEANUPSELECTION_H
#include "CleanupSelectionProof.h"
#include "ScalarConstruction.h"
#include "../../InsertSync/SyncScalarReplay.h"
#include "mlir/IR/Matchers.h"
namespace mlir::pto::frontiersynch::detail {
class CountedCleanupPredicate {
public:
    using Ref = ScalarToken;
    using Binary = CleanupPredicateBinary;
    explicit CountedCleanupPredicate(ScalarConstruction& construction) : construction(construction)
    {
        for (const auto& instruction : construction.trace()) {
            for (auto result : instruction.results) { definitions[result] = &instruction; }
        }
    }
    Ref get(Value value) const { return construction.input(value); }
    bool valid(Ref ref) const
    {
        if (!ref || !construction.owns(ref) || !ref.type) { return false; }
        auto retained = construction.lookup(ref.id);
        return retained.owner == ref.owner && retained.type == ref.type && retained.value == ref.value &&
            (ref.value || definitions.count(ref.id));
    }
    bool boolean(Ref ref) const { return valid(ref) && ref.type.isInteger(1); }
    Value original(Ref ref) const { return ref.value; }
    std::optional<uint64_t> number(Ref ref) const
    {
        if (!valid(ref)) { return std::nullopt; }
        APInt number;
        if (ref.value) {
            if (!matchPattern(ref.value, m_ConstantInt(&number))) { return std::nullopt; }
        } else {
            const auto* instruction = definitions.lookup(ref.id);
            auto properties = dyn_cast_or_null<DictionaryAttr>(instruction->properties);
            auto value = properties ? properties.getAs<IntegerAttr>("value") : IntegerAttr{};
            if (instruction->name.getValue() != "arith.constant" || !value) { return std::nullopt; }
            number = value.getValue();
        }
        if (!number.isSignedIntN(64)) { return std::nullopt; }
        return number.sextOrTrunc(64).getZExtValue();
    }
    std::optional<std::pair<Ref, Ref>> binary(Ref ref, Binary kind) const
    {
        if (!valid(ref)) { return std::nullopt; }
        if (ref.value) {
            auto* op = ref.value.getDefiningOp();
            bool matches = op && ((kind == Binary::And && isa<arith::AndIOp>(op) && boolean(ref)) ||
                (kind == Binary::Subtract && isa<arith::SubIOp>(op)) ||
                (kind == Binary::Divide && isa<arith::DivUIOp>(op)));
            if (kind == Binary::Equal) {
                auto compare = dyn_cast_or_null<arith::CmpIOp>(op);
                matches = compare && compare.getPredicate() == arith::CmpIPredicate::eq;
            }
            if (!matches) { return std::nullopt; }
            return std::make_pair(get(op->getOperand(0)), get(op->getOperand(1)));
        }
        const auto* instruction = definitions.lookup(ref.id);
        const auto name = instruction->name.getValue();
        if (instruction->operands.size() != 2) { return std::nullopt; }
        bool matches = (kind == Binary::And && name == "arith.andi" && boolean(ref)) ||
            (kind == Binary::Subtract && name == "arith.subi") ||
            (kind == Binary::Divide && name == "arith.divui");
        if (kind == Binary::Equal) {
            auto properties = dyn_cast_or_null<DictionaryAttr>(instruction->properties);
            auto predicate = properties ? properties.getAs<arith::CmpIPredicateAttr>("predicate") :
                arith::CmpIPredicateAttr{};
            matches = name == "arith.cmpi" && predicate && predicate.getValue() == arith::CmpIPredicate::eq;
        }
        if (!matches) { return std::nullopt; }
        return std::make_pair(construction.lookup(instruction->operands[0]),
                              construction.lookup(instruction->operands[1]));
    }
    std::optional<std::pair<Ref, Ref>> masked(Ref ref) const
    {
        if (!valid(ref)) { return std::nullopt; }
        if (ref.value) {
            auto select = ref.value.getDefiningOp<arith::SelectOp>();
            if (!select || number(get(select.getFalseValue())) != 0) { return std::nullopt; }
            return std::make_pair(get(select.getCondition()), get(select.getTrueValue()));
        }
        const auto* instruction = definitions.lookup(ref.id);
        if (instruction->name.getValue() != "arith.select" || instruction->operands.size() != 3) {
            return std::nullopt;
        }
        auto fallback = construction.lookup(instruction->operands[2]);
        if (number(fallback) != 0) { return std::nullopt; }
        return std::make_pair(construction.lookup(instruction->operands[0]),
                              construction.lookup(instruction->operands[1]));
    }
    bool appendOperands(Ref ref, SmallVectorImpl<Ref>& operands) const
    {
        if (!valid(ref)) { return false; }
        if (ref.value) {
            auto* op = ref.value.getDefiningOp();
            if (!mlir::pto::detail::canReplayScalar(op)) { return false; }
            for (auto operand : op->getOperands()) { operands.push_back(get(operand)); }
        } else {
            for (auto operand : definitions.lookup(ref.id)->operands) {
                operands.push_back(construction.lookup(operand));
            }
        }
        return true;
    }
private:
    ScalarConstruction& construction;
    llvm::DenseMap<uint64_t, const ScalarInstruction*> definitions;
};
} // namespace mlir::pto::frontiersynch::detail
#endif
