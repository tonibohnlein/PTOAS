// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Match a deliberately small modular language. Add/multiply syntax is reported
// with an arithmetic obligation until equivalence under IR overflow is proved.
#include "RotationPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

namespace mlir::pto::frontiersynch::detail {
namespace {
std::optional<int64_t> integer(Value value)
{
    APInt number;
    if (!value || !matchPattern(value, m_ConstantInt(&number)) || !number.isSignedIntN(64)) {
        return std::nullopt;
    }
    return number.getSExtValue();
}
uint64_t residue(int64_t value, uint64_t modulus)
{
    const auto remainder = value % static_cast<int64_t>(modulus);
    return remainder < 0 ? modulus - static_cast<uint64_t>(-(remainder + 1)) - 1 :
                           static_cast<uint64_t>(remainder);
}
std::optional<SlotPattern> term(Value value, Value induction, uint64_t count)
{
    if (value == induction) {
        return SlotPattern{1 % count, 0, true};
    }
    if (auto number = integer(value)) {
        return SlotPattern{0, residue(*number, count), true};
    }
    auto multiply = value.getDefiningOp<arith::MulIOp>();
    if (!multiply) {
        return std::nullopt;
    }
    Value scale = multiply.getLhs() == induction ? multiply.getRhs() :
                  multiply.getRhs() == induction ? multiply.getLhs() : Value{};
    if (auto number = integer(scale)) {
        return SlotPattern{residue(*number, count), 0, false};
    }
    return std::nullopt;
}
std::optional<SlotPattern> affine(Value value, Value induction, uint64_t count)
{
    if (auto direct = term(value, induction, count)) {
        return direct;
    }
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
        auto left = term(add.getLhs(), induction, count), right = term(add.getRhs(), induction, count);
        if (left && right) {
            // count <= INT64_MAX, so the sum of two residues fits uint64_t.
            return SlotPattern{(left->stride + right->stride) % count,
                               (left->offset + right->offset) % count, false};
        }
    }
    return std::nullopt;
}
} // namespace

std::optional<SlotPattern> matchSlot(Value slot, Value induction, uint64_t count)
{
    if (!count || count > static_cast<uint64_t>(INT64_MAX)) {
        return std::nullopt;
    }
    if (auto number = integer(slot)) {
        if (*number >= 0 && static_cast<uint64_t>(*number) < count) {
            return SlotPattern{0, static_cast<uint64_t>(*number), true};
        }
        return std::nullopt;
    }
    Value numerator, divisor;
    if (auto rem = slot.getDefiningOp<arith::RemUIOp>()) {
        numerator = rem.getLhs();
        divisor = rem.getRhs();
    } else if (auto rem = slot.getDefiningOp<arith::RemSIOp>()) {
        numerator = rem.getLhs();
        divisor = rem.getRhs();
    } else {
        return std::nullopt;
    }
    auto modulus = integer(divisor);
    if (!modulus || *modulus <= 0 || static_cast<uint64_t>(*modulus) != count) {
        return std::nullopt;
    }
    auto pattern = affine(numerator, induction, count);
    // Only the nonnegative canonical induction is immediately certified.
    // Signed/unsigned constants and arithmetic require their own semantic check.
    if (pattern && numerator != induction) {
        pattern->arithmeticProven = false;
    }
    return pattern;
}

std::optional<std::pair<uint64_t, uint64_t>> scalarAtom(const SyncStorageEffect& effect, uint64_t bytes)
{
    Value operand, offset;
    if (auto get = dyn_cast<TGetValOp>(effect.phase->elementOp); get && effect.mode == SyncAccessMode::Read) {
        operand = get.getSrc();
        offset = get.getOffset();
    } else if (auto set = dyn_cast<TSetValOp>(effect.phase->elementOp);
               set && effect.mode == SyncAccessMode::Write) {
        operand = set.getDst();
        offset = set.getOffset();
    } else {
        return std::nullopt;
    }
    auto type = dyn_cast<TileBufType>(operand.getType());
    auto number = integer(offset);
    if (!type || operand != effect.memory->baseBuffer || !number || *number < 0 ||
        !type.getElementType().isIntOrFloat() ||
        type.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox) ||
        type.getCompactModeI32() == static_cast<int32_t>(CompactMode::RowPlusOne)) {
        return std::nullopt;
    }
    const auto bits = type.getElementType().getIntOrFloatBitWidth();
    if (!bits || bits % 8 != 0 || static_cast<uint64_t>(*number) >= bytes / (bits / 8)) {
        return std::nullopt;
    }
    const auto begin = static_cast<uint64_t>(*number) * (bits / 8);
    return std::make_pair(begin, begin + bits / 8);
}
} // namespace mlir::pto::frontiersynch::detail
