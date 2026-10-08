// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Affine machine congruences. Unlike ScalarEvolution's integer equalities,
// every coefficient here is interpreted modulo the supplied bit width.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCSCALARCONGRUENCE_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCSCALARCONGRUENCE_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/MapVector.h"
#include <functional>
#include <optional>
namespace mlir::pto::detail {
struct ScalarCongruence {
    explicit ScalarCongruence(unsigned bits) : constant(bits, 0) {}
    APInt constant;
    llvm::MapVector<Value, APInt> coefficients;
};
// The caller supplies the index width from the target data layout and decides
// which original SSA values may remain symbols (coordinates or entry values).
// No division, cast across widths, or unknown loop-local operation is expanded.
class ScalarCongruenceNormalizer {
public:
    using Symbol = std::function<bool(Value)>;
    ScalarCongruenceNormalizer(unsigned bits, Symbol permitted)
        : bits(bits), permitted(std::move(permitted)) {}
    std::optional<ScalarCongruence> value(Value current) { return resolve(current, 0); }
private:
    std::optional<ScalarCongruence> resolve(Value current, unsigned depth)
    {
        if (!current || !bits || bits > 64 || depth == 64 ||
            (!current.getType().isIndex() && !current.getType().isInteger(bits))) { return std::nullopt; }
        if (auto found = memo.find(current); found != memo.end()) { return found->second; }
        ScalarCongruence result(bits);
        APInt literal;
        if (matchPattern(current, m_ConstantInt(&literal))) {
            result.constant = literal.sextOrTrunc(bits); return result;
        }
        if (permitted(current)) {
            result.coefficients.insert({current, APInt(bits, 1)}); return result;
        }
        auto* op = current.getDefiningOp();
        if (!op || !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) { return std::nullopt; }
        auto left = resolve(op->getOperand(0), depth + 1);
        auto right = resolve(op->getOperand(1), depth + 1);
        if (!left || !right) { return memo[current] = std::nullopt; }
        if (isa<arith::MulIOp>(op)) {
            if (left->coefficients.empty()) { std::swap(left, right); }
            if (!right->coefficients.empty()) { return memo[current] = std::nullopt; }
            left->constant *= right->constant;
            for (auto& term : left->coefficients) { term.second *= right->constant; }
        } else {
            const bool subtract = isa<arith::SubIOp>(op);
            left->constant += subtract ? -right->constant : right->constant;
            for (const auto& term : right->coefficients) {
                auto [position, inserted] = left->coefficients.insert({term.first, APInt(bits, 0)});
                position->second += subtract ? -term.second : term.second;
            }
        }
        left->coefficients.remove_if([](const auto& term) { return term.second.isZero(); });
        return memo[current] = std::move(left);
    }
    unsigned bits;
    Symbol permitted;
    DenseMap<Value, std::optional<ScalarCongruence>> memo;
};
} // namespace mlir::pto::detail
#endif
