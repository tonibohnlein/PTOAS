// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCSLOTMAPPING_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCSLOTMAPPING_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

namespace mlir::pto {
// Finite ORIGINAL scalar evolution, independent of synchronization or payload
// opcodes. Only nonnegative, nonoverflowing arithmetic is admitted. Unsupported
// expressions retain the ordinary conservative alias contract.
struct SyncSlotMapping {
  unsigned period = 1;
  std::vector<llvm::DenseMap<Value, uint64_t>> values;

  static std::optional<uint64_t> literal(Value value) {
    auto op = value.getDefiningOp<arith::ConstantOp>();
    if (!op) return {};
    auto attr = dyn_cast<IntegerAttr>(op.getValue());
    if (!attr || attr.getValue().getBitWidth() > 64 || attr.getValue().isNegative()) return {};
    return attr.getValue().getZExtValue();
  }
  static std::optional<uint64_t> evaluate(Value value, llvm::DenseMap<Value, uint64_t> &known) {
    auto found = known.find(value);
    if (found != known.end()) return found->second;
    if (auto constant = literal(value)) return constant;
    auto *op = value.getDefiningOp();
    if (!op) return {};
    unsigned width = isa<IndexType>(value.getType()) ? 64 :
        (isa<IntegerType>(value.getType()) ? cast<IntegerType>(value.getType()).getWidth() : 0);
    if (!width || width > 64) return {};
    const uint64_t maximum = (uint64_t(1) << (width - 1)) - 1;
    std::optional<uint64_t> result;
    if (auto cast = dyn_cast<arith::IndexCastOp>(op)) {
      result = evaluate(cast.getIn(), known);
    } else if (isa<arith::AddIOp, arith::MulIOp, arith::RemSIOp, arith::RemUIOp>(op)) {
      // Stop on an unqualified operand. Evaluating both children eagerly can
      // revisit a shared unknown scalar DAG exponentially (x = add(y, y)).
      auto a = evaluate(op->getOperand(0), known);
      if (!a) return {};
      auto b = evaluate(op->getOperand(1), known);
      if (!b) return {};
      if (isa<arith::AddIOp>(op)) {
        if (*a > maximum || *b > maximum - *a) return {};
        result = *a + *b;
      } else if (isa<arith::MulIOp>(op)) {
        if (*b && *a > maximum / *b) return {};
        result = *a * *b;
      } else {
        if (!*b) return {};
        result = *a % *b;
      }
    }
    if (!result || *result > maximum) return {};
    known[value] = *result;
    return result;
  }

  static std::optional<SyncSlotMapping> derive(scf::ForOp loop, unsigned maximumPeriod) {
    if (loop.getInitArgs().empty() || literal(loop.getLowerBound()) != std::optional<uint64_t>(0) ||
        literal(loop.getStep()) != std::optional<uint64_t>(1) ||
        !isa<IndexType>(loop.getInductionVar().getType()) ||
        loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp")) return {};
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    SyncSlotMapping result;
    llvm::DenseMap<Value, uint64_t> initial;
    for (unsigned i = 0; i < loop.getInitArgs().size(); ++i) {
      Value arg = loop.getRegionIterArgs()[i], next = yield.getOperand(i);
      const auto init = literal(loop.getInitArgs()[i]);
      if (!init || !isa<IndexType>(arg.getType())) return {};
      auto *rem = next.getDefiningOp();
      if (!rem || !isa<arith::RemSIOp, arith::RemUIOp>(rem)) return {};
      auto modulus = literal(rem->getOperand(1));
      auto add = rem->getOperand(0).getDefiningOp<arith::AddIOp>();
      if (!modulus || !*modulus || !add || *init >= *modulus) return {};
      Value delta = add.getLhs() == arg ? add.getRhs() :
                    (add.getRhs() == arg ? add.getLhs() : Value{});
      auto step = delta ? literal(delta) : std::optional<uint64_t>{};
      if (!step || *step > uint64_t(std::numeric_limits<int64_t>::max()) - (*modulus - 1)) return {};
      const auto period = *modulus / std::gcd(*modulus, *step);
      const auto factor = period / std::gcd(uint64_t(result.period), period);
      if (!maximumPeriod || factor > maximumPeriod / result.period) return {};
      result.period *= unsigned(factor);
      initial[arg] = *init;
    }
    auto state = initial;
    for (unsigned residue = 0; residue < result.period; ++residue) {
      result.values.push_back(state);
      llvm::DenseMap<Value, uint64_t> next;
      for (unsigned i = 0; i < loop.getInitArgs().size(); ++i) {
        auto value = evaluate(yield.getOperand(i), state);
        if (!value) return {};
        next[loop.getRegionIterArgs()[i]] = *value;
      }
      state = std::move(next);
    }
    if (state != initial) return {};
    return result;
  }
};
} // namespace mlir::pto
#endif
