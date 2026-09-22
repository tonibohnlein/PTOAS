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
#include "mlir/Interfaces/CastInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

namespace mlir::pto {
// Finite ORIGINAL scalar evolution, independent of synchronization or payload
// opcodes. Checked nonnegative arithmetic and dialect scalar folding preserve
// integer widths and conversion obligations. Unknown expressions retain the
// ordinary conservative alias contract.
struct SyncSlotMapping {
  unsigned period = 1;
  std::vector<llvm::DenseMap<Value, uint64_t>> values;

  static std::optional<uint64_t> literal(Value value) {
    auto op = value.getDefiningOp<arith::ConstantOp>();
    if (!op) return {};
    auto attr = dyn_cast<IntegerAttr>(op.getValue());
    if (!attr || attr.getValue().getBitWidth() > 64 ||
        (attr.getValue().getBitWidth() != 1 && attr.getValue().isNegative())) {
      return {};
    }
    return attr.getValue().getZExtValue();
  }
  // Only for immutable UNSEEDED evaluation. Failed facts must not leak into
  // the separate residue valuations, where an unknown slot can become known.
  struct ConstantCache {
    llvm::DenseMap<Value, uint64_t> known;
    llvm::DenseSet<Value> notConstant;
    uint64_t evaluations = 0;
    void clear() { known.clear(); notConstant.clear(); evaluations = 0; }
  };
  static std::optional<uint64_t> evaluateConstant(Value value, ConstantCache &cache) {
    return evaluateImpl(value, cache.known, &cache.notConstant, &cache.evaluations);
  }
  static std::optional<uint64_t> evaluate(Value value, llvm::DenseMap<Value, uint64_t> &known) {
    return evaluateImpl(value, known, nullptr, nullptr);
  }
private:
  static std::optional<uint64_t> evaluateImpl(Value value, llvm::DenseMap<Value, uint64_t> &known,
                                             llvm::DenseSet<Value> *failed, uint64_t *evaluations) {
    auto found = known.find(value);
    if (found != known.end()) return found->second;
    if (failed && failed->contains(value)) return {};
    if (evaluations) ++*evaluations;
    const auto result = [&]() -> std::optional<uint64_t> {
      if (auto constant = literal(value)) {
        return constant;
      }
      auto *op = value.getDefiningOp();
      if (!op) return {};
      unsigned width = isa<IndexType>(value.getType()) ? 64 :
          (isa<IntegerType>(value.getType()) ? cast<IntegerType>(value.getType()).getWidth() : 0);
      if (!width || width > 64) return {};
      const uint64_t maximum = (uint64_t(1) << (width - 1)) - 1;
      std::optional<uint64_t> result;
      if (auto cast = dyn_cast<arith::IndexCastOp>(op)) {
        result = evaluateImpl(cast.getIn(), known, failed, evaluations);
        if (cast.getIn().getType().isInteger(1) && result == std::optional<uint64_t>(1)) {
          return {}; // signed widening of i1 true is -1, not a positive address
        }
      } else if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
        // The input must already be a known nonnegative signed value. Both
        // extensions preserve it; truncation is admitted only if it fits the
        // destination's nonnegative range (checked below). Never strip a lossy
        // conversion or reinterpret a negative literal as a physical address.
        result = evaluateImpl(op->getOperand(0), known, failed, evaluations);
        if (isa<arith::ExtSIOp>(op) && op->getOperand(0).getType().isInteger(1) &&
            result == std::optional<uint64_t>(1)) {
          return {};
        }
      } else if (isa<arith::AddIOp, arith::MulIOp, arith::RemSIOp, arith::RemUIOp>(op)) {
        // Stop on an unqualified operand. Evaluating both children eagerly can
        // revisit a shared unknown scalar DAG exponentially (x = add(y, y)).
        auto a = evaluateImpl(op->getOperand(0), known, failed, evaluations);
        if (!a) return {};
        auto b = evaluateImpl(op->getOperand(1), known, failed, evaluations);
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
      if (!result && op->getName().getDialectNamespace() == "arith" &&
          !op->getNumRegions() && op->getNumResults() == 1) {
        // Let the owning dialect define scalar operations, rather than adding
        // selector spellings here. Keep the checked arithmetic/conversions
        // above authoritative, including their nonoverflow premise.
        if (isa<arith::AddIOp, arith::MulIOp, arith::RemSIOp, arith::RemUIOp,
                arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
          return {};
        }
        if (auto flags = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(op)) {
          // The dialect folder does not check poison-producing overflow flags.
          // Such operations need a range proof before using its folded value.
          if (flags.hasNoSignedWrap() || flags.hasNoUnsignedWrap()) {
            return {};
          }
        }
        SmallVector<Attribute> operands;
        for (Value operand : op->getOperands()) {
          if (!isa<IndexType, IntegerType>(operand.getType())) {
            return {};
          }
          auto integer = evaluateImpl(operand, known, failed, evaluations);
          if (!integer) {
            return {};
          }
          operands.push_back(IntegerAttr::get(operand.getType(), *integer));
        }
        // Folding may mutate its operation. Never mutate the original program
        // while deriving immutable physical facts.
        auto *copy = op->clone();
        SmallVector<OpFoldResult> folded;
        if (succeeded(copy->fold(operands, folded)) && folded.size() == 1) {
          if (auto attr = dyn_cast_if_present<IntegerAttr>(folded.front().dyn_cast<Attribute>())) {
            if (attr.getValue().getBitWidth() <= 64 &&
                (attr.getValue().getBitWidth() == 1 || !attr.getValue().isNegative())) {
              result = attr.getValue().getZExtValue();
            }
          } else if (auto replacement = folded.front().dyn_cast<Value>()) {
            // A replacement result owned by the clone is not an original fact.
            if (replacement.getDefiningOp() != copy) {
              result = evaluateImpl(replacement, known, failed, evaluations);
            }
          }
        }
        copy->destroy();
      }
      if (!result || *result > (width == 1 ? 1 : maximum)) {
        return {};
      }
      return result;
    }();
    if (result) known[value] = *result;
    else if (failed) failed->insert(value);
    return result;
  }
public:

  // Original scalar identities from the dialect's partial folder, including
  // identity arithmetic and lossless cast round trips. This rewrites no IR and
  // does not infer an equality from sampled values.
  static Value canonicalValue(Value value, ConstantCache &constants) {
    llvm::DenseSet<Value> seen;
    while (seen.insert(value).second) {
      auto *op = value.getDefiningOp();
      if (!op || op->getName().getDialectNamespace() != "arith" ||
          op->getNumRegions() || op->getNumResults() != 1) {
        break;
      }
      // Equal-width integer/index casts preserve the represented bits. This
      // supplies the width fact used by index-cast canonicalization without
      // running a rewriting pass over the original IR. Narrowing is not an
      // identity. Index uses this analysis's existing 64-bit target contract.
      if (isa<CastOpInterface>(op) && op->getNumOperands() == 1) {
        const auto width = [](Type type) -> unsigned {
          if (isa<IndexType>(type)) {
            return 64;
          }
          if (auto integer = dyn_cast<IntegerType>(type)) {
            return integer.getWidth();
          }
          return 0;
        };
        const unsigned sourceWidth = width(op->getOperand(0).getType());
        if (sourceWidth && sourceWidth == width(value.getType())) {
          value = op->getOperand(0);
          continue;
        }
      }
      SmallVector<Attribute> operands;
      for (Value operand : op->getOperands()) {
        const auto integer = evaluateConstant(operand, constants);
        operands.push_back(integer ? IntegerAttr::get(operand.getType(), *integer) : Attribute{});
      }
      auto *copy = op->clone();
      SmallVector<OpFoldResult> folded;
      Value next;
      if (succeeded(copy->fold(operands, folded)) && folded.size() == 1) {
        next = folded.front().dyn_cast<Value>();
        if (next && next.getDefiningOp() == copy) {
          next = {};
        }
      }
      copy->destroy();
      if (!next || next == value) {
        break;
      }
      value = next;
    }
    return value;
  }

  // Exploration is compiler work, not a physical event-pool limit. Each query
  // follows one observation and its transitive backedge dependencies; unrelated
  // carried values never enter the state. Closing the entire dependency state
  // proves periodicity, unlike observing a repeated address in a finite sample.
  static constexpr unsigned ExplorationLimit = 256;
  static std::optional<SyncSlotMapping> derive(
      scf::ForOp loop, Value observation, unsigned maximumStates = ExplorationLimit) {
    if (!observation || !maximumStates || !isa<IndexType>(loop.getInductionVar().getType()) ||
        loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp")) {
      return {};
    }
    ConstantCache constants;
    const auto lower = evaluateConstant(loop.getLowerBound(), constants);
    const auto step = evaluateConstant(loop.getStep(), constants);
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    std::vector<unsigned> carried;
    struct Projection { Value value; uint64_t modulus, mask; bool bitwise; };
    std::vector<Projection> projections;
    llvm::DenseSet<Value> visited;
    std::function<bool(Value)> collect = [&](Value value) {
      value = canonicalValue(value, constants);
      if (!visited.insert(value).second || evaluateConstant(value, constants)) {
        return true;
      }
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        if (arg.getOwner() != loop.getBody() || !arg.getArgNumber()) {
          return false;
        }
        const unsigned index = arg.getArgNumber() - 1;
        carried.push_back(index);
        return collect(yield.getOperand(index));
      }
      auto *op = value.getDefiningOp();
      if (!op || op->getName().getDialectNamespace() != "arith" ||
          op->getNumRegions() || op->getNumResults() != 1) {
        return false;
      }
      // Congruence projections of the actual nonnegative induction sequence.
      // Other uses of the unbounded IV remain unknown. A masked projection
      // needs only the bits through its highest selected bit.
      if (op->getNumOperands() == 2 && lower && step && *step &&
          (isa<arith::RemSIOp, arith::RemUIOp>(op) || isa<arith::AndIOp>(op))) {
        Value variable = canonicalValue(op->getOperand(0), constants);
        Value constant = canonicalValue(op->getOperand(1), constants);
        if (isa<arith::AndIOp>(op) && constant == loop.getInductionVar()) {
          std::swap(variable, constant);
        }
        const auto bound = evaluateConstant(constant, constants);
        if (variable == loop.getInductionVar() && bound && *bound &&
            *bound <= uint64_t(std::numeric_limits<int64_t>::max())) {
          uint64_t modulus = *bound;
          const bool bitwise = isa<arith::AndIOp>(op);
          if (bitwise) {
            modulus = 1;
            while (modulus <= *bound) {
              modulus <<= 1;
            }
          }
          projections.push_back({value, modulus, *bound, bitwise});
          return true;
        }
      }
      for (Value operand : op->getOperands()) {
        if (!collect(operand)) {
          return false;
        }
      }
      return true;
    };
    if (!collect(observation)) {
      return {};
    }
    std::vector<uint64_t> initial;
    for (unsigned index : carried) {
      const auto value = evaluateConstant(loop.getInitArgs()[index], constants);
      if (!value) {
        return {};
      }
      initial.push_back(*value);
    }
    for (const auto &projection : projections) {
      initial.push_back(*lower % projection.modulus);
    }
    SyncSlotMapping result;
    auto state = initial;
    for (unsigned visit = 0; visit < maximumStates; ++visit) {
      llvm::DenseMap<Value, uint64_t> values;
      for (unsigned i = 0; i < carried.size(); ++i) {
        values[loop.getRegionIterArgs()[carried[i]]] = state[i];
      }
      for (unsigned i = 0; i < projections.size(); ++i) {
        const auto &p = projections[i];
        const auto phase = state[carried.size() + i];
        values[p.value] = p.bitwise ? phase & p.mask : phase;
      }
      if (!evaluate(observation, values)) {
        return {};
      }
      std::vector<uint64_t> next;
      for (unsigned index : carried) {
        auto value = evaluate(yield.getOperand(index), values);
        if (!value) {
          return {};
        }
        next.push_back(*value);
      }
      for (unsigned i = 0; i < projections.size(); ++i) {
        const auto modulus = projections[i].modulus;
        // Both summands are below 2^63, so the unsigned addition fits.
        next.push_back((state[carried.size() + i] + *step % modulus) % modulus);
      }
      result.values.push_back(std::move(values));
      if (next == initial) {
        result.period = result.values.size();
        return result;
      }
      state = std::move(next);
    }
    return {};
  }
};
} // namespace mlir::pto
#endif
