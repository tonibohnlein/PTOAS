// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/SyncAddressAnalysis.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <limits>
using namespace mlir;
using namespace mlir::pto;

SyncAddressEvaluator::SyncAddressEvaluator(func::FuncOp function) {
  auto size = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
  index64 = !size.isScalable() && size.getFixedValue() == 64;
}
std::optional<unsigned> SyncAddressEvaluator::width(Type type) const {
  if (type.isIndex()) return index64 ? std::optional<unsigned>(64) : std::nullopt;
  if (auto integer = dyn_cast<IntegerType>(type))
    if (integer.getWidth() <= 4096) return integer.getWidth();
  return {};
}
std::optional<llvm::APInt> SyncAddressEvaluator::evaluate(Value value) {
  return evaluate(value, 0);
}
std::optional<llvm::APInt> SyncAddressEvaluator::evaluate(Value value, unsigned depth) {
  if (!value) return {};
  if (auto found = cache.find(value); found != cache.end()) return found->second;
  if (visited >= 100000) return {}; // bounded cache as well as bounded work
  ++visited;
  std::optional<llvm::APInt> answer;
  auto bits = width(value.getType());
  Operation *op = value.getDefiningOp();
  if (bits && op && depth < 128) {
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      if (auto attr = dyn_cast<IntegerAttr>(constant.getValue()))
        answer = attr.getValue().sextOrTrunc(*bits);
    } else if (isa<arith::IndexCastOp, arith::IndexCastUIOp,
                   arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
      auto input = evaluate(op->getOperand(0), depth + 1);
      if (input) {
        if (isa<arith::ExtUIOp, arith::IndexCastUIOp>(op)) answer = input->zextOrTrunc(*bits);
        else answer = input->sextOrTrunc(*bits);
      }
    }
  }
  // Depth/budget imprecision can only enlarge aliasing, never prove an address.
  cache.try_emplace(value, answer);
  return answer;
}
std::optional<int64_t> SyncAddressEvaluator::signedValue(Value value) {
  auto bits = evaluate(value);
  if (!bits || !bits->isSignedIntN(64)) return {};
  return bits->getSExtValue();
}
uint64_t mlir::pto::getSyncTileFootprintBytes(TileBufType type) {
  if (!type) return 0;
  auto shape = type.getShape();
  uint64_t bytes = getPTOStorageElemByteSize(type.getElementType());
  if (!bytes) return 0;
  auto multiply = [](uint64_t a, uint64_t b) -> uint64_t {
    return b && a <= uint64_t(INT64_MAX) / b ? a * b : 0;
  };
  if (type.getCompactModeI32() == int32_t(CompactMode::RowPlusOne)) {
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) return 0;
    bool row = type.getBLayoutValueI32() == int32_t(BLayout::RowMajor);
    uint64_t major = shape[row ? 0 : 1], minor = shape[row ? 1 : 0];
    // ((major - 1) * (minor + 1) + minor) == major * (minor + 1) - 1.
    uint64_t elements = multiply(major, minor + 1);
    return elements ? multiply(elements - 1, bytes) : 0;
  }
  for (int64_t dim : shape) {
    if (dim <= 0 || !(bytes = multiply(bytes, uint64_t(dim)))) return 0;
  }
  return bytes;
}
SyncAddressAdmission mlir::pto::qualifySyncPhysicalAddresses(func::FuncOp function) {
  SyncAddressAdmission result;
  SyncAddressEvaluator evaluator(function);
  auto reject = [&](const char *reason) {
    result = {SyncAddressAdmission::Rejected, reason};
    return WalkResult::interrupt();
  };
  auto address = [&](Value value) -> std::optional<uint64_t> {
    auto bits = evaluator.evaluate(value);
    if (!bits) {
      result = {SyncAddressAdmission::Conservative, "unresolved explicit physical address"};
      return {};
    }
    if (!bits->isSignedIntN(64) || bits->isNegative()) {
      result = {SyncAddressAdmission::Rejected, "physical address is outside nonnegative i64 range"};
      return {};
    }
    return bits->getZExtValue();
  };
  function.walk([&](Operation *op) {
    if (auto alloc = dyn_cast<AllocTileOp>(op)) {
      uint64_t bytes = getSyncTileFootprintBytes(alloc.getResult().getType());
      if (!bytes) return reject("unrepresentable physical tile footprint");
      if (auto value = alloc.getAddr()) {
        auto base = address(value);
        if (result.status == SyncAddressAdmission::Rejected) return WalkResult::interrupt();
        if (base && *base > uint64_t(INT64_MAX) - bytes)
          return reject("physical tile interval exceeds nonnegative i64 range");
      }
    } else if (auto alloc = dyn_cast<AllocMultiTileOp>(op)) {
      auto type = alloc.getResult().getType();
      auto layout = getPTOStaticMultiTileSlotLayout(type.getSlotType());
      if (failed(layout)) return reject("unrepresentable multi-tile physical layout");
      if (auto planned = alloc->getAttrOfType<DenseI64ArrayAttr>(kPtoMultiBufferAddrsAttrName)) {
        if (alloc.getAddr() || planned.size() != type.getCount())
          return reject("inconsistent planned multi-tile address table");
        for (int64_t base : planned.asArrayRef()) {
          if (base < 0 || failed(getPTOStaticMultiTileSlotAddress(*layout, uint64_t(base), 0)))
            return reject("planned multi-tile interval exceeds nonnegative i64 range");
          if (uint64_t(base) % layout->alignmentBytes)
            return reject("unaligned planned multi-tile physical interval");
        }
      } else if (auto value = alloc.getAddr()) {
        if (!type.getCount() || failed(getPTOStaticMultiTileSlotOffset(*layout, type.getCount() - 1)))
          return reject("multi-tile slot offset exceeds nonnegative i64 range");
        auto base = address(value);
        if (result.status == SyncAddressAdmission::Rejected) return WalkResult::interrupt();
        if (base && failed(getPTOStaticMultiTileSlotAddress(*layout, *base, type.getCount() - 1)))
          return reject("multi-tile physical interval exceeds nonnegative i64 range");
        if (base && *base % layout->alignmentBytes)
          return reject("unaligned multi-tile physical base");
      } else return reject("multi-tile physical addresses are unavailable");
    }
    return WalkResult::advance();
  });
  return result;
}
