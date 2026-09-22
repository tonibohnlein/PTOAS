// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCACCUMULATORORDERING_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCACCUMULATORORDERING_H

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/SyncTileDescriptorState.h"
#include <limits>

namespace mlir::pto {
// Access ordering, NOT completion: ordinary A2/A3 Mmad K-accumulation needs
// PIPE_M only below (m/16)*(n/16) == 10. Supported dtypes come from the ordinary
// operation/lowering contract, not a synchronization-specific whitelist.
// Source: asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/
//         mmad_compute/Mmad.html (master ee1721f1b399, checked 2026-09-22).
// No UnitFlag, operand-release or FIX-completion credit follows from this rule.
struct SyncAccumulatorOrder {
  TileBufType destinationType;
  std::array<uint64_t, 3> signature; // physical ACC base, effective M, N (not K)

  bool compatible(const SyncAccumulatorOrder &other) const {
    return signature == other.signature &&
        destinationType.getShape() == other.destinationType.getShape() &&
        destinationType.getElementType() == other.destinationType.getElementType() &&
        destinationType.getConfigAttr() == other.destinationType.getConfigAttr();
  }
};

// Coordinates are may-footprints, so a view also needs a layout/address proof.
// Address-preserving views inherit the shared origin. A boxed subview whose
// offset/stride the translator widened is unknown unless it is an identity view.
inline bool syncPreservesTileAddress(Value value, Value root) {
  SyncSlotMapping::ConstantCache constants;
  Value current = value;
  while (current != root) {
    auto *definition = current.getDefiningOp();
    if (auto reshape = dyn_cast_or_null<TReshapeOp>(definition)) {
      current = reshape.getSrc();
    } else if (auto bitcast = dyn_cast_or_null<BitcastOp>(definition)) {
      current = bitcast.getSrc();
    } else if (auto view = dyn_cast_or_null<SubViewOp>(definition)) {
      const auto type = cast<TileBufType>(current.getType());
      const auto parent = cast<TileBufType>(view.getSource().getType());
      if (type.getShape() != parent.getShape() || type.getConfigAttr() != parent.getConfigAttr()) {
        return false;
      }
      for (Value offset : view.getOffsets()) {
        if (SyncSlotMapping::evaluateConstant(offset, constants) != std::optional<uint64_t>(0)) {
          return false;
        }
      }
      current = view.getSource();
    } else {
      return false;
    }
  }
  return true;
}

inline std::optional<SyncStorageCoordinates> syncExactTileStorage(Value value, const Buffer2MemInfoMap &buffers) {
  const auto type = dyn_cast<TileBufType>(value.getType());
  auto found = buffers.find(value);
  if (!type || found == buffers.end() || found->second.size() != 1) {
    return {};
  }
  const auto &memory = *found->second.front();
  const auto storage = MemoryDependentAnalyzer::storageCoordinates(memory);
  if (!storage || !syncPreservesTileAddress(value, memory.rootBuffer)) {
    return {};
  }
  // A queue envelope or a widened view is not the precise tile allocation.
  // Rank/dimension and overflow checks precede each footprint multiplication.
  uint64_t bytes = getPTOStorageElemByteSize(type.getElementType());
  if (!bytes || type.getShape().size() != 2) {
    return {};
  }
  for (int64_t dim : type.getShape()) {
    if (dim <= 0 || uint64_t(dim) > std::numeric_limits<uint64_t>::max() / bytes) {
      return {};
    }
    bytes *= uint64_t(dim);
  }
  if (bytes != storage->size) {
    return {};
  }
  return storage;
}

inline std::optional<SyncAccumulatorOrder> syncAccumulatorOrder(
    mlir::Operation *op, const SyncTileDescriptorState &descriptors, const Buffer2MemInfoMap &buffers) {
  Value lhs, rhs, dst, input;
  if (auto mm = dyn_cast<TMatmulOp>(op)) {
    if (mm.getAccPhase() != AccPhase::Unspecified) {
      return {};
    }
    lhs = mm.getLhs(); rhs = mm.getRhs(); dst = mm.getDst();
  } else if (auto mm = dyn_cast<TMatmulAccOp>(op)) {
    if (mm.getAccPhase() != AccPhase::Unspecified) {
      return {};
    }
    lhs = mm.getLhs(); rhs = mm.getRhs(); dst = mm.getDst(); input = mm.getAccIn();
  } else {
    return {};
  }
  const auto a = descriptors.at(lhs, op), b = descriptors.at(rhs, op), c = descriptors.at(dst, op);
  const auto valid = [](const SyncTileDescriptorState::Shape &shape) {
    return shape && (*shape)[0] > 0 && (*shape)[0] <= 4095 && (*shape)[1] > 0 && (*shape)[1] <= 4095;
  };
  if (!valid(a) || !valid(b) || !valid(c) || (*a)[1] != (*b)[0] || (*a)[0] != (*c)[0] ||
      (*b)[1] != (*c)[1] || ((*a)[0] / 16) * ((*b)[1] / 16) < 10) {
    return {};
  }
  const auto type = dyn_cast<TileBufType>(dst.getType());
  const auto storage = syncExactTileStorage(dst, buffers);
  if (!type || !storage || !storage->absolute) {
    return {};
  }
  SyncAccumulatorOrder result{type, {storage->begin, (*c)[0], (*c)[1]}};
  if (input && input != dst) {
    const auto prior = syncExactTileStorage(input, buffers);
    const auto priorType = dyn_cast<TileBufType>(input.getType());
    const auto dims = descriptors.at(input, op);
    if (!prior || !priorType || !dims || !prior->absolute || prior->size != storage->size ||
        !result.compatible({priorType, {prior->begin, (*dims)[0], (*dims)[1]}})) {
      return {};
    }
  }
  return result;
}
} // namespace mlir::pto
#endif
