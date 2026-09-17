// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCACCUMULATORORDERING_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCACCUMULATORORDERING_H
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncSlotMapping.h"
#include <array>
#include <optional>

namespace mlir::pto {
// Access ordering, NOT completion: the A2/A3 ordinary Mmad K-accumulation
// contract permits omitting PIPE_M when (m/16)*(n/16) >= 10. PTO lowers these
// two operations through TMATMUL[_ACC] to mad with actual valid dimensions.
// Source: asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/
//         mmad_compute/Mmad.html
// Keep modes, precision and layouts narrow; no UnitFlag or phase credit.
struct SyncAccumulatorOrder {
  Value destination;
  Type destinationType;
  std::array<uint64_t, 4> signature; // absolute ACC base, M, N, K
};
inline std::optional<SyncAccumulatorOrder> syncAccumulatorOrder(mlir::Operation *op) {
  Value lhs, rhs, dst;
  if (auto mm = dyn_cast<TMatmulOp>(op)) {
    if (mm.getAccPhase() != AccPhase::Unspecified) return {};
    lhs = mm.getLhs(); rhs = mm.getRhs(); dst = mm.getDst();
  } else if (auto mm = dyn_cast<TMatmulAccOp>(op)) {
    if (mm.getAccPhase() != AccPhase::Unspecified || mm.getAccIn() != mm.getDst()) return {};
    lhs = mm.getLhs(); rhs = mm.getRhs(); dst = mm.getDst();
  } else return {};
  auto dimensions = [](Value value) -> std::optional<std::array<uint64_t, 2>> {
    auto alloc = value.getDefiningOp<AllocTileOp>();
    auto type = dyn_cast<TileBufType>(value.getType());
    if (!alloc || !type || type.getShape().size() != 2 || type.getValidShape().size() != 2) return {};
    std::array<uint64_t, 2> result{};
    for (unsigned i = 0; i < 2; ++i) {
      Value explicitSize = i ? alloc.getValidCol() : alloc.getValidRow();
      auto size = explicitSize ? SyncSlotMapping::literal(explicitSize) :
          (type.getValidShape()[i] > 0 ? std::optional<uint64_t>(type.getValidShape()[i]) : std::nullopt);
      if (!size || !*size || *size > 4095 || *size % 16 || type.getShape()[i] != int64_t(*size)) return {};
      result[i] = *size;
    }
    return result;
  };
  auto a = dimensions(lhs), b = dimensions(rhs), c = dimensions(dst);
  auto at = dyn_cast<TileBufType>(lhs.getType()), bt = dyn_cast<TileBufType>(rhs.getType());
  auto ct = dyn_cast<TileBufType>(dst.getType());
  if (!a || !b || !c || !at.getElementType().isF16() || !bt.getElementType().isF16() ||
      !ct.getElementType().isF32() || (*a)[1] != (*b)[0] || (*a)[0] != (*c)[0] ||
      (*b)[1] != (*c)[1] || ((*a)[0] / 16) * ((*b)[1] / 16) < 10) return {};
  auto alloc = dst.getDefiningOp<AllocTileOp>();
  auto base = alloc.getAddr() ? SyncSlotMapping::literal(alloc.getAddr()) : std::nullopt;
  if (!base) return {};
  return SyncAccumulatorOrder{dst, dst.getType(), {*base, (*a)[0], (*b)[1], (*a)[1]}};
}
} // namespace mlir::pto
#endif
