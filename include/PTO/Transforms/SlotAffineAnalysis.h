// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SlotAffineAnalysis.h - Multi-buffer slot affine compare --*- C++ -*-===//
//
// Small affine helper used by the multi-buffer sync path. Both InsertSync
// and GraphSyncSolver consume it to decide, for two `pto.multi_tile_get`
// slot-index SSA expressions, whether they are provably equal modulo N,
// provably disjoint modulo N, or indeterminate. The result lets sync
// shrink event-id count or skip same-iter forward syncs entirely when
// producer and consumer touch different slots in every iteration.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_SLOTAFFINEANALYSIS_H
#define PTO_TRANSFORMS_SLOTAFFINEANALYSIS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {

/// Three-valued relation between two multi-buffer slot SSA expressions
/// taken modulo `N`. Anything we cannot prove statically degrades to
/// `kUnknown`, which the callers treat conservatively (i.e. fall back to
/// the existing all-slots-may-overlap path).
enum class SlotRelation {
  kEqual,    // a(iv) == b(iv)  (mod N) for every iv
  kDisjoint, // a(iv) != b(iv)  (mod N) for every iv
  kUnknown,  // can neither prove equal nor disjoint
};

struct SlotOrdinalPair {
  std::uint32_t first = 0;
  std::uint32_t second = 0;

  bool operator==(const SlotOrdinalPair &other) const {
    return first == other.first && second == other.second;
  }
};

/// Return the slot SSA value carried by `pto.multi_tile_get`. Returns null
/// if the chain does not pass through a multi_tile_get.
mlir::Value findMultiTileSlotExpr(mlir::Value v);

/// Compare two slot SSA expressions modulo `N`. The analysis is
/// intentionally narrow: it accepts the forms commonly produced by
/// frontends and lowerings (`iv % N`, `(iv + c) % N`, `c`, and same-SSA
/// equality) and bails to `kUnknown` for anything else.
/// Examples (all with N == 2):
///   compareSlotSSA(%iv % 2, %iv % 2)         -> kEqual
///   compareSlotSSA((%iv + 1) % 2, %iv % 2)   -> kDisjoint
///   compareSlotSSA((%iv + 3) % 2, %iv % 2)   -> kDisjoint  // 3 % 2 == 1
///   compareSlotSSA((%iv + 2) % 2, %iv % 2)   -> kEqual     // 2 % 2 == 0
///   compareSlotSSA(%iv % 2, %j % 2)          -> kUnknown   // diff symbols
///   compareSlotSSA(arith.constant 0, arith.constant 1) -> kDisjoint
SlotRelation compareSlotSSA(mlir::Value a, mlir::Value b, uint32_t N);

/// Compare `a(symbol)` with `b(symbol + rhsSymbolOffset)` modulo `N`.
/// Returns `kUnknown` when the canonical symbolic form does not use
/// `shiftedSymbol`. This is used for fixed-distance loop recurrences.
SlotRelation compareSlotSSAWithOffset(mlir::Value a, mlir::Value b, uint32_t N,
                                      mlir::Value shiftedSymbol,
                                      int64_t rhsSymbolOffset);

/// Enumerate the exact selected ordinal pairs over every residue modulo `N`.
/// Returns nullopt unless both expressions are normalized constants or
/// modulo-N forms whose symbolic relationship is fully known. This is the
/// stronger relation required when physical-slot provenance is recorded.
std::optional<llvm::SmallVector<SlotOrdinalPair, 4>>
enumerateSlotSSAOrdinalPairs(mlir::Value a, mlir::Value b, std::uint32_t N,
                             mlir::Value shiftedSymbol = {},
                             std::int64_t rhsSymbolOffset = 0);

/// Enumerate selected ordinal pairs only for the supplied residues of the
/// common loop symbol. This is the phase-sensitive variant used when a
/// periodic guard restricts which loop iterations can execute both accesses.
std::optional<llvm::SmallVector<SlotOrdinalPair, 4>>
enumerateSlotSSAOrdinalPairsForResidues(
    mlir::Value a, mlir::Value b, std::uint32_t N,
    llvm::ArrayRef<std::uint32_t> symbolResidues,
    mlir::Value shiftedSymbol = {}, std::int64_t rhsSymbolOffset = 0);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_SLOTAFFINEANALYSIS_H
