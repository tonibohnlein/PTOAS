// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Symbolic physical geometry shared by overlap and dependence analysis.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCACCESSREGION_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCACCESSREGION_H
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>
namespace mlir::pto {
struct SyncIterationDomain {
    Value induction;
    Value lower;
    Value upper;
    Value step;
};
// Coordinates d_i range over [0, extents[i]). byteOffset maps each element
// to its first byte; elementBytes consecutive bytes are accessed there.
// Symbols refer to unchanged SSA values. A null base denotes an absolute
// address in the effect's memory space, so allocation roots do not hide reuse.
struct SyncAccessRegion {
    Value base;
    AffineExpr byteOffset;
    SmallVector<AffineExpr> extents;
    SmallVector<Value> symbols;
    unsigned elementBytes = 0;
    // Original counted-loop domains, outermost first. No trip-count unfolding.
    SmallVector<SyncIterationDomain> iterations;
    bool empty() const;
};
struct SyncStorageEffect;
class SyncInput;
// Describe the operand's valid buffer region at a program point. This says
// where descriptor coordinates map, not which bytes an instruction accesses.
std::optional<SyncAccessRegion> resolveBufferRegion(const SyncInput& input, Value operand, Operation* at);
// Compose an exact shared access selection with the operand's descriptor map.
// Invalid, unknown or unavailable contracts return no region.
std::optional<SyncAccessRegion> resolveSelectedRegion(const SyncInput& input, Value operand,
                                                     Operation* at, DictionaryAttr contract);
// Equal symbolic terms cancel before comparison. Unknown is not disjoint.
bool regionsProvablyDisjoint(const SyncAccessRegion& a, const SyncAccessRegion& b);
// Only function-entry GM pointers establish invariant relative coordinates.
bool isCanonicalGMBase(Value base);
} // namespace mlir::pto
#endif
