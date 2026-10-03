// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numerical minimum-excess partition of a selected finite one-way staircase.
// Target, available-mechanism, matching, cut and invocation premises belong to
// the caller; this helper neither recovers requirements nor certifies them.
#ifndef PTO_FRONTIERSYNCH_ONE_WAY_REPAIR_H
#define PTO_FRONTIERSYNCH_ONE_WAY_REPAIR_H
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <string>
namespace mlir::pto::frontiersynch {
struct OneWayCover {
    llvm::DynamicAPInt sourceRank, consumerRank; // One-based payload ranks.
};
struct OneWayBlock {
    std::size_t first, last; // Inclusive zero-based cover indices.
    unsigned id;
};
enum class OneWayRepairStatus { Ready, NoCapacity, Unsupported };
struct OneWayRepair {
    OneWayRepairStatus status = OneWayRepairStatus::Unsupported;
    llvm::SmallVector<OneWayBlock> blocks;
    llvm::DynamicAPInt excess = llvm::DynamicAPInt(0);
    std::string reason;
};
// Implements prop:one-way-excess / app:one-way-excess-optimization. Strictly
// increasing source/consumer ranks are validated against the explicit counts.
// A block publishes after its last source and acquires before its first
// consumer. IDs are distinct actual eligible values, including reserved holes.
// O(t*m) arithmetic/comparison work and backpointer storage, t=min(E,m),
// excluding encoded-integer costs and input qualification. Value ties choose
// the smaller last cut, recursively. Empty covers require no event ID.
OneWayRepair repairOneWay(llvm::ArrayRef<OneWayCover> covers,
    const llvm::DynamicAPInt& nP, const llvm::DynamicAPInt& nQ,
    llvm::ArrayRef<unsigned> eligibleIds);
} // namespace mlir::pto::frontiersynch
#endif
