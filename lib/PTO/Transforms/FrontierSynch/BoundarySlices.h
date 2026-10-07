// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_BOUNDARYSLICES_H
#define PTO_FRONTIERSYNCH_BOUNDARYSLICES_H
#include "PhaseNormalization.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingRegional.h"
namespace mlir::pto::frontiersynch {
struct BoundarySlice {
    PeriodicSlice interval;
    DenseMap<Value, RegionExpressions::Id> bindings;
    // Certified bound on max(end-begin,0), independent of runtime parameters.
    std::optional<uint64_t> maximumLength;
};
// Partition only this loop's changing Boolean atoms. Child-local predicates
// remain child obligations; outer AND/OR expressions need not themselves be
// comparisons. All cuts and bindings use original absolute loop ordinals.
std::optional<std::vector<BoundarySlice>> collectBoundarySlices(scf::ForOp loop,
    const PhaseIndex& index, RegionExpressions& arena, RegionExpressions::Id trips,
    const DenseMap<Value, RegionExpressions::Id>& inherited, std::string& error);
// Evaluate a Boolean formula under certified atom bindings. nullopt means an
// unbound operand still depends on the repeated loop, not an unknown truth value.
std::optional<RegionExpressions::Id> boundaryGuard(Value value, PhaseNormalization& normalizer,
    const DenseMap<Value, RegionExpressions::Id>& bindings, RegionExpressions& arena);
} // namespace mlir::pto::frontiersynch
#endif
