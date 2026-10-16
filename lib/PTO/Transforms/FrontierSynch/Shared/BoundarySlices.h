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
#include "PTO/Transforms/FrontierSynch/Structured/GuardedRotatingRegional.h"
namespace mlir::pto::frontiersynch {
enum class BoundaryControlPolicy { RequirePartition, RetainChildControl };
struct BoundarySlice {
    PeriodicSlice interval;
    DenseMap<Value, RegionExpressions::Id> bindings;
    // Certified bound on max(end-begin,0), independent of runtime parameters.
    std::optional<uint64_t> maximumLength;
};
// Owned evidence from collectBoundarySlices: intervals are ordered, disjoint
// subsets of the original loop's visit domain. Child adapters must match both
// original endpoints before using this evidence; it is not a general alias fact.
struct BoundaryPartition {
    scf::ForOp loop;
    std::shared_ptr<RegionExpressions> expressions;
    std::vector<PeriodicSlice> intervals;
};
// Partition Boolean atoms using original absolute loop ordinals, including
// outer AND/OR expressions. RetainChildControl stops at proper nested loops
// and leaves unpartitioned predicates with their original child provider.
// Callers must prove that the resulting skeleton is used at most once before
// dispensing with control invariance.
std::optional<std::vector<BoundarySlice>> collectBoundarySlices(scf::ForOp loop,
    const PhaseIndex& index, RegionExpressions& arena, RegionExpressions::Id trips,
    const DenseMap<Value, RegionExpressions::Id>& inherited, std::string& error,
    BoundaryControlPolicy policy = BoundaryControlPolicy::RequirePartition);
// Evaluate a Boolean formula under certified atom bindings. nullopt means an
// unbound operand still depends on the repeated loop, not an unknown truth value.
std::optional<RegionExpressions::Id> boundaryGuard(Value value, PhaseNormalization& normalizer,
    const DenseMap<Value, RegionExpressions::Id>& bindings, RegionExpressions& arena);
} // namespace mlir::pto::frontiersynch
#endif
