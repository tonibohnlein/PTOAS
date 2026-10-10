// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Immutable shared input normalization, never a mathematical demand certificate.
#ifndef PTO_FRONTIERSYNCH_NORMALIZEDCONTROL_H
#define PTO_FRONTIERSYNCH_NORMALIZEDCONTROL_H
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include <memory>
namespace mlir::pto::frontiersynch {
enum class NormalizedControlKind { Leaf, Sequence, Conditional, RetainedLoop, ExpandedLoop };
struct NormalizedControlNode {
    NormalizedControlKind kind = NormalizedControlKind::Leaf;
    Operation* original = nullptr;
    SmallVector<FixedLoopCoordinate> fixedCoordinates;
    SmallVector<std::size_t> children;
};
// Nodes retain both branch arms and all scalar/control operations. Coordinates
// name original induction values; effects, prerequisites and cuts remain in the
// unchanged shared input/index. Backend rejection cannot consume this owner.
struct NormalizedControlDescription {
    // Present only when this is a distinct expanded alternative.
    std::shared_ptr<const NormalizedControlDescription> original;
    ArithmeticRegionContext context;
    const PhaseIndex* index = nullptr;
    const SyncInput* input = nullptr;
    FiniteExpansionLimits limits;
    RecognitionResult result;
    SmallVector<NormalizedControlNode, 0> nodes;
    SmallVector<std::size_t> roots;
    uint64_t planningVisits = 0, expandedLoops = 0, payloads = 0, effects = 0;
};
std::shared_ptr<const NormalizedControlDescription> normalizeSmallCountControl(
    ArithmeticRegionContext context, const PhaseIndex& index, const SyncInput& input,
    const FiniteExpansionLimits& limits = {});
namespace detail {
std::optional<int64_t> normalizedInteger(Value value, ArrayRef<FixedLoopCoordinate> coordinates,
                                         MLIRContext* context);
std::optional<bool> normalizedCondition(Value value, const ArithmeticSite& site, uint64_t& work);
}
} // namespace mlir::pto::frontiersynch
#endif
