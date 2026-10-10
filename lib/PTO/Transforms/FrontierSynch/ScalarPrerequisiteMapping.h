// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_SCALARPREREQUISITEMAPPING_H
#define PTO_FRONTIERSYNCH_SCALARPREREQUISITEMAPPING_H
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "mlir/IR/Dominance.h"
namespace mlir::pto::frontiersynch::detail {
// This certificate concerns SSA dependence, not replay of scalar instructions.
// Exact occurrence domains supply guards; shared original coordinates identify
// the one enclosing producer visit for each consumer visit.
inline bool directPrerequisiteMapping(const ValuePrerequisite& edge,
    const ArithmeticSite& source, const ArithmeticSite& target, DominanceInfo& dominance)
{
    const bool identities = edge.directSSA && source.phase && target.phase &&
        source.phase == edge.producer && target.phase->elementOp == edge.consumer;
    const bool originalCoordinates = source.loops.size() <= target.loops.size();
    if (!identities || !originalCoordinates) { return false; }
    for (const auto& fixed : source.fixedCoordinates) {
        auto match = llvm::find_if(target.fixedCoordinates, [&](const auto& candidate) {
            return candidate.loop == fixed.loop && candidate.induction == fixed.induction;
        });
        if (match == target.fixedCoordinates.end()) { return false; }
    }
    for (unsigned i = 0; i < source.loops.size(); ++i) {
        if (source.loops[i] != target.loops[i]) { return false; }
    }
    return source.phase->elementOp && edge.consumer &&
        dominance.properlyDominates(source.phase->elementOp, edge.consumer);
}
} // namespace mlir::pto::frontiersynch::detail
#endif
