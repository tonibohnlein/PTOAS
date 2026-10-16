// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bind only the reached original/replay inputs in one immutable cut environment.
#ifndef PTO_FRONTIERSYNCH_MAPPED_CUT_INPUTS_H
#define PTO_FRONTIERSYNCH_MAPPED_CUT_INPUTS_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "../../InsertSync/SyncScalarReplay.h"
#include "mlir/IR/IRMapping.h"
namespace mlir::pto::frontiersynch::detail {
inline void bindMappedCutInputList(
    ArrayRef<std::pair<RegionExpressions::Id, Value>> inputs, Operation* cut, const IRMapping& mapping,
    DominanceInfo& dominance, RegionExpressions::CutEmission& context, uint64_t* traversed = nullptr)
{
    SmallVector<Value> pending;
    for (auto [id, original] : inputs) {
        if (context.values.count(id)) { continue; }
        pending.push_back(original);
        auto value = mapping.lookupOrNull(original);
        if (value && dominance.properlyDominates(original, cut)) {
            context.inputs[original] = value;
            // Integer inputs still need the expression service's conversion to
            // its index arithmetic type. A raw i32/i64 value is not that result.
            // An identity-mapped original Boolean still admits facts from its
            // enclosing original branch. Treating it as an opaque expression
            // binding would suppress that cofactor and demand unreachable inputs.
            // Nonidentity replacements (including historical values) remain opaque.
            const bool originalBoolean = value == original && value.getType().isInteger(1);
            if (!originalBoolean && (value.getType().isIndex() || value.getType().isInteger(1))) {
                context.values[id] = value;
                context.cofactors[id] = id;
            }
        }
    }
    DenseSet<Value> seen;
    while (!pending.empty()) {
        auto value = pending.pop_back_val();
        if (context.inputs.count(value) || !seen.insert(value).second) { continue; }
        if (dominance.properlyDominates(value, cut)) {
            if (auto replacement = mapping.lookupOrNull(value)) { context.inputs[value] = replacement; }
        } else if (auto* definition = value.getDefiningOp(); mlir::pto::detail::canReplayScalar(definition)) {
            llvm::append_range(pending, definition->getOperands());
        }
    }
    if (traversed) {
        *traversed = inputs.size() > UINT64_MAX - seen.size() ? UINT64_MAX : inputs.size() + seen.size();
    }
}
inline void bindMappedCutInputs(
    RegionExpressions& arena, RegionExpressions::Id root, Operation* cut, const IRMapping& mapping,
    DominanceInfo& dominance, RegionExpressions::CutEmission& context, uint64_t* traversed = nullptr)
{
    uint64_t nodes = 0, inputsVisited = 0;
    auto inputs = arena.referencedInputs(ArrayRef<RegionExpressions::Id>(root), &nodes);
    bindMappedCutInputList(inputs, cut, mapping, dominance, context, &inputsVisited);
    if (traversed) {
        *traversed = nodes > UINT64_MAX - inputsVisited ? UINT64_MAX : nodes + inputsVisited;
    }
}
} // namespace mlir::pto::frontiersynch::detail
#endif
