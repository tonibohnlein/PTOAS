// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bind nested occurrence coordinates without cloning source IR or DAG paths.
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::pto::frontiersynch {
RegionExpressions::Id RegionExpressions::substitute(Id expression, Substitution& context)
{
    if (!valid(expression) || (context.owner && context.owner != this)) {
        return reject("regional substitution requires a valid root in its owning arena");
    }
    if (!context.owner) {
        for (auto [source, target] : context.bindings) {
            if (!valid(source) || !valid(target) || isBoolean(source) != isBoolean(target)) {
                return reject("regional substitution has an invalid or differently typed binding");
            }
            auto [entry, added] = context.memo.try_emplace(source, target);
            if (!added && entry->second != target) {
                return reject("regional substitution has conflicting bindings");
            }
        }
        context.owner = this;
    }
    SmallVector<Id> pending{expression}, order;
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        const Id next = pending.pop_back_val();
        if (context.memo.count(next) || !seen.insert(next).second) { continue; }
        order.push_back(next);
        for (Id operand : {nodes[next].a, nodes[next].b, nodes[next].c}) {
            if (operand != invalid) { pending.push_back(operand); }
        }
    }
    llvm::sort(order);
    for (Id id : order) {
        const auto node = nodes[id]; // Rebuilding can grow the arena.
        Id result = id;
        if (node.kind == Kind::Select) {
            result = select(context.memo.lookup(node.a), context.memo.lookup(node.b), context.memo.lookup(node.c));
        } else if (node.kind == Kind::Not) {
            result = lnot(context.memo.lookup(node.a));
        } else if (node.a != invalid && node.b != invalid) {
            result = binary(node.kind, context.memo.lookup(node.a), context.memo.lookup(node.b));
        }
        if (result == invalid) { return invalid; }
        context.memo[id] = result;
    }
    return context.memo.lookup(expression);
}
} // namespace mlir::pto::frontiersynch
