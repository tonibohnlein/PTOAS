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
        appendOperands(nodes[next], pending);
    }
    llvm::sort(order);
    for (Id id : order) {
        const auto node = nodes[id]; // Rebuilding can grow the arena.
        Id result = id;
        if (node.kind == Kind::Integer) {
            result = rebuildInteger(node, context.memo);
        } else if (node.kind == Kind::Select) {
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
FailureOr<SmallVector<RegionExpressions::Id>> RegionExpressions::import(
    const RegionExpressions& source, ArrayRef<Id> roots, uint64_t* work)
{
    const bool validRoots = constructionMessage.empty() && source.constructionMessage.empty() &&
        llvm::all_of(roots, [&](Id id) { return source.valid(id); });
    if (!validRoots) { return failure(); }
    if (&source == this || roots.empty()) { return SmallVector<Id>(roots.begin(), roots.end()); }
    Transaction transaction(*this);
    SmallVector<Id> pending(roots.begin(), roots.end()), order;
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        const Id id = pending.pop_back_val();
        if (!seen.insert(id).second) { continue; }
        order.push_back(id); source.appendOperands(source.nodes[id], pending);
    }
    llvm::sort(order);
    if (work) {
        const auto count = static_cast<uint64_t>(order.size());
        *work = count > UINT64_MAX - *work ? UINT64_MAX : *work + count;
    }
    llvm::DenseMap<Id, Id> memo;
    for (Id id : order) {
        const auto& node = source.nodes[id];
        Id result = invalid;
        if (node.kind == Kind::Constant) {
            result = node.boolean ? boolean(node.literal != 0) : constant(node.literal);
        }
        else if (node.kind == Kind::Input) { result = input(node.value); }
        else if (node.kind == Kind::Integer) { result = rebuildInteger(node, memo); }
        else if (node.kind == Kind::Select) {
            result = select(memo.lookup(node.a), memo.lookup(node.b), memo.lookup(node.c));
        } else if (node.kind == Kind::Not) { result = lnot(memo.lookup(node.a)); }
        else { result = binary(node.kind, memo.lookup(node.a), memo.lookup(node.b)); }
        if (result == invalid) { return failure(); }
        memo[id] = result;
    }
    SmallVector<Id> imported;
    for (Id id : roots) { imported.push_back(memo.lookup(id)); }
    // Query placeholders have owners separate from source IR. Preserve them
    // and the monotone payload replay prohibition for the target's lifetime.
    const std::owner_less<std::shared_ptr<void>> before;
    for (const auto& owner : source.inputOwners) {
        const bool retained = llvm::any_of(inputOwners, [&](const auto& other) {
            return !before(owner, other) && !before(other, owner);
        });
        if (!retained) { inputOwners.push_back(owner); }
    }
    forbiddenRecomputation.insert(source.forbiddenRecomputation.begin(), source.forbiddenRecomputation.end());
    transaction.commit();
    return imported;
}
} // namespace mlir::pto::frontiersynch
