// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Cache scalar proofs within an exact structured-control and index-width
// context. Fixed-coordinate clients keep their separate specialized evaluator.
#include "PTO/Transforms/InsertSync/SyncScalarFacts.h"
#include "SyncScalarEvolution.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
namespace mlir::pto {
namespace {
struct Context {
    detail::ScalarEvolution scalars;
    DenseMap<Value, unsigned> ids;
    SmallVector<Value> symbols;
    explicit Context(Operation* access) : scalars(access->getContext(), access) {}
};
AffineExpr rebind(AffineExpr expression, ArrayRef<Value> symbols, llvm::function_ref<AffineExpr(Value)> symbol)
{
    if (!expression) {
        return {};
    }
    // Shared normalization can retain a long DAG across many operations. Use an
    // explicit postorder stack instead of recursing along its expression depth.
    DenseMap<AffineExpr, AffineExpr> cache;
    SmallVector<std::pair<AffineExpr, bool>> pending{{expression, false}};
    while (!pending.empty()) {
        auto [current, childrenReady] = pending.pop_back_val();
        if (cache.count(current)) {
            continue;
        }
        if (auto binary = dyn_cast<AffineBinaryOpExpr>(current)) {
            if (!childrenReady) {
                pending.emplace_back(current, true);
                pending.emplace_back(binary.getRHS(), false);
                pending.emplace_back(binary.getLHS(), false);
                continue;
            }
            auto left = cache.lookup(binary.getLHS()), right = cache.lookup(binary.getRHS());
            if (!left || !right) {
                return {};
            }
            cache[current] = getAffineBinaryOpExpr(binary.getKind(), left, right);
        } else if (auto id = dyn_cast<AffineSymbolExpr>(current)) {
            if (id.getPosition() >= symbols.size()) {
                return {};
            }
            auto rebound = symbol(symbols[id.getPosition()]);
            if (!rebound) {
                return {};
            }
            cache[current] = rebound;
        } else {
            cache[current] = current;
        }
    }
    return cache.lookup(expression);
}
} // namespace
struct SyncScalarFacts::Impl {
    // Branch assumptions and induction scopes are identical within one block.
    // Width is included because data-layout anchors may refine index arithmetic.
    DenseMap<std::pair<Block*, uint64_t>, std::unique_ptr<Context>> contexts;
};
SyncScalarFacts::SyncScalarFacts() : impl(std::make_unique<Impl>()) {}
SyncScalarFacts::~SyncScalarFacts() = default;
AffineExpr SyncScalarFacts::value(Value value, Operation* access, llvm::function_ref<AffineExpr(Value)> symbol) const
{
    if (!value || !access || !access->getBlock()) {
        return {};
    }
    auto bits = DataLayout::closest(access).getTypeSizeInBits(IndexType::get(access->getContext()));
    const uint64_t width = bits.isScalable() ? 0 : bits.getFixedValue();
    auto& owned = impl->contexts[{access->getBlock(), width}];
    if (!owned) {
        owned = std::make_unique<Context>(access);
    }
    auto* context = owned.get();
    auto expression = context->scalars.value(value, [&](Value original) {
        auto inserted = context->ids.try_emplace(original, context->symbols.size());
        if (inserted.second) {
            context->symbols.push_back(original);
        }
        return getAffineSymbolExpr(inserted.first->second, access->getContext());
    });
    return rebind(expression, context->symbols, symbol);
}
} // namespace mlir::pto
