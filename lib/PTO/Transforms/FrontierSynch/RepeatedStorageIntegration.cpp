// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// The evolving-storage theorem changes inter-visit witnesses, not body graphs.
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::pto::frontiersynch {
RepeatedRegionAnalysis repeatEvolvingRegion(func::FuncOp function, std::shared_ptr<RepeatedStorage> storage)
{
    RepeatedRegionAnalysis result;
    if (!storage) { result.error = "evolving repetition has no validated storage certificate"; return result; }
    auto crossingView = storage->body();
    // The new certificate reconstructed every symbolic effect from its complete
    // coordinate domain. Its byte selectors supersede the child's only in this
    // inter-visit view; internal queries and endpoint recipes remain unchanged.
    if (llvm::any_of(crossingView.symbolicStorageEffects,
                    [&](std::size_t effect) { return !storage->contains(effect); })) {
        result.error = "nested storage projection did not cover every symbolic effect";
        return result;
    }
    crossingView.storageSelectors = {};
    crossingView.symbolicStorageEffects.clear();
    crossingView.symbolicStorage.reset();
    crossingView.arithmeticRelations.reset(); crossingView.relations.reset();
    llvm::erase_if(crossingView.accessBoundary,
                   [&](const auto& access) { return storage->contains(access.effect); });
    // The unchanged query and preparation closures still contain every payload
    // and internal dependency. Only certified conflict-free inter-visit effects
    // are absent from this temporary boundary view.
    result = repeatInvariantRegion(function, storage->loop(), std::move(crossingView), storage->trips());
    if (!result.error.empty()) { return result; }
    auto& out = result.regional;
    auto& e = *out.expressions;
    const auto nonempty = e.lt(e.constant(0), storage->trips());
    for (auto access : storage->body().accessBoundary) {
        if (!storage->contains(access.effect)) { continue; }
        access.first.event.visits.insert(access.first.event.visits.begin(), e.constant(0));
        access.last.event.visits.insert(access.last.event.visits.begin(), e.sub(storage->trips(), e.constant(1)));
        access.first.present = e.land(nonempty, access.first.present);
        access.last.present = e.land(nonempty, access.last.present);
        access.representedByCells = false;
        out.accessBoundary.push_back(std::move(access));
    }
    out.symbolicStorageEffects.assign(storage->effects().begin(), storage->effects().end());
    out.symbolicStorage = storage->certificate();
    out.storageSelectors = [storage](RegionalByteAddress address) { return storage->selectors(address); };
    return result;
}
} // namespace mlir::pto::frontiersynch
