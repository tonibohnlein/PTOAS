// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_PHASELOGICALVIEW_H
#define PTO_FRONTIERSYNCH_PHASELOGICALVIEW_H
#include "PhaseNormalization.h"
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch::detail {
using Expr = RegionExpressions::Id;
// Specialize the logical circuit's actual inputs. Physical-only SSA values are
// kept in the original access model and checked by the joint storage proof.
struct PhaseLogicalView {
    RegionalAnalysis original;
    std::shared_ptr<PhaseIndex> index;
    std::shared_ptr<Block> symbols = std::make_shared<Block>();
    PhaseNormalization normalizer;
    uint64_t phase, period;
    SmallVector<Expr> coordinates;
    unsigned width = 1;
    PhaseLogicalView(RegionalAnalysis original, std::shared_ptr<PhaseIndex> index,
                     scf::ForOp outer, uint64_t phase, uint64_t period)
        : original(std::move(original)), index(std::move(index)),
          normalizer(outer, *this->index, *this->original.expressions), phase(phase), period(period)
    {
        this->original.expressions->retainInputOwner(symbols);
        for (const auto& loops : this->original.outerLoops) { width = std::max(width, unsigned(loops.size() + 1)); }
        for (unsigned i = 0; i < 2 * width; ++i) {
            coordinates.push_back(this->original.expressions->input(
                symbols->addArgument(IndexType::get(outer.getContext()), outer.getLoc())));
        }
    }
    std::optional<Expr> specialize(Expr root)
    {
        auto& e = *original.expressions;
        SmallVector<std::pair<Expr, Expr>> bindings;
        for (auto [id, value] : e.referencedInputs(root)) {
            if (normalizer.independent(value)) { continue; }
            auto representative = normalizer.atPhase(value, phase, period);
            if (!representative) { return std::nullopt; }
            bindings.emplace_back(id, *representative);
        }
        RegionExpressions::Substitution context(bindings);
        return e.substitute(root, context);
    }
    void protect(Expr& coordinate, unsigned position, SmallVectorImpl<std::pair<Expr, Expr>>& values)
    {
        // Literal explicit ordinals must remain visible to the child provider.
        for (auto [id, value] : original.expressions->referencedInputs(coordinate)) {
            (void)id;
            if (normalizer.independent(value)) { continue; }
            values.emplace_back(coordinates[position], coordinate);
            coordinate = coordinates[position];
            return;
        }
    }
    RegionalEvent bind(RegionalEvent event, unsigned start, SmallVectorImpl<std::pair<Expr, Expr>>& values)
    {
        protect(event.ordinal, start, values);
        for (unsigned i = 0; i < event.visits.size(); ++i) { protect(event.visits[i], start + i + 1, values); }
        return event;
    }
    std::optional<Expr> restore(std::optional<Expr> root, ArrayRef<std::pair<Expr, Expr>> values)
    {
        if (!root) { return std::nullopt; }
        auto result = specialize(*root);
        if (!result) { return std::nullopt; }
        RegionExpressions::Substitution arguments(values);
        return original.expressions->substitute(*result, arguments);
    }
    std::optional<Expr> query(RegionalEvent a, std::optional<RegionalEvent> b, bool reference)
    {
        if (a.visits.size() >= width || (b && b->visits.size() >= width)) { return std::nullopt; }
        SmallVector<std::pair<Expr, Expr>> values;
        auto first = bind(a, 0, values);
        std::optional<Expr> result;
        if (!b) { result = regionalPresence(original, first); }
        else {
            auto second = bind(*b, width, values);
            result = reference ? regionalReferenceBefore(original, first, second) :
                                 regionalReachability(original, first, second);
        }
        return restore(result, values);
    }
    bool rewrite(RegionalSelector& value, ArrayRef<std::pair<Expr, Expr>> arguments = {})
    {
        auto apply = [&](Expr& root) {
            auto result = restore(root, arguments);
            if (!result) { return false; }
            root = *result; return true;
        };
        if (!apply(value.present) || !apply(value.event.ordinal)) { return false; }
        return llvm::all_of(value.event.visits, apply);
    }
    bool rewrite(RegionalStorageSelectors& selectors, ArrayRef<std::pair<Expr, Expr>> arguments = {})
    {
        for (auto* side : {&selectors.firstWriters, &selectors.lastWriters}) {
            for (auto& value : *side) { if (!rewrite(value, arguments)) { return false; } }
        }
        for (auto* side : {&selectors.firstReaders, &selectors.lastReaders}) {
            for (auto& [pipe, values] : *side) {
                for (auto& value : values) { if (!rewrite(value, arguments)) { return false; } }
            }
        }
        return true;
    }
};
inline bool bindLogicalPhase(RegionalAnalysis& view, std::shared_ptr<PhaseIndex> index, scf::ForOp outer,
                      uint64_t phase, uint64_t period)
{
    auto state = std::make_shared<PhaseLogicalView>(view, std::move(index), outer, phase, period);
    for (auto* side : {&view.firstPayloads, &view.lastPayloads, &view.firstSitePayloads}) {
        for (auto& [pipe, values] : *side) {
            for (auto& value : values) { if (!state->rewrite(value)) { return false; } }
        }
    }
    for (auto& cell : view.storageBoundary) {
        for (auto* side : {&cell.firstWriters, &cell.lastWriters}) {
            for (auto& value : *side) { if (!state->rewrite(value)) { return false; } }
        }
        for (auto* side : {&cell.firstReaders, &cell.lastReaders}) {
            for (auto& [pipe, values] : *side) {
                for (auto& value : values) { if (!state->rewrite(value)) { return false; } }
            }
        }
    }
    for (auto* side : {&view.accessBoundary, &view.deferredAccessBoundary}) {
        for (auto& access : *side) {
            if (!state->rewrite(access.first) || !state->rewrite(access.last)) { return false; }
        }
    }
    for (auto* root : {&view.firstOrdinal, &view.endpointSiteGuard, &view.endpointInvocationGuard}) {
        if (!*root) { continue; }
        auto result = state->specialize(**root);
        if (!result) { return false; }
        *root = result;
    }
    view.presence = [state](RegionalEvent a) { return state->query(a, std::nullopt, false); };
    view.reachability = [state](RegionalEvent a, RegionalEvent b) { return state->query(a, b, false); };
    view.referenceBefore = [state](RegionalEvent a, RegionalEvent b) { return state->query(a, b, true); };
    if (view.endpointEventGuard) {
        view.endpointEventGuard = [state](RegionalEvent event) -> std::optional<Expr> {
            if (event.visits.size() >= state->width) { return std::nullopt; }
            SmallVector<std::pair<Expr, Expr>> values;
            event = state->bind(event, 0, values);
            return state->restore(state->original.endpointEventGuard(event), values);
        };
    }
    if (view.storageSelectors) {
        view.storageSelectors = [state](RegionalByteAddress address) -> std::optional<RegionalStorageSelectors> {
            SmallVector<std::pair<Expr, Expr>> values;
            state->protect(address.offset, 0, values);
            auto selectors = state->original.storageSelectors(address);
            if (!selectors || !state->rewrite(*selectors, values)) { return std::nullopt; }
            return selectors;
        };
    }
    view.arithmeticRelations.reset(); view.relations.reset(); view.symbolicStorage.reset(); view.numerical.reset();
    return true;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
