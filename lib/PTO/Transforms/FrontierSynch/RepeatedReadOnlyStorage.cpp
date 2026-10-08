// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Removing a reader from the crossing view needs disjointness from all writers.
#include "RepeatedReadOnlyStorage.h"
#include "RepeatedRegionInternal.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::pto::frontiersynch {
namespace {
// Inner coordinates may vary within a visit, but their domains and scalar maps
// must be independent of this outer visit. Loop-carried SSA is not a proof.
bool independent(Value value, scf::ForOp loop, DenseMap<Value, bool>& memo)
{
    if (!value) { return true; }
    if (auto found = memo.find(value); found != memo.end()) { return found->second; }
    memo[value] = false;
    auto* owner = value.getDefiningOp();
    auto argument = dyn_cast<BlockArgument>(value);
    if (argument) { owner = argument.getOwner()->getParentOp(); }
    if (!owner) { return false; }
    if (owner != loop && !loop->isProperAncestor(owner)) { return memo[value] = true; }
    if (argument) {
        auto inner = dyn_cast<scf::ForOp>(owner);
        if (!inner || inner == loop || argument != inner.getInductionVar()) { return false; }
        return memo[value] = independent(inner.getLowerBound(), loop, memo) &&
            independent(inner.getUpperBound(), loop, memo) && independent(inner.getStep(), loop, memo);
    }
    auto dialect = owner->getName().getDialectNamespace();
    if ((dialect != "arith" && dialect != "index") || owner->getNumRegions() ||
        !isMemoryEffectFree(owner) || !isSpeculatable(owner)) { return false; }
    return memo[value] = llvm::all_of(owner->getOperands(),
        [&](Value operand) { return independent(operand, loop, memo); });
}
bool invariantMap(const SyncStorageEffect& effect, scf::ForOp loop)
{
    DenseMap<Value, bool> memo;
    if (effect.selection && !independent(effect.selection->selector, loop, memo)) { return false; }
    // Missing exact access maps cannot establish the symbolic family premise.
    if (effect.regions.empty()) { return false; }
    for (const auto& region : effect.regions) {
        if (!region.byteOffset || !independent(region.base, loop, memo)) { return false; }
        for (unsigned i = 0; i < region.symbols.size(); ++i) {
            const bool used = region.byteOffset.isFunctionOfSymbol(i) || llvm::any_of(region.extents,
                [i](AffineExpr extent) { return extent.isFunctionOfSymbol(i); });
            if (used && !independent(region.symbols[i], loop, memo)) { return false; }
        }
    }
    return true;
}
std::string validate(const RegionalAnalysis& body, scf::ForOp loop, llvm::DenseSet<std::size_t>& readers)
{
    if (!body.storageSelectors || body.symbolicStorageEffects.empty() || !body.accessModel) {
        return "symbolic repetition requires byte selectors and complete effect identities";
    }
    const auto effects = body.accessModel->effects();
    for (auto id : body.symbolicStorageEffects) {
        if (id >= effects.size() || !readers.insert(id).second || !effects[id].phase ||
            !effects[id].phase->elementOp || !loop->isProperAncestor(effects[id].phase->elementOp) ||
            !effects[id].memory) { return "invalid repeated symbolic effect identity"; }
        if (!invariantMap(effects[id], loop)) {
            return "symbolic effect map invariance is unavailable for the outer visit";
        }
    }
    return {};
}
bool conflictFreeReaders(const RegionalAnalysis& body, scf::ForOp loop,
                         const llvm::DenseSet<std::size_t>& readers)
{
    const auto effects = body.accessModel->effects();
    for (auto id : readers) {
        if (effects[id].mode != SyncAccessMode::Read) { return false; }
    }
    // Do not restrict this scan to exported selectors: a discharged or omitted
    // writer can still invalidate the read-only class after outer re-entry.
    for (std::size_t id = 0; id < effects.size(); ++id) {
        const auto& writer = effects[id];
        if (writer.mode != SyncAccessMode::Write || !writer.phase || !writer.phase->elementOp ||
            !loop->isProperAncestor(writer.phase->elementOp)) { continue; }
        if (!writer.memory) { return false; }
        for (auto reader : readers) {
            if (body.accessModel->mayOverlap(reader, id)) {
                return false;
            }
        }
    }
    return true;
}
void lift(RegionalStorageSelectors& selectors, RegionExpressions& e, RegionExpressions::Id trips)
{
    const auto nonempty = e.lt(e.constant(0), trips);
    auto appendVisit = [&](std::vector<RegionalSelector>& values, bool final) {
        for (auto& value : values) {
            value.event.visits.insert(value.event.visits.begin(), final ? e.sub(trips, e.constant(1)) : e.constant(0));
            value.present = e.land(nonempty, value.present);
        }
    };
    appendVisit(selectors.firstWriters, false);
    appendVisit(selectors.lastWriters, true);
    for (auto& [pipe, values] : selectors.firstReaders) { appendVisit(values, false); }
    for (auto& [pipe, values] : selectors.lastReaders) { appendVisit(values, true); }
}
} // namespace
std::string prepareRepeatedSymbolicStorage(RegionalAnalysis& body, scf::ForOp loop,
    RegionExpressions::Id trips, ArrayRef<std::size_t> periodEffects)
{
    if (!loop || !body.expressions || trips >= body.expressions->size() || body.expressions->isBoolean(trips)) {
        return "symbolic repetition requires an original loop and integer trip expression";
    }
    if (body.symbolicStorageEffects.empty()) {
        // A byte-query callback can describe an already complete finite
        // boundary. It is an additional export, not a symbolic-storage debt.
        // The caller retains the original callback and lifts it after ordinary
        // repetition validates and composes this finite crossing view.
        body.storageSelectors = {};
        body.symbolicStorage.reset();
        body.arithmeticRelations.reset();
        return {};
    }
    llvm::DenseSet<std::size_t> readers;
    auto error = validate(body, loop, readers);
    if (!error.empty()) { return error; }
    auto crossingView = body;
    const bool conflictFree = conflictFreeReaders(body, loop, readers);
    if (!conflictFree) {
        error = materializeRepeatedSymbolicStorage(crossingView, loop, trips, periodEffects);
        if (!error.empty()) { return error; }
    }
    crossingView.storageSelectors = {};
    crossingView.symbolicStorage.reset();
    crossingView.arithmeticRelations.reset();
    crossingView.symbolicStorageEffects.clear();
    if (conflictFree) {
        llvm::erase_if(crossingView.accessBoundary, [&](const auto& access) { return readers.count(access.effect); });
        llvm::erase_if(crossingView.deferredAccessBoundary,
            [&](const auto& access) { return readers.count(access.effect); });
    } else {
        // Finite atoms do not discharge unknown relationships to other effects.
        // Keep effect extrema for the existing uniform/residual bridge adapter.
        for (auto& access : crossingView.accessBoundary) {
            if (readers.count(access.effect)) { access.representedByCells = true; }
        }
        for (auto access : crossingView.deferredAccessBoundary) {
            if (readers.count(access.effect)) { access.representedByCells = true; }
            crossingView.accessBoundary.push_back(std::move(access));
        }
        crossingView.deferredAccessBoundary.clear();
    }
    body = std::move(crossingView);
    return {};
}
RepeatedRegionAnalysis repeatSymbolicStorageRegion(func::FuncOp function, scf::ForOp loop,
    RegionalAnalysis body, RegionExpressions::Id trips)
{
    RepeatedRegionAnalysis result;
    if (!function || !loop || loop->getParentOfType<func::FuncOp>() != function) {
        result.error = "symbolic repetition requires an original enclosing function";
        return result;
    }
    auto original = std::make_shared<RegionalAnalysis>(std::move(body));
    auto crossingView = *original;
    result.error = prepareRepeatedSymbolicStorage(crossingView, loop, trips);
    if (!result.error.empty()) { return result; }
    // Internal queries and recipes remain complete. Only independently proven
    // conflict-free readers are removed from the two-copy access boundary.
    result = repeatInvariantRegion(function, loop, std::move(crossingView), trips);
    if (!result.error.empty()) { return result; }
    auto lifted = *original;
    liftRepeatedSelectors(lifted, trips);
    auto& out = result.regional;
    out.accessBoundary = std::move(lifted.accessBoundary);
    out.deferredAccessBoundary = std::move(lifted.deferredAccessBoundary);
    out.symbolicStorageEffects = original->symbolicStorageEffects;
    out.storageSelectors = [original, trips](RegionalByteAddress address)
        -> std::optional<RegionalStorageSelectors> {
        if (original->expressions->constantValue(trips) == uint64_t(0)) { return RegionalStorageSelectors{}; }
        auto selected = original->storageSelectors(address);
        if (selected) { lift(*selected, *original->expressions, trips); }
        return selected;
    };
    out.cost.expressionNodes = out.expressions->size();
    return result;
}
} // namespace mlir::pto::frontiersynch
