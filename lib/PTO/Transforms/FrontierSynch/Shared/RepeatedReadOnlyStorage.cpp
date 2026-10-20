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
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
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
// An invariant child can retain non-enumerated GM interfaces when its local
// crossing graph already covers every possible residual conflict. This is a
// sufficient reduction proof, never an extra edge or an alias discharge.
RepeatedRegionAnalysis coveredResidualRepeat(func::FuncOp function, scf::ForOp loop,
    const RegionalAnalysis& original, RegionExpressions::Id trips)
{
    RepeatedRegionAnalysis refused;
    refused.error = "residual GM crossings are not covered by the retained local order";
    if (!original.accessModel || original.anchors.empty() || original.symbolicStorageEffects.empty() ||
        !original.capabilities.exactQueries || !original.presence || !original.reachability) { return refused; }
    llvm::DenseSet<std::size_t> validated;
    if (!validate(original, loop, validated).empty()) { return refused; }
    const auto effects = original.accessModel->effects();
    llvm::DenseSet<std::size_t> omitted;
    for (auto id : original.symbolicStorageEffects) {
        if (id >= effects.size() || !effects[id].memory || effects[id].memory->scope != AddressSpace::GM ||
            effects[id].regions.empty()) { return refused; }
        omitted.insert(id);
    }
    auto local = original;
    local.storageSelectors = {}; local.symbolicStorage.reset();
    local.arithmeticRelations.reset(); local.relations.reset();
    local.symbolicStorageEffects.clear();
    llvm::erase_if(local.accessBoundary, [&](const auto& access) { return omitted.count(access.effect); });
    llvm::erase_if(local.deferredAccessBoundary, [&](const auto& access) { return omitted.count(access.effect); });
    auto candidate = repeatInvariantRegion(function, loop, std::move(local), trips);
    if (!candidate.error.empty() || !candidate.state) { return refused; }
    std::vector<RegionalAccessBoundary> accesses = original.accessBoundary;
    llvm::append_range(accesses, original.deferredAccessBoundary);
    // Every occurrence of an omitted effect must have certified extrema.
    for (auto id : omitted) {
        if (llvm::none_of(accesses, [id](const auto& access) { return access.effect == id; })) { return refused; }
    }
    for (const auto& anchor : original.anchors) {
        for (auto id : original.accessModel->effectsFor(anchor.phase)) {
            const bool relevant = llvm::any_of(original.symbolicStorageEffects,
                [&](std::size_t other) { return original.accessModel->mayConflict(id, other); });
            if (!relevant) { continue; }
            if (llvm::none_of(accesses, [id](const auto& access) { return access.effect == id; })) { return refused; }
            // Do not turn an unclassified residual writer into a complete
            // selector export merely because another effect is symbolic.
            if (!omitted.count(id) && effects[id].memory && effects[id].memory->scope == AddressSpace::GM &&
                llvm::any_of(accesses, [id](const auto& access) {
                    return access.effect == id && !access.representedByCells;
                })) { return refused; }
        }
    }
    auto& e = *original.expressions;
    for (const auto& source : accesses) {
        for (const auto& target : accesses) {
            if ((!omitted.count(source.effect) && !omitted.count(target.effect)) ||
                !original.accessModel->mayConflict(source.effect, target.effect)) { continue; }
            if (source.effect >= effects.size() || target.effect >= effects.size() ||
                !validRegionalEvent(original, source.last.event) || !validRegionalEvent(original, target.first.event) ||
                original.anchors[source.last.event.type].phase != effects[source.effect].phase ||
                original.anchors[target.first.event.type].phase != effects[target.effect].phase ||
                source.last.present >= e.size() || target.first.present >= e.size() ||
                !e.isBoolean(source.last.present) || !e.isBoolean(target.first.present)) { return refused; }
            auto from = source.last.event, to = target.first.event;
            from.kind = PeriodicEventKind::Completion;
            to.kind = PeriodicEventKind::Start;
            auto active = e.land(source.last.present, target.first.present);
            if (e.constantValue(active) == uint64_t(0)) { continue; }
            // relativeQuery excludes finite invocation clipping and is reused
            // for all outer translations. A writer's self-pair is included:
            // intrinsic/native chains then extend this proof to every gap > 1.
            auto reached = candidate.state->relativeQuery(from, to, 1);
            accumulateCost(candidate.regional.cost.implicationChecks, 1);
            if (!reached || !e.implies(active, *reached)) { return refused; }
        }
    }
    return candidate;
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
        body.arithmeticRelations.reset(); body.relations.reset();
        return {};
    }
    llvm::DenseSet<std::size_t> readers;
    auto error = validate(body, loop, readers);
    if (!error.empty()) { return error; }
    auto crossingView = body;
    const bool conflictFree = conflictFreeReaders(body, loop, readers);
    llvm::DenseSet<std::size_t> independentReaders;
    for (auto id : readers) {
        llvm::DenseSet<std::size_t> single;
        single.insert(id);
        if (conflictFreeReaders(body, loop, single)) { independentReaders.insert(id); }
    }
    if (!conflictFree) {
        error = materializeRepeatedSymbolicStorage(crossingView, loop, trips, periodEffects);
        if (!error.empty()) { return error; }
    }
    crossingView.storageSelectors = {};
    crossingView.symbolicStorage.reset();
    crossingView.arithmeticRelations.reset(); crossingView.relations.reset();
    crossingView.symbolicStorageEffects.clear();
    if (conflictFree) {
        llvm::erase_if(crossingView.accessBoundary, [&](const auto& access) { return readers.count(access.effect); });
        llvm::erase_if(crossingView.deferredAccessBoundary,
            [&](const auto& access) { return readers.count(access.effect); });
    } else {
        // Finite atoms do not discharge unknown relationships to other effects.
        // Keep effect extrema for the existing uniform/residual bridge adapter.
        for (auto& access : crossingView.accessBoundary) {
            if (readers.count(access.effect) && !independentReaders.count(access.effect)) {
                access.representedByCells = true;
            }
        }
        for (auto access : crossingView.deferredAccessBoundary) {
            if (readers.count(access.effect) && !independentReaders.count(access.effect)) {
                access.representedByCells = true;
            }
            crossingView.accessBoundary.push_back(std::move(access));
        }
        crossingView.deferredAccessBoundary.clear();
    }
    // This removal belongs only to this enclosing visit's crossing proof.
    // The exported original interface below retains every reader and refusal.
    llvm::erase_if(crossingView.accessBoundary,
        [&](const auto& access) { return independentReaders.count(access.effect); });
    llvm::erase_if(crossingView.deferredAccessBoundary,
        [&](const auto& access) { return independentReaders.count(access.effect); });
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
    if (!result.error.empty()) {
        auto covered = coveredResidualRepeat(function, loop, *original, trips);
        if (!covered.error.empty()) {
            result.error += "; " + covered.error;
            return result;
        }
        result = std::move(covered);
    } else {
        result = repeatInvariantRegion(function, loop, std::move(crossingView), trips);
        if (!result.error.empty()) { return result; }
    }
    // Internal queries and recipes remain complete. Only independently proven
    // conflict-free readers are removed from the two-copy access boundary.
    auto lifted = *original;
    liftRepeatedSelectors(lifted, trips);
    auto& out = result.regional;
    out.accessBoundary = std::move(lifted.accessBoundary);
    out.deferredAccessBoundary = std::move(lifted.deferredAccessBoundary);
    out.symbolicStorageEffects = original->symbolicStorageEffects;
    if (original->symbolicStorage &&
        original->symbolicStorage->expressions == original->expressions &&
        original->symbolicStorage->accessModel == original->accessModel &&
        original->symbolicStorage->gmAliasPolicy == original->gmAliasPolicy) {
        auto partition = std::make_shared<RegionalSymbolicStorageCertificate>();
        partition->expressions = out.expressions;
        partition->accessModel = out.accessModel;
        partition->gmAliasPolicy = out.gmAliasPolicy;
        partition->selectorCuts = original->symbolicStorage->selectorCuts;
        partition->selectorSupport = original->symbolicStorage->selectorSupport;
        partition->selectorDomainSupport = original->symbolicStorage->selectorDomainSupport;
        out.symbolicStorage = std::move(partition);
    }
    out.storageSelectors = [original, trips](RegionalByteAddress address)
        -> std::optional<RegionalStorageSelectors> {
        if (original->expressions->constantValue(trips) == uint64_t(0)) { return RegionalStorageSelectors{}; }
        auto selected = regionalStorageSelectors(*original, address);
        if (selected) { lift(*selected, *original->expressions, trips); }
        return selected;
    };
    out.cost.expressionNodes = out.expressions->size();
    return result;
}
} // namespace mlir::pto::frontiersynch
