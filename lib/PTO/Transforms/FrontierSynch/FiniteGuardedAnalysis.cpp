// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "FiniteGuardedInternal.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
bool collectEffects(FiniteGuardedState& state, const GuardedRecognition& recognized, const SyncInput& input)
{
    HardwareProtectionBuilder protection;
    std::optional<std::size_t> previousGuard;
    Operation* previous = nullptr;
    std::vector<SmallVector<const CompoundInstanceElement*>> scopes;
    std::vector<std::size_t> scopeOf;
    for (const auto& guarded : recognized.phases) {
        bool reset = previousGuard != guarded.guard;
        previousGuard = guarded.guard;
        auto* operation = guarded.phase->elementOp;
        if (previous && previous->getBlock() == operation->getBlock()) {
            for (auto* between = previous->getNextNode(); between && between != operation;
                 between = between->getNextNode()) {
                if (between->getNumRegions()) { reset = true; }
            }
        } else { reset = true; }
        if (reset || scopes.empty()) { protection.endScope(); scopes.emplace_back(); }
        scopes.back().push_back(guarded.phase);
        scopeOf.push_back(scopes.size() - 1);
        previous = operation;
        auto* phase = guarded.phase;
        ExplicitEffects occurrence;
        occurrence.payload = state.effects.size();
        occurrence.pipe = static_cast<uint32_t>(phase->kPipeValue);
        std::map<uint32_t, CellAccess> modes;
        for (auto id : input.accesses().effectsFor(phase)) {
            const auto& effect = input.accesses().effects()[id];
            if (input.accesses().independentOfOtherPhases(id)) { continue; }
            state.cost.physicalFragments += effect.ranges.size();
            for (auto cell : effect.cells) {
                if (cell >= input.accesses().cells().size() || cell > UINT32_MAX) { return false; }
                auto& mode = modes[static_cast<uint32_t>(cell)];
                mode.atom = static_cast<uint32_t>(cell);
                mode.read |= effect.mode == SyncAccessMode::Read;
                mode.write |= effect.mode == SyncAccessMode::Write;
            }
        }
        SmallVector<uint32_t> accumulator;
        for (auto [id, mode] : modes) {
            occurrence.accesses.push_back(mode);
            if (input.accesses().cells()[id].space == AddressSpace::ACC) { accumulator.push_back(id); }
        }
        protection.observe(phase->elementOp, occurrence, accumulator);
        state.effects.push_back(std::move(occurrence));
    }
    std::vector<uint64_t> modeledGroups(input.accesses().effects().size(), 0);
    for (const auto& scope : scopes) {
        const auto groups = modeledProtectionGroups(input, scope);
        for (const auto* phase : scope) {
            for (auto effect : input.accesses().effectsFor(phase)) { modeledGroups[effect] = groups[effect]; }
        }
    }
    for (uint32_t a = 0; a < recognized.phases.size(); ++a) {
        for (uint32_t b = a + 1; b < recognized.phases.size(); ++b) {
            bool conflict = false;
            for (auto x : input.accesses().effectsFor(recognized.phases[a].phase)) {
                for (auto y : input.accesses().effectsFor(recognized.phases[b].phase)) {
                    const bool protectedPair = scopeOf[a] == scopeOf[b] && hardwareProtectsConflict(
                        static_cast<uint32_t>(recognized.phases[a].phase->kPipeValue), modeledGroups[x],
                        static_cast<uint32_t>(recognized.phases[b].phase->kPipeValue), modeledGroups[y]);
                    conflict |= input.accesses().residualConflict(x, y) && !protectedPair;
                }
            }
            if (conflict) { state.residual.push_back({a, b}); }
        }
    }
    return true;
}
bool validRoots(func::FuncOp function, ArrayRef<Operation*> roots)
{
    Operation* previous = nullptr;
    for (auto* root : roots) {
        if (!root || root->getBlock() != &function.front() || (previous && previous->getNextNode() != root)) {
            return false;
        }
        previous = root;
    }
    return true;
}
} // namespace
FiniteGuardedAnalysis analyzeFiniteGuarded(func::FuncOp function, ArrayRef<Operation*> roots,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions)
{
    FiniteGuardedAnalysis result;
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock() || !validRoots(function, roots)) {
        result.error = "finite guarded analysis requires consecutive function-body roots"; return result;
    }
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (bits.isScalable() || bits.getFixedValue() != 64) {
        result.error = "finite guarded endpoint arithmetic requires a 64-bit index representation"; return result;
    }
    auto recognized = recognizeFiniteGuarded(roots, index, input.accesses());
    if (recognized.result.state != RecognitionState::Applicable) {
        result.error = "finite guarded analysis has unsupported control or additional prerequisites"; return result;
    }
    if (recognized.phases.size() > UINT32_MAX / 2 ||
        recognized.phases.size() > std::numeric_limits<std::size_t>::max() / 2) {
        result.error = "finite guarded event identity overflow"; return result;
    }
    auto state = std::make_shared<FiniteGuardedState>();
    state->function = function;
    state->accessModel = &input.accesses();
    state->gmAliasPolicy = input.memory().gmPolicy();
    state->arena = expressions ? std::move(expressions) : std::make_shared<RegionExpressions>();
    std::vector<RegionExpressions::Id> guards;
    for (const auto& guard : recognized.guards) {
        auto literal = state->arena->input(guard.condition);
        if (!guard.takeThen) { literal = state->negate(literal); }
        // An inner predicate is evaluated only on its parent path. Speculating
        // a pure expression may produce poison off that path; select masks it
        // before any eager Boolean closure operation consumes this presence.
        guards.push_back(state->arena->select(guard.parent ? guards[*guard.parent] : state->yes(),
                                              literal,state->no()));
    }
    for (const auto& guarded : recognized.phases) {
        auto* op = guarded.phase->elementOp;
        if (!op->getNextNode()) {
            result.error = "finite guarded payload has no legal after cut"; return result;
        }
        state->anchors.push_back({guarded.phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
        state->presence.push_back(guarded.guard ? guards[*guarded.guard] : state->yes());
        state->arena->forbidRecomputation(op);
    }
    if (!collectEffects(*state, recognized, input)) {
        result.error = "finite guarded analysis has invalid modeled storage references";
        return result;
    }
    state->closeAndReduce();
    state->summarize(input);
    if (!state->arena->constructionError().empty()) { result.error = state->arena->constructionError(); return result; }
    state->cost.children = 1;
    state->cost.ports = state->anchors.size();
    state->cost.crossings = state->retained.size();
    state->cost.cells = state->storageBoundary.size();
    state->cost.expressionNodes = state->arena->size();
    result.cost = state->cost;
    result.state = std::move(state);
    return result;
}
RegionalAnalysis finiteGuardedRegionalResult(const FiniteGuardedAnalysis& analysis)
{
    RegionalAnalysis out;
    auto state = analysis.state;
    if (!state || !analysis.error.empty()) { return out; }
    out.expressions = state->arena;
    out.anchors = state->anchors;
    out.occurrenceLoops.resize(out.anchors.size());
    out.storageBoundary = state->storageBoundary;
    out.firstPayloads = state->firstPayloads; out.lastPayloads = state->lastPayloads;
    out.cost = state->cost;
    out.gmAliasPolicy = state->gmAliasPolicy;
    out.capabilities = {true, true, true, true, true};
    out.accessModel = state->accessModel;
    for (uint32_t type = 0; type < out.anchors.size(); ++type) {
        RegionalSelector selector{{type, state->arena->constant(0), PeriodicEventKind::Start}, state->presence[type]};
        for (auto effect : out.accessModel->effectsFor(out.anchors[type].phase)) {
            out.accessBoundary.push_back({effect, selector, selector,
                out.accessModel->effects()[effect].rangesMaterialized});
        }
    }
    auto valid = [state](RegionalEvent event) {
        return event.type < state->anchors.size() && event.ordinal < state->arena->size() &&
            !state->arena->isBoolean(event.ordinal) &&
            (event.kind == PeriodicEventKind::Start || event.kind == PeriodicEventKind::Completion);
    };
    out.presence = [state, valid](RegionalEvent event) -> std::optional<RegionExpressions::Id> {
        if (!valid(event)) { return std::nullopt; }
        return state->both(state->presence[event.type], state->arena->eq(event.ordinal, state->arena->constant(0)));
    };
    out.reachability = [state, valid](RegionalEvent a, RegionalEvent b) -> std::optional<RegionExpressions::Id> {
        if (!valid(a) || !valid(b)) { return std::nullopt; }
        auto zero = state->arena->constant(0);
        return state->both(state->graph[2*a.type + (a.kind == PeriodicEventKind::Completion)]
            [2*b.type + (b.kind == PeriodicEventKind::Completion)], state->both(
                state->arena->eq(a.ordinal, zero), state->arena->eq(b.ordinal, zero)));
    };
    out.prepare = [state]() { return state->prepare(); };
    out.prepareFiltered = [state](const RegionalDemandFilter& filter) { return state->prepare(filter); };
    return out;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFiniteGuardedInsertion(FiniteGuardedAnalysis& analysis)
{
    if (!analysis.error.empty() || !analysis.state) { return failure(); }
    auto result = analysis.state->prepare();
    if (succeeded(result)) {
        (*result)->allocationCertificate =
            finiteRegionalAllocationCertificate(finiteGuardedRegionalResult(analysis), **result);
    }
    analysis.insertionError = analysis.state->insertionError;
    return result;
}
} // namespace mlir::pto::frontiersynch
