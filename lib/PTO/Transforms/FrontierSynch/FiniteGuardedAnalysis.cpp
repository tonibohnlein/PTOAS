// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "FiniteGuardedInternal.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
bool collectEffects(FiniteGuardedState& state, const GuardedRecognition& recognized, const SyncInput& input)
{
    const auto invocation = structuredProtection(input.accesses());
    SmallVector<const CompoundInstanceElement*> phases;
    for (const auto& guarded : recognized.phases) {
        phases.push_back(guarded.phase);
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
        for (auto [id, mode] : modes) {
            occurrence.accesses.push_back(mode);
        }
        if (auto group = invocation.at(phase)) {
            for (auto& access : occurrence.accesses) {
                if (input.accesses().cells()[access.atom].space == AddressSpace::ACC) {
                    access.protectionGroup = group;
                }
            }
        }
        state.effects.push_back(std::move(occurrence));
    }
    const auto modeledGroups = modeledProtectionGroups(input, phases, invocation);
    for (uint32_t a = 0; a < recognized.phases.size(); ++a) {
        for (uint32_t b = a + 1; b < recognized.phases.size(); ++b) {
            if (recognized.phases[a].phase->elementOp == recognized.phases[b].phase->elementOp ||
                ptoStorageProtection().protectsScalar(state.effects[a].pipe, state.effects[b].pipe)) { continue; }
            bool conflict = false;
            for (auto x : input.accesses().effectsFor(recognized.phases[a].phase)) {
                for (auto y : input.accesses().effectsFor(recognized.phases[b].phase)) {
                    const bool protectedPair = hardwareProtectsConflict(
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
    Block* block = roots.empty() || !roots.front() ? nullptr : roots.front()->getBlock();
    for (auto* root : roots) {
        if (!root || !block || root->getBlock() != block) { return false; }
        const bool originalFunction = root->getParentOfType<func::FuncOp>() == function;
        if (!originalFunction || (previous && previous->getNextNode() != root)) {
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
        result.error = "finite guarded analysis requires consecutive roots in one original function block";
        return result;
    }
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (bits.isScalable() || bits.getFixedValue() != 64) {
        result.error = "finite guarded endpoint arithmetic requires a 64-bit index representation"; return result;
    }
    auto recognized = recognizeFiniteGuarded(roots, index, input.accesses());
    if (recognized.result.state != RecognitionState::Applicable) {
        result.error = "finite guarded analysis cannot export this region";
        for (const auto& diagnostic : recognized.result.diagnostics) {
            result.error += " / " + recognitionName(diagnostic.issue).str();
        }
        return result;
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
    SmallVector<const CompoundInstanceElement*> phases;
    for (const auto& anchor : state->anchors) { phases.push_back(anchor.phase); }
    auto prerequisites = index.mapPrerequisites(phases);
    if (!prerequisites.error.empty()) { result.error = prerequisites.error; return result; }
    state->nativePrerequisites = std::move(prerequisites.native);
    llvm::append_range(state->residual, prerequisites.demands);
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
            regionalAllocationCertificate(finiteGuardedRegionalResult(analysis), **result);
    }
    analysis.insertionError = analysis.state->insertionError;
    return result;
}
} // namespace mlir::pto::frontiersynch
