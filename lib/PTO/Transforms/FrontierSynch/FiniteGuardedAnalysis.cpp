// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "FiniteGuardedInternal.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
bool independentGlobal(std::size_t id, const SyncInput& input)
{
    const auto& effect = input.accesses().effects()[id];
    if (!effect.memory || effect.memory->scope != AddressSpace::GM) { return false; }
    for (std::size_t other = 0; other < input.accesses().effects().size(); ++other) {
        if (input.accesses().effects()[other].phase != effect.phase && input.accesses().mayConflict(id, other)) {
            return false;
        }
    }
    return true;
}
bool collectEffects(FiniteGuardedState& state, const GuardedRecognition& recognized, const SyncInput& input)
{
    HardwareProtectionBuilder protection;
    std::optional<std::size_t> previousGuard;
    Operation* previous = nullptr;
    for (const auto& guarded : recognized.phases) {
        if (previousGuard != guarded.guard) { protection.endScope(); }
        previousGuard = guarded.guard;
        auto* operation = guarded.phase->elementOp;
        if (previous && previous->getBlock() == operation->getBlock()) {
            for (auto* between = previous->getNextNode(); between && between != operation;
                 between = between->getNextNode()) {
                if (between->getNumRegions()) { protection.endScope(); }
            }
        } else { protection.endScope(); }
        previous = operation;
        auto* phase = guarded.phase;
        ExplicitEffects occurrence;
        occurrence.payload = state.effects.size();
        occurrence.pipe = static_cast<uint32_t>(phase->kPipeValue);
        std::map<uint32_t, CellAccess> modes;
        for (auto id : input.accesses().effectsFor(phase)) {
            const auto& effect = input.accesses().effects()[id];
            if (independentGlobal(id, input)) { continue; }
            if (effect.precision != SyncAccessPrecision::Exact || !effect.exactRanges ||
                (effect.memory && effect.memory->scope == AddressSpace::GM)) { return false; }
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
    for (const auto& diagnostic : recognized.result.diagnostics) {
        if (diagnostic.issue != RecognitionIssue::InexactFootprint &&
            diagnostic.issue != RecognitionIssue::SymbolicGeometry &&
            diagnostic.issue != RecognitionIssue::UnknownGeometry) {
            result.error = "finite guarded analysis has unsupported control or additional prerequisites"; return result;
        }
    }
    if (recognized.phases.size() > UINT32_MAX / 2 ||
        recognized.phases.size() > std::numeric_limits<std::size_t>::max() / 2) {
        result.error = "finite guarded event identity overflow"; return result;
    }
    auto state = std::make_shared<FiniteGuardedState>();
    state->function = function;
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
        result.error = "finite guarded analysis requires exact physical ranges or globally independent GM effects";
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
    out.capabilities = {true, true, true, true, true};
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
    return out;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFiniteGuardedInsertion(FiniteGuardedAnalysis& analysis)
{
    if (!analysis.error.empty() || !analysis.state) { return failure(); }
    auto result = analysis.state->prepare();
    analysis.insertionError = analysis.state->insertionError;
    return result;
}
} // namespace mlir::pto::frontiersynch
