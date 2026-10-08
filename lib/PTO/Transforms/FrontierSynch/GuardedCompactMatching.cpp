// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Presence proof and cut availability are independent obligations.
#include "PTO/Transforms/FrontierSynch/GuardedCompactMatching.h"
#include "CircuitEndpoints.h"
#include "IterationPredicates.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
bool integer(const RegionExpressions& arena, Id value)
{
    return value < arena.size() && !arena.isBoolean(value);
}
bool presence(RegionExpressions& arena, const GuardedCompactPresence& value)
{
    if (value.guard >= arena.size() || !arena.isBoolean(value.guard)) { return false; }
    if (value.enclosing.empty()) { return true; }
    auto conjunction = arena.boolean(true);
    for (auto term : value.enclosing) {
        if (term >= arena.size() || !arena.isBoolean(term)) { return false; }
        conjunction = arena.land(conjunction, term);
    }
    return conjunction == value.guard;
}
GuardedCompactProof prove(RegionExpressions& arena, const GuardedCompactPresence& source,
                         const GuardedCompactPresence& target)
{
    if (arena.constantValue(source.guard) == 1) { return GuardedCompactProof::MandatorySource; }
    if (source.guard == target.guard) { return GuardedCompactProof::IdenticalGuard; }
    if (source.enclosing.empty() || target.enclosing.empty()) { return GuardedCompactProof::Unknown; }
    llvm::DenseSet<Id> conditions(target.enclosing.begin(), target.enclosing.end());
    for (auto term : source.enclosing) {
        if (!conditions.contains(term)) { return GuardedCompactProof::Unknown; }
    }
    return GuardedCompactProof::EnclosingBranch;
}
struct PreparationState {
    PreparationState(scf::ForOp loop, RegionExpressions& arena, ArrayRef<const CompoundInstanceElement*> phases)
        : predicates(loop, arena, phases) {}
    IterationPredicates predicates;
    std::vector<std::vector<std::pair<Value, bool>>> paths;
};
bool collectPaths(scf::ForOp loop, const PhaseIndex& index, ArrayRef<const CompoundInstanceElement*> phases,
                  ArrayRef<PeriodicRecord> records, PreparationState& state,
                  std::vector<TemplateEndpointAnchor>& anchors)
{
    DenseMap<Operation*, uint64_t> positions;
    loop.walk<WalkOrder::PreOrder>([&positions](Operation* operation) {
        positions.try_emplace(operation, positions.size());
    });
    llvm::DenseSet<uint32_t> used;
    for (const auto& record : records) { used.insert(record.source); used.insert(record.target); }
    std::optional<uint64_t> previous;
    for (std::size_t site = 0; site < phases.size(); ++site) {
        const auto* phase = phases[site];
        auto* operation = phase ? phase->elementOp : nullptr;
        if (!operation || !positions.count(operation) || index.phasesFor(operation).size() != 1 ||
            index.phasesFor(operation)[0] != phase || (previous && positions.lookup(operation) <= *previous)) {
            return false;
        }
        previous = positions.lookup(operation);
        std::vector<std::pair<Value, bool>> path;
        auto* region = operation->getParentRegion();
        while (used.contains(static_cast<uint32_t>(site)) && region != &loop.getRegion()) {
            auto branch = dyn_cast_or_null<scf::IfOp>(region ? region->getParentOp() : nullptr);
            if (!branch) { return false; }
            path.push_back({branch.getCondition(), region == &branch.getThenRegion()});
            region = branch->getParentRegion();
        }
        std::reverse(path.begin(), path.end());
        state.paths.push_back(std::move(path));
        anchors.push_back({phase, {}, {operation->getBlock(), operation},
                          {operation->getBlock(), operation->getNextNode()}});
    }
    return true;
}
} // namespace
GuardedCompactMatching buildGuardedCompactMatching(RegionExpressions& arena, Id ordinal, Id trips,
    llvm::ArrayRef<PeriodicPayload> payloads, llvm::ArrayRef<PeriodicRecord> records,
    const GuardedCompactGuards& guards)
{
    GuardedCompactMatching out;
    for (auto record : records) { out.records.push_back({record}); }
    const auto initial = arena.size();
    if (!arena.constructionError().empty() || !integer(arena, ordinal) || !integer(arena, trips) || !guards ||
        payloads.size() > UINT32_MAX || records.size() > UINT32_MAX) {
        out.error = "invalid guarded compact arena, coordinates, guards or record count";
        return out;
    }
    out.sourcePresenceComplete = true;
    for (auto& recipe : out.records) {
        const auto record = recipe.original;
        if (record.source >= payloads.size() || record.target >= payloads.size() ||
            (!record.displacement && record.source >= record.target)) {
            out.error = "nonforward or absent guarded compact endpoint";
            out.sourcePresenceComplete = false;
            return out;
        }
        const auto shift = arena.constant(record.displacement);
        const auto next = arena.add(ordinal, shift);
        auto source = guards(record.source, ordinal), target = guards(record.target, next);
        out.guardEvaluations += 2;
        out.constantBits += std::max(1U, llvm::APInt(64, record.displacement).getActiveBits());
        if (!source || !target) { out.sourcePresenceComplete = false; continue; }
        if (!presence(arena, *source) || !presence(arena, *target)) {
            out.error = "invalid guarded compact participation circuit or structural path";
            out.sourcePresenceComplete = false;
            return out;
        }
        recipe.proof = payloads[record.source].pipe == payloads[record.target].pipe ?
            GuardedCompactProof::Local : prove(arena, *source, *target);
        if (recipe.proof == GuardedCompactProof::Unknown || recipe.proof == GuardedCompactProof::Local) {
            out.sourcePresenceComplete = false;
            continue;
        }
        // The subtraction guard proves natural addition at active source cuts,
        // even for UINT64_MAX displacement. Every arithmetic node stays total.
        const auto inRange = arena.land(arena.lt(ordinal, trips), arena.lt(shift, arena.sub(trips, ordinal)));
        recipe.publication = arena.land(inRange, target->guard);
        recipe.acquisition = arena.le(shift, ordinal);
        recipe.sourceIdentity = ordinal;
        recipe.targetIdentity = arena.sub(ordinal, shift);
    }
    out.addedCircuitNodes = arena.size() - initial;
    if (!arena.constructionError().empty()) {
        out.error = arena.constructionError(); out.sourcePresenceComplete = false;
    }
    return out;
}
GuardedCompactPreparation prepareGuardedCompactMatching(func::FuncOp function, scf::ForOp loop,
    const PhaseIndex& index, llvm::ArrayRef<const CompoundInstanceElement*> phases,
    llvm::ArrayRef<PeriodicRecord> records)
{
    GuardedCompactPreparation out;
    for (auto record : records) { out.matching.records.push_back({record}); }
    auto domain = CountedLoop::get(loop);
    if (!function || !domain || phases.size() > UINT32_MAX ||
        loop->getParentOfType<func::FuncOp>() != function) {
        out.placementError = "guarded compact matching needs an original counted loop"; return out;
    }
    for (auto* parent = loop->getParentOp(); parent && parent != function; parent = parent->getParentOp()) {
        if (isa<LoopLikeOpInterface>(parent)) {
            out.placementError = "guarded compact nested invocation needs enclosing identity coordinates"; return out;
        }
    }
    for (const auto* phase : phases) {
        if (!phase || !phase->elementOp || phase->kPipeValue == PipelineType::PIPE_UNASSIGNED) {
            out.placementError = "guarded compact site has no original phase or pipe"; return out;
        }
    }
    out.expressions = std::make_shared<RegionExpressions>();
    auto& arena = *out.expressions;
    SmallVector<const CompoundInstanceElement*> allPhases;
    function.walk([&index, &allPhases](Operation* operation) {
        for (const auto* phase : index.phasesFor(operation)) { allPhases.push_back(phase); }
    });
    auto state = std::make_shared<PreparationState>(loop, arena, allPhases);
    out.expressionOwner = state;
    std::vector<TemplateEndpointAnchor> anchors;
    if (!collectPaths(loop, index, phases, records, *state, anchors)) {
        out.placementError = "guarded compact sites lack distinct original phase/branch paths"; return out;
    }
    std::vector<PeriodicPayload> payloads;
    for (const auto* phase : phases) { payloads.push_back({static_cast<uint32_t>(phase->kPipeValue)}); }
    const auto ordinal = arena.div(arena.sub(arena.input(loop.getInductionVar()), arena.input(loop.getLowerBound())),
                                    arena.constant(domain->step));
    auto guards = [state, &arena, &index, loop](uint32_t site, Id at) -> std::optional<GuardedCompactPresence> {
        GuardedCompactPresence result;
        result.guard = arena.boolean(true);
        for (auto [condition, takeThen] : state->paths[site]) {
            // A value available before the loop is the same evaluation on every
            // visit. Other predicates carry an explicit iteration recipe.
            auto term = index.valueAvailable(condition, loop, Boundary::Before) ?
                arena.input(condition) : state->predicates.at(condition, at);
            if (!takeThen) { term = arena.lnot(term); }
            result.enclosing.push_back(term);
            result.guard = arena.land(result.guard, term);
        }
        return result;
    };
    out.matching = buildGuardedCompactMatching(arena, ordinal, domain->trips(arena), payloads, records, guards);
    if (!out.matching.error.empty() || !out.matching.sourcePresenceComplete) {
        out.placementError = out.matching.error.empty() ?
            "guarded compact source presence or executed local adjacency is unavailable" : out.matching.error;
        return out;
    }
    CircuitEndpoints emitter(function, arena, anchors);
    emitter.recover = [state](Id root, OpBuilder& builder, Operation* cut, RegionExpressions::CutEmission& context) {
        return state->predicates.recover(root, builder, cut, context);
    };
    for (const auto& recipe : out.matching.records) {
        if (!emitter.add(recipe.original.source, recipe.original.target, recipe.publication, recipe.acquisition,
                         {recipe.sourceIdentity}, {recipe.targetIdentity})) {
            out.placementError = state->predicates.error.empty() ? arena.lastEmissionError() : state->predicates.error;
            if (out.placementError.empty()) { out.placementError = "guarded compact original cut is unavailable"; }
            return out;
        }
    }
    out.plan = emitter.take();
    return out;
}
OptionalIndirectReadExcess optionalIndirectReadExcess(RegionExpressions& arena, Id ordinal, Id participation,
                                                     uint64_t delta, uint64_t trips)
{
    OptionalIndirectReadExcess out;
    if (!integer(arena, ordinal) || participation >= arena.size() || !arena.isBoolean(participation)) { return out; }
    out.contribution = arena.select(participation, arena.minimum(ordinal, arena.constant(delta)), arena.constant(0));
    const llvm::APInt d(128, delta), t(128, trips), first(128, std::min(delta, trips));
    out.allPresent = first.isZero() ? llvm::APInt(128, 0) : (first * (first - 1)).udiv(llvm::APInt(128, 2));
    out.allPresent += (t - first) * d;
    out.deltaTimesTrips = d * t;
    return out;
}
} // namespace mlir::pto::frontiersynch
