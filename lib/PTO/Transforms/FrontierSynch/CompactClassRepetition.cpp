// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CompactClassRepetition.h"
#include "CountedLoop.h"
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "../InsertSync/SyncScalarReplay.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
class UniformInvocation {
public:
    UniformInvocation(scf::ForOp loop, const PhaseIndex& index) : loop(loop), index(index) {}
    uint64_t values = 0;
    bool value(Value current)
    {
        if (!current) { return false; }
        if (auto found = memo.find(current); found != memo.end()) { return found->second; }
        memo[current] = false; ++values;
        if (auto argument = dyn_cast<BlockArgument>(current)) {
            auto* owner = argument.getOwner()->getParentOp();
            if (!owner || owner == loop) { return false; }
            if (!loop->isProperAncestor(owner)) { return memo[current] = true; }
            // A nested IV denotes the same inner coordinate in each outer
            // invocation. Its bounds are checked separately over the entire body.
            auto inner = dyn_cast<scf::ForOp>(owner);
            return memo[current] = inner && argument == inner.getInductionVar();
        }
        auto* definition = current.getDefiningOp();
        if (!definition) { return false; }
        if (!loop->isProperAncestor(definition)) { return memo[current] = true; }
        if (!mlir::pto::detail::canReplayScalar(definition) || !index.phasesFor(definition).empty()) { return false; }
        for (auto operand : definition->getOperands()) { if (!value(operand)) { return false; } }
        return memo[current] = true;
    }
    bool check()
    {
        bool valid = !index.hasRelevantCarriedState(loop);
        loop.getBody()->walk([&](Operation* operation) {
            if (auto inner = dyn_cast<scf::ForOp>(operation)) {
                valid &= !index.hasRelevantCarriedState(inner) && value(inner.getLowerBound()) &&
                    value(inner.getUpperBound()) && value(inner.getStep());
            } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
                valid &= value(branch.getCondition());
            }
        });
        return valid;
    }
private:
    scf::ForOp loop;
    const PhaseIndex& index;
    DenseMap<Value, bool> memo;
};
class BodyBuilder {
public:
    BodyBuilder(func::FuncOp function, const SyncInput& input, std::shared_ptr<RegionExpressions> arena,
                CompactClassPreparation prepare = {})
        : function(function), input(input), arena(std::move(arena)), prepare(std::move(prepare)) {}
    PhaseIndex index;
    std::vector<CompactClasses> captured;
    std::string error;
    CompactClasses body(Block& invocation)
    {
        std::vector<CompactClasses> children;
        SmallVector<Operation*> run;
        auto flush = [&]() {
            if (run.empty()) { return true; }
            const auto contract = recognizeExplicitRun(run, index, input.accesses());
            for (const auto& diagnostic : contract.diagnostics) {
                if (diagnostic.issue != RecognitionIssue::AdditionalPrerequisite) {
                    error = "class repetition has an unsupported shared leaf/control contract"; return false;
                }
            }
            SmallVector<const CompoundInstanceElement*> phases;
            for (auto* operation : run) { llvm::append_range(phases, index.phasesFor(operation)); }
            run.clear();
            if (phases.empty()) { return true; }
            auto leaf = finite(invocation, phases);
            if (!leaf) { return false; }
            captured.push_back(leaf); children.push_back(std::move(leaf)); return true;
        };
        for (Operation& operation : invocation) {
            if (auto inner = dyn_cast<scf::ForOp>(operation)) {
                if (!flush()) { return {}; }
                auto child = region(inner);
                if (!child) { return {}; }
                captured.push_back(child); children.push_back(std::move(child));
            } else { run.push_back(&operation); }
        }
        if (!flush()) { return {}; }
        if (children.empty()) { return finite(invocation, {}); }
        if (children.size() == 1) { return children.front(); }
        auto combined = composeCompactClassBoundariesInBlock(function, invocation, input, std::move(children));
        if (!combined.boundary) { error = combined.error; return {}; }
        captured.push_back(combined.boundary);
        return combined.boundary;
    }
private:
    CompactClasses finite(Block& invocation, llvm::ArrayRef<const CompoundInstanceElement*> phases)
    {
        if (nextProducer == UINT64_MAX) { error = "class repetition producer identity overflow"; return {}; }
        std::vector<RequirementGroupId> groups(input.accesses().cells().size(), 1);
        auto frame = captureFiniteRequirementsInBlock(function, invocation, input, phases, arena,
            nextProducer++, groups, 2, error);
        auto result = frame ? captureFiniteClassBoundary(frame, {}, error) : CompactClasses{};
        return result && prepare ? prepare(result) : result;
    }
    CompactClasses region(scf::ForOp loop)
    {
        bool nested = false;
        loop.getBody()->walk([&](scf::ForOp) { nested = true; });
        if (nested) {
            auto inner = body(*loop.getBody());
            if (!inner) { return {}; }
            auto repeated = repeatCompactClassBoundary(function, loop, inner);
            if (!repeated.boundary) { error = repeated.error; return {}; }
            return repeated.boundary;
        }
        const auto counted = CountedLoop::get(loop);
        if (!counted) { error = "compact class child needs a represented counted domain"; return {}; }
        auto context = captureCompactSlotContextInBlock(loop, *loop->getBlock(), input, index,
            arena, counted->trips(*arena), error);
        auto result = context ? captureCompactClassBoundary(loop, index, context, {}, error) : CompactClasses{};
        return result && prepare ? prepare(result) : result;
    }
    func::FuncOp function;
    const SyncInput& input;
    std::shared_ptr<RegionExpressions> arena;
    uint64_t nextProducer = 1;
    CompactClassPreparation prepare;
};
} // namespace
CompactClassRepetition repeatCompactClassBoundary(func::FuncOp function, scf::ForOp loop, CompactClasses body)
{
    CompactClassRepetition result;
    result.original = body;
    if (!function || function.isDeclaration() || !loop || loop->getParentOfType<func::FuncOp>() != function ||
        !body || !body->bounds().context || body->invocationBlock() != loop.getBody()) {
        result.error = "class repetition requires its original relative body invocation"; return result;
    }
    const auto& input = body->bounds().context->input();
    auto arena = body->bounds().context->expressions();
    PhaseIndex index;
    if (failed(index.build(function, input))) {
        result.error = "class repetition shared index unavailable"; return result;
    }
    if (!body->exportError().empty()) { result.error = body->exportError(); return result; }
    llvm::DenseSet<const CompoundInstanceElement*> included(body->originalPhases.begin(), body->originalPhases.end());
    for (const auto* phase : input.instructions()) {
        if (loop->isProperAncestor(phase->elementOp) != included.contains(phase)) {
            result.error = "class repetition schema is not the complete original loop body"; return result;
        }
    }
    bool missingInterface = false;
    loop.getBody()->walk([&](Operation* operation) {
        if (!index.phasesFor(operation).empty()) { return; }
        missingInterface |= index.needsValuePrerequisite(operation);
        for (const auto& requirement : index.prerequisitesFor(operation)) {
            missingInterface |= !requirement.native;
        }
    });
    if (missingInterface) {
        result.error = "class repetition phase-less completion prerequisite needs an interface mapping";
        return result;
    }
    UniformInvocation invariant(loop, index);
    const bool uniform = invariant.check();
    result.scalarValues = invariant.values;
    if (!uniform) {
        result.error = "class repetition needs uniform inner counts/control and no unresolved carried prerequisite";
        return result;
    }
    const auto counted = CountedLoop::get(loop);
    if (!counted) { result.error = "class repetition counted domain unavailable"; return result; }
    const auto trips = counted->trips(*arena);
    // Two schema descriptions select the same per-visit word. Their distinct
    // child labels are only for computing next-visit bridges; no IR is cloned.
    auto links = collectCompactClassCrossings({body, body});
    result.accessPairs = links.accessPairs; result.effectPairs = links.effectPairs;
    if (!links.error.empty()) { result.error = links.error; return result; }
    BoundingRepetitionInput specification;
    specification.trips = trips;
    for (auto* ancestor = loop->getParentOp(); ancestor; ancestor = ancestor->getParentOp()) {
        if (auto enclosing = dyn_cast<scf::ForOp>(ancestor)) { specification.enclosing.push_back(enclosing); }
    }
    std::reverse(specification.enclosing.begin(), specification.enclosing.end());
    specification.lower.bridges = specification.upper.bridges = BoundingSequenceBridges::SuppliedCrossings;
    specification.guarantee = InputOrderGuarantee::InputOrderCovering;
    BoundingSequenceChild phase;
    phase.bounds = body->bounds(); phase.lowerExports = phase.upperExports = body->nativeExports();
    phase.lowerExports->prepare = {}; phase.lowerExports->prepareWithVisits = {};
    phase.lowerExports->prepareFiltered = {}; phase.lowerExports->capabilities.endpointRecipes = false;
    phase.mathematicalOwner = body; phase.placementMayStrengthen = true;
    specification.phases.push_back(std::move(phase));
    for (const auto& link : links.upper) {
        RepeatedCrossing crossing;
        crossing.source = {link.source.type, link.source.ordinal, PeriodicEventKind::Completion, link.source.visits};
        crossing.target = {link.target.type, link.target.ordinal, PeriodicEventKind::Start, link.target.visits};
        crossing.guard = link.active;
        specification.upper.crossings.push_back({std::move(crossing), link.owners});
    }
    // Native target start chains extend every next-visit requirement to every
    // later visit. Uniform presence is essential: no absent intermediate slot
    // is used as a forwarding point. Empty bodies create no storage links.
    auto repeated = repeatBoundingRegion(function, loop, input, std::move(specification));
    result.mathematical = std::make_shared<const BoundingRepetitionResult>(std::move(repeated));
    if (!result.mathematical->error.empty()) { result.error = result.mathematical->error; return result; }
    if (!result.mathematical->bounds.upper || !result.mathematical->bounds.lower ||
        !result.mathematical->upper.repeated) {
        result.error = !result.mathematical->upper.exportError.empty() ? result.mathematical->upper.exportError :
            result.mathematical->lower.exportError;
        return result;
    }
    auto out = std::shared_ptr<CompactClassBoundary>(new CompactClassBoundary());
    out->order = result.mathematical->bounds;
    out->selectors = result.mathematical->upper.repeated->regional;
    out->repetition = result.mathematical;
    out->effects = body->effects; out->sites = body->sites; out->originalPhases = body->originalPhases;
    out->firstAnchor = out->lastAnchor = loop; out->invocation = loop->getBlock();
    const auto zero = arena->constant(0), active = arena->lt(zero, trips);
    const auto last = arena->select(active, arena->sub(trips, arena->constant(1)), zero);
    for (auto& access : out->sites) {
        access.first.event.visits.insert(access.first.event.visits.begin(), zero);
        access.last.event.visits.insert(access.last.event.visits.begin(), last);
        access.first.present = arena->land(active, access.first.present);
        access.last.present = arena->land(active, access.last.present);
    }
    out->selectors.capabilities.completeStorageModel = false;
    out->selectors.capabilities.exactSelectors = false;
    if (!arena->constructionError().empty()) { result.error = arena->constructionError(); return result; }
    result.boundary = std::move(out);
    return result;
}
CompactClassRepetition analyzeCompactClassRepetition(func::FuncOp function, scf::ForOp loop,
    const SyncInput& input, std::shared_ptr<RegionExpressions> arena)
{
    CompactClassRepetition result;
    if (!function || function.isDeclaration() || !loop || loop->getParentOfType<func::FuncOp>() != function ||
        !arena || !arena->constructionError().empty()) {
        result.error = "class repetition analysis needs an original loop and valid shared arena"; return result;
    }
    BodyBuilder builder(function, input, std::move(arena));
    if (failed(builder.index.build(function, input))) {
        result.error = "class repetition shared input indexing failed"; return result;
    }
    auto body = builder.body(*loop.getBody());
    if (!body) { result.error = builder.error; result.captured = std::move(builder.captured); return result; }
    result = repeatCompactClassBoundary(function, loop, std::move(body));
    result.captured = std::move(builder.captured);
    return result;
}
CompactClassRepetition captureCompactClassSequence(func::FuncOp function, Block& invocation,
    const SyncInput& input, std::shared_ptr<RegionExpressions> arena, CompactClassPreparation prepare)
{
    CompactClassRepetition result;
    if (!function || function.isDeclaration() || !arena || !invocation.getParentOp() ||
        (invocation.getParentOp() != function && !function->isProperAncestor(invocation.getParentOp()))) {
        result.error = "compact block construction needs its original function invocation"; return result;
    }
    BodyBuilder builder(function, input, std::move(arena), std::move(prepare));
    if (failed(builder.index.build(function, input))) {
        result.error = "compact block shared input indexing failed"; return result;
    }
    result.boundary = builder.body(invocation);
    result.original = result.boundary;
    result.error = builder.error; result.captured = std::move(builder.captured);
    return result;
}
} // namespace mlir::pto::frontiersynch
