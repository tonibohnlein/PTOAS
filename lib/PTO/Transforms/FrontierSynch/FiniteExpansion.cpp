// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A finite occurrence representation over the unchanged shared access model.
#include "ArithmeticProgramInternal.h"
#include "FiniteExpansionPlan.h"
#include "NormalizedControl.h"
#include "FiniteGuardedInternal.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include "RecognitionInternal.h"
#include "../InsertSync/SyncScalarEvolution.h"
#include "../InsertSync/SyncScalarReplay.h"
#include "llvm/Support/MathExtras.h"
namespace mlir::pto::frontiersynch {
FiniteExpansionPlan preflightFiniteExpansion(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, const FiniteExpansionLimits& limits,
    std::shared_ptr<const NormalizedControlDescription> normalized)
{
    FiniteExpansionPlan plan;
    plan.context = std::move(context); plan.index = &index; plan.input = &input; plan.limits = limits;
    auto& sites = plan.sites;
    auto& visits = plan.visits;
    auto region = plan.context;
    const bool valid = region.function && region.root && !region.function.isDeclaration() &&
        region.function.getBody().hasOneBlock() &&
        (region.root == region.function.getOperation() || region.function->isProperAncestor(region.root));
    if (!valid) {
        plan.result.note(RecognitionIssue::UnsupportedControl, region.root, true);
        return plan;
    }
    if (!region.roots.empty()) {
        auto* previous = region.roots.front();
        bool adjacent = previous == region.root;
        for (auto* selected : llvm::drop_begin(region.roots)) {
            adjacent = adjacent && selected && previous && previous->getNextNode() == selected;
            if (!adjacent) { break; }
            previous = selected;
        }
        if (!adjacent) {
            plan.result.note(RecognitionIssue::UnsupportedControl, region.root, true);
            return plan;
        }
    }
    if (!normalized) { normalized = normalizeSmallCountControl(region, index, input, limits); }
    plan.normalized = std::move(normalized);
    const auto& description = *plan.normalized;
    const bool sameContext = description.index == &index && description.input == &input &&
        description.context.function == region.function && description.context.root == region.root &&
        description.context.roots == region.roots && description.result.state == RecognitionState::Applicable;
    if (!sameContext) {
        plan.result.note(RecognitionIssue::TemplateContext, region.root); return plan;
    }
    bool admitted = true;
    auto issue = RecognitionIssue::TemplateExpansionLimit;
    std::function<void(std::size_t, ArithmeticSite, unsigned)> walk;
    walk = [&](std::size_t id, ArithmeticSite context, unsigned depth) {
        if (!admitted) { return; }
        const auto& node = description.nodes[id];
        for (auto fixed : node.fixedCoordinates) {
            auto old = llvm::find_if(context.fixedCoordinates,
                [&](auto coordinate) { return coordinate.loop == fixed.loop; });
            if (old == context.fixedCoordinates.end()) { context.fixedCoordinates.push_back(fixed); }
            else if (old->induction != fixed.induction) {
                issue = RecognitionIssue::TemplateContext; admitted = false; return;
            }
        }
        auto* op = node.original;
        if (!op) { for (auto child : node.children) { walk(child, context, depth); } return; }
        const bool exhausted = depth > limits.depth || visits.size() >= limits.visits;
        if (exhausted) { admitted = false; return; }
        visits.push_back({op, context});
        if (node.kind == NormalizedControlKind::ExpandedLoop) {
            plan.expandsLoops = true;
            for (auto child : node.children) { walk(child, context, depth + 1); }
            return;
        }
        if (node.kind == NormalizedControlKind::RetainedLoop) {
            auto loop = cast<scf::ForOp>(op);
            plan.expandsLoops = true;
            auto lower = detail::normalizedInteger(loop.getLowerBound(), context.fixedCoordinates,
                region.function.getContext());
            auto upper = detail::normalizedInteger(loop.getUpperBound(), context.fixedCoordinates,
                region.function.getContext());
            auto step = detail::normalizedInteger(loop.getStep(), context.fixedCoordinates,
                region.function.getContext());
            if (!lower || !upper || !step || *step <= 0) {
                issue = RecognitionIssue::LoopDomain; admitted = false; return;
            }
            const __int128 distance = static_cast<__int128>(*upper) - *lower;
            const __int128 count = distance <= 0 ? 0 : (distance + *step - 1) / *step;
            if (count > limits.visits - visits.size()) { admitted = false; return; }
            for (__int128 visit = 0; visit < count && admitted; ++visit) {
                auto body = context;
                const auto induction = static_cast<__int128>(*lower) + visit * *step;
                if (induction + *step > INT64_MAX) { admitted = false; return; }
                body.fixedCoordinates.push_back({loop, static_cast<int64_t>(induction)});
                for (auto child : node.children) { walk(child, body, depth + 1); }
            }
            return;
        }
        if (node.kind == NormalizedControlKind::Conditional) {
            auto branch = cast<scf::IfOp>(op);
            const auto selected = detail::normalizedCondition(branch.getCondition(), context, plan.foldOperations);
            visits.back().fixedBranch = selected.has_value();
            for (auto [arm, child] : llvm::enumerate(node.children)) {
                const bool inactive = selected && ((arm == 0) != *selected);
                if (inactive) {
                    if (plan.prunedArms != UINT64_MAX) { ++plan.prunedArms; }
                    continue;
                }
                auto body = context;
                body.guards.push_back({branch, arm == 0, selected.has_value()});
                walk(child, body, depth + 1);
            }
            return;
        }
        const bool functionRoot = op == region.function.getOperation();
        if (functionRoot) {
            for (auto child : node.children) { walk(child, context, depth); }
            return;
        }
        if (op->getNumRegions()) { issue = RecognitionIssue::UnsupportedControl; admitted = false; return; }
        auto phases = index.phasesFor(op);
        const bool multiple = phases.size() > 1;
        if (multiple) { issue = RecognitionIssue::UnsupportedControl; admitted = false; return; }
        const bool payload = phases.size() == 1;
        if (payload) {
            const bool full = sites.size() >= limits.payloads;
            if (full) { admitted = false; return; }
            context.phase = phases.front(); sites.push_back(std::move(context));
        }
    };
    for (auto root : description.roots) { walk(root, {}, 0); }
    const auto count = static_cast<uint64_t>(sites.size());
    const bool pairsFit = !count || count <= limits.pairs / count;
    if (!admitted || !pairsFit) {
        plan.result.note(issue, region.root);
        plan.sites.clear(); plan.visits.clear();
        return plan;
    }
    return plan;
}
} // namespace mlir::pto::frontiersynch
namespace mlir::pto::frontiersynch::detail {
void collectExpanded(ProgramBuilder& builder, const FiniteExpansionLimits& limits,
    const FiniteExpansionPlan* supplied)
{
    std::optional<FiniteExpansionPlan> local;
    if (!supplied) {
        local = preflightFiniteExpansion(builder.output.context, builder.index, *builder.output.modeledInput, limits);
        supplied = &*local;
    }
    const auto& plan = *supplied;
    auto& output = builder.output;
    output.extraction = plan.result;
    if (plan.result.state != RecognitionState::Applicable) { return; }
    output.expandedVisits = plan.visits.size();
    output.expandedFoldOperations = plan.foldOperations;
    output.expandedPrunedArms = plan.prunedArms;
    output.sites = plan.sites;
    for (const auto& [op, context, fixedBranch] : plan.visits) {
        for (auto prerequisite : builder.index.prerequisitesFor(op)) {
            auto* producer = prerequisite.producer->elementOp;
            const bool internal = output.context.roots.empty() ? output.context.root->isAncestor(producer) :
                llvm::any_of(output.context.roots, [&](Operation* selected) { return selected->isAncestor(producer); });
            if (!internal) {
                output.incomingPrerequisites.push_back(prerequisite);
            }
        }
        const bool controlPrerequisite = op->getNumRegions() && builder.index.needsValuePrerequisite(op);
        if (controlPrerequisite) {
            output.extraction.note(RecognitionIssue::AdditionalPrerequisite, op);
        }
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
            const bool prepared = fixedBranch || builder.prepareGuard(branch.getCondition(), context);
            if (!prepared) {
                output.extraction.note(RecognitionIssue::IndexArithmetic, op);
            }
        } else if (!op->getNumRegions()) {
            inspectLeaf(*op, builder.index, output.extraction);
        }
    }
}
} // namespace mlir::pto::frontiersynch::detail

namespace mlir::pto::frontiersynch {
static FiniteGuardedAnalysis analyzeExpandedFiniteContext(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, const FiniteExpansionPlan* plan = nullptr,
    std::shared_ptr<RegionExpressions> expressions = {})
{
    FiniteGuardedAnalysis result;
    auto program = std::make_shared<ArithmeticProgram>(
        plan ? materializeFiniteExpansion(*plan, index, input) :
            expandFiniteArithmeticProgram(std::move(context), index, input, input.accesses()));
    if (program->extraction.state != RecognitionState::Applicable ||
        program->recognition.state != RecognitionState::Applicable) {
        result.error = "finite expansion requires a bounded exact occurrence/access adapter";
        for (const auto& issue : program->extraction.diagnostics) {
            result.error += " / " + recognitionName(issue.issue).str();
        }
        return result;
    }
    uint64_t accesses = 0, overlapPairs = 0;
    DenseMap<Value, std::map<unsigned, std::pair<uint64_t, uint64_t>>> groups;
    for (const auto& relation : program->primitives.relations) {
        const bool read = relation.kind == PrimitiveKind::Reads;
        const bool write = relation.kind == PrimitiveKind::Writes;
        const bool access = (read || write) && !relation.pieces.empty();
        if (!access) { continue; }
        accesses += relation.pieces.size();
        auto& count = groups[relation.storageBase][static_cast<unsigned>(*relation.storageSpace)];
        (read ? count.first : count.second) += relation.pieces.size();
    }
    const auto limits = plan ? plan->limits : FiniteExpansionLimits{};
    for (const auto& entry : groups) {
        for (auto [space, count] : entry.second) {
            (void)space;
            // Counts are bounded by the primitive-fragment cap, so products
            // fit uint64_t. Read/read pairs never enter the overlap join.
            overlapPairs += count.second * (count.second + 2 * count.first);
        }
    }
    if (overlapPairs > limits.pairs) {
        result.error = "finite expansion overlap adapter exceeds its pair budget"; return result;
    }
    auto protection = structuredProtection(input.accesses());
    llvm::MapVector<std::pair<AddressSpace, Value>, AffineExpr> translations;
    auto translated = detail::normalizeFiniteDemandAccesses(*program, &program->expandedTranslationFragments,
                                                          &translations);
    auto stage = analyzeGeneralArithmeticGenerators(translated ? *translated : *program, &protection);
    if (!stage.analysis().error.empty()) { result.error = stage.analysis().error; return result; }
    auto arena = expressions ? std::move(expressions) : std::make_shared<RegionExpressions>();
    if (!arena->constructionError().empty()) {
        result.error = "finite expansion requires an unpoisoned expression context"; return result;
    }
    const auto initialExpressions = arena->size();
    RegionExpressions::Transaction transaction(*arena);
    auto state = std::make_shared<FiniteGuardedState>();
    state->expandedStorageProgram = translated ?
        std::make_shared<const ArithmeticProgram>(std::move(*translated)) : program;
    state->expandedStorageTranslations = std::move(translations);
    state->function = program->context.function;
    state->arena = arena;
    state->accessModel = &input.accesses();
    state->gmAliasPolicy = input.memory().gmPolicy();
    SmallVector<RegionExpressions::Id> parameters;
    for (auto value : program->parameters) { parameters.push_back(state->arena->input(value)); }
    for (auto [id, site] : llvm::enumerate(program->sites)) {
        auto* op = site.phase->elementOp;
        state->anchors.push_back({site.phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
        state->presence.push_back(state->no());
        ExplicitEffects effect;
        effect.payload = id;
        effect.pipe = static_cast<uint32_t>(site.phase->kPipeValue);
        state->effects.push_back(std::move(effect));
    }
    auto predicate = [&](const IntegerSystem& system, ArrayRef<uint64_t> residues) {
        return state->arena->integerPredicate(system, parameters, program->primitives.period, residues);
    };
    for (const auto& domain : stage.occurrences()) {
        const bool invalidDomain = domain.site >= state->presence.size() || !domain.residues.empty();
        if (invalidDomain) {
            result.error = "finite expansion has a free occurrence coordinate"; return result;
        }
        state->presence[domain.site] = state->either(state->presence[domain.site],
            predicate(domain.system, domain.parameterResidues));
    }
    auto import = [&](const GeneralArithmeticRelation& relation, bool native) {
        for (const auto& [key, pieces] : relation) {
            if (key.source.event != ArithmeticEvent::Completion || key.target.event != ArithmeticEvent::Start) {
                continue;
            }
            auto guard = state->no();
            for (const auto& piece : pieces) { guard = state->either(guard, predicate(piece, key.parameterResidues)); }
            auto& edges = native ? state->guardedNative : state->guardedResidual;
            edges.push_back({static_cast<uint32_t>(key.source.site), static_cast<uint32_t>(key.target.site), guard});
        }
    };
    import(stage.analysis().generators, false);
    import(stage.analysis().nativeOrder, true);
    state->cost.numericVisits = program->expandedVisits;
    state->cost.phaseDescriptions = program->sites.size();
    state->cost.physicalFragments = accesses;
    state->cost.crossingCandidates = stage.analysis().cost.pieceJoins;
    state->cost.crossings = state->guardedResidual.size();
    state->closeAndReduce();
    state->cost.expressionNodes = state->arena->size() - initialExpressions;
    const bool failedReduction = !state->rankIndex.error.empty() || !state->arena->error().empty();
    if (failedReduction) {
        result.error = state->rankIndex.error.empty() ? state->arena->error() : state->rankIndex.error;
        return result;
    }
    transaction.commit();
    result.state = std::move(state);
    result.expandedProgram = std::move(program);
    result.cost = result.state->cost;
    return result;
}
FiniteGuardedAnalysis analyzeExpandedFinite(const FiniteExpansionPlan& plan,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions)
{
    return analyzeExpandedFiniteContext(plan.context, index, input, &plan, std::move(expressions));
}
FiniteGuardedAnalysis analyzeExpandedFinite(func::FuncOp function, Operation* root,
    const PhaseIndex& index, const SyncInput& input)
{
    return analyzeExpandedFiniteContext({function, root}, index, input);
}
FiniteGuardedAnalysis analyzeExpandedFinite(func::FuncOp function, ArrayRef<Operation*> roots,
    const PhaseIndex& index, const SyncInput& input)
{
    if (roots.empty()) {
        FiniteGuardedAnalysis result;
        result.error = "finite expansion root list is empty";
        return result;
    }
    ArithmeticRegionContext context{function, roots.front()};
    llvm::append_range(context.roots, roots);
    return analyzeExpandedFiniteContext(std::move(context), index, input);
}
} // namespace mlir::pto::frontiersynch
