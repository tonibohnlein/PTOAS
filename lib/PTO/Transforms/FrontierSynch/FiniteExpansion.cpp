// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A finite occurrence representation over the unchanged shared access model.
#include "ArithmeticProgramInternal.h"
#include "FiniteGuardedInternal.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include "RecognitionInternal.h"
#include "../InsertSync/SyncScalarEvolution.h"
#include "../InsertSync/SyncScalarReplay.h"
#include "llvm/Support/MathExtras.h"
namespace mlir::pto::frontiersynch::detail {
namespace {
// This memo belongs to exactly one fixed-coordinate environment. Dialect
// folders operate on detached copies and never rewrite the shared input IR.
class FixedCondition {
public:
    FixedCondition(const ArithmeticSite& site, uint64_t& work) : site(site), work(work) {}
    std::optional<bool> evaluate(Value value)
    {
        auto result = dyn_cast_or_null<IntegerAttr>(fold(value, 0));
        if (!result || !result.getType().isInteger(1)) { return std::nullopt; }
        return result.getValue().isOne();
    }
private:
    const ArithmeticSite& site;
    uint64_t& work;
    DenseMap<Value, Attribute> memo;
    Attribute fold(Value value, unsigned depth)
    {
        if (!value || depth >= 64) { return {}; }
        auto inserted = memo.try_emplace(value, Attribute{});
        if (!inserted.second) { return inserted.first->second; }
        auto remember = [&](Attribute result) { memo[value] = result; return result; };
        for (auto coordinate : site.fixedCoordinates) {
            if (value == coordinate.loop.getInductionVar()) {
                return remember(IntegerAttr::get(value.getType(), coordinate.induction));
            }
        }
        Attribute literal;
        if (matchPattern(value, m_Constant(&literal))) {
            // Poison/undefined fold results are not concrete scalar values.
            if (isa<IntegerAttr, FloatAttr>(literal)) { return remember(literal); }
            return {};
        }
        auto* operation = value.getDefiningOp();
        const bool replay = mlir::pto::detail::canReplayScalar(operation);
        const bool arithmetic = replay && isa<arith::ArithDialect>(operation->getDialect());
        const bool singleResult = arithmetic && operation->getNumResults() == 1;
        if (!singleResult) {
            return {};
        }
        // LLVM folders may ignore poison-producing overflow/fast-math
        // promises. Until separately proved, those operations stay symbolic.
        if (auto flags = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(operation)) {
            const bool unflagged = flags.getOverflowAttr().getValue() == arith::IntegerOverflowFlags::none;
            if (!unflagged) { return {}; }
        }
        if (auto flags = dyn_cast<arith::ArithFastMathInterface>(operation)) {
            const bool unflagged = flags.getFastMathFlagsAttr().getValue() == arith::FastMathFlags::none;
            if (!unflagged) { return {}; }
        }
        const bool usesIndex = value.getType().isIndex() ||
            llvm::any_of(operation->getOperandTypes(), [](Type type) { return type.isIndex(); });
        if (usesIndex) {
            const auto width = DataLayout::closest(operation).getTypeSizeInBits(IndexType::get(value.getContext()));
            const bool supportedWidth = !width.isScalable() &&
                width.getFixedValue() == IndexType::kInternalStorageBitWidth;
            if (!supportedWidth) { return {}; }
        }
        SmallVector<Attribute> operands;
        for (Value operand : operation->getOperands()) {
            auto constant = fold(operand, depth + 1);
            if (!constant) { return {}; }
            operands.push_back(constant);
        }
        if (work != UINT64_MAX) { ++work; }
        OwningOpRef<Operation*> copy(operation->cloneWithoutRegions());
        SmallVector<OpFoldResult> results;
        const bool folded = succeeded(copy->fold(operands, results));
        const bool singleFold = folded && results.size() == 1;
        if (!singleFold) { return {}; }
        auto result = dyn_cast<Attribute>(results.front());
        auto typed = dyn_cast_or_null<TypedAttr>(result);
        const bool scalar = typed && isa<IntegerAttr, FloatAttr>(result);
        const bool sameType = scalar && typed.getType() == value.getType();
        if (!sameType) { return {}; }
        return remember(result);
    }
};
std::optional<int64_t> fixedValue(Value value, const ArithmeticSite& site, MLIRContext* context)
{
    mlir::pto::detail::ScalarEvolution evolution(context, value.getDefiningOp() ? value.getDefiningOp() :
        value.getParentRegion()->getParentOp());
    auto expression = evolution.value(value, [&](Value current) -> AffineExpr {
        for (auto fixed : site.fixedCoordinates) {
            auto loop = fixed.loop;
            if (current == loop.getInductionVar()) {
                return getAffineConstantExpr(fixed.induction, context);
            }
        }
        return {};
    });
    auto constant = dyn_cast_or_null<AffineConstantExpr>(expression);
    return constant ? std::optional<int64_t>(constant.getValue()) : std::nullopt;
}
}
void collectExpanded(ProgramBuilder& builder, const FiniteExpansionLimits& limits)
{
    auto& output = builder.output;
    SmallVector<ArithmeticSite> sites;
    struct Visit { Operation* op; ArithmeticSite context; bool fixedBranch = false; };
    SmallVector<Visit> visits;
    bool admitted = true;
    auto issue = RecognitionIssue::TemplateExpansionLimit;
    std::function<void(Operation*, ArithmeticSite, unsigned)> walk;
    walk = [&](Operation* op, ArithmeticSite context, unsigned depth) {
        if (!admitted) { return; }
        const bool exhausted = depth > limits.depth || visits.size() >= limits.visits;
        if (exhausted) {
            admitted = false;
            return;
        }
        visits.push_back({op, context});
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
            auto lower = fixedValue(loop.getLowerBound(), context, builder.context);
            auto upper = fixedValue(loop.getUpperBound(), context, builder.context);
            auto step = fixedValue(loop.getStep(), context, builder.context);
            if (!lower || !upper || !step || *step <= 0) {
                issue = RecognitionIssue::LoopDomain; admitted = false; return;
            }
            // Wide arithmetic proves count and every actual IV before narrowing.
            const __int128 distance = static_cast<__int128>(*upper) - *lower;
            const __int128 count = distance <= 0 ? 0 : (distance + *step - 1) / *step;
            if (count > limits.visits - visits.size()) { admitted = false; return; }
            for (__int128 visit = 0; visit < count && admitted; ++visit) {
                auto body = context;
                const auto induction = static_cast<__int128>(*lower) + visit * *step;
                // The final increment must also be representable in index width.
                if (induction + *step > INT64_MAX) { admitted = false; return; }
                body.fixedCoordinates.push_back({loop, static_cast<int64_t>(induction)});
                for (auto& child : loop.getBody()->getOperations()) { walk(&child, body, depth + 1); }
            }
            return;
        }
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
            FixedCondition condition(context, output.expandedFoldOperations);
            const auto selected = condition.evaluate(branch.getCondition());
            visits.back().fixedBranch = selected.has_value();
            for (auto [arm, region] : llvm::enumerate(op->getRegions())) {
                const bool inactive = selected && ((arm == 0) != *selected);
                if (inactive) {
                    if (output.expandedPrunedArms != UINT64_MAX) { ++output.expandedPrunedArms; }
                    continue;
                }
                auto body = context;
                body.guards.push_back({branch, arm == 0, selected.has_value()});
                for (auto& block : region) {
                    for (auto& child : block) { walk(&child, body, depth + 1); }
                }
            }
            return;
        }
        if (op == output.context.function.getOperation()) {
            for (auto& child : output.context.function.front()) { walk(&child, context, depth); }
            return;
        }
        if (op->getNumRegions()) { issue = RecognitionIssue::UnsupportedControl; admitted = false; return; }
        auto phases = builder.index.phasesFor(op);
        const bool multiplePhases = phases.size() > 1;
        if (multiplePhases) { issue = RecognitionIssue::UnsupportedControl; admitted = false; return; }
        const bool payload = phases.size() == 1;
        if (payload) {
            const bool payloadLimit = sites.size() >= limits.payloads;
            if (payloadLimit) { admitted = false; return; }
            context.phase = phases.front();
            sites.push_back(std::move(context));
        }
    };
    if (output.context.roots.empty()) { walk(output.context.root, {}, 0); }
    else {
        for (auto* selected : output.context.roots) { walk(selected, {}, 0); }
    }
    const auto count = static_cast<uint64_t>(sites.size());
    const bool pairsFit = !count || count <= limits.pairs / count;
    if (!admitted || !pairsFit) {
        output.extraction.note(issue, output.context.root);
        return;
    }
    // Only bounded scalar folding precedes the complete control/payload and
    // quadratic-pair preflight. Relation/access construction follows it.
    output.expandedVisits = visits.size();
    output.sites = std::move(sites);
    for (const auto& [op, context, fixedBranch] : visits) {
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
    const PhaseIndex& index, const SyncInput& input)
{
    FiniteGuardedAnalysis result;
    auto program = std::make_shared<ArithmeticProgram>(
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
    const FiniteExpansionLimits limits;
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
    auto translated = detail::normalizeFiniteDemandAccesses(*program, &program->expandedTranslationFragments);
    auto stage = analyzeGeneralArithmeticGenerators(translated ? *translated : *program, &protection);
    if (!stage.analysis().error.empty()) { result.error = stage.analysis().error; return result; }
    auto state = std::make_shared<FiniteGuardedState>();
    state->function = program->context.function;
    state->arena = std::make_shared<RegionExpressions>();
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
    state->cost.expressionNodes = state->arena->size();
    const bool failedReduction = !state->rankIndex.error.empty() || !state->arena->error().empty();
    if (failedReduction) {
        result.error = state->rankIndex.error.empty() ? state->arena->error() : state->rankIndex.error;
        return result;
    }
    result.state = std::move(state);
    result.expandedProgram = std::move(program);
    result.cost = result.state->cost;
    return result;
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
