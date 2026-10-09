// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Derive domains and strict order from the original structured loop tree.
// Native primitives denote its strict native closure, not adjacent edges.
#include "ArithmeticProgramInternal.h"
#include "../InsertSync/SyncRegionArithmetic.h"
#include "RecognitionInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
namespace {
SmallVector<scf::ForOp> enclosing(Operation* op, Operation* root)
{
    SmallVector<scf::ForOp> loops;
    for (auto* parent = op->getParentOp(); parent && op != root && parent != root->getParentOp();
         parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            loops.push_back(loop);
        }
    }
    std::reverse(loops.begin(), loops.end());
    return loops;
}
SmallVector<ArithmeticGuard> enclosingGuards(Operation* op, Operation* root)
{
    SmallVector<ArithmeticGuard> guards;
    auto* child = op;
    for (auto* parent = op->getParentOp(); parent && op != root && parent != root->getParentOp();
         child = parent, parent = parent->getParentOp()) {
        if (auto branch = dyn_cast<scf::IfOp>(parent)) {
            guards.push_back({branch, child->getParentRegion() == &branch.getThenRegion()});
        }
    }
    std::reverse(guards.begin(), guards.end());
    return guards;
}
bool supportedDomain(scf::ForOp loop, detail::ProgramBuilder& builder)
{
    APInt step;
    if (!matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isSignedIntN(64) ||
        step.getSExtValue() <= 0) {
        builder.output.extraction.note(RecognitionIssue::LoopDomain, loop, true);
        return false;
    }
    // The relation retains original IVs. Congruence is represented by the
    // exact local-quotient importer, not by expanding the configured period.
    ArithmeticSite context{nullptr, enclosing(loop, builder.output.context.root), {}};
    if (!builder.prepareBound(loop.getLowerBound(), context) ||
        !builder.prepareBound(loop.getUpperBound(), context)) {
        builder.output.extraction.note(RecognitionIssue::LoopDomain, loop);
        return false;
    }
    return true;
}
void collect(const PhaseIndex& index, detail::ProgramBuilder& builder)
{
    auto& output = builder.output;
    auto* root = output.context.root;
    root->walk<WalkOrder::PreOrder>([&](Operation* op) {
        if (op == output.context.function.getOperation()) {
            return;
        }
        auto phases = index.phasesFor(op);
        for (const auto& prerequisite : index.prerequisitesFor(op)) {
            if (!root->isAncestor(prerequisite.producer->elementOp)) {
                output.incomingPrerequisites.push_back(prerequisite);
            }
        }
        const bool controlPrerequisite = op->getNumRegions() && index.needsValuePrerequisite(op);
        if (controlPrerequisite) {
            output.extraction.note(RecognitionIssue::AdditionalPrerequisite, op);
        }
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
            supportedDomain(loop, builder);
            auto bodyLoops = enclosing(loop, builder.output.context.root);
            bodyLoops.push_back(loop);
            // Retain the original recurrence. The shared scalar semantics must
            // prove every carried argument as a function of these coordinates;
            // unrecognized state never becomes a free execution parameter.
            for (Value argument : loop.getRegionIterArgs()) {
                if (index.isRelevant(argument) && !builder.prepareValue(argument, {nullptr, bodyLoops, {}})) {
                    output.extraction.note(RecognitionIssue::LoopCarriedState, op);
                }
            }
            return;
        }
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
            ArithmeticSite context{nullptr, enclosing(op, root), {}};
            builder.prepareGuard(branch.getCondition(), context);
            return;
        }
        if (op->getNumRegions()) {
            output.extraction.note(RecognitionIssue::UnsupportedControl, op, true);
            return;
        }
        detail::inspectLeaf(*op, index, output.extraction);
        // The shared leaf contract already distinguishes metadata from payloads
        // and unknown effects. Any metadata used by a domain or footprint must
        // additionally pass that consumer's exact scalar/geometry extraction.
        if (phases.size() == 1) {
            output.sites.push_back({phases.front(), enclosing(op, root), enclosingGuards(op, root)});
        }
    });
}
void occurrence(detail::ProgramBuilder& builder, std::size_t id)
{
    const auto& site = builder.output.sites[id];
    auto relation = builder.relation(PrimitiveKind::Occurrences, site.loops.size());
    relation.sourceSite = id;
    relation.sourceDimensions = site.loops.size();
    builder.emitForSites(relation, builder.domain(site, 0), {{&site, 0}});
    builder.output.primitives.relations.push_back(std::move(relation));
}
void order(detail::ProgramBuilder& builder, std::size_t a, std::size_t b)
{
    const auto& first = builder.output.sites[a];
    const auto& second = builder.output.sites[b];
    const unsigned left = first.loops.size(), right = second.loops.size();
    auto relation = builder.relation(PrimitiveKind::Order, left + right);
    relation.sourceSite = a;
    relation.targetSite = b;
    relation.sourceDimensions = left;
    relation.targetDimensions = right;
    auto rows = builder.domain(first, 0);
    llvm::append_range(rows, builder.domain(second, left));
    unsigned common = 0;
    while (common < std::min(left, right) && first.loops[common] == second.loops[common]) {
        auto x = getAffineDimExpr(common, builder.context);
        auto y = getAffineDimExpr(left + common, builder.context);
        auto earlier = rows;
        earlier.push_back(y - x - 1);
        builder.emitForSites(relation, earlier, {{&first, 0}, {&second, left}});
        rows.push_back(x - y);
        rows.push_back(y - x);
        ++common;
    }
    if (a < b) {
        builder.emitForSites(relation, rows, {{&first, 0}, {&second, left}});
    }
    builder.output.primitives.relations.push_back(relation);
    if (first.phase->kPipeValue != second.phase->kPipeValue) {
        return;
    }
    relation.kind = PrimitiveKind::Native;
    for (auto events : {std::make_pair(ArithmeticEvent::Start, ArithmeticEvent::Start),
                        std::make_pair(ArithmeticEvent::Completion, ArithmeticEvent::Completion),
                        std::make_pair(ArithmeticEvent::Start, ArithmeticEvent::Completion)}) {
        relation.sourceEvent = events.first;
        relation.targetEvent = events.second;
        builder.output.primitives.relations.push_back(relation);
    }
    if (a == b) {
        // I_a -> C_a also for the same occurrence, using both endpoint tuples.
        relation.pieces.clear();
        rows = builder.domain(first, 0);
        for (unsigned i = 0; i < left; ++i) {
            auto difference = getAffineDimExpr(i, builder.context) - getAffineDimExpr(left + i, builder.context);
            rows.push_back(difference);
            rows.push_back(-difference);
        }
        builder.emitForSites(relation, rows, {{&first, 0}, {&second, left}});
        builder.output.primitives.relations.push_back(std::move(relation));
    }
}
void prerequisites(detail::ProgramBuilder& builder, const PhaseIndex& index)
{
    for (std::size_t b = 0; b < builder.output.sites.size(); ++b) {
        const auto& target = builder.output.sites[b];
        for (const auto& edge : index.prerequisitesFor(target.phase->elementOp)) {
            for (std::size_t a = 0; a < b; ++a) {
                const auto& source = builder.output.sites[a];
                if (source.phase != edge.producer) { continue; }
                bool sameVisit = true;
                for (auto x : source.fixedCoordinates) {
                    for (auto y : target.fixedCoordinates) {
                        if (x.loop == y.loop && x.induction != y.induction) { sameVisit = false; }
                    }
                }
                if (!sameVisit) { continue; }
                const unsigned left = source.loops.size(), right = target.loops.size();
                auto relation = builder.relation(edge.native ? PrimitiveKind::Native : PrimitiveKind::Prerequisites,
                                                  left + right);
                relation.sourceSite = a; relation.targetSite = b;
                relation.sourceDimensions = left; relation.targetDimensions = right;
                relation.sourceEvent = ArithmeticEvent::Completion;
                relation.targetEvent = ArithmeticEvent::Start;
                auto rows = builder.domain(source, 0);
                llvm::append_range(rows, builder.domain(target, left));
                for (unsigned i = 0; i < std::min(left, right) && source.loops[i] == target.loops[i]; ++i) {
                    auto equal = getAffineDimExpr(i, builder.context) - getAffineDimExpr(left+i, builder.context);
                    rows.push_back(equal); rows.push_back(-equal);
                }
                builder.emitForSites(relation, rows, {{&source, 0}, {&target, left}});
                builder.output.primitives.relations.push_back(std::move(relation));
            }
        }
    }
}
void clearExports(ArithmeticProgram& output)
{
    output.primitives = {};
    output.sites.clear();
    output.parameters.clear();
    output.uniformConflicts.clear();
    output.incomingPrerequisites.clear();
    output.extraction.dischargedEffects.clear();
}
} // namespace
static ArithmeticProgram buildArithmeticProgram(ArithmeticRegionContext region, const PhaseIndex& index,
                                             const SyncInput& input, const SyncStorageEffects& effects,
                                             const ArithmeticLimits& limits, ArithmeticEntryConstant entryConstant,
                                             const FiniteExpansionLimits* expansion)
{
    ArithmeticProgram output;
    output.context = region;
    output.finiteExpansion = expansion != nullptr;
    output.expansionFragmentLimit = expansion ? expansion->fragments : 0;
    output.modeledInput = &input;
    output.phaseIndex = &index;
    output.specializedEntry = static_cast<bool>(entryConstant);
    auto function = region.function;
    output.recognition.state = RecognitionState::MissingPremise;
    if (!function || !region.root ||
        (region.root != function.getOperation() && !function->isAncestor(region.root)) ||
        function.isDeclaration() || !function.getBody().hasOneBlock()) {
        output.extraction.note(RecognitionIssue::UnsupportedControl, function, true);
        return output;
    }
    if (!region.roots.empty()) {
        auto* previous = region.roots.front();
        bool valid = expansion && previous == region.root && previous->getBlock();
        if (valid) {
            for (auto* selected : llvm::drop_begin(region.roots)) {
                const bool adjacent = selected && previous->getNextNode() == selected;
                if (!adjacent) { valid = false; break; }
                previous = selected;
            }
        }
        if (!valid) {
            output.extraction.note(RecognitionIssue::UnsupportedControl, region.root);
            return output;
        }
    }
    if (!limits.pipes || !limits.coefficient || !limits.period) {
        output.extraction.note(RecognitionIssue::ArithmeticConfiguration, function);
        return output;
    }
    // P and D are declared class parameters, not universal restrictions.
    // The residue producer checks representability of its actual tuple count.
    if (limits.period > static_cast<uint64_t>(INT64_MAX)) {
        output.extraction.note(RecognitionIssue::ArithmeticConfiguration, function);
        return output;
    }
    auto configured = limits;
    detail::ProgramBuilder builder{output, configured, function.getContext(), index, DenseMap<Value, unsigned>(),
                                   std::move(entryConstant)};
    if (expansion) { detail::collectExpanded(builder, *expansion); }
    else { collect(index, builder); }
    if (output.extraction.state != RecognitionState::Applicable) {
        clearExports(output);
        return output;
    }
    // An independent read cannot generate a requirement under any regional
    // re-entry. Apply the shared whole-input proof to every regional root,
    // including conditional/explicit phase siblings. Keep the effect identity
    // as a deferred export; consumers need not invent a symbolic byte domain
    // solely for an access that has no possible writer in the supplied input.
    if (region.root != function.getOperation()) {
        for (const auto& site : output.sites) {
            for (auto id : effects.effectsFor(site.phase)) {
                if (detail::dischargeGlobalReadOnlyEffect(id, effects)) {
                    output.extraction.dischargedEffects.push_back(id);
                }
            }
        }
    }
    // Reuse the same whole-input GM discharge as rotating extraction. Only
    // a single visit of the root loop is covered here: nested writer loops
    // require a separate repeated-visit certificate. Retain these effect IDs
    // so the regional adapter can export them for an enclosing owner.
    if (auto loop = dyn_cast<scf::ForOp>(region.root)) {
        for (const auto& site : output.sites) {
            if (site.loops.size() != 1 || site.loops.front() != loop) { continue; }
            for (auto id : effects.effectsFor(site.phase)) {
                if (!llvm::is_contained(output.extraction.dischargedEffects, id) &&
                    detail::dischargeGlobalEffect(id, loop, input, index)) {
                    output.extraction.dischargedEffects.push_back(id);
                }
            }
        }
    }
    // Register symbolic footprint inputs before fixing each relation's shared
    // parameter tuple. Geometry remains Step 0's authoritative access map.
    for (const auto& site : output.sites) {
        if (builder.staticallyEmpty(site)) {
            continue;
        }
        for (auto id : effects.effectsFor(site.phase)) {
            if (llvm::is_contained(output.extraction.dischargedEffects, id)) { continue; }
            for (const auto& region : effects.effects()[id].regions) {
                for (auto symbol : region.symbols) {
                    if (!builder.prepareValue(symbol, site)) {
                        output.extraction.note(RecognitionIssue::IndexArithmetic, site.phase->elementOp);
                    }
                }
            }
        }
    }
    if (output.extraction.state != RecognitionState::Applicable) {
        clearExports(output);
        return output;
    }
    bool preferParity = false;
    if (expansion) {
        bool symbolicParity = false;
        for (const auto& site : output.sites) {
            if (builder.staticallyEmpty(site)) { continue; }
            for (auto id : effects.effectsFor(site.phase)) {
                if (llvm::is_contained(output.extraction.dischargedEffects, id)) { continue; }
                for (const auto& region : effects.effects()[id].regions) {
                    SmallVector<AffineExpr> symbols;
                    for (auto symbol : region.symbols) { symbols.push_back(builder.value(symbol, site, 0)); }
                    SmallVector<AffineExpr> dimensions;
                    for (unsigned i = 0; i < region.extents.size(); ++i) {
                        dimensions.push_back(getAffineDimExpr(i, function.getContext()));
                    }
                    auto specialized = mlir::pto::detail::substitute(region.byteOffset, dimensions, symbols);
                    if (!specialized) { continue; }
                    specialized.walk([&](AffineExpr expression) {
                        auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
                        const bool modulo = binary && binary.getKind() == AffineExprKind::Mod;
                        if (!modulo) { return; }
                        auto divisor = dyn_cast<AffineConstantExpr>(binary.getRHS());
                        const bool parity = divisor && divisor.getValue() == 2;
                        if (!parity) { return; }
                        for (unsigned i = 0; i < output.parameters.size(); ++i) {
                            symbolicParity |= binary.getLHS().isFunctionOfSymbol(i);
                        }
                    });
                }
            }
        }
        // Resolve a retained parity bank in the declared two-residue adapter.
        // It changes representation only; original SSA parameter bindings and
        // all residue cases are preserved. No geometry or class is excluded.
        preferParity = symbolicParity;
    }
    output.primitives.period = configured.period;
    DenseSet<unsigned> pipes;
    for (const auto& site : output.sites) {
        pipes.insert(static_cast<unsigned>(site.phase->kPipeValue));
    }
    output.primitives.pipeCount = pipes.size();
    std::size_t depth = 0;
    for (const auto& site : output.sites) {
        depth = std::max(depth, site.loops.size());
    }
    // Both endpoint tuples and all parameters count, including accesses' byte.
    if (output.parameters.size() > limits.dimensions || depth > limits.dimensions ||
        std::max(2 * depth, depth + 1) + output.parameters.size() > limits.dimensions) {
        output.extraction.note(RecognitionIssue::ArithmeticDimension, function, true);
    }
    if (pipes.size() > limits.pipes) {
        output.extraction.note(RecognitionIssue::ArithmeticPipeLimit, function, true);
    }
    if (output.extraction.state != RecognitionState::Applicable) {
        clearExports(output);
        return output;
    }
    auto construct = [&]() {
        output.primitives.period = configured.period;
        output.primitives.relations.clear();
        output.uniformConflicts.clear();
        output.expandedFragments = 0;
        // Explicit empty roles differ from absent primitive information.
        for (auto kind : {PrimitiveKind::Context, PrimitiveKind::Occurrences, PrimitiveKind::Order,
                          PrimitiveKind::Native, PrimitiveKind::Reads, PrimitiveKind::Writes,
                          PrimitiveKind::Prerequisites}) {
            auto relation = builder.relation(kind, 0);
            if (kind == PrimitiveKind::Context) {
                builder.emit(relation, {});
            }
            output.primitives.relations.push_back(std::move(relation));
        }
        for (std::size_t a = 0; a < output.sites.size(); ++a) {
            occurrence(builder, a);
            for (std::size_t b = 0; b < output.sites.size(); ++b) {
                order(builder, a, b);
            }
        }
        prerequisites(builder, index);
        detail::extractAccesses(builder, input, effects);
        if (output.extraction.state != RecognitionState::Applicable) {
            return false;
        }
        output.recognition = recognizeArithmetic(output.primitives, configured);
        return output.recognition.state == RecognitionState::Applicable;
    };
    const bool constructed = construct();
    if (!constructed) { clearExports(output); return output; }
    if (preferParity) {
        uint64_t estimate = 0;
        bool fits = true;
        for (const auto& relation : output.primitives.relations) {
            if (relation.pieces.empty()) { continue; }
            uint64_t copies = 1;
            for (std::size_t i = 0; i < relation.coordinates.size(); ++i) {
                if (copies > expansion->fragments / 2) { fits = false; break; }
                copies *= 2;
            }
            const bool remaining = fits &&
                relation.pieces.size() <= (expansion->fragments - estimate) / copies;
            if (!remaining) { fits = false; break; }
            estimate += relation.pieces.size() * copies;
        }
        if (fits) {
            auto original = std::move(output.primitives);
            auto recognized = std::move(output.recognition);
            auto conflicts = std::move(output.uniformConflicts);
            const auto extraction = output.extraction;
            const auto fragments = output.expandedFragments;
            output.primitives.parameters = original.parameters;
            output.primitives.pipeCount = original.pipeCount;
            configured.period = 2;
            const bool success = construct();
            const auto attempted = output.expandedFragments;
            if (!success) {
                output.primitives = std::move(original);
                output.recognition = std::move(recognized);
                output.uniformConflicts = std::move(conflicts);
                output.extraction = extraction;
            }
            // Both adapter normalizations are charged. The original occurrence
            // expansion and shared parameter table were constructed only once.
            output.expandedFragments = fragments + attempted;
        }
    }
    return output;
}
ArithmeticProgram recognizeArithmeticProgram(ArithmeticRegionContext region, const PhaseIndex& index,
    const SyncInput& input, const SyncStorageEffects& effects, const ArithmeticLimits& limits,
    ArithmeticEntryConstant entryConstant)
{
    return buildArithmeticProgram(region, index, input, effects, limits, std::move(entryConstant), nullptr);
}
ArithmeticProgram expandFiniteArithmeticProgram(ArithmeticRegionContext region, const PhaseIndex& index,
    const SyncInput& input, const SyncStorageEffects& effects, const FiniteExpansionLimits& limits)
{
    // Only normalize the specialized affine representation here. These are
    // representability bounds, not a restricted-arithmetic membership claim.
    const ArithmeticLimits adapter{UINT_MAX, UINT_MAX, 1, UINT64_MAX};
    return buildArithmeticProgram(region, index, input, effects, adapter, {}, &limits);
}
ArithmeticProgram recognizeArithmeticProgram(func::FuncOp function, const PhaseIndex& index,
                                             const SyncInput& input, const SyncStorageEffects& effects,
                                             const ArithmeticLimits& limits)
{
    return recognizeArithmeticProgram({function, function.getOperation()}, index, input, effects, limits);
}
} // namespace mlir::pto::frontiersynch
