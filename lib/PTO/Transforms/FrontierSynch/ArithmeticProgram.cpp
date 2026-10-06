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
    APInt lower, step;
    const bool constants = matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
                           matchPattern(loop.getStep(), m_ConstantInt(&step));
    const bool positive = constants && lower.isSignedIntN(64) && step.isSignedIntN(64) &&
                          lower.getSExtValue() >= 0 && step.getSExtValue() > 0;
    const bool supported = positive && builder.limits.period % step.getSExtValue() == 0;
    if (!supported) {
        return false;
    }
    return builder.prepareValue(loop.getUpperBound(), {nullptr, enclosing(loop, builder.output.context.root), {}});
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
            if (!supportedDomain(loop, builder)) {
                output.extraction.note(RecognitionIssue::LoopDomain, op, true);
            }
            auto bodyLoops = enclosing(loop, builder.output.context.root);
            bodyLoops.push_back(loop);
            // Retain the original recurrence. The shared scalar semantics must
            // prove every carried argument as a function of these coordinates;
            // unrecognized state never becomes a free execution parameter.
            for (Value argument : loop.getRegionIterArgs()) {
                if (!builder.prepareValue(argument, {nullptr, bodyLoops, {}})) {
                    output.extraction.note(RecognitionIssue::LoopCarriedState, op, true);
                }
            }
            return;
        }
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
            ArithmeticSite context{nullptr, enclosing(op, root), {}};
            if (!builder.prepareGuard(branch.getCondition(), context)) {
                output.extraction.note(RecognitionIssue::UnsupportedControl, op, true);
            }
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
}
} // namespace
ArithmeticProgram recognizeArithmeticProgram(ArithmeticRegionContext region, const PhaseIndex& index,
                                             const SyncInput& input, const SyncStorageEffects& effects,
                                             const ArithmeticLimits& limits)
{
    ArithmeticProgram output;
    output.context = region;
    auto function = region.function;
    output.recognition.state = RecognitionState::MissingPremise;
    if (!function || !region.root ||
        (region.root != function.getOperation() && !function->isAncestor(region.root)) ||
        function.isDeclaration() || !function.getBody().hasOneBlock()) {
        output.extraction.note(RecognitionIssue::UnsupportedControl, function, true);
        return output;
    }
    if (!limits.pipes || !limits.coefficient) {
        output.extraction.note(RecognitionIssue::ArithmeticConfiguration, function, true);
        return output;
    }
    if (!limits.period || limits.period > 2) {
        output.extraction.note(RecognitionIssue::ArithmeticPeriod, function, true);
        return output;
    }
    // This first producer has a deliberately bounded residue language: P<=2,
    // D<=8 (at most 256 residue tuples per conjunction). The supplied-bundle
    // checker is independent of these producer limits.
    if (limits.dimensions > 8) {
        output.extraction.note(RecognitionIssue::ArithmeticDimension, function, true);
        return output;
    }
    detail::ProgramBuilder builder{output, limits, function.getContext(), index, DenseMap<Value, unsigned>()};
    collect(index, builder);
    if (output.extraction.state != RecognitionState::Applicable) {
        clearExports(output);
        return output;
    }
    // Register symbolic footprint inputs before fixing each relation's shared
    // parameter tuple. Geometry remains Step 0's authoritative access map.
    for (const auto& site : output.sites) {
        if (builder.staticallyEmpty(site)) {
            continue;
        }
        for (auto id : effects.effectsFor(site.phase)) {
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
    output.primitives.period = limits.period;
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
        clearExports(output);
        return output;
    }
    output.recognition = recognizeArithmetic(output.primitives, limits);
    if (output.recognition.state != RecognitionState::Applicable) {
        clearExports(output);
    }
    return output;
}
ArithmeticProgram recognizeArithmeticProgram(func::FuncOp function, const PhaseIndex& index,
                                             const SyncInput& input, const SyncStorageEffects& effects,
                                             const ArithmeticLimits& limits)
{
    return recognizeArithmeticProgram({function, function.getOperation()}, index, input, effects, limits);
}
} // namespace mlir::pto::frontiersynch
