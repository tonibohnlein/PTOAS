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
SmallVector<scf::ForOp> enclosing(Operation* op)
{
    SmallVector<scf::ForOp> loops;
    for (auto* parent = op->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            loops.push_back(loop);
        }
    }
    std::reverse(loops.begin(), loops.end());
    return loops;
}
bool supportedBound(scf::ForOp loop, detail::ProgramBuilder& builder, func::FuncOp function)
{
    auto bound = loop.getUpperBound();
    APInt number;
    if (matchPattern(bound, m_ConstantInt(&number))) {
        return number.isSignedIntN(64);
    }
    if (auto argument = dyn_cast<BlockArgument>(bound);
        argument && argument.getOwner() == &function.front() && isa<IndexType>(argument.getType())) {
        if (builder.parameterIds.try_emplace(bound, builder.output.parameters.size()).second) {
            builder.output.parameters.push_back(bound);
            builder.output.primitives.parameters.push_back("p" + std::to_string(argument.getArgNumber()));
        }
        return true;
    }
    return llvm::any_of(enclosing(loop), [&](scf::ForOp parent) { return parent.getInductionVar() == bound; });
}
void collect(func::FuncOp function, const PhaseIndex& index, detail::ProgramBuilder& builder)
{
    auto& output = builder.output;
    // Mark values transitively derived from a payload result. Such SSA uses
    // may impose extra completion prerequisites absent from the storage model.
    DenseSet<Value> payloadValues;
    function.walk<WalkOrder::PreOrder>([&](Operation* op) {
        if (op == function.getOperation()) {
            return;
        }
        auto phases = index.phasesFor(op);
        const bool derived = llvm::any_of(op->getOperands(), [&](Value value) {
            return payloadValues.contains(value);
        });
        if (derived && (!phases.empty() || op->getNumRegions() || isa<func::ReturnOp, scf::YieldOp>(op))) {
            output.extraction.note(RecognitionIssue::AdditionalPrerequisite, op);
        }
        if (derived || !phases.empty()) {
            payloadValues.insert(op->getResults().begin(), op->getResults().end());
        }
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
            detail::checkRotatingDomain(loop, output.extraction);
            if (!supportedBound(loop, builder, function)) {
                output.extraction.note(RecognitionIssue::LoopDomain, op, true);
            }
            return;
        }
        if (op->getNumRegions()) {
            output.extraction.note(RecognitionIssue::UnsupportedControl, op, true);
            return;
        }
        detail::inspectLeaf(*op, index, output.extraction);
        // Only these structural/value operations have no separate prerequisites.
        if (phases.empty() && !isa<AllocTileOp, AllocMultiTileOp, MultiTileGetOp, SubViewOp,
                                   scf::YieldOp, func::ReturnOp>(op) &&
            op->getName().getDialectNamespace() != "arith") {
            output.extraction.note(RecognitionIssue::UnmodeledOperation, op);
        }
        if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp, BarrierOp>(op)) {
            output.extraction.note(RecognitionIssue::AdditionalPrerequisite, op);
        }
        if (phases.size() == 1) {
            output.sites.push_back({phases.front(), enclosing(op)});
        }
    });
}
void occurrence(detail::ProgramBuilder& builder, std::size_t id)
{
    const auto& site = builder.output.sites[id];
    auto relation = builder.relation(PrimitiveKind::Occurrences, site.loops.size());
    relation.sourceSite = id;
    relation.sourceDimensions = site.loops.size();
    builder.emit(relation, builder.domain(site, 0));
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
        builder.emit(relation, earlier);
        rows.push_back(x - y);
        rows.push_back(y - x);
        ++common;
    }
    if (a < b) {
        builder.emit(relation, rows);
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
        builder.emit(relation, rows);
        builder.output.primitives.relations.push_back(std::move(relation));
    }
}
void clearExports(ArithmeticProgram& output)
{
    output.primitives = {};
    output.sites.clear();
    output.parameters.clear();
}
} // namespace
ArithmeticProgram recognizeArithmeticProgram(func::FuncOp function, const PhaseIndex& index,
                                             const SyncInput& input, const SyncStorageEffects& effects,
                                             const ArithmeticLimits& limits)
{
    ArithmeticProgram output;
    output.recognition.state = RecognitionState::MissingPremise;
    if (function.isDeclaration() || !function.getBody().hasOneBlock()) {
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
    detail::ProgramBuilder builder{output, limits, function.getContext(), DenseMap<Value, unsigned>()};
    collect(function, index, builder);
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
} // namespace mlir::pto::frontiersynch
