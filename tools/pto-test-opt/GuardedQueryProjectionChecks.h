// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_GUARDEDQUERYPROJECTIONCHECKS_H
#define PTO_GUARDEDQUERYPROJECTIONCHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/GuardedQueryProjection.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/PrefixExpression.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "RegionAlternativeChecks.h"
namespace guardedQueryProjectionChecks {
namespace fs = mlir::pto::frontiersynch;
using Status = fs::detail::GuardedQueryProjectionStatus;
using Kind = fs::PeriodicEventKind;
inline bool dependencies(mlir::MLIRContext& context)
{
    context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @projection(%q: i1, %n: index, %f: f32) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %i = %zero to %n step %one {
      %p = arith.cmpf ogt, %f, %f : f32
    }
    return
  }
})mlir", &context);
    if (!module) { return false; }
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    auto loop = *function.getOps<mlir::scf::ForOp>().begin();
    auto predicate = *loop.getBody()->getOps<mlir::arith::CmpFOp>().begin();
    mlir::Block symbols;
    fs::RegionExpressions e;
    const auto q = e.input(function.getArgument(0));
    const auto p = e.evaluatedInput(symbols.addArgument(mlir::IntegerType::get(&context, 1), function.getLoc()),
        predicate.getResult(), loop, e.constant(0));
    const auto yes = e.boolean(true);
    const auto n = e.input(function.getArgument(1));
    const auto numericPresence = e.lt(n, e.constant(10));
    // The last B completion covers its earlier completion. The numeric
    // reducer discards the older guarded incoming edge; a raw Boolean OR can
    // still mention that edge's opaque guard despite equivalent truth values.
    const std::vector<fs::GuardedRankPayload> alternatives{{0, yes}, {1, yes}, {1, yes}, {2, yes}};
    const auto ranks = fs::reduceGuardedRanks(e, alternatives,
        {{0, 1, q}, {0, 2, numericPresence}, {1, 3, p}, {2, 3, yes}}, {},
        fs::GuardedRankLabels::Reference, true);
    const auto original = ranks.query(e, {0, Kind::Completion}, {3, Kind::Start});
    if (!original || llvm::any_of(e.referencedInputs(*original),
            [&](const auto& input) { return input.first == p; })) { return false; }
    const auto before = e.size();
    const auto refused = fs::detail::projectGuardedRankQuery(e, ranks, {0, Kind::Completion}, {3, Kind::Start});
    if (refused.status != Status::AdditionalInput || refused.additionalInput != p ||
        refused.answer != original || e.size() != before || !e.constructionError().empty()) { return false; }
    for (unsigned values = 0; values < 8; ++values) {
        fs::RegionExpressions::Transaction scratch(e);
        fs::RegionExpressions::Substitution bindings({{q, e.boolean(values & 1)},
            {n, e.constant(values & 2 ? 0 : 20)}, {p, e.boolean(values & 4)}});
        const auto answer = e.constantValue(e.substitute(*original, bindings));
        if (!answer || bool(*answer) != bool(values & 3)) { return false; }
    }
    // The original root is available after the loop, while the added opaque
    // evaluation is not. Refusal must retain exact original cut availability.
    mlir::Block scratch;
    mlir::OpBuilder builder(&context); builder.setInsertionPointToEnd(&scratch);
    fs::RegionExpressions::CutEmission emission;
    if (mlir::failed(e.emitContextual(*refused.answer, builder, function.front().getTerminator(), emission))) {
        return false;
    }
    const std::vector<fs::GuardedRankPayload> payloads{{0, q}, {0, yes}, {1, yes}};
    const auto necessary = fs::reduceGuardedRanks(e, payloads, {{0, 2, p}}, {}, fs::GuardedRankLabels::Reference, true);
    const auto accepted = fs::detail::projectGuardedRankQuery(e, necessary, {0, Kind::Completion}, {2, Kind::Start});
    if ((accepted.status != Status::Projected && accepted.status != Status::NoReduction) ||
        !accepted.answer) { return false; }
    const auto inputs = e.referencedInputs(*accepted.answer);
    if (llvm::none_of(inputs, [&](const auto& input) { return input.first == p; }) ||
        !e.evaluation(p) || e.evaluation(p)->original != predicate.getResult()) { return false; }
    fs::RegionExpressions::CutEmission unavailable;
    if (mlir::succeeded(e.emitContextual(*accepted.answer, builder, function.front().getTerminator(), unavailable))) {
        return false;
    }
    const auto ordinary = fs::reduceGuardedRanks(e, payloads, {}, {}, fs::GuardedRankLabels::Reference);
    if (ordinary.queryGraph || fs::detail::projectGuardedRankQuery(e, ordinary,
            {0, Kind::Start}, {0, Kind::Start}).status != Status::MissingGraph) { return false; }
    auto foreign = necessary;
    auto foreignGraph = std::make_shared<fs::detail::GuardedQueryGraph>(*foreign.queryGraph);
    fs::RegionExpressions other;
    foreignGraph->arena = &other; foreign.queryGraph = std::move(foreignGraph);
    if (fs::detail::projectGuardedRankQuery(e, foreign,
            {0, Kind::Completion}, {2, Kind::Start}).status != Status::ForeignArena) { return false; }
    // Conservative totality qualification inspects BOTH arms, even when the
    // numeric row discards this entire redundant edge. No witness admission is
    // broadened by an equal set of referenced original inputs.
    const auto witness = e.integerWitness({{fs::BoundInteger(1)}, fs::BoundInteger(0)},
        fs::BoundInteger(2), {n}, 1, {0}, 0);
    const auto guard = e.select(numericPresence, e.lt(witness, e.constant(3)), e.boolean(false));
    const std::vector<fs::GuardedRankPayload> numericPayloads{{0, numericPresence}, {0, yes}, {1, yes}};
    const auto partial = fs::reduceGuardedRanks(e, numericPayloads, {{0, 2, guard}, {1, 2, yes}}, {},
        fs::GuardedRankLabels::Reference, true);
    const auto fallback = fs::detail::projectGuardedRankQuery(e, partial, {0, Kind::Completion}, {2, Kind::Start});
    return fallback.status == Status::PartialGuard && fallback.answer == numericPresence &&
           e.referencedInputs(guard) == e.referencedInputs(numericPresence) && e.constructionError().empty();
}
inline bool scalingAndRollback(mlir::MLIRContext& context)
{
    for (unsigned count : {16U, 64U, 256U}) {
        mlir::Block symbols;
        fs::RegionExpressions e;
        std::vector<fs::GuardedRankPayload> payloads;
        std::vector<fs::GuardedRankEdge> edges;
        for (unsigned i = 0; i < count; ++i) {
            const auto present = e.input(symbols.addArgument(mlir::IntegerType::get(&context, 1),
                mlir::UnknownLoc::get(&context)));
            payloads.push_back({i % 2, present});
            if (i) { edges.push_back({i - 1, i, e.boolean(true)}); }
        }
        const auto ranks = fs::reduceGuardedRanks(e, payloads, edges, {}, fs::GuardedRankLabels::Reference, true);
        const auto old = ranks.query(e, {0, Kind::Completion}, {count - 1, Kind::Start});
        if (!old || !ranks.queryGraph || !ranks.queryGraph->totalGuards) { return false; }
        const auto before = e.size();
        {
            fs::RegionExpressions::Transaction outer(e);
            const auto projected = fs::detail::projectGuardedRankQuery(e, ranks,
                {0, Kind::Completion}, {count - 1, Kind::Start});
            if (projected.status != Status::Projected || !projected.answer ||
                projected.occurrences != count || projected.edges != edges.size() ||
                projected.projectedNodes >= projected.originalNodes) { return false; }
            const auto stable = e.size();
            const auto repeated = fs::detail::projectGuardedRankQuery(e, ranks,
                {0, Kind::Completion}, {count - 1, Kind::Start});
            if (repeated.answer != projected.answer || e.size() != stable) { return false; }
            fs::detail::GuardedQueryProjectionWork work;
            work.record(projected); work.record(repeated);
            if (work.outcomes[Status::Projected] != 2 || work.occurrences != 2 * count ||
                work.edges != 2 * edges.size() || work.constructedNodes != projected.constructedNodes ||
                work.dependencyNodes != 2 * (projected.originalNodes + projected.projectedNodes) ||
                !ranks.queryGraph->qualificationNodes) { return false; }
        }
        if (e.size() != before) { return false; }
        // Reuse the rolled-back suffix with unrelated nodes. The graph owns
        // only original guards; no cached projected answer may survive.
        for (unsigned i = 0; i < 4; ++i) {
            e.select(e.input(symbols.getArgument(0)), e.constant(1000 + i), e.constant(2000 + i));
        }
        const auto retried = fs::detail::projectGuardedRankQuery(e, ranks,
            {0, Kind::Completion}, {count - 1, Kind::Start});
        if (retried.status != Status::Projected || !retried.answer || *retried.answer == e.invalid) { return false; }
        llvm::outs() << "Boolean guarded query " << count << ": " << retried.originalNodes << " -> "
                     << retried.projectedNodes << " nodes\n";
    }
    return true;
}
inline bool historicalAvailability(mlir::MLIRContext& context)
{
    for (bool unavailable : {false, true}) {
        auto reject = [&](const char* stage) {
            llvm::errs() << "Boolean query history " << unavailable << " failed at " << stage << "\n";
            return false;
        };
        auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @history(%n: index, %x: f32, %y: f32) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %i = %zero to %n step %one {
      %predicate = arith.cmpf ogt, %x, %y : f32
    }
    return
  }
})mlir", &context);
        if (!module) { return reject("fixture parse"); }
        auto function = *module->getOps<mlir::func::FuncOp>().begin();
        auto loop = *function.getOps<mlir::scf::ForOp>().begin();
        auto predicate = *loop.getBody()->getOps<mlir::arith::CmpFOp>().begin();
        auto arena = std::make_shared<fs::RegionExpressions>();
        auto symbols = std::make_shared<mlir::Block>(); arena->retainInputOwner(symbols);
        auto& e = *arena;
        std::vector<fs::GuardedRankPayload> payloads, historicalPayloads;
        std::vector<fs::GuardedRankEdge> edges, historicalEdges;
        constexpr unsigned count = 8;
        for (unsigned i = 0; i < count; ++i) {
            const auto present = e.slt(e.constant(i), e.input(function.getArgument(0)));
            const auto visit = e.constant(unavailable && i == 0 ? count + 1 : i);
            const auto guard = e.evaluatedInput(
                symbols->addArgument(mlir::IntegerType::get(&context, 1), function.getLoc()),
                predicate.getResult(), loop, visit);
            payloads.push_back({i % 2, present});
            historicalPayloads.push_back({i % 2, e.land(present, guard)});
            if (i + 1 < count) {
                edges.push_back({i, i + 1, guard});
                historicalEdges.push_back({i, i + 1, e.boolean(true)});
            }
        }
        // Opaque edge guards make this candidate larger. Preserve the exact
        // numeric root and exercise its original historical emission below.
        const auto residualRanks = fs::reduceGuardedRanks(
            e, payloads, edges, {}, fs::GuardedRankLabels::Reference, true);
        const auto residual = residualRanks.query(e, {0, Kind::Completion}, {count - 1, Kind::Start});
        const auto beforeResidual = e.size();
        const auto unchanged = fs::detail::projectGuardedRankQuery(
            e, residualRanks, {0, Kind::Completion}, {count - 1, Kind::Start});
        if (!residual || unchanged.status != Status::NoReduction || unchanged.answer != residual ||
            unchanged.projectedNodes < unchanged.originalNodes || e.size() != beforeResidual ||
            fs::detail::retainGuardedRankAlternative(
                e, residualRanks, {0, Kind::Completion}, {count - 1, Kind::Start}) != residual ||
            fs::detail::RegionConstructionAccess::alternative(e, *residual) || e.size() != beforeResidual) {
            return reject("unchanged larger candidate");
        }
        // Conditional occurrences with unconditional consecutive edges expose
        // the smaller Boolean prefix, while retaining per-visit opaque inputs.
        const auto ranks = fs::reduceGuardedRanks(
            e, historicalPayloads, historicalEdges, {}, fs::GuardedRankLabels::Reference, true);
        const auto old = ranks.query(e, {0, Kind::Completion}, {count - 1, Kind::Start});
        const auto projected = fs::detail::projectGuardedRankQuery(e, ranks,
            {0, Kind::Completion}, {count - 1, Kind::Start});
        if (!old || !projected.answer || projected.status != Status::Projected) {
            llvm::errs() << "history query status=" << static_cast<unsigned>(projected.status)
                         << " old=" << bool(old) << " answer=" << bool(projected.answer)
                         << " nodes=" << projected.originalNodes << "/" << projected.projectedNodes << "\n";
            return reject("candidate qualification");
        }
        unsigned route = 0;
        for (const auto root : {*residual, *old, *projected.answer, *old}) {
            if (route == 3) {
                if (fs::detail::retainGuardedRankAlternative(e, ranks,
                        {0, Kind::Completion}, {count - 1, Kind::Start}) != old) {
                    return reject("authoritative query registration");
                }
                fs::RegionExpressions imported;
                auto copied = imported.import(e, {*old});
                if (mlir::failed(copied) ||
                    !fs::detail::RegionConstructionAccess::alternative(imported, copied->front())) {
                    return reject("imported alternate ownership");
                }
                for (auto input : imported.referencedInputs(copied->front())) {
                    const auto evaluation = imported.evaluation(input.first);
                    if (!evaluation) { continue; }
                    const auto originalId = e.input(input.second);
                    const auto originalEvaluation = e.evaluation(originalId);
                    if (!originalEvaluation || evaluation->original != originalEvaluation->original ||
                        evaluation->loop != originalEvaluation->loop ||
                        imported.constantValue(evaluation->ordinal) !=
                            e.constantValue(originalEvaluation->ordinal)) {
                        return reject("imported evaluation identity");
                    }
                }
            }
            fs::InvocationPrefixCollection collection(function);
            auto* cut = function.front().getTerminator();
            auto emit = fs::preparePrefixExpression(arena, root, cut, collection, "query-projection-test");
            collection.recipe().actions.push_back({cut, fs::PrivateStateCut::Before,
                [emit](mlir::OpBuilder& builder, const mlir::IRMapping& mapping,
                    llvm::MutableArrayRef<mlir::Value> state, std::string& error) {
                    return mlir::success(mlir::succeeded(emit(builder, mapping, state, error)));
                }});
            collection.recipe().privateScalarControl = true;
            fs::PrivateStateError error;
            const auto prepared = fs::materializeInvocationPrefix(function, collection.recipe(), error);
            if (mlir::succeeded(prepared) == unavailable) {
                llvm::errs() << "Boolean query history " << unavailable << " root=" << root
                             << ": " << error.message << "\n";
                return false;
            }
            if (route == 3) {
                const auto report = collection.recipe().expressionProfile->report();
                const auto* producer = report.getObject("alternate_producer_construction");
                if (!producer || producer->getInteger("graphs").value_or(0) == 0 ||
                    producer->getInteger("queries").value_or(0) == 0 ||
                    producer->getInteger("guard_qualification_visits").value_or(0) == 0 ||
                    producer->getInteger("registration_visits").value_or(0) == 0) {
                    if (producer) {
                        llvm::errs() << "history producer graphs=" << producer->getInteger("graphs").value_or(-1)
                            << " queries=" << producer->getInteger("queries").value_or(-1)
                            << " qualification=" << producer->getInteger("guard_qualification_visits").value_or(-1)
                            << " registration=" << producer->getInteger("registration_visits").value_or(-1) << "\n";
                    }
                    return reject("retained producer accounting");
                }
                if (report.getInteger("alternate_attempts").value_or(0) == 0 ||
                    (!unavailable && report.getInteger("alternate_successes").value_or(0) == 0) ||
                    (unavailable && report.getInteger("alternate_fallbacks").value_or(0) == 0)) {
                    llvm::errs() << "history alternate attempts="
                        << report.getInteger("alternate_attempts").value_or(-1)
                        << " successes=" << report.getInteger("alternate_successes").value_or(-1)
                        << " fallbacks=" << report.getInteger("alternate_fallbacks").value_or(-1) << "\n";
                    return reject("emission accounting");
                }
            }
            ++route;
        }
    }
    return true;
}
inline bool run(mlir::MLIRContext& context)
{
    if (!regionAlternativeChecks::run(context)) { return false; }
    if (!dependencies(context)) { llvm::errs() << "guarded query dependency qualification failed\n"; return false; }
    if (!scalingAndRollback(context)) { llvm::errs() << "guarded query scaling/rollback failed\n"; return false; }
    if (!historicalAvailability(context)) {
        llvm::errs() << "guarded query history qualification failed\n"; return false;
    }
    return true;
}
} // namespace guardedQueryProjectionChecks
#endif
