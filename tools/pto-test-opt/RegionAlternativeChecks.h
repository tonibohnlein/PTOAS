// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_REGIONALTERNATIVECHECKS_H
#define PTO_REGIONALTERNATIVECHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RegionExpressionAlternatives.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/PrefixExpression.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/PrefixExpressionProfile.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/PrefixExpressionSession.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include <iterator>
namespace regionAlternativeChecks {
namespace fs = mlir::pto::frontiersynch;
using Access = fs::detail::RegionConstructionAccess;
using Id = fs::RegionExpressions::Id;
struct Pair { Id original, alternate; };
inline Pair equivalent(fs::RegionExpressions& e, Id x, Id q, Id p, bool originalAvailable)
{
    auto left = e.select(p, x, e.add(x, e.constant(1))), right = x;
    for (unsigned i = 1; i <= 8; ++i) {
        left = e.add(left, e.constant(i)); right = e.add(right, e.constant(i));
    }
    // (x+7)==(x-(-7)) is true for every uint64 bit pattern, including wrap.
    // Keeping this unfactored identity distinguishes the existing cut prover.
    const auto identity = e.eq(e.add(x, e.constant(7)), e.sub(x, e.constant(uint64_t(0) - 7)));
    const auto otherQ = e.select(q, identity, e.boolean(false));
    return {e.select(originalAvailable ? q : otherQ, e.boolean(true), e.eq(left, right)),
            e.select(originalAvailable ? otherQ : q, e.boolean(true), p)};
}
inline bool values(fs::RegionExpressions& e, Pair pair, Id x, Id q, Id p)
{
    for (uint64_t value : {uint64_t(0), uint64_t(1), uint64_t(7), uint64_t(INT64_MAX), UINT64_MAX}) {
        for (bool a : {false, true}) { for (bool b : {false, true}) {
            fs::RegionExpressions::Transaction scratch(e);
            fs::RegionExpressions::Substitution bind({{x, e.constant(value)}, {q, e.boolean(a)}, {p, e.boolean(b)}});
            if (e.constantValue(e.substitute(pair.original, bind)) != uint64_t(a || b) ||
                e.constantValue(e.substitute(pair.alternate, bind)) != uint64_t(a || b)) { return false; }
        } }
    }
    return true;
}
inline bool transport(mlir::MLIRContext& context)
{
    mlir::Block symbols;
    fs::RegionExpressions e;
    const auto loc = mlir::UnknownLoc::get(&context);
    const auto x = e.input(symbols.addArgument(mlir::IndexType::get(&context), loc));
    const auto q = e.input(symbols.addArgument(mlir::IntegerType::get(&context, 1), loc));
    const auto p = e.input(symbols.addArgument(mlir::IntegerType::get(&context, 1), loc));
    const auto pair = equivalent(e, x, q, p, true);
    if (!values(e, pair, x, q, p)) { return false; }
    // Journal registrations whose original AND alternate predate a transaction.
    const auto before = e.size();
    {
        fs::RegionExpressions::Transaction outer(e);
        {
            fs::RegionExpressions::Transaction inner(e);
            if (!Access::retainAlternative(e, pair.original, pair.alternate) ||
                Access::alternative(e, pair.original) != pair.alternate) { return false; }
            inner.commit();
        }
    }
    if (e.size() != before || Access::alternative(e, pair.original)) { return false; }
    if (!Access::retainAlternative(e, pair.original, pair.alternate)) { return false; }
    const auto y = e.input(symbols.addArgument(mlir::IndexType::get(&context), loc));
    const auto replacement = e.add(y, e.constant(11));
    fs::RegionExpressions::Substitution binding({{x, replacement}});
    const auto changed = e.substitute(pair.original, binding);
    auto changedAlternative = Access::alternative(e, changed);
    fs::RegionExpressions::Substitution expectedBinding({{x, replacement}});
    const auto expected = e.substitute(pair.alternate, expectedBinding);
    if (!changedAlternative || *changedAlternative != expected) { return false; }
    // An internal root replacement is atomic and must not transport its old
    // relation to a semantically unrelated replacement expression.
    const auto unrelated = e.land(q, p);
    fs::RegionExpressions::Substitution internal({{pair.original, unrelated}});
    if (e.substitute(pair.original, internal) != unrelated || Access::alternative(e, unrelated)) { return false; }
    fs::RegionExpressions imported;
    auto roots = imported.import(e, {changed});
    if (mlir::failed(roots)) { return false; }
    auto importedAlternative = Access::alternative(imported, roots->front());
    auto expectedImported = imported.import(e, {*changedAlternative});
    if (!importedAlternative || mlir::failed(expectedImported) ||
        *importedAlternative != expectedImported->front()) { return false; }
    // An original-only input can acquire an unsupported typed recipe through
    // substitution. Requalify BOTH spellings, not only the reduced alternate.
    const auto dropsX = equivalent(e, x, q, p, false);
    if (!Access::retainAlternative(e, dropsX.original, dropsX.alternate)) { return false; }
    const auto typed = e.integerWitness({{fs::BoundInteger(1)}, fs::BoundInteger(0)},
        fs::BoundInteger(2), {y}, 1, {0}, 0);
    fs::RegionExpressions::Substitution typedBinding({{x, typed}});
    const auto typedOriginal = e.substitute(dropsX.original, typedBinding);
    if (typedOriginal >= e.size() || Access::alternative(e, typedOriginal) ||
        !e.constructionError().empty()) { return false; }
    // Destroying a speculative imported suffix also removes its relation.
    fs::RegionExpressions destination;
    Id removed = fs::RegionExpressions::invalid;
    {
        fs::RegionExpressions::Transaction rollback(destination);
        auto attempt = destination.import(e, {changed});
        if (mlir::failed(attempt) || !Access::alternative(destination, attempt->front())) { return false; }
        removed = attempt->front();
    }
    if (destination.size() || Access::alternative(destination, removed)) { return false; }
    // Imported diagnostic ownership is deduplicated and survives discarded
    // optional work. It contains counts only, never speculative expression IDs.
    fs::PrefixExpressionProfile profile;
    profile.recordAlternativeConstruction(e);
    profile.recordAlternativeConstruction(imported);
    const auto firstReport = profile.report();
    const auto* firstWork = firstReport.getObject("alternate_producer_construction");
    if (!firstWork || firstWork->getInteger("registration_visits").value_or(0) == 0) { return false; }
    const auto attempts = firstWork->getInteger("registration_attempts");
    profile.recordAlternativeConstruction(e);
    profile.recordAlternativeConstruction(imported);
    const auto repeatedReport = profile.report();
    const auto* repeatedWork = repeatedReport.getObject("alternate_producer_construction");
    if (!repeatedWork || repeatedWork->getInteger("registration_attempts") != attempts) { return false; }
    return true;
}
// Exercise the producer's mixed predicate bindings without authorizing general
// internal-node replacement. The authoritative memo exists before registration.
inline bool qualifiedBinding(mlir::MLIRContext& context)
{
    for (unsigned mode = 0; mode != 6; ++mode) {
        auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @bound(%n: index, %f: f32) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %i = %zero to %n step %one {
      %predicate = arith.cmpf oeq, %f, %f : f32
    }
    return
  }
})mlir", &context);
        if (!module) { return false; }
        auto function = *module->getOps<mlir::func::FuncOp>().begin();
        auto loop = *function.getOps<mlir::scf::ForOp>().begin();
        auto predicate = *loop.getBody()->getOps<mlir::arith::CmpFOp>().begin();
        fs::RegionExpressions e;
        mlir::Block symbols;
        const auto loc = function.getLoc();
        const auto x = e.input(symbols.addArgument(mlir::IndexType::get(&context), loc));
        const auto y = e.input(symbols.addArgument(mlir::IndexType::get(&context), loc));
        const auto ordinal = e.add(x, e.constant(1));
        const auto p = e.evaluatedInput(symbols.addArgument(mlir::IntegerType::get(&context, 1), loc),
            predicate.getResult(), loop, ordinal);
        const auto replacement = e.add(y, e.constant(11));
        const auto newOrdinal = e.add(replacement, e.constant(1));
        const auto nextP = e.evaluatedInput(symbols.addArgument(mlir::IntegerType::get(&context, 1), loc),
            predicate.getResult(), loop, newOrdinal);
        const auto q = e.lt(x, e.constant(10));
        const auto newQ = e.lt(replacement, e.constant(10));
        const auto pair = equivalent(e, x, q, p, true);
        llvm::SmallVector<std::pair<Id, Id>> mappings{{x, replacement}, {p, nextP}};
        if (mode == 1 || mode == 2) {
            // This composite is irrelevant to both queries, but every mapping
            // still requires qualification; an unrelated rewrite is not trusted.
            mappings.push_back({e.lt(x, e.constant(99)),
                mode == 1 ? e.lt(replacement, e.constant(99)) : e.boolean(false)});
        } else { mappings.push_back({q, newQ}); }
        if (mode == 3) { mappings.push_back({pair.original, e.boolean(false)}); }
        if (mode == 4) { mappings.push_back({ordinal, newOrdinal}); }
        if (mode == 5) {
            const auto partial = e.integerWitness({{fs::BoundInteger(1)}, fs::BoundInteger(0)},
                fs::BoundInteger(2), {y}, 1, {0}, 0);
            mappings[0].second = partial;
            mappings.back().second = e.lt(partial, e.constant(10));
        }
        fs::RegionExpressions::Substitution mixed(mappings);
        const auto bound = e.substitute(pair.original, mixed);
        if (bound >= e.size() || !Access::retainAlternative(e, pair.original, pair.alternate)) { return false; }
        // Cached numeric query lookup does not perform generic transport.
        if (e.substitute(pair.original, mixed) != bound || Access::alternative(e, bound)) { return false; }
        const auto before = e.size();
        const bool supported = mode != 2 && mode != 3 && mode != 5;
        {
            fs::RegionExpressions::Transaction rollback(e);
            Access::transportQualifiedAlternative(e, pair.original, bound, mixed);
            const auto selected = Access::alternative(e, bound);
            if (bool(selected) != supported || !e.constructionError().empty()) { return false; }
            if (selected) {
                fs::RegionExpressions::Substitution inputOnly({{x, replacement}, {p, nextP}});
                const auto expected = e.substitute(pair.alternate, inputOnly);
                if (*selected != expected) { return false; }
                for (uint64_t value : {uint64_t(0), uint64_t(1), uint64_t(7), UINT64_MAX}) {
                    for (bool flag : {false, true}) {
                        fs::RegionExpressions::Transaction valuation(e);
                        fs::RegionExpressions::Substitution concrete({{y, e.constant(value)},
                            {nextP, e.boolean(flag)}});
                        const uint64_t oracle = (value + uint64_t(11) < 10) || flag;
                        if (e.constantValue(e.substitute(bound, concrete)) != oracle ||
                            e.constantValue(e.substitute(*selected, concrete)) != oracle) { return false; }
                    }
                }
                const auto history = e.evaluation(nextP);
                if (!history || history->original != predicate.getResult() || history->loop != loop ||
                    history->ordinal != newOrdinal) { return false; }
            }
        }
        if (e.size() != before || Access::alternative(e, bound) ||
            e.substitute(pair.original, mixed) != bound) { return false; }
        Access::transportQualifiedAlternative(e, pair.original, bound, mixed);
        if (bool(Access::alternative(e, bound)) != supported || !e.constructionError().empty()) { return false; }
        const auto& work = Access::alternativeConstructionWork(e);
        if (!work.bindingVisits || !work.finalLookups ||
            (supported && !work.outcomes.count("binding-retained"))) { return false; }
    }
    return true;
}
inline bool availability(mlir::MLIRContext& context)
{
    for (bool originalAvailable : {false, true}) {
        auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @alternate(%x: index, %q: i1) {
    %one = arith.constant 1 : index
    scf.if %q {
      scf.yield
    }
    scf.if %q {
      %opaque = arith.cmpi eq, %x, %one : index
    }
    return
  }
})mlir", &context);
        if (!module) { return false; }
        auto function = *module->getOps<mlir::func::FuncOp>().begin();
        auto branches = function.getOps<mlir::scf::IfOp>();
        auto first = *branches.begin(), second = *std::next(branches.begin());
        auto opaque = *second.getThenRegion().front().getOps<mlir::arith::CmpIOp>().begin();
        auto* cut = first.getThenRegion().front().getTerminator();
        auto arena = std::make_shared<fs::RegionExpressions>();
        auto& e = *arena;
        e.forbidRecomputation(opaque);
        const auto x = e.input(function.getArgument(0)), q = e.input(function.getArgument(1));
        const auto p = e.input(opaque.getResult());
        const auto pair = equivalent(e, x, q, p, originalAvailable);
        if (!values(e, pair, x, q, p)) { return false; }
        // Independently establish the deliberate shape-sensitive availability
        // contrast before installing any equivalence record.
        for (auto test : {std::make_pair(pair.original, originalAvailable),
                          std::make_pair(pair.alternate, !originalAvailable)}) {
            mlir::Block scratch;
            mlir::OpBuilder builder(&context); builder.setInsertionPointToEnd(&scratch);
            fs::RegionExpressions::CutEmission emission;
            if (mlir::succeeded(e.emitContextual(test.first, builder, cut, emission)) != test.second) {
                llvm::errs() << "alternate direct availability premise " << originalAvailable << "\n";
                return false;
            }
        }
        if (!Access::retainAlternative(e, pair.original, pair.alternate)) { return false; }
        // Unlike the identity mappings below, a replacement Boolean is an
        // opaque binding. Neither spelling may borrow the original branch fact
        // to erase its unavailable future input in that environment.
        {
            mlir::Block replacements, scratch;
            auto replacement = replacements.addArgument(mlir::IntegerType::get(&context, 1), function.getLoc());
            mlir::IRMapping mapped;
            mapped.map(function.getArgument(0), function.getArgument(0));
            mapped.map(function.getArgument(1), replacement);
            fs::InvocationPrefixCollection opaqueCollection(function);
            auto opaqueEmit = fs::preparePrefixExpression(
                arena, pair.original, cut, opaqueCollection, "alternate-mapped-opaque");
            const auto size = e.size();
            const auto actionCount = opaqueCollection.recipe().actions.size();
            mlir::OpBuilder builder(&context); builder.setInsertionPointToEnd(&scratch);
            std::string error;
            if (mlir::succeeded(opaqueEmit(builder, mapped, {}, error)) || error.empty() || !scratch.empty() ||
                e.size() != size || opaqueCollection.recipe().actions.size() != actionCount) {
                llvm::errs() << "alternate nonidentity binding lost opacity " << originalAvailable << "\n";
                return false;
            }
            const auto report = opaqueCollection.recipe().expressionProfile->report();
            if (report.getInteger("alternate_attempts") != 1 || report.getInteger("alternate_fallbacks") != 1 ||
                report.getInteger("alternate_successes") != 0) { return false; }
        }
        fs::InvocationPrefixCollection collection(function);
        auto emit = fs::preparePrefixExpression(arena, pair.original, cut, collection, "alternate-control");
        const auto actions = collection.recipe().actions.size();
        mlir::IRMapping mapping;
        for (auto argument : function.getArguments()) { mapping.map(argument, argument); }
        const auto before = e.size();
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            mlir::Block scratch;
            mlir::OpBuilder builder(&context); builder.setInsertionPointToEnd(&scratch);
            std::string error;
            auto result = emit(builder, mapping, {}, error);
            llvm::APInt value;
            if (mlir::failed(result) || !mlir::matchPattern(*result, mlir::m_ConstantInt(&value)) ||
                !value.isOne() || collection.recipe().actions.size() != actions || e.size() != before) {
                llvm::errs() << "alternate exact fallback " << originalAvailable << ": " << error << "\n";
                return false;
            }
        }
        const auto report = collection.recipe().expressionProfile->report();
        if (report.getInteger("alternate_attempts") != 2 ||
            report.getInteger("alternate_successes") != (originalAvailable ? 0 : 2) ||
            report.getInteger("alternate_fallbacks") != (originalAvailable ? 2 : 0) ||
            report.getInteger("emissions") != 2) { return false; }
        // A real shared emission scope must not retain failed candidate SSA.
        // Repeat in one block, then change blocks and recreate the scope: each
        // exact cut/environment may reuse its own successful result only.
        collection.recipe().expressionProfile->resetEmission();
        for (unsigned scopeIndex = 0; scopeIndex < 2; ++scopeIndex) {
            mlir::Block outputs[2];
            auto scope = collection.recipe().expressionSession->begin();
            if (!scope) { return false; }
            for (auto& output : outputs) {
                mlir::OpBuilder builder(&context); builder.setInsertionPointToEnd(&output);
                mlir::Value previous;
                for (unsigned repeat = 0; repeat < 2; ++repeat) {
                    std::string error;
                    auto result = emit(builder, mapping, {}, error);
                    llvm::APInt value;
                    if (mlir::failed(result) || !mlir::matchPattern(*result, mlir::m_ConstantInt(&value)) ||
                        !value.isOne() || (previous && previous != *result) ||
                        collection.recipe().actions.size() != actions || e.size() != before) {
                        llvm::errs() << "alternate shared-session isolation " << originalAvailable << "\n";
                        return false;
                    }
                    previous = *result;
                }
                if (output.getOperations().size() != 1) { return false; }
            }
        }
        const auto shared = collection.recipe().expressionProfile->report();
        if (shared.getInteger("shared_returns").value_or(0) < 4 ||
            shared.getInteger("retained_operations") != 4) { return false; }
    }
    return true;
}
inline bool run(mlir::MLIRContext& context)
{
    context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect>();
    if (!transport(context)) { llvm::errs() << "regional alternate transport failed\n"; return false; }
    if (!qualifiedBinding(context)) { llvm::errs() << "qualified alternate binding failed\n"; return false; }
    if (!availability(context)) { llvm::errs() << "regional alternate availability failed\n"; return false; }
    return true;
}
} // namespace regionAlternativeChecks
#endif
