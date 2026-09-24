// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
// Native public-interface regression. This is intentionally separate from the
// standard-library-only core test: it needs this checkout's generated PTO/MLIR
// headers and tests real imported instruction effects and insertion positions.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

namespace fs = mlir::pto::frontiersynch;
using fs::OriginalCut;
static void check(bool condition, const char* message)
{
    if (!condition) {
        llvm::errs() << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
static std::string print(mlir::Operation* op)
{
    std::string text;
    llvm::raw_string_ostream out(text);
    op->print(out);
    return text;
}
static bool has(const fs::OriginalMayAfter& answer, std::size_t op)
{
    for (const auto& origin : answer.witnesses) {
        if (origin.operation == op) {
            return true;
        }
    }
    return false;
}
static void run(mlir::func::FuncOp function)
{
    mlir::pto::SyncInput input;
    check(mlir::succeeded(input.build(function)), "shared SyncInput import");
    fs::OriginalStructure original;
    check(mlir::succeeded(fs::importOriginalStructure(function, input, original)), "original structure import");
    std::size_t outer = fs::NoControlId, inner = fs::NoControlId, choice = fs::NoControlId;
    for (std::size_t site = 0; site < original.originalSites.size(); ++site) {
        auto* op = original.originalSites[site];
        if (mlir::isa<mlir::scf::IfOp>(op)) {
            choice = site;
        }
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
            if (loop->getParentOfType<mlir::scf::ForOp>()) {
                inner = site;
            } else {
                outer = site;
            }
        }
    }
    check(outer != fs::NoControlId && inner != fs::NoControlId && choice != fs::NoControlId, "fixture controls");
    std::vector<std::size_t> updates;
    std::size_t overwrite = fs::NoControlId;
    for (std::size_t i = 0; i < original.operations.size(); ++i) {
        auto* op = original.operations[i].instruction->elementOp;
        const auto name = op->getName().getStringRef();
        if (name == "pto.taxpy") {
            updates.push_back(i);
        }
        if (name == "pto.tload" && op->getParentOfType<mlir::scf::ForOp>()) {
            overwrite = i;
        }
    }
    check(updates.size() == 2 && overwrite != fs::NoControlId, "fixture payloads");
    std::size_t cell = fs::NoControlId;
    for (const auto& a : original.operations[updates[0]].accesses) {
        for (const auto& b : original.operations[overwrite].accesses) {
            if (a.read && b.write && a.cell == b.cell) {
                cell = a.cell;
            }
        }
    }
    check(cell != fs::NoControlId, "real physical overlap imported through shared effects");
    const auto join = OriginalCut::scope(choice, OriginalCut::After);
    const auto joinIR = fs::resolveOriginalCut(original, join);
    check(joinIR && joinIR->before == original.originalSites[choice]->getNextNode(), "real if-join insertion point");
    const auto exitIR = fs::resolveOriginalCut(original, OriginalCut::scope(inner, OriginalCut::After));
    check(exitIR && exitIR->before == original.originalSites[inner]->getNextNode(), "real loop-exit insertion point");
    const auto childExit = fs::resolveOriginalCut(original, OriginalCut::childBoundary(inner, 0, OriginalCut::After));
    check(
        childExit && childExit->before->hasTrait<mlir::OpTrait::IsTerminator>(),
        "child exit precedes original terminator");
    check(
        !fs::resolveOriginalCut(original, OriginalCut::childBoundary(choice, 1, OriginalCut::Before)),
        "absent else region does not acquire a fabricated insertion point");

    // Semantic adapter fixture: two translated phases of the SAME real original
    // operation. This tests the lowering boundary, not recognition of a native
    // multi-phase opcode. No instruction effects are changed in SyncInput.
    fs::OriginalStructure phases;
    phases.function = function;
    phases.originalSites = original.originalSites;
    phases.operations = {original.operations[updates[0]], original.operations[updates[0]]};
    phases.operations[0].beforeExecutable = true;
    phases.operations[0].afterExecutable = false;
    phases.operations[1].beforeExecutable = false;
    phases.operations[1].afterExecutable = true;
    check(
        !fs::resolveOriginalCut(phases, {0, OriginalCut::After}) &&
            !fs::resolveOriginalCut(phases, {1, OriginalCut::Before}),
        "internal phase cuts rejected");
    check(
        bool(fs::resolveOriginalCut(phases, {0, OriginalCut::Before})) &&
            bool(fs::resolveOriginalCut(phases, {1, OriginalCut::After})),
        "outer phase cuts remain executable");

    fs::ProgramAnalysis analysis(input, std::move(original));
    check(analysis.complete(), "native Phase A preparation");
    auto prepare = [&](fs::OriginalIntervalRequest query) {
        const auto result = analysis.prepareInterval(std::move(query));
        check(result.valid, "public interval preparation");
        return result.interval;
    };
    fs::OriginalIntervalRequest q;
    q.selector.cell = cell;
    q.occurrence.source = updates[0];
    q.occurrence.target = updates[1];
    q.occurrence.stopVisit = fs::OriginalOccurrenceContext::StopVisit::FirstReach;
    q.start = {updates[0], OriginalCut::After};
    q.stop = OriginalCut::scope(inner, OriginalCut::After);
    const auto local = prepare(q);
    check(local.owner == inner, "least physical owner is child for child-exit horizon");
    q.continuationOwner = outer;
    const auto parent = prepare(q);
    check(parent.owner == outer && parent != local, "same endpoints with distinct declared continuation");
    const auto beforeHits = analysis.lifetimes().stats().intervalCacheHits;
    analysis.mayAfter(local);
    analysis.mayAfter(local);
    check(analysis.lifetimes().stats().intervalCacheHits == beforeHits + 1, "full-context cache hit");
    analysis.mayAfter(parent);
    check(
        analysis.lifetimes().stats().intervalCacheHits == beforeHits + 1, "different owner/horizon is not a cache hit");
    auto stale = local;
    stale.query.version = fs::OriginalProgramVersion::fresh();
    check(
        analysis.mayAfter(stale).status == fs::OriginalMayAfter::Status::Unknown, "foreign original version rejected");

    q.continuationOwner.reset();
    q.start = OriginalCut::scope(inner, OriginalCut::After);
    q.stop = q.start;
    check(
        analysis.mayAfter(prepare(q)).status == fs::OriginalMayAfter::Status::NoHit,
        "child-exit same-visit interval empty");
    q.stop = {overwrite, OriginalCut::After};
    q.includeStoppingAccess = true;
    q.occurrence.target = overwrite;
    auto wider = prepare(q);
    check(wider.owner == outer && has(analysis.mayAfter(wider), overwrite), "enclosing overwrite is not child exit");
    q.start = {overwrite, OriginalCut::Before};
    q.occurrence.source = overwrite;
    for (const auto side : {OriginalCut::Before, OriginalCut::After}) {
        q.stop = {overwrite, side};
        for (bool include : {false, true}) {
            q.includeStoppingAccess = include;
            const auto interval = prepare(q);
            const auto result = analysis.mayAfter(interval);
            check(has(result, overwrite) == include, "explicit stop inclusion, before and after");
            check(
                result.status == (include ? fs::OriginalMayAfter::Status::May : fs::OriginalMayAfter::Status::NoHit),
                "inclusion changes actual native result, not just its label");
        }
    }
    q.stop = {overwrite, OriginalCut::After};
    q.includeStoppingAccess = true;
    const auto single = prepare(q);
    const auto first = analysis.firstMayUse(single), last = analysis.lastMayUse(single);
    check(
        first.status == fs::PhysicalUseFrontier::Status::Present &&
            first.cuts == std::vector<OriginalCut>{{overwrite, OriginalCut::Before}},
        "native first may-use returns legal cut");
    check(
        last.status == fs::PhysicalUseFrontier::Status::Present &&
            last.cuts == std::vector<OriginalCut>{{overwrite, OriginalCut::After}},
        "native last may-use returns legal cut");
    q.occurrence.incomingInterface = 71;
    const auto incoming = analysis.mayAfter(prepare(q));
    check(incoming.cases.incoming && has(incoming, overwrite), "incoming identity retained without new origins");

    bool decodedPair = false;
    const auto& requirements = analysis.requirementsAt(updates[1]);
    for (std::size_t i = 0; i < requirements.size(); ++i) {
        if (requirements[i].relationship.source.operation != updates[0] || requirements[i].relationship.cell != cell) {
            continue;
        }
        auto request = analysis.intervalFor(updates[1], i);
        request.continuationOwner = inner;
        const auto& a = analysis.interpretAt(updates[1], i, request);
        request.continuationOwner = outer;
        const auto& b = analysis.interpretAt(updates[1], i, request);
        check(
            &a != &b && a.queryInterval && b.queryInterval && a.queryInterval->owner == inner &&
                b.queryInterval->owner == outer,
            "construction-facing decoder and interpreter key the full continuation");
        check(
            a.decoded->requirement.relationship.source.operation == updates[0] &&
                b.decoded->requirement.relationship.target.operation == updates[1],
            "different descriptors retain unchanged original requirement endpoints");
        decodedPair = true;
        break;
    }
    check(decodedPair, "real imported requirement exercises context-keyed decoding");
}
int main(int argc, char** argv)
{
    if (argc != 2) {
        llvm::errs() << "usage: pto-frontier-interval-test input.pto\n";
        return 1;
    }
    mlir::DialectRegistry dialects;
    dialects.insert<
        mlir::pto::PTODialect, mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
        mlir::cf::ControlFlowDialect>();
    mlir::MLIRContext context(dialects);
    context.disableMultithreading();
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    check(module && mlir::succeeded(mlir::verify(*module)), "verified native fixture");
    const auto before = print(module->getOperation());
    unsigned functions = 0;
    for (auto function : module->getOps<mlir::func::FuncOp>()) {
        run(function);
        ++functions;
    }
    check(functions == 1, "one native interval fixture");
    check(before == print(module->getOperation()), "Phase A preserved original IR");
    llvm::outs() << "native original intervals passed; original-ir-unchanged; construction-not-run\n";
    return 0;
}
