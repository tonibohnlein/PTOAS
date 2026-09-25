// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
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
static void expectCut(const fs::OriginalCoveringBoundary& result, const fs::ProgramAnalysis& analysis, OriginalCut cut)
{
    if (result.status != fs::CoveringBoundary::Status::Covering) {
        llvm::errs() << "covering obstruction: " << result.reason << '\n';
    }
    check(result.status == fs::CoveringBoundary::Status::Covering && result.cut && *result.cut == cut,
          "native prescribed boundary");
    check(bool(fs::resolveOriginalCut(analysis.structure(), cut)), "native cut resolves to unchanged original IR");
    check(result.referenceCoverage && result.exactlyOneVisitPerInterval && result.guardIsTrue,
          "reference coverage and explicit per-interval occurrence count");
    check(result.endpointQualification.available(), "unconditional endpoint needs no unavailable last-use guard");
}
static void run(mlir::func::FuncOp function)
{
    mlir::pto::SyncInput input;
    check(mlir::succeeded(input.build(function)), "shared SyncInput import");
    fs::OriginalStructure original;
    check(mlir::succeeded(fs::importOriginalStructure(function, input, original)), "native original structure import");
    std::size_t choice = fs::NoControlId, counted = fs::NoControlId, unqualified = fs::NoControlId;
    std::vector<std::size_t> uses, loads;
    for (std::size_t i = 0; i < original.originalSites.size(); ++i) {
        auto* op = original.originalSites[i];
        if (mlir::isa<mlir::scf::IfOp>(op) && !op->getParentOfType<mlir::scf::ForOp>()) {
            choice = i;
        }
        if (mlir::isa<mlir::scf::ForOp>(op)) {
            counted = i;
        }
        if (mlir::isa<mlir::scf::WhileOp>(op)) {
            unqualified = i;
        }
    }
    for (std::size_t i = 0; i < original.operations.size(); ++i) {
        auto* op = original.operations[i].instruction->elementOp;
        if (op->getName().getStringRef() == "pto.taxpy" && !op->getParentOfType<mlir::scf::ForOp>() &&
            !op->getParentOfType<mlir::scf::WhileOp>()) {
            uses.push_back(i);
        }
        if (op->getName().getStringRef() == "pto.tload") {
            loads.push_back(i);
        }
    }
    check(uses.size() == 3 && loads.size() == 3 && choice != fs::NoControlId && counted != fs::NoControlId &&
          unqualified != fs::NoControlId, "native covering fixture population");
    const auto A = uses[0], B = uses[1], U = uses[2], overwrite = loads.back();
    std::size_t cell = fs::NoControlId;
    for (const auto& a : original.operations[A].accesses) {
        if (!a.read) {
            continue;
        }
        const bool written = llvm::any_of(original.operations[overwrite].accesses, [&](const fs::Access& w) {
            return w.write && w.cell == a.cell;
        });
        const bool unrelated = llvm::none_of(original.operations[U].accesses, [&](const fs::Access& u) {
            return (u.read || u.write) && u.cell == a.cell;
        });
        if (written && unrelated) {
            cell = a.cell;
        }
    }
    check(cell != fs::NoControlId, "shared native effects prove that U does not touch the original input cell");
    // Supply one relation-tagged incidence and one same-cell incidence whose
    // selector relation is unknown. This tests the adapter's conservative
    // witness filtering; it does not claim importer discovery of a new selector.
    const auto selectedRelation = original.physicalAddresses.size();
    original.physicalAddresses.emplace_back();
    for (auto& access : original.operations[A].accesses) {
        if (access.cell == cell && access.read) {
            access.physicalRelation = selectedRelation;
            original.physicalAddresses.back().memory = access.memory;
        }
    }
    for (auto& access : original.operations[B].accesses) {
        if (access.cell == cell && access.read) { access.physicalRelation = fs::NoControlId; }
    }
    const auto pipe = original.operations[A].instruction->kPipeValue;
    fs::ProgramAnalysis analysis(input, std::move(original));
    check(analysis.complete(), "native Phase A preparation");
    const auto& fixed = analysis.structure();
    const auto g = mlir::cast<mlir::scf::IfOp>(fixed.originalSites[choice]).getCondition();
    const auto observation = analysis.originalValues().qualify(g, analysis.originalValues().phaseCut(A, true));
    check(observation.status == fs::OriginalValueQualification::Status::NotObservableHere,
          "later original SSA guard is unavailable immediately after A");
    auto prepare = [&](fs::OriginalIntervalRequest q) {
        const auto r = analysis.prepareInterval(std::move(q));
        check(r.valid, "native interval preparation");
        return r.interval;
    };
    fs::OriginalIntervalRequest q;
    q.selector.cell = cell;
    q.selector.read = true;
    q.selector.write = false;
    q.selector.engine = unsigned(pipe);
    q.occurrence.stopVisit = fs::OriginalOccurrenceContext::StopVisit::FirstReach;
    q.start = {A, OriginalCut::Before};
    q.stop = {overwrite, OriginalCut::Before};
    const auto interval = prepare(q);
    const auto pairs = analysis.lifetimes().stats().requirements;
    const auto firstBefore = analysis.firstMayUse(interval);
    const auto lastBefore = analysis.lastMayUse(interval);
    const auto source = analysis.sourceCovering(interval);
    const auto target = analysis.targetCovering(interval);
    expectCut(source, analysis, OriginalCut::scope(choice, OriginalCut::After));
    expectCut(target, analysis, {A, OriginalCut::Before});
    check(source.mayAccesses == std::vector<std::size_t>({A, B}), "both real native reader incidences retained");
    check(source.trimmedWork.items.size() == 1 && source.trimmedWork.items.front().operation == U,
          "real unrelated suffix excluded before original overwrite");
    const auto point = fs::resolveOriginalCut(fixed, *source.cut);
    check(point && point->before == fixed.operations[U].instruction->elementOp,
          "covering publication is before U, not at the containing region exit");
    check(analysis.lifetimes().stats().requirements == pairs,
          "covering does not eagerly materialize legacy source-target pairs or duplicate obligations");
    check(analysis.firstMayUse(interval).cuts == firstBefore.cuts && analysis.lastMayUse(interval).cuts == lastBefore.cuts,
          "covering leaves independent existing frontier answers intact");
    const auto hits = analysis.coveringQueries().stats().cacheHits;
    analysis.sourceCovering(interval);
    check(analysis.coveringQueries().stats().cacheHits == hits + 1, "native full-interval directional cache");
    auto relationQuery = q;
    relationQuery.selector.physicalRelation = selectedRelation;
    const auto conservative = analysis.sourceCovering(prepare(relationQuery));
    expectCut(conservative, analysis, OriginalCut::scope(choice, OriginalCut::After));
    check(conservative.conservativeEffects &&
          llvm::is_contained(conservative.mayAccesses, B),
          "same-cell unknown relation cannot be trimmed as a no-hit suffix");
    q.start = OriginalCut::scope(choice, OriginalCut::Before);
    const auto optional = prepare(q);
    expectCut(analysis.sourceCovering(optional), analysis, OriginalCut::scope(choice, OriginalCut::After));
    expectCut(analysis.targetCovering(optional), analysis, OriginalCut::scope(choice, OriginalCut::Before));
    check(analysis.sourceCovering(optional).mayExecuteWithoutAccess && analysis.targetCovering(optional).mayExecuteWithoutAccess,
          "optional empty arm executes its covering boundaries and retains that extra execution");
    q.start = {U, OriginalCut::Before};
    const auto empty = analysis.sourceCovering(prepare(q));
    check(empty.status == fs::CoveringBoundary::Status::NoHit && !empty.cut, "native all-no-hit has no invented endpoint");
    q.start = OriginalCut::scope(counted, OriginalCut::Before);
    q.stop = OriginalCut::scope(counted, OriginalCut::After);
    const auto loop = prepare(q);
    const auto release = analysis.sourceCovering(loop);
    expectCut(release, analysis, q.stop);
    expectCut(analysis.targetCovering(loop), analysis, q.start);
    check(release.mayExecuteWithoutAccess && !release.mayRepeatInInvocation && release.controlQualifications.size() == 1 &&
          release.controlQualifications.front().owner == counted &&
          release.controlQualifications.front().qualification.executableAfterPrerequisites(),
          "finite loop uses shared width-correct arithmetic; variable reader participation stays supported");
    q.start = OriginalCut::scope(unqualified, OriginalCut::Before);
    q.stop = OriginalCut::scope(unqualified, OriginalCut::After);
    const auto unknown = analysis.sourceCovering(prepare(q));
    check(unknown.status == fs::CoveringBoundary::Status::Unknown &&
          unknown.obstruction == fs::CoveringBoundary::Obstruction::RepetitionUnqualified &&
          !unknown.mayAccesses.empty() && !unknown.cut && !unknown.controlQualifications.empty(),
          "unqualified while retains accesses and precise missing finite-exit premise");
    auto stale = interval;
    ++stale.query.version.revision;
    check(analysis.sourceCovering(stale).status == fs::CoveringBoundary::Status::Unknown, "native stale snapshot rejected");
}
int main(int argc, char** argv)
{
    if (argc != 2) {
        llvm::errs() << "usage: pto-frontier-covering-test input.pto\n";
        return 1;
    }
    mlir::DialectRegistry dialects;
    dialects.insert<mlir::pto::PTODialect, mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                    mlir::cf::ControlFlowDialect>();
    mlir::MLIRContext context(dialects);
    context.disableMultithreading();
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    check(module && mlir::succeeded(mlir::verify(*module)), "verified native covering fixture");
    const auto before = print(module->getOperation());
    unsigned functions = 0;
    for (auto function : module->getOps<mlir::func::FuncOp>()) {
        run(function);
        ++functions;
    }
    check(functions == 1, "one native covering fixture");
    check(before == print(module->getOperation()), "covering queries preserve original IR");
    llvm::outs() << "native directional covering passed; original-ir-unchanged; construction-not-run\n";
}
