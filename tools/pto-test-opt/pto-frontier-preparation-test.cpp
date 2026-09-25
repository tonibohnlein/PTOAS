// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
// Native semantic fixtures exercise the public immutable preparation service.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include <cstdlib>
#include <iostream>
using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::frontiersynch;
static void check(bool b, int line = __builtin_LINE())
{
    if (!b) { std::cerr << "preparation assertion failed at " << line << '\n'; std::exit(1); }
}
static frontiersynch::Region phase(std::size_t id)
{
    frontiersynch::Region r; r.kind = frontiersynch::Region::Operation; r.operation = id; return r;
}
static OriginalStructure fixture(func::FuncOp function,
    std::vector<std::unique_ptr<CompoundInstanceElement>>& instructions, bool loop, bool internal)
{
    OriginalStructure p; p.function = function; p.cells.resize(3); p.operations.resize(3); p.originalSites.resize(4);
    function.walk([&](mlir::Operation* op) {
        if (isa<scf::ForOp, scf::IfOp>(op)) { p.originalSites[3] = op; }
        if (auto loc = dyn_cast<NameLoc>(op->getLoc())) {
            for (unsigned i = 0; i < 3; ++i) {
                if (loc.getName().getValue() == "p" + std::to_string(i)) { p.originalSites[i] = op; }
            }
        }
    });
    if (!p.originalSites[3]) { p.originalSites.resize(3); }
    for (unsigned i = 0; i < 3; ++i) {
        check(p.originalSites[i]);
        instructions.push_back(std::make_unique<CompoundInstanceElement>(i,
            SmallVector<const BaseMemInfo*>{}, SmallVector<const BaseMemInfo*>{},
            i == 1 ? PipelineType::PIPE_V : PipelineType::PIPE_MTE2, p.originalSites[i]->getName()));
        instructions.back()->elementOp = p.originalSites[i];
        auto& op = p.operations[i]; op.instruction = instructions.back().get(); op.original = i;
        op.beforeExecutable = op.afterExecutable = true; op.enclosingAfter = i;
    }
    p.operations[0].accesses = {{0,false,true,true}, {1,false,true,true}};
    p.operations[1].accesses = {{1,true,false,false}, {2,false,true,true}};
    p.operations[2].accesses = {{0,true,false,false}, {2,true,false,false}};
    if (internal) { p.operations[0].afterExecutable = false; p.operations[0].enclosingAfter = 1; }
    p.body.children = {phase(0), phase(1), phase(2)};
    if (loop) {
        frontiersynch::Region repeat; repeat.kind = frontiersynch::Region::For;
        repeat.originalOwner = 3; repeat.qualifiedCounted = true; repeat.zeroTripPossible = true;
        repeat.children.push_back(std::move(p.body)); p.body = {};
        p.body.children.push_back(std::move(repeat));
    }
    return p;
}
static OriginalObligationId due(const ProgramAnalysis& analysis)
{
    for (auto id : analysis.obligationsAt(2)) {
        const auto* f = analysis.obligations().get(id);
        if (f->key.kind == OriginalObligationKey::Kind::RAW && f->key.cell == 0) { return {id, {0,false}}; }
    }
    check(false); return {};
}
static OriginalInterval interval(const ProgramAnalysis& analysis)
{
    OriginalIntervalRequest q; q.version = analysis.structure().version;
    q.selector.cell = 0; q.selector.read = true; q.selector.write = false;
    q.start = {0,OriginalCut::After}; q.stop = {2,OriginalCut::Before};
    q.occurrence.source = 0; q.occurrence.target = 2;
    q.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::FirstReach;
    auto result = analysis.prepareInterval(q); check(result.valid); return result.interval;
}
int main()
{
    MLIRContext context; context.loadDialect<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    auto module = parseSourceString<ModuleOp>(R"mlir(module {
      func.func @test() {
        %p0 = arith.constant 0 : i32 loc("p0")
        %p1 = arith.constant 1 : i32 loc("p1")
        %p2 = arith.constant 2 : i32 loc("p2")
        return
      }
      func.func @conditional(%g: i1) {
        %p0 = arith.constant 0 : i32 loc("p0")
        scf.if %g { %p1 = arith.constant 1 : i32 loc("p1") }
        %p2 = arith.constant 2 : i32 loc("p2")
        return
      }
      func.func @branches(%g: i1) {
        scf.if %g {
          %p0 = arith.constant 0 : i32 loc("p0")
        } else {
          %p1 = arith.constant 1 : i32 loc("p1")
        }
        %p2 = arith.constant 2 : i32 loc("p2")
        return
      }
      func.func @loop(%n: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %p0 = arith.constant 0 : i32 loc("p0")
          %p1 = arith.constant 1 : i32 loc("p1")
          %p2 = arith.constant 2 : i32 loc("p2")
        }
        return
      }
    })mlir", &context);
    check(module && succeeded(verify(*module)));
    SyncInput input;
    std::vector<std::unique_ptr<CompoundInstanceElement>> instructions;
    ProgramAnalysis analysis(input, fixture(module->lookupSymbol<func::FuncOp>("test"), instructions, false, false));
    check(analysis.complete()); const auto& prep = *analysis.preparation();
    check(prep.formed()); check(prep.repertoireComplete()); check(prep.hooks().size() >= 2);
    check(analysis.obligations().stats().enumeratedOrigins == 0);
    check(analysis.lifetimes().stats().requirements == 0);
    check(!prep.bucketsAt(0).empty()); check(!prep.atDeadline(2).empty());
    const auto query = interval(analysis); const auto obligation = due(analysis);
    const auto& result = prep.consequences(obligation, query);
    check(result.obstructions.empty()); check(result.opportunities.size() == 1);
    check(result.opportunities[0].middle == 1);
    check(result.opportunities[0].due == obligation);
    check(result.predicates.evaluate(result.opportunities[0].condition, {}) == true);
    const auto computations = prep.stats().consequenceComputations;
    check(&prep.consequences(obligation, query) == &result);
    check(prep.stats().consequenceComputations == computations);
    auto changed = query; changed.query.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::AfterBackedge;
    check(prep.consequences(obligation, changed).opportunities.empty());
    check(prep.stats().consequenceComputations == computations + 1);
    auto filtered = query; filtered.query.selector.engine = unsigned(PipelineType::PIPE_V);
    check(prep.consequences(obligation, filtered).opportunities.empty());
    auto includingDeadline = query; includingDeadline.query.includeStoppingAccess = true;
    check(prep.consequences(obligation, includingDeadline).opportunities.empty());
    // Simulate traversal: all named cuts already exist before reaching source 0.
    const auto count = prep.hooks().size();
    for (unsigned phaseId = 0; phaseId < 3; ++phaseId) {
        check(prep.hooks().size() == count);
        if (phaseId < 2) {
            bool present = false;
            for (const auto& hook : prep.hooks()) { present |= hook.sufficient == OriginalCut{phaseId,OriginalCut::After}; }
            check(present);
        }
    }
    std::vector<std::unique_ptr<CompoundInstanceElement>> conditionalInstructions;
    auto conditionalOriginal = fixture(module->lookupSymbol<func::FuncOp>("conditional"), conditionalInstructions, false, false);
    frontiersynch::Region optional; optional.kind = frontiersynch::Region::Choice; optional.originalOwner = 3;
    optional.children = {phase(1), frontiersynch::Region{}};
    conditionalOriginal.body.children = {phase(0), optional, phase(2)};
    ProgramAnalysis conditional(input, std::move(conditionalOriginal)); check(conditional.complete());
    const auto& conditionalResult = conditional.preparation()->consequences(due(conditional), interval(conditional));
    check(conditionalResult.opportunities.size() == 1);
    const auto condition = conditionalResult.opportunities[0].condition;
    check(conditionalResult.predicates.evaluate(condition, [](std::size_t) { return true; }) == true);
    check(conditionalResult.predicates.evaluate(condition, [](std::size_t) { return false; }) == false);
    std::vector<std::unique_ptr<CompoundInstanceElement>> branchInstructions;
    auto branchOriginal = fixture(module->lookupSymbol<func::FuncOp>("branches"), branchInstructions, false, false);
    frontiersynch::Region choice; choice.kind = frontiersynch::Region::Choice; choice.originalOwner = 3;
    choice.children = {phase(0), phase(1)}; branchOriginal.body.children = {choice, phase(2)};
    ProgramAnalysis branches(input, std::move(branchOriginal)); check(branches.complete());
    const auto& branchResult = branches.preparation()->consequences(due(branches), interval(branches));
    check(branchResult.opportunities.empty());
    std::vector<std::unique_ptr<CompoundInstanceElement>> repeatedInstructions;
    ProgramAnalysis repeated(input, fixture(module->lookupSymbol<func::FuncOp>("loop"), repeatedInstructions, true, false));
    check(repeated.complete());
    const auto& repeatedResult = repeated.preparation()->consequences(due(repeated), interval(repeated));
    check(repeatedResult.opportunities.empty()); check(!repeatedResult.obstructions.empty());
    std::vector<std::unique_ptr<CompoundInstanceElement>> internalInstructions;
    ProgramAnalysis internal(input, fixture(module->lookupSymbol<func::FuncOp>("test"), internalInstructions, false, true));
    check(internal.complete()); check(!internal.preparation()->repertoireComplete());
    bool analytical = false;
    for (const auto& hook : internal.preparation()->hooks()) {
        if (hook.sufficient == OriginalCut{0,OriginalCut::After}) {
            analytical = true; check(hook.executable == OriginalCut{1,OriginalCut::After}); check(!hook.obstruction.empty());
        }
    }
    check(analytical);
    // Finite declared cycle: shifts remain relations, no unfolding per visit.
    bool declared = false;
    OriginalRequestBoundaryProvider provider = [&](const ProgramAnalysis& a, const OriginalRequestReference&) {
        OriginalRequestAnswers answer;
        if (declared) { return answer; } declared = true;
        auto roleUniverse = a.structure().version;
        DescriptorFactRef x{DescriptorFactRef::Kind::SupportRole, 0, 0, roleUniverse};
        DescriptorFactRef y{DescriptorFactRef::Kind::SupportRole, 0, 1, roleUniverse};
        OriginalSupportRole first, second;
        first.id = x; second.id = y;
        first.version = second.version = a.structure().version;
        first.interval = second.interval = interval(a);
        first.source = {0,OriginalCut::After}; second.source = {1,OriginalCut::After};
        first.qualified = second.qualified = true;
        first.links.push_back({y, {}, 1, {}}); second.links.push_back({x, {}, -1, {}});
        DescriptorFactRef missing{DescriptorFactRef::Kind::SupportRole, 0, 2, roleUniverse};
        second.links.push_back({missing, {}, 0, {}});
        answer.declaredRoles = {first,second}; answer.support.push_back({x, {}, 0, {}});
        return answer;
    };
    std::vector<std::unique_ptr<CompoundInstanceElement>> supportedInstructions;
    ProgramAnalysis supported(input, fixture(module->lookupSymbol<func::FuncOp>("test"), supportedInstructions, false, false), provider);
    check(supported.complete()); check(supported.preparation()->stats().roles == 2);
    check(supported.preparation()->supportRoles().size() == 2);
    check(supported.preparation()->stats().links < 10);
    bool missingRole = false;
    for (const auto& obstruction : supported.preparation()->obstructions()) {
        missingRole |= obstruction == "support link has no declared role";
    }
    check(missingRole);
    check(!supported.preparation()->repertoireComplete()); // Shift premises retained, not invented.
    check(supported.obligations().stats().enumeratedOrigins == 0);
    ++const_cast<OriginalStructure&>(analysis.structure()).version.revision;
    check(!prep.formed()); check(prep.hooks().empty());
    check(prep.consequences(obligation, query).opportunities.empty());
    std::cout << "PASS: prepared source hooks, finite support closure, scoped cross-cell consequences, stale snapshots\n";
}
