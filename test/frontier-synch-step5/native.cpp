// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/OriginalValueQueries.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>
#include <map>
static void check(bool condition, int line = __builtin_LINE())
{
    if (!condition) {
        std::cerr << "step-5 semantic assertion failed at line " << line << "\n";
        std::exit(1);
    }
}
using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::frontiersynch;
using Status = OriginalValueQualification::Status;

// Actual MLIR scalar/control IR with explicit translated-phase semantic
// fixtures. Marking an arith operation as a phase here is test input, not a
// production opcode recognizer or a claimed real-PTO importer regression.
struct Fixture {
    OwningOpRef<ModuleOp> module;
    OriginalStructure original;
    std::map<std::string, Operation*> named;
    std::vector<std::unique_ptr<CompoundInstanceElement>> phases;
    Fixture(MLIRContext& context, StringRef source)
    {
        module = parseSourceString<ModuleOp>(source, &context);
        check(module && succeeded(verify(*module)));
        original.function = *module->getOps<func::FuncOp>().begin();
        original.function.walk<WalkOrder::PreOrder>([&](Operation* op) {
            if (op != original.function.getOperation()) {
                original.originalSites.push_back(op);
            }
            if (auto loc = dyn_cast<NameLoc>(op->getLoc())) {
                named[loc.getName().getValue().str()] = op;
            }
        });
    }
    Operation* op(const char* name) const { return named.at(name); }
    Value value(const char* name) const { return op(name)->getResult(0); }
    void asyncPhases(const char* name)
    {
        auto* operation = op(name);
        const auto site = std::find(original.originalSites.begin(), original.originalSites.end(), operation) -
                          original.originalSites.begin();
        for (unsigned i = 0; i < 2; ++i) {
            auto phase = std::make_unique<CompoundInstanceElement>(
                i, SmallVector<const BaseMemInfo*>{}, SmallVector<const BaseMemInfo*>{},
                i ? PipelineType::PIPE_V : PipelineType::PIPE_MTE2, operation->getName());
            phase->elementOp = operation;
            original.operations.push_back({phase.get(), std::size_t(site), {}, i == 0, i == 1, 1});
            phases.push_back(std::move(phase));
        }
    }
    std::string text() const
    {
        std::string result;
        llvm::raw_string_ostream stream(result);
        module.get().print(stream);
        stream.flush();
        return result;
    }
};
int main()
{
    MLIRContext context;
    context.loadDialect<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    Fixture f(context, R"mlir(
module {
  func.func @values(%g: i1, %n: index, %x: i32) {
    %zero = arith.constant 0 : index loc("zero")
    %one = arith.constant 1 : index
    %z32 = arith.constant 0 : i32
    %true = arith.constant true
    %async = arith.addi %x, %z32 : i32 loc("async")
    %condition = arith.cmpi sgt, %async, %z32 : i32 loc("condition")
    scf.if %condition { scf.yield } loc("use_async")
    scf.if %g { scf.yield } loc("g1")
    scf.if %g { scf.yield } loc("g2")
    %out = scf.for %i = %zero to %n step %one iter_args(%carried = %g) -> (i1) {
      %vary = arith.cmpi eq, %i, %zero : index loc("vary")
      scf.if %carried { scf.yield } loc("use_carried")
      %has_previous = arith.cmpi sgt, %i, %zero : index
      scf.if %has_previous {
        scf.if %carried { scf.yield } loc("previous_use")
      }
      %next = arith.xori %carried, %true : i1 loc("next")
      scf.yield %next : i1
    } loc("loop")
    %unchanged = scf.for %j = %zero to %n step %one iter_args(%same = %g) -> (i1) {
      scf.if %same { scf.yield } loc("same")
      scf.yield %same : i1
    } loc("unchanged")
    return loc("return")
  }
}
)mlir");
    f.asyncPhases("async");
    const auto unchanged = f.text();
    OriginalValueQueries q(f.original);
    auto loop = cast<scf::ForOp>(f.op("loop"));
    const auto g = f.original.function.getArgument(0);
    check(q.qualify(g, q.before(loop)).available());
    check(q.qualify(f.value("condition"), q.before(f.op("async"))).status == Status::NotObservableHere);
    const auto condition = q.qualify(f.value("condition"), q.before(f.op("use_async")));
    check(condition.status == Status::NeedsCompletion && condition.prerequisites.size() == 2);
    for (const auto& p : condition.prerequisites) {
        check(p.deadline == q.before(f.op("condition")));
        check(p.source == q.after(f.op("async")) && p.executableSourcePhase == 1);
        check(p.applicability.empty());
    }
    check(!q.legal(q.phaseCut(0, true)));
    check(q.legal(q.phaseCut(1, true)));
    check(
        q.qualifyEnabled(f.value("condition"), q.before(f.op("use_async")), {f.value("condition")}).status ==
        Status::Unresolved);
    check(q.qualifyEnabled(f.value("condition"), q.before(f.op("use_async")), {g}).status == Status::NeedsCompletion);
    check(q.qualify(f.value("vary"), q.before(loop)).status == Status::NotObservableHere);
    check(q.qualify(f.value("vary"), q.after(f.op("vary"))).available());
    auto carried = loop.getRegionIterArgs().front();
    check(q.identity(carried).value == carried && q.identity(carried).value != g);
    check(q.qualify(carried, q.before(f.op("use_carried"))).available());
    check(q.qualify(carried, q.before(loop)).status == Status::NotObservableHere);
    check(q.qualify({carried, q.siteOf(loop), -1}, q.before(f.op("use_carried"))).status == Status::NotObservableHere);
    const auto retained = q.qualify({f.value("next"), q.siteOf(loop), -1}, q.before(f.op("previous_use")));
    check(retained.available() && retained.representative == carried);
    check(retained.reference.value == f.value("next") && retained.reference.visitOffset == -1);
    check(
        q.qualify({f.value("next"), q.siteOf(loop), -1}, q.before(f.op("use_carried"))).status ==
        Status::NotObservableHere);
    check(q.canonicalGuardOwner(q.siteOf(f.op("g1"))) == q.canonicalGuardOwner(q.siteOf(f.op("g2"))));
    check(q.canonicalGuardOwner(q.siteOf(f.op("g1"))) != q.canonicalGuardOwner(q.siteOf(f.op("use_carried"))));
    check(q.canonicalGuardOwner(q.siteOf(f.op("g1"))) == q.canonicalGuardOwner(q.siteOf(f.op("same"))));
    const auto last = q.atom({ObservationAtom::LoopHasNext, q.siteOf(loop), 1, 0}, q.before(f.op("use_carried")));
    check(last.available() && last.recipe.kind == OriginalValueRecipe::LoopHasNext);
    check(
        q.atom({ObservationAtom::LoopHasNext, q.siteOf(loop), 1, 0}, q.after(loop)).status ==
        Status::NotObservableHere);
    const auto work = q.evaluationCount();
    q.qualify(f.value("condition"), q.before(f.op("use_async")));
    check(q.evaluationCount() == work);
    check(f.text() == unchanged);

    Fixture arithmetic(context, R"mlir(
module {
  func.func @arithmetic(%d: i32, %n: index) {
    %z = arith.constant 0 : i32
    %o = arith.constant 1 : i32
    %nz = arith.cmpi ne, %d, %z : i32
    %conditional = scf.if %nz -> (i32) {
      %v = arith.divui %o, %d : i32 loc("guarded_divide")
      scf.yield %v : i32
    } else {
      scf.yield %z : i32
    } loc("conditional")
    %bad = arith.divui %o, %d : i32 loc("unqualified_divide")
    %max = arith.constant 127 : i8
    %one = arith.constant 1 : i8
    %wrapped = arith.addi %max, %one : i8 loc("wrapped")
    %poison = arith.addi %max, %one overflow<nsw> : i8 loc("poison")
    %zero = arith.constant 0 : index loc("zero")
    %large = arith.constant 9223372036854775807 : index
    %almost = arith.constant 9223372036854775806 : index
    %two = arith.constant 2 : index
    scf.for %i = %almost to %large step %two { scf.yield } loc("overflow_loop")
    scf.for %i = %large to %large step %two { scf.yield } loc("empty_loop")
    return loc("return")
  }
}
)mlir");
    const auto before = arithmetic.text();
    OriginalValueQueries a(arithmetic.original);
    check(a.qualify(arithmetic.value("conditional"), a.after(arithmetic.op("conditional"))).available());
    check(
        a.qualify(arithmetic.value("conditional"), a.before(arithmetic.op("conditional"))).status ==
        Status::NotObservableHere);
    check(
        a.qualify(arithmetic.value("conditional"), a.before(arithmetic.op("guarded_divide"))).status ==
        Status::NotObservableHere);
    check(
        a.qualify(arithmetic.value("unqualified_divide"), a.after(arithmetic.op("unqualified_divide"))).status ==
        Status::Unresolved);
    check(a.qualify(arithmetic.value("wrapped"), a.after(arithmetic.op("wrapped"))).available());
    check(a.qualify(arithmetic.value("poison"), a.after(arithmetic.op("poison"))).status == Status::Unresolved);
    check(a.counted(cast<scf::ForOp>(arithmetic.op("overflow_loop"))).status == Status::Unresolved);
    check(a.counted(cast<scf::ForOp>(arithmetic.op("empty_loop"))).available());
    const auto extrema = a.guardedLast(
        arithmetic.value("zero"), arithmetic.original.function.getArgument(1), a.before(arithmetic.op("return")),
        false);
    check(extrema.available() && extrema.recipe.kind == OriginalValueRecipe::GuardedLast);
    check(arithmetic.text() == before);
    f.original.version = OriginalProgramVersion::fresh();
    check(!q.current());
    check(q.qualify(g, q.before(loop)).status == Status::Unresolved);
    check(q.counted(loop).status == Status::Unresolved);
    std::cout << "PASS: original-value/endpoint semantic fixtures\n";
}
