// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
// Real MLIR control and production Phase A; effects in the semantic fixtures
// are supplied explicitly. This is not a replacement for importer/device tests.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <map>
#include <string>

namespace fs = mlir::pto::frontiersynch;
using namespace mlir;
using namespace mlir::pto;
using H = fs::FactoredUseNode::Hazard;
using S = fs::OriginalValueQualification::Status;
static void check(bool condition, int line = __builtin_LINE())
{
    if (!condition) {
        llvm::errs() << "D1 native assertion failed at " << line << "\n";
        std::exit(1);
    }
}
static fs::Region op(std::size_t id)
{
    fs::Region r;
    r.kind = fs::Region::Operation;
    r.operation = id;
    return r;
}
struct Fixture {
    fs::OriginalStructure original;
    SyncInput input;
    std::vector<std::unique_ptr<CompoundInstanceElement>> instructions;
    std::map<std::string, std::size_t> sites;
    Fixture(func::FuncOp function, std::vector<fs::Access> effects)
    {
        original.function = function;
        original.cells.resize(1);
        function.walk([&](mlir::Operation* operation) {
            const auto site = original.originalSites.size();
            original.originalSites.push_back(operation);
            if (auto loc = dyn_cast<NameLoc>(operation->getLoc())) {
                sites.emplace(loc.getName().getValue().str(), site);
            }
        });
        for (std::size_t i = 0; i < effects.size(); ++i) {
            const auto site = sites.at("p" + std::to_string(i));
            auto* anchor = original.originalSites[site];
            instructions.push_back(std::make_unique<CompoundInstanceElement>(
                i, SmallVector<const BaseMemInfo*>{}, SmallVector<const BaseMemInfo*>{},
                effects[i].write ? PipelineType::PIPE_MTE2 : PipelineType::PIPE_V, anchor->getName()));
            instructions.back()->elementOp = anchor;
            fs::PhysicalOperation phase;
            phase.instruction = instructions.back().get();
            phase.original = site;
            phase.beforeExecutable = phase.afterExecutable = true;
            phase.enclosingAfter = i;
            phase.accesses.push_back(effects[i]);
            original.operations.push_back(phase);
        }
    }
    fs::Region choice(std::string name, fs::Region yes, fs::Region no = {})
    {
        fs::Region r;
        r.kind = fs::Region::Choice;
        r.originalOwner = sites.at(name);
        r.children = {std::move(yes), std::move(no)};
        return r;
    }
    std::unique_ptr<fs::ProgramAnalysis> analyze()
    {
        return std::make_unique<fs::ProgramAnalysis>(input, std::move(original));
    }
};
static const fs::FixedVisitEndpointPair& endpoint(const fs::FixedVisitSourceFrontier& frontier, std::size_t source)
{
    for (const auto& pair : frontier.endpoints) {
        if (pair.source == source) {
            return pair;
        }
    }
    check(false);
    std::abort();
}
static std::string printIR(mlir::Operation* op)
{
    std::string result;
    llvm::raw_string_ostream stream(result);
    op->print(stream);
    return result;
}
int main()
{
    MLIRContext context;
    context.loadDialect<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    context.disableMultithreading();
    auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @conditional(%g: i1) {
    %p0 = arith.constant 0 : i32 loc("p0")
    %p1 = arith.constant 1 : i32 loc("p1")
    scf.if %g { %p2 = arith.constant 2 : i32 loc("p2") } loc("select")
    %p3 = arith.constant 3 : i32 loc("p3")
    %p4 = arith.constant 4 : i32 loc("p4")
    return
  }
  func.func @late(%x: i32, %y: i32) {
    %p0 = arith.constant 0 : i32 loc("p0")
    %g = arith.cmpi slt, %x, %y : i32
    %p1 = arith.constant 1 : i32 loc("p1")
    scf.if %g { %p2 = arith.constant 2 : i32 loc("p2") } loc("select")
    %p3 = arith.constant 3 : i32 loc("p3")
    %p4 = arith.constant 4 : i32 loc("p4")
    return
  }
  func.func @arms(%g: i1) {
    scf.if %g { %p0 = arith.constant 0 : i32 loc("p0") }
    else { %p1 = arith.constant 1 : i32 loc("p1") } loc("select")
    %p2 = arith.constant 2 : i32 loc("p2")
    return
  }
  func.func @pair() {
    %p0 = arith.constant 0 : i32 loc("p0")
    %p1 = arith.constant 1 : i32 loc("p1")
    return
  }
  func.func @loop() {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %i = %zero to %four step %one {
      %p0 = arith.constant 0 : i32 loc("p0")
      %p1 = arith.constant 1 : i32 loc("p1")
      %p2 = arith.constant 2 : i32 loc("p2")
    } loc("repeat")
    return
  }
})mlir", &context);
    check(module && succeeded(verify(*module)));
    const auto before = printIR(module->getOperation());
    const fs::Access W{0, false, true, true}, R{0, true, false, false};
    for (unsigned variant = 0; variant < 5; ++variant) {
        const bool late = variant == 3, missing = variant == 1, partial = variant == 2;
        Fixture f(module->lookupSymbol<func::FuncOp>(late ? "late" : "conditional"),
                  {missing ? R : W, R, partial ? fs::Access{0, false, true, false} : W, R, W});
        f.original.body.children = {op(0), op(1), f.choice("select", op(2)), op(3), op(4)};
        if (variant == 4) {
            f.original.operations[0].afterExecutable = false;
        }
        auto a = f.analyze();
        check(a->complete());
        const auto size = a->originalUsesAt(3, 0).nodes().size();
        const auto frontier = a->occurrences().fixedSourcesAt(3, 0, H::RAW);
        check(frontier->sources.complete && frontier->sources.alternatives.size() == 2);
        check(frontier->hasIncoming == missing);
        check(frontier->sources.exclusiveWriters == !partial);
        check(frontier->physicalRolesQualified);
        check(endpoint(*frontier, 2).sourceCondition == 1);
        check(endpoint(*frontier, 2).targetCondition != 1);
        if (late || variant == 4) {
            const auto& old = endpoint(*frontier, 0);
            check(old.sourceQualification.status == S::NotObservableHere);
            check(old.targetQualification.available());
            check(!a->occurrences().fixedVisit(0, 3, 0, H::RAW).exact);
            check(a->occurrences().fixedVisit(2, 3, 0, H::RAW).exact);
        } else if (!partial) {
            check(frontier->localGuardsQualified);
        }
        if (!missing && !late && variant != 4) {
            check(a->occurrences().fixedVisit(0, 1, 0, H::RAW).exact);
        }
        if (partial) {
            check(!a->occurrences().fixedVisit(0, 3, 0, H::RAW).exact);
        }
        bool familySeen = false;
        for (auto id : a->obligationsAt(a->structure().operations[3].original)) {
            if (a->obligations().get(id)->key.kind == fs::OriginalObligationKey::Kind::RAW) {
                check(a->fixedSourcesFor(id) == frontier);
                familySeen = true;
            }
        }
        check(familySeen && a->lifetimes().stats().requirements == 0);
        check(a->originalUsesAt(3, 0).nodes().size() == size);
        check(!a->fixedSourcesFor({})->sources.complete);
    }
    {
        Fixture f(module->lookupSymbol<func::FuncOp>("arms"), {W, R, R});
        f.original.body.children = {f.choice("select", op(0), op(1)), op(2)};
        auto a = f.analyze();
        check(a->complete());
        auto q = a->occurrences().fixedSourcesAt(1, 0, H::RAW);
        check(q->sources.complete && q->hasIncoming && q->endpoints.size() == 1);
        check(!a->occurrences().fixedVisit(0, 1, 0, H::RAW).exact);
        bool seen = false;
        for (auto id : a->obligationsAt(a->structure().operations[1].original)) {
            if (a->obligations().get(id)->key.kind == fs::OriginalObligationKey::Kind::RAW) {
                check(a->fixedSourcesFor(id)->hasIncoming);
                seen = true;
            }
        }
        check(seen);
    }
    {
        // WAR and WAW have identical source/target/cell/cuts but DIFFERENT
        // source roots: WAR is a reader; WAW retains simultaneous weak origins.
        Fixture f(module->lookupSymbol<func::FuncOp>("pair"), {fs::Access{0, true, true, false}, W});
        f.original.body.children = {op(0), op(1)};
        auto a = f.analyze();
        check(a->complete());
        bool war = false, waw = false;
        const auto& requests = a->requirementsAt(1);
        for (std::size_t i = 0; i < requests.size(); ++i) {
            if (requests[i].relationship.source.operation != 0) {
                continue;
            }
            auto result = a->fixedVisitFor(a->decodeAt(1, i));
            const auto kind = requests[i].relationship.kind;
            if (kind == fs::StorageRelationship::WAR) {
                check(result.exact);
                war = true;
            } else if (kind == fs::StorageRelationship::WAW) {
                check(!result.exact);
                waw = true;
            }
            check(a->interpretAt(1, i).fixedVisit.alternatives == result.alternatives);
        }
        check(war && waw);
    }
    {
        Fixture f(module->lookupSymbol<func::FuncOp>("loop"), {W, R, R});
        fs::Region body;
        body.children = {op(0), op(1), op(2)};
        fs::Region repeat;
        repeat.kind = fs::Region::For;
        repeat.originalOwner = f.sites.at("repeat");
        repeat.children = {body};
        f.original.body.children = {repeat};
        fs::OriginalValueQueries values(f.original);
        fs::OriginalLifetimes storage(f.original, &values);
        fs::OccurrenceQueries queries(f.original, &values, &storage);
        check(queries.fixedVisit(0, 2, 0, H::RAW).exact);
        const auto q = queries.fixedSourcesAt(2, 0, H::RAW);
        check(q->sources.frame.kind == fs::FactoredUseFrame::Kind::ForBody);
        ++f.original.version.revision;
        check(!queries.fixedVisit(0, 2, 0, H::RAW).exact);
    }
    check(before == printIR(module->getOperation()));
    llvm::outs() << "PASS: native D1 sources, incoming paths, endpoint guards, hazard cache, local loop visit; IR unchanged\n";
}
