// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
// Executes production OriginalLifetimes/Control on real MLIR control with
// explicitly supplied semantic phase effects. Real PTO import is tested separately.
#include "PTO/Transforms/FrontierSynch/OriginalLifetimes.h"
#include "PTO/Transforms/FrontierSynch/OriginalObligations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include <cstdlib>
#include <iostream>
using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::frontiersynch;
static void check(bool condition, int line = __builtin_LINE())
{
    if (!condition) {
        std::cerr << "lifetime assertion failed at " << line << "\n";
        std::exit(1);
    }
}

frontiersynch::Region op(std::size_t id)
{
    frontiersynch::Region r;
    r.kind = frontiersynch::Region::Operation;
    r.operation = id;
    return r;
}
void initialize(
    OriginalStructure& p, std::vector<std::unique_ptr<CompoundInstanceElement>>& instructions, std::size_t n)
{
    p.cells.resize(1);
    p.operations.resize(n);
    instructions.resize(n);
    p.originalSites.resize(21);
    p.function.walk([&](mlir::Operation* operation) {
        if (isa<scf::IfOp>(operation)) {
            p.originalSites[10] = operation;
        }
        if (isa<scf::ForOp>(operation)) {
            p.originalSites[20] = operation;
        }
    });
    for (std::size_t i = 0; i < n; ++i) {
        mlir::Operation* anchor = nullptr;
        p.function.walk([&](mlir::Operation* operation) {
            auto location = dyn_cast<NameLoc>(operation->getLoc());
            if (location && location.getName().getValue() == "p" + std::to_string(i)) {
                anchor = operation;
            }
        });
        check(anchor != nullptr);
        instructions[i] = std::make_unique<CompoundInstanceElement>(
            i, SmallVector<const BaseMemInfo*>{}, SmallVector<const BaseMemInfo*>{},
            i % 2 ? PipelineType::PIPE_V : PipelineType::PIPE_MTE2, anchor->getName());
        instructions[i]->elementOp = anchor;
        p.operations[i].instruction = instructions[i].get();
        p.originalSites[i] = anchor;
        p.operations[i].original = i;
        p.operations[i].beforeExecutable = p.operations[i].afterExecutable = true;
        p.operations[i].enclosingAfter = i;
    }
}
int main()
{
    MLIRContext context;
    context.loadDialect<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    auto module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @acyclic(%g: i1) {
    %p0 = arith.constant 0 : i32 loc("p0")
    %p1 = arith.constant 1 : i32 loc("p1")
    scf.if %g { %p2 = arith.constant 2 : i32 loc("p2") }
    %p3 = arith.constant 3 : i32 loc("p3")
    %p4 = arith.constant 4 : i32 loc("p4")
    return
  }
  func.func @repeated() {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %i = %zero to %four step %one {
      %p0 = arith.constant 0 : i32 loc("p0")
      %p1 = arith.constant 1 : i32 loc("p1")
    }
    return
  }
}
)mlir",
        &context);
    check(module && succeeded(verify(*module)));
    OriginalStructure p;
    p.function = module->lookupSymbol<func::FuncOp>("acyclic");
    std::vector<std::unique_ptr<CompoundInstanceElement>> instructions;
    initialize(p, instructions, 5);
    p.operations[0].accesses.push_back({0, false, true, true});
    p.operations[1].accesses.push_back({0, true, false, false});
    p.operations[2].accesses.push_back({0, false, true, true});
    p.operations[3].accesses.push_back({0, true, false, false});
    p.operations[3].accesses.push_back({0, true, false, false});
    p.operations[4].accesses.push_back({0, false, true, true});
    frontiersynch::Region choice;
    choice.kind = frontiersynch::Region::Choice;
    choice.originalOwner = 10;
    choice.children = {op(2), frontiersynch::Region{}};
    p.body.children = {op(0), op(1), choice, op(3), op(4)};
    OriginalLifetimes storage(p);
    check(storage.complete());
    check(storage.stats().requirements == 0);
    check(storage.mayOriginAt(3, 0, 0, false) == true);
    check(storage.mayOriginAt(3, 0, 2, false) == true);
    check(storage.mayOriginAt(3, 0, 1, false) == false);
    check(storage.mayOriginAt(0, 0, NoControlId, false, true) == true);
    check(storage.mayOriginAt(0, 0, NoControlId, true, true) == true);
    check(storage.mayOriginAt(3, 0, NoControlId, true, true) == false);
    check(!storage.mayOriginAt(99, 0, 0, false).has_value());
    check(!storage.hasMayOriginsAt(0, 99, false).has_value());
    check(storage.hasMayOriginsAt(0, 0, true) == true); // explicit incoming
    check(storage.hasMayOriginsAt(1, 0, true) == false);
    check(storage.effectIncidences(3, 0, false).size() == 2);
    check(storage.factored(0).complete);
    check(storage.stats().requirements == 0);
    check(storage.mayOriginsAt(3, 0, false).size() == 2);
    check(storage.stats().requirements == 0);
    // Explicit compatibility output keeps every effect and subscription index.
    const auto& subscriptions = storage.subscriptionsAt(0);
    check(!subscriptions.empty() && storage.stats().requirements > 0);
    const auto pairs = storage.stats().requirements;
    for (std::size_t target = 0; target < p.operations.size(); ++target) {
        const auto& requests = storage.requirementsAt(target);
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const auto& request = requests[index];
            bool found = false;
            for (const auto& hook : storage.subscriptionsAt(request.source.operation)) {
                found |= hook.deadlineOperation == target && hook.requirementIndex == index;
            }
            check(found);
            check(request.sourceEffects && !request.sourceEffects->empty());
            check(request.targetEffects && !request.targetEffects->empty());
        }
    }
    check(storage.stats().requirements == pairs);
    OriginalLifetimes supportFirst(p);
    check(supportFirst.stats().requirements == 0);
    auto support = supportFirst.supportBetween(0, 4, 0);
    check(support.complete && !support.affectedRequirements.empty());
    check(supportFirst.stats().requirements == pairs);

    // Unbounded original control retains marginals and incoming cases; no false
    // absence is inferred from the unavailable factored loop projection.
    OriginalStructure repeated;
    repeated.function = module->lookupSymbol<func::FuncOp>("repeated");
    std::vector<std::unique_ptr<CompoundInstanceElement>> loopInstructions;
    initialize(repeated, loopInstructions, 2);
    repeated.operations[0].accesses.push_back({0, false, true, true});
    repeated.operations[1].accesses.push_back({0, true, false, false});
    frontiersynch::Region body;
    body.children = {op(0), op(1)};
    frontiersynch::Region loop;
    loop.kind = frontiersynch::Region::For;
    loop.originalOwner = 20;
    loop.children = {body};
    repeated.body.children = {loop};
    OriginalLifetimes cyclic(repeated);
    check(cyclic.complete() && !cyclic.factored(0).complete);
    check(cyclic.mayOriginAt(0, 0, 0, false) == true);
    check(cyclic.mayOriginAt(0, 0, 1, true) == true);
    check(cyclic.mayOriginAt(0, 0, NoControlId, true, true) == true);
    check(cyclic.stats().requirements == 0);
    std::cout << "PASS: production lifetime membership, lazy pairs/subscriptions, support-first query, loop fallback\n";
}
