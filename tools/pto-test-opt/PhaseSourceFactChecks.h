// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Compare scoped source-fact views with independently rebuilt phase indexes.
#ifndef PTO_TEST_PHASESOURCEFACTCHECKS_H
#define PTO_TEST_PHASESOURCEFACTCHECKS_H
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <set>
#include <tuple>
namespace {
namespace phasefacts = mlir::pto::frontiersynch;
using Phase = mlir::pto::CompoundInstanceElement;
bool samePhaseView(mlir::func::FuncOp function, const phasefacts::PhaseIndex& a,
                   const phasefacts::PhaseIndex& b, llvm::ArrayRef<const Phase*> selected)
{
    bool equal = true;
    auto compareValue = [&](mlir::Value value) {
        if (a.isRelevant(value) != b.isRelevant(value)) { equal = false; }
        auto* x = a.carriedOrdinal(value); auto* y = b.carriedOrdinal(value);
        if (bool(x) != bool(y) || (x && std::tie(x->seed, x->stride, x->modulus) !=
                                       std::tie(y->seed, y->stride, y->modulus))) { equal = false; }
    };
    function.walk([&](mlir::Operation* op) {
        for (auto value : op->getOperands()) { compareValue(value); }
        for (auto value : op->getResults()) { compareValue(value); }
        for (auto& region : op->getRegions()) {
            for (auto& block : region) {
                for (auto value : block.getArguments()) { compareValue(value); }
            }
        }
        if (a.needsValuePrerequisite(op) != b.needsValuePrerequisite(op) ||
            a.hasRelevantResults(op) != b.hasRelevantResults(op) ||
            a.hasRelevantCarriedState(op) != b.hasRelevantCarriedState(op) ||
            a.hasUnprovedCarriedState(op) != b.hasUnprovedCarriedState(op)) { equal = false; }
        using Key = std::tuple<const Phase*, mlir::Operation*, bool, bool>;
        auto prerequisites = [&](const phasefacts::PhaseIndex& index) {
            std::set<Key> result;
            for (const auto& edge : index.prerequisitesFor(op)) {
                result.emplace(edge.producer, edge.consumer, edge.native, edge.directSSA);
            }
            return result;
        };
        if (prerequisites(a) != prerequisites(b)) { equal = false; }
    });
    auto edges = [](const std::vector<phasefacts::StorageGenerator>& list) {
        std::set<std::pair<uint32_t, uint32_t>> result;
        for (auto edge : list) { result.emplace(edge.source, edge.target); }
        return result;
    };
    auto x = a.mapPrerequisites(selected), y = b.mapPrerequisites(selected);
    return equal && x.error == y.error && edges(x.native) == edges(y.native) && edges(x.demands) == edges(y.demands);
}
bool checkSharedPhaseFacts(mlir::MLIRContext* context)
{
    using namespace mlir;
    auto module = parseSourceString<ModuleOp>(R"(
module {
  func.func @views(%n: index, %guard: i1) -> index {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %three = arith.constant 3 : index
    %a = arith.addi %n, %one {test.phase} : index
    %b = arith.addi %a, %one {test.phase, test.multiple} : index
    %c = arith.addi %b, %one {test.phase} : index
    %unused = scf.for %i = %zero to %n step %one iter_args(%bank = %zero) -> index {
      %probe = arith.addi %bank, %one {test.phase} : index
      %sum = arith.addi %bank, %one : index
      %next = arith.remui %sum, %three : index
      scf.yield %next : index
    }
    %chosen = scf.if %guard -> index {
      scf.yield %c : index
    } else {
      scf.yield %zero : index
    }
    return %chosen : index
  }
  func.func @foreign() { return }
})", context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("views");
    std::vector<std::unique_ptr<Phase>> owners;
    SmallVector<const Phase*> phases;
    scf::ForOp loop;
    function.walk([&](Operation* op) {
        if (auto found = dyn_cast<scf::ForOp>(op)) { loop = found; }
        if (!op->hasAttr("test.phase")) { return; }
        for (unsigned i = 0; i < (op->hasAttr("test.multiple") ? 2U : 1U); ++i) {
            owners.push_back(std::make_unique<Phase>(static_cast<unsigned>(owners.size()),
                SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
                pto::PipelineType::PIPE_S, op->getName()));
            owners.back()->elementOp = op;
            phases.push_back(owners.back().get());
        }
    });
    auto root = std::make_unique<phasefacts::PhaseIndex>();
    if (failed(root->build(function, phases))) { return false; }
    auto facts = root->sourceFacts();
    const auto originalWork = root->sourceWork();
    if (!originalWork.sourceVisits || !originalWork.recurrenceAttempts) { return false; }
    root.reset(); // The retained source facts must not borrow their root index.
    for (unsigned mask = 0; mask < (1U << phases.size()); ++mask) {
        SmallVector<const Phase*> selected;
        for (unsigned i = 0; i < phases.size(); ++i) {
            if (mask & (1U << i)) { selected.push_back(phases[i]); }
        }
        phasefacts::PhaseIndex fresh, shared;
        if (failed(fresh.build(function, selected)) || failed(shared.build(function, selected, facts)) ||
            shared.sourceFacts() != facts || shared.sourceFactsFor(function, selected) != facts ||
            shared.relevanceVisits() != fresh.relevanceVisits() ||
            !samePhaseView(function, fresh, shared, selected)) { return false; }
        const auto work = shared.sourceWork();
        if (std::tie(work.sourceVisits, work.baselineVisits, work.recurrenceAttempts) !=
            std::tie(originalWork.sourceVisits, originalWork.baselineVisits, originalWork.recurrenceAttempts)) {
            return false;
        }
        const bool deadStateSelected = mask & (1U << (phases.size() - 1));
        if (bool(shared.carriedOrdinal(loop.getRegionIterArgs().front())) != deadStateSelected) { return false; }
    }
    ScopedDiagnosticHandler expected(context, [](Diagnostic&) { return success(); });
    phasefacts::PhaseIndex rejected, narrow;
    if (failed(narrow.build(function, {}))) { return false; }
    auto foreignFunction = module->lookupSymbol<func::FuncOp>("foreign");
    if (narrow.sourceFactsFor(foreignFunction, {}) || narrow.sourceFactsFor(function, phases)) { return false; }
    auto expanded = narrow.sourceFactsFor(function, phases);
    if (failed(narrow.build(function, phases, std::move(expanded))) || !narrow.sourceFactsFor(function, phases)) {
        return false;
    }
    Phase foreign(99, {}, {}, pto::PipelineType::PIPE_S, phases.front()->elementOp->getName());
    foreign.elementOp = phases.front()->elementOp;
    const Phase* unknown[] = {&foreign};
    return failed(rejected.build(module->lookupSymbol<func::FuncOp>("foreign"), {}, facts)) &&
           failed(rejected.build(function, unknown, facts)) && !rejected.sourceFacts();
}
} // namespace
#endif
